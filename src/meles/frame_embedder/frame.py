"""
The frame-level embedder: detect a face in every frame, align it, and run it through an ONNX
recognition model in batches.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from meles import weights
from meles.common import Frame, Frames, OptEmbeddings
from meles.detection import (
    TEMPLATE_SIZE,
    Detector,
    default_detector,
    prepare_face,
    whole_frame,
)
from meles.frame_embedder.embedder import Embedder
from meles.frame_embedder.session import OnnxSession

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EmbedderSpec:
    """
    Everything that distinguishes one recognition model from another.

    :param name: The model name, used in log messages and the progress bar
    :param weights: The weights to run
    :param output_dim: The embedding dimensionality the model produces
    :param mean: The value subtracted from every pixel before inference
    :param std: The value every pixel is divided by after the mean is subtracted
    :param swap_rb: Whether to convert the BGR crop to RGB first
    :param layout: The dimension order the graph declares, "NCHW" or "NHWC"
    :param input_size: The side length of the crop the model takes
    """

    name: str
    weights: weights.WeightSpec
    output_dim: int
    mean: float = 127.5
    std: float = 127.5
    swap_rb: bool = True
    layout: Literal["NCHW", "NHWC"] = "NCHW"
    input_size: int = TEMPLATE_SIZE


class FrameEmbedder(Embedder):
    """
    Embeds each frame on its own by detecting the face in it, aligning it to 112x112 and
    running the aligned crop through a recognition model.

    Detection is inherently per-image but the recognition forward pass is much cheaper per
    frame when batched, so frames are accumulated into a batch and flushed together.
    """

    def __init__(
        self,
        spec: EmbedderSpec,
        detector: Detector | None = None,
        batch_size: int = 64,
        on_missing: Literal["whole_frame", "none"] = "whole_frame",
        threads: int = 0,
    ):
        """
        :param spec: The model to run
        :param detector: The detector to locate faces with; the shared YuNet one by default
        :param batch_size: The number of crops per forward pass
        :param on_missing: What to embed when no face is detected. "whole_frame" embeds the
            frame as a whole, which is what DeepFace did with ``enforce_detection=False`` and
            keeps one embedding per frame; "none" reports no embedding for that frame
        :param threads: The intra-operator thread budget passed to ONNX Runtime
        """
        self.spec = spec
        self.detector = detector if detector is not None else default_detector
        self.batch_size = batch_size
        self.on_missing = on_missing
        self.threads = threads
        self._session: OnnxSession | None = None

    def metric(self) -> str:
        return "cosine"

    @property
    def session(self) -> OnnxSession:
        """
        The model, downloaded and loaded on first use.

        Deliberately lazy: ``workflow/scripts/embed_frame.py`` builds every registry entry at
        import time, so a job that runs one model must not load the other two.

        :return: The open session
        """
        if self._session is None:
            self._session = OnnxSession(weights.resolve(self.spec.weights), threads=self.threads)
        return self._session

    def _read(self, frame: Frame) -> NDArray[Any] | None:
        """
        Decode one frame into a BGR image.

        :param frame: A path, an already-decoded image, or an open binary stream
        :return: The image, or None if it could not be decoded
        """
        if isinstance(frame, np.ndarray):
            return frame
        if isinstance(frame, (str, Path)):
            return cv2.imread(str(frame))
        return cv2.imdecode(np.frombuffer(frame.read(), dtype=np.uint8), cv2.IMREAD_COLOR)

    def _crop(self, frame: Frame) -> NDArray[np.uint8] | None:
        """
        The aligned crop to embed for one frame.

        :param frame: The frame to prepare
        :return: The crop, or None if there is nothing to embed
        """
        image = self._read(frame)
        if image is None:
            logger.warning("Could not read frame", extra={"frame": str(frame)[:200]})
            return None

        # One face per frame, as DeepFace also did -- see Detector.detect_best for why that is
        # the largest detection rather than the most confident one.
        area = self.detector.detect_best(image)
        if area is not None:
            return prepare_face(image, area, self.spec.input_size)

        if self.on_missing == "none":
            return None
        return whole_frame(image, self.spec.input_size)[0]

    def _blob(self, crops: list[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """
        Preprocess a batch of crops into the model's input tensor.

        :param crops: The aligned crops, BGR
        :return: The preprocessed batch in the model's declared layout
        """
        size = self.spec.input_size
        blob = cv2.dnn.blobFromImages(
            crops,
            1.0 / self.spec.std,
            (size, size),
            (self.spec.mean,) * 3,
            self.spec.swap_rb,
        )
        # 'blobFromImages' always builds NCHW. GhostFaceNet was exported from Keras and declares
        # NHWC, so its batch has to be transposed back.
        if self.spec.layout == "NHWC":
            blob = np.ascontiguousarray(blob.transpose(0, 2, 3, 1))
        # OpenCV already produces float32; the cast is what lets the declared type hold.
        return blob.astype(np.float32, copy=False)

    def embed(self, frames: Frames) -> OptEmbeddings:
        # Preallocated and filled by index, so the result lines up with the input row by row
        # even though undetected frames never reach a batch.
        embeddings: OptEmbeddings = [None] * len(frames)
        pending: list[tuple[int, NDArray[np.uint8]]] = []

        def flush() -> None:
            if not pending:
                return
            vectors = self.session.run(self._blob([crop for _, crop in pending]))
            for (index, _), vector in zip(pending, vectors):
                embeddings[index] = np.asarray(vector, dtype=float).ravel().tolist()
            pending.clear()

        with tqdm(
            total=len(frames),
            desc=f"Embedding ({self.spec.name})",
            unit="frame",
            # The old per-frame bar wrote 5 MiB of carriage returns into every job log.
            mininterval=5.0,
        ) as progress:
            for index, frame in enumerate(frames):
                crop = self._crop(frame)
                if crop is not None:
                    pending.append((index, crop))
                    if len(pending) >= self.batch_size:
                        flush()
                progress.update(1)
            flush()

        missing = sum(1 for embedding in embeddings if embedding is None)
        if missing:
            logger.warning(
                "Frames without an embedding",
                extra={"model": self.spec.name, "missing": missing, "len": len(frames)},
            )
        return embeddings
