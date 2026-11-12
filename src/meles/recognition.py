"""
The face recognition models the experiment compares, and the step that turns frames into
embeddings.

A recognizer takes one aligned 112x112 crop and returns a vector; two faces are the same person
when their vectors point in nearly the same direction. The preprocessing constants below were
determined by measurement rather than from documentation, because none of the three models
documents its own completely:

- SFace: verified against ``cv2.FaceRecognizerSF.feature``, the reference implementation for this
  exact file. Raw 0-255 RGB reproduces it to a cosine of 1.000000, because the graph applies
  ``(x - 127.5) / 128`` itself in its leading Sub/Mul nodes.
- ArcFace: the preprocessing InsightFace's own ``ArcFaceONNX`` applies to this file.
- GhostFaceNet: chosen empirically. RGB with ``(x - 127.5) / 128`` separated identities best
  (mean same-identity cosine minus mean different-identity cosine of 0.760 over 72 frames of six
  identities, against 0.158 for plain 0-1 scaling), and it agrees with the quantisation range the
  upstream model card states.
"""

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import cv2
import numpy as np
from numpy.typing import NDArray
from tqdm.auto import tqdm

from meles import weights
from meles.alignment import TEMPLATE_SIZE, align_face, letterbox
from meles.detection import YUNET, Detector, FacialArea, default_detector
from meles.inference import OnnxSession

logger = logging.getLogger(__name__)

# A frame to embed: a path to an image file, or an already decoded BGR image.
Frame = str | Path | NDArray[np.uint8]

# The OpenCV model zoo stores '*.onnx' with git-lfs, so 'raw.githubusercontent.com' would serve a
# pointer file rather than the model.
OPENCV_ZOO = "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models"


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


SFACE = EmbedderSpec(
    name="SFace",
    weights=weights.WeightSpec(
        file_name="face_recognition_sface_2021dec.onnx",
        url=f"{OPENCV_ZOO}/face_recognition_sface/face_recognition_sface_2021dec.onnx",
        sha256="0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79",
    ),
    output_dim=128,
    # The graph normalises internally, so it takes raw pixels.
    mean=0.0,
    std=1.0,
)

# The recognition model from InsightFace's buffalo_l pack: a ResNet50 trained on WebFace600K. Its
# weights are published for non-commercial research use only, which this project is.
ARCFACE = EmbedderSpec(
    name="ArcFace",
    weights=weights.WeightSpec(
        file_name="w600k_r50.onnx",
        url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
        sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f",
        # The archive members are flat, with no 'buffalo_l/' prefix.
        archive_member="w600k_r50.onnx",
    ),
    output_dim=512,
)

# GhostFaceNet has no first-party export that avoids TensorFlow: every release asset upstream is a
# Keras '.h5'. This is a third-party ONNX conversion reporting 99.683% on LFW, a figure matching
# none of the published upstream checkpoints, so treat it as "a GhostFaceNetV1-class model" rather
# than as a specific citable checkpoint.
GHOSTFACENET = EmbedderSpec(
    name="GhostFaceNet",
    weights=weights.WeightSpec(
        file_name="ghostface_fp32.onnx",
        url=(
            "https://raw.githubusercontent.com/andestech/ModelZoo/"
            "4d25c9ffe36ecb805ae7a6443b9079ab5b9b98d3/GhostFaceNet/Model/ghostface_fp32.onnx"
        ),
        sha256="e8c60ea19ec09ae0b1d30460426fac783c290b4e76c5c7e582e159c8906480ce",
    ),
    output_dim=512,
    std=128.0,
    # Exported from Keras, so it takes NHWC and transposes internally.
    layout="NHWC",
)

# The models keyed by the name the experiment configuration uses.
SPECS: dict[str, EmbedderSpec] = {
    "sface": SFACE,
    "arcface": ARCFACE,
    "ghostfacenet": GHOSTFACENET,
}


def resolve(method: str) -> EmbedderSpec:
    """
    The model a configuration key names.

    :param method: The key, for example 'arcface'
    :return: The model specification
    :raises ValueError: If nothing is registered under that key
    """
    spec = SPECS.get(method)
    if spec is None:
        raise ValueError(f"Unknown recognizer {method!r}, expected one of {sorted(SPECS)}")
    return spec


def prefetch() -> None:
    """Download every model, including the detector, so that a long run does not have to."""
    weights.resolve(YUNET)
    for spec in SPECS.values():
        weights.resolve(spec.weights)


class FrameEmbedder:
    """
    Embeds each frame on its own: detect the face, align it to 112x112, run the crop through a
    recognition model.

    Detection is inherently per image, but the recognition forward pass is far cheaper per frame
    when batched, so crops are accumulated and flushed together.
    """

    def __init__(
        self,
        spec: EmbedderSpec,
        detector: Detector | None = None,
        batch_size: int = 64,
        threads: int = 0,
    ) -> None:
        """
        :param spec: The model to run
        :param detector: The detector locating faces; the shared YuNet one by default
        :param batch_size: The number of crops per forward pass
        :param threads: The intra-operator thread budget passed to ONNX Runtime
        """
        self.spec = spec
        self.detector = detector if detector is not None else default_detector
        self.batch_size = batch_size
        self.threads = threads
        self._session: OnnxSession | None = None

    @property
    def session(self) -> OnnxSession:
        """
        The model, downloaded and loaded on first use, so that constructing an embedder is free.

        :return: The open session
        """
        if self._session is None:
            self._session = OnnxSession(weights.resolve(self.spec.weights), threads=self.threads)
        return self._session

    def crop(self, frame: Frame, area: FacialArea | None = None) -> NDArray[np.uint8]:
        """
        The aligned crop to embed for one frame.

        A frame the detector finds no face in is scaled whole into the crop instead of being
        dropped: the anonymizations are meant to destroy faces, so dropping those frames would
        change which frames the experiment scores.

        :param frame: A path to an image file, or an already decoded BGR image
        :param area: The face to align on, or None to detect it in the frame
        :return: The aligned crop with shape (input_size, input_size, 3), BGR
        :raises FileNotFoundError: If the frame is a path that cannot be read
        """
        decoded = frame if isinstance(frame, np.ndarray) else cv2.imread(str(frame))
        if decoded is None:
            raise FileNotFoundError(f"Could not read frame {frame}")
        image = np.asarray(decoded, dtype=np.uint8)

        found = self.detector.detect(image) if area is None else area
        if found is None:
            return letterbox(image, self.spec.input_size)
        return align_face(image, found.landmarks, self.spec.input_size)

    def blob(self, crops: Sequence[NDArray[np.uint8]]) -> NDArray[np.float32]:
        """
        Preprocess a batch of crops into the model's input tensor.

        :param crops: The aligned crops, BGR
        :return: The preprocessed batch in the model's declared layout
        """
        size = self.spec.input_size
        batch = cv2.dnn.blobFromImages(
            list(crops),
            1.0 / self.spec.std,
            (size, size),
            (self.spec.mean,) * 3,
            self.spec.swap_rb,
        )
        # 'blobFromImages' always builds NCHW; a model declaring NHWC needs it transposed back.
        if self.spec.layout == "NHWC":
            batch = np.ascontiguousarray(batch.transpose(0, 2, 3, 1))
        return batch.astype(np.float32, copy=False)

    def embed(
        self,
        frames: Sequence[Frame],
        areas: Sequence[FacialArea | None] | None = None,
        progress: bool = True,
    ) -> NDArray[np.float32]:
        """
        Embed every frame, one row per input frame in the order given.

        :param frames: The frames to embed
        :param areas: The face to align each frame on, or None to detect it. Supplying them is
            what lets a caller align on a face located somewhere else, for example in the frame
            before it was anonymized
        :param progress: Whether to show a progress bar
        :return: The embeddings with shape (len(frames), output_dim)
        :raises ValueError: If ``areas`` is given and does not match ``frames`` in length
        """
        if areas is not None and len(areas) != len(frames):
            raise ValueError(f"Got {len(areas)} areas for {len(frames)} frames")

        embeddings: NDArray[np.float32] = np.empty((len(frames), self.spec.output_dim), dtype=np.float32)
        pending: list[tuple[int, NDArray[np.uint8]]] = []

        def flush() -> None:
            if not pending:
                return
            vectors = self.session.run(self.blob([crop for _, crop in pending]))
            for (row, _), vector in zip(pending, vectors):
                embeddings[row] = vector
            pending.clear()

        stepped = (
            tqdm(frames, desc=f"Embedding ({self.spec.name})", unit="frame", mininterval=5.0) if progress else frames
        )
        for index, frame in enumerate(stepped):
            pending.append((index, self.crop(frame, None if areas is None else areas[index])))
            if len(pending) >= self.batch_size:
                flush()
        flush()
        return embeddings
