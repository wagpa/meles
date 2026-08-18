"""
Face detection for the anonymization and the embedding stage.

Both stages need the same thing -- where the face is in a frame -- so they share one detector.
YuNet replaces DeepFace's Haar-cascade backend: it is a 227 KiB ONNX model, it is an order of
magnitude faster than ``cv2.CascadeClassifier.detectMultiScale``, and unlike Haar it returns
the five landmarks the embedders need to align a crop.
"""

import logging
import threading
from abc import ABC, abstractmethod
from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from meles import weights
from meles.common import FacialArea

logger = logging.getLogger(__name__)

# The YuNet weights from the OpenCV model zoo. The '*.onnx' files there are stored with
# git-lfs, so the usual 'raw.githubusercontent.com' URL serves a 131-byte pointer file rather
# than the model -- the 'media.githubusercontent.com/media' form serves the real blob.
YUNET = weights.WeightSpec(
    file_name="face_detection_yunet_2023mar.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)

# The number of columns 'cv2.FaceDetectorYN.detect' returns per face: the box, five landmarks
# as (x, y) pairs, and the score.
_DETECTION_COLUMNS = 15


class Detector(ABC):
    """Locates faces in an image."""

    @abstractmethod
    def detect(self, image: NDArray[Any]) -> list[FacialArea]:
        """
        Find every face in an image.

        :param image: The image to search, BGR
        :return: The detected faces, ordered by descending confidence, empty if there are none
        """
        raise NotImplementedError

    def detect_best(self, image: NDArray[Any]) -> FacialArea | None:
        """
        The face to embed when a frame is assumed to show one subject.

        Selected by box area, not by confidence. The frames are pre-cropped around a single
        subject, so the subject's face is by far the largest thing in them -- but YuNet also
        picks up small spurious or background faces, and those score high enough to outrank the
        subject: over 300 sample frames, taking the most confident detection landed on an
        off-centre face 13 times, while taking the largest never did. Confidence cannot separate
        the two either, since the spurious detections reach 0.905 while genuine subjects start
        at 0.889.

        :param image: The image to search, BGR
        :return: The largest detected face, or None if there is none
        """
        faces = self.detect(image)
        if not faces:
            return None
        return max(faces, key=lambda face: face.w * face.h)


class YuNetDetector(Detector):
    """
    The YuNet detector, run through OpenCV's ONNX-backed ``cv2.FaceDetectorYN``.

    The underlying model wraps a ``cv::dnn::Net``, which keeps mutable state across a forward
    pass and is not thread-safe, so each thread gets its own. They are cheap (the model is
    227 KiB) and are built on first use, which keeps constructing a detector free -- the
    embedder registry in ``workflow/scripts/embed_frame.py`` instantiates every entry at
    import time and must not pay for weights it will not use.
    """

    def __init__(
        self,
        score_threshold: float = 0.6,
        nms_threshold: float = 0.3,
        top_k: int = 50,
        spec: weights.WeightSpec = YUNET,
    ):
        """
        :param score_threshold: The lowest confidence still reported as a face. Below OpenCV's
            own 0.9 default on purpose: the frames are low-resolution video stills and, in
            three of the four branches, already anonymized, so recall matters more than
            precision here
        :param nms_threshold: The IoU above which overlapping boxes are suppressed
        :param top_k: The number of boxes kept before suppression
        :param spec: The weights to run
        """
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.top_k = top_k
        self.spec = spec
        self._local = threading.local()

    @property
    def _model(self) -> Any:
        """
        This thread's detector, built on first use.

        :return: The thread's ``cv2.FaceDetectorYN``
        """
        model = getattr(self._local, "model", None)
        if model is None:
            path = weights.resolve(self.spec)
            model = cv2.FaceDetectorYN.create(
                str(path),
                "",
                (320, 320),
                self.score_threshold,
                self.nms_threshold,
                self.top_k,
            )
            self._local.model = model
            self._local.size = None
        return model

    def detect(self, image: NDArray[Any]) -> list[FacialArea]:
        model = self._model

        # YuNet regenerates its anchor priors on every 'setInputSize', so only call it when
        # the frame size actually changes. The dataset mixes sizes, but consecutive frames of
        # the same video usually agree.
        size = (image.shape[1], image.shape[0])
        if self._local.size != size:
            model.setInputSize(size)
            self._local.size = size

        # On no detection this returns (retval, None) rather than an empty array.
        found, detections = model.detect(image)
        if not found or detections is None or len(detections) == 0:
            return []

        faces = [
            FacialArea(
                x=int(round(float(row[0]))),
                y=int(round(float(row[1]))),
                w=int(round(float(row[2]))),
                h=int(round(float(row[3]))),
                landmarks=np.asarray(row[4:14], dtype=np.float32).reshape(5, 2),
                confidence=float(row[_DETECTION_COLUMNS - 1]),
            )
            for row in detections
        ]
        faces.sort(key=lambda face: face.confidence or 0.0, reverse=True)
        return faces


# The detector the pipeline shares. Constructing it is free; the weights load on first use.
default_detector = YuNetDetector()
