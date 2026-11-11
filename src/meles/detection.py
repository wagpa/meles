"""
Locating the face in a frame.

Both the anonymization and the recognition stage need the same thing — where the face is, and
where its eyes, nose and mouth corners are — so they share one detector. YuNet is a 227 KiB ONNX
model run through OpenCV, and unlike a plain box detector it returns the five landmarks the
recognition models need to align a crop.

A frame of the corpus is a face track: it shows exactly one subject by construction, so there is
no decision to make about whether a face is present. The detector is therefore asked for its
single highest-scoring candidate with no score threshold, which leaves nothing to tune.
"""

from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from meles import weights

# The YuNet weights from the OpenCV model zoo. The '*.onnx' files there are stored with git-lfs,
# so the usual 'raw.githubusercontent.com' URL serves a 131-byte pointer file rather than the
# model; the 'media.githubusercontent.com/media' form serves the real blob.
YUNET = weights.WeightSpec(
    file_name="face_detection_yunet_2023mar.onnx",
    url=(
        "https://media.githubusercontent.com/media/opencv/opencv_zoo/main/models/"
        "face_detection_yunet/face_detection_yunet_2023mar.onnx"
    ),
    sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
)


@dataclass(frozen=True)
class FacialArea:
    """
    A detected face: its bounding box in image coordinates, its landmarks, and its score.

    The landmarks are (5, 2) float32 in the order YuNet emits, which is also the order
    :data:`meles.alignment.ARCFACE_TEMPLATE` expects: eye at image-left, eye at image-right, nose
    tip, mouth corner at image-left, mouth corner at image-right. They stay sub-pixel on purpose,
    because rounding them before fitting the similarity transform degrades the aligned crop.
    """

    x: int
    y: int
    w: int
    h: int
    landmarks: NDArray[np.float32]
    confidence: float


class Detector(Protocol):
    """Locates the subject's face in a frame."""

    def detect(self, image: NDArray[np.uint8]) -> FacialArea | None:
        """
        Find the subject's face.

        :param image: The frame to search, BGR
        :return: The detected face, or None if the detector returned nothing
        """
        ...


class YuNetDetector(Detector):
    """
    YuNet, run through OpenCV's ``cv2.FaceDetectorYN``.

    The model wraps a ``cv::dnn::Net``, which keeps mutable state across a forward pass and is not
    thread-safe, so a detector belongs to one thread. It is built on first use, which keeps
    constructing a detector free for code that may never call it.
    """

    def __init__(self, spec: weights.WeightSpec = YUNET):
        """
        :param spec: The weights to run
        """
        self.spec = spec
        self._model: Any = None

    @property
    def model(self) -> Any:
        """
        The underlying detector, built on first use.

        Created with no score threshold and a budget of one detection, so that it returns its
        highest-scoring candidate whatever that score is.

        :return: The ``cv2.FaceDetectorYN``
        """
        if self._model is None:
            self._model = cv2.FaceDetectorYN.create(
                str(weights.resolve(self.spec)),
                "",
                (0, 0),
                score_threshold=0.0,
                nms_threshold=0.3,
                top_k=1,
            )
        return self._model

    def detect(self, image: NDArray[np.uint8]) -> FacialArea | None:
        model = self.model
        model.setInputSize((image.shape[1], image.shape[0]))

        # On no detection this returns (retval, None) rather than an empty array.
        found, detections = model.detect(image)
        if not found or detections is None or len(detections) == 0:
            return None

        # Columns are the box, five landmarks as (x, y) pairs, and the score.
        row = detections[0]
        return FacialArea(
            x=int(round(float(row[0]))),
            y=int(round(float(row[1]))),
            w=int(round(float(row[2]))),
            h=int(round(float(row[3]))),
            landmarks=np.asarray(row[4:14], dtype=np.float32).reshape(5, 2),
            confidence=float(row[14]),
        )


# The detector the pipeline shares. Constructing it is free; the weights load on first use.
default_detector = YuNetDetector()
