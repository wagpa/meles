from dataclasses import dataclass
from typing import TypeAlias, Any, Sequence, List, Union, IO

import numpy as np
from numpy.typing import NDArray

Image: TypeAlias = NDArray[np.uint8]

Images: TypeAlias = Sequence[Image]

Frame: TypeAlias = Union[str, NDArray[Any], IO[bytes]]

Frames: TypeAlias = Sequence[Frame]

OptEmbeddings: TypeAlias = List[Union[List[float], None]]

Embeddings: TypeAlias = List[List[float]]


@dataclass(frozen=True)
class FacialArea:
    """
    A detected face: its bounding box in image coordinates and, when the detector provides
    them, the five landmarks used to align the face before embedding it.

    This replaces DeepFace's ``FacialAreaRegion``. It lives here rather than next to the
    detector so that ``meles.anonymization`` keeps depending on nothing but the box.

    The landmarks are ``(5, 2)`` float32 in the order ``cv2.FaceDetectorYN`` emits, which is
    also the order ``meles.detection.alignment.ARCFACE_TEMPLATE`` expects: eye at
    image-left, eye at image-right, nose tip, mouth corner at image-left, mouth corner at
    image-right. They stay sub-pixel on purpose -- rounding them to whole pixels before
    fitting the similarity transform visibly degrades the 112x112 crop.
    """

    x: int
    y: int
    w: int
    h: int
    landmarks: NDArray[np.float32] | None = None
    confidence: float | None = None
