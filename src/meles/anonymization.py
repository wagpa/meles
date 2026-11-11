"""
The anonymizations the experiment attacks.

Each one replaces the pixels inside the detected face box and leaves the rest of the frame alone.
Their strengths are defined relative to the size of that box rather than in pixels, so the same
method applies equally to a large and a small face.

The registry lives here rather than in the workflow script because training uses it too: a model
is trained to undo an anonymization and must see exactly the one the experiment later applies.
"""

from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from meles.detection import FacialArea


def facial_region(image: NDArray[np.uint8], area: FacialArea) -> tuple[int, int, int, int]:
    """
    Clip a detected face box to the bounds of the image so it can be used to slice it.

    :param image: The frame the face was detected in
    :param area: The detected face
    :return: The clipped region as (x, y, w, h)
    """
    height, width = image.shape[:2]
    x = max(0, min(area.x, width))
    y = max(0, min(area.y, height))
    return x, y, max(0, min(area.w, width - x)), max(0, min(area.h, height - y))


class Anonymizer(Protocol):
    """Hides the face in a frame."""

    def apply(self, image: NDArray[np.uint8], area: FacialArea) -> NDArray[np.uint8]:
        """
        Anonymize the detected face.

        :param image: The frame to anonymize
        :param area: The face to hide
        :return: The anonymized frame, which is the input itself when nothing changed
        """
        ...


class NoAnonymizer(Anonymizer):
    """The baseline: the frame is left exactly as it is."""

    def apply(self, image: NDArray[np.uint8], area: FacialArea) -> NDArray[np.uint8]:
        return image


class GaussianBlurAnonymizer(Anonymizer):
    """Blurs the face with a kernel sized as a fraction of the face itself."""

    def __init__(self, strength: float = 0.25):
        """
        :param strength: The blur kernel as a fraction of the longer side of the face box
        """
        self.strength = strength

    def apply(self, image: NDArray[np.uint8], area: FacialArea) -> NDArray[np.uint8]:
        x, y, w, h = facial_region(image, area)
        if w == 0 or h == 0:
            return image

        # OpenCV needs an odd kernel size; sigma=0 derives the deviation from the kernel.
        kernel = max(3, int(max(w, h) * self.strength) | 1)
        anonymized = image.copy()
        anonymized[y : y + h, x : x + w] = cv2.GaussianBlur(anonymized[y : y + h, x : x + w], (kernel, kernel), 0)
        return anonymized


class PixelationAnonymizer(Anonymizer):
    """Reduces the face to a fixed number of blocks across, whatever its size in pixels."""

    def __init__(self, blocks: int = 8):
        """
        :param blocks: How many blocks the face is reduced to along each axis
        """
        self.blocks = blocks

    def apply(self, image: NDArray[np.uint8], area: FacialArea) -> NDArray[np.uint8]:
        x, y, w, h = facial_region(image, area)
        if w == 0 or h == 0:
            return image

        # INTER_AREA averages all pixels of a block, INTER_NEAREST keeps the block edges sharp.
        reduced = cv2.resize(
            image[y : y + h, x : x + w],
            (min(self.blocks, w), min(self.blocks, h)),
            interpolation=cv2.INTER_AREA,
        )
        anonymized = image.copy()
        anonymized[y : y + h, x : x + w] = cv2.resize(reduced, (w, h), interpolation=cv2.INTER_NEAREST)
        return anonymized


class BlackBarAnonymizer(Anonymizer):
    """Fills the face box with black, destroying it rather than degrading it."""

    def apply(self, image: NDArray[np.uint8], area: FacialArea) -> NDArray[np.uint8]:
        x, y, w, h = facial_region(image, area)
        anonymized = image.copy()
        cv2.rectangle(anonymized, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
        return anonymized


# Every anonymization the experiment can name. Constructing an entry has to stay free of side
# effects, because the whole dict is built at import time.
ANONYMIZERS: dict[str, Anonymizer] = {
    "none": NoAnonymizer(),
    "gaussianblur": GaussianBlurAnonymizer(),
    "pixelation": PixelationAnonymizer(),
    "blackbar": BlackBarAnonymizer(),
}


def resolve(method: str) -> Anonymizer:
    """
    The anonymizer a configuration key names.

    :param method: The key, for example 'pixelation'
    :return: The shared anonymizer instance
    :raises ValueError: If nothing is registered under that key
    """
    anonymizer = ANONYMIZERS.get(method)
    if anonymizer is None:
        raise ValueError(f"Unknown anonymization method {method!r}, expected one of {sorted(ANONYMIZERS)}")
    return anonymizer
