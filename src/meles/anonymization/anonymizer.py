from abc import ABC, abstractmethod

import cv2
import numpy as np

from meles.common import FacialArea


def facial_region(
    image: np.ndarray, area: FacialArea
) -> tuple[int, int, int, int]:
    """
    Clips a detected facial area to the bounds of the image so that it can safely be
    used to slice it.

    :param image: The image, the facial area was detected in
    :param area: The detected facial area
    :return: The clipped region as (x, y, w, h)
    """

    height, width = image.shape[:2]
    x = max(0, min(area.x, width))
    y = max(0, min(area.y, height))
    return x, y, max(0, min(area.w, width - x)), max(0, min(area.h, height - y))


class Anonymizer(ABC):
    """Anonymizes a facial area in an image."""

    @abstractmethod
    def apply(self, image: np.ndarray, area: FacialArea) -> np.ndarray:
        """
        Anonymizes the facial area.

        :param image: The image to anonymize
        :param area: The facial area to anonymize
        :return: A copy of the image with the facial anonymized
        """
        raise NotImplementedError


class NoAnonymizer(Anonymizer):
    def apply(self, image: np.ndarray, area: FacialArea) -> np.ndarray:
        return image


class GaussianBlurAnonymizer(Anonymizer):
    strength: float

    def __init__(self, strength: float = 0.25):
        self.strength = strength

    def apply(self, image: np.ndarray, area: FacialArea) -> np.ndarray:
        x, y, w, h = facial_region(image, area)
        if w == 0 or h == 0:
            return image.copy()

        # OpenCV requires an odd kernel size, sigma=0 derives the deviation from the kernel.
        kernel = max(3, int(max(w, h) * self.strength) | 1)
        anonymized = image.copy()
        anonymized[y : y + h, x : x + w] = cv2.GaussianBlur(
            anonymized[y : y + h, x : x + w], (kernel, kernel), 0
        )
        return anonymized


class PixelationAnonymizer(Anonymizer):
    blocks: int = 8

    def __init__(self, blocks: int = 8):
        self.blocks = blocks

    def apply(self, image: np.ndarray, area: FacialArea) -> np.ndarray:
        x, y, w, h = facial_region(image, area)
        if w == 0 or h == 0:
            return image.copy()

        # INTER_AREA averages all pixels of a block, INTER_NEAREST keeps the block edges.
        reduced = cv2.resize(
            image[y : y + h, x : x + w],
            (min(self.blocks, w), min(self.blocks, h)),
            interpolation=cv2.INTER_AREA,
        )
        anonymized = image.copy()
        anonymized[y : y + h, x : x + w] = cv2.resize(
            reduced, (w, h), interpolation=cv2.INTER_NEAREST
        )
        return anonymized


class BlackBarAnonymizer(Anonymizer):
    def __init__(self):
        pass

    def apply(self, image: np.ndarray, area: FacialArea) -> np.ndarray:
        x, y, w, h = facial_region(image, area)
        anonymized = image.copy()
        cv2.rectangle(anonymized, (x, y), (x + w, y + h), (0, 0, 0), thickness=-1)
        return anonymized
