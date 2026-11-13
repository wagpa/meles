"""
Combining the frames of a group before anything has looked at them.

This is one of the three places the experiment can aggregate, and the earliest: many anonymized
observations become the image a recognizer then sees once. Its counterpart at the other end of the
recognizer is :mod:`meles.embedding_aggregation`, where each observation is recognized on its own
first and the vectors are combined instead.

Two kinds of thing live here, and the difference matters when reading results:

- a **real aggregation** returns fewer frames than it was given, usually one;
- a **frame-wise de-anonymization** returns one frame per input and combines nothing. Resampling
  is of this kind: it is an attack on the anonymization, not on the shortage of observations.

Returning the input unchanged means "no new pixels", which is what lets the baseline cost nothing
on disk - the stage can point its rows at the frames that are already there instead of copying
them.
"""

from collections.abc import Sequence
from typing import Protocol

import cv2
import numpy as np
from numpy.typing import NDArray


class FrameAggregator(Protocol):
    """Combines the frames of one group into one or more frames."""

    def aggregate(self, images: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        """
        :param images: The group's frames, BGR, at their native resolution
        :return: The frames the group becomes, which may be the input itself when nothing changed
        """
        ...


class NoFrameAggregator(FrameAggregator):
    """The baseline: the frames are handed back untouched."""

    def aggregate(self, images: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        return images


class ResampleFrameAggregator(FrameAggregator):
    """
    De-anonymizes frame by frame: every frame is scaled down and straight back up.

    This combines nothing - one frame goes in, one comes out - which is why it returns as many
    frames as it was given.

    Scaling down with ``INTER_AREA`` averages each neighbourhood into one pixel and scaling back up
    interpolates between them, which turns the hard block edges pixelation leaves behind into a
    smooth face again. Nothing the anonymization destroyed is recovered; what changes is how much
    of what survived a recognizer can still use.

    The factor is a constant of the method rather than something fitted. Fitting it against the
    clear frames, as the attack this reproduces does, is fitting against the answer: the clear
    frames are what the gallery holds. Sweep it by registering more entries instead.
    """

    def __init__(self, factor: float = 0.5, interpolation: int = cv2.INTER_AREA) -> None:
        """
        :param factor: How far down each frame is scaled before being scaled back
        :param interpolation: The kernel used on the way back up
        """
        self.factor = factor
        self.interpolation = interpolation

    def resample(self, image: NDArray[np.uint8]) -> NDArray[np.uint8]:
        """
        :param image: One frame, BGR
        :return: The frame, scaled down and back up again
        """
        height, width = image.shape[:2]
        # At least one pixel each way: a frame small enough to round to zero would fail in resize.
        reduced = cv2.resize(
            image,
            (max(1, int(width * self.factor)), max(1, int(height * self.factor))),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(reduced, (width, height), interpolation=self.interpolation)
        return np.asarray(restored, dtype=np.uint8)

    def aggregate(self, images: Sequence[NDArray[np.uint8]]) -> Sequence[NDArray[np.uint8]]:
        return [self.resample(image) for image in images]


class BicubicFrameAggregator(ResampleFrameAggregator):
    """Resamples back up with bicubic interpolation rather than area averaging."""

    def __init__(self, factor: float = 0.5) -> None:
        """
        :param factor: How far down each frame is scaled before being scaled back
        """
        super().__init__(factor=factor, interpolation=cv2.INTER_CUBIC)


# Every frame aggregation the experiment can name. Constructing an entry has to stay free of side
# effects, because the whole dict is built at import time.
FRAME_AGGREGATORS: dict[str, FrameAggregator] = {
    "none": NoFrameAggregator(),
    "resample": ResampleFrameAggregator(),
    "bicubic": BicubicFrameAggregator(),
}


def resolve(method: str) -> FrameAggregator:
    """
    The frame aggregator a configuration key names.

    :param method: The key, for example 'bicubic'
    :return: The shared aggregator instance
    :raises ValueError: If nothing is registered under that key
    """
    aggregator = FRAME_AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown frame aggregation {method!r}, expected one of {sorted(FRAME_AGGREGATORS)}")
    return aggregator
