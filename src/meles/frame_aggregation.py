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
from typing import TYPE_CHECKING, Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

from meles.alignment import align_face, letterbox
from meles.detection import FacialArea

if TYPE_CHECKING:
    from aggface.model import AggFace


class FrameAggregator(Protocol):
    """Combines the frames of one group into one or more frames."""

    # Whether the output frames are already aligned 112x112 faces rather than frames at their native resolution.
    aligned: bool = False

    def aggregate(
        self,
        images: Sequence[NDArray[np.uint8]],
        areas: Sequence[FacialArea | None],
        anonymization: str,
    ) -> Sequence[NDArray[np.uint8]]:
        """
        :param images: The group's frames, BGR, at their native resolution and already anonymized
        :param areas: The face found in the clear version of each frame, or None where there was none
        :param anonymization: The key of the anonymization the frames carry, for aggregators that depend on it
        :return: The frames the group becomes, which may be the input itself when nothing changed
        """
        ...


class NoFrameAggregator(FrameAggregator):
    """The baseline: the frames are handed back untouched."""

    def aggregate(
        self,
        images: Sequence[NDArray[np.uint8]],
        areas: Sequence[FacialArea | None],
        anonymization: str,
    ) -> Sequence[NDArray[np.uint8]]:
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

    def aggregate(
        self,
        images: Sequence[NDArray[np.uint8]],
        areas: Sequence[FacialArea | None],
        anonymization: str,
    ) -> Sequence[NDArray[np.uint8]]:
        return [self.resample(image) for image in images]


class BicubicFrameAggregator(ResampleFrameAggregator):
    """Resamples back up with bicubic interpolation rather than area averaging."""

    def __init__(self, factor: float = 0.5) -> None:
        """
        :param factor: How far down each frame is scaled before being scaled back
        """
        super().__init__(factor=factor, interpolation=cv2.INTER_CUBIC)


class AggFaceFrameAggregator(FrameAggregator):
    """
    Reconstructs a face from many anonymized observations of it with a trained AggFace model.

    The frames are aligned on the clear frames' faces and the model turns each run of them into one 112x112
    face. The model is conditioned on the anonymization, so the attacker is assumed to recognise which one was
    used. The checkpoint is the one of seed 0 at its best epoch.

    Neither torch nor the model is loaded until the first call, because the registry is built at import time.
    """

    aligned = True

    def __init__(self, seed: int = 0, frames: int | None = None) -> None:
        """
        :param seed: The seed of the trained checkpoint
        :param frames: How many consecutive frames make one reconstruction, or None for the whole group
        """
        self.seed = seed
        self.frames = frames
        self._model: AggFace | None = None
        self._methods: list[str] = []

    def load(self) -> tuple["AggFace", list[str]]:
        """
        :return: The model and the anonymizations it was trained on, loaded on first use
        """
        if self._model is None:
            from aggface.model import checkpoint_path
            from aggface.training import load_model

            checkpoint = checkpoint_path("voxceleb2", self.seed)
            best = checkpoint.with_suffix(f".best{checkpoint.suffix}")
            self._model, metadata = load_model(best if best.exists() else checkpoint)
            self._methods = list(metadata["methods"])
        return self._model, self._methods

    def aggregate(
        self,
        images: Sequence[NDArray[np.uint8]],
        areas: Sequence[FacialArea | None],
        anonymization: str,
    ) -> Sequence[NDArray[np.uint8]]:
        import torch

        from aggface.tensors import to_image, to_tensor
        from aggface.training import reconstruct

        model, methods = self.load()
        method = methods.index(anonymization)

        # The frames are already anonymized, so they are only aligned; a frame without a face is used whole.
        crops: list[Any] = [
            letterbox(image) if area is None else align_face(image, area.landmarks)
            for image, area in zip(images, areas)
        ]
        tensors = [to_tensor(crop) for crop in crops]

        step = self.frames or len(tensors)
        runs = [tensors[start : start + step] for start in range(0, len(tensors), step)]
        # Runs of equal length go through together; a shorter final run needs a call of its own.
        batches = [runs] if len({len(run) for run in runs}) == 1 else [[run] for run in runs]
        reconstructions = [
            reconstruct(model, torch.stack([torch.stack(run) for run in batch]), method) for batch in batches
        ]
        return [to_image(face) for faces in reconstructions for face in faces]


# Every frame aggregation the experiment can name. Constructing an entry has to stay free of side
# effects, because the whole dict is built at import time.
FRAME_AGGREGATORS: dict[str, FrameAggregator] = {
    "none": NoFrameAggregator(),
    "resample": ResampleFrameAggregator(),
    "bicubic": BicubicFrameAggregator(),
    "aggface": AggFaceFrameAggregator(),
    "aggface8": AggFaceFrameAggregator(frames=8),
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
