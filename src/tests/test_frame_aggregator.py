"""
Tests for the pixel-level aggregators in ``meles.frame_aggregation``.

The frame-wise de-anonymizations are aggregators that aggregate nothing, which makes two properties
load-bearing rather than incidental. They have to return one frame per input, because
``aggregate_frame`` reads a matching count as "this worked frame-wise" and keeps the original frame
names. And they have to return *new* arrays, because the same rule reads "the input handed back" as
"no new pixels" and then writes no images at all -- a de-anonymizer that returned its input in place
would silently score as the anonymization it was supposed to undo.
"""

import numpy as np
import pytest

from meles.frame_aggregation.aggregator import NoFrameAggregator
from meles.frame_aggregation.resample import BicubicFrameAggregator

SIZE = 64


@pytest.fixture
def blocky() -> list[np.ndarray]:
    """Frames pixelated into hard 8x8 blocks, the artefact the resampling is meant to smooth out."""
    rng = np.random.default_rng(0)
    blocks = rng.integers(0, 256, size=(3, 8, 8, 3), dtype=np.uint8)
    return [np.kron(block, np.ones((SIZE // 8, SIZE // 8, 1), np.uint8)) for block in blocks]


def test_the_baseline_hands_its_frames_straight_back() -> None:
    """'none' has to return the very objects it was given -- that is what marks it as a no-op."""
    images = [np.zeros((SIZE, SIZE, 3), np.uint8) for _ in range(3)]

    aggregated = NoFrameAggregator().aggregate(images)

    assert all(a is b for a, b in zip(images, aggregated))


def test_bicubic_returns_one_new_frame_per_input(blocky: list[np.ndarray]) -> None:
    """Frame-wise means the count and the shape survive; de-anonymizing means the pixels do not."""
    aggregated = BicubicFrameAggregator().aggregate(blocky)

    assert len(aggregated) == len(blocky)
    assert not any(a is b for a, b in zip(blocky, aggregated))
    for before, after in zip(blocky, aggregated):
        assert after.shape == before.shape
        assert after.dtype == np.uint8
        assert not np.array_equal(after, before)


def test_bicubic_softens_the_block_edges(blocky: list[np.ndarray]) -> None:
    """
    The point of the method: pixelation leaves hard block edges, and scaling down and back up with
    a smooth kernel interpolates across them. Measured as the largest step between two neighbouring
    pixels, which is what an edge is.
    """
    aggregated = BicubicFrameAggregator().aggregate(blocky)

    for before, after in zip(blocky, aggregated):
        step_before = np.abs(np.diff(before.astype(int), axis=1)).max()
        step_after = np.abs(np.diff(after.astype(int), axis=1)).max()
        assert step_after < step_before


@pytest.mark.parametrize("factor", [0.1, 0.5, 0.9])
def test_the_factor_never_rounds_a_frame_away(factor: float) -> None:
    """A frame small enough for the scaled size to round to zero would fail inside cv2.resize."""
    tiny = [np.zeros((4, 4, 3), np.uint8)]

    aggregated = BicubicFrameAggregator(factor=factor).aggregate(tiny)

    assert aggregated[0].shape == (4, 4, 3)
