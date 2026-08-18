"""
Tests for the anonymizers and the box clipping they all share.

Pure geometry, no weights and no dataset needed. Worth having because ``facial_region`` is the
one place a detected box meets a numpy slice: a negative coordinate or a box reaching past the
edge would silently produce an empty slice, and an empty slice means a frame that was never
actually anonymized.
"""

import numpy as np
import pytest

from meles.anonymization.anonymizer import (
    BlackBarAnonymizer,
    GaussianBlurAnonymizer,
    NoAnonymizer,
    PixelationAnonymizer,
    facial_region,
)
from meles.common import FacialArea

ANONYMIZERS = {
    "none": NoAnonymizer(),
    "gaussianblur": GaussianBlurAnonymizer(),
    "pixelation": PixelationAnonymizer(),
    "blackbar": BlackBarAnonymizer(),
}


@pytest.fixture
def image() -> np.ndarray:
    return np.full((100, 120, 3), 200, dtype=np.uint8)


@pytest.mark.parametrize(
    "area",
    [
        FacialArea(10, 10, 30, 30),
        FacialArea(0, 0, 120, 100),
        FacialArea(100, 80, 60, 60),
        FacialArea(-20, -20, 50, 50),
        FacialArea(200, 200, 10, 10),
        FacialArea(10, 10, 0, 0),
    ],
    ids=["inside", "exact", "overhang", "negative", "outside", "empty"],
)
def test_facial_region_always_yields_a_valid_slice(image: np.ndarray, area: FacialArea) -> None:
    x, y, w, h = facial_region(image, area)

    assert 0 <= x <= image.shape[1]
    assert 0 <= y <= image.shape[0]
    assert w >= 0 and h >= 0
    assert x + w <= image.shape[1]
    assert y + h <= image.shape[0]
    # The slice must be indexable without raising, whatever the box was.
    assert image[y : y + h, x : x + w].shape == (h, w, 3)


def test_black_bar_masks_the_region(image: np.ndarray) -> None:
    area = FacialArea(20, 30, 40, 25)

    anonymized = BlackBarAnonymizer().apply(image, area)

    assert anonymized[30:55, 20:60].max() == 0
    # 'cv2.rectangle' fills inclusively of both corners, so it covers one row and one column
    # more than the half-open slice the other anonymizers write to. Asserted rather than fixed:
    # changing it would shift every anonymized frame by a pixel for no benefit.
    outside = anonymized.copy()
    outside[30:56, 20:61] = 200
    np.testing.assert_array_equal(outside, image)


def test_pixelation_reduces_detail_without_moving_the_region(image: np.ndarray) -> None:
    detailed = image.copy()
    detailed[30:60, 20:50] = np.random.default_rng(0).integers(0, 256, (30, 30, 3), dtype=np.uint8)

    anonymized = PixelationAnonymizer(blocks=4).apply(detailed, FacialArea(20, 30, 30, 30))

    assert anonymized.shape == detailed.shape
    region = anonymized[30:60, 20:50]
    # Four blocks per axis over 30 pixels means far fewer distinct colours than before.
    assert len({tuple(pixel) for pixel in region.reshape(-1, 3)}) <= 16
    np.testing.assert_array_equal(anonymized[:30], detailed[:30])


def test_gaussian_blur_changes_the_region_only(image: np.ndarray) -> None:
    detailed = image.copy()
    detailed[30:60, 20:50] = np.random.default_rng(1).integers(0, 256, (30, 30, 3), dtype=np.uint8)

    anonymized = GaussianBlurAnonymizer().apply(detailed, FacialArea(20, 30, 30, 30))

    assert not np.array_equal(anonymized[30:60, 20:50], detailed[30:60, 20:50])
    np.testing.assert_array_equal(anonymized[60:], detailed[60:])


@pytest.mark.parametrize("name", sorted(ANONYMIZERS))
def test_anonymizers_do_not_mutate_their_input(name: str, image: np.ndarray) -> None:
    """
    'anonymize_frame.py' feeds the same frame through one anonymizer per detected face, so an
    in-place implementation would make the result depend on how many faces were found.
    """
    original = image.copy()

    ANONYMIZERS[name].apply(image, FacialArea(20, 30, 40, 25))

    np.testing.assert_array_equal(image, original)


@pytest.mark.parametrize("name", sorted(ANONYMIZERS))
def test_anonymizers_preserve_shape_and_dtype(name: str, image: np.ndarray) -> None:
    anonymized = ANONYMIZERS[name].apply(image, FacialArea(20, 30, 40, 25))

    assert anonymized.shape == image.shape
    assert anonymized.dtype == image.dtype


@pytest.mark.parametrize("name", sorted(ANONYMIZERS))
def test_anonymizers_tolerate_a_degenerate_box(name: str, image: np.ndarray) -> None:
    anonymized = ANONYMIZERS[name].apply(image, FacialArea(10, 10, 0, 0))
    assert anonymized.shape == image.shape
