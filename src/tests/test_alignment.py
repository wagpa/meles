"""
Tests for the face alignment every embedding model depends on.

The important one is ``test_matches_opencv_align_crop``. Alignment is the one step in this
pipeline where a mistake stays invisible: swap the two eye landmarks and every crop comes out
mirrored, every embedding still looks like a perfectly ordinary vector, and the only symptom
is recognition accuracy that is quietly worse than it should be. Checking our crop against
OpenCV's own implementation pins the landmark order down.
"""

import cv2
import numpy as np
import pytest

from meles import weights
from meles.common import FacialArea
from meles.detection import (
    ARCFACE_TEMPLATE,
    TEMPLATE_SIZE,
    align_face,
    crop_face,
    letterbox,
    prepare_face,
    umeyama,
    whole_frame,
)
from meles.recognizer import models


def test_umeyama_recovers_a_known_similarity_transform() -> None:
    angle, scale, shift = np.deg2rad(23.0), 1.7, np.array([12.0, -5.0])
    rotation = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [2.0, 3.0], [-1.0, 2.0]])
    target = (scale * rotation @ source.T).T + shift

    matrix = umeyama(source, target)

    recovered = (matrix[:, :2] @ source.T).T + matrix[:, 2]
    np.testing.assert_allclose(recovered, target, atol=1e-9)


def test_umeyama_does_not_mirror() -> None:
    """A reflection must be fitted as the nearest rotation, never as a flip."""
    source = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    target = np.array([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0]])

    matrix = umeyama(source, target)

    assert np.linalg.det(matrix[:, :2]) > 0


def test_template_maps_onto_itself() -> None:
    matrix = umeyama(ARCFACE_TEMPLATE, ARCFACE_TEMPLATE)
    np.testing.assert_allclose(matrix[:, :2], np.eye(2), atol=1e-9)
    np.testing.assert_allclose(matrix[:, 2], np.zeros(2), atol=1e-7)


def test_align_face_shape_and_dtype() -> None:
    image = np.random.default_rng(0).integers(0, 256, (200, 240, 3), dtype=np.uint8)
    landmarks = ARCFACE_TEMPLATE * 1.5 + 20

    crop = align_face(image, landmarks)

    assert crop.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 3)
    assert crop.dtype == np.uint8


def test_align_face_honours_the_requested_size() -> None:
    image = np.random.default_rng(1).integers(0, 256, (200, 240, 3), dtype=np.uint8)
    crop = align_face(image, ARCFACE_TEMPLATE * 1.5 + 20, size=224)
    assert crop.shape == (224, 224, 3)


@pytest.mark.weights
@pytest.mark.dataset
def test_matches_opencv_align_crop(detector, images: list[np.ndarray]) -> None:
    """
    Our closed-form alignment must reproduce ``cv2.FaceRecognizerSF.alignCrop``, which
    consumes a raw YuNet detection row. Agreement proves three things at once: that YuNet's
    landmark order matches ARCFACE_TEMPLATE, that the template values are right, and that the
    warp uses the same interpolation.
    """
    recognizer = cv2.FaceRecognizerSF.create(str(weights.resolve(models.SFACE.weights)), "")
    model = detector._model  # the raw cv2 detector, to get the unconverted rows

    compared = 0
    for image in images[:20]:
        model.setInputSize((image.shape[1], image.shape[0]))
        found, rows = model.detect(image)
        if not found or rows is None or len(rows) == 0:
            continue
        row = rows[int(np.argmax(rows[:, 14]))]

        ours = align_face(image, row[4:14].reshape(5, 2)).astype(int)
        theirs = recognizer.alignCrop(image, row.reshape(1, -1)).astype(int)

        difference = np.abs(ours - theirs).mean()
        mirrored = np.abs(ours[:, ::-1] - theirs).mean()

        assert difference < 1.0, f"alignment differs from OpenCV by {difference:.3f} grey levels"
        assert mirrored > 10 * max(difference, 0.1), "mirrored crop is indistinguishable"
        compared += 1

    assert compared >= 5, "not enough faces detected to compare alignment"


def test_letterbox_pads_instead_of_stretching() -> None:
    image = np.full((50, 200, 3), 255, dtype=np.uint8)

    padded = letterbox(image, 112)

    assert padded.shape == (112, 112, 3)
    # A wide image fits by width, so the padding ends up as black bands top and bottom.
    assert padded[0].max() == 0
    assert padded[-1].max() == 0
    assert padded[56].max() == 255


def test_crop_face_clips_a_box_reaching_outside_the_image() -> None:
    image = np.random.default_rng(2).integers(0, 256, (100, 100, 3), dtype=np.uint8)

    crop = crop_face(image, FacialArea(x=80, y=80, w=60, h=60))

    assert crop.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 3)


def test_crop_face_falls_back_to_the_frame_for_a_degenerate_box() -> None:
    image = np.random.default_rng(3).integers(0, 256, (100, 100, 3), dtype=np.uint8)

    crop = crop_face(image, FacialArea(x=10, y=10, w=0, h=0))

    assert crop.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 3)
    assert crop.max() > 0


def test_prepare_face_uses_the_box_when_there_are_no_landmarks() -> None:
    image = np.random.default_rng(4).integers(0, 256, (120, 120, 3), dtype=np.uint8)

    area = FacialArea(x=10, y=10, w=50, h=50)

    np.testing.assert_array_equal(prepare_face(image, area), crop_face(image, area))


def test_whole_frame_covers_the_entire_image() -> None:
    image = np.random.default_rng(5).integers(0, 256, (90, 130, 3), dtype=np.uint8)

    crop, area = whole_frame(image)

    assert crop.shape == (TEMPLATE_SIZE, TEMPLATE_SIZE, 3)
    assert (area.x, area.y, area.w, area.h) == (0, 0, 130, 90)
    assert area.landmarks is None
