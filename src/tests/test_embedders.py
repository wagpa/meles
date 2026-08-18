"""
Tests for the detector, the ONNX sessions and the three embedders.

The load-bearing test here is ``test_separates_identities``: it is the only check that would
notice a model being fed the wrong channel order or the wrong pixel scaling. Both mistakes
produce embeddings that are finite, correctly shaped and completely useless, so shape
assertions alone would pass.
"""

import cv2
import numpy as np
import pandas as pd
import pytest
from helpers import paths, separation

from meles import weights
from meles.detection import YuNetDetector
from meles.recognizer import models
from meles.recognizer.frame import FrameEmbedder
from meles.recognizer.session import OnnxSession, providers

# The mean same-identity cosine similarity minus the mean cross-identity one, below which a
# model is not usefully discriminating. All three measure around 0.71 to 0.78 on this sample,
# so 0.40 leaves room for sampling noise while still failing hard on a preprocessing mistake.
MIN_SEPARATION = 0.40


def test_providers_are_available() -> None:
    """Whatever provider list is chosen must consist of providers this build actually has."""
    available = set(__import__("onnxruntime").get_available_providers())
    assert set(providers()) <= available
    assert providers(), "at least the CPU provider must be usable"


@pytest.mark.weights
@pytest.mark.dataset
def test_detector_finds_a_face_in_every_frame(detector: YuNetDetector, images: list[np.ndarray]) -> None:
    """
    Every frame must yield at least one detection. A miss is what actually hurts: the
    anonymizer leaves an undetected frame untouched, which weakens the effect the experiment is
    trying to measure.

    More than one detection is normal and not a problem -- these are video stills that often
    catch a bystander, and the anonymizer masks every region it is given.
    """
    counts = [len(detector.detect(image)) for image in images]

    assert counts.count(0) == 0, "some frames had no detectable face"


@pytest.mark.weights
@pytest.mark.dataset
def test_detect_best_picks_the_subject_not_a_bystander(detector: YuNetDetector, images: list[np.ndarray]) -> None:
    """
    The frames are cropped around one subject, whose face is the largest thing in them. Picking
    the most confident detection instead of the largest lands on a small background face often
    enough to matter, so this pins the selection down.
    """
    for image in images:
        height, width = image.shape[:2]
        area = detector.detect_best(image)
        assert area is not None

        centre_x = (area.x + area.w / 2) / width
        centre_y = (area.y + area.h / 2) / height
        assert abs(centre_x - 0.5) < 0.2, f"selected face is off-centre horizontally: {centre_x:.2f}"
        assert abs(centre_y - 0.5) < 0.2, f"selected face is off-centre vertically: {centre_y:.2f}"
        # The subject fills a good part of the frame; a bystander does not.
        assert area.w * area.h > 0.02 * width * height


@pytest.mark.weights
@pytest.mark.dataset
def test_detected_boxes_and_landmarks_lie_inside_the_frame(detector: YuNetDetector, images: list[np.ndarray]) -> None:
    for image in images[:20]:
        height, width = image.shape[:2]
        area = detector.detect_best(image)
        assert area is not None

        assert area.w > 0 and area.h > 0
        assert -width <= area.x <= width and -height <= area.y <= height
        assert area.landmarks.shape == (5, 2)
        assert np.isfinite(area.landmarks).all()
        assert 0.0 <= area.confidence <= 1.0


@pytest.mark.weights
def test_detector_reports_nothing_for_a_blank_image(detector: YuNetDetector) -> None:
    assert detector.detect(np.zeros((200, 200, 3), dtype=np.uint8)) == []
    assert detector.detect_best(np.zeros((200, 200, 3), dtype=np.uint8)) is None


@pytest.mark.weights
def test_detector_handles_changing_frame_sizes(detector: YuNetDetector) -> None:
    """The detector caches its input size, so alternating sizes must not confuse it."""
    for size in (64, 200, 64, 331):
        assert detector.detect(np.zeros((size, size, 3), dtype=np.uint8)) == []


@pytest.mark.weights
@pytest.mark.parametrize("spec", list(models.SPECS.values()), ids=list(models.SPECS))
def test_batched_inference_matches_single_images(spec) -> None:
    """
    Two of the three graphs pin their batch dimension to one and are loaded with it relaxed.
    That rewrite is only valid if batching changes nothing, so assert exactly that.
    """
    session = OnnxSession(weights.resolve(spec.weights))
    embedder = FrameEmbedder(spec)

    rng = np.random.default_rng(0)
    faces = [rng.integers(0, 256, (spec.input_size, spec.input_size, 3), dtype=np.uint8) for _ in range(5)]
    blob = embedder._blob(faces)

    batched = session.run(blob)
    singles = np.concatenate([session.run(blob[index : index + 1]) for index in range(len(faces))])

    assert batched.shape == (len(faces), spec.output_dim)
    np.testing.assert_allclose(batched, singles, atol=1e-4)


@pytest.mark.weights
@pytest.mark.dataset
@pytest.mark.parametrize("name", list(models.SPECS))
def test_separates_identities(name: str, crops: tuple[list[np.ndarray], list[str]]) -> None:
    aligned, labels = crops
    spec = models.SPECS[name]
    embedder = FrameEmbedder(spec)

    vectors = embedder.session.run(embedder._blob(aligned))

    assert vectors.shape == (len(aligned), spec.output_dim)
    assert np.isfinite(vectors).all()

    same, cross = separation(vectors, labels)
    assert same - cross > MIN_SEPARATION, (
        f"{name} barely separates identities (same {same:.3f}, cross {cross:.3f}); "
        "this usually means the channel order or the pixel scaling is wrong"
    )


@pytest.mark.weights
@pytest.mark.dataset
@pytest.mark.parametrize("name", list(models.SPECS))
def test_embed_returns_one_entry_per_frame(name: str, frames: pd.DataFrame) -> None:
    """
    'embed_frame.py' assigns the result straight into a dataframe column, so the output has to
    line up with the input row by row -- including for a frame with no detectable face.
    """
    embedder = FrameEmbedder(models.SPECS[name], batch_size=8)
    sample = paths(frames)[:10]

    embeddings = embedder.embed(sample)

    assert len(embeddings) == len(sample)
    assert all(embedding is not None for embedding in embeddings)
    assert all(len(embedding) == models.SPECS[name].output_dim for embedding in embeddings)


@pytest.mark.weights
def test_embed_falls_back_to_the_whole_frame(tmp_path) -> None:
    """
    DeepFace embedded the whole frame when it found no face, so every frame got an embedding.
    That behaviour is preserved: the anonymizers deliberately destroy faces, and dropping the
    frames where detection then fails would change which frames the experiment scores.
    """
    blank = tmp_path / "blank.jpg"
    cv2.imwrite(str(blank), np.full((150, 150, 3), 127, dtype=np.uint8))

    fallback = FrameEmbedder(models.SFACE).embed([str(blank)])
    assert fallback[0] is not None
    assert len(fallback[0]) == models.SFACE.output_dim

    dropped = FrameEmbedder(models.SFACE, on_missing="none").embed([str(blank)])
    assert dropped == [None]


@pytest.mark.weights
def test_embed_reports_an_unreadable_frame_as_missing(tmp_path) -> None:
    broken = tmp_path / "broken.jpg"
    broken.write_bytes(b"not an image")

    assert FrameEmbedder(models.SFACE, on_missing="none").embed([str(broken)]) == [None]


@pytest.mark.weights
@pytest.mark.dataset
def test_embed_accepts_arrays_and_streams(frames: pd.DataFrame) -> None:
    """The Frame alias allows a path, an array or an open stream; all three must work."""
    path = paths(frames)[0]
    embedder = FrameEmbedder(models.SFACE)

    from_path = embedder.embed([path])[0]
    from_array = embedder.embed([cv2.imread(path)])[0]
    with open(path, "rb") as handle:
        from_stream = embedder.embed([handle])[0]

    np.testing.assert_allclose(from_path, from_array, atol=1e-4)
    np.testing.assert_allclose(from_path, from_stream, atol=1e-4)


@pytest.mark.weights
def test_sface_matches_the_opencv_reference_implementation() -> None:
    """
    SFace's preprocessing is not documented anywhere: the graph normalises internally and
    OpenCV's wrapper swaps the channels for you. This asserts the spec reproduces
    'cv2.FaceRecognizerSF.feature', which is the reference implementation for this exact file.
    """
    path = weights.resolve(models.SFACE.weights)
    reference = cv2.FaceRecognizerSF.create(str(path), "")
    embedder = FrameEmbedder(models.SFACE)

    rng = np.random.default_rng(7)
    faces = [rng.integers(0, 256, (112, 112, 3), dtype=np.uint8) for _ in range(4)]

    ours = embedder.session.run(embedder._blob(faces))
    theirs = np.concatenate([reference.feature(face) for face in faces])

    np.testing.assert_allclose(ours, theirs, atol=1e-3)
