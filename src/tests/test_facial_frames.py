"""
``FacialFrames`` builds one training item per identity, and the ways it can go wrong are all
silent: the target leaking into its own inputs, a validation item redrawn every epoch, or a flip
applied to one end of the item only. These pin each of those.

A stub detector stands in for YuNet so the tests need neither the model weights nor the dataset.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from meles.common import FacialArea
from meles.detection import Detector
from meles.training.facial import FacialFrames, split_identities

IDENTITIES = ["alice", "bob", "carol"]
VIDEOS = ["0", "1", "2"]
FRAMES_PER_VIDEO = 4
SIZE = 112


class StubDetector(Detector):
    """Always finds one centred face, with the landmarks a similarity fit needs."""

    def detect(self, image):
        height, width = image.shape[:2]
        box = FacialArea(
            x=width // 4,
            y=height // 4,
            w=width // 2,
            h=height // 2,
            landmarks=np.array(
                [
                    [width * 0.38, height * 0.45],
                    [width * 0.62, height * 0.45],
                    [width * 0.50, height * 0.58],
                    [width * 0.40, height * 0.72],
                    [width * 0.60, height * 0.72],
                ],
                dtype=np.float32,
            ),
            confidence=0.99,
        )
        return [box]


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, Path]:
    """A synthetic identity/video/frame tree, plus a structure and a target per identity."""
    rows = []
    rng = np.random.default_rng(0)
    for identity in IDENTITIES:
        for video in VIDEOS:
            directory = tmp_path / "ytfaces" / identity / video
            directory.mkdir(parents=True)
            for index in range(FRAMES_PER_VIDEO):
                name = f"aligned_detect_{video}.{index}.jpg"
                cv2.imwrite(str(directory / name), rng.integers(0, 255, (80, 80, 3), dtype=np.uint8))
                rows.append(
                    {
                        "identity": identity,
                        "video": video,
                        "frame": name,
                        "path": f"ytfaces/{identity}/{video}/{name}",
                    }
                )

    structure = pd.DataFrame(rows)
    # Every identity's target is the first frame of video "1", so there is a video to exclude.
    targets = (
        structure[(structure["video"] == "1") & (structure["frame"].str.endswith(".0.jpg"))]
        .reset_index(drop=True)
        .copy()
    )
    return structure, targets, tmp_path


def build(dataset, **kwargs) -> FacialFrames:
    structure, targets, root = dataset
    options = {"anonymizer": "pixelation", "frames": 4, "detector": StubDetector(), "size": SIZE}
    return FacialFrames(structure, targets, root, **(options | kwargs))


def test_item_has_the_shape_facial_forward_takes(dataset):
    anonymized, clear = build(dataset)[0]
    assert anonymized.shape == (4, 3, SIZE, SIZE)
    assert clear.shape == (3, SIZE, SIZE)


def test_tensors_are_in_the_tanh_range(dataset):
    anonymized, clear = build(dataset)[0]
    for tensor in (anonymized, clear):
        assert tensor.dtype.is_floating_point
        assert -1.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_default_collate_batches_without_a_custom_fn(dataset):
    loader = DataLoader(build(dataset), batch_size=3)
    anonymized, clear = next(iter(loader))
    assert anonymized.shape == (3, 4, 3, SIZE, SIZE)
    assert clear.shape == (3, 3, SIZE, SIZE)


def test_one_item_per_identity(dataset):
    frames = build(dataset)
    assert len(frames) == len(IDENTITIES)
    assert frames.identities == sorted(IDENTITIES)


def test_target_never_appears_among_its_own_inputs(dataset):
    """The decoder could otherwise copy an input straight through."""
    frames = build(dataset, exclude_target_video=False)
    for identity, pool in frames.pool.items():
        assert frames.target[identity] not in pool


def test_excluding_the_target_video_drops_the_whole_shot(dataset):
    """Same-shot inputs share a background, which is a shortcut around the face."""
    frames = build(dataset)
    for identity, pool in frames.pool.items():
        assert all(f"/{identity}/1/" not in path for path in pool)
        assert pool, "excluding one video should not empty the pool"


def test_fewer_frames_than_n_still_fills_the_item(dataset):
    """An identity the attacker saw less of is a truthful item, not a broken one."""
    frames = build(dataset, frames=32)
    anonymized, _ = frames[0]
    assert anonymized.shape[0] == 32


def test_deterministic_draws_the_same_item_every_epoch(dataset):
    """A validation loss over freshly drawn frames is not comparable between epochs."""
    frames = build(dataset, deterministic=True)
    first, _ = frames[0]
    second, _ = frames[0]
    assert first.equal(second)


def test_non_deterministic_redraws(dataset):
    """Training items should vary, which is where this dataset's augmentation comes from."""
    frames = build(dataset, frames=2, flip=False)
    draws = {frames[0][0].numpy().tobytes() for _ in range(25)}
    assert len(draws) > 1


def test_anonymization_actually_changes_the_input(dataset):
    """A 'none' baseline aside, an item whose input equals its target teaches copying."""
    pixelated = build(dataset, deterministic=True)[0][0]
    clear = build(dataset, anonymizer="none", deterministic=True)[0][0]
    assert not pixelated.equal(clear)


def test_identities_can_be_restricted_to_one_split(dataset):
    frames = build(dataset, identities=["alice"])
    assert frames.identities == ["alice"]
    assert len(frames) == 1


def test_split_identities_holds_out_identities_not_frames():
    train, validation = split_identities([f"id{i}" for i in range(20)], validation=0.25, seed=0)
    assert len(validation) == 5
    assert len(train) == 15
    assert not set(train) & set(validation)
    assert sorted(train + validation) == sorted(f"id{i}" for i in range(20))


def test_split_identities_is_reproducible():
    first = split_identities(IDENTITIES * 4, validation=0.5, seed=7)
    assert first == split_identities(IDENTITIES * 4, validation=0.5, seed=7)
