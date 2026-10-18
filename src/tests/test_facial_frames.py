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
from meles.training.facial import CURRICULUM, FacialFrames, split_identities

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
    options = {"curriculum": ["pixelation"], "frames": 4, "detector": StubDetector(), "size": SIZE}
    return FacialFrames(structure, targets, root, **(options | kwargs))


def test_item_has_the_shape_facial_forward_takes(dataset):
    anonymized, clear, _ = build(dataset)[0]
    assert anonymized.shape == (4, 3, SIZE, SIZE)
    assert clear.shape == (3, SIZE, SIZE)


def test_tensors_are_in_the_tanh_range(dataset):
    anonymized, clear, _ = build(dataset)[0]
    for tensor in (anonymized, clear):
        assert tensor.dtype.is_floating_point
        assert -1.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0


def test_default_collate_batches_without_a_custom_fn(dataset):
    loader = DataLoader(build(dataset), batch_size=3)
    anonymized, clear, method = next(iter(loader))
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
    anonymized, _, _ = frames[0]
    assert anonymized.shape[0] == 32


def test_deterministic_draws_the_same_item_every_epoch(dataset):
    """A validation loss over freshly drawn frames is not comparable between epochs."""
    frames = build(dataset, deterministic=True)
    first, _, _ = frames[0]
    second, _, _ = frames[0]
    assert first.equal(second)


def test_non_deterministic_redraws(dataset):
    """Training items should vary, which is where this dataset's augmentation comes from."""
    frames = build(dataset, frames=2, flip=False)
    draws = {frames[0][0].numpy().tobytes() for _ in range(25)}
    assert len(draws) > 1


def test_anonymization_actually_changes_the_input(dataset):
    """A 'none' baseline aside, an item whose input equals its target teaches copying."""
    pixelated = build(dataset, deterministic=True)[0][0]
    clear = build(dataset, curriculum=["none"], deterministic=True)[0][0]
    assert not pixelated.equal(clear)


def test_identities_can_be_restricted_to_one_split(dataset):
    frames = build(dataset, identities=["alice"])
    assert frames.identities == ["alice"]
    assert len(frames) == 1


def test_stage_zero_draws_only_the_first_method(dataset):
    """The curriculum starts on clear frames; nothing harder may leak into stage 0."""
    frames = build(dataset, curriculum=CURRICULUM, stage=0)
    assert frames.active == ["none"]
    assert {frames[i % len(frames)][2] for i in range(40)} == {0}


def test_each_stage_keeps_everything_before_it(dataset):
    """Incremental means cumulative -- a later stage must not drop an earlier method."""
    frames = build(dataset, curriculum=CURRICULUM)
    for stage in range(frames.stages):
        frames.set_stage(stage)
        assert frames.active == list(CURRICULUM[: stage + 1])
        drawn = {frames[i % len(frames)][2] for i in range(120)}
        assert drawn <= set(range(stage + 1))


def test_the_last_stage_reaches_every_method(dataset):
    frames = build(dataset, curriculum=CURRICULUM, stage=len(CURRICULUM) - 1)
    assert {frames[i % len(frames)][2] for i in range(300)} == set(range(len(CURRICULUM)))


def test_method_indexes_the_whole_curriculum_not_the_active_prefix(dataset):
    """A per-method loss is only comparable across stages if the index means one thing."""
    frames = build(dataset, curriculum=CURRICULUM, stage=3)
    assert frames.names.index("blackbar") == 3
    assert frames.stage == 3 and len(frames.active) == 4


def test_a_whole_item_shares_one_anonymization(dataset):
    """A sweep leg pins one method, so an item mixing them is a distribution that never occurs."""
    frames = build(dataset, curriculum=["none", "blackbar"], stage=1, frames=6, flip=False)
    for index in range(len(frames)):
        anonymized, _, method = frames[index]
        # Under 'blackbar' every frame is mostly destroyed; under 'none' none of them is. A mixed
        # item would show both extremes at once, so compare each frame against the item's method.
        darkest = [float(frame.mean()) for frame in anonymized]
        assert (max(darkest) - min(darkest) < 1.5) or method == 0


def test_all_methods_yields_every_identity_under_every_method(dataset):
    """Drawing one method per identity leaves the per-method counts to chance."""
    frames = build(dataset, curriculum=CURRICULUM, stage=3, all_methods=True, deterministic=True)
    assert len(frames) == len(IDENTITIES) * len(CURRICULUM)

    seen = [frames[i][2] for i in range(len(frames))]
    assert sorted(seen) == sorted(list(range(len(CURRICULUM))) * len(IDENTITIES))


def test_all_methods_grows_with_the_stage(dataset):
    frames = build(dataset, curriculum=CURRICULUM, all_methods=True, deterministic=True)
    for stage in range(frames.stages):
        frames.set_stage(stage)
        assert len(frames) == len(IDENTITIES) * (stage + 1)
        assert {frames[i][2] for i in range(len(frames))} == set(range(stage + 1))


def test_all_methods_pairs_the_comparison_on_identical_frames(dataset):
    """Every method of one identity must see the same frames, or the difference is not the method."""
    frames = build(dataset, curriculum=["none", "none"], stage=1, all_methods=True, deterministic=True)
    # Two identical anonymizations over the same identity, so the items must match exactly.
    first, _, method_a = frames[0]
    second, _, method_b = frames[1]
    assert (method_a, method_b) == (0, 1)
    assert first.equal(second)


def test_set_stage_rejects_a_stage_outside_the_curriculum(dataset):
    frames = build(dataset, curriculum=CURRICULUM)
    with pytest.raises(IndexError):
        frames.set_stage(len(CURRICULUM))
    with pytest.raises(IndexError):
        frames.set_stage(-1)


def test_an_empty_curriculum_is_refused(dataset):
    structure, targets, root = dataset
    with pytest.raises(ValueError):
        FacialFrames(structure, targets, root, curriculum=[], detector=StubDetector())


def test_split_identities_holds_out_identities_not_frames():
    train, validation = split_identities([f"id{i}" for i in range(20)], validation=0.25, seed=0)
    assert len(validation) == 5
    assert len(train) == 15
    assert not set(train) & set(validation)
    assert sorted(train + validation) == sorted(f"id{i}" for i in range(20))


def test_split_identities_is_reproducible():
    first = split_identities(IDENTITIES * 4, validation=0.5, seed=7)
    assert first == split_identities(IDENTITIES * 4, validation=0.5, seed=7)
