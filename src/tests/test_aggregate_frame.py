"""
Tests for the pixel-level aggregation rule.

Two things here are easy to get wrong and impossible to notice downstream. The first is the group
key: the 'video' column is only a per-identity number ("0", "1", ...), so a grouping that left the
identity out would average unrelated people into one image and report a recognition accuracy for
frames that no longer show anybody. The second is the path the output names. Every ``path`` is
relative to the parquet holding it, and the baseline aggregator writes no images at all -- it points
back at the anonymized frames where they already lie -- so a path that is merely *wrong* still reads
as a plausible string and only fails two rules later, inside the embedder.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import cv2
import numpy as np
import pandas as pd
import pytest

from meles.common import Images
from meles.frame_aggregation.aggregator import FrameAggregator
from meles.frame_aggregation.resample import BicubicFrameAggregator

SCRIPT = Path("workflow/scripts/aggregate_frame.py")

# One grey level per identity, so an image can be traced back to who it came from. A merge of two
# identities lands between them, which no single identity's level is.
LEVELS = {"alice": 40, "bob": 120, "carol": 200}
VIDEOS = ["0", "1", "2"]
FRAMES = 4
SIZE = 8


class MeanFrameAggregator(FrameAggregator):
    """Collapses a group to its pixel-wise mean, i.e. an aggregator that does produce new pixels."""

    def aggregate(self, images: Images) -> Images:
        return [np.mean(np.stack(images), axis=0).astype(np.uint8)]


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_workflow_aggregate_frame", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Frames on disk and a split, for three identities that all reuse the same video keys."""
    frames_path = tmp_path / "anonymize_frame" / "structure.parquet"
    frames_path.parent.mkdir(parents=True)

    rows = []
    for identity, level in LEVELS.items():
        for video in VIDEOS:
            for frame in range(FRAMES):
                name = f"{frame}.png"
                path = Path(identity) / video / name
                (frames_path.parent / path).parent.mkdir(parents=True, exist_ok=True)
                # PNG rather than JPEG so the grey level survives the round trip exactly.
                cv2.imwrite(str(frames_path.parent / path), np.full((SIZE, SIZE, 3), level, np.uint8))
                rows.append(
                    {
                        "identity": identity,
                        "video": video,
                        "frame": name,
                        "frame_suffix": ".png",
                        "path": str(path),
                    }
                )
    frames = pd.DataFrame(rows)
    frames.to_parquet(frames_path)

    split = frames[["identity", "video"]].drop_duplicates().reset_index(drop=True)
    split["fold"] = split.groupby("identity").cumcount() % len(VIDEOS) + 1
    split_path = tmp_path / "split.parquet"
    split.to_parquet(split_path)
    return frames_path, split_path


def read_all(output_path: Path) -> tuple[pd.DataFrame, list[np.ndarray]]:
    """The written structure, and every frame it names, resolved relative to the structure itself."""
    aggregated = pd.read_parquet(output_path)
    images = []
    for path in aggregated["path"]:
        image = cv2.imread(str(output_path.parent / path))
        assert image is not None, f"{path} does not resolve relative to {output_path}"
        images.append(image)
    return aggregated, images


def test_the_group_key_always_contains_the_identity(script: ModuleType) -> None:
    """The identity is part of the group key of every scope, not something a scope may leave out."""
    group_by, _ = script.resolve("video_none")

    assert "identity" in group_by
    assert "fold" in group_by


def test_the_identity_scope_is_refused(script: ModuleType) -> None:
    """Collapsing the videos of an identity would leave no video for the later rules to key on."""
    with pytest.raises(ValueError, match="not supported"):
        script.resolve("identity_none")


def test_the_baseline_reuses_the_frames_where_they_lie(script: ModuleType, dataset: tuple[Path, Path]) -> None:
    """
    'none' produces no new pixels, so it must write no images -- copying the whole anonymized
    dataset into every aggregation directory would double the largest thing on disk for nothing.
    """
    frames_path, split_path = dataset
    output_path = frames_path.parent.parent / "aggregate_frame" / "video_none" / "structure.parquet"

    script.main(frames_path, split_path, output_path, "video_none")
    aggregated, images = read_all(output_path)

    # Every frame survives, still named where the input left it, and nothing was written next to
    # the output: the paths walk back up to the input directory instead.
    assert len(aggregated) == len(LEVELS) * len(VIDEOS) * FRAMES
    assert list(output_path.parent.iterdir()) == [output_path]
    assert all(path.startswith("..") for path in aggregated["path"])
    for identity, rows in aggregated.groupby("identity"):
        levels = {int(images[index][0, 0, 0]) for index in rows.index}
        assert levels == {LEVELS[identity]}, f"{identity} was aggregated with another identity"


def test_new_pixels_are_written_out_per_group(
    script: ModuleType, dataset: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """An aggregator that does collapse its group writes the result, one row per returned frame."""
    frames_path, split_path = dataset
    output_path = frames_path.parent.parent / "aggregate_frame" / "video_mean" / "structure.parquet"
    monkeypatch.setitem(script.AGGREGATORS, "mean", MeanFrameAggregator())

    script.main(frames_path, split_path, output_path, "video_mean")
    aggregated, images = read_all(output_path)

    # One frame per identity-video now, written beside the structure rather than pointing away.
    assert len(aggregated) == len(LEVELS) * len(VIDEOS)
    assert set(aggregated["video"]) == set(VIDEOS)
    assert not any(path.startswith("..") for path in aggregated["path"])
    # Each mean is still its own identity's level; a group spanning two would land between them.
    for identity, rows in aggregated.groupby("identity"):
        levels = {int(images[index][0, 0, 0]) for index in rows.index}
        assert levels == {LEVELS[identity]}, f"{identity} was aggregated with another identity"


def test_a_frame_wise_aggregator_keeps_the_frame_names(
    script: ModuleType, dataset: tuple[Path, Path], monkeypatch: pytest.MonkeyPatch
) -> None:
    """
    A de-anonymizer returns one frame per input and combines nothing, so every frame still answers
    to the name it arrived under and the output stays comparable to the anonymized input.
    """
    frames_path, split_path = dataset
    output_path = frames_path.parent.parent / "aggregate_frame" / "video_frame_bicubic" / "structure.parquet"
    monkeypatch.setitem(script.AGGREGATORS, "frame_bicubic", BicubicFrameAggregator())

    script.main(frames_path, split_path, output_path, "video_frame_bicubic")
    aggregated, images = read_all(output_path)

    # Nothing collapsed, the names came through, and the frames were written here rather than
    # pointed at: bicubic produces new pixels, so the reuse shortcut must not have fired.
    original = pd.read_parquet(frames_path)
    assert len(aggregated) == len(original)
    assert sorted(aggregated["frame"]) == sorted(original["frame"])
    assert not any(path.startswith("..") for path in aggregated["path"])
    assert all(image.shape == (SIZE, SIZE, 3) for image in images)


def test_the_fold_is_not_written_out(script: ModuleType, dataset: tuple[Path, Path]) -> None:
    """
    'aggregate_embedding' merges the split onto its own input, so a 'fold' arriving from here would
    collide with it and take the column away from both.
    """
    frames_path, split_path = dataset
    output_path = frames_path.parent.parent / "aggregate_frame" / "video_none" / "structure.parquet"

    script.main(frames_path, split_path, output_path, "video_none")

    assert "fold" not in pd.read_parquet(output_path).columns
