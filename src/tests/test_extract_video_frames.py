"""
``extract_video_frames`` turns a tree of clips into the ``identity/video/frame`` layout the rest
of the pipeline walks. The ways it can go wrong are quiet: frames taken consecutively instead of
spread across a clip (which makes one observation look like many), the video level collapsing, or
paths that do not resolve against the data root.

Synthetic clips keep it off any real corpus, and a stub detector keeps it off the model weights.
"""

import logging
import os
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from meles.common import FacialArea
from meles.dataset import extract_video_frames, read_dataset
from meles.detection import Detector

IDENTITIES = ["id00001", "id00002"]
VIDEOS = ["AbCdEf12345", "ZyXwVu98765"]
CLIPS = 2
CLIP_FRAMES = 40


class AlwaysFinds(Detector):
    def detect(self, image):
        height, width = image.shape[:2]
        return [FacialArea(x=0, y=0, w=width, h=height, confidence=0.9)]


class NeverFinds(Detector):
    def detect(self, image):
        return []


@pytest.fixture
def clips(tmp_path: Path) -> Path:
    """A VoxCeleb2-shaped tree: identity / youtube video / utterance.mp4."""
    source = tmp_path / "vox2" / "dev" / "mp4"
    rng = np.random.default_rng(0)
    for identity in IDENTITIES:
        for video in VIDEOS:
            directory = source / identity / video
            directory.mkdir(parents=True)
            for clip in range(CLIPS):
                writer = cv2.VideoWriter(
                    str(directory / f"{clip:05d}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 25.0, (64, 64)
                )
                for index in range(CLIP_FRAMES):
                    # A ramp, so a frame's content identifies where in the clip it came from.
                    frame = np.full((64, 64, 3), index * 6, dtype=np.uint8)
                    frame[:8, :8] = rng.integers(0, 255, (8, 8, 3), dtype=np.uint8)
                    writer.write(frame)
                writer.release()
    return source


def test_it_produces_the_layout_read_dataset_walks(clips, tmp_path):
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(clips, destination, frames_per_video=8, progress=False, require_face=False)

    assert set(structure["identity"]) == set(IDENTITIES)
    assert set(structure["video"]) == set(VIDEOS)
    assert list(structure.columns) == ["identity", "video", "frame", "frame_suffix", "path"]

    # read_dataset over the same tree agrees with what was returned.
    walked = read_dataset(destination, root=destination.parent)
    assert len(walked) == len(structure)
    assert set(walked["path"]) == set(structure["path"])


def test_paths_resolve_against_the_data_root(clips, tmp_path):
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(clips, destination, frames_per_video=4, progress=False, require_face=False)
    assert all((destination.parent / path).is_file() for path in structure["path"])
    assert all(path.startswith("voxceleb/") for path in structure["path"])


def test_frames_are_spread_across_a_clip_not_consecutive(clips, tmp_path):
    """Consecutive frames are one observation copied -- the thing the premise needs to avoid."""
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(clips, destination, frames_per_video=8, progress=False, require_face=False)

    one = structure[(structure["identity"] == IDENTITIES[0]) & (structure["video"] == VIDEOS[0])]
    # The synthetic frames ramp in brightness, so distinct means spread through the clip.
    levels = sorted({int(cv2.imread(str(destination.parent / p))[32, 32, 0]) for p in one["path"]})
    assert len(levels) >= 4, f"frames look consecutive, brightness levels {levels}"
    assert max(levels) - min(levels) > 100, "frames should span most of the clip's duration"


def test_every_video_contributes(clips, tmp_path):
    """Sessions are the scarce axis, so no video may be skipped."""
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(clips, destination, frames_per_video=6, progress=False, require_face=False)
    per_video = structure.groupby(["identity", "video"]).size()
    assert len(per_video) == len(IDENTITIES) * len(VIDEOS)
    assert (per_video > 0).all()


def test_the_frame_budget_is_per_video_and_split_over_its_clips(clips, tmp_path):
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(clips, destination, frames_per_video=6, progress=False, require_face=False)
    per_video = structure.groupby(["identity", "video"]).size()
    assert (per_video <= 6).all(), f"budget exceeded: {per_video.to_dict()}"

    # Both clips of a video contribute, rather than the first supplying everything.
    stems = structure.groupby(["identity", "video"])["frame"].apply(lambda s: {n.split(".")[0] for n in s})
    assert all(len(s) == CLIPS for s in stems), stems.to_dict()


def test_videos_per_identity_can_be_capped(clips, tmp_path):
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(
        clips, destination, frames_per_video=4, videos_per_identity=1, progress=False, require_face=False
    )
    assert (structure.groupby("identity")["video"].nunique() == 1).all()


def test_identities_can_be_renamed(clips, tmp_path):
    """vox2_meta.csv maps id00012 to a name, which is what makes the YTF overlap check possible."""
    destination = tmp_path / "data" / "voxceleb"
    structure = extract_video_frames(
        clips,
        destination,
        frames_per_video=4,
        names={"id00001": "Aaron_Eckhart"},
        progress=False,
        require_face=False,
    )
    assert set(structure["identity"]) == {"Aaron_Eckhart", "id00002"}
    assert all(
        path.startswith("voxceleb/Aaron_Eckhart/")
        for path in structure[structure["identity"] == "Aaron_Eckhart"]["path"]
    )


def test_frames_without_a_face_are_dropped_when_asked(clips, tmp_path):
    """A frame with no face is not an observation of anybody."""
    kept = extract_video_frames(
        clips, tmp_path / "a" / "voxceleb", frames_per_video=4, detector=AlwaysFinds(), progress=False
    )
    dropped = extract_video_frames(
        clips, tmp_path / "b" / "voxceleb", frames_per_video=4, detector=NeverFinds(), progress=False
    )
    assert len(kept) == 4 * len(IDENTITIES) * len(VIDEOS)
    assert dropped.empty


def test_an_empty_source_is_refused(tmp_path):
    empty = tmp_path / "nothing"
    empty.mkdir()
    with pytest.raises(FileNotFoundError, match="No identity directories"):
        extract_video_frames(empty, tmp_path / "out", progress=False)


def test_an_empty_result_still_has_the_columns(clips, tmp_path):
    """
    An archive whose frames are all filtered out must answer ``["identity"]``, not raise.

    Callers summarise what they got -- ``part["identity"].nunique()`` -- and a bare DataFrame()
    has no columns, so the summary line would be the thing that crashed the run.
    """
    structure = extract_video_frames(
        clips, tmp_path / "voxceleb", frames_per_video=4, detector=NeverFinds(), progress=False
    )
    assert structure.empty
    assert list(structure.columns) == ["identity", "video", "frame", "frame_suffix", "path"]
    assert structure["identity"].nunique() == 0


def test_workers_produce_exactly_the_same_structure(clips, tmp_path):
    """
    Parallelism is an optimisation and must not change a single row.

    Workers finish out of order, so this also pins that the structure is sorted rather than left
    in whatever order the pool happened to complete in -- otherwise the parquet would differ run
    to run and every downstream diff would be noise.
    """
    one = extract_video_frames(
        clips, tmp_path / "one" / "voxceleb", frames_per_video=6, progress=False, require_face=False, workers=1
    )
    many = extract_video_frames(
        clips, tmp_path / "many" / "voxceleb", frames_per_video=6, progress=False, require_face=False, workers=3
    )
    # ``path`` is relative to the destination's parent, so both sides read 'voxceleb/...'.
    pd.testing.assert_frame_equal(one, many)


def test_a_detector_cannot_be_handed_to_workers(clips, tmp_path):
    """cv2.FaceDetectorYN does not pickle, so say so rather than failing inside a worker."""
    with pytest.raises(ValueError, match="does not pickle"):
        extract_video_frames(clips, tmp_path / "out", detector=AlwaysFinds(), workers=2, progress=False)


def test_workers_none_leaves_one_core_free(clips, tmp_path, monkeypatch, caplog):
    monkeypatch.setattr(os, "cpu_count", lambda: 4)
    with caplog.at_level(logging.INFO, logger="meles.dataset"):
        extract_video_frames(
            clips, tmp_path / "voxceleb", frames_per_video=4, progress=False, require_face=False, workers=None
        )
    record = next(r for r in caplog.records if r.msg == "Extracted video frames")
    assert record.workers == 3


def test_unreadable_clips_are_counted_not_fatal(clips, tmp_path, caplog):
    """0.03% of VoxCeleb2's clips would not open; that is worth a number, not a crash."""
    (clips / IDENTITIES[0] / VIDEOS[0] / "99999.mp4").write_bytes(b"not a video")
    with caplog.at_level(logging.INFO, logger="meles.dataset"):
        structure = extract_video_frames(
            clips, tmp_path / "voxceleb", frames_per_video=6, progress=False, require_face=False
        )
    record = next(r for r in caplog.records if r.msg == "Extracted video frames")
    assert record.unreadable_clips == 1
    assert not structure.empty
