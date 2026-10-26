import logging
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray

if TYPE_CHECKING:
    from meles.detection import Detector

logger = logging.getLogger(__name__)


def read_dataset(path: Path, root: Path | None = None) -> pd.DataFrame:
    """
    Reads the YouTube Faces dataset from the given path and returns a DataFrame containing the structure of the dataset.
    The DataFrame has an entry for every frame in the dataset in the format:
    - identity: The identity of the person in the frame.
    - video: The name of the video containing the frame.
    - frame: The name of the frame file.
    - frame_suffix: The suffix of the frame file.
    - path: The relative path of the frame file from ``root``.

    ``path`` is where the ``identity/video/frame`` hierarchy starts, while ``root`` is what ``path`` is
    reported relative to. The two differ because the dataset lives in a subdirectory of the data root
    (``data/ytfaces``) but every consumer resolves ``path`` against the data root itself, so the column
    has to keep the ``ytfaces/`` segment.

    :param path: The path to the dataset, directly containing the identity directories
    :param root: The path that ``path`` entries are relative to; defaults to ``path``
    :return: The DataFrame containing the structure of the dataset
    """

    root = path if root is None else root

    _structure = []
    for identity_dir in path.iterdir():
        if not identity_dir.is_dir():
            continue

        # Iterate all videos for the identity. Each video itself contains a list of image files.
        for video_dir in identity_dir.iterdir():
            if not video_dir.is_dir():
                continue

            # Iterate all frames of the video. Each frame should be a image file.
            for frame_file in video_dir.iterdir():
                if not frame_file.is_file():
                    continue

                # The frame is added to the structure (flattened) in any order.
                # We can later sort them by their natural ordering. The path is relative to the data
                # directory such that it can be easily used in other files.
                _structure.append(
                    {
                        "identity": identity_dir.name,
                        "video": video_dir.name,
                        "frame": frame_file.name,
                        "frame_suffix": frame_file.suffix,
                        "path": str(frame_file.relative_to(root)),
                    }
                )
    return pd.DataFrame(_structure)


def truncate_dataset(
    dataset: pd.DataFrame,
    min_videos_per_identity: int = 3,
    max_frames_per_video: int = 1000,
) -> pd.DataFrame:
    # Keep only identities that have at least MIN_VIDEOS_PER_IDENTITY videos.
    videos_per_identity = dataset.groupby("identity")["video"].nunique()
    eligible_identities = videos_per_identity[videos_per_identity >= min_videos_per_identity].index
    dataset = dataset[dataset["identity"].isin(eligible_identities)].copy()

    # Keep only the first MAX_FRAMES_PER_VIDEO frames of each video. To do that, first
    # determine the natural ordering of frames within each video via the frame index
    # encoded in the file name (e.g. 'aligned_detect_5.1700.jpg' -> 1700) and then drop.
    dataset = dataset.sort_values(["identity", "video", "frame"])
    dataset = dataset.groupby(["identity", "video"], sort=False).head(max_frames_per_video).reset_index(drop=True)
    return dataset


def split_dataset(
    dataset: pd.DataFrame,
    train_videos: int = 2,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the dataset into train and test sets. It ensures that for every identity, there are at least ``min_train``
    videos in the train set. All other videos are placed into the test set. As such, some identities might only be present
    in the train set and not the test set. This mirrors expected real-world scenarios.

    :param dataset: The full dataset with the structure of ``read_ytfaces``.
    :param train_videos: The number of videos per identity to use for training.
    :return: The train and test sets.
    """
    # Rank the distinct videos of each identity by their natural (ascending) order, so
    # that the three lowest-numbered videos of every identity receive ranks 1, 2 and 3.
    video_rank = dataset.groupby("identity")["video"].transform(lambda videos: videos.rank(method="dense"))
    is_train = video_rank <= train_videos

    train = dataset[is_train].reset_index(drop=True)
    test = dataset[~is_train].reset_index(drop=True)

    # Every identity keeps at least its first video in the train set, so all test identities
    # are guaranteed to be present in the train set as well.
    assert set(test["identity"]).issubset(set(train["identity"]))
    return train, test


def k_fold_splits(
    df: pd.DataFrame,
    k: int,
    identity_col: str = "identity",
    video_col: str = "video",
    shuffle: bool = False,
    random_state: int | None = 0,
) -> pd.DataFrame:
    """Sets fold for each identity-video."""

    # 1. one row per (identity, video)
    videos = df[[identity_col, video_col]].drop_duplicates()

    # 2. guard the edge case: an identity with < k videos can't reach every fold
    counts = videos.groupby(identity_col).size()
    if (counts < k).any():
        bad = counts[counts < k].to_dict()
        raise ValueError(f"identities with fewer than k={k} videos: {bad}")

    # 3. assign each video a fold index within its identity (round-robin)
    if shuffle:
        videos = videos.sample(frac=1, random_state=random_state)
    videos["fold"] = videos.groupby(identity_col).cumcount() % k

    # 4. propagate the fold index to every frame
    return df.merge(videos, on=[identity_col, video_col], how="left")


def extract_video_frames(
    source: Path,
    destination: Path,
    frames_per_video: int = 25,
    videos_per_identity: int | None = None,
    names: Mapping[str, str] | None = None,
    detector: "Detector | None" = None,
    require_face: bool = True,
    progress: bool = True,
) -> pd.DataFrame:
    """
    Turn a tree of ``identity/video/clip.mp4`` into the ``identity/video/frame.jpg`` layout
    :func:`read_dataset` walks.

    Written for VoxCeleb2, whose ``dev/mp4/id00012/21Uxsk56VDQ/00001.mp4`` is exactly that shape.
    The middle level is the **YouTube video**, and that is the level that matters: clips sharing
    one video id are segments of a single recording session, while different video ids are
    genuinely separate sightings of the person. On YouTube Faces an identity offers 1.8 such
    sessions once its target's is held out, and validation rank-1 measured 26.7% for the
    identities with one against 63.6% for those with more -- so sessions, not frames, are what
    the aggregation has to work with.

    Frames are therefore spread **evenly across each video's whole duration** rather than taken
    consecutively. Neighbouring frames of one clip sit at 0.78 mean cosine against 0.42 across
    clips, so a consecutive run would be one observation copied ``frames_per_video`` times.

    :param source: The root holding ``identity/video/*.mp4``
    :param destination: Where to write ``identity/video/*.jpg``; created if needed
    :param frames_per_video: How many frames to keep per video, spread across its clips
    :param videos_per_identity: Cap on videos kept per identity, or None for all. A cap trades
        sessions per person against people, which is the wrong way round here
    :param names: Maps a source identity directory to the name to store, e.g. ``id00012`` to
        ``Aaron_Eckhart`` via ``vox2_meta.csv``. Without it the directory name is used
    :param detector: Used when ``require_face``; defaults to the pipeline's shared YuNet
    :param require_face: Whether to keep only frames a face is found in. A frame without one is
        not an observation of anybody, and as training data it teaches nothing
    :param progress: Whether to show a progress bar
    :return: The structure, with ``path`` relative to ``destination``'s parent so it reads like
        the YouTube Faces one
    """
    from tqdm.auto import tqdm

    if detector is None and require_face:
        from meles.detection import default_detector

        detector = default_detector

    source, destination = Path(source), Path(destination)
    identities = sorted(d for d in source.iterdir() if d.is_dir())
    if not identities:
        raise FileNotFoundError(f"No identity directories under {source}")

    rows: list[dict[str, str]] = []
    missed = 0
    stepped = tqdm(identities, desc="extracting", mininterval=5.0) if progress else identities

    for identity_dir in stepped:
        identity = names.get(identity_dir.name, identity_dir.name) if names else identity_dir.name
        videos = sorted(d for d in identity_dir.iterdir() if d.is_dir())[:videos_per_identity]

        for video_dir in videos:
            clips = sorted(video_dir.glob("*.mp4"))
            if not clips:
                continue

            # Spread the budget over the video's clips, so one long clip cannot supply all of it.
            per_clip = max(1, round(frames_per_video / len(clips)))
            written = 0
            for clip in clips:
                if written >= frames_per_video:
                    break
                for image in _sample_clip(clip, min(per_clip, frames_per_video - written)):
                    if require_face and detector is not None and detector.detect_best(image) is None:
                        missed += 1
                        continue
                    name = f"{clip.stem}.{written:04d}.jpg"
                    out = destination / identity / video_dir.name / name
                    out.parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(out), image)
                    rows.append(
                        {
                            "identity": identity,
                            "video": video_dir.name,
                            "frame": name,
                            "frame_suffix": ".jpg",
                            "path": str(out.relative_to(destination.parent)),
                        }
                    )
                    written += 1

    logger.info(
        "Extracted video frames",
        extra={"identities": len(identities), "frames": len(rows), "no_face": missed},
    )
    return pd.DataFrame(rows)


def _sample_clip(clip: Path, count: int) -> list[NDArray[np.uint8]]:
    """
    ``count`` frames spread evenly across a clip's duration.

    Evenly rather than consecutively on purpose: consecutive frames of one clip are near
    duplicates, so a run of them is a single observation wearing a crowd's costume.

    :param clip: The video file
    :param count: How many frames to return, fewer if the clip is shorter
    :return: The frames, BGR uint8
    """
    capture = cv2.VideoCapture(str(clip))
    try:
        if not capture.isOpened():
            logger.warning("Could not open clip", extra={"clip": str(clip)})
            return []

        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0:
            return []

        wanted = np.linspace(0, total - 1, num=min(count, total), dtype=int)
        frames = []
        for position in wanted:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(position))
            ok, image = capture.read()
            if ok:
                frames.append(image.astype(np.uint8, copy=False))
        return frames
    finally:
        capture.release()
