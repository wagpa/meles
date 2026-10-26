import logging
import multiprocessing
import os
from collections.abc import Mapping
from concurrent.futures import ProcessPoolExecutor, as_completed
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
    workers: int | None = 1,
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
    :param detector: Used when ``require_face``; defaults to the pipeline's shared YuNet. Cannot
        be combined with ``workers`` above one, because it would have to cross a process boundary
    :param require_face: Whether to keep only frames a face is found in. A frame without one is
        not an observation of anybody, and as training data it teaches nothing
    :param progress: Whether to show a progress bar
    :param workers: Processes to spread the identities over. One keeps everything in this process.
        Decoding and detection are both CPU bound and an identity never touches another's files,
        so this scales nearly linearly -- measured 6.1x on twelve cores; ``None`` uses every core
        but one. Above one the pool spawns rather than forks, so a *script* calling it needs its
        entry point under ``if __name__ == "__main__":`` or each worker re-runs the script.
        Notebooks need nothing
    :return: The structure, with ``path`` relative to ``destination``'s parent so it reads like
        the YouTube Faces one
    """
    from tqdm.auto import tqdm

    source, destination = Path(source), Path(destination)
    identities = sorted(d for d in source.iterdir() if d.is_dir())
    if not identities:
        raise FileNotFoundError(f"No identity directories under {source}")

    if workers is None:
        workers = max(1, (os.cpu_count() or 2) - 1)
    if workers > 1 and detector is not None:
        raise ValueError(
            "A detector cannot be handed to worker processes -- cv2.FaceDetectorYN does not "
            "pickle. Leave detector unset to let each worker build its own, or pass workers=1."
        )
    if detector is None and require_face and workers == 1:
        from meles.detection import default_detector

        detector = default_detector

    work = [(d, names.get(d.name, d.name) if names else d.name) for d in identities]
    rows: list[dict[str, str]] = []
    missed = empty = 0

    if workers == 1:
        stepped = tqdm(work, desc="extracting", mininterval=5.0) if progress else work
        for identity_dir, label in stepped:
            produced, no_face, unread = _extract_identity(
                identity_dir, destination, label, frames_per_video, videos_per_identity, require_face, detector
            )
            rows.extend(produced)
            missed += no_face
            empty += unread
    else:
        # Spawn rather than fork: the parent may already hold an ONNX Runtime session or a CUDA
        # context, neither of which survives being forked.
        context = multiprocessing.get_context("spawn")
        with ProcessPoolExecutor(max_workers=workers, mp_context=context, initializer=_configure_worker) as pool:
            futures = [
                pool.submit(
                    _extract_identity,
                    identity_dir,
                    destination,
                    label,
                    frames_per_video,
                    videos_per_identity,
                    require_face,
                )
                for identity_dir, label in work
            ]
            stepped = (
                tqdm(as_completed(futures), total=len(futures), desc="extracting", mininterval=5.0)
                if progress
                else as_completed(futures)
            )
            for future in stepped:
                produced, no_face, unread = future.result()
                rows.extend(produced)
                missed += no_face
                empty += unread

    # Workers finish out of order, so sort rather than leave the structure's order up to timing.
    rows.sort(key=lambda row: (row["identity"], row["video"], row["frame"]))
    logger.info(
        "Extracted video frames",
        extra={
            "identities": len(identities),
            "frames": len(rows),
            "no_face": missed,
            "unreadable_clips": empty,
            "workers": workers,
        },
    )
    # Named columns even when nothing was produced: an archive whose frames are all filtered out
    # should still answer structure["identity"] with an empty result rather than a KeyError.
    return pd.DataFrame(rows, columns=["identity", "video", "frame", "frame_suffix", "path"])


def _configure_worker() -> None:
    """
    Hold each worker to one OpenCV thread.

    Without this every worker sizes its pool to the whole machine, so ``workers`` processes each
    spawn ``cpu_count`` threads and they contend -- the same trap :mod:`meles.runtime` exists for
    on the Snakemake side.
    """
    cv2.setNumThreads(1)


def _extract_identity(
    identity_dir: Path,
    destination: Path,
    identity: str,
    frames_per_video: int,
    videos_per_identity: int | None,
    require_face: bool,
    detector: "Detector | None" = None,
) -> tuple[list[dict[str, str]], int, int]:
    """
    Extract one identity's frames. Module level and picklable so it can run in a worker process.

    :param identity_dir: The source directory holding this identity's video directories
    :param destination: Where to write ``identity/video/*.jpg``
    :param identity: The name to store, already resolved through the rename mapping
    :param frames_per_video: How many frames to keep per video, spread across its clips
    :param videos_per_identity: Cap on videos kept, or None for all
    :param require_face: Whether to drop frames with no detected face
    :param detector: The detector to use; built per process when omitted
    :return: The rows, the count of frames dropped for having no face, and the count of clips
        that could not be opened or held no frames
    """
    if detector is None and require_face:
        detector = _worker_detector()

    rows: list[dict[str, str]] = []
    missed = empty = 0
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
            images = _sample_clip(clip, min(per_clip, frames_per_video - written))
            if not images:
                empty += 1
            for image in images:
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

    return rows, missed, empty


_DETECTOR: "Detector | None" = None


def _worker_detector() -> "Detector":
    """
    This process's detector, built once and kept.

    A worker cannot be handed the parent's -- ``cv2.FaceDetectorYN`` does not pickle -- and
    rebuilding it per identity would reload the weights thousands of times.

    :return: The detector for this process
    """
    global _DETECTOR
    if _DETECTOR is None:
        from meles.detection import default_detector

        _DETECTOR = default_detector
    return _DETECTOR


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
