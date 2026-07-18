from pathlib import Path

import pandas as pd

def read_ytfaces(path: Path) -> pd.DataFrame:
    """
    Reads the YouTube Faces dataset from the given path and returns a DataFrame containing the structure of the dataset.
    The DataFrame has an entry for every frame in the dataset in the format:
    - identity: The identity of the person in the frame.
    - video: The name of the video containing the frame.
    - frame: The name of the frame file.
    - frame_suffix: The suffix of the frame file.
    - path: The relative path of the frame file from the dataset path.

    :param path: The path to the dataset
    :return: The DataFrame containing the structure of the dataset
    """

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
                _structure.append({
                    'identity': identity_dir.name,
                    'video': video_dir.name,
                    'frame': frame_file.name,
                    'frame_suffix': frame_file.suffix,
                    'path': str(frame_file.relative_to(path))
                })
    return pd.DataFrame(_structure)


def truncate_dataset(
    dataset: pd.DataFrame,
    min_videos_per_identity: int = 3,
    max_frames_per_video: int = 1000
) -> pd.DataFrame:
    # Keep only identities that have at least MIN_VIDEOS_PER_IDENTITY videos.
    videos_per_identity = dataset.groupby("identity")["video"].nunique()
    eligible_identities = videos_per_identity[videos_per_identity >= min_videos_per_identity].index
    dataset = dataset[dataset["identity"].isin(eligible_identities)].copy()

    # Keep only the first MAX_FRAMES_PER_VIDEO frames of each video. To do that, first
    # determine the natural ordering of frames within each video via the frame index
    # encoded in the file name (e.g. 'aligned_detect_5.1700.jpg' -> 1700) and then drop.
    dataset = dataset.sort_values(["identity", "video", "frame"])
    dataset = (
        dataset.groupby(["identity", "video"], sort=False)
        .head(max_frames_per_video)
        .reset_index(drop=True)
    )
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
    video_rank = dataset.groupby("identity")["video"].transform(
        lambda videos: videos.rank(method="dense")
    )
    is_train = video_rank <= train_videos

    train = dataset[is_train].reset_index(drop=True)
    test = dataset[~is_train].reset_index(drop=True)

    # Every identity keeps at least its first video in the train set, so all test identities
    # are guaranteed to be present in the train set as well.
    assert set(test["identity"]).issubset(set(train["identity"]))
    return train, test