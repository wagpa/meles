"""
How videos are split into folds and which columns define an aggregation group.

Both are keyed by identity, because a video key is only unique within one identity.
"""

import pandas as pd

# Identity and fold are in every scope on purpose: video keys repeat across identities, and a group must
# never span folds. "video" groups the frames of one video; "identity" groups everything a fold holds of
# that identity.
SCOPES: dict[str, list[str]] = {
    "video": ["identity", "fold", "video"],
    "identity": ["identity", "fold"],
}


def assign_folds(structure: pd.DataFrame, k: int, seed: int = 0) -> pd.DataFrame:
    """
    Assign a fold (1 to k) to each identity-video, spreading each identity's videos evenly over the folds.

    The result has one row per identity-video, never one per frame: it is merged onto per-frame tables
    later, and per-frame rows would duplicate them.

    :param structure: The dataset structure, with at least the columns ``identity`` and ``video``
    :param k: The number of folds
    :param seed: The seed of the shuffle that decides which videos share a fold
    :return: A frame with the columns ``identity``, ``video`` and ``fold``
    """
    videos = structure[["identity", "video"]].drop_duplicates()
    videos = videos.sample(frac=1, random_state=seed).reset_index(drop=True)

    # Round-robin within each identity over the shuffled order.
    videos["fold"] = videos.groupby("identity").cumcount() % k + 1
    return videos


def resolve_scope(key: str) -> tuple[list[str], str]:
    """
    Split an aggregator key of the form ``{scope}_{method}`` into its grouping columns and method name.

    :param key: The configuration key, e.g. ``video_pooling_mean``
    :return: The columns that define a group, and the method name to resolve
    """
    scope, _, method = key.partition("_")
    return SCOPES[scope], method
