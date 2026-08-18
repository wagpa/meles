"""Small helpers shared by the test modules."""

from pathlib import Path

import numpy as np
import pandas as pd

# The dataset structure the workflow runs on by default.
DATASET = Path("data/ytfaces.parquet")


def paths(frames: pd.DataFrame) -> list[str]:
    """The absolute paths of a sampled structure's frames."""
    return [str(DATASET.parent / path) for path in frames["path"]]


def separation(vectors: np.ndarray, labels: list[str]) -> tuple[float, float]:
    """
    How well a set of embeddings tells identities apart.

    :param vectors: The embeddings with shape (num_embeddings, embedding_dim)
    :param labels: The identity of every embedding
    :return: A tuple of the mean cosine similarity within an identity and across identities
    """
    unit = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)
    similarity = unit @ unit.T
    same = np.asarray(labels)[:, None] == np.asarray(labels)[None, :]
    off_diagonal = ~np.eye(len(labels), dtype=bool)
    return (
        float(similarity[same & off_diagonal].mean()),
        float(similarity[~same].mean()),
    )
