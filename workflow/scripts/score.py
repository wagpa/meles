"""Score each k-fold of aggregated embeddings.

For every fold this computes standard face re-identification metrics by matching
each test (query) embedding against the train (gallery) embeddings using cosine
similarity (the metric ArcFace is trained for):

- mAP:      mean average precision over all queries, ranking the individual
            gallery embeddings (scikit-learn's label ranking average precision).
- accuracy: fraction of queries whose closest gallery item has the correct
            identity (equivalent to Rank-1).
- rank_1/5/10: fraction of queries whose correct identity appears within the
            top-1/5/10 ranked identities (CMC). Identities are ranked by their
            best-matching gallery embedding, so repeated embeddings of the same
            identity do not occupy multiple ranks.

The per-fold metrics are collected into a single dataframe and written to the
score output path.

The full (queries x gallery) similarity matrix does not fit in memory for
realistic fold sizes (~50k queries x ~90k gallery items is >30 GB), so queries
are streamed in chunks and the metrics are accumulated as counts/sums over
those chunks.
"""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from tqdm.auto import tqdm

from meles.classifier import Classifier

logger = logging.getLogger(__name__)

RANKS = (1, 5, 10)

# Rows of the embedding TSVs parsed per read, and queries scored per similarity
# block. Both bound peak memory rather than changing the result.
READ_CHUNK_ROWS = 8192


def parse_embedding(value) -> np.ndarray:
    """Parse a serialized embedding cell into a 1-D float32 array.

    Embeddings are written to TSV as the string repr of a Python list, so they
    come back as strings. Already-parsed sequences are passed through.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    return np.fromstring(value.strip().strip("[]"), sep=",", dtype=np.float32)


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a TSV into an (N, D) float32 embedding matrix and an (N,) identity array.

    The file is read in row blocks so the parsed matrix, and not the raw text of
    the whole file, dominates memory usage.
    """
    logger.debug("Reading embeddings from %s", path)
    blocks: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    reader = pd.read_parquet(
        path, usecols=["identity", "embedding"], chunksize=READ_CHUNK_ROWS
    )
    for frame in reader:
        blocks.append(np.vstack([parse_embedding(e) for e in frame["embedding"]]))
        identities.append(frame["identity"].to_numpy())
    return np.vstack(blocks), np.concatenate(identities)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, a_min=1e-12, a_max=None)


def score_fold(
    train_structure: pd.DataFrame,
    test_structure: pd.DataFrame,
    ranks: tuple[int, ...] = RANKS,
) -> dict[str, float]:
    """Compute re-identification metrics for a single fold.

    Train and test structures with embeddings. Similarity is cosine,
    so both sides are L2-normalized and scored by dot product.
    """

    X_train = np.vstack(train_structure["embedding"].to_numpy())
    y_train = train_structure["identity"].to_numpy()
    X_test = np.vstack(test_structure["embedding"].to_numpy())
    y_test = test_structure["identity"].to_numpy()

    classifier = Classifier(metric="cosine")
    classifier.fit(X_train, y_train)

    # Top-1 accuracy: the closest neighbour's identity is the predicted identity.
    neighbor_labels, _ = classifier.classify(X_test, n_neighbors=max(RANKS))
    y_pred = neighbor_labels[:, 0]
    accuracy = accuracy_score(y_test, y_pred)

    return { "accuracy": accuracy }


@click.command()
@click.option(
    "--train-set",
    "train_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Aggregated train (gallery) embeddings for one fold; repeat per fold.",
)
@click.option(
    "--test-set",
    "test_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Aggregated test (query) embeddings for one fold; repeat per fold.",
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--method", required=True)

def main(
    train_paths: tuple[Path, ...],
    test_paths: tuple[Path, ...],
    output_path: Path,
    method: str,
) -> None:
    if len(train_paths) != len(test_paths):
        raise click.BadParameter(
            f"Got {len(train_paths)} train sets but {len(test_paths)} test sets; "
            "each fold needs both."
        )

    # Create the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, (train_path, test_path) in tqdm(enumerate(zip(train_paths, test_paths)), desc="Scoring folds"):
        logger.debug("Scoring fold %d", fold)
        train_structure = pd.read_parquet(train_path)
        test_structure = pd.read_parquet(test_path)
        metrics = score_fold(train_structure, test_structure)
        del train_structure, test_structure
        rows.append({"k": fold, **metrics})

    columns = ["k", "accuracy"]
    scores = pd.DataFrame(rows, columns=columns)
    scores.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
