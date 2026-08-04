"""
Score each k-fold of aggregated embeddings.
"""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import top_k_accuracy_score
from tqdm.auto import tqdm

from meles.classifier import Classifier

logger = logging.getLogger(__name__)

# The number of gallery neighbours every test frame votes with.
K = 20


def score_fold(
    train_structure: pd.DataFrame,
    test_structure: pd.DataFrame,
) -> dict[str, float]:
    """Compute re-identification metrics for a single fold.

    Train and test structures with embeddings. Similarity is cosine, so both
    sides are L2-normalized and scored by dot product. Every test frame votes
    with its K nearest gallery neighbours, the votes are pooled per test video,
    and the pooled scores are ranked to give top-k accuracy per video.
    """

    X_train = np.vstack(train_structure["embedding"].to_numpy())
    y_train = train_structure["identity"].to_numpy()
    X_test = np.vstack(test_structure["embedding"].to_numpy())
    y_test = test_structure["identity"].to_numpy()

    classifier = Classifier(metric="cosine")
    classifier.fit(X_train, y_train)

    # Neighbours per frame. The gallery may be smaller than K for aggregation
    # methods that pool every identity down to a single embedding.
    n_neighbors = min(K, len(X_train))
    neighbor_labels, distances = classifier.classify(X_test, n_neighbors=n_neighbors)

    # The gallery identities define the label space; test identities are a
    # subset of them but are unioned in so an unseen label cannot silently
    # disappear from the score matrix.
    classes = np.union1d(np.unique(y_train), np.unique(y_test))
    class_to_idx = {label: i for i, label in enumerate(classes)}

    # Per-frame soft score: every neighbour votes for its identity, weighted by
    # its cosine similarity. Plain unweighted votes are degenerate whenever the
    # gallery holds one embedding per identity, because then all K neighbours
    # are distinct identities with exactly one vote each and the ranking falls
    # back to an arbitrary tie-break. Weighting keeps the distances that decide
    # which of those neighbours is actually the closest.
    weights = np.clip(1.0 - distances, 0.0, None)
    frame_scores = np.zeros((len(neighbor_labels), len(classes)))
    frame_rows = np.arange(len(neighbor_labels))
    for j in range(n_neighbors):
        idx = np.fromiter(
            (class_to_idx[label] for label in neighbor_labels[:, j]),
            dtype=int,
            count=len(neighbor_labels),
        )
        frame_scores[frame_rows, idx] += weights[:, j]

    # Normalize each frame to a distribution so that frames contribute equally
    # when they are pooled per video. Frames whose neighbours are all further
    # away than orthogonal carry no weight and stay all-zero.
    totals = frame_scores.sum(axis=1, keepdims=True)
    np.divide(frame_scores, totals, out=frame_scores, where=totals > 0)

    # A video is only identified by the (identity, video) pair: the video column
    # holds the per-identity video number ("0", "1", ...), so grouping on it
    # alone merges the videos of every identity into a handful of groups.
    video_keys = pd.MultiIndex.from_arrays(
        [y_test, test_structure["video"].to_numpy()], names=["identity", "video"]
    )
    video_scores = pd.DataFrame(frame_scores, index=video_keys).groupby(
        level=["identity", "video"], sort=False
    ).mean()
    y_true = video_scores.index.get_level_values("identity").to_numpy()

    return {
        f"top-{k}": top_k_accuracy_score(
            y_true,
            video_scores.to_numpy(),
            k=min(k, len(classes) - 1),
            labels=classes,
        )
        for k in (1, 5, 10)
    }


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
    for fold, (train_path, test_path) in tqdm(
        enumerate(zip(train_paths, test_paths)), desc="Scoring folds"
    ):
        logger.debug("Scoring fold %d", fold)
        train_structure = pd.read_parquet(train_path)
        test_structure = pd.read_parquet(test_path)
        metrics = score_fold(train_structure, test_structure)
        del train_structure, test_structure
        rows.append({"k": fold, **metrics})

    scores = pd.DataFrame(rows)
    scores.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
