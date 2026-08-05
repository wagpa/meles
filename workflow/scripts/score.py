"""
Score each k-fold of aggregated embeddings.
"""

import logging
from pathlib import Path

import click
import pandas as pd
from tqdm.auto import tqdm

from meles import scoring

logger = logging.getLogger(__name__)


def score_fold(
    train_structure: pd.DataFrame,
    test_structure: pd.DataFrame,
) -> dict[str, float]:
    """Compute re-identification metrics for a single fold.

    Train and test structures with embeddings. Similarity is cosine, so both
    sides are L2-normalized and scored by dot product. Every test row votes with
    its K nearest gallery neighbours, the votes are pooled per test video, and
    the pooled scores are ranked to give top-k accuracy per video.

    ``frame-top-1`` reports the same rows scored without that pooling, so the
    gain the pooling contributes stays visible. Under an aggregation method that
    already fuses each video into one embedding the two are identical, because
    then there is nothing left to pool.
    """

    y_train = train_structure["identity"].to_numpy()
    y_test = test_structure["identity"].to_numpy()
    classes = scoring.label_space(y_train, y_test)

    classifier = scoring.fit_gallery(scoring.stack(train_structure), y_train)
    n_neighbors = min(scoring.K, len(train_structure))
    row_scores = scoring.vote_matrix(
        classifier, scoring.stack(test_structure), classes, n_neighbors=n_neighbors
    )

    keys = scoring.video_keys(test_structure)
    video_scores, y_true = scoring.pool_per_video(row_scores, keys)

    metrics = scoring.top_k_accuracies(y_true, video_scores, classes)
    metrics["frame-top-1"] = scoring.row_accuracy(row_scores, y_test, classes)
    return metrics


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
