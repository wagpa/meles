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


def score_fold(
    train_structure: pd.DataFrame,
    test_structure: pd.DataFrame,
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

    K = 20  # neighbors per frame; more = denser, less tie-degenerate scores
    neighbor_labels, _ = classifier.classify(X_test, n_neighbors=K)  # (n_frames, K)
    
    classes = np.union1d(test_structure["identity"].unique(), neighbor_labels.ravel())
    class_to_idx = {c: i for i, c in enumerate(classes)}
    
    # per-frame soft score = fraction of the K neighbors voting each class
    frame_scores = np.zeros((len(neighbor_labels), len(classes)))
    for j in range(K):
        idx = np.fromiter((class_to_idx[l] for l in neighbor_labels[:, j]), dtype=int)
        frame_scores[np.arange(len(idx)), idx] += 1
    frame_scores /= K
    
    sc = pd.DataFrame(frame_scores, columns=classes)
    sc["video"] = test_structure["video"].values
    sc["identity"] = test_structure["identity"].values
    
    video_scores = sc.groupby("video")[list(classes)].mean()           # dense, non-integer
    y_true = sc.groupby("video")["identity"].first().loc[video_scores.index]

    result = {}
    for k in [1, 5, 10]:
        acc = top_k_accuracy_score(y_true.to_numpy(), video_scores.to_numpy(), k=min(k, len(classes)), labels=classes)
        result[f"top-{k}"] = acc
    return result


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

    columns = ["k", "accuracy"]
    scores = pd.DataFrame(rows, columns=columns)
    scores.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
