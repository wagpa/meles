"""
Compare the fusion levels at which the frames of a query video can be combined.

The aggregation sweep in the Snakefile varies the aggregator on both the gallery and
the query side at once, which makes "no aggregation" hard to read: the scoring step
already fuses per-frame votes per video, so that baseline is not un-fused at all, and
its gallery differs from the aggregated ones too.

This analysis holds the gallery fixed and varies only how a query video's frames are
combined, so the three levels become directly comparable:

- single frame: one frame decides the video (the floor of one observation)
- score level: every frame is matched on its own, then the votes are fused per video
- feature level: the frames are fused into one embedding before it is matched

The gallery is additionally reported once as all enrolled frames and once mean-pooled
per identity, which separates the effect of aggregating the gallery from the effect of
aggregating the query.
"""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from meles import scoring
from meles.aggregation import pooling

logger = logging.getLogger(__name__)

# The feature-level aggregators to compare against the two frame-wise levels.
AGGREGATORS = {
    "feature-level mean": pooling.MeanPoolingAggregator(),
    "feature-level median": pooling.MedianPoolingAggregator(),
    "feature-level max": pooling.MaxPoolingAggregator(),
    "feature-level min": pooling.MinPoolingAggregator(),
}

# The galleries to enrol. Varying this separates gallery from query aggregation.
GALLERIES = ("all frames", "mean per identity")


def build_gallery(train_structure: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Build the enrolled gallery from the train side of a fold.

    :param train_structure: The train rows of a fold, with embeddings
    :param kind: Either "all frames" or "mean per identity"
    :return: A structure with the columns "identity" and "embedding"
    """
    if kind == "all frames":
        return train_structure[["identity", "embedding"]].reset_index(drop=True)
    if kind == "mean per identity":
        return (
            train_structure.groupby("identity")["embedding"]
            .apply(lambda e: np.mean(pooling.normalized(e.to_list()), axis=0))
            .reset_index()
        )
    raise ValueError(f"Unknown gallery: {kind}")


def fuse_features(
    test_structure: pd.DataFrame, aggregator: pooling.Aggregator
) -> pd.DataFrame:
    """
    Fuse the frames of every test video into a single embedding.

    :param test_structure: The test rows of a fold, with embeddings
    :param aggregator: The aggregator to fuse a video's frames with
    :return: A structure with one row per (identity, video)
    """
    return (
        test_structure.groupby(["identity", "video"])["embedding"]
        .apply(lambda e: np.asarray(aggregator.aggregate(e.to_list())).reshape(-1))
        .reset_index()
    )


def analyze_fold(
    train_structure: pd.DataFrame, test_structure: pd.DataFrame, repeats: int
) -> list[dict]:
    """
    Score every fusion level of one fold against every gallery.

    :param train_structure: The train rows of the fold, with embeddings
    :param test_structure: The test rows of the fold, with embeddings
    :param repeats: The number of one-frame-per-video draws for the single-frame level
    :return: One row per (gallery, fusion level) with its metrics
    """
    y_test = test_structure["identity"].to_numpy()
    test_keys = scoring.video_keys(test_structure)
    n_videos = len(test_keys.unique())

    rows = []
    for gallery_kind in GALLERIES:
        gallery = build_gallery(train_structure, gallery_kind)
        y_gallery = gallery["identity"].to_numpy()
        classes = scoring.label_space(y_gallery, y_test)
        classifier = scoring.fit_gallery(scoring.stack(gallery), y_gallery)
        n_neighbors = min(scoring.K, len(gallery))

        # One pass over all test frames serves both frame-wise levels.
        frame_scores = scoring.vote_matrix(
            classifier, scoring.stack(test_structure), classes, n_neighbors=n_neighbors
        )

        mean, std = scoring.single_frame_accuracy(
            frame_scores, test_keys, classes, repeats=repeats
        )
        # "matched" counts the vectors actually compared against the gallery, which is
        # what the matching costs; "decisions" counts the videos identified from them.
        # The two frame-wise levels match every frame, feature level matches one
        # fused vector per video.
        rows.append(
            {
                "gallery": gallery_kind,
                "fusion": "single frame",
                "gallery_size": len(gallery),
                "matched": len(test_structure),
                "decisions": n_videos,
                "top-1": mean,
                "top-1_std": std,
            }
        )

        video_scores, y_true = scoring.pool_per_video(frame_scores, test_keys)
        rows.append(
            {
                "gallery": gallery_kind,
                "fusion": "score-level voting",
                "gallery_size": len(gallery),
                "matched": len(test_structure),
                "decisions": len(y_true),
                **scoring.top_k_accuracies(y_true, video_scores, classes),
            }
        )

        for name, aggregator in AGGREGATORS.items():
            fused = fuse_features(test_structure, aggregator)
            fused_scores = scoring.vote_matrix(
                classifier, scoring.stack(fused), classes, n_neighbors=n_neighbors
            )
            rows.append(
                {
                    "gallery": gallery_kind,
                    "fusion": name,
                    "gallery_size": len(gallery),
                    "matched": len(fused),
                    "decisions": len(fused),
                    **scoring.top_k_accuracies(
                        fused["identity"].to_numpy(), fused_scores, classes
                    ),
                }
            )
    return rows


@click.command()
@click.option(
    "--fold",
    "fold_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="A k-fold split file with embeddings and a 'train' column; repeat per fold.",
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--repeats",
    default=20,
    show_default=True,
    help="Draws of one frame per video for the single-frame level.",
)
def main(fold_paths: tuple[Path, ...], output_path: Path, repeats: int) -> None:
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, fold_path in tqdm(
        enumerate(fold_paths), total=len(fold_paths), desc="Folds"
    ):
        structure = pd.read_parquet(fold_path).dropna()
        train_structure = structure[structure["train"]]
        test_structure = structure[~structure["train"]]
        for row in analyze_fold(train_structure, test_structure, repeats):
            rows.append({"k": fold, **row})
        del structure, train_structure, test_structure

    results = pd.DataFrame(rows)
    results.to_parquet(output_path, index=False)

    # A fold-averaged summary, ordered so the fusion levels read as a progression.
    order = ["single frame", "score-level voting", *AGGREGATORS]
    summary = (
        results.assign(fusion=pd.Categorical(results["fusion"], order, ordered=True))
        .groupby(["gallery", "fusion"], observed=True)[["top-1", "top-5", "top-10"]]
        .mean()
    )
    click.echo(summary.round(4).to_string())


if __name__ == "__main__":
    main()
