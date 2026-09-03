import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING, Dict

from meles import scoring
from meles.runtime import logs

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

def main(gallery_path: Path, samples_path: Path, output_path: Path, method: Dict):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the two legs of the experiment. They are the same videos aggregated under two different
    # conditions: the gallery is what the attacker holds, the samples are what they get to observe.
    gallery_df = pd.read_parquet(gallery_path)
    samples_df = pd.read_parquet(samples_path)
    if gallery_df.empty or samples_df.empty:
        raise ValueError(f"Empty leg: gallery has {len(gallery_df)} rows, samples have {len(samples_df)}")

    # The legs may differ in anonymization, recognizer and aggregator, but a cosine gallery only
    # means anything inside one embedding space. Mismatched recognizers would otherwise surface as
    # a shape error from deep inside sklearn.
    gallery_dim = len(gallery_df["embedding"].iloc[0])
    samples_dim = len(samples_df["embedding"].iloc[0])
    if gallery_dim != samples_dim:
        raise ValueError(
            f"The gallery leg embeds to {gallery_dim} dimensions and the samples leg to {samples_dim}; "
            "the two legs have to share a frame recognizer to be comparable"
        )

    # Score each fold separately
    metrics = []
    for fold in sorted(samples_df["fold"].unique()):
        # Every fold is queried once, against a gallery built from the other k-1 folds. Both legs
        # were split the same way, so holding the fold out is also what stops a query video from
        # matching its own, merely differently anonymized, copy in the gallery.
        gallery_k_df = gallery_df[gallery_df["fold"] != fold]
        samples_k_df = samples_df[samples_df["fold"] == fold]
        if gallery_k_df.empty or samples_k_df.empty:
            raise ValueError(
                f"Fold {fold} has an empty gallery ({len(gallery_k_df)} rows) or query set "
                f"({len(samples_k_df)} rows); check that both legs cover the same folds"
            )

        # Collect the embeddings. They match the 'y_gallery' labels by index.
        # Then, fit the classifier.
        x_gallery = scoring.stack(gallery_k_df)
        y_gallery = gallery_k_df["identity"].to_numpy()
        classifier = scoring.fit_gallery(x_gallery, y_gallery)

        # Now generate the scoring matrix for each query row.
        x_samples = scoring.stack(samples_k_df)
        y_samples = samples_k_df["identity"].to_numpy()
        n_neighbors = min(scoring.K, len(gallery_k_df))
        classes = scoring.label_space(y_gallery, y_samples)
        score_matrix = scoring.vote_matrix(
            classifier,
            embeddings=x_samples,
            classes=classes,
            n_neighbors=n_neighbors
        )

        # Lastly, generate the accuracy scores
        keys = scoring.video_keys(samples_k_df)
        video_scores, y_true = scoring.pool_per_video(score_matrix, keys)
        metrics_k = scoring.top_k_accuracies(y_true, video_scores, classes)
        metrics_k["frame-top-1"] = scoring.row_accuracy(score_matrix, y_samples, classes)
        metrics_k["fold"] = int(fold)
        metrics.append(metrics_k)

    # Store the results, tagged with the wildcard combination that produced them. Without the tag
    # the rows are indistinguishable once 'collect' concatenates every point of the experiment.
    logger.info("Scored experiment", extra={ **method, "folds": len(metrics) })
    pd.DataFrame(metrics).assign(**method).to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        gallery_path=Path(snakemake.input.gallery_dataset),
        samples_path=Path(snakemake.input.samples_dataset),
        output_path=Path(snakemake.output.dataset),
        method={
            "gallery_frame_anonymization": snakemake.wildcards.get("gallery_frame_anonymization"),
            "gallery_frame_embedding": snakemake.wildcards.get("gallery_frame_embedding"),
            "gallery_frame_aggregation": snakemake.wildcards.get("gallery_frame_aggregation"),
            "samples_frame_anonymization": snakemake.wildcards.get("samples_frame_anonymization"),
            "samples_frame_embedding": snakemake.wildcards.get("samples_frame_embedding"),
            "samples_frame_aggregation": snakemake.wildcards.get("samples_frame_aggregation"),
        }
    )
