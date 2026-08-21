import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING, Dict
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score
import numpy as np

from meles import logs, runtime, scoring

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# The number of points the silhouette is estimated from. Silhouette is O(n^2) in pairwise
# distances, and with 'video_aggregation=none' the input is one row per *frame* (~130k rows on
# the full dataset), where the exact value would cost ~1.7e10 distances. Subsampling is
# sklearn's own remedy for this; the seed keeps it reproducible across runs. ARI stays exact --
# it is O(n) and never subsampled.
SILHOUETTE_SAMPLE_SIZE = 10_000
SILHOUETTE_SEED = 42


def main(dataset_path: Path, output_path: Path, method: Dict):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input path
    dataset_df = pd.read_parquet(dataset_path)

    # Cluster the *pre-identity-aggregation* embeddings: one row per video (or per frame, when the
    # video aggregation is 'none'), so an identity spans several rows and asking whether they group
    # together is a real question. Clustering the identity aggregation instead would be degenerate,
    # since a pooling aggregator leaves exactly one row per identity and k == n forces every point
    # into its own cluster.
    #
    # There is deliberately no fold loop here. Clustering is unsupervised, so it needs no
    # gallery/query split, and restricting it to one fold would leave ~1.2 videos per identity --
    # degenerate for the same reason. The measurement is one global fit over every video.
    embeddings = scoring.stack(dataset_df)
    labels = dataset_df["identity"].to_numpy()
    n_identities = len(np.unique(labels))

    # Cluster on direction only. The rest of the pipeline scores cosine throughout, but the norms
    # arriving here are not constant -- raw frame embeddings run 15-41 and the pooled aggregators
    # 0.55-1.0 -- so Euclidean KMeans on them would partition partly by magnitude. Normalizing
    # first makes it spherical k-means, which is the cosine-consistent choice, and lets the
    # silhouette below use the same metric as 'scoring.fit_gallery'.
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    embeddings = embeddings / np.where(norms > 0, norms, 1.0)

    # Fit the KMeans model
    logger.debug(
        "Clustering embeddings",
        extra={"rows": len(embeddings), "identities": n_identities},
    )
    model = KMeans(n_clusters=n_identities, random_state=42)
    model_labels = model.fit_predict(embeddings)
    n_clusters = len(np.unique(model_labels))

    # Score the model. ARI is always defined; the silhouette is not -- it needs at least two
    # clusters and at least one cluster holding two points. Report NaN rather than raising, so a
    # single degenerate point of the grid does not take the whole run down with it.
    ari = adjusted_rand_score(labels, model_labels)
    if 2 <= n_clusters < len(embeddings):
        sample_size = min(SILHOUETTE_SAMPLE_SIZE, len(embeddings))
        if sample_size < len(embeddings):
            logger.info(
                "Estimating silhouette from a subsample",
                extra={"sample_size": sample_size, "rows": len(embeddings)},
            )
        silhouette_avg = float(
            silhouette_score(
                embeddings,
                model_labels,
                metric="cosine",
                sample_size=sample_size,
                random_state=SILHOUETTE_SEED,
            )
        )
    else:
        logger.warning(
            "Silhouette is undefined for this clustering",
            extra={"clusters": n_clusters, "rows": len(embeddings)},
        )
        silhouette_avg = float("nan")

    logger.info(
        "Scored clustering",
        extra={
            "clusters": n_clusters,
            "identities": n_identities,
            "rows": len(embeddings),
            "adjusted_rand_score": ari,
            "silhouette_avg": silhouette_avg,
        },
    )

    # Store the results, tagged with the wildcard combination that produced them. Without the tag
    # the rows are indistinguishable once 'collect' concatenates every point of the grid. The row
    # counts travel with the metrics because ARI is only interpretable against the number of
    # identities it had to recover.
    metrics = [{
        "silhouette_avg": silhouette_avg,
        "adjusted_rand_score": ari,
        "n_samples": int(len(embeddings)),
        "n_identities": int(n_identities),
        "n_clusters": int(n_clusters),
    }]
    pd.DataFrame(metrics).assign(**method).to_parquet(output_path)


if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    # noinspection unbound-local-variable
    main(
        dataset_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method={
            "frame_anonymization_method": snakemake.wildcards.get("frame_anonymization"),
            "frame_embedding_method": snakemake.wildcards.get("frame_embedding"),
            "video_aggregation_method": snakemake.wildcards.get("video_aggregation"),
            "video_embedding_method": snakemake.wildcards.get("video_embedding"),
        }
    )
