import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles import logs
from meles.aggregation import clustering, consensus, pooling
from meles.aggregation.aggregator import Aggregator, NoAggregator

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Every key listed under 'identity_aggregation_methods' in the config needs an entry here.
# noinspection SpellCheckingInspection
AGGREGATORS = {
    "none": NoAggregator(),
    "pooling_mean": pooling.MeanPoolingAggregator(),
    "pooling_min": pooling.MinPoolingAggregator(),
    "pooling_max": pooling.MaxPoolingAggregator(),
    "pooling_median": pooling.MedianPoolingAggregator(),
    "clustering_mean_d1_min3":
        clustering.MeanClusteringAggregator(distance_threshold=1.0, min_cluster_size=3),
    "clustering_mean_d08_min3":
        clustering.MeanClusteringAggregator(distance_threshold=0.8, min_cluster_size=3),
    "clustering_median_d1_min3":
        clustering.MedianClusteringAggregator(distance_threshold=1.0, min_cluster_size=3),
    "clustering_median_d08_min3":
        clustering.MedianClusteringAggregator(distance_threshold=0.8, min_cluster_size=3),
    "consensus_beta1": consensus.ConsensusAggregator(beta=1.0),
    "consensus_beta5": consensus.ConsensusAggregator(beta=5.0),
    "consensus_beta20": consensus.ConsensusAggregator(beta=20.0),
}

def identity_level(aggregator: Aggregator) -> Aggregator:
    """Adapt a video-level aggregator to the identity level.

    ``min_cluster_size`` is an outlier filter over *frame* embeddings. At this level the input is
    the handful of centroids 'aggregate_video' produced per video, so re-applying the filter drops
    every cluster and leaves the identity without any gallery embedding at all. Keep the clustering
    geometry, drop the size filter.
    """
    if isinstance(aggregator, clustering.ClusteringAggregator):
        return type(aggregator)(
            distance_threshold=aggregator.distance_threshold,
            n_clusters=aggregator.n_clusters,
            min_cluster_size=0,
        )
    return aggregator

def main(input_path: Path, split_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the aggregator based on the provided method key.
    aggregator = AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown identity aggregation method: {method}")
    aggregator = identity_level(aggregator)

    # Read the input dataset structure and join it with the split. The split holds one row per
    # identity-video, so the merge never duplicates an embedding.
    split_df = pd.read_parquet(split_path)
    structure_df = pd.read_parquet(input_path)
    structure_df = structure_df.merge(split_df, on=['identity', 'video'], how='left')

    # A video missing from the split keeps a null fold, and 'fold != f' holds for null in every
    # iteration below, which would quietly put it into every gallery including the held-out one.
    unassigned = structure_df["fold"].isna().sum()
    if unassigned:
        raise ValueError(f"{unassigned} rows have no fold assigned by the split")

    # Build one gallery per held-out fold: the gallery for fold f aggregates every video of an
    # identity *except* those in fold f, which is the fold that 'score' queries with. The 'fold'
    # column of the output therefore names the held-out fold, not the fold the videos came from.
    folds = sorted(split_df["fold"].unique())
    galleries = []
    for fold in folds:
        logger.debug("Aggregating identities for held-out fold", extra={ "fold": int(fold) })
        gallery_df = (
            structure_df[structure_df["fold"] != fold]
            .groupby(["identity"])["embedding"]
            .apply(lambda embeddings: aggregator.aggregate(embeddings.to_list()))
            .explode()
            .rename("embedding")
            .reset_index()
            .dropna(subset=["embedding"])
        )
        gallery_df["fold"] = fold
        galleries.append(gallery_df)

        # An identity without a single gallery embedding can never be matched, so it silently
        # caps the achievable accuracy. Report it instead of letting it disappear.
        lost = set(structure_df["identity"].unique()) - set(gallery_df["identity"].unique())
        if lost:
            logger.warning("Identities without a gallery embedding", extra={
                "fold": int(fold),
                "len": len(lost),
                "identities": sorted(lost),
            })
    structure_df = pd.concat(galleries, ignore_index=True)

    # Write the full dataset back. It now contains aggregated embeddings for each identity and fold.
    structure_df[["identity", "fold", "embedding"]].to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        split_path=Path(snakemake.input.k_split),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("identity_aggregation"),
    )
