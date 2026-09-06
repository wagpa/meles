import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles.runtime import logs, runtime
from meles.embedding_aggregation import clustering, consensus, pooling, normweighted
from meles.embedding_aggregation.aggregator import EmbeddingAggregator, NoEmbeddingAggregator

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Every group is confined to one identity and one fold, whatever the scope. The identity is not
# optional and is deliberately not left to the scopes to remember: 'video' is only a per-identity
# number ("0", "1", ...), so every identity has a video "0" and a grouping that left the identity
# out would aggregate unrelated people into one embedding.
ALWAYS_GROUPED = ["identity", "fold"]

# What *else* separates groups, per scope: 'video' keeps the videos of an identity apart, so it
# aggregates the frames of one video; 'identity' adds nothing, so everything the fold holds of that
# identity is aggregated together.
SCOPES = {
    "video": ["video"],
    "identity": [],
}

# The identity scope leaves no single video to name, but the column stays: 'score' keys its queries
# by identity-video, and one uniform output shape is worth more than a column that comes and goes.
ALL_VIDEOS = "*"

# Every 'embedding_aggregator' named by an experiment leg in the config needs an entry here.
# noinspection SpellCheckingInspection
AGGREGATORS = {
    "none": NoEmbeddingAggregator(),
    "pooling_mean": pooling.MeanPoolingEmbeddingAggregator(),
    "pooling_min": pooling.MinPoolingEmbeddingAggregator(),
    "pooling_max": pooling.MaxPoolingEmbeddingAggregator(),
    "pooling_median": pooling.MedianPoolingEmbeddingAggregator(),
    "clustering_mean_d1_min3": clustering.MeanClusteringAggregator(distance_threshold=1.0, min_cluster_size=3),
    "clustering_mean_d08_min3": clustering.MeanClusteringAggregator(distance_threshold=0.8, min_cluster_size=3),
    "clustering_median_d1_min3": clustering.MedianClusteringAggregator(distance_threshold=1.0, min_cluster_size=3),
    "clustering_median_d08_min3": clustering.MedianClusteringAggregator(distance_threshold=0.8, min_cluster_size=3),
    "consensus_beta1": consensus.ConsensusEmbeddingAggregator(beta=1.0),
    "consensus_beta5": consensus.ConsensusEmbeddingAggregator(beta=5.0),
    "consensus_beta20": consensus.ConsensusEmbeddingAggregator(beta=20.0),
    "normweighted": normweighted.NormWeightedEmbeddingAggregator(),
}


def resolve(embedding_aggregator: str) -> tuple[list[str], EmbeddingAggregator]:
    """
    Read a '{scope}_{method}' key as the columns defining a group and the aggregator to collapse it.

    :param embedding_aggregator: The method key, e.g. 'video_pooling_mean' or 'identity_none'
    :return: The complete grouping columns, always including the identity and the fold, and the
        aggregator to collapse each group with
    :raises ValueError: If either half names something the registries have no entry for
    """
    scope, _, method = embedding_aggregator.partition("_")

    separators = SCOPES.get(scope)
    if separators is None:
        raise ValueError(
            f"Unknown aggregation scope {scope!r} in {embedding_aggregator!r}, expected one of {sorted(SCOPES)}"
        )

    aggregator = AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown aggregation method {method!r} in {embedding_aggregator!r}")
    return ALWAYS_GROUPED + separators, aggregator


def main(input_path: Path, split_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the scope and the aggregator based on the provided method key.
    group_by, aggregator = resolve(method)

    # Read the input dataset structure and attach the fold of every video. The split holds one row
    # per identity-video, so the merge never duplicates an embedding. The fold only rides along:
    # this rule aggregates every video the same way, and 'score' is what decides which folds make
    # up a gallery and which are queried against it.
    split_df = pd.read_parquet(split_path)
    structure_df = pd.read_parquet(input_path).merge(split_df, on=["identity", "video"], how="left")

    # A video missing from the split keeps a null fold, which no fold comparison in 'score' selects,
    # so it would silently drop out of the experiment instead of failing here.
    unassigned = structure_df["fold"].isna().sum()
    if unassigned:
        raise ValueError(f"{unassigned} rows have no fold assigned by the split")

    # Collapse the frames of every group into the one or more embeddings the aggregator returns.
    # This drops the frame reference, as several frames are potentially combined into one, and under
    # the identity scope the video reference as well. The identity and the fold are part of every
    # group key, so they survive the aggregation and land back in the output for free.
    # An aggregator returns a list, hence the explode; a clustering aggregator whose clusters all
    # fall below 'min_cluster_size' returns none at all, which explodes to a null row and is dropped.
    logger.debug("Aggregating frames", extra={"method": method, "len": len(structure_df)})
    groups = structure_df.groupby(group_by).ngroups
    structure_df = (
        structure_df.groupby(group_by)["embedding"]
        .apply(lambda embeddings: aggregator.aggregate(embeddings.to_list()))
        .explode()
        .rename("embedding")
        .reset_index()
        .dropna(subset=["embedding"])
    )

    # A group that lost every embedding can never be matched, so it silently caps the achievable
    # accuracy. Report it instead of letting it disappear.
    lost = groups - structure_df.groupby(group_by).ngroups
    if lost:
        logger.warning(
            "Groups without an aggregated embedding",
            extra={
                "method": method,
                "lost": lost,
                "groups": groups,
            },
        )

    # Write the full dataset back. It now contains the aggregated embeddings per group.
    if "video" not in structure_df:
        structure_df["video"] = ALL_VIDEOS
    logger.info("Aggregated frames", extra={"method": method, "len": len(structure_df)})
    structure_df[["identity", "video", "fold", "embedding"]].to_parquet(output_path)


if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        split_path=Path(snakemake.input.k_split),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("embedding_aggregation"),
    )
