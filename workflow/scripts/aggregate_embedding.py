import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles.runtime import logs
from meles.aggregation import clustering, consensus, pooling, normweighted
from meles.aggregation.aggregator import NoAggregator

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Every 'frame_aggregator' named by an experiment leg in the config needs an entry here.
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
    "normweighted": normweighted.NormWeightedAggregator(),
}

def main(input_path: Path, split_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the aggregator based on the provided method key.
    aggregator = AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown frame aggregation method: {method}")

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

    # Collapse the frames of every identity-video into the one or more embeddings the aggregator
    # returns. This drops the frame reference, as several frames are potentially combined into one.
    # The fold is constant within a video, so grouping on it carries it into the output for free.
    # An aggregator returns a list, hence the explode; a clustering aggregator whose clusters all
    # fall below 'min_cluster_size' returns none at all, which explodes to a null row and is dropped.
    logger.debug("Aggregating frames", extra={ "method": method, "len": len(structure_df) })
    videos = structure_df.groupby(["identity", "video"]).ngroups
    structure_df = (
        structure_df.groupby(["identity", "video", "fold"])["embedding"]
        .apply(lambda embeddings: aggregator.aggregate(embeddings.to_list()))
        .explode()
        .rename("embedding")
        .reset_index()
        .dropna(subset=["embedding"])
    )

    # A video that lost every embedding can never be matched, so it silently caps the achievable
    # accuracy. Report it instead of letting it disappear.
    lost = videos - structure_df.groupby(["identity", "video"]).ngroups
    if lost:
        logger.warning("Videos without an aggregated embedding", extra={
            "method": method,
            "lost": lost,
            "videos": videos,
        })

    # Write the full dataset back. It now contains the aggregated embeddings per identity-video.
    logger.info("Aggregated frames", extra={ "method": method, "len": len(structure_df) })
    structure_df[["identity", "video", "fold", "embedding"]].to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        split_path=Path(snakemake.input.k_split),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("frame_aggregation"),
    )
