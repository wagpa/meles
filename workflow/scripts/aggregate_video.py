import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles import logs
from meles.aggregation import clustering, consensus, pooling, normweighted
from meles.aggregation.aggregator import NoAggregator

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Every key listed under 'video_aggregation_methods' in the config needs an entry here.
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

def main(input_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the aggregator based on the provided method key.
    aggregator = AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown video aggregation method: {method}")

    # Read the input dataset structure and pass it to the aggregator
    structure_df = pd.read_parquet(input_path)
    structure_df = (
        structure_df.groupby(["identity", "video"])["embedding"]
        .apply(lambda embeddings: aggregator.aggregate(embeddings.to_list()))
        .explode()
        .rename("embedding")
        .reset_index()
        .dropna(subset=["embedding"])
    )

    # Write the full dataset back. It now contains aggregated embedding for each video.
    structure_df[["identity", "video", "embedding"]].to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("video_aggregation"),
    )
