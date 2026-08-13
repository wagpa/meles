import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles.aggregation.aggregator import NoAggregator

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# noinspection SpellCheckingInspection
AGGREGATORS = {
    "none": NoAggregator(),
}

def main(input_path: Path, split_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the aggregator based on the provided method key.
    aggregator = AGGREGATORS[method]
    if aggregator is None:
        raise ValueError(f"Unknown video aggregation method: {method}")

    # Read the input dataset structure, join it with the split, and pass it to the aggregator
    split_df = pd.read_parquet(split_path)
    structure_df = pd.read_parquet(input_path)
    structure_df = structure_df.merge(split_df, on=['identity', 'video'], how='left')
    structure_df = (
        structure_df.groupby(["identity", "fold"])["embedding"]
        .apply(lambda embeddings: aggregator.aggregate(embeddings.to_list()))
        .explode()
        .rename("embedding")
        .reset_index()
        .dropna(subset=["embedding"])
    )

    # Write the full dataset back. It now contains aggregated embedding for each video.
    structure_df[["identity", "fold", "embedding"]].to_parquet(output_path)

if __name__ == "__main__":
    # noinspection unbound-local-variable
    main(
        input_path=snakemake.input.dataset,
        split_path=snakemake.input.k_split,
        output_path=snakemake.output.dataset,
        method=snakemake.wildcards.get("identity_aggregation"),
    )
