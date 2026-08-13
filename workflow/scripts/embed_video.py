import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles.recognizer.deepface import DeepFaceRecognizer

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# noinspection SpellCheckingInspection
EMBEDDERS = {
    # TODO add a real embedder
    "arcface": DeepFaceRecognizer(model_name="ArcFace"),
}

def main(input_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the anonymizer based on the provided method key.
    embedder = EMBEDDERS[method]
    if embedder is None:
        raise ValueError(f"Unknown video embedding method: {method}")

    # Read the input dataset structure and pass it to the anonymizer
    structure_df = pd.read_parquet(input_path)
    structure_df = (
        structure_df.groupby(["identity", "video"])["path"]
        .apply(lambda paths: embedder.embed(paths.to_list()))
        .explode()
        .rename("embedding")
        .reset_index()
        .dropna(subset=["embedding"])
    )

    # Write the full dataset back. It now contains aggregated embedding for each video.
    structure_df[["identity", "video", "embedding"]].to_parquet(output_path)


if __name__ == "__main__":
    # noinspection unbound-local-variable
    main(
        input_path=snakemake.input.dataset,
        output_path=snakemake.output.dataset,
        method=snakemake.wildcards.get("frame_embedding"),
    )
