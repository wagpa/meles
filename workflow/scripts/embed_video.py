import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles import logs
from meles.recognizer.deepface import DeepFaceRecognizer

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# TODO add real video-level embedders. Until then every key listed under 'video_embedding_methods'
# in the config maps to a frame-level stand-in so the branch is wired up end to end.
# noinspection SpellCheckingInspection
EMBEDDERS = {
    "dummy1": DeepFaceRecognizer(model_name="ArcFace"),
    "dummy2": DeepFaceRecognizer(model_name="SFace"),
    "dummy3": DeepFaceRecognizer(model_name="GhostFaceNet"),
}

def main(input_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the anonymizer based on the provided method key.
    embedder = EMBEDDERS.get(method)
    if embedder is None:
        raise ValueError(f"Unknown video embedding method: {method}")

    # Read the input dataset structure and pass it to the anonymizer. The frame paths are relative
    # to the input structure file, so they are resolved against its directory before being read.
    structure_df = pd.read_parquet(input_path)
    structure_df = (
        structure_df.groupby(["identity", "video"])["path"]
        .apply(lambda paths: embedder.embed([str(input_path.parent / path) for path in paths]))
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
        method=snakemake.wildcards.get("video_embedding"),
    )
