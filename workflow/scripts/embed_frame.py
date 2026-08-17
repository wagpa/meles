import logging
import os
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles import logs
from meles.recognizer.deepface import DeepFaceFrameEmbedder

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# noinspection SpellCheckingInspection
EMBEDDERS = {
    "arcface": DeepFaceFrameEmbedder(model_name="ArcFace"),
    "vggface": DeepFaceFrameEmbedder(model_name="VGG-Face"),
    "facenet": DeepFaceFrameEmbedder(model_name="Facenet"),
    "facenet512": DeepFaceFrameEmbedder(model_name="Facenet512"),
    "openface": DeepFaceFrameEmbedder(model_name="OpenFace"),
    "deepface": DeepFaceFrameEmbedder(model_name="DeepFace"),
    "deepid": DeepFaceFrameEmbedder(model_name="DeepID"),
    "dlib": DeepFaceFrameEmbedder(model_name="Dlib"),
    "sface": DeepFaceFrameEmbedder(model_name="SFace"),
    "ghostfacenet": DeepFaceFrameEmbedder(model_name="GhostFaceNet"),
}

def main(input_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the anonymizer based on the provided method key.
    embedder = EMBEDDERS.get(method)
    if embedder is None:
        raise ValueError(f"Unknown frame embedding method: {method}")

    # Read the input dataset structure and pass it to the anonymizer
    structure_df = pd.read_parquet(input_path)
    structure_df_paths = [str(input_path.parent / path) for path in structure_df["path"]]
    logger.debug("Embedding frames", extra={ "method": method, "len": len(structure_df) })
    # noinspection bad-index
    structure_df["embedding"] = embedder.embed(structure_df_paths)

    # Also, update the frame paths to be relative to the output directory. The frames live in the
    # anonymization output, which is a sibling of this directory, so the relative path has to walk
    # back up ('..') and Path.relative_to cannot express it.
    # noinspection bad-index
    structure_df["path"] = [os.path.relpath(path, output_path.parent) for path in structure_df_paths]

    # Write the full dataset back. It now contains created embedding for each frame.
    structure_df.to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("frame_embedding"),
    )
