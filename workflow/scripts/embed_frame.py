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
    "arcface": DeepFaceRecognizer(model_name="ArcFace"),
    "vggface": DeepFaceRecognizer(model_name="VGG-Face"),
    "facenet": DeepFaceRecognizer(model_name="Facenet"),
    "facenet512": DeepFaceRecognizer(model_name="Facenet512"),
    "openface": DeepFaceRecognizer(model_name="OpenFace"),
    "deepface": DeepFaceRecognizer(model_name="DeepFace"),
    "deepid": DeepFaceRecognizer(model_name="DeepID"),
    "dlib": DeepFaceRecognizer(model_name="Dlib"),
    "sface": DeepFaceRecognizer(model_name="SFace"),
    "ghostfacenet": DeepFaceRecognizer(model_name="GhostFaceNet"),
}

def main(input_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the anonymizer based on the provided method key.
    embedder = EMBEDDERS[method]
    if embedder is None:
        raise ValueError(f"Unknown embedding method: {method}")

    # Read the input dataset structure and pass it to the anonymizer
    structure_df = pd.read_parquet(input_path)
    structure_df_paths = [str(input_path.parent / path) for path in structure_df["path"]]
    logger.debug("Embedding frames", extra={ "method": method, "len": len(structure_df) })
    # noinspection bad-index
    structure_df["embedding"] = embedder.embed(structure_df_paths)

    # Also, update the frame paths to be relative to the output directory.
    # noinspection bad-index
    structure_df["path"] = [Path(path).relative_to(output_path.parent) for path in structure_df_paths]

    # Write the full dataset back. It now contains created embedding for each frame.
    structure_df.to_parquet(output_path)

if __name__ == "__main__":
    # noinspection unbound-local-variable
    main(
        input_path=snakemake.input.dataset,
        output_path=snakemake.output.dataset,
        method=snakemake.wildcards.get("frame_embedding"),
    )
