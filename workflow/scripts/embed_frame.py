import logging
import os
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles import runtime
from meles.runtime import logs
from meles.recognizer import models
from meles.recognizer.frame import FrameEmbedder

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Every key listed under 'frame_embedding_methods' in the config needs an entry here.
# Constructing an embedder is free: the weights are downloaded and the ONNX session is built
# on the first call to embed(), so a job only ever loads the one model it was asked for.
# noinspection SpellCheckingInspection
EMBEDDERS = {
    "sface": FrameEmbedder(models.SFACE),
    "arcface": FrameEmbedder(models.ARCFACE),
    "ghostfacenet": FrameEmbedder(models.GHOSTFACENET),
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
    # back up ('..') and Path.relative_to cannot express it. This has to happen before any rows
    # are dropped, because it assigns a list positionally and so needs the original row count.
    # noinspection bad-index
    structure_df["path"] = [os.path.relpath(path, output_path.parent) for path in structure_df_paths]

    # The embedder embeds the whole frame when it finds no face, so this should never fire.
    # It is a guard rather than a formality: the aggregators build a float matrix out of a
    # whole group at once, so a single None in a group raises there instead of here, several
    # rules downstream of the job that actually produced it.
    missing = int(structure_df["embedding"].isna().sum())
    if missing:
        logger.warning("Dropping frames without an embedding", extra={
            "method": method,
            "missing": missing,
            "len": len(structure_df),
        })
        structure_df = structure_df.dropna(subset=["embedding"]).reset_index(drop=True)

    # Write the full dataset back. It now contains created embedding for each frame.
    structure_df.to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    threads = runtime.configure_threads(snakemake.threads)
    for embedder in EMBEDDERS.values():
        embedder.threads = threads

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("frame_embedding"),
    )
