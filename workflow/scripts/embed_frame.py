"""Embed every frame with one recognizer."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from meles import recognition, runtime
from meles.detection import facial_areas
from meles.paths import rebase

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)

# Frames embedded per call, which bounds how many decoded frames and stored faces are alive at once.
CHUNK = 2048

# Whether each detection mode aligns on the stored faces instead of detecting in the frame.
USE_STORED_AREAS = {"oracle": True, "observed": False}


def main(input_path: Path, output_path: Path, embeddings_path: Path, key: str, threads: int) -> None:
    """
    Write the structure and the embedding of each of its rows.

    :param input_path: The structure of the frames to embed
    :param output_path: The structure to write, with paths relative to it
    :param embeddings_path: The float32 array to write, one row per structure row
    :param key: The recognizer key, ``{detection}_{model}``; ``oracle`` aligns on the stored face, ``observed``
        detects it in the frame as the recognizer sees it
    :param threads: The thread budget of the recognition model
    """
    detection, _, model = key.partition("_")
    use_stored = USE_STORED_AREAS[detection]
    embedder = recognition.FrameEmbedder(recognition.resolve(model), threads=threads)

    structure = pd.read_parquet(input_path)
    frames = [input_path.parent / path for path in structure["path"]]
    embeddings_path.parent.mkdir(parents=True, exist_ok=True)

    # Written straight to disk, so the whole array never has to fit in memory.
    embeddings = np.lib.format.open_memmap(
        embeddings_path, mode="w+", dtype=np.float32, shape=(len(structure), embedder.spec.output_dim)
    )
    for start in tqdm(range(0, len(structure), CHUNK), desc=f"Embedding ({key})", unit="chunk", mininterval=5.0):
        stop = min(start + CHUNK, len(structure))
        areas = facial_areas(structure.iloc[start:stop]) if use_stored else None
        embeddings[start:stop] = embedder.embed(frames[start:stop], areas=areas, progress=False)
    embeddings.flush()

    structure["path"] = rebase(structure["path"], input_path.parent, output_path.parent)
    structure.to_parquet(output_path)

    logger.info("Embedded frames", extra={"key": key, "frames": len(structure), "dim": embedder.spec.output_dim})


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])
    threads = runtime.configure_threads(snakemake.threads)

    main(
        input_path=Path(snakemake.input.structure),
        output_path=Path(snakemake.output.structure),
        embeddings_path=Path(snakemake.output.embeddings),
        key=snakemake.wildcards.frame_recognizer,
        threads=threads,
    )
