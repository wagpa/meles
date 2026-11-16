"""Score a gallery condition against a samples condition."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from meles import prediction_aggregation, runtime, scoring

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)

# The wildcards of a pairing, which tag every row of the result.
TAGS = [
    f"{leg}_{axis}"
    for leg in ("gallery", "samples")
    for axis in ("frame_anonymizer", "frame_aggregator", "frame_recognizer", "embedding_aggregator")
] + ["score_aggregator"]


def main(
    gallery_path: Path,
    gallery_embeddings_path: Path,
    samples_path: Path,
    samples_embeddings_path: Path,
    output_path: Path,
    tags: dict[str, str],
) -> None:
    """
    Write one row per query video, tagged with the pairing that produced it.

    :param gallery_path: The gallery leg's structure
    :param gallery_embeddings_path: The gallery leg's embeddings
    :param samples_path: The samples leg's structure
    :param samples_embeddings_path: The samples leg's embeddings
    :param output_path: The scores to write
    :param tags: The wildcards of the pairing; ``score_aggregator`` is ``{gallery collapse}_{prediction}``
    """
    collapse, _, prediction = tags["score_aggregator"].partition("_")

    results, diagnostics = scoring.score_folds(
        pd.read_parquet(gallery_path),
        np.load(gallery_embeddings_path, mmap_mode="r"),
        pd.read_parquet(samples_path),
        np.load(samples_embeddings_path, mmap_mode="r"),
        scoring.resolve_gallery(collapse),
        prediction_aggregation.resolve(prediction),
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    results.assign(**tags).to_parquet(output_path)

    logger.info("Scored pairing", extra=diagnostics)
    logger.info("Accuracy", extra={"top-1": float(results["top-1"].mean())})


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])

    main(
        gallery_path=Path(snakemake.input.gallery),
        gallery_embeddings_path=Path(snakemake.input.gallery_embeddings),
        samples_path=Path(snakemake.input.samples),
        samples_embeddings_path=Path(snakemake.input.samples_embeddings),
        output_path=Path(snakemake.output.structure),
        tags={tag: snakemake.wildcards[tag] for tag in TAGS},
    )
