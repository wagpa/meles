"""Assign each identity-video to a fold."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from meles import grouping, runtime

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)


def main(input_path: Path, output_path: Path, k: int, seed: int) -> None:
    """
    Write one row per identity-video with its fold.

    :param input_path: The structure to read the videos from
    :param output_path: The fold assignment to write
    :param k: The number of folds
    :param seed: The seed of the shuffle that decides which videos share a fold
    """
    folds = grouping.assign_folds(pd.read_parquet(input_path, columns=["identity", "video"]), k, seed)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    folds.to_parquet(output_path)

    logger.info("Split videos", extra={"videos": len(folds), "folds": k, "seed": seed})


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])

    main(
        input_path=Path(snakemake.input.structure),
        output_path=Path(snakemake.output.structure),
        k=int(snakemake.params.k),
        seed=int(snakemake.params.seed),
    )
