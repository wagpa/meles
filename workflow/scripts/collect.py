"""Gather every scored pairing into one table."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from meles import runtime

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)


def main(score_paths: list[Path], output_path: Path) -> None:
    """
    Concatenate the scores of all pairings; every row already names its pairing.

    :param score_paths: The score files to gather
    :param output_path: The table to write
    """
    collected = pd.concat([pd.read_parquet(path) for path in score_paths], ignore_index=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    collected.to_parquet(output_path)

    logger.info("Collected scores", extra={"files": len(score_paths), "rows": len(collected)})


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])

    main(
        score_paths=[Path(path) for path in snakemake.input.scores],
        output_path=Path(snakemake.output.table),
    )
