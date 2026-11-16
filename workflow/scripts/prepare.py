"""Select the frames of the configured identities and attach the face stored for each."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

from meles import runtime
from meles.paths import rebase

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)


def main(
    dataset_path: Path,
    splits_path: Path,
    areas_path: Path,
    output_path: Path,
    subset: str,
    identities: int | None,
) -> None:
    """
    Write the structure of the subset's frames with their stored faces.

    :param dataset_path: The dataset structure, one row per frame
    :param splits_path: The split of every identity
    :param areas_path: The face found in each clear frame, keyed by the dataset's path
    :param output_path: The structure to write
    :param subset: The split to keep
    :param identities: Keep only this many identities (the first ones, sorted), or None for all
    """
    splits = pd.read_parquet(splits_path)
    kept = sorted(splits.loc[splits["split"] == subset, "identity"])
    if identities is not None:
        kept = kept[:identities]

    dataset = pd.read_parquet(dataset_path)
    structure = dataset[dataset["identity"].isin(kept)].reset_index(drop=True)

    areas = pd.read_parquet(areas_path)
    areas = areas[areas["path"].isin(structure["path"])]
    structure = structure.merge(areas, on="path", how="left")
    # A frame without a stored area has no face to work on.
    structure["found"] = structure["found"].fillna(False).astype(bool)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    structure["path"] = rebase(structure["path"], dataset_path.parent, output_path.parent)
    structure.to_parquet(output_path)

    logger.info(
        "Prepared structure",
        extra={
            "subset": subset,
            "identities": len(kept),
            "frames": len(structure),
            "without_area": int((~structure["found"]).sum()),
        },
    )


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    main(
        dataset_path=Path(snakemake.input.dataset),
        splits_path=Path(snakemake.input.splits),
        areas_path=Path(snakemake.input.areas),
        output_path=Path(snakemake.output.structure),
        subset=snakemake.params.subset,
        identities=snakemake.params.identities,
    )
