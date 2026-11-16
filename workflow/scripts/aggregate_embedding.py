"""Combine the embeddings of each group."""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from meles import embedding_aggregation, grouping, runtime

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)


def main(input_path: Path, embeddings_path: Path, output_path: Path, output_embeddings_path: Path, key: str) -> None:
    """
    Write one row per aggregated embedding with the keys of its group.

    :param input_path: The structure of the embedded frames, with the fold
    :param embeddings_path: The embeddings of those frames, one row per structure row
    :param output_path: The structure to write: identity, video and fold
    :param output_embeddings_path: The aggregated embeddings to write, one row per output row
    :param key: The aggregator key, ``{scope}_{method}``
    """
    columns, method = grouping.resolve_scope(key)
    aggregator = embedding_aggregation.resolve(method)

    structure = pd.read_parquet(input_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")

    # Sorted so that every group is one run of rows; the embeddings follow the same order.
    order = structure.sort_values(columns, kind="stable").index.to_numpy()
    ordered = structure.loc[order, columns].reset_index(drop=True)
    starts = np.flatnonzero(~ordered.duplicated().to_numpy())

    aggregated, groups = embedding_aggregation.aggregate_groups(
        np.asarray(embeddings[order]), starts.tolist(), aggregator
    )

    # An aggregator may return several embeddings per group, or none, so rows are taken from their group.
    rows = ordered.iloc[starts].iloc[groups].reset_index(drop=True)
    if "video" not in columns:
        # An identity-wide group has no single video to name.
        rows["video"] = "*"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows[["identity", "video", "fold"]].to_parquet(output_path)
    np.save(output_embeddings_path, aggregated)

    logger.info(
        "Aggregated embeddings", extra={"key": key, "groups": len(starts), "frames": len(structure), "rows": len(rows)}
    )


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])

    main(
        input_path=Path(snakemake.input.structure),
        embeddings_path=Path(snakemake.input.embeddings),
        output_path=Path(snakemake.output.structure),
        output_embeddings_path=Path(snakemake.output.embeddings),
        key=snakemake.wildcards.embedding_aggregator,
    )
