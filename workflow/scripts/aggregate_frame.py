"""Combine the frames of each group at pixel level."""

import logging
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from tqdm.auto import tqdm

from meles import frame_aggregation, grouping, runtime
from meles.alignment import ARCFACE_TEMPLATE, TEMPLATE_SIZE
from meles.detection import facial_areas
from meles.paths import rebase

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)

# Which of the three outcomes a group had.
REUSED, FRAMEWISE, COLLAPSED = "reused", "framewise", "collapsed"


# The area of a frame that is already an aligned crop: the whole image, with the landmarks where the template puts
# them, so aligning on it again changes nothing.
ALIGNED_AREA: dict[str, object] = {
    "found": True,
    "x": 0,
    "y": 0,
    "w": TEMPLATE_SIZE,
    "h": TEMPLATE_SIZE,
    **{
        f"{axis}{point}": float(ARCFACE_TEMPLATE[point, index])
        for point in range(5)
        for index, axis in enumerate(("lx", "ly"))
    },
}


def read(path: Path) -> NDArray[np.uint8]:
    """Read one frame, BGR."""
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read frame {path}")
    return np.asarray(image, dtype=np.uint8)


def write(image: NDArray[np.uint8], path: Path) -> None:
    """Write one frame, creating its directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(path), image):
        raise OSError(f"Could not write frame {path}")


def unchanged(before: Sequence[NDArray[np.uint8]], after: Sequence[NDArray[np.uint8]]) -> bool:
    """
    Whether the aggregator handed its input back, i.e. produced no new pixels.

    Compared per element so that a fresh list holding the same arrays still counts.
    """
    return len(before) == len(after) and all(a is b for a, b in zip(before, after))


def aggregate_group(
    group: pd.DataFrame,
    columns: list[str],
    aggregator: frame_aggregation.FrameAggregator,
    anonymization: str,
    input_dir: Path,
    output_dir: Path,
) -> tuple[str, pd.DataFrame]:
    """
    Aggregate one group and describe the rows it becomes.

    :param group: The group's rows, in frame order
    :param columns: The columns that define a group
    :param aggregator: The aggregation to apply
    :param anonymization: The key of the anonymization the frames carry
    :param input_dir: The directory the group's paths are relative to
    :param output_dir: The directory the output structure lives in; new images go below it
    :return: The outcome (reused, framewise or collapsed) and the rows of the group's output
    """
    sources = [input_dir / path for path in group["path"]]
    images = [read(source) for source in sources]
    aggregated = aggregator.aggregate(images, facial_areas(group), anonymization)
    first = group.iloc[0]

    if unchanged(images, aggregated):
        rows = group.copy()
        rows["path"] = rebase(group["path"], input_dir, output_dir)
        return REUSED, rows

    # An aligned output is a new face even when it happens to number as many as the input.
    if len(aggregated) == len(group) and not aggregator.aligned:
        # Frame-wise: every frame keeps its name, so the directory stays comparable to the one it came from.
        targets = [output_dir / first["identity"] / first["video"] / frame for frame in group["frame"]]
        for image, target in zip(aggregated, targets):
            write(image, target)
        rows = group.copy()
        rows["path"] = [os.path.relpath(target, output_dir) for target in targets]
        return FRAMEWISE, rows

    # Combined frames have no original to be named after, so they are numbered. A group without a video in its
    # scope spans several, so it is named by fold instead.
    label = first["video"] if "video" in columns else f"fold{first['fold']}"
    names = [f"{index}{first['frame_suffix']}" for index in range(len(aggregated))]
    targets = [output_dir / first["identity"] / label / name for name in names]
    for image, target in zip(aggregated, targets):
        write(image, target)
    rows = pd.DataFrame(
        {
            **{column: first[column] for column in columns},
            "frame": names,
            "frame_suffix": first["frame_suffix"],
            "path": [os.path.relpath(target, output_dir) for target in targets],
        }
    )
    if "video" not in columns:
        rows["video"] = "*"
    if aggregator.aligned:
        rows = rows.assign(**ALIGNED_AREA)
    return COLLAPSED, rows


def main(input_path: Path, split_path: Path, output_path: Path, key: str, anonymization: str, threads: int) -> None:
    """
    Write the aggregated frames and a structure pointing at them.

    :param input_path: The anonymized structure
    :param split_path: The fold of every identity-video
    :param output_path: The structure to write; new images go into its directory
    :param key: The aggregator key, ``{scope}_{method}``
    :param anonymization: The key of the anonymization the input frames carry
    :param threads: How many groups are processed at once
    """
    columns, method = grouping.resolve_scope(key)
    aggregator = frame_aggregation.resolve(method)

    structure = pd.read_parquet(input_path).merge(pd.read_parquet(split_path), on=["identity", "video"], how="left")
    # Sorted so that the frames of a group are contiguous and in a stable order.
    structure = structure.sort_values(columns, kind="stable").reset_index(drop=True)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    if isinstance(aggregator, frame_aggregation.NoFrameAggregator):
        # Known to hand its input back, so the frames need not be decoded to find that out.
        result = structure.copy()
        result["path"] = rebase(structure["path"], input_path.parent, output_dir)
        n_groups = structure.groupby(columns, sort=False).ngroups
        outcomes = [REUSED] * n_groups
    else:
        groups = [group for _, group in structure.groupby(columns, sort=False)]
        n_groups = len(groups)
        outcomes = []
        parts: list[pd.DataFrame] = []
        # A model-backed aggregator owns the device, so its groups run one after another with torch given the
        # whole budget; `configure_threads` only reaches torch once it has been imported.
        if aggregator.aligned:
            import torch

            torch.set_num_threads(threads)
        with ThreadPoolExecutor(max_workers=1 if aggregator.aligned else threads) as pool:
            futures = pool.map(
                lambda group: aggregate_group(group, columns, aggregator, anonymization, input_path.parent, output_dir),
                groups,
            )
            for outcome, rows in tqdm(futures, total=n_groups, unit="group", mininterval=5.0):
                outcomes.append(outcome)
                parts.append(rows)

        # Combined frames lack the input's other columns, among them the stored face. Restoring the columns
        # as missing keeps `facial_areas` readable, and a missing face is `found = False`.
        result = pd.concat(parts, ignore_index=True).reindex(columns=structure.columns)
        if "found" in result:
            result["found"] = result["found"].eq(True)
    result.to_parquet(output_path)

    logger.info(
        "Aggregated frames",
        extra={
            "key": key,
            "groups": n_groups,
            "frames_in": len(structure),
            "frames_out": len(result),
            **{outcome: outcomes.count(outcome) for outcome in (REUSED, FRAMEWISE, COLLAPSED)},
        },
    )


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])
    threads = runtime.configure_threads(snakemake.threads)

    main(
        input_path=Path(snakemake.input.structure),
        split_path=Path(snakemake.input.split),
        output_path=Path(snakemake.output.structure),
        key=snakemake.wildcards.frame_aggregator,
        anonymization=snakemake.wildcards.frame_anonymizer,
        threads=threads,
    )
