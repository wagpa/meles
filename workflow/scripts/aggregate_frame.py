import logging
import os
from pathlib import Path
import pandas as pd
import cv2
from tqdm import tqdm
from typing import TYPE_CHECKING

from meles.common import Images
from meles.runtime import logs, runtime
from meles.frame_aggregation import resample
from meles.frame_aggregation.aggregator import FrameAggregator, NoFrameAggregator

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Every group is confined to one identity and one fold, whatever the scope. The identity is not
# optional and is deliberately not left to the scopes to remember: 'video' is only a per-identity
# number ("0", "1", ...), so every identity has a video "0" and a grouping that left the identity
# out would aggregate unrelated people into one frame.
ALWAYS_GROUPED = ["identity", "fold"]

# What *else* separates groups, per scope. Only 'video' exists here, unlike in 'aggregate_embedding'
# where both do: it keeps the videos of an identity apart, so a group is one identity-video, which
# is also the granularity the split assigns folds at.
SCOPES = {
    "video": ["video"],
}

# 'identity' is refused rather than half-supported. Collapsing the videos of an identity into one
# frame leaves no video to name, and every rule after this one is keyed by identity-video:
# 'aggregate_embedding' merges the split onto it and 'score' pools its per-frame votes by it. Both
# would quietly start measuring something else instead of failing, which is the worst outcome.
UNSUPPORTED_SCOPES = {
    "identity": (
        "it leaves no video to name, which 'aggregate_embedding' needs to merge the split onto and "
        "'score' needs to pool its votes by; aggregate across an identity at the embedding stage "
        "instead, with an 'identity_*' embedding_aggregator"
    ),
}

# Every 'frame_aggregator' named by an experiment leg in the config needs an entry here.
# A 'frame_*' method is frame-wise: it returns as many frames as it was given and combines nothing,
# which is how the de-anonymizations fit a rule otherwise about aggregation.
# noinspection SpellCheckingInspection
AGGREGATORS = {
    "none": NoFrameAggregator(),
    "frame_bicubic": resample.BicubicFrameAggregator(),
}


def resolve(frame_aggregator: str) -> tuple[list[str], FrameAggregator]:
    """
    Read a '{scope}_{method}' key as the columns defining a group and the aggregator to collapse it.

    :param frame_aggregator: The method key, e.g. 'video_none'
    :return: The complete grouping columns, always including the identity and the fold, and the
        aggregator to collapse each group with
    :raises ValueError: If either half names something the registries have no entry for
    """
    scope, _, method = frame_aggregator.partition("_")

    separators = SCOPES.get(scope)
    if separators is None:
        unsupported = UNSUPPORTED_SCOPES.get(scope)
        if unsupported is not None:
            raise ValueError(
                f"Aggregation scope {scope!r} in {frame_aggregator!r} is not supported here: {unsupported}"
            )
        raise ValueError(
            f"Unknown aggregation scope {scope!r} in {frame_aggregator!r}, expected one of {sorted(SCOPES)}"
        )

    aggregator = AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown aggregation method {method!r} in {frame_aggregator!r}")
    return ALWAYS_GROUPED + separators, aggregator


def unchanged(before: Images, after: Images) -> bool:
    """
    Whether an aggregator handed its input straight back, i.e. produced no new pixels.

    Checked per element rather than on the sequence, so that an aggregator returning a fresh list of
    the same images still counts. It is the baseline ('none') that this is for: copying the whole
    anonymized dataset into every aggregation directory just to leave it untouched would double the
    largest thing on disk for nothing.

    :param before: The frames handed to the aggregator
    :param after: The frames it returned
    :return: True if every returned frame is one of the originals, in the original order
    """
    return len(before) == len(after) and all(a is b for a, b in zip(before, after))


def read(path: Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"Image not found: {path}")
    return image


def main(input_path: Path, split_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the scope and the aggregator based on the provided method key.
    group_by, aggregator = resolve(method)

    # Read the input dataset structure and attach the fold of every video. The split holds one row
    # per identity-video, so the merge never duplicates a frame. The fold only rides along as part
    # of the group key, keeping a group from ever spanning folds; it is deliberately not written
    # out again, because 'aggregate_embedding' merges the split onto its own input and a 'fold'
    # already sitting there would collide with it.
    split_df = pd.read_parquet(split_path)
    structure_df = pd.read_parquet(input_path).merge(split_df, on=["identity", "video"], how="left")

    # A video missing from the split keeps a null fold, which no fold comparison in 'score' selects,
    # so it would silently drop out of the experiment instead of failing here.
    unassigned = structure_df["fold"].isna().sum()
    if unassigned:
        raise ValueError(f"{unassigned} rows have no fold assigned by the split")

    # Collapse the frames of every group into the one or more frames the aggregator returns. This is
    # the pixel-level counterpart of 'aggregate_embedding': many anonymized observations are combined
    # before anything has looked at them, rather than after each was embedded on its own.
    logger.debug("Aggregating frames", extra={"method": method, "len": len(structure_df)})
    grouped = structure_df.groupby(group_by, sort=False)
    # The bar advances once per group rather than once per frame, and 'mininterval' keeps even that
    # from writing megabytes of carriage returns into the job log.
    progress = tqdm(grouped, total=grouped.ngroups, desc="Aggregating", mininterval=5.0)
    # Counted rather than logged per group: the totals are what say whether the aggregator did
    # anything, and a line per group is a line per identity-video.
    reused, written, lost = 0, 0, 0
    aggregated_dfs = []
    for _, group in progress:
        # Read every frame of the group. The path is always relative to the input file (parent) path.
        sources = [input_path.parent / path for path in group["path"]]
        images = [read(source) for source in sources]
        aggregated = aggregator.aggregate(images)

        # An aggregator that produced no new pixels names the frames where they already lie, which
        # is a sibling directory, so the relative path has to walk back up ('..') and
        # Path.relative_to cannot express it -- the same reason 'embed_frame' uses os.path.relpath.
        if unchanged(images, aggregated):
            reused += len(group)
            kept_df = group[["identity", "video", "frame", "frame_suffix"]].copy()
            kept_df["path"] = [os.path.relpath(source, output_path.parent) for source in sources]
            aggregated_dfs.append(kept_df)
            continue

        # Otherwise the aggregated frames are new images and have to be written out. A group is one
        # identity-video, so the whole group shares both.
        identity, video = group["identity"].iloc[0], group["video"].iloc[0]
        suffix = group["frame_suffix"].iloc[0]
        # An aggregator that returned as many frames as it was given combined nothing -- it worked
        # frame-wise, which is what the 'frame_*' de-anonymizations do -- so every frame still
        # answers to the name it arrived under, and this directory stays comparable to the
        # anonymized one it was built from. A group that really collapsed has no original frame
        # left to name its result after, so those are numbered instead.
        if len(aggregated) == len(group):
            names = list(group["frame"])
        else:
            names = [f"{method}.{index}{suffix}" for index in range(len(aggregated))]
        made = []
        for frame, image in zip(names, aggregated):
            image_output_path = output_path.parent / identity / video / frame
            os.makedirs(image_output_path.parent, exist_ok=True)
            if not cv2.imwrite(str(image_output_path), image):
                raise ValueError(f"Failed to write aggregated frame: {image_output_path}")
            made.append(
                {
                    "identity": identity,
                    "video": video,
                    "frame": frame,
                    "frame_suffix": suffix,
                    "path": str(image_output_path.relative_to(output_path.parent)),
                }
            )

        # A group that lost every frame can never be matched, so it silently caps the achievable
        # accuracy. Report it instead of letting it disappear.
        if not made:
            lost += 1
            continue
        written += len(made)
        aggregated_dfs.append(pd.DataFrame(made))

    if lost:
        logger.warning(
            "Groups without an aggregated frame",
            extra={
                "method": method,
                "lost": lost,
                "groups": grouped.ngroups,
            },
        )

    # Write the full dataset back. It now contains the aggregated frames per group.
    structure_df = pd.concat(aggregated_dfs, ignore_index=True) if aggregated_dfs else structure_df.iloc[:0]
    logger.info(
        "Aggregated frames",
        extra={
            "method": method,
            "len": len(structure_df),
            "groups": grouped.ngroups,
            "reused": reused,
            "written": written,
        },
    )
    structure_df[["identity", "video", "frame", "frame_suffix", "path"]].to_parquet(output_path)


if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        split_path=Path(snakemake.input.k_split),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("frame_aggregation"),
    )
