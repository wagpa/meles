"""Anonymize the face of every frame, using the face stored for the clear frame."""

import logging
import os
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from meles import anonymization, runtime
from meles.detection import FacialArea, facial_areas

if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

logger = logging.getLogger(__name__)

# Rows handed to the thread pool at a time, so the stored faces of the whole dataset never exist at once.
CHUNK = 2048

# A frame without a stored face is left un-anonymized, which silently weakens the effect being measured.
MAX_MISSING_RATE = 0.02

WRITTEN, UNCHANGED, MISSING = "written", "unchanged", "missing"


def anonymize_one(
    source: Path, target: Path, area: FacialArea | None, anonymizer: anonymization.Anonymizer
) -> tuple[Path, str]:
    """
    Anonymize one frame and write it, unless there is nothing to write.

    :param source: The clear frame
    :param target: Where the anonymized frame goes
    :param area: The face to hide, or None if none was found
    :param anonymizer: The anonymization to apply
    :return: The file the row should point at, and what happened to the frame
    """
    if area is None:
        return source, MISSING

    decoded = cv2.imread(str(source))
    if decoded is None:
        raise FileNotFoundError(f"Could not read frame {source}")
    image = np.asarray(decoded, dtype=np.uint8)

    anonymized = anonymizer.apply(image, area)
    # The very same array means the frame was not changed: point at the original instead of copying it.
    if anonymized is image:
        return source, UNCHANGED

    target.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(target), anonymized):
        raise OSError(f"Could not write frame {target}")
    return target, WRITTEN


def main(input_path: Path, output_path: Path, method: str, threads: int) -> None:
    """
    Write the anonymized frames and a structure pointing at them.

    :param input_path: The structure of the clear frames, with the stored faces
    :param output_path: The structure to write; the images go into its directory
    :param method: The anonymizer key
    :param threads: How many frames are processed at once
    """
    anonymizer = anonymization.resolve(method)
    structure = pd.read_parquet(input_path)
    output_dir = output_path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = [input_path.parent / path for path in structure["path"]]
    # Mirrors identity/video/frame; the frame name carries the original suffix.
    targets = [output_dir / i / v / f for i, v, f in zip(structure["identity"], structure["video"], structure["frame"])]

    def chunks() -> Iterator[tuple[int, int]]:
        for start in range(0, len(structure), CHUNK):
            yield start, min(start + CHUNK, len(structure))

    pointers: list[Path] = []
    counts = {WRITTEN: 0, UNCHANGED: 0, MISSING: 0}
    with (
        ThreadPoolExecutor(max_workers=threads) as pool,
        tqdm(total=len(structure), unit="frame", mininterval=5.0) as bar,
    ):
        for start, stop in chunks():
            areas = facial_areas(structure.iloc[start:stop])
            results = pool.map(
                anonymize_one, sources[start:stop], targets[start:stop], areas, [anonymizer] * (stop - start)
            )
            for pointer, outcome in results:
                pointers.append(pointer)
                counts[outcome] += 1
            bar.update(stop - start)

    structure["path"] = [os.path.relpath(pointer, output_dir) for pointer in pointers]
    structure.to_parquet(output_path)

    logger.info("Anonymized frames", extra={"method": method, "frames": len(structure), **counts})
    if counts[MISSING] > MAX_MISSING_RATE * len(structure):
        logger.error(
            "Many frames had no stored face and were left unanonymized",
            extra={"missing": counts[MISSING], "frames": len(structure)},
        )


if __name__ == "__main__":
    runtime.configure_logging(snakemake.log[0])
    threads = runtime.configure_threads(snakemake.threads)

    main(
        input_path=Path(snakemake.input.structure),
        output_path=Path(snakemake.output.structure),
        method=snakemake.wildcards.frame_anonymizer,
        threads=threads,
    )
