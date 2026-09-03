"""
Confine a Snakemake job to the core budget it was given.

OpenCV defaults its thread pool to every hardware thread it can see, and ONNX Runtime does
the same. Snakemake happily runs several jobs at once, so without this every job grabs the
whole machine and they all contend: on this dataset four concurrent jobs each spawning twelve
OpenCV threads on twelve hardware threads measurably ran *slower* than four jobs of three.
"""

import logging
import os

import cv2

logger = logging.getLogger(__name__)


def configure_threads(threads: int | None) -> int:
    """
    Limit OpenCV and the OpenMP-based libraries to the given number of threads.

    ONNX Runtime is configured separately, when its session is built, because its thread pool
    is fixed at session creation and reads ``SessionOptions`` rather than the environment.

    :param threads: The thread budget, normally ``snakemake.threads``; None means one
    :return: The thread budget that was applied
    """
    budget = max(1, int(threads or 1))
    cv2.setNumThreads(budget)
    os.environ.setdefault("OMP_NUM_THREADS", str(budget))
    logger.debug("Configured thread budget", extra={"threads": budget})
    return budget
