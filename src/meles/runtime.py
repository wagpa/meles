"""
Per-job setup for Snakemake script rules: logging to the job's log file and a thread budget.

Snakemake's ``log:`` directive only *names* a file. A ``script:`` rule runs ``python <script>`` as a
subprocess that inherits the parent's stdout and stderr, so without redirecting, everything lands on the
console and the declared log file stays empty. :func:`configure_logging` redirects at the file-descriptor
level, because a job has three independent writers and only ``os.dup2`` catches all of them: the logging
module, tqdm's progress bars, and OpenCV / ONNX Runtime, which write to fd 2 from C and never see a
``sys.stderr`` reassignment.

Snakemake runs several jobs at once. OpenCV and ONNX Runtime each default to every hardware thread, so
concurrent jobs contend and run slower than jobs confined to their share; :func:`configure_threads` applies it.
"""

import logging
import os
import sys
from pathlib import Path

import cv2

logger = logging.getLogger(__name__)

# The attributes every LogRecord carries. Anything else was passed by the caller via 'extra' and is
# appended to the message, so that logger.debug("...", extra={"path": p}) actually shows the path.
_BUILTIN_RECORD_ATTRS = set(logging.LogRecord("", 0, "", 0, "", None, None).__dict__) | {
    "message",
    "asctime",
    "taskName",
}


class ExtraFormatter(logging.Formatter):
    """A formatter that appends the ``extra`` fields of a record as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        """Format the record, followed by its ``extra`` fields."""
        message = super().format(record)
        extras = {key: value for key, value in record.__dict__.items() if key not in _BUILTIN_RECORD_ATTRS}
        if extras:
            message += " " + " ".join(f"{key}={value!r}" for key, value in extras.items())
        return message


def configure_logging(log_path: str | Path, level: int = logging.INFO) -> Path:
    """
    Send this process' output to the Snakemake log file of the running job.

    :param log_path: The job's log file, i.e. ``snakemake.log[0]``
    :param level: The lowest level that is written; pass ``logging.DEBUG`` for the per-frame logs
    :return: The log file the output was redirected to
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Line buffered, so that a running job can be followed with 'tail -f'.
    stream = open(log_path, "a", buffering=1)
    os.dup2(stream.fileno(), sys.stdout.fileno())
    os.dup2(stream.fileno(), sys.stderr.fileno())

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(ExtraFormatter("%(asctime)s %(levelname)-7s %(name)s | %(message)s"))
    logging.basicConfig(level=level, handlers=[handler], force=True)
    return log_path


def configure_threads(threads: int | None) -> int:
    """
    Limit OpenCV and, if already imported, PyTorch to the given number of threads.

    ONNX Runtime is configured separately: its pool is fixed when the session is built, via the ``threads``
    argument of :class:`meles.recognition.FrameEmbedder` / :class:`meles.inference.OnnxSession`, so callers
    pass the returned budget there.

    :param threads: The thread budget, normally ``snakemake.threads``; None means one
    :return: The thread budget that was applied
    """
    budget = max(1, int(threads or 1))

    # Environment variables such as OMP_NUM_THREADS are only read when a library loads, so the libraries
    # are called directly. Torch is not imported here: it is slow to import and most jobs do not use it.
    cv2.setNumThreads(budget)
    if "torch" in sys.modules:
        sys.modules["torch"].set_num_threads(budget)

    logger.debug("Configured thread budget", extra={"threads": budget})
    return budget
