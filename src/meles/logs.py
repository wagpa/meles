"""
Per-job logging setup for the Snakemake scripts.

Snakemake's ``log:`` directive only *names* a file. A ``shell:`` rule redirects into it explicitly
(``> {log} 2>&1``), but a ``script:`` rule runs ``python <script>`` as a subprocess that inherits
the parent's stdout and stderr, so everything a job writes lands on the console interleaved with
Snakemake's own progress output — and the declared log file stays empty.

``configure`` points the job at its log file instead. It redirects at the file-descriptor level
because a job has three independent writers and only ``os.dup2`` catches all of them:

- the logging module, which without a handler falls back to writing warnings to stderr,
- tqdm's progress bars, which write to stderr,
- TensorFlow/OpenCV, which write to fd 2 from C and never see a ``sys.stderr`` reassignment.
"""

import logging
import os
import sys
from pathlib import Path

# The attributes every LogRecord carries. Anything else was passed by the caller via 'extra' and is
# appended to the message, so that logger.debug("...", extra={"path": p}) actually shows the path.
_BUILTIN_RECORD_ATTRS = set(
    logging.LogRecord("", 0, "", 0, "", None, None).__dict__
) | {"message", "asctime", "taskName"}


class ExtraFormatter(logging.Formatter):
    """A formatter that appends the ``extra`` fields of a record as key=value pairs."""

    def format(self, record: logging.LogRecord) -> str:
        message = super().format(record)
        extras = {
            key: value
            for key, value in record.__dict__.items()
            if key not in _BUILTIN_RECORD_ATTRS
        }
        if extras:
            message += " " + " ".join(f"{key}={value!r}" for key, value in extras.items())
        return message


def configure(log_path: str | Path, level: int = logging.INFO) -> Path:
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
