"""
Run an ONNX face-recognition graph on a batch of aligned crops.

The three models this project uses were exported by three different toolchains and disagree
about almost everything: SFace and GhostFaceNet declare a fixed batch size of one, ArcFace
declares a symbolic one; GhostFaceNet wants NHWC while the other two want NCHW; and SFace
bakes its input normalisation into the graph while the others expect it applied beforehand.
This module hides those differences behind one ``run(blob) -> (batch, dim)`` call.
"""

import contextlib
import io
import logging
import os
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# The environment variable overriding which execution providers are tried, as a comma
# separated list.
PROVIDERS_ENV = "MELES_ORT_PROVIDERS"

# The providers tried, in order. ONNX Runtime falls through to the next one when a provider
# cannot be initialised, so a machine without a usable CUDA stack still runs -- just on the
# CPU. Which one actually bound is logged, because a silent fall back to the CPU is otherwise
# indistinguishable from a slow GPU.
DEFAULT_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")


def providers() -> list[str]:
    """
    The execution providers to try, filtered to those this build of ONNX Runtime has.

    :return: The provider names, most preferred first
    """
    configured = os.environ.get(PROVIDERS_ENV)
    wanted = [name.strip() for name in configured.split(",") if name.strip()] if configured else list(DEFAULT_PROVIDERS)

    available = set(ort.get_available_providers())
    usable = [name for name in wanted if name in available]
    if not usable:
        logger.warning(
            "None of the requested execution providers are available, falling back to the CPU",
            extra={"requested": wanted, "available": sorted(available)},
        )
        return ["CPUExecutionProvider"]
    return usable


def _with_dynamic_batch(path: Path) -> bytes:
    """
    Relax a graph's leading input and output dimension to a symbolic batch size.

    SFace and GhostFaceNet were exported with the batch dimension pinned to one, which makes
    ONNX Runtime reject a stacked blob even though every shape-dependent operator inside them
    already derives the batch size at runtime (all of their ``Reshape`` targets start with
    -1). Rewriting the declared dimension is therefore sound, and
    ``test_batched_inference_matches_single_images`` in ``src/tests/test_embedders.py`` pins that
    down by asserting a batched run reproduces single-image runs exactly.

    :param path: The model to rewrite
    :return: The rewritten model, serialized
    """
    model = onnx.load(str(path))

    def relax(value: Any) -> None:
        dimension = value.type.tensor_type.shape.dim[0]
        dimension.ClearField("dim_value")
        dimension.dim_param = "batch"

    relax(model.graph.input[0])
    for output in model.graph.output:
        relax(output)
    return model.SerializeToString()


@contextlib.contextmanager
def _quiet_output() -> Iterator[None]:
    """
    Silence writes to stdout and stderr for the duration of the block.

    When a requested execution provider cannot be initialised, ONNX Runtime's Python wrapper
    prints a fifteen-line C++ diagnostic and then quietly retries on the CPU. It is neither
    raised nor routed through the logger, so it cannot be filtered by severity -- and with one
    session per job it would bury the actual contents of every Snakemake log. The interesting
    part (which provider ended up bound) is recovered from ``get_providers()`` afterwards and
    logged in one line.

    Both stdout and stderr are covered, and each at two levels. The provider diagnostic is a
    Python ``print``, so it sits in ``sys.stdout``'s buffer and would still surface after a
    file-descriptor swap was undone; the runtime's own warnings are written from C++ and never
    see ``sys.stdout`` at all. Only redirecting both levels silences both. Nothing else writes
    here -- the block only wraps the session construction.
    """
    sink = io.StringIO()
    saved = [os.dup(1), os.dup(2)]
    with open(os.devnull, "w") as null:
        os.dup2(null.fileno(), 1)
        os.dup2(null.fileno(), 2)
    try:
        with contextlib.redirect_stdout(sink), contextlib.redirect_stderr(sink):
            yield
    finally:
        for descriptor, backup in zip((1, 2), saved):
            os.dup2(backup, descriptor)
            os.close(backup)
    if sink.getvalue().strip():
        logger.debug("Suppressed runtime output while loading", extra={"output": sink.getvalue()})


class OnnxSession:
    """A face-recognition graph, held open for repeated batched inference."""

    def __init__(self, path: Path, threads: int = 0, dynamic_batch: bool = True):
        """
        :param path: The ONNX model to load
        :param threads: The intra-operator thread budget; 0 leaves ONNX Runtime's default
        :param dynamic_batch: Whether to relax a pinned batch dimension before loading
        """
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if threads > 0:
            options.intra_op_num_threads = threads
        # ONNX Runtime is chatty on stderr, which Snakemake captures into the job log.
        options.log_severity_level = 3

        model = _with_dynamic_batch(path) if dynamic_batch else str(path)
        requested = providers()
        with _quiet_output():
            self._session = ort.InferenceSession(model, options, providers=requested)
        self._input = self._session.get_inputs()[0].name

        bound = self._session.get_providers()
        logger.info(
            "Loaded recognition model",
            extra={"model": path.name, "providers": bound, "input": self._input},
        )
        # A silent fall back to the CPU is indistinguishable from a slow GPU, and on this dataset
        # that is the difference between minutes and hours, so it is worth a warning.
        unavailable = [name for name in requested if name not in bound]
        if unavailable:
            logger.warning(
                "Requested execution provider could not be initialised, running without it",
                extra={"model": path.name, "unavailable": unavailable, "providers": bound},
            )

    def run(self, blob: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        Embed a batch of preprocessed crops.

        :param blob: The preprocessed batch, in the layout the model declares
        :return: The embeddings with shape (batch, embedding_dim)
        """
        # 'run' is typed as possibly returning a sparse tensor or a container; these graphs all
        # have a single dense output, so narrowing it here is what makes the signature honest.
        return np.asarray(self._session.run(None, {self._input: blob})[0], dtype=np.float32)
