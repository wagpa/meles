"""
``FacialFrameAggregator`` is the trained model inside the sweep, and the ways it can go wrong are
quiet ones: weights read at import time (which would make every job load a model it may not need),
a group smaller than N silently shrinking the input, or the anonymization conditioning coming from
the registry key rather than from the job's wildcard.

No checkpoint is read here. Everything below is either construction, the draw, or a refusal.
"""

from pathlib import Path

import numpy as np
import pytest

from meles.frame_aggregation.aggregator import FrameAggregator
from meles.frame_aggregation.facial import (
    CHECKPOINT_STEM,
    INFERENCE_FRAMES,
    FacialFrameAggregator,
    facial_checkpoint_path,
)

FRAMES = [np.full((64, 64, 3), index, dtype=np.uint8) for index in range(40)]


def test_it_is_a_frame_aggregator():
    assert isinstance(FacialFrameAggregator("pixelation"), FrameAggregator)


def test_construction_reads_no_checkpoint(tmp_path, monkeypatch):
    """
    'resolve' builds this once per job, before the job knows whether its output is even stale,
    and the model is a few hundred megabytes.
    """

    def refuse(*args, **kwargs):
        raise AssertionError("the checkpoint was read during construction")

    monkeypatch.setattr("meles.training.facial.load_facial", refuse)
    aggregator = FacialFrameAggregator("pixelation", directory=tmp_path)
    assert aggregator._model is None


def test_the_inference_export_is_preferred(tmp_path):
    """Identical weights, a third of the bytes -- the optimiser is dead weight at inference."""
    full = tmp_path / f"{CHECKPOINT_STEM}.pt"
    export = tmp_path / f"{CHECKPOINT_STEM}.inference.pt"
    full.write_bytes(b"")
    assert facial_checkpoint_path(tmp_path) == full, "the full checkpoint is the fallback"
    export.write_bytes(b"")
    assert facial_checkpoint_path(tmp_path) == export, "the export wins where both exist"


def test_a_missing_checkpoint_says_where_to_put_it(tmp_path):
    with pytest.raises(FileNotFoundError, match="No trained Facial model"):
        FacialFrameAggregator("pixelation", directory=tmp_path).path()


def test_a_short_group_repeats_rather_than_shrinking():
    """
    The model takes a fixed N. A group holding fewer frames is a truthful item -- the attacker
    simply saw less -- so it repeats instead of handing the model a shape it cannot take.
    """
    aggregator = FacialFrameAggregator("none")
    assert len(aggregator.draw(FRAMES[:3])) == INFERENCE_FRAMES
    assert len(aggregator.draw(FRAMES)) == INFERENCE_FRAMES


def test_the_draw_is_reproducible():
    """A rerun has to aggregate the same frames, or the sweep is not deterministic."""
    first = FacialFrameAggregator("none").draw(FRAMES)
    second = FacialFrameAggregator("none").draw(FRAMES)
    assert [id(frame) for frame in first] == [id(frame) for frame in second]


def test_an_empty_group_produces_nothing():
    """'aggregate_frame' counts these as lost groups rather than crashing on them."""
    assert FacialFrameAggregator("none").aggregate([]) == []


def test_the_registry_entry_has_no_frame_prefix():
    """
    The 'frame_*' prefix means frame-wise, i.e. as many frames out as in. This one collapses a
    group to a single reconstruction, so prefixing it would make 'aggregate_frame' keep the
    original frame names for a result that has no original frame.
    """
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("_af", Path("workflow/scripts/aggregate_frame.py"))
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert "facial" in module.TARGETED_AGGREGATORS
    assert not any(key.startswith("frame_") and "facial" in key for key in module.TARGETED_AGGREGATORS)
    _, aggregator = module.resolve("video_facial", "blackbar")
    assert isinstance(aggregator, FacialFrameAggregator)
    assert aggregator.anonymizer == "blackbar"
