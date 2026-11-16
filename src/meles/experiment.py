"""
Parser for the ``experiment:`` block of the workflow config.

An experiment scores a pairing of two legs over the same videos: the gallery leg is what the attacker
holds, the samples leg is what they observe. Each leg lists methods for four axes (frame anonymizer,
frame aggregator, frame recognizer, embedding aggregator). The workflow runs the full cross product of
both legs and of ``score_aggregator``.

Two conventions are easy to misread:

- a field may be a bare string instead of a list; it becomes a one-element list;
- the gallery leg inherits every field it does not set from the samples leg (after the samples leg got
  its defaults), not from the global defaults.
"""

from dataclasses import dataclass
from typing import Any

# Aggregator keys are '{scope}_{method}' (see meles.grouping.resolve_scope); the score key is
# '{gallery collapse}_{prediction aggregation}' (see meles.scoring and meles.prediction_aggregation).
# The parser does not interpret any of these strings.
LEG_DEFAULTS: dict[str, str] = {
    "frame_anonymizer": "none",
    "frame_aggregator": "video_none",
    "frame_recognizer": "arcface",
    "embedding_aggregator": "video_none",
}
SCORE_DEFAULT = "max_none"


@dataclass(frozen=True)
class ExperimentLeg:
    """The methods one leg of an experiment sweeps over, per axis."""

    frame_anonymizer: list[str]
    frame_aggregator: list[str]
    frame_recognizer: list[str]
    embedding_aggregator: list[str]


@dataclass(frozen=True)
class Experiment:
    """A named pairing of a gallery leg and a samples leg, scored with each score aggregator."""

    name: str
    samples: ExperimentLeg
    gallery: ExperimentLeg
    score_aggregator: list[str]


def as_list(value: str | list[str]) -> list[str]:
    """Wrap a bare string into a one-element list."""
    return [value] if isinstance(value, str) else value


def parse_experiment(spec: dict[str, Any]) -> Experiment:
    """
    Parse the ``experiment:`` block of the workflow config.

    :param spec: The block as loaded from the config; ``name`` is required
    :return: The experiment, with every field of both legs as a list
    """
    samples = {**LEG_DEFAULTS, **(spec.get("samples") or {})}
    # The gallery falls back to the samples leg, not to the defaults.
    gallery = {**samples, **(spec.get("gallery") or {})}

    return Experiment(
        name=spec["name"],
        samples=ExperimentLeg(**{k: as_list(v) for k, v in samples.items()}),
        gallery=ExperimentLeg(**{k: as_list(v) for k, v in gallery.items()}),
        score_aggregator=as_list(spec.get("score_aggregator") or SCORE_DEFAULT),
    )
