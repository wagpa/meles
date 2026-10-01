from dataclasses import dataclass
from typing import Any, List, Dict
from itertools import product

SCORE_DEFAULT = "max_none"
LEG_DEFAULTS = {
    "frame_anonymizer": "none",
    # Both aggregators carry their scope, as '{scope}_{method}'. The frame aggregator only has the
    # 'video' scope, since collapsing the videos of an identity into one image would leave the rules
    # after it, all keyed by identity-video, nothing to key on.
    "frame_aggregator": "video_none",
    "frame_recognizer": "arcface",
    # The default keeps every video of an identity separate and aggregates nothing within them.
    "embedding_aggregator": "video_none",
}


@dataclass(frozen=True)
class ExperimentLeg:
    """A leg of an experiment."""

    frame_anonymizer: List[str]
    frame_aggregator: List[str]
    frame_recognizer: List[str]
    embedding_aggregator: List[str]


@dataclass(frozen=True)
class Experiment:
    """A named comparison of two legs, scored with one prediction aggregator."""

    name: str
    samples: ExperimentLeg
    gallery: ExperimentLeg
    score_aggregator: List[str]


def parse_experiment(spec: dict[str, Any]) -> Experiment:
    """Parse an experiment from a workflow config specification. The spec is flattened into the set of experiments."""
    name = spec["name"] or "unknown"
    score_spec = spec["score_aggregator"] or SCORE_DEFAULT

    # Get the experiment legs, the gallery defaults to the samples. Each should be a dict of string(s).
    samples_spec = {**LEG_DEFAULTS, **spec["samples"]}
    gallery_spec = {**samples_spec, **spec["gallery"]}

    def item_list(v: Any) -> List[str]:
        return [v] if isinstance(v, str) else v
    def dict_lists(d: Dict[str, Any]) -> Dict[str, List[str]]:
        return {k: item_list(v) for k, v in d.items()}

    samples_specs = dict_lists(samples_spec)
    gallery_specs = dict_lists(gallery_spec)

    return Experiment(
        name,
        ExperimentLeg(**samples_specs),
        ExperimentLeg(**gallery_specs),
        score_aggregator=item_list(score_spec),
    )


# Flatten the experiment legs into a list of dicts. Each should only have one string value per key.
def dict_combinations(d):
    keys = d.keys()
    values = [v if isinstance(v, (list, tuple)) else [v] for v in d.values()]
    for combo in product(*values):
        yield dict(zip(keys, combo))
