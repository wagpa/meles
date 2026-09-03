"""
Tests for parsing the experiment spec out of the workflow config.

The spec is deliberately forgiving -- a leg may name a method as a bare string, and the gallery
leg inherits whatever it does not override from the samples leg -- so every field comes out of
``parse_experiment`` normalised into a list of method names. Those lists are what the Snakefile
expands the grid over, so a leg that silently parses into the wrong thing does not fail the run,
it just runs the wrong experiment.
"""

from meles.experiment import parse_experiment

SPEC = {
    "name": "clear_gallery",
    "samples": {
        "frame_anonymizer": ["blackbar", "pixelation"],
        "frame_recognizer": "arcface",
        "frame_aggregator": "video_none",
    },
    "gallery": {
        "frame_anonymizer": ["none"],
        "frame_aggregator": ["video_none", "identity_pooling_mean"],
    },
}


def test_every_field_becomes_a_list_of_methods() -> None:
    """A bare string names one method and has to parse as a one-element list, not as its characters."""
    experiment = parse_experiment(SPEC)

    assert experiment.name == "clear_gallery"
    assert experiment.samples.frame_anonymizer == ["blackbar", "pixelation"]
    assert experiment.samples.frame_recognizer == ["arcface"]
    assert experiment.samples.frame_aggregator == ["video_none"]


def test_the_gallery_inherits_what_it_does_not_override() -> None:
    """The gallery leg names only what differs; the rest is the samples leg, not the global default."""
    experiment = parse_experiment(SPEC)

    assert experiment.gallery.frame_recognizer == ["arcface"]
    assert experiment.gallery.frame_anonymizer == ["none"]
    assert experiment.gallery.frame_aggregator == ["video_none", "identity_pooling_mean"]


def test_omitted_fields_fall_back_to_the_defaults() -> None:
    experiment = parse_experiment({"name": "minimal", "samples": {}, "gallery": {}})

    assert experiment.samples.frame_anonymizer == ["none"]
    assert experiment.samples.frame_recognizer == ["arcface"]
    assert experiment.gallery.frame_aggregator == ["video_none"]
