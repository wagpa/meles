"""
Tests that the config and the registries in the workflow scripts agree.

A rename once updated only one of a registry's two call sites, leaving a workflow script with an
import of a name that no longer existed. Nothing noticed, because the config had that branch
disabled, so the broken module was never imported by a run. Importing every script here is what
makes that class of breakage impossible to miss, whether or not a run currently reaches it.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

from meles.experiment import Experiment, parse_experiment

SCRIPTS = Path("workflow/scripts")
CONFIG = Path("workflow/config.yaml")

# Each workflow script's registry, and the experiment-leg field naming the methods it must cover.
# The two aggregation scripts are not here: their keys carry a scope ('{scope}_{method}'), so they
# are checked through each script's own 'resolve' rather than against the bare registry.
REGISTRIES = {
    "anonymize_frame": ("ANONYMIZERS", "frame_anonymizer"),
    "embed_frame": ("EMBEDDERS", "frame_recognizer"),
}

# Each aggregation script, the experiment-leg field whose scoped keys its 'resolve' must accept, and
# the field 'resolve' takes as its second argument, if it takes one. 'aggregate_frame' resolves
# against the anonymization as well, because a learned de-anonymization is trained against one
# method; 'aggregate_embedding' is a stage past any of that and resolves on the key alone.
SCOPED_REGISTRIES = {
    "aggregate_frame": ("frame_aggregator", "frame_anonymizer"),
    "aggregate_embedding": ("embedding_aggregator", None),
}

# Every script that has to import, whether or not the configured experiment currently reaches it.
ALL_SCRIPTS = [*REGISTRIES, *SCOPED_REGISTRIES]


def load_script(name: str) -> ModuleType:
    """
    Import a workflow script as a module.

    The scripts reference a ``snakemake`` global that only exists inside a job, but only under
    ``if __name__ == "__main__"``, so importing them under any other name is safe.

    :param name: The script's module name, without the '.py'
    :return: The imported module
    """
    spec = importlib.util.spec_from_file_location(f"_workflow_{name}", SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def experiment() -> Experiment:
    return parse_experiment(yaml.safe_load(CONFIG.read_text())["experiment"])


@pytest.mark.parametrize("script", sorted(ALL_SCRIPTS))
def test_script_imports(script: str) -> None:
    """Every script must import, whether or not the configured experiment currently reaches it."""
    assert load_script(script) is not None


@pytest.mark.parametrize("script", sorted(REGISTRIES))
@pytest.mark.parametrize("leg", ["samples", "gallery"])
def test_experiment_methods_have_a_registry_entry(script: str, leg: str, experiment: Experiment) -> None:
    attribute, field = REGISTRIES[script]
    registry = getattr(load_script(script), attribute)

    methods = getattr(getattr(experiment, leg), field)
    missing = [method for method in methods if method not in registry]
    assert not missing, (
        f"experiment.{leg}.{field} in {CONFIG} lists {missing}, which {script}.{attribute} has no entry for"
    )


@pytest.mark.parametrize("script", sorted(SCOPED_REGISTRIES))
@pytest.mark.parametrize("leg", ["samples", "gallery"])
def test_aggregators_name_a_scope_and_a_method(script: str, leg: str, experiment: Experiment) -> None:
    """
    Every configured aggregator has to resolve, i.e. name both a known scope and a known method.

    For 'aggregate_frame' that means against every anonymization its leg names, not just one of
    them: the workflow expands the leg as a product, so a leg pairing a method-specific
    de-anonymization with an anonymization it has no model for is a job that fails at run time.
    """
    resolve = load_script(script).resolve
    field, against = SCOPED_REGISTRIES[script]
    configured = getattr(experiment, leg)

    for method in getattr(configured, field):
        arguments = (
            [(method, anonymization) for anonymization in getattr(configured, against)] if against else [(method,)]
        )
        for argument in arguments:
            group_by, aggregator = resolve(*argument)
            assert group_by and aggregator is not None


def test_the_frame_aggregator_refuses_the_identity_scope() -> None:
    """
    Aggregating pixels across the videos of an identity leaves no video to name, and every rule
    after 'aggregate_frame' is keyed by identity-video. Refusing the scope is what keeps that from
    turning into a plausible-looking number instead of an error.
    """
    resolve = load_script("aggregate_frame").resolve

    with pytest.raises(ValueError, match="identity"):
        resolve("identity_none")


def test_embedder_registries_cover_every_model() -> None:
    """The frame embedder registry must expose every model 'meles.recognizer.models' defines."""
    from meles.recognizer import models

    registry = getattr(load_script("embed_frame"), "EMBEDDERS")
    assert set(registry) == set(models.SPECS)


def test_building_the_registry_does_not_load_any_weights(monkeypatch) -> None:
    """
    The registries instantiate every entry at import time, so construction has to stay free.
    If a constructor ever resolved its weights, each job would download and load all three
    models to use one.
    """
    from meles import weights

    def refuse(spec: weights.WeightSpec) -> Path:
        raise AssertionError(f"weights for {spec.file_name} were resolved at import time")

    monkeypatch.setattr(weights, "resolve", refuse)
    for name in ALL_SCRIPTS:
        sys.modules.pop(f"_workflow_{name}", None)
        load_script(name)


def test_resolving_a_trained_aggregator_loads_no_checkpoint(monkeypatch) -> None:
    """
    The same rule one level down, for the aggregators that are trained rather than downloaded.

    A trained de-anonymization is constructed by 'resolve' rather than by the registry, so it is
    built once per job instead of once per import -- but the model behind it is a few hundred
    megabytes, and construction still has to be free: a job resolves before it knows whether the
    output is even stale.
    """
    from meles.frame_aggregation import pix2pix

    def refuse(*args, **kwargs):
        raise AssertionError("the generator was loaded before a frame was aggregated")

    monkeypatch.setattr(pix2pix, "load_generator", refuse)
    _, aggregator = load_script("aggregate_frame").resolve("video_frame_pix2pix", "pixelation")
    assert isinstance(aggregator, pix2pix.Pix2PixFrameAggregator)


def test_a_trained_aggregator_refuses_to_resolve_without_an_anonymization() -> None:
    """
    One model is trained per anonymization method, so a key naming one is only half a resolution.
    Refusing is what keeps a missing wildcard from silently picking a model for the wrong frames.
    """
    resolve = load_script("aggregate_frame").resolve

    with pytest.raises(ValueError, match="trained against one anonymization"):
        resolve("video_frame_pix2pix")
