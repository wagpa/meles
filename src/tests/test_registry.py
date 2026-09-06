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
# 'aggregate_embedding' is not here: its keys carry a scope ('{video|identity}_{method}'), so they
# are checked through the script's own 'resolve' rather than against the bare registry.
REGISTRIES = {
    "anonymize_frame": ("ANONYMIZERS", "frame_anonymizer"),
    "embed_frame": ("EMBEDDERS", "frame_recognizer"),
}

# Every script that has to import, whether or not the configured experiment currently reaches it.
ALL_SCRIPTS = [*REGISTRIES, "aggregate_embedding"]


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


@pytest.mark.parametrize("leg", ["samples", "gallery"])
def test_embedding_aggregators_name_a_scope_and_a_method(leg: str, experiment: Experiment) -> None:
    """Every configured aggregator has to resolve, i.e. name both a known scope and a known method."""
    resolve = load_script("aggregate_embedding").resolve

    for method in getattr(experiment, leg).embedding_aggregator:
        group_by, aggregator = resolve(method)
        assert group_by and aggregator is not None


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
    for name in ("embed_frame", "anonymize_frame"):
        sys.modules.pop(f"_workflow_{name}", None)
        load_script(name)
