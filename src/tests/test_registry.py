"""
Tests that the config and the registries in the workflow scripts agree.

Commit 73ef67e renamed the embedder class but only updated one of its two call sites, leaving
``workflow/scripts/embed_video.py`` with an import of a name that no longer existed. Nothing
noticed, because the config disables that branch, so the broken module is never imported by a
run. Importing every script here is what makes that class of breakage impossible to miss.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

SCRIPTS = Path("workflow/scripts")
CONFIG = Path("workflow/config.yaml")

# Each workflow script's registry, and the config key whose entries it must cover.
REGISTRIES = {
    "anonymize_frame": ("ANONYMIZERS", "anonymization_methods"),
    "embed_frame": ("EMBEDDERS", "frame_embedding_methods"),
    "embed_video": ("EMBEDDERS", "video_embedding_methods"),
    "aggregate_video": ("AGGREGATORS", "video_aggregation_methods"),
    "aggregate_identity": ("AGGREGATORS", "identity_aggregation_methods"),
}


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
def config() -> dict:
    return yaml.safe_load(CONFIG.read_text())


@pytest.mark.parametrize("script", sorted(REGISTRIES))
def test_script_imports(script: str) -> None:
    """Every script must import, whether or not the config currently enables its branch."""
    assert load_script(script) is not None


@pytest.mark.parametrize("script", sorted(REGISTRIES))
def test_config_keys_have_a_registry_entry(script: str, config: dict) -> None:
    attribute, config_key = REGISTRIES[script]
    registry = getattr(load_script(script), attribute)

    missing = [key for key in config[config_key] if key not in registry]
    assert not missing, f"{config_key} in {CONFIG} lists {missing}, which {script}.{attribute} has no entry for"


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
    for name in ("embed_frame", "embed_video", "anonymize_frame"):
        sys.modules.pop(f"_workflow_{name}", None)
        load_script(name)
