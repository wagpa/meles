"""
Shared fixtures.

The tests split into three groups: ones that need nothing, ones that need the model weights
(marked ``weights``), and ones that additionally need the dataset (marked ``dataset``). The
latter two skip themselves when what they need is not there, so that ``pytest`` still passes
on a machine without the 300 MiB of models or the 10 GiB of frames.
"""

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from meles.detection import YuNetDetector
from meles.frame_embedder import models

# The dataset structure the workflow runs on by default.
DATASET = Path("data/ytfaces.parquet")


@pytest.fixture(scope="session")
def model_weights() -> None:
    """Ensure every model is cached, skipping the test if it cannot be downloaded."""
    try:
        models.prefetch()
    except Exception as error:  # noqa: BLE001 - any failure means the test cannot run
        pytest.skip(f"model weights unavailable: {error}")


@pytest.fixture(scope="session")
def detector(model_weights: None) -> YuNetDetector:
    return YuNetDetector()


@pytest.fixture(scope="session")
def frames() -> pd.DataFrame:
    """A small, deterministic sample of the dataset: six identities with twelve frames each."""
    if not DATASET.is_file():
        pytest.skip(f"{DATASET} not available")
    structure = pd.read_parquet(DATASET)
    identities = sorted(structure["identity"].unique())[:6]
    return (
        structure[structure["identity"].isin(identities)].groupby("identity", sort=True).head(12).reset_index(drop=True)
    )


@pytest.fixture(scope="session")
def images(frames: pd.DataFrame) -> list[np.ndarray]:
    return [cv2.imread(str(DATASET.parent / path)) for path in frames["path"]]


@pytest.fixture(scope="session")
def crops(
    detector: YuNetDetector, images: list[np.ndarray], frames: pd.DataFrame
) -> tuple[list[np.ndarray], list[str]]:
    """The aligned crops of the sampled frames, together with their identities."""
    from meles.detection import align_face

    aligned, labels = [], []
    for image, identity in zip(images, frames["identity"]):
        area = detector.detect_best(image)
        if area is None:
            continue
        aligned.append(align_face(image, area.landmarks))
        labels.append(identity)
    return aligned, labels
