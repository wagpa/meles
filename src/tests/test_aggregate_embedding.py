"""
Tests that no aggregation scope ever pools two identities together.

The 'video' column is only a per-identity number ("0", "1", ...), so every identity has a video
"0" and the column is not a key on its own. A grouping that left the identity out would therefore
average unrelated people into one embedding -- and it would not fail, it would quietly report a
recognition accuracy for a gallery that no longer describes anybody. That is what these pin down.
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pandas as pd
import pytest

SCRIPT = Path("workflow/scripts/aggregate_embedding.py")

# One direction per identity, mutually orthogonal. Direction rather than magnitude because the
# pooling aggregators normalise before they pool, which throws magnitude away: identities that
# differed only in length would all pool to the same unit vector and the test could not tell a
# cross-identity merge from a correct one. A merge of two of these lands between them instead.
SIGNALS = {
    "alice": [1.0, 0.0, 0.0, 0.0],
    "bob": [0.0, 1.0, 0.0, 0.0],
    "carol": [0.0, 0.0, 1.0, 0.0],
}
VIDEOS = ["0", "1", "2"]
FRAMES = 4


@pytest.fixture(scope="module")
def script() -> ModuleType:
    spec = importlib.util.spec_from_file_location("_workflow_aggregate_embedding", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def dataset(tmp_path: Path) -> tuple[Path, Path]:
    """Frames and a split for three identities that all reuse the same video keys."""
    frames = pd.DataFrame(
        [
            {"identity": identity, "video": video, "frame": f"{frame}.jpg", "embedding": list(signal)}
            for identity, signal in SIGNALS.items()
            for video in VIDEOS
            for frame in range(FRAMES)
        ]
    )
    frames_path = tmp_path / "frames.parquet"
    frames.to_parquet(frames_path)

    split = frames[["identity", "video"]].drop_duplicates().reset_index(drop=True)
    split["fold"] = split.groupby("identity").cumcount() % len(VIDEOS) + 1
    split_path = tmp_path / "split.parquet"
    split.to_parquet(split_path)
    return frames_path, split_path


@pytest.mark.parametrize("scope", ["video", "identity"])
def test_every_scope_groups_by_the_identity(script: ModuleType, scope: str) -> None:
    """The identity is part of the group key of every scope, not something a scope may leave out."""
    group_by, _ = script.resolve(f"{scope}_pooling_mean")

    assert "identity" in group_by
    assert "fold" in group_by


@pytest.mark.parametrize("method", ["video_none", "video_pooling_mean", "identity_none", "identity_pooling_mean"])
def test_identities_sharing_video_keys_are_never_pooled(
    script: ModuleType, dataset: tuple[Path, Path], tmp_path: Path, method: str
) -> None:
    frames_path, split_path = dataset
    output_path = tmp_path / f"{method}.parquet"

    script.main(frames_path, split_path, output_path, method)
    aggregated = pd.read_parquet(output_path)

    # Every identity survives, and every embedding still points along its own identity's direction.
    # Pooling across identities would land strictly between two of them.
    assert set(aggregated["identity"]) == set(SIGNALS)
    for identity, rows in aggregated.groupby("identity"):
        embeddings = np.vstack(rows["embedding"].to_numpy()).astype(float)
        directions = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        assert np.allclose(directions, SIGNALS[identity]), f"{identity} was pooled with another identity"
