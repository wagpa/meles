# meles

## Getting Started

**Meles** uses `uv` as its package manager.

```shell
# Install uv on your system (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
```

```shell
# Install all dependencies
uv sync
```

```shell
# Format the code
uv run ruff format
```

## Face models

Detection and embedding run on ONNX models rather than TensorFlow, so the CUDA 13 stack the
GPU needs is the only one in the environment. The models are downloaded on first use and cached
in `models/` (override with `MELES_WEIGHTS_DIR`), and each one is checksummed:

| Key | Model | Embedding | Source |
| --- | --- | --- | --- |
| detection | YuNet `face_detection_yunet_2023mar` | — | OpenCV model zoo |
| `sface` | SFace `face_recognition_sface_2021dec` | 128-d | OpenCV model zoo |
| `arcface` | ArcFace `w600k_r50` (ResNet50 @ WebFace600K) | 512-d | InsightFace `buffalo_l` |
| `ghostfacenet` | GhostFaceNet `ghostface_fp32` | 512-d | Andes ModelZoo |

```shell
# Optional: download every model up front rather than during the first run
uv run python -c "from meles.recognizer import models; models.prefetch()"
```

Inference tries CUDA and falls back to the CPU, logging which provider actually bound. Force a
choice with `MELES_ORT_PROVIDERS=CPUExecutionProvider`.

Two caveats worth knowing when writing results up. The InsightFace weights are published for
non-commercial research use only. And GhostFaceNet has no first-party export that avoids
TensorFlow, so the model used here is a third-party conversion whose reported LFW accuracy
(99.683%) matches no published upstream checkpoint — treat it as a GhostFaceNetV1-class model
rather than as a specific citable one.

## Running the experiment

```shell
# The full grid, writing to results/ (see workflow/config.yaml)
uv run snakemake --cores 12
```

```shell
# A scaled-down run over 20 identities that exercises every rule in a few minutes.
# Build the subset once, then run it against the smoke config.
uv run python -c "
import pandas as pd
from pathlib import Path
root = Path('data'); df = pd.read_parquet(root / 'ytfaces.parquet')
counts = df.groupby('identity')['video'].nunique()
ids = counts[counts >= 3].index[:20]
sub = df[df['identity'].isin(ids)]
keep = sub.groupby('identity')['video'].unique().apply(lambda v: set(sorted(v)[:3]))
sub = sub[[v in keep[i] for i, v in zip(sub['identity'], sub['video'])]]
sub.groupby(['identity', 'video'], sort=False).head(8).reset_index(drop=True).to_parquet(
    root / 'ytfaces_smoke.parquet')
"
uv run snakemake --cores 12 --configfile workflow/config.smoke.yaml
```

Expect top-1 near the ceiling for `none` and close to chance (1/20) for `blackbar`. If `none` is
not high, alignment is the first thing to check.

```shell
# Tests. The ones needing the models or the dataset skip themselves when either is absent.
uv run pytest
```