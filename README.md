# meles

Code for the master thesis *Extending Image Recognition Attacks by Embedding Aggregation over Time*.

Anonymizing a face in a single image (blurring, pixelating or covering it) can make it hard to
recognize. A video, however, shows the same person many times, under different poses, lighting and
days. This project asks whether an attacker who **aggregates many anonymized observations** of a
person can recover an identity that no single anonymized frame reveals.

The work has two parts:

1. **Preparation and the aggregation model** (this repository today): the dataset, face detection,
   the anonymizations, and *AggFace*, a model that turns several anonymized frames of one person into
   a single clear face.
2. **The recognition experiment** (in progress): a sweep over anonymization, aggregation method and
   recognition model, scoring how well each combination re-identifies people it has never seen.

## Repository layout

| Path | Contents |
| --- | --- |
| `src/meles` | Library: detection, alignment, anonymization, recognition, aggregation and scoring |
| `src/aggface` | The AggFace model, its training data, losses and training loop |
| `notebooks` | The steps of the thesis in order, each one documenting what it does and why |
| `images` | Figures exported by the notebooks |
| `data` | Datasets and derived tables (not tracked) |
| `models` | Downloaded weights and trained checkpoints (not tracked) |

## Getting started

The project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) to manage Python and
its dependencies.

```shell
# Install uv (macOS and Linux; see the link above for Windows)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Start Jupyter to run the notebooks
uv run jupyter lab
```

A CUDA-capable GPU is strongly recommended for training AggFace. Inference falls back to the CPU
when no GPU is available.

## Notebooks

Run them in order; each one reads the tables the previous ones wrote to `data/`.

| Notebook | What it does |
| --- | --- |
| `00_setup_verification` | Project introduction and environment setup |
| `01_dataset_download` | Downloads VoxCeleb2, inventories its clips and extracts one frame per second |
| `02_dataset_split` | Splits the identities into train (85%), validation (5%) and test (10%) with no person in two parts |
| `03_detection` | Detects and aligns the face in every frame, and checks where detection fails |
| `04_anonymization` | Defines Gaussian blur, pixelation and black bar anonymization and shows their effect |
| `05_target_selection` | Picks one clear, frontal target image per identity for training AggFace |
| `06_aggface_training` | Trains AggFace over a curriculum of anonymizations |
| `07_observation_count` | Measures how recognition changes with the number of observations given to AggFace |
| `08_observation_diversity` | Compares observations from one clip, one video and several videos |
| `09_exp_anonymizers_aggface` | Reads the experiment `anonymizers_aggface` that compares aggface against prediction aggregation. |

## Dataset

The experiments use [VoxCeleb2](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox2.html). Its
original distributors no longer provide the videos, so `01_dataset_download` fetches them from the
[`ProgramComputer/voxceleb`](https://huggingface.co/datasets/ProgramComputer/voxceleb) mirror on
Hugging Face. Accept the dataset conditions there and log in with `uv run hf auth login` (or set
`HF_TOKEN`) before running the notebook. The download is about 267 GB.

## Models

Detection and recognition run on ONNX models through ONNX Runtime. They are downloaded on first use,
checksummed, and cached in `models/`:

| Model | Used for | Embedding | Source |
| --- | --- | --- | --- |
| YuNet `face_detection_yunet_2023mar` | detection and landmarks | | OpenCV model zoo |
| SFace `face_recognition_sface_2021dec` | recognition | 128-d | OpenCV model zoo |
| ArcFace `w600k_r50` (ResNet50, WebFace600K) | recognition | 512-d | InsightFace `buffalo_l` |
| GhostFaceNet `ghostface_fp32` | recognition | 512-d | Andes ModelZoo |

```shell
# Optional: download every model up front instead of during the first run
uv run python -c "from meles import recognition; recognition.prefetch()"
```

AggFace's encoder starts from an `iresnet34` backbone of the InsightFace
[arcface_torch model zoo](https://github.com/deepinsight/insightface/tree/master/model_zoo). It is
distributed without a stable URL, so it has to be downloaded by hand to `models/iresnet34.pth`
(`06_aggface_training` links to it). Trained AggFace checkpoints are written to `models/aggface/`.

Two caveats when citing results:

- The InsightFace weights are published for non-commercial research use only.
- GhostFaceNet has no first-party export that avoids TensorFlow. The model used here is a
  third-party conversion whose reported LFW accuracy (99.683%) matches no published upstream
  checkpoint, so treat it as a GhostFaceNetV1-class model rather than a specific citable one.

### Environment variables

| Variable | Effect |
| --- | --- |
| `MELES_WEIGHTS_DIR` | Where weights are cached (default `models/`) |
| `MELES_ORT_PROVIDERS` | ONNX Runtime providers to try, comma separated, e.g. `CPUExecutionProvider` to force the CPU |

## Development

```shell
uv run ruff format && uv run ruff check   # formatting and linting
uv run ty check                           # type checking
```
