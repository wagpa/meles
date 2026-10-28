"""
What ``Facial`` produces, per anonymization, on identities it never trained on.

The figure exists because the headline numbers invite a fair objection: a model reported at 99%
rank-1 produces images that do not look like 99% of anything. Both halves are true, and the
reason is that ArcFace reads coarse facial geometry -- which survives reduction to 28x28 and
grayscaling -- while the eye reads sharpness and texture, which the L1 term never rewarded. Show
the images beside the numbers rather than letting a reader meet them separately.

The blackbar row is included precisely because it is weak: at 19.2% the reconstructions collapse
towards a mean face, which is what 'right one time in five' looks like.

Run from the repository root::

    uv run python src/figures/facial_reconstructions.py
"""

from pathlib import Path

import cv2
import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from meles.frame_aggregation.pix2pix import to_image  # noqa: E402
from meles.training import CURRICULUM, FacialFrames, split_identities  # noqa: E402
from meles.training.facial import load_facial, reconstruct  # noqa: E402

DATA = Path("data")
CHECKPOINT = Path("models/facial/facial.voxceleb.curriculum.w2d0.inference.pt")
OUTPUT = Path("src/figures/facial_reconstructions.png")

SHOWN = 6  # identities, as columns
FRAMES = 16  # N, matching what the model was trained with
SEED = 3  # which block of validation identities to show

# Measured on the 574-identity validation gallery at epoch 120; chance is 0.17%.
RANK1 = {"none": "99.0%", "gaussianblur": "97.0%", "pixelation": "78.0%", "blackbar": "19.2%"}


def main() -> None:
    model, metadata = load_facial(CHECKPOINT, device="cpu")
    targets = pd.read_parquet(DATA / "voxceleb_targets.parquet")
    structure = pd.read_parquet(DATA / "voxceleb.parquet")

    # The same split the training notebook holds out, so these identities are genuinely unseen.
    _, validation = split_identities(targets["identity"].tolist(), validation=0.1, seed=0)
    frames = FacialFrames(
        structure,
        targets,
        DATA,
        curriculum=CURRICULUM,
        identities=sorted(validation),
        frames=FRAMES,
        stage=len(CURRICULUM) - 1,
        deterministic=True,
        all_methods=True,
    )
    # 3.3M rows, and FacialFrames keeps no reference to it.
    del structure

    width = len(frames.active)
    rows = 1 + 2 * width  # the target, then an input and a reconstruction per method
    figure, axes = plt.subplots(rows, SHOWN, figsize=(1.55 * SHOWN, 1.62 * rows), squeeze=False, layout="constrained")
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])

    for column in range(SHOWN):
        # 'all_methods' lays the set out as identity-major blocks, one entry per active method.
        index = SEED * width + column
        axes[0, column].imshow(cv2.cvtColor(to_image(frames[index * width][1]), cv2.COLOR_BGR2RGB))
        axes[0, column].set_title(frames.identities[index], fontsize=7)
        for position in range(width):
            observations, _, method = frames[index * width + position]
            produced = reconstruct(model, observations, int(method))[0]
            axes[1 + 2 * position, column].imshow(cv2.cvtColor(to_image(observations[0]), cv2.COLOR_BGR2RGB))
            axes[2 + 2 * position, column].imshow(cv2.cvtColor(to_image(produced), cv2.COLOR_BGR2RGB))
        print(f"  {column + 1}/{SHOWN}", flush=True)

    axes[0, 0].set_ylabel("TARGET", fontsize=8, color="tab:green", weight="bold")
    for position, name in enumerate(frames.active):
        axes[1 + 2 * position, 0].set_ylabel(f"{name}\ninput (1 of {FRAMES})", fontsize=7)
        axes[2 + 2 * position, 0].set_ylabel(
            f"reconstructed\nrank-1 {RANK1[name]}", fontsize=7, color="tab:red", weight="bold"
        )

    figure.suptitle(
        f"Facial on held-out VoxCeleb2 identities — {FRAMES} anonymized frames in, one face out\n"
        f"epoch {metadata['epoch']}, width 2.0, 574-identity gallery (chance 0.17%)",
        fontsize=11,
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUTPUT, dpi=110)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
