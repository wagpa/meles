import logging
from pathlib import Path
import pandas as pd
import cv2
import os
from tqdm import tqdm
from typing import TYPE_CHECKING

from meles.runtime import logs, runtime
from meles import anonymization
from meles.detection import default_detector

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# Shared with 'meles.training', which anonymizes clear images to build the pix2pix training pairs
# and has to apply the very same anonymization this rule applies. Adding a method means adding it
# to 'meles.anonymization.ANONYMIZERS' and to the leg field in 'workflow/config.yaml'.
ANONYMIZERS = anonymization.ANONYMIZERS


def main(input_path: Path, output_path: Path, method: str):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get the anonymizer based on the provided method key.
    anonymizer = ANONYMIZERS.get(method)
    if anonymizer is None:
        raise ValueError(f"Unknown anonymization method: {method}")

    # Read the input dataset structure and pass it to the anonymizer
    structure_df = pd.read_parquet(input_path)
    # The bar used to advance once per frame, which wrote megabytes of carriage returns into
    # the job log; 'mininterval' keeps the live feedback without the noise.
    structure_df_rows = tqdm(structure_df.iterrows(), total=len(structure_df), desc="Anonymizing", mininterval=5.0)
    logger.debug("Anonymizing frames", extra={"method": method, "len": len(structure_df)})
    # Counted rather than logged per frame: a warning on every one of the 130k frames drowns
    # the log, but the totals are what actually tell you whether detection is working.
    missing, multiple = 0, 0
    for index, sample in structure_df_rows:
        # Read each image given the relative image path. The path is always relative to the input file (parent) path.
        image = cv2.imread(str(input_path.parent / sample["path"]))
        if image is None:
            raise ValueError(f"Image not found: {sample['path']}")

        # Detect all faces in the image. It should always be one as the images have been pre-processed as such.
        # However, we don't throw an error and only count it.
        faces = default_detector.detect(image)
        if len(faces) == 0:
            missing += 1
        elif len(faces) > 1:
            multiple += 1
        for face in faces:
            image = anonymizer.apply(image, face)

        # Write the anonymized image back into the output directory.
        image_output_path = output_path.parent / sample["identity"] / sample["video"] / sample["frame"]
        os.makedirs(image_output_path.parent, exist_ok=True)
        success = cv2.imwrite(str(image_output_path), image)
        if not success:
            raise ValueError(f"Failed to write anonymized image: {sample['path']}")
        structure_df.loc[index, "path"] = str(image_output_path.relative_to(output_path.parent))

    # A frame with no detection is left completely un-anonymized, so a high miss rate silently
    # weakens the very thing the experiment measures. Surface it loudly rather than let it show
    # up as unexpectedly good recognition accuracy three rules later.
    logger.info(
        "Anonymized frames",
        extra={
            "method": method,
            "len": len(structure_df),
            "no_face": missing,
            "several_faces": multiple,
        },
    )
    if missing > 0.02 * len(structure_df):
        logger.error(
            "Many frames had no detectable face and were left unanonymized",
            extra={
                "no_face": missing,
                "len": len(structure_df),
            },
        )

    # Write the full dataset back. It now contains updated path for each frame.
    structure_df.to_parquet(output_path)


if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("frame_anonymization"),
    )
