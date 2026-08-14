import logging
from pathlib import Path
import pandas as pd
import cv2
import os
from deepface import DeepFace
from tqdm import tqdm
from typing import TYPE_CHECKING

from meles import logs
from meles.anonymization.anonymizer import NoAnonymizer, GaussianBlurAnonymizer, BlackBarAnonymizer, \
    PixelationAnonymizer

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

ANONYMIZERS = {
    "none": NoAnonymizer(),
    "gaussianblur": GaussianBlurAnonymizer(),
    "blackbar": BlackBarAnonymizer(),
    "pixelation": PixelationAnonymizer(),
}

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
    structure_df_rows = tqdm(structure_df.iterrows(), total=len(structure_df), desc="Anonymizing")
    logger.debug("Anonymizing frames", extra={ "method": method, "len": len(structure_df) })
    for index, sample in structure_df_rows:
        # Read each image given the relative image path. The path is always relative to the input file (parent) path.
        logger.debug("Reading image from path", extra={ "path": sample["path"] })
        image = cv2.imread(str(input_path.parent / sample["path"]))
        if image is None:
            raise ValueError(f"Image not found: {sample["path"]}")

        # Detect all faces in the image. It should always be one as the images have been pre-processed as such.
        # However, we don't throw an error and only log a warning.
        logger.debug("Detecting facial regions")
        faces = DeepFace.detection.detect_faces(detector_backend="opencv", img=image)
        if len(faces) != 1:
            logger.warning("Received unexpected number of faces", extra={
                "len": len(faces),
                "path": sample["path"]
            })
        logger.debug("Anonymizing facial regions")
        for face in faces:
            image = anonymizer.apply(image, face.facial_area)

        # Write the anonymized image back into the output directory.
        image_output_path = output_path.parent / sample["identity"] / sample["video"] / sample["frame"]
        os.makedirs(image_output_path.parent, exist_ok=True)
        success = cv2.imwrite(image_output_path, image)
        if not success:
            raise ValueError(f"Failed to write anonymized image: {sample['path']}")
        structure_df.loc[index, "path"] = str(image_output_path.relative_to(output_path.parent))

    # Write the full dataset back. It now contains updated path for each frame.
    structure_df.to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("frame_anonymization"),
    )
