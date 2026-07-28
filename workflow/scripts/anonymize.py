import click
from pathlib import Path
import logging

import cv2
import pandas as pd
import os

from deepface import DeepFace
from tqdm import tqdm

from meles.anonymization.anonymizer import (
    Anonymizer,
    NoAnonymizer,
    GaussianBlurAnonymizer,
    BlackBarAnonymizer,
    PixelationAnonymizer,
)

logger = logging.getLogger(__name__)


def anonymize(input_path: Path, output_path: Path, anonymizer: Anonymizer) -> None:
    structure = pd.read_parquet(input_path)

    for index, sample in tqdm(
        structure.iterrows(), total=len(structure), desc="Anonymizing"
    ):
        image = cv2.imread(str(input_path.parent / sample["path"]))
        faces = DeepFace.detection.detect_faces(detector_backend="opencv", img=image)

        # Anonymize every detected region so that frames with multiple faces are
        # covered completely.
        for face in faces:
            image = anonymizer.apply(image, face.facial_area)

        # The frame is encoded in its original format to keep the dataframe
        # self-contained.
        path = (
            output_path.parent / sample["identity"] / sample["video"] / sample["frame"]
        )
        os.makedirs(path.parent, exist_ok=True)
        success = cv2.imwrite(path, image)
        if not success:
            raise ValueError(f"Failed to encode frame: {sample['path']}")
        structure.loc[index, "path"] = str(
            path.relative_to(output_path.parent)
        )  # TODO this does not update the path!
    structure.to_parquet(output_path)


@click.command()
@click.option(
    "--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--method", required=True)
def main(input_path: Path, output_path: Path, method: str) -> None:
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Embed each frame
    logger.debug("Applying anonymization %(method)", {method})
    match method:
        case "none":
            anonymize(input_path, output_path, NoAnonymizer())
        case "gaussianblur":
            anonymize(input_path, output_path, GaussianBlurAnonymizer())
        case "blackbar":
            anonymize(input_path, output_path, BlackBarAnonymizer())
        case "pixelation":
            anonymize(input_path, output_path, PixelationAnonymizer())
        case _:
            raise ValueError(f"Unknown anonymization method: {method}")


if __name__ == "__main__":
    main()
