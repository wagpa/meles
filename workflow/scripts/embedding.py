import click
from pathlib import Path
import logging
import pandas as pd
from tqdm.auto import tqdm

from meles.recognizer.arcface import ArcFaceRecognizer

logger = logging.getLogger(__name__)

# Allow the usage of progress bars on pandas dataframes
tqdm.pandas()

@click.command()
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option("--method", required=True)

def main(input_path: Path, output_path: Path, method: str) -> None:
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input file
    logger.debug("Reading input file")
    structure = pd.read_csv(input_path, sep="\t")
    paths = [str(input_path.parent / path) for path in structure["path"]]

    # Embed each frame
    logger.debug("Applying embedding %(method)", { method })
    match method:
        case "arcface":
            structure["embedding"] = ArcFaceRecognizer().embed(paths)
        case _:
            raise ValueError(f"Unknown embedding method: {method}")
    structure.to_csv(output_path, sep="\t", index=False)

if __name__ == "__main__":
    main()