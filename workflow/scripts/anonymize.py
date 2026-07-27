import click
from pathlib import Path
import logging
import pandas as pd
import os

logger = logging.getLogger(__name__)


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

    # Read the input file
    logger.debug("Reading input file")
    structure = pd.read_parquet(input_path)

    # Embed each frame
    logger.debug("Applying anonymization %(method)", {method})
    match method:
        case "none":
            # The 'none' anonymizer does nothing, it is used to generate a baseline comparison.
            # But it still has to ensure that the frame file paths are relative to the new output file
            structure["path"] = structure["path"].transform(
                lambda path: str(
                    Path(os.path.relpath(input_path.parent / path, output_path.parent))
                )
            )
        case _:
            raise ValueError(f"Unknown anonymization method: {method}")
    structure.to_parquet(output_path, index=False)


if __name__ == "__main__":
    main()
