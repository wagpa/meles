import click
from pathlib import Path
import logging
import pandas as pd
from tqdm.auto import tqdm

from meles.dataset import k_fold_splits

logger = logging.getLogger(__name__)

# Allow the usage of progress bars on pandas dataframes
tqdm.pandas()


@click.command()
@click.option(
    "--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path)
)
@click.argument(
    "output_paths", required=True, nargs=-1, type=click.Path(path_type=Path)
)
def main(input_path: Path, output_paths: tuple[Path, ...]) -> None:
    # Create the output directories if they do not exist
    logger.debug("Generating output directories")
    for output_path in output_paths:
        output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input file
    logger.debug("Reading input file")
    structure = pd.read_parquet(input_path)

    # Drop any rows with NaN values (frames without embeddings)
    with_nan = len(structure)
    structure = structure.dropna()
    if len(structure) != with_nan:
        logger.warning("Dropped %d rows with NaN values", with_nan - len(structure))

    # Split into 'k' folds and save
    k = len(output_paths)
    logger.debug("Splitting into %d folds", k)
    for i, fold in enumerate(k_fold_splits(structure, k)):
        fold.to_parquet(output_paths[i], index=False)


if __name__ == "__main__":
    main()
