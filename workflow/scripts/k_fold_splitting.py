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
@click.option("--input", "input_path", required=True, type=click.Path(exists=True, path_type=Path))
@click.option("--outputs", "output_paths", required=True, type=click.Path(path_type=Path), nargs=-1)

def main(input_path: Path, output_paths: tuple[Path, ...]) -> None:
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_paths[0].parent.mkdir(parents=True, exist_ok=True)

    # Read the input file
    logger.debug("Reading input file")
    structure = pd.read_csv(input_path, sep="\t")

    # Split into 'k' folds and save
    k = output_paths.__len__()
    logger.debug("Splitting into %(k) folds", { k })
    folds = enumerate(k_fold_splits(structure, k))
    for k, fold in folds:
        fold.to_csv(output_paths[k], sep="\t", index=False)

if __name__ == "__main__":
    main()