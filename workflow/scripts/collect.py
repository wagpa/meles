import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING, List, Tuple

from meles import logs

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

def main(dataset_paths: List[Tuple[str, Path]], output_path: Path):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate both result sets, adding 'pipeline' column with either 'frame' or 'video'
    logger.debug("Collecting scores")
    dfs = [pd.read_parquet(p).assign(pipeline=name) for (name, p) in dataset_paths]
    pd.concat(dfs, ignore_index=True).to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    inputs: List[Tuple[str, Path]] = []
    for (name, paths) in snakemake.input:
        inputs += [(name, Path(path)) for path in paths]
    main(
        dataset_paths=inputs,
        output_path=Path(snakemake.output.dataset),
    )
