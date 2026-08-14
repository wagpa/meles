import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING, List

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

def main(frame_paths: List[Path], video_paths: List[Path], output_path: Path):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Concatenate both result sets, adding 'pipeline' column with either 'frame' or 'video'
    logger.debug("Collecting scores")
    frame_dfs = [pd.read_parquet(p).assign(pipeline="frame") for p in frame_paths]
    video_dfs = [pd.read_parquet(p).assign(pipeline="video") for p in video_paths]
    pd.concat(frame_dfs + video_dfs, ignore_index=True).to_parquet(output_path)

if __name__ == "__main__":
    # noinspection unbound-local-variable
    main(
        frame_paths=[Path(path) for path in snakemake.input.frame_scores],
        video_paths=[Path(path) for path in snakemake.input.video_scores],
        output_path=Path(snakemake.output.dataset),
    )
