import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles.runtime import logs, runtime

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

def assign_folds(df: pd.DataFrame, k: int, shuffle: bool = True, random_state: int = 42) -> pd.DataFrame:
    """Assign a fold (1 to k) to each identity-video pair.
    Folds are assigned per-identity in round-robin fashion, so that each
    identity's videos are spread as evenly as possible across all k folds.
    If an identity has >= k videos, every fold will contain at least one
    of that identity's videos.

    The result holds one row per identity-video, never one per frame: it is merged onto
    per-video and per-frame structures downstream, and a per-frame split would replicate
    every embedding once per frame of its video.
    """
    # Groups each identity-video (multiple frames)
    logger.debug("Applying k-folds to identity-videos", extra={ "k": k, "shuffle": shuffle })
    groups = df[['identity', 'video']].drop_duplicates().reset_index(drop=True)
    if shuffle:
        groups = groups.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Every identity needs at least k videos, otherwise it cannot appear in every fold.
    short = groups.groupby('identity').size().loc[lambda sizes: sizes < k]
    if not short.empty:
        raise ValueError(f"Identities have less than {k} videos: {short.index.to_list()}")

    # Round-robin within each identity over the (optionally shuffled) video order.
    groups['fold'] = groups.groupby('identity').cumcount() % k + 1
    return groups

def main(input_path: Path, output_path: Path, k: int, shuffle: bool = True):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input dataset structure, assign its folds, and write it back. The response only contains
    # the listed columns and one row per identity-video.
    structure_df = pd.read_parquet(input_path)
    structure_df = assign_folds(structure_df, k=k, shuffle=shuffle)
    structure_df[['identity', 'video', 'fold']].to_parquet(output_path)

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    # noinspection unbound-local-variable
    main(
        input_path=Path(snakemake.input.dataset),
        output_path=Path(snakemake.output.dataset),
        k=snakemake.params.get("k"),
        shuffle=snakemake.params.get("shuffle"),
    )
