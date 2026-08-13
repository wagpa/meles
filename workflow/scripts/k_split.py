import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

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
    """
    def assign_within_identity(g: pd.DataFrame) -> pd.DataFrame:
        identity = g["identity"].iat[0]
        logger.debug("Applying k-folds to identity", extra={ "identity": identity })
        if shuffle:
            g = g.sample(frac=1, random_state=random_state)
        g = g.reset_index(drop=True)
        if len(g) < k:
            raise ValueError(f"Identity {identity} has less than {k} videos")
        g['fold'] = (g.index % k) + 1
        return g

    # Groups each identity-video (multiple frames)
    logger.debug("Applying k-folds to identity-videos", extra={ "k": k, "shuffle": shuffle })
    groups = (
        df[['identity', 'video']]
        .drop_duplicates()
        .reset_index(drop=True)
        .groupby('identity', group_keys=False)
        .apply(assign_within_identity)
        .reset_index(drop=True)
    )

    # Bring fold back onto the original dataframe
    df = df.merge(groups, on=['identity', 'video'], how='left')
    return df

def main(input_path: Path, output_path: Path, k: int, shuffle: bool = True):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the input dataset structure, assign its folds, and write it back. The response only contains
    # the listed columns.
    structure_df = pd.read_parquet(input_path)
    structure_df = assign_folds(structure_df, k=k, shuffle=shuffle)
    structure_df[['identity', 'video', 'fold']].to_parquet(output_path)

if __name__ == "__main__":
    # noinspection unbound-local-variable
    main(
        input_path=snakemake.input.dataset,
        output_path=snakemake.output.dataset,
        k=snakemake.params.get("k"),
        shuffle=snakemake.params.get("shuffle"),
    )
