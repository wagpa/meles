import logging
from pathlib import Path
import pandas as pd
from typing import TYPE_CHECKING

from meles.runtime import logs

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake
    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

def main(dataset_paths: list[tuple[str, Path]], output_path: Path):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if not dataset_paths:
        raise ValueError("No score files to collect; every branch of the grid is empty")

    # Concatenate both result sets, adding 'pipeline' column with either 'frame' or 'video'.
    # The two branches do not share a column set (one carries 'frame_embedding_method' and
    # 'video_aggregation_method', the other 'video_embedding_method'), so the union is ragged
    # and the missing side becomes NaN. That is intended: 'pipeline' says which columns to read.
    logger.debug("Collecting scores", extra={"files": len(dataset_paths)})
    dfs = [pd.read_parquet(p).assign(pipeline=name) for (name, p) in dataset_paths]
    collected = pd.concat(dfs, ignore_index=True)
    logger.info(
        "Collected scores",
        extra={"files": len(dataset_paths), "rows": len(collected)},
    )
    collected.to_parquet(output_path)

def named_inputs(inputs) -> list[tuple[str, Path]]:
    """
    Flatten a Snakemake input object into (group name, path) pairs, where the group name is the
    input keyword ("frame" or "video") that becomes the 'pipeline' tag.

    Iterating the input object directly yields bare path strings, not (name, paths) pairs, so it
    has to be walked by name. Two shapes have to be handled: a keyword bound to an ``expand()``
    of several files yields a list, but Snakemake collapses a keyword bound to a *single* file to
    a plain string, which would otherwise be iterated character by character.

    :param inputs: The ``snakemake.input`` object
    :return: One (name, path) pair per input file
    :raises ValueError: If any input file is not reachable through a keyword, since it would
                        otherwise be dropped from the collection without a trace
    """
    pairs: list[tuple[str, Path]] = []
    for name, value in inputs.items():
        paths = [value] if isinstance(value, str) else value
        pairs += [(name, Path(path)) for path in paths]

    # 'items()' only walks the *named* inputs, so an input added without a keyword is invisible
    # here. Comparing against the flat file count turns that silent omission into a failure:
    # a collection that quietly drops a branch reads as a complete comparison table.
    if len(pairs) != len(inputs):
        named = {str(path) for (_, path) in pairs}
        missing = sorted(str(path) for path in inputs if str(path) not in named)
        raise ValueError(
            f"{len(missing)} input file(s) are not bound to an input keyword and would be "
            f"dropped from the collection: {missing}"
        )
    return pairs

if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])

    # noinspection unbound-local-variable
    main(
        dataset_paths=named_inputs(snakemake.input),
        output_path=Path(snakemake.output.dataset),
    )
