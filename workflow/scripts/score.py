"""Score each k-fold of aggregated embeddings.

For every fold this computes standard face re-identification metrics by matching
each test (query) embedding against the train (gallery) embeddings using cosine
similarity (the metric ArcFace is trained for):

- mAP:      mean average precision over all queries, ranking the individual
            gallery embeddings (scikit-learn's label ranking average precision).
- accuracy: fraction of queries whose closest gallery item has the correct
            identity (equivalent to Rank-1).
- rank_1/5/10: fraction of queries whose correct identity appears within the
            top-1/5/10 ranked identities (CMC). Identities are ranked by their
            best-matching gallery embedding, so repeated embeddings of the same
            identity do not occupy multiple ranks.

The per-fold metrics are collected into a single dataframe and written to the
score output path.

The full (queries x gallery) similarity matrix does not fit in memory for
realistic fold sizes (~50k queries x ~90k gallery items is >30 GB), so queries
are streamed in chunks and the metrics are accumulated as counts/sums over
those chunks.
"""

import logging
from pathlib import Path

import click
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    label_ranking_average_precision_score,
    top_k_accuracy_score,
)
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

RANKS = (1, 5, 10)

# Rows of the embedding TSVs parsed per read, and queries scored per similarity
# block. Both bound peak memory rather than changing the result.
READ_CHUNK_ROWS = 8192


def parse_embedding(value) -> np.ndarray:
    """Parse a serialized embedding cell into a 1-D float32 array.

    Embeddings are written to TSV as the string repr of a Python list, so they
    come back as strings. Already-parsed sequences are passed through.
    """
    if isinstance(value, (list, tuple, np.ndarray)):
        return np.asarray(value, dtype=np.float32)
    return np.fromstring(value.strip().strip("[]"), sep=",", dtype=np.float32)


def load_embeddings(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a TSV into an (N, D) float32 embedding matrix and an (N,) identity array.

    The file is read in row blocks so the parsed matrix, and not the raw text of
    the whole file, dominates memory usage.
    """
    logger.debug("Reading embeddings from %s", path)
    blocks: list[np.ndarray] = []
    identities: list[np.ndarray] = []
    reader = pd.read_csv(
        path, sep="\t", usecols=["identity", "embedding"], chunksize=READ_CHUNK_ROWS
    )
    for frame in reader:
        blocks.append(np.vstack([parse_embedding(e) for e in frame["embedding"]]))
        identities.append(frame["identity"].to_numpy())
    return np.vstack(blocks), np.concatenate(identities)


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, a_min=1e-12, a_max=None)


def score_fold(
    gallery: np.ndarray,
    gallery_ids: np.ndarray,
    query: np.ndarray,
    query_ids: np.ndarray,
    ranks: tuple[int, ...] = RANKS,
    chunk_size: int = 1024,
) -> dict[str, float]:
    """Compute re-identification metrics for a single fold.

    gallery/query are (N, D) embedding matrices; *_ids are the matching
    identity labels. Similarity is cosine, so both sides are L2-normalized and
    scored by dot product.
    """
    gallery = _l2_normalize(gallery)
    query = _l2_normalize(query)

    # Group the gallery by identity so the per-identity best similarity is a
    # single segmented maximum over contiguous columns.
    grouped = np.argsort(gallery_ids, kind="stable")
    gallery = gallery[grouped]
    gallery_ids = gallery_ids[grouped]
    identities, identity_starts = np.unique(gallery_ids, return_index=True)

    n_queries = len(query_ids)
    rank_hits = dict.fromkeys(ranks, 0)
    top1_hits = 0
    average_precision_sum = 0.0
    scored_queries = 0

    for start in tqdm(range(0, n_queries, chunk_size), desc="Scoring queries"):
        stop = min(start + chunk_size, n_queries)
        chunk_ids = query_ids[start:stop]

        # (chunk, G) cosine similarities against every gallery embedding.
        similarities = query[start:stop] @ gallery.T
        relevant = gallery_ids[None, :] == chunk_ids[:, None]  # (chunk, G) bool

        # mAP over the gallery embeddings, skipping queries whose identity is
        # absent from the gallery (undefined average precision).
        has_relevant = relevant.any(axis=1)
        if has_relevant.any():
            average_precision_sum += label_ranking_average_precision_score(
                relevant[has_relevant], similarities[has_relevant]
            ) * int(has_relevant.sum())
            scored_queries += int(has_relevant.sum())

        # Collapse to (chunk, n_identities) by taking each identity's best
        # matching embedding, which is the ranking the CMC curve is defined on.
        identity_scores = np.maximum.reduceat(similarities, identity_starts, axis=1)
        del similarities, relevant

        top1_hits += accuracy_score(
            chunk_ids, identities[identity_scores.argmax(axis=1)], normalize=False
        )

        # top_k_accuracy_score requires every true label to be a known class, so
        # queries whose identity is missing from the gallery are counted as
        # misses without being passed in.
        known = np.isin(chunk_ids, identities)
        for k in ranks:
            # Rank-1 is the top-1 identity, i.e. the accuracy counted above.
            if k == 1:
                continue
            if not known.any():
                continue
            if k >= len(identities):
                rank_hits[k] += int(known.sum())
                continue
            rank_hits[k] += round(
                top_k_accuracy_score(
                    chunk_ids[known],
                    identity_scores[known],
                    k=k,
                    labels=identities,
                    normalize=False,
                )
            )

    if 1 in rank_hits:
        rank_hits[1] = top1_hits

    metrics = {
        "mAP": average_precision_sum / scored_queries if scored_queries else 0.0,
        "accuracy": top1_hits / n_queries if n_queries else 0.0,
    }
    metrics.update(
        {f"rank_{k}": rank_hits[k] / n_queries if n_queries else 0.0 for k in ranks}
    )
    return metrics


@click.command()
@click.option(
    "--train-set",
    "train_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Aggregated train (gallery) embeddings for one fold; repeat per fold.",
)
@click.option(
    "--test-set",
    "test_paths",
    required=True,
    multiple=True,
    type=click.Path(exists=True, path_type=Path),
    help="Aggregated test (query) embeddings for one fold; repeat per fold.",
)
@click.option("--output", "output_path", required=True, type=click.Path(path_type=Path))
@click.option(
    "--chunk-size",
    default=1024,
    show_default=True,
    type=click.IntRange(min=1),
    help="Number of queries scored per similarity block; lower it to use less memory.",
)
def main(
    train_paths: tuple[Path, ...],
    test_paths: tuple[Path, ...],
    output_path: Path,
    chunk_size: int,
) -> None:
    if len(train_paths) != len(test_paths):
        raise click.BadParameter(
            f"Got {len(train_paths)} train sets but {len(test_paths)} test sets; "
            "each fold needs both."
        )

    # Create the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = []
    for fold, (train_path, test_path) in enumerate(zip(train_paths, test_paths)):
        logger.debug("Scoring fold %d", fold)
        gallery, gallery_ids = load_embeddings(train_path)
        query, query_ids = load_embeddings(test_path)
        metrics = score_fold(
            gallery, gallery_ids, query, query_ids, chunk_size=chunk_size
        )
        del gallery, gallery_ids, query, query_ids
        rows.append({"k": fold, **metrics})

    columns = ["k", "mAP", "accuracy", *[f"rank_{k}" for k in RANKS]]
    scores = pd.DataFrame(rows, columns=columns)
    scores.to_csv(output_path, sep="\t", index=False)


if __name__ == "__main__":
    main()
