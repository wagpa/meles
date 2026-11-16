"""
Scoring one pairing of the experiment: a gallery leg against a samples leg.

The gallery is what the attacker holds; the samples are what they get to observe. The algorithm is
a dense, two-axis reduction of one cosine similarity matrix, and reading it as two collapses on
two different axes is the whole thing::

    S = samples @ gallery.T                     (observations, gallery rows)

    1. collapse the GALLERY axis  -> (observations, identities)
       How do the several gallery rows of one identity become one score for that identity?
    2. collapse the SAMPLES axis  -> (queries, identities)
       How do the several observations of one query video become one decision?

The first collapse must be **invariant to how many rows an identity owns**, or it confounds the
gallery's aggregation with the score: a gallery of per-frame embeddings and a gallery of pooled
ones would not be comparable. ``max`` has that property and ``sum`` does not, which is why only
the former is registered. A weaker bias does remain and is worth remembering when reading results:
the maximum of many draws is stochastically larger than the maximum of few, growing roughly like
``sqrt(2 log n)``. ``max`` removes the linear dependence on row count, not the extreme-value one,
so a clean comparison still holds the rows per identity roughly equal across conditions.

The second collapse has no such requirement. More observations are *supposed* to help - that is
the hypothesis the experiment exists to test - and the choices live in
:mod:`meles.prediction_aggregation`.

Scores stay raw cosine similarities and are never renormalised. After the gallery collapse every
row is a dense similarity profile over all identities, already on one bounded scale, so there is
no scale to correct for. A query that happens to sit close to everything lifts its whole row by
roughly a constant, and adding a constant changes neither its argmax nor the argmax of a sum of
such rows, so hub observations cancel on their own.

Output is **one row per query video**, carrying the fraction of its observations that were correct
at each k. That single property removes a class of weighting bugs: a mean over any subset of rows
is the correctly weighted macro-average, because each video counts once however many frames it
had.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from meles.embedding_aggregation import normalized
from meles.prediction_aggregation import PredictionAggregator

# The ranks reported. A query is correct at k when its true identity is among the k highest
# scoring identities, capped below the number of identities the gallery can offer.
RANKS = (1, 5, 10)

# How many bytes one chunk of the similarity matrix may occupy. The matrix is never materialised:
# a chunk of query rows is multiplied out, collapsed onto the identity axis straight away and
# thrown away, so the gallery axis only ever exists one chunk at a time.
CHUNK_BYTES = 512 * 1024**2

# How the several gallery rows of one identity become that identity's score. Every entry has to be
# invariant to the number of rows an identity owns; 'sum' is deliberately absent.
GALLERY_COLLAPSES: dict[str, np.ufunc] = {
    "max": np.maximum,
}


def resolve_gallery(method: str) -> np.ufunc:
    """
    The gallery collapse a configuration key names.

    :param method: The key, for example 'max'
    :return: The reduction to apply across one identity's gallery rows
    :raises ValueError: If nothing is registered under that key
    """
    collapse = GALLERY_COLLAPSES.get(method)
    if collapse is None:
        raise ValueError(f"Unknown gallery collapse {method!r}, expected one of {sorted(GALLERY_COLLAPSES)}")
    return collapse


@dataclass(frozen=True)
class Leg:
    """
    One side of a pairing: unit-norm embeddings and the keys they belong to, sorted by key.

    :param embeddings: Unit-norm embeddings, shape (n, dim)
    :param identity: Whose face each row is, shape (n,)
    :param video: Which video each row came from, shape (n,)
    """

    embeddings: NDArray[np.float32]
    identity: NDArray
    video: NDArray

    @classmethod
    def from_frame(cls, frame: pd.DataFrame, embeddings: NDArray[np.float32]) -> "Leg":
        """
        Read a leg out of a structure and its embeddings, sorting so that both collapses are runs of equal keys.

        Sorting here rather than relying on the upstream order keeps the contract local: both
        collapses reduce over contiguous runs, which is what makes them one vectorised call each.

        :param frame: A structure with 'identity' and 'video' columns
        :param embeddings: The structure's embeddings, shape (len(frame), dim), aligned by row order
        :return: The leg, sorted by identity then video
        """
        # Positional index, so the sorted order can be applied to the embeddings.
        ordered = frame.reset_index(drop=True).sort_values(["identity", "video"], kind="stable")
        return cls(
            embeddings=normalized(np.asarray(embeddings)[ordered.index.to_numpy()]),
            identity=ordered["identity"].to_numpy(),
            video=ordered["video"].to_numpy(),
        )


def group_starts(keys: NDArray) -> NDArray[np.intp]:
    """
    Where each run of equal keys begins, for keys that are already sorted.

    These are exactly the indices ``ufunc.reduceat`` wants: it reduces ``[starts[i], starts[i+1])``
    for each i, and the final group to the end of the axis.

    :param keys: A sorted array of shape (n,) or (n, columns)
    :return: The strictly increasing start index of every group
    """
    keys = np.atleast_2d(keys.T).T  # (n,) -> (n, 1); (n, k) stays (n, k)
    changed = np.ones(len(keys), dtype=bool)
    changed[1:] = (keys[1:] != keys[:-1]).any(axis=1)
    return np.flatnonzero(changed)


def identity_scores(
    samples: NDArray[np.float32],
    gallery: NDArray[np.float32],
    gallery_starts: NDArray[np.intp],
    collapse: np.ufunc,
    chunk_bytes: int = CHUNK_BYTES,
) -> NDArray[np.float32]:
    """
    The similarity matrix, collapsed onto the identity axis as it is produced.

    The two steps are fused on purpose. Computing all of ``S`` and collapsing afterwards is the
    same arithmetic but needs the whole (observations x gallery rows) product resident at once;
    collapsing inside the loop keeps only one chunk alive.

    :param samples: Unit-norm query embeddings, shape (observations, dim)
    :param gallery: Unit-norm gallery embeddings sorted by identity, shape (rows, dim)
    :param gallery_starts: Where each identity's run of gallery rows begins
    :param collapse: The reduction over one identity's rows
    :param chunk_bytes: How large one block of the product may be
    :return: One score per identity per observation, shape (observations, identities)
    """
    scores = np.empty((len(samples), len(gallery_starts)), dtype=np.float32)

    # Size the chunk so one block stays within the budget, but never below a single row.
    chunk = max(1, int(chunk_bytes // (gallery.shape[0] * np.dtype(np.float32).itemsize)))
    for start in range(0, len(samples), chunk):
        stop = min(start + chunk, len(samples))
        # Unit-norm rows on both sides, so the dot product is the cosine similarity.
        similarity = samples[start:stop] @ gallery.T
        scores[start:stop] = collapse.reduceat(similarity, gallery_starts, axis=1)
    return scores


def true_ranks(scores: NDArray[np.float32], truth: NDArray[np.intp]) -> tuple[NDArray[np.intp], int]:
    """
    The rank the true identity achieved, for every query.

    Ranking once and thresholding it is what makes every k consistent with every other, and with
    the top-1 prediction.

    Ties are resolved optimistically - only strictly greater scores push the true identity down.
    With float cosine similarities an exact tie at the top is vanishingly rare, so rather than pick
    a tie-break policy and hide the question, the count is returned: a non-zero tie count means the
    scores have degenerated and the metric is no longer measuring similarity.

    :param scores: Per-query identity scores, shape (queries, identities)
    :param truth: The column of each query's true identity, or -1 when that identity is absent
        from the gallery and so cannot be retrieved at any k
    :return: The 1-based rank of the true identity (identities + 1 when absent), and how many
        queries had a tied top score
    """
    ranks = np.full(len(scores), scores.shape[1] + 1, dtype=np.intp)

    known = truth >= 0
    if known.any():
        rows = np.flatnonzero(known)
        achieved = scores[rows, truth[known]][:, None]
        ranks[rows] = (scores[rows] > achieved).sum(axis=1) + 1

    best = scores.max(axis=1, keepdims=True)
    tied = int(((scores == best).sum(axis=1) > 1).sum())
    return ranks, tied


def score_fold(
    gallery: Leg,
    samples: Leg,
    collapse: np.ufunc,
    prediction: PredictionAggregator,
    ranks: tuple[int, ...] = RANKS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Score one fold end to end, returning one row per query video.

    :param gallery: What the attacker holds
    :param samples: What they observe
    :param collapse: The gallery-axis reduction
    :param prediction: How a query's observations become one decision
    :param ranks: Which rank-k accuracies to report
    :return: The per-video results, and a dict of diagnostics for the caller to log
    """
    # The identity axis is the gallery's identities, deliberately not the union with the samples'.
    # An identity the gallery does not hold cannot be retrieved at any k, and giving it an
    # all-zero column would let it be swept into a top-k by tie-breaking rather than by similarity.
    gallery_starts = group_starts(gallery.identity)
    classes = gallery.identity[gallery_starts]
    column_of = {identity: column for column, identity in enumerate(classes)}

    scores = identity_scores(samples.embeddings, gallery.embeddings, gallery_starts, collapse)

    # The query video every observation belongs to. 'video' is only a per-identity number, so the
    # identity has to be part of the key.
    video_keys = np.stack([samples.identity, samples.video], axis=1)
    video_starts = group_starts(video_keys)
    video_sizes = np.diff(np.append(video_starts, len(samples.embeddings))).astype(np.intp)

    fused = prediction.aggregate(scores, video_starts, video_sizes)
    if fused is None:
        # Nothing was fused, so the unit carrying a decision is still the observation and many
        # units map to one video. The per-video reduction then happens in the metric below.
        unit_scores, unit_identity = scores, samples.identity
        unit_starts, unit_sizes = video_starts, video_sizes
    else:
        unit_scores, unit_identity = fused, video_keys[video_starts, 0]
        unit_starts = np.arange(len(video_starts), dtype=np.intp)
        unit_sizes = np.ones(len(video_starts), dtype=np.intp)

    truth = np.fromiter(
        (column_of.get(identity, -1) for identity in unit_identity), dtype=np.intp, count=len(unit_identity)
    )
    achieved, tied = true_ranks(unit_scores, truth)
    predictions = classes[unit_scores.argmax(axis=1)]

    results = pd.DataFrame(
        {
            "identity": video_keys[video_starts, 0],
            "video": video_keys[video_starts, 1],
            "observations": video_sizes,
            # The rank the true identity reached, so a CMC curve at any k is derivable later
            # without rescoring. One unit per video for a fusing aggregation; for 'none' it is the
            # mean rank over that video's observations.
            "rank": np.add.reduceat(achieved.astype(np.float64), unit_starts) / unit_sizes,
        }
    )
    for k in ranks:
        hits = (achieved <= min(k, len(classes))).astype(np.float32)
        results[f"top-{k}"] = np.add.reduceat(hits, unit_starts) / unit_sizes.astype(np.float32)

    # The identity a video was most often assigned. Carried so that a degenerate condition stays
    # visible downstream: under heavy anonymization a large share of queries can land on one
    # identity, which an accuracy alone hides.
    if fused is None:
        video_of_unit = np.repeat(np.arange(len(unit_starts)), unit_sizes)
        modal = pd.Series(predictions).groupby(video_of_unit).agg(lambda votes: votes.mode().iloc[0])
        results["predicted"] = modal.to_numpy()
    else:
        results["predicted"] = predictions

    diagnostics = {
        "videos": len(results),
        "observations": int(video_sizes.sum()),
        "identities": len(classes),
        "ungalleried": int((truth < 0).sum()),
        "tied": tied,
    }
    return results, diagnostics


def score_folds(
    gallery: pd.DataFrame,
    gallery_embeddings: NDArray[np.float32],
    samples: pd.DataFrame,
    samples_embeddings: NDArray[np.float32],
    collapse: np.ufunc,
    prediction: PredictionAggregator,
    ranks: tuple[int, ...] = RANKS,
) -> tuple[pd.DataFrame, dict[str, int]]:
    """
    Score every fold, querying each one against a gallery built from the others.

    Both legs were split the same way, so holding the fold out is also what stops a query video
    from matching its own, merely differently anonymized, copy in the gallery.

    Nothing is averaged per fold: the folds hold different numbers of query videos, so a mean of
    per-fold means would weight them unequally. Emitting one row per video and letting the reader
    take a single mean is what keeps that mistake unavailable.

    :param gallery: The gallery leg's structure, with 'identity', 'video' and 'fold'
    :param gallery_embeddings: The gallery's embeddings, shape (len(gallery), dim), aligned by row order
    :param samples: The samples leg's structure, with the same columns
    :param samples_embeddings: The samples' embeddings, shape (len(samples), dim), aligned by row order
    :param collapse: The gallery-axis reduction
    :param prediction: How a query's observations become one decision
    :param ranks: Which rank-k accuracies to report
    :return: One row per query video across every fold, and the summed diagnostics
    :raises ValueError: If a fold has an empty gallery or no queries
    """
    per_video = []
    totals = {"videos": 0, "observations": 0, "ungalleried": 0, "tied": 0}

    for fold in sorted(samples["fold"].unique()):
        held_out_rows = (gallery["fold"] != fold).to_numpy()
        queried_rows = (samples["fold"] == fold).to_numpy()
        held_out, queried = gallery[held_out_rows], samples[queried_rows]
        if held_out.empty or queried.empty:
            raise ValueError(
                f"Fold {fold} has an empty gallery ({len(held_out)} rows) or no queries "
                f"({len(queried)} rows); check that both legs cover the same folds"
            )

        results, diagnostics = score_fold(
            Leg.from_frame(held_out, gallery_embeddings[held_out_rows]),
            Leg.from_frame(queried, samples_embeddings[queried_rows]),
            collapse,
            prediction,
            ranks,
        )
        per_video.append(results.assign(fold=int(fold)))
        for key in totals:
            totals[key] += diagnostics[key]

    return pd.concat(per_video, ignore_index=True), totals
