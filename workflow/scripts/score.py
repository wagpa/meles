"""
Score one pairing of the experiment: a gallery leg (what the attacker holds) against a samples leg
(what they get to observe).

The algorithm is a dense, two-axis reduction of one cosine similarity matrix. Reading it as "two
collapses on two different axes" is the whole thing:

    S = samples @ gallery.T                     (n_observations, n_gallery_rows)

    1. collapse the GALLERY axis  -> (n_observations, n_identities)
       How do the several gallery rows of one identity become one score for that identity?
    2. collapse the SAMPLES axis  -> (n_units, n_identities)
       How do the several observations of one query video become one decision?

'{score_aggregation}' names both halves as '{gallery}_{samples}', e.g. 'max_mean'.

Why this shape rather than a k-nearest-neighbour shortlist
---------------------------------------------------------
The previous implementation did the opposite order -- shortlist the K nearest gallery *rows*, then
accumulate their similarities into identity columns. That made the result depend on how many rows
an identity happens to own: with one row per frame, 20 neighbours covered only ~6.7 distinct
identities, so a gallery of per-frame embeddings and a gallery of pooled embeddings were never
comparable. Collapsing the gallery axis first removes that confound: every identity is represented
by its best-matching exemplar regardless of how many exemplars it has.

A residual, much weaker bias remains and is worth remembering when reading results: the maximum of
many draws is stochastically larger than the maximum of few, growing roughly like sqrt(2*log(n)).
'max' kills the *linear* cardinality bias, not the extreme-value one. The clean comparison still
holds the number of gallery rows per identity roughly equal across the conditions being compared.

Why the scores are raw cosine similarities and are never renormalised
--------------------------------------------------------------------
After the gallery collapse every row is a dense similarity profile over all identities, already on
one bounded scale (cosine in [-1, 1]) by construction, so there is no scale to correct for. A query
observation that happens to sit close to *everything* lifts its whole row by roughly a constant,
and adding a constant to a row changes neither its argmax nor the argmax of a sum of such rows --
hub observations therefore cancel on their own.

This is a deliberate departure from the old 'vote_matrix', which normalised each row to a
distribution. That was necessary there because the shortlist made rows sparse and incomparably
scaled; it is not necessary here, and it would compress exactly the differences the 'max' samples
collapse is supposed to read. A 'product' samples collapse *would* need normalisation plus
smoothing, which is why it is not implemented yet rather than implemented badly.

Output format: one row per query video
--------------------------------------
Every query video contributes exactly one row, carrying the fraction of its observations that were
correct at each k. That single property removes a whole class of weighting bugs:

  * ``mean()`` over any subset of rows is automatically the correctly weighted macro-average,
    because each video counts once no matter how many frames it had;
  * pooling across folds is just ``mean()`` over all rows -- no second average over folds, which
    would otherwise weight the three folds 0.40 / 0.31 / 0.29 by their differing video counts;
  * per-identity and per-fold breakdowns, and a cluster bootstrap over identities, are all still
    available downstream because nothing has been averaged away yet.

For the fused collapses ('mean', 'max') a video has one observation, so the fractions are 0.0 or
1.0. For 'none' they are that video's per-frame accuracy, which is the single-observation floor --
the one number the previous implementation could not report, because it averaged over rows and
"row" silently meant "frame" in one condition and "video" in another.
"""

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from numpy.typing import NDArray

from meles.runtime import logs, runtime

# Fix snakemake typechecking errors
if TYPE_CHECKING:
    from snakemake.script import Snakemake

    snakemake: Snakemake

# Configure the module logger. Logs are passed into the Snakemake configured logfile.
logger = logging.getLogger(__name__)

# The ranks reported. A video is "correct at k" when its true identity is among the k highest
# scoring identities. k is capped below the number of identities the gallery can offer, since a k
# equal to the identity count is trivially perfect.
KS = (1, 5, 10)

# How many bytes one chunk of the similarity matrix may occupy. The matrix itself is never
# materialised: a chunk of query rows is multiplied out, collapsed onto the identity axis straight
# away and thrown away, so the gallery axis only ever exists one chunk at a time. Without this the
# full product would be (n_observations x n_gallery_rows), which reaches ~16 GB for one fold of
# per-frame embeddings against a per-frame gallery.
CHUNK_BYTES = 512 * 1024**2

# How the several gallery rows of one identity become that identity's score. Every entry here must
# be invariant to the *number* of rows an identity owns, otherwise it confounds the gallery
# aggregation axis with this one -- see the module docstring. 'sum' is deliberately absent for that
# reason; add it only as an explicit legacy/reproduction point, never as a default.
GALLERY_COLLAPSES: dict[str, np.ufunc] = {
    "max": np.maximum,
}

# How the several observations of one query video become one decision. Unlike the gallery side this
# axis has no invariance requirement -- more observations are *supposed* to help, which is the
# hypothesis the experiment exists to test.
# 'none' is not a reduction at all: every observation stays its own query and makes its own
# decision, having seen no other observation's evidence. It is the degenerate member of the family
# (group size one) and therefore the single-observation floor. The per-video reduction that still
# happens for it lives in the metric, not here, and only exists so a 75 frame video does not
# outweigh a 48 frame one.
SAMPLES_COLLAPSES: dict[str, np.ufunc | None] = {
    "none": None,
    "mean": np.add,  # summed by reduceat, then divided by the group size
    "max": np.maximum,
}


def resolve(score_aggregation: str) -> tuple[np.ufunc, np.ufunc | None, str]:
    """
    Read a '{gallery}_{samples}' key as the two collapses it names.

    :param score_aggregation: The method key, e.g. 'max_mean' or 'max_none'
    :return: The gallery-axis ufunc, the samples-axis ufunc (None for 'none'), and the samples
        method name, which the caller needs because 'mean' is 'np.add' plus a division
    :raises ValueError: If either half names something the registries have no entry for
    """
    gallery_method, _, samples_method = score_aggregation.partition("_")

    gallery_collapse = GALLERY_COLLAPSES.get(gallery_method)
    if gallery_collapse is None:
        raise ValueError(
            f"Unknown gallery score collapse {gallery_method!r} in {score_aggregation!r}, "
            f"expected one of {sorted(GALLERY_COLLAPSES)}"
        )

    if samples_method not in SAMPLES_COLLAPSES:
        raise ValueError(
            f"Unknown samples score collapse {samples_method!r} in {score_aggregation!r}, "
            f"expected one of {sorted(SAMPLES_COLLAPSES)}"
        )
    return gallery_collapse, SAMPLES_COLLAPSES[samples_method], samples_method


def unit_rows(frame: pd.DataFrame) -> NDArray[np.float32]:
    """
    Stack the "embedding" column into an L2-normalised float32 matrix.

    Normalising here is what turns the cosine similarity into a plain dot product, so the whole
    retrieval step becomes one matrix multiply. float32 is deliberate: it halves the memory of the
    chunked product and the difference against float64 lands far below the precision anything
    downstream is read at.

    :param frame: A structure with an "embedding" column
    :return: The embeddings with shape (num_rows, embedding_dim) and unit-norm rows
    """
    embeddings = np.vstack(frame["embedding"].to_numpy()).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    # A zero embedding has no direction to compare; leaving the norm at 1 keeps it a zero row,
    # which scores 0 against everything rather than producing a NaN that silently poisons a column.
    return embeddings / np.where(norms == 0, 1, norms)


def group_starts(keys: NDArray) -> NDArray[np.intp]:
    """
    The index where each run of equal keys begins, for keys that are already sorted.

    These are exactly the indices ``ufunc.reduceat`` wants: it reduces ``[starts[i], starts[i+1])``
    for each i, and the final group to the end of the axis. Requiring sorted input is what makes
    both collapses a single vectorised call instead of a per-group loop.

    :param keys: A sorted array of shape (n,) or (n, num_key_columns)
    :return: The strictly increasing start index of every group
    """
    keys = np.atleast_2d(keys.T).T  # (n,) -> (n, 1), (n, k) stays (n, k)
    changed = np.ones(len(keys), dtype=bool)
    changed[1:] = (keys[1:] != keys[:-1]).any(axis=1)
    return np.flatnonzero(changed)


def identity_scores(
    samples: NDArray[np.float32],
    gallery: NDArray[np.float32],
    gallery_group_starts: NDArray[np.intp],
    collapse: np.ufunc,
) -> NDArray[np.float32]:
    """
    Steps 1 and 2: the similarity matrix, collapsed onto the identity axis as it is produced.

    The two steps are fused on purpose. Computing all of ``S`` first and collapsing afterwards is
    the same arithmetic but needs the whole (n_observations x n_gallery_rows) product resident at
    once; collapsing inside the loop keeps only (chunk x n_gallery_rows) alive and accumulates the
    much smaller (n_observations x n_identities) result.

    :param samples: Unit-norm query embeddings, shape (n_observations, embedding_dim)
    :param gallery: Unit-norm gallery embeddings sorted by identity, shape (n_gallery_rows, dim)
    :param gallery_group_starts: Where each identity's run of gallery rows begins
    :param collapse: The ufunc reducing an identity's rows to one score, e.g. ``np.maximum``
    :return: One score per identity per observation, shape (n_observations, n_identities)
    """
    n_identities = len(gallery_group_starts)
    scores = np.empty((len(samples), n_identities), dtype=np.float32)

    # Size the chunk so one similarity block stays within the byte budget, but never go below a
    # single row -- a gallery large enough to break that would be a different problem entirely.
    chunk = max(1, int(CHUNK_BYTES // (gallery.shape[0] * np.dtype(np.float32).itemsize)))
    for start in range(0, len(samples), chunk):
        stop = min(start + chunk, len(samples))
        # Unit-norm rows on both sides, so the dot product is the cosine similarity.
        similarity = samples[start:stop] @ gallery.T
        # Collapse the gallery axis: one score per identity, from that identity's rows only.
        scores[start:stop] = collapse.reduceat(similarity, gallery_group_starts, axis=1)
    return scores


def collapse_observations(
    scores: NDArray[np.float32],
    starts: NDArray[np.intp],
    sizes: NDArray[np.intp],
    collapse: np.ufunc,
    method: str,
) -> NDArray[np.float32]:
    """
    Step 3: collapse the samples axis, fusing the observations of one video into one score vector.

    Only reached for the fusing methods. 'none' never calls this -- it has nothing to fuse, and
    conflating "do not fuse" with "fuse by picking one" would make the floor an estimate with one
    sample per video instead of the exact expectation over all of them.

    :param scores: Per-observation identity scores, shape (n_observations, n_identities)
    :param starts: Where each video's run of observations begins
    :param sizes: How many observations each video has
    :param collapse: The ufunc reducing the observations, e.g. ``np.maximum`` or ``np.add``
    :param method: The samples method name, so that 'mean' can divide after ``np.add``
    :return: One score vector per video, shape (n_videos, n_identities)
    """
    fused = collapse.reduceat(scores, starts, axis=0)
    if method == "mean":
        # reduceat summed; dividing by the group size is what makes it a mean. Done here rather
        # than by normalising rows beforehand so that every observation weighs the same regardless
        # of how strongly it matched anything -- see the module docstring on normalisation.
        fused = fused / sizes[:, None].astype(np.float32)
    return fused


def true_ranks(
    scores: NDArray[np.float32],
    truth_columns: NDArray[np.intp],
) -> tuple[NDArray[np.intp], int]:
    """
    Step 4: the rank the true identity achieved, for every observation.

    Ranking once and thresholding it is what makes every k consistent with every other, and with
    the top-1 prediction. The previous implementation instead handed a sparse score matrix to
    ``sklearn.top_k_accuracy_score``, which fills the ranks below the last non-zero class by sorted
    *label order* -- so identities whose names sort late collected free credit at top-5 and top-10
    whenever the score vector was sparse, which it always was under a shortlist.

    Ties are resolved optimistically (only strictly greater scores push the true identity down).
    With float cosine similarities an exact tie at the top is vanishingly rare, so rather than pick
    a tie-break policy and hide the question, the count is returned and logged: a non-zero tie count
    means the scores have degenerated and the metric is no longer measuring similarity.

    :param scores: Per-unit identity scores, shape (n_units, n_identities)
    :param truth_columns: The column of each unit's true identity, or -1 when that identity is
        absent from the gallery and therefore cannot be retrieved at any k
    :return: The 1-based rank of the true identity (``n_identities + 1`` when absent), and the
        number of units whose top score was tied
    """
    ranks = np.full(len(scores), scores.shape[1] + 1, dtype=np.intp)

    known = truth_columns >= 0
    if known.any():
        known_rows = np.flatnonzero(known)
        truth_scores = scores[known_rows, truth_columns[known]][:, None]
        ranks[known_rows] = (scores[known_rows] > truth_scores).sum(axis=1) + 1

    # A tie at the very top means argmax picked arbitrarily among equals.
    best = scores.max(axis=1, keepdims=True)
    tied = int(((scores == best).sum(axis=1) > 1).sum())
    return ranks, tied


def score_fold(
    gallery_df: pd.DataFrame,
    samples_df: pd.DataFrame,
    gallery_collapse: np.ufunc,
    samples_collapse: np.ufunc | None,
    samples_method: str,
) -> tuple[pd.DataFrame, dict]:
    """
    Score one fold end to end, returning one row per query video.

    :param gallery_df: The gallery rows of this fold's gallery (the other k-1 folds)
    :param samples_df: The query rows of this fold
    :param gallery_collapse: The gallery-axis ufunc
    :param samples_collapse: The samples-axis ufunc, or None for 'none'
    :param samples_method: The samples method name
    :return: The per-video results, and a dict of diagnostics for the log
    """
    # Both collapses are 'reduceat' over runs of equal keys, which needs the rows sorted by those
    # keys. Sorting here rather than relying on the upstream order makes the contract local.
    gallery_df = gallery_df.sort_values(["identity"], kind="stable")
    samples_df = samples_df.sort_values(["identity", "video"], kind="stable")

    gallery = unit_rows(gallery_df)
    samples = unit_rows(samples_df)

    # The identity axis is the gallery's identities, sorted. Deliberately *not* the union with the
    # samples' identities: an identity the gallery does not hold cannot be retrieved at any k, and
    # giving it an all-zero column would let it be swept into a top-k by tie-breaking instead of by
    # similarity. Queries for such an identity are scored as incorrect, and counted below.
    gallery_identities = gallery_df["identity"].to_numpy()
    gallery_group_starts = group_starts(gallery_identities)
    classes = gallery_identities[gallery_group_starts]
    class_of = {identity: column for column, identity in enumerate(classes)}

    # Steps 1 + 2: similarity matrix, collapsed onto the identity axis chunk by chunk.
    scores = identity_scores(samples, gallery, gallery_group_starts, gallery_collapse)

    # The query video every observation belongs to. Sorted above, so each video is one contiguous
    # run; 'video' is only a per-identity number ("0", "1", ...) so the identity is part of the key.
    video_keys = samples_df[["identity", "video"]].to_numpy()
    video_starts = group_starts(video_keys)
    video_sizes = np.diff(np.append(video_starts, len(samples_df))).astype(np.intp)
    video_identities = video_keys[video_starts, 0]
    video_numbers = video_keys[video_starts, 1]

    # Step 3: collapse the samples axis, or do not.
    #
    # 'unit' is whatever carries one decision. For the fusing methods that is the video itself, and
    # each unit maps to its own video. For 'none' the unit stays the observation, and many units
    # map to one video -- the per-video reduction then happens in step 5, in the metric, which is
    # the separation that keeps all three methods measuring accuracy over the same population.
    if samples_collapse is None:
        unit_scores = scores
        unit_identities = samples_df["identity"].to_numpy()
        unit_starts, unit_sizes = video_starts, video_sizes
    else:
        unit_scores = collapse_observations(scores, video_starts, video_sizes, samples_collapse, samples_method)
        unit_identities = video_identities
        unit_starts = np.arange(len(video_starts), dtype=np.intp)
        unit_sizes = np.ones(len(video_starts), dtype=np.intp)

    # Step 4: rank the true identity for each unit, and read every k off that one ranking.
    truth_columns = np.fromiter(
        (class_of.get(identity, -1) for identity in unit_identities), dtype=np.intp, count=len(unit_identities)
    )
    ranks, tied = true_ranks(unit_scores, truth_columns)
    predictions = classes[unit_scores.argmax(axis=1)]

    # Step 5: reduce to one number per video. For the fusing methods every group holds one unit, so
    # this is the no-op it should be; for 'none' it is the video's per-frame accuracy, and dividing
    # by the group size is what stops a 75 frame video outweighing a 48 frame one.
    results = pd.DataFrame(
        {
            "identity": video_identities,
            "video": video_numbers,
            "observations": video_sizes,
        }
    )
    for k in KS:
        capped = min(k, len(classes))
        hits = (ranks <= capped).astype(np.float32)
        results[f"top-{k}"] = np.add.reduceat(hits, unit_starts) / unit_sizes.astype(np.float32)

    # The identity a video was most often assigned. One observation for the fusing methods, so it
    # is simply the prediction; for 'none' it is the modal per-frame prediction. Carried so the
    # collapse of a degenerate condition stays visible downstream -- under heavy anonymization a
    # large share of all queries can land on a single identity, which an accuracy alone hides.
    if samples_collapse is None:
        video_of_unit = np.repeat(np.arange(len(unit_starts)), unit_sizes)
        modal = pd.Series(predictions).groupby(video_of_unit).agg(lambda votes: votes.mode().iloc[0])
        results["predicted"] = modal.to_numpy()
    else:
        results["predicted"] = predictions

    diagnostics = {
        "videos": len(results),
        "observations": int(video_sizes.sum()),
        "identities": len(classes),
        "ungalleried": int((truth_columns < 0).sum()),
        "tied": tied,
    }
    return results, diagnostics


def main(gallery_path: Path, samples_path: Path, output_path: Path, method: str, experiment: dict):
    # Crate the output directory if it does not exist
    logger.debug("Generating output directory")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Read the two legs of the experiment. They are the same videos aggregated under two different
    # conditions: the gallery is what the attacker holds, the samples are what they get to observe.
    gallery_df = pd.read_parquet(gallery_path)
    samples_df = pd.read_parquet(samples_path)
    if gallery_df.empty or samples_df.empty:
        raise ValueError(f"Empty leg: gallery has {len(gallery_df)} rows, samples have {len(samples_df)}")

    # The legs may differ in anonymization, recognizer and aggregator, but a cosine gallery only
    # means anything inside one embedding space. Mismatched recognizers would otherwise surface as
    # a shape error from deep inside the matrix product.
    gallery_dim = len(gallery_df["embedding"].iloc[0])
    samples_dim = len(samples_df["embedding"].iloc[0])
    if gallery_dim != samples_dim:
        raise ValueError(
            f"The gallery leg embeds to {gallery_dim} dimensions and the samples leg to {samples_dim}; "
            "the two legs have to share a frame recognizer to be comparable"
        )

    gallery_collapse, samples_collapse, samples_method = resolve(method)

    # Score each fold separately, then concatenate. Nothing is averaged per fold: the folds differ
    # in size (they hold different numbers of query videos), so a mean of per-fold means would
    # weight them unequally. Emitting one row per video and letting the reader take a single mean
    # is what keeps that mistake unavailable.
    per_video = []
    totals = {"videos": 0, "observations": 0, "ungalleried": 0, "tied": 0}
    for fold in sorted(samples_df["fold"].unique()):
        # Every fold is queried once, against a gallery built from the other k-1 folds. Both legs
        # were split the same way, so holding the fold out is also what stops a query video from
        # matching its own, merely differently anonymized, copy in the gallery.
        gallery_k_df = gallery_df[gallery_df["fold"] != fold]
        samples_k_df = samples_df[samples_df["fold"] == fold]
        if gallery_k_df.empty or samples_k_df.empty:
            raise ValueError(
                f"Fold {fold} has an empty gallery ({len(gallery_k_df)} rows) or query set "
                f"({len(samples_k_df)} rows); check that both legs cover the same folds"
            )

        results, diagnostics = score_fold(
            gallery_k_df, samples_k_df, gallery_collapse, samples_collapse, samples_method
        )
        per_video.append(results.assign(fold=int(fold)))
        for key in totals:
            totals[key] += diagnostics[key]
        logger.debug("Scored fold", extra={"fold": int(fold), "method": method, **diagnostics})

    results = pd.concat(per_video, ignore_index=True)

    # A query whose identity the gallery never holds is scored as a miss at every k, which is
    # correct but caps the achievable accuracy. Say so rather than letting it look like a result.
    if totals["ungalleried"]:
        logger.warning(
            "Query units whose identity is absent from the gallery",
            extra={"method": method, **totals},
        )
    # A tie at the top means the scores have degenerated and argmax is choosing arbitrarily.
    if totals["tied"]:
        logger.warning("Query units with a tied top score", extra={"method": method, "tied": totals["tied"]})

    # One line per job, carrying the headline numbers and the one diagnostic an accuracy hides: how
    # far the predictions have collapsed onto a few identities. Under heavy anonymization a third
    # of all queries can land on one identity, which is far more informative about what the
    # condition is doing than a near-chance accuracy.
    prediction_counts = results["predicted"].value_counts()
    logger.info(
        "Scored experiment",
        extra={
            **experiment,
            "folds": len(per_video),
            "videos": len(results),
            "observations": totals["observations"],
            **{f"top-{k}": round(float(results[f"top-{k}"].mean()), 5) for k in KS},
            "predicted_identities": int(results["predicted"].nunique()),
            "largest_predicted_share": round(float(prediction_counts.iloc[0] / len(results)), 4),
        },
    )

    # Store the results, tagged with the wildcard combination that produced them. Without the tag
    # the rows are indistinguishable once 'collect' concatenates every point of the experiment.
    results.assign(**experiment).to_parquet(output_path)


if __name__ == "__main__":
    # Snakemake only names the log file; a script rule has to redirect into it itself.
    logs.configure(snakemake.log[0])
    runtime.configure_threads(snakemake.threads)

    # noinspection unbound-local-variable
    main(
        gallery_path=Path(snakemake.input.gallery_dataset),
        samples_path=Path(snakemake.input.samples_dataset),
        output_path=Path(snakemake.output.dataset),
        method=snakemake.wildcards.get("score_aggregation"),
        experiment={
            "gallery_frame_anonymization": snakemake.wildcards.get("gallery_frame_anonymization"),
            "gallery_frame_aggregation": snakemake.wildcards.get("gallery_frame_aggregation"),
            "gallery_frame_embedding": snakemake.wildcards.get("gallery_frame_embedding"),
            "gallery_embedding_aggregation": snakemake.wildcards.get("gallery_embedding_aggregation"),
            "samples_frame_anonymization": snakemake.wildcards.get("samples_frame_anonymization"),
            "samples_frame_aggregation": snakemake.wildcards.get("samples_frame_aggregation"),
            "samples_frame_embedding": snakemake.wildcards.get("samples_frame_embedding"),
            "samples_embedding_aggregation": snakemake.wildcards.get("samples_embedding_aggregation"),
            "score_aggregation": snakemake.wildcards.get("score_aggregation"),
        },
    )
