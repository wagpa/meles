"""
Scoring primitives for video-level re-identification.

A query video can be turned into a decision at three different fusion levels,
all of which share the same gallery:

- single frame: every frame is matched and decided on its own. This is the floor,
  the accuracy you get from one observation.
- score level: every frame is matched on its own, then the per-frame neighbour
  votes are fused into one decision per video. No embedding is ever modified.
- feature level: the frames of a video are fused into one embedding before it is
  matched, which is what the aggregators in ``meles.aggregation`` do.

The functions below keep those levels comparable by sharing one gallery, one
neighbour count and one vote weighting.
"""

from typing import Optional, Sequence

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.metrics import top_k_accuracy_score

from meles.classifier import Classifier

# The number of gallery neighbours every query vector votes with.
K = 20


def label_space(*label_arrays: Sequence) -> NDArray:
    """
    The sorted label space spanned by the given label arrays. The gallery labels define
    which identities can be predicted, the query labels are unioned in so that a label
    which never occurs in the gallery cannot silently disappear from the score matrix.

    :param label_arrays: The label arrays to span the label space over
    :return: The sorted unique labels
    """
    space = np.unique(np.concatenate([np.asarray(a) for a in label_arrays]))
    return space


def vote_matrix(
    classifier: Classifier,
    embeddings: NDArray,
    classes: NDArray,
    n_neighbors: int = K,
) -> NDArray:
    """
    Score every query embedding against the gallery and return the per-query class
    scores. Each of the ``n_neighbors`` nearest gallery neighbours votes for its
    identity, weighted by its cosine similarity.

    The weighting matters: with a gallery that holds a single embedding per identity
    all neighbours are distinct identities with exactly one unweighted vote each, so
    an unweighted count discards the distances that decide which neighbour is actually
    the closest and the ranking degenerates into an arbitrary tie-break.

    :param classifier: The fitted classifier holding the gallery
    :param embeddings: The query embeddings with shape (num_queries, embedding_dim)
    :param classes: The label space, sorted, indexing the columns of the result
    :param n_neighbors: The number of gallery neighbours to let vote
    :return: The row-normalized class scores with shape (num_queries, num_classes)
    """
    class_to_idx = {label: i for i, label in enumerate(classes)}
    neighbor_labels, distances = classifier.classify(
        embeddings, n_neighbors=n_neighbors
    )
    weights = np.clip(1.0 - distances, 0.0, None)

    scores = np.zeros((len(neighbor_labels), len(classes)))
    rows = np.arange(len(neighbor_labels))
    for j in range(neighbor_labels.shape[1]):
        idx = np.fromiter(
            (class_to_idx[label] for label in neighbor_labels[:, j]),
            dtype=int,
            count=len(neighbor_labels),
        )
        scores[rows, idx] += weights[:, j]

    # Normalize every query to a distribution so that queries contribute equally when
    # they are pooled. Queries whose neighbours all sit further away than orthogonal
    # carry no weight and stay all-zero.
    totals = scores.sum(axis=1, keepdims=True)
    np.divide(scores, totals, out=scores, where=totals > 0)
    return scores


def video_keys(structure: pd.DataFrame) -> pd.MultiIndex:
    """
    The key that identifies a video. The ``video`` column only holds the per-identity
    video number ("0", "1", ...), so it is not unique on its own and only the
    (identity, video) pair identifies a video.

    :param structure: A structure with the columns "identity" and "video"
    :return: The (identity, video) index of every row
    """
    return pd.MultiIndex.from_arrays(
        [structure["identity"].to_numpy(), structure["video"].to_numpy()],
        names=["identity", "video"],
    )


def pool_per_video(
    scores: NDArray, keys: pd.MultiIndex
) -> tuple[NDArray, NDArray]:
    """
    Fuse per-query class scores into one score vector per video by averaging.

    :param scores: The per-query class scores with shape (num_queries, num_classes)
    :param keys: The (identity, video) key of every query
    :return: A tuple (video_scores, identities) of the pooled scores and the true
             identity of every pooled video
    """
    pooled = (
        pd.DataFrame(scores, index=keys)
        .groupby(level=["identity", "video"], sort=False)
        .mean()
    )
    identities = pooled.index.get_level_values("identity").to_numpy()
    return pooled.to_numpy(), identities


def top_k_accuracies(
    y_true: NDArray,
    scores: NDArray,
    classes: NDArray,
    ks: Sequence[int] = (1, 5, 10),
) -> dict[str, float]:
    """
    Top-k accuracy for every requested k. ``k`` is capped below the number of classes
    because a ``k`` equal to the class count is trivially perfect.

    :param y_true: The true label of every row
    :param scores: The class scores with shape (num_rows, num_classes)
    :param classes: The label space, sorted, indexing the columns of ``scores``
    :param ks: The values of k to report
    :return: The accuracies keyed as "top-{k}"
    """
    return {
        f"top-{k}": top_k_accuracy_score(
            y_true, scores, k=min(k, len(classes) - 1), labels=classes
        )
        for k in ks
    }


def row_accuracy(scores: NDArray, y_true: Sequence, classes: NDArray) -> float:
    """
    Top-1 accuracy of every query row decided on its own, without any pooling.

    :param scores: The per-row class scores with shape (num_rows, num_classes)
    :param y_true: The true label of every row
    :param classes: The label space, sorted, indexing the columns of ``scores``
    :return: The fraction of rows whose highest scoring class is correct
    """
    predicted = classes[scores.argmax(axis=1)]
    return float((predicted == np.asarray(y_true)).mean())


def single_frame_accuracy(
    scores: NDArray,
    keys: pd.MultiIndex,
    classes: NDArray,
    repeats: int = 20,
    random_state: Optional[int] = 0,
) -> tuple[float, float]:
    """
    The accuracy of deciding a video from one single frame, measured over videos so
    that it is directly comparable to the fused levels. One frame is drawn per video
    and scored, repeated ``repeats`` times to give a spread.

    :param scores: The per-frame class scores with shape (num_frames, num_classes)
    :param keys: The (identity, video) key of every frame
    :param classes: The label space, sorted, indexing the columns of ``scores``
    :param repeats: The number of independent draws of one frame per video
    :param random_state: The seed of the first draw; later draws increment it
    :return: A tuple (mean, std) of the accuracy across the draws
    """
    predicted = classes[scores.argmax(axis=1)]
    correct = predicted == keys.get_level_values("identity").to_numpy()

    frame = pd.DataFrame({"correct": correct}, index=keys)
    accuracies = []
    for repeat in range(repeats):
        seed = None if random_state is None else random_state + repeat
        drawn = frame.groupby(level=["identity", "video"], sort=False).sample(
            n=1, random_state=seed
        )
        accuracies.append(drawn["correct"].mean())
    return float(np.mean(accuracies)), float(np.std(accuracies))


def fit_gallery(embeddings: NDArray, labels: Sequence) -> Classifier:
    """
    Fit a cosine nearest-neighbour gallery.

    :param embeddings: The gallery embeddings with shape (num_embeddings, embedding_dim)
    :param labels: The identity of every gallery embedding
    :return: The fitted classifier
    """
    classifier = Classifier(metric="cosine")
    classifier.fit(embeddings, labels)
    return classifier


def stack(structure: pd.DataFrame) -> NDArray:
    """
    Stack the "embedding" column of a structure into a float matrix.

    :param structure: A structure with an "embedding" column
    :return: The embeddings with shape (num_rows, embedding_dim)
    """
    return np.vstack(structure["embedding"].to_numpy()).astype(float)
