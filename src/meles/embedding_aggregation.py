"""
Combining the embeddings of a group after each frame has been recognized on its own.

The counterpart of :mod:`meles.frame_aggregation` at the other end of the recognizer, and the
difference between them is the point of having both: there many anonymized observations are
combined into one image that is recognized once, here each is recognized first and the vectors are
combined afterward.

An aggregator returns **one or more** embeddings. Pooling and consensus collapse a group to one;
an aggregator is free to return several, or none, meaning the group has no representative.
"""

from collections.abc import Callable, Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


def normalized(embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
    """
    Scale every row to unit length, which is what turns a cosine similarity into a dot product.

    A zero embedding has no direction to compare, so its norm is left at one: it stays a zero row,
    scoring zero against everything, rather than becoming a NaN that poisons whatever it touches.

    :param embeddings: The embeddings, shape (n, dim)
    :return: The embeddings with unit-norm rows
    """
    vectors = np.asarray(embeddings, dtype=np.float32)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return vectors / np.where(norms == 0, 1, norms)


class EmbeddingAggregator(Protocol):
    """Combines the embeddings of one group into one or more embeddings."""

    def aggregate(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        """
        :param embeddings: The group's embeddings, shape (n, dim)
        :return: The embeddings the group becomes, shape (k, dim) for some k >= 0
        """
        ...


class NoEmbeddingAggregator(EmbeddingAggregator):
    """The baseline: every observation stays its own embedding."""

    def aggregate(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        return embeddings


class PoolingEmbeddingAggregator(EmbeddingAggregator):
    """
    Reduces a group to one embedding, column by column.

    The embeddings are normalised first, so that a long vector does not simply outvote the others -
    without it the result would be ranked by embedding norm rather than by agreement.
    """

    def __init__(self, reduce: Callable[..., NDArray[np.float32]]) -> None:
        """
        :param reduce: The column-wise reduction, for example ``np.mean`` or ``np.median``
        """
        self.reduce = reduce

    def aggregate(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        pooled = self.reduce(normalized(embeddings), axis=0, keepdims=True)
        return np.asarray(pooled, dtype=np.float32)


class ConsensusEmbeddingAggregator(EmbeddingAggregator):
    """
    Weights each embedding by how far the rest of the group agrees with it, then averages.

    An embedding's weight is its leave-one-out mean cosine similarity to the others, put through a
    softmax. A frame whose embedding disagrees with the rest - a bad crop, a frame the
    anonymization destroyed - is down-weighted without anything being learned.

    ``beta`` is the softmax temperature: higher sharpens the down-weighting. Its best value depends
    on the anonymization and the recognizer, so it is part of the registry key rather than
    something tuned here.
    """

    def __init__(self, beta: float = 1.0) -> None:
        """
        :param beta: The softmax temperature over the agreement scores
        """
        self.beta = beta

    def aggregate(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        unit = normalized(embeddings)
        if len(unit) == 1:
            return unit

        # Unit-norm rows, so the dot product is the cosine similarity. Dropping the diagonal
        # leaves each embedding's agreement with the others rather than with itself.
        similarity = unit @ unit.T
        agreement = (similarity.sum(axis=1) - 1.0) / (len(unit) - 1)

        scaled = self.beta * agreement
        weights = np.exp(scaled - scaled.max())  # shifted before the exponential for stability
        weights /= weights.sum()
        return np.asarray(weights @ unit, dtype=np.float32).reshape(1, -1)


class NormWeightedEmbeddingAggregator(EmbeddingAggregator):
    """
    Averages the group weighted by each embedding's norm, then rescales to unit length.

    Unlike the pooling aggregators this does **not** normalise first: the norm is the weight. For
    recognizers whose embedding norm carries a notion of quality, that makes a confident frame
    count for more than an uncertain one.
    """

    def aggregate(self, embeddings: NDArray[np.float32]) -> NDArray[np.float32]:
        vectors = np.asarray(embeddings, dtype=np.float32)
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)

        total = float(norms.sum())
        if total == 0:
            return np.zeros((1, vectors.shape[1]), dtype=np.float32)

        pooled = (vectors * norms).sum(axis=0) / total
        length = float(np.linalg.norm(pooled))
        if length == 0:
            return pooled.reshape(1, -1)
        return (pooled / length).reshape(1, -1)


# Every embedding aggregation the experiment can name. Constructing an entry has to stay free of
# side effects, because the whole dict is built at import time.
EMBEDDING_AGGREGATORS: dict[str, EmbeddingAggregator] = {
    "none": NoEmbeddingAggregator(),
    "pooling_mean": PoolingEmbeddingAggregator(np.mean),
    "pooling_median": PoolingEmbeddingAggregator(np.median),
    "pooling_max": PoolingEmbeddingAggregator(np.max),
    "pooling_min": PoolingEmbeddingAggregator(np.min),
    "consensus_beta1": ConsensusEmbeddingAggregator(beta=1.0),
    "consensus_beta5": ConsensusEmbeddingAggregator(beta=5.0),
    "consensus_beta20": ConsensusEmbeddingAggregator(beta=20.0),
    "normweighted": NormWeightedEmbeddingAggregator(),
}


def resolve(method: str) -> EmbeddingAggregator:
    """
    The embedding aggregator a configuration key names.

    :param method: The key, for example 'pooling_mean'
    :return: The shared aggregator instance
    :raises ValueError: If nothing is registered under that key
    """
    aggregator = EMBEDDING_AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown embedding aggregation {method!r}, expected one of {sorted(EMBEDDING_AGGREGATORS)}")
    return aggregator


def aggregate_groups(
    embeddings: NDArray[np.float32],
    starts: Sequence[int],
    aggregator: EmbeddingAggregator,
) -> tuple[NDArray[np.float32], NDArray[np.intp]]:
    """
    Apply an aggregator to each run of rows, for rows already sorted into their groups.

    :param embeddings: Every group's embeddings stacked, shape (n, dim)
    :param starts: Where each group's run of rows begins, strictly increasing
    :param aggregator: The aggregation to apply within a group
    :return: The aggregated embeddings, and the group each one came from
    """
    bounds = [*starts, len(embeddings)]
    produced: list[NDArray[np.float32]] = []
    groups: list[int] = []
    for group, (start, stop) in enumerate(zip(bounds, bounds[1:])):
        result = aggregator.aggregate(embeddings[start:stop])
        produced.append(np.asarray(result, dtype=np.float32).reshape(-1, embeddings.shape[1]))
        groups.extend([group] * len(produced[-1]))

    stacked = np.concatenate(produced) if produced else np.empty((0, embeddings.shape[1]), dtype=np.float32)
    return stacked, np.asarray(groups, dtype=np.intp)
