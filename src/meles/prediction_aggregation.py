"""
Combining what several observations of one query say about who it is.

The last of the three places the experiment can aggregate, and the only one that happens after the
recognizer has already decided. By this point each observation is a row of similarity scores, one
per gallery identity, and the question is how several such rows become one decision.

Unlike the gallery side of scoring, this axis has **no invariance requirement**: more observations
are supposed to help, which is the hypothesis the experiment exists to test.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class PredictionAggregator:
    """
    How the several observations of one query become one decision.

    :param reduce: The reduction over the observation axis, or None not to reduce at all
    :param average: Whether to divide by the group size afterwards, which turns a sum into a mean
    """

    reduce: np.ufunc | None
    average: bool = False

    def aggregate(
        self,
        scores: NDArray[np.float32],
        starts: NDArray[np.intp],
        sizes: NDArray[np.intp],
    ) -> NDArray[np.float32] | None:
        """
        Fuse each query's observations into one row of identity scores.

        :param scores: One row of identity scores per observation, shape (n, identities), with the
            observations of a query in one contiguous run
        :param starts: Where each query's run of observations begins
        :param sizes: How many observations each query has
        :return: One row per query, or None when this aggregation does not reduce - in which case
            every observation stays its own decision
        """
        if self.reduce is None:
            return None
        fused = self.reduce.reduceat(scores, starts, axis=0)
        if self.average:
            fused = fused / sizes[:, None].astype(np.float32)
        return np.asarray(fused, dtype=np.float32)


# Every prediction aggregation the experiment can name.
#
# 'none' is not a reduction at all: every observation stays its own query and decides having seen
# no other observation's evidence. It is the degenerate member of the family, with a group size of
# one, and therefore the single-observation floor the others are read against.
PREDICTION_AGGREGATORS: dict[str, PredictionAggregator] = {
    "none": PredictionAggregator(reduce=None),
    "mean": PredictionAggregator(reduce=np.add, average=True),
    "max": PredictionAggregator(reduce=np.maximum),
}


def resolve(method: str) -> PredictionAggregator:
    """
    The prediction aggregator a configuration key names.

    :param method: The key, for example 'mean'
    :return: The aggregator
    :raises ValueError: If nothing is registered under that key
    """
    aggregator = PREDICTION_AGGREGATORS.get(method)
    if aggregator is None:
        raise ValueError(f"Unknown prediction aggregation {method!r}, expected one of {sorted(PREDICTION_AGGREGATORS)}")
    return aggregator
