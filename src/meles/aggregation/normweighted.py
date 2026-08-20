import numpy as np

from meles.aggregation.aggregator import Aggregator
from meles.common import Embeddings


class NormWeightedAggregator(Aggregator):
    """
    Norm-weighted average aggregator (embeddings-only).

    Each embedding is weighted by its norm before normalizing.
    """

    use_similarity: bool = True
    beta: float = 1.0
    eps: float = 1e-8

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        # Convert input to 2D numpy array: shape (N, D)
        X = np.array(embeddings, dtype=np.float64)

        # Compute weighted sum of feature vectors
        norms = np.linalg.norm(X, axis=1, keepdims=True)
        weighted_sum = np.sum(X * norms, axis=0)
        total_weight = np.sum(norms)

        # Avoid division by zero if all vectors are zero
        if total_weight == 0:
            return np.zeros(X.shape[1], dtype=np.float64)

        # Re-normalize the aggregated vector to unit length
        y_avg = weighted_sum / total_weight
        avg_norm = np.linalg.norm(y_avg)
        if avg_norm == 0:
            return y_avg

        return [y_avg / avg_norm]
