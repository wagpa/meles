import numpy as np

from meles.aggregation.aggregator import Aggregator
from meles.common import Embeddings


# Consensus self-attention (attention family, embeddings-only)
class ConsensusAggregator(Aggregator):
    use_similarity: bool = True
    eps: float = 1e-8

    def __init__(self, use_similarity: bool = True, eps: float = 1e-8):
        super().__init__()
        self.use_similarity = use_similarity
        self.eps = eps

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        X = np.asarray(embeddings, dtype=np.float64)
        centroid = X.mean(axis=0)

        cos_sim = (
            X
            @ centroid
            / (np.linalg.norm(X, axis=1) * np.linalg.norm(centroid) + self.eps)
        )
        weights = cos_sim if self.use_similarity else (1 - cos_sim)

        weights = np.clip(weights, 0, None)  # guard against negative weights
        weights /= weights.sum() + self.eps

        return [(weights @ X).tolist()]
