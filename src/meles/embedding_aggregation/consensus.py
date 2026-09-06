import numpy as np

from meles.embedding_aggregation.aggregator import EmbeddingAggregator, normalized
from meles.common import Embeddings


class ConsensusEmbeddingAggregator(EmbeddingAggregator):
    """
    Consensus self-attention (attention family, embeddings-only).

    Each embedding is weighted by how well it agrees with the rest of the set: a
    leave-one-out mean cosine similarity to every other embedding, fed through a
    softmax. Outlier/corrupted frames naturally get down-weighted, with no learned
    query/key/value projection. Same intuition as the SAB/PMA blocks in Set
    Transformer (points attend to each other, invariant to order and set size),
    just parameter-free.

    `beta` is the softmax temperature (higher = sharper down-weighting of
    low-consensus embeddings). Its optimal value is method/embedder-dependent
    (measured: low beta wins under gaussianblur/ghostfacenet, high beta wins
    under blackbar/arcface), so it's swept as separate aggregation_methods
    (consensus_beta1/5/20 in workflow/config.yaml and aggregate.py) rather than
    hardcoded here, the same way clustering sweeps distance_threshold.
    """

    use_similarity: bool = True
    beta: float = 1.0
    eps: float = 1e-8

    def __init__(self, use_similarity: bool = True, beta: float = 1.0, eps: float = 1e-8):
        super().__init__()
        self.use_similarity = use_similarity
        self.beta = beta
        self.eps = eps

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        Xn = np.asarray(normalized(embeddings))
        n = Xn.shape[0]
        if n == 1:
            return [Xn[0].tolist()]

        S = Xn @ Xn.T  # unit-norm inputs, so the dot product is the cosine similarity
        consensus = (S.sum(axis=1) - 1.0) / (n - 1)  # leave-one-out agreement with the rest

        sign = 1.0 if self.use_similarity else -1.0
        scaled = sign * self.beta * consensus
        weights = np.exp(scaled - scaled.max())  # softmax, shifted for numerical stability
        weights /= weights.sum() + self.eps

        return [(weights @ Xn).tolist()]
