import numpy as np
from meles.embedding_aggregation.aggregator import EmbeddingAggregator, Embeddings, normalized


class MeanPoolingEmbeddingAggregator(EmbeddingAggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.mean(normalized(embeddings), axis=0, keepdims=True).tolist()


class MedianPoolingEmbeddingAggregator(EmbeddingAggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.median(normalized(embeddings), axis=0, keepdims=True).tolist()


class MaxPoolingEmbeddingAggregator(EmbeddingAggregator):
    """
    The MaxPoolingAggregator pools all embeddings into a single embedding using the column wise max.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.max(normalized(embeddings), axis=0, keepdims=True).tolist()

# TODO: will potentially produce zero embeddings for regularized recognizers
class MinPoolingEmbeddingAggregator(EmbeddingAggregator):
    """
    The MinPoolingAggregator pools all embeddings into a single embedding using the column wise min.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.min(normalized(embeddings), axis=0, keepdims=True).tolist()
