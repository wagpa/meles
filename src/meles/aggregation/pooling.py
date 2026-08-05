import numpy as np
from meles.aggregation.aggregator import Aggregator, Embeddings, normalized


class MeanPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.mean(normalized(embeddings), axis=0, keepdims=True).tolist()


class MedianPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.median(normalized(embeddings), axis=0, keepdims=True).tolist()


class MaxPoolingAggregator(Aggregator):
    """
    The MaxPoolingAggregator pools all embeddings into a single embedding using the column wise max.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.max(normalized(embeddings), axis=0, keepdims=True).tolist()

# TODO: will potentially produce zero embeddings for regularized recognizers
class MinPoolingAggregator(Aggregator):
    """
    The MinPoolingAggregator pools all embeddings into a single embedding using the column wise min.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return np.min(normalized(embeddings), axis=0, keepdims=True).tolist()
