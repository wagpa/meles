import torch
from meles.aggregation.aggregator import Aggregator, Embeddings


class MeanPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __call__(self, embeddings: Embeddings) -> Embeddings:
        return torch.mean(embeddings, dim=0, keepdim=True)


class MaxPoolingAggregator(Aggregator):
    """
    The MaxPoolingAggregator pools all embeddings into a single embedding using the column wise max.
    """

    def __call__(self, embeddings: Embeddings) -> Embeddings:
        return torch.max(embeddings, dim=0, keepdim=True)


class MinPoolingAggregator(Aggregator):
    """
    The MinPoolingAggregator pools all embeddings into a single embedding using the column wise min.
    """

    def __call__(self, embeddings: Embeddings) -> Embeddings:
        return torch.min(embeddings, dim=0, keepdim=True)
