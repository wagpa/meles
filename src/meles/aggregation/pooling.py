import torch
from meles.aggregation.aggregator import Aggregator, AggregationInput, AggregationOutput


class MeanPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        Calculate the mean along dimension 0 (columns)
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The pooled embedding with shape (1, dim)
        """
        return torch.mean(embeddings, dim=0, keepdim=True)


class MaxPoolingAggregator(Aggregator):
    """
    The MaxPoolingAggregator pools all embeddings into a single embedding using the column wise max.
    """

    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        Calculate the max along dimension 0 (columns)
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The pooled embedding with shape (1, dim)
        """
        return torch.max(embeddings, dim=0, keepdim=True)


class MinPoolingAggregator(Aggregator):
    """
    The MinPoolingAggregator pools all embeddings into a single embedding using the column wise min.
    """

    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        Calculate the min along dimension 0 (columns)
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The pooled embedding with shape (1, dim)
        """
        return torch.min(embeddings, dim=0, keepdim=True)
