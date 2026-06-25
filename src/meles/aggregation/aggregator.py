from abc import abstractmethod
from typing import TypeAlias

# The input type used by the aggregator. It is a list of embeddings.
AggregationInput: TypeAlias = torch.Tensor

# The output type used by the aggregator. It is a single embedding or alist.
AggregationOutput: TypeAlias = torch.Tensor


class Aggregator:
    """
    The abstract class for aggregators. It aggregates a list of embeddings into a single embedding. In general, only the
    NoAggregator class should return a list of embeddings (baseline). The aggregated embedding still keeps the dimension.
    """

    @abstractmethod
    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        The abstract class for aggregators. It aggregates a list of embeddings into a single embedding.
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The aggregated embedding with shape (num_aggregated, dim)
        """
        return embeddings


class NoAggregator(Aggregator):
    """
    The NoAggregator class does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
    the experiment baseline.
    """

    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        Returns the embeddings as is.
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The aggregated embedding with shape (num_aggregated, dim)
        """
        return embeddings
