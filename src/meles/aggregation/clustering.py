import torch
from meles.aggregation.aggregator import Aggregator, AggregationInput, AggregationOutput


class ClusteringAggregator(Aggregator):
    """
    The ClusteringAggregator pools all embeddings into a single embedding while discarding outliers based on clustering.
    """

    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        Calculate the embedding by dropping outliers.
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The pooled embedding with shape (1, dim)
        """
        # TODO: implement me!
        return torch.mean(embeddings, dim=0, keepdim=True)
