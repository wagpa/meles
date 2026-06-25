import torch
from meles.aggregation.aggregator import Aggregator, AggregationInput, AggregationOutput


class QualityAggregator(Aggregator):
    """
    The QualityAggregator pools all embeddings into a single embedding by weighting the embeddings by their quality.
    """

    def __call__(self, embeddings: AggregationInput) -> AggregationOutput:
        """
        Calculate the embedding using the quality.
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The pooled embedding with shape (1, dim)
        """
        # TODO: implement the weighting using some model. But how to get the quality? -> needs original frames?
        return torch.mean(embeddings, dim=0, keepdim=True)
