import torch
from meles.aggregation.aggregator import Aggregator, Embeddings


class QualityAggregator(Aggregator):
    """
    The QualityAggregator pools all embeddings into a single embedding by weighting the embeddings by their quality.
    """

    def __call__(self, embeddings: Embeddings) -> Embeddings:
        # TODO: implement the weighting using some model. But how to get the quality? -> needs original frames?
        return torch.mean(embeddings, dim=0, keepdim=True)
