import torch
from meles.aggregation.aggregator import Aggregator, Embeddings


class ClusteringAggregator(Aggregator):
    """
    The ClusteringAggregator pools all embeddings into a single embedding while discarding outliers based on clustering.
    """

    def __call__(self, embeddings: Embeddings) -> Embeddings:
        # TODO: implement me!
        return torch.mean(embeddings, dim=0, keepdim=True)
