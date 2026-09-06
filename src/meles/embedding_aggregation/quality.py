import numpy as np
from meles.embedding_aggregation.aggregator import EmbeddingAggregator, Embeddings


class QualityEmbeddingAggregator(EmbeddingAggregator):
    """
    The QualityAggregator pools all embeddings into a single embedding by weighting the embeddings by their quality.
    """

    def __init__(self):
        super().__init__()

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        # TODO: implement the weighting using some model. But how to get the quality? -> needs original frames?
        return np.mean(embeddings, axis=0, keepdims=True).tolist()
