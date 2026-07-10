import numpy as np
from meles.aggregation.aggregator import Aggregator, Embeddings
from meles.common import Frames
from meles.recognizer.recognizer import Recognizer


class ClusteringAggregator(Aggregator):
    """
    The ClusteringAggregator pools all embeddings into a single embedding while discarding outliers based on clustering.
    """

    def __init__(self, recognizer: Recognizer):
        super().__init__(recognizer)

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = self.recognizer.embed(frames)
        return np.mean(embeddings, axis=0, keepdims=True).tolist()

    def name(self) -> str:
        return "cluster"
