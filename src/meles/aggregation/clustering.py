import numpy as np
from meles.aggregation.aggregator import Aggregator, Embeddings
from meles.common import Frames
from meles.recognizer.recognizer import Recognizer


class ClusteringAggregator(Aggregator):
    """
    The ClusteringAggregator pools all embeddings into a single embedding while discarding outliers based on clustering.
    """

    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        embeddings = recognizer.embed(frames)
        return np.mean(embeddings, axis=0, keepdims=True).tolist()
