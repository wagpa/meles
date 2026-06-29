import numpy as np
from meles.aggregation.aggregator import Aggregator, Embeddings
from meles.common import Frames
from meles.recognizer.recognizer import Recognizer


class MeanPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        embeddings = recognizer.embed(frames)
        return np.mean(embeddings, axis=0, keepdims=True).tolist()


class MaxPoolingAggregator(Aggregator):
    """
    The MaxPoolingAggregator pools all embeddings into a single embedding using the column wise max.
    """

    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        embeddings = recognizer.embed(frames)
        return np.max(embeddings, axis=0, keepdims=True).tolist()


class MinPoolingAggregator(Aggregator):
    """
    The MinPoolingAggregator pools all embeddings into a single embedding using the column wise min.
    """

    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        embeddings = recognizer.embed(frames)
        return np.min(embeddings, axis=0, keepdims=True).tolist()
