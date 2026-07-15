import numpy as np
from meles.aggregation.aggregator import Aggregator, Embeddings
from meles.common import Frames
from meles.recognizer.recognizer import Recognizer


class MeanPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __init__(self, recognizer: Recognizer):
        super().__init__(recognizer)

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = self.recognizer.embed(frames)
        return np.mean(embeddings, axis=0, keepdims=True).tolist()

    def name(self) -> str:
        return "mean"


class MedianPoolingAggregator(Aggregator):
    """
    The MeanPoolingAggregator pools all embeddings into a single embedding using the column wise mean.
    """

    def __init__(self, recognizer: Recognizer):
        super().__init__(recognizer)

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = self.recognizer.embed(frames)
        return np.median(embeddings, axis=0, keepdims=True).tolist()

    def name(self) -> str:
        return "median"


class MaxPoolingAggregator(Aggregator):
    """
    The MaxPoolingAggregator pools all embeddings into a single embedding using the column wise max.
    """

    def __init__(self, recognizer: Recognizer):
        super().__init__(recognizer)

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = self.recognizer.embed(frames)
        return np.max(embeddings, axis=0, keepdims=True).tolist()

    def name(self) -> str:
        return "max"


class MinPoolingAggregator(Aggregator):
    """
    The MinPoolingAggregator pools all embeddings into a single embedding using the column wise min.
    """

    def __init__(self, recognizer: Recognizer):
        super().__init__(recognizer)

    def embed(self, frames: Frames) -> Embeddings:
        embeddings = self.recognizer.embed(frames)
        return np.min(embeddings, axis=0, keepdims=True).tolist()

    def name(self) -> str:
        return "min"
