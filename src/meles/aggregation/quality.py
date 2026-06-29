import numpy as np
from meles.aggregation.aggregator import Aggregator, Embeddings
from meles.common import Frames
from meles.recognizer.recognizer import Recognizer


class QualityAggregator(Aggregator):
    """
    The QualityAggregator pools all embeddings into a single embedding by weighting the embeddings by their quality.
    """

    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        # TODO: implement the weighting using some model. But how to get the quality? -> needs original frames?
        embeddings = recognizer.embed(frames)
        return np.mean(embeddings, axis=0, keepdims=True).tolist()
