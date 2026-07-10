from abc import abstractmethod
from meles.common import Embeddings, Frames
from meles.recognizer.recognizer import Recognizer


class Aggregator(Recognizer):
    """
    The abstract class for an aggregator. It uses the internal recognizer to embed frames and then aggregates the embeddings.
    By default, the aggregator does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
    the experiment baseline.
    """

    recognizer: Recognizer

    def __init__(self, recognizer: Recognizer):
        self.recognizer = recognizer

    def metric(self) -> str:
        return self.recognizer.metric()

    @abstractmethod
    def name(self) -> str:
        return "base"

    @abstractmethod
    def embed(self, frames: Frames) -> Embeddings:
        return self.recognizer.embed(frames)


class NoAggregator(Aggregator):
    """
    The NoAggregator class does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
    the experiment baseline.
    """

    def __init__(self, recognizer: Recognizer):
        super().__init__(recognizer)

    def embed(self, frames: Frames) -> Embeddings:
        return self.recognizer.embed(frames)

    def name(self) -> str:
        return "no"
