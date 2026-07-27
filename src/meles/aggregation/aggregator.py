from abc import abstractmethod, ABC
from meles.common import Embeddings


class Aggregator(ABC):
    """
    The abstract class for an aggregator. It uses the internal recognizer to embed frames and then aggregates the embeddings.
    By default, the aggregator does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
    the experiment baseline.
    """

    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return embeddings


class NoAggregator(Aggregator):
    """
    The NoAggregator class does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
    the experiment baseline.
    """

    def aggregate(self, embeddings: Embeddings) -> Embeddings:
        return embeddings
