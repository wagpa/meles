from abc import abstractmethod, ABC
from meles.common import Embeddings, Frames
from meles.recognizer.recognizer import Recognizer


# TODO maybe rename Aggregator to AggregatingRecognizer taking the recognizer in the constructor? It then also implements the classifier
class Aggregator(ABC):
    """
    The abstract class for aggregators. It aggregates a list of embeddings into a single embedding. In general, only the
    NoAggregator class should return a list of embeddings (baseline). The aggregated embedding still keeps the dimension.
    """

    @abstractmethod
    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        """
        Embeds and aggregates the frames into a list of embeddings. In general, the resulting list will only contain a
        single embedding.
        :param frames: The list of frames to process shape (num_frames, dim_frame)
        :param recognizer: The recognizer to use for embedding the frames
        :return: The aggregated embeddings with shape (num_aggregated, dim_aggregated)
        """


class NoAggregator(Aggregator):
    """
    The NoAggregator class does not aggregate the embeddings. It simply returns the embeddings as is. This will be used as
    the experiment baseline.
    """

    def aggregate(self, frames: Frames, recognizer: Recognizer) -> Embeddings:
        return recognizer.embed(frames)
