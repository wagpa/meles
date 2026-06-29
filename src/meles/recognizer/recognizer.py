from abc import abstractmethod, ABC
from meles.common import Frames, Embeddings


class Recognizer(ABC):
    """
    The abstract class for recognizers. It embeds a list of frames into embeddings and handles the recognition.
    """

    @abstractmethod
    def metric(self) -> str:
        """
        The abstract class for recognizers. It gives the metric name (e.g., 'cosine') used by the recognizer to measure
        similarity between embeddings.
        :return: The metric used by the recognizer
        """
        pass

    @abstractmethod
    def embed(self, frames: Frames) -> Embeddings:
        """
        The abstract class for recognizers. It embeds a list f frames into embeddings.
        :param frames: The list of frames to embed with shape (num_frames, frame_dim)
        :return: The generated embeddings with shape (num_frames, embedding_dim)
        """
        pass
