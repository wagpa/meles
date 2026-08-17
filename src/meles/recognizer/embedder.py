from abc import abstractmethod, ABC
from meles.common import Frames, OptEmbeddings


class Embedder(ABC):
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
    def embed(self, frames: Frames) -> OptEmbeddings:
        """
        The abstract class for recognizers. It embeds a list f frames into embeddings.

        Frame-level embedders return one embedding per input frame (None where no face was found),
        so the result lines up with the input row-by-row. Video-level embedders instead treat the
        whole sequence as a single video and return one embedding per clip they sampled from it,
        which may be any number — 'embed_video.py' explodes the result into one row per embedding.

        :param frames: The list of frames to embed with shape (num_frames, frame_dim)
        :return: The generated embeddings with shape (num_embeddings, embedding_dim)
        """
        pass
