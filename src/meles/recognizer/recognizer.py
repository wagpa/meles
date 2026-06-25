from abc import abstractmethod
from typing import TypeAlias

# The input type used by the recognizer. It is a list of frames.
RecognizerInput: TypeAlias = torch.Tensor

# The output type used by the recognizer. It is list of embeddings.
RecognizerOutput: TypeAlias = torch.Tensor


class Recognizer:
    """
    The abstract class for recognizers. It embeds a list of frames into embeddings and handles the recognition.
    """

    # TODO decide on interface!
    @abstractmethod
    def __call__(self, frames: RecognizerInput) -> RecognizerOutput:
        """
        The abstract class for aggregators. It aggregates a list of embeddings into a single embedding.
        :param embeddings: The list of embeddings to pool with shape (num_frames, dim)
        :return: The aggregated embedding with shape (num_aggregated, dim)
        """
        return embeddings
