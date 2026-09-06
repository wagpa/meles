from abc import abstractmethod, ABC
from meles.common import Images


class FrameAggregator(ABC):
    """
    The abstract class for an aggregator. It aggregates frames. By default, the aggregator does not aggregate the frames.
    It simply returns the frames as is. This will be used as the experiment baseline.
    """

    def __init__(self):
        pass

    @abstractmethod
    def aggregate(self, images: Images) -> Images:
        return images


class NoFrameAggregator(FrameAggregator):
    """
    The NoAggregator class does not aggregate the frames. It simply returns the frames as is. This will be used as
    the experiment baseline.
    """

    def aggregate(self, images: Images) -> Images:
        return images
