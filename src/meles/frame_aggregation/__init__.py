from meles.frame_aggregation.aggregator import (
    FrameAggregator,
    NoFrameAggregator,
)
from meles.frame_aggregation.pix2pix import (
    Pix2PixFrameAggregator
)
from meles.frame_aggregation.resample import (
    ResampleFrameAggregator,
    BicubicFrameAggregator,
)

__all__ = [
    "FrameAggregator",
    "NoFrameAggregator",
    "Pix2PixFrameAggregator",
    "ResampleFrameAggregator",
    "BicubicFrameAggregator",
]