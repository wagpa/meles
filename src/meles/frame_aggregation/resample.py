import cv2
import numpy as np

from meles.common import Image, Images
from meles.frame_aggregation.aggregator import FrameAggregator


class ResampleFrameAggregator(FrameAggregator):
    """
    De-anonymizes frame-wise by resampling: every frame is scaled down and straight back up.

    This aggregates nothing -- one frame goes in, one frame comes out -- which is why it returns as
    many frames as it was given rather than collapsing the group.

    It is SEBA's 'resample' de-anonymization (kit-ps/seba, from the Fantomas paper). Scaling down
    with INTER_AREA averages each neighbourhood into one pixel and scaling back up interpolates
    between them, which turns the hard block edges pixelation leaves behind into a smooth face
    again. Nothing the anonymization destroyed is recovered; what changes is how much of what
    survived the recognizer can still make use of.

    SEBA fits 'factor' per dataset, searching 0.1 to 0.9 for the best SSIM against the clear frames.
    That is deliberately not done here: the clear frames are what the gallery holds, so fitting
    against them is fitting against the answer. The factor is a constant of the method instead, and
    so part of the registry key -- sweep it by registering more entries, not by learning it.
    """

    factor: float
    interpolation: int

    def __init__(self, factor: float = 0.5, interpolation: int = cv2.INTER_AREA):
        super().__init__()
        self.factor = factor
        self.interpolation = interpolation

    def resample(self, image: Image) -> Image:
        height, width = image.shape[:2]
        # At least one pixel each way: a frame small enough to round to zero would fail in resize.
        reduced = cv2.resize(
            image,
            (max(1, int(width * self.factor)), max(1, int(height * self.factor))),
            interpolation=cv2.INTER_AREA,
        )
        restored = cv2.resize(reduced, (width, height), interpolation=self.interpolation)
        # A no-op for the uint8 frames this is called with; it is here so the declared return type
        # holds for any input, since OpenCV's type stubs cannot express "same dtype as the input".
        return restored.astype(np.uint8, copy=False)

    def aggregate(self, images: Images) -> Images:
        return [self.resample(image) for image in images]


class BicubicFrameAggregator(ResampleFrameAggregator):
    """Resamples back up with bicubic interpolation, SEBA's 'bicubic' de-anonymization."""

    def __init__(self, factor: float = 0.5):
        super().__init__(factor=factor, interpolation=cv2.INTER_CUBIC)
