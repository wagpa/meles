"""
Moving frames between OpenCV's world and the network's.

Three conversions happen here, and all three are easy to get silently wrong: the pipeline speaks
**BGR** everywhere while the networks are trained on RGB, the layout goes from HWC to CHW, and the
decoder ends on a ``tanh`` so the range is **[-1, 1]** rather than [0, 1].
"""

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

from meles.alignment import TEMPLATE_SIZE


def default_device() -> torch.device:
    """
    :return: The CUDA device if there is one, otherwise the CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def interpolation(source: tuple[int, int], target: tuple[int, int]) -> int:
    """
    The resampling kernel to resize with: area averaging when shrinking, linear when growing.

    :param source: The (height, width) being resized from
    :param target: The (height, width) being resized to
    :return: The OpenCV interpolation flag
    """
    return cv2.INTER_AREA if target[0] * target[1] < source[0] * source[1] else cv2.INTER_LINEAR


def to_tensor(image: NDArray[np.uint8], size: int = TEMPLATE_SIZE) -> Tensor:
    """
    A frame as the tensor the networks take.

    :param image: The frame, BGR uint8
    :param size: The square edge length to resize to
    :return: A (3, size, size) float32 tensor in [-1, 1], RGB
    """
    height, width = image.shape[:2]
    resized = image
    if (height, width) != (size, size):
        resized = cv2.resize(image, (size, size), interpolation=interpolation((height, width), (size, size)))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float() / 127.5 - 1.0


def to_image(tensor: Tensor, size: tuple[int, int] | None = None) -> NDArray[np.uint8]:
    """
    The inverse of :func:`to_tensor`.

    Values are clamped before they are scaled back: the decoder's ``tanh`` cannot leave [-1, 1]
    but a target tensor can, and uint8 would wrap it round into the opposite brightness rather
    than saturate.

    :param tensor: A (3, H, W) tensor in [-1, 1], RGB, on any device
    :param size: The (height, width) to resize back to, or None to keep the tensor's own size
    :return: The frame, BGR uint8
    """
    array = ((tensor.detach().clamp(-1.0, 1.0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).round()
    bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if size is None or bgr.shape[:2] == size:
        return bgr
    return cv2.resize(bgr, (size[1], size[0]), interpolation=interpolation(bgr.shape[:2], size))
