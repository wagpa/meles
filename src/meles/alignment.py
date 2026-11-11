"""
Turning a detected face into the 112x112 crop the recognition models expect.

All three recognizers are ArcFace-family models trained on crops produced by the same five-point
similarity alignment, so one aligner serves all of them. Sharing it also keeps alignment from
becoming a difference between the models being compared.
"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

# The canonical ArcFace five-point reference for a 112x112 crop, in the landmark order YuNet
# emits: eye at image-left, eye at image-right, nose tip, mouth corner at image-left, mouth corner
# at image-right.
ARCFACE_TEMPLATE: NDArray[np.float32] = np.array(
    [
        [38.2946, 51.6963],
        [73.5318, 51.5014],
        [56.0252, 71.7366],
        [41.5493, 92.3655],
        [70.7299, 92.2041],
    ],
    dtype=np.float32,
)

# The side length of the crop the template is defined for, and the input size of the models.
TEMPLATE_SIZE = 112


def umeyama(source: NDArray[Any], target: NDArray[Any]) -> NDArray[np.float64]:
    """
    The least-squares similarity transform (rotation, uniform scale, translation) taking the
    source points onto the target points.

    Used instead of ``cv2.estimateAffinePartial2D`` because that runs RANSAC, which is a
    non-deterministic way to solve a five-point problem that has a closed form.

    :param source: The source points with shape (num_points, 2)
    :param target: The target points with shape (num_points, 2)
    :return: The transform as a (2, 3) matrix, ready for ``cv2.warpAffine``
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)

    source_mean, target_mean = source.mean(axis=0), target.mean(axis=0)
    source_centered, target_centered = source - source_mean, target - target_mean

    covariance = target_centered.T @ source_centered / len(source)
    rotation_u, singular, rotation_vt = np.linalg.svd(covariance)

    # Reflect the last axis when the naive solution would mirror rather than rotate.
    reflection = np.ones(2)
    if np.linalg.det(rotation_u) * np.linalg.det(rotation_vt) < 0:
        reflection[-1] = -1.0

    rotation = rotation_u @ np.diag(reflection) @ rotation_vt
    variance = (source_centered**2).sum() / len(source)
    scale = float((singular * reflection).sum() / variance)

    matrix = np.zeros((2, 3), dtype=np.float64)
    matrix[:, :2] = scale * rotation
    matrix[:, 2] = target_mean - scale * rotation @ source_mean
    return matrix


def align_face(image: NDArray[Any], landmarks: NDArray[Any], size: int = TEMPLATE_SIZE) -> NDArray[np.uint8]:
    """
    Warp a face onto the reference template using its five landmarks.

    :param image: The frame the face was detected in, BGR
    :param landmarks: The five detected landmarks with shape (5, 2)
    :param size: The side length of the resulting square crop
    :return: The aligned crop with shape (size, size, 3), BGR
    """
    template = ARCFACE_TEMPLATE.astype(np.float64) * size / TEMPLATE_SIZE
    matrix = umeyama(landmarks, template)
    warped = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    return warped.astype(np.uint8, copy=False)


def letterbox(image: NDArray[Any], size: int = TEMPLATE_SIZE) -> NDArray[np.uint8]:
    """
    Scale an image into a square, padding rather than stretching so the face keeps its aspect
    ratio. Used for frames the detector returned nothing for, which still have to be embedded.

    :param image: The image to fit, BGR
    :param size: The side length of the resulting square
    :return: The padded image with shape (size, size, 3), BGR
    """
    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized = cv2.resize(
        image,
        (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
    )

    padded = np.zeros((size, size, 3), dtype=image.dtype)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    padded[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return padded.astype(np.uint8, copy=False)
