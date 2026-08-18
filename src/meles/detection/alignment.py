"""
Turn a detected face into the 112x112 crop the embedding models were trained on.

All three embedders (SFace, ArcFace, GhostFaceNet) are ArcFace-family models trained on
crops produced by the same five-point similarity alignment, so one aligner serves all of
them. Sharing it also removes per-model alignment differences as a variable in the
experiment -- with DeepFace, each model brought its own detector configuration and cropping.
"""

from typing import Any

import cv2
import numpy as np
from numpy.typing import NDArray

from meles.common import FacialArea

# The canonical ArcFace five-point reference for a 112x112 crop, in the landmark order
# 'cv2.FaceDetectorYN' emits: eye at image-left, eye at image-right, nose tip, mouth corner
# at image-left, mouth corner at image-right. Fitting YuNet's landmarks onto this template
# reproduces 'cv2.FaceRecognizerSF.alignCrop' exactly (see src/tests/test_alignment.py),
# which is what confirms the two orderings agree.
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

# The side length of the crop the template is defined for.
TEMPLATE_SIZE = 112


def umeyama(source: NDArray[Any], target: NDArray[Any]) -> NDArray[np.float64]:
    """
    The least-squares similarity transform (rotation, uniform scale, translation) taking the
    source points onto the target points.

    Used instead of ``cv2.estimateAffinePartial2D`` because that runs RANSAC, which is a
    non-deterministic way to solve a five-point problem that has a closed form. The result is
    reproducible across runs, which matters for an experiment.

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

    :param image: The image the face was detected in, BGR
    :param landmarks: The five detected landmarks with shape (5, 2)
    :param size: The side length of the resulting square crop
    :return: The aligned crop with shape (size, size, 3), BGR
    """
    template = ARCFACE_TEMPLATE.astype(np.float64) * size / TEMPLATE_SIZE
    matrix = umeyama(landmarks, template)
    warped = cv2.warpAffine(image, matrix, (size, size), flags=cv2.INTER_LINEAR)
    # A no-op for the uint8 frames this is called with; it is here so the declared return type
    # holds for any input, since OpenCV's type stubs cannot express "same dtype as the input".
    return warped.astype(np.uint8, copy=False)


def letterbox(image: NDArray[Any], size: int = TEMPLATE_SIZE) -> NDArray[np.uint8]:
    """
    Scale an image into a square of the given side length, padding rather than stretching so
    that the face keeps its aspect ratio.

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
    return padded


def crop_face(image: NDArray[Any], area: FacialArea, size: int = TEMPLATE_SIZE) -> NDArray[np.uint8]:
    """
    Fall back to a plain bounding-box crop for a face without landmarks.

    :param image: The image the face was detected in, BGR
    :param area: The detected face
    :param size: The side length of the resulting square crop
    :return: The crop with shape (size, size, 3), BGR
    """
    height, width = image.shape[:2]
    x = max(0, min(area.x, width))
    y = max(0, min(area.y, height))
    box = image[y : y + max(0, min(area.h, height - y)), x : x + max(0, min(area.w, width - x))]
    return letterbox(image if box.size == 0 else box, size)


def prepare_face(image: NDArray[Any], area: FacialArea, size: int = TEMPLATE_SIZE) -> NDArray[np.uint8]:
    """
    The crop to embed for a detected face: landmark-aligned when possible, box-cropped
    otherwise.

    :param image: The image the face was detected in, BGR
    :param area: The detected face
    :param size: The side length of the resulting square crop
    :return: The crop with shape (size, size, 3), BGR
    """
    if area.landmarks is None:
        return crop_face(image, area, size)
    return align_face(image, area.landmarks, size)


def whole_frame(image: NDArray[Any], size: int = TEMPLATE_SIZE) -> tuple[NDArray[np.uint8], FacialArea]:
    """
    Treat a whole image as the face, for frames where detection finds nothing.

    This reproduces what DeepFace did with ``enforce_detection=False``: it substituted a
    region covering the entire frame, so every frame got an embedding. Keeping that behaviour
    matters because the anonymization methods deliberately destroy faces -- black bars
    especially -- and dropping those frames instead would silently change which frames the
    experiment scores.

    :param image: The image to treat as a face, BGR
    :param size: The side length of the resulting square crop
    :return: A tuple of the crop and the region standing in for the detection
    """
    height, width = image.shape[:2]
    return letterbox(image, size), FacialArea(0, 0, width, height, None, 0.0)
