"""
The (anonymized, clear) pairs a pix2pix de-anonymizer is trained on.

pix2pix needs aligned pairs, and this experiment has an unusual advantage there: the pairs do not
have to be collected, they can be *made*. Every clear face image is a target, and the input is
that same image with one of the pipeline's anonymizations applied to its facial area. The two are
pixel-aligned by construction, which is exactly the supervision the L1 term wants.

The anonymization is applied at the image's native resolution and only then resized, in that
order on purpose: the pipeline anonymizes a frame at whatever size the frame is, so anonymizing a
256x256 upsample of a 178x218 CelebA image would produce a blur radius (or a block size) the
model will never meet again at inference time.

The identities here are deliberately *not* the identities the experiment scores. The attack model
is that an attacker may train on any face data they like; training on YouTube Faces would make
the generator's prior contain the gallery, which is the same trap the fold split exists to avoid.
"""

import logging
import os
import random
from collections.abc import Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Literal

import cv2
import numpy as np
from torch import Tensor
from torch.utils.data import Dataset

from meles.anonymization import Anonymizer, resolve
from meles.common import FacialArea, Image
from meles.detection import Detector, default_detector
from meles.frame_aggregation.pix2pix import LOAD_SIZE, SIZE, to_tensor
from meles.training.sources import ImageSource

logger = logging.getLogger(__name__)

# What to do with an image the detector finds no face in.
OnMissing = Literal["whole_image", "clear"]


def augment(image: Image, crop: tuple[int, int], flip: bool, load_size: int, crop_size: int) -> Image:
    """
    pix2pix's ``resize_and_crop`` preprocessing, for one image of a pair.

    Upstream resizes to a **square** ``load_size`` -- aspect ratio is not preserved, so a 178x218
    CelebA image is squashed to 286x286 -- then takes a ``crop_size`` crop and optionally mirrors
    it. The crop and the flip are where pix2pix's only augmentation comes from.

    ``crop`` and ``flip`` are passed in rather than drawn here because both images of a pair must
    get the *same* ones; a pair cropped differently at each end is no longer aligned, and the L1
    term would be asking the generator to reconstruct a shifted face.

    :param image: The frame, BGR uint8
    :param crop: The (x, y) top-left corner of the crop, in the resized image
    :param flip: Whether to mirror horizontally
    :param load_size: The square edge to resize to before cropping
    :param crop_size: The square edge of the crop
    :return: The augmented frame, BGR uint8, ``crop_size`` square
    """
    # INTER_CUBIC to match upstream's PIL BICUBIC. The two kernels are not bit-identical -- PIL
    # antialiases on downscale where OpenCV does not -- but this is a training augmentation, and
    # the policy is what matters, not the last bit of the resample.
    resized = cv2.resize(image, (load_size, load_size), interpolation=cv2.INTER_CUBIC)
    x, y = crop
    cropped = resized[y : y + crop_size, x : x + crop_size]
    if flip:
        cropped = cropped[:, ::-1]
    # 'ascontiguousarray' because a reversed slice is a negative-stride view, which torch refuses.
    # The 'astype' is a no-op and is here only because OpenCV's stubs cannot express "same dtype
    # as the input", the same reason 'resample.py' carries one.
    return np.ascontiguousarray(cropped).astype(np.uint8, copy=False)


def augmentation_params(
    load_size: int = LOAD_SIZE, crop_size: int = SIZE, flip: bool = True
) -> tuple[tuple[int, int], bool]:
    """
    Draw one image pair's crop position and flip, i.e. upstream's ``get_params``.

    Drawn from the ``random`` module, as upstream does, which is also the one a ``DataLoader``
    reseeds per worker -- so the augmentation differs between workers instead of every worker
    replaying the same stream.

    :param load_size: The square edge the image is resized to
    :param crop_size: The square edge of the crop
    :param flip: Whether mirroring is enabled at all
    :return: The (x, y) crop corner and whether to mirror
    """
    span = max(0, load_size - crop_size)
    return (random.randint(0, span), random.randint(0, span)), flip and random.random() > 0.5


def as_image(item: Any) -> Image:
    """
    A dataset item as a frame in meles' own representation.

    :param item: A PIL image, an HWC array, or a ``(image, target)`` pair holding either
    :return: The frame, BGR uint8 -- BGR because that is what ``cv2`` and therefore every
        detector and anonymizer in meles expects
    :raises TypeError: If the item holds something that is not an image
    """
    # torchvision image datasets yield (image, target); the label is of no use here.
    if isinstance(item, tuple | list):
        item = item[0]

    if isinstance(item, np.ndarray):
        array = item
    elif hasattr(item, "convert"):
        # A PIL image. 'convert' also collapses greyscale and RGBA to three channels.
        array = np.asarray(item.convert("RGB"))
    else:
        raise TypeError(f"Expected a PIL image or an array, got {type(item).__name__}")

    if array.ndim != 3 or array.shape[2] != 3:
        raise TypeError(f"Expected a three-channel image, got shape {array.shape}")
    return cv2.cvtColor(array.astype(np.uint8, copy=False), cv2.COLOR_RGB2BGR)


class AnonymizedPairs(Dataset):
    """
    A clear-face dataset turned into pix2pix training pairs by anonymizing it.

    Yields ``(anonymized, clear)`` as the tensors the networks take: ``(3, size, size)``, RGB,
    in [-1, 1]. That order matches pix2pix's (condition, target) and the order the training loop
    unpacks.
    """

    def __init__(
        self,
        source: ImageSource,
        anonymizer: str | Anonymizer,
        detector: Detector = default_detector,
        size: int = SIZE,
        on_missing: OnMissing = "whole_image",
        areas: Sequence[FacialArea | None] | None = None,
        load_size: int | None = LOAD_SIZE,
        flip: bool = True,
    ):
        """
        :param source: The clear images, e.g. an ``meles.training.sources.ImageArchive`` over the
            CelebA zip. Any indexable, sized collection of RGB images does; if it is a torchvision
            dataset, construct it **without** ``transform``, since this class does the conversion
        :param anonymizer: The anonymization to train against, either a key in
            ``meles.anonymization.ANONYMIZERS`` or an anonymizer instance. This is the axis the
            whole thing turns on: one model is trained per anonymization method
        :param detector: The detector that locates the face to anonymize. The pipeline's shared
            YuNet by default, so that the model is trained against exactly the region the
            pipeline will anonymize at inference time
        :param size: The square edge length both frames are resized to, upstream's ``crop_size``
        :param on_missing: What to do when no face is detected. ``"whole_image"`` anonymizes the
            whole image; ``"clear"`` leaves it untouched, which is what ``anonymize_frame`` does
            to a frame but makes a *degenerate training pair* -- input equals target, teaching
            the generator to copy its input. Hence the different default here
        :param areas: Pre-detected facial areas, one per source item, as returned by
            :meth:`detect`. Passing them (or calling :meth:`detect` before the ``DataLoader`` is
            built) pays for detection once instead of once per epoch
        :param load_size: Upstream's ``load_size``: resize to this square edge and take a random
            ``size`` crop, which with the mirroring below is pix2pix's whole augmentation. None
            turns it off and resizes straight to ``size``, which is what a *validation* split
            wants -- a loss measured under a fresh random crop each epoch is not comparable
            between epochs
        :param flip: Whether to mirror horizontally at random, upstream's ``--no_flip`` inverted
        """
        self.source = source
        self.anonymizer = resolve(anonymizer) if isinstance(anonymizer, str) else anonymizer
        self.name = anonymizer if isinstance(anonymizer, str) else type(anonymizer).__name__
        self.detector = detector
        self.size = size
        self.on_missing = on_missing
        self.areas: list[FacialArea | None] | None = list(areas) if areas is not None else None
        self.load_size = load_size
        self.flip = flip

        # Counted rather than logged per image; a line per image is 160k lines on CelebA. Note
        # that with 'num_workers > 0' each worker counts only what it saw, so this is a hint
        # unless the areas were pre-detected -- which is the other reason to call 'detect'.
        self.missing = 0

    def __len__(self) -> int:
        return len(self.source)

    def image(self, index: int) -> Image:
        """
        The clear image at an index.

        :param index: The index into the source dataset
        :return: The frame, BGR uint8
        """
        return as_image(self.source[index])

    def area(self, image: Image, index: int | None = None) -> FacialArea | None:
        """
        Where the face is in an image.

        :param image: The frame to search, BGR
        :param index: Its index, used to look up a pre-detected area when there is one
        :return: The largest detected face, or None if there is none
        """
        if self.areas is not None and index is not None:
            return self.areas[index]
        return self.detector.detect_best(image)

    def anonymize(self, image: Image, index: int | None = None) -> Image:
        """
        Apply the selected anonymization to an image's facial area.

        This is the input half of a training pair. It goes through the same detector and the same
        anonymizer instance as ``workflow/scripts/anonymize_frame.py``, which is the point: the
        model has to learn the inverse of the anonymization the sweep actually produces, not of a
        near-enough imitation of it.

        :param image: The clear frame, BGR uint8
        :param index: Its index, used to look up a pre-detected area when there is one
        :return: The anonymized frame, BGR uint8. A copy, except under ``NoAnonymizer`` and under
            ``on_missing="clear"``, which both have nothing to change
        """
        area = self.area(image, index)
        if area is None:
            self.missing += 1
            if self.on_missing == "clear":
                return image
            # No face located, so the whole image is treated as the face. That keeps the pair
            # non-degenerate; see the 'on_missing' parameter.
            height, width = image.shape[:2]
            area = FacialArea(x=0, y=0, w=width, h=height)
        return self.anonymizer.apply(image, area)

    def detect(self, progress: bool = True, workers: int | None = None) -> list[FacialArea | None]:
        """
        Detect every image's face up front and remember the result.

        Worth doing before the ``DataLoader`` is built. Detection is otherwise repeated every
        epoch for every image, and doing it here -- in the parent process, before the workers
        fork -- both pays for it once and gives an honest miss count, which the per-worker
        counter cannot.

        :param progress: Whether to show a progress bar
        :param workers: How many threads to detect on, defaulting to the core count capped at 8.
            Threads rather than processes because ``YuNetDetector`` already keeps one model per
            thread (``cv::dnn::Net`` is not thread-safe) and the areas have to come back into
            this process anyway. Measured at 431 images/s on one thread against 1545 on eight,
            i.e. eight minutes down to two over CelebA
        :return: The detected areas, one per source item, None where there was no face
        """
        from tqdm import tqdm

        if workers is None:
            workers = min(8, os.cpu_count() or 1)

        # Detected against the *clear* image, which is the only place the face is reliably
        # findable -- a black bar or heavy pixelation is exactly what a detector cannot find.
        def locate(index: int) -> FacialArea | None:
            return self.detector.detect_best(self.image(index))

        indices = range(len(self))
        if workers > 1:
            with ThreadPoolExecutor(workers) as pool:
                located: Iterable[FacialArea | None] = pool.map(locate, indices)
                if progress:
                    located = tqdm(located, total=len(self), desc=f"Detecting ({self.name})", mininterval=5.0)
                self.areas = list(located)
        else:
            stepped: Any = tqdm(indices, desc=f"Detecting ({self.name})", mininterval=5.0) if progress else indices
            self.areas = [locate(index) for index in stepped]

        self.missing = sum(area is None for area in self.areas)
        logger.info(
            "Detected faces for the training pairs",
            extra={"anonymizer": self.name, "len": len(self.areas), "no_face": self.missing},
        )
        return self.areas

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        :param index: The index into the source dataset
        :return: The (anonymized, clear) pair, each ``(3, size, size)`` RGB in [-1, 1]
        """
        clear = self.image(index)
        anonymized = self.anonymize(clear, index)

        # One draw for the pair, applied to both. Cropping or mirroring the two ends
        # independently would break the pixel alignment the L1 term depends on.
        if self.load_size is not None:
            crop, flip = augmentation_params(self.load_size, self.size, self.flip)
            clear = augment(clear, crop, flip, self.load_size, self.size)
            anonymized = augment(anonymized, crop, flip, self.load_size, self.size)

        return to_tensor(anonymized, self.size), to_tensor(clear, self.size)
