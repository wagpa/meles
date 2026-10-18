"""
The ``(N anonymized frames, clear target)`` items the :class:`meles.frame_aggregation.facial.Facial`
aggregator is trained on.

``Facial`` maps ``B x N x 3 x 112 x 112`` onto ``B x 3 x 112 x 112``, so one training item is a
whole identity: several anonymized observations of a person, and the single clear image the
decoder should produce from them. That is the shape of the attack the experiment measures -- an
attacker holds many anonymized frames and wants one face back -- which is why the item is keyed by
identity rather than by frame, as it is for pix2pix in :mod:`meles.training.pairs`.

Two asymmetries with the inference pipeline are deliberate:

- The **face is located on the clear frame** and the resulting area is used to both anonymize and
  align. ``embed_frame`` detects on the anonymized frame, but a black bar is precisely what a
  detector cannot find, so detecting there would hand the model letterboxed whole frames instead
  of aligned faces for exactly the anonymization that matters most. :class:`AnonymizedPairs` makes
  the same trade for the same reason.
- The **anonymization is applied at the frame's native resolution**, before alignment. The
  pipeline anonymizes a frame at whatever size it is, so blurring a 112x112 crop would produce a
  radius the model never meets again at inference time.

Unlike pix2pix, this trains on the *same* dataset the experiment scores, because the model needs
many frames of one person and that is what YouTube Faces has. The identity split below is what
keeps that honest: hold identities out, not frames, or the decoder's prior contains the gallery --
the trap ``meles.training.pairs`` avoids by training on CelebA instead.
"""

import logging
import random
from collections.abc import Sequence
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from torch import Tensor, stack
from torch.utils.data import Dataset

from meles.anonymization import Anonymizer, resolve
from meles.common import FacialArea, Image
from meles.detection import TEMPLATE_SIZE, Detector, default_detector, prepare_face
from meles.frame_aggregation.pix2pix import to_tensor

logger = logging.getLogger(__name__)


def split_identities(identities: Sequence[str], validation: float = 0.1, seed: int = 0) -> tuple[list[str], list[str]]:
    """
    Split identities -- never frames -- into a train and a validation set.

    Splitting by frame would put the same face on both sides, and the validation loss would then
    measure how well the decoder memorised that face rather than whether it generalises.

    :param identities: Every identity available
    :param validation: The fraction held out for validation
    :param seed: The shuffle seed, so the split is reproducible
    :return: The (train, validation) identity lists, both sorted
    """
    shuffled = sorted(identities)
    random.Random(seed).shuffle(shuffled)
    held_out = int(round(len(shuffled) * validation))
    return sorted(shuffled[held_out:]), sorted(shuffled[:held_out])


class FacialFrames(Dataset):
    """
    One item per identity: ``(anonymized, clear)`` where ``anonymized`` is ``(N, 3, size, size)``
    and ``clear`` is ``(size, size)`` -- the order ``Facial.forward`` takes and the training loop
    unpacks.

    The default ``DataLoader`` collate stacks these into ``(B, N, 3, size, size)`` and
    ``(B, 3, size, size)``, so no custom ``collate_fn`` is needed.
    """

    def __init__(
        self,
        structure: pd.DataFrame,
        targets: pd.DataFrame,
        root: Path,
        anonymizer: str | Anonymizer,
        identities: Sequence[str] | None = None,
        frames: int = 8,
        size: int = TEMPLATE_SIZE,
        detector: Detector = default_detector,
        exclude_target_video: bool = True,
        flip: bool = True,
        deterministic: bool = False,
        seed: int = 0,
    ):
        """
        :param structure: One row per frame, as ``data/ytfaces.parquet`` holds it: ``identity``,
            ``video`` and ``path``. These are the observations the attacker gets
        :param targets: One row per identity, as ``data/ytfaces_targets.parquet`` holds it (see
            ``src/notebooks/07_target_selection.ipynb``): the frame the decoder should produce
        :param root: What the ``path`` column is relative to, i.e. the ``data`` directory
        :param anonymizer: The anonymization to train against, a key in
            ``meles.anonymization.ANONYMIZERS`` or an instance. One model is trained per method
        :param identities: Which identities this split covers, from :func:`split_identities`.
            None uses every identity that has a target
        :param frames: How many anonymized frames make up one item, ``N``. Fixed, because a batch
            has to stack them; ``FacialAggregator`` itself is indifferent to ``N``
        :param size: The square edge of every crop, the 112 the encoder was trained on
        :param detector: The detector that locates the face. The pipeline's shared YuNet, so the
            model trains against the region the pipeline anonymizes
        :param exclude_target_video: Whether to draw the input frames from the identity's *other*
            videos. Keeps the model from reconstructing the target off its own shot's background
            and lighting rather than off the face
        :param flip: Whether to mirror at random. One draw per item, applied to the target and all
            ``N`` inputs together -- mirroring only one end would ask the decoder for a face its
            inputs do not show
        :param deterministic: Whether to draw the same frames and skip the flip every epoch. What
            a *validation* split wants: a loss measured over freshly drawn frames each epoch is
            not comparable between epochs
        :param seed: Seeds the per-item draw under ``deterministic``
        """
        self.anonymizer = resolve(anonymizer) if isinstance(anonymizer, str) else anonymizer
        self.name = anonymizer if isinstance(anonymizer, str) else type(anonymizer).__name__
        self.root = Path(root)
        self.frames = frames
        self.size = size
        self.detector = detector
        self.flip = flip
        self.deterministic = deterministic
        self.seed = seed

        self.target: dict[str, str] = dict(zip(targets["identity"], targets["path"]))

        # The frames each identity's inputs are drawn from, resolved once so '__getitem__' only
        # samples from a list. An identity with no target cannot be trained on at all.
        wanted = set(self.target) if identities is None else set(identities) & set(self.target)
        target_video = dict(zip(targets["identity"], targets["video"]))
        target_path = set(self.target.values())

        self.pool: dict[str, list[str]] = {}
        for identity, group in structure[structure["identity"].isin(wanted)].groupby("identity"):
            if exclude_target_video:
                group = group[group["video"] != target_video[identity]]
            paths = [str(path) for path in group["path"] if path not in target_path]
            if paths:
                self.pool[str(identity)] = paths

        self.identities = sorted(self.pool)
        dropped = sorted(wanted - set(self.identities))
        if dropped:
            logger.warning(
                "Identities have no input frames left and were dropped",
                extra={"anonymizer": self.name, "dropped": len(dropped), "first": dropped[:5]},
            )

        # Detection is otherwise repeated for every frame in every epoch. Filled lazily and held
        # per instance, so with 'num_workers > 0' each worker warms its own copy -- which only
        # pays off across epochs, i.e. with 'persistent_workers=True'.
        self._areas: dict[str, FacialArea | None] = {}
        self.missing = 0

    def __len__(self) -> int:
        return len(self.identities)

    def read(self, path: str) -> Image:
        """
        A frame from the dataset.

        :param path: The ``path`` column's value, relative to ``root``
        :return: The frame, BGR uint8
        :raises FileNotFoundError: If the frame is not readable
        """
        image = cv2.imread(str(self.root / path))
        if image is None:
            raise FileNotFoundError(f"Could not read frame {self.root / path}")
        # A no-op for the uint8 frames 'imread' returns; it is here only so the declared return
        # type holds, since OpenCV's stubs cannot express "uint8" for 'imread'. Same reason
        # 'pairs.augment' and 'resample.py' each carry one.
        return image.astype(np.uint8, copy=False)

    def area(self, path: str, image: Image) -> FacialArea:
        """
        Where the face is, detected on the **clear** frame and remembered.

        :param path: The frame's path, the cache key
        :param image: The clear frame, BGR
        :return: The largest detected face, or the whole frame when there is none
        """
        if path not in self._areas:
            self._areas[path] = self.detector.detect_best(image)

        area = self._areas[path]
        if area is None:
            self.missing += 1
            # No face located, so the whole frame stands in for it -- what 'whole_frame' does
            # downstream, and what keeps the item non-degenerate under 'on_missing' in pix2pix.
            height, width = image.shape[:2]
            return FacialArea(x=0, y=0, w=width, h=height)
        return area

    def crop(self, path: str, anonymize: bool, flip: bool) -> Image:
        """
        One aligned crop, anonymized or clear.

        :param path: The frame to load
        :param anonymize: Whether to anonymize the facial area first
        :param flip: Whether to mirror the crop
        :return: The crop, ``(size, size, 3)`` BGR uint8
        """
        clear = self.read(path)
        area = self.area(path, clear)

        # Anonymize at the frame's native resolution, then align: the other order would fit the
        # blur radius or the block size to a 112x112 crop the pipeline never produces.
        image = self.anonymizer.apply(clear, area) if anonymize else clear
        cropped = prepare_face(image, area, self.size)
        return np.ascontiguousarray(cropped[:, ::-1]) if flip else cropped

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor]:
        """
        :param index: The index into this split's identities
        :return: The ``(anonymized, clear)`` item: ``(N, 3, size, size)`` and ``(3, size, size)``,
            both RGB in [-1, 1]
        """
        identity = self.identities[index]
        pool = self.pool[identity]

        # Seeded by index under 'deterministic' so a validation item is the same every epoch.
        # Otherwise drawn from the 'random' module, which a DataLoader reseeds per worker, so the
        # workers draw different frames instead of replaying one stream.
        draw = random.Random(self.seed + index) if self.deterministic else random
        # 'choices' rather than 'sample': an identity may hold fewer frames than N, and repeating
        # an observation is a truthful item -- the attacker simply saw less.
        chosen = draw.sample(pool, self.frames) if len(pool) >= self.frames else draw.choices(pool, k=self.frames)

        flip = self.flip and not self.deterministic and draw.random() > 0.5
        anonymized = [to_tensor(self.crop(path, anonymize=True, flip=flip), self.size) for path in chosen]
        clear = to_tensor(self.crop(self.target[identity], anonymize=False, flip=flip), self.size)
        return stack(anonymized), clear
