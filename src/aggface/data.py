"""
The items AggFace is trained on: ``(N anonymized frames, one clear target)`` per identity.

The model maps ``B x N x 3 x 112 x 112`` onto ``B x 3 x 112 x 112``, so one item is a whole
person - several anonymized observations of them, and the single clear image the decoder should
produce from those. That is the shape of the attack the experiment measures.

Two choices about how an item is built are worth knowing, because both differ from the obvious:

- The face is located on the **clear** frame, and that area is used both to anonymize and to
  align. Detecting on the anonymized frame instead would hand the model a crop that is not on the
  face for exactly the methods that matter most: a black bar is precisely what a detector cannot
  find.
- The anonymization is applied at the frame's **native resolution**, before alignment. Blurring a
  112x112 crop instead would produce a radius the model never meets again at inference.

One model covers every anonymization, and they are introduced as a curriculum - clear frames
first, then progressively harsher ones. The hardest of them removes the face outright, so a model
shown black bars from the first epoch has no signal to learn an identity from, while one that
already rebuilds faces from blurred input has something to fall back on.

An item carries a single anonymization, drawn per identity rather than per frame, because that is
how the experiment observes a condition: a sweep leg pins one method and every frame it sees was
anonymized that way.
"""

import logging
import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from torch import Tensor, stack
from torch.utils.data import Dataset

from aggface.tensors import to_tensor
from meles.alignment import TEMPLATE_SIZE, align_face, letterbox
from meles.anonymization import Anonymizer, resolve
from meles.detection import Detector, FacialArea, default_detector

logger = logging.getLogger(__name__)

# The order the anonymizations are introduced in, easiest first. Stage 0 trains on 'none' alone,
# which is the model learning to rebuild a face from several clear views of it before any of them
# is degraded; each later stage keeps everything before it and adds one method. 'blackbar' is last
# because it removes the face rather than obscuring it: there is no inverse to learn, only a prior
# over what the other frames imply.
CURRICULUM: tuple[str, ...] = ("none", "gaussianblur", "pixelation", "blackbar")


class Items(Dataset):
    """
    One item per identity: ``(anonymized, clear, method)``, where ``anonymized`` is
    ``(N, 3, size, size)``, ``clear`` is ``(3, size, size)`` and ``method`` is the index into the
    curriculum of the anonymization the item was built with.

    The default collate stacks these into ``(B, N, 3, size, size)``, ``(B, 3, size, size)`` and
    ``(B,)``, so no custom ``collate_fn`` is needed.

    ``method`` indexes the **whole** curriculum rather than the active prefix, so the same number
    means the same anonymization at every stage - which is what makes a per-method loss comparable
    across stages.
    """

    def __init__(
        self,
        structure: pd.DataFrame,
        targets: pd.DataFrame,
        root: Path,
        identities: list[str] | None = None,
        curriculum: tuple[str, ...] = CURRICULUM,
        stage: int = 0,
        frames: int = 8,
        size: int = TEMPLATE_SIZE,
        detector: Detector = default_detector,
        flip: bool = True,
        deterministic: bool = False,
        all_methods: bool = False,
        seed: int = 0,
    ) -> None:
        """
        :param structure: One row per frame, with ``identity``, ``video`` and ``path``. These are
            the observations the attacker gets
        :param targets: One row per identity, with ``identity``, ``video`` and ``path``: the frame
            the decoder should produce
        :param root: What the ``path`` column is relative to, the ``data`` directory
        :param identities: Which identities this split covers, or None for every identity that
            has a target
        :param curriculum: The anonymizations in the order they are introduced. A one-element
            sequence trains against a single method and nothing else
        :param stage: How far into the curriculum to start. Stage ``s`` draws uniformly from the
            first ``s + 1`` methods
        :param frames: How many anonymized frames make up one item, ``N``. Fixed, because a batch
            has to stack them; the aggregator itself is indifferent to it
        :param size: The square edge of every crop
        :param detector: The detector that locates the face, the same one the pipeline uses
        :param flip: Whether to mirror at random. One draw per item, applied to the target and all
            N inputs together, because mirroring one end would ask the decoder for a face its
            inputs do not show
        :param deterministic: Whether to draw the same frames, the same anonymization and no flip
            every epoch. What a validation split wants, since a loss measured over freshly drawn
            frames is not comparable between epochs
        :param all_methods: Whether to yield every identity under every active anonymization
            instead of one drawn per identity, making the dataset ``identities x methods`` long.
            The other half of what a validation split wants: it pairs the comparison, so every
            method sees the same identity on the same frames
        :param seed: Seeds the per-item draw under ``deterministic``
        :raises ValueError: If the curriculum is empty
        """
        if not curriculum:
            raise ValueError("The curriculum needs at least one anonymization")

        self.names = list(curriculum)
        self.curriculum: list[Anonymizer] = [resolve(name) for name in curriculum]
        self.root = Path(root)
        self.frames = frames
        self.size = size
        self.detector = detector
        self.flip = flip
        self.deterministic = deterministic
        self.all_methods = all_methods
        self.seed = seed

        self.target: dict[str, str] = dict(zip(targets["identity"], targets["path"]))
        target_video = dict(zip(targets["identity"], targets["video"]))
        target_paths = set(self.target.values())
        wanted = set(self.target) if identities is None else set(identities) & set(self.target)

        # The frames each identity's inputs are drawn from, grouped by video. Resolved once, so
        # '__getitem__' only samples from lists.
        self.videos: dict[str, list[list[str]]] = {}
        for identity, group in structure[structure["identity"].isin(list(wanted))].groupby("identity"):
            # The target's own video is left out, so the model cannot rebuild the target from its
            # own shot's background and lighting rather than from the face.
            group = group[(group["video"] != target_video[identity]) & ~group["path"].isin(target_paths)]
            if not group.empty:
                self.videos[str(identity)] = [list(shot["path"]) for _, shot in group.groupby("video", sort=True)]

        self.identities = sorted(self.videos)
        dropped = len(wanted) - len(self.identities)
        if dropped:
            logger.warning("Identities have no input frames left and were dropped", extra={"dropped": dropped})

        self.stage = 0
        self.set_stage(stage)

        # Detection is otherwise repeated for every frame in every epoch. Held per instance, so
        # with 'num_workers > 0' each worker warms its own copy.
        self.areas: dict[str, FacialArea | None] = {}
        self.undetected = 0

    def __len__(self) -> int:
        return len(self.identities) * (self.stage + 1 if self.all_methods else 1)

    @property
    def stages(self) -> int:
        """
        :return: How many stages the curriculum has
        """
        return len(self.curriculum)

    @property
    def active(self) -> list[str]:
        """
        :return: The names of the anonymizations currently being drawn from
        """
        return self.names[: self.stage + 1]

    def set_stage(self, stage: int) -> list[str]:
        """
        Widen the dataset to the first ``stage + 1`` anonymizations.

        A ``DataLoader`` with ``num_workers > 0`` hands each worker a copy of the dataset, so a
        stage change made after its iterator exists will not reach them. Rebuild the loader.

        :param stage: The index into the curriculum to widen to
        :return: The names of the anonymizations now in play
        :raises IndexError: If the stage is outside the curriculum
        """
        if not 0 <= stage < self.stages:
            raise IndexError(f"Stage {stage} is outside a curriculum of {self.stages}: {self.names}")
        self.stage = stage
        return self.active

    def read(self, path: str) -> NDArray[np.uint8]:
        """
        :param path: The ``path`` column's value, relative to ``root``
        :return: The frame, BGR uint8
        :raises FileNotFoundError: If the frame is not readable
        """
        image = cv2.imread(str(self.root / path))
        if image is None:
            raise FileNotFoundError(f"Could not read frame {self.root / path}")
        return np.asarray(image, dtype=np.uint8)

    def area(self, path: str, image: NDArray[np.uint8]) -> FacialArea | None:
        """
        Where the face is, detected on the clear frame and remembered.

        :param path: The frame's path, the cache key
        :param image: The clear frame, BGR
        :return: The detected face, or None if there is none
        """
        if path not in self.areas:
            self.areas[path] = self.detector.detect(image)
            if self.areas[path] is None:
                self.undetected += 1
        return self.areas[path]

    def draw(self, identity: str, source: random.Random) -> list[str]:
        """
        The ``frames`` paths one item is built from, spread across the identity's videos.

        Round-robin rather than uniform: frames of one video are near duplicates, so a uniform
        draw can hand the aggregator N copies of one observation, which is the premise of the
        experiment not being exercised. An identity with fewer frames than N repeats rather than
        shrinking - the model takes a fixed N, and a repeated observation is a truthful item,
        since the attacker simply saw less.

        :param identity: Whose frames to draw
        :param source: The random source, seeded under ``deterministic``
        :return: Exactly ``frames`` paths
        """
        shots = [list(shot) for shot in self.videos[identity]]
        source.shuffle(shots)
        for shot in shots:
            source.shuffle(shot)

        chosen: list[str] = []
        while len(chosen) < self.frames:
            served = [shot.pop() for shot in shots if shot]
            if not served:
                pool = [path for shot in self.videos[identity] for path in shot]
                return chosen + source.choices(pool, k=self.frames - len(chosen))
            chosen.extend(served[: self.frames - len(chosen)])
        return chosen

    def crop(self, path: str, method: int | None, flip: bool) -> NDArray[np.uint8]:
        """
        One aligned crop, anonymized or clear.

        :param path: The frame to load
        :param method: The index into the curriculum of the anonymization to apply, or None to
            leave the frame clear, which is what the target wants
        :param flip: Whether to mirror the crop
        :return: The crop, ``(size, size, 3)`` BGR uint8
        """
        clear = self.read(path)
        area = self.area(path, clear)

        if area is None:
            # No face to anonymize or align on. Measured at essentially zero on this corpus, so
            # the frame is used whole rather than dropped, which would change the item's N.
            cropped = letterbox(clear, self.size)
        else:
            # Anonymize at the frame's native resolution, then align. The other order would fit
            # the blur radius or the block size to a crop the pipeline never produces.
            image = clear if method is None else self.curriculum[method].apply(clear, area)
            cropped = align_face(image, area.landmarks, self.size)
        return np.ascontiguousarray(cropped[:, ::-1]) if flip else cropped

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        """
        :param index: The item index. Normally one per identity; under ``all_methods`` the
            identities are laid out in blocks, one entry per active method within each
        :return: ``(anonymized, clear, method)``: ``(N, 3, size, size)`` and ``(3, size, size)``,
            both RGB in [-1, 1], and the curriculum index every one of the N frames carries
        """
        # Under 'all_methods' the method comes from the index rather than a draw, and the frames
        # are seeded by the identity, so one identity's methods see the same frames and differ
        # only in how they were anonymized.
        position, fixed = divmod(index, self.stage + 1) if self.all_methods else (index, None)
        identity = self.identities[position]

        # Seeded by position under 'deterministic' so a validation item is the same every epoch.
        # Otherwise the module-level generator, which a DataLoader reseeds per worker.
        source = random.Random(self.seed + position) if self.deterministic else random.Random()
        chosen = self.draw(identity, source)
        method = source.randrange(self.stage + 1) if fixed is None else fixed
        flip = self.flip and not self.deterministic and source.random() > 0.5

        anonymized = [to_tensor(self.crop(path, method, flip), self.size) for path in chosen]
        clear = to_tensor(self.crop(self.target[identity], None, flip), self.size)
        return stack(anonymized), clear, method
