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

Also unlike pix2pix, **one model covers every anonymization** rather than one model per method.
The methods are introduced as a curriculum -- clear frames first, then progressively harsher
degradations -- because the hardest of them destroys the face outright: a model shown black bars
from the first step has no signal to learn an identity from, while one that already reconstructs
faces from blurred input has something to fall back on. :data:`CURRICULUM` is that order and
:meth:`FacialFrames.set_stage` is what widens it between stages.

A whole item shares one anonymization, drawn per identity rather than per frame. That matches how
the experiment observes a condition -- a sweep leg pins one ``frame_anonymizer`` and every frame
of it is anonymized the same way -- so an item of mixed methods is a distribution the model would
never meet again at inference.
"""

import logging
import random
import time
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as fun
from torch import Tensor, nn, stack
from torch.utils.data import DataLoader, Dataset

from meles.anonymization import Anonymizer, resolve
from meles.common import FacialArea, Image
from meles.detection import TEMPLATE_SIZE, Detector, default_detector, prepare_face
from meles.frame_aggregation.facial import Facial, FacialEncoder
from meles.frame_aggregation.pix2pix import default_device, to_tensor

logger = logging.getLogger(__name__)

# The two learning rates the encoder schedule uses: the decoder and the aggregator are trained from
# scratch, while the encoder arrives pretrained and is only nudged, an order of magnitude slower.
LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4

# The weight on the identity term next to an L1 of 1.0. Chosen so the two start out comparable in
# magnitude rather than by tuning: on the L1-only run the reconstruction error settled near 0.33,
# while '1 - cos' between two unrelated faces is close to 1.0. Worth sweeping -- it is the dial
# that decides whether the model prefers a sharp face or the right one.
IDENTITY_WEIGHT = 0.3

# Bumped when the checkpoint layout changes, so an older file fails loudly rather than loading
# into mismatched state dicts.
CHECKPOINT_FORMAT = 1

# The order the anonymizations are introduced in, easiest first. Stage 0 trains on 'none' alone,
# which is the model learning to rebuild a face from several clear views of it before any of them
# is degraded; each later stage keeps everything before it and adds one method. 'blackbar' is last
# because it removes the face rather than obscuring it -- there is no inverse to learn, only a
# prior over what the other frames imply, so it is the one that needs the earlier stages most.
CURRICULUM: tuple[str, ...] = ("none", "gaussianblur", "pixelation", "blackbar")

# A criterion maps (generated, target) to either one per-item loss, or named per-item
# weighted contributions as 'WeightedCriterion' does. The trainer accepts both.
Criterion = Callable[[Tensor, Tensor], "Tensor | dict[str, Tensor]"]


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
    One item per identity: ``(anonymized, clear, method)`` where ``anonymized`` is
    ``(N, 3, size, size)``, ``clear`` is ``(3, size, size)``, and ``method`` is the index into
    :attr:`curriculum` of the anonymization this item was built with.

    The default ``DataLoader`` collate stacks these into ``(B, N, 3, size, size)``,
    ``(B, 3, size, size)`` and ``(B,)``, so no custom ``collate_fn`` is needed.

    ``method`` indexes the **whole** curriculum, not the active prefix, so the same number means
    the same anonymization at every stage. That is what makes a per-method loss comparable across
    stages -- which is the number that says whether adding ``blackbar`` cost you ``gaussianblur``.
    """

    def __init__(
        self,
        structure: pd.DataFrame,
        targets: pd.DataFrame,
        root: Path,
        curriculum: Sequence[str | Anonymizer] = CURRICULUM,
        stage: int = 0,
        identities: Sequence[str] | None = None,
        frames: int = 8,
        size: int = TEMPLATE_SIZE,
        detector: Detector = default_detector,
        exclude_target_video: bool = True,
        flip: bool = True,
        deterministic: bool = False,
        all_methods: bool = False,
        seed: int = 0,
    ):
        """
        :param structure: One row per frame, as ``data/ytfaces.parquet`` holds it: ``identity``,
            ``video`` and ``path``. These are the observations the attacker gets
        :param targets: One row per identity, as ``data/ytfaces_targets.parquet`` holds it (see
            ``src/notebooks/07_target_selection.ipynb``): the frame the decoder should produce
        :param root: What the ``path`` column is relative to, i.e. the ``data`` directory
        :param curriculum: The anonymizations in the order they are introduced, each a key in
            ``meles.anonymization.ANONYMIZERS`` or an instance. Pass a one-element sequence to
            train against a single method and nothing else
        :param stage: How far into the curriculum to start. Stage ``s`` draws uniformly from the
            first ``s + 1`` methods; :meth:`set_stage` moves it
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
        :param deterministic: Whether to draw the same frames, the same anonymization and no flip
            every epoch. What a *validation* split wants: a loss measured over freshly drawn
            frames each epoch is not comparable between epochs
        :param all_methods: Whether to yield every identity under **every** active anonymization
            instead of one drawn per identity, making the dataset ``identities x methods`` long.
            The other half of what a validation split wants: drawing one method per identity
            leaves the per-method counts to chance, and a method that happens to draw no items has
            no loss at all. It also pairs the comparison -- every method sees the same identity on
            the same frames, so a difference between two methods is the anonymization and nothing
            else. Pointless for training, where the draw averages out over epochs
        :param seed: Seeds the per-item draw under ``deterministic``
        """
        self.curriculum = [resolve(entry) if isinstance(entry, str) else entry for entry in curriculum]
        self.names = [entry if isinstance(entry, str) else type(entry).__name__ for entry in curriculum]
        if not self.curriculum:
            raise ValueError("The curriculum needs at least one anonymization")

        self.root = Path(root)
        self.frames = frames
        self.size = size
        self.detector = detector
        self.flip = flip
        self.deterministic = deterministic
        self.all_methods = all_methods
        self.seed = seed

        self.target: dict[str, str] = dict(zip(targets["identity"], targets["path"]))

        # The frames each identity's inputs are drawn from, resolved once so '__getitem__' only
        # samples from a list. An identity with no target cannot be trained on at all.
        wanted = set(self.target) if identities is None else set(identities) & set(self.target)
        target_video = dict(zip(targets["identity"], targets["video"]))
        target_path = set(self.target.values())

        self.pool: dict[str, list[str]] = {}
        for identity, group in structure[structure["identity"].isin(list(wanted))].groupby("identity"):
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
                extra={"dropped": len(dropped), "first": dropped[:5]},
            )

        self.stage = 0
        self.set_stage(stage)

        # Detection is otherwise repeated for every frame in every epoch. Filled lazily and held
        # per instance, so with 'num_workers > 0' each worker warms its own copy -- which only
        # pays off across epochs, i.e. with 'persistent_workers=True'.
        self._areas: dict[str, FacialArea | None] = {}
        self.missing = 0

    def __len__(self) -> int:
        # Under 'all_methods' the dataset is one item per (identity, active method) pair, so it
        # grows as the curriculum widens.
        return len(self.identities) * (self.stage + 1 if self.all_methods else 1)

    @property
    def stages(self) -> int:
        """How many stages the curriculum has, i.e. the last valid :attr:`stage` plus one."""
        return len(self.curriculum)

    @property
    def active(self) -> list[str]:
        """The names of the anonymizations currently being drawn from."""
        return self.names[: self.stage + 1]

    def set_stage(self, stage: int) -> list[str]:
        """
        Widen the dataset to the first ``stage + 1`` anonymizations.

        Called between training stages. It mutates the dataset in place rather than building a new
        one so that the detection cache below survives the widening -- but note that a
        ``DataLoader`` with ``num_workers > 0`` hands each worker a *copy*, so a stage change made
        after the loader was built will not reach the workers. Rebuild the loader, or set the
        stage before the epoch that uses it.

        :param stage: The index into the curriculum to widen to
        :return: The names of the anonymizations now in play
        :raises IndexError: If the stage is outside the curriculum
        """
        if not 0 <= stage < self.stages:
            raise IndexError(f"Stage {stage} is outside a curriculum of {self.stages}: {self.names}")

        self.stage = stage
        logger.info("Curriculum stage set", extra={"stage": stage, "methods": ",".join(self.active)})
        return self.active

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

    def crop(self, path: str, method: int | None, flip: bool) -> Image:
        """
        One aligned crop, anonymized or clear.

        :param path: The frame to load
        :param method: The index into :attr:`curriculum` of the anonymization to apply, or None
            to leave the frame clear -- which is what the target wants
        :param flip: Whether to mirror the crop
        :return: The crop, ``(size, size, 3)`` BGR uint8
        """
        clear = self.read(path)
        area = self.area(path, clear)

        # Anonymize at the frame's native resolution, then align: the other order would fit the
        # blur radius or the block size to a 112x112 crop the pipeline never produces.
        image = clear if method is None else self.curriculum[method].apply(clear, area)
        cropped = prepare_face(image, area, self.size)
        return np.ascontiguousarray(cropped[:, ::-1]) if flip else cropped

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, int]:
        """
        :param index: The item index. Normally one per identity; under ``all_methods`` the
            identities are laid out in blocks, one entry per active method within each
        :return: The ``(anonymized, clear, method)`` item: ``(N, 3, size, size)`` and
            ``(3, size, size)`` both RGB in [-1, 1], and the curriculum index of the
            anonymization every one of the ``N`` frames was built with
        """
        # Under 'all_methods' the method comes from the index rather than a draw, and the frames
        # are seeded by the *identity* -- so one identity's methods all see the same frames and
        # differ only in how they were anonymized.
        position, fixed = divmod(index, self.stage + 1) if self.all_methods else (index, None)
        identity = self.identities[position]
        pool = self.pool[identity]

        # Seeded by position under 'deterministic' so a validation item is the same every epoch.
        # Otherwise drawn from the 'random' module, which a DataLoader reseeds per worker, so the
        # workers draw different frames instead of replaying one stream.
        draw = random.Random(self.seed + position) if self.deterministic else random
        # 'choices' rather than 'sample': an identity may hold fewer frames than N, and repeating
        # an observation is a truthful item -- the attacker simply saw less.
        chosen = draw.sample(pool, self.frames) if len(pool) >= self.frames else draw.choices(pool, k=self.frames)

        # One anonymization for the whole item, uniform over the stages unlocked so far. Drawing
        # per frame instead would build an item mixing methods, which no sweep leg ever produces.
        method = draw.randrange(self.stage + 1) if fixed is None else fixed

        flip = self.flip and not self.deterministic and draw.random() > 0.5
        anonymized = [to_tensor(self.crop(path, method, flip), self.size) for path in chosen]
        clear = to_tensor(self.crop(self.target[identity], None, flip), self.size)
        return stack(anonymized), clear, method


def l1_per_item(generated: Tensor, target: Tensor) -> Tensor:
    """
    Mean absolute reconstruction error, **per item** rather than reduced over the batch.

    Everything downstream depends on this not reducing. A scalar loss cannot be split by
    anonymization afterwards -- the batch has already been averaged over items that came from
    different methods -- so the per-method breakdown has to start here.

    :param generated: What the model produced, ``(B, 3, H, W)`` in [-1, 1]
    :param target: The clear target, the same shape
    :return: One loss per item, ``(B,)``
    """
    return (generated - target).abs().flatten(1).mean(dim=1)


class IdentityLoss(nn.Module):
    """
    Cosine distance between a recogniser's embeddings of the reconstruction and of the target:
    ``1 - cos``, per item, in [0, 2].

    This is the loss the experiment's question actually asks. L1 asks for the right *pixels*, and
    its cheapest answer under uncertainty is the conditional median -- a blurry average face,
    which has low error against everyone and identifies as no one. Measured on a 125-epoch L1-only
    run, the per-method validation loss came out *inverted*: ``blackbar`` scored better than
    ``none``, because the less the model could see, the harder it hedged towards that mean face.
    A recogniser is not fooled by the average face, so this term removes that incentive.

    The critic is a **separate frozen copy** of the recogniser, never the encoder inside
    ``Facial``. Two reasons, and the second is the dangerous one:

    - the encoder is fine-tuned in the second phase, so a shared critic could be satisfied by
      moving the encoder rather than by improving the image -- the loss would be grading itself;
    - a critic whose weights drift means something different each epoch, so its numbers would not
      be comparable across a run.

    ``iresnet34`` is used because it is the only **differentiable** recogniser here: ``ArcFace``
    (``w600k_r50``), ``SFace`` and ``GhostFaceNet`` are ONNX, and ONNX Runtime carries no
    gradients. That is also the better arrangement -- ``w600k_r50`` is what
    ``meles.frame_embedder.models.ARCFACE`` scores the experiment with, so optimising against it
    directly would be training on the evaluation metric.
    """

    def __init__(self, weights: str | Path | None = None, encoder: FacialEncoder | None = None):
        """
        :param weights: The recogniser weights, e.g. ``models/iresnet34.pth``. Ignored when
            ``encoder`` is given
        :param encoder: An already-built encoder to copy the critic from, for tests or to avoid
            reading the same file twice
        """
        super().__init__()
        self.encoder = encoder if encoder is not None else FacialEncoder(weights=weights)
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "IdentityLoss":
        """Pins the critic to ``eval``. Its BatchNorms must never move; see the class docstring."""
        super().train(mode)
        self.encoder.eval()
        return self

    def embed(self, images: Tensor) -> Tensor:
        """
        :param images: ``(B, 3, 112, 112)`` in [-1, 1] RGB, which is what ``to_tensor`` produces
            and what the recogniser was trained on -- so no renormalisation is needed
        :return: The L2-normalised ``(B, 512)`` embeddings
        """
        # Index 0 is the 512-d embedding; index 1 is the 7x7 feature map 'FacialEncoder.forward'
        # returns for the decoder. The identity lives in the former.
        return fun.normalize(self.encoder.model(images)[0], dim=1)

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        """
        :param generated: What the model produced, ``(B, 3, 112, 112)`` in [-1, 1]
        :param target: The clear target, the same shape
        :return: ``1 - cos`` per item, ``(B,)``
        """
        # The target's embedding is a constant: no gradient flows to it, and computing it under
        # 'no_grad' keeps the critic's activations off the tape.
        with torch.no_grad():
            wanted = self.embed(target)
        return 1.0 - (self.embed(generated) * wanted).sum(dim=1)


class ContrastiveIdentityLoss(IdentityLoss):
    """
    InfoNCE over the batch: the reconstruction must embed closer to **its own** target than to
    anybody else's.

    :class:`IdentityLoss` only pulls towards the right face; nothing pushes away from the wrong
    ones, and a reconstruction can grow more similar to its target while growing just as similar
    to everyone -- which moves the cosine and leaves identification where it was. Rank-1 is a
    ranking question, so this optimises the ranking directly.

    Items that share a target are masked out of each other's negatives. The validation split
    yields every identity once per anonymization, so without the mask an identity would be
    punished for resembling itself under a different method.
    """

    def __init__(
        self,
        weights: str | Path | None = None,
        encoder: FacialEncoder | None = None,
        temperature: float = 0.05,
        duplicate: float = 0.999,
    ):
        """
        :param weights: The recogniser weights, as for :class:`IdentityLoss`
        :param encoder: An already-built encoder to copy the critic from
        :param temperature: Softmax temperature over the cosine similarities. Low because cosines
            live in [-1, 1], so the logits need scaling before a softmax separates them at all
        :param duplicate: Cosine above which two targets are taken to be the same identity, and
            so masked out of each other's negatives
        """
        super().__init__(weights=weights, encoder=encoder)
        self.temperature = temperature
        self.duplicate = duplicate

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        """
        :param generated: What the model produced, ``(B, 3, 112, 112)`` in [-1, 1]
        :param target: The clear target, the same shape
        :return: The per-item cross entropy, ``(B,)``. A batch of one has no negatives, so it is
            zero by construction
        """
        with torch.no_grad():
            gallery = self.embed(target)
        probe = self.embed(generated)

        logits = probe @ gallery.T / self.temperature
        own = torch.arange(len(probe), device=logits.device)

        # Mask other items that carry the same target, but never the diagonal -- that is the
        # answer. -inf drops them out of the softmax entirely.
        shared = (gallery @ gallery.T) > self.duplicate
        shared[own, own] = False
        logits = logits.masked_fill(shared, float("-inf"))

        if len(probe) < 2:
            return torch.zeros(len(probe), device=logits.device)
        return fun.cross_entropy(logits, own, reduction="none")


class WeightedCriterion:
    """
    Several per-item losses, summed with weights.

    Returns the **weighted** contributions per term rather than one number, so the trainer can
    both backpropagate their sum and report them apart. Reporting them apart is the point: a
    falling total tells you nothing about whether it fell because the faces got sharper or because
    they got more recognisable, and those are the two things being traded against each other.
    """

    def __init__(self, terms: dict[str, tuple[float, Callable[[Tensor, Tensor], Tensor]]]):
        """
        :param terms: Name to ``(weight, loss)``, where each loss returns one value per item::

                WeightedCriterion({
                    "l1": (1.0, l1_per_item),
                    "identity": (IDENTITY_WEIGHT, IdentityLoss("models/iresnet34.pth")),
                })

        :raises ValueError: If no terms are given
        """
        if not terms:
            raise ValueError("A criterion needs at least one term")
        self.terms = dict(terms)

    def to(self, device: torch.device | str) -> "WeightedCriterion":
        """Move any term that is a module -- the identity loss carries a network."""
        for weight, loss in self.terms.values():
            if isinstance(loss, nn.Module):
                loss.to(device)
        return self

    def __call__(self, generated: Tensor, target: Tensor) -> dict[str, Tensor]:
        """
        :param generated: What the model produced, ``(B, 3, H, W)``
        :param target: The clear target, the same shape
        :return: Each term's **weighted** per-item contribution, ``(B,)`` apiece
        """
        return {name: weight * loss(generated, target) for name, (weight, loss) in self.terms.items()}


@dataclass(frozen=True)
class Losses:
    """
    What one epoch cost: the mean error overall, split by anonymization, and split by loss term.

    **The per-method split is the number a curriculum is steered by.** A single mean hides the
    thing worth seeing -- whether unlocking ``blackbar`` bought its gain by giving up
    ``gaussianblur`` -- because the newly unlocked method is also the one dragging the mean up, so
    a rising total is equally consistent with healthy progress and with quiet forgetting.

    **The per-term split is what a weighted criterion needs.** ``l1`` and ``identity`` pull in
    different directions -- a blurry average face is good L1 and terrible identity -- so the total
    alone cannot say which one moved.

    :param overall: The item-weighted mean over every method seen
    :param per_method: The mean per anonymization, NaN for one that did not appear
    :param counts: How many items each anonymization contributed, which is what says whether a
        per-method number is worth reading
    :param per_term: Each term's weighted mean, over everything
    :param per_term_method: Each term's weighted mean, split by anonymization
    """

    overall: float
    per_method: dict[str, float]
    counts: dict[str, int]
    per_term: dict[str, float] = field(default_factory=dict)
    per_term_method: dict[str, dict[str, float]] = field(default_factory=dict)

    def describe(self, places: int = 4) -> str:
        """A one-line summary, for a log line or a progress bar."""
        seen = " ".join(f"{name}={value:.{places}f}" for name, value in self.per_method.items() if value == value)
        terms = " ".join(f"{name}={value:.{places}f}" for name, value in self.per_term.items())
        described = f"{self.overall:.{places}f}"
        if terms:
            described += f" [{terms}]"
        return f"{described} ({seen})" if seen else described


class FacialTrainer:
    """
    A :class:`meles.frame_aggregation.facial.Facial` training run over the anonymization
    curriculum: the model, its optimiser, its history and its checkpoint.

    Two schedules run at once and they are independent, which is the easiest thing to confuse:

    - the **curriculum** widens which anonymizations the data holds (:meth:`fit` walks it);
    - the **encoder schedule** freezes the pretrained encoder and later fine-tunes it
      (:meth:`set_encoder_frozen`).

    Call ``fit`` once per encoder phase, not once per curriculum stage.
    """

    def __init__(
        self,
        model: Facial | None = None,
        methods: Sequence[str] = CURRICULUM,
        device: torch.device | str | None = None,
        learning_rate: float = LEARNING_RATE,
        encoder_learning_rate: float = ENCODER_LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        criterion: Criterion = l1_per_item,
        encoder_frozen: bool = True,
    ):
        """
        :param model: The model to train, or None to build one conditioned on ``methods``
        :param methods: The anonymization names, in curriculum order. Only used for labelling and
            for sizing the conditioning; the dataset decides what is actually drawn
        :param device: Where to train, defaulting to CUDA when there is one
        :param learning_rate: The step size for the decoder and the aggregator, trained from
            scratch
        :param encoder_learning_rate: The step size for the pretrained encoder, used only once it
            is unfrozen. An order of magnitude lower, because it arrives already trained
        :param weight_decay: AdamW's decoupled weight decay
        :param criterion: Maps ``(generated, target)`` to **one loss per item**. Swapping in an
            identity loss -- cosine distance between recogniser embeddings of the two -- is the
            obvious next step, since the experiment scores recognisability, not pixels
        :param encoder_frozen: Whether to start with the encoder frozen, which is the first phase
            of the encoder schedule
        """
        self.methods = list(methods)
        self.model = model if model is not None else Facial(methods=len(self.methods))
        self.device = torch.device(device) if device is not None else default_device()
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.weight_decay = weight_decay
        self.criterion = criterion
        # The identity term carries a network of its own, which has to follow the model.
        if isinstance(criterion, WeightedCriterion | nn.Module):
            criterion.to(self.device)

        self.epoch = 0
        self.stage = 0
        self.history: dict[str, list[float]] = {key: [] for key in self._history_keys()}

        self.optimizer: torch.optim.Optimizer
        self.set_encoder_frozen(encoder_frozen)

    def _history_keys(self) -> list[str]:
        keys = ["stage", "train", "val"]
        return keys + [f"{prefix}_{name}" for prefix in ("train", "val") for name in self.methods]

    def record(self, key: str, value: float) -> None:
        """
        Append to a history series, creating it if this is the first epoch that produced it.

        A new series is back-filled with NaN so that every series has one entry per epoch and
        plotting them against a shared epoch axis lines up.
        """
        series = self.history.setdefault(key, [float("nan")] * (self.epoch - 1))
        series.extend([float("nan")] * (self.epoch - 1 - len(series)))
        series.append(value)

    def set_encoder_frozen(self, frozen: bool) -> "FacialTrainer":
        """
        Switch the encoder between frozen and fine-tuned, rebuilding the optimiser for it.

        The optimiser has to be rebuilt because the unfrozen encoder is a new parameter group at
        its own learning rate. That discards AdamW's moment estimates, which is the intended
        behaviour at a phase change -- the moments describe a loss surface that just moved.

        :param frozen: Whether the encoder is frozen
        :return: This trainer, for chaining
        """
        self.model.set_encoder_frozen(frozen)

        groups: list[dict] = [{"params": self.model.train_parameters(), "lr": self.learning_rate}]
        fine = self.model.fine_parameters()
        if fine:
            groups.append({"params": fine, "lr": self.encoder_learning_rate})

        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        logger.info(
            "Encoder schedule set",
            extra={"frozen": frozen, "groups": len(groups), "fine_tuned": len(fine)},
        )
        return self

    def _batch(self, batch: tuple[Tensor, Tensor, Tensor]) -> tuple[Tensor, Tensor, Tensor]:
        anonymized, clear, method = batch
        return (
            anonymized.to(self.device, non_blocking=True),
            clear.to(self.device, non_blocking=True),
            method.to(self.device, non_blocking=True),
        )

    def _apply(self, generated: Tensor, clear: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Run the criterion, accepting either contract.

        A criterion may return one per-item tensor, or a dict of named per-item **weighted**
        contributions as :class:`WeightedCriterion` does. The sum is what gets backpropagated
        either way; the terms are only carried along so they can be reported apart.

        :return: The per-item total and the per-item terms
        """
        result = self.criterion(generated, clear)
        terms = {"loss": result} if isinstance(result, Tensor) else result
        return torch.stack(list(terms.values())).sum(dim=0), terms

    def step(self, anonymized: Tensor, clear: Tensor, method: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        One optimiser step over a batch.

        :param anonymized: ``(B, N, 3, H, W)``, the observations
        :param clear: ``(B, 3, H, W)``, the targets
        :param method: ``(B,)``, which anonymization each item carries
        :return: The detached per-item totals and per-item terms, so the caller can split both
            by method
        """
        self.optimizer.zero_grad(set_to_none=True)
        total, terms = self._apply(self.model(anonymized, method), clear)
        total.mean().backward()
        self.optimizer.step()
        return total.detach(), {name: value.detach() for name, value in terms.items()}

    def _totals(self) -> tuple[Tensor, Tensor]:
        return torch.zeros(len(self.methods)), torch.zeros(len(self.methods))

    @staticmethod
    def _accumulate(sums: Tensor, counts: Tensor, per_item: Tensor, method: Tensor) -> None:
        """Add a batch's per-item losses into the per-method running totals."""
        index = method.detach().to("cpu", torch.long)
        sums.index_add_(0, index, per_item.detach().to("cpu", torch.float32))
        counts.index_add_(0, index, torch.ones_like(index, dtype=torch.float32))

    def _accumulate_terms(self, term_sums: dict[str, Tensor], terms: dict[str, Tensor], method: Tensor) -> None:
        """The same, per loss term. The counts are shared, so only the sums are kept apart."""
        ignored = torch.zeros(len(self.methods))
        for name, value in terms.items():
            running = term_sums.setdefault(name, torch.zeros(len(self.methods)))
            self._accumulate(running, ignored, value, method)

    def _mean_by_method(self, sums: Tensor, counts: Tensor) -> dict[str, float]:
        return {name: float(sums[i] / counts[i]) if counts[i] else float("nan") for i, name in enumerate(self.methods)}

    def _losses(self, sums: Tensor, counts: Tensor, terms: dict[str, Tensor] | None = None) -> Losses:
        total = float(counts.sum())
        terms = terms or {}
        return Losses(
            overall=float(sums.sum() / total) if total else float("nan"),
            per_method=self._mean_by_method(sums, counts),
            counts={name: int(counts[i]) for i, name in enumerate(self.methods)},
            per_term={name: float(value.sum() / total) if total else float("nan") for name, value in terms.items()},
            per_term_method={name: self._mean_by_method(value, counts) for name, value in terms.items()},
        )

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[Tensor, Tensor, Tensor]]) -> Losses:
        """
        The mean losses over a loader, split by anonymization, with nothing updated.

        Runs in ``eval`` mode, and both halves of that matter. ``iresnet34`` carries BatchNorms,
        whose running statistics are updated inside the forward pass whenever the module is in
        ``train`` mode -- ``no_grad`` does not stop that -- so validating in ``train`` mode would
        spend every epoch quietly fitting normalisation to held-out identities.

        :param loader: Batches of ``(anonymized, clear, method)``
        :return: The losses, overall and per method
        """
        was_training = self.model.training
        self.model.eval()
        try:
            sums, counts = self._totals()
            term_sums: dict[str, Tensor] = {}
            for batch in loader:
                anonymized, clear, method = self._batch(batch)
                total, terms = self._apply(self.model(anonymized, method), clear)
                self._accumulate(sums, counts, total, method)
                self._accumulate_terms(term_sums, terms, method)
        finally:
            self.model.train(was_training)
        return self._losses(sums, counts, term_sums)

    def train_epoch(
        self, loader: Iterable[tuple[Tensor, Tensor, Tensor]], description: str = "", progress: bool = True
    ) -> Losses:
        """
        One pass over the training loader.

        :param loader: Batches of ``(anonymized, clear, method)``
        :param description: What to label the progress bar with
        :param progress: Whether to show a progress bar
        :return: The training losses, overall and per method
        """
        from tqdm.auto import tqdm

        self.model.train()
        sums, counts = self._totals()
        term_sums: dict[str, Tensor] = {}
        # 'mininterval' keeps the bar from writing megabytes of carriage returns into a log, the
        # same reason the workflow scripts set it.
        stepped = tqdm(loader, desc=description, mininterval=5.0) if progress else loader

        for batch in stepped:
            anonymized, clear, method = self._batch(batch)
            total, terms = self.step(anonymized, clear, method)
            self._accumulate(sums, counts, total, method)
            self._accumulate_terms(term_sums, terms, method)
        return self._losses(sums, counts, term_sums)

    def fit(
        self,
        train_loader_for: Callable[[int], DataLoader],
        validation_loader: Iterable[tuple[Tensor, Tensor, Tensor]] | None = None,
        epochs_per_stage: int = 25,
        stages: Sequence[int] | None = None,
        checkpoint: str | Path | None = None,
        best: str | None = None,
        maximize: bool = True,
        metrics: Callable[["FacialTrainer"], dict[str, float]] | None = None,
        on_epoch: Callable[["FacialTrainer", int], None] | None = None,
        progress: bool = True,
    ) -> dict[str, list[float]]:
        """
        Walk the curriculum, training ``epochs_per_stage`` epochs at each stage.

        :param train_loader_for: Builds the training loader for a curriculum stage. A **factory**
            rather than a loader because a ``DataLoader`` hands its workers a copy of the dataset
            when its iterator is first made: widening the dataset in this process after that would
            reach the parent and nothing else, and training would silently stay on the old stage
        :param validation_loader: Held-out batches, or None to skip validation. Pin it to the
            **last** stage so every method is measured from the first epoch -- otherwise a method
            has no baseline to be compared against on the epoch it unlocks
        :param epochs_per_stage: Epochs to train at each stage
        :param stages: Which curriculum stages to run, defaulting to all of them in order
        :param checkpoint: A file to write after every epoch, or None not to
        :param best: A history key to also keep the best epoch of, written alongside
            ``checkpoint`` with ``.best`` before the suffix. Worth setting whenever the metric is
            noisier than the improvement: rank-1 over a few dozen identities swings several points
            between neighbouring epochs, so the last epoch is an arbitrary draw from the
            trajectory rather than its best point
        :param maximize: Whether a larger ``best`` is better. True for rank-1, False for a loss
        :param metrics: Called with the trainer after each epoch; whatever it returns is recorded
            into the history under ``val:<key>``. Use it for numbers the loss cannot express --
            :func:`identification` is the one that matters, since rank-1 is what the experiment
            scores and the loss only approximates
        :param on_epoch: Called with ``(trainer, epoch)`` after each epoch, for plotting samples
        :param progress: Whether to show a per-epoch progress bar
        :return: The full history, including any epochs from before this call
        """
        started = time.time()
        schedule = list(range(len(self.methods))) if stages is None else list(stages)
        record = float("-inf") if maximize else float("inf")
        if best is not None and self.history.get(best):
            seen = [value for value in self.history[best] if value == value]
            record = (max if maximize else min)([record, *seen])

        for stage in schedule:
            loader = train_loader_for(stage)
            self.stage = stage
            active = ", ".join(self.methods[: stage + 1])

            for index in range(epochs_per_stage):
                described = f"stage {stage} ({active}) epoch {index + 1}/{epochs_per_stage}"
                training = self.train_epoch(loader, described, progress)
                validation = (
                    self.evaluate(validation_loader) if validation_loader is not None else self._losses(*self._totals())
                )
                self.epoch += 1

                self.history["stage"].append(float(stage))
                for prefix, losses in (("train", training), ("val", validation)):
                    self.history[prefix].append(losses.overall)
                    for name, value in losses.per_method.items():
                        self.history[f"{prefix}_{name}"].append(value)
                    # Term keys are only known once a criterion has run, so they are created on
                    # first sight rather than up front -- and back-filled with NaN so every
                    # series stays the same length as the epoch count.
                    for term, value in losses.per_term.items():
                        self.record(f"{prefix}:{term}", value)
                        for name, split in losses.per_term_method[term].items():
                            self.record(f"{prefix}:{term}_{name}", split)

                measured = metrics(self) if metrics is not None else {}
                for key, value in measured.items():
                    self.record(f"val:{key}", value)

                logger.info(
                    "Trained epoch",
                    extra={
                        "epoch": self.epoch,
                        "stage": stage,
                        "active": active,
                        "train": round(training.overall, 5),
                        "val": round(validation.overall, 5),
                        **{f"val_{name}": round(value, 5) for name, value in validation.per_method.items()},
                        **{f"val:{term}": round(value, 5) for term, value in validation.per_term.items()},
                        **{f"val:{key}": round(value, 5) for key, value in measured.items()},
                    },
                )

                if checkpoint is not None:
                    self.save(checkpoint)
                if best is not None and checkpoint is not None:
                    latest = self.history.get(best, [float("nan")])[-1]
                    # NaN compares False either way, so an epoch that did not produce the metric
                    # simply never becomes the record.
                    if (latest > record) if maximize else (latest < record):
                        record = latest
                        kept = Path(checkpoint)
                        self.save(kept.with_suffix(f".best{kept.suffix}"))
                        logger.info("New best epoch", extra={"epoch": self.epoch, best: latest})
                if on_epoch is not None:
                    on_epoch(self, self.epoch)

        logger.info(
            "Training completed",
            extra={"epochs": self.epoch, "stages": len(schedule), "seconds": round(time.time() - started, 1)},
        )
        return self.history

    def state(self, inference_only: bool = False) -> dict:
        """
        :param inference_only: Whether to leave the optimiser out. AdamW keeps two moment
            estimates per parameter, so dropping them is most of the file -- but what is left
            cannot be resumed, only run
        :return: Everything needed to resume this run, or to run it
        """
        state = {
            "format": CHECKPOINT_FORMAT,
            "model": self.model.state_dict(),
            "epoch": self.epoch,
            "stage": self.stage,
            "methods": self.methods,
            "encoder_frozen": self.model.encoder_frozen,
            "history": self.history,
        }
        if not inference_only:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def save(self, path: str | Path, inference_only: bool = False) -> Path:
        """
        Write a checkpoint.

        Written to a temporary file and renamed, because a run interrupted while overwriting the
        previous epoch's checkpoint would otherwise leave neither a complete old one nor a
        complete new one.

        :param path: The file to write; its directory is created if needed
        :param inference_only: Whether to drop the optimiser state, which is roughly two thirds
            of the file but makes it unresumable
        :return: The path written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state(inference_only), temporary)
        temporary.replace(path)
        return path

    def load(self, path: str | Path) -> "FacialTrainer":
        """
        Restore a checkpoint written by :meth:`save`.

        :param path: The file to read
        :return: This trainer
        :raises ValueError: If the file was written by a different checkpoint format, or for a
            different set of anonymizations
        """
        payload = torch.load(Path(path), map_location=self.device, weights_only=False)

        if payload.get("format") != CHECKPOINT_FORMAT:
            raise ValueError(f"Checkpoint format {payload.get('format')!r}, expected {CHECKPOINT_FORMAT}")
        if list(payload["methods"]) != self.methods:
            raise ValueError(f"Checkpoint covers {payload['methods']}, this trainer covers {self.methods}")

        # The encoder phase decides the optimiser's parameter groups, so it has to be restored
        # before the optimiser state is loaded into them.
        self.set_encoder_frozen(bool(payload["encoder_frozen"]))
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            logger.warning(
                "Checkpoint holds no optimiser state, so training restarts from fresh moments",
                extra={"epoch": payload["epoch"]},
            )
        self.epoch = int(payload["epoch"])
        self.stage = int(payload["stage"])
        self.history = {key: list(values) for key, values in payload["history"].items()}
        return self


def load_facial(path: str | Path, device: torch.device | str | None = None) -> tuple[Facial, dict]:
    """
    Load a trained model for inference, without building a trainer around it.

    :param path: A checkpoint written by :meth:`FacialTrainer.save`, with or without optimiser
        state
    :param device: Where to put the model, defaulting to CUDA when there is one
    :return: The model in ``eval`` mode, and the checkpoint's metadata minus the tensors --
        ``epoch``, ``stage``, ``methods`` and ``history``
    :raises ValueError: If the file was written by a different checkpoint format
    """
    where = torch.device(device) if device is not None else default_device()
    payload = torch.load(Path(path), map_location=where, weights_only=False)

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Checkpoint format {payload.get('format')!r}, expected {CHECKPOINT_FORMAT}")

    methods = list(payload["methods"])
    model = Facial(methods=len(methods))
    model.load_state_dict(payload["model"])
    model.to(where).eval()

    metadata = {key: payload[key] for key in ("epoch", "stage", "methods", "history") if key in payload}
    return model, metadata


@torch.no_grad()
def reconstruct(
    model: Facial,
    anonymized: Tensor,
    method: Tensor | int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Run observations through a trained model and get the faces back.

    Always runs in ``eval`` mode: ``iresnet34``'s BatchNorms otherwise normalise by the batch in
    front of them, so the same identity would reconstruct differently depending on which other
    identities happened to share its batch.

    :param model: A trained :class:`meles.frame_aggregation.facial.Facial`
    :param anonymized: ``(B, N, 3, H, W)`` observations, or ``(N, 3, H, W)`` for a single item
    :param method: Which anonymization the observations carry -- a ``(B,)`` tensor, one int for
        the whole batch, or None for a model trained without conditioning
    :param device: Where to run, defaulting to wherever the model already is
    :return: The reconstructions, ``(B, 3, H, W)`` in [-1, 1], on the CPU
    """
    where = torch.device(device) if device is not None else next(model.parameters()).device
    batched = anonymized if anonymized.dim() == 5 else anonymized.unsqueeze(0)
    batched = batched.to(where)

    if isinstance(method, int):
        method = torch.full((batched.shape[0],), method, dtype=torch.long)
    if method is not None:
        method = method.to(where)
        if method.dim() == 0:
            method = method.expand(batched.shape[0])

    was_training = model.training
    model.eval()
    try:
        return model(batched, method).cpu()
    finally:
        model.train(was_training)


@torch.no_grad()
def identification(
    model: Facial,
    frames: "FacialFrames",
    embedder: Callable[[Tensor], np.ndarray],
    ranks: Sequence[int] = (1, 5),
    chunk: int = 16,
) -> dict[str, float]:
    """
    Rank-k identification accuracy: how often a reconstruction's nearest gallery face is the
    right person.

    **This is the number the experiment scores, and the one to steer on.** Cosine distance
    compresses almost all of its interesting range into the last tenth -- unrelated faces sit at
    ``1 - cos`` near 1.0 -- so a large change in whether the right identity ranks first shows up
    there as a rounding error. Identification only asks that the correct face be *nearest*, not
    that it be absolutely close.

    The gallery is the clear target of every identity in ``frames``; the probes are the
    reconstructions, one per identity per anonymization.

    :param model: A trained model
    :param frames: The validation set, built with ``all_methods=True`` and ``deterministic=True``
        so the layout is identity-major and stable between epochs
    :param embedder: Maps ``(B, 3, H, W)`` in [-1, 1] to L2-normalised embeddings. Use a
        recogniser that is **not** the training critic, or this grades its own work
    :param ranks: Which rank-k accuracies to report
    :param chunk: How many identities to reconstruct at once
    :return: ``rank1``, ``rank1_<method>``, ... as fractions in [0, 1]
    :raises ValueError: If ``frames`` was not built with ``all_methods``
    """
    if not frames.all_methods:
        raise ValueError("identification needs a dataset built with all_methods=True")

    width = frames.stage + 1
    people = len(frames) // width
    gallery = embedder(torch.stack([frames[i * width][1] for i in range(people)]))
    answer = np.arange(people)

    scores: dict[str, float] = {}
    hits: dict[int, list[float]] = {rank: [] for rank in ranks}
    for index in range(width):
        produced = []
        for begin in range(0, people, chunk):
            block = torch.stack([frames[p * width + index][0] for p in range(begin, min(begin + chunk, people))])
            produced.append(reconstruct(model, block, index))
        similarity = embedder(torch.cat(produced)) @ gallery.T

        # argsort descending, then ask where the right answer landed.
        order = np.argsort(-similarity, axis=1)
        position = np.argmax(order == answer[:, None], axis=1)
        for rank in ranks:
            value = float((position < rank).mean())
            scores[f"rank{rank}_{frames.names[index]}"] = value
            hits[rank].append(value)

    for rank in ranks:
        scores[f"rank{rank}"] = float(np.mean(hits[rank]))
    return scores
