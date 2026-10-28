import logging
import random
from pathlib import Path

import torch
import torch.nn.functional as fun
from torch import nn

from meles.common import Images
from meles.frame_aggregation.aggregator import FrameAggregator
from meles.frame_aggregation.iresnet import iresnet34

logger = logging.getLogger(__name__)


class FacialEncoder(nn.Module):
    def __init__(self, weights=None):
        super().__init__()
        self.model = iresnet34()  # Hidden state: B x 512 x 7 x 7
        if weights:
            sd = torch.load(weights, map_location="cpu")
            sd = {k.removeprefix("module."): v for k, v in sd.items()}
            self.model.load_state_dict(sd, strict=True)

    def forward(self, x):
        return self.model(x)[1]


class FacialDecoderBlock(nn.Module):
    def __init__(self, cin, cout, groups=8, dropout=0.0):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(cin, cout, 3, 1, 1)
        self.conv2 = nn.Conv2d(cout, cout, 3, 1, 1)
        self.norm1 = nn.GroupNorm(groups, cout)
        self.norm2 = nn.GroupNorm(groups, cout)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()
        # Dropout2d, not Dropout: these are conv features, where neighbouring pixels of one
        # channel are strongly correlated, so dropping individual activations leaks through the
        # ones beside them. Dropping whole channels is what actually regularises here.
        # It sits on the residual branch only, leaving the skip path clean.
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x):
        x = self.up(x)
        h = fun.silu(self.norm1(self.conv1(x)))
        h = fun.silu(self.norm2(self.conv2(h)))
        return self.drop(h) + self.skip(x)


class FacialDecoder(nn.Module):
    def __init__(self, cin=512, chans=(256, 192, 128, 64), width=1.0, groups=8, dropout=0.0):
        """
        Expands the aggregator's ``512 x 7 x 7`` into a ``3 x 112 x 112`` face.

        The bottleneck is the thing to keep in mind: 25,088 values in, 37,632 out, and no skip
        path from the inputs -- unlike pix2pix's U-Net, every pixel of detail has to survive the
        pooling. ``width`` is the dial for testing whether that, or the decoder's own capacity, is
        what limits the result.

        :param cin: Channels arriving from the aggregator
        :param chans: The channel ladder, one entry per 2x upsampling step. Four steps take 7 to
            112, so changing the length changes the output size
        :param width: Multiplies every entry of ``chans``. 1.0 is the original decoder at 3.2 M
            parameters; 2.0 is roughly four times that. Each result is rounded to a multiple of
            ``groups``, which ``GroupNorm`` requires
        :param groups: ``GroupNorm`` groups
        :param dropout: Channel dropout on each block's residual branch. Worth turning on together
            with ``width``, since a wider decoder overfits harder -- on this data the train/val
            gap grew when the dataset was doubled, so capacity is not short of things to memorise
        """
        super().__init__()
        self.width = width
        self.dropout = dropout
        # Rounded up to a multiple of 'groups' so GroupNorm stays valid at any width, and never
        # to zero.
        self.chans = tuple(max(groups, int(round(c * width / groups)) * groups) for c in chans)

        self.blocks = nn.ModuleList()
        for c in self.chans:
            self.blocks.append(FacialDecoderBlock(cin, c, groups=groups, dropout=dropout))
            cin = c
        self.out = nn.Sequential(nn.GroupNorm(groups, cin), nn.SiLU(), nn.Conv2d(cin, 3, 3, 1, 1))

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        return torch.tanh(self.out(x))


class FacialAggregator(nn.Module):
    """N feature maps -> 1. One transformer block: cross-attention + FFN."""

    def __init__(self, cin=512, heads=8, grid=7, ff_mult=4, methods=0):
        """
        ``methods`` makes the pooling query conditional on which anonymization the inputs carry.
        The query is otherwise a single learned constant, identical for every input, so one model
        covering several degradations has to serve them all from one starting point -- undoing a
        blur and inventing a face behind a black bar are not the same operation. A learned
        embedding per method, added to the query, is the cheapest way to let it specialise.

        Pass 0 (the default) to keep the unconditional query, which is the right choice when one
        model is trained per anonymization.
        """
        super().__init__()
        self.grid = grid
        self.pos = nn.Parameter(torch.randn(1, grid * grid, cin) * 0.02)
        self.query = nn.Parameter(torch.randn(1, grid * grid, cin) * 0.02)
        # Zero-initialised, so an untrained conditional model starts out exactly as the
        # unconditional one does rather than with the query displaced at random.
        self.method = nn.Embedding(methods, cin) if methods else None
        if self.method is not None:
            nn.init.zeros_(self.method.weight)

        self.attn = nn.MultiheadAttention(cin, heads, batch_first=True)
        self.norm_q = nn.LayerNorm(cin)
        self.norm_kv = nn.LayerNorm(cin)
        self.norm_ff = nn.LayerNorm(cin)
        self.ff = nn.Sequential(
            nn.Linear(cin, cin * ff_mult),
            nn.GELU(),
            nn.Linear(cin * ff_mult, cin),
        )

    def forward(self, x, method=None):
        """x: B x N x C x H x W, method: B  ->  B x C x H x W"""
        b, n, c, h, w = x.shape
        t = x.flatten(0, 1).flatten(2).transpose(1, 2)  # B*N x 49 x C
        t = t + self.pos
        kv = t.reshape(b, n * h * w, c)  # B x N*49 x C
        q = self.query.expand(b, -1, -1)  # B x 49 x C
        if self.method is not None and method is not None:
            q = q + self.method(method).unsqueeze(1)  # B x 1 x C, broadcast over the grid

        h_ = q + self.attn(self.norm_q(q), self.norm_kv(kv), self.norm_kv(kv), need_weights=False)[0]
        h_ = h_ + self.ff(self.norm_ff(h_))
        return h_.transpose(1, 2).reshape(b, c, h, w)


class Facial(nn.Module):
    def __init__(self, weights=None, methods=0, width=1.0, dropout=0.0):
        """
        ``methods`` is how many anonymizations one model covers, i.e. ``len(CURRICULUM)`` when it
        is trained across all of them. It conditions the aggregator's pooling query on which one
        the inputs carry; 0 keeps the unconditional query. The anonymization is known at inference
        -- a sweep leg names it, and blur, blocks and a black bar are not hard to tell apart -- so
        supplying it is not information the attacker would lack.

        ``width`` and ``dropout`` size and regularise the decoder; see :class:`FacialDecoder`.
        Both default to the original decoder, so a checkpoint written before they existed still
        loads. **They change the parameter shapes**, so a checkpoint and the model that reads it
        have to agree on them -- give a run at a new width a new checkpoint file.
        """
        super().__init__()
        self.encoder_frozen = False
        self.methods = methods
        self.encoder = FacialEncoder(weights=weights)
        self.aggregator = FacialAggregator(methods=methods)
        self.decoder = FacialDecoder(width=width, dropout=dropout)

    def forward(self, x, method=None):
        """x: B x N x 3 x 112 x 112, method: B -> B x 3 x 112 x 112"""
        b, n = x.shape[:2]
        if self.encoder_frozen:
            with torch.no_grad():
                f = self.encoder(x.flatten(0, 1))
        else:
            f = self.encoder(x.flatten(0, 1))
        f = f.view(b, n, *f.shape[1:])
        return self.decoder(self.aggregator(f, method))

    def train(self, mode=True):
        """Keeps the encoder in eval mode while frozen, so its BatchNorm
        running statistics are not overwritten by the new data."""
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def set_encoder_frozen(self, frozen: bool):
        self.encoder_frozen = frozen
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()
        return self

    def fine_parameters(self):
        """Encoder parameters: fine-tuned at a low LR. Empty while frozen."""
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def train_parameters(self):
        """Everything else: trained from scratch at the normal LR."""
        enc = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in enc]


# The N the model is trained with. Inference below that is legal -- the aggregator pools with a
# fixed learned query, so any N runs -- but accuracy falls with it: measured on the w2.0 model,
# blackbar scored 9.5% at N=1 against 30.5% at N=16, and pixelation 29.0% against 87.0%.
INFERENCE_FRAMES = 16

# Where 'aggregate_frame' looks for the trained model, under the weights cache.
CHECKPOINT_DIR = "facial"
CHECKPOINT_STEM = "facial.voxceleb.curriculum.w2d0"

# The optimiser is two thirds of a full checkpoint and inference never touches it, so the export
# is preferred where both exist -- 190 MB against 467 MB for identical weights.
CHECKPOINT_SUFFIXES = (".inference.pt", ".pt")


def facial_checkpoint_path(directory: str | Path | None = None) -> Path:
    """
    The conventional file for the trained aggregator.

    Unlike pix2pix there is **one** model rather than one per anonymization: ``Facial`` is
    conditioned on which method its inputs carry, so the same weights serve every branch of the
    sweep and the anonymization selects an embedding rather than a file.

    :param directory: The directory to look in, defaulting to ``<weights cache>/facial``
    :return: The first existing candidate, or the inference export's path if neither is there
    """
    from meles import weights

    root = Path(directory) if directory is not None else weights.cache_dir() / CHECKPOINT_DIR
    candidates = [root / (CHECKPOINT_STEM + suffix) for suffix in CHECKPOINT_SUFFIXES]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


class FacialFrameAggregator(FrameAggregator):
    """
    The trained ``Facial`` model as a frame aggregator: many anonymized frames in, one clear-ish
    face out.

    This is a **real aggregation**, unlike the ``frame_*`` de-anonymizations beside it. It returns
    a single frame per group however many it was given, so ``aggregate_frame`` numbers the result
    rather than keeping an original name, and the group's ``len`` changes.

    It takes the anonymization its inputs carry because the model is conditioned on it -- one
    model covers all four methods, and the wildcard picks the embedding. Passing the wrong one
    produces plausible frames scored against the wrong conditioning, which is why this is resolved
    from the job's ``{frame_anonymization}`` rather than from the aggregator key.

    **The weights load on first use, not in the constructor**, for the same reason as
    ``Pix2PixFrameAggregator``: ``resolve`` runs before a job knows whether its output is stale.

    Two mismatches against training are worth knowing, because both cost accuracy and neither
    fails loudly:

    - training aligned its crops using a detection run on the **clear** frame, which an attacker
      does not have. Here detection runs on the anonymized frame and falls back to the whole frame
      when it finds nothing, matching what ``embed_frame`` does with ``on_missing="whole_frame"``.
    - a group here is one identity-video, so all N frames come from a **single** video, where a
      training item spanned ~14. Frames of one video sit at 0.78 mean cosine against 0.42 across
      videos, so a group is closer to one observation repeated than to N independent ones. Expect
      materially lower numbers than the notebook's, and read them as a video-scope result.
    """

    def __init__(
        self,
        anonymizer: str,
        directory: str | Path | None = None,
        frames: int = INFERENCE_FRAMES,
        seed: int = 0,
        device: str | torch.device | None = None,
    ):
        """
        :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS`` the frames carry, which
            selects the model's method conditioning
        :param directory: Where the checkpoint is, defaulting to ``<weights cache>/facial``
        :param frames: How many frames to draw from a group, matching the trained N
        :param seed: Seeds the draw, so a rerun aggregates the same frames
        :param device: Where to run, defaulting to CUDA when available
        """
        super().__init__()
        self.anonymizer = anonymizer
        self.directory = directory
        self.frames = frames
        self.seed = seed
        self.device = device
        self._model = None
        # Empty rather than None: 'model' fills it, and every read happens after that.
        self._methods: list[str] = []

    def path(self) -> Path:
        """
        :return: The checkpoint to read
        :raises FileNotFoundError: If it is not there
        """
        path = facial_checkpoint_path(self.directory)
        if not path.exists():
            from meles import weights

            raise FileNotFoundError(
                f"No trained Facial model at {path} (or its '.pt' fallback). Train it with "
                f"src/notebooks/06_facial.ipynb and copy the checkpoint there, or point "
                f"{weights.CACHE_ENV} at where it is."
            )
        return path

    @property
    def model(self):
        """
        The trained model, loaded once on first use.

        :return: The model in eval mode
        :raises ValueError: If the checkpoint was not trained on this aggregator's anonymization
        """
        if self._model is None:
            # Imported here rather than at module scope: 'meles.training.facial' imports the
            # networks from *this* module, so a top-level import would be circular -- and it
            # would pull the trainer into every job that merely resolves a registry.
            from meles.training.facial import load_facial

            model, metadata = load_facial(self.path(), device=self.device)
            methods = list(metadata["methods"])
            if self.anonymizer not in methods:
                raise ValueError(
                    f"{self.path()} was trained on {methods}, which does not include "
                    f"{self.anonymizer!r}; it cannot condition on an anonymization it never saw"
                )
            self._model, self._methods = model, methods
            logger.info(
                "Loaded Facial aggregator",
                extra={"path": str(self.path()), "epoch": metadata.get("epoch"), "methods": methods},
            )
        return self._model

    def draw(self, images: Images) -> Images:
        """
        The ``frames`` images one reconstruction is built from.

        Seeded, so the same group yields the same draw on a rerun. A group holding fewer than N
        repeats rather than shrinking the input: the model takes a fixed N, and a repeated
        observation is a truthful item -- the attacker simply saw less.

        :param images: Every frame of the group
        :return: Exactly ``frames`` images
        """
        draw = random.Random(self.seed + len(images))
        if len(images) >= self.frames:
            return draw.sample(list(images), self.frames)
        return draw.choices(list(images), k=self.frames)

    def aggregate(self, images: Images) -> Images:
        """
        Collapse a group of anonymized frames into one reconstructed face.

        :param images: The group's frames, BGR uint8 at their native resolution
        :return: A single 112x112 BGR frame, or an empty list if the group was empty
        """
        if not images:
            return []

        import torch

        from meles.detection import TEMPLATE_SIZE, default_detector, prepare_face, whole_frame
        from meles.frame_aggregation.pix2pix import to_image, to_tensor
        from meles.training.facial import reconstruct

        crops = []
        for image in self.draw(images):
            area = default_detector.detect_best(image)
            # A frame with no detectable face is still an observation: black bars are supposed to
            # destroy the face, and dropping those frames would change what the sweep scores.
            crops.append(
                prepare_face(image, area, TEMPLATE_SIZE) if area is not None else whole_frame(image, TEMPLATE_SIZE)[0]
            )

        batch = torch.stack([to_tensor(crop, TEMPLATE_SIZE) for crop in crops])
        # Bound first: reading 'model' is what loads the checkpoint and fills '_methods'.
        model = self.model
        produced = reconstruct(model, batch, self._methods.index(self.anonymizer), device=self.device)
        return [to_image(produced[0])]
