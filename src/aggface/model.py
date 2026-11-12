"""
AggFace: several anonymized frames of one person in, a single clear face out.

Three parts in a row:

- the **encoder** turns each frame into a 512 x 7 x 7 feature map. It is an ArcFace-style
  iresnet34 taken up to the map after ``layer4``/``bn2``, so it starts out already knowing what
  makes faces distinguishable rather than learning it from this data.
- the **aggregator** is one transformer block. A set of learned queries, one per cell of the 7 x 7
  grid, cross-attends over the position-embedded tokens of all N frames at once, so every output
  cell can draw on whichever frames happen to show it best. The result is one 512 x 7 x 7 map.
- the **decoder** upsamples that back to a 3 x 112 x 112 face.

The bottleneck is the point and the difficulty: 25,088 values carry the whole reconstruction, and
unlike a U-Net there is no skip path from the inputs, so every pixel of detail has to survive the
pooling.
"""

import logging
from collections.abc import Iterator
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch import nn

from aggface.iresnet import iresnet34
from meles import weights

logger = logging.getLogger(__name__)

# The encoder's starting weights: an iresnet34 backbone trained for face recognition, from the
# InsightFace arcface_torch model zoo
# (https://github.com/deepinsight/insightface/tree/master/model_zoo). Those backbones are
# distributed through OneDrive rather than from a stable URL, so there is nothing to download
# from automatically; the file is placed in the weights cache by hand and pinned by digest. The
# exact training recipe behind this particular file is not recorded upstream, which is why the
# digest rather than a name is what identifies it.
IRESNET34 = weights.WeightSpec(
    file_name="iresnet34.pth",
    sha256="d90db283c24048efd8e17488146219fc5c70876bcff61647a647b6a3c872ba5d",
    source="the InsightFace arcface_torch model zoo (see notebooks/06 for the download link)",
)

# The size of the feature grid the encoder produces, and the channels in it.
GRID = 7
CHANNELS = 512


class Encoder(nn.Module):
    """An iresnet34 up to its 7 x 7 feature map, one map per input frame."""

    def __init__(self, pretrained: bool = True) -> None:
        """
        :param pretrained: Whether to load the recognition weights. False leaves the encoder
            randomly initialised, which is only useful for tests
        """
        super().__init__()
        self.model = iresnet34()
        if pretrained:
            state = torch.load(weights.resolve(IRESNET34), map_location="cpu", weights_only=True)
            # Checkpoints written under DataParallel carry a 'module.' prefix.
            self.model.load_state_dict({key.removeprefix("module."): value for key, value in state.items()})

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: Frames, (B, 3, 112, 112), RGB in [-1, 1]
        :return: The feature maps, (B, 512, 7, 7)
        """
        # Stops at the feature map. The rest of the network produces a recognition embedding from
        # a 25088 -> 512 linear layer and a BatchNorm1d, which only the identity loss uses; running
        # it here would spend 12.8 M parameters on a discarded result, and would fail outright on a
        # single frame because a BatchNorm1d in training mode needs more than one sample.
        return self.model(x, hidden_only=True)


class Aggregator(nn.Module):
    """N feature maps in, one out. A single transformer block: cross-attention, then a feed-forward."""

    def __init__(
        self, channels: int = CHANNELS, heads: int = 8, grid: int = GRID, ff_mult: int = 4, methods: int = 0
    ) -> None:
        """
        :param channels: Channels in the feature maps
        :param heads: Attention heads
        :param grid: Side length of the feature grid
        :param ff_mult: How much wider the feed-forward layer is than the channels
        :param methods: How many anonymizations one model covers. The query is otherwise a single
            learned constant, identical for every input, so one model covering several
            degradations would have to serve them all from one starting point - and undoing a blur
            is not the same operation as inventing a face behind a black bar. A learned embedding
            per method, added to the query, is the cheapest way to let it specialise. 0 keeps the
            unconditional query, which is right when one model is trained per anonymization
        """
        super().__init__()
        self.grid = grid
        # One position embedding per grid cell, shared by every frame, so the attention can tell
        # where in the face a token came from. The queries are the same shape: one per output cell.
        self.position = nn.Parameter(torch.randn(1, grid * grid, channels) * 0.02)
        self.query = nn.Parameter(torch.randn(1, grid * grid, channels) * 0.02)
        # Zero-initialised, so an untrained conditional model starts out exactly as the
        # unconditional one rather than with the query displaced at random.
        self.method = nn.Embedding(methods, channels) if methods else None
        if self.method is not None:
            nn.init.zeros_(self.method.weight)

        self.attention = nn.MultiheadAttention(channels, heads, batch_first=True)
        self.norm_query = nn.LayerNorm(channels)
        self.norm_tokens = nn.LayerNorm(channels)
        self.norm_feedforward = nn.LayerNorm(channels)
        self.feedforward = nn.Sequential(
            nn.Linear(channels, channels * ff_mult),
            nn.GELU(),
            nn.Linear(channels * ff_mult, channels),
        )

    def forward(self, x: torch.Tensor, method: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: The frames' feature maps, (B, N, C, H, W)
        :param method: Which anonymization each item carries, (B,), or None when unconditional
        :return: One feature map per item, (B, C, H, W)
        """
        batch, frames, channels, height, width = x.shape
        tokens = x.flatten(0, 1).flatten(2).transpose(1, 2) + self.position  # (B*N, H*W, C)
        tokens = tokens.reshape(batch, frames * height * width, channels)  # every frame at once
        query = self.query.expand(batch, -1, -1)
        if self.method is not None and method is not None:
            query = query + self.method(method).unsqueeze(1)  # broadcast over the grid

        pooled = (
            query
            + self.attention(
                self.norm_query(query), self.norm_tokens(tokens), self.norm_tokens(tokens), need_weights=False
            )[0]
        )
        pooled = pooled + self.feedforward(self.norm_feedforward(pooled))
        return pooled.transpose(1, 2).reshape(batch, channels, height, width)


class DecoderBlock(nn.Module):
    """One 2x upsampling step: two convolutions on a residual branch, past a 1x1 skip."""

    def __init__(self, channels_in: int, channels_out: int, groups: int = 8, dropout: float = 0.0) -> None:
        """
        :param channels_in: Channels arriving
        :param channels_out: Channels produced
        :param groups: ``GroupNorm`` groups
        :param dropout: Channel dropout on the residual branch
        """
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv1 = nn.Conv2d(channels_in, channels_out, 3, 1, 1)
        self.conv2 = nn.Conv2d(channels_out, channels_out, 3, 1, 1)
        self.norm1 = nn.GroupNorm(groups, channels_out)
        self.norm2 = nn.GroupNorm(groups, channels_out)
        self.skip = nn.Conv2d(channels_in, channels_out, 1) if channels_in != channels_out else nn.Identity()
        # Dropout2d, not Dropout: these are convolutional features, where neighbouring pixels of
        # one channel are strongly correlated, so dropping individual activations leaks through
        # the ones beside them. Dropping whole channels is what regularises here.
        self.drop = nn.Dropout2d(dropout) if dropout else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The incoming map, (B, channels_in, H, W)
        :return: The map at twice the size, (B, channels_out, 2H, 2W)
        """
        x = self.up(x)
        hidden = functional.silu(self.norm1(self.conv1(x)))
        hidden = functional.silu(self.norm2(self.conv2(hidden)))
        return self.drop(hidden) + self.skip(x)


class Decoder(nn.Module):
    """Expands the aggregator's 512 x 7 x 7 into a 3 x 112 x 112 face."""

    def __init__(
        self,
        channels_in: int = CHANNELS,
        channels: tuple[int, ...] = (256, 192, 128, 64),
        width: float = 1.0,
        groups: int = 8,
        dropout: float = 0.0,
    ) -> None:
        """
        :param channels_in: Channels arriving from the aggregator
        :param channels: The channel ladder, one entry per 2x upsampling step. Four steps take 7
            to 112, so changing the length changes the output size
        :param width: Multiplies every entry of ``channels``, for testing whether the bottleneck
            or the decoder's own capacity is what limits the result. Each result is rounded to a
            multiple of ``groups``, which ``GroupNorm`` requires
        :param groups: ``GroupNorm`` groups
        :param dropout: Channel dropout on each block's residual branch, worth turning on
            together with ``width`` because a wider decoder overfits harder
        """
        super().__init__()
        self.width = width
        self.dropout = dropout
        # Rounded to a multiple of 'groups' so GroupNorm stays valid at any width, never to zero.
        self.channels = tuple(max(groups, int(round(c * width / groups)) * groups) for c in channels)

        self.blocks = nn.ModuleList()
        for channels_out in self.channels:
            self.blocks.append(DecoderBlock(channels_in, channels_out, groups=groups, dropout=dropout))
            channels_in = channels_out
        self.out = nn.Sequential(
            nn.GroupNorm(groups, channels_in),
            nn.SiLU(),
            nn.Conv2d(channels_in, 3, 3, 1, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        :param x: The pooled feature map, (B, channels_in, 7, 7)
        :return: The reconstructed face, (B, 3, 112, 112), RGB in [-1, 1]
        """
        for block in self.blocks:
            x = block(x)
        return torch.tanh(self.out(x))


class AggFace(nn.Module):
    """The whole model: encode every frame, pool them into one map, decode a face from it."""

    def __init__(self, methods: int = 0, pretrained: bool = True, width: float = 1.0, dropout: float = 0.0) -> None:
        """
        ``width`` and ``dropout`` size and regularise the decoder, and **they change the parameter
        shapes**, so a checkpoint and the model reading it have to agree on them.

        :param methods: How many anonymizations one model covers, conditioning the aggregator's
            query on which one its inputs carry. The anonymization is known at inference - a sweep
            leg names it - so supplying it is not information the attacker would lack
        :param pretrained: Whether the encoder starts from the recognition weights
        :param width: Decoder width multiplier
        :param dropout: Decoder channel dropout
        """
        super().__init__()
        self.encoder_frozen = False
        self.methods = methods
        self.encoder = Encoder(pretrained=pretrained)
        self.aggregator = Aggregator(methods=methods)
        self.decoder = Decoder(width=width, dropout=dropout)

    def forward(self, x: torch.Tensor, method: torch.Tensor | None = None) -> torch.Tensor:
        """
        :param x: The anonymized frames of one item, (B, N, 3, 112, 112), RGB in [-1, 1]
        :param method: Which anonymization each item carries, (B,), or None when unconditional
        :return: The reconstructed face, (B, 3, 112, 112), RGB in [-1, 1]
        """
        batch, frames = x.shape[:2]
        if self.encoder_frozen:
            with torch.no_grad():
                encoded = self.encoder(x.flatten(0, 1))
        else:
            encoded = self.encoder(x.flatten(0, 1))
        return self.decoder(self.aggregator(encoded.view(batch, frames, *encoded.shape[1:]), method))

    def train(self, mode: bool = True) -> "AggFace":
        """
        Keeps the encoder in eval mode while it is frozen, so that its BatchNorm running
        statistics are not overwritten by data it is not learning from.

        :param mode: Whether to enter training mode
        :return: This model
        """
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def freeze_encoder(self, frozen: bool = True) -> "AggFace":
        """
        :param frozen: Whether the encoder is held fixed
        :return: This model
        """
        self.encoder_frozen = frozen
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(not frozen)
        if frozen:
            self.encoder.eval()
        return self

    def fine_parameters(self) -> list[nn.Parameter]:
        """
        :return: The encoder's parameters, fine-tuned at a low rate; empty while it is frozen
        """
        return [parameter for parameter in self.encoder.parameters() if parameter.requires_grad]

    def new_parameters(self) -> list[nn.Parameter]:
        """
        :return: Everything outside the encoder, trained from scratch at the normal rate
        """
        encoder = {id(parameter) for parameter in self.encoder.parameters()}
        return [parameter for parameter in self.parameters() if id(parameter) not in encoder]


def parameter_counts(model: nn.Module) -> Iterator[tuple[str, int]]:
    """
    The number of parameters in each immediate child of a model.

    :param model: The model to count
    :return: Pairs of child name and parameter count
    """
    for name, child in model.named_children():
        yield name, sum(parameter.numel() for parameter in child.parameters())


def checkpoint_path(name: str, seed: int, directory: str | Path | None = None) -> Path:
    """
    Where a trained model is kept: one file per run name and seed.

    :param name: The run name
    :param seed: The seed the run used
    :param directory: Where to look, defaulting to ``<weights cache>/aggface``
    :return: The checkpoint path, which need not exist yet
    """
    root = Path(directory) if directory is not None else weights.cache_dir() / "aggface"
    return root / f"{name}.seed{seed}.pt"
