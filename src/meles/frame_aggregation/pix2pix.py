"""
pix2pix (Isola et al., CVPR 2017) as a learned de-anonymization.

The frame-wise de-anonymizations in ``resample.py`` are fixed functions: they smooth what the
anonymization left behind and recover nothing that is not still in the pixels. A conditional GAN
is the other kind of attack -- it learns a prior over faces and *hallucinates* the detail back.
What it hallucinates is not the subject's face, which is exactly why the experiment measures
whether a recognizer still matches it to the right identity.

One model is trained per anonymization method (see ``FrameModel.md``), because a generator that
undoes 8x8 pixelation has learnt a different inverse than one that undoes a Gaussian blur. The
method the model was trained against is therefore part of the checkpoint (``Pix2PixCheckpoint``),
not something the caller is trusted to remember.

This module holds the networks, the losses and the checkpoint format. The training loop and the
paired dataset that feeds it live in ``meles.training.pix2pix`` and ``meles.training.pairs``.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from meles.common import Image

logger = logging.getLogger(__name__)

# The resolution the U-Net works at. The encoder halves the spatial size eight times, so the
# input edge has to be a multiple of 256 for the innermost feature map to come out at 1x1 --
# 256 is the smallest such size and the one the paper trains at.
SIZE = 256

# The version of the checkpoint payload below. Stored in the file so that a checkpoint written
# by an older layout fails loudly instead of loading into mismatched state dicts.
CHECKPOINT_FORMAT = 1


def _interpolation(source: tuple[int, int], target: tuple[int, int]) -> int:
    """
    The resampling kernel to resize with.

    ``INTER_AREA`` averages the pixels falling into each output pixel, which is the right thing
    when shrinking and degenerates to something close to nearest-neighbour when growing -- and
    the frames go both ways here, since CelebA is 178x218 (up to 256) while a YouTube Faces frame
    may be larger (down to 256).

    :param source: The (height, width) being resized from
    :param target: The (height, width) being resized to
    :return: The ``cv2`` interpolation flag
    """
    shrinking = target[0] * target[1] < source[0] * source[1]
    return cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR


def to_tensor(image: Image, size: int = SIZE) -> Tensor:
    """
    A frame as the tensor the networks take.

    Three conversions happen here and all three are easy to get silently wrong, so they are in
    one place rather than left to each caller: meles speaks OpenCV's **BGR** everywhere
    (``meles.common.Image`` is what ``cv2.imread`` returns) while the networks are trained on
    RGB; the layout goes from HWC to CHW; and the generator ends on a ``Tanh``, so the value
    range has to be **[-1, 1]** rather than [0, 1].

    :param image: The frame, BGR uint8 as everywhere else in meles
    :param size: The square edge length to resize to
    :return: A ``(3, size, size)`` float32 tensor in [-1, 1], RGB
    """
    height, width = image.shape[:2]
    resized = cv2.resize(image, (size, size), interpolation=_interpolation((height, width), (size, size)))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def to_image(tensor: Tensor, size: tuple[int, int] | None = None) -> Image:
    """
    The inverse of :func:`to_tensor`.

    Values are clamped before they are scaled back: the generator's ``Tanh`` cannot leave [-1, 1]
    but a target tensor that has been through an augmentation can, and ``uint8`` would wrap it
    around into the opposite brightness rather than saturate.

    :param tensor: A ``(3, H, W)`` tensor in [-1, 1], RGB, on any device
    :param size: The (height, width) to resize back to, or None to keep the tensor's own size
    :return: The frame, BGR uint8
    """
    array = ((tensor.detach().clamp(-1.0, 1.0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).round()
    bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if size is None or bgr.shape[:2] == size:
        return bgr
    return cv2.resize(bgr, (size[1], size[0]), interpolation=_interpolation(bgr.shape[:2], size))


def init_weights(module: nn.Module) -> None:
    """
    The paper's initialisation: every convolution drawn from ``N(0, 0.02)``.

    Not cosmetic. A GAN at PyTorch's default (Kaiming-uniform) initialisation starts with a
    discriminator that is already confident, which is the usual reason pix2pix training collapses
    in the first epoch. Apply with ``model.apply(init_weights)``.

    :param module: The module to initialise, called once per submodule by ``nn.Module.apply``
    """
    if isinstance(module, nn.Conv2d | nn.ConvTranspose2d):
        nn.init.normal_(module.weight, 0.0, 0.02)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.normal_(module.weight, 1.0, 0.02)
        nn.init.constant_(module.bias, 0.0)


class Pix2PixDiscriminator(nn.Module):
    """
    The 70x70 PatchGAN.

    It judges overlapping 70x70 patches rather than the whole image, which is what keeps the
    generator from settling on a plausible *average* face: a blurred output fails the patch test
    everywhere even when it is globally about right.

    **It returns logits, not probabilities.** The losses in :class:`Pix2Pix` are
    ``BCEWithLogitsLoss``, which applies the sigmoid itself in a numerically stable way; ending
    this network on a ``Sigmoid`` as well would square it, flatten the gradient at both ends and
    leave training stuck near the point where the discriminator is already right.
    """

    def __init__(self, in_channels: int = 3):
        """
        :param in_channels: The channels of one image; the network sees twice that, because the
            condition (the anonymized frame) and the image being judged are concatenated
        """
        super().__init__()

        def conv_block(in_c: int, out_c: int, stride: int) -> nn.Sequential:
            return nn.Sequential(
                # The bias is redundant in front of a BatchNorm, which subtracts the mean again.
                nn.Conv2d(in_c, out_c, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_c),
                nn.LeakyReLU(0.2, inplace=True),
            )

        self.model = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, kernel_size=4, stride=2, padding=1),  # C64, no BatchNorm
            nn.LeakyReLU(0.2, inplace=True),
            conv_block(64, 128, stride=2),  # C128
            conv_block(128, 256, stride=2),  # C256
            conv_block(256, 512, stride=1),  # C512, stride 1 -- this is what makes the patch 70x70
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # one logit per patch
        )
        self.apply(init_weights)

    def forward(self, condition: Tensor, image: Tensor) -> Tensor:
        """
        :param condition: The anonymized frames, ``(N, C, H, W)``
        :param image: The frames being judged -- real targets or generator output -- same shape
        :return: One logit per patch, ``(N, 1, H/8 - 2, W/8 - 2)``; 30x30 at 256x256
        """
        return self.model(torch.cat([condition, image], dim=1))


class Pix2PixDownSample(nn.Module):
    """One encoder block: halve the spatial size, double (or cap) the channels."""

    def __init__(self, in_channels: int, out_channels: int, apply_batch_norm: bool = True):
        """
        :param in_channels: The incoming channels
        :param out_channels: The outgoing channels
        :param apply_batch_norm: Whether to normalise. Off for the first block (the paper leaves
            C64 unnormalised) and off for the innermost one, where the feature map is 1x1 and a
            BatchNorm would have a single spatial sample per channel to estimate from -- with a
            batch of one it does not merely lose signal, it raises
            "Expected more than 1 value per channel when training"
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=not apply_batch_norm)
        ]
        if apply_batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, inplace=True))

        self.down = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(x)


class Pix2PixUpSample(nn.Module):
    """One decoder block: double the spatial size, then concatenate the encoder's skip."""

    def __init__(self, in_channels: int, out_channels: int, apply_dropout: bool = False):
        """
        :param in_channels: The incoming channels, i.e. the *concatenated* width the previous
            block produced
        :param out_channels: The channels this block produces before the skip is concatenated
        :param apply_dropout: Whether to drop half the activations. In pix2pix this is the only
            source of stochasticity -- there is no noise input -- so it stays on at inference
            time as well, which is why :meth:`Pix2PixGenerator.deanonymize` does not call ``eval``
        """
        super().__init__()

        layers: list[nn.Module] = [
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        ]
        if apply_dropout:
            layers.append(nn.Dropout(0.5))

        self.up = nn.Sequential(*layers)

    def forward(self, x: Tensor, skip: Tensor) -> Tensor:
        return torch.cat([self.up(x), skip], dim=1)


class Pix2PixGenerator(nn.Module):
    """
    The U-Net generator: eight encoder blocks down to 1x1, eight decoder blocks back to 256x256,
    with a skip connection across every level.

    The skips are what make this a de-anonymization rather than a face generator. Everything the
    anonymization left intact -- pose, hair, background, lighting -- reaches the output without
    having to survive the 1x1 bottleneck, so the network only has to invent the part that was
    actually destroyed.
    """

    def __init__(self, in_channels: int = 3, out_channels: int = 3):
        super().__init__()

        # Encoder. 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        self.down1 = Pix2PixDownSample(in_channels, 64, apply_batch_norm=False)  # C64
        self.down2 = Pix2PixDownSample(64, 128)  # C128
        self.down3 = Pix2PixDownSample(128, 256)  # C256
        self.down4 = Pix2PixDownSample(256, 512)  # C512
        self.down5 = Pix2PixDownSample(512, 512)  # C512
        self.down6 = Pix2PixDownSample(512, 512)  # C512
        self.down7 = Pix2PixDownSample(512, 512)  # C512
        self.down8 = Pix2PixDownSample(512, 512, apply_batch_norm=False)  # C512, 1x1: see the block

        # Decoder. Each block's 'in_channels' is the previous block's output *after* its skip was
        # concatenated, which is why they read 1024 rather than 512.
        self.up1 = Pix2PixUpSample(512, 512, apply_dropout=True)  # CD512
        self.up2 = Pix2PixUpSample(1024, 512, apply_dropout=True)  # CD1024
        self.up3 = Pix2PixUpSample(1024, 512, apply_dropout=True)  # CD1024
        self.up4 = Pix2PixUpSample(1024, 512)  # C1024
        self.up5 = Pix2PixUpSample(1024, 256)  # C1024
        self.up6 = Pix2PixUpSample(512, 128)  # C512
        self.up7 = Pix2PixUpSample(256, 64)  # C256

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, kernel_size=4, stride=2, padding=1),
            # Matches the [-1, 1] range 'to_tensor' puts the targets in.
            nn.Tanh(),
        )
        self.apply(init_weights)

    def forward(self, x: Tensor) -> Tensor:
        """
        :param x: The anonymized frames, ``(N, 3, 256, 256)`` in [-1, 1]
        :return: The de-anonymized frames, the same shape and range
        """
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        d7 = self.down7(d6)
        d8 = self.down8(d7)

        u1 = self.up1(d8, d7)
        u2 = self.up2(u1, d6)
        u3 = self.up3(u2, d5)
        u4 = self.up4(u3, d4)
        u5 = self.up5(u4, d3)
        u6 = self.up6(u5, d2)
        u7 = self.up7(u6, d1)

        return self.final(u7)

    @torch.no_grad()
    def deanonymize(self, images: list[Image], size: int = SIZE) -> list[Image]:
        """
        Run the generator over frames, returning frames.

        Each output is resized back to the size of the frame it came from, so this is a drop-in
        replacement for an anonymized frame anywhere downstream.

        Note that the model is *not* switched to ``eval`` here, deliberately: pix2pix has no noise
        input, so the decoder's dropout is its only stochasticity and the paper keeps it on at
        test time. The BatchNorms are the awkward part of that -- they then normalise by the batch
        rather than by running statistics, which makes a frame's output depend slightly on what it
        was batched with. Call ``generator.eval()`` first if you need that determinism instead.

        :param images: The anonymized frames, BGR uint8, any sizes
        :param size: The square edge length the model was trained at
        :return: One de-anonymized frame per input, BGR uint8, each at its input's size
        """
        if not images:
            return []
        device = next(self.parameters()).device
        batch = torch.stack([to_tensor(image, size) for image in images]).to(device)
        generated = self(batch)
        return [to_image(frame, image.shape[:2]) for frame, image in zip(generated, images)]


class Pix2Pix(nn.Module):
    """
    The generator, the discriminator and the two losses that couple them.

    Holding both as submodules means ``.to(device)``, ``.train()`` and ``.eval()`` reach both,
    which is the thing most easily forgotten when they are two loose objects.
    """

    def __init__(self, generator: Pix2PixGenerator | None = None, discriminator: Pix2PixDiscriminator | None = None):
        """
        :param generator: The generator, or None to build a default one
        :param discriminator: The discriminator, or None to build a default one
        """
        super().__init__()

        self.generator = generator if generator is not None else Pix2PixGenerator()
        self.discriminator = discriminator if discriminator is not None else Pix2PixDiscriminator()
        # Takes logits. See the note on Pix2PixDiscriminator: the network must not sigmoid itself.
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

    def generator_loss(
        self,
        fake_verdict: Tensor,
        fake_images: Tensor,
        real_images: Tensor,
        lambda_l1: float = 100.0,
    ) -> Tensor:
        """
        How wrong the generator was: it wants the discriminator to call its output real, and it
        wants the output to be close to the target.

        The L1 term is what ties the output to *this* face rather than to a plausible face, and
        ``lambda_l1`` is large because the adversarial term is on a per-patch log scale while L1
        is a mean absolute pixel difference in [0, 2].

        :param fake_verdict: The discriminator's logits for the generated frames
        :param fake_images: The generated frames
        :param real_images: The clear frames they were generated from
        :param lambda_l1: The weight of the reconstruction term, 100 in the paper
        :return: The scalar loss to backpropagate
        """
        adversarial_loss = self.criterion_gan(fake_verdict, torch.ones_like(fake_verdict))
        l1_loss = self.criterion_l1(fake_images, real_images)
        return adversarial_loss + lambda_l1 * l1_loss

    def discriminator_loss(self, real_verdict: Tensor, fake_verdict: Tensor) -> Tensor:
        """
        How wrong the discriminator was: real frames should score high, generated ones low.

        Halved as in the paper, so that the discriminator learns at roughly half the generator's
        rate and does not run away from it.

        :param real_verdict: Its logits for the clear frames
        :param fake_verdict: Its logits for the generated frames
        :return: The scalar loss to backpropagate
        """
        real_loss = self.criterion_gan(real_verdict, torch.ones_like(real_verdict))
        fake_loss = self.criterion_gan(fake_verdict, torch.zeros_like(fake_verdict))
        return (real_loss + fake_loss) * 0.5


@dataclass
class Pix2PixCheckpoint:
    """
    Everything needed to either resume training or run the trained generator.

    ``anonymizer`` and ``size`` are in here on purpose. One model is trained per anonymization
    method, so a checkpoint that did not name its method would be indistinguishable from any
    other and could be pointed at the wrong branch of the sweep, where it would produce
    plausible-looking frames and a quietly meaningless number.

    :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS`` this was trained against
    :param size: The square edge length it was trained at
    :param epoch: The number of epochs completed
    :param generator: The generator's ``state_dict``
    :param discriminator: The discriminator's ``state_dict``, absent in an inference-only export
    :param generator_optimizer: The generator optimiser's ``state_dict``, for resuming
    :param discriminator_optimizer: The discriminator optimiser's ``state_dict``, for resuming
    :param history: The per-epoch losses, keyed as in ``Pix2PixTrainer.history``
    """

    anonymizer: str
    size: int = SIZE
    epoch: int = 0
    generator: dict[str, Any] = field(default_factory=dict)
    discriminator: dict[str, Any] | None = None
    generator_optimizer: dict[str, Any] | None = None
    discriminator_optimizer: dict[str, Any] | None = None
    history: dict[str, list[float]] = field(default_factory=dict)


def save_checkpoint(checkpoint: Pix2PixCheckpoint, path: str | Path) -> Path:
    """
    Write a checkpoint.

    Written to a temporary file and renamed, because a run interrupted while overwriting the
    previous epoch's checkpoint would otherwise leave neither a complete old one nor a complete
    new one.

    :param checkpoint: What to write
    :param path: The file to write it to; its directory is created if needed
    :return: The path written
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "format": CHECKPOINT_FORMAT,
        "anonymizer": checkpoint.anonymizer,
        "size": checkpoint.size,
        "epoch": checkpoint.epoch,
        "generator": checkpoint.generator,
        "discriminator": checkpoint.discriminator,
        "generator_optimizer": checkpoint.generator_optimizer,
        "discriminator_optimizer": checkpoint.discriminator_optimizer,
        "history": checkpoint.history,
    }

    temporary = path.with_name(path.name + ".part")
    torch.save(payload, temporary)
    temporary.replace(path)
    logger.info(
        "Saved pix2pix checkpoint",
        extra={"path": str(path), "anonymizer": checkpoint.anonymizer, "epoch": checkpoint.epoch},
    )
    return path


def load_checkpoint(path: str | Path, map_location: str | torch.device = "cpu") -> Pix2PixCheckpoint:
    """
    Read a checkpoint written by :func:`save_checkpoint`.

    Loaded with ``weights_only=True``: the payload is nothing but tensors, strings, numbers and
    plain containers, so there is no reason to let a checkpoint file execute anything.

    :param path: The file to read
    :param map_location: The device to materialise the tensors on
    :return: The checkpoint
    :raises ValueError: If the file was written by a different checkpoint format
    """
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)

    stored = payload.get("format")
    if stored != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{path} is a pix2pix checkpoint of format {stored!r}, but this build reads "
            f"{CHECKPOINT_FORMAT}. Retrain it, or convert it by hand."
        )

    return Pix2PixCheckpoint(
        anonymizer=payload["anonymizer"],
        size=payload["size"],
        epoch=payload["epoch"],
        generator=payload["generator"],
        discriminator=payload.get("discriminator"),
        generator_optimizer=payload.get("generator_optimizer"),
        discriminator_optimizer=payload.get("discriminator_optimizer"),
        history=payload.get("history", {}),
    )


def load_generator(
    path: str | Path,
    device: str | torch.device = "cpu",
    anonymizer: str | None = None,
) -> Pix2PixGenerator:
    """
    Import a trained generator for inference, without its discriminator or optimisers.

    :param path: The checkpoint to read
    :param device: The device to put the generator on
    :param anonymizer: The anonymization the caller believes this model undoes. Given, it is
        checked against the checkpoint and a mismatch raises -- pointing a pixelation model at
        blurred frames produces output that looks fine and scores meaninglessly
    :return: The generator, on ``device``, with the checkpoint's weights loaded
    :raises ValueError: If ``anonymizer`` is given and does not match the checkpoint
    """
    checkpoint = load_checkpoint(path, map_location=device)
    if anonymizer is not None and checkpoint.anonymizer != anonymizer:
        raise ValueError(
            f"{path} was trained against the {checkpoint.anonymizer!r} anonymization, not {anonymizer!r}. "
            "One model is trained per anonymization method; using the wrong one silently measures nothing."
        )

    generator = Pix2PixGenerator()
    generator.load_state_dict(checkpoint.generator)
    return generator.to(device)
