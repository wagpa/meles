"""
pix2pix (Isola et al., CVPR 2017) as a learned de-anonymization, as SEBA runs it.

SEBA (kit-ps/seba, the Fantomas paper) does not implement pix2pix. It clones
``junyanz/pytorch-CycleGAN-and-pix2pix`` and shells out to it
(``src/deanonymization/face/pix2pix.py``), so "the model SEBA uses" means that repository's
``--model pix2pix`` defaults, invoked as::

    train.py --model pix2pix --direction BtoA --n_epochs E/2 --n_epochs_decay E/2
    test.py  --model test --netG unet_256 --norm batch --direction BtoA

``--model pix2pix`` sets ``netG=unet_256``, ``norm=batch`` and ``gan_mode=vanilla``; everything
else is a base default. ``BtoA`` is what makes it a de-anonymization rather than an
anonymization: SEBA puts the clear images in ``A`` and the anonymized ones in ``B``, so B is the
input and A is the target.

The networks below are a **port, not a reimplementation**. The module nesting and therefore the
``state_dict`` keys match upstream exactly, which is checked by
``src/tests/test_pix2pix_upstream.py``. That is deliberate: a published
``latest_net_G.pth`` from anyone running that repository -- SEBA's, had they released it --
loads into :class:`UnetGenerator` through :func:`import_upstream_generator` with no conversion.

What this file does **not** copy is the surrounding harness. The dataset side of upstream is a
directory of side-by-side AB images produced by ``combine_A_and_B.py``; here the pairs are made
in memory by applying one of ``meles.anonymization.ANONYMIZERS`` to a clear image, which is the
point of the exercise -- SEBA's anonymizers are not this project's.

One model is trained per anonymization method, because a generator that undoes 8x8 pixelation has
learnt a different inverse than one that undoes a Gaussian blur. The method is recorded in the
checkpoint (:class:`Pix2PixCheckpoint`) rather than left to the caller to remember.
"""

import functools
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import Tensor, nn

from meles import weights
from meles.common import Image, Images
from meles.frame_aggregation.aggregator import FrameAggregator

logger = logging.getLogger(__name__)

# 'crop_size', the resolution the networks actually see. The U-Net halves the spatial size eight
# times, so the edge has to be a multiple of 256 for the innermost feature map to reach 1x1.
SIZE = 256

# 'load_size'. Upstream's default preprocessing is 'resize_and_crop': scale to 286 and take a
# random 256 crop, which is where pix2pix's translation augmentation comes from.
LOAD_SIZE = 286

# 'ngf' / 'ndf', the filter counts, and 'n_layers_D'. Upstream base defaults.
NGF = 64
NDF = 64
N_LAYERS_D = 3

# 'lambda_L1', the weight of the generator's reconstruction term against its adversarial one.
LAMBDA_L1 = 100.0

# 'lr' and 'beta1'. beta1 is 0.5 rather than Adam's usual 0.9 because a GAN's loss surface moves
# under the optimiser, so heavy momentum chases a direction the other network has invalidated.
LEARNING_RATE = 2e-4
BETAS = (0.5, 0.999)

# 'n_epochs' and 'n_epochs_decay': the learning rate is held for the first and decayed linearly
# towards zero across the second. SEBA passes 'epochs // 2' for both.
N_EPOCHS = 100
N_EPOCHS_DECAY = 100

# 'batch_size'. Upstream's default, and unusual enough to be worth stating: pix2pix trains on
# single images. It is not an arbitrary choice -- with batch norm, a batch of one makes the
# normalisation per-image, which is most of the way to the instance norm the paper discusses.
BATCH_SIZE = 1

# 'init_gain', the standard deviation of the normal initialisation.
INIT_GAIN = 0.02

# The version of the checkpoint payload below. Stored in the file so that a checkpoint written
# by an older layout fails loudly instead of loading into mismatched state dicts.
CHECKPOINT_FORMAT = 2

# Where trained models are kept, next to the downloaded ONNX weights. Overridden by the same
# MELES_WEIGHTS_DIR the download cache uses, and gitignored for the same reason.
CHECKPOINT_DIR = "pix2pix"

# The suffix of an inference-only export, i.e. one written with 'inference_only=True' and holding
# the generator alone. A third of the size of the full checkpoint, which also carries the
# discriminator and both optimisers, and all the attack needs.
GENERATOR_SUFFIX = ".generator.pt"

# How many frames go through the generator at once. Unlike BATCH_SIZE above this is not upstream's
# -- it batches nothing at test time -- and it is not part of the method either, only of how the
# method is run: a group here is one identity-video, 48-75 frames on YouTube Faces, and a U-Net
# holds every level's feature maps and their skip connections for the whole batch at once.
# Batching is only sound because 'Pix2PixFrameAggregator' switches the model to eval mode; with
# the BatchNorms on batch statistics, where 'deanonymize' otherwise leaves them, a frame's output
# would depend on which other frames landed in its chunk.
INFERENCE_BATCH_SIZE = 16


def _interpolation(source: tuple[int, int], target: tuple[int, int]) -> int:
    """
    The resampling kernel to resize with.

    ``INTER_AREA`` averages the pixels falling into each output pixel, which is right when
    shrinking and degenerates to something close to nearest-neighbour when growing.

    :param source: The (height, width) being resized from
    :param target: The (height, width) being resized to
    :return: The ``cv2`` interpolation flag
    """
    shrinking = target[0] * target[1] < source[0] * source[1]
    return cv2.INTER_AREA if shrinking else cv2.INTER_CUBIC


def to_tensor(image: Image, size: int = SIZE) -> Tensor:
    """
    A frame as the tensor the networks take, i.e. upstream's ``ToTensor`` + ``Normalize(0.5, 0.5)``.

    Three conversions happen here and all three are easy to get silently wrong, so they are in
    one place: meles speaks OpenCV's **BGR** everywhere (``meles.common.Image`` is what
    ``cv2.imread`` returns) while the networks are trained on RGB; the layout goes from HWC to
    CHW; and the generator ends on a ``Tanh``, so the range is **[-1, 1]**, not [0, 1].

    :param image: The frame, BGR uint8 as everywhere else in meles
    :param size: The square edge length to resize to
    :return: A ``(3, size, size)`` float32 tensor in [-1, 1], RGB
    """
    height, width = image.shape[:2]
    resized = image
    if (height, width) != (size, size):
        resized = cv2.resize(image, (size, size), interpolation=_interpolation((height, width), (size, size)))
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float()
    return tensor / 127.5 - 1.0


def to_image(tensor: Tensor, size: tuple[int, int] | None = None) -> Image:
    """
    The inverse of :func:`to_tensor`, i.e. upstream's ``tensor2im``.

    Values are clamped before they are scaled back: the generator's ``Tanh`` cannot leave [-1, 1]
    but an augmented target tensor can, and ``uint8`` would wrap it around into the opposite
    brightness rather than saturate.

    :param tensor: A ``(3, H, W)`` tensor in [-1, 1], RGB, on any device
    :param size: The (height, width) to resize back to, or None to keep the tensor's own size
    :return: The frame, BGR uint8
    """
    array = ((tensor.detach().clamp(-1.0, 1.0).cpu().permute(1, 2, 0).numpy() + 1.0) * 127.5).round()
    bgr = cv2.cvtColor(array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    if size is None or bgr.shape[:2] == size:
        return bgr
    return cv2.resize(bgr, (size[1], size[0]), interpolation=_interpolation(bgr.shape[:2], size))


def init_weights(net: nn.Module, init_gain: float = INIT_GAIN) -> None:
    """
    Upstream's ``init_weights(net, 'normal', 0.02)``.

    Not cosmetic. A GAN at PyTorch's default (Kaiming-uniform) initialisation starts with a
    discriminator that is already confident, which is the usual reason pix2pix collapses in the
    first epoch.

    Matched to upstream down to which modules it touches: it keys off the *class name* containing
    "Conv", so ``Conv2d`` and ``ConvTranspose2d`` are both caught, and treats BatchNorm
    separately because a normalisation weight is a scale around one, not a matrix around zero.

    :param net: The network to initialise, in place
    :param init_gain: The standard deviation of the normal draw
    """

    def initialise(module: nn.Module) -> None:
        name = module.__class__.__name__
        # 'nn.Module.__getattr__' widens every attribute to include submodules, so the tensors are
        # narrowed explicitly rather than reached through 'module.weight.data' as upstream does.
        weight = getattr(module, "weight", None)
        bias = getattr(module, "bias", None)

        if isinstance(weight, Tensor) and ("Conv" in name or "Linear" in name):
            nn.init.normal_(weight.data, 0.0, init_gain)
        elif isinstance(weight, Tensor) and "BatchNorm2d" in name:
            nn.init.normal_(weight.data, 1.0, init_gain)
        else:
            return

        if isinstance(bias, Tensor):
            nn.init.constant_(bias.data, 0.0)

    net.apply(initialise)


class UnetSkipConnectionBlock(nn.Module):
    """
    One level of the U-Net, holding the level below it. Upstream's ``UnetSkipConnectionBlock``.

    The recursion is what makes the skips work out: a block downsamples, hands the result to its
    submodule, upsamples what comes back, and concatenates its own input onto it. So every level's
    input reaches its output without passing through the bottleneck.

    Two things about the layer order are load-bearing and look like mistakes if you expect the
    usual ``conv -> norm -> activation``. The activation comes **before** the convolution here,
    which is what lets the outermost block start with a bare convolution and the outermost
    upsample end on ``Tanh`` with no normalisation in between. And the innermost block's
    downsample has **no normalisation**: its feature map is 1x1, where a BatchNorm has one spatial
    sample per channel and, on a batch of one, raises outright.
    """

    def __init__(
        self,
        outer_nc: int,
        inner_nc: int,
        input_nc: int | None = None,
        submodule: "UnetSkipConnectionBlock | None" = None,
        outermost: bool = False,
        innermost: bool = False,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = False,
    ):
        """
        :param outer_nc: The channels this block produces before its skip is concatenated
        :param inner_nc: The channels its downsample produces, i.e. what the level below sees
        :param input_nc: The channels arriving, defaulting to ``outer_nc``
        :param submodule: The level below, or None for the innermost
        :param outermost: Whether this is the level that takes the image and emits the image
        :param innermost: Whether this is the 1x1 bottleneck
        :param norm_layer: The normalisation to use, ``BatchNorm2d`` for pix2pix
        :param use_dropout: Whether to drop half the upsampled activations. pix2pix has no noise
            input, so this is its only stochasticity, and it stays on at test time
        """
        super().__init__()
        self.outermost = outermost

        # Upstream decides this by asking whether the norm is InstanceNorm; with BatchNorm the
        # bias is redundant, because the normalisation subtracts the mean straight back off.
        use_bias = isinstance(norm_layer, functools.partial) and norm_layer.func is nn.InstanceNorm2d

        if input_nc is None:
            input_nc = outer_nc

        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        model: list[nn.Module]
        if innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, uprelu, upconv, upnorm]
        elif submodule is None:
            # Only the innermost block has nothing below it. Upstream lets this through and fails
            # inside nn.Sequential with a much less obvious message.
            raise ValueError("Only the innermost UnetSkipConnectionBlock may have no submodule")
        elif outermost:
            # Takes 'inner_nc * 2' because the submodule concatenated its own input onto its output.
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1)
            model = [downconv, submodule, uprelu, upconv, nn.Tanh()]
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc, kernel_size=4, stride=2, padding=1, bias=use_bias)
            model = [downrelu, downconv, downnorm, submodule, uprelu, upconv, upnorm]
            if use_dropout:
                model.append(nn.Dropout(0.5))

        self.model = nn.Sequential(*model)

    def forward(self, x: Tensor) -> Tensor:
        if self.outermost:
            return self.model(x)
        # The input first and the block's output second, as upstream, so that a checkpoint's
        # channel order means the same thing to the convolution above.
        return torch.cat([x, self.model(x)], 1)


class UnetGenerator(nn.Module):
    """
    Upstream's ``unet_256`` generator: eight downsamples to 1x1 and eight back up, skipped at
    every level.

    The skips are what make this a de-anonymization rather than a face generator. Everything the
    anonymization left intact -- pose, hair, background, lighting -- reaches the output without
    having to survive the bottleneck, so the network only has to invent what was destroyed.

    Built innermost-first, exactly as upstream does, which is why the ``state_dict`` keys nest as
    ``model.model.1.model.3....``: every level is a submodule of the one outside it.
    """

    def __init__(
        self,
        input_nc: int = 3,
        output_nc: int = 3,
        num_downs: int = 8,
        ngf: int = NGF,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
        use_dropout: bool = True,
    ):
        """
        :param input_nc: The channels of the anonymized frame
        :param output_nc: The channels of the de-anonymized frame
        :param num_downs: How many times to halve the spatial size; 8 is ``unet_256``
        :param ngf: The filter count of the outermost level, doubling inwards up to ``ngf * 8``
        :param norm_layer: The normalisation, ``BatchNorm2d`` for ``--norm batch``
        :param use_dropout: Whether the three innermost ``ngf * 8`` levels drop out. Upstream's
            ``--no_dropout`` defaults off, so pix2pix trains *with* dropout
        """
        super().__init__()

        block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, submodule=None, norm_layer=norm_layer, innermost=True)
        # The levels that keep their width. These are the paper's CD512 blocks, and the only ones
        # upstream applies dropout to.
        for _ in range(num_downs - 5):
            block = UnetSkipConnectionBlock(
                ngf * 8, ngf * 8, submodule=block, norm_layer=norm_layer, use_dropout=use_dropout
            )
        block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, submodule=block, norm_layer=norm_layer)
        block = UnetSkipConnectionBlock(ngf, ngf * 2, submodule=block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(
            output_nc, ngf, input_nc=input_nc, submodule=block, outermost=True, norm_layer=norm_layer
        )

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: The anonymized frames, ``(N, 3, 256, 256)`` in [-1, 1]
        :return: The de-anonymized frames, the same shape and range
        """
        return self.model(input)

    @torch.no_grad()
    def deanonymize(self, images: list[Image], size: int = SIZE) -> list[Image]:
        """
        Run the generator over frames, returning frames. The inference entry point.

        Each output is resized back to the size of the frame it came from, so this is a drop-in
        replacement for an anonymized frame anywhere downstream.

        Two notes on fidelity to SEBA's ``test.py`` invocation. It passes no ``--eval``, so
        upstream leaves the model in training mode at test time: dropout stays on (pix2pix's only
        stochasticity) and the BatchNorms use batch rather than running statistics. This does the
        same, by not switching modes -- call ``generator.eval()`` first if you want the
        deterministic variant. It also passes no ``--preprocess``, so upstream takes a *random
        256 crop of a 286 resize* at test time as well; that is not copied, because losing a
        random 10% border off every frame is plainly an oversight rather than part of the method,
        and it would make the attack's framing non-deterministic.

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


class NLayerDiscriminator(nn.Module):
    """
    Upstream's ``basic`` discriminator: the 70x70 PatchGAN.

    It judges overlapping 70x70 patches rather than the whole image, which is what keeps the
    generator from settling on a plausible *average* face: a blurred output fails the patch test
    everywhere even when it is globally about right.

    **It returns logits, not probabilities.** ``--gan_mode vanilla`` is ``BCEWithLogitsLoss``,
    which applies the sigmoid itself in a numerically stable way; a ``Sigmoid`` here as well would
    square it and flatten the gradient at both ends.

    It takes the condition and the judged image already concatenated, as upstream does -- the
    caller cats them, see :meth:`Pix2Pix.discriminate`.
    """

    def __init__(
        self,
        input_nc: int = 6,
        ndf: int = NDF,
        n_layers: int = N_LAYERS_D,
        norm_layer: Callable[[int], nn.Module] = nn.BatchNorm2d,
    ):
        """
        :param input_nc: The channels arriving, i.e. twice an image's, since the anonymized frame
            and the frame being judged are concatenated
        :param ndf: The filter count of the first layer, doubling up to ``ndf * 8``
        :param n_layers: How many strided layers before the final pair; 3 is what makes it 70x70
        :param norm_layer: The normalisation, ``BatchNorm2d`` for ``--norm batch``
        """
        super().__init__()
        use_bias = isinstance(norm_layer, functools.partial) and norm_layer.func is nn.InstanceNorm2d

        sequence: list[nn.Module] = [
            nn.Conv2d(input_nc, ndf, kernel_size=4, stride=2, padding=1),  # C64, no normalisation
            nn.LeakyReLU(0.2, True),
        ]

        multiplier = 1
        for layer in range(1, n_layers):
            previous, multiplier = multiplier, min(2**layer, 8)
            sequence += [
                nn.Conv2d(ndf * previous, ndf * multiplier, kernel_size=4, stride=2, padding=1, bias=use_bias),
                norm_layer(ndf * multiplier),
                nn.LeakyReLU(0.2, True),
            ]

        # Stride 1 from here on: this is what fixes the receptive field at 70x70 rather than
        # letting it grow to the whole image.
        previous, multiplier = multiplier, min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * previous, ndf * multiplier, kernel_size=4, stride=1, padding=1, bias=use_bias),
            norm_layer(ndf * multiplier),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * multiplier, 1, kernel_size=4, stride=1, padding=1),  # one logit per patch
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, input: Tensor) -> Tensor:
        """
        :param input: The condition and the judged image concatenated, ``(N, 6, H, W)``
        :return: One logit per patch, ``(N, 1, 30, 30)`` at 256x256
        """
        return self.model(input)


class Pix2Pix(nn.Module):
    """
    The generator, the discriminator and the objective, i.e. upstream's ``Pix2PixModel`` minus its
    checkpointing and visualisation.

    Holding both networks as submodules means ``.to(device)``, ``.train()`` and ``.eval()`` reach
    both, which is the thing most easily forgotten when they are two loose objects.
    """

    def __init__(self, generator: UnetGenerator | None = None, discriminator: NLayerDiscriminator | None = None):
        """
        :param generator: The generator, or None to build and initialise a default one
        :param discriminator: The discriminator, or None to build and initialise a default one
        """
        super().__init__()

        self.generator = generator if generator is not None else UnetGenerator()
        self.discriminator = discriminator if discriminator is not None else NLayerDiscriminator()
        if generator is None:
            init_weights(self.generator)
        if discriminator is None:
            init_weights(self.discriminator)

        # '--gan_mode vanilla'. Takes logits; see the note on NLayerDiscriminator.
        self.criterion_gan = nn.BCEWithLogitsLoss()
        self.criterion_l1 = nn.L1Loss()

    def discriminate(self, condition: Tensor, image: Tensor) -> Tensor:
        """
        The discriminator's verdict on a conditioned image.

        The concatenation lives here rather than inside the network so that the network's
        ``forward`` keeps upstream's single-argument signature.

        :param condition: The anonymized frames
        :param image: The frames being judged, real targets or generator output
        :return: One logit per patch
        """
        return self.discriminator(torch.cat([condition, image], 1))

    def generator_terms(self, fake_verdict: Tensor, fake_images: Tensor, real_images: Tensor) -> tuple[Tensor, Tensor]:
        """
        The generator's two objectives, **unweighted and kept apart**.

        Upstream tracks these as ``G_GAN`` and ``G_L1`` for the same reason they are split here:
        the weighted sum is not a quality signal and reading it as one is misleading. The
        adversarial term measures how far ahead the discriminator is, which in a healthy run
        *grows*; the L1 term measures reconstruction, which is what should fall.

        :param fake_verdict: The discriminator's logits for the generated frames
        :param fake_images: The generated frames
        :param real_images: The clear frames they were generated from
        :return: The adversarial term and the L1 term, both scalars
        """
        adversarial = self.criterion_gan(fake_verdict, torch.ones_like(fake_verdict))
        l1 = self.criterion_l1(fake_images, real_images)
        return adversarial, l1

    def generator_loss(
        self,
        fake_verdict: Tensor,
        fake_images: Tensor,
        real_images: Tensor,
        lambda_l1: float = LAMBDA_L1,
    ) -> Tensor:
        """
        Upstream's ``backward_G`` objective: fool the discriminator, and stay close to the target.

        ``lambda_l1`` is large because the adversarial term is on a per-patch log scale while L1
        is a mean absolute pixel difference in [0, 2].

        :param fake_verdict: The discriminator's logits for the generated frames
        :param fake_images: The generated frames
        :param real_images: The clear frames they were generated from
        :param lambda_l1: The weight of the reconstruction term
        :return: The scalar loss to backpropagate
        """
        adversarial, l1 = self.generator_terms(fake_verdict, fake_images, real_images)
        return adversarial + lambda_l1 * l1

    def discriminator_loss(self, real_verdict: Tensor, fake_verdict: Tensor) -> Tensor:
        """
        Upstream's ``backward_D``: real frames should score high, generated ones low.

        Halved as upstream does, so that the discriminator learns at roughly half the generator's
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
    method, so a checkpoint that did not name its method would be indistinguishable from any other
    and could be pointed at the wrong branch of the sweep, where it would produce plausible-looking
    frames and a quietly meaningless number.

    :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS`` this was trained against
    :param size: The square edge length it was trained at
    :param epoch: The number of epochs completed
    :param generator: The generator's ``state_dict``, in upstream's key layout
    :param discriminator: The discriminator's ``state_dict``, absent in an inference-only export
    :param generator_optimizer: The generator optimiser's ``state_dict``, for resuming
    :param discriminator_optimizer: The discriminator optimiser's ``state_dict``, for resuming
    :param history: The per-epoch losses, keyed as in ``Pix2PixTrainer.history``
    :param schedule: The ``(n_epochs, n_epochs_decay)`` the run was configured with, so that a
        resumed run picks the learning rate schedule back up where it left it
    """

    anonymizer: str
    size: int = SIZE
    epoch: int = 0
    generator: dict[str, Any] = field(default_factory=dict)
    discriminator: dict[str, Any] | None = None
    generator_optimizer: dict[str, Any] | None = None
    discriminator_optimizer: dict[str, Any] | None = None
    history: dict[str, list[float]] = field(default_factory=dict)
    schedule: tuple[int, int] = (N_EPOCHS, N_EPOCHS_DECAY)


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
        "schedule": list(checkpoint.schedule),
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

    schedule = payload.get("schedule") or (N_EPOCHS, N_EPOCHS_DECAY)
    return Pix2PixCheckpoint(
        anonymizer=payload["anonymizer"],
        size=payload["size"],
        epoch=payload["epoch"],
        generator=payload["generator"],
        discriminator=payload.get("discriminator"),
        generator_optimizer=payload.get("generator_optimizer"),
        discriminator_optimizer=payload.get("discriminator_optimizer"),
        history=payload.get("history", {}),
        schedule=(int(schedule[0]), int(schedule[1])),
    )


def default_device() -> torch.device:
    """
    :return: The CUDA device if there is one, otherwise the CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(anonymizer: str, directory: str | Path | None = None) -> Path:
    """
    The conventional file for the model trained against one anonymization.

    This lives beside the checkpoint format rather than in ``meles.training`` because inference
    has to find the file too, and the frame aggregator cannot import the trainer: the trainer
    imports the networks from here.

    :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS``
    :param directory: The directory to put it in, defaulting to ``<weights cache>/pix2pix``
    :return: The checkpoint path, which need not exist yet
    """
    root = Path(directory) if directory is not None else weights.cache_dir() / CHECKPOINT_DIR
    return root / f"{anonymizer}.pt"


def load_generator(
    path: str | Path,
    device: str | torch.device = "cpu",
    anonymizer: str | None = None,
) -> UnetGenerator:
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

    generator = UnetGenerator()
    generator.load_state_dict(checkpoint.generator)
    return generator.to(device)


def import_upstream_generator(path: str | Path, device: str | torch.device = "cpu") -> UnetGenerator:
    """
    Load a ``latest_net_G.pth`` written by ``junyanz/pytorch-CycleGAN-and-pix2pix`` itself.

    This is the reason :class:`UnetGenerator` is a port rather than a reimplementation. Anyone
    who ran that repository -- SEBA, had they published their weights -- produced a file that
    drops straight in here, with no key remapping.

    It has to be a ``unet_256`` trained with ``--norm batch``, which is what ``--model pix2pix``
    selects; a ``resnet_9blocks`` or an instance-norm run will not match and will say so.

    :param path: The '.pth' to read
    :param device: The device to put the generator on
    :return: The generator, on ``device``
    :raises ValueError: If the file is not a ``unet_256`` batch-norm generator
    """
    state = torch.load(Path(path), map_location=device, weights_only=True)

    generator = UnetGenerator()
    try:
        generator.load_state_dict(state)
    except RuntimeError as mismatch:
        raise ValueError(
            f"{path} is not a 'unet_256' generator trained with '--norm batch'. pix2pix's own "
            f"defaults produce one; a different '--netG' or '--norm' does not. ({mismatch})"
        ) from mismatch

    logger.info("Imported an upstream pix2pix generator", extra={"path": str(path)})
    return generator.to(device)


class Pix2PixFrameAggregator(FrameAggregator):
    """
    The trained generator as a frame aggregator, i.e. SEBA's pix2pix attack inside the sweep.

    Like the resample attack this is a **de-anonymization, not an aggregation**: it returns one
    frame per input and combines nothing, which is what the ``frame_*`` prefix on its registry key
    says and what lets ``aggregate_frame`` keep each frame's original name.

    It is specific to the anonymization it undoes -- one model is trained per method -- so it takes
    the anonymizer rather than a path and finds the checkpoint by the same convention the trainer
    writes it under. ``load_generator`` is given the anonymizer as well, so a model pointed at the
    wrong branch of the sweep raises instead of producing plausible frames and a meaningless score.

    **The weights are loaded on first use, not in the constructor.** The workflow's registries build
    every entry at import time, so a constructor that read its checkpoint would load a few hundred
    megabytes into every ``aggregate_frame`` job, including the ones running the ``none`` baseline.
    """

    def __init__(
        self,
        anonymizer: str,
        directory: str | Path | None = None,
        batch_size: int = INFERENCE_BATCH_SIZE,
    ):
        """
        :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS`` this undoes, which both
            names the checkpoint and is checked against what the checkpoint says it was trained on
        :param directory: Where the checkpoints are, defaulting to ``<weights cache>/pix2pix``
        :param batch_size: How many frames to run through the generator at once
        """
        super().__init__()
        self.anonymizer = anonymizer
        self.directory = directory
        self.batch_size = batch_size
        self._generator: UnetGenerator | None = None

    def path(self) -> Path:
        """
        The checkpoint to read.

        The inference-only export is preferred and the full checkpoint is the fallback, because
        the two are written side by side and only the generator is needed here.

        :return: The generator export if it exists, otherwise the full checkpoint
        :raises FileNotFoundError: If neither exists
        """
        full = checkpoint_path(self.anonymizer, self.directory)
        export = full.with_suffix(GENERATOR_SUFFIX)
        if export.exists():
            return export
        if full.exists():
            return full
        raise FileNotFoundError(
            f"No pix2pix model for the {self.anonymizer!r} anonymization at {export} or {full}. "
            f"One model is trained per method; train it (see Pix2PixTraining.md) or point "
            f"{weights.CACHE_ENV} at where it is."
        )

    @property
    def generator(self) -> UnetGenerator:
        """
        :return: The generator, loaded on first access and kept for the rest of the job
        """
        generator = self._generator
        if generator is None:
            generator = load_generator(self.path(), device=default_device(), anonymizer=self.anonymizer)
            # 'deanonymize' deliberately leaves the mode alone, which keeps SEBA's test-time
            # behaviour: it passes no '--eval', so dropout stays on and the BatchNorms use batch
            # rather than running statistics. That is not wanted here. The sweep scores these
            # frames, so the same video has to de-anonymize the same way twice, and only the
            # running statistics make a frame's output independent of the chunk it landed in
            # (measured: identical but for float rounding, 1/255 on 2 pixels in 100k).
            generator.eval()
            self._generator = generator
        return generator

    def aggregate(self, images: Images) -> Images:
        """
        :param images: The anonymized frames of one group
        :return: One de-anonymized frame per input, each at its input's size
        """
        frames = list(images)
        deanonymized: list[Image] = []
        for start in range(0, len(frames), self.batch_size):
            deanonymized.extend(self.generator.deanonymize(frames[start : start + self.batch_size]))
        return deanonymized
