"""
Pins the networks to ``junyanz/pytorch-CycleGAN-and-pix2pix``'s ``--model pix2pix``.

SEBA does not implement pix2pix; it clones that repository and shells out to it. So "the model
SEBA uses" is defined by upstream's code, not by this one, and the networks in
``meles.frame_aggregation.pix2pix`` are a port whose ``state_dict`` is meant to be
*interchangeable* with it -- a ``latest_net_G.pth` from any run of that repository loads here
without conversion, which is the whole reason the module nesting is copied rather than tidied.

Nothing in this file can check that against upstream directly, because upstream is not a
dependency. What it can do is pin the observable consequences: the exact key/shape layout
(as a digest taken from the real thing), the parameter counts, and the 70x70 patch output. Any
edit that "cleans up" the block structure moves the digest and fails here, which is the point --
that edit would silently make published weights unloadable.

The digests below were taken from ``define_G(3, 3, 64, 'unet_256', norm='batch',
use_dropout=True)`` and ``define_D(6, 64, 'basic', n_layers_D=3, norm='batch')`` at
pytorch-CycleGAN-and-pix2pix HEAD, and the ported networks were additionally verified to produce
bit-identical output to upstream's when handed the same weights.
"""

import hashlib

import pytest
import torch

from meles.frame_aggregation.pix2pix import (
    LAMBDA_L1,
    LEARNING_RATE,
    N_EPOCHS,
    N_EPOCHS_DECAY,
    NLayerDiscriminator,
    Pix2Pix,
    UnetGenerator,
    import_upstream_generator,
)
from meles.training import Pix2PixTrainer

# sha256 over "key:shape" lines of the upstream state dicts, in order.
GENERATOR_FINGERPRINT = "6063f0577dbd94a587378b01c4ff7935f184335f9556da5fd18fd499c9f0ab67"
DISCRIMINATOR_FINGERPRINT = "ed406812769f9ca83b4a908270a89098ddc37b31466a1987d431558357b05801"

GENERATOR_PARAMETERS = 54_413_955
DISCRIMINATOR_PARAMETERS = 2_768_705


def fingerprint(module: torch.nn.Module) -> str:
    lines = [f"{key}:{tuple(value.shape)}" for key, value in module.state_dict().items()]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def test_the_generator_matches_upstreams_unet_256() -> None:
    """Same keys, same shapes, same order -- so upstream weights load with no remapping."""
    generator = UnetGenerator()

    assert fingerprint(generator) == GENERATOR_FINGERPRINT
    assert sum(parameter.numel() for parameter in generator.parameters()) == GENERATOR_PARAMETERS


def test_the_discriminator_matches_upstreams_basic_patchgan() -> None:
    assert fingerprint(NLayerDiscriminator()) == DISCRIMINATOR_FINGERPRINT
    assert sum(p.numel() for p in NLayerDiscriminator().parameters()) == DISCRIMINATOR_PARAMETERS


def test_the_nesting_is_upstreams_recursive_one() -> None:
    """
    A flat encoder/decoder would give the same parameter count and a different key layout, which
    is exactly the substitution that makes published weights silently unloadable.
    """
    keys = list(UnetGenerator().state_dict())

    assert keys[0] == "model.model.0.weight"
    # Eight nested levels, so the innermost weight is reached through seven submodules.
    assert max(key.count("model.") for key in keys) == 9


def test_an_upstream_checkpoint_imports(tmp_path) -> None:
    """The stated purpose of the port: someone else's 'latest_net_G.pth' drops straight in."""
    path = tmp_path / "latest_net_G.pth"
    torch.save(UnetGenerator().state_dict(), path)

    assert isinstance(import_upstream_generator(path), UnetGenerator)


def test_importing_a_different_architecture_is_refused(tmp_path) -> None:
    """A 'resnet_9blocks' or instance-norm run is not what '--model pix2pix' produces."""
    path = tmp_path / "wrong_net_G.pth"
    torch.save({"model.0.weight": torch.zeros(3, 3)}, path)

    with pytest.raises(ValueError, match="unet_256"):
        import_upstream_generator(path)


def test_the_discriminator_sees_the_pair_concatenated() -> None:
    """
    Upstream's D takes one six-channel tensor, not two three-channel ones, and the caller does the
    concatenation. Keeping that signature is part of being a drop-in for its weights.
    """
    model = Pix2Pix()

    with torch.no_grad():
        verdict = model.discriminate(torch.randn(2, 3, 256, 256), torch.randn(2, 3, 256, 256))

    # 70x70 receptive field over a 256x256 input lands on a 30x30 grid of patch logits.
    assert verdict.shape == (2, 1, 30, 30)
    assert float(verdict.min()) < 0.0 < float(verdict.max()), "a Sigmoid would have clamped these into (0, 1)"


def test_the_hyperparameters_are_upstreams_pix2pix_defaults() -> None:
    """These are what SEBA gets by overriding only the epoch counts and the direction."""
    assert (LEARNING_RATE, LAMBDA_L1) == (2e-4, 100.0)
    assert (N_EPOCHS, N_EPOCHS_DECAY) == (100, 100)

    trainer = Pix2PixTrainer("none", device="cpu")
    assert trainer.generator_optimizer.param_groups[0]["betas"] == (0.5, 0.999)
    assert trainer.discriminator_optimizer.param_groups[0]["lr"] == LEARNING_RATE


def test_the_learning_rate_follows_upstreams_linear_policy() -> None:
    """
    Constant across ``n_epochs``, then linear towards -- but never reaching -- zero across
    ``n_epochs_decay``. The divisor is ``n_epochs_decay + 1``, which is upstream's off-by-one and
    is reproduced deliberately.
    """
    trainer = Pix2PixTrainer("none", device="cpu", n_epochs=100, n_epochs_decay=100)

    assert trainer.epochs == 200
    assert trainer.learning_rate_at(1) == pytest.approx(2e-4)
    assert trainer.learning_rate_at(100) == pytest.approx(2e-4)
    assert trainer.learning_rate_at(101) == pytest.approx(2e-4 * (1 - 1 / 101))
    assert trainer.learning_rate_at(200) == pytest.approx(2e-4 * (1 - 100 / 101))
    assert trainer.learning_rate_at(200) > 0.0


def test_the_learning_rate_reaches_the_optimisers() -> None:
    """A schedule that is computed but never applied is the easy way to get this wrong."""
    trainer = Pix2PixTrainer("none", device="cpu", n_epochs=1, n_epochs_decay=1)

    applied = trainer._apply_learning_rate(2)

    assert applied == pytest.approx(2e-4 * 0.5)
    for optimizer in (trainer.generator_optimizer, trainer.discriminator_optimizer):
        assert optimizer.param_groups[0]["lr"] == pytest.approx(applied)


def test_a_batch_of_one_trains() -> None:
    """
    Upstream's ``--batch_size`` default is 1, so this has to work: the innermost encoder block
    runs at 1x1, where a BatchNorm would raise on a single sample. It does not, because upstream
    leaves the normalisation off that one block.
    """
    trainer = Pix2PixTrainer("none", device="cpu")

    losses = trainer.step(torch.randn(1, 3, 256, 256), torch.randn(1, 3, 256, 256))

    assert losses.l1 > 0.0
