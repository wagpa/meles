"""
Training a pix2pix de-anonymizer, one model per anonymization method.

This is upstream's ``train.py`` loop, which SEBA invokes as
``train.py --model pix2pix --direction BtoA --n_epochs E/2 --n_epochs_decay E/2``: the same step
order, the same objective, the same linear learning-rate decay. What it is not is upstream's
harness -- the visdom server, the AB-folder dataset and the per-iteration HTML are all gone, and
a validation pass is added, which upstream has none of (SEBA's ``train_rate`` writes a ``val``
folder that ``train.py`` never opens).

The trainer owns the model, the two optimisers, the schedule and the loss history together,
because those are what a checkpoint has to hold for training to be resumable -- and a 200-epoch
run will be interrupted.

The ordering inside a step is the part that is easy to get subtly wrong, so it is spelled out in
:meth:`Pix2PixTrainer.step`.
"""

import logging
import math
import time
from collections.abc import Callable, Iterable
from dataclasses import astuple, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader

from meles import weights
from meles.frame_aggregation.pix2pix import (
    BETAS,
    LAMBDA_L1,
    LEARNING_RATE,
    N_EPOCHS,
    N_EPOCHS_DECAY,
    SIZE,
    NLayerDiscriminator,
    Pix2Pix,
    Pix2PixCheckpoint,
    UnetGenerator,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# Where trained models are kept, next to the downloaded ONNX weights. Overridden by the same
# MELES_WEIGHTS_DIR the download cache uses, and gitignored for the same reason.
CHECKPOINT_DIR = "pix2pix"


@dataclass(frozen=True)
class Losses:
    """
    What one batch, or one epoch, cost.

    ``generator`` is ``adversarial + lambda_l1 * l1`` and is the only one of the four that is
    backpropagated. It is also the one least worth plotting: it rises whenever the discriminator
    pulls ahead, whether or not the reconstruction improved. **``l1`` is the quality signal.**

    :param generator: The generator's total loss
    :param discriminator: The discriminator's loss
    :param adversarial: The generator's adversarial term alone, i.e. how far ahead the
        discriminator is. Climbing while ``discriminator`` falls towards zero is the signature of
        a discriminator that has won and stopped returning a useful gradient
    :param l1: The mean absolute reconstruction error, unweighted, in the [-1, 1] tensor range
    """

    generator: float
    discriminator: float
    adversarial: float
    l1: float


def default_device() -> torch.device:
    """
    :return: The CUDA device if there is one, otherwise the CPU
    """
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def checkpoint_path(anonymizer: str, directory: str | Path | None = None) -> Path:
    """
    The conventional file for the model trained against one anonymization.

    :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS``
    :param directory: The directory to put it in, defaulting to ``<weights cache>/pix2pix``
    :return: The checkpoint path, which need not exist yet
    """
    root = Path(directory) if directory is not None else weights.cache_dir() / CHECKPOINT_DIR
    return root / f"{anonymizer}.pt"


class Pix2PixTrainer:
    """
    A pix2pix training run: the model, its optimisers, its history and its checkpoint.

    ``anonymizer`` is carried through into every checkpoint the run writes, so that a trained
    model can always say which anonymization it undoes.
    """

    def __init__(
        self,
        anonymizer: str,
        model: Pix2Pix | None = None,
        device: torch.device | str | None = None,
        learning_rate: float = LEARNING_RATE,
        betas: tuple[float, float] = BETAS,
        lambda_l1: float = LAMBDA_L1,
        size: int = SIZE,
        n_epochs: int = N_EPOCHS,
        n_epochs_decay: int = N_EPOCHS_DECAY,
    ):
        """
        The defaults are ``--model pix2pix``'s, i.e. what SEBA gets by not overriding them.

        :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS`` this run attacks
        :param model: The model to train, or None to build a fresh one
        :param device: Where to train, defaulting to CUDA when available
        :param learning_rate: Adam's step size, upstream's ``--lr``
        :param betas: Adam's momentum terms, upstream's ``--beta1`` and Adam's own default. The
            first is 0.5 rather than the usual 0.9 because a GAN's loss surface moves under the
            optimiser -- heavy momentum chases a direction the other network has invalidated
        :param lambda_l1: The weight of the generator's reconstruction term, ``--lambda_L1``
        :param size: The square edge length the pairs are fed at, upstream's ``--crop_size``
        :param n_epochs: How many epochs to hold the learning rate for, ``--n_epochs``
        :param n_epochs_decay: How many further epochs to decay it across, ``--n_epochs_decay``.
            SEBA passes ``epochs // 2`` for this and for ``n_epochs``, so its ``epochs: E`` means
            ``E / 2`` at full rate and ``E / 2`` decaying
        """
        self.anonymizer = anonymizer
        self.device = torch.device(device) if device is not None else default_device()
        self.lambda_l1 = lambda_l1
        self.size = size
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.n_epochs_decay = n_epochs_decay
        self.epoch = 0

        # '.to' reaches both submodules, which is the reason Pix2Pix holds them rather than the
        # caller keeping two loose networks -- a generator left on the CPU while the batches go
        # to the GPU is the classic first crash of a training script.
        self.model = (model if model is not None else Pix2Pix()).to(self.device)

        self.generator_optimizer = optim.Adam(self.model.generator.parameters(), lr=learning_rate, betas=betas)
        self.discriminator_optimizer = optim.Adam(self.model.discriminator.parameters(), lr=learning_rate, betas=betas)

        # 'g' and 'd' are what is optimised; 'adv' and 'l1' are what is worth *reading*. The
        # generator total is the weighted sum of the other two and moves with whichever is
        # winning, so it says nothing on its own about whether the reconstruction improved.
        self.history: dict[str, list[float]] = {
            key: [] for key in ("train_g", "train_d", "train_adv", "train_l1", "val_g", "val_d", "val_adv", "val_l1")
        }

    @property
    def generator(self) -> UnetGenerator:
        return self.model.generator

    @property
    def discriminator(self) -> NLayerDiscriminator:
        return self.model.discriminator

    @property
    def epochs(self) -> int:
        """
        :return: The total length of the run, constant rate plus decay
        """
        return self.n_epochs + self.n_epochs_decay

    def learning_rate_at(self, epoch: int) -> float:
        """
        The learning rate for a given epoch, i.e. upstream's ``lr_policy=linear``.

        Held at ``learning_rate`` for the first ``n_epochs`` and then decayed linearly across the
        next ``n_epochs_decay``. It divides by ``n_epochs_decay + 1`` rather than
        ``n_epochs_decay``, so the rate never quite reaches zero -- that off-by-one is upstream's
        and is reproduced rather than tidied, because it is what a comparison against SEBA's
        numbers has to match.

        :param epoch: The 1-based epoch about to run
        :return: The learning rate for that epoch
        """
        decayed = max(0, epoch - self.n_epochs) / float(self.n_epochs_decay + 1)
        return self.learning_rate * (1.0 - decayed)

    def _apply_learning_rate(self, epoch: int) -> float:
        """
        Set both optimisers' step size for an epoch.

        Upstream steps a ``LambdaLR`` at the end of every epoch; setting it directly from the
        epoch number is the same schedule and lets a resumed run pick it up without having to
        restore any scheduler state.

        :param epoch: The 1-based epoch about to run
        :return: The learning rate applied
        """
        rate = self.learning_rate_at(epoch)
        for optimizer in (self.generator_optimizer, self.discriminator_optimizer):
            for group in optimizer.param_groups:
                group["lr"] = rate
        return rate

    @staticmethod
    def _requires_grad(module: nn.Module, requires: bool) -> None:
        """
        Upstream's ``set_requires_grad``, used to skip the discriminator's gradients while the
        generator steps. Purely a saving -- the generator's optimiser holds no discriminator
        parameters either way -- but it is what upstream does and it is free.

        :param module: The network to switch
        :param requires: Whether its parameters should accumulate gradients
        """
        for parameter in module.parameters():
            parameter.requires_grad = requires

    def step(self, anonymized: Tensor, clear: Tensor) -> Losses:
        """
        One training step on one batch: update the discriminator, then the generator.

        The generator runs **once**, and the discriminator step takes its output ``detach``ed.
        Both halves of that matter. Without the detach, the discriminator's backward pass runs
        all the way through the generator -- wasted work at best, and a gradient in the
        generator's ``.grad`` that only happens to be harmless because the generator step zeroes
        it immediately afterwards. And because the detach cuts the graph rather than freeing it,
        the generator's own graph survives the discriminator's ``backward`` and the second
        forward pass the naive version needs is unnecessary.

        :param anonymized: The batch's anonymized frames, already on the device
        :param clear: The matching clear frames, already on the device
        :return: The batch's losses, with the generator's two terms kept apart
        """
        generated = self.model.generator(anonymized)

        # Discriminator: real frames should score high, generated ones low.
        self._requires_grad(self.model.discriminator, True)
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        real_verdict = self.model.discriminate(anonymized, clear)
        fake_verdict = self.model.discriminate(anonymized, generated.detach())
        discriminator_loss = self.model.discriminator_loss(real_verdict, fake_verdict)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Generator: fool the discriminator, and stay close to the clear frame. The discriminator
        # is run again because it has just been updated, so its verdict on the same frames has
        # changed -- and this pass is what carries a gradient back into the generator.
        self._requires_grad(self.model.discriminator, False)
        self.generator_optimizer.zero_grad(set_to_none=True)
        fake_verdict = self.model.discriminate(anonymized, generated)
        adversarial, l1 = self.model.generator_terms(fake_verdict, generated, clear)
        generator_loss = adversarial + self.lambda_l1 * l1
        generator_loss.backward()
        self.generator_optimizer.step()

        return Losses(generator_loss.item(), discriminator_loss.item(), adversarial.item(), l1.item())

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[Tensor, Tensor]]) -> Losses:
        """
        The mean losses over a loader, with nothing updated.

        Runs in ``eval`` mode, and both halves of that are deliberate. It makes ``val_l1`` the
        deterministic reconstruction error of the configuration the model is actually deployed
        in, rather than a number that depends on which frames happened to share a batch. And it
        stops the BatchNorms from folding the validation images into their running statistics --
        those updates happen inside the forward pass whenever a module is in ``train`` mode, so
        ``no_grad`` does not prevent them, and a run that validated in ``train`` mode would spend
        every epoch quietly training its normalisation on held-out data.

        :param loader: Batches of (anonymized, clear)
        :return: The mean losses, with the generator's two terms kept apart
        """
        was_training = self.model.training
        self.model.eval()
        try:
            totals = [0.0, 0.0, 0.0, 0.0]
            batches = 0
            for anonymized, clear in loader:
                anonymized = anonymized.to(self.device, non_blocking=True)
                clear = clear.to(self.device, non_blocking=True)

                generated = self.model.generator(anonymized)
                real_verdict = self.model.discriminate(anonymized, clear)
                fake_verdict = self.model.discriminate(anonymized, generated)

                adversarial, l1 = self.model.generator_terms(fake_verdict, generated, clear)
                batch = Losses(
                    generator=(adversarial + self.lambda_l1 * l1).item(),
                    discriminator=self.model.discriminator_loss(real_verdict, fake_verdict).item(),
                    adversarial=adversarial.item(),
                    l1=l1.item(),
                )
                totals = [total + value for total, value in zip(totals, astuple(batch))]
                batches += 1
        finally:
            self.model.train(was_training)

        if batches == 0:
            return Losses(*[float("nan")] * 4)
        return Losses(*[total / batches for total in totals])

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int | None = None,
        on_epoch: Callable[["Pix2PixTrainer", int], None] | None = None,
        checkpoint: str | Path | None = None,
        progress: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train, continuing from whatever epoch this trainer is at.

        :param train_loader: Batches of (anonymized, clear), e.g. over an
            ``meles.training.pairs.AnonymizedPairs``
        :param val_loader: The same over held-out images, or None to skip validation. Upstream
            has no validation pass at all -- SEBA's ``train_rate`` creates a ``val`` folder that
            ``train.py`` never reads -- so this is an addition, and it changes nothing that is
            trained
        :param on_epoch: Called after each epoch with ``(trainer, epoch)``, for plotting samples
        :param epochs: How many further epochs to run, or None to run out the configured
            ``n_epochs + n_epochs_decay`` schedule. Stopping short leaves the learning rate
            part-way down its ramp, which is a different model than the schedule describes
        :param checkpoint: A file to write after every epoch, or None not to. A 200-epoch run is
            long enough that not checkpointing is a decision, not a default
        :param progress: Whether to show a per-epoch progress bar
        :return: The full loss history, including any epochs from before this call
        """
        from tqdm import tqdm

        started = time.time()
        self.model.train()
        target = self.epoch + epochs if epochs is not None else self.epochs

        while self.epoch < target:
            totals = [0.0, 0.0, 0.0, 0.0]
            batches = 0

            # Upstream steps its scheduler at the end of an epoch; set from the epoch number, it
            # is the same schedule and survives a resume without any scheduler state.
            rate = self._apply_learning_rate(self.epoch + 1)

            described = f"Training {self.anonymizer} {self.epoch + 1}/{target}"
            # 'mininterval' keeps the bar from writing megabytes of carriage returns into a log,
            # the same reason the workflow scripts set it.
            progressed = tqdm(train_loader, desc=described, mininterval=5.0) if progress else train_loader

            for anonymized, clear in progressed:
                batch = self.step(
                    anonymized.to(self.device, non_blocking=True),
                    clear.to(self.device, non_blocking=True),
                )
                totals = [total + value for total, value in zip(totals, astuple(batch))]
                batches += 1

            self.epoch += 1
            training = Losses(*[total / max(batches, 1) for total in totals])
            validation = self.evaluate(val_loader) if val_loader is not None else Losses(*[float("nan")] * 4)

            for prefix, losses in (("train", training), ("val", validation)):
                self.history[f"{prefix}_g"].append(losses.generator)
                self.history[f"{prefix}_d"].append(losses.discriminator)
                self.history[f"{prefix}_adv"].append(losses.adversarial)
                self.history[f"{prefix}_l1"].append(losses.l1)

            logger.info(
                "Trained epoch",
                extra={
                    "anonymizer": self.anonymizer,
                    "epoch": self.epoch,
                    "lr": round(rate, 8),
                    "train_l1": round(training.l1, 5),
                    "val_l1": round(validation.l1, 5),
                    "train_adv": round(training.adversarial, 4),
                    "train_d": round(training.discriminator, 4),
                },
            )

            if checkpoint is not None:
                self.save(checkpoint)
            if on_epoch is not None:
                on_epoch(self, self.epoch)

        logger.info(
            "Training completed",
            extra={"anonymizer": self.anonymizer, "epochs": epochs, "seconds": round(time.time() - started, 1)},
        )
        return self.history

    def state(self, inference_only: bool = False) -> Pix2PixCheckpoint:
        """
        The current state as a checkpoint.

        :param inference_only: Whether to leave the discriminator and the optimisers out, which
            drops roughly three quarters of the file but makes it unresumable
        :return: The checkpoint
        """
        return Pix2PixCheckpoint(
            anonymizer=self.anonymizer,
            size=self.size,
            epoch=self.epoch,
            # Moved to the CPU so that a checkpoint written on a GPU loads on a machine without
            # one, without the caller having to know to pass 'map_location'.
            generator={key: value.cpu() for key, value in self.model.generator.state_dict().items()},
            discriminator=(
                None
                if inference_only
                else {key: value.cpu() for key, value in self.model.discriminator.state_dict().items()}
            ),
            generator_optimizer=None if inference_only else self.generator_optimizer.state_dict(),
            discriminator_optimizer=None if inference_only else self.discriminator_optimizer.state_dict(),
            history=self.history,
            schedule=(self.n_epochs, self.n_epochs_decay),
        )

    def save(self, path: str | Path | None = None, inference_only: bool = False) -> Path:
        """
        Export the run.

        :param path: Where to write, defaulting to :func:`checkpoint_path` for this anonymization
        :param inference_only: Whether to write the generator alone
        :return: The path written
        """
        return save_checkpoint(self.state(inference_only), path or checkpoint_path(self.anonymizer))

    def load(self, path: str | Path | None = None) -> "Pix2PixTrainer":
        """
        Import a run into this trainer, in place.

        :param path: The checkpoint to read, defaulting to :func:`checkpoint_path`
        :return: This trainer
        :raises ValueError: If the checkpoint was trained against a different anonymization, or
            at a different resolution
        """
        state = load_checkpoint(path or checkpoint_path(self.anonymizer), map_location=self.device)

        if state.anonymizer != self.anonymizer:
            raise ValueError(
                f"Checkpoint was trained against {state.anonymizer!r}, but this trainer attacks {self.anonymizer!r}."
            )
        if state.size != self.size:
            raise ValueError(f"Checkpoint was trained at size {state.size}, but this trainer uses {self.size}.")

        self.model.generator.load_state_dict(state.generator)
        if state.discriminator is not None:
            self.model.discriminator.load_state_dict(state.discriminator)
        if state.generator_optimizer is not None:
            self.generator_optimizer.load_state_dict(state.generator_optimizer)
        if state.discriminator_optimizer is not None:
            self.discriminator_optimizer.load_state_dict(state.discriminator_optimizer)

        self.epoch = state.epoch
        self.n_epochs, self.n_epochs_decay = state.schedule
        # Defaulted rather than replaced, so that a checkpoint missing a key (an inference-only
        # export, say) does not leave 'fit' appending to a dict without that list in it. A key
        # the checkpoint predates is padded with NaN rather than left empty, so that every series
        # stays index-alignable with every other and a plot of a resumed run does not silently
        # shift one curve against the rest.
        recorded = max((len(values) for values in state.history.values()), default=0)
        self.history = {
            key: list(state.history.get(key, [])) + [math.nan] * (recorded - len(state.history.get(key, [])))
            for key in self.history
        }
        self.model.to(self.device)
        return self

    @classmethod
    def resume(
        cls,
        anonymizer: str,
        path: str | Path | None = None,
        device: torch.device | str | None = None,
        **kwargs,
    ) -> "Pix2PixTrainer":
        """
        Build a trainer from a checkpoint if there is one, or a fresh one if there is not.

        This is the call a training notebook wants: re-running the cell after an interruption
        picks the run back up rather than silently starting over at epoch 0.

        :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS``
        :param path: The checkpoint to look for, defaulting to :func:`checkpoint_path`
        :param device: Where to train
        :param kwargs: Further arguments for the constructor
        :return: The trainer, at the checkpoint's epoch if one was found and 0 otherwise
        """
        trainer = cls(anonymizer, device=device, **kwargs)

        path = Path(path) if path is not None else checkpoint_path(anonymizer)
        if path.is_file():
            trainer.load(path)
            logger.info("Resuming pix2pix training", extra={"path": str(path), "epoch": trainer.epoch})
        return trainer
