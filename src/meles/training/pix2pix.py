"""
Training a pix2pix de-anonymizer, one model per anonymization method.

The trainer owns the model, the two optimisers and the loss history together, because those three
are what a checkpoint has to hold for training to be resumable -- and a 50-epoch run over CelebA
will be interrupted.

The ordering inside a step is the part that is easy to get subtly wrong, so it is spelled out in
:meth:`Pix2PixTrainer.step`.
"""

import logging
import time
from collections.abc import Callable, Iterable
from pathlib import Path

import torch
from torch import Tensor, optim
from torch.utils.data import DataLoader

from meles import weights
from meles.frame_aggregation.pix2pix import (
    SIZE,
    Pix2Pix,
    Pix2PixCheckpoint,
    Pix2PixDiscriminator,
    Pix2PixGenerator,
    load_checkpoint,
    save_checkpoint,
)

logger = logging.getLogger(__name__)

# Where trained models are kept, next to the downloaded ONNX weights. Overridden by the same
# MELES_WEIGHTS_DIR the download cache uses, and gitignored for the same reason.
CHECKPOINT_DIR = "pix2pix"


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
        learning_rate: float = 2e-4,
        betas: tuple[float, float] = (0.5, 0.999),
        lambda_l1: float = 100.0,
        size: int = SIZE,
    ):
        """
        :param anonymizer: The key in ``meles.anonymization.ANONYMIZERS`` this run attacks
        :param model: The model to train, or None to build a fresh one
        :param device: Where to train, defaulting to CUDA when available
        :param learning_rate: Adam's step size, 2e-4 in the paper
        :param betas: Adam's momentum terms. The first is 0.5 rather than the usual 0.9 because
            a GAN's loss surface moves under the optimiser -- heavy momentum chases a gradient
            direction that the other network has already invalidated
        :param lambda_l1: The weight of the generator's reconstruction term
        :param size: The square edge length the pairs are fed at, recorded in the checkpoint
        """
        self.anonymizer = anonymizer
        self.device = torch.device(device) if device is not None else default_device()
        self.lambda_l1 = lambda_l1
        self.size = size
        self.epoch = 0

        # '.to' reaches both submodules, which is the reason Pix2Pix holds them rather than the
        # caller keeping two loose networks -- a generator left on the CPU while the batches go
        # to the GPU is the classic first crash of a training script.
        self.model = (model if model is not None else Pix2Pix()).to(self.device)

        self.generator_optimizer = optim.Adam(self.model.generator.parameters(), lr=learning_rate, betas=betas)
        self.discriminator_optimizer = optim.Adam(self.model.discriminator.parameters(), lr=learning_rate, betas=betas)

        self.history: dict[str, list[float]] = {"train_g": [], "train_d": [], "val_g": [], "val_d": []}

    @property
    def generator(self) -> Pix2PixGenerator:
        return self.model.generator

    @property
    def discriminator(self) -> Pix2PixDiscriminator:
        return self.model.discriminator

    def step(self, anonymized: Tensor, clear: Tensor) -> tuple[float, float]:
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
        :return: The generator and discriminator losses for this batch
        """
        generated = self.model.generator(anonymized)

        # Discriminator: real frames should score high, generated ones low.
        self.discriminator_optimizer.zero_grad(set_to_none=True)
        real_verdict = self.model.discriminator(anonymized, clear)
        fake_verdict = self.model.discriminator(anonymized, generated.detach())
        discriminator_loss = self.model.discriminator_loss(real_verdict, fake_verdict)
        discriminator_loss.backward()
        self.discriminator_optimizer.step()

        # Generator: fool the discriminator, and stay close to the clear frame. The discriminator
        # is run again because it has just been updated, so its verdict on the same frames has
        # changed -- and this pass is what carries a gradient back into the generator.
        self.generator_optimizer.zero_grad(set_to_none=True)
        fake_verdict = self.model.discriminator(anonymized, generated)
        generator_loss = self.model.generator_loss(fake_verdict, generated, clear, self.lambda_l1)
        generator_loss.backward()
        self.generator_optimizer.step()

        return generator_loss.item(), discriminator_loss.item()

    @torch.no_grad()
    def evaluate(self, loader: Iterable[tuple[Tensor, Tensor]]) -> tuple[float, float]:
        """
        The mean losses over a loader, with nothing updated.

        The model stays in ``train`` mode on purpose. pix2pix has no noise input, so the
        decoder's dropout is its only stochasticity and the paper keeps it on at test time; and
        switching the BatchNorms to their running statistics here would make the validation loss
        measure a different function than the training loss did, which is not a comparison worth
        having between two numbers printed on the same line.

        :param loader: Batches of (anonymized, clear)
        :return: The mean generator and discriminator loss
        """
        generator_total, discriminator_total, batches = 0.0, 0.0, 0
        for anonymized, clear in loader:
            anonymized = anonymized.to(self.device, non_blocking=True)
            clear = clear.to(self.device, non_blocking=True)

            generated = self.model.generator(anonymized)
            real_verdict = self.model.discriminator(anonymized, clear)
            fake_verdict = self.model.discriminator(anonymized, generated)

            discriminator_total += self.model.discriminator_loss(real_verdict, fake_verdict).item()
            generator_total += self.model.generator_loss(fake_verdict, generated, clear, self.lambda_l1).item()
            batches += 1

        if batches == 0:
            return float("nan"), float("nan")
        return generator_total / batches, discriminator_total / batches

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader | None = None,
        epochs: int = 50,
        on_epoch: Callable[["Pix2PixTrainer", int], None] | None = None,
        checkpoint: str | Path | None = None,
        progress: bool = True,
    ) -> dict[str, list[float]]:
        """
        Train for a number of epochs, continuing from whatever epoch this trainer is at.

        :param train_loader: Batches of (anonymized, clear), e.g. over an
            ``meles.training.pairs.AnonymizedPairs``
        :param val_loader: The same over held-out images, or None to skip validation
        :param epochs: How many further epochs to run
        :param on_epoch: Called after each epoch with ``(trainer, epoch)``, for plotting samples
        :param checkpoint: A file to write after every epoch, or None not to. A 50-epoch run over
            CelebA is long enough that not checkpointing is a decision, not a default
        :param progress: Whether to show a per-epoch progress bar
        :return: The full loss history, including any epochs from before this call
        """
        from tqdm import tqdm

        started = time.time()
        self.model.train()
        target = self.epoch + epochs

        while self.epoch < target:
            generator_total, discriminator_total, batches = 0.0, 0.0, 0

            described = f"Training {self.anonymizer} {self.epoch + 1}/{target}"
            # 'mininterval' keeps the bar from writing megabytes of carriage returns into a log,
            # the same reason the workflow scripts set it.
            progressed = tqdm(train_loader, desc=described, mininterval=5.0) if progress else train_loader

            for anonymized, clear in progressed:
                generator_loss, discriminator_loss = self.step(
                    anonymized.to(self.device, non_blocking=True),
                    clear.to(self.device, non_blocking=True),
                )
                generator_total += generator_loss
                discriminator_total += discriminator_loss
                batches += 1

            self.epoch += 1
            self.history["train_g"].append(generator_total / max(batches, 1))
            self.history["train_d"].append(discriminator_total / max(batches, 1))

            validation = self.evaluate(val_loader) if val_loader is not None else (float("nan"), float("nan"))
            self.history["val_g"].append(validation[0])
            self.history["val_d"].append(validation[1])

            logger.info(
                "Trained epoch",
                extra={
                    "anonymizer": self.anonymizer,
                    "epoch": self.epoch,
                    "train_g": self.history["train_g"][-1],
                    "train_d": self.history["train_d"][-1],
                    "val_g": self.history["val_g"][-1],
                    "val_d": self.history["val_d"][-1],
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
        # Defaulted rather than replaced, so that a checkpoint missing a key (an inference-only
        # export, say) does not leave 'fit' appending to a dict without that list in it.
        self.history = {key: list(state.history.get(key, [])) for key in self.history}
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
