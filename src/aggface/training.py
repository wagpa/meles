"""
Training a model over the anonymization curriculum, and running a trained one.

Two schedules run at once and they are independent, which is the easiest thing to confuse:

- the **curriculum** widens which anonymizations the data holds, and :meth:`Trainer.fit` walks it;
- the **encoder schedule** freezes the pretrained encoder and later fine-tunes it.

Call ``fit`` once per encoder phase, not once per curriculum stage.
"""

import logging
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from aggface.data import CURRICULUM, Items
from aggface.losses import Criterion, Losses, WeightedCriterion, l1_per_item
from aggface.model import AggFace
from aggface.tensors import default_device

logger = logging.getLogger(__name__)

# The two learning rates: the decoder and the aggregator are trained from scratch, while the
# encoder arrives pretrained and is only nudged, an order of magnitude slower.
LEARNING_RATE = 1e-4
ENCODER_LEARNING_RATE = 1e-5
WEIGHT_DECAY = 5e-4

# Bumped when the checkpoint layout changes, so an older file fails loudly rather than loading
# into mismatched state dicts.
CHECKPOINT_FORMAT = 2


class Trainer:
    """A training run: the model, its optimiser, its history and its checkpoint."""

    def __init__(
        self,
        model: AggFace | None = None,
        methods: Sequence[str] = CURRICULUM,
        seed: int = 0,
        device: torch.device | str | None = None,
        learning_rate: float = LEARNING_RATE,
        encoder_learning_rate: float = ENCODER_LEARNING_RATE,
        weight_decay: float = WEIGHT_DECAY,
        criterion: Criterion = l1_per_item,
        encoder_frozen: bool = True,
    ) -> None:
        """
        :param model: The model to train, or None to build one conditioned on ``methods``
        :param methods: The anonymization names in curriculum order, which label the per-method
            losses and size the model's conditioning
        :param seed: Seeds the model's initialisation and the batch order. Recorded in the
            checkpoint, because the whole point of running several is to compare them
        :param device: Where to train, defaulting to CUDA when there is one
        :param learning_rate: The step size for the decoder and the aggregator
        :param encoder_learning_rate: The step size for the pretrained encoder, used once it is
            unfrozen. An order of magnitude lower, because it arrives already trained
        :param weight_decay: AdamW's decoupled weight decay
        :param criterion: Maps ``(generated, target)`` to one loss per item, or to named weighted
            per-item contributions
        :param encoder_frozen: Whether to start with the encoder frozen
        """
        self.methods = list(methods)
        self.seed = seed
        torch.manual_seed(seed)

        self.model = model if model is not None else AggFace(methods=len(self.methods))
        self.device = torch.device(device) if device is not None else default_device()
        self.model.to(self.device)

        self.learning_rate = learning_rate
        self.encoder_learning_rate = encoder_learning_rate
        self.weight_decay = weight_decay
        self.criterion = criterion
        # The identity term carries a network of its own, which has to follow the model.
        if isinstance(criterion, WeightedCriterion | torch.nn.Module):
            criterion.to(self.device)

        self.epoch = 0
        self.stage = 0
        keys = ["stage", "train", "val"]
        self.history: dict[str, list[float]] = {
            key: [] for key in keys + [f"{p}_{m}" for p in ("train", "val") for m in self.methods]
        }

        self.optimizer: torch.optim.Optimizer
        self.freeze_encoder(encoder_frozen)

    def record(self, key: str, value: float) -> None:
        """
        Append to a history series, creating it if this is the first epoch that produced it.

        A new series is back-filled with NaN so every series has one entry per epoch and plotting
        them against a shared epoch axis lines up.

        :param key: The series name
        :param value: The value for this epoch
        """
        series = self.history.setdefault(key, [float("nan")] * (self.epoch - 1))
        series.extend([float("nan")] * (self.epoch - 1 - len(series)))
        series.append(value)

    def freeze_encoder(self, frozen: bool) -> "Trainer":
        """
        Switch the encoder between frozen and fine-tuned, rebuilding the optimiser for it.

        The optimiser has to be rebuilt because the unfrozen encoder is a new parameter group at
        its own learning rate. That discards AdamW's moment estimates, which is intended at a
        phase change: the moments describe a loss surface that just moved.

        :param frozen: Whether the encoder is frozen
        :return: This trainer
        """
        self.model.freeze_encoder(frozen)

        groups: list[dict] = [{"params": self.model.new_parameters(), "lr": self.learning_rate}]
        fine = self.model.fine_parameters()
        if fine:
            groups.append({"params": fine, "lr": self.encoder_learning_rate})

        self.optimizer = torch.optim.AdamW(groups, weight_decay=self.weight_decay)
        logger.info("Encoder schedule set", extra={"frozen": frozen, "groups": len(groups), "fine_tuned": len(fine)})
        return self

    def apply(self, generated: Tensor, clear: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        Run the criterion, accepting either contract.

        :param generated: What the model produced
        :param clear: The clear target
        :return: The per-item total and the per-item terms
        """
        result = self.criterion(generated, clear)
        terms = {"loss": result} if isinstance(result, Tensor) else result
        return torch.stack(list(terms.values())).sum(dim=0), terms

    def step(self, anonymized: Tensor, clear: Tensor, method: Tensor) -> tuple[Tensor, dict[str, Tensor]]:
        """
        One optimiser step over a batch.

        :param anonymized: (B, N, 3, H, W), the observations
        :param clear: (B, 3, H, W), the targets
        :param method: (B,), which anonymization each item carries
        :return: The detached per-item totals and per-item terms
        """
        self.optimizer.zero_grad(set_to_none=True)
        total, terms = self.apply(self.model(anonymized, method), clear)
        total.mean().backward()
        self.optimizer.step()
        return total.detach(), {name: value.detach() for name, value in terms.items()}

    def totals(self) -> tuple[Tensor, Tensor]:
        """
        :return: Empty per-method sums and counts
        """
        return torch.zeros(len(self.methods)), torch.zeros(len(self.methods))

    @staticmethod
    def accumulate(sums: Tensor, counts: Tensor, per_item: Tensor, method: Tensor) -> None:
        """
        Add a batch's per-item losses into the per-method running totals.

        :param sums: The running sums, modified in place
        :param counts: The running counts, modified in place
        :param per_item: One loss per item
        :param method: Which anonymization each item carried
        """
        index = method.detach().to("cpu", torch.long)
        sums.index_add_(0, index, per_item.detach().to("cpu", torch.float32))
        counts.index_add_(0, index, torch.ones_like(index, dtype=torch.float32))

    def losses(self, sums: Tensor, counts: Tensor, terms: dict[str, Tensor] | None = None) -> Losses:
        """
        :param sums: Per-method loss sums
        :param counts: Per-method item counts
        :param terms: Per-method sums for each loss term
        :return: The epoch's losses
        """
        total = float(counts.sum())
        terms = terms or {}

        def by_method(values: Tensor) -> dict[str, float]:
            return {
                name: float(values[i] / counts[i]) if counts[i] else float("nan") for i, name in enumerate(self.methods)
            }

        return Losses(
            overall=float(sums.sum() / total) if total else float("nan"),
            per_method=by_method(sums),
            counts={name: int(counts[i]) for i, name in enumerate(self.methods)},
            per_term={name: float(value.sum() / total) if total else float("nan") for name, value in terms.items()},
            per_term_method={name: by_method(value) for name, value in terms.items()},
        )

    def run(
        self,
        loader: Iterable[tuple[Tensor, Tensor, Tensor]],
        training: bool,
        description: str = "",
        progress: bool = True,
    ) -> Losses:
        """
        One pass over a loader, either stepping the optimiser or only measuring.

        Evaluation runs in eval mode, and both halves of that matter: the encoder carries
        BatchNorms whose running statistics are updated inside the forward pass whenever the
        module is in train mode - ``no_grad`` does not stop that - so validating in train mode
        would spend every epoch quietly fitting normalisation to held-out identities.

        :param loader: Batches of ``(anonymized, clear, method)``
        :param training: Whether to update the model
        :param description: What to label the progress bar with
        :param progress: Whether to show a progress bar
        :return: The losses, overall, per method and per term
        """
        was_training = self.model.training
        self.model.train(training)
        sums, counts = self.totals()
        term_sums: dict[str, Tensor] = {}
        # 'mininterval' keeps the bar from writing megabytes of carriage returns into a log.
        stepped = tqdm(loader, desc=description, mininterval=5.0) if progress else loader

        try:
            for batch in stepped:
                anonymized, clear, method = (value.to(self.device, non_blocking=True) for value in batch)
                if training:
                    total, terms = self.step(anonymized, clear, method)
                else:
                    with torch.no_grad():
                        total, terms = self.apply(self.model(anonymized, method), clear)
                self.accumulate(sums, counts, total, method)
                ignored = torch.zeros(len(self.methods))
                for name, value in terms.items():
                    self.accumulate(term_sums.setdefault(name, torch.zeros(len(self.methods))), ignored, value, method)
        finally:
            self.model.train(was_training)
        return self.losses(sums, counts, term_sums)

    def fit(
        self,
        train_loader_for: Callable[[int], DataLoader],
        validation_loader: Iterable[tuple[Tensor, Tensor, Tensor]] | None = None,
        epochs_per_stage: int = 25,
        stages: Sequence[int] | None = None,
        checkpoint: str | Path | None = None,
        best: str | None = None,
        maximize: bool = True,
        metrics: Callable[["Trainer"], dict[str, float]] | None = None,
        on_epoch: Callable[["Trainer", int], None] | None = None,
        progress: bool = True,
    ) -> dict[str, list[float]]:
        """
        Walk the curriculum, training ``epochs_per_stage`` epochs at each stage.

        :param train_loader_for: Builds the training loader for a curriculum stage. A factory
            rather than a loader, because a ``DataLoader`` hands its workers a copy of the dataset
            when its iterator is first made: widening the dataset afterwards would reach the
            parent and nothing else, and training would silently stay on the old stage
        :param validation_loader: Held-out batches, or None to skip validation. Pin it to the last
            stage so every method is measured from the first epoch
        :param epochs_per_stage: Epochs to train at each stage
        :param stages: Which curriculum stages to run, defaulting to all of them in order
        :param checkpoint: A file to write after every epoch, or None not to
        :param best: A history key to also keep the best epoch of, written alongside ``checkpoint``
            with ``.best`` before the suffix. Worth setting whenever the metric is noisier than
            the improvement, so that the last epoch is not an arbitrary draw from the trajectory
        :param maximize: Whether a larger ``best`` is better. True for rank-1, False for a loss
        :param metrics: Called with the trainer after each epoch; whatever it returns is recorded
            under ``val:<key>``. Use it for numbers the loss cannot express
        :param on_epoch: Called with ``(trainer, epoch)`` after each epoch, for plotting samples
        :param progress: Whether to show a per-epoch progress bar
        :return: The full history, including epochs from before this call
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
                training = self.run(loader, True, described, progress)
                validation = (
                    self.run(validation_loader, False, progress=False)
                    if validation_loader is not None
                    else self.losses(*self.totals())
                )
                self.epoch += 1

                self.history["stage"].append(float(stage))
                for prefix, measured_losses in (("train", training), ("val", validation)):
                    self.history[prefix].append(measured_losses.overall)
                    for name, value in measured_losses.per_method.items():
                        self.history[f"{prefix}_{name}"].append(value)
                    # Term keys are only known once a criterion has run, so they are created on
                    # first sight and back-filled with NaN.
                    for term, value in measured_losses.per_term.items():
                        self.record(f"{prefix}:{term}", value)
                        for name, split in measured_losses.per_term_method[term].items():
                            self.record(f"{prefix}:{term}_{name}", split)

                measured = metrics(self) if metrics is not None else {}
                for key, value in measured.items():
                    self.record(f"val:{key}", value)

                logger.info(
                    "Trained epoch",
                    extra={
                        "epoch": self.epoch,
                        "stage": stage,
                        "seed": self.seed,
                        "train": round(training.overall, 5),
                        "val": round(validation.overall, 5),
                        **{f"val_{name}": round(value, 5) for name, value in validation.per_method.items()},
                        **{f"val:{key}": round(value, 5) for key, value in measured.items()},
                    },
                )

                if checkpoint is not None:
                    self.save(checkpoint)
                    if best is not None:
                        latest = self.history.get(best, [float("nan")])[-1]
                        # NaN compares False either way, so an epoch that did not produce the
                        # metric simply never becomes the record.
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
            estimates per parameter, so dropping them is most of the file - but what is left
            cannot be resumed, only run
        :return: Everything needed to resume this run, or to run it
        """
        state = {
            "format": CHECKPOINT_FORMAT,
            "model": self.model.state_dict(),
            # Recorded because they decide the parameter shapes: a checkpoint cannot be read back
            # without them.
            "width": self.model.decoder.width,
            "dropout": self.model.decoder.dropout,
            "methods": self.methods,
            "seed": self.seed,
            "epoch": self.epoch,
            "stage": self.stage,
            "encoder_frozen": self.model.encoder_frozen,
            "history": self.history,
        }
        if not inference_only:
            state["optimizer"] = self.optimizer.state_dict()
        return state

    def save(self, path: str | Path, inference_only: bool = False) -> Path:
        """
        Write a checkpoint, through a temporary file so that an interrupted run leaves neither a
        truncated new checkpoint nor a missing old one.

        :param path: The file to write; its directory is created if needed
        :param inference_only: Whether to drop the optimiser state
        :return: The path written
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state(inference_only), temporary)
        temporary.replace(path)
        return path

    def load(self, path: str | Path) -> "Trainer":
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
        self.freeze_encoder(bool(payload["encoder_frozen"]))
        self.model.load_state_dict(payload["model"])
        if "optimizer" in payload:
            self.optimizer.load_state_dict(payload["optimizer"])
        else:
            logger.warning("Checkpoint holds no optimiser state, so training restarts from fresh moments")
        self.seed = int(payload["seed"])
        self.epoch = int(payload["epoch"])
        self.stage = int(payload["stage"])
        self.history = {key: list(values) for key, values in payload["history"].items()}
        return self


def load_model(path: str | Path, device: torch.device | str | None = None) -> tuple[AggFace, dict]:
    """
    Load a trained model for inference, without building a trainer around it.

    :param path: A checkpoint written by :meth:`Trainer.save`, with or without optimiser state
    :param device: Where to put the model, defaulting to CUDA when there is one
    :return: The model in eval mode, and the checkpoint's metadata without the tensors
    :raises ValueError: If the file was written by a different checkpoint format
    """
    where = torch.device(device) if device is not None else default_device()
    payload = torch.load(Path(path), map_location=where, weights_only=False)

    if payload.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"Checkpoint format {payload.get('format')!r}, expected {CHECKPOINT_FORMAT}")

    methods = list(payload["methods"])
    # Built without the recognition weights: the checkpoint carries its own encoder, so reading
    # the pretrained file would only be overwritten a line later.
    model = AggFace(methods=len(methods), pretrained=False, width=payload["width"], dropout=payload.get("dropout", 0.0))
    model.load_state_dict(payload["model"])
    model.to(where).eval()

    metadata = {key: payload[key] for key in ("epoch", "stage", "methods", "seed", "history") if key in payload}
    return model, metadata


@torch.no_grad()
def reconstruct(
    model: AggFace,
    anonymized: Tensor,
    method: Tensor | int | None = None,
    device: torch.device | str | None = None,
) -> Tensor:
    """
    Run observations through a trained model and get the faces back.

    Always in eval mode: the encoder's BatchNorms otherwise normalise by the batch in front of
    them, so the same identity would reconstruct differently depending on which other identities
    happened to share its batch.

    :param model: A trained model
    :param anonymized: (B, N, 3, H, W) observations, or (N, 3, H, W) for a single item
    :param method: Which anonymization the observations carry - a (B,) tensor, one int for the
        whole batch, or None for a model trained without conditioning
    :param device: Where to run, defaulting to wherever the model already is
    :return: The reconstructions, (B, 3, H, W) in [-1, 1], on the CPU
    """
    where = torch.device(device) if device is not None else next(model.parameters()).device
    batched = (anonymized if anonymized.dim() == 5 else anonymized.unsqueeze(0)).to(where)

    conditioning = method
    if isinstance(conditioning, int):
        conditioning = torch.full((batched.shape[0],), conditioning, dtype=torch.long)
    if conditioning is not None:
        conditioning = conditioning.to(where)
        if conditioning.dim() == 0:
            conditioning = conditioning.expand(batched.shape[0])

    was_training = model.training
    model.eval()
    try:
        return model(batched, conditioning).cpu()
    finally:
        model.train(was_training)


@torch.no_grad()
def identification(
    model: AggFace,
    items: Items,
    embedder: Callable[[Tensor], np.ndarray],
    ranks: Sequence[int] = (1, 5),
    chunk: int = 16,
) -> dict[str, float]:
    """
    Rank-k identification accuracy: how often a reconstruction's nearest gallery face is the right
    person.

    **This is the number the experiment scores, and the one to steer on.** Cosine distance
    compresses almost all of its interesting range into the last tenth, so a large change in
    whether the right identity ranks first shows up there as a rounding error. Identification only
    asks that the correct face be nearest, not that it be absolutely close.

    The gallery is the clear target of every identity in ``items``; the probes are the
    reconstructions, one per identity per anonymization.

    :param model: A trained model
    :param items: The validation set, built with ``all_methods=True`` and ``deterministic=True``
        so the layout is identity-major and stable between epochs
    :param embedder: Maps (B, 3, H, W) in [-1, 1] to L2-normalised embeddings. Use a recognizer
        that is **not** the training critic, or this grades its own work
    :param ranks: Which rank-k accuracies to report
    :param chunk: How many identities to reconstruct at once
    :return: ``rank1``, ``rank1_<method>``, ... as fractions in [0, 1]
    :raises ValueError: If ``items`` was not built with ``all_methods``
    """
    if not items.all_methods:
        raise ValueError("identification needs a dataset built with all_methods=True")

    width = items.stage + 1
    people = len(items) // width
    gallery = embedder(torch.stack([items[i * width][1] for i in range(people)]))
    answer = np.arange(people)

    scores: dict[str, float] = {}
    hits: dict[int, list[float]] = {rank: [] for rank in ranks}
    for index in range(width):
        produced = []
        for begin in range(0, people, chunk):
            block = torch.stack([items[p * width + index][0] for p in range(begin, min(begin + chunk, people))])
            produced.append(reconstruct(model, block, index))
        similarity = embedder(torch.cat(produced)) @ gallery.T

        order = np.argsort(-similarity, axis=1)
        position = np.argmax(order == answer[:, None], axis=1)
        for rank in ranks:
            value = float((position < rank).mean())
            scores[f"rank{rank}_{items.names[index]}"] = value
            hits[rank].append(value)

    for rank in ranks:
        scores[f"rank{rank}"] = float(np.mean(hits[rank]))
    return scores
