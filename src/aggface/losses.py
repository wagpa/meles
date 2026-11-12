"""
What the reconstruction is scored against while training.

Two terms, pulling in different directions, which is why both are reported separately:

- **L1** asks for the right pixels. Its cheapest answer under uncertainty is the conditional
  median - a blurry average face, which has low error against everyone and identifies as no one.
  Measured on an L1-only run, the per-method validation loss came out *inverted*, with
  ``blackbar`` scoring better than ``none``, because the less the model could see the harder it
  hedged towards that mean face.
- **Identity** asks for the right person, as cosine distance between a recognizer's embeddings of
  the reconstruction and of the target. A recognizer is not fooled by the average face, so this
  term removes that incentive.

Every loss here returns **one value per item** rather than a reduced scalar. That is what lets the
trainer split a batch's loss by anonymization afterwards; a scalar has already averaged over items
that came from different methods.
"""

from collections.abc import Callable
from dataclasses import dataclass, field

import torch
import torch.nn.functional as functional
from torch import Tensor, nn

from aggface.model import Encoder

# A criterion maps (generated, target) to either one per-item loss, or named per-item weighted
# contributions as 'WeightedCriterion' does. The trainer accepts both.
Criterion = Callable[[Tensor, Tensor], "Tensor | dict[str, Tensor]"]


def l1_per_item(generated: Tensor, target: Tensor) -> Tensor:
    """
    Mean absolute reconstruction error, per item rather than reduced over the batch.

    :param generated: What the model produced, (B, 3, H, W) in [-1, 1]
    :param target: The clear target, the same shape
    :return: One loss per item, (B,)
    """
    return (generated - target).abs().flatten(1).mean(dim=1)


class IdentityLoss(nn.Module):
    """
    Cosine distance between a recognizer's embeddings of the reconstruction and of the target:
    ``1 - cos``, per item, in [0, 2].

    The critic is a **separate frozen copy** of the recognizer, never the encoder inside the
    model. Two reasons, and the second is the dangerous one: the encoder is fine-tuned in the
    second phase, so a shared critic could be satisfied by moving the encoder rather than by
    improving the image; and a critic whose weights drift means something different each epoch, so
    its numbers would not be comparable across a run.

    It is an ``iresnet34`` because that is the only **differentiable** recognizer here - ArcFace,
    SFace and GhostFaceNet all run through ONNX Runtime, which carries no gradients. That is also
    the better arrangement, since ArcFace is what scores the experiment and optimising against it
    directly would be training on the evaluation metric.
    """

    def __init__(self, encoder: Encoder | None = None) -> None:
        """
        :param encoder: An already-built encoder to take the critic from, to avoid reading the
            same weights twice. One is built when omitted
        """
        super().__init__()
        self.encoder = encoder if encoder is not None else Encoder()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        self.encoder.eval()

    def train(self, mode: bool = True) -> "IdentityLoss":
        """
        Pins the critic to eval. Its BatchNorms must never move.

        :param mode: Whether the surrounding model is entering training mode
        :return: This loss
        """
        super().train(mode)
        self.encoder.eval()
        return self

    def embed(self, images: Tensor) -> Tensor:
        """
        :param images: (B, 3, 112, 112) in [-1, 1] RGB, which is what the recognizer was trained
            on, so no renormalisation is needed
        :return: The L2-normalised (B, 512) embeddings
        """
        return functional.normalize(self.encoder.model(images), dim=1)

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        """
        :param generated: What the model produced, (B, 3, 112, 112) in [-1, 1]
        :param target: The clear target, the same shape
        :return: ``1 - cos`` per item, (B,)
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
    ones, so a reconstruction can grow more similar to its target while growing just as similar to
    everyone - which moves the cosine and leaves identification where it was. Rank-1 is a ranking
    question, so this optimises the ranking directly.

    Items sharing a target are masked out of each other's negatives: a validation split yields
    every identity once per anonymization, and without the mask an identity would be punished for
    resembling itself under a different method.
    """

    def __init__(self, encoder: Encoder | None = None, temperature: float = 0.05, duplicate: float = 0.999) -> None:
        """
        :param encoder: An already-built encoder to take the critic from
        :param temperature: Softmax temperature over the cosine similarities. Low because cosines
            live in [-1, 1], so the logits need scaling before a softmax separates them at all
        :param duplicate: Cosine above which two targets are taken to be the same identity, and so
            masked out of each other's negatives
        """
        super().__init__(encoder=encoder)
        self.temperature = temperature
        self.duplicate = duplicate

    def forward(self, generated: Tensor, target: Tensor) -> Tensor:
        """
        :param generated: What the model produced, (B, 3, 112, 112) in [-1, 1]
        :param target: The clear target, the same shape
        :return: The per-item cross entropy, (B,). A batch of one has no negatives, so it is zero
        """
        with torch.no_grad():
            gallery = self.embed(target)
        probe = self.embed(generated)

        logits = probe @ gallery.T / self.temperature
        own = torch.arange(len(probe), device=logits.device)

        # Mask other items carrying the same target, but never the diagonal, which is the answer.
        shared = (gallery @ gallery.T) > self.duplicate
        shared[own, own] = False
        logits = logits.masked_fill(shared, float("-inf"))

        if len(probe) < 2:
            return torch.zeros(len(probe), device=logits.device)
        return functional.cross_entropy(logits, own, reduction="none")


class WeightedCriterion:
    """
    Several per-item losses, summed with weights.

    Returns the **weighted** contribution of each term rather than one number, so the trainer can
    both backpropagate their sum and report them apart. Reporting them apart is the point: a
    falling total says nothing about whether the faces got sharper or more recognisable, and those
    are the two things being traded against each other.
    """

    def __init__(self, terms: dict[str, tuple[float, Callable[[Tensor, Tensor], Tensor]]]) -> None:
        """
        :param terms: Name to ``(weight, loss)``, where each loss returns one value per item
        :raises ValueError: If no terms are given
        """
        if not terms:
            raise ValueError("A criterion needs at least one term")
        self.terms = dict(terms)

    def to(self, device: torch.device | str) -> "WeightedCriterion":
        """
        Move any term that is a module; the identity loss carries a network.

        :param device: Where to move them
        :return: This criterion
        """
        for _, loss in self.terms.values():
            if isinstance(loss, nn.Module):
                loss.to(device)
        return self

    def __call__(self, generated: Tensor, target: Tensor) -> dict[str, Tensor]:
        """
        :param generated: What the model produced, (B, 3, H, W)
        :param target: The clear target, the same shape
        :return: Each term's weighted per-item contribution, (B,) apiece
        """
        return {name: weight * loss(generated, target) for name, (weight, loss) in self.terms.items()}


@dataclass(frozen=True)
class Losses:
    """
    What one epoch cost: the mean error overall, split by anonymization, and split by loss term.

    The per-method split is what a curriculum is steered by. A single mean hides whether unlocking
    ``blackbar`` bought its gain by giving up ``gaussianblur``, because the newly unlocked method
    is also the one dragging the mean up - so a rising total is equally consistent with healthy
    progress and with quiet forgetting.

    :param overall: The item-weighted mean over every method seen
    :param per_method: The mean per anonymization, NaN for one that did not appear
    :param counts: How many items each anonymization contributed
    :param per_term: Each term's weighted mean, over everything
    :param per_term_method: Each term's weighted mean, split by anonymization
    """

    overall: float
    per_method: dict[str, float]
    counts: dict[str, int]
    per_term: dict[str, float] = field(default_factory=dict)
    per_term_method: dict[str, dict[str, float]] = field(default_factory=dict)

    def describe(self, places: int = 4) -> str:
        """
        :param places: Decimal places to show
        :return: A one-line summary, for a log line or a progress bar
        """
        seen = " ".join(f"{name}={value:.{places}f}" for name, value in self.per_method.items() if value == value)
        terms = " ".join(f"{name}={value:.{places}f}" for name, value in self.per_term.items())
        described = f"{self.overall:.{places}f}"
        if terms:
            described += f" [{terms}]"
        return f"{described} ({seen})" if seen else described
