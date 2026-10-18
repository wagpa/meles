"""
``FacialTrainer``'s per-method accounting is the part that fails silently: a loss attributed to
the wrong anonymization, or averaged before it can be split, still produces a plausible number.
These pin the arithmetic against hand-computed values, then check the loop around it.

A tiny stand-in model keeps the tests off the 112x112 iresnet34, which would make them minutes long.
"""

import math
from pathlib import Path

import pytest
import torch
from torch import nn

from meles.training.facial import CURRICULUM, FacialTrainer, Losses, l1_per_item

SIZE, CHANNELS = 8, 3


def same(left, right) -> bool:
    """Equality that treats NaN as equal to NaN, which plain '==' does not."""
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(same(left[k], right[k]) for k in left)
    if isinstance(left, list):
        return len(left) == len(right) and all(same(a, b) for a, b in zip(left, right))
    if isinstance(left, Losses):
        return all(same(getattr(left, f), getattr(right, f)) for f in ("overall", "per_method", "counts"))
    if isinstance(left, float) and math.isnan(left):
        return isinstance(right, float) and math.isnan(right)
    return bool(left == right)


class TinyFacial(nn.Module):
    """Averages the N input frames and maps them through one conv. Same signature as Facial."""

    def __init__(self, methods=len(CURRICULUM)):
        super().__init__()
        self.encoder_frozen = False
        self.methods = methods
        self.encoder = nn.Conv2d(CHANNELS, CHANNELS, 1)
        self.aggregator = nn.Conv2d(CHANNELS, CHANNELS, 1)
        self.decoder = nn.Conv2d(CHANNELS, CHANNELS, 1)
        self.method = nn.Embedding(methods, CHANNELS) if methods else None

    def forward(self, x, method=None):
        pooled = self.encoder(x.flatten(0, 1)).view(*x.shape).mean(dim=1)
        out = self.decoder(self.aggregator(pooled))
        if self.method is not None and method is not None:
            out = out + self.method(method)[:, :, None, None]
        return torch.tanh(out)

    def train(self, mode=True):
        super().train(mode)
        if self.encoder_frozen:
            self.encoder.eval()
        return self

    def set_encoder_frozen(self, frozen):
        self.encoder_frozen = frozen
        for p in self.encoder.parameters():
            p.requires_grad_(not frozen)
        return self

    def fine_parameters(self):
        return [p for p in self.encoder.parameters() if p.requires_grad]

    def train_parameters(self):
        encoder = {id(p) for p in self.encoder.parameters()}
        return [p for p in self.parameters() if id(p) not in encoder]


def batch(methods, frames=2, seed=0):
    """A batch whose method column is exactly `methods`."""
    generator = torch.Generator().manual_seed(seed)
    size = (len(methods), frames, CHANNELS, SIZE, SIZE)
    return (
        torch.rand(size, generator=generator) * 2 - 1,
        torch.rand((len(methods), CHANNELS, SIZE, SIZE), generator=generator) * 2 - 1,
        torch.tensor(methods, dtype=torch.long),
    )


def trainer(**kwargs):
    return FacialTrainer(model=TinyFacial(), device="cpu", **kwargs)


def test_l1_per_item_does_not_reduce_across_the_batch():
    """The whole per-method split depends on this staying per item."""
    generated = torch.zeros(4, CHANNELS, SIZE, SIZE)
    target = torch.stack([torch.full((CHANNELS, SIZE, SIZE), v) for v in (0.0, 0.5, 1.0, 0.25)])
    losses = l1_per_item(generated, target)
    assert losses.shape == (4,)
    assert torch.allclose(losses, torch.tensor([0.0, 0.5, 1.0, 0.25]))


def test_per_method_means_are_computed_per_method():
    """Hand-checked: method 0 sees losses 0.0 and 1.0, method 2 sees 0.5."""
    run = trainer()
    sums, counts = run._totals()
    run._accumulate(sums, counts, torch.tensor([0.0, 1.0, 0.5]), torch.tensor([0, 0, 2]))
    losses = run._losses(sums, counts)

    assert losses.per_method["none"] == pytest.approx(0.5)
    assert losses.per_method["pixelation"] == pytest.approx(0.5)
    assert losses.counts == {"none": 2, "gaussianblur": 0, "pixelation": 1, "blackbar": 0}
    # Item-weighted, not a mean of the per-method means: (0 + 1 + 0.5) / 3.
    assert losses.overall == pytest.approx(1.5 / 3)


def test_a_method_that_never_appeared_is_nan_not_zero():
    """Zero would read as a perfect reconstruction of something never attempted."""
    run = trainer()
    sums, counts = run._totals()
    run._accumulate(sums, counts, torch.tensor([0.4]), torch.tensor([1]))
    losses = run._losses(sums, counts)

    assert losses.per_method["gaussianblur"] == pytest.approx(0.4)
    for absent in ("none", "pixelation", "blackbar"):
        assert losses.per_method[absent] != losses.per_method[absent], f"{absent} should be NaN"


def test_accumulation_is_order_independent():
    run = trainer()
    forwards, counts = run._totals()
    run._accumulate(forwards, counts, torch.tensor([0.2, 0.8]), torch.tensor([3, 1]))
    backwards, other = run._totals()
    run._accumulate(backwards, other, torch.tensor([0.8, 0.2]), torch.tensor([1, 3]))
    assert same(run._losses(forwards, counts), run._losses(backwards, other))


def test_evaluate_splits_a_loader_by_method():
    run = trainer()
    loader = [batch([0, 1]), batch([2, 3])]
    losses = run.evaluate(loader)
    assert losses.counts == dict.fromkeys(CURRICULUM, 1)
    assert all(value == value for value in losses.per_method.values())


def test_evaluate_leaves_the_model_untrained_and_restores_its_mode():
    run = trainer()
    before = [p.clone() for p in run.model.parameters()]
    run.model.train()
    run.evaluate([batch([0, 1])])
    assert run.model.training, "evaluate must restore the mode it found"
    assert all(torch.equal(a, b) for a, b in zip(before, run.model.parameters()))


def test_evaluate_runs_in_eval_mode():
    """BatchNorm folds held-out data into its running statistics unless eval mode stops it."""
    seen = []
    run = trainer()
    original = run.model.forward
    run.model.forward = lambda *a, **k: (seen.append(run.model.training), original(*a, **k))[1]
    run.evaluate([batch([0])])
    assert seen == [False]


def test_step_reduces_the_loss_on_a_repeated_batch():
    run = trainer()
    anonymized, clear, method = batch([0, 1, 2, 3])
    first = run.step(anonymized, clear, method)[0].mean()
    for _ in range(25):
        run.step(anonymized, clear, method)
    assert run.step(anonymized, clear, method)[0].mean() < first


def test_step_returns_one_loss_per_item():
    run = trainer()
    total, terms = run.step(*batch([0, 2, 3]))
    assert total.shape == (3,)
    assert all(value.shape == (3,) for value in terms.values())


def test_freezing_rebuilds_the_optimiser_groups():
    run = trainer(encoder_frozen=True)
    assert len(run.optimizer.param_groups) == 1

    run.set_encoder_frozen(False)
    assert len(run.optimizer.param_groups) == 2
    assert run.optimizer.param_groups[1]["lr"] < run.optimizer.param_groups[0]["lr"]


def test_fit_walks_every_stage_and_records_per_method_history():
    run = trainer()
    built = []

    def train_loader_for(stage):
        built.append(stage)
        return [batch(list(range(stage + 1)))]

    history = run.fit(train_loader_for, [batch([0, 1, 2, 3])], epochs_per_stage=2, progress=False)

    assert built == [0, 1, 2, 3], "the loader must be rebuilt once per stage, in order"
    assert run.epoch == 8
    assert history["stage"] == [0, 0, 1, 1, 2, 2, 3, 3]
    for name in CURRICULUM:
        assert len(history[f"val_{name}"]) == 8, f"val_{name} should have one entry per epoch"


def test_fit_can_run_a_subset_of_stages():
    run = trainer()
    history = run.fit(lambda stage: [batch([0])], None, epochs_per_stage=1, stages=[3], progress=False)
    assert history["stage"] == [3]
    assert run.stage == 3


def test_validation_history_covers_methods_training_has_not_reached():
    """Pinning validation to the last stage is what gives a method a baseline before it unlocks."""
    run = trainer()
    history = run.fit(
        lambda stage: [batch(list(range(stage + 1)))],
        [batch([0, 1, 2, 3])],
        epochs_per_stage=1,
        stages=[0],
        progress=False,
    )
    # Trained on 'none' alone, but every method still has a validation number.
    assert history["train_blackbar"][0] != history["train_blackbar"][0], "untrained method should be NaN in train"
    assert history["val_blackbar"][0] == history["val_blackbar"][0], "but measured in val"


def test_checkpoint_round_trips(tmp_path: Path):
    run = trainer()
    run.fit(lambda stage: [batch([0])], None, epochs_per_stage=1, stages=[1], progress=False)
    path = run.save(tmp_path / "facial.pt")

    restored = trainer()
    restored.load(path)
    assert restored.epoch == run.epoch
    assert restored.stage == run.stage
    assert same(restored.history, run.history)
    for a, b in zip(run.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)


def test_checkpoint_restores_the_encoder_phase(tmp_path: Path):
    """The phase decides the optimiser's groups, so it must be restored before its state is."""
    run = trainer(encoder_frozen=True)
    run.set_encoder_frozen(False)
    path = run.save(tmp_path / "facial.pt")

    restored = trainer(encoder_frozen=True)
    restored.load(path)
    assert restored.model.encoder_frozen is False
    assert len(restored.optimizer.param_groups) == 2


def test_a_checkpoint_for_other_methods_is_refused(tmp_path: Path):
    path = trainer().save(tmp_path / "facial.pt")
    with pytest.raises(ValueError, match="covers"):
        FacialTrainer(model=TinyFacial(methods=2), methods=["none", "blackbar"], device="cpu").load(path)


def test_describe_omits_methods_that_were_never_seen():
    losses = Losses(0.5, {"none": 0.25, "blackbar": float("nan")}, {"none": 4, "blackbar": 0})
    assert "none=0.2500" in losses.describe()
    assert "blackbar" not in losses.describe()


def test_inference_only_checkpoint_drops_the_optimiser(tmp_path: Path):
    """AdamW keeps two moments per parameter, so leaving them out is most of the file."""
    run = trainer()
    run.step(*batch([0, 1]))  # populate the optimiser's state

    full = run.save(tmp_path / "full.pt")
    small = run.save(tmp_path / "small.pt", inference_only=True)
    assert small.stat().st_size < full.stat().st_size

    payload = torch.load(small, map_location="cpu", weights_only=False)
    assert "optimizer" not in payload
    assert "model" in payload and payload["epoch"] == run.epoch


def test_an_inference_only_checkpoint_still_loads_and_runs(tmp_path: Path):
    """It cannot resume, but it must not fail to load."""
    run = trainer()
    run.fit(lambda stage: [batch([0])], None, epochs_per_stage=1, stages=[2], progress=False)
    path = run.save(tmp_path / "small.pt", inference_only=True)

    restored = trainer()
    restored.load(path)
    assert restored.epoch == run.epoch and restored.stage == run.stage
    for a, b in zip(run.model.parameters(), restored.model.parameters()):
        assert torch.equal(a, b)


def test_load_facial_returns_a_runnable_model_and_metadata(tmp_path: Path):
    from meles.training.facial import load_facial

    run = trainer()
    run.fit(lambda stage: [batch([0])], None, epochs_per_stage=1, stages=[1], progress=False)
    # 'load_facial' builds a real Facial, so the checkpoint has to come from one too.
    path = tmp_path / "facial.pt"
    torch.save(run.state(inference_only=True) | {"model": FacialTrainer(device="cpu").model.state_dict()}, path)

    model, meta = load_facial(path, device="cpu")
    assert meta["methods"] == list(CURRICULUM)
    assert meta["epoch"] == run.epoch
    assert not model.training, "a loaded model should arrive in eval mode"


def test_reconstruct_accepts_a_single_item_and_a_batch():
    from meles.training.facial import reconstruct

    run = trainer()
    anonymized, _, method = batch([0, 3])

    batched = reconstruct(run.model, anonymized, method)
    assert batched.shape == (2, CHANNELS, SIZE, SIZE)

    single = reconstruct(run.model, anonymized[0], 0)
    assert single.shape == (1, CHANNELS, SIZE, SIZE)


def test_reconstruct_runs_in_eval_mode_and_restores_it():
    """Batch-dependent normalisation would make a face depend on who shared its batch."""
    from meles.training.facial import reconstruct

    seen = []
    run = trainer()
    run.model.train()
    original = run.model.forward
    run.model.forward = lambda *a, **k: (seen.append(run.model.training), original(*a, **k))[1]

    reconstruct(run.model, batch([0])[0], 0)
    assert seen == [False]
    assert run.model.training


def test_reconstruct_broadcasts_an_int_method_over_the_batch():
    from meles.training.facial import reconstruct

    run = trainer()
    anonymized, _, _ = batch([0, 1, 2])
    by_int = reconstruct(run.model, anonymized, 2)
    by_tensor = reconstruct(run.model, anonymized, torch.tensor([2, 2, 2]))
    assert torch.allclose(by_int, by_tensor)


class TinyBackbone(nn.Module):
    """Returns (embedding, hidden) as IResNet.forward does."""

    def __init__(self, dim=16):
        super().__init__()
        self.fc = nn.Linear(CHANNELS * SIZE * SIZE, dim)

    def forward(self, x):
        return self.fc(x.flatten(1)), x


class TinyEncoder(nn.Module):
    """Stands in for FacialEncoder, which wraps the backbone as '.model'."""

    def __init__(self, dim=16):
        super().__init__()
        self.model = TinyBackbone(dim)

    def forward(self, x):
        return self.model(x)[1]


def identity_loss():
    from meles.training.facial import IdentityLoss

    return IdentityLoss(encoder=TinyEncoder())


def test_identity_loss_is_zero_for_an_exact_reconstruction():
    """1 - cos of a vector with itself is 0, whatever the encoder is."""
    loss = identity_loss()
    target = torch.rand(3, CHANNELS, SIZE, SIZE) * 2 - 1
    assert torch.allclose(loss(target, target), torch.zeros(3), atol=1e-5)


def test_identity_loss_is_per_item_and_in_range():
    loss = identity_loss()
    generated = torch.rand(4, CHANNELS, SIZE, SIZE) * 2 - 1
    target = torch.rand(4, CHANNELS, SIZE, SIZE) * 2 - 1
    values = loss(generated, target)
    assert values.shape == (4,)
    assert float(values.min()) >= -1e-5 and float(values.max()) <= 2.0 + 1e-5


def test_identity_loss_punishes_the_average_face():
    """The failure L1 rewards: one hedged output for everyone must score worse than the truth."""
    loss = identity_loss()
    targets = torch.rand(6, CHANNELS, SIZE, SIZE) * 2 - 1
    average = targets.mean(dim=0, keepdim=True).expand_as(targets)
    assert float(loss(average, targets).mean()) > float(loss(targets, targets).mean())


def test_identity_loss_backpropagates_to_the_generated_image():
    loss = identity_loss()
    generated = (torch.rand(2, CHANNELS, SIZE, SIZE) * 2 - 1).requires_grad_(True)
    target = torch.rand(2, CHANNELS, SIZE, SIZE) * 2 - 1
    loss(generated, target).mean().backward()
    assert generated.grad is not None and float(generated.grad.abs().sum()) > 0


def test_the_identity_critic_is_frozen_and_stays_in_eval():
    """A critic that trains would be grading itself; one in train mode would drift its BatchNorms."""
    loss = identity_loss()
    assert not any(p.requires_grad for p in loss.encoder.parameters())
    loss.train(True)
    assert not loss.encoder.training


def test_the_critic_takes_no_gradient_from_the_loss():
    loss = identity_loss()
    generated = (torch.rand(2, CHANNELS, SIZE, SIZE) * 2 - 1).requires_grad_(True)
    loss(generated, torch.rand(2, CHANNELS, SIZE, SIZE) * 2 - 1).mean().backward()
    assert all(p.grad is None for p in loss.encoder.parameters())


def test_weighted_criterion_returns_weighted_contributions():
    from meles.training.facial import WeightedCriterion

    criterion = WeightedCriterion({"a": (2.0, l1_per_item), "b": (0.5, l1_per_item)})
    generated = torch.zeros(3, CHANNELS, SIZE, SIZE)
    target = torch.full((3, CHANNELS, SIZE, SIZE), 0.4)
    terms = criterion(generated, target)

    assert set(terms) == {"a", "b"}
    assert torch.allclose(terms["a"], torch.full((3,), 0.8))
    assert torch.allclose(terms["b"], torch.full((3,), 0.2))


def test_an_empty_criterion_is_refused():
    from meles.training.facial import WeightedCriterion

    with pytest.raises(ValueError):
        WeightedCriterion({})


def test_the_trainer_backpropagates_the_weighted_sum():
    from meles.training.facial import WeightedCriterion

    run = trainer(criterion=WeightedCriterion({"a": (1.0, l1_per_item), "b": (3.0, l1_per_item)}))
    total, terms = run.step(*batch([0, 1]))
    assert set(terms) == {"a", "b"}
    assert torch.allclose(total, terms["a"] + terms["b"])
    assert torch.allclose(terms["b"], 3.0 * terms["a"])


def test_losses_split_by_term_as_well_as_by_method():
    from meles.training.facial import WeightedCriterion

    run = trainer(criterion=WeightedCriterion({"l1": (1.0, l1_per_item), "twice": (2.0, l1_per_item)}))
    losses = run.evaluate([batch([0, 3])])

    assert set(losses.per_term) == {"l1", "twice"}
    assert losses.per_term["twice"] == pytest.approx(2 * losses.per_term["l1"], rel=1e-4)
    assert losses.overall == pytest.approx(losses.per_term["l1"] + losses.per_term["twice"], rel=1e-4)
    # And each term is still split by anonymization.
    assert set(losses.per_term_method["l1"]) == set(CURRICULUM)
    assert losses.per_term_method["l1"]["none"] == losses.per_term_method["l1"]["none"]


def test_history_records_a_series_per_term_and_method():
    from meles.training.facial import WeightedCriterion

    run = trainer(criterion=WeightedCriterion({"l1": (1.0, l1_per_item), "id": (0.5, l1_per_item)}))
    history = run.fit(lambda s: [batch([0, 1])], [batch([0, 1, 2, 3])], epochs_per_stage=2, stages=[1], progress=False)

    assert len(history["val:l1"]) == 2
    assert len(history["val:id_blackbar"]) == 2
    assert len(history["train:l1_none"]) == 2
    # Every series lines up with the epoch axis.
    assert all(len(values) == 2 for key, values in history.items() if key.startswith(("train:", "val:")))


def test_a_plain_tensor_criterion_still_works():
    """The single-tensor contract must keep working; it is the default."""
    run = trainer(criterion=l1_per_item)
    total, terms = run.step(*batch([0]))
    assert list(terms) == ["loss"]
    assert torch.allclose(total, terms["loss"])
