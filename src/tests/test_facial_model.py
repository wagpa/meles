"""
``Facial`` pools N frames into one image. The conditioning added for multi-anonymization training
is optional, so the two things worth pinning are that the unconditional path still behaves exactly
as it did, and that the conditional one actually depends on the method it is handed.
"""

import torch

from meles.frame_aggregation.facial import Facial, FacialAggregator
from meles.training.facial import CURRICULUM

BATCH, FRAMES, SIZE = 2, 3, 112
CHANNELS, GRID = 512, 7


def features(batch=BATCH, frames=FRAMES, seed=0):
    torch.manual_seed(seed)
    return torch.randn(batch, frames, CHANNELS, GRID, GRID)


def test_aggregator_pools_any_number_of_frames_into_one():
    aggregator = FacialAggregator()
    for frames in (1, 3, 8):
        assert aggregator(features(frames=frames)).shape == (BATCH, CHANNELS, GRID, GRID)


def test_unconditional_aggregator_ignores_a_method_it_was_not_built_for():
    """Passing a method to a model built with methods=0 must not change or crash anything."""
    aggregator = FacialAggregator()
    x = features()
    assert aggregator.method is None
    assert torch.equal(aggregator(x), aggregator(x, torch.tensor([0, 1])))


def test_conditioning_starts_as_a_no_op():
    """The embedding is zero-initialised, so an untrained conditional model matches a plain one."""
    torch.manual_seed(7)
    conditional = FacialAggregator(methods=len(CURRICULUM))
    x = features()
    with torch.no_grad():
        without = conditional(x)
        with_method = conditional(x, torch.tensor([0, 3]))
    assert torch.allclose(without, with_method, atol=1e-6)


def test_a_trained_condition_changes_the_pooled_output():
    """Once the embedding is non-zero, which method is named has to matter."""
    torch.manual_seed(7)
    conditional = FacialAggregator(methods=len(CURRICULUM))
    with torch.no_grad():
        conditional.method.weight.normal_(0.0, 0.5)
        x = features()
        first = conditional(x, torch.tensor([0, 0]))
        second = conditional(x, torch.tensor([3, 3]))
    assert not torch.allclose(first, second, atol=1e-4)


def test_conditioning_is_per_item_not_per_batch():
    """Two items in one batch may carry different anonymizations."""
    torch.manual_seed(7)
    conditional = FacialAggregator(methods=len(CURRICULUM))
    with torch.no_grad():
        conditional.method.weight.normal_(0.0, 0.5)
        x = features()
        mixed = conditional(x, torch.tensor([0, 3]))
        uniform = conditional(x, torch.tensor([0, 0]))
    assert torch.allclose(mixed[0], uniform[0], atol=1e-6)
    assert not torch.allclose(mixed[1], uniform[1], atol=1e-4)


def test_facial_forward_takes_the_dataset_item_shape():
    model = Facial(methods=len(CURRICULUM))
    x = torch.randn(BATCH, FRAMES, 3, SIZE, SIZE)
    with torch.no_grad():
        out = model(x, torch.tensor([0, 2]))
    assert out.shape == (BATCH, 3, SIZE, SIZE)
    # The decoder ends on a tanh, which is the range 'to_tensor' puts the target in.
    assert float(out.min()) >= -1.0 and float(out.max()) <= 1.0


def test_facial_still_works_without_conditioning():
    model = Facial()
    assert model.methods == 0
    assert model(torch.randn(BATCH, FRAMES, 3, SIZE, SIZE)).shape == (BATCH, 3, SIZE, SIZE)


def test_freezing_the_encoder_keeps_the_rest_trainable():
    model = Facial(methods=len(CURRICULUM))
    model.set_encoder_frozen(True)
    assert model.fine_parameters() == []
    trainable = model.train_parameters()
    assert trainable and all(p.requires_grad for p in trainable)
    # The method embedding is new; it must be trained, not left frozen with the encoder.
    assert any(p is model.aggregator.method.weight for p in trainable)
