"""
``Facial`` pools N frames into one image. The conditioning added for multi-anonymization training
is optional, so the two things worth pinning are that the unconditional path still behaves exactly
as it did, and that the conditional one actually depends on the method it is handed.
"""

import pytest
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


def test_the_default_decoder_is_unchanged_so_old_checkpoints_still_load():
    """width/dropout were added after three runs were trained; their files must keep loading."""
    from meles.frame_aggregation.facial import FacialDecoder

    decoder = FacialDecoder()
    assert decoder.chans == (256, 192, 128, 64)
    # Dropout and Identity carry no parameters, so the state_dict is exactly what it always was:
    # same keys, same shapes, which is all load_state_dict checks.
    state = decoder.state_dict()
    assert not any("drop" in key for key in state)
    assert state["blocks.0.conv1.weight"].shape == (256, 512, 3, 3)
    assert state["blocks.3.conv2.weight"].shape == (64, 64, 3, 3)
    assert state["out.2.weight"].shape == (3, 64, 3, 3)


def test_width_scales_the_ladder_and_the_parameter_count():
    from meles.frame_aggregation.facial import FacialDecoder

    narrow, wide = FacialDecoder(), FacialDecoder(width=2.0)
    assert wide.chans == (512, 384, 256, 128)
    # Not quite 4x: the first block's input is fixed at 512 by the aggregator, so only its output
    # widens. The blocks after it scale on both sides.
    ratio = sum(p.numel() for p in wide.parameters()) / sum(p.numel() for p in narrow.parameters())
    assert 2.9 < ratio < 3.4, f"doubling the width should roughly triple the decoder, got {ratio:.2f}x"


def test_width_keeps_every_channel_valid_for_groupnorm():
    """GroupNorm needs channels divisible by its group count, at any width."""
    from meles.frame_aggregation.facial import FacialDecoder

    for width in (0.01, 0.25, 0.5, 1.0, 1.3, 1.75, 2.0, 3.0):
        decoder = FacialDecoder(width=width)
        assert all(c % 8 == 0 and c > 0 for c in decoder.chans), (width, decoder.chans)
        assert decoder(torch.randn(1, 512, 7, 7)).shape == (1, 3, 112, 112)


def test_dropout_is_active_in_train_and_inert_in_eval():
    from meles.frame_aggregation.facial import FacialDecoder

    torch.manual_seed(0)
    decoder = FacialDecoder(dropout=0.5)
    x = torch.randn(4, 512, 7, 7)

    decoder.eval()
    with torch.no_grad():
        assert torch.equal(decoder(x), decoder(x)), "eval must be deterministic"

    decoder.train()
    with torch.no_grad():
        assert not torch.equal(decoder(x), decoder(x)), "train must sample a fresh mask"


def test_dropout_zero_is_a_true_no_op():
    from meles.frame_aggregation.facial import FacialDecoder

    torch.manual_seed(0)
    plain = FacialDecoder(dropout=0.0).train()
    x = torch.randn(2, 512, 7, 7)
    with torch.no_grad():
        assert torch.equal(plain(x), plain(x))


def test_facial_passes_width_and_dropout_to_its_decoder():
    model = Facial(methods=len(CURRICULUM), width=2.0, dropout=0.1)
    assert model.decoder.chans == (512, 384, 256, 128)
    with torch.no_grad():
        out = model.eval()(torch.randn(2, 3, 3, SIZE, SIZE), torch.tensor([0, 1]))
    assert out.shape == (2, 3, SIZE, SIZE)


def test_a_default_checkpoint_loads_into_a_default_model_but_not_a_wider_one():
    """The shapes differ, so a width change needs its own checkpoint file -- loudly, not silently."""
    state = Facial(methods=len(CURRICULUM)).state_dict()
    Facial(methods=len(CURRICULUM)).load_state_dict(state)

    with pytest.raises(RuntimeError, match="size mismatch|shape"):
        Facial(methods=len(CURRICULUM), width=2.0).load_state_dict(state)


def test_a_wider_decoder_still_trains():
    """The extra capacity has to be reachable by the optimiser, not just allocated."""
    model = Facial(methods=len(CURRICULUM), width=2.0, dropout=0.1)
    x = torch.randn(2, 2, 3, SIZE, SIZE)
    target = torch.randn(2, 3, SIZE, SIZE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    first = (model(x, torch.tensor([0, 1])) - target).abs().mean()
    for _ in range(5):
        optimizer.zero_grad()
        (model(x, torch.tensor([0, 1])) - target).abs().mean().backward()
        optimizer.step()
    assert (model(x, torch.tensor([0, 1])) - target).abs().mean() < first
