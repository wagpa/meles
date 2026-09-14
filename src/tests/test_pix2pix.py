"""
Tests for the pix2pix de-anonymizer: the networks, the training pairs and the checkpoint.

Three things here are load-bearing rather than incidental, and each is a silent failure if it
breaks. The discriminator must return **logits**, because the losses are ``BCEWithLogitsLoss`` and
a sigmoid on both sides flattens the gradient instead of raising. The pairs must actually differ,
because an input equal to its target teaches the generator to copy. And a checkpoint must carry
the anonymization it was trained against, because a model pointed at the wrong branch of the sweep
produces plausible frames and a meaningless number.

The forward passes run at the full 256x256 -- the U-Net's eight halvings do not work out at any
smaller size -- so they are the slow part of this module. Batches are kept at two, which is the
smallest a BatchNorm at 1x1 tolerates.
"""

import pickle
import zipfile

import numpy as np
import pytest
import torch

from meles.anonymization import ANONYMIZERS
from meles.frame_aggregation.pix2pix import (
    SIZE,
    Pix2Pix,
    Pix2PixDiscriminator,
    Pix2PixGenerator,
    load_checkpoint,
    load_generator,
    to_image,
    to_tensor,
)
from meles.training import AnonymizedPairs, ImageArchive, ImageDirectory, Pix2PixTrainer, split


class Faces:
    """A stand-in for a torchvision image dataset: indexable, sized, yielding (image, target)."""

    def __init__(self, count: int = 4, size: tuple[int, int] = (218, 178)):
        rng = np.random.default_rng(0)
        self.images = [rng.integers(0, 256, (*size, 3), dtype=np.uint8) for _ in range(count)]

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int):
        # RGB, as a PIL image would convert to -- the wrapper is what turns it into meles' BGR.
        return self.images[index], 0


@pytest.fixture
def frame() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 256, (120, 90, 3), dtype=np.uint8)


def test_a_frame_survives_the_round_trip(frame: np.ndarray) -> None:
    """The tensor conversion changes colour order, layout and range; none of it may leak."""
    tensor = to_tensor(frame)

    assert tensor.shape == (3, SIZE, SIZE)
    assert tensor.dtype == torch.float32
    assert -1.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0

    restored = to_image(tensor, frame.shape[:2])
    assert restored.shape == frame.shape
    assert restored.dtype == np.uint8


def test_the_round_trip_keeps_the_colours(frame: np.ndarray) -> None:
    """
    A BGR/RGB slip is invisible in a shape assertion and visible in every generated face, so it is
    pinned on a frame whose channels are unmistakable.
    """
    blue = np.zeros((64, 64, 3), np.uint8)
    blue[:, :, 0] = 255  # BGR, so this is blue

    restored = to_image(to_tensor(blue), blue.shape[:2])

    assert restored[:, :, 0].min() > 250
    assert restored[:, :, 1:].max() < 5


def test_the_discriminator_returns_logits() -> None:
    """
    Not probabilities. The losses apply the sigmoid themselves, so a Sigmoid in the network would
    square it -- which does not fail, it just stops the model from learning.
    """
    with torch.no_grad():
        verdict = Pix2PixDiscriminator()(torch.randn(2, 3, SIZE, SIZE), torch.randn(2, 3, SIZE, SIZE))

    # 30x30 patches at 256x256, the receptive field the paper calls 70x70.
    assert verdict.shape == (2, 1, 30, 30)
    assert float(verdict.min()) < 0.0 < float(verdict.max()), "a sigmoid would have clamped these into (0, 1)"


def test_the_generator_returns_a_frame_shaped_tensor() -> None:
    """The skips only line up if every encoder level matches its decoder level."""
    with torch.no_grad():
        generated = Pix2PixGenerator()(torch.randn(2, 3, SIZE, SIZE))

    assert generated.shape == (2, 3, SIZE, SIZE)
    # The final Tanh has to match the range 'to_tensor' puts the targets in.
    assert -1.0 <= float(generated.min()) and float(generated.max()) <= 1.0


def test_the_generator_survives_a_batch_of_one() -> None:
    """
    The innermost encoder block works at 1x1, where a BatchNorm has one spatial sample per channel
    and raises on a batch of one. Inference with a single frame has to work regardless.
    """
    with torch.no_grad():
        assert Pix2PixGenerator()(torch.randn(1, 3, SIZE, SIZE)).shape == (1, 3, SIZE, SIZE)


def test_deanonymize_returns_one_frame_per_input_at_its_own_size() -> None:
    """The inference entry point has to behave like the other frame-wise de-anonymizations."""
    frames = [np.random.default_rng(seed).integers(0, 256, (60 + seed, 50, 3), dtype=np.uint8) for seed in range(3)]

    restored = Pix2PixGenerator().deanonymize(frames)

    assert len(restored) == len(frames)
    for before, after in zip(frames, restored):
        assert after.shape == before.shape
        assert after.dtype == np.uint8


@pytest.mark.parametrize("method", sorted(set(ANONYMIZERS) - {"none"}))
def test_the_pairs_anonymize_the_clear_image(method: str) -> None:
    """
    The input half of a pair is the target with the selected anonymization applied. If the two
    came out equal the L1 term would be trained to copy, so a differing pair is the whole point.
    """
    pairs = AnonymizedPairs(Faces(), method)

    anonymized, clear = pairs[0]

    assert anonymized.shape == clear.shape == (3, SIZE, SIZE)
    assert not torch.equal(anonymized, clear)


def test_the_baseline_pairs_are_identical() -> None:
    """'none' has nothing to undo, which is what makes it the baseline rather than a bug."""
    anonymized, clear = AnonymizedPairs(Faces(), "none")[0]

    assert torch.equal(anonymized, clear)


def test_an_undetected_face_still_makes_a_usable_pair() -> None:
    """
    'anonymize_frame' leaves an undetected frame untouched; doing that here would emit a pair whose
    input equals its target. The whole image is anonymized instead, and the miss is counted.
    """
    pairs = AnonymizedPairs(Faces(), "blackbar", on_missing="whole_image")

    anonymized, clear = pairs[0]

    # Random noise, so the detector finds nothing and the fallback is what produced this.
    assert pairs.missing == 1
    assert not torch.equal(anonymized, clear)
    # A black bar over the whole image leaves nothing but the floor of the value range.
    assert float(anonymized.max()) == pytest.approx(-1.0)


def test_detecting_up_front_agrees_with_detecting_per_item() -> None:
    """Pre-detection is an optimisation; it must not change which region gets anonymized."""
    lazy = AnonymizedPairs(Faces(), "pixelation")
    eager = AnonymizedPairs(Faces(), "pixelation")
    eager.detect(progress=False)

    assert eager.areas is not None
    assert eager.missing == len(eager)
    assert torch.equal(lazy[0][0], eager[0][0])


def test_a_checkpoint_round_trips(tmp_path) -> None:
    """Export then import has to return the same generator, and say what it was trained against."""
    trainer = Pix2PixTrainer("pixelation", device="cpu")
    trainer.epoch = 3
    trainer.history["train_g"].append(1.5)

    path = trainer.save(tmp_path / "pixelation.pt")
    loaded = Pix2PixTrainer.resume("pixelation", path, device="cpu")

    assert loaded.epoch == 3
    assert loaded.history["train_g"] == [1.5]
    for before, after in zip(trainer.generator.state_dict().values(), loaded.generator.state_dict().values()):
        assert torch.equal(before, after)


def test_an_inference_only_checkpoint_keeps_only_the_generator(tmp_path) -> None:
    """The discriminator and the optimisers are three quarters of the file and dead weight at inference."""
    path = Pix2PixTrainer("blackbar", device="cpu").save(tmp_path / "blackbar.pt", inference_only=True)

    checkpoint = load_checkpoint(path)

    assert checkpoint.anonymizer == "blackbar"
    assert checkpoint.generator
    assert checkpoint.discriminator is None
    assert checkpoint.generator_optimizer is None


def test_importing_a_model_for_the_wrong_anonymization_is_refused(tmp_path) -> None:
    """
    One model per anonymization method. A pixelation model handed blurred frames produces output
    that looks fine, so nothing downstream would notice -- which is why it has to fail here.
    """
    path = Pix2PixTrainer("pixelation", device="cpu").save(tmp_path / "pixelation.pt", inference_only=True)

    assert load_generator(path, anonymizer="pixelation") is not None
    with pytest.raises(ValueError, match="pixelation"):
        load_generator(path, anonymizer="gaussianblur")


def test_a_checkpoint_of_another_format_is_refused(tmp_path) -> None:
    """A silently mismatched payload would load into the wrong state dicts."""
    path = tmp_path / "stale.pt"
    torch.save({"format": 0, "anonymizer": "none"}, path)

    with pytest.raises(ValueError, match="format"):
        load_checkpoint(path)


def test_a_training_step_moves_both_networks() -> None:
    """
    The step updates the discriminator and then the generator. If the detach were missing the
    discriminator's backward would reach into the generator, which shows up as wasted work rather
    than as an error -- so what is pinned here is that both networks actually change.
    """
    torch.manual_seed(0)
    trainer = Pix2PixTrainer("pixelation", model=Pix2Pix(), device="cpu")
    before = {name: tensor.clone() for name, tensor in trainer.model.state_dict().items()}

    generator_loss, discriminator_loss = trainer.step(torch.randn(2, 3, SIZE, SIZE), torch.randn(2, 3, SIZE, SIZE))

    assert np.isfinite(generator_loss) and np.isfinite(discriminator_loss)
    after = trainer.model.state_dict()
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("generator."))
    assert any(not torch.equal(before[name], after[name]) for name in before if name.startswith("discriminator."))


def image_bytes(size: tuple[int, int] = (64, 48), seed: int = 0) -> bytes:
    """A decodable PNG, so that the sources are exercised on real encoded files."""
    import cv2

    pixels = np.random.default_rng(seed).integers(0, 256, (*size, 3), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", pixels)
    assert ok
    return encoded.tobytes()


def test_an_image_directory_finds_images_at_any_depth(tmp_path) -> None:
    """The CelebA mirrors disagree about how deeply they nest, so the walk has to be recursive."""
    (tmp_path / "nested" / "deeper").mkdir(parents=True)
    (tmp_path / "a.jpg").write_bytes(image_bytes())
    (tmp_path / "nested" / "b.PNG").write_bytes(image_bytes(seed=1))
    (tmp_path / "nested" / "deeper" / "c.jpeg").write_bytes(image_bytes(seed=2))
    (tmp_path / "nested" / "notes.txt").write_text("ignored")

    directory = ImageDirectory(tmp_path)

    assert len(directory) == 3
    assert directory[0].shape == (64, 48, 3)
    # RGB, as PIL and torchvision yield; 'as_image' is what converts to the BGR meles speaks.
    assert directory[0].dtype == np.uint8


def test_an_empty_directory_is_refused(tmp_path) -> None:
    """Better than a dataset of length zero, which only fails much later and less clearly."""
    with pytest.raises(FileNotFoundError, match="No"):
        ImageDirectory(tmp_path)


def test_an_archive_reads_images_without_extracting(tmp_path) -> None:
    """Reading the zip in place is what avoids unpacking 1.4 GiB of CelebA a second time."""
    path = tmp_path / "faces.zip"
    with zipfile.ZipFile(path, "w") as archive:
        # The nesting the Kaggle mirror uses, plus a stray file that is not an image.
        archive.writestr("img_align_celeba/img_align_celeba/000001.jpg", image_bytes())
        archive.writestr("img_align_celeba/img_align_celeba/000002.jpg", image_bytes(seed=1))
        archive.writestr("img_align_celeba/list_attr_celeba.csv", "not an image")

    source = ImageArchive(path)

    assert len(source) == 2
    assert source[0].shape == (64, 48, 3)


def test_an_archive_drops_its_handle_when_pickled(tmp_path) -> None:
    """
    A ZipFile has one file position, so a DataLoader worker reading through an inherited handle
    would interleave its seeks with the parent's and decode torn data -- intermittently.
    """
    path = tmp_path / "faces.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("000001.jpg", image_bytes())

    source = ImageArchive(path)
    first = source[0]  # opens the handle
    revived = pickle.loads(pickle.dumps(source))

    assert revived._archive is None
    assert np.array_equal(revived[0], first)


def test_the_split_is_disjoint_and_reproducible() -> None:
    """A rerun has to hold out the same images, or the validation loss is not comparable."""
    source = Faces(count=100)

    train, validation = split(source, validation=0.2, seed=0)
    again, _ = split(source, validation=0.2, seed=0)

    assert len(train) == 80 and len(validation) == 20
    assert set(train.indices).isdisjoint(validation.indices)
    assert set(train.indices) | set(validation.indices) == set(range(100))
    assert train.indices == again.indices


def test_detecting_on_threads_agrees_with_detecting_on_one() -> None:
    """Parallelism is an optimisation; it must not reorder the areas it returns."""
    pairs = AnonymizedPairs(Faces(count=6), "pixelation")

    serial = list(AnonymizedPairs(Faces(count=6), "pixelation").detect(progress=False, workers=1))
    threaded = pairs.detect(progress=False, workers=4)

    assert len(threaded) == 6
    assert [area is None for area in threaded] == [area is None for area in serial]
