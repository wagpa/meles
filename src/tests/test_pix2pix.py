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
import random
import zipfile
from dataclasses import astuple

import numpy as np
import pytest
import torch

from meles.anonymization import ANONYMIZERS
from meles.frame_aggregation.pix2pix import (
    LOAD_SIZE,
    SIZE,
    Pix2Pix,
    UnetGenerator,
    load_checkpoint,
    load_generator,
    save_checkpoint,
    to_image,
    to_tensor,
)
from meles.training import AnonymizedPairs, ImageArchive, ImageDirectory, Pix2PixTrainer, split
from meles.training.pairs import augment, augmentation_params


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


def test_the_generator_returns_a_frame_shaped_tensor() -> None:
    """The skips only line up if every encoder level matches its decoder level."""
    with torch.no_grad():
        generated = UnetGenerator()(torch.randn(2, 3, SIZE, SIZE))

    assert generated.shape == (2, 3, SIZE, SIZE)
    # The final Tanh has to match the range 'to_tensor' puts the targets in.
    assert -1.0 <= float(generated.min()) and float(generated.max()) <= 1.0


def test_the_generator_survives_a_batch_of_one() -> None:
    """
    The innermost encoder block works at 1x1, where a BatchNorm has one spatial sample per channel
    and raises on a batch of one. Inference with a single frame has to work regardless.
    """
    with torch.no_grad():
        assert UnetGenerator()(torch.randn(1, 3, SIZE, SIZE)).shape == (1, 3, SIZE, SIZE)


def test_deanonymize_returns_one_frame_per_input_at_its_own_size() -> None:
    """The inference entry point has to behave like the other frame-wise de-anonymizations."""
    frames = [np.random.default_rng(seed).integers(0, 256, (60 + seed, 50, 3), dtype=np.uint8) for seed in range(3)]

    restored = UnetGenerator().deanonymize(frames)

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
    pairs = AnonymizedPairs(Faces(), method, load_size=None)

    anonymized, clear = pairs[0]

    assert anonymized.shape == clear.shape == (3, SIZE, SIZE)
    assert not torch.equal(anonymized, clear)


def test_the_baseline_pairs_are_identical() -> None:
    """'none' has nothing to undo, which is what makes it the baseline rather than a bug."""
    anonymized, clear = AnonymizedPairs(Faces(), "none", load_size=None)[0]

    assert torch.equal(anonymized, clear)


def test_an_undetected_face_still_makes_a_usable_pair() -> None:
    """
    'anonymize_frame' leaves an undetected frame untouched; doing that here would emit a pair whose
    input equals its target. The whole image is anonymized instead, and the miss is counted.
    """
    pairs = AnonymizedPairs(Faces(), "blackbar", on_missing="whole_image", load_size=None)

    anonymized, clear = pairs[0]

    # Random noise, so the detector finds nothing and the fallback is what produced this.
    assert pairs.missing == 1
    assert not torch.equal(anonymized, clear)
    # A black bar over the whole image leaves nothing but the floor of the value range.
    assert float(anonymized.max()) == pytest.approx(-1.0)


def test_detecting_up_front_agrees_with_detecting_per_item() -> None:
    """Pre-detection is an optimisation; it must not change which region gets anonymized."""
    lazy = AnonymizedPairs(Faces(), "pixelation", load_size=None)
    eager = AnonymizedPairs(Faces(), "pixelation", load_size=None)
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

    losses = trainer.step(torch.randn(2, 3, SIZE, SIZE), torch.randn(2, 3, SIZE, SIZE))

    assert all(np.isfinite(value) for value in astuple(losses))
    # The total is the weighted sum of the two terms, which is what makes it useless on its own.
    assert losses.generator == pytest.approx(losses.adversarial + trainer.lambda_l1 * losses.l1, rel=1e-5)
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
    pairs = AnonymizedPairs(Faces(count=6), "pixelation", load_size=None)

    serial = list(AnonymizedPairs(Faces(count=6), "pixelation", load_size=None).detect(progress=False, workers=1))
    threaded = pairs.detect(progress=False, workers=4)

    assert len(threaded) == 6
    assert [area is None for area in threaded] == [area is None for area in serial]


def test_validation_does_not_move_the_batchnorm_statistics() -> None:
    """
    BatchNorm updates its running statistics inside the forward pass whenever it is in 'train'
    mode, and 'no_grad' does not stop it. A run that validated in train mode would spend every
    epoch fitting its normalisation to held-out data -- silently, and only visible later through
    'eval'.
    """
    trainer = Pix2PixTrainer("pixelation", device="cpu")
    trainer.model.train()
    before = {
        name: buffer.clone()
        for name, buffer in trainer.model.named_buffers()
        if name.endswith(("running_mean", "running_var"))
    }
    assert before, "the model should have BatchNorm buffers to begin with"

    batch = [(torch.randn(2, 3, SIZE, SIZE) * 4 + 2, torch.randn(2, 3, SIZE, SIZE))]
    trainer.evaluate(batch)

    after = dict(trainer.model.named_buffers())
    assert all(torch.equal(before[name], after[name]) for name in before)
    # And the mode is handed back as it was found, so 'fit' keeps training after validating.
    assert trainer.model.training


def test_evaluating_reports_the_generator_terms_apart() -> None:
    """The total conflates reconstruction with how far ahead the discriminator is."""
    trainer = Pix2PixTrainer("pixelation", device="cpu")

    losses = trainer.evaluate([(torch.randn(2, 3, SIZE, SIZE), torch.randn(2, 3, SIZE, SIZE))])

    assert losses.l1 > 0.0
    assert losses.generator == pytest.approx(losses.adversarial + trainer.lambda_l1 * losses.l1, rel=1e-5)


def test_an_empty_loader_evaluates_to_nan() -> None:
    """Better than 0.0, which would plot as a perfect epoch."""
    assert all(np.isnan(value) for value in astuple(Pix2PixTrainer("none", device="cpu").evaluate([])))


def test_resuming_a_checkpoint_without_the_newer_series_pads_it(tmp_path) -> None:
    """
    A checkpoint written before 'adv'/'l1' were recorded still has to resume, and its history has
    to stay index-alignable -- otherwise a plot of the resumed run shifts one curve against the
    rest without saying so.
    """
    trainer = Pix2PixTrainer("pixelation", device="cpu")
    trainer.epoch = 3
    state = trainer.state()
    state.history = {"train_g": [1.0, 2.0, 3.0], "train_d": [0.5, 0.4, 0.3], "val_g": [], "val_d": []}
    path = save_checkpoint(state, tmp_path / "old.pt")

    loaded = Pix2PixTrainer("pixelation", device="cpu").load(path)

    assert loaded.history["train_g"] == [1.0, 2.0, 3.0]
    assert len(loaded.history["train_l1"]) == 3
    assert all(np.isnan(value) for value in loaded.history["train_l1"])


def test_the_pair_is_cropped_and_flipped_together() -> None:
    """
    pix2pix's whole augmentation is a random 256 crop out of a 286 resize plus a random mirror,
    and both ends of a pair must get the *same* draw. Cropped independently the pair is no longer
    pixel-aligned, and the L1 term would be asking the generator to reconstruct a shifted face --
    which does not fail, it just trains a blur.
    """
    # 'none' leaves the image alone, so the two ends of the pair start out identical and can only
    # still be identical afterwards if they were cropped and mirrored together. Independent draws
    # would show up here as a shifted or mirrored copy, which is exactly the failure being ruled
    # out -- and one that trains a blur rather than raising.
    pairs = AnonymizedPairs(Faces(count=1), "none", load_size=LOAD_SIZE)

    random.seed(0)
    anonymized, clear = pairs[0]

    assert anonymized.shape == clear.shape == (3, SIZE, SIZE)
    assert torch.equal(anonymized, clear)

    # And the draw has to actually vary, or it is not an augmentation.
    varied = set()
    for seed in range(8):
        random.seed(seed)
        varied.add(pairs[0][1].numpy().tobytes())
    assert len(varied) > 1


def test_turning_the_augmentation_off_is_deterministic() -> None:
    """What a validation split wants: a loss that is comparable between epochs."""
    pairs = AnonymizedPairs(Faces(count=1), "pixelation", load_size=None)

    random.seed(0)
    first = pairs[0]
    random.seed(1)
    second = pairs[0]

    assert torch.equal(first[0], second[0]) and torch.equal(first[1], second[1])


def test_the_augmentation_draws_a_crop_inside_the_resized_image() -> None:
    """An out-of-range corner would slice a short array and yield a wrongly shaped tensor."""
    for seed in range(20):
        random.seed(seed)
        (x, y), _ = augmentation_params(LOAD_SIZE, SIZE)
        assert 0 <= x <= LOAD_SIZE - SIZE and 0 <= y <= LOAD_SIZE - SIZE


def test_the_augmentation_mirrors() -> None:
    """The flip has to actually flip, and only when asked."""
    image = np.zeros((64, 64, 3), np.uint8)
    image[:, :10] = 255

    kept = augment(image, (0, 0), False, 64, 64)
    mirrored = augment(image, (0, 0), True, 64, 64)

    assert np.array_equal(kept, image)
    assert np.array_equal(mirrored, image[:, ::-1])


def test_the_checkpoint_carries_the_schedule(tmp_path) -> None:
    """
    A resumed run has to pick the learning rate ramp back up where it left it. Losing the schedule
    would silently restart the decay, which is a different model than the one being described.
    """
    trainer = Pix2PixTrainer("pixelation", device="cpu", n_epochs=7, n_epochs_decay=3)
    trainer.epoch = 8

    path = trainer.save(tmp_path / "pixelation.pt")
    resumed = Pix2PixTrainer.resume("pixelation", path, device="cpu")

    assert (resumed.n_epochs, resumed.n_epochs_decay) == (7, 3)
    assert resumed.learning_rate_at(9) == trainer.learning_rate_at(9)


def test_a_stale_schedule_is_reported_rather_than_applied_silently(tmp_path, caplog) -> None:
    """
    Editing the schedule in the notebook and leaving the old checkpoint in place would otherwise
    train quietly to the old plan -- the checkpoint has to win, so the mismatch has to be said.
    """
    written = Pix2PixTrainer("pixelation", device="cpu", n_epochs=100, n_epochs_decay=100)
    written.epoch = 1
    path = written.save(tmp_path / "pixelation.pt")

    with caplog.at_level("WARNING"):
        resumed = Pix2PixTrainer.resume("pixelation", path, device="cpu", n_epochs=50, n_epochs_decay=50)

    assert (resumed.n_epochs, resumed.n_epochs_decay) == (100, 100)
    # Asserted on the record rather than on caplog.text: the useful part rides in 'extra', which
    # only 'meles.runtime.logs.ExtraFormatter' renders into the message.
    (record,) = [entry for entry in caplog.records if "schedule" in entry.message]
    assert record.configured == (50, 50)
    assert record.checkpoint == (100, 100)
    assert "delete" in record.remedy


def test_a_matching_schedule_resumes_quietly(tmp_path, caplog) -> None:
    """The warning has to stay rare enough to mean something."""
    path = Pix2PixTrainer("pixelation", device="cpu", n_epochs=50, n_epochs_decay=50).save(tmp_path / "p.pt")

    with caplog.at_level("WARNING"):
        Pix2PixTrainer.resume("pixelation", path, device="cpu", n_epochs=50, n_epochs_decay=50)

    assert "schedule" not in caplog.text
