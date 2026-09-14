The first step to a custom model will be a frame-based pix2pix deanonymization model. It takes an anonymized image and 
generates a de-anonymized variant from it.

Train one model per anonymization method, reference in workflow.

## Where it lives

- `src/meles/frame_aggregation/pix2pix.py` — the networks (U-Net generator, 70x70 PatchGAN
  discriminator), the two losses, the tensor/frame conversion, and the checkpoint format.
- `src/meles/training/sources.py` — where the clear faces come from: `ImageArchive` (reads a zip
  in place), `ImageDirectory`, and a seeded `split`.
- `src/meles/training/pairs.py` — `AnonymizedPairs`, which turns a clear-face dataset into
  (anonymized, clear) training pairs by applying one of `meles.anonymization.ANONYMIZERS` to each
  image's detected facial area. That registry is shared with
  `workflow/scripts/anonymize_frame.py`, so the model is trained against exactly the anonymization
  the sweep produces.
- `src/meles/training/pix2pix.py` — `Pix2PixTrainer`: model, optimisers, loss history, and
  `save`/`load`/`resume`.
- `src/notebooks/08_train_pix2pix.ipynb` — the run.

## The training images

**`torchvision.datasets.CelebA` is deliberately not used.** Its `_check_integrity` wants five
*space-separated* `.txt` annotation files matching MD5s of the original Google Drive upload, and
one of them (`identity_CelebA.txt`) is not in the common Kaggle mirror at all — that mirror ships
`.csv`. Any of them missing raises `Dataset not found or corrupted`, which reads as a broken
download but really means "these are not the bytes I expected". Moving files around cannot fix it,
and the `.txt` files cannot be reconstructed byte-identically from the CSVs.

None of that metadata is used here anyway — the pairs are made from the images alone. So the
sources read images and nothing else:

```py
from meles.training import ImageArchive, ImageDirectory, split

source = ImageArchive("data/celeba/img_align_celeba.zip")   # no extraction needed
source = ImageDirectory("data/celeba/img_align_celeba")     # or an already-unpacked folder
train, validation = split(source, validation=0.02)          # seeded, so a rerun holds out the same
```

`ImageArchive` matches members by suffix wherever they sit, so the mirrors that nest as
`img_align_celeba/*.jpg` and as `img_align_celeba/img_align_celeba/*.jpg` both work unchanged.

Any corpus of face images does, as long as it is **not** YouTube Faces — an attacker may train on
whatever they like, but training on the dataset the experiment scores would put the gallery into
the generator's prior. (Worth stating as a limitation either way: CelebA and YouTube Faces are
both celebrity datasets and their identities are not known to be disjoint.)

## Training

```py
from torch.utils.data import DataLoader
from meles.training import AnonymizedPairs, ImageArchive, Pix2PixTrainer, checkpoint_path, split

train, validation = split(ImageArchive("data/celeba/img_align_celeba.zip"))
pairs = AnonymizedPairs(train, "pixelation")
pairs.detect()                                  # ~2 min over CelebA on 8 threads, not once per epoch

loader = DataLoader(pairs, batch_size=32, shuffle=True, num_workers=8, drop_last=True)
trainer = Pix2PixTrainer.resume("pixelation")   # continues from a checkpoint if there is one
trainer.fit(loader, epochs=50, checkpoint=checkpoint_path("pixelation"))
```

`drop_last=True` is not optional: the U-Net's innermost block works at 1x1, where a BatchNorm
raises on a batch of one.

## Export and import

Checkpoints go to `models/pix2pix/{anonymizer}.pt` (`MELES_WEIGHTS_DIR` overrides the root, and
`models/` is gitignored).

```py
trainer.save()                        # generator + discriminator + optimisers + history, resumable
trainer.save(path, inference_only=True)   # generator only, ~207 MiB instead of ~654

from meles.frame_aggregation.pix2pix import load_generator
generator = load_generator(path, device="cuda", anonymizer="pixelation")
frames = generator.deanonymize(anonymized_frames)   # BGR uint8 in, BGR uint8 out, sizes preserved
```

Every checkpoint records the anonymization it was trained against, and `load_generator` refuses a
mismatch. A pixelation model handed blurred frames produces plausible output and a meaningless
score, so nothing downstream would catch it.

## Still to do

Wire it into the sweep: a `FrameAggregator` subclass over `generator.deanonymize`, registered in
`AGGREGATORS` in `workflow/scripts/aggregate_frame.py` as a `frame_*` method (it is frame-wise —
one frame in, one frame out — like the resample de-anonymizations). Open question there is how the
rule picks the checkpoint: the registry key would have to name the anonymization the model undoes,
and it should be checked against the `frame_anonymization` wildcard of the branch it runs in.
