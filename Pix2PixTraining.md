# Configuring a pix2pix training run

What each knob in `src/notebooks/08_train_pix2pix.ipynb` does, what it costs, and which ones are
deviations from [what SEBA runs](FrameModel.md).

## The unit that matters is steps, not epochs

pix2pix's defaults are written for the paired datasets it ships with: **facades is 400 images**,
maps is 1,096. `batch_size=1` and a 200-epoch schedule are calibrated for that, and upstream's own
facades run is 400 × 200 = **80,000 optimiser steps**.

SEBA inherits those defaults on a set that is just as small. Its selector takes `ids: 20`, CelebA
averages 19.9 images per identity, and its `train_rate` is 0.85, so SEBA's pix2pix trains on about
**340 images**; its sample config's `epochs: 100` becomes `--n_epochs 50 --n_epochs_decay 50`. That
is ~34,000 steps.

"Epochs" therefore says nothing on its own. An epoch over SEBA's corpus is 340 steps; an epoch over
all of CelebA is 172,209. **Decide the step count first**, then pick a corpus and a schedule that
multiply out to it.

```
steps = CORPUS * TRAIN_RATE * (N_EPOCHS + N_EPOCHS_DECAY) / BATCH_SIZE
```

The notebook prints this when you run the config cell. Compare it against the 80,000 above before
starting anything long.

## What a run costs

Measured at **72 optimiser steps/s** (one RTX-class GPU, `batch_size=1`, 256×256, 8 dataloader
workers). Measure your own: run one epoch and divide the training-set size by the seconds it took,
then rescale this table.

| `CORPUS` | training images | schedule | steps | min/epoch | wall time |
|---------:|----------------:|---------:|------:|----------:|----------:|
| 400 (SEBA's) | 340 | 50+50 | 34,000 | 0.1 | 8 min |
| 2,000 | 1,700 | 50+50 | 170,000 | 0.4 | 39 min |
| 5,000 | 4,250 | 100+100 | 850,000 | 1.0 | 3.3 h |
| 20,000 | 17,000 | 100+100 | 3,400,000 | 3.9 | 13.1 h |
| 202,599 (all) | 172,209 | 100+100 | 34,441,800 | 39.9 | 5.5 days |

`CORPUS` is what you set; the training half is `CORPUS * TRAIN_RATE`, and that is what multiplies
by the schedule.

The notebook ships with the first row.

## The knobs

### `CORPUS` — how many images, before the train/val split

The one that decides what the run costs. `None` uses all 202,599 CelebA images.

`split()` permutes with a fixed seed, so the notebook takes a prefix of each half and gets a
reproducible random subsample rather than whatever sorted first in the archive.

**How large should it be?** This is the one place where copying SEBA is arguably the wrong call.
SEBA trains the de-anonymizer on *the experiment's own* clear/anon pair set — the same 20
identities it then attacks — so 340 images is enough for it, because the generator only has to
learn those faces. Here the model is trained on CelebA and applied to YouTube Faces, so it needs a
prior that generalises to identities it has never seen. More identities is straightforwardly
better for that; 400 images is about 20 identities, 5,000 is about 250.

Starting at 400 is still the right first move: it gets a complete run end to end in ten minutes and
tells you whether the pipeline works before you spend a night on it.

### `N_EPOCHS` / `N_EPOCHS_DECAY` — the learning-rate schedule

Upstream's `--n_epochs` and `--n_epochs_decay`. The rate is held at `2e-4` for the first and
decayed linearly across the second:

```
lr(epoch) = 2e-4 * (1 - max(0, epoch - N_EPOCHS) / (N_EPOCHS_DECAY + 1))
```

The `+ 1` is upstream's off-by-one, reproduced deliberately — it means the rate never quite reaches
zero. SEBA passes `epochs // 2` for both, so its `epochs: 100` is `50 + 50`. Upstream's own default
is `100 + 100`.

Keep them equal unless you have a reason not to. **Don't stop a run early** by passing `epochs=` to
`fit` — that leaves the learning rate part-way down its ramp, which is a different model than the
schedule describes. `fit` with no `epochs` runs the whole thing and is resumable, so there is no
need to.

### `BATCH_SIZE` — leave it at 1

Upstream's default, and it is load-bearing rather than arbitrary. With batch norm, a batch of one
normalises each image by its own statistics, which is most of the way to the instance
normalisation the pix2pix paper discusses — so raising it changes what the network computes, not
just how fast it computes it.

Raising it is a real speedup (batch 32 measured ~3.8× better GPU utilisation per image: 276 vs 72
images/s) and a real deviation. If you take it, say so, and expect the BatchNorm behaviour to
differ from anything SEBA reports.

It also works at 1 only because the one block that runs at 1×1 — the innermost encoder, where a
BatchNorm would raise on a single sample — is the one block pix2pix leaves unnormalised. That is
pinned by `test_a_batch_of_one_trains` in `src/tests/test_pix2pix_upstream.py`.

### `TRAIN_RATE` — the train/validation split

SEBA's, at 0.85. Note that upstream **never validates**: SEBA's `train_rate` writes a `val` folder
that `train.py` never opens. The validation pass here is an addition, and it changes nothing that
gets trained — but it does cost time, roughly one forward pass per held-out image per epoch. At
`CORPUS=None` that is 30,390 extra images every epoch, another 10–15% on top.

Lower it (more training data, smaller val) if you want that time back.

### `ANONYMIZER` — which method this model undoes

A key in `meles.anonymization.ANONYMIZERS`: `none`, `gaussianblur`, `blackbar`, `pixelation`. The
same registry `workflow/scripts/anonymize_frame.py` reads, so the model learns the inverse of the
anonymization the sweep actually produces.

**One model per method.** A generator that undoes 8×8 pixelation has learnt a different inverse
than one that undoes a Gaussian blur. Each gets its own checkpoint at
`models/pix2pix/{anonymizer}.pt`, each records which method it was trained against, and
`load_generator(path, anonymizer=...)` refuses a mismatch — pointing a pixelation model at blurred
frames produces plausible output and a meaningless score.

To train the others, change `ANONYMIZER` and re-run the notebook. `none` is only useful as a
sanity check: its pairs are identical, so it trains an identity function.

### `WORKERS` — dataloader processes

Not a model setting. At `batch_size=1` the GPU is the bottleneck, not the data, so more workers
past ~8 buys nothing. Set it to 0 if you need a traceback out of the dataset code.

### `load_size` / `flip` — the augmentation

`load_size=LOAD_SIZE` (286) is upstream's `resize_and_crop`: scale to a square 286, take a random
256 crop, mirror at random. It is pix2pix's entire augmentation, and the draw is made **once per
pair** and applied to both ends — cropping the two independently would break the pixel alignment
the L1 term depends on, which trains a blur rather than raising.

`load_size=None` turns it off. That is what the validation pairs use, so their loss is comparable
between epochs rather than measured on a fresh random crop each time.

### `ARCHIVE` — where the faces come from

A zip (`ImageArchive`, read in place, no extraction) or a directory (`ImageDirectory`, recursive).
Any corpus of face images works. The one hard requirement is that it is **not** YouTube Faces: an
attacker may train on whatever they like, but training on the dataset the experiment scores would
put the gallery into the generator's prior.

Worth stating as a limitation either way — CelebA and YouTube Faces are both celebrity datasets and
their identities are not known to be disjoint.

## Interrupting and resuming

`fit(checkpoint=checkpoint_path(ANONYMIZER))` writes after every epoch, and
`Pix2PixTrainer.resume(...)` picks a run back up where it stopped, including its position on the
learning-rate ramp. Re-running the notebook after an interruption continues rather than restarting
at epoch 0.

A checkpoint holds the generator, the discriminator, both optimisers, the schedule and the loss
history (~654 MiB). `trainer.save(path, inference_only=True)` drops everything but the generator
(~207 MiB), which is what the frame aggregator needs.

Changing `CORPUS`, `BATCH_SIZE` or the schedule mid-run and then resuming is legal but produces a
model no single configuration describes. Delete the checkpoint and start over instead.

## Reading the output

Plot `train_l1` / `val_l1`. The generator total (`train_g` = `adversarial + 100 * l1`) is what gets
backpropagated and is close to useless as a progress signal — it rises whenever the discriminator
pulls ahead, whether or not reconstruction improved. `fit` logs all four separately every epoch,
along with the current learning rate.

A discriminator loss collapsing towards zero while `train_adv` climbs means the discriminator has
won and the adversarial half has stopped contributing. The run does not fail — L1 keeps it
producing faces — and pix2pix offers no brake for it. SEBA runs it at its defaults, so introducing
one is a deviation to state, not a default to add.
