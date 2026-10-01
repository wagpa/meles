The first step to a custom model will be a frame-based pix2pix deanonymization model. It takes an anonymized image and 
generates a de-anonymized variant from it.

Train one model per anonymization method, reference in workflow.

## What "the model SEBA uses" means

SEBA does not implement pix2pix. `src/deanonymization/face/pix2pix.py` clones
[junyanz/pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)
and shells out to it:

```
train.py --model pix2pix --direction BtoA --n_epochs E/2 --n_epochs_decay E/2
test.py  --model test --netG unet_256 --norm batch --direction BtoA
```

Everything not listed there is an upstream default, so "SEBA's model" is `--model pix2pix`'s
defaults: `unet_256` generator, `basic` (70×70 PatchGAN) discriminator, `--norm batch`,
`--gan_mode vanilla` (BCE-with-logits), `lambda_L1=100`, Adam at `lr=2e-4` / `beta1=0.5`,
`batch_size=1`, `resize_and_crop` from 286 to 256 with random horizontal flip, and a learning rate
held for `n_epochs` then decayed linearly across `n_epochs_decay`. `BtoA` is what makes it a
de-anonymization: SEBA puts clear images in `A` and anonymized in `B`, so B is the input.

The networks here are a **port, not a reimplementation**. Module nesting and `state_dict` keys
match upstream exactly — verified against the real thing: same 82/22 tensors, same keys, same
shapes, and bit-identical forward output when handed the same weights. `src/tests/test_pix2pix_upstream.py`
pins that with a digest, so a later "cleanup" of the block structure fails loudly. The practical
consequence: `import_upstream_generator()` loads anyone's `latest_net_G.pth` from that repository
— SEBA's, had they published it — with no conversion.

What is deliberately *not* copied is the harness: no visdom, no AB-folder dataset built by
`combine_A_and_B.py`, no subprocess. The pairs are made in memory from
`meles.anonymization.ANONYMIZERS` instead, which is the point of the exercise.

Two documented deviations, both stated in the docstrings:

- **Inference does not random-crop.** SEBA passes no `--preprocess` to `test.py`, so upstream
  takes a random 256 crop of a 286 resize at test time too. `deanonymize` resizes straight to 256:
  losing a random 10% border off every frame is an oversight rather than part of the method, and
  it would make the attack's framing non-deterministic.
- **The resize uses `cv2.INTER_CUBIC`, not PIL's BICUBIC.** Same kernel family, not bit-identical
  (PIL antialiases on downscale). It is a training augmentation; the policy is what matters.

One difference worth a decision rather than a footnote: SEBA trains the de-anonymizer on the
*experiment's own* clear/anon pair set (`train(clear_set, anon_set)`), for 20 selected identities.
This trains on a separate corpus (CelebA) and applies the result to YouTube Faces. That is a
stronger attacker model, not a weaker one, but it is not what SEBA measures.

## Where it lives

- `src/meles/frame_aggregation/pix2pix.py` — the ported networks (`UnetGenerator`,
  `NLayerDiscriminator`), the objective, the tensor/frame conversion, and the checkpoint format.
- `src/meles/training/sources.py` — where the clear faces come from: `ImageArchive` (reads a zip
  in place), `ImageDirectory`, and a seeded `split`.
- `src/meles/training/pairs.py` — `AnonymizedPairs`, which turns a clear-face dataset into
  (anonymized, clear) training pairs by applying one of `meles.anonymization.ANONYMIZERS` to each
  image's detected facial area. That registry is shared with
  `workflow/scripts/anonymize_frame.py`, so the model is trained against exactly the anonymization
  the sweep produces.
- `src/meles/training/pix2pix.py` — `Pix2PixTrainer`: model, optimisers, the linear
  learning-rate schedule, loss history, and `save`/`load`/`resume`.
- `src/tests/test_pix2pix_upstream.py` — pins the architecture and hyperparameters to upstream.
- `src/notebooks/01_train_pix2pix.ipynb` — the run; `Pix2PixTraining.md` is how to configure it.

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
from meles.frame_aggregation.pix2pix import LOAD_SIZE
from meles.training import AnonymizedPairs, ImageArchive, Pix2PixTrainer, checkpoint_path, split

train, validation = split(ImageArchive("data/celeba/img_align_celeba.zip"), validation=0.15)
pairs = AnonymizedPairs(train, "pixelation", load_size=LOAD_SIZE)   # 286 -> random 256 crop + flip
pairs.detect()                                  # ~2 min over CelebA on 8 threads, not once per epoch

loader = DataLoader(pairs, batch_size=1, shuffle=True, num_workers=8)
trainer = Pix2PixTrainer.resume("pixelation")   # continues from a checkpoint if there is one
trainer.fit(loader, checkpoint=checkpoint_path("pixelation"))       # runs out the 100+100 schedule
```

`batch_size=1` is upstream's default, and it works because the one block running at 1×1 — the
innermost encoder — is the one pix2pix leaves unnormalised. Passing no `epochs` to `fit` runs the
full `n_epochs + n_epochs_decay` schedule; stopping short leaves the learning rate part-way down
its ramp, which is a different model than the schedule describes.

Build the validation pairs with `load_size=None`. Upstream never evaluates (SEBA's `train_rate`
writes a `val` folder `train.py` never opens), so the validation pass here is an addition, and a
loss measured under a fresh random crop each epoch is not comparable between epochs.

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

## Reading the loss curves

Plot `l1`. The generator total (`train_g` = `adversarial + 100 * l1`) is what gets backpropagated
and is close to useless as a quality signal: it rises whenever the discriminator pulls ahead,
whether or not reconstruction improved. In the first 50-epoch pixelation run `train_g` went
5.66 → 10.99, which read as divergence but was not — at epoch 50 the total decomposed as **6.53
adversarial (62%) + 3.99 from L1**, and the L1 term had fallen from ≈4.8 to 3.99. The
discriminator had simply won: `train_d` 0.61 → 0.04, logits +6.9 on real against −6.5 on
generated, i.e. perfect separation and no useful gradient left for the generator.

The library logs `train_l1`, `train_adv` and `train_d` separately every epoch so this is legible
without decomposing anything by hand. No knobs are offered to rebalance it: SEBA runs pix2pix at
its defaults, and a discriminator that pulls ahead is what those defaults do on this data. If the
comparison ever calls for a brake, that is a deviation to introduce and state, not a default.

## Result of the first run (pixelation, 50 epochs, CelebA)

> Measured **before** the port to upstream's architecture, at batch size 32, without the
> `resize_and_crop` augmentation and without the learning-rate decay. Kept for reference; the
> numbers do not describe the current model, and that checkpoint no longer loads.

Measured on 300 held-out images, ArcFace cosine between each frame's embedding and the embedding
of its own clear image:

| condition                   | mean cosine |
|-----------------------------|-------------|
| anonymized (do nothing)     | 0.061       |
| `frame_bicubic` (SEBA)      | 0.068       |
| **pix2pix**                 | **0.229**   |
| different person (floor)    | 0.011       |

So the learned attack recovers roughly 3.7x the identity signal of the resample attack, but stays
well under a per-frame match — only 3.3% of frames clear a 0.4 cosine. That is the interesting
shape for this project: partial recovery per frame is exactly the regime where the aggregation
question has something to decide.

Reconstruction error tells the other half of the story:

| L1 vs the clear image | whole frame | inside the face box |
|-----------------------|-------------|---------------------|
| anonymized            | 0.0390      | 0.1499              |
| pix2pix               | 0.0398      | 0.1132              |

The model improves the face and **degrades everything else** — whole-frame L1 is worse than doing
nothing, because the generator repaints background that was already pixel-identical to the target.
Visible in the samples as speckle and checkerboard artefacts from the transposed convolutions.

## Still to do

- **Composite instead of replacing the whole frame.** The anonymization only touched the facial
  area, so everything outside it is already correct in the input; keeping those pixels and pasting
  in only the generated face would remove the degradation above for free. The open question is
  where the box comes from at inference — detecting in the *anonymized* frame works for pixelation
  but is exactly what fails for `blackbar`.
- Train the remaining methods: one checkpoint each for `gaussianblur` and `blackbar`.

Wire it into the sweep: a `FrameAggregator` subclass over `generator.deanonymize`, registered in
`AGGREGATORS` in `workflow/scripts/aggregate_frame.py` as a `frame_*` method (it is frame-wise —
one frame in, one frame out — like the resample de-anonymizations). Open question there is how the
rule picks the checkpoint: the registry key would have to name the anonymization the model undoes,
and it should be checked against the `frame_anonymization` wildcard of the branch it runs in.
