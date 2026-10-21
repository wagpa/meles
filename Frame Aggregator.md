# Frame Aggregator: Facial

The goal is to train a model that can aggregate anonymized frames into a single clear frame that represents the identity.
The resulting frame should be optimized to be recognized by arcface (or create an embedding of its own).

| Stage              | Description                                                                                      | Shape                   |
|--------------------|--------------------------------------------------------------------------------------------------|-------------------------|
| Input              | `N = 8` anonymized aligned crops of one identity, unordered, plus which anonymization they carry | `B x N x 3 x 112 x 112` |
| Target             | One clear frame per identity, picked by `07_target_selection` from a video the inputs exclude    | `B x 3 x 112 x 112`     |
| Encode (per image) | `iresnet34` with ArcFace weights, head dropped for the 7x7 map. Frozen, then fine-tuned at 1e-5  | `B*N x 512 x 7 x 7`     |
| Aggregate          | One transformer block, cross-attention + FFN, learned 7x7 query conditioned on the anonymization | `B x 512 x 7 x 7`       |
| Decode             | Four upsample + conv blocks, GroupNorm/SiLU, 512 -> 256 -> 192 -> 128 -> 64, `tanh` output       | `B x 3 x 112 x 112`     |
| Output             | A single aligned facial image to identify the person                                             | `B x 3 x 112 x 112`     |
| Loss               | `1.0 * L1 + 1.0 * identity + 0.2 * contrastive`, the last two under a frozen `iresnet34` critic  | per item                |
| Score              | Rank-1/rank-5 under ArcFace `w600k_r50`, the recogniser the experiment itself scores with        | per method              |

The target is a **single chosen frame**, not a mean embedding. Identity is `1 - cos` and
contrastive is InfoNCE over the batch; LPIPS and a symmetry term are not implemented, and the
contrastive term took their place — rank-1 is a *ranking* question, and a reconstruction can grow
more similar to its target while growing just as similar to everyone.

## The model

`meles.frame_aggregation.facial.Facial`. One item is a whole identity: several anonymized
observations in, one clear face out. `N` is free — the aggregator attends over whatever it is
given — but a batch has to stack them, so the dataset fixes it (`FRAMES = 8`).

```mermaid
flowchart LR
    IN["N anonymized crops<br/>B x N x 3 x 112 x 112<br/>RGB in [-1, 1]"] --> ENC
    ENC["FacialEncoder<br/>iresnet34, head removed<br/>frozen in phase 1"] --> FEAT["B x N x 512 x 7 x 7<br/>one feature map per observation"]
    FEAT --> AGG
    METHOD["method index<br/>B<br/>which anonymization"] -.->|"learned embedding,<br/>added to the query"| AGG
    AGG["FacialAggregator<br/>cross-attention + FFN<br/>learned 7x7 query pools N into 1"] --> POOL["B x 512 x 7 x 7"]
    POOL --> DEC["FacialDecoder<br/>4 upsample blocks<br/>512 → 256 → 192 → 128 → 64<br/>7 → 14 → 28 → 56 → 112"]
    DEC --> OUT["one reconstructed face<br/>B x 3 x 112 x 112<br/>tanh, so [-1, 1]"]
```

The encoder is shared across the `N` observations and the aggregator is permutation-invariant
except for the positional grid, so the frames are an unordered set, not a sequence.

## Training

`meles.training.facial`, driven by `src/notebooks/06_facial.ipynb`. One item per identity: the
observations come from the identity's *other* videos, the target is the one clear frame
`07_target_selection` picked, and both ends share one anonymization and one flip draw.

```mermaid
flowchart TB
    STRUCT["ytfaces_wide.parquet<br/>one row per frame"] --> FRAMES
    TARGETS["ytfaces_targets.parquet<br/>one clear frame per identity"] --> FRAMES
    SPLIT["split_identities<br/>hold out identities, never frames"] --> FRAMES

    FRAMES["FacialFrames<br/>draw N frames, detect on the clear frame,<br/>anonymize at native resolution, then align"]
    FRAMES --> ANON["N anonymized crops"]
    FRAMES --> CLEAR["clear target crop"]

    ANON --> MODEL["Facial"]
    MODEL --> RECON["reconstruction"]

    RECON --> CRIT
    CLEAR --> CRIT["WeightedCriterion<br/>1.0 L1 + 1.0 identity + 0.2 contrastive<br/>critic: frozen iresnet34"]
    CRIT -->|"backprop, per-item losses kept split by method"| MODEL

    RECON --> RANK["identification<br/>rank-1 / rank-5 under ArcFace w600k_r50<br/>never the critic — that would grade its own work"]
    RANK --> CKPT["checkpoint per epoch<br/>plus .best.pt on val:rank1"]
```

Two schedules run at once, and they are independent — `fit` is called **once per encoder phase**,
and walks the whole curriculum each time:

```mermaid
flowchart LR
    subgraph P1["phase 1: encoder frozen, lr 1e-4"]
        S0["stage 0<br/>none"] --> S1["stage 1<br/>+ gaussianblur"] --> S2["stage 2<br/>+ pixelation"] --> S3["stage 3<br/>+ blackbar"]
    end
    S3 --> P2
    subgraph P2["phase 2: encoder unfrozen, lr 1e-5"]
        S3B["stage 3 only<br/>the data is already as hard as it gets"]
    end
```

Each stage keeps everything before it and adds one method; a training item draws uniformly over
the unlocked prefix. The validation split is pinned to the **last** stage with `all_methods=True`
and `deterministic=True`, so every method is measured on the same identities and the same frames
from the first epoch onwards.

