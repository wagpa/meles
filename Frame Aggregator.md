# Frame Aggregator

The goal is to train a model that can aggregate anonymized frames into a single clear frame that represents the identity.
The resulting frame should be optimized to be recognized by arcface (or create an embedding of its own).

| Stage              | Description                                                       |
|--------------------|-------------------------------------------------------------------|
| Input              | anonymized images (unordered) and clear image with mean embedding |
| Encode (per image) | ResNet with ArcFace weights, last layer removed                   |
| Aggreagte          | Attention network over around 11 images                           |
| Decode             | Con upsampling network                                            |
| Output             | A single frontal facial image to identify the person              |
| Score              | Identity (ArcFace), LPIPS, L1, Symmetry?                          |

