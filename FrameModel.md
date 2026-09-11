The first step to a custom model will be a frame-based pix2pix deanonymization model. It takes an anonymized image and 
generates a de-anonymized variant from it.

```py
from torchvision.datasets import CelebA
dataset = CelebA(".data", split="train", download=True)
```

Train one model per anonymization method, reference in workflow.
