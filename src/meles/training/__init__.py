"""
Training the learned de-anonymizations.

Everything else in ``meles`` is inference: a method is applied, never fitted. pix2pix is the first
attack that has to be trained first, so it gets its own package rather than growing a training
loop inside ``frame_aggregation``. The networks themselves stay next to the other frame
aggregators, in ``meles.frame_aggregation.pix2pix``; what lives here is the corpus of clear faces,
the pairing of clear and anonymized images, and the loop that consumes it.
"""

from meles.training.facial import (
    CURRICULUM,
    IDENTITY_WEIGHT,
    FacialFrames,
    FacialTrainer,
    IdentityLoss,
    WeightedCriterion,
    l1_per_item,
    load_facial,
    reconstruct,
    split_identities,
)
from meles.training.facial import Losses as FacialLosses
from meles.training.pairs import AnonymizedPairs, as_image, augment, augmentation_params
from meles.training.pix2pix import Losses, Pix2PixTrainer, checkpoint_path, default_device
from meles.training.sources import ImageArchive, ImageDirectory, ImageSource, Slice, split

__all__ = [
    "CURRICULUM",
    "IDENTITY_WEIGHT",
    "AnonymizedPairs",
    "FacialFrames",
    "FacialLosses",
    "FacialTrainer",
    "IdentityLoss",
    "ImageArchive",
    "ImageDirectory",
    "ImageSource",
    "Losses",
    "Pix2PixTrainer",
    "Slice",
    "WeightedCriterion",
    "as_image",
    "augment",
    "augmentation_params",
    "checkpoint_path",
    "default_device",
    "l1_per_item",
    "load_facial",
    "reconstruct",
    "split",
    "split_identities",
]
