"""
The anonymizations the experiment attacks, keyed by the name ``workflow/config.yaml`` uses.

The registry lives here rather than only in ``workflow/scripts/anonymize_frame.py`` because a
second consumer appeared: ``meles.training`` builds the pix2pix training pairs by anonymizing
clear images, and it has to apply the *same* anonymization the pipeline applies. Two copies of
this dict would let a model be trained against, say, a different blur strength than the one it is
later asked to undo -- which does not fail, it just scores badly for a reason nothing reports.
"""

from meles.anonymization.anonymizer import (
    Anonymizer,
    BlackBarAnonymizer,
    GaussianBlurAnonymizer,
    NoAnonymizer,
    PixelationAnonymizer,
    facial_region,
)

# Every 'frame_anonymizer' an experiment leg in 'workflow/config.yaml' may name. Constructing an
# entry has to stay free of side effects: the whole dict is built at import time.
ANONYMIZERS: dict[str, Anonymizer] = {
    "none": NoAnonymizer(),
    "gaussianblur": GaussianBlurAnonymizer(),
    "blackbar": BlackBarAnonymizer(),
    "pixelation": PixelationAnonymizer(),
}


def resolve(method: str) -> Anonymizer:
    """
    The anonymizer a config key names.

    :param method: The key, e.g. 'pixelation'
    :return: The shared anonymizer instance
    :raises ValueError: If nothing is registered under that key
    """
    anonymizer = ANONYMIZERS.get(method)
    if anonymizer is None:
        raise ValueError(f"Unknown anonymization method {method!r}, expected one of {sorted(ANONYMIZERS)}")
    return anonymizer


__all__ = [
    "ANONYMIZERS",
    "Anonymizer",
    "BlackBarAnonymizer",
    "GaussianBlurAnonymizer",
    "NoAnonymizer",
    "PixelationAnonymizer",
    "facial_region",
    "resolve",
]
