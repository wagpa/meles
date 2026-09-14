"""
Where the clear face images for training come from.

``torchvision.datasets.CelebA`` is deliberately not used, and the reason is worth writing down
because the failure it produces is badly misleading. Its ``_check_integrity`` demands five
*space-separated* ``.txt`` annotation files, each matching an MD5 of the original Google Drive
upload, and one of them (``identity_CelebA.txt``) is not in the widely-used Kaggle mirror at all --
that mirror ships ``.csv`` files instead. When any of them is missing the class raises
``Dataset not found or corrupted``, which reads as "your download is broken" but is really
"your annotations are not the exact bytes I expected". No file location fixes it, and the
annotations cannot be reconstructed byte-identically from the CSVs.

None of that metadata is needed here. A pix2pix de-anonymizer is trained on (anonymized, clear)
pairs made from the images alone -- no attributes, no identities, no landmarks -- so these sources
read the images and nothing else. That also makes the training corpus swappable: a directory of
FFHQ or LFW crops works exactly as well as CelebA, which matters because the one hard requirement
is that it *not* be the dataset the experiment scores.
"""

import logging
import os
import zipfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)

# The image files a source picks up. Matched case-insensitively.
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp")


class ImageSource(Protocol):
    """
    What :class:`meles.training.pairs.AnonymizedPairs` needs from the dataset it wraps.

    Deliberately just "indexable, with a length", which is what every torchvision image dataset
    is. Items are **RGB**, either a PIL image or an ``(H, W, 3)`` array, optionally wrapped in an
    ``(image, target)`` pair as torchvision yields -- the target is discarded.
    """

    def __len__(self) -> int: ...

    def __getitem__(self, index: int) -> Any: ...


def _decode(data: bytes, name: str) -> NDArray[np.uint8]:
    """
    Decode image bytes to RGB.

    :param data: The encoded file contents
    :param name: What to name in the error, if it does not decode
    :return: The image, ``(H, W, 3)`` uint8 RGB
    :raises ValueError: If the bytes are not a decodable image
    """
    decoded = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if decoded is None:
        raise ValueError(f"Could not decode {name} as an image")
    # 'IMREAD_COLOR' gives BGR; the sources hand out RGB, as PIL and torchvision do, and it is
    # 'meles.training.pairs.as_image' that converts back to the BGR the rest of meles speaks.
    # The 'astype' is a no-op on what imdecode returns and is here only because OpenCV's stubs
    # cannot express "same dtype as the input", the same reason 'resample.py' carries one.
    return cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB).astype(np.uint8, copy=False)


class ImageDirectory:
    """
    Every image under a directory, in sorted order.

    Sorted so that an index means the same file on every machine and across runs, which is what
    makes a split by index reproducible.
    """

    def __init__(self, root: str | Path, suffixes: Sequence[str] = IMAGE_SUFFIXES, recursive: bool = True):
        """
        :param root: The directory to walk
        :param suffixes: The file extensions to pick up, matched case-insensitively
        :param recursive: Whether to descend into subdirectories. On by default, because the
            CelebA mirrors disagree about how deeply they nest their images
        :raises FileNotFoundError: If the directory does not exist or holds no images
        """
        self.root = Path(root)
        if not self.root.is_dir():
            raise FileNotFoundError(f"{self.root} is not a directory")

        wanted = {suffix.lower() for suffix in suffixes}
        walked = self.root.rglob("*") if recursive else self.root.glob("*")
        self.paths = sorted(path for path in walked if path.suffix.lower() in wanted and path.is_file())

        if not self.paths:
            raise FileNotFoundError(f"No {'/'.join(sorted(wanted))} files under {self.root}")
        logger.info("Read image directory", extra={"root": str(self.root), "len": len(self.paths)})

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        path = self.paths[index]
        return _decode(path.read_bytes(), str(path))


class ImageArchive:
    """
    Every image inside a zip, read without extracting it.

    Saves unpacking 1.4 GiB of CelebA onto disk a second time, and sidesteps the question of how
    deeply a given mirror nested its images -- the members are matched by suffix wherever they sit,
    so ``img_align_celeba/*.jpg`` and ``img_align_celeba/img_align_celeba/*.jpg`` both just work.

    The open archive handle is **not** shared across processes. A ``ZipFile`` keeps a single file
    position, so two ``DataLoader`` workers reading through an inherited handle would interleave
    their seeks and hand back torn data -- which decodes as a corrupt image, sometimes. The handle
    is therefore dropped on pickling and reopened per process.
    """

    def __init__(self, path: str | Path, suffixes: Sequence[str] = IMAGE_SUFFIXES):
        """
        :param path: The '.zip' to read
        :param suffixes: The member extensions to pick up, matched case-insensitively
        :raises FileNotFoundError: If the archive holds no images
        """
        self.path = Path(path)
        wanted = {suffix.lower() for suffix in suffixes}

        with zipfile.ZipFile(self.path) as archive:
            self.names = sorted(
                info.filename
                for info in archive.infolist()
                if not info.is_dir() and os.path.splitext(info.filename)[1].lower() in wanted
            )

        if not self.names:
            raise FileNotFoundError(f"No {'/'.join(sorted(wanted))} members in {self.path}")

        self._archive: zipfile.ZipFile | None = None
        self._pid: int | None = None
        logger.info("Read image archive", extra={"path": str(self.path), "len": len(self.names)})

    @property
    def archive(self) -> zipfile.ZipFile:
        """
        This process' archive handle, opened on first use.

        :return: The open ``ZipFile``
        """
        # A forked DataLoader worker inherits the parent's handle rather than a pickled copy, so
        # the pid is what catches that case; '__setstate__' covers the spawned one.
        if self._archive is None or self._pid != os.getpid():
            self._archive = zipfile.ZipFile(self.path)
            self._pid = os.getpid()
        return self._archive

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, index: int) -> NDArray[np.uint8]:
        name = self.names[index]
        return _decode(self.archive.read(name), f"{self.path}:{name}")

    def __getstate__(self) -> dict[str, Any]:
        return self.__dict__ | {"_archive": None, "_pid": None}

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)


class Slice:
    """
    A contiguous view of a source, which is how the training and validation halves are named.

    A view rather than a copy: an :class:`ImageArchive` scanned its own member list once, and
    slicing should not make it do that again.
    """

    def __init__(self, source: ImageSource, indices: Sequence[int]):
        """
        :param source: The source to view
        :param indices: The indices of it to expose, in the order to expose them
        """
        self.source = source
        self.indices = list(indices)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Any:
        return self.source[self.indices[index]]


def split(source: ImageSource, validation: float = 0.02, seed: int = 0) -> tuple[Slice, Slice]:
    """
    Split a source into a training and a validation half.

    Shuffled with a fixed seed rather than sliced off the end, because the sources sort by file
    name and CelebA's file names are ordered -- a trailing slice would hold whatever the last few
    thousand images happen to have in common.

    This is not CelebA's official ``list_eval_partition`` split, and does not need to be. The
    validation half only keeps the reported loss honest; it is not a benchmark number, and the
    identities on either side of it are not the ones the experiment scores either way.

    :param source: The source to split
    :param validation: The fraction to hold out. Small by default: 2% of CelebA is still 4,000
        images, and every one held out is one the generator does not learn its prior from
    :param seed: The seed of the permutation, fixed so that a rerun holds out the same images
    :return: The training and the validation half
    """
    order = np.random.default_rng(seed).permutation(len(source))
    held_out = int(round(len(source) * validation))
    return Slice(source, order[held_out:].tolist()), Slice(source, order[:held_out].tolist())
