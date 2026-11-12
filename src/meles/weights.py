"""
Download and cache the ONNX model weights the detector and the recognizers run on.

The weights are too large to commit, so they are fetched on first use and cached on disk. Several
workflow jobs may want the same file at the same time, so the download holds an exclusive lock and
finishes with an atomic rename: a reader either sees no file or sees the complete one.
"""

import fcntl
import hashlib
import logging
import os
import shutil
import tempfile
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from meles.paths import repository_root

logger = logging.getLogger(__name__)

# The environment variable overriding where weights are cached.
CACHE_ENV = "MELES_WEIGHTS_DIR"

# Where weights are cached when CACHE_ENV is unset, relative to the repository root.
DEFAULT_CACHE_DIR = "models"

# How much of a download is held in memory at a time.
CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class WeightSpec:
    """
    A cached weight file: where it ends up, where it comes from, and how to check it.

    :param file_name: The name the file is cached under
    :param url: The URL to download it from, or None when the file has no public download and
        has to be placed in the cache by hand
    :param sha256: The expected digest of the download, verified before the file is used
    :param archive_member: The member to extract, when the download is a zip archive
    :param source: Where the file comes from, shown when it is missing and cannot be fetched
    """

    file_name: str
    url: str | None = None
    sha256: str | None = None
    archive_member: str | None = None
    source: str | None = None


def cache_dir() -> Path:
    """
    The directory weights are cached in, which is not guaranteed to exist yet.

    Anchored to the repository root rather than the working directory, so that a notebook and a
    workflow job share one cache however they were started.

    :return: The cache directory, from ``MELES_WEIGHTS_DIR`` when it is set
    """
    override = os.environ.get(CACHE_ENV)
    return Path(override) if override else repository_root() / DEFAULT_CACHE_DIR


def digest(path: Path) -> str:
    """
    The SHA-256 digest of a file, read incrementally.

    :param path: The file to digest
    :return: The digest as a lowercase hex string
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def resolve(spec: WeightSpec) -> Path:
    """
    The path to a cached weight file, downloading it first if it is not there yet.

    :param spec: The weight file to resolve
    :return: The path to the complete, verified file
    :raises ValueError: If the download does not match its expected digest
    :raises FileNotFoundError: If the file has no download and is not in the cache
    """
    target = cache_dir() / spec.file_name
    if target.is_file():
        return target

    if spec.url is None:
        raise FileNotFoundError(
            f"{spec.file_name} is not in {cache_dir()} and cannot be downloaded. "
            + (f"Get it from {spec.source} and put it there." if spec.source else "Put it there by hand.")
        )

    target.parent.mkdir(parents=True, exist_ok=True)

    # The lock file is never deleted: unlinking it races with a process that has just opened it,
    # which would hand two processes locks on different inodes.
    with open(target.with_name(target.name + ".lock"), "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        # Another process may have finished the download while this one waited for the lock.
        if target.is_file():
            return target

        handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".part")
        os.close(handle)
        temporary_path = Path(temporary)
        try:
            logger.info("Downloading weights", extra={"url": spec.url, "target": str(target)})
            with urllib.request.urlopen(spec.url) as response, open(temporary_path, "wb") as output:
                shutil.copyfileobj(response, output, CHUNK_SIZE)

            if spec.sha256 is not None and (actual := digest(temporary_path)) != spec.sha256:
                raise ValueError(
                    f"Checksum mismatch for {spec.file_name}: expected {spec.sha256}, got {actual}. "
                    "The download is corrupt or the upstream file changed."
                )

            if spec.archive_member is not None:
                extracted = temporary_path.with_suffix(".member")
                with zipfile.ZipFile(temporary_path) as archive, archive.open(spec.archive_member) as source:
                    with open(extracted, "wb") as output:
                        shutil.copyfileobj(source, output, CHUNK_SIZE)
                temporary_path.unlink()
                temporary_path = extracted

            # mkstemp creates owner-only files, which is surprising for a cache that
            # MELES_WEIGHTS_DIR may point at a shared directory.
            os.chmod(temporary_path, 0o644)

            # Atomic within a filesystem, so no reader ever sees a truncated model.
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

    return target
