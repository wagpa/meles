"""
Download and cache the ONNX model weights the detector and the embedders run on.

The weights are far too large to commit (the ArcFace archive alone is 275 MiB), so they are
fetched on first use and cached on disk. Snakemake starts several jobs at once and they may
all want the same file, so the download has to be safe under concurrency: a job must never
observe a half-written ``.onnx``. That is what the lock and the atomic rename below are for.
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

logger = logging.getLogger(__name__)

# The environment variable overriding where weights are cached.
CACHE_ENV = "MELES_WEIGHTS_DIR"

# The cache directory relative to the working directory, used when CACHE_ENV is unset. The
# workflow always runs from the repository root (like the 'data' and 'results' paths in
# workflow/config.yaml), so a relative default resolves predictably.
DEFAULT_CACHE_DIR = "models"

# How much of a download is held in memory at a time.
_CHUNK_SIZE = 1024 * 1024


@dataclass(frozen=True)
class WeightSpec:
    """
    A cached weight file: where it ends up, where it comes from, and how to check it.

    :param file_name: The name the file is cached under
    :param url: The URL to download it from
    :param sha256: The expected digest of the *download*, verified before the file is used
    :param archive_member: The member to extract, when the download is a zip archive
    """

    file_name: str
    url: str
    sha256: str | None = None
    archive_member: str | None = None


def _umask() -> int:
    """
    The current umask.

    There is no way to read it without setting it, so it is set back immediately. Harmless here
    because this runs while the download lock is held, before any file is created.

    :return: The umask
    """
    mask = os.umask(0o022)
    os.umask(mask)
    return mask


def cache_dir() -> Path:
    """
    The directory the weights are cached in.

    :return: The cache directory, which is not guaranteed to exist yet
    """
    return Path(os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR)


def _digest(path: Path) -> str:
    """
    The SHA-256 digest of a file, read incrementally so that a 275 MiB archive does not have
    to be held in memory.

    :param path: The file to digest
    :return: The digest as a lowercase hex string
    """
    hasher = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(_CHUNK_SIZE):
            hasher.update(chunk)
    return hasher.hexdigest()


def _verify(path: Path, sha256: str | None) -> None:
    """
    Check a downloaded file against its expected digest.

    :param path: The file to check
    :param sha256: The expected digest, or None to skip the check
    :raises ValueError: If the digest does not match
    """
    if sha256 is None:
        return
    actual = _digest(path)
    if actual != sha256:
        raise ValueError(
            f"Checksum mismatch for {path.name}: expected {sha256}, got {actual}. "
            "The download is corrupt or the upstream file changed."
        )


def _download(url: str, target: Path) -> None:
    """
    Stream a URL to a file.

    :param url: The URL to fetch
    :param target: The file to write to
    """
    logger.info("Downloading weights", extra={"url": url, "target": str(target)})
    with urllib.request.urlopen(url) as response, open(target, "wb") as handle:
        shutil.copyfileobj(response, handle, _CHUNK_SIZE)


def resolve(spec: WeightSpec) -> Path:
    """
    The path to a cached weight file, downloading it first if it is not there yet.

    Safe to call from several processes at once. The exclusive lock serialises the
    downloaders, the second in line finds the file already there, and the final rename is
    atomic, so a reader either sees no file or sees the complete one -- never a partial
    write.

    :param spec: The weight file to resolve
    :return: The path to the complete, verified file
    """
    target = cache_dir() / spec.file_name
    if target.is_file():
        return target

    target.parent.mkdir(parents=True, exist_ok=True)

    # The lock file is never deleted: unlinking it races with another process that has just
    # opened it and would hand two processes locks on different inodes.
    with open(target.with_name(target.name + ".lock"), "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)

        # Another process may have finished the download while this one waited for the lock.
        if target.is_file():
            return target

        handle, temporary = tempfile.mkstemp(dir=target.parent, suffix=".part")
        os.close(handle)
        temporary_path = Path(temporary)
        try:
            _download(spec.url, temporary_path)
            _verify(temporary_path, spec.sha256)

            if spec.archive_member is not None:
                extracted = temporary_path.with_suffix(".member")
                with zipfile.ZipFile(temporary_path) as archive:
                    with archive.open(spec.archive_member) as source:
                        with open(extracted, "wb") as destination:
                            shutil.copyfileobj(source, destination, _CHUNK_SIZE)
                temporary_path.unlink()
                temporary_path = extracted

            # 'mkstemp' creates owner-only files, which is right for a temporary but surprising
            # for a cache that MELES_WEIGHTS_DIR may point at a shared directory.
            os.chmod(temporary_path, 0o644 & ~_umask())

            # Atomic within a filesystem, so no reader ever sees a truncated model.
            os.replace(temporary_path, target)
        finally:
            temporary_path.unlink(missing_ok=True)

    return target
