"""
Locating the repository, so that a path does not depend on the working directory a notebook or a
script happened to be started from.
"""

import os
from collections.abc import Iterable
from pathlib import Path


def repository_root(start: Path | None = None) -> Path:
    """
    The repository root: the nearest directory at or above ``start`` holding a ``pyproject.toml``.

    Called rather than exposed as a module constant on purpose. Resolving it at import time would
    make importing :mod:`meles` fail wherever the repository is not on disk, which is exactly the
    case an installed copy of the library is in.

    :param start: Where to start looking; defaults to the current working directory
    :return: The repository root
    :raises FileNotFoundError: If no ``pyproject.toml`` is found at or above ``start``
    """
    start = Path.cwd() if start is None else Path(start)
    for path in [start, *start.parents]:
        if (path / "pyproject.toml").is_file():
            return path
    raise FileNotFoundError(f"No pyproject.toml at or above {start}; is this a checkout of the repository?")


def rebase(paths: Iterable[str], source_dir: Path, target_dir: Path) -> list[str]:
    """
    Express paths that are relative to one directory as paths relative to another.

    A structure's ``path`` column is relative to the parquet that holds it, so moving a structure to a
    new directory means rewriting it. The result may walk up with ``..``, which ``Path.relative_to``
    cannot express.

    :param paths: Paths relative to ``source_dir``
    :param source_dir: The directory the paths are currently relative to
    :param target_dir: The directory the paths should be relative to
    :return: The same files, relative to ``target_dir``
    """
    return [os.path.relpath(source_dir / path, target_dir) for path in paths]
