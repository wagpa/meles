"""
Tests for the weight cache.

Snakemake starts several jobs at once and they can all want the same model, so the interesting
property is not "it downloads a file" but "concurrent jobs never see a partial one".
"""

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from meles import weights


@pytest.fixture
def cache(tmp_path: Path, monkeypatch) -> Path:
    monkeypatch.setenv(weights.CACHE_ENV, str(tmp_path))
    return tmp_path


def payload(size: int = 4096) -> bytes:
    return bytes(index % 251 for index in range(size))


def local_spec(tmp_path: Path, data: bytes, digest: str | None = None) -> weights.WeightSpec:
    source = tmp_path / "source.bin"
    source.write_bytes(data)
    return weights.WeightSpec(file_name="model.onnx", url=source.as_uri(), sha256=digest)


def test_cache_dir_follows_the_environment(monkeypatch) -> None:
    monkeypatch.setenv(weights.CACHE_ENV, "/somewhere/else")
    assert weights.cache_dir() == Path("/somewhere/else")

    monkeypatch.delenv(weights.CACHE_ENV)
    assert weights.cache_dir() == Path(weights.DEFAULT_CACHE_DIR)


def test_downloads_and_verifies(cache: Path, tmp_path: Path) -> None:
    data = payload()
    spec = local_spec(tmp_path, data, hashlib.sha256(data).hexdigest())

    path = weights.resolve(spec)

    assert path == cache / "model.onnx"
    assert path.read_bytes() == data


def test_rejects_a_wrong_checksum(cache: Path, tmp_path: Path) -> None:
    spec = local_spec(tmp_path, payload(), "0" * 64)

    with pytest.raises(ValueError, match="Checksum mismatch"):
        weights.resolve(spec)

    # A failed download must leave nothing behind that a later call would trust.
    assert not (cache / "model.onnx").exists()
    assert not list(cache.glob("*.part"))


def test_is_a_no_op_once_cached(cache: Path, tmp_path: Path) -> None:
    spec = local_spec(tmp_path, payload())

    first = weights.resolve(spec)
    stamp = first.stat().st_mtime_ns
    (tmp_path / "source.bin").unlink()  # a second download would now fail

    assert weights.resolve(spec) == first
    assert first.stat().st_mtime_ns == stamp


def test_concurrent_resolution_yields_one_complete_file(cache: Path, tmp_path: Path) -> None:
    data = payload(1024 * 512)
    spec = local_spec(tmp_path, data, hashlib.sha256(data).hexdigest())

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: weights.resolve(spec), range(8)))

    assert {str(path) for path in results} == {str(cache / "model.onnx")}
    assert (cache / "model.onnx").read_bytes() == data
    assert not list(cache.glob("*.part")), "a partial download was left behind"


def test_cached_file_is_readable(cache: Path, tmp_path: Path) -> None:
    """
    The download goes through 'mkstemp', which creates owner-only files. A model cache is not a
    secret and MELES_WEIGHTS_DIR may point somewhere shared, so the mode is normalised.
    """
    weights.resolve(local_spec(tmp_path, payload()))

    mode = (cache / "model.onnx").stat().st_mode & 0o777
    assert mode & 0o400, "not readable by its owner"
    assert mode == 0o644 & ~weights._umask()


def test_extracts_a_member_from_an_archive(cache: Path, tmp_path: Path) -> None:
    import zipfile

    data = payload(2048)
    archive = tmp_path / "pack.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("other.onnx", b"wrong one")
        handle.writestr("wanted.onnx", data)

    spec = weights.WeightSpec(file_name="wanted.onnx", url=archive.as_uri(), archive_member="wanted.onnx")

    assert weights.resolve(spec).read_bytes() == data
    assert not list(cache.glob("*.part"))
