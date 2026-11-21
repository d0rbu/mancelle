"""Disk-backed packed-row loading, cleanup, and eager-error parity."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
import torch as t

import steering.packed_batch_io as packed_io
from steering.overlay_io import save_overlay_shard
from steering.packed_batch_io import (
    PackedRowShard,
    load_packed_row_shard,
    open_packed_row_shard,
    save_packed_row_shard,
)


def _packed_shard(*, rows: int = 2, context_len: int = 8) -> PackedRowShard:
    segments_per_row = 2
    total_segments = rows * segments_per_row
    features_per_segment = 3
    total_features = total_segments * features_per_segment
    tokens = t.arange(rows * (context_len + 1), dtype=t.int64).reshape(
        rows, context_len + 1
    )
    return PackedRowShard(
        tokens=tokens.remainder(65_536).to(t.uint16),
        row_seg_offsets=t.arange(
            0,
            total_segments + 1,
            segments_per_row,
            dtype=t.int32,
        ),
        seg_token_start=t.tensor(
            [position for _ in range(rows) for position in (0, context_len // 2)],
            dtype=t.uint16,
        ),
        seg_token_end=t.tensor(
            [
                position
                for _ in range(rows)
                for position in (context_len // 2, context_len)
            ],
            dtype=t.uint16,
        ),
        seg_doc_id=t.arange(total_segments, dtype=t.int64).remainder(256).to(t.uint8),
        seg_feat_offsets=t.arange(
            0,
            total_features + 1,
            features_per_segment,
            dtype=t.int32,
        ),
        feature_ids=t.arange(total_features, dtype=t.int64)
        .remainder(65_536)
        .to(t.uint16),
        feature_pcts=t.arange(total_features, dtype=t.int64)
        .remainder(256)
        .to(t.uint8),
    )


def _write_packed(
    tmp_path: Path,
    *,
    rows: int = 2,
    context_len: int = 8,
) -> tuple[Path, PackedRowShard]:
    shard = _packed_shard(rows=rows, context_len=context_len)
    path = save_packed_row_shard(shard, tmp_path / "shard_00000")
    return path, shard


def _assert_shards_exact(
    actual: PackedRowShard,
    expected: PackedRowShard,
) -> None:
    for key in packed_io._TENSOR_KEYS:
        actual_tensor = getattr(actual, key)
        expected_tensor = getattr(expected, key)
        assert actual_tensor.dtype == expected_tensor.dtype, key
        assert actual_tensor.shape == expected_tensor.shape, key
        assert actual_tensor.stride() == expected_tensor.stride(), key
        assert actual_tensor.is_contiguous() == expected_tensor.is_contiguous(), key
        assert t.equal(actual_tensor, expected_tensor), key


def _capture_open_error(path: Path, spool_dir: Path | None) -> BaseException:
    try:
        with open_packed_row_shard(path, spool_dir=spool_dir):
            pass
    except BaseException as exc:
        return exc
    raise AssertionError("expected packed-row open to fail")


def _spool_files(spool_dir: Path) -> list[Path]:
    return sorted(spool_dir.glob("*.safetensors"))


def test_eager_context_is_exact_legacy_loader(tmp_path: Path) -> None:
    path, original = _write_packed(tmp_path)
    legacy = load_packed_row_shard(path)

    with open_packed_row_shard(path) as handle:
        assert handle.storage_mode == "eager"
        assert handle.backing_path is None
        assert handle.spool_bytes == 0
        assert handle.compressed_bytes == path.stat().st_size
        assert handle.tensor_bytes == packed_io._packed_tensor_bytes(original)
        assert handle.closed is False
        _assert_shards_exact(handle.shard, legacy)

    assert handle.closed is True
    _assert_shards_exact(handle.shard, legacy)


def test_disk_spool_is_tensor_exact_to_eager_loader(tmp_path: Path) -> None:
    path, _original = _write_packed(tmp_path, rows=17, context_len=32)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    legacy = load_packed_row_shard(path)

    with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
        assert handle.storage_mode == "disk_spool"
        assert handle.backing_path is not None
        assert handle.backing_path.parent == spool_dir
        assert handle.backing_path.is_file()
        assert handle.backing_path.stat().st_mode & 0o777 == 0o600
        assert handle.spool_bytes == handle.backing_path.stat().st_size
        assert handle.spool_bytes >= handle.tensor_bytes
        assert handle.tensor_bytes == packed_io._packed_tensor_bytes(legacy)
        assert handle.compressed_bytes == path.stat().st_size
        assert handle.closed is False
        _assert_shards_exact(handle.shard, legacy)

    assert handle.closed is True
    assert handle.backing_path is not None
    assert not handle.backing_path.exists()
    assert _spool_files(spool_dir) == []
    assert all(getattr(handle.shard, key).numel() == 0 for key in packed_io._TENSOR_KEYS)


@pytest.mark.skipif(not Path("/proc/self/maps").is_file(), reason="Linux maps required")
def test_disk_spool_mapping_exists_only_inside_context(tmp_path: Path) -> None:
    path, _original = _write_packed(tmp_path, rows=256, context_len=256)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()

    with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
        assert handle.backing_path is not None
        backing_text = str(handle.backing_path)
        # Force at least one page through every tensor mapping.
        for key in packed_io._TENSOR_KEYS:
            tensor = getattr(handle.shard, key)
            assert tensor.numel() > 0
            tensor.reshape(-1)[0].item()
        maps_during = Path("/proc/self/maps").read_text()
        assert backing_text in maps_during
        assert "(deleted)" not in next(
            line for line in maps_during.splitlines() if backing_text in line
        )
        # The API lifetime explicitly forbids retaining views past context exit.
        del tensor

    maps_after = Path("/proc/self/maps").read_text()
    assert backing_text not in maps_after
    assert not Path(backing_text).exists()


def test_disk_spool_follows_source_symlink_like_eager_loader(tmp_path: Path) -> None:
    target_dir = tmp_path / "target"
    target_dir.mkdir()
    target, _original = _write_packed(target_dir)
    link = tmp_path / target.name
    link.symlink_to(target)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()

    with open_packed_row_shard(link, spool_dir=spool_dir) as spooled:
        eager = load_packed_row_shard(link)
        _assert_shards_exact(spooled.shard, eager)
        assert spooled.compressed_bytes == target.stat().st_size


def test_absent_spool_directory_fails_without_creating_it(tmp_path: Path) -> None:
    path, _original = _write_packed(tmp_path)
    absent = tmp_path / "absent"

    error = _capture_open_error(path, absent)
    assert type(error) is NotADirectoryError
    assert str(error) == f"packed-row spool directory is absent: {absent}"
    assert not absent.exists()


def test_spool_path_that_is_a_file_fails_loud(tmp_path: Path) -> None:
    path, _original = _write_packed(tmp_path)
    not_dir = tmp_path / "not-a-directory"
    not_dir.write_text("file")

    error = _capture_open_error(path, not_dir)
    assert type(error) is NotADirectoryError
    assert str(error) == f"packed-row spool directory is absent: {not_dir}"
    assert not_dir.read_text() == "file"


@pytest.mark.parametrize("corruption", ["missing", "not-zstd", "truncated"])
def test_spooled_input_error_is_exact_eager_error(
    tmp_path: Path,
    corruption: str,
) -> None:
    path, _original = _write_packed(tmp_path)
    if corruption == "missing":
        path.unlink()
    elif corruption == "not-zstd":
        path.write_bytes(b"not a zstd stream")
    elif corruption == "truncated":
        payload = path.read_bytes()
        path.write_bytes(payload[: max(1, len(payload) // 2)])
    else:
        raise AssertionError(corruption)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()

    eager_error = _capture_open_error(path, None)
    spool_error = _capture_open_error(path, spool_dir)

    assert type(spool_error) is type(eager_error)
    assert str(spool_error) == str(eager_error)
    assert spool_error.args == eager_error.args
    assert _spool_files(spool_dir) == []


def test_spooled_missing_tensor_error_is_exact_eager_error(tmp_path: Path) -> None:
    path = save_overlay_shard(
        {"tokens": t.zeros((1, 2), dtype=t.uint16)},
        tmp_path / "shard_00000",
    )
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()

    eager_error = _capture_open_error(path, None)
    spool_error = _capture_open_error(path, spool_dir)

    assert type(spool_error) is type(eager_error) is KeyError
    assert str(spool_error) == str(eager_error)
    assert spool_error.args == eager_error.args
    assert _spool_files(spool_dir) == []


def test_wrong_extension_error_is_exact_eager_error(tmp_path: Path) -> None:
    path = tmp_path / "packed.bin"
    path.write_bytes(b"anything")
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()

    eager_error = _capture_open_error(path, None)
    spool_error = _capture_open_error(path, spool_dir)

    assert type(spool_error) is type(eager_error)
    assert str(spool_error) == str(eager_error)
    assert _spool_files(spool_dir) == []


def test_consumer_exception_is_not_misclassified_as_input_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _original = _write_packed(tmp_path)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    eager_calls = 0
    real_eager = packed_io.load_packed_row_shard

    def recording_eager(eager_path: Path) -> PackedRowShard:
        nonlocal eager_calls
        eager_calls += 1
        return real_eager(eager_path)

    monkeypatch.setattr(packed_io, "load_packed_row_shard", recording_eager)
    sentinel = RuntimeError("consumer sentinel")
    with pytest.raises(RuntimeError, match="consumer sentinel") as raised:
        with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
            assert handle.backing_path is not None
            raise sentinel

    assert raised.value is sentinel
    assert eager_calls == 0
    assert handle.closed is True
    assert handle.backing_path is not None
    assert not handle.backing_path.exists()
    assert _spool_files(spool_dir) == []


def test_spool_setup_resource_error_does_not_mask_with_eager_success(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _original = _write_packed(tmp_path)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    real_named_temporary_file = packed_io.tempfile.NamedTemporaryFile
    eager_calls = 0

    def forbidden_eager(_path: Path) -> PackedRowShard:
        nonlocal eager_calls
        eager_calls += 1
        raise AssertionError("resource failures must not trigger a multi-GiB eager retry")

    def fail_temp(*args, **kwargs):
        del args, kwargs
        raise OSError(28, "synthetic no space left")

    monkeypatch.setattr(packed_io.tempfile, "NamedTemporaryFile", fail_temp)
    monkeypatch.setattr(packed_io, "load_packed_row_shard", forbidden_eager)
    error = _capture_open_error(path, spool_dir)

    assert type(error) is OSError
    assert error.errno == 28
    assert "synthetic no space left" in str(error)
    assert eager_calls == 0
    assert _spool_files(spool_dir) == []
    monkeypatch.setattr(
        packed_io.tempfile,
        "NamedTemporaryFile",
        real_named_temporary_file,
    )


def test_spool_write_enospc_never_retries_the_eager_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _original = _write_packed(tmp_path, rows=31, context_len=64)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    eager_calls = 0
    real_named_temporary_file = packed_io.tempfile.NamedTemporaryFile

    class FailingOutput:
        def __init__(self, wrapped) -> None:
            self._wrapped = wrapped
            self.name = wrapped.name

        def __enter__(self):
            self._wrapped.__enter__()
            return self

        def __exit__(self, *args):
            return self._wrapped.__exit__(*args)

        def write(self, _payload: bytes) -> int:
            raise OSError(28, "synthetic spool write full")

        def flush(self) -> None:
            self._wrapped.flush()

    def failing_temp(*args, **kwargs):
        return FailingOutput(real_named_temporary_file(*args, **kwargs))

    def forbidden_eager(_path: Path) -> PackedRowShard:
        nonlocal eager_calls
        eager_calls += 1
        raise AssertionError("ENOSPC must not trigger eager packed loading")

    monkeypatch.setattr(packed_io.tempfile, "NamedTemporaryFile", failing_temp)
    monkeypatch.setattr(packed_io, "load_packed_row_shard", forbidden_eager)
    error = _capture_open_error(path, spool_dir)

    assert isinstance(error, OSError)
    assert error.errno == 28
    assert "synthetic spool write full" in str(error)
    assert eager_calls == 0
    assert _spool_files(spool_dir) == []


def test_spool_temp_is_cleaned_when_safe_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path, _original = _write_packed(tmp_path)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    real_safe_open = packed_io.safe_open

    def fail_safe_open(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("synthetic safe_open resource failure")

    monkeypatch.setattr(packed_io, "safe_open", fail_safe_open)
    error = _capture_open_error(path, spool_dir)

    assert type(error) is RuntimeError
    assert str(error) == "synthetic safe_open resource failure"
    assert _spool_files(spool_dir) == []
    monkeypatch.setattr(packed_io, "safe_open", real_safe_open)


def test_multiple_sequential_spools_leave_no_files_or_deleted_mappings(
    tmp_path: Path,
) -> None:
    path, _original = _write_packed(tmp_path, rows=128, context_len=128)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    backing_paths: list[str] = []

    for _ in range(10):
        with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
            assert handle.backing_path is not None
            backing_paths.append(str(handle.backing_path))
            assert int(handle.shard.feature_ids[0]) == 0
        assert _spool_files(spool_dir) == []

    assert len(set(backing_paths)) == len(backing_paths)
    if Path("/proc/self/maps").is_file():
        maps = Path("/proc/self/maps").read_text()
        assert not any(backing in maps for backing in backing_paths)


def test_spool_source_bytes_are_never_modified(tmp_path: Path) -> None:
    path, _original = _write_packed(tmp_path, rows=31, context_len=64)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    before = path.read_bytes()
    before_stat = path.stat()

    with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
        assert handle.shard.total_segments == 62

    after_stat = path.stat()
    assert path.read_bytes() == before
    assert after_stat.st_ino == before_stat.st_ino
    assert after_stat.st_size == before_stat.st_size
    assert after_stat.st_mtime_ns == before_stat.st_mtime_ns


def test_same_byte_source_replacement_is_read_transparently(tmp_path: Path) -> None:
    path, _original = _write_packed(tmp_path)
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()
    replacement = tmp_path / "replacement.zst"
    shutil.copyfile(path, replacement)
    replacement.replace(path)

    with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
        _assert_shards_exact(handle.shard, load_packed_row_shard(path))


@pytest.mark.requires_feature_data
def test_full_size_real_packed_shard_spools_exactly_when_configured(
    tmp_path: Path,
) -> None:
    raw_path = os.environ.get("SPLICE758_REAL_PACKED_SHARD")
    if not raw_path:
        pytest.skip("set SPLICE758_REAL_PACKED_SHARD to a production packed shard")
    path = Path(raw_path)
    if not path.is_file():
        pytest.fail(f"SPLICE758_REAL_PACKED_SHARD is not a file: {path}")
    spool_dir = tmp_path / "spool"
    spool_dir.mkdir()

    with open_packed_row_shard(path, spool_dir=spool_dir) as handle:
        assert handle.storage_mode == "disk_spool"
        assert handle.tensor_bytes >= 2 * 1024**3
        assert handle.spool_bytes >= handle.tensor_bytes
        assert handle.shard.tokens.dtype == t.uint16
        assert handle.shard.feature_ids.dtype == t.uint16
        assert handle.shard.feature_pcts.dtype == t.uint8
        assert handle.shard.total_segments > 100_000
        assert handle.shard.total_features > 100_000_000
        assert handle.backing_path is not None and handle.backing_path.is_file()
        eager = load_packed_row_shard(path)
        _assert_shards_exact(handle.shard, eager)

    assert _spool_files(spool_dir) == []
