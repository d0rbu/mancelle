from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
import stat
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import Mock

import hf_xet
import pytest
import torch as t
import torch.distributed as dist
from loguru import logger

import steering.remote_training as remote_module
from nanochat.checkpoint_manager import load_checkpoint, save_checkpoint
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
)
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    build_clustered_selection_shard_integrity,
    parse_clustered_selection_shard_inventory,
)
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    REMOTE_RUN_META_KEY,
    HfBucketBackend,
    RemoteCheckpointPublisher,
    RemoteCheckpointStore,
    RemoteMaterializationCancelled,
    RemoteObjectInfo,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
    assert_checkpoint_remote_run,
    build_checkpoint_marker,
    build_remote_loader_certificate,
    build_remote_run_fingerprint,
    ddp_collective_result,
    ddp_create_checkpoint_publisher,
    ddp_drain_checkpoint_publisher,
    ddp_poll_checkpoint_publisher,
    ddp_publish_saved_checkpoint,
    ddp_rank0_action,
    download_verified_object,
    parse_checkpoint_marker,
    parse_remote_loader_certificate,
    parse_verified_upload_claim_bytes,
    upload_verified_object,
    validate_local_verified_upload_claim,
    validate_remote_training_eligibility,
    verified_local_object_info,
)


class FakeBucket:
    def __init__(self, root: Path):
        self.root = root
        self.upload_order: list[str] = []
        self.download_failures: dict[str, int] = {}
        self.upload_failures: dict[str, int] = {}

    def _path(self, remote_path: str) -> Path:
        path = self.root / remote_path
        assert path.resolve().is_relative_to(self.root.resolve())
        return path

    @staticmethod
    def _info(path: Path, remote_path: str) -> RemoteObjectInfo:
        result = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(remote_path, path.stat().st_size, str(result.hash))

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        return self._info(path, remote_path) if path.is_file() else None

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        base = self._path(prefix)
        if not base.exists():
            return []
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        failures = self.download_failures.get(remote_path, 0)
        if failures:
            self.download_failures[remote_path] = failures - 1
            raise OSError("injected transient download failure")
        source = self._path(remote_path)
        if not source.is_file():
            raise FileNotFoundError(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        failures = self.upload_failures.get(remote_path, 0)
        if failures:
            self.upload_failures[remote_path] = failures - 1
            raise OSError("injected transient upload failure")
        target = self._path(remote_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(local_path, target)
        self.upload_order.append(remote_path)


def _stat_with_drift(observed: os.stat_result, **changes: int) -> SimpleNamespace:
    values = {
        name: getattr(observed, name)
        for name in (
            "st_dev",
            "st_ino",
            "st_mode",
            "st_size",
            "st_mtime_ns",
            "st_ctime_ns",
            "st_nlink",
        )
    }
    values.update(changes)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "drifted",
    [
        ("mtime_ns",),
        ("ctime_ns",),
        ("mtime_ns", "ctime_ns"),
        ("size",),
        ("size", "mtime_ns"),
        ("size", "mtime_ns", "ctime_ns"),
    ],
)
@pytest.mark.parametrize("allow_advisory_drift", [True, False])
def test_identity_drift_advisory_matrix(
    drifted: tuple[str, ...],
    allow_advisory_drift: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """size/mtime_ns/ctime_ns are one advisory class: tolerated (with a warning
    naming every field) only when the caller opted in, fatal otherwise."""
    before: remote_module._StableIdentity = (11, 22, 33, 44, 55, 66, 77)
    after_values = list(before)
    for name in drifted:
        after_values[remote_module._STABLE_IDENTITY_FIELDS.index(name)] -= 10
    after = cast(remote_module._StableIdentity, tuple(after_values))

    assert remote_module._identity_drift(before, after) == list(drifted)
    if allow_advisory_drift:
        remote_module._check_identity_drift(
            before,
            after,
            path=Path("artifact"),
            message="artifact changed",
            allow_advisory_drift=True,
        )
        assert len(caplog.records) == 1
        warning = caplog.records[0].message
        assert f"drifted_fields={','.join(drifted)}" in warning
        for name in drifted:
            index = remote_module._STABLE_IDENTITY_FIELDS.index(name)
            assert f"{name}(expected={before[index]}, observed={after[index]})" in warning
    else:
        with pytest.raises(RemoteTrainingError) as exc_info:
            remote_module._check_identity_drift(
                before,
                after,
                path=Path("artifact"),
                message="artifact changed",
                allow_advisory_drift=False,
            )
        assert not caplog.records
        error = str(exc_info.value)
        for name in drifted:
            index = remote_module._STABLE_IDENTITY_FIELDS.index(name)
            assert f"{name}(expected={before[index]}, observed={after[index]})" in error


@pytest.mark.parametrize("structural", ["dev", "ino", "mode", "nlink"])
@pytest.mark.parametrize("with_advisory_companions", [False, True])
@pytest.mark.parametrize("allow_advisory_drift", [True, False])
def test_identity_drift_structural_fields_always_fatal(
    structural: str,
    with_advisory_companions: bool,
    allow_advisory_drift: bool,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """dev/ino/mode/nlink drift (a file swap) is fatal regardless of the
    advisory opt-in, alone or mixed with advisory drift."""
    before: remote_module._StableIdentity = (11, 22, 33, 44, 55, 66, 77)
    after_values = list(before)
    after_values[remote_module._STABLE_IDENTITY_FIELDS.index(structural)] += 100
    if with_advisory_companions:
        for name in ("size", "mtime_ns"):
            after_values[remote_module._STABLE_IDENTITY_FIELDS.index(name)] -= 10
    after = cast(remote_module._StableIdentity, tuple(after_values))

    with pytest.raises(RemoteTrainingError, match=structural) as exc_info:
        remote_module._check_identity_drift(
            before,
            after,
            path=Path("artifact"),
            message="artifact changed",
            allow_advisory_drift=allow_advisory_drift,
        )
    assert not caplog.records
    index = remote_module._STABLE_IDENTITY_FIELDS.index(structural)
    assert f"{structural}(expected={before[index]}, observed={after[index]})" in str(exc_info.value)


_ADVISORY_STAT_DRIFTS = {
    "timestamps": {"st_mtime_ns": 1, "st_ctime_ns": 2},
    "size": {"st_size": -5},
}


def _inject_lstat_drift(
    monkeypatch: pytest.MonkeyPatch,
    *,
    calls: set[int],
    changes: dict[str, int],
) -> None:
    """Return drifted stats (relative field deltas) on the given 1-based
    Path.lstat call numbers."""
    real_lstat = Path.lstat
    seen = 0

    def drifting(candidate: Path) -> SimpleNamespace | os.stat_result:
        nonlocal seen
        observed = real_lstat(candidate)
        seen += 1
        if seen in calls:
            return _stat_with_drift(
                observed,
                **{name: getattr(observed, name) + delta for name, delta in changes.items()},
            )
        return observed

    monkeypatch.setattr(Path, "lstat", drifting)


def _count_hashes(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    real_hash_files = cast(Any, remote_module.hf_xet).hash_files
    counter = [0]

    def counting(paths: list[str]):
        counter[0] += 1
        return real_hash_files(paths)

    monkeypatch.setattr(cast(Any, remote_module.hf_xet), "hash_files", counting)
    return counter


@pytest.mark.parametrize("kind", sorted(_ADVISORY_STAT_DRIFTS))
def test_stable_file_identity_tolerates_advisory_baseline_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: str,
) -> None:
    """Advisory drift at the pass-0 baseline re-baselines against the open
    descriptor (which carries the authoritative size) in a single pass."""
    path = tmp_path / "artifact"
    path.write_bytes(b"stable bytes")
    _inject_lstat_drift(monkeypatch, calls={1}, changes=_ADVISORY_STAT_DRIFTS[kind])
    hashes = _count_hashes(monkeypatch)

    size, mode, sha256, xet_hash = remote_module._stable_file_identity(
        path,
        allow_advisory_drift=True,
    )

    assert size == len(b"stable bytes")
    assert stat.S_ISREG(mode)
    assert sha256 == hashlib.sha256(b"stable bytes").hexdigest()
    assert xet_hash
    assert hashes[0] == 1
    assert len(caplog.records) == 1
    assert str(path) in caplog.records[0].message


@pytest.mark.parametrize("kind", sorted(_ADVISORY_STAT_DRIFTS))
def test_stable_file_identity_reverifies_after_post_xet_advisory_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    kind: str,
) -> None:
    """Tolerated drift at the post-xet stat forces the strict full re-read
    pass, which returns the true content hashes."""
    path = tmp_path / "artifact"
    path.write_bytes(b"stable bytes")
    _inject_lstat_drift(monkeypatch, calls={3}, changes=_ADVISORY_STAT_DRIFTS[kind])
    hashes = _count_hashes(monkeypatch)

    size, _, sha256, xet_hash = remote_module._stable_file_identity(
        path,
        allow_advisory_drift=True,
    )

    assert size == len(b"stable bytes")
    assert sha256 == hashlib.sha256(b"stable bytes").hexdigest()
    assert xet_hash
    assert hashes[0] == 2
    assert any(
        "identity drift requires content re-verification" in record.message
        for record in caplog.records
    )


@pytest.mark.parametrize("kind", sorted(_ADVISORY_STAT_DRIFTS))
def test_stable_file_identity_rejects_advisory_drift_on_both_passes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """The re-verification pass is fully strict: drift that persists into
    pass 2 is fatal, not silently re-tolerated."""
    path = tmp_path / "artifact"
    path.write_bytes(b"stable bytes")
    _inject_lstat_drift(monkeypatch, calls={3, 6}, changes=_ADVISORY_STAT_DRIFTS[kind])

    with pytest.raises(RemoteTrainingError, match="drifted_fields=") as exc_info:
        remote_module._stable_file_identity(path, allow_advisory_drift=True)
    assert "artifact changed while computing its Xet hash" in str(exc_info.value)


@pytest.mark.parametrize("kind", sorted(_ADVISORY_STAT_DRIFTS))
def test_stable_file_identity_default_rejects_advisory_drift_first_occurrence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    """Callers without the download-path opt-in (training checkpoints,
    _verify_file) keep first-occurrence-fatal semantics."""
    path = tmp_path / "artifact"
    path.write_bytes(b"stable bytes")
    _inject_lstat_drift(monkeypatch, calls={3}, changes=_ADVISORY_STAT_DRIFTS[kind])
    hashes = _count_hashes(monkeypatch)

    with pytest.raises(RemoteTrainingError, match="drifted_fields="):
        remote_module._stable_file_identity(path)
    assert hashes[0] == 1


def test_verify_file_default_rejects_injected_timestamp_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"stable bytes")
    _inject_lstat_drift(monkeypatch, calls={1}, changes={"st_mtime_ns": 1, "st_ctime_ns": 2})

    with pytest.raises(RemoteTrainingError) as exc_info:
        remote_module._verify_file(
            path,
            byte_length=len(b"stable bytes"),
            content_sha256=hashlib.sha256(b"stable bytes").hexdigest(),
        )
    error = str(exc_info.value)
    assert "artifact changed before hashing" in error
    assert "mtime_ns" in error
    assert "ctime_ns" in error
    assert "expected=" in error
    assert "observed=" in error


def test_download_reverification_rejects_same_size_post_hash_rewrite(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    original_bytes = b"before-content"
    rewritten_bytes = b"after--content"
    assert len(original_bytes) == len(rewritten_bytes)
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(original_bytes)
    replacement = tmp_path / "replacement"
    replacement.write_bytes(rewritten_bytes)
    real_hash_files = cast(Any, remote_module.hf_xet).hash_files
    replacement_xet = str(real_hash_files([str(replacement)])[0].hash)
    hash_calls = 0

    def rewrite_after_first_xet(paths: list[str]):
        nonlocal hash_calls
        hash_calls += 1
        result = real_hash_files(paths)
        if hash_calls == 1:
            hashed_path = Path(paths[0])
            before = hashed_path.lstat()
            hashed_path.write_bytes(rewritten_bytes)
            os.utime(
                hashed_path,
                ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000),
            )
        return result

    monkeypatch.setattr(
        cast(Any, remote_module.hf_xet),
        "hash_files",
        rewrite_after_first_xet,
    )
    expected = RemoteObjectInfo(remote_path, len(rewritten_bytes), replacement_xet)

    with pytest.raises(RemoteTrainingError, match="has SHA-256"):
        remote_module._verified_download_with_retry(
            bucket,
            remote_path=remote_path,
            final_path=tmp_path / "cache" / "body",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
            expected=expected,
            expected_content_sha256=hashlib.sha256(original_bytes).hexdigest(),
            retry_verification_errors=False,
            verified_publication_callback=lambda _stat: None,
        )
    assert hash_calls == 2


@pytest.mark.parametrize("stale_lstat_call", [1, 5, 6])
def test_verified_download_tolerates_transient_stale_size_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    stale_lstat_call: int,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    payload = b"stable body bytes"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    real_hash_files = cast(Any, remote_module.hf_xet).hash_files
    expected_xet = str(real_hash_files([str(source)])[0].hash)
    real_lstat = Path.lstat
    lstat_calls = 0

    def stale_once(candidate: Path) -> SimpleNamespace | os.stat_result:
        nonlocal lstat_calls
        observed = real_lstat(candidate)
        lstat_calls += 1
        if lstat_calls == stale_lstat_call:
            return _stat_with_drift(observed, st_size=observed.st_size - 5)
        return observed

    monkeypatch.setattr(Path, "lstat", stale_once)
    final_path = tmp_path / "cache" / "body"

    info = remote_module._verified_download_with_retry(
        bucket,
        remote_path=remote_path,
        final_path=final_path,
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
        expected=RemoteObjectInfo(remote_path, len(payload), expected_xet),
        retry_verification_errors=False,
        verified_publication_callback=lambda _stat: None,
    )

    assert info.byte_length == len(payload)
    assert final_path.read_bytes() == payload
    assert any("identity drift tolerated" in record.message for record in caplog.records)


def test_verified_download_rejects_real_truncation_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    payload = b"stable body bytes that will be truncated mid-verification"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(payload)
    real_hash_files = cast(Any, remote_module.hf_xet).hash_files
    expected_xet = str(real_hash_files([str(source)])[0].hash)
    hash_calls = 0

    def truncate_after_first_xet(paths: list[str]):
        nonlocal hash_calls
        hash_calls += 1
        result = real_hash_files(paths)
        if hash_calls == 1:
            Path(paths[0]).write_bytes(payload[: len(payload) // 2])
        return result

    monkeypatch.setattr(
        cast(Any, remote_module.hf_xet),
        "hash_files",
        truncate_after_first_xet,
    )
    final_path = tmp_path / "cache" / "body"

    with pytest.raises(
        RemoteTrainingError,
        match="disagrees with remote size/Xet identity",
    ):
        remote_module._verified_download_with_retry(
            bucket,
            remote_path=remote_path,
            final_path=final_path,
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
            expected=RemoteObjectInfo(remote_path, len(payload), expected_xet),
            retry_verification_errors=False,
            verified_publication_callback=lambda _stat: None,
        )
    assert hash_calls == 2
    assert not final_path.exists()


@pytest.mark.parametrize("allow_advisory_drift", [True, False])
def test_stable_file_identity_rejects_content_mutation_during_hashing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    allow_advisory_drift: bool,
) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"before")
    real_hash_files = cast(Any, remote_module.hf_xet).hash_files

    def mutate_then_hash(paths: list[str]):
        path.write_bytes(b"after with a different size")
        return real_hash_files(paths)

    monkeypatch.setattr(cast(Any, remote_module.hf_xet), "hash_files", mutate_then_hash)

    with pytest.raises(RemoteTrainingError) as exc_info:
        remote_module._stable_file_identity(
            path,
            allow_advisory_drift=allow_advisory_drift,
        )
    error = str(exc_info.value)
    assert "size" in error.lower()


def test_stable_file_identity_rejects_short_sha256_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"stable bytes")
    real_open = Path.open

    class _ShortReader:
        def __init__(self, handle) -> None:
            self._handle = handle

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, traceback) -> None:
            self._handle.close()

        def fileno(self) -> int:
            return self._handle.fileno()

        def read(self, _size: int = -1) -> bytes:
            return b""

    def _short_open(candidate: Path, *args, **kwargs):
        handle = real_open(candidate, *args, **kwargs)
        if candidate == path and args and args[0] == "rb":
            return _ShortReader(handle)
        return handle

    monkeypatch.setattr(Path, "open", _short_open)
    with pytest.raises(RemoteTrainingError, match="expected=12, hashed=0"):
        remote_module._stable_file_identity(path)


def test_download_verified_object_promotes_exact_bytes_and_retries(tmp_path: Path) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified body")
    expected = bucket.stat(remote_path)
    assert expected is not None
    bucket.download_failures[remote_path] = 1
    final = tmp_path / "cache" / "body"
    phase_timings: dict[str, float] = {}

    observed = download_verified_object(
        bucket,
        expected=expected,
        final_path=final,
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
        phase_timings=phase_timings,
    )

    assert observed == expected
    assert final.read_bytes() == b"verified body"
    assert not list(final.parent.glob(".*.remote-tmp.*"))
    assert set(phase_timings) == {
        "download_seconds",
        "sha256_seconds",
        "xet_seconds",
    }
    assert phase_timings["download_seconds"] > 0
    assert phase_timings["xet_seconds"] > 0
    assert phase_timings["sha256_seconds"] == 0.0
    assert remote_module._stable_file_identity(final) == remote_module._stable_file_identity(
        final,
        phase_timings={},
    )


def test_download_tolerates_tmp_mtime_drift_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The verified_inode re-check spans the whole hash pass; mtime-only drift
    on the owner-private tmp file must be tolerated (structural drift stays fatal)."""
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified body")
    expected = bucket.stat(remote_path)
    assert expected is not None
    final = tmp_path / "cache" / "body"
    real_identity = remote_module._stable_file_identity

    def drift_mtime_then_identity(path: Path, **kwargs: Any):
        result = real_identity(path, **kwargs)
        observed = path.lstat()
        os.utime(path, ns=(observed.st_atime_ns, observed.st_mtime_ns + 5_000_000))
        return result

    monkeypatch.setattr(remote_module, "_stable_file_identity", drift_mtime_then_identity)
    published: list[os.stat_result] = []

    observed_info = download_verified_object(
        bucket,
        expected=expected,
        final_path=final,
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
        verified_publication_callback=published.append,
    )

    assert observed_info == expected
    assert final.read_bytes() == b"verified body"
    assert len(published) == 1
    assert any(
        "artifact identity drift tolerated" in record.message
        and "drifted_fields=" in record.message
        for record in caplog.records
    )


def test_download_rejects_tmp_replacement_after_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Structural identity drift (inode swap) after the hash pass stays fatal."""
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"verified body")
    expected = bucket.stat(remote_path)
    assert expected is not None
    final = tmp_path / "cache" / "body"
    real_identity = remote_module._stable_file_identity

    def replace_then_identity(path: Path, **kwargs: Any):
        result = real_identity(path, **kwargs)
        # Allocate the replacement while the original inode is still live so
        # the filesystem cannot hand the new file the just-freed inode.
        replacement = path.with_name(path.name + ".swap")
        replacement.write_bytes(path.read_bytes())
        os.replace(replacement, path)
        return result

    monkeypatch.setattr(remote_module, "_stable_file_identity", replace_then_identity)

    with pytest.raises(RemoteTrainingError, match="changed after verification"):
        download_verified_object(
            bucket,
            expected=expected,
            final_path=final,
            timeout_seconds=0.05,
            retry_interval_seconds=0.001,
            verified_publication_callback=lambda _stat: None,
        )


@pytest.mark.parametrize("field", ["byte_length", "xet_hash"])
def test_download_verified_object_rejects_snapshotted_identity_mismatch(
    tmp_path: Path,
    field: str,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"body")
    actual = bucket.stat(remote_path)
    assert actual is not None
    expected = (
        RemoteObjectInfo(actual.path, actual.byte_length + 1, actual.xet_hash)
        if field == "byte_length"
        else RemoteObjectInfo(actual.path, actual.byte_length, "0" * 64)
    )
    final = tmp_path / "cache" / "body"

    with pytest.raises(RemoteTrainingError, match="size/Xet identity"):
        download_verified_object(
            bucket,
            expected=expected,
            final_path=final,
            timeout_seconds=0.0,
            retry_interval_seconds=0.0,
        )
    assert not final.exists()


def test_download_verified_object_rejects_mutation_after_listing(tmp_path: Path) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"before")
    expected = bucket.stat(remote_path)
    assert expected is not None
    source.write_bytes(b"after mutation")

    with pytest.raises(RemoteTrainingError, match="size/Xet identity"):
        download_verified_object(
            bucket,
            expected=expected,
            final_path=tmp_path / "cache" / "body",
            timeout_seconds=0.0,
            retry_interval_seconds=0.0,
        )


def test_transport_download_skips_sha256_but_xet_still_rejects_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"before")
    expected = bucket.stat(remote_path)
    assert expected is not None
    source.write_bytes(b"after!")
    monkeypatch.setattr(
        remote_module.hashlib,
        "sha256",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("transport download computed SHA-256")
        ),
    )
    probe = tmp_path / "probe"
    probe.write_bytes(b"probe")
    timings: dict[str, float] = {}
    size, _, sha256, xet_hash = remote_module._stable_file_identity(
        probe,
        need_sha256=False,
        phase_timings=timings,
    )
    assert (size, sha256) == (len(b"probe"), "")
    assert xet_hash
    assert timings["sha256_seconds"] == 0.0

    with pytest.raises(RemoteTrainingError, match="size/Xet identity"):
        download_verified_object(
            bucket,
            expected=expected,
            final_path=tmp_path / "cache" / "body",
            timeout_seconds=0.0,
            retry_interval_seconds=0.0,
        )


@pytest.mark.parametrize("expected_digest", ["correct", "wrong"])
def test_checkpoint_style_download_keeps_sha256_verification(
    tmp_path: Path,
    expected_digest: str,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "checkpoints/model.pt"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"checkpoint")
    expected = bucket.stat(remote_path)
    assert expected is not None
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if expected_digest == "wrong":
        digest = "0" * 64

    def call():
        return remote_module._verified_download_with_retry(
            bucket,
            remote_path=remote_path,
            final_path=tmp_path / "cache" / "model.pt",
            timeout_seconds=0.0,
            retry_interval_seconds=0.0,
            expected=expected,
            expected_content_sha256=digest,
            retry_verification_errors=False,
        )

    if expected_digest == "wrong":
        with pytest.raises(RemoteTrainingError, match="has SHA-256"):
            call()
    else:
        assert call() == expected


def test_download_verified_object_fsync_failure_never_promotes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    source = bucket._path(remote_path)
    source.parent.mkdir(parents=True)
    source.write_bytes(b"body")
    expected = bucket.stat(remote_path)
    assert expected is not None
    final = tmp_path / "cache" / "body"
    monkeypatch.setattr(
        remote_module.os,
        "fsync",
        lambda _fd: (_ for _ in ()).throw(OSError("injected fsync failure")),
    )

    with pytest.raises(RemoteTrainingError, match="injected fsync failure"):
        download_verified_object(
            bucket,
            expected=expected,
            final_path=final,
            timeout_seconds=0.0,
            retry_interval_seconds=0.0,
        )
    assert not final.exists()


def test_download_verified_object_cancels_retry_wait_immediately(tmp_path: Path) -> None:
    class AlwaysFailDownloadBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.attempted = threading.Event()

        def download(self, remote_path: str, local_path: Path) -> None:
            self.attempted.set()
            raise OSError("injected transient failure")

    bucket = AlwaysFailDownloadBucket(tmp_path / "bucket")
    source = bucket._path("features/body")
    source.parent.mkdir(parents=True)
    source.write_bytes(b"body")
    expected = bucket.stat("features/body")
    assert expected is not None
    cancellation = threading.Event()
    failures: list[BaseException] = []

    def target() -> None:
        try:
            download_verified_object(
                bucket,
                expected=expected,
                final_path=tmp_path / "cache" / "body",
                timeout_seconds=30.0,
                retry_interval_seconds=30.0,
                cancellation_event=cancellation,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    thread = threading.Thread(target=target)
    thread.start()
    assert bucket.attempted.wait(timeout=5.0)
    cancellation.set()
    thread.join(timeout=1.0)

    assert not thread.is_alive()
    assert len(failures) == 1
    assert isinstance(failures[0], RemoteTrainingError)
    assert "download cancelled" in str(failures[0])
    assert not (tmp_path / "cache" / "body").exists()


def test_upload_verified_object_success_retry_and_idempotence(tmp_path: Path) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"body")
    remote_path = "features/body"
    bucket.upload_failures[remote_path] = 1

    first = upload_verified_object(
        bucket,
        local_path=local,
        remote_path=remote_path,
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )
    order_after_first = list(bucket.upload_order)
    second = upload_verified_object(
        bucket,
        local_path=local,
        remote_path=remote_path,
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )

    assert first == second == bucket.stat(remote_path)
    assert bucket.upload_order == order_after_first
    assert bucket.upload_order[-1] == remote_path
    assert len(bucket.upload_order) == 2
    assert remote_module.VERIFIED_UPLOAD_CLAIMS_DIRNAME in bucket.upload_order[0]
    claim_path = bucket._path(bucket.upload_order[0])
    assert json.loads(claim_path.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "remote_path": remote_path,
        "byte_length": first.byte_length,
        "xet_hash": first.xet_hash,
    }


def test_upload_verified_object_retries_final_identity_stat(tmp_path: Path) -> None:
    class FinalStatRetryBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.canonical_uploaded = False
            self.post_upload_stats = 0

        def upload(self, local_path: Path, remote_path: str) -> None:
            super().upload(local_path, remote_path)
            if remote_path == "features/body":
                self.canonical_uploaded = True

        def stat(self, remote_path: str) -> RemoteObjectInfo | None:
            if remote_path == "features/body" and self.canonical_uploaded:
                self.post_upload_stats += 1
                if self.post_upload_stats == 2:
                    raise OSError("injected transient final stat failure")
            return super().stat(remote_path)

    bucket = FinalStatRetryBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"body")

    uploaded = upload_verified_object(
        bucket,
        local_path=local,
        remote_path="features/body",
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )

    assert uploaded == bucket.stat("features/body")
    assert bucket.post_upload_stats >= 3


def test_upload_verified_object_recovers_from_claim_only_interruption(
    tmp_path: Path,
) -> None:
    class FailCanonicalOnceBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.failed = False

        def upload(self, local_path: Path, remote_path: str) -> None:
            if remote_path == "features/body" and not self.failed:
                self.failed = True
                raise OSError("injected interruption after claim")
            super().upload(local_path, remote_path)

    bucket = FailCanonicalOnceBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"body")

    with pytest.raises(RemoteTrainingError, match="timed out"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path="features/body",
            timeout_seconds=0.0,
            retry_interval_seconds=0.0,
        )
    claim_paths = [info.path for info in bucket.list("features/_verified_upload_claims/body")]
    assert len(claim_paths) == 1
    assert bucket.stat("features/body") is None

    uploaded = upload_verified_object(
        bucket,
        local_path=local,
        remote_path="features/body",
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )

    assert uploaded == bucket.stat("features/body")
    assert [
        info.path for info in bucket.list("features/_verified_upload_claims/body")
    ] == claim_paths


def _claim_bytes(info: RemoteObjectInfo) -> bytes:
    return (
        json.dumps(
            {
                "schema_version": 1,
                "remote_path": info.path,
                "byte_length": info.byte_length,
                "xet_hash": info.xet_hash,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode()


def test_verified_upload_claim_parser_and_local_validator(tmp_path: Path) -> None:
    expected = RemoteObjectInfo("features/body", 4, "a" * 64)
    payload = _claim_bytes(expected)
    assert (
        parse_verified_upload_claim_bytes(
            payload,
            expected_path=expected.path,
            expected_xet_hash=expected.xet_hash,
        )
        == expected
    )

    claim = tmp_path / "_verified_upload_claims" / "body" / f"{expected.xet_hash}.json"
    claim.parent.mkdir(parents=True)
    claim.write_bytes(payload)
    validate_local_verified_upload_claim(claim, expected=expected)
    with pytest.raises(RemoteTrainingError, match="claim path"):
        validate_local_verified_upload_claim(
            claim.with_name("wrong.json"),
            expected=expected,
        )


@pytest.mark.parametrize(
    "payload",
    [
        b"not-json",
        b'{"schema_version":1}\n',
        b'{"byte_length":4,"remote_path":"../body","schema_version":1,'
        b'"xet_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}\n',
        b'{"byte_length":4,"remote_path":"features/body","schema_version":1,"xet_hash":"bad"}\n',
        b'{"byte_length":4,"remote_path":"features/body","schema_version":1,'
        b'"xet_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"}',
        b'{"byte_length":4,"byte_length":4,"remote_path":"features/body",'
        b'"schema_version":1,"xet_hash":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'
        b'aaaaaaaaaaaaaaaaaaaaaaaa"}\n',
    ],
)
def test_verified_upload_claim_parser_rejects_malformed_bytes(payload: bytes) -> None:
    with pytest.raises(RemoteTrainingError):
        parse_verified_upload_claim_bytes(
            payload,
            expected_path="features/body",
            expected_xet_hash="a" * 64,
        )


def test_upload_verified_object_binds_immutable_claim_reopen_to_verified_inode(
    tmp_path: Path,
) -> None:
    class ReplaceClaimPathBucket(FakeBucket):
        replaced = False

        def upload(self, local_path: Path, remote_path: str) -> None:
            if (
                remote_module.VERIFIED_UPLOAD_CLAIMS_DIRNAME in PurePosixPath(remote_path).parts
                and not self.replaced
            ):
                self.replaced = True
                replacement = tmp_path / "claim-replacement"
                replacement.write_bytes(b"attacker claim")
                source_path = Path(os.readlink(local_path))
                replacement.replace(source_path)
            super().upload(local_path, remote_path)

    bucket = ReplaceClaimPathBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"body")
    with pytest.raises(RemoteTrainingError, match="changed during publication"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path="features/body",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
        )

    claim_infos = bucket.list("features/_verified_upload_claims/body")
    assert len(claim_infos) == 1
    claim_path = bucket._path(claim_infos[0].path)
    claim = json.loads(claim_path.read_text())
    assert claim["remote_path"] == "features/body"
    assert claim["byte_length"] == len(b"body")
    assert bucket.stat("features/body") is None

    local.write_bytes(b"body")
    assert upload_verified_object(
        bucket,
        local_path=local,
        remote_path="features/body",
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    ) == bucket.stat("features/body")


def test_upload_verified_object_allows_concurrent_identical_writers(
    tmp_path: Path,
) -> None:
    class IdenticalClaimBarrierBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.claim_barrier = threading.Barrier(2)

        def stat(self, remote_path: str) -> RemoteObjectInfo | None:
            if (
                remote_module.VERIFIED_UPLOAD_CLAIMS_DIRNAME in PurePosixPath(remote_path).parts
                and not self._path(remote_path).exists()
            ):
                self.claim_barrier.wait(timeout=5.0)
            return super().stat(remote_path)

    bucket = IdenticalClaimBarrierBucket(tmp_path / "bucket")
    local_a = tmp_path / "a"
    local_b = tmp_path / "b"
    local_a.write_bytes(b"identical")
    local_b.write_bytes(b"identical")
    results: list[RemoteObjectInfo] = []
    failures: list[BaseException] = []

    def publish(local_path: Path) -> None:
        try:
            results.append(
                upload_verified_object(
                    bucket,
                    local_path=local_path,
                    remote_path="features/body",
                    timeout_seconds=1.0,
                    retry_interval_seconds=0.001,
                )
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [threading.Thread(target=publish, args=(path,)) for path in (local_a, local_b)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10.0)
        assert not thread.is_alive()

    assert failures == []
    assert len(results) == 2
    assert results[0] == results[1] == bucket.stat("features/body")
    assert bucket._path("features/body").read_bytes() == b"identical"


def test_upload_verified_object_rejects_change_after_local_claim(tmp_path: Path) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"claimed")
    remote_path = "features/body"
    claimed = verified_local_object_info(
        local_path=local,
        remote_path=remote_path,
    )
    local.write_bytes(b"changed")

    with pytest.raises(RemoteTrainingError, match="changed after publication claim"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path=remote_path,
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
            expected_local=claimed,
        )

    assert bucket.upload_order == []

    local.write_bytes(b"deterministic")
    with pytest.raises(RemoteTrainingError, match="deterministic producer bytes"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path="features/deterministic",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
            expected_bytes=b"different",
        )
    assert bucket.stat("features/deterministic") is None


def test_upload_verified_object_binds_backend_reopen_to_verified_inode(
    tmp_path: Path,
) -> None:
    local = tmp_path / "body"
    local.write_bytes(b"verified original")

    class ReplaceCallerPathBucket(FakeBucket):
        replaced = False

        def upload(self, local_path: Path, remote_path: str) -> None:
            if remote_path == "features/body" and not self.replaced:
                self.replaced = True
                replacement = local.with_name("replacement")
                replacement.write_bytes(b"attacker replacement")
                replacement.replace(local)
            super().upload(local_path, remote_path)

    bucket = ReplaceCallerPathBucket(tmp_path / "bucket")
    with pytest.raises(RemoteTrainingError, match="changed during publication"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path="features/body",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
        )

    assert bucket._path("features/body").read_bytes() == b"verified original"
    local.write_bytes(b"verified original")
    assert upload_verified_object(
        bucket,
        local_path=local,
        remote_path="features/body",
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    ) == bucket.stat("features/body")


def test_upload_verified_object_rejects_every_bound_source_identity_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    local = tmp_path / "body"
    local.write_bytes(b"verified original")
    bucket = FakeBucket(tmp_path / "bucket")

    with monkeypatch.context() as scoped:
        scoped.delattr(remote_module.os, "O_NOFOLLOW")
        with pytest.raises(RemoteTrainingError, match="requires os.O_NOFOLLOW"):
            upload_verified_object(
                bucket,
                local_path=local,
                remote_path="features/no-nofollow",
                timeout_seconds=1.0,
                retry_interval_seconds=0.001,
            )

    hardlink = local.with_name("hardlink")
    hardlink.hardlink_to(local)
    try:
        with pytest.raises(RemoteTrainingError, match="single-link regular file"):
            upload_verified_object(
                bucket,
                local_path=local,
                remote_path="features/hardlink",
                timeout_seconds=1.0,
                retry_interval_seconds=0.001,
            )
    finally:
        hardlink.unlink()

    real_lstat = Path.lstat
    replaced_while_opening = False

    def replace_before_link_check(path: Path):
        nonlocal replaced_while_opening
        if path == local and not replaced_while_opening:
            replaced_while_opening = True
            replacement = local.with_name("opening-replacement")
            replacement.write_bytes(b"replacement")
            replacement.replace(local)
        return real_lstat(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(Path, "lstat", replace_before_link_check)
        with pytest.raises(RemoteTrainingError, match="changed while opening"):
            upload_verified_object(
                bucket,
                local_path=local,
                remote_path="features/open-race",
                timeout_seconds=1.0,
                retry_interval_seconds=0.001,
            )

    local.write_bytes(b"verified original")
    real_hash_files = cast(Any, hf_xet).hash_files

    def mutate_after_hash(paths):
        result = real_hash_files(paths)
        local.write_bytes(b"changed after hash")
        return result

    with monkeypatch.context() as scoped:
        scoped.setattr(hf_xet, "hash_files", mutate_after_hash)
        with pytest.raises(RemoteTrainingError, match="changed while hashing"):
            upload_verified_object(
                bucket,
                local_path=local,
                remote_path="features/hash-race",
                timeout_seconds=1.0,
                retry_interval_seconds=0.001,
            )

    local.write_bytes(b"verified original")

    class UnlinkCallerPathBucket(FakeBucket):
        def upload(self, local_path: Path, remote_path: str) -> None:
            if remote_path == "features/unlink-race":
                local.unlink()
            super().upload(local_path, remote_path)

    unlink_bucket = UnlinkCallerPathBucket(tmp_path / "unlink-bucket")
    with pytest.raises(RemoteTrainingError, match="changed during publication"):
        upload_verified_object(
            unlink_bucket,
            local_path=local,
            remote_path="features/unlink-race",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
        )
    assert unlink_bucket._path("features/unlink-race").read_bytes() == b"verified original"


def test_upload_verified_object_rejects_collision_and_non_regular_input(
    tmp_path: Path,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    remote_path = "features/body"
    prior = bucket._path(remote_path)
    prior.parent.mkdir(parents=True)
    prior.write_bytes(b"prior")
    local = tmp_path / "body"
    local.write_bytes(b"different")
    with pytest.raises(RemoteTrainingError, match="collision"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path=remote_path,
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
        )

    symlink = tmp_path / "symlink"
    symlink.symlink_to(local)
    with pytest.raises(RemoteTrainingError, match="direct regular file"):
        upload_verified_object(
            bucket,
            local_path=symlink,
            remote_path="features/symlink",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
        )


def test_upload_verified_object_retries_invisible_upload_and_times_out(
    tmp_path: Path,
) -> None:
    class InvisibleUploadBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.attempts = 0

        def upload(self, local_path: Path, remote_path: str) -> None:
            self.attempts += 1

    bucket = InvisibleUploadBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"body")
    with pytest.raises(RemoteTrainingError, match="timed out"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path="features/body",
            timeout_seconds=0.01,
            retry_interval_seconds=0.001,
        )
    assert bucket.attempts > 1
    assert bucket.stat("features/body") is None


def test_upload_verified_object_treats_visible_wrong_identity_as_terminal(
    tmp_path: Path,
) -> None:
    class WrongIdentityBucket(FakeBucket):
        def __init__(self, root: Path):
            super().__init__(root)
            self.attempts = 0

        def upload(self, local_path: Path, remote_path: str) -> None:
            self.attempts += 1
            super().upload(local_path, remote_path)

        def stat(self, remote_path: str) -> RemoteObjectInfo | None:
            info = super().stat(remote_path)
            if info is None:
                return None
            return RemoteObjectInfo(info.path, info.byte_length + 1, info.xet_hash)

    bucket = WrongIdentityBucket(tmp_path / "bucket")
    local = tmp_path / "body"
    local.write_bytes(b"body")
    with pytest.raises(RemoteTrainingError, match="failed size/Xet verification"):
        upload_verified_object(
            bucket,
            local_path=local,
            remote_path="features/body",
            timeout_seconds=1.0,
            retry_interval_seconds=0.001,
        )
    assert bucket.attempts == 1


class PrefixGuardedBucket:
    """Confine every bucket operation to one fresh test-owned prefix."""

    def __init__(self, backend: Any, *, allowed_prefix: str) -> None:
        if (
            not allowed_prefix
            or allowed_prefix.startswith("/")
            or allowed_prefix.endswith("/")
            or ".." in Path(allowed_prefix).parts
        ):
            raise RemoteTrainingError(f"unsafe live-test prefix {allowed_prefix!r}")
        self.backend = backend
        self.allowed_prefix = allowed_prefix
        self.successful_uploads: list[str] = []

    def _assert_allowed(self, remote_path: str) -> None:
        parts = PurePosixPath(remote_path).parts
        if (
            not remote_path
            or remote_path.startswith("/")
            or "\\" in remote_path
            or any(part in {"", ".", ".."} for part in remote_path.split("/"))
            or PurePosixPath(*parts).as_posix() != remote_path
        ):
            raise RemoteTrainingError(f"unsafe live-test remote path {remote_path!r}")
        if remote_path != self.allowed_prefix and not remote_path.startswith(
            self.allowed_prefix + "/"
        ):
            raise RemoteTrainingError(
                "live-test bucket operation escaped its nonce prefix: "
                f"allowed={self.allowed_prefix!r}, requested={remote_path!r}"
            )

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        self._assert_allowed(remote_path)
        return self.backend.stat(remote_path)

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        self._assert_allowed(prefix)
        return self.backend.list(prefix)

    def download(self, remote_path: str, local_path: Path) -> None:
        self._assert_allowed(remote_path)
        self.backend.download(remote_path, local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self._assert_allowed(remote_path)
        if self.backend.stat(remote_path) is not None:
            raise RemoteTrainingError(f"live test refuses to overwrite remote object {remote_path}")
        self.backend.upload(local_path, remote_path)
        self.successful_uploads.append(remote_path)


class MarkerInterruptingBucket:
    """Inject an interruption before any canonical marker upload API call."""

    def __init__(self, backend: PrefixGuardedBucket) -> None:
        self.backend = backend
        self.interrupt_markers = True

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        return self.backend.stat(remote_path)

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        return self.backend.list(prefix)

    def download(self, remote_path: str, local_path: Path) -> None:
        self.backend.download(remote_path, local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        if self.interrupt_markers and Path(remote_path).name.startswith("remote_complete_"):
            raise OSError("injected interruption before remote marker upload")
        self.backend.upload(local_path, remote_path)


class BlockingUploadBucket(FakeBucket):
    def __init__(
        self,
        root: Path,
        *,
        release: threading.Event,
        fail_after_release: bool = False,
        block_suffix: str | None = None,
        fail_suffix: str | None = None,
    ) -> None:
        super().__init__(root)
        self.release = release
        self.fail_after_release = fail_after_release
        self.block_suffix = block_suffix
        self.fail_suffix = fail_suffix
        self.upload_started = threading.Event()
        self._blocked_once = False

    def upload(self, local_path: Path, remote_path: str) -> None:
        should_block = self.block_suffix is None or remote_path.endswith(self.block_suffix)
        if not self._blocked_once and should_block:
            self._blocked_once = True
            self.upload_started.set()
            if not self.release.wait(timeout=10):
                raise TimeoutError("test did not release blocked checkpoint upload")
        if self.fail_after_release and (
            self.fail_suffix is None or remote_path.endswith(self.fail_suffix)
        ):
            raise OSError("injected asynchronous checkpoint upload failure")
        super().upload(local_path, remote_path)


def _remote_config(**changes: Any) -> RemoteTrainingConfig:
    raw = {
        "bucket_id": "owner/bucket",
        "body_prefetch": {"lookahead_shards_per_rank": 2},
        "selection_prefix": "features",
        "checkpoint_prefix": "checkpoints/conditional_encdec",
        "selection_wait_timeout_seconds": 0.2,
        "checkpoint_retry_timeout_seconds": 0.2,
        "retry_interval_seconds": 0.001,
        "resume_latest": True,
    }
    raw.update(changes)
    parsed = RemoteTrainingConfig.parse(raw)
    assert parsed is not None
    return parsed


def _xet_hash(path: Path) -> str:
    return str(cast(Any, hf_xet).hash_files([str(path)])[0].hash)


def _require_safe_live_test_bucket(bucket_id: str, confirmation: str | None) -> None:
    production_bucket = "d0rb/nano-gpt-pretrain-steer-artifacts"
    if bucket_id == production_bucket:
        raise RemoteTrainingError("refusing to run live test against the production bucket")
    if not bucket_id.endswith("-test"):
        raise RemoteTrainingError("live-test bucket id must end with '-test'")
    if confirmation != bucket_id:
        raise RemoteTrainingError(
            "HF_REMOTE_TRAINING_TEST_BUCKET_WRITE_CONFIRM must exactly match "
            "HF_REMOTE_TRAINING_TEST_BUCKET"
        )


@pytest.mark.parametrize(
    ("bucket_id", "confirmation", "error"),
    [
        (
            "d0rb/nano-gpt-pretrain-steer-artifacts",
            "d0rb/nano-gpt-pretrain-steer-artifacts",
            "production bucket",
        ),
        ("owner/not-throwaway", "owner/not-throwaway", "must end with '-test'"),
        ("owner/throwaway-test", None, "must exactly match"),
        ("owner/throwaway-test", "owner/other-test", "must exactly match"),
    ],
)
def test_live_bucket_safety_gate_fails_closed(
    bucket_id: str,
    confirmation: str | None,
    error: str,
) -> None:
    with pytest.raises(RemoteTrainingError, match=error):
        _require_safe_live_test_bucket(bucket_id, confirmation)


def test_prefix_guarded_bucket_rejects_escape_and_overwrite(tmp_path: Path) -> None:
    inner = FakeBucket(tmp_path / "bucket")
    guarded = PrefixGuardedBucket(inner, allowed_prefix="codex-tests/issue-874/nonce")
    local = tmp_path / "body"
    local.write_bytes(b"body")

    with pytest.raises(RemoteTrainingError, match="escaped"):
        guarded.upload(local, "codex-tests/issue-874/sibling/body")
    with pytest.raises(RemoteTrainingError, match="unsafe"):
        guarded.upload(local, "codex-tests/issue-874/nonce/../sibling/body")
    guarded.upload(local, "codex-tests/issue-874/nonce/body")
    with pytest.raises(RemoteTrainingError, match="refuses to overwrite"):
        guarded.upload(local, "codex-tests/issue-874/nonce/body")


def test_hf_bucket_stat_uses_one_exact_path_record() -> None:
    remote_path = "features/family/remote_packed_inventory.json"
    api = Mock()
    api.get_bucket_paths_info.return_value = [
        SimpleNamespace(
            path=remote_path,
            size=179_752,
            xet_hash="b" * 64,
        )
    ]
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    assert backend.stat(remote_path) == RemoteObjectInfo(
        path=remote_path,
        byte_length=179_752,
        xet_hash="b" * 64,
    )
    api.get_bucket_paths_info.assert_called_once_with("owner/bucket", [remote_path])
    api.get_bucket_file_metadata.assert_not_called()


def test_hf_bucket_stat_missing_exact_path_preserves_not_found_contract() -> None:
    api = Mock()
    api.get_bucket_paths_info.return_value = []
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    assert backend.stat("features/missing.json") is None


@pytest.mark.parametrize(
    ("records", "match"),
    [
        (
            [
                SimpleNamespace(path="features/object", size=1, xet_hash="a" * 64),
                SimpleNamespace(path="features/object", size=1, xet_hash="a" * 64),
            ],
            "returned 2 records",
        ),
        (
            [SimpleNamespace(path="features/other", size=1, xet_hash="a" * 64)],
            "returned the wrong path",
        ),
    ],
)
def test_hf_bucket_stat_rejects_ambiguous_or_wrong_path_metadata(
    records: list[SimpleNamespace],
    match: str,
) -> None:
    api = Mock()
    api.get_bucket_paths_info.return_value = records
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    with pytest.raises(RemoteTrainingError, match=match):
        backend.stat("features/object")


def test_hf_bucket_stat_propagates_metadata_api_failures() -> None:
    api = Mock()
    api.get_bucket_paths_info.side_effect = OSError("metadata service unavailable")
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    with pytest.raises(OSError, match="metadata service unavailable"):
        backend.stat("features/object")


def test_hf_bucket_list_filters_provider_prefix_collision_siblings() -> None:
    api = Mock()
    api.list_bucket_tree.return_value = [
        SimpleNamespace(
            type="file",
            path="features/base/shard_00000.bin",
            size=11,
            xet_hash="a" * 64,
        ),
        SimpleNamespace(
            type="file",
            path="features/base_smoke/shard_00000.bin",
            size=13,
            xet_hash="b" * 64,
        ),
    ]
    backend = HfBucketBackend("owner/bucket", api=cast(Any, api))

    assert backend.list("features/base") == [
        RemoteObjectInfo(
            path="features/base/shard_00000.bin",
            byte_length=11,
            xet_hash="a" * 64,
        )
    ]
    api.list_bucket_tree.assert_called_once_with(
        "owner/bucket",
        prefix="features/base",
        recursive=True,
    )


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"bucket_id": "bad"}, "owner/name"),
        ({"selection_prefix": "../features"}, "traversal"),
        ({"checkpoint_prefix": "/checkpoints"}, "normalized relative"),
        ({"retry_interval_seconds": True}, "positive number"),
        ({"resume_latest": 1}, "strict bool"),
        ({"async_checkpoint_upload": 1}, "strict bool"),
        ({"async_checkpoint_upload": None}, "strict bool"),
    ],
)
def test_remote_config_rejects_malformed_fields(change, match):
    with pytest.raises(RemoteTrainingError, match=match):
        _remote_config(**change)


def test_remote_config_rejects_unknown_keys():
    raw = _remote_config().__dict__ | {"extra": 1}
    with pytest.raises(RemoteTrainingError, match="exactly"):
        RemoteTrainingConfig.parse(raw)


def test_remote_config_without_body_prefetch_has_precise_migration_error() -> None:
    raw = {
        "bucket_id": "owner/bucket",
        "selection_prefix": "features",
        "checkpoint_prefix": "checkpoints/conditional_encdec",
        "selection_wait_timeout_seconds": 0.2,
        "checkpoint_retry_timeout_seconds": 0.2,
        "retry_interval_seconds": 0.001,
        "resume_latest": True,
    }
    with pytest.raises(
        RemoteTrainingError,
        match="synchronous demand downloader was removed",
    ):
        RemoteTrainingConfig.parse(raw)


@pytest.mark.parametrize(
    "body_prefetch",
    [
        None,
        [],
        {},
        {"lookahead_shards_per_rank": None},
        {"lookahead_shards_per_rank": True},
        {"lookahead_shards_per_rank": 0},
        {"lookahead_shards_per_rank": -1},
        {"lookahead_shards_per_rank": 1.0},
        {"lookahead_shards_per_rank": "2"},
        {"lookahead_shards_per_rank": 2, "extra": 1},
        {"lookahead_shards_per_rank": 2, "cache_shards": True},
        {"lookahead_shards_per_rank": 2, "cache_shards": 1},
        {"lookahead_shards_per_rank": 2, "cache_shards": 2.0},
        {"lookahead_shards_per_rank": 2, "cache_shards": "2"},
    ],
)
def test_remote_body_prefetch_parse_matrix_is_strict(body_prefetch: Any) -> None:
    with pytest.raises(RemoteTrainingError, match="body_prefetch"):
        _remote_config(body_prefetch=body_prefetch)


def test_remote_train_packed_absent_is_exactly_off_and_valid_value_parses():
    mode_off = _remote_config()
    assert mode_off.body_prefetch.lookahead_shards_per_rank == 2
    assert mode_off.body_prefetch.cache_shards == 2
    assert mode_off.async_checkpoint_upload is False
    assert mode_off.train_packed is None
    assert _remote_config(async_checkpoint_upload=True).async_checkpoint_upload is True
    assert _remote_config(async_checkpoint_upload=False).async_checkpoint_upload is False
    enabled = _remote_config(train_packed={})
    assert enabled.train_packed is not None
    assert enabled.train_packed.trust_existing_cache is False


def test_remote_body_cache_retention_parses_explicitly() -> None:
    enabled = _remote_config(body_prefetch={"lookahead_shards_per_rank": 2, "cache_shards": 4})
    assert enabled.body_prefetch.cache_shards == 4


def test_remote_train_packed_trusted_cache_is_explicit_and_warns() -> None:
    messages: list[Any] = []
    sink = logger.add(messages.append, format="{message}")
    try:
        disabled = _remote_config(train_packed={})
        enabled = _remote_config(
            train_packed={
                "trust_existing_cache": True,
            }
        )
    finally:
        logger.remove(sink)
    assert disabled.train_packed is not None
    assert disabled.train_packed.trust_existing_cache is False
    assert enabled.train_packed is not None
    assert enabled.train_packed.trust_existing_cache is True
    trusted_warnings = [
        message
        for message in messages
        if message.record["level"].name == "WARNING" and "trust_existing_cache=true" in str(message)
    ]
    assert len(trusted_warnings) == 1
    warning = str(trusted_warnings[0])
    assert "atomically published without owner-side content rereads" in warning
    assert "consumers skip content hashing after a publication receipt" in warning
    assert "preexisting packed and selection cache bodies" in warning.lower()
    assert "relies on transport completion and decode" in warning
    assert "use trust_existing_cache=false" in warning


@pytest.mark.parametrize(
    "train_packed",
    [
        {"wrong": 2},
        {"trust_existing_cache": 1},
        {"trust_existing_cache": True, "extra": 1},
        None,
        [],
    ],
)
def test_remote_train_packed_parse_matrix_is_strict(
    train_packed: Any,
) -> None:
    with pytest.raises(
        RemoteTrainingError,
        match="train_packed",
    ):
        _remote_config(train_packed=train_packed)


def test_trusted_cache_body_requires_direct_single_link_and_exact_size(
    tmp_path: Path,
) -> None:
    body = tmp_path / "body"
    body.write_bytes(b"abcd")
    remote_module._check_trusted_cache_body(body, byte_length=4)
    with pytest.raises(RemoteTrainingError, match="size mismatch"):
        remote_module._check_trusted_cache_body(body, byte_length=3)

    symlink = tmp_path / "symlink"
    symlink.symlink_to(body)
    with pytest.raises(RemoteTrainingError, match="direct regular"):
        remote_module._check_trusted_cache_body(symlink, byte_length=4)

    hardlink = tmp_path / "hardlink"
    os.link(body, hardlink)
    with pytest.raises(RemoteTrainingError, match="exactly one hard link"):
        remote_module._check_trusted_cache_body(body, byte_length=4)


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"use_precomputed_batches": False}, "precomputed"),
        ({"feature_clusters": None}, "feature_clusters"),
        ({"feature_delta": None}, "splice_counterfactual"),
        ({"max_seq_len": 2048}, "1024"),
        ({"model_tag": None}, "model_tag"),
        ({"resume_from_step": 10}, "incompatible"),
    ],
)
def test_remote_eligibility_is_exact(change, match):
    cfg = {
        "use_precomputed_batches": True,
        "feature_clusters": {"path": "clusters.json"},
        "feature_delta": {"mode": "splice_counterfactual"},
        "max_seq_len": 1024,
        "model_tag": "l22-clustered-c1024",
        "resume_from_step": -1,
    }
    cfg.update(change)
    with pytest.raises(RemoteTrainingError, match=match):
        validate_remote_training_eligibility(cfg, _remote_config())


def test_loader_certificate_strict_aggregate_and_field_sensitivity():
    cert = build_remote_loader_certificate(
        selection_checksum="1" * 64,
        selection_inventory_aggregate="2" * 64,
        packed_row_fingerprint="3" * 64,
        max_features_per_segment=17,
    )
    assert cert["schema_version"] == 2
    assert CLUSTER_DELTA_SEMANTICS_KEY not in cert
    assert parse_remote_loader_certificate(cert, surface="test") == cert
    for key in (
        "selection_checksum",
        "selection_inventory_aggregate",
        "packed_row_fingerprint",
        "max_features_per_segment",
    ):
        changed = copy.deepcopy(cert)
        changed[key] = 18 if key == "max_features_per_segment" else "4" * 64
        with pytest.raises(RemoteTrainingError, match="aggregate"):
            parse_remote_loader_certificate(changed, surface="test")


def test_loader_certificate_rejects_former_schema_v1_even_with_valid_aggregate():
    cert = build_remote_loader_certificate(
        selection_checksum="1" * 64,
        selection_inventory_aggregate="2" * 64,
        packed_row_fingerprint="3" * 64,
        max_features_per_segment=17,
    )
    former_signed = {key: value for key, value in cert.items() if key != "aggregate_sha256"}
    former_signed["schema_version"] = 1
    former_aggregate = hashlib.sha256(
        json.dumps(
            former_signed,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        ).encode("utf-8")
    ).hexdigest()

    with pytest.raises(RemoteTrainingError, match="schema_version must be 2"):
        parse_remote_loader_certificate(
            {**former_signed, "aggregate_sha256": former_aggregate},
            surface="test",
        )


def test_clustered_splice_loader_certificate_explicitly_binds_semantics():
    cert = build_remote_loader_certificate(
        selection_checksum="1" * 64,
        selection_inventory_aggregate="2" * 64,
        packed_row_fingerprint="3" * 64,
        max_features_per_segment=17,
        cluster_delta_semantics=CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    )
    assert cert["schema_version"] == 2
    assert cert[CLUSTER_DELTA_SEMANTICS_KEY] == CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
    assert parse_remote_loader_certificate(cert, surface="test") == cert

    stale = copy.deepcopy(cert)
    stale[CLUSTER_DELTA_SEMANTICS_KEY] = "post_mask_post_delta_max_percentile_v1"
    with pytest.raises(RemoteTrainingError, match="canonical clustered splice"):
        parse_remote_loader_certificate(stale, surface="test")


def _selection_remote_fixture(tmp_path: Path):
    feature_root = tmp_path / "features-local"
    selection_dir = feature_root / "family" / "c1024" / "sall" / "sel" / ("a" * 64)
    remote_source = tmp_path / "selection-source"
    remote_source.mkdir()
    body = remote_source / "shard_00000.safetensors.zst"
    body.write_bytes(b"compressed-selection-body")
    inventory = build_clustered_selection_shard_integrity([body])
    packed_fingerprint = "b" * 64
    loader = build_remote_loader_certificate(
        selection_checksum="a" * 64,
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=packed_fingerprint,
        max_features_per_segment=23,
    )
    manifest = {
        "schema_version": 2,
        "checksum": "a" * 64,
        "metadata": {
            SELECTION_SHARD_INTEGRITY_KEY: inventory,
            REMOTE_LOADER_KEY: loader,
            "feature_mask": {"mode": "whitelist", "ids": [1, 2]},
            "feature_clusters": {
                "descriptor": {"content_sha256": "c" * 64},
                "canonical": {"clusters": [[1, 2]]},
                "counts": {"source_clusters": 1},
                "status": "complete",
            },
            "packed_rows": {"packed_row_fingerprint": packed_fingerprint},
        },
    }
    bucket = FakeBucket(tmp_path / "bucket")
    remote_dir = "features/family/c1024/sall/sel/" + "a" * 64
    remote_manifest = bucket._path(remote_dir + "/manifest.json")
    remote_manifest.parent.mkdir(parents=True)
    remote_manifest.write_text(json.dumps(manifest), encoding="utf-8")
    shutil.copyfile(body, bucket._path(remote_dir + "/" + body.name))
    return bucket, feature_root, selection_dir, packed_fingerprint, body, remote_dir


def test_selection_discovery_is_metadata_only_and_reader_requires_worker(
    tmp_path: Path,
) -> None:
    bucket, root, selection_dir, packed_fp, body, _ = _selection_remote_fixture(tmp_path)
    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=_remote_config(),
        local_selection_dir=selection_dir,
        feature_root=root,
        expected_selection_checksum="a" * 64,
        packed_row_fingerprint=packed_fp,
        packed_stems={"shard_00000"},
        label="test",
    )
    final = selection_dir / body.name

    assert (selection_dir / "manifest.json").is_file()
    assert not final.exists()
    assert materializer.max_features_per_segment == 23
    assert "backend" not in vars(materializer)
    with pytest.raises(RemoteTrainingError, match="strict rank-striped"):
        materializer.read(final)


def test_selection_read_honors_preexisting_cancellation_without_cache_mutation(
    tmp_path: Path,
) -> None:
    bucket, root, selection_dir, packed_fp, body, _ = _selection_remote_fixture(tmp_path)
    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=_remote_config(),
        local_selection_dir=selection_dir,
        feature_root=root,
        expected_selection_checksum="a" * 64,
        packed_row_fingerprint=packed_fp,
        packed_stems={"shard_00000"},
        label="cancelled selection read",
    )
    final = selection_dir / body.name
    cancel_event = threading.Event()
    cancel_event.set()

    with pytest.raises(RemoteMaterializationCancelled, match="cancelled"):
        materializer.read(final, cancel_event=cancel_event)

    assert not final.exists()


def test_selection_discovery_verifies_and_records_preexisting_body(
    tmp_path: Path,
) -> None:
    bucket, root, selection_dir, packed_fp, body, _ = _selection_remote_fixture(tmp_path)
    selection_dir.mkdir(parents=True)
    final = selection_dir / body.name
    shutil.copyfile(body, final)

    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=_remote_config(),
        local_selection_dir=selection_dir,
        feature_root=root,
        expected_selection_checksum="a" * 64,
        packed_row_fingerprint=packed_fp,
        packed_stems={"shard_00000"},
        label="warm selection",
    )

    assert materializer.preexisting_descriptors == {
        body.name: remote_module._capture_cache_body_descriptor(final)
    }


def test_selection_discovery_rejects_corrupt_preexisting_body(
    tmp_path: Path,
) -> None:
    bucket, root, selection_dir, packed_fp, body, _ = _selection_remote_fixture(tmp_path)
    selection_dir.mkdir(parents=True)
    (selection_dir / body.name).write_bytes(b"corrupt")

    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        RemoteSelectionMaterializer.discover(
            backend=bucket,
            remote=_remote_config(),
            local_selection_dir=selection_dir,
            feature_root=root,
            expected_selection_checksum="a" * 64,
            packed_row_fingerprint=packed_fp,
            packed_stems={"shard_00000"},
            label="corrupt warm selection",
        )


def test_remote_discovery_accepts_current_cluster_delta_certificate_without_body(
    tmp_path: Path,
) -> None:
    bucket, root, selection_dir, packed_fp, body, remote_dir = _selection_remote_fixture(tmp_path)
    manifest_path = bucket._path(remote_dir + "/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    inventory = metadata[SELECTION_SHARD_INTEGRITY_KEY]
    metadata["splice_counterfactual"] = {
        "schema_version": 2,
        CLUSTER_DELTA_SEMANTICS_KEY: CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    }
    metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum="a" * 64,
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=packed_fp,
        max_features_per_segment=23,
        cluster_delta_semantics=CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=_remote_config(),
        local_selection_dir=selection_dir,
        feature_root=root,
        expected_selection_checksum="a" * 64,
        packed_row_fingerprint=packed_fp,
        packed_stems={"shard_00000"},
        label="test",
    )
    assert materializer.loader_certificate[CLUSTER_DELTA_SEMANTICS_KEY] == (
        CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
    )
    assert not (selection_dir / body.name).exists()


@pytest.mark.parametrize(
    ("splice_manifest", "loader_has_semantics", "match"),
    [
        (True, False, "cluster-delta semantics drift"),
        (False, True, "without splice_counterfactual manifest metadata"),
    ],
)
def test_remote_discovery_rejects_schema_v2_conditional_semantics_mismatch(
    tmp_path: Path,
    splice_manifest: bool,
    loader_has_semantics: bool,
    match: str,
) -> None:
    bucket, root, selection_dir, packed_fp, body, remote_dir = _selection_remote_fixture(tmp_path)
    manifest_path = bucket._path(remote_dir + "/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata = manifest["metadata"]
    if splice_manifest:
        metadata["splice_counterfactual"] = {
            "schema_version": 2,
            CLUSTER_DELTA_SEMANTICS_KEY: CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
        }
    inventory = metadata[SELECTION_SHARD_INTEGRITY_KEY]
    metadata[REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum="a" * 64,
        selection_inventory_aggregate=inventory["aggregate_sha256"],
        packed_row_fingerprint=packed_fp,
        max_features_per_segment=23,
        cluster_delta_semantics=(
            CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS if loader_has_semantics else None
        ),
    )
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RemoteTrainingError, match=match):
        RemoteSelectionMaterializer.discover(
            backend=bucket,
            remote=_remote_config(),
            local_selection_dir=selection_dir,
            feature_root=root,
            expected_selection_checksum="a" * 64,
            packed_row_fingerprint=packed_fp,
            packed_stems={"shard_00000"},
            label="test",
        )
    assert not (selection_dir / body.name).exists()


@pytest.mark.parametrize(
    "splice_metadata",
    [
        {"schema_version": 1},
        {
            "schema_version": 1,
            CLUSTER_DELTA_SEMANTICS_KEY: CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
        },
        {"schema_version": 2},
    ],
)
def test_remote_discovery_rejects_old_cluster_delta_manifest_before_body(
    tmp_path: Path,
    splice_metadata: dict[str, object],
) -> None:
    bucket, root, selection_dir, packed_fp, body, remote_dir = _selection_remote_fixture(tmp_path)
    manifest_path = bucket._path(remote_dir + "/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    # Old and hybrid shapes are all stale: neither adding the new semantic to
    # schema v1 nor relabeling the schema without its semantic is sufficient.
    # The fixture's delta-free-shaped loader certificate also remains body-free here.
    manifest["metadata"]["splice_counterfactual"] = splice_metadata
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(RemoteTrainingError, match="stale or missing"):
        RemoteSelectionMaterializer.discover(
            backend=bucket,
            remote=_remote_config(),
            local_selection_dir=selection_dir,
            feature_root=root,
            expected_selection_checksum="a" * 64,
            packed_row_fingerprint=packed_fp,
            packed_stems={"shard_00000"},
            label="test",
        )
    assert not (selection_dir / body.name).exists()


def test_selection_inventory_parser_rejects_traversal_and_duplicates(tmp_path):
    path = tmp_path / "shard_00000.safetensors.zst"
    path.write_bytes(b"x")
    inventory = build_clustered_selection_shard_integrity([path])
    bad = copy.deepcopy(inventory)
    bad["shards"][0]["name"] = "../shard_00000.safetensors.zst"
    with pytest.raises(Exception, match="direct shard filename"):
        parse_clustered_selection_shard_inventory(bad, surface="test")
    duplicate = copy.deepcopy(inventory)
    duplicate["shards"].append(copy.deepcopy(duplicate["shards"][0]))
    with pytest.raises(Exception, match="strictly sorted"):
        parse_clustered_selection_shard_inventory(duplicate, surface="test")


def _checkpoint_store(tmp_path: Path, *, world_size: int = 2):
    checkpoint_root = tmp_path / "checkpoints-local"
    checkpoint_dir = checkpoint_root / "model" / "leaf"
    checkpoint_dir.mkdir(parents=True)
    model = {"weight": t.arange(6).reshape(2, 3)}
    for rank in range(world_size):
        save_checkpoint(
            str(checkpoint_dir),
            10,
            model,
            {"state": {rank: {"step": t.tensor(rank)}}},
            {"step": 10, REMOTE_RUN_META_KEY: "d" * 64},
            rank=rank,
        )
    bucket = FakeBucket(tmp_path / "bucket")
    store = RemoteCheckpointStore(
        backend=bucket,
        remote=_remote_config(),
        checkpoint_dir=checkpoint_dir,
        checkpoint_root=checkpoint_root,
        remote_run_fingerprint="d" * 64,
        world_size=world_size,
    )
    return store, bucket, checkpoint_dir, checkpoint_root


def test_checkpoint_marker_exact_required_group_and_missing_rank(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker = build_checkpoint_marker(
        checkpoint_dir,
        step=10,
        world_size=2,
        remote_run_fingerprint="d" * 64,
    )
    assert [item["name"] for item in marker["files"]] == [
        "model_000010.pt",
        "meta_000010.json",
        "optim_000010_rank0.pt",
        "optim_000010_rank1.pt",
    ]
    assert parse_checkpoint_marker(marker, surface="test") == marker
    (checkpoint_dir / "optim_000010_rank1.pt").unlink()
    with pytest.raises((FileNotFoundError, RemoteTrainingError)):
        build_checkpoint_marker(
            checkpoint_dir,
            step=10,
            world_size=2,
            remote_run_fingerprint=store.remote_run_fingerprint,
        )


def test_checkpoint_publish_marker_last_idempotent_restore_and_roundtrip(tmp_path):
    store, bucket, checkpoint_dir, checkpoint_root = _checkpoint_store(tmp_path)
    marker = store.publish(10)
    assert bucket.upload_order[-1].endswith("remote_complete_000010.json")
    upload_order = list(bucket.upload_order)
    assert store.publish(10) == marker
    assert bucket.upload_order == upload_order
    assert store.latest_complete_step() == 10

    original_bytes = {path.name: path.read_bytes() for path in checkpoint_dir.iterdir()}
    fresh_root = tmp_path / "fresh" / "checkpoints-local"
    restored_dir = fresh_root / "model" / "leaf"
    restored = RemoteCheckpointStore(
        backend=bucket,
        remote=store.remote,
        checkpoint_dir=restored_dir,
        checkpoint_root=fresh_root,
        remote_run_fingerprint=store.remote_run_fingerprint,
        world_size=2,
    )
    restored.restore(10)
    for name, expected in original_bytes.items():
        assert (restored_dir / name).read_bytes() == expected
    model, optimizer, meta = load_checkpoint(
        str(restored_dir), 10, "cpu", load_optimizer=True, rank=1
    )
    assert t.equal(model["weight"], t.arange(6).reshape(2, 3))
    assert optimizer["state"][1]["step"].item() == 1
    assert_checkpoint_remote_run(meta, "d" * 64)


def _generation_remote_path(
    store: RemoteCheckpointStore,
    marker: Mapping[str, Any],
    name: str,
) -> str:
    return store.remote_dir + f"/_groups/{marker['aggregate_sha256']}/{name}"


def _install_legacy_flat_group(
    store: RemoteCheckpointStore,
    bucket: FakeBucket,
    marker: Mapping[str, Any],
) -> None:
    for entry in marker["files"]:
        source = bucket._path(_generation_remote_path(store, marker, entry["name"]))
        target = bucket._path(store.remote_dir + f"/{entry['name']}")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
    shutil.rmtree(bucket._path(store.remote_dir + f"/_groups/{marker['aggregate_sha256']}"))


def test_local_marker_match_probe_is_strict_and_non_authoritative(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker = build_checkpoint_marker(
        checkpoint_dir,
        step=10,
        world_size=store.world_size,
        remote_run_fingerprint=store.remote_run_fingerprint,
    )
    candidate = checkpoint_dir / "candidate-marker.json"
    assert store._local_marker_matches(candidate, marker) is False

    target = checkpoint_dir / "marker-target.json"
    remote_module.atomic_json_dump(marker, target)
    candidate.symlink_to(target)
    assert store._local_marker_matches(candidate, marker) is False
    candidate.unlink()

    candidate.write_text("{", encoding="utf-8")
    assert store._local_marker_matches(candidate, marker) is False
    remote_module.atomic_json_dump(marker, candidate)
    assert store._local_marker_matches(candidate, marker) is True


def test_checkpoint_resolver_restores_legacy_flat_and_generation_layouts(tmp_path):
    generation_store, generation_bucket, generation_dir, _ = _checkpoint_store(
        tmp_path / "generation"
    )
    generation_marker = generation_store.publish(10)
    assert generation_store._resolve_committed_group(generation_marker).layout == (
        "immutable_generation"
    )

    legacy_store, legacy_bucket, legacy_dir, _ = _checkpoint_store(tmp_path / "legacy")
    legacy_marker = legacy_store.publish(10)
    _install_legacy_flat_group(legacy_store, legacy_bucket, legacy_marker)
    resolved = legacy_store._resolve_committed_group(legacy_marker)
    assert resolved.layout == "legacy_flat"
    assert list(resolved.objects) == [entry["name"] for entry in legacy_marker["files"]]

    shutil.rmtree(legacy_dir)
    restored = legacy_store.restore(10)
    assert restored == legacy_marker
    for entry in legacy_marker["files"]:
        assert (legacy_dir / entry["name"]).read_bytes() == legacy_bucket._path(
            legacy_store.remote_dir + f"/{entry['name']}"
        ).read_bytes()
    assert (
        generation_dir.joinpath("remote_complete_000010.json").read_bytes()
        == (legacy_dir / "remote_complete_000010.json").read_bytes()
    )


def test_partial_committed_generation_never_falls_back_to_complete_flat_group(tmp_path):
    store, bucket, _, _ = _checkpoint_store(tmp_path)
    marker = store.publish(10)
    _install_legacy_flat_group(store, bucket, marker)
    first = marker["files"][0]
    generation_path = bucket._path(_generation_remote_path(store, marker, first["name"]))
    generation_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        bucket._path(store.remote_dir + f"/{first['name']}"),
        generation_path,
    )

    with pytest.raises(RemoteTrainingError, match="immutable_generation.*missing"):
        store.latest_complete_step()


def test_drifted_committed_generation_never_falls_back_to_complete_flat_group(tmp_path):
    store, bucket, _, _ = _checkpoint_store(tmp_path)
    marker = store.publish(10)
    for entry in marker["files"]:
        source = bucket._path(_generation_remote_path(store, marker, entry["name"]))
        flat = bucket._path(store.remote_dir + f"/{entry['name']}")
        flat.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, flat)
    bucket._path(_generation_remote_path(store, marker, marker["files"][0]["name"])).write_bytes(
        b"drifted-generation"
    )

    with pytest.raises(RemoteTrainingError, match="identity drift.*immutable_generation"):
        store.latest_complete_step()


def test_matching_partial_generation_is_reused_before_marker_commit(tmp_path):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker = build_checkpoint_marker(
        checkpoint_dir,
        step=10,
        world_size=store.world_size,
        remote_run_fingerprint=store.remote_run_fingerprint,
    )
    first = marker["files"][0]
    first_remote = _generation_remote_path(store, marker, first["name"])
    first_path = bucket._path(first_remote)
    first_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(checkpoint_dir / first["name"], first_path)

    assert store.publish(10) == marker
    assert first_remote not in bucket.upload_order
    assert bucket.upload_order[-1] == store.remote_dir + "/remote_complete_000010.json"


@pytest.mark.parametrize("local_marker_bytes", [b"{", b'{"stale": true}'])
def test_absent_remote_marker_replaces_malformed_local_marker_and_ignores_flat_debris(
    tmp_path,
    local_marker_bytes,
):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker_path = checkpoint_dir / "remote_complete_000010.json"
    marker_path.write_bytes(local_marker_bytes)
    flat_model = bucket._path(store.remote_dir + "/model_000010.pt")
    flat_meta = bucket._path(store.remote_dir + "/meta_000010.json")
    flat_model.parent.mkdir(parents=True, exist_ok=True)
    flat_model.write_bytes(b"interrupted-old-model")
    flat_meta.write_bytes(b"interrupted-old-meta")
    debris = {flat_model: flat_model.read_bytes(), flat_meta: flat_meta.read_bytes()}

    marker = store.publish(10)

    assert (
        parse_checkpoint_marker(
            json.loads(marker_path.read_text(encoding="utf-8")),
            surface="reconciled local marker",
        )
        == marker
    )
    assert {path: path.read_bytes() for path in debris} == debris
    for entry in marker["files"]:
        assert bucket._path(_generation_remote_path(store, marker, entry["name"])).is_file()
    assert bucket.upload_order[-1] == store.remote_dir + "/remote_complete_000010.json"


def test_equal_remote_marker_reconciles_stale_local_marker_without_upload(tmp_path):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker = store.publish(10)
    upload_order = list(bucket.upload_order)
    local_marker = checkpoint_dir / "remote_complete_000010.json"
    local_marker.write_text("malformed", encoding="utf-8")

    assert store.publish(10) == marker
    assert bucket.upload_order == upload_order
    assert (
        parse_checkpoint_marker(
            json.loads(local_marker.read_text(encoding="utf-8")),
            surface="reconciled local marker",
        )
        == marker
    )


def test_different_remote_marker_fails_before_local_marker_mutation(tmp_path):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    committed = store.publish(10)
    committed_local_marker = (checkpoint_dir / "remote_complete_000010.json").read_bytes()

    for rank in range(store.world_size):
        save_checkpoint(
            str(checkpoint_dir),
            10,
            {"weight": t.arange(6).reshape(2, 3) + 100},
            {"state": {rank: {"step": t.tensor(100 + rank)}}},
            {"step": 10, REMOTE_RUN_META_KEY: "d" * 64},
            rank=rank,
        )

    with pytest.raises(RemoteTrainingError, match="same-step remote checkpoint collision"):
        store.publish(10)
    assert (checkpoint_dir / "remote_complete_000010.json").read_bytes() == committed_local_marker
    assert (
        parse_checkpoint_marker(
            json.loads(committed_local_marker),
            surface="original local marker",
        )
        == committed
    )
    assert bucket.upload_order[-1] == store.remote_dir + "/remote_complete_000010.json"


def test_restore_replaces_stale_local_marker_only_after_local_files_verify(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker = store.publish(10)
    local_marker = checkpoint_dir / "remote_complete_000010.json"
    local_marker.write_text("malformed", encoding="utf-8")
    (checkpoint_dir / marker["files"][0]["name"]).write_bytes(b"corrupt-local-data")

    with pytest.raises(RemoteTrainingError, match="integrity mismatch"):
        store.restore(10)
    assert local_marker.read_text(encoding="utf-8") == "malformed"

    (checkpoint_dir / marker["files"][0]["name"]).unlink()
    assert store.restore(10) == marker
    assert (
        parse_checkpoint_marker(
            json.loads(local_marker.read_text(encoding="utf-8")),
            surface="restored local marker",
        )
        == marker
    )


@pytest.mark.parametrize("interrupted_file_count", range(11))
def test_world_size_eight_interrupted_same_step_relaunch_is_self_healing(
    tmp_path,
    interrupted_file_count,
):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path, world_size=8)
    committed_n = store.publish(10)
    _install_legacy_flat_group(store, bucket, committed_n)

    for rank in range(store.world_size):
        save_checkpoint(
            str(checkpoint_dir),
            510,
            {"weight": t.arange(6).reshape(2, 3) + 510},
            {"state": {rank: {"step": t.tensor(5100 + rank)}}, "rng": f"old-{rank}"},
            {
                "step": 510,
                "dataloader_state": {"cursor": 510},
                REMOTE_RUN_META_KEY: "d" * 64,
            },
            rank=rank,
        )
    interrupted_marker = build_checkpoint_marker(
        checkpoint_dir,
        step=510,
        world_size=store.world_size,
        remote_run_fingerprint=store.remote_run_fingerprint,
    )
    interrupted_names = [
        entry["name"] for entry in interrupted_marker["files"][:interrupted_file_count]
    ]
    debris: dict[str, bytes] = {}
    for name in interrupted_names:
        body = (checkpoint_dir / name).read_bytes()
        remote_path = bucket._path(store.remote_dir + f"/{name}")
        remote_path.parent.mkdir(parents=True, exist_ok=True)
        remote_path.write_bytes(body)
        debris[name] = body
    atomic_marker = checkpoint_dir / "remote_complete_000510.json"
    remote_module.atomic_json_dump(interrupted_marker, atomic_marker)

    assert store.latest_complete_step() == 10
    assert store.restore(10) == committed_n

    for rank in range(store.world_size):
        save_checkpoint(
            str(checkpoint_dir),
            510,
            {"weight": t.arange(6).reshape(2, 3) + 1510},
            {"state": {rank: {"step": t.tensor(15100 + rank)}}, "rng": f"new-{rank}"},
            {
                "step": 510,
                "dataloader_state": {"cursor": 1510},
                REMOTE_RUN_META_KEY: "d" * 64,
            },
            rank=rank,
        )
    marker = store.publish(510)

    assert {
        name: bucket._path(store.remote_dir + f"/{name}").read_bytes() for name in debris
    } == debris
    assert marker["aggregate_sha256"] != interrupted_marker["aggregate_sha256"]
    assert bucket.upload_order[-1] == store.remote_dir + "/remote_complete_000510.json"
    assert store.latest_complete_step() == 510
    for entry in marker["files"]:
        assert bucket._path(_generation_remote_path(store, marker, entry["name"])).is_file()

    fresh_root = tmp_path / "fresh" / "checkpoints-local"
    fresh_store = RemoteCheckpointStore(
        backend=bucket,
        remote=store.remote,
        checkpoint_dir=fresh_root / "model" / "leaf",
        checkpoint_root=fresh_root,
        remote_run_fingerprint=store.remote_run_fingerprint,
        world_size=store.world_size,
    )
    fresh_store.restore(510)
    for entry in marker["files"]:
        assert (fresh_store.checkpoint_dir / entry["name"]).read_bytes() == (
            checkpoint_dir / entry["name"]
        ).read_bytes()
    model, optimizer, meta = load_checkpoint(
        str(fresh_store.checkpoint_dir),
        510,
        "cpu",
        load_optimizer=True,
        rank=7,
    )
    assert t.equal(model["weight"], t.arange(6).reshape(2, 3) + 1510)
    assert optimizer["state"][7]["step"].item() == 15107
    assert optimizer["rng"] == "new-7"
    assert meta["dataloader_state"] == {"cursor": 1510}


def test_latest_ignores_higher_unmarked_debris_and_rejects_marked_corruption(tmp_path):
    store, bucket, _, _ = _checkpoint_store(tmp_path)
    store.publish(10)
    debris = bucket._path(store.remote_dir + "/model_000020.pt")
    debris.write_bytes(b"uncommitted")
    assert store.latest_complete_step() == 10
    marker = bucket._path(store.remote_dir + "/remote_complete_000020.json")
    marker.write_text("{}", encoding="utf-8")
    with pytest.raises(RemoteTrainingError, match="marker"):
        store.latest_complete_step()


def test_latest_rejects_malformed_completion_marker_name(tmp_path):
    store, bucket, _, _ = _checkpoint_store(tmp_path)
    malformed = bucket._path(store.remote_dir + "/remote_complete_latest.json")
    malformed.parent.mkdir(parents=True)
    malformed.write_text("{}", encoding="utf-8")
    with pytest.raises(RemoteTrainingError, match="malformed.*filename"):
        store.latest_complete_step()


def test_checkpoint_same_step_collision_is_loud(tmp_path):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    marker = store.publish(10)
    remote_model = bucket._path(
        store.remote_dir + f"/_groups/{marker['aggregate_sha256']}/model_000010.pt"
    )
    remote_model.write_bytes(b"different")
    (checkpoint_dir / "remote_complete_000010.json").unlink()
    bucket._path(store.remote_dir + "/remote_complete_000010.json").unlink()
    with pytest.raises(RemoteTrainingError, match="collision"):
        store.publish(10)


def _poll_publisher_until_complete(
    publisher: RemoteCheckpointPublisher,
    *,
    timeout_seconds: float = 5.0,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while publisher.poll():
        if time.monotonic() >= deadline:
            raise AssertionError("checkpoint publisher did not complete before test timeout")
        time.sleep(0.01)


def test_ddp_publish_saved_checkpoint_sync_and_validation_paths(tmp_path):
    store, bucket, _, _ = _checkpoint_store(tmp_path)
    created = ddp_create_checkpoint_publisher(
        store=store,
        world_size=1,
        rank=0,
    )
    assert created is not None
    assert created.poll() is False
    created.close()

    assert (
        ddp_publish_saved_checkpoint(
            store=store,
            publisher=None,
            async_upload=False,
            step=10,
            world_size=1,
            rank=0,
        )
        is False
    )
    assert bucket.upload_order[-1].endswith("remote_complete_000010.json")

    with pytest.raises(RemoteTrainingError, match="strict bool"):
        ddp_publish_saved_checkpoint(
            store=store,
            publisher=None,
            async_upload=cast(Any, 1),
            step=10,
            world_size=1,
            rank=0,
        )
    with pytest.raises(RemoteTrainingError, match="missing its checkpoint publisher"):
        ddp_publish_saved_checkpoint(
            store=store,
            publisher=None,
            async_upload=True,
            step=10,
            world_size=1,
            rank=0,
        )

    class WrongStepPublisher:
        def submit(self, step: int) -> int:
            return step + 1

    with pytest.raises(RemoteTrainingError, match="acknowledged the wrong step"):
        ddp_publish_saved_checkpoint(
            store=store,
            publisher=cast(Any, WrongStepPublisher()),
            async_upload=True,
            step=10,
            world_size=1,
            rank=0,
        )

    with pytest.raises(RemoteTrainingError, match="missing its checkpoint publisher"):
        ddp_poll_checkpoint_publisher(
            publisher=None,
            world_size=1,
            rank=0,
        )

    class NonBoolPollPublisher:
        def poll(self) -> int:
            return 1

    with pytest.raises(RemoteTrainingError, match="non-bool result"):
        ddp_poll_checkpoint_publisher(
            publisher=cast(Any, NonBoolPollPublisher()),
            world_size=1,
            rank=0,
        )


def test_checkpoint_publisher_creation_closes_after_collective_failure(
    tmp_path,
    monkeypatch,
):
    store, _, _, _ = _checkpoint_store(tmp_path)
    closed: list[bool] = []

    class TrackingPublisher(RemoteCheckpointPublisher):
        def close(self) -> None:
            super().close()
            closed.append(True)

    def _fail_after_creation(*, action, world_size: int, rank: int):
        assert world_size == 2
        assert rank == 0
        action()
        raise RuntimeError("injected collective acknowledgement failure")

    monkeypatch.setattr(remote_module, "RemoteCheckpointPublisher", TrackingPublisher)
    monkeypatch.setattr(remote_module, "ddp_rank0_action", _fail_after_creation)
    with pytest.raises(RuntimeError, match="collective acknowledgement failure"):
        ddp_create_checkpoint_publisher(
            store=store,
            world_size=2,
            rank=0,
        )
    assert closed == [True]


def test_async_checkpoint_submit_rejects_missing_group_member(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    (checkpoint_dir / "optim_000010_rank1.pt").unlink()
    publisher = RemoteCheckpointPublisher(store)
    try:
        with pytest.raises(RemoteTrainingError, match="cannot capture checkpoint file identity"):
            publisher.submit(10)
    finally:
        publisher.close()


def test_async_checkpoint_publisher_is_nonblocking_bounded_and_marker_last(tmp_path):
    store, _, _, _ = _checkpoint_store(tmp_path)
    release = threading.Event()
    bucket = BlockingUploadBucket(tmp_path / "blocking-bucket", release=release)
    store.backend = bucket
    publisher = RemoteCheckpointPublisher(store)

    assert publisher.submit(10) == 10
    assert bucket.upload_started.wait(timeout=5)
    assert publisher.pending_step == 10
    assert publisher.poll() is True
    assert not bucket._path(store.remote_dir + "/remote_complete_000010.json").exists()
    with pytest.raises(RemoteTrainingError, match="capacity is full"):
        publisher.submit(10)

    drain_started = threading.Event()
    drain_finished = threading.Event()
    drain_failures: list[BaseException] = []

    def _drain_before_next_boundary() -> None:
        drain_started.set()
        try:
            ddp_drain_checkpoint_publisher(
                publisher=publisher,
                world_size=1,
                rank=0,
            )
        except BaseException as exc:
            drain_failures.append(exc)
        finally:
            drain_finished.set()

    drain_thread = threading.Thread(target=_drain_before_next_boundary)
    drain_thread.start()
    assert drain_started.wait(timeout=5)
    assert not drain_finished.wait(timeout=0.1)
    release.set()
    drain_thread.join(timeout=5)
    assert not drain_thread.is_alive()
    assert drain_failures == []
    assert drain_finished.is_set()
    assert publisher.pending_step is None
    assert bucket.upload_order[-1].endswith("remote_complete_000010.json")
    assert bucket._path(store.remote_dir + "/remote_complete_000010.json").is_file()
    upload_order = list(bucket.upload_order)
    assert publisher.submit(10) == 10
    _poll_publisher_until_complete(publisher)
    assert bucket.upload_order == upload_order

    for rank in range(store.world_size):
        save_checkpoint(
            str(store.checkpoint_dir),
            11,
            {"weight": t.arange(6).reshape(2, 3)},
            {"state": {rank: {"step": t.tensor(rank)}}},
            {"step": 11, REMOTE_RUN_META_KEY: "d" * 64},
            rank=rank,
        )
    assert publisher.submit(11) == 11
    _poll_publisher_until_complete(publisher)
    assert bucket.upload_order[-1].endswith("remote_complete_000011.json")
    assert bucket._path(store.remote_dir + "/remote_complete_000010.json").is_file()
    assert bucket._path(store.remote_dir + "/remote_complete_000011.json").is_file()

    publisher.close()
    publisher.close()
    with pytest.raises(RemoteTrainingError, match="after publisher close"):
        publisher.submit(10)
    assert not any(
        thread.name.startswith("remote-checkpoint-publish") for thread in threading.enumerate()
    )


def test_async_checkpoint_publisher_close_surfaces_unpolled_failure(tmp_path):
    store, _, _, _ = _checkpoint_store(tmp_path)
    release = threading.Event()
    release.set()
    bucket = BlockingUploadBucket(
        tmp_path / "failing-bucket",
        release=release,
        fail_after_release=True,
    )
    store.backend = bucket
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(10)

    with pytest.raises(RemoteTrainingError, match="uploading and verifying"):
        publisher.close()
    publisher.close()
    assert not bucket._path(store.remote_dir + "/remote_complete_000010.json").exists()


def test_async_checkpoint_done_failure_keeps_capacity_until_observed(tmp_path):
    store, _, _, _ = _checkpoint_store(tmp_path)
    release = threading.Event()
    release.set()
    bucket = BlockingUploadBucket(
        tmp_path / "finished-failure-bucket",
        release=release,
        fail_after_release=True,
    )
    store.backend = bucket
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(10)
    future = publisher._future
    assert future is not None
    deadline = time.monotonic() + 5
    while not future.done():
        if time.monotonic() >= deadline:
            raise AssertionError("checkpoint failure did not finish before test timeout")
        time.sleep(0.01)

    with pytest.raises(RemoteTrainingError, match="capacity is full"):
        publisher.submit(10)
    with pytest.raises(RemoteTrainingError, match="uploading and verifying"):
        publisher.poll()
    assert publisher.pending_step is None
    publisher.close()


def test_async_checkpoint_preserves_data_and_marker_retries(tmp_path):
    store, bucket, _, _ = _checkpoint_store(tmp_path)
    marker = build_checkpoint_marker(
        store.checkpoint_dir,
        step=10,
        world_size=store.world_size,
        remote_run_fingerprint=store.remote_run_fingerprint,
    )
    model_remote = store.remote_dir + f"/_groups/{marker['aggregate_sha256']}/model_000010.pt"
    marker_remote = store.remote_dir + "/remote_complete_000010.json"
    bucket.upload_failures[model_remote] = 1
    bucket.upload_failures[marker_remote] = 1
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(10)
    publisher.close()

    assert bucket.upload_failures[model_remote] == 0
    assert bucket.upload_failures[marker_remote] == 0
    assert bucket.upload_order[-1] == marker_remote
    assert store.latest_complete_step() == 10


def test_async_marker_failure_leaves_data_debris_and_restores_latest_complete(tmp_path):
    store, bucket, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    committed = store.publish(10)
    step_10_bytes = {
        entry["name"]: (checkpoint_dir / entry["name"]).read_bytes() for entry in committed["files"]
    }
    for rank in range(store.world_size):
        save_checkpoint(
            str(checkpoint_dir),
            11,
            {"weight": t.arange(6).reshape(2, 3) + 11},
            {"state": {rank: {"step": t.tensor(11)}}},
            {"step": 11, REMOTE_RUN_META_KEY: "d" * 64},
            rank=rank,
        )

    marker_remote = store.remote_dir + "/remote_complete_000011.json"
    bucket.upload_failures[marker_remote] = 10_000
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(11)
    with pytest.raises(RemoteTrainingError, match="uploading and verifying"):
        publisher.close()

    assert not bucket._path(marker_remote).exists()
    marker = build_checkpoint_marker(
        checkpoint_dir,
        step=11,
        world_size=store.world_size,
        remote_run_fingerprint=store.remote_run_fingerprint,
    )
    for name in remote_module._checkpoint_names(11, store.world_size):
        assert bucket._path(
            store.remote_dir + f"/_groups/{marker['aggregate_sha256']}/{name}"
        ).is_file()
    assert store.latest_complete_step() == 10

    fresh_root = tmp_path / "fresh" / "checkpoints-local"
    fresh_dir = fresh_root / "model" / "leaf"
    fresh_store = RemoteCheckpointStore(
        backend=bucket,
        remote=store.remote,
        checkpoint_dir=fresh_dir,
        checkpoint_root=fresh_root,
        remote_run_fingerprint=store.remote_run_fingerprint,
        world_size=store.world_size,
    )
    fresh_store.restore(10)
    assert {name: (fresh_dir / name).read_bytes() for name in step_10_bytes} == step_10_bytes


def test_async_checkpoint_identity_rejects_same_byte_replacement_and_hardlink(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    expected = remote_module._checkpoint_group_identity(
        checkpoint_dir,
        step=10,
        world_size=2,
    )
    model_path = checkpoint_dir / "model_000010.pt"
    replacement = checkpoint_dir / ".replacement.pt"
    replacement.write_bytes(model_path.read_bytes())
    os.replace(replacement, model_path)

    with pytest.raises(RemoteTrainingError, match="changed after.*enqueue"):
        store.publish(10, expected_local_group=expected)

    expected = remote_module._checkpoint_group_identity(
        checkpoint_dir,
        step=10,
        world_size=2,
    )
    hardlink = checkpoint_dir / ".model-hardlink.pt"
    os.link(model_path, hardlink)
    with pytest.raises(RemoteTrainingError, match="changed after.*enqueue"):
        store.publish(10, expected_local_group=expected)


def test_async_checkpoint_mutation_during_upload_never_commits_marker(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    release = threading.Event()
    bucket = BlockingUploadBucket(tmp_path / "mutating-bucket", release=release)
    store.backend = bucket
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(10)
    assert bucket.upload_started.wait(timeout=5)

    (checkpoint_dir / "model_000010.pt").write_bytes(b"mutated-during-upload")
    release.set()
    with pytest.raises(RemoteTrainingError, match="collision|changed|verification"):
        publisher.close()
    assert not bucket._path(store.remote_dir + "/remote_complete_000010.json").exists()


def test_async_checkpoint_same_byte_replacement_during_upload_is_detected(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    release = threading.Event()
    bucket = BlockingUploadBucket(tmp_path / "replacing-bucket", release=release)
    store.backend = bucket
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(10)
    assert bucket.upload_started.wait(timeout=5)

    model_path = checkpoint_dir / "model_000010.pt"
    replacement = checkpoint_dir / ".replacement.pt"
    replacement.write_bytes(model_path.read_bytes())
    os.replace(replacement, model_path)
    release.set()
    with pytest.raises(RemoteTrainingError, match="changed after.*enqueue"):
        publisher.close()
    assert not bucket._path(store.remote_dir + "/remote_complete_000010.json").exists()


def test_async_checkpoint_mutation_during_marker_upload_fails_after_safe_commit(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    original_model = (checkpoint_dir / "model_000010.pt").read_bytes()
    release = threading.Event()
    bucket = BlockingUploadBucket(
        tmp_path / "marker-blocking-bucket",
        release=release,
        block_suffix="remote_complete_000010.json",
    )
    store.backend = bucket
    publisher = RemoteCheckpointPublisher(store)
    publisher.submit(10)
    assert bucket.upload_started.wait(timeout=5)

    (checkpoint_dir / "model_000010.pt").write_bytes(b"mutated-during-marker-upload")
    release.set()
    with pytest.raises(RemoteTrainingError, match="changed after.*enqueue"):
        publisher.close()

    remote_marker = bucket._path(store.remote_dir + "/remote_complete_000010.json")
    assert remote_marker.is_file()
    assert store.latest_complete_step() == 10
    fresh_root = tmp_path / "fresh" / "checkpoints-local"
    fresh_dir = fresh_root / "model" / "leaf"
    fresh_store = RemoteCheckpointStore(
        backend=bucket,
        remote=store.remote,
        checkpoint_dir=fresh_dir,
        checkpoint_root=fresh_root,
        remote_run_fingerprint=store.remote_run_fingerprint,
        world_size=store.world_size,
    )
    fresh_store.restore(10)
    assert (fresh_dir / "model_000010.pt").read_bytes() == original_model


def test_async_checkpoint_submit_rejects_symlinked_group_member(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    model_path = checkpoint_dir / "model_000010.pt"
    target = checkpoint_dir / ".model-target.pt"
    target.write_bytes(model_path.read_bytes())
    model_path.unlink()
    model_path.symlink_to(target)
    publisher = RemoteCheckpointPublisher(store)
    try:
        with pytest.raises(RemoteTrainingError, match="direct regular file"):
            publisher.submit(10)
    finally:
        publisher.close()


def test_async_checkpoint_submit_rejects_preexisting_hardlink(tmp_path):
    store, _, checkpoint_dir, _ = _checkpoint_store(tmp_path)
    model_path = checkpoint_dir / "model_000010.pt"
    os.link(model_path, checkpoint_dir / ".model-hardlink.pt")
    publisher = RemoteCheckpointPublisher(store)
    try:
        with pytest.raises(RemoteTrainingError, match="exactly one hard link"):
            publisher.submit(10)
    finally:
        publisher.close()


def test_async_and_sync_checkpoint_publication_are_byte_identical(tmp_path):
    sync_store, sync_bucket, sync_dir, _ = _checkpoint_store(tmp_path / "sync")
    async_root = tmp_path / "async" / "checkpoints-local"
    async_dir = async_root / "model" / "leaf"
    shutil.copytree(sync_dir, async_dir)
    async_bucket = FakeBucket(tmp_path / "async-bucket")
    async_store = RemoteCheckpointStore(
        backend=async_bucket,
        remote=sync_store.remote,
        checkpoint_dir=async_dir,
        checkpoint_root=async_root,
        remote_run_fingerprint=sync_store.remote_run_fingerprint,
        world_size=sync_store.world_size,
    )

    sync_marker = sync_store.publish(10)
    publisher = RemoteCheckpointPublisher(async_store)
    publisher.submit(10)
    publisher.close()
    async_marker_path = async_dir / "remote_complete_000010.json"
    async_marker = parse_checkpoint_marker(
        json.loads(async_marker_path.read_text(encoding="utf-8")),
        surface="async marker",
    )

    assert async_marker == sync_marker
    assert async_marker_path.read_bytes() == (sync_dir / "remote_complete_000010.json").read_bytes()
    assert sync_bucket.upload_order[-1].endswith("remote_complete_000010.json")
    assert async_bucket.upload_order[-1].endswith("remote_complete_000010.json")
    sync_remote_dir = sync_bucket._path(sync_store.remote_dir)
    async_remote_dir = async_bucket._path(async_store.remote_dir)
    for sync_path in sorted(path for path in sync_remote_dir.rglob("*") if path.is_file()):
        async_path = async_remote_dir / sync_path.relative_to(sync_remote_dir)
        assert async_path.read_bytes() == sync_path.read_bytes()


def test_remote_run_fingerprint_ignores_operations_but_binds_identity():
    cfg: dict[str, Any] = {
        "max_seq_len": 1024,
        "depth": 12,
        "feature_dir": "/machine/a/features",
        "feature_mask": {"mode": "whitelist", "path": "/machine/a/mask.yaml"},
        "feature_clusters": {"path": "/machine/a/clusters.yaml"},
        "feature_mask_freq_min": None,
        "feature_mask_freq_max": None,
        "resume_from_step": -1,
        "remote_training": {"retry_interval_seconds": 1},
    }
    contracts = {
        label: {"selection_checksum": char * 64}
        for label, char in zip(("train", "derived_val", "base_val"), "abc", strict=True)
    }
    first = build_remote_run_fingerprint(
        resolved_training_config=cfg,
        world_size=2,
        selection_contracts=contracts,
    )
    operational = copy.deepcopy(cfg)
    operational["feature_dir"] = "/machine/b/features"
    operational["feature_mask"] = {
        "mode": "whitelist",
        "path": "/machine/b/equivalent-mask.yaml",
    }
    operational["feature_clusters"] = {"path": "/machine/b/equivalent-clusters.yaml"}
    operational["feature_mask_freq_min"] = 0.123
    operational["feature_mask_freq_max"] = 0.456
    operational["resume_from_step"] = 100
    operational["remote_training"] = {
        "retry_interval_seconds": 99,
        "async_checkpoint_upload": False,
    }
    assert (
        build_remote_run_fingerprint(
            resolved_training_config=operational,
            world_size=2,
            selection_contracts=contracts,
        )
        == first
    )
    changed = copy.deepcopy(contracts)
    changed["train"]["selection_checksum"] = "f" * 64
    assert (
        build_remote_run_fingerprint(
            resolved_training_config=cfg,
            world_size=2,
            selection_contracts=changed,
        )
        != first
    )
    assert (
        build_remote_run_fingerprint(
            resolved_training_config=cfg,
            world_size=3,
            selection_contracts=contracts,
        )
        != first
    )


def _remote_ddp_coordination_target(rank: int, world_size: int, scenario: str):
    if scenario == "failure":
        try:
            ddp_collective_result(
                local_error=(OSError("rank-one-save-failed") if rank == 1 else None),
                world_size=world_size,
                rank=rank,
            )
        except RemoteTrainingError as exc:
            return str(exc)
        raise AssertionError("all ranks must observe the rank-one failure")
    if scenario == "publisher_init_failure":
        try:
            ddp_create_checkpoint_publisher(
                store=None,
                world_size=world_size,
                rank=rank,
            )
        except RemoteTrainingError as exc:
            return str(exc)
        raise AssertionError("all ranks must observe the rank-zero publisher failure")
    ddp_collective_result(local_error=None, world_size=world_size, rank=rank)
    return ddp_rank0_action(
        action=lambda: {"committed_step": 17},
        world_size=world_size,
        rank=rank,
    )


def _async_checkpoint_ddp_target(
    rank: int,
    world_size: int,
    root: Path,
    scenario: str,
) -> dict[str, Any]:
    checkpoint_root = root / "checkpoints-local"
    checkpoint_dir = checkpoint_root / "model" / "leaf"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def _save_group(step: int, *, byte_offset: int = 0) -> None:
        local_error: BaseException | None = None
        try:
            save_checkpoint(
                str(checkpoint_dir),
                step,
                {"weight": t.arange(6).reshape(2, 3) + step + byte_offset},
                {"state": {rank: {"step": t.tensor(step + byte_offset)}}},
                {
                    "step": step,
                    "byte_offset": byte_offset,
                    REMOTE_RUN_META_KEY: "d" * 64,
                },
                rank=rank,
            )
        except BaseException as exc:
            local_error = exc
        ddp_collective_result(
            local_error=local_error,
            world_size=world_size,
            rank=rank,
        )

    _save_group(17)

    release = threading.Event()
    bucket = BlockingUploadBucket(
        root / "bucket",
        release=release,
        fail_after_release=(scenario in {"failure", "recovery"}),
        block_suffix=("remote_complete_000017.json" if scenario == "recovery" else None),
        fail_suffix=("remote_complete_000017.json" if scenario == "recovery" else None),
    )
    store = RemoteCheckpointStore(
        backend=bucket,
        remote=_remote_config(),
        checkpoint_dir=checkpoint_dir,
        checkpoint_root=checkpoint_root,
        remote_run_fingerprint="d" * 64,
        world_size=world_size,
    )
    publisher = ddp_create_checkpoint_publisher(
        store=store,
        world_size=world_size,
        rank=rank,
    )

    try:
        assert ddp_publish_saved_checkpoint(
            store=store,
            publisher=publisher,
            async_upload=True,
            step=17,
            world_size=world_size,
            rank=rank,
        )
        advanced: list[bool | None] = [None] * world_size
        dist.all_gather_object(advanced, True)

        def _release_after_observing_progress() -> dict[str, bool]:
            assert bucket.upload_started.wait(timeout=5)
            marker = bucket._path(store.remote_dir + "/remote_complete_000017.json")
            step_18_absent = not any(
                (checkpoint_dir / name).exists()
                for name in remote_module._checkpoint_names(18, world_size)
            )
            release.set()
            return {
                "marker_absent": not marker.exists(),
                "step_18_absent": step_18_absent,
            }

        observation = ddp_rank0_action(
            action=_release_after_observing_progress,
            world_size=world_size,
            rank=rank,
        )
        try:
            ddp_drain_checkpoint_publisher(
                publisher=publisher,
                world_size=world_size,
                rank=rank,
            )
        except RemoteTrainingError as exc:
            if scenario == "recovery":
                if publisher is not None:
                    publisher.close()
                _save_group(17, byte_offset=1000)
                recovered_store = RemoteCheckpointStore(
                    backend=FakeBucket(root / "bucket"),
                    remote=_remote_config(),
                    checkpoint_dir=checkpoint_dir,
                    checkpoint_root=checkpoint_root,
                    remote_run_fingerprint="d" * 64,
                    world_size=world_size,
                )
                recovered_publisher = ddp_create_checkpoint_publisher(
                    store=recovered_store,
                    world_size=world_size,
                    rank=rank,
                )
                try:
                    assert ddp_publish_saved_checkpoint(
                        store=recovered_store,
                        publisher=recovered_publisher,
                        async_upload=True,
                        step=17,
                        world_size=world_size,
                        rank=rank,
                    )
                    ddp_drain_checkpoint_publisher(
                        publisher=recovered_publisher,
                        world_size=world_size,
                        rank=rank,
                    )
                finally:
                    if recovered_publisher is not None:
                        recovered_publisher.close()
                marker = parse_checkpoint_marker(
                    json.loads(
                        (checkpoint_dir / "remote_complete_000017.json").read_text(encoding="utf-8")
                    ),
                    surface="recovered DDP marker",
                )
                return {
                    "advanced": advanced,
                    **observation,
                    "initial_error": str(exc),
                    "recovered": bucket._path(
                        store.remote_dir + "/remote_complete_000017.json"
                    ).is_file(),
                    "generation_model": bucket._path(
                        _generation_remote_path(
                            recovered_store,
                            marker,
                            "model_000017.pt",
                        )
                    ).is_file(),
                }
            return {
                "advanced": advanced,
                **observation,
                "step_18_saved": any(
                    (checkpoint_dir / name).exists()
                    for name in remote_module._checkpoint_names(18, world_size)
                ),
                "error": str(exc),
            }

        _save_group(18)
        assert ddp_publish_saved_checkpoint(
            store=store,
            publisher=publisher,
            async_upload=True,
            step=18,
            world_size=world_size,
            rank=rank,
        )
        ddp_drain_checkpoint_publisher(
            publisher=publisher,
            world_size=world_size,
            rank=rank,
        )
        return {
            "advanced": advanced,
            **observation,
            "step_18_saved": all(
                (checkpoint_dir / name).is_file()
                for name in remote_module._checkpoint_names(18, world_size)
            ),
            "committed_17": bucket._path(
                store.remote_dir + "/remote_complete_000017.json"
            ).is_file(),
            "committed_18": bucket._path(
                store.remote_dir + "/remote_complete_000018.json"
            ).is_file(),
        }
    finally:
        if publisher is not None:
            publisher.close()


@pytest.mark.parametrize("scenario", ["success", "failure", "publisher_init_failure"])
def test_two_rank_gloo_checkpoint_coordination_has_no_deadlock(
    gloo_multiprocess,
    scenario: str,
):
    results = gloo_multiprocess(2, _remote_ddp_coordination_target, scenario)
    if scenario == "success":
        assert results == [
            {"committed_step": 17},
            {"committed_step": 17},
        ]
    elif scenario == "failure":
        assert len(results) == 2
        assert all("rank-one-save-failed" in result for result in results)
    else:
        assert len(results) == 2
        assert all("missing its remote checkpoint store" in result for result in results)


@pytest.mark.parametrize("scenario", ["success", "failure", "recovery"])
def test_two_rank_gloo_async_checkpoint_publication_is_symmetric_and_nonblocking(
    tmp_path: Path,
    gloo_multiprocess,
    scenario: str,
) -> None:
    results = gloo_multiprocess(
        2,
        _async_checkpoint_ddp_target,
        tmp_path,
        scenario,
    )
    assert all(result["advanced"] == [True, True] for result in results)
    assert all(result["marker_absent"] is True for result in results)
    assert all(result["step_18_absent"] is True for result in results)
    if scenario == "success":
        assert all(result["step_18_saved"] is True for result in results)
        assert all(result["committed_17"] is True for result in results)
        assert all(result["committed_18"] is True for result in results)
    elif scenario == "failure":
        assert all(result["step_18_saved"] is False for result in results)
        assert results[0]["error"] == results[1]["error"]
        assert "asynchronous checkpoint upload failure" in results[0]["error"]
    else:
        assert all(result["recovered"] is True for result in results)
        assert all(result["generation_model"] is True for result in results)
        assert results[0]["initial_error"] == results[1]["initial_error"]
        assert "asynchronous checkpoint upload failure" in results[0]["initial_error"]


@pytest.mark.slow
@pytest.mark.api
def test_real_throwaway_bucket_verified_single_object_roundtrip(tmp_path: Path):
    bucket_id = os.environ.get("HF_REMOTE_TRAINING_TEST_BUCKET")
    if not bucket_id:
        pytest.skip("set HF_REMOTE_TRAINING_TEST_BUCKET=owner/throwaway-bucket")
    _require_safe_live_test_bucket(
        bucket_id,
        os.environ.get("HF_REMOTE_TRAINING_TEST_BUCKET_WRITE_CONFIRM"),
    )
    test_prefix = f"codex-tests/splice-verified/{uuid.uuid4().hex}"
    backend = PrefixGuardedBucket(
        HfBucketBackend(bucket_id),
        allowed_prefix=test_prefix,
    )
    assert backend.list(test_prefix) == []
    local = tmp_path / "tiny.bin"
    local.write_bytes(b"bounded verified splice transport")
    remote_path = f"{test_prefix}/tiny.bin"

    uploaded = upload_verified_object(
        backend,
        local_path=local,
        remote_path=remote_path,
        timeout_seconds=60.0,
        retry_interval_seconds=1.0,
    )
    downloaded_path = tmp_path / "downloaded.bin"
    downloaded = download_verified_object(
        backend,
        expected=uploaded,
        final_path=downloaded_path,
        timeout_seconds=60.0,
        retry_interval_seconds=1.0,
    )
    assert downloaded == uploaded
    assert downloaded_path.read_bytes() == local.read_bytes()


@pytest.mark.slow
@pytest.mark.api
def test_real_throwaway_bucket_interrupted_changed_byte_recovery(tmp_path: Path):
    bucket_id = os.environ.get("HF_REMOTE_TRAINING_TEST_BUCKET")
    if not bucket_id:
        pytest.skip("set HF_REMOTE_TRAINING_TEST_BUCKET=owner/throwaway-bucket")
    _require_safe_live_test_bucket(
        bucket_id,
        os.environ.get("HF_REMOTE_TRAINING_TEST_BUCKET_WRITE_CONFIRM"),
    )
    nonce = uuid.uuid4().hex
    test_prefix = f"codex-tests/issue-874/{nonce}"
    hf_backend = HfBucketBackend(bucket_id)
    guarded_backend = PrefixGuardedBucket(
        hf_backend,
        allowed_prefix=test_prefix,
    )
    assert guarded_backend.list(test_prefix) == []
    backend = MarkerInterruptingBucket(guarded_backend)
    remote = _remote_config(
        bucket_id=bucket_id,
        checkpoint_prefix=f"{test_prefix}/checkpoints",
    )
    root = tmp_path / "root"
    checkpoint_dir = root / "run"
    checkpoint_dir.mkdir(parents=True)
    save_checkpoint(
        str(checkpoint_dir),
        1,
        {"weight": t.arange(4)},
        {"state": {0: {"step": t.tensor(1)}}, "rng": "interrupted"},
        {
            "step": 1,
            "dataloader_state": {"cursor": 1},
            REMOTE_RUN_META_KEY: "e" * 64,
        },
        rank=0,
    )
    store = RemoteCheckpointStore(
        backend=backend,
        remote=remote,
        checkpoint_dir=checkpoint_dir,
        checkpoint_root=root,
        remote_run_fingerprint="e" * 64,
        world_size=1,
    )
    with pytest.raises(
        RemoteTrainingError,
        match="timed out.*injected interruption before remote marker upload",
    ):
        store.publish(1)
    interrupted_marker = parse_checkpoint_marker(
        json.loads((checkpoint_dir / "remote_complete_000001.json").read_text(encoding="utf-8")),
        surface="interrupted live checkpoint marker",
    )
    interrupted_bytes = {
        entry["name"]: (checkpoint_dir / entry["name"]).read_bytes()
        for entry in interrupted_marker["files"]
    }
    interrupted_remote_paths = {
        entry["name"]: (
            f"{store.remote_dir}/_groups/{interrupted_marker['aggregate_sha256']}/{entry['name']}"
        )
        for entry in interrupted_marker["files"]
    }
    interrupted_infos = {
        name: guarded_backend.stat(remote_path)
        for name, remote_path in interrupted_remote_paths.items()
    }
    assert all(info is not None for info in interrupted_infos.values())
    assert guarded_backend.stat(store.remote_dir + "/remote_complete_000001.json") is None
    assert store.latest_complete_step() is None

    save_checkpoint(
        str(checkpoint_dir),
        1,
        {"weight": t.arange(4) + 100},
        {"state": {0: {"step": t.tensor(101)}}, "rng": "recovered"},
        {
            "step": 1,
            "dataloader_state": {"cursor": 101},
            REMOTE_RUN_META_KEY: "e" * 64,
        },
        rank=0,
    )
    backend.interrupt_markers = False
    marker = store.publish(1)

    assert marker["aggregate_sha256"] != interrupted_marker["aggregate_sha256"]
    assert guarded_backend.successful_uploads[-1] == (
        store.remote_dir + "/remote_complete_000001.json"
    )
    for name, remote_path in interrupted_remote_paths.items():
        assert guarded_backend.stat(remote_path) == interrupted_infos[name]
        downloaded = tmp_path / "interrupted" / name
        downloaded.parent.mkdir(parents=True, exist_ok=True)
        guarded_backend.download(remote_path, downloaded)
        assert downloaded.read_bytes() == interrupted_bytes[name]
    for entry in marker["files"]:
        generation_path = f"{store.remote_dir}/_groups/{marker['aggregate_sha256']}/{entry['name']}"
        assert guarded_backend.stat(generation_path) is not None
    assert store.latest_complete_step() == 1

    fresh_root = tmp_path / "fresh"
    fresh_store = RemoteCheckpointStore(
        backend=guarded_backend,
        remote=remote,
        checkpoint_dir=fresh_root / "run",
        checkpoint_root=fresh_root,
        remote_run_fingerprint="e" * 64,
        world_size=1,
    )
    fresh_store.restore(1)
    for entry in marker["files"]:
        assert (checkpoint_dir / entry["name"]).read_bytes() == (
            fresh_store.checkpoint_dir / entry["name"]
        ).read_bytes()
    model, optimizer, metadata = load_checkpoint(
        str(fresh_store.checkpoint_dir),
        1,
        "cpu",
        load_optimizer=True,
        rank=0,
    )
    assert t.equal(model["weight"], t.arange(4) + 100)
    assert optimizer["state"][0]["step"].item() == 101
    assert optimizer["rng"] == "recovered"
    assert metadata["dataloader_state"] == {"cursor": 101}
    logger.info(
        "live issue-874 recovery artifacts retained for inspection: "
        f"bucket={bucket_id} prefix={test_prefix}"
    )
