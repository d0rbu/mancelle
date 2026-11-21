"""Contracts for the bounded token-boundary HF transport."""

from __future__ import annotations

import fcntl
import os
import shutil
import threading
from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import hf_xet
import pytest

import steering.token_boundary_remote as remote_module
from steering.remote_feature_inventory import RemoteShardFamily
from steering.remote_training import RemoteObjectInfo
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.token_boundary_remote import (
    RemoteTokenBoundaryTransport,
    TokenBoundaryRemoteConfig,
    TokenBoundaryRemoteError,
    remote_output_lock,
    required_staging_capacity_bytes,
)
from steering.token_boundary_source import (
    TokenBoundarySourceManifest,
    serialize_token_boundary_source_manifest,
)


def _stale_size_publication_wrapper(real_download, *, size_delta: int = -1):
    """Wrap download_verified_object so the publication stat under-reports size."""

    def wrapped(*args, **kwargs):
        real_callback = kwargs["verified_publication_callback"]

        def doctored(published: os.stat_result) -> None:
            real_callback(
                os.stat_result(
                    (
                        published.st_mode,
                        published.st_ino,
                        published.st_dev,
                        published.st_nlink,
                        published.st_uid,
                        published.st_gid,
                        published.st_size + size_delta,
                        published.st_atime,
                        published.st_mtime,
                        published.st_ctime,
                    ),
                    {
                        "st_atime_ns": published.st_atime_ns,
                        "st_mtime_ns": published.st_mtime_ns,
                        "st_ctime_ns": published.st_ctime_ns,
                    },
                )
            )

        kwargs["verified_publication_callback"] = doctored
        return real_download(*args, **kwargs)

    return wrapped


class FakeBucket:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.downloads: list[str] = []
        self.list_calls = 0
        self.mutate_on_second_list = False
        self.block_after_download: dict[str, threading.Event] = {}

    def _path(self, remote_path: str) -> Path:
        path = self.root / remote_path
        assert path.resolve().is_relative_to(self.root.resolve())
        return path

    @staticmethod
    def _info(path: Path, remote_path: str) -> RemoteObjectInfo:
        xet = cast(Any, hf_xet).hash_files([str(path)])[0]
        return RemoteObjectInfo(remote_path, path.stat().st_size, str(xet.hash))

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        path = self._path(remote_path)
        return self._info(path, remote_path) if path.is_file() else None

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        self.list_calls += 1
        base = self._path(prefix)
        if self.mutate_on_second_list and self.list_calls == 2:
            (base / "shard_00003_features.safetensors.zst").write_bytes(b"late")
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"), reverse=True)
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)
        self.downloads.append(remote_path)
        event = self.block_after_download.get(remote_path)
        if event is not None:
            event.wait(timeout=5)

    def upload(self, local_path: Path, remote_path: str) -> None:
        raise AssertionError("token-boundary transport must never upload")


def _setup(tmp_path: Path, *, count: int = 3) -> tuple[FakeBucket, str]:
    bucket = FakeBucket(tmp_path / "bucket")
    prefix = "features/family"
    root = bucket._path(prefix)
    root.mkdir(parents=True)
    for index in range(count):
        (root / f"shard_{index:05d}_features.safetensors.zst").write_bytes(
            bytes([index + 1]) * (index + 3)
        )
    return bucket, prefix


def _config(tmp_path: Path, prefix: str, *, expected: int | None = 3):
    return TokenBoundaryRemoteConfig(
        bucket_id=HfBucketId.parse("owner/test-bucket"),
        remote_prefix=NormalizedRemotePrefix.parse(prefix),
        staging_dir=tmp_path / "staging",
        expected_shard_count=expected,
        retry_timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )


def test_snapshot_orders_numerically_and_requires_exact_family(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    assert [entry.index for entry in transport.inventory] == [0, 1, 2]
    assert len(transport.inventory_sha256) == 64

    bucket._path(f"{prefix}/shard_00001_features.safetensors.zst").unlink()
    with pytest.raises(TokenBoundaryRemoteError, match=r"missing=\[1\]"):
        RemoteTokenBoundaryTransport.snapshot(
            _config(tmp_path, prefix),
            backend=bucket,
        )


def test_snapshot_accepts_overlay_leaf_names_and_retains_control_listing(
    tmp_path: Path,
) -> None:
    bucket = FakeBucket(tmp_path / "bucket")
    prefix = "features/family/pct/checksum"
    root = bucket._path(prefix)
    root.mkdir(parents=True)
    for index in range(3):
        (root / f"shard_{index:05d}.safetensors.zst").write_bytes(bytes([index + 1]))
    (root / "manifest.json").write_text("{}")

    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
        family=RemoteShardFamily.OVERLAY_LEAF,
    )
    assert [entry.index for entry in transport.inventory] == [0, 1, 2]
    assert [
        info.path for info in transport.snapshot_listing if info.path.endswith("manifest.json")
    ] == [f"{prefix}/manifest.json"]


def test_snapshot_rejects_empty_family_before_download(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path, count=0)
    with pytest.raises(TokenBoundaryRemoteError, match="no direct token-boundary input"):
        RemoteTokenBoundaryTransport.snapshot(
            _config(tmp_path, prefix, expected=1),
            backend=bucket,
        )
    assert bucket.downloads == []


def test_snapshot_constructs_default_hf_backend(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
    )
    assert transport.backend is bucket


@pytest.mark.parametrize("expected", [True, False, 0, -1, 1.5])
def test_config_rejects_invalid_expected_count(tmp_path: Path, expected: object) -> None:
    with pytest.raises(ValueError, match="expected_shard_count"):
        TokenBoundaryRemoteConfig(
            bucket_id=HfBucketId.parse("owner/test-bucket"),
            remote_prefix=NormalizedRemotePrefix.parse("features/family"),
            staging_dir=tmp_path / "staging",
            expected_shard_count=cast(Any, expected),
            retry_timeout_seconds=1.0,
            retry_interval_seconds=0.1,
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("retry_timeout_seconds", True),
        ("retry_timeout_seconds", 0.0),
        ("retry_timeout_seconds", float("inf")),
        ("retry_interval_seconds", "1"),
        ("retry_interval_seconds", float("nan")),
    ],
)
def test_config_rejects_invalid_retry_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    kwargs: dict[str, object] = {
        "bucket_id": HfBucketId.parse("owner/test-bucket"),
        "remote_prefix": NormalizedRemotePrefix.parse("features/family"),
        "staging_dir": tmp_path / "staging",
        "expected_shard_count": 1,
        "retry_timeout_seconds": 1.0,
        "retry_interval_seconds": 0.1,
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        TokenBoundaryRemoteConfig(**cast(Any, kwargs))


def test_null_expected_count_warns_and_still_requires_contiguity(
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bucket, prefix = _setup(tmp_path)
    RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=None),
        backend=bucket,
    )
    assert any("EXPECTED COUNT IS NULL" in record.message for record in caplog.records)


def test_one_lookahead_and_identity_bound_release(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path)
    second_path = f"{prefix}/shard_00001_features.safetensors.zst"
    second_ready = threading.Event()
    unblock_second = threading.Event()

    original_download = bucket.download

    def observed_download(remote_path: str, local_path: Path) -> None:
        original_download(remote_path, local_path)
        if remote_path == second_path:
            second_ready.set()
            unblock_second.wait(timeout=5)

    monkeypatch.setattr(bucket, "download", observed_download)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    with transport:
        iterator = transport.stream()
        first = next(iterator)
        assert first.index == 0
        assert second_ready.wait(timeout=5)
        assert all("shard_00002" not in path for path in bucket.downloads)
        unblock_second.set()
        remaining = list(iterator)
        assert [prepared.index for prepared in remaining] == [1, 2]
    assert list((tmp_path / "staging").iterdir()) == []
    profile = transport.profile_payload()
    assert int(cast(int, profile["staging_high_water_bodies"])) <= 2
    assert profile["downloaded_bytes"] == transport.inventory_bytes


def test_download_retry_and_successful_final_fence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    original_download = bucket.download
    attempts = 0

    def flaky_download(remote_path: str, local_path: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            shutil.copyfile(bucket._path(remote_path), local_path)
            raise OSError("transient transport failure")
        original_download(remote_path, local_path)

    monkeypatch.setattr(bucket, "download", flaky_download)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    with transport:
        assert [item.index for item in transport.stream()] == [0]
        transport.final_inventory_fence()
    profile = transport.profile_payload()
    assert attempts == 2
    assert profile["downloaded_bytes"] == 2 * transport.inventory_bytes
    assert profile["final_fence_status"] == "passed"
    assert profile["terminal_error"] is None


def test_failed_download_records_materialized_bytes_wait_and_terminal_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)

    def fail_after_materialization(remote_path: str, local_path: Path) -> None:
        shutil.copyfile(bucket._path(remote_path), local_path)
        raise OSError("simulated transport failure")

    monkeypatch.setattr(bucket, "download", fail_after_materialization)
    config = TokenBoundaryRemoteConfig(
        bucket_id=HfBucketId.parse("owner/test-bucket"),
        remote_prefix=NormalizedRemotePrefix.parse(prefix),
        staging_dir=tmp_path / "staging",
        expected_shard_count=1,
        retry_timeout_seconds=0.01,
        retry_interval_seconds=0.001,
    )
    transport = RemoteTokenBoundaryTransport.snapshot(config, backend=bucket)

    with pytest.raises(RuntimeError, match="timed out"):
        with transport:
            list(transport.stream())
    profile = transport.profile_payload()
    assert int(cast(int, profile["downloaded_bytes"])) > 0
    assert float(cast(float, profile["total_transfer_seconds"])) > 0.0
    assert float(cast(float, profile["critical_path_wait_seconds"])) > 0.0
    assert cast(dict[str, str], profile["terminal_error"])["type"] == ("RemoteTrainingError")


def test_consumer_failure_cancels_and_removes_only_run_directory(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    orphan = staging / ".token-boundary-remote.orphan"
    orphan.mkdir()
    (orphan / "keep").write_bytes(b"keep")
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    with pytest.raises(RuntimeError, match="consumer failed"):
        with transport:
            for _prepared in transport.stream():
                raise RuntimeError("consumer failed")
    assert (orphan / "keep").read_bytes() == b"keep"
    assert sorted(path.name for path in staging.iterdir()) == [orphan.name]


def test_replaced_staged_body_fails_release(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    with pytest.raises(TokenBoundaryRemoteError, match="changed while consumed"):
        with transport:
            prepared = next(transport.stream())
            replacement = prepared.local_path.with_name("replacement")
            replacement.write_bytes(b"replacement")
            replacement.replace(prepared.local_path)


def test_missing_staged_body_fails_release_and_records_terminal_error(
    tmp_path: Path,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    with pytest.raises(TokenBoundaryRemoteError, match="cannot inspect consumed"):
        with transport:
            prepared = next(transport.stream())
            prepared.local_path.unlink()
    terminal = cast(dict[str, str], transport.profile_payload()["terminal_error"])
    assert terminal["type"] == "TokenBoundaryRemoteError"


def test_transport_rejects_nonregular_identity_and_helper_contract_violations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    hardlinked = tmp_path / "hardlinked"
    hardlinked.write_bytes(b"body")
    (tmp_path / "hardlinked-copy").hardlink_to(hardlinked)
    with pytest.raises(TokenBoundaryRemoteError, match="single-link"):
        remote_module._capture_direct_single_link(hardlinked, surface="test body")

    bucket, prefix = _setup(tmp_path / "identity", count=1)
    config = _config(tmp_path / "identity", prefix, expected=1)

    def invoke(mode: str) -> None:
        transport = RemoteTokenBoundaryTransport.snapshot(config, backend=bucket)
        run_dir = tmp_path / f"run-{mode}"
        run_dir.mkdir()
        transport._run_dir = run_dir

        def broken_download(*_args: object, **kwargs: object) -> None:
            final_path = cast(Path, kwargs["final_path"])
            callback = cast(Any, kwargs["verified_publication_callback"])
            if mode == "nonregular":
                directory = tmp_path / "published-directory"
                directory.mkdir(exist_ok=True)
                callback(directory.stat())
                return
            final_path.write_bytes(b"body")
            if mode == "missing-callback":
                return
            callback(final_path.stat())
            # Create the replacement while the original still exists so the swap
            # is visible as an inode change: size drift alone is advisory (stale
            # Lustre stats), and unlink-then-create can recycle the inode number.
            replacement = final_path.with_name(f"{final_path.name}.replacement")
            replacement.write_bytes(b"replacement")
            os.replace(replacement, final_path)

        monkeypatch.setattr(remote_module, "download_verified_object", broken_download)
        pattern = {
            "nonregular": "direct single-link",
            "missing-callback": "no bound file identity",
            "changed": "changed after publication",
        }[mode]
        with pytest.raises(TokenBoundaryRemoteError, match=pattern):
            transport._download(0)

    for mode in ("nonregular", "missing-callback", "changed"):
        invoke(mode)


@pytest.mark.parametrize(
    ("mode", "pattern"),
    [
        ("nonregular", "single-link regular file"),
        ("missing-callback", "no file identity"),
    ],
)
def test_source_manifest_download_requires_bound_regular_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    pattern: str,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    config = _config(tmp_path, prefix, expected=1)
    manifest_info = RemoteObjectInfo(
        f"{prefix}/manifest.json",
        1,
        "a" * 64,
    )

    def broken_download(*_args: object, **kwargs: object) -> None:
        if mode == "missing-callback":
            return
        directory = tmp_path / "published-manifest-directory"
        directory.mkdir()
        callback = cast(Any, kwargs["verified_publication_callback"])
        callback(directory.stat())

    monkeypatch.setattr(remote_module, "download_verified_object", broken_download)
    with pytest.raises(TokenBoundaryRemoteError, match=pattern):
        remote_module._download_source_manifest(
            backend=bucket,
            manifest_info=manifest_info,
            config=config,
        )


def test_staged_body_binds_verified_length_over_stale_published_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bucket, prefix = _setup(tmp_path)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    monkeypatch.setattr(
        remote_module,
        "download_verified_object",
        _stale_size_publication_wrapper(remote_module.download_verified_object),
    )

    with transport:
        for prepared in transport.stream():
            expected_length = transport.inventory[prepared.index].info.byte_length
            assert prepared.file_identity.size == expected_length
            assert prepared.local_path.stat().st_size == expected_length

    warnings = [
        record.message
        for record in caplog.records
        if "binding the verified length" in record.message
    ]
    assert len(warnings) == 3


def test_source_manifest_binds_verified_length_over_stale_published_stat(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    config = _config(tmp_path, prefix, expected=1)
    payload = serialize_token_boundary_source_manifest(
        TokenBoundarySourceManifest(
            base_shard_content_hash="a" * 64,
            raw_inventory_sha256="b" * 64,
            schema_version=1,
            shard_count=1,
            source_inventory_sha256="c" * 64,
        )
    )
    manifest_path = f"{prefix}/source-manifest"
    bucket._path(manifest_path).write_bytes(payload)
    manifest_xet = str(cast(Any, hf_xet).hash_files([str(bucket._path(manifest_path))])[0].hash)
    manifest_info = RemoteObjectInfo(manifest_path, len(payload), manifest_xet)
    monkeypatch.setattr(
        remote_module,
        "download_verified_object",
        _stale_size_publication_wrapper(remote_module.download_verified_object),
    )

    manifest = remote_module._download_source_manifest(
        backend=bucket,
        manifest_info=manifest_info,
        config=config,
    )

    assert manifest.shard_count == 1
    assert any("binding the verified length" in record.message for record in caplog.records)


def test_recording_backend_proxies_protocol_and_observes_nonregular_output(
    tmp_path: Path,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    observations = 0

    def observe() -> None:
        nonlocal observations
        observations += 1

    recording = remote_module._RecordingBackend(bucket, observe=observe)
    remote_path = f"{prefix}/shard_00000_features.safetensors.zst"
    assert recording.stat(remote_path) == bucket.stat(remote_path)
    assert recording.list(prefix) == bucket.list(prefix)
    with pytest.raises(AssertionError, match="never upload"):
        recording.upload(tmp_path / "unused", "unused")

    class DirectoryDownloadBucket(FakeBucket):
        def download(self, remote_path: str, local_path: Path) -> None:
            local_path.mkdir()

    directory_bucket = DirectoryDownloadBucket(bucket.root)
    directory_recording = remote_module._RecordingBackend(
        directory_bucket,
        observe=observe,
    )
    local_path = tmp_path / "downloaded-directory"
    directory_recording.download(remote_path, local_path)
    assert directory_recording.materialized_bytes == 0
    assert observations == 1

    class MissingDownloadBucket(FakeBucket):
        def download(self, remote_path: str, local_path: Path) -> None:
            raise OSError("failed before materialization")

    missing_recording = remote_module._RecordingBackend(
        MissingDownloadBucket(bucket.root),
        observe=observe,
    )
    with pytest.raises(OSError, match="before materialization"):
        missing_recording.download(remote_path, tmp_path / "never-created")
    assert observations == 2


def test_capacity_formula_and_admission_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    sizes = sorted(
        (entry.info.byte_length for entry in transport.inventory),
        reverse=True,
    )
    assert required_staging_capacity_bytes(()) == 1 << 30
    assert transport.required_capacity == sizes[0] + sizes[1] + (1 << 30)
    monkeypatch.setattr(
        remote_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=transport.required_capacity - 1),
    )
    with pytest.raises(TokenBoundaryRemoteError, match="admission failed"):
        with transport:
            pass
    assert bucket.downloads == []


def test_staging_must_be_a_real_directory(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    staging = tmp_path / "staging"
    staging.write_text("not a directory")
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    with pytest.raises(TokenBoundaryRemoteError, match="must be a real directory"):
        with transport:
            pass
    assert bucket.downloads == []


def test_enter_failure_records_terminal_error_and_cleans_partial_run_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )

    def fail_executor(**_kwargs: object) -> None:
        raise RuntimeError("executor construction failed")

    monkeypatch.setattr(remote_module, "ThreadPoolExecutor", fail_executor)
    with pytest.raises(RuntimeError, match="executor construction failed"):
        with transport:
            pass
    assert list((tmp_path / "staging").iterdir()) == []
    terminal = cast(dict[str, str], transport.profile_payload()["terminal_error"])
    assert terminal == {
        "type": "RuntimeError",
        "message": "executor construction failed",
    }


def test_staging_symlink_swap_after_creation_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    staging = tmp_path / "staging"
    real_staging = tmp_path / "real-staging"
    real_staging.mkdir()
    original_mkdir = Path.mkdir

    def swapped_mkdir(
        path: Path,
        mode: int = 0o777,
        parents: bool = False,
        exist_ok: bool = False,
    ) -> None:
        original_mkdir(path, mode=mode, parents=parents, exist_ok=exist_ok)
        if path == staging and not path.is_symlink():
            path.rmdir()
            path.symlink_to(real_staging, target_is_directory=True)

    monkeypatch.setattr(Path, "mkdir", swapped_mkdir)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    with pytest.raises(TokenBoundaryRemoteError, match="must be a real directory"):
        with transport:
            pass


def test_transport_rejects_inactive_or_duplicate_stream(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    with pytest.raises(TokenBoundaryRemoteError, match="not active"):
        next(transport.stream())
    with transport:
        iterator = transport.stream()
        with pytest.raises(TokenBoundaryRemoteError, match="already active"):
            transport.stream()
        assert [item.index for item in iterator] == [0]


def test_transport_stream_stop_position_bounds_downloads(tmp_path: Path) -> None:
    """stop_position bounds the stream without touching the sealed inventory."""
    bucket, prefix = _setup(tmp_path, count=3)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=3),
        backend=bucket,
    )
    sealed = transport.inventory
    with transport:
        iterator = transport.stream(stop_position=2)
        assert [item.index for item in iterator] == [0, 1]
    assert transport.inventory == sealed
    payload = transport.profile_payload()
    assert payload["inventory_count"] == 3
    records: Any = payload["per_body_transport"]
    assert [record["position"] for record in records] == [0, 1]


def test_transport_stream_start_position_seeks_without_prior_downloads(tmp_path: Path) -> None:
    """A resumed consumer starts at an absolute sealed-inventory position."""
    bucket, prefix = _setup(tmp_path, count=4)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=4),
        backend=bucket,
    )
    sealed = transport.inventory
    with transport:
        iterator = transport.stream(start_position=2, stop_position=4)
        assert [item.index for item in iterator] == [2, 3]
    assert transport.inventory == sealed
    assert bucket.downloads == [
        f"{prefix}/shard_00002_features.safetensors.zst",
        f"{prefix}/shard_00003_features.safetensors.zst",
    ]
    payload = transport.profile_payload()
    records: Any = payload["per_body_transport"]
    assert [record["position"] for record in records] == [2, 3]


@pytest.mark.parametrize("stop_position", [0, -1, 4, True])
def test_transport_stream_rejects_invalid_stop_position(
    tmp_path: Path,
    stop_position: object,
) -> None:
    bucket, prefix = _setup(tmp_path, count=3)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=3),
        backend=bucket,
    )
    with transport:
        with pytest.raises(TokenBoundaryRemoteError, match="stream bounds must satisfy"):
            transport.stream(stop_position=stop_position)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("start_position", [-1, 3, True])
def test_transport_stream_rejects_invalid_start_position(
    tmp_path: Path,
    start_position: object,
) -> None:
    bucket, prefix = _setup(tmp_path, count=3)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=3),
        backend=bucket,
    )
    with transport:
        with pytest.raises(TokenBoundaryRemoteError, match="start_position|stop_position"):
            transport.stream(start_position=start_position)  # ty: ignore[invalid-argument-type]


def test_transport_context_without_stream_and_owned_cleanup_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    transport._observe_staging()
    with transport:
        assert transport._run_dir is not None
        (transport._run_dir / "nested").mkdir()
        transport._observe_staging()
        unreadable = transport._run_dir / "unreadable"
        unreadable.write_bytes(b"body")
        original_lstat = Path.lstat

        def failing_lstat(path: Path):
            if path == unreadable:
                raise OSError("simulated disappearing staging entry")
            return original_lstat(path)

        with monkeypatch.context() as local_patch:
            local_patch.setattr(Path, "lstat", failing_lstat)
            transport._observe_staging()
    transport._cleanup_run_dir()

    unsafe = tmp_path / "unsafe-cleanup"
    unsafe.mkdir()
    transport._run_dir = unsafe
    with pytest.raises(TokenBoundaryRemoteError, match="refusing cleanup"):
        transport._cleanup_run_dir()
    assert unsafe.is_dir()


def test_exit_preserves_primary_error_and_attaches_both_cleanup_failures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    primary = RuntimeError("primary")
    stop_error = RuntimeError("stop failed")
    cleanup_error = RuntimeError("cleanup failed")
    monkeypatch.setattr(
        transport,
        "_stop_transport",
        lambda: (_ for _ in ()).throw(stop_error),
    )
    monkeypatch.setattr(
        transport,
        "_cleanup_run_dir",
        lambda: (_ for _ in ()).throw(cleanup_error),
    )
    transport.__exit__(RuntimeError, primary, None)
    assert any("stop failed" in note for note in getattr(primary, "__notes__", []))
    assert any(
        "additional staging cleanup failure" in note
        for note in getattr(stop_error, "__notes__", [])
    )


def test_exit_surfaces_cleanup_only_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bucket, prefix = _setup(tmp_path, count=1)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=1),
        backend=bucket,
    )
    monkeypatch.setattr(
        transport,
        "_cleanup_run_dir",
        lambda: (_ for _ in ()).throw(RuntimeError("cleanup only")),
    )
    with pytest.raises(RuntimeError, match="cleanup only"):
        transport.__exit__(None, None, None)
    terminal = cast(dict[str, str], transport.profile_payload()["terminal_error"])
    assert terminal["message"] == "cleanup only"


def test_final_fence_rejects_inventory_mutation(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix, expected=None),
        backend=bucket,
    )
    bucket.mutate_on_second_list = True
    with transport:
        list(transport.stream())
        with pytest.raises(TokenBoundaryRemoteError, match="changed before"):
            transport.final_inventory_fence()
    assert transport.profile_payload()["final_fence_status"] == "failed"


def test_final_fence_records_failed_when_family_validation_raises(tmp_path: Path) -> None:
    bucket, prefix = _setup(tmp_path)
    transport = RemoteTokenBoundaryTransport.snapshot(
        _config(tmp_path, prefix),
        backend=bucket,
    )
    with transport:
        list(transport.stream())
        bucket._path(f"{prefix}/shard_00001_features.safetensors.zst").unlink()
        with pytest.raises(TokenBoundaryRemoteError, match=r"missing=\[1\]"):
            transport.final_inventory_fence()
    assert transport.profile_payload()["final_fence_status"] == "failed"


def test_nonblocking_output_lock(tmp_path: Path) -> None:
    path = tmp_path / "output.lock"
    with remote_output_lock(path):
        descriptor = path.open("r+")
        try:
            with pytest.raises(BlockingIOError):
                fcntl.flock(descriptor.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        finally:
            descriptor.close()
        with pytest.raises(TokenBoundaryRemoteError, match="already in use"):
            with remote_output_lock(path):
                pass


def test_output_lock_rejects_hardlinked_file(tmp_path: Path) -> None:
    path = tmp_path / "output.lock"
    path.touch()
    (tmp_path / "other-link").hardlink_to(path)
    with pytest.raises(TokenBoundaryRemoteError, match="single-link"):
        with remote_output_lock(path):
            pass


def test_output_lock_without_nofollow_flag(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delattr(os, "O_NOFOLLOW")
    with remote_output_lock(tmp_path / "portable.lock"):
        pass
