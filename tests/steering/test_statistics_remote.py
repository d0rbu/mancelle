"""Functional contracts for identity-sealed remote feature statistics."""

from __future__ import annotations

import json
import os
import shutil
import socket
import stat
import tempfile
import threading
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import hf_xet
import pytest
from omegaconf import OmegaConf

import steering.compute_statistics as statistics_module
import steering.statistics_remote as remote_module
from steering.compute_statistics import compute_statistics
from steering.feature_io import load_features, save_features
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    build_feature_dir,
)
from steering.remote_training import RemoteObjectInfo
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.sae_resolver import ActivationType, SAEWidth
from steering.sft.mixture_fingerprint import mixture_fingerprint_json
from steering.statistics_remote import (
    RemoteStatisticsConfig,
    RemoteStatisticsError,
    RemoteStatisticsSource,
    required_staging_capacity_bytes,
)
from tests.steering.test_exact_statistics_integration import (
    _assert_safetensors_exact,
    _freeze_volatile_metadata,
    _shard,
    _write_shards,
)


class FakeBucket:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.downloads: list[str] = []
        self.list_calls = 0
        self.append_on_second_list = False

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
        self.list_calls += 1
        base = self._path(prefix)
        if self.append_on_second_list and self.list_calls == 2:
            source = base / "shard_00002_features.safetensors.zst"
            shutil.copyfile(source, base / "shard_00003_features.safetensors.zst")
        return [
            self._info(path, path.relative_to(self.root).as_posix())
            for path in sorted(base.rglob("*"))
            if path.is_file()
        ]

    def download(self, remote_path: str, local_path: Path) -> None:
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self._path(remote_path), local_path)
        self.downloads.append(remote_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        raise AssertionError("statistics transport must never upload")


def _setup_remote(
    tmp_path: Path,
    *,
    writer: Callable[[Path], list[Path]] = _write_shards,
) -> tuple[Path, str, FakeBucket]:
    local_root = tmp_path / "local" / "features"
    feature_dir = build_feature_dir(
        root=local_root,
        segment_size=4,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )
    feature_dir.mkdir(parents=True)
    paths = writer(feature_dir)
    relative = feature_dir.relative_to(local_root).as_posix()
    prefix = f"features/{relative}"
    bucket = FakeBucket(tmp_path / "bucket")
    for path in paths:
        destination = bucket._path(f"{prefix}/{path.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
    return feature_dir, prefix, bucket


def _write_edge_shards(directory: Path) -> list[Path]:
    smallest_f16_subnormal = 2**-24
    shards = [
        _shard(
            [0, 1, 3, 0, 3],
            [smallest_f16_subnormal, 1.0, 2.0, 0.5, 2.0],
            [0, 3, 5],
        ),
        _shard(
            [1, 3, 2, 1],
            [1.0, 4.0, 32768.0, 1.0],
            [0, 1, 4],
        ),
        _shard(
            [0, 3, 1],
            [smallest_f16_subnormal, 8.0, 1.0],
            [0, 3],
        ),
    ]
    return [
        save_features(shard, directory / f"shard_{index:05d}_features")
        for index, shard in enumerate(shards)
    ]


def _config(
    *,
    prefix: str,
    staging: Path,
    expected_count: int | None = 3,
) -> RemoteStatisticsConfig:
    return RemoteStatisticsConfig(
        bucket_id=HfBucketId.parse("owner/test-bucket"),
        remote_prefix=NormalizedRemotePrefix.parse(prefix),
        staging_dir=staging,
        expected_shard_count=expected_count,
        retry_timeout_seconds=1.0,
        retry_interval_seconds=0.001,
        expected_segment_size=4,
    )


@pytest.mark.parametrize("exclude_last_shard", [False, True])
def test_remote_output_matches_local_exactly_and_profiles_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    exclude_last_shard: bool,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    _freeze_volatile_metadata(monkeypatch)

    local_output = tmp_path / "local-statistics"
    eager_output = tmp_path / "eager-statistics"
    remote_output = tmp_path / "remote-statistics"
    eager = compute_statistics(
        str(feature_dir),
        str(eager_output),
        n_sae_features=4,
        streaming=False,
        device="cpu",
        exclude_last_shard=exclude_last_shard,
    )
    local = compute_statistics(
        str(feature_dir),
        str(local_output),
        n_sae_features=4,
        streaming=True,
        device="cpu",
        exclude_last_shard=exclude_last_shard,
    )
    remote = compute_statistics(
        str(feature_dir),
        str(remote_output),
        n_sae_features=4,
        streaming=True,
        device="cpu",
        exclude_last_shard=exclude_last_shard,
        profile_output=str(tmp_path / "remote-profile.json"),
        remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
    )

    assert remote == local == eager
    _assert_safetensors_exact(
        remote_output.with_suffix(".safetensors"),
        local_output.with_suffix(".safetensors"),
    )
    _assert_safetensors_exact(
        remote_output.with_suffix(".safetensors"),
        eager_output.with_suffix(".safetensors"),
    )
    expected_downloads = 2 if exclude_last_shard else 3
    assert len(bucket.downloads) == expected_downloads
    assert bucket.list_calls == 2
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]
    profile = json.loads((tmp_path / "remote-profile.json").read_text())
    assert profile["remote_transport"]["status"] == "complete"
    assert profile["remote_transport"]["full_inventory_count"] == 3
    assert profile["remote_transport"]["statistics_inventory_count"] == (expected_downloads)
    assert profile["remote_transport"]["verified_downloaded_bytes"] > 0
    assert profile["remote_transport"]["staging_high_water_bodies"] <= 2


def test_remote_config_defaults_to_henry_bucket() -> None:
    config_path = Path(__file__).parents[2] / "configs/extraction/compute_statistics.yaml"
    cfg = OmegaConf.load(config_path)
    assert cfg.remote.bucket_id == "henryac/nano-gpt-pretrain-steer"
    assert cfg.remote.remote_features_root == "l22_ci_nobos_raw"
    assert str(cfg.remote.local_features_root).endswith(
        "m_gemma-3-4b-pt/base/context_independent/no_bos"
    )


def test_remote_edge_case_artifact_matches_eager_and_local_exactly(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path, writer=_write_edge_shards)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    _freeze_volatile_metadata(monkeypatch)

    eager_path = tmp_path / "edge-eager"
    local_path = tmp_path / "edge-local"
    remote_path = tmp_path / "edge-remote"
    eager = compute_statistics(
        str(feature_dir),
        str(eager_path),
        n_sae_features=4,
        streaming=False,
        device="cpu",
    )
    local = compute_statistics(
        str(feature_dir),
        str(local_path),
        n_sae_features=4,
        streaming=True,
        device="cpu",
    )
    remote = compute_statistics(
        str(feature_dir),
        str(remote_path),
        n_sae_features=4,
        streaming=True,
        device="cpu",
        remote_config=_config(prefix=prefix, staging=tmp_path / "edge-staging"),
    )

    assert remote == local == eager
    _assert_safetensors_exact(
        remote_path.with_suffix(".safetensors"),
        local_path.with_suffix(".safetensors"),
    )
    _assert_safetensors_exact(
        remote_path.with_suffix(".safetensors"),
        eager_path.with_suffix(".safetensors"),
    )


def test_final_inventory_append_preserves_preexisting_output_and_cleans_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    bucket.append_on_second_list = True
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    output = tmp_path / "statistics.safetensors"
    output.write_bytes(b"sentinel")
    profile = tmp_path / "failed-profile.json"

    with pytest.raises(RemoteStatisticsError, match="contiguous family"):
        cast(Any, compute_statistics)(
            str(feature_dir),
            str(output),
            n_sae_features=4,
            streaming=True,
            device="cpu",
            profile_output=str(profile),
            remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
        )

    assert output.read_bytes() == b"sentinel"
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]
    failed = json.loads(profile.read_text())
    assert failed["remote_transport"]["status"] == "failed"
    assert failed["remote_transport"]["terminal_error"]["stage"] == "final_inventory_fence"


def test_inventory_snapshot_failure_writes_profile_before_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    profile = tmp_path / "snapshot-failure.json"
    staging = tmp_path / "staging"
    with pytest.raises(RemoteStatisticsError, match="contiguous family"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            profile_output=str(profile),
            remote_config=_config(
                prefix=prefix,
                staging=staging,
                expected_count=4,
            ),
        )
    assert not staging.exists()
    failed = json.loads(profile.read_text())["remote_transport"]
    assert failed["status"] == "failed"
    assert failed["terminal_error"]["stage"] == "inventory_snapshot"


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"streaming": False}, "streaming=True"),
        ({"activation_percentile": 0.5}, "activation_percentile=0.0"),
        ({"n_sae_features": None}, "explicit positive n_sae_features"),
        ({"n_sae_features": True}, "explicit positive n_sae_features"),
        ({"n_sae_features": 0}, "explicit positive n_sae_features"),
        ({"_preloaded": object()}, "incompatible with _preloaded"),
    ],
)
def test_invalid_mode_fails_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, object],
    match: str,
) -> None:
    feature_dir, prefix, _bucket = _setup_remote(tmp_path)
    called = False

    def forbidden_snapshot(*_args, **_kwargs):
        nonlocal called
        called = True
        raise AssertionError("backend construction was reached")

    monkeypatch.setattr(remote_module.RemoteStatisticsSource, "snapshot", forbidden_snapshot)
    kwargs: dict[str, object] = {
        "n_sae_features": 4,
        "streaming": True,
        "activation_percentile": 0.0,
    }
    kwargs.update(overrides)
    with pytest.raises(RemoteStatisticsError, match=match):
        cast(Any, compute_statistics)(
            str(feature_dir),
            str(tmp_path / "statistics"),
            remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
            **kwargs,
        )
    assert not called


def test_staging_overlap_fails_before_snapshot(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, _bucket = _setup_remote(tmp_path)
    called = False

    def forbidden_snapshot(*_args, **_kwargs):
        nonlocal called
        called = True

    monkeypatch.setattr(remote_module.RemoteStatisticsSource, "snapshot", forbidden_snapshot)
    with pytest.raises(RemoteStatisticsError, match="disjoint"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            remote_config=_config(prefix=prefix, staging=feature_dir / "staging"),
        )
    assert not called


def test_default_local_mode_never_constructs_backend_or_adds_remote_profile(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _prefix, _bucket = _setup_remote(tmp_path)

    def forbidden_backend(*_args, **_kwargs):
        raise AssertionError("local mode constructed an HF backend")

    monkeypatch.setattr(remote_module, "HfBucketBackend", forbidden_backend)
    profile = tmp_path / "local-profile.json"
    result = compute_statistics(
        str(feature_dir),
        str(tmp_path / "local-statistics"),
        n_sae_features=4,
        profile_output=str(profile),
    )
    assert result.corpus.n_shards_processed == 3
    assert "remote_transport" not in json.loads(profile.read_text())

    disabled = OmegaConf.create(
        {
            "remote": {
                "enabled": False,
                "bucket_id": None,
                "staging_dir": None,
                "expected_shard_count": "malformed but unused",
                "retry_timeout_seconds": "also unused",
            }
        }
    )
    assert statistics_module._remote_statistics_enabled(disabled) is False
    disabled.remote.enabled = "false"
    with pytest.raises(RemoteStatisticsError, match="must be a bool"):
        statistics_module._remote_statistics_enabled(disabled)


def test_completed_checkpoint_resume_downloads_nothing_and_relocates_staging(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    _freeze_volatile_metadata(monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    real_save = statistics_module.save_statistics

    def fail_save(*_args, **_kwargs):
        raise OSError("injected output failure")

    monkeypatch.setattr(statistics_module, "save_statistics", fail_save)
    failed_profile = tmp_path / "failed-output-profile.json"
    with pytest.raises(OSError, match="injected output failure"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "failed-output"),
            n_sae_features=4,
            checkpoint_dir=str(checkpoint),
            checkpoint_interval_shards=1,
            profile_output=str(failed_profile),
            remote_config=_config(
                prefix=prefix,
                staging=tmp_path / "first-staging",
                expected_count=None,
            ),
        )
    assert (checkpoint / "manifest.json").is_file()
    assert (
        json.loads(failed_profile.read_text())["remote_transport"]["terminal_error"]["stage"]
        == "output_serialize"
    )

    bucket.downloads.clear()
    monkeypatch.setattr(statistics_module, "save_statistics", real_save)
    relocated = tmp_path / "relocated-staging"
    result = compute_statistics(
        str(feature_dir),
        str(tmp_path / "resumed-output"),
        n_sae_features=4,
        checkpoint_dir=str(checkpoint),
        checkpoint_interval_shards=1,
        remote_config=RemoteStatisticsConfig(
            bucket_id=HfBucketId.parse("owner/test-bucket"),
            remote_prefix=NormalizedRemotePrefix.parse(prefix),
            staging_dir=relocated,
            expected_shard_count=3,
            retry_timeout_seconds=2.0,
            retry_interval_seconds=0.002,
            expected_segment_size=4,
        ),
    )
    assert result.corpus.n_shards_processed == 3
    assert bucket.downloads == []
    assert not relocated.exists()
    assert sorted(path.name for path in checkpoint.iterdir()) == ["checkpoint.lock"]


def test_intermediate_resume_starts_at_committed_cursor(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    checkpoint = tmp_path / "checkpoint"
    real_accumulate = statistics_module.accumulate_shard
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected reducer failure")
        return real_accumulate(*args, **kwargs)

    monkeypatch.setattr(statistics_module, "accumulate_shard", fail_second)
    with pytest.raises(RuntimeError, match="injected reducer failure"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "failed-output"),
            n_sae_features=4,
            checkpoint_dir=str(checkpoint),
            checkpoint_interval_shards=1,
            remote_config=_config(prefix=prefix, staging=tmp_path / "first-staging"),
        )
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    assert manifest["processed_shards"] == 1

    monkeypatch.setattr(statistics_module, "accumulate_shard", real_accumulate)
    bucket.downloads.clear()
    compute_statistics(
        str(feature_dir),
        str(tmp_path / "resumed-output"),
        n_sae_features=4,
        checkpoint_dir=str(checkpoint),
        checkpoint_interval_shards=1,
        remote_config=_config(prefix=prefix, staging=tmp_path / "second-staging"),
    )
    assert all("shard_00000_" not in path for path in bucket.downloads)
    assert {Path(path).name for path in bucket.downloads} == {
        "shard_00001_features.safetensors.zst",
        "shard_00002_features.safetensors.zst",
    }


def _owner_payload(prefix: str) -> dict[str, object]:
    return {
        "schema_version": 1,
        "bucket_id": "owner/test-bucket",
        "remote_prefix": prefix,
    }


def test_valid_staged_body_is_reused_without_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(_owner_payload(prefix)))
    name = "shard_00000_features.safetensors.zst"
    shutil.copyfile(bucket._path(f"{prefix}/{name}"), staging / name)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    real_verify = remote_module.verified_local_object_info
    verification_calls = 0

    def count_verification(*, local_path: Path, remote_path: str):
        nonlocal verification_calls
        verification_calls += 1
        return real_verify(local_path=local_path, remote_path=remote_path)

    monkeypatch.setattr(
        remote_module,
        "verified_local_object_info",
        count_verification,
    )
    prepared = source.prepare(0)
    assert prepared.disposition == "reused"
    assert verification_calls == 1
    assert all("shard_00000_" not in path for path in bucket.downloads)
    source.release(prepared)
    source.finish_transport()


def test_reused_body_identity_is_bound_during_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(_owner_payload(prefix)))
    name = "shard_00000_features.safetensors.zst"
    retained = staging / name
    shutil.copyfile(bucket._path(f"{prefix}/{name}"), retained)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    real_verify = remote_module.verified_local_object_info

    def replace_after_verification(*, local_path: Path, remote_path: str):
        result = real_verify(local_path=local_path, remote_path=remote_path)
        replacement = local_path.with_name("same-bytes-replacement")
        shutil.copyfile(local_path, replacement)
        os.replace(replacement, local_path)
        return result

    monkeypatch.setattr(
        remote_module,
        "verified_local_object_info",
        replace_after_verification,
    )
    with pytest.raises(RemoteStatisticsError, match="changed during verification"):
        source.prepare(0)
    assert retained.exists()
    assert bucket.downloads == []
    source.finish_transport()


@pytest.mark.parametrize("unsafe", ["wrong-marker", "unknown", "live-temp"])
def test_unsafe_staging_is_preserved_and_fails_before_download(
    tmp_path: Path,
    unsafe: str,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    owner = _owner_payload(prefix)
    if unsafe == "wrong-marker":
        owner["remote_prefix"] = "features/other"
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(owner))
    if unsafe == "unknown":
        (staging / "canary").write_text("keep")
    elif unsafe == "live-temp":
        (
            staging / f".shard_00000_features.safetensors.zst.remote-tmp.{os.getpid()}.deadbeef"
        ).write_bytes(b"partial")
    before = {path.name: path.read_bytes() for path in staging.iterdir() if path.is_file()}
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError):
        source.prepare(0)
    after = {
        path.name: path.read_bytes()
        for path in staging.iterdir()
        if path.is_file() and path.name != ".statistics-remote.lock"
    }
    assert after == before
    assert bucket.downloads == []


def test_lock_contention_fails_without_adopting_first_sources_files(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    config = _config(prefix=prefix, staging=tmp_path / "staging")
    first = RemoteStatisticsSource.snapshot(
        config,
        exclude_last_shard=False,
        backend=bucket,
    )
    prepared = first.prepare(0)
    second = RemoteStatisticsSource.snapshot(
        config,
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError, match="already in use"):
        second.prepare(0)
    first.release(prepared)
    first.finish_transport()


def test_replaced_consumed_path_is_preserved(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    prepared = source.prepare(0)
    replacement = prepared.local_path.with_name("replacement")
    replacement.write_bytes(b"canary replacement")
    os.replace(replacement, prepared.local_path)
    with pytest.raises(RemoteStatisticsError, match="replaced before unlink"):
        source.release(prepared)
    source.finish_transport()
    assert prepared.local_path.read_bytes() == b"canary replacement"


def test_capacity_formula_and_exact_admission_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    passing = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "passing"),
        exclude_last_shard=False,
        backend=bucket,
    )
    sizes = sorted(
        (entry.info.byte_length for entry in passing.statistics_inventory),
        reverse=True,
    )
    assert required_staging_capacity_bytes(()) == 1 << 30
    assert passing.required_capacity == sizes[0] + sizes[1] + (1 << 30)
    monkeypatch.setattr(
        remote_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=passing.required_capacity),
    )
    passing._ensure_transport(0)
    assert passing._executor is not None
    passing.finish_transport()

    failing = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "failing"),
        exclude_last_shard=False,
        backend=bucket,
    )
    failing.config.staging_dir.mkdir(parents=True)
    (failing.config.staging_dir / ".statistics-remote-owner.json").write_text(
        json.dumps(failing._owner_payload())
    )
    retained = failing.config.staging_dir / "shard_00000_features.safetensors.zst"
    retained.write_bytes(b"recognized retained body")
    monkeypatch.setattr(
        remote_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=failing.required_capacity - retained.stat().st_size - 1),
    )
    with pytest.raises(RemoteStatisticsError, match="admission failed"):
        failing.prepare(0)
    assert failing._executor is None
    assert not retained.exists()
    assert failing._lock_fd is None


def test_header_segment_size_mismatch_fails_and_cleans_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    remote_path = bucket._path(f"{prefix}/shard_00001_features.safetensors.zst")
    data = load_features(remote_path)
    data["segment_size"] = 8
    modified = save_features(data, tmp_path / "modified")
    os.replace(modified, remote_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)

    with pytest.raises(RemoteStatisticsError, match="segment_size=8"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
        )
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]


def test_admission_cleanup_failure_is_secondary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    profile = SimpleNamespace(remote_transport=None)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
        profile=profile,
    )
    monkeypatch.setattr(
        remote_module.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=source.required_capacity - 1),
    )

    def fail_cleanup() -> None:
        raise OSError("injected initialization cleanup failure")

    monkeypatch.setattr(source, "_cleanup_bodies", fail_cleanup)
    with pytest.raises(RemoteStatisticsError, match="admission failed") as caught:
        source.prepare(0)
    assert source._lock_fd is None
    assert any("initialization cleanup also failed" in note for note in caught.value.__notes__)
    assert profile.remote_transport["terminal_error"]["stage"] == ("staging_initialization")
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]


def test_loader_failure_releases_current_body(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, prefix, bucket = _setup_remote(tmp_path)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)

    def fail_load(*_args, **_kwargs):
        raise RuntimeError("injected stable loader failure")

    monkeypatch.setattr(statistics_module, "load_features_stable", fail_load)
    with pytest.raises(RuntimeError, match="injected stable loader failure"):
        compute_statistics(
            str(feature_dir),
            str(tmp_path / "statistics"),
            n_sae_features=4,
            remote_config=_config(prefix=prefix, staging=tmp_path / "staging"),
        )
    assert sorted(path.name for path in (tmp_path / "staging").iterdir()) == [
        ".statistics-remote-owner.json",
        ".statistics-remote.lock",
    ]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("expected_shard_count", True),
        ("expected_shard_count", 0),
        ("retry_timeout_seconds", True),
        ("retry_timeout_seconds", float("inf")),
        ("retry_interval_seconds", 0.0),
        ("expected_segment_size", True),
        ("expected_segment_size", 0),
    ],
)
def test_remote_config_rejects_noncanonical_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    values: dict[str, object] = {
        "bucket_id": HfBucketId.parse("owner/test-bucket"),
        "remote_prefix": NormalizedRemotePrefix.parse("features/base"),
        "staging_dir": tmp_path / "staging",
        "expected_shard_count": 3,
        "retry_timeout_seconds": 1.0,
        "retry_interval_seconds": 0.1,
        "expected_segment_size": 4,
    }
    values[field] = value
    with pytest.raises(ValueError, match="positive"):
        RemoteStatisticsConfig(**cast(Any, values))


def test_empty_gap_and_one_shard_inventory_policies(tmp_path: Path) -> None:
    empty = FakeBucket(tmp_path / "empty")
    with pytest.raises(RemoteStatisticsError, match="no direct raw feature"):
        RemoteStatisticsSource.snapshot(
            _config(prefix="features/base", staging=tmp_path / "empty-staging"),
            exclude_last_shard=False,
            backend=empty,
        )

    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    bucket._path(f"{prefix}/shard_00001_features.safetensors.zst").unlink()
    with pytest.raises(RemoteStatisticsError, match="contiguous family"):
        RemoteStatisticsSource.snapshot(
            _config(prefix=prefix, staging=tmp_path / "gap", expected_count=None),
            exclude_last_shard=False,
            backend=bucket,
        )

    bucket._path(f"{prefix}/shard_00002_features.safetensors.zst").unlink()
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "one", expected_count=1),
        exclude_last_shard=True,
        backend=bucket,
    )
    assert len(source.full_inventory) == 1
    assert source.statistics_inventory == source.full_inventory
    assert source.excluded_shard is None


def test_capture_regular_pid_and_download_delegator_edges(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RemoteStatisticsError, match="cannot inspect"):
        remote_module._capture_safe_regular(tmp_path / "missing", surface="test")
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(RemoteStatisticsError, match="single-link regular"):
        remote_module._capture_safe_regular(directory, surface="test")

    def missing_pid(_pid: int, _signal: int) -> None:
        raise ProcessLookupError

    monkeypatch.setattr(remote_module.os, "kill", missing_pid)
    assert not remote_module._pid_is_live_or_inaccessible(999999)

    def inaccessible_pid(_pid: int, _signal: int) -> None:
        raise PermissionError

    monkeypatch.setattr(remote_module.os, "kill", inaccessible_pid)
    assert remote_module._pid_is_live_or_inaccessible(999999)

    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    observed = 0

    def observe() -> None:
        nonlocal observed
        observed += 1

    wrapper = remote_module._DownloadRecordingBackend(
        bucket,
        observe_staging=observe,
    )
    remote_path = f"{prefix}/shard_00000_features.safetensors.zst"
    assert wrapper.stat(remote_path) == bucket.stat(remote_path)
    assert len(wrapper.list(prefix)) == 3
    destination = tmp_path / "delegated"
    wrapper.download(remote_path, destination)
    assert wrapper.attempts == 1
    assert wrapper.materialized_bytes == destination.stat().st_size
    assert observed == 1
    with pytest.raises(AssertionError, match="must never upload"):
        wrapper.upload(destination, remote_path)

    destination.unlink()

    class NoMaterializationBucket(FakeBucket):
        def download(self, remote_path: str, local_path: Path) -> None:
            raise OSError("no local file")

    no_file = remote_module._DownloadRecordingBackend(
        NoMaterializationBucket(bucket.root),
        observe_staging=observe,
    )
    with pytest.raises(OSError, match="no local file"):
        no_file.download(remote_path, destination)
    assert no_file.attempts == 1
    assert no_file.materialized_bytes == 0

    fifo = tmp_path / "download-fifo"
    os.mkfifo(fifo)

    class NonRegularBucket(FakeBucket):
        def download(self, remote_path: str, local_path: Path) -> None:
            del remote_path, local_path

    nonregular = remote_module._DownloadRecordingBackend(
        NonRegularBucket(bucket.root),
        observe_staging=observe,
    )
    nonregular.download(remote_path, fifo)
    assert nonregular.materialized_bytes == 0


def _checkpoint_metadata(*, l0: object = None, bos: object = False, sft: object = None):
    return {
        "schema_version": 1,
        "l0_tier": l0,
        "ignore_bos_features": bos,
        "sft_mixture_fingerprint_json": sft,
    }


def _canonical_sft_fingerprint() -> str:
    return mixture_fingerprint_json(
        {
            "mixture": "fixture",
            "split": "train",
            "tasks": [{"class_name": "FixtureTask", "length": 1}],
            "len_mixture": 1,
            "index_map_sha256": "a" * 64,
        }
    )


def test_checkpoint_metadata_strict_validation_and_resolution(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError, match="before shard metadata"):
        source.checkpoint_metadata()
    with pytest.raises(RemoteStatisticsError, match="missing"):
        source.restore_checkpoint_metadata({"other": 1})
    with pytest.raises(RemoteStatisticsError, match="malformed"):
        source._validate_checkpoint_metadata(None)
    with pytest.raises(RemoteStatisticsError, match="malformed"):
        source._validate_checkpoint_metadata({**_checkpoint_metadata(), "extra": None})
    with pytest.raises(RemoteStatisticsError, match="schema version"):
        source._validate_checkpoint_metadata({**_checkpoint_metadata(), "schema_version": 2})
    with pytest.raises(RemoteStatisticsError, match="l0_tier is malformed"):
        source._validate_checkpoint_metadata(_checkpoint_metadata(l0=1))
    with pytest.raises(RemoteStatisticsError, match="l0_tier is invalid"):
        source._validate_checkpoint_metadata(_checkpoint_metadata(l0="unknown"))
    with pytest.raises(RemoteStatisticsError, match="ignore_bos_features"):
        source._validate_checkpoint_metadata(_checkpoint_metadata(bos=1))
    with pytest.raises(RemoteStatisticsError, match="fingerprint is malformed"):
        source._validate_checkpoint_metadata(_checkpoint_metadata(sft=1))
    with pytest.raises(RemoteStatisticsError, match="fingerprint is invalid"):
        source._validate_checkpoint_metadata(_checkpoint_metadata(sft="{}"))
    canonical = _canonical_sft_fingerprint()
    noncanonical = json.dumps(json.loads(canonical), sort_keys=True)
    with pytest.raises(RemoteStatisticsError, match="not canonical"):
        source._validate_checkpoint_metadata(_checkpoint_metadata(sft=noncanonical))
    with pytest.raises(RemoteStatisticsError, match="never established"):
        source.resolved_metadata()

    restored = source.restore_checkpoint_metadata(
        {
            "remote_source_metadata": _checkpoint_metadata(
                l0="small",
                bos=True,
                sft=canonical,
            ),
            "retained": 7,
        }
    )
    assert restored == {"retained": 7}
    assert source.checkpoint_metadata()["remote_source_metadata"] == (
        _checkpoint_metadata(l0="small", bos=True, sft=canonical)
    )
    l0, bos, sft = source.resolved_metadata()
    assert l0 is not None and l0.value == "small"
    assert bos is True
    assert sft == canonical


def test_terminal_profile_preserves_first_error(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    profile = SimpleNamespace(remote_transport=None)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
        profile=profile,
    )
    source.record_terminal_error(stage="primary", error=RuntimeError("first"))
    source.record_terminal_error(stage="secondary", error=RuntimeError("second"))
    source._refresh_profile(status="complete")
    assert profile.remote_transport["status"] == "failed"
    assert profile.remote_transport["terminal_error"]["stage"] == "primary"

    source._bootstrap_index = 0
    bootstrap = source._profile_record(0)
    bootstrap["transfer_seconds"] = 4.0
    bootstrap["consumer_wait_seconds"] = 4.0
    overlapped = source._profile_record(1)
    overlapped["transfer_seconds"] = 10.0
    overlapped["consumer_wait_seconds"] = 3.0
    payload = source._profile_payload(status="running")
    assert payload["post_bootstrap_transfer_seconds"] == 10.0
    assert payload["post_bootstrap_hidden_transfer_seconds"] == 7.0
    assert payload["overlap_fraction"] == 0.7


def test_overlap_profile_excludes_resume_bootstrap(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )

    class PendingExecutor:
        def __init__(self) -> None:
            self.future: Future[remote_module._TransferResult] = Future()

        def submit(self, *_args, **_kwargs):
            return self.future

    executor = PendingExecutor()
    source._executor = cast(Any, executor)
    source._submit(1)
    assert source._bootstrap_index == 1

    resumed_bootstrap = source._profile_record(1)
    resumed_bootstrap["transfer_seconds"] = 4.0
    resumed_bootstrap["consumer_wait_seconds"] = 4.0
    assert source._profile_payload(status="running")["post_bootstrap_transfer_seconds"] == 0.0

    overlapped = source._profile_record(2)
    overlapped["transfer_seconds"] = 10.0
    overlapped["consumer_wait_seconds"] = 3.0
    payload = source._profile_payload(status="running")
    assert payload["post_bootstrap_transfer_seconds"] == 10.0
    assert payload["post_bootstrap_hidden_transfer_seconds"] == 7.0
    assert payload["overlap_fraction"] == 0.7
    assert executor.future.cancel()
    source._future = None
    source._future_index = None
    source._executor = None


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_staging_path_must_be_a_real_directory(
    tmp_path: Path,
    kind: str,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    if kind == "file":
        staging.write_text("canary")
    else:
        target = tmp_path / "target"
        target.mkdir()
        staging.symlink_to(target, target_is_directory=True)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError, match="real directory"):
        source.prepare(0)
    assert bucket.downloads == []


def test_post_mkdir_symlink_race_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    real_is_symlink = Path.is_symlink

    def raced_is_symlink(path: Path) -> bool:
        if path == staging and path.exists():
            return True
        return real_is_symlink(path)

    monkeypatch.setattr(Path, "is_symlink", raced_is_symlink)
    with pytest.raises(RemoteStatisticsError, match="real directory"):
        source.prepare(0)


def test_lock_open_and_validation_failure_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "no-open"),
        exclude_last_shard=False,
        backend=bucket,
    )

    def fail_open(*_args, **_kwargs):
        raise OSError("injected open failure")

    monkeypatch.setattr(remote_module.os, "open", fail_open)
    with pytest.raises(RemoteStatisticsError, match="cannot be locked"):
        source._open_lock_and_owner()
    assert source._lock_fd is None

    def interrupt_open(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(remote_module.os, "open", interrupt_open)
    with pytest.raises(KeyboardInterrupt):
        source._open_lock_and_owner()
    assert source._lock_fd is None

    monkeypatch.undo()
    unsafe = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "unsafe-lock"),
        exclude_last_shard=False,
        backend=bucket,
    )
    monkeypatch.setattr(
        remote_module.os,
        "fstat",
        lambda _fd: SimpleNamespace(st_mode=stat.S_IFDIR, st_nlink=1),
    )
    with pytest.raises(RemoteStatisticsError, match="lock is not"):
        unsafe._open_lock_and_owner()
    assert unsafe._lock_fd is None


def test_lock_without_o_nofollow_platform_branch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    monkeypatch.delattr(remote_module.os, "O_NOFOLLOW")
    source._open_lock_and_owner()
    source._release_lock()


def test_missing_or_corrupt_owner_marker_preserves_namespace(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    missing = tmp_path / "missing-owner"
    missing.mkdir()
    canary = missing / "canary"
    canary.write_text("keep")
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=missing),
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError, match="no ownership marker"):
        source.prepare(0)
    assert canary.read_text() == "keep"

    corrupt = tmp_path / "corrupt-owner"
    corrupt.mkdir()
    (corrupt / ".statistics-remote-owner.json").write_text("{")
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=corrupt),
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError, match="unreadable"):
        source.prepare(0)


@pytest.mark.parametrize(
    "kind",
    ["symlink", "hardlink", "fifo", "socket", "directory"],
)
def test_unsafe_canonical_staging_entry_is_preserved(
    tmp_path: Path,
    kind: str,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    short_staging: Path | None = None
    if kind == "socket":
        short_staging = Path(tempfile.mkdtemp(prefix="statistics-remote-test-"))
        staging = short_staging
    else:
        staging = tmp_path / "staging"
        staging.mkdir()
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(_owner_payload(prefix)))
    hostile = staging / "shard_00000_features.safetensors.zst"
    socket_handle: socket.socket | None = None
    if kind == "symlink":
        target = tmp_path / "target"
        target.write_text("keep")
        hostile.symlink_to(target)
    elif kind == "hardlink":
        target = tmp_path / "target"
        target.write_text("keep")
        os.link(target, hostile)
    elif kind == "fifo":
        os.mkfifo(hostile)
    elif kind == "socket":
        socket_handle = socket.socket(socket.AF_UNIX)
        socket_handle.bind(str(hostile))
    else:
        hostile.mkdir()
    try:
        source = RemoteStatisticsSource.snapshot(
            _config(prefix=prefix, staging=staging),
            exclude_last_shard=False,
            backend=bucket,
        )
        with pytest.raises(RemoteStatisticsError, match="single-link regular"):
            source.prepare(0)
        assert hostile.exists() or hostile.is_symlink()
    finally:
        if socket_handle is not None:
            socket_handle.close()
        if short_staging is not None:
            shutil.rmtree(short_staging)


def test_dead_temporary_and_stale_canonical_are_removed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(_owner_payload(prefix)))
    dead = staging / ".shard_00000_features.safetensors.zst.remote-tmp.99999999.deadbeef"
    dead.write_bytes(b"partial")
    stale = staging / "shard_00002_features.safetensors.zst"
    shutil.copyfile(bucket._path(f"{prefix}/{stale.name}"), stale)
    monkeypatch.setattr(remote_module, "_pid_is_live_or_inaccessible", lambda _pid: False)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    prepared = source.prepare(0)
    assert not dead.exists()
    assert not stale.exists()
    source.release(prepared)
    source.finish_transport()


def test_observe_staging_handles_absence_race_and_nonregular(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "missing"),
        exclude_last_shard=False,
        backend=bucket,
    )
    source._observe_staging()

    staging = source.config.staging_dir
    staging.mkdir()
    raced = staging / "shard_00000_features.safetensors.zst"
    raced.write_bytes(b"body")
    nonregular = staging / "shard_00001_features.safetensors.zst"
    nonregular.mkdir()
    real_lstat = Path.lstat

    def race_lstat(path: Path):
        if path == raced:
            raise OSError("injected disappearance")
        return real_lstat(path)

    monkeypatch.setattr(Path, "lstat", race_lstat)
    source._observe_staging()
    assert source._staging_high_water_bodies == 0


def test_observe_staging_is_serialized_by_profile_lock(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    source.config.staging_dir.mkdir()
    body = source.config.staging_dir / "shard_00000_features.safetensors.zst"
    body.write_bytes(b"body")
    started = threading.Event()

    def observe() -> None:
        started.set()
        source._observe_staging()

    source._profile_lock.acquire()
    thread = threading.Thread(target=observe)
    try:
        thread.start()
        assert started.wait(timeout=1.0)
        thread.join(timeout=0.1)
        assert thread.is_alive()
        assert source._staging_high_water_bodies == 0
    finally:
        source._profile_lock.release()
    thread.join(timeout=1.0)
    assert not thread.is_alive()
    assert source._staging_high_water_bodies == 1
    assert source._staging_high_water_bytes == len(b"body")


def test_transport_guard_branches_and_prepare_index_validation(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    with pytest.raises(RemoteStatisticsError, match="must be an int"):
        source.prepare(True)
    with pytest.raises(RemoteStatisticsError, match="out of range"):
        source.prepare(-1)
    with pytest.raises(RemoteStatisticsError, match="scheduler order"):
        source._consume_future(0)

    source._ensure_transport(0)
    executor = source._executor
    source._ensure_transport(0)
    assert source._executor is executor
    pending: Future[remote_module._TransferResult] = Future()
    source._future = pending
    source._future_index = 0
    with pytest.raises(RemoteStatisticsError, match="pending future"):
        source._submit(0)
    assert pending.cancel()
    source._future = None
    source._future_index = None
    source.finish_transport()
    source.finish_transport()

    initialized = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "initialized"),
        exclude_last_shard=False,
        backend=bucket,
    )
    initialized._staging_initialized = True
    initialized._ensure_transport(0)
    assert initialized._executor is not None
    initialized.finish_transport()


def test_scheduler_submits_lookahead_before_main_thread_header_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    main_thread = threading.get_ident()
    download_threads: list[int] = []
    real_download = bucket.download

    def observe_download(remote_path: str, local_path: Path) -> None:
        download_threads.append(threading.get_ident())
        real_download(remote_path, local_path)

    monkeypatch.setattr(bucket, "download", observe_download)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    scheduled: list[int] = []
    real_submit = source._submit

    def observe_submit(index: int) -> None:
        scheduled.append(index)
        real_submit(index)

    monkeypatch.setattr(source, "_submit", observe_submit)
    real_header = remote_module.read_shard_header_stable

    def observe_header(*args, **kwargs):
        assert threading.get_ident() == main_thread
        assert scheduled[:2] == [0, 1]
        return real_header(*args, **kwargs)

    monkeypatch.setattr(remote_module, "read_shard_header_stable", observe_header)
    prepared = source.prepare(0)
    assert scheduled[:2] == [0, 1]
    assert download_threads and all(thread != main_thread for thread in download_threads)
    source.release(prepared)
    source.finish_transport()


def test_corrupt_retained_body_is_removed_before_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(_owner_payload(prefix)))
    retained = staging / "shard_00000_features.safetensors.zst"
    retained.write_bytes(b"wrong")
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    prepared = source.prepare(0)
    assert prepared.disposition == "downloaded"
    assert retained.stat().st_size != len(b"wrong")
    source.release(prepared)
    source.finish_transport()

    shutil.copyfile(bucket._path(f"{prefix}/{retained.name}"), retained)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    real_verify = remote_module.verified_local_object_info
    calls = 0

    def fail_first_verify(*, local_path: Path, remote_path: str):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("injected retained verification failure")
        return real_verify(local_path=local_path, remote_path=remote_path)

    monkeypatch.setattr(
        remote_module,
        "verified_local_object_info",
        fail_first_verify,
    )
    prepared = source.prepare(0)
    assert prepared.disposition == "downloaded"
    assert calls == 1
    source.release(prepared)
    source.finish_transport()


def test_base_exception_during_retained_verification_is_not_swallowed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / ".statistics-remote-owner.json").write_text(json.dumps(_owner_payload(prefix)))
    name = "shard_00000_features.safetensors.zst"
    retained = staging / name
    shutil.copyfile(bucket._path(f"{prefix}/{name}"), retained)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )

    def interrupt_verify(**_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(
        remote_module,
        "verified_local_object_info",
        interrupt_verify,
    )
    with pytest.raises(KeyboardInterrupt):
        source.prepare(0)
    assert bucket.downloads == []
    source.finish_transport()


def test_downloaded_body_is_not_reverified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )

    def fail_reverification(**_kwargs):
        raise AssertionError("downloaded bodies must reuse verified publication")

    monkeypatch.setattr(
        remote_module,
        "verified_local_object_info",
        fail_reverification,
    )
    prepared = source.prepare(0)
    assert prepared.disposition == "downloaded"
    source.release(prepared)
    source.finish_transport()


@pytest.mark.parametrize(
    ("mode", "links"),
    [(stat.S_IFDIR, 1), (stat.S_IFREG, 2)],
)
def test_download_callback_rejects_unsafe_published_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: int,
    links: int,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )

    def publish_unsafe_identity(*_args, **kwargs):
        kwargs["verified_publication_callback"](SimpleNamespace(st_mode=mode, st_nlink=links))
        return kwargs["expected"]

    monkeypatch.setattr(
        remote_module,
        "download_verified_object",
        publish_unsafe_identity,
    )
    with pytest.raises(RemoteStatisticsError, match="single-link regular"):
        source.prepare(0)
    source.finish_transport()


def test_download_requires_bound_published_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    monkeypatch.setattr(
        remote_module,
        "download_verified_object",
        lambda *_args, **kwargs: kwargs["expected"],
    )
    with pytest.raises(RemoteStatisticsError, match="no bound file identity"):
        source.prepare(0)
    source.finish_transport()


def test_downloaded_body_identity_is_bound_before_prepare(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    profile = SimpleNamespace(remote_transport=None)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
        profile=profile,
    )
    real_download = remote_module.download_verified_object

    def replace_after_verified_publication(*args, **kwargs):
        result = real_download(*args, **kwargs)
        final_path = kwargs["final_path"]
        replacement = final_path.with_name("same-bytes-replacement")
        shutil.copyfile(final_path, replacement)
        os.replace(replacement, final_path)
        return result

    monkeypatch.setattr(
        remote_module,
        "download_verified_object",
        replace_after_verified_publication,
    )
    with pytest.raises(Exception, match="changed before stable header read"):
        source.prepare(0)
    assert profile.remote_transport["terminal_error"]["stage"] == "header"
    source.finish_transport()


def test_stale_published_stat_size_binds_verified_length(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    real_download = remote_module.download_verified_object

    def stale_size_publication(*args, **kwargs):
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
                        published.st_size - 5,
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

    monkeypatch.setattr(
        remote_module,
        "download_verified_object",
        stale_size_publication,
    )
    prepared = source.prepare(0)

    # The bound identity carries the inventory-verified length, so the strict
    # stable header read inside prepare() accepted the staged body.
    assert prepared.file_identity.size == prepared.remote_identity.byte_length
    assert prepared.file_identity.size == prepared.local_path.stat().st_size
    assert any("binding the verified length" in record.message for record in caplog.records)
    source.release(prepared)
    source.finish_transport()


def test_retry_profile_counts_partial_materialization(tmp_path: Path) -> None:
    _feature_dir, prefix, ordinary = _setup_remote(tmp_path)

    class RetryOnceBucket(FakeBucket):
        def __init__(self, root: Path) -> None:
            super().__init__(root)
            self.attempts = 0

        def download(self, remote_path: str, local_path: Path) -> None:
            self.attempts += 1
            if self.attempts == 1:
                local_path.write_bytes(b"partial")
                raise OSError("injected first-attempt failure")
            super().download(remote_path, local_path)

    bucket = RetryOnceBucket(ordinary.root)
    profile = SimpleNamespace(remote_transport=None)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
        profile=profile,
    )
    prepared = source.prepare(0)
    record = profile.remote_transport["per_shard"][0]
    assert record["attempts"] == 2
    assert record["retries"] == 1
    assert record["materialized_bytes"] == len(b"partial") + (prepared.remote_identity.byte_length)
    assert profile.remote_transport["materialized_bytes"] == record["materialized_bytes"]
    source.release(prepared)
    source.finish_transport()


def test_download_failure_records_primary_stage_and_attempts(tmp_path: Path) -> None:
    _feature_dir, prefix, ordinary = _setup_remote(tmp_path)

    class AlwaysFailBucket(FakeBucket):
        def download(self, remote_path: str, local_path: Path) -> None:
            del remote_path
            local_path.write_bytes(b"partial")
            raise OSError("persistent failure")

    profile = SimpleNamespace(remote_transport=None)
    source = RemoteStatisticsSource.snapshot(
        RemoteStatisticsConfig(
            bucket_id=HfBucketId.parse("owner/test-bucket"),
            remote_prefix=NormalizedRemotePrefix.parse(prefix),
            staging_dir=tmp_path / "staging",
            expected_shard_count=3,
            retry_timeout_seconds=0.001,
            retry_interval_seconds=0.001,
            expected_segment_size=4,
        ),
        exclude_last_shard=False,
        backend=AlwaysFailBucket(ordinary.root),
        profile=profile,
    )
    with pytest.raises(Exception, match="timed out"):
        source.prepare(0)
    record = profile.remote_transport["per_shard"][0]
    assert record["attempts"] >= 1
    assert record["materialized_bytes"] >= len(b"partial")
    assert record["consumer_ready_seconds"] is not None
    assert record["consumer_wait_seconds"] > 0.0
    assert (
        profile.remote_transport["critical_path_remote_wait_seconds"]
        == (record["consumer_wait_seconds"])
    )
    assert profile.remote_transport["post_bootstrap_transfer_seconds"] == 0.0
    assert profile.remote_transport["overlap_fraction"] == 0.0
    assert profile.remote_transport["terminal_error"]["stage"] == "download"
    source.finish_transport()


def test_header_metadata_parse_and_cross_shard_mismatch(
    tmp_path: Path,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    prepared = source.prepare(0)
    with pytest.raises(RemoteStatisticsError, match="malformed segment_size"):
        source._metadata_for_header(prepared, {"segment_size": "bad"})
    source.release(prepared)
    source.finish_transport()

    second = bucket._path(f"{prefix}/shard_00001_features.safetensors.zst")
    data = load_features(second)
    data["l0_tier"] = "small"
    modified = save_features(data, tmp_path / "modified-l0")
    os.replace(modified, second)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "mismatch"),
        exclude_last_shard=False,
        backend=bucket,
    )
    first = source.prepare(0)
    source.release(first)
    with pytest.raises(RemoteStatisticsError, match="metadata differs"):
        source.prepare(1)
    source.finish_transport()


def test_cleanup_failure_edges_preserve_primary_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    canonical = staging / "shard_00000_features.safetensors.zst"
    canonical.write_bytes(b"body")
    dead_temp = staging / ".shard_00001_features.safetensors.zst.remote-tmp.99999999.deadbeef"
    dead_temp.write_bytes(b"partial")
    unknown = staging / "canary"
    unknown.write_text("keep")
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    source._staging_initialized = True
    source._preserve_paths.add(canonical)
    monkeypatch.setattr(remote_module, "_pid_is_live_or_inaccessible", lambda _pid: True)
    with pytest.raises(RemoteStatisticsError, match="live/inaccessible"):
        source._cleanup_bodies()
    assert canonical.exists() and dead_temp.exists() and unknown.exists()

    primary = RuntimeError("primary transfer failure")
    future: Future[remote_module._TransferResult] = Future()
    future.set_exception(primary)
    source._future = future
    source._future_index = 1
    source._preserve_paths.add(dead_temp)
    with pytest.raises(RuntimeError, match="primary transfer failure") as caught:
        source.finish_transport()
    assert caught.value is primary
    assert any("cleanup also failed" in note for note in caught.value.__notes__)


def test_cleanup_unlink_failure_is_reported(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    staging = tmp_path / "staging"
    staging.mkdir()
    canonical = staging / "shard_00000_features.safetensors.zst"
    canonical.write_bytes(b"body")
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    source._staging_initialized = True

    def fail_unlink(_path: Path, **_kwargs) -> None:
        raise OSError("injected cleanup failure")

    monkeypatch.setattr(source, "_safe_unlink", fail_unlink)
    with pytest.raises(OSError, match="injected cleanup failure"):
        source.finish_transport()
    assert canonical.exists()

    dead_staging = tmp_path / "dead-temp-staging"
    dead_staging.mkdir()
    dead_temp = dead_staging / ".shard_00000_features.safetensors.zst.remote-tmp.99999999.deadbeef"
    dead_temp.write_bytes(b"partial")
    dead_source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=dead_staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    dead_source._staging_initialized = True
    monkeypatch.setattr(remote_module, "_pid_is_live_or_inaccessible", lambda _pid: False)
    monkeypatch.setattr(dead_source, "_safe_unlink", fail_unlink)
    with pytest.raises(OSError, match="injected cleanup failure"):
        dead_source._cleanup_bodies()


def test_identical_transfer_and_cleanup_failure_is_not_self_noted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    primary = RuntimeError("shared failure")
    future: Future[remote_module._TransferResult] = Future()
    future.set_exception(primary)
    source._future = future
    source._future_index = 0

    def raise_primary() -> None:
        raise primary

    class FailingExecutor:
        def shutdown(self, **_kwargs) -> None:
            raise OSError("secondary executor failure")

    def fail_release() -> None:
        raise OSError("secondary lock failure")

    source._executor = cast(Any, FailingExecutor())
    monkeypatch.setattr(source, "_cleanup_bodies", raise_primary)
    monkeypatch.setattr(source, "_release_lock", fail_release)
    with pytest.raises(RuntimeError, match="shared failure") as caught:
        source.finish_transport()
    assert caught.value is primary
    notes = getattr(primary, "__notes__", [])
    assert any("executor shutdown also failed" in note for note in notes)
    assert any("lock release also failed" in note for note in notes)
    assert not any("cleanup also failed" in note for note in notes)


def test_identical_future_executor_and_lock_failure_is_not_self_noted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    primary = RuntimeError("one shared failure")
    future: Future[remote_module._TransferResult] = Future()
    future.set_exception(primary)
    source._future = future
    source._future_index = 0

    class SameFailingExecutor:
        def shutdown(self, **_kwargs) -> None:
            raise primary

    def raise_primary() -> None:
        raise primary

    source._executor = cast(Any, SameFailingExecutor())
    monkeypatch.setattr(source, "_release_lock", raise_primary)
    with pytest.raises(RuntimeError, match="one shared failure") as caught:
        source.finish_transport()
    assert caught.value is primary
    assert not getattr(primary, "__notes__", [])


@pytest.mark.parametrize("stage", ["executor", "lock"])
def test_finish_surfaces_unaccompanied_shutdown_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    stage: str,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )

    class FailingExecutor:
        def shutdown(self, **_kwargs) -> None:
            raise OSError("executor failure")

    if stage == "executor":
        source._executor = cast(Any, FailingExecutor())
        match = "executor failure"
    else:
        monkeypatch.setattr(
            source,
            "_release_lock",
            lambda: (_ for _ in ()).throw(OSError("lock failure")),
        )
        match = "lock failure"
    with pytest.raises(OSError, match=match):
        source.finish_transport()


def test_cancelled_future_is_suppressed_and_repeated_finish_is_safe(
    tmp_path: Path,
) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=False,
        backend=bucket,
    )
    future: Future[remote_module._TransferResult] = Future()
    future.set_exception(remote_module.RemoteTrainingError("download cancelled: fixture"))
    source._future = future
    source._future_index = 0
    source.finish_transport()
    source.finish_transport()

    own_staging = tmp_path / "own-temp-staging"
    own_staging.mkdir()
    own_temp = (
        own_staging / f".shard_00000_features.safetensors.zst.remote-tmp.{os.getpid()}.deadbeef"
    )
    own_temp.write_bytes(b"partial")
    own_source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=own_staging),
        exclude_last_shard=False,
        backend=bucket,
    )
    own_source._staging_initialized = True
    own_source._cleanup_bodies()
    assert not own_temp.exists()

    completed = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "completed-future"),
        exclude_last_shard=False,
        backend=bucket,
    )
    completed_future: Future[remote_module._TransferResult] = Future()
    completed_future.set_result(cast(Any, object()))
    completed._future = completed_future
    completed._future_index = 0
    completed.finish_transport()

    pending = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "pending-future"),
        exclude_last_shard=False,
        backend=bucket,
    )
    pending._future = Future()
    pending._future_index = 0
    pending.finish_transport()


def test_final_inventory_identity_replacement_is_detected(tmp_path: Path) -> None:
    _feature_dir, prefix, bucket = _setup_remote(tmp_path)
    source = RemoteStatisticsSource.snapshot(
        _config(prefix=prefix, staging=tmp_path / "staging"),
        exclude_last_shard=True,
        backend=bucket,
    )
    contract = source.checkpoint_contract(
        n_sae_features=4,
        skip_per_feature_detail=False,
        numeric_runtime={"device": "cpu"},
    )
    assert contract["excluded_shard"] is not None

    replacement = bucket._path(f"{prefix}/shard_00000_features.safetensors.zst")
    replacement.write_bytes(replacement.read_bytes() + b"changed")
    with pytest.raises(RemoteStatisticsError, match="inventory changed"):
        source.assert_final_inventory_unchanged()
