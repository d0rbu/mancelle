"""Functional contracts for the bounded remote percentile runner."""

from __future__ import annotations

import gc
import json
import shutil
import weakref
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import hf_xet
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import steering.percentile_remote as remote_module
from scripts.preprocess.precompute_percentiles import run as run_local_percentiles
from steering.artifacts.sync import materialize_pulled_percentile_statistics
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import glob_features
from steering.overlay_io import (
    MANIFEST_FILENAME,
    OverlayKind,
    compute_base_shard_content_hash,
    discover_overlay,
    load_overlay_shard,
)
from steering.percentile_remote import (
    REMOTE_CONTRACT_FILENAME,
    REMOTE_RECEIPTS_DIRNAME,
    RemotePercentileError,
    remote_percentiles_enabled,
    run_remote_percentiles,
)
from steering.remote_training import RemoteObjectInfo
from steering.stats_io import (
    load_percentile_checksum,
    load_statistics,
    load_tokenizer_checksum,
    save_percentile_checksum,
    save_statistics,
    save_tokenizer_checksum,
)
from steering.validate_features import validate_feature_directory
from tests.scripts.test_production_manifest_v2 import _make_env


def _xet_hash(path: Path) -> str:
    return str(cast(Any, hf_xet).hash_files([str(path)])[0].hash)


class FilesystemBucket:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True)
        self.operations: list[tuple[str, str]] = []
        self.interrupt_after_upload: str | None = None
        self.interrupt_first_payload = False
        self.mutate_source_on_second_list = False
        self.mutate_source_on_list_call: int | None = None
        self.interrupt_before_upload: str | None = None
        self.transient_list_failures = 0
        self.transient_stat_failures = 0
        self.source_list_calls = 0

    def _path(self, remote_path: str) -> Path:
        return self.root.joinpath(*remote_path.split("/"))

    def stat(self, remote_path: str) -> RemoteObjectInfo | None:
        self.operations.append(("stat", remote_path))
        if self.transient_stat_failures > 0:
            self.transient_stat_failures -= 1
            raise OSError("injected transient stat failure")
        path = self._path(remote_path)
        if not path.is_file():
            return None
        return RemoteObjectInfo(remote_path, path.stat().st_size, _xet_hash(path))

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        self.operations.append(("list", prefix))
        if self.transient_list_failures > 0:
            self.transient_list_failures -= 1
            raise OSError("injected transient list failure")
        if prefix == "raw":
            self.source_list_calls += 1
            mutation_call = (
                2 if self.mutate_source_on_second_list else self.mutate_source_on_list_call
            )
            if mutation_call is not None and self.source_list_calls == mutation_call:
                source = self._path("raw/shard_00000_features.safetensors.zst")
                source.write_bytes(source.read_bytes() + b"mutated")
        root = self._path(prefix)
        if not root.exists():
            return []
        out: list[RemoteObjectInfo] = []
        for path in sorted(root.rglob("*")):
            if path.is_file():
                remote_path = path.relative_to(self.root).as_posix()
                out.append(RemoteObjectInfo(remote_path, path.stat().st_size, _xet_hash(path)))
        return out

    def download(self, remote_path: str, local_path: Path) -> None:
        self.operations.append(("download", remote_path))
        shutil.copyfile(self._path(remote_path), local_path)

    def upload(self, local_path: Path, remote_path: str) -> None:
        self.operations.append(("upload", remote_path))
        if self.interrupt_before_upload == remote_path:
            self.interrupt_before_upload = None
            raise KeyboardInterrupt(f"injected before {remote_path}")
        destination = self._path(remote_path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and destination.read_bytes() != local_path.read_bytes():
            raise RuntimeError(f"test bucket collision at {remote_path}")
        shutil.copyfile(local_path, destination)
        if self.interrupt_first_payload and remote_path.endswith("/shard_00000.safetensors.zst"):
            self.interrupt_first_payload = False
            raise KeyboardInterrupt("injected after first payload")
        if self.interrupt_after_upload == remote_path:
            self.interrupt_after_upload = None
            raise KeyboardInterrupt(f"injected interruption after {remote_path}")


def _remote_cfg(feature_dir: Path, staging_dir: Path, *, resume: bool = False) -> DictConfig:
    return OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "segment_size": 16,
            "device": "cpu",
            "resume": resume,
            "dry_run": False,
            "force": False,
            "zstd_level": 12,
            "profile": False,
            "profile_summary_output": None,
            "remote": {
                "enabled": True,
                "bucket_id": "owner/test-bucket",
                "local_features_root": str(feature_dir),
                "remote_features_root": "raw",
                "staging_dir": str(staging_dir),
                "expected_shard_count": 3,
                "max_output_shard_bytes": 1_000_000,
                "max_decompressed_shard_bytes": 1_000_000,
                "retry_timeout_seconds": 0.2,
                "retry_interval_seconds": 0.001,
            },
        }
    )


def _stage_sources(bucket: FilesystemBucket, feature_dir: Path) -> list[bytes]:
    bodies: list[bytes] = []
    for path in glob_features(feature_dir):
        bodies.append(path.read_bytes())
        destination = bucket._path(f"raw/{path.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        path.unlink()
    return bodies


def _output_prefix(bucket: FilesystemBucket) -> str:
    manifests = [
        path.relative_to(bucket.root).as_posix()
        for path in bucket.root.rglob(MANIFEST_FILENAME)
        if "_verified_upload_claims" not in path.parts
    ]
    assert len(manifests) == 1
    return manifests[0].removesuffix(f"/{MANIFEST_FILENAME}")


def test_remote_runner_publishes_manifest_last_and_fast_rerun_downloads_no_raw(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    source_bodies = _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")

    reference_dir = tmp_path / "reference" / "features" / "s16"
    reference_dir.mkdir(parents=True)
    shutil.copyfile(stats_path, reference_dir / stats_path.name)
    for index, body in enumerate(source_bodies):
        (reference_dir / f"shard_{index:05d}_features.safetensors.zst").write_bytes(body)
    local_cfg = OmegaConf.create(
        {
            "feature_dir": str(reference_dir),
            "segment_size": 16,
            "device": "cpu",
            "resume": False,
            "dry_run": False,
            "force": False,
            "zstd_level": 12,
            "profile": False,
        }
    )
    assert run_local_percentiles(local_cfg) == 0

    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    checksum = load_percentile_checksum(stats_path)
    assert checksum is not None
    prefix = _output_prefix(bucket)
    canonical_uploads = [
        path
        for operation, path in bucket.operations
        if operation == "upload" and "_verified_upload_claims" not in path
    ]
    assert canonical_uploads[-1] == f"{prefix}/{MANIFEST_FILENAME}"
    assert canonical_uploads[0] == f"{prefix}/{REMOTE_CONTRACT_FILENAME}"
    assert (
        sum(path.startswith(f"{prefix}/{REMOTE_RECEIPTS_DIRNAME}/") for path in canonical_uploads)
        == 3
    )

    manifest = json.loads(bucket._path(f"{prefix}/{MANIFEST_FILENAME}").read_text())
    assert manifest["schema_version"] == 2
    assert "created_at" not in manifest
    assert manifest["base_shard_content_hash"] == compute_base_shard_content_hash(
        [bucket._path(f"raw/shard_{index:05d}_features.safetensors.zst") for index in range(3)]
    )
    for index in range(3):
        payload = bucket._path(f"{prefix}/shard_{index:05d}.safetensors.zst")
        reference_payload = next(reference_dir.glob("pct/*")) / payload.name
        assert payload.read_bytes() == reference_payload.read_bytes()
        values = load_overlay_shard(payload)["feature_percentiles"]
        assert values.ndim == 1
        receipt = json.loads(
            bucket._path(f"{prefix}/{REMOTE_RECEIPTS_DIRNAME}/shard_{index:05d}.json").read_text()
        )
        assert receipt["n_values"] == values.numel()

    bucket.operations.clear()
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    raw_downloads = [
        path
        for operation, path in bucket.operations
        if operation == "download" and path.startswith("raw/shard_")
    ]
    assert raw_downloads == []
    assert source_bodies  # the logical feature directory had no raw resident bodies


def test_remote_rejects_zstd_contract_collision_without_mutation(tmp_path: Path) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    prefix = _output_prefix(bucket)
    canonical_paths = [
        f"{prefix}/{REMOTE_CONTRACT_FILENAME}",
        *(f"{prefix}/shard_{index:05d}.safetensors.zst" for index in range(3)),
        *(f"{prefix}/{REMOTE_RECEIPTS_DIRNAME}/shard_{index:05d}.json" for index in range(3)),
        f"{prefix}/feature_statistics.safetensors",
        f"{prefix}/{MANIFEST_FILENAME}",
    ]
    remote_bytes = {path: bucket._path(path).read_bytes() for path in canonical_paths}
    local_statistics = stats_path.read_bytes()

    bucket.operations.clear()
    cfg.zstd_level = 1
    cfg.remote.staging_dir = str(tmp_path / "level-1-staging")
    with pytest.raises(RemotePercentileError, match="immutable.*contract differs"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)

    assert all(operation != "upload" for operation, _ in bucket.operations)
    assert stats_path.read_bytes() == local_statistics
    for path, original in remote_bytes.items():
        assert bucket._path(path).read_bytes() == original


def test_remote_profile_uses_separate_schema_and_records_staging(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    profile_path = tmp_path / "profiles" / "remote.json"
    cfg.profile = True
    cfg.profile_summary_output = str(profile_path)

    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    assert profile["schema_version"] == 1
    assert profile["mode"] == "remote"
    assert profile["inventory"]["count"] == 3
    assert profile["staging"]["required_bytes"] > 1 << 30
    assert 0 < profile["staging"]["high_water_bytes"] <= profile["staging"]["required_bytes"]
    assert profile["totals"]["status"] == "complete"
    assert len(profile["per_shard"]) == 3


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
def test_remote_cuda_matches_local_cuda_and_rejects_cpu_contract(tmp_path: Path) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    source_bodies = _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    cfg.device = "cuda"

    reference_dir = tmp_path / "reference" / "features" / "s16"
    reference_dir.mkdir(parents=True)
    shutil.copyfile(stats_path, reference_dir / stats_path.name)
    for index, body in enumerate(source_bodies):
        (reference_dir / f"shard_{index:05d}_features.safetensors.zst").write_bytes(body)
    local_cfg = OmegaConf.create(
        {
            "feature_dir": str(reference_dir),
            "segment_size": 16,
            "device": "cuda",
            "resume": False,
            "dry_run": False,
            "force": False,
            "zstd_level": 12,
            "profile": False,
        }
    )
    assert run_local_percentiles(local_cfg) == 0
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0

    prefix = _output_prefix(bucket)
    contract = json.loads(bucket._path(f"{prefix}/{REMOTE_CONTRACT_FILENAME}").read_text())
    assert contract["producer"]["device_type"] == "cuda"
    for index in range(3):
        remote_payload = bucket._path(f"{prefix}/shard_{index:05d}.safetensors.zst")
        local_payload = next(reference_dir.glob("pct/*")) / remote_payload.name
        assert remote_payload.read_bytes() == local_payload.read_bytes()

        raw_source = bucket._path(f"raw/shard_{index:05d}_features.safetensors.zst")
        shutil.copyfile(raw_source, feature_dir / raw_source.name)
    materialized_overlay = feature_dir / "pct" / prefix.rsplit("/", 1)[-1]
    materialized_overlay.mkdir(parents=True)
    shutil.copyfile(
        bucket._path(f"{prefix}/{MANIFEST_FILENAME}"),
        materialized_overlay / MANIFEST_FILENAME,
    )
    for index in range(3):
        name = f"shard_{index:05d}.safetensors.zst"
        shutil.copyfile(bucket._path(f"{prefix}/{name}"), materialized_overlay / name)
    statistics = load_statistics(stats_path)
    save_statistics(
        replace(
            statistics,
            corpus=replace(
                statistics.corpus,
                n_shards_processed=3,
                n_docs=6,
            ),
            features=replace(statistics.features, n_segments=6),
        ),
        stats_path,
    )
    save_percentile_checksum(
        stats_path,
        compute_percentile_checksum(
            build_percentile_table(
                statistics.per_feature_stats,
                statistics.metadata.n_sae_features,
            )
        ),
    )
    validation = validate_feature_directory(feature_dir)
    assert validation.valid, validation.violations

    cfg.device = "cpu"
    with pytest.raises(RemotePercentileError, match="contract differs"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_remote_runner_resumes_payload_without_receipt(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    original_stats = stats_path.read_bytes()
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    # The payload upload itself succeeds, but the caller is interrupted before
    # it can publish the corresponding receipt.
    bucket.interrupt_first_payload = True
    with pytest.raises(KeyboardInterrupt, match="first payload"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert stats_path.read_bytes() == original_stats

    stable_reads: list[str] = []
    original_stable_read = remote_module.read_feature_compressed_bytes_stable

    def record_stable_read(path: Path, **kwargs) -> bytes:
        stable_reads.append(path.name)
        return original_stable_read(path, **kwargs)

    monkeypatch.setattr(
        remote_module,
        "read_feature_compressed_bytes_stable",
        record_stable_read,
    )
    bucket.operations.clear()
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    prefix = _output_prefix(bucket)
    assert bucket._path(f"{prefix}/{REMOTE_RECEIPTS_DIRNAME}/shard_00000.json").is_file()
    assert load_percentile_checksum(stats_path) is not None
    # Incomplete resume must still stable-read every raw source for the
    # canonical schema-v2 base hash. A bounded exact retained lookahead body
    # may avoid one network transfer without avoiding that reread.
    assert set(stable_reads) == {
        f"shard_{index:05d}_features.safetensors.zst" for index in range(3)
    }


def test_remote_dry_run_performs_no_download_or_write(tmp_path: Path) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    cfg.dry_run = True
    before = stats_path.read_bytes()
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert not (tmp_path / "staging").exists()
    assert stats_path.read_bytes() == before
    assert all(operation not in {"download", "upload"} for operation, _ in bucket.operations)


@pytest.mark.parametrize(
    "unknown_name",
    [
        "shard_99999_features.safetensors.zst",
        "unknown.tmp.99999999.aaaaaaaa",
    ],
)
def test_remote_dry_run_rejects_out_of_family_staging_name(
    tmp_path: Path,
    unknown_name: str,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    staging = tmp_path / "staging"
    staging.mkdir()
    unknown = staging / unknown_name
    unknown.write_bytes(b"not-owned")
    cfg = _remote_cfg(feature_dir, staging)
    cfg.dry_run = True
    with pytest.raises(RemotePercentileError, match="unknown entry"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert unknown.read_bytes() == b"not-owned"
    assert all(operation not in {"download", "upload"} for operation, _ in bucket.operations)


def test_remote_rejects_ownerless_dead_atomic_temporary_without_mutation(
    tmp_path: Path,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    staging = tmp_path / "staging"
    staging.mkdir()
    temporary = staging / "manifest.json.tmp.99999999.aaaaaaaa"
    temporary.write_bytes(b"interrupted atomic write")
    cfg = _remote_cfg(feature_dir, staging)

    cfg.dry_run = True
    with pytest.raises(RemotePercentileError, match="no owner record"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert temporary.read_bytes() == b"interrupted atomic write"


def test_remote_dry_run_rejects_partial_state_without_resume(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    bucket.interrupt_first_payload = True
    with pytest.raises(KeyboardInterrupt, match="first payload"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)

    cfg.resume = False
    cfg.dry_run = True
    bucket.operations.clear()
    with pytest.raises(RemotePercentileError, match="resume=true"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert all(operation not in {"download", "upload"} for operation, _ in bucket.operations)


def test_disabled_remote_fields_are_not_parsed(tmp_path: Path) -> None:
    cfg = OmegaConf.create({"remote": {"enabled": False, "expected_shard_count": "malformed"}})
    assert remote_percentiles_enabled(cfg) is False


@pytest.mark.parametrize("enabled", [None, 0, 1, "false"])
def test_remote_enabled_is_a_strict_bool(enabled: object) -> None:
    cfg = OmegaConf.create({"remote": {"enabled": enabled}})
    with pytest.raises(RemotePercentileError, match="must be a bool"):
        remote_percentiles_enabled(cfg)


def test_remote_rejects_auto_device_before_transport(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    cfg.device = "auto"
    with pytest.raises(RemotePercentileError, match="explicitly cpu or cuda"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert all(
        operation not in {"list", "download", "upload"} for operation, _ in bucket.operations
    )


def test_remote_rejects_force_and_non_single_rank(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    cfg.force = True
    with pytest.raises(RemotePercentileError, match="force=true"):
        run_remote_percentiles(
            cfg,
            feature_dir,
            backend=FilesystemBucket(tmp_path / "bucket"),
        )

    cfg.force = False
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(RemotePercentileError, match="WORLD_SIZE=1"):
        run_remote_percentiles(
            cfg,
            feature_dir,
            backend=FilesystemBucket(tmp_path / "bucket-2"),
        )


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("expected_shard_count", None, "positive integer"),
        ("expected_shard_count", True, "positive integer"),
        ("expected_shard_count", 0, "positive integer"),
        ("max_output_shard_bytes", False, "positive integer"),
        ("max_output_shard_bytes", -1, "positive integer"),
        ("retry_timeout_seconds", float("nan"), "positive finite"),
        ("retry_interval_seconds", 0.0, "positive finite"),
    ],
)
def test_remote_rejects_invalid_required_config(
    tmp_path: Path,
    field: str,
    value: object,
    message: str,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    cfg.remote[field] = value
    with pytest.raises(RemotePercentileError, match=message):
        run_remote_percentiles(cfg, feature_dir, backend=FilesystemBucket(tmp_path / "bucket"))


def test_remote_preserves_tokenizer_checksum_when_finalizing_root_stats(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    save_tokenizer_checksum(stats_path, "tokenizer-contract")
    statistics = load_statistics(stats_path)
    expected_percentile = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert load_tokenizer_checksum(stats_path) == "tokenizer-contract"
    assert load_percentile_checksum(stats_path) == expected_percentile


def test_remote_rejects_partial_state_without_resume(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    # A canonical contract from an interrupted run is partial state.
    contract = bucket._path("raw/pct/deadbeefdeadbeef/_remote_contract.json")
    contract.parent.mkdir(parents=True)
    contract.write_text("{}")
    # It is under another checksum prefix and therefore outside this run; the
    # current family remains fresh.  The assertion pins prefix scoping.
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0

    prefix = _output_prefix(bucket)
    bucket._path(f"{prefix}/{MANIFEST_FILENAME}").unlink()
    cfg.remote.staging_dir = str(tmp_path / "staging-second")
    with pytest.raises(RemotePercentileError, match="resume=true"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_remote_rejects_contract_tamper_and_receipt_without_payload(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    prefix = _output_prefix(bucket)

    contract_path = bucket._path(f"{prefix}/{REMOTE_CONTRACT_FILENAME}")
    original_contract = contract_path.read_bytes()
    contract_path.write_text("{}")
    with pytest.raises(RemotePercentileError, match="claim Xet hash differs"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    contract_path.write_bytes(original_contract)

    receipt_path = bucket._path(f"{prefix}/{REMOTE_RECEIPTS_DIRNAME}/shard_00000.json")
    original_receipt = receipt_path.read_bytes()
    malformed_receipt = json.loads(original_receipt)
    malformed_receipt["schema_version"] = True
    receipt_path.write_text(
        json.dumps(malformed_receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )
    with pytest.raises(RemotePercentileError, match="claim Xet hash differs"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    receipt_path.write_bytes(original_receipt)

    bucket._path(f"{prefix}/shard_00000.safetensors.zst").unlink()
    with pytest.raises(RemotePercentileError, match="without payload"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_source_mutation_before_final_fence_prevents_manifest(tmp_path: Path) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    original_stats = stats_path.read_bytes()
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    bucket.mutate_source_on_second_list = True
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    with pytest.raises(RemotePercentileError, match="source inventory changed"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert not list(bucket.root.glob("raw/pct/*/manifest.json"))
    assert stats_path.read_bytes() == original_stats


def test_local_statistics_failure_after_remote_commit_repairs_without_raw_download(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original_save = remote_module.save_percentile_checksum

    def fail_root_finalization(path: str | Path, checksum: str) -> None:
        if Path(path) == stats_path and list(bucket.root.glob("raw/pct/*/manifest.json")):
            raise OSError("injected local statistics finalization failure")
        original_save(path, checksum)

    monkeypatch.setattr(remote_module, "save_percentile_checksum", fail_root_finalization)
    with pytest.raises(OSError, match="finalization failure"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert list(bucket.root.glob("raw/pct/*/manifest.json"))
    assert load_percentile_checksum(stats_path) is None

    monkeypatch.setattr(remote_module, "save_percentile_checksum", original_save)
    bucket.operations.clear()
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert load_percentile_checksum(stats_path) is not None
    assert not [
        path
        for operation, path in bucket.operations
        if operation == "download" and path.startswith("raw/shard_")
    ]


def test_remote_runner_releases_output_tensor_before_next_shard(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    original_compute = remote_module._compute_streamed_percentile_values
    prior_values: weakref.ReferenceType[t.Tensor] | None = None
    calls = 0

    def tracked_compute(*args, **kwargs):
        nonlocal calls, prior_values
        gc.collect()
        if prior_values is not None:
            assert prior_values() is None
        values = original_compute(*args, **kwargs)
        prior_values = weakref.ref(values)
        calls += 1
        return values

    monkeypatch.setattr(
        remote_module,
        "_compute_streamed_percentile_values",
        tracked_compute,
    )
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    gc.collect()
    assert calls == 3
    assert prior_values is not None and prior_values() is None


def test_completed_remote_family_materializes_as_ordinary_overlay(tmp_path: Path) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    prefix = _output_prefix(bucket)
    manifest = json.loads(bucket._path(f"{prefix}/{MANIFEST_FILENAME}").read_text())
    checksum = manifest["checksum"]

    base_paths: list[Path] = []
    for index in range(3):
        source = bucket._path(f"raw/shard_{index:05d}_features.safetensors.zst")
        destination = feature_dir / source.name
        shutil.copyfile(source, destination)
        base_paths.append(destination)
    local_overlay = feature_dir / "pct" / checksum[:16]
    local_overlay.mkdir(parents=True)
    for name in [MANIFEST_FILENAME, *(f"shard_{i:05d}.safetensors.zst" for i in range(3))]:
        shutil.copyfile(bucket._path(f"{prefix}/{name}"), local_overlay / name)
    assert (
        discover_overlay(
            feature_dir,
            OverlayKind.PERCENTILE,
            checksum,
            base_shard_paths=base_paths,
        )
        == local_overlay
    )
    feature_index = FeatureIndex(str(feature_dir), percentile_checksum=checksum)
    segments = feature_index.lookup("shard_00000", 0)
    assert segments
    assert all(segment.feature_percentiles is not None for segment in segments)
    statistics = load_statistics(stats_path)
    save_statistics(
        replace(
            statistics,
            corpus=replace(statistics.corpus, n_shards_processed=3, n_docs=6),
            features=replace(statistics.features, n_segments=6),
        ),
        stats_path,
    )
    save_percentile_checksum(stats_path, checksum)
    validation = validate_feature_directory(feature_dir)
    assert validation.valid, validation.violations


def test_actual_remote_family_roundtrips_through_pull_materialization(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    prefix = _output_prefix(bucket)
    manifest = json.loads(bucket._path(f"{prefix}/{MANIFEST_FILENAME}").read_text())
    checksum = cast(str, manifest["checksum"])

    pulled_root = tmp_path / "pulled"
    pulled_feature_dir = pulled_root / "features" / "s16"
    pulled_feature_dir.mkdir(parents=True)
    for index in range(3):
        source = bucket._path(f"raw/shard_{index:05d}_features.safetensors.zst")
        shutil.copyfile(source, pulled_feature_dir / source.name)
    pulled_overlay = pulled_feature_dir / "pct" / checksum[:16]
    shutil.copytree(bucket._path(prefix), pulled_overlay)
    assert not (pulled_feature_dir / "feature_statistics.safetensors").exists()

    assert materialize_pulled_percentile_statistics(pulled_root) == 1
    assert (
        load_percentile_checksum(pulled_feature_dir / "feature_statistics.safetensors") == checksum
    )


def test_completed_family_rejects_every_canonical_replacement_and_claim_drift(
    tmp_path: Path,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    prefix = _output_prefix(bucket)
    canonical_paths = [
        f"{prefix}/{REMOTE_CONTRACT_FILENAME}",
        f"{prefix}/shard_00000.safetensors.zst",
        f"{prefix}/{REMOTE_RECEIPTS_DIRNAME}/shard_00000.json",
        f"{prefix}/feature_statistics.safetensors",
        f"{prefix}/{MANIFEST_FILENAME}",
    ]
    for remote_path in canonical_paths:
        path = bucket._path(remote_path)
        original = path.read_bytes()
        path.write_bytes(original + b"tampered")
        bucket.operations.clear()
        with pytest.raises(RemotePercentileError, match="claim Xet hash differs"):
            run_remote_percentiles(cfg, feature_dir, backend=bucket)
        assert not [
            path
            for operation, path in bucket.operations
            if operation == "download" and path.startswith("raw/shard_")
        ]
        path.write_bytes(original)

    contract_claim_dir = bucket._path(
        f"{prefix}/_verified_upload_claims/{REMOTE_CONTRACT_FILENAME}"
    )
    original_claim = next(contract_claim_dir.iterdir())
    conflicting_claim = contract_claim_dir / ("f" * 64 + ".json")
    shutil.copyfile(original_claim, conflicting_claim)
    with pytest.raises(RemotePercentileError, match="multiple verified-upload claims"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    conflicting_claim.unlink()

    original_claim.unlink()
    with pytest.raises(RemotePercentileError, match="lack verified-upload claims"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_claim_only_payload_is_partial_and_resumes_exactly(tmp_path: Path) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    statistics = load_statistics(next(feature_dir.glob("feature_statistics.safetensors")))
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    payload_path = f"raw/pct/{checksum[:16]}/shard_00000.safetensors.zst"
    bucket.interrupt_before_upload = payload_path
    with pytest.raises(KeyboardInterrupt, match="injected before"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    assert not bucket._path(payload_path).exists()
    assert bucket._path(
        f"raw/pct/{checksum[:16]}/_verified_upload_claims/shard_00000.safetensors.zst"
    ).is_dir()

    cfg.resume = False
    with pytest.raises(RemotePercentileError, match="partial remote percentile state"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    cfg.resume = True
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0


def test_fast_path_retries_transient_control_calls_and_refences_source(
    tmp_path: Path,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging")
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0

    bucket.operations.clear()
    bucket.transient_list_failures = 1
    bucket.transient_stat_failures = 1
    assert run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
    assert not [
        path
        for operation, path in bucket.operations
        if operation == "download" and path.startswith("raw/shard_")
    ]

    bucket.mutate_source_on_list_call = bucket.source_list_calls + 2
    with pytest.raises(RemotePercentileError, match="source inventory changed"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)


def test_fresh_saved_payload_is_reloaded_before_any_upload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
    bucket = FilesystemBucket(tmp_path / "bucket")
    _stage_sources(bucket, feature_dir)
    cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
    original_save = remote_module.save_overlay_shard

    def save_wrong_tensor(tensors, path, *, zstd_level):
        values = tensors["feature_percentiles"]
        return original_save(
            {"feature_percentiles": t.zeros_like(values)},
            path,
            zstd_level=zstd_level,
        )

    monkeypatch.setattr(remote_module, "save_overlay_shard", save_wrong_tensor)
    with pytest.raises(RemotePercentileError, match="saved staged output differs"):
        run_remote_percentiles(cfg, feature_dir, backend=bucket)
    canonical_uploads = [
        path
        for operation, path in bucket.operations
        if operation == "upload" and "_verified_upload_claims" not in path
    ]
    assert canonical_uploads == [
        next(path for path in canonical_uploads if path.endswith(REMOTE_CONTRACT_FILENAME))
    ]
    assert not list(bucket.root.glob("raw/pct/*/manifest.json"))
