"""Focused contracts for remote percentile decompression lookahead."""

from __future__ import annotations

import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path

import pytest
import torch as t
import zstandard

import steering.percentile_remote as remote
from steering.feature_io import RegularFileIdentity
from steering.overlay_io import BaseShardContentHasher
from steering.remote_feature_inventory import RemoteFeatureShard
from steering.remote_training import RemoteObjectInfo
from tests.scripts.test_production_manifest_v2 import _make_env
from tests.steering.test_percentile_remote import (
    FilesystemBucket,
    _output_prefix,
    _remote_cfg,
    _stage_sources,
)

_CHECKSUM = "a" * 64


def _config(tmp_path: Path) -> remote.RemotePercentileConfig:
    feature_dir = tmp_path / "features"
    feature_dir.mkdir(parents=True, exist_ok=True)
    return remote._resolve_config(
        _remote_cfg(feature_dir, tmp_path / "staging", resume=True),
        feature_dir=feature_dir,
        checksum=_CHECKSUM,
    )


def _canonical_output_bytes(bucket: FilesystemBucket) -> dict[str, bytes]:
    prefix = _output_prefix(bucket)
    root = bucket._path(prefix)
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file() and "_verified_upload_claims" not in path.parts
    }


class TestConfigurationAndAdmission:
    @pytest.mark.parametrize("value", [None, False, 0, -1])
    def test_cuda_requires_positive_decompressed_shard_bound(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        value: object,
    ) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        cfg = _remote_cfg(feature_dir, tmp_path / "staging")
        cfg.device = "cuda"
        cfg.remote.max_decompressed_shard_bytes = value
        monkeypatch.setattr(t.cuda, "is_available", lambda: True)

        with pytest.raises(remote.RemotePercentileError, match="positive integer"):
            remote._resolve_config(cfg, feature_dir=feature_dir, checksum=_CHECKSUM)

    def test_cpu_does_not_require_decompressed_shard_bound(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        feature_dir.mkdir()
        cfg = _remote_cfg(feature_dir, tmp_path / "staging")
        cfg.remote.max_decompressed_shard_bytes = None

        resolved = remote._resolve_config(
            cfg,
            feature_dir=feature_dir,
            checksum=_CHECKSUM,
        )

        assert resolved.max_decompressed_shard_bytes is None

    def test_decompressed_shard_bound_is_not_part_of_output_identity(
        self,
        tmp_path: Path,
    ) -> None:
        config = _config(tmp_path)
        contract = remote._producer_contract(
            config,
            inventory_sha256=_CHECKSUM,
            percentile_checksum=_CHECKSUM,
            n_sae_features=17,
        )
        changed_bound_contract = remote._producer_contract(
            replace(config, max_decompressed_shard_bytes=2_000_000),
            inventory_sha256=_CHECKSUM,
            percentile_checksum=_CHECKSUM,
            n_sae_features=17,
        )

        assert changed_bound_contract == contract

    def test_memory_envelope_and_fail_closed_admission(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        inventory = tuple(
            RemoteFeatureShard(
                index,
                RemoteObjectInfo(f"raw/shard_{index:05d}", size, f"{index + 1:064x}"),
            )
            for index, size in enumerate((100, 250, 175))
        )
        expected = 3 * 1_000 + 250 + remote.DECOMPRESSION_MEMORY_RESERVE_BYTES
        assert (
            remote.required_decompression_lookahead_memory_bytes(
                inventory,
                max_decompressed_shard_bytes=1_000,
            )
            == expected
        )

        monkeypatch.setattr(
            remote,
            "effective_available_memory_bytes",
            lambda: expected - 1,
        )
        with pytest.raises(remote.RemotePercentileError, match="memory admission failed"):
            remote._require_decompression_lookahead_memory(
                inventory,
                max_decompressed_shard_bytes=1_000,
            )

        monkeypatch.setattr(
            remote,
            "effective_available_memory_bytes",
            lambda: expected,
        )
        assert remote._require_decompression_lookahead_memory(
            inventory,
            max_decompressed_shard_bytes=1_000,
        ) == (expected, expected)

    def test_dry_run_memory_failure_precedes_staging_and_remote_writes(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
        bucket = FilesystemBucket(tmp_path / "bucket")
        _stage_sources(bucket, feature_dir)
        staging_dir = tmp_path / "staging"
        cfg = _remote_cfg(feature_dir, staging_dir)
        cfg.device = "cuda"
        cfg.dry_run = True
        monkeypatch.setattr(t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(remote, "effective_available_memory_bytes", lambda: 0)

        with pytest.raises(remote.RemotePercentileError, match="memory admission failed"):
            remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)

        assert not staging_dir.exists()
        assert not [
            operation
            for operation, _path in bucket.operations
            if operation in {"download", "upload"}
        ]


class TestBoundedPreparation:
    def test_decompressor_enforces_known_and_unknown_frame_bounds(self) -> None:
        plain = b"bounded-lookahead" * 100
        known = zstandard.ZstdCompressor().compress(plain)
        unknown = zstandard.ZstdCompressor(write_content_size=False).compress(plain)
        assert zstandard.frame_content_size(unknown) in {
            zstandard.CONTENTSIZE_UNKNOWN,
            -1,
        }

        assert remote._decompress_feature_source(
            known,
            max_decompressed_shard_bytes=len(plain),
            source_name="known",
        ) == plain
        assert remote._decompress_feature_source(
            unknown,
            max_decompressed_shard_bytes=len(plain),
            source_name="unknown",
        ) == plain
        with pytest.raises(remote.RemotePercentileError, match="declares.*exceeding"):
            remote._decompress_feature_source(
                known,
                max_decompressed_shard_bytes=len(plain) - 1,
                source_name="known",
            )
        with pytest.raises(remote.RemotePercentileError, match="configured bound"):
            remote._decompress_feature_source(
                unknown,
                max_decompressed_shard_bytes=len(plain) - 1,
                source_name="unknown",
            )
        with pytest.raises(remote.RemotePercentileError, match="invalid zstd frame"):
            remote._decompress_feature_source(
                b"not-zstd",
                max_decompressed_shard_bytes=len(plain),
                source_name="invalid",
            )

    @pytest.mark.parametrize("decompress", [False, True])
    def test_preparation_hashes_exact_compressed_bytes_and_cleans_source(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        decompress: bool,
    ) -> None:
        config = _config(tmp_path)
        plain = b"plain-safetensors-body"
        compressed = zstandard.ZstdCompressor().compress(plain)
        source = config.staging_dir / "source.safetensors.zst"
        config.staging_dir.mkdir(parents=True)
        source.write_bytes(compressed)
        identity = RegularFileIdentity.from_stat(source.stat())
        entry = RemoteFeatureShard(
            0,
            RemoteObjectInfo("raw/shard_00000_features.safetensors.zst", 1, "1" * 64),
        )
        monkeypatch.setattr(
            remote,
            "_download_source",
            lambda *_args, **_kwargs: (source, identity, "reused"),
        )
        if not decompress:
            monkeypatch.setattr(
                remote,
                "_decompress_feature_source",
                lambda *_args, **_kwargs: (_ for _ in ()).throw(
                    AssertionError("completed receipt source was decompressed")
                ),
            )
        hasher = BaseShardContentHasher()
        expected_hasher = BaseShardContentHasher()
        expected_hasher.update(Path(entry.info.path).name, compressed)

        prepared = remote._prepare_streamed_percentile_source(
            FilesystemBucket(tmp_path / "unused-bucket"),
            config=config,
            entry=entry,
            cancellation=threading.Event(),
            assert_staging=lambda: None,
            unlink_staged=lambda path, observed: (
                path.unlink()
                if RegularFileIdentity.from_stat(path.stat()) == observed
                else None
            ),
            base_hasher=hasher,
            decompress=decompress,
        )

        assert prepared.disposition == "reused"
        assert not source.exists()
        assert hasher.hexdigest() == expected_hasher.hexdigest()
        if decompress:
            assert prepared.take_safetensors_bytes() == plain
            with pytest.raises(remote.RemotePercentileError, match="no plain body"):
                prepared.take_safetensors_bytes()
        else:
            assert prepared.safetensors_bytes is None


class TestPipelineLifecycle:
    def test_pipeline_overlaps_one_worker_and_is_canonically_exact(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
        original_statistics = stats_path.read_bytes()
        sequential_bucket = FilesystemBucket(tmp_path / "sequential-bucket")
        _stage_sources(sequential_bucket, feature_dir)
        pipelined_bucket = FilesystemBucket(tmp_path / "pipelined-bucket")
        shutil.copytree(sequential_bucket.root / "raw", pipelined_bucket.root / "raw")

        sequential_cfg = _remote_cfg(feature_dir, tmp_path / "sequential-staging")
        with monkeypatch.context() as sequential:
            sequential.setattr(remote, "_decompression_lookahead_enabled", lambda *_args: False)
            assert remote.run_remote_percentiles(
                sequential_cfg,
                feature_dir,
                backend=sequential_bucket,
            ) == 0
        stats_path.write_bytes(original_statistics)

        pipelined_cfg = _remote_cfg(feature_dir, tmp_path / "pipelined-staging")
        pipelined_cfg.profile = True
        pipelined_cfg.profile_summary_output = str(tmp_path / "profile.json")
        original_prepare = remote._prepare_streamed_percentile_source
        original_compute = remote._compute_streamed_percentile_values_from_safetensors
        main_thread = threading.get_ident()
        preparation_threads: list[int] = []
        compute_threads: list[int] = []
        second_started = threading.Event()
        allow_second = threading.Event()

        def controlled_prepare(*args, **kwargs):
            preparation_threads.append(threading.get_ident())
            if kwargs["entry"].index == 1:
                second_started.set()
                assert allow_second.wait(timeout=5)
            return original_prepare(*args, **kwargs)

        def tracked_compute(*args, **kwargs):
            compute_threads.append(threading.get_ident())
            if len(compute_threads) == 1:
                assert second_started.wait(timeout=5)
                allow_second.set()
            return original_compute(*args, **kwargs)

        monkeypatch.setattr(remote, "_decompression_lookahead_enabled", lambda *_args: True)
        monkeypatch.setattr(remote, "effective_available_memory_bytes", lambda: 1 << 50)
        monkeypatch.setattr(remote, "_prepare_streamed_percentile_source", controlled_prepare)
        monkeypatch.setattr(
            remote,
            "_compute_streamed_percentile_values_from_safetensors",
            tracked_compute,
        )

        assert remote.run_remote_percentiles(
            pipelined_cfg,
            feature_dir,
            backend=pipelined_bucket,
        ) == 0

        pipelined_outputs = _canonical_output_bytes(pipelined_bucket)
        sequential_outputs = _canonical_output_bytes(sequential_bucket)
        pipelined_manifest = json.loads(pipelined_outputs.pop("manifest.json"))
        sequential_manifest = json.loads(sequential_outputs.pop("manifest.json"))
        pipelined_outputs.pop(remote.OVERLAY_STATISTICS_FILENAME)
        sequential_outputs.pop(remote.OVERLAY_STATISTICS_FILENAME)
        assert pipelined_outputs == sequential_outputs
        assert pipelined_manifest["checksum"] == sequential_manifest["checksum"]
        assert pipelined_manifest["kind"] == sequential_manifest["kind"]
        assert (
            pipelined_manifest["base_shard_content_hash"]
            == sequential_manifest["base_shard_content_hash"]
        )
        assert remote._statistics_checksum(
            pipelined_bucket._path(
                f"{_output_prefix(pipelined_bucket)}/{remote.OVERLAY_STATISTICS_FILENAME}"
            )
        ) == remote._statistics_checksum(
            sequential_bucket._path(
                f"{_output_prefix(sequential_bucket)}/{remote.OVERLAY_STATISTICS_FILENAME}"
            )
        )
        assert len(preparation_threads) == 3
        assert len(set(preparation_threads)) == 1
        assert set(preparation_threads).isdisjoint({main_thread})
        assert compute_threads == [main_thread] * 3
        assert not any(
            thread.name.startswith("percentile-remote-")
            for thread in threading.enumerate()
        )
        profile = json.loads((tmp_path / "profile.json").read_text())
        assert profile["config"]["decompression_lookahead_enabled"] is True
        assert profile["staging"]["lookahead_memory_required_bytes"] > 0
        assert profile["staging"]["lookahead_memory_available_bytes"] == 1 << 50

    def test_completed_receipt_is_hashed_without_decompression_on_resume(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir, stats_path = _make_env(tmp_path / "logical", n_shards=3)
        bucket = FilesystemBucket(tmp_path / "bucket")
        _stage_sources(bucket, feature_dir)
        cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
        checksum, _ = remote._statistics_checksum(stats_path)
        bucket.interrupt_after_upload = (
            f"raw/pct/{checksum[: remote.CHECKSUM_PREFIX_LEN]}"
            "/_receipts/shard_00000.json"
        )
        with pytest.raises(KeyboardInterrupt, match="injected interruption"):
            remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)

        original_prepare = remote._prepare_streamed_percentile_source
        scheduled: list[tuple[int, bool]] = []

        def tracked_prepare(*args, **kwargs):
            scheduled.append((kwargs["entry"].index, kwargs["decompress"]))
            return original_prepare(*args, **kwargs)

        monkeypatch.setattr(remote, "_decompression_lookahead_enabled", lambda *_args: True)
        monkeypatch.setattr(remote, "effective_available_memory_bytes", lambda: 1 << 50)
        monkeypatch.setattr(remote, "_prepare_streamed_percentile_source", tracked_prepare)

        assert remote.run_remote_percentiles(cfg, feature_dir, backend=bucket) == 0
        assert scheduled == [(0, False), (1, True), (2, True)]

    def test_primary_compute_error_joins_and_reports_lookahead_failure(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_dir, _ = _make_env(tmp_path / "logical", n_shards=3)
        bucket = FilesystemBucket(tmp_path / "bucket")
        _stage_sources(bucket, feature_dir)
        cfg = _remote_cfg(feature_dir, tmp_path / "staging", resume=True)
        original_prepare = remote._prepare_streamed_percentile_source
        second_started = threading.Event()
        fail_second = threading.Event()

        def controlled_prepare(*args, **kwargs):
            if kwargs["entry"].index == 1:
                second_started.set()
                assert fail_second.wait(timeout=5)
                raise OSError("secondary lookahead failure")
            return original_prepare(*args, **kwargs)

        def fail_compute(*_args, **_kwargs):
            assert second_started.wait(timeout=5)
            fail_second.set()
            raise RuntimeError("primary percentile compute failure")

        monkeypatch.setattr(remote, "_decompression_lookahead_enabled", lambda *_args: True)
        monkeypatch.setattr(remote, "effective_available_memory_bytes", lambda: 1 << 50)
        monkeypatch.setattr(remote, "_prepare_streamed_percentile_source", controlled_prepare)
        monkeypatch.setattr(
            remote,
            "_compute_streamed_percentile_values_from_safetensors",
            fail_compute,
        )

        with pytest.raises(RuntimeError, match="primary percentile compute failure") as caught:
            remote.run_remote_percentiles(cfg, feature_dir, backend=bucket)

        assert any(
            "secondary lookahead failure" in note
            for note in getattr(caught.value, "__notes__", ())
        )
        assert not any(
            thread.name.startswith("percentile-remote-")
            for thread in threading.enumerate()
        )
