"""Functional parity and publication tests for remote token boundaries."""

from __future__ import annotations

import json
import shutil
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast
from unittest.mock import patch

import hf_xet
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from scripts.preprocess import precompute_token_boundaries as token_module
from steering.feature_io import (
    CorruptFeatureShardError,
    glob_features,
    load_features,
    save_features,
)
from steering.feature_paths import (
    DatasetPhase,
    PoolingMode,
    SegmentSizeMismatch,
    build_feature_dir,
)
from steering.overlay_io import (
    BASE_SHARD_HASH_CACHE_FILENAME,
    MANIFEST_FILENAME,
    OverlayIncompleteError,
    OverlayKind,
    OverlayStaleError,
    compute_base_shard_content_hash,
    discover_overlay,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
)
from steering.remote_feature_inventory import (
    RemoteShardFamily,
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
)
from steering.remote_training import RemoteObjectInfo
from steering.remote_types import NormalizedRemotePrefix
from steering.sae_resolver import ActivationType, SAEWidth
from steering.segmentation import compute_tokenizer_checksum
from steering.token_boundary_remote import TokenBoundaryRemoteError
from steering.token_boundary_source import (
    MissingTokenBoundarySourceError,
    TokenBoundarySourceManifest,
    compact_token_boundary_payload_from_raw,
    derive_token_boundary_source_prefix,
    save_token_boundary_source_shard,
    serialize_token_boundary_source_manifest,
    token_boundary_source_manifest_path,
)
from tests.scripts.test_token_boundaries_manifest_v2 import (
    SEGMENT_SIZE,
    _make_env,
    _SimpleTokenizer,
)


class FakeBucket:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.downloads: list[str] = []
        self.list_calls = 0
        self.list_calls_by_prefix: dict[str, int] = {}
        self.mutate_prefix_on_second_list: str | None = None
        self.stat_calls_by_path: dict[str, int] = {}
        self.mutate_manifest_on_second_stat = False

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
        calls = self.stat_calls_by_path.get(remote_path, 0) + 1
        self.stat_calls_by_path[remote_path] = calls
        if (
            self.mutate_manifest_on_second_stat
            and remote_path.endswith("/manifest.json")
            and calls == 2
        ):
            path.write_bytes(path.read_bytes() + b" ")
        return self._info(path, remote_path) if path.is_file() else None

    def list(self, prefix: str) -> Sequence[RemoteObjectInfo]:
        self.list_calls += 1
        prefix_calls = self.list_calls_by_prefix.get(prefix, 0) + 1
        self.list_calls_by_prefix[prefix] = prefix_calls
        base = self._path(prefix)
        if self.mutate_prefix_on_second_list == prefix and prefix_calls == 2:
            (base / "shard_00002_features.safetensors.zst").write_bytes(b"late")
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
        raise AssertionError("remote token-boundary production must never upload")


def _canonical_dir(root: Path) -> Path:
    return build_feature_dir(
        root=root,
        segment_size=SEGMENT_SIZE,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
    )


def _publish_compact_fixture(bucket: FakeBucket, *, raw_prefix: str) -> str:
    normalized_raw = NormalizedRemotePrefix.parse(raw_prefix)
    raw_inventory = normalize_remote_feature_inventory(
        bucket.list(raw_prefix),
        prefix=normalized_raw,
        family=RemoteShardFamily.RAW_FEATURES,
    )
    raw_digest = remote_feature_inventory_digest(raw_inventory)
    compact_prefix = derive_token_boundary_source_prefix(normalized_raw, raw_digest)
    raw_paths = []
    for entry in raw_inventory:
        raw_path = bucket._path(entry.info.path)
        raw_paths.append(raw_path)
        payload = compact_token_boundary_payload_from_raw(load_features(raw_path))
        save_token_boundary_source_shard(
            payload,
            bucket._path(f"{compact_prefix}/{raw_path.name}"),
        )
    compact_inventory = normalize_remote_feature_inventory(
        bucket.list(str(compact_prefix)),
        prefix=compact_prefix,
        family=RemoteShardFamily.RAW_FEATURES,
    )
    manifest = TokenBoundarySourceManifest(
        base_shard_content_hash=compute_base_shard_content_hash(raw_paths),
        raw_inventory_sha256=raw_digest,
        schema_version=1,
        shard_count=len(raw_inventory),
        source_inventory_sha256=remote_feature_inventory_digest(compact_inventory),
    )
    manifest_path = bucket._path(token_boundary_source_manifest_path(compact_prefix))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(serialize_token_boundary_source_manifest(manifest))
    bucket.list_calls = 0
    bucket.list_calls_by_prefix.clear()
    return str(compact_prefix)


def _compact_manifest_path(bucket: FakeBucket) -> Path:
    matches = list(bucket.root.glob("raw/_token_boundary_source/v1/*/manifest.json"))
    assert len(matches) == 1
    return matches[0]


def _setup(
    tmp_path: Path,
    *,
    n_shards: int = 2,
) -> tuple[Path, Path, Path, _SimpleTokenizer, FakeBucket]:
    source, data_dir, tokenizer = _make_env(tmp_path / "source", n_shards=n_shards)
    if n_shards > 1:
        second = source / "shard_00001_features.safetensors.zst"
        data = load_features(second)
        permutation = t.tensor([2, 0, 3, 1], dtype=t.int64)
        original_offsets = data["offsets"]
        feature_ids = []
        feature_mags = []
        offsets = [0]
        keep_counts = (1, 2, 3, 1)
        for old_index, keep in zip(permutation.tolist(), keep_counts, strict=True):
            start = int(original_offsets[old_index].item())
            end = int(original_offsets[old_index + 1].item())
            feature_ids.append(data["feature_ids"][start:end][:keep])
            feature_mags.append(data["feature_mags"][start:end][:keep])
            offsets.append(offsets[-1] + keep)
        data["feature_ids"] = t.cat(feature_ids)
        data["feature_mags"] = t.cat(feature_mags)
        data["offsets"] = t.tensor(offsets, dtype=t.int64)
        for key in (
            "doc_ids",
            "segment_indices",
            "segment_positions",
            "segment_char_starts",
            "segment_char_ends",
        ):
            data[key] = data[key][permutation]
        data["doc_token_counts"] = t.tensor([41, 1024], dtype=t.int32)
        save_features(data, second)
    oracle = _canonical_dir(tmp_path / "oracle-root")
    logical = _canonical_dir(tmp_path / "logical-root")
    oracle.mkdir(parents=True)
    logical.mkdir(parents=True)
    bucket = FakeBucket(tmp_path / "bucket")
    bucket_prefix = bucket._path("raw")
    bucket_prefix.mkdir(parents=True)
    for path in glob_features(source):
        shutil.copyfile(path, oracle / path.name)
        shutil.copyfile(path, bucket_prefix / path.name)
    _publish_compact_fixture(bucket, raw_prefix="raw")
    return oracle, logical, data_dir, tokenizer, bucket


def _cfg(
    feature_dir: Path,
    data_dir: Path,
    staging: Path,
    *,
    remote: bool,
    n_shards: int = 2,
    **overrides: object,
):
    values: dict[str, object] = {
        "feature_dir": str(feature_dir),
        "segment_size": SEGMENT_SIZE,
        "data_dir": str(data_dir),
        "dry_run": False,
        "resume": False,
        "force": False,
        "workers": 1,
        "profile": False,
        "profile_trace": False,
    }
    if remote:
        values["remote"] = {
            "enabled": True,
            "bucket_id": "owner/test-bucket",
            "local_features_root": str(feature_dir),
            "remote_features_root": "raw",
            "staging_dir": str(staging),
            "expected_shard_count": n_shards,
            "retry_timeout_seconds": 1.0,
            "retry_interval_seconds": 0.001,
        }
    values.update(overrides)
    return OmegaConf.create(values)


def _run(cfg, tokenizer: _SimpleTokenizer, bucket: FakeBucket | None = None) -> int:
    patches = [
        patch.object(token_module, "get_tokenizer", return_value=tokenizer),
        patch("steering.overlay_io.time.gmtime", return_value=(2026, 1, 2, 3, 4, 5, 4, 2, 0)),
    ]
    if bucket is not None:
        patches.append(
            patch(
                "steering.token_boundary_remote.HfBucketBackend",
                return_value=bucket,
            )
        )
    with patches[0], patches[1]:
        if len(patches) == 3:
            with patches[2]:
                return token_module.run(cfg)
        return token_module.run(cfg)


def test_remote_missing_compact_manifest_fails_before_body_or_semantic_output(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    _compact_manifest_path(bucket).unlink()
    bucket.downloads.clear()
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)

    with pytest.raises(
        MissingTokenBoundarySourceError,
        match=r"raw_prefix=raw.*expected_compact_prefix=.*publish_token_boundary_source",
    ):
        _run(cfg, tokenizer, bucket)
    assert bucket.downloads == []
    assert not (logical / "tok").exists()


@pytest.mark.parametrize("field", ["raw_inventory_sha256", "source_inventory_sha256"])
def test_remote_rejects_mismatched_compact_manifest_binding(
    tmp_path: Path,
    field: str,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    manifest_path = _compact_manifest_path(bucket)
    manifest = json.loads(manifest_path.read_text())
    manifest[field] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n")
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)

    with pytest.raises(ValueError, match="inventory digest mismatch"):
        _run(cfg, tokenizer, bucket)
    assert len(bucket.downloads) == 1
    assert not (logical / "tok").exists()


def test_remote_artifacts_are_byte_identical_to_local_oracle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    assert (
        len(
            {
                path.stat().st_size
                for path in bucket._path("raw").glob("shard_*_features.safetensors.zst")
            }
        )
        == 2
    )
    assert (
        len(
            {
                path.read_bytes()
                for path in _compact_manifest_path(bucket).parent.glob(
                    "shard_*_features.safetensors.zst"
                )
            }
        )
        == 2
    )
    checksum = compute_tokenizer_checksum(tokenizer)
    oracle_stats = oracle / "feature_statistics.safetensors"
    remote_stats = logical / "feature_statistics.safetensors"
    save_file(
        {"sentinel": t.tensor([1, 2], dtype=t.int64)},
        str(oracle_stats),
        metadata={"source": "synthetic", "tokenizer_checksum": checksum},
    )
    shutil.copyfile(oracle_stats, remote_stats)
    stats_before = (logical / "feature_statistics.safetensors").read_bytes()
    assert (
        _run(
            _cfg(oracle, data_dir, tmp_path / "unused", remote=False),
            tokenizer,
        )
        == 0
    )
    real_precompute_shard = token_module.precompute_shard
    preloaded_calls = 0

    def check_preloaded_immutable(
        *args: Any,
        preloaded_data: Mapping[str, t.Tensor] | None = None,
        **kwargs: Any,
    ) -> bool:
        nonlocal preloaded_calls
        before = (
            None
            if preloaded_data is None
            else {key: value.clone() for key, value in preloaded_data.items()}
        )
        result = cast(Any, real_precompute_shard)(
            *args,
            preloaded_data=preloaded_data,
            **kwargs,
        )
        if preloaded_data is not None:
            assert before is not None
            preloaded_calls += 1
            assert set(before) == set(preloaded_data)
            for key, value in before.items():
                assert t.equal(value, preloaded_data[key])
        return result

    monkeypatch.setattr(token_module, "precompute_shard", check_preloaded_immutable)
    with monkeypatch.context() as no_raw_hash:
        no_raw_hash.setattr(
            "steering.overlay_io.BaseShardContentHasher",
            lambda: (_ for _ in ()).throw(
                AssertionError("compact consumer constructed a raw base hasher")
            ),
        )
        assert (
            _run(
                _cfg(logical, data_dir, tmp_path / "staging", remote=True),
                tokenizer,
                bucket,
            )
            == 0
        )
    assert preloaded_calls == 2

    oracle_output = overlay_dir(oracle, OverlayKind.SEG_BOUNDARY, checksum)
    remote_output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    assert (remote_output / MANIFEST_FILENAME).read_bytes() == (
        oracle_output / MANIFEST_FILENAME
    ).read_bytes()
    for index in range(2):
        stem = f"shard_{index:05d}"
        remote_path = overlay_shard_path(remote_output, stem)
        oracle_path = overlay_shard_path(oracle_output, stem)
        assert remote_path.read_bytes() == oracle_path.read_bytes()
        remote_tensors = load_overlay_shard(remote_path)
        oracle_tensors = load_overlay_shard(oracle_path)
        # safetensors exposes a mapping, not a serialized-key-order contract;
        # identical bytes can be materialized with different hash-map iteration
        # order across consecutive loads.
        assert set(remote_tensors) == set(oracle_tensors)
        for key in oracle_tensors:
            assert remote_tensors[key].dtype == oracle_tensors[key].dtype
            assert remote_tensors[key].shape == oracle_tensors[key].shape
            assert t.equal(remote_tensors[key], oracle_tensors[key])
    assert glob_features(logical) == []
    assert not (logical / BASE_SHARD_HASH_CACHE_FILENAME).exists()
    assert len(bucket.downloads) == 3
    assert not any(path.startswith("raw/shard_") for path in bucket.downloads)

    assert remote_stats.read_bytes() == oracle_stats.read_bytes() == stats_before
    assert set(load_file(str(remote_stats))) == set(load_file(str(oracle_stats)))
    assert t.equal(load_file(str(remote_stats))["sentinel"], t.tensor([1, 2]))
    with safe_open(oracle_stats, framework="pt") as oracle_handle:
        oracle_metadata = oracle_handle.metadata()
    with safe_open(remote_stats, framework="pt") as remote_handle:
        remote_metadata = remote_handle.metadata()
    assert remote_metadata == oracle_metadata
    assert remote_metadata["tokenizer_checksum"] == checksum

    for remote_path in sorted(bucket._path("raw").glob("*.safetensors.zst")):
        shutil.copyfile(remote_path, logical / remote_path.name)
    assert (
        discover_overlay(
            logical,
            OverlayKind.SEG_BOUNDARY,
            checksum,
            base_shard_paths=glob_features(logical),
        )
        == remote_output
    )


def test_remote_spawn_workers_match_sequential_oracle(tmp_path: Path) -> None:
    oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    profile_path = tmp_path / "spawn-profile.json"
    assert (
        _run(
            _cfg(oracle, data_dir, tmp_path / "unused", remote=False, workers=1),
            tokenizer,
        )
        == 0
    )
    assert (
        _run(
            _cfg(
                logical,
                data_dir,
                tmp_path / "staging",
                remote=True,
                workers=2,
                profile=True,
                profile_summary_output=str(profile_path),
            ),
            tokenizer,
            bucket,
        )
        == 0
    )

    checksum = compute_tokenizer_checksum(tokenizer)
    oracle_output = overlay_dir(oracle, OverlayKind.SEG_BOUNDARY, checksum)
    remote_output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    for index in range(2):
        stem = f"shard_{index:05d}"
        assert overlay_shard_path(remote_output, stem).read_bytes() == (
            overlay_shard_path(oracle_output, stem).read_bytes()
        )
    profile = json.loads(profile_path.read_text())
    assert profile["totals"]["status"] == "success"
    assert profile["worker_pool"]["spawn_count"] == 1
    assert profile["manifest_pre_pass"] == {
        "seconds": profile["manifest_pre_pass"]["seconds"],
        "bytes_read": 0,
        "cache_hit": True,
        "n_shards_hashed": 2,
    }
    assert profile["manifest_pre_pass"]["seconds"] >= 0.0
    assert profile["config"]["compact_prefix"].startswith("raw/_token_boundary_source/v1/")
    compact_sizes = {
        path.name.split("_features", 1)[0]: path.stat().st_size
        for path in _compact_manifest_path(bucket).parent.glob("shard_*_features.safetensors.zst")
    }
    assert {shard["stem"]: shard["load_bytes"] for shard in profile["per_shard"]} == compact_sizes
    assert all(shard["load_features_seconds"] >= 0.0 for shard in profile["per_shard"])
    assert profile["remote_transport"]["final_fence_status"] == "passed"
    assert profile["remote_transport"]["compact_final_fence_status"] == "passed"
    assert profile["remote_transport"]["raw_inventory_count"] == 2
    assert profile["remote_transport"]["inventory_count"] == 2


def test_compact_metadata_and_bos_helper_cover_fail_closed_branches(
    tmp_path: Path,
) -> None:
    _oracle, _logical, data_dir, _tokenizer, bucket = _setup(tmp_path)
    compact = compact_token_boundary_payload_from_raw(
        load_features(bucket._path("raw/shard_00001_features.safetensors.zst"))
    )
    compact["doc_token_counts"] = t.tensor([1024, 41], dtype=t.int32)
    assert (
        token_module._detect_source_bos_char_len_from_data(
            compact,
            data_dir / "shard_00001.parquet",
        )
        >= 0
    )
    compact["doc_token_counts"] = t.tensor([1024, 1024], dtype=t.int32)
    with pytest.raises(ValueError, match="paired.*parquet|from .*parquet"):
        token_module._detect_source_bos_char_len_from_data(
            compact,
            data_dir / "shard_00001.parquet",
        )

    with pytest.raises(ValueError, match="noncanonical shard"):
        token_module._compact_shard_metadata(
            index=0,
            shard_path=Path("wrong-name.safetensors.zst"),
            data=compact,
            expected_segment_size=SEGMENT_SIZE,
        )
    with pytest.raises(SegmentSizeMismatch, match="compact=129"):
        token_module._compact_shard_metadata(
            index=0,
            shard_path=Path("shard_00000_features.safetensors.zst"),
            data={**compact, "segment_size": t.tensor(129, dtype=t.int64)},
            expected_segment_size=SEGMENT_SIZE,
        )


def test_remote_profile_trace_uses_preloaded_compact_payload(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    exported: list[str] = []

    class FakeProfile:
        def __enter__(self):
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def export_chrome_trace(self, path: str) -> None:
            exported.append(path)

    monkeypatch.setattr(token_module, "torch_profile", lambda **_kwargs: FakeProfile())
    trace_path = tmp_path / "trace.json"
    summary_path = tmp_path / "summary.json"
    assert (
        _run(
            _cfg(
                logical,
                data_dir,
                tmp_path / "staging",
                remote=True,
                profile=True,
                profile_trace=True,
                profile_output=str(trace_path),
                profile_summary_output=str(summary_path),
            ),
            tokenizer,
            bucket,
        )
        == 0
    )
    assert exported == [str(trace_path)]
    assert json.loads(summary_path.read_text())["totals"]["status"] == "success"


def test_remote_worker_initialization_failure_restores_environment_and_cleans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    for name in token_module._WORKER_THREAD_ENV_VARS:
        monkeypatch.setenv(name, f"sentinel-{name}")

    def fail_pool(*_args: object, **_kwargs: object):
        raise RuntimeError("injected worker initialization failure")

    monkeypatch.setattr(token_module, "ProcessPoolExecutor", fail_pool)
    with pytest.raises(RuntimeError, match="worker initialization failure"):
        _run(
            _cfg(
                logical,
                data_dir,
                tmp_path / "staging",
                remote=True,
                workers=2,
            ),
            tokenizer,
            bucket,
        )
    for name in token_module._WORKER_THREAD_ENV_VARS:
        assert token_module.os.environ[name] == f"sentinel-{name}"
    output = overlay_dir(
        logical,
        OverlayKind.SEG_BOUNDARY,
        compute_tokenizer_checksum(tokenizer),
    )
    assert not output.exists()
    assert not any(logical.rglob(".seg-boundary-candidates.*"))
    assert list((tmp_path / "staging").iterdir()) == []


def test_remote_candidate_validation_failure_publishes_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    real_validate = token_module.validate_token_boundary_overlay_shard

    def fail_candidate(path: Path, *, expected_n_segments: int):
        if path.parent.name.startswith(".seg-boundary-candidates."):
            raise ValueError("injected candidate validation failure")
        return real_validate(path, expected_n_segments=expected_n_segments)

    monkeypatch.setattr(
        token_module,
        "validate_token_boundary_overlay_shard",
        fail_candidate,
    )
    with pytest.raises(ValueError, match="candidate validation failure"):
        _run(
            _cfg(logical, data_dir, tmp_path / "staging", remote=True),
            tokenizer,
            bucket,
        )
    output = overlay_dir(
        logical,
        OverlayKind.SEG_BOUNDARY,
        compute_tokenizer_checksum(tokenizer),
    )
    assert not output.exists()
    assert not any(logical.rglob(".seg-boundary-candidates.*"))
    assert list((tmp_path / "staging").iterdir()) == []


def test_remote_and_local_omit_empty_shard_identically(tmp_path: Path) -> None:
    oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    empty_path = oracle / "shard_00001_features.safetensors.zst"
    empty = load_features(empty_path)
    for key, value in list(empty.items()):
        if hasattr(value, "shape"):
            empty[key] = value[:0]
    empty["offsets"] = empty["offsets"].new_zeros(1)
    empty["n_docs"] = 0
    empty["n_segments"] = 0
    save_features(empty, empty_path)
    shutil.copyfile(empty_path, bucket._path(f"raw/{empty_path.name}"))
    _publish_compact_fixture(bucket, raw_prefix="raw")

    assert (
        _run(
            _cfg(oracle, data_dir, tmp_path / "unused", remote=False),
            tokenizer,
        )
        == 0
    )
    assert (
        _run(
            _cfg(logical, data_dir, tmp_path / "staging", remote=True),
            tokenizer,
            bucket,
        )
        == 0
    )
    checksum = compute_tokenizer_checksum(tokenizer)
    oracle_output = overlay_dir(oracle, OverlayKind.SEG_BOUNDARY, checksum)
    remote_output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    assert (remote_output / MANIFEST_FILENAME).read_bytes() == (
        oracle_output / MANIFEST_FILENAME
    ).read_bytes()
    assert not overlay_shard_path(remote_output, "shard_00001").exists()
    assert not overlay_shard_path(oracle_output, "shard_00001").exists()


def test_remote_resume_preserves_manifest_bytes(tmp_path: Path) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    manifest_before = (output / MANIFEST_FILENAME).read_bytes()
    overlay_shard_path(output, "shard_00001").unlink()
    (data_dir / "shard_00000.parquet").unlink()

    cfg.resume = True
    bucket.downloads.clear()
    assert _run(cfg, tokenizer, bucket) == 0
    assert (output / MANIFEST_FILENAME).read_bytes() == manifest_before
    assert overlay_shard_path(output, "shard_00001").is_file()
    assert len(bucket.downloads) == 3, "remote resume reauthenticates the full family"
    assert not any(path.startswith("raw/shard_") for path in bucket.downloads)


def test_remote_resume_rejects_corrupt_present_payload_without_mutation(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    manifest_before = (output / MANIFEST_FILENAME).read_bytes()
    corrupt = overlay_shard_path(output, "shard_00000")
    corrupt.write_bytes(b"corrupt present payload")
    cfg.resume = True

    with pytest.raises(CorruptFeatureShardError, match="unreadable"):
        _run(cfg, tokenizer, bucket)
    assert corrupt.read_bytes() == b"corrupt present payload"
    assert (output / MANIFEST_FILENAME).read_bytes() == manifest_before
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))


def test_remote_complete_v2_noop_needs_no_parquets_and_preserves_bytes(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    before = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    for parquet in data_dir.glob("*.parquet"):
        parquet.unlink()
    bucket.downloads.clear()

    assert _run(cfg, tokenizer, bucket) == 0
    after = {path.name: path.read_bytes() for path in output.iterdir() if path.is_file()}
    assert after == before
    assert len(bucket.downloads) == 3


def test_remote_incomplete_v2_without_resume_authenticates_then_fails(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    manifest_before = (output / MANIFEST_FILENAME).read_bytes()
    missing = overlay_shard_path(output, "shard_00001")
    missing.unlink()

    with pytest.raises(OverlayIncompleteError, match="Use resume=true"):
        _run(cfg, tokenizer, bucket)
    assert (output / MANIFEST_FILENAME).read_bytes() == manifest_before
    assert not missing.exists()


def test_remote_legacy_v1_resume_fills_body_without_restamping_manifest(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    manifest_path = output / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest.pop("schema_version")
    manifest.pop("base_shard_content_hash")
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_before = manifest_path.read_bytes()
    missing = overlay_shard_path(output, "shard_00000")
    missing.unlink()
    cfg.resume = True

    assert _run(cfg, tokenizer, bucket) == 0
    assert missing.is_file()
    assert manifest_path.read_bytes() == manifest_before


def test_remote_force_replaces_corrupt_body_and_writes_fresh_v2_manifest(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    body = overlay_shard_path(output, "shard_00000")
    valid_body = body.read_bytes()
    body.write_bytes(b"corrupt")
    cfg.force = True
    cfg.resume = True

    assert _run(cfg, tokenizer, bucket) == 0
    assert body.read_bytes() == valid_body
    assert json.loads((output / MANIFEST_FILENAME).read_text())["schema_version"] == 2


@pytest.mark.parametrize(
    ("shard_start", "shard_end"),
    [(0, 1), (1, -1), (1, 1)],
)
def test_remote_partial_force_fails_before_download_or_output_mutation(
    tmp_path: Path,
    shard_start: int,
    shard_end: int,
) -> None:
    oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    output_before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }

    bucket.downloads.clear()
    cfg.force = True
    cfg.shard_start = shard_start
    cfg.shard_end = shard_end

    with pytest.raises(ValueError, match="complete sealed shard range"):
        _run(cfg, tokenizer, bucket)
    assert len(bucket.downloads) == 1
    assert bucket.downloads[0].endswith("/manifest.json")
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == output_before
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))


def test_remote_fresh_partial_rejects_unselected_payloads_without_manifest(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    (output / MANIFEST_FILENAME).unlink()
    unselected = overlay_shard_path(output, "shard_00001")
    unselected.write_bytes(b"unauthenticated payload residue")
    output_before = {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    }
    bucket.downloads.clear()
    cfg.shard_start = 0
    cfg.shard_end = 1

    with pytest.raises(OverlayIncompleteError, match="unauthenticated unselected"):
        _run(cfg, tokenizer, bucket)
    assert len(bucket.downloads) == 1
    assert bucket.downloads[0].endswith("/manifest.json")
    assert {
        path.relative_to(output): path.read_bytes() for path in output.rglob("*") if path.is_file()
    } == output_before
    assert not (output / MANIFEST_FILENAME).exists()


def test_remote_reclassifies_output_after_acquiring_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    assert (
        _run(
            _cfg(oracle, data_dir, tmp_path / "unused", remote=False),
            tokenizer,
        )
        == 0
    )
    checksum = compute_tokenizer_checksum(tokenizer)
    oracle_output = overlay_dir(oracle, OverlayKind.SEG_BOUNDARY, checksum)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    old_manifest = (oracle_output / MANIFEST_FILENAME).read_bytes()

    real_lock = token_module.remote_output_lock

    @contextmanager
    def publish_stale_overlay_while_acquiring(lock_path: Path):
        with real_lock(lock_path):
            shutil.copytree(oracle_output, output)
            stale_manifest_path = output / MANIFEST_FILENAME
            stale_manifest = json.loads(stale_manifest_path.read_text())
            stale_manifest["base_shard_content_hash"] = "f" * 64
            stale_manifest_path.write_text(json.dumps(stale_manifest, indent=2) + "\n")
            yield

    monkeypatch.setattr(
        token_module,
        "remote_output_lock",
        publish_stale_overlay_while_acquiring,
    )
    cfg = _cfg(
        logical,
        data_dir,
        tmp_path / "staging",
        remote=True,
        shard_start=0,
        shard_end=1,
    )

    with pytest.raises(OverlayStaleError):
        _run(cfg, tokenizer, bucket)
    assert (output / MANIFEST_FILENAME).read_bytes() != old_manifest
    assert len(bucket.downloads) == 3
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))


def test_remote_shard_range_still_streams_complete_family(tmp_path: Path) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path, n_shards=3)
    cfg = _cfg(
        logical,
        data_dir,
        tmp_path / "staging",
        remote=True,
        n_shards=3,
        shard_start=1,
        shard_end=2,
    )
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    assert overlay_shard_path(output, "shard_00001").is_file()
    assert not overlay_shard_path(output, "shard_00000").exists()
    assert not overlay_shard_path(output, "shard_00002").exists()
    assert len(bucket.downloads) == 4


def test_remote_empty_selected_range_authenticates_family_without_payloads(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    for parquet in data_dir.glob("*.parquet"):
        parquet.unlink()
    cfg = _cfg(
        logical,
        data_dir,
        tmp_path / "staging",
        remote=True,
        shard_start=1,
        shard_end=1,
    )
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    assert (output / MANIFEST_FILENAME).is_file()
    assert list(output.glob("*.safetensors.zst")) == []
    assert len(bucket.downloads) == 3


def test_stale_remote_manifest_discards_private_candidate(tmp_path: Path) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    manifest_path = output / MANIFEST_FILENAME
    manifest = json.loads(manifest_path.read_text())
    manifest["base_shard_content_hash"] = "f" * 64
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    manifest_before = manifest_path.read_bytes()
    missing = overlay_shard_path(output, "shard_00000")
    missing.unlink()
    cfg.resume = True

    with pytest.raises(OverlayStaleError):
        _run(cfg, tokenizer, bucket)
    assert manifest_path.read_bytes() == manifest_before
    assert not missing.exists()
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))


def test_remote_dry_run_publishes_nothing(tmp_path: Path) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(
        logical,
        data_dir,
        tmp_path / "staging",
        remote=True,
        dry_run=True,
    )
    assert _run(cfg, tokenizer, bucket) == 0
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    assert not output.exists()
    assert len(bucket.downloads) == 3


@pytest.mark.parametrize("failure_point", ["second_body_move", "manifest_write"])
def test_remote_fresh_publication_failure_never_exposes_valid_overlay_and_restarts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure_point: str,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    real_replace = token_module.os.replace
    canonical_moves = 0

    def fail_second_body_move(source, destination) -> None:
        nonlocal canonical_moves
        source_path = Path(source)
        destination_path = Path(destination)
        if (
            source_path.parent.name.startswith(".seg-boundary-candidates.")
            and destination_path.parent == output
        ):
            canonical_moves += 1
            if canonical_moves == 2:
                raise OSError("simulated second candidate move failure")
        real_replace(source, destination)

    def fail_manifest_write(*_args: object, **_kwargs: object) -> None:
        raise OSError("simulated manifest write failure")

    with monkeypatch.context() as failure:
        if failure_point == "second_body_move":
            failure.setattr(token_module.os, "replace", fail_second_body_move)
            expected_error = "second candidate move failure"
        else:
            failure.setattr(token_module, "write_manifest", fail_manifest_write)
            expected_error = "manifest write failure"
        with pytest.raises(OSError, match=expected_error):
            _run(cfg, tokenizer, bucket)

    assert not (output / MANIFEST_FILENAME).exists()
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))
    assert list((tmp_path / "staging").iterdir()) == []
    if failure_point == "second_body_move":
        assert len(list(output.glob("*.safetensors.zst"))) == 1
    else:
        assert len(list(output.glob("*.safetensors.zst"))) == 2

    bucket.downloads.clear()
    assert _run(cfg, tokenizer, bucket) == 0
    assert (output / MANIFEST_FILENAME).is_file()
    assert len(list(output.glob("*.safetensors.zst"))) == 2
    assert len(bucket.downloads) == 3


def test_remote_statistics_failure_occurs_after_valid_overlay_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    stats_path = logical / "feature_statistics.safetensors"
    save_file(
        {"sentinel": t.tensor([7], dtype=t.int64)},
        str(stats_path),
        metadata={"source": "synthetic"},
    )
    stats_before = stats_path.read_bytes()
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)

    with monkeypatch.context() as failure:
        failure.setattr(
            token_module,
            "save_tokenizer_checksum",
            lambda *_args, **_kwargs: (_ for _ in ()).throw(
                OSError("simulated statistics update failure")
            ),
        )
        with pytest.raises(OSError, match="statistics update failure"):
            _run(cfg, tokenizer, bucket)

    assert stats_path.read_bytes() == stats_before
    manifest_before = (output / MANIFEST_FILENAME).read_bytes()
    bodies_before = {path.name: path.read_bytes() for path in output.glob("*.safetensors.zst")}
    assert (
        discover_overlay(
            logical,
            OverlayKind.SEG_BOUNDARY,
            checksum,
            precomputed_base_hash=compute_base_shard_content_hash(glob_features(oracle)),
        )
        == output
    )

    assert _run(cfg, tokenizer, bucket) == 0
    assert (output / MANIFEST_FILENAME).read_bytes() == manifest_before
    assert {
        path.name: path.read_bytes() for path in output.glob("*.safetensors.zst")
    } == bodies_before
    with safe_open(stats_path, framework="pt") as handle:
        assert handle.metadata()["tokenizer_checksum"] == checksum


def test_remote_final_fence_failure_keeps_candidates_noncanonical(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    bucket.mutate_prefix_on_second_list = "raw"
    profile_path = tmp_path / "remote-profile.json"
    cfg = _cfg(
        logical,
        data_dir,
        tmp_path / "staging",
        remote=True,
        profile=True,
        profile_summary_output=str(profile_path),
    )
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)

    with pytest.raises(TokenBoundaryRemoteError, match="exact contiguous family"):
        _run(cfg, tokenizer, bucket)
    assert not output.exists()
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))
    profile = json.loads(profile_path.read_text())
    assert profile["totals"]["status"] == "failed"
    assert profile["remote_transport"]["final_fence_status"] == "failed"
    assert profile["remote_transport"]["terminal_error"]["type"] == ("TokenBoundaryRemoteError")


@pytest.mark.parametrize("mutation", ["compact_inventory", "manifest_identity"])
def test_remote_compact_commit_fence_failure_keeps_candidates_private(
    tmp_path: Path,
    mutation: str,
) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    compact_prefix = _compact_manifest_path(bucket).parent.relative_to(bucket.root).as_posix()
    if mutation == "compact_inventory":
        bucket.mutate_prefix_on_second_list = compact_prefix
        expected = "exact contiguous family"
    else:
        bucket.mutate_manifest_on_second_stat = True
        expected = "manifest identity changed"
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)

    with pytest.raises(TokenBoundaryRemoteError, match=expected):
        _run(cfg, tokenizer, bucket)
    assert not output.exists()
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))


def test_remote_missing_pending_parquet_fails_before_download(tmp_path: Path) -> None:
    _oracle, logical, data_dir, tokenizer, bucket = _setup(tmp_path)
    (data_dir / "shard_00001.parquet").unlink()
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)

    with pytest.raises(token_module.MissingParquetInputError, match="shard_00001"):
        _run(cfg, tokenizer, bucket)
    assert len(bucket.downloads) == 1
    assert bucket.downloads[0].endswith("/manifest.json")
    assert not output.exists()
    assert not any(output.parent.glob(".seg-boundary-candidates.*"))


@pytest.mark.parametrize("row_delta", [-1, 1])
def test_alignment_error_reports_parquet_and_feature_counts(
    tmp_path: Path,
    row_delta: int,
) -> None:
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    parquet = data_dir / "shard_00000.parquet"
    table = pq.read_table(parquet)
    texts = table.column("text").to_pylist()
    if row_delta < 0:
        texts = texts[:-1]
    else:
        texts.append("extra row")
    pq.write_table(pa.table({"text": texts}), parquet)
    shard = glob_features(feature_dir)[0]

    with pytest.raises(
        token_module.TokenBoundaryAlignmentError,
        match=rf"parquet_rows={len(texts)}, metadata_n_docs=2.*observed_doc_id_range=\[0, 1\]",
    ):
        token_module.precompute_shard(
            shard,
            parquet,
            tokenizer,
            tokenizer.get_bos_token_id(),
            tmp_path / "output",
            "shard_00000",
            source_bos_char_len=5,
            source_segment_size=SEGMENT_SIZE,
            load_keys=token_module._PRECOMPUTE_SHARD_LOAD_KEYS,
        )


@pytest.mark.parametrize("bad_doc_id", [-1, 2])
def test_alignment_rejects_out_of_range_doc_id_before_indexing(
    tmp_path: Path,
    bad_doc_id: int,
) -> None:
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    shard = glob_features(feature_dir)[0]
    data = load_features(shard)
    data["doc_ids"][0] = bad_doc_id
    save_features(data, shard)

    with pytest.raises(
        token_module.TokenBoundaryAlignmentError,
        match=rf"outside \[0, n_docs\).*observed_doc_id_range=\[{min(bad_doc_id, 0)}, {max(bad_doc_id, 1)}\]",
    ):
        token_module.precompute_shard(
            shard,
            data_dir / "shard_00000.parquet",
            tokenizer,
            tokenizer.get_bos_token_id(),
            tmp_path / "output",
            "shard_00000",
            source_bos_char_len=5,
            source_segment_size=SEGMENT_SIZE,
            load_keys=token_module._PRECOMPUTE_SHARD_LOAD_KEYS,
        )


def test_alignment_rejects_tokenizer_batch_length_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, data_dir, tokenizer = _make_env(tmp_path)
    shard = glob_features(feature_dir)[0]
    original_encode = tokenizer.encode

    def short_batch(text, prepend=None, num_threads=None):
        encoded = original_encode(text, prepend=prepend, num_threads=num_threads)
        return encoded[:-1] if isinstance(text, list) else encoded

    monkeypatch.setattr(tokenizer, "encode", short_batch)
    with pytest.raises(
        token_module.TokenBoundaryAlignmentError,
        match=r"tokenizer batch-length mismatch.*tokenizer_outputs=1",
    ):
        token_module.precompute_shard(
            shard,
            data_dir / "shard_00000.parquet",
            tokenizer,
            tokenizer.get_bos_token_id(),
            tmp_path / "output",
            "shard_00000",
            source_bos_char_len=5,
            source_segment_size=SEGMENT_SIZE,
            load_keys=token_module._PRECOMPUTE_SHARD_LOAD_KEYS,
        )


def test_disabled_remote_ignores_malformed_subordinate_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = OmegaConf.create(
        {
            "remote": {
                "enabled": False,
                "bucket_id": None,
                "staging_dir": 123,
                "retry_timeout_seconds": float("nan"),
            }
        }
    )
    monkeypatch.setattr(token_module, "_run_local", lambda _cfg: 17)
    assert token_module.run(cfg) == 17


def test_hydra_accepts_declared_shard_range_overrides() -> None:
    config_dir = Path(__file__).parents[2] / "configs" / "extraction"
    with initialize_config_dir(version_base=None, config_dir=str(config_dir)):
        cfg = compose(
            config_name="precompute_token_boundaries",
            overrides=["shard_start=7", "shard_end=11"],
        )
    assert cfg.shard_start == 7
    assert cfg.shard_end == 11


@pytest.mark.parametrize("protected", ["feature", "data", "output"])
def test_remote_config_rejects_staging_overlap(
    tmp_path: Path,
    protected: str,
) -> None:
    _oracle, logical, data_dir, tokenizer, _bucket = _setup(tmp_path)
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    staging = {
        "feature": logical / "scratch",
        "data": data_dir / "scratch",
        "output": output / "scratch",
    }[protected]
    cfg = _cfg(logical, data_dir, staging, remote=True)
    with pytest.raises(ValueError, match="must be disjoint"):
        token_module._resolve_remote_transport_config(
            cfg,
            feature_dir=logical,
            output=output,
            data_dir=data_dir,
        )


def test_remote_config_requires_staging_and_feature_root_containment(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, _bucket = _setup(tmp_path)
    checksum = compute_tokenizer_checksum(tokenizer)
    output = overlay_dir(logical, OverlayKind.SEG_BOUNDARY, checksum)
    cfg = _cfg(logical, data_dir, tmp_path / "staging", remote=True)
    cfg.remote.staging_dir = None
    with pytest.raises(ValueError, match="staging_dir is required"):
        token_module._resolve_remote_transport_config(
            cfg,
            feature_dir=logical,
            output=output,
            data_dir=data_dir,
        )

    cfg.remote.staging_dir = str(tmp_path / "staging")
    cfg.remote.local_features_root = str(tmp_path / "unrelated-root")
    with pytest.raises(ValueError, match="escapes remote.local_features_root"):
        token_module._resolve_remote_transport_config(
            cfg,
            feature_dir=logical,
            output=output,
            data_dir=data_dir,
        )


def test_remote_rejects_cfg_path_segment_size_mismatch_before_bucket_access(
    tmp_path: Path,
) -> None:
    _oracle, logical, data_dir, tokenizer, _bucket = _setup(tmp_path)
    cfg = _cfg(
        logical,
        data_dir,
        tmp_path / "staging",
        remote=True,
        segment_size=SEGMENT_SIZE + 1,
    )
    with patch.object(token_module, "get_tokenizer", return_value=tokenizer):
        with pytest.raises(SegmentSizeMismatch, match="before transport"):
            token_module.run(cfg)
