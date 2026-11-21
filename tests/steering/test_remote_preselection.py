"""Strict transport and receipt contracts for remote splice pre-selection."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import shutil
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from steering._common.strict_json import canonical_json
from steering.conditional.feature_index import (
    FeatureIndex,
    MaterializedFeatureShard,
    TransportFeatureIndexCatalog,
)
from steering.feature_io import RegularFileIdentity, TensorKey, save_features
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.gradient_overlay import provenance_dir
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.percentile_remote import run_remote_percentiles
from steering.remote_feature_inventory import resolve_remote_feature_prefix
from steering.remote_preselection import (
    REMOTE_PRESELECTION_RECEIPTS_DIRNAME,
    RemotePreselectionConfig,
    RemotePreselectionError,
    build_remote_preselection_receipt,
    discover_remote_preselection_receipts,
    parse_remote_preselection_receipt,
    resolve_remote_preselection_prefixes,
    selection_integrity_from_receipts,
)
from steering.remote_training import (
    REMOTE_PACKED_INVENTORY_FILENAME,
    RemoteObjectInfo,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    build_remote_packed_inventory,
)
from steering.remote_types import NormalizedRemotePrefix
from steering.stats_io import load_percentile_checksum
from tests.steering._splice_cf_cache_fixtures import (
    SELECTION_SEG_SIZE,
    build_interleaved_env,
    run_selection_main,
    write_cluster_yaml,
)
from tests.steering.test_percentile_remote import FilesystemBucket


def _filesystem_worker_backend(_bucket_id: str) -> FilesystemBucket:
    """Construct a fake bucket inside a spawn worker, never in the parent."""
    bucket = FilesystemBucket.__new__(FilesystemBucket)
    bucket.root = Path(os.environ["REMOTE_PRESELECTION_TEST_BUCKET_ROOT"])
    bucket.operations = []
    bucket.interrupt_after_upload = None
    bucket.interrupt_first_payload = False
    bucket.mutate_source_on_second_list = False
    bucket.mutate_source_on_list_call = None
    bucket.interrupt_before_upload = None
    bucket.transient_list_failures = 0
    bucket.transient_stat_failures = 0
    bucket.source_list_calls = 0
    return bucket


def _receipt(name: str, *, byte_length: int = 12) -> dict[str, object]:
    return build_remote_preselection_receipt(
        selection_checksum="a" * 64,
        packed_inventory_aggregate="b" * 64,
        source_inventory_aggregate="c" * 64,
        packed_row_fingerprint="d" * 64,
        shard_name=name,
        expected_packed_segments=3,
        selection_body=RemoteObjectInfo(
            path=f"features/packed/sel/{'a' * 64}/{name}",
            byte_length=byte_length,
            xet_hash="e" * 64,
        ),
        content_sha256="f" * 64,
        splice_stats={
            "total_packed_segments": 3,
            "total_sidecar_splice_positions_referenced": 2,
            "total_delta_positions_selected": 1,
        },
        max_features_per_segment=7,
    )


def _resign(raw: dict[str, object]) -> None:
    signed = {key: value for key, value in raw.items() if key != "aggregate_sha256"}
    raw["aggregate_sha256"] = hashlib.sha256(canonical_json(signed).encode("utf-8")).hexdigest()


def test_receipt_round_trip_and_order_independent_integrity() -> None:
    first = _receipt("shard_00000.safetensors.zst")
    second = _receipt("shard_00001.safetensors.zst", byte_length=14)

    assert parse_remote_preselection_receipt(first, surface="test") == first
    forward = selection_integrity_from_receipts(
        {
            "shard_00000.safetensors.zst": first,
            "shard_00001.safetensors.zst": second,
        }
    )
    reverse = selection_integrity_from_receipts(
        {
            "shard_00001.safetensors.zst": second,
            "shard_00000.safetensors.zst": first,
        }
    )
    assert forward == reverse
    assert [entry["name"] for entry in cast(list[dict[str, object]], forward["shards"])] == [
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
    ]


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.__setitem__("aggregate_sha256", "0" * 64),
        lambda raw: raw["splice_stats"].__setitem__("total_delta_positions_selected", -1),
        lambda raw: raw["selection_body"].__setitem__("path", "../escape"),
        lambda raw: raw.__setitem__("expected_packed_segments", True),
    ],
)
def test_receipt_rejects_every_signed_mutation(mutate) -> None:
    raw = copy.deepcopy(_receipt("shard_00000.safetensors.zst"))
    mutate(raw)
    with pytest.raises(RemotePreselectionError):
        parse_remote_preselection_receipt(raw, surface="mutated receipt")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.__setitem__("schema_version", True),
        lambda raw: raw.__setitem__("expected_packed_segments", 4),
        lambda raw: raw.__setitem__("shard_name", "bad\\shard_00000.safetensors.zst"),
    ],
)
def test_receipt_rejects_resigned_noncanonical_fields(mutate) -> None:
    raw = copy.deepcopy(_receipt("shard_00000.safetensors.zst"))
    mutate(raw)
    _resign(raw)
    with pytest.raises(RemotePreselectionError):
        parse_remote_preselection_receipt(raw, surface="resigned malformed receipt")


def test_remote_config_is_exact_strict_and_disjoint(tmp_path: Path) -> None:
    staging = tmp_path / "stage"
    staging.mkdir()
    parsed = RemotePreselectionConfig.parse(
        {"staging_dir": str(staging), "max_output_shard_bytes": 10}
    )
    assert parsed is not None
    assert parsed.max_output_shard_bytes == 10
    with pytest.raises(RemotePreselectionError, match="exactly"):
        RemotePreselectionConfig.parse(
            {
                "staging_dir": str(staging),
                "max_output_shard_bytes": 10,
                "bucket_id": "forbidden",
            }
        )
    with pytest.raises(RemotePreselectionError, match="strict integer"):
        RemotePreselectionConfig.parse(
            {"staging_dir": str(staging), "max_output_shard_bytes": True}
        )
    with pytest.raises(RemotePreselectionError, match="absolute path"):
        RemotePreselectionConfig.parse(
            {"staging_dir": "relative-stage", "max_output_shard_bytes": 10}
        )
    with pytest.raises(RemotePreselectionError, match="disjoint"):
        parsed.assert_disjoint(staging / "child")


def test_remote_namespace_rejects_unknown_nested_and_receipt_without_body(
    tmp_path: Path,
) -> None:
    bucket = FilesystemBucket(tmp_path / "bucket")
    staging = tmp_path / "stage"
    staging.mkdir()
    name = "shard_00000.safetensors.zst"
    common = {
        "output_prefix": "selection/run",
        "expected_names": [name],
        "selection_checksum": "a" * 64,
        "packed_inventory_aggregate": "b" * 64,
        "source_inventory_aggregate": "c" * 64,
        "packed_row_fingerprint": "d" * 64,
        "staging_dir": staging,
        "timeout_seconds": 1.0,
        "retry_interval_seconds": 0.001,
    }
    receipts, manifest = discover_remote_preselection_receipts(bucket, **common)
    assert receipts == {}
    assert manifest is None

    unknown = bucket._path("selection/run/unknown/nested.json")
    unknown.parent.mkdir(parents=True)
    unknown.write_text("{}", encoding="utf-8")
    with pytest.raises(RemotePreselectionError, match="unknown object"):
        discover_remote_preselection_receipts(bucket, **common)
    unknown.unlink()

    receipt = bucket._path(f"selection/run/{REMOTE_PRESELECTION_RECEIPTS_DIRNAME}/{name}.json")
    receipt.parent.mkdir(parents=True)
    receipt.write_text("{}", encoding="utf-8")
    with pytest.raises(RemotePreselectionError, match="without its selection body"):
        discover_remote_preselection_receipts(bucket, **common)


class _CopyMaterializer:
    def __init__(self, raw_source: Path, pct_source: Path, raw: Path, pct: Path):
        self.raw_source = raw_source
        self.pct_source = pct_source
        self.raw = raw
        self.pct = pct
        self.loads = 0

    @contextmanager
    def materialize(self, stem: str) -> Iterator[MaterializedFeatureShard]:
        assert stem == "shard_00000"
        self.loads += 1
        self.raw.parent.mkdir(parents=True, exist_ok=True)
        self.pct.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(self.raw_source, self.raw)
        shutil.copyfile(self.pct_source, self.pct)
        raw_identity = RegularFileIdentity.capture(self.raw)
        pct_identity = RegularFileIdentity.capture(self.pct)
        try:
            yield MaterializedFeatureShard(
                raw_path=self.raw,
                raw_identity=raw_identity,
                percentile_path=self.pct,
                percentile_identity=pct_identity,
            )
        finally:
            assert RegularFileIdentity.capture(self.raw) == raw_identity
            assert RegularFileIdentity.capture(self.pct) == pct_identity
            self.pct.unlink()
            self.raw.unlink()


def test_transport_feature_index_matches_local_and_cleans_bodies(
    tmp_path: Path,
) -> None:
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    raw = save_features(
        {
            "offsets": t.tensor([0, 2, 3], dtype=t.int64),
            "feature_ids": t.tensor([3, 8, 13], dtype=t.int32),
            "feature_mags": t.tensor([0.25, 0.5, 0.75], dtype=t.float32),
            "doc_ids": t.tensor([0, 0], dtype=t.int32),
            "segment_indices": t.tensor([0, 1], dtype=t.int32),
            "segment_positions": t.tensor([0, 128], dtype=t.int32),
            "doc_token_counts": t.tensor([256], dtype=t.int32),
            "segment_char_starts": t.tensor([0, 768], dtype=t.int32),
            "segment_char_ends": t.tensor([768, 1536], dtype=t.int32),
            "n_docs": 1,
            "n_segments": 2,
            "layer": 12,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_mode": "fixed_tokens",
            "segment_size": 128,
        },
        source_dir / "shard_00000_features",
    )
    checksum = "9" * 64
    pct_dir = overlay_dir(source_dir, OverlayKind.PERCENTILE, checksum)
    write_manifest(
        pct_dir,
        checksum,
        OverlayKind.PERCENTILE,
        base_shard_paths=[raw],
    )
    pct = save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES: t.tensor([10, 20, 30], dtype=t.uint8)},
        overlay_shard_path(pct_dir, "shard_00000"),
    )
    local = FeatureIndex(str(source_dir), percentile_checksum=checksum)

    stage_raw = tmp_path / "stage" / "raw" / raw.name
    stage_pct = tmp_path / "stage" / "pct" / pct.name
    catalog = TransportFeatureIndexCatalog(
        feature_dir=str(stage_raw.parent.absolute()),
        percentile_overlay_dir=str(stage_pct.parent.absolute()),
        entries=(("shard_00000", str(stage_raw.absolute()), str(stage_pct.absolute())),),
        segment_size=128,
        percentile_checksum=checksum,
        validated_content_fingerprint="8" * 64,
    )
    materializer = _CopyMaterializer(raw, pct, stage_raw, stage_pct)
    remote = FeatureIndex.from_transport_catalog(
        catalog,
        materializer=materializer,
        max_resident_shards=1,
    )

    local_segments = local.lookup("shard_00000", 0)
    remote_segments = remote.lookup("shard_00000", 0)

    assert len(local_segments) == len(remote_segments)
    for expected, actual in zip(local_segments, remote_segments, strict=True):
        assert t.equal(actual.feature_ids, expected.feature_ids)
        assert t.equal(actual.feature_mags, expected.feature_mags)
        assert actual.feature_percentiles is not None
        assert expected.feature_percentiles is not None
        assert t.equal(actual.feature_percentiles, expected.feature_percentiles)
    assert materializer.loads == 1
    assert not stage_raw.exists()
    assert not stage_pct.exists()
    remote.lookup("shard_00000", 0)
    assert materializer.loads == 1


def _remote_selection_cfg(
    env: dict[str, object],
    *,
    feature_root: Path,
    clusters_path: str,
    staging_dir: Path,
    num_workers: int,
    shared_store: bool,
) -> DictConfig:
    return OmegaConf.create(
        {
            "packed_row_dir": str(env["packed_row_dir"]),
            "feature_dir": str(env["feature_dir"]),
            "feature_root": str(feature_root),
            "segment_size": SELECTION_SEG_SIZE,
            "feature_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 1,
                "max_k": 49,
                "mode": "top_k_percentile",
                "delta_weights": "percentile_delta",
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "min_k": 0,
                "max_k": 49,
            },
            "feature_delta": {
                "mode": "splice_counterfactual",
                "neg_delta_ratio": 1.0,
            },
            "feature_mask_freq_min": None,
            "feature_mask_freq_max": None,
            "feature_clusters": {"path": clusters_path},
            "selection_resume": True,
            "selection_splice_batch_size": 1,
            "selection_source_cache_shards": 1,
            "selection_sidecar_cache_shards": 1,
            "selection_num_workers": num_workers,
            "selection_shared_store": shared_store,
            "selection_shared_store_capacity": 4 if shared_store else 0,
            "selection_packed_spool_dir": None,
            "use_precomputed_batches": True,
            "num_buckets": 2,
            "seed": 42,
            "sae": {
                "release": "gemmascope_2",
                "model": "gemma-3-4b-pt",
                "activation_type": "residual",
                "layer": 8,
                "width": "65k",
                "l0": "medium",
            },
            "pooling": "max_pool",
            "dataset_phase": None,
            "remote_preselection": {
                "staging_dir": str(staging_dir),
                "max_output_shard_bytes": 1_000_000,
            },
            "remote_training": {
                "bucket_id": "owner/test-bucket",
                "body_prefetch": {"lookahead_shards_per_rank": 1},
                "selection_prefix": "features",
                "checkpoint_prefix": "checkpoints",
                "selection_wait_timeout_seconds": 2.0,
                "checkpoint_retry_timeout_seconds": 2.0,
                "retry_interval_seconds": 0.001,
                "resume_latest": False,
                "train_packed": {},
            },
        }
    )


def _copy_to_bucket(bucket: FilesystemBucket, source: Path, remote_path: str) -> None:
    destination = bucket._path(remote_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


@pytest.mark.parametrize(
    ("num_workers", "shared_store", "legacy_percentiles"),
    [(1, False, False), (2, True, False), (1, False, True)],
)
def test_remote_producer_matches_local_bytes_and_current_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    num_workers: int,
    shared_store: bool,
    legacy_percentiles: bool,
) -> None:
    env = build_interleaved_env(tmp_path / "corpus", n_shards=2)
    clusters_path = write_cluster_yaml(tmp_path)
    packed_dir = Path(env["packed_row_dir"])
    feature_dir = Path(str(env["feature_dir"]))

    staging = tmp_path / "staging"
    percentile_staging = tmp_path / "percentile-staging"
    staging.mkdir()
    percentile_staging.mkdir()
    bucket = FilesystemBucket(tmp_path / "bucket")
    shared_store_root = tmp_path / "shared-store"
    shared_store_root.mkdir()
    feature_root = tmp_path
    remote_root = NormalizedRemotePrefix.parse("features")
    source_prefix = str(
        resolve_remote_feature_prefix(
            feature_dir,
            local_features_root=feature_root,
            remote_features_root=remote_root,
        )
    )
    for source in sorted(feature_dir.glob("shard_*_features.safetensors.zst")):
        _copy_to_bucket(bucket, source, f"{source_prefix}/{source.name}")
    statistics = next(feature_dir.glob("feature_statistics.safetensors*"))
    _copy_to_bucket(bucket, statistics, f"{source_prefix}/feature_statistics.safetensors")
    for sidecar in sorted((feature_dir / PROVENANCE_DIRNAME).glob("*.json")):
        _copy_to_bucket(
            bucket,
            sidecar,
            f"{source_prefix}/{PROVENANCE_DIRNAME}/{sidecar.name}",
        )

    percentile_cfg = OmegaConf.create(
        {
            "feature_dir": str(feature_dir),
            "segment_size": SELECTION_SEG_SIZE,
            "device": "cpu",
            "resume": False,
            "dry_run": False,
            "force": False,
            "zstd_level": 12,
            "profile": False,
            "profile_summary_output": None,
            "remote": {
                "enabled": True,
                "bucket_id": "owner/test-bucket",
                "local_features_root": str(feature_root),
                "remote_features_root": str(remote_root),
                "staging_dir": str(percentile_staging),
                "expected_shard_count": 2,
                "max_output_shard_bytes": 1_000_000,
                "max_decompressed_shard_bytes": 1_000_000,
                "retry_timeout_seconds": 2.0,
                "retry_interval_seconds": 0.001,
            },
        }
    )
    assert run_remote_percentiles(percentile_cfg, feature_dir, backend=bucket) == 0
    checksum = load_percentile_checksum(statistics)
    assert checksum is not None
    remote_pct_prefix = f"{source_prefix}/pct/{checksum[:16]}"
    completed_statistics = bucket._path(f"{remote_pct_prefix}/feature_statistics.safetensors")
    if legacy_percentiles:
        _copy_to_bucket(
            bucket,
            statistics,
            f"{source_prefix}/feature_statistics.safetensors",
        )
        manifest_path = bucket._path(f"{remote_pct_prefix}/manifest.json")
        modern_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        legacy_manifest = {
            "checksum": modern_manifest["checksum"],
            "checksum_prefix": modern_manifest["checksum_prefix"],
            "kind": modern_manifest["kind"],
            "created_at": "2026-07-06T15:20:05Z",
            "metadata": {
                "n_sae_features": modern_manifest["metadata"]["n_sae_features"],
                "stats_file": "feature_statistics.safetensors",
            },
            "base_shard_content_hash": modern_manifest["base_shard_content_hash"],
            "schema_version": 2,
        }
        manifest_path.write_text(json.dumps(legacy_manifest, indent=2) + "\n", encoding="utf-8")
        completed_statistics.unlink()
        shutil.rmtree(bucket._path(f"{remote_pct_prefix}/_receipts"))
    else:
        shutil.copyfile(completed_statistics, statistics)

    # Repack from the actual remotely produced percentile bytes. The local
    # oracle and streaming path then consume one identical source contract.
    local_pct_dir = feature_dir / "pct" / checksum[:16]
    local_pct_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "shard_00000.safetensors.zst",
        "shard_00001.safetensors.zst",
        "manifest.json",
    ):
        shutil.copyfile(bucket._path(f"{remote_pct_prefix}/{name}"), local_pct_dir / name)
    shutil.rmtree(packed_dir)
    packed_dir = local_pct_dir / "tok" / "fixture" / "c16" / "sall"
    env["packed_row_dir"] = packed_dir
    packed_prefix = str(
        resolve_remote_feature_prefix(
            packed_dir,
            local_features_root=feature_root,
            remote_features_root=remote_root,
        )
    )
    feature_index = FeatureIndex(
        str(feature_dir),
        percentile_checksum=checksum,
        tokenizer_checksum=str(env["tok_checksum"]),
    )
    total_rows, max_n_seg, max_seg_size = precompute_packed_rows(
        parquet_paths=[str(path) for path in sorted(Path(str(env["data_dir"])).glob("*.parquet"))],
        tokenizer=env["tokenizer"],
        feature_index=feature_index,
        context_len=16,
        output_dir=packed_dir,
        buffer_size=3,
        rows_per_shard=1,
        compress_workers=1,
    )
    assert total_rows > 0
    write_manifest(
        packed_dir,
        "packed-interleaved",
        OverlayKind.PACKED_ROW,
        metadata={
            "base_shard_stems": feature_index.shard_stems,
            "provenance_shards": len(glob_overlay_shards(provenance_dir(packed_dir))),
            "max_n_seg": max_n_seg,
            "max_seg_size": max_seg_size,
            "context_len": 16,
            "pct_checksum": checksum,
            "tok_checksum": str(env["tok_checksum"]),
        },
    )
    local = run_selection_main(
        env,
        cache_shards=1,
        sidecar_shards=1,
        num_workers=num_workers,
        shared_store=shared_store,
        shared_store_capacity=4,
        shm_dir=str(shared_store_root),
        splice_batch_size=1,
        clusters_path=clusters_path,
        monkeypatch=monkeypatch,
    )
    expected_bodies = {
        name: payload for name, payload in local.items() if name.endswith(".safetensors.zst")
    }
    local_selection_checksum = next((packed_dir / "sel").iterdir()).name
    packed_inventory = build_remote_packed_inventory(packed_dir)
    shutil.rmtree(packed_dir / "sel")

    for body in sorted(packed_dir.glob("*.safetensors.zst")):
        _copy_to_bucket(bucket, body, f"{packed_prefix}/{body.name}")
    inventory_path = bucket._path(f"{packed_prefix}/{REMOTE_PACKED_INVENTORY_FILENAME}")
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(canonical_json(packed_inventory), encoding="utf-8")

    import steering.remote_preselection as remote_module
    from scripts.preprocess import precompute_selected_features as psf

    monkeypatch.setattr(psf, "HfBucketBackend", lambda _bucket_id: bucket)
    monkeypatch.setattr(remote_module, "HfBucketBackend", lambda _bucket_id: bucket)
    if num_workers > 1:
        original_remote_runner = psf._run_remote_splice_shards

        def run_with_worker_bucket(**kwargs) -> None:
            original_remote_runner(
                **kwargs,
                remote_worker_backend_factory=_filesystem_worker_backend,
            )

        monkeypatch.setattr(psf, "_run_remote_splice_shards", run_with_worker_bucket)
        monkeypatch.setenv("REMOTE_PRESELECTION_TEST_BUCKET_ROOT", str(tmp_path / "bucket"))
        monkeypatch.setenv("NANOCHAT_SHM_DIR", str(shared_store_root))
    cfg = _remote_selection_cfg(
        env,
        feature_root=feature_root,
        clusters_path=clusters_path,
        staging_dir=staging,
        num_workers=num_workers,
        shared_store=shared_store,
    )
    remote = RemoteTrainingConfig.parse(cfg.remote_training)
    assert remote is not None
    prefixes = resolve_remote_preselection_prefixes(
        remote=remote,
        feature_root=feature_root,
        feature_dir=feature_dir,
        packed_row_dir=packed_dir,
        selection_checksum=local_selection_checksum,
    )
    first_name = sorted(expected_bodies)[0]
    first_body_path = f"{prefixes.selection_output}/{first_name}"
    bucket.interrupt_after_upload = first_body_path
    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        psf.main.__wrapped__(cfg)
    assert bucket._path(first_body_path).is_file()
    assert not bucket._path(
        f"{prefixes.selection_output}/{REMOTE_PRESELECTION_RECEIPTS_DIRNAME}/{first_name}.json"
    ).exists()
    assert not bucket._path(f"{prefixes.selection_output}/manifest.json").exists()

    # Restart recomputes the body-only orphan, proves the existing bytes are
    # identical, and then completes receipt and manifest publication.
    psf.main.__wrapped__(cfg)

    local_selection_dirs = list((packed_dir / "sel").iterdir())
    assert len(local_selection_dirs) == 1
    selection_dir = local_selection_dirs[0]
    selection_checksum = selection_dir.name
    assert selection_checksum == local_selection_checksum
    actual_bodies = {
        name: bucket._path(f"{prefixes.selection_output}/{name}").read_bytes()
        for name in expected_bodies
    }
    assert actual_bodies == expected_bodies

    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=remote,
        local_selection_dir=selection_dir,
        feature_root=feature_root,
        expected_selection_checksum=selection_checksum,
        packed_row_fingerprint=str(packed_inventory["packed_row_fingerprint"]),
        packed_stems={name.removesuffix(".safetensors.zst") for name in expected_bodies},
        label="remote pre-selection integration",
    )
    assert set(materializer.inventory) == set(expected_bodies)
    manifest = json.loads((selection_dir / "manifest.json").read_text())
    assert "created_at" not in manifest
    local_manifest = json.loads(local["manifest.json"])
    local_manifest.pop("created_at", None)
    assert manifest == local_manifest
    canonical_uploads = [
        path
        for operation, path in bucket.operations
        if operation == "upload"
        and path.startswith(f"{prefixes.selection_output}/")
        and "_verified_upload_claims" not in path
    ]
    receipt_path = (
        f"{prefixes.selection_output}/{REMOTE_PRESELECTION_RECEIPTS_DIRNAME}/{first_name}.json"
    )
    assert canonical_uploads[-1] == f"{prefixes.selection_output}/manifest.json"
    assert canonical_uploads.index(first_body_path) < canonical_uploads.index(receipt_path)
    assert not any(staging.iterdir())

    bucket.operations.clear()
    psf.main.__wrapped__(cfg)
    assert not any(operation == "upload" for operation, _path in bucket.operations)
