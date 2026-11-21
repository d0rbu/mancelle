"""Strict receipts and namespace states for packed-only ordinary selection."""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

from scripts.preprocess import precompute_selected_features as psf
from steering._common.strict_json import canonical_json
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.remote_feature_inventory import resolve_remote_feature_prefix
from steering.remote_preselection import (
    REMOTE_PRESELECTION_RECEIPTS_DIRNAME,
    RemotePreselectionError,
    build_remote_preselection_receipt,
    build_remote_unspliced_preselection_receipt,
    discover_remote_unspliced_preselection_receipts,
    parse_remote_preselection_receipt,
    parse_remote_unspliced_preselection_receipt,
    selection_integrity_from_unspliced_receipts,
)
from steering.remote_training import (
    REMOTE_PACKED_INVENTORY_FILENAME,
    RemoteObjectInfo,
    RemoteSelectionMaterializer,
    RemoteTrainingConfig,
    build_remote_packed_inventory,
)
from steering.remote_types import NormalizedRemotePrefix
from steering.stats_io import find_statistics_path, load_statistics
from tests.steering._splice_cf_cache_fixtures import (
    SELECTION_SEG_SIZE,
    build_interleaved_env,
    write_cluster_yaml,
)
from tests.steering.test_percentile_remote import FilesystemBucket

_SELECTION_CHECKSUM = "a" * 64
_PACKED_AGGREGATE = "b" * 64
_PACKED_FINGERPRINT = "c" * 64
_OUTPUT_PREFIX = f"packed/sel/{_SELECTION_CHECKSUM}"


def _receipt(
    name: str = "shard_00000.safetensors.zst",
    *,
    body: RemoteObjectInfo | None = None,
    segments: int = 3,
    width: int = 7,
    rng_sha256: str = "f" * 64,
) -> dict[str, object]:
    return build_remote_unspliced_preselection_receipt(
        selection_checksum=_SELECTION_CHECKSUM,
        packed_inventory_aggregate=_PACKED_AGGREGATE,
        packed_row_fingerprint=_PACKED_FINGERPRINT,
        shard_name=name,
        expected_packed_segments=segments,
        selection_body=(
            body
            if body is not None
            else RemoteObjectInfo(
                path=f"{_OUTPUT_PREFIX}/{name}",
                byte_length=12,
                xet_hash="d" * 64,
            )
        ),
        content_sha256="e" * 64,
        max_features_per_segment=width,
        post_selection_rng_sha256=rng_sha256,
    )


def _resign(raw: dict[str, object]) -> None:
    signed = {key: value for key, value in raw.items() if key != "aggregate_sha256"}
    raw["aggregate_sha256"] = hashlib.sha256(
        canonical_json(signed).encode("utf-8")
    ).hexdigest()


def test_ordinary_receipt_round_trip_is_canonical_and_mode_specific() -> None:
    first = _receipt()
    second = _receipt("shard_00001.safetensors.zst", width=0)

    assert parse_remote_unspliced_preselection_receipt(first, surface="test") == first
    assert canonical_json(first).encode() == canonical_json(
        parse_remote_unspliced_preselection_receipt(first, surface="repeat")
    ).encode()
    integrity = selection_integrity_from_unspliced_receipts(
        {
            "shard_00001.safetensors.zst": second,
            "shard_00000.safetensors.zst": first,
        }
    )
    assert [
        entry["name"] for entry in cast(list[dict[str, object]], integrity["shards"])
    ] == ["shard_00000.safetensors.zst", "shard_00001.safetensors.zst"]

    with pytest.raises(RemotePreselectionError, match="unexpected or missing keys"):
        parse_remote_preselection_receipt(first, surface="ordinary as splice")
    splice = build_remote_preselection_receipt(
        selection_checksum=_SELECTION_CHECKSUM,
        packed_inventory_aggregate=_PACKED_AGGREGATE,
        source_inventory_aggregate="9" * 64,
        packed_row_fingerprint=_PACKED_FINGERPRINT,
        shard_name="shard_00000.safetensors.zst",
        expected_packed_segments=3,
        selection_body=RemoteObjectInfo(
            path=f"{_OUTPUT_PREFIX}/shard_00000.safetensors.zst",
            byte_length=12,
            xet_hash="d" * 64,
        ),
        content_sha256="e" * 64,
        splice_stats={
            "total_packed_segments": 3,
            "total_sidecar_splice_positions_referenced": 0,
            "total_delta_positions_selected": 0,
        },
        max_features_per_segment=7,
    )
    with pytest.raises(RemotePreselectionError, match="unexpected or missing keys"):
        parse_remote_unspliced_preselection_receipt(splice, surface="splice as ordinary")


@pytest.mark.parametrize(
    "mutate",
    [
        lambda raw: raw.pop("post_selection_rng_sha256"),
        lambda raw: raw.__setitem__("future_extension", 1),
        lambda raw: raw["selection_body"].pop("xet_hash"),
        lambda raw: raw["selection_body"].__setitem__("future_extension", 1),
        lambda raw: raw.__setitem__("expected_packed_segments", True),
        lambda raw: raw.__setitem__("max_features_per_segment", True),
        lambda raw: raw["selection_body"].__setitem__("byte_length", True),
        lambda raw: raw.__setitem__("expected_packed_segments", -1),
        lambda raw: raw.__setitem__("max_features_per_segment", -1),
        lambda raw: raw["selection_body"].__setitem__("byte_length", 0),
        lambda raw: raw.__setitem__("selection_checksum", "A" * 64),
        lambda raw: raw.__setitem__("packed_inventory_aggregate", "b" * 63),
        lambda raw: raw.__setitem__("packed_row_fingerprint", "z" * 64),
        lambda raw: raw.__setitem__("post_selection_rng_sha256", "F" * 64),
        lambda raw: raw.__setitem__("shard_name", "nested/shard_00000.safetensors.zst"),
        lambda raw: raw.__setitem__("shard_name", "shard_00000.safetensors"),
        lambda raw: raw["selection_body"].__setitem__("path", "/absolute/body"),
        lambda raw: raw["selection_body"].__setitem__("path", "packed//body"),
        lambda raw: raw["selection_body"].__setitem__("path", "packed/../body"),
    ],
)
def test_ordinary_receipt_rejects_resigned_noncanonical_values(mutate) -> None:
    raw = copy.deepcopy(_receipt())
    mutate(raw)
    _resign(raw)
    with pytest.raises(RemotePreselectionError):
        parse_remote_unspliced_preselection_receipt(raw, surface="mutated ordinary receipt")


def _publish_receipt(
    bucket: FilesystemBucket,
    *,
    name: str,
    with_receipt: bool,
) -> dict[str, object] | None:
    body_path = bucket._path(f"{_OUTPUT_PREFIX}/{name}")
    body_path.parent.mkdir(parents=True, exist_ok=True)
    body_path.write_bytes(f"body:{name}".encode())
    if not with_receipt:
        return None
    body = bucket.stat(f"{_OUTPUT_PREFIX}/{name}")
    assert body is not None
    receipt = _receipt(name, body=body)
    receipt_path = bucket._path(
        f"{_OUTPUT_PREFIX}/{REMOTE_PRESELECTION_RECEIPTS_DIRNAME}/{name}.json"
    )
    receipt_path.parent.mkdir(parents=True, exist_ok=True)
    receipt_path.write_text(canonical_json(receipt), encoding="utf-8")
    return receipt


def _discover(
    bucket: FilesystemBucket, staging: Path, names: list[str]
) -> tuple[dict[str, dict[str, object]], RemoteObjectInfo | None, tuple[str, ...]]:
    return discover_remote_unspliced_preselection_receipts(
        bucket,
        output_prefix=_OUTPUT_PREFIX,
        expected_names=names,
        selection_checksum=_SELECTION_CHECKSUM,
        packed_inventory_aggregate=_PACKED_AGGREGATE,
        packed_row_fingerprint=_PACKED_FINGERPRINT,
        staging_dir=staging,
        timeout_seconds=1.0,
        retry_interval_seconds=0.001,
    )


def test_ordinary_namespace_accepts_only_prefix_receipts_and_first_orphan(
    tmp_path: Path,
) -> None:
    names = [f"shard_{index:05d}.safetensors.zst" for index in range(3)]
    staging = tmp_path / "staging"
    staging.mkdir()
    bucket = FilesystemBucket(tmp_path / "bucket")

    receipts, manifest, body_only = _discover(bucket, staging, names)
    assert receipts == {}
    assert manifest is None
    assert body_only == ()

    first = _publish_receipt(bucket, name=names[0], with_receipt=True)
    assert first is not None
    _publish_receipt(bucket, name=names[1], with_receipt=False)
    receipts, manifest, body_only = _discover(bucket, staging, names)
    assert receipts == {names[0]: first}
    assert manifest is None
    assert body_only == (names[1],)

    _publish_receipt(bucket, name=names[2], with_receipt=False)
    with pytest.raises(RemotePreselectionError, match="first missing shard"):
        _discover(bucket, staging, names)


def test_ordinary_namespace_rejects_receipt_gap_and_signed_path_drift(
    tmp_path: Path,
) -> None:
    names = [f"shard_{index:05d}.safetensors.zst" for index in range(2)]
    staging = tmp_path / "staging"
    staging.mkdir()
    bucket = FilesystemBucket(tmp_path / "bucket")
    _publish_receipt(bucket, name=names[1], with_receipt=True)
    with pytest.raises(RemotePreselectionError, match="leading prefix"):
        _discover(bucket, staging, names)

    bucket = FilesystemBucket(tmp_path / "bucket-path")
    receipt = _publish_receipt(bucket, name=names[0], with_receipt=True)
    assert receipt is not None
    receipt_body = cast(dict[str, object], receipt["selection_body"])
    receipt_body["path"] = f"elsewhere/{names[0]}"
    _resign(receipt)
    receipt_path = bucket._path(
        f"{_OUTPUT_PREFIX}/{REMOTE_PRESELECTION_RECEIPTS_DIRNAME}/{names[0]}.json"
    )
    receipt_path.write_text(canonical_json(receipt), encoding="utf-8")
    with pytest.raises(RemotePreselectionError, match="noncanonical body path"):
        _discover(bucket, staging, names)


def _ordinary_cfg(
    env: dict[str, object],
    *,
    feature_root: Path | None,
    clusters_path: str,
    staging_dir: Path | None,
) -> DictConfig:
    cfg: dict[str, object] = {
        "packed_row_dir": str(env["packed_row_dir"]),
        "feature_dir": str(env["feature_dir"]),
        "feature_root": str(feature_root) if feature_root is not None else None,
        "segment_size": SELECTION_SEG_SIZE,
        "feature_selection": {
            "distribution": "geometric",
            "mean": 12.0,
            "std": 12.0,
            "min_k": 1,
            "max_k": 49,
            "mode": "top_k_percentile",
            "delta_weights": None,
        },
        "negative_selection": {
            "distribution": "geometric",
            "mean": 12.0,
            "min_k": 0,
            "max_k": 49,
        },
        "feature_delta": None,
        "feature_mask_freq_min": None,
        "feature_mask_freq_max": None,
        "feature_clusters": {"path": clusters_path},
        "selection_resume": True,
        "selection_splice_batch_size": 1,
        "selection_source_cache_shards": 3,
        "selection_sidecar_cache_shards": 4,
        "selection_num_workers": 1,
        "selection_shared_store": True,
        "selection_shared_store_capacity": 9,
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
        "remote_preselection": None,
        "remote_training": None,
    }
    if staging_dir is not None:
        cfg["remote_preselection"] = {
            "staging_dir": str(staging_dir),
            "max_output_shard_bytes": 1_000_000,
        }
        cfg["remote_training"] = {
            "bucket_id": "owner/test-bucket",
            "body_prefetch": {"lookahead_shards_per_rank": 1},
            "selection_prefix": "remote",
            "checkpoint_prefix": "checkpoints",
            "selection_wait_timeout_seconds": 2.0,
            "checkpoint_retry_timeout_seconds": 2.0,
            "retry_interval_seconds": 0.001,
            "resume_latest": False,
            "train_packed": {},
        }
    return OmegaConf.create(cfg)


def test_remote_ordinary_matches_local_rng_resume_manifest_and_consumer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    corpus_root = tmp_path / "corpus"
    env = build_interleaved_env(corpus_root, n_shards=4)
    packed_dir = Path(cast(Path, env["packed_row_dir"]))
    feature_dir = Path(cast(str, env["feature_dir"]))
    clusters_path = write_cluster_yaml(tmp_path)
    statistics_path = find_statistics_path(feature_dir)
    assert statistics_path is not None
    statistics = load_statistics(statistics_path)
    real_pct_checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    packed_manifest_path = packed_dir / "manifest.json"
    packed_manifest = json.loads(packed_manifest_path.read_text(encoding="utf-8"))
    packed_manifest["metadata"]["pct_checksum"] = real_pct_checksum
    packed_manifest_path.write_text(
        json.dumps(packed_manifest, indent=2) + "\n", encoding="utf-8"
    )

    built_pipelines = []
    original_builder = psf.build_feature_pipeline_from_cfg_block

    def _capture_pipeline(*args, **kwargs):
        pipeline = original_builder(*args, **kwargs)
        built_pipelines.append(pipeline)
        return pipeline

    monkeypatch.setattr(psf, "build_feature_pipeline_from_cfg_block", _capture_pipeline)
    local_cfg = _ordinary_cfg(
        env,
        feature_root=None,
        clusters_path=clusters_path,
        staging_dir=None,
    )
    psf.main.__wrapped__(local_cfg)
    selection_dirs = list((packed_dir / "sel").iterdir())
    assert len(selection_dirs) == 1
    local_selection_dir = selection_dirs[0]
    selection_checksum = local_selection_dir.name
    local_bodies = {
        path.name: path.read_bytes()
        for path in sorted(local_selection_dir.glob("*.safetensors.zst"))
    }
    assert len(local_bodies) >= 2
    local_manifest = json.loads(
        (local_selection_dir / "manifest.json").read_text(encoding="utf-8")
    )
    local_manifest.pop("created_at", None)
    local_rng_state = built_pipelines[-1].selection_config.rng.get_state().clone()
    local_next_draw = t.randint(
        0, 2**31, (1,), generator=built_pipelines[-1].selection_config.rng
    )

    packed_inventory = build_remote_packed_inventory(packed_dir)
    shutil.rmtree(packed_dir / "sel")
    staging = tmp_path / "staging"
    staging.mkdir()
    bucket = FilesystemBucket(tmp_path / "bucket")
    remote_prefix = NormalizedRemotePrefix.parse("remote")
    packed_prefix = str(
        resolve_remote_feature_prefix(
            packed_dir,
            local_features_root=corpus_root,
            remote_features_root=remote_prefix,
        )
    )
    for body_path in sorted(packed_dir.glob("*.safetensors.zst")):
        destination = bucket._path(f"{packed_prefix}/{body_path.name}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(body_path, destination)
    inventory_path = bucket._path(
        f"{packed_prefix}/{REMOTE_PACKED_INVENTORY_FILENAME}"
    )
    inventory_path.parent.mkdir(parents=True, exist_ok=True)
    inventory_path.write_text(canonical_json(packed_inventory), encoding="utf-8")

    monkeypatch.setattr(psf, "HfBucketBackend", lambda _bucket_id: bucket)
    parsed_components = SimpleNamespace(segment_size=SELECTION_SEG_SIZE)
    monkeypatch.setattr(psf, "parse_feature_dir", lambda _path: parsed_components)
    remote_cfg = _ordinary_cfg(
        env,
        feature_root=corpus_root,
        clusters_path=clusters_path,
        staging_dir=staging,
    )
    remote = RemoteTrainingConfig.parse(remote_cfg.remote_training)
    assert remote is not None
    prefixes = psf.resolve_remote_preselection_prefixes(
        remote=remote,
        feature_root=corpus_root,
        feature_dir=feature_dir,
        packed_row_dir=packed_dir,
        selection_checksum=selection_checksum,
    )
    first_name = sorted(local_bodies)[0]
    first_receipt_path = (
        f"{prefixes.selection_output}/{REMOTE_PRESELECTION_RECEIPTS_DIRNAME}/"
        f"{first_name}.json"
    )
    bucket.interrupt_after_upload = first_receipt_path
    with pytest.raises(KeyboardInterrupt, match="injected interruption"):
        psf.main.__wrapped__(remote_cfg)
    assert bucket._path(first_receipt_path).is_file()
    assert not bucket._path(f"{prefixes.selection_output}/manifest.json").exists()

    receipt_control_path = bucket._path(first_receipt_path)
    original_receipt_bytes = receipt_control_path.read_bytes()
    corrupted_receipt = json.loads(original_receipt_bytes)
    corrupted_receipt["post_selection_rng_sha256"] = "0" * 64
    _resign(corrupted_receipt)
    receipt_control_path.write_text(
        canonical_json(corrupted_receipt), encoding="utf-8"
    )
    with pytest.raises(RemotePreselectionError, match="prefix replay drift"):
        psf.main.__wrapped__(remote_cfg)
    receipt_control_path.write_bytes(original_receipt_bytes)

    psf.main.__wrapped__(remote_cfg)
    remote_selection_dir = packed_dir / "sel" / selection_checksum
    remote_manifest = json.loads(
        (remote_selection_dir / "manifest.json").read_text(encoding="utf-8")
    )
    assert "created_at" not in remote_manifest
    assert remote_manifest == local_manifest
    assert {
        name: bucket._path(f"{prefixes.selection_output}/{name}").read_bytes()
        for name in local_bodies
    } == local_bodies
    remote_rng_state = built_pipelines[-1].selection_config.rng.get_state().clone()
    assert t.equal(remote_rng_state, local_rng_state)
    remote_next_draw = t.randint(
        0, 2**31, (1,), generator=built_pipelines[-1].selection_config.rng
    )
    assert t.equal(remote_next_draw, local_next_draw)

    materializer = RemoteSelectionMaterializer.discover(
        backend=bucket,
        remote=remote,
        local_selection_dir=remote_selection_dir,
        feature_root=corpus_root,
        expected_selection_checksum=selection_checksum,
        packed_row_fingerprint=str(packed_inventory["packed_row_fingerprint"]),
        packed_stems={
            name.removesuffix(".safetensors.zst") for name in local_bodies
        },
        label="remote ordinary pre-selection integration",
    )
    assert set(materializer.inventory) == set(local_bodies)

    bucket._path(f"{prefixes.selection_output}/manifest.json").unlink()
    (remote_selection_dir / "manifest.json").unlink()
    bucket.operations.clear()
    psf.main.__wrapped__(remote_cfg)
    packed_remote_paths = {f"{packed_prefix}/{name}" for name in local_bodies}
    assert not any(
        operation == "download" and path in packed_remote_paths
        for operation, path in bucket.operations
    )

    bucket.operations.clear()
    psf.main.__wrapped__(remote_cfg)
    assert not any(operation == "upload" for operation, _path in bucket.operations)
    assert not any(staging.iterdir())
