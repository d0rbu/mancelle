"""Unit tests for sync_push / sync_pull — mocks the actual huggingface_hub call."""

from __future__ import annotations

import hashlib
import json
import shutil
from pathlib import Path, PurePosixPath
from unittest.mock import patch

import pytest
import torch as t

import steering.artifacts.sync as sync_module
from steering.artifacts.config import ArtifactsConfig, CategoryConfig
from steering.artifacts.sync import (
    materialize_pulled_percentile_statistics,
    sync_pull,
    sync_push,
)
from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.feature_io import (
    COMPRESSED_EXT,
    CorruptFeatureShardError,
    TensorKey,
    glob_features,
    read_feature_value_count,
)
from steering.overlay_io import (
    CHECKSUM_PREFIX_LEN,
    OverlayKind,
    glob_overlay_shards,
    overlay_dir,
    overlay_tensors_content_digest,
    save_overlay_shard,
    write_manifest,
)
from steering.percentile_remote import RemotePercentileError, RemoteReceipt
from steering.remote_feature_inventory import (
    RemoteFeatureShard,
    remote_feature_inventory_digest,
)
from steering.remote_training import (
    VERIFIED_UPLOAD_CLAIMS_DIRNAME,
    RemoteObjectInfo,
    verified_local_object_info,
)
from steering.stats_io import (
    load_percentile_checksum,
    load_statistics,
    load_tokenizer_checksum,
    save_percentile_checksum,
    save_statistics,
    save_tokenizer_checksum,
)
from tests.scripts.test_production_manifest_v2 import _make_env

_CFG = ArtifactsConfig(
    bucket_owner="acme",
    bucket_name="bkt",
    categories={
        "base_checkpoints": CategoryConfig(
            local_root="/cache/base_checkpoints",
            remote_subpath="checkpoints/base",
            default_exclude=["**/optim_*.pt"],
        ),
    },
)


def _write_verified_claim(
    scoped_dir: Path,
    *,
    output_prefix: PurePosixPath,
    info: RemoteObjectInfo,
) -> None:
    remote_claim = (
        PurePosixPath(info.path).parent
        / VERIFIED_UPLOAD_CLAIMS_DIRNAME
        / PurePosixPath(info.path).name
        / f"{info.xet_hash}.json"
    )
    local_claim = scoped_dir.joinpath(*remote_claim.relative_to(output_prefix).parts)
    local_claim.parent.mkdir(parents=True, exist_ok=True)
    local_claim.write_text(
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
        + "\n",
        encoding="utf-8",
    )


def _write_complete_percentile_overlay(
    feature_dir: Path,
    stats_path: Path,
    checksum: str,
) -> Path:
    scoped_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    scoped_dir.mkdir(parents=True)
    receipts_dir = scoped_dir / "_receipts"
    receipts_dir.mkdir()
    feature_paths = glob_features(feature_dir)
    source_infos = [
        verified_local_object_info(
            local_path=feature_path,
            remote_path=f"raw/{feature_path.name}",
        )
        for feature_path in feature_paths
    ]
    inventory_sha256 = remote_feature_inventory_digest(
        tuple(
            RemoteFeatureShard(index=index, info=source_info)
            for index, source_info in enumerate(source_infos)
        )
    )
    remote_contract_path = f"raw/pct/{checksum[:16]}/_remote_contract.json"
    contract = {
        "schema_version": 1,
        "percentile_algorithm_version": 1,
        "source": {
            "bucket_id": "owner/test-bucket",
            "prefix": "raw",
            "shard_count": len(feature_paths),
            "inventory_sha256": inventory_sha256,
        },
        "expected_segment_size": 16,
        "percentile_table": {
            "checksum": checksum,
            "n_sae_features": load_statistics(stats_path).metadata.n_sae_features,
        },
        "output": {"prefix": f"raw/pct/{checksum[:16]}", "zstd_level": 12},
        "producer": {
            "device_type": "cpu",
            "torch_version": "test",
            "cuda_runtime": None,
            "gpu_name": None,
            "gpu_capability": None,
        },
        "serialization": {
            "safetensors_version": "test",
            "zstandard_version": "test",
        },
    }
    contract_payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_payload).hexdigest()
    contract_path = scoped_dir / "_remote_contract.json"
    contract_path.write_bytes(contract_payload)
    output_prefix = PurePosixPath(f"raw/pct/{checksum[:16]}")
    canonical_objects = [
        verified_local_object_info(
            local_path=contract_path,
            remote_path=remote_contract_path,
        )
    ]
    receipt_objects = []
    for index, feature_path in enumerate(feature_paths):
        payload_name = feature_path.name.removesuffix(f"_features{COMPRESSED_EXT}") + COMPRESSED_EXT
        values = t.zeros(read_feature_value_count(feature_path), dtype=t.uint8)
        tensors = {TensorKey.FEATURE_PERCENTILES.value: values}
        payload_path = save_overlay_shard(
            tensors,
            scoped_dir / payload_name.removesuffix(COMPRESSED_EXT),
        )
        source_info = source_infos[index]
        output_info = verified_local_object_info(
            local_path=payload_path,
            remote_path=f"raw/pct/{checksum[:16]}/{payload_name}",
        )
        receipt = RemoteReceipt(
            contract_sha256=contract_sha256,
            shard_index=index,
            source=source_info,
            output=output_info,
            semantic_content_sha256=overlay_tensors_content_digest(tensors),
            n_values=values.numel(),
        )
        receipt_path = receipts_dir / f"shard_{index:05d}.json"
        receipt_path.write_text(
            json.dumps(receipt.to_json(), sort_keys=True, separators=(",", ":"))
        )
        canonical_objects.append(output_info)
        receipt_objects.append(
            verified_local_object_info(
                local_path=receipt_path,
                remote_path=(output_prefix / "_receipts" / receipt_path.name).as_posix(),
            )
        )
    scoped_statistics = scoped_dir / stats_path.name
    shutil.copyfile(stats_path, scoped_statistics)
    save_percentile_checksum(scoped_statistics, checksum)
    remote_statistics_path = f"raw/pct/{checksum[:16]}/{stats_path.name}"
    statistics_info = verified_local_object_info(
        local_path=scoped_statistics,
        remote_path=remote_statistics_path,
    )
    write_manifest(
        scoped_dir,
        checksum,
        OverlayKind.PERCENTILE,
        metadata={
            "stats_file": stats_path.name,
            "remote_contract_path": remote_contract_path,
            "remote_contract_sha256": contract_sha256,
            "source_bucket_id": "owner/test-bucket",
            "source_prefix": "raw",
            "source_shard_count": len(feature_paths),
            "source_inventory_sha256": inventory_sha256,
            "n_sae_features": load_statistics(stats_path).metadata.n_sae_features,
            "producer": contract["producer"],
            "statistics_object": {
                "path": statistics_info.path,
                "byte_length": statistics_info.byte_length,
                "xet_hash": statistics_info.xet_hash,
            },
            "receipt_count": len(feature_paths),
        },
        base_shard_paths=feature_paths,
        include_created_at=False,
    )
    canonical_objects.extend(
        [
            statistics_info,
            verified_local_object_info(
                local_path=scoped_dir / "manifest.json",
                remote_path=(output_prefix / "manifest.json").as_posix(),
            ),
        ]
    )
    for info in [*canonical_objects, *receipt_objects]:
        _write_verified_claim(
            scoped_dir,
            output_prefix=output_prefix,
            info=info,
        )
    return scoped_dir


def _statistics_checksum(path: Path) -> str:
    statistics = load_statistics(path)
    return compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )


def _alternate_statistics(stats_path: Path, destination: Path) -> tuple[Path, str]:
    statistics = load_statistics(stats_path)
    statistics.per_feature_stats["0"].percentiles["p50"] += 0.037
    alternate = save_statistics(statistics, destination / "feature_statistics")
    return alternate, _statistics_checksum(alternate)


def test_verified_claim_inventory_rejects_wrong_parent_and_non_directory(
    tmp_path: Path,
) -> None:
    info = RemoteObjectInfo("wrong/body", 1, "a" * 64)
    with pytest.raises(ValueError, match="canonical-object inventory"):
        sync_module._validate_verified_claim_directory(
            tmp_path / "claims",
            expected_objects=[info],
            canonical_parent=PurePosixPath("expected"),
        )

    claim_root = tmp_path / "claims"
    claim_root.mkdir()
    (claim_root / "body").write_bytes(b"not a directory")
    with pytest.raises(ValueError, match="claim is unreadable"):
        sync_module._validate_verified_claim_directory(
            claim_root,
            expected_objects=[info],
            canonical_parent=PurePosixPath("wrong"),
        )


@pytest.fixture(autouse=True)
def _stub_local_root_exists(monkeypatch):
    """Stub `Path.exists` to True for any /cache/* path so the sync_push
    "skip-if-missing" guard doesn't trip on these unit tests' fictitious
    paths. The `test_sync_push_skips_when_local_root_missing` test
    overrides this with its own monkeypatch to exercise the skip path."""
    real_exists = Path.exists

    def _exists_true_for_cache_paths(self):
        if str(self).startswith("/cache/"):
            return True
        return real_exists(self)

    monkeypatch.setattr(Path, "exists", _exists_true_for_cache_paths)


def test_sync_push_calls_sync_bucket_with_correct_endpoints(tmp_path):
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
            delete=False,
        )
    args, kwargs = mock_sync.call_args
    assert args[0].endswith("/cache/base_checkpoints") or args[0] == "/cache/base_checkpoints"
    assert args[1] == "hf://buckets/acme/bkt/checkpoints/base"
    assert kwargs.get("exclude") == ["**/optim_*.pt"]
    assert kwargs.get("dry_run") is False
    assert "delete" not in kwargs or kwargs["delete"] is False


def test_sync_pull_swaps_endpoints():
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_pull(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
        )
    args, _ = mock_sync.call_args
    # pull: source is remote, dest is local
    assert args[0] == "hf://buckets/acme/bkt/checkpoints/base"
    assert args[1].endswith("/cache/base_checkpoints") or args[1] == "/cache/base_checkpoints"


def test_feature_pull_materializes_overlay_statistics_and_preserves_root_metadata(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    save_tokenizer_checksum(stats_path, "root-tokenizer")
    _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    cfg = ArtifactsConfig(
        bucket_owner="acme",
        bucket_name="bkt",
        categories={
            "features": CategoryConfig(
                local_root=str(tmp_path / "pulled"),
                remote_subpath="features",
            )
        },
    )

    with patch("steering.artifacts.sync.sync_bucket"):
        sync_pull(
            cfg=cfg,
            category_name="features",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
        )
    assert load_percentile_checksum(stats_path) == checksum
    assert load_tokenizer_checksum(stats_path) == "root-tokenizer"


def test_feature_pull_dry_run_never_materializes_statistics(tmp_path: Path) -> None:
    cfg = ArtifactsConfig(
        bucket_owner="acme",
        bucket_name="bkt",
        categories={
            "features": CategoryConfig(
                local_root=str(tmp_path / "pulled"),
                remote_subpath="features",
            )
        },
    )
    with (
        patch("steering.artifacts.sync.sync_bucket"),
        patch("steering.artifacts.sync.materialize_pulled_percentile_statistics") as materialize,
    ):
        sync_pull(
            cfg=cfg,
            category_name="features",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=True,
        )
    materialize.assert_not_called()


def test_pull_materializer_ignores_ordinary_overlays_in_a_mixed_tree(
    tmp_path: Path,
) -> None:
    ordinary_feature_dir, ordinary_stats = _make_env(
        tmp_path / "pulled" / "ordinary",
        n_shards=1,
    )
    ordinary_statistics = load_statistics(ordinary_stats)
    ordinary_checksum = compute_percentile_checksum(
        build_percentile_table(
            ordinary_statistics.per_feature_stats,
            ordinary_statistics.metadata.n_sae_features,
        )
    )
    ordinary_dir = overlay_dir(
        ordinary_feature_dir,
        OverlayKind.PERCENTILE,
        ordinary_checksum,
    )
    ordinary_dir.mkdir(parents=True)
    write_manifest(
        ordinary_dir,
        ordinary_checksum,
        OverlayKind.PERCENTILE,
        base_shard_paths=glob_features(ordinary_feature_dir),
    )

    remote_feature_dir, remote_stats = _make_env(
        tmp_path / "pulled" / "remote",
        n_shards=1,
    )
    remote_statistics = load_statistics(remote_stats)
    remote_checksum = compute_percentile_checksum(
        build_percentile_table(
            remote_statistics.per_feature_stats,
            remote_statistics.metadata.n_sae_features,
        )
    )
    _write_complete_percentile_overlay(
        remote_feature_dir,
        remote_stats,
        remote_checksum,
    )

    assert materialize_pulled_percentile_statistics(tmp_path / "pulled") == 1
    assert load_percentile_checksum(ordinary_stats) is None
    assert load_percentile_checksum(remote_stats) == remote_checksum


def test_pull_materializer_selects_family_matching_root_statistics(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    current_checksum = _statistics_checksum(stats_path)
    old_stats, old_checksum = _alternate_statistics(stats_path, tmp_path / "old")
    old_dir = _write_complete_percentile_overlay(feature_dir, old_stats, old_checksum)
    _write_complete_percentile_overlay(feature_dir, stats_path, current_checksum)
    # If the stale family were validated, this deliberately broken claim would fail.
    next(
        path
        for path in (old_dir / VERIFIED_UPLOAD_CLAIMS_DIRNAME).rglob("*.json")
        if path.is_file()
    ).write_text(
        "stale-family-claim",
        encoding="utf-8",
    )

    assert materialize_pulled_percentile_statistics(tmp_path / "pulled") == 1
    assert load_percentile_checksum(stats_path) == current_checksum


def test_pull_materializer_ignores_stale_family_when_root_selects_no_candidate(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    old_stats, old_checksum = _alternate_statistics(stats_path, tmp_path / "old")
    _write_complete_percentile_overlay(feature_dir, old_stats, old_checksum)

    assert materialize_pulled_percentile_statistics(tmp_path / "pulled") == 0
    assert load_percentile_checksum(stats_path) is None


def test_pull_materializer_rejects_ambiguous_families_before_root_mutation(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    current_checksum = _statistics_checksum(stats_path)
    old_stats, old_checksum = _alternate_statistics(stats_path, tmp_path / "old")
    _write_complete_percentile_overlay(feature_dir, old_stats, old_checksum)
    _write_complete_percentile_overlay(feature_dir, stats_path, current_checksum)
    stats_path.unlink()

    with pytest.raises(ValueError, match="multiple remote percentile families"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert not stats_path.exists()


@pytest.mark.parametrize(
    "mode",
    [
        "missing_top",
        "missing_contract",
        "missing_payload",
        "missing_statistics",
        "missing_manifest",
        "missing_receipt",
        "wrong_target",
        "wrong_size",
        "wrong_xet",
        "duplicate",
    ],
)
def test_pull_materializer_requires_exact_verified_upload_claims(
    tmp_path: Path,
    mode: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    scoped_dir = _write_complete_percentile_overlay(
        feature_dir,
        stats_path,
        _statistics_checksum(stats_path),
    )
    top_claims = scoped_dir / VERIFIED_UPLOAD_CLAIMS_DIRNAME
    receipt_claims = scoped_dir / "_receipts" / VERIFIED_UPLOAD_CLAIMS_DIRNAME
    if mode == "missing_top":
        shutil.rmtree(top_claims)
    elif mode == "missing_receipt":
        shutil.rmtree(receipt_claims)
    elif mode.startswith("missing_"):
        canonical_name = {
            "missing_contract": "_remote_contract.json",
            "missing_payload": next(iter(glob_overlay_shards(scoped_dir))).name,
            "missing_statistics": "feature_statistics.safetensors",
            "missing_manifest": "manifest.json",
        }[mode]
        shutil.rmtree(top_claims / canonical_name)
    else:
        claim = next(path for path in top_claims.rglob("*.json") if path.is_file())
        if mode == "duplicate":
            shutil.copyfile(claim, claim.with_name("f" * 64 + ".json"))
        else:
            body = json.loads(claim.read_text())
            key, value = {
                "wrong_target": ("remote_path", "wrong/path"),
                "wrong_size": ("byte_length", body["byte_length"] + 1),
                "wrong_xet": ("xet_hash", "0" * 64),
            }[mode]
            body[key] = value
            claim.write_text(
                json.dumps(body, sort_keys=True, separators=(",", ":")) + "\n",
                encoding="utf-8",
            )

    before = stats_path.read_bytes()
    expected_error = "receipt coverage" if mode == "missing_receipt" else "verified-upload claim"
    with pytest.raises(ValueError, match=expected_error):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize(
    ("mode", "match"),
    [
        ("payload", "payload coverage"),
        ("malformed_receipt", "receipt is malformed"),
        ("noncanonical_receipt", "receipt is not canonical"),
        ("noncontiguous_receipt", "indices are not contiguous"),
        ("semantic_digest", "semantic digest differs"),
    ],
)
def test_pull_materializer_rejects_structural_family_corruption_before_claims(
    tmp_path: Path,
    mode: str,
    match: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    scoped_dir = _write_complete_percentile_overlay(
        feature_dir,
        stats_path,
        _statistics_checksum(stats_path),
    )
    receipt_path = scoped_dir / "_receipts" / "shard_00000.json"
    if mode == "payload":
        next(iter(glob_overlay_shards(scoped_dir))).unlink()
    else:
        receipt = json.loads(receipt_path.read_text())
        if mode == "malformed_receipt":
            receipt["schema_version"] = True
        elif mode == "noncontiguous_receipt":
            receipt["shard_index"] = 1
            receipt_path = receipt_path.rename(receipt_path.with_name("shard_00001.json"))
        elif mode == "semantic_digest":
            receipt["output"]["semantic_content_sha256"] = "f" * 64
        receipt_path.write_text(
            (
                json.dumps(receipt, indent=2)
                if mode == "noncanonical_receipt"
                else json.dumps(receipt, sort_keys=True, separators=(",", ":"))
            ),
            encoding="utf-8",
        )

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match=match):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pull_materializer_refences_raw_base_hash_after_claim_validation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    _write_complete_percentile_overlay(
        feature_dir,
        stats_path,
        _statistics_checksum(stats_path),
    )
    original_hash = sync_module.compute_base_shard_content_hash
    calls = 0

    def changes_after_initial_validation(paths):
        nonlocal calls
        calls += 1
        observed = original_hash(paths)
        return observed if calls == 1 else "f" * 64

    monkeypatch.setattr(
        sync_module,
        "compute_base_shard_content_hash",
        changes_after_initial_validation,
    )
    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="changed during pulled percentile validation"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert calls == 2
    assert stats_path.read_bytes() == before


def test_pull_claims_reject_self_consistent_payload_and_receipt_replacement(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    scoped_dir = _write_complete_percentile_overlay(
        feature_dir,
        stats_path,
        _statistics_checksum(stats_path),
    )
    payload_path = next(iter(glob_overlay_shards(scoped_dir)))
    receipt_path = scoped_dir / "_receipts" / "shard_00000.json"
    receipt = json.loads(receipt_path.read_text())
    replacement = t.ones(read_feature_value_count(glob_features(feature_dir)[0]), dtype=t.uint8)
    save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES.value: replacement},
        Path(str(payload_path).removesuffix(COMPRESSED_EXT)),
    )
    output_info = verified_local_object_info(
        local_path=payload_path,
        remote_path=receipt["output"]["path"],
    )
    receipt["output"] = {
        "path": output_info.path,
        "byte_length": output_info.byte_length,
        "xet_hash": output_info.xet_hash,
        "semantic_content_sha256": overlay_tensors_content_digest(
            {TensorKey.FEATURE_PERCENTILES.value: replacement}
        ),
    }
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, separators=(",", ":")),
        encoding="utf-8",
    )

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="verified-upload claim"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize(
    ("manifest_mutation", "match"),
    [
        (lambda manifest: "not-json", "no readable manifest"),
        (
            lambda manifest: {**manifest, "kind": OverlayKind.SEG_BOUNDARY.value},
            "does not identify this percentile overlay",
        ),
        (
            lambda manifest: {**manifest, "checksum": "A" * 64},
            "does not identify this percentile overlay",
        ),
        (
            lambda manifest: {**manifest, "unexpected": "field"},
            "unexpected or missing keys",
        ),
        (
            lambda manifest: {
                **manifest,
                "metadata": {**manifest["metadata"], "unexpected": "field"},
            },
            "metadata differs",
        ),
        (
            lambda manifest: {
                **manifest,
                "metadata": {
                    **manifest["metadata"],
                    "producer": {
                        **manifest["metadata"]["producer"],
                        "torch_version": "wrong",
                    },
                },
            },
            "metadata differs",
        ),
    ],
)
def test_pulled_percentile_materialization_fails_closed_on_manifest_identity(
    tmp_path: Path,
    manifest_mutation,
    match: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    manifest_path = scoped_dir / "manifest.json"
    manifest = manifest_mutation(json.loads(manifest_path.read_text()))
    if isinstance(manifest, str):
        manifest_path.write_text(manifest, encoding="utf-8")
    else:
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match=match):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_manifest_must_have_deterministic_producer_bytes(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    scoped_dir = _write_complete_percentile_overlay(
        feature_dir,
        stats_path,
        _statistics_checksum(stats_path),
    )
    manifest_path = scoped_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_bytes())
    manifest_path.write_text(json.dumps(manifest, sort_keys=True), encoding="utf-8")

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="manifest is not canonical"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_manifest_disappearance_fails_before_root_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    scoped_dir = _write_complete_percentile_overlay(
        feature_dir,
        stats_path,
        _statistics_checksum(stats_path),
    )
    manifest_path = scoped_dir / "manifest.json"
    original_read_bytes = Path.read_bytes

    def fail_manifest_read(path: Path) -> bytes:
        if path == manifest_path:
            raise OSError("manifest disappeared")
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", fail_manifest_read)
    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="manifest is unreadable"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_rejects_manifest_and_stats_only(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    scoped_dir.mkdir(parents=True)
    scoped_statistics = scoped_dir / stats_path.name
    shutil.copyfile(stats_path, scoped_statistics)
    save_percentile_checksum(scoped_statistics, checksum)
    remote_statistics_path = f"raw/pct/{checksum[:16]}/{stats_path.name}"
    statistics_info = verified_local_object_info(
        local_path=scoped_statistics,
        remote_path=remote_statistics_path,
    )
    write_manifest(
        scoped_dir,
        checksum,
        OverlayKind.PERCENTILE,
        metadata={
            "stats_file": stats_path.name,
            "remote_contract_path": f"raw/pct/{checksum[:16]}/_remote_contract.json",
            "statistics_object": {
                "path": statistics_info.path,
                "byte_length": statistics_info.byte_length,
                "xet_hash": statistics_info.xet_hash,
            },
            "receipt_count": 1,
        },
        base_shard_paths=glob_features(feature_dir),
    )

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="no readable remote contract"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_rejects_stale_local_raw_family(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    raw_path = glob_features(feature_dir)[0]
    raw_path.write_bytes(raw_path.read_bytes() + b"replacement")

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="base-shard identity"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_rechecks_raw_family_before_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    raw_path = glob_features(feature_dir)[0]
    original_count = read_feature_value_count
    mutated = False

    def count_then_mutate(path: Path) -> int:
        nonlocal mutated
        count = original_count(path)
        if not mutated:
            mutated = True
            path.write_bytes(path.read_bytes() + b"concurrent replacement")
        return count

    monkeypatch.setattr(
        "steering.artifacts.sync.read_feature_value_count",
        count_then_mutate,
    )
    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="receipt"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert raw_path.read_bytes().endswith(b"concurrent replacement")
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize("corruption", ["unreadable", "key", "dtype", "length"])
def test_pulled_percentile_materialization_rejects_corrupt_payload(
    tmp_path: Path,
    corruption: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    payload = next(scoped_dir.glob(f"*{COMPRESSED_EXT}"))
    if corruption == "unreadable":
        payload.write_bytes(b"not an overlay")
    elif corruption == "key":
        save_overlay_shard({"wrong": t.zeros(30, dtype=t.uint8)}, payload)
    elif corruption == "dtype":
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES.value: t.zeros(30, dtype=t.int16)},
            payload,
        )
    else:
        save_overlay_shard(
            {TensorKey.FEATURE_PERCENTILES.value: t.zeros(0, dtype=t.uint8)},
            payload,
        )

    before = stats_path.read_bytes()
    with pytest.raises(CorruptFeatureShardError):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_rejects_same_shape_payload_drift(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    payload = next(scoped_dir.glob(f"*{COMPRESSED_EXT}"))
    save_overlay_shard(
        {
            TensorKey.FEATURE_PERCENTILES.value: t.ones(
                read_feature_value_count(glob_features(feature_dir)[0]),
                dtype=t.uint8,
            )
        },
        payload,
    )

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="differs from its receipt"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_finally_fences_payload_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    payload = next(scoped_dir.glob(f"*{COMPRESSED_EXT}"))
    original_verify = verified_local_object_info
    payload_checks = 0

    def verify_then_mutate(*, local_path: Path, remote_path: str):
        nonlocal payload_checks
        info = original_verify(local_path=local_path, remote_path=remote_path)
        if local_path == payload:
            payload_checks += 1
            if payload_checks == 1:
                save_overlay_shard(
                    {
                        TensorKey.FEATURE_PERCENTILES.value: t.ones(
                            read_feature_value_count(glob_features(feature_dir)[0]),
                            dtype=t.uint8,
                        )
                    },
                    payload,
                )
        return info

    monkeypatch.setattr(
        "steering.artifacts.sync.verified_local_object_info",
        verify_then_mutate,
    )
    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="changed during final validation"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert payload_checks == 2
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["missing", "malformed"])
def test_pulled_percentile_materialization_requires_exact_receipts(
    tmp_path: Path,
    mode: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    receipt = next((scoped_dir / "_receipts").glob("*.json"))
    if mode == "missing":
        receipt.unlink()
        expected = "receipt coverage"
    else:
        receipt.write_text("not json")
        expected = "receipt is unreadable"

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match=expected):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["missing", "noncanonical", "tampered"])
def test_pulled_percentile_materialization_requires_exact_contract(
    tmp_path: Path,
    mode: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    contract = scoped_dir / "_remote_contract.json"
    if mode == "missing":
        contract.unlink()
        expected = "no readable remote contract"
    else:
        payload = json.loads(contract.read_bytes())
        if mode == "noncanonical":
            contract.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            expected = "contract identity is invalid"
        else:
            payload["producer"]["torch_version"] = "tampered"
            contract.write_text(
                json.dumps(payload, sort_keys=True, separators=(",", ":")),
                encoding="utf-8",
            )
            expected = "manifest disagrees"

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match=expected):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["count", "inventory", "bool_schema", "extra_key"])
def test_pulled_percentile_contract_is_bound_to_the_actual_source_inventory(
    tmp_path: Path,
    mode: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=2)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    contract_path = scoped_dir / "_remote_contract.json"
    manifest_path = scoped_dir / "manifest.json"
    contract = json.loads(contract_path.read_bytes())
    manifest = json.loads(manifest_path.read_bytes())
    if mode == "count":
        contract["source"]["shard_count"] += 1
        manifest["metadata"]["source_shard_count"] += 1
        manifest["metadata"]["receipt_count"] += 1
        expected = "source count differs"
    elif mode == "inventory":
        contract["source"]["inventory_sha256"] = "f" * 64
        manifest["metadata"]["source_inventory_sha256"] = "f" * 64
        expected = "receipt inventory differs"
    elif mode == "bool_schema":
        contract["schema_version"] = True
        expected = "contract is malformed"
    else:
        contract["unexpected"] = "field"
        expected = "contract is malformed"
    contract_payload = json.dumps(
        contract,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    contract_sha256 = hashlib.sha256(contract_payload).hexdigest()
    contract_path.write_bytes(contract_payload)
    manifest["metadata"]["remote_contract_sha256"] = contract_sha256
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    for receipt_path in (scoped_dir / "_receipts").glob("*.json"):
        receipt = json.loads(receipt_path.read_bytes())
        receipt["contract_sha256"] = contract_sha256
        receipt_path.write_text(
            json.dumps(receipt, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match=expected):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_path_must_match_contract_checksum(tmp_path: Path) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    wrong_dir = scoped_dir.with_name("f" * CHECKSUM_PREFIX_LEN)
    scoped_dir.rename(wrong_dir)
    stats_path.unlink()

    with pytest.raises(ValueError, match="path disagrees with its checksum"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert not stats_path.exists()


@pytest.mark.parametrize("malformed_manifest", [False, True])
def test_pulled_percentile_manifest_requires_scoped_statistics(
    tmp_path: Path,
    malformed_manifest: bool,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    (scoped_dir / stats_path.name).unlink()
    if malformed_manifest:
        (scoped_dir / "manifest.json").write_text("not-json", encoding="utf-8")
        expected = "no readable manifest"
    else:
        expected = "no scoped statistics"

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match=expected):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


@pytest.mark.parametrize("mode", ["filename", "extra"])
def test_pulled_percentile_materialization_rejects_receipt_inventory_anomalies(
    tmp_path: Path,
    mode: str,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    receipts_dir = scoped_dir / "_receipts"
    receipt = next(receipts_dir.glob("*.json"))
    if mode == "filename":
        receipt.rename(receipts_dir / "shard_00001.json")
    else:
        (receipts_dir / "unexpected.txt").write_text("extra")

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="receipt coverage|filename/index"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_rejects_statistics_byte_drift(
    tmp_path: Path,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    save_tokenizer_checksum(scoped_dir / stats_path.name, "unmanifested-tokenizer")

    before = stats_path.read_bytes()
    with pytest.raises(ValueError, match="metadata differs"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert stats_path.read_bytes() == before


def test_pulled_percentile_materialization_binds_final_statistics_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    feature_dir, stats_path = _make_env(tmp_path / "pulled", n_shards=1)
    statistics = load_statistics(stats_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(
            statistics.per_feature_stats,
            statistics.metadata.n_sae_features,
        )
    )
    scoped_dir = _write_complete_percentile_overlay(feature_dir, stats_path, checksum)
    scoped_statistics = scoped_dir / stats_path.name
    original_verify = verified_local_object_info
    mutated = False

    def verify_then_mutate(*, local_path: Path, remote_path: str):
        nonlocal mutated
        info = original_verify(local_path=local_path, remote_path=remote_path)
        if local_path == scoped_statistics and not mutated:
            mutated = True
            save_tokenizer_checksum(local_path, "concurrent-unmanifested-tokenizer")
        return info

    monkeypatch.setattr(
        "steering.artifacts.sync.verified_local_object_info",
        verify_then_mutate,
    )
    before = stats_path.read_bytes()
    with pytest.raises(RemotePercentileError, match="manifest identity"):
        materialize_pulled_percentile_statistics(tmp_path / "pulled")
    assert mutated
    assert stats_path.read_bytes() == before


def test_sync_push_dry_run_propagates():
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=True,
            delete=False,
        )
    _, kwargs = mock_sync.call_args
    assert kwargs["dry_run"] is True


def test_sync_push_delete_requires_explicit_opt_in():
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
            delete=True,
        )
    _, kwargs = mock_sync.call_args
    assert kwargs["delete"] is True


def test_sync_user_filters_extend_defaults():
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=["*.safetensors"],
            user_exclude=["scratch/**"],
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
            delete=False,
        )
    _, kwargs = mock_sync.call_args
    assert kwargs.get("include") == ["*.safetensors"]
    assert sorted(kwargs.get("exclude", [])) == sorted(["**/optim_*.pt", "scratch/**"])


def test_sync_unknown_category_raises():
    with pytest.raises(KeyError):
        sync_push(
            cfg=_CFG,
            category_name="nope",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
            delete=False,
        )


# ---------------------------------------------------------------------------
# Robustness: error propagation, filter passing, mkdir safety
# ---------------------------------------------------------------------------


def test_sync_push_propagates_sync_bucket_exception():
    """If huggingface_hub.sync_bucket raises, sync_push must propagate the error
    rather than swallowing it (no silent failures on real network/auth issues)."""
    with patch("steering.artifacts.sync.sync_bucket", side_effect=RuntimeError("auth failed")):
        with pytest.raises(RuntimeError, match="auth failed"):
            sync_push(
                cfg=_CFG,
                category_name="base_checkpoints",
                user_include=None,
                user_exclude=None,
                no_default_include=False,
                no_default_exclude=False,
                dry_run=False,
                delete=False,
            )


def test_sync_pull_propagates_sync_bucket_exception():
    """sync_pull must propagate sync_bucket errors too."""
    with patch("steering.artifacts.sync.sync_bucket", side_effect=ConnectionError("network down")):
        with pytest.raises(ConnectionError, match="network down"):
            sync_pull(
                cfg=_CFG,
                category_name="base_checkpoints",
                user_include=None,
                user_exclude=None,
                no_default_include=False,
                no_default_exclude=False,
                dry_run=False,
            )


def test_sync_push_passes_none_when_filters_empty():
    """When the resolved filters are empty, sync_bucket receives include=None / exclude=None
    (NOT the empty list, which would mean 'match nothing' in some hub versions)."""
    cfg_no_defaults = ArtifactsConfig(
        bucket_owner="acme",
        bucket_name="bkt",
        categories={
            "empty_cat": CategoryConfig(
                local_root="/cache/empty",
                remote_subpath="empty",
                # default_include and default_exclude default to []
            ),
        },
    )
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=cfg_no_defaults,
            category_name="empty_cat",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
            delete=False,
        )
    _, kwargs = mock_sync.call_args
    assert kwargs["include"] is None
    assert kwargs["exclude"] is None


def test_sync_pull_dry_run_does_not_create_destination(tmp_path):
    """Destination creation belongs to sync_bucket's execution path, not the wrapper."""
    local_root = tmp_path / "missing" / "pull_destination"
    cfg = ArtifactsConfig(
        bucket_owner="acme",
        bucket_name="bkt",
        categories={
            "x": CategoryConfig(
                local_root=str(local_root),
                remote_subpath="x",
            ),
        },
    )
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_pull(
            cfg=cfg,
            category_name="x",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=True,
        )
    assert not local_root.exists()
    _, kwargs = mock_sync.call_args
    assert kwargs["dry_run"] is True


def test_sync_push_no_default_exclude_clears_default():
    """no_default_exclude=True clears the category's default_exclude before applying user excludes."""
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=True,
            dry_run=False,
            delete=False,
        )
    _, kwargs = mock_sync.call_args
    # default was ["**/optim_*.pt"]; cleared and no user excludes → None
    assert kwargs["exclude"] is None


def test_sync_pull_does_not_pass_delete_kwarg():
    """sync_pull must never pass `delete=` to sync_bucket — pull is non-destructive."""
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_pull(
            cfg=_CFG,
            category_name="base_checkpoints",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
        )
    _, kwargs = mock_sync.call_args
    assert "delete" not in kwargs


def test_sync_push_skips_when_local_root_missing(monkeypatch):
    """When the category's local_root does not exist, sync_push logs a warning
    and returns without calling sync_bucket — supports layouts where some
    categories simply have no local artifacts (e.g., this project's
    sft_checkpoints lives nested under base_checkpoints, not as its own dir)."""
    # Override the autouse fixture: make this category's path NOT exist.
    monkeypatch.setattr(Path, "exists", lambda self: False)
    cfg = ArtifactsConfig(
        bucket_owner="acme",
        bucket_name="bkt",
        categories={
            "missing": CategoryConfig(
                local_root="/definitely/does/not/exist/anywhere",
                remote_subpath="checkpoints/missing",
            ),
        },
    )
    with patch("steering.artifacts.sync.sync_bucket") as mock_sync:
        sync_push(
            cfg=cfg,
            category_name="missing",
            user_include=None,
            user_exclude=None,
            no_default_include=False,
            no_default_exclude=False,
            dry_run=False,
            delete=False,
        )
    assert not mock_sync.called  # skipped, not called
