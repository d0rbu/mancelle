"""Issue #806 trainer discovery composition and cross-mode identity tests."""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

import pytest
import torch as t

from errors import PackedRowInvariantError
from scripts.train.conditional_encdec_train import (
    PackedSelectionDiscovery,
    discover_packed_selection,
)
from steering.conditional.delta import (
    CLUSTER_DELTA_SEMANTICS_KEY,
    CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS,
    DeltaMode,
    FeatureDeltaConfig,
    splice_counterfactual_manifest_identity,
)
from steering.conditional.encoder_decoder.fp_target_source import (
    certify_fp_target_source,
    validate_fp_target_source_certificate_remote,
)
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    build_clustered_selection_shard_integrity,
    canonical_feature_clusters_descriptor,
    resolve_feature_clusters,
)
from steering.conditional.splice_counterfactual import (
    canonical_feature_delta_dict,
    derived_feature_fingerprint,
    splice_sidecar_fingerprint,
    validate_packed_provenance,
)
from steering.conditional.splice_provenance import SpliceProvenanceError
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import OverlayKind, save_overlay_shard, write_manifest
from steering.packed_batch_io import (
    compute_selection_hash,
    load_packed_row_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.refined_types import PositiveInt
from steering.remote_training import (
    REMOTE_LOADER_KEY,
    REMOTE_PACKED_INVENTORY_FILENAME,
    RemotePackedRowMaterializer,
    RemoteTrainingConfig,
    RemoteTrainingError,
    build_remote_loader_certificate,
    build_remote_run_fingerprint,
    publish_remote_packed_inventory,
)
from steering.stats_io import save_percentile_checksum, save_tokenizer_checksum
from tests.encdec_test_helpers import _make_identity_selection, _make_test_shard
from tests.scripts.test_checksum_pipeline_wiring import _make_stats_file
from tests.steering.test_remote_training import FakeBucket

_PERCENTILE_CHECKSUM = "1" * 64
_TOKENIZER_CHECKSUM = "2" * 64
_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(32)
_CONTEXT_LEN = 1024
_SEGMENT_SIZE = 16
_NUM_BUCKETS = 4
_SEED = 17
_BOUNDARY_FINGERPRINT = "f" * 64
_FEATURE_SELECTION = {
    "threshold_type": "segment_topk",
    "threshold": 3,
}
_FEATURE_DELTA = {
    "mode": "splice_counterfactual",
    "neg_delta_ratio": 1.0,
}


@dataclass(frozen=True)
class _DiscoveryFixture:
    producer_root: Path
    derived_dir: Path
    base_dir: Path
    cluster_path: Path
    train_packed_dir: Path
    derived_val_packed_dir: Path
    base_val_packed_dir: Path
    train_selection_hash: str
    base_selection_hash: str
    derived_feature_fingerprint: str
    bucket: FakeBucket


def _packed_train_dir(feature_dir: Path) -> Path:
    return (
        feature_dir
        / "pct"
        / _PERCENTILE_CHECKSUM[:16]
        / "tok"
        / _TOKENIZER_CHECKSUM[:16]
        / f"c{_CONTEXT_LEN}"
        / "sall"
    )


def _write_stats(feature_dir: Path) -> Path:
    stats_path = _make_stats_file(
        feature_dir,
        n_sae=_N_SAE_FEATURES,
    )
    save_percentile_checksum(stats_path, _PERCENTILE_CHECKSUM)
    save_tokenizer_checksum(stats_path, _TOKENIZER_CHECKSUM)
    return stats_path


def _write_derived_lineage(feature_dir: Path) -> None:
    feature_dir.mkdir(parents=True, exist_ok=True)
    for index in range(2):
        (feature_dir / f"source_{index:05d}_features.safetensors.zst").write_bytes(
            f"derived-feature-body-{index}".encode()
        )
    sidecar_dir = feature_dir / "_splice_provenance"
    sidecar_dir.mkdir()
    for index in range(2):
        (sidecar_dir / f"source_{index:05d}.json").write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "source": f"source_{index:05d}",
                }
            ),
            encoding="utf-8",
        )


def _write_packed_family(
    packed_dir: Path,
    *,
    n_shards: int,
    seed_offset: int,
) -> None:
    prov_dir = provenance_dir(packed_dir)
    prov_dir.mkdir(parents=True)
    packed_paths: list[Path] = []
    for index in range(n_shards):
        shard = _make_test_shard(
            n_rows=1,
            context_len=_CONTEXT_LEN,
            segs_per_row=2,
            feats_per_seg=3,
            n_features=_N_SAE_FEATURES,
            seed=seed_offset + index,
        )
        stem = f"shard_{index:05d}"
        packed_paths.append(save_packed_row_shard(shard, packed_dir / stem))
        save_overlay_shard(
            {
                PROVENANCE_BASE_SHARD_IDX_KEY: t.tensor(
                    [index % 2, index % 2],
                    dtype=t.int32,
                ),
                PROVENANCE_DOC_IDX_KEY: t.tensor(
                    [index, index],
                    dtype=t.int32,
                ),
                PROVENANCE_SEG_ORDINAL_KEY: t.tensor(
                    [0, 1],
                    dtype=t.int32,
                ),
            },
            prov_dir / stem,
        )
    write_manifest(
        packed_dir,
        f"{_PERCENTILE_CHECKSUM}:{_TOKENIZER_CHECKSUM}",
        OverlayKind.PACKED_ROW,
        metadata={
            "type": "packed_rows",
            "context_len": _CONTEXT_LEN,
            "pct_checksum": _PERCENTILE_CHECKSUM,
            "tok_checksum": _TOKENIZER_CHECKSUM,
            "base_shard_stems": [
                "source_00000",
                "source_00001",
            ],
            "provenance_shards": n_shards,
            "max_n_seg": 2,
            "max_seg_size": _CONTEXT_LEN,
        },
        base_shard_paths=packed_paths,
    )
    certify_fp_target_source(
        packed_dir,
        n_sae_features=_N_SAE_FEATURES,
    )


def _write_selection_family(
    *,
    packed_dir: Path,
    selection_hash: str,
    clusters: FeatureClusterConsolidation,
    derived_dir: Path | None,
) -> Path:
    packed_info = validate_packed_provenance(
        packed_dir,
        cache_dir=packed_dir,
    )
    selection_dir = packed_dir / "sel" / selection_hash
    selection_paths: list[Path] = []
    for packed_path in packed_info.packed_shard_paths:
        selection_paths.append(
            save_selection_shard(
                _make_identity_selection(load_packed_row_shard(packed_path)),
                selection_dir / packed_path.name,
            )
        )
    integrity = build_clustered_selection_shard_integrity(selection_paths)
    cluster_metadata = clusters.metadata_dict()
    cluster_metadata["status"] = "complete"
    metadata: dict[str, Any] = {
        "feature_mask": None,
        "feature_clusters": cluster_metadata,
        "packed_rows": {"packed_row_fingerprint": (packed_info.packed_row_fingerprint)},
        "packed_provenance": {
            "packed_stem_fingerprint": (packed_info.packed_stem_fingerprint),
            "packed_row_fingerprint": (packed_info.packed_row_fingerprint),
            "provenance_shard_count": len(packed_info.provenance_shard_paths),
            "provenance_fingerprint": (packed_info.provenance_fingerprint),
        },
        SELECTION_SHARD_INTEGRITY_KEY: integrity,
    }
    cluster_delta_semantics: str | None = None
    if derived_dir is not None:
        metadata["feature_delta"] = canonical_feature_delta_dict(
            FeatureDeltaConfig(
                mode=DeltaMode.splice_counterfactual,
                neg_delta_ratio=1.0,
            )
        )
        metadata["splice_counterfactual"] = {
            **splice_counterfactual_manifest_identity(clustered=True),
            "status": "complete",
            "derived_shard_stems": (packed_info.base_shard_stems),
            "sidecar_fingerprint": splice_sidecar_fingerprint(derived_dir),
            "derived_feature_fingerprint": (
                derived_feature_fingerprint(
                    derived_dir,
                    expected_stems=(packed_info.base_shard_stems),
                )
            ),
            "boundary_fingerprint": _BOUNDARY_FINGERPRINT,
        }
        cluster_delta_semantics = CLUSTERED_SPLICE_COUNTERFACTUAL_SEMANTICS
    loader = build_remote_loader_certificate(
        selection_checksum=selection_hash,
        selection_inventory_aggregate=integrity["aggregate_sha256"],
        packed_row_fingerprint=packed_info.packed_row_fingerprint,
        max_features_per_segment=3,
        cluster_delta_semantics=cluster_delta_semantics,
    )
    metadata[REMOTE_LOADER_KEY] = loader
    (selection_dir / "manifest.json").write_text(
        json.dumps(
            {
                "checksum": selection_hash,
                "metadata": metadata,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return selection_dir


def _copy_remote_file(
    fixture_root: Path,
    bucket: FakeBucket,
    path: Path,
) -> None:
    remote_path = "features/" + path.relative_to(fixture_root).as_posix()
    target = bucket._path(remote_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, target)


def _upload_remote_contract(
    fixture_root: Path,
    bucket: FakeBucket,
    train_packed_dir: Path,
    selection_dirs: list[Path],
) -> None:
    for name in (
        "manifest.json",
        REMOTE_PACKED_INVENTORY_FILENAME,
        *[path.name for path in sorted(train_packed_dir.glob("*.safetensors.zst"))],
    ):
        _copy_remote_file(
            fixture_root,
            bucket,
            train_packed_dir / name,
        )
    for selection_dir in selection_dirs:
        _copy_remote_file(
            fixture_root,
            bucket,
            selection_dir / "manifest.json",
        )
        for path in sorted(selection_dir.glob("*.safetensors.zst")):
            _copy_remote_file(fixture_root, bucket, path)


def _build_discovery_fixture(tmp_path: Path) -> _DiscoveryFixture:
    producer_root = tmp_path / "producer"
    derived_dir = producer_root / "derived"
    base_dir = producer_root / "base"
    _write_derived_lineage(derived_dir)
    _write_stats(derived_dir)
    _write_stats(base_dir)
    cluster_path = tmp_path / "clusters.json"
    cluster_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "clusters": [[1, 2], [3, 4]],
            }
        ),
        encoding="utf-8",
    )
    resolved_clusters = resolve_feature_clusters(
        {"feature_clusters": {"path": str(cluster_path)}},
        _N_SAE_FEATURES,
    )
    assert resolved_clusters is not None
    clusters = FeatureClusterConsolidation.build(
        resolved_clusters,
        None,
    )
    cluster_descriptor = canonical_feature_clusters_descriptor(resolved_clusters)
    train_selection_hash = compute_selection_hash(
        positive_selection=_FEATURE_SELECTION,
        negative_selection=None,
        delta_config=_FEATURE_DELTA,
        num_buckets=_NUM_BUCKETS,
        seed=_SEED,
        ddp_rank=0,
        feature_mask=None,
        feature_clusters=cluster_descriptor,
        splice_batch_size=None,
    )
    base_feature_selection: dict[str, object] = dict(_FEATURE_SELECTION)
    base_feature_selection["delta_weights"] = None
    base_selection_hash = compute_selection_hash(
        positive_selection=base_feature_selection,
        negative_selection=None,
        delta_config=None,
        num_buckets=_NUM_BUCKETS,
        seed=_SEED,
        ddp_rank=0,
        feature_mask=None,
        feature_clusters=cluster_descriptor,
        splice_batch_size=None,
    )

    train_packed_dir = _packed_train_dir(derived_dir)
    derived_val_packed_dir = train_packed_dir.parent / "val_sall"
    base_train_packed_dir = _packed_train_dir(base_dir)
    base_val_packed_dir = base_train_packed_dir.parent / "val_sall"
    _write_packed_family(
        train_packed_dir,
        n_shards=2,
        seed_offset=10,
    )
    _write_packed_family(
        derived_val_packed_dir,
        n_shards=1,
        seed_offset=20,
    )
    _write_packed_family(
        base_val_packed_dir,
        n_shards=1,
        seed_offset=30,
    )
    train_selection_dir = _write_selection_family(
        packed_dir=train_packed_dir,
        selection_hash=train_selection_hash,
        clusters=clusters,
        derived_dir=derived_dir,
    )
    derived_val_selection_dir = _write_selection_family(
        packed_dir=derived_val_packed_dir,
        selection_hash=train_selection_hash,
        clusters=clusters,
        derived_dir=derived_dir,
    )
    base_val_selection_dir = _write_selection_family(
        packed_dir=base_val_packed_dir,
        selection_hash=base_selection_hash,
        clusters=clusters,
        derived_dir=None,
    )
    publish_remote_packed_inventory(train_packed_dir)
    bucket = FakeBucket(tmp_path / "bucket")
    _upload_remote_contract(
        producer_root,
        bucket,
        train_packed_dir,
        [
            train_selection_dir,
            derived_val_selection_dir,
            base_val_selection_dir,
        ],
    )
    return _DiscoveryFixture(
        producer_root=producer_root,
        derived_dir=derived_dir,
        base_dir=base_dir,
        cluster_path=cluster_path,
        train_packed_dir=train_packed_dir,
        derived_val_packed_dir=derived_val_packed_dir,
        base_val_packed_dir=base_val_packed_dir,
        train_selection_hash=train_selection_hash,
        base_selection_hash=base_selection_hash,
        derived_feature_fingerprint=(
            derived_feature_fingerprint(
                derived_dir,
                expected_stems=[
                    "source_00000",
                    "source_00001",
                ],
            )
        ),
        bucket=bucket,
    )


def _remote_config(
    *,
    stream_train_packed: bool,
    timeout: float = 0.05,
) -> RemoteTrainingConfig:
    raw: dict[str, Any] = {
        "bucket_id": "owner/bucket",
        "body_prefetch": {"lookahead_shards_per_rank": 2},
        "selection_prefix": "features",
        "checkpoint_prefix": ("checkpoints/conditional_encdec"),
        "selection_wait_timeout_seconds": timeout,
        "checkpoint_retry_timeout_seconds": timeout,
        "retry_interval_seconds": 0.001,
        "resume_latest": True,
    }
    if stream_train_packed:
        raw["train_packed"] = {}
    parsed = RemoteTrainingConfig.parse(raw)
    assert parsed is not None
    return parsed


def _stage_all_local(
    fixture: _DiscoveryFixture,
    root: Path,
) -> tuple[Path, Path]:
    shutil.copytree(fixture.producer_root, root)
    return root / "derived", root / "base"


def _stage_streaming(
    fixture: _DiscoveryFixture,
    root: Path,
) -> tuple[Path, Path]:
    derived_dir = root / "derived"
    base_dir = root / "base"
    derived_dir.mkdir(parents=True)
    base_dir.mkdir(parents=True)
    shutil.copyfile(
        fixture.derived_dir / "feature_statistics.safetensors",
        derived_dir / "feature_statistics.safetensors",
    )
    shutil.copytree(
        fixture.derived_dir / "_splice_provenance",
        derived_dir / "_splice_provenance",
    )
    shutil.copyfile(
        fixture.base_dir / "feature_statistics.safetensors",
        base_dir / "feature_statistics.safetensors",
    )

    local_train = root / fixture.train_packed_dir.relative_to(fixture.producer_root)
    local_train.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        provenance_dir(fixture.train_packed_dir),
        provenance_dir(local_train),
    )
    local_derived_val = root / fixture.derived_val_packed_dir.relative_to(fixture.producer_root)
    shutil.copytree(
        fixture.derived_val_packed_dir,
        local_derived_val,
    )
    shutil.rmtree(local_derived_val / "sel")
    local_base_val = root / fixture.base_val_packed_dir.relative_to(fixture.producer_root)
    shutil.copytree(
        fixture.base_val_packed_dir,
        local_base_val,
    )
    shutil.rmtree(local_base_val / "sel")
    return derived_dir, base_dir


def _discover(
    fixture: _DiscoveryFixture,
    *,
    feature_root: Path,
    derived_dir: Path,
    base_dir: Path,
    remote_training: RemoteTrainingConfig | None,
) -> PackedSelectionDiscovery:
    return discover_packed_selection(
        feature_dir=derived_dir,
        original_val_feature_dir=base_dir,
        feature_selection=_FEATURE_SELECTION,
        negative_selection=None,
        feature_delta=_FEATURE_DELTA,
        num_buckets=_NUM_BUCKETS,
        seed=_SEED,
        feature_mask=None,
        feature_clusters={"path": str(fixture.cluster_path)},
        n_sae_features=_N_SAE_FEATURES,
        use_precomputed_batches=True,
        percentile_checksum=_PERCENTILE_CHECKSUM,
        tokenizer_checksum=_TOKENIZER_CHECKSUM,
        context_len=_CONTEXT_LEN,
        segment_size=_SEGMENT_SIZE,
        max_train_shards=None,
        compile_static=False,
        fp8=False,
        eval_every=1,
        original_val_data_dir=None,
        boundary_fingerprint=_BOUNDARY_FINGERPRINT,
        selection_splice_batch_size=None,
        remote_training=remote_training,
        feature_root=(feature_root if remote_training is not None else None),
        remote_backend=(fixture.bucket if remote_training is not None else None),
    )


def _normalize_path(value: Any, root: Path) -> Any:
    if isinstance(value, Path):
        return value.relative_to(root).as_posix()
    if isinstance(value, dict):
        return {key: _normalize_path(item, root) for key, item in value.items()}
    return value


def _semantic_discovery(
    result: PackedSelectionDiscovery,
    *,
    root: Path,
) -> dict[str, Any]:
    excluded = {
        "packed_identity_materializer",
        "packed_materializer",
        "remote_derived_feature_fingerprint",
        "remote_materializers",
    }
    return {
        field.name: _normalize_path(
            getattr(result, field.name),
            root,
        )
        for field in fields(PackedSelectionDiscovery)
        if field.name not in excluded
    }


def _selection_contracts(
    result: PackedSelectionDiscovery,
) -> dict[str, dict[str, Any]]:
    paths = {
        "train": result.train_selection_dir,
        "derived_val": result.val_selection_dir,
        "base_val": result.original_val_selection_dir,
    }
    contracts: dict[str, dict[str, Any]] = {}
    for label, path in paths.items():
        assert path is not None
        materializer = result.remote_materializers[str(path.resolve())]
        contracts[label] = materializer.identity_contract
    return contracts


def test_discovery_streaming_composes_and_preserves_every_semantic_field(
    tmp_path: Path,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    local_root = tmp_path / "all-local"
    local_derived, local_base = _stage_all_local(
        fixture,
        local_root,
    )
    pure_local = _discover(
        fixture,
        feature_root=local_root,
        derived_dir=local_derived,
        base_dir=local_base,
        remote_training=None,
    )
    selection_remote = _discover(
        fixture,
        feature_root=local_root,
        derived_dir=local_derived,
        base_dir=local_base,
        remote_training=_remote_config(stream_train_packed=False),
    )

    streaming_root = tmp_path / "streaming"
    streaming_derived, streaming_base = _stage_streaming(
        fixture,
        streaming_root,
    )
    streaming = _discover(
        fixture,
        feature_root=streaming_root,
        derived_dir=streaming_derived,
        base_dir=streaming_base,
        remote_training=_remote_config(stream_train_packed=True),
    )

    assert _semantic_discovery(
        pure_local,
        root=local_root,
    ) == _semantic_discovery(
        streaming,
        root=streaming_root,
    )
    assert _selection_contracts(selection_remote) == (_selection_contracts(streaming))
    assert isinstance(
        streaming.packed_materializer,
        RemotePackedRowMaterializer,
    )
    assert streaming.remote_derived_feature_fingerprint == (fixture.derived_feature_fingerprint)
    assert streaming.train_packed_dir is not None
    assert not list(streaming.train_packed_dir.glob("*.safetensors.zst"))
    assert not list(streaming_derived.glob("*_features.safetensors.zst"))
    assert len(streaming.remote_materializers) == 3

    local_contracts = _selection_contracts(selection_remote)
    streaming_contracts = _selection_contracts(streaming)
    local_fingerprint = build_remote_run_fingerprint(
        resolved_training_config={
            "max_seq_len": 1024,
            "remote_training": {"train_packed": None},
        },
        world_size=2,
        selection_contracts=local_contracts,
    )
    streaming_fingerprint = build_remote_run_fingerprint(
        resolved_training_config={
            "max_seq_len": 1024,
            "remote_training": {"train_packed": {}},
        },
        world_size=2,
        selection_contracts=streaming_contracts,
    )
    assert streaming_fingerprint == local_fingerprint


def test_selection_remote_uses_signed_inventory_with_partial_local_packed_cache(
    tmp_path: Path,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    complete_root = tmp_path / "complete"
    complete_derived, complete_base = _stage_all_local(
        fixture,
        complete_root,
    )
    complete = _discover(
        fixture,
        feature_root=complete_root,
        derived_dir=complete_derived,
        base_dir=complete_base,
        remote_training=_remote_config(stream_train_packed=False),
    )

    partial_root = tmp_path / "partial"
    partial_derived, partial_base = _stage_all_local(
        fixture,
        partial_root,
    )
    partial_train = partial_root / fixture.train_packed_dir.relative_to(fixture.producer_root)
    packed_paths = sorted(partial_train.glob("*.safetensors.zst"))
    assert len(packed_paths) == 2
    packed_paths[-1].unlink()

    partial = _discover(
        fixture,
        feature_root=partial_root,
        derived_dir=partial_derived,
        base_dir=partial_base,
        remote_training=_remote_config(stream_train_packed=False),
    )

    assert _semantic_discovery(
        complete,
        root=complete_root,
    ) == _semantic_discovery(
        partial,
        root=partial_root,
    )
    assert _selection_contracts(complete) == _selection_contracts(partial)
    assert partial.packed_materializer is None
    assert isinstance(
        partial.packed_identity_materializer,
        RemotePackedRowMaterializer,
    )
    assert partial.remote_derived_feature_fingerprint == (fixture.derived_feature_fingerprint)
    assert len(list(partial_train.glob("*.safetensors.zst"))) == 1


@pytest.mark.parametrize(
    ("missing", "match"),
    [
        ("remote_manifest", r"timed out.*sall/manifest\.json"),
        ("remote_inventory", r"timed out.*remote_packed_inventory\.json"),
        ("provenance", "pre-staged packed provenance"),
        ("sidecars", "splice provenance dir"),
        ("derived_val_bodies", "derived/spliced val packed rows"),
        ("base_val_family", "original/base val packed rows"),
        ("derived_stats", "feature statistics"),
        ("base_selection_manifest", r"timed out.*sel/.*manifest\.json"),
    ],
)
def test_discovery_missing_required_artifact_fails_early(
    tmp_path: Path,
    missing: str,
    match: str,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    streaming_root = tmp_path / "streaming"
    derived_dir, base_dir = _stage_streaming(
        fixture,
        streaming_root,
    )
    local_train = streaming_root / fixture.train_packed_dir.relative_to(fixture.producer_root)
    local_derived_val = streaming_root / fixture.derived_val_packed_dir.relative_to(
        fixture.producer_root
    )
    local_base_val = streaming_root / fixture.base_val_packed_dir.relative_to(fixture.producer_root)
    train_remote = (
        "features/" + fixture.train_packed_dir.relative_to(fixture.producer_root).as_posix()
    )
    base_selection_remote = (
        "features/"
        + (fixture.base_val_packed_dir / "sel" / fixture.base_selection_hash / "manifest.json")
        .relative_to(fixture.producer_root)
        .as_posix()
    )
    if missing == "remote_manifest":
        fixture.bucket._path(f"{train_remote}/manifest.json").unlink()
    elif missing == "remote_inventory":
        fixture.bucket._path(f"{train_remote}/{REMOTE_PACKED_INVENTORY_FILENAME}").unlink()
    elif missing == "provenance":
        shutil.rmtree(provenance_dir(local_train))
    elif missing == "sidecars":
        shutil.rmtree(derived_dir / "_splice_provenance")
    elif missing == "derived_val_bodies":
        for path in local_derived_val.glob("*.safetensors.zst"):
            path.unlink()
    elif missing == "base_val_family":
        shutil.rmtree(local_base_val)
    elif missing == "derived_stats":
        (derived_dir / "feature_statistics.safetensors").unlink()
    elif missing == "base_selection_manifest":
        fixture.bucket._path(base_selection_remote).unlink()
    else:
        raise AssertionError(missing)

    with pytest.raises(
        (
            FileNotFoundError,
            RemoteTrainingError,
            SpliceProvenanceError,
            ValueError,
        ),
        match=match,
    ):
        _discover(
            fixture,
            feature_root=streaming_root,
            derived_dir=derived_dir,
            base_dir=base_dir,
            remote_training=_remote_config(
                stream_train_packed=True,
                timeout=0.01,
            ),
        )


def test_discovery_streaming_validates_packed_provenance_contract(
    tmp_path: Path,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    remote_manifest = fixture.bucket._path(
        "features/"
        + (fixture.train_packed_dir / "sel" / fixture.train_selection_hash / "manifest.json")
        .relative_to(fixture.producer_root)
        .as_posix()
    )
    manifest = json.loads(remote_manifest.read_text(encoding="utf-8"))
    manifest["metadata"]["packed_provenance"]["provenance_fingerprint"] = "0" * 64
    remote_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    streaming_root = tmp_path / "streaming"
    derived_dir, base_dir = _stage_streaming(
        fixture,
        streaming_root,
    )

    with pytest.raises(
        ValueError,
        match="packed_provenance mismatch",
    ):
        _discover(
            fixture,
            feature_root=streaming_root,
            derived_dir=derived_dir,
            base_dir=base_dir,
            remote_training=_remote_config(stream_train_packed=True),
        )


def test_discovery_rejects_inventory_vs_remote_loader_packed_fingerprint_drift(
    tmp_path: Path,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    remote_manifest = fixture.bucket._path(
        "features/"
        + (fixture.train_packed_dir / "sel" / fixture.train_selection_hash / "manifest.json")
        .relative_to(fixture.producer_root)
        .as_posix()
    )
    manifest = json.loads(remote_manifest.read_text(encoding="utf-8"))
    loader = manifest["metadata"][REMOTE_LOADER_KEY]
    manifest["metadata"][REMOTE_LOADER_KEY] = build_remote_loader_certificate(
        selection_checksum=loader["selection_checksum"],
        selection_inventory_aggregate=loader["selection_inventory_aggregate"],
        packed_row_fingerprint="0" * 64,
        max_features_per_segment=loader["max_features_per_segment"],
        cluster_delta_semantics=loader[CLUSTER_DELTA_SEMANTICS_KEY],
    )
    remote_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    streaming_root = tmp_path / "streaming"
    derived_dir, base_dir = _stage_streaming(
        fixture,
        streaming_root,
    )

    with pytest.raises(
        RemoteTrainingError,
        match="packed-row fingerprint drift",
    ):
        _discover(
            fixture,
            feature_root=streaming_root,
            derived_dir=derived_dir,
            base_dir=base_dir,
            remote_training=_remote_config(stream_train_packed=True),
        )


def test_discovery_rejects_cross_family_derived_fingerprint_drift(
    tmp_path: Path,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    remote_manifest = fixture.bucket._path(
        "features/"
        + (fixture.derived_val_packed_dir / "sel" / fixture.train_selection_hash / "manifest.json")
        .relative_to(fixture.producer_root)
        .as_posix()
    )
    manifest = json.loads(remote_manifest.read_text(encoding="utf-8"))
    manifest["metadata"]["splice_counterfactual"]["derived_feature_fingerprint"] = "d" * 64
    remote_manifest.write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    streaming_root = tmp_path / "streaming"
    derived_dir, base_dir = _stage_streaming(
        fixture,
        streaming_root,
    )

    with pytest.raises(
        RemoteTrainingError,
        match="disagree.*derived_feature_fingerprint",
    ):
        _discover(
            fixture,
            feature_root=streaming_root,
            derived_dir=derived_dir,
            base_dir=base_dir,
            remote_training=_remote_config(stream_train_packed=True),
        )


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        ("byte_length", "shard inventory"),
        ("packed_fingerprint", "fingerprint.*stale"),
    ],
)
def test_discovery_remote_fp_certificate_variant_is_exact(
    tmp_path: Path,
    mutation: str,
    match: str,
) -> None:
    fixture = _build_discovery_fixture(tmp_path)
    streaming_root = tmp_path / "streaming"
    derived_dir, base_dir = _stage_streaming(
        fixture,
        streaming_root,
    )
    result = _discover(
        fixture,
        feature_root=streaming_root,
        derived_dir=derived_dir,
        base_dir=base_dir,
        remote_training=_remote_config(stream_train_packed=True),
    )
    assert result.train_packed_dir is not None
    assert result.packed_materializer is not None
    inventory = copy.deepcopy(result.packed_materializer.inventory)
    validate_fp_target_source_certificate_remote(
        result.train_packed_dir,
        n_sae_features=_N_SAE_FEATURES,
        inventory=inventory,
    )
    if mutation == "byte_length":
        inventory["shards"][0]["byte_length"] += 1
    else:
        inventory["packed_row_fingerprint"] = "0" * 64

    with pytest.raises(PackedRowInvariantError, match=match):
        validate_fp_target_source_certificate_remote(
            result.train_packed_dir,
            n_sae_features=_N_SAE_FEATURES,
            inventory=inventory,
        )
