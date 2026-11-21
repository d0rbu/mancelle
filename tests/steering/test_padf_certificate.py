"""Selection-metadata authority for precomputed-loader feature padding."""

from __future__ import annotations

import contextlib
import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch as t
from loguru import logger

from errors import FeatureClustersError
from scripts.preprocess.precompute_selected_features import (
    _selection_max_features_per_segment,
)
from steering.conditional.clustered_selection_validation import (
    CLUSTERED_SELECTION_VALIDATION_KEY,
    build_clustered_selection_validation,
)
from steering.conditional.encoder_decoder import dataloader as dataloader_module
from steering.conditional.encoder_decoder.dataloader import (
    _max_selected_features_per_segment,
    _validate_generic_selection_dir,
    _validate_splice_counterfactual_selection_dir,
    precomputed_data_loader,
)
from steering.conditional.feature_clusters import (
    SELECTION_SHARD_INTEGRITY_KEY,
    FeatureClusterConsolidation,
    build_clustered_selection_shard_integrity,
)
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.splice_counterfactual import (
    resolve_packed_row_fingerprint_strict,
    validate_packed_provenance,
)
from steering.conditional.splice_selection_certificate import (
    SPLICE_RESUME_METADATA_KEY,
    SpliceSelectionStats,
    build_splice_resume_metadata,
)
from steering.overlay_io import OverlayKind, glob_overlay_shards, read_manifest, write_manifest
from steering.packed_batch_io import (
    SelectionShard,
    load_selection_shard,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.remote_training import RemoteSelectionMaterializer
from tests.steering.test_feature_clusters_precomputed_loader import (
    N_SAE,
    _base_shard,
    _cluster_state,
    _pipeline,
    _selection_shard,
    _streaming_loader,
    _streaming_materializers,
    _streaming_splice_pipeline,
    _write_streaming_splice_fixture,
)
from tests.steering.test_precomputed_shard_lookahead import (
    _assert_batches_equal,
    _assert_streams_equal,
)


@dataclass(frozen=True)
class _GenericFixture:
    packed_dir: Path
    selection_dir: Path
    clusters: FeatureClusterConsolidation
    pipeline: FeaturePipelineConfig
    certificate: dict[str, Any]
    selection_paths: tuple[Path, ...]


def _wide_selection_shard() -> SelectionShard:
    """Two valid clustered segments with a global width of seven."""
    return SelectionShard(
        sel_offsets=t.tensor([0, 4, 5], dtype=t.int32),
        sel_fids=t.tensor([3, 7, 10, 11, 15], dtype=t.uint16),
        sel_buckets=t.tensor([3, 2, 1, 0, 3], dtype=t.uint8),
        sel_pcts=t.tensor([240, 220, 180, 140, 100], dtype=t.uint8),
        neg_offsets=t.tensor([0, 3, 4], dtype=t.int32),
        neg_fids=t.tensor([12, 13, 14, 16], dtype=t.uint16),
    )


def _zero_width_selection_shard() -> SelectionShard:
    return SelectionShard(
        sel_offsets=t.tensor([0, 0, 0], dtype=t.int32),
        sel_fids=t.empty(0, dtype=t.uint16),
        sel_buckets=t.empty(0, dtype=t.uint8),
        sel_pcts=t.empty(0, dtype=t.uint8),
        neg_offsets=t.tensor([0, 0, 0], dtype=t.int32),
        neg_fids=t.empty(0, dtype=t.uint16),
    )


def _write_generic_fixture(
    root: Path,
    *,
    selection_dir_name: str = "a" * 64,
    n_shards: int = 4,
    zero_width: bool = False,
) -> _GenericFixture:
    packed_dir = root / "packed"
    selection_dir = packed_dir / "sel" / selection_dir_name
    packed_dir.mkdir(parents=True)
    selection_dir.mkdir(parents=True)
    clusters = _cluster_state(root / "cluster-input")
    pipeline = _pipeline(clusters.feature_mask)

    packed_paths: list[Path] = []
    selection_paths: list[Path] = []
    segment_counts: dict[str, int] = {}
    max_features: dict[str, int] = {}
    for index in range(n_shards):
        stem = f"shard_{index:05d}"
        packed_paths.append(
            save_packed_row_shard(
                _base_shard(token_offset=3 * index),
                packed_dir / stem,
            )
        )
        selection = (
            _zero_width_selection_shard()
            if zero_width
            else _wide_selection_shard()
            if index == 1
            else _selection_shard()
        )
        selection_paths.append(save_selection_shard(selection, selection_dir / stem))
        segment_counts[stem] = selection.total_segments
        max_features[stem] = _max_selected_features_per_segment(
            selection.sel_offsets,
            selection.neg_offsets,
        )

    write_manifest(
        packed_dir,
        "packed-padf-test",
        OverlayKind.PACKED_ROW,
        metadata={"max_n_seg": 1, "max_seg_size": 8},
        base_shard_paths=packed_paths,
    )
    integrity = build_clustered_selection_shard_integrity(selection_paths)
    packed_proof = resolve_packed_row_fingerprint_strict(packed_dir)
    certificate = build_clustered_selection_validation(
        selection_checksum="a" * 64,
        selection_integrity=integrity,
        packed_row_fingerprint=packed_proof.aggregate_sha256,
        packed_content_by_name={
            entry.name.removesuffix(".safetensors.zst"): entry.content_sha256
            for entry in packed_proof.files
        },
        feature_clusters_descriptor=clusters.resolved.descriptor_dict(),
        feature_delta=None,
        effective_buckets=pipeline.effective_buckets,
        segment_counts_by_stem=segment_counts,
        max_features_by_stem=max_features,
    )
    cluster_metadata = clusters.metadata_dict()
    cluster_metadata["status"] = "complete"
    write_manifest(
        selection_dir,
        "a" * 64,
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_proof.aggregate_sha256,
            },
            "feature_clusters": cluster_metadata,
            SELECTION_SHARD_INTEGRITY_KEY: integrity,
            CLUSTERED_SELECTION_VALIDATION_KEY: certificate,
        },
        base_shard_paths=packed_paths,
        cache_dir=packed_dir,
    )
    return _GenericFixture(
        packed_dir=packed_dir,
        selection_dir=selection_dir,
        clusters=clusters,
        pipeline=pipeline,
        certificate=certificate,
        selection_paths=tuple(selection_paths),
    )


def _manifest_metadata(selection_dir: Path) -> dict[str, Any]:
    manifest = read_manifest(selection_dir)
    assert manifest is not None
    metadata = manifest.get("metadata")
    assert isinstance(metadata, dict)
    return metadata


def _write_manifest_metadata(selection_dir: Path, metadata: dict[str, Any]) -> None:
    manifest_path = selection_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["metadata"] = metadata
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


def _set_generic_certificate(fixture: _GenericFixture, *, present: bool) -> None:
    metadata = _manifest_metadata(fixture.selection_dir)
    if present:
        metadata[CLUSTERED_SELECTION_VALIDATION_KEY] = copy.deepcopy(fixture.certificate)
    else:
        metadata.pop(CLUSTERED_SELECTION_VALIDATION_KEY, None)
    _write_manifest_metadata(fixture.selection_dir, metadata)


def _add_splice_certificate(fixture: Any) -> dict[str, Any]:
    metadata = _manifest_metadata(fixture.selection_dir)
    selection_paths = tuple(glob_overlay_shards(fixture.selection_dir))
    integrity = build_clustered_selection_shard_integrity(list(selection_paths))
    packed_info = validate_packed_provenance(
        fixture.packed_dir,
        cache_dir=fixture.packed_dir,
    )
    stats_by_stem: dict[str, SpliceSelectionStats] = {}
    storage_validated: dict[str, str] = {}
    widths: list[int] = []
    integrity_by_name = {
        entry["name"]: entry["content_sha256"] for entry in integrity["shards"]
    }
    for path in selection_paths:
        selection = load_selection_shard(path)
        stem = path.name.removesuffix(".safetensors.zst")
        stats_by_stem[stem] = SpliceSelectionStats(
            total_packed_segments=selection.total_segments,
            total_sidecar_splice_positions_referenced=0,
            total_delta_positions_selected=0,
        )
        storage_validated[stem] = integrity_by_name[path.name]
        widths.append(
            _max_selected_features_per_segment(
                selection.sel_offsets,
                selection.neg_offsets,
            )
        )
    resume = build_splice_resume_metadata(
        stats_by_stem=stats_by_stem,
        selection_integrity=integrity,
        storage_validated_by_stem=storage_validated,
        validated_max_features_per_segment=max(widths),
        packed_rows_hash=packed_info.packed_row_fingerprint,
        effective_buckets=_streaming_splice_pipeline(
            fixture.clusters
        ).effective_buckets,
        status="complete",
    )
    metadata[SELECTION_SHARD_INTEGRITY_KEY] = integrity
    metadata[SPLICE_RESUME_METADATA_KEY] = resume
    splice = metadata["splice_counterfactual"]
    splice["total_packed_segments_selected"] = sum(
        stats.total_packed_segments for stats in stats_by_stem.values()
    )
    splice["total_sidecar_splice_positions_referenced"] = 0
    splice["total_delta_positions_selected"] = 0
    _write_manifest_metadata(fixture.selection_dir, metadata)
    return resume


def _set_splice_certificate(fixture: Any, resume: dict[str, Any] | None) -> None:
    metadata = _manifest_metadata(fixture.selection_dir)
    if resume is None:
        metadata.pop(SPLICE_RESUME_METADATA_KEY, None)
    else:
        metadata[SPLICE_RESUME_METADATA_KEY] = copy.deepcopy(resume)
    _write_manifest_metadata(fixture.selection_dir, metadata)


def _widen_middle_splice_selection(fixture: Any) -> None:
    selection_paths = tuple(glob_overlay_shards(fixture.selection_dir))
    assert len(selection_paths) >= 3
    save_selection_shard(_wide_selection_shard(), selection_paths[1])
    metadata = _manifest_metadata(fixture.selection_dir)
    metadata[SELECTION_SHARD_INTEGRITY_KEY] = (
        build_clustered_selection_shard_integrity(list(selection_paths))
    )
    _write_manifest_metadata(fixture.selection_dir, metadata)


def _selection_paths_by_stem(selection_dir: Path) -> dict[str, Path]:
    return {
        path.name.removesuffix(".safetensors.zst"): path
        for path in glob_overlay_shards(selection_dir)
    }


def _scan_width(paths: tuple[Path, ...] | list[Path]) -> int:
    width = 0
    for path in paths:
        selection = load_selection_shard(path)
        width = max(
            width,
            _max_selected_features_per_segment(
                selection.sel_offsets,
                selection.neg_offsets,
            ),
        )
    return width


def _validate_generic(fixture: _GenericFixture) -> int | None:
    return _validate_generic_selection_dir(
        packed_row_dir=fixture.packed_dir,
        shard_paths=glob_overlay_shards(fixture.packed_dir),
        selection_dir=fixture.selection_dir,
        sel_shard_paths=_selection_paths_by_stem(fixture.selection_dir),
        feature_pipeline=fixture.pipeline,
        feature_clusters=fixture.clusters,
    )


def _validate_splice(fixture: Any) -> int | None:
    return _validate_splice_counterfactual_selection_dir(
        packed_row_dir=fixture.packed_dir,
        shard_paths=glob_overlay_shards(fixture.packed_dir),
        selection_dir=fixture.selection_dir,
        sel_shard_paths=_selection_paths_by_stem(fixture.selection_dir),
        feature_pipeline=_streaming_splice_pipeline(fixture.clusters),
        splice_feature_dir=fixture.splice_feature_dir,
        feature_clusters=fixture.clusters,
    )


def _generic_loader(
    fixture: _GenericFixture,
    *,
    resume_state_dict: dict[str, Any] | None = None,
    batch_size: int = 2,
    ddp_rank: int = 1,
    ddp_world_size: int = 2,
):
    return precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=batch_size,
        feature_pipeline=fixture.pipeline,
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=fixture.pipeline.effective_buckets,
        n_sae_features=N_SAE,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        resume_state_dict=resume_state_dict,
        feature_clusters=fixture.clusters,
    )


def _splice_loader(
    fixture: Any,
    *,
    resume_state_dict: dict[str, Any] | None = None,
    batch_size: int = 2,
    ddp_rank: int = 1,
    ddp_world_size: int = 2,
):
    return precomputed_data_loader(
        packed_row_dir=fixture.packed_dir,
        B=batch_size,
        feature_pipeline=_streaming_splice_pipeline(fixture.clusters),
        device="cpu",
        selection_dir=fixture.selection_dir,
        effective_buckets=4,
        n_sae_features=N_SAE,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        resume_state_dict=resume_state_dict,
        splice_feature_dir=fixture.splice_feature_dir,
        feature_clusters=fixture.clusters,
    )


def _drain(loader: Any, count: int) -> list[tuple[Any, ...]]:
    with contextlib.closing(loader):
        return [next(loader) for _ in range(count)]


def _selection_with_offsets(
    sel_offsets: list[int],
    neg_offsets: list[int],
) -> SelectionShard:
    n_positive = sel_offsets[-1]
    n_negative = neg_offsets[-1]
    return SelectionShard(
        sel_offsets=t.tensor(sel_offsets, dtype=t.int32),
        sel_fids=t.arange(n_positive, dtype=t.int64).to(t.uint16),
        sel_buckets=t.zeros(n_positive, dtype=t.uint8),
        sel_pcts=t.zeros(n_positive, dtype=t.uint8),
        neg_offsets=t.tensor(neg_offsets, dtype=t.int32),
        neg_fids=t.arange(n_negative, dtype=t.int64).to(t.uint16),
    )


@pytest.mark.parametrize(
    ("sel_offsets", "neg_offsets"),
    [
        pytest.param([0, 2], [0, 3], id="single-segment"),
        pytest.param([0, 1, 1, 4, 6], [0, 0, 2, 2, 3], id="many-segments"),
        pytest.param([0, 1, 4], [0, 0, 0], id="empty-negative-csr"),
        pytest.param([0, 0, 0], [0, 2, 5], id="empty-positive-csr"),
        pytest.param([0, 0, 0], [0, 0, 0], id="zero-width"),
        pytest.param([0, 1, 5, 5, 7], [0, 3, 3, 4, 4], id="ragged-widths"),
        pytest.param([0, 5, 6], [0, 2, 2], id="maximum-first"),
        pytest.param([0, 1, 5, 6], [0, 1, 4, 5], id="maximum-middle"),
        pytest.param([0, 1, 2, 6], [0, 1, 2, 5], id="maximum-last"),
    ],
)
def test_producer_and_loader_selection_width_formulas_are_equivalent(
    sel_offsets: list[int],
    neg_offsets: list[int],
) -> None:
    selection = _selection_with_offsets(sel_offsets, neg_offsets)
    assert _selection_max_features_per_segment(selection) == (
        _max_selected_features_per_segment(
            selection.sel_offsets,
            selection.neg_offsets,
        )
    )


def test_generic_certificate_and_body_scan_return_the_exact_width(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    expected = _scan_width(list(fixture.selection_paths))
    assert _validate_generic(fixture) == expected

    _set_generic_certificate(fixture, present=False)
    assert _validate_generic(fixture) == expected

    with contextlib.closing(_generic_loader(fixture)) as loader:
        scan_batch = next(loader)
    _set_generic_certificate(fixture, present=True)
    with contextlib.closing(_generic_loader(fixture)) as loader:
        certificate_batch = next(loader)
    assert scan_batch[9]["pad_F"] == certificate_batch[9]["pad_F"] == max(expected, 1)
    _assert_batches_equal(certificate_batch, scan_batch)


def test_splice_certificate_and_legacy_body_scan_return_the_exact_width(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _widen_middle_splice_selection(fixture)
    _add_splice_certificate(fixture)
    paths = glob_overlay_shards(fixture.selection_dir)
    expected = _scan_width(paths)
    assert _validate_splice(fixture) == expected

    with contextlib.closing(
        _streaming_loader(
            fixture,
            packed_dir=fixture.packed_dir,
            selection_dir=fixture.selection_dir,
            splice_feature_dir=fixture.splice_feature_dir,
            B=2,
        )
    ) as loader:
        certificate_batch = next(loader)
    _set_splice_certificate(fixture, None)
    assert _validate_splice(fixture) == expected
    with contextlib.closing(
        _streaming_loader(
            fixture,
            packed_dir=fixture.packed_dir,
            selection_dir=fixture.selection_dir,
            splice_feature_dir=fixture.splice_feature_dir,
            B=2,
        )
    ) as loader:
        scan_batch = next(loader)
    assert certificate_batch[9]["pad_F"] == scan_batch[9]["pad_F"] == max(expected, 1)
    _assert_batches_equal(certificate_batch, scan_batch)


def test_generic_certificate_is_bitwise_identical_across_epochs_and_resume(
    tmp_path: Path,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    _set_generic_certificate(fixture, present=False)
    scan_stream = _drain(_generic_loader(fixture), 8)
    assert scan_stream[-1][9]["epoch"] >= 3

    _set_generic_certificate(fixture, present=True)
    certificate_stream = _drain(_generic_loader(fixture), 8)
    _assert_streams_equal(certificate_stream, scan_stream)

    scan_state = copy.deepcopy(scan_stream[2][9])
    resumed_with_certificate = _drain(
        _generic_loader(fixture, resume_state_dict=scan_state),
        4,
    )
    _assert_streams_equal(resumed_with_certificate, scan_stream[2:6])

    certificate_state = copy.deepcopy(certificate_stream[2][9])
    _set_generic_certificate(fixture, present=False)
    resumed_with_scan = _drain(
        _generic_loader(fixture, resume_state_dict=certificate_state),
        4,
    )
    _assert_streams_equal(resumed_with_scan, certificate_stream[2:6])


def test_splice_certificate_is_bitwise_identical_across_ddp_epochs(
    tmp_path: Path,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    certificate_stream = _drain(_splice_loader(fixture), 8)
    assert certificate_stream[-1][9]["epoch"] >= 3

    _set_splice_certificate(fixture, None)
    scan_stream = _drain(_splice_loader(fixture), 8)
    assert scan_stream[-1][9]["epoch"] >= 3
    _assert_streams_equal(certificate_stream, scan_stream)


def test_unclustered_generic_body_scan_returns_the_exact_width(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    metadata = _manifest_metadata(fixture.selection_dir)
    metadata.pop("feature_clusters")
    metadata.pop(SELECTION_SHARD_INTEGRITY_KEY)
    metadata.pop(CLUSTERED_SELECTION_VALIDATION_KEY)
    _write_manifest_metadata(fixture.selection_dir, metadata)
    expected = _scan_width(list(fixture.selection_paths))
    assert (
        _validate_generic_selection_dir(
            packed_row_dir=fixture.packed_dir,
            shard_paths=glob_overlay_shards(fixture.packed_dir),
            selection_dir=fixture.selection_dir,
            sel_shard_paths=_selection_paths_by_stem(fixture.selection_dir),
            feature_pipeline=_pipeline(None),
            feature_clusters=None,
        )
        == expected
    )


def test_malformed_generic_certificate_fails_loud(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    metadata = _manifest_metadata(fixture.selection_dir)
    metadata[CLUSTERED_SELECTION_VALIDATION_KEY]["aggregate_sha256"] = "0" * 64
    _write_manifest_metadata(fixture.selection_dir, metadata)
    with pytest.raises(FeatureClustersError, match="clustered validation aggregate mismatch"):
        _validate_generic(fixture)


def test_malformed_splice_certificate_fails_loud(tmp_path: Path) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    metadata = _manifest_metadata(fixture.selection_dir)
    metadata[SPLICE_RESUME_METADATA_KEY]["aggregate_sha256"] = "0" * 64
    _write_manifest_metadata(fixture.selection_dir, metadata)
    with pytest.raises(FeatureClustersError, match="splice resume aggregate mismatch"):
        _validate_splice(fixture)


def test_non_sha_selection_dir_treats_certificate_as_absent_and_scans(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path, selection_dir_name="selection")
    calls = 0
    real_load = dataloader_module.load_selection_shard

    def recording_load(path: Path) -> SelectionShard:
        nonlocal calls
        calls += 1
        return real_load(path)

    monkeypatch.setattr(dataloader_module, "load_selection_shard", recording_load)
    with contextlib.closing(
        _generic_loader(fixture, batch_size=1, ddp_rank=0, ddp_world_size=1)
    ) as loader:
        next(loader)
    assert calls == len(fixture.selection_paths) + 1


def test_selection_load_counter_pins_body_scan_and_certificate_counts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_generic_fixture(tmp_path)
    shard_count = len(fixture.selection_paths)
    calls = 0
    real_load = dataloader_module.load_selection_shard

    def recording_load(path: Path) -> SelectionShard:
        nonlocal calls
        calls += 1
        return real_load(path)

    monkeypatch.setattr(dataloader_module, "load_selection_shard", recording_load)
    _set_generic_certificate(fixture, present=False)
    with contextlib.closing(
        _generic_loader(fixture, batch_size=1, ddp_rank=0, ddp_world_size=1)
    ) as loader:
        next(loader)
    assert calls == shard_count + 1

    calls = 0
    _set_generic_certificate(fixture, present=True)
    with contextlib.closing(
        _generic_loader(fixture, batch_size=1, ddp_rank=0, ddp_world_size=1)
    ) as loader:
        next(loader)
    assert calls == 1


def test_splice_selection_load_counter_pins_certificate_width_handoff(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path)
    _add_splice_certificate(fixture)
    shard_count = len(glob_overlay_shards(fixture.selection_dir))
    calls = 0
    real_load = dataloader_module.load_selection_shard

    def recording_load(path: Path) -> SelectionShard:
        nonlocal calls
        calls += 1
        return real_load(path)

    monkeypatch.setattr(dataloader_module, "load_selection_shard", recording_load)
    _set_splice_certificate(fixture, None)
    with contextlib.closing(
        _splice_loader(
            fixture,
            batch_size=1,
            ddp_rank=0,
            ddp_world_size=1,
        )
    ) as loader:
        next(loader)
    assert calls == shard_count + 1

    calls = 0
    _add_splice_certificate(fixture)
    with contextlib.closing(
        _splice_loader(
            fixture,
            batch_size=1,
            ddp_rank=0,
            ddp_world_size=1,
        )
    ) as loader:
        next(loader)
    assert calls == 1


def test_validated_zero_width_is_clamped_to_one(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path, n_shards=1, zero_width=True)
    messages: list[str] = []
    sink_id = logger.add(messages.append, format="{message}", level="INFO")
    try:
        with contextlib.closing(
            _generic_loader(
                fixture,
                batch_size=1,
                ddp_rank=0,
                ddp_world_size=1,
            )
        ) as loader:
            batch = next(loader)
    finally:
        logger.remove(sink_id)

    assert batch[9]["pad_F"] == 1
    assert any(
        message.strip() == "PrecomputedBatchLoader: max_F=1 from selection validation"
        for message in messages
    )


def test_remote_inventory_width_precedes_splice_certificate_width(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture = _write_streaming_splice_fixture(tmp_path / "fixture")
    resume = _add_splice_certificate(fixture)
    certificate_width = resume["output_validation"][
        "validated_max_features_per_segment"
    ]
    remote_width = certificate_width + 5
    (
        fresh_packed,
        fresh_selection,
        fresh_splice,
        packed_materializer,
        selection_materializer,
    ) = _streaming_materializers(fixture, tmp_path / "consumer")
    monkeypatch.setattr(
        RemoteSelectionMaterializer,
        "max_features_per_segment",
        property(lambda _self: remote_width),
    )

    assert remote_width != certificate_width
    with contextlib.closing(
        _streaming_loader(
            fixture,
            packed_dir=fresh_packed,
            selection_dir=fresh_selection,
            splice_feature_dir=fresh_splice,
            B=2,
            packed_materializer=packed_materializer,
            selection_materializer=selection_materializer,
        )
    ) as loader:
        batch = next(loader)
    assert batch[9]["pad_F"] == remote_width


def test_scan_checkpoint_padf_resumes_under_certificate(tmp_path: Path) -> None:
    fixture = _write_generic_fixture(tmp_path)
    _set_generic_certificate(fixture, present=False)
    with contextlib.closing(_generic_loader(fixture)) as loader:
        scan_batch = next(loader)
    scan_state = copy.deepcopy(scan_batch[9])
    assert scan_state["pad_F"] == _scan_width(list(fixture.selection_paths))

    _set_generic_certificate(fixture, present=True)
    with contextlib.closing(
        _generic_loader(fixture, resume_state_dict=scan_state)
    ) as resumed:
        resumed_batch = next(resumed)
    assert resumed_batch[9]["pad_F"] == scan_state["pad_F"]
