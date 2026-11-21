"""Preselected packed training for splice_counterfactual delta selection."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

import steering.conditional.splice_counterfactual as splice_module
from nanochat.dataset import DatasetSplit, list_parquet_files
from scripts.preprocess.precompute_packed_rows import precompute_packed_rows
from scripts.preprocess.precompute_selected_features import (
    _assert_splice_counterfactual_sel_manifest_current,
    assert_precompute_supports_pipeline,
    select_splice_counterfactual_features_for_shard,
)
from scripts.preprocess.precompute_token_boundaries import (
    _detect_source_bos_char_len,
)
from scripts.preprocess.precompute_token_boundaries import (
    precompute_shard as precompute_token_boundary_shard,
)
from scripts.train.conditional_encdec_train import (
    DualValidationMode,
    _dual_validation_mode,
    _packed_row_inputs_for_pipeline,
)
from steering.conditional.delta import DeltaMode, FeatureDeltaConfig, parse_feature_delta
from steering.conditional.encoder_decoder.dataloader import (
    encoder_decoder_data_loader_with_state,
    precomputed_data_loader,
)
from steering.conditional.feature_clusters import (
    FeatureClusterConsolidation,
    resolve_feature_clusters,
)
from steering.conditional.feature_index import FeatureIndex
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
    build_feature_pipeline_from_cfg_block,
)
from steering.conditional.selection_artifact_validation import (
    validate_complete_selection_dir,
)
from steering.conditional.splice_counterfactual import (
    _PACKED_PROVENANCE_VALIDATION_CACHE,
    _PACKED_ROW_FINGERPRINT_CACHE,
    _PROVENANCE_FINGERPRINT_CACHE,
    canonical_feature_delta_dict,
    derived_feature_fingerprint,
    splice_sidecar_fingerprint,
    tensor_content_fingerprint,
    validate_packed_provenance,
)
from steering.conditional.splice_provenance import (
    SpliceProvenanceError,
    SpliceProvenanceIndex,
)
from steering.feature_io import (
    TensorKey,
    glob_features,
    load_features,
    save_features,
    shard_stem_from_path,
)
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    OverlayKind,
    glob_overlay_shards,
    load_overlay_shard,
    overlay_dir,
    overlay_shard_path,
    save_overlay_shard,
    write_manifest,
)
from steering.packed_batch_io import (
    PackedRowShard,
    compute_selection_hash,
    load_packed_row_shard,
    load_selection_shard,
    save_packed_row_shard,
)
from steering.refined_types import PositiveInt
from steering.segmentation import compute_tokenizer_checksum
from steering.stats_types import PerFeatureStat
from tests.steering._splice_cf_fixtures import build_derived_env


def _stats(n: int = 4096) -> dict[str, PerFeatureStat]:
    percentiles = {f"p{k}": 0.01 * k for k in range(1, 100)}
    return {
        str(i): PerFeatureStat(
            count=10,
            pct_segments_active=0.5,
            mean_mag=1.0,
            mean_mag_incl_zeros=0.5,
            std_mag=0.1,
            min_mag=0.0,
            max_mag=2.0,
            median_mag=1.0,
            percentiles=dict(percentiles),
        )
        for i in range(n)
    }


def _pipeline(*, mode: str = "splice_counterfactual") -> FeaturePipelineConfig:
    delta = parse_feature_delta({"mode": mode, "neg_delta_ratio": 1.0})
    cfg = OmegaConf.create(
        {
            "feature_selection": {
                "threshold_type": "segment_topk",
                "threshold": 999,
            },
            "negative_selection": None,
        }
    )
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=4096,
        num_buckets=4,
        seed=7,
        feature_delta=delta,
        feature_mask=None,
    )


def _sampling_pipeline(mask: FeatureMask) -> FeaturePipelineConfig:
    delta = parse_feature_delta({"mode": "splice_counterfactual", "neg_delta_ratio": 1.0})
    cfg = OmegaConf.create(
        {
            "feature_selection": {
                "distribution": "normal",
                "mean": 2.0,
                "std": 0.01,
                "min_k": 1,
                "max_k": 4,
                "mode": "top_k_percentile",
                "delta_weights": "percentile_delta",
            },
            "negative_selection": None,
        }
    )
    return build_feature_pipeline_from_cfg_block(
        cfg,
        per_feature_stats=_stats(),
        n_sae_features=4096,
        num_buckets=4,
        seed=714,
        feature_delta=delta,
        feature_mask=mask,
    )


def _choose_spliced_doc(feature_dir: Path) -> tuple[FeatureIndex, str, int, int]:
    fi = FeatureIndex(str(feature_dir))
    splice_index = SpliceProvenanceIndex(feature_dir)
    for stem in fi.shard_stems:
        for doc_idx in fi.doc_ids_for_shard(stem):
            if splice_index.doc_splices(stem, doc_idx):
                return fi, stem, doc_idx, fi.shard_stems.index(stem)
    raise AssertionError("fixture produced no splice positions")


def _write_one_doc_packed(
    tmp_path: Path,
    *,
    feature_dir: Path,
    fi: FeatureIndex,
    stem: str,
    doc_idx: int,
    base_idx: int,
) -> tuple[Path, Path]:
    segments = fi.lookup(stem, doc_idx)
    n_seg = len(segments)
    feat_counts = [len(seg.feature_ids) for seg in segments]
    feat_offsets = [0]
    for count in feat_counts:
        feat_offsets.append(feat_offsets[-1] + count)
    packed = PackedRowShard(
        tokens=t.arange(n_seg + 1, dtype=t.int64).to(t.uint16).unsqueeze(0),
        row_seg_offsets=t.tensor([0, n_seg], dtype=t.int32),
        seg_token_start=t.arange(n_seg, dtype=t.int64).to(t.uint16),
        seg_token_end=t.arange(1, n_seg + 1, dtype=t.int64).to(t.uint16),
        seg_doc_id=t.zeros(n_seg, dtype=t.uint8),
        seg_feat_offsets=t.tensor(feat_offsets, dtype=t.int32),
        feature_ids=t.cat([seg.feature_ids.to(t.uint16) for seg in segments]),
        feature_pcts=t.full((feat_offsets[-1],), 128, dtype=t.uint8),
    )
    packed_dir = tmp_path / "packed"
    packed_dir.mkdir()
    shard_path = save_packed_row_shard(packed, packed_dir / "shard_00000")
    prov_dir = provenance_dir(packed_dir)
    prov_dir.mkdir()
    save_overlay_shard(
        {
            PROVENANCE_BASE_SHARD_IDX_KEY: t.full((n_seg,), base_idx, dtype=t.int32),
            PROVENANCE_DOC_IDX_KEY: t.full((n_seg,), doc_idx, dtype=t.int32),
            PROVENANCE_SEG_ORDINAL_KEY: t.arange(n_seg, dtype=t.int32),
        },
        prov_dir / "shard_00000",
    )
    write_manifest(
        packed_dir,
        "packed-test",
        OverlayKind.PACKED_ROW,
        metadata={
            "base_shard_stems": fi.shard_stems,
            "provenance_shards": 1,
            "max_n_seg": n_seg,
            "max_seg_size": 1,
        },
    )
    assert shard_path.is_file()
    assert feature_dir.is_dir()
    return packed_dir, shard_path


def _build_preselected_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, FeaturePipelineConfig, Path]:
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    feature_dir = Path(env["feature_dir"])
    fi, stem, doc_idx, base_idx = _choose_spliced_doc(feature_dir)
    packed_dir, shard_path = _write_one_doc_packed(
        tmp_path,
        feature_dir=feature_dir,
        fi=fi,
        stem=stem,
        doc_idx=doc_idx,
        base_idx=base_idx,
    )
    pipeline = _pipeline()
    packed_info = validate_packed_provenance(packed_dir)
    selection_hash = compute_selection_hash(
        positive_selection={
            "threshold_type": "segment_topk",
            "threshold": 999,
        },
        negative_selection=None,
        delta_config={"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
        num_buckets=4,
        seed=7,
        ddp_rank=0,
        feature_mask=None,
    )
    sel_dir = packed_dir / "sel" / selection_hash
    sel_dir.mkdir(parents=True)
    select_splice_counterfactual_features_for_shard(
        shard_path,
        pipeline,
        sel_dir,
        "shard_00000",
        provenance_path=provenance_dir(packed_dir) / shard_path.name,
        packed_info=packed_info,
        feature_index=fi,
        splice_index=SpliceProvenanceIndex(feature_dir),
        seed=7,
    )
    derived_hash = derived_feature_fingerprint(
        feature_dir,
        expected_stems=packed_info.base_shard_stems,
    )
    boundary_hash = tensor_content_fingerprint(pipeline.boundaries)
    write_manifest(
        sel_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_info.packed_row_fingerprint,
            },
            "feature_delta": canonical_feature_delta_dict(
                FeatureDeltaConfig(DeltaMode.splice_counterfactual)
            ),
            "splice_counterfactual": {
                "schema_version": 1,
                "status": "complete",
                "derived_shard_stems": packed_info.base_shard_stems,
                "total_packed_segments_selected": load_selection_shard(
                    sel_dir / shard_path.name
                ).total_segments,
                "total_sidecar_splice_positions_referenced": 1,
                "derived_feature_fingerprint": derived_hash,
                "boundary_fingerprint": boundary_hash,
                "sidecar_fingerprint": splice_sidecar_fingerprint(feature_dir),
            },
            "packed_provenance": {
                "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
                "packed_row_fingerprint": packed_info.packed_row_fingerprint,
                "provenance_shard_count": len(packed_info.provenance_shard_paths),
                "provenance_fingerprint": packed_info.provenance_fingerprint,
            },
        },
    )
    return packed_dir, sel_dir, pipeline, feature_dir


def _selection_manifest_inputs(
    feature_dir: Path,
    packed_info,
    pipeline: FeaturePipelineConfig,
) -> tuple[str, str, str]:
    return (
        splice_sidecar_fingerprint(feature_dir),
        derived_feature_fingerprint(
            feature_dir,
            expected_stems=packed_info.base_shard_stems,
        ),
        tensor_content_fingerprint(pipeline.boundaries),
    )


def _selection_hash() -> str:
    return compute_selection_hash(
        positive_selection={
            "threshold_type": "segment_topk",
            "threshold": 999,
        },
        negative_selection=None,
        delta_config={"mode": "splice_counterfactual", "neg_delta_ratio": 1.0},
        num_buckets=4,
        seed=7,
        ddp_rank=0,
        feature_mask=None,
    )


class TestPackedProvenanceValidationCache:
    """A warm certificate may skip bodies only while all certified bytes are exact."""

    def _packed_dir(self, tmp_path: Path) -> Path:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        feature_dir = Path(env["feature_dir"])
        fi, stem, doc_idx, base_idx = _choose_spliced_doc(feature_dir)
        packed_dir, _shard_path = _write_one_doc_packed(
            tmp_path,
            feature_dir=feature_dir,
            fi=fi,
            stem=stem,
            doc_idx=doc_idx,
            base_idx=base_idx,
        )
        return packed_dir

    def test_warm_hit_returns_exact_info_without_loading_tensor_bodies(self, tmp_path, monkeypatch):
        packed_dir = self._packed_dir(tmp_path)
        cold = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert (packed_dir / _PACKED_PROVENANCE_VALIDATION_CACHE).is_file()
        assert (packed_dir / _PACKED_ROW_FINGERPRINT_CACHE).is_file()
        assert (packed_dir / _PROVENANCE_FINGERPRINT_CACHE).is_file()

        def forbidden_load(_path):
            raise AssertionError("warm validation hit loaded a tensor body")

        monkeypatch.setattr(splice_module, "load_packed_row_shard", forbidden_load)
        monkeypatch.setattr(splice_module, "load_overlay_shard", forbidden_load)
        warm = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert warm == cold

    def test_mutation_during_certificate_lookup_fails_instead_of_returning_hit(
        self, tmp_path, monkeypatch
    ):
        packed_dir = self._packed_dir(tmp_path)
        info = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        real_load = splice_module._load_packed_provenance_validation_cache

        def mutate_after_cache_read(*args, **kwargs):
            cached = real_load(*args, **kwargs)
            shard_path = info.packed_shard_paths[0]
            shard_path.write_bytes(shard_path.read_bytes() + b"raced-hit")
            return cached

        monkeypatch.setattr(
            splice_module,
            "_load_packed_provenance_validation_cache",
            mutate_after_cache_read,
        )
        with pytest.raises(SpliceProvenanceError, match="validating a cached"):
            validate_packed_provenance(packed_dir, cache_dir=packed_dir)

    def test_packed_replacement_invalidates_and_revalidates(self, tmp_path, monkeypatch):
        packed_dir = self._packed_dir(tmp_path)
        old = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        shard_path = old.packed_shard_paths[0]
        packed = load_packed_row_shard(shard_path)
        packed.tokens[0, 0] = int(packed.tokens[0, 0].item()) + 1
        save_packed_row_shard(packed, shard_path)

        real_load = load_packed_row_shard
        loads = 0

        def recording_load(path):
            nonlocal loads
            loads += 1
            return real_load(path)

        monkeypatch.setattr(splice_module, "load_packed_row_shard", recording_load)
        new = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert loads == 1
        assert new.packed_row_fingerprint != old.packed_row_fingerprint
        assert new.provenance_fingerprint == old.provenance_fingerprint

    def test_provenance_replacement_cannot_reuse_semantic_certificate(self, tmp_path):
        packed_dir = self._packed_dir(tmp_path)
        info = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        provenance_path = info.provenance_shard_paths[0]
        provenance = load_overlay_shard(provenance_path)
        provenance[PROVENANCE_DOC_IDX_KEY][0] = -1
        save_overlay_shard(provenance, provenance_path)

        with pytest.raises(SpliceProvenanceError, match="negative"):
            validate_packed_provenance(packed_dir, cache_dir=packed_dir)

    def test_corrupt_certificate_falls_back_to_full_validation(self, tmp_path, monkeypatch):
        packed_dir = self._packed_dir(tmp_path)
        expected = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        cache_path = packed_dir / _PACKED_PROVENANCE_VALIDATION_CACHE
        cache_path.write_text("{}")
        real_packed_load = load_packed_row_shard
        real_provenance_load = load_overlay_shard
        loads = {"packed": 0, "provenance": 0}

        def recording_packed_load(path):
            loads["packed"] += 1
            return real_packed_load(path)

        def recording_provenance_load(path):
            loads["provenance"] += 1
            return real_provenance_load(path)

        monkeypatch.setattr(splice_module, "load_packed_row_shard", recording_packed_load)
        monkeypatch.setattr(splice_module, "load_overlay_shard", recording_provenance_load)
        actual = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert actual == expected
        assert loads == {"packed": 1, "provenance": 1}
        rewritten = json.loads(cache_path.read_text())
        assert rewritten["schema_version"] == 2

    def test_valid_shaped_certificate_corruption_cannot_define_fingerprints(
        self, tmp_path, monkeypatch
    ):
        packed_dir = self._packed_dir(tmp_path)
        expected = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        cache_path = packed_dir / _PACKED_PROVENANCE_VALIDATION_CACHE
        cache = json.loads(cache_path.read_text())
        assert cache["packed_row_fingerprint"] != "0" * 64
        cache["packed_row_fingerprint"] = "0" * 64
        cache_path.write_text(json.dumps(cache))
        real_load = load_packed_row_shard
        packed_loads = 0

        def recording_load(path):
            nonlocal packed_loads
            packed_loads += 1
            return real_load(path)

        monkeypatch.setattr(splice_module, "load_packed_row_shard", recording_load)
        actual = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert packed_loads == 1
        assert actual == expected
        rewritten = json.loads(cache_path.read_text())
        assert rewritten["packed_row_fingerprint"] == expected.packed_row_fingerprint
        rewritten_aggregate = rewritten.pop("aggregate_sha256")
        assert rewritten_aggregate == splice_module._fingerprint_json(rewritten)

    def test_all_cache_write_failures_preserve_full_validation_result(self, tmp_path, monkeypatch):
        packed_dir = self._packed_dir(tmp_path)
        expected = validate_packed_provenance(packed_dir)

        def fail_write(_payload, _path):
            raise OSError("read-only packed-row directory")

        monkeypatch.setattr(splice_module, "atomic_json_dump", fail_write)
        actual = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert actual == expected
        assert not (packed_dir / _PACKED_PROVENANCE_VALIDATION_CACHE).exists()
        assert not (packed_dir / _PACKED_ROW_FINGERPRINT_CACHE).exists()
        assert not (packed_dir / _PROVENANCE_FINGERPRINT_CACHE).exists()

    def test_manifest_byte_change_invalidates_certificate(self, tmp_path, monkeypatch):
        packed_dir = self._packed_dir(tmp_path)
        expected = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        manifest_path = packed_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metadata"]["test_only_addition"] = "changes exact manifest bytes"
        manifest_path.write_text(json.dumps(manifest))
        real_load = load_packed_row_shard
        loads = 0

        def recording_load(path):
            nonlocal loads
            loads += 1
            return real_load(path)

        monkeypatch.setattr(splice_module, "load_packed_row_shard", recording_load)
        actual = validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert loads == 1
        assert actual == expected

    def test_mutation_during_full_validation_fails_without_certificate(self, tmp_path, monkeypatch):
        packed_dir = self._packed_dir(tmp_path)
        provenance_path = next(provenance_dir(packed_dir).glob("*.safetensors.zst"))
        real_load = load_packed_row_shard

        def mutate_after_packed_load(path):
            packed = real_load(path)
            provenance = load_overlay_shard(provenance_path)
            save_overlay_shard(provenance, provenance_path)
            return packed

        monkeypatch.setattr(splice_module, "load_packed_row_shard", mutate_after_packed_load)
        with pytest.raises(SpliceProvenanceError, match="changed during full"):
            validate_packed_provenance(packed_dir, cache_dir=packed_dir)
        assert not (packed_dir / _PACKED_PROVENANCE_VALIDATION_CACHE).exists()


def _write_splice_selection_manifest(
    *,
    sel_dir: Path,
    selection_hash: str,
    packed_info,
    pipeline: FeaturePipelineConfig,
    feature_dir: Path,
    total_segments: int,
    sidecar_positions: int = 1,
) -> None:
    sidecar_hash, derived_hash, boundary_hash = _selection_manifest_inputs(
        feature_dir, packed_info, pipeline
    )
    write_manifest(
        sel_dir,
        selection_hash,
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_info.packed_row_fingerprint,
            },
            "feature_delta": canonical_feature_delta_dict(
                FeatureDeltaConfig(DeltaMode.splice_counterfactual)
            ),
            "splice_counterfactual": {
                "schema_version": 1,
                "status": "complete",
                "derived_shard_stems": packed_info.base_shard_stems,
                "total_packed_segments_selected": total_segments,
                "total_sidecar_splice_positions_referenced": sidecar_positions,
                "derived_feature_fingerprint": derived_hash,
                "boundary_fingerprint": boundary_hash,
                "sidecar_fingerprint": sidecar_hash,
            },
            "packed_provenance": {
                "packed_stem_fingerprint": packed_info.packed_stem_fingerprint,
                "packed_row_fingerprint": packed_info.packed_row_fingerprint,
                "provenance_shard_count": len(packed_info.provenance_shard_paths),
                "provenance_fingerprint": packed_info.provenance_fingerprint,
            },
        },
    )


def _write_token_boundary_overlay(
    *,
    feature_dir: Path,
    data_dir: str,
    tokenizer,
    segment_size: int,
) -> str:
    checksum = compute_tokenizer_checksum(tokenizer)
    output_dir = overlay_dir(feature_dir, OverlayKind.SEG_BOUNDARY, checksum)
    shard_paths = glob_features(feature_dir)
    assert shard_paths
    write_manifest(
        output_dir,
        checksum,
        OverlayKind.SEG_BOUNDARY,
        base_shard_paths=shard_paths,
    )
    parquet_by_stem = {Path(p).stem: Path(p) for p in list_parquet_files(data_dir)}
    first_stem = shard_stem_from_path(shard_paths[0])
    source_bos_char_len = _detect_source_bos_char_len(
        shard_paths[0],
        parquet_by_stem[first_stem],
    )
    bos_token = tokenizer.get_bos_token_id()
    assert bos_token is not None
    for shard_path in shard_paths:
        stem = shard_stem_from_path(shard_path)
        assert precompute_token_boundary_shard(
            shard_path=shard_path,
            parquet_path=parquet_by_stem[stem],
            tokenizer=tokenizer,
            bos_token=bos_token,
            output_dir=output_dir,
            shard_stem=stem,
            source_bos_char_len=source_bos_char_len,
            source_segment_size=segment_size,
        )
    return checksum


def _write_all_ones_percentile_overlay(feature_dir: Path) -> str:
    checksum = "f" * 64
    output_dir = overlay_dir(feature_dir, OverlayKind.PERCENTILE, checksum)
    shard_paths = glob_features(feature_dir)
    assert shard_paths
    write_manifest(
        output_dir,
        checksum,
        OverlayKind.PERCENTILE,
        base_shard_paths=shard_paths,
    )
    for shard_path in shard_paths:
        stem = shard_stem_from_path(shard_path)
        shard = load_features(shard_path)
        save_overlay_shard(
            {
                TensorKey.FEATURE_PERCENTILES: t.full(
                    (len(shard["feature_ids"]),),
                    255,
                    dtype=t.uint8,
                )
            },
            overlay_shard_path(output_dir, stem),
        )
    return checksum


def _selection_max_f(sel_dir: Path) -> int:
    max_f = 1
    for path in glob_overlay_shards(sel_dir):
        shard = load_selection_shard(path)
        sel_offsets = shard.sel_offsets.tolist()
        neg_offsets = shard.neg_offsets.tolist()
        for seg_idx in range(shard.total_segments):
            n_pos = sel_offsets[seg_idx + 1] - sel_offsets[seg_idx]
            n_neg = neg_offsets[seg_idx + 1] - neg_offsets[seg_idx]
            max_f = max(max_f, n_pos + n_neg)
    return max_f


def _build_real_packed_preselected_fixture(
    tmp_path: Path,
) -> tuple[dict, Path, Path, FeaturePipelineConfig, int, int, str, str]:
    env = build_derived_env(tmp_path, cut_fraction=0.6)
    feature_dir = Path(env["feature_dir"])
    pct_checksum = _write_all_ones_percentile_overlay(feature_dir)
    tok_checksum = _write_token_boundary_overlay(
        feature_dir=feature_dir,
        data_dir=env["data_dir"],
        tokenizer=env["tokenizer"],
        segment_size=int(env["seg_size"]),
    )
    feature_index = FeatureIndex(
        str(feature_dir),
        percentile_checksum=pct_checksum,
        tokenizer_checksum=tok_checksum,
    )
    packed_dir = tmp_path / "real_packed"
    context_len = 32
    total_rows, max_n_seg, max_seg_size = precompute_packed_rows(
        parquet_paths=list_parquet_files(env["data_dir"])[:-1],
        tokenizer=env["tokenizer"],
        feature_index=feature_index,
        context_len=context_len,
        output_dir=packed_dir,
        buffer_size=4,
        rows_per_shard=2,
        compress_workers=1,
    )
    assert total_rows > 0
    write_manifest(
        packed_dir,
        "packed-real",
        OverlayKind.PACKED_ROW,
        metadata={
            "base_shard_stems": feature_index.shard_stems,
            "provenance_shards": len(glob_overlay_shards(provenance_dir(packed_dir))),
            "max_n_seg": max_n_seg,
            "max_seg_size": max_seg_size,
            "context_len": context_len,
            "pct_checksum": pct_checksum,
            "tok_checksum": tok_checksum,
        },
    )

    pipeline = _pipeline()
    packed_info = validate_packed_provenance(packed_dir)
    selection_hash = _selection_hash()
    sel_dir = packed_dir / "sel" / selection_hash
    sel_dir.mkdir(parents=True)
    splice_index = SpliceProvenanceIndex(feature_dir)
    stats = []
    for shard_path in packed_info.packed_shard_paths:
        stats.append(
            select_splice_counterfactual_features_for_shard(
                shard_path,
                pipeline,
                sel_dir,
                shard_path.stem.split(".")[0],
                provenance_path=provenance_dir(packed_dir) / shard_path.name,
                packed_info=packed_info,
                feature_index=FeatureIndex(
                    str(feature_dir),
                    percentile_checksum=pct_checksum,
                ),
                splice_index=splice_index,
                seed=7,
            )
        )
    total_segments = sum(s.total_packed_segments for s in stats)
    sidecar_positions = sum(s.total_sidecar_splice_positions_referenced for s in stats)
    _write_splice_selection_manifest(
        sel_dir=sel_dir,
        selection_hash=selection_hash,
        packed_info=packed_info,
        pipeline=pipeline,
        feature_dir=feature_dir,
        total_segments=total_segments,
        sidecar_positions=sidecar_positions,
    )
    return (
        env,
        packed_dir,
        sel_dir,
        pipeline,
        max_n_seg,
        _selection_max_f(sel_dir),
        pct_checksum,
        tok_checksum,
    )


class TestSupportGate:
    def test_precompute_allows_only_splice_counterfactual_delta(self) -> None:
        assert_precompute_supports_pipeline(_pipeline())
        with pytest.raises(ValueError, match="prev_segment or cum_max"):
            assert_precompute_supports_pipeline(_pipeline(mode="prev_segment"))


class TestProducerAndLoader:
    def test_identity_after_mask_splice_shard_is_bit_exact_to_unclustered(
        self,
        tmp_path: Path,
    ) -> None:
        """The complete compressed splice artifact preserves legacy bytes."""
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        feature_dir = Path(env["feature_dir"])
        feature_index, stem, doc_idx, base_idx = _choose_spliced_doc(feature_dir)
        packed_dir, shard_path = _write_one_doc_packed(
            tmp_path,
            feature_dir=feature_dir,
            fi=feature_index,
            stem=stem,
            doc_idx=doc_idx,
            base_idx=base_idx,
        )
        used_ids = {
            int(feature_id)
            for segment in feature_index.lookup(stem, doc_idx)
            for feature_id in segment.feature_ids.tolist()
        }
        survivor = min(used_ids)
        masked_member = next(
            feature_id for feature_id in range(4095, -1, -1) if feature_id not in used_ids
        )
        mask = FeatureMask(FeatureMaskMode.blacklist, (masked_member,))
        cluster_path = tmp_path / "identity_after_mask.json"
        cluster_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "clusters": [[masked_member, survivor]],
                }
            ),
            encoding="utf-8",
        )
        resolved = resolve_feature_clusters(
            {"feature_clusters": {"path": str(cluster_path)}},
            4096,
        )
        assert resolved is not None
        clusters = FeatureClusterConsolidation.build(resolved, mask)
        assert not bool(clusters.consolidated_away.any())
        assert t.equal(clusters.rep_of, t.arange(4096, dtype=t.int64))
        assert clusters.effective_negative_mask() is mask

        packed_info = validate_packed_provenance(packed_dir)
        provenance_path = provenance_dir(packed_dir) / shard_path.name
        splice_index = SpliceProvenanceIndex(feature_dir)
        unclustered_dir = tmp_path / "unclustered_splice_selection"
        clustered_dir = tmp_path / "identity_clustered_splice_selection"
        unclustered_dir.mkdir()
        clustered_dir.mkdir()
        unclustered_stats = select_splice_counterfactual_features_for_shard(
            shard_path,
            _sampling_pipeline(mask),
            unclustered_dir,
            "shard_00000",
            provenance_path=provenance_path,
            packed_info=packed_info,
            feature_index=feature_index,
            splice_index=splice_index,
            seed=714,
        )
        clustered_stats = select_splice_counterfactual_features_for_shard(
            shard_path,
            _sampling_pipeline(mask),
            clustered_dir,
            "shard_00000",
            provenance_path=provenance_path,
            packed_info=packed_info,
            feature_index=feature_index,
            splice_index=splice_index,
            seed=714,
            feature_clusters=clusters,
        )
        assert clustered_stats == unclustered_stats

        unclustered_path = overlay_shard_path(unclustered_dir, "shard_00000")
        clustered_path = overlay_shard_path(clustered_dir, "shard_00000")
        unclustered = load_selection_shard(unclustered_path)
        clustered = load_selection_shard(clustered_path)
        assert unclustered.total_segments > 0
        assert unclustered.total_selected > 0
        for name in (
            "sel_offsets",
            "sel_fids",
            "sel_buckets",
            "sel_pcts",
            "neg_offsets",
            "neg_fids",
        ):
            expected = getattr(unclustered, name)
            actual = getattr(clustered, name)
            assert actual.dtype == expected.dtype
            assert t.equal(actual, expected), name
        assert clustered_path.read_bytes() == unclustered_path.read_bytes()

    def test_splice_counterfactual_selection_shard_loads(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        sel = load_selection_shard(sel_dir / "shard_00000.safetensors.zst")
        base_segments = validate_packed_provenance(packed_dir).packed_shard_paths
        assert base_segments
        assert sel.total_segments > 0
        assert sel.total_selected > 0

        batch = next(
            precomputed_data_loader(
                packed_dir,
                B=1,
                feature_pipeline=pipeline,
                device="cpu",
                selection_dir=sel_dir,
                effective_buckets=pipeline.effective_buckets,
                splice_feature_dir=feature_dir,
            )
        )
        enc_mask = batch[4]
        assert bool(enc_mask.any())

    def test_deterministic_preselected_batch_matches_live_loader_bit_for_bit(
        self, tmp_path
    ) -> None:
        (
            env,
            packed_dir,
            sel_dir,
            _pipeline_for_manifest,
            max_n_seg,
            max_f,
            pct_checksum,
            tok_checksum,
        ) = _build_real_packed_preselected_fixture(tmp_path)
        feature_dir = Path(env["feature_dir"])

        live_batch = next(
            encoder_decoder_data_loader_with_state(
                env["tokenizer"],
                B=1,
                T=32,
                split=DatasetSplit.TRAIN,
                feature_index=FeatureIndex(
                    str(feature_dir),
                    percentile_checksum=pct_checksum,
                    tokenizer_checksum=tok_checksum,
                ),
                pipeline=_pipeline(),
                segment_size=PositiveInt.parse(int(env["seg_size"])),
                device="cpu",
                tokenizer_batch_size=128,
                buffer_size=4,
                data_dir=env["data_dir"],
                label_dropout=0.0,
                n_sae_features=4096,
                context_len=32,
                pad_n_seg=max_n_seg,
                pad_F=max_f,
            )
        )
        # Fresh pipeline instance proves loader validation matches the manifest
        # by canonical values, not by object identity.
        packed_pipeline = _pipeline()
        packed_batch = next(
            precomputed_data_loader(
                packed_dir,
                B=1,
                feature_pipeline=packed_pipeline,
                device="cpu",
                selection_dir=sel_dir,
                effective_buckets=packed_pipeline.effective_buckets,
                n_sae_features=4096,
                splice_feature_dir=feature_dir,
            )
        )

        for idx, name in enumerate(
            (
                "tokens",
                "targets",
                "enc_fids",
                "enc_mags",
                "enc_mask",
                "enc_pcts",
                "seg_map",
                "seg_doc_id",
                "fp_targets",
            )
        ):
            live_tensor = live_batch[idx]
            packed_tensor = packed_batch[idx]
            assert live_tensor is not None
            assert packed_tensor is not None
            assert t.equal(live_tensor, packed_tensor), name

    def test_splice_counterfactual_loader_requires_selection_dir(self, tmp_path) -> None:
        packed_dir, _sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        with pytest.raises(ValueError, match="requires selection_dir"):
            next(
                precomputed_data_loader(
                    packed_dir,
                    B=1,
                    feature_pipeline=pipeline,
                    device="cpu",
                    selection_dir=None,
                    effective_buckets=pipeline.effective_buckets,
                    splice_feature_dir=feature_dir,
                )
            )

    def test_splice_counterfactual_loader_requires_splice_feature_dir(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        with pytest.raises(ValueError, match="requires splice_feature_dir"):
            next(
                precomputed_data_loader(
                    packed_dir,
                    B=1,
                    feature_pipeline=pipeline,
                    device="cpu",
                    selection_dir=sel_dir,
                    effective_buckets=pipeline.effective_buckets,
                )
            )

    def test_splice_counterfactual_loader_rejects_stale_sidecar(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        sidecar = next((feature_dir / "_splice_provenance").glob("*.json"))
        sidecar.write_text("{}")
        with pytest.raises(ValueError, match="sidecar_fingerprint mismatch"):
            next(
                precomputed_data_loader(
                    packed_dir,
                    B=1,
                    feature_pipeline=pipeline,
                    device="cpu",
                    selection_dir=sel_dir,
                    effective_buckets=pipeline.effective_buckets,
                    splice_feature_dir=feature_dir,
                )
            )


class TestStaleSelectionRefusal:
    def test_splice_selection_refuses_stale_packed_rows_before_skip(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        shard_path = packed_dir / "shard_00000.safetensors.zst"
        packed = load_packed_row_shard(shard_path)
        packed.tokens[0, 0] = int(packed.tokens[0, 0].item()) + 1
        save_packed_row_shard(packed, packed_dir / "shard_00000")
        packed_info = validate_packed_provenance(packed_dir)
        sidecar_hash, derived_hash, boundary_hash = _selection_manifest_inputs(
            feature_dir, packed_info, pipeline
        )
        with pytest.raises(SpliceProvenanceError, match="packed_row_fingerprint"):
            _assert_splice_counterfactual_sel_manifest_current(
                sel_dir,
                pipeline,
                packed_info,
                sidecar_hash,
                derived_hash,
                boundary_hash,
            )

    def test_splice_loader_rejects_stale_packed_rows(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        shard_path = packed_dir / "shard_00000.safetensors.zst"
        packed = load_packed_row_shard(shard_path)
        packed.tokens[0, 0] = int(packed.tokens[0, 0].item()) + 1
        save_packed_row_shard(packed, packed_dir / "shard_00000")
        with pytest.raises(ValueError, match="packed_provenance mismatch"):
            next(
                precomputed_data_loader(
                    packed_dir,
                    B=1,
                    feature_pipeline=pipeline,
                    device="cpu",
                    selection_dir=sel_dir,
                    effective_buckets=pipeline.effective_buckets,
                    splice_feature_dir=feature_dir,
                )
            )

    def test_packed_provenance_rejects_boolean_shard_count(self, tmp_path) -> None:
        env = build_derived_env(tmp_path, cut_fraction=0.6)
        feature_dir = Path(env["feature_dir"])
        fi, stem, doc_idx, base_idx = _choose_spliced_doc(feature_dir)
        packed_dir, _shard_path = _write_one_doc_packed(
            tmp_path,
            feature_dir=feature_dir,
            fi=fi,
            stem=stem,
            doc_idx=doc_idx,
            base_idx=base_idx,
        )
        manifest_path = packed_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["metadata"]["provenance_shards"] = True
        manifest_path.write_text(json.dumps(manifest))
        with pytest.raises(SpliceProvenanceError, match="missing integer"):
            validate_packed_provenance(packed_dir)

    def test_splice_selection_refuses_stale_sidecar_manifest(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        packed_info = validate_packed_provenance(packed_dir)
        _sidecar_hash, derived_hash, boundary_hash = _selection_manifest_inputs(
            feature_dir, packed_info, pipeline
        )
        with pytest.raises(SpliceProvenanceError, match="stale"):
            _assert_splice_counterfactual_sel_manifest_current(
                sel_dir,
                pipeline,
                packed_info,
                "not-the-current-sidecar-fingerprint",
                derived_hash,
                boundary_hash,
            )

    def test_splice_selection_refuses_stale_derived_feature_magnitudes(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        packed_info = validate_packed_provenance(packed_dir)
        feature_path = feature_dir / f"{packed_info.base_shard_stems[0]}_features.safetensors.zst"
        shard = load_features(feature_path)
        shard["feature_mags"] = shard["feature_mags"].clone()
        shard["feature_mags"][0] = shard["feature_mags"][0] + 0.5
        save_features(shard, feature_path)
        sidecar_hash, derived_hash, boundary_hash = _selection_manifest_inputs(
            feature_dir, packed_info, pipeline
        )
        with pytest.raises(SpliceProvenanceError, match="derived_feature_fingerprint"):
            _assert_splice_counterfactual_sel_manifest_current(
                sel_dir,
                pipeline,
                packed_info,
                sidecar_hash,
                derived_hash,
                boundary_hash,
            )

    def test_splice_loader_rejects_stale_derived_feature_magnitudes(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        packed_info = validate_packed_provenance(packed_dir)
        feature_path = feature_dir / f"{packed_info.base_shard_stems[0]}_features.safetensors.zst"
        shard = load_features(feature_path)
        shard["feature_mags"] = shard["feature_mags"].clone()
        shard["feature_mags"][0] = shard["feature_mags"][0] + 0.5
        save_features(shard, feature_path)
        with pytest.raises(ValueError, match="derived_feature_fingerprint mismatch"):
            next(
                precomputed_data_loader(
                    packed_dir,
                    B=1,
                    feature_pipeline=pipeline,
                    device="cpu",
                    selection_dir=sel_dir,
                    effective_buckets=pipeline.effective_buckets,
                    splice_feature_dir=feature_dir,
                )
            )

    def test_splice_selection_refuses_existing_shards_without_manifest(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        (sel_dir / "manifest.json").unlink()
        packed_info = validate_packed_provenance(packed_dir)
        sidecar_hash, derived_hash, boundary_hash = _selection_manifest_inputs(
            feature_dir, packed_info, pipeline
        )
        with pytest.raises(SpliceProvenanceError, match="no valid manifest"):
            _assert_splice_counterfactual_sel_manifest_current(
                sel_dir,
                pipeline,
                packed_info,
                sidecar_hash,
                derived_hash,
                boundary_hash,
            )


class TestTrainingRouting:
    def test_dual_validation_mode_depends_on_precomputed_flag(self, tmp_path) -> None:
        assert (
            _dual_validation_mode(
                original_val_feature_dir=tmp_path,
                splice_cf_enabled=True,
                use_precomputed=True,
            )
            is DualValidationMode.packed
        )
        assert (
            _dual_validation_mode(
                original_val_feature_dir=tmp_path,
                splice_cf_enabled=True,
                use_precomputed=False,
            )
            is DualValidationMode.live
        )

    def test_packed_row_inputs_allow_splice_only_with_selection(self, tmp_path) -> None:
        packed = tmp_path / "packed"
        sel = tmp_path / "sel"
        pipeline = _pipeline()
        assert _packed_row_inputs_for_pipeline(
            packed_row_dir=packed,
            selection_dir=sel,
            feature_pipeline=pipeline,
        ) == (packed, sel, False)
        with pytest.raises(ValueError, match="requires a precomputed selection_dir"):
            _packed_row_inputs_for_pipeline(
                packed_row_dir=packed,
                selection_dir=None,
                feature_pipeline=pipeline,
            )


class TestSelectionDirBoundaryFingerprint:
    """The training-script selection-dir assert's boundary check — previously
    only source-pattern-pinned, so a mis-threaded fingerprint or a wrong gate
    would have passed CI silently (pre-merge review finding on the
    discover_packed_selection extraction). The fingerprint is derived from the
    live boundaries tensor exactly the way build_shared_state derives it."""

    def test_matching_fingerprint_passes_and_mismatch_raises(self, tmp_path) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)

        def assert_selection_dir(boundary_fingerprint: str) -> Path:
            return validate_complete_selection_dir(
                packed_row_dir=packed_dir,
                selection_hash=sel_dir.name,
                configured_mask=None,
                label="boundary-parity",
                boundary_fingerprint=boundary_fingerprint,
                delta_config={
                    "mode": "splice_counterfactual",
                    "neg_delta_ratio": 1.0,
                },
                splice_feature_dir=feature_dir,
            )

        fingerprint = tensor_content_fingerprint(pipeline.boundaries)
        assert assert_selection_dir(fingerprint) == sel_dir
        with pytest.raises(ValueError, match="boundary_fingerprint mismatch"):
            assert_selection_dir("0" * 64)
        altered = tensor_content_fingerprint(pipeline.boundaries.flip(-1) + 1)
        assert altered != fingerprint
        with pytest.raises(ValueError, match="boundary_fingerprint mismatch"):
            assert_selection_dir(altered)

    def test_training_requires_exact_splice_batching_manifest_descriptor(
        self,
        tmp_path,
    ) -> None:
        packed_dir, sel_dir, pipeline, feature_dir = _build_preselected_fixture(tmp_path)
        delta = {
            "mode": "splice_counterfactual",
            "neg_delta_ratio": 1.0,
        }

        def assert_selection_dir(splice_batch_size: int) -> Path:
            return validate_complete_selection_dir(
                packed_row_dir=packed_dir,
                selection_hash=sel_dir.name,
                configured_mask=None,
                label="batching-parity",
                boundary_fingerprint=tensor_content_fingerprint(pipeline.boundaries),
                delta_config=delta,
                splice_feature_dir=feature_dir,
                splice_batch_size=splice_batch_size,
            )

        assert assert_selection_dir(1) == sel_dir

        manifest_path = sel_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        splice = manifest["metadata"]["splice_counterfactual"]
        splice["selection_batching"] = {
            "schema_version": 1,
            "algorithm": "ordinary_execution_runs_v1",
            "batch_size": 8,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="selection batching mismatch"):
            assert_selection_dir(1)
        assert assert_selection_dir(8) == sel_dir

        splice.pop("selection_batching")
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="selection batching mismatch"):
            assert_selection_dir(8)

    def test_training_rejects_missing_splice_manifest_metadata(self, tmp_path) -> None:
        packed_dir, sel_dir, _pipeline_config, _feature_dir = _build_preselected_fixture(tmp_path)
        manifest_path = sel_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"].pop("splice_counterfactual")
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="missing splice_counterfactual"):
            validate_complete_selection_dir(
                packed_row_dir=packed_dir,
                selection_hash=sel_dir.name,
                configured_mask=None,
                label="missing-splice-metadata",
                delta_config={
                    "mode": "splice_counterfactual",
                    "neg_delta_ratio": 1.0,
                },
            )

    def test_training_rejects_splice_source_for_non_splice_delta(self, tmp_path) -> None:
        packed_dir, sel_dir, _pipeline_config, feature_dir = _build_preselected_fixture(tmp_path)
        delta = {
            "mode": "prev_segment",
            "neg_delta_ratio": 1.0,
        }
        parsed_delta = parse_feature_delta(delta)
        assert parsed_delta is not None
        manifest_path = sel_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["metadata"]["feature_delta"] = canonical_feature_delta_dict(parsed_delta)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="valid only for.*splice_counterfactual"):
            validate_complete_selection_dir(
                packed_row_dir=packed_dir,
                selection_hash=sel_dir.name,
                configured_mask=None,
                label="non-splice-source",
                delta_config=delta,
                splice_feature_dir=feature_dir,
            )
