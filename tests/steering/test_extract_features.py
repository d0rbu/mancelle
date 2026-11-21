"""Unit and property-based tests for steering.extract_features.

Tests pure functions and data format invariants without loading real models.
Run with:
    uv run pytest tests/test_extract_features.py -v
"""

import ast
import inspect
import tempfile
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock, patch

import hypothesis.strategies as st
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from hypothesis import given, settings
from omegaconf import OmegaConf

import steering.extract_features as ef
from errors import EmptyExtractionSegmentError
from steering.compute_statistics import (
    PERCENTILES,
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
    accumulate_shard,
    compute_per_feature_percentiles,
    compute_statistics,
    estimate_10b,
    init_accumulators,
    load_all_shards,
    log_summary,
)
from steering.extract_features import (
    ShardData,
    _build_nnterp_gemma3,
    _coerce_config_l0,
    _get_sae_activations,
    _offload_transformer_tail_for_early_stop,
    _resolve_extraction_sae,
    _validate_csr_invariants,
    extract_sparse_gpu,
    load_sae_and_model,
    pool_segments_batched_gpu,
)
from steering.feature_io import load_features, save_features
from steering.feature_paths import PoolingMode
from steering.refined_types import NonNegativeInt
from steering.sae_resolver import (
    ActivationType,
    L0Tier,
    SAERelease,
    SAEWidth,
    resolve_sae,
)
from steering.segmentation import SegmentMode
from steering.stats_io import load_statistics
from steering.stats_schema import HistogramData
from tests.steering._compute_statistics_reference import build_histogram


def _extract_features_func_def(name: str) -> ast.FunctionDef:
    src = Path(load_sae_and_model.__code__.co_filename).read_text()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise AssertionError(f"function {name!r} not found in steering.extract_features")


# ---------------------------------------------------------------------------
# Reference implementations (moved from production code for test comparisons)
# ---------------------------------------------------------------------------


def _ref_pool_segment(
    sae_acts: t.Tensor, start: int, end: int, pooling: PoolingMode
) -> t.Tensor:
    """Reference: pool SAE activations for a single segment."""
    segment = sae_acts[start:end]
    if pooling == PoolingMode.last_token:
        return segment[-1]
    if pooling == PoolingMode.max_pool:
        return segment.max(dim=0).values
    if pooling == PoolingMode.mean_pool:
        return segment.mean(dim=0)
    raise ValueError(f"Unknown pooling mode: {pooling}")


def _ref_extract_sparse_features(pooled: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
    """Reference: extract non-zero features, sorted by magnitude descending."""
    nonzero_mask = pooled > 0
    feature_ids = nonzero_mask.nonzero(as_tuple=True)[0].to(t.int32)
    feature_mags = pooled[nonzero_mask].to(t.float32)
    sort_idx = feature_mags.argsort(descending=True)
    return feature_ids[sort_idx].cpu(), feature_mags[sort_idx].cpu()


# ---------------------------------------------------------------------------
# PoolingMode enum
# ---------------------------------------------------------------------------


class TestPoolingMode:
    def test_round_trip_from_string(self):
        for mode in PoolingMode:
            assert PoolingMode(mode.value) is mode

    def test_invalid_mode_raises(self):
        with pytest.raises(ValueError):
            PoolingMode("nonexistent_mode")


# ---------------------------------------------------------------------------
# estimate_10b — property-based tests
# ---------------------------------------------------------------------------


class TestEstimate10b:
    def test_basic_extrapolation(self):
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=100000,
            mean_tokens_per_doc=1000.0,
            median_tokens_per_doc=900,
        )
        storage = StorageStats(
            total_bytes=10000000,
            mean_bytes_per_doc=100_000.0,
            mean_bytes_per_token=100.0,
        )
        result = estimate_10b(corpus, storage)
        assert result.target_tokens == 10_000_000_000
        assert result.estimated_docs == 10_000_000
        expected_gb = round(10_000_000 * 100_000 / (1024**3), 1)
        assert result.estimated_storage_gb == expected_gb

    def test_zero_tokens_per_doc_raises(self):
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=0,
            n_tokens_gpt2=0,
            mean_tokens_per_doc=0,
            median_tokens_per_doc=0,
        )
        storage = StorageStats(
            total_bytes=0, mean_bytes_per_doc=100.0, mean_bytes_per_token=1.0
        )
        with pytest.raises(ValueError, match="Cannot extrapolate"):
            estimate_10b(corpus, storage)

    def test_custom_target(self):
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=50000,
            mean_tokens_per_doc=500.0,
            median_tokens_per_doc=450,
        )
        storage = StorageStats(
            total_bytes=5000000, mean_bytes_per_doc=50_000.0, mean_bytes_per_token=100.0
        )
        result = estimate_10b(corpus, storage, target_tokens=1_000_000_000)
        assert result.target_tokens == 1_000_000_000
        assert result.estimated_docs == 2_000_000

    @given(
        tokens_per_doc=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False),
        bytes_per_doc=st.floats(min_value=1.0, max_value=1e9, allow_nan=False),
        target=st.integers(min_value=1_000_000, max_value=100_000_000_000),
    )
    def test_scaling_is_linear(self, tokens_per_doc, bytes_per_doc, target):
        """Doubling target approximately doubles estimated docs.

        Uses int() truncation so we allow +-1 rounding tolerance.
        """
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=int(100 * tokens_per_doc),
            mean_tokens_per_doc=tokens_per_doc,
            median_tokens_per_doc=int(tokens_per_doc),
        )
        storage = StorageStats(
            total_bytes=int(100 * bytes_per_doc),
            mean_bytes_per_doc=bytes_per_doc,
            mean_bytes_per_token=bytes_per_doc / max(tokens_per_doc, 1),
        )
        r1 = estimate_10b(corpus, storage, target_tokens=target)
        r2 = estimate_10b(corpus, storage, target_tokens=target * 2)
        # int(target / tpd) * 2 vs int(target * 2 / tpd) can differ by 1
        assert abs(r2.estimated_docs - r1.estimated_docs * 2) <= 1

    @given(
        tokens_per_doc=st.floats(min_value=1.0, max_value=100_000.0, allow_nan=False),
        bytes_per_doc=st.floats(min_value=1.0, max_value=1e9, allow_nan=False),
    )
    def test_estimated_docs_positive(self, tokens_per_doc, bytes_per_doc):
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=int(100 * tokens_per_doc),
            mean_tokens_per_doc=tokens_per_doc,
            median_tokens_per_doc=int(tokens_per_doc),
        )
        storage = StorageStats(
            total_bytes=int(100 * bytes_per_doc),
            mean_bytes_per_doc=bytes_per_doc,
            mean_bytes_per_token=bytes_per_doc / max(tokens_per_doc, 1),
        )
        result = estimate_10b(corpus, storage)
        assert result.estimated_docs > 0
        assert result.estimated_storage_gb >= 0

    def test_wall_time_extrapolation(self):
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=100_000,
            mean_tokens_per_doc=1000.0,
            median_tokens_per_doc=900,
        )
        storage = StorageStats(
            total_bytes=10_000_000,
            mean_bytes_per_doc=100_000.0,
            mean_bytes_per_token=100.0,
        )
        result = estimate_10b(corpus, storage, elapsed_seconds=360.0)
        # scale_factor = 10B / 100K = 100_000
        # estimated_hours = 360 * 100_000 / 3600 = 10_000.0
        assert result.estimated_wall_time_hours == 10_000.0

    def test_wall_time_zero_when_no_elapsed(self):
        corpus = CorpusStats(
            n_shards_processed=1,
            n_docs=100,
            n_tokens_gpt2=100_000,
            mean_tokens_per_doc=1000.0,
            median_tokens_per_doc=900,
        )
        storage = StorageStats(
            total_bytes=10_000_000,
            mean_bytes_per_doc=100_000.0,
            mean_bytes_per_token=100.0,
        )
        result = estimate_10b(corpus, storage)
        assert result.estimated_wall_time_hours == 0.0


# ---------------------------------------------------------------------------
# CSR format invariants — tests on synthetic feature files
# ---------------------------------------------------------------------------


def _make_synthetic_feature_file(
    path: str,
    n_docs: int = 100,
    features_per_doc: int = 50,
    n_total_features: int = 24576,
):
    """Create a synthetic feature shard file with valid CSR structure."""
    offsets = [0]
    all_ids = []
    all_mags = []
    doc_token_counts = []

    for _ in range(n_docs):
        # Issue #522: a base shard with n_segments > 0 may not contain an empty
        # segment. Each doc here is one segment, so guarantee >=1 feature per
        # segment (realistic: real SAEs fire >=1 feature per pooled segment).
        # The property test still varies density across 1..100.
        n_feat = max(1, features_per_doc)
        ids = t.randint(0, n_total_features, (n_feat,), dtype=t.int32)
        mags = t.rand(n_feat, dtype=t.float32).abs() + 0.01  # all > 0
        all_ids.append(ids)
        all_mags.append(mags)
        offsets.append(offsets[-1] + n_feat)
        doc_token_counts.append(t.randint(100, 5000, (1,)).item())

    save_data = {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.cat(all_ids) if all_ids else t.tensor([], dtype=t.int32),
        "feature_mags": t.cat(all_mags) if all_mags else t.tensor([], dtype=t.float32),
        "doc_token_counts": t.tensor(doc_token_counts, dtype=t.int32),
        "n_docs": n_docs,
        "doc_ids": t.arange(n_docs, dtype=t.int32),
        "segment_indices": t.zeros(n_docs, dtype=t.int32),
        "segment_positions": t.zeros(n_docs, dtype=t.int32),
        "n_segments": n_docs,
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "layer": 8,
        "context_window": 1024,
        "pooling": "mean_pool",
    }
    actual_path = save_features(save_data, path)
    return save_data, str(actual_path)


class TestCSRFormatInvariants:
    """Verify CSR sparse format invariants on synthetic feature files."""

    @pytest.fixture
    def feature_file(self, tmp_path):
        _, actual_path = _make_synthetic_feature_file(
            str(tmp_path / "shard_00000_features"), n_docs=50, features_per_doc=30
        )
        return actual_path

    def test_offsets_monotonically_increasing(self, feature_file):
        data = load_features(feature_file)
        offsets = data["offsets"]
        diffs = offsets[1:] - offsets[:-1]
        assert (diffs >= 0).all()

    def test_offsets_start_at_zero(self, feature_file):
        data = load_features(feature_file)
        assert data["offsets"][0].item() == 0

    def test_offsets_end_matches_feature_count(self, feature_file):
        data = load_features(feature_file)
        assert data["offsets"][-1].item() == len(data["feature_ids"])

    def test_feature_ids_and_mags_same_length(self, feature_file):
        data = load_features(feature_file)
        assert len(data["feature_ids"]) == len(data["feature_mags"])

    def test_offsets_length_is_n_segments_plus_one(self, feature_file):
        data = load_features(feature_file)
        assert len(data["offsets"]) == data["n_segments"] + 1

    def test_doc_token_counts_length_matches_n_docs(self, feature_file):
        data = load_features(feature_file)
        assert len(data["doc_token_counts"]) == data["n_docs"]

    def test_dtypes_correct(self, feature_file):
        data = load_features(feature_file)
        assert data["offsets"].dtype == t.int64
        assert data["feature_ids"].dtype == t.int32
        assert data["feature_mags"].dtype == t.float32
        assert data["doc_token_counts"].dtype == t.int32

    def test_all_magnitudes_positive(self, feature_file):
        data = load_features(feature_file)
        assert (data["feature_mags"] > 0).all()

    def test_per_segment_ids_sorted_ascending(self, feature_file):
        data = load_features(feature_file)
        offsets = data["offsets"]
        ids = data["feature_ids"]
        n_segments = data["n_segments"]
        for i in range(n_segments):
            start, end = int(offsets[i].item()), int(offsets[i + 1].item())
            seg_ids = ids[start:end]
            if len(seg_ids) > 1:
                diffs = seg_ids[1:] - seg_ids[:-1]
                assert (diffs >= 0).all(), f"Segment {i} IDs not sorted ascending"

    def test_segment_metadata_present(self, feature_file):
        data = load_features(feature_file)
        assert "doc_ids" in data
        assert "segment_indices" in data
        assert "segment_positions" in data
        assert "n_segments" in data
        assert "segment_mode" in data

    def test_metadata_fields_present(self, feature_file):
        data = load_features(feature_file)
        assert "layer" in data
        assert "context_window" in data
        assert "pooling" in data
        assert isinstance(data["layer"], int)
        assert isinstance(data["context_window"], int)
        assert isinstance(data["pooling"], str)

    @given(
        n_docs=st.integers(min_value=1, max_value=200),
        features_per_doc=st.integers(min_value=0, max_value=100),
    )
    @settings(max_examples=20, deadline=None)
    def test_csr_invariants_property(self, n_docs, features_per_doc):
        """CSR invariants hold for arbitrary doc/feature counts."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _, actual_path = _make_synthetic_feature_file(
                str(tmpdir) + "/shard_features",
                n_docs=n_docs,
                features_per_doc=features_per_doc,
            )
            data = load_features(actual_path)

        offsets = data["offsets"]
        assert offsets[0].item() == 0
        assert offsets[-1].item() == len(data["feature_ids"])
        assert len(offsets) == data["n_segments"] + 1
        assert len(data["feature_ids"]) == len(data["feature_mags"])
        assert len(data["doc_token_counts"]) == n_docs


def _csr_kwargs(offsets, n_feats):
    n_seg = len(offsets) - 1
    return {
        "offsets": t.tensor(offsets, dtype=t.int64),
        "feature_ids": t.arange(n_feats, dtype=t.int32),
        "feature_mags": t.ones(n_feats, dtype=t.float32),
        "n_segments": n_seg,
        "doc_ids": t.zeros(n_seg, dtype=t.int32),
        "segment_indices": t.arange(n_seg, dtype=t.int32),
        "segment_positions": t.arange(n_seg, dtype=t.int32),
        "segment_char_starts": t.arange(n_seg, dtype=t.int32),
        "segment_char_ends": t.arange(1, n_seg + 1, dtype=t.int32),
    }


def test_validate_csr_invariants_raises_on_empty_segment():
    with pytest.raises(EmptyExtractionSegmentError, match="segment 1"):
        _validate_csr_invariants(**_csr_kwargs([0, 2, 2, 4], 4))


def test_validate_csr_invariants_accepts_nonempty():
    _validate_csr_invariants(**_csr_kwargs([0, 2, 4], 4))  # no raise


def test_validate_csr_invariants_accepts_zero_segments():
    _validate_csr_invariants(
        offsets=t.tensor([0], dtype=t.int64),
        feature_ids=t.tensor([], dtype=t.int32),
        feature_mags=t.tensor([], dtype=t.float32),
        n_segments=0,
        doc_ids=t.tensor([], dtype=t.int32),
        segment_indices=t.tensor([], dtype=t.int32),
        segment_positions=t.tensor([], dtype=t.int32),
        segment_char_starts=t.tensor([], dtype=t.int32),
        segment_char_ends=t.tensor([], dtype=t.int32),
    )  # no raise


def test_validate_csr_invariants_empty_segment_doc_id_sentinel():
    """The empty-segment check runs BEFORE the len(doc_ids)==n_segments check,
    so a doc_ids SHORTER than the offending segment index hits the ``-1``
    sentinel branch (``doc = ... if i < len(doc_ids) else -1``). The empty
    segment at i=1 with a length-1 doc_ids must report "doc -1" (issue #522)."""
    with pytest.raises(EmptyExtractionSegmentError, match="doc -1") as exc:
        _validate_csr_invariants(
            offsets=t.tensor([0, 2, 2, 4], dtype=t.int64),  # n_segments=3, empty i=1
            feature_ids=t.arange(4, dtype=t.int32),
            feature_mags=t.ones(4, dtype=t.float32),
            n_segments=3,
            doc_ids=t.zeros(1, dtype=t.int32),  # malformed: shorter than n_segments
            segment_indices=t.arange(3, dtype=t.int32),
            segment_positions=t.arange(3, dtype=t.int32),
            segment_char_starts=t.arange(3, dtype=t.int32),
            segment_char_ends=t.arange(1, 4, dtype=t.int32),
        )
    assert "segment 1 (doc -1)" in str(exc.value)


# ---------------------------------------------------------------------------
# compute_statistics
# ---------------------------------------------------------------------------


class TestComputeStatistics:
    @pytest.fixture
    def feature_dir(self, tmp_path):
        """Create a temp directory with two synthetic feature shards."""
        _make_synthetic_feature_file(
            str(tmp_path / "shard_00000_features"),
            n_docs=20,
            features_per_doc=10,
        )
        _make_synthetic_feature_file(
            str(tmp_path / "shard_00001_features"),
            n_docs=30,
            features_per_doc=15,
        )
        return str(tmp_path)

    def test_returns_all_sections(self, feature_dir, tmp_path):
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        assert hasattr(stats, "metadata")
        assert hasattr(stats, "corpus")
        assert hasattr(stats, "features")
        assert hasattr(stats, "storage")
        assert hasattr(stats, "per_feature_stats")
        assert hasattr(stats, "extrapolation_10b")

    def test_corpus_counts(self, feature_dir, tmp_path):
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        assert stats.corpus.n_shards_processed == 2
        assert stats.corpus.n_docs == 50  # 20 + 30

    def test_writes_valid_safetensors(self, feature_dir, tmp_path):
        output_path = str(tmp_path / "stats")
        compute_statistics(feature_dir, output_path)
        loaded = load_statistics(tmp_path / "stats.safetensors")
        assert loaded.corpus.n_docs == 50

    def test_empty_dir_raises(self, tmp_path):
        output_path = str(tmp_path / "stats.json")
        with pytest.raises(FileNotFoundError):
            compute_statistics(str(tmp_path), output_path)

    def test_feature_stats_non_negative(self, feature_dir, tmp_path):
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        features = stats.features
        assert features.n_features_seen > 0
        assert features.mean_active_per_segment > 0
        assert features.min_active_per_segment >= 0
        assert features.max_active_per_segment >= features.min_active_per_segment

    def test_storage_stats_positive(self, feature_dir, tmp_path):
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        storage = stats.storage
        assert storage.total_bytes > 0
        assert storage.mean_bytes_per_doc > 0
        assert storage.mean_bytes_per_token > 0

    def test_per_feature_stats_have_count_and_mag(self, feature_dir, tmp_path):
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        for fstats in stats.per_feature_stats.values():
            assert fstats.count > 0
            assert fstats.mean_mag > 0

    def test_has_metadata(self, feature_dir, tmp_path):
        """Verify metadata fields are populated from synthetic shard data."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        meta = stats.metadata
        assert meta.pooling_mode == "mean_pool"
        assert meta.segment_mode == "fixed_tokens"
        assert meta.segment_size == 128
        assert meta.context_window == 1024
        assert meta.layer == 8
        assert meta.n_sae_features > 0
        assert len(meta.generated_at) > 0

    def test_features_has_n_segments(self, feature_dir, tmp_path):
        """Verify n_segments is populated and positive."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        assert stats.features.n_segments > 0

    def test_per_feature_pct_segments_active(self, feature_dir, tmp_path):
        """Verify pct_segments_active is computed correctly per feature."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        n_segments = stats.features.n_segments
        for fid, fstats in stats.per_feature_stats.items():
            expected = round(fstats.count / max(n_segments, 1) * 100, 4)
            assert fstats.pct_segments_active == expected, (
                f"Feature {fid}: pct_segments_active={fstats.pct_segments_active} != expected {expected}"
            )

    def test_features_has_count_percentiles(self, feature_dir, tmp_path):
        """Verify decile keys exist and values are non-decreasing."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        pcts = stats.features.count_percentiles
        expected_keys = [f"p{p}" for p in PERCENTILES]
        for key in expected_keys:
            assert key in pcts, f"Missing key {key} in count_percentiles"
        # Non-decreasing
        values = [pcts[k] for k in expected_keys]
        for i in range(1, len(values)):
            assert values[i] >= values[i - 1], (
                f"count_percentiles not non-decreasing: {expected_keys[i - 1]}={values[i - 1]} > {expected_keys[i]}={values[i]}"
            )

    def test_features_has_magnitude_stats(self, feature_dir, tmp_path):
        """Verify mean, std, min, max, percentiles keys in magnitude_stats."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        ms = stats.features.magnitude_stats
        assert "mean" in ms
        assert "std" in ms
        assert "min" in ms
        assert "max" in ms
        assert "percentiles" in ms
        pcts = ms["percentiles"]
        for p in PERCENTILES:
            assert f"p{p}" in pcts  # ty: ignore[unsupported-operator]

    def test_features_pct_active(self, feature_dir, tmp_path):
        """Verify 0 < pct_features_active <= 100."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        assert 0 < stats.features.pct_features_active <= 100

    def test_per_feature_stats_enriched(self, feature_dir, tmp_path):
        """Verify std_mag, min_mag, max_mag, median_mag, percentiles for each feature."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        for fid, fstats in stats.per_feature_stats.items():
            assert fstats.std_mag >= 0, f"Feature {fid}: std_mag should be >= 0"
            assert fstats.min_mag > 0, f"Feature {fid}: min_mag should be > 0"
            assert fstats.max_mag >= fstats.min_mag, (
                f"Feature {fid}: max_mag ({fstats.max_mag}) < min_mag ({fstats.min_mag})"
            )
            assert fstats.median_mag > 0, f"Feature {fid}: median_mag should be > 0"
            assert isinstance(fstats.percentiles, dict)
            pct_keys = [f"p{p}" for p in PERCENTILES]
            for key in pct_keys:
                assert key in fstats.percentiles, (
                    f"Feature {fid}: missing {key} in percentiles"
                )
            # Percentiles should be non-decreasing
            vals = [fstats.percentiles[k] for k in pct_keys]
            assert vals == sorted(vals), (
                f"Feature {fid}: percentiles not non-decreasing: {vals}"
            )

    def test_per_feature_stats_sorted_by_count_desc(self, feature_dir, tmp_path):
        """Verify per-feature stats are sorted by count descending."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        counts = [fstats.count for fstats in stats.per_feature_stats.values()]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], (
                f"per_feature_stats not sorted descending at index {i}: "
                f"{counts[i - 1]} < {counts[i]}"
            )

    def test_per_feature_stats_sorted_in_safetensors(self, feature_dir, tmp_path):
        """Verify safetensors output preserves descending count order."""
        output_path = str(tmp_path / "stats")
        compute_statistics(feature_dir, output_path)
        loaded = load_statistics(tmp_path / "stats.safetensors")
        counts = [fstat.count for fstat in loaded.per_feature_stats.values()]
        for i in range(1, len(counts)):
            assert counts[i] <= counts[i - 1], (
                f"safetensors per_feature_stats not sorted descending at index {i}"
            )

    def test_extrapolation_has_wall_time(self, feature_dir, tmp_path):
        """Verify estimated_wall_time_hours field exists and is >= 0."""
        output_path = str(tmp_path / "stats.json")
        stats = compute_statistics(feature_dir, output_path)
        assert hasattr(stats.extrapolation_10b, "estimated_wall_time_hours")
        assert stats.extrapolation_10b.estimated_wall_time_hours >= 0


# ---------------------------------------------------------------------------
# load_all_shards
# ---------------------------------------------------------------------------


class TestLoadAllShards:
    def test_single_shard(self, tmp_path):
        _make_synthetic_feature_file(
            str(tmp_path / "shard_00000_features"),
            n_docs=10,
            features_per_doc=5,
        )
        result = load_all_shards(str(tmp_path))
        assert result.n_shards == 1
        assert result.total_docs == 10
        assert len(result.shard_data_list) == 1
        assert result.feature_ids.ndim == 1
        assert result.feature_mags.ndim == 1
        assert len(result.feature_ids) == 10 * 5
        assert len(result.feature_mags) == 10 * 5

    def test_multiple_shards(self, tmp_path):
        _make_synthetic_feature_file(
            str(tmp_path / "shard_00000_features"),
            n_docs=10,
            features_per_doc=5,
        )
        _make_synthetic_feature_file(
            str(tmp_path / "shard_00001_features"),
            n_docs=20,
            features_per_doc=8,
        )
        result = load_all_shards(str(tmp_path))
        assert result.n_shards == 2
        assert result.total_docs == 30
        assert len(result.feature_ids) == 10 * 5 + 20 * 8
        assert result.metadata.pooling == "mean_pool"
        assert result.metadata.segment_mode == "fixed_tokens"
        assert result.metadata.layer == 8

    def test_empty_dir_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_all_shards(str(tmp_path))


# ---------------------------------------------------------------------------
# Accumulators / init_accumulators / accumulate_shard
# ---------------------------------------------------------------------------


class TestAccumulateShard:
    def test_updates_occurrence(self):
        acc = init_accumulators(10)
        feat_ids = t.tensor([0, 1, 1, 3], dtype=t.int32)
        feat_mags = t.tensor([0.5, 0.3, 0.7, 0.2], dtype=t.float32)
        accumulate_shard(acc, feat_ids, feat_mags)
        assert acc.occurrence[0].item() == 1
        assert acc.occurrence[1].item() == 2
        assert acc.occurrence[3].item() == 1
        assert acc.occurrence[2].item() == 0

    def test_updates_mag_sum(self):
        acc = init_accumulators(10)
        feat_ids = t.tensor([0, 0], dtype=t.int32)
        feat_mags = t.tensor([0.3, 0.7], dtype=t.float32)
        accumulate_shard(acc, feat_ids, feat_mags)
        assert abs(acc.mag_sum[0].item() - 1.0) < 1e-5

    def test_auto_expands(self):
        acc = init_accumulators(5)
        assert len(acc.occurrence) == 5
        feat_ids = t.tensor([10], dtype=t.int32)
        feat_mags = t.tensor([1.0], dtype=t.float32)
        accumulate_shard(acc, feat_ids, feat_mags)
        assert len(acc.occurrence) == 11
        assert acc.occurrence[10].item() == 1

    def test_empty_shard_noop(self):
        acc = init_accumulators(5)
        accumulate_shard(
            acc, t.tensor([], dtype=t.int32), t.tensor([], dtype=t.float32)
        )
        assert acc.occurrence.sum().item() == 0


# ---------------------------------------------------------------------------
# build_histogram
# ---------------------------------------------------------------------------


class TestBuildHistogram:
    def test_basic(self):
        data = t.randn(100)
        result = build_histogram(data, n_bins=10)
        assert result is not None
        assert isinstance(result, HistogramData)
        assert len(result.bin_edges) == 11
        assert len(result.counts) == 10
        assert sum(result.counts) == 100

    def test_empty_returns_none(self):
        data = t.tensor([])
        result = build_histogram(data)
        assert result is None


# ---------------------------------------------------------------------------
# compute_per_feature_percentiles
# ---------------------------------------------------------------------------


class TestComputePerFeaturePercentiles:
    def test_basic(self):
        feat_ids = t.tensor([0, 0, 0, 1, 1], dtype=t.int32)
        feat_mags = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=t.float32)
        details, shared_edges = compute_per_feature_percentiles(feat_ids, feat_mags)
        assert 0 in details
        assert 1 in details
        assert details[0].percentiles  # non-empty dict
        assert details[0].median is not None
        assert details[0].histogram is not None
        assert shared_edges is not None

    def test_produces_median(self):
        feat_ids = t.tensor([0, 0, 0, 0, 0], dtype=t.int32)
        feat_mags = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0], dtype=t.float32)
        details, _ = compute_per_feature_percentiles(feat_ids, feat_mags)
        assert abs(details[0].median - 3.0) < 1e-5

    def test_empty_input(self):
        feat_ids = t.tensor([], dtype=t.int32)
        feat_mags = t.tensor([], dtype=t.float32)
        details, shared_edges = compute_per_feature_percentiles(feat_ids, feat_mags)
        assert details == {}
        assert shared_edges is None


# ---------------------------------------------------------------------------
# Enriched dataclass tests
# ---------------------------------------------------------------------------


class TestEnrichedDataclasses:
    """Tests for all new dataclass fields added in the enrichment."""

    def test_per_feature_stat_new_fields(self):
        pfs = PerFeatureStat(
            count=10,
            pct_segments_active=1.0,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.2,
            max_mag=0.9,
            median_mag=0.45,
            percentiles={"p10": 0.25, "p50": 0.45, "p90": 0.85},
        )
        assert pfs.std_mag == 0.1
        assert pfs.min_mag == 0.2
        assert pfs.max_mag == 0.9
        assert pfs.median_mag == 0.45
        assert pfs.percentiles["p50"] == 0.45
        assert pfs.pct_segments_active == 1.0

    def test_per_feature_stat_is_frozen(self):
        pfs = PerFeatureStat(
            count=10,
            pct_segments_active=1.0,
            mean_mag=0.5,
            mean_mag_incl_zeros=0.25,
            std_mag=0.1,
            min_mag=0.2,
            max_mag=0.9,
            median_mag=0.45,
            percentiles={},
        )
        with pytest.raises(AttributeError):
            pfs.count = 20  # ty: ignore[invalid-assignment]

    def test_metadata_stats_fields(self):
        meta = MetadataStats(
            pooling_mode="mean_pool",
            segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
            segment_size=128,
            context_window=1024,
            layer=8,
            n_sae_features=24576,
            generated_at="2025-01-01T00:00:00+00:00",
        )
        assert meta.pooling_mode == "mean_pool"
        assert meta.segment_mode == "fixed_tokens"
        assert meta.segment_size == 128
        assert meta.context_window == 1024
        assert meta.layer == 8
        assert meta.n_sae_features == 24576
        assert "2025" in meta.generated_at

    def test_metadata_stats_is_frozen(self):
        meta = MetadataStats(
            pooling_mode="mean_pool",
            segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
            segment_size=128,
            context_window=1024,
            layer=8,
            n_sae_features=24576,
            generated_at="now",
        )
        with pytest.raises(AttributeError):
            meta.layer = 10  # ty: ignore[invalid-assignment]

    def test_segment_feature_stats_new_fields(self):
        sfs = SegmentFeatureStats(
            n_segments=1000,
            n_features_seen=5000,
            mean_active_per_segment=500.0,
            median_active_per_segment=450,
            p90_active_per_segment=800,
            max_active_per_segment=1000,
            min_active_per_segment=100,
            n_total_sae_features=24576,
            pct_features_active=20.35,
            count_percentiles={"p50": 10, "p90": 100},
            magnitude_stats={"mean": 0.5, "std": 0.1},
            active_per_segment_histogram={},
            active_per_segment_percentiles={},
        )
        assert sfs.n_total_sae_features == 24576
        assert sfs.pct_features_active == 20.35
        assert sfs.count_percentiles["p50"] == 10
        assert sfs.magnitude_stats["mean"] == 0.5

    def test_extrapolation_new_field(self):
        ext = Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=10_000_000,
            estimated_storage_gb=500.0,
            estimated_wall_time_hours=24.5,
        )
        assert ext.estimated_wall_time_hours == 24.5

    def test_extrapolation_is_frozen(self):
        ext = Extrapolation(
            target_tokens=10_000_000_000,
            estimated_docs=10_000_000,
            estimated_storage_gb=500.0,
            estimated_wall_time_hours=24.5,
        )
        with pytest.raises(AttributeError):
            ext.estimated_wall_time_hours = 0.0  # ty: ignore[invalid-assignment]

    def test_extraction_statistics_has_metadata(self):
        meta = MetadataStats(
            pooling_mode="mean_pool",
            segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
            segment_size=128,
            context_window=1024,
            layer=8,
            n_sae_features=24576,
            generated_at="now",
        )
        stats = ExtractionStatistics(
            metadata=meta,
            corpus=CorpusStats(
                n_shards_processed=1,
                n_docs=100,
                n_tokens_gpt2=100_000,
                mean_tokens_per_doc=1000.0,
                median_tokens_per_doc=900,
            ),
            features=SegmentFeatureStats(
                n_segments=1000,
                n_features_seen=5000,
                mean_active_per_segment=500.0,
                median_active_per_segment=450,
                p90_active_per_segment=800,
                max_active_per_segment=1000,
                min_active_per_segment=100,
                n_total_sae_features=24576,
                pct_features_active=20.35,
                count_percentiles={},
                magnitude_stats={},
                active_per_segment_histogram={},
                active_per_segment_percentiles={},
            ),
            storage=StorageStats(
                total_bytes=10_000_000,
                mean_bytes_per_doc=100_000.0,
                mean_bytes_per_token=100.0,
            ),
            per_feature_stats={},
            extrapolation_10b=Extrapolation(
                target_tokens=10_000_000_000,
                estimated_docs=10_000_000,
                estimated_storage_gb=1000.0,
                estimated_wall_time_hours=0.0,
            ),
        )
        assert stats.metadata.pooling_mode == "mean_pool"


# ---------------------------------------------------------------------------
# extract_sparse_features
# ---------------------------------------------------------------------------


class TestExtractSparseFeatures:
    """Tests for _ref_extract_sparse_features (reference implementation)."""

    def test_basic_extraction(self):
        pooled = t.tensor([0.0, 0.5, 0.0, 0.3, 0.8])
        ids, mags = _ref_extract_sparse_features(pooled)
        assert ids.tolist() == [4, 1, 3]
        assert t.allclose(mags, t.tensor([0.8, 0.5, 0.3]))

    def test_all_zeros(self):
        pooled = t.zeros(10)
        ids, mags = _ref_extract_sparse_features(pooled)
        assert ids.numel() == 0
        assert mags.numel() == 0

    def test_output_dtypes(self):
        pooled = t.tensor([0.0, 0.5, 0.0, 0.3, 0.8])
        ids, mags = _ref_extract_sparse_features(pooled)
        assert ids.dtype == t.int32
        assert mags.dtype == t.float32

    def test_on_cpu(self):
        pooled = t.tensor([0.0, 0.5, 0.0, 0.3, 0.8])
        ids, mags = _ref_extract_sparse_features(pooled)
        assert ids.device.type == "cpu"
        assert mags.device.type == "cpu"


# ---------------------------------------------------------------------------
# log_summary — smoke test (should not crash)
# ---------------------------------------------------------------------------


class TestLogSummary:
    def test_does_not_crash(self):
        stats = ExtractionStatistics(
            metadata=MetadataStats(
                pooling_mode="mean_pool",
                segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
                segment_size=128,
                context_window=1024,
                layer=8,
                n_sae_features=24576,
                generated_at="2025-01-01T00:00:00+00:00",
            ),
            corpus=CorpusStats(
                n_shards_processed=1,
                n_docs=100,
                n_tokens_gpt2=100_000,
                mean_tokens_per_doc=1000.0,
                median_tokens_per_doc=900,
            ),
            features=SegmentFeatureStats(
                n_segments=1000,
                n_features_seen=5000,
                mean_active_per_segment=500.0,
                median_active_per_segment=450,
                p90_active_per_segment=800,
                max_active_per_segment=1000,
                min_active_per_segment=100,
                n_total_sae_features=24576,
                pct_features_active=20.35,
                count_percentiles={"p50": 10, "p90": 100},
                magnitude_stats={
                    "mean": 0.5,
                    "std": 0.1,
                    "min": 0.01,
                    "max": 2.0,
                    "percentiles": {},
                },
                active_per_segment_histogram={},
                active_per_segment_percentiles={},
            ),
            storage=StorageStats(
                total_bytes=10_000_000,
                mean_bytes_per_doc=100_000.0,
                mean_bytes_per_token=100.0,
            ),
            per_feature_stats={},
            extrapolation_10b=Extrapolation(
                target_tokens=10_000_000_000,
                estimated_docs=10_000_000,
                estimated_storage_gb=1000.0,
                estimated_wall_time_hours=5.0,
            ),
        )
        log_summary(stats)  # should not raise

    def test_handles_minimal_data(self):
        stats = ExtractionStatistics(
            metadata=MetadataStats(
                pooling_mode="last_token",
                segment_mode="fixed_tokens",  # ty: ignore[invalid-argument-type]
                segment_size=64,
                context_window=512,
                layer=6,
                n_sae_features=24576,
                generated_at="2025-01-01T00:00:00+00:00",
            ),
            corpus=CorpusStats(
                n_shards_processed=1,
                n_docs=10,
                n_tokens_gpt2=1000,
                mean_tokens_per_doc=100.0,
                median_tokens_per_doc=90,
            ),
            features=SegmentFeatureStats(
                n_segments=100,
                n_features_seen=100,
                mean_active_per_segment=50.0,
                median_active_per_segment=45,
                p90_active_per_segment=80,
                max_active_per_segment=100,
                min_active_per_segment=10,
                n_total_sae_features=24576,
                pct_features_active=0.41,
                count_percentiles={},
                magnitude_stats={},
                active_per_segment_histogram={},
                active_per_segment_percentiles={},
            ),
            storage=StorageStats(
                total_bytes=1000, mean_bytes_per_doc=100.0, mean_bytes_per_token=1.0
            ),
            per_feature_stats={},
            extrapolation_10b=Extrapolation(
                target_tokens=10_000_000_000,
                estimated_docs=100_000_000,
                estimated_storage_gb=10.0,
                estimated_wall_time_hours=0.0,
            ),
        )
        log_summary(stats)  # should not raise


# ---------------------------------------------------------------------------
# _get_sae_activations — stop-layer and device co-location
# ---------------------------------------------------------------------------


def test_get_sae_activations_uses_stop_layer_and_sae_device():
    hook_name = "blocks.8.hook_resid_post"
    captured: dict[str, object] = {}

    class _FakeModel:
        def __init__(self) -> None:
            self.param = t.zeros(1)

        def parameters(self):
            yield self.param

        def run_with_cache(self, tokens, **kwargs):
            captured["tokens_device"] = tokens.device
            captured["names_filter"] = kwargs["names_filter"]
            captured["stop_at_layer"] = kwargs["stop_at_layer"]
            acts = t.ones(1, tokens.shape[1], 3, device=tokens.device)
            return None, {hook_name: acts}

    class _FakeSae:
        def __init__(self) -> None:
            self.param = t.zeros(1)

        def parameters(self):
            yield self.param

        def encode(self, activations):
            captured["sae_input_device"] = activations.device
            return t.ones(activations.shape[0], 5, device=activations.device)

    tokens = t.tensor([1, 2, 3], dtype=t.long)
    out = _get_sae_activations(
        _FakeModel(),  # ty: ignore[invalid-argument-type]
        _FakeSae(),  # ty: ignore[invalid-argument-type]
        tokens,
        layer=8,
        hook_name=hook_name,
    )

    assert captured["names_filter"] == [hook_name]
    assert captured["stop_at_layer"] == 9
    assert captured["sae_input_device"] == next(_FakeSae().parameters()).device
    assert out.shape == (3, 5)


# ---------------------------------------------------------------------------
# ActivationType import from sae_resolver
# ---------------------------------------------------------------------------


class TestActivationTypeImport:
    def test_activation_type_enum_accessible(self):
        """Verify ActivationType is importable and has expected members."""
        assert ActivationType.residual.value == "residual"
        assert ActivationType.mlp.value == "mlp"
        assert ActivationType.attention.value == "attention"


# ---------------------------------------------------------------------------
# Raw-coordinate loader path
# ---------------------------------------------------------------------------


class TestRawCoordinateLoader:
    """Verify artifact-supplied raw coordinates reach SAELens unchanged."""

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_legacy_load_passes_release_and_sae_id(self, mock_ht, mock_sae):
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.return_value = MagicMock()

        load_sae_and_model(
            "gpt2",
            layer=8,
            device="cpu",
            sae_release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
        )

        mock_ht.from_pretrained.assert_called_once_with("gpt2", device="cpu")
        mock_sae.from_pretrained.assert_called_once_with(
            release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
            device="cpu",
        )

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_legacy_load_returns_model_sae_tokenizer(self, mock_ht, mock_sae):
        mock_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_model.tokenizer = mock_tokenizer
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae_obj = MagicMock()
        mock_sae.from_pretrained.return_value = mock_sae_obj

        model, sae, tokenizer = load_sae_and_model(
            "gpt2",
            layer=8,
            device="cpu",
            sae_release="gpt2-small-res-jb",
            sae_id="blocks.8.hook_resid_pre",
        )

        assert model is mock_model
        assert sae is mock_sae_obj
        assert tokenizer is mock_tokenizer

    @patch("steering.extract_features.HookedTransformer")
    def test_load_raises_without_release_or_resolved_before_model_io(self, mock_ht):
        with pytest.raises(ValueError, match="Must provide"):
            load_sae_and_model("gpt2", layer=8, device="cpu")
        mock_ht.from_pretrained.assert_not_called()

    @pytest.mark.parametrize(
        "raw_kwargs",
        [
            {"sae_release": "gpt2-small-res-jb"},
            {"sae_id": "blocks.8.hook_resid_pre"},
        ],
    )
    @patch("steering.extract_features.HookedTransformer")
    def test_partial_raw_identity_raises_before_model_io(self, mock_ht, raw_kwargs):
        with pytest.raises(ValueError, match="both sae_release and sae_id"):
            load_sae_and_model(
                "gpt2",
                layer=8,
                device="cpu",
                **raw_kwargs,
            )
        mock_ht.from_pretrained.assert_not_called()

    @patch("steering.extract_features.HookedTransformer")
    def test_resolved_and_raw_identity_conflict_raises_before_model_io(self, mock_ht):
        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            8,
            SAEWidth.W_16K,
        )
        with pytest.raises(ValueError, match="cannot be combined"):
            load_sae_and_model(
                "gemma-2-2b",
                layer=8,
                device="cpu",
                resolved=resolved,
                sae_release=resolved.release,
                sae_id=resolved.sae_id,
            )
        mock_ht.from_pretrained.assert_not_called()


# ---------------------------------------------------------------------------
# Structured config path: GemmaScope via ResolvedSAE
# ---------------------------------------------------------------------------


class TestStructuredConfigPath:
    """Verify the structured sae config path resolves and loads correctly."""

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_resolved_load_passes_resolved_release_and_sae_id(self, mock_ht, mock_sae):
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.return_value = MagicMock()

        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_16K,
        )

        load_sae_and_model(
            "gemma-2-2b",
            layer=12,
            device="cpu",
            resolved=resolved,
        )

        mock_ht.from_pretrained.assert_called_once_with(
            "gemma-2-2b", device="cpu"
        )
        mock_sae.from_pretrained.assert_called_once_with(
            release="gemma-scope-2b-pt-res-canonical",
            sae_id="layer_12/width_16k/canonical",
            device="cpu",
        )

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_opt_in_early_stop_offloads_only_unreachable_tail_before_sae_load(
        self,
        mock_ht,
        mock_sae,
    ):
        events: list[str] = []
        blocks = [MagicMock() for _ in range(5)]
        for index, block in enumerate(blocks):
            block.to.side_effect = lambda device, index=index: events.append(
                f"block_{index}_to_{device}"
            )
        mock_model = MagicMock(blocks=blocks)
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.side_effect = lambda **_kwargs: (
            events.append("load_sae") or MagicMock()
        )
        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            2,
            SAEWidth.W_16K,
        )

        load_sae_and_model(
            "gemma-2-2b",
            layer=2,
            device="cpu",
            resolved=resolved,
            offload_transformer_tail=True,
        )

        assert events == ["block_3_to_cpu", "block_4_to_cpu", "load_sae"]
        for block in blocks[:3]:
            block.to.assert_not_called()

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_default_loader_does_not_offload_transformer_tail(
        self,
        mock_ht,
        mock_sae,
    ):
        blocks = [MagicMock() for _ in range(3)]
        mock_model = MagicMock(blocks=blocks)
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.return_value = MagicMock()
        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            1,
            SAEWidth.W_16K,
        )

        load_sae_and_model(
            "gemma-2-2b",
            layer=1,
            device="cpu",
            resolved=resolved,
        )

        for block in blocks:
            block.to.assert_not_called()

    @settings(max_examples=500, derandomize=True)
    @given(
        st.one_of(
            st.integers(max_value=-1),
            st.booleans(),
            st.floats(allow_nan=True, allow_infinity=True),
            st.text(),
            st.none(),
        )
    )
    def test_offload_rejects_every_invalid_layer(self, invalid_layer: object) -> None:
        with pytest.raises(BeartypeCallHintParamViolation, match="last_required_layer"):
            _offload_transformer_tail_for_early_stop(
                MagicMock(),
                last_required_layer=cast(Any, invalid_layer),
                device="cpu",
            )

    @patch("steering.extract_features.HookedTransformer")
    def test_invalid_offload_layer_fails_before_model_load(self, mock_ht) -> None:
        with pytest.raises(TypeError, match="NonNegativeInt"):
            load_sae_and_model(
                "gemma-2-2b",
                layer=-1,
                device="cpu",
                offload_transformer_tail=True,
            )

        mock_ht.from_pretrained.assert_not_called()

    def test_offload_rejects_model_without_blocks(self):
        mock_model = MagicMock(blocks=None)
        with pytest.raises(ValueError, match="no blocks collection"):
            _offload_transformer_tail_for_early_stop(
                mock_model,
                last_required_layer=NonNegativeInt.parse(0),
                device="cpu",
            )

    def test_offload_rejects_out_of_range_layer(self):
        blocks = [MagicMock() for _ in range(5)]
        mock_model = MagicMock(blocks=blocks)

        with pytest.raises(ValueError, match="outside"):
            _offload_transformer_tail_for_early_stop(
                mock_model,
                last_required_layer=NonNegativeInt.parse(5),
                device="cpu",
            )

        for block in blocks:
            block.to.assert_not_called()

    def test_offload_cpu_device_skips_cuda_cache_clear(self, monkeypatch):
        mock_empty_cache = MagicMock()
        monkeypatch.setattr(t.cuda, "empty_cache", mock_empty_cache)
        blocks = [MagicMock() for _ in range(5)]
        mock_model = MagicMock(blocks=blocks)

        _offload_transformer_tail_for_early_stop(
            mock_model,
            last_required_layer=NonNegativeInt.parse(2),
            device="cpu",
        )

        mock_empty_cache.assert_not_called()
        for block in blocks[3:]:
            block.to.assert_called_once_with("cpu")
        for block in blocks[:3]:
            block.to.assert_not_called()

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_resolved_mlp_passes_correct_release(self, mock_ht, mock_sae):
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.return_value = MagicMock()

        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.mlp,
            5,
            SAEWidth.W_16K,
        )

        load_sae_and_model(
            "gemma-2-2b",
            layer=5,
            device="cpu",
            resolved=resolved,
        )

        mock_sae.from_pretrained.assert_called_once_with(
            release="gemma-scope-2b-pt-mlp-canonical",
            sae_id="layer_5/width_16k/canonical",
            device="cpu",
        )

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_resolved_attention_passes_correct_release(self, mock_ht, mock_sae):
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.return_value = MagicMock()

        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.attention,
            12,
            SAEWidth.W_16K,
        )

        load_sae_and_model(
            "gemma-2-2b",
            layer=12,
            device="cpu",
            resolved=resolved,
        )

        mock_sae.from_pretrained.assert_called_once_with(
            release="gemma-scope-2b-pt-att-canonical",
            sae_id="layer_12/width_16k/canonical",
            device="cpu",
        )

    @patch("steering.extract_features.SAE")
    @patch("steering.extract_features.HookedTransformer")
    def test_resolved_with_l0_passes_specific_sae_id(self, mock_ht, mock_sae):
        mock_model = MagicMock()
        mock_model.tokenizer = MagicMock()
        mock_ht.from_pretrained.return_value = mock_model
        mock_sae.from_pretrained.return_value = MagicMock()

        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_65K,
            l0=62,
        )

        load_sae_and_model(
            "gemma-2-2b",
            layer=12,
            device="cpu",
            resolved=resolved,
        )

        mock_sae.from_pretrained.assert_called_once_with(
            release="gemma-scope-2b-pt-res",
            sae_id="layer_12/width_65k/average_l0_62",
            device="cpu",
        )

    def test_resolved_hook_name_for_residual(self):
        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.residual,
            12,
            SAEWidth.W_16K,
        )
        assert resolved.hook_name == "blocks.12.hook_resid_post"

    def test_resolved_hook_name_for_mlp(self):
        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.mlp,
            5,
            SAEWidth.W_16K,
        )
        assert resolved.hook_name == "blocks.5.hook_mlp_out"

    def test_resolved_hook_name_for_attention(self):
        resolved = resolve_sae(
            "gemma-2-2b",
            SAERelease.gemmascope,
            ActivationType.attention,
            12,
            SAEWidth.W_16K,
        )
        assert resolved.hook_name == "blocks.12.attn.hook_z"


# ---------------------------------------------------------------------------
# Extraction SAE config validation
# ---------------------------------------------------------------------------


class TestExtractionSaeConfig:
    def test_removed_raw_key_fails_with_migration_guidance(self):
        cfg = OmegaConf.create(
            {
                "sae_id": "gpt2-small-res-jb",
                "sae": {
                    "release": "gemmascope",
                    "num_features": "16k",
                    "l0": None,
                },
            }
        )
        with pytest.raises(ValueError, match=r"sae_id.*removed.*sae\.release"):
            _resolve_extraction_sae(
                cfg,
                model_name="gemma-2-2b",
                activation_type=ActivationType.residual,
                layer=8,
            )

    def test_missing_structured_config_fails(self):
        cfg = OmegaConf.create({"sae": None})
        with pytest.raises(ValueError, match="structured sae config"):
            _resolve_extraction_sae(
                cfg,
                model_name="gemma-2-2b",
                activation_type=ActivationType.residual,
                layer=8,
            )

    def test_structured_config_resolves_exact_identity(self):
        cfg = OmegaConf.create(
            {
                "sae": {
                    "release": "gemmascope",
                    "num_features": "16k",
                    "l0": None,
                }
            }
        )
        resolved = _resolve_extraction_sae(
            cfg,
            model_name="gemma-2-2b",
            activation_type=ActivationType.residual,
            layer=8,
        )
        assert resolved.release == "gemma-scope-2b-pt-res-canonical"
        assert resolved.sae_id == "layer_8/width_16k/canonical"
        assert resolved.hook_name == "blocks.8.hook_resid_post"


# ---------------------------------------------------------------------------
# pool_segments_batched_gpu correctness tests
# ---------------------------------------------------------------------------


class TestPoolSegmentsBatchedGpu:
    """Compare pool_segments_batched_gpu against per-segment pool_segment()."""

    def _make_sae_acts(self, seq_len=32, n_features=64, seed=42):
        t.manual_seed(seed)
        return t.rand(seq_len, n_features)

    def _segments(self, seq_len=32, seg_size=8):
        return [(i, min(i + seg_size, seq_len)) for i in range(0, seq_len, seg_size)]

    def test_last_token_matches(self):
        sae_acts = self._make_sae_acts()
        segments = self._segments()
        batched = pool_segments_batched_gpu(sae_acts, segments, PoolingMode.last_token)
        for i, (start, end) in enumerate(segments):
            ref = _ref_pool_segment(sae_acts, start, end, PoolingMode.last_token)
            assert t.allclose(batched[i], ref, atol=1e-6), f"Segment {i} mismatch"

    def test_max_pool_matches(self):
        sae_acts = self._make_sae_acts()
        segments = self._segments()
        batched = pool_segments_batched_gpu(sae_acts, segments, PoolingMode.max_pool)
        for i, (start, end) in enumerate(segments):
            ref = _ref_pool_segment(sae_acts, start, end, PoolingMode.max_pool)
            assert t.allclose(batched[i], ref, atol=1e-6), f"Segment {i} mismatch"

    def test_mean_pool_matches(self):
        sae_acts = self._make_sae_acts()
        segments = self._segments()
        batched = pool_segments_batched_gpu(sae_acts, segments, PoolingMode.mean_pool)
        for i, (start, end) in enumerate(segments):
            ref = _ref_pool_segment(sae_acts, start, end, PoolingMode.mean_pool)
            assert t.allclose(batched[i], ref, atol=1e-6), f"Segment {i} mismatch"

    def test_short_last_segment(self):
        """Fixed-tokens with a short last segment (tail of sequence)."""
        sae_acts = self._make_sae_acts(seq_len=50)
        segments = [(0, 20), (20, 40), (40, 50)]  # last segment is shorter
        for mode in PoolingMode:
            batched = pool_segments_batched_gpu(
                sae_acts,
                segments,
                mode,
                SegmentMode.fixed_tokens,
            )
            for i, (start, end) in enumerate(segments):
                ref = _ref_pool_segment(sae_acts, start, end, mode)
                assert t.allclose(batched[i], ref, atol=1e-6)

    def test_output_shape(self):
        sae_acts = self._make_sae_acts(seq_len=40, n_features=100)
        segments = self._segments(seq_len=40, seg_size=10)
        for mode in PoolingMode:
            result = pool_segments_batched_gpu(sae_acts, segments, mode)
            assert result.shape == (len(segments), 100)


# ---------------------------------------------------------------------------
# extract_sparse_gpu correctness tests
# ---------------------------------------------------------------------------


class TestExtractSparseGpu:
    """Compare extract_sparse_gpu against per-segment extract_sparse_features()."""

    def test_matches_per_segment(self):
        t.manual_seed(0)
        pooled = t.rand(8, 64)
        # Some zeros
        pooled[pooled < 0.3] = 0.0

        gpu_ids, gpu_mags = extract_sparse_gpu(pooled)

        for i in range(8):
            ref_ids, ref_mags = _ref_extract_sparse_features(pooled[i])
            # extract_sparse_features sorts descending; extract_sparse_gpu doesn't sort
            # Compare as sets
            gpu_set = set(
                zip(gpu_ids[i].tolist(), [round(m, 6) for m in gpu_mags[i].tolist()], strict=True)
            )
            ref_set = set(
                zip(ref_ids.tolist(), [round(m, 6) for m in ref_mags.tolist()], strict=True)
            )
            assert gpu_set == ref_set, f"Segment {i} feature mismatch"

    def test_all_zeros(self):
        pooled = t.zeros(4, 32)
        gpu_ids, gpu_mags = extract_sparse_gpu(pooled)
        assert len(gpu_ids) == 4
        for i in range(4):
            assert len(gpu_ids[i]) == 0
            assert len(gpu_mags[i]) == 0

    def test_all_nonzero(self):
        t.manual_seed(1)
        pooled = t.rand(3, 16) + 0.1  # All positive
        gpu_ids, gpu_mags = extract_sparse_gpu(pooled)
        for i in range(3):
            assert len(gpu_ids[i]) == 16  # All features active

    def test_output_dtypes(self):
        pooled = t.tensor([[0.0, 1.5, 0.0, 3.2]])
        gpu_ids, gpu_mags = extract_sparse_gpu(pooled)
        assert gpu_ids[0].dtype == t.int32
        assert gpu_mags[0].dtype == t.float32

    def test_bf16_input(self):
        """extract_sparse_gpu handles bfloat16 input correctly."""
        t.manual_seed(7)
        pooled = t.rand(4, 32, dtype=t.bfloat16)
        pooled[pooled < 0.5] = 0.0
        gpu_ids, gpu_mags = extract_sparse_gpu(pooled)
        for i in range(4):
            assert gpu_ids[i].dtype == t.int32
            assert gpu_mags[i].dtype == t.float32
            # All returned mags should be > 0
            if len(gpu_mags[i]) > 0:
                assert (gpu_mags[i] > 0).all()

    def test_single_segment(self):
        """Single segment works correctly."""
        pooled = t.tensor([[0.0, 0.5, 0.0, 0.3]])
        gpu_ids, gpu_mags = extract_sparse_gpu(pooled)
        assert len(gpu_ids) == 1
        assert set(gpu_ids[0].tolist()) == {1, 3}


class TestShardDataContextIndependentField:
    def _minimal(self, **overrides) -> ShardData:
        defaults = {
            "offsets": t.tensor([0], dtype=t.int64),
            "feature_ids": t.tensor([], dtype=t.int32),
            "feature_mags": t.tensor([], dtype=t.float32),
            "doc_ids": t.tensor([], dtype=t.int32),
            "segment_indices": t.tensor([], dtype=t.int32),
            "segment_positions": t.tensor([], dtype=t.int32),
            "doc_token_counts": t.tensor([], dtype=t.int32),
            "n_docs": 0,
            "n_segments": 0,
            "layer": 8,
            "context_window": 1024,
            "pooling": "mean_pool",
            "segment_size": 128,
            "segment_char_starts": t.tensor([], dtype=t.int32),
            "segment_char_ends": t.tensor([], dtype=t.int32),
            "segment_mode": SegmentMode.fixed_tokens,
        }
        defaults.update(overrides)
        return ShardData(**defaults)  # ty: ignore[invalid-argument-type]

    def test_default_false(self):
        sd = self._minimal()
        assert sd.context_independent_features is False

    def test_set_true(self):
        sd = self._minimal(context_independent_features=True)
        assert sd.context_independent_features is True

    def test_to_dict_includes_flag_true(self):
        sd = self._minimal(context_independent_features=True)
        d = sd.to_dict()
        assert d["context_independent_features"] is True

    def test_to_dict_includes_flag_false(self):
        sd = self._minimal()
        d = sd.to_dict()
        assert d["context_independent_features"] is False

    def test_from_dict_round_trip_true(self):
        sd = self._minimal(context_independent_features=True)
        sd2 = ShardData.from_dict(sd.to_dict())
        assert sd2.context_independent_features is True

    def test_from_dict_defaults_false_if_absent(self):
        """Legacy dicts without the key should default False after from_dict."""
        sd = self._minimal()
        d = sd.to_dict()
        del d["context_independent_features"]
        sd2 = ShardData.from_dict(d)
        assert sd2.context_independent_features is False


# ---------------------------------------------------------------------------
# extract_features.yaml Hydra config
# ---------------------------------------------------------------------------


class TestExtractFeaturesConfig:
    def test_context_independent_default_false(self):
        config_dir = str(Path(__file__).parent.parent.parent / "configs" / "extraction")
        GlobalHydra.instance().clear()
        with initialize_config_dir(config_dir=config_dir, version_base=None):
            cfg = compose(config_name="extract_features")
        assert hasattr(cfg, "context_independent_features")
        assert cfg.context_independent_features is False


class TestCoerceConfigL0:
    """`_coerce_config_l0` bridges OmegaConf's untyped ``sae.l0`` to
    ``resolve_sae``'s ``int | L0Tier | None`` contract.

    Regression: the structured-SAE path coerced ``release`` and
    ``num_features`` to enums but passed ``sae.l0`` through as a raw
    string, so the documented gemma-scope-2 config (``sae.l0: medium``)
    crashed inside ``_resolve_gemmascope_2`` with
    ``AttributeError: 'str' object has no attribute 'value'``.
    """

    def test_gemmascope2_string_tier_resolves_end_to_end(self):
        l0 = _coerce_config_l0(SAERelease.gemmascope_2, "medium")
        assert l0 is L0Tier.medium

        resolved = resolve_sae(
            model="gemma-3-4b-it",
            release=SAERelease.gemmascope_2,
            activation_type=ActivationType.residual,
            layer=22,
            width=SAEWidth.W_65K,
            l0=l0,
        )
        assert resolved.sae_id == "layer_22_width_65k_l0_medium"
        assert resolved.release == "gemma-scope-2-4b-it-res"

    def test_gemmascope2_already_enum_passthrough(self):
        assert (
            _coerce_config_l0(SAERelease.gemmascope_2, L0Tier.big)
            is L0Tier.big
        )

    def test_gemmascope2_invalid_tier_raises_valueerror(self):
        with pytest.raises(ValueError):
            _coerce_config_l0(SAERelease.gemmascope_2, "enormous")

    def test_v1_gemmascope_int_l0_passthrough_unchanged(self):
        """v1 uses an int average-L0; it must NOT be coerced to L0Tier."""
        assert _coerce_config_l0(SAERelease.gemmascope, 176) == 176

    def test_v1_gemmascope_none_l0_passthrough_unchanged(self):
        """v1 canonical (l0=None) must stay None."""
        assert _coerce_config_l0(SAERelease.gemmascope, None) is None

    def test_v1_gemmascope_string_l0_rejected_fast(self):
        """A v1 ``gemmascope`` string l0 is a misconfiguration (v1 uses an
        int average-L0). Fail fast with a clear ValueError instead of
        leaking an untyped str into resolve_sae."""
        with pytest.raises(ValueError, match="sae.l0"):
            _coerce_config_l0(SAERelease.gemmascope, "176")

    def test_invalid_l0_type_rejected_fast(self):
        """Non-int / non-str / non-L0Tier / non-None l0 is rejected with a
        clear ValueError (was an opaque AttributeError before the fix)."""
        with pytest.raises(ValueError, match="sae.l0"):
            _coerce_config_l0(SAERelease.gemmascope_2, 3.5)

    def test_real_omegaconf_string_l0_resolves(self):
        """OmegaConf returns a native str for a YAML scalar; the helper must
        accept it and resolve end-to-end (the exact production path)."""
        cfg = OmegaConf.create(
            {"sae": {"release": "gemmascope_2", "l0": "medium"}}
        )
        l0 = _coerce_config_l0(
            SAERelease(cfg.sae.release), cfg.sae.get("l0")
        )
        assert l0 is L0Tier.medium

    def test_bool_is_int_subclass_passthrough_is_characterized(self):
        """bool is an int subclass; the helper passes it through for BOTH
        releases (documented behavior). resolve_sae then rejects it for
        gemma-scope-2."""
        assert _coerce_config_l0(SAERelease.gemmascope, True) is True
        assert _coerce_config_l0(SAERelease.gemmascope_2, True) is True
        with pytest.raises(ValueError):
            resolve_sae(
                model="gemma-3-4b-it",
                release=SAERelease.gemmascope_2,
                activation_type=ActivationType.residual,
                layer=22,
                width=SAEWidth.W_65K,
                l0=_coerce_config_l0(SAERelease.gemmascope_2, True),
            )

    def test_v1_with_l0tier_passes_through_then_resolver_rejects(self):
        """v1 + an L0Tier is a misconfig; the helper passes it through and
        resolve_sae fails fast with a clear v1-vs-v2 message."""
        l0 = _coerce_config_l0(SAERelease.gemmascope, L0Tier.medium)
        assert l0 is L0Tier.medium
        with pytest.raises(ValueError, match="v1|average L0|L0Tier"):
            resolve_sae(
                model="gemma-2-2b",
                release=SAERelease.gemmascope,
                activation_type=ActivationType.residual,
                layer=8,
                width=SAEWidth.W_16K,
                l0=l0,
            )


class TestLoadSaeAndModelNoDtypeParam:
    """Regression guard: commit 2's `model_dtype` param was reverted. The
    gemma-3 OOM is fixed at the root cause (TL offloaded to CPU before the HF
    Gemma-3 load inside _build_nnterp_gemma3), NOT by loading the shared TL
    model in bf16 (which silently broke the gemma-2 nnterp verify gate)."""

    def test_signature_has_no_model_dtype_param(self):
        sig = inspect.signature(load_sae_and_model)
        assert "model_dtype" not in sig.parameters, (
            "load_sae_and_model must not take model_dtype again — see "
            "spec B1; the OOM fix lives in _build_nnterp_gemma3."
        )

    def test_main_load_sae_and_model_calls_omit_model_dtype(self):
        """The load_sae_and_model(...) call in main() must NOT forward
        model_dtype (the param was reverted). Backend-builder calls
        legitimately keep model_dtype=model_dtype and must not be matched."""
        func_def = _extract_features_func_def("main")
        found = 0
        for node in ast.walk(func_def):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "load_sae_and_model"
            ):
                continue
            assert all(kw.arg != "model_dtype" for kw in node.keywords), (
                f"main() must not forward model_dtype into "
                f"load_sae_and_model; offending call:\n{ast.dump(node)}"
            )
            found += 1
        assert found == 1, (
            f"expected exactly one load_sae_and_model call in main(), found {found}"
        )

    def test_extraction_resolver_retains_coerce_config_l0_wiring(self):
        func_def = _extract_features_func_def("_resolve_extraction_sae")
        calls = [
            node
            for node in ast.walk(func_def)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id == "_coerce_config_l0"
        ]
        assert calls, (
            "the sound commit-1 sae.l0->L0Tier coercion must be retained."
        )

    def test_coerce_config_l0_still_callable(self):
        assert callable(_coerce_config_l0)


class TestBuildNnterpGemma3OffloadsTL:
    """Root-cause gemma-3 OOM fix: the fp32 TL companion must be offloaded
    to CPU (+empty_cache) BEFORE the HF Gemma-3 load, mirroring
    _build_nnterp_gemma2. Otherwise TL (fp32, ~16GB) double-residents with
    the bf16 HF build and OOMs a 24GB GPU. Spec C1."""

    def _run(self, tl_model):
        calls: list[tuple] = []
        with (
            patch(
                "steering.extract_features.load_gemma3_text_decoder",
                side_effect=lambda *a, **k: calls.append(("hf_from_pretrained",))
                or MagicMock(),
            ),
            patch("steering.extract_features.StandardizedTransformer"),
            patch(
                "steering.extract_features.resolve_tokenizer",
                return_value=None,
            ),
            patch(
                "steering.extract_features.t.cuda.empty_cache",
                side_effect=lambda *a, **k: calls.append(("empty_cache",)),
            ),
        ):
            _build_nnterp_gemma3(
                model_name="gemma-3-4b-it",
                tl_model=tl_model,
                device=t.device("cpu"),
                model_dtype=t.bfloat16,
            )
        return calls

    def test_tl_offloaded_to_cpu_before_hf_load(self):
        tl_model = MagicMock()
        tl_model.to.side_effect = lambda *a, **k: None
        calls = self._run(tl_model)
        tl_model.to.assert_any_call("cpu")
        names = [c[0] for c in calls]
        assert "empty_cache" in names and "hf_from_pretrained" in names
        assert names.index("empty_cache") < names.index("hf_from_pretrained")

    def test_tl_to_called_before_hf_from_pretrained(self):
        order: list[str] = []
        tl_model = MagicMock()
        tl_model.to.side_effect = lambda *a, **k: order.append("tl_to")
        with (
            patch(
                "steering.extract_features.load_gemma3_text_decoder",
                side_effect=lambda *a, **k: order.append("hf") or MagicMock(),
            ),
            patch("steering.extract_features.StandardizedTransformer"),
            patch(
                "steering.extract_features.resolve_tokenizer",
                return_value=None,
            ),
            patch("steering.extract_features.t.cuda.empty_cache"),
        ):
            _build_nnterp_gemma3(
                model_name="gemma-3-4b-it",
                tl_model=tl_model,
                device=t.device("cpu"),
                model_dtype=t.bfloat16,
            )
        assert order.index("tl_to") < order.index("hf")

    def test_tl_model_none_is_safe(self):
        calls = self._run(None)
        assert ("hf_from_pretrained",) in calls


class TestF65kConfigResolution:
    """All gemma-3 l22 f65k configs (incl. the new validate config) resolve
    to the documented effective values. Spec C3/C4."""

    @staticmethod
    def _compose(name):
        cfg_dir = str(
            Path(__file__).resolve().parents[2] / "configs" / "extraction"
        )
        GlobalHydra.instance().clear()
        with initialize_config_dir(version_base=None, config_dir=cfg_dir):
            return compose(config_name=name)

    def test_validate_config_resolves(self):
        cfg = self._compose("validate_features_gemma3_l22_f65k")
        assert cfg.model == "gemma-3-4b-it"
        assert cfg.sae_release is None and cfg.sae_id is None
        assert cfg.n_spot_check == 0
        assert cfg.parquet_dir is None
        assert cfg.resolve_parquet_default is True
        assert cfg.feature_dir.endswith(
            "s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-it/base"
        )
        # sae_release/sae_id stay null via base inheritance (no GPU path)
        assert cfg.sae_release is None and cfg.sae_id is None
        assert cfg.model_dtype == "bfloat16"

    def test_extract_config_resolves(self):
        cfg = self._compose("extract_features_gemma3_l22_f65k")
        assert "sae_id" not in cfg
        assert cfg.model == "gemma-3-4b-it"
        assert cfg.layer == 22
        assert cfg.sae.release == "gemmascope_2"
        assert str(cfg.sae.num_features) == "65k"
        assert cfg.sae.l0 == "medium"
        assert int(cfg.gpu_batch_size) == 2
        assert int(cfg.cpu_batch_size) == 8
        assert cfg.model_dtype == "bfloat16"

    def test_compute_statistics_config_resolves(self):
        cfg = self._compose("compute_statistics_gemma3_l22_f65k")
        assert str(cfg.sae_width) == "65k"
        assert int(cfg.n_sae_features) == 65536
        assert cfg.device == "cuda"
        assert cfg.exclude_last_shard is True
        assert cfg.feature_dir.endswith("f65k/m_gemma-3-4b-it/base")

    def test_percentiles_config_resolves(self):
        cfg = self._compose("precompute_percentiles_gemma3_l22_f65k")
        assert str(cfg.sae_width) == "65k"
        assert cfg.resume is True
        assert cfg.device == "cuda"
        assert cfg.feature_dir.endswith(
            "s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-it/base"
        )

    def test_token_boundaries_config_resolves(self):
        cfg = self._compose("precompute_token_boundaries_gemma3_l22_f65k")
        assert str(cfg.sae_width) == "65k"
        assert cfg.resume is True
        assert cfg.model == "gemma-3-4b-it"
        assert cfg.feature_dir.endswith(
            "s128/l22/a_res/p_max_pool/f65k/m_gemma-3-4b-it/base"
        )

    def test_packed_rows_config_resolves(self):
        cfg = self._compose("precompute_packed_rows_gemma3_l22_f65k")
        assert str(cfg.sae_width) == "65k"
        assert cfg.context_len == 1024
        assert cfg.max_shards is None
        assert "decoder_segment_size" not in cfg
        assert cfg.resume is True


class TestGateModelWeightLoadErrorPropagation:
    """F7: compile/rowpack gates must NOT re-type a propagating ModelWeightLoadError.

    If build_native_model raises ModelWeightLoadError (e.g. gemma-3 weight load
    failure per PR #549), the gate must let it propagate unchanged rather than
    re-wrapping it in CompileGateError / RowPackGateError with a misleading
    "typically OOM" message.
    """

    def _raise_model_weight_load_error(self, *a, **k):
        raise ef.ModelWeightLoadError("boom")

    def test_compile_gate_does_not_retype_model_weight_load_error(self, monkeypatch):
        native_model = MagicMock()
        native_model.model_name = "gpt2"
        native_model.device = t.device("cpu")
        native_model.truncated_layer = 0
        sae = MagicMock()
        probe_input_ids = t.zeros(1, 4, dtype=t.long)

        monkeypatch.setattr(ef, "build_native_model", self._raise_model_weight_load_error)

        with pytest.raises(ef.ModelWeightLoadError):
            ef._native_compile_gate(
                native_model,
                sae,
                probe_input_ids=probe_input_ids,
                model_dtype=t.bfloat16,
            )

    def test_rowpack_gate_does_not_retype_model_weight_load_error(self, monkeypatch):
        native_model = MagicMock()
        native_model.model_name = "gpt2"
        native_model.device = t.device("cpu")
        native_model.truncated_layer = 0
        sae = MagicMock()
        probe = ef.RowpackGateProbe(
            token_id_lists=[[1, 2, 3]],
            segments_by_sequence=[[(0, 3)]],
            keep_masks_by_sequence=[None],
            context_independent_features=False,
        )

        monkeypatch.setattr(ef, "build_native_model", self._raise_model_weight_load_error)

        with pytest.raises(ef.ModelWeightLoadError):
            ef._native_rowpack_gate(
                native_model,
                sae,
                probe=probe,
                pooling=ef.PoolingMode.max_pool,
                model_dtype=t.bfloat16,
            )
