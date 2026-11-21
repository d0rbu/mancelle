"""Tests for the indentation bug fix in the Stage 3 generate loop, and
for the precomputed percentile lookup table optimization.

Verifies that _generate_completions processes every (feature, doc) pair —
not just the last doc per feature — and that the results are complete.

Also verifies that the optimized _activation_to_percentile (using
PercentileLookup + bisect) produces results exactly identical to the original
linear-scan implementation across a wide range of activation values and edge
cases.
"""

from __future__ import annotations

import random
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import torch as t

from nanochat.engine import KVCache
from scripts.eval.completion_steering_eval import _generate_completions
from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.conditional.thresholding import ThresholdConfig, ThresholdType
from steering.eval.completion_judge import CompletionJudgeClient, JudgementScore
from steering.eval.completion_steering_eval import (
    EvalMode,
    FeatureSelection,
    _activation_to_percentile,
    _build_percentile_lookup,
    _vectorized_percentile,
    scan_validation_features_vectorized,
)
from steering.eval.multiseg_generator import (
    CompletionSample,
    DocumentCompletion,
    _decode_from_prefill,
    _prefill_condition,
    ablate_feature,
    build_multiseg_inputs,
    generate_completion,
    generate_for_document,
    prepare_segments,
)
from steering.feature_io import save_features
from steering.stats_types import PerFeatureStat
from tests.padded_encoder_reference import pack_dense_encoder_output_reference
from tests.steering._completion_steering_reference import (
    scan_results_to_dict,
    scan_validation_features,
)

# Passthrough config: segment_topk with k large enough to keep all features.
_PASSTHROUGH_CFG = ThresholdConfig(ThresholdType.segment_topk, 99999)


def _load_index_shard(
    shard_path: Path, shard_stem: str
) -> dict[int, list[SegmentFeatures]]:
    """Load a fixture through a fully initialized current FeatureIndex."""
    return dict(FeatureIndex(str(shard_path.parent))._load_shard(shard_stem))


# ---------------------------------------------------------------------------
# Reference implementation (moved from production code for test comparison)
# ---------------------------------------------------------------------------


def _activation_to_percentile_original(
    activation: float,
    percentiles: dict[str, float],
) -> float:
    """Original linear-scan implementation for verifying optimized version.

    Convert a raw activation value to an approximate percentile (0-1 scale).
    Uses linear interpolation between the nearest percentile keys.
    """
    pct_pairs: list[tuple[float, float]] = []
    for key, val in percentiles.items():
        if key.startswith("p") and key[1:].isdigit():
            pct_pairs.append((float(key[1:]) / 100.0, val))
    if not pct_pairs:
        raise ValueError(
            f"No valid percentile keys (p1..p99) found in percentiles dict "
            f"(keys: {list(percentiles.keys())[:10]})"
        )
    pct_pairs.sort(key=lambda x: x[0])

    lo_pct, lo_val = pct_pairs[0]
    hi_pct, hi_val = pct_pairs[-1]

    if activation <= lo_val:
        return lo_pct
    if activation >= hi_val:
        return hi_pct

    for i in range(len(pct_pairs) - 1):
        p0, v0 = pct_pairs[i]
        p1, v1 = pct_pairs[i + 1]
        if v0 <= activation <= v1:
            if v1 == v0:
                return p0
            frac = (activation - v0) / (v1 - v0)
            return p0 + frac * (p1 - p0)

    return hi_pct


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_FEATURES = 3
DOCS_PER_FEATURE = 5
N_DOC_TEXTS = 100  # large enough that no doc_idx is out of range


def _make_selections(
    n_features: int = N_FEATURES,
    docs_per_feature: int = DOCS_PER_FEATURE,
) -> list[FeatureSelection]:
    """Build FeatureSelection objects with predictable doc lists."""
    selections = []
    for fid in range(n_features):
        selected_docs = list(
            range(fid * docs_per_feature, (fid + 1) * docs_per_feature)
        )
        selections.append(
            FeatureSelection(
                feature_id=fid,
                neuronpedia_description=f"Feature {fid}",
                n_docs_activated=docs_per_feature,
                selected_docs=selected_docs,
            )
        )
    return selections


def _make_fake_segment(seg_idx: int = 0) -> MagicMock:
    """Return a minimal SegmentFeatures-like mock."""
    seg = MagicMock()
    seg.segment_idx = seg_idx
    seg.char_start = seg_idx * 512
    seg.char_end = (seg_idx + 1) * 512
    return seg


def _make_feature_index_mock(n_segs_per_doc: int = 3) -> MagicMock:
    """Return a FeatureIndex mock whose lookup always returns n_segs_per_doc segments."""
    idx = MagicMock()
    idx.lookup.side_effect = lambda shard_stem, doc_idx: [
        _make_fake_segment(i) for i in range(n_segs_per_doc)
    ]
    return idx


def _make_doc_completion(
    feature_id: int, doc_idx: int, n_samples: int
) -> DocumentCompletion:
    """Build a DocumentCompletion with the given number of samples."""
    samples = [
        CompletionSample(
            condition="ablated",
            text=f"text_{feature_id}_{doc_idx}_{i}",
            seed=i,
            bucket_index=None,
        )
        for i in range(n_samples)
    ]
    return DocumentCompletion(
        doc_idx=doc_idx,
        feature_id=feature_id,
        pre_segment_text="context",
        ground_truth_segment="truth",
        target_segment_idx=2,
        samples=samples,
    )


def _common_kwargs(
    selections: list[FeatureSelection],
    feature_index: MagicMock,
    n_samples: int = 5,
) -> dict:
    """Shared keyword arguments for _generate_completions."""
    # Model mock needs dec_config.sequence_len for the context-window filter
    model = MagicMock()
    model.dec_config.sequence_len = 2048

    # Tokenizer mock needs decode/encode for the context-window filter
    tokenizer = MagicMock()
    tokenizer.decode.return_value = "x"  # short BOS prefix
    tokenizer.encode.return_value = [2, 10, 11]  # short prefix (fits in ctx)

    return {
        "selections": selections,
        "target_segment_maps": None,
        "doc_texts": [f"doc_text_{i}" for i in range(N_DOC_TEXTS)],
        "feature_index": feature_index,
        "val_shard_stem": "shard_0000",
        "model": model,
        "tokenizer": tokenizer,
        "mode": EvalMode.ABLATION,
        "boundaries": t.zeros(10),
        "feature_cfg": _PASSTHROUGH_CFG,
        "precomputed_thresholds": None,
        "device": t.device("cpu"),
        "num_buckets": 8,
        "samples_per_condition": n_samples,
        "temperature": 0.7,
        "top_k_sampling": 50,
        "bos_id": 2,
    }


# ---------------------------------------------------------------------------
# TestIndentationFix
# ---------------------------------------------------------------------------


class TestIndentationFix:
    """Verify the loop processes all (feature, doc) pairs after the indent fix."""

    def test_all_docs_processed(self) -> None:
        """generate_for_document must be called exactly N_FEATURES * DOCS_PER_FEATURE times."""
        selections = _make_selections()
        feature_index = _make_feature_index_mock()

        captured_calls: list[tuple[int, int]] = []

        def fake_generate_for_document(**kwargs):
            fid = kwargs["feature_id"]
            doc_idx = kwargs["doc_idx"]
            captured_calls.append((fid, doc_idx))
            return _make_doc_completion(fid, doc_idx, kwargs["samples_per_condition"])

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate_for_document,
        ):
            result = _generate_completions(**_common_kwargs(selections, feature_index))

        expected_total = N_FEATURES * DOCS_PER_FEATURE
        assert len(result) == expected_total, (
            f"Expected {expected_total} completions, got {len(result)}"
        )
        assert len(captured_calls) == expected_total, (
            f"generate_for_document called {len(captured_calls)} times, "
            f"expected {expected_total}"
        )

        # Verify every expected (feature_id, doc_idx) pair was processed
        expected_pairs = {
            (fid, doc_idx)
            for fid in range(N_FEATURES)
            for doc_idx in range(fid * DOCS_PER_FEATURE, (fid + 1) * DOCS_PER_FEATURE)
        }
        actual_pairs = set(captured_calls)
        assert actual_pairs == expected_pairs, (
            f"Missing pairs: {expected_pairs - actual_pairs}"
        )

    def test_doc_idx_order(self) -> None:
        """Doc indices must be passed to generate_for_document in selected_docs order."""
        selections = _make_selections()
        feature_index = _make_feature_index_mock()

        calls_by_feature: dict[int, list[int]] = {}

        def fake_generate_for_document(**kwargs):
            fid = kwargs["feature_id"]
            doc_idx = kwargs["doc_idx"]
            calls_by_feature.setdefault(fid, []).append(doc_idx)
            return _make_doc_completion(fid, doc_idx, kwargs["samples_per_condition"])

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate_for_document,
        ):
            _generate_completions(**_common_kwargs(selections, feature_index))

        for sel in selections:
            observed = calls_by_feature[sel.feature_id]
            assert observed == sel.selected_docs, (
                f"Feature {sel.feature_id}: expected doc order {sel.selected_docs}, "
                f"got {observed}"
            )

    def test_completions_count(self) -> None:
        """With samples_per_condition=5 and 2 ablation conditions each DocumentCompletion
        returned by generate_for_document is stored; 3 features * 5 docs = 15 total."""
        n_features = N_FEATURES
        docs_per_feature = DOCS_PER_FEATURE
        samples_per_condition = 5
        # Ablation mode has 2 conditioned variants: real and ablated.
        n_conditions = 2
        expected_completions = n_features * docs_per_feature

        selections = _make_selections(n_features, docs_per_feature)
        feature_index = _make_feature_index_mock()

        def fake_generate_for_document(**kwargs):
            fid = kwargs["feature_id"]
            doc_idx = kwargs["doc_idx"]
            spc = kwargs["samples_per_condition"]
            # Return a completion with samples for all ablation conditions.
            all_samples = [
                CompletionSample(
                    condition=cond,
                    text=f"text_{fid}_{doc_idx}_{cond}_{i}",
                    seed=i,
                    bucket_index=None,
                )
                for cond in ("real", "ablated")
                for i in range(spc)
            ]
            return DocumentCompletion(
                doc_idx=doc_idx,
                feature_id=fid,
                pre_segment_text="context",
                ground_truth_segment="truth",
                target_segment_idx=2,
                samples=all_samples,
            )

        with patch(
            "scripts.eval.completion_steering_eval.generate_for_document",
            side_effect=fake_generate_for_document,
        ):
            result = _generate_completions(
                **_common_kwargs(
                    selections, feature_index, n_samples=samples_per_condition
                )
            )

        assert len(result) == expected_completions, (
            f"Expected {expected_completions} DocumentCompletion objects, got {len(result)}"
        )

        expected_samples_per_completion = n_conditions * samples_per_condition
        for comp in result:
            assert len(comp.samples) == expected_samples_per_completion, (
                f"DocumentCompletion for feature={comp.feature_id} doc={comp.doc_idx} "
                f"has {len(comp.samples)} samples, expected {expected_samples_per_completion}"
            )


# ---------------------------------------------------------------------------
# Helpers for TestPercentileLookupOptimization
# ---------------------------------------------------------------------------

# Representative percentiles dict: p1..p99 with linearly spaced values 0.1..9.9
_REPRESENTATIVE_PERCENTILES: dict[str, float] = {
    f"p{i}": float(i) * 0.1 for i in range(1, 100)
}


def _make_per_feature_stat(percentiles: dict[str, float]) -> PerFeatureStat:
    """Build a minimal PerFeatureStat with the given percentiles."""
    return PerFeatureStat(
        count=100,
        pct_segments_active=0.5,
        mean_mag=1.0,
        mean_mag_incl_zeros=0.5,
        std_mag=0.3,
        min_mag=0.0,
        max_mag=10.0,
        median_mag=1.0,
        percentiles=percentiles,
    )


def _make_segment_features(
    segment_idx: int,
    feature_ids: list[int],
    feature_mags: list[float],
) -> SegmentFeatures:
    """Build a minimal SegmentFeatures for scan tests."""
    return SegmentFeatures(
        segment_idx=segment_idx,
        position=0,
        feature_ids=t.tensor(feature_ids, dtype=t.int32),
        feature_mags=t.tensor(feature_mags, dtype=t.float32),
        char_start=0,
        char_end=10,
    )


# ---------------------------------------------------------------------------
# TestPercentileLookupOptimization
# ---------------------------------------------------------------------------


class TestPercentileLookupOptimization:
    """Tests for the PercentileLookup dataclass and optimized _activation_to_percentile."""

    def test_matches_original_implementation(self) -> None:
        """Optimized version must produce identical float results to the original."""
        percentiles = _REPRESENTATIVE_PERCENTILES
        lookup = _build_percentile_lookup(percentiles)

        rng = random.Random(12345)
        # Activation values spanning the full range and beyond.
        activations = [rng.uniform(-1.0, 11.0) for _ in range(100)]

        for act in activations:
            expected = _activation_to_percentile_original(act, percentiles)
            got = _activation_to_percentile(act, lookup)
            assert got == expected, (
                f"Mismatch for activation={act}: original={expected}, optimized={got}"
            )

    def test_edge_cases(self) -> None:
        """Test values at exact percentile boundaries, below p1, above p99, and midpoints."""
        percentiles = _REPRESENTATIVE_PERCENTILES
        lookup = _build_percentile_lookup(percentiles)

        # Exact p1 boundary: activation == vals[0]
        p1_val = percentiles["p1"]
        assert _activation_to_percentile(
            p1_val, lookup
        ) == _activation_to_percentile_original(p1_val, percentiles)

        # Exact p50 boundary
        p50_val = percentiles["p50"]
        assert _activation_to_percentile(
            p50_val, lookup
        ) == _activation_to_percentile_original(p50_val, percentiles)

        # Exact p99 boundary: activation == vals[-1]
        p99_val = percentiles["p99"]
        assert _activation_to_percentile(
            p99_val, lookup
        ) == _activation_to_percentile_original(p99_val, percentiles)

        # Below p1 — should clamp to lo_pct
        below_p1 = p1_val - 1.0
        assert _activation_to_percentile(
            below_p1, lookup
        ) == _activation_to_percentile_original(below_p1, percentiles)

        # Above p99 — should clamp to hi_pct
        above_p99 = p99_val + 1.0
        assert _activation_to_percentile(
            above_p99, lookup
        ) == _activation_to_percentile_original(above_p99, percentiles)

        # Midpoint between p48 and p49
        p48_val = percentiles["p48"]
        p49_val = percentiles["p49"]
        midpoint = (p48_val + p49_val) / 2.0
        assert _activation_to_percentile(
            midpoint, lookup
        ) == _activation_to_percentile_original(midpoint, percentiles)

    def test_zero_activation(self) -> None:
        """Activation of 0.0 should be handled correctly (typically clamps to lo_pct)."""
        percentiles = _REPRESENTATIVE_PERCENTILES
        lookup = _build_percentile_lookup(percentiles)

        expected = _activation_to_percentile_original(0.0, percentiles)
        got = _activation_to_percentile(0.0, lookup)
        assert got == expected, f"Zero activation: expected={expected}, got={got}"

    def test_identical_percentile_values(self) -> None:
        """When v1 == v0 for adjacent percentile keys, return p0 (no division by zero)."""
        # Make p5 and p6 have the same value.
        percentiles: dict[str, float] = {f"p{i}": float(i) * 0.1 for i in range(1, 100)}
        percentiles["p6"] = percentiles["p5"]

        lookup = _build_percentile_lookup(percentiles)

        # activation exactly at the shared threshold value
        shared_val = percentiles["p5"]
        expected = _activation_to_percentile_original(shared_val, percentiles)
        got = _activation_to_percentile(shared_val, lookup)
        assert got == expected, (
            f"Identical pct values branch: expected={expected}, got={got}"
        )

        # activation midway into the equal bracket — should still be p0
        # Use a value that falls exactly in the [p5, p6] bracket (same val).
        # In the optimized path: bisect_right(vals, shared_val) points just past
        # the last occurrence of shared_val. Check that p0 is returned.
        p5 = 5.0 / 100.0
        assert got in (p5, expected)  # both resolve to p5 = 0.05

    def test_full_scan_equivalence(self) -> None:
        """scan_validation_features with optimized code must match original algorithm."""
        # Build 3 features with linearly spaced percentile dicts.
        feature_ids_list = [10, 20, 30]
        per_feature_stats: dict[str, PerFeatureStat] = {}
        for fid in feature_ids_list:
            pcts_dict = {f"p{i}": float(i) * 0.05 * (fid / 10) for i in range(1, 100)}
            per_feature_stats[str(fid)] = _make_per_feature_stat(pcts_dict)

        # Build doc_segments: 5 docs, each with 3 segments (segment 0 + 2 non-first).
        doc_segments: dict[int, list[SegmentFeatures]] = {}
        rng = random.Random(999)
        for doc_idx in range(5):
            segs = []
            # segment 0 — should be skipped by scan
            segs.append(_make_segment_features(0, [10], [0.5]))
            # non-first segments with activations for features 10, 20, 30
            for seg_idx in (1, 2):
                fids_for_seg = feature_ids_list
                mags_for_seg = [rng.uniform(0.1, 4.0) for _ in feature_ids_list]
                segs.append(_make_segment_features(seg_idx, fids_for_seg, mags_for_seg))
            doc_segments[doc_idx] = segs

        # Run the optimized scan.
        results = scan_validation_features(doc_segments, per_feature_stats)

        # Manually verify each DocFeatureActivation's percentile matches the
        # original linear-scan algorithm.
        for fid, scan_result in results.items():
            stat = per_feature_stats[str(fid)]
            for dfa in scan_result.doc_activations:
                expected_pct = _activation_to_percentile_original(
                    dfa.max_activation, stat.percentiles
                )
                assert dfa.percentile == expected_pct, (
                    f"Feature {fid}, doc {dfa.doc_idx}: "
                    f"expected percentile={expected_pct}, got={dfa.percentile}"
                )


# ---------------------------------------------------------------------------
# TestVectorizedPercentile
# ---------------------------------------------------------------------------


class TestVectorizedPercentile:
    """Verify _vectorized_percentile matches _activation_to_percentile exactly."""

    def _make_per_feature_stats(
        self, feature_ids: list[int]
    ) -> dict[str, PerFeatureStat]:
        """Build per-feature stats with known percentile tables."""
        rng = random.Random(42)
        stats = {}
        for fid in feature_ids:
            # Generate monotonically increasing percentile values
            vals = sorted([rng.uniform(0.0, 10.0) for _ in range(99)])
            percentiles = {f"p{i + 1}": v for i, v in enumerate(vals)}
            stats[str(fid)] = PerFeatureStat(
                count=1000,
                pct_segments_active=0.5,
                mean_mag=5.0,
                mean_mag_incl_zeros=2.5,
                std_mag=1.0,
                min_mag=0.0,
                max_mag=10.0,
                median_mag=5.0,
                percentiles=percentiles,
            )
        return stats

    def test_matches_scalar_reference_many_values(self) -> None:
        """Vectorized percentile must match scalar _activation_to_percentile
        for many activations across multiple features."""
        feature_ids = [10, 20, 30]
        per_feature_stats = self._make_per_feature_stats(feature_ids)

        # Generate test activations: below min, above max, and many in between
        test_activations = [
            -1.0,
            0.0,
            0.001,
            0.5,
            1.0,
            2.5,
            5.0,
            7.5,
            9.99,
            10.0,
            15.0,
        ]

        for fid in feature_ids:
            fid_tensor = t.full((len(test_activations),), fid, dtype=t.int64)
            act_tensor = t.tensor(test_activations, dtype=t.float32)

            vec_results = _vectorized_percentile(
                fid_tensor, act_tensor, per_feature_stats
            )

            lookup = _build_percentile_lookup(per_feature_stats[str(fid)].percentiles)
            for i, act in enumerate(test_activations):
                expected = _activation_to_percentile(act, lookup)
                assert vec_results[i].item() == pytest.approx(expected, abs=1e-5), (
                    f"Feature {fid}, activation {act}: "
                    f"vectorized={vec_results[i].item()}, scalar={expected}"
                )

    def test_matches_on_exact_boundary_values(self) -> None:
        """Vectorized percentile matches scalar at exact percentile boundary values."""
        per_feature_stats = self._make_per_feature_stats([42])
        lookup = _build_percentile_lookup(per_feature_stats["42"].percentiles)

        # Test at exact boundary values
        boundary_vals = lookup.vals
        fid_tensor = t.full((len(boundary_vals),), 42, dtype=t.int64)
        act_tensor = t.tensor(boundary_vals, dtype=t.float32)

        vec_results = _vectorized_percentile(fid_tensor, act_tensor, per_feature_stats)

        for i, val in enumerate(boundary_vals):
            expected = _activation_to_percentile(val, lookup)
            assert vec_results[i].item() == pytest.approx(expected, abs=1e-5), (
                f"Boundary val {val}: vectorized={vec_results[i].item()}, scalar={expected}"
            )

    def test_mixed_features_single_batch(self) -> None:
        """Multiple features in one batch produces correct per-feature percentiles."""
        per_feature_stats = self._make_per_feature_stats([10, 20])

        # Interleaved feature IDs
        fids = t.tensor([10, 20, 10, 20, 10], dtype=t.int64)
        acts = t.tensor([1.0, 1.0, 5.0, 5.0, 9.0], dtype=t.float32)

        vec_results = _vectorized_percentile(fids, acts, per_feature_stats)

        for i in range(len(fids)):
            fid = int(fids[i].item())
            act = float(acts[i].item())
            lookup = _build_percentile_lookup(per_feature_stats[str(fid)].percentiles)
            expected = _activation_to_percentile(act, lookup)
            assert vec_results[i].item() == pytest.approx(expected, abs=1e-5)

    def test_empty_input(self) -> None:
        """Empty inputs produce empty output."""
        per_feature_stats = self._make_per_feature_stats([10])
        result = _vectorized_percentile(
            t.empty(0, dtype=t.int64),
            t.empty(0, dtype=t.float32),
            per_feature_stats,
        )
        assert result.numel() == 0


# ---------------------------------------------------------------------------
# Helpers for TestPersistentSQLiteConnection
# ---------------------------------------------------------------------------


def _make_client(tmp_path: str) -> CompletionJudgeClient:
    """Create a CompletionJudgeClient pointing to a temp DB with no rate limiting."""
    return CompletionJudgeClient(
        model="test-model",
        base_url="http://localhost",
        api_key="test-key",
        cache_path=tmp_path,
        requests_per_minute=0,
    )


# ---------------------------------------------------------------------------
# TestPersistentSQLiteConnection
# ---------------------------------------------------------------------------


class TestPersistentSQLiteConnection:
    """Verify that CompletionJudgeClient uses a single persistent SQLite connection."""

    def test_cache_hit_returns_same_result(self) -> None:
        """Write a value to cache and read it back; result must be identical."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        client = _make_client(db_path)
        try:
            key = "test-key-hit"
            value = "coherence: 4\nfeature_alignment: 3"
            client._cache_put(key, value)
            result = client._cache_get(key)
            assert result == value, f"Expected {value!r}, got {result!r}"
        finally:
            client.close()

    def test_cache_miss_returns_none(self) -> None:
        """Reading a non-existent key must return None."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        client = _make_client(db_path)
        try:
            result = client._cache_get("nonexistent-key-xyz")
            assert result is None, f"Expected None for cache miss, got {result!r}"
        finally:
            client.close()

    def test_overwrite_updates_value(self) -> None:
        """Writing the same key twice must return the most recent value."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        client = _make_client(db_path)
        try:
            key = "overwrite-key"
            client._cache_put(key, "coherence: 1\nfeature_alignment: 1")
            client._cache_put(key, "coherence: 5\nfeature_alignment: 5")
            result = client._cache_get(key)
            assert result == "coherence: 5\nfeature_alignment: 5", (
                f"Expected updated value, got {result!r}"
            )
        finally:
            client.close()

    def test_concurrent_reads_after_write(self) -> None:
        """Write 100 keys then read them all back; every value must match."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        client = _make_client(db_path)
        try:
            n = 100
            entries = {
                f"key-{i}": f"coherence: {(i % 5) + 1}\nfeature_alignment: {(i % 5) + 1}"
                for i in range(n)
            }
            for key, value in entries.items():
                client._cache_put(key, value)
            for key, expected in entries.items():
                result = client._cache_get(key)
                assert result == expected, (
                    f"Key {key!r}: expected {expected!r}, got {result!r}"
                )
        finally:
            client.close()

    def test_judge_with_cache_hit(self) -> None:
        """Calling judge() twice with the same inputs must hit the API only once."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        client = _make_client(db_path)
        try:
            api_response = "coherence: 4\nfeature_alignment: 3"
            mock_post = MagicMock()
            mock_post.return_value.json.return_value = {
                "choices": [{"message": {"content": api_response}}]
            }
            mock_post.return_value.raise_for_status = MagicMock()

            with patch("steering.eval.completion_judge._requests.post", mock_post):
                result1 = client.judge(
                    context="The sky is",
                    completion="blue and clear.",
                    feature_description="mentions sky color",
                    ground_truth="blue.",
                )
                result2 = client.judge(
                    context="The sky is",
                    completion="blue and clear.",
                    feature_description="mentions sky color",
                    ground_truth="blue.",
                )

            assert mock_post.call_count == 1, (
                f"API should be called exactly once (cache hit on second call), "
                f"but was called {mock_post.call_count} times"
            )
            assert result1 == result2, (
                f"Both calls should return identical results, got {result1} and {result2}"
            )
        finally:
            client.close()

    def test_judge_scores_unchanged(self) -> None:
        """Judge returns correct JudgementScore objects for known API responses."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name
        client = _make_client(db_path)
        try:
            test_cases = [
                (
                    "coherence: 5\nfeature_alignment: 5",
                    JudgementScore(coherence=5, feature_alignment=5),
                ),
                (
                    "coherence: 1\nfeature_alignment: 2",
                    JudgementScore(coherence=1, feature_alignment=2),
                ),
                (
                    "coherence: 3\nfeature_alignment: 4",
                    JudgementScore(coherence=3, feature_alignment=4),
                ),
            ]
            for i, (api_response, expected_score) in enumerate(test_cases):
                mock_post = MagicMock()
                mock_post.return_value.json.return_value = {
                    "choices": [{"message": {"content": api_response}}]
                }
                mock_post.return_value.raise_for_status = MagicMock()

                with patch("steering.eval.completion_judge._requests.post", mock_post):
                    result = client.judge(
                        context=f"Context for test case {i}",
                        completion=f"Completion for test case {i}",
                        feature_description=f"Feature {i}",
                        ground_truth=f"Truth {i}",
                    )

                assert result == expected_score, (
                    f"Test case {i}: expected {expected_score}, got {result}"
                )
        finally:
            client.close()


# ---------------------------------------------------------------------------
# Helpers for TestEncoderCaching
# ---------------------------------------------------------------------------


def _make_segments_for_caching(
    n_segs: int = 3,
    features_per_seg: int = 5,
    segment_size: int = 128,
) -> list[SegmentFeatures]:
    """Create synthetic SegmentFeatures with known feature IDs and mags."""
    segs: list[SegmentFeatures] = []
    for i in range(n_segs):
        fids = t.arange(i * 10, i * 10 + features_per_seg, dtype=t.int32)
        mags = t.ones(features_per_seg, dtype=t.float32) * (i + 1.0)
        segs.append(
            SegmentFeatures(
                segment_idx=i,
                position=i * segment_size,
                feature_ids=fids,
                feature_mags=mags,
                char_start=i * 512,
                char_end=(i + 1) * 512,
            )
        )
    return segs


class _FixedWidthTokenizer:
    """Mock tokenizer where each token decodes to exactly 4 characters."""

    def decode(self, token_ids):
        return "x" * (len(token_ids) * 4)


_MOCK_TOK = _FixedWidthTokenizer()


class _FakeTokenizerCaching:
    """Minimal tokenizer for encoder caching tests.

    Each token decodes to exactly 4 characters so that char offsets
    align with the test segment boundaries (multiples of 512).
    """

    bos_id = 2

    def encode(
        self, text: str, prepend: int | None = None, append: int | None = None
    ) -> list[int]:
        tokens = list(range(10, 10 + len(text) // 4))
        if prepend is not None:
            tokens = [prepend] + tokens
        if append is not None:
            tokens = tokens + [append]
        return tokens

    def decode(self, ids: list[int]) -> str:
        return "x" * (len(ids) * 4)


class _FakeModelCaching:
    """Minimal model mock that tracks encoder calls for caching tests.

    Returns deterministic (seeded) logits from forward() and has the
    required attributes.  The encoder is a MagicMock whose call count
    we can inspect.
    """

    def __init__(
        self, vocab_size: int = 100, seq_len: int = 1024, num_buckets: int = 4
    ):
        self.vocab_size = vocab_size

        self.dec_config = MagicMock()
        self.dec_config.sequence_len = seq_len
        self.dec_config.n_kv_head = 4
        self.dec_config.n_head = 8
        self.dec_config.n_embd = 256
        self.dec_config.n_layer = 2

        def fake_encoder_call(fids, mags, mask):
            n_seg = fids.shape[1]
            # Use a fixed seed so encoder output is deterministic for same inputs
            gen = t.Generator().manual_seed(42)
            dense = t.randn(1, n_seg, mask.shape[2], 256, generator=gen)
            return pack_dense_encoder_output_reference(dense, mask)

        self.encoder = MagicMock(side_effect=fake_encoder_call)
        self.encoder.config = MagicMock()
        self.encoder.config.num_buckets = num_buckets

    def __call__(self, idx, **kwargs):
        B = idx.shape[0]
        T = idx.shape[1]
        logits = t.randn(B, T, self.vocab_size)
        # Make token 50 most likely to keep generation deterministic-ish
        logits[:, :, 50] = 10.0
        return logits


class _FakeModelStochastic(_FakeModelCaching):
    """Like _FakeModelCaching but with varied logits for stochastic sampling tests.

    Returns logits seeded from the input token IDs so that:
    - Same inputs always produce identical logits (deterministic).
    - Different token inputs produce different logits.
    - No single token dominates, so sampling with different RNG seeds
      produces genuinely different output sequences.
    """

    def __call__(self, idx, **kwargs):
        B = idx.shape[0]
        T = idx.shape[1]
        # Seed from input token IDs for deterministic-but-varied logits
        seed = int(idx.sum().item()) % (2**31)
        gen = t.Generator().manual_seed(seed)
        return t.randn(B, T, self.vocab_size, generator=gen)


# ---------------------------------------------------------------------------
# TestEncoderCaching
# ---------------------------------------------------------------------------


class TestEncoderCaching:
    """Verify encoder output caching avoids redundant forward passes."""

    def test_encoder_output_deterministic(self) -> None:
        """For fixed encoder inputs, model.encoder() twice returns bitwise identical outputs."""
        model = _FakeModelCaching()
        segs = _make_segments_for_caching(n_segs=3)
        inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )

        enc_fids = inputs.enc_feature_ids
        enc_mags = inputs.enc_magnitudes
        enc_mask = inputs.enc_mask

        out1 = model.encoder(enc_fids, enc_mags, enc_mask)
        out2 = model.encoder(enc_fids, enc_mags, enc_mask)

        assert t.equal(out1.values, out2.values), (
            "Encoder output should be bitwise identical for same inputs"
        )
        assert t.equal(out1.layout.valid_mask, out2.layout.valid_mask), (
            "Encoder layout should be bitwise identical for same inputs"
        )

    def test_precomputed_matches_fresh(self) -> None:
        """generate_completion with precomputed_encoder must produce same tokens as without."""
        model = _FakeModelCaching()
        segs = _make_segments_for_caching(n_segs=3)
        inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        prefix_tokens = list(range(2, 258))  # 256 tokens with BOS
        device = t.device("cpu")
        seed = 42

        # Without precomputed
        tokens_fresh = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=inputs,
            max_new_tokens=10,
            device=device,
            temperature=0.0,
            seed=seed,
            precomputed_encoder=None,
        )

        # Precompute encoder output
        enc_fids = inputs.enc_feature_ids.to(device)
        enc_mags = inputs.enc_magnitudes.to(device)
        enc_mask = inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

        # With precomputed
        tokens_cached = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=inputs,
            max_new_tokens=10,
            device=device,
            temperature=0.0,
            seed=seed,
            precomputed_encoder=encoder_out,
        )

        assert tokens_fresh == tokens_cached, (
            f"Precomputed encoder should produce identical tokens.\n"
            f"Fresh: {tokens_fresh}\nCached: {tokens_cached}"
        )

    def test_precomputed_none_unchanged(self) -> None:
        """When precomputed_encoder=None, encoder is called internally (regression test)."""
        model = _FakeModelCaching()
        segs = _make_segments_for_caching(n_segs=3)
        inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        prefix_tokens = list(range(2, 258))
        device = t.device("cpu")

        # Reset the call count
        model.encoder.reset_mock()

        tokens = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=inputs,
            max_new_tokens=5,
            device=device,
            temperature=0.0,
            precomputed_encoder=None,
        )

        # Encoder should have been called exactly once internally
        assert model.encoder.call_count == 1, (
            f"Expected encoder called once when precomputed_encoder=None, "
            f"got {model.encoder.call_count}"
        )
        assert isinstance(tokens, list)
        assert len(tokens) > 0

    def test_different_conditions_get_different_encoder(self) -> None:
        """For ablation mode, real and ablated produce different encoder outputs."""
        n_segs = 3
        segs = _make_segments_for_caching(n_segs=n_segs)
        target_idx = 2
        feature_id = segs[target_idx].feature_ids[0].item()
        n_prefix = 256
        segment_size = 128

        boundaries = t.zeros(100, 3, dtype=t.float64)
        for i in range(3):
            boundaries[:, i] = float(i + 1)

        prepared = prepare_segments(segs, boundaries, _PASSTHROUGH_CFG)

        prefix_tokens = list(range(n_prefix))

        # "real" condition
        real_inputs = build_multiseg_inputs(
            segments=prepared,
            prefix_tokens=prefix_tokens,
            n_gen_tokens=segment_size,
            target_segment_idx=target_idx,
            tokenizer=_MOCK_TOK,
        )

        # "ablated" condition
        ablated_segs = ablate_feature(prepared, target_idx, feature_id)  # ty: ignore[invalid-argument-type]
        ablated_inputs = build_multiseg_inputs(
            segments=ablated_segs,
            prefix_tokens=prefix_tokens,
            n_gen_tokens=segment_size,
            target_segment_idx=target_idx,
            tokenizer=_MOCK_TOK,
        )

        # The two encoder inputs should differ (masks at minimum)
        assert not t.equal(real_inputs.enc_mask, ablated_inputs.enc_mask), (
            "Real and ablated encoder masks should differ"
        )

    def test_generate_for_document_encoder_calls_equal_conditions(self) -> None:
        """generate_for_document should call encoder once per condition, not per sample."""
        model = _FakeModelCaching()
        tokenizer = _FakeTokenizerCaching()
        n_segs = 3
        segment_size = 128
        num_buckets = 4
        samples_per_condition = 3

        doc_text = "A" * 512 * n_segs
        segs = _make_segments_for_caching(n_segs=n_segs, segment_size=segment_size)

        boundaries = t.zeros(100, num_buckets - 1, dtype=t.float64)
        for i in range(num_buckets - 1):
            boundaries[:, i] = float(i + 1)

        feature_id = segs[2].feature_ids[0].item()

        model.encoder.reset_mock()

        result = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segs,
            target_segment_idx=2,
            feature_id=feature_id,  # ty: ignore[invalid-argument-type]
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=t.device("cpu"),
            num_buckets=num_buckets,
            samples_per_condition=samples_per_condition,
            temperature=0.0,
        )

        # Ablation mode has 2 conditioned variants: real and ablated.
        n_conditions = 2
        expected_encoder_calls = n_conditions
        # Previously it would be n_conditions * samples_per_condition.
        # Now it should be exactly n_conditions.
        assert model.encoder.call_count == expected_encoder_calls, (
            f"Expected {expected_encoder_calls} encoder calls (one per condition), "
            f"got {model.encoder.call_count} "
            f"(would have been {n_conditions * samples_per_condition} without caching)"
        )

        # Verify we still got the right number of samples
        assert len(result.samples) == n_conditions * samples_per_condition


# ---------------------------------------------------------------------------
# TestKVCachePrefillReuse
# ---------------------------------------------------------------------------


class TestKVCachePrefillReuse:
    """Verify KV cache prefill reuse produces correct and consistent results."""

    def test_kv_cache_prefill_copy_correctness(self) -> None:
        """Copy via KVCache.prefill() must exactly match original, and be independent."""
        device = t.device("cpu")
        dtype = t.float32
        num_layers = 2
        num_heads = 4
        head_dim = 32
        seq_len = 384
        n_prefix = 256

        # Create a base KV cache and fill it with random data to simulate prefill
        base_kv = KVCache(
            batch_size=1,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )

        # Simulate prefill: write random data into the first n_prefix positions
        gen = t.Generator().manual_seed(123)
        base_kv.k_cache[:, :, :n_prefix, :, :] = t.randn(
            num_layers, 1, n_prefix, num_heads, head_dim, generator=gen, dtype=dtype
        )
        base_kv.v_cache[:, :, :n_prefix, :, :] = t.randn(
            num_layers, 1, n_prefix, num_heads, head_dim, generator=gen, dtype=dtype
        )
        base_kv.cache_seqlens.fill_(n_prefix)

        # Create a fresh empty KV cache and copy via prefill
        copy_kv = KVCache(
            batch_size=1,
            num_heads=num_heads,
            seq_len=seq_len,
            head_dim=head_dim,
            num_layers=num_layers,
            device=device,
            dtype=dtype,
        )
        copy_kv.prefill(base_kv)

        # Assert the copy's k/v tensors equal the original's up to cache_seqlens
        base_pos = int(base_kv.get_pos())
        assert base_pos == n_prefix, (
            f"Base cache should have {n_prefix} positions, got {base_pos}"
        )
        assert int(copy_kv.get_pos()) == base_pos, (
            "Copy cache should have same position as base"
        )

        assert t.equal(
            copy_kv.k_cache[:, :, :base_pos, :, :],
            base_kv.k_cache[:, :, :base_pos, :, :],
        ), "Copy k_cache should match base up to cache_seqlens"
        assert t.equal(
            copy_kv.v_cache[:, :, :base_pos, :, :],
            base_kv.v_cache[:, :, :base_pos, :, :],
        ), "Copy v_cache should match base up to cache_seqlens"

        # Mutate the copy (simulate a decode step by writing random data)
        copy_kv.k_cache[:, :, base_pos, :, :] = t.randn_like(
            copy_kv.k_cache[:, :, base_pos, :, :]
        )
        copy_kv.v_cache[:, :, base_pos, :, :] = t.randn_like(
            copy_kv.v_cache[:, :, base_pos, :, :]
        )
        copy_kv.cache_seqlens += 1

        # Assert original is unchanged at the mutated position
        assert t.all(base_kv.k_cache[:, :, base_pos, :, :] == 0), (
            "Base k_cache should be unchanged after mutating copy"
        )
        assert t.all(base_kv.v_cache[:, :, base_pos, :, :] == 0), (
            "Base v_cache should be unchanged after mutating copy"
        )
        assert int(base_kv.get_pos()) == base_pos, (
            "Base cache position should be unchanged after mutating copy"
        )

    def test_copied_prefill_matches_fresh(self) -> None:
        """Prefill once + copy + decode must match full fresh generation."""
        model = _FakeModelCaching()
        segs = _make_segments_for_caching(n_segs=3)
        inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        prefix_tokens = list(range(2, 258))
        device = t.device("cpu")
        seed = 42

        # Compute encoder output
        enc_fids = inputs.enc_feature_ids.to(device)
        enc_mags = inputs.enc_magnitudes.to(device)
        enc_mask = inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

        # 1. Full fresh generation (prefill + decode from scratch)
        tokens_fresh = generate_completion(
            model=model,
            prefix_tokens=prefix_tokens,
            encoder_inputs=inputs,
            max_new_tokens=10,
            device=device,
            temperature=0.0,
            seed=seed,
            precomputed_encoder=encoder_out,
        )

        # 2. Prefill once into base cache, copy, then decode
        dec_config = model.dec_config
        max_seq_len = dec_config.sequence_len
        truncated_prefix = prefix_tokens[-(max_seq_len - 10) :]
        n_prefix = len(truncated_prefix)
        total_len = n_prefix + 10
        dtype = t.float32

        seg_map = inputs.seg_map.to(device)

        base_kv = KVCache(
            batch_size=1,
            num_heads=dec_config.n_kv_head,
            seq_len=total_len,
            head_dim=dec_config.n_embd // dec_config.n_head,
            num_layers=dec_config.n_layer,
            device=device,
            dtype=dtype,
        )

        base_last_logits = _prefill_condition(
            model=model,
            prefix_tokens=truncated_prefix,
            encoder_out=encoder_out,
            seg_map=seg_map,
            kv_cache=base_kv,
            device=device,
        )

        # Clone the prefilled cache
        sample_kv = KVCache(
            batch_size=1,
            num_heads=dec_config.n_kv_head,
            seq_len=total_len,
            head_dim=dec_config.n_embd // dec_config.n_head,
            num_layers=dec_config.n_layer,
            device=device,
            dtype=dtype,
        )
        sample_kv.prefill(base_kv)

        tokens_from_copy = list(
            _decode_from_prefill(
                model=model,
                kv_cache=sample_kv,
                last_logits=base_last_logits,
                encoder_out=encoder_out,
                seg_map=seg_map,
                n_prefix=n_prefix,
                max_new_tokens=10,
                total_len=total_len,
                device=device,
                temperature=0.0,
                seed=seed,
            )
        )

        assert tokens_fresh == tokens_from_copy, (
            f"Tokens from copied prefill should match fresh generation.\n"
            f"Fresh: {tokens_fresh}\nFrom copy: {tokens_from_copy}"
        )

    def test_multiple_seeds_from_same_prefill(self) -> None:
        """Different seeds from the same prefill must produce different outputs,
        matching what fresh generation would produce with those same seeds."""
        model = _FakeModelStochastic()
        segs = _make_segments_for_caching(n_segs=3)
        inputs = build_multiseg_inputs(
            segments=segs,
            prefix_tokens=list(range(256)),
            n_gen_tokens=128,
            target_segment_idx=2,
            tokenizer=_MOCK_TOK,
        )
        prefix_tokens = list(range(2, 258))
        device = t.device("cpu")
        seeds = [42, 43, 44]

        # Compute encoder output
        enc_fids = inputs.enc_feature_ids.to(device)
        enc_mags = inputs.enc_magnitudes.to(device)
        enc_mask = inputs.enc_mask.to(device)
        encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)

        dec_config = model.dec_config
        max_seq_len = dec_config.sequence_len
        max_new_tokens = 10
        truncated_prefix = prefix_tokens[-(max_seq_len - max_new_tokens) :]
        n_prefix = len(truncated_prefix)
        total_len = n_prefix + max_new_tokens
        dtype = t.float32
        seg_map = inputs.seg_map.to(device)

        # Prefill once
        base_kv = KVCache(
            batch_size=1,
            num_heads=dec_config.n_kv_head,
            seq_len=total_len,
            head_dim=dec_config.n_embd // dec_config.n_head,
            num_layers=dec_config.n_layer,
            device=device,
            dtype=dtype,
        )

        base_last_logits = _prefill_condition(
            model=model,
            prefix_tokens=truncated_prefix,
            encoder_out=encoder_out,
            seg_map=seg_map,
            kv_cache=base_kv,
            device=device,
        )

        # Decode with 3 different seeds from the same prefill
        outputs_from_copy: list[list[int]] = []
        for seed in seeds:
            sample_kv = KVCache(
                batch_size=1,
                num_heads=dec_config.n_kv_head,
                seq_len=total_len,
                head_dim=dec_config.n_embd // dec_config.n_head,
                num_layers=dec_config.n_layer,
                device=device,
                dtype=dtype,
            )
            sample_kv.prefill(base_kv)

            tokens = list(
                _decode_from_prefill(
                    model=model,
                    kv_cache=sample_kv,
                    last_logits=base_last_logits,
                    encoder_out=encoder_out,
                    seg_map=seg_map,
                    n_prefix=n_prefix,
                    max_new_tokens=max_new_tokens,
                    total_len=total_len,
                    device=device,
                    temperature=0.7,
                    top_k=50,
                    seed=seed,
                )
            )
            outputs_from_copy.append(tokens)

        # All three should differ (stochastic sampling with different seeds)
        assert outputs_from_copy[0] != outputs_from_copy[1], (
            "Seeds 42 and 43 should produce different outputs"
        )
        assert outputs_from_copy[0] != outputs_from_copy[2], (
            "Seeds 42 and 44 should produce different outputs"
        )
        assert outputs_from_copy[1] != outputs_from_copy[2], (
            "Seeds 43 and 44 should produce different outputs"
        )

        # Repeat from scratch (fresh prefill each time, same seeds) and compare
        for i, seed in enumerate(seeds):
            tokens_fresh = generate_completion(
                model=model,
                prefix_tokens=prefix_tokens,
                encoder_inputs=inputs,
                max_new_tokens=max_new_tokens,
                device=device,
                temperature=0.7,
                top_k=50,
                seed=seed,
                precomputed_encoder=encoder_out,
            )
            assert tokens_fresh == outputs_from_copy[i], (
                f"Seed {seed}: fresh generation should match copy-based generation.\n"
                f"Fresh: {tokens_fresh}\nFrom copy: {outputs_from_copy[i]}"
            )

    def test_different_conditions_use_separate_prefill(self) -> None:
        """Real and ablated conditions must produce identical outputs whether
        using the optimized (prefill-reuse) or original (per-sample prefill) path."""
        model = _FakeModelCaching()
        tokenizer = _FakeTokenizerCaching()
        n_segs = 3
        segment_size = 128
        num_buckets = 4
        samples_per_condition = 2

        doc_text = "A" * 512 * n_segs
        segs = _make_segments_for_caching(n_segs=n_segs, segment_size=segment_size)

        boundaries = t.zeros(100, num_buckets - 1, dtype=t.float64)
        for i in range(num_buckets - 1):
            boundaries[:, i] = float(i + 1)

        feature_id = segs[2].feature_ids[0].item()
        device = t.device("cpu")

        # Optimized path (generate_for_document uses prefill reuse internally)
        result_optimized = generate_for_document(
            model=model,
            tokenizer=tokenizer,
            doc_text=doc_text,
            segments=segs,
            target_segment_idx=2,
            feature_id=feature_id,  # ty: ignore[invalid-argument-type]
            mode=EvalMode.ABLATION,
            boundaries=boundaries,
            feature_cfg=_PASSTHROUGH_CFG,
            precomputed_thresholds=None,
            device=device,
            num_buckets=num_buckets,
            samples_per_condition=samples_per_condition,
            temperature=0.0,
            base_seed=42,
        )

        # Original path: call generate_completion per sample (no prefill reuse)
        prefix_tokens = tokenizer.encode(doc_text[: segs[2].char_start], prepend=2)
        prepared = prepare_segments(
            segs[: 2 + 1],
            boundaries,
            _PASSTHROUGH_CFG,
        )

        conditions_map = {
            "real": prepared,
            "ablated": ablate_feature(prepared, 2, feature_id),  # ty: ignore[invalid-argument-type]
        }

        original_samples: list[CompletionSample] = []
        seed_counter = 0
        for cond_name, cond_segs in conditions_map.items():
            encoder_inputs = build_multiseg_inputs(
                segments=cond_segs,
                prefix_tokens=prefix_tokens,
                n_gen_tokens=segment_size,
                target_segment_idx=2,
                tokenizer=tokenizer,
            )
            enc_fids = encoder_inputs.enc_feature_ids.to(device)
            enc_mags = encoder_inputs.enc_magnitudes.to(device)
            enc_mask = encoder_inputs.enc_mask.to(device)
            encoder_out = model.encoder(enc_fids, enc_mags, enc_mask)
            precomputed = encoder_out

            for _i in range(samples_per_condition):
                seed = 42 + seed_counter
                seed_counter += 1
                gen_tokens = generate_completion(
                    model=model,
                    prefix_tokens=prefix_tokens,
                    encoder_inputs=encoder_inputs,
                    max_new_tokens=segment_size,
                    device=device,
                    temperature=0.0,
                    seed=seed,
                    precomputed_encoder=precomputed,
                )
                text = tokenizer.decode(gen_tokens)
                original_samples.append(
                    CompletionSample(
                        condition=cond_name,
                        text=text,
                        seed=seed,
                        bucket_index=None,
                    )
                )

        # Compare: optimized vs original should produce identical texts per seed
        for opt_sample, orig_sample in zip(result_optimized.samples, original_samples, strict=True):
            assert opt_sample.condition == orig_sample.condition, (
                f"Condition mismatch: {opt_sample.condition} vs {orig_sample.condition}"
            )
            assert opt_sample.seed == orig_sample.seed, (
                f"Seed mismatch: {opt_sample.seed} vs {orig_sample.seed}"
            )
            assert opt_sample.text == orig_sample.text, (
                f"Text mismatch for condition={opt_sample.condition}, seed={opt_sample.seed}:\n"
                f"Optimized: {opt_sample.text!r}\nOriginal:  {orig_sample.text!r}"
            )


# ---------------------------------------------------------------------------
# Helpers for TestVectorizedScan
# ---------------------------------------------------------------------------


def _build_synthetic_shard(
    n_docs: int,
    segs_per_doc: int,
    features_per_seg: int,
    feature_pool: list[int],
    rng: random.Random,
) -> dict:
    """Build a synthetic shard dict compatible with save_features / load_features.

    Args:
        n_docs: Number of documents.
        segs_per_doc: Number of segments per document (including segment 0).
        features_per_seg: Number of features per segment.
        feature_pool: Pool of feature IDs to sample from.
        rng: Random generator for reproducibility.

    Returns:
        Dict with all required shard keys.
    """
    all_doc_ids: list[int] = []
    all_seg_indices: list[int] = []
    all_seg_positions: list[int] = []
    all_char_starts: list[int] = []
    all_char_ends: list[int] = []
    all_feature_ids: list[int] = []
    all_feature_mags: list[float] = []
    offsets: list[int] = [0]

    segment_size = 64

    for doc_id in range(n_docs):
        for seg_idx in range(segs_per_doc):
            all_doc_ids.append(doc_id)
            all_seg_indices.append(seg_idx)
            all_seg_positions.append(seg_idx * segment_size)
            all_char_starts.append(seg_idx * 256)
            all_char_ends.append((seg_idx + 1) * 256)

            # Pick random features and magnitudes
            n_feats = features_per_seg
            fids = rng.choices(feature_pool, k=n_feats)
            mags = [rng.uniform(0.1, 5.0) for _ in range(n_feats)]
            all_feature_ids.extend(fids)
            all_feature_mags.extend(mags)
            offsets.append(offsets[-1] + n_feats)

    n_segments = len(all_doc_ids)
    return {
        "n_segments": n_segments,
        "segment_size": segment_size,
        "offsets": t.tensor(offsets, dtype=t.int64),
        "doc_ids": t.tensor(all_doc_ids, dtype=t.int32),
        "segment_indices": t.tensor(all_seg_indices, dtype=t.int32),
        "segment_positions": t.tensor(all_seg_positions, dtype=t.int32),
        "feature_ids": t.tensor(all_feature_ids, dtype=t.int32),
        "feature_mags": t.tensor(all_feature_mags, dtype=t.float32),
        "segment_char_starts": t.tensor(all_char_starts, dtype=t.int32),
        "segment_char_ends": t.tensor(all_char_ends, dtype=t.int32),
    }


def _build_per_feature_stats_for_pool(
    feature_pool: list[int],
) -> dict[str, PerFeatureStat]:
    """Build minimal per-feature stats for the given feature IDs."""
    stats: dict[str, PerFeatureStat] = {}
    for fid in set(feature_pool):
        pcts = {f"p{i}": float(i) * 0.05 for i in range(1, 100)}
        stats[str(fid)] = _make_per_feature_stat(pcts)
    return stats


def _compare_scan_results(
    result_orig: dict,
    result_vec: dict,
) -> None:
    """Assert that two scan result dicts are equivalent.

    Compares feature IDs, doc activations (doc_idx, max_activation,
    max_segment_idx, percentile, total_segments) after sorting for
    deterministic comparison.
    """
    assert set(result_orig.keys()) == set(result_vec.keys()), (
        f"Feature ID mismatch: orig has {sorted(result_orig.keys())}, "
        f"vec has {sorted(result_vec.keys())}"
    )

    for fid, orig in result_orig.items():
        vec = result_vec[fid]
        assert orig.feature_id == vec.feature_id
        assert orig.n_docs_activated == vec.n_docs_activated, (
            f"Feature {fid}: n_docs_activated mismatch "
            f"(orig={orig.n_docs_activated}, vec={vec.n_docs_activated})"
        )

        # Sort doc activations by doc_idx for stable comparison
        orig_acts = sorted(orig.doc_activations, key=lambda a: a.doc_idx)
        vec_acts = sorted(vec.doc_activations, key=lambda a: a.doc_idx)

        assert len(orig_acts) == len(vec_acts), (
            f"Feature {fid}: activation count mismatch "
            f"(orig={len(orig_acts)}, vec={len(vec_acts)})"
        )

        for oa, va in zip(orig_acts, vec_acts, strict=True):
            assert oa.doc_idx == va.doc_idx, (
                f"Feature {fid}: doc_idx mismatch ({oa.doc_idx} vs {va.doc_idx})"
            )
            assert abs(oa.max_activation - va.max_activation) < 1e-5, (
                f"Feature {fid}, doc {oa.doc_idx}: max_activation mismatch "
                f"({oa.max_activation} vs {va.max_activation})"
            )
            assert oa.max_segment_idx == va.max_segment_idx, (
                f"Feature {fid}, doc {oa.doc_idx}: max_segment_idx mismatch "
                f"({oa.max_segment_idx} vs {va.max_segment_idx})"
            )
            assert abs(oa.percentile - va.percentile) < 1e-5, (
                f"Feature {fid}, doc {oa.doc_idx}: percentile mismatch "
                f"({oa.percentile} vs {va.percentile})"
            )
            assert oa.total_segments == va.total_segments, (
                f"Feature {fid}, doc {oa.doc_idx}: total_segments mismatch "
                f"({oa.total_segments} vs {va.total_segments})"
            )


# ---------------------------------------------------------------------------
# TestVectorizedScan
# ---------------------------------------------------------------------------


class TestVectorizedScan:
    """Verify that scan_validation_features_vectorized produces identical
    output to the original scan_validation_features."""

    def test_matches_original_on_small_fixture(self, tmp_path: Path) -> None:
        """Create a small synthetic shard, run both scan versions, assert identical output."""
        rng = random.Random(42)
        feature_pool = [10, 20, 30, 40, 50]
        n_docs = 10
        segs_per_doc = 5
        features_per_seg = 20

        shard_data = _build_synthetic_shard(
            n_docs=n_docs,
            segs_per_doc=segs_per_doc,
            features_per_seg=features_per_seg,
            feature_pool=feature_pool,
            rng=rng,
        )
        per_feature_stats = _build_per_feature_stats_for_pool(feature_pool)

        # Save the shard and get the actual written path
        shard_path = save_features(shard_data, tmp_path / "test_shard_features")

        # --- Original path: FeatureIndex load + scan_validation_features ---
        doc_segments = _load_index_shard(shard_path, "test_shard")
        result_orig = scan_validation_features(doc_segments, per_feature_stats)

        # --- Vectorized path ---
        result_vec_tensors = scan_validation_features_vectorized(
            shard_path, per_feature_stats
        )
        result_vec = scan_results_to_dict(result_vec_tensors)

        # --- Compare ---
        _compare_scan_results(result_orig, result_vec)

    def test_handles_single_segment_docs(self, tmp_path: Path) -> None:
        """Docs with only segment_idx=0 should produce no results in either version."""
        rng = random.Random(123)
        feature_pool = [100, 200]

        # Build shard where all docs have only 1 segment (segment_idx=0)
        shard_data = _build_synthetic_shard(
            n_docs=5,
            segs_per_doc=1,
            features_per_seg=10,
            feature_pool=feature_pool,
            rng=rng,
        )
        per_feature_stats = _build_per_feature_stats_for_pool(feature_pool)

        shard_path = save_features(shard_data, tmp_path / "single_seg_features")

        # Original path
        doc_segments = _load_index_shard(shard_path, "single_seg")
        result_orig = scan_validation_features(doc_segments, per_feature_stats)

        # Vectorized path
        result_vec_tensors = scan_validation_features_vectorized(
            shard_path, per_feature_stats
        )
        result_vec = scan_results_to_dict(result_vec_tensors)

        # Both should be empty since there are no non-first segments
        assert len(result_orig) == 0, (
            f"Original should be empty, got {len(result_orig)} features"
        )
        assert len(result_vec) == 0, (
            f"Vectorized should be empty, got {len(result_vec)} features"
        )

    def test_empty_features(self, tmp_path: Path) -> None:
        """The scan must tolerate sparse segments whose features are all outside
        the validated feature pool (they contribute nothing to the scan output),
        and original vs vectorized must agree.

        Issue #522: a base shard with n_segments > 0 may not contain an empty
        segment, so the sparse segments each carry a single out-of-pool feature
        (ID 99, absent from ``feature_pool``) rather than zero features. This
        preserves the variable-density / irrelevant-segment scenario this test
        exercises (those segments still produce no scan results) while keeping
        the base shard valid.
        """
        # CSR offsets must be monotonically non-decreasing.
        # seg 0: 1 feat (99), seg 1: 5 feats, seg 2: 1 feat (99), seg 3: 3 feats,
        # seg 4: 1 feat (99), seg 5: 2 feats  => offsets = [0, 1, 6, 7, 10, 11, 13]
        n_segments = 6  # 3 docs x 2 segments each
        offsets = [0, 1, 6, 7, 10, 11, 13]
        doc_ids = [0, 0, 1, 1, 2, 2]
        seg_indices = [0, 1, 0, 1, 0, 1]
        seg_positions = [0, 64, 0, 64, 0, 64]
        char_starts = [0, 256, 0, 256, 0, 256]
        char_ends = [256, 512, 256, 512, 256, 512]

        # seg 0: [99]; seg 1: 5 feats; seg 2: [99]; seg 3: 3 feats; seg 4: [99];
        # seg 5: 2 feats. The 99-bearing segments (0/2/4) are each a doc's FIRST
        # segment (seg_index 0), which the scan skips entirely; feature 99 is
        # also absent from feature_pool. Either way these sparse segments
        # contribute nothing to the scan — the same observable result as the
        # previous empty-segment fixture — while keeping the base shard valid.
        feature_ids = [99, 10, 20, 30, 10, 20, 99, 10, 20, 30, 99, 10, 20]
        feature_mags = [0.7, 1.0, 2.0, 3.0, 1.5, 2.5, 0.7, 0.5, 1.0, 1.5, 0.7, 0.8, 0.9]

        shard_data = {
            "n_segments": n_segments,
            "segment_size": 64,
            "offsets": t.tensor(offsets, dtype=t.int64),
            "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(seg_indices, dtype=t.int32),
            "segment_positions": t.tensor(seg_positions, dtype=t.int32),
            "feature_ids": t.tensor(feature_ids, dtype=t.int32),
            "feature_mags": t.tensor(feature_mags, dtype=t.float32),
            "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        }

        feature_pool = [10, 20, 30]
        per_feature_stats = _build_per_feature_stats_for_pool(feature_pool)

        shard_path = save_features(shard_data, tmp_path / "empty_feats_features")

        # Original path
        doc_segments = _load_index_shard(shard_path, "empty_feats")
        result_orig = scan_validation_features(doc_segments, per_feature_stats)

        # Vectorized path
        result_vec_tensors = scan_validation_features_vectorized(
            shard_path, per_feature_stats
        )
        result_vec = scan_results_to_dict(result_vec_tensors)

        # Should produce identical results and not crash
        _compare_scan_results(result_orig, result_vec)

    def test_feature_max_across_segments(self, tmp_path: Path) -> None:
        """When a feature appears in multiple non-first segments of the same doc,
        only the max magnitude should be reported."""
        # Doc 0: seg 0 (skip), seg 1 (feature 10 @ 1.0), seg 2 (feature 10 @ 3.0)
        # Expected: feature 10 -> doc 0 -> max_activation=3.0, max_segment_idx=2
        n_segments = 3
        offsets = [0, 2, 4, 6]
        doc_ids = [0, 0, 0]
        seg_indices = [0, 1, 2]
        seg_positions = [0, 64, 128]
        char_starts = [0, 256, 512]
        char_ends = [256, 512, 768]
        feature_ids = [10, 20, 10, 20, 10, 20]
        feature_mags = [0.5, 0.5, 1.0, 4.0, 3.0, 2.0]

        shard_data = {
            "n_segments": n_segments,
            "segment_size": 64,
            "offsets": t.tensor(offsets, dtype=t.int64),
            "doc_ids": t.tensor(doc_ids, dtype=t.int32),
            "segment_indices": t.tensor(seg_indices, dtype=t.int32),
            "segment_positions": t.tensor(seg_positions, dtype=t.int32),
            "feature_ids": t.tensor(feature_ids, dtype=t.int32),
            "feature_mags": t.tensor(feature_mags, dtype=t.float32),
            "segment_char_starts": t.tensor(char_starts, dtype=t.int32),
            "segment_char_ends": t.tensor(char_ends, dtype=t.int32),
        }

        feature_pool = [10, 20]
        per_feature_stats = _build_per_feature_stats_for_pool(feature_pool)

        shard_path = save_features(shard_data, tmp_path / "max_feats_features")

        # Original path
        doc_segments = _load_index_shard(shard_path, "max_feats")
        result_orig = scan_validation_features(doc_segments, per_feature_stats)

        # Vectorized path
        result_vec_tensors = scan_validation_features_vectorized(
            shard_path, per_feature_stats
        )
        result_vec = scan_results_to_dict(result_vec_tensors)

        _compare_scan_results(result_orig, result_vec)

        # Additionally verify the specific expected values
        # Feature 10: max should be 3.0 at segment 2
        assert 10 in result_vec
        acts_10 = result_vec[10].doc_activations
        assert len(acts_10) == 1
        assert abs(acts_10[0].max_activation - 3.0) < 1e-5
        assert acts_10[0].max_segment_idx == 2

        # Feature 20: max should be 4.0 at segment 1
        assert 20 in result_vec
        acts_20 = result_vec[20].doc_activations
        assert len(acts_20) == 1
        assert abs(acts_20[0].max_activation - 4.0) < 1e-5
        assert acts_20[0].max_segment_idx == 1


# ---------------------------------------------------------------------------
# Helpers for TestEndToEndEquivalence and TestPipelineSmoke
# ---------------------------------------------------------------------------

# _make_segments_for_caching produces segments where segs[i].feature_ids
# = arange(i*10, i*10+features_per_seg).  With the default features_per_seg=5
# and target_segment_idx=2:
#   segs[2].feature_ids = [20, 21, 22, 23, 24]  <- for ablation (feature present)
#   insertion feature must NOT be in segs[2], e.g., feature_id=0 is in segs[0]
_E2E_N_SEGS = 3
_E2E_SEGMENT_SIZE = 128
_E2E_NUM_BUCKETS = 4
_E2E_TARGET_SEG_IDX = 2
# The first feature_id in segs[2] — used for ablation tests
_E2E_ABLATION_FEATURE = 20
# A feature_id NOT in segs[2] — used for insertion tests (it's in segs[0])
_E2E_INSERTION_FEATURE = 0


def _make_e2e_boundaries(max_fid: int, num_buckets: int) -> t.Tensor:
    """Return (max_fid+1, num_buckets-1) zero boundaries (all mags map to bucket 0)."""
    return t.zeros(max_fid + 1, num_buckets - 1, dtype=t.float64)


def _make_e2e_feature_index(
    selections: list[FeatureSelection],
    n_segs: int = _E2E_N_SEGS,
    segment_size: int = _E2E_SEGMENT_SIZE,
) -> MagicMock:
    """Feature index mock whose .lookup() returns real SegmentFeatures for ablation.

    For each selection's feature_id the segments at _E2E_TARGET_SEG_IDX will
    contain that feature_id so ablate_feature() can find and remove it.
    """
    idx = MagicMock()

    def _lookup(shard_stem: str, doc_idx: int) -> list[SegmentFeatures]:
        # All selections share the same segment layout; use the first
        # selection's feature_id to populate segs[2].
        # We include ALL selection feature_ids in the target segment so that
        # each call works regardless of which feature is being tested.
        feature_ids_for_target = [sel.feature_id for sel in selections]
        segs: list[SegmentFeatures] = []
        for i in range(n_segs):
            if i == _E2E_TARGET_SEG_IDX:
                fids = t.tensor(feature_ids_for_target, dtype=t.int32)
                mags = t.ones(len(feature_ids_for_target), dtype=t.float32)
            else:
                fids = t.tensor([i * 10], dtype=t.int32)
                mags = t.ones(1, dtype=t.float32)
            segs.append(
                SegmentFeatures(
                    segment_idx=i,
                    position=i * segment_size,
                    feature_ids=fids,
                    feature_mags=mags,
                    char_start=i * 512,
                    char_end=(i + 1) * 512,
                )
            )
        return segs

    idx.lookup.side_effect = _lookup
    return idx


def _make_e2e_selections_ablation(
    n_features: int,
    docs_per_feature: int,
    start_feature_id: int = _E2E_ABLATION_FEATURE,
) -> list[FeatureSelection]:
    """Selections with feature IDs that exist in segs[2] of the e2e index."""
    selections = []
    for i in range(n_features):
        fid = start_feature_id + i
        selected_docs = list(range(i * docs_per_feature, (i + 1) * docs_per_feature))
        selections.append(
            FeatureSelection(
                feature_id=fid,
                neuronpedia_description=f"Feature {fid}",
                n_docs_activated=docs_per_feature,
                selected_docs=selected_docs,
            )
        )
    return selections


def _make_e2e_common_kwargs(
    selections: list[FeatureSelection],
    feature_index: MagicMock,
    mode: EvalMode,
    num_buckets: int,
    samples_per_condition: int,
    max_fid: int,
) -> dict:
    """Build kwargs for _generate_completions tuned for e2e tests."""
    boundaries = _make_e2e_boundaries(max_fid, num_buckets)
    return {
        "selections": selections,
        "target_segment_maps": None,
        "doc_texts": [f"{'A' * 512 * _E2E_N_SEGS}" for _ in range(N_DOC_TEXTS)],
        "feature_index": feature_index,
        "val_shard_stem": "shard_0000",
        "model": _FakeModelCaching(num_buckets=num_buckets),
        "tokenizer": _FakeTokenizerCaching(),
        "mode": mode,
        "boundaries": boundaries,
        "feature_cfg": _PASSTHROUGH_CFG,
        "precomputed_thresholds": None,
        "device": t.device("cpu"),
        "num_buckets": num_buckets,
        "samples_per_condition": samples_per_condition,
        "temperature": 0.0,
        "top_k_sampling": None,
        "bos_id": 2,
    }


# ---------------------------------------------------------------------------
# TestEndToEndEquivalence
# ---------------------------------------------------------------------------


class TestEndToEndEquivalence:
    """Verify that _generate_completions works end-to-end with real generate_for_document."""

    def test_ablation_mode_equivalence(self) -> None:
        """With 2 features, 2 docs each, ablation mode returns correct DocumentCompletions."""
        n_features = 2
        docs_per_feature = 2
        samples_per_condition = 2
        num_buckets = _E2E_NUM_BUCKETS
        # feature IDs 20 and 21 — both present in segs[2] of the e2e index
        selections = _make_e2e_selections_ablation(n_features, docs_per_feature)
        max_fid = max(sel.feature_id for sel in selections) + 1
        feature_index = _make_e2e_feature_index(selections)

        kwargs = _make_e2e_common_kwargs(
            selections=selections,
            feature_index=feature_index,
            mode=EvalMode.ABLATION,
            num_buckets=num_buckets,
            samples_per_condition=samples_per_condition,
            max_fid=max_fid,
        )

        result = _generate_completions(**kwargs)

        # Should have n_features * docs_per_feature DocumentCompletion objects
        expected_count = n_features * docs_per_feature
        assert len(result) == expected_count, (
            f"Expected {expected_count} DocumentCompletions, got {len(result)}"
        )

        # Ablation mode: 2 conditioned variants x samples_per_condition samples each.
        expected_samples = 2 * samples_per_condition
        for comp in result:
            assert len(comp.samples) == expected_samples, (
                f"feature={comp.feature_id} doc={comp.doc_idx}: "
                f"expected {expected_samples} samples, got {len(comp.samples)}"
            )
            conditions_seen = {s.condition for s in comp.samples}
            assert conditions_seen == {"real", "ablated"}, (
                f"feature={comp.feature_id} doc={comp.doc_idx}: "
                f"expected conditions {{real, ablated}}, got {conditions_seen}"
            )

    def test_insertion_mode_equivalence(self) -> None:
        """With 2 features (not in target seg), insertion mode returns correct DocumentCompletions."""
        n_features = 2
        docs_per_feature = 2
        samples_per_condition = 2
        num_buckets = 4

        # Use feature IDs that are NOT in segs[2] for insertion
        # segs[2].feature_ids contains only the selection feature_ids from _make_e2e_feature_index
        # For insertion we need features absent from the target segment.
        # We build custom selections with a feature in segs[0] territory.
        # We'll use a custom feature_index for insertion mode where target seg has NO features.
        selections = []
        for i in range(n_features):
            fid = 100 + i  # arbitrary IDs not in any seg
            selected_docs = list(
                range(i * docs_per_feature, (i + 1) * docs_per_feature)
            )
            selections.append(
                FeatureSelection(
                    feature_id=fid,
                    neuronpedia_description=f"Feature {fid}",
                    n_docs_activated=docs_per_feature,
                    selected_docs=selected_docs,
                )
            )

        # Custom feature_index: segs[2] has NO features (empty), so add_feature() can insert
        def _lookup_insertion(shard_stem: str, doc_idx: int) -> list[SegmentFeatures]:
            segs: list[SegmentFeatures] = []
            for i in range(_E2E_N_SEGS):
                if i == _E2E_TARGET_SEG_IDX:
                    # Empty feature set — so add_feature can insert without conflict
                    fids = t.tensor([], dtype=t.int32)
                    mags = t.tensor([], dtype=t.float32)
                else:
                    fids = t.tensor([i * 10], dtype=t.int32)
                    mags = t.ones(1, dtype=t.float32)
                segs.append(
                    SegmentFeatures(
                        segment_idx=i,
                        position=i * _E2E_SEGMENT_SIZE,
                        feature_ids=fids,
                        feature_mags=mags,
                        char_start=i * 512,
                        char_end=(i + 1) * 512,
                    )
                )
            return segs

        feature_index = MagicMock()
        feature_index.lookup.side_effect = _lookup_insertion

        max_fid = max(sel.feature_id for sel in selections) + 1
        kwargs = _make_e2e_common_kwargs(
            selections=selections,
            feature_index=feature_index,
            mode=EvalMode.INSERTION,
            num_buckets=num_buckets,
            samples_per_condition=samples_per_condition,
            max_fid=max_fid,
        )

        result = _generate_completions(**kwargs)

        expected_count = n_features * docs_per_feature
        assert len(result) == expected_count, (
            f"Expected {expected_count} DocumentCompletions, got {len(result)}"
        )

        # Insertion mode: real plus one added-bucket variant per bucket.
        expected_samples = (1 + num_buckets) * samples_per_condition
        for comp in result:
            assert len(comp.samples) == expected_samples, (
                f"feature={comp.feature_id} doc={comp.doc_idx}: "
                f"expected {expected_samples} samples, got {len(comp.samples)}"
            )
            conditions_seen = {s.condition for s in comp.samples}
            expected_conditions = {"real"} | {
                f"added_bucket_{b}" for b in range(num_buckets)
            }
            assert conditions_seen == expected_conditions, (
                f"feature={comp.feature_id} doc={comp.doc_idx}: "
                f"conditions mismatch. Expected {expected_conditions}, got {conditions_seen}"
            )

    def test_generation_seeds_preserved(self) -> None:
        """Seed pattern: base_seed + offset, where offset increments per sample across conditions."""
        n_features = 1
        docs_per_feature = 2
        samples_per_condition = 3
        num_buckets = _E2E_NUM_BUCKETS

        selections = _make_e2e_selections_ablation(n_features, docs_per_feature)
        max_fid = max(sel.feature_id for sel in selections) + 1
        feature_index = _make_e2e_feature_index(selections)

        kwargs = _make_e2e_common_kwargs(
            selections=selections,
            feature_index=feature_index,
            mode=EvalMode.ABLATION,
            num_buckets=num_buckets,
            samples_per_condition=samples_per_condition,
            max_fid=max_fid,
        )

        result = _generate_completions(**kwargs)

        assert len(result) == docs_per_feature, (
            f"Expected {docs_per_feature} DocumentCompletions, got {len(result)}"
        )

        # base_seed for doc 0 = 42 + 1 * 1000 = 1042
        # base_seed for doc 1 = 42 + 2 * 1000 = 2042
        # Within each doc: seeds are base_seed + 0, +1, ... across both conditions.
        for doc_position, comp in enumerate(result):
            expected_base = 42 + (doc_position + 1) * 1000
            expected_seeds = list(
                range(expected_base, expected_base + 2 * samples_per_condition)
            )
            actual_seeds = [s.seed for s in comp.samples]
            assert actual_seeds == expected_seeds, (
                f"doc position {doc_position} (feature={comp.feature_id}, doc={comp.doc_idx}): "
                f"expected seeds {expected_seeds}, got {actual_seeds}"
            )

    def test_fixed_loop_with_encoder_caching(self) -> None:
        """Encoder is called once per condition per doc, not once per sample."""
        n_features = 2
        docs_per_feature = 3
        samples_per_condition = 2
        num_buckets = _E2E_NUM_BUCKETS

        selections = _make_e2e_selections_ablation(n_features, docs_per_feature)
        max_fid = max(sel.feature_id for sel in selections) + 1
        feature_index = _make_e2e_feature_index(selections)

        model = _FakeModelCaching(num_buckets=num_buckets)
        model.encoder.reset_mock()

        boundaries = _make_e2e_boundaries(max_fid, num_buckets)
        kwargs = {
            "selections": selections,
            "target_segment_maps": None,
            "doc_texts": [f"{'A' * 512 * _E2E_N_SEGS}" for _ in range(N_DOC_TEXTS)],
            "feature_index": feature_index,
            "val_shard_stem": "shard_0000",
            "model": model,
            "tokenizer": _FakeTokenizerCaching(),
            "mode": EvalMode.ABLATION,
            "boundaries": boundaries,
            "feature_cfg": _PASSTHROUGH_CFG,
            "precomputed_thresholds": None,
            "device": t.device("cpu"),
            "num_buckets": num_buckets,
            "samples_per_condition": samples_per_condition,
            "temperature": 0.0,
            "top_k_sampling": None,
            "bos_id": 2,
        }

        _generate_completions(**kwargs)  # ty: ignore[invalid-argument-type]

        # Ablation mode: 2 conditioned variants per doc.
        # NOT n_conditions * samples_per_condition calls per doc.
        n_conditions = 2
        expected_encoder_calls = n_features * docs_per_feature * n_conditions
        assert model.encoder.call_count == expected_encoder_calls, (
            f"Expected {expected_encoder_calls} encoder calls (one per condition per doc), "
            f"got {model.encoder.call_count}. "
            f"Without caching it would be "
            f"{n_features * docs_per_feature * n_conditions * samples_per_condition}."
        )


# ---------------------------------------------------------------------------
# TestPipelineSmoke
# ---------------------------------------------------------------------------


class TestPipelineSmoke:
    """Minimal smoke test: _generate_completions runs end-to-end without error."""

    def test_pipeline_smoke(self) -> None:
        """1 feature, 1 doc, 1 sample_per_condition — verify structure and non-empty text."""
        n_features = 1
        docs_per_feature = 1
        samples_per_condition = 1
        num_buckets = _E2E_NUM_BUCKETS

        selections = _make_e2e_selections_ablation(n_features, docs_per_feature)
        max_fid = max(sel.feature_id for sel in selections) + 1
        feature_index = _make_e2e_feature_index(selections)

        kwargs = _make_e2e_common_kwargs(
            selections=selections,
            feature_index=feature_index,
            mode=EvalMode.ABLATION,
            num_buckets=num_buckets,
            samples_per_condition=samples_per_condition,
            max_fid=max_fid,
        )

        result = _generate_completions(**kwargs)

        # Should return exactly 1 DocumentCompletion
        assert len(result) == 1, f"Expected 1 DocumentCompletion, got {len(result)}"

        comp = result[0]

        # Check structure
        assert comp.doc_idx == selections[0].selected_docs[0], (
            f"Expected doc_idx={selections[0].selected_docs[0]}, got {comp.doc_idx}"
        )
        assert comp.feature_id == selections[0].feature_id, (
            f"Expected feature_id={selections[0].feature_id}, got {comp.feature_id}"
        )
        assert comp.pre_segment_text != "", (
            "pre_segment_text should be non-empty (text before target segment)"
        )

        # 2 conditioned variants x 1 sample each.
        expected_samples = 2 * samples_per_condition
        assert len(comp.samples) == expected_samples, (
            f"Expected {expected_samples} samples, got {len(comp.samples)}"
        )

        # All samples should have non-empty text
        for sample in comp.samples:
            assert sample.text != "", (
                f"Sample condition={sample.condition} seed={sample.seed} has empty text"
            )
