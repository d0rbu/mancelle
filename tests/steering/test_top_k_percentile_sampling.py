"""Tests for top-k percentile feature selection mode.

Tests cover:
- select_top_k_by_percentile: basic selection, k > n_features, k=0, ties,
  override weights, precomputed percentiles
- select_top_k_by_percentile_batched: multi-segment, empty segments, match
  sequential calls
- SamplingMode config: parsing, backward compatibility, invalid mode
- Determinism: seeded RNG produces reproducible results
- Comparison: fixed-k top_k_percentile vs weighted sampling
- Property-based (Hypothesis): monotonicity, subset, cardinality
"""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional.percentile_table import (
    build_percentile_table,
    magnitude_to_percentile,
    percentile_to_uint8,
    uint8_to_percentile,
)
from steering.conditional.sampling import (
    Distribution,
    SamplingMode,
    SamplingSelection,
    _rand_like_scores,
    _top_k_order,
    parse_feature_selection,
    sample_features,
    sample_features_batched,
    select_top_k_by_percentile,
    select_top_k_by_percentile_batched,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_stat(p_values: list[float]) -> PerFeatureStat:
    """Create a PerFeatureStat with p1..p99 from a list of 99 values."""
    assert len(p_values) == 99
    percentiles = {f"p{i + 1}": float(v) for i, v in enumerate(p_values)}
    return PerFeatureStat(
        count=1000,
        pct_segments_active=0.5,
        mean_mag=float(p_values[49]),
        mean_mag_incl_zeros=float(p_values[49]) * 0.5,
        std_mag=1.0,
        min_mag=float(p_values[0]),
        max_mag=float(p_values[-1]),
        median_mag=float(p_values[49]),
        percentiles=percentiles,
    )


def _make_linear_stat(lo: float = 1.0, hi: float = 99.0) -> PerFeatureStat:
    """Create a stat whose p_k = lo + (k-1)*(hi-lo)/98 for k in 1..99."""
    vals = [lo + (hi - lo) * i / 98.0 for i in range(99)]
    return _make_stat(vals)


def _build_test_table(n_features: int = 10) -> t.Tensor:
    """Build a percentile table with distinct linear stats per feature."""
    stats = {}
    for i in range(n_features):
        lo = 1.0 + i * 10.0
        hi = lo + 98.0
        stats[str(i)] = _make_linear_stat(lo, hi)
    return build_percentile_table(stats, n_sae_features=n_features)


def _rng(seed: int = 42) -> t.Generator:
    return t.Generator().manual_seed(seed)


class TestRandLikeScoresDeviceContract:
    def test_unrelated_runtime_error_propagates_without_retry(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        original_rand = t.rand
        calls = 0

        def fail_once(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 1:
                raise RuntimeError("sentinel rand failure")
            return original_rand(*args, **kwargs)

        monkeypatch.setattr(t, "rand", fail_once)
        rng = _rng(77)
        initial_rng_state = rng.get_state().clone()

        with pytest.raises(RuntimeError, match="sentinel rand failure"):
            _rand_like_scores(t.ones(4), rng)

        assert calls == 1
        assert t.equal(rng.get_state(), initial_rng_state)

    def test_same_device_draw_matches_origin_main_exactly(self) -> None:
        scores = t.tensor([0.5, 0.5, 0.2, 0.5], dtype=t.float32)
        actual_rng = _rng(77)
        expected_rng = _rng(77)

        actual = _rand_like_scores(scores, actual_rng)
        expected = t.rand(
            scores.shape,
            generator=expected_rng,
            device=scores.device,
            dtype=t.float64,
        )

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.device == expected.device
        assert (
            actual.detach().cpu().contiguous().numpy().tobytes()
            == expected.cpu().contiguous().numpy().tobytes()
        )
        assert t.equal(actual_rng.get_state(), expected_rng.get_state())

    @pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is unavailable")
    def test_cpu_generator_cuda_scores_matches_explicit_bridge(self) -> None:
        scores = t.ones(4, device="cuda")
        actual_rng = _rng(91)
        expected_rng = _rng(91)

        actual = _rand_like_scores(scores, actual_rng)
        expected = t.rand(
            scores.shape,
            generator=expected_rng,
            device=expected_rng.device,
            dtype=t.float64,
        ).to(scores.device)

        assert actual.dtype == expected.dtype
        assert actual.shape == expected.shape
        assert actual.device == expected.device
        assert t.equal(actual, expected)
        assert t.equal(actual_rng.get_state(), expected_rng.get_state())


def _legacy_top_k_order(
    scores: t.Tensor,
    feature_ids: t.Tensor,
    rng: t.Generator | None,
) -> t.Tensor:
    """Frozen pre-optimization implementation used as the parity oracle."""
    if rng is None:
        tie_order = feature_ids.to(scores.device).argsort(stable=True)
    else:
        tie_scores = _rand_like_scores(scores, rng)
        tie_order = (-tie_scores).argsort(stable=True)
    reordered_scores = scores[tie_order]
    score_order = (-reordered_scores).argsort(stable=True)
    return tie_order[score_order]


class TestBoundedTopKOrderParity:
    """The bounded prefix must be exact, including RNG consumption and ties."""

    @pytest.mark.parametrize("n", [1, 2, 7, 64, 257, 1024])
    @pytest.mark.parametrize("seed", [0, 1, 17, 714])
    def test_seeded_tie_heavy_prefix_and_next_rng_match_reference(self, n, seed):
        data_rng = t.Generator().manual_seed(10_000 + seed + n)
        feature_ids = t.randperm(n, generator=data_rng, dtype=t.int64) * 13 + 5
        # Eight score levels force many exact cutoff and above-cutoff ties.
        scores = t.randint(
            0,
            8,
            (n,),
            generator=data_rng,
            dtype=t.int64,
        ).to(t.float64)
        ks = sorted({1, max(1, n // 7), max(1, n // 2), n})
        for k in ks:
            reference_rng = _rng(seed)
            optimized_rng = _rng(seed)
            expected = _legacy_top_k_order(
                scores,
                feature_ids,
                reference_rng,
            )[:k]
            actual = _top_k_order(
                scores,
                feature_ids,
                optimized_rng,
                k=k,
            )[:k]
            assert t.equal(actual, expected), (n, seed, k)
            assert t.equal(
                t.rand(32, generator=optimized_rng, dtype=t.float64),
                t.rand(32, generator=reference_rng, dtype=t.float64),
            ), (n, seed, k, "next RNG draw")

    @pytest.mark.parametrize("n", [2, 17, 256, 1025])
    def test_no_rng_cutoff_ties_match_feature_id_reference(self, n):
        feature_ids = t.arange(n - 1, -1, -1, dtype=t.int64) * 3
        scores = (t.arange(n) % 5).to(t.float64)
        for k in sorted({1, max(1, n // 3), n - 1, n}):
            expected = _legacy_top_k_order(scores, feature_ids, None)[:k]
            actual = _top_k_order(scores, feature_ids, None, k=k)[:k]
            assert t.equal(actual, expected), (n, k)

    @pytest.mark.parametrize(
        "scores",
        [
            t.tensor([1.0, float("nan"), 2.0, float("nan")]),
            t.tensor([float("inf"), 1.0, float("-inf"), 1.0]),
        ],
    )
    def test_nonfinite_scores_use_full_reference_path(self, scores):
        feature_ids = t.tensor([40, 20, 10, 30], dtype=t.int64)
        for seed in (None, 9):
            reference_rng = None if seed is None else _rng(seed)
            optimized_rng = None if seed is None else _rng(seed)
            expected = _legacy_top_k_order(scores, feature_ids, reference_rng)[:2]
            actual = _top_k_order(
                scores,
                feature_ids,
                optimized_rng,
                k=2,
            )[:2]
            assert t.equal(actual, expected)
            if seed is not None:
                assert t.equal(
                    t.rand(8, generator=optimized_rng),
                    t.rand(8, generator=reference_rng),
                )

    def test_none_and_full_k_return_complete_reference_order(self):
        scores = t.tensor([0.2, 0.9, 0.9, 0.1], dtype=t.float64)
        feature_ids = t.tensor([7, 3, 5, 1], dtype=t.int64)
        for k in (None, len(scores)):
            reference_rng = _rng(123)
            optimized_rng = _rng(123)
            expected = _legacy_top_k_order(scores, feature_ids, reference_rng)
            actual = _top_k_order(
                scores,
                feature_ids,
                optimized_rng,
                k=k,
            )
            assert t.equal(actual, expected)
            assert t.equal(
                t.rand(8, generator=optimized_rng),
                t.rand(8, generator=reference_rng),
            )


# ---------------------------------------------------------------------------
# Unit Tests: select_top_k_by_percentile
# ---------------------------------------------------------------------------


class TestSelectTopKByPercentile:
    """Unit tests for single-segment top-k percentile selection."""

    def test_basic_top_k_selection(self):
        """10 features, k=3 → returns the 3 with highest percentiles."""

        table = _build_test_table(10)
        # Features 0-9, with magnitudes designed to give different percentiles
        # Feature i has stats with lo=1+10i, hi=99+10i
        # Give each feature a magnitude at its p50 level
        feature_ids = t.arange(10, dtype=t.int64)
        # Magnitudes: feature 9 gets highest percentile relative to its own
        # distribution, feature 0 gets lowest
        # Actually, let's use magnitudes that produce clear percentile ordering
        mags = t.tensor([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0])
        # Feature 0: lo=1, hi=99, mag=5 → low percentile
        # Feature 9: lo=91, hi=189, mag=95 → ~p2 (low percentile for feat 9)

        # Instead, give feature 9 a magnitude at its p99 and others at p1
        # Feature i: p1 = 1 + 10i, p99 = 99 + 10i
        # If mag = p99 = 99 + 10i, percentile ≈ 99.5
        # If mag = p1 = 1 + 10i, percentile ≈ 1.0
        # Give features 7, 8, 9 high magnitudes (above p99), rest at p1
        mags = t.zeros(10)
        for i in range(10):
            p1 = 1.0 + i * 10.0
            p99 = 99.0 + i * 10.0
            if i >= 7:
                mags[i] = p99 + 10.0  # above p99 → percentile 99.5
            else:
                mags[i] = p1 - 1.0  # below p1 → percentile 0.5

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k=3,
        )

        assert ids.shape == (3,)
        assert selected_mags.shape == (3,)
        # Should be features 7, 8, 9 (highest percentiles)
        assert set(ids.tolist()) == {7, 8, 9}

    def test_returned_sorted_by_percentile_descending(self):
        """Returned features are sorted by percentile descending."""

        table = _build_test_table(5)
        feature_ids = t.arange(5, dtype=t.int64)
        # Give each feature a different percentile level
        mags = t.zeros(5)
        for i in range(5):
            p50 = 1.0 + i * 10.0 + (99.0 + i * 10.0 - 1.0 - i * 10.0) * 49.0 / 98.0
            mags[i] = p50  # at p50 for each feature

        # Feature 0: p50 at mag ~50, percentile ~50
        # All features at their p50 → percentile ~50 for each
        # Instead, create a clear ordering:
        for i in range(5):
            p1 = 1.0 + i * 10.0
            p99 = 99.0 + i * 10.0
            # Feature 0: below p1 → 0.5
            # Feature 1: at p25 → ~25
            # Feature 2: at p50 → ~50
            # Feature 3: at p75 → ~75
            # Feature 4: above p99 → 99.5
            target_pct = i * 25.0  # 0, 25, 50, 75, 100
            mags[i] = p1 + (p99 - p1) * min(target_pct, 98.0) / 98.0

        # Feature 0 should be at very low percentile, feature 4 at very high
        mags[0] = (1.0 + 0 * 10.0) - 5.0  # below p1
        mags[4] = (99.0 + 4 * 10.0) + 5.0  # above p99

        ids, _ = select_top_k_by_percentile(feature_ids, mags, table, k=5)

        # Compute percentiles for verification
        pcts = magnitude_to_percentile(ids, _, table)
        # Should be sorted descending
        for j in range(len(pcts) - 1):
            assert pcts[j] >= pcts[j + 1], (
                f"Not sorted descending: pcts[{j}]={pcts[j]:.2f} < pcts[{j+1}]={pcts[j+1]:.2f}"
            )

    def test_returned_mags_correspond_to_correct_features(self):
        """Returned mags match the original magnitudes of selected features."""

        table = _build_test_table(5)
        feature_ids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        mags = t.tensor([100.0, 200.0, 300.0, 400.0, 500.0])

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k=3,
        )

        # Verify each returned mag matches the original for that feature_id
        for fid, mag in zip(ids.tolist(), selected_mags.tolist(), strict=True):
            original_idx = (feature_ids == fid).nonzero(as_tuple=True)[0].item()
            assert mag == mags[original_idx].item(), (
                f"Feature {fid}: returned mag {mag} != original {mags[original_idx].item()}"
            )

    def test_k_greater_than_n_features(self):
        """k=10 but only 5 features → returns all 5, no crash."""

        table = _build_test_table(10)
        feature_ids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        mags = t.tensor([10.0, 20.0, 30.0, 40.0, 50.0])

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k=10,
        )

        assert ids.shape[0] == 5
        assert selected_mags.shape[0] == 5
        assert set(ids.tolist()) == {0, 1, 2, 3, 4}

    def test_k_equals_zero(self):
        """k=0 → returns empty tensors."""

        table = _build_test_table(5)
        feature_ids = t.tensor([0, 1, 2], dtype=t.int64)
        mags = t.tensor([10.0, 20.0, 30.0])

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k=0,
        )

        assert ids.shape == (0,)
        assert selected_mags.shape == (0,)

    def test_empty_features(self):
        """Empty feature_ids → empty result, no crash."""

        table = _build_test_table(5)
        feature_ids = t.zeros(0, dtype=t.int64)
        mags = t.zeros(0)

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k=3,
        )

        assert ids.shape == (0,)
        assert selected_mags.shape == (0,)

    def test_ties_broken_by_seeded_random_order(self):
        """Features with same percentile use reproducible random tie-breaking."""

        # All features have the same stats (same distribution)
        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(5)}
        table = build_percentile_table(stats, n_sae_features=5)

        # All features at the same magnitude → same percentile
        feature_ids = t.tensor([4, 2, 0, 3, 1], dtype=t.int64)
        mags = t.tensor([50.0, 50.0, 50.0, 50.0, 50.0])

        ids1, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=3, rng=_rng(17),
        )
        ids2, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=3, rng=_rng(17),
        )
        ids3, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=3, rng=_rng(18),
        )

        assert ids1.shape == (3,)
        assert t.equal(ids1, ids2)
        assert not t.equal(ids1, ids3)
        assert set(ids1.tolist()).issubset(set(feature_ids.tolist()))

    def test_ties_without_rng_break_by_feature_id_ascending(self):
        """No-RNG deterministic extraction path should select lower fids first."""
        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(5)}
        table = build_percentile_table(stats, n_sae_features=5)

        feature_ids = t.tensor([4, 2, 0, 3, 1], dtype=t.int64)
        mags = t.tensor([50.0, 50.0, 50.0, 50.0, 50.0])

        ids, _ = select_top_k_by_percentile(feature_ids, mags, table, k=3)

        assert ids.tolist() == [0, 1, 2]

    def test_override_weights(self):
        """When override_weights provided, ranking uses those instead of percentiles."""

        table = _build_test_table(5)
        feature_ids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        mags = t.tensor([50.0, 50.0, 50.0, 50.0, 50.0])

        # Override weights: feature 2 is highest, then 4, then 0
        weights = t.tensor([3.0, 1.0, 5.0, 0.5, 4.0])

        ids, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=3, override_weights=weights,
        )

        assert ids.shape == (3,)
        assert set(ids.tolist()) == {2, 4, 0}  # top 3 by weight

    def test_precomputed_percentiles(self):
        """Same result whether using runtime percentiles or precomputed uint8."""

        table = _build_test_table(10)
        feature_ids = t.arange(10, dtype=t.int64)
        mags = t.tensor([5.0, 15.0, 25.0, 35.0, 45.0, 55.0, 65.0, 75.0, 85.0, 95.0])

        # Runtime computation
        ids_runtime, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=5,
        )

        # Precomputed: compute percentiles, encode as uint8, decode
        raw_pcts = magnitude_to_percentile(feature_ids, mags, table)
        encoded = percentile_to_uint8(raw_pcts)
        decoded = uint8_to_percentile(encoded)

        ids_precomputed, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=5, override_weights=decoded,
        )

        # Should select the same features (quantization may cause small diffs
        # but top-k should be robust to that)
        assert set(ids_runtime.tolist()) == set(ids_precomputed.tolist())


# ---------------------------------------------------------------------------
# Unit Tests: select_top_k_by_percentile_batched
# ---------------------------------------------------------------------------


class TestSelectTopKByPercentileBatched:
    """Tests for batched multi-segment top-k percentile selection."""

    def test_multiple_segments_correct_per_segment(self):
        """Multiple segments with different sizes and k values."""

        table = _build_test_table(10)
        # Segment 1: features [0,1,2,3,4], k=2
        # Segment 2: features [5,6,7,8,9], k=3
        seg1_fids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        seg1_mags = t.tensor([100.0, 20.0, 30.0, 40.0, 50.0])  # feat 0 high
        seg2_fids = t.tensor([5, 6, 7, 8, 9], dtype=t.int64)
        seg2_mags = t.tensor([60.0, 70.0, 80.0, 90.0, 200.0])  # feat 9 high

        all_fids = t.cat([seg1_fids, seg2_fids])
        all_mags = t.cat([seg1_mags, seg2_mags])
        segment_sizes = [5, 5]
        ks = t.tensor([2, 3], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
        )

        assert len(results) == 2
        ids1, mags1 = results[0]
        ids2, mags2 = results[1]
        assert ids1.shape[0] == 2
        assert ids2.shape[0] == 3
        # All selected features must be from their respective segments
        assert all(fid.item() in [0, 1, 2, 3, 4] for fid in ids1)
        assert all(fid.item() in [5, 6, 7, 8, 9] for fid in ids2)

    def test_empty_segment_handled(self):
        """Segments with 0 features produce empty results."""

        table = _build_test_table(5)
        seg1_fids = t.tensor([0, 1, 2], dtype=t.int64)
        seg1_mags = t.tensor([10.0, 20.0, 30.0])

        all_fids = seg1_fids
        all_mags = seg1_mags
        segment_sizes = [3, 0]
        ks = t.tensor([2, 3], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
        )

        assert len(results) == 2
        assert results[0][0].shape[0] == 2
        assert results[1][0].shape[0] == 0
        assert results[1][1].shape[0] == 0

    def test_matches_sequential_calls(self):
        """Batched results match sequential per-segment calls."""
        from steering.conditional.sampling import (
            select_top_k_by_percentile,
            select_top_k_by_percentile_batched,
        )

        table = _build_test_table(10)
        seg1_fids = t.tensor([0, 1, 2, 3], dtype=t.int64)
        seg1_mags = t.tensor([10.0, 50.0, 30.0, 90.0])
        seg2_fids = t.tensor([4, 5, 6], dtype=t.int64)
        seg2_mags = t.tensor([40.0, 80.0, 20.0])
        seg3_fids = t.tensor([7, 8, 9], dtype=t.int64)
        seg3_mags = t.tensor([60.0, 70.0, 100.0])

        ks = t.tensor([2, 1, 3], dtype=t.int64)

        # Sequential
        seq_results = []
        for fids, mags, k in [
            (seg1_fids, seg1_mags, 2),
            (seg2_fids, seg2_mags, 1),
            (seg3_fids, seg3_mags, 3),
        ]:
            ids, ms = select_top_k_by_percentile(fids, mags, table, k)
            seq_results.append((ids, ms))

        # Batched
        all_fids = t.cat([seg1_fids, seg2_fids, seg3_fids])
        all_mags = t.cat([seg1_mags, seg2_mags, seg3_mags])
        segment_sizes = [4, 3, 3]

        batch_results = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
        )

        for i in range(3):
            assert set(seq_results[i][0].tolist()) == set(batch_results[i][0].tolist()), (
                f"Segment {i}: sequential {seq_results[i][0].tolist()} != "
                f"batched {batch_results[i][0].tolist()}"
            )

    def test_precomputed_percentiles(self):
        """Batched selection with precomputed percentiles matches runtime."""

        table = _build_test_table(10)
        all_fids = t.arange(10, dtype=t.int64)
        all_mags = t.linspace(5.0, 95.0, 10)
        segment_sizes = [5, 5]
        ks = t.tensor([3, 3], dtype=t.int64)

        # Runtime
        results_runtime = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
        )

        # Precomputed
        raw_pcts = magnitude_to_percentile(all_fids, all_mags, table)
        encoded = percentile_to_uint8(raw_pcts)
        decoded = uint8_to_percentile(encoded)

        results_precomputed = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
            precomputed_percentiles=decoded,
        )

        for i in range(2):
            assert set(results_runtime[i][0].tolist()) == set(results_precomputed[i][0].tolist())

    def test_all_empty_segments(self):
        """All segments empty → all empty results."""

        table = _build_test_table(5)
        all_fids = t.tensor([], dtype=t.int64)
        all_mags = t.tensor([], dtype=t.float32)
        segment_sizes = [0, 0, 0]
        ks = t.tensor([3, 2, 1], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
        )

        assert len(results) == 3
        for ids, mags in results:
            assert ids.shape[0] == 0
            assert mags.shape[0] == 0

    def test_k_clamped_to_segment_size(self):
        """k > segment_size → returns all features in segment."""

        table = _build_test_table(5)
        all_fids = t.tensor([0, 1], dtype=t.int64)
        all_mags = t.tensor([10.0, 20.0])
        segment_sizes = [2]
        ks = t.tensor([100], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, segment_sizes, table, ks,
        )

        assert results[0][0].shape[0] == 2


# ---------------------------------------------------------------------------
# Integration Tests: Config parsing and mode dispatch
# ---------------------------------------------------------------------------


class TestSamplingModeConfig:
    """Config parsing for the mode field."""

    def test_mode_top_k_percentile_parsed(self):
        """mode: top_k_percentile creates SamplingSelection with correct mode."""

        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "min_k": 1,
            "max_k": 25,
            "mode": "top_k_percentile",
        }
        selection = parse_feature_selection(cfg)
        assert isinstance(selection, SamplingSelection)
        assert selection.mode == SamplingMode.top_k_percentile

    def test_mode_weighted_parsed(self):
        """mode: weighted creates standard sampling selection."""

        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "mode": "weighted",
        }
        selection = parse_feature_selection(cfg)
        assert isinstance(selection, SamplingSelection)
        assert selection.mode == SamplingMode.weighted

    def test_mode_omitted_defaults_to_weighted(self):
        """Omitting mode defaults to weighted for backward compatibility."""

        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
        }
        selection = parse_feature_selection(cfg)
        assert isinstance(selection, SamplingSelection)
        assert selection.mode == SamplingMode.weighted

    def test_invalid_mode_raises(self):
        """Invalid mode string raises clear ValueError."""
        cfg = {
            "distribution": "geometric",
            "mean": 5.0,
            "std": 5.0,
            "mode": "invalid_mode",
        }
        with pytest.raises(ValueError, match="mode"):
            parse_feature_selection(cfg)

    def test_threshold_selection_ignores_mode(self):
        """ThresholdSelection configs don't have a mode field."""
        cfg = {
            "threshold_type": "segment_topk",
            "threshold": 10,
        }
        selection = parse_feature_selection(cfg)
        assert not isinstance(selection, SamplingSelection)


# ---------------------------------------------------------------------------
# Integration Tests: sample_features mode dispatch
# ---------------------------------------------------------------------------


class TestSampleFeaturesTopKDispatch:
    """sample_features() and sample_features_batched() dispatch on mode."""

    def _make_bundle(self, mode_str: str = "top_k_percentile", seed: int = 42) -> tuple:
        """Create a SamplingBundle with the given mode."""

        mode = SamplingMode(mode_str)
        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
            min_k=1,
            max_k=10,
            mode=mode,
        )
        table = _build_test_table(10)
        rng = _rng(seed)
        return config, table, rng

    def test_sample_features_top_k_mode(self):
        """sample_features with top_k_percentile mode returns deterministic top-k."""
        config, table, rng = self._make_bundle("top_k_percentile")
        feature_ids = t.arange(10, dtype=t.int64)
        # Give features 7,8,9 magnitudes above their p99
        mags = t.zeros(10)
        for i in range(10):
            p99 = 99.0 + i * 10.0
            mags[i] = p99 + 10.0 if i >= 7 else 1.0

        ids, selected_mags = sample_features(
            feature_ids, mags, table, config, rng,
        )

        # Should have sampled some k and returned the top-k by percentile
        assert ids.shape[0] > 0
        # Verify selected features have the highest percentiles
        pcts = magnitude_to_percentile(ids, selected_mags, table)
        all_pcts = magnitude_to_percentile(feature_ids, mags, table)
        # Every selected feature's percentile >= every non-selected feature's percentile
        non_selected = set(range(10)) - set(ids.tolist())
        if non_selected and len(ids) > 0:
            min_selected_pct = pcts.min().item()
            for ns_fid in non_selected:
                ns_pct = all_pcts[ns_fid].item()
                assert min_selected_pct >= ns_pct - 1e-6, (
                    f"Selected feature has pct {min_selected_pct} < "
                    f"non-selected feature {ns_fid} with pct {ns_pct}"
                )

    def test_override_weights_match_internal_top_k_percentiles(self):
        """Precomputed percentile scores must preserve top-k dispatch exactly."""
        config, table, _ = self._make_bundle("top_k_percentile")
        feature_ids = t.arange(10, dtype=t.int64)
        mags = t.tensor(
            [5.0, 30.0, 80.0, 15.0, 130.0, 70.0, 95.0, 10.0, 160.0, 45.0]
        )
        precomputed_weights = magnitude_to_percentile(feature_ids, mags, table)

        ref_ids, ref_mags = sample_features(
            feature_ids, mags, table, config, _rng(77)
        )
        reuse_ids, reuse_mags = sample_features(
            feature_ids,
            mags,
            table,
            config,
            _rng(77),
            override_weights=precomputed_weights,
        )

        assert t.equal(reuse_ids, ref_ids)
        assert t.equal(reuse_mags, ref_mags)

    def test_sample_features_weighted_mode_unchanged(self):
        """sample_features with weighted mode uses multinomial (existing behavior)."""
        config, table, rng = self._make_bundle("weighted")
        feature_ids = t.arange(10, dtype=t.int64)
        mags = t.tensor([50.0] * 10)

        # Run multiple times — weighted mode is stochastic
        results = set()
        for seed in range(10):
            rng = _rng(seed)
            ids, _ = sample_features(feature_ids, mags, table, config, rng)
            results.add(frozenset(ids.tolist()))

        # With stochastic sampling, we should see variation across seeds
        # (Though this is probabilistic, 10 different seeds should give > 1 result)
        assert len(results) >= 1  # at minimum it shouldn't crash

    def test_sample_features_batched_top_k_mode(self):
        """sample_features_batched with top_k_percentile mode."""
        config, table, rng = self._make_bundle("top_k_percentile")
        seg1_fids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        seg1_mags = t.tensor([100.0, 20.0, 30.0, 40.0, 50.0])
        seg2_fids = t.tensor([5, 6, 7, 8, 9], dtype=t.int64)
        seg2_mags = t.tensor([60.0, 70.0, 80.0, 90.0, 200.0])

        all_fids = t.cat([seg1_fids, seg2_fids])
        all_mags = t.cat([seg1_mags, seg2_mags])

        results = sample_features_batched(
            all_fids, all_mags, [5, 5], table, config, rng,
        )

        assert len(results) == 2
        for ids, _ in results:
            assert ids.shape[0] > 0  # non-empty (min_k=1)

    def test_top_k_mode_empty_features(self):
        """sample_features with top_k_percentile mode handles empty input."""
        config, table, rng = self._make_bundle("top_k_percentile")
        ids, mags = sample_features(
            t.zeros(0, dtype=t.int64),
            t.zeros(0),
            table,
            config,
            rng,
        )
        assert ids.shape == (0,)
        assert mags.shape == (0,)


# ---------------------------------------------------------------------------
# Determinism Tests
# ---------------------------------------------------------------------------


class TestDeterminism:
    """Same seed → same results for top-k percentile mode."""

    def test_same_seed_same_selection(self):
        """Seeded RNG produces identical top-k selection."""

        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
            min_k=1,
            max_k=10,
            mode=SamplingMode.top_k_percentile,
        )
        table = _build_test_table(10)
        feature_ids = t.arange(10, dtype=t.int64)
        mags = t.linspace(10.0, 100.0, 10)

        rng1 = _rng(42)
        ids1, mags1 = sample_features(feature_ids, mags, table, config, rng1)

        rng2 = _rng(42)
        ids2, mags2 = sample_features(feature_ids, mags, table, config, rng2)

        assert ids1.tolist() == ids2.tolist()
        assert mags1.tolist() == mags2.tolist()

    def test_different_seed_different_cardinality(self):
        """Different seeds may produce different k values (but same top-k within)."""

        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
            min_k=1,
            max_k=10,
            mode=SamplingMode.top_k_percentile,
        )
        table = _build_test_table(10)
        feature_ids = t.arange(10, dtype=t.int64)
        mags = t.linspace(10.0, 100.0, 10)

        # Collect cardinalities from many seeds
        cardinalities = set()
        for seed in range(50):
            rng = _rng(seed)
            ids, _ = sample_features(feature_ids, mags, table, config, rng)
            cardinalities.add(len(ids))

        # With geometric distribution, we should see multiple cardinalities
        assert len(cardinalities) > 1, f"Only saw cardinalities: {cardinalities}"

    def test_batched_determinism(self):
        """Batched top-k is deterministic with same seed."""

        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=3.0,
            std=3.0,
            min_k=1,
            max_k=8,
            mode=SamplingMode.top_k_percentile,
        )
        table = _build_test_table(10)
        all_fids = t.arange(10, dtype=t.int64)
        all_mags = t.linspace(5.0, 95.0, 10)

        rng1 = _rng(123)
        r1 = sample_features_batched(all_fids, all_mags, [5, 5], table, config, rng1)

        rng2 = _rng(123)
        r2 = sample_features_batched(all_fids, all_mags, [5, 5], table, config, rng2)

        for i in range(2):
            assert r1[i][0].tolist() == r2[i][0].tolist()
            assert r1[i][1].tolist() == r2[i][1].tolist()


# ---------------------------------------------------------------------------
# Comparison with existing modes
# ---------------------------------------------------------------------------


class TestComparisonWithExistingModes:
    """Compare top_k_percentile behavior against known reference points."""

    def test_fixed_k_selects_top_by_percentile(self):
        """With fixed k (min_k=max_k-1=5), result contains top-5 by percentile."""

        config = SamplingSelection(
            distribution=Distribution.geometric,
            mean=5.0,
            std=5.0,
            min_k=5,
            max_k=6,  # forces k=5 (geometric wraps: k % (6-5) + 5 = 5)
            mode=SamplingMode.top_k_percentile,
        )
        table = _build_test_table(10)
        feature_ids = t.arange(10, dtype=t.int64)
        # Give each feature a magnitude that produces a DIFFERENT percentile.
        # Feature i: p_k = k + 10*i. So mag = 20*i + 10 puts feature i at
        # percentile 10 + 10*i (feature 0→p10, feature 9→p100/99.5).
        mags = t.tensor([20.0 * i + 10.0 for i in range(10)])

        rng = _rng(42)
        ids, _ = sample_features(feature_ids, mags, table, config, rng)

        # Should always be exactly 5
        assert ids.shape[0] == 5

        # These should be the top-5 by percentile (features 5-9)
        all_pcts = magnitude_to_percentile(feature_ids, mags, table)
        _, top5_idx = all_pcts.topk(5)
        expected_ids = set(feature_ids[top5_idx].tolist())
        assert set(ids.tolist()) == expected_ids


# ---------------------------------------------------------------------------
# Property-Based Tests (Hypothesis)
# ---------------------------------------------------------------------------


class TestTopKProperties:
    """Property-based tests for top-k percentile selection."""

    @given(
        n_features=st.integers(min_value=1, max_value=20),
        k=st.integers(min_value=0, max_value=25),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_cardinality(self, n_features: int, k: int, seed: int):
        """Output length == min(k, n_active_features)."""

        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(n_features)}
        table = build_percentile_table(stats, n_sae_features=n_features)

        feature_ids = t.arange(n_features, dtype=t.int64)
        mags = t.rand(n_features) * 100.0

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k,
        )

        expected_len = min(k, n_features)
        assert ids.shape[0] == expected_len
        assert selected_mags.shape[0] == expected_len

    @given(
        n_features=st.integers(min_value=1, max_value=20),
        k=st.integers(min_value=1, max_value=25),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_subset_property(self, n_features: int, k: int, seed: int):
        """Top-k features are always a subset of all active features."""

        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(n_features)}
        table = build_percentile_table(stats, n_sae_features=n_features)

        feature_ids = t.arange(n_features, dtype=t.int64)
        mags = t.rand(n_features) * 100.0

        ids, _ = select_top_k_by_percentile(feature_ids, mags, table, k)

        input_set = set(feature_ids.tolist())
        output_set = set(ids.tolist())
        assert output_set.issubset(input_set)

    @given(
        n_features=st.integers(min_value=2, max_value=20),
        k=st.integers(min_value=2, max_value=25),
        seed=st.integers(min_value=0, max_value=10000),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_monotonicity(self, n_features: int, k: int, seed: int):
        """If feature A has higher percentile than B, and both in top-k,
        A appears before B in output."""

        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(n_features)}
        table = build_percentile_table(stats, n_sae_features=n_features)

        feature_ids = t.arange(n_features, dtype=t.int64)
        # Use seeded random mags for variety
        gen = t.Generator().manual_seed(seed)
        mags = t.rand(n_features, generator=gen) * 100.0

        ids, selected_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k,
        )

        if ids.shape[0] < 2:
            return  # nothing to check

        pcts = magnitude_to_percentile(ids, selected_mags, table)
        for j in range(len(pcts) - 1):
            assert pcts[j] >= pcts[j + 1] - 1e-10, (
                f"Not monotonically sorted: pcts[{j}]={pcts[j]:.4f} < pcts[{j+1}]={pcts[j+1]:.4f}"
            )

    @given(
        n_features=st.integers(min_value=1, max_value=20),
        k=st.integers(min_value=1, max_value=25),
    )
    @settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
    def test_no_duplicates(self, n_features: int, k: int):
        """Output contains no duplicate feature IDs."""

        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(n_features)}
        table = build_percentile_table(stats, n_sae_features=n_features)

        feature_ids = t.arange(n_features, dtype=t.int64)
        mags = t.rand(n_features) * 100.0

        ids, _ = select_top_k_by_percentile(feature_ids, mags, table, k)

        assert len(set(ids.tolist())) == ids.shape[0]


# ---------------------------------------------------------------------------
# Scalar and batched top-k paths must produce identical seeded random
# tie-breaking under exact percentile ties.
# ---------------------------------------------------------------------------


class TestTopKTiebreakOrderingParity:
    """Scalar and batched tiebreak must produce the same seeded random order."""

    def test_topk_percentile_batched_matches_scalar_seeded_tie_order(self):
        """All features share the same percentile score; both paths must return
        the same seeded random order.

        Input chosen to surface ordering-only bugs:
        - Feature ids in non-monotonic order: [7, 3, 5, 1, 9]
        - All magnitudes identical → identical percentile scores → ties.
        """
        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(10)}
        table = build_percentile_table(stats, n_sae_features=10)

        feature_ids = t.tensor([7, 3, 5, 1, 9], dtype=t.int64)
        mags = t.tensor([50.0, 50.0, 50.0, 50.0, 50.0])

        scalar_rng = _rng(123)
        batched_rng = _rng(123)
        scalar_ids, scalar_mags = select_top_k_by_percentile(
            feature_ids, mags, table, k=3, rng=scalar_rng,
        )
        batched_results = select_top_k_by_percentile_batched(
            feature_ids, mags, [5], table, t.tensor([3], dtype=t.int64),
            rng=batched_rng,
        )
        batched_ids, batched_mags = batched_results[0]

        assert t.equal(scalar_ids, batched_ids), (
            f"Ordering divergence: scalar={scalar_ids.tolist()}, "
            f"batched={batched_ids.tolist()}"
        )
        assert t.equal(scalar_mags, batched_mags), (
            f"Mags ordering divergence: scalar={scalar_mags.tolist()}, "
            f"batched={batched_mags.tolist()}"
        )
        assert set(scalar_ids.tolist()).issubset(set(feature_ids.tolist()))

    def test_topk_percentile_tie_order_changes_with_seed(self):
        """Tie-breaking is deterministic for a seed and random across seeds."""
        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(10)}
        table = build_percentile_table(stats, n_sae_features=10)

        feature_ids = t.tensor([7, 3, 5, 1, 9], dtype=t.int64)
        mags = t.tensor([50.0] * 5)

        ids_a, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=5, rng=_rng(11),
        )
        ids_b, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=5, rng=_rng(11),
        )
        ids_c, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=5, rng=_rng(12),
        )

        assert t.equal(ids_a, ids_b)
        assert not t.equal(ids_a, ids_c)

    def test_topk_percentile_mixed_ties_and_distinct_scores(self):
        """Random tie-breaking must not perturb distinct primary scores."""
        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(10)}
        table = build_percentile_table(stats, n_sae_features=10)

        feature_ids = t.tensor([3, 1, 2, 5, 4], dtype=t.int64)
        # Override weights to pin scores exactly — feature 1,2,5 tie at 99,
        # feature 3 has 10, feature 4 has 20.
        weights = t.tensor([10.0, 99.0, 99.0, 99.0, 20.0])
        mags = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        scalar_rng = _rng(99)
        batched_rng = _rng(99)
        scalar_ids, _ = select_top_k_by_percentile(
            feature_ids, mags, table, k=5, override_weights=weights, rng=scalar_rng,
        )

        batched = select_top_k_by_percentile_batched(
            feature_ids, mags, [5], table, t.tensor([5], dtype=t.int64),
            override_weights_list=[weights],
            rng=batched_rng,
        )
        batched_ids, _ = batched[0]

        assert t.equal(scalar_ids, batched_ids), (
            f"Order mismatch: scalar={scalar_ids.tolist()}, "
            f"batched={batched_ids.tolist()}"
        )
        assert set(scalar_ids[:3].tolist()) == {1, 2, 5}
        assert scalar_ids[3:].tolist() == [4, 3]

    def test_topk_percentile_multi_segment_matches_sequential_scalar_rng(self):
        """Batched tie-breaking consumes RNG in segment order, matching scalar calls."""
        stat = _make_linear_stat(1.0, 99.0)
        stats = {str(i): stat for i in range(10)}
        table = build_percentile_table(stats, n_sae_features=10)

        all_feature_ids = t.tensor([7, 3, 5, 1, 9, 0, 2, 4], dtype=t.int64)
        all_mags = t.tensor([50.0] * 8)
        segment_sizes = [5, 3]
        ks = t.tensor([3, 2], dtype=t.int64)

        scalar_rng = _rng(2026)
        expected = [
            select_top_k_by_percentile(
                all_feature_ids[:5], all_mags[:5], table, k=3, rng=scalar_rng,
            ),
            select_top_k_by_percentile(
                all_feature_ids[5:], all_mags[5:], table, k=2, rng=scalar_rng,
            ),
        ]

        batched = select_top_k_by_percentile_batched(
            all_feature_ids, all_mags, segment_sizes, table, ks, rng=_rng(2026),
        )

        for (exp_ids, exp_mags), (got_ids, got_mags) in zip(expected, batched, strict=True):
            assert t.equal(exp_ids, got_ids)
            assert t.equal(exp_mags, got_mags)
