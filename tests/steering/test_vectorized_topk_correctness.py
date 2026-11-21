"""Rigorous correctness tests for the vectorized rewrite of
select_top_k_by_percentile_batched.

The vectorized implementation pads variable-length segments into
[n_segments, max_seg_size] tensors with -inf padding, adds a fid-based
tiebreaker (fid / (max_fid * 1e10)) subtracted from scores, then uses a
single padded_scores_tb.topk() call.  These tests verify equivalence with
the sequential reference implementation across tiebreaking, padding,
variable-k, override weights, and randomized regression scenarios.
"""

from __future__ import annotations

import torch as t

from steering.conditional.percentile_table import (
    build_percentile_table,
)
from steering.conditional.sampling import (
    select_top_k_by_percentile,
    select_top_k_by_percentile_batched,
)
from steering.stats_types import PerFeatureStat

# ---------------------------------------------------------------------------
# Helpers (mirrored from test_top_k_percentile_sampling.py)
# ---------------------------------------------------------------------------


def _make_linear_stat(lo: float = 1.0, hi: float = 99.0) -> PerFeatureStat:
    vals = [lo + (hi - lo) * i / 98.0 for i in range(99)]
    percentiles = {f"p{i + 1}": float(v) for i, v in enumerate(vals)}
    return PerFeatureStat(
        count=1000,
        pct_segments_active=0.5,
        mean_mag=float(vals[49]),
        mean_mag_incl_zeros=float(vals[49]) * 0.5,
        std_mag=1.0,
        min_mag=float(vals[0]),
        max_mag=float(vals[-1]),
        median_mag=float(vals[49]),
        percentiles=percentiles,
    )


def _build_test_table(n_features: int = 10) -> t.Tensor:
    stats = {}
    for i in range(n_features):
        lo = 1.0 + i * 10.0
        hi = lo + 98.0
        stats[str(i)] = _make_linear_stat(lo, hi)
    return build_percentile_table(stats, n_sae_features=n_features)


def _uniform_stat() -> PerFeatureStat:
    """A stat where every feature has the identical distribution."""
    return _make_linear_stat(1.0, 99.0)


# ---------------------------------------------------------------------------
# TestVectorizedTopKTiebreaking
# ---------------------------------------------------------------------------


class TestVectorizedTopKTiebreaking:
    """Verify the fid-based tiebreaker matches the sequential double-argsort."""

    def test_tied_scores_select_lower_fid(self):
        """5 features with IDENTICAL percentile scores; k=3 selects lowest fids."""
        stat = _uniform_stat()
        stats = {str(i): stat for i in range(5)}
        table = build_percentile_table(stats, n_sae_features=5)

        # All features at the same magnitude -> identical percentile scores.
        fids = t.tensor([4, 2, 0, 3, 1], dtype=t.int64)
        mags = t.tensor([50.0, 50.0, 50.0, 50.0, 50.0])

        results = select_top_k_by_percentile_batched(
            fids, mags, [5], table, t.tensor([3], dtype=t.int64),
        )
        selected = set(results[0][0].tolist())
        assert selected == {0, 1, 2}, f"Expected {{0,1,2}}, got {selected}"

    def test_tiebreak_does_not_affect_non_tied(self):
        """Features with clearly distinct scores: tiebreaker must not reorder."""
        table = _build_test_table(5)
        fids = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        # Give each feature a magnitude that maps to a very different percentile.
        # Feature i has distribution lo=1+10i, hi=99+10i.
        # Place feature 4 above its p99, feature 3 near its p75, etc.
        mags = t.zeros(5)
        for i in range(5):
            p99 = 99.0 + i * 10.0
            mags[i] = p99 + (4 - i) * 50.0  # 4 -> highest, 0 -> lowest

        seq_ids, _ = select_top_k_by_percentile(fids, mags, table, k=3)
        batch_results = select_top_k_by_percentile_batched(
            fids, mags, [5], table, t.tensor([3], dtype=t.int64),
        )
        assert set(seq_ids.tolist()) == set(batch_results[0][0].tolist())

    def test_tiebreak_with_large_fids(self):
        """Large feature IDs (10000+) must not overflow the tiebreaker scaling."""
        n = 20000
        stat = _uniform_stat()
        stats = {str(i): stat for i in range(n)}
        table = build_percentile_table(stats, n_sae_features=n)

        # 5 features all with same magnitude -> tied percentiles
        fids = t.tensor([19999, 10000, 15000, 12000, 18000], dtype=t.int64)
        mags = t.tensor([50.0] * 5)

        results = select_top_k_by_percentile_batched(
            fids, mags, [5], table, t.tensor([3], dtype=t.int64),
        )
        selected = set(results[0][0].tolist())
        # Lower fids should win: 10000, 12000, 15000
        assert selected == {10000, 12000, 15000}, f"Expected {{10000,12000,15000}}, got {selected}"

    def test_tiebreak_matches_sequential_reference(self):
        """Batched vs sequential for 5 segments with deliberate ties."""
        stat = _uniform_stat()
        n_feat = 100
        stats = {str(i): stat for i in range(n_feat)}
        table = build_percentile_table(stats, n_sae_features=n_feat)

        segments_fids = []
        segments_mags = []
        seg_sizes = []
        ks_list = []

        rng = t.Generator().manual_seed(123)
        for seg_idx in range(5):
            sz = 5 + seg_idx * 3  # 5, 8, 11, 14, 17
            # Use a SINGLE magnitude for all features in each segment -> all tied
            mag_val = 30.0 + seg_idx * 10.0
            fids = t.randperm(n_feat, generator=rng)[:sz].to(t.int64)
            mags = t.full((sz,), mag_val)
            segments_fids.append(fids)
            segments_mags.append(mags)
            seg_sizes.append(sz)
            ks_list.append(min(3 + seg_idx, sz))

        all_fids = t.cat(segments_fids)
        all_mags = t.cat(segments_mags)
        ks = t.tensor(ks_list, dtype=t.int64)

        batch_results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
        )

        for i in range(5):
            seq_ids, _ = select_top_k_by_percentile(
                segments_fids[i], segments_mags[i], table, ks_list[i],
            )
            assert set(seq_ids.tolist()) == set(batch_results[i][0].tolist()), (
                f"Segment {i}: sequential {sorted(seq_ids.tolist())} != "
                f"batched {sorted(batch_results[i][0].tolist())}"
            )


# ---------------------------------------------------------------------------
# TestVectorizedTopKPadding
# ---------------------------------------------------------------------------


class TestVectorizedTopKPadding:
    """Verify padding (-inf scores, 0-fid, 0-mag) never leaks into results."""

    def test_padding_never_appears_in_results(self):
        """Segments [5,2,8] with fids starting at 100; no fid=0 in results.
        Also verifies batched matches sequential reference."""
        n_feat = 200
        table = _build_test_table(n_feat)

        seg_fids = [
            t.arange(100, 105, dtype=t.int64),
            t.arange(110, 112, dtype=t.int64),
            t.arange(120, 128, dtype=t.int64),
        ]
        seg_mags = [
            t.linspace(10.0, 50.0, 5),
            t.linspace(20.0, 30.0, 2),
            t.linspace(10.0, 80.0, 8),
        ]

        all_fids = t.cat(seg_fids)
        all_mags = t.cat(seg_mags)
        seg_sizes = [5, 2, 8]
        ks_list = [3, 2, 5]
        ks = t.tensor(ks_list, dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
        )

        for i, (ids, _) in enumerate(results):
            # No padding leakage
            for fid in ids.tolist():
                assert fid != 0, (
                    f"Segment {i}: padding fid=0 leaked into results"
                )
            # Reference equality
            ref_ids, _ = select_top_k_by_percentile(
                seg_fids[i], seg_mags[i], table, ks_list[i],
            )
            assert set(ids.tolist()) == set(ref_ids.tolist()), (
                f"Segment {i}: batched {sorted(ids.tolist())} != "
                f"sequential {sorted(ref_ids.tolist())}"
            )

    def test_highly_variable_segment_sizes(self):
        """Segments [1, 500, 0, 3] all produce correct results vs reference."""
        n_feat = 1000
        stat = _uniform_stat()
        stats = {str(i): stat for i in range(n_feat)}
        table = build_percentile_table(stats, n_sae_features=n_feat)

        rng = t.Generator().manual_seed(7)
        seg_fids = [
            t.tensor([42], dtype=t.int64),
            t.randperm(n_feat, generator=rng)[:500].to(t.int64),
            t.tensor([], dtype=t.int64),
            t.tensor([10, 20, 30], dtype=t.int64),
        ]
        seg_mags = [
            t.tensor([50.0]),
            t.rand(500) * 100.0,
            t.tensor([], dtype=t.float32),
            t.tensor([10.0, 20.0, 30.0]),
        ]

        all_fids = t.cat(seg_fids)
        all_mags = t.cat(seg_mags)
        seg_sizes = [1, 500, 0, 3]
        ks_list = [1, 10, 5, 2]
        ks = t.tensor(ks_list, dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
        )

        expected_lengths = [1, 10, 0, 2]  # min(k, seg_size)
        for i, (ids, _) in enumerate(results):
            assert ids.shape[0] == expected_lengths[i], (
                f"Segment {i}: expected {expected_lengths[i]} results, got {ids.shape[0]}"
            )
            # Reference equality for non-empty segments
            if seg_sizes[i] > 0 and ks_list[i] > 0:
                ref_ids, _ = select_top_k_by_percentile(
                    seg_fids[i], seg_mags[i], table, ks_list[i],
                )
                assert set(ids.tolist()) == set(ref_ids.tolist()), (
                    f"Segment {i}: batched {sorted(ids.tolist())} != "
                    f"sequential {sorted(ref_ids.tolist())}"
                )

    def test_single_feature_segment(self):
        """Segment with exactly 1 feature, k=1 returns that feature."""
        table = _build_test_table(10)
        fid = t.tensor([7], dtype=t.int64)
        mag = t.tensor([42.0])

        results = select_top_k_by_percentile_batched(
            fid, mag, [1], table, t.tensor([1], dtype=t.int64),
        )
        assert results[0][0].tolist() == [7]
        assert results[0][1].tolist() == [42.0]

    def test_max_k_larger_than_some_segments(self):
        """k=[10,10,10] but segments=[3,15,7]; each gets min(k, seg_size)."""
        n_feat = 50
        table = _build_test_table(n_feat)

        seg1_fids = t.arange(0, 3, dtype=t.int64)
        seg2_fids = t.arange(10, 25, dtype=t.int64)
        seg3_fids = t.arange(30, 37, dtype=t.int64)

        seg1_mags = t.rand(3) * 100
        seg2_mags = t.rand(15) * 100
        seg3_mags = t.rand(7) * 100

        all_fids = t.cat([seg1_fids, seg2_fids, seg3_fids])
        all_mags = t.cat([seg1_mags, seg2_mags, seg3_mags])
        seg_sizes = [3, 15, 7]
        ks = t.tensor([10, 10, 10], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
        )

        expected = [3, 10, 7]  # min(10, seg_size)
        for i, (ids, _) in enumerate(results):
            assert ids.shape[0] == expected[i], (
                f"Segment {i}: expected {expected[i]}, got {ids.shape[0]}"
            )
            # All returned fids must come from the correct segment
            valid_fids = [seg1_fids, seg2_fids, seg3_fids][i].tolist()
            for fid in ids.tolist():
                assert fid in valid_fids, (
                    f"Segment {i}: fid {fid} not in valid set {valid_fids}"
                )


# ---------------------------------------------------------------------------
# TestVectorizedTopKVariableK
# ---------------------------------------------------------------------------


class TestVectorizedTopKVariableK:
    """Verify correct behavior with different k values per segment."""

    def test_different_k_per_segment(self):
        """k=[1, 5, 2] for 3 segments; verify lengths and reference equality."""
        table = _build_test_table(20)

        t.manual_seed(77)
        seg_fids = [
            t.arange(0, 6, dtype=t.int64),
            t.arange(6, 16, dtype=t.int64),
            t.arange(16, 20, dtype=t.int64),
        ]
        seg_mags = [t.rand(6) * 100, t.rand(10) * 100, t.rand(4) * 100]

        all_fids = t.cat(seg_fids)
        all_mags = t.cat(seg_mags)
        seg_sizes = [6, 10, 4]
        ks_list = [1, 5, 2]
        ks = t.tensor(ks_list, dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
        )

        for i in range(3):
            assert results[i][0].shape[0] == ks_list[i]
            ref_ids, _ = select_top_k_by_percentile(
                seg_fids[i], seg_mags[i], table, ks_list[i],
            )
            assert set(results[i][0].tolist()) == set(ref_ids.tolist()), (
                f"Segment {i}: batched {sorted(results[i][0].tolist())} != "
                f"sequential {sorted(ref_ids.tolist())}"
            )

    def test_k_zero_produces_empty(self):
        """Some segments have k=0; must produce empty tensors."""
        table = _build_test_table(10)

        fids = t.arange(0, 6, dtype=t.int64)
        mags = t.rand(6) * 100
        seg_sizes = [3, 3]
        ks = t.tensor([0, 2], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            fids, mags, seg_sizes, table, ks,
        )

        assert results[0][0].shape[0] == 0
        assert results[0][1].shape[0] == 0
        assert results[1][0].shape[0] == 2

    def test_k_equals_segment_size(self):
        """k exactly equals segment size; must return all features."""
        table = _build_test_table(10)

        fids = t.arange(0, 5, dtype=t.int64)
        mags = t.tensor([10.0, 50.0, 30.0, 20.0, 40.0])
        seg_sizes = [5]
        ks = t.tensor([5], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            fids, mags, seg_sizes, table, ks,
        )

        assert results[0][0].shape[0] == 5
        assert set(results[0][0].tolist()) == {0, 1, 2, 3, 4}


# ---------------------------------------------------------------------------
# TestVectorizedTopKOverrideWeights
# ---------------------------------------------------------------------------


class TestVectorizedTopKOverrideWeights:
    """Verify override_weights_list is respected in vectorized path."""

    def test_override_weights_respected(self):
        """Override weights reverse natural order; verify via reference."""
        table = _build_test_table(10)

        fids = t.arange(0, 5, dtype=t.int64)
        mags = t.tensor([10.0, 20.0, 30.0, 40.0, 50.0])
        override = t.tensor([100.0, 80.0, 60.0, 40.0, 20.0])

        # Batched with override
        results = select_top_k_by_percentile_batched(
            fids, mags, [5], table, t.tensor([3], dtype=t.int64),
            override_weights_list=[override],
        )

        # Sequential reference: select_top_k_by_percentile with override_weights
        ref_ids, _ = select_top_k_by_percentile(
            fids, mags, table, k=3, override_weights=override,
        )

        selected = set(results[0][0].tolist())
        ref_set = set(ref_ids.tolist())
        assert selected == ref_set, (
            f"Batched {sorted(selected)} != sequential {sorted(ref_set)}"
        )
        # With override, features 0, 1, 2 should be selected (highest weights)
        assert selected == {0, 1, 2}, (
            f"Expected {{0,1,2}} (highest override weights), got {selected}"
        )

    def test_mixed_override_and_percentile(self):
        """Segment 0 uses percentiles, segment 1 uses override weights."""
        table = _build_test_table(10)

        seg1_fids = t.arange(0, 5, dtype=t.int64)
        seg2_fids = t.arange(5, 10, dtype=t.int64)

        # Segment 1: feature 4 has highest magnitude -> highest percentile
        seg1_mags = t.tensor([10.0, 20.0, 30.0, 40.0, 200.0])
        # Segment 2: feature 5 has highest magnitude, but override will reverse
        seg2_mags = t.tensor([200.0, 10.0, 20.0, 30.0, 40.0])
        # Override for segment 2 reverses: feature 9 highest weight
        seg2_override = t.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

        all_fids = t.cat([seg1_fids, seg2_fids])
        all_mags = t.cat([seg1_mags, seg2_mags])
        seg_sizes = [5, 5]
        ks = t.tensor([2, 2], dtype=t.int64)

        results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
            override_weights_list=[None, seg2_override],
        )

        # Segment 0: percentile-based; feature 4 has mag=200 -> very high pct
        assert 4 in results[0][0].tolist(), (
            f"Segment 0: feature 4 (highest mag) should be selected, got {results[0][0].tolist()}"
        )

        # Segment 1: override-based; features 8 and 9 have weights 4.0 and 5.0
        selected_seg1 = set(results[1][0].tolist())
        assert selected_seg1 == {8, 9}, (
            f"Segment 1: expected {{8,9}} (highest override weights), got {selected_seg1}"
        )


# ---------------------------------------------------------------------------
# TestVectorizedTopKRegression
# ---------------------------------------------------------------------------


class TestVectorizedTopKRegression:
    """Regression tests comparing vectorized batched vs sequential reference."""

    def test_batched_matches_sequential_many_segments(self):
        """20 random segments: batched fid sets must match sequential calls."""
        n_feat = 10000
        stat = _uniform_stat()
        stats = {str(i): stat for i in range(n_feat)}
        table = build_percentile_table(stats, n_sae_features=n_feat)

        rng = t.Generator().manual_seed(999)
        segments_fids = []
        segments_mags = []
        seg_sizes = []
        ks_list = []

        for _ in range(20):
            sz = int(t.randint(1, 51, (1,), generator=rng).item())
            fids = t.randperm(n_feat, generator=rng)[:sz].to(t.int64)
            mags = t.rand(sz, generator=rng) * 100.0
            k = int(t.randint(1, sz + 1, (1,), generator=rng).item())
            segments_fids.append(fids)
            segments_mags.append(mags)
            seg_sizes.append(sz)
            ks_list.append(k)

        all_fids = t.cat(segments_fids)
        all_mags = t.cat(segments_mags)
        ks = t.tensor(ks_list, dtype=t.int64)

        batch_results = select_top_k_by_percentile_batched(
            all_fids, all_mags, seg_sizes, table, ks,
        )

        for i in range(20):
            seq_ids, _ = select_top_k_by_percentile(
                segments_fids[i], segments_mags[i], table, ks_list[i],
            )
            batch_set = set(batch_results[i][0].tolist())
            seq_set = set(seq_ids.tolist())
            assert batch_set == seq_set, (
                f"Segment {i} (size={seg_sizes[i]}, k={ks_list[i]}): "
                f"batched {sorted(batch_set)} != sequential {sorted(seq_set)}"
            )

    def test_deterministic_across_calls(self):
        """Same inputs, two calls produce identical results."""
        table = _build_test_table(20)

        rng = t.Generator().manual_seed(42)
        fids = t.randperm(20, generator=rng).to(t.int64)
        mags = t.rand(20, generator=rng) * 100.0
        seg_sizes = [7, 5, 8]
        ks = t.tensor([3, 4, 2], dtype=t.int64)

        r1 = select_top_k_by_percentile_batched(
            fids, mags, seg_sizes, table, ks,
        )
        r2 = select_top_k_by_percentile_batched(
            fids, mags, seg_sizes, table, ks,
        )

        for i in range(3):
            assert r1[i][0].tolist() == r2[i][0].tolist(), (
                f"Segment {i}: first call {r1[i][0].tolist()} != "
                f"second call {r2[i][0].tolist()}"
            )
            assert r1[i][1].tolist() == r2[i][1].tolist()
