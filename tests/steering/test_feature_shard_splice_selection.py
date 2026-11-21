"""Tests for seeded selection and derangement primitives (Task 3, issue #590).

Tests the seed, selection, derangement primitives that will power the new
batch select→derange→stitch engine.
"""

from __future__ import annotations

import random
from collections import deque

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.feature_shard_splice import (
    SpliceInternalError,
    _draw_cut_selection,
    _draw_derangement,
    _enumerate_splice_points,
    recombination_seed,
)


class TestRecombinationSeed:
    """Tests for recombination_seed: deterministic per-shard seeding."""

    def test_recombination_seed_golden(self):
        """Golden test: fixed inputs yield exact deterministic outputs.

        These values are pinned to the cross-version RNG contract;
        they were computed once and verified deterministic across runs.
        """
        # Computed once and verified deterministic
        assert recombination_seed(42, 0, 0.15) == 12946493011878250801
        assert recombination_seed(42, 1, 0.15) == 5952213469957664253
        assert recombination_seed(42, 0, 0.25) == 1477061519133991247

    def test_seed_varies_with_each_component(self):
        """Seed varies when any input component changes."""
        base_seed = recombination_seed(42, 0, 0.15)

        # Different global seed
        assert recombination_seed(43, 0, 0.15) != base_seed

        # Different shard index
        assert recombination_seed(42, 1, 0.15) != base_seed

        # Different cut fraction
        assert recombination_seed(42, 0, 0.25) != base_seed


class TestDrawCutSelection:
    """Tests for _draw_cut_selection: shuffled selection with pool."""

    def test_selection_golden(self):
        """Golden test: fixed seed yields exact selected and pool lists.

        These values are pinned to the RNG contract for the selection shuffle.
        """
        seed = recombination_seed(42, 0, 0.15)
        rng = random.Random(seed)

        n_points = 10
        target_k = 3
        selected, pool = _draw_cut_selection(rng, n_points, target_k)

        # Exact selected and pool from golden run
        assert selected == [4, 7, 8]
        assert list(pool) == [1, 6, 3, 2, 5, 0, 9]

    def test_selection_partition(self):
        """Selected and pool partition all points exactly once."""
        rng = random.Random(42)
        n_points = 20
        target_k = 7

        selected, pool = _draw_cut_selection(rng, n_points, target_k)

        assert len(selected) == target_k
        assert len(pool) == n_points - target_k
        assert sorted(selected + list(pool)) == list(range(n_points))

    def test_selection_deterministic_with_same_seed(self):
        """Two RNGs with the same seed produce identical selections."""
        rng1 = random.Random(12345)
        rng2 = random.Random(12345)

        n_points = 15
        target_k = 5

        selected1, pool1 = _draw_cut_selection(rng1, n_points, target_k)
        selected2, pool2 = _draw_cut_selection(rng2, n_points, target_k)

        assert selected1 == selected2
        assert list(pool1) == list(pool2)

    def test_selection_returns_deque_for_pool(self):
        """Pool is returned as a deque for efficient popleft redraw."""
        rng = random.Random(42)
        selected, pool = _draw_cut_selection(rng, 10, 3)

        assert isinstance(pool, deque)


class TestDrawDerangement:
    """Tests for _draw_derangement: uniform derangements via rejection sampling."""

    @given(seed=st.integers(min_value=0, max_value=2**31 - 1), k=st.integers(min_value=2, max_value=200))
    @settings(max_examples=50)
    def test_derangement_no_fixed_points(self, seed, k):
        """Derangements have no fixed points: sigma[i] != i for all i, across varied seeds."""
        rng = random.Random(seed)
        sigma = _draw_derangement(rng, k)

        assert len(sigma) == k
        assert all(sigma[i] != i for i in range(k))

    def test_derangement_uniform_smoke(self):
        """All 9 derangements of 4 elements appear across 2000 random seeds.

        Smoke test that _draw_derangement samples the uniform distribution
        (or at least approximates it well enough to hit all representatives).
        """
        # The 9 derangements of [0, 1, 2, 3] (no element in its original position)
        expected_derangements = [
            (1, 0, 3, 2),  # swap pairs
            (1, 2, 3, 0),  # cycle
            (1, 3, 0, 2),  # cycle
            (2, 0, 3, 1),  # cycle
            (2, 3, 0, 1),  # cycle
            (2, 3, 1, 0),  # mixed
            (3, 0, 1, 2),  # cycle
            (3, 2, 0, 1),  # mixed
            (3, 2, 1, 0),  # reverse
        ]

        seen = set()
        for seed in range(2000):
            rng = random.Random(seed)
            sigma = _draw_derangement(rng, 4)
            seen.add(tuple(sigma))

        # All expected derangements should appear
        for d in expected_derangements:
            assert d in seen, f"Derangement {d} not seen in {len(seen)} samples"

    def test_derangement_k1_raises(self):
        """k < 2 raises SpliceInternalError."""
        with pytest.raises(SpliceInternalError):
            _draw_derangement(random.Random(42), 0)

        with pytest.raises(SpliceInternalError):
            _draw_derangement(random.Random(42), 1)

    def test_derangement_deterministic_with_same_seed(self):
        """Two RNGs with the same seed produce identical derangements."""
        rng1 = random.Random(54321)
        rng2 = random.Random(54321)

        sigma1 = _draw_derangement(rng1, 10)
        sigma2 = _draw_derangement(rng2, 10)

        assert sigma1 == sigma2


class TestEnumerateSplicePoints:
    """Tests for _enumerate_splice_points: enumerate valid boundary pairs."""

    def test_enumerate_splice_points_shapes(self):
        """Splice points enumerate boundaries in lexicographic order.

        Single-segment docs contribute no boundaries (range(1-1)=range(0) is empty).
        Zero-segment docs likewise contribute nothing.
        Multi-segment docs contribute all boundaries in order.
        """
        # Three docs: first has 0 segments, second has 1 segment (the stated claim),
        # third has 3 segments (2 boundaries at 0 and 1).
        source_segments = [
            [],          # doc 0: 0 segments → no splice points
            [None],      # doc 1: 1 segment  → no splice points (demonstrates stated claim)
            [None, None, None],  # doc 2: 3 segments → boundaries at 0, 1
        ]

        points = _enumerate_splice_points(source_segments)  # ty: ignore[invalid-argument-type]

        # Only doc 2 contributes splice points.
        assert points == [(2, 0), (2, 1)]

    def test_enumerate_splice_points_all_single_segment(self):
        """All single-segment docs contribute no splice points."""
        source_segments = [
            [None],  # 1 segment
            [None],  # 1 segment
            [None],  # 1 segment
        ]

        points = _enumerate_splice_points(source_segments)  # ty: ignore[invalid-argument-type]

        assert points == []

    def test_enumerate_splice_points_multi_doc(self):
        """Multiple multi-segment docs enumerate in lexicographic order."""
        source_segments = [
            [None, None],  # doc 0: 2 segments (boundary 0)
            [None, None, None],  # doc 1: 3 segments (boundaries 0, 1)
            [None],  # doc 2: 1 segment (no boundaries)
            [None, None, None, None],  # doc 3: 4 segments (boundaries 0, 1, 2)
        ]

        points = _enumerate_splice_points(source_segments)  # ty: ignore[invalid-argument-type]

        expected = [
            (0, 0),
            (1, 0), (1, 1),
            (3, 0), (3, 1), (3, 2),
        ]
        assert points == expected

    def test_enumerate_splice_points_only_uses_len(self):
        """Only the lengths of source_segments are used, not contents."""
        # Create two equivalent source_segment lists (different contents, same lengths)
        ss1 = [
            [None, None],
            [None, None, None],
        ]

        ss2 = [
            [1, 2],
            ["a", "b", "c"],
        ]

        points1 = _enumerate_splice_points(ss1)  # ty: ignore[invalid-argument-type]
        points2 = _enumerate_splice_points(ss2)  # ty: ignore[invalid-argument-type]

        assert points1 == points2
