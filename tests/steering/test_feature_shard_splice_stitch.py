"""Tests for stitch, chain assembly, loop detection (Task 4, issue #590).

Tests the successor-graph construction and chain/loop reader that will power
the batch select→derange→stitch recombination engine.
"""

from __future__ import annotations

import random
from collections import Counter

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.feature_shard_splice import (
    SpliceInternalError,
    _build_successors,
    _read_chains,
)


class TestBuildSuccessors:
    """Tests for _build_successors: successor-map construction."""

    def test_successor_identity_no_cuts(self):
        """With no cuts, successors are the source segment adjacencies.

        One document with 3 segments => (0,0)->(0,1), (0,1)->(0,2), (0,2)->None.
        """
        source_segments = [[None, None, None]]  # 3 segments in doc 0
        succ = _build_successors(source_segments, [], [])  # ty: ignore[invalid-argument-type]

        assert succ[(0, 0)] == (0, 1)
        assert succ[(0, 1)] == (0, 2)
        assert succ[(0, 2)] is None

    def test_successor_multiple_docs_no_cuts(self):
        """Multiple docs, no cuts => each doc is a chain ending at None."""
        source_segments = [[None, None], [None, None, None]]  # doc 0: 2 segs, doc 1: 3 segs
        succ = _build_successors(source_segments, [], [])  # ty: ignore[invalid-argument-type]

        # Doc 0: (0,0)->(0,1)->(None)
        assert succ[(0, 0)] == (0, 1)
        assert succ[(0, 1)] is None

        # Doc 1: (1,0)->(1,1)->(1,2)->(None)
        assert succ[(1, 0)] == (1, 1)
        assert succ[(1, 1)] == (1, 2)
        assert succ[(1, 2)] is None

    def test_successor_cut_removes_edge_and_stitches(self):
        """A cut at (d, b) removes (d,b)->(d,b+1) and stitches to (dp,bp+1)."""
        source_segments = [[None] * 4]  # 4 segments in doc 0
        cuts = [(0, 1)]  # Cut at boundary 1: remove (0,1)->(0,2)
        sigma = [0]  # Stitch cut 0 to cut 0 (cycle back to itself)

        succ = _build_successors(source_segments, cuts, sigma)  # ty: ignore[invalid-argument-type]

        # Source adjacencies, then overwrite the cut edge
        assert succ[(0, 0)] == (0, 1)
        assert succ[(0, 1)] == (0, 2)  # Stitches to (cuts[0].boundary + 1) = (0, 2)
        assert succ[(0, 2)] == (0, 3)
        assert succ[(0, 3)] is None

    def test_successor_two_cuts_swap_derangement(self):
        """Spec minimal example: 1 doc, 4 segs, cuts at 0 and 2, sigma=[1,0]."""
        source_segments = [[None] * 4]  # Segments: D0, D1, D2, D3
        cuts = [(0, 0), (0, 2)]  # Cut at boundaries 0 and 2
        sigma = [1, 0]  # Swap: cut 0 stitches to cut 1, cut 1 stitches to cut 0

        succ = _build_successors(source_segments, cuts, sigma)  # ty: ignore[invalid-argument-type]

        # Original adjacencies: (0,0)->(0,1), (0,1)->(0,2), (0,2)->(0,3), (0,3)->None
        # After cuts:
        #   - (0,0) stitches to (cuts[1].doc, cuts[1].boundary+1) = (0, 3)
        #   - (0,2) stitches to (cuts[0].doc, cuts[0].boundary+1) = (0, 1)
        # Result:
        #   - (0,0) -> (0,3)
        #   - (0,1) -> (0,2)
        #   - (0,2) -> (0,1)
        #   - (0,3) -> None

        assert succ[(0, 0)] == (0, 3)
        assert succ[(0, 1)] == (0, 2)
        assert succ[(0, 2)] == (0, 1)
        assert succ[(0, 3)] is None

    def test_successor_cross_doc_stitch(self):
        """Cuts in different docs stitch across them."""
        source_segments = [[None, None], [None, None]]  # 2 docs, 2 segs each
        cuts = [(0, 0), (1, 0)]  # Cuts at (0,0) and (1,0)
        sigma = [1, 0]  # Swap

        succ = _build_successors(source_segments, cuts, sigma)  # ty: ignore[invalid-argument-type]

        # (0,0) stitches to (1, 1)
        # (1,0) stitches to (0, 1)
        assert succ[(0, 0)] == (1, 1)
        assert succ[(1, 0)] == (0, 1)


class TestReadChains:
    """Tests for _read_chains: chain and loop assembly."""

    def test_chains_identity_no_cuts(self):
        """No cuts => each doc is one chain from (d,0) to (d,len-1)."""
        source_segments = [[None, None, None], [None, None]]
        succ = {
            (0, 0): (0, 1),
            (0, 1): (0, 2),
            (0, 2): None,
            (1, 0): (1, 1),
            (1, 1): None,
        }
        chains, loops = _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]

        assert len(chains) == 2
        assert chains[0] == [(0, 0), (0, 1), (0, 2)]
        assert chains[1] == [(1, 0), (1, 1)]
        assert loops == []

    def test_spec_minimal_example(self):
        """Spec example: 1 doc, 4 segs, cuts at 0 and 2, sigma=[1,0].

        Expected: chain [(0,0), (0,3)], loop [(0,1), (0,2)].
        """
        source_segments = [[None] * 4]
        succ = {
            (0, 0): (0, 3),
            (0, 1): (0, 2),
            (0, 2): (0, 1),
            (0, 3): None,
        }
        chains, loops = _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]

        assert len(chains) == 1
        assert chains[0] == [(0, 0), (0, 3)]
        assert len(loops) == 1
        # Loop can be entered at (0,1) or (0,2); both are in the loop
        assert set(loops[0]) == {(0, 1), (0, 2)}
        # Check the loop is a 2-cycle
        assert len(loops[0]) == 2

    def test_multiple_separate_loops(self):
        """Multiple disjoint loops are detected separately."""
        source_segments = [[None] * 6]
        succ = {
            (0, 0): (0, 3),  # Chain root to tail
            (0, 1): (0, 2),  # Loop 1: (0,1) <-> (0,2)
            (0, 2): (0, 1),
            (0, 3): None,    # Chain ends
            (0, 4): (0, 5),  # Loop 2: (0,4) <-> (0,5)
            (0, 5): (0, 4),
        }
        chains, loops = _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]

        assert len(chains) == 1
        assert chains[0] == [(0, 0), (0, 3)]
        assert len(loops) == 2
        loop_sets = [set(loop) for loop in loops]
        assert {(0, 1), (0, 2)} in loop_sets
        assert {(0, 4), (0, 5)} in loop_sets

    def test_loop_walk_detects_re_entry(self):
        """Re-entering a visited segment in a chain raises SpliceInternalError."""
        source_segments = [[None] * 3]
        succ = {
            (0, 0): (0, 1),
            (0, 1): (0, 0),  # Cycle back to chain start
            (0, 2): None,
        }
        try:
            _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]
            raise AssertionError("Expected SpliceInternalError on cycle in chain")
        except SpliceInternalError as e:
            assert "chain re-entered" in str(e)

    def test_loop_walk_detects_none_in_loop(self):
        """A loop walk hitting None raises SpliceInternalError."""
        source_segments = [[None] * 3]
        succ = {
            (0, 0): None,
            (0, 1): (0, 2),
            (0, 2): None,  # Loop end hits None
        }
        try:
            _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]
            raise AssertionError("Expected SpliceInternalError on None in loop")
        except SpliceInternalError as e:
            assert "loop walk hit chain end" in str(e)


    def test_chain_ends_at_source_tail(self):
        """Every chain ends at a source-doc tail (d, len(doc_d)-1)."""
        source_segments = [[None] * 3, [None] * 2]
        succ = {
            (0, 0): (0, 1),
            (0, 1): (0, 2),
            (0, 2): None,
            (1, 0): (1, 1),
            (1, 1): None,
        }
        chains, loops = _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]

        for chain in chains:
            last_ref = chain[-1]
            d, seg = last_ref
            assert seg == len(source_segments[d]) - 1, \
                f"Chain doesn't end at tail: {chain}"

    def test_loop_never_contains_doc_root(self):
        """No loop contains any (d, 0) — roots are chain starts."""
        source_segments = [[None] * 6]
        succ = {
            (0, 0): (0, 3),
            (0, 1): (0, 2),
            (0, 2): (0, 1),
            (0, 3): None,
            (0, 4): (0, 5),
            (0, 5): (0, 4),
        }
        chains, loops = _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]

        for loop in loops:
            for ref in loop:
                d, seg = ref
                assert seg != 0, f"Loop contains root {ref}"

    def test_two_doc_cross_stitch_any_index(self):
        """Two docs with different segment counts, cuts at DIFFERENT boundaries.

        doc0: 5 segments, doc1: 4 segments
        Cuts at (0, 1) and (1, 2); sigma=[1, 0] (swap).
        Expected chains:
          - root 0: [(0,0), (0,1), (1,3)]
          - root 1: [(1,0), (1,1), (1,2), (0,2), (0,3), (0,4)]
        No loops.
        Verify any-index graft: at least one position has different ordinal.
        """
        source_segments = [[None] * 5, [None] * 4]  # doc0: 5 segs, doc1: 4 segs
        cuts = [(0, 1), (1, 2)]
        sigma = [1, 0]  # Swap: cut 0 stitches to cut 1, cut 1 stitches to cut 0

        succ = _build_successors(source_segments, cuts, sigma)  # ty: ignore[invalid-argument-type]

        chains, loops = _read_chains(source_segments, succ)  # ty: ignore[invalid-argument-type]

        # Verify chain count and structure
        assert len(chains) == 2
        assert chains[0] == [(0, 0), (0, 1), (1, 3)]
        assert chains[1] == [(1, 0), (1, 1), (1, 2), (0, 2), (0, 3), (0, 4)]
        assert loops == []

        # Verify any-index graft: some position has ordinal != position
        any_index_mismatch = any(
            ref[1] != pos for chain in chains for pos, ref in enumerate(chain)
        )
        assert any_index_mismatch, "Expected at least one position with different ordinal"

        # Explicit graft check: chains[0][2] == (1, 3) at output position 2 with src_ordinal 3
        assert chains[0][2] == (1, 3)


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    doc_count=st.integers(min_value=1, max_value=8),
    seg_per_doc=st.lists(
        st.integers(min_value=1, max_value=6),
        min_size=1,
        max_size=8,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_bijection_all_refs_covered_once(doc_count, seg_per_doc, seed):
    """Random doc shapes and valid cuts => refs partition correctly."""
    # Adjust seg_per_doc to match doc_count
    seg_per_doc = seg_per_doc[:doc_count]
    rng = random.Random(seed)
    while len(seg_per_doc) < doc_count:
        seg_per_doc.append(rng.randint(1, 6))

    source_segments = [[None] * n_seg for n_seg in seg_per_doc]

    # Compute all possible splice points
    all_splice_points = []
    for d in range(len(source_segments)):
        for b in range(len(source_segments[d]) - 1):
            all_splice_points.append((d, b))

    if len(all_splice_points) < 2:
        # Skip if we can't form valid cuts (need k >= 2)
        return

    # Draw k >= 2 random distinct cuts
    k = min(len(all_splice_points), rng.randint(2, min(6, len(all_splice_points))))
    cuts = rng.sample(all_splice_points, k)

    # Draw a random derangement
    sigma = list(range(k))
    while any(sigma[i] == i for i in range(k)):
        rng.shuffle(sigma)

    # Build successor map
    succ = _build_successors(source_segments, cuts, sigma)

    # Read chains and loops
    chains, loops = _read_chains(source_segments, succ)

    # Collect all refs from chains and loops as a list
    all_refs = []
    for chain in chains:
        for ref in chain:
            all_refs.append(ref)
    for loop in loops:
        for ref in loop:
            all_refs.append(ref)

    # Compute expected refs
    expected_refs = []
    for d in range(len(source_segments)):
        for seg in range(len(source_segments[d])):
            expected_refs.append((d, seg))

    assert Counter(all_refs) == Counter(expected_refs), \
        f"Refs mismatch: all_refs={Counter(all_refs)}, expected={Counter(expected_refs)}"


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    doc_count=st.integers(min_value=1, max_value=8),
    seg_per_doc=st.lists(
        st.integers(min_value=1, max_value=6),
        min_size=1,
        max_size=8,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_bijection_chain_ends_at_tail(doc_count, seg_per_doc, seed):
    """Every chain in bijection test ends at a source tail."""
    seg_per_doc = seg_per_doc[:doc_count]
    rng = random.Random(seed)
    while len(seg_per_doc) < doc_count:
        seg_per_doc.append(rng.randint(1, 6))

    source_segments = [[None] * n_seg for n_seg in seg_per_doc]

    all_splice_points = []
    for d in range(len(source_segments)):
        for b in range(len(source_segments[d]) - 1):
            all_splice_points.append((d, b))

    if len(all_splice_points) < 2:
        return

    k = min(len(all_splice_points), rng.randint(2, min(6, len(all_splice_points))))
    cuts = rng.sample(all_splice_points, k)

    sigma = list(range(k))
    while any(sigma[i] == i for i in range(k)):
        rng.shuffle(sigma)

    succ = _build_successors(source_segments, cuts, sigma)
    chains, loops = _read_chains(source_segments, succ)

    for chain in chains:
        last_ref = chain[-1]
        d, seg = last_ref
        assert seg == len(source_segments[d]) - 1


@settings(max_examples=50, suppress_health_check=[HealthCheck.too_slow])
@given(
    doc_count=st.integers(min_value=1, max_value=8),
    seg_per_doc=st.lists(
        st.integers(min_value=1, max_value=6),
        min_size=1,
        max_size=8,
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_bijection_loop_never_has_root(doc_count, seg_per_doc, seed):
    """No loop contains any (d, 0)."""
    seg_per_doc = seg_per_doc[:doc_count]
    rng = random.Random(seed)
    while len(seg_per_doc) < doc_count:
        seg_per_doc.append(rng.randint(1, 6))

    source_segments = [[None] * n_seg for n_seg in seg_per_doc]

    all_splice_points = []
    for d in range(len(source_segments)):
        for b in range(len(source_segments[d]) - 1):
            all_splice_points.append((d, b))

    if len(all_splice_points) < 2:
        return

    k = min(len(all_splice_points), rng.randint(2, min(6, len(all_splice_points))))
    cuts = rng.sample(all_splice_points, k)

    sigma = list(range(k))
    while any(sigma[i] == i for i in range(k)):
        rng.shuffle(sigma)

    succ = _build_successors(source_segments, cuts, sigma)
    chains, loops = _read_chains(source_segments, succ)

    for loop in loops:
        for ref in loop:
            d, seg = ref
            assert seg != 0
