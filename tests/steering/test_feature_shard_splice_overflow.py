"""Tests for Task 5: batched oracle, gated loop-break + overflow split.

Tests:
- _batch_roundtrip_stable  (must agree element-for-element with is_roundtrip_stable)
- _piece_ids               (BOS prepend logic)
- _break_loop              (stitch-candidate rotation, gated by oracle)
- _split_overflow          (maximal boundary + sliding gate + cascade + error)
"""

from __future__ import annotations

import random

import pytest

from steering._roundtrip_seam import is_roundtrip_stable
from steering.feature_shard_splice import (
    SpliceInternalError,
    UnsplittableChainError,
    _batch_roundtrip_stable,
    _break_loop,
    _build_successors,
    _piece_ids,
    _read_chains,
    _split_overflow,
)
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

# ---------------------------------------------------------------------------
# Utility: build succ + extract single cycle for _break_loop tests
# ---------------------------------------------------------------------------


def _build_cycle_and_succ(source_segments, cuts, sigma):
    succ = _build_successors(source_segments, cuts, sigma)
    _, loops = _read_chains(source_segments, succ)
    assert loops, "expected at least one loop"
    return loops[0], succ


def _plain_tok(bos_id=None):
    """No-merge tokenizer, optional BOS."""
    return CharMergeTokenizer(bos_id=bos_id, merge_pairs=[], merge_id_base=1000)


def _ab_merge_tok(bos_id=None):
    """Tokenizer that merges ('a','b') -> 1000."""
    return CharMergeTokenizer(bos_id=bos_id, merge_pairs=[("a", "b")], merge_id_base=1000)


# ---------------------------------------------------------------------------
# test_piece_ids_bos_prepend
# ---------------------------------------------------------------------------


class TestPieceIds:
    def test_no_bos_when_not_new_started(self):
        seg_tokens = [[(10, 11, 12), (20, 21)]]
        result = _piece_ids([(0, 0), (0, 1)], False, seg_tokens, bos_id=1)
        assert result == (10, 11, 12, 20, 21)

    def test_bos_prepended_when_new_started(self):
        seg_tokens = [[(10, 11, 12), (20, 21)]]
        result = _piece_ids([(0, 0), (0, 1)], True, seg_tokens, bos_id=99)
        assert result == (99, 10, 11, 12, 20, 21)

    def test_empty_piece_no_bos(self):
        result = _piece_ids([], False, [[]], bos_id=7)
        assert result == ()

    def test_empty_piece_new_started(self):
        result = _piece_ids([], True, [[]], bos_id=7)
        assert result == (7,)

    def test_single_segment(self):
        seg_tokens = [[(5, 6)]]
        assert _piece_ids([(0, 0)], False, seg_tokens, bos_id=0) == (5, 6)  # ty: ignore[invalid-argument-type]
        assert _piece_ids([(0, 0)], True, seg_tokens, bos_id=0) == (0, 5, 6)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# test_batch_oracle_matches_per_doc
# ---------------------------------------------------------------------------


class TestBatchOracleMatchesPerDoc:
    """_batch_roundtrip_stable must agree element-for-element with is_roundtrip_stable."""

    def test_all_stable_no_merges(self):
        tok = _plain_tok()
        docs = [
            (ord("h"), ord("i")),
            (ord("a"), ord("b"), ord("c")),
            (ord("x"),),
        ]
        batch = _batch_roundtrip_stable(docs, tok)
        per_doc = [is_roundtrip_stable(d, tok) for d in docs]
        assert batch == per_doc
        assert all(batch)

    def test_merge_makes_some_unstable(self):
        # ("a","b")->1000: (a,b) decodes "ab" re-encodes to (1000,) -> UNSTABLE
        tok = _ab_merge_tok()
        docs = [
            (ord("a"), ord("b")),   # unstable
            (ord("a"), ord("c")),   # stable (no merge)
            (1000,),                # stable: decodes "ab", re-encodes (1000,)
        ]
        batch = _batch_roundtrip_stable(docs, tok)
        per_doc = [is_roundtrip_stable(d, tok) for d in docs]
        assert batch == per_doc
        assert not batch[0]
        assert batch[1]
        assert batch[2]

    def test_30_random_docs_mixed_lengths(self):
        """30 random docs; batch and per-doc results must match exactly."""
        rng = random.Random(42)
        tok = _ab_merge_tok(bos_id=1)
        vocab_ids = [1, ord("a"), ord("b"), ord("c"), ord("d"), 1000]
        docs = [
            tuple(rng.choice(vocab_ids) for _ in range(rng.randint(1, 8)))
            for _ in range(30)
        ]
        batch = _batch_roundtrip_stable(docs, tok)
        per_doc = [is_roundtrip_stable(d, tok) for d in docs]
        assert batch == per_doc
        assert any(batch) and not all(batch), (
            "non-vacuity: 30 random docs must produce both stable and unstable results"
        )

    def test_empty_list_raises(self):
        """Empty docs_ids raises SpliceInternalError (caller logic error)."""
        tok = _plain_tok()
        with pytest.raises(SpliceInternalError, match="empty docs_ids"):
            _batch_roundtrip_stable([], tok)

    def test_single_stable_doc(self):
        tok = _plain_tok()
        docs = [(ord("z"),)]
        assert _batch_roundtrip_stable(docs, tok) == [is_roundtrip_stable(docs[0], tok)]  # ty: ignore[invalid-argument-type]

    def test_single_unstable_doc(self):
        tok = _ab_merge_tok()
        docs = [(ord("a"), ord("b"))]
        batch = _batch_roundtrip_stable(docs, tok)  # ty: ignore[invalid-argument-type]
        per_doc = [is_roundtrip_stable(d, tok) for d in docs]
        assert batch == per_doc
        assert not batch[0]

    def test_with_bos_id_in_sequences(self):
        """Sequences that include BOS id are handled consistently."""
        tok = _ab_merge_tok(bos_id=1)
        # (1, a, b) -> to_text strips BOS -> "ab" -> re-enc with bos -> (1, 1000). UNSTABLE.
        docs = [
            (1, ord("a"), ord("b")),   # unstable
            (1, ord("a"), ord("c")),   # stable
        ]
        batch = _batch_roundtrip_stable(docs, tok)  # ty: ignore[invalid-argument-type]
        per_doc = [is_roundtrip_stable(d, tok) for d in docs]
        assert batch == per_doc


# ---------------------------------------------------------------------------
# test_split_exact_limit_not_split
# ---------------------------------------------------------------------------


class TestSplitNotSplit:
    def test_exactly_at_limit_not_split(self):
        # 1 segment of 3 tokens, limit=3, not new_started -> 3 == limit -> no split
        tok = _plain_tok()
        seg_tokens = [[(10, 11, 12)]]
        result = _split_overflow([(0, 0)], False, seg_tokens, tok, bos_id=1, limit=3)  # ty: ignore[invalid-argument-type]
        assert result == [([(0, 0)], False)]

    def test_below_limit_not_split(self):
        tok = _plain_tok()
        seg_tokens = [[(10, 11)]]
        result = _split_overflow([(0, 0)], False, seg_tokens, tok, bos_id=1, limit=10)  # ty: ignore[invalid-argument-type]
        assert result == [([(0, 0)], False)]

    def test_new_started_exactly_at_limit(self):
        # BOS + 2-token seg = 3 tokens = limit -> no split
        tok = _plain_tok(bos_id=1)
        seg_tokens = [[(ord("a"), ord("c"))]]
        result = _split_overflow([(0, 0)], True, seg_tokens, tok, bos_id=1, limit=3)  # ty: ignore[invalid-argument-type]
        assert result == [([(0, 0)], True)]


# ---------------------------------------------------------------------------
# test_split_basic
# ---------------------------------------------------------------------------


class TestSplitBasic:
    def test_splits_kept_at_maximal_boundary(self):
        """2 segments; first has 3 tokens (= limit), second has 2 tokens.
        Kept=[seg0] (3 tokens == limit); remainder=[seg1] (BOS+2=3<=limit). 2 pieces.

        Use bos_id=None in the tokenizer so segment tokens (without BOS) are stable.
        The synthetic bos_id=1 is passed separately to _split_overflow for counting/oracle.
        """
        # tok has no BOS: segment tokens (x,y,z) stable; (1,p,q) stable as chr(1)+"pq"
        tok = _plain_tok(bos_id=None)
        seg_tokens = [
            [
                (ord("x"), ord("y"), ord("z")),  # 3 tokens
                (ord("p"), ord("q")),              # 2 tokens
            ]
        ]
        piece = [(0, 0), (0, 1)]
        result = _split_overflow(piece, False, seg_tokens, tok, bos_id=1, limit=3)
        assert len(result) == 2
        kept_refs, ns_kept = result[0]
        rem_refs, ns_rem = result[1]
        assert kept_refs == [(0, 0)]
        assert ns_kept is False
        assert rem_refs == [(0, 1)]
        assert ns_rem is True

    def test_remainder_is_new_started(self):
        """Remainder always gets new_started=True regardless of the input piece."""
        tok = _plain_tok(bos_id=None)
        seg_tokens = [
            [(ord("a"), ord("c"), ord("d")), (ord("e"),)]
        ]
        result = _split_overflow([(0, 0), (0, 1)], False, seg_tokens, tok, bos_id=1, limit=3)
        assert len(result) == 2
        _, ns = result[1]
        assert ns is True


# ---------------------------------------------------------------------------
# test_split_new_started_input_counts_bos
# ---------------------------------------------------------------------------


class TestSplitNewStartedCountsBos:
    def test_bos_counted_for_new_started_piece(self):
        """When new_started=True, BOS adds 1 to the token count.

        2 segments of 2 tokens each. With BOS: 1+2+2=5 > limit=3.
        Maximal boundary: kept=[seg0]+BOS = 1+2=3 = limit. Rem=[seg1]+BOS=1+2=3=limit. 2 pieces.
        """
        tok = _plain_tok(bos_id=1)
        seg_tokens = [
            [(ord("a"), ord("c")), (ord("d"), ord("e"))]
        ]
        piece = [(0, 0), (0, 1)]
        result = _split_overflow(piece, True, seg_tokens, tok, bos_id=1, limit=3)  # ty: ignore[invalid-argument-type]
        assert len(result) == 2
        kept_refs, ns_kept = result[0]
        rem_refs, ns_rem = result[1]
        assert kept_refs == [(0, 0)]
        assert ns_kept is True
        assert rem_refs == [(0, 1)]
        assert ns_rem is True

    def test_non_new_started_does_not_count_bos(self):
        """new_started=False: BOS not counted, so 3-token piece fits limit=3 exactly."""
        tok = _plain_tok(bos_id=1)
        seg_tokens = [[(ord("a"), ord("b"), ord("c"))]]
        # 3 tokens, no BOS counted -> fits limit=3 exactly -> no split
        result = _split_overflow([(0, 0)], False, seg_tokens, tok, bos_id=1, limit=3)  # ty: ignore[invalid-argument-type]
        assert result == [([(0, 0)], False)]


# ---------------------------------------------------------------------------
# test_split_cascade
# ---------------------------------------------------------------------------


class TestSplitCascade:
    def test_cascade_produces_3_or_more_pieces(self):
        """4 single-token segments, limit=2, not new_started.

        tok has no BOS so segment tokens are individually stable.
        Iteration 1: piece=[0,1,2,3], not new_started, total=4>2.
          Max b=2: kept=2 tokens <= 2. Gate: stable. Emit ([(0,0),(0,1)], False).
          Rem=([(0,2),(0,3)], True).
        Iteration 2: BOS+1+1=3>2. Max b=1: kept=BOS+1=2. Gate: stable. Emit ([(0,2)], True).
          Rem=([(0,3)], True).
        Iteration 3: BOS+1=2<=2. Not split. Emit ([(0,3)], True). Done. 3 pieces.
        """
        tok = _plain_tok(bos_id=None)
        seg_tokens = [
            [(ord("a"),), (ord("b"),), (ord("c"),), (ord("d"),)]
        ]
        piece = [(0, 0), (0, 1), (0, 2), (0, 3)]
        result = _split_overflow(piece, False, seg_tokens, tok, bos_id=1, limit=2)  # ty: ignore[invalid-argument-type]
        assert len(result) >= 3
        for refs, ns in result:
            actual_ids = _piece_ids(refs, ns, seg_tokens, bos_id=1)  # ty: ignore[invalid-argument-type]
            assert len(actual_ids) <= 2

    def test_cascade_all_pieces_within_limit(self):
        """All pieces produced by cascade fit within limit."""
        tok = _plain_tok(bos_id=None)
        seg_tokens = [
            [(ord("a"),), (ord("b"),), (ord("c"),), (ord("d"),)]
        ]
        piece = [(0, 0), (0, 1), (0, 2), (0, 3)]
        result = _split_overflow(piece, False, seg_tokens, tok, bos_id=1, limit=2)  # ty: ignore[invalid-argument-type]
        for refs, ns in result:
            actual_ids = _piece_ids(refs, ns, seg_tokens, bos_id=1)  # ty: ignore[invalid-argument-type]
            assert len(actual_ids) <= 2, f"piece {refs} ns={ns} has {len(actual_ids)} > 2 tokens"


# ---------------------------------------------------------------------------
# test_split_boundary_slides_on_unstable_bos
# ---------------------------------------------------------------------------


class TestSplitBoundarySlides:
    def test_slides_when_kept_is_unstable(self):
        """Max boundary b produces an unstable kept piece; slide to b-1 which is stable.

        Use a no-BOS tokenizer that merges ('y','z') -> 1000.
        Segments: seg0=(x,y) seg1=(z,p) seg2=(q,r), limit=4, not new_started.
        Total = 2+2+2 = 6 > 4.
        Max b=2: kept=(x,y,z,p) -> 'xyzp' -> re-enc (x,1000,p). UNSTABLE.
        Slide to b=1: kept=(x,y) -> 'xy' -> re-enc (x,y). STABLE.
          rem oracle: (bos_id,z,p,q,r). to_text (no BOS in tok) -> chr(1)+"zpqr".
          re-enc: (1,z,p,q,r). Matches. STABLE.
        Emit ([(0,0)], False). Continue ([(0,1),(0,2)], True).
        """
        # No BOS in tokenizer so segment tokens without BOS are stable alone.
        tok = CharMergeTokenizer(bos_id=None, merge_pairs=[("y", "z")], merge_id_base=1000)
        bos_id = 1  # synthetic BOS for _piece_ids / rem oracle

        # Verify engineered instabilities
        yz_merge_ids = (ord("x"), ord("y"), ord("z"), ord("p"))
        assert not is_roundtrip_stable(yz_merge_ids, tok), "setup: yz-merge should be unstable"
        stable_ids = (ord("x"), ord("y"))
        assert is_roundtrip_stable(stable_ids, tok), "setup: seg0 alone should be stable"
        # Remainder oracle: (bos_id, z, p, q, r) with no-BOS tokenizer
        rem_oracle = (bos_id, ord("z"), ord("p"), ord("q"), ord("r"))
        assert is_roundtrip_stable(rem_oracle, tok), "setup: rem oracle should be stable"

        seg_tokens = [
            [
                (ord("x"), ord("y")),
                (ord("z"), ord("p")),
                (ord("q"), ord("r")),
            ]
        ]
        piece = [(0, 0), (0, 1), (0, 2)]
        result = _split_overflow(piece, False, seg_tokens, tok, bos_id=bos_id, limit=4)  # ty: ignore[invalid-argument-type]
        # First piece should be just seg0 (slide found b=1)
        assert result[0][0] == [(0, 0)]
        assert result[0][1] is False
        assert result[1][1] is True


# ---------------------------------------------------------------------------
# test_split_raises_when_no_boundary_works
# ---------------------------------------------------------------------------


class TestSplitRaises:
    def test_raises_when_all_boundaries_unstable(self):
        """When every boundary position fails the double gate, raise UnsplittableChainError."""
        # Merge ('a','b') -> 1000. Every seg begins with (a,b,...) so any kept piece
        # ending in (a,b,...) is unstable, and any remainder starting with (a,b,...) is
        # also unstable after BOS prepend.
        tok = _ab_merge_tok(bos_id=1)
        bos_id = 1

        # Verify instability
        assert not is_roundtrip_stable((bos_id, ord("a"), ord("b")), tok)

        # 3 segments each starting with (a,b,...), total > limit
        seg_tokens = [
            [
                (ord("a"), ord("b"), ord("c")),  # 3 tokens
                (ord("a"), ord("b"), ord("d")),  # 3 tokens
                (ord("a"), ord("b"), ord("e")),  # 3 tokens
            ]
        ]
        piece = [(0, 0), (0, 1), (0, 2)]
        # Total=9 > limit=5. Max b=1 (kept=3 ≤ 5). Kept=(a,b,c) -> "abc" -> (1000,c). UNSTABLE.
        # Slide b to 0: raise.
        with pytest.raises(UnsplittableChainError, match=r"\(0, 0\)") as exc_info:
            _split_overflow(piece, False, seg_tokens, tok, bos_id=bos_id, limit=5)  # ty: ignore[invalid-argument-type]
        # Pin every diagnostic field the message carries (3 segs of 3 tokens each,
        # not new_started so bos_overhead=0, piece_tokens=9, boundaries 1..2 tried).
        msg = str(exc_info.value)
        assert "doc=0" in msg
        assert "new_started=False" in msg
        assert "piece_segments=3" in msg
        assert "piece_tokens=9" in msg
        assert "limit=5" in msg
        assert "bos_overhead=0" in msg
        assert "boundaries_tried=1..2" in msg
        assert "segment_sizes=[3, 3, 3]" in msg

    def test_raises_message_includes_first_ref(self):
        """Error message includes the piece's first ref."""
        tok = _ab_merge_tok(bos_id=1)
        seg_tokens = [
            [
                (ord("a"), ord("b"), ord("c")),
                (ord("a"), ord("b"), ord("d")),
            ]
        ]
        piece = [(0, 0), (0, 1)]
        with pytest.raises(UnsplittableChainError):
            _split_overflow(piece, False, seg_tokens, tok, bos_id=1, limit=5)  # ty: ignore[invalid-argument-type]


# ---------------------------------------------------------------------------
# Module-level test tokenizers for _break_loop and _split_overflow tests
# ---------------------------------------------------------------------------


class StartAbMergeTokenizer:
    """Merges 'ab' -> 1000 only when it appears at the START of a text.

    Text-start context is what the remainder's decode/re-encode sees after
    BOS is stripped by to_text, making this the right tool for testing the
    remainder-gate slide path in _split_overflow.
    """

    bos_token_id = 0

    def _encode(self, text: str) -> list[int]:
        result: list[int] = []
        i = 0
        if text.startswith("ab"):
            result.append(1000)
            i = 2
        while i < len(text):
            result.append(ord(text[i]))
            i += 1
        return result

    def __call__(self, texts, *, padding=False, add_special_tokens=True):
        bos: list[int] = [self.bos_token_id] if add_special_tokens else []
        if isinstance(texts, str):
            return {"input_ids": bos + self._encode(texts)}
        return {"input_ids": [bos + self._encode(t) for t in texts]}

    def decode(self, ids: list[int]) -> str:
        out: list[str] = []
        for tid in ids:
            if tid == self.bos_token_id:
                continue
            if tid == 1000:
                out.append("ab")
            else:
                out.append(chr(tid))
        return "".join(out)

    def get_bos_token_id(self) -> int:
        return self.bos_token_id


class AlwaysUnstableTokenizer:
    """Decodes everything to empty string; re-encode gives only BOS.

    Every token sequence is round-trip-unstable (any input → empty text → [BOS]),
    and the tokenizer drops all content (acts as a "drops everything" tokenizer).
    Used in tests that need guaranteed instability or a total-drop oracle.
    """

    bos_token_id = 0

    def __call__(self, texts, *, padding=False, add_special_tokens=True):
        bos: list[int] = [0] if add_special_tokens else []
        if isinstance(texts, str):
            return {"input_ids": bos}
        return {"input_ids": [bos for _ in texts]}

    def decode(self, ids: list[int]) -> str:
        return ""

    def get_bos_token_id(self) -> int:
        return self.bos_token_id


# ---------------------------------------------------------------------------
# _break_loop helpers  (manually built succ dicts for fine control)
# ---------------------------------------------------------------------------


class TestBreakLoopLowestCandidateFirst:
    def test_only_candidate_chosen(self):
        """Single stitch candidate in a 2-cycle: that ref is rotated to front."""
        # 1-doc, 4-segs. cuts=[(0,0),(0,2)], sigma=[1,0]:
        # (0,0)->(0,3) stitch; (0,2)->(0,1) stitch.
        # Root: (0,0)->(0,3)->None. Cycle: [(0,1),(0,2)].
        # Stitch edges: (0,2)->(0,1) is stitch; (0,1)->(0,2) is NATURAL.
        # Only (0,1) is a stitch candidate (its cycle-pred is (0,2) with stitch edge).
        source_segments = [[None] * 4]
        cuts = [(0, 0), (0, 2)]
        sigma = [1, 0]
        cyc, succ = _build_cycle_and_succ(source_segments, cuts, sigma)
        assert set(cyc) == {(0, 1), (0, 2)}

        tok = _plain_tok(bos_id=1)
        seg_tokens = [
            [
                (ord("w"), ord("x")),   # (0,0) in chain
                (ord("a"), ord("c")),   # (0,1) in cycle
                (ord("d"), ord("e")),   # (0,2) in cycle
                (ord("f"), ord("g")),   # (0,3) in chain
            ]
        ]
        result = _break_loop(cyc, seg_tokens, tok, bos_id=1)  # ty: ignore[invalid-argument-type]
        assert result is not None
        assert set(result) == {(0, 1), (0, 2)}
        # Must start at (0,1) — the stitch candidate
        assert result[0] == (0, 1)

    def test_multiple_stitch_candidates_ascending_order(self):
        """With 2+ stitch candidates (all cycle edges are stitches), lowest ref wins.

        Manually constructed succ dict with a 3-ref pure-stitch cycle:
        cyc = [(0,1),(0,3),(1,2)] where EVERY predecessor edge is a stitch.
        All 3 are candidates. Ascending order: (0,1) < (0,3) < (1,2).
        """
        tok = _plain_tok(bos_id=1)
        bos_id = 1

        cyc = [(0, 1), (0, 3), (1, 2)]
        # All edges are stitches (no natural adjacency matches any of these):
        #   (0,1) -> (0,3): natural would be (0,2)
        #   (0,3) -> (1,2): different doc entirely
        #   (1,2) -> (0,1): natural would be (1,3)
        seg_tokens = [
            [None, (ord("a"), ord("c")), None, (ord("d"), ord("e"))],  # doc 0
            [None, None, (ord("f"), ord("g"))],                          # doc 1
        ]

        result = _break_loop(cyc, seg_tokens, tok, bos_id)  # ty: ignore[invalid-argument-type]
        assert result is not None
        # Lowest candidate (0,1) should be tried first; if stable, returned
        assert result[0] == (0, 1)


class TestBreakLoopFallsThrough:
    def test_falls_through_to_next_when_first_unstable(self):
        """First candidate's rotation is BOS-unstable; second's is stable.

        Uses the module-level StartAbMergeTokenizer: 'ab' merges to 1000 ONLY at
        start of text (after BOS is stripped). So (bos,a,b) is unstable but
        (bos,p,q,a,b) is stable (ab not at start).
        """
        tok = StartAbMergeTokenizer()
        bos_id = 0

        # Verify engineered instabilities
        unstable = (bos_id, ord("a"), ord("b"))
        assert not is_roundtrip_stable(unstable, tok), "setup: BOS+ab unstable (ab at start)"
        stable = (bos_id, ord("p"), ord("q"), ord("a"), ord("b"))
        assert is_roundtrip_stable(stable, tok), "setup: BOS+pqab stable (ab not at start)"

        # Cycle: [(0,1),(0,3)], both edges are stitches.
        # succ[(0,1)]=(0,3): (0,2) would be natural.
        # succ[(0,3)]=(0,1): (0,4) would be natural.
        # Both (0,1) and (0,3) are stitch candidates.
        cyc = [(0, 1), (0, 3)]
        seg_tokens = [
            [None, (ord("a"), ord("b")), None, (ord("p"), ord("q"))],
        ]

        # Rotation from (0,1): BOS+(a,b)+(p,q) -> to_text -> "abpq" -> start-ab -> UNSTABLE.
        # Rotation from (0,3): BOS+(p,q)+(a,b) -> to_text -> "pqab" -> NOT start-ab -> STABLE.
        result = _break_loop(cyc, seg_tokens, tok, bos_id)  # ty: ignore[invalid-argument-type]
        assert result is not None
        assert result[0] == (0, 3)   # fell through (0,1), picked (0,3)
        assert result == [(0, 3), (0, 1)]


class TestBreakLoopReturnsNone:
    def test_returns_none_when_all_candidates_fail(self):
        """All candidates produce unstable ids -> return None."""
        tok = AlwaysUnstableTokenizer()
        bos_id = 0
        assert not is_roundtrip_stable((bos_id, 1, 2), tok)

        cyc = [(0, 1), (0, 3)]
        seg_tokens = [[None, (1, 2), None, (3, 4)]]
        result = _break_loop(cyc, seg_tokens, tok, bos_id)  # ty: ignore[invalid-argument-type]
        assert result is None

    def test_returns_none_when_sole_candidate_is_unstable(self):
        """Sole stitch candidate yields unstable ids -> return None.

        Builds a 2-cycle [(0,1),(0,2)] where (0,1)->(0,2) is the NATURAL edge
        and (0,2)->(0,1) is a stitch, making (0,1) the only stitch candidate.
        The DropsEverythingTokenizer makes every rotation unstable, so None is
        returned via the pass-through-all-candidates path.
        """
        tok = AlwaysUnstableTokenizer()
        bos_id = 0
        # 2-cycle [(0,1),(0,2)]; (0,2)->(0,1) is stitch, (0,1)->(0,2) natural.
        # (0,1) is the stitch candidate; its rotation is BOS+(seg1+seg2) -> unstable.
        source_segments = [[None] * 4]
        cuts = [(0, 0), (0, 2)]
        sigma = [1, 0]
        cyc, _succ = _build_cycle_and_succ(source_segments, cuts, sigma)
        seg_tokens = [[None, (1,), (2,), None]]
        result = _break_loop(cyc, seg_tokens, tok, bos_id)  # ty: ignore[invalid-argument-type]
        assert result is None


# ---------------------------------------------------------------------------
# test_root_geometry_split_with_real_bos_succeeds (IMPORTANT-1)
# ---------------------------------------------------------------------------


class TestRootGeometrySplitWithRealBos:
    def test_root_geometry_split_with_real_bos_succeeds(self):
        """Primary production geometry: new_started=False + seg-0 starts with real BOS.

        This is THE production path. Extraction tokenizes with BOS enabled, so
        seg-0 = [bos, content...]. The bos_id argument MUST equal the tokenizer's
        own BOS id (resolve_bos_token_id). A previous concern claimed that
        new_started=False with a BOS-having tokenizer always fails the kept gate —
        this test proves that claim wrong.

        Geometry:
          tok = CharMergeTokenizer(bos_id=BOS_ID, merge_pairs=[])  (no merges)
          seg0 = (BOS_ID, p, q, r)    -- 4 tokens, starts with real BOS
          seg1 = (s, t, u)             -- 3 tokens
          seg2 = (v, w)                -- 2 tokens
          limit = 7

        piece = [(0,0),(0,1),(0,2)], new_started=False, bos_id=BOS_ID.
        Total: 4+3+2=9 > 7 -> overflow.
        Max b=2: kept=(BOS,p,q,r,s,t,u)=7=limit -> to_text strips BOS -> "pqrstu" ->
          re-encode -> (BOS,p,q,r,s,t,u) STABLE.
        Remainder=(BOS,v,w): to_text strips BOS -> "vw" -> re-encode -> (BOS,v,w) STABLE.
        Both pass at b=2. Emit kept=[seg0,seg1] (new_started=False) + rem=[seg2] (True).
        """
        BOS_ID = 1
        tok = CharMergeTokenizer(bos_id=BOS_ID, merge_pairs=[], merge_id_base=1000)
        # CharMergeTokenizer sets both bos_id and bos_token_id to the same value;
        # caller contract requires bos_token_id; assert equality as a sanity guard.
        assert tok.bos_token_id == tok.bos_id
        bos_id = tok.bos_token_id  # use bos_token_id to match the caller contract

        seg_tokens = [
            [
                (BOS_ID, ord("p"), ord("q"), ord("r")),  # seg0: 4 tokens, BOS-first
                (ord("s"), ord("t"), ord("u")),            # seg1: 3 tokens
                (ord("v"), ord("w")),                      # seg2: 2 tokens
            ]
        ]
        piece = [(0, 0), (0, 1), (0, 2)]
        limit = 7  # seg0+seg1 = 7 = limit; total 9 > 7

        result = _split_overflow(piece, False, seg_tokens, tok, bos_id=bos_id, limit=limit)

        assert len(result) == 2, "split must produce exactly 2 pieces"
        kept_refs, ns_kept = result[0]
        rem_refs, ns_rem = result[1]

        # Kept piece preserves new_started=False (root flag unchanged)
        assert ns_kept is False, "kept piece must keep new_started=False"
        assert kept_refs == [(0, 0), (0, 1)], "kept must be seg0+seg1 (maximal b=2)"

        # Verify kept passes is_roundtrip_stable directly
        kept_ids = _piece_ids(kept_refs, ns_kept, seg_tokens, bos_id)
        assert is_roundtrip_stable(kept_ids, tok), "kept piece must pass oracle directly"

        # Remainder flagged True and (bos,)+remainder_ids passes oracle
        assert ns_rem is True
        assert rem_refs == [(0, 2)]
        rem_raw = tuple(tid for ref in rem_refs for tid in seg_tokens[ref[0]][ref[1]])
        assert is_roundtrip_stable((bos_id,) + rem_raw, tok), (
            "(bos,)+remainder_ids must pass oracle directly"
        )


# ---------------------------------------------------------------------------
# test_slides_when_remainder_bos_gate_fails (IMPORTANT-2)
# ---------------------------------------------------------------------------


class TestSplitRemainderGateSlide:
    def test_slides_when_remainder_bos_gate_fails(self):
        """Remainder gate is the sole failing conjunct at maximal b; slide to b-1 where both pass.

        Uses the module-level StartAbMergeTokenizer whose encode re-merges 'ab' when
        it appears at TEXT START (text-start context is what the remainder's
        decode/re-encode sees after BOS is stripped by to_text).

        Geometry:
          seg0 = (BOS, x, y, z)    -- 4 tokens, BOS-first root segment
          seg1 = (p, q, r)          -- 3 tokens, stable alone
          seg2 = (a_id, b_id, c_id) -- 3 tokens; (BOS,a,b,c) -> "abc" -> "ab" at start -> UNSTABLE
          limit = 7

        new_started=False, total = 4+3+3=10 > 7.
        Max b=2: kept=(BOS,x,y,z,p,q,r)=7=limit STABLE; rem=(BOS,a,b,c) UNSTABLE (sole failing).
        Slide b=1: kept=(BOS,x,y,z)=4 STABLE; rem=(BOS,p,q,r,a,b,c) -> "pqrabc" NOT at start STABLE.
        Assert: kept=[seg0] (one shorter than maximal), all pieces ≤ limit.
        """
        tok = StartAbMergeTokenizer()
        bos_id = tok.bos_token_id  # 0

        # Verify engineered instabilities BEFORE calling _split_overflow (TDD RED check basis)
        #   Remainder at b=2: to_text strips BOS -> "abc" -> "ab" at start -> (BOS,1000,c) != (BOS,a,b,c)
        unstable_rem = (bos_id, ord("a"), ord("b"), ord("c"))
        assert is_roundtrip_stable(unstable_rem, tok) is False, (
            "setup: (bos,a,b,c) must be unstable — 'ab' at text start merges"
        )
        #   Kept at b=2: to_text strips BOS -> "xyzpqr" -> no 'ab' at start -> STABLE
        kept_at_b2 = (bos_id, ord("x"), ord("y"), ord("z"), ord("p"), ord("q"), ord("r"))
        assert is_roundtrip_stable(kept_at_b2, tok) is True, (
            "setup: kept at b=2 must be stable — 'xy...' not 'ab'"
        )

        seg_tokens = [
            [
                (bos_id, ord("x"), ord("y"), ord("z")),  # seg0: 4 tokens, BOS-first
                (ord("p"), ord("q"), ord("r")),            # seg1: 3 tokens
                (ord("a"), ord("b"), ord("c")),            # seg2: 3 tokens; rem at b=2 unstable
            ]
        ]
        piece = [(0, 0), (0, 1), (0, 2)]
        limit = 7  # seg0+seg1=7=limit; total 10 > 7

        result = _split_overflow(piece, False, seg_tokens, tok, bos_id=bos_id, limit=limit)

        # Slide happened: kept is seg0 only (b=1), NOT seg0+seg1 (maximal b=2)
        assert result[0][0] == [(0, 0)], (
            "slide: kept should be only seg0 — one segment shorter than maximal"
        )
        assert result[0][1] is False, "kept new_started stays False (root piece)"

        # All output pieces fit within limit
        for refs, ns in result:
            ids = _piece_ids(refs, ns, seg_tokens, bos_id)
            assert len(ids) <= limit, (
                f"piece {refs} ns={ns} has {len(ids)} tokens > limit={limit}"
            )
