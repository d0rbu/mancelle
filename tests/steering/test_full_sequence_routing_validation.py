"""Validation / edge-case coverage for full-sequence segment routing.

Targets ``steering.conditional.encoder_decoder.full_sequence_routing`` — the
module that turns a full rendered decoder token sequence plus parallel feature
segment char-ranges into a ``seg_map`` and per-segment feature lists.

These tests pin the loud-failure contract (inverted / zero-length segments,
first-segment-not-at-0, empty feature rows, degenerate token id lists) and the
``cummax`` monotonicity guarantee on the produced ``seg_map``.  They use real
char-level tokenizers (no mocks of the routing internals) so the
extractor↔router coordinate contract is exercised for real.
"""

from __future__ import annotations

from typing import cast

import pytest
import torch as t

from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    compute_full_coverage_gap,
    source_aligned_char_offsets,
)
from steering.segmentation import assign_tokens_to_segments
from tests.conftest import MockTokenizer

# ---------------------------------------------------------------------------
# Test tokenizers
# ---------------------------------------------------------------------------


def _bos_and_chars(tokenizer: MockTokenizer, text: str) -> list[int]:
    """Return BOS followed by one token id per character of ``text``.

    ``MockTokenizer`` decodes each content token to exactly one character and
    BOS to the empty string, so ``token_to_char_offsets`` yields a clean
    ``[0, 0, 1, 2, ...]`` ladder: every content token spans exactly one source
    character.  That makes char-range geometry trivial to reason about.
    """
    char_ids = cast(list[int], tokenizer.encode(text))
    return [tokenizer.get_bos_token_id(), *char_ids]


class _MultiCharTokenizer:
    """Char-level tokenizer whose tokens decode to *multiple* characters.

    Needed for the straddling-token monotonicity test: a token whose character
    span crosses a segment boundary can only exist when a single token decodes
    to more than one character (``MockTokenizer`` is strictly 1-char-per-token
    and so can never straddle an integer boundary).
    """

    _bos_id = 0
    vocab_size = 128
    # id -> decoded text. BOS decodes to "" (zero-width, like nanochat BOS).
    _decode_map = {0: "", 10: "AB", 11: "CDE", 12: "F", 13: "GH"}

    def get_bos_token_id(self) -> int:
        return self._bos_id

    def decode(self, ids: list[int]) -> str:
        return "".join(self._decode_map.get(i, "?") for i in ids)


# ---------------------------------------------------------------------------
# Segment-geometry validation raises (build_full_sequence_segment_routing)
# ---------------------------------------------------------------------------


class TestSegmentGeometryValidation:
    """Malformed segment char-ranges must raise before any token assignment."""

    def test_zero_length_segment_raises(self) -> None:
        """A segment with ``char_end == char_start`` has non-positive length."""
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abcdef")
        seg_features = ([t.tensor([1])], [t.tensor([1])], [0], [0])  # end == start
        with pytest.raises(
            ValueError, match=r"Segment 0 has non-positive length: start=0, end=0"
        ):
            build_full_sequence_segment_routing(
                ids=ids, tokenizer=tok, seg_features=seg_features
            )

    def test_inverted_segment_raises(self) -> None:
        """A segment with ``char_end < char_start`` is rejected as non-positive."""
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abcdef")
        seg_features = ([t.tensor([1])], [t.tensor([1])], [4], [2])  # end < start
        with pytest.raises(
            ValueError, match=r"Segment 0 has non-positive length: start=4, end=2"
        ):
            build_full_sequence_segment_routing(
                ids=ids, tokenizer=tok, seg_features=seg_features
            )

    def test_first_segment_not_at_zero_raises(self) -> None:
        """The first segment must begin at source char 0."""
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abcdef")
        seg_features = ([t.tensor([1])], [t.tensor([1])], [2], [5])  # starts at 2
        with pytest.raises(ValueError, match=r"First segment must start at 0, got 2"):
            build_full_sequence_segment_routing(
                ids=ids, tokenizer=tok, seg_features=seg_features
            )

    def test_empty_segment_list_raises(self) -> None:
        """Zero segments is rejected — at least one segment is required."""
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abcdef")
        seg_features = ([], [], [], [])
        with pytest.raises(
            ValueError, match=r"No segments provided \(need at least one\)"
        ):
            build_full_sequence_segment_routing(
                ids=ids, tokenizer=tok, seg_features=seg_features
            )

    def test_token_zero_is_always_in_range_so_no_overlap_guard_is_unreachable(
        self,
    ) -> None:
        """Document that the "no conversation tokens overlap the feature
        ranges" guard (full_sequence_routing.py ~line 155) is unreachable from
        this public entry point.

        ``source_aligned_char_offsets`` always preserves ``offset[0] == 0`` (it
        only shifts offsets *after* the first token), so the first decoder token
        always starts at source char 0.  The validation already enforces
        ``char_starts[0] == 0`` with a positive-length first segment, so
        ``seg_range_end = char_ends[-1] > 0`` and ``in_range`` is True for token
        0 — ``n_in_range`` can never be 0.  Any attempt to push the segment
        range to end at char 0 instead raises earlier inside
        ``align_char_ranges_to_decoded_prefix``.

        We therefore assert the CLOSEST reachable behaviour: a "tokens fall
        before the trailing segment" geometry still routes successfully (the
        leading token anchors the in-range slice) rather than tripping the
        line-155 guard.  See the reported coverage note.
        """
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "ab")  # source chars: t0 BOS[0,0), t1 a[0,1), t2 b[1,2)
        offsets = source_aligned_char_offsets(tok, ids)
        assert offsets[0] == 0  # the load-bearing fact

        # Two tiling segments; every real token lives in the FIRST segment.
        seg_features = (
            [t.tensor([3]), t.tensor([4])],
            [t.tensor([1]), t.tensor([2])],
            [0, 2],
            [2, 4],  # second segment [2,4) is past all token char spans
        )
        # The trailing segment gets dropped by decoded-prefix alignment (only
        # 2 source chars), and routing succeeds — it does NOT raise the
        # line-155 "no tokens overlap" error.
        routing = build_full_sequence_segment_routing(
            ids=ids, tokenizer=tok, seg_features=seg_features
        )
        assert bool((routing.seg_map > 0).any()), routing.seg_map.tolist()


# ---------------------------------------------------------------------------
# compute_full_coverage_gap edge case
# ---------------------------------------------------------------------------


class TestComputeFullCoverageGap:
    """``compute_full_coverage_gap`` guards against an empty feature row."""

    def test_empty_feature_list_raises(self) -> None:
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abcdef")
        with pytest.raises(
            ValueError,
            match=r"Cannot compute full-sequence coverage for an empty feature row",
        ):
            compute_full_coverage_gap(ids, [], tok)

    def test_gap_is_total_chars_minus_coverage_end(self) -> None:
        """Happy-path control: gap = total source chars - max segment char_end.

        Without this control a regression that made the empty-row path also
        cover the non-empty path could pass the raise test by accident.
        """

        class _Seg:
            def __init__(self, char_end: int) -> None:
                self.char_end = char_end

        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abcdef")  # 6 source chars
        total = source_aligned_char_offsets(tok, ids)[-1]
        assert total == 6
        gap = compute_full_coverage_gap(ids, [_Seg(4), _Seg(2)], tok)
        assert gap == total - 4 == 2


# ---------------------------------------------------------------------------
# source_aligned_char_offsets degenerate token id lists
# ---------------------------------------------------------------------------


class TestSourceAlignedCharOffsets:
    """``source_aligned_char_offsets`` short-circuits for <= 1 token ids."""

    def test_empty_ids_short_circuits_to_single_zero(self) -> None:
        """An empty id list yields ``token_to_char_offsets == [0]`` (len 1),
        which trips the ``len(offsets) <= 1`` short-circuit and is returned
        verbatim — no BOS-len shift is attempted."""
        tok = MockTokenizer()
        assert source_aligned_char_offsets(tok, []) == [0]
        # The short-circuit fires regardless of the requested bos char len:
        # there is no offset[1] to read.
        assert source_aligned_char_offsets(tok, [], source_bos_char_len=99) == [0]

    def test_single_id_is_not_short_circuited_and_is_shifted(self) -> None:
        """A *single* token id produces ``len(offsets) == 2`` (``[0, bos_len]``),
        which is past the ``<= 1`` short-circuit, so the BOS-length realignment
        IS applied: offset[1] becomes ``source_bos_char_len``.

        This pins the off-by-one boundary of the guard at line ~44 — only the
        empty-id case (one offset) skips the shift, not the one-id case.
        """
        tok = MockTokenizer()
        bos = tok.get_bos_token_id()  # decodes to "" -> nanochat_bos_len == 0
        # delta == 0 path (source_bos_char_len == nanochat_bos_len == 0).
        assert source_aligned_char_offsets(tok, [bos]) == [0, 0]
        # delta != 0 path: offset[1] is shifted to source_bos_char_len.
        assert source_aligned_char_offsets(tok, [bos], source_bos_char_len=99) == [
            0,
            99,
        ]

    def test_negative_source_bos_char_len_raises(self) -> None:
        """A negative ``source_bos_char_len`` is a programming error."""
        tok = MockTokenizer()
        ids = _bos_and_chars(tok, "abc")
        with pytest.raises(
            ValueError, match=r"feature_source_bos_char_len must be non-negative"
        ):
            source_aligned_char_offsets(tok, ids, source_bos_char_len=-1)


# ---------------------------------------------------------------------------
# Monotonicity / cummax guarantee
# ---------------------------------------------------------------------------


class TestSegMapMonotonicity:
    """The produced ``seg_map`` is monotonically non-decreasing over the
    in-range decoder tokens.

    ``build_full_sequence_segment_routing`` applies ``t.cummax`` to the raw
    token→segment assignment (full_sequence_routing.py ~line 171) precisely so
    the per-position segment index can never decrease as the decoder advances.
    """

    def test_straddling_token_keeps_seg_map_monotonic(self) -> None:
        """A token whose char span straddles a segment boundary must not
        produce a backwards (non-monotonic) ``seg_map``.

        ``_MultiCharTokenizer`` ids ``[BOS, AB, CDE, F, GH]`` decode to source
        char offsets ``[0, 0, 2, 5, 6, 8]``.  With a segment split at char 3,
        the token ``CDE`` spans ``[2, 5)`` and straddles the boundary; the
        midpoint rule assigns it to segment 1, and every later token stays at
        segment 1, so the result is monotonic.
        """
        tok = _MultiCharTokenizer()
        ids = [0, 10, 11, 12, 13]
        offsets = source_aligned_char_offsets(tok, ids)
        assert offsets == [0, 0, 2, 5, 6, 8], offsets  # CDE straddles char 3

        seg_features = (
            [t.tensor([7]), t.tensor([8])],
            [t.tensor([1]), t.tensor([2])],
            [0, 3],  # seg0 [0,3), seg1 [3,8)
            [3, 8],
        )
        routing = build_full_sequence_segment_routing(
            ids=ids, tokenizer=tok, seg_features=seg_features
        )
        seg_map = routing.seg_map
        assert seg_map.shape == (len(ids),)
        # Both segments are actually used (the straddling token did not collapse
        # routing onto a single segment).
        assert set(seg_map.tolist()) == {1, 2}, seg_map.tolist()
        # The load-bearing invariant the cummax enforces:
        assert bool((seg_map[1:] >= seg_map[:-1]).all()), (
            f"seg_map must be monotonic non-decreasing, got {seg_map.tolist()}"
        )

    def test_cummax_repairs_a_non_monotonic_assignment(self) -> None:
        """The exact ``cummax`` operation used in the router repairs a raw,
        genuinely non-monotonic token→segment assignment.

        ``assign_tokens_to_segments`` on *non-tiling* (overlapping/gapped)
        segments takes the O(n×m) argmax path, which can emit a non-monotonic
        raw vector (a later token's max-overlap segment preceding an earlier
        one).  This reproduces such a vector and asserts that
        ``t.cummax(..., dim=0).values`` — the same call the router applies at
        full_sequence_routing.py ~line 171 — yields a monotonic result without
        losing the high-water segment.
        """
        # Tokens (start, end) and NON-tiling segments seg0=[0,5), seg1=[6,15):
        #   t0=[0,5): overlap seg0=5, seg1=0  -> 0
        #   t1=[5,7): overlap seg0=0, seg1=1  -> 1
        #   t2=[1,3): overlap seg0=2, seg1=0  -> 0  (BACKWARDS vs t1)
        tok_starts = t.tensor([0, 5, 1])
        tok_ends = t.tensor([5, 7, 3])
        seg_starts = t.tensor([0, 6])
        seg_ends = t.tensor([5, 15])
        raw = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts, seg_ends)
        assert raw.tolist() == [0, 1, 0], raw.tolist()
        assert not bool((raw[1:] >= raw[:-1]).all()), "fixture must be non-monotonic"

        repaired = t.cummax(raw, dim=0).values
        assert repaired.tolist() == [0, 1, 1], repaired.tolist()
        assert bool((repaired[1:] >= repaired[:-1]).all())
