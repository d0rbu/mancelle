"""Producer-level regressions for byte-fallback token-boundary content (issue #1269)."""

from __future__ import annotations

import itertools

import pytest
import torch as t

from nanochat.tokenizer import get_tokenizer
from scripts.preprocess.precompute_token_boundaries import (
    _TINY_TRAILING_SEG_MAX_CHARS,
    compute_doc_token_boundaries_from_tokens,
)
from steering.segmentation import assign_tokens_to_segments
from tests.steering.test_token_offsets_tiktoken_bytes import (
    _reference_token_to_char_offsets,
)

# Frozen pre-#1269 reference implementation (copied from
# scripts/preprocess/precompute_token_boundaries.py @ 78c841725).
# Do NOT "fix" or update; the differential is the point.
_REFERENCE_TINY_TRAILING_SEG_MAX_CHARS = 20


def _reference_boundaries_from_offsets(
    text_tokens: list[int],
    seg_char_starts: list[int],
    seg_char_ends: list[int],
    tokenizer,
    source_bos_char_len: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Compute token-boundary ranges from decoded character offsets."""
    n_tokens = len(text_tokens)
    n_segs = len(seg_char_starts)
    if n_tokens == 0 or n_segs == 0:
        return t.zeros(n_segs, dtype=t.int32), t.zeros(n_segs, dtype=t.int32)

    char_offsets = _reference_token_to_char_offsets(tokenizer, text_tokens)[1]
    nanochat_bos_len = char_offsets[1] if len(char_offsets) > 1 else 0
    delta = source_bos_char_len - nanochat_bos_len
    if delta != 0:
        char_offsets = [char_offsets[0]] + [o + delta for o in char_offsets[1:]]

    tok_starts_full = t.tensor(char_offsets[:-1], dtype=t.int64)
    tok_ends_full = t.tensor(char_offsets[1:], dtype=t.int64)

    zero_width = tok_starts_full == tok_ends_full
    if zero_width.any():
        expanded_ends = tok_ends_full.clone()
        for i in range(len(expanded_ends) - 2, -1, -1):
            if expanded_ends[i] == tok_starts_full[i]:
                expanded_ends[i] = expanded_ends[i + 1]
        tok_ends_full = expanded_ends

    seg_range_end = seg_char_ends[-1]
    in_range = tok_starts_full < seg_range_end
    n_in_range = int(in_range.sum().item())
    tok_starts = tok_starts_full[:n_in_range]
    tok_ends = tok_ends_full[:n_in_range]

    seg_starts_t = t.tensor(seg_char_starts, dtype=t.int64)
    seg_ends_t = t.tensor(seg_char_ends, dtype=t.int64)
    assignments = assign_tokens_to_segments(tok_starts, tok_ends, seg_starts_t, seg_ends_t)
    assignments = t.cummax(assignments, dim=0).values
    assignments = _reference_fix_trailing_empty_last_segment(
        assignments, seg_char_starts, seg_char_ends
    )
    return _reference_extract_ranges_from_monotonic(assignments, n_segs, n_in_range)


def _reference_fix_trailing_empty_last_segment(
    assignments: t.Tensor,
    seg_char_starts: list[int],
    seg_char_ends: list[int],
) -> t.Tensor:
    """Repair the degenerate tiny-trailing-segment case."""
    n_segs = len(seg_char_starts)
    if len(assignments) == 0 or n_segs < 2:
        return assignments
    max_assigned = int(assignments[-1].item())
    if max_assigned >= n_segs - 1:
        return assignments
    last_seg_chars = seg_char_ends[-1] - seg_char_starts[-1]
    if last_seg_chars <= _REFERENCE_TINY_TRAILING_SEG_MAX_CHARS:
        assignments[-1] = n_segs - 1
    return assignments


def _reference_extract_ranges_from_monotonic(
    assignments: t.Tensor,
    n_segs: int,
    n_tokens: int,
) -> tuple[t.Tensor, t.Tensor]:
    """Extract per-segment [start, end) ranges from monotonic assignments."""
    if n_tokens == 0:
        raise ValueError(f"Cannot extract ranges from 0 tokens for {n_segs} segments")
    changes = (assignments[1:] != assignments[:-1]).nonzero(as_tuple=True)[0] + 1
    boundaries = t.cat([t.tensor([0]), changes, t.tensor([n_tokens])])
    run_starts = boundaries[:-1].to(t.int32)
    run_ends = boundaries[1:].to(t.int32)
    n_runs = len(run_starts)
    if n_runs != n_segs:
        run_seg_ids = assignments[boundaries[:-1]].tolist()
        missing = [s for s in range(n_segs) if s not in run_seg_ids]
        raise ValueError(
            f"Expected {n_segs} contiguous runs but found {n_runs}. "
            f"Segments with no tokens: {missing}. "
            f"Every extraction segment must cover at least one token."
        )
    return run_starts, run_ends


@pytest.fixture(scope="module")
def tok():
    try:
        return get_tokenizer()
    except (FileNotFoundError, ValueError) as exc:
        pytest.skip(f"nanochat tokenizer cache unavailable: {exc}")


def _segment_chars_from_token_stride(
    tokenizer, token_ids: list[int], stride: int
) -> tuple[list[int], list[int]]:
    _, offsets = _reference_token_to_char_offsets(tokenizer, token_ids)
    token_boundaries = list(range(0, len(token_ids), stride)) + [len(token_ids)]
    char_boundaries = [offsets[i] for i in token_boundaries]
    starts: list[int] = []
    ends: list[int] = []
    for start, end in itertools.pairwise(char_boundaries):
        if end > start:
            starts.append(start)
            ends.append(end)
    assert starts and ends and ends[-1] == offsets[-1]
    return starts, ends


def _assert_case(
    tokenizer,
    text_tokens: list[int],
    seg_char_starts: list[int],
    seg_char_ends: list[int],
) -> tuple[t.Tensor, t.Tensor]:
    bos_chars = len(tokenizer.decode([tokenizer.get_bos_token_id()]))
    actual = compute_doc_token_boundaries_from_tokens(
        text_tokens,
        seg_char_starts,
        seg_char_ends,
        tokenizer,
        source_bos_char_len=bos_chars,
    )
    expected = _reference_boundaries_from_offsets(
        text_tokens,
        seg_char_starts,
        seg_char_ends,
        tokenizer,
        source_bos_char_len=bos_chars,
    )
    assert all(t.equal(got, want) for got, want in zip(actual, expected, strict=True))
    starts, ends = actual
    assert bool((starts >= 0).all())
    assert bool((ends >= starts).all())
    assert bool((starts[1:] >= starts[:-1]).all())
    assert bool((ends[1:] >= ends[:-1]).all())
    assert bool((ends > starts).all())
    return actual


class TestTokenBoundaryContentRegressions:
    def test_byte_fallback_old_italic_doc(self, tok) -> None:
        text = "x " + "𐌇𐌀𐌓𐌉𐌙𐌀𐌔𐌕𐌉𐌕𐌄𐌉𐌅𐌀" * 4 + " y"
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        starts, ends = _segment_chars_from_token_stride(tok, ids, stride=32)
        _assert_case(tok, ids, starts, ends)

    def test_cjk_heavy_doc(self, tok) -> None:
        text = "안녕하세요 세계。日本語のテキストです。" * 40
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        path, _ = _reference_token_to_char_offsets(tok, ids)
        assert len(ids) > 16 and path == "slow"
        starts, ends = _segment_chars_from_token_stride(tok, ids, stride=64)
        _assert_case(tok, ids, starts, ends)

    def test_long_document_multi_segment(self, tok) -> None:
        text = "word " * 2500
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        assert len(ids) >= 2000
        starts, ends = _segment_chars_from_token_stride(tok, ids, stride=256)
        assert len(starts) >= 8
        _assert_case(tok, ids, starts, ends)

    def test_tiny_trailing_segment_repair(self, tok) -> None:
        ids = tok.encode("a" * 100, prepend=tok.get_bos_token_id())
        _, offsets = _reference_token_to_char_offsets(tok, ids)
        starts = [0, offsets[-1] - 1]
        ends = [offsets[-1] - 1, offsets[-1]]
        assert ends[-1] - starts[-1] <= _TINY_TRAILING_SEG_MAX_CHARS
        actual = _assert_case(tok, ids, starts, ends)
        assert actual[1][-1].item() == len(ids)

    def test_zero_width_token_expansion(self, tok) -> None:
        text = "𐌀" * 30
        ids = tok.encode(text, prepend=tok.get_bos_token_id())
        _, offsets = _reference_token_to_char_offsets(tok, ids)
        assert any(start == end for start, end in itertools.pairwise(offsets))
        starts, ends = _segment_chars_from_token_stride(tok, ids, stride=16)
        actual = _assert_case(tok, ids, starts, ends)
        assert bool((actual[1] > actual[0]).all())
