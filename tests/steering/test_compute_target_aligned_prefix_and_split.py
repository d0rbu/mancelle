"""Unit tests for ``steering.eval.multiseg_generator._compute_target_aligned_prefix_and_split``.

The helper extracts the M4b alignment block + split-index computation
from ``generate_for_document``. Isolating it in a dedicated function
lets us unit-test the full M4b control flow without standing up a real
``EncoderDecoderGPT`` model.

The 8 tests below cover:
  * no-mismatch happy path
  * forward extension within window
  * reverse clip within window
  * reverse / forward over-window raises
  * synthetic sub-window-segment filter dropping the target -> RuntimeError
  * edge case where target_segment_idx never appears in cummax -> split_idx
    falls back to len(all_tokens) (skipped - production-unreachable)
  * counter name = completion_eval (lock the site-scoped name contract).
  * ground_truth / pre_segment_text slice semantics
"""

from __future__ import annotations

import ast
import inspect
from dataclasses import dataclass

import pytest
import torch as t

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.multiseg_generator import (
    _compute_target_aligned_prefix_and_split,
    _TargetAlignedSplit,
    generate_for_document,
)


@dataclass
class _StubTokenizer:
    """Each token id decodes to one char (id == ord(char)). BOS id = 2."""

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def encode(self, text, prepend=None):
        ids = [ord(c) for c in text]
        if prepend is not None:
            return [int(prepend), *ids]
        return ids

    def get_bos_token_id(self):
        return 2


@dataclass
class _ForwardDeltaTokenizer:
    """Cross-tokenizer mismatch sim: ``decode`` emits a spurious trailing
    character whenever it sees ``end_marker`` (default ``'D'``), so
    ``decode(encode(text)) = text + 'X' * count(end_marker, text)``.

    Encode is 1-1 char->id (BOS prepended). This models the forward
    direction of M4b: consumer tokenizer decodes MORE characters than
    the source covered.
    """

    end_marker: str = "D"
    extra: str = "X"

    def decode(self, ids):
        parts = []
        for i in ids:
            c = chr(int(i))
            parts.append(c)
            if c == self.end_marker:
                parts.append(self.extra)
        return "".join(parts)

    def encode(self, text, prepend=None):
        ids = [ord(c) for c in text]
        if prepend is not None:
            return [int(prepend), *ids]
        return ids

    def get_bos_token_id(self):
        return 2


@dataclass
class _ReverseDeltaTokenizer:
    """Cross-tokenizer mismatch sim: ``decode`` drops the trailing
    character whenever the last token is ``drop_id`` (default ord('Z')),
    so ``decode(encode(text)) = text[:-1]`` when ``text`` ends with the
    drop char. Encode is 1-1 char->id (BOS prepended). This models the
    reverse direction of M4b: consumer tokenizer decodes FEWER
    characters than the source covered.
    """

    drop_id: int = ord("Z")

    def decode(self, ids):
        if not ids:
            return ""
        parts = []
        for i in ids[:-1]:
            parts.append(chr(int(i)))
        last = int(ids[-1])
        if last != self.drop_id:
            parts.append(chr(last))
        return "".join(parts)

    def encode(self, text, prepend=None):
        ids = [ord(c) for c in text]
        if prepend is not None:
            return [int(prepend), *ids]
        return ids

    def get_bos_token_id(self):
        return 2


def _make_seg(idx, char_start, char_end):
    return SegmentFeatures(
        segment_idx=idx,
        position=idx * 4,
        feature_ids=t.tensor([1], dtype=t.int32),
        feature_mags=t.tensor([1.0], dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
    )


def test_no_mismatch_split_at_target_start(patch_counter):
    """Aligned segments + decoded_len matches -> split_idx points at the
    first token of the target segment."""
    tok = _StubTokenizer()
    segments = [
        _make_seg(0, char_start=1, char_end=4),
        _make_seg(1, char_start=4, char_end=7),
    ]
    doc_text = "ABCDEF"  # 6 chars; bos_text = "\x02ABCDEF" = 7 chars total.

    out = _compute_target_aligned_prefix_and_split(
        tokenizer=tok,
        segments=segments,
        target_segment_idx=1,
        doc_text=doc_text,
        bos_id=2,
    )

    assert isinstance(out, _TargetAlignedSplit)
    assert len(out.seg_slice) == 2
    assert out.seg_slice[-1].char_end == 7
    # ground_truth slices bos_text[4:7] = "DEF" (BOS occupies position 0).
    assert out.ground_truth == "DEF"
    # pre_segment_text = bos_text[1:4] = "ABC"
    assert out.pre_segment_text == "ABC"
    # all_tokens = [2, A, B, C, D, E, F]
    assert list(out.all_tokens) == [2, ord("A"), ord("B"), ord("C"), ord("D"), ord("E"), ord("F")]
    # Split: first token whose cummax-assignment is the target (idx 1).
    # strict_segment_bounds_for_assignment folds the BOS span into seg 0,
    # so seg 0 covers [0, 4) (tokens BOS, A, B, C) and seg 1 covers [4, 7)
    # (tokens D, E, F). split_idx = 4 (where D starts).
    assert out.split_idx == 4
    # No counter - no mismatch.
    canon = [c for c in patch_counter if c[0].startswith("segmentation.char_end_extended")]
    assert canon == []


@dataclass
class _BigExpandTokenizer:
    """Each non-BOS token decodes to ``multiplier`` repeated chars. BOS stays 1 char.

    Used to drive forward-direction mismatches well past the 20-char
    window without needing huge documents.
    """

    multiplier: int = 25

    def decode(self, ids):
        parts = []
        for raw_id in ids:
            i = int(raw_id)
            if i == 2:
                parts.append(chr(2))
            else:
                parts.append(chr(i) * self.multiplier)
        return "".join(parts)

    def encode(self, text, prepend=None):
        ids = [ord(c) for c in text]
        if prepend is not None:
            return [int(prepend), *ids]
        return ids

    def get_bos_token_id(self):
        return 2


def test_forward_within_window_extends_target(patch_counter):
    """char_end short by <= 20 chars -> seg_slice[-1].char_end extended,
    counter fires with direction=forward."""
    tok = _ForwardDeltaTokenizer(end_marker="D", extra="X")
    # bos_text = chr(2) + doc_text = "\x02ABCDE" (6 chars)
    # seg 1 covers [4, 5) = "D" (1 char in source view).
    # decode(all_tokens) for [2, A, B, C, D] = chr(2) + "A" + "B" + "C" + "DX" = 6 chars.
    # coverage_end=5, decoded_len=6 -> forward delta=1 -> within window.
    segments = [
        _make_seg(0, 1, 4),
        _make_seg(1, 4, 5),
    ]
    doc_text = "ABCDE"

    out = _compute_target_aligned_prefix_and_split(
        tokenizer=tok,
        segments=segments,
        target_segment_idx=1,
        doc_text=doc_text,
        bos_id=2,
    )

    assert out.seg_slice[-1].char_end == 6
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.completion_eval"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == 1


def test_reverse_within_window_clips_target(patch_counter):
    """char_end over by <= 20 chars -> seg_slice[-1].char_end clipped,
    counter fires with direction=reverse."""
    tok = _ReverseDeltaTokenizer(drop_id=ord("Z"))
    # bos_text = "\x02ABCDZ" (6 chars), seg 1 covers [4, 6) in source view.
    # encode("ABCDZ", prepend=2) = [2, A, B, C, D, Z] (6 tokens).
    # decode drops the trailing Z -> chr(2) + "ABCD" = 5 chars.
    # coverage_end=6, decoded_len=5 -> reverse, overall_delta=1.
    segments = [
        _make_seg(0, 1, 4),
        _make_seg(1, 4, 6),
    ]
    doc_text = "ABCDZ"

    out = _compute_target_aligned_prefix_and_split(
        tokenizer=tok,
        segments=segments,
        target_segment_idx=1,
        doc_text=doc_text,
        bos_id=2,
    )

    assert out.seg_slice[-1].char_end == 5
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.completion_eval"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 1


def test_forward_over_window_raises():
    """Forward delta > 20 -> ValueError from the alignment primitive."""
    tok = _BigExpandTokenizer(multiplier=25)
    # decode([2, X, X, X, X]) = chr(2) + "X"*25*4 = 101 chars.
    # coverage_end=5, decoded_len=101 -> forward delta = 96 (>> 20).
    segments = [_make_seg(0, 1, 5)]
    doc_text = "X" * 50

    with pytest.raises(ValueError, match="[Ff]orward.*window|exceeds"):
        _compute_target_aligned_prefix_and_split(
            tokenizer=tok,
            segments=segments,
            target_segment_idx=0,
            doc_text=doc_text,
            bos_id=2,
        )


def test_reverse_over_window_raises():
    """Reverse delta > 20 -> ValueError from the alignment primitive."""
    tok = _StubTokenizer()
    # coverage_end=100, decoded_len=4 (chr(2)+"ABC") -> reverse delta=96 (>> 20).
    segments = [_make_seg(0, 1, 100)]
    doc_text = "ABC"

    with pytest.raises(ValueError, match="[Rr]everse.*window|exceeds"):
        _compute_target_aligned_prefix_and_split(
            tokenizer=tok,
            segments=segments,
            target_segment_idx=0,
            doc_text=doc_text,
            bos_id=2,
        )


def test_filter_drops_target_raises_runtimeerror():
    """Synthetic sub-window segments where the reverse-direction filter
    drops the target index -> RuntimeError from the helper's defensive
    check.

    seg 2 starts at char 8 but decoded_len=6, so kept_indices=[0,1] and
    the target (idx 2) survives the bound check but is dropped by the
    starts-past-decoded filter.
    """
    tok = _StubTokenizer()
    segments = [
        _make_seg(0, 1, 4),
        _make_seg(1, 4, 8),
        _make_seg(2, 8, 12),
    ]
    doc_text = "ABCDE"  # bos_text = "\x02ABCDE" (6 chars decoded)

    with pytest.raises(RuntimeError, match="filtered the target segment"):
        _compute_target_aligned_prefix_and_split(
            tokenizer=tok,
            segments=segments,
            target_segment_idx=2,
            doc_text=doc_text,
            bos_id=2,
        )


def test_no_target_position_in_assignments_split_idx_fallback():
    """Edge case: split_idx=len(all_tokens) fallback path is production-
    unreachable when the target segment survives alignment with positive
    width. Document via skip."""
    pytest.skip(
        "split_idx=len(all_tokens) fallback path is production-unreachable "
        "when target survives alignment with positive width; documented in "
        "helper docstring."
    )


def test_counter_name_is_completion_eval(patch_counter):
    """Lock the site-scoped counter name contract: the helper must emit
    only ``segmentation.char_end_extended.completion_eval`` (never the
    default ``segmentation.char_end_extended``)."""
    tok = _ForwardDeltaTokenizer(end_marker="D", extra="X")
    segments = [
        _make_seg(0, 1, 4),
        _make_seg(1, 4, 5),
    ]
    doc_text = "ABCDE"

    _compute_target_aligned_prefix_and_split(
        tokenizer=tok,
        segments=segments,
        target_segment_idx=1,
        doc_text=doc_text,
        bos_id=2,
    )

    completion = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.completion_eval"]
    default = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(completion) == 1
    assert default == []


def test_ground_truth_pre_segment_text_slices(patch_counter):
    """Verify ``ground_truth`` + ``pre_segment_text`` + ``text_through_target``
    are sliced from ``bos_text`` using the ORIGINAL (pre-alignment) target
    char range."""
    tok = _StubTokenizer()
    segments = [
        _make_seg(0, 1, 4),
        _make_seg(1, 4, 7),
    ]
    doc_text = "ABCDEF"  # bos_text = "\x02ABCDEF" (7 chars)

    out = _compute_target_aligned_prefix_and_split(
        tokenizer=tok,
        segments=segments,
        target_segment_idx=1,
        doc_text=doc_text,
        bos_id=2,
    )

    # bos_text[4:7] = "DEF" (positions 4=D, 5=E, 6=F).
    assert out.ground_truth == "DEF"
    # bos_text[1:4] = "ABC".
    assert out.pre_segment_text == "ABC"
    # text_through_target = bos_text[1:7] = "ABCDEF".
    assert out.text_through_target == "ABCDEF"


def test_empty_segments_raises_valueerror():
    tok = _StubTokenizer()
    with pytest.raises(ValueError, match="non-empty segments|at least one segment"):
        _compute_target_aligned_prefix_and_split(
            tokenizer=tok, segments=[], target_segment_idx=0,
            doc_text="ABC", bos_id=2,
        )


def test_target_segment_idx_out_of_range_raises_valueerror():
    tok = _StubTokenizer()
    segments = [_make_seg(0, 1, 4)]
    with pytest.raises(ValueError, match="out of range"):
        _compute_target_aligned_prefix_and_split(
            tokenizer=tok, segments=segments, target_segment_idx=5,
            doc_text="ABC", bos_id=2,
        )


def test_generate_for_document_invokes_helper_via_ast_inspection():
    """Pin: generate_for_document MUST call _compute_target_aligned_prefix_and_split.

    A future refactor that inlines the alignment logic back into
    generate_for_document (or replaces the helper with something else)
    would silently revert the testability win. This AST-based test
    catches such a refactor at CI time.
    """
    source = inspect.getsource(generate_for_document)
    tree = ast.parse(source)

    helper_name = "_compute_target_aligned_prefix_and_split"
    found = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id == helper_name:
                found = True
                break
            if isinstance(func, ast.Attribute) and func.attr == helper_name:
                found = True
                break

    assert found, (
        f"generate_for_document must call {helper_name}; not found in its body. "
        f"This indicates either a refactor that removed the helper, or a "
        f"rename. Update this test if the rename is intentional."
    )
