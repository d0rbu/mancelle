"""Pin the direct full-context slot contract for ``build_multiseg_inputs``.

Generated-token positions always map to ``target_segment_idx``. Encoder slots
are direct source-coordinate slots, with no offset mode.
"""

from __future__ import annotations

import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.multiseg_generator import build_multiseg_inputs
from tests.conftest import MockTokenizer


def _make_segments(
    n_seg: int, *, char_step: int = 4
) -> list[SegmentFeatures]:
    """Build a deterministic list of ``SegmentFeatures`` covering
    ``[0, n_seg * char_step]``. Each segment has one feature.
    """
    segments: list[SegmentFeatures] = []
    for i in range(n_seg):
        segments.append(
            SegmentFeatures(
                segment_idx=i,
                position=i,
                feature_ids=t.tensor([i + 1], dtype=t.int64),
                feature_mags=t.tensor([1], dtype=t.int64),
                char_start=i * char_step,
                char_end=(i + 1) * char_step,
            )
        )
    return segments


def _prefix_for_segments(n_seg: int, char_step: int = 4) -> list[int]:
    """A small prefix; content doesn't matter for cond_slot logic."""
    # Use an alphanumeric string of size n_seg * char_step — the
    # MockTokenizer encodes char-level so we land in segment 0..n_seg-1
    # roughly. The decoded prefix tokens align by char-overlap.
    text = "abcdefghijklmnopqrstuvwxyz0123456789"
    return list(MockTokenizer().encode(text[: n_seg * char_step], prepend=0))  # ty: ignore[invalid-return-type]


def test_cond_slot_writes_target_seg_idx():
    n_seg = 3
    target = 2
    segments = _make_segments(n_seg)
    prefix = _prefix_for_segments(n_seg)
    inputs = build_multiseg_inputs(
        segments=segments,
        prefix_tokens=prefix,
        n_gen_tokens=8,
        target_segment_idx=target,
        tokenizer=MockTokenizer(),
    )
    n_prefix = len(prefix)
    gen_part = inputs.seg_map[0, n_prefix:]
    assert (gen_part == target).all(), (
        f"target_segment_idx={target} must be written to all "
        f"generated-token slots, got {gen_part.tolist()}"
    )


@settings(deadline=None, max_examples=30, derandomize=True)
@given(
    n_seg=st.integers(min_value=1, max_value=8),
    target_offset=st.integers(min_value=0, max_value=7),
)
def test_cond_slot_property(n_seg, target_offset):
    """Property test: generated-token slots reflect direct target routing."""
    target = target_offset % n_seg
    segments = _make_segments(n_seg)
    prefix = _prefix_for_segments(n_seg)
    inputs = build_multiseg_inputs(
        segments=segments,
        prefix_tokens=prefix,
        n_gen_tokens=4,
        target_segment_idx=target,
        tokenizer=MockTokenizer(),
    )
    n_prefix = len(prefix)
    gen_part = inputs.seg_map[0, n_prefix:]
    assert (gen_part == target).all(), (
        f"For n_seg={n_seg}, target={target}, expected gen_part={target} "
        f"but got {gen_part.tolist()}"
    )


def test_cond_slot_seg_map_prefix_is_monotone_non_decreasing():
    """Prefix slots in ``seg_map`` are cummax-monotonised; this is the
    documented invariant that anchors the cond_slot contract on the prefix
    side."""
    n_seg = 4
    segments = _make_segments(n_seg)
    prefix = _prefix_for_segments(n_seg)
    inputs = build_multiseg_inputs(
        segments=segments,
        prefix_tokens=prefix,
        n_gen_tokens=2,
        target_segment_idx=2,
        tokenizer=MockTokenizer(),
    )
    prefix_part = inputs.seg_map[0, : len(prefix)]
    diffs = prefix_part[1:] - prefix_part[:-1]
    assert (diffs >= 0).all(), (
        f"Prefix seg_map must be monotone non-decreasing post-cummax, "
        f"got {prefix_part.tolist()}"
    )


def test_cond_slot_always_populates_encoder_features():
    """The multiseg builder has no empty-feature branch."""
    n_seg = 3
    target = 1
    segments = _make_segments(n_seg)
    prefix = _prefix_for_segments(n_seg)
    inputs = build_multiseg_inputs(
        segments=segments,
        prefix_tokens=prefix,
        n_gen_tokens=4,
        target_segment_idx=target,
        tokenizer=MockTokenizer(),
    )
    assert inputs.enc_mask.any().item()
    assert inputs.enc_magnitudes.sum().item() > 0
