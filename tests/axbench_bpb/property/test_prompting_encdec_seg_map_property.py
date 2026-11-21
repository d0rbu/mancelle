"""Hypothesis property tests for ``_build_encdec_seg_map_and_encoder``.

For ANY valid combination with at least one cached prefix segment, the
helper produces a seg_map that satisfies:

P1. Shape == ``(1, len(full_prefix) + n_output)``.
P2. ``seg_map.min() >= 0``.
P3. ``seg_map.max() < n_seg`` (computed n_seg matches return shape).
P4. Non-zero values monotonically non-decreasing (cross-attention contract).
P5. ``_padded_idx_unique_per_row`` does not raise (G9 contract).
P6. When ``needs_empty_prompt_slot`` triggers, ``seg_map[0, :n_prompt_surviving] == 0``.
P7. When ``needs_empty_prompt_slot`` triggers, every value in the
    inputs-prefix slice is in {1, ..., k+1}.
P8. ``seg_map[0, -n_output:].unique() == [generation_slot]``.
"""
from __future__ import annotations

import torch
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb.teacher_force import (
    _build_encdec_seg_map_and_encoder,
    _padded_idx_unique_per_row,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer


def _check_non_zero_monotonic(seg_map: torch.Tensor) -> bool:
    sm = seg_map[0]
    nonzero_cummax = torch.where(sm > 0, sm, torch.zeros_like(sm)).cummax(dim=0).values
    return bool(((sm == 0) | (sm >= nonzero_cummax)).all().item())


def _build_segments(n_segments: int, n_inputs_chars: int):
    """Build ``n_segments`` contiguous segments over [0, n_inputs_chars)."""
    if n_segments == 0:
        return ()
    seg_width = max(1, n_inputs_chars // n_segments)
    segments = []
    for i in range(n_segments):
        start = i * seg_width
        end = min(n_inputs_chars, start + seg_width)
        if start >= end:
            continue
        segments.append(ContextFeatureSegment(
            feature_ids=(i,), buckets=(0,),
            char_start=start, char_end=end,
        ))
    return tuple(segments)


@given(
    n_prompt_surviving=st.integers(min_value=0, max_value=15),
    n_inputs=st.integers(min_value=1, max_value=30),
    n_output=st.integers(min_value=1, max_value=10),
    n_segments=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=200, deadline=None)
def test_seg_map_invariants(n_prompt_surviving, n_inputs, n_output, n_segments):
    """Every produced seg_map satisfies P1-P5."""
    # Skip combos where there are more segments than inputs chars (degenerate).
    assume(n_segments <= max(1, n_inputs))
    full_prefix = list(range(n_prompt_surviving + n_inputs))
    segments = _build_segments(n_segments, max(1, n_inputs))
    assume(bool(segments))

    tok = SimpleTokenizer()
    enc_fids, _enc_mags, _enc_mask, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
        tokenizer=tok,
        full_prefix=full_prefix,
        prefix_segments=segments,
        feature_conditions=None,
        n_output=n_output,
        n_prompt_surviving=n_prompt_surviving,
        device=torch.device("cpu"),
    )
    # P1: shape.
    assert seg_map.shape == (1, len(full_prefix) + n_output)
    # P2: min >= 0.
    assert int(seg_map.min().item()) >= 0
    # P3: max < n_seg.
    n_seg = enc_fids.shape[1]
    assert int(seg_map.max().item()) < n_seg
    # P4: non-zero monotonic.
    assert _check_non_zero_monotonic(seg_map)
    # P5: G9 unique padded_idx.
    _padded_idx_unique_per_row(seg_map, n_seg_max=n_seg)


@given(
    n_prompt_surviving=st.integers(min_value=1, max_value=15),
    n_inputs=st.integers(min_value=1, max_value=30),
    n_output=st.integers(min_value=1, max_value=10),
    n_segments=st.integers(min_value=1, max_value=8),
)
@settings(max_examples=200, deadline=None)
def test_shifted_branch_routing(n_prompt_surviving, n_inputs, n_output, n_segments):
    """When the shifted branch fires, P6/P7/P8 hold."""
    assume(n_segments <= n_inputs)
    full_prefix = list(range(n_prompt_surviving + n_inputs))
    segments = _build_segments(n_segments, n_inputs)
    if not segments:
        return  # branch did not trigger

    tok = SimpleTokenizer()
    _, _, _, seg_map, gen_slot = _build_encdec_seg_map_and_encoder(
        tokenizer=tok,
        full_prefix=full_prefix,
        prefix_segments=segments,
        feature_conditions=None,
        n_output=n_output,
        n_prompt_surviving=n_prompt_surviving,
        device=torch.device("cpu"),
    )
    # P6: prompt-prefix positions are zero.
    assert seg_map[0, :n_prompt_surviving].eq(0).all().item()
    # P7: inputs-prefix positions in {1..len(segments)+1}.
    inputs_slice = seg_map[0, n_prompt_surviving:n_prompt_surviving + n_inputs]
    if inputs_slice.numel() > 0:
        assert (inputs_slice >= 1).all().item()
        assert (inputs_slice <= len(segments) + 1).all().item()
    # P8: output positions == generation_slot, and generation_slot == k+1.
    assert gen_slot == 1 + len(segments)
    assert seg_map[0, -n_output:].eq(gen_slot).all().item()
