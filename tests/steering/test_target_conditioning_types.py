"""Unit tests for the target-conditioning tagged union types.

These types replace the wrapper's old ``last_segment_features: dict`` field with
a type-level mutual exclusion between single-slot (broadcast) and multi-slot
(per-token routing) target conditioning.
"""

from __future__ import annotations

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

from steering.conditional.encoder_decoder.eval_adapter import SegmentedEncoderState
from steering.eval.target_conditioning import (
    MultiTargetSegments,
    SelfConditionedTarget,
    SingleTargetSegment,
    TargetConditioning,
    TargetConditioningMode,
)
from tests.padded_encoder_reference import pack_dense_encoder_output_reference


def test_single_target_segment_empty_features_is_context_only_delta():
    seg = SingleTargetSegment(features={})
    assert seg.features == {}


def test_single_target_segment_nonempty_features_round_trip():
    seg = SingleTargetSegment(features={42: 3, 108: 7})
    assert seg.features == {42: 3, 108: 7}


def test_multi_target_segments_requires_at_least_one_segment():
    with pytest.raises(BeartypeCallHintParamViolation, match="segments"):
        MultiTargetSegments(
            segments=[],
            token_assignments=t.tensor([0], dtype=t.int64),
        )


def test_multi_target_segments_requires_int64_dtype():
    with pytest.raises(ValueError, match="1-D int64"):
        MultiTargetSegments(
            segments=[{0: 1}],
            token_assignments=t.tensor([0.0]),
        )


def test_multi_target_segments_requires_one_dim():
    with pytest.raises(ValueError, match="1-D int64"):
        MultiTargetSegments(
            segments=[{0: 1}],
            token_assignments=t.tensor([[0]], dtype=t.int64),
        )


def test_multi_target_segments_requires_non_empty_assignments():
    with pytest.raises(ValueError, match="non-empty"):
        MultiTargetSegments(
            segments=[{0: 1}],
            token_assignments=t.tensor([], dtype=t.int64),
        )


def test_multi_target_segments_rejects_index_too_high():
    with pytest.raises(ValueError, match=r"outside \[0, 2\)"):
        MultiTargetSegments(
            segments=[{0: 1}, {1: 2}],
            token_assignments=t.tensor([0, 1, 2], dtype=t.int64),
        )


def test_multi_target_segments_rejects_negative_index():
    with pytest.raises(ValueError, match="outside"):
        MultiTargetSegments(
            segments=[{0: 1}],
            token_assignments=t.tensor([0, -1], dtype=t.int64),
        )


def test_multi_target_segments_rejects_non_monotonic_assignments():
    """SFT routing contract requires monotonic non-decreasing assignments
    (enforced via cummax in full-sequence routing). A non-monotonic
    vector would imply a token rejoining an earlier segment; reject loudly.
    """
    with pytest.raises(ValueError, match="monotonic non-decreasing"):
        MultiTargetSegments(
            segments=[{0: 1}, {0: 2}, {0: 3}],
            token_assignments=t.tensor([0, 1, 2, 1], dtype=t.int64),
        )


def test_multi_target_segments_accepts_monotonic_with_repeats():
    """Repeats and plateaus are fine: [0,0,1,1,2,2] is non-decreasing."""
    seg = MultiTargetSegments(
        segments=[{0: 1}, {0: 2}, {0: 3}],
        token_assignments=t.tensor([0, 0, 1, 1, 2, 2], dtype=t.int64),
    )
    assert seg.token_assignments.tolist() == [0, 0, 1, 1, 2, 2]


def test_multi_target_segments_rejects_negative_feature_id():
    with pytest.raises(BeartypeCallHintParamViolation, match="segments"):
        MultiTargetSegments(
            segments=[{-5: 1}],
            token_assignments=t.tensor([0], dtype=t.int64),
        )


def test_multi_target_segments_rejects_negative_bucket():
    with pytest.raises(BeartypeCallHintParamViolation, match="segments"):
        MultiTargetSegments(
            segments=[{42: -1}],
            token_assignments=t.tensor([0], dtype=t.int64),
        )


def test_multi_target_segments_valid_construction_round_trips():
    seg = MultiTargetSegments(
        segments=[{0: 1}, {1: 2}, {2: 3}],
        token_assignments=t.tensor([0, 0, 1, 2, 2], dtype=t.int64),
    )
    assert len(seg.segments) == 3
    assert seg.token_assignments.tolist() == [0, 0, 1, 2, 2]


def test_target_conditioning_union_accepts_all_branches():
    s: TargetConditioning = SingleTargetSegment(features={})
    m: TargetConditioning = MultiTargetSegments(
        segments=[{0: 1}],
        token_assignments=t.tensor([0], dtype=t.int64),
    )
    sc: TargetConditioning = SelfConditionedTarget()
    assert isinstance(s, SingleTargetSegment)
    assert isinstance(m, MultiTargetSegments)
    assert isinstance(sc, SelfConditionedTarget)
    assert not isinstance(sc, MultiTargetSegments)


def test_target_conditioning_mode_parses_current_and_next_aliases():
    assert TargetConditioningMode.parse("current") is TargetConditioningMode.CURRENT
    assert TargetConditioningMode.parse("current_segment") is TargetConditioningMode.CURRENT
    assert TargetConditioningMode.parse("next") is TargetConditioningMode.NEXT
    assert TargetConditioningMode.parse("next_segment") is TargetConditioningMode.NEXT
    assert (
        TargetConditioningMode.from_next_segment_offset(False)
        is TargetConditioningMode.CURRENT
    )
    assert (
        TargetConditioningMode.from_next_segment_offset(True)
        is TargetConditioningMode.NEXT
    )


def test_target_conditioning_mode_rejects_removed_unconditioned_modes():
    with pytest.raises(ValueError, match="target_conditioning_mode"):
        TargetConditioningMode.parse("empty")
    with pytest.raises(ValueError, match="target_conditioning_mode"):
        TargetConditioningMode.parse(None)


def test_single_target_segment_is_frozen():
    seg = SingleTargetSegment(features={1: 2})
    with pytest.raises((AttributeError, TypeError, Exception)):
        seg.features = {3: 4}  # ty: ignore[invalid-assignment]


def test_multi_target_segments_is_frozen():
    seg = MultiTargetSegments(
        segments=[{0: 1}],
        token_assignments=t.tensor([0], dtype=t.int64),
    )
    with pytest.raises((AttributeError, TypeError, Exception)):
        seg.segments = []  # ty: ignore[invalid-assignment]


def test_segmented_encoder_state_carries_target_token_assignments():
    """Extended state used at decode time: ``target_token_assignments``
    + ``target_base_idx`` + ``prefill_end_pos``.
    """
    state = SegmentedEncoderState(
        encoder_out=pack_dense_encoder_output_reference(
            t.zeros(1, 2, 4, 8),
            t.ones(1, 2, 4, dtype=t.bool),
        ),
        target_base_idx=1,
        target_token_assignments=t.tensor([0, 0, 1], dtype=t.int64),
        prefill_end_pos=12,
    )
    assert state.target_base_idx == 1
    assert state.target_token_assignments.tolist() == [0, 0, 1]
    assert state.prefill_end_pos == 12
