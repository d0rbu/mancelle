"""Unit tests for steering.segmentation.align_char_ranges_to_decoded_prefix.

The Layer 1 primitive that powers both align_segments_to_decoded_prefix
(Layer 2 dataclass wrapper) and the parallel-list call sites in
full-sequence SFT routing, steering/eval/target_segments.py, and
steering/eval/multiseg_generator.py.
"""

from __future__ import annotations

from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from phantom.sized import NonEmptyStr

from steering.refined_types import NonNegativeInt
from steering.segmentation import (
    CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    align_char_ranges_to_decoded_prefix,
)


def _window(value: int) -> NonNegativeInt:
    return NonNegativeInt.parse(value)


def _counter(value: str) -> NonEmptyStr:
    return NonEmptyStr.parse(value)


def test_layer1_forward_within_window_extends(patch_counter):
    starts = [0]
    ends = [3]
    new_starts, new_ends, kept = align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=5,
    )
    assert new_starts == [0]
    assert new_ends == [5]
    assert kept == [0]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    name, value, attrs = canon[0]
    assert attrs.get("direction") == "forward"
    assert attrs.get("delta") == 2


def test_layer1_forward_multi_segment_only_extends_last(patch_counter):
    """Forward extension touches ONLY the last segment's char_end.
    Earlier segments' char_start / char_end are preserved verbatim.
    This is critical for the M1 / M2 migration sites where parallel
    arrays (fids_list, mags_list) must remain aligned with the char
    ranges — any unexpected shift in earlier segment boundaries would
    desynchronise the lockstep filter.
    """
    new_starts, new_ends, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3, 6],
        char_ends=[3, 6, 9],
        decoded_len=11,  # delta = 2, within window
    )
    assert new_starts == [0, 3, 6]
    assert new_ends == [3, 6, 11]  # only last extended
    assert kept == [0, 1, 2]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == 2


# --- Layer 1: equal / empty no-op ---

def test_layer1_equal_returns_inputs_unchanged(patch_counter):
    starts, ends = [0, 5], [5, 10]
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=10,
    )
    assert new_s == [0, 5]
    assert new_e == [5, 10]
    assert kept == [0, 1]
    assert [c for c in patch_counter if "char_end_extended" in c[0]] == []


def test_layer1_empty_returns_empty_no_counter(patch_counter):
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[], char_ends=[], decoded_len=10,
    )
    assert new_s == []
    assert new_e == []
    assert kept == []
    assert [c for c in patch_counter if "char_end_extended" in c[0]] == []


# --- Layer 1: forward boundary (window + 1 must raise) ---

def test_layer1_forward_at_window_exactly_extends(patch_counter):
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0], char_ends=[10], decoded_len=10 + 20,
    )
    assert new_e == [30]
    assert kept == [0]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == 20


def test_layer1_forward_window_plus_one_raises():
    with pytest.raises(ValueError, match="[Ff]orward.*window|exceeds"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0], char_ends=[10], decoded_len=10 + 20 + 1,
        )


# --- Layer 1: reverse direction ---

def test_layer1_reverse_within_window_filter_clip(patch_counter):
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 5], char_ends=[5, 15], decoded_len=10,
    )
    assert kept == [0, 1]
    assert new_s == [0, 5]
    assert new_e == [5, 10]  # last clipped
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 5  # overall_delta = 15 - 10


def test_layer1_reverse_no_clip_path_emits_counter(patch_counter):
    """Surviving tail ends exactly at decoded_len; no clipping needed.

    starts=[0,3,6] ends=[3,6,9] decoded=6 → kept=[0,1], last_end=6, no clip.
    overall_delta = 9-6 = 3.
    """
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3, 6], char_ends=[3, 6, 9], decoded_len=6,
    )
    assert kept == [0, 1]
    assert new_e == [3, 6]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 3


def test_layer1_reverse_at_window_exactly_filter_clip(patch_counter):
    """overall_delta == window must filter/clip (no raise)."""
    decoded = 10
    coverage_end = decoded + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS  # 30
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0], char_ends=[coverage_end], decoded_len=decoded,
    )
    assert kept == [0]
    assert new_e == [decoded]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS


def test_layer1_reverse_window_plus_one_raises():
    decoded = 10
    over = decoded + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1  # 31
    with pytest.raises(ValueError, match="[Rr]everse.*window|exceeds"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0], char_ends=[over], decoded_len=decoded,
        )


def test_layer1_reverse_over_window_overrides_filter_check():
    """When overall_delta > window, the window-bound raises BEFORE the
    all-filtered branch (even if every range would be filtered)."""
    with pytest.raises(ValueError, match="exceeds"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[10, 20], char_ends=[20, 30], decoded_len=5,
        )


def test_layer1_reverse_all_segments_filtered_raises():
    """All ranges start past decoded_len → ValueError (within window)."""
    with pytest.raises(ValueError, match="every range"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[10], char_ends=[20], decoded_len=5,
            end_marker_window=_window(100),
        )


# --- Layer 1: kept_indices invariants ---

def test_layer1_kept_indices_full_keep_returns_range():
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3], char_ends=[3, 6], decoded_len=6,
    )
    assert kept == [0, 1]


def test_layer1_kept_indices_partial_filter_preserves_order():
    _, _, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0, 3, 6, 9], char_ends=[3, 6, 9, 12], decoded_len=7,
    )
    # Only segments with char_start < 7 survive: indices 0, 1, 2.
    assert kept == [0, 1, 2]


def test_layer1_length_mismatch_raises():
    with pytest.raises(ValueError, match="length mismatch"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0, 1], char_ends=[1], decoded_len=2,
        )


def test_layer1_length_mismatch_one_empty_raises():
    """One-empty + one-non-empty is a length mismatch, not an empty-input no-op."""
    with pytest.raises(ValueError, match="length mismatch"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[1, 2], char_ends=[], decoded_len=5,
        )
    with pytest.raises(ValueError, match="length mismatch"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[], char_ends=[1], decoded_len=5,
        )


# --- Layer 1: kwarg validation (end_marker_window / counter_name) ---


def test_layer1_negative_window_raises():
    bad_window: Any = -1
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter end_marker_window=.*NonNegativeInt",
    ):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0],
            char_ends=[5],
            decoded_len=5,
            end_marker_window=bad_window,
        )


def test_layer1_empty_counter_name_raises():
    bad_counter_name: Any = ""
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter counter_name=.*NonEmptyStr",
    ):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0],
            char_ends=[5],
            decoded_len=5,
            counter_name=bad_counter_name,
        )


@pytest.mark.parametrize("field", ["char_starts", "char_ends"])
def test_layer1_non_integer_character_bound_raises_from_typed_boundary(field):
    kwargs: dict[str, Any] = {
        "char_starts": [0],
        "char_ends": [5],
        "decoded_len": 5,
    }
    kwargs[field] = [0.0]
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        align_char_ranges_to_decoded_prefix(**kwargs)


def test_layer1_zero_window_strict_mode_raises_on_any_mismatch():
    """end_marker_window=0 is a valid strict mode: any non-equal coverage
    must raise, since 0 > 0 is False but 1 > 0 is True."""
    # Forward delta = 1 > 0, raises:
    with pytest.raises(ValueError, match="exceeds.*window"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0],
            char_ends=[5],
            decoded_len=6,
            end_marker_window=_window(0),
        )
    # Equal still works:
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0],
        char_ends=[5],
        decoded_len=5,
        end_marker_window=_window(0),
    )
    assert new_e == [5]


# --- Layer 1: parametrised window sizes ---

@pytest.mark.parametrize("ws", [1, 2, 5, 20, 64])
def test_layer1_forward_window_boundary_parametrised(ws):
    """delta == ws must extend; delta == ws + 1 must raise. Both directions."""
    # Forward at boundary
    _, ends, _ = align_char_ranges_to_decoded_prefix(
        char_starts=[0], char_ends=[10], decoded_len=10 + ws,
        end_marker_window=_window(ws),
    )
    assert ends == [10 + ws]
    # Forward over boundary
    with pytest.raises(ValueError):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0], char_ends=[10], decoded_len=10 + ws + 1,
            end_marker_window=_window(ws),
        )


@pytest.mark.parametrize("ws", [1, 2, 5, 20, 64])
def test_layer1_reverse_window_boundary_parametrised(ws):
    """Reverse at boundary and over boundary."""
    # Reverse at boundary
    _, ends, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[0], char_ends=[10 + ws], decoded_len=10,
        end_marker_window=_window(ws),
    )
    assert ends == [10]
    assert kept == [0]
    # Reverse over boundary
    with pytest.raises(ValueError):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0], char_ends=[10 + ws + 1], decoded_len=10,
            end_marker_window=_window(ws),
        )


# --- Layer 1: Hypothesis property ---

@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=10),
    starts_first=st.integers(min_value=0, max_value=20),
    seg_len=st.integers(min_value=1, max_value=20),
    delta=st.integers(min_value=-20, max_value=20),
)
def test_layer1_property_within_window_invariants(n, starts_first, seg_len, delta):
    """For randomly-generated contiguous ranges with |delta| within window,
    the primitive returns a valid result, never raises (within bound), and
    preserves the order of kept_indices."""
    starts = [starts_first + i * seg_len for i in range(n)]
    ends = [s + seg_len for s in starts]
    coverage_end = ends[-1]
    decoded_len = coverage_end + delta

    # If decoded_len is so low every range is filtered, the primitive
    # raises legitimately. Skip those cases here; the dedicated test
    # ``test_layer1_reverse_all_segments_filtered_raises`` covers that
    # branch. After this pre-guard, the within-window invariants hold
    # unconditionally — no ValueError should reach the caller below.
    if decoded_len <= starts[0]:
        return

    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=decoded_len,
        end_marker_window=CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    )

    # Invariants:
    assert len(new_s) == len(new_e) == len(kept)
    assert len(new_s) <= n
    assert kept == sorted(kept), "kept_indices must be in original order"
    assert len(set(kept)) == len(kept), "kept_indices must not have duplicates"
    # Coverage is now ≤ decoded_len (reverse) or == decoded_len (forward).
    if new_e:
        assert new_e[-1] <= decoded_len
    # All kept ranges are within original; starts unchanged.
    for new_i, orig_i in enumerate(kept):
        assert new_s[new_i] == starts[orig_i]


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=10),
    starts_first=st.integers(min_value=0, max_value=20),
    seg_len=st.integers(min_value=1, max_value=20),
    delta=st.integers(min_value=-100, max_value=100),
)
def test_layer1_property_over_window_raises_invariants(
    n, starts_first, seg_len, delta,
):
    """For |delta| > window, the primitive MUST raise ValueError.

    The window-bound check fires before any filtering/clipping; this is
    asserted directly via ``pytest.raises``. ``delta == ±window`` is the
    boundary that absorbs (extend or filter+clip without raising), so
    explicitly exclude it via ``assume`` to keep the strategy focused on
    the over-window branch.
    """
    window = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    assume(abs(delta) > window)

    starts = [starts_first + i * seg_len for i in range(n)]
    ends = [s + seg_len for s in starts]
    coverage_end = ends[-1]
    decoded_len = coverage_end + delta

    if decoded_len < 0:
        with pytest.raises(BeartypeCallHintParamViolation, match="decoded_len"):
            align_char_ranges_to_decoded_prefix(
                char_starts=starts, char_ends=ends, decoded_len=decoded_len,
                end_marker_window=_window(window),
            )
    else:
        with pytest.raises(ValueError, match="exceeds"):
            align_char_ranges_to_decoded_prefix(
                char_starts=starts, char_ends=ends, decoded_len=decoded_len,
                end_marker_window=_window(window),
            )


# --- Layer 1: load-bearing-invariant Hypothesis properties (T1, T6, T8) ---


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=10),
    starts_first=st.integers(min_value=0, max_value=20),
    seg_len=st.integers(min_value=1, max_value=20),
    delta=st.integers(min_value=-20, max_value=20),
)
def test_layer1_property_kept_indices_is_prefix(n, starts_first, seg_len, delta):
    """LOAD-BEARING invariant for eval_adapter alignment-then-assignment:
    the reverse-direction filter must drop a CONTIGUOUS TAIL of indices,
    never a head or middle. Captured in segmentation.py's docstring under
    'Invariants - Filter is tail-only'.

    Post-lift, the eval_adapter packs encoder slots from POST-alignment
    context_segments and reads seg_map values pointing into those slots.
    If the filter dropped a head or middle index (kept_indices = [0, 2]
    rather than [0, 1]), the surviving segments would still be re-indexed
    [0, 1] in the seg_map, but the encoder slot at index 1 would carry
    the WRONG feature payload (originally the middle segment, not the
    tail). Tail-only filtering preserves the index correspondence.
    """
    starts = [starts_first + i * seg_len for i in range(n)]
    ends = [s + seg_len for s in starts]
    coverage_end = ends[-1]
    decoded_len = coverage_end + delta

    if decoded_len <= starts[0]:
        return  # exclude all-filtered cases (helper raises)

    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=decoded_len,
        end_marker_window=CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    )

    # The load-bearing invariant: kept is always a prefix of range(n).
    assert kept == list(range(len(kept))), (
        f"Filter is NOT tail-only: kept={kept} for starts={starts}, "
        f"ends={ends}, decoded_len={decoded_len}. The eval_adapter "
        f"encoder-slot routing depends on kept being a contiguous prefix."
    )


@settings(suppress_health_check=[HealthCheck.too_slow], max_examples=200, deadline=None)
@given(
    n=st.integers(min_value=1, max_value=10),
    starts_first=st.integers(min_value=0, max_value=20),
    seg_len=st.integers(min_value=1, max_value=20),
    delta=st.integers(min_value=1, max_value=20),  # forward only
)
def test_layer1_property_forward_only_touches_last(n, starts_first, seg_len, delta):
    """Forward extension MUST only mutate new_ends[-1]; earlier
    char_start and char_end values are preserved verbatim.

    Required by full-sequence routing and target_segments.py, which
    rely on parallel-array indices staying aligned across the call.
    """
    starts = [starts_first + i * seg_len for i in range(n)]
    ends = [s + seg_len for s in starts]
    coverage_end = ends[-1]
    decoded_len = coverage_end + delta

    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=decoded_len,
        end_marker_window=CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    )

    assert new_s == starts, "starts must be unchanged on forward extension"
    assert new_e[:-1] == ends[:-1], "all char_ends except the last must be unchanged"
    assert new_e[-1] == decoded_len, "last char_end must equal decoded_len"
    assert kept == list(range(n)), "no segments should be filtered on forward path"


@settings(
    suppress_health_check=[HealthCheck.too_slow, HealthCheck.function_scoped_fixture],
    max_examples=200,
    deadline=None,
)
@given(
    n=st.integers(min_value=1, max_value=8),
    starts_first=st.integers(min_value=0, max_value=20),
    seg_len=st.integers(min_value=1, max_value=20),
    delta=st.integers(min_value=-20, max_value=20),
)
def test_layer1_property_counter_delta_matches_magnitude(
    n, starts_first, seg_len, delta, patch_counter,
):
    """The counter's ``delta`` attribute must equal the absolute
    cross-tokenizer mismatch magnitude (== |coverage_end - decoded_len|)
    on every successful call that emits a counter.

    Catches mutations that emit clip_delta instead of overall_delta in
    the reverse-clip path, or that emit something else entirely.

    Function-scoped ``patch_counter`` is acceptable here: we use
    ``pre_count = len(patch_counter)`` to slice only the events emitted
    by THIS hypothesis example, so the accumulated history across
    generated inputs is correctly partitioned. The health check is
    suppressed to make this explicit.
    """
    if delta == 0:
        return  # equal case emits no counter
    starts = [starts_first + i * seg_len for i in range(n)]
    ends = [s + seg_len for s in starts]
    coverage_end = ends[-1]
    decoded_len = coverage_end + delta

    if decoded_len <= starts[0]:
        return  # all-filtered case raises

    # Capture only counter events from this call
    pre_count = len(patch_counter)
    align_char_ranges_to_decoded_prefix(
        char_starts=starts, char_ends=ends, decoded_len=decoded_len,
        end_marker_window=CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    )
    new_events = [
        c for c in patch_counter[pre_count:]
        if c[0] == "segmentation.char_end_extended"
    ]
    assert len(new_events) == 1, (
        f"Expected exactly 1 counter event, got {len(new_events)}"
    )
    name, value, attrs = new_events[0]
    assert attrs.get("delta") == abs(delta), (
        f"Counter delta mismatch: attrs.delta={attrs.get('delta')}, "
        f"|delta|={abs(delta)}, direction={attrs.get('direction')}"
    )
    expected_direction = "forward" if delta > 0 else "reverse"
    assert attrs.get("direction") == expected_direction


def test_layer1_custom_counter_name_propagates_to_record_counter(patch_counter):
    """When a consumer passes a site-scoped counter_name, Layer 1 emits
    under that exact name, not under the canonical default."""
    align_char_ranges_to_decoded_prefix(
        char_starts=[0], char_ends=[3], decoded_len=5,
        counter_name=_counter("segmentation.char_end_extended.foo"),
    )
    foo = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.foo"]
    default = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(foo) == 1
    assert default == []
