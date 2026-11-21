"""Unit tests for :func:`steering.segmentation.align_segments_to_decoded_prefix`.

This helper centralises the cross-tokenizer end-marker handling that the
W5 audit shipped inline in ``steering/eval/axbench_bpb/seg_map.py`` (PR
#448, commit 98235860). The helper extends or filters+clips a tuple of
:class:`ContextFeatureSegment` so its char extent matches the nanochat
decoder's view of the prefix.

Four directions × edge cases:

* forward small : nanochat decode > Gemma char_end, delta <= window
* forward large : nanochat decode > Gemma char_end, delta > window
* reverse small : nanochat decode < Gemma char_end, delta <= window
* reverse large : nanochat decode < Gemma char_end, delta > window
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from phantom.sized import NonEmptyStr

from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import NonNegativeInt
from steering.segmentation import (
    CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    align_segments_to_decoded_prefix,
)


def _window(value: int) -> NonNegativeInt:
    return NonNegativeInt.parse(value)


def _counter(value: str) -> NonEmptyStr:
    return NonEmptyStr.parse(value)


@dataclass
class _StubTokenizer:
    """Each token id decodes to 1 char (id == ord(char))."""

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def get_bos_token_id(self):
        return ord("<")


# The ``patch_counter`` fixture lives in tests/steering/conftest.py so it
# can be reused by both this file and test_align_char_ranges_to_decoded_prefix.py
# without duplicating a module-level import inside the fixture body.


# ---------------------------------------------------------------------------
# Forward (nanochat decode > Gemma char_end) — small delta extends
# ---------------------------------------------------------------------------


def test_align_extends_last_segment_when_within_window(patch_counter):
    """Small forward delta extends the last segment's char_end."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]  # 5 chars
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    assert len(out) == 1
    assert out[0].char_end == 5
    assert out[0].char_start == 0
    # Other fields preserved.
    assert out[0].feature_ids == (1,)
    assert out[0].buckets == (0,)


def test_align_at_window_boundary_extends(patch_counter):
    """delta exactly equal to the window must extend, not raise."""
    tok = _StubTokenizer()
    # delta = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS (= 20) exactly.
    decoded_len = 30
    tokens = [ord("A")] * decoded_len
    char_end = decoded_len - CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS
    segs = (
        ContextFeatureSegment(
            feature_ids=(7,), buckets=(2,), char_start=0, char_end=char_end,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    assert out[0].char_end == decoded_len
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS


# ---------------------------------------------------------------------------
# Forward (nanochat decode > Gemma char_end) — large delta raises
# ---------------------------------------------------------------------------


def test_align_raises_when_forward_delta_exceeds_window():
    """delta > window must raise ValueError with informative message."""
    tok = _StubTokenizer()
    decoded_len = CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 50  # 70
    tokens = [ord("A")] * decoded_len
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=1,
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
        )
    msg = str(excinfo.value)
    # Message must surface both lengths + delta.
    assert str(decoded_len) in msg
    assert "1" in msg
    assert str(decoded_len - 1) in msg  # delta = decoded_len - char_end


# ---------------------------------------------------------------------------
# Reverse (nanochat decode < Gemma char_end) — small delta filters + clips
# ---------------------------------------------------------------------------


def test_align_filter_clip_when_reverse_within_window(patch_counter):
    """Small reverse mismatch clips last surviving segment's char_end."""
    tok = _StubTokenizer()
    decoded_len = 10
    tokens = [ord("A")] * decoded_len
    # Two segments: [0,5), [5,15). nanochat decodes 10 chars only.
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=5,
        ),
        ContextFeatureSegment(
            feature_ids=(2,), buckets=(1,), char_start=5, char_end=15,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    assert len(out) == 2  # both retained, last clipped.
    assert out[0].char_end == 5
    assert out[1].char_end == decoded_len
    assert out[1].char_start == 5


def test_align_filters_fully_uncovered_segments(patch_counter):
    """Segments fully past decoded_len are dropped; last survivor clipped."""
    tok = _StubTokenizer()
    decoded_len = 4
    tokens = [ord("A")] * decoded_len
    # Three segments: [0,3), [3,6), [6,9). nanochat decodes 4 chars only.
    # Seg 0 retained; seg 1 retained + clipped; seg 2 fully past → dropped.
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
        ContextFeatureSegment(
            feature_ids=(2,), buckets=(1,), char_start=3, char_end=6,
        ),
        ContextFeatureSegment(
            feature_ids=(3,), buckets=(2,), char_start=6, char_end=9,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    assert len(out) == 2
    assert out[0].feature_ids == (1,)
    assert out[0].char_end == 3
    assert out[1].feature_ids == (2,)
    assert out[1].char_end == decoded_len  # 4


# ---------------------------------------------------------------------------
# Reverse (nanochat decode < Gemma char_end) — large delta raises
# ---------------------------------------------------------------------------


def test_align_raises_when_reverse_delta_exceeds_window():
    """Large reverse mismatch must raise."""
    tok = _StubTokenizer()
    decoded_len = 3
    tokens = [ord("A")] * decoded_len
    # One segment covering 200 chars, nanochat only decodes 3 → delta = 197.
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=200,
        ),
    )
    with pytest.raises(ValueError) as excinfo:
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
        )
    msg = str(excinfo.value)
    # Must mention both lengths + delta.
    assert str(decoded_len) in msg
    assert "200" in msg
    assert "197" in msg


# ---------------------------------------------------------------------------
# Exact match — no change, no counter
# ---------------------------------------------------------------------------


def test_align_returns_unchanged_when_equal(patch_counter):
    """When decoded_len == segments[-1].char_end, no counter, no change."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C")]  # 3 chars
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    assert out == segs  # identical
    extensions = [c for c in patch_counter if "char_end_extended" in c[0]]
    assert extensions == []


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


def test_align_works_with_empty_prefix_tokens():
    """Empty prefix tokens → segments returned unchanged (no extension)."""
    tok = _StubTokenizer()
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=[],
        segments=segs,
    )
    assert out == segs


def test_align_works_with_empty_segments():
    """Empty segments tuple → returns empty tuple unchanged."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B")]
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=(),
    )
    assert out == ()


# ---------------------------------------------------------------------------
# Counter emission
# ---------------------------------------------------------------------------


def test_align_emits_counter_with_direction_attr_forward(patch_counter):
    """Forward extension must emit counter with direction='forward'."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    canonical = [
        c for c in patch_counter
        if c[0] == "segmentation.char_end_extended"
    ]
    assert len(canonical) == 1, (
        f"Expected exactly 1 'segmentation.char_end_extended' event, "
        f"got {patch_counter}"
    )
    name, value, attrs = canonical[0]
    assert value == 1.0
    assert attrs.get("direction") == "forward"
    assert attrs.get("delta") == 2  # 5 - 3


def test_align_emits_counter_with_direction_attr_reverse(patch_counter):
    """Reverse filter+clip must emit counter with direction='reverse'."""
    tok = _StubTokenizer()
    decoded_len = 10
    tokens = [ord("A")] * decoded_len
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=5,
        ),
        ContextFeatureSegment(
            feature_ids=(2,), buckets=(1,), char_start=5, char_end=15,
        ),
    )
    align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
    )
    canonical = [
        c for c in patch_counter
        if c[0] == "segmentation.char_end_extended"
    ]
    assert len(canonical) == 1
    name, value, attrs = canonical[0]
    assert value == 1.0
    assert attrs.get("direction") == "reverse"
    assert attrs.get("delta") == 5  # 15 - 10


def test_align_preserves_feature_ids_and_buckets(patch_counter):
    """Extension preserves non-char_end fields exactly."""
    tok = _StubTokenizer()
    tokens = [ord("X") for _ in range(7)]
    seg = ContextFeatureSegment(
        feature_ids=(11, 22, 33), buckets=(3, 5, 7), char_start=0, char_end=3,
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=(seg,),
    )
    assert out[0].feature_ids == (11, 22, 33)
    assert out[0].buckets == (3, 5, 7)
    assert out[0].char_start == 0
    assert out[0].char_end == 7


def test_align_custom_window_size_extends():
    """Caller can override the window size; small custom window still works."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5  # decoded = 5 chars
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    # delta = 2, custom_window = 2 → still within bound.
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
        end_marker_window=_window(2),
    )
    assert out[0].char_end == 5


def test_align_custom_window_size_raises_when_exceeded():
    """Custom window can shrink the allowed gap; over the bound raises."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 10  # decoded = 10 chars
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    # delta = 7, custom_window = 5 → over bound → raise.
    with pytest.raises(ValueError):
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
            end_marker_window=_window(5),
        )


# ---------------------------------------------------------------------------
# Off-by-one boundary tests (window + 1 must raise; window exactly absorbs)
# ---------------------------------------------------------------------------


def test_align_raises_at_window_plus_one_forward():
    tok = _StubTokenizer()
    char_end = 10
    decoded_len = char_end + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1
    tokens = [ord("A")] * decoded_len
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=char_end,
        ),
    )
    with pytest.raises(ValueError):
        align_segments_to_decoded_prefix(
            tokenizer=tok, prefix_tokens=tokens, segments=segs,
        )


def test_align_raises_at_window_plus_one_reverse():
    tok = _StubTokenizer()
    decoded_len = 5
    tokens = [ord("A")] * decoded_len
    coverage_end = decoded_len + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=coverage_end,
        ),
    )
    with pytest.raises(ValueError):
        align_segments_to_decoded_prefix(
            tokenizer=tok, prefix_tokens=tokens, segments=segs,
        )


def test_align_extends_at_window_exactly_reverse(patch_counter):
    """overall_delta == window must filter/clip, not raise."""
    tok = _StubTokenizer()
    decoded_len = 10
    tokens = [ord("A")] * decoded_len
    coverage_end = decoded_len + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS  # 30
    segs = (
        ContextFeatureSegment(
            feature_ids=(11,), buckets=(2,), char_start=0, char_end=coverage_end,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok, prefix_tokens=tokens, segments=segs,
    )
    assert len(out) == 1
    assert out[0].char_end == decoded_len
    assert out[0].feature_ids == (11,)
    # Counter emitted with overall_delta.
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS


# ---------------------------------------------------------------------------
# Reverse-direction counter must emit overall_delta, not clip_delta
# ---------------------------------------------------------------------------


def test_align_emits_overall_delta_counter_on_reverse_with_filtering(patch_counter):
    """Filter-then-clip path: counter delta must be overall_delta, not clip_delta.

    segs=[(0,3),(3,6),(6,9)], decoded_len=4 → kept=[0,1], last clipped from 6 to 4.
    overall_delta = 9 - 4 = 5; clip_delta = 6 - 4 = 2. Counter must emit 5.
    """
    tok = _StubTokenizer()
    decoded_len = 4
    tokens = [ord("A")] * decoded_len
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=3, char_end=6),
        ContextFeatureSegment(feature_ids=(3,), buckets=(2,), char_start=6, char_end=9),
    )
    align_segments_to_decoded_prefix(
        tokenizer=tok, prefix_tokens=tokens, segments=segs,
    )
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    name, value, attrs = canon[0]
    assert attrs.get("direction") == "reverse"
    assert attrs.get("delta") == 5  # overall_delta, not clip_delta=2


# ---------------------------------------------------------------------------
# Preservation on reverse clip
# ---------------------------------------------------------------------------


def test_align_preserves_feature_ids_and_buckets_on_reverse_clip(patch_counter):
    tok = _StubTokenizer()
    decoded_len = 8
    tokens = [ord("A")] * decoded_len
    # Two segments: [0,5),[5,12). decoded=8 → kept=both, last clipped to 8.
    segs = (
        ContextFeatureSegment(
            feature_ids=(11, 22), buckets=(3, 5), char_start=0, char_end=5,
        ),
        ContextFeatureSegment(
            feature_ids=(7, 9, 13), buckets=(1, 4, 6), char_start=5, char_end=12,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok, prefix_tokens=tokens, segments=segs,
    )
    assert len(out) == 2
    assert out[0].feature_ids == (11, 22)
    assert out[0].buckets == (3, 5)
    assert out[1].feature_ids == (7, 9, 13)
    assert out[1].buckets == (1, 4, 6)
    assert out[1].char_end == decoded_len  # clipped


# ---------------------------------------------------------------------------
# Reverse no-clip path still emits a counter
# ---------------------------------------------------------------------------


def test_align_no_clip_reverse_path_emits_counter(patch_counter):
    """Surviving tail ends exactly at decoded_len; no clip needed; counter fires."""
    tok = _StubTokenizer()
    decoded_len = 6
    tokens = [ord("A")] * decoded_len
    # segs=[(0,3),(3,6),(6,9)] decoded=6 → kept=[0,1] (seg 2 char_start=6 not < 6),
    # last_end of kept = 6 (matches decoded), no clip needed.
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=3, char_end=6),
        ContextFeatureSegment(feature_ids=(3,), buckets=(2,), char_start=6, char_end=9),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok, prefix_tokens=tokens, segments=segs,
    )
    assert len(out) == 2
    assert out[-1].char_end == decoded_len
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 9 - 6  # overall_delta = 3


# ---------------------------------------------------------------------------
# Window-size parametrised boundary tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws", [1, 2, 5, 20, 64])
def test_align_forward_at_and_over_window_parametrised(ws):
    """delta == ws extends; delta == ws + 1 raises."""
    tok = _StubTokenizer()
    base = 5
    # At boundary
    tokens = [ord("A")] * (base + ws)
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=base,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
        end_marker_window=_window(ws),
    )
    assert out[0].char_end == base + ws
    # Over boundary
    tokens = [ord("A")] * (base + ws + 1)
    with pytest.raises(ValueError):
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
            end_marker_window=_window(ws),
        )


@pytest.mark.parametrize("ws", [1, 2, 5, 20, 64])
def test_align_reverse_at_and_over_window_parametrised(ws):
    """overall_delta == ws clips; overall_delta == ws + 1 raises."""
    tok = _StubTokenizer()
    decoded_len = 5
    tokens = [ord("A")] * decoded_len
    # At boundary
    coverage_end = decoded_len + ws
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=coverage_end,
        ),
    )
    out = align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
        end_marker_window=_window(ws),
    )
    assert out[0].char_end == decoded_len
    # Over boundary
    segs_over = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=coverage_end + 1,
        ),
    )
    with pytest.raises(ValueError):
        align_segments_to_decoded_prefix(
            tokenizer=tok, prefix_tokens=tokens, segments=segs_over,
            end_marker_window=_window(ws),
        )


# ---------------------------------------------------------------------------
# Layer-2 wrapper counter_name kwarg propagation
# ---------------------------------------------------------------------------


def test_align_segments_passes_custom_counter_name_to_layer_1(patch_counter):
    """The Layer-2 wrapper must accept counter_name and forward it.

    Site-scoped counter names let dashboards attribute cross-tokenizer
    mismatches to specific consumer call sites.
    """
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5  # decoded = 5 chars
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    align_segments_to_decoded_prefix(
        tokenizer=tok,
        prefix_tokens=tokens,
        segments=segs,
        counter_name=_counter("segmentation.char_end_extended.test_consumer"),
    )
    canon = [
        c for c in patch_counter
        if c[0] == "segmentation.char_end_extended.test_consumer"
    ]
    assert len(canon) == 1, (
        f"Expected exactly 1 site-scoped event, got patches={patch_counter}"
    )
    name, value, attrs = canon[0]
    assert attrs.get("direction") == "forward"
    assert attrs.get("delta") == 2
    # No event under the default name
    default = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert default == [], (
        f"Default counter must not fire when site-scoped name is supplied: {default}"
    )


# ---------------------------------------------------------------------------
# Layer-2 refined runtime boundary
# ---------------------------------------------------------------------------


def test_align_segments_empty_counter_name_fails_at_refined_boundary():
    """Layer 2 rejects an empty counter name before alignment begins."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    bad_counter_name: Any = ""
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter counter_name=.*NonEmptyStr",
    ):
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
            counter_name=bad_counter_name,
        )


def test_align_segments_none_counter_name_fails_at_refined_boundary():
    """Layer 2 rejects ``None`` at its own runtime-typed boundary."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    bad_counter_name: Any = None
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter counter_name=.*NonEmptyStr",
    ):
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
            counter_name=bad_counter_name,
        )


def test_align_segments_negative_window_fails_at_refined_boundary():
    """Layer 2 rejects a negative window before alignment begins."""
    tok = _StubTokenizer()
    tokens = [ord("A")] * 5
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    bad_window: Any = -1
    with pytest.raises(
        BeartypeCallHintParamViolation,
        match=r"parameter end_marker_window=.*NonNegativeInt",
    ):
        align_segments_to_decoded_prefix(
            tokenizer=tok,
            prefix_tokens=tokens,
            segments=segs,
            end_marker_window=bad_window,
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
