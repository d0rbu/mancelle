"""W5 unit tests: build_multi_segment_seg_map must extend the last
prefix segment's char_end when the nanochat tokenizer's decode of the
prefix produces longer offsets than the Gemma extraction tokenizer used
to compute the segment boundaries.

The extension is implemented via
:func:`steering.segmentation.align_segments_to_decoded_prefix`, which
emits the canonical ``segmentation.char_end_extended`` counter (replacing
the historical ``bpb.seg_map.char_end_extended`` name).
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
from steering.eval.target_conditioning import ContextFeatureSegment

# Canonical counter name emitted by
# ``steering.segmentation.align_segments_to_decoded_prefix``.
_COUNTER_NAME = "segmentation.char_end_extended.bpb_seg_map"


@dataclass
class _StubTokenizer:
    """Each token id decodes to 1 char (id == ord(char))."""

    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def get_bos_token_id(self):
        return ord("<")


@pytest.fixture(autouse=True)
def patch_counter(monkeypatch):
    """Capture ``record_counter`` calls so tests assert on them.

    The counter is now emitted from :mod:`steering.segmentation`; the
    legacy :mod:`steering.eval.axbench_bpb.seg_map` no longer calls
    ``record_counter`` directly. We patch both call sites for backward
    compatibility (the seg_map module simply has no remaining call).
    """
    captured: list[tuple[str, float, dict]] = []

    def fake(name, value=1, **attrs):
        captured.append((name, float(value), attrs))

    import steering.segmentation as seg
    monkeypatch.setattr(seg, "record_counter", fake)
    return captured


def test_no_extension_when_decoded_len_matches(patch_counter):
    """Gemma char_end equals nanochat decoded length → no extension."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C")]
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=tokens,
        prefix_segments=segs,
        n_output=1,
        generation_slot=1,
    )
    extensions = [c for c in patch_counter if c[0] == _COUNTER_NAME]
    assert extensions == []


def test_extends_last_segment_when_decoded_len_exceeds_char_end(patch_counter):
    """nanochat decodes longer than Gemma → last segment extended +
    counter emitted."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]
    # Gemma segments say char_end=3 but nanochat decodes 5 chars.
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=tokens,
        prefix_segments=segs,
        n_output=1,
        generation_slot=1,
    )
    extensions = [c for c in patch_counter if c[0] == _COUNTER_NAME]
    assert len(extensions) == 1, (
        f"Expected 1 extension counter event, got {len(extensions)}: {patch_counter}"
    )


def test_decoded_len_less_than_char_end_filters_and_clips(patch_counter):
    """If nanochat decodes SHORTER than Gemma char_end, the helper clips the
    last surviving segment's ``char_end`` to the decoded length and emits a
    reverse-direction counter event."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B")]  # 2 chars
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=5,
        ),
    )
    build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=tokens,
        prefix_segments=segs,
        n_output=1,
        generation_slot=1,
    )
    extensions = [c for c in patch_counter if c[0] == _COUNTER_NAME]
    # delta = 3 chars (5 - 2) <= 20 char window, so filter+clip applies.
    assert len(extensions) == 1
    name, value, attrs = extensions[0]
    assert attrs.get("direction") == "reverse"
    assert attrs.get("delta") == 3


def test_no_op_when_no_prefix_segments(patch_counter):
    """Empty prefix segments tuple → no extension attempted."""
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B")]
    build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=tokens,
        prefix_segments=(),
        n_output=1,
        generation_slot=0,
    )
    extensions = [c for c in patch_counter if c[0] == _COUNTER_NAME]
    assert extensions == []


def test_no_op_when_prefix_tokens_empty(patch_counter):
    tok = _StubTokenizer()
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=3,
        ),
    )
    build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=[],
        prefix_segments=segs,
        n_output=1,
        generation_slot=1,
    )
    extensions = [c for c in patch_counter if c[0] == _COUNTER_NAME]
    assert extensions == []


def test_extension_preserves_other_segment_fields(patch_counter):
    """Extended segment must preserve all non-char_end fields."""
    tok = _StubTokenizer()
    tokens = [ord("X") for _ in range(7)]
    seg = ContextFeatureSegment(
        feature_ids=(11, 22), buckets=(3, 5), char_start=0, char_end=3,
    )
    # The extension trigger fires because 7 > 3.
    build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=tokens,
        prefix_segments=(seg,),
        n_output=1,
        generation_slot=1,
    )
    extensions = [c for c in patch_counter if c[0] == _COUNTER_NAME]
    assert len(extensions) == 1


def test_multibyte_prefix_round_trip_through_compute_context_token_segment_assignments(patch_counter):
    """End-to-end smoke: a tokenizer that decodes ids to characters in a
    range that produces longer-than-Gemma offsets must NOT crash inside
    compute_context_token_segment_assignments.

    Pre-fix this would raise ValueError because trailing nanochat tokens
    fall past the last segment's char_end. Post-fix the seg_map is built
    successfully.
    """
    tok = _StubTokenizer()
    # 5 tokens decoded as single chars (length 5), but Gemma's char_end
    # claims only 2 chars. The extension fix bridges the gap.
    tokens = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]
    segs = (
        ContextFeatureSegment(
            feature_ids=(1,), buckets=(0,), char_start=0, char_end=2,
        ),
    )
    seg_map = build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=tokens,
        prefix_segments=segs,
        n_output=1,
        generation_slot=1,
    )
    # 5 prefix tokens + 1 output token = shape (1, 6).
    assert seg_map.shape == (1, 6)
    # All 5 prefix tokens land on the (extended) single segment id 0.
    assert (seg_map[0, :5] == 0).all().item()
    # Output position lands on generation_slot=1.
    assert seg_map[0, 5].item() == 1
