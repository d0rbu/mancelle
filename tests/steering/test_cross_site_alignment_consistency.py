"""Cross-site consistency: the M1 (finetune SFT) and M3 (encdec eval
adapter) paths must produce the SAME segment layout (char_start,
char_end, feature_ids, buckets) on the same input data.

Why this test exists
--------------------

The five consumer sites of ``align_*_to_decoded_prefix`` all share the
same canonical 20-char window and counter, but they differ in their
adapter shape (parallel lists vs ContextFeatureSegment tuple). If a
future PR mistakenly changes the end_marker_window default on one path
(e.g. via the ``counter_name`` knob on Layer 1), or swaps the order
of operations at one call site, the M1 and M3 paths could silently
diverge on identical input data.

The test pins:
  * forward extension produces the same surviving char ranges on both
    sides for the same input segment tuple,
  * reverse filter+clip produces the same surviving char ranges + the
    same kept_indices order.
"""

from __future__ import annotations

import pytest

from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import (
    align_char_ranges_to_decoded_prefix,
    align_segments_to_decoded_prefix,
)
from tests.cross_tokenizer_helpers import SimpleTokenizer


def test_m1_layer1_and_m3_layer2_produce_consistent_layout_forward():
    """Same segments + decoded_len; Layer 1 (M1-shape) and Layer 2
    (M3-shape) produce structurally identical results."""
    tok = SimpleTokenizer()
    # Decoded_len = 5 chars, segments cover 3 chars (forward extension).
    prefix_tokens = [ord("A")] * 5
    segs_dataclass = (
        ContextFeatureSegment(feature_ids=(11,), buckets=(2,), char_start=0, char_end=3),
    )
    # Layer 2 (M3 shape)
    out_layer2 = align_segments_to_decoded_prefix(
        tokenizer=tok, prefix_tokens=prefix_tokens, segments=segs_dataclass,
    )
    # Layer 1 (M1 shape)
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[s.char_start for s in segs_dataclass],
        char_ends=[s.char_end for s in segs_dataclass],
        decoded_len=5,
    )
    # Same char ranges, same survivors:
    assert new_s == [s.char_start for s in out_layer2]
    assert new_e == [s.char_end for s in out_layer2]
    assert len(out_layer2) == len(kept)


def test_m1_layer1_and_m3_layer2_produce_consistent_layout_reverse_clip():
    """Reverse-clip case: both layers drop the same indices and clip
    the same final segment."""
    tok = SimpleTokenizer()
    prefix_tokens = [ord("A")] * 4
    segs_dataclass = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=3, char_end=6),
        ContextFeatureSegment(feature_ids=(3,), buckets=(2,), char_start=6, char_end=9),
    )
    out_layer2 = align_segments_to_decoded_prefix(
        tokenizer=tok, prefix_tokens=prefix_tokens, segments=segs_dataclass,
    )
    new_s, new_e, kept = align_char_ranges_to_decoded_prefix(
        char_starts=[s.char_start for s in segs_dataclass],
        char_ends=[s.char_end for s in segs_dataclass],
        decoded_len=4,
    )
    # Both drop seg 2:
    assert len(out_layer2) == 2
    assert kept == [0, 1]
    # Same surviving char ranges:
    assert new_s == [s.char_start for s in out_layer2]
    assert new_e == [s.char_end for s in out_layer2]
    # Surviving feature_ids on Layer 2 match expected (seg 0, seg 1):
    assert out_layer2[0].feature_ids == (1,)
    assert out_layer2[1].feature_ids == (2,)


def test_m1_layer1_and_m3_layer2_both_raise_on_over_window_forward():
    """Both paths raise ValueError when delta exceeds the window."""
    tok = SimpleTokenizer()
    prefix_tokens = [ord("A")] * 50  # delta = 47 > 20
    segs_dataclass = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
    )
    with pytest.raises(ValueError, match="[Ff]orward.*window|exceeds"):
        align_segments_to_decoded_prefix(
            tokenizer=tok, prefix_tokens=prefix_tokens, segments=segs_dataclass,
        )
    with pytest.raises(ValueError, match="[Ff]orward.*window|exceeds"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[0], char_ends=[3], decoded_len=50,
        )
