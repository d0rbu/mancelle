"""Unit tests for the multi-segment seg_map builder. Reference:
TargetAwareEncoderDecoderEvalWrapper._fill_context_assignments and
multiseg_generator.build_multiseg_inputs."""
from __future__ import annotations

import torch

from steering.eval.axbench_bpb.seg_map import build_multi_segment_seg_map
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.cross_tokenizer_helpers import SimpleTokenizer


def test_seg_map_single_prefix_segment_single_output_segment():
    tok = SimpleTokenizer()
    # Prefix "abcde" (5 tokens), one segment covering chars [0, 5)
    prefix_tokens = list(tok.encode("abcde"))
    prefix_segments = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(3,), char_start=0, char_end=5),
    )
    # Output "xyz" (3 tokens)
    n_output = 3
    seg_map = build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=prefix_tokens,
        prefix_segments=prefix_segments,
        n_output=n_output,
        generation_slot=1,  # one prompt seg + one target seg
    )
    # All prefix positions -> 0; all output positions -> 1
    expected = torch.tensor([[0, 0, 0, 0, 0, 1, 1, 1]], dtype=torch.long)
    assert torch.equal(seg_map, expected)


def test_seg_map_monotonic_cummax():
    tok = SimpleTokenizer()
    prefix_tokens = list(tok.encode("abcdef"))
    # Two segments: [0,3) and [3,6)
    prefix_segments = (
        ContextFeatureSegment(feature_ids=(10,), buckets=(3,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(20,), buckets=(4,), char_start=3, char_end=6),
    )
    seg_map = build_multi_segment_seg_map(
        tokenizer=tok,
        prefix_tokens=prefix_tokens,
        prefix_segments=prefix_segments,
        n_output=2,
        generation_slot=2,
    )
    # Monotonic non-decreasing
    for i in range(seg_map.shape[1] - 1):
        assert seg_map[0, i].item() <= seg_map[0, i + 1].item()
    # Output positions at end
    assert seg_map[0, -2].item() == 2
    assert seg_map[0, -1].item() == 2
