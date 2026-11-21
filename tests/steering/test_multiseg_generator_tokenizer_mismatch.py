"""Integration tests for the tokenizer-mismatch defence in
``steering.eval.multiseg_generator``.

The Layer-1 ``align_char_ranges_to_decoded_prefix`` primitive is wired
into the per-target build site (``generate_for_document``-style call site
around lines 1100-1125). Small deltas are absorbed via the primitive's
canonical counter ``segmentation.char_end_extended``; large deltas
raise ``ValueError``.

Note on coverage: the ``build_multiseg_inputs`` entry-point at this
module's first call site (around lines 369-393) operates on a
``segments`` list whose final element is the target segment, while the
``prefix_tokens`` parameter only covers chars up to ``target_seg.char_start``.
That structural asymmetry (``coverage_end >> decoded_len`` by design)
makes the Layer-1 primitive's reverse-direction bound check inappropriate
for that site — the per-segment width naturally exceeds the
``CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS`` window. The
``_fill_context_assignments`` site (eval_adapter.py) and the per-target
build (this file) DO have ``coverage_end ≈ decoded_len`` invariants by
design and use the primitive. See plan
``docs/superpowers/plans/2026-05-11-centralise-segmentation-alignment.md``
Commit 4 / Step 4.8 for the structural distinction.
"""

from __future__ import annotations

import ast
import dataclasses
import inspect
from dataclasses import dataclass

import pytest
import torch as t

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.multiseg_generator import build_multiseg_inputs
from steering.segmentation import align_char_ranges_to_decoded_prefix


@dataclass
class _StubTokenizer:
    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)


def _make_segment(idx: int, char_start: int, char_end: int) -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=idx,
        position=idx * 8,
        feature_ids=t.tensor([1, 2], dtype=t.int32),
        feature_mags=t.tensor([1, 2], dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
    )


def test_build_multiseg_no_mismatch_succeeds(patch_counter):
    """No mismatch: build_multiseg_inputs succeeds without emitting the counter.

    This exercises the build_multiseg_inputs path under the synthetic
    ``coverage_end == decoded_len`` invariant. The reverse-direction
    defence at this site is intentionally NOT migrated (see module
    docstring) because the production caller passes segments whose
    coverage extends past the prefix-tokens decoded length by design.
    """
    tok = _StubTokenizer()
    segments = [
        _make_segment(0, 0, 3),
        _make_segment(1, 3, 6),
    ]
    prefix_tokens = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E"), ord("F")]  # 6 chars decoded
    inputs = build_multiseg_inputs(
        segments=segments,
        prefix_tokens=prefix_tokens,
        n_gen_tokens=1,
        target_segment_idx=1,
        tokenizer=tok,
    )
    assert inputs.seg_map.shape == (1, 7)
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert canon == []


# ---------------------------------------------------------------------------
# M4b alignment-block tests
#
# generate_for_document is heavy (requires a real model + lots of setup).
# Instead of building that machinery, these tests exercise the exact
# primitive-sequence the M4b block performs:
#   1. compute char_offsets (skipped here; we pass decoded_len directly)
#   2. seg_slice = segments[: target_segment_idx + 1]
#   3. align_char_ranges_to_decoded_prefix(...)
#   4. lockstep rebuild via dataclasses.replace
# This locks in the M4b code path's behavior under each mismatch direction.
# ---------------------------------------------------------------------------


def _make_seg_features_for_m4b(idx: int, char_start: int, char_end: int) -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=idx,
        position=idx * 4,
        feature_ids=t.tensor([1], dtype=t.int32),
        feature_mags=t.tensor([1.0], dtype=t.float32),
        char_start=char_start,
        char_end=char_end,
    )


def test_m4b_alignment_forward_extension_preserves_target(patch_counter):
    """The M4b alignment block in generate_for_document forward-extends
    the target segment's char_end when decoded_len exceeds it. Verifies
    the lockstep rebuild correctly produces a new SegmentFeatures
    tuple with the extended char_end on the last (target) segment.
    """
    segments = [
        _make_seg_features_for_m4b(0, 0, 3),
        _make_seg_features_for_m4b(1, 3, 5),  # target — coverage_end = 5
    ]
    target_segment_idx = 1
    decoded_len = 7  # > coverage_end, delta=2 within window
    seg_slice = segments[: target_segment_idx + 1]

    new_starts, new_ends, kept_indices = align_char_ranges_to_decoded_prefix(
        char_starts=[s.char_start for s in seg_slice],
        char_ends=[s.char_end for s in seg_slice],
        decoded_len=decoded_len,
    )
    assert kept_indices == [0, 1]
    assert new_ends == [3, 7]  # last extended

    # Apply the lockstep rebuild (mirrors generate_for_document code)
    rebuilt = [
        dataclasses.replace(
            seg_slice[orig_idx],
            char_start=new_starts[new_idx],
            char_end=new_ends[new_idx],
        )
        for new_idx, orig_idx in enumerate(kept_indices)
    ]
    # target index unchanged; target.char_end extended
    assert rebuilt[target_segment_idx].char_end == 7
    # Non-char_end fields preserved
    assert rebuilt[target_segment_idx].segment_idx == 1
    assert rebuilt[target_segment_idx].feature_ids.tolist() == [1]

    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == 2


def test_m4b_alignment_reverse_clip_within_window(patch_counter):
    """The M4b alignment block reverse-clips the target's char_end when
    decoded_len falls short of it within the canonical window."""
    segments = [
        _make_seg_features_for_m4b(0, 0, 3),
        _make_seg_features_for_m4b(1, 3, 10),  # target — coverage_end = 10
    ]
    target_segment_idx = 1
    decoded_len = 8  # < coverage_end, delta=2 within window
    seg_slice = segments[: target_segment_idx + 1]

    new_starts, new_ends, kept_indices = align_char_ranges_to_decoded_prefix(
        char_starts=[s.char_start for s in seg_slice],
        char_ends=[s.char_end for s in seg_slice],
        decoded_len=decoded_len,
    )
    assert kept_indices == [0, 1]  # both retained (char_start of seg 1 = 3 < 8)
    assert new_ends == [3, 8]  # target clipped from 10 to 8

    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 2  # overall_delta = 10 - 8


def test_m4b_alignment_over_window_raises():
    """Above the window, the helper raises — protecting M4b from
    silently truncating data with a large cross-tokenizer mismatch."""
    segments = [
        _make_seg_features_for_m4b(0, 0, 5),
        _make_seg_features_for_m4b(1, 5, 50),  # target — coverage_end = 50
    ]
    target_segment_idx = 1
    decoded_len = 5  # decoded_len far short → delta = 45 > window
    seg_slice = segments[: target_segment_idx + 1]
    with pytest.raises(ValueError, match="[Rr]everse|exceeds"):
        align_char_ranges_to_decoded_prefix(
            char_starts=[s.char_start for s in seg_slice],
            char_ends=[s.char_end for s in seg_slice],
            decoded_len=decoded_len,
        )


# ---------------------------------------------------------------------------
# M4a invariant pin
# ---------------------------------------------------------------------------


def test_m4a_build_multiseg_inputs_does_not_invoke_alignment_helper():
    """Pin: build_multiseg_inputs (M4a) intentionally does NOT call the
    alignment helper. The plan documents that M4a's production input
    shape (coverage_end ≈ target.char_end, decoded_len ≈ target.char_start,
    delta ≈ one segment width) is structurally incompatible with the
    helper's window-bound check. The overlay-precomputed boundaries
    used by callers via scripts.preprocess.precompute_token_boundaries
    already absorb cross-tokenizer mismatch upstream, so a defence at
    M4a would be a no-op AND would break production data.

    This test asserts the absence of the helper call to catch a future
    contributor who naively migrates M4a.
    """
    source = inspect.getsource(build_multiseg_inputs)
    tree = ast.parse(source)

    helper_names = {
        "align_char_ranges_to_decoded_prefix",
        "align_segments_to_decoded_prefix",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in helper_names:
                raise AssertionError(
                    f"build_multiseg_inputs at line {node.lineno} calls "
                    f"{func.id}; the M4a code path is intentionally NOT "
                    f"migrated. See PR feat-centralise-segmentation-alignment "
                    f"Commit 4 body for the rationale (production "
                    f"coverage_end >> decoded_len by one segment width)."
                )
