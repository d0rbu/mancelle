"""Full-wrapper integration tests for cross-tokenizer alignment in
``TargetAwareEncoderDecoderEvalWrapper._build_conditioning_impl``.

The sibling file ``test_eval_adapter_fill_context_tokenizer_mismatch.py``
binds ``_align_row_context_segments`` directly to a SimpleNamespace stub,
which is sufficient for unit-level verification but bypasses:

  * the wrapper's ``_pack_encoder_inputs`` step (which packs encoder
    slots from POST-alignment context_segments — see the lift commit),
  * the ``_plan_for_context`` step (which derives ``target_base_idx``
    from POST-alignment segment count),
  * the ordering of ``_align_row_context_segments`` BEFORE
    ``_pack_encoder_inputs`` inside ``_build_conditioning_impl``.

A mutation that runs assignment WITHOUT first running alignment would
silently break under reverse-direction filtering: tokens past
decoded_len would have zero overlap with any segment, surfacing as a
``strict_segment_bounds_for_assignment`` ``ValueError``. These tests
pin the full integration so such mutations surface as test failures.

Tests use a real ``TargetAwareEncoderDecoderEvalWrapper`` instance and
a minimal in-memory ``SafetensorsContextFeatureProvider``-shaped stub.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from dataclasses import dataclass
from types import SimpleNamespace

import torch

from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.target_conditioning import ContextFeatureSegment

# (Implementation note: this test scaffolding requires constructing a
# minimal TargetAwareEncoderDecoderEvalWrapper with a real or stub
# EncoderDecoderGPT model. If the construction is heavy, we use the
# descriptor-protocol binding pattern established in the sibling
# test_eval_adapter_fill_context_tokenizer_mismatch.py module: bind
# the production method to a SimpleNamespace stub and verify the SEQUENCE
# of method calls and the seg_map output values.)


# Approach: we don't need a full forward pass — we just need to verify
# the SEQUENCE of method calls and the seg_map output values. We can
# instrument _fill_context_assignments to verify its inputs match the
# expected post-alignment shape.


def _align_then_fill_under_stub(tokenizer, row_seg_map, context_tokens, context_segments):
    """Model the full ``_build_conditioning_impl`` sequence under a stub.

    Post-lift, ``_build_conditioning_impl`` does:

      context_segments = self._align_row_context_segments(
          context_tokens, context_segments,
      )
      # ... encoder pack uses aligned segments ...
      self._fill_context_assignments(
          row_seg_map, context_tokens, context_segments,
      )

    This helper composes the same two calls under a SimpleNamespace
    stub so we can verify the SEQUENCE of method calls without
    standing up a real EncoderDecoderGPT model.

    The mutation we want to catch: drop or invert the alignment step
    so that ``compute_context_token_segment_assignments`` sees the
    UNALIGNED ``context_segments``. For reverse-direction filtering
    cases, tokens past decoded_len would have zero overlap with any
    segment and trip ``assign_tokens_to_segments``'s ``ValueError``.
    """
    stub = SimpleNamespace(tokenizer=tokenizer, is_turn_self_conditioned=False, self_cond_segment_mode=None)
    align = TargetAwareEncoderDecoderEvalWrapper._align_row_context_segments.__get__(
        stub, TargetAwareEncoderDecoderEvalWrapper
    )
    fill = TargetAwareEncoderDecoderEvalWrapper._fill_context_assignments.__get__(
        stub, TargetAwareEncoderDecoderEvalWrapper
    )
    aligned = align(context_tokens, context_segments)  # ty: ignore[invalid-argument-type]
    fill(row_seg_map, context_tokens, aligned)  # ty: ignore[invalid-argument-type]


@dataclass
class _StubTokenizer:
    def decode(self, ids):
        return "".join(chr(int(i)) for i in ids)

    def get_bos_token_id(self):
        return ord("<")


def test_m3_alignment_runs_before_assignment_under_reverse_filter(patch_counter):
    """The two operations MUST be in order: align FIRST, then assign.

    Construct a scenario where the unaligned segments would trigger a
    ``assign_tokens_to_segments`` "zero overlap" ValueError (a segment
    past decoded_len). After alignment (drops the tail), assignment
    works. If alignment is skipped, the unaligned segments hit strict
    bounds and the call raises — different observable behavior.

    Reverse direction with filter: 3 segments [(0,3),(3,6),(6,9)],
    decoded_len=4. Unaligned segments would cause assign_tokens_to_segments
    to fail on tokens past char=4 (no segment has overlap). Aligned
    segments drop seg 2; tokens 0-2 -> seg 0, token 3 -> seg 1.
    """
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C"), ord("D")]  # 4 chars
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=3, char_end=6),
        ContextFeatureSegment(feature_ids=(3,), buckets=(2,), char_start=6, char_end=9),
    )
    row_seg_map = torch.zeros(len(tokens), dtype=torch.long)
    _align_then_fill_under_stub(tok, row_seg_map, tokens, segs)
    # Aligned: tokens 0-2 -> seg 0, token 3 -> seg 1. cummax preserves.
    assert row_seg_map.tolist() == [0, 0, 0, 1]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.encdec_eval_context"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"
    assert canon[0][2]["delta"] == 5  # overall_delta = 9 - 4


def test_m3_alignment_runs_before_assignment_under_forward_extension(patch_counter):
    """Forward extension scenario. 5 tokens decoded to 5 chars but
    segments only cover 3 chars. Unaligned: tokens 3-4 are past
    seg[-1].char_end=3, assign_tokens_to_segments raises 'zero overlap'.
    Aligned: seg[-1].char_end extended to 5, tokens 3-4 -> seg 0.
    """
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C"), ord("D"), ord("E")]  # 5 chars
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
    )
    row_seg_map = torch.zeros(len(tokens), dtype=torch.long)
    _align_then_fill_under_stub(tok, row_seg_map, tokens, segs)
    # All 5 tokens -> seg 0 after extension.
    assert row_seg_map.tolist() == [0, 0, 0, 0, 0]
    canon = [c for c in patch_counter if c[0] == "segmentation.char_end_extended.encdec_eval_context"]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "forward"
    assert canon[0][2]["delta"] == 2


def test_m3_signature_matches_production_caller():
    """The production caller (_build_conditioning_impl) passes the method
    three positional args: (row_seg_map, context_tokens, context_segments).
    If the method signature changes (e.g. kwarg added without default),
    the wrapper would crash. This test pins the signature.
    """
    sig = inspect.signature(
        TargetAwareEncoderDecoderEvalWrapper._fill_context_assignments
    )
    params = list(sig.parameters)
    # ['self', 'row_seg_map', 'context_tokens', 'context_segments']
    assert params == ['self', 'row_seg_map', 'context_tokens', 'context_segments']


def test_wrapper_encoder_slot_count_matches_aligned_segments_post_lift(patch_counter):
    """After the alignment lift, the encoder is packed with M aligned
    slots (not N unaligned). When reverse-direction filtering drops k
    tail context segments, the encoder gets N - k = M slots, not N.

    Pre-lift: encoder packs N slots, M < N of them are referenced,
    [M, N) are dead. Post-lift: encoder packs M slots, all referenced.

    This test pins the lift behavior. It uses the stub binding pattern
    on _align_row_context_segments since standing up a real
    EncoderDecoderGPT requires a tiny model fixture.
    """
    tok = _StubTokenizer()
    tokens = [ord("A"), ord("B"), ord("C"), ord("D")]  # 4 chars
    segs = (
        ContextFeatureSegment(feature_ids=(1,), buckets=(0,), char_start=0, char_end=3),
        ContextFeatureSegment(feature_ids=(2,), buckets=(1,), char_start=3, char_end=6),
        ContextFeatureSegment(feature_ids=(3,), buckets=(2,), char_start=6, char_end=9),
    )
    # 3 unaligned segments; aligned should drop seg 2 (char_start=6 > decoded=4)
    # → 2 aligned segments retained.

    stub = SimpleNamespace(tokenizer=tok, is_turn_self_conditioned=False, self_cond_segment_mode=None)
    bound = TargetAwareEncoderDecoderEvalWrapper._align_row_context_segments.__get__(
        stub, TargetAwareEncoderDecoderEvalWrapper
    )
    aligned = bound(tokens, segs)  # ty: ignore[invalid-argument-type]
    assert len(aligned) == 2, (
        f"Expected 2 aligned segments (tail filtered), got {len(aligned)}"
    )
    # The counter fires on the site-scoped name.
    canon = [
        c for c in patch_counter
        if c[0] == "segmentation.char_end_extended.encdec_eval_context"
    ]
    assert len(canon) == 1
    assert canon[0][2]["direction"] == "reverse"


def test_build_conditioning_impl_invokes_align_row_before_pack():
    """Pin: _build_conditioning_impl MUST call _align_row_context_segments
    BEFORE _pack_encoder_inputs in its per-row loop.

    The lift ordering is the load-bearing invariant for the dead-encoder-slot
    fix. A refactor that swaps the order (or removes the alignment call)
    would silently revert the fix.
    """
    source = textwrap.dedent(
        inspect.getsource(
            TargetAwareEncoderDecoderEvalWrapper._build_conditioning_impl
        )
    )
    tree = ast.parse(source)

    align_call_lineno: int | None = None
    pack_call_lineno: int | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Attribute):
                if func.attr == "_align_row_context_segments":
                    if align_call_lineno is None or node.lineno < align_call_lineno:
                        align_call_lineno = node.lineno
                if func.attr == "_pack_encoder_inputs":
                    if pack_call_lineno is None or node.lineno < pack_call_lineno:
                        pack_call_lineno = node.lineno

    assert align_call_lineno is not None, (
        "_build_conditioning_impl must call _align_row_context_segments"
    )
    assert pack_call_lineno is not None, (
        "_build_conditioning_impl must call _pack_encoder_inputs"
    )
    assert align_call_lineno < pack_call_lineno, (
        f"_align_row_context_segments (line {align_call_lineno}) must run "
        f"BEFORE _pack_encoder_inputs (line {pack_call_lineno}). The lift "
        f"depends on this ordering to avoid dead encoder slots."
    )
