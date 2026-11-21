"""Integration proof that expanded lockstep assignments reach decoder seg_map."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t

from nanochat.engine import KVCache
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.eval.context_cache_io import ContextCacheEntry, ContextFeatureSegment
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    MultiTargetSegments,
)
from steering.eval.target_real_dispatch import build_target_for_real_config
from tests.conftest import _make_encdec


class _OneCharTokenizer:
    def decode(self, token_ids):
        return "".join(chr(ord("a") + int(token)) for token in token_ids)


class _FullSpanProvider:
    def __init__(self, span_end: int):
        self._segments = (
            ContextFeatureSegment(
                feature_ids=(0,),
                buckets=(1,),
                char_start=0,
                char_end=span_end,
            ),
        )

    def get(self, token_ids):
        return self._segments


def _build_forced_gap_target() -> MultiTargetSegments:
    tokenizer = _OneCharTokenizer()
    full_token_ids = list(range(8))
    target_segments = [
        ContextFeatureSegment(
            feature_ids=(idx + 1,),
            buckets=(idx % 4,),
            char_start=idx + 1,
            char_end=idx + 2,
        )
        for idx in range(7)
    ]
    target = build_target_for_real_config(
        cfg=LockstepTargetRealConfig(name="tool-gap", cache_dir=Path("/tmp/x")),
        entry=ContextCacheEntry.from_sequences(full_token_ids, target_segments),
        target_char_start=1,
        target_char_end=8,
        full_token_ids=full_token_ids,
        target_token_loss_mask=[0, 1, 1, 0, 0, 0, 1, 1],
        tokenizer=tokenizer,
    )
    assert isinstance(target, MultiTargetSegments)
    return target


def test_real_tiny_encoder_decoder_consumes_every_forced_gap_position() -> None:
    """Exercise dispatch → wrapper → real model forward/KV decode.

    The target vector is the concrete spelling-bee-shaped regression: four
    routed positions, a three-position forced tool span, then two routed
    positions.  The wrapper must expose the expanded sequence to the decoder
    rather than compressing it back to routed-token order.
    """
    target = _build_forced_gap_target()
    assert target.token_assignments.tolist() == [0, 1, 1, 1, 1, 5, 6]

    model = _make_encdec(device="cpu")
    tokenizer = _OneCharTokenizer()
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=target,
        context_provider=_FullSpanProvider(span_end=4),
    )
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.dec_config.n_kv_head,
        seq_len=16,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer,
        device=t.device("cpu"),
        dtype=t.float32,
    )
    input_ids = t.tensor([[0, 1, 2, 3]], dtype=t.long)
    captured_seg_maps: list[list[int]] = []
    original_forward = model.forward

    def record_forward(*args, **kwargs):
        captured_seg_maps.append(kwargs["seg_map"].detach().cpu().tolist()[0])
        return original_forward(*args, **kwargs)

    with patch.object(model, "forward", side_effect=record_forward):
        with wrapper.conditioning_context(context_token_ids=[[0, 1, 2, 3]]):
            with t.inference_mode():
                wrapper(input_ids, kv_cache=kv_cache)
                for token in range(7):
                    wrapper(t.tensor([[token + 4]], dtype=t.long), kv_cache=kv_cache)

    assert len(captured_seg_maps) == 8
    # One context slot precedes the target slots. Prefill uses assignment[0]
    # for its final prompt position; each one-token decode then advances one
    # absolute position, including the three forced positions.
    assert captured_seg_maps[0] == [0, 0, 0, 1]
    decode_slots = [row[0] for row in captured_seg_maps[1:]]
    assert decode_slots == [1, 2, 2, 2, 2, 6, 7]
    assert decode_slots[5:] == [6, 7], "post-tool target slots must not shift"


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
def test_cuda_bfloat16_wrapper_preserves_forced_gap_routing() -> None:  # pragma: no cover
    """Run the same production wrapper/attention path on a real CUDA device."""
    target = _build_forced_gap_target()
    model = _make_encdec(device="cuda").to(dtype=t.bfloat16)
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=_OneCharTokenizer(),
        target=target,
        context_provider=_FullSpanProvider(span_end=4),
    )
    kv_cache = KVCache(
        batch_size=1,
        num_heads=model.dec_config.n_kv_head,
        seq_len=16,
        head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer,
        device=t.device("cuda"),
        dtype=t.bfloat16,
    )
    seen: list[list[int]] = []
    original_forward = model.forward

    def record_forward(*args, **kwargs):
        seen.append(kwargs["seg_map"].detach().cpu().tolist()[0])
        return original_forward(*args, **kwargs)

    with patch.object(model, "forward", side_effect=record_forward):
        with wrapper.conditioning_context(context_token_ids=[[0, 1, 2, 3]]):
            with t.inference_mode():
                wrapper(t.tensor([[0, 1, 2, 3]], device="cuda"), kv_cache=kv_cache)
                for token in range(7):
                    wrapper(
                        t.tensor([[token + 4]], device="cuda"),
                        kv_cache=kv_cache,
                    )
    assert [row[0] for row in seen[1:]] == [1, 2, 2, 2, 2, 6, 7]
