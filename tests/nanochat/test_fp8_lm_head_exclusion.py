"""Tests that lm_head / unembedding is excluded from FP8 conversion.

Audit: phase2/03-numerical-correctness.md §2.2 — Phase 1 report 01 §2 flagged
that `lm_head` meets `fp8_module_filter` criteria (both dims divisible by 16,
min dim >= 128) and is silently converted to FP8, despite the standard LLM
training recipe keeping lm_head in higher precision (bf16/fp32) because the
output is softcapped and cross-entropy is sensitive to tail logits.

These tests assert that after `apply_fp8_training` on a decoder-only GPT and
on an EncoderDecoderGPT, the module named `lm_head` is NOT a Float8Linear.
"""

from __future__ import annotations

import pytest
import torch

from nanochat.fp8_utils import apply_fp8_training
from nanochat.gpt import GPT, GPTConfig

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA required for FP8 conversion"
)


def _is_fp8(module: torch.nn.Module) -> bool:
    return "Float8" in type(module).__name__


class TestLmHeadFP8Exclusion:
    """lm_head must not be converted to FP8 after apply_fp8_training."""

    def test_decoder_gpt_lm_head_not_fp8(self):
        """Decoder-only GPT: lm_head stays in bf16/fp32 after FP8 conversion."""
        config = GPTConfig(
            sequence_len=128,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=256,
            window_pattern="L",
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cuda")
        model.init_weights()

        apply_fp8_training(model, recipe="tensorwise")

        assert not _is_fp8(model.lm_head), (
            f"lm_head was converted to FP8 (type={type(model.lm_head).__name__}); "
            f"it must be excluded per standard LLM training recipe."
        )
        # Sanity: some other Linear should still be FP8, else the test is vacuous.
        num_fp8 = sum(1 for m in model.modules() if _is_fp8(m))
        assert num_fp8 > 0, (
            "No FP8 conversion happened at all — test is vacuous. Model config "
            "may need larger dims."
        )

    def test_encdec_gpt_lm_head_not_fp8(self):
        """EncoderDecoderGPT: lm_head stays in bf16/fp32 after FP8 conversion."""
        from nanochat.gpt import GPTConfig
        from steering.conditional.encoder_decoder.config import (
            CrossAttentionConfig,
            EncoderConfig,
        )
        from steering.conditional.encoder_decoder.model import EncoderDecoderGPT

        dec_config = GPTConfig(
            sequence_len=128,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=256,
            window_pattern="L",
        )
        enc_config = EncoderConfig(n_embd=128, depth=2, head_dim=64)
        xattn_config = CrossAttentionConfig(frequency=2)

        with torch.device("meta"):
            model = EncoderDecoderGPT(
                dec_config,
                enc_config,
                xattn_config,
                n_sae_features=16384,
                num_buckets=10,
            )
        model.to_empty(device="cuda")
        model.init_weights()

        apply_fp8_training(model, recipe="tensorwise")

        assert not _is_fp8(model.lm_head), (
            f"lm_head was converted to FP8 "
            f"(type={type(model.lm_head).__name__}) in EncoderDecoderGPT; "
            f"it must be excluded per standard LLM training recipe."
        )

        # No module named exactly 'lm_head' should be FP8.
        fp8_lm_head_count = sum(
            1 for n, m in model.named_modules() if n == "lm_head" and _is_fp8(m)
        )
        assert fp8_lm_head_count == 0

    def test_opt_in_fp8_lm_head_possible(self):
        """A caller who explicitly wants lm_head in FP8 can request it via kwarg."""
        config = GPTConfig(
            sequence_len=128,
            vocab_size=256,
            n_layer=2,
            n_head=4,
            n_kv_head=4,
            n_embd=256,
            window_pattern="L",
        )
        with torch.device("meta"):
            model = GPT(config)
        model.to_empty(device="cuda")
        model.init_weights()

        apply_fp8_training(model, recipe="tensorwise", exclude_lm_head=False)

        # When the user opts in, lm_head IS FP8.
        assert _is_fp8(model.lm_head), (
            "exclude_lm_head=False should allow lm_head to be converted to FP8"
        )
