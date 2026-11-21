"""EncoderDecoderGPT support in ``_capture_per_row_continuation``.

Stage 3a (DiffMean) and 3b (LinearProbe) dispatch scalar-baseline
training across both ``"decoder_only"`` and ``"encdec"`` targets. The
shared capture helper at
``steering.eval.axbench_base.baselines._capture._capture_per_row_continuation``
runs one forward pass through the target model to collect the
layer-``L`` residual, then mean-reduces the continuation positions.

For a plain :class:`nanochat.gpt.GPT`, ``model(idx)`` does the right
thing. For a :class:`steering.conditional.encoder_decoder.model.EncoderDecoderGPT`
the bare call raises ``TypeError`` because ``targets`` is a required
positional arg, and the encoder-side tensors
(``enc_feature_ids`` / ``enc_magnitudes`` / ``enc_mask`` / ``seg_map``)
are also required. That makes the shared helper crash whenever Stage 3
reaches the encdec target — a silent latent bug because the Stage 3
integration tests only exercise decoder-only :class:`GPT`.

These tests pin the contract that the capture helper works on BOTH
:class:`GPT` and :class:`EncoderDecoderGPT` so the Stage 3 dispatch
can run to completion for every (baseline, target) combination the
method registry already advertises.
"""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.axbench_base.baselines._capture import (
    _capture_per_row_continuation,
)
from steering.eval.axbench_base.baselines.conditioning import (
    EncoderConditioning,
)
from steering.eval.axbench_base.types import AxBenchBaseError

TINY_VOCAB: int = 128
TINY_N_EMBD: int = 64
TINY_N_LAYER: int = 2
TINY_N_HEAD: int = 2
TINY_N_KV_HEAD: int = 2
TINY_SEQ_LEN: int = 128
TINY_HEAD_DIM: int = 32
TINY_N_SAE_FEATURES: int = 64
TINY_NUM_BUCKETS: int = 4
TINY_BOS_ID: int = 1
TINY_LAYER: int = 0


def _make_tiny_gpt_config() -> GPTConfig:
    return GPTConfig(
        sequence_len=TINY_SEQ_LEN,
        vocab_size=TINY_VOCAB,
        n_layer=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        n_embd=TINY_N_EMBD,
        window_pattern="L",
    )


def _make_tiny_gpt() -> GPT:
    config = _make_tiny_gpt_config()
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=TINY_VOCAB)
    model = model.to_empty(device="cpu")
    model.init_weights()
    model.eval()
    return model


def _make_tiny_encdec() -> EncoderDecoderGPT:
    dec_config = _make_tiny_gpt_config()
    enc_config = EncoderConfig(
        n_embd=TINY_N_EMBD,
        depth=TINY_N_LAYER,
        n_head=TINY_N_HEAD,
        n_kv_head=TINY_N_KV_HEAD,
        head_dim=TINY_HEAD_DIM,
    )
    xattn_config = CrossAttentionConfig(frequency=1)
    model = EncoderDecoderGPT(
        dec_config=dec_config,
        enc_config=enc_config,
        xattn_config=xattn_config,
        n_sae_features=TINY_N_SAE_FEATURES,
        num_buckets=TINY_NUM_BUCKETS,
        pad_vocab_size_to=TINY_VOCAB,
    )
    model.init_weights()
    model.eval()
    return model


class _StubNanochatTokenizer:
    """Deterministic nanochat-like tokenizer matching the real BOS-prepend call."""

    def __init__(
        self, vocab_size: int = TINY_VOCAB, bos_id: int = TINY_BOS_ID,
    ) -> None:
        self.vocab_size = vocab_size
        self.bos_id = bos_id

    def get_bos_token_id(self) -> int:
        return self.bos_id

    def encode(
        self, text: str, *, prepend: int | None = None, **_: Any,
    ) -> list[int]:
        ids = [ord(c) % self.vocab_size for c in text]
        if prepend is not None:
            return [prepend] + ids
        return ids

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


# #1249: encdec capture is CONDITIONED -- the unconditioned build_no_feature_inputs
# path was removed, so every encdec capture here supplies a bag (ids/buckets are
# within the tiny model's n_sae_features / num_buckets).
_TEST_CONDITIONING = EncoderConditioning(entries=((2, 1), (5, 0)))


class TestCaptureOnEncoderDecoderGPT:
    """``_capture_per_row_continuation`` must run on encoder-decoder models.

    The helper is called with a live target model during Stage 3 — if
    the model is an :class:`EncoderDecoderGPT`, the helper must build
    the mandatory encoder-side tensors on the fly so the forward pass
    completes. Under the ``build_no_feature_inputs`` convention the
    cross-attention block yields NaN on its all-False mask and
    ``nan_to_num`` collapses it back to zero, so the captured residual
    is well-defined.
    """

    def test_returns_float_tensor_on_encdec_model(self) -> None:
        model = _make_tiny_encdec()
        tok = _StubNanochatTokenizer()
        prefix_text = "prefix"
        output_text = "output"
        n_prefix_nanochat_tokens = len(
            tok.encode(prefix_text, prepend=TINY_BOS_ID),
        )

        result, acts = _capture_per_row_continuation(
            model=model,
            layer=TINY_LAYER,
            tokenizer=tok,
            prefix_text=prefix_text,
            output_text=output_text,
            n_prefix_nanochat_tokens=n_prefix_nanochat_tokens,
            encoder_conditioning=_TEST_CONDITIONING,
        )

        assert result is not None, (
            "Capture helper returned None for a non-empty continuation slice "
            "on an EncoderDecoderGPT"
        )
        assert result.dtype == t.float32
        assert result.shape == (TINY_N_EMBD,), (
            f"Expected (C,) vector of shape ({TINY_N_EMBD},), got {result.shape}"
        )
        assert acts.dtype == t.float32
        assert acts.ndim == 2 and acts.shape[1] == TINY_N_EMBD
        assert acts.shape[0] >= 1
        assert t.equal(acts.mean(dim=0), result), (
            "Paired capture contract: the mean must be the reduction of the "
            "returned token matrix (CPU: reduction orders coincide bitwise)"
        )
        assert not t.isnan(result).any(), (
            "Capture helper produced NaNs on EncoderDecoderGPT — the "
            "build_no_feature_inputs path should route NaN cross-attention "
            "output through nan_to_num"
        )

    def test_empty_continuation_fails_loudly_on_encdec(self) -> None:
        """An empty committed continuation is an artifact-contract error."""
        model = _make_tiny_encdec()
        tok = _StubNanochatTokenizer()
        prefix_text = "anything"
        output_text = ""
        n_prefix_nanochat_tokens = len(
            tok.encode(prefix_text, prepend=TINY_BOS_ID),
        )

        with pytest.raises(AxBenchBaseError, match="empty output_text"):
            _capture_per_row_continuation(
                model=model,
                layer=TINY_LAYER,
                tokenizer=tok,
                prefix_text=prefix_text,
                output_text=output_text,
                n_prefix_nanochat_tokens=n_prefix_nanochat_tokens,
                encoder_conditioning=_TEST_CONDITIONING,
            )

    @pytest.mark.parametrize(
        ("field", "value", "message"),
        [
            pytest.param("prefix_text", None, "prefix_text", id="null-prefix"),
            pytest.param("output_text", None, "output_text", id="null-output"),
            pytest.param(
                "n_prefix_nanochat_tokens", None, "positive integer", id="null-boundary"
            ),
            pytest.param(
                "n_prefix_nanochat_tokens", True, "positive integer", id="bool-boundary"
            ),
            pytest.param(
                "n_prefix_nanochat_tokens", "7", "positive integer", id="text-boundary"
            ),
            pytest.param(
                "n_prefix_nanochat_tokens", 0, "positive integer", id="zero-boundary"
            ),
        ],
    )
    def test_malformed_capture_inputs_fail_without_coercion(
        self,
        field: str,
        value: Any,
        message: str,
    ) -> None:
        """Committed synth values are exact types, never best-effort casts."""
        kwargs: dict[str, Any] = {
            "model": _make_tiny_encdec(),
            "layer": TINY_LAYER,
            "tokenizer": _StubNanochatTokenizer(),
            "prefix_text": "prefix",
            "output_text": "output",
            "n_prefix_nanochat_tokens": 7,
        }
        kwargs[field] = value

        with pytest.raises(AxBenchBaseError, match=message):
            _capture_per_row_continuation(**kwargs)

    def test_encdec_capture_is_deterministic(self) -> None:
        """Two captures of the same (model, prefix, output) must agree
        to float-32 precision. Catches hidden stochasticity in the
        encdec forward path (e.g. un-fixed dropout)."""
        model = _make_tiny_encdec()
        tok = _StubNanochatTokenizer()
        prefix_text = "prefix-text"
        output_text = "continuation-text"
        n_prefix_nanochat_tokens = len(
            tok.encode(prefix_text, prepend=TINY_BOS_ID),
        )

        result_a, acts_a = _capture_per_row_continuation(
            model=model,
            layer=TINY_LAYER,
            tokenizer=tok,
            prefix_text=prefix_text,
            output_text=output_text,
            n_prefix_nanochat_tokens=n_prefix_nanochat_tokens,
            encoder_conditioning=_TEST_CONDITIONING,
        )
        result_b, acts_b = _capture_per_row_continuation(
            model=model,
            layer=TINY_LAYER,
            tokenizer=tok,
            prefix_text=prefix_text,
            output_text=output_text,
            n_prefix_nanochat_tokens=n_prefix_nanochat_tokens,
            encoder_conditioning=_TEST_CONDITIONING,
        )

        assert result_a is not None and result_b is not None
        assert t.allclose(result_a, result_b, atol=1e-6), (
            "Repeat calls produced different activations — the encdec "
            "capture path is leaking non-determinism"
        )
        assert t.allclose(acts_a, acts_b, atol=1e-6)

    def test_gpt_still_works_regression(self) -> None:
        """Pins that the decoder-only GPT path is preserved after adding
        encdec support — the isinstance dispatch must not regress the
        plain GPT call site."""
        model = _make_tiny_gpt()
        tok = _StubNanochatTokenizer()
        prefix_text = "prefix"
        output_text = "output"
        n_prefix_nanochat_tokens = len(
            tok.encode(prefix_text, prepend=TINY_BOS_ID),
        )

        result, _acts = _capture_per_row_continuation(
            model=model,
            layer=TINY_LAYER,
            tokenizer=tok,
            prefix_text=prefix_text,
            output_text=output_text,
            n_prefix_nanochat_tokens=n_prefix_nanochat_tokens,
        )

        assert result is not None
        assert result.shape == (TINY_N_EMBD,)
        assert not t.isnan(result).any()

    def test_encdec_training_mode_restored(self) -> None:
        """Calling the helper on a model in ``model.train()`` mode must
        leave it in training mode afterwards. Capture only runs in eval
        during the forward — the original mode must be restored even for
        encdec."""
        model = _make_tiny_encdec()
        model.train()
        assert model.training

        tok = _StubNanochatTokenizer()
        prefix_text = "prefix"
        output_text = "output"
        n_prefix_nanochat_tokens = len(
            tok.encode(prefix_text, prepend=TINY_BOS_ID),
        )

        _capture_per_row_continuation(
            model=model,
            layer=TINY_LAYER,
            tokenizer=tok,
            prefix_text=prefix_text,
            output_text=output_text,
            n_prefix_nanochat_tokens=n_prefix_nanochat_tokens,
            encoder_conditioning=_TEST_CONDITIONING,
        )

        assert model.training, (
            "Capture helper must restore training mode after running on "
            "EncoderDecoderGPT"
        )
