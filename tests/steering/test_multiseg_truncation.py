"""Hard-fail tests for multiseg generators that previously silently truncated.

Map SB-13-P0-05:

    ``multiseg_generator.generate_completion`` and
    ``generate_completion_stream`` previously silently left-truncated
    ``prefix_tokens`` when over budget; sister functions
    (``generate_for_document``, ``generate_samples_for_condition``) raise
    explicitly. Silent truncation drops BOS, breaking
    ``token_to_char_offsets`` alignment.

Map SB-13-P0-06:

    ``_decode_from_prefill`` indexes ``seg_map[:, cur_pos-1:cur_pos]``;
    if a caller plumbs ``max_new_tokens`` larger than the seg_map's gen
    budget, the slice silently produces a ``(1, 0)`` tensor.

Both fixes raise loudly.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.eval.multiseg_generator import (
    MultisegInputs,
    _decode_from_prefill,
    generate_completion,
    generate_completion_stream,
)
from tests.conftest import _make_encdec
from tests.padded_encoder_reference import pack_dense_encoder_output_reference

_DEVICE = t.device("cpu")


def _empty_encoder_inputs(seg_map_T: int) -> MultisegInputs:
    """Build a minimal encoder-inputs payload sized to a given seg_map T.

    Used to drive the truncation / size-assertion checks without requiring
    real encoder content (the relevant code paths fail before the encoder
    is invoked).
    """
    enc_feature_ids = t.zeros(1, 1, 1, dtype=t.int64)
    enc_magnitudes = t.zeros(1, 1, 1, dtype=t.int64)
    enc_mask = t.zeros(1, 1, 1, dtype=t.bool)
    seg_map = t.zeros(1, seg_map_T, dtype=t.int64)
    return MultisegInputs(
        enc_feature_ids=enc_feature_ids,
        enc_magnitudes=enc_magnitudes,
        enc_mask=enc_mask,
        seg_map=seg_map,
    )


# ---------------------------------------------------------------------------
# A7 / SB-13-P0-05: silent left-truncation → hard error.
# ---------------------------------------------------------------------------


def test_generate_completion_overrun_raises():
    """``generate_completion`` raises if prefix is too long."""
    model = _make_encdec(device="cpu")
    seq_len = model.dec_config.sequence_len  # 128 in conftest TINY config
    max_new_tokens = 8
    too_long_prefix = list(range(seq_len - max_new_tokens + 1))  # too long by 1
    encoder_inputs = _empty_encoder_inputs(seg_map_T=len(too_long_prefix) + max_new_tokens)
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        generate_completion(
            model=model,
            prefix_tokens=too_long_prefix,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new_tokens,
            device=_DEVICE,
        )


def test_generate_completion_stream_overrun_raises():
    """``generate_completion_stream`` raises (parity with batch sister)."""
    model = _make_encdec(device="cpu")
    seq_len = model.dec_config.sequence_len
    max_new_tokens = 8
    too_long_prefix = list(range(seq_len - max_new_tokens + 1))
    encoder_inputs = _empty_encoder_inputs(seg_map_T=len(too_long_prefix) + max_new_tokens)
    # generate_completion_stream is a generator; the raise fires when we
    # advance it for the first time (or when we collect into list).
    with pytest.raises(ValueError, match="exceeds max_seq_len"):
        list(
            generate_completion_stream(
                model=model,
                prefix_tokens=too_long_prefix,
                encoder_inputs=encoder_inputs,
                max_new_tokens=max_new_tokens,
                device=_DEVICE,
            )
        )


def test_generate_completion_at_budget_does_not_raise_truncation():
    """Prefix exactly at the budget edge does NOT raise the truncation
    guard; if the model errors later, that's a different concern."""
    model = _make_encdec(device="cpu")
    seq_len = model.dec_config.sequence_len
    max_new_tokens = 8
    # Build a prefix that fits exactly: prefix + max_new_tokens == seq_len.
    valid_prefix = list(range(2, seq_len - max_new_tokens + 2))
    # We need a seg_map sized for the full sequence so the decoder doesn't
    # also raise the seg_map-size assertion. Encoder inputs need 1 segment.
    encoder_inputs = MultisegInputs(
        enc_feature_ids=t.zeros(1, 1, 1, dtype=t.int64),
        enc_magnitudes=t.zeros(1, 1, 1, dtype=t.int64),
        enc_mask=t.zeros(1, 1, 1, dtype=t.bool),
        seg_map=t.zeros(1, seq_len, dtype=t.int64),
    )
    # The function may still error somewhere downstream (non-finite logits
    # from a fresh init etc.), but the truncation guard should NOT fire.
    try:
        generate_completion(
            model=model,
            prefix_tokens=valid_prefix,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new_tokens,
            device=_DEVICE,
            temperature=0.0,
            top_k=None,
            seed=0,
            bos_id=0,
        )
    except ValueError as e:
        # An overrun ValueError would name "exceeds max_seq_len".
        assert "exceeds max_seq_len" not in str(e), (
            f"Truncation guard fired at the budget edge: {e}"
        )
    except FloatingPointError:
        # Acceptable: a tiny CPU model fresh-initialised can produce NaN
        # logits; the NaN guard fires.
        pass


def test_generate_completion_under_budget_does_not_raise_truncation():
    """A prefix shorter than the budget is unconditionally fine for the
    truncation guard."""
    model = _make_encdec(device="cpu")
    max_new_tokens = 8
    short_prefix = list(range(2, 5))
    encoder_inputs = _empty_encoder_inputs(
        seg_map_T=len(short_prefix) + max_new_tokens,
    )
    try:
        generate_completion(
            model=model,
            prefix_tokens=short_prefix,
            encoder_inputs=encoder_inputs,
            max_new_tokens=max_new_tokens,
            device=_DEVICE,
            temperature=0.0,
            top_k=None,
            seed=0,
            bos_id=0,
        )
    except ValueError as e:
        assert "exceeds max_seq_len" not in str(e)
    except FloatingPointError:
        pass


# ---------------------------------------------------------------------------
# A8 / SB-13-P0-06: seg_map size assertion in _decode_from_prefill.
# ---------------------------------------------------------------------------


def test_decode_from_prefill_seg_map_too_short_raises():
    """``_decode_from_prefill`` rejects a seg_map smaller than ``total_len``.

    Previously the slice ``seg_map[:, cur_pos-1:cur_pos]`` would silently
    produce a ``(1, 0)`` tensor once ``cur_pos > seg_map.shape[1]``."""
    model = _make_encdec(device="cpu")
    n_prefix = 4
    max_new_tokens = 4
    total_len = n_prefix + max_new_tokens
    # seg_map intentionally too short.
    short_seg_map = t.zeros(1, total_len - 2, dtype=t.int64)
    fake_logits = t.zeros(1, model.dec_config.vocab_size, dtype=t.float32)
    fake_enc_mask = t.ones(1, 1, dtype=t.bool)
    fake_enc_out = pack_dense_encoder_output_reference(
        t.zeros(1, 1, 1, model.encoder.config.n_embd),
        fake_enc_mask.unsqueeze(-1),
    )
    with pytest.raises(ValueError, match="seg_map covers"):
        list(
            _decode_from_prefill(
                model=model,
                kv_cache=None,  # never reached  # ty: ignore[invalid-argument-type]
                last_logits=fake_logits,
                encoder_out=fake_enc_out,
                seg_map=short_seg_map,
                n_prefix=n_prefix,
                max_new_tokens=max_new_tokens,
                total_len=total_len,
                device=_DEVICE,
                temperature=0.0,
                top_k=None,
                seed=0,
                bos_id=0,
            )
        )


def test_decode_from_prefill_seg_map_exact_size_does_not_raise_seg_map_check():
    """A seg_map with shape ``(1, total_len)`` passes the assertion."""
    model = _make_encdec(device="cpu")
    n_prefix = 4
    max_new_tokens = 4
    total_len = n_prefix + max_new_tokens
    fits_seg_map = t.zeros(1, total_len, dtype=t.int64)
    fake_logits = t.zeros(1, model.dec_config.vocab_size, dtype=t.float32)
    fake_enc_mask = t.ones(1, 1, dtype=t.bool)
    fake_enc_out = pack_dense_encoder_output_reference(
        t.zeros(1, 1, 1, model.encoder.config.n_embd),
        fake_enc_mask.unsqueeze(-1),
    )
    # We expect either successful generation, FloatingPointError (NaN
    # logits from a CPU encdec fresh init), or some downstream error —
    # but NOT the seg_map-size assertion.
    try:
        list(
            _decode_from_prefill(
                model=model,
                kv_cache=None,  # ty: ignore[invalid-argument-type]
                last_logits=fake_logits,
                encoder_out=fake_enc_out,
                seg_map=fits_seg_map,
                n_prefix=n_prefix,
                max_new_tokens=max_new_tokens,
                total_len=total_len,
                device=_DEVICE,
                temperature=0.0,
                top_k=None,
                seed=0,
                bos_id=0,
            )
        )
    except ValueError as e:
        assert "seg_map covers" not in str(e), (
            f"seg_map-size check fired even though seg_map was sized "
            f"correctly: {e}"
        )
    except (FloatingPointError, AttributeError, RuntimeError):
        # AttributeError: kv_cache=None means model.forward will eventually
        # fail. That's expected for this size-assertion test.
        pass
