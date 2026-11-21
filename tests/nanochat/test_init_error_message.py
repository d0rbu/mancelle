"""Slice-09 audit: GQA invariant assertion error message (map P3-4).

The construction-time assertion ``n_kv_head <= n_head and n_head % n_kv_head == 0``
previously raised a bare ``AssertionError`` with no message. The audit adds a
helpful message so configuration mistakes surface their cause.
"""

import pytest

from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
)


def _config(n_head: int, n_kv_head: int) -> CrossAttentionLayerConfig:
    return CrossAttentionLayerConfig(
        dec_n_embd=32,
        enc_n_embd=16,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=8,
    )


def test_n_kv_head_greater_than_n_head_raises_with_message() -> None:
    cfg = _config(n_head=4, n_kv_head=8)
    with pytest.raises(AssertionError, match="GQA invariant violated"):
        CrossAttentionLayer(cfg)


def test_n_head_not_divisible_by_n_kv_head_raises_with_message() -> None:
    cfg = _config(n_head=4, n_kv_head=3)
    with pytest.raises(AssertionError, match="GQA invariant violated"):
        CrossAttentionLayer(cfg)


def test_error_message_includes_actual_values() -> None:
    cfg = _config(n_head=4, n_kv_head=3)
    with pytest.raises(AssertionError) as excinfo:
        CrossAttentionLayer(cfg)
    msg = str(excinfo.value)
    assert "n_head=4" in msg
    assert "n_kv_head=3" in msg


def test_valid_gqa_config_does_not_raise() -> None:
    """Sanity check: legal GQA configurations still succeed."""
    for n_head, n_kv_head in [(4, 1), (4, 2), (4, 4), (8, 2)]:
        layer = CrossAttentionLayer(_config(n_head=n_head, n_kv_head=n_kv_head))
        assert layer.n_head == n_head
        assert layer.n_kv_head == n_kv_head
