"""Slice-09 audit: ``seg_map`` contract assertions in CrossAttentionLayer.

Maps:
- map.md P2-1 (manifest §4.18): non-monotonic seg_map among *non-zero*
  entries must raise. ``seg=0`` is the SFT pipeline's no-encoder-context
  sentinel and may repeat freely between non-zero runs, so the assertion
  only rejects a non-zero value followed by a smaller non-zero value.
- map.md P2-2: out-of-range seg_map must raise.

The assertions are eager-only (gated by ``torch.compiler.is_compiling()``) so
they don't break compile traces.
"""

from unittest.mock import patch

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
)


def _make_layer(dec_n_embd: int = 32, enc_n_embd: int = 16, n_head: int = 4, n_kv_head: int = 2, head_dim: int = 8) -> CrossAttentionLayer:
    cfg = CrossAttentionLayerConfig(
        dec_n_embd=dec_n_embd,
        enc_n_embd=enc_n_embd,
        n_head=n_head,
        n_kv_head=n_kv_head,
        head_dim=head_dim,
    )
    layer = CrossAttentionLayer(cfg)
    layer.init_weights()
    return layer


def _make_inputs(B: int = 2, T: int = 8, n_seg: int = 2, F_feat: int = 4, dec_n_embd: int = 32, enc_n_embd: int = 16):
    g = t.Generator().manual_seed(20260505)
    decoder_hidden = t.randn(B, T, dec_n_embd, generator=g)
    encoder_kv = t.randn(B, n_seg, F_feat, enc_n_embd, generator=g)
    encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)
    seg_size = T // n_seg
    seg_map = t.arange(T).unsqueeze(0).expand(B, -1).contiguous() // seg_size
    return decoder_hidden, encoder_kv, encoder_mask, seg_map


# ----------------------------- Functional ------------------------------------


def test_non_monotonic_nonzero_seg_map_raises_with_helpful_message() -> None:
    """A non-zero value followed by a smaller non-zero value must raise.

    ``seg=0`` is a sentinel that may repeat (multi-turn SFT data uses this
    pattern) so the assertion is scoped to non-zero entries.
    """
    layer = _make_layer(dec_n_embd=32, enc_n_embd=16, n_head=4, n_kv_head=2, head_dim=8)
    B, T, n_seg, F_feat = 2, 8, 3, 4
    g = t.Generator().manual_seed(20260505)
    decoder_hidden = t.randn(B, T, 32, generator=g)
    encoder_kv = t.randn(B, n_seg, F_feat, 16, generator=g)
    encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)
    # Row 0 violates the non-zero monotonicity: 2 is followed by 1.
    seg_map = t.tensor(
        [
            [0, 0, 1, 1, 2, 2, 1, 1],
            [0, 0, 1, 1, 2, 2, 2, 2],
        ],
        dtype=t.long,
    )
    with pytest.raises(AssertionError, match="non-decreasing"):
        layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)


def test_zero_sentinel_between_nonzero_runs_does_not_raise() -> None:
    """``seg=0`` may appear anywhere for empty/padding slots."""
    layer = _make_layer(dec_n_embd=32, enc_n_embd=16, n_head=4, n_kv_head=2, head_dim=8)
    B, T, n_seg, F_feat = 2, 8, 3, 4
    g = t.Generator().manual_seed(20260505)
    decoder_hidden = t.randn(B, T, 32, generator=g)
    encoder_kv = t.randn(B, n_seg, F_feat, 16, generator=g)
    encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)
    # The 0 -> 1 -> 0 -> 2 interleave pattern can occur when padding or
    # explicit empty routing is interspersed with real feature segments.
    seg_map = t.tensor(
        [
            [0, 0, 1, 1, 0, 0, 2, 2],
            [0, 1, 1, 0, 0, 2, 2, 2],
        ],
        dtype=t.long,
    )
    out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
    assert out.shape == (B, T, 32)


def test_seg_map_out_of_range_raises_with_helpful_message() -> None:
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs(n_seg=2)
    seg_map = seg_map.clone()
    # Push the trailing positions past n_seg=2.
    seg_map[:, -1] = 3
    # Make the first entries follow monotonicity, ending at 3.
    seg_map[0, -2] = 2
    seg_map[1, -2] = 2
    with pytest.raises(AssertionError, match=r"seg_map values must be < n_seg"):
        layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)


def test_seg_map_negative_values_raise() -> None:
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _make_inputs()
    # We want monotonicity to pass but min < 0 to fire. Build a non-decreasing
    # map starting at -1.
    seg_map = t.tensor([[-1, -1, 0, 0, 1, 1, 1, 1]] * decoder_hidden.size(0))
    with pytest.raises(AssertionError, match=r"seg_map values must be >= 0"):
        layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)


def test_assertions_silenced_under_torch_compile() -> None:
    """When torch.compiler.is_compiling() is True the validation block is
    short-circuited so it doesn't break compile traces."""
    layer = _make_layer(dec_n_embd=32, enc_n_embd=16, n_head=4, n_kv_head=2, head_dim=8)
    B, T, n_seg, F_feat = 2, 8, 3, 4
    g = t.Generator().manual_seed(20260505)
    decoder_hidden = t.randn(B, T, 32, generator=g)
    encoder_kv = t.randn(B, n_seg, F_feat, 16, generator=g)
    encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)
    # Non-zero monotonicity violation that would normally raise.
    seg_map = t.tensor(
        [
            [0, 0, 1, 1, 2, 2, 1, 1],
            [0, 0, 1, 1, 2, 2, 2, 2],
        ],
        dtype=t.long,
    )
    with patch("torch.compiler.is_compiling", return_value=True):
        # The cummax-based pos_in_seg will compute *something* (semantically
        # garbage) but no assertion fires. We just want to confirm the call
        # path doesn't raise the audit-added assertion.
        try:
            layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
        except AssertionError as exc:
            if "non-decreasing" in str(exc) or "seg_map values must" in str(exc):
                pytest.fail("audit-added assertion fired under is_compiling=True")
            # Other assertions (e.g. unrelated SDPA path) don't violate the
            # contract under test; re-raise so they remain visible.
            raise
        except RuntimeError:
            # Out-of-bounds scatter etc. is the documented unsafe behaviour
            # the assertion guards against; the test only cares that the
            # *audit assertion* itself is suppressed under compile.
            pass


def test_single_token_skips_monotonicity_check() -> None:
    """T=1 has no pair to compare; the monotonicity guard must not raise."""
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, _ = _make_inputs(B=2, T=1, n_seg=1, F_feat=4)
    seg_map = t.zeros(2, 1, dtype=t.long)
    out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
    assert out.shape == (2, 1, layer.dec_n_embd)


# ----------------------------- Property --------------------------------------


@settings(
    derandomize=True,
    max_examples=50,
    deadline=None,
    suppress_health_check=[HealthCheck.too_slow],
)
@given(
    n_seg=st.integers(min_value=1, max_value=4),
    seg_lens=st.lists(st.integers(min_value=1, max_value=4), min_size=1, max_size=4),
)
def test_property_monotonic_seg_map_never_raises(n_seg: int, seg_lens: list[int]) -> None:
    """Any properly-constructed monotonic seg_map with values in [0, n_seg)
    passes the audit assertion."""
    seg_lens = seg_lens[:n_seg]
    if not seg_lens:
        return
    n_seg_eff = len(seg_lens)
    T = sum(seg_lens)
    seg_map_row = []
    for s, length in enumerate(seg_lens):
        seg_map_row.extend([s] * length)
    seg_map = t.tensor([seg_map_row], dtype=t.long)
    B = 1
    F_feat = 2
    layer = _make_layer(dec_n_embd=16, enc_n_embd=8, n_head=2, n_kv_head=1, head_dim=8)
    g = t.Generator().manual_seed(20260505)
    decoder_hidden = t.randn(B, T, 16, generator=g)
    encoder_kv = t.randn(B, n_seg_eff, F_feat, 8, generator=g)
    encoder_mask = t.ones(B, n_seg_eff, F_feat, dtype=t.bool)
    out = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
    assert out.shape == (B, T, 16)
