"""Slice-09 audit: ``static_max_seg_size`` validation in CrossAttentionLayer.

Issue #723 raw-bound contract:
- A too-small ``static_max_seg_size`` (less than ``pos_in_seg.amax()+1``) must raise
  rather than silently corrupting via out-of-bounds scatter.
- The packed-row manifest value is a raw semantic maximum and need not be a
  multiple of ``FP8_ALIGNMENT=16``.
- The CPU SDPA reference must round the raw value to an aligned allocation
  extent without weakening the raw stale-bound check. CUDA packed attention
  uses the value only as a semantic bound.
- The CPU dynamic and static paths must produce numerically equivalent output
  for identical inputs when they resolve to the same allocation extent.
- Eager and compiled execution must accept valid raw values and reject stale
  raw values, including when required and stale values round to the same extent.
"""

import pytest
import torch as t

from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.cross_attention import (
    CrossAttentionLayer,
    CrossAttentionLayerConfig,
)


def _make_layer() -> CrossAttentionLayer:
    cfg = CrossAttentionLayerConfig(dec_n_embd=32, enc_n_embd=16, n_head=4, n_kv_head=2, head_dim=8)
    layer = CrossAttentionLayer(cfg)
    layer.init_weights()
    return layer


def _inputs(B: int = 2, T: int = 32, n_seg: int = 2, F_feat: int = 4):
    g = t.Generator().manual_seed(20260505)
    decoder_hidden = t.randn(B, T, 32, generator=g)
    encoder_kv = t.randn(B, n_seg, F_feat, 16, generator=g)
    encoder_mask = t.ones(B, n_seg, F_feat, dtype=t.bool)
    seg_size = T // n_seg
    seg_map = t.arange(T).unsqueeze(0).expand(B, -1).contiguous() // seg_size
    return decoder_hidden, encoder_kv, encoder_mask, seg_map


def test_too_small_static_max_seg_size_raises() -> None:
    layer = _make_layer()
    # Each segment has 32 tokens. Give an aligned-but-stale raw bound of 16 so
    # this targets the semantic upper-bound guard rather than positivity.
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs(B=1, T=64, n_seg=2, F_feat=4)
    with pytest.raises(AssertionError, match="smaller than"):
        layer(
            decoder_hidden,
            encoder_kv,
            encoder_mask,
            seg_map,
            static_max_seg_size=FP8_ALIGNMENT,  # = 16, but each seg has 32 toks
        )


def test_raw_unaligned_static_max_seg_size_is_accepted_and_allocates_aligned() -> None:
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs(
        B=1,
        T=268,
        n_seg=2,
    )
    query_shapes: list[tuple[int, ...]] = []
    handle = layer.c_q.register_forward_pre_hook(
        lambda _module, args: query_shapes.append(tuple(args[0].shape))
    )
    try:
        out = layer(
            decoder_hidden,
            encoder_kv,
            encoder_mask,
            seg_map,
            static_max_seg_size=134,
        )
    finally:
        handle.remove()

    assert out.shape == decoder_hidden.shape
    assert t.isfinite(out).all()
    assert query_shapes == [(2, 144, 32)]


def test_stale_raw_bound_fails_even_inside_same_alignment_bucket() -> None:
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs(
        B=1,
        T=268,
        n_seg=2,
    )
    # required=134 and stale=133 both round to allocation extent 144. The raw
    # semantic bound must still be checked before allocation alignment.
    with pytest.raises(
        AssertionError,
        match=r"static_max_seg_size=133.*required=134",
    ):
        layer(
            decoder_hidden,
            encoder_kv,
            encoder_mask,
            seg_map,
            static_max_seg_size=133,
        )


@pytest.mark.parametrize("bad_bound", [0, -1])
def test_non_positive_static_max_seg_size_raises(bad_bound: int) -> None:
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs()
    with pytest.raises(AssertionError, match="must be > 0"):
        layer(
            decoder_hidden,
            encoder_kv,
            encoder_mask,
            seg_map,
            static_max_seg_size=bad_bound,
        )


def test_static_path_matches_dynamic_path_numerically() -> None:
    """Raw-static, dynamic, and explicitly aligned paths are bit-identical."""
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs(
        B=2,
        T=34,
        n_seg=2,
    )
    # required/raw=17; every path allocates 32 positions per segment.
    out_dynamic = layer(decoder_hidden, encoder_kv, encoder_mask, seg_map)
    out_raw_static = layer(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=17,
    )
    out_aligned_static = layer(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=32,
    )
    t.testing.assert_close(out_dynamic, out_raw_static, atol=0.0, rtol=0.0)
    t.testing.assert_close(out_dynamic, out_aligned_static, atol=0.0, rtol=0.0)


def test_static_path_larger_than_required_works() -> None:
    """A larger raw value is accepted and aligned internally."""
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs(B=2, T=32, n_seg=2)
    # required=16; raw=31 and explicit allocation extent=32 must agree.
    out_raw = layer(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=31,
    )
    out_aligned = layer(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=32,
    )
    assert out_raw.shape == decoder_hidden.shape
    assert t.isfinite(out_raw).all()
    t.testing.assert_close(out_raw, out_aligned, atol=0.0, rtol=0.0)


@pytest.mark.parametrize("dynamic", [False, True, None])
def test_compiled_path_accepts_raw_bound_and_rejects_same_bucket_stale_bound(
    dynamic: bool | None,
) -> None:
    """Compiled execution enforces the same raw semantic bound as eager."""
    t._dynamo.reset()
    layer = _make_layer()
    decoder_hidden, encoder_kv, encoder_mask, seg_map = _inputs(
        B=1,
        T=34,
        n_seg=2,
    )
    eager = layer(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=17,
    )
    compiled = t.compile(layer, fullgraph=True, dynamic=dynamic)

    out_raw = compiled(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=17,
    )
    out_aligned = compiled(
        decoder_hidden,
        encoder_kv,
        encoder_mask,
        seg_map,
        static_max_seg_size=32,
    )
    assert out_raw.shape == decoder_hidden.shape
    assert t.isfinite(out_raw).all()
    # The two compiled calls have the identical allocation extent and are
    # bit-exact. Inductor may reorder floating-point operations relative to
    # eager; the observed CPU error is <=2.4e-7, so 1e-6 is a tight guard.
    t.testing.assert_close(out_raw, out_aligned, atol=0.0, rtol=0.0)
    t.testing.assert_close(out_raw, eager, atol=1e-6, rtol=1e-6)

    # Reuse the fixed raw bound and tensor shapes that produced the valid graph,
    # but change the runtime segment distribution from [17, 17] to [18, 16].
    # required=18 and raw=17 both round to 32, so an implementation that merely
    # aligns the manifest value would silently accept this later batch.
    stale_seg_map = seg_map.clone()
    stale_seg_map[:, :18] = 0
    stale_seg_map[:, 18:] = 1
    with pytest.raises(RuntimeError, match=r"static_max_seg_size is smaller"):
        compiled(
            decoder_hidden,
            encoder_kv,
            encoder_mask,
            stale_seg_map,
            static_max_seg_size=17,
        )
