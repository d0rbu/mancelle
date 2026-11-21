"""CRITICAL sign-convention tests for the collector's gradient SENSITIVITY.

The collector computes ``max over rows of ReLU(-dL/df)`` (gated by activity) —
how strongly INCREASING a feature would DECREASE the unit's loss; a feature
scores > 0 iff it fired AND its increase helps. (The producer separately
multiplies this by the activation percentile to form the stored
pct*sensitivity score — that weighting is tested at the producer level.) A
sign inversion anywhere (the relu argument, the decoder VJP, the loss sign)
would silently rank loss-*increasing* features instead; these tests construct
decoder directions where the correct sign is provable and corroborate with
finite differences.
"""

from __future__ import annotations

import pytest
import torch as t

from errors import GradientAttributionError
from steering.gradient_attribution import (
    GradientAttributionCollector,
    decoder_linearity_gate,
)
from steering.gradient_overlay import GradientAttributionMode
from tests.reference_gradient_attribution import (
    TINY_LAYER,
    fixed_segments,
    make_tiny_sae,
    ref_capture,
    ref_unit_gradient,
    ref_unit_loss,
    seeded_ids,
    tiny_gemma2,
)


def _suffix_positions(doc_len: int, seg_end: int) -> list[int]:
    return [j for j in range(doc_len - 1) if j + 1 >= seg_end]


def _planted_sae(direction: t.Tensor):
    """A tiny SAE whose feature 0 decodes to ``-direction`` (so increasing it
    moves the residual AGAINST the gradient => decreases loss to first order)
    and feature 1 decodes to ``+direction`` (increases loss). Both features'
    encoders are built to fire on the given resid rows (positive gate)."""
    sae = make_tiny_sae("standard", seed=123)
    unit = direction / direction.norm()
    with t.no_grad():
        sae.W_dec[0] = -unit
        sae.W_dec[1] = unit
        # Make features 0 and 1 fire on every row: encoder weight 0 with a
        # positive bias guarantees a positive pre-activation.
        sae.W_enc[:, 0] = 0.0
        sae.W_enc[:, 1] = 0.0
        sae.b_enc[0] = 1.0
        sae.b_enc[1] = 1.0
    return sae


def test_helpful_direction_scores_positive_harmful_scores_zero():
    model = tiny_gemma2(softcap=True)
    doc_len, seg_size = 24, 8
    ids = seeded_ids(doc_len, seed=31)
    segs = fixed_segments(doc_len, seg_size)
    seg_idx = 0
    b, e = segs[seg_idx]

    positions = _suffix_positions(doc_len, e)
    g = ref_unit_gradient(model, ids, TINY_LAYER, positions)  # [T, d]
    # Pick the read row with the largest gradient so the planted effect is
    # well above numerical noise.
    row = int(g[b:e].norm(dim=-1).argmax().item()) + b
    direction = g[row]
    assert direction.norm() > 0

    sae = _planted_sae(direction)
    fids = [t.arange(sae.W_dec.shape[0], dtype=t.int64) for _ in segs]
    collector = GradientAttributionCollector(
        model, sae,
        layer=TINY_LAYER,
        mode=GradientAttributionMode.future_half_split,
        min_final_segment_tokens=8,
    )
    scores = collector.score_documents([ids], [segs], [fids])[0][seg_idx]

    # Feature 0 decodes along -grad: dL/df0 = grad . (-unit) < 0 => helpful.
    expected = float(direction.norm())
    assert scores[0].item() > 0, "anti-gradient feature must score positive"
    assert scores[0].item() == pytest.approx(expected, rel=2e-3), (
        "score should equal |grad| at the dominant row for a unit "
        "anti-gradient decoder direction"
    )
    # Feature 1 decodes along +grad: dL/df1 = |grad| > 0 => ReLU(-g) == 0.
    assert scores[1].item() == 0.0, (
        "a feature whose increase raises future loss must score exactly 0"
    )
    # NOTE: the collector returns the bare gradient SENSITIVITY here; the
    # activation-percentile weighting (pct * sensitivity) is applied by the
    # producer and exercised in the producer-integration tests.


def test_finite_difference_corroboration():
    """Moving the residual along the planted helpful decoder direction must
    actually LOWER the suffix loss (first-order slope check, tolerance)."""
    model = tiny_gemma2(softcap=True)
    doc_len = 16
    ids = seeded_ids(doc_len, seed=77)
    seg_end = 8
    positions = _suffix_positions(doc_len, seg_end)

    g = ref_unit_gradient(model, ids, TINY_LAYER, positions)
    row = int(g[:seg_end].norm(dim=-1).argmax().item())
    direction = -g[row] / g[row].norm()  # helpful: against the gradient

    def loss_with_offset(alpha: float) -> float:
        captured = []

        def _hook(_m, _a, output):
            h = output[0] if isinstance(output, tuple) else output
            h = h.clone()
            h[0, row] = h[0, row] + alpha * direction
            captured.append(h)
            return (h, *output[1:]) if isinstance(output, tuple) else h

        handle = model.model.layers[TINY_LAYER].register_forward_hook(_hook)
        try:
            input_ids = t.tensor(ids).unsqueeze(0)
            hidden = model.model(
                input_ids=input_ids, attention_mask=None, use_cache=False
            ).last_hidden_state
            return float(ref_unit_loss(model, hidden, ids, positions))
        finally:
            handle.remove()

    base = loss_with_offset(0.0)
    bumped = loss_with_offset(1e-3)
    assert bumped < base, (
        f"loss must decrease along the helpful direction "
        f"(base={base:.6f}, bumped={bumped:.6f})"
    )
    slope = (bumped - base) / 1e-3
    assert slope == pytest.approx(-float(g[row].norm()), rel=5e-2)


def test_gate_blocks_inactive_features():
    """A helpful decoder direction must score 0 when the feature never fires
    in the read window (the activity gate)."""
    model = tiny_gemma2(softcap=True)
    doc_len, seg_size = 24, 8
    ids = seeded_ids(doc_len, seed=31)
    segs = fixed_segments(doc_len, seg_size)
    b, e = segs[0]
    positions = _suffix_positions(doc_len, e)
    g = ref_unit_gradient(model, ids, TINY_LAYER, positions)
    row = int(g[b:e].norm(dim=-1).argmax().item()) + b

    sae = _planted_sae(g[row])
    with t.no_grad():
        sae.b_enc[0] = -1e6  # feature 0 can never fire
    fids = [t.arange(sae.W_dec.shape[0], dtype=t.int64) for _ in segs]
    collector = GradientAttributionCollector(
        model, sae,
        layer=TINY_LAYER,
        mode=GradientAttributionMode.future_half_split,
        min_final_segment_tokens=8,
    )
    scores = collector.score_documents([ids], [segs], [fids])[0][0]
    assert scores[0].item() == 0.0, "gate must zero an inactive feature"


def test_jumprelu_gate_vs_preactivation():
    """JumpReLU gating is on the POST-jump activation (act > 0 means
    pre > threshold), NOT on pre > 0: a feature with 0 < pre <= threshold
    must be gated OFF."""
    model = tiny_gemma2(softcap=True)
    doc_len, seg_size = 16, 8
    ids = seeded_ids(doc_len, seed=5)
    segs = fixed_segments(doc_len, seg_size)
    b, e = segs[0]
    positions = _suffix_positions(doc_len, e)
    g = ref_unit_gradient(model, ids, TINY_LAYER, positions)
    row = int(g[b:e].norm(dim=-1).argmax().item()) + b

    sae = make_tiny_sae("jumprelu", seed=123)
    unit = g[row] / g[row].norm()
    with t.no_grad():
        sae.W_dec[0] = -unit
        sae.W_dec[1] = -unit
        sae.W_enc[:, 0] = 0.0
        sae.W_enc[:, 1] = 0.0
        sae.b_enc[0] = 0.5
        sae.b_enc[1] = 0.5
        sae.threshold[0] = 0.1  # pre (0.5) > threshold => fires
        sae.threshold[1] = 0.9  # 0 < pre (0.5) <= threshold => silent
    _, capt = ref_capture(model, ids, TINY_LAYER)
    with t.no_grad():
        acts = sae.encode(capt[0, b:e].detach())
    assert bool((acts[:, 0] > 0).all()) and bool((acts[:, 1] == 0).all())

    fids = [t.arange(sae.W_dec.shape[0], dtype=t.int64) for _ in segs]
    collector = GradientAttributionCollector(
        model, sae,
        layer=TINY_LAYER,
        mode=GradientAttributionMode.future_half_split,
        min_final_segment_tokens=8,
    )
    scores = collector.score_documents([ids], [segs], [fids])[0][0]
    assert scores[0].item() > 0.0
    assert scores[1].item() == 0.0, (
        "sub-threshold JumpReLU pre-activation must gate the feature OFF "
        "(gate is act > 0, not pre > 0)"
    )


def test_non_finite_scores_raise():
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("standard", seed=1)
    with t.no_grad():
        sae.W_dec.mul_(t.inf)  # poison the decoder => non-finite VJP
        sae.b_enc.fill_(1.0)
        sae.W_enc.zero_()  # every feature fires
    ids = seeded_ids(16, seed=2)
    segs = fixed_segments(16, 8)
    fids = [t.arange(sae.W_dec.shape[0], dtype=t.int64) for _ in segs]
    collector = GradientAttributionCollector(
        model, sae,
        layer=TINY_LAYER,
        mode=GradientAttributionMode.future_half_split,
        min_final_segment_tokens=8,
    )
    with pytest.raises(GradientAttributionError, match="non-finite"):
        collector.score_documents([ids], [segs], [fids])


def test_finite_but_saturating_scores_raise():
    """The OTHER half of the persist guard: a FINITE score >= FP16_SAFE_MAX
    (60000) still raises rather than silently saturating the fp16 overlay.
    test_non_finite_scores_raise covers the non-finite half; here W_dec stays
    finite (scaled by 1e7, not inf), so the isfinite clause is False and the
    raise is specifically the magnitude clause ``(score >= FP16_SAFE_MAX).any()``."""
    model = tiny_gemma2(softcap=True)
    sae = make_tiny_sae("standard", seed=1)
    with t.no_grad():
        sae.W_dec.mul_(1e7)  # finite but huge => finite score >> 60000
        sae.b_enc.fill_(1.0)
        sae.W_enc.zero_()  # every feature fires
    assert bool(t.isfinite(sae.W_dec).all())  # finite => the non-finite clause is False
    ids = seeded_ids(16, seed=2)
    segs = fixed_segments(16, 8)
    fids = [t.arange(sae.W_dec.shape[0], dtype=t.int64) for _ in segs]
    collector = GradientAttributionCollector(
        model, sae,
        layer=TINY_LAYER,
        mode=GradientAttributionMode.future_half_split,
        min_final_segment_tokens=8,
    )
    with pytest.raises(GradientAttributionError, match="refusing to persist"):
        collector.score_documents([ids], [segs], [fids])


class TestDecoderLinearityGateRejections:
    def test_rejects_finetuning_scaling_factor(self):
        sae = make_tiny_sae("standard")
        sae.cfg.finetuning_scaling_factor = True  # type: ignore[attr-defined]
        with pytest.raises(GradientAttributionError, match="finetuning_scaling_factor"):
            decoder_linearity_gate(sae)

    def test_rejects_unsupported_architecture(self):
        class FakeCfg:
            architecture = "topk"
            normalize_activations = "none"

        class FakeSAE:
            cfg = FakeCfg()

        with pytest.raises(GradientAttributionError, match="architectures"):
            decoder_linearity_gate(FakeSAE())

    def test_rejects_non_residual_hook(self):
        sae = make_tiny_sae("standard")
        sae.cfg.metadata.hook_name = "blocks.2.attn.hook_z"
        try:
            with pytest.raises(GradientAttributionError, match="residual-only"):
                decoder_linearity_gate(sae)
        finally:
            sae.cfg.metadata.hook_name = None

    def test_rejects_nonlinear_decode(self):
        sae = make_tiny_sae("standard")
        original = sae.decode
        sae.decode = lambda acts: original(acts).tanh()  # type: ignore[method-assign]
        try:
            with pytest.raises(GradientAttributionError, match="decoder-linearity"):
                decoder_linearity_gate(sae)
        finally:
            sae.decode = original  # type: ignore[method-assign]
