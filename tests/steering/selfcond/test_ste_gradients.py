"""T-G1/T-G2/T-G5: exact STE gradient contracts on a tiny fp64 encoder."""

from __future__ import annotations

import copy
from collections.abc import Callable

import pytest
import torch as t
import torch.nn.functional as F

import nanochat.flash_attention as _fa_mod
import steering.conditional.encoder_decoder.encoder as encoder_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.encoder import (
    FeatureEncoder,
    FeatureEncoderConfig,
    SteForm,
    SteInputs,
)
from tests.padded_encoder_reference import pack_dense_encoder_output_reference


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch):
    """Force SDPA and fp64 activations for exact CPU backward assertions."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float64)
    yield
    _fa_mod._override_impl = old


def _encoder(seed: int = 29) -> FeatureEncoder:
    cfg = FeatureEncoderConfig(
        n_sae_features=8,
        num_buckets=2,
        n_embd=4,
        n_layer=1,
        n_head=1,
        n_kv_head=1,
        head_dim=4,
        ve_gate_channels=4,
    )
    enc = FeatureEncoder(cfg).double()
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for parameter in enc.parameters():
            parameter.copy_(t.randn(parameter.shape, generator=generator, dtype=t.float64) * 0.2)
    return enc


def _batch() -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    feature_ids = t.tensor([[[1, 5]]], dtype=t.int64)
    magnitudes = t.tensor([[[0, 1]]], dtype=t.int64)
    mask = t.ones_like(feature_ids, dtype=t.bool)
    sign = t.tensor([[[1.0, -1.0]]], dtype=t.float64)
    return feature_ids, magnitudes, mask, sign


def _loss_weight(output: t.Tensor) -> t.Tensor:
    shape = (1,) * (output.ndim - 1) + (4,)
    return t.tensor([0.7, -0.2, 0.4, 1.1], dtype=t.float64).view(shape).expand_as(output)


def _value_embed(encoder: FeatureEncoder, key: str) -> t.nn.Embedding:
    module = encoder.value_embeds[key]
    assert isinstance(module, t.nn.Embedding)
    return module


def _capture_activation(
    store: dict[str, t.Tensor], name: str
) -> Callable[[t.nn.Module, tuple[t.Tensor, ...], t.Tensor], None]:
    def hook(
        _module: t.nn.Module,
        _inputs: tuple[t.Tensor, ...],
        output: t.Tensor,
    ) -> None:
        output.retain_grad()
        store[name] = output

    return hook


def _run_with_hooks(
    encoder: FeatureEncoder,
    form: SteForm,
    sign: t.Tensor,
) -> tuple[t.Tensor, dict[str, t.Tensor], t.Tensor]:
    feature_ids, magnitudes, mask, _ = _batch()
    generator = t.Generator().manual_seed(101)
    scores = t.randn(1, 1, 8, generator=generator, dtype=t.float64).requires_grad_()
    activations: dict[str, t.Tensor] = {}
    handles = [
        encoder.feature_embed.register_forward_hook(_capture_activation(activations, "feature")),
        _value_embed(encoder, "0").register_forward_hook(
            _capture_activation(activations, "value_0")
        ),
    ]
    try:
        output = encoder(
            feature_ids,
            magnitudes,
            mask,
            ste=SteInputs(scores=scores, sign=sign, form=form),
        ).values
        loss = (output * _loss_weight(output)).sum()
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()
    assert scores.grad is not None
    assert activations["feature"].grad is not None
    assert activations["value_0"].grad is not None
    return scores, activations, loss.detach()


def _padded_sign(sign: t.Tensor) -> t.Tensor:
    nf = sign.shape[-1]
    padded = ((nf + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
    return F.pad(sign, (0, padded - nf)).reshape(-1, padded)


def test_dense_scores_gradient_matches_hand_projection_exactly() -> None:
    """T-G1: dense dL/ds equals the feature plus every-VE projection."""
    encoder = _encoder()
    _, _, _, sign = _batch()
    scores, activations, _ = _run_with_hooks(encoder, SteForm.DENSE, sign)
    sign_flat = _padded_sign(sign)
    feature_grad = activations["feature"].grad
    value_grad = activations["value_0"].grad
    assert feature_grad is not None
    assert value_grad is not None

    expected_feature = (feature_grad * sign_flat.unsqueeze(-1)).sum(dim=1)
    expected_feature = expected_feature @ encoder.feature_embed.weight.detach().T
    expected_value = (value_grad * sign_flat.unsqueeze(-1)).sum(dim=1)
    expected_value = expected_value @ _value_embed(encoder, "0").weight.detach().T
    expected = (expected_feature + expected_value).reshape_as(scores)

    assert scores.grad is not None
    assert t.equal(scores.grad, expected)


def test_selected_scores_gradient_matches_gathered_row_dots_exactly() -> None:
    """T-G2a: selected dL/ds is nonzero only on gathered feature ids."""
    encoder = _encoder()
    feature_ids, _, _, sign = _batch()
    scores, activations, _ = _run_with_hooks(encoder, SteForm.SELECTED, sign)
    feature_grad = activations["feature"].grad
    value_grad = activations["value_0"].grad
    assert feature_grad is not None
    assert value_grad is not None

    expected = t.zeros_like(scores).reshape(-1, 8)
    for column, feature_id in enumerate(feature_ids.reshape(-1).tolist()):
        contribution = sign.reshape(-1)[column] * (
            (
                feature_grad.reshape(-1, feature_grad.shape[-1])[column]
                * encoder.feature_embed.weight.detach()[feature_id]
            ).sum()
            + (
                value_grad.reshape(-1, value_grad.shape[-1])[column]
                * _value_embed(encoder, "0").weight.detach()[feature_id]
            ).sum()
        )
        expected[0, feature_id] += contribution

    assert scores.grad is not None
    assert t.equal(scores.grad, expected.reshape_as(scores))


@pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
def test_negative_sign_exactly_negates_score_gradient(form: SteForm) -> None:
    """T-G2b/R13/D5: a negative slot implements the derivative of 1-s."""
    base = _encoder()
    positive = copy.deepcopy(base)
    negative = copy.deepcopy(base)
    _, _, _, sign = _batch()

    positive_scores, _, _ = _run_with_hooks(positive, form, sign.abs())
    negative_scores, _, _ = _run_with_hooks(negative, form, -sign.abs())

    assert positive_scores.grad is not None
    assert negative_scores.grad is not None
    assert t.equal(negative_scores.grad, -positive_scores.grad)


@pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
def test_ste_does_not_contaminate_feature_indexed_parameter_gradients(
    form: SteForm,
) -> None:
    """T-G2c/R11: detached STE tables retain only ordinary lookup gradients."""
    baseline = _encoder()
    injected = copy.deepcopy(baseline)
    feature_ids, magnitudes, mask, sign = _batch()

    baseline_output = baseline(feature_ids, magnitudes, mask).values
    (baseline_output * _loss_weight(baseline_output)).sum().backward()
    _run_with_hooks(injected, form, sign)

    assert baseline.feature_embed.weight.grad is not None
    assert injected.feature_embed.weight.grad is not None
    assert t.equal(
        baseline.feature_embed.weight.grad,
        injected.feature_embed.weight.grad,
    )
    for key in baseline.value_embeds:
        baseline_grad = _value_embed(baseline, key).weight.grad
        injected_grad = _value_embed(injected, key).weight.grad
        assert baseline_grad is not None
        assert injected_grad is not None
        assert t.equal(baseline_grad, injected_grad)


def _local_surrogate_loss(
    encoder: FeatureEncoder,
    scores: t.Tensor,
    base_scores: t.Tensor,
    sign: t.Tensor,
    form: SteForm,
) -> t.Tensor:
    """Evaluate the STE's local surrogate with its detach baseline held fixed."""
    feature_ids, magnitudes, mask, _ = _batch()
    batch, n_seg, nf = feature_ids.shape
    padded = ((nf + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
    pad_nf = padded - nf
    ids_flat = F.pad(feature_ids, (0, pad_nf)).reshape(batch * n_seg, padded)
    mags_flat = F.pad(magnitudes, (0, pad_nf)).reshape(batch * n_seg, padded)
    mask_flat = F.pad(mask, (0, pad_nf)).reshape(batch * n_seg, padded)
    ids_flat = encoder._to_compact(ids_flat, mask_flat)
    sign_flat = F.pad(sign, (0, pad_nf)).reshape(batch * n_seg, padded)
    scores_flat = scores.reshape(batch * n_seg, -1)
    base_flat = base_scores.reshape(batch * n_seg, -1)

    x = encoder.feature_embed(ids_flat) + encoder.magnitude_embed(mags_flat)
    if form is SteForm.DENSE:
        current = scores_flat @ encoder.feature_embed.weight.detach()
        baseline = base_flat @ encoder.feature_embed.weight.detach()
        x = x + sign_flat.unsqueeze(-1) * (current - baseline).unsqueeze(1)
    else:
        current = scores_flat.gather(1, ids_flat).unsqueeze(-1)
        baseline = base_flat.gather(1, ids_flat).unsqueeze(-1)
        rows = encoder.feature_embed.weight.detach()[ids_flat]
        x = x + sign_flat.unsqueeze(-1) * (current - baseline) * rows

    for index, block in enumerate(encoder.blocks):
        value_embed = _value_embed(encoder, str(index))
        value = value_embed(ids_flat)
        if form is SteForm.DENSE:
            current = scores_flat @ value_embed.weight.detach()
            baseline = base_flat @ value_embed.weight.detach()
            value = value + sign_flat.unsqueeze(-1) * (current - baseline).unsqueeze(1)
        else:
            current = scores_flat.gather(1, ids_flat).unsqueeze(-1)
            baseline = base_flat.gather(1, ids_flat).unsqueeze(-1)
            rows = value_embed.weight.detach()[ids_flat]
            value = value + sign_flat.unsqueeze(-1) * (current - baseline) * rows
        x = block(x, ve=value, mask=mask_flat)

    dense_output = encoder_module.norm(x).reshape(batch, n_seg, padded, -1)
    packed_output = pack_dense_encoder_output_reference(
        dense_output,
        mask_flat.view(batch, n_seg, padded),
    ).values
    return (packed_output * _loss_weight(packed_output)).sum()


@pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
def test_scores_gradient_matches_local_surrogate_finite_difference(
    form: SteForm,
) -> None:
    """T-G5: central difference checks the STE tangent with a fixed detach point."""
    encoder = _encoder()
    _, _, _, sign = _batch()
    scores, _, _ = _run_with_hooks(encoder, form, sign)
    assert scores.grad is not None
    base_scores = scores.detach().clone()
    coordinate = (0, 0, 1)
    epsilon = 1e-6

    plus = base_scores.clone()
    minus = base_scores.clone()
    plus[coordinate] += epsilon
    minus[coordinate] -= epsilon
    with t.no_grad():
        finite_difference = (
            _local_surrogate_loss(encoder, plus, base_scores, sign, form)
            - _local_surrogate_loss(encoder, minus, base_scores, sign, form)
        ) / (2 * epsilon)

    t.testing.assert_close(scores.grad[coordinate], finite_difference, rtol=1e-5, atol=1e-7)


def _deep_encoder(seed: int = 41) -> FeatureEncoder:
    """A 3-layer encoder: has_ve blocks at i=0 and i=2 (two VE tables).

    Every parameter (including the residual projections that init_weights zeroes)
    is randomized, so both value-embedding paths carry nonzero gradient — the
    degenerate zero-c_proj point would make the VE STE terms invisible.
    """
    cfg = FeatureEncoderConfig(
        n_sae_features=8,
        num_buckets=2,
        n_embd=4,
        n_layer=3,
        n_head=1,
        n_kv_head=1,
        head_dim=4,
        ve_gate_channels=4,
    )
    enc = FeatureEncoder(cfg).double()
    generator = t.Generator().manual_seed(seed)
    with t.no_grad():
        for parameter in enc.parameters():
            parameter.copy_(t.randn(parameter.shape, generator=generator, dtype=t.float64) * 0.2)
    return enc


def test_dense_gradient_sums_every_ve_block_and_pins_bs_flattening() -> None:
    """T-G1 (strengthened): multi-VE-block accumulation + (B,S)->B*S mapping.

    Reviewer-consensus gaps: (1) with a single has_ve block, dropping the
    "sum over EVERY value-embedding table" is invisible; (2) with B=S=1, a
    transposed (B,S) flatten produces identical gradients. A 3-layer encoder
    (VE tables at blocks 0 and 2) with B=2, S=2 and per-(b,s)-distinct scores
    and signs kills both mutants against the independent hand projection.
    """
    encoder = _deep_encoder()
    assert sorted(encoder.value_embeds.keys()) == ["0", "2"]

    B, S, F_cols, V = 2, 2, 2, 8
    feature_ids = t.tensor(
        [[[1, 5], [2, 7]], [[0, 3], [6, 4]]],
        dtype=t.int64,
    )
    magnitudes = t.tensor(
        [[[0, 1], [1, 0]], [[0, 0], [1, 1]]],
        dtype=t.int64,
    )
    mask = t.ones(B, S, F_cols, dtype=t.bool)
    sign = t.tensor(
        [[[1.0, -1.0], [-1.0, 0.0]], [[0.0, 1.0], [1.0, -1.0]]],
        dtype=t.float64,
    )
    generator = t.Generator().manual_seed(202)
    scores = t.randn(B, S, V, generator=generator, dtype=t.float64).requires_grad_()

    activations: dict[str, t.Tensor] = {}
    handles = [
        encoder.feature_embed.register_forward_hook(_capture_activation(activations, "feature")),
        _value_embed(encoder, "0").register_forward_hook(
            _capture_activation(activations, "value_0")
        ),
        _value_embed(encoder, "2").register_forward_hook(
            _capture_activation(activations, "value_2")
        ),
    ]
    try:
        output = encoder(
            feature_ids,
            magnitudes,
            mask,
            ste=SteInputs(scores=scores, sign=sign, form=SteForm.DENSE),
        ).values
        loss_weight = t.randn(output.shape, generator=generator, dtype=t.float64)
        (output * loss_weight).sum().backward()
    finally:
        for handle in handles:
            handle.remove()

    sign_flat = _padded_sign(sign)  # (B*S, Nf_padded), rows ordered r = b*S + s
    expected = t.zeros(B * S, V, dtype=t.float64)
    for name, table in [
        ("feature", encoder.feature_embed.weight),
        ("value_0", _value_embed(encoder, "0").weight),
        ("value_2", _value_embed(encoder, "2").weight),
    ]:
        activation_grad = activations[name].grad
        assert activation_grad is not None
        summed = (activation_grad * sign_flat.unsqueeze(-1)).sum(dim=1)  # (B*S, d)
        expected += summed @ table.detach().T

    assert scores.grad is not None
    # Not t.equal: autograd accumulates the three per-table contributions into
    # scores.grad in reverse-forward order while the reference sums
    # feature+ve0+ve2; IEEE addition is non-associative, and the measured
    # difference is 1-2 ulps (3.6e-15 abs / 3.5e-16 rel). Any real mutant this
    # test exists to kill — a dropped VE table, a transposed (B,S) flatten, a
    # mis-mapped slot — differs at O(1) relative, thirteen orders of magnitude
    # above this bound.
    t.testing.assert_close(scores.grad, expected.reshape(B, S, V), rtol=1e-13, atol=1e-13)
    # Every (b, s) row carries a distinct nonzero gradient, so a transposed
    # flatten (S-major instead of B-major) could not silently pass above.
    per_row_norms = scores.grad.reshape(B * S, V).norm(dim=1)
    assert bool((per_row_norms > 0).all())
