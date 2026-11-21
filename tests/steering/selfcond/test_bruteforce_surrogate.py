"""T-G6: exact fp64 oracle for the specified one-hop STE surrogate."""

from __future__ import annotations

import copy
from typing import cast

import pytest
import torch as t
import torch.nn.functional as F

import nanochat.flash_attention as _fa_mod
import steering.conditional.encoder_decoder.encoder as encoder_module
import steering.conditional.encoder_decoder.model as model_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    RolloutResult,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    freeze_for_head_only,
    rollout,
    selfcond_loss,
)
from steering.conditional.encoder_decoder.selfcond_training import (
    pool_fp_logits_per_segment as _pool_fp_logits_per_segment,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.refined_types import PositiveInt
from tests.padded_encoder_reference import pack_dense_encoder_output_reference
from tests.steering.selfcond.helpers import (
    SelfcondBatch,
    make_batch,
    perturb_residual_projections,
    tiny_head_model,
)


def pool_fp_logits_per_segment(
    fp_logits: t.Tensor,
    seg_map: t.Tensor,
    n_seg: int,
    mode: FeaturePredLossMode,
) -> t.Tensor:
    """Pass raw oracle values through the production beartype boundary."""
    return _pool_fp_logits_per_segment(
        fp_logits,
        seg_map,
        cast(PositiveInt, n_seg),
        mode,
    )


@pytest.fixture(autouse=True)
def _fp64_sdpa(monkeypatch: pytest.MonkeyPatch):
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float64)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float64)
    yield
    _fa_mod._override_impl = old


def _config(form: SteForm, scope: SelfCondTrainScope) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=scope,
        ste_form=form,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=3,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=0,
            max_k=2,
        ),
        seed=84,
    )


def _explicit_surrogate_encoder(
    model,
    result: RolloutResult,
    shifted_scores: t.Tensor,
    form: SteForm,
) -> tuple[t.Tensor, t.Tensor]:
    """Rebuild every STE feature/value mixture without calling encoder.forward."""
    encoder = model.encoder
    ids = result.conditioning.fids
    mags = result.conditioning.mags
    mask = result.conditioning.mask
    sign = result.conditioning.sign
    batch_size, n_seg, width = ids.shape
    padded_width = ((width + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
    padding = padded_width - width
    ids_flat = F.pad(ids, (0, padding)).reshape(batch_size * n_seg, padded_width)
    mags_flat = F.pad(mags, (0, padding)).reshape(batch_size * n_seg, padded_width)
    mask_padded = F.pad(mask, (0, padding))
    mask_flat = mask_padded.reshape(batch_size * n_seg, padded_width)
    sign_flat = F.pad(sign, (0, padding)).reshape(batch_size * n_seg, padded_width)
    scores_flat = shifted_scores.reshape(batch_size * n_seg, -1)

    feature_rows = encoder.feature_embed(ids_flat)
    x = feature_rows + encoder.magnitude_embed(mags_flat)
    score_delta = scores_flat - scores_flat.detach()
    selected_scores = scores_flat.gather(1, ids_flat)
    selected_delta = selected_scores - selected_scores.detach()
    if form is SteForm.DENSE:
        feature_mix = score_delta.to(feature_rows.dtype) @ encoder.feature_embed.weight
        x = x + sign_flat.unsqueeze(-1) * feature_mix.unsqueeze(1)
    else:
        x = (
            x
            + sign_flat.unsqueeze(-1)
            * selected_delta.to(feature_rows.dtype).unsqueeze(-1)
            * feature_rows
        )

    for key, block in enumerate(encoder.blocks):
        value_embed = encoder.value_embeds[str(key)] if str(key) in encoder.value_embeds else None
        assert value_embed is None or isinstance(value_embed, t.nn.Embedding)
        ve = value_embed(ids_flat) if value_embed is not None else None
        if ve is not None:
            assert value_embed is not None
            if form is SteForm.DENSE:
                value_mix = score_delta.to(ve.dtype) @ value_embed.weight
                ve = ve + sign_flat.unsqueeze(-1) * value_mix.unsqueeze(1)
            else:
                ve = ve + sign_flat.unsqueeze(-1) * selected_delta.to(ve.dtype).unsqueeze(-1) * ve
        x = block(x, ve=ve, mask=mask_flat)
    x = encoder_module.norm(x)
    return (
        x.reshape(batch_size, n_seg, padded_width, encoder.config.n_embd),
        mask_padded,
    )


def _explicit_surrogate_loss(
    model,
    batch: SelfcondBatch,
    result: RolloutResult,
    cfg: SelfCondTrainConfig,
) -> t.Tensor:
    _, fp_logits = model(
        batch.inputs,
        None,
        result.conditioning.fids,
        result.conditioning.mags,
        result.conditioning.mask,
        batch.seg_map,
        return_fp_logits=True,
    )
    pooled = pool_fp_logits_per_segment(
        fp_logits,
        batch.seg_map,
        batch.seg_doc_id.shape[1],
        cfg.pool_mode,
    )
    scores = t.sigmoid(pooled.float())
    shifted = t.zeros_like(scores)
    shifted[:, 1:] = scores[:, :-1]
    encoder_out, encoder_mask = _explicit_surrogate_encoder(
        model,
        result,
        shifted,
        cfg.ste_form,
    )
    packed_encoder_out = pack_dense_encoder_output_reference(
        encoder_out,
        encoder_mask,
    )
    return model(
        batch.inputs,
        batch.targets,
        seg_map=batch.seg_map,
        encoder_out=packed_encoder_out,
    )


@pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
@pytest.mark.parametrize("scope", [SelfCondTrainScope.HEAD_ONLY, SelfCondTrainScope.FULL])
def test_all_parameter_grads_match_explicit_one_hop_surrogate_exactly(
    form: SteForm,
    scope: SelfCondTrainScope,
) -> None:
    template = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=85).double().eval()
    batch = make_batch([[2]], T=8, F=4, n_sae=8, seed=86)
    cfg = _config(form, scope)
    result = rollout(
        template,
        batch,
        cfg,
        t.Generator().manual_seed(87),
        static_max_seg_size=None,
    )
    actual_model = copy.deepcopy(template)
    reference_model = copy.deepcopy(template)
    if scope is SelfCondTrainScope.HEAD_ONLY:
        freeze_for_head_only(actual_model)
        freeze_for_head_only(reference_model)

    actual_loss, _ = selfcond_loss(
        actual_model,
        batch,
        result,
        cfg,
        static_max_seg_size=None,
    )
    reference_loss = _explicit_surrogate_loss(reference_model, batch, result, cfg)
    actual_loss.backward()
    reference_loss.backward()

    assert t.equal(actual_loss, reference_loss)
    for (actual_name, actual_parameter), (reference_name, reference_parameter) in zip(
        actual_model.named_parameters(), reference_model.named_parameters(), strict=True
    ):
        assert actual_name == reference_name
        if actual_parameter.grad is None or reference_parameter.grad is None:
            assert actual_parameter.grad is None and reference_parameter.grad is None
        else:
            assert t.equal(actual_parameter.grad, reference_parameter.grad), actual_name


@pytest.mark.parametrize("form", [SteForm.DENSE, SteForm.SELECTED])
@pytest.mark.parametrize("scope", [SelfCondTrainScope.HEAD_ONLY, SelfCondTrainScope.FULL])
def test_surrogate_match_with_three_hop_chain_and_multiple_live_ve_blocks(
    form: SteForm,
    scope: SelfCondTrainScope,
) -> None:
    """T-G6 (strengthened): S>=3 + two live VE tables + B=2.

    Reviewer-consensus gaps this closes against the SAME independent surrogate:
    - A 2-slot batch cannot distinguish the specified one-hop estimator from a
      compounding mutant (the forbidden chain L -> slot2 -> s1 -> hidden1 ->
      slot1-STE -> s0 needs a third slot). The [[3]] document provides it.
    - A single-VE-block encoder with zero-init residual projections makes the
      value-embedding STE terms gradient-invisible. encoder_depth=3 yields VE
      tables at blocks 0 and 2, and perturbing the zero-init projections makes
      both paths carry real gradient, so dropping either VE mixture (or the
      per-block loop) now diverges from the surrogate.
    - B=2 with unequal per-row documents pins the batch flattening.
    """
    template = (
        tiny_head_model(
            FeaturePredLossMode.MAX_POOL,
            n_sae=8,
            seed=88,
            encoder_depth=3,
        )
        .double()
        .eval()
    )
    assert sorted(template.encoder.value_embeds.keys()) == ["0", "2"]
    perturb_residual_projections(template, seed=88)
    batch = make_batch([[3], [2, 1]], T=12, F=4, n_sae=8, seed=89)
    cfg = _config(form, scope)
    result = rollout(
        template,
        batch,
        cfg,
        t.Generator().manual_seed(90),
        static_max_seg_size=None,
    )
    assert int(result.predicted.sum()) >= 3  # the 3-segment doc predicts slots 1 and 2

    actual_model = copy.deepcopy(template)
    reference_model = copy.deepcopy(template)
    if scope is SelfCondTrainScope.HEAD_ONLY:
        freeze_for_head_only(actual_model)
        freeze_for_head_only(reference_model)

    actual_loss, _ = selfcond_loss(actual_model, batch, result, cfg, static_max_seg_size=None)
    reference_loss = _explicit_surrogate_loss(reference_model, batch, result, cfg)
    actual_loss.backward()
    reference_loss.backward()

    assert t.equal(actual_loss, reference_loss)
    for (actual_name, actual_parameter), (reference_name, reference_parameter) in zip(
        actual_model.named_parameters(), reference_model.named_parameters(), strict=True
    ):
        assert actual_name == reference_name
        if actual_parameter.grad is None or reference_parameter.grad is None:
            assert actual_parameter.grad is None and reference_parameter.grad is None
        else:
            assert t.equal(actual_parameter.grad, reference_parameter.grad), actual_name
    if scope is SelfCondTrainScope.FULL:
        for key in ("0", "2"):
            value_grad = actual_model.encoder.value_embeds[key].weight.grad
            assert value_grad is not None
            assert bool((value_grad != 0).any()), f"VE table {key} carries no gradient"  # ty: ignore[unresolved-attribute]
