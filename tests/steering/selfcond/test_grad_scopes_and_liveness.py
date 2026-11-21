"""T-G4/T-G7: scope freezing, R16 zero grads, and pass-1 liveness."""

from __future__ import annotations

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm, SteInputs
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondTrainConfig,
    SelfCondTrainScope,
    freeze_for_head_only,
    materialize_head_zero_grads,
    rollout,
    selfcond_loss,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from tests.steering.selfcond.helpers import (
    make_batch,
    perturb_residual_projections,
    tiny_head_model,
)


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _config(scope: SelfCondTrainScope) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=scope,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=2,
            max_k=4,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=3,
        ),
        seed=71,
    )


def _loss_for_layout(model, layout: list[list[int]], scope: SelfCondTrainScope):
    batch = make_batch(layout, T=12, F=4, n_sae=16, seed=72)
    cfg = _config(scope)
    result = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(73),
        static_max_seg_size=None,
    )
    loss, _ = selfcond_loss(model, batch, result, cfg, static_max_seg_size=None)
    return loss


def test_head_only_scope_and_r16_zero_slot_states() -> None:
    """T-G4: frozen params stay empty; R16 changes None only to exact zeros."""
    predicted_model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=74).eval()
    freeze_for_head_only(predicted_model)
    _loss_for_layout(predicted_model, [[3], [2, 1]], SelfCondTrainScope.HEAD_ONLY).backward()
    assert predicted_model.feature_pred_head is not None
    for name, parameter in predicted_model.named_parameters():
        if name.startswith("feature_pred_head."):
            assert parameter.grad is not None
            assert bool((parameter.grad != 0).any())
        else:
            assert parameter.grad is None or not bool(parameter.grad.any())

    zero_model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=75).eval()
    freeze_for_head_only(zero_model)
    _loss_for_layout(zero_model, [[1], [1]], SelfCondTrainScope.HEAD_ONLY).backward()
    assert zero_model.feature_pred_head is not None
    assert all(parameter.grad is None for parameter in zero_model.feature_pred_head.parameters())
    materialize_head_zero_grads(zero_model)
    for parameter in zero_model.feature_pred_head.parameters():
        assert parameter.grad is not None
        assert t.equal(parameter.grad, t.zeros_like(parameter))


def test_full_scope_gives_every_parameter_a_gradient() -> None:
    """T-G4: the multi-document full-scope loss reaches every parameter."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=76).eval()
    # Move off the zero-init degenerate point: with the residual projections at
    # zero, upstream weights (c_qkv/c_fc/value_embeds) route their gradient
    # through a zero projection and are exactly zero even though they ARE
    # connected. Perturbing makes the whole surface live, so the strictly
    # stronger "every parameter receives a NON-ZERO gradient" bar (matching the
    # head-only sibling above) is meaningful rather than a false failure.
    perturb_residual_projections(model, seed=76)
    _loss_for_layout(model, [[3], [2, 1]], SelfCondTrainScope.FULL).backward()
    missing = [name for name, parameter in model.named_parameters() if parameter.grad is None]
    assert missing == []
    dead = [
        name
        for name, parameter in model.named_parameters()
        if not bool((parameter.grad != 0).any())  # ty: ignore[unresolved-attribute]
    ]
    assert dead == []


def test_pass_one_scores_reach_head_weight_before_backward() -> None:
    """T-G7: the scores handed to the STE still own the pass-1 graph."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=77).eval()
    batch = make_batch([[2], [2]], T=10, F=4, n_sae=16, seed=78)
    cfg = _config(SelfCondTrainScope.FULL)
    result = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(79),
        static_max_seg_size=None,
    )
    captured: list[t.Tensor] = []

    def capture_ste(_module, _args, kwargs) -> None:
        ste = kwargs.get("ste")
        if isinstance(ste, SteInputs):
            captured.append(ste.scores)

    handle = model.encoder.register_forward_pre_hook(capture_ste, with_kwargs=True)
    try:
        selfcond_loss(model, batch, result, cfg, static_max_seg_size=None)
    finally:
        handle.remove()

    assert len(captured) == 1
    scores = captured[0]
    assert scores.requires_grad is True
    assert model.feature_pred_head is not None
    gradient = t.autograd.grad(
        scores.sum(),
        model.feature_pred_head.weight,
        retain_graph=True,
    )[0]
    assert gradient is not None
    assert bool((gradient != 0).any())
