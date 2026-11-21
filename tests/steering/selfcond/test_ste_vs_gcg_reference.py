"""T-G3: dense self-conditioning gradients match a GCG-style hook oracle."""

from __future__ import annotations

from collections.abc import Callable

import pytest
import torch as t
import torch.nn.functional as F

import nanochat.flash_attention as _fa_mod
import steering.conditional.encoder_decoder.encoder as encoder_module
import steering.conditional.encoder_decoder.model as model_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder import FP8_ALIGNMENT
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm, SteInputs
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondTrainConfig,
    SelfCondTrainScope,
    rollout,
    selfcond_loss,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from tests.steering.selfcond.helpers import make_batch, tiny_head_model


@pytest.fixture(autouse=True)
def _fp64_sdpa(monkeypatch: pytest.MonkeyPatch):
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    monkeypatch.setattr(encoder_module, "COMPUTE_DTYPE", t.float64)
    monkeypatch.setattr(model_module, "COMPUTE_DTYPE", t.float64)
    yield
    _fa_mod._override_impl = old


def _config() -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.FULL,
        ste_form=SteForm.DENSE,
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
        seed=80,
    )


def _capture(
    store: dict[str, t.Tensor], name: str
) -> Callable[[t.nn.Module, tuple[t.Tensor, ...], t.Tensor], None]:
    def hook(_module, _inputs, output: t.Tensor) -> None:
        if output.requires_grad:
            output.retain_grad()
            store[name] = output

    return hook


def test_dense_scores_match_gcg_hook_reconstruction_exactly_fp64() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=81).double().eval()
    batch = make_batch([[2]], T=8, F=4, n_sae=8, seed=82)
    cfg = _config()
    result = rollout(
        model,
        batch,
        cfg,
        t.Generator().manual_seed(83),
        static_max_seg_size=None,
    )
    activations: dict[str, t.Tensor] = {}
    shifted_scores: list[t.Tensor] = []
    handles = [model.encoder.feature_embed.register_forward_hook(_capture(activations, "feature"))]
    for key, value_embed in model.encoder.value_embeds.items():
        handles.append(value_embed.register_forward_hook(_capture(activations, f"value_{key}")))

    def capture_ste(_module, _args, kwargs) -> None:
        ste = kwargs.get("ste")
        if isinstance(ste, SteInputs):
            ste.scores.retain_grad()
            shifted_scores.append(ste.scores)

    handles.append(model.encoder.register_forward_pre_hook(capture_ste, with_kwargs=True))
    try:
        loss, _ = selfcond_loss(model, batch, result, cfg, static_max_seg_size=None)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    assert len(shifted_scores) == 1
    scores = shifted_scores[0]
    assert scores.grad is not None
    width = result.conditioning.sign.shape[-1]
    padded_width = ((width + FP8_ALIGNMENT - 1) // FP8_ALIGNMENT) * FP8_ALIGNMENT
    sign = F.pad(result.conditioning.sign, (0, padded_width - width)).reshape(-1, padded_width)
    feature_activation = activations["feature"]
    assert feature_activation.grad is not None
    projected = (feature_activation.grad * sign.unsqueeze(-1)).sum(dim=1)
    projected = projected @ model.encoder.feature_embed.weight.detach().T
    for key, value_embed in model.encoder.value_embeds.items():
        assert isinstance(value_embed, t.nn.Embedding)
        activation = activations[f"value_{key}"]
        assert activation.grad is not None
        contribution = (activation.grad * sign.unsqueeze(-1)).sum(dim=1)
        projected = projected + contribution @ value_embed.weight.detach().T
    expected = projected.to(dtype=scores.dtype).reshape_as(scores)
    assert t.equal(scores.grad, expected)
