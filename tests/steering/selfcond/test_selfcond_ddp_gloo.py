"""T-I5: real gloo/DistMuonAdamW self-conditioning smoke tests."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from nanochat.optim import DistMuonAdamW
from scripts.train.conditional_encdec_selfcond import (
    run_selfcond_train_step as _run_selfcond_train_step,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondTrainConfig,
    SelfCondTrainScope,
    build_head_only_optimizer,
    freeze_for_head_only,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from tests.padded_encoder_reference import use_test_owned_cpu_encoder_reference
from tests.steering.selfcond.helpers import make_batch, tiny_head_model


def run_selfcond_train_step(*args: Any, **kwargs: Any) -> Any:
    """Pass raw DDP smoke values through the production beartype boundary."""
    return _run_selfcond_train_step(*args, **kwargs)


def _config() -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=0,
            max_k=1,
        ),
        seed=700,
    )


def _head_snapshot(model) -> tuple[tuple[float, ...], ...]:
    assert model.feature_pred_head is not None
    return tuple(
        tuple(float(value) for value in parameter.detach().reshape(-1).tolist())
        for parameter in model.feature_pred_head.parameters()
    )


def _selfcond_gloo_target_inner(rank: int, world_size: int, scenario: str) -> dict:
    assert world_size == 2
    assert t.distributed.is_initialized()
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=701)
    freeze_for_head_only(model)
    optimizer = build_head_only_optimizer(
        model,
        unembedding_lr=0.004,
        adam_betas=(0.8, 0.95),
    )
    assert isinstance(optimizer, DistMuonAdamW)
    initial = _head_snapshot(model)

    if scenario == "live":
        layout = [[2]]
    elif scenario == "one_zero":
        layout = [[2]] if rank == 0 else [[1]]
    elif scenario == "both_zero":
        layout = [[1]]
    else:
        raise AssertionError(f"unknown test scenario {scenario!r}")
    batch = make_batch(layout, T=8, F=3, n_sae=8, seed=710 + rank)
    result = run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        _config(),
        global_step=0,
        ddp_rank=rank,
        num_iterations=1,
        grad_clip_max_norm=1.0,
        static_max_seg_size=None,
    )
    return {
        "optimizer": type(optimizer).__name__,
        "initial": initial,
        "final": _head_snapshot(model),
        "predicted_fraction": result.metrics.frac_predicted_slots,
    }


def _selfcond_gloo_target(rank: int, world_size: int, scenario: str) -> dict:
    """Install the test-owned CPU model boundary inside the spawned worker."""
    with use_test_owned_cpu_encoder_reference():
        return _selfcond_gloo_target_inner(rank, world_size, scenario)


@pytest.mark.parametrize(
    ("scenario", "expected_fractions", "expect_update"),
    [
        pytest.param("live", [0.5, 0.5], True, id="both-ranks-live"),
        pytest.param("one_zero", [0.5, 0.0], True, id="rank-one-zero-slots"),
        pytest.param("both_zero", [0.0, 0.0], False, id="both-ranks-zero-slots"),
    ],
)
def test_two_rank_step_reduces_head_grads_and_survives_zero_slot_r16_cases(
    gloo_multiprocess,
    scenario: str,
    expected_fractions: list[float],
    expect_update: bool,
) -> None:
    results = gloo_multiprocess(2, _selfcond_gloo_target, scenario)

    assert [result["optimizer"] for result in results] == [
        "DistMuonAdamW",
        "DistMuonAdamW",
    ]
    assert [result["predicted_fraction"] for result in results] == expected_fractions
    assert results[0]["initial"] == results[1]["initial"]
    assert results[0]["final"] == results[1]["final"]
    if expect_update:
        assert results[0]["final"] != results[0]["initial"]
    else:
        assert results[0]["final"] == results[0]["initial"]
