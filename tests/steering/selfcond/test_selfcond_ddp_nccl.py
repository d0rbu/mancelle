"""Opt-in production-NCCL self-conditioning optimizer-step coverage."""

from __future__ import annotations

import os
from typing import Any

import pytest
import torch as t
import torch.distributed as dist

import nanochat.flash_attention as flash_attention_module
from nanochat.flash_attention import FlashAttnImpl
from nanochat.optim import DistMuonAdamW
from scripts.train.conditional_encdec_selfcond import run_selfcond_train_step
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
from steering.refined_types import PositiveFloat, PositiveInt
from tests.nccl_multiprocess_helpers import run_nccl_multiprocess
from tests.steering.selfcond.helpers import (
    SelfcondBatch,
    make_batch,
    perturb_residual_projections,
    tiny_head_model,
)


def _config(scope: SelfCondTrainScope, ste_form: SteForm) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=scope,
        ste_form=ste_form,
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
        seed=1700,
    )


def _to_device(batch: SelfcondBatch, device: t.device) -> SelfcondBatch:
    return SelfcondBatch(
        inputs=batch.inputs.to(device),
        targets=batch.targets.to(device),
        enc_fids=batch.enc_fids.to(device),
        enc_mags=batch.enc_mags.to(device),
        enc_mask=batch.enc_mask.to(device),
        enc_pcts=batch.enc_pcts.to(device),
        seg_map=batch.seg_map.to(device),
        seg_doc_id=batch.seg_doc_id.to(device),
    )


def _assert_model_identical_across_ranks(model: t.nn.Module) -> None:
    """Byte-compare every parameter to rank zero after production collectives."""
    for name, parameter in model.named_parameters():
        rank_zero = parameter.detach().clone()
        dist.broadcast(rank_zero, src=0)
        if parameter.dtype != rank_zero.dtype or not t.equal(parameter, rank_zero):
            raise AssertionError(f"model parameter differs across NCCL ranks: {name}")


def _nccl_selfcond_target(
    rank: int,
    world_size: int,
    scenario: str,
) -> dict[str, Any]:
    assert world_size == 2
    assert dist.is_initialized()
    device = t.device("cuda", rank)
    flash_attention_module._override_impl = FlashAttnImpl.SDPA
    t._dynamo.config.cache_size_limit = 64
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=1701)
    perturb_residual_projections(model, seed=1702)
    model.to(device)

    if scenario == "head_only_dense_zero_rank":
        scope = SelfCondTrainScope.HEAD_ONLY
        ste_form = SteForm.DENSE
        freeze_for_head_only(model)
        optimizer = build_head_only_optimizer(
            model,
            unembedding_lr=0.004,
            adam_betas=(0.8, 0.95),
        )
        layout = [[2]] if rank == 0 else [[1]]
        expected_fraction = 0.5 if rank == 0 else 0.0
    elif scenario == "full_selected_different_shapes":
        scope = SelfCondTrainScope.FULL
        ste_form = SteForm.SELECTED
        optimizer = model.setup_optimizer(
            unembedding_lr=0.004,
            embedding_lr=0.02,
            matrix_lr=0.002,
            scalar_lr=0.05,
            weight_decay=0.0,
            adam_betas=(0.8, 0.95),
            encoder_embedding_lr=0.02,
            encoder_matrix_lr=0.002,
            xattn_matrix_lr=0.002,
        )
        layout = [[3]] if rank == 0 else [[2]]
        expected_fraction = 2 / 3 if rank == 0 else 0.5
    else:
        raise AssertionError(f"unknown NCCL self-conditioning scenario: {scenario!r}")

    assert isinstance(optimizer, DistMuonAdamW)
    initial = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    batch = _to_device(
        make_batch(layout, T=8, F=3, n_sae=8, seed=1710 + rank),
        device,
    )
    result = run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        _config(scope, ste_form),
        global_step=0,
        ddp_rank=rank,
        num_iterations=PositiveInt.parse(1),
        grad_clip_max_norm=PositiveFloat.parse(1.0),
        static_max_seg_size=None,
    )
    _assert_model_identical_across_ranks(model)
    updated = sorted(
        name
        for name, parameter in model.named_parameters()
        if not t.equal(parameter.detach(), initial[name])
    )
    if not updated:
        raise AssertionError("the production NCCL optimizer step updated no parameters")
    return {
        "optimizer": type(optimizer).__name__,
        "predicted_fraction": result.metrics.frac_predicted_slots,
        "expected_fraction": expected_fraction,
        "updated": updated,
    }


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_DDP_SMOKE") != "1"
    or t.cuda.device_count() < 2
    or not dist.is_nccl_available(),
    reason="requires RUN_DDP_SMOKE=1, NCCL, and at least two local CUDA devices",
)
@pytest.mark.parametrize(
    "scenario",
    [
        "head_only_dense_zero_rank",
        "full_selected_different_shapes",
    ],
)
def test_two_rank_nccl_selfcond_step_is_collective_safe_and_rank_identical(
    scenario: str,
) -> None:
    results = run_nccl_multiprocess(2, _nccl_selfcond_target, scenario)

    assert [result["optimizer"] for result in results] == [
        "DistMuonAdamW",
        "DistMuonAdamW",
    ]
    assert [result["predicted_fraction"] for result in results] == [
        result["expected_fraction"] for result in results
    ]
    assert results[0]["updated"] == results[1]["updated"]
