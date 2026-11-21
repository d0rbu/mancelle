"""Real two-rank gloo coverage for SFT-mode self-conditioning."""

from __future__ import annotations

import math
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from scripts.train.conditional_encdec_selfcond import (
    evaluate_selfcond_bpb_sft,
    run_selfcond_train_step,
)
from scripts.train.conditional_encdec_sft import SFTState, _SFTValidationLoader
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.loss_eval import accumulate_bpb_nats_bytes
from steering.conditional.encoder_decoder.selfcond_training import (
    VAL_ROLLOUT_STEP_SENTINEL,
    SelfCondBatch,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    build_head_only_optimizer,
    freeze_for_head_only,
    rollout,
    selfcond_step_seed,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.refined_types import PositiveFloat, PositiveInt
from tests.encdec_test_helpers import VOCAB_SIZE
from tests.padded_encoder_reference import use_test_owned_cpu_encoder_reference
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch, tiny_head_model


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


def _as_sft_tuple(batch: SelfcondBatch) -> tuple[t.Tensor, ...] | tuple[Any, ...]:
    """Convert the shared hand-built batch to the production eight-item SFT order."""
    return (
        batch.inputs.to(t.int32),
        batch.targets,
        batch.enc_fids,
        batch.enc_mags,
        batch.enc_mask,
        batch.seg_map,
        None,
        batch.seg_doc_id,
    )


def _with_target_only_slot(batch: SelfcondBatch) -> tuple[Any, ...]:
    """Append one real-doc encoder slot with no decoder-token coverage."""
    assert batch.seg_doc_id.shape == (1, 2)
    assert batch.seg_doc_id.tolist() == [[0, 0]]
    batch_size, _n_seg, feature_width = batch.enc_fids.shape
    target_doc_id = batch.seg_doc_id[:, -1:]
    raw = _as_sft_tuple(batch)
    return (
        raw[0],
        raw[1],
        t.cat((batch.enc_fids, t.zeros(batch_size, 1, feature_width, dtype=t.int64)), dim=1),
        t.cat((batch.enc_mags, t.zeros(batch_size, 1, feature_width, dtype=t.int64)), dim=1),
        t.cat((batch.enc_mask, t.zeros(batch_size, 1, feature_width, dtype=t.bool)), dim=1),
        batch.seg_map,
        None,
        t.cat((batch.seg_doc_id, target_doc_id), dim=1),
    )


def _head_snapshot(model: Any) -> tuple[tuple[float, ...], ...]:
    head = model.feature_pred_head
    assert head is not None
    return tuple(
        tuple(float(value) for value in parameter.detach().reshape(-1).tolist())
        for parameter in head.parameters()
    )


def _sft_train_step_worker_inner(rank: int, world_size: int, scenario: str) -> dict[str, Any]:
    """Run one SFT-adapted optimizer step on differently shaped rank-local rows."""
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
    initial = _head_snapshot(model)

    if scenario == "different_n_seg":
        layout = [[4]] if rank == 0 else [[2]]
        raw_batch = _as_sft_tuple(make_batch(layout, T=8, F=3, n_sae=8, seed=710 + rank))
        expected_fraction = 0.75 if rank == 0 else 0.5
    elif scenario == "target_only":
        if rank == 0:
            raw_batch = _as_sft_tuple(make_batch([[4]], T=8, F=3, n_sae=8, seed=720))
            expected_fraction = 0.75
        else:
            raw_batch = _with_target_only_slot(make_batch([[2]], T=8, F=3, n_sae=8, seed=721))
            assert raw_batch[5].max().item() == 1
            assert raw_batch[7].shape[1] == 3
            expected_fraction = 0.5
    elif scenario == "rank_one_zero_live":
        layout = [[2]] if rank == 0 else [[1]]
        raw_batch = _as_sft_tuple(make_batch(layout, T=8, F=3, n_sae=8, seed=730 + rank))
        expected_fraction = 0.5 if rank == 0 else 0.0
    else:
        raise AssertionError(f"unknown SFT DDP scenario {scenario!r}")

    batch = SelfCondBatch.from_sft(raw_batch)
    assert batch.seg_doc_id.shape[1] == raw_batch[2].shape[1]
    result = run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        _config(),
        global_step=0,
        ddp_rank=rank,
        num_iterations=PositiveInt.parse(1),
        grad_clip_max_norm=PositiveFloat.parse(1.0),
        static_max_seg_size=None,
    )
    return {
        "initial": initial,
        "final": _head_snapshot(model),
        "expected_fraction": expected_fraction,
        "predicted_fraction": result.metrics.frac_predicted_slots,
        "n_seg": batch.seg_doc_id.shape[1],
    }


def _sft_train_step_worker(rank: int, world_size: int, scenario: str) -> dict[str, Any]:
    """Install the test-owned CPU model boundary inside the spawned worker."""
    with use_test_owned_cpu_encoder_reference():
        return _sft_train_step_worker_inner(rank, world_size, scenario)


@pytest.mark.parametrize(
    "scenario",
    ["different_n_seg", "target_only", "rank_one_zero_live"],
)
def test_sft_step_reduces_head_grads_across_different_rank_local_shapes(
    gloo_multiprocess,
    scenario: str,
) -> None:
    results = gloo_multiprocess(2, _sft_train_step_worker, scenario)

    assert results[0]["initial"] == results[1]["initial"]
    assert results[0]["final"] == results[1]["final"]
    assert results[0]["final"] != results[0]["initial"]
    assert [result["predicted_fraction"] for result in results] == [
        result["expected_fraction"] for result in results
    ]
    assert results[0]["n_seg"] != results[1]["n_seg"]


def _sft_eval_lockstep_worker_inner(rank: int, world_size: int) -> dict[str, Any]:
    """Evaluate rank-local real tails followed by production-faithful padding."""
    assert world_size == 2
    assert t.distributed.is_initialized()
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=741)
    model.eval()
    state = SFTState()
    local_batch_limit = 1 if rank == 0 else 4
    yielded = 0
    real_yielded = 0
    token_bytes = t.ones(VOCAB_SIZE, dtype=t.int64)

    def eval_batch(batch_index: int) -> tuple[Any, ...]:
        raw = _as_sft_tuple(
            make_batch(
                [[2]],
                T=8,
                F=3,
                n_sae=8,
                seed=750 + 100 * rank + batch_index,
            )
        )
        if rank == 0:
            return raw
        targets = raw[1].clone()
        targets[:, -1] = -1
        return raw[:1] + (targets,) + raw[2:]

    manual_rng = t.Generator().manual_seed(
        selfcond_step_seed(_config().seed, rank, VAL_ROLLOUT_STEP_SENTINEL)
    )
    local_nats = t.tensor(0.0, dtype=t.float32)
    local_bytes = t.tensor(0, dtype=t.int64)
    with t.no_grad():
        for batch_index in range(local_batch_limit):
            manual_batch = SelfCondBatch.from_sft(eval_batch(batch_index))
            manual_rollout = rollout(model, manual_batch, _config(), manual_rng, None)
            manual_loss = model(
                manual_batch.inputs,
                manual_batch.targets,
                manual_rollout.conditioning.fids,
                manual_rollout.conditioning.mags,
                manual_rollout.conditioning.mask,
                manual_batch.seg_map,
                loss_reduction="none",
            ).reshape(-1)
            batch_nats, batch_bytes = accumulate_bpb_nats_bytes(
                manual_loss,
                manual_batch.targets.reshape(-1),
                token_bytes,
            )
            local_nats += batch_nats
            local_bytes += batch_bytes

    def batches():
        nonlocal real_yielded, yielded
        for batch_index in range(local_batch_limit):
            yielded += 1
            real_yielded += 1
            state.consumed += 1
            state.exhausted = real_yielded == local_batch_limit
            yield eval_batch(batch_index)
        while True:
            # Production's single-pass SFT generator remains iterable after
            # local exhaustion. Its targets are fully masked so slower ranks'
            # real tail batches can finish without changing pooled BPB.
            yielded += 1
            padded = eval_batch(local_batch_limit - 1)
            targets = t.full_like(padded[1], -1)
            yield padded[:1] + (targets,) + padded[2:]

    bpb = evaluate_selfcond_bpb_sft(
        model,
        _SFTValidationLoader(batches(), state),
        eval_steps=6,
        token_bytes=token_bytes,
        cfg=_config(),
        ddp_rank=rank,
        device="cpu",
    )
    return {
        "bpb": bpb,
        "exhausted": state.exhausted,
        "local_batch_limit": local_batch_limit,
        "yielded": yielded,
        "real_yielded": real_yielded,
        "local_nats": float(local_nats.item()),
        "local_bytes": int(local_bytes.item()),
    }


def _sft_eval_lockstep_worker(rank: int, world_size: int) -> dict[str, Any]:
    """Install the test-owned CPU model boundary inside the spawned worker."""
    with use_test_owned_cpu_encoder_reference():
        return _sft_eval_lockstep_worker_inner(rank, world_size)


def test_sft_eval_waits_for_every_rank_without_dropping_real_tail_batches(
    gloo_multiprocess,
) -> None:
    results = gloo_multiprocess(2, _sft_eval_lockstep_worker)

    assert [result["local_batch_limit"] for result in results] == [1, 4]
    assert [result["real_yielded"] for result in results] == [1, 4]
    assert [result["yielded"] for result in results] == [4, 4]
    assert [result["exhausted"] for result in results] == [True, True]
    assert results[0]["bpb"] == results[1]["bpb"]
    assert results[0]["local_nats"] != results[1]["local_nats"]
    assert results[0]["local_bytes"] != results[1]["local_bytes"]
    pooled_nats = float(
        t.tensor(
            [result["local_nats"] for result in results],
            dtype=t.float32,
        )
        .sum()
        .item()
    )
    pooled_bytes = sum(result["local_bytes"] for result in results)
    expected_bpb = pooled_nats / (math.log(2) * pooled_bytes)
    assert results[0]["bpb"] == expected_bpb
    assert results[1]["bpb"] == expected_bpb
    assert t.isfinite(t.tensor(results[0]["bpb"]))
