"""Production optimizer ownership, collectives, and continuation in head-only GRPO."""

from __future__ import annotations

import copy
import io
import os
from dataclasses import replace
from typing import Any
from unittest.mock import Mock, call, patch

import pytest
import torch as t
import torch.distributed as dist
from loguru import logger

from nanochat.optim import DistMuonAdamW
from scripts.train import conditional_encdec_sft_grpo as trainer
from steering.conditional.encoder_decoder.config import FeaturePredictionConfig, FeaturePredLossMode
from steering.conditional.encoder_decoder.sft_grpo import ActionMode
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    run_grpo_optimizer_step,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    configure_grpo_trainable_parameters,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.sft.structured_dataset import RankCursorState, StagedEnvironmentBatch
from tests.encdec_test_helpers import build_tiny_model
from tests.nccl_multiprocess_helpers import run_nccl_multiprocess
from tests.padded_encoder_reference import use_test_owned_cpu_encoder_reference
from tests.scripts.test_conditional_encdec_sft_grpo_ntp import _optimizer_cfg
from tests.steering.selfcond.test_sft_grpo_training import (
    _config,
    _environment,
    _three_segment_environment,
)
from tests.steering.test_encdec_topology_optimizer_ddp import _collective_patches


def _worker(rank, world_size, mode, width, zero_rank, device_type):
    t.set_num_threads(1)
    t.manual_seed(44)
    t.compiler.reset()
    device = t.device(device_type, rank) if device_type == "cuda" else t.device("cpu")
    if device_type == "cuda":
        attention = trainer._assert_attention_impl(device_type="cuda")
        trainer._configure_deterministic_algorithms(device_type="cuda", attention_impl=attention)
    vocab = FeatureVocab(width * 2, tuple(range(0, width * 2, 2)))
    model = (
        build_tiny_model(
            n_sae_features=width * 2,
            num_buckets=2,
            feature_vocab=vocab,
            feature_pred_config=FeaturePredictionConfig(
                layer=-1, loss_mode=FeaturePredLossMode.MAX_POOL
            ),
            n_layer=2,
            n_embd=64,
            seq_len=16,
            xattn_frequency=2,
        )
        .to(device)
        .eval()
    )
    if device_type == "cuda":
        model.init_weights()  # Exercise the production BF16 embedding / FP32 head contract.
    reference = copy.deepcopy(model).requires_grad_(False)
    initial = copy.deepcopy(model.state_dict())
    manifest = configure_grpo_trainable_parameters(
        model, train_lm_head=False, train_feature_head_only=True
    )
    optimizer = trainer._build_optimizer(model, _optimizer_cfg(), fp_head_lr=None)
    assert isinstance(optimizer, DistMuonAdamW)
    assert len(optimizer.param_groups) == 1
    assert [id(p) for p in optimizer.param_groups[0]["params"]] == [
        id(p) for p in model.feature_pred_head.parameters()
    ]

    environment = _three_segment_environment() if rank == 0 else _environment()
    if zero_rank and rank == 1:
        # Score only a C0 target: all sampled actions then receive the same reward.
        environment = replace(environment, reward_mask=(True, False, False, False))
    staged = StagedEnvironmentBatch(
        (environment,), RankCursorState("a", "b", rank, world_size, 1, 1, ()), {}, "base"
    )
    kwargs: dict[str, Any] = {
        "staged": staged,
        "geometry": GRPOBatchGeometry(world_size, 8, 1, 2, world_size),
        "policy": _config(ActionMode(mode), action_policy_version=2),
        "parameter_manifest": manifest,
        "device": device,
        "kl_beta": 0.0 if zero_rank else 0.2,
        "reference_model": None if zero_rank else reference,
    }
    cursor = Mock(spec=["commit"])
    state_buffer = io.BytesIO()
    with (
        patch.object(dist, "all_reduce", wraps=dist.all_reduce) as all_reduce,
        patch.object(dist, "reduce_scatter_tensor", wraps=dist.reduce_scatter_tensor) as scatter,
        patch.object(dist, "all_gather_into_tensor", wraps=dist.all_gather_into_tensor) as gather,
    ):
        for step in range(2):
            metrics = run_grpo_optimizer_step(
                model, optimizer, cursor=cursor, completed_optimizer_step=step, **kwargs
            )
            if step == 0:
                assert (metrics.gradient_norm == 0) is (zero_rank and rank == 1), (rank, metrics)
                t.save(
                    {"model": model.state_dict(), "optimizer": optimizer.state_dict()}, state_buffer
                )
            elif not zero_rank:
                assert metrics.kl_loss > 0
            for name, parameter in model.named_parameters():
                if not parameter.requires_grad:
                    assert parameter.grad is None and parameter not in optimizer.state
                    assert t.equal(parameter, initial[name]), name
            for name, parameter in reference.named_parameters():
                assert not parameter.requires_grad and parameter.grad is None
                assert t.equal(parameter, initial[name]), name
            for name, value in model.state_dict().items():
                if not name.startswith("feature_pred_head."):
                    assert t.equal(value, initial[name]), name
            for parameter in model.parameters():
                rank_zero = parameter.detach().clone()
                dist.broadcast(rank_zero, src=0)
                assert t.equal(parameter, rank_zero)

    assert cursor.commit.call_args_list == [call(staged), call(staged)]
    assert all(
        not t.equal(parameter, initial[name])
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    )
    assert len(optimizer.state) == 2
    weight_shape = (width, 64)
    reduced_shapes = [tuple(call.args[0].shape) for call in all_reduce.call_args_list]
    scattered_shapes = [tuple(call.args[1].shape) for call in scatter.call_args_list]
    gathered_shapes = [tuple(call.args[0].shape) for call in gather.call_args_list]
    assert (width,) in reduced_shapes
    if width % world_size == 0:
        assert weight_shape in scattered_shapes
        assert weight_shape in gathered_shapes
        assert optimizer.state[model.feature_pred_head.weight]["exp_avg"].shape == (
            width // world_size,
            64,
        )
    else:
        assert weight_shape in reduced_shapes
        assert not scattered_shapes

    # Serialize and restore each rank's actual optimizer shard, then repeat step 2.
    state_buffer.seek(0)
    saved = t.load(state_buffer, map_location=device, weights_only=True)
    restored = copy.deepcopy(model)
    restored.load_state_dict(saved["model"], strict=True, assign=False)
    restored_manifest = configure_grpo_trainable_parameters(
        restored, train_lm_head=False, train_feature_head_only=True
    )
    assert restored_manifest == manifest
    restored_optimizer = trainer._build_optimizer(restored, _optimizer_cfg(), fp_head_lr=None)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_cursor = Mock(spec=["commit"])
    run_grpo_optimizer_step(
        restored, restored_optimizer, cursor=restored_cursor, completed_optimizer_step=1, **kwargs
    )
    t.testing.assert_close(restored.state_dict(), model.state_dict(), rtol=0, atol=0)
    assert restored_optimizer.state_dict()["param_groups"] == optimizer.state_dict()["param_groups"]
    t.testing.assert_close(
        restored_optimizer.state_dict()["state"], optimizer.state_dict()["state"], rtol=0, atol=0
    )
    restored_cursor.commit.assert_called_once_with(staged)
    assert all(
        p.grad is None and p not in restored_optimizer.state
        for p in restored.parameters()
        if not p.requires_grad
    )
    t.compiler.reset()
    return {"manifest": manifest, "commits": cursor.commit.call_count, "resumed_exactly": True}


def _gloo_worker(rank, world_size, mode, width, zero_rank):
    reduce_patch, gather_patch = _collective_patches()
    # Only adapt Gloo's missing Work.get_future; retain real CPU collectives.
    with use_test_owned_cpu_encoder_reference(), reduce_patch, gather_patch:
        try:
            return _worker(rank, world_size, mode, width, zero_rank, "cpu")
        except Exception:
            logger.exception("head-only GRPO test failed on rank {}", rank)
            raise


@pytest.mark.parametrize(
    "mode,width,zero_rank",
    [
        (ActionMode.WEIGHTED_WITHOUT_REPLACEMENT.value, 32, False),
        (ActionMode.BERNOULLI_TOPK_BOTTOMK.value, 33, True),
    ],
)
def test_two_rank_gloo_head_only_update_and_resume(gloo_multiprocess, mode, width, zero_rank):
    results = gloo_multiprocess(2, _gloo_worker, mode, width, zero_rank)
    assert results[0] == results[1]


@pytest.mark.slow
@pytest.mark.skipif(
    not t.cuda.is_available() or not dist.is_nccl_available(), reason="CUDA/NCCL required"
)
@pytest.mark.parametrize("mode", [action.value for action in ActionMode])
def test_one_rank_nccl_head_only_update_and_resume(monkeypatch, mode):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", trainer._target_attention_impl())
    results = run_nccl_multiprocess(1, _worker, mode, 32, False, "cuda")
    assert results[0]["resumed_exactly"]


@pytest.mark.slow
@pytest.mark.skipif(
    os.environ.get("RUN_DDP_SMOKE") != "1"
    or t.cuda.device_count() < 2
    or not dist.is_nccl_available(),
    reason="requires RUN_DDP_SMOKE=1 and two CUDA/NCCL devices",
)
@pytest.mark.parametrize(
    "mode,width,zero_rank",
    [
        (ActionMode.WEIGHTED_WITHOUT_REPLACEMENT.value, 32, False),
        (ActionMode.BERNOULLI_TOPK_BOTTOMK.value, 33, True),
    ],
)
def test_two_rank_nccl_head_only_update_and_resume(monkeypatch, mode, width, zero_rank):
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setenv("NANOCHAT_ATTN_IMPL", trainer._target_attention_impl())
    results = run_nccl_multiprocess(2, _worker, mode, width, zero_rank, "cuda")
    assert results[0] == results[1]
