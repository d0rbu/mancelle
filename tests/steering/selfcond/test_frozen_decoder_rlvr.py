"""Exact CUDA RLVR parity through three updates with frozen decoder shadows."""

from __future__ import annotations

import copy
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as flash_attention
import steering.eval.chat_self_conditioned_generation as chat_generation
from nanochat.flash_attention import FlashAttnImpl
from nanochat.optim import MuonAdamW
from steering.conditional.encoder_decoder.config import FeaturePredictionBlock
from steering.conditional.encoder_decoder.frozen_decoder import stage_frozen_decoder_linears
from steering.conditional.encoder_decoder.rlvr_policy import (
    RLVRGroupRollout,
    RLVRReplayMetrics,
    backward_rlvr_group,
    collect_rlvr_group,
)
from steering.conditional.encoder_decoder.rlvr_runtime import configure_rlvr_compiler
from steering.conditional.encoder_decoder.sft_grpo_training import (
    assert_grpo_gradients_ready,
    configure_grpo_trainable_parameters,
)
from tests.steering.selfcond.rlvr_test_helpers import factory, policy
from tests.steering.test_chat_self_conditioned_generation import make_fixture


@dataclass
class _StepReceipt:
    group: RLVRGroupRollout
    token_logits: list[t.Tensor]
    metrics: RLVRReplayMetrics
    gradients: dict[str, t.Tensor]
    parameters: dict[str, t.Tensor]
    optimizer: dict[str, Any]


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="native compiled encoder requires CUDA")
def test_frozen_decoder_shadows_preserve_three_generated_updates(
    tmp_path, monkeypatch, *, head_block=FeaturePredictionBlock.MLP,
):
    """Script token choices, retaining real logits, feature draws, and backward.

    Both arms retain compiled kernels between updates; the candidate keeps one
    frozen-shadow context open across all three updates. Mixed binary rewards
    make this a nonzero-gradient qualification fixture, not a benchmark or an
    accuracy experiment. Packed encoder/cross-attention and the repository's
    fused single-device optimizer remain native CUDA code.
    """
    monkeypatch.setenv("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    for key in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "fail_on_recompile_limit_hit",
        "suppress_errors",
    ):
        monkeypatch.setattr(t._dynamo.config, key, getattr(t._dynamo.config, key))
    deterministic = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    precision = t.get_float32_matmul_precision()
    cudnn_sdp = t.backends.cuda.cudnn_sdp_enabled()
    t.use_deterministic_algorithms(True)
    t.set_float32_matmul_precision("highest")
    t.backends.cuda.enable_cudnn_sdp(False)
    t.compiler.reset()
    try:
        device = t.device("cuda")
        fixture = make_fixture(tmp_path, monkeypatch, K=2, head_depth=2, head_block=head_block)
        source = fixture.model.to(device)
        source.init_weights()  # Production BF16 embeddings and FP32 head weights.
        with t.no_grad():
            for name, parameter in source.named_parameters():
                if "cross_attn" in name and parameter.ndim >= 2:
                    parameter.normal_(std=0.04)
                if name.startswith("feature_pred_head.residual_blocks."):
                    parameter.normal_(std=0.03)
        initial = {
            name: parameter.detach().cpu().clone() for name, parameter in source.named_parameters()
        }
        initial_buffers = {
            name: buffer.detach().cpu().clone() for name, buffer in source.named_buffers()
        }
        initial_cpu_rng = t.get_rng_state()
        initial_cuda_rng = t.cuda.get_rng_state_all()
        good = fixture.tokenizer.encode("x")[0]
        bad = fixture.tokenizer.encode("y")[0]
        end = fixture.tokenizer.encode_special("<assistant_end>")
        references: list[_StepReceipt] = []
        for staged in (False, True):
            t.compiler.reset()
            model = copy.deepcopy(source).eval()
            fixture.model = model
            manifest = configure_grpo_trainable_parameters(
                model, train_lm_head=False, train_feature_head_only=True
            )
            # Projection plus both complete MLP or transformer residual blocks.
            assert len(manifest) == {
                FeaturePredictionBlock.MLP: 6,
                FeaturePredictionBlock.TRANSFORMER: 10,
            }[head_block]
            optimizer = model.setup_optimizer(trainable_only=True, fp_head_lr=1e-4)
            assert isinstance(optimizer, MuonAdamW)
            configure_rlvr_compiler(model)
            t.set_rng_state(initial_cpu_rng)
            t.cuda.set_rng_state_all(initial_cuda_rng)
            context = (
                stage_frozen_decoder_linears(
                    model, max_shadow_bytes=64 * 1024**2, min_free_bytes=0,
                    train_feature_head_only=True, world_size=1,
                )
                if staged
                else nullcontext()
            )
            with context as report:
                if staged:
                    assert report is not None
                    assert report.applied and report.reason == "staged"
                    assert report.staged_shadow_bytes > 0
                    assert report.staged_shadow_bytes == report.planned_shadow_bytes
                    assert report.module_names
                    assert all(
                        name.startswith(("transformer.h.", "lm_head"))
                        for name in report.module_names
                    )
                for step in range(3):
                    optimizer.zero_grad(set_to_none=True)
                    token_logits: list[t.Tensor] = []
                    tokens = iter([good] * (2 + step) + [end] + [bad] * (4 + step) + [end])

                    def sample(logits, *_args, observed=token_logits, remaining=tokens):
                        assert bool(t.isfinite(logits).all())
                        observed.append(logits.detach().cpu().clone())
                        return t.tensor([[next(remaining)]], device=logits.device)

                    monkeypatch.setattr(chat_generation, "sample_next_token", sample)
                    group = collect_rlvr_group(
                        model,
                        factory(fixture, device),
                        fixture.prompt,
                        lambda result: float(result.token_ids[len(fixture.prompt)] == good),
                        group_size=2,
                        policy_config=policy(),
                        group_seed=991 + step,
                        max_tokens=5 + step,
                        extraction_tokenizer=fixture.source,
                    )
                    assert next(tokens, None) is None
                    assert [trajectory.reward for trajectory in group.trajectories] == [1.0, 0.0]
                    assert group.advantages == (1.0, -1.0)
                    assert all(
                        len(trajectory.live_actions) >= 2 for trajectory in group.trajectories
                    )
                    metrics = backward_rlvr_group(
                        model, group, policy_config=policy(), train_feature_head_only=True
                    )
                    assert_grpo_gradients_ready(model, expected_manifest=manifest)
                    gradients = {}
                    before = {}
                    for name, parameter in model.named_parameters():
                        if parameter.requires_grad:
                            assert parameter.grad is not None
                            assert bool(parameter.grad.ne(0).any()), (step, name)
                            gradients[name] = parameter.grad.detach().cpu().clone()
                            before[name] = parameter.detach().cpu().clone()
                    t.nn.utils.clip_grad_norm_(model.feature_pred_head.parameters(), 1.0)
                    optimizer.step()
                    t.cuda.synchronize()
                    parameters = {}
                    for name, parameter in model.named_parameters():
                        actual = parameter.detach().cpu().clone()
                        if parameter.requires_grad:
                            assert not t.equal(actual, before[name]), (step, name)
                            parameters[name] = actual
                        else:
                            assert parameter.grad is None and parameter not in optimizer.state
                            t.testing.assert_close(actual, initial[name], rtol=0, atol=0)
                    for name, buffer in model.named_buffers():
                        t.testing.assert_close(
                            buffer.detach().cpu(), initial_buffers[name], rtol=0, atol=0
                        )
                    receipt = _StepReceipt(
                        group,
                        token_logits,
                        metrics,
                        gradients,
                        parameters,
                        copy.deepcopy(optimizer.state_dict()),
                    )
                    if not staged:
                        references.append(receipt)
                    else:
                        reference = references[step]
                        assert (
                            group.prompt.prompt_token_ids == reference.group.prompt.prompt_token_ids
                        )
                        assert (
                            group.prompt.encoder_slot_count
                            == reference.group.prompt.encoder_slot_count
                        )
                        for name in (
                            "enc_feature_ids",
                            "enc_magnitudes",
                            "enc_mask",
                            "prefix_seg_map",
                        ):
                            t.testing.assert_close(
                                getattr(group.prompt, name),
                                getattr(reference.group.prompt, name),
                                rtol=0,
                                atol=0,
                            )
                        assert group.trajectories == reference.group.trajectories
                        assert group.advantages == reference.group.advantages
                        assert metrics == reference.metrics
                        t.testing.assert_close(token_logits, reference.token_logits, rtol=0, atol=0)
                        t.testing.assert_close(gradients, reference.gradients, rtol=0, atol=0)
                        t.testing.assert_close(parameters, reference.parameters, rtol=0, atol=0)
                        assert (
                            receipt.optimizer["param_groups"] == reference.optimizer["param_groups"]
                        )
                        t.testing.assert_close(
                            receipt.optimizer["state"],
                            reference.optimizer["state"],
                            rtol=0,
                            atol=0,
                        )
                if staged:
                    assert report is not None
                    assert report.fallback_calls == 0
                    assert report.invalidated_modules == 0
            if staged:
                assert report is not None
                assert report.cleaned_up
                assert report.fallback_calls == report.invalidated_modules == 0
                assert all(
                    "forward" not in model.get_submodule(name).__dict__
                    for name in report.module_names
                )
    finally:
        t.compiler.reset()
        t.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        t.set_float32_matmul_precision(precision)
        t.backends.cuda.enable_cudnn_sdp(cudnn_sdp)
