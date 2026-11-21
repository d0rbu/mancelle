"""Training-scope invariants over real GRPO collection, replay, and updates."""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, cast
from unittest.mock import Mock

import pytest
import torch as t
from torch import nn

from scripts.train.conditional_encdec_sft_grpo import _build_optimizer
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    run_grpo_optimizer_step,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    NoneInitialConditioningProvider,
    SeedContext,
    assert_grpo_gradients_ready,
    backward_replay_group,
    collect_conversation_group,
    configure_grpo_trainable_parameters,
)
from steering.sft.structured_dataset import RankCursorState, StagedEnvironmentBatch
from tests.scripts.test_conditional_encdec_sft_grpo_ntp import _optimizer_cfg, _optimizer_model
from tests.steering.selfcond.test_sft_grpo_training import (
    TinySequentialPolicy,
    _config,
    _environment,
    _three_segment_environment,
)


class _ResidualBlock(nn.Module):
    def __init__(self, width: int) -> None:
        super().__init__()
        self.c_fc = nn.Linear(width, 4 * width, bias=False)
        self.c_proj = nn.Linear(4 * width, width, bias=False)
        nn.init.zeros_(self.c_proj.weight)

    def forward(self, x: t.Tensor) -> t.Tensor:
        normalized = t.nn.functional.rms_norm(x, (x.shape[-1],))
        return x + self.c_proj(self.c_fc(normalized).relu().square())


class _NestedHead(nn.Module):
    """Test module ownership without depending on companion issue #1681."""

    def __init__(self, output: nn.Linear, depth: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([_ResidualBlock(output.in_features) for _ in range(depth)])
        self.output = output
        self.out_features = output.out_features

    def forward(self, x: t.Tensor) -> t.Tensor:
        for block in self.blocks:
            x = block(x)
        return self.output(x)


def _manifest(model: nn.Module) -> tuple[str, ...]:
    return tuple(name for name, p in model.named_parameters() if p.requires_grad)


def _group(model, *, mode=ActionMode.WEIGHTED_WITHOUT_REPLACEMENT, version=2, microbatch=2):
    return collect_conversation_group(
        model,
        _three_segment_environment(),
        group_size=8,
        config=_config(mode, action_policy_version=version),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=microbatch,
    )


@pytest.mark.parametrize("train_lm_head", [False, True])
@pytest.mark.parametrize("explicit_false", [False, True])
def test_default_manifest_and_optimizer_are_exactly_legacy(train_lm_head, explicit_false):
    legacy = _optimizer_model()
    candidate = copy.deepcopy(legacy)
    # Independent pre-change oracle: do not call the selector on both sides.
    legacy.requires_grad_(True)
    legacy.lm_head.requires_grad_(train_lm_head)
    candidate.requires_grad_(False)
    kwargs = {"train_feature_head_only": False} if explicit_false else {}
    actual = configure_grpo_trainable_parameters(candidate, train_lm_head=train_lm_head, **kwargs)
    assert actual == _manifest(legacy)
    t.testing.assert_close(candidate.state_dict(), legacy.state_dict(), rtol=0, atol=0)
    old_optimizer = _build_optimizer(legacy, _optimizer_cfg(), fp_head_lr=None)
    new_optimizer = _build_optimizer(candidate, _optimizer_cfg(), fp_head_lr=None)
    assert new_optimizer.state_dict() == old_optimizer.state_dict()
    for model, optimizer in ((legacy, old_optimizer), (candidate, new_optimizer)):
        names = {id(p): name for name, p in model.named_parameters()}
        groups = [[names[id(p)] for p in g["params"]] for g in optimizer.param_groups]
        if model is legacy:
            expected_groups = groups
        else:
            assert groups == expected_groups


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_head_selection_is_recursive_idempotent_and_optimizer_complete(depth):
    model = _optimizer_model()
    if depth:
        model.feature_pred_head = _NestedHead(model.feature_pred_head, depth)
    initial = copy.deepcopy(model.state_dict())
    for index, p in enumerate(model.parameters()):
        p.requires_grad_(index % 2 == 0)
    expected = tuple(
        f"feature_pred_head.{name}" for name, _p in model.feature_pred_head.named_parameters()
    )
    for _ in range(2):
        assert (
            configure_grpo_trainable_parameters(
                model, train_lm_head=False, train_feature_head_only=True
            )
            == expected
        )
    assert _manifest(model) == expected
    t.testing.assert_close(model.state_dict(), initial, rtol=0, atol=0)
    optimizer = _build_optimizer(
        model, _optimizer_cfg(lr_multiplier=2, init_lr_frac=0.5), fp_head_lr=None
    )
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert [id(p) for p in group["params"]] == [id(p) for p in model.feature_pred_head.parameters()]
    assert group["kind"] == "adamw"
    assert group["lr"] == group["initial_lr"] == 0.004 * (64 / 768) ** -0.5
    assert group["betas"] == (0.8, 0.95)
    assert group["eps"] == 1e-10 and group["weight_decay"] == 0.0
    assert optimizer.state == {}


def test_head_only_rejects_missing_empty_or_conflicting_heads():
    model = TinySequentialPolicy()
    before = _manifest(model)
    with pytest.raises(SFTGRPOError, match="cannot also train the LM head"):
        configure_grpo_trainable_parameters(model, train_lm_head=True, train_feature_head_only=True)
    assert _manifest(model) == before
    del model.feature_pred_head
    with pytest.raises(SFTGRPOError, match="requires a feature-prediction head"):
        configure_grpo_trainable_parameters(
            model, train_lm_head=False, train_feature_head_only=True
        )
    model.add_module("feature_pred_head", nn.Identity())
    with pytest.raises(SFTGRPOError, match="no trainable parameters"):
        configure_grpo_trainable_parameters(
            model, train_lm_head=False, train_feature_head_only=True
        )


@pytest.mark.parametrize("mode", list(ActionMode))
@pytest.mark.parametrize("version", [1, 2])
@pytest.mark.parametrize("microbatch", [1, 2, 8])
def test_frozen_trunk_preserves_behavior_and_unclipped_head_gradients(mode, version, microbatch):
    t.manual_seed(44)
    full = TinySequentialPolicy().eval()
    head = copy.deepcopy(full)
    configure_grpo_trainable_parameters(full, train_lm_head=False)
    manifest = configure_grpo_trainable_parameters(
        head, train_lm_head=False, train_feature_head_only=True
    )
    group = _group(full, mode=mode, version=version, microbatch=microbatch)
    assert _group(head, mode=mode, version=version, microbatch=microbatch) == group
    kwargs: dict[str, Any] = {
        "config": _config(mode, action_policy_version=version),
        "trajectory_microbatch_size": microbatch,
    }
    expected_metrics = backward_replay_group(full, group, **kwargs)
    assert backward_replay_group(head, group, **kwargs) == expected_metrics
    assert_grpo_gradients_ready(head, expected_manifest=manifest)
    for name, p in head.named_parameters():
        if p.requires_grad:
            gradient = p.grad
            expected_gradient = dict(full.named_parameters())[name].grad
            assert gradient is not None and expected_gradient is not None
            assert t.equal(gradient, expected_gradient)
            assert bool(gradient.any()), name
        else:
            assert p.grad is None, name


@pytest.mark.parametrize("depth", [1, 2])
def test_nested_zero_initialized_head_learns_in_two_stages(depth):
    t.manual_seed(44)
    model = TinySequentialPolicy().eval()
    head = _NestedHead(model.feature_pred_head, depth)
    model.add_module("feature_pred_head", head)
    manifest = configure_grpo_trainable_parameters(
        model, train_lm_head=False, train_feature_head_only=True
    )
    optimizer = t.optim.AdamW(model.feature_pred_head.parameters(), lr=0.01, weight_decay=0)
    initial = copy.deepcopy(model.state_dict())
    for step in range(2):
        optimizer.zero_grad(set_to_none=True)
        backward_replay_group(
            model,
            _group(model),
            config=_config(action_policy_version=2),
            trajectory_microbatch_size=2,
        )
        assert_grpo_gradients_ready(model, expected_manifest=manifest)
        for entry in head.blocks:
            block = cast(_ResidualBlock, entry)
            assert block.c_proj.weight.grad is not None and block.c_fc.weight.grad is not None
            assert bool(block.c_proj.weight.grad.any())
            assert bool(block.c_fc.weight.grad.any()) is (step > 0)
        optimizer.step()
        for name, p in model.named_parameters():
            if not p.requires_grad:
                assert p.grad is None and t.equal(p, initial[name])
                assert p not in optimizer.state
    assert all(
        not t.equal(p, initial[name]) for name, p in model.named_parameters() if p.requires_grad
    )


@pytest.mark.parametrize(
    "mode,zero_case",
    [
        (ActionMode.WEIGHTED_WITHOUT_REPLACEMENT, "zero_advantages"),
        (ActionMode.BERNOULLI_TOPK_BOTTOMK, "zero_advantages"),
        (ActionMode.BERNOULLI_TOPK_BOTTOMK, "empty_actions"),
    ],
)
def test_head_only_zero_gradients_are_materialized_by_real_replay(mode, zero_case):
    model = TinySequentialPolicy().eval()
    manifest = configure_grpo_trainable_parameters(
        model, train_lm_head=False, train_feature_head_only=True
    )
    config = _config(mode, action_policy_version=2)
    if zero_case == "empty_actions":
        # Valid counts always request at least one feature. Only Bernoulli
        # shortages can realize an empty action under that contract.
        with t.no_grad():
            model.feature_pred_head.weight.zero_()
            model.feature_pred_head.bias.fill_(-1000)
    group = collect_conversation_group(
        model,
        _environment(),
        group_size=4,
        config=config,
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )
    if zero_case == "empty_actions":
        assert all(
            trace.selected_count == 0
            for trajectory in group.trajectories
            for trace in trajectory.actions
        )
    else:
        group = replace(group, advantages=(0.0,) * 4)
    backward_replay_group(model, group, config=config, trajectory_microbatch_size=2)
    assert_grpo_gradients_ready(model, expected_manifest=manifest)
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None and not bool(p.grad.any())
        else:
            assert p.grad is None


@pytest.mark.parametrize("fault", ["missing", "nonfinite", "scope"])
def test_head_readiness_fails_before_optimizer_and_cursor_commit(monkeypatch, fault):
    model = TinySequentialPolicy().eval()
    manifest = configure_grpo_trainable_parameters(
        model, train_lm_head=False, train_feature_head_only=True
    )
    optimizer = t.optim.AdamW(model.feature_pred_head.parameters(), lr=0.01)
    cursor = Mock(spec=["commit"])
    initial = copy.deepcopy(model.state_dict())
    real_backward = backward_replay_group

    def faulty_backward(*args, **kwargs):
        metrics = real_backward(*args, **kwargs)
        if fault == "missing":
            model.feature_pred_head.bias.grad = None
        elif fault == "nonfinite":
            assert model.feature_pred_head.bias.grad is not None
            model.feature_pred_head.bias.grad.fill_(float("nan"))
        else:
            model.lm_head.requires_grad_(True)
        return metrics

    monkeypatch.setattr(
        "steering.conditional.encoder_decoder.sft_grpo_runtime.backward_replay_group",
        faulty_backward,
    )
    staged = StagedEnvironmentBatch(
        (_environment(),), RankCursorState("a", "b", 0, 1, 1, 1, ()), {}, "base"
    )
    with pytest.raises(SFTGRPOError, match="manifest changed|not optimizer-ready"):
        run_grpo_optimizer_step(
            model,
            optimizer,
            staged=staged,
            cursor=cursor,
            geometry=GRPOBatchGeometry(1, 8, 1, 2, 1),
            policy=_config(),
            completed_optimizer_step=0,
            parameter_manifest=manifest,
            device="cpu",
        )
    cursor.commit.assert_not_called()
    assert optimizer.state == {}
    t.testing.assert_close(model.state_dict(), initial, rtol=0, atol=0)
