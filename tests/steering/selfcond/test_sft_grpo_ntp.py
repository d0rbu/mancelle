from __future__ import annotations

import copy
from dataclasses import asdict, replace

import pytest
import torch as t
from torch import nn

import steering.conditional.encoder_decoder.sft_grpo_runtime as runtime_module
from steering.conditional.encoder_decoder.sft_grpo import (
    ActionMode,
    SFTGRPOError,
    masked_mean_token_ce,
)
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    GRPOStepMetrics,
    reduce_grpo_step_metrics,
    run_grpo_optimizer_step,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    NoneInitialConditioningProvider,
    NTPObjective,
    SeedContext,
    assert_grpo_gradients_ready,
    backward_ntp_group,
    backward_replay_group,
    collect_conversation_group,
    configure_grpo_trainable_parameters,
)
from steering.profiling.markers import TimingRecorder, use_recorder
from steering.sft.structured_dataset import RankCursorState, StagedEnvironmentBatch
from tests.steering.selfcond.test_sft_grpo_training import (
    TinySequentialPolicy,
    _collect,
    _config,
    _environment,
    _three_segment_environment,
)


def _gradients(model: TinySequentialPolicy) -> dict[str, t.Tensor | None]:
    return {
        name: None if parameter.grad is None else parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
    }


def _assert_gradients_close(
    actual: dict[str, t.Tensor | None],
    expected: dict[str, t.Tensor | None],
    *,
    atol: float = 0.0,
    rtol: float = 0.0,
) -> None:
    assert actual.keys() == expected.keys()
    for name, actual_gradient in actual.items():
        if expected[name] is None:
            assert actual_gradient is None, name
        else:
            assert actual_gradient is not None, name
            t.testing.assert_close(actual_gradient, expected[name], atol=atol, rtol=rtol)


def test_masked_mean_token_ce_contracts() -> None:
    ce = t.tensor([[1.0, 10.0, 3.0], [4.0, 6.0, 8.0]])
    mask = t.tensor([[True, False, True], [False, True, True]])
    t.testing.assert_close(masked_mean_token_ce(ce, mask), t.tensor([2.0, 7.0]))
    with pytest.raises(SFTGRPOError, match=r"share shape \(B,T\)"):
        masked_mean_token_ce(ce[:, :2], mask)
    with pytest.raises(SFTGRPOError, match=r"share shape \(B,T\)"):
        masked_mean_token_ce(ce, mask.float())
    with pytest.raises(SFTGRPOError, match="contains no valid target"):
        masked_mean_token_ce(ce, t.zeros_like(mask))
    # A PARTIALLY empty batch must fail just as loudly as a wholly empty one.
    # With every row empty, `(valid == 0).any()` and `(valid == 0).all()` agree,
    # so the all-empty case above cannot distinguish them; relaxing the guard to
    # `.all()` would let this row divide by zero and return NaN instead of
    # raising. One valid row plus one empty row is what pins `.any()`.
    with pytest.raises(SFTGRPOError, match="contains no valid target"):
        masked_mean_token_ce(ce, t.tensor([[True, False, True], [False, False, False]]))
    # The rank guard needs its own case: a (T,) vector shares neither ndim nor
    # the 2-D contract, and passing the same shape for both arguments keeps the
    # shape-equality clause from masking it.
    with pytest.raises(SFTGRPOError, match=r"share shape \(B,T\)"):
        masked_mean_token_ce(ce[0], mask[0])


def test_ntp_gradients_match_an_independent_autograd_oracle() -> None:
    """Pin the gradient itself, not just the reported scalar.

    The returned loss is read off ``row`` and is therefore blind to what
    ``.backward()`` actually propagated: negating the backward leaves the
    reported loss identical, so the ``-reward`` oracle cannot see a sign flip
    that would turn "minimize next-token CE" into "maximize" it. This rebuilds
    the masked-CE objective from scratch with ``t.autograd.grad`` and compares
    every parameter gradient, which pins sign, magnitude, ``loss_scale`` and the
    mean-over-k divisor together.
    """
    t.manual_seed(107)
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    trajectory_count, loss_scale = 2, 0.75

    environment = group.environment
    inputs = t.tensor([environment.inputs], dtype=t.int64)
    targets = t.tensor([environment.targets], dtype=t.int64)
    seg_map = t.tensor([environment.seg_map], dtype=t.int64)
    reward_mask = t.tensor([environment.reward_mask], dtype=t.bool)
    n_segments = len(environment.segments)

    def independent_row_loss(trajectory) -> t.Tensor:
        width = max(1, max(len(action.feature_ids) for action in trajectory.conditioning))
        feature_ids = t.zeros((1, n_segments, width), dtype=t.int64)
        magnitudes = t.zeros_like(feature_ids)
        mask = t.zeros_like(feature_ids, dtype=t.bool)
        for slot, action in enumerate(trajectory.conditioning):
            selected = len(action.feature_ids)
            if selected:
                feature_ids[0, slot, :selected] = t.tensor(action.feature_ids, dtype=t.int64)
                magnitudes[0, slot, :selected] = t.tensor(action.magnitudes, dtype=t.int64)
                mask[0, slot, :selected] = True
        per_token = model(
            inputs, targets, feature_ids, magnitudes, mask, seg_map, loss_reduction="none"
        ).reshape_as(targets)
        # Written out longhand on purpose: a mean over the VALID targets only,
        # never over the padded sequence, and positive because the objective is
        # to make the next token more likely.
        valid = reward_mask.sum(dim=1)
        return (per_token.float().masked_fill(~reward_mask, 0.0).sum(dim=1) / valid.float())[0]

    scored = group.trajectories[:trajectory_count]
    oracle_objective = (
        sum(independent_row_loss(trajectory) for trajectory in scored)
        * loss_scale
        / trajectory_count
    )
    parameters = [p for p in model.parameters() if p.requires_grad]
    oracle = dict(
        zip(
            (name for name, p in model.named_parameters() if p.requires_grad),
            t.autograd.grad(oracle_objective, parameters, allow_unused=True),
            strict=True,
        )
    )

    model.zero_grad(set_to_none=True)
    backward_ntp_group(
        model,
        group,
        trajectory_count=trajectory_count,
        loss_scale=loss_scale,
    )
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        if oracle.get(name) is None:
            # A parameter the oracle never reached must not have acquired a
            # gradient from somewhere else either.
            assert parameter.grad is None, name
            continue
        assert parameter.grad is not None, name
        t.testing.assert_close(parameter.grad, oracle[name], atol=1e-6, rtol=1e-5)
    # Non-vacuity, on the parameter this whole objective exists to train: an
    # "any parameter is nonzero" check would pass even with lm_head's gradient
    # identically zero, which is precisely the regression worth catching.
    assert oracle["lm_head.weight"] is not None
    assert bool((oracle["lm_head.weight"] != 0).any())


def test_ntp_loss_is_exact_negative_collected_reward() -> None:
    t.manual_seed(101)
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    rewards = [trajectory.reward for trajectory in group.trajectories]
    assert all(reward != 0.0 for reward in rewards)
    assert len(set(rewards)) >= 3
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    for trajectory in group.trajectories:
        model.zero_grad(set_to_none=True)
        single = replace(group, trajectories=(trajectory,), advantages=(0.0,))
        metrics = backward_ntp_group(
            model,
            single,
            trajectory_count=1,
            loss_scale=1.0,
        )
        assert metrics == -trajectory.reward


def test_ntp_loss_averages_the_first_k_distinct_trajectories() -> None:
    """Identical trajectories cannot see a first-k selection or ordering bug.

    ``test_ntp_gradient_is_mean_over_k`` deliberately uses four copies of one
    trajectory so that pure scaling is isolated -- but that also makes "which k"
    unobservable. This pins the selection with DISTINCT per-trajectory losses:
    the reported mean must equal the mean of the first k, not of any k, and not
    of all of them.
    """
    t.manual_seed(109)
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)

    def single_loss(trajectory) -> float:
        model.zero_grad(set_to_none=True)
        single = replace(group, trajectories=(trajectory,), advantages=(0.0,))
        return backward_ntp_group(model, single, trajectory_count=1, loss_scale=0.0)

    # Several rollouts sample the SAME conditioning, so their losses coincide.
    # Keep one trajectory per distinct loss: equal values would make a wrong
    # subset produce the right mean by accident.
    distinct: dict[float, object] = {}
    for trajectory in group.trajectories:
        distinct.setdefault(single_loss(trajectory), trajectory)
    assert len(distinct) >= 3, "fixture must expose at least three distinguishable rollouts"
    losses = list(distinct)[:3]
    ordered = replace(
        group,
        trajectories=tuple(distinct[loss] for loss in losses),
        advantages=(0.0,) * 3,
    )
    for count in (1, 2, 3):
        model.zero_grad(set_to_none=True)
        metrics = backward_ntp_group(
            model,
            ordered,
            trajectory_count=count,
            loss_scale=0.0,
        )
        assert metrics == pytest.approx(sum(losses[:count]) / count, rel=1e-12, abs=1e-12)


def test_ntp_gradient_is_mean_over_k() -> None:
    t.manual_seed(102)
    source = TinySequentialPolicy().eval()
    collected = _collect(source)
    identical = replace(
        collected,
        trajectories=(collected.trajectories[0],) * 4,
        advantages=(0.0,) * 4,
    )
    results = []
    losses = []
    for count in (1, 2, 4):
        model = copy.deepcopy(source)
        configure_grpo_trainable_parameters(model, train_lm_head=True)
        metrics = backward_ntp_group(
            model,
            identical,
            trajectory_count=count,
            loss_scale=1.0,
        )
        results.append(_gradients(model))
        losses.append(metrics)
    assert losses[0] == losses[1] == losses[2]
    _assert_gradients_close(results[1], results[0])
    _assert_gradients_close(results[2], results[0])


def test_policy_gradient_is_mean_over_group_size() -> None:
    t.manual_seed(103)
    source = TinySequentialPolicy().eval()
    group = _collect(source)
    doubled = replace(
        group,
        trajectories=group.trajectories * 2,
        advantages=group.advantages * 2,
    )
    baseline = copy.deepcopy(source)
    candidate = copy.deepcopy(source)
    configure_grpo_trainable_parameters(baseline, train_lm_head=False)
    configure_grpo_trainable_parameters(candidate, train_lm_head=False)
    backward_replay_group(
        baseline,
        group,
        config=_config(),
        trajectory_microbatch_size=8,
    )
    backward_replay_group(
        candidate,
        doubled,
        config=_config(),
        trajectory_microbatch_size=8,
    )
    _assert_gradients_close(
        _gradients(candidate),
        _gradients(baseline),
        atol=2e-7,
        rtol=2e-6,
    )


class _NoStepOptimizer:
    def __init__(self, model: TinySequentialPolicy) -> None:
        self.model = model
        self.stepped = False

    def zero_grad(self, *, set_to_none: bool) -> None:
        self.model.zero_grad(set_to_none=set_to_none)

    def step(self) -> None:
        self.stepped = True


class _Cursor:
    def __init__(self, optimizer: _NoStepOptimizer) -> None:
        self.optimizer = optimizer
        self.committed = False

    def commit(self, staged: StagedEnvironmentBatch) -> None:
        del staged
        assert self.optimizer.stepped
        self.committed = True


def _staged(count: int) -> StagedEnvironmentBatch:
    state = RankCursorState("a", "b", 0, 1, count, count, ())
    return StagedEnvironmentBatch((_environment(),) * count, state, {}, "base")


def test_step_gradients_are_mean_over_conversations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t.manual_seed(104)
    source = TinySequentialPolicy().eval()
    fixed_group = _collect(source)
    replay_scales: list[str] = []
    ntp_scales: list[str] = []
    real_replay = runtime_module.backward_replay_group
    real_ntp = runtime_module.backward_ntp_group

    def replay_spy(*args, **kwargs):
        replay_scales.append(float(kwargs["loss_scale"]).hex())
        return real_replay(*args, **kwargs)

    def ntp_spy(*args, **kwargs):
        ntp_scales.append(float(kwargs["loss_scale"]).hex())
        return real_ntp(*args, **kwargs)

    monkeypatch.setattr(
        runtime_module, "collect_conversation_group", lambda *args, **kwargs: fixed_group
    )
    monkeypatch.setattr(runtime_module, "backward_replay_group", replay_spy)
    monkeypatch.setattr(runtime_module, "backward_ntp_group", ntp_spy)
    gradients = []
    for conversations in (1, 2):
        model = copy.deepcopy(source)
        manifest = configure_grpo_trainable_parameters(model, train_lm_head=True)
        optimizer = _NoStepOptimizer(model)
        cursor = _Cursor(optimizer)
        run_grpo_optimizer_step(
            model,
            optimizer,
            staged=_staged(conversations),
            cursor=cursor,
            geometry=GRPOBatchGeometry(conversations, 8, 1, 2, 1),
            policy=_config(),
            completed_optimizer_step=0,
            parameter_manifest=manifest,
            device="cpu",
            grad_clip_max_norm=1e9,
            ntp=NTPObjective(1, 5.0),
        )
        assert cursor.committed
        gradients.append(_gradients(model))
    assert replay_scales == [(1.0).hex(), (0.5).hex(), (0.5).hex()]
    assert ntp_scales == [(5.0).hex(), (2.5).hex(), (2.5).hex()]
    _assert_gradients_close(
        gradients[1],
        gradients[0],
        atol=2e-7,
        rtol=2e-6,
    )


def test_ntp_loss_scale_is_linear() -> None:
    t.manual_seed(105)
    source = TinySequentialPolicy().eval()
    group = _collect(source)
    results = []
    for scale in (0.75, 1.5):
        model = copy.deepcopy(source)
        configure_grpo_trainable_parameters(model, train_lm_head=True)
        backward_ntp_group(model, group, trajectory_count=2, loss_scale=scale)
        results.append(_gradients(model))
    for name, baseline in results[0].items():
        actual = results[1][name]
        if baseline is None:
            assert actual is None, name
        else:
            assert actual is not None, name
            t.testing.assert_close(actual, baseline * 2, atol=0, rtol=0)


def test_zero_ntp_weight_materializes_zero_lm_head_gradient() -> None:
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    backward_ntp_group(model, group, trajectory_count=1, loss_scale=0.0)
    assert model.lm_head.weight.grad is not None
    assert t.count_nonzero(model.lm_head.weight.grad).item() == 0


@pytest.mark.parametrize("count", [0, -1, 9, 1.5, True])
def test_ntp_trajectory_count_contract(count: object) -> None:
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    with pytest.raises(SFTGRPOError, match="trajectory_count must be an integer"):
        backward_ntp_group(model, group, trajectory_count=count, loss_scale=1.0)  # ty: ignore[invalid-argument-type]


def test_ntp_requires_grad_enabled_eval_mode() -> None:
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    with t.no_grad(), pytest.raises(SFTGRPOError, match="requires gradients enabled"):
        backward_ntp_group(model, group, trajectory_count=1, loss_scale=1.0)
    model.train()
    with pytest.raises(SFTGRPOError, match="requires eval-mode model"):
        backward_ntp_group(model, group, trajectory_count=1, loss_scale=1.0)


def test_ntp_rejects_a_malformed_forward_return() -> None:
    """The forward-shape guard needs a model that actually violates it."""

    class WrongShapePolicy(TinySequentialPolicy):
        # Armed only AFTER collection: collection runs the same teacher-forced
        # forward and would otherwise trip its own guard first, so the test
        # would never reach the one it is named for.
        truncate = False

        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            if self.truncate and kwargs.get("loss_reduction") == "none":
                return result[:-1]
            return result

    model = WrongShapePolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    model.truncate = True
    with pytest.raises(SFTGRPOError, match="one CE value per target"):
        backward_ntp_group(model, group, trajectory_count=1, loss_scale=1.0)


def test_ntp_rejects_a_conditioning_history_with_the_wrong_slot_count() -> None:
    """A trajectory whose history does not tile the segments must fail loud.

    Silently accepting it would route conditioning into the wrong segments. The
    guard lives in the shared ``_conditioning_tensors``, so this pins that the
    NTP path actually reaches it rather than building tensors of its own.
    """
    t.manual_seed(110)
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    truncated = replace(
        group,
        trajectories=(
            replace(group.trajectories[0], conditioning=group.trajectories[0].conditioning[:-1]),
        ),
        advantages=(0.0,),
    )
    with pytest.raises(SFTGRPOError, match="one slot per segment"):
        backward_ntp_group(model, truncated, trajectory_count=1, loss_scale=1.0)


def test_reported_ntp_loss_is_a_mean_over_conversations_not_a_sum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two conversations must report the per-conversation loss, not twice it.

    A single-conversation step cannot tell a mean from a sum apart, so pin it at
    C=2: with both conversations scoring the same fixed group, the mean is that
    conversation's own loss while a sum would be double it.
    """
    t.manual_seed(114)
    source = TinySequentialPolicy().eval()
    fixed_group = _collect(source)
    monkeypatch.setattr(
        runtime_module, "collect_conversation_group", lambda *args, **kwargs: fixed_group
    )

    class _Cursor:
        def commit(self, staged) -> None:
            del staged

    def step(conversations: int) -> float:
        # A fresh model each time: the optimizer step mutates parameters, and
        # replaying a group collected at the ORIGINAL ones would trip the
        # fail-loud behavior/replay equality check.
        model = copy.deepcopy(source)
        manifest = configure_grpo_trainable_parameters(model, train_lm_head=True)
        optimizer = t.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
        return run_grpo_optimizer_step(
            model,
            optimizer,
            staged=_staged(conversations),
            cursor=_Cursor(),  # ty: ignore[invalid-argument-type]
            geometry=GRPOBatchGeometry(conversations, 8, 1, 2, 1),
            policy=_config(),
            completed_optimizer_step=0,
            parameter_manifest=manifest,
            device="cpu",
            ntp=NTPObjective(1, 1.0),
        ).ntp_loss

    assert step(2) == pytest.approx(step(1))


def test_objective_gradient_sources_are_disjoint_at_heads() -> None:
    t.manual_seed(106)
    source = TinySequentialPolicy().eval()
    group = _collect(source)
    policy = copy.deepcopy(source)
    ntp = copy.deepcopy(source)
    configure_grpo_trainable_parameters(policy, train_lm_head=True)
    configure_grpo_trainable_parameters(ntp, train_lm_head=True)
    backward_replay_group(
        policy,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )
    backward_ntp_group(ntp, group, trajectory_count=2, loss_scale=1.0)
    assert policy.lm_head.weight.grad is None
    assert ntp.feature_pred_head.weight.grad is None
    assert ntp.feature_pred_head.bias.grad is None


def test_objective_gradients_add_exactly() -> None:
    t.manual_seed(107)
    source = TinySequentialPolicy().eval()
    group = _collect(source)
    policy = copy.deepcopy(source)
    ntp = copy.deepcopy(source)
    combined = copy.deepcopy(source)
    for model in (policy, ntp, combined):
        configure_grpo_trainable_parameters(model, train_lm_head=True)
    backward_replay_group(
        policy,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )
    backward_ntp_group(ntp, group, trajectory_count=2, loss_scale=1.0)
    backward_replay_group(
        combined,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )
    backward_ntp_group(combined, group, trajectory_count=2, loss_scale=1.0)
    policy_grads = _gradients(policy)
    ntp_grads = _gradients(ntp)
    combined_grads = _gradients(combined)
    for name, actual in combined_grads.items():
        assert actual is not None, name
        expected = t.zeros_like(actual)
        if policy_grads[name] is not None:
            expected.add_(policy_grads[name])
        if ntp_grads[name] is not None:
            expected.add_(ntp_grads[name])
        t.testing.assert_close(actual, expected, atol=4e-9, rtol=1e-7)


def test_manifest_modes_and_missing_ntp_backward_are_fail_loud() -> None:
    class NoBiasLMHeadPolicy(TinySequentialPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.lm_head = nn.Linear(8, 32, bias=False)

    model = NoBiasLMHeadPolicy().eval()
    policy_manifest = configure_grpo_trainable_parameters(
        model,
        train_lm_head=False,
    )
    assert policy_manifest == (
        "token_embedding.weight",
        "feature_embedding.weight",
        "magnitude_embedding.weight",
        "policy_block.weight",
        "policy_block.bias",
        "feature_pred_head.weight",
        "feature_pred_head.bias",
    )
    assert not model.lm_head.weight.requires_grad
    ntp_manifest = configure_grpo_trainable_parameters(model, train_lm_head=True)
    # The manifest is a resume-identity field, so its ORDER is a checkpoint
    # compatibility contract, not an incidental detail.
    assert ntp_manifest == (*policy_manifest, "lm_head.weight")
    assert all(parameter.requires_grad for parameter in model.parameters())
    group = _collect(model)
    backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )
    with pytest.raises(SFTGRPOError, match=r"missing=\['lm_head.weight'\]"):
        assert_grpo_gradients_ready(model, expected_manifest=ntp_manifest)


def test_enabled_step_updates_both_heads_and_reports_ntp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    t.manual_seed(108)
    model = TinySequentialPolicy().eval()
    fixed_group = _collect(model)
    monkeypatch.setattr(
        runtime_module,
        "collect_conversation_group",
        lambda *args, **kwargs: fixed_group,
    )
    manifest = configure_grpo_trainable_parameters(model, train_lm_head=True)
    optimizer = t.optim.SGD(model.parameters(), lr=0.01)
    optimizer.stepped = False  # ty: ignore[unresolved-attribute]
    real_step = optimizer.step

    def step_spy(*args, **kwargs):
        result = real_step(*args, **kwargs)
        optimizer.stepped = True  # ty: ignore[unresolved-attribute]
        return result

    optimizer.step = step_spy  # ty: ignore[invalid-assignment]
    cursor = _Cursor(optimizer)  # ty: ignore[invalid-argument-type]
    before_lm = model.lm_head.weight.detach().clone()
    before_fp = model.feature_pred_head.weight.detach().clone()
    recorder = TimingRecorder()
    with use_recorder(recorder):
        metrics = run_grpo_optimizer_step(
            model,
            optimizer,
            staged=_staged(1),
            cursor=cursor,  # ty: ignore[invalid-argument-type]
            geometry=GRPOBatchGeometry(1, 8, 1, 2, 1),
            policy=_config(),
            completed_optimizer_step=0,
            parameter_manifest=manifest,
            device="cpu",
            ntp=NTPObjective(2, 1.0),
        )
    expected = sum(-trajectory.reward for trajectory in fixed_group.trajectories[:2]) / 2
    assert metrics.ntp_loss == expected
    # "Changed" would accept a sign-flipped update. SGD(lr) with no momentum or
    # decay applies exactly `w - lr * grad`, and clip/step leave `.grad` readable,
    # so pin the actual arithmetic on both heads -- each fed by a DIFFERENT
    # objective, so this also proves neither head consumed the other's gradient.
    assert model.lm_head.weight.grad is not None
    assert bool((model.lm_head.weight.grad != 0).any())
    assert model.feature_pred_head.weight.grad is not None
    assert bool((model.feature_pred_head.weight.grad != 0).any())
    t.testing.assert_close(
        model.lm_head.weight,
        before_lm - 0.01 * model.lm_head.weight.grad,
    )
    t.testing.assert_close(
        model.feature_pred_head.weight,
        before_fp - 0.01 * model.feature_pred_head.weight.grad,
    )
    assert cursor.committed
    assert {row["name"] for row in recorder.rows()} == {
        "grpo/zero_grad",
        "grpo/collection",
        "grpo/replay_and_backward",
        "grpo/ntp_backward",
        "grpo/gradient_readiness_and_clip",
        "grpo/optimizer_step",
        "grpo/cursor_commit",
    }


def test_single_rank_metric_reduction_round_trips_ntp_fields() -> None:
    metrics = runtime_module.GRPOStepMetrics(
        policy_gradient_loss=1.0,
        kl_loss=2.0,
        total_loss=3.0,
        reward_mean=4.0,
        reward_std=5.0,
        zero_variance_groups=6,
        ratio_min=0.9,
        ratio_max=1.1,
        live_decisions=7,
        requested_positive=8,
        requested_negative=9,
        realized_positive=10,
        realized_negative=11,
        shortage_actions=12,
        empty_actions=13,
        gradient_norm=14.0,
        ntp_loss=15.0,
    )
    assert reduce_grpo_step_metrics(metrics, device="cpu") == metrics


# ---------------------------------------------------------------------------
# Distributed reduction: the ONLY place the new all-reduce indices execute.
# ---------------------------------------------------------------------------

# Per-rank inputs are all mutually distinct and cross-field distinct, so ANY
# index confusion in the packed all-reduce tensor changes at least one output.
_RANK_METRICS = (
    {
        "policy_gradient_loss": 1.0,
        "kl_loss": 2.0,
        "total_loss": 3.0,
        "reward_mean": 4.0,
        "reward_std": 0.0,
        "zero_variance_groups": 1,
        "ratio_min": 0.5,
        "ratio_max": 1.5,
        "live_decisions": 10,
        "requested_positive": 11,
        "requested_negative": 12,
        "realized_positive": 13,
        "realized_negative": 14,
        "shortage_actions": 15,
        "empty_actions": 16,
        "gradient_norm": 7.0,
        "ntp_loss": 20.0,
    },
    {
        "policy_gradient_loss": 3.0,
        "kl_loss": 6.0,
        "total_loss": 9.0,
        "reward_mean": 8.0,
        "reward_std": 0.0,
        "zero_variance_groups": 2,
        "ratio_min": 0.25,
        "ratio_max": 2.5,
        "live_decisions": 100,
        "requested_positive": 110,
        "requested_negative": 120,
        "realized_positive": 130,
        "realized_negative": 140,
        "shortage_actions": 150,
        "empty_actions": 160,
        "gradient_norm": 9.0,
        "ntp_loss": 40.0,
    },
)


def _reduce_on_rank(rank: int, world_size: int) -> dict[str, float]:
    """Reduce this rank's metrics inside a REAL gloo process group.

    Module-level so ``spawn`` can pickle it; the child re-imports this module,
    which is also how it gets ``_RANK_METRICS``.
    """
    assert world_size == 2
    return asdict(reduce_grpo_step_metrics(GRPOStepMetrics(**_RANK_METRICS[rank]), device="cpu"))


@pytest.mark.slow
def test_metric_reduction_over_two_real_ranks_averages_and_sums_each_field() -> None:
    """Execute the distributed branch for real, on two processes.

    Single-rank ``reduce_grpo_step_metrics`` returns its input untouched, so the
    packed all-reduce tensor -- including the two new NTP entries appended at
    indices 13 and 14 -- never runs in any other test. An off-by-one there would
    silently mis-report every field after it.
    """
    from tests.axbench_base._multiprocess_helpers import run_gloo_multiprocess

    results = run_gloo_multiprocess(2, _reduce_on_rank)
    assert len(results) == 2
    for reduced in results:
        # Averaged across ranks.
        assert reduced["policy_gradient_loss"] == pytest.approx(2.0)
        assert reduced["kl_loss"] == pytest.approx(4.0)
        assert reduced["total_loss"] == pytest.approx(6.0)
        assert reduced["reward_mean"] == pytest.approx(6.0)
        # second moments 16 and 64 -> mean 40; variance 40 - 6^2 = 4
        assert reduced["reward_std"] == pytest.approx(2.0)
        assert reduced["ntp_loss"] == pytest.approx(30.0)
        # Summed across ranks.
        assert reduced["zero_variance_groups"] == 3
        assert reduced["live_decisions"] == 110
        assert reduced["requested_positive"] == 121
        assert reduced["requested_negative"] == 132
        assert reduced["realized_positive"] == 143
        assert reduced["realized_negative"] == 154
        assert reduced["shortage_actions"] == 165
        assert reduced["empty_actions"] == 176
        # Bounds, not means.
        assert reduced["ratio_min"] == pytest.approx(0.25)
        assert reduced["ratio_max"] == pytest.approx(2.5)
        assert reduced["gradient_norm"] == pytest.approx(9.0)
    assert results[0] == results[1], "every rank must observe identical reduced metrics"


# ---------------------------------------------------------------------------
# Fail-loud guards. Each of these lines was previously unexecuted, so nothing
# proved the guard fires at all.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("loss_scale", [-1.0, float("nan"), float("inf")])
def test_backward_ntp_group_rejects_an_invalid_loss_scale(loss_scale: float) -> None:
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    with pytest.raises(SFTGRPOError, match="NTP loss_scale must be finite and non-negative"):
        backward_ntp_group(model, group, trajectory_count=1, loss_scale=loss_scale)


def test_trainable_parameters_requires_a_model_that_exposes_lm_head() -> None:
    class Headless(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Linear(4, 4)

    with pytest.raises(SFTGRPOError, match="must expose an untied lm_head"):
        configure_grpo_trainable_parameters(Headless(), train_lm_head=False)


def test_trainable_parameters_rejects_a_tied_lm_head_when_ntp_needs_to_train_it() -> None:
    """A TIED head is invisible to ``named_parameters()`` and must not pass.

    ``named_parameters()`` de-duplicates by tensor identity and keeps the FIRST
    name, so tying ``lm_head.weight`` to the token embedding removes it from the
    manifest entirely. NTP would then believe it was training the readout while
    the optimizer never saw it -- and, worse, freezing it would silently freeze
    the tied embedding. This objective requires an untied head.
    """

    class Tied(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.token_embedding = nn.Embedding(8, 4)
            self.lm_head = nn.Linear(4, 8, bias=False)
            self.lm_head.weight = self.token_embedding.weight

    assert not any(name.startswith("lm_head") for name, _ in Tied().named_parameters()), (
        "fixture must actually hide the tied head, or the guard is untested"
    )
    with pytest.raises(AssertionError, match="trainable lm_head is absent"):
        configure_grpo_trainable_parameters(Tied(), train_lm_head=True)


def test_trainable_parameters_detects_a_freeze_that_silently_did_not_take() -> None:
    """Defense in depth: prove the leak assertion is not dead code."""

    class StubbornHead(nn.Linear):
        def requires_grad_(self, requires_grad: bool = True) -> StubbornHead:
            return self  # deliberately ignores the freeze

    class Model(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.block = nn.Linear(4, 4)
            self.lm_head = StubbornHead(4, 8, bias=False)

    with pytest.raises(AssertionError, match="frozen lm_head leaked"):
        configure_grpo_trainable_parameters(Model(), train_lm_head=False)


def test_ntp_scores_the_complete_history_over_three_segments() -> None:
    """Slot routing needs more than the two-segment fixture to be observable.

    Every other NTP test uses a 2-segment environment with a single action, where
    history ORDER cannot matter. The spec's distinguishing claim -- that NTP scores
    the COMPLETE conditioning history, unlike replay which masks future slots --
    only has room to fail with three or more segments. Reversing the history must
    therefore change the loss.
    """
    t.manual_seed(111)
    model = TinySequentialPolicy().eval()
    group = collect_conversation_group(
        model,
        _three_segment_environment(),
        group_size=4,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=1,
    )
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    trajectory = group.trajectories[0]
    assert len(trajectory.conditioning) == 3
    straight = backward_ntp_group(
        model,
        replace(group, trajectories=(trajectory,), advantages=(0.0,)),
        trajectory_count=1,
        loss_scale=0.0,
    )
    assert straight == pytest.approx(-trajectory.reward)
    reversed_history = replace(
        group,
        trajectories=(replace(trajectory, conditioning=tuple(reversed(trajectory.conditioning))),),
        advantages=(0.0,),
    )
    shuffled = backward_ntp_group(
        model,
        reversed_history,
        trajectory_count=1,
        loss_scale=0.0,
    )
    assert shuffled != straight, "slot order must reach the encoder"


def test_ntp_runs_on_a_bernoulli_action_group() -> None:
    """NTP must be action-mode agnostic: it replays conditioning, not traces.

    Bernoulli selection can yield different per-slot widths than the weighted
    policy, and every prior NTP test used only weighted sampling.
    """
    t.manual_seed(112)
    model = TinySequentialPolicy().eval()
    group = _collect(model, ActionMode.BERNOULLI_TOPK_BOTTOMK)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    metrics = backward_ntp_group(model, group, trajectory_count=2, loss_scale=1.0)
    expected = sum(-trajectory.reward for trajectory in group.trajectories[:2]) / 2
    assert metrics == pytest.approx(expected)
    assert model.lm_head.weight.grad is not None
    assert model.feature_pred_head.weight.grad is None


def test_replay_loss_scale_multiplies_the_policy_and_kl_terms_together() -> None:
    """``loss_scale`` multiplies ``pg + kl_beta*kl`` as a single unit.

    The reference anchor therefore scales with the policy gradient rather than
    holding constant. Nothing else records that, and a refactor to two separate
    scales would silently change the anchor's strength.
    """
    t.manual_seed(113)
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    reference = copy.deepcopy(model).eval()
    reference.requires_grad_(False)
    configure_grpo_trainable_parameters(model, train_lm_head=True)

    def kl_scaled_gradient(weight: float) -> t.Tensor:
        model.zero_grad(set_to_none=True)
        backward_replay_group(
            model,
            group,
            config=_config(),
            trajectory_microbatch_size=2,
            loss_scale=weight,
            kl_beta=0.5,
            reference_model=reference,
        )
        grad = model.feature_pred_head.weight.grad
        assert grad is not None
        return grad.clone()

    single = kl_scaled_gradient(1.0)
    double = kl_scaled_gradient(2.0)
    assert bool((single != 0).any()), "fixture must produce a nonzero policy+KL gradient"
    t.testing.assert_close(double, single * 2.0, atol=1e-6, rtol=1e-5)
