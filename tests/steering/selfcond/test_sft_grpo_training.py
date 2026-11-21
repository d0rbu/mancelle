from __future__ import annotations

import copy

import pytest
import torch as t
import torch.nn.functional as F
from torch import nn

import steering.conditional.encoder_decoder.sft_grpo as sft_grpo_module
import steering.conditional.encoder_decoder.sft_grpo_training as sft_grpo_training_module
from nanochat.tokenizer import LogicalTurnSpan
from steering.conditional.encoder_decoder.sft_grpo import ActionMode, SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    run_grpo_optimizer_step,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    GRPOPolicyConfig,
    NoneInitialConditioningProvider,
    SeedContext,
    assert_grpo_gradients_ready,
    backward_replay_group,
    collect_conversation_group,
    configure_grpo_trainable_parameters,
)
from steering.conditional.sampling import NegativeSamplingSelection, SamplingSelection
from steering.profiling.markers import TimingRecorder, use_recorder
from steering.sft.structured_dataset import RankCursorState, StagedEnvironmentBatch
from steering.sft.structured_rollout import ConversationEnvironment, TargetSegment


class TinySequentialPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(32, 8)
        self.feature_embedding = nn.Embedding(6, 8)
        self.magnitude_embedding = nn.Embedding(2, 8)
        self.policy_block = nn.Linear(8, 8)
        self.feature_pred_head = nn.Linear(8, 6)
        self.lm_head = nn.Linear(8, 32)

    def forward(
        self,
        inputs: t.Tensor,
        targets: t.Tensor | None,
        feature_ids: t.Tensor,
        magnitudes: t.Tensor,
        mask: t.Tensor,
        seg_map: t.Tensor,
        *,
        fp_logits_only: bool = False,
        loss_reduction: str = "mean",
    ) -> t.Tensor:
        token_hidden = self.token_embedding(inputs)
        conditioning = (
            self.feature_embedding(feature_ids) + self.magnitude_embedding(magnitudes)
        ) * mask.unsqueeze(-1)
        slots = conditioning.sum(dim=2)
        routed = slots.gather(
            1,
            seg_map.unsqueeze(-1).expand(-1, -1, slots.shape[-1]),
        )
        hidden = t.tanh(self.policy_block(token_hidden + routed))
        if fp_logits_only:
            return self.feature_pred_head(hidden)
        assert targets is not None
        logits = self.lm_head(hidden)
        return F.cross_entropy(
            logits.flatten(0, 1),
            targets.flatten(),
            reduction=loss_reduction,
        )


def _environment() -> ConversationEnvironment:
    return ConversationEnvironment(
        original_index=17,
        inputs=(0, 4, 5, 6),
        targets=(4, 5, 6, 7),
        reward_mask=(False, True, True, True),
        logical_turns=(
            LogicalTurnSpan("user", 0, 2),
            LogicalTurnSpan("assistant", 2, 4),
        ),
        segments=(
            TargetSegment(0, 0, 2, None),
            TargetSegment(1, 2, 4, None),
        ),
        seg_map=(0, 0, 0, 1),
    )


def _three_segment_environment() -> ConversationEnvironment:
    return ConversationEnvironment(
        original_index=23,
        inputs=(0, 4, 5, 6, 7, 8),
        targets=(4, 5, 6, 7, 8, 9),
        reward_mask=(False, True, True, True, True, True),
        logical_turns=(
            LogicalTurnSpan("user", 0, 2),
            LogicalTurnSpan("assistant", 2, 6),
        ),
        segments=(
            TargetSegment(0, 0, 2, None),
            TargetSegment(1, 2, 4, None),
            TargetSegment(2, 4, 6, None),
        ),
        seg_map=(0, 0, 0, 1, 1, 2),
    )


def _config(
    mode: ActionMode = ActionMode.WEIGHTED_WITHOUT_REPLACEMENT,
    *,
    action_policy_version: int = 1,
) -> GRPOPolicyConfig:
    return GRPOPolicyConfig(
        action_mode=mode,
        positive_counts=SamplingSelection(mean=1, std=1, min_k=1, max_k=2),
        negative_counts=NegativeSamplingSelection(
            mean=1,
            std=1,
            min_k=0,
            max_k=1,
        ),
        seed=1234,
        action_policy_version=action_policy_version,
    )


def _collect(
    model: TinySequentialPolicy, mode: ActionMode = ActionMode.WEIGHTED_WITHOUT_REPLACEMENT
):
    model.eval()
    return collect_conversation_group(
        model,
        _environment(),
        group_size=8,
        config=_config(mode),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
    )


def test_sequential_collection_and_replay_are_on_policy_and_differentiable() -> None:
    t.manual_seed(9)
    model = TinySequentialPolicy().eval()
    group = _collect(model)

    assert group.initial_conditioning_mode.value == "none"
    assert group.sae_loaded is False
    assert all(len(trajectory.actions) == 1 for trajectory in group.trajectories)
    assert all(len(trajectory.conditioning) == 2 for trajectory in group.trajectories)
    assert all(trajectory.conditioning[0].feature_ids == () for trajectory in group.trajectories)
    assert sum(group.advantages) == pytest.approx(0.0, abs=5e-5)

    manifest = configure_grpo_trainable_parameters(model, train_lm_head=False)
    metrics = backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
    assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
    assert metrics.policy_gradient_loss == pytest.approx(0.0, abs=5e-5)
    assert metrics.live_decisions == 8
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    assert_grpo_gradients_ready(model, expected_manifest=manifest)
    assert model.lm_head.weight.grad is None


def test_weighted_training_reuses_whole_forward_finite_validation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = sft_grpo_module._validate_logits

    def counted(logits: t.Tensor) -> t.Tensor:
        nonlocal calls
        calls += 1
        return original(logits)

    monkeypatch.setattr(sft_grpo_module, "_validate_logits", counted)
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=False)
    backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )
    assert calls == 0


def test_weighted_training_transforms_log_weights_by_trajectory_microbatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shapes = []
    original = sft_grpo_training_module._weighted_log_weights_from_finite_forward

    def captured(logits: t.Tensor) -> tuple[t.Tensor, t.Tensor]:
        shapes.append(tuple(logits.shape))
        return original(logits)

    monkeypatch.setattr(
        sft_grpo_training_module,
        "_weighted_log_weights_from_finite_forward",
        captured,
    )
    model = TinySequentialPolicy().eval()
    group = collect_conversation_group(
        model,
        _environment(),
        group_size=8,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )
    configure_grpo_trainable_parameters(model, train_lm_head=False)
    backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )

    assert shapes == [(2, 6)] * 8


def test_rng_v2_weighted_collection_batches_sampling_and_replays_on_policy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batch_shapes = []
    original = sft_grpo_training_module._sample_weighted_without_replacement_batch_from_log_weights

    def captured(positive_log_weights, negative_log_weights, **kwargs):
        batch_shapes.append(tuple(positive_log_weights.shape))
        return original(positive_log_weights, negative_log_weights, **kwargs)

    def unexpected_legacy_sampler(*args, **kwargs):
        pytest.fail("action-policy v2 reached the per-trajectory weighted sampler")

    monkeypatch.setattr(
        sft_grpo_training_module,
        "_sample_weighted_without_replacement_batch_from_log_weights",
        captured,
    )
    monkeypatch.setattr(
        sft_grpo_training_module,
        "_sample_weighted_without_replacement_from_log_weights",
        unexpected_legacy_sampler,
    )
    config = _config(action_policy_version=2)
    model = TinySequentialPolicy().eval()
    first = collect_conversation_group(
        model,
        _environment(),
        group_size=8,
        config=config,
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )
    second = collect_conversation_group(
        model,
        _environment(),
        group_size=8,
        config=config,
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )

    assert first.trajectories == second.trajectories
    assert batch_shapes == [(2, 6)] * 8
    manifest = configure_grpo_trainable_parameters(model, train_lm_head=False)
    metrics = backward_replay_group(
        model,
        first,
        config=config,
        trajectory_microbatch_size=2,
    )
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
    assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
    assert_grpo_gradients_ready(model, expected_manifest=manifest)


def test_whole_forward_finite_validation_remains_fail_loud() -> None:
    class NonFinitePolicy(TinySequentialPolicy):
        def forward(self, *args, **kwargs):
            result = super().forward(*args, **kwargs)
            if kwargs.get("fp_logits_only"):
                result = result.clone()
                result[..., 0] = t.nan
            return result

    with pytest.raises(SFTGRPOError, match="feature-policy forward produced non-finite logits"):
        _collect(NonFinitePolicy().eval())


def test_compact_encoder_mapping_is_materialized_on_host_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeFeatureVocab:
        def __init__(self) -> None:
            self.devices: list[t.device] = []

        def new_to_old_tensor(self, *, device: t.device) -> t.Tensor:
            self.devices.append(t.device(device))
            return t.tensor([5, 4, 3, 2, 1, 0], dtype=t.int64, device=device)

    model = TinySequentialPolicy().eval()
    vocab = FakeFeatureVocab()
    object.__setattr__(model, "feature_vocab", vocab)
    original = sft_grpo_training_module.trace_to_conditioning
    mapping_ids: set[int] = set()

    def captured(trace, *, compact_to_original):
        assert isinstance(compact_to_original, tuple)
        assert compact_to_original == (5, 4, 3, 2, 1, 0)
        mapping_ids.add(id(compact_to_original))
        return original(trace, compact_to_original=compact_to_original)

    monkeypatch.setattr(sft_grpo_training_module, "trace_to_conditioning", captured)
    group = _collect(model)
    assert len(group.trajectories) == 8
    assert mapping_ids and len(mapping_ids) == 1
    assert vocab.devices.count(t.device("cpu")) == 2


def test_real_trajectory_microbatch_materializes_contiguous_reward_targets() -> None:
    class ContiguityGuardPolicy(TinySequentialPolicy):
        def forward(self, inputs, targets, *args, **kwargs):
            if targets is not None:
                assert targets.is_contiguous()
            return super().forward(inputs, targets, *args, **kwargs)

    model = ContiguityGuardPolicy().eval()
    group = collect_conversation_group(
        model,
        _environment(),
        group_size=4,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )
    assert len(group.trajectories) == 4


@pytest.mark.parametrize(
    ("environment", "expected_policy_forwards"),
    [(_environment(), 1), (_three_segment_environment(), 3)],
)
def test_shared_initial_policy_forward_is_cached_across_collection_microbatches(
    environment: ConversationEnvironment,
    expected_policy_forwards: int,
) -> None:
    class PolicyForwardCounter(TinySequentialPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.policy_forwards = 0

        def forward(self, *args, **kwargs):
            if kwargs.get("fp_logits_only"):
                self.policy_forwards += 1
            return super().forward(*args, **kwargs)

    model = PolicyForwardCounter().eval()
    group = collect_conversation_group(
        model,
        environment,
        group_size=4,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )

    assert len(group.trajectories) == 4
    assert model.policy_forwards == expected_policy_forwards


def test_replay_reconstructs_behavior_prefix_and_masks_future_slots() -> None:
    class ReplayMaskSpy(TinySequentialPolicy):
        def __init__(self) -> None:
            super().__init__()
            self.replay_masks: list[t.Tensor] = []
            self.record_replay = False

        def forward(self, *args, **kwargs):
            if self.record_replay and kwargs.get("fp_logits_only") and not t.is_grad_enabled():
                self.replay_masks.append(args[4].detach().clone())
            return super().forward(*args, **kwargs)

    t.manual_seed(27)
    model = ReplayMaskSpy().eval()
    group = collect_conversation_group(
        model,
        _three_segment_environment(),
        group_size=4,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=2,
    )
    model.record_replay = True
    configure_grpo_trainable_parameters(model, train_lm_head=False)
    metrics = backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=2,
    )

    assert metrics.live_decisions == 8
    assert len(model.replay_masks) == 4
    for first_decision, second_decision in zip(
        model.replay_masks[::2],
        model.replay_masks[1::2],
        strict=True,
    ):
        assert not bool(first_decision.any())
        assert bool(second_decision[:, 1].any())
        assert not bool(second_decision[:, 2].any())


def test_replay_visible_forward_uses_no_grad_behavior_kernel() -> None:
    class GradModeSensitivePolicy(TinySequentialPolicy):
        def forward(self, *args, **kwargs):
            output = super().forward(*args, **kwargs)
            if kwargs.get("fp_logits_only") and t.is_grad_enabled():
                return output + 0.25
            return output

    t.manual_seed(29)
    model = GradModeSensitivePolicy().eval()
    group = _collect(model)
    configure_grpo_trainable_parameters(model, train_lm_head=False)
    metrics = backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=8,
    )

    assert metrics.ratio_min == 1.0
    assert metrics.ratio_max == 1.0
    assert any(
        parameter.grad is not None and bool((parameter.grad != 0).any())
        for parameter in model.parameters()
        if parameter.requires_grad
    )


@pytest.mark.parametrize("microbatch_size", [1, 2, 4, 8])
def test_microbatching_preserves_replay_gradients(microbatch_size: int) -> None:
    t.manual_seed(31)
    reference = TinySequentialPolicy().eval()
    candidate = copy.deepcopy(reference)
    group = _collect(reference)
    configure_grpo_trainable_parameters(reference, train_lm_head=False)
    configure_grpo_trainable_parameters(candidate, train_lm_head=False)

    backward_replay_group(
        reference,
        group,
        config=_config(),
        trajectory_microbatch_size=8,
    )
    backward_replay_group(
        candidate,
        group,
        config=_config(),
        trajectory_microbatch_size=microbatch_size,
    )
    for (name, expected), (candidate_name, actual) in zip(
        reference.named_parameters(),
        candidate.named_parameters(),
        strict=True,
    ):
        assert name == candidate_name
        if expected.requires_grad:
            assert expected.grad is not None
            assert actual.grad is not None
            t.testing.assert_close(actual.grad, expected.grad, atol=2e-6, rtol=2e-5)


def test_future_tokens_cannot_change_preceding_policy_logits() -> None:
    t.manual_seed(2)
    model = TinySequentialPolicy().eval()
    environment = _environment()
    empty_ids = t.zeros((1, 2, 1), dtype=t.int64)
    empty_magnitudes = t.zeros_like(empty_ids)
    empty_mask = t.zeros_like(empty_ids, dtype=t.bool)
    seg_map = t.tensor([environment.seg_map], dtype=t.int64)
    first = model(
        t.tensor([environment.inputs]),
        None,
        empty_ids,
        empty_magnitudes,
        empty_mask,
        seg_map,
        fp_logits_only=True,
    )
    mutated = list(environment.inputs)
    mutated[-1] = 29
    second = model(
        t.tensor([mutated]),
        None,
        empty_ids,
        empty_magnitudes,
        empty_mask,
        seg_map,
        fp_logits_only=True,
    )
    t.testing.assert_close(first[:, :3], second[:, :3])


def test_bernoulli_rollout_and_exact_zero_reference_path() -> None:
    t.manual_seed(5)
    model = TinySequentialPolicy().eval()
    group = _collect(model, ActionMode.BERNOULLI_TOPK_BOTTOMK)
    configure_grpo_trainable_parameters(model, train_lm_head=False)
    metrics = backward_replay_group(
        model,
        group,
        config=_config(ActionMode.BERNOULLI_TOPK_BOTTOMK),
        trajectory_microbatch_size=1,
        kl_beta=0,
        reference_model=None,
    )
    assert metrics.kl_loss == 0
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)


def test_reference_contract_and_missing_gradients_fail_loudly() -> None:
    model = TinySequentialPolicy().eval()
    group = _collect(model)
    manifest = configure_grpo_trainable_parameters(model, train_lm_head=False)
    with pytest.raises(SFTGRPOError, match="no-reference path"):
        backward_replay_group(
            model,
            group,
            config=_config(),
            trajectory_microbatch_size=1,
            reference_model=TinySequentialPolicy(),
        )
    with pytest.raises(SFTGRPOError, match="missing="):
        assert_grpo_gradients_ready(model, expected_manifest=manifest)


def test_complete_optimizer_step_commits_only_after_update(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class CursorSpy:
        committed = False

        def commit(self, staged) -> None:
            del staged
            self.committed = True

    t.manual_seed(44)
    model = TinySequentialPolicy().eval()
    manifest = configure_grpo_trainable_parameters(model, train_lm_head=False)
    optimizer = t.optim.SGD(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=0.01,
    )
    before = model.feature_pred_head.weight.detach().clone()
    state = RankCursorState("a", "b", 0, 1, 1, 1, ())
    # Two rank-local conversations, deliberately. With one, the expected
    # ``1.0 / C`` collapses to ``1.0`` and a regression that hard-coded the
    # constant would pass unnoticed; ``C == 2`` makes the divisor observable.
    staged = StagedEnvironmentBatch(
        (_environment(), _three_segment_environment()),
        state,
        {},
        "base",
    )
    cursor = CursorSpy()
    recorder = TimingRecorder()
    scales = []
    real_replay = sft_grpo_training_module.backward_replay_group

    def replay_spy(*args, **kwargs):
        scales.append(float(kwargs["loss_scale"]).hex())
        return real_replay(*args, **kwargs)

    monkeypatch.setattr(
        "steering.conditional.encoder_decoder.sft_grpo_runtime.backward_replay_group",
        replay_spy,
    )
    monkeypatch.setattr(
        "steering.conditional.encoder_decoder.sft_grpo_runtime.backward_ntp_group",
        lambda *args, **kwargs: pytest.fail("disabled step called NTP backward"),
    )
    with use_recorder(recorder):
        metrics = run_grpo_optimizer_step(
            model,
            optimizer,
            staged=staged,
            cursor=cursor,  # ty: ignore[invalid-argument-type]
            geometry=GRPOBatchGeometry(2, 8, 1, 2, 1),
            policy=_config(),
            completed_optimizer_step=0,
            parameter_manifest=manifest,
            device="cpu",
        )
    assert cursor.committed
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
    assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
    assert metrics.requested_positive == metrics.realized_positive
    assert metrics.requested_negative == metrics.realized_negative
    assert metrics.shortage_actions == 0
    assert metrics.gradient_norm > 0
    assert metrics.ntp_loss == 0.0
    assert scales == [(1.0 / 2).hex(), (1.0 / 2).hex()]
    assert not t.equal(model.feature_pred_head.weight, before)
    assert {row["name"] for row in recorder.rows()} == {
        "grpo/zero_grad",
        "grpo/collection",
        "grpo/replay_and_backward",
        "grpo/gradient_readiness_and_clip",
        "grpo/optimizer_step",
        "grpo/cursor_commit",
    }


@pytest.mark.parametrize(
    "geometry",
    [
        (3, 8, 1, 2, 2),
        (4, 8, 3, 2, 2),
        (4, 8, 1, 3, 2),
        (4, 1, 1, 1, 2),
    ],
)
def test_invalid_batch_geometry_fails(geometry: tuple[int, ...]) -> None:
    with pytest.raises(SFTGRPOError):
        GRPOBatchGeometry(*geometry)
