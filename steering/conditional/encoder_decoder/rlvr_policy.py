"""On-policy RLVR collection and canonical differentiable feature replay.

The token generator remains the native scalar calculator generator.  This
module replaces only its feature-selection provider and records enough causal
state to replay each feature action from a full, batch-one, no-KV prefix.
"""

from __future__ import annotations

import hashlib
import math
import sys
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager, nullcontext
from contextvars import copy_context
from dataclasses import dataclass
from functools import partial
from typing import Any

import torch as t
from loguru import logger
from torch.utils.checkpoint import checkpoint

from steering.conditional.encoder_decoder import packed_encoder as packed
from steering.conditional.encoder_decoder._rlvr_replay_guard import NativeBinding
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.feature_prediction import pool_fp_logits
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    append_packed_encoder_segment,
)
from steering.conditional.encoder_decoder.rlvr_replay_cache import (
    HiddenReplayReport,
    HiddenReplaySettings,
    RLVRReplaySession,
    _bump,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
    SelfCondState,
    final_real_prompt_source_mask,
)
from steering.conditional.encoder_decoder.sft_grpo import (
    ActionMode,
    ConditioningAction,
    SFTGRPOError,
    SparseActionTrace,
    behavior_checked_replay_ratio,
    conversation_group_advantages,
    sample_weighted_without_replacement,
    trace_to_conditioning,
    validate_count_support,
)
from steering.conditional.encoder_decoder.sft_grpo_training import GRPOPolicyConfig
from steering.conditional.sampling import sample_action_counts
from steering.eval.chat_self_conditioned_generation import (
    ChatSelfConditionedGeneration,
    generate_self_conditioned_chat,
)
from steering.eval.inference_engine import InferenceEngine

RewardFn = Callable[[ChatSelfConditionedGeneration], float]
EngineFactory = Callable[[int], InferenceEngine]


@dataclass(frozen=True)
class RLVRPromptReplay:
    """Exact raw prompt conditioning consumed by the authoritative prefill."""

    prompt_token_ids: tuple[int, ...]
    enc_feature_ids: t.Tensor
    enc_magnitudes: t.Tensor
    enc_mask: t.Tensor
    prefix_seg_map: t.Tensor
    encoder_slot_count: int


@dataclass(frozen=True)
class RLVRActionTrace:
    """One sampled feature action and its causal canonical policy state."""

    decision_index: int
    source: str
    committed_prefix: tuple[int, ...]
    outgoing_seg_map: tuple[int, ...]
    pool_positions: tuple[int, ...]
    first_committed_position: int
    feature_history: tuple[ConditioningAction, ...]
    conditioning: ConditioningAction
    policy_trace: SparseActionTrace

    def is_live(self, last_nonforced_logits_prefix_length: int) -> bool:
        return self.first_committed_position < last_nonforced_logits_prefix_length


@dataclass(frozen=True)
class RLVRTrajectoryTrace:
    """One fully materialized calculator trajectory and its feature decisions."""

    token_ids: tuple[int, ...]
    sampled_mask: tuple[int, ...]
    reward: float
    stop_reason: str
    actions: tuple[RLVRActionTrace, ...]
    last_nonforced_logits_prefix_length: int
    structurally_valid: bool
    invalid_structural_token: int | None

    @property
    def live_actions(self) -> tuple[RLVRActionTrace, ...]:
        return tuple(
            action
            for action in self.actions
            if action.is_live(self.last_nonforced_logits_prefix_length)
        )


@dataclass(frozen=True)
class RLVRGroupRollout:
    """One prompt-local rollout group with population-normalized advantages."""

    prompt: RLVRPromptReplay
    trajectories: tuple[RLVRTrajectoryTrace, ...]
    advantages: tuple[float, ...]


@dataclass(frozen=True)
class RLVRReplayMetrics:
    """Detached policy-surrogate diagnostics for one prompt group."""

    loss: float
    ratio_min: float
    ratio_max: float
    live_decisions: int


@dataclass(frozen=True)
class _PreparedBoundary:
    committed_prefix: tuple[int, ...]
    pool_positions: tuple[int, ...]
    first_committed_position: int
    source: str


def _seed(
    group_seed: int,
    rollout_index: int,
    decision_index: int,
    stream: str,
) -> int:
    payload = (
        f"rlvr-feature-v1\0{group_seed}\0{rollout_index}\0"
        f"{decision_index}\0{stream}"
    ).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _token_seed(group_seed: int, rollout_index: int) -> int:
    payload = f"rlvr-token-v1\0{group_seed}\0{rollout_index}".encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & (2**63 - 1)


def _action_tensors(
    action: ConditioningAction,
    *,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    if not action.feature_ids:
        raise SFTGRPOError("RLVR actions must contain at least one feature")
    count = len(action.feature_ids)
    return (
        t.tensor(action.feature_ids, dtype=t.long, device=device).reshape(1, 1, count),
        t.tensor(action.magnitudes, dtype=t.long, device=device).reshape(1, 1, count),
        t.tensor(action.mask, dtype=t.bool, device=device).reshape(1, 1, count),
    )


def _encode_canonical_state(
    model: Any,
    prompt: RLVRPromptReplay,
    feature_history: Sequence[ConditioningAction],
    *,
    device: t.device,
    replay_cache: RLVRReplaySession | None = None,
) -> PackedEncoderOutput:
    inputs = (prompt.enc_feature_ids.to(device), prompt.enc_magnitudes.to(device), prompt.enc_mask.to(device))
    if replay_cache is not None and replay_cache.settings.prompt_encoder:
        memoized = replay_cache.encode_prompt_and_history(model, inputs, feature_history, device=device)
        if memoized is not None:
            return memoized
        # A decline has performed no native encoding. Consume these conversions
        # exactly once and keep the whole fallback native, without another guard.
        replay_cache = None
    encoder_out = model.encoder(*inputs)
    del inputs
    if replay_cache is not None:
        memoized = replay_cache.encode_history(model, encoder_out, feature_history, device=device)
        if memoized is not None:
            return memoized
    encode_history = model.encoder
    if t.is_grad_enabled():
        replay_context = copy_context()

        def replay_encoder(*inputs: t.Tensor) -> PackedEncoderOutput:
            # Nested backward also needs the caller's compiler ContextVars.
            return replay_context.copy().run(model.encoder, *inputs)

        # Bound per-segment activations and casts during outer-checkpoint replay.
        # Nonreentrant checkpointing supports integer inputs and packed output.
        encode_history = partial(checkpoint, replay_encoder, use_reentrant=False, preserve_rng_state=False)
    for action in feature_history:
        encoded = encode_history(*_action_tensors(action, device=device))
        encoder_out = append_packed_encoder_segment(encoder_out, encoded)
    return encoder_out


def canonical_fp_logits(
    model: Any,
    prompt: RLVRPromptReplay,
    *,
    committed_prefix: Sequence[int],
    outgoing_seg_map: Sequence[int],
    pool_positions: Sequence[int],
    feature_history: Sequence[ConditioningAction],
    train_feature_head_only: bool = False,
    replay_cache: RLVRReplaySession | None = None,
) -> t.Tensor:
    """Evaluate the canonical batch-one, full-prefix, no-KV feature policy."""

    request = None if replay_cache is None else replay_cache.prepare_request(
        model, prompt, committed_prefix=committed_prefix, outgoing_seg_map=outgoing_seg_map,
        pool_positions=pool_positions, feature_history=feature_history,
        train_feature_head_only=train_feature_head_only,
    )
    device = next(model.parameters()).device
    prefix = tuple(int(token) for token in committed_prefix)
    routing = tuple(int(slot) for slot in outgoing_seg_map)
    positions = tuple(int(position) for position in pool_positions)
    if not prefix or len(routing) != len(prefix):
        raise SFTGRPOError("canonical RLVR prefix and outgoing seg_map must be nonempty and aligned")
    if not positions or any(position < 0 or position >= len(prefix) for position in positions):
        raise SFTGRPOError("canonical RLVR pool positions must be nonempty prefix indices")
    if tuple(sorted(set(positions))) != positions:
        raise SFTGRPOError("canonical RLVR pool positions must be unique and increasing")
    cached = (
        replay_cache.lookup(request)
        if replay_cache is not None and request is not None and train_feature_head_only else None
    )
    capture_hidden = replay_cache is not None and request is not None and not train_feature_head_only
    hidden_output = train_feature_head_only or capture_hidden
    try:
        # Frozen parameters alone do not preserve the behavior-time numerics of
        # compiled trunk kernels. Keep encoder, decoder, and normalization in the
        # same no-grad mode used for collection; only the complete head needs grad.
        with t.no_grad() if train_feature_head_only else nullcontext():
            if cached is None:
                encoder_out = _encode_canonical_state(
                    model, prompt, feature_history, device=device,
                    replay_cache=replay_cache if capture_hidden else None,
                )
                encoder_slots = encoder_out.layout.n_segments
            else:
                encoder_out = None
                encoder_slots = cached.encoder_slots
            if any(slot < 0 or slot >= encoder_slots for slot in routing):
                raise SFTGRPOError("canonical RLVR seg_map addresses a nonexistent encoder slot")
            output = cached.hidden if cached is not None else model(
                t.tensor(prefix, dtype=t.long, device=device).unsqueeze(0), None,
                seg_map=t.tensor(routing, dtype=t.long, device=device).unsqueeze(0), encoder_out=encoder_out,
                encoder_strict=True, fp_logits_only=not hidden_output, fp_hidden_only=hidden_output,
            )
        logits = model.apply_feature_prediction_head(output) if hidden_output else output
        if logits.ndim != 3 or logits.shape[:2] != (1, len(prefix)):
            raise SFTGRPOError("canonical fp_logits_only forward returned an invalid shape")
        selected = logits[:, t.tensor(positions, dtype=t.long, device=device), :]
        pooled = pool_fp_logits(selected, model.feature_pred_config.loss_mode)
        if pooled.ndim != 2 or pooled.shape[0] != 1 or not bool(t.isfinite(pooled).all()):
            raise SFTGRPOError("canonical RLVR pooled policy logits must be finite with shape (1,V)")
        if capture_hidden:
            assert replay_cache is not None and request is not None
            replay_cache.capture(request, hidden=output, encoder_slots=encoder_slots)
        return pooled[0]
    finally:
        if capture_hidden:
            assert replay_cache is not None
            replay_cache.discard_pending_history()


class RLVRFeatureProvider(SelfConditionedFeatureProvider):
    """Native self-conditioning provider with canonical weighted RLVR actions."""

    def __init__(
        self,
        model: Any,
        cfg: Any,
        device: t.device,
        *,
        policy_config: GRPOPolicyConfig,
        group_seed: int,
        rollout_index: int,
        s0_logits_cache: list[t.Tensor],
        replay_cache: RLVRReplaySession | None = None,
    ) -> None:
        super().__init__(model, cfg, device)
        if policy_config.action_mode is not ActionMode.WEIGHTED_WITHOUT_REPLACEMENT:
            raise SFTGRPOError("RLVR supports only weighted_without_replacement actions")
        feature_count = (
            int(cfg.n_sae_features)
            if self._compact_candidate_feature_ids is None
            else int(self._compact_candidate_feature_ids.numel())
        )
        positive_max = policy_config.positive_counts.max_k
        negative_max = policy_config.negative_counts.max_k
        if positive_max is None or negative_max is None:
            raise SFTGRPOError("RLVR action count supports must have finite upper bounds")
        validate_count_support(
            positive_min=int(policy_config.positive_counts.min_k),
            positive_max_exclusive=int(positive_max),
            negative_min=int(policy_config.negative_counts.min_k),
            negative_max_exclusive=int(negative_max),
            feature_count=feature_count,
        )
        self.policy_config = policy_config
        self.group_seed = group_seed
        self.rollout_index = rollout_index
        self.s0_logits_cache = s0_logits_cache
        self.replay_cache = replay_cache
        self.prompt_replay: RLVRPromptReplay | None = None
        self.actions: list[RLVRActionTrace] = []
        self.last_nonforced_logits_prefix_length = 0
        self.invalid_structural_token: int | None = None
        self._prepared_boundary: _PreparedBoundary | None = None

    def capture_prompt(
        self,
        *,
        input_ids: t.Tensor,
        enc_feature_ids: t.Tensor,
        enc_magnitudes: t.Tensor,
        enc_mask: t.Tensor,
        seg_map: t.Tensor,
        encoder_slot_count: int,
    ) -> None:
        if self.prompt_replay is not None:
            raise RuntimeError("RLVR trajectory prompt was captured more than once")
        prompt_ids = tuple(int(token) for token in input_ids[0].tolist())
        source_mask = final_real_prompt_source_mask(
            seg_map,
            encoder_slot_count=encoder_slot_count,
        )
        pool_positions = tuple(int(position) for position in source_mask.nonzero().flatten().tolist())
        self.prompt_replay = RLVRPromptReplay(
            prompt_token_ids=prompt_ids,
            enc_feature_ids=enc_feature_ids.detach().clone(),
            enc_magnitudes=enc_magnitudes.detach().clone(),
            enc_mask=enc_mask.detach().clone(),
            prefix_seg_map=seg_map[:, :-1].detach().clone(),
            encoder_slot_count=int(encoder_slot_count),
        )
        self.prepare_boundary(
            committed_prefix=prompt_ids[:-1],
            pool_positions=pool_positions,
            first_committed_position=len(prompt_ids) - 1,
            source="prompt",
        )

    def prepare_boundary(
        self,
        *,
        committed_prefix: Sequence[int],
        pool_positions: Sequence[int],
        first_committed_position: int,
        source: str,
    ) -> None:
        if self._prepared_boundary is not None:
            raise RuntimeError("RLVR boundary context was overwritten before consumption")
        prefix = tuple(int(token) for token in committed_prefix)
        positions = tuple(int(position) for position in pool_positions)
        if source not in {"prompt", "assistant", "tool"}:
            raise ValueError(f"unknown RLVR boundary source {source!r}")
        self._prepared_boundary = _PreparedBoundary(
            committed_prefix=prefix,
            pool_positions=positions,
            first_committed_position=int(first_committed_position),
            source=source,
        )

    def record_nonforced_logits_prefix_length(self, prefix_length: int) -> None:
        if prefix_length <= self.last_nonforced_logits_prefix_length:
            raise RuntimeError("non-forced policy prefixes must advance strictly")
        self.last_nonforced_logits_prefix_length = int(prefix_length)

    def record_invalid_structural_token(self, token: int) -> None:
        if self.invalid_structural_token is not None:
            raise RuntimeError("trajectory already recorded an invalid structural token")
        self.invalid_structural_token = int(token)

    def _outgoing_seg_map(
        self,
        prefix_length: int,
        history: Sequence[RLVRActionTrace],
    ) -> tuple[int, ...]:
        prompt = self.prompt_replay
        if prompt is None:
            raise RuntimeError("RLVR prompt trace is unavailable")
        prompt_prefix_length = len(prompt.prompt_token_ids) - 1
        if prefix_length < prompt_prefix_length:
            raise SFTGRPOError("RLVR action prefix truncates the real prompt")
        routing = [int(slot) for slot in prompt.prefix_seg_map[0].tolist()]
        routing.extend([0] * (prefix_length - prompt_prefix_length))
        previous_start = -1
        for index, action in enumerate(history):
            start = action.first_committed_position
            if action.decision_index != index:
                raise SFTGRPOError("RLVR action history decision indices must be contiguous")
            if start <= previous_start:
                raise SFTGRPOError(
                    "RLVR action history first committed positions must increase strictly"
                )
            if start < 0 or start >= prefix_length:
                raise SFTGRPOError(
                    "RLVR action history must already own a token in the canonical prefix"
                )
            slot = prompt.encoder_slot_count + index
            routing[start:] = [slot] * (prefix_length - start)
            previous_start = start
        return tuple(routing)

    def _append_from_fp_logits(
        self,
        state: SelfCondState,
        fp_logits_tail: t.Tensor,
        *,
        owner: str,
        max_time: int | None,
    ) -> SelfCondState:
        boundary = self._prepared_boundary
        prompt = self.prompt_replay
        if boundary is None or prompt is None:
            raise RuntimeError("RLVR policy boundary was not prepared by the native generator")
        self._prepared_boundary = None
        history_traces = tuple(self.actions)
        history = tuple(action.conditioning for action in history_traces)
        outgoing_seg_map = self._outgoing_seg_map(len(boundary.committed_prefix), history_traces)
        decision_index = len(self.actions)
        if decision_index == 0 and self.s0_logits_cache:
            policy_logits = self.s0_logits_cache[0]
        else:
            with t.no_grad():
                policy_logits = canonical_fp_logits(
                    self._model,
                    prompt,
                    committed_prefix=boundary.committed_prefix,
                    outgoing_seg_map=outgoing_seg_map,
                    pool_positions=boundary.pool_positions,
                    feature_history=history,
                    replay_cache=self.replay_cache,
                )
            if decision_index == 0:
                self.s0_logits_cache.append(policy_logits.detach())
        count_generator = t.Generator(device="cpu").manual_seed(
            _seed(self.group_seed, self.rollout_index, decision_index, "count")
        )
        positive_counts, negative_counts = sample_action_counts(
            self.policy_config.positive_counts,
            self.policy_config.negative_counts,
            count_generator,
            1,
        )
        action_generator = t.Generator(device=policy_logits.device).manual_seed(
            _seed(self.group_seed, self.rollout_index, decision_index, "action")
        )
        policy_trace = sample_weighted_without_replacement(
            policy_logits,
            positive_count=int(positive_counts[0]),
            negative_count=int(negative_counts[0]),
            generator=action_generator,
        )
        compact_to_original = (
            None
            if self._compact_candidate_feature_ids is None
            else tuple(int(value) for value in self._compact_candidate_feature_ids.tolist())
        )
        conditioning = trace_to_conditioning(
            policy_trace,
            compact_to_original=compact_to_original,
        )
        explicit = dict(zip(conditioning.feature_ids, conditioning.magnitudes, strict=True))
        self.queue_explicit_segment(explicit)
        new_state = super()._append_from_fp_logits(
            state,
            fp_logits_tail,
            owner=owner,
            max_time=max_time,
        )
        self.actions.append(
            RLVRActionTrace(
                decision_index=decision_index,
                source=boundary.source,
                committed_prefix=boundary.committed_prefix,
                outgoing_seg_map=outgoing_seg_map,
                pool_positions=boundary.pool_positions,
                first_committed_position=boundary.first_committed_position,
                feature_history=history,
                conditioning=conditioning,
                policy_trace=policy_trace,
            )
        )
        return new_state


def _same_prompt(left: RLVRPromptReplay, right: RLVRPromptReplay) -> bool:
    return (
        left.prompt_token_ids == right.prompt_token_ids
        and left.encoder_slot_count == right.encoder_slot_count
        and t.equal(left.enc_feature_ids, right.enc_feature_ids)
        and t.equal(left.enc_magnitudes, right.enc_magnitudes)
        and t.equal(left.enc_mask, right.enc_mask)
        and t.equal(left.prefix_seg_map, right.prefix_seg_map)
    )


def collect_rlvr_group(
    model: Any,
    engine_factory: EngineFactory,
    prompt_token_ids: Sequence[int],
    reward_fn: RewardFn,
    *,
    group_size: int,
    policy_config: GRPOPolicyConfig,
    group_seed: int,
    max_tokens: int,
    temperature: float = 1.0,
    top_k: int | None = None,
    extraction_tokenizer: Any,
    replay_cache: RLVRReplaySession | None = None,
) -> RLVRGroupRollout:
    """Collect a prompt-local group through fresh native scalar runtimes."""

    if type(group_size) is not int or group_size < 1:
        raise SFTGRPOError("RLVR group_size must be a positive integer")
    if type(max_tokens) is not int or max_tokens <= 0:
        raise SFTGRPOError("RLVR max_tokens must be a positive integer")
    if not math.isfinite(temperature) or temperature < 0:
        raise SFTGRPOError("RLVR temperature must be finite and non-negative")
    prompt_ids = tuple(int(token) for token in prompt_token_ids)
    if len(prompt_ids) < 2:
        raise SFTGRPOError("RLVR prompt must contain at least two tokens")
    if replay_cache is not None and not replay_cache.begin_collection(model):
        replay_cache = None
    trajectories: list[RLVRTrajectoryTrace] = []
    canonical_prompt: RLVRPromptReplay | None = None
    s0_logits_cache: list[t.Tensor] = []
    for rollout_index in range(group_size):
        engine = engine_factory(rollout_index)
        wrapper = engine._model
        if not isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper):
            raise TypeError("RLVR engine factory must return a target-aware encoder-decoder engine")
        if wrapper.model is not model:
            raise SFTGRPOError("RLVR engine factory must reuse the exact training model instance")
        if not wrapper.is_turn_self_conditioned or wrapper._self_cond is None:
            raise SFTGRPOError("RLVR requires turn-delimited self-conditioned generation")
        if wrapper._self_cond.state is not None:
            raise SFTGRPOError("RLVR engine factory must return a fresh runtime per trajectory")
        provider = RLVRFeatureProvider(
            model,
            wrapper._self_cond.config,
            wrapper.get_device(),
            policy_config=policy_config,
            group_seed=group_seed,
            rollout_index=rollout_index,
            s0_logits_cache=s0_logits_cache,
            replay_cache=replay_cache,
        )
        wrapper.swap_self_cond_provider(provider, wrapper._self_cond.config)
        wrapper._self_cond_prompt_trace_callback = provider.capture_prompt
        wrapper._self_cond_boundary_trace_callback = provider.prepare_boundary
        try:
            generation = generate_self_conditioned_chat(
                engine,
                list(prompt_ids),
                max_tokens=max_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=_token_seed(group_seed, rollout_index),
                extraction_tokenizer=extraction_tokenizer,
                trace_callback=provider,
            )
        finally:
            wrapper._self_cond_prompt_trace_callback = None
            wrapper._self_cond_boundary_trace_callback = None
        if generation.token_ids[: len(prompt_ids)] != prompt_ids:
            raise SFTGRPOError("native RLVR generation did not preserve the exact prompt prefix")
        if len(generation.token_ids) != len(generation.mask):
            raise SFTGRPOError("native RLVR generation returned misaligned tokens and mask")
        structurally_valid = provider.invalid_structural_token is None
        reward = 0.0 if not structurally_valid else float(reward_fn(generation))
        if reward not in {0.0, 1.0}:
            raise SFTGRPOError(f"RLVR reward must be exactly 0 or 1, got {reward!r}")
        if provider.prompt_replay is None:
            raise RuntimeError("native RLVR prefill did not publish raw prompt conditioning")
        if canonical_prompt is None:
            canonical_prompt = provider.prompt_replay
        elif not _same_prompt(canonical_prompt, provider.prompt_replay):
            raise SFTGRPOError("fresh RLVR runtimes reconstructed different prompt conditioning")
        trajectory = RLVRTrajectoryTrace(
            token_ids=generation.token_ids,
            sampled_mask=generation.mask,
            reward=reward,
            stop_reason=generation.stop_reason,
            actions=tuple(provider.actions),
            last_nonforced_logits_prefix_length=provider.last_nonforced_logits_prefix_length,
            structurally_valid=structurally_valid,
            invalid_structural_token=provider.invalid_structural_token,
        )
        if not trajectory.live_actions:
            raise SFTGRPOError("accepted RLVR trajectory contains no live feature action")
        trajectories.append(trajectory)
    assert canonical_prompt is not None
    rewards = t.tensor([trajectory.reward for trajectory in trajectories], dtype=t.float32)
    advantages = (
        t.zeros_like(rewards)
        if group_size == 1
        else conversation_group_advantages(rewards)
    )
    group = RLVRGroupRollout(
        prompt=canonical_prompt,
        trajectories=tuple(trajectories),
        advantages=tuple(float(value) for value in advantages.tolist()),
    )
    if replay_cache is not None:
        replay_cache.seal_collection(model, group)
    return group


def _checkpointed_canonical_fp_logits(
    model: Any,
    prompt: RLVRPromptReplay,
    action: RLVRActionTrace,
    *,
    train_feature_head_only: bool = False,
    replay_cache: RLVRReplaySession | None = None,
) -> t.Tensor:
    def replay(_sentinel: t.Tensor | None = None) -> t.Tensor:
        logits = canonical_fp_logits(
            model,
            prompt,
            committed_prefix=action.committed_prefix,
            outgoing_seg_map=action.outgoing_seg_map,
            pool_positions=action.pool_positions,
            feature_history=action.feature_history,
            train_feature_head_only=train_feature_head_only,
            replay_cache=replay_cache,
        )
        return logits if _sentinel is None else logits + _sentinel * 0.0

    if train_feature_head_only:
        # The canonical forward explicitly keeps the frozen trunk in no_grad.
        # Retain the head graph without checkpointing/recomputing that trunk.
        return replay()

    device = next(model.parameters()).device
    sentinel = t.ones((), dtype=t.float32, device=device, requires_grad=True)

    # Torch 2.13 stores compiler overrides in ContextVars that CUDA autograd
    # workers do not inherit. Fresh copies avoid leaks and concurrent entry.
    replay_context = copy_context()

    def replay_in_context(_sentinel: t.Tensor) -> t.Tensor:
        return replay_context.copy().run(replay, _sentinel)

    return checkpoint(
        replay_in_context,
        sentinel,
        use_reentrant=True,
        preserve_rng_state=False,
    )


def _validate_trajectory_history(trajectory: RLVRTrajectoryTrace) -> None:
    expected_history: list[ConditioningAction] = []
    previous_start = -1
    for index, action in enumerate(trajectory.actions):
        if action.decision_index != index:
            raise SFTGRPOError("RLVR action decision indices must be contiguous")
        if action.feature_history != tuple(expected_history):
            raise SFTGRPOError("RLVR action feature history must preserve exact insertion order")
        if action.first_committed_position <= previous_start:
            raise SFTGRPOError(
                "RLVR action first committed positions must increase strictly"
            )
        if action.first_committed_position < 0:
            raise SFTGRPOError("RLVR action first committed positions must be non-negative")
        if index and previous_start >= len(action.committed_prefix):
            raise SFTGRPOError(
                "RLVR action history contains a state that has not committed a prefix token"
            )
        expected_history.append(action.conditioning)
        previous_start = action.first_committed_position


def backward_rlvr_group(
    model: Any,
    group: RLVRGroupRollout,
    *,
    policy_config: GRPOPolicyConfig,
    loss_scale: float = 1.0,
    train_feature_head_only: bool = False,
    replay_cache: RLVRReplaySession | None = None,
) -> RLVRReplayMetrics:
    """Replay the length-normalized surrogate and backpropagate a scaled loss.

    ``RLVRReplayMetrics.loss`` remains the unscaled prompt-local loss.  The
    caller can pass its prompt-mean factor as ``loss_scale`` without changing
    behavioral replay checks or diagnostics.
    """

    if replay_cache is None:
        return _backward_rlvr_group(model, group, policy_config=policy_config,
            loss_scale=loss_scale, train_feature_head_only=train_feature_head_only)
    with replay_cache.replaying(model, group, train_feature_head_only=train_feature_head_only) as effective_cache:
        return _backward_rlvr_group(model, group, policy_config=policy_config,
            loss_scale=loss_scale, train_feature_head_only=train_feature_head_only, replay_cache=effective_cache)


def _backward_rlvr_group(
    model: Any, group: RLVRGroupRollout, *, policy_config: GRPOPolicyConfig,
    loss_scale: float, train_feature_head_only: bool, replay_cache: RLVRReplaySession | None = None,
) -> RLVRReplayMetrics:
    if model.training:
        raise SFTGRPOError("RLVR replay requires deterministic eval-mode model")
    if len(group.trajectories) != len(group.advantages) or not group.trajectories:
        raise SFTGRPOError("RLVR group trajectories and advantages must be nonempty and aligned")
    if any(not math.isfinite(advantage) for advantage in group.advantages):
        raise SFTGRPOError("RLVR advantages must be finite and detached")
    if not math.isfinite(loss_scale) or loss_scale <= 0:
        raise SFTGRPOError("RLVR loss_scale must be finite and positive")
    group_size = len(group.trajectories)
    ratios: list[t.Tensor] = []
    weighted_terms: list[t.Tensor] = []
    for trajectory in group.trajectories:
        _validate_trajectory_history(trajectory)

    # S0 has an identical state across G trajectories, so one differentiable
    # probe serves every independently sampled S0 action.
    if any(not trajectory.live_actions for trajectory in group.trajectories):
        raise SFTGRPOError("every RLVR replay trajectory must contain a live action")
    first_actions = [trajectory.live_actions[0] for trajectory in group.trajectories]
    if any(action.decision_index != 0 for action in first_actions):
        raise SFTGRPOError("every RLVR trajectory must begin with live S0")
    shared_logits = _checkpointed_canonical_fp_logits(
        model, group.prompt, first_actions[0], train_feature_head_only=train_feature_head_only,
        replay_cache=replay_cache,
    )
    for trajectory_index, (trajectory, advantage) in enumerate(
        zip(group.trajectories, group.advantages, strict=True)
    ):
        live_actions = trajectory.live_actions
        first = live_actions[0]
        if (
            first.committed_prefix != first_actions[0].committed_prefix
            or first.outgoing_seg_map != first_actions[0].outgoing_seg_map
            or first.pool_positions != first_actions[0].pool_positions
            or first.feature_history
        ):
            raise SFTGRPOError("RLVR S0 canonical state drifted within one prompt group")
        weight = -float(advantage) / (group_size * len(live_actions))
        for action_index, action in enumerate(live_actions):
            logits = (
                shared_logits
                if action_index == 0
                else _checkpointed_canonical_fp_logits(
                    model, group.prompt, action, train_feature_head_only=train_feature_head_only,
                    replay_cache=replay_cache,
                )
            )
            ratio = behavior_checked_replay_ratio(
                logits,
                action.policy_trace,
                atol=policy_config.replay_atol,
                rtol=policy_config.replay_rtol,
            )
            ratios.append(ratio)
            weighted_terms.append(ratio * weight)
        logger.debug(
            "RLVR replay trajectory {}: reward={}, advantage={}, live_actions={}",
            trajectory_index,
            trajectory.reward,
            advantage,
            len(live_actions),
        )
    loss = t.stack([term.reshape(()) for term in weighted_terms]).sum()
    (loss * loss_scale).backward()
    ratio_values = t.stack([ratio.detach().float().reshape(()) for ratio in ratios])
    return RLVRReplayMetrics(
        loss=float(loss.detach()),
        ratio_min=float(ratio_values.min()),
        ratio_max=float(ratio_values.max()),
        live_decisions=len(ratios),
    )


_REPLAY_DEPENDENCIES = tuple(
    NativeBinding.capture(sys.modules[__name__], name, value)
    for name, value in (
        ("canonical_fp_logits", canonical_fp_logits), ("_encode_canonical_state", _encode_canonical_state),
        ("_action_tensors", _action_tensors), ("append_packed_encoder_segment", append_packed_encoder_segment),
        ("pool_fp_logits", pool_fp_logits), ("checkpoint", checkpoint),
    )
) + tuple(
    NativeBinding.capture(packed, name, value)
    for name, value in (("PackedEncoderOutput", packed.PackedEncoderOutput),
                        ("PackedEncoderLayout", packed.PackedEncoderLayout))
)


@contextmanager
def rlvr_group_replay_cache(
    model: Any, *, settings: HiddenReplaySettings, train_feature_head_only: bool,
    world_size: int, report: HiddenReplayReport,
) -> Iterator[RLVRReplaySession | None]:
    """Own one training group's optional hidden inputs and bounded worker counts."""
    if type(train_feature_head_only) is not bool or type(world_size) is not int or world_size < 1:
        raise ValueError("Hidden replay requires boolean head-only scope and positive integer world size")
    report.requested_groups += 1
    reason = ("disabled" if settings.max_cache_bytes == 0 else "full_policy" if not train_feature_head_only
              else "unsupported_world_size" if world_size != 1 else None)
    if reason is not None:
        report.inactive_groups += 1
        _bump(report.inactive_reasons, reason)
        yield None
        return
    session = RLVRReplaySession(model, settings=settings, report=report,
        prompt_type=RLVRPromptReplay, group_type=RLVRGroupRollout, policy_dependencies=_REPLAY_DEPENDENCIES)
    completed = False
    try:
        yield session
        completed = True
    except BaseException:
        session.mark_failed()
        raise
    finally:
        session.close()
        if completed and not session.failed and (report.active_groups != 0 or report.current_bytes != 0
                          or report.owned_tensor_objects_alive_after_cleanup != 0):
            raise RuntimeError("Hidden replay retained group state after successful completion")


__all__ = [
    "RLVRActionTrace",
    "RLVRFeatureProvider",
    "RLVRGroupRollout",
    "RLVRPromptReplay",
    "RLVRReplayMetrics",
    "RLVRTrajectoryTrace",
    "backward_rlvr_group",
    "canonical_fp_logits",
    "collect_rlvr_group",
    "rlvr_group_replay_cache",
]
