"""Canonical turn-RLVR collection and replay on the real tiny encoder-decoder."""

from __future__ import annotations

import copy
from collections import deque
from dataclasses import replace
from datetime import timedelta
from pathlib import Path
from time import perf_counter
from typing import Any

import pytest
import torch as t
import torch.distributed as dist

import nanochat.flash_attention as flash_attention
import steering.conditional.encoder_decoder.rlvr_policy as rlvr_policy
import steering.eval.chat_self_conditioned_generation as chat_generation
from nanochat.engine import RowState, use_calculator
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import OptimizerKind
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.cross_attention import CrossAttentionLayer
from steering.conditional.encoder_decoder.encoder import FeatureEncoder
from steering.conditional.encoder_decoder.rlvr_policy import (
    RLVRActionTrace,
    RLVRFeatureProvider,
    RLVRGroupRollout,
    RLVRPromptReplay,
    RLVRTrajectoryTrace,
    backward_rlvr_group,
    canonical_fp_logits,
    collect_rlvr_group,
)
from steering.conditional.encoder_decoder.rlvr_runtime import configure_rlvr_compiler
from steering.conditional.encoder_decoder.self_conditioned import (
    self_conditioned_decode_step,
)
from steering.conditional.encoder_decoder.sft_grpo import (
    SFTGRPOError,
    replay_ratio,
    sample_weighted_without_replacement,
    trace_to_conditioning,
    weighted_ordered_log_likelihood,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    assert_grpo_gradients_ready,
    configure_grpo_trainable_parameters,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.chat_self_conditioned_generation import ChatSelfConditionedGeneration
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.sft.turn_spans import ChatSpecialTokens, TurnRole
from tests.encdec_test_helpers import build_tiny_model
from tests.gloo_two_rank_helpers import run_two_ranks
from tests.padded_encoder_reference import (
    cross_attention_forward_with_cpu_reference,
    feature_encoder_forward_packed_with_cpu_reference,
    feature_encoder_forward_with_cpu_reference,
    use_test_owned_cpu_encoder_reference,
)
from tests.steering.selfcond.rlvr_test_helpers import factory as _factory
from tests.steering.selfcond.rlvr_test_helpers import policy as _policy
from tests.steering.selfcond.rlvr_test_helpers import script as _script
from tests.steering.test_chat_self_conditioned_generation import instrument, make_fixture


class _FilteredContextProvider:
    """Preserve the real turn cache while admitting one compact vocabulary."""

    def __init__(self, provider: Any, allowed: tuple[int, ...], turn_contract: Any) -> None:
        self._provider = provider
        self._allowed = frozenset(allowed)
        self.turn_contract = turn_contract

    def __getattr__(self, name: str) -> Any:
        return getattr(self._provider, name)

    def get(self, token_ids):
        segments = self._provider.get(token_ids)
        if segments is None:
            return None
        filtered = []
        for segment in segments:
            pairs = [
                (feature_id, bucket)
                for feature_id, bucket in zip(
                    segment.feature_ids,
                    segment.buckets,
                    strict=True,
                )
                if feature_id in self._allowed
            ]
            assert pairs
            filtered.append(
                ContextFeatureSegment(
                    feature_ids=tuple(feature_id for feature_id, _bucket in pairs),
                    buckets=tuple(bucket for _feature_id, bucket in pairs),
                    char_start=segment.char_start,
                    char_end=segment.char_end,
                )
            )
        return tuple(filtered)


class _FrozenMainTurnTransition:
    """Turn transition copied from current-main before the RLVR trace seam."""

    def __init__(self, wrapper, extraction_tokenizer) -> None:
        if wrapper.turn_contract is None or wrapper._self_cond is None:
            raise ValueError
        self.wrapper = wrapper
        self.runtime = wrapper._self_cond
        self.contract = wrapper.turn_contract
        self.extraction_tokenizer = extraction_tokenizer
        self.specials = ChatSpecialTokens.from_tokenizer(wrapper.tokenizer)
        self.python_ids = frozenset(
            wrapper.tokenizer.encode_special(name)
            for name in ("<python_start>", "<python_end>")
        )
        self.role = TurnRole.ASSISTANT
        self.plans: deque[tuple[int, ...]] = deque()
        self.tool_rows: list[t.Tensor] = []

    def plan_output(self, prefix: list[int], forced_tokens: tuple[int, ...]) -> None:
        self.plans.extend(
            chat_generation.plan_tool_windows(
                prefix,
                forced_tokens,
                tokenizer=self.wrapper.tokenizer,
                extraction_tokenizer=self.extraction_tokenizer,
                contract=self.contract,
            )
        )

    def commit(self, token: int, kv_cache) -> t.Tensor:
        runtime, device = self.runtime, self.wrapper.get_device()
        state = runtime.state
        if state is None:
            raise RuntimeError
        ids = t.tensor([[token]], dtype=t.long, device=device)
        entering_tool = token == self.specials.output_start and self.role is TurnRole.ASSISTANT
        if entering_tool and not self.plans:
            raise ValueError
        if entering_tool or (self.role is TurnRole.TOOL and self.plans):
            plan = self.plans[0]
            if token != plan[len(self.tool_rows)]:
                raise ValueError
            if entering_tool and runtime.fp_buffer:
                state = runtime.provider.step(state, t.cat(runtime.fp_buffer, dim=1))
            logits, fp = self.wrapper.model(
                ids,
                None,
                seg_map=t.full(
                    (1, 1),
                    state.target_slot_idx,
                    device=device,
                    dtype=t.long,
                ),
                kv_cache=kv_cache,
                encoder_out=state.encoder_out.to(device),
                encoder_strict=True,
                return_fp_logits=True,
            )
            self.tool_rows.append(fp.detach())
            if len(self.tool_rows) == len(plan):
                state = runtime.provider.append_tool_window(
                    state,
                    t.cat(self.tool_rows, dim=1),
                )
                self.tool_rows.clear()
                self.plans.popleft()
            runtime.state, runtime.fp_buffer, runtime.tokens_in_seg = state, (), 0
            self.role = TurnRole.TOOL
            return logits
        if token in self.wrapper.generation_stop_token_ids and token not in self.python_ids:
            raise ValueError
        result = self_conditioned_decode_step(
            model=self.wrapper.model,
            provider=runtime.provider,
            state=state,
            fp_buffer=runtime.fp_buffer,
            tokens_in_seg=runtime.tokens_in_seg,
            emitted_token=ids,
            kv_cache=kv_cache,
            device=device,
        )
        runtime.state = result.state
        runtime.fp_buffer = result.fp_buffer
        runtime.tokens_in_seg = result.tokens_in_seg
        self.role = TurnRole.ASSISTANT
        return result.logits


@t.inference_mode()
def _frozen_main_generate(
    engine,
    token_ids: list[int],
    *,
    max_tokens: int,
    extraction_tokenizer,
) -> ChatSelfConditionedGeneration:
    """Exact no-hook current-main scalar generator used as an independent oracle."""

    wrapper = engine._model
    budget = min(max_tokens, wrapper.config.sequence_len - len(token_ids))
    if budget == 0:
        return ChatSelfConditionedGeneration(tuple(token_ids), (0,) * len(token_ids))
    transition = _FrozenMainTurnTransition(wrapper, extraction_tokenizer)
    prefill = engine._prepare_self_conditioned_prefill_impl(
        token_ids,
        max_new_tokens=budget,
    )
    logits, kv_cache = prefill.logits, prefill.kv_cache
    rng = t.Generator(device=engine._device).manual_seed(1234)
    row = RowState(token_ids.copy())
    masks = [0] * len(token_ids)
    python_start, python_end, output_start, output_end, assistant_end = (
        engine._tokenizer.encode_special(name)
        for name in (
            "<python_start>",
            "<python_end>",
            "<output_start>",
            "<output_end>",
            "<assistant_end>",
        )
    )
    terminal_ids = {assistant_end, engine._tokenizer.get_bos_token_id()}
    for index in range(budget):
        sampled = int(chat_generation.sample_next_token(logits, rng, 0.0, None).item())
        forced = bool(row.forced_tokens)
        token = row.forced_tokens.popleft() if forced else sampled
        if token in terminal_ids:
            break
        row.current_tokens.append(token)
        masks.append(0 if forced else 1)
        if token == python_start:
            row.in_python_block, row.python_expr_tokens = True, []
        elif token == python_end and row.in_python_block:
            row.in_python_block = False
            if row.python_expr_tokens:
                result = use_calculator(engine._tokenizer.decode(row.python_expr_tokens))
                if result is not None:
                    output = (
                        output_start,
                        *engine._tokenizer.encode(str(result)),
                        output_end,
                    )
                    row.forced_tokens.extend(output)
                    if index + 1 < budget:
                        transition.plan_output(row.current_tokens, output)
            row.python_expr_tokens = []
        elif row.in_python_block:
            row.python_expr_tokens.append(token)
        if index + 1 == budget:
            break
        logits = transition.commit(token, kv_cache)[:, -1, :]
    return ChatSelfConditionedGeneration(tuple(row.current_tokens), tuple(masks))


def _install_cpu_model_reference() -> None:
    flash_attention._override_impl = FlashAttnImpl.SDPA
    setattr(FeatureEncoder, "forward", feature_encoder_forward_with_cpu_reference)  # noqa: B010
    setattr(  # noqa: B010
        FeatureEncoder,
        "forward_packed",
        feature_encoder_forward_packed_with_cpu_reference,
    )
    setattr(  # noqa: B010
        CrossAttentionLayer,
        "forward",
        cross_attention_forward_with_cpu_reference,
    )


def _gloo_model(*, head_layers: int = 0, block: str = "mlp") -> Any:
    t.manual_seed(737)
    model = build_tiny_model(
        n_sae_features=8,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            head_layers=head_layers,
            block=FeaturePredictionBlock(block),
        ),
        n_layer=1,
        seq_len=16,
        xattn_frequency=1,
        vocab_size=32,
    )
    return model.eval()


def _gloo_group(model: Any, rank: int) -> RLVRGroupRollout:
    prompt = RLVRPromptReplay(
        prompt_token_ids=(1, 2, 3, 4),
        enc_feature_ids=t.tensor([[[1, 5]]], dtype=t.long),
        enc_magnitudes=t.tensor([[[0, 1]]], dtype=t.long),
        enc_mask=t.tensor([[[True, True]]]),
        prefix_seg_map=t.tensor([[0, 0, 0]], dtype=t.long),
        encoder_slot_count=1,
    )
    with t.no_grad():
        logits = canonical_fp_logits(
            model,
            prompt,
            committed_prefix=(1, 2, 3),
            outgoing_seg_map=(0, 0, 0),
            pool_positions=(1, 2),
            feature_history=(),
        )
    trajectories = []
    for rollout_index in range(2):
        trace = sample_weighted_without_replacement(
            logits,
            positive_count=1,
            negative_count=1,
            generator=t.Generator(device=logits.device).manual_seed(
                1000 + rank * 17 + rollout_index
            ),
        )
        action = RLVRActionTrace(
            decision_index=0,
            source="prompt",
            committed_prefix=(1, 2, 3),
            outgoing_seg_map=(0, 0, 0),
            pool_positions=(1, 2),
            first_committed_position=2,
            feature_history=(),
            conditioning=trace_to_conditioning(trace, compact_to_original=None),
            policy_trace=trace,
        )
        trajectories.append(
            RLVRTrajectoryTrace(
                token_ids=(1, 2, 3, 4 + rollout_index),
                sampled_mask=(0, 0, 0, 1),
                reward=float(rollout_index == 0),
                stop_reason="eos",
                actions=(action,),
                last_nonforced_logits_prefix_length=3,
                structurally_valid=True,
                invalid_structural_token=None,
            )
        )
    return RLVRGroupRollout(
        prompt=prompt,
        trajectories=tuple(trajectories),
        advantages=(1.0, -1.0),
    )


def _gradient_bytes(model: Any, names: tuple[str, ...]) -> bytes:
    flat = t.cat(
        [
            dict(model.named_parameters())[name].grad.detach().float().reshape(-1)
            for name in names
        ]
    )
    return flat.numpy().tobytes()


def _gloo_policy_worker(rank, world_size, init_file, result_queue) -> None:
    _install_cpu_model_reference()
    dist.init_process_group(
        backend="gloo",
        init_method=Path(init_file).as_uri(),
        rank=rank,
        world_size=world_size,
        timeout=timedelta(seconds=60),
    )
    try:
        model = _gloo_model()
        names = configure_grpo_trainable_parameters(model, train_lm_head=False)
        backward_rlvr_group(
            model,
            _gloo_group(model, rank),
            policy_config=_policy(),
            loss_scale=1.0 / world_size,
        )
        parameters = dict(model.named_parameters())
        for name in names:
            assert parameters[name].grad is not None
            dist.all_reduce(parameters[name].grad, op=dist.ReduceOp.SUM)
        result_queue.put((rank, names, _gradient_bytes(model, names)))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("head_depth", [0, 1, 2])
def test_collection_replays_exact_canonical_policy_and_freezes_lm_head(
    tmp_path, monkeypatch: pytest.MonkeyPatch, head_depth
) -> None:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    fixture = make_fixture(tmp_path, monkeypatch, K=2, head_depth=head_depth)
    good = fixture.tokenizer.encode("x")[0]
    bad = fixture.tokenizer.encode("y")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [good, end, bad, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda result: float(result.token_ids[len(fixture.prompt)] == good),
        group_size=2,
        policy_config=_policy(),
        group_seed=991,
        max_tokens=2,
        extraction_tokenizer=fixture.source,
    )

    assert [trajectory.reward for trajectory in group.trajectories] == [1.0, 0.0]
    assert group.advantages == pytest.approx((1.0, -1.0))
    assert all(len(trajectory.actions) == 2 for trajectory in group.trajectories)
    # S1 was formed after the final committed token, but no non-forced logits
    # were ever evaluated under it.  It remains in history and is excluded from
    # the length-normalized surrogate.
    assert all(len(trajectory.live_actions) == 1 for trajectory in group.trajectories)
    first = group.trajectories[0].actions[0]
    with t.no_grad():
        replay = canonical_fp_logits(
            fixture.model,
            group.prompt,
            committed_prefix=first.committed_prefix,
            outgoing_seg_map=first.outgoing_seg_map,
            pool_positions=first.pool_positions,
            feature_history=first.feature_history,
        )
    assert replay.shape == (8,)

    # The policy law never reads token logits: even an arbitrary frozen
    # unembedding produces the same feature-policy loss and every policy grad.
    altered_head = copy.deepcopy(fixture.model).eval()
    with t.no_grad():
        altered_head.lm_head.weight.normal_(mean=17.0, std=3.0)
    manifest = configure_grpo_trainable_parameters(fixture.model, train_lm_head=False)
    altered_manifest = configure_grpo_trainable_parameters(
        altered_head,
        train_lm_head=False,
    )
    metrics = backward_rlvr_group(fixture.model, group, policy_config=_policy())
    altered_metrics = backward_rlvr_group(
        altered_head,
        group,
        policy_config=_policy(),
    )
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
    assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
    assert metrics.live_decisions == 2
    assert fixture.model.lm_head.weight.requires_grad is False
    assert fixture.model.lm_head.weight.grad is None
    assert all(
        parameter.grad is not None and bool(t.isfinite(parameter.grad).all())
        for name, parameter in fixture.model.named_parameters()
        if name in manifest
    )
    assert any(
        bool(parameter.grad.ne(0).any())
        for name, parameter in fixture.model.named_parameters()
        if name in manifest and parameter.grad is not None
    )

    assert altered_metrics == metrics
    original_parameters = dict(fixture.model.named_parameters())
    altered_parameters = dict(altered_head.named_parameters())
    assert altered_manifest == manifest
    for name in manifest:
        assert t.allclose(
            original_parameters[name].grad,
            altered_parameters[name].grad,
            atol=2e-6,
            rtol=2e-5,
        ), name

    optimizer = fixture.model.setup_optimizer(trainable_only=True)
    owned = [id(p) for group in optimizer.param_groups for p in group["params"]]
    assert len(owned) == len(set(owned))
    assert set(owned) == {id(original_parameters[name]) for name in manifest}
    head_names = [name for name in manifest if name.startswith("feature_pred_head.")]
    assert len(head_names) == 2 + 2 * head_depth
    for name in head_names:
        assert t.count_nonzero(original_parameters[name].grad) > 0, name
    for prefix in ("encoder.", "transformer.", "cross_attn"):
        assert any(
            name.startswith(prefix) and t.count_nonzero(original_parameters[name].grad) > 0
            for name in manifest
        ), prefix
    before = {name: p.detach().clone() for name, p in original_parameters.items()}
    with t.compiler.set_stance("force_eager"):
        optimizer.step()
    for name in head_names:
        assert not t.equal(original_parameters[name], before[name]), name
    assert t.equal(fixture.model.lm_head.weight, before["lm_head.weight"])


def test_head_only_replay_matches_default_gradients_without_recomputing_trunk(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end, token, token, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda result: float(len(result.token_ids) == len(fixture.prompt)),
        group_size=2,
        policy_config=_policy(),
        group_seed=8,
        max_tokens=3,
        extraction_tokenizer=fixture.source,
    )
    assert [len(trajectory.live_actions) for trajectory in group.trajectories] == [1, 2]
    results = []
    for scope in (None, False, True):
        model = copy.deepcopy(fixture.model)
        manifest = configure_grpo_trainable_parameters(
            model, train_lm_head=False, train_feature_head_only=scope is True
        )
        head_inputs = []
        encoder_modes, decoder_modes = [], []

        def observe_head(_module, inputs, *, observed=head_inputs):
            observed.append(
                (t.is_grad_enabled(), inputs[0].requires_grad, inputs[0].grad_fn is None)
            )

        kwargs: dict[str, bool] = {} if scope is None else {"train_feature_head_only": scope}
        with (
            model.feature_pred_head.register_forward_pre_hook(observe_head),
            model.encoder.register_forward_pre_hook(
                lambda _module, _inputs, observed=encoder_modes: observed.append(t.is_grad_enabled())
            ),
            model._decoder_blocks[-1].register_forward_pre_hook(
                lambda _module, _inputs, observed=decoder_modes: observed.append(t.is_grad_enabled())
            ),
        ):
            metrics = backward_rlvr_group(model, group, policy_config=_policy(), **kwargs)
        assert_grpo_gradients_ready(model, expected_manifest=manifest)
        assert head_inputs == (
            [(True, False, True)] * 2
            if scope
            else [(False, False, True)] * 2 + [(True, True, False)] * 2
        )
        # S0 encodes the prompt; S1 also encodes its one historical action.
        # Default replay also recomputes its checkpointed historical encoding.
        assert encoder_modes == ([False] * 3 if scope else [False] * 3 + [True] * 4)
        assert decoder_modes == ([False] * 2 if scope else [False] * 2 + [True] * 2)
        gradients = {}
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                assert parameter.grad is not None
                gradients[name] = parameter.grad.detach().clone()
            else:
                assert parameter.grad is None, name
        results.append((metrics, gradients))
    default_metrics, default_gradients = results[0]
    for metrics, gradients in results[1:]:
        assert metrics == default_metrics
        for name, gradient in gradients.items():
            assert t.equal(gradient, default_gradients[name]), name
    assert results[1][1].keys() == default_gradients.keys()
    assert set(results[2][1]) == {"feature_pred_head.weight", "feature_pred_head.bias"}
    assert all(bool(gradient.any()) for gradient in results[2][1].values())


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("block_kind", ["mlp", "transformer"])
def test_head_only_replay_preserves_no_grad_trunk_outputs_and_head_gradients(monkeypatch, depth, block_kind):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    with (
        use_test_owned_cpu_encoder_reference(),
        t.autocast("cpu", dtype=t.bfloat16, cache_enabled=False),
    ):
        model = _gloo_model(head_layers=depth, block=block_kind)
        configure_grpo_trainable_parameters(
            model, train_lm_head=False, train_feature_head_only=True
        )
        block = model._decoder_blocks[-1]
        original_forward = block.forward
        trunk_modes = []

        def mode_sensitive_forward(*args, **kwargs):
            hidden = original_forward(*args, **kwargs)
            trunk_modes.append(t.is_grad_enabled())
            # Model the observed Hopper specialization difference without a
            # GPU dependency. A nonuniform shift survives the final RMS norm.
            if t.is_grad_enabled():
                hidden = hidden + t.arange(hidden.shape[-1], dtype=hidden.dtype)
            return hidden

        monkeypatch.setattr(block, "forward", mode_sensitive_forward)
        captured_inputs = []
        with model.feature_pred_head.register_forward_pre_hook(
            lambda _module, inputs: captured_inputs.append(inputs[0].detach().clone())
        ):
            group = _gloo_group(model, 0)
        assert trunk_modes == [False]
        assert len(captured_inputs) == 1
        action = group.trajectories[0].live_actions[0]
        reference_head = copy.deepcopy(model.feature_pred_head)
        # Independent oracle: apply only the complete head to its actual
        # behavior-time normalized input, with no trunk replay or readout seam.
        head_kwargs = {} if not depth else {
            "cos_sin": (model.cos[:, :captured_inputs[0].shape[1]], model.sin[:, :captured_inputs[0].shape[1]]),
            "window_size": model.window_sizes[model._feature_pred_layer_idx],
        }
        reference_logits = reference_head(captured_inputs[0], **head_kwargs)[:, action.pool_positions, :].amax(1)[
            0
        ]
        trunk_modes.clear()
        actual_logits = rlvr_policy._checkpointed_canonical_fp_logits(
            model, group.prompt, action, train_feature_head_only=True
        )
        t.testing.assert_close(actual_logits, reference_logits, rtol=0, atol=0)
        weighted_ordered_log_likelihood(actual_logits, action.policy_trace).backward()
        weighted_ordered_log_likelihood(reference_logits, action.policy_trace).backward()
        for (name, parameter), (_, expected) in zip(
            model.feature_pred_head.named_parameters(),
            reference_head.named_parameters(),
            strict=True,
        ):
            assert parameter.grad is not None and expected.grad is not None, name
            t.testing.assert_close(parameter.grad, expected.grad, rtol=0, atol=0)
        assert trunk_modes == [False]
        assert all(
            parameter.grad is None
            for parameter in model.parameters()
            if not parameter.requires_grad
        )


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("block_kind", ["mlp", "transformer"])
def test_head_only_optimizer_updates_complete_nested_head(monkeypatch, depth, block_kind):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    # Exercise BF16 without retaining cached weight casts across optimizer updates.
    with (
        use_test_owned_cpu_encoder_reference(),
        t.autocast("cpu", dtype=t.bfloat16, cache_enabled=False),
    ):
        model = _gloo_model(head_layers=depth, block=block_kind)
        initial = copy.deepcopy(model.state_dict())
        manifest = configure_grpo_trainable_parameters(
            model, train_lm_head=False, train_feature_head_only=True
        )
        optimizer = model.setup_optimizer(trainable_only=True)
        head_ids = {id(parameter) for parameter in model.feature_pred_head.parameters()}
        grouped = [
            parameter for group in optimizer.param_groups for parameter in group["params"]
        ]
        assert len(grouped) == len(head_ids)
        assert {id(parameter) for parameter in grouped} == head_ids
        if depth:
            block_matrix_ids = {
                id(parameter)
                for parameter in model.feature_pred_head.residual_blocks.parameters()
                if parameter.ndim == 2
            }
            assert {
                group["kind"]
                for group in optimizer.param_groups
                if any(id(parameter) in block_matrix_ids for parameter in group["params"])
            } == {OptimizerKind.MUON}
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            metrics = backward_rlvr_group(
                model, _gloo_group(model, 0), policy_config=_policy(), train_feature_head_only=True
            )
            assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
            assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
            assert_grpo_gradients_ready(model, expected_manifest=manifest)
            if depth:
                for block in model.feature_pred_head.residual_blocks:
                    mlp = block.mlp if block_kind == "transformer" else block
                    if block_kind == "transformer":
                        assert block.attn.c_proj.weight.grad is not None and block.attn.c_proj.weight.grad.any()
                        assert bool(block.attn.c_qkv.weight.grad.any()) is (step > 0)
                    assert mlp.c_proj.weight.grad is not None
                    assert mlp.c_fc.weight.grad is not None
                    assert bool(mlp.c_proj.weight.grad.any())
                    assert bool(mlp.c_fc.weight.grad.any()) is (step > 0)
            optimizer.step()
            for name, parameter in model.named_parameters():
                assert bool(t.isfinite(parameter).all()), name
                if not parameter.requires_grad:
                    assert parameter.grad is None and parameter not in optimizer.state
                    assert t.equal(parameter, initial[name]), name
        expected_state_parameters = {
            parameter
            for group in optimizer.param_groups
            if group["kind"] == OptimizerKind.ADAMW
            for parameter in group["params"]
        }
        expected_state_parameters.update(
            group["params"][0]
            for group in optimizer.param_groups
            if group["kind"] == OptimizerKind.MUON
        )
        assert set(optimizer.state) == expected_state_parameters
        assert all(
            not t.equal(parameter, initial[name])
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA replay qualification")
def test_cuda_head_only_replay_matches_full_and_reports_step_time(monkeypatch, record_property):
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    for key in (
        "recompile_limit",
        "accumulated_recompile_limit",
        "fail_on_recompile_limit_hit",
        "suppress_errors",
    ):
        monkeypatch.setattr(t._dynamo.config, key, getattr(t._dynamo.config, key))
    source = _gloo_model().to("cuda")
    source.init_weights()
    configure_rlvr_compiler(source)
    initial = copy.deepcopy(source.state_dict())
    first_gradients = []
    for head_only in (False, True):
        model = copy.deepcopy(source)
        manifest = configure_grpo_trainable_parameters(
            model, train_lm_head=False, train_feature_head_only=head_only
        )
        optimizer = model.setup_optimizer(trainable_only=True)
        for step in range(3):
            group = _gloo_group(model, 0)
            optimizer.zero_grad(set_to_none=True)
            t.cuda.synchronize()
            start = perf_counter()
            metrics = backward_rlvr_group(
                model, group, policy_config=_policy(), train_feature_head_only=head_only
            )
            assert_grpo_gradients_ready(model, expected_manifest=manifest)
            if step == 0:
                first_gradients.append(
                    tuple(
                        parameter.grad.detach().clone()
                        for parameter in model.feature_pred_head.parameters()
                    )
                )
            optimizer.step()
            t.cuda.synchronize()
            elapsed = perf_counter() - start
            assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
            assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
        label = "head_only" if head_only else "full"
        record_property(f"{label}_step_seconds", elapsed)
        print(f"RLVR {label} warmed replay/optimizer step: {elapsed:.6f}s")
        for name, parameter in model.named_parameters():
            assert bool(t.isfinite(parameter).all()), name
            if not parameter.requires_grad:
                assert parameter.grad is None and parameter not in optimizer.state
                assert t.equal(parameter, initial[name]), name
        assert any(
            not t.equal(parameter, initial[name])
            for name, parameter in model.named_parameters()
            if name.startswith("feature_pred_head.")
        )
    t.testing.assert_close(first_gradients[1], first_gradients[0], rtol=0, atol=0)


def test_collection_clears_trace_callbacks_after_success(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end])

    collect_rlvr_group(
        fixture.model,
        lambda _rollout: fixture.engine,
        fixture.prompt,
        lambda _generation: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=7,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )

    assert fixture.engine._model._self_cond_prompt_trace_callback is None
    assert fixture.engine._model._self_cond_boundary_trace_callback is None


def test_collection_clears_trace_callbacks_when_generation_raises(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    wrapper = fixture.engine._model

    def fail_generation(engine, *_args, **_kwargs):
        assert engine._model._self_cond_prompt_trace_callback is not None
        assert engine._model._self_cond_boundary_trace_callback is not None
        raise RuntimeError("injected native generation failure")

    monkeypatch.setattr(
        rlvr_policy,
        "generate_self_conditioned_chat",
        fail_generation,
    )
    with pytest.raises(RuntimeError, match="injected native generation failure"):
        collect_rlvr_group(
            fixture.model,
            lambda _rollout: fixture.engine,
            fixture.prompt,
            lambda _generation: 0.0,
            group_size=1,
            policy_config=_policy(),
            group_seed=7,
            max_tokens=1,
            extraction_tokenizer=fixture.source,
        )

    assert wrapper._self_cond_prompt_trace_callback is None
    assert wrapper._self_cond_boundary_trace_callback is None


def test_liveness_uses_strict_prefix_inequality() -> None:
    trace = sample_weighted_without_replacement(
        t.zeros(4),
        positive_count=1,
        negative_count=1,
        generator=t.Generator().manual_seed(1),
    )
    action = RLVRActionTrace(
        decision_index=0,
        source="prompt",
        committed_prefix=(1, 2, 3, 4, 5),
        outgoing_seg_map=(0, 0, 0, 0, 0),
        pool_positions=(4,),
        first_committed_position=5,
        feature_history=(),
        conditioning=trace_to_conditioning(trace, compact_to_original=None),
        policy_trace=trace,
    )
    assert action.is_live(5) is False
    assert action.is_live(6) is True


def test_invalid_structural_sample_is_terminal_zero_and_suffix_is_preserved(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    invalid = fixture.tokenizer.encode_special("<user_start>")
    _script(monkeypatch, [invalid])
    grader_called = False

    def grade(_result):
        nonlocal grader_called
        grader_called = True
        return 1.0

    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        grade,
        group_size=1,
        policy_config=_policy(),
        group_seed=22,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )
    trajectory = group.trajectories[0]
    assert trajectory.reward == 0.0
    assert trajectory.structurally_valid is False
    assert trajectory.invalid_structural_token == invalid
    assert trajectory.token_ids == (*fixture.prompt, invalid)
    assert grader_called is False


def test_k1_marker_boundary_is_traced_before_generation(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=1)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [token, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 1.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=23,
        max_tokens=2,
        extraction_tokenizer=fixture.source,
    )
    trajectory = group.trajectories[0]
    assert [action.source for action in trajectory.actions[:2]] == ["prompt", "assistant"]
    assert trajectory.actions[1].committed_prefix == (*fixture.prompt,)
    assert len(trajectory.live_actions) == 2


@pytest.mark.parametrize("segment_size", [1, 2, 33])
def test_tool_window_actions_preserve_forced_suffix_and_causal_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch, segment_size: int
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=segment_size)
    tok = fixture.tokenizer
    special = tok.encode_special
    assistant = [special("<python_start>"), *tok.encode("1+1"), special("<python_end>")]
    forced = [special("<output_start>"), *tok.encode("2"), special("<output_end>")]
    resumed = tok.encode(" done")
    script = [*assistant, *forced, *resumed, special("<assistant_end>")]
    _script(monkeypatch, script)
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 1.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=313,
        max_tokens=len(script),
        extraction_tokenizer=fixture.source,
    )
    trajectory = group.trajectories[0]
    assert trajectory.stop_reason == "eos"
    assert trajectory.token_ids == (*fixture.prompt, *assistant, *forced, *resumed)
    assert trajectory.sampled_mask[len(fixture.prompt) + len(assistant) :][
        : len(forced)
    ] == (0,) * len(forced)
    assert "tool" in {action.source for action in trajectory.actions}
    tool_actions = [action for action in trajectory.actions if action.source == "tool"]
    assert tool_actions and all(action in trajectory.live_actions for action in tool_actions)
    for action in trajectory.actions:
        # The action being sampled cannot route any token in its own state.
        assert max(action.outgoing_seg_map) < (
            group.prompt.encoder_slot_count + action.decision_index
        )
        assert all(position < len(action.committed_prefix) for position in action.pool_positions)


def test_budget_inside_forced_tool_suffix_keeps_tokens_without_tool_action(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    tok = fixture.tokenizer
    special = tok.encode_special
    assistant = [special("<python_start>"), *tok.encode("1+1"), special("<python_end>")]
    forced = [special("<output_start>"), *tok.encode("2"), special("<output_end>")]
    # One forced token fits after python_end; the next budget position is never
    # committed, but its dummy categorical draw is still consumed natively.
    budget = len(assistant) + 1
    _script(monkeypatch, [*assistant, *forced])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=314,
        max_tokens=budget,
        extraction_tokenizer=fixture.source,
    )
    trajectory = group.trajectories[0]
    assert trajectory.stop_reason == "budget"
    assert trajectory.token_ids == (*fixture.prompt, *assistant, forced[0])
    assert trajectory.sampled_mask[-1] == 0
    assert not any(action.source == "tool" for action in trajectory.actions)


def test_collection_rejects_zero_budget_before_building_an_engine() -> None:
    called = False

    def factory(_index):
        nonlocal called
        called = True
        raise AssertionError

    with pytest.raises(SFTGRPOError, match="max_tokens must be a positive"):
        collect_rlvr_group(
            object(),
            factory,
            (1, 2),
            lambda _result: 0.0,
            group_size=1,
            policy_config=_policy(),
            group_seed=0,
            max_tokens=0,
            extraction_tokenizer=object(),
        )
    assert called is False


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"group_size": 0}, "group_size"),
        ({"temperature": float("nan")}, "temperature"),
        ({"temperature": -0.1}, "temperature"),
        ({"prompt_token_ids": (1,)}, "at least two tokens"),
    ],
)
def test_collection_rejects_invalid_public_inputs_before_engine(
    kwargs: dict[str, Any], message: str
) -> None:
    called = False

    def factory(_index):
        nonlocal called
        called = True
        raise AssertionError

    arguments: dict[str, Any] = {
        "model": object(),
        "engine_factory": factory,
        "prompt_token_ids": (1, 2),
        "reward_fn": lambda _result: 0.0,
        "group_size": 1,
        "policy_config": _policy(),
        "group_seed": 0,
        "max_tokens": 1,
        "temperature": 1.0,
        "extraction_tokenizer": object(),
    }
    arguments.update(kwargs)
    with pytest.raises(SFTGRPOError, match=message):
        collect_rlvr_group(**arguments)
    assert called is False


def test_canonical_policy_rejects_noncausal_public_state(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=30,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )
    action = group.trajectories[0].actions[0]
    cases = [
        ((), (), action.pool_positions, "prefix"),
        (action.committed_prefix, action.outgoing_seg_map, (), "pool positions"),
        (action.committed_prefix, action.outgoing_seg_map, (0, 0), "unique"),
        (
            action.committed_prefix,
            (999,) * len(action.committed_prefix),
            action.pool_positions,
            "nonexistent encoder slot",
        ),
    ]
    for prefix, routing, positions, message in cases:
        with pytest.raises(SFTGRPOError, match=message):
            canonical_fp_logits(
                fixture.model,
                group.prompt,
                committed_prefix=prefix,
                outgoing_seg_map=routing,
                pool_positions=positions,
                feature_history=(),
            )


def test_rlvr_uses_no_grad_without_changing_legacy_inference_mode(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    token = fixture.tokenizer.encode("x")[0]
    modes: list[tuple[bool, bool]] = []
    original = fixture.model.forward

    def observed(*args, **kwargs):
        modes.append((t.is_grad_enabled(), t.is_inference_mode_enabled()))
        return original(*args, **kwargs)

    monkeypatch.setattr(fixture.model, "forward", observed)
    _script(monkeypatch, [token])
    chat_generation.generate_self_conditioned_chat(
        fixture.engine,
        fixture.prompt,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )
    assert modes and all(not grad and inference_mode for grad, inference_mode in modes)

    modes.clear()
    _script(monkeypatch, [token])
    with t.inference_mode():
        group = collect_rlvr_group(
            fixture.model,
            _factory(fixture),
            fixture.prompt,
            lambda _result: 0.0,
            group_size=1,
            policy_config=_policy(),
            group_seed=5,
            max_tokens=1,
            extraction_tokenizer=fixture.source,
        )
    assert modes and all(not grad and not inference_mode for grad, inference_mode in modes)
    configure_grpo_trainable_parameters(fixture.model, train_lm_head=False)
    metrics = backward_rlvr_group(fixture.model, group, policy_config=_policy())
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)


def test_canonical_pool_is_causal_to_future_prefix_tokens(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 0.0,
        group_size=1,
        policy_config=_policy(),
        group_seed=6,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )
    action = group.trajectories[0].actions[0]
    prefix = list(action.committed_prefix)
    assert len(prefix) > 2
    mutated = prefix.copy()
    mutated[-1] = (mutated[-1] + 1) % fixture.model.dec_config.vocab_size
    with t.no_grad():
        before = canonical_fp_logits(
            fixture.model,
            group.prompt,
            committed_prefix=prefix,
            outgoing_seg_map=action.outgoing_seg_map,
            pool_positions=(0,),
            feature_history=(),
        )
        after = canonical_fp_logits(
            fixture.model,
            group.prompt,
            committed_prefix=mutated,
            outgoing_seg_map=action.outgoing_seg_map,
            pool_positions=(0,),
            feature_history=(),
        )
    assert t.equal(before, after)


def test_replay_rejects_mutated_behavior_and_pool_before_backward(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 0.0,
        group_size=2,
        policy_config=_policy(),
        group_seed=7,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )
    configure_grpo_trainable_parameters(fixture.model, train_lm_head=False)
    first = group.trajectories[0].actions[0]
    bad_policy = replace(
        first.policy_trace,
        behavior_log_terms=(first.policy_trace.behavior_log_terms[0] + 0.25,),
    )
    bad_action = replace(first, policy_trace=bad_policy)
    bad_trajectory = replace(
        group.trajectories[0],
        actions=(bad_action, *group.trajectories[0].actions[1:]),
    )
    bad_group = replace(group, trajectories=(bad_trajectory, group.trajectories[1]))
    with pytest.raises(SFTGRPOError, match="behavior/replay log terms disagree"):
        backward_rlvr_group(fixture.model, bad_group, policy_config=_policy())
    assert all(parameter.grad is None for parameter in fixture.model.parameters())

    bad_pool_action = replace(first, pool_positions=(0,))
    bad_pool_trajectory = replace(
        group.trajectories[0],
        actions=(bad_pool_action, *group.trajectories[0].actions[1:]),
    )
    bad_pool_group = replace(
        group,
        trajectories=(bad_pool_trajectory, group.trajectories[1]),
    )
    with pytest.raises(SFTGRPOError, match="behavior/replay log terms disagree"):
        backward_rlvr_group(fixture.model, bad_pool_group, policy_config=_policy())
    assert all(parameter.grad is None for parameter in fixture.model.parameters())


@pytest.mark.parametrize("mutation", ["history", "position"])
def test_replay_rejects_malformed_ordered_action_history(
    tmp_path, monkeypatch: pytest.MonkeyPatch, mutation: str
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=1)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [token, token, end, token, token, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda result: float(result.token_ids[-1] == token),
        group_size=2,
        policy_config=_policy(),
        group_seed=70,
        max_tokens=3,
        extraction_tokenizer=fixture.source,
    )
    configure_grpo_trainable_parameters(fixture.model, train_lm_head=False)
    actions = group.trajectories[0].actions
    assert len(actions) >= 2
    provider = RLVRFeatureProvider(
        fixture.model,
        fixture.engine._model._self_cond.config,
        t.device("cpu"),
        policy_config=_policy(),
        group_seed=70,
        rollout_index=0,
        s0_logits_cache=[],
    )
    provider.prompt_replay = group.prompt
    if mutation == "history":
        bad_history = (replace(actions[0], decision_index=1),)
        match = "decision indices"
    else:
        bad_history = (
            actions[0],
            replace(
                actions[1],
                first_committed_position=actions[0].first_committed_position,
            ),
        )
        match = "increase strictly"
    with pytest.raises(SFTGRPOError, match=match):
        provider._outgoing_seg_map(len(actions[-1].committed_prefix), bad_history)
    with pytest.raises(SFTGRPOError, match="already own a token"):
        provider._outgoing_seg_map(
            len(actions[0].committed_prefix),
            (
                replace(
                    actions[0],
                    first_committed_position=len(actions[0].committed_prefix),
                ),
            ),
        )
    changed = (
        replace(actions[1], feature_history=())
        if mutation == "history"
        else replace(actions[1], first_committed_position=actions[0].first_committed_position)
    )
    trajectory = replace(
        group.trajectories[0],
        actions=(actions[0], changed, *actions[2:]),
    )
    malformed = replace(group, trajectories=(trajectory, group.trajectories[1]))
    with pytest.raises(SFTGRPOError, match="history|positions"):
        backward_rlvr_group(fixture.model, malformed, policy_config=_policy())
    assert all(parameter.grad is None for parameter in fixture.model.parameters())


def test_replay_rejects_invalid_group_state_before_backward(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 0.0,
        group_size=2,
        policy_config=_policy(),
        group_seed=71,
        max_tokens=1,
        extraction_tokenizer=fixture.source,
    )
    configure_grpo_trainable_parameters(fixture.model, train_lm_head=False)

    fixture.model.train()
    with pytest.raises(SFTGRPOError, match="eval-mode"):
        backward_rlvr_group(fixture.model, group, policy_config=_policy())
    fixture.model.eval()

    no_live_trajectory = replace(
        group.trajectories[0],
        last_nonforced_logits_prefix_length=(
            group.trajectories[0].actions[0].first_committed_position
        ),
    )
    drifted_s0 = replace(
        group.trajectories[1].actions[0],
        committed_prefix=(*group.trajectories[1].actions[0].committed_prefix, 1),
        outgoing_seg_map=(*group.trajectories[1].actions[0].outgoing_seg_map, 0),
    )
    drifted_trajectory = replace(group.trajectories[1], actions=(drifted_s0,))
    cases: list[tuple[RLVRGroupRollout, dict[str, Any], str]] = [
        (replace(group, advantages=()), {}, "nonempty and aligned"),
        (replace(group, advantages=(float("nan"), 0.0)), {}, "finite"),
        (group, {"loss_scale": 0.0}, "loss_scale"),
        (
            replace(
                group,
                trajectories=(no_live_trajectory, group.trajectories[1]),
            ),
            {},
            "live action",
        ),
        (
            replace(
                group,
                trajectories=(group.trajectories[0], drifted_trajectory),
            ),
            {},
            "S0 canonical state drifted",
        ),
    ]
    for malformed, kwargs, message in cases:
        with pytest.raises(SFTGRPOError, match=message):
            backward_rlvr_group(
                fixture.model,
                malformed,
                policy_config=_policy(),
                **kwargs,
            )
        assert all(parameter.grad is None for parameter in fixture.model.parameters())


@pytest.mark.parametrize("head_depth", [0, 2])
def test_variable_decision_counts_and_loss_scale_match_independent_gradient_factor(
    tmp_path, monkeypatch: pytest.MonkeyPatch, head_depth
) -> None:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    fixture = make_fixture(tmp_path, monkeypatch, K=2, head_depth=head_depth)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [end, token, token, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda result: float(len(result.token_ids) == len(fixture.prompt)),
        group_size=2,
        policy_config=_policy(),
        group_seed=8,
        max_tokens=3,
        extraction_tokenizer=fixture.source,
    )
    assert [len(trajectory.live_actions) for trajectory in group.trajectories] == [1, 2]
    full = copy.deepcopy(fixture.model).eval()
    scaled = copy.deepcopy(fixture.model).eval()
    configure_grpo_trainable_parameters(full, train_lm_head=False)
    configure_grpo_trainable_parameters(scaled, train_lm_head=False)
    with t.no_grad():
        oracle_ratios = t.stack([
            replay_ratio(
                canonical_fp_logits(
                    full, group.prompt, committed_prefix=action.committed_prefix,
                    outgoing_seg_map=action.outgoing_seg_map,
                    pool_positions=action.pool_positions,
                    feature_history=action.feature_history,
                ), action.policy_trace,
            )
            for trajectory in group.trajectories for action in trajectory.live_actions
        ])
    # Independently fixed coefficients for G=2 and D=[1,2]. Retain the actual
    # finite-precision ratios: summing sampled log terms and scoring an ordered
    # action can differ by one FP32 ULP even with identical policy logits.
    oracle_loss = (oracle_ratios * t.tensor([-0.5, 0.25, 0.25])).sum().item()
    full_metrics = backward_rlvr_group(full, group, policy_config=_policy())
    # At ratio 1, per-trajectory decision means cancel exactly. Summing the
    # D=[1,2] decisions would instead yield +0.5 and fail this oracle.
    assert full_metrics.loss == pytest.approx(oracle_loss, abs=1e-8)
    if head_depth == 0:
        assert full_metrics.loss == pytest.approx(0.0, abs=1e-8)
    scaled_metrics = backward_rlvr_group(
        scaled,
        group,
        policy_config=_policy(),
        loss_scale=0.25,
    )
    assert scaled_metrics.loss == pytest.approx(full_metrics.loss, abs=1e-8)
    for (full_name, full_parameter), (scaled_name, scaled_parameter) in zip(
        full.named_parameters(),
        scaled.named_parameters(),
        strict=True,
    ):
        assert full_name == scaled_name
        if full_parameter.requires_grad:
            assert full_parameter.grad is not None and scaled_parameter.grad is not None
            assert t.allclose(
                scaled_parameter.grad,
                full_parameter.grad * 0.25,
                atol=2e-6,
                rtol=2e-5,
            ), full_name


@pytest.mark.parametrize("head_depth", [0, 2])
@pytest.mark.parametrize("head_only", [False, True])
def test_all_zero_advantages_still_allocate_every_connected_gradient(
    tmp_path, monkeypatch: pytest.MonkeyPatch, head_only: bool, head_depth: int
) -> None:
    monkeypatch.setattr(flash_attention, "_override_impl", FlashAttnImpl.SDPA)
    fixture = make_fixture(tmp_path, monkeypatch, K=2, head_depth=head_depth)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [token, end, token, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        lambda _result: 0.0,
        group_size=2,
        policy_config=_policy(),
        group_seed=9,
        max_tokens=2,
        extraction_tokenizer=fixture.source,
    )
    assert group.advantages == (0.0, 0.0)
    manifest = configure_grpo_trainable_parameters(
        fixture.model, train_lm_head=False, train_feature_head_only=head_only
    )
    metrics = backward_rlvr_group(
        fixture.model, group, policy_config=_policy(), train_feature_head_only=head_only
    )
    assert metrics.loss == pytest.approx(0.0, abs=1e-8)
    assert all(
        parameter.grad is not None and not bool(parameter.grad.ne(0).any())
        for name, parameter in fixture.model.named_parameters()
        if name in manifest
    )


def test_no_hook_generator_is_exactly_frozen_current_main_with_native_tool_calls(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    current = make_fixture(tmp_path / "current", monkeypatch, K=2)
    frozen = make_fixture(tmp_path / "frozen", monkeypatch, K=2)
    current_forwards, current_appends = instrument(current, monkeypatch)
    frozen_forwards, frozen_appends = instrument(frozen, monkeypatch)
    tok = current.tokenizer
    script = [
        tok.encode_special("<python_start>"),
        *tok.encode("1+1"),
        tok.encode_special("<python_end>"),
        tok.encode_special("<output_start>"),
        *tok.encode("2"),
        tok.encode_special("<output_end>"),
        *tok.encode(" done"),
        tok.encode_special("<assistant_end>"),
    ]
    _script(monkeypatch, script)
    current_result = chat_generation.generate_self_conditioned_chat(
        current.engine,
        current.prompt,
        max_tokens=len(script),
        extraction_tokenizer=current.source,
    )
    _script(monkeypatch, script)
    frozen_result = _frozen_main_generate(
        frozen.engine,
        frozen.prompt,
        max_tokens=len(script),
        extraction_tokenizer=frozen.source,
    )
    assert current_result.token_ids == frozen_result.token_ids
    assert current_result.mask == frozen_result.mask
    assert len(current_forwards) == len(frozen_forwards)
    for actual, expected in zip(current_forwards, frozen_forwards, strict=True):
        assert (actual.position, actual.ids, actual.slots) == (
            expected.position,
            expected.ids,
            expected.slots,
        )
        assert (actual.fp is None) is (expected.fp is None)
        assert (actual.logits is None) is (expected.logits is None)
        if actual.fp is not None:
            assert t.equal(actual.fp, expected.fp)
        if actual.logits is not None:
            assert t.equal(actual.logits, expected.logits)
    assert [entry[0] for entry in current_appends] == [
        entry[0] for entry in frozen_appends
    ]
    for actual, expected in zip(current_appends, frozen_appends, strict=True):
        assert actual[2:] == expected[2:]
        assert t.equal(actual[1], expected[1])


@pytest.mark.skipif(not t.cuda.is_available(), reason="compacted production encoder requires CUDA")
@pytest.mark.parametrize("head_depth", [0, 2])
def test_compacted_original_id_rollout_and_replay_preserve_sample_order(
    tmp_path, monkeypatch: pytest.MonkeyPatch, head_depth
) -> None:
    device = t.device("cuda")
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    allowed = (1, 2, 3, 7)
    feature_vocab = FeatureVocab(n_sae_features=8, original_ids=allowed)
    fixture.contract = replace(fixture.contract, feature_vocab=feature_vocab)
    fixture.provider = _FilteredContextProvider(
        fixture.provider,
        allowed,
        fixture.contract,
    )
    sequence_len = fixture.model.dec_config.sequence_len
    t.manual_seed(418)
    fixture.model = build_tiny_model(
        n_sae_features=8,
        num_buckets=4,
        feature_pred_config=FeaturePredictionConfig(
            loss_mode=FeaturePredLossMode.MAX_POOL,
            head_layers=head_depth,
        ),
        seq_len=sequence_len,
        n_layer=2,
        xattn_frequency=1,
        feature_vocab=feature_vocab,
        vocab_size=512,
    ).to(device).eval()
    with t.no_grad():
        for name, parameter in fixture.model.named_parameters():
            if "cross_attn" in name and parameter.ndim >= 2:
                parameter.normal_(std=0.04)
            if name.startswith("feature_pred_head.residual_blocks."):
                parameter.normal_(std=0.03)
    token = fixture.tokenizer.encode("x")[0]
    end = fixture.tokenizer.encode_special("<assistant_end>")
    _script(monkeypatch, [token, end, token, end])
    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture, device),
        fixture.prompt,
        lambda result: float(result.token_ids[len(fixture.prompt)] == token),
        group_size=2,
        policy_config=_policy(),
        group_seed=2027,
        max_tokens=2,
        extraction_tokenizer=fixture.source,
    )
    assert fixture.model.feature_vocab is not None
    assert fixture.model.feature_vocab.original_ids == allowed
    for trajectory in group.trajectories:
        for action in trajectory.actions:
            compact_rows = (*action.policy_trace.positive_rows, *action.policy_trace.negative_rows)
            assert action.conditioning.feature_ids == tuple(allowed[row] for row in compact_rows)
            assert set(action.conditioning.feature_ids) <= set(allowed)
    manifest = configure_grpo_trainable_parameters(fixture.model, train_lm_head=False)
    metrics = backward_rlvr_group(fixture.model, group, policy_config=_policy())
    assert metrics.ratio_min == pytest.approx(1.0, abs=2e-5)
    assert metrics.ratio_max == pytest.approx(1.0, abs=2e-5)
    parameters = dict(fixture.model.named_parameters())
    assert all(parameters[name].grad is not None for name in manifest)
    assert fixture.model.lm_head.weight.grad is None


@pytest.mark.skipif(
    not dist.is_available() or not dist.is_gloo_available(),
    reason="requires torch.distributed Gloo",
)
def test_two_rank_gloo_policy_gradient_matches_serial_prompt_mean() -> None:
    results = run_two_ranks(_gloo_policy_worker, "rlvr_policy_gradient", join_timeout_s=180)
    rank0_names, rank0_bytes = results[0]
    rank1_names, rank1_bytes = results[1]
    assert rank0_names == rank1_names
    distributed = t.frombuffer(bytearray(rank0_bytes), dtype=t.float32).clone()
    assert t.equal(
        distributed,
        t.frombuffer(bytearray(rank1_bytes), dtype=t.float32),
    )

    _install_cpu_model_reference()
    oracle = _gloo_model()
    oracle_names = configure_grpo_trainable_parameters(oracle, train_lm_head=False)
    assert oracle_names == rank0_names
    for rank in range(2):
        backward_rlvr_group(
            oracle,
            _gloo_group(oracle, rank),
            policy_config=_policy(),
            loss_scale=0.5,
        )
    serial = t.frombuffer(
        bytearray(_gradient_bytes(oracle, oracle_names)),
        dtype=t.float32,
    ).clone()
    assert t.allclose(distributed, serial, atol=2e-6, rtol=2e-5)
