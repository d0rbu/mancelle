"""Exact CUDA contracts for the explicit, group-scoped RLVR replay cache."""

from __future__ import annotations

import copy
import os
import sys
import weakref
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as flash
from nanochat.gpt import Linear
from steering.conditional.encoder_decoder import _rlvr_replay_guard as native_guard
from steering.conditional.encoder_decoder import rlvr_policy as policy_module
from steering.conditional.encoder_decoder import rlvr_replay_cache as candidate
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.frozen_decoder import stage_frozen_decoder_linears
from steering.conditional.encoder_decoder.rlvr_runtime import configure_rlvr_compiler
from steering.conditional.encoder_decoder.sft_grpo import (
    sample_weighted_without_replacement,
    trace_to_conditioning,
)
from steering.conditional.encoder_decoder.sft_grpo_training import (
    configure_grpo_trainable_parameters,
)
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.selfcond import test_frozen_decoder_rlvr as native_qualification
from tests.steering.selfcond.rlvr_test_helpers import policy

pytestmark = pytest.mark.slow
NATIVE_CANONICAL = policy_module.canonical_fp_logits


@pytest.fixture(autouse=True)
def _install_test_owned_cpu_encoder_reference():
    """This CUDA suite qualifies the native compiled encoder/cross-attention."""
    yield


@pytest.fixture(scope="module", autouse=True)
def cuda_contract(record_testsuite_property):
    assert t.cuda.is_available(), "Real CUDA is required"
    assert os.environ.get("CUBLAS_WORKSPACE_CONFIG") == ":4096:8"
    deterministic = t.are_deterministic_algorithms_enabled()
    warn_only = t.is_deterministic_algorithms_warn_only_enabled()
    precision = t.get_float32_matmul_precision()
    previous_attention = flash._override_impl
    cudnn = t.backends.cuda.cudnn_sdp_enabled()
    t.use_deterministic_algorithms(True)
    t.set_float32_matmul_precision("highest")
    t.backends.cuda.enable_cudnn_sdp(False)
    flash._override_impl = flash.FlashAttnImpl.SDPA
    record_testsuite_property("worker_pid", str(os.getpid()))
    record_testsuite_property("torch", str(t.__version__))
    record_testsuite_property("cuda", str(t.version.cuda))
    record_testsuite_property("gpu", t.cuda.get_device_name())
    t.compiler.reset()
    try:
        yield
    finally:
        t.cuda.synchronize()
        t.compiler.reset()
        flash._override_impl = previous_attention
        t.use_deterministic_algorithms(deterministic, warn_only=warn_only)
        t.set_float32_matmul_precision(precision)
        t.backends.cuda.enable_cudnn_sdp(cudnn)


def make_model(depth=0, *, block=FeaturePredictionBlock.MLP):
    t.manual_seed(737)
    model = build_tiny_model(n_sae_features=8, num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(loss_mode=FeaturePredLossMode.MAX_POOL,
                                                  head_layers=depth, block=block),
        n_layer=1, seq_len=16, xattn_frequency=1, vocab_size=32).cuda().eval()
    model.init_weights()
    with t.no_grad():
        for name, value in model.named_parameters():
            if ("cross_attn" in name and value.ndim >= 2) or name.startswith("feature_pred_head.residual_blocks."):
                value.normal_(std=0.04)
    configure_grpo_trainable_parameters(model, train_lm_head=False, train_feature_head_only=True)
    configure_rlvr_compiler(model)
    return model


@pytest.fixture
def model():
    return make_model()


@contextmanager
def staged(model):
    with stage_frozen_decoder_linears(model, max_shadow_bytes=64 * 1024**2, min_free_bytes=0,
                                     train_feature_head_only=True, world_size=1) as report:
        assert report.applied
        yield report
    assert report.cleaned_up


def cache_context(model, *, report=None, **changes):
    return policy_module.rlvr_group_replay_cache(
        model, settings=candidate.HiddenReplaySettings(
            max_cache_bytes=changes.get("budget", 16 * 1024**2), min_free_bytes=changes.get("min_free_bytes", 0)),
        train_feature_head_only=changes.get("head_only", True), world_size=changes.get("world_size", 1),
        report=report if report is not None else candidate.HiddenReplayReport(),
    )


def clone_prompt(prompt):
    return replace(prompt, **{name: getattr(prompt, name).detach().clone() for name in
                              ("enc_feature_ids", "enc_magnitudes", "enc_mask", "prefix_seg_map")})


def make_group(model, *, second_action=True, group_size=2, seed=1000, cloned_prompts=False, replay_cache=None):
    """Real native CUDA trunk/head and sampler, with explicit causal trace rows."""
    if replay_cache is not None:
        assert replay_cache.begin_collection(model)
    device = next(model.parameters()).device
    prompt = policy_module.RLVRPromptReplay((1, 2, 3, 4),
        t.tensor([[[1, 5]]], device=device), t.tensor([[[0, 1]]], device=device),
        t.tensor([[[True, True]]], device=device), t.tensor([[0, 0, 0]], device=device), 1)
    with t.no_grad():
        s0 = policy_module.canonical_fp_logits(model, prompt, committed_prefix=(1, 2, 3),
            outgoing_seg_map=(0, 0, 0), pool_positions=(1, 2), feature_history=(), replay_cache=replay_cache)
    trajectories = []
    for index in range(group_size):
        trace = sample_weighted_without_replacement(s0, positive_count=1, negative_count=1,
            generator=t.Generator(device=device).manual_seed(seed + index))
        conditioning = trace_to_conditioning(trace, compact_to_original=None)
        first = policy_module.RLVRActionTrace(0, "prompt", (1, 2, 3), (0, 0, 0), (1, 2), 3, (), conditioning, trace)
        actions = [first]
        if second_action:
            current_prompt = clone_prompt(prompt) if cloned_prompts else prompt
            with t.no_grad():
                later = policy_module.canonical_fp_logits(model, current_prompt, committed_prefix=(1, 2, 3, 4, 5),
                    outgoing_seg_map=(0, 0, 0, 1, 1), pool_positions=(3, 4), feature_history=(conditioning,), replay_cache=replay_cache)
            later_trace = sample_weighted_without_replacement(later, positive_count=1, negative_count=1,
                generator=t.Generator(device=device).manual_seed(seed + 100 + index))
            actions.append(policy_module.RLVRActionTrace(1, "assistant", (1, 2, 3, 4, 5), (0, 0, 0, 1, 1),
                (3, 4), 5, (conditioning,), trace_to_conditioning(later_trace, compact_to_original=None), later_trace))
        trajectories.append(policy_module.RLVRTrajectoryTrace((1, 2, 3, 4, 5, 6), (0, 0, 0, 1, 1, 1),
            float(index == 0), "eos", tuple(actions), 6, True, None))
    group = policy_module.RLVRGroupRollout(prompt, tuple(trajectories), (0.0,) if group_size == 1 else (1.0, -1.0))
    if replay_cache is not None:
        replay_cache.seal_collection(model, group)
    return group


def backward(model, group, *, replay_cache=None):
    return policy_module.backward_rlvr_group(model, group, policy_config=policy(),
                                             train_feature_head_only=True, replay_cache=replay_cache)


def gradients(model):
    return {name: None if value.grad is None else value.grad.detach().clone()
            for name, value in model.named_parameters() if value.requires_grad}


def exact(actual, expected):
    t.testing.assert_close(actual, expected, rtol=0, atol=0)


def outcome(model, callback):
    model.zero_grad(set_to_none=True)
    try:
        result = callback()
    except (RuntimeError, ValueError, AssertionError, IndexError, TypeError) as error:
        return (type(error), str(error)), gradients(model)
    return result, gradients(model)


def compare_native_replay(model, group, cache):
    # Omitting the explicit session evaluates the same native replay first.
    # Neither callback mutates parameters or consumes RNG.
    expected, expected_gradients = outcome(model, lambda: backward(model, group))
    actual, actual_gradients = outcome(model, lambda: backward(model, group, replay_cache=cache))
    assert actual == expected
    exact(actual_gradients, expected_gradients)
    return actual, actual_gradients


def clean(cache, *, allow_reported_consumers=False):
    report = cache.report
    assert report.cleaned_up and report.active_groups == 0
    assert report.current_bytes == report.current_hidden_bytes == report.current_prompt_bytes == 0
    assert not cache._entries and cache._model is cache._group is None
    if not allow_reported_consumers:
        assert report.owned_tensor_objects_alive_after_cleanup == 0
        assert report.max_owned_tensor_objects_alive_after_cleanup == 0
    assert policy_module.canonical_fp_logits is NATIVE_CANONICAL


@pytest.mark.parametrize("depth,block", [(0, FeaturePredictionBlock.MLP), (2, FeaturePredictionBlock.MLP),
                                      (2, FeaturePredictionBlock.TRANSFORMER)], ids=["0", "2", "transformer-2"])
@pytest.mark.parametrize("group_size", [1, 2])
def test_native_cuda_head_gradients_and_s0_sharing(depth, group_size, block):
    model = make_model(depth, block=block)
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    with staged(model) as decoder, cache_context(model) as cache:
        group = make_group(model, group_size=group_size, cloned_prompts=True, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        cpu_rng, gpu_rng = t.get_rng_state(), t.cuda.get_rng_state()
        result, grads = compare_native_replay(model, group, cache)
        assert isinstance(result, policy_module.RLVRReplayMetrics)
        assert all(value is not None for value in grads.values())
        if group_size == 2:
            assert all(bool(t.count_nonzero(value)) for value in grads.values())
        else:
            assert all(not bool(t.count_nonzero(value)) for value in grads.values())
        assert cache.report.replay_hits == 1 + group_size, asdict(cache.report)
        assert cache.report.replay_misses == 0 and cache.report.invalidations == 0
        assert t.equal(cpu_rng, t.get_rng_state()) and t.equal(gpu_rng, t.cuda.get_rng_state())
        assert decoder.fallback_calls == decoder.invalidated_modules == 0
    clean(cache)
    exact(model.state_dict(), before)


@pytest.mark.parametrize("head_block", [FeaturePredictionBlock.MLP, FeaturePredictionBlock.TRANSFORMER])
def test_depth2_three_generated_updates_with_staging_in_both_arms(tmp_path, monkeypatch, head_block):
    native_stage = native_qualification.stage_frozen_decoder_linears
    native_configure = native_qualification.configure_grpo_trainable_parameters
    native_collect = native_qualification.collect_rlvr_group
    native_backward = native_qualification.backward_rlvr_group
    models, decoder_reports, sessions = [], [], []
    report = candidate.HiddenReplayReport()
    active: dict[str, Any] = {"enabled": False, "context": None, "session": None}

    def configure(model, **kwargs):
        models.append(model)
        return native_configure(model, **kwargs)

    @contextmanager
    def reference_context():
        with native_stage(models[-1], max_shadow_bytes=64 * 1024**2, min_free_bytes=0,
                          train_feature_head_only=True, world_size=1) as decoder:
            decoder_reports.append(decoder)
            yield None

    @contextmanager
    def candidate_context(model, **kwargs):
        with native_stage(model, **kwargs) as decoder:
            decoder_reports.append(decoder)
            active["enabled"] = True
            try:
                yield decoder
            finally:
                try:
                    if active["context"] is not None:
                        active["context"].__exit__(*sys.exc_info())
                finally:
                    active.update(enabled=False, context=None, session=None)

    def collect(model, *args, **kwargs):
        if not active["enabled"]:
            return native_collect(model, *args, **kwargs)
        assert active["context"] is None
        context = cache_context(model, report=report)
        session = context.__enter__()
        active.update(context=context, session=session)
        sessions.append(session)
        return native_collect(model, *args, **kwargs, replay_cache=session)

    def replay(model, group, **kwargs):
        if not active["enabled"]:
            return native_backward(model, group, **kwargs)
        try:
            return native_backward(model, group, **kwargs, replay_cache=active["session"])
        finally:
            try:
                active["context"].__exit__(*sys.exc_info())
            finally:
                active.update(context=None, session=None)

    monkeypatch.setattr(native_qualification, "configure_grpo_trainable_parameters", configure)
    monkeypatch.setattr(native_qualification, "nullcontext", reference_context)
    monkeypatch.setattr(native_qualification, "stage_frozen_decoder_linears", candidate_context)
    monkeypatch.setattr(native_qualification, "collect_rlvr_group", collect)
    monkeypatch.setattr(native_qualification, "backward_rlvr_group", replay)
    native_qualification.test_frozen_decoder_shadows_preserve_three_generated_updates(
        tmp_path, monkeypatch, head_block=head_block)
    assert len(models) == len(decoder_reports) == 2 and len(sessions) == 3
    assert all(item.applied and item.cleaned_up and item.fallback_calls == item.invalidated_modules == 0
               for item in decoder_reports)
    assert report.requested_groups == report.groups_completed == report.groups_closed == 3
    assert report.replay_hits > 0 and report.replay_misses == report.invalidations == 0
    assert report.groups_failed == report.groups_abandoned == report.inactive_groups == 0
    for session in sessions:
        clean(session)


@pytest.mark.parametrize("field", ["enc_feature_ids", "enc_magnitudes", "enc_mask", "prefix_seg_map"])
@pytest.mark.parametrize("write", ["versioned", "data", "independent_alias"])
def test_fresh_prompt_contents_force_native_replay(model, field, write):
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        target = getattr(group.prompt, field)
        if write == "independent_alias":
            target = t.empty(0, dtype=target.dtype, device=target.device).set_(
                target.untyped_storage(), target.storage_offset(), target.shape, target.stride())
        elif write == "data":
            target = target.data
        with t.no_grad():
            target.reshape(-1)[0] = 2 if field == "enc_feature_ids" else (not bool(target.reshape(-1)[0]))
        compare_native_replay(model, group, cache)
        assert cache.report.replay_hits == 0 and cache.report.replay_misses > 0
    clean(cache)


def test_equal_content_prompt_rebinding_matches(model):
    with staged(model), cache_context(model) as cache:
        group = make_group(model, cloned_prompts=True, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        object.__setattr__(group, "prompt", clone_prompt(group.prompt))
        compare_native_replay(model, group, cache)
        assert cache.report.replay_hits == 3 and cache.report.replay_misses == 0
    clean(cache)


@pytest.mark.parametrize("change", ["decoder_weight", "encoder_weight", "cross_weight", "rotary_buffer", "module",
                                    "parameter", "config", "head_layout", "embedding_max_norm", "submodule_training"])
def test_frozen_state_changes_delegate_native(model, monkeypatch, change):
    replacement = copy.deepcopy(model.transformer["h"][0].mlp) if change == "module" else None
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        with t.no_grad():
            if change == "decoder_weight":
                model.transformer["h"][0].mlp.c_fc.weight.add_(0.01)
            elif change == "encoder_weight":
                next(model.encoder.parameters()).add_(0.01)
            elif change == "cross_weight":
                model.cross_attn_layers["0"].c_q.weight.add_(0.01)
            elif change == "rotary_buffer":
                model.cos.add_(0.01)
            elif change == "module":
                model.transformer["h"][0].mlp = replacement
            elif change == "parameter":
                layer = model.transformer["h"][0].mlp.c_fc
                layer.weight = t.nn.Parameter(layer.weight.detach().clone(), requires_grad=False)
            elif change == "config":
                monkeypatch.setattr(model.feature_pred_config, "loss_mode", FeaturePredLossMode.LAST_TOKEN)
            elif change == "head_layout":
                model.feature_pred_head.weight = t.nn.Parameter(model.feature_pred_head.weight.detach().clone())
            elif change == "embedding_max_norm":
                model.transformer["wte"].max_norm = 1000.0
            else:
                model.encoder.train()
        compare_native_replay(model, group, cache)
        assert cache.report.replay_hits == 0 and cache.report.invalidations > 0
    clean(cache)


def test_live_head_update_remains_live_and_native_behavior_check_fails(model):
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        references = [weakref.ref(entry.hidden) for entries in cache._entries.values() for entry in entries]
        with t.no_grad():
            model.feature_pred_head.weight[0].add_(2.0)
        result, grads = compare_native_replay(model, group, cache)
        assert isinstance(result, tuple) and issubclass(result[0], Exception)
        assert all(value is None for value in grads.values())
        assert cache.report.replay_hits > 0 and cache.report.invalidations == 0
    assert all(reference() is None for reference in references)
    clean(cache, allow_reported_consumers=True)


@pytest.mark.parametrize("kind", ["module_hook", "callback", "class_descriptor", "class_lookup", "precision", "autocast", "stream", "policy_dependency"])
def test_observable_execution_changes_use_native(model, monkeypatch, kind):
    handle = None
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        try:
            if kind == "module_hook":
                handle = model.encoder.register_forward_pre_hook(lambda _module, _args: None)
            elif kind == "callback":
                monkeypatch.setattr(model, "_post_cross_attention_eval_callback", lambda _index, x: x)
            elif kind == "class_descriptor":
                monkeypatch.setattr(Linear, "weight", property(lambda self: self._parameters["weight"]), raising=False)
            elif kind == "class_lookup":
                original = Linear.__getattribute__
                monkeypatch.setattr(Linear, "__getattribute__", lambda self, name: original(self, name))
            elif kind == "policy_dependency":
                original_encode = policy_module._encode_canonical_state
                monkeypatch.setattr(policy_module, "_encode_canonical_state", lambda *args, **kwargs: original_encode(*args, **kwargs))
            elif kind == "precision":
                t.set_float32_matmul_precision("high")
            if kind == "autocast":
                with t.autocast("cuda", dtype=t.bfloat16):
                    compare_native_replay(model, group, cache)
            elif kind == "stream":
                t.cuda.synchronize()
                stream = t.cuda.Stream()
                with t.cuda.stream(stream):
                    compare_native_replay(model, group, cache)
                stream.synchronize()
            else:
                compare_native_replay(model, group, cache)
            assert cache.report.replay_hits == 0 and cache.report.invalidations > 0
        finally:
            if handle is not None:
                handle.remove()
            t.set_float32_matmul_precision("highest")
    clean(cache)


@pytest.mark.parametrize("case", ["empty_pool", "reordered_pool", "wrong_history", "wrong_decision", "behavior_terms", "dormant", "s0_drift"])
def test_native_causal_validation_and_dormant_actions_remain_authoritative(model, case):
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        assert cache.report.admitted_entries > 0, asdict(cache.report)
        trajectory = group.trajectories[0]
        first, later = trajectory.actions
        if case == "empty_pool":
            first = replace(first, pool_positions=())
        elif case == "reordered_pool":
            first = replace(first, pool_positions=(2, 1))
        elif case == "wrong_history":
            later = replace(later, feature_history=())
        elif case == "wrong_decision":
            later = replace(later, decision_index=2)
        elif case == "behavior_terms":
            first = replace(first, policy_trace=replace(first.policy_trace,
                behavior_log_terms=tuple(value + 0.5 for value in first.policy_trace.behavior_log_terms)))
        changed = replace(trajectory, actions=(first, later),
                          last_nonforced_logits_prefix_length=4 if case == "dormant" else trajectory.last_nonforced_logits_prefix_length)
        second = group.trajectories[1]
        if case == "s0_drift":
            second = replace(second, actions=(replace(second.actions[0], committed_prefix=(1, 2, 4)), second.actions[1]))
        object.__setattr__(group, "trajectories", (changed, second))
        result, grads = compare_native_replay(model, group, cache)
        if case == "dormant":
            assert isinstance(result, policy_module.RLVRReplayMetrics)
            assert cache.report.replay_hits == 2
        else:
            assert isinstance(result, tuple) and issubclass(result[0], Exception)
            assert all(value is None for value in grads.values())
    clean(cache, allow_reported_consumers=case != "dormant")


@pytest.mark.parametrize("budget", [1, 16 * 1024**2])
def test_budget_admission_is_bounded_and_native_results_exact(model, budget):
    with staged(model), cache_context(model, budget=budget) as cache:
        group = make_group(model, replay_cache=cache)
        compare_native_replay(model, group, cache)
        assert cache.report.peak_bytes <= budget
        if budget == 1:
            assert cache.report.capture_skips > 0 and cache.report.replay_hits == 0
        else:
            assert cache.report.replay_hits == 3
    clean(cache)


@pytest.mark.parametrize("allocation_index", [1, 3, 5])
def test_optional_clone_oom_falls_back_without_swallowing_native_error(model, monkeypatch, allocation_index):
    calls = []

    native_clone = candidate._clone_for_cache

    def oom(value):
        calls.append(1)
        if len(calls) % allocation_index == 0:
            raise t.cuda.OutOfMemoryError("injected optional cache allocation")
        return native_clone(value)

    monkeypatch.setattr(candidate, "_clone_for_cache", oom)
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        compare_native_replay(model, group, cache)
        assert calls and cache.report.capture_skips > 0 and cache.report.replay_hits == 0
    clean(cache)


@pytest.mark.parametrize("changes", [{"budget": 0}, {"head_only": False}, {"world_size": 2}])
def test_disabled_scope_preserves_native_results_and_globals(model, changes):
    report = candidate.HiddenReplayReport()
    with staged(model), cache_context(model, report=report, **changes) as cache:
        assert cache is None and policy_module.canonical_fp_logits is NATIVE_CANONICAL
        group = make_group(model, replay_cache=cache)
        compare_native_replay(model, group, cache)
    assert report.requested_groups == report.inactive_groups == 1
    assert report.active_groups == report.admitted_entries == report.groups_closed == 0
    assert report.cleaned_up and not report.applied
    assert policy_module.canonical_fp_logits is NATIVE_CANONICAL


@pytest.mark.parametrize("failure_phase", ["collect", "backward"])
def test_exception_cleanup_and_fresh_group_reentry(model, failure_phase):
    report = candidate.HiddenReplayReport()
    with staged(model):
        with pytest.raises(ValueError, match="injected failure"):
            with cache_context(model, report=report) as cache:
                group = make_group(model, replay_cache=cache)
                assert report.admitted_entries > 0
                if failure_phase == "collect":
                    raise ValueError("injected failure")
                with cache.replaying(model, group, train_feature_head_only=True):
                    raise ValueError("injected failure")
        assert report.current_bytes == report.active_groups == 0
        assert report.groups_failed == report.groups_closed == 1
        with cache_context(model, report=report) as next_cache:
            next_group = make_group(model, replay_cache=next_cache)
            compare_native_replay(model, next_group, next_cache)
        assert report.groups_completed == 1 and report.groups_closed == 2
    clean(cache)
    clean(next_cache)


def test_abandoned_and_foreign_group_lifetimes(model):
    report = candidate.HiddenReplayReport()
    with staged(model):
        with cache_context(model, report=report) as first:
            make_group(model, replay_cache=first)
            assert report.admitted_entries > 0
        assert report.groups_abandoned == 1
        with cache_context(model, report=report) as cache:
            group = make_group(model, seed=2000, replay_cache=cache)
            foreign = replace(group)
            compare_native_replay(model, foreign, cache)
        assert report.replay_hits == 0 and report.current_bytes == 0
        assert report.groups_abandoned == report.groups_closed == 2
    clean(first)
    clean(cache)


def test_shared_report_rejects_overlap_and_closed_session_cannot_reenter(model):
    report = candidate.HiddenReplayReport()
    with cache_context(model, report=report) as outer:
        with pytest.raises(RuntimeError, match="overlap"):
            with cache_context(model, report=report):
                pytest.fail("Overlapping group sessions admitted")
        assert report.active_groups == 1
        assert policy_module.canonical_fp_logits is NATIVE_CANONICAL
    with pytest.raises(RuntimeError, match="fresh group"):
        outer.begin_collection(model)
    assert report.active_groups == 0 and report.groups_closed == 1
    clean(outer)


@pytest.mark.parametrize("offset", [-1, 0])
def test_exact_first_entry_byte_boundary(model, offset):
    # S0 is BF16 [1,3,D]; all four contiguous prompt clones occupy 58 bytes.
    expected_bytes = 3 * model.dec_config.n_embd * 2 + (2 * 8 + 2 * 8 + 2 + 3 * 8)
    budget = expected_bytes + offset
    with staged(model), cache_context(model, budget=budget) as cache:
        group = make_group(model, replay_cache=cache)
        compare_native_replay(model, group, cache)
        assert cache.report.peak_bytes == (expected_bytes if offset == 0 else 0)
        assert cache.report.replay_hits == (1 if offset == 0 else 0)
        assert cache.report.capture_skips > 0
    clean(cache)


@pytest.mark.parametrize("cap", ["MAX_ENTRIES_PER_GROUP", "MAX_KEY_ELEMENTS_PER_GROUP", "MAX_KEY_BYTES_PER_GROUP"])
def test_entry_and_host_key_caps_remain_native(model, monkeypatch, cap):
    monkeypatch.setattr(candidate, cap, 1)
    with staged(model), cache_context(model) as cache:
        group = make_group(model, replay_cache=cache)
        compare_native_replay(model, group, cache)
        assert cache.report.capture_skips > 0
        if cap == "MAX_ENTRIES_PER_GROUP":
            assert cache.report.peak_entries == cache.report.replay_hits == 1
        else:
            assert cache.report.peak_entries == cache.report.replay_hits == 0
    clean(cache)


@pytest.mark.parametrize("kind", ["distinct_storage_alias", "same_parameter_alias", "same_parameter_buffer_alias", "trunk_trainable", "head_frozen",
                                  "preexisting_hook", "preexisting_callback", "inference_mode", "compiled_call"])
def test_guard_rejects_unsupported_admission(model, kind):
    handle = None
    if kind == "distinct_storage_alias":
        frozen = model.transformer["h"][0].mlp.c_fc.weight
        model.feature_pred_head.weight = t.nn.Parameter(frozen[:model.feature_pred_head.out_features])
    elif kind == "same_parameter_alias":
        model.transformer["h"][0].mlp.c_fc.weight = model.feature_pred_head.weight
    elif kind == "same_parameter_buffer_alias":
        model.register_buffer("head_alias", model.feature_pred_head.weight)
    elif kind == "trunk_trainable":
        model.transformer["h"][0].mlp.c_fc.weight.requires_grad_(True)
    elif kind == "head_frozen":
        model.feature_pred_head.weight.requires_grad_(False)
    elif kind == "compiled_call":
        model.encoder._compiled_call_impl = lambda *args, **kwargs: None
    elif kind == "preexisting_hook":
        handle = model.encoder.register_forward_pre_hook(lambda _module, _args: None)
    elif kind == "preexisting_callback":
        model._post_cross_attention_eval_callback = lambda _layer, value: value
    try:
        with t.inference_mode(kind == "inference_mode"):
            guard, reason = native_guard.NativeFrozenState.capture(model, policy_dependencies=policy_module._REPLAY_DEPENDENCIES)
        assert guard is None and reason not in {"admitted", "matched"}
    finally:
        if handle is not None:
            handle.remove()


def test_unsupported_sequence_is_delegated_without_extra_iteration(model):
    class ObservedList(list):
        def __init__(self, values):
            super().__init__(values)
            self.iterations = 0

        def __iter__(self):
            self.iterations += 1
            return super().__iter__()

    with staged(model):
        prompt = make_group(model, second_action=False).prompt
        reference_prefix, candidate_prefix = ObservedList([1, 2, 3]), ObservedList([1, 2, 3])
        kwargs: dict[str, Any] = {"outgoing_seg_map": (0, 0, 0), "pool_positions": (1, 2), "feature_history": ()}
        with t.no_grad():
            expected = NATIVE_CANONICAL(model, prompt, committed_prefix=reference_prefix, **kwargs)
            with cache_context(model) as cache:
                assert cache.begin_collection(model)
                actual = policy_module.canonical_fp_logits(
                    model, prompt, committed_prefix=candidate_prefix, **kwargs, replay_cache=cache)
                exact(actual, expected)
                assert reference_prefix.iterations == candidate_prefix.iterations == 1
                assert cache.report.capture_attempts == cache.report.admitted_entries == 0
        clean(cache)



def test_free_memory_admission_preserves_native_results(model):
    with staged(model), cache_context(model, min_free_bytes=2**60) as cache:
        group = make_group(model, replay_cache=cache)
        compare_native_replay(model, group, cache)
        assert cache.report.peak_bytes == cache.report.admitted_entries == cache.report.replay_hits == 0
        assert cache.report.capture_skips > 0
    clean(cache)


def test_same_frozen_tensor_buffer_alias_is_admitted_and_version_guarded(model):
    model.register_buffer("frozen_alias", model.cos)
    guard, reason = native_guard.NativeFrozenState.capture(model, policy_dependencies=policy_module._REPLAY_DEPENDENCIES)
    assert guard is not None and reason == "admitted", reason
    assert guard.matches() == (True, "matched")
    with t.no_grad():
        model.frozen_alias.add_(0.01)
    valid, reason = guard.matches()
    assert not valid and reason.startswith("frozen_tensor_changed:")



def test_session_rejects_foreign_thread_without_gpu_work_there(model):
    with cache_context(model) as cache:
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(cache.begin_collection, model)
            with pytest.raises(RuntimeError, match="owning thread"):
                future.result()
        assert cache.report.active_groups == 1
    clean(cache)
