"""Native CUDA contracts for group-local canonical history encoder reuse."""

from __future__ import annotations

import sys
import weakref
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Any

import pytest
import torch as t

from steering.conditional.encoder_decoder import packed_encoder as packed_module
from steering.conditional.encoder_decoder import rlvr_policy as policy_module
from steering.conditional.encoder_decoder import rlvr_replay_cache as cache_module
from steering.conditional.encoder_decoder.sft_grpo import ConditioningAction
from tests.steering.selfcond import test_frozen_decoder_rlvr as native_qualification
from tests.steering.selfcond.test_rlvr_replay_cache import (
    clean,
    cuda_contract,  # noqa: F401 -- use the native deterministic CUDA fixture
    exact,
    make_model,
    staged,
)

pytestmark = pytest.mark.slow

A = ConditioningAction((1, 3), (0, 1), (True, True))
B = ConditioningAction((2, 5), (1, 0), (True, True))


@pytest.fixture(autouse=True)
def _install_test_owned_cpu_encoder_reference():
    """Keep the real compiled CUDA encoder and packed cross-attention active."""
    yield


@pytest.fixture
def model():
    result = make_model()
    # Exercise attention/MLP residual arithmetic, not the zero-initialized path.
    with t.no_grad():
        for block in result.encoder.blocks:
            block.attn.c_proj.weight.normal_(std=0.04)
            block.mlp.c_proj.weight.normal_(std=0.04)
    return result


def _context(model, *, enabled=True, report=None, budget=16 * 1024**2):
    return policy_module.rlvr_group_replay_cache(
        model, settings=cache_module.HiddenReplaySettings(
            max_cache_bytes=budget, min_free_bytes=0, history_encoder=enabled),
        train_feature_head_only=True, world_size=1,
        report=cache_module.HiddenReplayReport() if report is None else report,
    )


def _prompt(model):
    device = model.get_device()
    return policy_module.RLVRPromptReplay(
        (1, 2, 3, 4), t.tensor([[[1, 5]]], device=device),
        t.tensor([[[0, 1]]], device=device), t.tensor([[[True, True]]], device=device),
        t.tensor([[0, 0, 0]], device=device), 1,
    )


def _canonical(model, prompt, history, *, session=None, extra=0, **overrides):
    prefix = (1, 2, 3) + (4,) * (len(history) + extra)
    routing = (0, 0, 0) + tuple(range(1, len(history) + 1)) + (len(history),) * extra
    kwargs: dict[str, Any] = {"committed_prefix": prefix, "outgoing_seg_map": routing,
                              "pool_positions": (len(prefix) - 1,), "feature_history": history}
    kwargs.update(overrides)
    with t.no_grad():
        return policy_module.canonical_fp_logits(model, prompt, replay_cache=session, **kwargs)


def _result(callback):
    try:
        return callback().detach().clone()
    except (RuntimeError, ValueError, TypeError, AssertionError, IndexError) as error:
        return type(error), str(error)


def _same_result(actual, expected):
    if isinstance(expected, t.Tensor):
        exact(actual, expected)
    else:
        assert actual == expected


def _clean(session):
    clean(session)
    assert session.report.current_history_bytes == 0


def _packed_tensors(output):
    return (output.values, output.layout.flat_indices, output.layout.valid_mask,
            output.layout.batch_indices, output.layout.segment_indices)


def _packed_exact(actual, expected):
    for name in ("batch_size", "n_segments", "feature_count"):
        assert getattr(actual.layout, name) == getattr(expected.layout, name)
    for actual_tensor, expected_tensor in zip(_packed_tensors(actual), _packed_tensors(expected), strict=True):
        assert actual_tensor.dtype == expected_tensor.dtype and actual_tensor.device == expected_tensor.device
        assert actual_tensor.shape == expected_tensor.shape
        assert actual_tensor.stride() == expected_tensor.stride()
        assert actual_tensor.storage_offset() == expected_tensor.storage_offset()
        exact(actual_tensor, expected_tensor)
        assert t.equal(actual_tensor.view(t.uint8), expected_tensor.view(t.uint8))
        for field in ("_dynamo_dynamic_indices", "_dynamo_dynamic_range", "_dynamo_hint_overrides",
                      "_specialize_on", "_has_dynamo_dim_marking"):
            assert getattr(actual_tensor, field) == getattr(expected_tensor, field)


def test_collection_history_reuses_ordered_actions_with_hidden_replay_enabled_in_both_arms(model):
    prompt = _prompt(model)
    histories = ((A,), (A, B), (A, B, A))
    results, reports = [], []
    with staged(model):
        for enabled in (False, True):
            with _context(model, enabled=enabled) as session:
                assert session.begin_collection(model)
                rng = t.get_rng_state(), t.cuda.get_rng_state()
                results.append([_canonical(model, prompt, history, session=session).detach().clone()
                                for history in histories])
                assert t.equal(rng[0], t.get_rng_state()) and t.equal(rng[1], t.cuda.get_rng_state())
                assert session.report.admitted_entries == 3, asdict(session.report)
                reports.append(session.report)
            _clean(session)
    exact(results[1], results[0])
    assert reports[0].history_hits == reports[0].history_admitted_entries == 0
    assert reports[1].history_action_calls == 6
    assert reports[1].history_hits == 4 and reports[1].history_misses == 2
    assert reports[1].history_admitted_entries == 2
    assert reports[1].history_guard_calls == reports[1].history_calls == 3
    assert reports[1].history_clone_ooms == reports[1].invalidations == 0


@pytest.mark.parametrize("changed", [
    ConditioningAction((3, 1), (0, 1), (True, True)),
    ConditioningAction((1, 3), (1, 0), (True, True)),
    ConditioningAction((1, 3), (0, 1), (False, True)),
])
def test_action_key_rebinding_order_magnitudes_and_mask_remain_native(model, changed):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        rebound = replace(A, feature_ids=changed.feature_ids, magnitudes=changed.magnitudes, mask=changed.mask)
        expected = _canonical(model, prompt, (rebound,), extra=1)
        actual = _canonical(model, prompt, (rebound,), session=session, extra=1)
        exact(actual, expected)
        assert session.report.history_hits == 0 and session.report.history_admitted_entries == 2
    _clean(session)


@pytest.mark.parametrize("mutation", ["encoder_weight", "encoder_hook", "policy_dependency",
                                     "packed_output_constructor", "packed_layout_constructor", "packed_property"])
def test_changes_between_collection_calls_invalidate_before_history_reuse(model, monkeypatch, mutation):
    prompt = _prompt(model)
    handle = None
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        assert session.report.history_admitted_entries == 1
        try:
            if mutation == "encoder_weight":
                with t.no_grad():
                    model.encoder.feature_embed.weight[1].add_(0.125)
            elif mutation == "encoder_hook":
                handle = model.encoder.register_forward_pre_hook(lambda _module, _args: None)
            elif mutation == "policy_dependency":
                original = policy_module._action_tensors
                monkeypatch.setattr(policy_module, "_action_tensors", lambda *args, **kwargs: original(*args, **kwargs))
            elif mutation == "packed_property":
                original_property = packed_module.PackedEncoderOutput.is_cuda
                monkeypatch.setattr(packed_module.PackedEncoderOutput, "is_cuda",
                                    property(lambda self: original_property.fget(self)))
            else:
                kind = (packed_module.PackedEncoderOutput if mutation == "packed_output_constructor"
                        else packed_module.PackedEncoderLayout)
                original = kind.__init__

                def initialize(self, *args, **kwargs):
                    return original(self, *args, **kwargs)

                monkeypatch.setattr(kind, "__init__", initialize)
            expected = _canonical(model, prompt, (A,), extra=1)
            actual = _canonical(model, prompt, (A,), session=session, extra=1)
            exact(actual, expected)
            assert session.report.history_hits == 0 and session.report.invalidations == 1
            assert session.report.current_history_bytes == 0
        finally:
            if handle is not None:
                handle.remove()
    _clean(session)


@pytest.mark.parametrize("name", ["PackedEncoderOutput", "PackedEncoderLayout"])
def test_packed_class_binding_replacement_declines_before_processing_history(model, monkeypatch, name):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        with t.no_grad():
            initial = model.encoder(prompt.enc_feature_ids, prompt.enc_magnitudes, prompt.enc_mask)
        original = getattr(packed_module, name)
        monkeypatch.setattr(packed_module, name, type("ForeignPackedClass", (original,), {}))
        with t.no_grad():
            result = session.encode_history(model, initial, (A,), device=model.get_device())
        assert result is None and session.report.invalidations == 1
        assert session.report.current_history_bytes == 0
    _clean(session)


def test_pending_duplicates_are_not_reused_before_successful_canonical_capture(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A, A), session=session)
        assert session.report.history_hits == 0 and session.report.history_misses == 2
        assert session.report.history_admitted_entries == 1 and not session._history_pending
        actual = _canonical(model, prompt, (A, A), session=session, extra=1)
        exact(actual, _canonical(model, prompt, (A, A), extra=1))
        assert session.report.history_hits == 2
    _clean(session)


def test_changed_guard_during_history_capture_never_commits_pending_outputs(model, monkeypatch):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        original_clone = session._clone_history_output

        def changed_clone(output):
            result = original_clone(output)
            with t.no_grad():
                model.encoder.feature_embed.weight[1].add_(0.125)
            return result

        monkeypatch.setattr(session, "_clone_history_output", changed_clone)
        _canonical(model, prompt, (A,), session=session)
        assert session.report.invalidations == 1
        assert session.report.history_admitted_entries == session.report.admitted_entries == 0
        assert session.report.current_bytes == 0 and not session._history_pending
        actual = _canonical(model, prompt, (A,), session=session, extra=1)
        exact(actual, _canonical(model, prompt, (A,), extra=1))
    _clean(session)


def test_history_operation_requires_a_fresh_guard_not_an_earlier_request(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        assert session.report.history_admitted_entries == 1
        with t.no_grad():
            encoder_out = model.encoder(prompt.enc_feature_ids, prompt.enc_magnitudes, prompt.enc_mask)
            model.encoder.feature_embed.weight[1].add_(0.125)
            result = session.encode_history(model, encoder_out, (A,), device=model.get_device())
        assert result is None
        assert session.report.history_hits == 0 and session.report.invalidations == 1
        assert session.report.current_history_bytes == 0
    _clean(session)


def test_history_memo_keeps_the_current_feature_head_live(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        first = _canonical(model, prompt, (A,), session=session).detach().clone()
        with t.no_grad():
            model.feature_pred_head.bias.add_(1.0)
        expected = _canonical(model, prompt, (A,), extra=1)
        actual = _canonical(model, prompt, (A,), session=session, extra=1)
        exact(actual, expected)
        assert not t.equal(actual, first)
        assert session.report.history_hits == 1 and session.report.invalidations == 0
    _clean(session)


@pytest.mark.parametrize("count", [31, 32, 33, 64, 65])
def test_packed_history_boundaries_preserve_all_bytes_metadata_and_dynamic_marks(model, monkeypatch, count):
    prompt = _prompt(model)
    action = ConditioningAction(tuple(index % 8 for index in range(count)),
                                tuple(index % 2 for index in range(count)), (True,) * count)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        original_clone = session._clone_history_output
        cloned_lengths = []

        def checked_clone(output):
            result = original_clone(output)
            if result is not None:
                _packed_exact(result, output)
                for copied, source in zip(_packed_tensors(result), _packed_tensors(output), strict=True):
                    assert copied.untyped_storage()._cdata != source.untyped_storage()._cdata
                cloned_lengths.append(result.values.size(1))
            return result

        monkeypatch.setattr(session, "_clone_history_output", checked_clone)
        _canonical(model, prompt, (action,), session=session)
        assert session.report.history_admitted_entries == 1
        with t.no_grad():
            initial = model.encoder(prompt.enc_feature_ids, prompt.enc_magnitudes, prompt.enc_mask)
            expected = policy_module._encode_canonical_state(model, prompt, (action,), device=model.get_device())
            actual = session.encode_history(model, initial, (action,), device=model.get_device())
        assert actual is not None and session.report.history_hits == 1
        _packed_exact(actual, expected)
        assert actual.layout.n_segments == 2 and int(actual.layout.valid_mask.sum()) == count + 2
        assert actual.values.size(1) % 32 == 0
        assert cloned_lengths == [((count + 31) // 32) * 32] * 2
        session.discard_pending_history()
        del actual, expected, initial
    _clean(session)


@pytest.mark.parametrize("write", ["versioned", "data", "independent_alias"])
def test_mutating_merged_state_cannot_change_retained_history(model, write):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        with t.no_grad():
            initial = model.encoder(prompt.enc_feature_ids, prompt.enc_magnitudes, prompt.enc_mask)
            expected = policy_module._encode_canonical_state(model, prompt, (A,), device=model.get_device())
            first = session.encode_history(model, initial, (A,), device=model.get_device())
            assert first is not None
            for tensor in _packed_tensors(first):
                target = tensor
                if write == "data":
                    target = tensor.data
                elif write == "independent_alias":
                    target = t.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                        tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())
                target.reshape(-1)[0] = (not bool(target.reshape(-1)[0])) if target.dtype == t.bool else 99
            second = session.encode_history(model, initial, (A,), device=model.get_device())
        assert second is not None
        _packed_exact(second, expected)
        for left, right in zip(_packed_tensors(first), _packed_tensors(second), strict=True):
            assert left.untyped_storage()._cdata != right.untyped_storage()._cdata
        assert session.report.history_hits == 2 and session.report.invalidations == 0
        session.discard_pending_history()
        del first, second, expected, initial, target, tensor
    _clean(session)


@pytest.mark.parametrize("phase", ["admission", "hit"])
@pytest.mark.parametrize("allocation_index", [1, 3, 5])
def test_partial_history_clone_oom_delegates_native_and_releases_copies(model, monkeypatch, phase, allocation_index):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        if phase == "hit":
            _canonical(model, prompt, (A,), session=session)
            assert session.report.history_admitted_entries == 1
        original_clone = cache_module._clone_for_cache
        original_packed_clone = session._clone_history_output
        state = {"inside": False, "index": 0, "failures": 0}
        refs = []

        def tensor_clone(value):
            if state["inside"]:
                state["index"] += 1
                if state["index"] == allocation_index:
                    state["failures"] += 1
                    raise t.cuda.OutOfMemoryError("injected optional history clone")
            result = original_clone(value)
            if state["inside"]:
                refs.append(weakref.ref(result))
            return result

        def packed_clone(*args, **kwargs):
            assert not state["inside"]
            state.update(inside=True, index=0)
            try:
                return original_packed_clone(*args, **kwargs)
            finally:
                state["inside"] = False

        monkeypatch.setattr(cache_module, "_clone_for_cache", tensor_clone)
        monkeypatch.setattr(session, "_clone_history_output", packed_clone)
        expected = _canonical(model, prompt, (A,), extra=1)
        actual = _canonical(model, prompt, (A,), session=session, extra=1)
        exact(actual, expected)
        assert state["failures"] > 0 and session.report.history_clone_ooms > 0
        assert session.report.admitted_entries > 0
        assert session.report.peak_bytes <= session.settings.max_cache_bytes
        assert all(reference() is None for reference in refs)
    _clean(session)


def test_history_pressure_preserves_the_existing_hidden_entry_budget(model):
    prompt = _prompt(model)
    prompt_bytes = sum(getattr(prompt, name).numel() * getattr(prompt, name).element_size()
                       for name in ("enc_feature_ids", "enc_magnitudes", "enc_mask", "prefix_seg_map"))
    budget = (4 + 5) * model.dec_config.n_embd * 2 + 2 * prompt_bytes
    reports, values = [], []
    with staged(model):
        for enabled in (False, True):
            with _context(model, enabled=enabled, budget=budget) as session:
                assert session.begin_collection(model)
                values.append([_canonical(model, prompt, (A,), session=session, extra=extra).detach().clone()
                               for extra in (0, 1)])
                assert session.report.admitted_entries == 2
                assert session.report.peak_bytes <= budget
                reports.append(session.report)
            _clean(session)
    exact(values[1], values[0])
    assert reports[1].history_admission_skips > 0


@pytest.mark.parametrize("cap", ["MAX_ENTRIES_PER_GROUP", "MAX_KEY_ELEMENTS_PER_GROUP", "MAX_KEY_BYTES_PER_GROUP"])
def test_successful_history_admission_is_evicted_when_hidden_capture_needs_shared_caps(model, monkeypatch, cap):
    prompt = _prompt(model)
    _, elements, key_bytes, reason = cache_module._key(prompt, (1, 2, 3, 4), (0, 0, 0, 1), (3,), (A,))
    assert reason == "eligible"
    bound = {"MAX_ENTRIES_PER_GROUP": 1, "MAX_KEY_ELEMENTS_PER_GROUP": elements,
             "MAX_KEY_BYTES_PER_GROUP": key_bytes}[cap]
    monkeypatch.setattr(cache_module, cap, bound)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        actual = _canonical(model, prompt, (A,), session=session)
        exact(actual, _canonical(model, prompt, (A,)))
        assert session.report.history_admitted_entries == session.report.history_evictions == 1
        assert session.report.admitted_entries == 1
        assert session.report.current_history_bytes == 0 and not session._history_pending
    _clean(session)


def test_new_history_copy_errors_propagate_and_fresh_group_reenters(model, monkeypatch):
    prompt = _prompt(model)
    report = cache_module.HiddenReplayReport()
    with staged(model):
        with pytest.raises(ValueError, match="non-OOM history copy failure"):
            with _context(model, report=report) as session:
                assert session.begin_collection(model)
                original_packed_clone = session._clone_history_output
                original_clone = cache_module._clone_for_cache
                calls = []

                def tensor_copy(value):
                    calls.append(1)
                    if len(calls) == 3:
                        raise ValueError("non-OOM history copy failure")
                    return original_clone(value)

                def failed_copy(output):
                    with monkeypatch.context() as temporary:
                        temporary.setattr(cache_module, "_clone_for_cache", tensor_copy)
                        return original_packed_clone(output)

                monkeypatch.setattr(session, "_clone_history_output", failed_copy)
                _canonical(model, prompt, (A,), session=session)
        _clean(session)
        assert report.current_bytes == report.active_groups == 0 and report.groups_failed == 1
        with _context(model, report=report) as following:
            assert following.begin_collection(model)
            _canonical(model, prompt, (A,), session=following)
            actual = _canonical(model, prompt, (A,), session=following, extra=1)
            exact(actual, _canonical(model, prompt, (A,), extra=1))
            assert report.history_hits > 0
        _clean(following)


@pytest.mark.parametrize("case", ["empty_pool", "reordered_pool", "routing", "empty_action", "shape", "all_masked"])
def test_native_history_error_order_after_a_successful_capture(model, case):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        history, kwargs = (A,), {}
        if case == "empty_pool":
            kwargs["pool_positions"] = ()
        elif case == "reordered_pool":
            kwargs["pool_positions"] = (3, 2)
        elif case == "routing":
            history = (A, B)
            kwargs["outgoing_seg_map"] = (0, 0, 0, 1, 99, 99)
        elif case == "empty_action":
            history = (ConditioningAction((), (), ()),)
        elif case == "shape":
            history = (ConditioningAction((1, 3), (0,), (True, True)),)
        else:
            history = (ConditioningAction(A.feature_ids, A.magnitudes, (False, False)),)
        before = session.report.history_admitted_entries
        before_bytes = session.report.current_bytes
        expected = _result(lambda: _canonical(model, prompt, history, extra=1, **kwargs))
        actual = _result(lambda: _canonical(model, prompt, history, session=session, extra=1, **kwargs))
        assert isinstance(expected, tuple), case
        _same_result(actual, expected)
        assert session.report.history_admitted_entries == before
        assert not session._history_pending
        assert session.report.current_bytes == before_bytes
    _clean(session)


def test_depth2_three_generated_updates_change_only_history_memoization(tmp_path, monkeypatch):
    native_stage = native_qualification.stage_frozen_decoder_linears
    native_configure = native_qualification.configure_grpo_trainable_parameters
    native_collect = native_qualification.collect_rlvr_group
    native_backward = native_qualification.backward_rlvr_group
    models, decoder_reports, sessions = [], [], []
    reports = [cache_module.HiddenReplayReport(), cache_module.HiddenReplayReport()]
    rng_states: list[list[tuple[t.Tensor, t.Tensor]]] = [[], []]
    active: dict[str, Any] = {"arm": None, "context": None, "session": None}

    def configure(model, **kwargs):
        models.append(model)
        return native_configure(model, **kwargs)

    @contextmanager
    def arm_context(model, arm, **kwargs):
        with native_stage(model, **kwargs) as decoder:
            decoder_reports.append(decoder)
            active["arm"] = arm
            try:
                yield decoder
            finally:
                try:
                    if active["context"] is not None:
                        active["context"].__exit__(*sys.exc_info())
                finally:
                    active.update(arm=None, context=None, session=None)

    @contextmanager
    def reference_context():
        with arm_context(models[-1], 0, max_shadow_bytes=64 * 1024**2, min_free_bytes=0,
                         train_feature_head_only=True, world_size=1):
            yield None

    @contextmanager
    def candidate_context(model, **kwargs):
        with arm_context(model, 1, **kwargs) as decoder:
            yield decoder

    def collect(model, *args, **kwargs):
        arm = active["arm"]
        assert arm in (0, 1) and active["context"] is None
        context = _context(model, enabled=bool(arm), report=reports[arm])
        session = context.__enter__()
        active.update(context=context, session=session)
        sessions.append(session)
        result = native_collect(model, *args, **kwargs, replay_cache=session)
        rng_states[arm].append((t.get_rng_state(), t.cuda.get_rng_state()))
        return result

    def replay(model, group, **kwargs):
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
    native_qualification.test_frozen_decoder_shadows_preserve_three_generated_updates(tmp_path, monkeypatch)
    assert len(models) == len(decoder_reports) == 2 and len(sessions) == 6
    assert all(item.applied and item.cleaned_up and item.fallback_calls == item.invalidated_modules == 0
               for item in decoder_reports)
    for report in reports:
        assert report.requested_groups == report.groups_completed == report.groups_closed == 3
        assert report.replay_hits > 0 and report.replay_misses == report.invalidations == 0
        assert report.groups_failed == report.groups_abandoned == report.inactive_groups == 0
        assert report.history_clone_ooms == 0
    assert reports[0].history_hits == 0 and reports[1].history_hits > 0
    assert reports[0].replay_hits == reports[1].replay_hits
    exact(rng_states[0], rng_states[1])
    for session in sessions:
        _clean(session)
