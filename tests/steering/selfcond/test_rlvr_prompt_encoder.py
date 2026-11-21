"""Native CUDA contracts for canonical prompt reuse alongside history memoization."""

from __future__ import annotations

import sys
import weakref
from contextlib import contextmanager
from dataclasses import asdict, replace
from typing import Any

import pytest
import torch as t

from steering.conditional.encoder_decoder import rlvr_policy as policy_module
from steering.conditional.encoder_decoder import rlvr_replay_cache as cache_module
from steering.conditional.encoder_decoder.config import FeaturePredictionBlock
from steering.conditional.encoder_decoder.sft_grpo import ConditioningAction
from tests.steering.selfcond import test_frozen_decoder_rlvr as native_qualification
from tests.steering.selfcond.test_rlvr_history_encoder import (
    A,
    B,
    _packed_exact,
    _packed_tensors,
    _result,
    _same_result,
)
from tests.steering.selfcond.test_rlvr_history_encoder import (
    model as native_model,
)
from tests.steering.selfcond.test_rlvr_replay_cache import (
    clean,
    cuda_contract,  # noqa: F401 -- native deterministic CUDA fixture
    exact,
    staged,
)

pytestmark = pytest.mark.slow
model = native_model


@pytest.fixture(autouse=True)
def _install_test_owned_cpu_encoder_reference():
    """Keep the compiled CUDA encoder and packed cross-attention active."""
    yield


def _context(model, *, enabled=True, report=None, budget=16 * 1024**2):
    return policy_module.rlvr_group_replay_cache(
        model, settings=cache_module.HiddenReplaySettings(
            max_cache_bytes=budget, min_free_bytes=0, history_encoder=True, prompt_encoder=enabled),
        train_feature_head_only=True, world_size=1,
        report=cache_module.HiddenReplayReport() if report is None else report,
    )


def _prompt(model, *, segments=1, features=2):
    device = model.get_device()
    indices = t.arange(segments * features, device=device).reshape(1, segments, features)
    return policy_module.RLVRPromptReplay(
        (1, 2, 3, 4), indices % 8, indices % 2, t.ones_like(indices, dtype=t.bool),
        t.tensor([[0, 0, segments - 1]], device=device), segments,
    )


def _inputs(prompt, device):
    return (prompt.enc_feature_ids.to(device), prompt.enc_magnitudes.to(device), prompt.enc_mask.to(device))


def _canonical(model, prompt, history=(), *, session=None, extra=0, **overrides):
    last = prompt.encoder_slot_count - 1
    prefix = (1, 2, 3) + (4,) * (len(history) + extra)
    routing = (0, 0, last) + tuple(range(last + 1, last + len(history) + 1)) + (last + len(history),) * extra
    kwargs: dict[str, Any] = {"committed_prefix": prefix, "outgoing_seg_map": routing,
                              "pool_positions": (len(prefix) - 1,), "feature_history": history}
    kwargs.update(overrides)
    with t.no_grad():
        return policy_module.canonical_fp_logits(model, prompt, replay_cache=session, **kwargs)


def _clean(session):
    clean(session)
    assert session.report.current_history_bytes == session.report.current_canonical_prompt_bytes == 0
    assert session._canonical_prompt_entry is session._canonical_prompt_pending is None


@contextmanager
def _encoder_calls(model):
    """Observe the original eager encoder entry without changing its bindings."""
    code = type(model.encoder).forward.__code__
    counts = {"calls": 0}
    previous = sys.getprofile()
    assert previous is None

    def observe(frame, event, arg):
        if event == "call" and frame.f_code is code and frame.f_locals.get("self") is model.encoder:
            counts["calls"] += 1

    sys.setprofile(observe)
    try:
        yield counts
    finally:
        assert sys.getprofile() is observe
        sys.setprofile(previous)


def test_collection_changes_only_prompt_reuse_with_prior_optimizations_in_both_arms(model):
    prompt = _prompt(model, segments=2)
    histories = ((), (A,), (A, B), (A, B, A))
    results, reports = [], []
    with staged(model):
        for enabled in (False, True):
            with _context(model, enabled=enabled) as session:
                assert session.begin_collection(model)
                rng = t.get_rng_state(), t.cuda.get_rng_state()
                results.append([_canonical(model, prompt, history, session=session).detach().clone()
                                for history in histories])
                exact((t.get_rng_state(), t.cuda.get_rng_state()), rng)
                assert session.report.admitted_entries == 4, asdict(session.report)
                assert session.report.history_hits == 4 and session.report.history_misses == 2
                assert session.report.history_guard_calls == 3
                reports.append(session.report)
            _clean(session)
    exact(results[1], results[0])
    assert reports[0].canonical_prompt_hits == reports[0].canonical_prompt_admitted_entries == 0
    assert reports[1].canonical_prompt_hits == 3 and reports[1].canonical_prompt_misses == 1
    assert reports[1].canonical_prompt_admitted_entries == 1
    assert reports[1].canonical_prompt_clone_ooms == reports[1].invalidations == 0


def test_empty_history_runs_native_even_with_a_committed_prompt(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        for extra in (0, 1):
            with _encoder_calls(model) as observed:
                actual = _canonical(model, prompt, session=session, extra=extra)
            assert observed["calls"] == 1
            exact(actual, _canonical(model, prompt, extra=extra))
        assert session.report.canonical_prompt_hits == 0
        assert session.report.canonical_prompt_misses == 2
        assert session.report.history_guard_calls == 0
        actual = _canonical(model, prompt, (A,), session=session, extra=1)
        exact(actual, _canonical(model, prompt, (A,), extra=1))
        assert session.report.canonical_prompt_hits == 1
    _clean(session)


@pytest.mark.parametrize(("segments", "features"), [(1, 31), (1, 32), (1, 33), (2, 16), (2, 33)])
def test_multisegment_prompt_bytes_layout_dynamic_marks_and_independent_output(model, monkeypatch, segments, features):
    prompt = _prompt(model, segments=segments, features=features)
    # Include a hole so equal packed length alone cannot establish layout parity.
    if segments > 1:
        prompt.enc_mask[0, 0, 1] = False
    cloned = []
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        original = session._clone_canonical_prompt_output

        def checked(output):
            result = original(output)
            if result is not None:
                _packed_exact(result, output)
                for copied, source in zip(_packed_tensors(result), _packed_tensors(output), strict=True):
                    assert copied.untyped_storage()._cdata != source.untyped_storage()._cdata
                cloned.append((result.layout.n_segments, result.values.size(1)))
            return result

        monkeypatch.setattr(session, "_clone_canonical_prompt_output", checked)
        _canonical(model, prompt, session=session)
        assert session.report.canonical_prompt_admitted_entries == 1
        entry = session._canonical_prompt_entry
        assert entry is not None
        for copied, source in zip(entry.inputs, _inputs(prompt, model.get_device()), strict=True):
            exact(copied, source)
            assert copied.dtype == source.dtype and copied.shape == source.shape
            assert copied.stride() == source.stride() and copied.storage_offset() == source.storage_offset()
            assert copied.untyped_storage()._cdata != source.untyped_storage()._cdata
        with t.no_grad():
            expected = policy_module._encode_canonical_state(model, prompt, (A,), device=model.get_device())
            actual = session.encode_prompt_and_history(model, _inputs(prompt, model.get_device()), (A,),
                                                       device=model.get_device())
        assert actual is not None and session.report.canonical_prompt_hits == 1
        _packed_exact(actual, expected)
        assert actual.layout.n_segments == segments + 1
        assert int(actual.layout.valid_mask.sum()) == int(prompt.enc_mask.sum()) + 2
        expected_packed = ((int(prompt.enc_mask.sum()) + 31) // 32) * 32
        assert cloned == [(segments, expected_packed)] * 2
        for source, merged in zip(_packed_tensors(entry.output), _packed_tensors(actual), strict=True):
            assert source.untyped_storage()._cdata != merged.untyped_storage()._cdata
        session.discard_pending_history()
        del entry, actual, expected, source, merged, copied
    _clean(session)


@pytest.mark.parametrize("field", ["enc_feature_ids", "enc_magnitudes", "enc_mask"])
@pytest.mark.parametrize("write", ["versioned", "data", "independent_alias"])
def test_fresh_actual_input_content_rejects_stale_prompt_even_without_version_changes(model, field, write):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        tensor = getattr(prompt, field)
        target = tensor if write == "versioned" else tensor.data
        if write == "independent_alias":
            target = t.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())
        target.reshape(-1)[0] = False if field == "enc_mask" else 1
        expected = _canonical(model, prompt, (A,))
        actual = _canonical(model, prompt, (A,), session=session)
        exact(actual, expected)
        assert session.report.canonical_prompt_hits == 0
        assert session.report.canonical_prompt_misses == 2
        assert session.report.invalidations == 0
    _clean(session)


@pytest.mark.parametrize("change", ["cpu_rebinding", "stride", "offset"])
def test_prompt_matching_uses_actual_device_inputs_and_exact_metadata(model, change):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        fields = {}
        for name in ("enc_feature_ids", "enc_magnitudes", "enc_mask"):
            source = getattr(prompt, name)
            if change == "cpu_rebinding":
                value = source.cpu()
            elif change == "stride":
                value = t.stack((source, source), dim=-1)[..., 0]
            else:
                value = t.cat((source.new_zeros(1), source.reshape(-1)))[1:].view_as(source)
            fields[name] = value
        rebound = replace(prompt, **fields)
        actual = _canonical(model, rebound, (A,), session=session)
        exact(actual, _canonical(model, rebound, (A,)))
        assert session.report.canonical_prompt_hits == int(change == "cpu_rebinding")
        assert session.report.invalidations == 0
    _clean(session)


@pytest.mark.parametrize("failure", [None, "enc_mask", "history"])
def test_device_conversion_is_evaluated_once_in_native_order_without_retry(model, failure):
    prompt = _prompt(model)
    calls = []

    class ToOnce:
        def __init__(self, name, value):
            self.name, self.value = name, value

        def to(self, device):
            calls.append(self.name)
            if self.name == failure:
                raise ValueError("conversion sentinel")
            return self.value.to(device)

    proxy = replace(prompt, **{name: ToOnce(name, getattr(prompt, name)) for name in
                               ("enc_feature_ids", "enc_magnitudes", "enc_mask")})
    history = (ConditioningAction((), (), ()),) if failure == "history" else (A,)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        with t.no_grad():
            with _encoder_calls(model) as observed:
                try:
                    result = policy_module._encode_canonical_state(model, proxy, history,
                                                                   device=model.get_device(), replay_cache=session)
                except ValueError as error:
                    assert failure is not None
                    assert ("conversion sentinel" in str(error) if failure == "enc_mask"
                            else "at least one feature" in str(error))
                else:
                    assert failure is None
            if failure is None:
                expected = policy_module._encode_canonical_state(model, prompt, history, device=model.get_device())
                _packed_exact(result, expected)
                del result, expected
        assert calls == ["enc_feature_ids", "enc_magnitudes", "enc_mask"]
        assert observed["calls"] == int(failure is None)
        session.discard_pending_history()
    _clean(session)


def test_changed_then_reverted_prompt_reuses_original_entry_with_the_live_head(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        first_entry = session._canonical_prompt_entry
        assert first_entry is not None
        prompt.enc_feature_ids.data[0, 0, 0] = 2
        actual = _canonical(model, prompt, (A,), session=session)
        exact(actual, _canonical(model, prompt, (A,)))
        assert session.report.canonical_prompt_hits == 0
        assert session._canonical_prompt_entry is first_entry
        prompt.enc_feature_ids.data[0, 0, 0] = 0
        with t.no_grad():
            model.feature_pred_head.bias.add_(1.0)
        actual = _canonical(model, prompt, (A,), session=session, extra=1)
        exact(actual, _canonical(model, prompt, (A,), extra=1))
        assert session.report.canonical_prompt_hits == 1 and session.report.invalidations == 0
        assert session._canonical_prompt_entry is first_entry
        del first_entry
    _clean(session)


def test_pending_prompt_is_not_reused_before_successful_canonical_capture(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session, t.no_grad():
        assert session.begin_collection(model)
        request = session.prepare_request(model, prompt, committed_prefix=(1, 2, 3),
            outgoing_seg_map=(0, 0, 0), pool_positions=(2,), feature_history=(), train_feature_head_only=False)
        assert request is not None
        first = session.encode_prompt_and_history(model, _inputs(prompt, model.get_device()), (), device=model.get_device())
        assert first is not None and session._canonical_prompt_pending is not None
        assert session._canonical_prompt_entry is None
        second = session.encode_prompt_and_history(model, _inputs(prompt, model.get_device()), (A,), device=model.get_device())
        assert second is not None
        assert session.report.canonical_prompt_hits == session.report.canonical_prompt_admitted_entries == 0
        assert session.report.canonical_prompt_misses == 2
        session.discard_pending_history()
        assert session.report.current_bytes == 0
        del first, second, request
    _clean(session)


def test_postcanonical_guard_rejects_pending_prompt_after_encoder_mutation(model, monkeypatch):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        original = session._stage_canonical_prompt

        def changed(*args, **kwargs):
            result = original(*args, **kwargs)
            with t.no_grad():
                model.encoder.feature_embed.weight[0].add_(0.125)
            return result

        monkeypatch.setattr(session, "_stage_canonical_prompt", changed)
        _canonical(model, prompt, session=session)
        assert session.report.invalidations == 1
        assert session.report.canonical_prompt_admitted_entries == session.report.admitted_entries == 0
        assert session.report.current_bytes == 0 and session._canonical_prompt_pending is None
    _clean(session)


def test_prompt_and_history_hits_require_a_fresh_guard_after_an_earlier_request(model):
    prompt = _prompt(model)
    with staged(model), _context(model) as session, t.no_grad():
        assert session.begin_collection(model)
        _canonical(model, prompt, (A,), session=session)
        request = session.prepare_request(model, prompt, committed_prefix=(1, 2, 3, 4, 4),
            outgoing_seg_map=(0, 0, 0, 1, 1), pool_positions=(4,), feature_history=(A,), train_feature_head_only=False)
        assert request is not None
        model.encoder.feature_embed.weight[0].add_(0.125)
        result = session.encode_prompt_and_history(model, _inputs(prompt, model.get_device()), (A,), device=model.get_device())
        assert result is None
        assert session.report.canonical_prompt_hits == session.report.history_hits == 0
        assert session.report.invalidations == 1 and session.report.current_bytes == 0
    _clean(session)


@pytest.mark.parametrize(("phase", "allocation_index"),
                         [("admission", index) for index in (1, 2, 3, 4, 6, 8)]
                         + [("hit", index) for index in (1, 3, 5)])
def test_partial_prompt_copy_oom_preserves_native_result_and_releases_owned_tensors(model, monkeypatch,
                                                                                  phase, allocation_index):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        if phase == "hit":
            _canonical(model, prompt, session=session)
            assert session.report.canonical_prompt_admitted_entries == 1
        original_copy = cache_module._clone_for_cache
        name = "_stage_canonical_prompt" if phase == "admission" else "_clone_canonical_prompt_output"
        original = getattr(session, name)
        state = {"inside": False, "index": 0, "failures": 0}
        refs = []

        def copy(value):
            if state["inside"]:
                state["index"] += 1
                if state["index"] == allocation_index:
                    state["failures"] += 1
                    raise t.cuda.OutOfMemoryError("injected optional canonical prompt copy")
            result = original_copy(value)
            if state["inside"]:
                refs.append(weakref.ref(result))
            return result

        def optional(*args, **kwargs):
            assert not state["inside"]
            state.update(inside=True, index=0)
            try:
                return original(*args, **kwargs)
            finally:
                state["inside"] = False

        monkeypatch.setattr(cache_module, "_clone_for_cache", copy)
        monkeypatch.setattr(session, name, optional)
        history = (A,) if phase == "hit" else ()
        actual = _canonical(model, prompt, history, session=session, extra=1)
        exact(actual, _canonical(model, prompt, history, extra=1))
        assert state["failures"] > 0 and session.report.canonical_prompt_clone_ooms > 0
        assert session.report.admitted_entries > 0 and session.report.peak_bytes <= session.settings.max_cache_bytes
        assert all(reference() is None for reference in refs)
    _clean(session)


@pytest.mark.parametrize("write", ["versioned", "data", "independent_alias"])
def test_mutating_appended_state_cannot_change_owned_prompt(model, write):
    prompt = _prompt(model, segments=2)
    with staged(model), _context(model) as session, t.no_grad():
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        inputs = _inputs(prompt, model.get_device())
        first = session.encode_prompt_and_history(model, inputs, (A,), device=model.get_device())
        assert first is not None
        for tensor in _packed_tensors(first):
            target = tensor if write == "versioned" else tensor.data
            if write == "independent_alias":
                target = t.empty(0, dtype=tensor.dtype, device=tensor.device).set_(
                    tensor.untyped_storage(), tensor.storage_offset(), tensor.shape, tensor.stride())
            target.reshape(-1)[0] = False if target.dtype == t.bool else 99
        second = session.encode_prompt_and_history(model, inputs, (A,), device=model.get_device())
        expected = policy_module._encode_canonical_state(model, prompt, (A,), device=model.get_device())
        assert second is not None and session.report.canonical_prompt_hits == 2
        _packed_exact(second, expected)
        session.discard_pending_history()
        del first, second, expected, tensor, target
    _clean(session)


def test_non_oom_prompt_copy_error_propagates_and_fresh_session_reenters(model, monkeypatch):
    prompt = _prompt(model)
    report = cache_module.HiddenReplayReport()
    with staged(model):
        with pytest.raises(ValueError, match="non-OOM prompt copy failure"):
            with _context(model, report=report) as session:
                assert session.begin_collection(model)
                original_stage = session._stage_canonical_prompt
                original_copy = cache_module._clone_for_cache
                calls = []

                def copy(value):
                    calls.append(1)
                    if len(calls) == 2:
                        raise ValueError("non-OOM prompt copy failure")
                    return original_copy(value)

                def failed_stage(*args, **kwargs):
                    with monkeypatch.context() as temporary:
                        temporary.setattr(cache_module, "_clone_for_cache", copy)
                        return original_stage(*args, **kwargs)

                monkeypatch.setattr(session, "_stage_canonical_prompt", failed_stage)
                _canonical(model, prompt, session=session)
        _clean(session)
        assert report.groups_failed == 1 and report.active_groups == report.current_bytes == 0
        with _context(model, report=report) as following:
            assert following.begin_collection(model)
            _canonical(model, prompt, session=following)
            actual = _canonical(model, prompt, (A,), session=following)
            exact(actual, _canonical(model, prompt, (A,)))
            assert report.canonical_prompt_hits == 1
        _clean(following)


@pytest.mark.parametrize("where", ["encoder", "append"])
def test_native_compute_oom_is_propagated_without_retry_or_optional_copy_suppression(model, monkeypatch, where):
    prompt = _prompt(model)
    calls = []
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        before = session.report.current_bytes
        before_optional_ooms = session.report.optional_copy_ooms

        def fail(*args, **kwargs):
            calls.append(1)
            raise t.cuda.OutOfMemoryError("native compute OOM sentinel")

        with monkeypatch.context() as temporary:
            if where == "encoder":
                prompt.enc_feature_ids[0, 0, 0] = 2
                original_guard = session._check_guard

                def checked(*args, **kwargs):
                    result = original_guard(*args, **kwargs)
                    assert result[0]
                    # Fault after admission so the exception comes from the
                    # native miss inside the synchronous operation under test.
                    temporary.setattr(model.encoder, "forward", fail)
                    return result

                temporary.setattr(session, "_check_guard", checked)
            else:
                temporary.setattr(session, "_append_history", fail)
            with t.no_grad(), pytest.raises(t.cuda.OutOfMemoryError, match="native compute OOM sentinel"):
                session.encode_prompt_and_history(model, _inputs(prompt, model.get_device()), (A,),
                                                  device=model.get_device())
        assert calls == [1]
        assert session.report.optional_copy_ooms == before_optional_ooms
        assert session.report.current_bytes == before
        assert session._canonical_prompt_pending is None and not session._history_pending
        assert session.report.canonical_prompt_hits == int(where == "append")
    _clean(session)


@pytest.mark.parametrize("case", ["empty_action", "all_masked", "routing"])
def test_native_error_after_prompt_hit_never_publishes_pending_history_or_leaks_copy(model, case):
    prompt = _prompt(model)
    with staged(model), _context(model) as session:
        assert session.begin_collection(model)
        _canonical(model, prompt, session=session)
        history, kwargs = (A,), {}
        if case == "empty_action":
            history = (ConditioningAction((), (), ()),)
        elif case == "all_masked":
            history = (ConditioningAction(A.feature_ids, A.magnitudes, (False, False)),)
        else:
            kwargs["outgoing_seg_map"] = (0, 0, 0, 99)
        before = session.report.current_bytes
        expected = _result(lambda: _canonical(model, prompt, history, **kwargs))
        actual = _result(lambda: _canonical(model, prompt, history, session=session, **kwargs))
        assert isinstance(expected, tuple)
        _same_result(actual, expected)
        assert session.report.canonical_prompt_hits == 1
        assert session.report.history_admitted_entries == 0
        assert session.report.current_bytes == before
        assert session._canonical_prompt_pending is None and not session._history_pending
    _clean(session)


def test_prompt_memo_is_evicted_before_hidden_capture_exceeds_shared_bytes(model):
    prompt = _prompt(model)
    with staged(model):
        with _context(model) as sizing:
            assert sizing.begin_collection(model)
            _canonical(model, prompt, session=sizing)
            memo_bytes = sizing.report.current_canonical_prompt_bytes
            hidden_bytes = sizing.report.current_bytes - memo_bytes
            assert memo_bytes > hidden_bytes > 0
        _clean(sizing)
        with _context(model, budget=memo_bytes) as session:
            assert session.begin_collection(model)
            actual = _canonical(model, prompt, session=session)
            exact(actual, _canonical(model, prompt))
            assert session.report.canonical_prompt_admitted_entries == session.report.canonical_prompt_evictions == 1
            assert session.report.admitted_entries == 1
            assert session.report.current_canonical_prompt_bytes == 0
            assert session.report.current_bytes == hidden_bytes and session.report.peak_bytes <= memo_bytes
        _clean(session)


@pytest.mark.parametrize("cap", ["MAX_ENTRIES_PER_GROUP", "MAX_KEY_ELEMENTS_PER_GROUP", "MAX_KEY_BYTES_PER_GROUP"])
def test_shared_key_caps_preserve_hidden_priority_over_successfully_staged_prompt(model, monkeypatch, cap):
    prompt = _prompt(model)
    with staged(model):
        with _context(model) as sizing:
            assert sizing.begin_collection(model)
            _canonical(model, prompt, session=sizing)
            entry = sizing._canonical_prompt_entry
            assert entry is not None
            _, hidden_elements, hidden_bytes, reason = cache_module._key(prompt, (1, 2, 3), (0, 0, 0), (2,), ())
            assert reason == "eligible"
            bound = {"MAX_ENTRIES_PER_GROUP": 1,
                     "MAX_KEY_ELEMENTS_PER_GROUP": max(entry.key_elements, hidden_elements),
                     "MAX_KEY_BYTES_PER_GROUP": max(entry.key_bytes, hidden_bytes)}[cap]
            del entry
        _clean(sizing)
        monkeypatch.setattr(cache_module, cap, bound)
        with _context(model) as session:
            assert session.begin_collection(model)
            actual = _canonical(model, prompt, session=session)
            exact(actual, _canonical(model, prompt))
            assert session.report.canonical_prompt_admitted_entries == session.report.canonical_prompt_evictions == 1
            assert session.report.admitted_entries == 1 and session.report.current_canonical_prompt_bytes == 0
        _clean(session)


@pytest.mark.parametrize("head_block", [FeaturePredictionBlock.MLP, FeaturePredictionBlock.TRANSFORMER])
def test_depth2_three_generated_updates_change_only_canonical_prompt_memoization(tmp_path, monkeypatch, head_block):
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
    native_qualification.test_frozen_decoder_shadows_preserve_three_generated_updates(
        tmp_path, monkeypatch, head_block=head_block,
    )
    assert len(models) == len(decoder_reports) == 2 and len(sessions) == 6
    assert all(item.applied and item.cleaned_up and item.fallback_calls == item.invalidated_modules == 0
               for item in decoder_reports)
    for report in reports:
        assert report.requested_groups == report.groups_completed == report.groups_closed == 3
        assert report.replay_hits > 0 and report.replay_misses == report.invalidations == 0
        assert report.groups_failed == report.groups_abandoned == report.inactive_groups == 0
        assert report.history_hits > 0 and report.history_clone_ooms == report.canonical_prompt_clone_ooms == 0
    assert reports[0].canonical_prompt_hits == 0 and reports[1].canonical_prompt_hits > 0
    assert reports[0].history_hits == reports[1].history_hits
    assert reports[0].replay_hits == reports[1].replay_hits
    exact(rng_states[0], rng_states[1])
    for session in sessions:
        _clean(session)
