"""CPU-only lifecycle checks for explicit RLVR replay-cache sessions."""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.encoder_decoder import _rlvr_replay_guard as guard
from steering.conditional.encoder_decoder import rlvr_policy as policy
from steering.conditional.encoder_decoder.rlvr_replay_cache import (
    HiddenReplayReport,
    HiddenReplaySettings,
)


def scope(model, report, *, budget=1024, head_only=True, world_size=1):
    return policy.rlvr_group_replay_cache(model, settings=HiddenReplaySettings(budget, 0),
                                         train_feature_head_only=head_only, world_size=world_size, report=report)


@pytest.mark.parametrize("changes", [{"budget": 0}, {"head_only": False}, {"world_size": 2}])
def test_inactive_factory_does_not_inspect_model_or_framework(monkeypatch, changes):
    class Uninspectable:
        def __getattribute__(self, name):
            raise AssertionError(f"Inactive scope inspected model: {name}")

    def forbidden(*_args, **_kwargs):
        pytest.fail("Inactive factory initialized or inspected the native framework guard")

    monkeypatch.setattr(guard.NativeFrozenState, "capture", forbidden)
    report = HiddenReplayReport()
    canonical = policy.canonical_fp_logits
    with scope(Uninspectable(), report, **changes) as session:
        assert session is None
        assert policy.canonical_fp_logits is canonical
    assert report.requested_groups == report.inactive_groups == 1
    assert report.active_groups == report.groups_closed == report.admitted_entries == 0
    assert report.cleaned_up and not report.applied


@pytest.mark.parametrize("same_model,same_report", [(True, True), (True, False), (False, True)])
def test_overlapping_model_or_report_rejected_without_leaking_ownership(same_model, same_report):
    first_model = t.nn.Module()
    second_model = first_model if same_model else t.nn.Module()
    first_report = HiddenReplayReport()
    second_report = first_report if same_report else HiddenReplayReport()
    with scope(first_model, first_report) as first:
        with pytest.raises(RuntimeError, match="overlap|owned|active"):
            with scope(second_model, second_report):
                pytest.fail("Overlapping ownership was admitted")
        assert first_report.active_groups == 1
    assert first_report.active_groups == 0 and first_report.groups_closed == 1
    with scope(second_model, second_report) as second:
        assert second is not first
    assert second_report.active_groups == 0 and second_report.cleaned_up


def test_independent_models_and_reports_can_have_independent_sessions():
    first_report, second_report = HiddenReplayReport(), HiddenReplayReport()
    with scope(t.nn.Module(), first_report) as first, scope(t.nn.Module(), second_report) as second:
        assert first is not second and first_report.active_groups == second_report.active_groups == 1
    assert first_report.cleaned_up and second_report.cleaned_up
    assert first_report.groups_closed == second_report.groups_closed == 1


def test_exception_releases_model_owner_for_fresh_session():
    model = t.nn.Module()
    report = HiddenReplayReport()
    with pytest.raises(ValueError, match="native failure"):
        with scope(model, report):
            raise ValueError("native failure")
    assert report.groups_failed == report.groups_closed == 1 and report.active_groups == 0
    with scope(model, report):
        assert report.active_groups == 1
    assert report.groups_closed == 2 and report.active_groups == 0
    assert report.current_bytes == report.max_owned_tensor_objects_alive_after_cleanup == 0


def test_unavailable_guard_refuses_capture_without_owning_tensors(monkeypatch):
    calls = []

    def unavailable(*_args, **_kwargs):
        calls.append(1)
        return None, "unavailable_framework_capability"

    monkeypatch.setattr(guard.NativeFrozenState, "capture", unavailable)
    model = t.nn.Module()
    prompt = policy.RLVRPromptReplay((1, 2), t.tensor([[[1]]]), t.tensor([[[0]]]),
                                    t.tensor([[[True]]]), t.tensor([[0]]), 1)
    report = HiddenReplayReport()
    with scope(model, report) as session, t.no_grad():
        assert session.begin_collection(model)
        request = session.prepare_request(model, prompt, committed_prefix=(1,), outgoing_seg_map=(0,),
                                          pool_positions=(0,), feature_history=(), train_feature_head_only=False)
        assert request is None and calls == [1]
        assert report.capture_skips == report.invalidations == 1
        assert report.capture_attempts == report.admitted_entries == report.current_bytes == 0
    assert report.cleaned_up and report.active_groups == 0
