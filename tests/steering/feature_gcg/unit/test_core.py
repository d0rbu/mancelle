from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
import torch as t
import yaml

from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.edit_space import GroupResult
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.profiler import AxBenchProfiler
from tests.steering.feature_gcg.unit.integer_edit_space import IntegerEditSpace, IntGroup


@dataclass
class Cfg:
    max_steps: int = 10
    candidate_batch_size: int = 4
    temperature: float = 0.1
    master_seed: int = 0
    universal_minibatch_size: int = 16


def _objective_batch(items):
    # item = (member, target); minimize |target - member| averaged later by the core.
    return [float(abs(target - member)) for (member, target) in items]


def _grad_batch(items, step):
    return [t.zeros(1) for _ in items]


def test_converges_toward_member_mean_and_retains_best():
    space = IntegerEditSpace()
    groups = [IntGroup(key_str="g", members=[10, 10], is_control=False)]
    results = run_gcg(edit_space=space, groups=groups, objective_batch=_objective_batch,
                      gradient_provider_batch=_grad_batch, cfg=Cfg())
    assert len(results) == 1
    payload = results[0].payload
    # init=10 (members[0]); objective already 0 -> no improvement possible, retained=init.
    assert payload["target"] == 10
    assert payload["objective"] == 0.0


def test_run_gcg_emits_per_phase_profile_regions(tmp_path: Path) -> None:
    """Issue #657: the SHARED production GCG loop emits per-phase sub-regions +
    step counters when a profiler is bound, so a profiling run shows WHERE
    feature-GCG time goes (gradient vs candidate-eval). Without a bound profiler
    these are no-ops (every other run_gcg test above exercises that path and
    asserts the search result is unchanged)."""
    # Suboptimal init (members[0]=0, median 10) -> the loop actually steps and
    # accepts, exercising gradient + candidate_eval + steps_accepted.
    space = IntegerEditSpace()
    groups = [IntGroup(key_str="g", members=[0, 20], is_control=False)]

    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={"run_tag": "unit-test"}):
        run_gcg(
            edit_space=space,
            groups=groups,
            objective_batch=_objective_batch,
            gradient_provider_batch=_grad_batch,
            cfg=Cfg(max_steps=5),
        )

    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text("utf-8"))
    region_names = {r["name"] for r in summary["regions"]}
    counter_names = {c["name"] for c in summary["counters"]}
    assert "feature_gcg.run_gcg.gradient" in region_names
    assert "feature_gcg.run_gcg.rank_candidates" in region_names
    assert "feature_gcg.run_gcg.candidate_eval" in region_names
    assert "feature_gcg.run_gcg.steps" in counter_names
    steps = next(
        c for c in summary["counters"] if c["name"] == "feature_gcg.run_gcg.steps"
    )
    assert steps["total"] == 5  # one group x max_steps=5
    region_counts = {row["name"]: row["count"] for row in summary["regions"]}
    assert region_counts["feature_gcg.run_gcg.gradient"] == 5
    assert region_counts["feature_gcg.run_gcg.rank_candidates"] == 5
    # Ranking still runs after the tiny visited space saturates; candidate eval
    # is correctly skipped on steps whose sampler returns no unseen candidate.
    assert 0 < region_counts["feature_gcg.run_gcg.candidate_eval"] <= 5


@pytest.mark.parametrize(
    "groups,cfg",
    [
        ([IntGroup(key_str="control", members=[1], is_control=True)], Cfg()),
        ([IntGroup(key_str="zero-step", members=[1], is_control=False)], Cfg(max_steps=0)),
    ],
)
def test_control_and_zero_step_runs_emit_no_rank_region(
    tmp_path: Path,
    groups: list[IntGroup],
    cfg: Cfg,
) -> None:
    profile_dir = tmp_path / groups[0].key_str
    with AxBenchProfiler(profile_dir, metadata={"run_tag": groups[0].key_str}):
        run_gcg(
            edit_space=IntegerEditSpace(),
            groups=groups,
            objective_batch=_objective_batch,
            gradient_provider_batch=_grad_batch,
            cfg=cfg,
        )

    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text("utf-8"))
    region_names = {row["name"] for row in summary["regions"]}
    assert "feature_gcg.run_gcg.gradient" not in region_names
    assert "feature_gcg.run_gcg.rank_candidates" not in region_names
    assert "feature_gcg.run_gcg.candidate_eval" not in region_names


def test_walks_to_median_when_init_is_suboptimal():
    space = IntegerEditSpace()
    # members [4, 10, 16]: init = members[0] = 4 is suboptimal; the abs-mean
    # objective is minimized at the median 10 (obj 4.0 < init obj 6.0). The loop
    # must demonstrably walk 4 -> 10 via accepted strict-improvement steps.
    groups = [IntGroup(key_str="g", members=[4, 10, 16], is_control=False)]
    results = run_gcg(edit_space=space, groups=groups, objective_batch=_objective_batch,
                      gradient_provider_batch=_grad_batch, cfg=Cfg(max_steps=20))
    payload = results[0].payload
    assert payload["target"] == 10
    assert payload["objective"] == 4.0
    assert payload["accepted_steps"] >= 1


def test_control_group_short_circuits():
    space = IntegerEditSpace()
    groups = [IntGroup(key_str="c", members=[1, 2], is_control=True)]
    results = run_gcg(edit_space=space, groups=groups, objective_batch=_objective_batch,
                      gradient_provider_batch=_grad_batch, cfg=Cfg())
    assert results[0].payload == {"control": True}


def test_progress_events_have_no_attack_row_id():
    space = IntegerEditSpace()
    events: list[dict] = []
    groups = [IntGroup(key_str="g", members=[10, 10], is_control=False)]
    run_gcg(edit_space=space, groups=groups, objective_batch=_objective_batch,
            gradient_provider_batch=_grad_batch, cfg=Cfg(),
            progress_callback=events.append)
    assert events and all("attack_row_id" not in e for e in events)
    assert any(e["event"] == "group_start" for e in events)


def test_accept_is_strict_rejects_boundary_ties():
    # members [4, 6]: abs-mean objective is flat (=1.0) on [4, 6].
    # init target = 4: objective = mean(|4-4|, |4-6|) = mean(0, 2) = 1.0
    # candidate 5:     objective = mean(|5-4|, |5-6|) = mean(1, 1) = 1.0
    # Strict `<` rejects the tie (target stays 4, accepted_steps == 0).
    # `<=` would accept candidate 5 (accepted_steps > 0).
    # This test pins core.py's `best_obj < current_minibatch_obj` as STRICT.
    space = IntegerEditSpace()
    groups = [IntGroup(key_str="g", members=[4, 6], is_control=False)]
    results = run_gcg(
        edit_space=space,
        groups=groups,
        objective_batch=_objective_batch,
        gradient_provider_batch=_grad_batch,
        cfg=Cfg(max_steps=20),
    )
    payload = results[0].payload
    assert payload["target"] == 4
    assert payload["accepted_steps"] == 0
    assert payload["objective"] == 1.0


class _ReservedKeyEditSpace(IntegerEditSpace):
    """Subclass that returns a reserved key in step_event_fields to test the guard."""

    def step_event_fields(self, validated: Any) -> dict[str, Any]:
        return {"accepted": 1}  # "accepted" is a reserved event key


def test_reserved_event_key_raises_feature_gcg_error():
    # An EditSpace returning a reserved key in step_event_fields must raise FeatureGcgError.
    # This pins the _check_event_fields guard in core.py.
    space = _ReservedKeyEditSpace()
    groups = [IntGroup(key_str="g", members=[4, 6], is_control=False)]
    with pytest.raises(FeatureGcgError, match="accepted"):
        run_gcg(
            edit_space=space,
            groups=groups,
            objective_batch=_objective_batch,
            gradient_provider_batch=_grad_batch,
            cfg=Cfg(),
        )


def test_strict_zip_raises_value_error_on_gradient_count_mismatch():
    # A gradient_provider_batch that returns fewer gradients than the minibatch
    # must cause run_gcg to raise ValueError (from the strict=True zip in core.py).
    # group has 2 members; minibatch size >= 2 so both are selected; provider returns [].
    def _short_grad_batch(items, step):
        return []  # too few: should be len(items), not 0

    space = IntegerEditSpace()
    groups = [IntGroup(key_str="g", members=[4, 6], is_control=False)]
    with pytest.raises(ValueError):
        run_gcg(
            edit_space=space,
            groups=groups,
            objective_batch=_objective_batch,
            gradient_provider_batch=_short_grad_batch,
            cfg=Cfg(max_steps=1, universal_minibatch_size=2),
        )


class _CallbackTraceSpace(IntegerEditSpace):
    def __init__(self, calls: list[tuple[str, Any]]) -> None:
        self.calls = calls

    def validate(self, group: IntGroup):
        self.calls.append(("validate", group.key_str))
        return super().validate(group)

    def is_trivial(self, validated):
        self.calls.append(("is_trivial", validated.key_str))
        return super().is_trivial(validated)

    def finalize_trivial(self, validated):
        self.calls.append(("finalize_trivial", validated.key_str))
        return super().finalize_trivial(validated)

    def init_target(self, validated):
        self.calls.append(("init_target", validated.key_str))
        return super().init_target(validated)

    def step_event_fields(self, validated):
        self.calls.append(("step_event_fields", validated.key_str))
        return super().step_event_fields(validated)

    def rank_candidates(self, agg_grad, current, cfg):
        self.calls.append(("rank_candidates", current))
        return super().rank_candidates(agg_grad, current, cfg)

    def sample_candidates(
        self, proposals, current, *, batch_size, temperature, rng, visited
    ):
        self.calls.append(("sample_candidates", current))
        return super().sample_candidates(
            proposals,
            current,
            batch_size=batch_size,
            temperature=temperature,
            rng=rng,
            visited=visited,
        )

    def finalize(
        self,
        validated,
        target,
        *,
        objective,
        accepted_steps,
        visited_count,
    ):
        self.calls.append(("finalize", validated.key_str))
        return super().finalize(
            validated,
            target,
            objective=objective,
            accepted_steps=accepted_steps,
            visited_count=visited_count,
        )


def test_callback_seam_preserves_order_and_skips_every_restored_search_path() -> None:
    calls: list[tuple[str, Any]] = []
    boundaries: list[str] = []
    restored_calls: list[str] = []
    commits: list[tuple[str, GroupResult]] = []
    events: list[dict[str, Any]] = []
    groups = [
        IntGroup(key_str="control", members=[0], is_control=True),
        IntGroup(key_str="restored-before", members=[100], is_control=False),
        IntGroup(key_str="fresh", members=[10], is_control=False),
        IntGroup(key_str="restored-after", members=[200], is_control=False),
    ]
    restored = {
        "restored-before": GroupResult(payload={"restored": "before"}),
        "restored-after": GroupResult(payload={"restored": "after"}),
    }

    def _restore(validated):
        restored_calls.append(validated.key_str)
        return restored.get(validated.key_str)

    def _objective(items):
        assert all(target not in {100, 200} for _member, target in items)
        calls.append(("objective_batch", tuple(items)))
        return _objective_batch(items)

    def _gradient(items, step):
        assert all(target not in {100, 200} for _member, target in items)
        calls.append(("gradient_provider_batch", step))
        return _grad_batch(items, step)

    results = run_gcg(
        edit_space=_CallbackTraceSpace(calls),
        groups=groups,
        objective_batch=_objective,
        gradient_provider_batch=_gradient,
        cfg=Cfg(max_steps=1),
        progress_callback=events.append,
        group_boundary_callback=lambda: boundaries.append("boundary"),
        restore_group_result=_restore,
        commit_group_result=lambda validated, result: commits.append(
            (validated.key_str, result)
        ),
    )

    assert [result.payload for result in results] == [
        {"control": True},
        {"restored": "before"},
        {
            "target": 10,
            "objective": 0.0,
            "accepted_steps": 0,
            "visited_count": 3,
        },
        {"restored": "after"},
    ]
    assert boundaries == ["boundary"] * len(groups)
    assert restored_calls == ["restored-before", "fresh", "restored-after"]
    assert [(key, result.payload) for key, result in commits] == [
        ("fresh", results[2].payload)
    ]
    assert ("init_target", "restored-before") not in calls
    assert ("init_target", "restored-after") not in calls
    assert ("step_event_fields", "restored-before") not in calls
    assert ("step_event_fields", "restored-after") not in calls
    assert [event["event"] for event in events] == [
        "group_done",
        "group_start",
        "step_done",
        "group_done",
    ]
    assert [event["group_index"] for event in events] == [1, 3, 3, 3]


def test_commit_is_after_finalize_and_before_group_done() -> None:
    calls: list[tuple[str, Any]] = []
    events: list[dict[str, Any]] = []

    def _commit(validated, result):
        assert calls[-1] == ("finalize", validated.key_str)
        assert isinstance(result, GroupResult)
        assert not any(event["event"] == "group_done" for event in events)
        calls.append(("commit", validated.key_str))

    run_gcg(
        edit_space=_CallbackTraceSpace(calls),
        groups=[IntGroup(key_str="fresh", members=[10], is_control=False)],
        objective_batch=_objective_batch,
        gradient_provider_batch=_grad_batch,
        cfg=Cfg(max_steps=0),
        progress_callback=events.append,
        commit_group_result=_commit,
    )
    assert calls[-2:] == [("finalize", "fresh"), ("commit", "fresh")]
    assert [event["event"] for event in events] == ["group_start", "group_done"]


def test_commit_failure_has_no_false_done_and_committed_result_can_restore() -> None:
    durable: dict[str, GroupResult] = {}
    first_events: list[dict[str, Any]] = []

    def _commit_then_crash(validated, result):
        durable[validated.key_str] = result
        raise RuntimeError("simulated crash after durable commit")

    with pytest.raises(RuntimeError, match="after durable commit"):
        run_gcg(
            edit_space=IntegerEditSpace(),
            groups=[IntGroup(key_str="g", members=[10], is_control=False)],
            objective_batch=_objective_batch,
            gradient_provider_batch=_grad_batch,
            cfg=Cfg(max_steps=0),
            progress_callback=first_events.append,
            commit_group_result=_commit_then_crash,
        )
    assert durable.keys() == {"g"}
    assert [event["event"] for event in first_events] == ["group_start"]

    def _tripwire(*_args, **_kwargs):
        pytest.fail("a durably committed group was recomputed")

    second_events: list[dict[str, Any]] = []
    restored = run_gcg(
        edit_space=IntegerEditSpace(),
        groups=[IntGroup(key_str="g", members=[10], is_control=False)],
        objective_batch=_tripwire,
        gradient_provider_batch=_tripwire,
        cfg=Cfg(max_steps=10),
        progress_callback=second_events.append,
        restore_group_result=lambda validated: durable.get(validated.key_str),
        commit_group_result=_tripwire,
    )
    assert restored == [durable["g"]]
    assert second_events == []


def test_trivial_groups_never_touch_restore_or_commit_callbacks() -> None:
    def _tripwire(*_args, **_kwargs):
        pytest.fail("control group touched a non-control checkpoint callback")

    events: list[dict[str, Any]] = []
    results = run_gcg(
        edit_space=IntegerEditSpace(),
        groups=[IntGroup(key_str="control", members=[1], is_control=True)],
        objective_batch=_tripwire,
        gradient_provider_batch=_tripwire,
        cfg=Cfg(),
        progress_callback=events.append,
        restore_group_result=_tripwire,
        commit_group_result=_tripwire,
    )
    assert results[0].payload == {"control": True}
    assert events == [
        {
            "event": "group_done",
            "group_index": 1,
            "total_groups": 1,
            "objective_value": None,
            "accepted_steps": 0,
            "visited_sequences": 0,
            "control": True,
            "key_str": "control",
            "n_members": 1,
        }
    ]


def test_explicit_none_callbacks_preserve_results_and_full_event_oracle() -> None:
    groups = [
        IntGroup(key_str="control", members=[1], is_control=True),
        IntGroup(key_str="fresh", members=[4, 10, 16], is_control=False),
    ]
    baseline_events: list[dict[str, Any]] = []
    explicit_events: list[dict[str, Any]] = []
    baseline = run_gcg(
        edit_space=IntegerEditSpace(),
        groups=groups,
        objective_batch=_objective_batch,
        gradient_provider_batch=_grad_batch,
        cfg=Cfg(max_steps=3),
        progress_callback=baseline_events.append,
    )
    explicit = run_gcg(
        edit_space=IntegerEditSpace(),
        groups=groups,
        objective_batch=_objective_batch,
        gradient_provider_batch=_grad_batch,
        cfg=Cfg(max_steps=3),
        progress_callback=explicit_events.append,
        restore_group_result=None,
        commit_group_result=None,
    )
    assert explicit == baseline
    assert explicit_events == baseline_events
