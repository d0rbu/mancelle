from __future__ import annotations

import random
from collections.abc import Callable, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass
from typing import Any

import torch as t

from steering.conditional.feature_gcg.aggregation import minibatch_for_step
from steering.conditional.feature_gcg.candidate_search import deterministic_step_rng
from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.edit_space import GroupResult, ValidatedGroup
from steering.refined_types import NonNegativeInt, PositiveInt


@dataclass
class _Cfg:
    max_steps: int = 3
    candidate_batch_size: int = 1
    temperature: float = 1.0
    master_seed: int = 0
    universal_minibatch_size: int = 16


@dataclass
class _Group:
    key_str: str
    members: list[int]
    initial_target: int = 10
    control: bool = False


class _InstrumentedSpace:
    def __init__(self, sample_outputs: Sequence[Sequence[int]]) -> None:
        self._sample_outputs = [list(values) for values in sample_outputs]
        self.rank_targets: list[int] = []
        self.proposals: list[object] = []
        self.sample_proposal_ids: list[int] = []
        self.sample_rng_values: list[float] = []
        self.sample_targets: list[int] = []
        self.sample_calls = 0

    def validate(self, group: _Group) -> ValidatedGroup:
        return ValidatedGroup(
            key_str=group.key_str,
            members=list(group.members),
            payload=group,
        )

    def is_trivial(self, validated: ValidatedGroup) -> bool:
        return bool(validated.payload.control)

    def finalize_trivial(self, validated: ValidatedGroup) -> GroupResult:
        return GroupResult(payload={"control": True})

    def init_target(self, validated: ValidatedGroup) -> int:
        return int(validated.payload.initial_target)

    def digest(self, target: int) -> str:
        return str(int(target))

    def aggregate_gradient(self, per_member: Any) -> None:
        list(per_member)

    def rank_candidates(self, agg_grad: Any, current: int, cfg: Any) -> object:
        del agg_grad, cfg
        proposal = object()
        self.rank_targets.append(current)
        self.proposals.append(proposal)
        return proposal

    def sample_candidates(
        self,
        proposals: object,
        current: int,
        *,
        batch_size: int,
        temperature: float,
        rng: random.Random,
        visited: set[str],
    ) -> list[int]:
        del batch_size, temperature
        self.sample_proposal_ids.append(id(proposals))
        self.sample_rng_values.append(rng.random())
        self.sample_targets.append(current)
        output = (
            self._sample_outputs[self.sample_calls]
            if self.sample_calls < len(self._sample_outputs)
            else []
        )
        self.sample_calls += 1
        for candidate in output:
            visited.add(self.digest(candidate))
        return list(output)

    def model_item(self, member: int, target: int) -> tuple[int, int]:
        return member, target

    def group_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        return {"key_str": validated.key_str}

    def step_event_fields(self, validated: ValidatedGroup) -> dict[str, Any]:
        return {"key_str": validated.key_str}

    def finalize(
        self,
        validated: ValidatedGroup,
        target: int,
        *,
        objective: float,
        accepted_steps: int,
        visited_count: int,
    ) -> GroupResult:
        return GroupResult(
            payload={
                "key": validated.key_str,
                "target": target,
                "objective": objective,
                "accepted_steps": accepted_steps,
                "visited_count": visited_count,
            }
        )


def _constant_objective(items: Sequence[tuple[int, int]]) -> list[float]:
    return [0.0 for _item in items]


def _target_objective(items: Sequence[tuple[int, int]]) -> list[float]:
    return [float(target) for _member, target in items]


def _run(
    *,
    space: _InstrumentedSpace,
    groups: Sequence[_Group],
    cfg: _Cfg,
    objective: Callable[[Sequence[tuple[int, int]]], list[float]] = _constant_objective,
    exhaustion: Callable[[object, int, AbstractSet[str]], bool] | None,
    events: list[dict[str, Any]] | None = None,
) -> tuple[list[GroupResult], list[list[tuple[int, int]]]]:
    gradient_items: list[list[tuple[int, int]]] = []

    def gradient_provider(
        items: Sequence[tuple[int, int]], _step: int
    ) -> list[t.Tensor]:
        gradient_items.append(list(items))
        return [t.zeros(1) for _item in items]

    results = run_gcg(
        edit_space=space,
        groups=groups,
        objective_batch=objective,
        gradient_provider_batch=gradient_provider,
        cfg=cfg,
        progress_callback=None if events is None else events.append,
        full_batch_pool_exhausted=exhaustion,
    )
    return results, gradient_items


def test_full_batch_reuses_pool_but_keeps_step_rng_and_batch_schedule() -> None:
    space = _InstrumentedSpace([[101], [102], [103]])
    objective_calls: list[list[tuple[int, int]]] = []

    def objective(items: Sequence[tuple[int, int]]) -> list[float]:
        objective_calls.append(list(items))
        return _constant_objective(items)

    _results, gradients = _run(
        space=space,
        groups=[_Group("stable", [1, 2, 3])],
        cfg=_Cfg(max_steps=3, universal_minibatch_size=3),
        objective=objective,
        exhaustion=lambda _p, _t, _v: False,
    )

    assert len(gradients) == 1
    assert space.rank_targets == [10]
    assert len(set(space.sample_proposal_ids)) == 1
    assert space.sample_rng_values == [
        deterministic_step_rng(
            candidate_seed="stable",
            step=step,
            master_seed=0,
        ).random()
        for step in range(3)
    ]
    candidate_eval_calls = [
        call for call in objective_calls if call and call[0][1] >= 100
    ]
    expected_batches = []
    for step in range(3):
        idx = minibatch_for_step(
            NonNegativeInt.parse(3),
            step=NonNegativeInt.parse(step),
            minibatch_size=PositiveInt.parse(3),
            group_seed="stable|seed0",
        )
        expected_batches.append([[1, 2, 3][i] for i in idx])
    assert [[member for member, _target in call] for call in candidate_eval_calls] == (
        expected_batches
    )


def test_acceptance_invalidates_only_the_old_target_pool() -> None:
    space = _InstrumentedSpace([[9], [10], [8]])
    (result,), gradients = _run(
        space=space,
        groups=[_Group("accept", [1, 2])],
        cfg=_Cfg(max_steps=3, universal_minibatch_size=2),
        objective=_target_objective,
        exhaustion=lambda _p, _t, _v: False,
    )

    assert space.rank_targets == [10, 9]
    assert len(gradients) == 2
    assert space.sample_proposal_ids[0] != space.sample_proposal_ids[1]
    assert space.sample_proposal_ids[1] == space.sample_proposal_ids[2]
    assert result.payload["target"] == 8
    assert result.payload["objective"] == 8.0
    assert result.payload["accepted_steps"] == 2


def test_unlucky_empty_sample_continues_with_cached_pool() -> None:
    space = _InstrumentedSpace([[], [11]])
    callback_calls = 0

    def not_exhausted(
        _proposals: object, _target: int, _visited: AbstractSet[str]
    ) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return False

    _results, gradients = _run(
        space=space,
        groups=[_Group("unlucky", [1, 2])],
        cfg=_Cfg(max_steps=2, universal_minibatch_size=2),
        exhaustion=not_exhausted,
    )

    assert callback_calls == 1
    assert len(gradients) == 1
    assert space.sample_calls == 2
    assert len(set(space.sample_proposal_ids)) == 1


def test_exact_exhaustion_emits_terminal_empty_step_before_group_done() -> None:
    space = _InstrumentedSpace([[], [999]])
    events: list[dict[str, Any]] = []

    _results, gradients = _run(
        space=space,
        groups=[_Group("exhausted", [1, 2])],
        cfg=_Cfg(max_steps=5, universal_minibatch_size=2),
        exhaustion=lambda _p, _t, _v: True,
        events=events,
    )

    assert len(gradients) == 1
    assert len(space.rank_targets) == 1
    assert space.sample_calls == 1
    assert [event["event"] for event in events] == [
        "group_start",
        "step_done",
        "group_done",
    ]
    assert events[1]["candidate_count"] == 0
    assert events[1]["accepted"] is False


def test_partial_minibatch_never_reuses_or_consults_exhaustion() -> None:
    space = _InstrumentedSpace([[], [], [], []])

    def forbidden(
        _proposals: object, _target: int, _visited: AbstractSet[str]
    ) -> bool:
        raise AssertionError("partial-minibatch group consulted exhaustion callback")

    _results, gradients = _run(
        space=space,
        groups=[_Group("partial", [1, 2, 3])],
        cfg=_Cfg(max_steps=4, universal_minibatch_size=2),
        exhaustion=forbidden,
    )

    assert len(gradients) == 4
    assert len(space.rank_targets) == 4
    assert space.sample_calls == 4
    expected = []
    for step in range(4):
        idx = minibatch_for_step(
            NonNegativeInt.parse(3),
            step=NonNegativeInt.parse(step),
            minibatch_size=PositiveInt.parse(2),
            group_seed="partial|seed0",
        )
        expected.append([(member, 10) for member in ([1, 2, 3][i] for i in idx)])
    assert gradients == expected


def test_proposal_cache_is_group_local() -> None:
    space = _InstrumentedSpace([[], []])
    callback_calls = 0

    def not_exhausted(
        _proposals: object, _target: int, _visited: AbstractSet[str]
    ) -> bool:
        nonlocal callback_calls
        callback_calls += 1
        return False

    _results, gradients = _run(
        space=space,
        groups=[_Group("first", [1]), _Group("second", [2])],
        cfg=_Cfg(max_steps=1),
        exhaustion=not_exhausted,
    )

    assert callback_calls == 2
    assert len(gradients) == 2
    assert len(space.proposals) == 2
    assert space.sample_proposal_ids[0] != space.sample_proposal_ids[1]


def test_control_and_zero_step_never_touch_optimizer_or_exhaustion() -> None:
    space = _InstrumentedSpace([])

    def forbidden(
        _proposals: object, _target: int, _visited: AbstractSet[str]
    ) -> bool:
        raise AssertionError("zero-work group consulted exhaustion callback")

    results, gradients = _run(
        space=space,
        groups=[_Group("control", [1], control=True), _Group("zero", [2])],
        cfg=_Cfg(max_steps=0),
        exhaustion=forbidden,
    )

    assert len(results) == 2
    assert gradients == []
    assert space.rank_targets == []
    assert space.sample_calls == 0


def test_exhaustion_callback_exception_propagates() -> None:
    space = _InstrumentedSpace([[]])

    def fail(_proposals: object, _target: int, _visited: AbstractSet[str]) -> bool:
        raise RuntimeError("exact exhaustion failed")

    try:
        _run(
            space=space,
            groups=[_Group("error", [1])],
            cfg=_Cfg(max_steps=2),
            exhaustion=fail,
        )
    except RuntimeError as exc:
        assert str(exc) == "exact exhaustion failed"
    else:
        raise AssertionError("exhaustion callback exception was swallowed")
