"""Real-GEPA-0.1.1 functional tests over deterministic fake runtimes."""

from __future__ import annotations

import importlib
import importlib.metadata
import tomllib
from pathlib import Path
from typing import Any

import pytest
from gepa.core.result import GEPAResult

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import (
    CallAccounting,
    OptimizationExample,
    RawEvaluation,
    optimize_suffix,
)


class CachedRuntime:
    def __init__(self) -> None:
        self.seen: set[tuple[str, int]] = set()
        self.calls: list[tuple[str, int]] = []

    def __call__(self, suffix: str, input_id: int) -> RawEvaluation:
        key = (suffix, input_id)
        paid = 0 if key in self.seen else 2
        self.seen.add(key)
        self.calls.append(key)
        concept = {"seed": 2.0, "better": 1.0, "best": 0.0}.get(suffix, 2.0)
        return RawEvaluation(
            generation=f"A useful substantive answer for {input_id}.",
            concept_rating=concept,
            fluency_rating=2.0,
            judge_rationale=f"concept={concept}",
            paid_judge_cache_misses=paid,
        )


class SequenceReflection:
    def __init__(self, *suffixes: Any) -> None:
        self.suffixes = list(suffixes)
        self.prompts: list[object] = []
        self._paid = 0

    @property
    def paid_cache_misses(self) -> int:
        return self._paid

    def __call__(self, prompt: object) -> Any:
        self.prompts.append(prompt)
        self._paid += 1
        value = self.suffixes.pop(0)
        if isinstance(value, Exception):
            raise value
        if value == "RAW_BLANK":
            return ""
        if value == "RAW_NONSTRING":
            return 7
        if value == "RAW_FENCES":
            return "``````"
        return f"```{value}```"


def _examples() -> list[OptimizationExample]:
    return [OptimizationExample(i, f"request {i}") for i in range(4)]


def _run(
    tmp_path: Path,
    *,
    name: str,
    proposals: int,
    reflection_values: tuple[Any, ...],
) -> tuple[Any, CachedRuntime, SequenceReflection]:
    runtime = CachedRuntime()
    reflection = SequenceReflection(*reflection_values)
    result = optimize_suffix(
        examples=_examples(),
        runtime=runtime,
        reflection_lm=reflection,
        objective="Avoid the target concept while remaining responsive.",
        run_dir=tmp_path / name,
        proposal_count=proposals,
        gepa_seed=17,
        seed_suffix="seed",
        reflection_minibatch_size=3,
    )
    return result, runtime, reflection


def test_real_gepa_improves_candidate_and_accounts_cache_hits(tmp_path: Path) -> None:
    assert importlib.metadata.version("gepa") == "0.1.1"
    result, runtime, reflection = _run(
        tmp_path, name="improvement", proposals=1, reflection_values=("better",)
    )
    assert result.suffix == "better"
    assert result.best_fitness == 0.5
    assert [value.raw.generation for value in result.evaluations] == [
        f"A useful substantive answer for {i}." for i in range(4)
    ]
    # GEPA nominal: seed full=4, parent minibatch=3, proposal minibatch=3,
    # accepted proposal full=4. Its per-run cache evaluates only the eight
    # distinct (suffix, input) pairs, and root retains the selected full eval.
    assert result.accounting == CallAccounting(14, 8, 8, 16, 1)
    assert len(runtime.calls) == 8
    assert len(set(runtime.calls)) == len(runtime.calls)
    assert len(reflection.prompts) == 1


def test_gepa_dependency_is_exact_in_project_and_lock() -> None:
    root = Path(__file__).resolve().parents[3]
    project = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    dependencies = project["project"]["dependencies"]
    assert [value for value in dependencies if value.startswith("gepa")] == ["gepa==0.1.1"]

    lock = tomllib.loads((root / "uv.lock").read_text(encoding="utf-8"))
    package = next(value for value in lock["package"] if value["name"] == "nano-gpt-pretrain-steer")
    requirements = package["metadata"]["requires-dist"]
    assert [value for value in requirements if value["name"] == "gepa"] == [
        {"name": "gepa", "specifier": "==0.1.1"}
    ]


def test_real_gepa_tie_keeps_seed_and_first_max_is_pinned(tmp_path: Path) -> None:
    result, _runtime, _reflection = _run(
        tmp_path, name="tie", proposals=1, reflection_values=("unknown",)
    )
    assert result.suffix == "seed"
    assert result.best_fitness == 0.0
    # Directly pin GEPAResult's documented first-maximum behavior as well as
    # the real engine's strict-improvement rejection above.
    tied = GEPAResult(
        candidates=[{"suffix": "first"}, {"suffix": "second"}],
        parents=[[None], [0]],
        val_aggregate_scores=[0.5, 0.5],
        val_subscores=[{0: 0.5}, {0: 0.5}],
        per_val_instance_best_candidates={0: {0, 1}},
        discovery_eval_counts=[0, 1],
    )
    assert tied.best_idx == 0
    assert tied.best_candidate == {"suffix": "first"}


def test_proposal_levels_are_independent_and_deterministic(tmp_path: Path) -> None:
    one, runtime_one, _ = _run(
        tmp_path, name="level1", proposals=1, reflection_values=("better",)
    )
    two, runtime_two, _ = _run(
        tmp_path, name="level2", proposals=2, reflection_values=("better", "best")
    )
    two_again, runtime_again, _ = _run(
        tmp_path, name="level2-repeat", proposals=2, reflection_values=("better", "best")
    )
    assert one.suffix == "better"
    assert two.suffix == "best"
    assert two == two_again
    assert runtime_two.calls == runtime_again.calls
    assert runtime_one.calls != runtime_two.calls
    assert one.accounting.gepa_nominal_metric_calls == 14
    assert two.accounting.gepa_nominal_metric_calls == 24
    assert one.accounting.actual_generation_calls == 8
    assert two.accounting.actual_generation_calls == 12
    assert len(runtime_one.calls) == len(set(runtime_one.calls))
    assert len(runtime_two.calls) == len(set(runtime_two.calls))


@pytest.mark.parametrize(
    "bad",
    (RuntimeError("provider failed"), "RAW_BLANK", "RAW_NONSTRING", "RAW_FENCES"),
)
def test_reflection_failures_stop_after_one_attempt_and_raise(
    bad: object, tmp_path: Path
) -> None:
    runtime = CachedRuntime()
    reflection = SequenceReflection(bad, "must-not-be-used")
    with pytest.raises(AxBenchAdvError, match="GEPA reflection failed") as exc_info:
        optimize_suffix(
            examples=_examples(),
            runtime=runtime,
            reflection_lm=reflection,
            objective="Improve.",
            run_dir=tmp_path / f"bad-{type(bad).__name__}",
            proposal_count=2,
            gepa_seed=17,
            seed_suffix="seed",
        )
    assert len(reflection.prompts) == 1
    assert exc_info.value.__cause__ is not None


def test_litellm_string_path_is_unreachable(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    optimize_module = importlib.import_module("gepa.optimize_anything")
    monkeypatch.setattr(
        optimize_module,
        "make_litellm_lm",
        lambda _model: pytest.fail("LiteLLM path must be unreachable"),
    )
    result, _runtime, _reflection = _run(
        tmp_path, name="no-litellm", proposals=1, reflection_values=("better",)
    )
    assert result.suffix == "better"
