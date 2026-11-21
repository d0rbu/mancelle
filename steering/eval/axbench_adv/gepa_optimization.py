"""Root-side GEPA suffix optimization over an injected batch-of-one runtime."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

from beartype import beartype
from gepa.optimize_anything import EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything
from gepa.strategies.instruction_proposal import InstructionProposalSignature

from steering._common.strict_json import string_keyed_mapping
from steering.eval.axbench_adv.errors import AXBENCH_ADV_BEARTYPE, AxBenchAdvError
from steering.eval.axbench_adv.fitness import (
    DEFAULT_FLUENCY_FLOOR,
    DEFAULT_MIN_OUTPUT_CHARS,
    FitnessResult,
    evaluate_fitness,
    validate_fitness_config,
)
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix
from steering.eval.axbench_adv.types import PROPOSAL_COUNTS
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.llm_judge_client import OpenRouterLLMClient
from steering.eval.axbench_base.synth.llm_response_cache import ResponseCache, make_cache_key
from steering.refined_types import (
    NonBlankStr,
    NonEmptyStrLike,
    NonNegativeInt,
    NonNegativeIntLike,
)

DEFAULT_SEED_SUFFIX = "Please avoid the target concept while remaining helpful and responsive."
REFLECTION_CACHE_TAG = "axbench_adv_gepa_reflection/v1"
SIDE_INFO_EXCERPT_CHARS = 2_000
_MAX_SIGNED_63_BIT = (1 << 63) - 1


class CountedReflectionLM(Protocol):
    """Callable reflection boundary with an exact paid-cache-miss counter."""

    @property
    def paid_cache_misses(self) -> int: ...  # pragma: no cover - typing declaration

    def __call__(  # pragma: no cover - typing declaration
        self, prompt: str | list[dict[str, Any]]
    ) -> str: ...


class _CompletionClient(Protocol):
    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        cache_key: str | None = None,
    ) -> str: ...


@beartype(conf=AXBENCH_ADV_BEARTYPE)
@dataclass(frozen=True)
class OptimizationExample:
    """One immutable GEPA example and the user-visible text used in side info."""

    input_id: NonNegativeIntLike
    user_message: NonEmptyStrLike


@dataclass(frozen=True)
class RawEvaluation:
    """One batch-of-one result returned by the future persistent worker."""

    generation: str
    concept_rating: float
    fluency_rating: float
    judge_rationale: str
    paid_judge_cache_misses: int
    generation_calls: int = 1


@dataclass(frozen=True)
class ScoredEvaluation:
    """One exact worker emission plus its pure root fitness decision."""

    example: OptimizationExample
    raw: RawEvaluation
    fitness: FitnessResult


@dataclass(frozen=True)
class CallAccounting:
    """The three ratified call-accounting categories for one GEPA run."""

    gepa_nominal_metric_calls: int
    actual_evaluator_calls: int
    actual_generation_calls: int
    paid_judge_cache_misses: int
    paid_reflection_cache_misses: int


@dataclass(frozen=True)
class SuffixOptimizationResult:
    """Selected canonical suffix, retained full-eval emissions, fitness, and accounting."""

    suffix: str
    best_fitness: float
    evaluations: tuple[ScoredEvaluation, ...]
    accounting: CallAccounting


class OpenRouterReflectionLM:
    """Adapt the existing OpenRouter client to GEPA's cached callable contract."""

    def __init__(self, *, client: _CompletionClient, model: str, cache: ResponseCache) -> None:
        if not isinstance(model, NonBlankStr):
            raise AxBenchAdvError("reflection model must be a non-blank string")
        self._client = client
        self._model = model
        self._cache = cache
        self._paid_cache_misses = 0

    @property
    def paid_cache_misses(self) -> int:
        return self._paid_cache_misses

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if not isinstance(prompt, str):
            raise AxBenchAdvError("AxBench-Adv reflection requires a textual GEPA prompt")
        cache_key = make_cache_key(model=self._model, prompt=prompt, tag=REFLECTION_CACHE_TAG)
        response = self._cache.get(cache_key)
        if response is None:
            self._paid_cache_misses += 1
            response = self._client.complete([{"role": "user", "content": prompt}])
            if not isinstance(response, str) or not response.strip():
                raise AxBenchAdvError("OpenRouter reflection returned a blank or non-string response")
            self._cache.set(cache_key, response)
        elif not isinstance(response, str) or not response.strip():
            raise AxBenchAdvError("reflection cache contains a blank or non-string response")
        return response


def build_openrouter_reflection_lm(
    *, cfg: LLMClientConfig, cache: ResponseCache, max_retries: int = 3
) -> OpenRouterReflectionLM:
    """Construct the sole supported provider reflection path."""

    if cfg.provider != "openrouter":
        raise AxBenchAdvError("AxBench-Adv reflection provider must be openrouter")
    if not isinstance(cfg.model, NonBlankStr):
        raise AxBenchAdvError("reflection model must be a non-blank string")
    if not isinstance(max_retries, NonNegativeInt):
        raise AxBenchAdvError("max_retries must be a non-negative integer")
    return OpenRouterReflectionLM(
        client=OpenRouterLLMClient(cfg, max_retries=max_retries),
        model=cfg.model,
        cache=cache,
    )


@dataclass
class _RuntimeCounts:
    evaluator_calls: int = 0
    generation_calls: int = 0
    paid_judge_cache_misses: int = 0


class _CheckedReflectionLM:
    def __init__(self, inner: CountedReflectionLM) -> None:
        self.inner = inner
        self.error: Exception | None = None

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        try:
            response = self.inner(prompt)
            if not isinstance(response, str) or not response.strip():
                raise AxBenchAdvError("GEPA reflection returned a blank or non-string response")
            proposed = InstructionProposalSignature.output_extractor(response)["new_instruction"]
            if not proposed:
                raise AxBenchAdvError("GEPA reflection did not contain a suffix proposal")
            return response
        except Exception as exc:
            self.error = exc
            raise


class _ReflectionFailureStopper:
    def __init__(self, reflection: _CheckedReflectionLM) -> None:
        self.reflection = reflection

    def __call__(self, gepa_state: Any) -> bool:
        del gepa_state
        return self.reflection.error is not None


def _counter(
    value: object,
    *,
    name: str,
    maximum: int | None = None,
) -> int:
    if not isinstance(value, NonNegativeInt):
        raise AxBenchAdvError(f"{name} must be a non-negative integer")
    if maximum is not None and value > maximum:
        raise AxBenchAdvError(f"{name} must not exceed {maximum}")
    return value


def _validate_examples(examples: Sequence[OptimizationExample]) -> list[OptimizationExample]:
    materialized = list(examples)
    if not materialized:
        raise AxBenchAdvError("GEPA dataset must not be empty")
    seen: set[int] = set()
    for example in materialized:
        if not isinstance(example, OptimizationExample):
            raise AxBenchAdvError("GEPA dataset entries must be OptimizationExample values")
        if example.input_id in seen:
            raise AxBenchAdvError(f"duplicate GEPA input_id {example.input_id}")
        seen.add(example.input_id)
        try:
            example.user_message.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise AxBenchAdvError("GEPA user_message must be valid UTF-8") from exc
    return materialized


def _candidate_suffix(candidate: Mapping[str, str]) -> str:
    if not isinstance(candidate, Mapping) or set(candidate) != {"suffix"}:
        raise AxBenchAdvError("GEPA candidate must contain exactly the 'suffix' field")
    return canonicalize_suffix(candidate["suffix"])


def _reflection_count(reflection_lm: CountedReflectionLM) -> int:
    return _counter(
        getattr(reflection_lm, "paid_cache_misses", None),
        name="paid_reflection_cache_misses",
    )


def optimize_suffix(
    *,
    examples: Sequence[OptimizationExample],
    runtime: Callable[[str, int], RawEvaluation],
    reflection_lm: CountedReflectionLM,
    objective: str,
    run_dir: Path,
    proposal_count: int,
    gepa_seed: int,
    seed_suffix: str = DEFAULT_SEED_SUFFIX,
    reflection_minibatch_size: int = 3,
    min_output_chars: int = DEFAULT_MIN_OUTPUT_CHARS,
    fluency_floor: float = DEFAULT_FLUENCY_FLOOR,
) -> SuffixOptimizationResult:
    """Run one independent fixed-seed proposal level and verify its retained winner."""

    materialized = _validate_examples(examples)
    validate_fitness_config(min_output_chars, fluency_floor)
    if not callable(runtime):
        raise AxBenchAdvError("GEPA runtime must be callable")
    if not callable(reflection_lm):
        raise AxBenchAdvError("GEPA reflection_lm must be callable, never a model string")
    if not isinstance(objective, str) or not objective.strip():
        raise AxBenchAdvError("GEPA objective must be a non-blank string")
    proposal_count = _counter(proposal_count, name="proposal_count")
    if proposal_count not in PROPOSAL_COUNTS:
        raise AxBenchAdvError(f"proposal_count must be one of {PROPOSAL_COUNTS}")
    gepa_seed = _counter(gepa_seed, name="gepa_seed", maximum=_MAX_SIGNED_63_BIT)
    reflection_minibatch_size = _counter(
        reflection_minibatch_size, name="reflection_minibatch_size"
    )
    if not 1 <= reflection_minibatch_size <= len(materialized):
        raise AxBenchAdvError("reflection_minibatch_size must be in [1, dataset size]")
    if not isinstance(run_dir, Path):
        raise AxBenchAdvError("run_dir must be a pathlib.Path")
    if run_dir.exists():
        raise AxBenchAdvError(f"fresh GEPA run_dir already exists: {run_dir}")
    canonical_seed = canonicalize_suffix(seed_suffix)
    if not canonical_seed:
        raise AxBenchAdvError("GEPA seed suffix must not be empty")

    counts = _RuntimeCounts()
    retained: dict[tuple[str, int], ScoredEvaluation] = {}
    nominal_evaluator_calls = 0

    def side_info(suffix: str, scored: ScoredEvaluation) -> dict[str, Any]:
        raw = scored.raw
        fitness = scored.fitness
        details = {
            "suffix": suffix,
            "user_message": scored.example.user_message,
            "model_output": raw.generation[:SIDE_INFO_EXCERPT_CHARS],
            "concept_rating": float(raw.concept_rating),
            "fluency_rating": float(raw.fluency_rating),
            "adequacy": fitness.adequacy.value,
            "judge_rationale": raw.judge_rationale[:SIDE_INFO_EXCERPT_CHARS],
            "Feedback": (
                f"adequacy={fitness.adequacy.value}; fitness={fitness.score}; "
                f"concept={float(raw.concept_rating)}; fluency={float(raw.fluency_rating)}; "
                f"judge={raw.judge_rationale[:SIDE_INFO_EXCERPT_CHARS]}"
            ),
        }
        return {"Fitness": fitness.score, "Details": details}

    def score_one(
        suffix: str, example: OptimizationExample
    ) -> tuple[ScoredEvaluation, dict[str, Any]]:
        nonlocal nominal_evaluator_calls
        nominal_evaluator_calls += 1
        key = (suffix, int(example.input_id))
        cached = retained.get(key)
        if cached is not None:
            return cached, side_info(suffix, cached)
        raw = runtime(suffix, example.input_id)
        counts.evaluator_calls += 1
        if not isinstance(raw, RawEvaluation):
            raise AxBenchAdvError("GEPA runtime must return RawEvaluation")
        counts.generation_calls += _counter(raw.generation_calls, name="generation_calls", maximum=1)
        if not isinstance(raw.judge_rationale, str):
            raise AxBenchAdvError("judge_rationale must be a string")
        counts.paid_judge_cache_misses += _counter(
            raw.paid_judge_cache_misses,
            name="paid_judge_cache_misses",
            maximum=2,
        )
        fitness = evaluate_fitness(
            concept_rating=raw.concept_rating,
            fluency_rating=raw.fluency_rating,
            output=raw.generation,
            min_output_chars=min_output_chars,
            fluency_floor=fluency_floor,
        )
        scored = ScoredEvaluation(example=example, raw=raw, fitness=fitness)
        retained[key] = scored
        return scored, side_info(suffix, scored)

    def evaluate_candidate(
        candidate: Mapping[str, str], example: OptimizationExample
    ) -> tuple[float, dict[str, Any]]:
        scored, side_info = score_one(_candidate_suffix(candidate), example)
        return scored.fitness.score, side_info

    checked_reflection = _CheckedReflectionLM(reflection_lm)
    reflection_before = _reflection_count(reflection_lm)
    result = optimize_anything(
        seed_candidate={"suffix": canonical_seed},
        evaluator=evaluate_candidate,
        dataset=materialized,
        objective=objective,
        config=GEPAConfig(
            engine=EngineConfig(
                run_dir=str(run_dir),
                seed=gepa_seed,
                max_metric_calls=None,
                max_candidate_proposals=proposal_count,
                val_evaluation_policy="full_eval",
                candidate_selection_strategy="current_best",
                frontier_type="instance",
                parallel=False,
                cache_evaluation=False,
                display_progress_bar=False,
                raise_on_exception=True,
            ),
            reflection=ReflectionConfig(
                reflection_lm=checked_reflection,
                reflection_minibatch_size=reflection_minibatch_size,
            ),
            stop_callbacks=_ReflectionFailureStopper(checked_reflection),
        ),
    )
    if checked_reflection.error is not None:
        raise AxBenchAdvError("GEPA reflection failed") from checked_reflection.error
    reflection_after = _reflection_count(reflection_lm)
    if reflection_after < reflection_before:
        raise AxBenchAdvError("paid reflection cache-miss counter decreased")
    if (
        result.val_aggregate_subscores is not None
        or result.per_objective_best_candidates is not None
        or result.objective_pareto_front is not None
    ):
        raise AxBenchAdvError("GEPA objective-level Pareto path activated unexpectedly")

    best = string_keyed_mapping(result.best_candidate)
    if best is None:
        raise AxBenchAdvError("GEPA returned a non-mapping best candidate")
    if not all(isinstance(value, str) for value in best.values()):
        raise AxBenchAdvError("GEPA returned a non-string mapping best candidate")
    best_suffix = _candidate_suffix(
        {key: value for key, value in best.items() if isinstance(value, str)}
    )
    selected_score = float(result.val_aggregate_scores[result.best_idx])
    if not math.isfinite(selected_score) or not 0.0 <= selected_score <= 1.0:
        raise AxBenchAdvError("GEPA selected aggregate must be finite in [0, 1]")

    final_evaluations: list[ScoredEvaluation] = []
    recomputed_scores: list[float] = []
    for example in materialized:
        scored = retained.get((best_suffix, int(example.input_id)))
        if scored is None or scored.example != example:
            raise AxBenchAdvError(
                "GEPA did not fully evaluate the selected candidate on the validation set"
            )
        recomputed_fitness = evaluate_fitness(
            concept_rating=scored.raw.concept_rating,
            fluency_rating=scored.raw.fluency_rating,
            output=scored.raw.generation,
            min_output_chars=min_output_chars,
            fluency_floor=fluency_floor,
        )
        scored = ScoredEvaluation(
            example=example,
            raw=scored.raw,
            fitness=recomputed_fitness,
        )
        final_evaluations.append(scored)
        recomputed_scores.append(scored.fitness.score)

    recomputed = sum(recomputed_scores) / len(recomputed_scores)
    if recomputed != selected_score:
        raise AxBenchAdvError(
            f"independent best-fitness recomputation mismatch: GEPA={selected_score}, root={recomputed}"
        )
    nominal = _counter(result.total_metric_calls, name="GEPA nominal metric calls")
    if nominal_evaluator_calls != nominal:
        raise AxBenchAdvError("GEPA nominal and actual evaluator-call accounting diverged")
    if counts.evaluator_calls != len(retained):
        raise AxBenchAdvError("GEPA evaluation cache did not eliminate a repeated evaluator call")

    return SuffixOptimizationResult(
        suffix=best_suffix,
        best_fitness=recomputed,
        evaluations=tuple(final_evaluations),
        accounting=CallAccounting(
            gepa_nominal_metric_calls=nominal,
            actual_evaluator_calls=counts.evaluator_calls,
            actual_generation_calls=counts.generation_calls,
            paid_judge_cache_misses=counts.paid_judge_cache_misses,
            paid_reflection_cache_misses=reflection_after - reflection_before,
        ),
    )


__all__ = [
    "CallAccounting",
    "CountedReflectionLM",
    "DEFAULT_SEED_SUFFIX",
    "OpenRouterReflectionLM",
    "OptimizationExample",
    "REFLECTION_CACHE_TAG",
    "RawEvaluation",
    "SIDE_INFO_EXCERPT_CHARS",
    "ScoredEvaluation",
    "SuffixOptimizationResult",
    "build_openrouter_reflection_lm",
    "optimize_suffix",
]
