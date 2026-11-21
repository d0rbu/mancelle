"""SAE Feature Steering Evaluation Pipeline.

Orchestrates generation, LLM-as-a-judge pairwise comparison, dose-response
analysis, and aggregate reporting for evaluating SAE feature conditioning.

Specificity disclaimer
----------------------
This evaluator's primary metric, ``standard_win_rate_at_middle_magnitude``
(historically ``standard_win_rate``), is the pairwise-judge win rate at the
middle magnitude index of ``mag_levels`` — it is **not** a specificity
estimator. The canonical specificity metric in this codebase is defined in
``steering.eval.inline_steering_eval`` as
``real_probe_score − random_doc_random_seg_probe_score`` and is the ONLY
metric that emits keys containing the substring ``specificity``. A high
``standard_win_rate_at_middle_magnitude`` and a high canonical specificity
are not directly comparable — they answer different questions. Downstream
consumers should migrate from the historical key name to the explicit
middle-magnitude win-rate key.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol, runtime_checkable

import matplotlib
import numpy as np
import requests
import torch as t

matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import yaml
from loguru import logger
from scipy.stats import spearmanr as _scipy_spearmanr

from errors import DataLoadError, ValidationError
from nanochat.gpt import GPT
from steering.conditional.encoder_decoder.eval_adapter import ContextProvider
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.distractor_sampler import DistractorSampler
from steering.eval.harmfulness_judge import JudgeClient
from steering.eval.inference_engine import build_engine_for_features
from steering.eval.target_conditioning import TargetConditioningMode

# ---------------------------------------------------------------------------
# Retry configuration for PairwiseJudge.judge() (matches JudgeClient.score_pair)
# ---------------------------------------------------------------------------

_PAIRWISE_JUDGE_MAX_RETRIES = 3
_PAIRWISE_JUDGE_RETRYABLE_STATUS: frozenset[int] = frozenset({429, 500, 502, 503, 504})
_PAIRWISE_JUDGE_BACKOFF_BASE_S = 1.0  # doubled each attempt, ±20% jitter
_INVALID_RATE_WARN_THRESHOLD = 0.05  # warn when fraction invalid > 5%


# ---------------------------------------------------------------------------
# Enums for judge responses and verdicts
# ---------------------------------------------------------------------------


class JudgeChoice(StrEnum):
    """Possible single-token responses from the LLM judge."""

    A = "A"
    B = "B"
    TIE = "TIE"


class Verdict(StrEnum):
    """Reconciled outcome of a position-swapped pairwise comparison."""

    WIN = "win"
    LOSS = "loss"
    TIE = "tie"
    INCONSISTENT = "inconsistent"
    INVALID = "invalid"


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GenerationConfig:
    """Parameters controlling text generation."""

    max_new_tokens: int = 256
    temperature: float = 0.7
    top_k: int = 50
    samples_per_condition: int = 50


@dataclass(frozen=True)
class JudgeConfig:
    """Parameters for the LLM judge."""

    model: str = "openai/gpt-5-nano"


@dataclass(frozen=True)
class FeatureSpec:
    """Specification for a single SAE feature to evaluate."""

    id: int
    description: str
    confidence: int  # 1-5


@dataclass(frozen=True)
class DistractorConfig:
    """Configuration for distractor feature mode."""

    stats_path: str
    distractor_counts: list[int] = field(default_factory=lambda: [0, 1, 5, 20, 50])
    seed: int = 12345


@dataclass(frozen=True)
class SteeringEvalConfig:
    """Top-level configuration for the steering evaluation pipeline."""

    features: list[FeatureSpec]
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)
    magnitude_levels: list[int] | None = None
    prompts: list[str] | None = None
    distractors: DistractorConfig | None = None


def load_config(path: Path) -> SteeringEvalConfig:
    """Load and validate a steering eval YAML config file.

    Raises ``ValueError`` for missing or invalid fields.
    """
    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise ValueError(f"Config must be a YAML mapping, got {type(raw).__name__}")

    # -- Features (required) --
    raw_features = raw.get("features")
    if not raw_features:
        raise ValueError("Config must contain a non-empty 'features' list")

    features: list[FeatureSpec] = []
    for i, feat in enumerate(raw_features):
        if "id" not in feat:
            raise ValueError(f"Feature at index {i} is missing required field 'id'")
        if "description" not in feat:
            raise ValueError(
                f"Feature at index {i} (id={feat.get('id')}) is missing required field 'description'"
            )
        if "confidence" not in feat:
            raise ValueError(
                f"Feature at index {i} (id={feat['id']}) is missing required field 'confidence'"
            )
        conf = feat["confidence"]
        if not isinstance(conf, int) or conf < 1 or conf > 5:
            raise ValueError(
                f"Feature {feat['id']}: confidence must be an integer 1-5, got {conf}"
            )
        features.append(
            FeatureSpec(id=feat["id"], description=feat["description"], confidence=conf)
        )

    # -- Generation (optional, with defaults) --
    raw_gen = raw.get("generation", {})
    generation = GenerationConfig(
        max_new_tokens=raw_gen.get("max_new_tokens", 256),
        temperature=raw_gen.get("temperature", 0.7),
        top_k=raw_gen.get("top_k", 50),
        samples_per_condition=raw_gen.get("samples_per_condition", 50),
    )

    # -- Judge (optional, with defaults) --
    raw_judge = raw.get("judge", {})
    judge = JudgeConfig(model=raw_judge.get("model", "openai/gpt-5-nano"))

    # -- Optional fields --
    magnitude_levels = raw.get("magnitude_levels")
    prompts = raw.get("prompts")

    # -- Distractors (optional) --
    distractors: DistractorConfig | None = None
    raw_distractors = raw.get("distractors")
    if raw_distractors is not None:
        if "stats_path" not in raw_distractors:
            raise ValueError("distractors section requires 'stats_path'")
        distractor_counts = raw_distractors.get("distractor_counts", [0, 1, 5, 20, 50])
        if 0 not in distractor_counts:
            distractor_counts = [0] + distractor_counts
        distractors = DistractorConfig(
            stats_path=raw_distractors["stats_path"],
            distractor_counts=distractor_counts,
            seed=raw_distractors.get("seed", 12345),
        )

    return SteeringEvalConfig(
        features=features,
        generation=generation,
        judge=judge,
        magnitude_levels=magnitude_levels,
        prompts=prompts,
        distractors=distractors,
    )


# ---------------------------------------------------------------------------
# Judge protocol and position-swap comparison
# ---------------------------------------------------------------------------

JUDGE_PROMPT_TEMPLATE = """\
You are evaluating whether a piece of text exhibits a specific behavioral trait.

Trait: {description}

Text A:
---
{text_a}
---

Text B:
---
{text_b}
---

Which text more strongly exhibits the described trait? Answer with exactly one of: "A", "B", or "TIE".
Output only the single word answer, nothing else."""


@dataclass(frozen=True)
class PairwiseResult:
    """Result of a position-swapped pairwise comparison."""

    verdict: Verdict
    ab_response: JudgeChoice | str = ""
    ba_response: JudgeChoice | str = ""


def format_judge_prompt(description: str, text_a: str, text_b: str) -> str:
    """Format the judge prompt with the given texts and trait description."""
    return JUDGE_PROMPT_TEMPLATE.format(
        description=description,
        text_a=text_a,
        text_b=text_b,
    )


def reconcile_judgments(ab_response: JudgeChoice, ba_response: JudgeChoice) -> Verdict:
    """Reconcile two position-swapped judge responses into a verdict.

    Parameters
    ----------
    ab_response:
        Judge response when conditioned=A, baseline=B.
    ba_response:
        Judge response when conditioned=B, baseline=A.

    Returns
    -------
    A :class:`Verdict` enum member.
    """
    if ab_response == JudgeChoice.A and ba_response == JudgeChoice.B:
        return Verdict.WIN
    if ab_response == JudgeChoice.B and ba_response == JudgeChoice.A:
        return Verdict.LOSS
    if ab_response == JudgeChoice.TIE and ba_response == JudgeChoice.TIE:
        return Verdict.TIE
    return Verdict.INCONSISTENT


@runtime_checkable
class _JudgeProtocol(Protocol):
    """Structural type for judge objects (duck-typed)."""

    def judge(self, prompt: str) -> str: ...


def judge_pair(
    conditioned_text: str,
    baseline_text: str,
    description: str,
    judge: _JudgeProtocol,
) -> PairwiseResult:
    """Run position-swapped pairwise comparison via the judge.

    Calls ``judge.judge(prompt)`` twice -- once with conditioned=A/baseline=B
    and once with the order flipped. Returns a ``PairwiseResult`` with a
    consistent verdict.

    Parameters
    ----------
    conditioned_text:
        Text generated with SAE feature conditioning.
    baseline_text:
        Text generated without conditioning.
    description:
        Human-readable description of the expected behavioral trait.
    judge:
        Object with a ``judge(prompt: str) -> str`` method.
    """
    prompt_ab = format_judge_prompt(
        description, text_a=conditioned_text, text_b=baseline_text
    )
    prompt_ba = format_judge_prompt(
        description, text_a=baseline_text, text_b=conditioned_text
    )

    ab_raw = judge.judge(prompt_ab).strip().upper()
    ba_raw = judge.judge(prompt_ba).strip().upper()

    try:
        ab_response = JudgeChoice(ab_raw)
    except ValueError:
        return PairwiseResult(
            verdict=Verdict.INVALID, ab_response=ab_raw, ba_response=ba_raw
        )
    try:
        ba_response = JudgeChoice(ba_raw)
    except ValueError:
        return PairwiseResult(
            verdict=Verdict.INVALID, ab_response=ab_response, ba_response=ba_raw
        )

    verdict = reconcile_judgments(ab_response, ba_response)
    return PairwiseResult(
        verdict=verdict, ab_response=ab_response, ba_response=ba_response
    )


# ---------------------------------------------------------------------------
# Dose-response statistics
# ---------------------------------------------------------------------------


def compute_spearman(levels: list[int], win_rates: list[float]) -> tuple[float, float]:
    """Compute Spearman rank correlation between magnitude levels and win rates.

    Returns ``(rho, p_value)``. If the data is constant (NaN from scipy),
    returns ``(0.0, 1.0)``.
    """
    result = _scipy_spearmanr(levels, win_rates)
    rho = result.statistic
    p = result.pvalue
    if math.isnan(rho):
        raise ValidationError(
            "Correlation computation produced NaN -- likely insufficient "
            "data variance"
        )
    return float(rho), float(p)


def isotonic_regression_pava(values: list[float]) -> list[float]:
    """Pool Adjacent Violators Algorithm for isotonic (non-decreasing) regression.

    Returns a list of the same length with non-decreasing values that minimize
    squared error to the input.
    """
    n = len(values)
    result = list(values)
    blocks: list[tuple[float, int]] = []

    for i in range(n):
        blocks.append((result[i], 1))
        while len(blocks) >= 2:
            (s1, c1) = blocks[-2]
            (s2, c2) = blocks[-1]
            if s1 / c1 > s2 / c2:
                blocks.pop()
                blocks[-1] = (s1 + s2, c1 + c2)
            else:
                break

    idx = 0
    for s, c in blocks:
        mean = s / c
        for _ in range(c):
            result[idx] = mean
            idx += 1

    return result


def compute_isotonic_r2(values: list[float]) -> float:
    """Compute R-squared of isotonic regression fit on the given values.

    Treats the index as the independent variable. R² = 1 - SS_res / SS_tot.
    """
    fitted = isotonic_regression_pava(values)
    mean_val = sum(values) / len(values)

    ss_res = sum((v - f) ** 2 for v, f in zip(values, fitted, strict=True))
    ss_tot = sum((v - mean_val) ** 2 for v in values)

    if ss_tot < 1e-15:
        return 1.0
    return 1.0 - ss_res / ss_tot


def bootstrap_ci(
    wins: int,
    losses: int,
    ties: int,
    n_boot: int = 1000,
    seed: int = 42,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Compute bootstrapped confidence interval for win rate.

    Win rate is defined as ``wins / (wins + losses + ties)``.
    Returns ``(lower, upper)`` bounds for the ``(1 - alpha)`` CI.
    """
    total = wins + losses + ties
    if total == 0:
        return 0.0, 0.0

    sample = [1] * wins + [0] * (losses + ties)

    rng = random.Random(seed)
    boot_rates: list[float] = []
    for _ in range(n_boot):
        resample = rng.choices(sample, k=total)
        boot_rates.append(sum(resample) / total)

    boot_rates.sort()
    lo_idx = int(n_boot * alpha / 2)
    hi_idx = int(n_boot * (1.0 - alpha / 2)) - 1
    return boot_rates[lo_idx], boot_rates[hi_idx]


# ---------------------------------------------------------------------------
# Aggregate metrics
# ---------------------------------------------------------------------------


@dataclass
class FeatureResult:
    """Summary results for a single feature (used for aggregation)."""

    feature_id: int
    confidence: int
    standard_win_rate: float
    spearman_rho: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Confidence-weighted aggregate metrics across all features."""

    confidence_weighted_win_rate: float
    confidence_weighted_monotonicity: float  # NaN if no conf >= 3
    fraction_monotonic: float  # NaN if no conf >= 3
    total_features: int
    high_confidence_features: int  # confidence >= 3


def compute_aggregate_metrics(results: list[FeatureResult]) -> AggregateMetrics:
    """Compute confidence-weighted aggregate metrics.

    Map B-10 fix: returns NaN-filled metrics when ``results`` is empty or
    every entry has confidence == 0 (avoids ``ZeroDivisionError`` in the
    weighted-mean denominator).

    Parameters
    ----------
    results:
        Per-feature results with confidence scores and statistics.
    """
    total_weight = sum(r.confidence for r in results)
    if total_weight == 0:
        # Either empty results, or all confidences are 0 — no meaningful
        # weighted mean is computable. Surface as NaN rather than raising.
        return AggregateMetrics(
            confidence_weighted_win_rate=float("nan"),
            confidence_weighted_monotonicity=float("nan"),
            fraction_monotonic=float("nan"),
            total_features=len(results),
            high_confidence_features=0,
        )

    weighted_wr = (
        sum(r.confidence * r.standard_win_rate for r in results) / total_weight
    )

    high_conf = [r for r in results if r.confidence >= 3]

    if not high_conf:
        return AggregateMetrics(
            confidence_weighted_win_rate=weighted_wr,
            confidence_weighted_monotonicity=float("nan"),
            fraction_monotonic=float("nan"),
            total_features=len(results),
            high_confidence_features=0,
        )

    hc_weight = sum(r.confidence for r in high_conf)
    weighted_mono = sum(r.confidence * r.spearman_rho for r in high_conf) / hc_weight
    n_monotonic = sum(1 for r in high_conf if r.spearman_rho > 0.8)
    frac_mono = n_monotonic / len(high_conf)

    return AggregateMetrics(
        confidence_weighted_win_rate=weighted_wr,
        confidence_weighted_monotonicity=weighted_mono,
        fraction_monotonic=frac_mono,
        total_features=len(results),
        high_confidence_features=len(high_conf),
    )


# ---------------------------------------------------------------------------
# Raw data structures for persistence and resumability
# ---------------------------------------------------------------------------


@dataclass
class RawJudgment:
    """A single pairwise judgment record."""

    pair_index: int
    conditioned_text: str
    baseline_text: str
    ab_response: JudgeChoice | str
    ba_response: JudgeChoice | str
    verdict: Verdict


@dataclass
class RawMagnitudeData:
    """Raw data for a single magnitude level of a single feature."""

    level: int
    conditioned_samples: list[tuple[str, str]]  # (prompt, generation)
    baseline_samples: list[tuple[str, str]]
    judgments: list[RawJudgment]


@dataclass(frozen=True)
class LevelStats:
    """Telemetry counters for one magnitude level."""

    n_wins: int
    n_losses: int
    n_ties: int
    n_inconsistent_verdicts: int
    n_invalid_verdicts: int
    n_total: int


@dataclass
class RawFeatureData:
    """Raw data for a single feature across all magnitude levels."""

    feature_id: int
    description: str
    confidence: int
    magnitude_data: dict[int, RawMagnitudeData]


def save_raw_data(data: dict[int, RawFeatureData], path: Path) -> None:
    """Serialize raw evaluation data to a YAML file."""
    serialized: dict = {"features": []}
    for feat_data in data.values():
        feat_entry: dict = {
            "id": feat_data.feature_id,
            "description": feat_data.description,
            "confidence": feat_data.confidence,
            "magnitude_levels": [],
        }
        for level in sorted(feat_data.magnitude_data.keys()):
            mag = feat_data.magnitude_data[level]
            mag_entry: dict = {
                "level": mag.level,
                "conditioned_samples": [
                    {"prompt": p, "generation": g} for p, g in mag.conditioned_samples
                ],
                "baseline_samples": [
                    {"prompt": p, "generation": g} for p, g in mag.baseline_samples
                ],
                "judgments": [
                    {
                        "pair_index": j.pair_index,
                        "conditioned_text": j.conditioned_text,
                        "baseline_text": j.baseline_text,
                        "ab_response": getattr(j.ab_response, "value", j.ab_response),
                        "ba_response": getattr(j.ba_response, "value", j.ba_response),
                        "verdict": j.verdict.value,
                    }
                    for j in mag.judgments
                ],
            }
            feat_entry["magnitude_levels"].append(mag_entry)
        serialized["features"].append(feat_entry)

    with open(path, "w") as f:
        yaml.dump(
            serialized, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def load_raw_data(path: Path) -> dict[int, RawFeatureData]:
    """Load raw evaluation data from a YAML file.

    Returns an empty dict if the file does not exist.
    """
    if not path.exists():
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not isinstance(raw, dict):
        raise DataLoadError(
            f"Corrupted YAML in {path}: expected dict, got "
            f"{type(raw).__name__}"
        )
    if "features" not in raw:
        raise DataLoadError(
            f"YAML in {path} missing expected key 'features'"
        )

    result: dict[int, RawFeatureData] = {}
    for feat_entry in raw["features"]:
        mag_data: dict[int, RawMagnitudeData] = {}
        for mag_entry in feat_entry.get("magnitude_levels", []):
            judgments = []
            for j in mag_entry.get("judgments", []):
                verdict = Verdict(j["verdict"])
                # INVALID verdicts may have non-standard responses (e.g. unparseable judge output)
                if verdict == Verdict.INVALID:
                    ab: JudgeChoice | str = j["ab_response"]
                    ba: JudgeChoice | str = j["ba_response"]
                else:
                    try:
                        ab = JudgeChoice(j["ab_response"])
                    except ValueError as exc:
                        raise DataLoadError(
                            f"invalid ab_response {j['ab_response']!r} "
                            f"(expected one of {[e.value for e in JudgeChoice]})"
                        ) from exc
                    try:
                        ba = JudgeChoice(j["ba_response"])
                    except ValueError as exc:
                        raise DataLoadError(
                            f"invalid ba_response {j['ba_response']!r} "
                            f"(expected one of {[e.value for e in JudgeChoice]})"
                        ) from exc
                judgments.append(
                    RawJudgment(
                        pair_index=j["pair_index"],
                        conditioned_text=j["conditioned_text"],
                        baseline_text=j["baseline_text"],
                        ab_response=ab,
                        ba_response=ba,
                        verdict=verdict,
                    )
                )
            mag_data[mag_entry["level"]] = RawMagnitudeData(
                level=mag_entry["level"],
                conditioned_samples=[
                    (s["prompt"], s["generation"])
                    for s in mag_entry.get("conditioned_samples", [])
                ],
                baseline_samples=[
                    (s["prompt"], s["generation"])
                    for s in mag_entry.get("baseline_samples", [])
                ],
                judgments=judgments,
            )
        result[feat_entry["id"]] = RawFeatureData(
            feature_id=feat_entry["id"],
            description=feat_entry["description"],
            confidence=feat_entry["confidence"],
            magnitude_data=mag_data,
        )

    return result


def save_distractor_data(
    data: dict[int, dict[int, RawFeatureData]], path: Path
) -> None:
    """Save unified eval data keyed by ``{feature_id: {n_distractors: RawFeatureData}}``."""
    serialized: dict = {"distractor_features": []}
    for feat_id in sorted(data.keys()):
        by_ndist = data[feat_id]
        feat_entry: dict = {"id": feat_id, "distractor_counts": []}
        for n_dist in sorted(by_ndist.keys()):
            feat_data = by_ndist[n_dist]
            mag_entries = []
            for level in sorted(feat_data.magnitude_data.keys()):
                mag = feat_data.magnitude_data[level]
                mag_entries.append(
                    {
                        "level": mag.level,
                        "conditioned_samples": [
                            {"prompt": p, "generation": g}
                            for p, g in mag.conditioned_samples
                        ],
                        "baseline_samples": [
                            {"prompt": p, "generation": g}
                            for p, g in mag.baseline_samples
                        ],
                        "judgments": [
                            {
                                "pair_index": j.pair_index,
                                "conditioned_text": j.conditioned_text,
                                "baseline_text": j.baseline_text,
                                "ab_response": getattr(
                                    j.ab_response, "value", j.ab_response
                                ),
                                "ba_response": getattr(
                                    j.ba_response, "value", j.ba_response
                                ),
                                "verdict": j.verdict.value,
                            }
                            for j in mag.judgments
                        ],
                    }
                )
            feat_entry["distractor_counts"].append(
                {
                    "n_distractors": n_dist,
                    "description": feat_data.description,
                    "confidence": feat_data.confidence,
                    "magnitude_levels": mag_entries,
                }
            )
        serialized["distractor_features"].append(feat_entry)

    with open(path, "w") as f:
        yaml.dump(
            serialized, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


def load_distractor_data(path: Path) -> dict[int, dict[int, RawFeatureData]]:
    """Load unified eval data. Returns empty dict if file doesn't exist."""
    if not path.exists():
        return {}

    with open(path) as f:
        raw = yaml.safe_load(f)

    if not raw or "distractor_features" not in raw:
        return {}

    result: dict[int, dict[int, RawFeatureData]] = {}
    for feat_entry in raw["distractor_features"]:
        feat_id = feat_entry["id"]
        result[feat_id] = {}
        for dc_entry in feat_entry.get("distractor_counts", []):
            n_dist = dc_entry["n_distractors"]
            mag_data: dict[int, RawMagnitudeData] = {}
            for mag_entry in dc_entry.get("magnitude_levels", []):
                judgments = []
                for j in mag_entry.get("judgments", []):
                    verdict = Verdict(j["verdict"])
                    if verdict == Verdict.INVALID:
                        ab: JudgeChoice | str = j["ab_response"]
                        ba: JudgeChoice | str = j["ba_response"]
                    else:
                        try:
                            ab = JudgeChoice(j["ab_response"])
                        except ValueError as exc:
                            raise DataLoadError(
                                f"invalid ab_response {j['ab_response']!r} "
                                f"(expected one of {[e.value for e in JudgeChoice]})"
                            ) from exc
                        try:
                            ba = JudgeChoice(j["ba_response"])
                        except ValueError as exc:
                            raise DataLoadError(
                                f"invalid ba_response {j['ba_response']!r} "
                                f"(expected one of {[e.value for e in JudgeChoice]})"
                            ) from exc
                    judgments.append(
                        RawJudgment(
                            pair_index=j["pair_index"],
                            conditioned_text=j["conditioned_text"],
                            baseline_text=j["baseline_text"],
                            ab_response=ab,
                            ba_response=ba,
                            verdict=verdict,
                        )
                    )
                mag_data[mag_entry["level"]] = RawMagnitudeData(
                    level=mag_entry["level"],
                    conditioned_samples=[
                        (s["prompt"], s["generation"])
                        for s in mag_entry.get("conditioned_samples", [])
                    ],
                    baseline_samples=[
                        (s["prompt"], s["generation"])
                        for s in mag_entry.get("baseline_samples", [])
                    ],
                    judgments=judgments,
                )
            result[feat_id][n_dist] = RawFeatureData(
                feature_id=feat_id,
                description=dc_entry["description"],
                confidence=dc_entry["confidence"],
                magnitude_data=mag_data,
            )

    return result


def generate_report(
    feature_results: list[FeatureResult],
    dose_response_data: dict[int, dict],
    config_file: str,
    output_path: Path,
) -> None:
    """Generate the summary YAML report.

    Parameters
    ----------
    feature_results:
        Per-feature result summaries.
    dose_response_data:
        Mapping from feature ID to dose-response statistics dict.
    config_file:
        Path to the config file used (for metadata).
    output_path:
        Where to write the YAML report.
    """
    agg = compute_aggregate_metrics(feature_results)

    features_section = []
    for r in feature_results:
        dose = dose_response_data.get(r.feature_id, {})
        features_section.append(
            {
                "id": r.feature_id,
                "confidence": r.confidence,
                "standard_win_rate": r.standard_win_rate,
                "dose_response": dose,
            }
        )

    report = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "config_file": config_file,
            "num_features": len(feature_results),
        },
        "features": features_section,
        "aggregate": {
            "confidence_weighted_win_rate": agg.confidence_weighted_win_rate,
            "confidence_weighted_monotonicity": (
                None
                if math.isnan(agg.confidence_weighted_monotonicity)
                else agg.confidence_weighted_monotonicity
            ),
            "fraction_monotonic": (
                None if math.isnan(agg.fraction_monotonic) else agg.fraction_monotonic
            ),
            "total_features": agg.total_features,
            "high_confidence_features": agg.high_confidence_features,
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(
            report, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_dose_response(
    levels: list[int],
    win_rates: list[float],
    ci_lower: list[float],
    ci_upper: list[float],
    feature_id: int,
    description: str,
    confidence: int,
    spearman_rho: float,
    output_path: Path,
) -> None:
    """Plot a single feature's dose-response curve with error bars.

    Low-confidence features (confidence <= 2) get a 'LOW CONFIDENCE' watermark.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    yerr_lower = [max(0.0, wr - lo) for wr, lo in zip(win_rates, ci_lower, strict=True)]
    yerr_upper = [max(0.0, hi - wr) for wr, hi in zip(win_rates, ci_upper, strict=True)]
    ax.errorbar(
        levels,
        win_rates,
        yerr=[yerr_lower, yerr_upper],
        fmt="o-",
        capsize=4,
        color="steelblue",
        linewidth=2,
        markersize=6,
    )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="chance")

    ax.set_xlabel("Magnitude Level", fontsize=12)
    ax.set_ylabel("Consistent Win Rate", fontsize=12)
    ax.set_title(
        f"Feature {feature_id}: {description}\n(confidence={confidence}, \u03c1={spearman_rho:.2f})",
        fontsize=11,
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(levels)
    ax.legend(loc="upper left")

    if confidence <= 2:
        ax.text(
            0.5,
            0.5,
            "LOW CONFIDENCE",
            transform=ax.transAxes,
            fontsize=28,
            color="red",
            alpha=0.3,
            ha="center",
            va="center",
            rotation=30,
            fontweight="bold",
        )

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_summary_grid(
    features_data: list[dict],
    output_path: Path,
    cols: int = 3,
) -> None:
    """Plot a grid of dose-response curves for all features.

    Parameters
    ----------
    features_data:
        List of dicts, each with keys: feature_id, description, confidence,
        levels, win_rates, ci_lower, ci_upper, spearman_rho.
    output_path:
        Where to save the PNG.
    cols:
        Number of columns in the grid.
    """
    n = len(features_data)
    rows = (n + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(6 * cols, 4 * rows), squeeze=False)

    for idx, fd in enumerate(features_data):
        r, c = divmod(idx, cols)
        ax = axes[r][c]

        yerr_lo = [max(0.0, wr - lo) for wr, lo in zip(fd["win_rates"], fd["ci_lower"], strict=True)]
        yerr_hi = [max(0.0, hi - wr) for wr, hi in zip(fd["win_rates"], fd["ci_upper"], strict=True)]
        ax.errorbar(
            fd["levels"],
            fd["win_rates"],
            yerr=[yerr_lo, yerr_hi],
            fmt="o-",
            capsize=3,
            color="steelblue",
            linewidth=1.5,
            markersize=4,
        )
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.4)
        ax.set_title(
            f"F{fd['feature_id']}: {fd['description'][:30]}\n\u03c1={fd['spearman_rho']:.2f}",
            fontsize=9,
        )
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(fd["levels"])

        if fd["confidence"] <= 2:
            ax.text(
                0.5,
                0.5,
                "LOW CONF",
                transform=ax.transAxes,
                fontsize=16,
                color="red",
                alpha=0.3,
                ha="center",
                va="center",
                rotation=30,
                fontweight="bold",
            )

    for idx in range(n, rows * cols):
        r, c = divmod(idx, cols)
        axes[r][c].set_visible(False)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_robustness_overlay(
    levels: list[int],
    win_rates_by_ndist: dict[int, list[float]],
    feature_id: int,
    description: str,
    output_path: Path,
) -> None:
    """Plot overlaid dose-response curves for each distractor count.

    ``n_dist=0`` is the darkest line; higher counts get progressively lighter.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    sorted_counts = sorted(win_rates_by_ndist.keys())
    n_counts = len(sorted_counts)

    for idx, n_dist in enumerate(sorted_counts):
        alpha = 1.0 - 0.6 * (idx / max(n_counts - 1, 1))
        label = f"n_dist={n_dist}"
        ax.plot(
            levels,
            win_rates_by_ndist[n_dist],
            "o-",
            alpha=alpha,
            label=label,
            linewidth=2,
            markersize=5,
        )

    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Magnitude Level", fontsize=12)
    ax.set_ylabel("Consistent Win Rate", fontsize=12)
    ax.set_title(
        f"Feature {feature_id}: {description}\nRobustness Overlay", fontsize=11
    )
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(levels)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def plot_robustness_summary(
    feature_ids: list[int],
    median_win_rates: dict[int, dict[int, float]],
    distractor_counts: list[int],
    output_path: Path,
) -> None:
    """Grouped bar chart of median-magnitude win rate per feature at each distractor count.

    Parameters
    ----------
    feature_ids:
        Feature IDs to include.
    median_win_rates:
        ``{feature_id: {n_dist: win_rate}}``.
    distractor_counts:
        Distractor counts (x-groups within each feature).
    output_path:
        Where to save the PNG.
    """
    n_features = len(feature_ids)
    n_groups = len(distractor_counts)
    bar_width = 0.8 / max(n_groups, 1)

    fig, ax = plt.subplots(figsize=(max(8, n_features * 1.5), 5))

    x = np.arange(n_features)
    for g_idx, n_dist in enumerate(distractor_counts):
        vals = [median_win_rates.get(fid, {}).get(n_dist, 0.0) for fid in feature_ids]
        offset = (g_idx - n_groups / 2 + 0.5) * bar_width
        ax.bar(x + offset, vals, bar_width, label=f"n_dist={n_dist}", alpha=0.8)

    ax.set_xlabel("Feature ID", fontsize=12)
    ax.set_ylabel("Median-Magnitude Win Rate", fontsize=12)
    ax.set_title("Robustness Summary", fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels([str(fid) for fid in feature_ids])
    ax.set_ylim(0.0, 1.0)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Distractor report generation
# ---------------------------------------------------------------------------


def generate_distractor_report(
    distractor_results: dict[int, dict[int, FeatureResult]],
    distractor_dose_data: dict[int, dict[int, dict]],
    distractor_counts: list[int],
    config_file: str,
    output_path: Path,
) -> None:
    """Generate the unified YAML report with distractor count as a first-class dimension.

    Parameters
    ----------
    distractor_results:
        ``{feature_id: {n_dist: FeatureResult}}``.
    distractor_dose_data:
        ``{feature_id: {n_dist: dose_response_dict}}``.
    distractor_counts:
        All distractor counts evaluated.
    config_file:
        Config file path (for metadata).
    output_path:
        Where to write the report.
    """
    features_section = []
    all_feature_ids = sorted(distractor_results.keys())

    for fid in all_feature_ids:
        by_ndist = distractor_results[fid]
        # Use n_dist=0 result for top-level confidence
        base_result = by_ndist.get(0, next(iter(by_ndist.values())))

        results_by_ndist: dict[int, dict] = {}
        for n_dist in sorted(by_ndist.keys()):
            r = by_ndist[n_dist]
            dose = distractor_dose_data.get(fid, {}).get(n_dist, {})
            results_by_ndist[n_dist] = {
                # Renamed from the misleading "median_win_rate" (Wave 3D/1):
                # this is the win rate at the MIDDLE MAGNITUDE INDEX of
                # `mag_levels`, not a statistical median of per-level win
                # rates. It is NOT a specificity estimator — see the module
                # docstring and inline_steering_eval for canonical specificity.
                "standard_win_rate_at_middle_magnitude": r.standard_win_rate,
                "spearman_rho": r.spearman_rho,
                "dose_response": dose,
            }

        features_section.append(
            {
                "id": fid,
                "confidence": base_result.confidence,
                "results_by_distractor_count": results_by_ndist,
            }
        )

    # Aggregate metrics by distractor count
    agg_by_ndist: dict[int, dict] = {}
    for n_dist in distractor_counts:
        results_at_ndist = []
        for fid in all_feature_ids:
            if n_dist in distractor_results.get(fid, {}):
                results_at_ndist.append(distractor_results[fid][n_dist])
        if results_at_ndist:
            agg = compute_aggregate_metrics(results_at_ndist)
            agg_by_ndist[n_dist] = {
                "confidence_weighted_win_rate": agg.confidence_weighted_win_rate,
                "confidence_weighted_monotonicity": (
                    None
                    if math.isnan(agg.confidence_weighted_monotonicity)
                    else agg.confidence_weighted_monotonicity
                ),
            }

    # Mean robustness ratio at highest distractor count
    max_ndist = max(distractor_counts)
    ratios = []
    for fid in all_feature_ids:
        wr_0 = distractor_results.get(fid, {}).get(0)
        wr_max = distractor_results.get(fid, {}).get(max_ndist)
        if wr_0 and wr_max and wr_0.standard_win_rate > 0:
            ratios.append(wr_max.standard_win_rate / wr_0.standard_win_rate)

    report = {
        "metadata": {
            "timestamp": datetime.now(UTC).isoformat(),
            "config_file": config_file,
            "num_features": len(all_feature_ids),
            "distractor_counts": distractor_counts,
        },
        "features": features_section,
        "aggregate": {
            "by_distractor_count": agg_by_ndist,
            "mean_robustness_ratio_at_max": (
                sum(ratios) / len(ratios) if ratios else None
            ),
        },
    }

    with open(output_path, "w") as f:
        yaml.dump(
            report, f, default_flow_style=False, sort_keys=False, allow_unicode=True
        )


# ---------------------------------------------------------------------------
# Pipeline orchestrator
# ---------------------------------------------------------------------------


def _evaluate_feature_at_magnitude(
    feature: FeatureSpec,
    magnitude: int,
    model: GPT | EncoderDecoderGPT,
    tokenizer: object,
    device: t.device,
    judge: _JudgeProtocol,
    config: SteeringEvalConfig,
    existing_mag_data: RawMagnitudeData | None,
    context_provider: ContextProvider,
    target_mode: TargetConditioningMode,
    distractor_sampler: DistractorSampler | None = None,
    n_distractors: int = 0,
) -> RawMagnitudeData:
    """Generate samples and run judge comparisons for one feature at one magnitude.

    Resumes from existing data if available.

    When ``n_distractors > 0`` and a ``distractor_sampler`` is provided,
    each sample gets a unique set of distractor features. Conditioned samples
    include the target + distractors; baseline samples include only the
    distractors. This isolates the incremental effect of the target feature.
    """
    n_samples = config.generation.samples_per_condition

    # Resume or start fresh
    if existing_mag_data and len(existing_mag_data.judgments) >= n_samples:
        logger.debug(
            f"  Magnitude {magnitude}: already complete ({len(existing_mag_data.judgments)} judgments)"
        )
        return existing_mag_data

    prompts = config.prompts or [""]

    conditioned_samples: list[tuple[str, str]] = []
    baseline_samples: list[tuple[str, str]] = []
    judgments: list[RawJudgment] = []

    if existing_mag_data:
        conditioned_samples = list(existing_mag_data.conditioned_samples)
        baseline_samples = list(existing_mag_data.baseline_samples)
        judgments = list(existing_mag_data.judgments)

    start_idx = len(judgments)

    # Distractor seed components (from plan):
    # base_seed + feat_id * 10000 + n_dist * 100 + sample_idx
    dist_config = config.distractors

    for i in range(start_idx, n_samples):
        prompt = prompts[i % len(prompts)]

        # Sample distractors for this sample (shared between conditioned & baseline)
        distractor_conditions: dict[int, int] = {}
        if (
            n_distractors > 0
            and distractor_sampler is not None
            and dist_config is not None
        ):
            sample_seed = (
                dist_config.seed + feature.id * 10000 + n_distractors * 100 + i
            )
            dset = distractor_sampler.sample(n_distractors, sample_seed)
            distractor_conditions = dset.to_feature_conditions()

        conditioned_features = {feature.id: magnitude, **distractor_conditions}
        baseline_features = distractor_conditions if distractor_conditions else None

        # Generate conditioned sample
        if i >= len(conditioned_samples):
            cond_engine = build_engine_for_features(
                model,
                tokenizer,
                device,
                features=conditioned_features,
                context_provider=context_provider,
                target_mode=target_mode,
            )
            cond_text = cond_engine.generate(
                prompt,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_k=config.generation.top_k,
                seed=magnitude * 10000 + i,
                render_as_chat=True,
            )
            conditioned_samples.append((prompt, cond_text))
        else:
            cond_text = conditioned_samples[i][1]

        # Generate baseline sample
        if i >= len(baseline_samples):
            base_engine = build_engine_for_features(
                model,
                tokenizer,
                device,
                features=baseline_features,
                context_provider=context_provider,
                target_mode=target_mode,
            )
            base_text = base_engine.generate(
                prompt,
                max_new_tokens=config.generation.max_new_tokens,
                temperature=config.generation.temperature,
                top_k=config.generation.top_k,
                seed=magnitude * 10000 + n_samples + i,
                render_as_chat=True,
            )
            baseline_samples.append((prompt, base_text))
        else:
            base_text = baseline_samples[i][1]

        # Judge with position swap
        result = judge_pair(cond_text, base_text, feature.description, judge)

        judgments.append(
            RawJudgment(
                pair_index=i,
                conditioned_text=cond_text,
                baseline_text=base_text,
                ab_response=result.ab_response,
                ba_response=result.ba_response,
                verdict=result.verdict,
            )
        )

    return RawMagnitudeData(
        level=magnitude,
        conditioned_samples=conditioned_samples,
        baseline_samples=baseline_samples,
        judgments=judgments,
    )


def _compute_level_stats(
    feat_mag_data: dict[int, RawMagnitudeData],
    mag_levels: list[int],
) -> tuple[list[float], list[float], list[float], list[LevelStats]]:
    """Compute per-level win rates and CIs from magnitude data.

    Returns ``(win_rates, ci_lower, ci_upper, level_stats)`` where
    ``level_stats`` contains per-level telemetry counters.

    **Accounting convention:** INCONSISTENT (swap disagrees) and INVALID
    (unparseable judge output) verdicts are counted as LOSSES for the method
    being evaluated. The conservative choice prevents a flaky judge from
    silently inflating the win rate when its errors drop out of the
    denominator. A warning is emitted when the combined invalid rate exceeds
    ``_INVALID_RATE_WARN_THRESHOLD`` (5%) so downstream consumers see
    judge flakiness.

    This accounting applies only to the pairwise judge evaluator; the other
    specificity metrics use different baselines and are unchanged.
    """
    level_win_rates: list[float] = []
    level_ci_lower: list[float] = []
    level_ci_upper: list[float] = []
    level_stats: list[LevelStats] = []

    for mag in mag_levels:
        md = feat_mag_data[mag]
        wins = sum(1 for j in md.judgments if j.verdict == Verdict.WIN)
        losses = sum(1 for j in md.judgments if j.verdict == Verdict.LOSS)
        ties = sum(1 for j in md.judgments if j.verdict == Verdict.TIE)
        inconsistent = sum(
            1 for j in md.judgments if j.verdict == Verdict.INCONSISTENT
        )
        invalid = sum(1 for j in md.judgments if j.verdict == Verdict.INVALID)
        # Denominator: all judgments (INVALID/INCONSISTENT count as losses).
        total = wins + losses + ties + inconsistent + invalid

        win_rate = wins / total if total > 0 else 0.0
        level_win_rates.append(win_rate)

        # For bootstrap CI, treat inconsistent + invalid as losses so the
        # CI is drawn from the same distribution as the point estimate.
        lo, hi = bootstrap_ci(wins, losses + inconsistent + invalid, ties)
        level_ci_lower.append(lo)
        level_ci_upper.append(hi)

        stats = LevelStats(
            n_wins=wins,
            n_losses=losses,
            n_ties=ties,
            n_inconsistent_verdicts=inconsistent,
            n_invalid_verdicts=invalid,
            n_total=total,
        )
        level_stats.append(stats)

        # Telemetry: warn if flaky-judge rate exceeds threshold.
        if total > 0:
            flaky = inconsistent + invalid
            flaky_rate = flaky / total
            if flaky_rate > _INVALID_RATE_WARN_THRESHOLD:
                logger.warning(
                    f"High invalid/inconsistent verdict rate at magnitude {mag}: "
                    f"{flaky_rate:.2%} ({flaky}/{total}) — "
                    f"invalid={invalid}, inconsistent={inconsistent}. "
                    "Reported win rate may understate true steering effect."
                )

    return level_win_rates, level_ci_lower, level_ci_upper, level_stats


def run_steering_eval(
    config: SteeringEvalConfig,
    judge: _JudgeProtocol,
    output_dir: Path,
    *,
    model: GPT | EncoderDecoderGPT,
    tokenizer: object,
    device: t.device,
    context_provider: ContextProvider,
    target_mode: TargetConditioningMode,
) -> None:
    """Run the full steering evaluation pipeline.

    Parameters
    ----------
    config:
        Parsed evaluation configuration.
    judge:
        An object with a ``judge(prompt: str) -> str`` method.
    output_dir:
        Directory for all outputs (report, raw data, plots).
    model:
        The model to generate from (GPT or EncoderDecoderGPT).
    tokenizer:
        Tokenizer with ``encode`` and ``render_for_completion`` methods.
    device:
        Torch device for inference.
    context_provider:
        Provider for real prompt/context features.
    """
    if not isinstance(model, EncoderDecoderGPT):
        raise ValueError(
            "Steering eval requires an EncoderDecoderGPT with full-context "
            "conditioning; decoder-only unconditioned baselines are not supported."
        )
    if context_provider is None:
        raise ValueError(
            "Encoder-decoder steering eval requires a context_provider with "
            "real prompt/context features."
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # Determine magnitude levels
    if config.magnitude_levels is not None:
        mag_levels = config.magnitude_levels
    else:
        mag_levels = list(range(10))  # default: 10 buckets

    if config.distractors is not None:
        _run_distractor_eval(
            config,
            judge,
            output_dir,
            plots_dir,
            mag_levels,
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_provider=context_provider,
            target_mode=target_mode,
        )
    else:
        _run_isolated_eval(
            config,
            judge,
            output_dir,
            plots_dir,
            mag_levels,
            model=model,
            tokenizer=tokenizer,
            device=device,
            context_provider=context_provider,
            target_mode=target_mode,
        )

    logger.info(f"Evaluation complete. Results saved to {output_dir}")


def _run_isolated_eval(
    config: SteeringEvalConfig,
    judge: _JudgeProtocol,
    output_dir: Path,
    plots_dir: Path,
    mag_levels: list[int],
    *,
    model: GPT | EncoderDecoderGPT,
    tokenizer: object,
    device: t.device,
    context_provider: ContextProvider,
    target_mode: TargetConditioningMode,
) -> None:
    """Run the original isolated (no-distractor) evaluation pipeline."""
    raw_data_path = output_dir / "raw_data.yaml"
    raw_data = load_raw_data(raw_data_path)

    feature_results: list[FeatureResult] = []
    all_plot_data: list[dict] = []

    for feat in config.features:
        logger.info(
            f"Evaluating feature {feat.id}: {feat.description} (confidence={feat.confidence})"
        )

        existing_feat = raw_data.get(feat.id)
        feat_mag_data: dict[int, RawMagnitudeData] = {}

        for mag in mag_levels:
            existing_mag = (
                existing_feat.magnitude_data.get(mag)
                if existing_feat is not None
                else None
            )

            mag_data = _evaluate_feature_at_magnitude(
                feature=feat,
                magnitude=mag,
                model=model,
                tokenizer=tokenizer,
                device=device,
                context_provider=context_provider,
                target_mode=target_mode,
                judge=judge,
                config=config,
                existing_mag_data=existing_mag,
            )
            feat_mag_data[mag] = mag_data

        # Save raw data after each feature (crash safety)
        raw_data[feat.id] = RawFeatureData(
            feature_id=feat.id,
            description=feat.description,
            confidence=feat.confidence,
            magnitude_data=feat_mag_data,
        )
        save_raw_data(raw_data, raw_data_path)

        # Compute per-level statistics (level_stats ignored in summary plot)
        (
            level_win_rates,
            level_ci_lower,
            level_ci_upper,
            _level_stats,
        ) = _compute_level_stats(feat_mag_data, mag_levels)

        # Dose-response statistics
        try:
            rho, p_val = compute_spearman(mag_levels, level_win_rates)
        except ValidationError:
            logger.warning(
                f"  Feature {feat.id}: Spearman correlation is undefined "
                "(constant win rates); setting rho=0.0, p=1.0"
            )
            rho, p_val = 0.0, 1.0
        iso_r2 = compute_isotonic_r2(level_win_rates)

        # Standard magnitude win rate (median level)
        median_idx = len(mag_levels) // 2
        standard_wr = level_win_rates[median_idx]

        feature_results.append(
            FeatureResult(
                feature_id=feat.id,
                confidence=feat.confidence,
                standard_win_rate=standard_wr,
                spearman_rho=rho,
            )
        )

        # Plot individual feature
        plot_dose_response(
            levels=mag_levels,
            win_rates=level_win_rates,
            ci_lower=level_ci_lower,
            ci_upper=level_ci_upper,
            feature_id=feat.id,
            description=feat.description,
            confidence=feat.confidence,
            spearman_rho=rho,
            output_path=plots_dir / f"feature_{feat.id}.png",
        )

        all_plot_data.append(
            {
                "feature_id": feat.id,
                "description": feat.description,
                "confidence": feat.confidence,
                "levels": mag_levels,
                "win_rates": level_win_rates,
                "ci_lower": level_ci_lower,
                "ci_upper": level_ci_upper,
                "spearman_rho": rho,
            }
        )

        logger.info(
            f"  Feature {feat.id}: win_rate={standard_wr:.2f}, rho={rho:.2f}, R2={iso_r2:.2f}"
        )

    # Summary grid
    if all_plot_data:
        plot_summary_grid(all_plot_data, plots_dir / "summary_grid.png")

    # Build dose-response data for report
    dose_report_data: dict[int, dict] = {}
    for pd_entry in all_plot_data:
        fid = pd_entry["feature_id"]
        try:
            rho_val, p_val = compute_spearman(pd_entry["levels"], pd_entry["win_rates"])
        except ValidationError:
            rho_val, p_val = 0.0, 1.0
        iso_r2_val = compute_isotonic_r2(pd_entry["win_rates"])
        dose_report_data[fid] = {
            "magnitude_levels": pd_entry["levels"],
            "win_rates": pd_entry["win_rates"],
            "spearman_rho": rho_val,
            "spearman_p": p_val,
            "isotonic_r2": iso_r2_val,
        }

    # Generate report
    generate_report(
        feature_results=feature_results,
        dose_response_data=dose_report_data,
        config_file="config.yaml",
        output_path=output_dir / "report.yaml",
    )


def _run_distractor_eval(
    config: SteeringEvalConfig,
    judge: _JudgeProtocol,
    output_dir: Path,
    plots_dir: Path,
    mag_levels: list[int],
    *,
    model: GPT | EncoderDecoderGPT,
    tokenizer: object,
    device: t.device,
    context_provider: ContextProvider,
    target_mode: TargetConditioningMode,
) -> None:
    """Run evaluation with distractor features at multiple counts."""
    assert config.distractors is not None

    excluded_ids = {f.id for f in config.features}
    distractor_data_path = output_dir / "distractor_raw_data.yaml"
    distractor_data = load_distractor_data(distractor_data_path)

    # Collect results for report: {feat_id: {n_dist: FeatureResult}}
    distractor_results: dict[int, dict[int, FeatureResult]] = {}
    distractor_dose_data: dict[int, dict[int, dict]] = {}

    # Track win rates for robustness overlay: {feat_id: {n_dist: win_rates}}
    robustness_win_rates: dict[int, dict[int, list[float]]] = {}
    # Track median-magnitude win rates: {feat_id: {n_dist: wr}}
    median_win_rates: dict[int, dict[int, float]] = {}

    for feat in config.features:
        logger.info(
            f"Evaluating feature {feat.id}: {feat.description} (confidence={feat.confidence})"
        )

        distractor_results.setdefault(feat.id, {})
        distractor_dose_data.setdefault(feat.id, {})
        robustness_win_rates.setdefault(feat.id, {})
        median_win_rates.setdefault(feat.id, {})

        for n_dist in config.distractors.distractor_counts:
            logger.info(f"  n_distractors={n_dist}")

            # Build sampler only when needed
            sampler: DistractorSampler | None = None
            if n_dist > 0:
                sampler = DistractorSampler(
                    stats_path=Path(config.distractors.stats_path),
                    magnitude_levels=mag_levels,
                    excluded_feature_ids=excluded_ids,
                )

            # Check existing data for resume
            existing_feat_ndist = distractor_data.get(feat.id, {}).get(n_dist)

            feat_mag_data: dict[int, RawMagnitudeData] = {}
            for mag in mag_levels:
                existing_mag = (
                    existing_feat_ndist.magnitude_data.get(mag)
                    if existing_feat_ndist is not None
                    else None
                )
                feat_mag_data[mag] = _evaluate_feature_at_magnitude(
                    feature=feat,
                    magnitude=mag,
                    model=model,
                    tokenizer=tokenizer,
                    device=device,
                    context_provider=context_provider,
                    target_mode=target_mode,
                    judge=judge,
                    config=config,
                    existing_mag_data=existing_mag,
                    distractor_sampler=sampler,
                    n_distractors=n_dist,
                )

            # Save to unified distractor data after each (feature, n_dist)
            feat_raw = RawFeatureData(
                feature_id=feat.id,
                description=feat.description,
                confidence=feat.confidence,
                magnitude_data=feat_mag_data,
            )
            distractor_data.setdefault(feat.id, {})[n_dist] = feat_raw
            save_distractor_data(distractor_data, distractor_data_path)

            # Compute stats (level_stats ignored in summary plot for distractor)
            (
                level_win_rates,
                level_ci_lower,
                level_ci_upper,
                _level_stats,
            ) = _compute_level_stats(feat_mag_data, mag_levels)
            try:
                rho, p_val = compute_spearman(mag_levels, level_win_rates)
            except ValidationError:
                logger.warning(
                    f"    Feature {feat.id} n_dist={n_dist}: Spearman correlation "
                    "is undefined (constant win rates); setting rho=0.0, p=1.0"
                )
                rho, p_val = 0.0, 1.0
            iso_r2 = compute_isotonic_r2(level_win_rates)

            median_idx = len(mag_levels) // 2
            standard_wr = level_win_rates[median_idx]

            distractor_results[feat.id][n_dist] = FeatureResult(
                feature_id=feat.id,
                confidence=feat.confidence,
                standard_win_rate=standard_wr,
                spearman_rho=rho,
            )
            distractor_dose_data[feat.id][n_dist] = {
                "magnitude_levels": mag_levels,
                "win_rates": level_win_rates,
                "spearman_rho": rho,
                "spearman_p": p_val,
                "isotonic_r2": iso_r2,
            }

            robustness_win_rates[feat.id][n_dist] = level_win_rates
            median_win_rates[feat.id][n_dist] = standard_wr

            # Individual dose-response plot for each (feature, n_dist)
            plot_dose_response(
                levels=mag_levels,
                win_rates=level_win_rates,
                ci_lower=level_ci_lower,
                ci_upper=level_ci_upper,
                feature_id=feat.id,
                description=feat.description,
                confidence=feat.confidence,
                spearman_rho=rho,
                output_path=plots_dir / f"feature_{feat.id}_ndist{n_dist}.png",
            )

            logger.info(
                f"    Feature {feat.id} n_dist={n_dist}: "
                f"win_rate={standard_wr:.2f}, rho={rho:.2f}, R2={iso_r2:.2f}"
            )

        # Robustness overlay for this feature
        plot_robustness_overlay(
            levels=mag_levels,
            win_rates_by_ndist=robustness_win_rates[feat.id],
            feature_id=feat.id,
            description=feat.description,
            output_path=plots_dir / f"feature_{feat.id}_robustness.png",
        )

    # Robustness summary across features
    feature_ids = [f.id for f in config.features]
    plot_robustness_summary(
        feature_ids=feature_ids,
        median_win_rates=median_win_rates,
        distractor_counts=config.distractors.distractor_counts,
        output_path=plots_dir / "robustness_summary.png",
    )

    # Generate unified report
    generate_distractor_report(
        distractor_results=distractor_results,
        distractor_dose_data=distractor_dose_data,
        distractor_counts=config.distractors.distractor_counts,
        config_file="config.yaml",
        output_path=output_dir / "report.yaml",
    )


# ---------------------------------------------------------------------------
# Pairwise judge adapter (wraps JudgeClient for pairwise comparison)
# ---------------------------------------------------------------------------


class PairwiseJudge:
    """Adapts JudgeClient's OpenRouter API for pairwise text comparison.

    The steering eval uses a pairwise comparison protocol where the judge
    receives a formatted prompt comparing two texts and responds with
    "A", "B", or "TIE". This adapter reuses JudgeClient's connection
    and rate-limiting infrastructure for those API calls.

    Includes a retry loop with exponential backoff (±20% jitter) on transient
    HTTP errors (429/500/502/503/504) and network errors, matching the policy
    used by :meth:`JudgeClient.score_pair` (PR #339). A single transient API
    failure must not kill a whole steering-eval run.
    """

    def __init__(self, client: JudgeClient) -> None:
        self._client = client

    def judge(self, prompt: str) -> str:
        """Send a pairwise comparison prompt and return the judge response.

        Retries up to ``_PAIRWISE_JUDGE_MAX_RETRIES`` times on retryable
        errors (HTTP 429/5xx and connection errors) with exponential backoff
        (1s, 2s, 4s with ±20% jitter). Non-retryable HTTP errors (e.g. 401,
        404) and other exceptions are raised immediately on the first attempt.

        Raises
        ------
        requests.HTTPError
            When the API returns a non-retryable status, or when a retryable
            status is returned on every attempt.
        requests.ConnectionError / requests.Timeout
            When the connection fails on every attempt.
        """
        messages = [{"role": "user", "content": prompt}]
        backoff = _PAIRWISE_JUDGE_BACKOFF_BASE_S
        last_exc: Exception | None = None

        for attempt in range(_PAIRWISE_JUDGE_MAX_RETRIES):
            try:
                self._client._rate_limit()
                return self._client._call_api(messages)
            except requests.HTTPError as exc:
                status = (
                    exc.response.status_code if exc.response is not None else None
                )
                if status not in _PAIRWISE_JUDGE_RETRYABLE_STATUS:
                    # Non-retryable: authentication error, bad request, etc.
                    raise
                last_exc = exc
                logger.warning(
                    f"PairwiseJudge HTTP {status} "
                    f"(attempt {attempt + 1}/{_PAIRWISE_JUDGE_MAX_RETRIES})"
                )
            except (requests.ConnectionError, requests.Timeout) as exc:
                last_exc = exc
                logger.warning(
                    f"PairwiseJudge connection error "
                    f"(attempt {attempt + 1}/{_PAIRWISE_JUDGE_MAX_RETRIES}): {exc}"
                )

            if attempt < _PAIRWISE_JUDGE_MAX_RETRIES - 1:
                # ±20% jitter so retries from concurrent clients do not
                # synchronise.
                jitter = random.uniform(0.8, 1.2)
                time.sleep(backoff * jitter)
                backoff *= 2.0

        # All retries exhausted.
        assert last_exc is not None
        logger.error(
            f"PairwiseJudge failed after {_PAIRWISE_JUDGE_MAX_RETRIES} attempts: "
            f"{last_exc}"
        )
        raise last_exc


# ---------------------------------------------------------------------------
# Config validation
# ---------------------------------------------------------------------------


def validate_config_against_model(
    config: SteeringEvalConfig,
    n_sae_features: int,
    num_buckets: int,
) -> None:
    """Validate steering eval config against model encoder parameters.

    Checks that all feature IDs are in ``[0, n_sae_features)`` and all
    magnitude levels are in ``[0, num_buckets)``.

    Parameters
    ----------
    config:
        Parsed steering evaluation config.
    n_sae_features:
        Number of SAE features the encoder supports.
    num_buckets:
        Number of magnitude discretization buckets.

    Raises
    ------
    ValueError
        If any feature ID or magnitude level is out of range.
    """
    for feat in config.features:
        if feat.id < 0 or feat.id >= n_sae_features:
            raise ValueError(
                f"Feature ID {feat.id} is out of range [0, {n_sae_features})"
            )

    mag_levels = (
        config.magnitude_levels
        if config.magnitude_levels is not None
        else list(range(10))
    )
    for level in mag_levels:
        if level < 0 or level >= num_buckets:
            raise ValueError(
                f"Magnitude level {level} is out of range [0, {num_buckets})"
            )
