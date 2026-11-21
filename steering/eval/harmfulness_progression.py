"""Harmfulness progression aggregation + plotting.

Reads ``harmfulness_results*/**/*.yaml`` from a base directory, filters by
``(model_tag, num_harmful, config_name)``, and renders three families of
matplotlib charts:

- ``render_progression_chart`` — one line per signal across conditioned SFT
  branches.
- ``render_standalone_progression`` — cross-lineage chart combining
  d13_base + d12_baseline progression results plus standalone-eval results
  on the same step-proportional x-axis.

Schema contract (input YAMLs):

::

    config:
      model_tag: str
      num_harmful: int
      source: str
      feature_configs: str  # config_label_map key
    harmful_prompts:
      asr: float                  # required
      safe_response_rate: float   # required; score == 1 on harmful pool
    benign_prompts:
      compliance_rate: float      # required; non-refusal
    overall:
      mean_score: float           # required

A YAML file lacking a top-level ``config`` key is treated as "not a
harmfulness result" and silently skipped (so callers can ``rglob`` freely).
A YAML file *with* ``config`` but missing required metric fields raises
``KeyError`` with the file path. There are no silent defaults on REQUIRED
metric fields (ASR, harmful safety, benign compliance, mean score); missing
required fields raise. The loaders DO have non-strict defaults for
filter/sort keys (e.g., ``num_harmful``, ``phase``, ``step``) — files
missing these are filtered out rather than raised on, since some legacy
YAMLs predate those fields.

Renderers return ``matplotlib.figure.Figure``; they never call
``plt.savefig`` themselves. The script driver is responsible for ``savefig``
+ ``plt.close(fig)``.

Example:

>>> from pathlib import Path
>>> from steering.eval.harmfulness_progression import (
...     load_harmfulness_records,
...     render_progression_chart,
...     DEFAULT_CONFIG_LABEL_MAP,
...     DEFAULT_COND_SOURCE_MAP,
...     DEFAULT_STAGES,
...     DEFAULT_SIGNAL_ORDER,
...     DEFAULT_SIGNAL_STYLES,
... )
>>> records = load_harmfulness_records(
...     Path("nanochat_cache"), model_tag="d12_baseline", num_harmful=700,
...     config_label_map=DEFAULT_CONFIG_LABEL_MAP,
... )
>>> fig = render_progression_chart(
...     records, source_to_stage=DEFAULT_COND_SOURCE_MAP,
...     stage_labels=DEFAULT_STAGES,
...     signal_order=DEFAULT_SIGNAL_ORDER,
...     signal_styles=DEFAULT_SIGNAL_STYLES,
...     metric="asr", ylabel="Attack Success Rate",
...     title="d12_baseline cond progression",
... )
>>> fig.savefig("out.png", dpi=150, bbox_inches="tight")
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.figure import Figure

from steering.eval.harmfulness_artifact import HARMFULNESS_EVAL_SCHEMA_VERSION
from steering.eval.harmfulness_generation import (
    STRENGTH_SWEEP_SPEC_SHA256_KEY,
    validate_harmfulness_completion_provenance,
)
from steering.eval.progression_results import SuiteResultKind
from steering.refined_types import LowercaseSha256

# ── types ──────────────────────────────────────────────────────────────────


class HarmfulnessMetric(StrEnum):
    """Metric fields shared by harmfulness records and progression points."""

    ASR = "asr"
    SAFE_RATE = "safe_rate"
    COMPLIANCE_RATE = "compliance_rate"
    MEAN_SCORE = "mean_score"


@dataclass(frozen=True)
class HarmfulnessMetrics:
    """Four explicit metrics from one harmfulness eval run.

    ``safe_rate`` is retained as an internal plotting attribute for API
    compatibility, but now unambiguously means the harmful-prompt score-1 rate.
    """

    asr: float
    mean_score: float
    safe_rate: float
    compliance_rate: float = 0.0


@dataclass(frozen=True)
class HarmfulnessRecord:
    """One harmfulness eval result for (model_tag, source, signal)."""

    model_tag: str
    source: str
    signal: str
    full_conditioning: bool
    metrics: HarmfulnessMetrics


@dataclass(frozen=True)
class ProgressionPoint:
    """One progression eval result point (mean over seeds)."""

    step: int
    phase: str  # "pretrain" | "safety_sft" | "capabilities_sft"
    asr: float
    safe_rate: float
    mean_score: float
    compliance_rate: float = 0.0


@dataclass(frozen=True)
class SignalStyle:
    """Style for one signal across all renderers."""

    label: str
    color: str
    linestyle: str
    linewidth: float
    marker: str
    markersize: float


@dataclass(frozen=True)
class HarmfulnessMetricSpec:
    """Rendering metadata for one :class:`HarmfulnessMetric`."""

    metric: HarmfulnessMetric
    ylabel: str
    suffix: str
    fmt: str


# ── defaults ───────────────────────────────────────────────────────────────


DEFAULT_STAGES: tuple[str, ...] = (
    "Base (pretrain)",
    "Safety SFT",
    "Capabilities SFT",
)

DENSE_SIGNAL_COLORS = ("#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b")
SELF_CONDITIONED_CONFIG = SuiteResultKind.SELF_CONDITIONED.value

DEFAULT_COND_SOURCE_MAP: Mapping[str, int] = {
    "base": 0,
    "base/safety_cond": 1,
    "base/safety_cond/capabilities_cond": 2,
}

DEFAULT_REVERSED_COND_SOURCE_MAP: Mapping[str, int] = {
    "base": 0,
    "base/capabilities_cond": 1,
    "base/capabilities_cond/safety_cond": 2,
}

DEFAULT_REVERSED_STAGES: tuple[str, ...] = (
    "Base (pretrain)",
    "Capabilities SFT",
    "Safety SFT",
)

DEFAULT_SIGNAL_ORDER: tuple[str, ...] = (
    "Safety p90",
    "Safety mean-k standalone",
    "Safety mean-k pretrain-ref",
    "Safety delta-base",
    "Capability p90",
    "Capability mean-k standalone",
    "Capability mean-k pretrain-ref",
    "Capability delta-base",
)


def _build_default_signal_styles() -> Mapping[str, SignalStyle]:
    colors = {
        "Safety p90": "#1f77b4",
        "Safety mean-k standalone": "#4a9bd9",
        "Safety mean-k pretrain-ref": "#7ec8e3",
        "Safety delta-base": "#0b3d91",
        "Capability p90": "#d62728",
        "Capability mean-k standalone": "#e77c7c",
        "Capability mean-k pretrain-ref": "#f0a8a8",
        "Capability delta-base": "#8b0000",
    }
    styles_extra = {
        "Safety p90": ("-", 1.5, "o", 6.0),
        "Safety mean-k standalone": ("--", 1.5, "o", 6.0),
        "Safety mean-k pretrain-ref": (":", 1.5, "o", 6.0),
        "Safety delta-base": ("-.", 1.5, "D", 6.0),
        "Capability p90": ("-", 1.5, "^", 6.0),
        "Capability mean-k standalone": ("--", 1.5, "^", 6.0),
        "Capability mean-k pretrain-ref": (":", 1.5, "^", 6.0),
        "Capability delta-base": ("-.", 1.5, "v", 6.0),
    }
    return {
        sig: SignalStyle(
            label=sig,
            color=colors[sig],
            linestyle=ls,
            linewidth=lw,
            marker=mk,
            markersize=ms,
        )
        for sig, (ls, lw, mk, ms) in styles_extra.items()
    }


DEFAULT_SIGNAL_STYLES: Mapping[str, SignalStyle] = _build_default_signal_styles()


def _build_default_config_label_map() -> Mapping[str, str]:
    labels: dict[str, str] = {}
    for phase in ("safety_sft", "capabilities_sft"):
        prefix = "Safety" if phase.startswith("safety") else "Capability"
        labels[f"{phase}/f0.9/m0.5"] = f"{prefix} p90"
        labels[f"{phase}/geometric_m10_min1_max40_d12_seed1234/standalone"] = (
            f"{prefix} mean-k standalone"
        )
        labels[f"{phase}/geometric_m10_min1_max40_d12_seed1234/pretrain_ref_local_l8_1shard"] = (
            f"{prefix} mean-k pretrain-ref"
        )
        labels[
            f"{phase}/delta_base/dist_geometric/mean_10.0/std_10.0/min_1/"
            "max_40/neg_dist_geometric/mean_10.0/std_10.0/min_0/max_40/"
            "seed_42/r_mean_pct_incl_zeros/m0.5"
        ] = f"{prefix} delta-base"
    return labels


DEFAULT_CONFIG_LABEL_MAP: Mapping[str, str] = _build_default_config_label_map()

DEFAULT_SFT_STEPS: Mapping[str, tuple[int, int]] = {
    "d13_base": (904, 1140),
    "cond": (886, 1106),
}

METRIC_SPECS: tuple[HarmfulnessMetricSpec, ...] = (
    HarmfulnessMetricSpec(
        metric=HarmfulnessMetric.ASR,
        ylabel="Attack Success Rate",
        suffix="asr",
        fmt=".1%",
    ),
    HarmfulnessMetricSpec(
        metric=HarmfulnessMetric.SAFE_RATE,
        ylabel="Harmful-Prompt Safe Response Rate (score = 1)",
        suffix="harmful_safe_response_rate",
        fmt=".1%",
    ),
    HarmfulnessMetricSpec(
        metric=HarmfulnessMetric.COMPLIANCE_RATE,
        ylabel="Benign Compliance Rate (non-refusal)",
        suffix="benign_compliance_rate",
        fmt=".1%",
    ),
    HarmfulnessMetricSpec(
        metric=HarmfulnessMetric.MEAN_SCORE,
        ylabel="Mean Judge Score (1–5)",
        suffix="mean_score",
        fmt=".2f",
    ),
)

# ── loaders ────────────────────────────────────────────────────────────────


def _req(d: Mapping[str, Any], key: str, full_path: str, path: Path) -> Any:
    if key not in d:
        raise KeyError(f"required field {full_path!r} missing from {path}")
    return d[key]


def _bounded_float(
    value: Any,
    *,
    full_path: str,
    source_path: Path,
    minimum: float,
    maximum: float,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{source_path}: {full_path} must be numeric, got {value!r}")
    numeric = float(value)
    if not math.isfinite(numeric) or not minimum <= numeric <= maximum:
        raise ValueError(
            f"{source_path}: {full_path} must be finite and in "
            f"[{minimum}, {maximum}], got {value!r}"
        )
    return numeric


def _parse_harmfulness_metrics(data: Mapping[str, Any], path: Path) -> HarmfulnessMetrics:
    """Use the same required fields and numeric bounds for YAML runs and JSON seeds."""

    def read(section: str, key: str, minimum: float = 0.0, maximum: float = 1.0) -> float:
        block = _req(data, section, section, path)
        field = f"{section}.{key}"
        return _bounded_float(
            _req(block, key, field, path),
            full_path=field, source_path=path, minimum=minimum, maximum=maximum,
        )

    return HarmfulnessMetrics(
        asr=read("harmful_prompts", "asr"),
        mean_score=read("overall", "mean_score", 1.0, 5.0),
        safe_rate=read("harmful_prompts", "safe_response_rate"),
        compliance_rate=read("benign_prompts", "compliance_rate"),
    )


def discover_harmfulness_yaml_files(base_dir: Path) -> list[Path]:
    """Return sorted YAML paths under base_dir that look like harmfulness
    results, excluding ``*_responses*.yaml`` (which are full-text dumps,
    not metric summaries)."""
    return sorted(
        p for p in base_dir.rglob("harmfulness_results*/**/*.yaml") if "_responses" not in str(p)
    )


def _parse_harmfulness_record(data: Any, path: Path) -> HarmfulnessRecord | None:
    """Parse decoded YAML, retaining its path in schema/metric errors.

    Non-results (no config) are skipped. Results require every metric and a
    current schema; a non-empty judge run with no successful results fails.
    """
    if not isinstance(data, dict):
        return None
    cfg = data.get("config")
    if cfg is None:
        return None
    schema_version = data.get("harmfulness_eval_schema_version")
    if schema_version != HARMFULNESS_EVAL_SCHEMA_VERSION:
        raise ValueError(
            f"{path}: harmfulness_eval_schema_version must equal "
            f"{HARMFULNESS_EVAL_SCHEMA_VERSION}, got {schema_version!r}"
        )

    model_tag = str(_req(cfg, "model_tag", "config.model_tag", path))
    source = str(_req(cfg, "source", "config.source", path))
    config_name = _req(cfg, "feature_configs", "config.feature_configs", path)

    hp = _req(data, "harmful_prompts", "harmful_prompts", path)
    bp = _req(data, "benign_prompts", "benign_prompts", path)
    overall = _req(data, "overall", "overall", path)
    if not isinstance(hp, dict) or not isinstance(bp, dict) or not isinstance(overall, dict):
        raise KeyError(
            f"expected dict at harmful_prompts, benign_prompts, and overall "
            f"in {path}; got {type(hp).__name__}, {type(bp).__name__}, "
            f"and {type(overall).__name__}"
        )

    num_scored = overall.get("num_scored")
    num_failed = overall.get("num_failed")
    if num_scored == 0 and num_failed is not None and num_failed > 0:
        raise ValueError(
            f"{path}: harmfulness result has no successful judge results ({num_failed} failed)"
        )

    return HarmfulnessRecord(
        model_tag=model_tag,
        source=source,
        signal=str(config_name),  # raw config name; caller maps via label map
        full_conditioning="full_conditioning" in str(path),
        metrics=_parse_harmfulness_metrics(data, path),
    )


def load_harmfulness_records(
    base_dir: Path,
    *,
    model_tag: str,
    num_harmful: int,
    config_label_map: Mapping[str, str],
    safety_steering: Mapping[str, Any] | None = None,
) -> list[HarmfulnessRecord]:
    """Discover, parse, filter, and label harmfulness eval results.

    Filtering: a record is included iff:
      - the file has the harmfulness schema,
      - the file's ``config.model_tag`` equals the requested ``model_tag``,
      - the file's ``config.num_harmful`` equals the requested
        ``num_harmful``,
      - the file's ``config.feature_configs`` is a key in
        ``config_label_map``.

    The returned record's ``signal`` is the *human label* from
    ``config_label_map`` (not the raw ``feature_configs`` string).

    Schema files matching the rglob but missing required metric fields
    raise. Filter mismatches are exclusions,
    not raises.
    """
    if safety_steering is not None:
        _validate_safety_plot_receipt(safety_steering)
    paths = discover_harmfulness_yaml_files(base_dir)
    out: list[HarmfulnessRecord] = []
    safety_targets_by_cell: dict[tuple[str, str], tuple[str, Path]] = {}
    for p in paths:
        with open(p) as f:
            data = yaml.safe_load(f)
        rec = _parse_harmfulness_record(data, p)
        if rec is None:
            continue  # not a harmfulness result
        if rec.model_tag != model_tag:
            continue
        n_harm = int(
            data["config"].get("num_harmful", 0)
        )  # filter key, not a metric — file w/o this gets filtered out
        if n_harm != num_harmful:
            continue
        generation = data["config"].get("generation")
        if generation is not None and not isinstance(generation, Mapping):
            raise ValueError(f"{p}: config.generation must be a mapping or null")
        completion = generation.get("completion") if generation is not None else None
        if completion is not None:
            validate_harmfulness_completion_provenance(completion)
        receipt = completion.get("safety_steering") if completion is not None else None
        if receipt is not None:
            label = _safety_signal_label(receipt)
        elif completion is not None:
            label = "Self-conditioned"
        else:
            label = config_label_map.get(rec.signal)
        if safety_steering is not None:
            if receipt != dict(safety_steering):
                continue
        if label is None:
            continue  # signal not in caller's selection
        if receipt is not None:
            cell = (rec.source, label)
            digest = receipt["targets_sha256"]
            if cell in safety_targets_by_cell:
                previous_digest, previous_path = safety_targets_by_cell[cell]
                if previous_digest != digest:
                    raise ValueError(
                        f"Conflicting safety targets for plot cell source={rec.source!r}, "
                        f"signal={label!r}: {previous_path} ({previous_digest}) and "
                        f"{p} ({digest}). Select an exact safety_steering receipt filter "
                        "or use separate result roots."
                    )
                raise ValueError(
                    f"Duplicate safety results for plot cell source={rec.source!r}, "
                    f"signal={label!r}: {previous_path} and {p}. "
                    "Use separate result roots or aggregate progression seeds explicitly."
                )
            safety_targets_by_cell[cell] = (digest, p)
        # Replace rec.signal (raw config_name) with the human label.
        out.append(replace(rec, signal=label))
    return out


def _validate_safety_plot_receipt(receipt: Any) -> Mapping[str, Any]:
    if not isinstance(receipt, Mapping) or set(receipt) != {"targets_sha256", "strength"}:
        raise ValueError("safety_steering plot receipt requires targets_sha256 and strength")
    LowercaseSha256.parse(receipt["targets_sha256"])
    strength = receipt["strength"]
    if type(strength) not in (int, float) or not 0 <= strength <= 6:
        raise ValueError("safety_steering strength must be finite and within [0, 6]")
    return receipt


def _safety_signal_label(receipt: Mapping[str, Any]) -> str:
    # The default :g precision merges distinct strengths into one plotted cell.
    strength = float(receipt["strength"])
    rendered = "0" if strength == 0 else str(strength).removesuffix(".0")
    return f"Safety interpolation (strength={rendered})"


def harmfulness_plot_signals(
    records: Sequence[HarmfulnessRecord],
) -> tuple[tuple[str, ...], Mapping[str, SignalStyle]]:
    """Keep legacy styles and add every observed dense-interpolation signal."""
    extra = tuple(sorted({record.signal for record in records} - set(DEFAULT_SIGNAL_ORDER)))
    styles = dict(DEFAULT_SIGNAL_STYLES)
    for index, signal in enumerate(extra):
        styles[signal] = SignalStyle(signal, DENSE_SIGNAL_COLORS[index % len(DENSE_SIGNAL_COLORS)], "-", 1.5, "o", 6.0)
    return DEFAULT_SIGNAL_ORDER + extra, styles


def load_progression_data(
    path: Path,
    *,
    min_step: int = 53000,
    conditioned_config: str | None = None,
) -> list[ProgressionPoint]:
    """Load mean-across-seeds progression points from a results.json.

    Requires issue #1338's nested per-seed result shape. Legacy flat seeds are
    rejected because their ``safe_response_rate`` was the overall-pool value
    and cannot be losslessly reinterpreted as harmful-prompt safety.

    Pretrain points with step < min_step are excluded. SFT phases
    (``safety_sft``, ``capabilities_sft``) are not subject to min_step.

    Returns sorted by (phase_order, step).
    """
    with open(path) as f:
        return _parse_progression_data(
            json.load(f), path, min_step=min_step, conditioned_config=conditioned_config
        )


def _parse_progression_data(
    data: Mapping[str, Any], path: Path, *, min_step: int, conditioned_config: str | None,
) -> list[ProgressionPoint]:
    phase_order = {"pretrain": 0, "safety_sft": 1, "capabilities_sft": 2}
    out: list[ProgressionPoint] = []

    for ckpt in sorted(
        data.get("checkpoints", []),  # filter key, not a metric — file w/o this gets filtered out
        key=lambda c: (
            phase_order.get(c.get("phase", ""), 99),
            c.get("step", 0),
        ),  # sort keys, not metrics — checkpoints w/o these get sorted last and filtered out
    ):
        phase = ckpt.get(
            "phase", ""
        )  # filter key, not a metric — checkpoint w/o this gets filtered out
        step = int(
            ckpt.get("step", 0)
        )  # filter key, not a metric — checkpoint w/o this gets filtered out
        if phase == "pretrain" and step < min_step:
            continue

        h = ckpt.get("evals", {}).get("harmfulness")
        if h is None:
            continue
        if not isinstance(h, dict):
            raise ValueError(
                f"{path}: checkpoint phase={phase} step={step} has "
                f"non-dict harmfulness evals: {type(h).__name__}"
            )

        # Empty harmfulness dict: skip silently — no run yet.
        if not h:
            continue

        if conditioned_config is not None:
            conditioned = h.get("conditioned")
            configs = conditioned.get("configs") if isinstance(conditioned, dict) else None
            if not isinstance(configs, dict):
                raise ValueError(f"{path}: conditioned harmfulness requires a configs mapping")
            if conditioned_config not in configs:
                raise ValueError(f"{path}: missing conditioned config {conditioned_config!r}")
            selected = configs[conditioned_config]
            if not isinstance(selected, dict) or "seeds" not in selected:
                raise ValueError(f"{path}: conditioned config {conditioned_config!r} requires seeds")
            seeds = selected["seeds"]
            if not isinstance(seeds, dict):
                raise ValueError(f"{path}: conditioned config {conditioned_config!r} seeds must be a mapping")
        elif "seeds" in h:
            seeds = h["seeds"]
        else:
            raise ValueError(
                f"{path}: checkpoint phase={phase} step={step} harmfulness "
                f"evals contains no 'seeds' mapping: "
                f"keys={sorted(h.keys())}"
            )

        if not isinstance(seeds, dict) or not seeds:
            continue  # no seeds run yet for this checkpoint

        parsed_seeds: list[HarmfulnessMetrics] = []
        for seed_id, seed_data in seeds.items():
            if not isinstance(seed_data, dict):
                raise ValueError(
                    f"{path}: checkpoint phase={phase} step={step} seed "
                    f"{seed_id!r}: expected dict, got "
                    f"{type(seed_data).__name__}"
                )
            schema_version = seed_data.get("harmfulness_eval_schema_version")
            if schema_version != HARMFULNESS_EVAL_SCHEMA_VERSION:
                raise ValueError(
                    f"{path}: checkpoint phase={phase} step={step} seed "
                    f"{seed_id!r}: harmfulness_eval_schema_version must equal "
                    f"{HARMFULNESS_EVAL_SCHEMA_VERSION}, got {schema_version!r}"
                )
            legacy_fields = {"asr", "safe_response_rate", "mean_score"}
            if legacy_fields & set(seed_data):
                raise ValueError(
                    f"{path}: checkpoint phase={phase} step={step} seed "
                    f"{seed_id!r}: legacy flat harmfulness seed cannot be "
                    f"interpreted under schema version {HARMFULNESS_EVAL_SCHEMA_VERSION}"
                )
            try:
                parsed_seeds.append(_parse_harmfulness_metrics(seed_data, path))
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(
                    f"{path}: checkpoint phase={phase} step={step} seed "
                    f"{seed_id!r}: invalid nested harmfulness schema: {exc}"
                ) from exc

        out.append(
            ProgressionPoint(
                step=step,
                phase=phase,
                **{
                    metric.value: float(np.mean([getattr(seed, metric) for seed in parsed_seeds]))
                    for metric in HarmfulnessMetric
                },
            )
        )

    return out


def load_safety_progression_runs(
    paths: Sequence[Path], *, model_tag: str,
) -> tuple[list[HarmfulnessRecord], Mapping[str, int]]:
    """Combine separately collected strengths, preserving each checkpoint receipt.

    Each input is a training_progression results.json for the same checkpoints.
    Metrics retain load_progression_data's schema and mean-across-seeds semantics.
    """
    records: list[HarmfulnessRecord] = []
    stages: dict[str, int] = {}
    seen: set[tuple[str, str]] = set()
    targets_by_checkpoint: dict[tuple[str, int], str] = {}
    comparisons_by_checkpoint: dict[tuple[str, int], dict[str, Any]] = {}
    reference_checkpoints = None
    if not paths:
        raise ValueError("strength sweep requires at least one progression file")
    for path in paths:
        data = json.loads(path.read_text())
        if data.get("model_tag") != model_tag:
            raise ValueError(f"{path}: strength sweep model_tag disagrees with {model_tag!r}")
        points = _parse_progression_data(
            data, path, min_step=0, conditioned_config=SELF_CONDITIONED_CONFIG
        )
        if not points:
            raise ValueError(f"{path}: strength sweep has no judged self-conditioned points")
        checkpoints = {}
        for checkpoint in data["checkpoints"]:
            key = (checkpoint["phase"], checkpoint["step"])
            if key in checkpoints:
                raise ValueError(f"{path}: duplicate progression checkpoint {key}")
            checkpoints[key] = checkpoint
        identities = {(point.phase, point.step): checkpoints[(point.phase, point.step)].get("checkpoint_identity")
                      for point in points}
        if any(not isinstance(identity, dict) or not identity for identity in identities.values()):
            raise ValueError(f"{path}: strength sweep requires checkpoint_identity for every judged point")
        if reference_checkpoints is not None and identities != reference_checkpoints:
            raise ValueError(f"{path}: strength sweep checkpoint identities or cells disagree")
        reference_checkpoints = identities
        run_strength = None
        for point in points:
            checkpoint = checkpoints[(point.phase, point.step)]
            seeds = checkpoint["evals"]["harmfulness"]["conditioned"]["configs"][SELF_CONDITIONED_CONFIG]["seeds"]
            receipts = []
            comparisons = {}
            for seed_id, seed in seeds.items():
                completion = validate_harmfulness_completion_provenance(seed.get("completion"))
                receipt = _validate_safety_plot_receipt(completion.get("safety_steering"))
                receipts.append(receipt)
                provenance = seed.get("progression_provenance")
                digest = provenance.get(STRENGTH_SWEEP_SPEC_SHA256_KEY) if isinstance(provenance, Mapping) else None
                try:
                    LowercaseSha256.parse(digest)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"{path}: strength sweep requires {STRENGTH_SWEEP_SPEC_SHA256_KEY} "
                        f"for seed {seed_id!r}; recollect this progression with the current evaluator"
                    ) from exc
                comparisons[seed_id] = {
                    "spec": digest,
                    "completion": {key: value for key, value in completion.items()
                                   if key not in {"strength", "safety_steering"}},
                }
            if any(receipt != receipts[0] for receipt in receipts):
                raise ValueError(f"{path}: strength sweep seeds disagree on safety_steering receipt")
            receipt = receipts[0]
            if run_strength is not None and receipt["strength"] != run_strength:
                raise ValueError(f"{path}: each strength sweep file must use one strength for all checkpoints")
            run_strength = receipt["strength"]
            checkpoint_key = (point.phase, point.step)
            if comparisons_by_checkpoint.setdefault(checkpoint_key, comparisons) != comparisons:
                raise ValueError(
                    f"{path}: strength sweep seeds or evaluation settings disagree "
                    f"for checkpoint {checkpoint_key}"
                )
            if targets_by_checkpoint.setdefault(checkpoint_key, receipt["targets_sha256"]) != receipt["targets_sha256"]:
                raise ValueError(f"{path}: strength sweep safety targets disagree for checkpoint {checkpoint_key}")
            source = f"{point.phase} / {point.step}"
            label = _safety_signal_label(receipt)
            cell = (source, label)
            if cell in seen:
                raise ValueError(f"{path}: duplicate strength sweep plot cell {cell}")
            seen.add(cell)
            stages.setdefault(source, len(stages))
            records.append(HarmfulnessRecord(model_tag=model_tag, source=source, signal=label,
                full_conditioning=False, metrics=HarmfulnessMetrics(asr=point.asr, mean_score=point.mean_score,
                    safe_rate=point.safe_rate, compliance_rate=point.compliance_rate)))
    return records, stages


# ── renderers ──────────────────────────────────────────────────────────────


def _records_indexed(
    records: Sequence[HarmfulnessRecord],
) -> Mapping[tuple[str, str], HarmfulnessMetrics]:
    """Index by (source, signal) → metrics for fast lookup. Last record
    wins on duplicate keys (caller's responsibility to dedupe upstream)."""
    return {(r.source, r.signal): r.metrics for r in records}


def _format_value(value: float, fmt: str) -> str:
    return format(value, fmt)


def render_progression_chart(
    records: Sequence[HarmfulnessRecord],
    *,
    source_to_stage: Mapping[str, int],
    stage_labels: Sequence[str],
    signal_order: Sequence[str],
    signal_styles: Mapping[str, SignalStyle],
    metric: HarmfulnessMetric,
    ylabel: str,
    title: str,
    fmt: str = ".1%",
    invert_y: bool = False,
) -> Figure:
    """Render a one-line-per-signal progression chart.

    For each signal in ``signal_order``, plot one line whose y-values are
    the chosen ``metric`` indexed by ``source_to_stage[record.source]``.
    Signals with no data points across any source are omitted entirely
    (no empty line, no legend entry).

    Returns a fresh Figure; does not call savefig.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    n_stages = len(stage_labels)
    x_positions = np.arange(n_stages)

    indexed = _records_indexed(records)

    for signal in signal_order:
        # Walk stages in order; pick metric value if (source, signal) exists.
        ys: list[float | None] = [None] * n_stages
        for source, stage_idx in source_to_stage.items():
            if 0 <= stage_idx < n_stages:
                key = (source, signal)
                if key in indexed:
                    ys[stage_idx] = float(getattr(indexed[key], metric))

        # Skip signals with no data at all.
        plotted = [(i, y) for i, y in enumerate(ys) if y is not None]
        if not plotted:
            continue

        xs = [i for i, _ in plotted]
        ys_real = [y for _, y in plotted]

        style = signal_styles.get(signal)
        color = style.color if style is not None else "#999999"
        ls = style.linestyle if style is not None else "-"
        lw = style.linewidth if style is not None else 1.5
        marker = style.marker if style is not None else "o"
        ms = style.markersize if style is not None else 6.0

        ax.plot(
            xs,
            ys_real,
            color=color,
            linestyle=ls,
            linewidth=lw,
            marker=marker,
            markersize=ms,
            label=signal,
        )

        # Endpoint annotations.
        for xi, yi in zip(xs, ys_real, strict=True):
            ax.annotate(
                _format_value(yi, fmt),
                (xi, yi),
                textcoords="offset points",
                xytext=(0, 10),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xticks(x_positions)
    ax.set_xticklabels(list(stage_labels), fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=13)

    # Phase transition vertical lines between every consecutive pair of stages.
    for i in range(1, n_stages):
        ax.axvline(x=i - 0.5, color="gray", linestyle="--", alpha=0.3)

    ax.legend(
        bbox_to_anchor=(1.02, 1),
        loc="upper left",
        fontsize=8,
        framealpha=0.9,
    )
    ax.grid(axis="y", alpha=0.2)

    if invert_y:
        ax.invert_yaxis()

    fig.tight_layout()
    return fig


def _signal_metric_for_source(
    indexed: Mapping[tuple[str, str], HarmfulnessMetrics],
    *,
    source: str,
    signal: str,
    metric: HarmfulnessMetric,
) -> float | None:
    key = (source, signal)
    if key not in indexed:
        return None
    return float(getattr(indexed[key], metric))


def render_standalone_progression(
    records: Sequence[HarmfulnessRecord],
    d13_progression: Sequence[ProgressionPoint],
    d12_progression: Sequence[ProgressionPoint],
    *,
    metric: HarmfulnessMetric,
    ylabel: str,
    title: str,
    last_pretrain_step: int,
    sft_steps: Mapping[str, tuple[int, int]],
    signal_order: Sequence[str],
    signal_styles: Mapping[str, SignalStyle],
) -> Figure:
    """Cross-lineage chart combining d13_base + d12_baseline progression
    plus standalone-eval branches on a step-proportional x-axis."""
    fig, ax = plt.subplots(figsize=(14, 7))

    # ── d13_base decoder-only progression line ──
    d13_pretrain = [pt for pt in d13_progression if pt.phase == "pretrain"]
    d13_safety = [pt for pt in d13_progression if pt.phase == "safety_sft"]
    d13_cap = [pt for pt in d13_progression if pt.phase == "capabilities_sft"]

    d13_x = [pt.step for pt in d13_pretrain]
    d13_y = [getattr(pt, metric) for pt in d13_pretrain]
    if d13_safety:
        d13_x.append(last_pretrain_step + sft_steps["d13_base"][0])
        d13_y.append(getattr(d13_safety[0], metric))
    if d13_cap:
        d13_x.append(last_pretrain_step + sft_steps["d13_base"][0] + sft_steps["d13_base"][1])
        d13_y.append(getattr(d13_cap[0], metric))

    if d13_x:
        ax.plot(
            d13_x,
            d13_y,
            color="#888888",
            linestyle="--",
            linewidth=2,
            marker="D",
            markersize=6,
            zorder=5,
            label="d13_base (decoder-only)",
        )

    # ── d12_baseline progression (encoder-decoder, pretrain only) ──
    d12_pretrain = [pt for pt in d12_progression if pt.phase == "pretrain"]
    d12_x = [pt.step for pt in d12_pretrain]
    d12_y = [getattr(pt, metric) for pt in d12_pretrain]

    if d12_x:
        ax.plot(
            d12_x,
            d12_y,
            color="#2ca02c",
            linestyle="-",
            linewidth=2,
            marker="o",
            markersize=6,
            zorder=5,
            label="d12_baseline (encdec pretrain)",
        )

    # ── SFT branches (standalone-eval results) anchored at last_pretrain_step ──
    indexed = _records_indexed(records)

    pretrain_anchor = getattr(d12_pretrain[-1], metric) if d12_pretrain else None

    branches: list[tuple[str, str, str, str, float, Sequence[str]]] = [
        # (safety_src, cap_src, lineage, label, alpha, signals_to_plot)
        (
            "base/safety_cond",
            "base/safety_cond/capabilities_cond",
            "cond",
            "cond SFT",
            0.85,
            list(signal_order),
        ),
    ]

    for safety_src, cap_src, lineage, label, alpha, signals in branches:
        ls = "-" if lineage == "cond" else "--"
        lw = 1.8 if lineage == "cond" else 1.5
        for signal in signals:
            style = signal_styles.get(signal)
            color = style.color if style is not None else "#999999"
            marker = "s" if "Safety" in signal else ("^" if "Capability" in signal else "o")

            safety_val = _signal_metric_for_source(
                indexed, source=safety_src, signal=signal, metric=metric
            )
            cap_val = _signal_metric_for_source(
                indexed, source=cap_src, signal=signal, metric=metric
            )
            if safety_val is None and cap_val is None:
                continue

            xs: list[float] = []
            ys: list[float] = []
            if pretrain_anchor is not None:
                xs.append(float(last_pretrain_step))
                ys.append(pretrain_anchor)
            if safety_val is not None:
                xs.append(float(last_pretrain_step + sft_steps[lineage][0]))
                ys.append(safety_val)
            if cap_val is not None:
                xs.append(float(last_pretrain_step + sft_steps[lineage][0] + sft_steps[lineage][1]))
                ys.append(cap_val)
            if len(xs) < 2:
                continue

            ax.plot(
                xs,
                ys,
                color=color,
                linestyle=ls,
                linewidth=lw,
                marker=marker,
                markersize=5,
                alpha=alpha,
                zorder=4,
                label=f"{label} — {signal}",
            )

    # Phase-transition vline at last_pretrain_step.
    ax.axvline(x=last_pretrain_step, color="gray", linestyle=":", alpha=0.4, linewidth=1.5)

    ax.set_xlabel("Training step", fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12)
    ax.grid(axis="y", alpha=0.2)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=7, framealpha=0.9)
    fig.tight_layout()
    return fig
