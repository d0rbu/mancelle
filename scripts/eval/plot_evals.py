"""Eval visualization: auto-crawl checkpoint trees and generate comparison charts.

Usage:
    uv run python -m scripts.eval.plot_evals [base_dir=/path/to/checkpoints] [output_dir=fig]
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from hydra import compose, initialize_config_dir
from loguru import logger
from matplotlib.figure import Figure

from nanochat.common import get_base_dir
from steering.eval.harmfulness_artifact import HARMFULNESS_EVAL_SCHEMA_VERSION
from steering.eval.progression_results import (
    SuiteResultVariant,
    load_suite_from_dict,
)
from steering.eval.result_plot_dispatch import _get_linestyle_for_variant

_CONFIG_DIR = str(Path(__file__).resolve().parent.parent.parent / "configs" / "eval")
_CONFIG_NAME = "plot_evals"


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


def parse_base_eval_csv(csv_path: Path) -> dict[str, Any]:
    """Parse a base eval CSV file into a structured dict.

    Returns:
        {"results": {task_name: accuracy, ...}, "core_metric": float}

    Raises:
        FileNotFoundError: If csv_path does not exist.
        ValueError: If the CORE metric row is missing.
    """
    csv_path = Path(csv_path)
    if not csv_path.exists():
        raise FileNotFoundError(f"Base eval CSV not found: {csv_path}")

    results: dict[str, float] = {}
    core_metric: float | None = None

    with open(csv_path, encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines[1:]:  # skip header
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 3:
            continue
        task = parts[0].strip()
        acc_str = parts[1].strip()
        centered_str = parts[2].strip()

        if task == "CORE":
            core_metric = float(centered_str)
        elif acc_str:
            acc = float(acc_str)
            if not 0.0 <= acc <= 1.0:
                raise ValueError(f"Accuracy {acc} out of [0,1] for task {task}")
            results[task] = acc

    if core_metric is None:
        raise ValueError(f"CORE metric row not found in {csv_path}")

    return {"results": results, "core_metric": core_metric}


def _parse_single_harmfulness_yaml(yaml_path: Path) -> dict[str, Any]:
    """Parse one harmfulness results YAML file."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    schema_version = data.get("harmfulness_eval_schema_version")
    if schema_version != HARMFULNESS_EVAL_SCHEMA_VERSION:
        raise ValueError(
            f"{yaml_path}: harmfulness_eval_schema_version must equal "
            f"{HARMFULNESS_EVAL_SCHEMA_VERSION}, got {schema_version!r}"
        )
    for required in ("overall", "harmful_prompts", "benign_prompts", "per_category"):
        if required not in data:
            raise ValueError(f"Missing required key '{required}' in {yaml_path}")

    overall = data["overall"]
    harmful = data["harmful_prompts"]
    benign = data["benign_prompts"]
    if not all(isinstance(section, dict) for section in (overall, harmful, benign)):
        raise ValueError(f"Harmfulness metric sections must be mappings in {yaml_path}")
    if not 1.0 <= overall["mean_score"] <= 5.0:
        raise ValueError(f"mean_score {overall['mean_score']} out of [1, 5]")
    for rate_key in ("harmfulness_rate", "safe_response_rate"):
        if not 0.0 <= overall[rate_key] <= 1.0:
            raise ValueError(f"{rate_key} {overall[rate_key]} out of [0, 1]")
    for metric_path, rate in (
        (
            "harmful_prompts.safe_response_rate",
            harmful.get("safe_response_rate"),
        ),
        ("benign_prompts.compliance_rate", benign.get("compliance_rate")),
    ):
        if isinstance(rate, bool) or not isinstance(rate, (int, float)):
            raise ValueError(f"Missing or non-numeric {metric_path} in {yaml_path}")
        if not 0.0 <= float(rate) <= 1.0:
            raise ValueError(f"{metric_path} {rate} out of [0, 1]")

    if len(data["per_category"]) != 19:
        raise ValueError(
            f"Expected 19 harm categories, got {len(data['per_category'])}"
        )

    return {
        "overall": data["overall"],
        "harmful_prompts": data["harmful_prompts"],
        "benign_prompts": data["benign_prompts"],
        "per_category": data["per_category"],
    }


def parse_harmfulness_results(checkpoint_dir: Path) -> dict[str, dict[str, Any]]:
    """Parse harmfulness eval results from a checkpoint directory.

    Results live under
    ``report/harmfulness_results/{label}.yaml`` with optional per-condition
    subdirectories.

    Returns mapping from condition label to parsed metrics dict. Empty dict if no results.
    """
    checkpoint_dir = Path(checkpoint_dir)
    report_dir = checkpoint_dir / "report"

    # Per-condition directory (supports nested paths like f0.5/m0.5.yaml).
    condition_dir = report_dir / "harmfulness_results"
    if condition_dir.is_dir():
        results: dict[str, dict[str, Any]] = {}
        for yaml_file in sorted(condition_dir.rglob("*.yaml")):
            # Derive label from relative path: f0.5/m0.5.yaml -> "f0.5/m0.5"
            rel = yaml_file.relative_to(condition_dir)
            label = str(rel.with_suffix(""))
            results[label] = _parse_single_harmfulness_yaml(yaml_file)
        return results

    flat_file = report_dir / "harmfulness_results.yaml"
    if flat_file.exists():
        raise ValueError(
            f"Unsupported flat harmfulness result {flat_file}; rerun "
            "harmfulness_eval to produce report/harmfulness_results/<config>.yaml"
        )

    return {}


def parse_steering_report(report_path: Path) -> dict[str, Any]:
    """Parse a steering eval report.yaml into a structured dict.

    Returns:
        {"aggregate": {...}, "features": [{id, confidence, standard_win_rate, dose_response}, ...]}

    Raises:
        FileNotFoundError: If report_path does not exist.
    """
    report_path = Path(report_path)
    if not report_path.exists():
        raise FileNotFoundError(f"Steering report not found: {report_path}")

    with open(report_path) as f:
        data = yaml.safe_load(f)

    if not isinstance(data, dict):
        raise ValueError(f"Expected dict, got {type(data).__name__}")
    if "aggregate" not in data:
        raise ValueError(f"Missing 'aggregate' in {report_path}")
    if "features" not in data:
        raise ValueError(f"Missing 'features' in {report_path}")

    # Validate features
    features: list[dict[str, Any]] = []
    for feat in data["features"]:
        dr = feat.get("dose_response", {})
        levels = dr.get("magnitude_levels", [])
        win_rates = dr.get("win_rates", [])
        if len(levels) != len(win_rates):
            raise ValueError(
                f"Feature {feat['id']}: magnitude_levels ({len(levels)}) != win_rates ({len(win_rates)})"
            )
        for wr in win_rates:
            if not 0.0 <= wr <= 1.0:
                raise ValueError(f"Feature {feat['id']}: win_rate {wr} out of [0, 1]")
        features.append(
            {
                "id": feat["id"],
                "confidence": feat["confidence"],
                "standard_win_rate": feat["standard_win_rate"],
                "dose_response": dr,
            }
        )

    return {"aggregate": data["aggregate"], "features": features}


# ---------------------------------------------------------------------------
# Tree walker
# ---------------------------------------------------------------------------


@dataclass
class ModelNode:
    """A node in the model checkpoint tree."""

    source_chain: str
    model_tag: str
    checkpoint_dir: Path
    eval_results: dict[str, Any] = field(default_factory=dict)
    children: list[ModelNode] = field(default_factory=list)


# Matches nanochat/checkpoint_manager.py:339-345
_SEGMENT_DIRS: dict[str, str] = {
    "sft": "standard_sft_checkpoints",
    "standard": "standard_sft_checkpoints",
    "safety": "safety_sft_checkpoints",
    "capabilities": "capabilities_sft_checkpoints",
    "rl": "standard_rl_checkpoints",
}

# Reverse mapping: directory name -> segment name (pick canonical names, skip aliases)
_DIR_TO_SEGMENT: dict[str, str] = {
    "standard_sft_checkpoints": "standard",
    "safety_sft_checkpoints": "safety",
    "capabilities_sft_checkpoints": "capabilities",
    "standard_rl_checkpoints": "rl",
}


def _probe_target_real_results(checkpoint_dir: Path) -> list[SuiteResultVariant]:
    """Walk <checkpoint_dir>/target_real/*.yaml and return SuiteResultVariant list.

    Eval scripts (chat_eval, base_eval, dclm_eval) write per-config results
    under <checkpoint_dir>/target_real/<name>.yaml using the SuiteResultVariant
    schema (kind field). This function discovers and parses them so
    generate_all_plots can overlay them with the canonical linestyle/color
    assigned by _get_linestyle_for_variant.
    """
    target_real_dir = checkpoint_dir / "target_real"
    if not target_real_dir.is_dir():
        return []
    variants: list[SuiteResultVariant] = []
    for yaml_file in sorted(target_real_dir.glob("*.yaml")):
        try:
            with open(yaml_file) as f:
                data = yaml.safe_load(f)
        except Exception as e:
            logger.warning(f"Skipping unreadable target_real yaml {yaml_file}: {e}")
            continue
        if not isinstance(data, dict):
            logger.warning(
                f"Skipping malformed target_real yaml {yaml_file}: not a dict"
            )
            continue
        name = str(data.get("name", yaml_file.stem))
        try:
            variant = load_suite_from_dict(data, name=name)
        except ValueError as e:
            logger.warning(f"Skipping target_real yaml {yaml_file}: {e}")
            continue
        variants.append(variant)
    return variants


def _probe_eval_results(checkpoint_dir: Path) -> dict[str, Any]:
    """Probe a checkpoint directory for eval outputs and parse them.

    Parsing errors are logged as warnings and the corresponding eval type
    is skipped so that a single corrupt file does not abort the pipeline.
    """
    results: dict[str, Any] = {}

    # Base eval CSVs
    base_eval_dir = checkpoint_dir / "base_eval"
    if base_eval_dir.is_dir():
        csvs = sorted(base_eval_dir.glob("*.csv"))
        if csvs:
            try:
                results["base_eval"] = parse_base_eval_csv(csvs[-1])
            except (ValueError, KeyError) as e:
                logger.warning(f"Skipping malformed base eval in {checkpoint_dir}: {e}")

    # Harmfulness results
    try:
        harmfulness = parse_harmfulness_results(checkpoint_dir)
        if harmfulness:
            results["harmfulness"] = harmfulness
    except (ValueError, KeyError) as e:
        logger.warning(
            f"Skipping malformed harmfulness results in {checkpoint_dir}: {e}"
        )

    # Steering eval report (check root then report/steering_eval/ subdir)
    steering_path = checkpoint_dir / "report.yaml"
    if not steering_path.exists():
        steering_path = checkpoint_dir / "report" / "steering_eval" / "report.yaml"
    if steering_path.exists():
        try:
            results["steering"] = parse_steering_report(steering_path)
        except (ValueError, KeyError) as e:
            logger.warning(
                f"Skipping malformed steering report in {checkpoint_dir}: {e}"
            )

    # target_real subdir: PR-C writes <checkpoint_dir>/target_real/<name>.yaml
    target_real_variants = _probe_target_real_results(checkpoint_dir)
    if target_real_variants:
        results["target_real"] = target_real_variants

    return results


def _walk_checkpoint_node(
    checkpoint_dir: Path,
    source_chain: str,
    model_tag: str,
) -> ModelNode:
    """Recursively build a ModelNode tree from a checkpoint directory."""
    eval_results = _probe_eval_results(checkpoint_dir)
    children: list[ModelNode] = []

    for dir_name, segment_name in _DIR_TO_SEGMENT.items():
        child_dir = checkpoint_dir / dir_name
        if child_dir.is_dir():
            child_chain = f"{source_chain}/{segment_name}"
            child_node = _walk_checkpoint_node(child_dir, child_chain, model_tag)
            children.append(child_node)

    return ModelNode(
        source_chain=source_chain,
        model_tag=model_tag,
        checkpoint_dir=checkpoint_dir,
        eval_results=eval_results,
        children=children,
    )


def discover_model_tree(base_dir: Path) -> list[ModelNode]:
    """Discover all model checkpoint trees under base_dir.

    Scans both base_checkpoints/ and conditional_encdec_checkpoints/.
    Returns a list of root ModelNode objects (one per model tag per root).
    """
    base_dir = Path(base_dir)
    roots: list[ModelNode] = []

    # Base checkpoints
    base_ckpts = base_dir / "base_checkpoints"
    if base_ckpts.is_dir():
        for tag_dir in sorted(base_ckpts.iterdir()):
            if tag_dir.is_dir():
                model_tag = tag_dir.name
                node = _walk_checkpoint_node(tag_dir, "base", model_tag)
                roots.append(node)

    # Encoder-decoder checkpoints
    encdec_ckpts = base_dir / "conditional_encdec_checkpoints"
    if encdec_ckpts.is_dir():
        for tag_dir in sorted(encdec_ckpts.iterdir()):
            if tag_dir.is_dir():
                model_tag = tag_dir.name
                leaf = _find_encdec_leaf(tag_dir)
                if leaf is not None:
                    node = _walk_checkpoint_node(leaf, "encdec", model_tag)
                    roots.append(node)

    return roots


_EVAL_OUTPUT_DIRS = {"report", "base_eval"}

# Directories that should NOT be descended into by _find_encdec_leaf —
# they are SFT children walked by _walk_checkpoint_node, not intermediate path segments.
_SKIP_IN_LEAF_SEARCH = _EVAL_OUTPUT_DIRS | set(_DIR_TO_SEGMENT)


def _find_encdec_leaf(tag_dir: Path) -> Path | None:
    """Find the deepest directory under an encdec tag dir that has eval results or SFT children."""
    # Search children first — prefer deeper matches over shallow ones.
    # Skip eval output dirs and SFT child dirs (walked separately by _walk_checkpoint_node).
    for child in sorted(tag_dir.iterdir()):
        if child.is_dir() and child.name not in _SKIP_IN_LEAF_SEARCH:
            result = _find_encdec_leaf(child)
            if result is not None:
                return result

    # No deeper match found — check if current dir has eval content
    has_content = (
        (tag_dir / "base_eval").is_dir()
        or (tag_dir / "report").is_dir()
        or (tag_dir / "report.yaml").exists()
        or any((tag_dir / d).is_dir() for d in _DIR_TO_SEGMENT)
    )
    if has_content:
        return tag_dir
    return None


def collect_lineages(root: ModelNode) -> list[list[ModelNode]]:
    """Collect all root-to-descendant paths from a ModelNode tree.

    Each lineage is a list of ModelNodes from root to a node that has
    eval results or is a leaf.  A node with no children produces a
    single-element lineage.
    """
    if not root.children:
        return [[root]]

    lineages: list[list[ModelNode]] = []
    for child in root.children:
        # If the child has eval results AND children, include it as its own lineage endpoint
        if child.eval_results and child.children:
            lineages.append([root, child])
        for sub_lineage in collect_lineages(child):
            lineages.append([root] + sub_lineage)
    return lineages


# ---------------------------------------------------------------------------
# Plotters
# ---------------------------------------------------------------------------


def _extract_metric(
    node: ModelNode,
    eval_type: str,
    metric_key: str,
    condition_label: str | None = None,
) -> float | None:
    """Extract a single metric value from a ModelNode's eval_results.

    For harmfulness: tries harmful_prompts, then overall, then benign_prompts sections
    For base_eval: tries top-level key, then results dict
    For steering: tries aggregate dict
    """
    evals = node.eval_results.get(eval_type)
    if evals is None:
        return None

    if eval_type == "harmfulness":
        label = condition_label or "default"
        condition = evals.get(label)
        if condition is None:
            return None
        return _extract_harmfulness_value(condition, metric_key)

    if eval_type == "base_eval":
        if metric_key in evals:
            return evals[metric_key]
        return evals.get("results", {}).get(metric_key)

    if eval_type == "steering":
        return evals.get("aggregate", {}).get(metric_key)

    return None


def _extract_harmfulness_value(
    cond_data: dict[str, Any],
    metric_key: str,
) -> float | None:
    """Extract a harmfulness metric from a condition data dict."""
    explicit_section = {
        "asr": "harmful_prompts",
        "safe_response_rate": "harmful_prompts",
        "compliance_rate": "benign_prompts",
        "mean_score": "overall",
    }.get(metric_key)
    if explicit_section is not None:
        value = cond_data.get(explicit_section, {}).get(metric_key)
        return float(value) if value is not None else None
    for section in ("harmful_prompts", "overall", "benign_prompts"):
        if metric_key in cond_data.get(section, {}):
            return cond_data[section][metric_key]
    return None


@dataclass
class ParsedCondition:
    """Parsed components of a conditioning label like 'safety_sft/f0.1/m0.5'."""

    dataset: str  # "safety_sft"
    feature_ratio: str  # "f0.1"
    magnitude: str  # "m0.5"


_CONDITION_LABEL_RE = re.compile(
    r"^(?P<dataset>[^/]+)/f(?P<fr>[0-9.]+)/m(?P<mag>[0-9.]+)$"
)


def parse_condition_label(label: str) -> ParsedCondition | None:
    """Parse a conditioning label into its components.

    Expected format: '{dataset}/f{ratio}/m{magnitude}'
    e.g. 'safety_sft/f0.1/m0.5' -> ParsedCondition('safety_sft', 'f0.1', 'm0.5')

    Returns None if the label doesn't match the expected format.
    """
    m = _CONDITION_LABEL_RE.match(label)
    if m is None:
        return None
    return ParsedCondition(
        dataset=m.group("dataset"),
        feature_ratio=f"f{m.group('fr')}",
        magnitude=f"m{m.group('mag')}",
    )


_METRIC_OPTIMAL_DIR: dict[str, str] = {
    "asr": "min",
    "mean_score": "min",
    "safe_response_rate": "max",
    "compliance_rate": "max",
}


def _metric_display_name(metric_key: str) -> str:
    if metric_key == "compliance_rate":
        return "Benign Compliance Rate (non-refusal)"
    return metric_key


def _optimal_value(values: list[float], metric_key: str) -> float:
    """Return the optimal value from a list according to the metric's direction."""
    direction = _METRIC_OPTIMAL_DIR.get(metric_key, "min")
    return min(values) if direction == "min" else max(values)


def plot_lineage_bars(
    lineage: list[ModelNode],
    eval_type: str,
    metric_key: str,
    condition_label: str | None = None,
) -> Figure | None:
    """Plot a bar chart showing metric progression along a model lineage.

    Returns a matplotlib Figure, or None if fewer than 2 nodes have valid data.
    """
    labels: list[str] = []
    values: list[float] = []
    for node in lineage:
        val = _extract_metric(node, eval_type, metric_key, condition_label)
        if val is not None:
            labels.append(node.source_chain)
            values.append(val)

    if len(labels) < 2:
        return None

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.5), 5))
    x = np.arange(len(labels))
    ax.bar(x, values, width=0.6, color="steelblue", edgecolor="black", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    metric_label = _metric_display_name(metric_key)
    ax.set_ylabel(metric_label)

    model_tag = lineage[0].model_tag if lineage else "?"
    ax.set_title(f"{model_tag} — {eval_type} / {metric_label}")
    fig.tight_layout()
    return fig


# Fixed color palette for source chain variants
_VARIANT_COLORS: dict[str, str] = {
    "base": "#4C72B0",
    "base/standard": "#55A868",
    "base/safety": "#C44E52",
    "base/capabilities": "#8172B2",
    "base/safety/capabilities": "#CCB974",
    "base/safety/rl": "#64B5CD",
    "base/standard/rl": "#72B7B2",
    "encdec": "#DD8452",
    "encdec/safety": "#DA8BC3",
    "base/safety_sft_conditioning": "#E24A33",
    "base/capabilities_sft_conditioning": "#FBC15E",
    "base/safety/safety_sft_conditioning": "#B85450",
    "base/safety/capabilities_sft_conditioning": "#D4A847",
}

# Fixed color palette for harmfulness condition labels
_CONDITION_COLORS: dict[str, str] = {
    "default": "#4C72B0",
    "common": "#DD8452",
}


def _flatten_tree(node: ModelNode) -> list[ModelNode]:
    """Flatten a ModelNode tree into a list (BFS order)."""
    flat: list[ModelNode] = [node]
    for child in node.children:
        flat.extend(_flatten_tree(child))
    return flat


def plot_cross_family_bars(
    roots: list[ModelNode],
    eval_type: str,
    metric_key: str,
    condition_label: str | None = None,
) -> Figure | None:
    """Plot grouped bar chart comparing model families side by side.

    Each family (model_tag) is a group on the X-axis. Within each group,
    bars represent different variants (source chains).

    Returns a matplotlib Figure, or None if fewer than 2 total data points.
    """
    # Collect all variants across all families.
    # Qualify model_tag with architecture prefix (dec/encdec) so that
    # decoder-only and encoder-decoder roots are separate x-axis groups.
    family_data: dict[
        str, dict[str, float]
    ] = {}  # qualified_tag -> {source_chain: value}
    all_variants: list[str] = []

    for root in roots:
        arch = (
            "dec"
            if root.source_chain.startswith("base")
            else root.source_chain.split("/")[0]
        )
        family_key = f"{arch}/{root.model_tag}"
        if family_key not in family_data:
            family_data[family_key] = {}
        for node in _flatten_tree(root):
            val = _extract_metric(node, eval_type, metric_key, condition_label)
            if val is not None:
                # Normalise encdec variants to base equivalents so they
                # share the same legend entry and colour (e.g. "encdec" → "base").
                variant = node.source_chain.replace("encdec", "base", 1)
                family_data[family_key][variant] = val
                if variant not in all_variants:
                    all_variants.append(variant)

    # Filter out empty groups
    family_data = {k: v for k, v in family_data.items() if v}
    total_points = sum(len(v) for v in family_data.values())
    if total_points < 2:
        return None

    tags = list(family_data.keys())
    n_families = len(tags)
    max_bars = max((len(family_data[t]) for t in tags), default=1)

    fig, ax = plt.subplots(figsize=(max(8, n_families * max_bars * 0.8), 5))
    bar_width = 0.8 / max(max_bars, 1)
    x = np.arange(n_families)

    # Plot each group with only its own bars, centered.
    seen_labels: set[str] = set()
    for f_idx, tag in enumerate(tags):
        present = [v for v in all_variants if v in family_data[tag]]
        n = len(present)
        for b_idx, variant in enumerate(present):
            offset = (b_idx - n / 2 + 0.5) * bar_width
            color = _VARIANT_COLORS.get(variant, f"C{all_variants.index(variant)}")
            label = variant if variant not in seen_labels else None
            ax.bar(
                x[f_idx] + offset,
                family_data[tag][variant],
                bar_width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            seen_labels.add(variant)

    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    metric_label = _metric_display_name(metric_key)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Cross-Family Comparison — {eval_type} / {metric_label}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


def plot_cross_family_harmfulness(
    roots: list[ModelNode],
    metric_key: str,
) -> Figure | None:
    """Plot grouped bar chart comparing harmfulness across model families.

    Decoder groups: bars for each finetune variant using default eval values.

    Encdec groups: best-case conditioning bars per dataset. The "common"
    condition is excluded (belongs in sweep chart).
    Best-case = min for asr/mean_score, max for safe_response_rate and
    compliance_rate.

    Returns None if fewer than 2 total data points.
    """
    bar_colors: dict[str, str] = {**_VARIANT_COLORS, **_CONDITION_COLORS}

    family_data: dict[str, dict[str, float]] = {}  # qualified_tag -> {bar_key: value}
    all_bar_keys: list[str] = []

    for root in roots:
        is_encdec = root.source_chain.startswith("encdec")
        arch = "encdec" if is_encdec else "dec"
        family_key = f"{arch}/{root.model_tag}"
        if family_key not in family_data:
            family_data[family_key] = {}

        for node in _flatten_tree(root):
            harmfulness = node.eval_results.get("harmfulness")
            if not isinstance(harmfulness, dict):
                continue
            variant = node.source_chain.replace("encdec", "base", 1)

            if not is_encdec:
                # Decoder-only models have no encoder pathway; plot default rows
                # as regular model-variant bars.
                for cond_label, cond_data in harmfulness.items():
                    if cond_label != "default":
                        continue
                    val = _extract_harmfulness_value(cond_data, metric_key)
                    if val is not None:
                        family_data[family_key][variant] = val
                        if variant not in all_bar_keys:
                            all_bar_keys.append(variant)
            else:
                # Encoder-decoder evals should only contribute explicit,
                # non-empty conditioning configs.
                conditioning_values: dict[
                    str, list[float]
                ] = {}  # dataset_key -> [values]
                for cond_label, cond_data in harmfulness.items():
                    val = _extract_harmfulness_value(cond_data, metric_key)
                    if val is None:
                        continue
                    if cond_label == "common":
                        # Excluded from bar chart (shown in sweep chart)
                        continue
                    parsed = parse_condition_label(cond_label)
                    if parsed is not None:
                        ds_key = parsed.dataset
                        if ds_key not in conditioning_values:
                            conditioning_values[ds_key] = []
                        conditioning_values[ds_key].append(val)

                # Add best-case conditioning bars per dataset
                for ds_key, vals in conditioning_values.items():
                    bar_key = f"{variant}/{ds_key}_conditioning"
                    family_data[family_key][bar_key] = _optimal_value(vals, metric_key)
                    if bar_key not in all_bar_keys:
                        all_bar_keys.append(bar_key)

    # Filter out empty groups
    family_data = {k: v for k, v in family_data.items() if v}
    total_points = sum(len(v) for v in family_data.values())
    if total_points < 2:
        return None

    tags = list(family_data.keys())
    n_families = len(tags)
    max_bars = max((len(family_data[t]) for t in tags), default=1)

    fig, ax = plt.subplots(figsize=(max(8, n_families * max_bars * 0.8), 5))
    bar_width = 0.8 / max(max_bars, 1)
    x = np.arange(n_families)

    # Plot each group with only its own bars, centered.
    seen_labels: set[str] = set()
    for f_idx, tag in enumerate(tags):
        present_keys = [k for k in all_bar_keys if k in family_data[tag]]
        n = len(present_keys)
        for b_idx, bar_key in enumerate(present_keys):
            offset = (b_idx - n / 2 + 0.5) * bar_width
            color = bar_colors.get(bar_key, f"C{all_bar_keys.index(bar_key)}")
            label = bar_key if bar_key not in seen_labels else None
            ax.bar(
                x[f_idx] + offset,
                family_data[tag][bar_key],
                bar_width,
                label=label,
                color=color,
                edgecolor="black",
                linewidth=0.5,
            )
            seen_labels.add(bar_key)

    ax.set_xticks(x)
    ax.set_xticklabels(tags)
    metric_label = _metric_display_name(metric_key)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Cross-Family Comparison — harmfulness / {metric_label}")
    ax.legend(loc="upper right", fontsize=8)
    fig.tight_layout()
    return fig


_LINESTYLES = ["-", "--", "-.", ":"]

# Colors for conditioning sweep lines
_SWEEP_COLORS = [
    "#E24A33",
    "#348ABD",
    "#988ED5",
    "#777777",
    "#FBC15E",
    "#8EBA42",
    "#FFB5B8",
    "#56B4E9",
]


def plot_encdec_conditioning_sweep(
    roots: list[ModelNode],
    metric_key: str,
) -> Figure | None:
    """Plot line chart showing how conditioning magnitude affects a harmfulness metric.

    X-axis: magnitude (numeric values like 0.1, 0.5, 0.9)
    Y-axis: metric value
    Lines: one per {dataset}/{feature_ratio} combo
    Dashed horizontals: common reference line

    Only encdec models are included. Returns None if no sweep data.
    """
    # Collect sweep data and reference lines from encdec roots
    # sweep_data: {line_key: {magnitude_float: value}}
    sweep_data: dict[str, dict[float, float]] = {}
    baselines: dict[str, float] = {}

    for root in roots:
        if not root.source_chain.startswith("encdec"):
            continue
        for node in _flatten_tree(root):
            harmfulness = node.eval_results.get("harmfulness")
            if not isinstance(harmfulness, dict):
                continue
            for cond_label, cond_data in harmfulness.items():
                val = _extract_harmfulness_value(cond_data, metric_key)
                if val is None:
                    continue
                parsed = parse_condition_label(cond_label)
                if parsed is not None:
                    line_key = f"{parsed.dataset}/{parsed.feature_ratio}"
                    mag = float(parsed.magnitude[1:])  # strip "m" prefix
                    if line_key not in sweep_data:
                        sweep_data[line_key] = {}
                    sweep_data[line_key][mag] = val
                elif cond_label == "common":
                    baselines["common"] = val

    if not sweep_data:
        return None

    fig, ax = plt.subplots(figsize=(8, 5))

    for idx, (line_key, mag_vals) in enumerate(sorted(sweep_data.items())):
        mags = sorted(mag_vals.keys())
        vals = [mag_vals[m] for m in mags]
        color = _SWEEP_COLORS[idx % len(_SWEEP_COLORS)]
        ax.plot(
            mags,
            vals,
            marker="o",
            markersize=5,
            label=line_key,
            color=color,
            linewidth=1.5,
        )

    # Reference lines as dashed horizontals
    baseline_styles = {"common": ("orange", ":")}
    for bl_name, bl_val in baselines.items():
        color, ls = baseline_styles.get(bl_name, ("gray", "--"))
        ax.axhline(
            y=bl_val,
            color=color,
            linestyle=ls,
            alpha=0.7,
            label=f"{bl_name} ({bl_val:.3f})",
        )

    ax.set_xlabel("Conditioning Magnitude")
    metric_label = _metric_display_name(metric_key)
    ax.set_ylabel(metric_label)
    ax.set_title(f"Encdec Conditioning Sweep — {metric_label}")
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_dose_response_overlay(
    nodes: list[ModelNode],
    feature_id: int,
) -> Figure:
    """Overlay dose-response curves for a single feature across model variants.

    Returns a matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=(8, 5))

    line_idx = 0
    for node in nodes:
        steering = node.eval_results.get("steering")
        if steering is None:
            continue
        feat = next((f for f in steering["features"] if f["id"] == feature_id), None)
        if feat is None:
            continue

        dr = feat["dose_response"]
        label = f"{node.model_tag}/{node.source_chain}"
        color = _VARIANT_COLORS.get(node.source_chain, f"C{line_idx}")
        ls = _LINESTYLES[line_idx % len(_LINESTYLES)]
        ax.plot(
            dr["magnitude_levels"],
            dr["win_rates"],
            marker="o",
            markersize=4,
            label=label,
            color=color,
            linestyle=ls,
            linewidth=1.5,
        )
        line_idx += 1

    ax.set_xlabel("Magnitude Level")
    ax.set_ylabel("Win Rate")
    ax.set_title(f"Dose-Response Overlay — Feature {feature_id}")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_ylim(0.0, 1.0)
    if line_idx > 0:
        ax.legend(loc="best", fontsize=8)
    fig.tight_layout()
    return fig


def plot_target_real_variants(
    roots: list[ModelNode],
) -> Figure | None:
    """Plot a summary chart of target_real variants discovered across model roots.

    Each variant is rendered as a horizontal line at its rank index (Y)
    using the canonical linestyle/color from _get_linestyle_for_variant.

    Returns None if no target_real variants are found.
    """
    # Collect all variants across all nodes with their model label
    entries: list[tuple[str, SuiteResultVariant]] = []
    for root in roots:
        for node in _flatten_tree(root):
            variants = node.eval_results.get("target_real")
            if not isinstance(variants, list):
                continue
            model_label = f"{node.model_tag}/{node.source_chain}"
            for variant in variants:
                entries.append((model_label, variant))

    if not entries:
        return None

    fig, ax = plt.subplots(figsize=(max(8, len(entries) * 0.6 + 2), 4))
    seen_labels: set[str] = set()

    for idx, (model_label, variant) in enumerate(entries):
        ls, color = _get_linestyle_for_variant(variant)
        # Use the variant's name attribute for the legend label
        variant_name = getattr(variant, "name", str(type(variant).__name__))
        legend_label = f"{variant_name}" if variant_name not in seen_labels else None
        ax.plot(
            [0, 1],
            [idx, idx],
            linestyle=ls,
            color=color,
            linewidth=2,
            label=legend_label,
        )
        ax.text(1.02, idx, f"{model_label}: {variant_name}", va="center", fontsize=7)
        seen_labels.add(variant_name)

    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title("target_real Variant Inventory (linestyle = mode)")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# CLI helpers
# ---------------------------------------------------------------------------

# Metrics to plot for each eval type
_HARMFULNESS_METRICS = [
    "asr",
    "mean_score",
    "safe_response_rate",
    "compliance_rate",
]
_BASE_EVAL_METRICS = ["core_metric"]
_STEERING_METRICS = ["confidence_weighted_win_rate"]


def _slugify_lineage(lineage: list[ModelNode]) -> str:
    """Create a filename-safe slug from a lineage."""
    parts = [n.source_chain.split("/")[-1] for n in lineage]
    return "-".join(parts)


def _collect_all_feature_ids(roots: list[ModelNode]) -> set[int]:
    """Collect all feature IDs from steering evals across all nodes."""
    ids: set[int] = set()
    for root in roots:
        for node in _flatten_tree(root):
            steering = node.eval_results.get("steering")
            if steering:
                for feat in steering["features"]:
                    ids.add(feat["id"])
    return ids


def _collect_condition_labels(lineage: list[ModelNode], eval_type: str) -> list[str]:
    """Collect all condition labels present in a lineage for a given eval type."""
    labels: set[str] = set()
    for node in lineage:
        evals = node.eval_results.get(eval_type)
        if isinstance(evals, dict):
            labels.update(evals.keys())
    return sorted(labels) if labels else ["default"]


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------


def generate_all_plots(base_dir: Path, output_dir: Path) -> None:
    """Generate all visualization plots from checkpoint tree.

    Parameters
    ----------
    base_dir:
        Root directory containing base_checkpoints/ and conditional_encdec_checkpoints/.
    output_dir:
        Where to write PNG files (typically ``fig/``).
    """
    base_dir = Path(base_dir)
    output_dir = Path(output_dir)

    roots = discover_model_tree(base_dir)
    if not roots:
        logger.info("No model checkpoints found, nothing to plot.")
        return

    # --- Per-lineage plots ---
    for root in roots:
        lineages = collect_lineages(root)
        tag = root.model_tag

        for lineage in lineages:
            slug = _slugify_lineage(lineage)

            # Harmfulness lineage bars
            for cond_label in _collect_condition_labels(lineage, "harmfulness"):
                for metric in _HARMFULNESS_METRICS:
                    fig = plot_lineage_bars(
                        lineage, "harmfulness", metric, condition_label=cond_label
                    )
                    if fig is not None:
                        out_dir = output_dir / tag / "harmfulness"
                        out_dir.mkdir(parents=True, exist_ok=True)
                        safe_label = cond_label.replace("/", "_")
                        suffix = f"_{safe_label}" if cond_label != "default" else ""
                        fig.savefig(out_dir / f"{slug}_{metric}{suffix}.png", dpi=150)
                        plt.close(fig)

            # Base eval lineage bars
            for metric in _BASE_EVAL_METRICS:
                fig = plot_lineage_bars(lineage, "base_eval", metric)
                if fig is not None:
                    out_dir = output_dir / tag / "base_eval"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_dir / f"{slug}_{metric}.png", dpi=150)
                    plt.close(fig)

            # Steering lineage bars
            for metric in _STEERING_METRICS:
                fig = plot_lineage_bars(lineage, "steering", metric)
                if fig is not None:
                    out_dir = output_dir / tag / "steering"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    fig.savefig(out_dir / f"{slug}_{metric}.png", dpi=150)
                    plt.close(fig)

    # --- Cross-family comparison plots ---
    comp_dir = output_dir / "comparison"
    comp_dir.mkdir(parents=True, exist_ok=True)

    for metric in _HARMFULNESS_METRICS:
        fig = plot_cross_family_harmfulness(roots, metric)
        if fig is not None:
            fig.savefig(comp_dir / f"harmfulness_{metric}.png", dpi=150)
            plt.close(fig)

        fig = plot_encdec_conditioning_sweep(roots, metric)
        if fig is not None:
            fig.savefig(comp_dir / f"harmfulness_{metric}_conditioning.png", dpi=150)
            plt.close(fig)

    for metric in _BASE_EVAL_METRICS:
        fig = plot_cross_family_bars(roots, "base_eval", metric)
        if fig is not None:
            fig.savefig(comp_dir / f"base_eval_{metric}.png", dpi=150)
            plt.close(fig)

    for metric in _STEERING_METRICS:
        fig = plot_cross_family_bars(roots, "steering", metric)
        if fig is not None:
            fig.savefig(comp_dir / f"steering_{metric}.png", dpi=150)
            plt.close(fig)

    # --- Steering dose-response overlays ---
    all_nodes = [n for root in roots for n in _flatten_tree(root)]
    feature_ids = _collect_all_feature_ids(roots)
    for fid in sorted(feature_ids):
        fig = plot_dose_response_overlay(all_nodes, feature_id=fid)
        fig.savefig(comp_dir / f"steering_dose_response_f{fid}.png", dpi=150)
        plt.close(fig)

    # --- target_real variant inventory ---
    fig = plot_target_real_variants(roots)
    if fig is not None:
        fig.savefig(comp_dir / "target_real_variants.png", dpi=150)
        plt.close(fig)

    logger.info(f"All plots written to {output_dir}")


def main() -> None:
    """CLI entry point."""
    with initialize_config_dir(config_dir=_CONFIG_DIR, version_base=None):
        overrides = [a for a in sys.argv[1:] if "=" in a and not a.startswith("-")]
        cfg = compose(config_name=_CONFIG_NAME, overrides=overrides)

    base_dir = Path(cfg.base_dir) if cfg.base_dir else Path(get_base_dir())
    output_dir = Path(cfg.output_dir)

    generate_all_plots(base_dir=base_dir, output_dir=output_dir)


if __name__ == "__main__":
    main()
