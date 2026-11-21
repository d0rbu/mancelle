"""Offline, equally weighted distributions of AxBench per-concept judge means.

The JSONL supplies the exact plotted means; the companion Parquet proves their
population and raw-score consistency. This reader validates the plotting inputs,
not generation provenance or the complete producer schema.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from scripts.eval.axbench_adv.pressures import Pressure, parse_pressures
from scripts.eval.axbench_upstream.merge_methods import (
    INPUTS_PER_CONCEPT,
    SUPPORTED_METHOD_FACTORS,
)
from scripts.eval.axbench_upstream.merge_methods import METHOD_FACTORS as METHOD_FACTORS
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.execution_scope import validate_concept_ids
from steering.eval.axbench_adv.method_grid import ALL_METHOD_FACTORS
from steering.eval.axbench_adv.types import (
    PLOT_EFFORT_LEVELS,
    validate_method_set,
    validate_method_subset,
)

AXES = {
    "relevance_concept_ratings": "Concept relevance",
    "fluency_ratings": "Fluency",
    "relevance_instruction_ratings": "Instruction-following",
}
# Preserve the existing Adv palette; the four encoder-decoder analogs get
# distinct, fixed colors independent of the selected method subset.
COLORS = {
    "DiffMean": "#1f77b4",
    "LinearProbe": "#ff7f0e",
    "LoRA": "#2ca02c",
    "PromptSteering": "#9467bd",
    "SelfConditioning": "#d62728",
    "SelfConditioningMasked": "#17becf",
    "EncDecDiffMean": "#8c564b",
    "EncDecLinearProbe": "#e377c2",
    "EncDecPromptSteering": "#7f7f7f",
    "EncDecLoRA": "#bcbd22",
}
_IDENTITY = ["concept_id", "input_id", "dataset_name", "input_concept", "original_prompt"]


def _object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _json(text: str) -> Any:
    return json.loads(text, object_pairs_hook=_object)


def _number(value: Any, *, label: str, rating: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be numeric, not {value!r}")
    result = float(value)
    if not math.isfinite(result) or (rating and not 0 <= result <= 2):
        raise ValueError(f"{label} must be finite" + (" in [0, 2]" if rating else ""))
    return result


def _factors(values: Any, grid: Mapping[str, tuple[float, ...]]) -> dict[str, float]:
    if not isinstance(values, Mapping) or not values:
        raise ValueError("method_factors must be a nonempty method-to-factor mapping")
    result = {}
    for method, value in values.items():
        factor = _number(value, label=f"{method}.factor")
        if method not in grid or factor not in grid[method]:
            raise ValueError(f"unknown method or undeclared factor: {method}={factor}")
        result[method] = factor
    return dict(sorted(result.items()))


def _snapshot(path: Path, sources: dict[str, str]) -> bytes:
    data = path.read_bytes()
    sources[str(path)] = hashlib.sha256(data).hexdigest()
    return data


def _load_scores(
    directory: Path, factors: Mapping[str, float], sources: dict[str, str],
    *, expected_concept_ids: tuple[int, ...] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    entries = [
        _json(line)
        for line in _snapshot(directory / "steering.jsonl", sources).decode("utf-8").splitlines()
        if line.strip()
    ]
    if not entries or any(
        not isinstance(entry, dict) or set(entry) != {"concept_id", "results"} for entry in entries
    ):
        raise ValueError("JSONL must contain concept_id/results objects")
    ids = [entry["concept_id"] for entry in entries]
    expected_ids = list(range(len(ids))) if expected_concept_ids is None else list(expected_concept_ids)
    if any(type(value) is not int for value in ids) or ids != expected_ids:
        raise ValueError("JSONL concepts must occur exactly once in the expected ascending population")
    frame = pd.read_parquet(io.BytesIO(_snapshot(directory / "steering_data.parquet", sources)))
    required = [*_IDENTITY, "factor"]
    if not frame.columns.is_unique or not set(required).issubset(frame.columns):
        raise ValueError("Parquet has duplicate columns or missing identity columns")
    if frame[required].isna().any().any():
        raise ValueError("Parquet identity columns contain nulls")
    for column in ("concept_id", "input_id"):
        if not pd.api.types.is_integer_dtype(frame[column].dtype):
            raise ValueError(f"{column} must have integer dtype")
    for column in ("dataset_name", "input_concept", "original_prompt"):
        if not frame[column].map(lambda x: isinstance(x, str) and bool(x.strip())).all():
            raise ValueError(f"{column} must contain nonblank strings")
    for value in frame["factor"].tolist():
        _number(value, label="Parquet factor")
    if frame.duplicated(["concept_id", "input_id", "factor"]).any():
        raise ValueError("duplicate Parquet concept/input/factor key")
    if sorted(frame["concept_id"].unique().tolist()) != ids:
        raise ValueError("JSONL/Parquet concept coverage differs")
    if frame["dataset_name"].nunique() != 1:
        raise ValueError("one report requires exactly one dataset")
    if (frame.groupby("concept_id")["input_concept"].nunique() != 1).any():
        raise ValueError("concept description varies within a concept")
    if (frame.groupby(["concept_id", "input_id"])["original_prompt"].nunique() != 1).any():
        raise ValueError("prompt identity varies across factors")

    rows: list[dict[str, Any]] = []
    population = pd.DataFrame()
    for method, factor in factors.items():
        selected = frame.loc[frame["factor"].eq(factor)].sort_values(
            ["concept_id", "input_id"], kind="stable"
        )
        expected = [(concept, i) for concept in ids for i in range(INPUTS_PER_CONCEPT)]
        if list(zip(selected["concept_id"], selected["input_id"], strict=True)) != expected:
            raise ValueError(f"{method}: selected factor lacks the complete concept/input grid")
        identity = selected[_IDENTITY].reset_index(drop=True)
        # Complete selected grids plus the factor-invariant identity checks above
        # already prove that every method uses the same population.
        population = identity
        for entry in entries:
            concept = entry["concept_id"]
            results = entry["results"]
            evaluator = results.get("LMJudgeEvaluator") if isinstance(results, dict) else None
            block = evaluator.get(method) if isinstance(evaluator, dict) else None
            if not isinstance(block, dict) or not isinstance(block.get("factor"), list):
                raise ValueError(f"concept {concept}: missing {method} judge block/factors")
            block_factors = [_number(v, label=f"{method}.factor") for v in block["factor"]]
            if len(set(block_factors)) != len(block_factors) or factor not in block_factors:
                raise ValueError(f"{method}: duplicate factors or selected factor absent in JSONL")
            index = block_factors.index(factor)
            concept_frame = selected.loc[selected["concept_id"].eq(concept)]
            source_concept = frame.loc[frame["concept_id"].eq(concept)]
            declared = source_concept["factor"].eq(factor).tolist()
            for field in AXES:
                values = block.get(field)
                column = f"{method}_LMJudgeEvaluator_{field}"
                if not isinstance(values, list) or len(values) != len(block_factors):
                    raise ValueError(f"{method}.{field}: factor/score length mismatch")
                score = _number(values[index], label=f"{method}.{field}", rating=True)
                if column not in frame:
                    raise ValueError(f"missing score column: {column}")
                raw = [
                    _number(value, label=column, rating=True)
                    for value in concept_frame[column].tolist()
                ]
                saved_raw = block.get(f"raw_{field}")
                if not isinstance(saved_raw, list) or len(saved_raw) != len(source_concept):
                    raise ValueError(f"{method}.{field}: raw JSONL coverage differs")
                selected_raw = [
                    _number(value, label=f"raw_{field}", rating=True)
                    for value, used in zip(saved_raw, declared, strict=True)
                    if used
                ]
                if (
                    selected_raw
                    != source_concept.loc[source_concept["factor"].eq(factor), column].tolist()
                ):
                    raise ValueError(f"{method}.{field}: raw JSONL/Parquet ratings differ")
                if not math.isclose(score, sum(raw) / len(raw), rel_tol=1e-12, abs_tol=1e-12):
                    raise ValueError(f"{method}.{field}: JSONL mean disagrees with Parquet")
                rows.append(
                    {
                        "method": method,
                        "factor": factor,
                        "concept_id": concept,
                        "concept": concept_frame["input_concept"].iloc[0],
                        "axis": field,
                        "score": score,
                        "n_prompts": len(raw),
                    }
                )
    return pd.DataFrame(rows), population


def _histogram(values: list[float], edges: np.ndarray) -> tuple[list[int], list[float]]:
    counts = np.histogram(values, bins=edges)[0]
    return counts.tolist(), (counts * (100.0 / len(values))).tolist()


def _render(
    scores: pd.DataFrame, edges: np.ndarray, title: str, path: Path
) -> list[dict[str, Any]]:
    colors = {**COLORS, "NLSteering": "#393b79"}
    methods = sorted(scores["method"].unique().tolist())
    if not methods:
        raise ValueError("scores must contain at least one method")
    figure = Figure(figsize=(15, 3.2 * len(methods)), layout="constrained")
    FigureCanvasAgg(figure)
    axes = figure.subplots(len(methods), len(AXES), sharex=True, sharey=True, squeeze=False)
    summaries: list[dict[str, Any]] = []
    try:
        for row, method in enumerate(methods):
            method_scores = scores.loc[scores["method"].eq(method)]
            factor = float(method_scores["factor"].iloc[0])
            for column, (field, label) in enumerate(AXES.items()):
                ax = axes[row, column]
                group = method_scores.loc[method_scores["axis"].eq(field)]
                values = group["score"].tolist()
                counts, percentages = _histogram(values, edges)
                mean = float(np.mean(values))
                std = float(np.std(values, ddof=1)) if len(values) > 1 else None
                variance = std * std if std is not None else None
                summaries.append(
                    {
                        "method": method,
                        "factor": factor,
                        "axis": field,
                        "n_concepts": len(values),
                        "mean": mean,
                        "sample_std": std,
                        "sample_variance": variance,
                        "n_prompts_per_concept": group["n_prompts"].tolist(),
                        "counts": counts,
                        "percentages": percentages,
                    }
                )
                bars = ax.bar(
                    edges[:-1],
                    percentages,
                    width=np.diff(edges),
                    align="edge",
                    color=colors[method],
                    alpha=0.4,
                    edgecolor=colors[method],
                    linewidth=0.8,
                    label=f"{method} (factor {factor:g})",
                )
                for bar in bars:
                    bar.set_label(f"{method} (factor {factor:g})")
                variance_text = f"{variance:.3f}" if variance is not None else "n/a"
                ax.text(
                    0.03,
                    0.97,
                    f"mean {mean:.2f}  var {variance_text}",
                    transform=ax.transAxes,
                    ha="left",
                    va="top",
                )
                ax.set(title=label, xlabel="Per-concept mean judge score", xlim=(0, 2), ylim=(0, 100))
                ax.grid(axis="y", linestyle=":", alpha=0.4)
            axes[row, 0].set_ylabel(f"{method} (factor {factor:g})\nConcepts (%)")
        count = int(scores["concept_id"].nunique())
        figure.suptitle(
            f"{title}\n{count} concepts; one equally weighted mean per concept", fontsize=13
        )
        figure.savefig(path, dpi=150)
    finally:
        figure.clear()
    return summaries


def generate_histogram_reports(
    *,
    input_dir: Path,
    output_dir: Path,
    pipeline: str,
    method_factors: Mapping[str, float],
    bins: int = 20,
    pressure: str | Pressure = Pressure.GEPA,
) -> Path:
    """Validate all inputs, then write one upstream report or five Adv reports.

    Adv uses the completed manifest's factors and hashes, never historical
    source/control scores. ``metadata.json`` is published last; its absence
    denotes an incomplete output. Existing nonempty outputs are never replaced.
    """
    input_dir, output_dir = input_dir.expanduser().resolve(), output_dir.expanduser().resolve()
    if pipeline not in {"upstream", "adv"}:
        raise ValueError("pipeline must be upstream or adv")
    if not isinstance(method_factors, Mapping):
        raise ValueError("method_factors must be a mapping")
    try:
        selected_pressure = Pressure(pressure)
    except ValueError as exc:
        raise ValueError(f"unknown AxBench-Adv pressure {pressure!r}") from exc
    if pipeline == "upstream" and selected_pressure != Pressure.GEPA:
        raise ValueError("upstream histograms do not have an adversarial pressure")
    if selected_pressure in (
        Pressure.LINEAR_PROBE_COUNTER,
        Pressure.DIFFMEAN_COUNTER,
    ):
        raise ValueError("additive-counter histogram selection is not implemented")
    if type(bins) is not int or not 1 <= bins <= 100:
        raise ValueError("bins must be an integer in [1, 100]")
    if output_dir.is_relative_to(input_dir) or input_dir.is_relative_to(output_dir):
        raise ValueError("input and output directories must be path-disjoint")
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise ValueError("output directory must be new or empty")
    sources: dict[str, str] = {}
    manifest: dict[str, Any] = {}
    concept_ids: tuple[int, ...] | None = None
    if pipeline == "adv":
        if method_factors:
            raise ValueError("Adv uses recorded factors; method_factors must be empty")
        manifest = _json(_snapshot(input_dir / "manifest.json", sources).decode("utf-8"))
        if not isinstance(manifest, dict) or manifest.get("schema_version") != 2:
            raise ValueError("Adv requires a completed schema-2 run manifest")
        factors = _factors(manifest.get("best_factors"), ALL_METHOD_FACTORS)
        try:
            methods = (
                validate_method_subset(factors)
                if "methods" in manifest
                else validate_method_set(factors)
            )
        except AxBenchAdvError as exc:
            raise ValueError(
                "Adv method subsets require an explicit manifest methods list"
            ) from exc
        if ("methods" in manifest or "NLSteering" in factors) and manifest.get("methods") != list(
            methods
        ):
            raise ValueError("Adv manifest method order differs from its factors")
        if type(manifest.get("max_concepts")) is not int or manifest["max_concepts"] <= 0:
            raise ValueError("Adv manifest max_concepts must be positive")
        declared = manifest.get("concept_ids", list(range(manifest["max_concepts"])))
        if not isinstance(declared, list):
            raise ValueError("Adv manifest concept_ids must be a list")
        try:
            concept_ids = validate_concept_ids(declared, max_concepts=manifest["max_concepts"])
        except AxBenchAdvError as exc:
            raise ValueError(f"Adv manifest has invalid concept_ids: {exc}") from exc
        if not isinstance(manifest.get("files"), dict):
            raise ValueError("Adv manifest must contain file hashes")
        try:
            available = parse_pressures(manifest.get("pressures", [Pressure.GEPA]), environ={})
        except (TypeError, ValueError) as exc:
            raise ValueError("Adv manifest has invalid pressures") from exc
        if selected_pressure not in available:
            raise ValueError(f"Adv manifest does not contain pressure {selected_pressure}")
        if selected_pressure == Pressure.GEPA:
            points = {
                f"p{level}": input_dir / "points" / f"p{level}" / "judged"
                for level in PLOT_EFFORT_LEVELS
            }
        elif selected_pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
            config = manifest.get(selected_pressure)
            search = (
                config.get("search")
                if selected_pressure == Pressure.GCG and isinstance(config, Mapping)
                else config
            )
            counts: Any = search.get("optimized_token_counts") if isinstance(search, Mapping) else None
            if (
                not isinstance(counts, list)
                or not counts
                or any(type(value) is not int or value < 0 for value in counts)
                or counts != sorted(set(cast(list[int], counts)))
                or counts[0] != 0
            ):
                label = "GCG" if selected_pressure == Pressure.GCG else "soft-prompt"
                raise ValueError(f"Adv {label} manifest has an invalid token-count grid")
            points = {
                f"k{count}": (
                    input_dir / "points" / "p0" / "judged"
                    if count == 0
                    else input_dir / selected_pressure / "points" / f"k{count}" / "judged"
                )
                for count in counts
            }
        else:
            raise ValueError(f"unsupported histogram pressure {selected_pressure}")
    else:
        factors = _factors(method_factors, SUPPORTED_METHOD_FACTORS)
        points = {"upstream": input_dir}
    reports = {}
    population: pd.DataFrame | None = None
    for name, directory in points.items():
        scores, identity = _load_scores(
            directory,
            factors,
            sources,
            expected_concept_ids=concept_ids,
        )
        if population is not None and not population.equals(identity):
            raise ValueError("Adv proposal levels have different concept/prompt populations")
        population = identity
        reports[name] = scores
    if pipeline == "adv":
        for source, digest in sources.items():
            relative = str(Path(source).relative_to(input_dir))
            if relative != "manifest.json" and manifest["files"].get(relative) != digest:
                raise ValueError(f"Adv judged file hash differs from manifest: {relative}")
    assert population is not None
    edges = np.linspace(0, 2, bins + 1)
    root = Path(__file__).resolve().parents[2]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, check=True, capture_output=True, text=True
    ).stdout.strip()
    metadata: dict[str, Any] = {
        "schema_version": 1,
        "pipeline": pipeline,
        "sources": sources,
        "method_factors": factors,
        "plotting_revision": revision,
        "plotting_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": population["dataset_name"].iloc[0],
        "bin_edges": edges.tolist(),
        "bins": bins,
        "layout": {"rows": "method", "columns": list(AXES)},
        "bin_convention": "[left, right), final bin includes 2",
        "normalization": "percentage of concepts",
        "score_semantics": "raw judge relevance; no suppression inversion",
        "reports": {},
    }
    if pipeline == "adv":
        metadata["pressure"] = selected_pressure
    output_dir.mkdir(parents=True, exist_ok=True)
    for name, scores in reports.items():
        target = output_dir / name if pipeline == "adv" else output_dir
        target.mkdir(exist_ok=True)
        if pipeline == "upstream":
            suffix = ""
        elif selected_pressure == Pressure.GEPA:
            suffix = (
                "; p0 regenerated empty-suffix control"
                if name == "p0"
                else f"; {name[1:]} GEPA proposals"
            )
        elif selected_pressure in (Pressure.GCG, Pressure.SOFT_PROMPT):
            label = "GCG suffix tokens" if selected_pressure == Pressure.GCG else "soft suffix positions"
            suffix = (
                "; k0 shared no-adversary control"
                if name == "k0"
                else f"; {name[1:]} retained {label}"
            )
        else:
            raise ValueError(f"unsupported histogram pressure {selected_pressure}")
        title = (
            f"{'AxBench-Adv' if pipeline == 'adv' else 'AxBench'} — {metadata['dataset']}{suffix}"
        )
        summaries = _render(scores, edges, title, target / "score_histograms.png")
        scores.to_csv(target / "concept_scores.csv", index=False)
        pd.DataFrame(summaries).drop(
            columns=["counts", "percentages", "n_prompts_per_concept"]
        ).to_csv(target / "summary.csv", index=False)
        metadata["reports"][name] = summaries
    destination = output_dir / "metadata.json"
    temporary = output_dir / ".metadata.json.tmp"
    temporary.write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(destination)
    return destination
