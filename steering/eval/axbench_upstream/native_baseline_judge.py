"""Target-only AxBench judging for one native standalone baseline artifact."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, cast

import pandas as pd
import yaml
from omegaconf import DictConfig

from nanochat.atomic_io import atomic_write_bytes, atomic_write_text
from scripts.eval.axbench_upstream.run_evaluation import (
    EVALUATE_CONFIG,
    EVALUATE_SCRIPT,
    VENDOR_ROOT,
)
from steering.eval.axbench_upstream.native_baselines import (
    METHOD_FACTORS,
    METHODS,
    load_native_baseline_output,
)
from steering.eval.axbench_upstream.native_contract import sha256
from steering.eval.axbench_upstream.native_judge import FIELDS, SUMMARY, _attempt_accounting


def validate_native_baseline_judged(
    frame: pd.DataFrame, output: Path, *, method: str
) -> dict[str, dict[str, float]]:
    """Validate exact method/factor coverage and judge arithmetic."""

    if method not in METHODS:
        raise ValueError(f"unsupported native standalone judge method {method!r}")
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    expected_columns = [*frame.columns, *(f"{method}_{key}" for key in FIELDS)]
    if set(evaluated.columns) != set(expected_columns) or not evaluated.columns.is_unique:
        raise ValueError("native standalone judged columns differ")
    if list(evaluated.columns[: len(frame.columns)]) != list(frame.columns) or not evaluated.loc[
        :, frame.columns
    ].equals(frame):
        raise ValueError("native standalone judge changed its input")
    entries = [
        json.loads(line)
        for line in (output / "steering.jsonl").read_text().splitlines()
        if line.strip()
    ]
    concepts = sorted(frame["concept_id"].unique().tolist())
    if [entry.get("concept_id") for entry in entries] != concepts:
        raise ValueError("native standalone judged concept coverage differs")
    factors = METHOD_FACTORS[method]
    totals = {name: {factor: [] for factor in factors} for name in SUMMARY}
    for entry in entries:
        if (
            set(entry) != {"concept_id", "results"}
            or set(entry["results"]) != {"LMJudgeEvaluator"}
            or set(entry["results"]["LMJudgeEvaluator"]) != {method}
        ):
            raise ValueError("native standalone judge method/evaluator differs")
        result = entry["results"]["LMJudgeEvaluator"][method]
        selected = evaluated.loc[evaluated["concept_id"].eq(entry["concept_id"])]
        if result.get("factor") != list(factors):
            raise ValueError("native standalone judge factor grid differs")
        for suffix, field in FIELDS.items():
            values = result.get(field)
            if not isinstance(values, list) or len(values) != len(selected):
                raise ValueError(f"native standalone judge coverage differs: {field}")
            for value in values:
                if field.endswith("completions"):
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("native standalone judge completion is blank")
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 2
                ):
                    raise ValueError("native standalone judge rating outside [0,2]")
            if selected[f"{method}_{suffix}"].tolist() != values:
                raise ValueError("native standalone judge JSONL/Parquet disagreement")
        for summary, raw in SUMMARY.items():
            summaries = result.get(summary)
            if not isinstance(summaries, list) or len(summaries) != len(factors):
                raise ValueError("native standalone judge summary coverage differs")
            for factor, stated in zip(factors, summaries, strict=True):
                positions = selected["factor"].eq(factor).tolist()
                values = [value for value, keep in zip(result[raw], positions, strict=True) if keep]
                expected = sum(values) / len(values)
                if not math.isclose(cast(float, stated), expected, abs_tol=1e-12):
                    raise ValueError("native standalone judge summary disagrees with raw ratings")
                totals[summary][factor].extend(values)
        for row_index in range(len(selected)):
            ratings = [
                result[f"raw_{name}_ratings"][row_index]
                for name in ("relevance_concept", "relevance_instruction", "fluency")
            ]
            expected = 0.0 if 0 in ratings else 3 / sum(1 / value for value in ratings)
            if not math.isclose(
                result["raw_aggregated_ratings"][row_index], expected, abs_tol=1e-12
            ):
                raise ValueError("native standalone judge aggregate is not AxBench harmonic mean")
    return {
        name: {
            float(factor).hex(): sum(values) / len(values) for factor, values in by_factor.items()
        }
        for name, by_factor in totals.items()
    }


def _publish(
    *,
    frame: pd.DataFrame,
    method: str,
    output: Path,
    completed: Path,
    attempts: list[Path],
    identity: dict[str, Any],
) -> None:
    scores = validate_native_baseline_judged(frame, completed, method=method)
    reports = [
        {
            "path": str(path.relative_to(output)),
            "report": _attempt_accounting(path),
        }
        for path in attempts
    ]
    complete = all(item["report"] is not None for item in reports)
    calls = 0
    hits = 0
    for item in reports:
        report = cast(dict[str, int] | None, item["report"])
        if report is not None:
            calls += report["total_calls"]
            hits += report["total_cache_hits"]
    accounting = {
        "total_calls": calls if complete else None,
        "total_cache_hits": hits if complete else None,
        "reported_total_calls": calls,
        "reported_total_cache_hits": hits,
        "reported_noncache_calls": calls - hits,
        "accounting_complete": complete,
        "unreported_attempts": [item["path"] for item in reports if item["report"] is None],
        "semantics": "completed logical judge calls; provider retries and billable requests are not counted",
        "attempts": reports,
    }
    for name in ("steering.jsonl", "steering_data.parquet"):
        atomic_write_bytes((completed / name).read_bytes(), output / name)
    atomic_write_text(
        json.dumps(accounting, indent=2) + "\n", output / "judge_call_accounting.json"
    )
    receipt = {
        **identity,
        "scores_by_factor_hex": scores,
        "rows": len(frame),
        "baseline_requests": 0,
        "files": {
            name: sha256(output / name)
            for name in ("steering.jsonl", "steering_data.parquet", "judge_call_accounting.json")
        },
        "call_accounting": accounting,
    }
    atomic_write_text(
        json.dumps(receipt, indent=2) + "\n", output / "native_baseline_judge_receipt.json"
    )


def run_native_baseline_evaluation(cfg: DictConfig) -> None:
    """Judge one complete method artifact without any baseline-model request."""

    method = str(cfg.method)
    if method not in METHODS:
        raise ValueError(f"method must be one of {list(METHODS)}")
    input_path = Path(str(cfg.input_path)).resolve()
    manifest_path = Path(str(cfg.manifest_path)).resolve()
    output = Path(str(cfg.output_dir)).resolve()
    cache = Path(str(cfg.judge_cache)).resolve()
    if input_path.name != "steering_data.parquet":
        raise ValueError("vendor judge requires steering_data.parquet")
    if (
        cache.is_relative_to(output)
        or output.is_relative_to(cache)
        or input_path.is_relative_to(output)
    ):
        raise ValueError("judge input/cache/output must be disjoint")
    manifest, frame = load_native_baseline_output(manifest_path, input_path)
    if manifest["method"] != method:
        raise ValueError("native standalone judge method differs from manifest")
    expected_concepts = list(range(int(cfg.concept_lo), int(cfg.concept_hi)))
    if manifest["source"]["concept_ids"] != expected_concepts:
        raise ValueError("native standalone judge concept subset differs from manifest")
    if any(
        column.endswith("_steered_generation") and column != f"{method}_steered_generation"
        for column in frame.columns
    ):
        raise ValueError("native standalone judge refuses other model generations")
    identity = {
        "method": method,
        "factors": list(METHOD_FACTORS[method]),
        "concept_ids": expected_concepts,
        "input_sha256": sha256(input_path),
        "manifest_sha256": sha256(manifest_path),
        "judge_model": "openai/gpt-5.6-luna",
        "vendor_lock_sha256": sha256(VENDOR_ROOT / "uv.lock"),
        "evaluator_source_sha256": {
            str(path.relative_to(VENDOR_ROOT)): sha256(path)
            for path in (
                EVALUATE_SCRIPT,
                EVALUATE_CONFIG,
                VENDOR_ROOT / "axbench/evaluators/lm_judge.py",
                VENDOR_ROOT / "axbench/evaluators/prompt_templates.py",
                VENDOR_ROOT / "axbench/models/language_models.py",
            )
        },
        "num_workers": int(cfg.num_workers),
        "api_base_url": str(cfg.api_base_url),
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "native_baseline_judge_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("native standalone judge resume identity changed")
    else:
        if any(output.iterdir()):
            raise ValueError("native standalone judge refuses unbound existing output")
        atomic_write_text(json.dumps(identity, indent=2) + "\n", identity_path)
    receipt_path = output / "native_baseline_judge_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("native standalone judge receipt identity changed")
        if any(sha256(output / name) != digest for name, digest in receipt["files"].items()):
            raise ValueError("native standalone judge receipt file hashes changed")
        if validate_native_baseline_judged(frame, output, method=method) != receipt.get(
            "scores_by_factor_hex"
        ):
            raise ValueError("native standalone judge receipt scores changed")
        return
    attempts_dir = output / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    attempts = sorted(path for path in attempts_dir.iterdir() if path.is_dir())
    for prior in reversed(attempts):
        try:
            validate_native_baseline_judged(frame, prior, method=method)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        _publish(
            frame=frame,
            method=method,
            output=output,
            completed=prior,
            attempts=attempts,
            identity=identity,
        )
        return
    if not bool(cfg.cache_only) and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("live native standalone judge requires OPENROUTER_API_KEY")
    attempt = attempts_dir / f"attempt_{len(attempts) + 1:05d}"
    attempt.mkdir(exist_ok=False)
    attempts.append(attempt)
    loaded = yaml.safe_load(EVALUATE_CONFIG.read_text())
    if loaded["evaluate"].get("lm_model") != identity["judge_model"] or loaded["evaluate"].get(
        "steering_evaluators"
    ) != ["LMJudgeEvaluator"]:
        raise ValueError("native standalone judge configuration changed")
    loaded["evaluate"].update(models=[method], num_of_workers=int(cfg.num_workers))
    config_path = attempt / "native_baseline_evaluate.yaml"
    atomic_write_text(yaml.safe_dump(loaded, sort_keys=False), config_path)
    command = [
        "uv",
        "run",
        "--project",
        str(VENDOR_ROOT),
        "--frozen",
        "python",
        str(EVALUATE_SCRIPT),
        "--config",
        str(config_path),
        "--mode",
        "steering",
        "--overwrite_inference_dump_dir",
        str(input_path.parent),
        "--overwrite_evaluate_dump_dir",
        str(attempt),
        "--judge_cache",
        str(cache),
        "--cache_only",
        str(bool(cfg.cache_only)).lower(),
        "--api_base_url",
        str(cfg.api_base_url),
    ]
    error: subprocess.CalledProcessError | None = None
    with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
        try:
            subprocess.run(command, cwd=VENDOR_ROOT, check=True, stdout=stdout, stderr=stderr)
        except subprocess.CalledProcessError as exc:
            error = exc
    if error is not None:
        atomic_write_text(
            json.dumps({"exit_code": error.returncode}) + "\n", attempt / "failure.json"
        )
        try:
            validate_native_baseline_judged(frame, attempt, method=method)
        except (OSError, ValueError, KeyError, TypeError):
            raise error from None
    _publish(
        frame=frame,
        method=method,
        output=output,
        completed=attempt,
        attempts=attempts,
        identity=identity,
    )


__all__ = (
    "FIELDS",
    "SUMMARY",
    "run_native_baseline_evaluation",
    "validate_native_baseline_judged",
)
