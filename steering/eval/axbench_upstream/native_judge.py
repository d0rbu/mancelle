"""Target-only native decoder evaluation through the unchanged AxBench judge."""

from __future__ import annotations

import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd
import yaml
from omegaconf import DictConfig

from nanochat.atomic_io import atomic_write_bytes, atomic_write_text
from scripts.eval.axbench_upstream.run_evaluation import (
    EVALUATE_CONFIG,
    EVALUATE_SCRIPT,
    VENDOR_ROOT,
)
from steering.eval.axbench_upstream.native_contract import METHOD, load_native_output, sha256

FIELDS = {
    "LMJudgeEvaluator": "raw_aggregated_ratings",
    "LMJudgeEvaluator_relevance_concept_ratings": "raw_relevance_concept_ratings",
    "LMJudgeEvaluator_relevance_concept_completions": "relevance_concept_completions",
    "LMJudgeEvaluator_relevance_instruction_ratings": "raw_relevance_instruction_ratings",
    "LMJudgeEvaluator_relevance_instruction_completions": "relevance_instruction_completions",
    "LMJudgeEvaluator_fluency_ratings": "raw_fluency_ratings",
    "LMJudgeEvaluator_fluency_completions": "fluency_completions",
}
SUMMARY = {
    "lm_judge_rating": "raw_aggregated_ratings",
    "relevance_concept_ratings": "raw_relevance_concept_ratings",
    "relevance_instruction_ratings": "raw_relevance_instruction_ratings",
    "fluency_ratings": "raw_fluency_ratings",
}


def validate_native_judged(frame: pd.DataFrame, output: Path) -> dict[str, Any]:
    """Prove preserved inputs, exact raw/summary coverage and metric arithmetic."""
    evaluated = pd.read_parquet(output / "steering_data.parquet")
    expected = [*frame.columns, *(f"{METHOD}_{key}" for key in FIELDS)]
    if set(evaluated.columns) != set(expected) or not evaluated.columns.is_unique:
        raise ValueError("native judged columns differ")
    if list(evaluated.columns[: len(frame.columns)]) != list(frame.columns) or not evaluated.loc[
        :, frame.columns
    ].equals(frame):
        raise ValueError("native judge changed its input")
    entries = [
        json.loads(line)
        for line in (output / "steering.jsonl").read_text().splitlines()
        if line.strip()
    ]
    concepts = sorted(frame.concept_id.unique().tolist())
    if [entry.get("concept_id") for entry in entries] != concepts:
        raise ValueError("native judged concept coverage differs")
    for entry in entries:
        if (
            set(entry) != {"concept_id", "results"}
            or set(entry["results"]) != {"LMJudgeEvaluator"}
            or set(entry["results"]["LMJudgeEvaluator"]) != {METHOD}
        ):
            raise ValueError("native judge method/evaluator differs")
        result = entry["results"]["LMJudgeEvaluator"][METHOD]
        selected = evaluated.loc[evaluated.concept_id.eq(entry["concept_id"])]
        if result.get("factor") != [1.0]:
            raise ValueError("native judge factor differs")
        for suffix, field in FIELDS.items():
            values = result.get(field)
            if not isinstance(values, list) or len(values) != len(selected):
                raise ValueError(f"native judge coverage differs: {field}")
            for value in values:
                if field.endswith("completions"):
                    if not isinstance(value, str) or not value.strip():
                        raise ValueError("native judge completion is blank")
                elif (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(value)
                    or not 0 <= value <= 2
                ):
                    raise ValueError("native judge rating outside [0,2]")
            if selected[f"{METHOD}_{suffix}"].tolist() != values:
                raise ValueError("native judge JSONL/Parquet disagreement")
        for summary, raw in SUMMARY.items():
            if len(result.get(summary, [])) != 1 or not math.isclose(
                result[summary][0], sum(result[raw]) / len(selected), abs_tol=1e-12
            ):
                raise ValueError("native judge summary disagrees with raw ratings")
        for i in range(len(selected)):
            ratings = [
                result[f"raw_{field}_ratings"][i]
                for field in ("relevance_concept", "relevance_instruction", "fluency")
            ]
            expected_score = 0.0 if 0 in ratings else 3 / sum(1 / value for value in ratings)
            if not math.isclose(result["raw_aggregated_ratings"][i], expected_score, abs_tol=1e-12):
                raise ValueError("native judge aggregate is not AxBench harmonic mean")
    return {
        name: float(evaluated[f"{METHOD}_{suffix}"].mean())
        for name, suffix in (
            ("concept", "LMJudgeEvaluator_relevance_concept_ratings"),
            ("instruction", "LMJudgeEvaluator_relevance_instruction_ratings"),
            ("fluency", "LMJudgeEvaluator_fluency_ratings"),
            ("aggregate", "LMJudgeEvaluator"),
        )
    }


def _attempt_accounting(path: Path) -> dict[str, int] | None:
    report = path / "judge_call_accounting.json"
    if not report.exists():
        return None
    value = json.loads(report.read_text())
    if (set(value) != {"total_calls", "total_cache_hits"}
            or any(type(item) is not int or item < 0 for item in value.values())
            or value["total_cache_hits"] > value["total_calls"]):
        raise ValueError(f"invalid native judge attempt accounting: {report}")
    return value


def _publish_judged_attempt(
    frame: pd.DataFrame, output: Path, completed: Path,
    attempts: list[Path], identity: dict[str, Any], scores: dict[str, Any],
) -> None:
    reports: list[dict[str, Any]] = [{"path": str(path.relative_to(output)),
                "report": _attempt_accounting(path)} for path in attempts]
    complete_accounting = all(item["report"] is not None for item in reports)
    calls = sum(item["report"]["total_calls"] for item in reports if item["report"] is not None)
    hits = sum(item["report"]["total_cache_hits"] for item in reports if item["report"] is not None)
    accounting = {
        "total_calls": calls if complete_accounting else None,
        "total_cache_hits": hits if complete_accounting else None,
        "reported_total_calls": calls, "reported_total_cache_hits": hits,
        "reported_noncache_calls": calls - hits,
        "accounting_complete": complete_accounting,
        "unreported_attempts": [item["path"] for item in reports if item["report"] is None],
        "semantics": "completed logical judge calls; provider retries and billable requests are not counted",
        "attempts": reports,
    }
    for name in ("steering.jsonl", "steering_data.parquet"):
        atomic_write_bytes((completed / name).read_bytes(), output / name)
    atomic_write_text(json.dumps(accounting, indent=2) + "\n", output / "judge_call_accounting.json")
    receipt = {
        **identity, "scores": scores, "rows": len(frame), "baseline_requests": 0,
        "files": {name: sha256(output / name) for name in (
            "steering.jsonl", "steering_data.parquet", "judge_call_accounting.json")},
        "call_accounting": accounting,
    }
    atomic_write_text(json.dumps(receipt, indent=2) + "\n", output / "native_judge_receipt.json")


def run_native_evaluation(cfg: DictConfig) -> None:
    """Bind a resumable vendor execution to immutable native inputs and config."""
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
    manifest, frame = load_native_output(manifest_path, output_path=input_path)
    if manifest["source"]["max_concepts"] != int(cfg.max_concepts):
        raise ValueError("native judge concept count differs from source manifest")
    if any(
        column.endswith("_steered_generation") and column != f"{METHOD}_steered_generation"
        for column in frame.columns
    ):
        raise ValueError("native judge refuses other model generations")
    identity = {
        "method": METHOD,
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
        "max_concepts": int(cfg.max_concepts),
    }
    output.mkdir(parents=True, exist_ok=True)
    identity_path = output / "native_judge_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text()) != identity:
            raise ValueError("native judge resume identity changed")
    else:
        if any(output.iterdir()):
            raise ValueError("native judge refuses unbound existing output")
        atomic_write_text(json.dumps(identity, indent=2) + "\n", identity_path)
    receipt_path = output / "native_judge_receipt.json"
    if receipt_path.exists():
        receipt = json.loads(receipt_path.read_text())
        if any(receipt.get(key) != value for key, value in identity.items()):
            raise ValueError("native judge receipt identity changed")
        if set(receipt.get("files", {})) != {
            "steering.jsonl",
            "steering_data.parquet",
            "judge_call_accounting.json",
        } or any(sha256(output / name) != digest for name, digest in receipt["files"].items()):
            raise ValueError("native judge receipt file hashes changed")
        if validate_native_judged(frame, output) != receipt.get("scores"):
            raise ValueError("native judge receipt scores changed")
        return
    attempts_dir = output / "attempts"
    attempts_dir.mkdir(exist_ok=True)
    attempts = sorted(path for path in attempts_dir.iterdir() if path.is_dir())
    # A previous invocation may have finished judging and then failed during plotting
    # or receipt publication. Validate its durable result before any provider access.
    for prior in reversed(attempts):
        try:
            scores = validate_native_judged(frame, prior)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        _publish_judged_attempt(frame, output, prior, attempts, identity, scores)
        return
    if not cfg.cache_only and not os.environ.get("OPENROUTER_API_KEY", "").strip():
        raise ValueError("live native judge requires OPENROUTER_API_KEY")
    # Never trust vendor partial-resume state: it precedes JSONL/Parquet commits.
    # Reevaluate all logical rows in a new attempt, reusing the durable judge cache.
    attempt = attempts_dir / f"attempt_{len(attempts) + 1:05d}"
    attempt.mkdir(exist_ok=False)
    attempts.append(attempt)
    loaded = yaml.safe_load(EVALUATE_CONFIG.read_text())
    expected = {"lm_model": identity["judge_model"], "steering_evaluators": ["LMJudgeEvaluator"]}
    if any(loaded["evaluate"].get(key) != value for key, value in expected.items()):
        raise ValueError("native judge configuration changed its judge model/evaluator")
    loaded["evaluate"].update(models=[METHOD], num_of_workers=int(cfg.num_workers))
    config_path = attempt / "native_evaluate.yaml"
    atomic_write_text(yaml.safe_dump(loaded, sort_keys=False), config_path)
    command = [
        "uv", "run", "--project", str(VENDOR_ROOT), "--frozen", "python", str(EVALUATE_SCRIPT),
        "--config", str(config_path), "--mode", "steering",
        "--overwrite_inference_dump_dir", str(input_path.parent),
        "--overwrite_evaluate_dump_dir", str(attempt), "--judge_cache", str(cache),
        "--cache_only", str(bool(cfg.cache_only)).lower(), "--api_base_url", str(cfg.api_base_url),
    ]
    command_error: subprocess.CalledProcessError | None = None
    with (attempt / "stdout.log").open("w") as stdout, (attempt / "stderr.log").open("w") as stderr:
        try:
            subprocess.run(command, cwd=VENDOR_ROOT, check=True, stdout=stdout, stderr=stderr)
        except subprocess.CalledProcessError as exc:
            command_error = exc
    if command_error is not None:
        atomic_write_text(json.dumps({"exit_code": command_error.returncode}) + "\n", attempt / "failure.json")
    try:
        scores = validate_native_judged(frame, attempt)
    except (OSError, ValueError, KeyError, TypeError):
        if command_error is not None:
            raise command_error from None
        raise
    _publish_judged_attempt(frame, output, attempt, attempts, identity, scores)
