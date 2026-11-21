"""Versioned artifact contract shared by harmfulness evaluation surfaces."""

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import yaml
from loguru import logger

from nanochat.atomic_io import atomic_write_text
from steering._common.strict_json import string_keyed_mapping
from steering.eval.compacted_eval_loading import EvaluationConditioningMode
from steering.eval.harmfulness_generation import validate_generation_provenance
from steering.eval.harmfulness_judge import (
    COMPLIANCE_CACHE_NAMESPACE,
    COMPLIANCE_RESULT_SCHEMA_VERSION,
    COMPLIANCE_SYSTEM_PROMPT,
    COMPLIANCE_USER_TEMPLATE,
    HARM_CATEGORIES,
    HARMFULNESS_RESULT_SCHEMA_VERSION,
    JUDGE_SYSTEM_PROMPT,
    JUDGE_USER_TEMPLATE,
    ComplianceJudgeResult,
    compute_compliance_metrics,
)

HARMFULNESS_EVAL_SCHEMA_VERSION = 4
HARMFULNESS_RESPONSES_SCHEMA_VERSION = 2

_PROVENANCE_KEYS = {
    "model",
    "harmfulness_prompt_sha256",
    "harmfulness_result_schema_version",
    "compliance_prompt_sha256",
    "compliance_result_schema_version",
    "compliance_cache_namespace",
    "ordered_pairs_sha256",
}
_COMPLIANCE_KEYS = {
    "compliance_rate",
    "compliance_num_scored",
    "compliance_num_failed",
    "compliance_judgments",
}
_COMPLIANCE_RECORD_KEYS = {
    "pair_index",
    "instruction",
    "response",
    "status",
    "helpfulness_score",
    "is_refusal",
    "reasoning",
    "is_compliant",
    "error",
}
_CONDITIONED_KEYS = {
    "conditioning_mode",
    "eval_name",
    "feature_vocab",
    "context_cache",
    "coverage",
}
_UNCONDITIONED_KEYS = {"conditioning_mode", "eval_name"}
_FEATURE_VOCAB_KEYS = {"n_sae_features", "vocab_size", "sha256"}
_CONTEXT_CACHE_KEYS = {"manifest_path", "identity", "cached_rows"}
_CACHE_IDENTITY_KEYS = {
    "model_id",
    "sae_id",
    "layer",
    "pooling",
    "feature_selection_hash",
}
_COVERAGE_KEYS = {
    "expected_prompts",
    "expected_unique_keys",
    "extra_cached_rows",
}
_RESPONSES_ARTIFACT_KEYS = {
    "harmfulness_responses_schema_version",
    "conditioning",
    "generation",
    "responses",
}
_SHA256_RE = re.compile(r"[0-9a-f]{64}")


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def harmfulness_prompt_sha256() -> str:
    """Hash the complete invariant harmfulness prompt templates."""
    return _sha256_text(f"{JUDGE_SYSTEM_PROMPT}\0{JUDGE_USER_TEMPLATE}")


def compliance_prompt_sha256() -> str:
    """Hash the complete versioned compliance prompt templates."""
    return _sha256_text(f"{COMPLIANCE_SYSTEM_PROMPT}\0{COMPLIANCE_USER_TEMPLATE}")


def ordered_pairs_sha256(pairs: list[dict[str, Any]]) -> str:
    """Hash the ordered inputs that determine every persisted metric."""
    canonical_pairs: list[dict[str, str]] = []
    for index, pair in enumerate(pairs):
        if not isinstance(pair, dict):
            raise ValueError(
                f"Harmfulness pair {index} must be a mapping, got {type(pair).__name__}"
            )
        canonical: dict[str, str] = {}
        for key in ("prompt", "response", "pool", "category"):
            value = pair.get(key)
            if not isinstance(value, str):
                raise ValueError(
                    f"Harmfulness pair {index} field {key!r} must be a string, "
                    f"got {type(value).__name__}"
                )
            canonical[key] = value
        if canonical["pool"] not in {"harmful", "benign"}:
            raise ValueError(f"Harmfulness pair {index} has invalid pool {canonical['pool']!r}")
        canonical_pairs.append(canonical)
    rendered = json.dumps(
        canonical_pairs,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(rendered)


def context_cache_manifest_sha256(manifest: dict[str, Any]) -> str:
    """Hash the exact manifest mapping loaded by the context provider."""

    canonical = json.dumps(
        manifest,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return _sha256_text(canonical)


def build_judge_provenance(
    *,
    model: str,
    pairs: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build complete judge and input provenance for resume validation."""
    if not isinstance(model, str) or not model or model != model.strip():
        raise ValueError("Harmfulness judge model must be a non-empty, stripped string")
    return {
        "model": model,
        "harmfulness_prompt_sha256": harmfulness_prompt_sha256(),
        "harmfulness_result_schema_version": HARMFULNESS_RESULT_SCHEMA_VERSION,
        "compliance_prompt_sha256": compliance_prompt_sha256(),
        "compliance_result_schema_version": COMPLIANCE_RESULT_SCHEMA_VERSION,
        "compliance_cache_namespace": COMPLIANCE_CACHE_NAMESPACE,
        "ordered_pairs_sha256": ordered_pairs_sha256(pairs),
    }


def build_harmfulness_payload(
    *,
    metrics: dict[str, Any],
    compliance_metrics: dict[str, Any],
    model: str,
    pairs: list[dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build and validate one standalone or progression result payload."""
    benign = string_keyed_mapping(metrics.get("benign_prompts"))
    if benign is None:
        raise ValueError("Harmfulness metrics missing benign_prompts mapping")
    overlap = set(benign) & set(compliance_metrics)
    if overlap:
        raise ValueError(f"Compliance fields would overwrite harmfulness fields: {sorted(overlap)}")

    payload: dict[str, Any] = {
        "harmfulness_eval_schema_version": HARMFULNESS_EVAL_SCHEMA_VERSION,
    }
    if config is not None:
        payload["config"] = config
    payload["judge_provenance"] = build_judge_provenance(
        model=model,
        pairs=pairs,
    )
    for key in ("overall", "harmful_prompts", "benign_prompts", "per_category"):
        if key not in metrics:
            raise ValueError(f"Harmfulness metrics missing required key {key!r}")
        payload[key] = metrics[key]
    payload["benign_prompts"] = {**benign, **compliance_metrics}

    validate_harmfulness_payload(payload, model=model, pairs=pairs)
    return payload


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be a mapping, got {type(value).__name__}")
    return value


def _require_nonnegative_int(value: Any, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{path} must be a non-negative integer, got {value!r}")
    return value


def _require_positive_int(value: Any, path: str) -> int:
    parsed = _require_nonnegative_int(value, path)
    if parsed == 0:
        raise ValueError(f"{path} must be a positive integer, got {value!r}")
    return parsed


def _require_nonempty_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value or value != value.strip():
        raise ValueError(f"{path} must be a non-empty, stripped string, got {value!r}")
    return value


def _require_exact_keys(
    value: dict[str, Any],
    expected: set[str],
    path: str,
) -> None:
    if set(value) != expected:
        raise ValueError(f"{path} keys must be exactly {sorted(expected)}, got {sorted(value)}")


def _validate_conditioning_provenance(config: dict[str, Any]) -> None:
    """Validate the report discriminator and its complete mode-specific block."""

    conditioning = _require_mapping(
        config.get("conditioning"),
        "config.conditioning",
    )
    raw_mode = conditioning.get("conditioning_mode")
    try:
        mode = EvaluationConditioningMode(raw_mode)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "config.conditioning.conditioning_mode must be one of "
            f"{[item.value for item in EvaluationConditioningMode]}, got {raw_mode!r}"
        ) from exc
    _require_nonempty_string(
        conditioning.get("eval_name"),
        "config.conditioning.eval_name",
    )

    if mode is EvaluationConditioningMode.UNCONDITIONED:
        _require_exact_keys(
            conditioning,
            _UNCONDITIONED_KEYS,
            "config.conditioning",
        )
        return

    _require_exact_keys(
        conditioning,
        _CONDITIONED_KEYS,
        "config.conditioning",
    )
    feature_vocab_raw = conditioning["feature_vocab"]
    if feature_vocab_raw is not None:
        feature_vocab = _require_mapping(
            feature_vocab_raw,
            "config.conditioning.feature_vocab",
        )
        _require_exact_keys(
            feature_vocab,
            _FEATURE_VOCAB_KEYS,
            "config.conditioning.feature_vocab",
        )
        n_sae_features = _require_positive_int(
            feature_vocab["n_sae_features"],
            "config.conditioning.feature_vocab.n_sae_features",
        )
        vocab_size = _require_positive_int(
            feature_vocab["vocab_size"],
            "config.conditioning.feature_vocab.vocab_size",
        )
        if vocab_size >= n_sae_features:
            raise ValueError(
                "config.conditioning.feature_vocab must describe a genuine "
                f"compaction, got vocab_size={vocab_size}, "
                f"n_sae_features={n_sae_features}"
            )
        fingerprint = feature_vocab["sha256"]
        if not isinstance(fingerprint, str) or not _SHA256_RE.fullmatch(fingerprint):
            raise ValueError(
                "config.conditioning.feature_vocab.sha256 must be 64 lowercase "
                "hexadecimal characters"
            )

    context_cache = _require_mapping(
        conditioning["context_cache"],
        "config.conditioning.context_cache",
    )
    _require_exact_keys(
        context_cache,
        _CONTEXT_CACHE_KEYS,
        "config.conditioning.context_cache",
    )
    _require_nonempty_string(
        context_cache["manifest_path"],
        "config.conditioning.context_cache.manifest_path",
    )
    cached_rows = _require_nonnegative_int(
        context_cache["cached_rows"],
        "config.conditioning.context_cache.cached_rows",
    )
    identity = _require_mapping(
        context_cache["identity"],
        "config.conditioning.context_cache.identity",
    )
    _require_exact_keys(
        identity,
        _CACHE_IDENTITY_KEYS,
        "config.conditioning.context_cache.identity",
    )
    for key in ("model_id", "sae_id", "pooling"):
        _require_nonempty_string(
            identity[key],
            f"config.conditioning.context_cache.identity.{key}",
        )
    _require_nonnegative_int(
        identity["layer"],
        "config.conditioning.context_cache.identity.layer",
    )
    selection_hash = identity["feature_selection_hash"]
    if not isinstance(selection_hash, str) or not _SHA256_RE.fullmatch(selection_hash):
        raise ValueError(
            "config.conditioning.context_cache.identity.feature_selection_hash "
            "must be 64 lowercase hexadecimal characters"
        )

    coverage = _require_mapping(
        conditioning["coverage"],
        "config.conditioning.coverage",
    )
    _require_exact_keys(
        coverage,
        _COVERAGE_KEYS,
        "config.conditioning.coverage",
    )
    expected_prompts = _require_positive_int(
        coverage["expected_prompts"],
        "config.conditioning.coverage.expected_prompts",
    )
    expected_unique_keys = _require_positive_int(
        coverage["expected_unique_keys"],
        "config.conditioning.coverage.expected_unique_keys",
    )
    extra_cached_rows = _require_nonnegative_int(
        coverage["extra_cached_rows"],
        "config.conditioning.coverage.extra_cached_rows",
    )
    if expected_unique_keys > expected_prompts:
        raise ValueError(
            "config.conditioning.coverage.expected_unique_keys cannot exceed expected_prompts"
        )
    if cached_rows != expected_unique_keys + extra_cached_rows:
        raise ValueError(
            "config.conditioning context-cache cardinality mismatch: "
            f"cached_rows={cached_rows}, expected_unique_keys={expected_unique_keys}, "
            f"extra_cached_rows={extra_cached_rows}"
        )


def validate_conditioning_provenance(conditioning: Any) -> dict[str, Any]:
    """Validate and return one complete conditioning discriminator block."""

    config = {"conditioning": conditioning}
    _validate_conditioning_provenance(config)
    assert isinstance(conditioning, dict)
    return conditioning


def parse_harmfulness_responses_artifact(
    payload: Any,
) -> tuple[list[Any], dict[str, Any], dict[str, Any]]:
    """Validate a response artifact and return its responses and provenance."""

    root = _require_mapping(payload, "harmfulness response artifact")
    schema_version = root.get("harmfulness_responses_schema_version")
    if schema_version == 1:
        raise ValueError(
            "schema-v1 harmfulness response artifacts are no longer valid; "
            "regenerate responses with the current evaluator"
        )
    _require_exact_keys(
        root,
        _RESPONSES_ARTIFACT_KEYS,
        "harmfulness response artifact",
    )
    schema_version = root["harmfulness_responses_schema_version"]
    if schema_version != HARMFULNESS_RESPONSES_SCHEMA_VERSION:
        raise ValueError(
            "harmfulness_responses_schema_version mismatch: expected "
            f"{HARMFULNESS_RESPONSES_SCHEMA_VERSION}, got {schema_version!r}"
        )
    responses = root["responses"]
    if not isinstance(responses, list):
        raise ValueError(
            "harmfulness response artifact.responses must be a list, got "
            f"{type(responses).__name__}"
        )
    conditioning = validate_conditioning_provenance(root["conditioning"])
    if conditioning["eval_name"] != "harmfulness_eval":
        raise ValueError(
            "harmfulness response artifact conditioning.eval_name must be "
            f"'harmfulness_eval', got {conditioning['eval_name']!r}"
        )
    generation = validate_generation_provenance(root["generation"])
    return responses, conditioning, generation


def build_harmfulness_responses_artifact(
    *,
    responses: list[dict[str, object]],
    conditioning: dict[str, Any],
    generation: dict[str, Any],
) -> dict[str, Any]:
    """Build a versioned response artifact bound to generation conditioning."""

    artifact = {
        "harmfulness_responses_schema_version": (HARMFULNESS_RESPONSES_SCHEMA_VERSION),
        "conditioning": conditioning,
        "generation": generation,
        "responses": responses,
    }
    parse_harmfulness_responses_artifact(artifact)
    return artifact


def _require_rate(value: Any, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric, got {value!r}")
    rate = float(value)
    if not math.isfinite(rate) or not 0.0 <= rate <= 1.0:
        raise ValueError(f"{path} must be finite and in [0, 1], got {value!r}")
    return rate


def _require_mean_score(value: Any, path: str, *, num_scored: int) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be numeric, got {value!r}")
    score = float(value)
    expected_range = (1.0, 5.0) if num_scored else (0.0, 0.0)
    if not math.isfinite(score) or not expected_range[0] <= score <= expected_range[1]:
        raise ValueError(
            f"{path} must be finite and in [{expected_range[0]}, {expected_range[1]}] "
            f"when num_scored={num_scored}, got {value!r}"
        )
    return score


def _validate_category_metrics(payload: dict[str, Any]) -> None:
    per_category = _require_mapping(payload.get("per_category"), "per_category")
    if set(per_category) != set(HARM_CATEGORIES):
        raise ValueError("per_category keys must exactly match the 19-category harm taxonomy")
    for category, value in per_category.items():
        _require_rate(value, f"per_category.{category}")


def _validate_harmfulness_counts(
    payload: dict[str, Any],
    pairs: list[dict[str, Any]],
) -> None:
    expected_counts = {
        "overall": len(pairs),
        "harmful_prompts": sum(pair["pool"] == "harmful" for pair in pairs),
        "benign_prompts": sum(pair["pool"] == "benign" for pair in pairs),
    }
    for section_name, expected_count in expected_counts.items():
        section = _require_mapping(payload.get(section_name), section_name)
        num_scored = _require_nonnegative_int(
            section.get("num_scored"),
            f"{section_name}.num_scored",
        )
        num_failed = _require_nonnegative_int(
            section.get("num_failed"),
            f"{section_name}.num_failed",
        )
        if num_scored + num_failed != expected_count:
            raise ValueError(
                f"{section_name} count mismatch: num_scored={num_scored} + "
                f"num_failed={num_failed} != expected pairs={expected_count}"
            )
        _require_mean_score(
            section.get("mean_score"),
            f"{section_name}.mean_score",
            num_scored=num_scored,
        )
        for rate_key in ("harmfulness_rate", "safe_response_rate"):
            _require_rate(section.get(rate_key), f"{section_name}.{rate_key}")
    harmful = _require_mapping(payload["harmful_prompts"], "harmful_prompts")
    asr = _require_rate(harmful.get("asr"), "harmful_prompts.asr")
    harmfulness_rate = _require_rate(
        harmful.get("harmfulness_rate"),
        "harmful_prompts.harmfulness_rate",
    )
    if asr != harmfulness_rate:
        raise ValueError("harmful_prompts.asr must equal harmfulness_rate")
    _validate_category_metrics(payload)


def _validate_compliance_summary(benign: dict[str, Any]) -> None:
    missing_compliance = sorted(_COMPLIANCE_KEYS - set(benign))
    if missing_compliance:
        raise ValueError(f"benign_prompts missing compliance fields {missing_compliance}")
    _require_rate(
        benign.get("compliance_rate"),
        "benign_prompts.compliance_rate",
    )
    _require_nonnegative_int(
        benign.get("compliance_num_scored"),
        "benign_prompts.compliance_num_scored",
    )
    _require_nonnegative_int(
        benign.get("compliance_num_failed"),
        "benign_prompts.compliance_num_failed",
    )


def _compliance_results_from_records(
    records: Any,
) -> list[ComplianceJudgeResult | None]:
    if not isinstance(records, list):
        raise ValueError("benign_prompts.compliance_judgments must be a list")
    results: list[ComplianceJudgeResult | None] = []
    for index, raw_record in enumerate(records):
        record = _require_mapping(
            raw_record,
            f"benign_prompts.compliance_judgments[{index}]",
        )
        if set(record) != _COMPLIANCE_RECORD_KEYS:
            rendered_keys = sorted(repr(key) for key in record)
            raise ValueError(
                f"Compliance record {index} keys must be exactly "
                f"{sorted(_COMPLIANCE_RECORD_KEYS)}, got {rendered_keys}"
            )
        pair_index = record["pair_index"]
        if isinstance(pair_index, bool) or not isinstance(pair_index, int) or pair_index != index:
            raise ValueError(f"Compliance record {index} has pair_index={record['pair_index']!r}")
        for field_name in ("instruction", "response"):
            if not isinstance(record[field_name], str):
                raise ValueError(
                    f"Compliance record {index} has non-string {field_name} {record[field_name]!r}"
                )
        status = record["status"]
        if status == "failed":
            expected_nulls = {
                "helpfulness_score",
                "is_refusal",
                "reasoning",
                "is_compliant",
            }
            if any(record[key] is not None for key in expected_nulls):
                raise ValueError(f"Failed compliance record {index} has non-null result fields")
            if record["error"] != "judge_retries_exhausted":
                raise ValueError(
                    f"Failed compliance record {index} has invalid error {record['error']!r}"
                )
            results.append(None)
            continue
        if status != "scored":
            raise ValueError(f"Compliance record {index} has invalid status {status!r}")
        if record["error"] is not None:
            raise ValueError(f"Scored compliance record {index} must have error=null")
        score = record["helpfulness_score"]
        refusal = record["is_refusal"]
        reasoning = record["reasoning"]
        is_compliant = record["is_compliant"]
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"Compliance record {index} has invalid helpfulness_score {score!r}")
        if not isinstance(refusal, bool):
            raise ValueError(f"Compliance record {index} has invalid is_refusal {refusal!r}")
        if not isinstance(reasoning, str) or not reasoning.strip():
            raise ValueError(f"Compliance record {index} has invalid reasoning {reasoning!r}")
        if not isinstance(is_compliant, bool):
            raise ValueError(f"Compliance record {index} has invalid is_compliant {is_compliant!r}")
        expected_compliant = not refusal
        if is_compliant is not expected_compliant:
            raise ValueError(
                f"Compliance record {index} has is_compliant={is_compliant!r}, "
                f"expected {expected_compliant!r}"
            )
        results.append(
            ComplianceJudgeResult(
                helpfulness_score=score,
                is_refusal=refusal,
                reasoning=reasoning,
            )
        )
    return results


def validate_harmfulness_payload(
    payload: Any,
    *,
    model: str,
    pairs: list[dict[str, Any]],
) -> None:
    """Validate provenance, cardinality, and recomputable compliance fields."""
    root = _require_mapping(payload, "harmfulness result")
    if root.get("harmfulness_eval_schema_version") != HARMFULNESS_EVAL_SCHEMA_VERSION:
        raise ValueError(
            "harmfulness_eval_schema_version mismatch: expected "
            f"{HARMFULNESS_EVAL_SCHEMA_VERSION}, got "
            f"{root.get('harmfulness_eval_schema_version')!r}"
        )

    if "config" in root:
        config = _require_mapping(root["config"], "config")
        _validate_conditioning_provenance(config)
        if "generation" not in config:
            raise ValueError("config.generation is required for standalone harmfulness results")
        validate_generation_provenance(config["generation"])

    provenance = _require_mapping(root.get("judge_provenance"), "judge_provenance")
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError(
            f"judge_provenance keys must be exactly {sorted(_PROVENANCE_KEYS)}, "
            f"got {sorted(provenance)}"
        )
    expected_provenance = build_judge_provenance(model=model, pairs=pairs)
    if provenance != expected_provenance:
        differing = sorted(
            key for key in _PROVENANCE_KEYS if provenance.get(key) != expected_provenance.get(key)
        )
        raise ValueError(
            "judge_provenance mismatch for fields "
            f"{differing}: expected current model/prompt/schema/pairs"
        )

    _validate_harmfulness_counts(root, pairs)
    benign = _require_mapping(root.get("benign_prompts"), "benign_prompts")
    _validate_compliance_summary(benign)

    benign_pairs = [pair for pair in pairs if pair["pool"] == "benign"]
    records = benign.get("compliance_judgments")
    results = _compliance_results_from_records(records)
    if len(results) != len(benign_pairs):
        raise ValueError(
            "Compliance record cardinality mismatch: "
            f"{len(results)} records != {len(benign_pairs)} benign pairs"
        )
    recomputed = compute_compliance_metrics(benign_pairs, results)
    actual_compliance = {key: benign.get(key) for key in _COMPLIANCE_KEYS}
    if actual_compliance != recomputed:
        differing = sorted(
            key for key in _COMPLIANCE_KEYS if actual_compliance.get(key) != recomputed.get(key)
        )
        raise ValueError(f"Benign compliance aggregate/record mismatch for fields {differing}")


def validate_harmfulness_contract(
    payload: Any,
    *,
    model: str,
) -> None:
    """Validate a persisted seed before its source pairs are regenerated.

    Progression scheduling has the result payload and current evaluation spec,
    but intentionally has not loaded a model or regenerated responses. This
    validator checks every static judge contract field and all internally
    recomputable cardinality/compliance invariants. The progression-specific
    caller separately checks its evaluation-spec digest.
    """
    root = _require_mapping(payload, "harmfulness result")
    if root.get("harmfulness_eval_schema_version") != HARMFULNESS_EVAL_SCHEMA_VERSION:
        raise ValueError(
            "harmfulness_eval_schema_version mismatch: expected "
            f"{HARMFULNESS_EVAL_SCHEMA_VERSION}, got "
            f"{root.get('harmfulness_eval_schema_version')!r}"
        )
    provenance = _require_mapping(root.get("judge_provenance"), "judge_provenance")
    if set(provenance) != _PROVENANCE_KEYS:
        raise ValueError("judge_provenance does not match the current exact schema")
    expected_static = {
        "model": model,
        "harmfulness_prompt_sha256": harmfulness_prompt_sha256(),
        "harmfulness_result_schema_version": HARMFULNESS_RESULT_SCHEMA_VERSION,
        "compliance_prompt_sha256": compliance_prompt_sha256(),
        "compliance_result_schema_version": COMPLIANCE_RESULT_SCHEMA_VERSION,
        "compliance_cache_namespace": COMPLIANCE_CACHE_NAMESPACE,
    }
    for key, expected in expected_static.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"judge_provenance.{key} mismatch: expected {expected!r}, "
                f"got {provenance.get(key)!r}"
            )
    pair_digest = provenance.get("ordered_pairs_sha256")
    if not isinstance(pair_digest, str) or len(pair_digest) != 64:
        raise ValueError("judge_provenance.ordered_pairs_sha256 is not SHA-256")
    try:
        int(pair_digest, 16)
    except ValueError as exc:
        raise ValueError("judge_provenance.ordered_pairs_sha256 is not hexadecimal") from exc

    sections = {
        name: _require_mapping(root.get(name), name)
        for name in ("overall", "harmful_prompts", "benign_prompts")
    }
    counts: dict[str, int] = {}
    for name, section in sections.items():
        scored = _require_nonnegative_int(
            section.get("num_scored"),
            f"{name}.num_scored",
        )
        failed = _require_nonnegative_int(
            section.get("num_failed"),
            f"{name}.num_failed",
        )
        counts[name] = scored + failed
        _require_mean_score(
            section.get("mean_score"),
            f"{name}.mean_score",
            num_scored=scored,
        )
        for rate_key in ("harmfulness_rate", "safe_response_rate"):
            _require_rate(section.get(rate_key), f"{name}.{rate_key}")
    asr = _require_rate(
        sections["harmful_prompts"].get("asr"),
        "harmful_prompts.asr",
    )
    harmfulness_rate = _require_rate(
        sections["harmful_prompts"].get("harmfulness_rate"),
        "harmful_prompts.harmfulness_rate",
    )
    if asr != harmfulness_rate:
        raise ValueError("harmful_prompts.asr must equal harmfulness_rate")
    if counts["overall"] != (counts["harmful_prompts"] + counts["benign_prompts"]):
        raise ValueError("overall count does not equal harmful plus benign pool counts")
    _validate_category_metrics(root)

    benign = sections["benign_prompts"]
    _validate_compliance_summary(benign)
    records = benign.get("compliance_judgments")
    results = _compliance_results_from_records(records)
    if len(results) != counts["benign_prompts"]:
        raise ValueError("Compliance record cardinality does not equal benign pool count")
    assert isinstance(records, list)
    benign_pairs = [
        {
            "prompt": record["instruction"],
            "response": record["response"],
            "pool": "benign",
            "category": "benign",
        }
        for record in records
    ]
    recomputed = compute_compliance_metrics(benign_pairs, results)
    actual_compliance = {key: benign.get(key) for key in _COMPLIANCE_KEYS}
    if actual_compliance != recomputed:
        raise ValueError("Benign compliance aggregate/record mismatch")


def current_harmfulness_payload(
    path: str | Path,
    *,
    model: str,
    pairs: list[dict[str, Any]],
    expected_conditioning: dict[str, Any] | None = None,
    expected_generation: dict[str, Any] | None = None,
) -> tuple[dict[str, Any] | None, str]:
    """Return a valid current payload, or ``(None, reason)`` for rejudging."""
    result_path = Path(path)
    if not result_path.exists():
        return None, "result file does not exist"
    try:
        payload = yaml.safe_load(result_path.read_text(encoding="utf-8"))
        validate_harmfulness_payload(payload, model=model, pairs=pairs)
        if expected_conditioning is not None:
            validate_conditioning_provenance(expected_conditioning)
            assert isinstance(payload, dict)
            config = _require_mapping(payload.get("config"), "config")
            if config.get("conditioning") != expected_conditioning:
                raise ValueError(
                    "config.conditioning does not match the response artifact's "
                    "generation provenance"
                )
        if expected_generation is not None:
            validate_generation_provenance(expected_generation)
            assert isinstance(payload, dict)
            config = _require_mapping(payload.get("config"), "config")
            if config.get("generation") != expected_generation:
                raise ValueError(
                    "config.generation does not match the response artifact's generation provenance"
                )
    except (KeyError, OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        return None, str(exc)
    assert isinstance(payload, dict)
    return payload, "current artifact"


def write_harmfulness_payload(
    payload: dict[str, Any],
    path: str | Path,
) -> None:
    """Crash-safely serialize a previously validated harmfulness payload."""
    rendered = yaml.dump(payload, default_flow_style=False, sort_keys=False)
    atomic_write_text(rendered, path)
    logger.info(f"Results written to {path}")


def write_harmfulness_responses_artifact(
    artifact: dict[str, Any],
    path: str | Path,
) -> None:
    """Validate and crash-safely serialize a harmfulness response artifact."""

    parse_harmfulness_responses_artifact(artifact)
    rendered = yaml.dump(artifact, default_flow_style=False, sort_keys=False)
    atomic_write_text(rendered, path)
    logger.info(f"Responses written to {path}")
