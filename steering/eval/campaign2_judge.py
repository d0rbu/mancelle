"""Cell-local persistence adapter for the shared harmfulness judge engine."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from loguru import logger

from errors import ParseError
from nanochat.atomic_io import atomic_json_dump
from scripts.utils.campaign2 import required_file_digest
from steering.eval.harmfulness_artifact import build_harmfulness_payload, build_judge_provenance
from steering.eval.harmfulness_judge import (
    JUDGE_MAX_ATTEMPTS,
    JUDGE_MAX_CONCURRENT,
    JudgeClient,
    JudgeScoreEnvelope,
    JudgmentKind,
    compute_compliance_metrics,
    compute_metrics,
)

JUDGE_RECORD_SCHEMA_VERSION = 1
JUDGE_RECORDS_DIR = "judge_records"
BENIGN_POOL = "benign"


def _digest(value: Any) -> str:
    rendered = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(rendered.encode("utf-8")).hexdigest()


def _cached_envelope(path: Path, identity: dict[str, Any], kind: JudgmentKind,
                     client: JudgeClient) -> JudgeScoreEnvelope | None:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (ValueError, UnicodeError) as exc:
        logger.warning("Rejudging invalid record {}: {}", path, exc)
        return None
    try:
        fields = {"identity", "raw", "error", "attempts", "judge_model", "sha256"}
        if not isinstance(record, dict) or set(record) != fields:
            raise ValueError("invalid judge record fields")
        checksum = record.pop("sha256")
        if checksum != _digest(record) or _digest(record["identity"]) != _digest(identity):
            raise ValueError("judge record checksum or identity mismatch")
        judge_model = record["judge_model"]
        allowed_models = {client.model}
        if client.refusal_fallback_model is not None:
            allowed_models.add(client.refusal_fallback_model)
        if judge_model not in allowed_models:
            raise ValueError("judge record model is not configured for this run")
        max_attempts = JUDGE_MAX_ATTEMPTS * (2 if judge_model != client.model else 1)
        if type(record["attempts"]) is not int or not 1 <= record["attempts"] <= max_attempts:
            raise ValueError("invalid judge attempt count")
        if record["error"] is not None:
            return None
        if not isinstance(record["raw"], str):
            raise ValueError("missing judge response")
        return JudgeScoreEnvelope(raw=record["raw"],
            result=client._parse_score(kind, record["raw"], judge_model),
            model=judge_model, attempts=record["attempts"], error=None)
    except (ValueError, OverflowError, ParseError) as exc:
        logger.warning("Rejudging invalid record {}: {}", path, exc)
        return None


def judge_records(client: JudgeClient, records: list[dict], output: Path) -> dict[str, Any]:
    """Resume cell-bound judgments and aggregate in original prompt order."""
    cell_sha256 = required_file_digest(output / "identity.json").sha256
    requests: list[tuple[str, str, JudgmentKind]] = []
    assignments: list[tuple[JudgmentKind, int, Path, dict[str, Any]]] = []
    results: dict[JudgmentKind, dict[int, Any]] = {kind: {} for kind in JudgmentKind}
    for index, row in enumerate(records):
        identity = {
            "schema_version": JUDGE_RECORD_SCHEMA_VERSION,
            "cell_sha256": cell_sha256,
            "prompt_index": index,
            **build_judge_provenance(model=client.model, pairs=[row]),
        }
        for kind in JudgmentKind:
            if kind is JudgmentKind.COMPLIANCE and row["pool"] != BENIGN_POOL:
                continue
            bound_identity = {**identity, "kind": kind.value}
            path = output / JUDGE_RECORDS_DIR / kind.value / f"{index:06d}.json"
            cached = _cached_envelope(path, bound_identity, kind, client)
            if cached is not None:
                results[kind][index] = cached.result
                continue
            requests.append((row["prompt"], row["response"], kind))
            assignments.append((kind, index, path, bound_identity))

    for request_index, envelope in client.score_envelopes(requests, max_concurrent=JUDGE_MAX_CONCURRENT):
        kind, index, path, bound_identity = assignments[request_index]
        record = {"identity": bound_identity, "raw": envelope.raw, "error": envelope.error,
                  "attempts": envelope.attempts, "judge_model": envelope.model}
        atomic_json_dump({**record, "sha256": _digest(record)}, path)
        results[kind][index] = envelope.result

    errors = []
    for kind in JudgmentKind:
        missing = sorted(index for index, result in results[kind].items() if result is None)
        if missing:
            label = "harmfulness" if kind is JudgmentKind.HARMFULNESS else "benign compliance"
            errors.append(f"{label} judge returned incomplete scores; remaining prompt indexes: {missing}")
    if errors:
        raise ValueError("; ".join(errors))

    benign_indexes = [index for index, row in enumerate(records) if row["pool"] == BENIGN_POOL]
    payload = build_harmfulness_payload(
        metrics=compute_metrics(records,
            [results[JudgmentKind.HARMFULNESS][i] for i in range(len(records))]),
        compliance_metrics=compute_compliance_metrics([records[i] for i in benign_indexes],
            [results[JudgmentKind.COMPLIANCE][i] for i in benign_indexes]),
        model=client.model, pairs=records,
    )
    model_assignments = []
    fallback_pair_indexes = set()
    for index in range(len(records)):
        harmfulness_model = results[JudgmentKind.HARMFULNESS][index].judge_model
        compliance_result = results[JudgmentKind.COMPLIANCE].get(index)
        compliance_model = None if compliance_result is None else compliance_result.judge_model
        model_assignments.append({"prompt_index": index, "harmfulness": harmfulness_model,
                                  "compliance": compliance_model})
        fallback_model = client.refusal_fallback_model
        if fallback_model is not None and fallback_model in {harmfulness_model, compliance_model}:
            fallback_pair_indexes.add(index)
    payload["judge_model_assignments"] = model_assignments
    payload["fallback_scored_pair_count"] = len(fallback_pair_indexes)
    return payload
