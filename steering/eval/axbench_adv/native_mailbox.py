"""Durable artifact handoff for native Adv judge/reflection CPU workers.

The GPU process never opens an API connection. Operators may synchronize these
ordinary JSON artifacts to an authorized CPU host; this is not an HTTP proxy.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

from nanochat.atomic_io import atomic_write_text
from steering._common.strict_json import reject_nonstandard_json_constant, strict_json_object
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.final_judge import _rationale
from steering.eval.axbench_adv.fitness import evaluate_fitness
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))


def digest_json(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(), object_pairs_hook=strict_json_object,
                       parse_constant=reject_nonstandard_json_constant)
    if not isinstance(value, dict):
        raise AxBenchAdvError(f"expected JSON mapping: {path}")
    return value


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(canonical_json(value) + "\n", path)


def checked_raw(result: dict[str, Any], generation: str) -> RawEvaluation:
    expected = {"generation", "concept_rating", "fluency_rating", "judge_rationale", "paid_judge_cache_misses"}
    if set(result) not in (expected, expected | {"generation_calls"}):
        raise AxBenchAdvError("native judge result fields differ")
    if result["generation"] != generation or not isinstance(result["judge_rationale"], str):
        raise AxBenchAdvError("native judge generation or rationale mismatch")
    if type(result["paid_judge_cache_misses"]) is not int or not 0 <= result["paid_judge_cache_misses"] <= 2:
        raise AxBenchAdvError("native judge paid-miss counter invalid")
    raw = RawEvaluation(**result)
    _rationale(raw)
    if type(raw.generation_calls) is not int or raw.generation_calls not in {0, 1}:
        raise AxBenchAdvError("invalid native generation-call count")
    evaluate_fitness(concept_rating=raw.concept_rating, fluency_rating=raw.fluency_rating, output=raw.generation)
    return raw


class ArtifactMailbox:
    """Content-addressed requests and validated result receipts, with bounded waits."""

    def __init__(self, directory: Path, *, timeout_seconds: float = 3600.0) -> None:
        self.directory = directory
        self.timeout_seconds = timeout_seconds
        if timeout_seconds <= 0:
            raise ValueError("mailbox timeout must be positive")

    def request(self, operation: str, payload: dict[str, Any]) -> dict[str, Any]:
        if operation not in {"judge", "reflection"}:
            raise AxBenchAdvError("unknown native provider operation")
        body = {"schema_version": 1, "operation": operation, "payload": payload}
        request_id = digest_json(body)
        request = {"request_id": request_id, **body}
        request_path = self.directory / "requests" / f"{request_id}.json"
        response_path = self.directory / "responses" / f"{request_id}.json"
        preexisting_response = response_path.exists()
        if request_path.exists() and read_json(request_path) != request:
            raise AxBenchAdvError("mailbox request identity collision")
        if not request_path.exists():
            write_json(request_path, request)
        started = time.monotonic()
        while not response_path.exists():
            if time.monotonic() - started > self.timeout_seconds:
                raise AxBenchAdvError(f"provider artifact response timed out: {request_id}")
            time.sleep(1.0)
        response = read_json(response_path)
        if set(response) != {"request_id", "result", "result_sha256"}:
            raise AxBenchAdvError("mailbox response fields differ from contract")
        if response["request_id"] != request_id or digest_json(response["result"]) != response["result_sha256"]:
            raise AxBenchAdvError("mailbox response identity mismatch")
        if not isinstance(response["result"], dict):
            raise AxBenchAdvError("mailbox result must be a mapping")
        result = dict(response["result"])
        if preexisting_response:
            for field in ("paid_cache_misses", "paid_judge_cache_misses"):
                if field in result:
                    result[field] = 0
        return result


class MailboxReflection:
    def __init__(
        self,
        mailbox: ArtifactMailbox,
        model: str,
        *,
        run_identity_sha256: str | None = None,
    ) -> None:
        self.mailbox = mailbox
        self.model = model
        self.run_identity_sha256 = run_identity_sha256
        self.paid_cache_misses = 0

    def __call__(self, prompt: str | list[dict[str, Any]]) -> str:
        if not isinstance(prompt, str):
            raise AxBenchAdvError("reflection requires textual prompt")
        payload = {"model": self.model, "prompt": prompt}
        if self.run_identity_sha256 is not None:
            payload["run_identity_sha256"] = self.run_identity_sha256
        result = self.mailbox.request("reflection", payload)
        if set(result) != {"response", "paid_cache_misses"} or not isinstance(result["response"], str) or not result["response"].strip():
            raise AxBenchAdvError("invalid reflection response")
        misses = result["paid_cache_misses"]
        if type(misses) is not int or misses not in {0, 1}:
            raise AxBenchAdvError("invalid reflection paid-miss count")
        self.paid_cache_misses += misses
        return result["response"]
