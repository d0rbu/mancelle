"""Authorized CPU consumer for immutable native Adv provider artifacts."""

from __future__ import annotations

import math
import time
from contextlib import ExitStack
from pathlib import Path

from loguru import logger
from omegaconf import DictConfig

from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import build_openrouter_reflection_lm
from steering.eval.axbench_adv.method_grid import ALL_METHOD_FACTORS
from steering.eval.axbench_adv.native_mailbox import digest_json, read_json, write_json
from steering.eval.axbench_adv.types import NATIVE_METHOD, QWEN_METHODS
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient
from steering.eval.axbench_base.config import LLMClientConfig
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache

JUDGE_MODEL = "openai/gpt-5.6-luna"


def run_provider(cfg: DictConfig) -> None:
    mailbox = Path(str(cfg.mailbox_dir)).resolve()
    if cfg.reflection_model != JUDGE_MODEL:
        raise AxBenchAdvError("native campaign reflection must use the approved judge model")
    client = None
    client_key = None
    with (
        SynthResponseCache(Path(str(cfg.reflection_cache))) as reflection_cache,
        ExitStack() as judge_stack,
    ):
        reflection = build_openrouter_reflection_lm(
            cfg=LLMClientConfig(
                model=JUDGE_MODEL,
                requests_per_minute=int(cfg.requests_per_minute),
                max_concurrent=1,
                transport_retry_budget_seconds=300.0,
            ),
            cache=reflection_cache,
        )
        last_activity = time.monotonic()
        while time.monotonic() - last_activity < float(cfg.idle_timeout_seconds):
            for path in sorted((mailbox / "requests").glob("*.json")):
                response_path = mailbox / "responses" / path.name
                if response_path.exists():
                    continue
                request = read_json(path)
                if set(request) != {"request_id", "schema_version", "operation", "payload"}:
                    raise AxBenchAdvError("provider request fields differ from contract")
                body = {key: request[key] for key in ("schema_version", "operation", "payload")}
                request_id = digest_json(body)
                if (
                    request["schema_version"] != 1
                    or request_id != request["request_id"]
                    or path.stem != request_id
                ):
                    raise AxBenchAdvError("provider request identity mismatch")
                payload = request["payload"]
                if not isinstance(payload, dict):
                    raise AxBenchAdvError("provider request payload must be a mapping")
                run_identity_sha256 = payload.get("run_identity_sha256")
                if run_identity_sha256 is not None and (
                    not isinstance(run_identity_sha256, str)
                    or len(run_identity_sha256) != 64
                    or any(character not in "0123456789abcdef" for character in run_identity_sha256)
                ):
                    raise AxBenchAdvError("provider run identity SHA-256 is invalid")
                if request["operation"] == "reflection":
                    reflection_fields = {"model", "prompt"}
                    if (
                        frozenset(payload)
                        not in {
                            frozenset(reflection_fields),
                            frozenset(reflection_fields | {"run_identity_sha256"}),
                        }
                        or payload["model"] != JUDGE_MODEL
                    ):
                        raise AxBenchAdvError(
                            "reflection request differs from approved model contract"
                        )
                    before = reflection.paid_cache_misses
                    response = reflection(payload["prompt"])
                    result = {
                        "response": response,
                        "paid_cache_misses": reflection.paid_cache_misses - before,
                    }
                elif request["operation"] == "judge":
                    legacy_fields = {"concept_id", "concept", "generation", "input_id", "model"}
                    method_fields = legacy_fields | {"method", "factor"}
                    authenticated_method_fields = method_fields | {"run_identity_sha256"}
                    if (
                        frozenset(payload)
                        not in {
                            frozenset(legacy_fields),
                            frozenset(method_fields),
                            frozenset(authenticated_method_fields),
                        }
                        or payload["model"] != JUDGE_MODEL
                    ):
                        raise AxBenchAdvError("judge request differs from approved model contract")
                    routed_method = payload.get("method", NATIVE_METHOD)
                    routed_factor = payload.get("factor", 1.0)
                    if routed_method not in (*QWEN_METHODS[:-1], NATIVE_METHOD):
                        raise AxBenchAdvError("native judge request method is unsupported")
                    if (
                        isinstance(routed_factor, bool)
                        or not isinstance(routed_factor, (int, float))
                        or not math.isfinite(float(routed_factor))
                    ):
                        raise AxBenchAdvError("native judge request factor is invalid")
                    allowed_factors = (
                        (1.0,)
                        if routed_method == NATIVE_METHOD
                        else ALL_METHOD_FACTORS[routed_method]
                    )
                    if float(routed_factor) not in tuple(float(value) for value in allowed_factors):
                        raise AxBenchAdvError("native judge request factor is outside its grid")
                    concept_id = payload["concept_id"]
                    key = (concept_id, payload["concept"])
                    if client is None or key != client_key:
                        judge_stack.close()
                        # The inner evaluator is exactly the existing vendored concept/fluency judge.
                        client = judge_stack.enter_context(
                            AxBenchAdvJudgeClient(
                                init_payload={
                                    "runtime_kind": "judge",
                                    "method": NATIVE_METHOD,
                                    "concept_id": concept_id,
                                    "concept": payload["concept"],
                                    "factor": 1.0,
                                    "judge_model": JUDGE_MODEL,
                                    "judge_cache": str(Path(str(cfg.judge_cache)).resolve()),
                                    "cache_only": False,
                                    "api_base_url": "https://openrouter.ai/api/v1",
                                },
                                timeout_seconds=600.0,
                            )
                        )
                        client_key = key
                    raw = client.evaluate(
                        generation=payload["generation"], input_id=payload["input_id"]
                    )
                    result = {
                        "generation": raw.generation,
                        "concept_rating": raw.concept_rating,
                        "fluency_rating": raw.fluency_rating,
                        "judge_rationale": raw.judge_rationale,
                        "paid_judge_cache_misses": raw.paid_judge_cache_misses,
                    }
                else:
                    raise AxBenchAdvError("unknown provider operation")
                write_json(
                    response_path,
                    {
                        "request_id": request_id,
                        "result": result,
                        "result_sha256": digest_json(result),
                    },
                )
                last_activity = time.monotonic()
                logger.info("Provider artifact completed: {} {}", request["operation"], request_id)
            if (mailbox / "STOP").exists() and all(
                (mailbox / "responses" / path.name).exists()
                for path in (mailbox / "requests").glob("*.json")
            ):
                logger.info("Native provider STOP received; request backlog complete")
                return
            if bool(cfg.once):
                return
            time.sleep(float(cfg.poll_seconds))
