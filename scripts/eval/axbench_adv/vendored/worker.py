"""Persistent AxBench generation and inner-judge worker for one method/concept."""

from __future__ import annotations

import asyncio
import json
import math
import os
import random
import sys
import time
import traceback
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from typing import Any, BinaryIO, cast

import axbench
import numpy as np
import openai
import pandas as pd
import torch as t
from axbench.evaluators.lm_judge import LMJudgeEvaluator
from axbench.evaluators.prompt_templates import (
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE,
    UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE,
)
from axbench.models.language_models import OPENROUTER_LUNA_MODEL, LanguageModel
from axbench.scripts.inference import (
    load_metadata_flatten,
    run_steering_generation,
)
from axbench.utils.constants import is_chat_model_name
from axbench.utils.model_utils import get_prefix_length
from axbench.utils.openrouter import DEFAULT_OPENROUTER_BASE_URL, create_openrouter_client
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedTokenizerBase

from scripts.eval.axbench_adv.counter_hooks import load_vendor_direction, residual_counter
from scripts.eval.axbench_adv.pressures import DIRECTION_METHOD, Pressure
from scripts.eval.axbench_adv.vendored.gcg_worker import GCGSession
from scripts.eval.axbench_adv.vendored.protocol import (
    PROTOCOL_VERSION,
    decode_frame,
    write_frame,
)

_METHODS = frozenset({"DiffMean", "LinearProbe", "LoRA", "PromptSteering"})
_ADDITIVE_METHODS = frozenset({"DiffMean", "LinearProbe"})
_ATTACK_COLUMN = {
    "DiffMean": "input",
    "LinearProbe": "input",
    "LoRA": "input",
    "PromptSteering": "steered_input",
}
_MAX_SIGNED_63_BIT = (1 << 63) - 1

# A provider outage or a network blip surfaces here as ``openai.APIConnectionError`` (``APITimeoutError``
# is a subclass). The SDK's own ``max_retries`` covers seconds; these cover the minutes-long outages that
# otherwise turn one lost request into a dead worker and a discarded slice. Waits grow linearly
# (30 s, 60 s, ...) and plateau at the cap from attempt 10, about 72 minutes of tolerance in total.
JUDGE_CONNECTION_ATTEMPTS = 20
JUDGE_CONNECTION_BASE_WAIT_SECONDS = 30.0
JUDGE_CONNECTION_MAX_WAIT_SECONDS = 300.0


def set_generation_seed(seed: int) -> None:
    """Seed Python, NumPy, CPU Torch, and every CUDA device for one row."""

    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed <= _MAX_SIGNED_63_BIT:
        raise ValueError("row_seed must be an integer in [0, 2^63 - 1]")
    random.seed(seed)
    np.random.seed(seed & 0xFFFFFFFF)
    t.manual_seed(seed)
    t.cuda.manual_seed_all(seed)


def _exact_keys(value: object, expected: set[str], *, label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        observed = sorted(value) if isinstance(value, dict) else type(value).__name__
        raise ValueError(f"{label} fields must be exactly {sorted(expected)}, got {observed}")
    return value


def _text(value: object, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonblank string")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise ValueError(f"{name} must be valid UTF-8") from exc
    return value


def _integer(value: object, *, name: str, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return value


def _finite_float(value: object, *, name: str, minimum: float | None = None) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be numeric")
    parsed = float(value)
    if not math.isfinite(parsed) or (minimum is not None and parsed < minimum):
        raise ValueError(f"{name} must be finite and within its allowed range")
    return parsed


def _validate_judge_fields(
    *,
    judge_model: object,
    judge_cache: object,
    cache_only: object,
    api_base_url: object,
) -> tuple[str, Path, bool, str]:
    if type(cache_only) is not bool:
        raise ValueError("cache_only must be a boolean")
    parsed_model = _text(judge_model, name="judge_model")
    if parsed_model != OPENROUTER_LUNA_MODEL:
        raise ValueError(f"judge_model must be {OPENROUTER_LUNA_MODEL!r}")
    parsed_base_url = _text(api_base_url, name="api_base_url").strip().rstrip("/")
    configured_base_url = (
        os.environ.get("OPENROUTER_BASE_URL", DEFAULT_OPENROUTER_BASE_URL).strip().rstrip("/")
    )
    if not cache_only and parsed_base_url != configured_base_url:
        raise ValueError("api_base_url must match OPENROUTER_BASE_URL for live worker requests")
    cache_path = Path(_text(judge_cache, name="judge_cache")).resolve()
    return parsed_model, cache_path, cache_only, parsed_base_url


class _InnerJudge:
    """Exact vendored concept/fluency judge shared by both runtime kinds."""

    def __init__(
        self,
        *,
        method: str,
        concept_id: int,
        concept: str,
        judge_model: object,
        judge_cache: object,
        cache_only: object,
        api_base_url: object,
        pair_concurrency: int = 1,
    ) -> None:
        if type(pair_concurrency) is not int or pair_concurrency not in (1, 2):
            raise ValueError("pair_concurrency must be 1 or 2")
        parsed_model, cache_path, parsed_cache_only, parsed_base_url = _validate_judge_fields(
            judge_model=judge_model,
            judge_cache=judge_cache,
            cache_only=cache_only,
            api_base_url=api_base_url,
        )
        client = None if parsed_cache_only else create_openrouter_client("for AxBench-Adv judging")
        self.language_model = LanguageModel(
            parsed_model,
            client,
            use_cache=False,
            judge_cache=cache_path,
            cache_only=parsed_cache_only,
            api_base_url=parsed_base_url,
            temperature=0.0,
        )
        self.evaluator = LMJudgeEvaluator(
            method,
            concept_id=concept_id,
            lm_model=self.language_model,
        )
        self.method = method
        self.concept = concept
        self.pair_concurrency = pair_concurrency

    def judge(self, generation: str) -> tuple[float, float, str, int]:
        """Judge one generation, retrying the two-call unit across transient connection failures.

        Call accounting is re-snapshotted per attempt, so a concept rating cached by a failed attempt is
        counted as a cache hit on the retry and the reported paid-miss count stays exact.
        """
        for attempt in range(1, JUDGE_CONNECTION_ATTEMPTS + 1):
            try:
                return self._judge_once(generation)
            except openai.APIConnectionError as exc:
                if attempt == JUDGE_CONNECTION_ATTEMPTS:
                    raise
                wait = min(
                    JUDGE_CONNECTION_MAX_WAIT_SECONDS,
                    JUDGE_CONNECTION_BASE_WAIT_SECONDS * attempt,
                )
                print(
                    f"inner judge connection failure (attempt {attempt}/{JUDGE_CONNECTION_ATTEMPTS}): "
                    f"{type(exc).__name__}: {exc}; retrying in {wait:g}s",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(wait)
        raise AssertionError("unreachable: the final attempt re-raises")

    def _judge_once(self, generation: str) -> tuple[float, float, str, int]:
        concept_prompt = UNIDIRECTIONAL_PAIRWISE_EVALUATION_CONCEPT_RELEVANCE_TEMPLATE.format(
            concept=self.concept,
            sentence=generation,
        )
        fluency_prompt = UNIDIRECTIONAL_PAIRWISE_EVALUATION_FLUENCY_TEMPLATE.format(
            sentence=generation
        )
        before_calls = self.language_model.stats.total_call
        before_hits = self.language_model.stats.total_cache_hit
        if getattr(self, "pair_concurrency", 1) == 2:
            names = [
                f"{self.method}_concept_{self.method}_LMJudgeEvaluator",
                f"{self.method}_fluency_{self.method}_LMJudgeEvaluator",
            ]
            prompts = [concept_prompt, fluency_prompt]

            async def judge_pair() -> list[str]:
                settled = await asyncio.gather(
                    *(
                        self.language_model.chat_completion(
                            self.language_model.client,
                            prompt,
                            name,
                            validator=self.evaluator._get_rating_from_completion,
                        )
                        for name, prompt in zip(names, prompts, strict=True)
                    ),
                    return_exceptions=True,
                )
                completions: list[str] = []
                for name, prompt, result in zip(names, prompts, settled, strict=True):
                    if isinstance(result, BaseException):
                        continue
                    completion, usage, cache_hit = result
                    self.language_model.stats.record(
                        name, usage, prompt=prompt, completion=completion, cache_hit=cache_hit
                    )
                    completions.append(completion)
                for result in settled:
                    if isinstance(result, BaseException):
                        raise result
                return completions

            concept_completion, fluency_completion = asyncio.run(judge_pair())
            ratings = self.evaluator._get_ratings_from_completions(
                [concept_completion, fluency_completion]
            )
            concept_ratings, fluency_ratings = [ratings[0]], [ratings[1]]
            concept_completions, fluency_completions = [concept_completion], [fluency_completion]
        else:
            concept_ratings, concept_completions = self.evaluator._get_ratings_from_prompts(
                [concept_prompt], f"{self.method}_concept"
            )
            fluency_ratings, fluency_completions = self.evaluator._get_ratings_from_prompts(
                [fluency_prompt], f"{self.method}_fluency"
            )
        call_delta = self.language_model.stats.total_call - before_calls
        hit_delta = self.language_model.stats.total_cache_hit - before_hits
        if call_delta != 2 or not 0 <= hit_delta <= call_delta:
            raise RuntimeError(
                f"inner judge accounting must be two calls, got calls={call_delta}, hits={hit_delta}"
            )
        rationale = json.dumps(
            {
                "concept": concept_completions[0],
                "fluency": fluency_completions[0],
            },
            allow_nan=False,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        return (
            float(concept_ratings[0]),
            float(fluency_ratings[0]),
            rationale,
            call_delta - hit_delta,
        )

    def close(self) -> None:
        asyncio.run(self.language_model.close())


class WorkerRuntime:
    """One loaded generation method plus one persistent exact judge/cache."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        sidecar_value = payload.pop("gcg_sidecar_dir", None)
        fields = _exact_keys(
            payload,
            {
                "api_base_url",
                "aux_dump_dir",
                "cache_only",
                "concept",
                "concept_id",
                "device",
                "factor",
                "intervene_on_prompt",
                "intervention_type",
                "judge_cache",
                "judge_model",
                "layer",
                "main_dump_dir",
                "max_concepts",
                "method",
                "model_name",
                "model_revision",
                "output_length",
                "prepared_dir",
                "steering_layers",
                "temperature",
                "use_bf16",
                "worker_max_prompt_tokens",
            },
            label="init payload",
        )
        self.method = _text(fields["method"], name="method")
        if self.method not in _METHODS:
            raise ValueError(f"unsupported method {self.method!r}")
        self.concept_id = _integer(fields["concept_id"], name="concept_id")
        self.concept = _text(fields["concept"], name="concept")
        self.factor = _finite_float(fields["factor"], name="factor")
        self.model_name = _text(fields["model_name"], name="model_name")
        self.model_revision = (
            None
            if fields["model_revision"] is None
            else _text(fields["model_revision"], name="model_revision")
        )
        self.layer = _integer(fields["layer"], name="layer")
        self.max_concepts = _integer(fields["max_concepts"], name="max_concepts", minimum=1)
        self.temperature = _finite_float(fields["temperature"], name="temperature", minimum=0.0)
        if self.temperature <= 0:
            raise ValueError("temperature must be positive for sampled generation")
        self.output_length = _integer(fields["output_length"], name="output_length", minimum=1)
        self.worker_max_prompt_tokens = _integer(
            fields["worker_max_prompt_tokens"], name="worker_max_prompt_tokens", minimum=1
        )
        self.device = _text(fields["device"], name="device")
        if self.device.startswith("cuda") and not t.cuda.is_available():
            raise ValueError("CUDA worker requested but torch.cuda.is_available() is false")
        if fields["use_bf16"] is not True:
            raise ValueError("worker requires the source-pinned use_bf16=true runtime")
        if fields["intervention_type"] != "addition":
            raise ValueError("worker requires steering_intervention_type='addition'")
        if fields["intervene_on_prompt"] is not True:
            raise ValueError("worker requires intervene_on_prompt=true")
        if fields["steering_layers"] != [self.layer]:
            raise ValueError("steering_layers must contain exactly the configured layer")
        _validate_judge_fields(
            judge_model=fields["judge_model"],
            judge_cache=fields["judge_cache"],
            cache_only=fields["cache_only"],
            api_base_url=fields["api_base_url"],
        )
        prepared_dir = Path(_text(fields["prepared_dir"], name="prepared_dir")).resolve()
        main_dump_dir = Path(_text(fields["main_dump_dir"], name="main_dump_dir")).resolve()
        aux_dump_dir = Path(_text(fields["aux_dump_dir"], name="aux_dump_dir")).resolve()
        sidecar_dir = None
        if sidecar_value is not None:
            sidecar_dir = Path(_text(sidecar_value, name="gcg_sidecar_dir")).resolve(strict=True)
            sidecar_stat = sidecar_dir.stat()
            if (
                not sidecar_dir.is_dir()
                or sidecar_stat.st_uid != os.getuid()
                or sidecar_stat.st_mode & 0o077
            ):
                raise ValueError("gcg_sidecar_dir must be a private current-user directory")
        metadata = load_metadata_flatten(prepared_dir / "generate", self.max_concepts)
        metadata_by_id = {int(row["concept_id"]): row for row in metadata}
        if set(metadata_by_id) != set(range(self.max_concepts)):
            raise ValueError("prepared metadata concept ids must be contiguous from zero")
        if self.concept_id not in metadata_by_id:
            raise ValueError(f"concept_id {self.concept_id} is absent from prepared metadata")
        if metadata_by_id[self.concept_id]["concept"] != self.concept:
            raise ValueError("worker concept does not match prepared metadata")

        self.tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(
                self.model_name,
                revision=self.model_revision,
                use_fast=True,
                model_max_length=self.worker_max_prompt_tokens,
            ),
        )
        self.tokenizer.padding_side = "right"
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            revision=self.model_revision,
            torch_dtype=t.bfloat16,
            device_map=self.device,
        ).eval()
        if self.tokenizer.unk_token is None and self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({"pad_token": "[PAD]"})
            base_model.resize_token_embeddings(len(self.tokenizer))

        model_class = getattr(axbench, self.method)
        self.benchmark_model = model_class(
            base_model,
            self.tokenizer,
            layer=self.layer,
            training_args=None,
            low_rank_dimension=self.max_concepts,
            device=self.device,
            steering_layers=[self.layer],
        )
        train_dir = aux_dump_dir / "train" if self.method == "LoRA" else main_dump_dir / "train"
        self.benchmark_model.load(
            dump_dir=train_dir,
            sae_path=metadata[0]["ref"],
            mode="steering",
            priority_mode="compute_priority",
            intervention_type="addition",
            concept_id=self.concept_id,
            low_rank_dimension=1,
        )
        self.benchmark_model.to(self.device)
        if hasattr(self.benchmark_model, "ax"):
            self.benchmark_model.ax.eval()
            self.benchmark_model.ax.to(t.bfloat16)
        if self.method in _ADDITIVE_METHODS:
            self.benchmark_model.pre_compute_mean_activations(main_dump_dir / "inference")
            if self.concept_id not in self.benchmark_model.max_activations:
                raise ValueError(
                    f"{self.method} has no max activation for concept {self.concept_id}"
                )
        self._gcg_main_dump_dir = main_dump_dir
        self._gcg_aux_dump_dir = aux_dump_dir
        self._gcg_sidecar_dir = sidecar_dir
        self._gcg_session: GCGSession | None = None
        self.prefix_length = (
            get_prefix_length(self.tokenizer) if is_chat_model_name(self.model_name) else 1
        )
        self.inner_judge = _InnerJudge(
            method=self.method,
            concept_id=self.concept_id,
            concept=self.concept,
            judge_model=fields["judge_model"],
            judge_cache=fields["judge_cache"],
            cache_only=fields["cache_only"],
            api_base_url=fields["api_base_url"],
        )
        # Compatibility aliases preserve the reviewed Qwen worker surface.
        self.language_model = self.inner_judge.language_model
        self.evaluator = self.inner_judge.evaluator

    def _judge(self, generation: str) -> tuple[float, float, str, int]:
        inner = getattr(self, "inner_judge", None)
        if inner is not None:
            return inner.judge(generation)
        compatibility = object.__new__(_InnerJudge)
        compatibility.method = self.method
        compatibility.concept = self.concept
        compatibility.language_model = self.language_model
        compatibility.evaluator = self.evaluator
        return compatibility.judge(generation)

    def evaluate(
        self, payload: Mapping[str, Any], *, soft_embeddings: t.Tensor | None = None
    ) -> dict[str, Any]:
        payload = dict(payload)
        counter = payload.pop("counter", None)
        context = nullcontext()
        if counter is not None:
            if not isinstance(counter, dict):
                raise ValueError("counter must be an object")
            pressure = Pressure(counter["pressure"])
            if (
                pressure not in DIRECTION_METHOD
                or counter["concept_id"] != self.concept_id
                or counter["method"] != self.method
                or counter["layer"] != self.layer
            ):
                raise ValueError("counter identity differs from worker cell")
            path = Path(counter["path"])
            if path.name != f"{DIRECTION_METHOD[pressure]}_weight.pt":
                raise ValueError("counter artifact filename differs from pressure")
            direction = load_vendor_direction(path, self.concept_id, counter["sha256"])
            context = residual_counter(
                self.benchmark_model.model,
                layer=self.layer,
                direction=direction,
                magnitude=counter["magnitude"],
            )
        fields = _exact_keys(
            payload,
            {"input", "input_id", "row_seed", "steered_input"},
            label="evaluate payload",
        )
        input_id = _integer(fields["input_id"], name="input_id")
        plain_input = _text(fields["input"], name="input")
        steered_input = _text(fields["steered_input"], name="steered_input")
        seed = _integer(fields["row_seed"], name="row_seed")
        if seed > _MAX_SIGNED_63_BIT:
            raise ValueError("row_seed must not exceed 2^63 - 1")
        target_prompt = plain_input if _ATTACK_COLUMN[self.method] == "input" else steered_input
        encoded = self.tokenizer(
            target_prompt,
            add_special_tokens=True,
            truncation=False,
        )["input_ids"]
        if not isinstance(encoded, list) or not all(type(token_id) is int for token_id in encoded):
            raise ValueError("tokenizer returned invalid input_ids for prompt preflight")
        if len(encoded) > self.worker_max_prompt_tokens:
            raise ValueError(
                f"attacked prompt has {len(encoded)} tokens, exceeding configured prompt-token "
                f"limit {self.worker_max_prompt_tokens}"
            )
        if soft_embeddings is not None:
            if counter is not None:
                raise ValueError("soft generation cannot combine a counter intervention")
            context = self._gcg().soft_generation(
                soft_embeddings, prompt=target_prompt, token_ids=encoded
            )
        set_generation_seed(seed)
        frame = pd.DataFrame(
            {
                "input": [plain_input],
                "steered_input": [steered_input],
                "factor": [self.factor],
                "concept_id": [self.concept_id],
                "sae_id": [self.concept_id],
            }
        )
        with t.inference_mode(), context:
            results = run_steering_generation(
                self.benchmark_model,
                frame,
                steering_batch_size=1,
                steering_output_length=self.output_length,
                temperature=self.temperature,
                prefix_length=self.prefix_length,
                positions="all_prompt" if self.method in _ADDITIVE_METHODS else None,
                disable_neuronpedia_max_act=False,
                intervene_on_prompt=True,
                compute_perplexity=False,
            )
        generations = results.get("steered_generation") if isinstance(results, Mapping) else None
        if (
            not isinstance(generations, list)
            or len(generations) != 1
            or not isinstance(generations[0], str)
        ):
            raise ValueError("steering generation must return exactly one text value")
        generation = generations[0]
        concept_rating, fluency_rating, rationale, paid_misses = self._judge(generation)
        return {
            "concept_rating": concept_rating,
            "fluency_rating": fluency_rating,
            "generation": generation,
            "input_id": input_id,
            "judge_rationale": rationale,
            "paid_judge_cache_misses": paid_misses,
        }

    def _gcg(self) -> GCGSession:
        if self._gcg_session is None:
            self._gcg_session = GCGSession(self)
        return self._gcg_session

    def gcg_prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().prepare(payload)

    def gcg_filter(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().filter(payload)

    def gcg_score(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().score(payload)

    def gcg_gradient(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().gradient(payload)

    def soft_prepare(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().prepare_soft_prompt(payload)

    def soft_score(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().soft_score(payload)

    def soft_gradient(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        return self._gcg().soft_gradient(payload)

    def evaluate_soft(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        fields = _exact_keys(
            payload,
            {"input", "input_id", "row_seed", "steered_input"},
            label="evaluate_soft payload",
        )
        embeddings = self._gcg()._load_soft_embeddings()
        return self.evaluate(fields, soft_embeddings=embeddings)

    def close(self) -> None:
        if self._gcg_session is not None:
            self._gcg_session.close()
        self.inner_judge.close()


class JudgeRuntime:
    """Judge-only child runtime for root-generated SelfConditioning text."""

    def __init__(self, payload: Mapping[str, Any]) -> None:
        payload = dict(payload)
        pair_concurrency = payload.pop("pair_concurrency", 1)
        if type(pair_concurrency) is not int or pair_concurrency not in (1, 2):
            raise ValueError("pair_concurrency must be 1 or 2")
        fields = _exact_keys(
            payload,
            {
                "api_base_url",
                "cache_only",
                "concept",
                "concept_id",
                "factor",
                "judge_cache",
                "judge_model",
                "method",
                "runtime_kind",
            },
            label="init payload",
        )
        if fields["runtime_kind"] != "judge":
            raise ValueError("judge runtime_kind must be 'judge'")
        self.method = _text(fields["method"], name="method")
        project_methods = {"SelfConditioning", "SelfConditioningMasked", "NLSteering"}
        allowed_methods = project_methods | {
            "NativeDecoderPromptSteering",
            "EncDecPromptSteering",
            "EncDecDiffMean",
            "EncDecLinearProbe",
            "EncDecLoRA",
        }
        if self.method not in allowed_methods:
            raise ValueError(
                "judge runtime method is not a supported project, native, or EncDec baseline"
            )
        if self.method not in project_methods and pair_concurrency != 1:
            raise ValueError("pair_concurrency=2 is project-only")
        self.concept_id = _integer(fields["concept_id"], name="concept_id")
        self.concept = _text(fields["concept"], name="concept")
        self.factor = _finite_float(fields["factor"], name="factor")
        self.inner_judge = _InnerJudge(
            method=self.method,
            concept_id=self.concept_id,
            concept=self.concept,
            judge_model=fields["judge_model"],
            judge_cache=fields["judge_cache"],
            cache_only=fields["cache_only"],
            api_base_url=fields["api_base_url"],
            **({"pair_concurrency": pair_concurrency} if pair_concurrency != 1 else {}),
        )

    def evaluate(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        fields = _exact_keys(payload, {"generation", "input_id"}, label="evaluate payload")
        input_id = _integer(fields["input_id"], name="input_id")
        generation = fields["generation"]
        if not isinstance(generation, str):
            raise ValueError("generation must be a string")
        try:
            generation.encode("utf-8", errors="strict")
        except UnicodeEncodeError as exc:
            raise ValueError("generation must be valid UTF-8") from exc
        concept_rating, fluency_rating, rationale, paid_misses = self.inner_judge.judge(generation)
        return {
            "concept_rating": concept_rating,
            "fluency_rating": fluency_rating,
            "generation": generation,
            "input_id": input_id,
            "judge_rationale": rationale,
            "paid_judge_cache_misses": paid_misses,
        }

    def close(self) -> None:
        self.inner_judge.close()


Runtime = WorkerRuntime | JudgeRuntime
RuntimeFactory = Callable[[Mapping[str, Any]], Runtime]


def _runtime_factory(payload: Mapping[str, Any]) -> Runtime:
    return (
        JudgeRuntime(payload) if payload.get("runtime_kind") == "judge" else WorkerRuntime(payload)
    )


def _request(value: object) -> tuple[int, str, dict[str, Any]]:
    frame = _exact_keys(value, {"id", "op", "payload", "version"}, label="request")
    if type(frame["version"]) is not int or frame["version"] != PROTOCOL_VERSION:
        raise ValueError(f"protocol version must be {PROTOCOL_VERSION}, got {frame['version']!r}")
    request_id = _integer(frame["id"], name="request id", minimum=1)
    operation = _text(frame["op"], name="operation")
    payload = frame["payload"]
    if not isinstance(payload, dict):
        raise ValueError("request payload must be a JSON object")
    return request_id, operation, payload


def serve(
    reader: BinaryIO,
    writer: BinaryIO,
    *,
    runtime_factory: RuntimeFactory = _runtime_factory,
) -> int:
    """Serve one init/evaluate*/close session, failing closed on every error."""

    runtime: Runtime | None = None
    while True:
        request_id: int | None = None
        operation = "invalid"
        try:
            request_id, operation, payload = _request(decode_frame(reader))
            if operation == "init":
                if runtime is not None:
                    raise ValueError("worker is already initialized")
                runtime = runtime_factory(payload)
                result: dict[str, Any] = {
                    "concept_id": runtime.concept_id,
                    "factor": runtime.factor,
                    "method": runtime.method,
                }
            elif operation == "rebind":
                if not isinstance(runtime, JudgeRuntime) or payload.get("runtime_kind") != "judge":
                    raise ValueError("rebind requires an initialized judge runtime")
                runtime.close()
                runtime = None
                runtime = runtime_factory(payload)
                if not isinstance(runtime, JudgeRuntime):
                    raise ValueError("rebind created a non-judge runtime")
                result = {
                    "concept_id": runtime.concept_id,
                    "factor": runtime.factor,
                    "method": runtime.method,
                }
            elif operation == "evaluate":
                if runtime is None:
                    raise ValueError("worker must be initialized before evaluate")
                result = runtime.evaluate(payload)
            elif operation in {
                "gcg_prepare",
                "gcg_filter",
                "gcg_score",
                "gcg_gradient",
                "soft_prepare",
                "soft_score",
                "soft_gradient",
                "evaluate_soft",
            }:
                if runtime is None:
                    raise ValueError(f"worker must be initialized before {operation}")
                if not isinstance(runtime, WorkerRuntime):
                    raise ValueError(f"judge runtime does not support {operation}")
                result = getattr(runtime, operation)(payload)
            elif operation == "close":
                if runtime is None:
                    raise ValueError("worker must be initialized before close")
                _exact_keys(payload, set(), label="close payload")
                runtime.close()
                write_frame(
                    writer,
                    {
                        "id": request_id,
                        "op": operation,
                        "result": {},
                        "status": "ok",
                        "version": PROTOCOL_VERSION,
                    },
                )
                return 0
            else:
                raise ValueError(f"unknown worker operation {operation!r}")
            write_frame(
                writer,
                {
                    "id": request_id,
                    "op": operation,
                    "result": result,
                    "status": "ok",
                    "version": PROTOCOL_VERSION,
                },
            )
        except BaseException as exc:
            traceback.print_exc(file=sys.stderr)
            if request_id is not None:
                try:
                    write_frame(
                        writer,
                        {
                            "error": {
                                "message": str(exc),
                                "type": type(exc).__name__,
                            },
                            "id": request_id,
                            "op": operation,
                            "status": "error",
                            "version": PROTOCOL_VERSION,
                        },
                    )
                except BaseException:
                    traceback.print_exc(file=sys.stderr)
            return 1


def main() -> int:
    protocol_writer = sys.stdout.buffer
    sys.stdout = sys.stderr
    return serve(sys.stdin.buffer, protocol_writer)


if __name__ == "__main__":
    raise SystemExit(main())
