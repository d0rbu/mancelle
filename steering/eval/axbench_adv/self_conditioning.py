"""Thin root orchestration for real-prompt AxBench-Adv SelfConditioning."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import unicodedata
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import AbstractContextManager
from dataclasses import dataclass, replace
from pathlib import Path
from queue import SimpleQueue
from types import TracebackType
from typing import TYPE_CHECKING, Any, Protocol

import pandas as pd
import torch as t
from beartype import beartype

from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import KVCache
from nanochat.gpt import Linear
from nanochat.tokenizer import SPECIAL_TOKENS, get_tokenizer
from scripts.eval.axbench_adv.gcg_math import (
    activation_l2_pair,
    activation_l2_row,
    response_margin_gradient,
    response_margin_value,
    weighted_bpb,
)
from scripts.eval.axbench_adv.gcg_runtime_state import preserve_gcg_model_state
from steering.conditional.encoder_decoder.eval_adapter import (
    TargetAwareEncoderDecoderEvalWrapper,
    seed_self_conditioning_from_prompt,
)
from steering.conditional.encoder_decoder.eval_residual import (
    post_cross_attention_residual_counter,
    scoped_post_cross_attention_callback,
)
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderLayout,
    PackedEncoderOutput,
    stack_packed_encoder_outputs,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
)
from steering.conditional.sampling import SamplingBundle
from steering.eval.axbench_adv.counter import CounterDirection, GcgActivationDirection
from steering.eval.axbench_adv.errors import AXBENCH_ADV_BEARTYPE, AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample, RawEvaluation
from steering.eval.axbench_adv.method_grid import (
    ADV_FULL_FACTORS,
    ADV_INPUTS_PER_CONCEPT,
    ALL_METHOD_FACTORS,
)
from steering.eval.axbench_adv.project_soft_prompt import (
    generate_project_soft_prompt,
    project_suffix_prefix,
    replace_soft_embedding_rows,
    validate_project_soft_embeddings,
)
from steering.eval.axbench_adv.prompt_surgery import canonicalize_suffix
from steering.eval.axbench_adv.types import NL_METHOD, ROOT_METHODS
from steering.eval.axbench_adv.worker_client import (
    AxBenchAdvJudgeClient,
    GcgFilteredCandidate,
    GcgPrepared,
)
from steering.eval.axbench_upstream.data import ConceptTrainingTexts
from steering.eval.axbench_upstream.methods import (
    MethodSetup,
    PreparedPromptContext,
    assert_masked_diffmean_feasible,
    assert_prepared_prompt_matches,
    build_self_conditioning_config,
    build_self_conditioning_masked_policy,
    fit_concept_dense_diffmeans,
    generate_self_conditioning,
    load_extraction_stack,
    load_target_model,
    prepare_prompt_context,
    self_conditioning_weight,
)
from steering.eval.axbench_upstream.run_methods import (
    derive_generation_seed,
    derive_selection_seed,
)
from steering.eval.chat_generation import (
    ChatCompletionPrompt,
    render_prompt_for_chat_completion,
)
from steering.eval.dense_feature_targets import load_weight_free_gemma3_extraction_stack
from steering.eval.target_conditioning import (
    SelfConditionedTarget,
    SingleEntryContextProvider,
    validate_feature_dict,
)
from steering.refined_types import (
    PositiveIntLike,
    UniqueNonEmptyNonNegativeIntSequenceLike,
)

if TYPE_CHECKING:
    _RawEvaluationsLike = tuple[RawEvaluation, ...]
    _RenderedPromptLike = tuple[str, ChatCompletionPrompt]
else:
    # Avoid beartype's randomized sampling of parameterized return containers.
    _RawEvaluationsLike = tuple
    _RenderedPromptLike = tuple

_LEGACY_SPECIAL_TOKENS = tuple(
    f"<|{token[1:-1]}|>"
    for token in SPECIAL_TOKENS
    if token.startswith("<") and token.endswith(">")
)
_TARGET_SPECIAL_TOKEN_SPELLINGS = (*SPECIAL_TOKENS, *_LEGACY_SPECIAL_TOKENS)


class EvaluationRuntime(Protocol):
    """The runtime surface shared by Qwen workers and SelfConditioning cells."""

    def evaluate(
        self,
        suffix: str,
        input_ids: UniqueNonEmptyNonNegativeIntSequenceLike,
    ) -> _RawEvaluationsLike: ...  # pragma: no cover - typing declaration

    def evaluate_counter(
        self,
        counter: CounterDirection,
        magnitude: float,
        input_ids: UniqueNonEmptyNonNegativeIntSequenceLike,
    ) -> _RawEvaluationsLike: ...


# Both runtime implementations support explicit additive points with empty prompts.
JudgeFactory = Callable[..., AxBenchAdvJudgeClient]


def _valid_utf8_text(value: str, *, name: str) -> str:
    if "\x00" in value:
        raise AxBenchAdvError(f"{name} must not contain NUL")
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError as exc:
        raise AxBenchAdvError(f"{name} must be valid UTF-8 without surrogate code points") from exc
    return value


def _canonical_self_conditioning_suffix(suffix: str) -> str:
    canonical = canonicalize_suffix(suffix)
    if any(token in canonical for token in _TARGET_SPECIAL_TOKEN_SPELLINGS):
        raise AxBenchAdvError("suffix must not contain nanochat special tokens")
    return canonical


@beartype(conf=AXBENCH_ADV_BEARTYPE)
def _render_canonical_prompt(
    tokenizer: Any,
    original_prompt: str,
    canonical_suffix: str,
    *,
    max_prompt_tokens: PositiveIntLike,
) -> _RenderedPromptLike:
    original = _valid_utf8_text(original_prompt, name="original_prompt")
    attacked_raw = original if not canonical_suffix else original + "\n\n" + canonical_suffix
    rendered = render_prompt_for_chat_completion(tokenizer, attacked_raw)
    content_tokens = tokenizer.encode(attacked_raw)
    if not isinstance(content_tokens, list) or not all(
        type(token_id) is int for token_id in content_tokens
    ):
        raise AxBenchAdvError("nanochat tokenizer returned invalid content token ids")
    structural = {
        "bos": tokenizer.get_bos_token_id(),
        "user_start": tokenizer.encode_special("<user_start>"),
        "user_end": tokenizer.encode_special("<user_end>"),
        "assistant_start": tokenizer.encode_special("<assistant_start>"),
    }
    if any(type(token_id) is not int for token_id in structural.values()):
        raise AxBenchAdvError("nanochat tokenizer returned invalid structural token ids")
    expected = [
        structural["bos"],
        structural["user_start"],
        *content_tokens,
        structural["user_end"],
        structural["assistant_start"],
    ]
    if rendered.token_ids != expected:
        raise AxBenchAdvError(
            "nanochat chat rendering differs from the exact one-user-turn structural sequence"
        )
    decoded = tokenizer.decode(rendered.token_ids[2:-2])
    if decoded != attacked_raw:
        raise AxBenchAdvError(
            "nanochat user-content tokens do not decode exactly to the raw prompt"
        )
    if len(rendered.token_ids) > max_prompt_tokens:
        raise AxBenchAdvError(
            f"rendered prompt has {len(rendered.token_ids)} tokens, exceeding decoder prompt "
            f"budget {max_prompt_tokens}"
        )
    return attacked_raw, rendered


def render_self_conditioning_prompt(
    tokenizer: Any,
    original_prompt: str,
    suffix: str,
    *,
    max_prompt_tokens: PositiveIntLike,
) -> tuple[str, ChatCompletionPrompt]:
    """Append a canonical suffix to raw text and prove exact nanochat rendering."""

    canonical = _canonical_self_conditioning_suffix(suffix)
    return _render_canonical_prompt(
        tokenizer,
        original_prompt,
        canonical,
        max_prompt_tokens=max_prompt_tokens,
    )


@dataclass(frozen=True)
class _OriginalContext:
    raw_prompt: str
    rendered: ChatCompletionPrompt
    prepared: PreparedPromptContext
    rng_state: t.Tensor


def _metadata_sequence_len(setup: MethodSetup) -> int:
    model_config = setup.meta.get("model_config")
    value = model_config.get("sequence_len") if isinstance(model_config, Mapping) else None
    if type(value) is not int or value <= 0:
        raise AxBenchAdvError("target metadata model_config.sequence_len must be positive")
    return value


def _prompt_rng(setup: MethodSetup) -> t.Generator:
    selection = setup.prompt_pipeline.selection_config
    negative = setup.prompt_pipeline.negative_bundle
    if not isinstance(selection, SamplingBundle):
        raise AxBenchAdvError("SelfConditioning prompt selection must use SamplingBundle")
    if negative is None:
        raise AxBenchAdvError("SelfConditioning prompt selection must include negative sampling")
    if negative.rng is not selection.rng:
        raise AxBenchAdvError("positive and negative prompt sampling must share one generator")
    if selection.rng.device.type != "cpu":
        raise AxBenchAdvError("SelfConditioning prompt sampling generator must be on CPU")
    return selection.rng


def _source_prompts(
    frame: pd.DataFrame,
    *,
    max_concepts: int,
) -> tuple[dict[tuple[int, int], str], dict[int, str]]:
    required = {"concept_id", "input_id", "factor", "input_concept", "original_prompt"}
    if not isinstance(frame, pd.DataFrame) or not required.issubset(frame.columns):
        raise AxBenchAdvError("SelfConditioning source frame is missing required prompt columns")
    expected_keys = {
        (concept_id, input_id, float(factor))
        for concept_id in range(max_concepts)
        for input_id in range(ADV_INPUTS_PER_CONCEPT)
        for factor in ADV_FULL_FACTORS
    }
    observed_keys = {
        (int(concept_id), int(input_id), float(factor))
        for concept_id, input_id, factor in zip(
            frame["concept_id"], frame["input_id"], frame["factor"], strict=True
        )
    }
    if observed_keys != expected_keys or len(frame) != len(expected_keys):
        raise AxBenchAdvError("SelfConditioning source frame lacks the canonical key product")
    prompts: dict[tuple[int, int], str] = {}
    concepts: dict[int, str] = {}
    for concept_id in range(max_concepts):
        for input_id in range(ADV_INPUTS_PER_CONCEPT):
            rows = frame.loc[frame["concept_id"].eq(concept_id) & frame["input_id"].eq(input_id)]
            prompt_values = rows["original_prompt"].drop_duplicates().tolist()
            concept_values = rows["input_concept"].drop_duplicates().tolist()
            if len(prompt_values) != 1 or not isinstance(prompt_values[0], str):
                raise AxBenchAdvError(
                    "SelfConditioning original_prompt varies across source factors"
                )
            if (
                len(concept_values) != 1
                or not isinstance(concept_values[0], str)
                or not concept_values[0]
            ):
                raise AxBenchAdvError("SelfConditioning concept text varies across source factors")
            prompts[(concept_id, input_id)] = prompt_values[0]
            prior = concepts.setdefault(concept_id, concept_values[0])
            if prior != concept_values[0]:
                raise AxBenchAdvError("SelfConditioning concept text varies across inputs")
    return prompts, concepts


def _validate_prepared(setup: MethodSetup, prepared: PreparedPromptContext, *, label: str) -> None:
    if not prepared.segments:
        raise AxBenchAdvError(f"{label} has no real feature segments")
    for segment_index, segment in enumerate(prepared.segments):
        if len(segment.feature_ids) != len(segment.buckets):
            raise AxBenchAdvError(f"{label} segment {segment_index} has mismatched feature arrays")
        selected = {
            int(feature_id): int(bucket)
            for feature_id, bucket in zip(segment.feature_ids, segment.buckets, strict=True)
        }
        if len(selected) != len(segment.feature_ids):
            raise AxBenchAdvError(f"{label} segment {segment_index} has duplicate feature ids")
        validate_feature_dict(
            selected,
            setup.n_sae_features,
            2,
            name=f"{label} segment {segment_index}",
            feature_vocab=setup.feature_vocab,
        )


@dataclass(frozen=True)
class _ProjectGcgRow:
    raw_prompt: str
    prompt_prefix: tuple[int, ...]
    prompt_prefix_seg: tuple[int, ...]
    prompt_tail: tuple[int, ...]
    clean_prompt: tuple[int, ...]
    response: tuple[int, ...]
    response_byte_weights: t.Tensor | None
    clean_seg_map: t.Tensor
    encoder_out: PackedEncoderOutput
    target_slot: int
    source_slot: int
    clean_l2: t.Tensor | None


@dataclass(frozen=True)
class _ProjectGcgPair:
    positive: _ProjectGcgRow
    negative: _ProjectGcgRow


def _clone_packed(output: PackedEncoderOutput) -> PackedEncoderOutput:
    layout = output.layout
    return PackedEncoderOutput(
        values=output.values.detach().clone(),
        layout=PackedEncoderLayout(
            flat_indices=layout.flat_indices.detach().clone(),
            valid_mask=layout.valid_mask.detach().clone(),
            batch_indices=layout.batch_indices.detach().clone(),
            segment_indices=layout.segment_indices.detach().clone(),
            batch_size=layout.batch_size,
            n_segments=layout.n_segments,
            feature_count=layout.feature_count,
        ),
    )


def _value_embedding_modules(model: Any) -> list[tuple[int, Any]]:
    values = getattr(model, "value_embeds", None)
    if values is None:
        return []
    items: list[tuple[int, Any]] = []
    for key, module in values.items():
        try:
            numeric = int(key)
        except (TypeError, ValueError) as exc:
            raise AxBenchAdvError(f"project value embedding key is nonnumeric: {key!r}") from exc
        items.append((numeric, module))
    return sorted(items)


def _project_embedding_weight(model: Any) -> t.Tensor:
    parts = [model.transformer["wte"].weight.detach().float().cpu()]
    parts.extend(
        module.weight.detach().float().cpu() for _layer, module in _value_embedding_modules(model)
    )
    rows = {part.shape[0] for part in parts}
    if len(rows) != 1:
        raise AxBenchAdvError("project GCG embedding tables disagree in vocabulary width")
    return t.cat(parts, dim=1).contiguous()


def _project_token_bytes(tokenizer: Any, token_ids: Sequence[int]) -> t.Tensor:
    weights = [len(tokenizer.decode([int(token_id)]).encode("utf-8")) for token_id in token_ids]
    result = t.tensor(weights, dtype=t.float32)
    if not weights or int(result.sum().item()) <= 0:
        raise AxBenchAdvError("project GCG response tokens decode to zero bytes")
    return result


def _validate_project_gcg_context_budget(
    *, prompt_tokens: int, response_tokens: int, model_window: int, prompt_budget: int
) -> None:
    if prompt_tokens + response_tokens > model_window:
        raise AxBenchAdvError("project GCG training context exceeds the model window")
    if prompt_tokens > prompt_budget:
        raise AxBenchAdvError("project GCG seed exceeds the decoder prompt budget")


def _preserve_project_model(model: Any) -> AbstractContextManager[None]:
    return preserve_gcg_model_state((model,))


class _BoundSelfConditioningCell(AbstractContextManager[EvaluationRuntime]):
    def __init__(
        self,
        runtime: SelfConditioningMethodRuntime,
        *,
        method: str,
        concept_id: int,
        concept: str,
        factor: float,
        examples: Sequence[OptimizationExample],
    ) -> None:
        self._runtime = runtime
        self._closed = False
        if method not in ROOT_METHODS:
            raise AxBenchAdvError(f"unsupported root method: {method!r}")
        self._method = method
        if type(concept_id) is not int or concept_id not in runtime._concepts:
            raise AxBenchAdvError("SelfConditioning cell concept_id is invalid")
        if concept != runtime._concepts[concept_id]:
            raise AxBenchAdvError(
                "SelfConditioning cell concept text differs from validated source"
            )
        if isinstance(factor, bool) or not isinstance(factor, (int, float)):
            raise AxBenchAdvError("SelfConditioning factor must be numeric")
        self._factor = float(factor)
        if not math.isfinite(self._factor) or self._factor not in ALL_METHOD_FACTORS[method]:
            raise AxBenchAdvError(f"{method} factor is outside its declared grid")
        materialized = list(examples)
        if not all(isinstance(example, OptimizationExample) for example in materialized):
            raise AxBenchAdvError(
                "SelfConditioning cell examples must be OptimizationExample values"
            )
        if [example.input_id for example in materialized] != list(range(ADV_INPUTS_PER_CONCEPT)):
            raise AxBenchAdvError(
                "SelfConditioning cell examples must cover input ids 0..9 in order"
            )
        for example in materialized:
            if (
                example.user_message
                != runtime._originals[(concept_id, example.input_id)].raw_prompt
            ):
                raise AxBenchAdvError(
                    "SelfConditioning example text differs from validated raw prompt"
                )
        self._concept_id = concept_id
        self._concept = concept
        self._cache: dict[tuple[int, str], PreparedPromptContext] = {
            (input_id, ""): runtime._originals[(concept_id, input_id)].prepared
            for input_id in range(ADV_INPUTS_PER_CONCEPT)
        }
        self._judges: tuple[AxBenchAdvJudgeClient, ...] = ()
        self._gcg_pairs: tuple[_ProjectGcgPair, ...] = ()
        self._gcg_seed: tuple[int, ...] = ()
        self._gcg_alpha = 0.0
        self._gcg_device_batch_size = 1
        self._gcg_direction: GcgActivationDirection | None = None
        self._gcg_allowed: frozenset[int] = frozenset()
        self._soft_prepared = False

    def _policy(self):
        scores = (
            self._runtime._nl_target_scores[self._concept_id]
            if self._method == NL_METHOD
            else self._runtime._dense_diffmeans[self._concept_id]
        )
        if self._method in {"SelfConditioning", NL_METHOD}:
            return SelfCondScoreInterpolation(
                target_scores=scores,
                negative_selection=self._runtime._setup.output_negative_selection,
                weight=self_conditioning_weight(self._factor),
            )
        if self._method == "SelfConditioningMasked":
            return build_self_conditioning_masked_policy(
                self._runtime._setup,
                scores,
                self._factor,
            )
        raise AxBenchAdvError(f"unsupported root method: {self._method!r}")

    def __enter__(self) -> _BoundSelfConditioningCell:
        if self._runtime._active_cell is not None:
            raise AxBenchAdvError("only one SelfConditioning concept cell may be active")
        opened: list[AxBenchAdvJudgeClient] = []
        try:
            payload = self._runtime._judge_payload(
                method=self._method,
                concept_id=self._concept_id,
                concept=self._concept,
                factor=self._factor,
            )
            log_path = self._runtime._stderr_dir / self._method / f"c{self._concept_id}.log"
            if self._runtime._reuse_judge_workers:
                opened.extend(self._runtime._bind_judge_pool(payload))
            else:
                for index in range(self._runtime._judge_workers):
                    opened.append(
                        self._runtime._judge_factory(
                            init_payload=payload,
                            timeout_seconds=self._runtime._timeout_seconds,
                            stderr_path=(
                                log_path
                                if self._runtime._judge_workers == 1
                                else log_path.with_name(f"{log_path.stem}.worker{index}.log")
                            ),
                        )
                    )
        except BaseException as exc:
            if not self._runtime._reuse_judge_workers:
                for judge in opened:
                    try:
                        judge.close()
                    except BaseException as cleanup_exc:
                        exc.add_note(f"judge startup cleanup failed: {cleanup_exc}")
            self._cache.clear()
            raise
        self._judges = tuple(opened)
        self._runtime._active_cell = self
        return self

    def _context(self, input_id: int, suffix: str) -> PreparedPromptContext:
        canonical = _canonical_self_conditioning_suffix(suffix)
        key = (input_id, canonical)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        original = self._runtime._originals[(self._concept_id, input_id)]
        _attacked, rendered = _render_canonical_prompt(
            self._runtime._nanochat_tokenizer,
            original.raw_prompt,
            canonical,
            max_prompt_tokens=self._runtime._max_prompt_tokens,
        )
        self._runtime._prompt_rng.set_state(original.rng_state.clone())
        prepared = prepare_prompt_context(
            self._runtime._setup,
            self._runtime._extraction_model,
            self._runtime._sae,
            self._runtime._nanochat_tokenizer,
            token_ids=rendered.token_ids,
            assistant_end_id=rendered.assistant_end_id,
        )
        _validate_prepared(
            self._runtime._setup,
            prepared,
            label=f"candidate c{self._concept_id}/i{input_id}",
        )
        self._cache[key] = prepared
        return prepared

    def _prepare_gcg_row(
        self,
        *,
        prompt: str,
        response: str,
        sample_id: int,
        seed_text: str,
        objective: str,
    ) -> tuple[_ProjectGcgRow, tuple[int, ...]]:
        tokenizer = self._runtime._nanochat_tokenizer
        _clean_raw, clean_rendered = _render_canonical_prompt(
            tokenizer,
            prompt,
            "",
            max_prompt_tokens=self._runtime._max_prompt_tokens,
        )
        rng_state = self._runtime._prompt_rng.get_state().clone()
        try:
            self._runtime._prompt_rng.manual_seed(sample_id)
            prepared = prepare_prompt_context(
                self._runtime._setup,
                self._runtime._extraction_model,
                self._runtime._sae,
                tokenizer,
                token_ids=clean_rendered.token_ids,
                assistant_end_id=clean_rendered.assistant_end_id,
            )
        finally:
            self._runtime._prompt_rng.set_state(rng_state)
        config = build_self_conditioning_config(
            self._runtime._setup,
            self._runtime._target_model,
            self._policy(),
            selection_seed=sample_id,
        )
        provider = SingleEntryContextProvider(
            prepared.token_ids,
            prepared.segments,
            segment_mode=prepared.segment_mode,
            token_assignments=prepared.token_assignments,
        )
        wrapper = TargetAwareEncoderDecoderEvalWrapper(
            model=self._runtime._target_model,
            tokenizer=self._runtime._target_tokenizer,
            target=SelfConditionedTarget(),
            context_provider=provider,
            self_cond_config=config,
        )
        model = self._runtime._target_model
        device = model.get_device()
        model_cfg = model.dec_config
        dtype = t.bfloat16 if device.type == "cuda" else t.float32
        cache = KVCache(
            batch_size=1,
            num_heads=model_cfg.n_kv_head,
            seq_len=model_cfg.sequence_len,
            head_dim=model_cfg.n_embd // model_cfg.n_head,
            num_layers=model.num_cache_layers,
            device=device,
            dtype=dtype,
        )
        clean_ids = t.tensor([prepared.token_ids], dtype=t.long, device=device)
        with _preserve_project_model(model), t.no_grad():
            encoder_out, clean_seg_map = wrapper._build_conditioning(clean_ids, kv_cache=cache)
            runtime = wrapper._self_cond
            if runtime is None:
                raise AxBenchAdvError("project GCG failed to construct self-conditioning runtime")
            prompt_seed = seed_self_conditioning_from_prompt(
                model=model,
                runtime=runtime,
                input_ids=clean_ids,
                encoder_out=encoder_out,
                seg_map=clean_seg_map,
                kv_cache=cache,
            )
        state = _clone_packed(prompt_seed.state.encoder_out)
        tail = tuple(int(token) for token in clean_rendered.token_ids[-2:])
        prompt_prefix, prompt_prefix_seg, seed_ids, source_slot = project_suffix_prefix(
            tokenizer, prompt, clean_rendered.token_ids, clean_seg_map, seed_text,
        )
        response_ids = tuple(int(token) for token in tokenizer.encode(response))
        if not response_ids:
            raise AxBenchAdvError("project GCG training response tokenizes to empty")
        _validate_project_gcg_context_budget(
            prompt_tokens=len(prompt_prefix) + len(seed_ids) + len(tail),
            response_tokens=len(response_ids),
            model_window=int(model.dec_config.sequence_len),
            prompt_budget=self._runtime._max_prompt_tokens,
        )
        target_slot = prompt_seed.state.target_slot_idx
        response_byte_weights = None
        if objective == "response_margin":
            # Later backward calls must be able to save these fixed weights,
            # including when preparation runs inside inference mode.
            with t.inference_mode(False):
                response_byte_weights = _project_token_bytes(
                    self._runtime._target_tokenizer, response_ids
                ).to(device)
        return (
            _ProjectGcgRow(
                raw_prompt=prompt,
                prompt_prefix=prompt_prefix,
                prompt_prefix_seg=prompt_prefix_seg,
                prompt_tail=tail,
                clean_prompt=tuple(int(token) for token in clean_rendered.token_ids),
                response=response_ids,
                response_byte_weights=response_byte_weights,
                clean_seg_map=clean_seg_map.detach().clone(),
                encoder_out=state,
                target_slot=target_slot,
                source_slot=source_slot,
                clean_l2=None,
            ),
            seed_ids,
        )

    def prepare_gcg(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str,
    ) -> GcgPrepared:
        self._soft_prepared = False
        if separator != "\n\n" or objective not in {"response_margin", "activation_l2"}:
            raise AxBenchAdvError("project GCG received unsupported separator/objective")
        if type(device_batch_size) is not int or device_batch_size <= 0:
            raise AxBenchAdvError("project GCG device_batch_size must be positive")
        seed_text = " ".join(["!"] * suffix_token_count)
        pairs: list[_ProjectGcgPair] = []
        shared_seed: tuple[int, ...] | None = None
        for pair_id, raw_pair in enumerate(rows):
            if (
                set(raw_pair) != {"pair_id", "positive", "negative"}
                or raw_pair["pair_id"] != pair_id
            ):
                raise AxBenchAdvError("project GCG training pairs must have ordered exact fields")
            class_rows: dict[str, _ProjectGcgRow] = {}
            for label in ("positive", "negative"):
                raw = raw_pair[label]
                if not isinstance(raw, Mapping) or set(raw) != {"sample_id", "prompt", "response"}:
                    raise AxBenchAdvError("project GCG class row fields are malformed")
                if (
                    type(raw["sample_id"]) is not int
                    or not isinstance(raw["prompt"], str)
                    or not isinstance(raw["response"], str)
                ):
                    raise AxBenchAdvError("project GCG class row values are malformed")
                class_row, seed_ids = self._prepare_gcg_row(
                    prompt=raw["prompt"],
                    response=raw["response"],
                    sample_id=raw["sample_id"],
                    seed_text=seed_text,
                    objective=objective,
                )
                if len(seed_ids) != suffix_token_count:
                    raise AxBenchAdvError(
                        "project GCG independent seed does not encode to exactly k tokens"
                    )
                if shared_seed is None:
                    shared_seed = seed_ids
                elif seed_ids != shared_seed:
                    raise AxBenchAdvError("project GCG seed token ids vary across training prompts")
                class_rows[label] = class_row
            pairs.append(_ProjectGcgPair(class_rows["positive"], class_rows["negative"]))
        if not pairs or shared_seed is None:
            raise AxBenchAdvError("project GCG training population must not be empty")
        self._gcg_pairs = tuple(pairs)
        self._gcg_seed = shared_seed
        self._gcg_alpha = float(alpha)
        self._gcg_device_batch_size = device_batch_size
        self._gcg_direction = self._runtime._gcg_activation_directions.get(self._concept_id)
        if objective == "activation_l2" and self._gcg_direction is None:
            raise AxBenchAdvError("project activation-L2 requires a calibrated DiffMean direction")
        if objective == "response_margin" and self._gcg_direction is not None:
            self._gcg_direction = None
        embedding = _project_embedding_weight(self._runtime._target_model)
        tokenizer = self._runtime._target_tokenizer
        special_ids = {
            int(tokenizer.get_bos_token_id()),
            *(int(tokenizer.encode_special(name)) for name in SPECIAL_TOKENS),
        }
        allowed: list[int] = []
        for token_id in range(tokenizer.get_vocab_size()):
            if token_id in special_ids:
                continue
            try:
                text = tokenizer.decode([token_id])
            except (UnicodeError, RuntimeError, ValueError):
                continue
            if not text or "\x00" in text:
                continue
            # Keep the Unicode category policy for non-ASCII and str subclasses.
            if type(text) is str and text.isascii():
                if not text.isprintable():
                    continue
            elif any(unicodedata.category(character) in {"Cc", "Cf", "Cs"} for character in text):
                continue
            try:
                text.encode("utf-8", errors="strict")
            except UnicodeEncodeError:
                continue
            if tokenizer.encode(text) == [token_id]:
                allowed.append(token_id)
        if not allowed or any(token_id not in allowed for token_id in shared_seed):
            raise AxBenchAdvError("project GCG seed is absent from the canonical token pool")
        self._gcg_allowed = frozenset(allowed)
        layouts_payload = json.dumps(
            [
                [
                    len(row.prompt_prefix),
                    len(row.clean_prompt),
                    len(row.response),
                    row.source_slot,
                    row.target_slot,
                ]
                for pair in pairs
                for row in (pair.positive, pair.negative)
            ],
            separators=(",", ":"),
        ).encode()
        identity = {
            "runtime": "project_encoder_decoder",
            "concept_id": self._concept_id,
            "rank_sentinel_id": int(self._gcg_pairs[0].positive.clean_prompt[0]),
            "vocab_size": tokenizer.get_vocab_size(),
            "embedding_dim": embedding.shape[1],
            "method": self._method,
            "factor": self._factor,
            "frozen_state": "clean_real_prompt_S0_before_assistant_marker",
        }
        calibration = None
        if self._gcg_direction is not None:
            calibration = {
                **self._gcg_direction.provenance,
                "factor": self._factor,
                "resolved_scale": self._gcg_direction.max_activation * self._factor,
                "coefficient_semantics": (
                    "saved_diffmean * own_positive_max_activation * selected_method_factor"
                ),
            }
        return GcgPrepared(
            embedding_weight=embedding,
            allowed_token_ids=tuple(allowed),
            seed_token_ids=shared_seed,
            member_count=len(pairs),
            identity=identity,
            layouts_sha256=hashlib.sha256(layouts_payload).hexdigest(),
            direction_calibration=calibration,
        )

    def _soft_embedding_modules(self) -> tuple[tuple[str, Any], ...]:
        model = self._runtime._target_model
        return (
            ("wte", model.transformer["wte"]),
            *((f"value:{layer}", module) for layer, module in _value_embedding_modules(model)),
        )

    def prepare_soft_prompt(
        self,
        *,
        objective: str,
        alpha: float,
        rows: Sequence[Mapping[str, Any]],
        suffix_token_count: int,
        device_batch_size: int,
        separator: str,
    ) -> GcgPrepared:
        """Prepare the existing fixed surrogate while binding the continuous coordinate layout."""

        self._soft_prepared = False
        if self._closed or self._runtime._active_cell is not self:
            raise AxBenchAdvError("SelfConditioning cell is not active")
        if type(suffix_token_count) is not int or suffix_token_count <= 0:
            raise AxBenchAdvError("project soft suffix_token_count must be a positive integer")
        if (
            isinstance(alpha, bool) or not isinstance(alpha, (int, float))
            or not math.isfinite(alpha) or alpha < 0
        ):
            raise AxBenchAdvError("project soft alpha must be finite and nonnegative")
        prepared = self.prepare_gcg(
            objective=objective, alpha=alpha, rows=rows,
            suffix_token_count=suffix_token_count, device_batch_size=device_batch_size,
            separator=separator,
        )
        modules = self._soft_embedding_modules()
        validate_project_soft_embeddings(
            prepared.embedding_weight[list(prepared.seed_token_ids)], modules, suffix_token_count,
        )
        result = replace(prepared, identity={
            **prepared.identity,
            "embedding_layout": [
                {"name": name, "width": int(module.weight.shape[1]), "dtype": str(module.weight.dtype)}
                for name, module in modules
            ],
            "generation_conditioning": "clean_prompt_S0_dynamic_decode",
            "compute_dtype": str(COMPUTE_DTYPE),
        })
        self._soft_prepared = True
        return result

    def _validate_soft_request(
        self, embeddings: t.Tensor, member_indices: Sequence[int] | None = None
    ) -> None:
        if not self._soft_prepared or self._closed or self._runtime._active_cell is not self:
            raise AxBenchAdvError("prepare_soft_prompt must succeed before a soft prompt operation")
        validate_project_soft_embeddings(embeddings, self._soft_embedding_modules(), len(self._gcg_seed))
        if member_indices is not None and (
            not member_indices
            or any(type(index) is not int or not 0 <= index < len(self._gcg_pairs) for index in member_indices)
            or len(set(member_indices)) != len(member_indices)
        ):
            raise AxBenchAdvError("project soft member indices must be unique prepared pair indices")

    def gcg_filter(
        self, candidate_token_ids: Sequence[Sequence[int]]
    ) -> tuple[GcgFilteredCandidate, ...]:
        tokenizer = self._runtime._target_tokenizer
        out: list[GcgFilteredCandidate] = []
        for raw_ids in candidate_token_ids:
            ids: tuple[int, ...] = ()
            error: str | None = None
            text = ""
            try:
                if any(type(token_id) is not int for token_id in raw_ids):
                    raise AxBenchAdvError("candidate token ids must be exact integers")
                ids = tuple(raw_ids)
                if len(ids) != len(self._gcg_seed):
                    raise AxBenchAdvError("candidate token count differs from k")
                if any(token_id not in self._gcg_allowed for token_id in ids):
                    raise AxBenchAdvError("candidate token id is outside the canonical pool")
                text = tokenizer.decode(list(ids))
                if _canonical_self_conditioning_suffix(text) != text or tokenizer.encode(
                    text
                ) != list(ids):
                    raise AxBenchAdvError("candidate text does not round-trip canonically")
                for pair in self._gcg_pairs:
                    for row in (pair.positive, pair.negative):
                        _raw, rendered = _render_canonical_prompt(
                            tokenizer,
                            row.raw_prompt,
                            text,
                            max_prompt_tokens=self._runtime._max_prompt_tokens,
                        )
                        expected = [*row.prompt_prefix, *ids, *row.prompt_tail]
                        if rendered.token_ids != expected:
                            raise AxBenchAdvError("candidate changes fixed project prompt framing")
            except (AxBenchAdvError, UnicodeError, RuntimeError, ValueError) as exc:
                error = str(exc)
            out.append(GcgFilteredCandidate(ids, text, error is None, error))
        return tuple(out)

    def _project_batch_values(
        self,
        items: Sequence[tuple[_ProjectGcgRow, tuple[int, ...] | None]],
        *,
        gradient: bool,
        embeddings: t.Tensor | None = None,
    ) -> list[tuple[t.Tensor, t.Tensor | None]]:
        """Evaluate one compatible packed-state microbatch in input order."""

        if not items:
            return []
        model = self._runtime._target_model
        device = model.get_device()
        direction = self._gcg_direction
        capture_layer = None if direction is None else int(direction.provenance["layer"])
        n_segments = {row.encoder_out.layout.n_segments for row, _tokens in items}
        if len(n_segments) != 1:
            raise AxBenchAdvError("project GCG microbatch mixes packed segment counts")

        sequences: list[list[int]] = []
        prompts: list[list[int]] = []
        suffix_starts: list[int] = []
        seg_rows: list[list[int]] = []
        for row, maybe_tokens in items:
            clean = maybe_tokens is None
            tokens = () if clean else maybe_tokens
            assert tokens is not None
            prompt = (
                list(row.clean_prompt) if clean else [*row.prompt_prefix, *tokens, *row.prompt_tail]
            )
            all_tokens = [*prompt, *row.response]
            full_length = len(all_tokens) if capture_layer is not None else len(all_tokens) - 1
            if clean:
                seg_values = [
                    *row.clean_seg_map[0, :-1].tolist(),
                    row.target_slot,
                    *([row.target_slot] * (full_length - len(prompt))),
                ]
            else:
                seg_values = [
                    *row.prompt_prefix_seg,
                    *([row.source_slot] * len(tokens)),
                    *([row.source_slot] * (len(row.prompt_tail) - 1)),
                    row.target_slot,
                    *([row.target_slot] * (full_length - len(prompt))),
                ]
            if len(seg_values) != full_length:
                raise AxBenchAdvError("project GCG frozen routing length mismatch")
            prompts.append(prompt)
            sequences.append(all_tokens)
            seg_rows.append(seg_values)
            suffix_starts.append(len(row.clean_prompt) - 2 if clean else len(row.prompt_prefix))

        lengths = [
            len(values) if capture_layer is not None else len(values) - 1 for values in sequences
        ]
        maximum = max(lengths)
        pad_id = int(self._runtime._target_tokenizer.get_bos_token_id())
        dec_input = t.full((len(items), maximum), pad_id, dtype=t.long, device=device)
        seg_map = t.empty((len(items), maximum), dtype=t.long, device=device)
        targets = (
            None
            if capture_layer is not None
            else t.full((len(items), maximum), -1, dtype=t.long, device=device)
        )
        for index, (sequence, seg_values, length) in enumerate(
            zip(sequences, seg_rows, lengths, strict=True)
        ):
            dec_input[index, :length] = t.tensor(sequence[:length], dtype=t.long, device=device)
            seg_map[index, :length] = t.tensor(seg_values, dtype=t.long, device=device)
            seg_map[index, length:] = seg_values[-1]
            if targets is not None:
                targets[index, :length] = t.tensor(sequence[1:], dtype=t.long, device=device)

        captured: dict[str, t.Tensor] = {}
        soft_parts = (
            dict(zip(
                (name for name, _module in self._soft_embedding_modules()),
                embeddings.split(
                    [int(module.weight.shape[1]) for _name, module in self._soft_embedding_modules()],
                    dim=1,
                ), strict=True,
            ))
            if embeddings is not None else {}
        )

        def capture_embedding(name: str):
            def hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> t.Tensor | None:
                if not isinstance(output, t.Tensor):
                    raise AxBenchAdvError(f"project GCG embedding {name} returned non-tensor")
                if embeddings is not None:
                    output = replace_soft_embedding_rows(output, soft_parts[name], suffix_starts)
                if gradient and not output.requires_grad:
                    output.requires_grad_(True)
                captured[name] = output
                return output if embeddings is not None else None

            return hook

        value_modules = _value_embedding_modules(model)
        hooks = [model.transformer["wte"].register_forward_hook(capture_embedding("wte"))]
        hooks.extend(
            module.register_forward_hook(capture_embedding(f"value:{layer}"))
            for layer, module in value_modules
        )
        activation: list[t.Tensor] = []

        def capture_activation(layer: int, residual: t.Tensor) -> t.Tensor:
            if layer == capture_layer:
                activation.append(residual)
            return residual

        try:
            with _preserve_project_model(model):
                encoder_out = stack_packed_encoder_outputs(
                    tuple(row.encoder_out for row, _tokens in items)
                )
                if capture_layer is None:
                    assert targets is not None
                    output = model(
                        dec_input,
                        targets,
                        seg_map=seg_map,
                        encoder_out=encoder_out,
                        encoder_strict=True,
                        loss_reduction="none",
                    ).view(len(items), maximum)
                else:
                    with scoped_post_cross_attention_callback(model, capture_activation):
                        model(
                            dec_input,
                            None,
                            seg_map=seg_map,
                            encoder_out=encoder_out,
                            encoder_strict=True,
                            logits_to_keep=1,
                        )
                    if len(activation) != 1:
                        raise AxBenchAdvError(
                            "project GCG activation capture count differs from one"
                        )
                    output = activation[0]

                roots = [
                    captured["wte"],
                    *(captured[f"value:{layer}"] for layer, _module in value_modules),
                ]
                objectives: list[t.Tensor] = []
                for index, ((row, maybe_tokens), sequence, prompt, suffix_start) in enumerate(
                    zip(items, sequences, prompts, suffix_starts, strict=True)
                ):
                    tokens = () if maybe_tokens is None else maybe_tokens
                    if direction is None:
                        losses = output[index, len(prompt) - 1 : len(sequence) - 1]
                        assert row.response_byte_weights is not None
                        weights = row.response_byte_weights.to(output.device)
                        objective = weighted_bpb(losses, weights, tensor_ln2=True)
                    else:
                        tail_start = suffix_start + len(tokens)
                        attacked_hidden = output[index, tail_start : len(sequence)]
                        if row.clean_l2 is None and maybe_tokens is None:
                            objectives.append(attacked_hidden)
                            continue
                        if row.clean_l2 is None:
                            raise AxBenchAdvError(
                                "project activation-L2 clean reference is unavailable"
                            )
                        clean_hidden = row.clean_l2.to(attacked_hidden.device)
                        runtime_direction = direction.direction.to(attacked_hidden.dtype)
                        offset = (direction.max_activation * self._factor) * runtime_direction
                        objective = activation_l2_row(attacked_hidden, clean_hidden, offset)
                    objectives.append(objective)

                results: list[tuple[t.Tensor, t.Tensor | None]] = []
                for index, (objective, (_row, maybe_tokens), suffix_start) in enumerate(
                    zip(objectives, items, suffix_starts, strict=True)
                ):
                    if not gradient:
                        results.append((objective.detach(), None))
                        continue
                    assert maybe_tokens is not None
                    root_gradients = t.autograd.grad(
                        objective,
                        roots,
                        allow_unused=True,
                        retain_graph=index + 1 < len(objectives),
                    )
                    parts: list[t.Tensor] = []
                    for root_index, root_gradient in enumerate(root_gradients):
                        layer = -1 if root_index == 0 else value_modules[root_index - 1][0]
                        if root_gradient is None:
                            if direction is not None and layer > capture_layer:
                                parts.append(
                                    t.zeros(
                                        (len(maybe_tokens), roots[root_index].shape[-1]),
                                        dtype=t.float32,
                                    )
                                )
                                continue
                            raise AxBenchAdvError("project GCG missing upstream embedding gradient")
                        parts.append(
                            root_gradient[index, suffix_start : suffix_start + len(maybe_tokens)]
                            .detach()
                            .float()
                            .cpu()
                        )
                    results.append((objective.detach(), t.cat(parts, dim=1)))
                return results
        finally:
            for hook in hooks:
                hook.remove()

    def _batched_project_values(
        self,
        items: Sequence[tuple[_ProjectGcgRow, tuple[int, ...] | None]],
        *,
        gradient: bool,
        batch_size: int,
        embeddings: t.Tensor | None = None,
    ) -> list[tuple[t.Tensor, t.Tensor | None]]:
        results: list[tuple[t.Tensor, t.Tensor | None] | None] = [None] * len(items)
        groups: dict[int, list[int]] = {}
        for index, (row, _tokens) in enumerate(items):
            groups.setdefault(row.encoder_out.layout.n_segments, []).append(index)
        for indices in groups.values():
            for start in range(0, len(indices), batch_size):
                selected = indices[start : start + batch_size]
                values = self._project_batch_values(
                    [items[index] for index in selected], gradient=gradient,
                    embeddings=embeddings,
                )
                for index, value in zip(selected, values, strict=True):
                    results[index] = value
        if any(result is None for result in results):
            raise AxBenchAdvError("project GCG failed to fill a batched result")
        return [result for result in results if result is not None]

    def _ensure_clean_l2(self) -> None:
        if self._gcg_direction is None:
            return
        if all(
            row.clean_l2 is not None
            for pair in self._gcg_pairs
            for row in (pair.positive, pair.negative)
        ):
            return
        original_rows = [row for pair in self._gcg_pairs for row in (pair.positive, pair.negative)]
        with t.no_grad():
            clean_values = self._batched_project_values(
                [(row, None) for row in original_rows],
                gradient=False,
                batch_size=self._gcg_device_batch_size,
            )
        clean_rows = [
            replace(row, clean_l2=value.detach().float().cpu().contiguous())
            for row, (value, _gradient) in zip(original_rows, clean_values, strict=True)
        ]
        updated = [
            _ProjectGcgPair(clean_rows[index], clean_rows[index + 1])
            for index in range(0, len(clean_rows), 2)
        ]
        self._gcg_pairs = tuple(updated)

    def gcg_score(
        self,
        candidate_token_ids: Sequence[Sequence[int]],
        member_indices: Sequence[int],
        *,
        candidate_eval_device_batch_size: int,
        embeddings: t.Tensor | None = None,
    ) -> list[float]:
        self._ensure_clean_l2()
        if candidate_eval_device_batch_size <= 0:
            raise AxBenchAdvError("project GCG score batch size must be positive")
        items: list[tuple[_ProjectGcgRow, tuple[int, ...] | None]] = []
        for raw_tokens in candidate_token_ids:
            tokens = tuple(raw_tokens)
            for member_index in member_indices:
                pair = self._gcg_pairs[member_index]
                items.extend(((pair.negative, tokens), (pair.positive, tokens)))
        with t.no_grad():
            sides = self._batched_project_values(
                items,
                gradient=False,
                batch_size=candidate_eval_device_batch_size,
                embeddings=embeddings,
            )
        values: list[float] = []
        for offset in range(0, len(sides), 2):
            negative, positive = sides[offset][0], sides[offset + 1][0]
            pair_value = (
                activation_l2_pair(positive, negative)
                if self._gcg_direction is not None
                else response_margin_value(negative, positive, alpha=self._gcg_alpha)
            )
            value = float(pair_value)
            if not math.isfinite(value):
                raise AxBenchAdvError("project GCG score is non-finite")
            values.append(value)
        return values

    def gcg_gradient(
        self,
        token_ids: Sequence[int],
        member_indices: Sequence[int],
        *,
        embeddings: t.Tensor | None = None,
    ) -> list[t.Tensor]:
        self._ensure_clean_l2()
        tokens = tuple(token_ids)
        items: list[tuple[_ProjectGcgRow, tuple[int, ...] | None]] = []
        for member_index in member_indices:
            pair = self._gcg_pairs[member_index]
            items.extend(((pair.negative, tokens), (pair.positive, tokens)))
        sides = self._batched_project_values(
            items,
            gradient=True,
            batch_size=self._gcg_device_batch_size,
            embeddings=embeddings,
        )
        gradients: list[t.Tensor] = []
        for offset in range(0, len(sides), 2):
            negative, positive = sides[offset][1], sides[offset + 1][1]
            assert positive is not None and negative is not None
            gradient = (
                0.5 * (positive + negative)
                if self._gcg_direction is not None
                else response_margin_gradient(negative, positive, alpha=self._gcg_alpha)
            )
            if not bool(t.isfinite(gradient).all().item()):
                raise AxBenchAdvError("project GCG gradient is non-finite")
            if embeddings is not None and gradient.shape != embeddings.shape:
                raise AxBenchAdvError("project soft gradient has invalid shape")
            gradients.append(gradient)
        return gradients

    def soft_score(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[float]:
        """Score one continuous suffix on the requested ordered training pairs."""

        self._validate_soft_request(embeddings, member_indices)
        return self.gcg_score(
            [self._gcg_seed], member_indices,
            candidate_eval_device_batch_size=self._gcg_device_batch_size,
            embeddings=embeddings,
        )

    def soft_gradient(self, embeddings: t.Tensor, member_indices: Sequence[int]) -> list[t.Tensor]:
        """Return per-pair suffix VJPs in main-then-value-table coordinate order."""

        self._validate_soft_request(embeddings, member_indices)
        with t.inference_mode(False), t.enable_grad():
            return self.gcg_gradient(
                self._gcg_seed, member_indices, embeddings=embeddings,
            )

    @beartype(conf=AXBENCH_ADV_BEARTYPE)
    def evaluate(
        self,
        suffix: str,
        input_ids: UniqueNonEmptyNonNegativeIntSequenceLike,
    ) -> _RawEvaluationsLike:
        if self._closed or self._runtime._active_cell is not self or not self._judges:
            raise AxBenchAdvError("SelfConditioning cell is not active")
        if any(input_id >= ADV_INPUTS_PER_CONCEPT for input_id in input_ids):
            raise AxBenchAdvError("SelfConditioning evaluate input_ids must be integers in 0..9")

        def generate(input_id: int) -> str:
            prepared = self._context(input_id, suffix)
            selection_seed = derive_selection_seed(
                self._runtime._master_seed, self._concept_id, input_id
            )
            token_seed = derive_generation_seed(
                self._runtime._master_seed,
                self._method,
                self._concept_id,
                input_id,
                self._factor,
            )
            policy = self._policy()
            return generate_self_conditioning(
                self._runtime._setup,
                self._runtime._target_model,
                self._runtime._target_tokenizer,
                prepared,
                policy,
                selection_seed=selection_seed,
                seed=token_seed,
                max_new_tokens=self._runtime._max_new_tokens,
                **(
                    {"reuse_single_token_mask": True}
                    if self._runtime._reuse_single_token_mask
                    else {}
                ),
            )

        return self._evaluate_generations(input_ids, generate)

    @beartype(conf=AXBENCH_ADV_BEARTYPE)
    def evaluate_soft(
        self, embeddings: t.Tensor, input_ids: UniqueNonEmptyNonNegativeIntSequenceLike
    ) -> _RawEvaluationsLike:
        """Generate with clean SAE/S0, an attacked decoder prefix, and normal later refreshes."""

        self._validate_soft_request(embeddings)
        if not self._judges or any(input_id >= ADV_INPUTS_PER_CONCEPT for input_id in input_ids):
            raise AxBenchAdvError("SelfConditioning soft evaluation needs active judges and input ids 0..9")
        seed_text = self._runtime._target_tokenizer.decode(list(self._gcg_seed))

        def generate(input_id: int) -> str:
            original = self._runtime._originals[(self._concept_id, input_id)]
            prepared = self._context(input_id, "")
            _raw, attacked = _render_canonical_prompt(
                self._runtime._nanochat_tokenizer, original.raw_prompt, seed_text,
                max_prompt_tokens=self._runtime._max_prompt_tokens,
            )
            selection_seed = derive_selection_seed(
                self._runtime._master_seed, self._concept_id, input_id,
            )
            token_seed = derive_generation_seed(
                self._runtime._master_seed, self._method, self._concept_id, input_id, self._factor,
            )
            config = build_self_conditioning_config(
                self._runtime._setup, self._runtime._target_model, self._policy(),
                selection_seed=selection_seed,
            )
            return generate_project_soft_prompt(
                model=self._runtime._target_model, tokenizer=self._runtime._target_tokenizer,
                prepared=prepared, self_cond_config=config,
                attacked_token_ids=attacked.token_ids, raw_prompt=original.raw_prompt,
                seed_text=seed_text, embeddings=embeddings,
                embedding_modules=self._soft_embedding_modules(), seed=token_seed,
                max_new_tokens=self._runtime._max_new_tokens,
                reuse_single_token_mask=self._runtime._reuse_single_token_mask,
            )

        return self._evaluate_generations(input_ids, generate)

    def _evaluate_generations(
        self, input_ids: Sequence[int], generate: Callable[[int], str]
    ) -> tuple[RawEvaluation, ...]:
        """Judge generated texts through the same bounded, ordered existing worker path."""

        def checked(judged: RawEvaluation, generation: str) -> RawEvaluation:
            if judged.generation != generation:
                raise AxBenchAdvError(f"{self._method} judge changed the generated text")
            return judged

        available: SimpleQueue[AxBenchAdvJudgeClient] = SimpleQueue()
        for judge in self._judges:
            available.put(judge)

        def judge_generation(generation: str, input_id: int) -> RawEvaluation:
            judge = available.get()
            try:
                return judge.evaluate(generation=generation, input_id=input_id)
            finally:
                available.put(judge)

        results: list[RawEvaluation] = []
        if self._runtime._judge_inflight == 1 or len(input_ids) == 1:
            for input_id in input_ids:
                generation = generate(input_id)
                results.append(
                    checked(judge_generation(generation, input_id), generation)
                )
            return tuple(results)

        pending: list[tuple[str, Future[RawEvaluation]]] = []
        with ThreadPoolExecutor(
            max_workers=len(self._judges), thread_name_prefix="axbench-adv-judge"
        ) as executor:
            try:
                for input_id in input_ids:
                    generation = generate(input_id)
                    # The cache key includes the generated text. A duplicate
                    # must observe the first response's committed cache rows.
                    while any(text == generation for text, _future in pending):
                        text, future = pending.pop(0)
                        results.append(checked(future.result(), text))
                    if pending and pending[0][1].done():
                        text, future = pending.pop(0)
                        results.append(checked(future.result(), text))
                    pending.append(
                        (
                            generation,
                            executor.submit(
                                judge_generation, generation, input_id
                            ),
                        )
                    )
                    if len(pending) == self._runtime._judge_inflight:
                        text, future = pending.pop(0)
                        results.append(checked(future.result(), text))
                for text, future in pending:
                    results.append(checked(future.result(), text))
            except BaseException:
                for _text, future in pending:
                    future.cancel()
                raise
        return tuple(results)

    def evaluate_counter(
        self,
        counter: CounterDirection,
        magnitude: float,
        input_ids: UniqueNonEmptyNonNegativeIntSequenceLike,
    ) -> _RawEvaluationsLike:
        with post_cross_attention_residual_counter(
            self._runtime._target_model,
            layer=counter.provenance["layer"],
            direction=counter.direction,
            magnitude=magnitude,
        ):
            return self.evaluate("", input_ids)

    def close(self) -> None:
        if self._closed:
            return
        error: BaseException | None = None
        if not self._runtime._reuse_judge_workers:
            for judge in self._judges:
                try:
                    judge.close()
                except BaseException as exc:
                    if error is None:
                        error = exc
                    else:
                        error.add_note(f"additional judge cleanup failure: {exc}")
        self._judges = ()
        self._cache.clear()
        self._gcg_pairs = ()
        self._gcg_seed = ()
        self._gcg_allowed = frozenset()
        self._gcg_direction = None
        self._soft_prepared = False
        if self._runtime._active_cell is self:
            self._runtime._active_cell = None
        self._closed = True
        if error is not None:
            raise error

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        if exc_value is not None and self._runtime._reuse_judge_workers:
            try:
                self._runtime._close_judge_pool(abort=True)
            except BaseException as cleanup_exc:
                exc_value.add_note(f"judge pool abort failed: {cleanup_exc}")
        try:
            self.close()
        except BaseException as cleanup_exc:
            if exc_value is None:
                raise
            exc_value.add_note(
                f"SelfConditioning cell cleanup failed: {type(cleanup_exc).__name__}: {cleanup_exc}"
            )


class SelfConditioningMethodRuntime(AbstractContextManager["SelfConditioningMethodRuntime"]):
    """One co-resident extraction/target runtime reused across concept cells."""

    def __init__(
        self,
        *,
        setup: MethodSetup,
        training: Mapping[int, ConceptTrainingTexts],
        main_frame: pd.DataFrame,
        max_concepts: int,
        max_new_tokens: int,
        master_seed: int,
        device: str,
        extraction_batch_size: int,
        judge_factory: JudgeFactory,
        judge_cache: Path,
        judge_model: str,
        api_base_url: str,
        cache_only: bool,
        timeout_seconds: float,
        stderr_dir: Path,
        judge_inflight: int = 1,
        judge_workers: int = 1,
        judge_pair_concurrency: int = 1,
        reuse_judge_workers: bool = False,
        prewarm_judge_cell: tuple[str, int, str, float] | None = None,
        reuse_single_token_mask: bool = False,
        bf16_decoder_linears: bool = False,
        reuse_diffmean_texts: bool = False,
        weight_free_extraction: bool = False,
        native_extraction: bool = False,
        fit_concept_ids: Sequence[int] | None = None,
        nl_target_scores: Mapping[int, t.Tensor] | None = None,
        gcg_activation_directions: Mapping[int, GcgActivationDirection] | None = None,
    ) -> None:
        self._setup = setup
        self._training = dict(training)
        self._max_new_tokens = max_new_tokens
        self._master_seed = master_seed
        self._device = device
        self._judge_factory = judge_factory
        self._judge_cache = Path(judge_cache).resolve()
        self._judge_model = judge_model
        self._api_base_url = api_base_url.strip().rstrip("/")
        self._cache_only = cache_only
        self._timeout_seconds = timeout_seconds
        if type(judge_inflight) is not int or judge_inflight not in (1, 2, 3):
            raise AxBenchAdvError("SelfConditioning judge_inflight must be 1, 2, or 3")
        self._judge_inflight = judge_inflight
        if type(judge_workers) is not int or judge_workers not in (1, 2):
            raise AxBenchAdvError("SelfConditioning judge_workers must be 1 or 2")
        if judge_workers == 2 and judge_inflight < 2:
            raise AxBenchAdvError("SelfConditioning two judge_workers require judge_inflight>=2")
        if judge_inflight == 3 and judge_workers != 2:
            raise AxBenchAdvError("SelfConditioning judge_inflight=3 requires two judge_workers")
        self._judge_workers = judge_workers
        if type(judge_pair_concurrency) is not int or judge_pair_concurrency not in (1, 2):
            raise AxBenchAdvError("SelfConditioning judge_pair_concurrency must be 1 or 2")
        self._judge_pair_concurrency = judge_pair_concurrency
        if type(reuse_judge_workers) is not bool:
            raise AxBenchAdvError("SelfConditioning reuse_judge_workers must be bool")
        if prewarm_judge_cell is not None and not reuse_judge_workers:
            raise AxBenchAdvError("prewarm_judge_cell requires reuse_judge_workers")
        self._reuse_judge_workers = reuse_judge_workers
        self._judge_pool: tuple[AxBenchAdvJudgeClient, ...] = ()
        if type(reuse_single_token_mask) is not bool:
            raise AxBenchAdvError("SelfConditioning reuse_single_token_mask must be bool")
        self._reuse_single_token_mask = reuse_single_token_mask
        if type(bf16_decoder_linears) is not bool:
            raise AxBenchAdvError("SelfConditioning bf16_decoder_linears must be bool")
        if bf16_decoder_linears and (
            t.device(device).type != "cuda" or COMPUTE_DTYPE != t.bfloat16
        ):
            raise AxBenchAdvError("bf16_decoder_linears requires CUDA BF16 evaluation")
        self._bf16_decoder_linears = bf16_decoder_linears
        if type(reuse_diffmean_texts) is not bool:
            raise AxBenchAdvError("SelfConditioning reuse_diffmean_texts must be bool")
        if type(weight_free_extraction) is not bool:
            raise AxBenchAdvError("SelfConditioning weight_free_extraction must be bool")
        if type(native_extraction) is not bool:
            raise AxBenchAdvError("SelfConditioning native_extraction must be bool")
        if native_extraction and (not weight_free_extraction or t.device(device).type != "cuda"):
            raise AxBenchAdvError("native_extraction requires weight-free CUDA extraction")
        self._stderr_dir = Path(stderr_dir).resolve()
        self._active_cell: _BoundSelfConditioningCell | None = None
        self._closed = False
        self._extraction_model: Any = None
        self._sae: Any = None
        self._target_model: Any = None
        self._target_tokenizer: Any = None
        self._dense_diffmeans: dict[int, t.Tensor] = {}
        self._nl_target_scores = {
            concept_id: scores.clone() for concept_id, scores in (nl_target_scores or {}).items()
        }
        self._gcg_activation_directions = dict(gcg_activation_directions or {})
        self._originals: dict[tuple[int, int], _OriginalContext] = {}
        self._concepts: dict[int, str] = {}
        prewarm_executor: ThreadPoolExecutor | None = None
        prewarm_future: Future[tuple[AxBenchAdvJudgeClient, ...]] | None = None
        try:
            if prewarm_judge_cell is not None:
                method, concept_id, concept, factor = prewarm_judge_cell
                payload = self._judge_payload(
                    method=method, concept_id=concept_id, concept=concept, factor=factor,
                )
                prewarm_executor = ThreadPoolExecutor(max_workers=1)
                prewarm_future = prewarm_executor.submit(self._start_judge_pool, payload)
            if nl_target_scores is not None:
                width = (
                    setup.n_sae_features
                    if setup.feature_vocab is None
                    else setup.feature_vocab.size
                )
                if set(self._nl_target_scores) != set(range(max_concepts)) or any(
                    scores.dtype is not t.float32
                    or scores.device.type != "cpu"
                    or scores.shape != (width,)
                    or not scores.is_contiguous()
                    or not bool(t.isfinite(scores).all())
                    or bool((scores.abs() > 1).any())
                    for scores in self._nl_target_scores.values()
                ):
                    raise AxBenchAdvError("NLSteering target tensors violate the source contract")
            self._prompt_rng = _prompt_rng(setup)
            if type(max_concepts) is not int or max_concepts <= 0:
                raise AxBenchAdvError("SelfConditioning max_concepts must be positive")
            if set(self._training) != set(range(max_concepts)):
                raise AxBenchAdvError("SelfConditioning training concepts are incomplete")
            if fit_concept_ids is not None and (
                not fit_concept_ids
                or any(type(concept_id) is not int for concept_id in fit_concept_ids)
                or len(set(fit_concept_ids)) != len(fit_concept_ids)
                or not set(fit_concept_ids).issubset(self._training)
            ):
                raise AxBenchAdvError(
                    "SelfConditioning fit_concept_ids must be unique source concept IDs"
                )
            if (
                isinstance(max_new_tokens, bool)
                or not isinstance(max_new_tokens, int)
                or max_new_tokens <= 0
            ):
                raise AxBenchAdvError("SelfConditioning max_new_tokens must be positive")
            if type(master_seed) is not int or not 0 <= master_seed <= (1 << 63) - 1:
                raise AxBenchAdvError("SelfConditioning master_seed must be in [0, 2^63 - 1]")
            if type(extraction_batch_size) is not int or extraction_batch_size <= 0:
                raise AxBenchAdvError("SelfConditioning extraction_batch_size must be positive")
            if reuse_diffmean_texts and extraction_batch_size != 1:
                raise AxBenchAdvError("SelfConditioning reuse_diffmean_texts requires extraction_batch_size=1")
            sequence_len = _metadata_sequence_len(setup)
            self._max_prompt_tokens = sequence_len - max_new_tokens
            if self._max_prompt_tokens <= 0:
                raise AxBenchAdvError("SelfConditioning generation budget exhausts decoder context")
            prompts, self._concepts = _source_prompts(main_frame, max_concepts=max_concepts)
            for concept_id, examples in self._training.items():
                if examples.concept != self._concepts[concept_id]:
                    raise AxBenchAdvError("SelfConditioning training concept differs from source")
            self._nanochat_tokenizer = get_tokenizer()
            rendered_by_pair: dict[tuple[int, int], ChatCompletionPrompt] = {}
            for pair in sorted(prompts):
                _raw, rendered_by_pair[pair] = render_self_conditioning_prompt(
                    self._nanochat_tokenizer,
                    prompts[pair],
                    "",
                    max_prompt_tokens=self._max_prompt_tokens,
                )
            self._extraction_model, self._sae = (
                load_weight_free_gemma3_extraction_stack(
                    sae_model=setup.sae_model, resolved_sae=setup.resolved_sae, device=device,
                    **(
                        {"native": True, "sae_layer": setup.sae_layer}
                        if native_extraction else {}
                    ),
                )
                if weight_free_extraction
                else load_extraction_stack(setup, device)
            )
            prepared_by_tokens: dict[tuple[int, ...], tuple[PreparedPromptContext, t.Tensor]] = {}
            for pair in sorted(prompts):
                rendered = rendered_by_pair[pair]
                exact_tokens = tuple(rendered.token_ids)
                cached = prepared_by_tokens.get(exact_tokens)
                if cached is None:
                    state = self._prompt_rng.get_state().clone()
                    prepared = prepare_prompt_context(
                        setup,
                        self._extraction_model,
                        self._sae,
                        self._nanochat_tokenizer,
                        token_ids=rendered.token_ids,
                        assistant_end_id=rendered.assistant_end_id,
                    )
                    cached = (prepared, state)
                    prepared_by_tokens[exact_tokens] = cached
                self._originals[pair] = _OriginalContext(
                    raw_prompt=prompts[pair],
                    rendered=rendered,
                    prepared=cached[0],
                    rng_state=cached[1].clone(),
                )
            self._dense_diffmeans = fit_concept_dense_diffmeans(
                setup,
                self._extraction_model,
                self._sae,
                (
                    self._training
                    if fit_concept_ids is None
                    else {
                        cid: examples
                        for cid, examples in self._training.items()
                        if cid in fit_concept_ids
                    }
                ),
                extraction_batch_size=extraction_batch_size,
                reuse_duplicate_texts=reuse_diffmean_texts,
            )
            for concept_id, scores in self._dense_diffmeans.items():
                assert_masked_diffmean_feasible(
                    self._setup,
                    scores,
                    concept_id=concept_id,
                )
            self._target_model, self._target_tokenizer = load_target_model(setup, device)
            if int(self._target_model.dec_config.sequence_len) != sequence_len:
                raise AxBenchAdvError("loaded target decoder sequence length differs from metadata")
            if self._bf16_decoder_linears:
                with t.no_grad():
                    for block in self._target_model._decoder_blocks:
                        for module in block.modules():
                            if isinstance(module, Linear):
                                module.to(dtype=t.bfloat16)
            for pair, original in self._originals.items():
                assert_prepared_prompt_matches(
                    self._target_tokenizer,
                    original.raw_prompt,
                    original.prepared,
                )
                _validate_prepared(
                    setup, original.prepared, label=f"original c{pair[0]}/i{pair[1]}"
                )
            if prewarm_future is not None:
                self._judge_pool = prewarm_future.result()
        except BaseException as exc:
            if prewarm_future is not None and not self._judge_pool:
                try:
                    self._judge_pool = prewarm_future.result()
                except BaseException as startup_exc:
                    exc.add_note(f"judge prewarm failed: {type(startup_exc).__name__}: {startup_exc}")
            self._cleanup(primary=exc)
            raise
        finally:
            if prewarm_executor is not None:
                prewarm_executor.shutdown(wait=True)

    def _judge_payload(
        self, *, method: str, concept_id: int, concept: str, factor: float
    ) -> dict[str, Any]:
        if method not in ROOT_METHODS:
            raise AxBenchAdvError(f"unsupported root method: {method!r}")
        payload = {
            "api_base_url": self._api_base_url,
            "cache_only": self._cache_only,
            "concept": concept,
            "concept_id": concept_id,
            "factor": factor,
            "judge_cache": str(self._judge_cache),
            "judge_model": self._judge_model,
            "method": method,
            "runtime_kind": "judge",
        }
        if self._judge_pair_concurrency != 1:
            payload["pair_concurrency"] = self._judge_pair_concurrency
        return payload

    def _close_judge_pool(self, *, abort: bool) -> None:
        pool, self._judge_pool = self._judge_pool, ()
        error: BaseException | None = None
        for judge in pool:
            try:
                judge.abort() if abort else judge.close()
            except BaseException as exc:
                if error is None:
                    error = exc
                else:
                    error.add_note(f"additional judge pool cleanup failure: {exc}")
        if error is not None:
            raise error

    def _start_judge_pool(self, payload: Mapping[str, Any]) -> tuple[AxBenchAdvJudgeClient, ...]:
        opened: list[AxBenchAdvJudgeClient] = []
        try:
            for index in range(self._judge_workers):
                opened.append(
                    self._judge_factory(
                        init_payload=payload,
                        timeout_seconds=self._timeout_seconds,
                        stderr_path=self._stderr_dir / "judge-pool" / f"worker{index}.log",
                    )
                )
        except BaseException as exc:
            for judge in opened:
                try:
                    judge.abort()
                except BaseException as cleanup_exc:
                    exc.add_note(f"judge pool startup cleanup failed: {cleanup_exc}")
            raise
        return tuple(opened)

    def _bind_judge_pool(self, payload: Mapping[str, Any]) -> tuple[AxBenchAdvJudgeClient, ...]:
        if self._judge_pool:
            try:
                for judge in self._judge_pool:
                    judge.rebind(payload)
            except BaseException as exc:
                try:
                    self._close_judge_pool(abort=True)
                except BaseException as cleanup_exc:
                    exc.add_note(f"judge pool abort failed: {cleanup_exc}")
                raise
            return self._judge_pool
        self._judge_pool = self._start_judge_pool(payload)
        return self._judge_pool

    def bind_cell(
        self,
        *,
        method: str,
        concept_id: int,
        concept: str,
        factor: float,
        examples: Sequence[OptimizationExample],
    ) -> AbstractContextManager[EvaluationRuntime]:
        if self._closed:
            raise AxBenchAdvError("SelfConditioning method runtime is closed")
        if method not in ROOT_METHODS:
            raise AxBenchAdvError(f"unsupported root method: {method!r}")
        if method == NL_METHOD and not self._nl_target_scores:
            raise AxBenchAdvError("NLSteering was not enabled for this runtime")
        if method != NL_METHOD and concept_id not in self._dense_diffmeans:
            raise AxBenchAdvError(
                "SelfConditioning concept has no fitted DiffMean direction"
            )
        return _BoundSelfConditioningCell(
            self,
            method=method,
            concept_id=concept_id,
            concept=concept,
            factor=factor,
            examples=examples,
        )

    def _cleanup(self, *, primary: BaseException | None) -> None:
        failures: list[BaseException] = []
        if self._active_cell is not None:
            try:
                self._active_cell.close()
            except BaseException as exc:
                failures.append(exc)
        self._active_cell = None
        try:
            self._close_judge_pool(abort=primary is not None)
        except BaseException as exc:
            failures.append(exc)
        self._target_model = None
        self._target_tokenizer = None
        self._extraction_model = None
        self._sae = None
        self._dense_diffmeans.clear()
        self._nl_target_scores.clear()
        gcg_directions = getattr(self, "_gcg_activation_directions", None)
        if gcg_directions is not None:
            gcg_directions.clear()
        self._originals.clear()
        try:
            gc.collect()
            if self._device.startswith("cuda") and t.cuda.is_available():
                t.cuda.empty_cache()
        except BaseException as exc:
            failures.append(exc)
        self._closed = True
        if failures:
            if primary is not None:
                for failure in failures:
                    primary.add_note(
                        f"SelfConditioning cleanup failed: {type(failure).__name__}: {failure}"
                    )
            else:
                first = failures[0]
                for failure in failures[1:]:
                    first.add_note(
                        f"additional SelfConditioning cleanup failure: "
                        f"{type(failure).__name__}: {failure}"
                    )
                raise first

    def close(self) -> None:
        if not self._closed:
            self._cleanup(primary=None)

    def __enter__(self) -> SelfConditioningMethodRuntime:
        if self._closed:
            raise AxBenchAdvError("SelfConditioning method runtime is closed")
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        del exc_type, traceback
        try:
            self.close()
        except BaseException as cleanup_exc:
            if exc_value is None:
                raise
            exc_value.add_note(
                f"SelfConditioning runtime cleanup failed: "
                f"{type(cleanup_exc).__name__}: {cleanup_exc}"
            )


__all__ = [
    "EvaluationRuntime",
    "SelfConditioningMethodRuntime",
    "render_self_conditioning_prompt",
]
