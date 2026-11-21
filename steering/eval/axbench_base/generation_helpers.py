"""Stage 4 generation helpers (Task 36).

Generation adapters used by Stage 4 and round-trip scoring:

  * :func:`_generate_with_nanochat_tokens` — builds a per-call
    :class:`~steering.eval.inference_engine.InferenceEngine` via
    :func:`~steering.eval.inference_engine.build_engine_for_features`,
    runs one generation, and returns a :class:`GenerationResult`.
  * :func:`_generate_samples_with_nanochat_tokens` — builds one engine for a
    shared prompt and maps its single-RNG batched rows to
    :class:`GenerationResult` values.

The helper is kept deliberately small: it performs no sampling logic
itself, it does not prepend BOS (the caller owns tokenization), and it
hands the full engine kwargs-dict through to ``generate_stream`` so
a future engine refactor is a single-file change.

Per-call engine build
---------------------
The helper calls :func:`~steering.eval.inference_engine.build_engine_for_features`
once per invocation. For decoder-only targets the engine is a thin
:class:`InferenceEngine` wrapper (no wrapping, microseconds). For
encoder-decoder targets the wrapper is rebuilt with the caller-supplied
``features`` (``encoder_input``) and ``context_provider``
so each generation call gets the correct feature conditioning without
needing a pre-built engine that carries stale feature state.

Stop-reason detection
---------------------
``InferenceEngine.generate_stream`` either hits a stop token (in which
case it stops yielding *before* emitting the stop token) or runs out of
budget. We distinguish the two cases by counting the yielded tokens:

  * ``len(generated) < max_new_tokens`` → stopped on a stop token
    ``stop_reason = "bos_token"``.
  * ``len(generated) == max_new_tokens`` → exhausted budget
    ``stop_reason = "max_tokens"``.

An empty generation (zero tokens) maps to ``stop_reason = "bos_token"``
because the engine must have hit a stop token on the very first sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch as t

from steering.conditional.encoder_decoder.eval_adapter import ContextProvider
from steering.eval.axbench_base.config import GenerationConfig
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.inference_engine import build_engine_for_features


@dataclass(frozen=True)
class GenerationResult:
    """A single generation result returned by the generation helper."""

    text: str
    token_count: int
    stop_reason: str  # "bos_token" | "max_tokens"


def _generate_with_nanochat_tokens(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    nanochat_tokens: list[int],
    sampling_seed: int,
    gen_cfg: GenerationConfig,
    encoder_input: dict[int, int] | None,
    context_provider: ContextProvider | None,
) -> GenerationResult:
    """Build a per-call engine and run one generation via ``generate_stream``.

    Parameters
    ----------
    model:
        The live model (decoder-only ``GPT`` or ``EncoderDecoderGPT``).
        :func:`~steering.eval.inference_engine.build_engine_for_features`
        wraps it appropriately for the target architecture.
    tokenizer:
        Any object with a ``decode(list[int]) -> str`` method. Not typed
        more strictly because the real nanochat tokenizer is a union of
        two concrete classes that share only this duck-typed surface here.
    device:
        Device string (``"cuda"`` / ``"cpu"``). Coerced to
        :class:`torch.device` inside ``build_engine_for_features``.
    nanochat_tokens:
        The fully constructed model-ready prompt. The caller owns BOS and
        prompting-prefix ordering and preflights the target checkpoint's
        sequence limit before dispatch; ``generate_stream`` independently
        guards its own runtime budget.
    sampling_seed:
        Forwarded verbatim to ``generate_stream``'s ``seed`` argument.
    gen_cfg:
        Supplies ``max_new_tokens``, ``temperature``, and
        ``top_k_sampling``.
    encoder_input:
        Mapping ``{feature_id: bucket_index}`` used as the explicit
        target-delta signal. ``None`` or an empty dict means no target
        delta; encoder-decoder calls still require ``context_provider``
        so prompt tokens receive real full-context features.
    context_provider:
        Provider for real prompt/context features for encoder-decoder models.
        Ignored for decoder-only models.

    Returns
    -------
    GenerationResult
        A frozen dataclass with the newly-generated text, token count, and
        stop reason. The prefix is NOT included in the returned text.
    """
    record_counter("axbench_base.generation.requests")
    record_counter("axbench_base.generation.prompt_tokens", len(nanochat_tokens))
    with profile_region(
        "axbench_base.generation.build_engine",
        prompt_tokens=len(nanochat_tokens),
        has_encoder_input=bool(encoder_input),
        has_context_provider=context_provider is not None,
    ):
        engine = build_engine_for_features(
            model,
            tokenizer,
            t.device(device),
            features=encoder_input,
            context_provider=context_provider,
        )
    record_counter("axbench_base.generation.engine_constructions")
    with profile_region(
        "stage4.generate_stream",
        prompt_tokens=len(nanochat_tokens),
        has_encoder_input=bool(encoder_input),
    ):
        generated: list[int] = list(
            engine.generate_stream(
                token_ids=nanochat_tokens,
                max_new_tokens=gen_cfg.max_new_tokens,
                temperature=gen_cfg.temperature,
                top_k=gen_cfg.top_k_sampling,
                seed=sampling_seed,
            )
        )
    token_count = len(generated)
    record_counter("axbench_base.generation.generated_tokens", token_count)
    text = tokenizer.decode(generated)
    stop_reason = "max_tokens" if token_count >= gen_cfg.max_new_tokens else "bos_token"
    return GenerationResult(
        text=text,
        token_count=token_count,
        stop_reason=stop_reason,
    )


def _generate_samples_with_nanochat_tokens(
    *,
    model: Any,
    tokenizer: Any,
    device: str,
    nanochat_tokens: list[int],
    sampling_seed: int,
    n_samples: int,
    gen_cfg: GenerationConfig,
    encoder_input: dict[int, int] | None,
    context_provider: ContextProvider | None,
) -> list[GenerationResult]:
    """Build one engine and generate a batch through one shared RNG."""
    record_counter("axbench_base.generation.requests", n_samples)
    record_counter("axbench_base.generation.prompt_tokens", len(nanochat_tokens))
    with profile_region(
        "axbench_base.generation.build_engine",
        prompt_tokens=len(nanochat_tokens),
        has_encoder_input=bool(encoder_input),
        has_context_provider=context_provider is not None,
    ):
        engine = build_engine_for_features(
            model,
            tokenizer,
            t.device(device),
            features=encoder_input,
            context_provider=context_provider,
        )
    record_counter("axbench_base.generation.engine_constructions")
    with profile_region(
        "stage4.generate_samples",
        prompt_tokens=len(nanochat_tokens),
        has_encoder_input=bool(encoder_input),
        n_samples=n_samples,
    ):
        generated_rows = engine.generate_samples(
            token_ids=nanochat_tokens,
            seed=sampling_seed,
            n_samples=n_samples,
            max_new_tokens=gen_cfg.max_new_tokens,
            temperature=gen_cfg.temperature,
            top_k=gen_cfg.top_k_sampling,
        )
    record_counter(
        "axbench_base.generation.generated_tokens",
        sum(len(row) for row in generated_rows),
    )
    return [
        GenerationResult(
            text=tokenizer.decode(row),
            token_count=len(row),
            stop_reason=(
                "max_tokens"
                if len(row) >= gen_cfg.max_new_tokens
                else "bos_token"
            ),
        )
        for row in generated_rows
    ]
