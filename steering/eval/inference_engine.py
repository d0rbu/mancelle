"""InferenceEngine — unified generation for GPT and EncoderDecoderGPT models.

Supports optional SAE feature conditioning, streaming single-sample decode,
and shared-prefill vectorized decode for multiple samples.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from dataclasses import dataclass, field
from typing import Any

import torch as t
import torch.nn.functional as F
from loguru import logger

from nanochat import flash_attention
from nanochat import sampling as nanochat_sampling
from nanochat.engine import KVCache
from nanochat.gpt import GPT
from nanochat.sampling import sample_next_token
from steering.conditional.encoder_decoder.eval_adapter import (
    ContextProvider,
    SelfCondRuntimeSnapshot,
    TargetAwareEncoderDecoderEvalWrapper,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.packed_encoder import (
    stack_packed_encoder_outputs,
)
from steering.conditional.encoder_decoder.self_conditioned import SelfCondConfig
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.chat_generation import (
    render_prompt_for_chat_completion,
    require_chat_render_support,
)
from steering.eval.target_conditioning import (
    ContextRoutingMode,
    MultiTargetSegments,
    SelfConditionedTarget,
    SingleTargetSegment,
    TargetConditioningMode,
)
from steering.eval.turn_context_contract import TurnContextContract
from steering.profiling.markers import timed_region
from steering.segmentation import SegmentMode


@dataclass(frozen=True)
class SelfConditionedPrefill:
    """One independent batch-one prefill ready for bounded batched decode.

    ``segment_mode`` is required, immutable provenance stamped by the producing
    wrapper: it is the only signal that lets ``generate_self_conditioned_batch``
    reject a turn-produced row, so a producer must never be able to omit it.
    """

    logits: t.Tensor
    kv_cache: KVCache
    runtime: SelfCondRuntimeSnapshot
    decoder: EncoderDecoderGPT = field(repr=False, compare=False)
    segment_size: int
    segment_mode: SegmentMode


def _sample_next_token_from_finite_logits(
    logits: t.Tensor,
    rng: t.Generator | None,
    temperature: float,
    top_k: int | None,
) -> t.Tensor:
    """Match nanochat sampling after a caller has checked a larger logits batch.

    Self-conditioned project rows require independent generators, so they cannot
    use one batched ``torch.multinomial`` call. The batch decode checks all active
    rows once, then calls this arithmetic-only helper in stable row order. Keep
    this branch structure identical to ``nanochat.sampling.sample_next_token``.
    """
    assert temperature >= 0.0, "temperature must be non-negative"
    if temperature == 0.0:
        return t.argmax(logits, dim=-1, keepdim=True)
    if top_k is not None and top_k > 0:
        k = min(top_k, logits.size(-1))
        values, indices = t.topk(logits, k, dim=-1)
        probabilities = F.softmax(values / temperature, dim=-1)
        choice = t.multinomial(probabilities, num_samples=1, generator=rng)
        return indices.gather(1, choice)
    probabilities = F.softmax(logits / temperature, dim=-1)
    return t.multinomial(probabilities, num_samples=1, generator=rng)


def _allocate_kv_cache(
    *,
    batch_size: int,
    num_heads: int,
    seq_len: int,
    head_dim: int,
    num_layers: int,
    device: t.device,
    dtype: t.dtype,
) -> KVCache:
    cache = KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device=device,
        dtype=dtype,
    )
    # Match flash_attn_with_kvcache's dispatch, including forced SDPA on FA3
    # hardware. SDPA reads positions on the host in every decoder layer; keeping
    # this authoritative counter on CPU avoids those CUDA synchronizations.
    uses_fa3 = (
        flash_attention.HAS_FA3
        and flash_attention._override_impl != flash_attention.FlashAttnImpl.SDPA
    )
    if device.type == "cuda" and not uses_fa3:
        cache.cache_seqlens = t.zeros(batch_size, dtype=t.int32, device="cpu")
    return cache


def stack_prefilled_kv_caches(caches: tuple[KVCache, ...]) -> KVCache:
    """Stack uniform-position batch-one caches without mutating their rows."""
    if not caches:
        raise ValueError("KV-cache stacking requires at least one cache")
    first = caches[0]
    position = int(first.get_pos())
    if first.k_cache.device != first.v_cache.device:
        raise ValueError(
            "KV-cache stacking requires K and V on one device; "
            f"row 0 has K on {first.k_cache.device} and V on {first.v_cache.device}"
        )
    if first.k_cache.dtype != first.v_cache.dtype:
        raise TypeError(
            "KV-cache stacking requires one K/V dtype; "
            f"row 0 has K={first.k_cache.dtype}, V={first.v_cache.dtype}"
        )
    if not (0 <= position <= first.max_seq_len):
        raise ValueError(
            "KV-cache stacking requires an in-capacity decode position; "
            f"row 0 has position={position}, capacity={first.max_seq_len}"
        )
    expected = (
        first.n_layers,
        first.n_heads,
        first.head_dim,
        first.max_seq_len,
        first.k_cache.device,
        first.k_cache.dtype,
        first.v_cache.device,
        first.v_cache.dtype,
    )
    for index, cache in enumerate(caches):
        if cache.batch_size != 1:
            raise ValueError(
                "KV-cache stacking requires batch-one sources; "
                f"row {index} has batch_size={cache.batch_size}"
            )
        observed = (
            cache.n_layers,
            cache.n_heads,
            cache.head_dim,
            cache.max_seq_len,
            cache.k_cache.device,
            cache.k_cache.dtype,
            cache.v_cache.device,
            cache.v_cache.dtype,
        )
        if observed != expected:
            raise ValueError(
                "KV-cache stacking requires identical shapes, capacities, devices, "
                f"and dtypes; row 0={expected}, row {index}={observed}"
            )
        row_position = int(cache.get_pos())
        if row_position != position:
            raise ValueError(
                "KV-cache stacking requires one decode position; "
                f"row 0 is at {position}, row {index} is at {row_position}"
            )

    stacked = _allocate_kv_cache(
        batch_size=len(caches),
        num_heads=first.n_heads,
        seq_len=first.max_seq_len,
        head_dim=first.head_dim,
        num_layers=first.n_layers,
        device=first.k_cache.device,
        dtype=first.k_cache.dtype,
    )
    for row, cache in enumerate(caches):
        stacked.k_cache[:, row : row + 1, :position] = cache.k_cache[:, :, :position]
        stacked.v_cache[:, row : row + 1, :position] = cache.v_cache[:, :, :position]
    stacked.cache_seqlens.fill_(position)
    return stacked


class InferenceEngine:
    """Unified generation interface for GPT and wrapped encoder-decoder models.

    Handles KV-cache management and autoregressive token generation, including
    one-prompt multi-sample generation with a shared prefill, one batch RNG,
    and compacting batched decode.
    Encoder-decoder conditioning is handled externally by wrapping the model
    with ``TargetAwareEncoderDecoderEvalWrapper`` (see ``build_engine_for_features``).

    Parameters
    ----------
    model:
        A ``GPT`` instance or a ``TargetAwareEncoderDecoderEvalWrapper`` in eval mode.
    tokenizer:
        Tokenizer with ``encode``, ``decode``, and ``get_bos_token_id`` methods.
    device:
        Device the model lives on.
    """

    def __init__(
        self,
        model,
        tokenizer: Any,
        device: t.device,
    ) -> None:
        self._model: Any = model
        self._tokenizer: Any = tokenizer
        self._device = device
        # Decoder config for KV cache sizing. Works for both raw GPT
        # (model.config) and wrapped encoder-decoder (wrapper.config = model.dec_config).
        self._dec_config: Any = model.config
        self._bos_id = tokenizer.get_bos_token_id()
        self._supports_chat_render = (
            hasattr(tokenizer, "render_for_completion")
            and hasattr(tokenizer, "encode_special")
        )

    # ------------------------------------------------------------------
    # Validation helpers
    # ------------------------------------------------------------------

    def _validate_prompt_length(
        self, n_tokens: int, max_new_tokens: int = 0
    ) -> None:
        """Raise ``ValueError`` if prompt + new tokens exceeds the context window.

        The KV cache is allocated to ``len(token_ids) + max_new_tokens``; the
        decoder's RoPE bounds-check uses ``T0 + T <= cos.size(1)``. So a
        prompt of exactly ``sequence_len`` is OK only when
        ``max_new_tokens == 0``. Previously the helper rejected only
        ``n_tokens > sequence_len``, allowing
        ``n_tokens == sequence_len`` with ``max_new_tokens > 0`` to slip
        through and assert deep inside ``gpt.py`` (see map SB-13-P1-01).

        ``max_new_tokens`` defaults to 0 for backwards-compatible
        prompt-only validation; entry points that allocate a decode budget
        pass it explicitly.
        """
        max_len = self._dec_config.sequence_len
        if n_tokens + max_new_tokens > max_len:
            raise ValueError(
                f"Prompt length ({n_tokens}) + max_new_tokens "
                f"({max_new_tokens}) exceeds context window ({max_len})"
            )

    def _require_chat_render_support(self) -> None:
        """Fail fast (and loudly) when a caller asks for ``render_as_chat=True``
        against a tokenizer that doesn't implement the nanochat chat template.

        Only ``RustBPETokenizer`` defines ``render_for_completion`` today;
        ``TransformersTokenizer`` and ``HuggingFaceTokenizer`` do not. Without
        this guard the caller would get a cryptic ``AttributeError`` from deep
        inside ``generate``; instead we raise a ``ValueError`` that names the
        tokenizer class and the remedy.
        """
        require_chat_render_support(self._tokenizer)

    def _build_chat_prompt(
        self, prompt: str
    ) -> tuple[list[int], set[int]]:
        """Render *prompt* as an SFT-style chat completion prefix.

        Returns ``(token_ids, stop_token_ids)`` where ``token_ids`` primes
        the model for an assistant response (ends with
        ``<assistant_start>``) and ``stop_token_ids`` covers both the BOS
        sentinel and the ``<assistant_end>`` terminator that SFT taught the
        model to emit.

        ``_require_chat_render_support()`` MUST be called by the entrypoint
        before invoking this method; this helper assumes the capability.

        Raises ``ValueError`` if ``<assistant_end>`` cannot be resolved to
        a concrete token id (e.g. ``HuggingFaceTokenizer.encode_special``
        returns ``None`` for missing tokens, and ``TransformersTokenizer``
        raises ``ParseError``). Without this guard a ``None`` would leak
        into ``stop_token_ids`` and silently disable the assistant-end
        stop — sampled int ids never equal ``None``, so generation would
        run to ``max_new_tokens`` every time.
        """
        rendered = render_prompt_for_chat_completion(self._tokenizer, prompt)
        stop_token_ids = {self._bos_id, rendered.assistant_end_id}
        return rendered.token_ids, stop_token_ids

    # ------------------------------------------------------------------
    # KV cache
    # ------------------------------------------------------------------

    def _make_kv_cache(self, batch_size: int, seq_len: int) -> KVCache:
        """Allocate a fresh KV cache sized for the decoder."""
        cfg = self._dec_config
        dtype = t.bfloat16 if self._device.type == "cuda" else t.float32
        return _allocate_kv_cache(
            batch_size=batch_size,
            num_heads=cfg.n_kv_head,
            seq_len=seq_len,
            head_dim=cfg.n_embd // cfg.n_head,
            num_layers=getattr(self._model, "num_cache_layers", cfg.n_layer),
            device=self._device,
            dtype=dtype,
        )

    def _compact_kv_cache(
        self,
        kv_cache: KVCache,
        keep_indices: t.Tensor,
    ) -> KVCache:
        """Copy surviving rows into a fresh uniformly positioned KV cache."""
        position = int(kv_cache.get_pos())
        survivors = int(keep_indices.numel())
        removed = int(kv_cache.batch_size - survivors)
        with profile_region(
            "axbench_base.generation.compact_decode_batch",
            survivors=survivors,
            removed=removed,
            position=position,
        ):
            compacted = self._make_kv_cache(
                batch_size=survivors,
                seq_len=kv_cache.max_seq_len,
            )
            compacted.k_cache[:, :, :position] = kv_cache.k_cache[
                :, keep_indices, :position
            ]
            compacted.v_cache[:, :, :position] = kv_cache.v_cache[
                :, keep_indices, :position
            ]
            compacted.cache_seqlens.fill_(position)
        record_counter("axbench_base.generation.batch_compactions")
        return compacted

    # ------------------------------------------------------------------
    # Streaming generation
    # ------------------------------------------------------------------

    def prepare_self_conditioned_prefill(
        self,
        token_ids: list[int],
        *,
        max_new_tokens: int = 0,
    ) -> SelfConditionedPrefill:
        """Use no-grad only for an explicitly traced RLVR prefill."""

        trace_callback = getattr(
            self._model,
            "_self_cond_prompt_trace_callback",
            None,
        )
        with t.inference_mode(trace_callback is None), t.no_grad():
            return self._prepare_self_conditioned_prefill_impl(
                token_ids,
                max_new_tokens=max_new_tokens,
            )

    def _prepare_self_conditioned_prefill_impl(
        self,
        token_ids: list[int],
        *,
        max_new_tokens: int = 0,
    ) -> SelfConditionedPrefill:
        """Return marker logits and KV state without sampling or feeding an answer.

        Zero reserved decode positions supports categorical scoring even when
        the exact completion prefix fills the model context.
        """
        if not isinstance(self._model, TargetAwareEncoderDecoderEvalWrapper) or not bool(
            getattr(self._model, "is_self_conditioned", False)
        ):
            raise ValueError(
                "prepare_self_conditioned_prefill requires a self-conditioned "
                "TargetAwareEncoderDecoderEvalWrapper"
            )
        if not token_ids:
            raise ValueError("self-conditioned prefill token_ids must be nonempty")
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be nonnegative")
        self._validate_prompt_length(len(token_ids), max_new_tokens)

        total_len = len(token_ids) + max_new_tokens
        kv_cache = self._make_kv_cache(batch_size=1, seq_len=total_len)
        record_counter("axbench_base.generation.prefill_tokens", len(token_ids))
        idx = t.tensor([token_ids], dtype=t.long, device=self._device)
        with profile_region(
            "axbench_base.generation.prefill",
            prompt_tokens=len(token_ids),
            max_new_tokens=max_new_tokens,
            target_aware=True,
            self_conditioned=True,
            physical_batch_size=1,
        ):
            with self._model.conditioning_context(context_token_ids=[token_ids]):
                with timed_region("generation/prefill_model_forward", sync_cuda=False):
                    logits = self._model(idx, kv_cache=kv_cache)
        return SelfConditionedPrefill(
            logits=logits[:, -1, :],
            kv_cache=kv_cache,
            runtime=self._model.self_cond_runtime_snapshot(),
            decoder=self._model.model,
            segment_size=self._model.self_cond_segment_size(),
            segment_mode=self._model.self_cond_segment_mode,
        )

    @t.inference_mode()
    def generate_self_conditioned_batch(
        self,
        prefills: Sequence[SelfConditionedPrefill],
        *,
        seeds: Sequence[int],
        max_new_tokens: int,
        temperature: float,
        top_k: int | None,
        stop_token_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Decode independent scalar prefills together before their first boundary.

        Each row keeps its own sampling generator and stop state. Finished rows
        remain as inert physical filler rows so surviving rows never change
        their batch index or residual addend. The underlying model still emits
        feature-prediction logits on every forwarded token, matching the scalar
        non-boundary arithmetic path; those rows are intentionally not retained
        because this method rejects any budget that could consume them at a
        self-conditioning boundary.

        This optimization is fixed-token-only. Turn-delimited generation can
        refresh more than once within a completion and is rejected before any
        prefill validation, state stacking, or sampling work.
        """
        if not isinstance(self._model, TargetAwareEncoderDecoderEvalWrapper) or not bool(
            getattr(self._model, "is_self_conditioned", False)
        ):
            raise ValueError(
                "generate_self_conditioned_batch requires a self-conditioned "
                "TargetAwareEncoderDecoderEvalWrapper"
            )
        if self._model.is_turn_self_conditioned:
            raise ValueError(
                "generate_self_conditioned_batch does not support turn-delimited "
                "self-conditioning; use independent scalar generate_stream calls"
            )
        rows = tuple(prefills)
        row_seeds = tuple(seeds)
        if not rows:
            raise ValueError("self-conditioned batch requires at least one prefill")
        for index, prefill in enumerate(rows):
            if prefill.segment_mode is not SegmentMode.fixed_tokens:
                raise ValueError(
                    "self-conditioned batched decode requires fixed_tokens prefill "
                    f"provenance; row {index} has segment_mode={prefill.segment_mode!r}"
                )
        if len(row_seeds) != len(rows):
            raise ValueError(
                "self-conditioned batch requires one seed per prefill; "
                f"got {len(row_seeds)} seeds for {len(rows)} rows"
            )
        if isinstance(max_new_tokens, bool) or not isinstance(max_new_tokens, int):
            raise TypeError("max_new_tokens must be an integer")
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        segment_size = self._model.self_cond_segment_size()
        if max_new_tokens > segment_size:
            raise ValueError(
                "self-conditioned batched decode cannot cross a conditioning boundary: "
                f"max_new_tokens={max_new_tokens}, segment_size={segment_size}"
            )
        if temperature < 0.0:
            raise ValueError("temperature must be nonnegative")
        if temperature > 0.0 and any(
            isinstance(seed, bool) or not isinstance(seed, int) for seed in row_seeds
        ):
            raise ValueError(
                "self-conditioned batched decode requires one non-bool integer "
                "seed per row when temperature is positive"
            )
        if stop_token_ids is None:
            stop_token_ids = {self._bos_id}

        for index, prefill in enumerate(rows):
            if prefill.decoder is not self._model.model:
                raise ValueError(
                    "self-conditioned prefills must come from the active decoder; "
                    f"row {index} has a different decoder instance"
                )
            if prefill.segment_size != segment_size:
                raise ValueError(
                    "self-conditioned prefills must share the active segment window; "
                    f"row {index} has segment_size={prefill.segment_size}, "
                    f"active={segment_size}"
                )
            if prefill.logits.ndim != 2 or prefill.logits.size(0) != 1:
                raise ValueError(
                    "self-conditioned prefill logits must have shape (1, V); "
                    f"row {index} has {tuple(prefill.logits.shape)}"
                )
            if prefill.logits.device != self._device:
                raise ValueError(
                    "self-conditioned prefill logits must be on the engine device; "
                    f"row {index} is on {prefill.logits.device}, engine is on {self._device}"
                )
            if index > 0 and (
                prefill.logits.shape != rows[0].logits.shape
                or prefill.logits.dtype != rows[0].logits.dtype
            ):
                raise ValueError(
                    "self-conditioned prefill logits must share one vocabulary and dtype; "
                    f"row 0 has shape={tuple(rows[0].logits.shape)}, "
                    f"dtype={rows[0].logits.dtype}; row {index} has "
                    f"shape={tuple(prefill.logits.shape)}, dtype={prefill.logits.dtype}"
                )
            if prefill.runtime.tokens_in_seg != 0 or prefill.runtime.fp_buffer:
                raise ValueError(
                    "self-conditioned batched decode requires boundary prefills with "
                    f"an empty FP buffer; row {index} has tokens_in_seg="
                    f"{prefill.runtime.tokens_in_seg} and buffer size "
                    f"{len(prefill.runtime.fp_buffer)}"
                )
            target_slot = prefill.runtime.state.target_slot_idx
            segment_count = prefill.runtime.state.encoder_out.layout.n_segments
            if type(target_slot) is not int or not (0 <= target_slot < segment_count):
                raise ValueError(
                    "self-conditioned prefill target slot must identify a packed segment; "
                    f"row {index} has target_slot_idx={target_slot!r}, "
                    f"n_segments={segment_count}"
                )
            cache_position = int(prefill.kv_cache.get_pos())
            if prefill.kv_cache.max_seq_len - cache_position < max_new_tokens:
                raise ValueError(
                    "self-conditioned prefill cache has insufficient decode capacity; "
                    f"row {index} has position={cache_position}, "
                    f"capacity={prefill.kv_cache.max_seq_len}, "
                    f"max_new_tokens={max_new_tokens}"
                )

        logits = t.cat([prefill.logits for prefill in rows], dim=0)
        encoder_out = stack_packed_encoder_outputs(
            tuple(prefill.runtime.state.encoder_out for prefill in rows)
        )
        kv_cache = stack_prefilled_kv_caches(tuple(prefill.kv_cache for prefill in rows))
        seg_map = t.tensor(
            [[prefill.runtime.state.target_slot_idx] for prefill in rows],
            dtype=t.long,
            device=self._device,
        )
        generators: list[t.Generator | None] = []
        for seed in row_seeds:
            if temperature == 0.0:
                generators.append(None)
            else:
                generator = t.Generator(device=self._device)
                generator.manual_seed(seed)
                generators.append(generator)

        generated: list[list[int]] = [[] for _ in rows]
        finished = [False] * len(rows)
        filler = t.tensor([[self._bos_id]], dtype=t.long, device=self._device)
        logical_decode_steps = 0
        batched_iterations = 0
        with profile_region(
            "axbench_base.generation.decode",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            target_aware=True,
            self_conditioned=True,
            batch_size=len(rows),
            independent_rngs=True,
            fixed_physical_rows=True,
        ):
            try:
                for token_index in range(max_new_tokens):
                    batched_iterations += 1
                    active_rows = [
                        row_index
                        for row_index, row_finished in enumerate(finished)
                        if not row_finished
                    ]
                    active_logits = logits if len(active_rows) == len(rows) else logits[active_rows]
                    with timed_region(
                        "generation/sample_next_token_batch",
                        sync_cuda=False,
                    ):
                        # ``nan`` bypasses the comparison-based input check above.
                        # Replay the first scalar attempt so the canonical
                        # assertion still precedes the finite guard and the
                        # legacy failed-attempt timing marker is retained.
                        if not temperature >= 0.0:
                            with timed_region(
                                "generation/sample_next_token",
                                sync_cuda=False,
                            ):
                                first_active_row = active_rows[0]
                                sample_next_token(
                                    logits[first_active_row : first_active_row + 1],
                                    generators[first_active_row],
                                    temperature,
                                    top_k,
                                )
                                raise AssertionError(
                                    "canonical sampler accepted a non-finite temperature"
                                )
                        batch_guard_error: FloatingPointError | None = None
                        try:
                            nanochat_sampling._assert_finite_logits(active_logits)
                        except FloatingPointError as batch_error:
                            batch_guard_error = batch_error
                        if batch_guard_error is not None:
                            # Preserve the legacy exceptional path exactly: the
                            # scalar sampler advances earlier finite rows, then
                            # raises with the first bad row's original (1, V)
                            # diagnostic. This branch is irrelevant to the hot
                            # finite path but keeps profiler counters stable.
                            for row_index in active_rows:
                                with timed_region(
                                    "generation/sample_next_token",
                                    sync_cuda=False,
                                ):
                                    sample_next_token(
                                        logits[row_index : row_index + 1],
                                        generators[row_index],
                                        temperature,
                                        top_k,
                                    )
                                logical_decode_steps += 1
                            raise AssertionError(
                                "batched finite guard failed without a bad active row"
                            ) from batch_guard_error
                        sampled_active: list[t.Tensor] = []
                        for active_index, row_index in enumerate(active_rows):
                            with timed_region(
                                "generation/sample_next_token",
                                sync_cuda=False,
                            ):
                                sampled_active.append(
                                    _sample_next_token_from_finite_logits(
                                        active_logits[active_index : active_index + 1],
                                        generators[row_index],
                                        temperature,
                                        top_k,
                                    )
                                )
                            logical_decode_steps += 1
                        active_tokens = t.cat(sampled_active, dim=0)[:, 0].tolist()

                    sampled: list[t.Tensor] = []
                    active_index = 0
                    for row_index, row_finished in enumerate(finished):
                        if row_finished:
                            sampled.append(filler)
                            continue
                        next_id = sampled_active[active_index]
                        token = int(active_tokens[active_index])
                        active_index += 1
                        if token in stop_token_ids:
                            finished[row_index] = True
                        else:
                            generated[row_index].append(token)
                        sampled.append(next_id)

                    if all(finished):
                        break
                    if token_index + 1 < max_new_tokens:
                        emitted_tokens = t.cat(sampled, dim=0)
                        with timed_region(
                            "generation/decode_model_forward",
                            sync_cuda=False,
                        ):
                            logits, _fp_logits = self._model.model(
                                emitted_tokens,
                                None,
                                seg_map=seg_map,
                                kv_cache=kv_cache,
                                encoder_out=encoder_out,
                                encoder_strict=True,
                                return_fp_logits=True,
                            )
                        logits = logits[:, -1, :]
            finally:
                record_counter(
                    "axbench_base.generation.decode_steps",
                    logical_decode_steps,
                )
                record_counter(
                    "axbench_base.generation.batched_decode_iterations",
                    batched_iterations,
                )
        return generated

    @t.inference_mode()
    def generate_stream(
        self,
        token_ids: list[int],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = None,
        seed: int | None = None,
        stop_token_ids: set[int] | None = None,
    ) -> Iterator[int]:
        """Yield generated token IDs one at a time."""
        if stop_token_ids is None:
            stop_token_ids = {self._bos_id}
        if isinstance(self._model, TargetAwareEncoderDecoderEvalWrapper):
            stop_token_ids = stop_token_ids | self._model.generation_stop_token_ids

        self._validate_prompt_length(len(token_ids), max_new_tokens)
        if max_new_tokens == 0:
            return

        if bool(getattr(self._model, "is_self_conditioned", False)):
            prefill = self.prepare_self_conditioned_prefill(
                token_ids,
                max_new_tokens=max_new_tokens,
            )
            kv_cache = prefill.kv_cache
            logits = prefill.logits
        else:
            total_len = len(token_ids) + max_new_tokens
            kv_cache = self._make_kv_cache(batch_size=1, seq_len=total_len)
            record_counter("axbench_base.generation.prefill_tokens", len(token_ids))

            idx = t.tensor([token_ids], dtype=t.long, device=self._device)
            prefill_ctx: Any
            if getattr(self._model, "is_target_aware_encdec", False):
                prefill_ctx = self._model.conditioning_context(
                    context_token_ids=[token_ids],
                )
            else:
                prefill_ctx = nullcontext()
            with profile_region(
                "axbench_base.generation.prefill",
                prompt_tokens=len(token_ids),
                max_new_tokens=max_new_tokens,
                target_aware=bool(getattr(self._model, "is_target_aware_encdec", False)),
            ):
                with prefill_ctx:
                    with timed_region("generation/prefill_model_forward", sync_cuda=False):
                        logits = self._model(idx, kv_cache=kv_cache)
            logits = logits[:, -1, :]

        rng: t.Generator | None = None
        if temperature > 0.0:
            if seed is None:
                raise ValueError("seed is required when temperature > 0")
            rng = t.Generator(device=self._device)
            rng.manual_seed(seed)

        decode_steps = 0
        with profile_region(
            "axbench_base.generation.decode",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            target_aware=bool(getattr(self._model, "is_target_aware_encdec", False)),
        ):
            try:
                for token_index in range(max_new_tokens):
                    decode_steps += 1
                    with timed_region("generation/sample_next_token", sync_cuda=False):
                        next_id = sample_next_token(logits, rng, temperature, top_k)
                    tok = int(next_id.item())
                    if tok in stop_token_ids:
                        return
                    yield tok
                    if token_index + 1 == max_new_tokens:
                        return
                    idx = next_id
                    with timed_region("generation/decode_model_forward", sync_cuda=False):
                        logits = self._model(idx, kv_cache=kv_cache)
                    logits = logits[:, -1, :]
            finally:
                record_counter("axbench_base.generation.decode_steps", decode_steps)

    @t.inference_mode()
    def generate_samples(
        self,
        token_ids: list[int],
        *,
        seed: int | None,
        n_samples: int,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = None,
        stop_token_ids: set[int] | None = None,
    ) -> list[list[int]]:
        """Generate continuations of one shared prompt with one batch RNG.

        The prompt is prefilled once at batch size one. Its KV rows are then
        broadcast into ``n_samples`` rows. Every iteration samples all active
        rows in one call, and finished rows are removed from both the active
        logits batch and KV cache before the next decode forward.

        A quiet iteration (no row stops) performs one loop-local stop-detection
        host sync at ``bool(finished_now.any())``. This is deliberately not an
        end-to-end sync-count claim. The sampler's finite-logit ``.all`` also
        reads a CUDA scalar. SDPA caches keep positions on CPU, so cached-forward
        ``get_pos`` calls, batch-uniformity checks, per-layer position reads, and
        the position read per compaction do not synchronize CUDA. FA3 caches
        retain CUDA position counters, so their ``get_pos`` calls still require
        a device-to-host scalar read.

        Self-conditioned target-aware wrappers support exactly one sample because
        their dynamically growing encoder state is intentionally batch-size-one.
        """
        if not isinstance(n_samples, int) or isinstance(n_samples, bool) or n_samples < 1:
            raise ValueError(f"n_samples must be an int >= 1, got {n_samples!r}")
        if getattr(self._model, "is_self_conditioned", False) and n_samples > 1:
            raise ValueError(
                "generate_samples does not support self-conditioned mode with "
                f"N={n_samples}; pass n_samples=1."
            )
        if temperature > 0.0 and (
            not isinstance(seed, int) or isinstance(seed, bool)
        ):
            raise ValueError(
                "seed must be a non-None, non-bool int when temperature > 0.0; "
                f"got seed={seed!r}, temperature={temperature}"
            )

        self._validate_prompt_length(len(token_ids), max_new_tokens)
        if stop_token_ids is None:
            stop_token_ids = {self._bos_id}
        if isinstance(self._model, TargetAwareEncoderDecoderEvalWrapper):
            stop_token_ids = stop_token_ids | self._model.generation_stop_token_ids

        total_len = len(token_ids) + max_new_tokens
        prefill_cache = self._make_kv_cache(batch_size=1, seq_len=total_len)
        record_counter("axbench_base.generation.prefill_tokens", len(token_ids))

        idx = t.tensor([token_ids], dtype=t.long, device=self._device)
        prefill_ctx: Any
        if getattr(self._model, "is_target_aware_encdec", False):
            prefill_ctx = self._model.conditioning_context(
                context_token_ids=[token_ids],
            )
        else:
            prefill_ctx = nullcontext()
        with profile_region(
            "axbench_base.generation.prefill",
            prompt_tokens=len(token_ids),
            max_new_tokens=max_new_tokens,
            target_aware=bool(getattr(self._model, "is_target_aware_encdec", False)),
            physical_batch_size=1,
            n_samples=n_samples,
        ):
            with prefill_ctx:
                with timed_region("generation/prefill_model_forward", sync_cuda=False):
                    logits = self._model(idx, kv_cache=prefill_cache)
        logits = logits[:, -1, :].expand(n_samples, -1)

        with profile_region(
            "axbench_base.generation.expand_prefill_cache",
            prefill_tokens=len(token_ids),
            n_samples=n_samples,
        ):
            decode_cache = self._make_kv_cache(
                batch_size=n_samples,
                seq_len=total_len,
            )
            decode_cache.prefill(prefill_cache)
        del prefill_cache

        rng: t.Generator | None = None
        if temperature > 0.0:
            assert isinstance(seed, int) and not isinstance(seed, bool)
            rng = t.Generator(device=self._device)
            rng.manual_seed(seed)

        rows_with_lengths = t.zeros(
            (n_samples, max_new_tokens + 1),
            dtype=t.long,
            device=self._device,
        )
        rows_with_lengths[:, 0] = max_new_tokens
        active_original_rows = t.arange(
            n_samples,
            dtype=t.long,
            device=self._device,
        )
        sorted_stop_token_ids = sorted(stop_token_ids)
        if sorted_stop_token_ids:
            stop_ids = t.tensor(
                sorted_stop_token_ids,
                dtype=t.long,
                device=self._device,
            )
        else:
            stop_ids = t.empty(0, dtype=t.long, device=self._device)
        active_count = n_samples
        decode_steps = 0
        batched_iterations = 0
        with profile_region(
            "axbench_base.generation.decode",
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            target_aware=bool(getattr(self._model, "is_target_aware_encdec", False)),
            batch_size=n_samples,
        ):
            try:
                for iteration in range(max_new_tokens):
                    batched_iterations += 1
                    decode_steps += active_count
                    with timed_region("generation/sample_next_token", sync_cuda=False):
                        next_ids = sample_next_token(logits, rng, temperature, top_k)
                    with timed_region("generation/stop_detection", sync_cuda=False):
                        finished_now = t.isin(next_ids[:, 0], stop_ids)
                        any_finished = bool(finished_now.any())

                    if not any_finished:
                        rows_with_lengths[
                            active_original_rows, iteration + 1
                        ] = next_ids[:, 0]
                        # No cache state is consumed after the final budgeted draw.
                        if iteration + 1 == max_new_tokens:
                            break
                        # Keep the sampled tensor directly: boolean indexing would
                        # make an avoidable copy on every quiet decode iteration.
                        idx = next_ids
                        with timed_region(
                            "generation/decode_model_forward",
                            sync_cuda=False,
                        ):
                            logits = self._model(idx, kv_cache=decode_cache)[:, -1, :]
                        continue

                    keep_mask = ~finished_now
                    surviving_original_rows = active_original_rows[keep_mask]
                    rows_with_lengths[
                        surviving_original_rows, iteration + 1
                    ] = next_ids[keep_mask, 0]
                    rows_with_lengths[active_original_rows[finished_now], 0] = iteration

                    # No cache state is consumed after the final budgeted draw.
                    # Avoid copying the full prefix into a smaller cache only to
                    # break immediately afterward.
                    if iteration + 1 == max_new_tokens:
                        break
                    if bool(finished_now.all()):
                        break

                    keep_indices = keep_mask.nonzero(as_tuple=True)[0]
                    with timed_region("generation/compact_kv_cache", sync_cuda=False):
                        decode_cache = self._compact_kv_cache(
                            decode_cache,
                            keep_indices,
                        )
                    active_original_rows = surviving_original_rows
                    active_count = keep_indices.numel()
                    idx = next_ids[keep_mask]
                    with timed_region("generation/decode_model_forward", sync_cuda=False):
                        logits = self._model(idx, kv_cache=decode_cache)[:, -1, :]
            finally:
                record_counter("axbench_base.generation.decode_steps", decode_steps)
                record_counter(
                    "axbench_base.generation.batched_decode_iterations",
                    batched_iterations,
                )

        # The zero-budget case naturally has shape (N, 1). One conversion moves
        # the initialized length/token rectangle to Python without a transient cat.
        converted_rows = rows_with_lengths.tolist()
        return [row[1 : 1 + row[0]] for row in converted_rows]

    # ------------------------------------------------------------------
    # Generation
    # ------------------------------------------------------------------

    @t.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = None,
        seed: int | None = None,
        render_as_chat: bool = False,
    ) -> str:
        if render_as_chat:
            self._require_chat_render_support()
            token_ids, stop_token_ids = self._build_chat_prompt(prompt)
        else:
            token_ids = self._tokenizer.encode(prompt, prepend=self._bos_id)
            stop_token_ids = None
        generated = list(
            self.generate_stream(
                token_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed,
                stop_token_ids=stop_token_ids,
            )
        )
        return self._tokenizer.decode(generated)

    @t.inference_mode()
    def generate_batch(
        self,
        prompts: list[str],
        max_new_tokens: int = 256,
        temperature: float = 0.7,
        top_k: int | None = None,
        seed: int | None = None,
        render_as_chat: bool = False,
    ) -> list[str]:
        """Generate text for multiple prompts.

        For ``temperature != 0`` OR a wrapped encoder-decoder model, generation
        runs sequentially with per-prompt seeds (``seed + i``). For greedy
        (``temperature == 0``) on a raw decoder-only model, prompts are
        left-padded with BOS to equal length and processed as a true batch.

        The wrapper (TargetAwareEncoderDecoderEvalWrapper) enforces B==1 prefill
        with kv_cache (eval_adapter.py:_build_conditioning_impl), so wrapped encdec callers must
        go through the sequential path.
        """
        if len(prompts) == 0:
            logger.debug("generate_batch called with empty prompts list")
            return []

        if render_as_chat:
            self._require_chat_render_support()

        # Force sequential when wrapped or when temperature implies sampling.
        if temperature != 0.0 or getattr(self._model, "is_target_aware_encdec", False):
            return [
                self.generate(
                    p,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_k=top_k,
                    seed=seed + i if seed is not None else None,
                    render_as_chat=render_as_chat,
                )
                for i, p in enumerate(prompts)
            ]

        # Greedy batched (decoder-only only).
        # Greedy is mathematically argmax-only; a caller-supplied ``top_k``
        # would silently be dropped (see map SB-13-P0-03). Reject it
        # explicitly so the contract drift is loud.
        if top_k is not None:
            raise ValueError(
                "generate_batch greedy path (temperature=0.0, decoder-only) "
                f"does not honour top_k (got top_k={top_k}). Drop top_k for "
                "greedy generation, or pass temperature>0 to take the "
                "sequential sampling path that does honour top_k."
            )
        # Tokenize all prompts. In chat mode every prompt is rendered through
        # the SFT template; in raw mode prompts are encoded with a BOS prepend.
        if render_as_chat:
            all_ids_and_stops = [self._build_chat_prompt(p) for p in prompts]
            all_ids = [ids for ids, _ in all_ids_and_stops]
            # Every prompt rendered by the same tokenizer must yield the same
            # stop set; UNIONing across prompts lets a future divergence go
            # silent (see map SB-13-P0-04). Assert uniformity defensively.
            per_prompt_stops = [stops for _, stops in all_ids_and_stops]
            first_stops = per_prompt_stops[0]
            for i, stops in enumerate(per_prompt_stops):
                if stops != first_stops:
                    raise RuntimeError(
                        "generate_batch render_as_chat: stop_token_ids must "
                        "be identical across all prompts (single tokenizer "
                        "per engine), but prompt[0] yielded "
                        f"{sorted(first_stops)} and prompt[{i}] yielded "
                        f"{sorted(stops)}. Investigate the tokenizer or the "
                        "chat-template renderer."
                    )
            stop_token_ids: set[int] = set(first_stops)
        else:
            all_ids = [
                self._tokenizer.encode(p, prepend=self._bos_id) for p in prompts
            ]
            stop_token_ids = {self._bos_id}
        for ids in all_ids:
            self._validate_prompt_length(len(ids), max_new_tokens)

        max_prompt_len = max(len(ids) for ids in all_ids)
        batch_size = len(prompts)

        # Left-pad with BOS to equal length.
        padded = []
        for ids in all_ids:
            pad_len = max_prompt_len - len(ids)
            padded.append([self._bos_id] * pad_len + ids)

        total_len = max_prompt_len + max_new_tokens
        kv_cache = self._make_kv_cache(batch_size, total_len)

        idx = t.tensor(padded, dtype=t.long, device=self._device)
        logits = self._model(idx, kv_cache=kv_cache)
        logits = logits[:, -1, :]  # (B, V)

        generated: list[list[int]] = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_new_tokens):
            next_ids = sample_next_token(logits, None, 0.0, None)  # (B, 1)
            tokens = next_ids.squeeze(-1).tolist()

            for i, tok in enumerate(tokens):
                if finished[i]:
                    continue
                if tok in stop_token_ids:
                    finished[i] = True
                else:
                    generated[i].append(tok)

            if all(finished):
                break

            idx = next_ids
            logits = self._model(idx, kv_cache=kv_cache)
            logits = logits[:, -1, :]

        return [self._tokenizer.decode(g) for g in generated]


def build_wrapper_for_features(
    model: GPT | EncoderDecoderGPT,
    tokenizer: Any,
    *,
    features: dict[int, int] | None = None,
    target: SingleTargetSegment | MultiTargetSegments | None = None,
    context_provider: ContextProvider | None,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
    context_routing_mode: ContextRoutingMode = ContextRoutingMode.DECODER_ALIGNED,
    turn_contract: TurnContextContract | None = None,
) -> Any:
    """Wrap an EncoderDecoderGPT for feature-conditioned eval.

    Decoder-only models pass through (with empty/None features only).
    Decoder-only + non-empty features raises ValueError.
    Pass ``context_provider=None`` explicitly only for decoder-only paths.
    Encoder-decoder paths require a provider for real prompt/context features.

    Pass ``features`` (dict[int, int] | None) for the common case — the
    helper wraps it as ``SingleTargetSegment(features=features or {})``.

    Pass ``target`` for explicit ``MultiTargetSegments`` (PR-C lockstep
    consumer). Passing both ``features`` and ``target`` raises ValueError.
    """
    if features is not None and target is not None:
        raise ValueError(
            "build_wrapper_for_features: pass features OR target, not both."
        )
    if not isinstance(model, EncoderDecoderGPT):
        if turn_contract is not None or context_routing_mode is not ContextRoutingMode.DECODER_ALIGNED:
            raise ValueError("turn raw-source routing requires an encoder-decoder model")
        has_active_features = bool(features)
        has_explicit_target = target is not None
        if has_active_features or has_explicit_target:
            raise ValueError(
                "decoder-only models do not support feature/target conditioning; "
                f"got features={features!r}, target={target!r}. Pass None / "
                "an empty dict / no target, or use an EncoderDecoderGPT."
            )
        return model
    if context_provider is None:
        raise ValueError(
            "Encoder-decoder feature eval requires a context_provider with "
            "real prompt/context features."
        )
    resolved_target = (
        target if target is not None
        else SingleTargetSegment(features=dict(features or {}))
    )
    return TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=resolved_target,
        context_provider=context_provider,
        target_conditioning_mode=target_mode,
        context_routing_mode=context_routing_mode,
        turn_contract=turn_contract,
    )


def build_engine_for_features(
    model: GPT | EncoderDecoderGPT,
    tokenizer: Any,
    device: t.device,
    features: dict[int, int] | None,
    *,
    context_provider: ContextProvider | None,
    target_mode: TargetConditioningMode | str = TargetConditioningMode.CURRENT,
) -> InferenceEngine:
    """Build an InferenceEngine for full-context encoder-decoder conditioning.

    For decoder-only models, ``features`` must be None or empty. For
    encoder-decoder models, ``features`` are the explicit target-delta signal,
    a context provider is required for real prompt/context features, and the
    model is wrapped with ``TargetAwareEncoderDecoderEvalWrapper``.
    """
    wrapper_or_decoder = build_wrapper_for_features(
        model, tokenizer,
        features=features,
        context_provider=context_provider,
        target_mode=target_mode,
    )
    return InferenceEngine(wrapper_or_decoder, tokenizer, device)


def build_engine_self_conditioned(
    model: EncoderDecoderGPT,
    tokenizer: Any,
    device: t.device,
    self_cond_config: SelfCondConfig,
    *,
    context_provider: ContextProvider,
    context_routing_mode: ContextRoutingMode = ContextRoutingMode.DECODER_ALIGNED,
    turn_contract: TurnContextContract | None = None,
) -> InferenceEngine:
    """Build an InferenceEngine for self-conditioned generation.

    Prefill requires ``context_provider`` so prompt tokens are conditioned
    on real per-segment SAE features before self-conditioning begins.

    Fixed mode retains its historical decode cadence.  In turn-delimited mode,
    the terminal ``<assistant_start>`` prompt token is row one and every
    ``self_cond_config.self_cond_segment_size`` committed assistant inputs a
    fresh slot is appended via :meth:`SelfConditionedFeatureProvider.step`.

    Decoder-only models cannot self-condition (no encoder), so this
    helper requires :class:`EncoderDecoderGPT`.
    """
    if not isinstance(model, EncoderDecoderGPT):
        raise ValueError(
            "build_engine_self_conditioned requires an EncoderDecoderGPT; "
            f"got {type(model).__name__}. Decoder-only models cannot "
            "self-condition (no encoder)."
        )
    target = SelfConditionedTarget()
    wrapper = TargetAwareEncoderDecoderEvalWrapper(
        model=model,
        tokenizer=tokenizer,
        target=target,
        context_provider=context_provider,
        self_cond_config=self_cond_config,
        context_routing_mode=context_routing_mode,
        turn_contract=turn_contract,
    )
    return InferenceEngine(wrapper, tokenizer, device)
