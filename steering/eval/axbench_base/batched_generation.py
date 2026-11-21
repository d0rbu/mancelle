"""Custom batched-sample generation kernel for Stage 0c verification.

Stage 0c verification uses `batched_generate_seeds` to run its per-seed
generations in a single batched call. Each row maintains its own
`torch.Generator` so its token output is byte-identical to independent
single-seed Hugging Face generation.

Byte-identity rationale
-----------------------
Independent Hugging Face generation seeds one global RNG per call, then its
internal sampler draws from it. This kernel replicates the same draw sequence
per row by using `torch.multinomial(..., generator=row_gen)` on each row's
explicit generator. The parity tests in
``test_batched_vs_sequential_parity.py`` gate the merge on byte-identity
over a parametrised seed-set matrix.
"""

from __future__ import annotations

from typing import Any

import torch as t

from steering.eval.axbench_base.config import VerificationPrefillMode
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_base.types import AxBenchBaseError


def _hf_generation_model(base_model: Any) -> Any:
    """Locate the Hugging Face causal LM attached to ``base_model``."""
    nnterp_model = getattr(base_model, "nnterp_model", None)
    hf_model = getattr(nnterp_model, "_model", None)
    if hf_model is None:
        raise AxBenchBaseError(
            "batched_generate_seeds: base_model.nnterp_model must expose "
            "the StandardizedTransformer _model"
        )
    return hf_model


def _module_device(model: Any) -> t.device:
    """Best-effort device lookup; returns CPU if model has no parameters."""
    parameters = getattr(model, "parameters", None)
    if parameters is None:
        return t.device("cpu")
    try:
        return next(parameters()).device
    except StopIteration:
        return t.device("cpu")


def _prefill_forward_last_logits(hf_model: Any, **kwargs: Any) -> Any:
    """Run the prefill forward while materializing only last-token logits.

    Stage 0c generation needs the KV cache for the full prompt, but it only
    samples from the final prompt position. Gemma2's default CausalLM forward
    computes logits for every prompt token, which is prohibitively large for
    cross-prefix batches on a 256k-token vocab. The repository's minimum
    Transformers version exposes ``logits_to_keep=1`` for exactly this case.
    """
    return hf_model(**kwargs, logits_to_keep=1)


def _apply_top_k(logits: t.Tensor, top_k: int) -> t.Tensor:
    """Mask all but the top_k logits to -inf per row."""
    if top_k <= 0 or top_k >= logits.shape[-1]:
        return logits
    topk_vals, _ = logits.topk(top_k, dim=-1)
    threshold = topk_vals[..., -1, None]
    return t.where(
        logits < threshold,
        t.full_like(logits, float("-inf")),
        logits,
    )


def _generation_pad_token_id(hf_model: Any) -> int:
    """Best-effort pad token lookup for padded prefill batches."""
    config = getattr(hf_model, "config", None)
    if config is None:
        return 0
    pad_token_id = getattr(config, "pad_token_id", None)
    if pad_token_id is not None:
        return int(pad_token_id)
    eos_token_id = getattr(config, "eos_token_id", None)
    if isinstance(eos_token_id, int):
        return int(eos_token_id)
    if isinstance(eos_token_id, list) and eos_token_id:
        return int(eos_token_id[0])
    return 0


def _left_pad_token_rows(
    token_rows: list[list[int]],
    *,
    pad_token_id: int,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Left-pad token rows and build masks/positions for causal LMs."""
    if not token_rows:
        raise AxBenchBaseError("cross_prefix_batched_generate_seeds: no rows")
    row_lengths = [len(row) for row in token_rows]
    if any(length <= 0 for length in row_lengths):
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds received an empty prompt"
        )
    max_len = max(row_lengths)
    n_rows = len(token_rows)
    input_ids = t.full(
        (n_rows, max_len),
        int(pad_token_id),
        dtype=t.long,
        device=device,
    )
    attention_mask = t.zeros((n_rows, max_len), dtype=t.long, device=device)
    position_ids = t.zeros((n_rows, max_len), dtype=t.long, device=device)
    for row_idx, row in enumerate(token_rows):
        row_len = len(row)
        start = max_len - row_len
        input_ids[row_idx, start:] = t.tensor(row, dtype=t.long, device=device)
        attention_mask[row_idx, start:] = 1
        position_ids[row_idx, start:] = t.arange(
            row_len,
            dtype=t.long,
            device=device,
        )
    prompt_lengths = t.tensor(row_lengths, dtype=t.long, device=device)
    return input_ids, attention_mask, position_ids, prompt_lengths


def _seeded_row_generators(seeds: list[int], *, device: t.device) -> list[t.Generator]:
    gen_device = t.device("cpu") if device.type == "meta" else device
    row_generators: list[t.Generator] = []
    for s in seeds:
        g = t.Generator(device=gen_device)
        g.manual_seed(int(s))
        row_generators.append(g)
    return row_generators


def _flatten_prefix_seed_rows(
    token_id_rows: list[list[int]],
    seed_rows: list[list[int]],
) -> tuple[list[list[int]], list[int], list[int]]:
    if len(token_id_rows) != len(seed_rows):
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: token_id_rows and "
            f"seed_rows length mismatch ({len(token_id_rows)} != "
            f"{len(seed_rows)})"
        )
    flat_token_rows: list[list[int]] = []
    flat_seeds: list[int] = []
    seed_counts: list[int] = []
    for token_ids, seeds in zip(token_id_rows, seed_rows, strict=True):
        seed_counts.append(len(seeds))
        for seed in seeds:
            flat_token_rows.append(token_ids)
            flat_seeds.append(int(seed))
    return flat_token_rows, flat_seeds, seed_counts


def _unflatten_continuations(
    flat_continuations: list[list[int]],
    seed_counts: list[int],
) -> list[list[list[int]]]:
    out: list[list[list[int]]] = []
    offset = 0
    for count in seed_counts:
        out.append(flat_continuations[offset : offset + count])
        offset += count
    return out


def _build_prefix_indices(
    seed_counts: list[int],
    *,
    device: t.device,
) -> t.Tensor:
    """Map prefix-major decode rows back to their unique prefill rows."""
    return t.repeat_interleave(
        t.arange(len(seed_counts), dtype=t.long, device=device),
        t.tensor(seed_counts, dtype=t.long, device=device),
    )


def _cache_type_name(cache: object) -> str:
    cache_type = type(cache)
    return f"{cache_type.__module__}.{cache_type.__qualname__}"


def _expand_shared_prefill_cache(
    past_key_values: object,
    prefix_indices: t.Tensor,
    *,
    source_rows: int,
    rows: int,
    prefill_mode: VerificationPrefillMode,
) -> None:
    """Expand a public Transformers cache in place for shared prefill."""
    cache_type = _cache_type_name(past_key_values)
    select_indices = getattr(past_key_values, "batch_select_indices", None)
    if past_key_values is None or not callable(select_indices):
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: cache type "
            f"{cache_type} does not support prefill_mode={prefill_mode.value!r}; "
            "shared mode requires the Transformers 5.6.2 public "
            "Cache.batch_select_indices API. Use "
            "verification_prefill_mode=repeated only as an explicit operator "
            "choice."
        )
    try:
        select_indices(prefix_indices)
    except Exception as exc:
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: shared prefill cache "
            f"expansion failed for cache type {cache_type} from "
            f"source_rows={source_rows} to rows={rows} with "
            f"prefill_mode={prefill_mode.value!r}"
        ) from exc


@t.inference_mode()
def batched_generate_seeds(
    base_model: Any,
    token_ids: list[int],
    *,
    seeds: list[int],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    stop_token_ids: set[int],
) -> list[list[int]]:
    """Generate one continuation per seed, batched.

    Args:
        base_model: The base SAE LM as returned by RuntimeState.base_model().
            Must have ``.nnterp_model`` exposing an HF causal LM through
            :class:`nnterp.StandardizedTransformer`'s ``_model`` attribute.
        token_ids: Shared prompt token IDs (Gemma vocabulary).
        seeds: Per-row sampling seeds. Must be non-empty unless
            ``max_new_tokens == 0``.
        max_new_tokens: Maximum continuation length per row.
        temperature: Sampling temperature; 0 means greedy. Must be
            non-negative.
        top_k: Top-k sampling cutoff. ``None`` or ``<=0`` means disabled.
        stop_token_ids: Tokens that, when generated, terminate that row.
            Trailing stop tokens are removed from the returned per-row
            list.

    Returns:
        ``list[list[int]]`` of length ``len(seeds)``; inner list ``i`` is
        the trimmed continuation tokens for ``seeds[i]``.

    Raises:
        AxBenchBaseError: on invalid arguments (empty prompt, negative
            temperature, missing seeds).
    """
    if not token_ids:
        raise AxBenchBaseError(
            "batched_generate_seeds received an empty prompt"
        )
    if temperature < 0.0:
        raise AxBenchBaseError(
            "batched_generate_seeds: temperature must be non-negative, "
            f"got {temperature}"
        )
    if max_new_tokens <= 0:
        return [[] for _ in seeds]
    if not seeds:
        raise AxBenchBaseError(
            "batched_generate_seeds: at least one seed required when "
            "max_new_tokens > 0"
        )

    n_rows = len(seeds)
    hf_model = _hf_generation_model(base_model)
    device = _module_device(hf_model)

    # Build per-row generators on the model's device. Note: gloo/cpu
    # backends require generator device == cpu; cuda backends use cuda.
    # Pass the full torch.device (not just .type) so the CUDA index
    # ("cuda:1") is preserved — t.Generator(device="cuda") would default
    # to cuda:0 and crash on multi-GPU setups.
    row_generators = _seeded_row_generators(seeds, device=device)

    # Build the (N, prompt_len) input
    input_ids_t = t.tensor([token_ids] * n_rows, dtype=t.long, device=device)
    attention_mask_t = t.ones_like(input_ids_t)

    # Initial prefill: one batched forward to populate KV cache.
    with profile_region(
        "stage0c.batched_generate_seeds.prefill",
        rows=n_rows,
        prompt_len=len(token_ids),
    ):
        out = _prefill_forward_last_logits(
            hf_model,
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            use_cache=True,
        )
    # Upcast to fp32 to match HF generate's pre-sampling upcast
    # (transformers/generation/utils.py: next_token_logits.to(... fp32)).
    # Without this, bf16 division/softmax rounding diverges from sequential
    # per-seed output. .clone() detaches from the parent so `del out` can
    # free the full logits tensor.
    past_key_values = out.past_key_values
    last_logits = out.logits[:, -1, :].to(dtype=t.float32).clone()  # (N, V)
    del out

    continuations: list[list[int]] = [[] for _ in range(n_rows)]
    done_mask = t.zeros(n_rows, dtype=t.bool, device=device)

    do_sample = temperature > 0.0
    effective_top_k = top_k if (top_k is not None and top_k > 0) else 0

    for step in range(max_new_tokens):
        if bool(done_mask.all().item()):
            break

        if do_sample:
            scaled = last_logits / temperature
            if effective_top_k > 0:
                scaled = _apply_top_k(scaled, effective_top_k)
            probs = t.softmax(scaled, dim=-1)  # (N, V)
            next_tokens = t.empty(n_rows, dtype=t.long, device=device)
            for i in range(n_rows):
                if bool(done_mask[i].item()):
                    next_tokens[i] = 0  # placeholder; ignored
                    continue
                tok = t.multinomial(
                    probs[i],
                    num_samples=1,
                    generator=row_generators[i],
                )
                next_tokens[i] = tok.item()
        else:
            next_tokens = last_logits.argmax(dim=-1)

        for i in range(n_rows):
            if bool(done_mask[i].item()):
                continue
            tok = int(next_tokens[i].item())
            if tok in stop_token_ids:
                done_mask[i] = True
                continue
            continuations[i].append(tok)

        # Advance KV cache only if at least one row is still active.
        if not bool(done_mask.all().item()):
            with profile_region(
                "stage0c.batched_generate_seeds.sample_step",
                step=step,
                active_rows=int((~done_mask).sum().item()),
            ):
                out = hf_model(
                    input_ids=next_tokens.unsqueeze(1),
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            # See prefill comment: upcast to fp32 + clone + del out so the
            # parent logits tensor can be freed and downstream sampling
            # matches HF's fp32 path byte-identically.
            past_key_values = out.past_key_values
            last_logits = out.logits[:, -1, :].to(dtype=t.float32).clone()
            del out

    record_counter(
        "stage0c.batched_generate_seeds.tokens_per_call",
        sum(len(c) for c in continuations),
        rows=n_rows,
    )
    record_counter(
        "stage0c.batched_generate_seeds.rows_done_at_stop_token",
        int(done_mask.sum().item()),
        rows=n_rows,
    )

    return continuations


@t.inference_mode()
def cross_prefix_batched_generate_seeds(
    base_model: Any,
    token_id_rows: list[list[int]],
    *,
    seed_rows: list[list[int]],
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    stop_token_ids: set[int],
    prefill_mode: VerificationPrefillMode | str,
) -> list[list[list[int]]]:
    """Generate continuations for K prefixes x N seeds in one decode loop.

    ``shared`` prefills K rows and expands them to the prefix-major M-row
    decode layout. ``repeated`` prefills the legacy M repeated rows. The
    latter still delegates K=1 to :func:`batched_generate_seeds` exactly.
    """
    if len(token_id_rows) != len(seed_rows):
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: token_id_rows and "
            f"seed_rows length mismatch ({len(token_id_rows)} != "
            f"{len(seed_rows)})"
        )
    if not token_id_rows:
        return []
    if temperature < 0.0:
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: temperature must be "
            f"non-negative, got {temperature}"
        )
    if any(not token_ids for token_ids in token_id_rows):
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds received an empty prompt"
        )
    if max_new_tokens <= 0:
        return [[[] for _ in seeds] for seeds in seed_rows]
    if any(not seeds for seeds in seed_rows):
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: every prefix needs at "
            "least one seed when max_new_tokens > 0"
        )
    try:
        resolved_prefill_mode = VerificationPrefillMode(prefill_mode)
    except ValueError as exc:
        raise AxBenchBaseError(
            "cross_prefix_batched_generate_seeds: prefill_mode must be "
            f"'shared' or 'repeated', got {prefill_mode!r}"
        ) from exc

    flat_token_rows, flat_seeds, seed_counts = _flatten_prefix_seed_rows(
        token_id_rows, seed_rows,
    )
    if (
        resolved_prefill_mode is VerificationPrefillMode.REPEATED
        and len(token_id_rows) == 1
    ):
        return [
            batched_generate_seeds(
                base_model,
                token_id_rows[0],
                seeds=seed_rows[0],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                stop_token_ids=stop_token_ids,
            )
        ]

    n_rows = len(flat_seeds)
    hf_model = _hf_generation_model(base_model)
    device = _module_device(hf_model)
    row_generators = _seeded_row_generators(flat_seeds, device=device)
    pad_token_id = _generation_pad_token_id(hf_model)
    if resolved_prefill_mode is VerificationPrefillMode.SHARED:
        prefill_token_rows = token_id_rows
    else:
        prefill_token_rows = flat_token_rows
    (
        input_ids_t,
        attention_mask_t,
        position_ids_t,
        prompt_lengths_t,
    ) = _left_pad_token_rows(
        prefill_token_rows,
        pad_token_id=pad_token_id,
        device=device,
    )
    prefill_row_count = len(prefill_token_rows)
    avoided_prefill_rows = n_rows - prefill_row_count

    with profile_region(
        "stage0c.cross_prefix_batched_generate.prefill",
        prefixes=len(token_id_rows),
        rows=prefill_row_count,
        decode_rows=n_rows,
        avoided_prefill_rows=avoided_prefill_rows,
        prefill_mode=resolved_prefill_mode.value,
        max_prompt_len=int(input_ids_t.shape[1]),
    ):
        out = _prefill_forward_last_logits(
            hf_model,
            input_ids=input_ids_t,
            attention_mask=attention_mask_t,
            position_ids=position_ids_t,
            use_cache=True,
        )
    past_key_values = out.past_key_values
    last_logits = out.logits[:, -1, :].to(dtype=t.float32).clone()
    del out

    expansion_calls = 0
    if resolved_prefill_mode is VerificationPrefillMode.SHARED:
        prefix_indices = _build_prefix_indices(seed_counts, device=device)
        if prefix_indices.numel() != n_rows:
            raise AxBenchBaseError(
                "cross_prefix_batched_generate_seeds: malformed shared-prefill "
                f"expansion index has {prefix_indices.numel()} rows; expected "
                f"{n_rows}"
            )
        with profile_region(
            "stage0c.cross_prefix_batched_generate.expand_prefill_state",
            source_rows=len(token_id_rows),
            rows=n_rows,
            prefixes=len(token_id_rows),
            prefill_mode=resolved_prefill_mode.value,
        ):
            _expand_shared_prefill_cache(
                past_key_values,
                prefix_indices,
                source_rows=len(token_id_rows),
                rows=n_rows,
                prefill_mode=resolved_prefill_mode,
            )
            last_logits = last_logits.index_select(0, prefix_indices).clone()
            attention_mask_t = attention_mask_t.index_select(
                0,
                prefix_indices,
            ).clone()
            prompt_lengths_t = prompt_lengths_t.index_select(
                0,
                prefix_indices,
            ).clone()
        expansion_calls = 1
        del prefix_indices

    del input_ids_t, position_ids_t, prefill_token_rows, flat_token_rows

    counter_attrs = {
        "prefixes": len(token_id_rows),
        "prefill_mode": resolved_prefill_mode.value,
    }
    record_counter(
        "stage0c.cross_prefix_batched_generate.prefill_rows",
        prefill_row_count,
        **counter_attrs,
    )
    record_counter(
        "stage0c.cross_prefix_batched_generate.decode_rows",
        n_rows,
        **counter_attrs,
    )
    record_counter(
        "stage0c.cross_prefix_batched_generate.avoided_prefill_rows",
        avoided_prefill_rows,
        **counter_attrs,
    )
    record_counter(
        "stage0c.cross_prefix_batched_generate.expansion_calls",
        expansion_calls,
        **counter_attrs,
    )

    continuations: list[list[int]] = [[] for _ in range(n_rows)]
    done_flags = [False for _ in range(n_rows)]
    active_count = n_rows
    do_sample = temperature > 0.0
    effective_top_k = top_k if (top_k is not None and top_k > 0) else 0

    for step in range(max_new_tokens):
        if active_count == 0:
            break

        if do_sample:
            scaled = last_logits / temperature
            if effective_top_k > 0:
                scaled = _apply_top_k(scaled, effective_top_k)
            probs = t.softmax(scaled, dim=-1)
            next_tokens = t.zeros(n_rows, dtype=t.long, device=device)
            for row_idx, done in enumerate(done_flags):
                if done:
                    continue
                tok = t.multinomial(
                    probs[row_idx],
                    num_samples=1,
                    generator=row_generators[row_idx],
                )
                next_tokens[row_idx] = tok.squeeze(0)
        else:
            next_tokens = last_logits.argmax(dim=-1)
            if any(done_flags):
                done_idx = [
                    row_idx for row_idx, done in enumerate(done_flags) if done
                ]
                next_tokens[done_idx] = 0

        next_token_values = [int(tok) for tok in next_tokens.tolist()]
        for row_idx, tok in enumerate(next_token_values):
            if done_flags[row_idx]:
                continue
            if tok in stop_token_ids:
                done_flags[row_idx] = True
                active_count -= 1
                continue
            continuations[row_idx].append(tok)

        if active_count > 0:
            attention_mask_t = t.cat(
                [
                    attention_mask_t,
                    t.ones((n_rows, 1), dtype=t.long, device=device),
                ],
                dim=1,
            )
            decode_position_ids = (prompt_lengths_t + step).unsqueeze(1)
            with profile_region(
                "stage0c.cross_prefix_batched_generate.sample_step",
                step=step,
                active_rows=active_count,
                rows=n_rows,
            ):
                out = hf_model(
                    input_ids=next_tokens.unsqueeze(1),
                    attention_mask=attention_mask_t,
                    position_ids=decode_position_ids,
                    past_key_values=past_key_values,
                    use_cache=True,
                )
            past_key_values = out.past_key_values
            last_logits = out.logits[:, -1, :].to(dtype=t.float32).clone()
            del out

    record_counter(
        "stage0c.cross_prefix_batched_generate.tokens_per_call",
        sum(len(c) for c in continuations),
        prefixes=len(token_id_rows),
        rows=n_rows,
    )
    record_counter(
        "stage0c.cross_prefix_batched_generate.rows_done_at_stop_token",
        sum(1 for done in done_flags if done),
        prefixes=len(token_id_rows),
        rows=n_rows,
    )

    return _unflatten_continuations(continuations, seed_counts)


__all__ = ["batched_generate_seeds", "cross_prefix_batched_generate_seeds"]
