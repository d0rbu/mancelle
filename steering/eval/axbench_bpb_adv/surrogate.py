"""Teacher-forced BPB losses and prompt-token gradients for BPB adversaries."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

import torch as t
from beartype import beartype
from loguru import logger

from nanochat.engine import KVCache
from scripts.eval.axbench_adv.gcg_math import weighted_bpb
from steering.conditional.encoder_decoder.packed_encoder import (
    PackedEncoderOutput,
    repeat_packed_encoder_output,
)
from steering.conditional.feature_gcg.edit_primitives import EditableSpan
from steering.eval.axbench_base.profiler import profile_region, record_counter
from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.seg_map import project_segments_to_truncated_prefix
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    _build_encdec_seg_map_and_encoder,
    _padded_idx_unique_per_row,
    assert_enc_ids_in_vocab,
    assert_enc_ids_match_cluster_contract,
    build_output_only_mask,
)
from steering.eval.axbench_bpb.tokenization import truncate_prefix_left
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb.vocab_restriction import feature_vocab_allowed_ids
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt, PositiveIntLike


@dataclass(frozen=True)
class BpbGradientResult:
    """BPB scalar and GCG-compatible embedding gradient for one prompt."""

    loss: float
    token_gradient: t.Tensor
    embedding_weight: t.Tensor


def _embedding_owner(model: Any) -> Any:
    if hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
        return model
    inner = getattr(model, "model", None)
    if inner is not None and hasattr(inner, "transformer") and hasattr(
        inner.transformer,
        "wte",
    ):
        return inner
    raise AxBenchBpbAdvError(
        f"model {type(model).__name__} does not expose transformer.wte"
    )


def _value_embedding_modules(owner: Any) -> list[Any]:
    value_embeds = getattr(owner, "value_embeds", None)
    if value_embeds is None:
        return []
    try:
        items = list(value_embeds.items())
    except AttributeError:
        return []

    def _sort_key(item: tuple[str, Any]) -> tuple[int, str]:
        key = str(item[0])
        try:
            return (int(key), key)
        except ValueError:
            return (10**9, key)

    return [module for _, module in sorted(items, key=_sort_key)]


def _concatenate_embedding_parts(parts: list[t.Tensor]) -> t.Tensor:
    if not parts:
        raise AxBenchBpbAdvError("no embedding weights were captured")
    row_counts = {int(part.shape[0]) for part in parts}
    if len(row_counts) != 1:
        raise AxBenchBpbAdvError(
            "all embedding tables used for Faster-GCG must have the same "
            f"vocab row count; got {sorted(row_counts)}"
        )
    return t.cat(parts, dim=1)


def bpb_surrogate_embedding_weight(model: Any) -> t.Tensor:
    """Return the concatenated token/value embedding matrix used by gradients."""

    owner = _embedding_owner(model)
    parts = [owner.transformer.wte.weight.detach().float().cpu()]
    for module in _value_embedding_modules(owner):
        parts.append(module.weight.detach().float().cpu())
    return _concatenate_embedding_parts(parts)


@contextmanager
def _freeze_model_parameters(model: Any) -> Iterator[None]:
    """Temporarily disable model parameter gradients without touching ``.grad``."""

    parameters = tuple(model.parameters())
    requires_grad = tuple(parameter.requires_grad for parameter in parameters)
    try:
        for parameter in parameters:
            parameter.requires_grad_(False)
        yield
    finally:
        for parameter, original_requires_grad in zip(
            parameters, requires_grad, strict=True
        ):
            parameter.requires_grad_(original_requires_grad)


def _output_byte_weights(
    *,
    output_tokens: list[int],
    token_bytes: t.Tensor,
    device: t.device,
) -> t.Tensor:
    output_tensor = t.tensor(output_tokens, dtype=t.long)
    try:
        weights = token_bytes[output_tensor]
    except IndexError as exc:
        raise AxBenchBpbAdvError("output token outside tokenizer vocab") from exc
    if int(weights.sum().item()) <= 0:
        raise AxBenchBpbAdvError("output tokens decode to zero bytes")
    return weights.to(device=device, dtype=t.float32)


def _prompt_gradient_from_full_gradient(
    *,
    full_gradient: t.Tensor,
    prompt_len: int,
    prompt_prefix_len: int,
    tokens_dropped_total: int,
) -> t.Tensor:
    out = t.zeros((prompt_len, full_gradient.shape[1]), dtype=full_gradient.dtype)
    for prompt_index in range(prompt_len):
        pre_truncation_index = prompt_prefix_len + prompt_index
        if pre_truncation_index < tokens_dropped_total:
            continue
        truncated_index = pre_truncation_index - tokens_dropped_total
        if truncated_index >= full_gradient.shape[0]:
            continue
        out[prompt_index] = full_gradient[truncated_index]
    return out


@dataclass(frozen=True)
class _TeacherForcedLayout:
    prompt: list[int]
    output: list[int]
    prompt_prefix_tokens: list[int]
    full_prefix: list[int]
    tokens_dropped_total: int
    n_full_prefix: int
    truncated_prefix_segments: tuple[ContextFeatureSegment, ...]
    n_prompt_surviving: int


@dataclass(frozen=True)
class _EncoderTensors:
    """Encoder-side teacher-forcing tensors (all ``None`` for decoder-only)."""

    feature_ids: t.Tensor | None = None
    magnitudes: t.Tensor | None = None
    mask: t.Tensor | None = None
    seg_map: t.Tensor | None = None
    # Checkpoint's compacted vocabulary V (None if uncompacted); carried
    # alongside the tensors it was already derived from feeding into
    # _build_encdec_seg_map_and_encoder, so _teacher_forced_model_forward's
    # eager assert_enc_ids_in_vocab call can reuse it without re-deriving.
    allowed_ids: t.Tensor | None = None


@dataclass(frozen=True)
class _SharedPrefixEncoderCpuState:
    """CPU structural encoder state retained for one compatibility group."""

    feature_ids: t.Tensor
    magnitudes: t.Tensor
    mask: t.Tensor
    full_seg_map: t.Tensor
    allowed_ids: t.Tensor | None
    generation_slot: int


@dataclass(frozen=True)
class _SharedPrefixPreparedSample:
    """Validated suffix slices for one caller-ordered teacher-forcing sample."""

    layout: _TeacherForcedLayout
    shared_prefix_len: int
    editable_count: int
    dec_input: tuple[int, ...]
    dec_target: tuple[int, ...]
    loss_mask: tuple[bool, ...]
    full_seg_map: t.Tensor | None
    generation_slot: int | None


@dataclass
class _SharedPrefixCompatibilityGroup:
    """Samples whose decoder and encoder prefix state can share one KV prefill."""

    prefix_tokens: tuple[int, ...]
    sample_indices: list[int]
    encoder: _SharedPrefixEncoderCpuState | None


def _cpu_tensor_structural_key(tensor: t.Tensor) -> tuple[Any, ...]:
    """Hashable exact tensor identity without a device synchronization.

    Shared-prefix grouping builds encoder tensors directly on CPU. Keeping this
    helper CPU-only is load-bearing: calling ``tolist`` on a CUDA tensor for each
    candidate would serialize the hot path on one synchronization per sample.
    """

    if tensor.device.type != "cpu":
        raise AxBenchBpbAdvError(
            "shared-prefix compatibility keys must be constructed from CPU tensors"
        )
    contiguous = tensor.detach().contiguous()
    return (
        tuple(int(dim) for dim in contiguous.shape),
        str(contiguous.dtype),
        tuple(contiguous.view(-1).tolist()),
    )


@dataclass(frozen=True)
class _CachedCpuTensorStructuralKey:
    shape: tuple[int, ...]
    dtype: str
    values: tuple[Any, ...]
    cached_hash: int

    def __hash__(self) -> int:
        return self.cached_hash


def _cpu_tensor_cached_structural_key(
    tensor: t.Tensor,
) -> _CachedCpuTensorStructuralKey:
    """Return an exact structural key whose expensive hash is computed once.

    The values remain in the key, so equality is exact rather than digest-based.
    Caching the composite hash avoids re-hashing tens of thousands of compacted
    vocabulary IDs every time the same call-level key is nested into a sample's
    compatibility key.
    """

    if tensor.device.type != "cpu":
        raise AxBenchBpbAdvError(
            "shared-prefix cached keys must be constructed from CPU tensors"
        )
    contiguous = tensor.detach().contiguous()
    shape = tuple(int(dim) for dim in contiguous.shape)
    dtype = str(contiguous.dtype)
    values = tuple(contiguous.view(-1).tolist())
    return _CachedCpuTensorStructuralKey(
        shape=shape,
        dtype=dtype,
        values=values,
        cached_hash=hash((shape, dtype, values)),
    )


def _cluster_contract_identity(target_bundle: TargetModelBundle) -> str | None:
    contract = target_bundle.cluster_contract
    if contract is None:
        return None
    fingerprint = getattr(contract, "feature_contract_fingerprint", None)
    if not isinstance(fingerprint, str) or not fingerprint:
        raise AxBenchBpbAdvError(
            "clustered shared-prefix scoring requires a non-empty feature-contract fingerprint"
        )
    return fingerprint


def _build_teacher_forced_layout(
    *,
    target_bundle: TargetModelBundle,
    prompt_tokens: list[int],
    output_tokens: list[int],
    prefix_context_segments: tuple[ContextFeatureSegment, ...],
    intervention: Intervention,
) -> _TeacherForcedLayout:
    prompt = [int(token_id) for token_id in prompt_tokens]
    output = [int(token_id) for token_id in output_tokens]
    if not prompt:
        raise AxBenchBpbAdvError("prompt_tokens must be non-empty")
    if not output:
        raise AxBenchBpbAdvError("output_tokens must be non-empty")

    prompt_prefix_tokens = list(intervention.prompt_prefix_tokens or [])
    full_prefix_pre_truncation = [*prompt_prefix_tokens, *prompt]
    full_prefix, truncated = truncate_prefix_left(
        prefix=full_prefix_pre_truncation,
        n_output=len(output),
        max_seq_len=target_bundle.max_seq_len,
    )
    if truncated and not full_prefix:
        raise AxBenchBpbAdvError("prompt was fully truncated before BPB objective")
    tokens_dropped_total = len(full_prefix_pre_truncation) - len(full_prefix)

    truncated_prefix_segments = prefix_context_segments
    n_prompt_surviving = 0
    if target_bundle.target_name == "encdec":
        n_prompt_prefix = len(prompt_prefix_tokens)
        n_inputs_prefix = len(prompt)
        n_remaining_in_full = len(full_prefix)
        tokens_of_inputs_dropped = max(0, tokens_dropped_total - n_prompt_prefix)
        n_inputs_surviving = n_inputs_prefix - tokens_of_inputs_dropped
        n_prompt_surviving = n_remaining_in_full - n_inputs_surviving
        if (
            prefix_context_segments
            and tokens_of_inputs_dropped > 0
            and len(full_prefix) < len(full_prefix_pre_truncation)
        ):
            truncated_inputs_prefix = prompt[tokens_of_inputs_dropped:]
            truncated_prefix_segments = project_segments_to_truncated_prefix(
                prefix_context_segments,
                original_prefix_tokens=prompt,
                truncated_prefix_tokens=truncated_inputs_prefix,
                tokenizer=target_bundle.tokenizer,
            )

    return _TeacherForcedLayout(
        prompt=prompt,
        output=output,
        prompt_prefix_tokens=prompt_prefix_tokens,
        full_prefix=full_prefix,
        tokens_dropped_total=tokens_dropped_total,
        n_full_prefix=len(full_prefix),
        truncated_prefix_segments=truncated_prefix_segments,
        n_prompt_surviving=n_prompt_surviving,
    )


def _build_teacher_forced_forward_inputs(
    *,
    target_bundle: TargetModelBundle,
    layout: _TeacherForcedLayout,
    intervention: Intervention,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, _EncoderTensors]:
    """Build the decoder inputs/targets/mask and (encdec) encoder tensors.

    Shared by the gradient and loss-only paths so both feed the model byte-
    identical inputs.
    """

    all_tokens = [*layout.full_prefix, *layout.output]
    dec_input = t.tensor([all_tokens[:-1]], dtype=t.long, device=device)
    dec_target = t.tensor([all_tokens[1:]], dtype=t.long, device=device)
    mask = build_output_only_mask(
        n_full_prefix=PositiveInt.parse(layout.n_full_prefix),
        n_output=PositiveInt.parse(len(layout.output)),
    ).to(device)

    encoder = _EncoderTensors()
    if target_bundle.target_name == "encdec":
        # Derived ONCE and reused both for the seg-map builder and (via the
        # returned _EncoderTensors.allowed_ids) for
        # _teacher_forced_model_forward's eager pre-forward vocabulary
        # assert — avoids a second feature_vocab_ids read for this call.
        allowed_ids = feature_vocab_allowed_ids(target_bundle.model)
        enc_fids, enc_mags, enc_mask, seg_map, _generation_slot = (
            _build_encdec_seg_map_and_encoder(
                tokenizer=target_bundle.tokenizer,
                full_prefix=layout.full_prefix,
                prefix_segments=layout.truncated_prefix_segments,
                feature_conditions=intervention.feature_conditions,
                n_output=len(layout.output),
                n_prompt_surviving=layout.n_prompt_surviving,
                device=device,
                allowed_ids=allowed_ids,
                cluster_contract=target_bundle.cluster_contract,
            )
        )
        encoder = _EncoderTensors(
            feature_ids=enc_fids,
            magnitudes=enc_mags,
            mask=enc_mask,
            seg_map=seg_map[:, :-1],
            allowed_ids=allowed_ids,
        )
    return dec_input, dec_target, mask, encoder


def _teacher_forced_model_forward(
    *,
    target_bundle: TargetModelBundle,
    dec_input: t.Tensor,
    dec_target: t.Tensor,
    encoder: _EncoderTensors,
) -> t.Tensor:
    """Run the (encdec or decoder-only) teacher-forced forward.

    This is the SINGLE forward call shared by the gradient and loss-only paths,
    so the per-token loss tensor is computed by byte-identical code in both.
    """

    if target_bundle.target_name == "encdec":
        assert encoder.feature_ids is not None and encoder.mask is not None
        # Eager backstop (issue #695 PR-1 Task 4): every REAL
        # (enc_mask-True) id must be in the checkpoint's compacted
        # vocabulary V, checked immediately before this forward — reuses
        # the allowed_ids already derived in
        # _build_teacher_forced_forward_inputs. See
        # steering.eval.axbench_bpb.teacher_force.assert_enc_ids_in_vocab's
        # docstring for why this exists alongside
        # FeatureEncoder._to_compact's own (async) OOV assert.
        assert_enc_ids_in_vocab(encoder.feature_ids, encoder.mask, encoder.allowed_ids)
        if target_bundle.cluster_contract is not None:
            assert_enc_ids_match_cluster_contract(
                encoder.feature_ids,
                encoder.mask,
                target_bundle.cluster_contract,
            )
        target_bundle.ensure_ready_for_forward()
        return target_bundle.model(
            dec_input,
            dec_target,
            enc_feature_ids=encoder.feature_ids,
            enc_magnitudes=encoder.magnitudes,
            enc_mask=encoder.mask,
            seg_map=encoder.seg_map,
            loss_reduction="none",
        )
    target_bundle.ensure_ready_for_forward()
    return target_bundle.model(
        dec_input,
        targets=dec_target,
        loss_reduction="none",
    )


def _bpb_tensor_from_loss(
    *,
    loss: t.Tensor,
    mask: t.Tensor,
    layout: _TeacherForcedLayout,
    output_token_bytes: t.Tensor,
    device: t.device,
) -> t.Tensor:
    """Reduce the per-token loss to the scalar BPB objective.

    Shared by the gradient and loss-only paths, so the BPB scalar is computed by
    byte-identical code in both (and identical guards fire in both).
    """

    loss_flat = loss.view(-1)
    if loss_flat.shape != mask.shape:
        raise AxBenchBpbAdvError(
            f"loss/mask shape mismatch: {loss_flat.shape} vs {mask.shape}"
        )
    in_mask_loss = loss_flat[mask]
    if not bool(t.isfinite(in_mask_loss).all()):
        raise AxBenchBpbAdvError("non-finite BPB objective loss")
    byte_weights = _output_byte_weights(
        output_tokens=layout.output,
        token_bytes=output_token_bytes,
        device=device,
    )
    if byte_weights.numel() != in_mask_loss.numel():
        raise AxBenchBpbAdvError(
            "byte-weight/output loss length mismatch: "
            f"{byte_weights.numel()} vs {in_mask_loss.numel()}"
        )
    return weighted_bpb(in_mask_loss, byte_weights, tensor_ln2=True)


def _pack_decoder_microbatch(
    *,
    layouts: list[_TeacherForcedLayout],
    device: t.device,
    t_pad: int,
) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Pad a list of decoder layouts into one ``(B, t_pad-1)`` teacher-forced batch.

    Returns ``(dec_input, dec_target, masks)``: ``dec_input`` is right-padded with 0,
    ``dec_target`` with -1 (the model's cross-entropy ignore_index), and ``masks``
    marks each sample's output positions. The per-sample packing is byte-identical
    to the single-prompt :func:`_build_teacher_forced_forward_inputs`
    (``[*full_prefix, *output]``, shifted), so a B=1 batch reproduces the
    single-prompt forward exactly. Shared by the loss and gradient microbatch paths
    so both feed the model the same tensors.

    ``t_pad`` is a CALL-level constant (the max sample length over the whole batched
    call), not the microbatch-local max: padding every microbatch to the same length
    holds the ``(B, T)`` forward shape constant across steps, so the ``dynamic=False``
    compiled flex-attention kernel is compiled ONCE and stays fused instead of
    blowing the recompile limit and falling back to the eager (full-scores-matrix)
    path. Right-padding + causal attention keeps the extra padded tail invisible to
    every valid position.
    """

    all_tokens = [[*lay.full_prefix, *lay.output] for lay in layouts]
    lengths = [len(toks) for toks in all_tokens]
    if t_pad < max(lengths):
        raise AxBenchBpbAdvError(
            f"t_pad ({t_pad}) shorter than longest sample ({max(lengths)})"
        )
    b = len(layouts)
    dec_input = t.zeros(b, t_pad - 1, dtype=t.long, device=device)
    dec_target = t.full((b, t_pad - 1), -1, dtype=t.long, device=device)
    masks = t.zeros(b, t_pad - 1, dtype=t.bool, device=device)
    for i, toks in enumerate(all_tokens):
        length = lengths[i]
        dec_input[i, : length - 1] = t.tensor(toks[:-1], dtype=t.long, device=device)
        dec_target[i, : length - 1] = t.tensor(toks[1:], dtype=t.long, device=device)
        masks[i, : length - 1] = build_output_only_mask(
            n_full_prefix=PositiveInt.parse(layouts[i].n_full_prefix),
            n_output=PositiveInt.parse(len(layouts[i].output)),
        ).to(device)
    return dec_input, dec_target, masks


def _microbatch_bpb_tensors(
    *,
    loss: t.Tensor,
    masks: t.Tensor,
    layouts: list[_TeacherForcedLayout],
    output_token_bytes: t.Tensor,
    device: t.device,
) -> list[t.Tensor]:
    """Reduce a ``(B, T)`` per-token loss to one differentiable BPB scalar per sample.

    Each sample reduces over ITS OWN output mask + byte weights with the exact same
    formula as the single-prompt :func:`_bpb_tensor_from_loss`, so the returned
    tensors are byte-identical to the single-prompt path (and carry grad: the
    gradient path sums them; the loss path detaches them). Raises on a non-finite
    in-mask loss (the finite-loss guard) and on a byte-weight/output length
    mismatch."""

    tensors: list[t.Tensor] = []
    for i in range(len(layouts)):
        in_mask_loss = loss[i][masks[i]]
        if not bool(t.isfinite(in_mask_loss).all()):
            raise AxBenchBpbAdvError("non-finite BPB objective loss (batched)")
        byte_weights = _output_byte_weights(
            output_tokens=layouts[i].output,
            token_bytes=output_token_bytes,
            device=device,
        )
        if byte_weights.numel() != in_mask_loss.numel():
            raise AxBenchBpbAdvError(
                "byte-weight/output loss length mismatch (batched): "
                f"{byte_weights.numel()} vs {in_mask_loss.numel()}"
            )
        tensors.append(weighted_bpb(in_mask_loss, byte_weights, tensor_ln2=False))
    return tensors


def _validate_seg_map_for_shared_prefix(
    seg_map: t.Tensor,
    *,
    n_segments: int,
    surface: str,
) -> None:
    if seg_map.device.type != "cpu":
        raise AxBenchBpbAdvError(f"{surface}: seg_map validation requires a CPU tensor")
    if seg_map.dim() != 2 or seg_map.shape[0] < 1 or seg_map.shape[1] < 1:
        raise AxBenchBpbAdvError(
            f"{surface}: seg_map must have shape (B, T) with B,T >= 1, got "
            f"{tuple(seg_map.shape)}"
        )
    if n_segments < 1:
        raise AxBenchBpbAdvError(
            f"{surface}: encoder must contain at least one segment, got {n_segments}"
        )
    minimum = int(seg_map.min().item())
    maximum = int(seg_map.max().item())
    if minimum < 0 or maximum >= n_segments:
        raise AxBenchBpbAdvError(
            f"{surface}: seg_map range [{minimum}, {maximum}] is outside "
            f"[0, {n_segments})"
        )
    _padded_idx_unique_per_row(seg_map, n_segments)


def _prepare_shared_prefix_groups(
    *,
    target_bundle: TargetModelBundle,
    prompts: list[list[int]],
    source_prompts: list[list[int]],
    outputs: list[list[int]],
    prefix_context_segments_list: list[tuple[ContextFeatureSegment, ...]],
    editable_spans: list[EditableSpan],
    intervention: Intervention,
) -> tuple[list[_SharedPrefixPreparedSample], list[_SharedPrefixCompatibilityGroup]]:
    """Validate candidate layouts and partition exact cache-compatible groups."""

    n = len(prompts)
    lengths = {
        "source_prompts": len(source_prompts),
        "outputs": len(outputs),
        "prefix_context_segments_list": len(prefix_context_segments_list),
        "editable_spans": len(editable_spans),
    }
    mismatched = {name: value for name, value in lengths.items() if value != n}
    if mismatched:
        raise AxBenchBpbAdvError(
            "teacher_forced_bpb_loss_shared_prefix_kv_batch: malformed batch "
            f"lengths prompts={n}, {mismatched}"
        )
    if n == 0:
        return [], []

    editable_counts = {span.count for span in editable_spans}
    if len(editable_counts) != 1:
        raise AxBenchBpbAdvError(
            "shared-prefix candidate batch has inconsistent editable-token counts: "
            f"{sorted(editable_counts)}"
        )
    editable_count = next(iter(editable_counts))
    if editable_count <= 0:
        raise AxBenchBpbAdvError(
            f"shared-prefix KV reuse requires k > 0, got k={editable_count}"
        )

    allowed_ids_cpu: t.Tensor | None = None
    allowed_ids_key: _CachedCpuTensorStructuralKey | None = None
    if target_bundle.target_name == "encdec":
        allowed_ids = feature_vocab_allowed_ids(target_bundle.model)
        if allowed_ids is not None:
            # One call-level transfer, before the per-sample structural loop.
            allowed_ids_cpu = allowed_ids.detach().to(
                device="cpu", dtype=t.int64
            ).contiguous()
            # The compacted vocabulary is constant for the whole scorer call.
            # Serializing its potentially tens-of-thousands of IDs per sample
            # would turn grouping into O(samples * vocabulary_size) Python work.
            allowed_ids_key = _cpu_tensor_cached_structural_key(allowed_ids_cpu)
    cluster_identity = _cluster_contract_identity(target_bundle)

    prepared: list[_SharedPrefixPreparedSample] = []
    group_by_key: dict[tuple[Any, ...], _SharedPrefixCompatibilityGroup] = {}
    for index in range(n):
        prompt = [int(token_id) for token_id in prompts[index]]
        source_prompt = [int(token_id) for token_id in source_prompts[index]]
        span = editable_spans[index]
        if span.start < 0 or span.count <= 0:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: invalid editable span "
                f"start={span.start}, count={span.count}"
            )
        if span.count != editable_count:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: editable span width {span.count} "
                f"!= batch k={editable_count}"
            )
        if len(prompt) != len(source_prompt):
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: replacement changed prompt length "
                f"from {len(source_prompt)} to {len(prompt)}"
            )
        if span.stop != len(source_prompt) or span.stop != len(prompt):
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: editable span [{span.start}, "
                f"{span.stop}) must end at prompt boundary {len(prompt)}"
            )
        if prompt[: span.start] != source_prompt[: span.start]:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: token replacement modified the "
                "fixed prefix outside the editable span"
            )

        layout = _build_teacher_forced_layout(
            target_bundle=target_bundle,
            prompt_tokens=prompt,
            output_tokens=outputs[index],
            prefix_context_segments=prefix_context_segments_list[index],
            intervention=intervention,
        )
        editable_start_before_truncation = len(layout.prompt_prefix_tokens) + span.start
        if layout.tokens_dropped_total > editable_start_before_truncation:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: left truncation dropped "
                "at least one editable trigger token"
            )
        if layout.n_full_prefix < editable_count:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: post-truncation prefix length "
                f"{layout.n_full_prefix} is shorter than k={editable_count}"
            )
        shared_prefix_len = layout.n_full_prefix - editable_count
        if shared_prefix_len <= 0:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: no fixed-prefix token remains "
                f"after truncation (n_full_prefix={layout.n_full_prefix}, "
                f"k={editable_count})"
            )

        all_tokens = [*layout.full_prefix, *layout.output]
        dec_input = tuple(all_tokens[shared_prefix_len:-1])
        dec_target = tuple(all_tokens[shared_prefix_len + 1 :])
        if len(dec_input) != len(dec_target) or not dec_input:
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: malformed suffix shift lengths "
                f"input={len(dec_input)}, target={len(dec_target)}"
            )
        full_mask = build_output_only_mask(
            n_full_prefix=PositiveInt.parse(layout.n_full_prefix),
            n_output=PositiveInt.parse(len(layout.output)),
        )
        suffix_mask = full_mask[shared_prefix_len:]
        independent_suffix_mask = build_output_only_mask(
            n_full_prefix=PositiveInt.parse(editable_count),
            n_output=PositiveInt.parse(len(layout.output)),
        )
        if not t.equal(suffix_mask, independent_suffix_mask):
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: sliced output mask does not "
                "equal the independently constructed suffix mask"
            )
        first_output_position = editable_count - 1
        true_positions = suffix_mask.nonzero(as_tuple=False).view(-1)
        if (
            true_positions.numel() != len(layout.output)
            or int(true_positions[0].item()) != first_output_position
        ):
            raise AxBenchBpbAdvError(
                f"shared-prefix sample {index}: output loss must begin at suffix "
                f"index k-1={first_output_position}"
            )

        encoder_state: _SharedPrefixEncoderCpuState | None = None
        encoder_key: tuple[Any, ...] = ("decoder_only",)
        full_seg_map: t.Tensor | None = None
        generation_slot: int | None = None
        if target_bundle.target_name == "encdec":
            (
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map_with_final,
                generation_slot,
            ) = _build_encdec_seg_map_and_encoder(
                tokenizer=target_bundle.tokenizer,
                full_prefix=layout.full_prefix,
                prefix_segments=layout.truncated_prefix_segments,
                feature_conditions=intervention.feature_conditions,
                n_output=len(layout.output),
                n_prompt_surviving=layout.n_prompt_surviving,
                device=t.device("cpu"),
                allowed_ids=allowed_ids_cpu,
                cluster_contract=target_bundle.cluster_contract,
            )
            full_seg_map = seg_map_with_final[:, :-1].contiguous()
            expected_full_length = len(all_tokens) - 1
            if full_seg_map.shape != (1, expected_full_length):
                raise AxBenchBpbAdvError(
                    f"shared-prefix sample {index}: full seg_map shape "
                    f"{tuple(full_seg_map.shape)} != (1, {expected_full_length})"
                )
            prefix_seg_map = full_seg_map[:, :shared_prefix_len].contiguous()
            _validate_seg_map_for_shared_prefix(
                prefix_seg_map,
                n_segments=int(enc_fids.shape[1]),
                surface=f"shared-prefix sample {index} prefill",
            )
            encoder_state = _SharedPrefixEncoderCpuState(
                feature_ids=enc_fids.contiguous(),
                magnitudes=enc_mags.contiguous(),
                mask=enc_mask.contiguous(),
                full_seg_map=full_seg_map,
                allowed_ids=allowed_ids_cpu,
                generation_slot=generation_slot,
            )
            encoder_key = (
                "encdec",
                _cpu_tensor_structural_key(prefix_seg_map),
                _cpu_tensor_structural_key(enc_fids),
                _cpu_tensor_structural_key(enc_mags),
                _cpu_tensor_structural_key(enc_mask),
                generation_slot,
                allowed_ids_key,
                cluster_identity,
            )

        key = (
            shared_prefix_len,
            tuple(layout.full_prefix[:shared_prefix_len]),
            encoder_key,
        )
        group = group_by_key.get(key)
        if group is None:
            group = _SharedPrefixCompatibilityGroup(
                prefix_tokens=tuple(layout.full_prefix[:shared_prefix_len]),
                sample_indices=[],
                encoder=encoder_state,
            )
            group_by_key[key] = group
        group.sample_indices.append(index)
        prepared.append(
            _SharedPrefixPreparedSample(
                layout=layout,
                shared_prefix_len=shared_prefix_len,
                editable_count=editable_count,
                dec_input=dec_input,
                dec_target=dec_target,
                loss_mask=tuple(bool(value) for value in suffix_mask.tolist()),
                full_seg_map=full_seg_map,
                generation_slot=generation_slot,
            )
        )
    return prepared, list(group_by_key.values())


def _decoder_cache_config(target_bundle: TargetModelBundle) -> Any:
    model = target_bundle.model
    config = getattr(model, "dec_config", None)
    if config is None:
        config = getattr(model, "config", None)
    required = ("n_layer", "n_kv_head", "n_head", "n_embd")
    if config is None or any(not hasattr(config, field) for field in required):
        raise AxBenchBpbAdvError(
            "shared-prefix scoring cannot derive decoder KV dimensions from model config"
        )
    return config


def _canonical_shared_prefix_device(
    *,
    requested: t.device,
    model_device: t.device,
) -> t.device:
    """Resolve an index-free device against the model's concrete device.

    PyTorch tensors report an indexed CUDA device (for example ``cuda:0``),
    while the shipped single-GPU config requests bare ``cuda``. Those identify
    the same operational device, but ``torch.device`` compares them as unequal.
    The model tensor is authoritative so every cache and scorer input is placed
    beside the model. Explicit, conflicting device indices still fail loudly.
    """

    if requested.type != model_device.type or (
        requested.index is not None and requested != model_device
    ):
        raise AxBenchBpbAdvError(
            f"model rotary device {model_device} != scoring device {requested}"
        )
    return model_device


def _new_shared_prefix_cache(
    *,
    target_bundle: TargetModelBundle,
    batch_size: int,
    max_seq_len: int,
    device: t.device,
) -> KVCache:
    config = _decoder_cache_config(target_bundle)
    n_head = int(config.n_head)
    n_embd = int(config.n_embd)
    if n_head <= 0 or n_embd <= 0 or n_embd % n_head != 0:
        raise AxBenchBpbAdvError(
            f"invalid decoder cache dimensions n_head={n_head}, n_embd={n_embd}"
        )
    rotary = getattr(target_bundle.model, "cos", None)
    if not isinstance(rotary, t.Tensor):
        raise AxBenchBpbAdvError("shared-prefix scoring requires model.cos rotary cache")
    cache_device = _canonical_shared_prefix_device(
        requested=device,
        model_device=rotary.device,
    )
    cache = KVCache(
        batch_size=batch_size,
        num_heads=int(config.n_kv_head),
        seq_len=max_seq_len,
        head_dim=n_embd // n_head,
        num_layers=int(config.n_layer),
        device=cache_device,
        dtype=rotary.dtype,
    )
    # ``KVCache`` allocates ordinary contiguous storage. Make the materialization
    # contract explicit so a future expand/stride-zero rewrite fails here before
    # suffix attention performs in-place K/V writes.
    if batch_size > 1 and (
        cache.k_cache.stride(1) == 0 or cache.v_cache.stride(1) == 0
    ):
        raise AxBenchBpbAdvError(
            "shared-prefix destination cache batch rows must have independent storage"
        )
    return cache


def _assert_shared_prefix_cache_ready(
    *,
    cache: KVCache,
    expected_position: int,
    suffix_length: int,
    rotary_length: int,
    surface: str,
) -> None:
    position = cache.get_pos()
    if position != expected_position:
        raise AxBenchBpbAdvError(
            f"{surface}: KV cache position {position} != expected {expected_position}"
        )
    expected_positions = t.full_like(cache.cache_seqlens, expected_position)
    if not bool(t.equal(cache.cache_seqlens, expected_positions)):
        raise AxBenchBpbAdvError(
            f"{surface}: KV cache batch positions disagree: "
            f"{cache.cache_seqlens.detach().cpu().tolist()}"
        )
    required = expected_position + suffix_length
    if required > cache.max_seq_len:
        raise AxBenchBpbAdvError(
            f"{surface}: required cache length {required} exceeds "
            f"allocated length {cache.max_seq_len}"
        )
    if required > rotary_length:
        raise AxBenchBpbAdvError(
            f"{surface}: required rotary length {required} exceeds table {rotary_length}"
        )


def _pack_shared_prefix_suffix_microbatch(
    *,
    samples: list[_SharedPrefixPreparedSample],
    suffix_pad_length: int,
    device: t.device,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor | None]:
    batch_size = len(samples)
    if batch_size < 1 or suffix_pad_length < 1:
        raise AxBenchBpbAdvError(
            "shared-prefix suffix packing requires a non-empty batch and pad length"
        )
    dec_input = t.zeros(
        batch_size, suffix_pad_length, dtype=t.long, device=device
    )
    dec_target = t.full(
        (batch_size, suffix_pad_length), -1, dtype=t.long, device=device
    )
    masks = t.zeros(
        batch_size, suffix_pad_length, dtype=t.bool, device=device
    )
    is_encdec = samples[0].full_seg_map is not None
    if any((sample.full_seg_map is not None) != is_encdec for sample in samples):
        raise AxBenchBpbAdvError(
            "shared-prefix suffix batch mixes decoder-only and encoder-decoder samples"
        )
    seg_map_cpu = (
        t.zeros(batch_size, suffix_pad_length, dtype=t.long)
        if is_encdec
        else None
    )
    n_segments: int | None = None
    for row, sample in enumerate(samples):
        length = len(sample.dec_input)
        if length > suffix_pad_length or len(sample.dec_target) != length:
            raise AxBenchBpbAdvError(
                f"shared-prefix suffix row {row}: malformed/pad length "
                f"input={length}, target={len(sample.dec_target)}, "
                f"pad={suffix_pad_length}"
            )
        if len(sample.loss_mask) != length:
            raise AxBenchBpbAdvError(
                f"shared-prefix suffix row {row}: mask length "
                f"{len(sample.loss_mask)} != input length {length}"
            )
        dec_input[row, :length] = t.tensor(
            sample.dec_input, dtype=t.long, device=device
        )
        dec_target[row, :length] = t.tensor(
            sample.dec_target, dtype=t.long, device=device
        )
        masks[row, :length] = t.tensor(
            sample.loss_mask, dtype=t.bool, device=device
        )
        if seg_map_cpu is not None:
            if sample.full_seg_map is None or sample.generation_slot is None:
                raise AxBenchBpbAdvError(
                    f"shared-prefix suffix row {row}: missing encoder routing state"
                )
            suffix_seg_map = sample.full_seg_map[
                :, sample.shared_prefix_len : sample.shared_prefix_len + length
            ]
            if suffix_seg_map.shape != (1, length):
                raise AxBenchBpbAdvError(
                    f"shared-prefix suffix row {row}: seg_map slice shape "
                    f"{tuple(suffix_seg_map.shape)} != (1, {length})"
                )
            seg_map_cpu[row, :length] = suffix_seg_map[0]
            seg_map_cpu[row, length:] = sample.generation_slot
            row_n_segments = int(sample.full_seg_map.max().item()) + 1
            n_segments = (
                row_n_segments if n_segments is None else max(n_segments, row_n_segments)
            )
    if seg_map_cpu is not None:
        assert n_segments is not None
        _validate_seg_map_for_shared_prefix(
            seg_map_cpu,
            n_segments=n_segments,
            surface="shared-prefix suffix microbatch",
        )
        return dec_input, dec_target, masks, seg_map_cpu.to(device)
    return dec_input, dec_target, masks, None


def _copied_kv_bytes(cache: KVCache, *, prefix_len: int) -> int:
    return (
        2
        * cache.n_layers
        * cache.batch_size
        * prefix_len
        * cache.n_heads
        * cache.head_dim
        * cache.k_cache.element_size()
    )


@beartype
def teacher_forced_bpb_loss_shared_prefix_kv_batch(
    *,
    target_bundle: TargetModelBundle,
    prompts: list[list[int]],
    source_prompts: list[list[int]],
    outputs: list[list[int]],
    prefix_context_segments_list: list[tuple[ContextFeatureSegment, ...]],
    editable_spans: list[EditableSpan],
    output_token_bytes: t.Tensor,
    intervention: Intervention,
    device: t.device,
    lora: LoraParams | None = None,
    device_batch_size: PositiveIntLike = 32,
) -> list[float]:
    """Score Faster-GCG candidates with exact shared-prefix KV fan-out.

    Callers must supply the original prompt and final editable span for every
    sample; the function refuses non-final, length-changing, truncated, or
    empty-prefix layouts rather than heuristically falling back. The previous
    full-sequence batch implementation lives only in the test oracle.
    """

    with profile_region(
        "axbench_bpb_adv.candidate_eval.shared_prefix.layout_grouping",
        samples=len(prompts),
    ):
        prepared, groups = _prepare_shared_prefix_groups(
            target_bundle=target_bundle,
            prompts=prompts,
            source_prompts=source_prompts,
            outputs=outputs,
            prefix_context_segments_list=prefix_context_segments_list,
            editable_spans=editable_spans,
            intervention=intervention,
        )
    if not prepared:
        return []

    group_cache_lengths = [
        len(group.prefix_tokens)
        + max(len(prepared[index].dec_input) for index in group.sample_indices)
        for group in groups
    ]
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.compatibility_groups",
        len(groups),
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.prefills",
        len(groups),
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.full_token_positions",
        sum(
            sample.layout.n_full_prefix + len(sample.layout.output) - 1
            for sample in prepared
        ),
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.suffix_token_positions",
        sum(len(sample.dec_input) for sample in prepared),
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.max_source_cache_batch",
        1,
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.max_source_cache_length",
        max(group_cache_lengths),
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.max_destination_cache_batch",
        min(int(device_batch_size), max(len(group.sample_indices) for group in groups)),
    )
    record_counter(
        "axbench_bpb_adv.candidate_eval.shared_prefix.max_destination_cache_length",
        max(group_cache_lengths),
    )

    results: list[float | None] = [None] * len(prepared)
    model = target_bundle.model
    rotary = getattr(model, "cos", None)
    if not isinstance(rotary, t.Tensor):
        raise AxBenchBpbAdvError("shared-prefix scoring requires model.cos rotary cache")
    device = _canonical_shared_prefix_device(
        requested=device,
        model_device=rotary.device,
    )
    rotary_length = int(rotary.size(1))
    was_training = bool(getattr(model, "training", False))
    try:
        if hasattr(model, "eval"):
            model.eval()
        with apply_intervention(model, intervention, lora=lora), t.no_grad():
            for group_index, group in enumerate(groups):
                shared_prefix_len = len(group.prefix_tokens)
                group_samples = [prepared[index] for index in group.sample_indices]
                suffix_pad_length = max(len(sample.dec_input) for sample in group_samples)
                max_cache_length = shared_prefix_len + suffix_pad_length
                if max_cache_length > rotary_length:
                    raise AxBenchBpbAdvError(
                        f"shared-prefix group {group_index}: cache length "
                        f"{max_cache_length} exceeds rotary table {rotary_length}"
                    )

                source_cache = _new_shared_prefix_cache(
                    target_bundle=target_bundle,
                    batch_size=1,
                    max_seq_len=max_cache_length,
                    device=device,
                )
                prefix_input = t.tensor(
                    [group.prefix_tokens], dtype=t.long, device=device
                )
                dummy_targets = t.full_like(prefix_input, -1)
                encoder_out: PackedEncoderOutput | None = None
                prefix_seg_map: t.Tensor | None = None
                if group.encoder is not None:
                    with profile_region(
                        "axbench_bpb_adv.candidate_eval.shared_prefix.encoder_preparation",
                        group_index=group_index,
                    ):
                        enc_feature_ids = group.encoder.feature_ids.to(device)
                        enc_magnitudes = group.encoder.magnitudes.to(device)
                        enc_input_mask = group.encoder.mask.to(device)
                        assert_enc_ids_in_vocab(
                            enc_feature_ids,
                            enc_input_mask,
                            None
                            if group.encoder.allowed_ids is None
                            else group.encoder.allowed_ids.to(device),
                        )
                        if target_bundle.cluster_contract is not None:
                            assert_enc_ids_match_cluster_contract(
                                enc_feature_ids,
                                enc_input_mask,
                                target_bundle.cluster_contract,
                            )
                        target_bundle.ensure_ready_for_forward()
                        encoder_out = model.encoder(
                            enc_feature_ids,
                            enc_magnitudes,
                            enc_input_mask,
                        )
                        prefix_seg_map = group.encoder.full_seg_map[
                            :, :shared_prefix_len
                        ].to(device)

                with profile_region(
                    "axbench_bpb_adv.candidate_eval.shared_prefix.prefill",
                    group_index=group_index,
                    prefix_tokens=shared_prefix_len,
                ):
                    _assert_shared_prefix_cache_ready(
                        cache=source_cache,
                        expected_position=0,
                        suffix_length=shared_prefix_len,
                        rotary_length=rotary_length,
                        surface=f"shared-prefix group {group_index} source prefill",
                    )
                    if group.encoder is None:
                        target_bundle.ensure_ready_for_forward()
                        model(
                            prefix_input,
                            targets=dummy_targets,
                            kv_cache=source_cache,
                            loss_reduction="none",
                        )
                    else:
                        assert encoder_out is not None
                        assert prefix_seg_map is not None
                        target_bundle.ensure_ready_for_forward()
                        model(
                            prefix_input,
                            dummy_targets,
                            seg_map=prefix_seg_map,
                            kv_cache=source_cache,
                            encoder_out=encoder_out,
                            encoder_strict=True,
                            loss_reduction="none",
                        )
                    _assert_shared_prefix_cache_ready(
                        cache=source_cache,
                        expected_position=shared_prefix_len,
                        suffix_length=0,
                        rotary_length=rotary_length,
                        surface=f"shared-prefix group {group_index} populated source",
                    )

                full_destination: KVCache | None = None
                n_group_samples = len(group_samples)
                full_batch_count = n_group_samples // int(device_batch_size)
                if full_batch_count:
                    full_destination = _new_shared_prefix_cache(
                        target_bundle=target_bundle,
                        batch_size=int(device_batch_size),
                        max_seq_len=max_cache_length,
                        device=device,
                    )
                for start in range(0, n_group_samples, int(device_batch_size)):
                    stop = min(start + int(device_batch_size), n_group_samples)
                    chunk = group_samples[start:stop]
                    if len(chunk) == int(device_batch_size):
                        assert full_destination is not None
                        destination = full_destination
                        destination.reset()
                    else:
                        # Drop the reusable full-size allocation before the optional
                        # short-tail cache so peak memory is max(full, tail), not sum.
                        full_destination = None
                        destination = _new_shared_prefix_cache(
                            target_bundle=target_bundle,
                            batch_size=len(chunk),
                            max_seq_len=max_cache_length,
                            device=device,
                        )
                    with profile_region(
                        "axbench_bpb_adv.candidate_eval.shared_prefix.materialized_kv_copy",
                        group_index=group_index,
                        batch_size=len(chunk),
                    ):
                        destination.prefill(source_cache)
                    record_counter(
                        "axbench_bpb_adv.candidate_eval.shared_prefix.copied_kv_bytes",
                        _copied_kv_bytes(destination, prefix_len=shared_prefix_len),
                    )
                    _assert_shared_prefix_cache_ready(
                        cache=destination,
                        expected_position=shared_prefix_len,
                        suffix_length=suffix_pad_length,
                        rotary_length=rotary_length,
                        surface=(
                            f"shared-prefix group {group_index} suffix batch "
                            f"{start // int(device_batch_size)}"
                        ),
                    )
                    dec_input, dec_target, masks, suffix_seg_map = (
                        _pack_shared_prefix_suffix_microbatch(
                            samples=chunk,
                            suffix_pad_length=suffix_pad_length,
                            device=device,
                        )
                    )
                    with profile_region(
                        "axbench_bpb_adv.candidate_eval.shared_prefix.suffix_forward_bpb",
                        group_index=group_index,
                        batch_size=len(chunk),
                        suffix_tokens=suffix_pad_length,
                    ):
                        if group.encoder is None:
                            target_bundle.ensure_ready_for_forward()
                            loss = model(
                                dec_input,
                                targets=dec_target,
                                kv_cache=destination,
                                loss_reduction="none",
                            )
                        else:
                            assert encoder_out is not None
                            assert suffix_seg_map is not None
                            encoder_out_batch = repeat_packed_encoder_output(
                                encoder_out,
                                len(chunk),
                            )
                            target_bundle.ensure_ready_for_forward()
                            loss = model(
                                dec_input,
                                dec_target,
                                seg_map=suffix_seg_map,
                                kv_cache=destination,
                                encoder_out=encoder_out_batch,
                                encoder_strict=True,
                                loss_reduction="none",
                            )
                        loss_batch = loss.view(len(chunk), suffix_pad_length)
                        bpb_tensors = _microbatch_bpb_tensors(
                            loss=loss_batch,
                            masks=masks,
                            layouts=[sample.layout for sample in chunk],
                            output_token_bytes=output_token_bytes,
                            device=device,
                        )
                    for chunk_offset, bpb in enumerate(bpb_tensors):
                        original_index = group.sample_indices[start + chunk_offset]
                        results[original_index] = float(bpb.detach().item())
                    # Drop the loop-local alias. The reusable full cache remains
                    # reachable through ``full_destination``; a short-tail cache
                    # does not overlap the preceding full-size allocation.
                    del destination
                full_destination = None
                del source_cache
        scored: list[float] = []
        for index, value in enumerate(results):
            if value is None:
                raise AxBenchBpbAdvError(
                    f"shared-prefix scorer left sample {index} unscored"
                )
            scored.append(value)
        return scored
    finally:
        if hasattr(model, "train"):
            model.train(was_training)


def teacher_forced_bpb_gradient(
    *,
    target_bundle: TargetModelBundle,
    prompt_tokens: list[int],
    output_tokens: list[int],
    prefix_context_segments: tuple[ContextFeatureSegment, ...],
    output_token_bytes: t.Tensor,
    intervention: Intervention,
    device: t.device,
    lora: LoraParams | None = None,
    embedding_weight: t.Tensor | None = None,
) -> BpbGradientResult:
    """Return BPB and gradient with respect to the original prompt tokens.

    ``embedding_weight`` (OPT #1): the concatenated token/value embedding matrix
    is a STATIC model parameter that does not change during a GCG search (the
    search optimizes input tokens, not weights; the model is in eval mode and
    model parameter gradients are disabled around every call). Rebuilding
    it -- a ~720MB GPU->CPU float32 copy on d12_base -- on every call dominates the
    eval. When the caller passes the cached matrix (identical to what
    ``bpb_surrogate_embedding_weight`` returns), the rebuild is SKIPPED and the
    passed tensor is returned verbatim as ``BpbGradientResult.embedding_weight``.
    This is bit-identical: the embedding-output VJP and the loss are
    computed exactly as before; only the PROVENANCE of the returned static
    embedding matrix changes (cached vs freshly-copied from the same params).
    When ``None`` the original rebuild path is preserved (backward-compatible).
    """

    layout = _build_teacher_forced_layout(
        target_bundle=target_bundle,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=prefix_context_segments,
        intervention=intervention,
    )
    dec_input, dec_target, mask, encoder = _build_teacher_forced_forward_inputs(
        target_bundle=target_bundle,
        layout=layout,
        intervention=intervention,
        device=device,
    )

    owner = _embedding_owner(target_bundle.model)
    captured: dict[str, t.Tensor] = {}

    def _capture(name: str):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output_tensor: Any) -> None:
            if not isinstance(output_tensor, t.Tensor):
                raise AxBenchBpbAdvError(
                    f"embedding hook {name} returned non-Tensor output "
                    f"{type(output_tensor).__name__}"
                )
            if not output_tensor.requires_grad:
                output_tensor.requires_grad_(True)
            captured[name] = output_tensor

        return _hook

    hooks = [owner.transformer.wte.register_forward_hook(_capture("wte"))]
    value_modules = _value_embedding_modules(owner)
    for index, module in enumerate(value_modules):
        hooks.append(module.register_forward_hook(_capture(f"value:{index}")))

    was_training = bool(getattr(target_bundle.model, "training", False))
    try:
        if hasattr(target_bundle.model, "eval"):
            target_bundle.model.eval()
        with apply_intervention(target_bundle.model, intervention, lora=lora):
            with _freeze_model_parameters(target_bundle.model):
                loss = _teacher_forced_model_forward(
                    target_bundle=target_bundle,
                    dec_input=dec_input,
                    dec_target=dec_target,
                    encoder=encoder,
                )
                bpb_tensor = _bpb_tensor_from_loss(
                    loss=loss,
                    mask=mask,
                    layout=layout,
                    output_token_bytes=output_token_bytes,
                    device=device,
                )
                wte_output = captured.get("wte")
                if wte_output is None:
                    raise AxBenchBpbAdvError(
                        "failed to capture gradient from transformer.wte"
                    )
                value_outputs: list[t.Tensor] = []
                for index in range(len(value_modules)):
                    value_output = captured.get(f"value:{index}")
                    if value_output is None:
                        raise AxBenchBpbAdvError(
                            f"failed to capture gradient from value_embeds[{index}]"
                        )
                    value_outputs.append(value_output)
                if not bpb_tensor.requires_grad:
                    raise AxBenchBpbAdvError(
                        "failed to capture gradient from transformer.wte"
                    )
                gradients = t.autograd.grad(
                    bpb_tensor,
                    (wte_output, *value_outputs),
                    allow_unused=True,
                )

        gradient_parts: list[t.Tensor] = []
        # OPT #1: only rebuild the static embedding matrix when the caller did not
        # pass a cached one. The per-module gradient capture below is unaffected.
        rebuild_embedding = embedding_weight is None
        weight_parts: list[t.Tensor] = []
        wte_gradient = gradients[0]
        if wte_gradient is None:
            raise AxBenchBpbAdvError("failed to capture gradient from transformer.wte")
        gradient_parts.append(
            _prompt_gradient_from_full_gradient(
                full_gradient=wte_gradient.detach()[0].float().cpu(),
                prompt_len=len(layout.prompt),
                prompt_prefix_len=len(layout.prompt_prefix_tokens),
                tokens_dropped_total=layout.tokens_dropped_total,
            )
        )
        if rebuild_embedding:
            weight_parts.append(owner.transformer.wte.weight.detach().float().cpu())
        for index, (module, value_gradient) in enumerate(
            zip(value_modules, gradients[1:], strict=True)
        ):
            if value_gradient is None:
                raise AxBenchBpbAdvError(
                    f"failed to capture gradient from value_embeds[{index}]"
                )
            gradient_parts.append(
                _prompt_gradient_from_full_gradient(
                    full_gradient=value_gradient.detach()[0].float().cpu(),
                    prompt_len=len(layout.prompt),
                    prompt_prefix_len=len(layout.prompt_prefix_tokens),
                    tokens_dropped_total=layout.tokens_dropped_total,
                )
            )
            if rebuild_embedding:
                weight_parts.append(module.weight.detach().float().cpu())

        token_gradient = t.cat(gradient_parts, dim=1)
        # Clamp non-finite gradient entries to 0 so the Faster-GCG ranking/sampling
        # degrades gracefully instead of crashing on NaN weights. Unlike the
        # finite-LOSS guard (which raises), a non-finite gradient with a finite
        # loss is recoverable -- but it must NOT be silent: log it so a persistent
        # upstream numerical blowup is observable rather than masked.
        n_non_finite = int((~t.isfinite(token_gradient)).sum().item())
        if n_non_finite:
            logger.warning(
                f"teacher_forced_bpb_gradient: clamped {n_non_finite} non-finite "
                "token-gradient entries to 0; the GCG search continues, but a "
                "persistent/large count signals an upstream numerical blowup the "
                "finite-loss guard does not catch."
            )
        token_gradient = t.nan_to_num(
            token_gradient, nan=0.0, posinf=0.0, neginf=0.0
        )
        result_embedding_weight = (
            embedding_weight
            if embedding_weight is not None
            else _concatenate_embedding_parts(weight_parts)
        )
        if token_gradient.shape[1] != result_embedding_weight.shape[1]:
            raise AxBenchBpbAdvError(
                "gradient/embedding dimension mismatch: "
                f"{token_gradient.shape[1]} vs {result_embedding_weight.shape[1]}"
            )
        return BpbGradientResult(
            loss=float(bpb_tensor.detach().item()),
            token_gradient=token_gradient,
            embedding_weight=result_embedding_weight,
        )
    finally:
        for hook in hooks:
            hook.remove()
        if hasattr(target_bundle.model, "train"):
            target_bundle.model.train(was_training)


def _decoder_microbatch_gradient(
    *,
    layouts: list[_TeacherForcedLayout],
    target_bundle: TargetModelBundle,
    output_token_bytes: t.Tensor,
    device: t.device,
    t_pad: int,
    embedding_weight: t.Tensor | None,
) -> list[BpbGradientResult]:
    """Per-sample BPB + prompt-token gradient for ONE decoder microbatch.

    Intervention + ``eval()`` and parameter freezing are applied by the caller.
    Runs ONE padded forward, sums the per-sample BPB into a single scalar, and
    evaluates one embedding-only VJP. The samples
    are independent (right-padding + causal attention), so
    ``d(sum_j BPB_j)/d(emb_output_i) == d(BPB_i)/d(emb_output_i)`` -- the captured
    ``(B, seq, D)`` embedding-output gradient sliced at row ``i`` is byte-identical
    (exact model) to the single-prompt :func:`teacher_forced_bpb_gradient`'s
    ``[0]`` slice. One forward + VJP over B samples replaces B single-prompt
    forward + VJP evaluations.

    Mirrors the single path exactly: same hooks (capture wte + value-embed outputs
    as explicit graph roots), same OPT#1 cached embedding, same per-prompt slicing,
    same non-finite clamp + warning. The caller owns the parameter-freeze context."""

    dec_input, dec_target, masks = _pack_decoder_microbatch(
        layouts=layouts, device=device, t_pad=t_pad
    )
    b = len(layouts)
    owner = _embedding_owner(target_bundle.model)
    captured: dict[str, t.Tensor] = {}

    def _capture(name: str):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output_tensor: Any) -> None:
            if not isinstance(output_tensor, t.Tensor):
                raise AxBenchBpbAdvError(
                    f"embedding hook {name} returned non-Tensor output "
                    f"{type(output_tensor).__name__}"
                )
            if not output_tensor.requires_grad:
                output_tensor.requires_grad_(True)
            captured[name] = output_tensor

        return _hook

    hooks = [owner.transformer.wte.register_forward_hook(_capture("wte"))]
    value_modules = _value_embedding_modules(owner)
    for index, module in enumerate(value_modules):
        hooks.append(module.register_forward_hook(_capture(f"value:{index}")))

    try:
        target_bundle.ensure_ready_for_forward()
        loss = target_bundle.model(
            dec_input, targets=dec_target, loss_reduction="none"
        ).view(b, t_pad - 1)
        bpb_tensors = _microbatch_bpb_tensors(
            loss=loss,
            masks=masks,
            layouts=layouts,
            output_token_bytes=output_token_bytes,
            device=device,
        )
        bpb_sum = t.stack(bpb_tensors).sum()

        rebuild_embedding = embedding_weight is None
        wte_output = captured.get("wte")
        if wte_output is None:
            raise AxBenchBpbAdvError("failed to capture gradient from transformer.wte")
        value_outputs: list[t.Tensor] = []
        for index in range(len(value_modules)):
            value_output = captured.get(f"value:{index}")
            if value_output is None:
                raise AxBenchBpbAdvError(
                    f"failed to capture gradient from value_embeds[{index}]"
                )
            value_outputs.append(value_output)
        if not bpb_sum.requires_grad:
            raise AxBenchBpbAdvError("failed to capture gradient from transformer.wte")
        gradients = t.autograd.grad(
            bpb_sum,
            (wte_output, *value_outputs),
            allow_unused=True,
        )
        wte_gradient = gradients[0]
        if wte_gradient is None:
            raise AxBenchBpbAdvError("failed to capture gradient from transformer.wte")
        wte_grad = wte_gradient.detach()
        value_grads: list[t.Tensor] = []
        weight_parts: list[t.Tensor] = []
        if rebuild_embedding:
            weight_parts.append(owner.transformer.wte.weight.detach().float().cpu())
        for index, (module, value_gradient) in enumerate(
            zip(value_modules, gradients[1:], strict=True)
        ):
            if value_gradient is None:
                raise AxBenchBpbAdvError(
                    f"failed to capture gradient from value_embeds[{index}]"
                )
            value_grads.append(value_gradient.detach())
            if rebuild_embedding:
                weight_parts.append(module.weight.detach().float().cpu())
        result_embedding_weight = (
            embedding_weight
            if embedding_weight is not None
            else _concatenate_embedding_parts(weight_parts)
        )

        results: list[BpbGradientResult] = []
        for i, layout in enumerate(layouts):
            gradient_parts = [
                _prompt_gradient_from_full_gradient(
                    full_gradient=wte_grad[i].float().cpu(),
                    prompt_len=len(layout.prompt),
                    prompt_prefix_len=len(layout.prompt_prefix_tokens),
                    tokens_dropped_total=layout.tokens_dropped_total,
                )
            ]
            gradient_parts.extend(
                _prompt_gradient_from_full_gradient(
                    full_gradient=value_grad[i].float().cpu(),
                    prompt_len=len(layout.prompt),
                    prompt_prefix_len=len(layout.prompt_prefix_tokens),
                    tokens_dropped_total=layout.tokens_dropped_total,
                )
                for value_grad in value_grads
            )
            token_gradient = t.cat(gradient_parts, dim=1)
            n_non_finite = int((~t.isfinite(token_gradient)).sum().item())
            if n_non_finite:
                logger.warning(
                    f"teacher_forced_bpb_gradient_batch: clamped {n_non_finite} "
                    "non-finite token-gradient entries to 0; the GCG search "
                    "continues, but a persistent/large count signals an upstream "
                    "numerical blowup the finite-loss guard does not catch."
                )
            token_gradient = t.nan_to_num(
                token_gradient, nan=0.0, posinf=0.0, neginf=0.0
            )
            if token_gradient.shape[1] != result_embedding_weight.shape[1]:
                raise AxBenchBpbAdvError(
                    "gradient/embedding dimension mismatch: "
                    f"{token_gradient.shape[1]} vs {result_embedding_weight.shape[1]}"
                )
            results.append(
                BpbGradientResult(
                    loss=float(bpb_tensors[i].detach().item()),
                    token_gradient=token_gradient,
                    embedding_weight=result_embedding_weight,
                )
            )
        return results
    finally:
        for hook in hooks:
            hook.remove()


@beartype
def teacher_forced_bpb_gradient_batch(
    *,
    target_bundle: TargetModelBundle,
    prompts: list[list[int]],
    outputs: list[list[int]],
    prefix_context_segments_list: list[tuple[ContextFeatureSegment, ...]],
    output_token_bytes: t.Tensor,
    intervention: Intervention,
    device: t.device,
    lora: LoraParams | None = None,
    embedding_weight: t.Tensor | None = None,
    device_batch_size: PositiveIntLike = 8,
) -> list[BpbGradientResult]:
    """BATCHED teacher-forced BPB + prompt-token gradient for N samples.

    The universal GCG step computes one prompt-token gradient per minibatch pair
    (rich AND avoid completions = 2 per pair), and at single-prompt granularity
    that is the dominant per-step cost (each is a grad-enabled forward plus an
    embedding-only VJP, serialized, under eager flex-attention). This pads the
    decoder samples into ``(B, t_pad-1)`` forwards of up to
    ``device_batch_size`` and runs ONE forward + VJP per microbatch, returning the
    SAME per-sample :class:`BpbGradientResult` the single-prompt
    :func:`teacher_forced_bpb_gradient` returns -- byte-identical on an exact
    model (the samples are independent under right-padding + causal attention),
    up to bf16 batch-shape rounding on the real model (the same accepted cost as
    the batched loss path).

    Unlike the loss batch this runs WITH autograd (it must evaluate a VJP), so the
    live graph is bounded by ``device_batch_size`` rather than
    ``torch.no_grad()``; keep it small (the graph for B samples is held until the
    per-microbatch VJP).
    The same intervention/LoRA is applied ONCE around the whole chunked loop. The
    encdec path loops the single-sample primitive (its encoder tensors are
    per-sample; encdec faster_gcg is deferred)."""

    n = len(prompts)
    if not (n == len(outputs) == len(prefix_context_segments_list)):
        raise AxBenchBpbAdvError(
            "teacher_forced_bpb_gradient_batch: prompts/outputs/segments length mismatch"
        )
    if n == 0:
        return []
    if target_bundle.target_name == "encdec":
        return [
            teacher_forced_bpb_gradient(
                target_bundle=target_bundle,
                prompt_tokens=prompts[i],
                output_tokens=outputs[i],
                prefix_context_segments=prefix_context_segments_list[i],
                output_token_bytes=output_token_bytes,
                intervention=intervention,
                device=device,
                lora=lora,
                embedding_weight=embedding_weight,
            )
            for i in range(n)
        ]

    layouts = [
        _build_teacher_forced_layout(
            target_bundle=target_bundle,
            prompt_tokens=prompts[i],
            output_tokens=outputs[i],
            prefix_context_segments=prefix_context_segments_list[i],
            intervention=intervention,
        )
        for i in range(n)
    ]
    t_pad = max(len(lay.full_prefix) + len(lay.output) for lay in layouts)
    results: list[BpbGradientResult] = []
    was_training = bool(getattr(target_bundle.model, "training", False))
    try:
        if hasattr(target_bundle.model, "eval"):
            target_bundle.model.eval()
        with apply_intervention(target_bundle.model, intervention, lora=lora):
            with _freeze_model_parameters(target_bundle.model):
                for start in range(0, n, device_batch_size):
                    results.extend(
                        _decoder_microbatch_gradient(
                            layouts=layouts[start : start + device_batch_size],
                            target_bundle=target_bundle,
                            output_token_bytes=output_token_bytes,
                            device=device,
                            t_pad=t_pad,
                            embedding_weight=embedding_weight,
                        )
                    )
        return results
    finally:
        if hasattr(target_bundle.model, "train"):
            target_bundle.model.train(was_training)
