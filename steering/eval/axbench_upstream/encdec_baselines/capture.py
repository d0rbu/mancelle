"""Real-conditioned residual capture and loss forwards for encdec baselines."""

from __future__ import annotations

from collections.abc import Sequence
from contextlib import ExitStack
from dataclasses import dataclass
from typing import Any, Literal

import torch as t
import torch.nn as nn
from phantom.sized import NonEmptyStr

from steering.conditional.encoder_decoder.eval_adapter import (
    build_segmented_feature_encoder_inputs,
)
from steering.conditional.encoder_decoder.eval_residual import (
    scoped_post_cross_attention_callback,
)
from steering.conditional.encoder_decoder.full_sequence_routing import (
    context_token_assignments,
)
from steering.eval.axbench_base.baselines.hooks import _get_block
from steering.eval.axbench_upstream.methods import (
    MethodSetup,
    build_extracted_context_entry,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.segmentation import (
    align_segments_to_decoded_prefix,
)

_FIT_ALIGNMENT_COUNTER: NonEmptyStr = NonEmptyStr.parse(
    "segmentation.char_end_extended.encdec_baseline_fit"
)


@dataclass(frozen=True)
class FitRowConditioning:
    """Exact fit tokens, aligned real segments, and their token routing."""

    token_ids: tuple[int, ...]
    segments: tuple[ContextFeatureSegment, ...]
    segment_assignments: tuple[int, ...]


def build_fit_row_conditioning(
    setup: MethodSetup,
    extraction_model: Any,
    sae: Any,
    nanochat_tokenizer: Any,
    *,
    token_ids: Sequence[int],
    surface: str,
) -> FitRowConditioning:
    """Extract, align, and route one exact full rendered fit row in memory."""

    exact_tokens = tuple(int(token_id) for token_id in token_ids)
    entry = build_extracted_context_entry(
        setup,
        extraction_model,
        sae,
        nanochat_tokenizer,
        token_ids=exact_tokens,
        source=f"{surface}: fit row",
    )
    if not entry.segments:
        raise ValueError(f"{surface}: fit row has no valid real feature segments")

    aligned_segments = align_segments_to_decoded_prefix(
        tokenizer=nanochat_tokenizer,
        prefix_tokens=list(exact_tokens),
        segments=entry.segments,
        counter_name=_FIT_ALIGNMENT_COUNTER,
    )
    assignments = context_token_assignments(
        segment_mode=setup.segment_mode,
        tokenizer=nanochat_tokenizer,
        context_tokens=list(exact_tokens),
        context_segments=aligned_segments,
        token_assignments=entry.token_assignments,
    )
    return FitRowConditioning(
        token_ids=exact_tokens,
        segments=aligned_segments,
        segment_assignments=tuple(int(value) for value in assignments.tolist()),
    )


def conditioned_forward_inputs(
    row: FitRowConditioning,
    device: t.device | str,
) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
    """Build raw encoder tensors and the aligned ``(1, T)`` segment map."""

    if not row.token_ids:
        raise ValueError("AxBench encdec-baseline fit row token tuple must be nonempty")
    if not row.segments:
        raise ValueError("AxBench encdec-baseline fit row has no real feature segments")
    if len(row.segment_assignments) != len(row.token_ids):
        raise ValueError(
            "AxBench encdec-baseline fit row assignment count must match token count: "
            f"{len(row.segment_assignments)} != {len(row.token_ids)}"
        )
    for segment_index, segment in enumerate(row.segments):
        if len(segment.feature_ids) != len(segment.buckets):
            raise ValueError(
                "AxBench encdec-baseline fit row segment feature/bucket length "
                f"mismatch at segment {segment_index}: "
                f"{len(segment.feature_ids)} != {len(segment.buckets)}"
            )

    segment_conditions: list[dict[int, int] | None] = [
        {
            int(feature_id): int(bucket)
            for feature_id, bucket in zip(
                segment.feature_ids,
                segment.buckets,
                strict=True,
            )
        }
        for segment in row.segments
    ]
    enc_feature_ids, enc_magnitudes, enc_mask = build_segmented_feature_encoder_inputs(
        segment_conditions,
        device,
    )
    seg_map = t.tensor(
        [row.segment_assignments],
        dtype=t.int64,
        device=device,
    )
    if int(seg_map.min().item()) < 0 or int(seg_map.max().item()) >= len(row.segments):
        raise ValueError(
            "AxBench encdec-baseline fit row segment assignment is outside the "
            f"real segment range [0, {len(row.segments)})"
        )
    return enc_feature_ids, enc_magnitudes, enc_mask, seg_map


def capture_post_prefix_activations(
    model: Any,
    row: FitRowConditioning,
    *,
    layer: int,
    prefix_length: int,
    capture_site: Literal["block_output", "post_cross_attention"] = "block_output",
) -> t.Tensor:
    """Capture one decoder residual site after the excluded prefix as fp32.

    ``block_output`` is the historical pre-cross-attention default. The
    explicit ``post_cross_attention`` option uses the project-owned model seam
    and cannot silently change existing upstream fit artifacts.
    """

    if prefix_length < 0 or prefix_length > len(row.token_ids):
        raise ValueError(
            "AxBench encdec-baseline fit row prefix length must be within the "
            f"token sequence: {prefix_length} not in [0, {len(row.token_ids)}]"
        )
    if capture_site not in {"block_output", "post_cross_attention"}:
        raise ValueError(f"unknown encoder-decoder capture site {capture_site!r}")
    if type(layer) is not int or not 0 <= layer < int(model.dec_config.n_layer):
        raise ValueError(
            f"capture layer {layer!r} is outside model with {model.dec_config.n_layer} layers"
        )
    device = next(model.parameters()).device
    enc_feature_ids, enc_magnitudes, enc_mask, seg_map = conditioned_forward_inputs(
        row,
        device,
    )
    idx = t.tensor([row.token_ids], dtype=t.int64, device=device)
    captured: list[t.Tensor] = []

    def capture_hook(
        _module: nn.Module,
        _inputs: tuple[Any, ...],
        output: t.Tensor,
    ) -> None:
        # Verified in nanochat/gpt.py: Block.forward returns the residual tensor,
        # not a tuple; this is the same output contract used by residual hooks.
        captured.append(output)

    def capture_post_cross_attention(layer_index: int, output: t.Tensor) -> t.Tensor:
        if layer_index == layer:
            captured.append(output)
        return output

    with ExitStack() as stack:
        if capture_site == "block_output":
            handle = _get_block(model, layer).register_forward_hook(capture_hook)
            stack.callback(handle.remove)
        else:
            stack.enter_context(
                scoped_post_cross_attention_callback(
                    model,
                    capture_post_cross_attention,
                )
            )
        # The conditional model owns its compute dtype; no autocast is applied.
        with t.no_grad():
            model(
                idx,
                None,
                enc_feature_ids=enc_feature_ids,
                enc_magnitudes=enc_magnitudes,
                enc_mask=enc_mask,
                seg_map=seg_map,
                encoder_strict=True,
            )
    if len(captured) != 1:
        raise RuntimeError(
            "AxBench encdec-baseline fit row capture expected exactly one block "
            f"output, got {len(captured)}"
        )
    return captured[0][0, prefix_length:, :].detach().to(t.float32)


def conditioned_loss_forward(
    model: Any,
    row: FitRowConditioning,
    *,
    targets: t.Tensor,
    loss_reduction: str,
) -> t.Tensor:
    """Run a differentiable teacher-forced loss under real row conditioning."""

    expected_shape = (1, len(row.token_ids))
    if tuple(targets.shape) != expected_shape:
        raise ValueError(
            "AxBench encdec-baseline fit row targets must have shape "
            f"{expected_shape}, got {tuple(targets.shape)}"
        )
    if targets.dtype != t.int64:
        raise ValueError(
            f"AxBench encdec-baseline fit row targets must have dtype int64, got {targets.dtype}"
        )
    device = next(model.parameters()).device
    enc_feature_ids, enc_magnitudes, enc_mask, seg_map = conditioned_forward_inputs(
        row,
        device,
    )
    idx = t.tensor([row.token_ids], dtype=t.int64, device=device)
    loss = model(
        idx,
        targets.to(device),
        enc_feature_ids=enc_feature_ids,
        enc_magnitudes=enc_magnitudes,
        enc_mask=enc_mask,
        seg_map=seg_map,
        loss_reduction=loss_reduction,
        encoder_strict=True,
    )
    if not isinstance(loss, t.Tensor):
        raise TypeError(
            "AxBench encdec-baseline fit row loss forward must return a tensor, "
            f"got {type(loss).__name__}"
        )
    return loss
