"""Test-only full-sequence oracle for shared-prefix candidate scoring.

Production Faster-GCG and GEPA have one candidate-loss implementation:
shared-prefix KV reuse. This module retains the previous padded full-sequence
algorithm only for numerical parity and non-regression tests.
"""

from __future__ import annotations

import torch as t
from beartype import beartype

from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.surrogate import (
    _bpb_tensor_from_loss,
    _build_teacher_forced_forward_inputs,
    _build_teacher_forced_layout,
    _microbatch_bpb_tensors,
    _pack_decoder_microbatch,
    _teacher_forced_model_forward,
    _TeacherForcedLayout,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveIntLike


def teacher_forced_bpb_loss(
    *,
    target_bundle: TargetModelBundle,
    prompt_tokens: list[int],
    output_tokens: list[int],
    prefix_context_segments: tuple[ContextFeatureSegment, ...],
    output_token_bytes: t.Tensor,
    intervention: Intervention,
    device: t.device,
    lora: LoraParams | None = None,
) -> float:
    """Run the retired single-prompt loss path as a parity oracle."""
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

    was_training = bool(getattr(target_bundle.model, "training", False))
    try:
        if hasattr(target_bundle.model, "eval"):
            target_bundle.model.eval()
        with apply_intervention(target_bundle.model, intervention, lora=lora):
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
        return float(bpb_tensor.detach().item())
    finally:
        if hasattr(target_bundle.model, "train"):
            target_bundle.model.train(was_training)


def _decoder_microbatch_bpb_reference(
    *,
    layouts: list[_TeacherForcedLayout],
    target_bundle: TargetModelBundle,
    output_token_bytes: t.Tensor,
    device: t.device,
    t_pad: int,
) -> list[float]:
    dec_input, dec_target, masks = _pack_decoder_microbatch(
        layouts=layouts, device=device, t_pad=t_pad
    )
    target_bundle.ensure_ready_for_forward()
    loss = target_bundle.model(
        dec_input, targets=dec_target, loss_reduction="none"
    ).view(len(layouts), t_pad - 1)
    bpb_tensors = _microbatch_bpb_tensors(
        loss=loss,
        masks=masks,
        layouts=layouts,
        output_token_bytes=output_token_bytes,
        device=device,
    )
    return [float(bpb.detach().item()) for bpb in bpb_tensors]


@beartype
def teacher_forced_bpb_loss_full_sequence_reference(
    *,
    target_bundle: TargetModelBundle,
    prompts: list[list[int]],
    outputs: list[list[int]],
    prefix_context_segments_list: list[tuple[ContextFeatureSegment, ...]],
    output_token_bytes: t.Tensor,
    intervention: Intervention,
    device: t.device,
    lora: LoraParams | None = None,
    device_batch_size: PositiveIntLike = 32,
) -> list[float]:
    """Previous production algorithm, retained only as a test oracle."""

    n = len(prompts)
    if not (n == len(outputs) == len(prefix_context_segments_list)):
        raise AxBenchBpbAdvError(
            "full-sequence reference: prompts/outputs/segments length mismatch"
        )
    if n == 0:
        return []
    if target_bundle.target_name == "encdec":
        return [
            teacher_forced_bpb_loss(
                target_bundle=target_bundle,
                prompt_tokens=prompts[i],
                output_tokens=outputs[i],
                prefix_context_segments=prefix_context_segments_list[i],
                output_token_bytes=output_token_bytes,
                intervention=intervention,
                device=device,
                lora=lora,
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
    t_pad = max(len(layout.full_prefix) + len(layout.output) for layout in layouts)
    results: list[float] = []
    was_training = bool(getattr(target_bundle.model, "training", False))
    try:
        if hasattr(target_bundle.model, "eval"):
            target_bundle.model.eval()
        with apply_intervention(target_bundle.model, intervention, lora=lora):
            with t.no_grad():
                for start in range(0, n, device_batch_size):
                    results.extend(
                        _decoder_microbatch_bpb_reference(
                            layouts=layouts[start : start + device_batch_size],
                            target_bundle=target_bundle,
                            output_token_bytes=output_token_bytes,
                            device=device,
                            t_pad=t_pad,
                        )
                    )
        return results
    finally:
        if hasattr(target_bundle.model, "train"):
            target_bundle.model.train(was_training)
