"""Test-only pre-#1226 activation-``.grad`` surrogate oracle.

Production must not import this module. It deliberately retains the previous
full-parameter ``backward()`` behavior so the embedding-only VJP can be checked
against an independent extraction path.
"""

from __future__ import annotations

from typing import Any

import torch as t

from steering.eval.axbench_bpb.intervention import apply_intervention
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb_adv import surrogate
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.surrogate import BpbGradientResult
from steering.eval.target_conditioning import ContextFeatureSegment


def _snapshot_and_clear_parameter_grads(
    model: Any,
) -> dict[t.nn.Parameter, t.Tensor | None]:
    snapshot: dict[t.nn.Parameter, t.Tensor | None] = {}
    for parameter in model.parameters():
        snapshot[parameter] = (
            None if parameter.grad is None else parameter.grad.detach().clone()
        )
        parameter.grad = None
    return snapshot


def _restore_parameter_grads(
    snapshot: dict[t.nn.Parameter, t.Tensor | None],
) -> None:
    for parameter, gradient in snapshot.items():
        parameter.grad = gradient


def _embedding_hooks(
    model: Any,
) -> tuple[Any, list[Any], dict[str, t.Tensor], list[Any]]:
    owner = surrogate._embedding_owner(model)
    value_modules = surrogate._value_embedding_modules(owner)
    captured: dict[str, t.Tensor] = {}

    def _capture(name: str):
        def _hook(_module: Any, _inputs: tuple[Any, ...], output: Any) -> None:
            if not isinstance(output, t.Tensor):
                raise AxBenchBpbAdvError(
                    f"embedding hook {name} returned non-Tensor output "
                    f"{type(output).__name__}"
                )
            output.retain_grad()
            captured[name] = output

        return _hook

    hooks = [owner.transformer.wte.register_forward_hook(_capture("wte"))]
    for index, module in enumerate(value_modules):
        hooks.append(module.register_forward_hook(_capture(f"value:{index}")))
    return owner, value_modules, captured, hooks


def _assemble_result(
    *,
    layout: surrogate._TeacherForcedLayout,
    bpb_tensor: t.Tensor,
    owner: Any,
    value_modules: list[Any],
    captured: dict[str, t.Tensor],
    embedding_weight: t.Tensor | None,
    batch_index: int,
) -> BpbGradientResult:
    wte_output = captured.get("wte")
    if wte_output is None or wte_output.grad is None:
        raise AxBenchBpbAdvError("failed to capture gradient from transformer.wte")
    gradient_parts = [
        surrogate._prompt_gradient_from_full_gradient(
            full_gradient=wte_output.grad.detach()[batch_index].float().cpu(),
            prompt_len=len(layout.prompt),
            prompt_prefix_len=len(layout.prompt_prefix_tokens),
            tokens_dropped_total=layout.tokens_dropped_total,
        )
    ]
    rebuild_embedding = embedding_weight is None
    weight_parts = (
        [owner.transformer.wte.weight.detach().float().cpu()]
        if rebuild_embedding
        else []
    )
    for index, module in enumerate(value_modules):
        value_output = captured.get(f"value:{index}")
        if value_output is None or value_output.grad is None:
            raise AxBenchBpbAdvError(
                f"failed to capture gradient from value_embeds[{index}]"
            )
        gradient_parts.append(
            surrogate._prompt_gradient_from_full_gradient(
                full_gradient=value_output.grad.detach()[batch_index].float().cpu(),
                prompt_len=len(layout.prompt),
                prompt_prefix_len=len(layout.prompt_prefix_tokens),
                tokens_dropped_total=layout.tokens_dropped_total,
            )
        )
        if rebuild_embedding:
            weight_parts.append(module.weight.detach().float().cpu())

    token_gradient = t.nan_to_num(
        t.cat(gradient_parts, dim=1), nan=0.0, posinf=0.0, neginf=0.0
    )
    result_embedding_weight = (
        embedding_weight
        if embedding_weight is not None
        else surrogate._concatenate_embedding_parts(weight_parts)
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


def legacy_teacher_forced_bpb_gradient(
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
    """Run the former single-sample full-backward implementation."""

    layout = surrogate._build_teacher_forced_layout(
        target_bundle=target_bundle,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        prefix_context_segments=prefix_context_segments,
        intervention=intervention,
    )
    dec_input, dec_target, mask, encoder = (
        surrogate._build_teacher_forced_forward_inputs(
            target_bundle=target_bundle,
            layout=layout,
            intervention=intervention,
            device=device,
        )
    )
    owner, value_modules, captured, hooks = _embedding_hooks(target_bundle.model)
    snapshot = _snapshot_and_clear_parameter_grads(target_bundle.model)
    was_training = bool(getattr(target_bundle.model, "training", False))
    try:
        if hasattr(target_bundle.model, "eval"):
            target_bundle.model.eval()
        with apply_intervention(target_bundle.model, intervention, lora=lora):
            loss = surrogate._teacher_forced_model_forward(
                target_bundle=target_bundle,
                dec_input=dec_input,
                dec_target=dec_target,
                encoder=encoder,
            )
            bpb_tensor = surrogate._bpb_tensor_from_loss(
                loss=loss,
                mask=mask,
                layout=layout,
                output_token_bytes=output_token_bytes,
                device=device,
            )
            bpb_tensor.backward()
        return _assemble_result(
            layout=layout,
            bpb_tensor=bpb_tensor,
            owner=owner,
            value_modules=value_modules,
            captured=captured,
            embedding_weight=embedding_weight,
            batch_index=0,
        )
    finally:
        for hook in hooks:
            hook.remove()
        _restore_parameter_grads(snapshot)
        if hasattr(target_bundle.model, "train"):
            target_bundle.model.train(was_training)


def _legacy_decoder_microbatch_gradient(
    *,
    layouts: list[surrogate._TeacherForcedLayout],
    target_bundle: TargetModelBundle,
    output_token_bytes: t.Tensor,
    device: t.device,
    t_pad: int,
    embedding_weight: t.Tensor | None,
) -> list[BpbGradientResult]:
    dec_input, dec_target, masks = surrogate._pack_decoder_microbatch(
        layouts=layouts,
        device=device,
        t_pad=t_pad,
    )
    owner, value_modules, captured, hooks = _embedding_hooks(target_bundle.model)
    snapshot = _snapshot_and_clear_parameter_grads(target_bundle.model)
    try:
        target_bundle.ensure_ready_for_forward()
        loss = target_bundle.model(
            dec_input,
            targets=dec_target,
            loss_reduction="none",
        ).view(len(layouts), t_pad - 1)
        bpb_tensors = surrogate._microbatch_bpb_tensors(
            loss=loss,
            masks=masks,
            layouts=layouts,
            output_token_bytes=output_token_bytes,
            device=device,
        )
        t.stack(bpb_tensors).sum().backward()
        return [
            _assemble_result(
                layout=layout,
                bpb_tensor=bpb_tensors[index],
                owner=owner,
                value_modules=value_modules,
                captured=captured,
                embedding_weight=embedding_weight,
                batch_index=index,
            )
            for index, layout in enumerate(layouts)
        ]
    finally:
        for hook in hooks:
            hook.remove()
        _restore_parameter_grads(snapshot)


def legacy_teacher_forced_bpb_gradient_batch(
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
    device_batch_size: int = 8,
) -> list[BpbGradientResult]:
    """Run the former public batched full-backward implementation."""

    if target_bundle.target_name == "encdec":
        return [
            legacy_teacher_forced_bpb_gradient(
                target_bundle=target_bundle,
                prompt_tokens=prompts[index],
                output_tokens=outputs[index],
                prefix_context_segments=prefix_context_segments_list[index],
                output_token_bytes=output_token_bytes,
                intervention=intervention,
                device=device,
                lora=lora,
                embedding_weight=embedding_weight,
            )
            for index in range(len(prompts))
        ]

    layouts = [
        surrogate._build_teacher_forced_layout(
            target_bundle=target_bundle,
            prompt_tokens=prompts[index],
            output_tokens=outputs[index],
            prefix_context_segments=prefix_context_segments_list[index],
            intervention=intervention,
        )
        for index in range(len(prompts))
    ]
    if not layouts:
        return []
    t_pad = max(len(layout.full_prefix) + len(layout.output) for layout in layouts)
    results: list[BpbGradientResult] = []
    was_training = bool(getattr(target_bundle.model, "training", False))
    try:
        if hasattr(target_bundle.model, "eval"):
            target_bundle.model.eval()
        with apply_intervention(target_bundle.model, intervention, lora=lora):
            for start in range(0, len(layouts), device_batch_size):
                results.extend(
                    _legacy_decoder_microbatch_gradient(
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
