"""LoRA adapter loader for axbench_base generation (Task 25, Stage 3b).

Complements Task 24's training loop. The training loop writes a
per-concept safetensors file at
``AxBenchBasePaths.lora_adapter(mode, concept_id)`` containing one
``lora_A`` / ``lora_B`` tensor per
attached nanochat sublayer. At evaluation time we need to load that
file back and wire it onto a fresh ``GPT`` so the generation loop
produces adapted outputs; when generation is done we restore the
original ``Linear`` modules so the next concept can be evaluated on a
clean model.

This module exposes exactly one public helper,
:func:`attach_lora_from_disk`, which returns the attached adapters and
an idempotent ``detach_fn`` that the caller invokes when it's done
with this concept.

Keying convention
-----------------
Weights are stored under keys of the form::

    layer_{layer}_{parent}_{child}_lora_A
    layer_{layer}_{parent}_{child}_lora_B

where ``(parent, child)`` ranges over
:data:`steering.eval.axbench_base.baselines.lora._attach._ATTACH_PATHS`.
This matches ``training._save_lora_adapter``.

Error handling
--------------
  * Nonexistent file → :class:`FileNotFoundError` (from
    ``safetensors.torch.load_file``).
  * Malformed / non-safetensors file → wrapped in an
    :class:`AxBenchBaseError` so callers in the pipeline can catch a
    single error type.
  * Missing required tensor key → :class:`AxBenchBaseError` with a
    clear "Missing adapter key: ..." message.

On any error raised AFTER attach has begun, we detach before
re-raising so the caller's model is not left half-adapted.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import torch as t
import torch.nn as nn
from safetensors.torch import load_file

from steering.eval.axbench_base.baselines.lora._attach import (
    _attach_lora_to_model,
    _collect_attached_adapters,
    _detach_lora_from_model,
)
from steering.eval.axbench_base.baselines.lora.module import LoRAAdapter
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_base.types import AxBenchBaseError

__all__ = ("attach_lora_from_disk",)


def _load_adapter_state(adapter_path: Path) -> dict[str, t.Tensor]:
    """Load a LoRA adapter safetensors file, surfacing errors clearly.

    ``safetensors.torch.load_file`` raises a ``safetensors_rust.SafetensorError``
    on malformed input. We wrap that in an :class:`AxBenchBaseError`
    so upstream code can catch a single error type. A missing file
    raises ``FileNotFoundError`` and is allowed to propagate as-is
    (it's already a clear, well-known exception).
    """
    if not adapter_path.exists():
        raise FileNotFoundError(
            f"LoRA adapter file not found: {adapter_path}"
        )
    try:
        return load_file(str(adapter_path))
    except Exception as exc:
        raise AxBenchBaseError(
            f"Failed to load LoRA adapter file {adapter_path}: {exc}"
        ) from exc


def attach_lora_from_disk(
    model: nn.Module,
    adapter_path: Path,
    lora_cfg: LoraConfig,
    *,
    layer: int,
) -> tuple[list[LoRAAdapter], Callable[[], None]]:
    """Load LoRA weights from disk and attach them to ``model`` at ``layer``.

    This is the eval-time counterpart of ``training.train_lora_all``.
    It wraps the four ``_ATTACH_PATHS`` sublayers of ``model.transformer.h[layer]``
    with fresh :class:`LoRAAdapter` instances (LoRA B = 0 so initially
    the adapter forward is bit-exact equal to the base forward), then
    copies the ``lora_A`` / ``lora_B`` weights from ``adapter_path``
    onto each wrapper. The base ``nn.Linear`` weights are unchanged
    (but are frozen by :class:`LoRAAdapter.__init__`; ``detach_fn`` restores
    their exact pre-attach ``requires_grad`` state with the originals).

    Args:
        model: The target GPT to adapt. Must expose
            ``model.transformer.h[layer]`` with the standard
            ``attn.c_qkv``, ``attn.c_proj``, ``mlp.c_fc``,
            ``mlp.c_proj`` sublayers.
        adapter_path: Path to the safetensors file written by Task 24's
            training loop
            (``AxBenchBasePaths.lora_adapter(mode, concept_id)``).
        lora_cfg: The LoRA hyperparameters used at training time. Must
            match (rank, alpha) or the loaded tensors will not fit
            their adapter's parameter shapes.
        layer: The residual-stream layer index to attach to.
            Normally ``cfg.layer_hooks.diffmean_and_probe_layer``.

    Returns:
        A tuple ``(adapters, detach_fn)`` where:

          * ``adapters`` is the list of four :class:`LoRAAdapter` instances
            in the fixed order given by
            :data:`steering.eval.axbench_base.baselines.lora._attach._ATTACH_PATHS`.
          * ``detach_fn`` is a zero-argument callable that restores the
            original ``Linear`` modules byte-for-byte. Calling it twice
            is a no-op (idempotent).

    Raises:
        FileNotFoundError: ``adapter_path`` does not exist.
        AxBenchBaseError: the file is malformed, or is missing a
            required tensor key, or the model layout does not match
            the expected nanochat ``Block`` shape.
    """
    loaded = _load_adapter_state(adapter_path)

    # Attach first, then copy weights. If weight assignment fails (e.g.
    # missing key), we detach before re-raising so the caller sees a
    # clean model.
    original = _attach_lora_to_model(model, layer=layer, lora_cfg=lora_cfg)
    try:
        refs = _collect_attached_adapters(model, layer=layer)
        adapters: list[LoRAAdapter] = []
        for ref in refs:
            prefix = f"layer_{ref.layer}_{ref.parent_name}_{ref.child_name}"
            key_a = f"{prefix}_lora_A"
            key_b = f"{prefix}_lora_B"
            if key_a not in loaded:
                raise AxBenchBaseError(
                    f"Missing adapter key: {key_a} (in {adapter_path})"
                )
            if key_b not in loaded:
                raise AxBenchBaseError(
                    f"Missing adapter key: {key_b} (in {adapter_path})"
                )

            # Match dtype to the base Linear's weight dtype so the
            # adapter forward stays in the model's native dtype.
            base_dtype = ref.adapter.base.weight.dtype
            with t.no_grad():
                ref.adapter.lora_A.copy_(
                    loaded[key_a].to(dtype=base_dtype)
                )
                ref.adapter.lora_B.copy_(
                    loaded[key_b].to(dtype=base_dtype)
                )
            adapters.append(ref.adapter)
    except Exception:
        # Restore the model before re-raising so callers never see a
        # half-adapted state after an error.
        _detach_lora_from_model(model, original=original)
        raise

    detached = False

    def detach_fn() -> None:
        """Restore the original ``nn.Linear`` modules. Idempotent."""
        nonlocal detached
        if detached:
            return
        _detach_lora_from_model(model, original=original)
        detached = True

    return adapters, detach_fn
