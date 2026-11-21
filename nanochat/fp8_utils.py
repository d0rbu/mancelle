"""Shared FP8 training utilities.

Used by base_train.py and conditional_encdec_train.py
to avoid duplicating the FP8 conversion logic and disable_fp8 context manager.
"""

from contextlib import contextmanager

import torch.nn as nn
from loguru import logger
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from nanochat.gpt import Linear


def fp8_module_filter(mod: nn.Module, fqn: str) -> bool:
    """Filter for FP8-eligible Linear layers.

    Requires:
    - Module is nn.Linear
    - Both dimensions divisible by 16 (FP8 hardware requirement)
    - Min dimension >= 128 (efficiency threshold)
    """
    if not isinstance(mod, nn.Linear):
        return False
    if mod.in_features % 16 != 0 or mod.out_features % 16 != 0:
        return False
    if min(mod.in_features, mod.out_features) < 128:
        return False
    return True


def apply_fp8_training(
    model: nn.Module,
    recipe: str = "tensorwise",
    exclude_lm_head: bool = True,
) -> tuple[int, int]:
    """Apply FP8 training conversion to eligible Linear layers.

    Args:
        model: model to mutate in-place.
        recipe: torchao FP8 recipe name (e.g. "tensorwise", "rowwise").
        exclude_lm_head: if True (default), any module named `lm_head`
            (the unembedding projection) is kept in bf16/fp32 rather than
            converted to FP8. This matches standard LLM training recipes:
            the output is softcapped and cross-entropy is sensitive to
            tail logits, so rounding error in the last projection is
            disproportionately damaging. Audit: Wave 3C/7,
            phase2/03-numerical-correctness.md §2.2.

    Returns (num_converted, num_skipped).
    """
    fp8_config = Float8LinearConfig.from_recipe_name(recipe)
    num_linear = sum(1 for m in model.modules() if isinstance(m, nn.Linear))

    def _filter(mod: nn.Module, fqn: str) -> bool:
        if exclude_lm_head and (fqn == "lm_head" or fqn.endswith(".lm_head")):
            return False
        return fp8_module_filter(mod, fqn)

    convert_to_float8_training(model, config=fp8_config, module_filter_fn=_filter)
    num_fp8 = sum(1 for m in model.modules() if "Float8" in type(m).__name__)
    num_skipped = num_linear - num_fp8
    logger.info(
        f"FP8 training enabled ({recipe} scaling) - "
        f"converted {num_fp8}/{num_linear} linear layers, skipped {num_skipped} "
        f"({'lm_head excluded, ' if exclude_lm_head else ''}small/non-aligned)"
    )
    return num_fp8, num_skipped


@contextmanager
def disable_fp8(model: nn.Module):
    """Temporarily swap Float8Linear modules with nn.Linear for BF16 evaluation.

    CastConfig is a frozen dataclass, so we can't mutate scaling_type. Instead,
    we swap out Float8Linear modules entirely and restore them after.
    """
    fp8_locations = []
    for name, module in model.named_modules():
        if "Float8" in type(module).__name__:
            if "." in name:
                parent_name, attr_name = name.rsplit(".", 1)
                parent = model.get_submodule(parent_name)
            else:
                parent = model
                attr_name = name
            fp8_locations.append((parent, attr_name, module))

    if not fp8_locations:
        yield
        return

    for parent, attr_name, fp8_module in fp8_locations:
        linear = Linear(
            fp8_module.in_features,
            fp8_module.out_features,
            bias=fp8_module.bias is not None,
            device=fp8_module.weight.device,
            dtype=fp8_module.weight.dtype,
        )
        linear.weight = fp8_module.weight
        if fp8_module.bias is not None:
            linear.bias = fp8_module.bias
        setattr(parent, attr_name, linear)

    try:
        yield
    finally:
        for parent, attr_name, fp8_module in fp8_locations:
            setattr(parent, attr_name, fp8_module)
