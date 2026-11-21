"""Sanity-check: DCLM CORE evaluation on the decoder checkpoint."""

from __future__ import annotations

import os
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import build_model, find_largest_model, find_last_step
from nanochat.common import autodetect_device_type, get_base_dir
from steering.eval.dclm_eval import evaluate_dclm_core

EXPECTED_CORE = 0.2461  # from report.md
TOLERANCE = 0.005


def main() -> None:
    device_type = autodetect_device_type()
    device = torch.device(device_type)

    # Load the decoder checkpoint
    base_dir = os.path.join(get_base_dir(), "base_checkpoints")
    model_tag = find_largest_model(base_dir)
    ckpt_dir = f"{base_dir}/{model_tag}"
    step = find_last_step(ckpt_dir)
    print(f"Loading model: tag={model_tag}, step={step}")

    model, tokenizer, meta = build_model(ckpt_dir, step, device, "eval")

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    print("\n" + "=" * 60)
    print("DCLM CORE EVALUATION (plain decoder-only, max_per_task=500)")
    print("=" * 60)

    with autocast_ctx:
        result = evaluate_dclm_core(
            model,
            tokenizer,
            device_type,
            max_per_task=500,
            context_provider=None,
        )

    core_score = result["core_metric"]
    per_task = result["results"]
    centered = result["centered_results"]

    print(f"\nCORE metric: {core_score:.4f}")
    print(f"Expected CORE: {EXPECTED_CORE:.4f}")
    print(f"Difference: {abs(core_score - EXPECTED_CORE):.4f}")

    print("\nPer-task scores:")
    for task, score in sorted(per_task.items()):
        c = centered.get(task, float("nan"))
        print(f"  {task}: {score:.4f} (centered: {c:.4f})")

    if abs(core_score - EXPECTED_CORE) > TOLERANCE:
        print(f"\nWARNING: CORE score differs from expected by more than {TOLERANCE}!")
        print("This may be due to using max_per_task=500 vs full eval.")
    else:
        print(f"\nCORE score matches expected within tolerance ({TOLERANCE}).")

    print("\n" + "=" * 60)
    print("DCLM CORE EVALUATION COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()
