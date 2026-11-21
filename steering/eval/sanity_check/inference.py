"""Sanity-check: InferenceEngine smoke tests on the decoder checkpoint."""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext

import torch

from nanochat.checkpoint_manager import build_model, find_largest_model, find_last_step
from nanochat.common import autodetect_device_type, get_base_dir
from steering.eval.inference_engine import InferenceEngine, build_engine_for_features


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
    print(f"Model loaded. Config: {meta.get('model_config', {})}")

    # Use autocast for bfloat16 models on CUDA (same as dclm_eval.py)
    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    engine = InferenceEngine(model, tokenizer, device)

    with autocast_ctx:
        # ---- Test 1: Greedy completions ----
        print("\n" + "=" * 60)
        print("TEST 1: Greedy completions (temperature=0)")
        print("=" * 60)

        prompts = [
            "The capital of France is",
            "def fibonacci(n):",
            "Once upon a time",
            "The meaning of life is",
            "User: How do I make pasta?\nAssistant:",
        ]

        outputs = engine.generate_batch(
            prompts,
            temperature=0,
            max_new_tokens=64,
        )

        for prompt, output in zip(prompts, outputs, strict=True):
            print(f"\nPROMPT: {prompt!r}")
            print(f"OUTPUT: {output!r}")

        # Sanity checks
        for i, output in enumerate(outputs):
            if len(output.strip()) == 0:
                print(f"WARNING: Output {i} is empty!")
                sys.exit(1)

        # Check for degenerate repetition
        for i, output in enumerate(outputs):
            words = output.split()
            if len(words) > 10:
                unique_ratio = len(set(words)) / len(words)
                if unique_ratio < 0.1:
                    print(
                        f"WARNING: Output {i} looks like degenerate repetition (unique ratio: {unique_ratio:.2f})"
                    )
                    sys.exit(1)

        print("\nAll greedy completions look reasonable.")

        # ---- Test 2: Reproducibility with seeds ----
        print("\n" + "=" * 60)
        print("TEST 2: Reproducibility with seeds")
        print("=" * 60)

        seed = 42
        prompt = "The capital of France is"

        samples_a = []
        for i in range(10):
            out = engine.generate(
                prompt,
                temperature=0.7,
                max_new_tokens=64,
                seed=seed + i,
            )
            samples_a.append(out)

        samples_b = []
        for i in range(10):
            out = engine.generate(
                prompt,
                temperature=0.7,
                max_new_tokens=64,
                seed=seed + i,
            )
            samples_b.append(out)

        all_match = True
        for i, (a, b) in enumerate(zip(samples_a, samples_b, strict=True)):
            if a != b:
                print(f"MISMATCH at sample {i}:")
                print(f"  A: {a[:100]!r}")
                print(f"  B: {b[:100]!r}")
                all_match = False

        if all_match:
            print("All 10 samples match across runs with same seeds.")
        else:
            print("ERROR: Reproducibility test FAILED!")
            sys.exit(1)

        # Verify samples are diverse (not all identical)
        unique_samples = set(samples_a)
        print(f"Unique samples out of 10: {len(unique_samples)}")
        if len(unique_samples) == 1:
            print(
                "WARNING: All samples are identical despite different seeds — temperature may not be working."
            )

        # ---- Test 3: build_engine_for_features rejects decoder-only + features ----
        print("\n" + "=" * 60)
        print("TEST 3: build_engine_for_features rejects decoder-only + features")
        print("=" * 60)

        try:
            build_engine_for_features(
                model, tokenizer, device,
                features={0: 5},
                context_provider=None,
            )
            print("  FAIL: build_engine_for_features did not raise on decoder-only + features")
            sys.exit(1)
        except ValueError as e:
            if "decoder-only" in str(e):
                print(f"  PASS: build_engine_for_features raised ValueError ({e!r})")
            else:
                print(f"  FAIL: wrong error message: {e!r}")
                sys.exit(1)

    print("\n" + "=" * 60)
    print("ALL INFERENCE ENGINE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
