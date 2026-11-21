"""Sanity-check: decoder-only steering-eval control.

This script intentionally stays decoder-only. It samples two unconditioned
continuations from the same checkpoint and sends the pair to the steering
pairwise judge. The expected behavior is no systematic preference for either
side; encoder-decoder feature conditioning is covered by the main steering
eval with a real context provider.
"""

from __future__ import annotations

import os
import sys
from contextlib import nullcontext
from pathlib import Path

import torch
import yaml

from nanochat.checkpoint_manager import build_model, find_largest_model, find_last_step
from nanochat.common import autodetect_device_type, get_base_dir, load_project_dotenv
from steering.eval.harmfulness_judge import JudgeClient
from steering.eval.inference_engine import InferenceEngine
from steering.eval.steering_eval import (
    PairwiseJudge,
    Verdict,
    judge_pair,
    load_config,
)


def _verdict_key(verdict: Verdict) -> str:
    return str(verdict.value)


def main() -> None:
    load_project_dotenv()

    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key:
        print("ERROR: OPENROUTER_API_KEY not set. Cannot run judge.")
        sys.exit(1)

    device_type = autodetect_device_type()
    device = torch.device(device_type)

    base_dir = os.path.join(get_base_dir(), "base_checkpoints")
    model_tag = find_largest_model(base_dir)
    ckpt_dir = f"{base_dir}/{model_tag}"
    step = find_last_step(ckpt_dir)
    print(f"Loading decoder-only model: tag={model_tag}, step={step}")

    model, tokenizer, _meta = build_model(ckpt_dir, step, device, "eval")
    engine = InferenceEngine(model, tokenizer, device)

    config_path = Path(__file__).parent / "steering_config.yaml"
    config = load_config(config_path)
    prompts = config.prompts or [""]
    n_samples = int(config.generation.samples_per_condition)

    judge_client = JudgeClient(model=config.judge.model, requests_per_minute=60)
    judge = PairwiseJudge(judge_client)

    output_dir = Path(get_base_dir()) / "steering_eval_decoder_sanity"
    output_dir.mkdir(parents=True, exist_ok=True)

    autocast_ctx = (
        torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
        if device_type == "cuda"
        else nullcontext()
    )

    print("\n" + "=" * 60)
    print("STEERING EVAL DECODER-ONLY CONTROL")
    print("=" * 60)
    print("Comparing two unconditioned samples from the same decoder-only model.\n")

    feature_reports = []
    for feature in config.features:
        verdict_counts = dict.fromkeys(("win", "loss", "tie", "inconsistent", "invalid"), 0)
        judgments = []
        with autocast_ctx:
            for sample_idx in range(n_samples):
                prompt = prompts[sample_idx % len(prompts)]
                seed_a = feature.id * 10000 + sample_idx
                seed_b = feature.id * 10000 + 5000 + sample_idx
                text_a = engine.generate(
                    prompt,
                    max_new_tokens=config.generation.max_new_tokens,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                    seed=seed_a,
                    render_as_chat=True,
                )
                text_b = engine.generate(
                    prompt,
                    max_new_tokens=config.generation.max_new_tokens,
                    temperature=config.generation.temperature,
                    top_k=config.generation.top_k,
                    seed=seed_b,
                    render_as_chat=True,
                )
                result = judge_pair(text_a, text_b, feature.description, judge)
                key = _verdict_key(result.verdict)
                verdict_counts[key] += 1
                judgments.append(
                    {
                        "prompt": prompt,
                        "seed_a": seed_a,
                        "seed_b": seed_b,
                        "sample_a": text_a,
                        "sample_b": text_b,
                        "ab_response": getattr(result.ab_response, "value", result.ab_response),
                        "ba_response": getattr(result.ba_response, "value", result.ba_response),
                        "verdict": key,
                    }
                )

        preference_rate = (
            verdict_counts["win"] / n_samples
            if n_samples > 0
            else 0.0
        )
        feature_reports.append(
            {
                "id": feature.id,
                "description": feature.description,
                "confidence": feature.confidence,
                "sample_a_preference_rate": preference_rate,
                "verdict_counts": verdict_counts,
                "judgments": judgments,
            }
        )
        print(
            f"Feature {feature.id}: sample-A preference={preference_rate:.3f}, "
            f"counts={verdict_counts}"
        )

    report = {
        "metadata": {
            "model_tag": model_tag,
            "step": int(step),
            "control": "decoder_only_two_unconditioned_samples",
            "num_samples_per_feature": n_samples,
        },
        "features": feature_reports,
    }
    report_path = output_dir / "report.yaml"
    report_path.write_text(yaml.safe_dump(report, sort_keys=False))
    print(f"\nWrote decoder-only steering sanity report to {report_path}")


if __name__ == "__main__":
    main()
