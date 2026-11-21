"""Profiler contract tests for the generation hot path."""

from __future__ import annotations

from pathlib import Path

import torch as t
import yaml
from torch.profiler import ProfilerActivity, profile

from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.inference_engine import InferenceEngine


def _counter_totals(profile_dir: Path) -> dict[str, float]:
    summary = yaml.safe_load(
        (profile_dir / "summary.yaml").read_text(encoding="utf-8")
    )
    return {str(row["name"]): float(row["total"]) for row in summary["counters"]}


def _region_names(profile_dir: Path) -> set[str]:
    summary = yaml.safe_load(
        (profile_dir / "summary.yaml").read_text(encoding="utf-8")
    )
    return {str(row["name"]) for row in summary["regions"]}


def test_generate_stream_emits_prefill_and_decode_profile_data(
    gpt_model,
    mock_tokenizer,
    tmp_path: Path,
) -> None:
    engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    prompt_tokens = mock_tokenizer.encode(
        "profile me",
        prepend=mock_tokenizer.get_bos_token_id(),
    )
    profile_dir = tmp_path / "profile"

    with AxBenchProfiler(profile_dir, metadata={"test": True}):
        generated = list(
            engine.generate_stream(
                prompt_tokens,
                max_new_tokens=3,
                temperature=0.0,
            )
        )

    counters = _counter_totals(profile_dir)
    regions = _region_names(profile_dir)
    assert counters["axbench_base.generation.prefill_tokens"] == len(prompt_tokens)
    decode_steps = counters["axbench_base.generation.decode_steps"]
    assert len(generated) <= decode_steps <= 3
    assert "axbench_base.generation.prefill" in regions
    assert "axbench_base.generation.decode" in regions


def test_generate_stream_emits_phase_markers_under_torch_profiler(
    gpt_model,
    mock_tokenizer,
) -> None:
    engine = InferenceEngine(gpt_model, mock_tokenizer, t.device("cpu"))
    prompt_tokens = mock_tokenizer.encode(
        "profile phase markers",
        prepend=mock_tokenizer.get_bos_token_id(),
    )
    expected = list(
        engine.generate_stream(
            prompt_tokens,
            max_new_tokens=3,
            temperature=0.0,
            stop_token_ids=set(),
        )
    )

    with profile(activities=[ProfilerActivity.CPU], with_stack=False) as profiler:
        generated = list(
            engine.generate_stream(
                prompt_tokens,
                max_new_tokens=3,
                temperature=0.0,
                stop_token_ids=set(),
            )
        )

    names = {event.name for event in profiler.events()}
    assert generated == expected
    assert len(generated) == 3
    assert "generation/prefill_model_forward" in names
    assert "generation/sample_next_token" in names
    assert "generation/decode_model_forward" in names
