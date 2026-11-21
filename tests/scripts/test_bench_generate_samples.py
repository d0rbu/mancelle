"""Smoke coverage for the generate_samples decode-loop microbenchmark."""

from __future__ import annotations

import json
from argparse import Namespace
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import scripts.benchmark.bench_generate_samples as benchmark
import steering.eval.axbench_bpb.runtime_state as runtime_state_module
import steering.eval.inference_engine as inference_engine_module


def _install_tiny_loader(
    monkeypatch,
    gpt_model,
    mock_tokenizer,
) -> None:
    def _fake_load_model(**_kwargs: Any):
        return gpt_model, mock_tokenizer, {}

    monkeypatch.setattr(runtime_state_module, "load_model", _fake_load_model)
    monkeypatch.setattr(
        benchmark,
        "_build_prompt_tokens",
        lambda _tokenizer: [mock_tokenizer.get_bos_token_id(), 2, 3],
    )


def test_bench_generate_samples_prints_json_with_tiny_loader(
    gpt_model,
    mock_tokenizer,
    monkeypatch,
    capsys,
) -> None:
    _install_tiny_loader(monkeypatch, gpt_model, mock_tokenizer)

    benchmark.main(["--device", "cpu", "--iterations", "2"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["device"] == "cpu"
    assert payload["iterations"] == 2
    assert payload["n_samples"] == 4
    assert payload["prompt_tokens"] == 3
    assert payload["max_new_tokens"] == 32
    assert payload["temperature"] == 0.7
    assert payload["seed"] == 1260
    assert payload["stop_mode"] == "none"
    assert payload["stop_token_count"] == 0
    assert payload["aggregate_wall_seconds"] > 0
    assert payload["wall_seconds_per_call"] > 0
    assert len(payload["per_iteration_wall_seconds"]) == 2
    assert payload["median_wall_seconds"] > 0
    assert payload["p95_wall_seconds"] > 0
    assert payload["end_to_end_tokens_per_second"] > 0
    assert payload["decode_region_seconds"] > 0
    counters = payload["profiler_counters"]
    assert counters["axbench_base.generation.decode_steps"] == 2 * 4 * 32
    assert counters["axbench_base.generation.batched_decode_iterations"] == 2 * 32
    assert counters["axbench_base.generation.prefill_tokens"] == 2 * 3


def test_benchmark_synchronizes_exactly_around_the_timed_loop(
    gpt_model,
    mock_tokenizer,
    monkeypatch,
) -> None:
    events: list[str] = []
    generate_calls = 0
    timer_calls = 0

    def _fake_generate(_engine, _prompt_tokens, stop_token_ids, temperature):
        nonlocal generate_calls
        assert stop_token_ids == set()
        assert temperature == 0.7
        phase = "warmup" if generate_calls < 5 else "timed"
        generate_calls += 1
        events.append(phase)
        return [[1] * 32 for _ in range(4)]

    def _fake_synchronize(_device: t.device) -> None:
        events.append("synchronize")

    def _fake_perf_counter() -> float:
        nonlocal timer_calls
        events.append(f"timer-{timer_calls}")
        timer_calls += 1
        return float(timer_calls)

    _install_tiny_loader(monkeypatch, gpt_model, mock_tokenizer)
    monkeypatch.setattr(benchmark, "_generate", _fake_generate)
    monkeypatch.setattr(benchmark, "_synchronize", _fake_synchronize)
    monkeypatch.setattr(
        benchmark,
        "_load_profile_totals",
        lambda _profile_dir: (
            {"axbench_base.generation.decode_steps": 256.0},
            {"axbench_base.generation.decode": 1.0},
        ),
    )
    monkeypatch.setattr(
        benchmark,
        "time",
        SimpleNamespace(perf_counter=_fake_perf_counter),
    )

    payload = benchmark.run_benchmark(Namespace(device="cpu", iterations=2, stop_mode="none"))

    assert payload["seed"] == 1260
    assert events == [
        "warmup",
        "warmup",
        "warmup",
        "warmup",
        "warmup",
        "synchronize",
        "timer-0",
        "timer-1",
        "timed",
        "timer-2",
        "timer-3",
        "timed",
        "timer-4",
        "synchronize",
        "timer-5",
    ]


def test_bench_generate_samples_compaction_mode_exercises_stop_path(
    gpt_model,
    mock_tokenizer,
    monkeypatch,
    capsys,
) -> None:
    sampler_calls = 0

    def _sample(logits, _rng, temperature, _top_k):
        nonlocal sampler_calls
        sampler_calls += 1
        assert temperature == 1.5
        if logits.size(0) == 4:
            tokens = [0, 1, 2, 3]
        else:
            assert logits.size(0) == 3
            tokens = [64, 64, 64]
        return t.tensor(tokens, dtype=t.long, device=logits.device).unsqueeze(1)

    _install_tiny_loader(monkeypatch, gpt_model, mock_tokenizer)
    monkeypatch.setattr(inference_engine_module, "sample_next_token", _sample)

    benchmark.main(["--device", "cpu", "--iterations", "1", "--stop-mode", "compaction"])
    payload = json.loads(capsys.readouterr().out)

    assert payload["stop_mode"] == "compaction"
    assert payload["stop_token_count"] == 4
    assert payload["temperature"] == 1.5
    assert payload["seed"] == 1260
    assert sampler_calls == 2 * (5 + 1)
    assert payload["profiler_counters"]["axbench_base.generation.batch_compactions"] == 1


def test_benchmark_helpers_cover_prompt_growth_validation_and_cuda_sync(
    mock_tokenizer,
    monkeypatch,
) -> None:
    with pytest.raises(
        benchmark.argparse.ArgumentTypeError,
        match=r"expected an integer >= 1",
    ):
        benchmark._positive_int("0")

    prompt = benchmark._build_prompt_tokens(mock_tokenizer, target_tokens=400)
    assert len(prompt) == 400
    assert prompt[0] == mock_tokenizer.get_bos_token_id()
    assert benchmark._percentile([4.0, 1.0, 2.0, 3.0], 0.5) == 2.5

    synchronized: list[t.device] = []
    monkeypatch.setattr(
        benchmark.t.cuda,
        "synchronize",
        lambda device: synchronized.append(device),
    )
    cuda_device = t.device("cuda:0")
    benchmark._synchronize(cuda_device)
    assert synchronized == [cuda_device]


def test_benchmark_fails_loudly_for_unavailable_cuda_and_context_overflow(
    gpt_model,
    mock_tokenizer,
    monkeypatch,
) -> None:
    monkeypatch.setattr(benchmark.t.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match=r"CUDA is unavailable"):
        benchmark.run_benchmark(Namespace(device="cuda", iterations=1, stop_mode="none"))

    _install_tiny_loader(monkeypatch, gpt_model, mock_tokenizer)
    monkeypatch.setattr(
        benchmark,
        "_build_prompt_tokens",
        lambda _tokenizer: list(range(100)),
    )
    with pytest.raises(
        ValueError,
        match=r"benchmark prompt and decode budget exceed checkpoint context",
    ):
        benchmark.run_benchmark(Namespace(device="cpu", iterations=1, stop_mode="none"))


@pytest.mark.parametrize(
    ("stop_mode", "rows", "message"),
    [
        pytest.param(
            "none",
            [[], [1] * 32, [1] * 32, [1] * 32],
            "empty stop set must consume the full decode budget",
            id="none-short-row",
        ),
        pytest.param(
            "compaction",
            [[1] * 32 for _ in range(4)],
            "compaction stop set did not produce a finish event",
            id="compaction-no-finish",
        ),
    ],
)
def test_benchmark_rejects_workloads_that_do_not_match_the_selected_mode(
    stop_mode: str,
    rows: list[list[int]],
    message: str,
    gpt_model,
    mock_tokenizer,
    monkeypatch,
) -> None:
    _install_tiny_loader(monkeypatch, gpt_model, mock_tokenizer)
    monkeypatch.setattr(
        benchmark,
        "_generate",
        lambda _engine, _prompt_tokens, _stop_token_ids, _temperature: rows,
    )

    with pytest.raises(AssertionError, match=message):
        benchmark.run_benchmark(Namespace(device="cpu", iterations=1, stop_mode=stop_mode))


@pytest.mark.parametrize(
    ("stop_mode", "rows", "profile_totals", "message"),
    [
        pytest.param(
            "none",
            [[1] * 32 for _ in range(4)],
            ({}, {"axbench_base.generation.decode": 1.0}),
            "did not emit the decode_steps counter",
            id="missing-decode-steps",
        ),
        pytest.param(
            "none",
            [[1] * 32 for _ in range(4)],
            ({"axbench_base.generation.decode_steps": 128.0}, {}),
            "did not emit the decode region",
            id="missing-decode-region",
        ),
        pytest.param(
            "compaction",
            [[], [1], [1], [1]],
            (
                {"axbench_base.generation.decode_steps": 7.0},
                {"axbench_base.generation.decode": 1.0},
            ),
            "compaction mode did not compact the active batch",
            id="missing-compaction-counter",
        ),
    ],
)
def test_benchmark_requires_profiler_evidence_for_every_reported_mode(
    stop_mode: str,
    rows: list[list[int]],
    profile_totals: tuple[dict[str, float], dict[str, float]],
    message: str,
    gpt_model,
    mock_tokenizer,
    monkeypatch,
) -> None:
    _install_tiny_loader(monkeypatch, gpt_model, mock_tokenizer)
    monkeypatch.setattr(
        benchmark,
        "_generate",
        lambda _engine, _prompt_tokens, _stop_token_ids, _temperature: rows,
    )
    monkeypatch.setattr(
        benchmark,
        "_load_profile_totals",
        lambda _profile_dir: profile_totals,
    )

    with pytest.raises(AssertionError, match=message):
        benchmark.run_benchmark(Namespace(device="cpu", iterations=1, stop_mode=stop_mode))
