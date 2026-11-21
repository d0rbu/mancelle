"""Microbenchmark the tensor-native :meth:`InferenceEngine.generate_samples` loop.

Loads the canonical ``d12_base`` decoder-only checkpoint without creating an
evaluation run directory, warms up five times, then prints one JSON object with
aggregate and per-iteration wall time, end-to-end throughput, decode-region
time, and production profiler counters. ``--stop-mode compaction`` uses seed
1260, temperature 1.5, and token ids divisible by 32 to exercise active-batch
compaction; those values are calibrated against the pinned ``d12_base`` step
54600 checkpoint, where the fixed trajectory finishes two rows at different
steps (two compaction events) while the survivors run to the full budget. At
temperature 0.7 the repetitive prompt keeps the sampled rows in lockstep, so
every finish is simultaneous and Case-B compaction never runs.
"""

from __future__ import annotations

import argparse
import json
import statistics
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch as t
import yaml

from steering.eval.axbench_base.profiler import AxBenchProfiler
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.inference_engine import InferenceEngine

_N_SAMPLES = 4
_MAX_NEW_TOKENS = 32
_TEMPERATURE = 0.7
_COMPACTION_TEMPERATURE = 1.5
_COMPACTION_STOP_MODULUS = 32
_SEED = 1260
_PROMPT_TOKENS = 400
_WARMUP_ITERATIONS = 5
_DECODER_ONLY = {"source": "base", "model_tag": "d12_base", "step": 54600}
_PROMPT_TEXT = (
    "A careful performance benchmark should preserve the exact sampling "
    "contract while measuring only the execution strategy of the decode loop. "
)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value!r}")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        choices=("cuda", "cpu"),
        default="cuda",
        help="Device used to load and benchmark the d12_base decoder.",
    )
    parser.add_argument(
        "--iterations",
        type=_positive_int,
        default=20,
        help="Number of timed generate_samples calls after five warmups.",
    )
    parser.add_argument(
        "--stop-mode",
        choices=("none", "compaction"),
        default="none",
        help=(
            "Stop-set workload: none consumes the full budget; compaction uses "
            "the deterministic ids divisible by 32 at temperature 1.5 with "
            "seed 1260."
        ),
    )
    return parser.parse_args(argv)


def _build_prompt_tokens(tokenizer: Any, target_tokens: int = _PROMPT_TOKENS) -> list[int]:
    """Build a deterministic prompt containing exactly ``target_tokens`` ids."""
    bos_id = tokenizer.get_bos_token_id()
    repetitions = 1
    while True:
        encoded = tokenizer.encode(
            _PROMPT_TEXT * repetitions,
            prepend=bos_id,
        )
        token_ids = [int(token_id) for token_id in encoded]
        if len(token_ids) >= target_tokens:
            return token_ids[:target_tokens]
        repetitions *= 2


def _synchronize(device: t.device) -> None:
    if device.type == "cuda":
        t.cuda.synchronize(device)


def _percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _load_profile_totals(
    profile_dir: Path,
) -> tuple[dict[str, float], dict[str, float]]:
    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text(encoding="utf-8"))
    counters = {str(row["name"]): float(row["total"]) for row in summary["counters"]}
    regions = {str(row["name"]): float(row["total_s"]) for row in summary["regions"]}
    return counters, regions


def _stop_token_ids(tokenizer: Any, stop_mode: str) -> set[int]:
    if stop_mode == "none":
        return set()
    vocab_size = int(tokenizer.get_vocab_size())
    return set(range(0, vocab_size, _COMPACTION_STOP_MODULUS))


def _generate(
    engine: InferenceEngine,
    prompt_tokens: list[int],
    stop_token_ids: set[int],
    temperature: float,
) -> list[list[int]]:
    return engine.generate_samples(
        prompt_tokens,
        seed=_SEED,
        n_samples=_N_SAMPLES,
        max_new_tokens=_MAX_NEW_TOKENS,
        temperature=temperature,
        stop_token_ids=stop_token_ids,
    )


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the production decoder bundle and return the JSON-ready result."""
    device = t.device(args.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")

    cfg = AxBenchBpbConfig(
        decoder_only=dict(_DECODER_ONLY),
        device=args.device,
    )
    state = RuntimeState(cfg=cfg, ddp_rank=0, ddp_world_size=1)
    try:
        bundle = state.decoder_only()
        prompt_tokens = _build_prompt_tokens(bundle.tokenizer)
        if len(prompt_tokens) + _MAX_NEW_TOKENS > bundle.max_seq_len:
            raise ValueError(
                "benchmark prompt and decode budget exceed checkpoint context: "
                f"{len(prompt_tokens)} + {_MAX_NEW_TOKENS} > {bundle.max_seq_len}"
            )
        engine = InferenceEngine(bundle.model, bundle.tokenizer, device)
        stop_token_ids = _stop_token_ids(bundle.tokenizer, args.stop_mode)
        temperature = _COMPACTION_TEMPERATURE if args.stop_mode == "compaction" else _TEMPERATURE

        for _ in range(_WARMUP_ITERATIONS):
            _generate(engine, prompt_tokens, stop_token_ids, temperature)
        _synchronize(device)

        with tempfile.TemporaryDirectory(prefix="bench-generate-samples-") as tmp:
            profile_dir = Path(tmp) / "profile"
            with AxBenchProfiler(
                profile_dir,
                metadata={"benchmark": "generate_samples"},
                synchronize_cuda=False,
            ):
                started_at = time.perf_counter()
                per_iteration_wall_s: list[float] = []
                for _ in range(args.iterations):
                    iteration_started_at = time.perf_counter()
                    rows = _generate(engine, prompt_tokens, stop_token_ids, temperature)
                    per_iteration_wall_s.append(time.perf_counter() - iteration_started_at)
                    if args.stop_mode == "none":
                        if any(len(row) != _MAX_NEW_TOKENS for row in rows):
                            raise AssertionError(
                                "empty stop set must consume the full decode budget"
                            )
                    elif not any(len(row) < _MAX_NEW_TOKENS for row in rows):
                        raise AssertionError("compaction stop set did not produce a finish event")
                _synchronize(device)
                elapsed_s = time.perf_counter() - started_at
            counters, regions = _load_profile_totals(profile_dir)

        decode_steps = counters.get("axbench_base.generation.decode_steps")
        if decode_steps is None:
            raise AssertionError("generate_samples did not emit the decode_steps counter")
        decode_region_s = regions.get("axbench_base.generation.decode")
        if decode_region_s is None:
            raise AssertionError("generate_samples did not emit the decode region")
        if args.stop_mode == "compaction" and not counters.get(
            "axbench_base.generation.batch_compactions", 0.0
        ):
            raise AssertionError("compaction mode did not compact the active batch")
        return {
            "device": str(device),
            "iterations": args.iterations,
            "n_samples": _N_SAMPLES,
            "prompt_tokens": len(prompt_tokens),
            "max_new_tokens": _MAX_NEW_TOKENS,
            "temperature": temperature,
            "seed": _SEED,
            "stop_mode": args.stop_mode,
            "stop_token_count": len(stop_token_ids),
            "aggregate_wall_seconds": elapsed_s,
            "wall_seconds_per_call": elapsed_s / args.iterations,
            "per_iteration_wall_seconds": per_iteration_wall_s,
            "median_wall_seconds": statistics.median(per_iteration_wall_s),
            "p95_wall_seconds": _percentile(per_iteration_wall_s, 0.95),
            "end_to_end_tokens_per_second": decode_steps / elapsed_s,
            "decode_region_seconds": decode_region_s,
            "profiler_counters": counters,
        }
    finally:
        state.offload_models()


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    print(json.dumps(run_benchmark(args), sort_keys=True))


if __name__ == "__main__":
    main()
