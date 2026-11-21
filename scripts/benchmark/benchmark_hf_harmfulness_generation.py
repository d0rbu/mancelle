"""Benchmark repository-HF harmfulness batching against the serial shape.

This is a generation-only benchmark: it loads one strict local repository SFT
export, uses real harmfulness/benign prompts, and never constructs a judge
client. The serial reference calls the production engine with one row at a
time, preserving the former model-forward shape without retaining a second
generation implementation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import torch as t
from torch.profiler import ProfilerActivity, profile, record_function

from steering.eval.harmfulness_prompt_pool import resolve_harmfulness_prompt_pool
from steering.eval.hf_chat_generation import (
    HuggingFaceChatGenerationEngine,
    load_repository_hf_checkpoint,
)

_DEFAULT_MAX_NEW_TOKENS = 32
_DEFAULT_TEMPERATURE = 0.7
_DEFAULT_TOP_K = 50
_DEFAULT_SEED = 4242


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError(f"expected an integer >= 1, got {value!r}")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not parsed > 0.0 or not t.isfinite(t.tensor(parsed)).item():
        raise argparse.ArgumentTypeError(f"expected a finite number > 0, got {value!r}")
    return parsed


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--iterations", type=_positive_int, default=3)
    parser.add_argument("--num-harmful", type=_positive_int, default=4)
    parser.add_argument("--num-benign", type=_positive_int, default=4)
    parser.add_argument("--max-new-tokens", type=_positive_int, default=_DEFAULT_MAX_NEW_TOKENS)
    parser.add_argument("--temperature", type=_positive_float, default=_DEFAULT_TEMPERATURE)
    parser.add_argument("--top-k", type=_positive_int, default=_DEFAULT_TOP_K)
    parser.add_argument("--seed", type=int, default=_DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--batch-sizes",
        type=_positive_int,
        nargs="+",
        help=(
            "Measure production outer-batch scaling for one or more sizes instead of the "
            "serial-versus-single-batch comparison."
        ),
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        help="Optionally capture one serial and one batched torch.profiler trace after timing.",
    )
    args = parser.parse_args(argv)
    if args.seed < 0:
        parser.error("--seed must be non-negative")
    if args.batch_sizes is not None and args.profile_dir is not None:
        parser.error("--profile-dir is unavailable with --batch-sizes")
    return args


def _synchronize(device: t.device) -> None:
    if device.type == "cuda":
        t.cuda.synchronize(device)


def _generate_serial(
    engine: HuggingFaceChatGenerationEngine,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> list[str]:
    responses: list[str] = []
    for index, prompt in enumerate(prompts):
        responses.extend(
            engine.generate_batch(
                [prompt],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed + index,
                render_as_chat=True,
            )
        )
    return responses


def _generate_batched(
    engine: HuggingFaceChatGenerationEngine,
    prompts: list[str],
    *,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> list[str]:
    return engine.generate_batch(
        prompts,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        seed=seed,
        render_as_chat=True,
    )


def _generate_outer_batched(
    engine: HuggingFaceChatGenerationEngine,
    prompts: list[str],
    *,
    batch_size: int,
    max_new_tokens: int,
    temperature: float,
    top_k: int | None,
    seed: int,
) -> list[str]:
    responses: list[str] = []
    for start in range(0, len(prompts), batch_size):
        responses.extend(
            _generate_batched(
                engine,
                prompts[start : start + batch_size],
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_k=top_k,
                seed=seed + start,
            )
        )
    return responses


def _measure(
    fn: Callable[[], list[str]],
    *,
    device: t.device,
    iterations: int,
) -> tuple[list[float], list[list[str]]]:
    samples: list[float] = []
    outputs: list[list[str]] = []
    for _ in range(iterations):
        _synchronize(device)
        started_at = time.perf_counter()
        outputs.append(fn())
        _synchronize(device)
        samples.append(time.perf_counter() - started_at)
    return samples, outputs


def _measure_with_peak(
    fn: Callable[[], list[str]],
    *,
    device: t.device,
    iterations: int,
) -> tuple[list[float], list[list[str]], list[float]]:
    samples: list[float] = []
    outputs: list[list[str]] = []
    peaks_mib: list[float] = []
    for _ in range(iterations):
        if device.type == "cuda":
            t.cuda.empty_cache()
            t.cuda.reset_peak_memory_stats(device)
        _synchronize(device)
        started_at = time.perf_counter()
        outputs.append(fn())
        _synchronize(device)
        samples.append(time.perf_counter() - started_at)
        if device.type == "cuda":
            peaks_mib.append(t.cuda.max_memory_allocated(device) / 1024**2)
    return samples, outputs, peaks_mib


def _sha256_json(value: Any) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _profile_call(
    fn: Callable[[], list[str]],
    *,
    device: t.device,
    name: str,
    output_dir: Path,
) -> dict[str, str]:
    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)
    with profile(activities=activities, record_shapes=True, with_stack=False) as profiler:
        with record_function(f"hf_harmfulness_{name}"):
            fn()
        _synchronize(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = output_dir / f"{name}_trace.json"
    averages_path = output_dir / f"{name}_key_averages.txt"
    profiler.export_chrome_trace(str(trace_path))
    averages_path.write_text(
        profiler.key_averages().table(sort_by="self_cuda_time_total", row_limit=80),
        encoding="utf-8",
    )
    return {"trace": str(trace_path), "key_averages": str(averages_path)}


def _run_batch_scaling(
    *,
    engine: HuggingFaceChatGenerationEngine,
    prompts: list[str],
    batch_sizes: list[int],
    common: dict[str, Any],
    device: t.device,
    iterations: int,
    checkpoint_identity: dict[str, Any],
) -> dict[str, Any]:
    if len(set(batch_sizes)) != len(batch_sizes):
        raise ValueError("--batch-sizes values must be unique")
    if any(batch_size > len(prompts) for batch_size in batch_sizes):
        raise ValueError(
            f"--batch-sizes cannot exceed the fixed prompt count {len(prompts)}"
        )

    measurements: dict[str, dict[str, Any]] = {}
    for batch_size in batch_sizes:
        _generate_outer_batched(
            engine,
            prompts,
            batch_size=batch_size,
            max_new_tokens=min(4, int(common["max_new_tokens"])),
            temperature=float(common["temperature"]),
            top_k=int(common["top_k"]),
            seed=int(common["seed"]),
        )
        samples, outputs, peaks_mib = _measure_with_peak(
            lambda size=batch_size: _generate_outer_batched(
                engine,
                prompts,
                batch_size=size,
                **common,
            ),
            device=device,
            iterations=iterations,
        )
        median_seconds = statistics.median(samples)
        measurements[str(batch_size)] = {
            "latency_seconds": samples,
            "median_seconds": median_seconds,
            "prompts_per_second": len(prompts) / median_seconds,
            "requested_tokens_per_second": (
                len(prompts) * int(common["max_new_tokens"]) / median_seconds
            ),
            "peak_allocated_mib": max(peaks_mib) if peaks_mib else None,
            "repeatable": all(rows == outputs[0] for rows in outputs[1:]),
            "output_hashes": [_sha256_json(rows) for rows in outputs],
        }

    reference_throughput = measurements[str(batch_sizes[0])]["prompts_per_second"]
    for measurement in measurements.values():
        measurement["throughput_gain_vs_first"] = (
            measurement["prompts_per_second"] / reference_throughput
        )
    return {
        "schema_version": 1,
        "mode": "batch_scaling",
        "checkpoint_identity": checkpoint_identity,
        "device": str(device),
        "gpu": t.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": t.__version__,
        "prompt_count": len(prompts),
        "prompt_sha256": _sha256_json(prompts),
        "generation": common,
        "iterations": iterations,
        "batch_sizes": batch_sizes,
        "measurements": measurements,
    }


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    device = t.device(args.device)
    if device.type == "cuda" and not t.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but unavailable: {device}")
    checkpoint = load_repository_hf_checkpoint(args.checkpoint_dir, device=str(device))
    engine = HuggingFaceChatGenerationEngine(checkpoint)
    pool = resolve_harmfulness_prompt_pool(
        seed=42,
        max_harmful=args.num_harmful,
        num_benign=args.num_benign,
    )
    prompts = [str(row["prompt"]) for row in pool.rows]
    common = {
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_k": args.top_k,
        "seed": args.seed,
    }
    if args.batch_sizes is not None:
        return _run_batch_scaling(
            engine=engine,
            prompts=prompts,
            batch_sizes=args.batch_sizes,
            common=common,
            device=device,
            iterations=args.iterations,
            checkpoint_identity=checkpoint.model_identity,
        )

    # Short, shape-specific warmups keep compilation/cache effects out of both
    # headline samples without spending a full measurement-sized pass.
    warmup_prompts = prompts[: min(2, len(prompts))]
    _generate_serial(
        engine,
        warmup_prompts,
        max_new_tokens=min(4, args.max_new_tokens),
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )
    _generate_batched(
        engine,
        warmup_prompts,
        max_new_tokens=min(4, args.max_new_tokens),
        temperature=args.temperature,
        top_k=args.top_k,
        seed=args.seed,
    )

    serial_samples, serial_outputs = _measure(
        lambda: _generate_serial(engine, prompts, **common),
        device=device,
        iterations=args.iterations,
    )
    batched_samples, batched_outputs = _measure(
        lambda: _generate_batched(engine, prompts, **common),
        device=device,
        iterations=args.iterations,
    )
    serial_median = statistics.median(serial_samples)
    batched_median = statistics.median(batched_samples)

    greedy_serial = _generate_serial(
        engine,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=None,
        seed=args.seed,
    )
    greedy_batched = _generate_batched(
        engine,
        prompts,
        max_new_tokens=args.max_new_tokens,
        temperature=0.0,
        top_k=None,
        seed=args.seed,
    )
    result = {
        "schema_version": 1,
        "checkpoint_identity": checkpoint.model_identity,
        "device": str(device),
        "gpu": t.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch": t.__version__,
        "prompt_count": len(prompts),
        "prompt_sha256": _sha256_json(prompts),
        "generation": common,
        "iterations": args.iterations,
        "serial": {
            "latency_seconds": serial_samples,
            "median_seconds": serial_median,
            "output_hashes": [_sha256_json(rows) for rows in serial_outputs],
        },
        "batched": {
            "latency_seconds": batched_samples,
            "median_seconds": batched_median,
            "output_hashes": [_sha256_json(rows) for rows in batched_outputs],
        },
        "speedup": serial_median / batched_median,
        "latency_reduction_percent": 100.0 * (1.0 - batched_median / serial_median),
        "serial_repeatable": all(rows == serial_outputs[0] for rows in serial_outputs[1:]),
        "batched_repeatable": all(rows == batched_outputs[0] for rows in batched_outputs[1:]),
        "sampled_serial_batched_exact": serial_outputs[0] == batched_outputs[0],
        "greedy_serial_batched_exact": greedy_serial == greedy_batched,
    }
    if args.profile_dir is not None:
        result["profiles"] = {
            "serial": _profile_call(
                lambda: _generate_serial(engine, prompts, **common),
                device=device,
                name="serial",
                output_dir=args.profile_dir,
            ),
            "batched": _profile_call(
                lambda: _generate_batched(engine, prompts, **common),
                device=device,
                name="batched",
                output_dir=args.profile_dir,
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    result = run_benchmark(args)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")


if __name__ == "__main__":
    main()
