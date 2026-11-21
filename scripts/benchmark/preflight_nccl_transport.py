"""Bounded NCCL correctness and timing probe for a reserved training allocation.

Run this under ``torchrun`` with an explicit ``NCCL_TIMEOUT_SEC``.  The probe
starts with a one-element all-reduce, then exercises all-reduce,
reduce-scatter, and all-gather at configured per-rank shard sizes.  It emits
rank-critical timings as JSON from rank zero.  This is a transport health gate,
not an end-to-end training benchmark or a replacement for ``nccl-tests``.
"""

from __future__ import annotations

import json
import os
import socket
import statistics
from collections.abc import Callable, Sequence
from typing import Any

import hydra
import torch as t
import torch.distributed as dist
from omegaconf import DictConfig
from torch.cuda import nccl

from nanochat.common import compute_cleanup, compute_init


def _percentile(values: Sequence[float], q: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _summary_ms(samples: Sequence[float]) -> dict[str, float]:
    return {
        "median_ms": statistics.median(samples),
        "p95_ms": _percentile(samples, 0.95),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def _nccl_version() -> str | None:
    try:
        version = nccl.version()
    except (AttributeError, RuntimeError):
        return None
    if isinstance(version, tuple):
        return ".".join(str(part) for part in version)
    return str(version)


def _positive_int(value: Any, *, name: str, allow_zero: bool = False) -> int:
    minimum = 0 if allow_zero else 1
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        comparison = ">= 0" if allow_zero else "> 0"
        raise ValueError(f"{name} must be an integer {comparison}, got {value!r}")
    return value


def _resolve_dtype(name: str) -> t.dtype:
    supported = {"bfloat16": t.bfloat16, "float32": t.float32}
    try:
        return supported[name]
    except KeyError:
        raise ValueError(
            f"transport.dtype must be one of {sorted(supported)}, got {name!r}"
        ) from None


def _elements_for_mib(shard_mib: int, dtype: t.dtype) -> int:
    shard_bytes = _positive_int(shard_mib, name="transport.shard_mib item") * 2**20
    if shard_bytes % dtype.itemsize != 0:
        raise ValueError(
            f"{shard_bytes} bytes is not divisible by dtype item size {dtype.itemsize}"
        )
    return shard_bytes // dtype.itemsize


def _time_collective(
    collective: Callable[[], None],
    *,
    prepare: Callable[[], None] | None,
    warmup_steps: int,
    steps: int,
) -> list[float]:
    for _ in range(warmup_steps):
        if prepare is not None:
            prepare()
        collective()
    dist.barrier()
    t.cuda.synchronize()
    samples: list[float] = []
    for _ in range(steps):
        if prepare is not None:
            prepare()
        started = t.cuda.Event(enable_timing=True)
        finished = t.cuda.Event(enable_timing=True)
        started.record()
        collective()
        finished.record()
        finished.synchronize()
        samples.append(started.elapsed_time(finished))
    return samples


def _assert_constant(tensor: t.Tensor, expected: float, *, operation: str) -> None:
    if not bool(t.all(tensor == expected).item()):
        raise RuntimeError(f"{operation} correctness check failed; expected {expected}")


def _fill_tensor(tensor: t.Tensor, value: float) -> None:
    tensor.fill_(value)


def _probe_tiny_all_reduce(device: t.device, rank: int, world_size: int) -> None:
    tensor = t.tensor([rank + 1.0], dtype=t.float32, device=device)
    dist.all_reduce(tensor)
    t.cuda.synchronize()
    expected = world_size * (world_size + 1) / 2
    _assert_constant(tensor, expected, operation="one-element all-reduce")


def _reduce_scatter(output: t.Tensor, input_: t.Tensor) -> None:
    implementation = getattr(dist, "reduce_scatter_single", dist.reduce_scatter_tensor)
    implementation(output, input_, op=dist.ReduceOp.SUM)


def _all_gather(output: t.Tensor, input_: t.Tensor) -> None:
    implementation = getattr(dist, "all_gather_single", dist.all_gather_into_tensor)
    implementation(output, input_)


def _probe_shard_size(
    *,
    device: t.device,
    rank: int,
    world_size: int,
    shard_mib: int,
    dtype: t.dtype,
    warmup_steps: int,
    steps: int,
) -> list[dict[str, Any]]:
    elements = _elements_for_mib(shard_mib, dtype)
    expected_sum = float(world_size * (world_size + 1) // 2)
    results: list[dict[str, Any]] = []

    all_reduce = t.empty(elements, dtype=dtype, device=device)
    all_reduce_samples = _time_collective(
        lambda tensor=all_reduce: dist.all_reduce(tensor),
        prepare=lambda tensor=all_reduce: _fill_tensor(tensor, rank + 1),
        warmup_steps=warmup_steps,
        steps=steps,
    )
    _assert_constant(all_reduce, expected_sum, operation="all-reduce")
    results.append(
        {
            "operation": "all_reduce",
            "shard_mib": shard_mib,
            "rank_samples_ms": all_reduce_samples,
        }
    )
    del all_reduce

    reduce_scatter_input = t.full(
        (elements * world_size,),
        rank + 1,
        dtype=dtype,
        device=device,
    )
    reduce_scatter_output = t.empty(elements, dtype=dtype, device=device)
    reduce_scatter_samples = _time_collective(
        lambda output=reduce_scatter_output, input_=reduce_scatter_input: _reduce_scatter(
            output,
            input_,
        ),
        prepare=None,
        warmup_steps=warmup_steps,
        steps=steps,
    )
    _assert_constant(reduce_scatter_output, expected_sum, operation="reduce-scatter")
    results.append(
        {
            "operation": "reduce_scatter",
            "shard_mib": shard_mib,
            "rank_samples_ms": reduce_scatter_samples,
        }
    )
    del reduce_scatter_input, reduce_scatter_output
    t.cuda.empty_cache()

    all_gather_input = t.full(
        (elements,),
        rank,
        dtype=dtype,
        device=device,
    )
    all_gather_output = t.empty(
        elements * world_size,
        dtype=dtype,
        device=device,
    )
    all_gather_samples = _time_collective(
        lambda output=all_gather_output, input_=all_gather_input: _all_gather(
            output,
            input_,
        ),
        prepare=None,
        warmup_steps=warmup_steps,
        steps=steps,
    )
    gathered = all_gather_output.view(world_size, elements)
    for source_rank in range(world_size):
        _assert_constant(
            gathered[source_rank],
            float(source_rank),
            operation=f"all-gather source rank {source_rank}",
        )
    results.append(
        {
            "operation": "all_gather",
            "shard_mib": shard_mib,
            "rank_samples_ms": all_gather_samples,
        }
    )
    del all_gather_input, all_gather_output, gathered
    t.cuda.empty_cache()
    return results


def _aggregate_rank_results(rank_results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    run_count = len(rank_results[0]["runs"])
    aggregated: list[dict[str, Any]] = []
    for run_index in range(run_count):
        first = rank_results[0]["runs"][run_index]
        operation = str(first["operation"])
        shard_mib = int(first["shard_mib"])
        if any(
            result["runs"][run_index]["operation"] != operation
            or int(result["runs"][run_index]["shard_mib"]) != shard_mib
            for result in rank_results
        ):
            raise RuntimeError("rank transport probe order disagreed")
        steps = len(first["rank_samples_ms"])
        critical = [
            max(
                float(result["runs"][run_index]["rank_samples_ms"][step]) for result in rank_results
            )
            for step in range(steps)
        ]
        aggregated.append(
            {
                "operation": operation,
                "shard_mib": shard_mib,
                "critical_samples_ms": critical,
                "critical": _summary_ms(critical),
                "rank_samples_ms": [
                    [float(sample) for sample in result["runs"][run_index]["rank_samples_ms"]]
                    for result in rank_results
                ],
            }
        )
    return aggregated


def preflight_transport(cfg: DictConfig) -> dict[str, Any] | None:
    """Run the bounded transport probe and return JSON-ready rank-zero evidence."""
    if os.environ.get("NCCL_TIMEOUT_SEC") is None:
        raise RuntimeError(
            "NCCL_TIMEOUT_SEC must be set explicitly so a broken transport cannot "
            "consume the reservation indefinitely"
        )
    transport_cfg = cfg.transport
    warmup_steps = _positive_int(
        transport_cfg.warmup_steps,
        name="transport.warmup_steps",
        allow_zero=True,
    )
    steps = _positive_int(transport_cfg.steps, name="transport.steps")
    dtype = _resolve_dtype(str(transport_cfg.dtype))
    shard_mib = [
        _positive_int(value, name="transport.shard_mib item") for value in transport_cfg.shard_mib
    ]
    if not shard_mib:
        raise ValueError("transport.shard_mib must contain at least one size")

    ddp, rank, local_rank, world_size, device = compute_init("cuda")
    if not ddp:
        compute_cleanup()
        raise RuntimeError("transport preflight requires torchrun")
    try:
        max_buffer_gib = float(transport_cfg.max_buffer_gib)
        if max_buffer_gib <= 0:
            raise ValueError("transport.max_buffer_gib must be positive")
        largest_buffer_bytes = max(shard_mib) * 2**20 * world_size
        if largest_buffer_bytes > max_buffer_gib * 2**30:
            raise ValueError(
                "largest reduce-scatter/all-gather buffer exceeds the configured guard: "
                f"{largest_buffer_bytes / 2**30:.3f} GiB > {max_buffer_gib:.3f} GiB"
            )

        _probe_tiny_all_reduce(device, rank, world_size)
        runs: list[dict[str, Any]] = []
        for size in shard_mib:
            runs.extend(
                _probe_shard_size(
                    device=device,
                    rank=rank,
                    world_size=world_size,
                    shard_mib=size,
                    dtype=dtype,
                    warmup_steps=warmup_steps,
                    steps=steps,
                )
            )

        rank_result = {
            "rank": rank,
            "local_rank": local_rank,
            "hostname": socket.gethostname(),
            "gpu": t.cuda.get_device_name(device),
            "runs": runs,
        }
        gathered: list[dict[str, Any] | None] = [None] * world_size
        dist.all_gather_object(gathered, rank_result)
        if rank != 0:
            return None
        rank_results = [result for result in gathered if result is not None]
        env_keys = (
            "NCCL_NET_GDR_LEVEL",
            "NCCL_PXN_DISABLE",
            "NCCL_P2P_PXN_LEVEL",
            "NCCL_IB_HCA",
            "NCCL_SOCKET_IFNAME",
        )
        return {
            "evidence": "measured_nccl_correctness_and_rank_critical_timing",
            "limitations": (
                "Synthetic collectives and fixed buffers only; this does not measure model "
                "compute, optimizer overlap, remote data, or end-to-end training throughput."
            ),
            "software": {
                "torch": t.__version__,
                "cuda": t.version.cuda,
                "nccl": _nccl_version(),
            },
            "world_size": world_size,
            "nodes": sorted({str(result["hostname"]) for result in rank_results}),
            "gpus": [str(result["gpu"]) for result in rank_results],
            "dtype": str(transport_cfg.dtype),
            "warmup_steps": warmup_steps,
            "steps": steps,
            "largest_buffer_gib": largest_buffer_bytes / 2**30,
            "max_buffer_guard_gib": max_buffer_gib,
            "transport_env": {key: os.environ.get(key) for key in env_keys},
            "runs": _aggregate_rank_results(rank_results),
        }
    finally:
        compute_cleanup()


@hydra.main(
    config_path="../../configs/profiling",
    config_name="nccl_transport_preflight",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    result = preflight_transport(cfg)
    if result is not None:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
