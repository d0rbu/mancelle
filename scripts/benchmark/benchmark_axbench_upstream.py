"""Compare, profile, and project AxBench-Upstream producer campaigns.

Examples::

    uv run python -m scripts.benchmark.benchmark_axbench_upstream compare \
        --kind project --reference-manifest ... --candidate-manifest ...

    uv run python -m scripts.benchmark.benchmark_axbench_upstream profile \
        --kind encdec --output-dir out/axbench/profile/encdec-c2 \
        --override main_path=... --override manifest_path=...

    uv run python -m scripts.benchmark.benchmark_axbench_upstream throughput \
        --kind encdec --output-dir out/axbench/throughput/encdec-c2-sample-1 \
        --override main_path=... --override manifest_path=...

    uv run python -m scripts.benchmark.benchmark_axbench_upstream attention \
        --sequence-length 127 --packed-count 1024 --output-json attention.json

    uv run python -m scripts.benchmark.benchmark_axbench_upstream cost-model \
        --samples-json out/axbench/profile/samples.json --workers 8
"""

from __future__ import annotations

import argparse
import json
import math
import platform
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from contextlib import nullcontext
from pathlib import Path
from statistics import median
from typing import Any, cast

import pandas as pd
import torch as t
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf
from torch.nn.attention.flex_attention import FlexKernelOptions, flex_attention

from nanochat.atomic_io import atomic_write_text
from nanochat.common import compute_cleanup, compute_init, is_ddp_requested
from scripts.eval.axbench_upstream.merge_methods import INPUTS_PER_CONCEPT, resolve_concept_ids
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder.cross_attention import (
    build_packed_cross_attention_block_mask,
    norm,
)
from steering.conditional.encoder_decoder.packed_encoder import PackedEncoderLayout
from steering.eval.axbench_base.profiler import (
    AxBenchProfiler,
    resolve_profile_gpu_device_index,
)
from steering.eval.axbench_upstream.comparison import BundleKind, compare_bundle
from steering.eval.axbench_upstream.encdec_baselines.run import (
    run_encdec_baselines_from_dictconfig,
)
from steering.eval.axbench_upstream.profiling import (
    CellTraceController,
    CellTraceSelector,
    activate_cell_tracing,
)
from steering.eval.axbench_upstream.project_input import load_project_input_artifact
from steering.eval.axbench_upstream.run_methods import (
    _resolve_input_source,
    run_project_methods_from_dictconfig,
)
from steering.eval.axbench_upstream.work_grid import (
    CellKey,
    build_encdec_cell_grid,
    build_project_cell_grid,
    owned_encdec_cells,
    owned_project_cells,
)

_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "eval" / "axbench_upstream"
_CELLS_PER_CONCEPT: dict[BundleKind, int] = {"project": 240, "encdec": 260}
_CELLS_PER_PROMPT_GROUP: dict[BundleKind, int] = {
    kind: cells // INPUTS_PER_CONCEPT for kind, cells in _CELLS_PER_CONCEPT.items()
}
_DEFAULT_CONFIG: dict[BundleKind, str] = {
    "project": "methods",
    "encdec": "encdec_baselines",
}
_ATTENTION_HEADS = 6
_ATTENTION_HEAD_DIM = 128
_ATTENTION_WIDTH = _ATTENTION_HEADS * _ATTENTION_HEAD_DIM
_ATTENTION_KERNEL_OPTIONS: FlexKernelOptions = {"BACKEND": "TRITON"}


def _strict_json_object(path: Path, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite constant {value!r}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate key {key!r}")
            result[key] = value
        return result

    try:
        payload = json.loads(
            Path(path).read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{label} must be a readable strict-JSON object: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return payload


def _json_text(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _write_or_print(payload: Mapping[str, Any], output_json: Path | None) -> None:
    text = _json_text(payload)
    if output_json is None:
        sys.stdout.write(text)
        return
    atomic_write_text(text, Path(output_json).expanduser().resolve())


def _runtime_metadata(
    cfg: DictConfig, *, rank: int, local_rank: int, world_size: int
) -> dict[str, Any]:
    configured_device = cfg.get("device")
    metadata: dict[str, Any] = {
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "python": platform.python_version(),
        "torch": str(t.__version__),
        "cuda_build": str(t.version.cuda),
        "cudnn": t.backends.cudnn.version(),
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    if (
        isinstance(configured_device, str)
        and configured_device.startswith("cuda")
        and t.cuda.is_available()
    ):
        configured_index = t.device(configured_device).index
        device_index = t.cuda.current_device() if configured_index is None else configured_index
        metadata.update(
            {
                "gpu_name": t.cuda.get_device_name(device_index),
                "gpu_capability": list(t.cuda.get_device_capability(device_index)),
            }
        )
    return metadata


def _default_trace_selectors(
    kind: BundleKind, concept_id: int = 0,
) -> tuple[CellTraceSelector, ...]:
    method_factors = (
        (("SelfConditioning", 6.0),)
        if kind == "project"
        else (("EncDecDiffMean", 6.0), ("EncDecLoRA", 1.0))
    )
    return tuple(
        CellTraceSelector(
            driver=kind,
            method=method,
            concept_id=concept_id,
            input_id=0,
            factor_hex=factor.hex(),
        )
        for method, factor in method_factors
    )


def _cell_grid(kind: BundleKind, cfg: DictConfig) -> tuple[CellKey, ...]:
    config = cast(dict[str, Any], OmegaConf.to_container(cfg, resolve=True))
    seed = config.get("seed")
    if type(seed) is not int:
        raise ValueError("profile config seed must be an integer")
    if kind == "project":
        input_path, _, manifest_path = _resolve_input_source(
            config.get("main_path"),
            config.get("project_input_path"),
            config.get("project_input_manifest_path"),
        )
    else:
        main_path = config.get("main_path")
        if not isinstance(main_path, str) or not main_path:
            raise ValueError("profile config main_path must be a nonempty string")
        input_path, manifest_path = Path(main_path).expanduser().resolve(), None
    if manifest_path is None:
        frame = pd.read_parquet(input_path)
    else:
        frame, _, _ = load_project_input_artifact(
            input_path=input_path,
            manifest_path=manifest_path,
            prepared_dir=Path(config["prepared_dir"]),
            max_concepts=config["max_concepts"],
        )
    if kind == "project":
        if config.get("concept_ids") is not None:
            selected_ids = resolve_concept_ids(config["max_concepts"], config["concept_ids"])
            frame = frame.loc[frame.concept_id.isin(selected_ids)]
        return build_project_cell_grid(frame, seed=seed)
    return build_encdec_cell_grid(frame, seed=seed)


def _owned_trace_selectors(
    kind: BundleKind,
    cfg: DictConfig,
    *,
    rank: int,
    world_size: int,
) -> tuple[CellTraceSelector, ...]:
    cells = _cell_grid(kind, cfg)
    owner = owned_project_cells if kind == "project" else owned_encdec_cells
    rank_cells = frozenset(owner(cells, rank=rank, world_size=world_size))
    selected: list[CellTraceSelector] = []
    for selector in _default_trace_selectors(kind, min(cell.concept_id for cell in cells)):
        matches = [cell for cell in cells if selector.matches(cell)]
        if len(matches) != 1:
            raise ValueError(
                f"representative selector must match exactly one logical cell: "
                f"selector={selector}, matches={len(matches)}"
            )
        if matches[0] in rank_cells:
            selected.append(selector)
    return tuple(selected)


def _require_fresh_rank_profile_dir(path: Path) -> None:
    if path.exists():
        if not path.is_dir():
            raise FileExistsError(f"profile output path is not a directory: {path}")
        if any(path.iterdir()):
            raise FileExistsError(f"profile output directory must be fresh and empty: {path}")


def run_profile(
    *,
    kind: BundleKind,
    output_dir: Path,
    config_name: str,
    overrides: Sequence[str],
    torch_traces: bool,
    synchronize_cuda: bool = True,
    measurement_mode: str = "profile",
    cuda_event_regions: Sequence[str] = (),
) -> None:
    """Run the normal producer with region timing and optional selected-cell traces."""

    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=list(overrides))
    configured_device = cfg.get("device")
    if not isinstance(configured_device, str) or not configured_device:
        raise ValueError("profile config device must be a nonempty string")
    device_type = configured_device.split(":", 1)[0]
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported profile device type: {device_type!r}")
    deferred_cuda_regions = tuple(cuda_event_regions)
    if deferred_cuda_regions and device_type != "cuda":
        raise ValueError("deferred CUDA region timing requires device=cuda")
    if deferred_cuda_regions and synchronize_cuda:
        raise ValueError("deferred CUDA region timing requires synchronize_cuda=False")
    ddp = False
    rank = 0
    local_rank = 0
    world_size = 1
    requested_ddp = is_ddp_requested()
    if requested_ddp:
        matmul_precision = t.get_float32_matmul_precision()
        try:
            try:
                ddp, rank, local_rank, world_size, _device = compute_init(device_type)
            finally:
                # Profiling must observe the producer's historical arithmetic,
                # not compute_init's training-oriented TF32 mutation.
                t.set_float32_matmul_precision(matmul_precision)
        except BaseException:
            # compute_init may fail after creating a process group.
            compute_cleanup()
            raise
    try:
        if ddp and configured_device != "cuda":
            raise ValueError("torchrun AxBench-Upstream profiling requires device=cuda")
        rank_dir = Path(output_dir).expanduser().resolve() / f"rank_{rank:05d}"
        _require_fresh_rank_profile_dir(rank_dir)
        selectors = (
            _owned_trace_selectors(kind, cfg, rank=rank, world_size=world_size)
            if torch_traces
            else ()
        )
        controller = CellTraceController(rank_dir / "torch", selectors) if selectors else None
        trace_context = (
            activate_cell_tracing(controller) if controller is not None else nullcontext()
        )
        with AxBenchProfiler(
            rank_dir / "regions",
            metadata={
                "kind": kind,
                "measurement_mode": measurement_mode,
                "representative_selectors": [
                    selector.to_dict() for selector in selectors
                ],
                **(
                    {"cuda_event_regions": list(deferred_cuda_regions)}
                    if deferred_cuda_regions
                    else {}
                ),
                **_runtime_metadata(cfg, rank=rank, local_rank=local_rank, world_size=world_size),
            },
            synchronize_cuda=synchronize_cuda,
            gpu_utilization=device_type == "cuda",
            gpu_device_index=resolve_profile_gpu_device_index(
                configured_device,
                ddp_local_rank=local_rank,
                ddp_world_size=world_size,
            ),
            cuda_event_regions=deferred_cuda_regions,
        ) as profiler:
            with trace_context:
                if kind == "project":
                    run_project_methods_from_dictconfig(cfg)
                else:
                    run_encdec_baselines_from_dictconfig(cfg)
            with rank_synchronized_stage(
                "AxBench-Upstream selected-cell profile completion",
                device=configured_device if device_type == "cuda" else "cpu",
            ):
                if controller is not None:
                    controller.require_complete()
                if deferred_cuda_regions:
                    profiler.require_cuda_event_regions_complete()
    finally:
        if requested_ddp:
            compute_cleanup()


def _attention_delta(reference: t.Tensor, candidate: t.Tensor) -> dict[str, float | bool]:
    if reference.shape != candidate.shape:
        raise ValueError(
            "attention delta tensors must have identical shapes: "
            f"{tuple(reference.shape)} != {tuple(candidate.shape)}"
        )
    reference_float = reference.detach().float()
    candidate_float = candidate.detach().float()
    absolute = (reference_float - candidate_float).abs()
    relative = absolute / reference_float.abs().clamp_min(t.finfo(t.float32).tiny)
    return {
        "bitwise_equal": bool(t.equal(reference, candidate)),
        "max_absolute": float(absolute.max().item()),
        "mean_absolute": float(absolute.mean().item()),
        "max_relative": float(relative.max().item()),
    }


def _historical_attention_value_view(
    projected: t.Tensor,
    *,
    packed_count: int,
) -> t.Tensor:
    """Reproduce the current production V view for isolated measurement."""

    return projected.view(
        1,
        packed_count,
        _ATTENTION_HEADS,
        _ATTENTION_HEAD_DIM,
    ).transpose(1, 2)


def _extent_invariant_attention_value_view(
    projected: t.Tensor,
    *,
    packed_count: int,
) -> t.Tensor:
    """Build the logically equal experimental V view with a stable batch stride."""

    return (
        projected.view(
            packed_count,
            _ATTENTION_HEADS,
            _ATTENTION_HEAD_DIM,
        )
        .transpose(0, 1)
        .unsqueeze(0)
    )


def _cuda_latency_samples(
    call: Callable[[], t.Tensor],
    *,
    warmups: int,
    iterations: int,
) -> tuple[t.Tensor, dict[str, Any]]:
    """Measure one isolated CUDA callable after explicit warmup."""

    _positive_int(warmups, label="attention warmups")
    _positive_int(iterations, label="attention iterations")
    output: t.Tensor | None = None
    with t.inference_mode():
        for _ in range(warmups):
            output = call()
        t.cuda.synchronize()
        wall_samples: list[float] = []
        cuda_samples: list[float] = []
        for _ in range(iterations):
            start_event = t.cuda.Event(enable_timing=True)
            end_event = t.cuda.Event(enable_timing=True)
            started = time.perf_counter()
            start_event.record()
            output = call()
            end_event.record()
            end_event.synchronize()
            wall_samples.append(time.perf_counter() - started)
            cuda_samples.append(float(start_event.elapsed_time(end_event)) / 1000.0)
    if output is None:
        raise AssertionError("positive attention iteration count produced no output")
    with t.inference_mode():
        repeat = call()
    t.cuda.synchronize()
    if not bool(t.isfinite(output).all()) or not bool(t.isfinite(repeat).all()):
        raise RuntimeError("attention benchmark produced a non-finite output")
    return output, {
        "iterations": iterations,
        "warmups": warmups,
        "wall_seconds": _summary(wall_samples),
        "cuda_seconds": _summary(cuda_samples),
        "same_kernel_repeat_bitwise_equal": bool(t.equal(output, repeat)),
    }


def run_attention_microbenchmark(
    *,
    sequence_length: int,
    packed_count: int,
    warmups: int,
    iterations: int,
) -> dict[str, Any]:
    """Compare the historical and guard-stable V layouts on production-width CUDA."""

    _positive_int(sequence_length, label="attention sequence_length")
    _positive_int(packed_count, label="attention packed_count")
    _positive_int(warmups, label="attention warmups")
    _positive_int(iterations, label="attention iterations")
    if not t.cuda.is_available():
        raise RuntimeError("the AxBench-Upstream attention benchmark requires CUDA")

    device = t.device("cuda", t.cuda.current_device())
    t.manual_seed(1511)
    q_projected = t.randn(
        1,
        sequence_length,
        _ATTENTION_HEADS,
        _ATTENTION_HEAD_DIM,
        device=device,
        dtype=t.bfloat16,
    )
    kv_projected = t.randn(
        packed_count,
        2 * _ATTENTION_WIDTH,
        device=device,
        dtype=t.bfloat16,
    )
    k_projected, v_projected = kv_projected.split(
        [_ATTENTION_WIDTH, _ATTENTION_WIDTH],
        dim=-1,
    )
    q = norm(q_projected).transpose(1, 2).nan_to_num()
    k = (
        norm(
            k_projected.view(
                1,
                packed_count,
                _ATTENTION_HEADS,
                _ATTENTION_HEAD_DIM,
            )
        )
        .transpose(1, 2)
        .nan_to_num()
    )
    historical_v = _historical_attention_value_view(
        v_projected,
        packed_count=packed_count,
    )
    candidate_v = _extent_invariant_attention_value_view(
        v_projected,
        packed_count=packed_count,
    )
    if not t.equal(historical_v, candidate_v):
        raise AssertionError("historical and candidate V views must be logically identical")

    indices = t.arange(packed_count, device=device, dtype=t.int64)
    layout = PackedEncoderLayout(
        flat_indices=indices,
        valid_mask=t.ones(packed_count, device=device, dtype=t.bool),
        batch_indices=t.zeros(packed_count, device=device, dtype=t.int64),
        segment_indices=t.zeros(packed_count, device=device, dtype=t.int64),
        batch_size=1,
        n_segments=1,
        feature_count=packed_count,
    )
    seg_map = t.zeros(1, sequence_length, device=device, dtype=t.int64)
    block_mask = build_packed_cross_attention_block_mask(seg_map, layout)

    def eager_historical() -> t.Tensor:
        result = flex_attention(
            q,
            k,
            historical_v,
            block_mask=block_mask,
            enable_gqa=False,
            kernel_options=_ATTENTION_KERNEL_OPTIONS,
        )
        assert isinstance(result, t.Tensor)
        return result

    def eager_candidate() -> t.Tensor:
        result = flex_attention(
            q,
            k,
            candidate_v,
            block_mask=block_mask,
            enable_gqa=False,
            kernel_options=_ATTENTION_KERNEL_OPTIONS,
        )
        assert isinstance(result, t.Tensor)
        return result

    def candidate_flex_body(
        query: t.Tensor,
        key: t.Tensor,
        value: t.Tensor,
    ) -> t.Tensor:
        result = flex_attention(
            query,
            key,
            value,
            block_mask=block_mask,
            enable_gqa=False,
            kernel_options=_ATTENTION_KERNEL_OPTIONS,
        )
        assert isinstance(result, t.Tensor)
        return result

    compiled_candidate_body = t.compile(candidate_flex_body, dynamic=True)

    def compiled_candidate() -> t.Tensor:
        return compiled_candidate_body(q, k, candidate_v)

    def sdpa_reference() -> t.Tensor:
        # This all-one-segment benchmark makes the dense SDPA mask exactly
        # equivalent to the production BlockMask. SDPA is a test-only ceiling,
        # not a production fallback for general segmented packed inputs.
        return F.scaled_dot_product_attention(q, k, candidate_v)

    calls = {
        "eager_flex_historical_stride": eager_historical,
        "eager_flex_candidate_stride": eager_candidate,
        "compiled_flex_candidate_stride": compiled_candidate,
        "sdpa_all_one_segment_reference": sdpa_reference,
    }
    outputs: dict[str, t.Tensor] = {}
    measurements: dict[str, Any] = {}
    for name, call in calls.items():
        outputs[name], measurements[name] = _cuda_latency_samples(
            call,
            warmups=warmups,
            iterations=iterations,
        )
    historical = outputs["eager_flex_historical_stride"]
    deltas = {name: _attention_delta(historical, output) for name, output in outputs.items()}
    return {
        "schema_version": 1,
        "benchmark": "axbench_upstream_packed_cross_attention",
        "device": str(device),
        "gpu_name": t.cuda.get_device_name(device),
        "python": platform.python_version(),
        "torch": str(t.__version__),
        "cuda_build": str(t.version.cuda),
        "dtype": str(q.dtype),
        "shape": {
            "batch_size": 1,
            "query_heads": _ATTENTION_HEADS,
            "kv_heads": _ATTENTION_HEADS,
            "sequence_length": sequence_length,
            "packed_count": packed_count,
            "head_dim": _ATTENTION_HEAD_DIM,
        },
        "layouts": {
            "q_stride": list(q.stride()),
            "k_stride": list(k.stride()),
            "historical_v_stride": list(historical_v.stride()),
            "candidate_v_stride": list(candidate_v.stride()),
            "historical_v_storage_offset": historical_v.storage_offset(),
            "candidate_v_storage_offset": candidate_v.storage_offset(),
        },
        "measurements": measurements,
        "deltas_against_eager_historical_stride": deltas,
        "scope": (
            "SDPA is legal only because this isolated benchmark has one batch and one "
            "segment; it is not a production replacement for packed segmented routing"
        ),
    }


def _finite_nonnegative(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite nonnegative number")
    number = float(value)
    if not math.isfinite(number) or number < 0.0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return number


def _positive_int(value: Any, *, label: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"{label} must be a positive integer excluding bool")
    return value


def _percentile_95(values: Sequence[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        raise ValueError("cost-model statistic requires at least one sample")
    return {
        "sample_count": len(values),
        "median": median(values),
        "p95": _percentile_95(values),
        "min": min(values),
        "max": max(values),
    }


def _canonical_generation_cells_per_worker(
    kind: BundleKind,
    *,
    concepts: int,
    workers: int,
) -> int:
    # The c2/c500 campaign inputs are concept-major with ten contiguous prompt
    # groups per concept. Both producers synchronize after the complete first
    # concept before bulk generation, so unused capacity in one stage cannot
    # hide work in the other stage.
    first_stage_groups = INPUTS_PER_CONCEPT
    bulk_stage_groups = INPUTS_PER_CONCEPT * (concepts - 1)
    maximum_groups = math.ceil(first_stage_groups / workers) + math.ceil(
        bulk_stage_groups / workers
    )
    return maximum_groups * _CELLS_PER_PROMPT_GROUP[kind]


def _arbitrary_order_generation_cells_upper_bound(
    kind: BundleKind,
    *,
    concepts: int,
    workers: int,
) -> int:
    """Return a conservative stage-load upper bound over supported source-row orders."""

    total_groups = INPUTS_PER_CONCEPT * concepts
    maximum_groups_on_one_rank = math.ceil(total_groups / workers)
    first_stage_groups = min(INPUTS_PER_CONCEPT, maximum_groups_on_one_rank)
    bulk_stage_groups = min(
        INPUTS_PER_CONCEPT * (concepts - 1),
        maximum_groups_on_one_rank,
    )
    return (first_stage_groups + bulk_stage_groups) * _CELLS_PER_PROMPT_GROUP[kind]


def build_cost_model(samples: Mapping[str, Any], *, workers: int) -> dict[str, Any]:
    """Build labeled c2/c500 projections from explicit complete-run samples."""

    _positive_int(workers, label="workers")
    if samples.get("schema_version") != 1:
        raise ValueError("cost-model samples schema_version must equal 1")
    arms = samples.get("arms")
    if not isinstance(arms, dict) or set(arms) != {"project", "encdec"}:
        raise ValueError("cost-model samples arms must contain exactly project and encdec")
    modeled_arms: dict[str, Any] = {}
    for kind in ("project", "encdec"):
        raw_samples = arms[kind]
        if not isinstance(raw_samples, list) or len(raw_samples) < 3:
            raise ValueError(f"cost-model {kind} arm requires at least three samples")
        fit_per_concept: list[float] = []
        generation_per_cell: list[float] = []
        other_seconds: list[float] = []
        tokens_per_cell: list[float] = []
        for index, raw_sample in enumerate(raw_samples):
            label = f"cost-model {kind} sample {index}"
            if not isinstance(raw_sample, dict):
                raise ValueError(f"{label} must be an object")
            if set(raw_sample) != {
                "concepts",
                "generation_cells",
                "fit_seconds",
                "generation_seconds",
                "other_seconds",
                "actual_output_tokens",
            }:
                raise ValueError(f"{label} has missing or unexpected fields")
            concepts = _positive_int(raw_sample["concepts"], label=f"{label}.concepts")
            cells = _positive_int(raw_sample["generation_cells"], label=f"{label}.generation_cells")
            expected_cells = _CELLS_PER_CONCEPT[kind] * concepts
            if cells != expected_cells:
                raise ValueError(
                    f"{label}.generation_cells must equal {expected_cells} for {concepts} concepts"
                )
            fit_seconds = _finite_nonnegative(
                raw_sample["fit_seconds"], label=f"{label}.fit_seconds"
            )
            generation_seconds = _finite_nonnegative(
                raw_sample["generation_seconds"], label=f"{label}.generation_seconds"
            )
            overhead = _finite_nonnegative(
                raw_sample["other_seconds"], label=f"{label}.other_seconds"
            )
            tokens = _finite_nonnegative(
                raw_sample["actual_output_tokens"], label=f"{label}.actual_output_tokens"
            )
            fit_per_concept.append(fit_seconds / concepts)
            generation_per_cell.append(generation_seconds / cells)
            other_seconds.append(overhead)
            tokens_per_cell.append(tokens / cells)

        components = {
            "fit_seconds_per_concept": _summary(fit_per_concept),
            "generation_seconds_per_cell": _summary(generation_per_cell),
            "other_seconds_per_run": _summary(other_seconds),
            "actual_output_tokens_per_cell": _summary(tokens_per_cell),
        }
        projections: dict[str, Any] = {}
        for concepts in (2, 500):
            cells = _CELLS_PER_CONCEPT[kind] * concepts
            fit = median(fit_per_concept) * math.ceil(concepts / workers)
            generation = median(generation_per_cell) * _canonical_generation_cells_per_worker(
                kind,
                concepts=concepts,
                workers=workers,
            )
            arbitrary_order_generation_upper_bound = median(
                generation_per_cell
            ) * _arbitrary_order_generation_cells_upper_bound(
                kind,
                concepts=concepts,
                workers=workers,
            )
            other = median(other_seconds)
            projections[f"c{concepts}"] = {
                "kind": "projection",
                "concepts": concepts,
                "generation_cells": cells,
                "workers": workers,
                "fit_seconds": fit,
                "generation_seconds": generation,
                "generation_seconds_arbitrary_order_upper_bound": (
                    arbitrary_order_generation_upper_bound
                ),
                "other_seconds": other,
                "total_seconds": fit + generation + other,
                "total_seconds_arbitrary_order_upper_bound": (
                    fit + arbitrary_order_generation_upper_bound + other
                ),
                "projected_output_tokens": median(tokens_per_cell) * cells,
            }
        modeled_arms[kind] = {"normalized_samples": components, "projections": projections}
    return {
        "schema_version": 1,
        "result_kind": "projection_not_measurement",
        "workers": workers,
        "assumptions": [
            "cell and concept fit costs use medians of complete-run normalized samples",
            "generation_seconds assumes concept-major input with ten contiguous prompt groups per concept",
            "the arbitrary-order bound covers every supported interleaving of those prompt groups",
            "both generation estimates sum the slowest rank in each synchronized stage",
            "other_seconds_per_run does not scale with concept count or workers",
            "no multi-GPU scaling claim follows from this arithmetic",
        ],
        "arms": modeled_arms,
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare", help="strictly compare two artifact bundles")
    compare.add_argument("--kind", choices=("project", "encdec"), required=True)
    compare.add_argument("--reference-manifest", type=Path, required=True)
    compare.add_argument("--candidate-manifest", type=Path, required=True)
    compare.add_argument("--output-json", type=Path)
    compare.add_argument(
        "--include-nl-steering",
        action="store_true",
        help="compare the explicit three-project-method NL product",
    )

    profile_parser = subparsers.add_parser("profile", help="profile a normal producer run")
    profile_parser.add_argument("--kind", choices=("project", "encdec"), required=True)
    profile_parser.add_argument("--output-dir", type=Path, required=True)
    profile_parser.add_argument("--config-name")
    profile_parser.add_argument("--override", action="append", default=[])
    profile_parser.add_argument("--no-torch-traces", action="store_true")

    throughput = subparsers.add_parser(
        "throughput",
        help="run one fresh complete producer sample with low-overhead region timing",
    )
    throughput.add_argument("--kind", choices=("project", "encdec"), required=True)
    throughput.add_argument("--output-dir", type=Path, required=True)
    throughput.add_argument("--config-name")
    throughput.add_argument("--override", action="append", default=[])
    throughput.add_argument(
        "--cuda-event-region",
        action="append",
        default=[],
        help=(
            "record deferred current-stream CUDA event timings for this exact "
            "profile-region name; repeat for multiple regions"
        ),
    )

    attention = subparsers.add_parser(
        "attention",
        help="benchmark production-width packed cross-attention layouts/backends",
    )
    attention.add_argument("--sequence-length", type=int, default=127)
    attention.add_argument("--packed-count", type=int, default=1024)
    attention.add_argument("--warmups", type=int, default=5)
    attention.add_argument("--iterations", type=int, default=20)
    attention.add_argument("--output-json", type=Path)

    cost = subparsers.add_parser("cost-model", help="project c2/c500 from measured samples")
    cost.add_argument("--samples-json", type=Path, required=True)
    cost.add_argument("--workers", type=int, required=True)
    cost.add_argument("--output-json", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "compare":
        report = compare_bundle(
            cast(BundleKind, args.kind),
            args.reference_manifest,
            args.candidate_manifest,
            include_nl_steering=args.include_nl_steering,
        )
        _write_or_print(report.to_dict(), args.output_json)
        return
    if args.command == "profile":
        kind = cast(BundleKind, args.kind)
        run_profile(
            kind=kind,
            output_dir=args.output_dir,
            config_name=args.config_name or _DEFAULT_CONFIG[kind],
            overrides=args.override,
            torch_traces=not args.no_torch_traces,
        )
        return
    if args.command == "attention":
        report = run_attention_microbenchmark(
            sequence_length=args.sequence_length,
            packed_count=args.packed_count,
            warmups=args.warmups,
            iterations=args.iterations,
        )
        _write_or_print(report, args.output_json)
        return
    if args.command == "throughput":
        kind = cast(BundleKind, args.kind)
        run_profile(
            kind=kind,
            output_dir=args.output_dir,
            config_name=args.config_name or _DEFAULT_CONFIG[kind],
            overrides=args.override,
            torch_traces=False,
            synchronize_cuda=False,
            measurement_mode="throughput",
            cuda_event_regions=tuple(args.cuda_event_region),
        )
        return
    samples = _strict_json_object(args.samples_json, label="cost-model samples")
    _write_or_print(build_cost_model(samples, workers=args.workers), args.output_json)


if __name__ == "__main__":
    main()
