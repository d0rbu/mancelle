"""Profile the Stage 0c per-prefix verification hot path.

This is a profiling-only wrapper around the normal AxBench-Base Stage 0c
dispatcher. It does not implement cross-prefix batching and does not change
Stage 0c semantics. The wrapper installs timing hooks around the existing
``_verify_candidate`` path, captures a torch CPU/CUDA trace, records cProfile
output, and stops after ``STAGE0C_PROFILE_VERIFY_CANDIDATES`` candidate-prefix
verification calls.

Because this profiler wraps the serial ``_verify_candidate`` function, it
requires ``verification_cross_prefix_batch_size=1``. The production default is
larger, so the script fails loudly if that override is omitted.

Example H100 run:

    STAGE0C_PROFILE_VERIFY_CANDIDATES=50 \
    uv run python -m scripts.benchmark.profile_stage0c_verification \
        stage=0c \
        run_tag=stage0c-profile-h100 \
        enable_concept_examples_qc=false \
        verification_cross_prefix_batch_size=1 \
        verification_candidate_scan_limit=200 \
        verification_candidate_attempt_limit=80 \
        feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
        feature_statistics_path=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base/feature_statistics.safetensors \
        decoder_only.source=base decoder_only.model_tag=d12_base \
        decoder_only.step=54600 \
        encdec.source=encdec encdec.model_tag=d12_baseline \
        encdec.step=54600

Outputs land under:

    <run_output_root>/profile/stage0c_verify_candidate/
        torch_trace.json
        cprofile_top.txt
        summary.json
        summary.md
"""

from __future__ import annotations

import cProfile
import io
import itertools
import json
import math
import os
import pstats
import statistics
import time
from collections import defaultdict
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import hydra
import torch as t
from loguru import logger
from omegaconf import DictConfig
from torch.profiler import ProfilerActivity, record_function
from torch.profiler import profile as torch_profile

import scripts.eval.axbench_base_eval as axbench_cli
from nanochat.common import compute_cleanup, compute_init
from steering.eval.axbench_base import _stage0c_internals as _si
from steering.eval.axbench_base import batched_generation as _bg
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import AxBenchBaseConfig, validate_config
from steering.eval.axbench_base.config_loader import cfg_to_dataclass
from steering.eval.axbench_base.runtime_state import RuntimeState, validate_env_for_stages
from steering.eval.axbench_base.synth import qc as _qc
from steering.eval.axbench_base.types import AxBenchBaseError, Stage
from steering.eval.cluster_config import axbench_base_config_to_dict

_DEFAULT_VERIFY_LIMIT = 50
_VERIFY_RECORD_PREFIX = "stage0c_profile.verify_candidate_"
_DEVICE_EVENT_CATEGORIES = {"kernel", "gpu_memcpy", "gpu_memset"}


class _Stage0cProfileComplete(Exception):
    """Private sentinel used to stop after the requested number of calls."""


class _TimingTokenizer:
    """Proxy that times tokenizer encode/decode while preserving attributes."""

    def __init__(self, inner: Any, state: _ProfileState) -> None:
        self._inner = inner
        self._state = state

    def encode(self, *args: Any, **kwargs: Any) -> Any:
        with self._state.timed("tokenizer.encode"):
            return self._inner.encode(*args, **kwargs)

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        with self._state.timed("tokenizer.decode"):
            return self._inner.decode(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


class _ProfileState:
    """Mutable profiling state shared by all installed wrappers."""

    def __init__(self, *, verify_limit: int, output_dir: Path) -> None:
        self.verify_limit = int(verify_limit)
        self.output_dir = output_dir
        self.timers: dict[str, list[float]] = defaultdict(list)
        self.call_records: list[dict[str, Any]] = []
        self.forward_events: list[dict[str, Any]] = []
        self._tokenizer_wrappers: dict[int, _TimingTokenizer] = {}
        self._hooked_models: set[int] = set()
        self._hook_handles: list[Any] = []
        self.current_call_index: int | None = None

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            self.timers[name].append(time.perf_counter() - started)

    def synchronize(self) -> None:
        if t.cuda.is_available():
            t.cuda.synchronize()

    def wrap_tokenizer(self, tokenizer: Any) -> _TimingTokenizer:
        key = id(tokenizer)
        wrapped = self._tokenizer_wrappers.get(key)
        if wrapped is None:
            wrapped = _TimingTokenizer(tokenizer, self)
            self._tokenizer_wrappers[key] = wrapped
        return wrapped

    def ensure_forward_hooks(self, base_model: Any) -> None:
        hf_model = _bg._hf_generation_model(base_model)
        key = id(hf_model)
        if key in self._hooked_models:
            return
        self._hooked_models.add(key)

        def _pre_hook(_module: Any, _inputs: Any) -> None:
            self.forward_events.append(
                {
                    "kind": "pre",
                    "candidate_call_index": self.current_call_index,
                    "perf_s": time.perf_counter(),
                }
            )

        def _post_hook(_module: Any, _inputs: Any, _output: Any) -> None:
            self.forward_events.append(
                {
                    "kind": "post",
                    "candidate_call_index": self.current_call_index,
                    "perf_s": time.perf_counter(),
                }
            )

        self._hook_handles.append(hf_model.register_forward_pre_hook(_pre_hook))
        self._hook_handles.append(hf_model.register_forward_hook(_post_hook))

    def close_hooks(self) -> None:
        for handle in self._hook_handles:
            remove = getattr(handle, "remove", None)
            if callable(remove):
                remove()
        self._hook_handles.clear()
        self._hooked_models.clear()


class _InstalledWrappers:
    """Install and restore monkeypatch-style wrappers for profiling."""

    def __init__(self, state: _ProfileState) -> None:
        self.state = state
        self._originals: dict[str, Any] = {}

    def install(self) -> None:
        self._originals = {
            "si_get_tokenizer": _si.get_extraction_tokenizer,
            "qc_get_tokenizer": _qc.get_extraction_tokenizer,
            "si_batched_generate": _si.batched_generate_seeds,
            "si_compute_fc_percentiles": _si.compute_fc_percentiles,
            "qc_tokenize_prefix_and_full": _qc._tokenize_prefix_and_full,
            "qc_run_sae_on_full": _qc._run_sae_on_full,
            "qc_max_percentile": _qc._max_percentile,
            "si_verify_candidate": _si._verify_candidate,
        }

        def _si_get_tokenizer(tokenizer_id: str) -> _TimingTokenizer:
            with self.state.timed("stage0c.get_extraction_tokenizer"):
                tokenizer = self._originals["si_get_tokenizer"](tokenizer_id)
            return self.state.wrap_tokenizer(tokenizer)

        def _qc_get_tokenizer(tokenizer_id: str = _qc.DEFAULT_EXTRACTION_TOKENIZER_ID) -> _TimingTokenizer:
            with self.state.timed("synth_qc.get_extraction_tokenizer"):
                tokenizer = self._originals["qc_get_tokenizer"](tokenizer_id)
            return self.state.wrap_tokenizer(tokenizer)

        def _batched_generate(base_model: Any, *args: Any, **kwargs: Any) -> Any:
            self.state.ensure_forward_hooks(base_model)
            self.state.synchronize()
            with self.state.timed("stage0c.verify_candidate_batched_generate"):
                out = self._originals["si_batched_generate"](
                    base_model, *args, **kwargs,
                )
            self.state.synchronize()
            return out

        def _compute_fc_percentiles(*args: Any, **kwargs: Any) -> Any:
            self.state.synchronize()
            with self.state.timed("stage0c.verify_candidate_compute_percentile"):
                out = self._originals["si_compute_fc_percentiles"](
                    *args, **kwargs,
                )
            self.state.synchronize()
            return out

        def _tokenize_prefix_and_full(*args: Any, **kwargs: Any) -> Any:
            with self.state.timed("synth_qc.tokenize_prefix_and_full"):
                return self._originals["qc_tokenize_prefix_and_full"](
                    *args, **kwargs,
                )

        def _run_sae_on_full(*args: Any, **kwargs: Any) -> Any:
            self.state.synchronize()
            with self.state.timed("synth_qc.run_sae_on_full"):
                out = self._originals["qc_run_sae_on_full"](*args, **kwargs)
            self.state.synchronize()
            return out

        def _max_percentile(*args: Any, **kwargs: Any) -> Any:
            with self.state.timed("synth_qc.max_percentile_lookup"):
                return self._originals["qc_max_percentile"](*args, **kwargs)

        def _verify_candidate(*args: Any, **kwargs: Any) -> Any:
            call_index = len(self.state.call_records)
            if call_index >= self.state.verify_limit:
                raise _Stage0cProfileComplete(
                    f"captured {self.state.verify_limit} _verify_candidate calls"
                )
            name = f"{_VERIFY_RECORD_PREFIX}{call_index:06d}"
            self.state.current_call_index = call_index
            self.state.synchronize()
            started = time.perf_counter()
            errored = False
            result = None
            try:
                with record_function(name):
                    result = self._originals["si_verify_candidate"](
                        *args, **kwargs,
                    )
                return result
            except BaseException:
                errored = True
                raise
            finally:
                self.state.synchronize()
                elapsed = time.perf_counter() - started
                self.state.call_records.append(
                    {
                        "call_index": call_index,
                        "wall_s": elapsed,
                        "errored": errored,
                        "verified": result is not None,
                    }
                )
                self.state.current_call_index = None
                if (
                    not errored
                    and len(self.state.call_records) >= self.state.verify_limit
                ):
                    raise _Stage0cProfileComplete(
                        f"captured {self.state.verify_limit} _verify_candidate calls"
                    )

        si_module: Any = _si
        qc_module: Any = _qc
        si_module.get_extraction_tokenizer = _si_get_tokenizer
        qc_module.get_extraction_tokenizer = _qc_get_tokenizer
        si_module.batched_generate_seeds = _batched_generate
        si_module.compute_fc_percentiles = _compute_fc_percentiles
        qc_module._tokenize_prefix_and_full = _tokenize_prefix_and_full
        qc_module._run_sae_on_full = _run_sae_on_full
        qc_module._max_percentile = _max_percentile
        si_module._verify_candidate = _verify_candidate

    def restore(self) -> None:
        if not self._originals:
            return
        si_module: Any = _si
        qc_module: Any = _qc
        si_module.get_extraction_tokenizer = self._originals["si_get_tokenizer"]
        qc_module.get_extraction_tokenizer = self._originals["qc_get_tokenizer"]
        si_module.batched_generate_seeds = self._originals["si_batched_generate"]
        si_module.compute_fc_percentiles = self._originals[
            "si_compute_fc_percentiles"
        ]
        qc_module._tokenize_prefix_and_full = self._originals[
            "qc_tokenize_prefix_and_full"
        ]
        qc_module._run_sae_on_full = self._originals["qc_run_sae_on_full"]
        qc_module._max_percentile = self._originals["qc_max_percentile"]
        si_module._verify_candidate = self._originals["si_verify_candidate"]
        self.state.close_hooks()
        self._originals = {}


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    try:
        value = int(raw)
    except ValueError as exc:
        raise AxBenchBaseError(
            f"{name} must be an integer, got {raw!r}"
        ) from exc
    if value <= 0:
        raise AxBenchBaseError(f"{name} must be > 0, got {value}")
    return value


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "total_s": 0.0,
            "mean_ms": None,
            "median_ms": None,
            "p95_ms": None,
            "min_ms": None,
            "max_ms": None,
        }
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "total_s": sum(values),
        "mean_ms": statistics.mean(values) * 1000.0,
        "median_ms": statistics.median(values) * 1000.0,
        "p95_ms": ordered[p95_idx] * 1000.0,
        "min_ms": ordered[0] * 1000.0,
        "max_ms": ordered[-1] * 1000.0,
    }


def _summary_unitless(values: list[float]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "min": None,
            "max": None,
        }
    ordered = sorted(values)
    p95_idx = min(len(ordered) - 1, math.ceil(0.95 * len(ordered)) - 1)
    return {
        "count": len(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "p95": ordered[p95_idx],
        "min": ordered[0],
        "max": ordered[-1],
    }


def _pair_forward_events(
    events: list[dict[str, Any]],
) -> list[tuple[int | None, float, float]]:
    pairs: list[tuple[int | None, float, float]] = []
    pending: tuple[int | None, float] | None = None
    for event in events:
        kind = event["kind"]
        call_index = event["candidate_call_index"]
        timestamp = float(event["perf_s"])
        if kind == "pre":
            pending = (call_index, timestamp)
        elif kind == "post" and pending is not None:
            pre_call_index, pre_timestamp = pending
            pairs.append((pre_call_index, pre_timestamp, timestamp))
            pending = None
    return pairs


def _forward_gap_summary(
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    pairs = _pair_forward_events(events)
    all_gaps: list[float] = []
    within_candidate_gaps: list[float] = []
    cross_candidate_gaps: list[float] = []
    for left, right in itertools.pairwise(pairs):
        left_call, _left_pre, left_post = left
        right_call, right_pre, _right_post = right
        gap = right_pre - left_post
        all_gaps.append(gap)
        if left_call == right_call:
            within_candidate_gaps.append(gap)
        else:
            cross_candidate_gaps.append(gap)
    return {
        "forward_calls": len(pairs),
        "all_gaps": _summary(all_gaps),
        "within_candidate_gaps": _summary(within_candidate_gaps),
        "cross_candidate_gaps": _summary(cross_candidate_gaps),
    }


def _merged_interval_duration(intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return 0.0
    ordered = sorted(intervals)
    total = 0.0
    cur_start, cur_end = ordered[0]
    for start, end in ordered[1:]:
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        total += cur_end - cur_start
        cur_start, cur_end = start, end
    total += cur_end - cur_start
    return total


def _parse_trace_gpu_activity(trace_path: Path) -> dict[int, dict[str, float]]:
    if not trace_path.exists():
        return {}
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", [])
    call_windows: dict[int, tuple[float, float]] = {}
    device_events: list[tuple[float, float]] = []
    for event in events:
        if event.get("ph") != "X":
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", ""))
        ts = float(event.get("ts", 0.0) or 0.0)
        dur = float(event.get("dur", 0.0) or 0.0)
        if category == "user_annotation" and name.startswith(_VERIFY_RECORD_PREFIX):
            raw_idx = name.removeprefix(_VERIFY_RECORD_PREFIX)
            try:
                call_windows[int(raw_idx)] = (ts, ts + dur)
            except ValueError:
                continue
        elif category in _DEVICE_EVENT_CATEGORIES:
            device_events.append((ts, dur))

    out: dict[int, dict[str, float]] = {}
    for call_index, (start_us, end_us) in call_windows.items():
        wall_us = max(0.0, end_us - start_us)
        active_intervals: list[tuple[float, float]] = []
        for event_ts, event_dur in device_events:
            event_end = event_ts + event_dur
            overlap_start = max(start_us, event_ts)
            overlap_end = min(end_us, event_end)
            if overlap_end > overlap_start:
                active_intervals.append((overlap_start, overlap_end))
        active_us = _merged_interval_duration(active_intervals)
        out[call_index] = {
            "trace_wall_s": wall_us / 1_000_000.0,
            "gpu_active_s": active_us / 1_000_000.0,
            "gpu_active_fraction": (
                active_us / wall_us if wall_us > 0.0 else 0.0
            ),
        }
    return out


def _build_summary_payload(
    *,
    cfg: AxBenchBaseConfig,
    state: _ProfileState,
    status: str,
    torch_trace_path: Path,
    cprofile_path: Path,
) -> dict[str, Any]:
    trace_gpu = _parse_trace_gpu_activity(torch_trace_path)
    call_records = []
    for record in state.call_records:
        enriched = dict(record)
        enriched.update(trace_gpu.get(int(record["call_index"]), {}))
        call_records.append(enriched)

    call_wall_values = [float(r["wall_s"]) for r in call_records]
    gpu_fractions = [
        float(r["gpu_active_fraction"])
        for r in call_records
        if "gpu_active_fraction" in r
    ]

    timed_regions = {
        name: _summary(values)
        for name, values in sorted(state.timers.items())
    }
    generation_total_s = sum(
        state.timers.get("stage0c.verify_candidate_batched_generate", []),
    )
    percentile_total_s = sum(
        state.timers.get("stage0c.verify_candidate_compute_percentile", []),
    )
    tokenizer_total_s = sum(
        sum(values)
        for name, values in state.timers.items()
        if "tokenizer" in name or "tokenize" in name
    )
    observed_total_s = sum(call_wall_values)
    residual_total_s = max(
        0.0,
        observed_total_s - generation_total_s - percentile_total_s,
    )

    return {
        "status": status,
        "run_tag": cfg.run_tag,
        "stage": cfg.stage,
        "device": cfg.device,
        "verify_limit": state.verify_limit,
        "n_verify_candidate_calls": len(call_records),
        "torch_trace_path": str(torch_trace_path),
        "cprofile_path": str(cprofile_path),
        "config": axbench_base_config_to_dict(cfg),
        "call_wall": _summary(call_wall_values),
        "per_call_gpu_active_fraction": _summary_unitless(gpu_fractions),
        "timed_regions": timed_regions,
        "forward_gaps": _forward_gap_summary(state.forward_events),
        "breakdown_totals": {
            "observed_verify_candidate_wall_s": observed_total_s,
            "generation_total_s": generation_total_s,
            "percentile_total_s": percentile_total_s,
            "tokenizer_related_total_s": tokenizer_total_s,
            "residual_setup_teardown_s": residual_total_s,
            "generation_fraction_of_wall": (
                generation_total_s / observed_total_s
                if observed_total_s > 0.0 else None
            ),
            "percentile_fraction_of_wall": (
                percentile_total_s / observed_total_s
                if observed_total_s > 0.0 else None
            ),
        },
        "call_records": call_records,
    }


def _fmt_ms(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_s(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.3f}"


def _fmt_float(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"{float(value):.4f}"


def _summary_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {summary['count']} | {_fmt_ms(summary['mean_ms'])} | "
        f"{_fmt_ms(summary['median_ms'])} | {_fmt_ms(summary['p95_ms'])} | "
        f"{_fmt_s(summary['total_s'])} |"
    )


def _fraction_row(label: str, summary: dict[str, Any]) -> str:
    return (
        f"| {label} | {summary['count']} | {_fmt_float(summary['mean'])} | "
        f"{_fmt_float(summary['median'])} | {_fmt_float(summary['p95'])} | "
        f"{_fmt_float(summary['min'])} | {_fmt_float(summary['max'])} |"
    )


def _write_markdown_summary(path: Path, payload: dict[str, Any]) -> None:
    lines = [
        "# Stage 0c Verification Profile",
        "",
        f"- status: `{payload['status']}`",
        f"- run_tag: `{payload['run_tag']}`",
        f"- device: `{payload['device']}`",
        f"- verify calls captured: `{payload['n_verify_candidate_calls']}`",
        f"- torch trace: `{payload['torch_trace_path']}`",
        f"- cProfile: `{payload['cprofile_path']}`",
        "",
        "## Per-Call Summary",
        "",
        "| Measurement | Count | Mean ms | Median ms | P95 ms | Total s |",
        "|---|---:|---:|---:|---:|---:|",
        _summary_row("verify_candidate wall", payload["call_wall"]),
        "",
        "## GPU Active Fraction",
        "",
        "| Measurement | Count | Mean | Median | P95 | Min | Max |",
        "|---|---:|---:|---:|---:|---:|---:|",
        _fraction_row(
            "per-call GPU active fraction",
            payload["per_call_gpu_active_fraction"],
        ),
        "",
        "## Timed Regions",
        "",
        "| Region | Count | Mean ms | Median ms | P95 ms | Total s |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, summary in payload["timed_regions"].items():
        lines.append(_summary_row(name, summary))

    forward = payload["forward_gaps"]
    lines.extend(
        [
            "",
            "## Forward Gaps",
            "",
            f"- forward calls: `{forward['forward_calls']}`",
            "",
            "| Gap | Count | Mean ms | Median ms | P95 ms | Total s |",
            "|---|---:|---:|---:|---:|---:|",
            _summary_row("all", forward["all_gaps"]),
            _summary_row("within candidate", forward["within_candidate_gaps"]),
            _summary_row("cross candidate", forward["cross_candidate_gaps"]),
            "",
            "## Breakdown Totals",
            "",
            "| Component | Seconds |",
            "|---|---:|",
        ]
    )
    for key, value in payload["breakdown_totals"].items():
        if key.endswith("_fraction_of_wall"):
            formatted = _fmt_float(value)
        else:
            formatted = _fmt_s(value)
        lines.append(f"| `{key}` | {formatted} |")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_reports(
    *,
    cfg: AxBenchBaseConfig,
    state: _ProfileState,
    status: str,
    torch_prof: Any,
    cprofiler: cProfile.Profile,
) -> None:
    state.output_dir.mkdir(parents=True, exist_ok=True)
    trace_path = state.output_dir / "torch_trace.json"
    cprofile_path = state.output_dir / "cprofile_top.txt"
    summary_json_path = state.output_dir / "summary.json"
    summary_md_path = state.output_dir / "summary.md"

    torch_prof.export_chrome_trace(str(trace_path))
    stream = io.StringIO()
    pstats.Stats(cprofiler, stream=stream).sort_stats("cumtime").print_stats(80)
    cprofile_path.write_text(stream.getvalue(), encoding="utf-8")

    payload = _build_summary_payload(
        cfg=cfg,
        state=state,
        status=status,
        torch_trace_path=trace_path,
        cprofile_path=cprofile_path,
    )
    summary_json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_markdown_summary(summary_md_path, payload)
    logger.info(
        "Stage 0c verification profile written to {path}",
        path=str(state.output_dir),
    )


def _require_serial_prefix_profile_mode(cfg_obj: AxBenchBaseConfig) -> None:
    if cfg_obj.verification_cross_prefix_batch_size != 1:
        raise AxBenchBaseError(
            "profile_stage0c_verification wraps the serial _verify_candidate "
            "path and requires verification_cross_prefix_batch_size=1. "
            "The production default is cross-prefix batched, so add "
            "`verification_cross_prefix_batch_size=1` to profile the "
            "per-prefix hot path."
        )


def _run(cfg: DictConfig) -> None:
    cfg_obj = cfg_to_dataclass(cfg)
    validate_config(cfg_obj)
    if Stage.coerce(cfg_obj.stage) is not Stage.PREPARE_NEGATIVE_PREFIXES:
        raise AxBenchBaseError(
            "profile_stage0c_verification must be run with stage=0c. "
            f"Got stage={cfg_obj.stage!r}."
        )
    _require_serial_prefix_profile_mode(cfg_obj)
    if cfg_obj.run_tag is None:
        cfg_obj = cfg_obj.replace(run_tag=axbench_cli._generate_run_tag(cfg_obj))
    assert cfg_obj.run_tag is not None
    output_paths = paths.AxBenchBasePaths.from_config(cfg_obj)
    paths.assert_no_legacy_output_shadow(cfg_obj)

    verify_limit = _env_int(
        "STAGE0C_PROFILE_VERIFY_CANDIDATES",
        _DEFAULT_VERIFY_LIMIT,
    )
    record_shapes = _env_bool("STAGE0C_PROFILE_RECORD_SHAPES", False)
    profile_memory = _env_bool("STAGE0C_PROFILE_MEMORY", False)

    stages = [Stage.PREPARE_NEGATIVE_PREFIXES]
    validate_env_for_stages(stages, cfg=cfg_obj)
    compute_init(device_type=axbench_cli._device_type_from_cfg(cfg_obj.device))
    state: _ProfileState | None = None
    wrappers: _InstalledWrappers | None = None
    prof: Any | None = None
    status = "ok"
    try:
        ddp_rank, ddp_world_size = axbench_cli._resolve_ddp_coordinates()
        output_dir = output_paths.profile_dir() / "stage0c_verify_candidate"
        state = _ProfileState(verify_limit=verify_limit, output_dir=output_dir)
        wrappers = _InstalledWrappers(state)
        runtime_state = RuntimeState(
            cfg=cfg_obj,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )

        try:
            activities = [ProfilerActivity.CPU]
            if t.cuda.is_available():
                activities.append(ProfilerActivity.CUDA)
            cprofiler = cProfile.Profile()
            try:
                wrappers.install()
                with torch_profile(
                    activities=activities,
                    record_shapes=record_shapes,
                    profile_memory=profile_memory,
                    with_stack=False,
                ) as prof:
                    cprofiler.enable()
                    try:
                        axbench_cli._dispatch_stage(cfg_obj, runtime_state)
                    except _Stage0cProfileComplete:
                        status = "stopped_after_limit"
                    finally:
                        cprofiler.disable()
            finally:
                wrappers.restore()
        finally:
            runtime_state.close()
    except BaseException:
        status = "error"
        raise
    finally:
        try:
            if prof is not None and state is not None:
                _write_reports(
                    cfg=cfg_obj,
                    state=state,
                    status=status,
                    torch_prof=prof,
                    cprofiler=cprofiler,
                )
        finally:
            compute_cleanup()


@hydra.main(
    version_base=None,
    config_path="../../configs/eval/axbench_base",
    config_name="axbench_base",
)
def main(cfg: DictConfig) -> None:
    _run(cfg)


__all__ = ["_run", "main"]


if __name__ == "__main__":
    main()
