"""Phase-instrumented benchmark for autointerp token-example materialization (#678).

Drives CachedTokenExampleProvider + TokenExampleSampler directly (no runner,
no judge, no network). Instrumentation wraps module attributes / instance
methods from THIS script only; production modules are unmodified. Run one
path-variant per process so ru_maxrss and CUDA peaks stay attributable.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import platform
import resource
import sqlite3
import subprocess
import threading
import time
from collections.abc import Callable, Iterator, Sequence
from contextlib import closing, contextmanager
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import torch as t
from hydra import compose, initialize_config_dir

import steering.eval.autointerp_filter.cache as cache_module
import steering.eval.autointerp_filter.source_index as source_index_module
import steering.eval.autointerp_filter.token_examples as token_examples_module
import steering.eval.autointerp_filter.types as types_module
from steering.eval.autointerp_filter.batch_validation import (
    BatchDifferenceMetrics,
    BatchToleranceProfile,
    assert_batch_metrics_within_profile,
)
from steering.eval.autointerp_filter.batching import build_target_blocks
from steering.eval.autointerp_filter.cli import build_config
from steering.eval.autointerp_filter.config import AutoInterpFilterConfig, validate_config
from steering.eval.autointerp_filter.parity import (
    compare_rendered_trial_surfaces,
    compare_token_example_surface,
    compare_trial_surfaces,
    load_token_example_set_by_target,
    sqlite_ro_uri,
)
from steering.eval.autointerp_filter.targets import resolve_targets
from steering.eval.autointerp_filter.token_examples import (
    CachedTokenExampleProvider,
    TokenExampleSampler,
    infer_n_sae_features,
)
from steering.eval.autointerp_filter.types import AutoInterpDataError, ScoringTarget

_NS = 1_000_000_000

@dataclass
class PhaseStat:
    inclusive_ns: int = 0
    exclusive_ns: int = 0
    count: int = 0


@dataclass
class _Frame:
    """Typed timer frame (a heterogeneous list is not provably numeric for ty)."""

    start_ns: int
    child_ns: int = 0


class PhaseRecorder:
    """Nested-aware flat phase timer: exclusive = inclusive minus child time.

    ``cuda_device`` scopes every synchronize to the benchmarked device so
    ``--device cuda:N`` runs measure the right stream (argument-less
    ``torch.cuda.synchronize()`` would sync device 0 only).
    """

    def __init__(self, cuda_device: t.device | None = None) -> None:
        self._stats: dict[str, PhaseStat] = {}
        self._local = threading.local()
        self._stats_lock = threading.Lock()
        self._cuda_device = cuda_device

    def _thread_stack(self) -> list[_Frame]:
        stack = getattr(self._local, "stack", None)
        if stack is None:
            stack = []
            self._local.stack = stack
        return stack

    def _sync(self) -> None:
        if self._cuda_device is not None and t.cuda.is_available():
            t.cuda.synchronize(self._cuda_device)

    @contextmanager
    def phase(self, name: str, *, cuda_sync: bool = False) -> Iterator[None]:
        if cuda_sync:
            self._sync()
        frame = _Frame(start_ns=time.perf_counter_ns())
        stack = self._thread_stack()
        stack.append(frame)
        try:
            yield
        finally:
            # The post-sync may itself raise (async CUDA errors surface at
            # sync): stack/stat bookkeeping must still run or every ancestor
            # frame pops wrong afterwards.
            try:
                if cuda_sync:
                    self._sync()
            finally:
                stack.pop()
                duration = time.perf_counter_ns() - frame.start_ns
                if stack:
                    stack[-1].child_ns += duration
                with self._stats_lock:
                    stat = self._stats.setdefault(name, PhaseStat())
                    stat.inclusive_ns += duration
                    stat.exclusive_ns += duration - frame.child_ns
                    stat.count += 1

    def wrap(self, fn: Callable, name: str, *, cuda_sync: bool = False) -> Callable:
        @functools.wraps(fn)  # sets __wrapped__ = fn itself; no manual assignment
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            with self.phase(name, cuda_sync=cuda_sync):
                return fn(*args, **kwargs)

        return wrapped

    def snapshot(self) -> dict[str, tuple[int, int, int]]:
        with self._stats_lock:
            return {
                name: (stat.inclusive_ns, stat.exclusive_ns, stat.count)
                for name, stat in self._stats.items()
            }

    def calibrate_wrapper_overhead_ns(self, iters: int = 100_000) -> float:
        probe = self.wrap(lambda: None, "_calibration")
        start = time.perf_counter_ns()
        for _ in range(iters):
            probe()
        per_call = (time.perf_counter_ns() - start) / iters
        with self._stats_lock:
            self._stats.pop("_calibration", None)
        return per_call


class TokenizerTimingProxy:
    """Delegating proxy that times only .decode; all else passes through."""

    def __init__(self, inner: Any, recorder: PhaseRecorder) -> None:
        self._inner = inner
        self._recorder = recorder

    def decode(self, *args: Any, **kwargs: Any) -> Any:
        with self._recorder.phase("token_decode"):
            return self._inner.decode(*args, **kwargs)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self._inner(*args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)


def positive_ref_counts(index_path: Path) -> list[tuple[int, int]]:
    if not index_path.is_file():
        raise AutoInterpDataError(
            f"--select-targets needs an existing source index, none at {index_path}"
        )
    with closing(sqlite3.connect(sqlite_ro_uri(index_path), uri=True)) as conn:
        rows = conn.execute(
            "SELECT feature_id, COUNT(*) FROM positive_refs GROUP BY feature_id ORDER BY feature_id"
        ).fetchall()
    return [(int(fid), int(count)) for fid, count in rows]




def _terciles(ordered: list[int]) -> dict[int, int]:
    n = len(ordered)
    return {fid: min(2, (3 * idx) // n) for idx, fid in enumerate(ordered)}


def select_benchmark_targets(
    counts: Sequence[tuple[int, int]],
    n: int,
    include: Sequence[int] = (0,),
    allowed: set[int] | None = None,
) -> tuple[int, ...]:
    """``allowed`` restricts candidates to the config's RESOLVED target ids —
    an index built for a superset whitelist would otherwise nominate ids the
    run cannot benchmark."""
    nonzero = sorted(
        (int(f), int(c)) for f, c in counts if int(c) > 0 and (allowed is None or int(f) in allowed)
    )
    present = {fid for fid, _count in nonzero}
    missing = [fid for fid in include if fid not in present]
    if missing:
        raise AutoInterpDataError(
            f"include ids absent from the benchmarkable candidate set: {missing} "
            "(not positive in the index, or excluded by the config's resolved "
            "targets; pass --targets explicitly for such configs)"
        )
    if n < len(include) or n > len(nonzero):
        raise AutoInterpDataError(f"n={n} outside [{len(include)}, {len(nonzero)}] for this index")
    by_id = [fid for fid, _count in nonzero]
    by_density = [fid for fid, _count in sorted(nonzero, key=lambda fc: (fc[1], fc[0]))]
    id_terc, dens_terc = _terciles(by_id), _terciles(by_density)
    cells: dict[tuple[int, int], list[int]] = {}
    for fid in by_id:
        cells.setdefault((id_terc[fid], dens_terc[fid]), []).append(fid)
    selected: list[int] = sorted({int(fid) for fid in include})
    chosen = set(selected)
    cell_order = [(i, j) for i in range(3) for j in range(3)]
    while len(selected) < n:
        progressed = False
        for cell in cell_order:
            bucket = cells.get(cell, [])
            pick = next((fid for fid in bucket if fid not in chosen), None)
            if pick is None:
                continue
            selected.append(pick)
            chosen.add(pick)
            progressed = True
            if len(selected) == n:
                break
        if not progressed:
            raise AutoInterpDataError("target selection exhausted cells before reaching n")
    return tuple(sorted(selected))




def build_anchor_profile_observations(
    report: dict[str, Any],
    *,
    profile: BatchToleranceProfile,
) -> list[dict[str, Any]]:
    """Extract only predeclared-anchor surfaces after exact output checks.

    Startup probes may include block-filler targets, so they are intentionally
    excluded. Periodic/calibration shadows cover every declared anchor row;
    final token-window metrics additionally calibrate the exact downstream
    surface on which p99/mean acceptance is enforced.
    """
    batch_validation = report.get("batch_validation")
    parity = report.get("parity")
    if not isinstance(batch_validation, dict) or not isinstance(parity, dict):
        raise AutoInterpDataError(
            "anchor profile observations require batch-validation and parity reports"
        )
    raw_observations = batch_validation.get("observations")
    parity_targets = parity.get("per_target")
    if not isinstance(raw_observations, list) or not isinstance(parity_targets, dict):
        raise AutoInterpDataError("anchor profile report surfaces are malformed")

    observations = [
        dict(observation)
        for observation in raw_observations
        if isinstance(observation, dict)
        and observation.get("surface")
        in {"selected_sae_periodic_shadow", "raw_residual_calibration_shadow"}
    ]
    observed_kinds = {observation.get("surface_kind") for observation in observations}
    if observed_kinds != {"selected_sae", "raw_residual"}:
        raise AutoInterpDataError(
            "anchor validation did not capture both selected-SAE and raw-residual shadows"
        )
    if not all(isinstance(target_key, str) for target_key in parity_targets):
        raise AutoInterpDataError("parity report target keys must be strings")
    expected_target_keys = set(cast(dict[str, object], parity_targets))
    for surface_kind in ("selected_sae", "raw_residual"):
        raw_observed_target_keys = [
            observation.get("target_key")
            for observation in observations
            if observation.get("surface_kind") == surface_kind
        ]
        if not all(isinstance(target_key, str) for target_key in raw_observed_target_keys):
            raise AutoInterpDataError(
                f"anchor validation {surface_kind} target keys must be strings"
            )
        observed_target_keys = set(cast(list[str], raw_observed_target_keys))
        if observed_target_keys != expected_target_keys:
            missing = sorted(expected_target_keys - observed_target_keys)
            extra = sorted(
                str(target_key) for target_key in observed_target_keys - expected_target_keys
            )
            raise AutoInterpDataError(
                f"anchor validation {surface_kind} target coverage mismatch: "
                f"missing={missing} extra={extra}"
            )

    for target_key in sorted(parity_targets):
        target_report = parity_targets[target_key]
        if not isinstance(target_report, dict):
            raise AutoInterpDataError(f"parity report for {target_key!r} is malformed")
        example_report = target_report.get("examples")
        trial_report = target_report.get("trials")
        rendered_report = target_report.get("rendered_prompts")
        if (
            not isinstance(example_report, dict)
            or not isinstance(trial_report, dict)
            or not isinstance(rendered_report, dict)
        ):
            raise AutoInterpDataError(f"parity surfaces for {target_key!r} are incomplete")
        if (
            trial_report.get("reference_count") != trial_report.get("candidate_count")
            or trial_report.get("same_prompt_surface_count") != trial_report.get("reference_count")
            or rendered_report.get("all_rendered_examples_exact") is not True
            or rendered_report.get("all_message_bytes_exact") is not True
        ):
            raise AutoInterpDataError(
                f"anchor {target_key!r} changed trial or rendered prompt bytes"
            )
        for group_name in ("positives", "non_activating"):
            group = example_report.get(group_name)
            if not isinstance(group, dict):
                raise AutoInterpDataError(f"anchor {target_key!r} is missing {group_name} parity")
            reference_count = group.get("reference_count")
            if (
                group.get("identity_set_equal") is not True
                or group.get("candidate_count") != reference_count
                or group.get("matched_window_count") != reference_count
                or group.get("same_identity_order_count") != reference_count
                or group.get("quantile_exact_count") != reference_count
                or group.get("highlight_exact_count") != reference_count
            ):
                raise AutoInterpDataError(
                    f"anchor {target_key!r} changed the exact {group_name} window surface"
                )
            for metric_name in ("activation", "max_activation"):
                metric = group.get(metric_name)
                if not isinstance(metric, dict):
                    raise AutoInterpDataError(
                        f"anchor {target_key!r} is missing {group_name}/{metric_name} metrics"
                    )
                if metric.get("reference_values") == metric.get("candidate_values") == 0:
                    if (
                        metric.get("support_mismatches") != 0
                        or metric.get("max_abs") != 0.0
                        or metric.get("p99_abs") != 0.0
                        or metric.get("mean_abs") != 0.0
                    ):
                        raise AutoInterpDataError(
                            f"empty anchor metric {target_key!r}/{group_name}/{metric_name} "
                            "has nonzero difference values"
                        )
                    continue
                if metric.get("candidate_values") != metric.get("reference_values"):
                    raise AutoInterpDataError(
                        f"anchor metric {target_key!r}/{group_name}/{metric_name} "
                        "changed value count"
                    )
                metrics = BatchDifferenceMetrics(
                    count=metric.get("reference_values"),
                    support_mismatches=metric.get("support_mismatches"),
                    max_abs=metric.get("max_abs"),
                    p99_abs=metric.get("p99_abs"),
                    mean_abs=metric.get("mean_abs"),
                )
                context = f"output target={target_key} group={group_name} metric={metric_name}"
                assert_batch_metrics_within_profile(
                    metrics,
                    profile=profile,
                    context=context,
                    surface_kind="selected_sae",
                )
                observations.append(
                    {
                        "target_key": target_key,
                        "surface": f"selected_sae_output_{group_name}_{metric_name}",
                        "surface_kind": "selected_sae",
                        "context": context,
                        "require_support_bound": True,
                        **asdict(metrics),
                    }
                )
    return observations




def build_steady_state_measurement(
    *,
    targets: Sequence[ScoringTarget],
    materialization_batch_size: int,
    per_target: Sequence[dict[str, Any]],
    batch_pair_timings: Sequence[dict[str, object]],
) -> dict[str, Any]:
    """Exclude the first two feature blocks and report like-for-like build time.

    The first canonical block pair contains CUDA/kernel warmup, the candidate's
    startup correctness gate, and the serial path's first source-index load.
    Removing exactly those target keys from both variants defines the approved
    steady-state unit without estimating or subtracting overhead.
    """
    block_size = 128 if materialization_batch_size == 1 else materialization_batch_size
    blocks = build_target_blocks(list(targets), batch_size=block_size)
    warmup_blocks = blocks[:2]
    warmup_keys = tuple(target.target_key for block in warmup_blocks for target in block.targets)
    measured_keys = tuple(target.target_key for block in blocks[2:] for target in block.targets)
    if materialization_batch_size == 1:
        rows_by_key = {str(row["target_key"]): row for row in per_target}
        if set(rows_by_key) != {target.target_key for target in targets}:
            raise AutoInterpDataError(
                "serial steady-state measurement requires one timing row per target"
            )
        elapsed_s = sum(
            float(rows_by_key[target_key]["phase_deltas"]["examples"]["inclusive_s"])
            for target_key in measured_keys
        )
    else:
        if len(batch_pair_timings) != (len(blocks) + 1) // 2:
            raise AutoInterpDataError(
                "batch steady-state timing count does not match canonical block pairs"
            )
        normalized_timings: list[tuple[tuple[str, ...], int]] = []
        for index, timing in enumerate(batch_pair_timings):
            target_keys = timing.get("target_keys")
            elapsed_ns = timing.get("elapsed_ns")
            scope = timing.get("scope")
            if (
                not isinstance(target_keys, tuple)
                or any(not isinstance(target_key, str) for target_key in target_keys)
                or isinstance(elapsed_ns, bool)
                or not isinstance(elapsed_ns, int)
                or elapsed_ns <= 0
                or scope != "complete_bulk_partition_v1"
            ):
                raise AutoInterpDataError(f"batch steady-state timing {index} is malformed")
            normalized_timings.append(
                (tuple(str(target_key) for target_key in target_keys), elapsed_ns)
            )
        timed_keys = tuple(
            str(target_key)
            for target_keys, _elapsed_ns in normalized_timings
            for target_key in target_keys
        )
        if timed_keys != (*warmup_keys, *measured_keys):
            raise AutoInterpDataError(
                "batch steady-state timings do not cover the canonical target plan"
            )
        elapsed_s = sum(elapsed_ns / _NS for _target_keys, elapsed_ns in normalized_timings[1:])
    available = bool(measured_keys) and elapsed_s > 0.0
    return {
        "policy": "exclude_first_canonical_two_feature_blocks_v1",
        "block_size": block_size,
        "warmup_block_ids": [block.block_id for block in warmup_blocks],
        "warmup_target_keys": list(warmup_keys),
        "measured_target_keys_sha256": hashlib.sha256(
            json.dumps(measured_keys, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "target_count": len(measured_keys),
        "elapsed_s": elapsed_s,
        "targets_per_second": len(measured_keys) / elapsed_s if available else None,
        "available": available,
    }


@dataclass
class InstrumentationHandle:
    recorder: PhaseRecorder
    _restores: list[Callable[[], None]] = field(default_factory=list)

    def add(self, restore: Callable[[], None]) -> None:
        self._restores.append(restore)

    def restore(self) -> None:
        for restore in reversed(self._restores):
            restore()
        self._restores.clear()


def _patch_module_attr(
    handle: InstrumentationHandle,
    module: Any,
    attr: str,
    wrapped: Callable,
) -> None:
    original = getattr(module, attr)
    setattr(module, attr, wrapped)
    handle.add(lambda: setattr(module, attr, original))


def _patch_instance_method(
    handle: InstrumentationHandle,
    obj: Any,
    attr: str,
    recorder: PhaseRecorder,
    name: str,
    *,
    cuda_sync: bool = False,
) -> None:
    bound = getattr(obj, attr)
    obj.__dict__[attr] = recorder.wrap(bound, name, cuda_sync=cuda_sync)
    handle.add(lambda: obj.__dict__.pop(attr, None))


def install_instrumentation(
    provider: CachedTokenExampleProvider,
    *,
    recorder: PhaseRecorder,
    tier: int,
    cuda_sync: bool,
) -> InstrumentationHandle:
    if tier not in (1, 2):
        raise AutoInterpDataError(f"instrumentation tier must be 1 or 2, got {tier}")
    handle = InstrumentationHandle(recorder=recorder)
    try:
        _install_all_seams(
            provider,
            handle=handle,
            recorder=recorder,
            tier=tier,
            cuda_sync=cuda_sync,
        )
    except BaseException:
        handle.restore()
        raise
    return handle


def _install_all_seams(
    provider: CachedTokenExampleProvider,
    *,
    handle: InstrumentationHandle,
    recorder: PhaseRecorder,
    tier: int,
    cuda_sync: bool,
) -> None:
    module_seams: list[tuple[Any, str, str, bool]] = [
        (token_examples_module, "load_or_build_source_index", "source_index_load", False),
        (source_index_module, "_load_index", "sqlite_row_load", False),
        (token_examples_module, "_shard_row_group_offsets", "parquet_text_load", False),
        (token_examples_module, "_load_shard_row_group", "parquet_text_load", False),
        (token_examples_module, "_truncate_to_training_window", "tokenization", False),
        (token_examples_module, "_truncate_to_focused_window", "tokenization", False),
        (token_examples_module, "_get_activations_native", "model_forward", cuda_sync),
        (token_examples_module, "_project_selected_sae_activations", "sae_projection", cuda_sync),
    ]
    for module, attr, phase, sync in module_seams:
        _patch_module_attr(
            handle,
            module,
            attr,
            recorder.wrap(getattr(module, attr), phase, cuda_sync=sync),
        )
    _patch_module_attr(
        handle,
        cache_module.TokenExampleCache,
        "__init__",
        recorder.wrap(cache_module.TokenExampleCache.__init__, "token_cache_open"),
    )
    _patch_module_attr(
        handle,
        cache_module.TokenExampleCache,
        "get",
        recorder.wrap(cache_module.TokenExampleCache.get, "token_cache_read"),
    )
    _patch_module_attr(
        handle,
        cache_module.TokenExampleCache,
        "set",
        recorder.wrap(cache_module.TokenExampleCache.set, "token_cache_write"),
    )
    _patch_module_attr(
        handle,
        cache_module.TokenExampleCache,
        "get_many",
        recorder.wrap(cache_module.TokenExampleCache.get_many, "token_cache_read"),
    )
    _patch_module_attr(
        handle,
        cache_module.TokenExampleCache,
        "set_many",
        recorder.wrap(cache_module.TokenExampleCache.set_many, "token_cache_write"),
    )
    _patch_instance_method(handle, provider, "_apply_source_index", recorder, "sqlite_apply")
    _patch_instance_method(
        handle, provider, "_positive_windows_for_doc", recorder, "window_selection"
    )
    _patch_instance_method(
        handle, provider, "_non_activating_windows_for_doc", recorder, "window_selection"
    )
    _patch_instance_method(handle, provider, "_doc_activations", recorder, "doc_activations")
    if tier == 2:
        _patch_module_attr(
            handle,
            types_module.TokenWindowRef,
            "stable_key",
            recorder.wrap(types_module.TokenWindowRef.stable_key, "stable_key_hash_window"),
        )
        _patch_module_attr(
            handle,
            types_module.SourceDocRef,
            "stable_key",
            recorder.wrap(types_module.SourceDocRef.stable_key, "stable_key_hash_docref"),
        )


def install_model_instrumentation(
    provider: CachedTokenExampleProvider,
    *,
    handle: InstrumentationHandle,
    tier: int,
    cuda_sync: bool = False,
) -> None:
    recorder = handle.recorder
    if provider._model is not None:
        # The TL forward launches kernels and returns; without a sync here a
        # cuda+transformerlens run would attribute the whole GPU forward to
        # whichever frame is open at the NEXT sync point.
        _patch_instance_method(
            handle,
            provider._model,
            "run_with_cache",
            recorder,
            "model_forward",
            cuda_sync=cuda_sync,
        )
    if tier == 2 and provider._tokenizer is not None:
        original = provider._tokenizer
        provider._tokenizer = TokenizerTimingProxy(original, recorder)

        def _restore_tokenizer() -> None:
            provider._tokenizer = original

        handle.add(_restore_tokenizer)


REQUIRED_REPORT_FIELDS: frozenset[str] = frozenset(
    {
        "source_index_load_elapsed_s",
        "sqlite_row_load_elapsed_s",
        "sqlite_apply_elapsed_s",
        "parquet_text_load_elapsed_s",
        "tokenization_elapsed_s",
        "token_decode_elapsed_s",
        "model_forward_elapsed_s",
        "sae_projection_elapsed_s",
        "window_selection_elapsed_s",
        "doc_activations_elapsed_s",
        "stable_key_hash_elapsed_s",
        "token_cache_open_elapsed_s",
        "token_cache_read_elapsed_s",
        "token_cache_write_elapsed_s",
        "trial_build_elapsed_s",
        "prepare_elapsed_s",
        "model_load_elapsed_s",
        "examples_elapsed_s",
        "ru_maxrss_kib",
        "cuda_max_memory_allocated_bytes",
        "cuda_max_memory_reserved_bytes",
        "phases",
        "per_target",
        "parity",
        "preconditions",
        "environment",
        "config",
        "git",
        "instrumentation",
        "targets",
        "benchmark",
    }
)


def _phase_seconds(snapshot: dict[str, tuple[int, int, int]], name: str) -> float:
    inclusive_ns, _exclusive_ns, _count = snapshot.get(name, (0, 0, 0))
    return inclusive_ns / _NS


def build_report(
    *,
    recorder: PhaseRecorder,
    per_target: list[dict],
    meta: dict,
    preconditions: dict,
    parity: dict | None,
    instrumentation: dict,
    cuda_device: t.device | None,
) -> dict:
    """Build a JSON-safe benchmark report from recorder and run metadata.

    ``cuda_device`` must be derived from the configured device (None for CPU
    variants) rather than CUDA availability, so CPU runs never initialize a
    CUDA context and cuda:N runs read the right device's peaks. Every flat
    ``*_elapsed_s`` field is an INCLUSIVE phase time (they overlap; use
    ``phases[*]["exclusive_s"]`` for non-overlapping sums).
    """
    snapshot = recorder.snapshot()
    if cuda_device is not None:
        allocated: int | None = int(t.cuda.max_memory_allocated(cuda_device))
        reserved: int | None = int(t.cuda.max_memory_reserved(cuda_device))
    else:
        allocated = reserved = None
    report: dict[str, Any] = {
        "phases": {
            name: {
                "inclusive_s": inc / _NS,
                "exclusive_s": exc / _NS,
                "count": count,
            }
            for name, (inc, exc, count) in sorted(snapshot.items())
        },
        "per_target": per_target,
        "parity": parity,
        "preconditions": preconditions,
        "instrumentation": instrumentation,
        "ru_maxrss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_max_memory_allocated_bytes": allocated,
        "cuda_max_memory_reserved_bytes": reserved,
        "stable_key_hash_elapsed_s": (
            _phase_seconds(snapshot, "stable_key_hash_window")
            + _phase_seconds(snapshot, "stable_key_hash_docref")
        ),
        **meta,
    }
    for name in (
        "prepare",
        "model_load",
        "source_index_load",
        "sqlite_row_load",
        "sqlite_apply",
        "parquet_text_load",
        "tokenization",
        "token_decode",
        "model_forward",
        "sae_projection",
        "window_selection",
        "doc_activations",
        "token_cache_open",
        "token_cache_read",
        "token_cache_write",
        "trial_build",
        "examples",
    ):
        report[f"{name}_elapsed_s"] = _phase_seconds(snapshot, name)
    missing = REQUIRED_REPORT_FIELDS - set(report)
    if missing:
        raise AutoInterpDataError(f"benchmark report missing fields: {sorted(missing)}")
    return report


def _require_existing_source_index(cfg: AutoInterpFilterConfig) -> None:
    if not cfg.source_index.enabled:
        return
    if cfg.source_index.force_rebuild:
        raise AutoInterpDataError(
            "benchmark forbids source_index.force_rebuild=true: provider prepare "
            "would eagerly UNLINK and rebuild the production index "
            "(token_examples.py:497-500, source_index.py:531-533)"
        )
    if cfg.source_index.shared_cache_dir is not None:
        raise AutoInterpDataError(
            "benchmark forbids source_index.shared_cache_dir: the request would "
            "resolve to hash-named files under that dir (source_index.py:571-582), "
            "bypassing this existence check and possibly building a missing entry; "
            "point source_index.path/manifest_path at the index directly"
        )
    missing = [
        str(path)
        for path in (Path(cfg.source_index.path), Path(cfg.source_index.manifest_path))
        if not path.is_file()
    ]
    if missing:
        raise AutoInterpDataError(
            "benchmark requires an ALREADY-BUILT source index (a missing path "
            f"would trigger an implicit in-process rebuild): missing {missing}"
        )
    index_path = Path(cfg.source_index.path)
    hot = [
        str(sidecar)
        for suffix in ("-journal", "-wal", "-shm")
        if (sidecar := index_path.with_name(index_path.name + suffix)).exists()
    ]
    if hot:
        raise AutoInterpDataError(
            f"source index has hot sqlite sidecar files {hot}: the production "
            "loaders open the index WRITABLE and would roll the journal back, "
            "mutating a read-only benchmark input; resolve the stale "
            "journal/WAL first"
        )


def _require_fresh_cache(cfg: AutoInterpFilterConfig) -> None:
    """Any pre-existing cache file — matching rows, disjoint targets, or a
    foreign/empty sqlite — means the output dir is not fresh and the run's
    cache/size/parity semantics would be polluted."""
    path = Path(cfg.token_example_cache_path)
    if path.exists():
        raise AutoInterpDataError(
            f"{path} already exists; benchmark runs must be token-cache misses "
            "in a fresh output dir so the cache stays single-variant"
        )


def _require_reference_cache(path: Path, target_keys: list[str]) -> None:
    """Fail BEFORE hours of materialization, not after: the reference cache
    must exist and hold exactly one namespace's rows for every selected target."""
    if not path.is_file():
        raise AutoInterpDataError(f"--reference-cache does not exist: {path}")
    try:
        with closing(sqlite3.connect(sqlite_ro_uri(path), uri=True)) as conn:
            rows = conn.execute(
                "SELECT target_key, COUNT(DISTINCT namespace) "
                "FROM token_example_sets GROUP BY target_key"
            ).fetchall()
    except sqlite3.OperationalError as exc:
        raise AutoInterpDataError(
            f"--reference-cache {path} is not a readable token example cache: {exc}"
        ) from exc
    by_target = {str(key): int(count) for key, count in rows}
    missing = sorted(key for key in target_keys if key not in by_target)
    if missing:
        raise AutoInterpDataError(f"--reference-cache {path} is missing target keys: {missing}")
    ambiguous = sorted(key for key in target_keys if by_target[key] > 1)
    if ambiguous:
        raise AutoInterpDataError(
            f"--reference-cache {path} has multiple namespaces for: {ambiguous}"
        )


def _stat_tuple(path: Path) -> tuple[int, int] | None:
    if not path.exists():
        return None
    stat = path.stat()
    return (stat.st_size, stat.st_mtime_ns)


def _require_unchanged_stat(
    *,
    label: str,
    before: tuple[int, int] | None,
    after: tuple[int, int] | None,
) -> None:
    if before != after:
        raise AutoInterpDataError(f"{label} stat changed during benchmark: {before} -> {after}")


def _phase_delta(
    before: dict[str, tuple[int, int, int]],
    after: dict[str, tuple[int, int, int]],
) -> dict[str, dict[str, float | int]]:
    delta: dict[str, dict[str, float | int]] = {}
    for name in sorted(set(before) | set(after)):
        before_inc, before_exc, before_count = before.get(name, (0, 0, 0))
        after_inc, after_exc, after_count = after.get(name, (0, 0, 0))
        delta[name] = {
            "inclusive_s": (after_inc - before_inc) / _NS,
            "exclusive_s": (after_exc - before_exc) / _NS,
            "count": after_count - before_count,
        }
    return delta


def _proc_meminfo() -> tuple[int | None, int | None]:
    values: dict[str, int] = {}
    try:
        for line in Path("/proc/meminfo").read_text().splitlines():
            key, value, *_unit = line.split()
            values[key.rstrip(":")] = int(value)
    except (OSError, ValueError):
        return None, None
    return values.get("MemTotal"), values.get("MemAvailable")


def _nvidia_compute_process_count() -> int:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return 0
    if result.returncode != 0:
        return 0
    return len([line for line in result.stdout.splitlines() if line.strip()])


def _environment_metadata(*, cuda_device: t.device | None) -> dict:
    mem_total_kib, mem_available_kib = _proc_meminfo()
    try:
        uptime_s: float | None = float(Path("/proc/uptime").read_text().split()[0])
    except (OSError, ValueError, IndexError):
        uptime_s = None
    return {
        "python_version": platform.python_version(),
        "torch_version": str(t.__version__),
        "torch_cuda_version": t.version.cuda,
        "hostname": platform.node(),
        "gpu_name": t.cuda.get_device_name(cuda_device) if cuda_device is not None else None,
        "load_average": list(os.getloadavg()),
        "uptime_seconds": uptime_s,
        "mem_total_kib": mem_total_kib,
        "mem_available_kib": mem_available_kib,
        "nvidia_compute_process_count": _nvidia_compute_process_count(),
    }


def _git_metadata(repo_root: Path) -> dict:
    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    status = subprocess.run(
        ["git", "status", "--porcelain"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {"sha": sha, "dirty": bool(status.strip())}


def _target_metadata(target: ScoringTarget) -> dict:
    return {
        "kind": target.kind.value,
        "feature_ids": list(target.feature_ids),
        "target_key": target.target_key,
        "display_name": target.display_name,
    }


def run_benchmark(
    cfg: AutoInterpFilterConfig,
    target_ids: Sequence[int],
    *,
    materialization_target_ids: Sequence[int] | None = None,
    tier: int = 2,
    reference_cache_path: Path | None = None,
    trial_n_positive: int = 2,
    trial_count: int = 8,
    trial_quantiles: int = 5,
    post_prepare_hook: Callable[[CachedTokenExampleProvider], None] | None = None,
    instrument: bool = True,
    materialization_prefetch: bool = True,
    calibration_mode: bool = False,
    calibration_profile: BatchToleranceProfile | None = None,
    validate_all_anchor_surfaces: bool = False,
) -> dict:
    if not isinstance(materialization_prefetch, bool):
        raise AutoInterpDataError("materialization_prefetch must be a strict bool")
    if not isinstance(calibration_mode, bool):
        raise AutoInterpDataError("calibration_mode must be a strict bool")
    if not isinstance(validate_all_anchor_surfaces, bool):
        raise AutoInterpDataError("validate_all_anchor_surfaces must be a strict bool")
    if calibration_mode and validate_all_anchor_surfaces:
        raise AutoInterpDataError(
            "calibration_mode already validates all anchor surfaces; do not set both modes"
        )
    if calibration_mode != (calibration_profile is not None):
        raise AutoInterpDataError(
            "calibration_profile may be injected only with calibration_mode=true, "
            "and calibration_mode requires an explicit profile"
        )
    if calibration_mode:
        if cfg.token_examples.materialization_batch_size <= 1:
            raise AutoInterpDataError("calibration_mode requires materialization batch size > 1")
        if reference_cache_path is None:
            raise AutoInterpDataError("calibration_mode requires a serial reference cache")
    if validate_all_anchor_surfaces:
        if cfg.token_examples.materialization_batch_size <= 1:
            raise AutoInterpDataError(
                "validate_all_anchor_surfaces requires materialization batch size > 1"
            )
        if reference_cache_path is None:
            raise AutoInterpDataError(
                "validate_all_anchor_surfaces requires a serial reference cache"
            )
    use_cuda = cfg.token_examples.device.startswith("cuda")
    feature_dir = Path(cfg.feature_dir)
    n_sae_features = infer_n_sae_features(feature_dir, cfg.n_sae_features)
    all_targets = resolve_targets(
        cfg,
        n_sae_features=n_sae_features,
        feature_dir=feature_dir,
    )
    targets_by_id = {
        target.feature_ids[0]: target for target in all_targets if len(target.feature_ids) == 1
    }
    unknown = [int(target_id) for target_id in target_ids if int(target_id) not in targets_by_id]
    if unknown:
        raise AutoInterpDataError(f"unknown benchmark target ids: {unknown}")
    if len({int(target_id) for target_id in target_ids}) != len(list(target_ids)):
        raise AutoInterpDataError(f"duplicate benchmark target ids: {list(target_ids)}")
    selected_targets = [targets_by_id[int(target_id)] for target_id in target_ids]
    if not selected_targets:
        raise AutoInterpDataError("benchmark requires at least one target id")
    resolved_materialization_ids = (
        list(target_ids)
        if materialization_target_ids is None
        else [int(target_id) for target_id in materialization_target_ids]
    )
    unknown_materialization = [
        target_id for target_id in resolved_materialization_ids if target_id not in targets_by_id
    ]
    if unknown_materialization:
        raise AutoInterpDataError(
            f"unknown benchmark materialization target ids: {unknown_materialization}"
        )
    if len(set(resolved_materialization_ids)) != len(resolved_materialization_ids):
        raise AutoInterpDataError(
            f"duplicate benchmark materialization target ids: {resolved_materialization_ids}"
        )
    materialization_targets = [
        targets_by_id[target_id] for target_id in resolved_materialization_ids
    ]
    if not materialization_targets:
        raise AutoInterpDataError("benchmark requires at least one materialization target")
    missing_anchors = sorted(
        {int(target_id) for target_id in target_ids} - set(resolved_materialization_ids)
    )
    if missing_anchors:
        raise AutoInterpDataError(
            "benchmark anchor targets must be included in materialization targets: "
            f"missing={missing_anchors}"
        )

    _require_existing_source_index(cfg)
    _require_fresh_cache(cfg)
    if reference_cache_path is not None:
        _require_reference_cache(
            reference_cache_path,
            [target.target_key for target in selected_targets],
        )
    git_meta = _git_metadata(Path(__file__).resolve().parents[2])

    if cfg.source_index.enabled:
        index_path: Path | None = Path(cfg.source_index.path)
        manifest_path: Path | None = Path(cfg.source_index.manifest_path)
    else:
        index_path = manifest_path = None
    index_before = _stat_tuple(index_path) if index_path is not None else None
    manifest_before = _stat_tuple(manifest_path) if manifest_path is not None else None
    preconditions: dict[str, Any] = {
        "source_index_required": cfg.source_index.enabled,
        "index_stat_before": index_before,
        "manifest_stat_before": manifest_before,
    }

    cuda_device = t.device(cfg.token_examples.device) if use_cuda else None
    if cuda_device is not None:
        t.cuda.reset_peak_memory_stats(cuda_device)

    recorder = PhaseRecorder(cuda_device=cuda_device)
    batch_pair_timings: list[dict[str, object]] = []
    provider = CachedTokenExampleProvider(
        cfg,
        feature_dir=feature_dir,
        n_sae_features=n_sae_features,
        materialization_prefetch=materialization_prefetch,
        batch_tolerance_profile=calibration_profile,
        batch_validation_anchor_target_keys=(
            frozenset(target.target_key for target in selected_targets)
            if calibration_mode or validate_all_anchor_surfaces
            else None
        ),
        batch_pair_timing_sink=batch_pair_timings,
        allow_unvalidated_batch_profile=calibration_mode,
    )
    handle = (
        install_instrumentation(
            provider,
            recorder=recorder,
            tier=tier,
            cuda_sync=use_cuda,
        )
        if instrument
        else InstrumentationHandle(recorder=recorder)
    )
    per_target: list[dict] = []
    candidate_examples: dict[str, Any] = {}
    try:
        with recorder.phase("prepare"):
            provider.prepare(all_targets, scoring_targets=materialization_targets)
        if post_prepare_hook is not None:
            post_prepare_hook(provider)
        with recorder.phase("model_load"):
            provider._ensure_model_loaded()
        if instrument:
            install_model_instrumentation(provider, handle=handle, tier=tier, cuda_sync=use_cuda)

        sampler_cfg = replace(
            cfg,
            n_positive_examples=trial_n_positive,
            n_trials_per_target=trial_count,
            n_quantiles=trial_quantiles,
        )
        sampler = TokenExampleSampler(sampler_cfg, provider=provider)
        batch_materialization = cfg.token_examples.materialization_batch_size > 1
        bulk_phase_delta: dict[str, dict[str, float | int]] | None = None
        if batch_materialization:
            before_bulk = recorder.snapshot()
            with recorder.phase("examples"):
                candidate_examples = provider.examples_for_targets(materialization_targets)
            bulk_phase_delta = _phase_delta(before_bulk, recorder.snapshot())

        for target in selected_targets:
            before = recorder.snapshot()
            if batch_materialization:
                examples = candidate_examples[target.target_key]
            else:
                with recorder.phase("examples"):
                    examples = provider.examples_for_target(target)
                candidate_examples[target.target_key] = examples
            with recorder.phase("trial_build"):
                trials = sampler._build_trials_from_examples(target, examples)
            after = recorder.snapshot()
            # MUST match TokenExampleCache.set's exact dump convention
            # (cache.py:196-201, ensure_ascii=False): the recorded #660 gate
            # hashes are sha256 of the raw stored payload text, and real
            # payloads contain non-ASCII token strings.
            payload = json.dumps(
                examples.to_json(),
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            )
            per_target.append(
                {
                    "target_key": target.target_key,
                    "namespace": provider._cache_namespace(
                        target,
                        serial_reference=not batch_materialization,
                    ),
                    "payload_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
                    "positive_windows": len(examples.positives),
                    "non_activating_windows": len(examples.non_activating),
                    "trials_built": len(trials),
                    "trial_keys_sha256": hashlib.sha256(
                        json.dumps([trial.trial_key for trial in trials]).encode("utf-8")
                    ).hexdigest(),
                    "phase_deltas": _phase_delta(before, after),
                    "bulk_materialization_phase_deltas": bulk_phase_delta,
                }
            )
    finally:
        handle.restore()
        index_after = _stat_tuple(index_path) if index_path is not None else None
        manifest_after = _stat_tuple(manifest_path) if manifest_path is not None else None
        preconditions["index_stat_after"] = index_after
        preconditions["manifest_stat_after"] = manifest_after
        preconditions["index_stat_unchanged"] = index_before == index_after
        preconditions["manifest_stat_unchanged"] = manifest_before == manifest_after
    # Raised OUTSIDE the finally: a stat mismatch must never mask the body's
    # real exception (e.g. a CUDA OOM) as the surfaced error.
    _require_unchanged_stat(label="source index", before=index_before, after=index_after)
    _require_unchanged_stat(
        label="source index manifest",
        before=manifest_before,
        after=manifest_after,
    )

    parity: dict | None = None
    if reference_cache_path is not None:
        parity_targets: dict[str, dict] = {}
        for target in selected_targets:
            reference_examples = load_token_example_set_by_target(
                reference_cache_path,
                target.target_key,
            )
            examples = candidate_examples[target.target_key]
            parity_targets[target.target_key] = {
                "examples": compare_token_example_surface(
                    reference_examples,
                    examples,
                ).to_json(),
                "trials": compare_trial_surfaces(
                    reference_examples,
                    examples,
                    n_positive_examples=trial_n_positive,
                    n_trials=trial_count,
                    n_quantiles=trial_quantiles,
                    target=target,
                    seed=cfg.seed,
                    example_ctx_len=cfg.token_examples.example_ctx_len,
                    activation_threshold_ratio=(cfg.token_examples.activation_threshold_ratio),
                ).to_json(),
                "rendered_prompts": compare_rendered_trial_surfaces(
                    reference_examples,
                    examples,
                    n_positive_examples=trial_n_positive,
                    n_trials=trial_count,
                    n_quantiles=trial_quantiles,
                    target=target,
                    seed=cfg.seed,
                    example_ctx_len=cfg.token_examples.example_ctx_len,
                    activation_threshold_ratio=(cfg.token_examples.activation_threshold_ratio),
                    cot=cfg.judge.cot,
                    prompt_variant=cfg.judge.prompt_variant,
                    few_shot_set=cfg.judge.few_shot_set,
                ).to_json(),
            }
        parity = {
            "reference_cache": str(reference_cache_path),
            "trial_params": {
                "n_positive_examples": trial_n_positive,
                "n_trials": trial_count,
                "n_quantiles": trial_quantiles,
            },
            "per_target": parity_targets,
        }

    wrapper_overhead = recorder.calibrate_wrapper_overhead_ns()
    snapshot = recorder.snapshot()
    instrumentation = {
        "tier": tier,
        "instrumented": instrument,
        "wrapper_overhead_ns_per_call": wrapper_overhead,
        "wrapper_overhead_excludes_cuda_sync": True,
        "stable_key_calls": (
            snapshot.get("stable_key_hash_window", (0, 0, 0))[2]
            + snapshot.get("stable_key_hash_docref", (0, 0, 0))[2]
        ),
        "stable_key_window_calls": snapshot.get("stable_key_hash_window", (0, 0, 0))[2],
        "stable_key_docref_calls": snapshot.get("stable_key_hash_docref", (0, 0, 0))[2],
        "decode_calls": snapshot.get("token_decode", (0, 0, 0))[2],
    }
    cache_path = Path(cfg.token_example_cache_path)
    steady_state = build_steady_state_measurement(
        targets=(
            selected_targets
            if cfg.token_examples.materialization_batch_size == 1
            else materialization_targets
        ),
        materialization_batch_size=(cfg.token_examples.materialization_batch_size),
        per_target=per_target,
        batch_pair_timings=batch_pair_timings,
    )
    profile_observations = (
        build_anchor_profile_observations(
            {
                "batch_validation": {
                    "observations": list(provider._batch_validation_observations),
                },
                "parity": parity,
            },
            profile=provider._batch_tolerance_profile,
        )
        if calibration_mode or validate_all_anchor_surfaces
        else None
    )
    meta = {
        "benchmark": {
            "name": "autointerp_token_examples_issue678",
            "tier": tier,
            "target_count": len(selected_targets),
            "materialization_target_count": len(materialization_targets),
            "materialization_prefetch": materialization_prefetch,
            "unvalidated_candidate": calibration_mode,
            "full_anchor_validation": calibration_mode or validate_all_anchor_surfaces,
            "steady_state": steady_state,
            "batch_pair_timings": batch_pair_timings,
        },
        "git": git_meta,
        "environment": _environment_metadata(cuda_device=cuda_device),
        "config": {
            "token_examples": asdict(cfg.token_examples),
            "source_index": {
                "enabled": cfg.source_index.enabled,
                "path": cfg.source_index.path,
                "manifest_path": cfg.source_index.manifest_path,
            },
            "seed": cfg.seed,
            "feature_dir": cfg.feature_dir,
            "data_dir": cfg.data_dir,
            "n_sae_features": n_sae_features,
            "n_targets_total": len(all_targets),
            "trial_params": {
                "n_positive_examples": trial_n_positive,
                "n_trials": trial_count,
                "n_quantiles": trial_quantiles,
            },
        },
        "targets": [_target_metadata(target) for target in selected_targets],
        "materialization_targets": [_target_metadata(target) for target in materialization_targets],
        "batch_validation": {
            "profile": (
                None
                if cfg.token_examples.materialization_batch_size == 1
                else provider._batch_tolerance_profile.identity()
            ),
            "unvalidated_candidate": calibration_mode,
            "observations": list(provider._batch_validation_observations),
            "profile_observations": profile_observations,
        },
        "token_cache_size_bytes": cache_path.stat().st_size if cache_path.exists() else 0,
    }
    return build_report(
        recorder=recorder,
        per_target=per_target,
        meta=meta,
        preconditions=preconditions,
        parity=parity,
        instrumentation=instrumentation,
        cuda_device=cuda_device,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark autointerp token-example materialization phases."
    )
    parser.add_argument(
        "--config-name",
        default="precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--feature-dir")
    parser.add_argument("--data-dir")
    parser.add_argument("--source-index-path")
    parser.add_argument("--source-index-manifest-path")
    parser.add_argument("--device")
    parser.add_argument("--activation-backend")
    parser.add_argument("--model-dtype")
    targets = parser.add_mutually_exclusive_group(required=True)
    targets.add_argument("--targets")
    targets.add_argument("--select-targets", type=int)
    parser.add_argument(
        "--materialization-targets",
        help=(
            "optional comma-separated superset materialized by the candidate; "
            "--targets remain the reported/reference anchors"
        ),
    )
    parser.add_argument(
        "--instrumentation",
        choices=("off", "tier1", "tier2"),
        default="tier2",
        help=(
            "off measures headline throughput without wrappers/CUDA sync; "
            "tier1 and tier2 enable progressively finer diagnostics"
        ),
    )
    parser.add_argument(
        "--materialization-prefetch",
        choices=("on", "off"),
        default="on",
        help="benchmark-only scheduling toggle; excluded from cache identity",
    )
    parser.add_argument(
        "--tier",
        type=int,
        choices=(1, 2),
        default=None,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--reference-cache")
    parser.add_argument("--trial-n-positive", type=int, default=2)
    parser.add_argument("--trial-count", type=int, default=8)
    parser.add_argument("--trial-quantiles", type=int, default=5)
    parser.add_argument("--report", required=True)
    parser.add_argument("--extra-override", action="append", default=[])
    args = parser.parse_args(argv)

    if args.tier is not None:
        if args.instrumentation != "tier2":
            parser.error("--tier cannot be combined with --instrumentation")
        args.instrumentation = f"tier{args.tier}"
    args.tier = 1 if args.instrumentation == "tier1" else 2
    args.instrument = args.instrumentation != "off"

    if bool(args.source_index_path) != bool(args.source_index_manifest_path):
        parser.error(
            "--source-index-path and --source-index-manifest-path must be provided together"
        )
    pinned_keys = {"output_dir", "token_example_cache_path", "neuronpedia.enabled"}
    if args.feature_dir is not None:
        pinned_keys.add("feature_dir")
    if args.data_dir is not None:
        pinned_keys.add("data_dir")
    if args.device is not None:
        pinned_keys.add("token_examples.device")
    if args.activation_backend is not None:
        pinned_keys.add("token_examples.activation_backend")
    if args.model_dtype is not None:
        pinned_keys.add("token_examples.model_dtype")
    for override in args.extra_override:
        # Normalize hydra override grammar: +/++ add-prefixes AND ~ deletions
        # (a bare '~source_index' silently flips the run onto the full-shard
        # scan path with every source-index guard no-oping).
        key = override.split("=", 1)[0].lstrip("+~")
        if key == "source_index" or key.startswith("source_index."):
            parser.error(
                "source_index --extra-override values (including +/++/~ forms) "
                "are forbidden; use the dedicated source-index flags"
            )
        if key in pinned_keys:
            parser.error(
                f"{key} must not be smuggled through --extra-override; hydra "
                "last-wins would silently override the dedicated flag/value"
            )
    if args.targets is not None:
        try:
            parsed_targets = [int(part) for part in args.targets.split(",") if part]
        except ValueError:
            parser.error("--targets must be a comma-separated list of integers")
        if not parsed_targets or len(parsed_targets) != len(args.targets.split(",")):
            parser.error("--targets must be a non-empty comma-separated list of integers")
        args.targets = parsed_targets
    if args.materialization_targets is not None:
        try:
            parsed_materialization_targets = [
                int(part) for part in args.materialization_targets.split(",") if part
            ]
        except ValueError:
            parser.error("--materialization-targets must be a comma-separated list of integers")
        if not parsed_materialization_targets or len(parsed_materialization_targets) != len(
            args.materialization_targets.split(",")
        ):
            parser.error(
                "--materialization-targets must be a non-empty comma-separated list of integers"
            )
        args.materialization_targets = parsed_materialization_targets
    return args


def build_cli_overrides(args: argparse.Namespace) -> list[str]:
    overrides = [
        f"output_dir={Path(args.output_dir).resolve()}",
        "neuronpedia.enabled=false",
    ]
    if args.feature_dir is not None:
        overrides.append(f"feature_dir={Path(args.feature_dir).resolve()}")
    if args.data_dir is not None:
        overrides.append(f"data_dir={Path(args.data_dir).resolve()}")
    if args.source_index_path is not None:
        overrides.extend(
            [
                "source_index.enabled=true",
                f"source_index.path={Path(args.source_index_path).resolve()}",
                (f"source_index.manifest_path={Path(args.source_index_manifest_path).resolve()}"),
            ]
        )
    if args.device is not None:
        overrides.append(f"token_examples.device={args.device}")
    if args.activation_backend is not None:
        overrides.append(f"token_examples.activation_backend={args.activation_backend}")
    if args.model_dtype is not None:
        overrides.append(f"token_examples.model_dtype={args.model_dtype}")
    overrides.extend(args.extra_override)
    return overrides


def compose_benchmark_config(
    config_name: str,
    overrides: list[str],
) -> AutoInterpFilterConfig:
    config_dir = Path(__file__).resolve().parents[2] / "configs" / "extraction"
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        composed = compose(config_name=config_name, overrides=overrides)
    cfg = build_config(composed)
    validate_config(cfg)
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = compose_benchmark_config(args.config_name, build_cli_overrides(args))
    if args.targets is not None:
        target_ids = args.targets
    else:
        if not cfg.source_index.enabled:
            raise AutoInterpDataError(
                "--select-targets requires source_index.enabled=true and an existing index"
            )
        n_sae_features = infer_n_sae_features(Path(cfg.feature_dir), cfg.n_sae_features)
        allowed = {
            target.feature_ids[0]
            for target in resolve_targets(
                cfg, n_sae_features=n_sae_features, feature_dir=Path(cfg.feature_dir)
            )
            if len(target.feature_ids) == 1
        }
        target_ids = list(
            select_benchmark_targets(
                positive_ref_counts(Path(cfg.source_index.path)),
                args.select_targets,
                include=(0,),
                allowed=allowed,
            )
        )
    report = run_benchmark(
        cfg,
        target_ids,
        materialization_target_ids=args.materialization_targets,
        tier=args.tier,
        reference_cache_path=(
            Path(args.reference_cache) if args.reference_cache is not None else None
        ),
        trial_n_positive=args.trial_n_positive,
        trial_count=args.trial_count,
        trial_quantiles=args.trial_quantiles,
        instrument=args.instrument,
        materialization_prefetch=args.materialization_prefetch == "on",
    )
    rendered = json.dumps(report, indent=2, sort_keys=True)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = report_path.with_name(f"{report_path.name}.tmp")
    tmp_path.write_text(f"{rendered}\n")
    os.replace(tmp_path, report_path)  # never leave truncated JSON at the advertised path
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
