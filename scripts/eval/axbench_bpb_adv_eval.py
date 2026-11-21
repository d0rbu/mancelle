"""CLI entry point for the AxBench-BPB adversarial robustness eval."""

from __future__ import annotations

import datetime as dt
import functools
import hashlib
import importlib
import json
import sys
from collections.abc import Callable, Mapping
from contextlib import nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, NoReturn, cast

import hydra
import pyarrow as pa
import pyarrow.parquet as pq
import torch as _torch
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig

from nanochat.common import compute_cleanup, compute_init
from nanochat.tokenizer import get_tokenizer
from scripts.eval.axbench_bpb_eval import (
    _device_type_from_cfg,
    _maybe_barrier,
)
from scripts.eval.axbench_bpb_eval import (
    _stage_prepare_context_cache as _run_bpb_context_cache_stage,
)
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.profiler import (
    AxBenchProfiler,
    get_current_profiler,
    profile_region,
    record_counter,
    resolve_profile_gpu_device_index,
)
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb.runtime_state import RuntimeState
from steering.eval.axbench_bpb.sample_loader import load_raw_synth_rows
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.tokenization import (
    assert_bos_lengths_match,
    assert_tokenizer_checksum_matches,
)
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv import config_loader as _config_loader
from steering.eval.axbench_bpb_adv.aggregation import (
    PRIMARY_MARGIN_YLABEL,
    ROUNDTRIP_PRIMARY_YLABEL,
    aggregate_bpb_adv_scores,
    aggregate_roundtrip_adv_scores,
    plot_bpb_adv_summaries,
    plot_roundtrip_adv_summaries,
    plot_steering_strength_families,
)
from steering.eval.axbench_bpb_adv.artifact_identity import (
    assert_table_cluster_identity,
    bind_clustered_table,
    load_clustered_candidate_snapshot,
    write_clustered_table_atomic,
)
from steering.eval.axbench_bpb_adv.attack_rows import (
    bpb_attack_sample_from_synth_row,
    pair_samples_into_prefixes,
    prepare_bpb_attack_rows,
    work_specs_from_method_registry,
)
from steering.eval.axbench_bpb_adv.cluster_preflight import (
    build_cluster_contract_request,
)
from steering.eval.axbench_bpb_adv.clustered_context_cache import (
    build_clustered_context_cache,
    canonicalize_prefix_population,
)
from steering.eval.axbench_bpb_adv.clustered_stage_source import (
    ClusteredStageSource,
    resolve_clustered_stage_source,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    Stage,
    SteeringMetric,
    validate_all_stage_representative_target_contract,
    validate_config,
)
from steering.eval.axbench_bpb_adv.direction_requirements import (
    requirements_from_config,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    evaluation_prefixes_for_scoring,
    write_evaluation_population,
)
from steering.eval.axbench_bpb_adv.general_val import run_general_val_bpb_stage
from steering.eval.axbench_bpb_adv.model_optimize import (
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.roundtrip_scoring import (
    run_model_backed_roundtrip_score_stage,
)
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    MACRO_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import (
    _bpb_config_from_adv_config,
    load_candidate_table,
    run_model_backed_bpb_adv_score_stage,
)
from steering.eval.axbench_bpb_adv.source_validation import (
    BpbAdvSourceValidation,
    _read_source_concept_ids,
    validate_stage_0_3_source_run,
    write_source_contract,
)
from steering.eval.cluster_config import axbench_bpb_adv_config_to_dict
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ensure_feature_contract_run_marker,
    resolve_cluster_feature_contract_ddp,
)

# `resource` is POSIX-only (absent on Windows). Resolve via `importlib` at
# module load so `_cpu_rss_kb` can degrade to None on non-POSIX platforms
# without using a deferred `import` statement (project rule: imports at top).
_resource: ModuleType | None
try:
    _resource = importlib.import_module("resource")
except ImportError:
    _resource = None


def _cpu_rss_kb() -> int | None:
    """Process peak RSS in kilobytes; ``None`` when ``resource`` is unavailable.

    Posix ``ru_maxrss`` is reported in KB on Linux and bytes on macOS — we
    normalize to KB. Used to attach a CPU-memory counter to each stage so the
    profiler captures memory growth alongside wall-clock time.
    """
    if _resource is None:
        return None
    try:
        ru = _resource.getrusage(_resource.RUSAGE_SELF)
        rss_kb = int(ru.ru_maxrss)
        if sys.platform == "darwin":
            rss_kb //= 1024
        return rss_kb
    except (OSError, AttributeError):
        return None


def _record_memory_counters(stage: str) -> None:
    """Emit per-stage CPU RSS + (optional) CUDA memory counters at stage exit.

    Called from the ``@_profiled`` decorator's exit hook (once per stage,
    NOT entry+exit). Each counter name is namespaced so the per-stage
    breakdown stays clean in the profiler's summary.csv.

    The CPU counter is process **peak** RSS (POSIX ``ru_maxrss``) since
    process start; the CUDA counters are **current** values
    (``memory_allocated`` / ``memory_reserved``) at stage exit. The
    profiler's per-region snapshot also captures peak CUDA automatically —
    these per-stage counters are complementary, surfacing the running totals
    in a single row of summary.csv.

    CUDA counters are recorded only when CUDA is available; on CPU-only
    hosts the helper still emits the CPU RSS counter.
    """
    rss_kb = _cpu_rss_kb()
    if rss_kb is not None:
        record_counter(f"axbench_bpb_adv.{stage}.cpu_rss_kb", rss_kb)
    try:
        if _torch.cuda.is_available():
            record_counter(
                f"axbench_bpb_adv.{stage}.cuda_allocated_bytes",
                int(_torch.cuda.memory_allocated()),
            )
            record_counter(
                f"axbench_bpb_adv.{stage}.cuda_reserved_bytes",
                int(_torch.cuda.memory_reserved()),
            )
    except Exception:  # noqa: BLE001 - profiling must never break the run
        pass


def _profiled(
    region_name: str,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that wraps a stage handler in `profile_region(region_name)`.

    **No measurable overhead when no profiler is bound:** the wrapper
    short-circuits immediately if ``get_current_profiler() is None``, fully
    eliminating the decorator's memory-probe machinery (``_cpu_rss_kb`` /
    ``_record_memory_counters``) and the wrapping ``profile_region`` — no
    syscalls, no allocations. The wrapped stage body still runs its inline
    ``record_counter`` / nested ``profile_region`` calls, but each of those is
    a single constant-time ``get_current_profiler()`` lookup that returns None
    and no-ops (the O(rows) token/file scans are additionally gated behind an
    explicit ``if get_current_profiler() is not None`` so they never run on the
    production path). Net effect: behaviour is identical and the residual cost
    is a handful of per-stage dict lookups, not per-row work. When the
    `scripts.benchmark.profile_axbench_bpb_adv` wrapper binds an
    `AxBenchProfiler`, every stage call shows up as a region in the summary
    with wall-clock + (optional) CUDA memory deltas attached automatically.

    A counterpart `_record_memory_counters` call after the wrapped function
    emits per-stage CPU/CUDA-memory counters, complementary to the per-region
    peak memory the profiler captures automatically.
    """

    def deco(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if get_current_profiler() is None:
                # Strict zero-overhead path: skip every profiler-only syscall
                # (cpu_rss / cuda probes) so the production run is byte-identical
                # to the un-instrumented form.
                return fn(*args, **kwargs)
            stage_attr = region_name.split(".", 1)[1] if "." in region_name else region_name
            with profile_region(region_name):
                result = fn(*args, **kwargs)
                _record_memory_counters(stage_attr)
                return result

        return wrapper

    return deco


def _generate_run_tag(cfg: AxBenchBpbAdvConfig) -> str:
    now = dt.datetime.now(tz=dt.UTC).strftime("%Y%m%d-%H%M%S")
    payload = json.dumps(axbench_bpb_adv_config_to_dict(cfg), sort_keys=True, default=str)
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"{now}-{digest}"


def _install_feature_contract_marker_ddp(
    cfg: AxBenchBpbAdvConfig,
    contract: ClusterFeatureContract | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Commit clustered ownership on rank 0 and broadcast any failure."""

    out_paths = _paths(cfg)
    marker = out_paths.root / "feature_contract.json"
    error: str | None = None
    if rank == 0:
        try:
            if contract is None:
                if marker.exists():
                    raise AxBenchBpbAdvError(
                        "unclustered BPB run rejects cluster-stamped output; choose a fresh run_tag"
                    )
            else:
                ensure_feature_contract_run_marker(out_paths.root, contract)
        except Exception as exc:
            logger.exception(
                "AxBench-BPB-Adv rank-0 feature-contract marker installation failed"
            )
            error = f"{type(exc).__name__}: {exc}"
    payload: list[str | None] = [error]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise AxBenchBpbAdvError(f"feature-contract run-marker validation failed: {payload[0]}")


def _preflight_source_ddp(
    cfg: AxBenchBpbAdvConfig,
    contract: ClusterFeatureContract | None,
    *,
    rank: int,
    world_size: int,
) -> BpbAdvSourceValidation | None:
    """Validate source identity and selected directions before any run marker."""

    error: str | None = None
    validation: BpbAdvSourceValidation | None = None
    if rank == 0:
        try:
            validation = validate_stage_0_3_source_run(
                cfg.source_run_dir,
                filter_unaccepted_samples=cfg.filter_unaccepted_samples,
            )
            if contract is not None and validation.contract.feature_contract_fingerprint != (
                contract.feature_contract_fingerprint
            ):
                raise AxBenchBpbAdvError(
                    "clustered source validation feature-contract fingerprint "
                    "mismatch: source="
                    f"{validation.contract.feature_contract_fingerprint!r}, "
                    f"target={contract.feature_contract_fingerprint!r}"
                )
            if contract is None and validation.contract.feature_contract_fingerprint is not None:
                raise AxBenchBpbAdvError(
                    "unclustered BPB-Adv run rejects a clustered source contract"
                )
            validation.require_directions(
                requirements_from_config(
                    cfg,
                    concept_ids=validation.concept_ids,
                ),
                surface="BPB-Adv runtime preflight",
            )
            _assert_source_contract_installable(
                _paths(cfg).source_contract(),
                validation.contract.to_json_dict(),
            )
        except Exception as exc:
            logger.exception("AxBench-BPB-Adv rank-0 source preflight failed")
            error = f"{type(exc).__name__}: {exc}"
    payload: list[str | None] = [error]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise AxBenchBpbAdvError(
            f"source preflight failed before run-marker creation: {payload[0]}"
        )
    return validation


def _preflight_and_install_feature_contract_marker_ddp(
    cfg: AxBenchBpbAdvConfig,
    contract: ClusterFeatureContract | None,
    *,
    rank: int,
    world_size: int,
) -> None:
    """Commit feature and source ownership from one successful preflight."""

    validation = _preflight_source_ddp(
        cfg,
        contract,
        rank=rank,
        world_size=world_size,
    )
    error: str | None = None
    if rank == 0:
        try:
            if validation is None:
                raise AxBenchBpbAdvError(
                    "source preflight completed without a validation snapshot"
                )
            if contract is None:
                _commit_exact_source_contract(cfg, validation)
            else:
                _commit_exact_clustered_source_contract(cfg, validation, contract)
        except Exception as exc:
            logger.exception(
                "AxBench-BPB-Adv rank-0 source contract commit failed"
            )
            error = f"{type(exc).__name__}: {exc}"
    payload: list[str | None] = [error]
    if world_size > 1:
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is not None:
        raise AxBenchBpbAdvError(
            f"source preflight commit failed before run-marker creation: {payload[0]}"
        )
    _install_feature_contract_marker_ddp(
        cfg,
        contract,
        rank=rank,
        world_size=world_size,
    )


def _paths(cfg: AxBenchBpbAdvConfig) -> AxBenchBpbAdvPaths:
    if cfg.run_tag is None:
        raise AxBenchBpbAdvError("cfg.run_tag must be set before resolving paths")
    return AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag=cfg.run_tag)


def _tokenizer_signature(tokenizer: Any) -> tuple[str, int | None, int | None]:
    cls = type(tokenizer)
    bos = tokenizer.get_bos_token_id() if hasattr(tokenizer, "get_bos_token_id") else None
    vocab = tokenizer.get_vocab_size() if hasattr(tokenizer, "get_vocab_size") else None
    return (f"{cls.__module__}.{cls.__qualname__}", bos, vocab)


def _target_runtime_info_from_runtime(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> tuple[dict[str, TargetModelMetadata], Any]:
    metas: dict[str, TargetModelMetadata] = {}
    tokenizers: list[Any] = []
    runtime_kwargs: dict[str, Any] = {
        "cfg": _bpb_config_from_adv_config(cfg),
        "ddp_rank": 0,
        "ddp_world_size": 1,
    }
    if cluster_contract is not None:
        runtime_kwargs["cluster_contract"] = cluster_contract
    runtime_state = RuntimeState(**runtime_kwargs)
    try:
        if "decoder_only" in cfg.target_models:
            decoder_bundle = runtime_state.decoder_only()
            metas["decoder_only"] = detect_target_model_metadata(
                decoder_bundle.model,
                runtime_state.cfg,
            )
            tokenizers.append(decoder_bundle.tokenizer)
        if "encdec" in cfg.target_models:
            encdec_bundle = runtime_state.encdec()
            detect_kwargs = (
                {} if cluster_contract is None else {"cluster_contract": cluster_contract}
            )
            metas["encdec"] = detect_target_model_metadata(
                encdec_bundle.model,
                runtime_state.cfg,
                **detect_kwargs,
            )
            tokenizers.append(encdec_bundle.tokenizer)
    finally:
        if cfg.offload_between_stages:
            runtime_state.offload_models()
    if not tokenizers:
        raise AxBenchBpbAdvError("target_models must contain at least one target")
    first = tokenizers[0]
    first_signature = _tokenizer_signature(first)
    for tokenizer in tokenizers[1:]:
        if tokenizer is first:
            continue
        signature = _tokenizer_signature(tokenizer)
        if signature != first_signature:
            raise AxBenchBpbAdvError(
                "target models resolved incompatible tokenizers: "
                f"{first_signature!r} vs {signature!r}"
            )
    return metas, first


def _synchronize_run_tag(
    cfg: AxBenchBpbAdvConfig,
    *,
    ddp_rank: int,
    ddp_world_size: int,
) -> None:
    local_tag = cfg.run_tag
    if ddp_rank == 0 and local_tag is None:
        local_tag = _generate_run_tag(cfg)
    payload: list[str | None] = [local_tag]
    if ddp_world_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise AxBenchBpbAdvError(
                "distributed run_tag synchronization requires initialized torch.distributed"
            )
        dist.broadcast_object_list(payload, src=0)
    if payload[0] is None:
        payload[0] = _generate_run_tag(cfg)
    cfg.run_tag = payload[0]


def _filter_synth_rows_to_source_concepts(
    rows: list[dict[str, Any]],
    *,
    concept_ids: set[int],
    mode: SteeringMode,
) -> list[dict[str, Any]]:
    filtered = [row for row in rows if int(row["concept_id"]) in concept_ids]
    skipped = len(rows) - len(filtered)
    if skipped:
        logger.info(
            "filtered {} {} synth row(s) whose concept_id is not in concepts.json",
            skipped,
            mode.value,
        )
    return filtered


def _assert_source_contract_installable(
    path: Path,
    expected: Mapping[str, Any],
) -> None:
    """Read-only immutability check used before feature-marker publication."""

    if not path.exists():
        return
    try:
        existing = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise AxBenchBpbAdvError(
            f"BPB-Adv source contract at {path} is unreadable; use a fresh run_tag"
        ) from exc
    if isinstance(existing, dict) and existing.get("schema_version") in {4, 5}:
        raise AxBenchBpbAdvError(
            f"legacy BPB-Adv source contract at {path} cannot be overwritten; use a fresh run tag"
        )
    if existing != dict(expected):
        raise AxBenchBpbAdvError(
            "BPB-Adv source contract changed within an existing run; source "
            "drift requires migration or a fresh run_tag"
        )


def _commit_exact_source_contract(
    cfg: AxBenchBpbAdvConfig,
    validation: BpbAdvSourceValidation,
) -> None:
    path = _paths(cfg).source_contract()
    _assert_source_contract_installable(path, validation.contract.to_json_dict())
    write_source_contract(path, validation.contract)


def _commit_exact_clustered_source_contract(
    cfg: AxBenchBpbAdvConfig,
    validation: Any,
    cluster_contract: ClusterFeatureContract,
) -> None:
    actual_fingerprint = validation.contract.feature_contract_fingerprint
    if actual_fingerprint != cluster_contract.feature_contract_fingerprint:
        raise AxBenchBpbAdvError(
            "clustered source validation feature-contract fingerprint mismatch: "
            f"source={actual_fingerprint!r}, "
            f"target={cluster_contract.feature_contract_fingerprint!r}"
        )
    _commit_exact_source_contract(cfg, validation)


def _validate_clustered_source_for_stage(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract,
) -> ClusteredStageSource:
    validation = validate_stage_0_3_source_run(
        cfg.source_run_dir,
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    validation.require_directions(
        requirements_from_config(cfg, concept_ids=validation.concept_ids),
        surface="BPB-Adv standalone clustered stage",
    )
    _commit_exact_clustered_source_contract(cfg, validation, cluster_contract)
    return resolve_clustered_stage_source(
        validation=validation,
        paths=_paths(cfg),
        contract=cluster_contract,
        expected_source_run_dir=cfg.source_run_dir,
        expected_filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )


@_profiled("axbench_bpb_adv.validate_source")
def _stage_validate_source(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    validation = validate_stage_0_3_source_run(
        cfg.source_run_dir,
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    validation.require_directions(
        requirements_from_config(cfg, concept_ids=validation.concept_ids),
        surface="BPB-Adv validate_source stage",
    )
    if cluster_contract is None:
        _commit_exact_source_contract(cfg, validation)
    else:
        _commit_exact_clustered_source_contract(cfg, validation, cluster_contract)
    record_counter("axbench_bpb_adv.validate_source.concepts", validation.n_concepts)
    record_counter(
        "axbench_bpb_adv.validate_source.positive_rows",
        validation.n_positive_synth_rows,
    )
    record_counter(
        "axbench_bpb_adv.validate_source.negative_rows",
        validation.n_negative_synth_rows,
    )
    record_counter(
        "axbench_bpb_adv.validate_source.synth_rows_total",
        validation.n_positive_synth_rows + validation.n_negative_synth_rows,
    )
    # Retained for source-contract/profiler schema compatibility. Incomplete
    # prefixes are fail-loud errors, so successful validation always records 0.
    record_counter("axbench_bpb_adv.validate_source.dropped", validation.n_dropped)
    logger.info(
        "validate_source: validated axbench-bpb-adv source: "
        f"concepts={validation.n_concepts}, "
        f"positive_rows={validation.n_positive_synth_rows}, "
        f"negative_rows={validation.n_negative_synth_rows}, "
        f"n_dropped={validation.n_dropped}"
    )


@_profiled("axbench_bpb_adv.prepare_context_cache")
def _stage_prepare_context_cache(
    cfg: AxBenchBpbAdvConfig,
    *,
    ddp_rank: int = 0,
    ddp_world_size: int = 1,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    # NOTE: profile_region is rank-aware via _CURRENT_PROFILER (rank 0 only —
    # ranks > 0 see no profiler, so the wrapping `@_profiled` collapses to a
    # no-op). This stage delegates to AxBench-BPB's existing context-cache
    # extraction; any per-batch instrumentation belongs in axbench_bpb itself.
    prefix_population = None
    if cluster_contract is not None:
        source_payload: list[dict[str, Any] | None] = [None]
        if ddp_rank == 0:
            try:
                stage_source = _validate_clustered_source_for_stage(
                    cfg,
                    cluster_contract,
                )
                source_payload[0] = {
                    "error": None,
                    "prefixes": stage_source.validation.context_prefixes(),
                }
            except Exception as exc:
                logger.exception(
                    "AxBench-BPB-Adv rank-0 context-cache source preflight failed"
                )
                source_payload[0] = {
                    "error": f"{type(exc).__name__}: {exc}",
                    "prefixes": None,
                }
        if ddp_world_size > 1:
            dist.broadcast_object_list(source_payload, src=0)
        payload = source_payload[0]
        if not isinstance(payload, dict):
            raise AxBenchBpbAdvError(
                "clustered context-cache source preflight broadcast returned "
                "a malformed payload"
            )
        if set(payload) != {"error", "prefixes"}:
            raise AxBenchBpbAdvError(
                "clustered context-cache source preflight broadcast returned "
                "malformed fields"
            )
        source_error = payload.get("error")
        if source_error is not None and not isinstance(source_error, str):
            raise AxBenchBpbAdvError(
                "clustered context-cache source preflight broadcast returned "
                "a malformed error"
            )
        if source_error is not None:
            raise AxBenchBpbAdvError(
                "clustered context-cache source preflight failed before model "
                f"load or cache writes: {source_error}"
            )
        prefixes = payload.get("prefixes")
        if not isinstance(prefixes, Mapping):
            raise AxBenchBpbAdvError(
                "clustered context-cache source preflight completed without a "
                "prefix population"
            )
        prefix_population = canonicalize_prefix_population(prefixes)

    bpb_cfg = _bpb_config_from_adv_config(cfg)
    runtime_kwargs: dict[str, Any] = {
        "cfg": bpb_cfg,
        "ddp_rank": ddp_rank,
        "ddp_world_size": ddp_world_size,
    }
    if cluster_contract is not None:
        runtime_kwargs["cluster_contract"] = cluster_contract
    runtime_state = RuntimeState(**runtime_kwargs)
    if cluster_contract is None:
        _run_bpb_context_cache_stage(runtime_state)
        return

    base_model, base_sae = runtime_state.base_gemma_and_sae()
    # The selected cache stores extraction-tokenizer BOS-inclusive character
    # coordinates and later projects nanochat target tokens through them.  Run
    # both tokenizer guards before the cache builder can forward the extractor
    # or commit its incomplete marker.  Rank-zero owns the canonical result;
    # broadcasting it keeps every DDP rank on the same side of the builder's
    # first collective.
    prebuild_guard_status: list[str | None] = [None]
    if ddp_rank == 0:
        try:
            target_tokenizer = get_tokenizer()
            assert_tokenizer_checksum_matches(
                tokenizer=target_tokenizer,
                feature_dir=bpb_cfg.feature_dir,
            )
            assert_bos_lengths_match(
                tokenizer_extraction=runtime_state.extraction_tokenizer(),
                tokenizer_nanochat=target_tokenizer,
            )
        except Exception as exc:
            logger.exception(
                "AxBench-BPB-Adv rank-0 pre-build tokenizer guard failed"
            )
            prebuild_guard_status[0] = f"{type(exc).__name__}: {exc}"
    if ddp_world_size > 1:
        dist.broadcast_object_list(prebuild_guard_status, src=0)
    if prebuild_guard_status[0] is not None:
        raise AxBenchBpbAdvError(
            "clustered context-cache tokenizer compatibility preflight failed "
            f"before extraction or cache writes: {prebuild_guard_status[0]}"
        )

    if prefix_population is None:
        raise AxBenchBpbAdvError(
            "clustered context-cache stage lost its canonical prefix population"
        )
    build_clustered_context_cache(
        cfg=bpb_cfg,
        output_paths=_paths(cfg),
        prefix_population=prefix_population,
        base_model=base_model,
        base_sae=base_sae,
        cluster_contract=cluster_contract,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    # The extraction Gemma/SAE pair can consume essentially the entire GPU.
    # Drop every local + RuntimeState owner before loading the nanochat target
    # for the compatibility guards below. The extraction tokenizer remains
    # cached on the host, so this transition does not reload or change it.
    del base_model, base_sae
    runtime_state.release_base_gemma_and_sae()

    # Preserve the legacy tokenizer/BOS compatibility guards without letting a
    # rank-zero exception strand the other DDP ranks at a barrier.  The
    # clustered cache builder has completed all of its collectives before this
    # status broadcast.
    guard_status: list[str | None] = [None]
    if ddp_rank == 0:
        try:
            decoder_bundle = runtime_state.decoder_only()
            assert_tokenizer_checksum_matches(
                tokenizer=decoder_bundle.tokenizer,
                feature_dir=bpb_cfg.feature_dir,
            )
            assert_bos_lengths_match(
                tokenizer_extraction=runtime_state.extraction_tokenizer(),
                tokenizer_nanochat=decoder_bundle.tokenizer,
            )
        except Exception as exc:
            logger.exception(
                "AxBench-BPB-Adv rank-0 post-build tokenizer guard failed"
            )
            guard_status[0] = f"{type(exc).__name__}: {exc}"
    if ddp_world_size > 1:
        dist.broadcast_object_list(guard_status, src=0)
    if guard_status[0] is not None:
        raise AxBenchBpbAdvError(
            f"clustered context-cache tokenizer compatibility guard failed: {guard_status[0]}"
        )


@_profiled("axbench_bpb_adv.prepare_attack_rows")
def _stage_prepare_attack_rows(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    validation = validate_stage_0_3_source_run(
        cfg.source_run_dir,
        filter_unaccepted_samples=cfg.filter_unaccepted_samples,
    )
    if cluster_contract is not None:
        _commit_exact_clustered_source_contract(cfg, validation, cluster_contract)
    with profile_region("axbench_bpb_adv.prepare_attack_rows.target_runtime"):
        if cluster_contract is None:
            target_metas, tokenizer = _target_runtime_info_from_runtime(cfg)
        else:
            target_metas, tokenizer = _target_runtime_info_from_runtime(
                cfg,
                cluster_contract,
            )
    work_specs = work_specs_from_method_registry(
        base_methods=cfg.base_methods,
        target_models=cfg.target_models,
        target_model_metas=target_metas,
        cfg=cfg,
    )
    record_counter("axbench_bpb_adv.prepare_attack_rows.work_specs", len(work_specs))
    if cluster_contract is None:
        # Preserve the unclustered reload/filter route exactly. Clustered
        # attack rows instead consume the tables retained by the source
        # validation whose contract was committed above.
        source_paths = SourcePaths(Path(cfg.source_run_dir))
        positive_rows = load_raw_synth_rows(
            source_paths.synth_training_positive(),
            filter_unaccepted=cfg.filter_unaccepted_samples,
            require_positive_qc_contract=True,
            expected_continuation_role="stage1b_positive",
        )
        negative_rows = load_raw_synth_rows(
            source_paths.synth_training_negative(),
            filter_unaccepted=cfg.filter_unaccepted_samples,
            expected_continuation_role="stage1c_negative",
        )
        concept_ids = _read_source_concept_ids(source_paths.concepts_json())
        positive_rows = _filter_synth_rows_to_source_concepts(
            positive_rows,
            concept_ids=concept_ids,
            mode=SteeringMode.POSITIVE,
        )
        negative_rows = _filter_synth_rows_to_source_concepts(
            negative_rows,
            concept_ids=concept_ids,
            mode=SteeringMode.NEGATIVE,
        )
    else:
        positive_rows = validation.positive_rows()
        negative_rows = validation.negative_rows()
    samples = [
        bpb_attack_sample_from_synth_row(row, mode=SteeringMode.POSITIVE, tokenizer=tokenizer)
        for row in positive_rows
    ]
    samples.extend(
        bpb_attack_sample_from_synth_row(row, mode=SteeringMode.NEGATIVE, tokenizer=tokenizer)
        for row in negative_rows
    )
    n_groups = len(
        {
            (int(sample.concept_id), SteeringMode(sample.mode).value, sample.prefix_text)
            for sample in samples
        }
    )
    prefixes = pair_samples_into_prefixes(cast(Any, samples))
    active_modes = {SteeringMode(spec.mode) for spec in work_specs}
    evaluation_prefixes = evaluation_prefixes_for_scoring(
        prefixes,
        modes=active_modes,
        ours=cfg.ours_gcg,
    )
    n_dropped = 0
    record_counter("axbench_bpb_adv.prepare_attack_rows.samples", len(samples))
    # `_kept` suffix disambiguates these POST-filter counts from the
    # validate_source stage's pre-filter `positive_rows`/`negative_rows`.
    record_counter("axbench_bpb_adv.prepare_attack_rows.positive_rows_kept", len(positive_rows))
    record_counter("axbench_bpb_adv.prepare_attack_rows.negative_rows_kept", len(negative_rows))
    # #505 class-margin: samples are paired into core prefixes (each owning a
    # concept_rich + concept_avoiding completion). Incomplete groups now raise
    # before artifact write; `dropped` remains as a zero-valued compatibility
    # counter.
    record_counter("axbench_bpb_adv.prepare_attack_rows.prefix_groups", n_groups)
    record_counter("axbench_bpb_adv.prepare_attack_rows.core_prefixes", len(prefixes))
    record_counter(
        "axbench_bpb_adv.prepare_attack_rows.evaluation_core_prefixes",
        len(evaluation_prefixes),
    )
    record_counter("axbench_bpb_adv.prepare_attack_rows.dropped", n_dropped)
    attack_rows = prepare_bpb_attack_rows(
        prefixes=evaluation_prefixes,
        work_specs=work_specs,
        cfg=cfg,
    )
    if not attack_rows.schema.equals(ATTACK_ROWS_SCHEMA):
        raise AxBenchBpbAdvError("internal error: attack-row schema drift")
    record_counter("axbench_bpb_adv.prepare_attack_rows.attack_rows", attack_rows.num_rows)
    out_paths = _paths(cfg)
    if cluster_contract is None:
        write_source_contract(out_paths.source_contract(), validation.contract)
    write_evaluation_population(
        out_paths.evaluation_population(),
        attack_rows.to_pylist(),
        ours=cfg.ours_gcg,
        **(
            {}
            if cluster_contract is None
            else {"feature_contract_fingerprint": (cluster_contract.feature_contract_fingerprint)}
        ),
    )
    out_paths.attack_rows_dir().mkdir(parents=True, exist_ok=True)
    attack_rows = bind_clustered_table(
        attack_rows,
        paths=out_paths,
        contract=cluster_contract,
        artifact_family="attack_rows",
    )
    if cluster_contract is None:
        pq.write_table(attack_rows, out_paths.attack_rows_path())
    else:
        write_clustered_table_atomic(
            attack_rows,
            out_paths.attack_rows_path(),
            paths=out_paths,
            contract=cluster_contract,
            artifact_family="attack_rows",
        )
    logger.info(
        f"prepare_attack_rows: prepared {attack_rows.num_rows} axbench-bpb-adv "
        f"attack rows at {out_paths.attack_rows_path()} "
        f"(n_dropped={n_dropped})"
    )


def _sum_list_lengths(table: pa.Table, column_name: str) -> int:
    """Sum ``len(...)`` across all rows of a pyarrow list-typed column.

    O(rows) but only ever called from profiler-gated sites — the production
    path never pays this when no profiler is bound.

    Null cells (BPB-Adv list-typed schema fields are ``nullable=True`` by
    pyarrow default) contribute 0. No production writer emits nulls today,
    but the helper is defensive so a future writer change can't crash the
    profiled stage region — preserving the "profiling must never break the
    run" invariant the surrounding helpers document.
    """
    return int(sum(0 if v is None else len(v) for v in table.column(column_name).to_pylist()))


@_profiled("axbench_bpb_adv.build_candidates")
def _stage_build_candidates(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    stage_source = None
    if cluster_contract is not None:
        stage_source = _validate_clustered_source_for_stage(
            cfg,
            cluster_contract,
        )
    candidate_kwargs: dict[str, Any] = {"cfg": cfg, "paths": _paths(cfg)}
    if cluster_contract is not None:
        candidate_kwargs["cluster_contract"] = cluster_contract
        candidate_kwargs["stage_source"] = stage_source
    written = run_model_backed_bpb_candidate_stage(**candidate_kwargs)
    record_counter("axbench_bpb_adv.build_candidates.candidates_written", written)
    out_paths = _paths(cfg)
    candidate_snapshot = None
    if cluster_contract is None:
        candidate_table = pq.read_table(out_paths.candidates_path())
        assert_table_cluster_identity(
            candidate_table,
            paths=out_paths,
            contract=None,
            artifact_family="candidates",
        )
    else:
        assert stage_source is not None
        candidate_snapshot = load_clustered_candidate_snapshot(
            out_paths.candidates_path(),
            paths=out_paths,
            contract=cluster_contract,
            source_identity=stage_source.source_identity,
        )
        candidate_table = candidate_snapshot.table
    if not candidate_table.schema.equals(CANDIDATES_SCHEMA):
        raise AxBenchBpbAdvError("internal error: candidate schema drift")
    record_counter(
        "axbench_bpb_adv.build_candidates.candidate_rows",
        candidate_table.num_rows,
    )
    # The candidate stage also writes the canonical universal-triggers artifact
    # (one row per non-control faster_gcg concept group; always present, possibly
    # zero rows). Read it back and assert the schema so a drift in the
    # general_val-facing artifact is caught here (mirrors the candidates guard).
    universal_triggers_table = pq.read_table(out_paths.universal_triggers_path())
    if not universal_triggers_table.schema.equals(UNIVERSAL_TRIGGERS_SCHEMA):
        raise AxBenchBpbAdvError("internal error: universal_triggers schema drift")
    assert_table_cluster_identity(
        universal_triggers_table,
        paths=out_paths,
        contract=cluster_contract,
        artifact_family="universal_triggers",
        source_identity=(
            None if stage_source is None else stage_source.source_identity
        ),
        candidate_snapshot=candidate_snapshot,
    )
    record_counter(
        "axbench_bpb_adv.build_candidates.universal_triggers_rows",
        universal_triggers_table.num_rows,
    )
    # Per #501 — token-count counters for the profiler. Gated on
    # `get_current_profiler()` so production avoids the O(rows) `to_pylist` scan
    # when nothing will consume the count (mirroring the plot-stage gate).
    # #505 class-margin: each core prefix owns TWO completions — concept_rich
    # (concept-favoring) and concept_avoiding (contrast) — so the per-completion
    # output-token totals are surfaced as a split rather than one merged total.
    if get_current_profiler() is not None:
        record_counter(
            "axbench_bpb_adv.build_candidates.source_prompt_tokens_total",
            _sum_list_lengths(candidate_table, "source_prompt_tokens"),
        )
        record_counter(
            "axbench_bpb_adv.build_candidates.attacked_prompt_tokens_total",
            _sum_list_lengths(candidate_table, "attacked_prompt_tokens"),
        )
        record_counter(
            "axbench_bpb_adv.build_candidates.concept_rich_output_tokens_total",
            _sum_list_lengths(candidate_table, "concept_rich_output_tokens"),
        )
        record_counter(
            "axbench_bpb_adv.build_candidates.concept_avoiding_output_tokens_total",
            _sum_list_lengths(candidate_table, "concept_avoiding_output_tokens"),
        )
    logger.info(f"built {written} candidate rows")


def _score_path(cfg: AxBenchBpbAdvConfig) -> Path:
    if cfg.scores_path:
        return Path(cfg.scores_path)
    out_paths = _paths(cfg)
    metric = SteeringMetric.coerce(cfg.steering_metric)
    if metric is SteeringMetric.ROUNDTRIP:
        return out_paths.roundtrip_scores_path()
    return out_paths.scores_path()


@_profiled("axbench_bpb_adv.score")
def _stage_score(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    stage_source = None
    if cluster_contract is not None:
        stage_source = _validate_clustered_source_for_stage(
            cfg,
            cluster_contract,
        )
    source_identity = (
        None if stage_source is None else stage_source.source_identity
    )
    identity_kwargs: dict[str, Any] = (
        {}
        if source_identity is None
        else {"source_identity": source_identity}
    )
    out_paths = _paths(cfg)
    candidate_snapshot = None
    if cluster_contract is None:
        candidates = load_candidate_table(out_paths.candidates_path())
    else:
        candidate_snapshot = load_clustered_candidate_snapshot(
            out_paths.candidates_path(),
            paths=out_paths,
            contract=cluster_contract,
            **identity_kwargs,
        )
        candidates = candidate_snapshot.table
    record_counter("axbench_bpb_adv.score.candidates_loaded", candidates.num_rows)
    dedup_stats: dict[str, int] = {}
    metric = SteeringMetric.coerce(cfg.steering_metric)
    if metric is SteeringMetric.ROUNDTRIP:
        scores = (
            run_model_backed_roundtrip_score_stage(
                cfg=cfg,
                paths=out_paths,
                candidates=candidates,
                stats=dedup_stats,
            )
            if cluster_contract is None
            else run_model_backed_roundtrip_score_stage(
                cfg=cfg,
                paths=out_paths,
                candidates=candidates,
                stats=dedup_stats,
                cluster_contract=cluster_contract,
                stage_source=stage_source,
                candidate_snapshot=candidate_snapshot,
            )
        )
        if not scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA):
            raise AxBenchBpbAdvError("internal error: roundtrip score schema drift")
    else:
        scores = (
            run_model_backed_bpb_adv_score_stage(
                cfg=cfg,
                paths=out_paths,
                candidates=candidates,
                stats=dedup_stats,
            )
            if cluster_contract is None
            else run_model_backed_bpb_adv_score_stage(
                cfg=cfg,
                paths=out_paths,
                candidates=candidates,
                stats=dedup_stats,
                cluster_contract=cluster_contract,
                stage_source=stage_source,
                candidate_snapshot=candidate_snapshot,
            )
        )
        if not scores.schema.equals(SCORE_SCHEMA):
            raise AxBenchBpbAdvError("internal error: score schema drift")
    record_counter("axbench_bpb_adv.score.score_rows", scores.num_rows)
    # Per #501 — token-count counters for the profiler. Gated on
    # `get_current_profiler()` so production avoids the per-row scan when no
    # profiler will consume the value. Prompt tokens come from the candidates
    # table (already in memory); output tokens come from SCORE_SCHEMA's
    # per-completion ``n_rich_output_tokens`` / ``n_avoid_output_tokens``
    # int32 columns (already aggregated per row by the dual-completion scorer
    # — #505 class-margin replaced the single ``n_output_tokens`` column).
    if get_current_profiler() is not None:
        record_counter(
            "axbench_bpb_adv.score.scoring_cache_hits",
            dedup_stats.get(
                "cache_hits",
                dedup_stats.get("generation_cache_hits", 0),
            ),
        )
        record_counter(
            "axbench_bpb_adv.score.scored_unique",
            dedup_stats.get(
                "scored_unique",
                dedup_stats.get("generated_unique", 0),
            ),
        )
        record_counter(
            "axbench_bpb_adv.score.source_prompt_tokens_total",
            _sum_list_lengths(candidates, "source_prompt_tokens"),
        )
        record_counter(
            "axbench_bpb_adv.score.attacked_prompt_tokens_total",
            _sum_list_lengths(candidates, "attacked_prompt_tokens"),
        )
        if metric is SteeringMetric.ROUNDTRIP:
            record_counter(
                "axbench_bpb_adv.score.bpb_scoring_cache_hits",
                dedup_stats.get("cache_hits", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.bpb_scored_unique",
                dedup_stats.get("scored_unique", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_generation_cache_hits",
                dedup_stats.get("generation_cache_hits", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_generated_unique",
                dedup_stats.get("generated_unique", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_control_generation_cache_hits",
                dedup_stats.get("control_generation_cache_hits", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_control_generated_unique",
                dedup_stats.get("control_generated_unique", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_generated_tokens_total",
                dedup_stats.get(
                    "generated_tokens",
                    int(sum(v or 0 for v in scores.column("n_generated_tokens").to_pylist())),
                ),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_control_generated_tokens_total",
                dedup_stats.get(
                    "control_generated_tokens",
                    int(
                        sum(v or 0 for v in scores.column("control_n_generated_tokens").to_pylist())
                    ),
                ),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_probe_rows",
                dedup_stats.get("probe_rows", scores.num_rows),
            )
            # Empty steered/control completions are skipped (not scored) because
            # the SAE reactivation probe has no text. Surface the dropped counts
            # so the metric's reliability is auditable.
            record_counter(
                "axbench_bpb_adv.score.roundtrip_empty_skipped",
                dedup_stats.get("empty_skipped", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_empty_steered",
                dedup_stats.get("empty_steered", 0),
            )
            record_counter(
                "axbench_bpb_adv.score.roundtrip_empty_control",
                dedup_stats.get("empty_control", 0),
            )
        else:
            # int32 non-list columns; null cells (from synthesized fixtures)
            # contribute 0 — production scorers always populate these.
            record_counter(
                "axbench_bpb_adv.score.rich_output_tokens_total",
                int(sum(v or 0 for v in scores.column("n_rich_output_tokens").to_pylist())),
            )
            record_counter(
                "axbench_bpb_adv.score.avoid_output_tokens_total",
                int(sum(v or 0 for v in scores.column("n_avoid_output_tokens").to_pylist())),
            )
    out_paths.root.mkdir(parents=True, exist_ok=True)
    score_path = _score_path(cfg)
    score_family = (
        "roundtrip_scores" if metric is SteeringMetric.ROUNDTRIP else "class_margin_scores"
    )
    scores = bind_clustered_table(
        scores,
        paths=out_paths,
        contract=cluster_contract,
        artifact_family=score_family,
        candidate_snapshot=candidate_snapshot,
        **identity_kwargs,
    )
    if cluster_contract is None:
        pq.write_table(scores, score_path)
    else:
        assert candidate_snapshot is not None
        write_clustered_table_atomic(
            scores,
            score_path,
            paths=out_paths,
            contract=cluster_contract,
            artifact_family=score_family,
            candidate_snapshot=candidate_snapshot,
            **identity_kwargs,
        )
    logger.info(f"wrote {scores.num_rows} score rows to {score_path}")


@_profiled("axbench_bpb_adv.score_general_val")
def _stage_score_general_val(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    # Opt-in: when disabled the stage is a no-op and the general_val_bpb agg
    # column stays null, so the eval is byte-identical to its pre-feature form.
    if not cfg.general_val_bpb.enabled:
        logger.info(
            "score_general_val: general_val_bpb.enabled=false; skipping "
            "(general_val_bpb metric will be null)"
        )
        return
    stage_source = None
    if cluster_contract is not None:
        stage_source = _validate_clustered_source_for_stage(
            cfg,
            cluster_contract,
        )
    general_val_kwargs: dict[str, Any] = {"cfg": cfg, "paths": _paths(cfg)}
    if cluster_contract is not None:
        general_val_kwargs["cluster_contract"] = cluster_contract
        general_val_kwargs["stage_source"] = stage_source
    table = run_general_val_bpb_stage(**general_val_kwargs)
    record_counter("axbench_bpb_adv.score_general_val.rows", table.num_rows)
    logger.info(f"score_general_val: wrote {table.num_rows} general_val_bpb rows")


def _general_val_path_for_join(cfg: AxBenchBpbAdvConfig) -> Path | None:
    """The general_val table to left-join at aggregation, or None when the
    feature is disabled (so aggregation skips the join and the column stays
    null)."""
    if not cfg.general_val_bpb.enabled:
        return None
    return _paths(cfg).general_val_bpb_path()


@_profiled("axbench_bpb_adv.aggregate")
def _stage_aggregate(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    stage_source = None
    if cluster_contract is not None:
        # Aggregation is independently invocable.  Recompute the clustered
        # source contract from current bytes before trusting a score table that
        # was stamped against an earlier source_contract.json.
        stage_source = _validate_clustered_source_for_stage(cfg, cluster_contract)
    metric = SteeringMetric.coerce(cfg.steering_metric)
    paths = _paths(cfg)
    general_val_path = _general_val_path_for_join(cfg)
    score_table: pa.Table | None = None
    general_val_table: pa.Table | None = None
    if cluster_contract is not None:
        assert stage_source is not None
        candidate_snapshot = load_clustered_candidate_snapshot(
            paths.candidates_path(),
            paths=paths,
            contract=cluster_contract,
            source_identity=stage_source.source_identity,
        )
        score_family = (
            "roundtrip_scores" if metric is SteeringMetric.ROUNDTRIP else "class_margin_scores"
        )
        score_table = pq.read_table(_score_path(cfg))
        assert_table_cluster_identity(
            score_table,
            paths=paths,
            contract=cluster_contract,
            artifact_family=score_family,
            source_identity=stage_source.source_identity,
            candidate_snapshot=candidate_snapshot,
        )
        if general_val_path is not None:
            general_val_table = pq.read_table(general_val_path)
            assert_table_cluster_identity(
                general_val_table,
                paths=paths,
                contract=cluster_contract,
                artifact_family="general_val_bpb",
                source_identity=stage_source.source_identity,
                candidate_snapshot=candidate_snapshot,
            )
    aggregate_kwargs: dict[str, Any] = {}
    if cluster_contract is not None:
        aggregate_kwargs = {
            "scores_table": score_table,
            "general_val_table": general_val_table,
        }
    if metric is SteeringMetric.ROUNDTRIP:
        aggregate_roundtrip_adv_scores(
            _score_path(cfg),
            paths.roundtrip_aggregates_dir(),
            general_val_path=general_val_path,
            **aggregate_kwargs,
        )
        return
    aggregate_bpb_adv_scores(
        _score_path(cfg),
        paths.aggregates_dir(),
        general_val_path=general_val_path,
        **aggregate_kwargs,
    )


@_profiled("axbench_bpb_adv.plot")
def _stage_plot(
    cfg: AxBenchBpbAdvConfig,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    if cluster_contract is not None:
        # A schema-current macro has no artifact-local cluster stamp.  Always
        # rebuild it from the fully validated clustered score inputs so a
        # direct ``stage=plot`` invocation cannot consume a stale aggregate.
        _stage_aggregate(cfg, cluster_contract)
    out_paths = _paths(cfg)
    metric = SteeringMetric.coerce(cfg.steering_metric)
    if metric is SteeringMetric.ROUNDTRIP:
        aggregates_dir = out_paths.roundtrip_aggregates_dir()
        macro_schema = ROUNDTRIP_MACRO_AGG_SCHEMA
    else:
        aggregates_dir = out_paths.aggregates_dir()
        macro_schema = MACRO_AGG_SCHEMA
    macro_path = aggregates_dir / "macro.parquet"
    # Re-aggregate when the macro is absent OR was written by an older schema
    # (e.g. before is_control was added) -- the plot path reads is_control and
    # would otherwise KeyError on a stale-artifact resume. Other stages validate
    # their inputs on read; this mirrors that for the resume path. Read only the
    # parquet footer (read_schema) for the check, not the whole table.
    fresh = macro_path.exists() and pq.read_schema(macro_path).equals(macro_schema)
    if not fresh:
        if metric is SteeringMetric.ROUNDTRIP:
            aggregate_roundtrip_adv_scores(
                _score_path(cfg),
                aggregates_dir,
                general_val_path=_general_val_path_for_join(cfg),
            )
        else:
            aggregate_bpb_adv_scores(
                _score_path(cfg),
                out_paths.aggregates_dir(),
                general_val_path=_general_val_path_for_join(cfg),
            )
    macro = pq.read_table(macro_path)
    record_counter("axbench_bpb_adv.plot.macro_rows", macro.num_rows)
    if metric is SteeringMetric.ROUNDTRIP:
        plot_roundtrip_adv_summaries(
            macro,
            out_paths.roundtrip_plots_dir(),
            scatter_baseline_x=cfg.scatter_baseline_x,
            plot_methods=cfg.plot_methods,
        )
        plots_dir = out_paths.roundtrip_plots_dir()
        strength_metric_column = "primary_roundtrip_y"
        strength_metric_label = ROUNDTRIP_PRIMARY_YLABEL
    else:
        plot_bpb_adv_summaries(
            macro,
            out_paths.plots_dir(),
            scatter_baseline_x=cfg.scatter_baseline_x,
            plot_methods=cfg.plot_methods,
        )
        plots_dir = out_paths.plots_dir()
        strength_metric_column = "primary_margin_y"
        strength_metric_label = PRIMARY_MARGIN_YLABEL
    if cfg.plot_steering_strength_families:
        strength_written = plot_steering_strength_families(
            macro,
            out_paths.steering_strength_plots_dir(),
            metric_column=strength_metric_column,
            metric_label=strength_metric_label,
            plot_methods=cfg.plot_methods,
        )
        record_counter(
            "axbench_bpb_adv.plot.steering_strength_files_written",
            len([path for path in strength_written if path.is_file()]),
        )
    # Count plot files written (PNG/PDF/SVG) so #502 can see how much I/O the
    # plot stage does relative to its compute. Gated on `get_current_profiler()`
    # so production avoids the directory scan when nothing will consume it
    # (the surrounding `record_counter` already no-ops without a profiler).
    # `rglob` (recursive) is required because `plot_bpb_adv_summaries` writes
    # PNGs into nested ``<mode>/`` subdirs (incl. the per-subset
    # scatter plot) — a non-recursive ``glob("*")`` would miss every PNG and
    # count only the top-level index CSVs. The nested layout is pinned by
    # `tests/.../test_steerability_scatter_flow.py::test_stage_plot_recursive_count_includes_scatter`.
    if get_current_profiler() is not None:
        n_files = sum(1 for p in plots_dir.rglob("*") if p.is_file())
        record_counter("axbench_bpb_adv.plot.files_written", n_files)


def _not_implemented(stage: Stage) -> NoReturn:
    raise NotImplementedError(f"axbench-bpb-adv stage {stage.value!r} is not implemented")


_STAGE_HANDLERS: dict[
    Stage,
    Callable[[AxBenchBpbAdvConfig, ClusterFeatureContract | None], None],
] = {
    Stage.VALIDATE_SOURCE: _stage_validate_source,
    Stage.PREPARE_ATTACK_ROWS: _stage_prepare_attack_rows,
    Stage.BUILD_CANDIDATES: _stage_build_candidates,
    Stage.SCORE: _stage_score,
    Stage.SCORE_GENERAL_VAL: _stage_score_general_val,
    Stage.AGGREGATE: _stage_aggregate,
    Stage.PLOT: _stage_plot,
}


def _resolve_stages(stage: Stage) -> list[Stage]:
    if stage == Stage.ALL:
        return [
            Stage.VALIDATE_SOURCE,
            Stage.PREPARE_CONTEXT_CACHE,
            Stage.PREPARE_ATTACK_ROWS,
            Stage.BUILD_CANDIDATES,
            Stage.SCORE,
            Stage.SCORE_GENERAL_VAL,
            Stage.AGGREGATE,
            Stage.PLOT,
        ]
    return [stage]


def _run_stage(
    stage: Stage,
    cfg_obj: AxBenchBpbAdvConfig,
    *,
    ddp_rank: int,
    ddp_world_size: int,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    if ddp_rank == 0:
        logger.info(f"→ Stage: {stage.value}")
    if stage == Stage.PREPARE_CONTEXT_CACHE:
        context_kwargs: dict[str, Any] = {
            "ddp_rank": ddp_rank,
            "ddp_world_size": ddp_world_size,
        }
        if cluster_contract is not None:
            context_kwargs["cluster_contract"] = cluster_contract
        _stage_prepare_context_cache(cfg_obj, **context_kwargs)
        return
    if ddp_rank != 0:
        return
    handler = _STAGE_HANDLERS.get(stage)
    if handler is None:
        _not_implemented(stage)
    handler(cfg_obj, cluster_contract)


def _abort_all_ranks_on_stage_failure(
    error: BaseException | None,
    *,
    stage: Stage,
    world_size: int,
) -> None:
    """Make a stage failure on ANY rank abort every rank deterministically (#942).

    The stage loop mixes collective and rank-0-only stages: ``PREPARE_CONTEXT_CACHE``
    runs everywhere, the rest return early on non-zero ranks and are followed by
    ``_maybe_barrier``. So if one rank raises, the others are already committed to a
    collective the failing rank will never reach. In practice the survivors then block
    until the failing process finishes unwinding and its transport drops, at which point
    they surface an opaque ``Connection closed by peer`` from gloo rather than the real
    cause; if the failing rank does not exit promptly, they block indefinitely.

    Every rank therefore captures its own exception, all ranks meet at one exchange, and
    all raise together: the rank that actually failed re-raises its original exception
    (traceback intact), and the others raise a summary naming the stage.

    Deliberately symmetric, unlike ``axbench_bpb_eval._propagate_rank0_derive``, which
    broadcasts from rank 0 because only rank 0 can fail in the region it guards. Here a
    collective stage can fail on any rank, so a broadcast would leave a non-zero rank's
    failure unpropagated.
    """

    if world_size > 1:
        if not (dist.is_available() and dist.is_initialized()):
            raise AxBenchBpbAdvError(
                "distributed BPB-Adv stage dispatch requires initialized torch.distributed"
            )
        flag = _torch.tensor([1 if error is not None else 0], dtype=_torch.int64)
        dist.all_reduce(flag, op=dist.ReduceOp.MAX)
        any_rank_failed = bool(flag.item())
    else:
        any_rank_failed = error is not None
    if error is not None:
        raise error
    if any_rank_failed:
        raise AxBenchBpbAdvError(
            f"another rank failed during stage {stage.value}; all ranks aborting "
            "(see that rank's log for the original error)"
        )


def _run_resolved(
    cfg_obj: AxBenchBpbAdvConfig,
    *,
    ddp_rank: int,
    ddp_world_size: int,
    cluster_contract: ClusterFeatureContract | None = None,
) -> AxBenchBpbAdvConfig:
    if ddp_rank == 0:
        logger.info(f"axbench-bpb-adv run_tag={cfg_obj.run_tag}")
        logger.info(f"source_run_dir={cfg_obj.source_run_dir}")
        logger.info(f"output_root={cfg_obj.output_root}")
        logger.info(f"stage={cfg_obj.stage}, world_size={ddp_world_size}")
    for stage in _resolve_stages(Stage.coerce(cfg_obj.stage)):
        stage_error: BaseException | None = None
        try:
            _run_stage(
                stage,
                cfg_obj,
                ddp_rank=ddp_rank,
                ddp_world_size=ddp_world_size,
                cluster_contract=cluster_contract,
            )
        except BaseException as exc:  # noqa: BLE001 - re-raised below on this rank
            stage_error = exc
        _abort_all_ranks_on_stage_failure(
            stage_error,
            stage=stage,
            world_size=ddp_world_size,
        )
        _maybe_barrier(ddp_world_size)
    return cfg_obj


def _run(cfg: DictConfig) -> AxBenchBpbAdvConfig:
    cfg_obj = _config_loader.cfg_to_dataclass(cfg)
    validate_config(cfg_obj)
    validate_all_stage_representative_target_contract(cfg_obj)
    contract_request = build_cluster_contract_request(cfg_obj)
    try:
        _, _, ddp_local_rank, _, device = compute_init(
            device_type=_device_type_from_cfg(cfg_obj.device)
        )
        if dist.is_available() and dist.is_initialized():
            ddp_rank = dist.get_rank()
            ddp_world_size = dist.get_world_size()
        else:
            ddp_rank, ddp_world_size = 0, 1
        cluster_contract: ClusterFeatureContract | None = None
        if contract_request is not None:
            cluster_contract = resolve_cluster_feature_contract_ddp(contract_request)
            cfg_obj.encdec_target = {
                **cfg_obj.encdec_target,
                "step": cluster_contract.checkpoint.step,
            }
        _synchronize_run_tag(
            cfg_obj,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
        assert cfg_obj.run_tag is not None
        _preflight_and_install_feature_contract_marker_ddp(
            cfg_obj,
            cluster_contract,
            rank=ddp_rank,
            world_size=ddp_world_size,
        )
        out_paths = _paths(cfg_obj)
        profile_output_dir = (
            Path(cfg_obj.profile.output_dir)
            if cfg_obj.profile.output_dir is not None
            else out_paths.profile_dir()
        )
        profile_gpu_device_index = resolve_profile_gpu_device_index(
            cfg_obj.device,
            ddp_local_rank=ddp_local_rank,
            ddp_world_size=ddp_world_size,
        )
        profile_metadata = {
            "pipeline": "axbench_bpb_adv",
            "run_tag": cfg_obj.run_tag,
            "stage": cfg_obj.stage,
            "stages": [stage.value for stage in _resolve_stages(Stage.coerce(cfg_obj.stage))],
            "source_run_dir": cfg_obj.source_run_dir,
            "output_root": cfg_obj.output_root,
            "feature_dir": cfg_obj.feature_dir,
            "feature_statistics_path": cfg_obj.feature_statistics_path,
            "target_models": list(cfg_obj.target_models),
            "steering_metric": cfg_obj.steering_metric,
            "ddp_rank": ddp_rank,
            "ddp_world_size": ddp_world_size,
            "ddp_local_rank": ddp_local_rank,
            "profile_gpu_device_index": profile_gpu_device_index,
            "device": str(device),
            "config": axbench_bpb_adv_config_to_dict(cfg_obj),
        }
        profile_cm = (
            AxBenchProfiler(
                profile_output_dir,
                metadata=profile_metadata,
                synchronize_cuda=cfg_obj.profile.synchronize_cuda,
                gpu_utilization=cfg_obj.profile.gpu_utilization,
                gpu_poll_interval_ms=cfg_obj.profile.gpu_poll_interval_ms,
                gpu_idle_threshold=cfg_obj.profile.gpu_idle_threshold,
                gpu_device_index=profile_gpu_device_index,
            )
            if cfg_obj.profile.enabled and ddp_rank == 0
            else nullcontext()
        )
        with profile_cm:
            run_kwargs: dict[str, Any] = {
                "ddp_rank": ddp_rank,
                "ddp_world_size": ddp_world_size,
            }
            if cluster_contract is not None:
                run_kwargs["cluster_contract"] = cluster_contract
            return _run_resolved(cfg_obj, **run_kwargs)
    finally:
        compute_cleanup()


@hydra.main(
    config_path="../../configs/eval/axbench_bpb_adv",
    config_name="axbench_bpb_adv_eval",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    _run(cfg)


if __name__ == "__main__":
    main()
