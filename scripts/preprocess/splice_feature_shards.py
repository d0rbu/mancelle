"""Post-extraction feature-shard splicer (issue #590).

Recombines already-computed feature-shard segments (carrying features verbatim)
into a separate derived feature directory, leaving the base shards immutable.
CPU-only, weight-free. Runs after extract_features + compute_statistics.
Optional remote mode stages verified feature bodies from an HF bucket while
keeping source parquets local and publishing the existing output triplet.

    uv run -m scripts.preprocess.splice_feature_shards \
        feature_dir=out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base \
        cut_fraction=0.15
"""

from __future__ import annotations

import fcntl
import json
import math
import multiprocessing as mp
import os
import pickle
import re
import stat
import threading
import time
from collections.abc import Iterator, Mapping, Sequence
from concurrent.futures import Future, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any

import hydra
import pyarrow as pa
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from nanochat.dataset import list_parquet_files
from scripts.preprocess._splice_profile import SpliceRunProfile, SpliceShardProfile
from steering.conditional.percentile_table import build_percentile_table
from steering.feature_io import (
    COMPRESSED_EXT,
    glob_features,
    read_shard_metadata,
    save_features_presorted,
    shard_stem_from_path,
)
from steering.feature_paths import parse_feature_dir
from steering.feature_shard_pairing import OppositePercentilePairing
from steering.feature_shard_splice import (
    PARQUET_DIRNAME,
    PROVENANCE_DIRNAME,
    recombination_seed,
    splice_feature_shard,
)
from steering.models import resolve_sae_max_context, resolve_tokenizer
from steering.parquet_io import shard_index_from_path
from steering.remote_feature_inventory import (
    normalize_remote_feature_inventory,
    remote_feature_inventory_digest,
    resolve_remote_feature_prefix,
)
from steering.remote_training import (
    BucketBackend,
    HfBucketBackend,
    RemoteObjectInfo,
    download_verified_object,
    upload_verified_object,
    verified_local_object_info,
)
from steering.remote_types import HfBucketId, NormalizedRemotePrefix
from steering.sae_resolver import SAE_WIDTH_TO_NUM_FEATURES
from steering.stats_io import load_statistics, statistics_path

_SPLICE_CONFIG_PATH = (
    Path(__file__).resolve().parents[2] / "configs" / "extraction" / "splice_feature_shards.yaml"
)

_REMOTE_FEATURE_NAME_RE = re.compile(
    rf"^shard_(?P<index>\d{{5}})_features{re.escape(COMPRESSED_EXT)}$"
)
_REMOTE_PARQUET_NAME_RE = re.compile(r"^shard_(?P<index>\d{5})\.parquet$")
_REMOTE_PROVENANCE_NAME_RE = re.compile(r"^shard_(?P<index>\d{5})\.json$")
_LOWERCASE_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_REMOTE_DOWNLOAD_DEPTH = 1
_REMOTE_UPLOAD_DEPTH = 1


def load_splice_config(
    *overrides: Mapping[str, object] | DictConfig,
) -> DictConfig:
    """Load the canonical splice schema and merge recognized overrides."""
    base = OmegaConf.load(_SPLICE_CONFIG_PATH)
    if not isinstance(base, DictConfig):
        raise TypeError(f"splice config must be a mapping, got {type(base).__name__}")
    OmegaConf.set_struct(base, True)
    merged = OmegaConf.merge(base, *overrides)
    if not isinstance(merged, DictConfig):
        raise TypeError(f"merged splice config must be a mapping, got {type(merged).__name__}")
    return merged


class SpliceJobStatus(StrEnum):
    """Terminal status for a single shard splice job."""

    ok = "ok"
    failed = "failed"


@dataclass(frozen=True)
class SpliceRunParams:
    model: str
    segment_size: int
    context_window: int
    sae_max_context: int | None
    cut_fraction: float
    seed: int
    write_provenance: bool
    skip_existing: bool
    pairing: OppositePercentilePairing | None = None
    pairing_metadata: dict[str, str | None] | None = None
    zstd_threads: int = 0
    zstd_level: int | None = None


@dataclass(frozen=True)
class SpliceJob:
    base_shard_path: Path
    parquet_path: Path
    shard_index: int
    shard_name: str
    derived_shard_path: Path  # the stem path (save_features adds _features.safetensors.zst)
    derived_parquet_path: Path
    provenance_path: Path
    # Remote-publication jobs replace the staged source path with this logical
    # base directory and require deterministic bytes for immutable upload claims.
    recorded_base_feature_dir: str | None = None


@dataclass(frozen=True)
class SpliceJobResult:
    shard_name: str
    shard_index: int
    status: SpliceJobStatus
    n_docs: int
    n_switches: int
    error_type: str | None = None
    error_message: str | None = None
    profile: dict[str, Any] | None = None


@dataclass(frozen=True)
class _PreparedFeatureSave:
    job: SpliceJob
    derived: dict
    n_docs: int
    n_switches: int
    profile: SpliceShardProfile | None
    manual_cpu_profile: bool = False


@dataclass(frozen=True)
class _PendingFeatureSave:
    prepared: _PreparedFeatureSave
    future: Future[SpliceJobResult]


@dataclass(frozen=True)
class _RemoteSettings:
    bucket_id: HfBucketId
    local_features_root: Path
    remote_features_root: NormalizedRemotePrefix
    staging_dir: Path
    expected_shard_count: int | None
    retry_timeout_seconds: float
    retry_interval_seconds: float


@dataclass(frozen=True)
class _RemoteLayout:
    feature_dir: Path
    output_dir: Path
    data_dir: Path
    feature_prefix: str
    output_prefix: str
    rank: int
    world_size: int
    settings: _RemoteSettings


@dataclass(frozen=True)
class _RemoteStaging:
    rank_dir: Path
    source_dir: Path
    output_dir: Path
    parquet_dir: Path
    provenance_dir: Path
    verify_dir: Path


@dataclass(frozen=True)
class _RemoteOutputState:
    parquet: RemoteObjectInfo | None = None
    provenance: RemoteObjectInfo | None = None
    feature: RemoteObjectInfo | None = None

    @property
    def is_complete(self) -> bool:
        return self.parquet is not None and self.provenance is not None and self.feature is not None


@dataclass(frozen=True)
class _RemoteCommitResult:
    job_result: SpliceJobResult
    output_bytes: int


@dataclass(frozen=True)
class _RemotePendingJob:
    source: RemoteObjectInfo
    job: SpliceJob
    preexisting_remote_paths: frozenset[str]


# Worker-process globals (set by the pool initializer).
_WORKER_TOKENIZER: Any | None = None
_WORKER_PARAMS: SpliceRunParams | None = None
_WORKER_PROFILE_ENABLED: bool = False


def _resolve_pairing_pool_size(cfg: DictConfig) -> int | None:
    """Validate the opt-in pairing mode without touching statistics or outputs."""
    mode = cfg.get("pairing_mode", "random")
    if mode == "random":
        return None  # The inactive search bound is deliberately ignored.
    if mode != "opposite_percentile":
        raise ValueError(
            "pairing_mode must be 'random' or 'opposite_percentile'; "
            f"got {mode!r}."
        )
    pool_size = cfg.get("pairing_candidate_pool_size", 64)
    if type(pool_size) is not int or pool_size < 2:
        raise ValueError(
            "pairing_candidate_pool_size must be an integer >= 2 for "
            f"opposite_percentile pairing; got {pool_size!r}."
        )
    return pool_size


def _default_output_dir(
    feature_dir: Path,
    cut_fraction: float,
    seed: int,
    *,
    pairing_candidate_pool_size: int | None = None,
) -> Path:
    seed_leaf = f"seed{seed}"
    if pairing_candidate_pool_size is not None:
        seed_leaf += f"_opposite_percentile_pool{pairing_candidate_pool_size}"
    return feature_dir / "splice" / f"cut_frac{cut_fraction}" / seed_leaf


def _validate_pairing_metadata(
    expected: Mapping[str, str | None],
    base_metadata: Mapping[str, str],
    *,
    source: str,
) -> None:
    """Bind each scored shard to the statistics' available extraction metadata."""
    observed = {"segment_mode": "fixed_tokens", "ignore_bos_features": "false", **base_metadata}
    mismatches = [
        f"{key}: statistics={value!r}, base={observed.get(key)!r}"
        for key, value in expected.items()
        # Older extracted shards have no width header; present widths must agree
        # even for an explicit-model run outside the canonical directory layout.
        if (key != "n_sae_features" or key in observed) and observed.get(key) != value
    ]
    if mismatches:
        raise ValueError(
            f"opposite_percentile statistics do not match base shard {source}: "
            + "; ".join(mismatches)
        )


def _load_pairing(
    feature_dir: Path, base_metadata: Mapping[str, str], pool_size: int
) -> tuple[OppositePercentilePairing, dict[str, str | None]]:
    """Load the base directory's local statistics control file once per run."""
    stats_path = statistics_path(feature_dir)
    stats = load_statistics(stats_path)
    expected: dict[str, str | None] = {
        "segment_size": str(stats.metadata.segment_size),
        "context_window": str(stats.metadata.context_window),
        "layer": str(stats.metadata.layer),
        "pooling": stats.metadata.pooling_mode,
        "segment_mode": stats.metadata.segment_mode.value,
        "l0_tier": stats.l0_tier.value if stats.l0_tier is not None else None,
        "ignore_bos_features": str(stats.ignore_bos_features).lower(),
        "n_sae_features": str(stats.metadata.n_sae_features),
    }
    _validate_pairing_metadata(expected, base_metadata, source=str(feature_dir))
    try:
        feature_stack = parse_feature_dir(feature_dir)
    except ValueError:
        # Explicit-model callers may use noncanonical directories. Feature-ID
        # bounds are still checked against the loaded table before scoring.
        pass
    else:
        expected_width = SAE_WIDTH_TO_NUM_FEATURES[feature_stack.width]
        if stats.metadata.n_sae_features != expected_width:
            raise ValueError(
                f"opposite_percentile statistics {stats_path} do not match feature_dir: "
                f"n_sae_features: statistics={stats.metadata.n_sae_features}, "
                f"feature_dir={expected_width}"
            )
    table = build_percentile_table(stats.per_feature_stats, stats.metadata.n_sae_features)
    return OppositePercentilePairing(table, candidate_pool_size=pool_size), expected


def _resolve_zstd_threads(cfg: DictConfig) -> int:
    value = cfg.get("zstd_threads", 0)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("zstd_threads must be an integer >= 0")
    return value


def _resolve_zstd_level(cfg: DictConfig) -> int | None:
    value = cfg.get("zstd_level")
    if value is not None and (type(value) is not int or not 1 <= value <= 22):
        raise ValueError("zstd_level must be null or an integer in [1, 22]")
    return value


def _resolve_run_params(
    cfg: DictConfig, feature_dir: Path, first_base_shard: Path
) -> SpliceRunParams:
    """Resolve model / segment_size / context_window from config or the base
    shard metadata + feature_dir path; an explicit config value must match."""
    zstd_threads = _resolve_zstd_threads(cfg)
    zstd_level = _resolve_zstd_level(cfg)
    pairing_pool_size = _resolve_pairing_pool_size(cfg)
    meta = read_shard_metadata(first_base_shard)
    if "segment_size" not in meta or "context_window" not in meta:
        raise ValueError(
            f"base shard {first_base_shard} is missing segment_size/context_window "
            f"metadata; cannot resolve splice params."
        )
    meta_seg = int(meta["segment_size"])
    meta_ctx = int(meta["context_window"])
    seg_size = int(cfg.segment_size) if cfg.segment_size is not None else meta_seg
    ctx = int(cfg.context_window) if cfg.context_window is not None else meta_ctx
    if seg_size != meta_seg:
        raise ValueError(f"config segment_size={seg_size} != base shard metadata {meta_seg}.")
    if ctx != meta_ctx:
        raise ValueError(f"config context_window={ctx} != base shard metadata {meta_ctx}.")
    if cfg.model is not None:
        model = str(cfg.model)
    else:
        try:
            model = parse_feature_dir(feature_dir).model
        except ValueError as exc:
            raise ValueError(
                f"could not parse model from feature_dir {feature_dir}; pass model= "
                f"explicitly. ({exc})"
            ) from exc
    sae_max_context = resolve_sae_max_context(model)
    if ctx > sae_max_context:
        raise ValueError(
            f"context_window={ctx} exceeds the model's max context "
            f"({sae_max_context}) for {model!r}."
        )
    limit = min(ctx, sae_max_context)
    if limit % seg_size != 0:
        raise ValueError(
            f"min(context_window, sae_max_context)={limit} must be a multiple of "
            f"segment_size={seg_size} so no recombined doc exceeds the truncation limit."
        )
    cut_fraction = float(cfg.cut_fraction)
    if not (0.0 <= cut_fraction <= 1.0):
        raise ValueError(f"cut_fraction must be in [0.0, 1.0]; got {cut_fraction}.")
    write_provenance = bool(cfg.write_provenance)
    skip_existing = bool(cfg.skip_existing)
    if skip_existing and not write_provenance:
        raise ValueError(
            "skip_existing requires write_provenance: existing outputs cannot be "
            "verified without provenance sidecars. Set write_provenance=true (so a "
            "resume can verify recorded params) or skip_existing=false (to always "
            "recompute)."
        )
    pairing, pairing_metadata = (
        _load_pairing(feature_dir, meta, pairing_pool_size)
        if pairing_pool_size is not None
        else (None, None)
    )
    return SpliceRunParams(
        model=model,
        segment_size=seg_size,
        context_window=ctx,
        sae_max_context=sae_max_context,
        cut_fraction=cut_fraction,
        seed=int(cfg.seed),
        write_provenance=write_provenance,
        skip_existing=skip_existing,
        pairing=pairing,
        pairing_metadata=pairing_metadata,
        zstd_threads=zstd_threads,
        zstd_level=zstd_level,
    )


def _verify_matching_provenance(
    provenance_path: Path,
    shard_index: int,
    stem: str,
    run_params: SpliceRunParams,
    *,
    expected_base_feature_dir: str | None = None,
) -> None:
    """(a) shard + parquet + sidecar all present: compare the sidecar's recorded
    derivation params against the current run and raise on any mismatch.

    A mismatch means the output_dir already holds a shard produced with DIFFERENT
    params — a mixed-provenance directory — so we fail loudly instead of silently
    skipping (or, worse, appending incompatible shards). Compares the authoritative
    keys ``build_provenance`` records (model, segment_size, context_window,
    cut_fraction, seed) plus the derived ``shard_seed`` (which folds seed +
    shard_index + cut_fraction).
    """
    try:
        sidecar = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(
            f"{stem}: provenance sidecar {provenance_path} is unreadable/corrupt "
            f"({exc}); refusing to skip. Use a fresh output_dir."
        ) from exc
    sc_schema = sidecar.get("schema_version")
    if not isinstance(sc_schema, int) or sc_schema < 2:
        raise ValueError(
            f"{stem}: provenance sidecar {provenance_path} is schema_version="
            f"{sc_schema!r} (pre-#596). skip_existing cannot upgrade it in place — "
            f"the would-have-followed splice records are missing and "
            f"feature_delta.mode=splice_counterfactual would reject it. Delete the "
            f"derived output_dir and re-splice (then recompute overlays), or run "
            f"with skip_existing=false to regenerate."
        )
    sc_params = sidecar.get("params")
    if not isinstance(sc_params, dict):
        raise ValueError(
            f"{stem}: provenance sidecar {provenance_path} has no 'params' block; "
            f"the existing output is corrupt. Use a fresh output_dir."
        )
    expected_shard_seed = recombination_seed(run_params.seed, shard_index, run_params.cut_fraction)
    recorded_threads = sc_params.get("zstd_threads", 0)
    if type(recorded_threads) is not int or recorded_threads < 0:
        raise ValueError(f"{stem}: provenance zstd_threads must be an integer >= 0")
    recorded_level = sc_params.get("zstd_level")
    if recorded_level is not None and (
        type(recorded_level) is not int or not 1 <= recorded_level <= 22
    ):
        raise ValueError(f"{stem}: provenance zstd_level must be null or an integer in [1, 22]")
    comparisons: tuple[tuple[str, object, object], ...] = (
        ("zstd_threads", recorded_threads, run_params.zstd_threads),
        ("zstd_level", recorded_level, run_params.zstd_level),
        ("model", sc_params.get("model"), run_params.model),
        ("segment_size", sc_params.get("segment_size"), run_params.segment_size),
        ("context_window", sc_params.get("context_window"), run_params.context_window),
        ("cut_fraction", sc_params.get("cut_fraction"), run_params.cut_fraction),
        ("seed", sc_params.get("seed"), run_params.seed),
        ("shard_seed", sidecar.get("shard_seed"), expected_shard_seed),
    )
    pairing_params = (
        run_params.pairing.provenance_params() if run_params.pairing is not None else {}
    )
    comparisons += tuple(
        (key, sc_params.get(key, default), pairing_params.get(key, default))
        for key, default in (
            ("pairing_mode", "random"),
            ("pairing_candidate_pool_size", None),
            ("pairing_percentile_checksum", None),
            ("pairing_algorithm", None),
        )
    )
    if expected_base_feature_dir is not None:
        comparisons += (
            (
                "base_feature_dir",
                sc_params.get("base_feature_dir"),
                expected_base_feature_dir,
            ),
        )
    mismatches = [(k, rec, cur) for k, rec, cur in comparisons if rec != cur]
    if mismatches:
        detail = "; ".join(
            f"{k}: recorded={rec!r} != requested={cur!r}" for k, rec, cur in mismatches
        )
        raise ValueError(
            f"{stem}: mixed-provenance output_dir — the existing derived shard was "
            f"produced with different params ({detail}); use a fresh output_dir or "
            f"matching params."
        )


def _build_jobs(
    base_shards: list[Path],
    parquet_by_stem: dict[str, Path],
    output_dir: Path,
    *,
    rank: int,
    world_size: int,
    skip_existing: bool,
    run_params: SpliceRunParams,
) -> tuple[list[SpliceJob], int]:
    """Pair each base shard with its stem-matching parquet; stripe across ranks by
    shard index; honor skip_existing with STRICT verification. All shards are
    selected (no range filter).

    skip_existing resume semantics for an existing output_dir:

    (a) shard + parquet + sidecar all present -> verify the sidecar's recorded params
        match this run (:func:`_verify_matching_provenance`); match -> skip, mismatch
        -> raise.
    (b) shard + parquet present, sidecar missing (write_provenance on) -> raise ValueError.
        There is no basis to verify these outputs belong to the current run params;
        delete the outputs or use a fresh output_dir.
    (c) exactly one of shard/parquet present -> raise (partial == corrupt state).

    NOTE: run() validates skip_existing=True => write_provenance=True (see
    :func:`_resolve_run_params`), so case (b) only occurs when write_provenance is on.
    """
    indexed: list[tuple[int, Path]] = []
    for path in base_shards:
        stem = shard_stem_from_path(path)
        idx = shard_index_from_path(stem)
        if idx is None:
            raise ValueError(
                f"base shard {path} is not a 'shard_NNNNN' file; the index seeds "
                f"per-shard determinism."
            )
        indexed.append((idx, path))
    indexed.sort(key=lambda ip: ip[0])
    selected = indexed[rank::world_size]

    jobs: list[SpliceJob] = []
    n_skipped = 0
    for idx, base_path in selected:
        stem = shard_stem_from_path(base_path)
        parquet_path = parquet_by_stem.get(stem)
        if parquet_path is None:
            raise ValueError(
                f"no parquet matching base shard stem {stem!r} in the data dir; "
                f"the derived parquet must be derived from the original."
            )
        derived_shard_path = output_dir / stem  # save_features appends the suffix
        derived_parquet_path = output_dir / PARQUET_DIRNAME / f"{stem}.parquet"
        provenance_path = output_dir / PROVENANCE_DIRNAME / f"{stem}.json"
        if skip_existing:
            shard_exists = (output_dir / f"{stem}_features{COMPRESSED_EXT}").exists()
            parquet_exists = derived_parquet_path.exists()
            if shard_exists != parquet_exists:
                raise ValueError(
                    f"{stem}: partial derived outputs in {output_dir} "
                    f"(shard present={shard_exists}, parquet present={parquet_exists}); "
                    f"partial outputs are a corrupted state, not a resumable one. Use a "
                    f"fresh output_dir or remove the stray file."
                )
            if shard_exists and parquet_exists:
                if provenance_path.exists():
                    _verify_matching_provenance(provenance_path, idx, stem, run_params)
                    logger.info(f"Skipping {stem} (verified derived shard + parquet + provenance)")
                    n_skipped += 1
                    continue
                # (b) shard+parquet exist but sidecar is missing — cannot verify provenance.
                raise ValueError(
                    f"existing outputs at {stem!r} have no provenance sidecar to verify "
                    f"against; delete them or use a fresh output_dir "
                    f"(shard: {output_dir / f'{stem}_features{COMPRESSED_EXT}'}, "
                    f"parquet: {derived_parquet_path})"
                )
        jobs.append(
            SpliceJob(
                base_shard_path=base_path,
                parquet_path=parquet_path,
                shard_index=idx,
                shard_name=stem,
                derived_shard_path=derived_shard_path,
                derived_parquet_path=derived_parquet_path,
                provenance_path=provenance_path,
            )
        )
    return jobs, n_skipped


def _start_job_profile(
    job: SpliceJob,
    profile_enabled: bool,
    *,
    track_process_cpu: bool = True,
) -> SpliceShardProfile | None:
    profile = (
        SpliceShardProfile.from_paths(
            shard_stem=job.shard_name,
            shard_index=job.shard_index,
            base_shard_path=job.base_shard_path,
            parquet_path=job.parquet_path,
        )
        if profile_enabled
        else None
    )
    if profile is not None:
        profile.start_total(track_process_cpu=track_process_cpu)
    return profile


def _failed_job_result(
    job: SpliceJob,
    profile: SpliceShardProfile | None,
    exc: Exception,
) -> SpliceJobResult:
    if profile is not None:
        profile.finish(
            status=SpliceJobStatus.failed.value,
            error_type=type(exc).__name__,
            error_message=str(exc),
        )
    return SpliceJobResult(
        shard_name=job.shard_name,
        shard_index=job.shard_index,
        status=SpliceJobStatus.failed,
        n_docs=0,
        n_switches=0,
        error_type=type(exc).__name__,
        error_message=str(exc),
        profile=profile.to_dict() if profile is not None else None,
    )


def _prepare_feature_save(
    job: SpliceJob,
    tokenizer: Any,
    params: SpliceRunParams,
    *,
    profile_enabled: bool = False,
    manual_cpu_profile: bool = False,
) -> _PreparedFeatureSave | SpliceJobResult:
    profile = _start_job_profile(job, profile_enabled, track_process_cpu=not manual_cpu_profile)
    manual_cpu_start = time.thread_time() if profile is not None and manual_cpu_profile else None
    try:
        if params.pairing is not None:
            if params.pairing_metadata is None:
                raise ValueError("opposite_percentile run params require statistics metadata")
            _validate_pairing_metadata(
                params.pairing_metadata,
                read_shard_metadata(job.base_shard_path),
                source=str(job.base_shard_path),
            )
        derived, texts, provenance = splice_feature_shard(
            job.base_shard_path,
            job.parquet_path,
            tokenizer,
            model=params.model,
            cut_fraction=params.cut_fraction,
            seed=params.seed,
            shard_index=job.shard_index,
            shard_name=job.shard_name,
            segment_size=params.segment_size,
            context_window=params.context_window,
            sae_max_context=params.sae_max_context,
            profile=profile,
            pairing=params.pairing,
        )
        if job.recorded_base_feature_dir is not None:
            provenance["params"]["base_feature_dir"] = job.recorded_base_feature_dir
        if params.zstd_threads:
            provenance["params"]["zstd_threads"] = params.zstd_threads
        if params.zstd_level is not None:
            provenance["params"]["zstd_level"] = params.zstd_level
        n_docs = int(derived["n_docs"])
        n_switches = sum(d["n_switches"] for d in provenance["docs"])
        if profile is not None:
            profile.record_derived(derived, n_switches)
        # Atomic writes: parquet + provenance first, then the feature shard last so
        # skip_existing only treats a shard as complete once all three are present.
        if profile is None:
            atomic_write_parquet(pa.table({"text": texts}), job.derived_parquet_path)
        else:
            with profile.time_phase("write_parquet"):
                atomic_write_parquet(pa.table({"text": texts}), job.derived_parquet_path)
        if params.write_provenance:
            if profile is None:
                atomic_json_dump(provenance, job.provenance_path)
            else:
                with profile.time_phase("write_provenance"):
                    atomic_json_dump(provenance, job.provenance_path)
        if profile is not None and manual_cpu_start is not None:
            profile.add_cpu_seconds(time.thread_time() - manual_cpu_start)
        return _PreparedFeatureSave(
            job=job,
            derived=derived,
            n_docs=n_docs,
            n_switches=n_switches,
            profile=profile,
            manual_cpu_profile=manual_cpu_profile,
        )
    except Exception as exc:  # noqa: BLE001 — recorded per-shard; run() decides fatality
        if profile is not None and manual_cpu_start is not None:
            profile.add_cpu_seconds(time.thread_time() - manual_cpu_start)
        return _failed_job_result(job, profile, exc)


def _save_prepared_features(
    prepared: _PreparedFeatureSave, *, zstd_threads: int = 0, zstd_level: int | None = None
) -> Path:
    profile = prepared.profile
    deterministic = prepared.job.recorded_base_feature_dir is not None
    if profile is None:
        return save_features_presorted(
            prepared.derived, prepared.job.derived_shard_path, deterministic=deterministic,
            zstd_threads=zstd_threads, zstd_level=zstd_level,
        )
    with profile.time_phase("save_features"):
        return save_features_presorted(
            prepared.derived, prepared.job.derived_shard_path,
            profile=profile, deterministic=deterministic,
            zstd_threads=zstd_threads, zstd_level=zstd_level,
        )


def _successful_job_result(
    prepared: _PreparedFeatureSave,
    feature_path: Path,
    params: SpliceRunParams,
) -> SpliceJobResult:
    profile = prepared.profile
    if profile is not None:
        profile.record_outputs(
            feature_path=feature_path,
            parquet_path=prepared.job.derived_parquet_path,
            provenance_path=prepared.job.provenance_path if params.write_provenance else None,
        )
        profile.finish(status=SpliceJobStatus.ok.value)
    return SpliceJobResult(
        shard_name=prepared.job.shard_name,
        shard_index=prepared.job.shard_index,
        status=SpliceJobStatus.ok,
        n_docs=prepared.n_docs,
        n_switches=prepared.n_switches,
        profile=profile.to_dict() if profile is not None else None,
    )


def _finish_prepared_feature_save(
    prepared: _PreparedFeatureSave,
    params: SpliceRunParams,
) -> SpliceJobResult:
    save_cpu_start = (
        time.thread_time() if prepared.profile is not None and prepared.manual_cpu_profile else None
    )
    try:
        feature_path = _save_prepared_features(
            prepared, zstd_threads=params.zstd_threads, zstd_level=params.zstd_level
        )
        if prepared.profile is not None and save_cpu_start is not None:
            prepared.profile.add_cpu_seconds(time.thread_time() - save_cpu_start)
        return _successful_job_result(prepared, feature_path, params)
    except Exception as exc:  # noqa: BLE001 — recorded per-shard; run() decides fatality
        if prepared.profile is not None and save_cpu_start is not None:
            prepared.profile.add_cpu_seconds(time.thread_time() - save_cpu_start)
        return _failed_job_result(prepared.job, prepared.profile, exc)


def _finish_pending_feature_save(pending: _PendingFeatureSave) -> SpliceJobResult:
    try:
        return pending.future.result()
    except Exception as exc:  # noqa: BLE001 — recorded per-shard; run() decides fatality
        return _failed_job_result(pending.prepared.job, pending.prepared.profile, exc)


def _process_job(
    job: SpliceJob,
    tokenizer: Any,
    params: SpliceRunParams,
    *,
    profile_enabled: bool = False,
) -> SpliceJobResult:
    prepared = _prepare_feature_save(job, tokenizer, params, profile_enabled=profile_enabled)
    if isinstance(prepared, SpliceJobResult):
        return prepared
    return _finish_prepared_feature_save(prepared, params)


def _init_worker(tokenizer: Any, params: SpliceRunParams, profile_enabled: bool) -> None:
    global _WORKER_TOKENIZER, _WORKER_PARAMS, _WORKER_PROFILE_ENABLED
    _WORKER_TOKENIZER = tokenizer
    _WORKER_PARAMS = params
    _WORKER_PROFILE_ENABLED = profile_enabled


def _process_job_in_worker(job: SpliceJob) -> SpliceJobResult:
    if _WORKER_TOKENIZER is None or _WORKER_PARAMS is None:
        raise RuntimeError("splice worker was not initialized")
    return _process_job(
        job,
        _WORKER_TOKENIZER,
        _WORKER_PARAMS,
        profile_enabled=_WORKER_PROFILE_ENABLED,
    )


def _run_serial(
    jobs: list[SpliceJob],
    tokenizer: Any,
    params: SpliceRunParams,
    *,
    run_profile: SpliceRunProfile | None = None,
    profile_path: Path | None = None,
) -> list[SpliceJobResult]:
    results: list[SpliceJobResult] = []
    for job in jobs:
        result = _process_job(job, tokenizer, params, profile_enabled=run_profile is not None)
        results.append(result)
        _record_profile_result(run_profile, profile_path, result)
    return results


def _record_profile_result(
    run_profile: SpliceRunProfile | None,
    profile_path: Path | None,
    result: SpliceJobResult,
) -> None:
    if run_profile is not None and profile_path is not None and result.profile is not None:
        run_profile.add_shard(result.profile)
        run_profile.write(profile_path)


def _run_serial_async_feature_save(
    jobs: list[SpliceJob],
    tokenizer: Any,
    params: SpliceRunParams,
    *,
    async_feature_save_depth: int,
    run_profile: SpliceRunProfile | None = None,
    profile_path: Path | None = None,
) -> list[SpliceJobResult]:
    """Serial splice with one bounded async feature-shard save worker.

    Parquet/provenance writes stay in the main thread and complete before a save
    future is submitted. The feature shard remains the final completion marker:
    each future is joined before that shard is reported successful, and save
    failures are returned as failed shard results.
    """
    if async_feature_save_depth < 1:
        raise ValueError("async_feature_save_depth must be >= 1 when using async feature saves.")
    results: list[SpliceJobResult] = []
    pending: list[_PendingFeatureSave] = []

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="splice-feature-save") as pool:
        for job in jobs:
            prepared = _prepare_feature_save(
                job,
                tokenizer,
                params,
                profile_enabled=run_profile is not None,
                manual_cpu_profile=run_profile is not None,
            )
            if isinstance(prepared, SpliceJobResult):
                results.append(prepared)
                _record_profile_result(run_profile, profile_path, prepared)
                continue

            while len(pending) >= async_feature_save_depth:
                result = _finish_pending_feature_save(pending.pop(0))
                results.append(result)
                _record_profile_result(run_profile, profile_path, result)

            pending.append(
                _PendingFeatureSave(
                    prepared=prepared,
                    future=pool.submit(_finish_prepared_feature_save, prepared, params),
                )
            )

        while pending:
            result = _finish_pending_feature_save(pending.pop(0))
            results.append(result)
            _record_profile_result(run_profile, profile_path, result)

    return sorted(results, key=lambda r: r.shard_index)


def _run_process_pool(
    jobs: list[SpliceJob],
    tokenizer: Any,
    params: SpliceRunParams,
    shard_parallelism: int,
    *,
    run_profile: SpliceRunProfile | None = None,
    profile_path: Path | None = None,
) -> list[SpliceJobResult]:
    if not jobs:
        return []
    try:
        pickle.dumps(tokenizer)
    except Exception as exc:
        raise ValueError(
            "shard_parallelism>1 requires a picklable tokenizer; set shard_parallelism=1."
        ) from exc
    workers = min(shard_parallelism, len(jobs))
    results: list[SpliceJobResult] = []
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=mp.get_context("spawn"),
        initializer=_init_worker,
        initargs=(tokenizer, params, run_profile is not None),
    ) as pool:
        futures = {pool.submit(_process_job_in_worker, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            _record_profile_result(run_profile, profile_path, result)
    return sorted(results, key=lambda r: r.shard_index)


def _remote_enabled(cfg: DictConfig) -> bool:
    remote = cfg.get("remote")
    if remote is None:
        return False
    enabled = remote.get("enabled", False)
    if not isinstance(enabled, bool):
        raise ValueError(f"remote.enabled must be a bool, got {enabled!r}")
    return enabled


def _strict_remote_float(value: object, *, surface: str, positive: bool) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{surface} must be a finite number, got {value!r}")
    resolved = float(value)
    if not math.isfinite(resolved) or (resolved <= 0.0 if positive else resolved < 0.0):
        relation = "> 0" if positive else ">= 0"
        raise ValueError(f"{surface} must be finite and {relation}, got {value!r}")
    return resolved


def _strict_remote_int(value: object, *, surface: str, minimum: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f"{surface} must be an integer >= {minimum}, got {value!r}")
    return int(value)


def _paths_overlap(left: Path, right: Path) -> bool:
    left = left.resolve()
    right = right.resolve()
    return left == right or left.is_relative_to(right) or right.is_relative_to(left)


def _remote_prefix_for_local(
    path: Path,
    *,
    local_root: Path,
    remote_root: NormalizedRemotePrefix,
    surface: str,
) -> str:
    return str(
        resolve_remote_feature_prefix(
            path,
            local_features_root=local_root,
            remote_features_root=remote_root,
            surface=surface,
        )
    )


def _resolve_remote_layout(cfg: DictConfig) -> _RemoteLayout:
    _resolve_zstd_threads(cfg)
    _resolve_zstd_level(cfg)
    if cfg.feature_dir is None:
        raise ValueError("feature_dir is required (the logical base feature directory).")
    if cfg.data_dir is None:
        raise ValueError("data_dir is required and must remain local in remote splice mode.")
    remote = cfg.remote
    if remote.staging_dir is None:
        raise ValueError("remote.staging_dir is required when remote.enabled=true.")

    feature_dir = Path(cfg.feature_dir)
    data_dir = Path(cfg.data_dir)
    cut_fraction = _strict_remote_float(
        cfg.cut_fraction,
        surface="cut_fraction",
        positive=False,
    )
    if cut_fraction > 1.0:
        raise ValueError(f"cut_fraction must be in [0.0, 1.0]; got {cut_fraction}.")
    if isinstance(cfg.seed, bool) or not isinstance(cfg.seed, int):
        raise ValueError(f"seed must be an integer, got {cfg.seed!r}")
    seed = int(cfg.seed)
    pairing_pool_size = _resolve_pairing_pool_size(cfg)
    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir is not None
        else _default_output_dir(
            feature_dir, cut_fraction, seed,
            pairing_candidate_pool_size=pairing_pool_size,
        )
    )
    if output_dir.resolve() == feature_dir.resolve():
        raise ValueError(
            f"output_dir ({output_dir}) must not equal the base feature_dir ({feature_dir}): "
            "derived shards reuse the base filenames."
        )

    bucket_id = HfBucketId.parse(str(remote.bucket_id))
    remote_root = NormalizedRemotePrefix.parse(str(remote.remote_features_root))
    local_root = Path(remote.local_features_root)
    staging_dir = Path(remote.staging_dir).resolve()
    feature_prefix = _remote_prefix_for_local(
        feature_dir,
        local_root=local_root,
        remote_root=remote_root,
        surface="feature_dir",
    )
    output_prefix = _remote_prefix_for_local(
        output_dir,
        local_root=local_root,
        remote_root=remote_root,
        surface="output_dir",
    )
    if feature_prefix == output_prefix:
        raise ValueError("remote feature and output prefixes must be distinct")
    if _paths_overlap(staging_dir, feature_dir) or _paths_overlap(staging_dir, output_dir):
        raise ValueError(
            "remote.staging_dir must be disjoint from the logical feature_dir and output_dir"
        )

    expected_raw = remote.expected_shard_count
    if expected_raw is None:
        expected_count = None
    elif isinstance(expected_raw, bool) or not isinstance(expected_raw, int) or expected_raw < 1:
        raise ValueError(
            f"remote.expected_shard_count must be null or a positive integer, got {expected_raw!r}"
        )
    else:
        expected_count = int(expected_raw)

    shard_parallelism = _strict_remote_int(
        cfg.shard_parallelism,
        surface="shard_parallelism",
        minimum=1,
    )
    async_depth = _strict_remote_int(
        cfg.async_feature_save_depth,
        surface="async_feature_save_depth",
        minimum=0,
    )
    if shard_parallelism != 1:
        raise ValueError("remote splice mode requires shard_parallelism=1")
    if async_depth != 0:
        raise ValueError("remote splice mode requires async_feature_save_depth=0")
    if cfg.skip_existing is not True:
        raise ValueError("remote splice mode requires skip_existing=true")
    if cfg.write_provenance is not True:
        raise ValueError("remote splice mode requires write_provenance=true")

    try:
        rank = int(os.environ.get("RANK") or 0)
        world_size = int(os.environ.get("WORLD_SIZE") or 1)
    except ValueError as exc:
        raise ValueError("RANK and WORLD_SIZE must be integers in remote splice mode") from exc
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must satisfy 0 <= RANK < WORLD_SIZE, got {rank}/{world_size}")

    settings = _RemoteSettings(
        bucket_id=bucket_id,
        local_features_root=local_root,
        remote_features_root=remote_root,
        staging_dir=staging_dir,
        expected_shard_count=expected_count,
        retry_timeout_seconds=_strict_remote_float(
            remote.retry_timeout_seconds,
            surface="remote.retry_timeout_seconds",
            positive=True,
        ),
        retry_interval_seconds=_strict_remote_float(
            remote.retry_interval_seconds,
            surface="remote.retry_interval_seconds",
            positive=False,
        ),
    )
    return _RemoteLayout(
        feature_dir=feature_dir,
        output_dir=output_dir,
        data_dir=data_dir,
        feature_prefix=feature_prefix,
        output_prefix=output_prefix,
        rank=rank,
        world_size=world_size,
        settings=settings,
    )


def _relative_remote_parts(remote_path: str, prefix: str) -> tuple[str, ...]:
    try:
        normalized = str(NormalizedRemotePrefix.parse(remote_path))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"bucket returned a non-normalized object path {remote_path!r}") from exc
    try:
        return PurePosixPath(normalized).relative_to(PurePosixPath(prefix)).parts
    except ValueError as exc:
        raise ValueError(
            f"bucket listing for {prefix!r} returned object outside the prefix: {remote_path!r}"
        ) from exc


def _validate_remote_info(info: RemoteObjectInfo, *, surface: str) -> None:
    if isinstance(info.byte_length, bool) or info.byte_length <= 0:
        raise ValueError(f"{surface} {info.path!r} has invalid byte length {info.byte_length!r}")
    if _LOWERCASE_SHA256_RE.fullmatch(info.xet_hash) is None:
        raise ValueError(f"{surface} {info.path!r} has invalid Xet hash {info.xet_hash!r}")


def _discover_remote_sources(
    listing: Sequence[RemoteObjectInfo],
    *,
    prefix: str,
    expected_shard_count: int | None,
) -> tuple[dict[int, RemoteObjectInfo], str]:
    normalized = normalize_remote_feature_inventory(
        listing,
        prefix=NormalizedRemotePrefix.parse(prefix),
    )
    sources = {entry.index: entry.info for entry in normalized}
    if not sources:
        raise ValueError(f"no direct raw feature shard objects found under {prefix}")
    if expected_shard_count is not None:
        expected = set(range(expected_shard_count))
        actual = set(sources)
        if actual != expected:
            missing = sorted(expected - actual)
            extra = sorted(actual - expected)
            raise ValueError(
                "remote source inventory is not the exact expected contiguous set "
                f"0..{expected_shard_count - 1}: missing={missing[:20]}, extra={extra[:20]}"
            )
    return sources, remote_feature_inventory_digest(normalized)


def _discover_remote_output_states(
    listing: Sequence[RemoteObjectInfo],
    *,
    prefix: str,
    source_indices: set[int],
) -> dict[int, _RemoteOutputState]:
    mutable: dict[int, dict[str, RemoteObjectInfo]] = {index: {} for index in source_indices}
    seen_paths: set[str] = set()
    for info in listing:
        parts = _relative_remote_parts(info.path, prefix)
        kind: str | None = None
        match: re.Match[str] | None = None
        if len(parts) == 1:
            match = _REMOTE_FEATURE_NAME_RE.fullmatch(parts[0])
            kind = "feature" if match is not None else None
        elif len(parts) == 2 and parts[0] == PARQUET_DIRNAME:
            match = _REMOTE_PARQUET_NAME_RE.fullmatch(parts[1])
            kind = "parquet" if match is not None else None
        elif len(parts) == 2 and parts[0] == PROVENANCE_DIRNAME:
            match = _REMOTE_PROVENANCE_NAME_RE.fullmatch(parts[1])
            kind = "provenance" if match is not None else None
        if match is None or kind is None:
            continue
        index = int(match.group("index"))
        if index not in source_indices:
            continue
        _validate_remote_info(info, surface="output object")
        if info.path in seen_paths or kind in mutable[index]:
            raise ValueError(f"duplicate remote output object in bucket listing: {info.path}")
        seen_paths.add(info.path)
        mutable[index][kind] = info
    return {
        index: _RemoteOutputState(
            parquet=items.get("parquet"),
            provenance=items.get("provenance"),
            feature=items.get("feature"),
        )
        for index, items in mutable.items()
    }


def _validate_remote_output_states(
    states: Mapping[int, _RemoteOutputState],
    *,
    owned_indices: Sequence[int],
) -> None:
    for index in owned_indices:
        state = states[index]
        parquet = state.parquet is not None
        provenance = state.provenance is not None
        feature = state.feature is not None
        if (
            (not parquet and not provenance and not feature)
            or (parquet and not feature)
            or (parquet and provenance and feature)
        ):
            continue
        raise ValueError(
            f"shard_{index:05d}: impossible remote publication state under feature-last "
            f"ordering (parquet={parquet}, provenance={provenance}, feature={feature})"
        )


def _local_parquets_for_sources(
    data_dir: Path,
    source_indices: Sequence[int],
) -> dict[str, Path]:
    parquet_paths = list_parquet_files(str(data_dir))
    parquet_by_stem = {Path(path).stem: Path(path) for path in parquet_paths}
    missing = [
        f"shard_{index:05d}"
        for index in source_indices
        if f"shard_{index:05d}" not in parquet_by_stem
    ]
    if missing:
        raise ValueError(
            "missing local source parquet(s) required by the fixed remote snapshot: "
            + ", ".join(missing[:20])
        )
    return parquet_by_stem


def _ensure_direct_directory(path: Path) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError:
        path.mkdir(parents=False)
        metadata = path.lstat()
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"runner staging path must be a direct directory: {path}")


def _is_atomic_tmp(name: str, final_pattern: re.Pattern[str]) -> bool:
    marker = ".tmp."
    if marker not in name:
        return False
    final_name, suffix = name.rsplit(marker, 1)
    return (
        final_pattern.fullmatch(final_name) is not None
        and re.fullmatch(r"\d+\.[0-9a-f]{8}", suffix) is not None
    )


def _is_download_tmp(name: str, final_pattern: re.Pattern[str]) -> bool:
    match = re.fullmatch(r"\.(.+)\.remote-tmp\.\d+\.[0-9a-f]{8}", name)
    return match is not None and final_pattern.fullmatch(match.group(1)) is not None


def _classify_staging_file(role: str, name: str) -> bool:
    if role == "source":
        return _REMOTE_FEATURE_NAME_RE.fullmatch(name) is not None or _is_download_tmp(
            name, _REMOTE_FEATURE_NAME_RE
        )
    if role == "feature":
        return _REMOTE_FEATURE_NAME_RE.fullmatch(name) is not None or _is_atomic_tmp(
            name, _REMOTE_FEATURE_NAME_RE
        )
    if role == "parquet":
        return _REMOTE_PARQUET_NAME_RE.fullmatch(name) is not None or _is_atomic_tmp(
            name, _REMOTE_PARQUET_NAME_RE
        )
    if role == "provenance":
        return _REMOTE_PROVENANCE_NAME_RE.fullmatch(name) is not None or _is_atomic_tmp(
            name, _REMOTE_PROVENANCE_NAME_RE
        )
    if role == "verify":
        return (
            _REMOTE_PROVENANCE_NAME_RE.fullmatch(name) is not None
            or _is_download_tmp(name, _REMOTE_PROVENANCE_NAME_RE)
            or _is_atomic_tmp(name, _REMOTE_PROVENANCE_NAME_RE)
        )
    raise AssertionError(f"unknown staging role {role}")


def _cleanup_rank_staging(staging: _RemoteStaging, lock_path: Path) -> None:
    expected_top = {
        lock_path.name: "lock",
        staging.source_dir.name: "dir",
        staging.output_dir.name: "dir",
        staging.verify_dir.name: "dir",
    }
    unknown: list[Path] = []
    recognized: list[Path] = []
    for child in staging.rank_dir.iterdir():
        expected = expected_top.get(child.name)
        metadata = child.lstat()
        if expected == "lock" and stat.S_ISREG(metadata.st_mode):
            continue
        if expected == "dir" and stat.S_ISDIR(metadata.st_mode):
            continue
        unknown.append(child)
    for child in staging.output_dir.iterdir():
        metadata = child.lstat()
        if child in {staging.parquet_dir, staging.provenance_dir} and stat.S_ISDIR(
            metadata.st_mode
        ):
            continue
        if stat.S_ISREG(metadata.st_mode) and _classify_staging_file("feature", child.name):
            recognized.append(child)
            continue
        unknown.append(child)
    for directory, role in (
        (staging.source_dir, "source"),
        (staging.parquet_dir, "parquet"),
        (staging.provenance_dir, "provenance"),
        (staging.verify_dir, "verify"),
    ):
        for child in directory.iterdir():
            metadata = child.lstat()
            if stat.S_ISREG(metadata.st_mode) and _classify_staging_file(role, child.name):
                recognized.append(child)
            else:
                unknown.append(child)
    if unknown:
        rendered = ", ".join(str(path) for path in sorted(set(unknown)))
        raise ValueError(
            "unknown file or directory in runner-owned staging namespace; refusing cleanup: "
            + rendered
        )
    for path in recognized:
        path.unlink()


@contextmanager
def _locked_rank_staging(layout: _RemoteLayout) -> Iterator[_RemoteStaging]:
    root = layout.settings.staging_dir
    root.mkdir(parents=True, exist_ok=True)
    rank_dir = root / f"rank{layout.rank:05d}"
    _ensure_direct_directory(rank_dir)
    source_dir = rank_dir / "source"
    output_dir = rank_dir / "output"
    verify_dir = rank_dir / "verify"
    for directory in (source_dir, output_dir, verify_dir):
        _ensure_direct_directory(directory)
    parquet_dir = output_dir / PARQUET_DIRNAME
    provenance_dir = output_dir / PROVENANCE_DIRNAME
    for directory in (parquet_dir, provenance_dir):
        _ensure_direct_directory(directory)
    staging = _RemoteStaging(
        rank_dir=rank_dir,
        source_dir=source_dir,
        output_dir=output_dir,
        parquet_dir=parquet_dir,
        provenance_dir=provenance_dir,
        verify_dir=verify_dir,
    )
    lock_path = rank_dir / ".lock"
    flags = os.O_RDWR | os.O_CREAT | getattr(os, "O_CLOEXEC", 0)
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    try:
        lock_fd = os.open(lock_path, flags, 0o600)
    except OSError as exc:
        raise ValueError(f"cannot open remote splice staging lock {lock_path}: {exc}") from exc
    try:
        opened = os.fstat(lock_fd)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"remote splice staging lock is not a direct file: {lock_path}")
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError(
                f"remote splice staging is already in use for rank {layout.rank}: {rank_dir}"
            ) from exc
        _cleanup_rank_staging(staging, lock_path)
        yield staging
    finally:
        os.close(lock_fd)


def _remote_object_path(prefix: str, *parts: str) -> str:
    return PurePosixPath(prefix, *parts).as_posix()


def _assert_runner_output(path: Path, expected_parent: Path) -> int:
    if path.parent != expected_parent:
        raise ValueError(f"runner output escaped its staging directory: {path}")
    metadata = path.lstat()
    if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
        raise ValueError(f"runner output must be a direct single-link regular file: {path}")
    return metadata.st_size


def _remote_job_paths(job: SpliceJob) -> tuple[Path, Path, Path]:
    return (
        job.derived_parquet_path,
        job.provenance_path,
        job.derived_shard_path.parent / f"{job.shard_name}_features{COMPRESSED_EXT}",
    )


def _commit_remote_job(
    backend: BucketBackend,
    layout: _RemoteLayout,
    staging: _RemoteStaging,
    job: SpliceJob,
    result: SpliceJobResult,
    preexisting_remote_paths: frozenset[str] = frozenset(),
) -> _RemoteCommitResult:
    parquet_path, provenance_path, feature_path = _remote_job_paths(job)
    _assert_runner_output(parquet_path, staging.parquet_dir)
    _assert_runner_output(provenance_path, staging.provenance_dir)
    _assert_runner_output(feature_path, staging.output_dir)
    remote_paths = (
        _remote_object_path(layout.output_prefix, PARQUET_DIRNAME, parquet_path.name),
        _remote_object_path(layout.output_prefix, PROVENANCE_DIRNAME, provenance_path.name),
        _remote_object_path(layout.output_prefix, feature_path.name),
    )
    local_paths = (parquet_path, provenance_path, feature_path)
    claimed_identities = tuple(
        verified_local_object_info(local_path=local_path, remote_path=remote_path)
        for local_path, remote_path in zip(local_paths, remote_paths, strict=True)
    )
    output_bytes = 0
    for local_path, remote_path, claimed_identity in zip(
        local_paths, remote_paths, claimed_identities, strict=True
    ):
        verified = upload_verified_object(
            backend,
            local_path=local_path,
            remote_path=remote_path,
            timeout_seconds=layout.settings.retry_timeout_seconds,
            retry_interval_seconds=layout.settings.retry_interval_seconds,
            expected_local=claimed_identity,
        )
        if remote_path not in preexisting_remote_paths:
            output_bytes += verified.byte_length
    for path in (parquet_path, provenance_path, feature_path):
        path.unlink()
    return _RemoteCommitResult(job_result=result, output_bytes=output_bytes)


def _publication_failed_result(
    result: SpliceJobResult,
    exc: BaseException,
) -> SpliceJobResult:
    profile = dict(result.profile) if result.profile is not None else None
    if profile is not None:
        profile["status"] = SpliceJobStatus.failed.value
        profile["error_type"] = type(exc).__name__
        profile["error_message"] = str(exc)
    return SpliceJobResult(
        shard_name=result.shard_name,
        shard_index=result.shard_index,
        status=SpliceJobStatus.failed,
        n_docs=result.n_docs,
        n_switches=result.n_switches,
        error_type=type(exc).__name__,
        error_message=str(exc),
        profile=profile,
    )


def _record_remote_profile(
    run_profile: SpliceRunProfile | None,
    profile_path: Path | None,
    result: SpliceJobResult,
    *,
    committed: int,
    source_bytes: int,
    output_bytes: int,
) -> None:
    if run_profile is None or profile_path is None:
        return
    if result.profile is not None:
        run_profile.add_shard(result.profile)
    run_profile.update_runtime_totals(
        committed=committed,
        source_bytes_downloaded=source_bytes,
        output_bytes_uploaded=output_bytes,
    )
    run_profile.write(profile_path)


def _download_remote_source(
    backend: BucketBackend,
    layout: _RemoteLayout,
    expected: RemoteObjectInfo,
    final_path: Path,
    cancellation_event: threading.Event,
) -> RemoteObjectInfo:
    return download_verified_object(
        backend,
        expected=expected,
        final_path=final_path,
        timeout_seconds=layout.settings.retry_timeout_seconds,
        retry_interval_seconds=layout.settings.retry_interval_seconds,
        cancellation_event=cancellation_event,
    )


def _completed_download(
    info: RemoteObjectInfo,
) -> Future[tuple[RemoteObjectInfo, bool]]:
    future: Future[tuple[RemoteObjectInfo, bool]] = Future()
    future.set_result((info, False))
    return future


def _run_remote_pipeline(
    *,
    backend: BucketBackend,
    layout: _RemoteLayout,
    staging: _RemoteStaging,
    pending: Sequence[_RemotePendingJob],
    bootstrap: RemoteObjectInfo,
    tokenizer: Any,
    params: SpliceRunParams,
    run_profile: SpliceRunProfile | None,
    profile_path: Path | None,
    initial_source_bytes: int,
) -> tuple[int, int, int]:
    if not pending:
        return 0, initial_source_bytes, 0
    download_pool = ThreadPoolExecutor(
        max_workers=_REMOTE_DOWNLOAD_DEPTH, thread_name_prefix="splice-remote-download"
    )
    upload_pool = ThreadPoolExecutor(
        max_workers=_REMOTE_UPLOAD_DEPTH, thread_name_prefix="splice-remote-upload"
    )
    source_bytes = initial_source_bytes
    output_bytes = 0
    committed = 0
    upload_future: Future[_RemoteCommitResult] | None = None
    upload_result: SpliceJobResult | None = None
    next_download: Future[tuple[RemoteObjectInfo, bool]] | None = None
    download_cancellation = threading.Event()
    first_pending = pending[0]
    first_info = first_pending.source
    first_job = first_pending.job
    if first_info.path == bootstrap.path and first_job.base_shard_path.exists():
        next_download = _completed_download(first_info)
    else:
        next_download = download_pool.submit(
            lambda info=first_info, path=first_job.base_shard_path: (
                _download_remote_source(backend, layout, info, path, download_cancellation),
                True,
            )
        )
    try:
        for position, pending_job in enumerate(pending):
            source_info = pending_job.source
            job = pending_job.job
            assert next_download is not None
            downloaded, transferred = next_download.result()
            if downloaded != source_info:
                raise RuntimeError(
                    f"download scheduler returned the wrong source for {job.shard_name}"
                )
            if transferred:
                source_bytes += downloaded.byte_length
            if position + 1 < len(pending):
                following = pending[position + 1]
                following_info = following.source
                following_job = following.job
                next_download = download_pool.submit(
                    lambda info=following_info, path=following_job.base_shard_path: (
                        _download_remote_source(backend, layout, info, path, download_cancellation),
                        True,
                    )
                )
            else:
                next_download = None

            result = _process_job(
                job,
                tokenizer,
                params,
                profile_enabled=run_profile is not None,
            )
            if result.status == SpliceJobStatus.failed:
                download_cancellation.set()
                if next_download is not None:
                    next_download.cancel()
                if upload_future is not None and upload_result is not None:
                    try:
                        completed = upload_future.result()
                    except BaseException as exc:
                        failed_upload = _publication_failed_result(upload_result, exc)
                        _record_remote_profile(
                            run_profile,
                            profile_path,
                            failed_upload,
                            committed=committed,
                            source_bytes=source_bytes,
                            output_bytes=output_bytes,
                        )
                        _record_remote_profile(
                            run_profile,
                            profile_path,
                            result,
                            committed=committed,
                            source_bytes=source_bytes,
                            output_bytes=output_bytes,
                        )
                        raise
                    committed += 1
                    output_bytes += completed.output_bytes
                    _record_remote_profile(
                        run_profile,
                        profile_path,
                        completed.job_result,
                        committed=committed,
                        source_bytes=source_bytes,
                        output_bytes=output_bytes,
                    )
                _record_remote_profile(
                    run_profile,
                    profile_path,
                    result,
                    committed=committed,
                    source_bytes=source_bytes,
                    output_bytes=output_bytes,
                )
                raise RuntimeError(
                    f"Splice FAILED {result.shard_name}: {result.error_type}: "
                    f"{result.error_message}"
                )

            parquet_path, provenance_path, feature_path = _remote_job_paths(job)
            _assert_runner_output(parquet_path, staging.parquet_dir)
            _assert_runner_output(provenance_path, staging.provenance_dir)
            _assert_runner_output(feature_path, staging.output_dir)
            job.base_shard_path.unlink()

            if upload_future is not None and upload_result is not None:
                try:
                    completed = upload_future.result()
                except BaseException as exc:
                    failed = _publication_failed_result(upload_result, exc)
                    _record_remote_profile(
                        run_profile,
                        profile_path,
                        failed,
                        committed=committed,
                        source_bytes=source_bytes,
                        output_bytes=output_bytes,
                    )
                    raise
                committed += 1
                output_bytes += completed.output_bytes
                _record_remote_profile(
                    run_profile,
                    profile_path,
                    completed.job_result,
                    committed=committed,
                    source_bytes=source_bytes,
                    output_bytes=output_bytes,
                )

            upload_result = result
            upload_future = upload_pool.submit(
                _commit_remote_job,
                backend,
                layout,
                staging,
                job,
                result,
                pending_job.preexisting_remote_paths,
            )

        if upload_future is not None and upload_result is not None:
            try:
                completed = upload_future.result()
            except BaseException as exc:
                failed = _publication_failed_result(upload_result, exc)
                _record_remote_profile(
                    run_profile,
                    profile_path,
                    failed,
                    committed=committed,
                    source_bytes=source_bytes,
                    output_bytes=output_bytes,
                )
                raise
            committed += 1
            output_bytes += completed.output_bytes
            _record_remote_profile(
                run_profile,
                profile_path,
                completed.job_result,
                committed=committed,
                source_bytes=source_bytes,
                output_bytes=output_bytes,
            )
        return committed, source_bytes, output_bytes
    finally:
        download_cancellation.set()
        if next_download is not None:
            next_download.cancel()
        download_pool.shutdown(wait=True, cancel_futures=True)
        upload_pool.shutdown(wait=True, cancel_futures=True)


def _run_remote(cfg: DictConfig) -> None:
    wall_start = time.perf_counter()
    layout = _resolve_remote_layout(cfg)
    backend: BucketBackend = HfBucketBackend(layout.settings.bucket_id)
    source_listing = backend.list(layout.feature_prefix)
    output_listing = backend.list(layout.output_prefix)
    sources, inventory_digest = _discover_remote_sources(
        source_listing,
        prefix=layout.feature_prefix,
        expected_shard_count=layout.settings.expected_shard_count,
    )
    source_indices = sorted(sources)
    parquet_by_stem = _local_parquets_for_sources(layout.data_dir, source_indices)
    owned_indices = [index for index in source_indices if index % layout.world_size == layout.rank]
    states = _discover_remote_output_states(
        output_listing,
        prefix=layout.output_prefix,
        source_indices=set(source_indices),
    )
    _validate_remote_output_states(states, owned_indices=owned_indices)
    logger.info(
        f"Remote splice snapshot: rank={layout.rank}/{layout.world_size} "
        f"sources={len(source_indices)} owned={len(owned_indices)} "
        f"source_bytes={sum(entry.byte_length for entry in sources.values())} "
        f"inventory_digest={inventory_digest}"
    )
    if not owned_indices:
        logger.info(
            f"Remote splice complete: 0 committed, 0 failed, 0 skipped; "
            f"inventory_digest={inventory_digest}"
        )
        return

    run_profile: SpliceRunProfile | None = None
    profile_path: Path | None = None
    committed = 0
    source_bytes = 0
    output_bytes = 0
    n_skipped = 0
    with _locked_rank_staging(layout) as staging:
        complete_indices = [index for index in owned_indices if states[index].is_complete]
        pending_indices = [index for index in owned_indices if not states[index].is_complete]
        # A pending body is both the independent metadata authority and useful
        # pipeline input. All-skip runs still read one snapshotted raw body so
        # completed provenance can never define the values used to verify itself.
        bootstrap_index = pending_indices[0] if pending_indices else owned_indices[0]
        bootstrap = sources[bootstrap_index]
        bootstrap_path = staging.source_dir / PurePosixPath(bootstrap.path).name
        tokenizer: Any | None = None
        download_verified_object(
            backend,
            expected=bootstrap,
            final_path=bootstrap_path,
            timeout_seconds=layout.settings.retry_timeout_seconds,
            retry_interval_seconds=layout.settings.retry_interval_seconds,
        )
        source_bytes += bootstrap.byte_length
        params = _resolve_run_params(cfg, layout.feature_dir, bootstrap_path)
        if pending_indices:
            tokenizer = resolve_tokenizer(params.model, load_chat_template=False)

        for index in complete_indices:
            provenance = states[index].provenance
            assert provenance is not None
            stem = f"shard_{index:05d}"
            verify_path = staging.verify_dir / f"{stem}.json"
            try:
                download_verified_object(
                    backend,
                    expected=provenance,
                    final_path=verify_path,
                    timeout_seconds=layout.settings.retry_timeout_seconds,
                    retry_interval_seconds=layout.settings.retry_interval_seconds,
                )
                _verify_matching_provenance(
                    verify_path,
                    index,
                    stem,
                    params,
                    expected_base_feature_dir=str(layout.feature_dir),
                )
            finally:
                verify_path.unlink(missing_ok=True)
            n_skipped += 1
            logger.info(f"Skipping {stem} (verified remote triplet + provenance)")

        if bootstrap_path.exists() and bootstrap_index not in pending_indices:
            bootstrap_path.unlink()
        pending: list[_RemotePendingJob] = []
        for index in pending_indices:
            stem = f"shard_{index:05d}"
            source = sources[index]
            state = states[index]
            pending.append(
                _RemotePendingJob(
                    source=source,
                    job=SpliceJob(
                        base_shard_path=staging.source_dir / PurePosixPath(source.path).name,
                        parquet_path=parquet_by_stem[stem],
                        shard_index=index,
                        shard_name=stem,
                        derived_shard_path=staging.output_dir / stem,
                        derived_parquet_path=staging.parquet_dir / f"{stem}.parquet",
                        provenance_path=staging.provenance_dir / f"{stem}.json",
                        recorded_base_feature_dir=str(layout.feature_dir),
                    ),
                    preexisting_remote_paths=frozenset(
                        info.path
                        for info in (state.parquet, state.provenance, state.feature)
                        if info is not None
                    ),
                )
            )

        if cfg.profile:
            profile_path = _profile_path(cfg, layout.rank, layout.world_size)
            run_profile = SpliceRunProfile(
                config={
                    "feature_dir": str(layout.feature_dir),
                    "output_dir": str(layout.output_dir),
                    "model": params.model,
                    "segment_size": params.segment_size,
                    "context_window": params.context_window,
                    "cut_fraction": params.cut_fraction,
                    "seed": params.seed,
                    "async_feature_save_depth": 0,
                    "zstd_threads": params.zstd_threads,
                    "zstd_level": params.zstd_level,
                    **(params.pairing.provenance_params() if params.pairing is not None else {}),
                    "remote": {
                        "enabled": True,
                        "bucket_id": str(layout.settings.bucket_id),
                        "feature_prefix": layout.feature_prefix,
                        "output_prefix": layout.output_prefix,
                        "source_inventory_digest": inventory_digest,
                        "expected_shard_count": layout.settings.expected_shard_count,
                        "download_lookahead": _REMOTE_DOWNLOAD_DEPTH,
                        "upload_lookahead": _REMOTE_UPLOAD_DEPTH,
                    },
                },
                rank=layout.rank,
                world_size=layout.world_size,
                shard_parallelism=1,
                n_skipped=n_skipped,
                runtime_totals={
                    "committed": 0,
                    "source_bytes_downloaded": source_bytes,
                    "output_bytes_uploaded": 0,
                },
            )
            run_profile.start()

        pipeline_completed = False
        try:
            if pending:
                assert tokenizer is not None
                committed, source_bytes, output_bytes = _run_remote_pipeline(
                    backend=backend,
                    layout=layout,
                    staging=staging,
                    pending=pending,
                    bootstrap=bootstrap,
                    tokenizer=tokenizer,
                    params=params,
                    run_profile=run_profile,
                    profile_path=profile_path,
                    initial_source_bytes=source_bytes,
                )
            pipeline_completed = True
        finally:
            if run_profile is not None and profile_path is not None:
                if pipeline_completed:
                    run_profile.update_runtime_totals(
                        committed=committed,
                        source_bytes_downloaded=source_bytes,
                        output_bytes_uploaded=output_bytes,
                    )
                run_profile.finish(time.perf_counter() - wall_start)
                run_profile.write(profile_path)

    wall_seconds = time.perf_counter() - wall_start
    logger.info(
        f"Remote splice complete: {committed} committed, 0 failed, {n_skipped} skipped; "
        f"source_bytes_downloaded={source_bytes} output_bytes_uploaded={output_bytes} "
        f"wall_seconds={wall_seconds:.3f} inventory_digest={inventory_digest}"
    )


def run(cfg: DictConfig) -> None:
    if _remote_enabled(cfg):
        _run_remote(cfg)
        return

    if cfg.feature_dir is None:
        raise ValueError("feature_dir is required (the base feature directory).")
    feature_dir = Path(cfg.feature_dir)
    base_shards = glob_features(feature_dir)
    if not base_shards:
        raise ValueError(f"no base feature shards found in {feature_dir}.")

    params = _resolve_run_params(cfg, feature_dir, base_shards[0])
    tokenizer = resolve_tokenizer(params.model, load_chat_template=False)

    data_dir = cfg.data_dir
    parquet_paths = list_parquet_files(str(data_dir) if data_dir is not None else None)
    parquet_by_stem = {Path(p).stem: Path(p) for p in parquet_paths}

    output_dir = (
        Path(cfg.output_dir)
        if cfg.output_dir is not None
        else _default_output_dir(
            feature_dir, params.cut_fraction, params.seed,
            pairing_candidate_pool_size=(
                params.pairing.candidate_pool_size if params.pairing is not None else None
            ),
        )
    )
    # Derived shards reuse the base shard filenames, so output_dir == feature_dir would
    # os.replace() the IMMUTABLE base shards in place (reporting success). Forbid exact
    # equality (by resolved path); the default <feature_dir>/splice/... subdir stays legal.
    if output_dir.resolve() == feature_dir.resolve():
        raise ValueError(
            f"output_dir ({output_dir}) must not equal the base feature_dir "
            f"({feature_dir}): derived shards reuse the base filenames, so writing there "
            f"would overwrite the immutable base shards. Pick a separate output_dir (the "
            f"default <feature_dir>/splice/... subdir is fine)."
        )
    rank = int(os.environ.get("RANK") or 0)
    world_size = int(os.environ.get("WORLD_SIZE") or 1)

    jobs, n_skipped = _build_jobs(
        base_shards,
        parquet_by_stem,
        output_dir,
        rank=rank,
        world_size=world_size,
        skip_existing=params.skip_existing,
        run_params=params,
    )
    # Validate shard_parallelism BEFORE any output-dir mkdir (mirrors cut_fraction's
    # fail-before-side-effects ordering) so an invalid value creates no directories.
    shard_parallelism = int(cfg.shard_parallelism)
    if shard_parallelism < 1:
        raise ValueError(f"shard_parallelism must be >= 1, got {shard_parallelism}")
    async_feature_save_depth = int(cfg.async_feature_save_depth)
    if async_feature_save_depth < 0:
        raise ValueError(f"async_feature_save_depth must be >= 0, got {async_feature_save_depth}")
    if async_feature_save_depth > 0 and shard_parallelism > 1:
        raise ValueError(
            "async_feature_save_depth>0 is only supported with shard_parallelism=1; "
            "process-pool shard parallelism already isolates per-shard saves."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / PARQUET_DIRNAME).mkdir(parents=True, exist_ok=True)
    if params.write_provenance:
        (output_dir / PROVENANCE_DIRNAME).mkdir(parents=True, exist_ok=True)

    run_profile: SpliceRunProfile | None = None
    profile_path: Path | None = None
    if cfg.profile:
        profile_path = _profile_path(cfg, rank, world_size)
        run_profile = SpliceRunProfile(
            config={
                "feature_dir": str(feature_dir),
                "output_dir": str(output_dir),
                "model": params.model,
                "segment_size": params.segment_size,
                "context_window": params.context_window,
                "cut_fraction": params.cut_fraction,
                "seed": params.seed,
                "async_feature_save_depth": async_feature_save_depth,
                "zstd_threads": params.zstd_threads,
                "zstd_level": params.zstd_level,
                **(params.pairing.provenance_params() if params.pairing is not None else {}),
            },
            rank=rank,
            world_size=world_size,
            shard_parallelism=shard_parallelism,
            n_skipped=n_skipped,
        )
        run_profile.start()

    wall_start = time.perf_counter()
    if shard_parallelism > 1:
        results = _run_process_pool(
            jobs,
            tokenizer,
            params,
            shard_parallelism,
            run_profile=run_profile,
            profile_path=profile_path,
        )
    elif async_feature_save_depth > 0:
        results = _run_serial_async_feature_save(
            jobs,
            tokenizer,
            params,
            async_feature_save_depth=async_feature_save_depth,
            run_profile=run_profile,
            profile_path=profile_path,
        )
    else:
        results = _run_serial(
            jobs, tokenizer, params, run_profile=run_profile, profile_path=profile_path
        )
    wall_seconds = time.perf_counter() - wall_start

    n_ok = sum(1 for r in results if r.status == SpliceJobStatus.ok)
    n_failed = sum(1 for r in results if r.status == SpliceJobStatus.failed)
    for r in results:
        if r.status == SpliceJobStatus.ok:
            logger.info(f"Spliced {r.shard_name}: {r.n_docs} docs, {r.n_switches} switches")
        else:
            logger.error(f"Splice FAILED {r.shard_name}: {r.error_type}: {r.error_message}")

    if run_profile is not None and profile_path is not None:
        run_profile.finish(wall_seconds)
        run_profile.write(profile_path)

    logger.info(
        f"Splice complete: {n_ok} ok, {n_failed} failed, {n_skipped} skipped "
        f"(of {len(jobs) + n_skipped} selected) -> {output_dir}"
    )
    if n_failed:
        raise RuntimeError(
            f"{n_failed} shard(s) failed to splice; see the logged errors above. "
            f"A failed shard signals corrupt/misaligned data, not a skippable state."
        )


def _profile_path(cfg: DictConfig, rank: int, world_size: int) -> Path:
    path = Path(cfg.profile_summary_output)
    if world_size <= 1:
        return path
    return path.with_name(f"{path.stem}.rank{rank:05d}{path.suffix}")


@hydra.main(
    version_base=None, config_path="../../configs/extraction", config_name="splice_feature_shards"
)
def main(cfg: DictConfig) -> None:
    run(cfg)


if __name__ == "__main__":
    main()
