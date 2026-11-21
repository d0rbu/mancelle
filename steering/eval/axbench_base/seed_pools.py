"""Stage 0b: exact BOS-free seed slicing and independent prefix extraction."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import platform
import random
import shutil
import sys
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import torch.distributed as dist
from loguru import logger

from nanochat.atomic_io import atomic_json_dump, atomic_write_parquet
from nanochat.dataset import list_parquet_files
from nanochat.tokenizer import get_tokenizer
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    effective_extraction_model_id,
)
from steering.eval.axbench_base.extraction import _sae_hook_name
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
    SeedPoolExpectation,
    compute_extraction_fingerprint,
    compute_seed_fingerprint,
    expectation_from_config,
    load_seed_pool,
    save_seed_pool,
    seed_config_fingerprint,
    seed_config_payload,
    sha256_file,
    validate_seed_row,
)
from steering.eval.axbench_base.types import AxBenchBaseError, SeedRow
from steering.eval.segment_mode_guard import require_fixed_feature_dir_segment_mode
from steering.extract_features import process_shard
from steering.feature_io import MetadataKey, feature_path, load_features
from steering.feature_paths import parse_feature_dir, resolve_data_dir
from steering.sae_resolver import L0Tier, parse_neuronpedia_source, resolve_sae
from steering.segmentation import (
    SegmentMode,
    compute_tokenizer_checksum,
    resolve_bos_token_id,
)

_FRESH_RUN_TAG = (
    "Existing Stage 0b artifacts cannot be migrated because BOS ownership is "
    "not recoverable; use a fresh run tag."
)

_SKIP_REASONS = (
    "source_too_short",
    "empty_prefix",
    "empty_continuation",
    "extraction_empty",
    "extraction_over_limit",
    "extraction_entirely_unpoolable",
)

_CANDIDATE_SCHEMA = pa.schema(
    [
        ("candidate_index", pa.int64()),
        ("text", pa.string()),
        ("shard_idx", pa.int32()),
        ("doc_idx", pa.int32()),
        ("source_nanochat_token_count", pa.int32()),
        ("prefix_source_token_start", pa.int32()),
        ("prefix_source_token_end", pa.int32()),
        ("prefix_content_nanochat_tokens", pa.list_(pa.int32())),
        ("continuation_text", pa.string()),
        ("continuation_content_nanochat_tokens", pa.list_(pa.int32())),
        ("extraction_token_count", pa.int32()),
    ]
)


@dataclass(frozen=True)
class SeedCandidate:
    candidate_index: int
    text: str
    shard_idx: int
    doc_idx: int
    source_nanochat_token_count: int
    prefix_source_token_start: int
    prefix_source_token_end: int
    prefix_content_nanochat_tokens: list[int]
    continuation_text: str
    continuation_content_nanochat_tokens: list[int]
    extraction_token_count: int

    def parquet_row(self) -> dict[str, Any]:
        return asdict(self)


def _list_parquets_for_feature_dir(cfg: AxBenchBaseConfig) -> list[Path]:
    phase = parse_feature_dir(Path(cfg.feature_dir)).dataset_phase.value
    data_dir = resolve_data_dir(None, phase)
    return [Path(path) for path in list_parquet_files(str(data_dir))]


def _resolve_parquet(parquets: list[Path], shard_idx: int) -> Path:
    if shard_idx < 0 or shard_idx >= len(parquets):
        raise AxBenchBaseError(
            f"validation_shards contains {shard_idx}, but only "
            f"{len(parquets)} parquet shards exist"
        )
    path = parquets[shard_idx]
    if not path.is_file():
        raise AxBenchBaseError(f"Validation parquet does not exist: {path}")
    return path


def _flat_token_ids(encoding: Any) -> list[int]:
    ids = encoding["input_ids"] if isinstance(encoding, dict) else encoding.input_ids
    if ids and isinstance(ids[0], list):
        ids = ids[0]
    return [int(token) for token in ids]


def _build_structural_candidate(
    *,
    nanochat_tokenizer: Any,
    extraction_tokenizer: Any,
    text: str,
    shard_idx: int,
    doc_idx: int,
    candidate_index: int,
    continuation_length: int,
    prefix_cap: int,
    extraction_limit: int,
    ignore_bos_features: bool,
) -> tuple[SeedCandidate | None, str | None]:
    """Build one exact source-slice candidate, returning a named skip reason."""

    source_tokens = [int(token) for token in nanochat_tokenizer.encode(text)]
    source_count = len(source_tokens)
    if source_count < 2 * continuation_length:
        return None, "source_too_short"
    prefix_end = source_count - continuation_length
    prefix_start = max(0, prefix_end - prefix_cap)
    prefix_tokens = source_tokens[prefix_start:prefix_end]
    continuation_tokens = source_tokens[prefix_end:]
    prefix_text = nanochat_tokenizer.decode(prefix_tokens)
    continuation_text = nanochat_tokenizer.decode(continuation_tokens)
    if not prefix_tokens or not prefix_text:
        return None, "empty_prefix"
    if not continuation_tokens:
        return None, "empty_continuation"

    extraction_tokens = _flat_token_ids(
        extraction_tokenizer(prefix_text, padding=False)
    )
    if not extraction_tokens:
        return None, "extraction_empty"
    if len(extraction_tokens) > extraction_limit:
        return None, "extraction_over_limit"
    if ignore_bos_features:
        extraction_bos = resolve_bos_token_id(extraction_tokenizer)
        if extraction_bos is None:
            raise AxBenchBaseError(
                "feature_dir requires BOS feature masking but the extraction "
                "tokenizer has no BOS token id"
            )
        if all(token == extraction_bos for token in extraction_tokens):
            return None, "extraction_entirely_unpoolable"

    return (
        SeedCandidate(
            candidate_index=candidate_index,
            text=prefix_text,
            shard_idx=shard_idx,
            doc_idx=doc_idx,
            source_nanochat_token_count=source_count,
            prefix_source_token_start=prefix_start,
            prefix_source_token_end=prefix_end,
            prefix_content_nanochat_tokens=prefix_tokens,
            continuation_text=continuation_text,
            continuation_content_nanochat_tokens=continuation_tokens,
            extraction_token_count=len(extraction_tokens),
        ),
        None,
    )


def _stream_candidates(
    *,
    cfg: AxBenchBaseConfig,
    nanochat_tokenizer: Any,
    extraction_tokenizer: Any,
    extraction_limit: int,
) -> tuple[list[SeedCandidate], dict[str, int], dict[str, str]]:
    parquets = _list_parquets_for_feature_dir(cfg)
    parsed = parse_feature_dir(Path(cfg.feature_dir))
    candidates: list[SeedCandidate] = []
    skips: Counter[str] = Counter()
    source_hashes: dict[str, str] = {}
    for shard_idx in cfg.validation_shards:
        parquet_path = _resolve_parquet(parquets, shard_idx)
        source_hashes[str(parquet_path.resolve())] = sha256_file(parquet_path)
        parquet = pq.ParquetFile(parquet_path)
        if "text" not in parquet.schema_arrow.names:
            raise AxBenchBaseError(f"Validation parquet has no text column: {parquet_path}")
        doc_idx = 0
        for row_group in range(parquet.num_row_groups):
            texts = parquet.read_row_group(row_group, columns=["text"])["text"].to_pylist()
            for raw_text in texts:
                if not isinstance(raw_text, str):
                    raise AxBenchBaseError(
                        f"Validation source text must be str at shard={shard_idx} "
                        f"doc={doc_idx}, got {type(raw_text).__name__}"
                    )
                candidate, reason = _build_structural_candidate(
                    nanochat_tokenizer=nanochat_tokenizer,
                    extraction_tokenizer=extraction_tokenizer,
                    text=raw_text,
                    shard_idx=shard_idx,
                    doc_idx=doc_idx,
                    candidate_index=len(candidates),
                    continuation_length=cfg.seed_continuation_nanochat_tokens,
                    prefix_cap=cfg.seed_prefix_max_nanochat_tokens,
                    extraction_limit=extraction_limit,
                    ignore_bos_features=parsed.ignore_bos_features,
                )
                if candidate is None:
                    assert reason is not None
                    skips[reason] += 1
                else:
                    candidates.append(candidate)
                doc_idx += 1
    skip_counts = {reason: int(skips[reason]) for reason in _SKIP_REASONS}
    unknown_reasons = sorted(set(skips) - set(_SKIP_REASONS))
    if unknown_reasons:
        raise AxBenchBaseError(
            f"Stage 0b produced unknown skip reasons: {unknown_reasons}"
        )
    return candidates, skip_counts, source_hashes


def _candidate_digest(
    candidates: list[SeedCandidate],
    skips: dict[str, int],
    source_hashes: dict[str, str],
) -> str:
    digest = hashlib.sha256()
    for candidate in candidates:
        digest.update(canonical_json(candidate.parquet_row()).encode("utf-8"))
        digest.update(b"\n")
    digest.update(canonical_json({"skip_counts": skips}).encode("utf-8"))
    digest.update(canonical_json({"source_parquet_hashes": source_hashes}).encode("utf-8"))
    return digest.hexdigest()


def _require_rank_agreement(label: str, value: Any, *, world_size: int) -> None:
    if world_size == 1:
        return
    if not dist.is_available() or not dist.is_initialized():
        raise AxBenchBaseError(
            f"Stage 0b world_size={world_size} requires an initialized process group"
        )
    gathered: list[Any] = [None] * world_size
    dist.all_gather_object(gathered, value)
    if any(item != gathered[0] for item in gathered[1:]):
        raise AxBenchBaseError(
            f"Stage 0b ranks disagree on {label}: {gathered}"
        )


def _collective_call(
    label: str,
    operation: Any,
    *,
    rank: int,
    world_size: int,
) -> Any:
    """Run rank-local Python work without stranding peers on failure."""

    result: Any = None
    error: str | None = None
    try:
        result = operation()
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
    record = {"rank": rank, "error": error}
    records: list[dict[str, Any]] = [record]
    if world_size > 1:
        gathered: list[Any] = [None] * world_size
        dist.all_gather_object(gathered, record)
        records = gathered
    errors = [item for item in records if item["error"] is not None]
    if errors:
        raise AxBenchBaseError(
            f"Stage 0b {label} failed collectively: {canonical_json(errors)}"
        )
    return result


def _barrier(world_size: int) -> None:
    if world_size > 1:
        dist.barrier()


def _candidate_shards(
    candidates: list[SeedCandidate], *, shard_size: int
) -> list[list[SeedCandidate]]:
    return [candidates[start : start + shard_size] for start in range(0, len(candidates), shard_size)]


def _write_or_validate_candidate_shards(
    *,
    root: Path,
    candidate_shards: list[list[SeedCandidate]],
    candidate_digest: str,
    skip_counts: dict[str, int],
    source_hashes: dict[str, str],
) -> list[Path]:
    candidate_root = root / "candidates"
    candidate_root.mkdir(parents=True, exist_ok=True)
    expected_paths: list[Path] = []
    shard_entries: list[dict[str, Any]] = []
    for shard_index, candidates in enumerate(candidate_shards):
        path = candidate_root / f"candidate_{shard_index:05d}.parquet"
        table = pa.Table.from_pylist(
            [candidate.parquet_row() for candidate in candidates],
            schema=_CANDIDATE_SCHEMA,
        )
        if path.exists():
            existing = pq.read_table(path)
            if not existing.schema.equals(_CANDIDATE_SCHEMA) or not existing.equals(table):
                raise AxBenchBaseError(
                    f"Committed candidate shard does not match this run: {path}. "
                    f"{_FRESH_RUN_TAG}"
                )
        else:
            atomic_write_parquet(table, path)
        expected_paths.append(path)
        shard_entries.append(
            {
                "name": path.name,
                "sha256": sha256_file(path),
                "document_count": len(candidates),
            }
        )
    stale = sorted(
        path.name
        for path in candidate_root.glob("candidate_*.parquet")
        if path not in set(expected_paths)
    )
    if stale:
        raise AxBenchBaseError(
            f"Stale committed candidate shards are present: {stale}. {_FRESH_RUN_TAG}"
        )
    plan = {
        "schema_version": SEED_SCHEMA_VERSION,
        "candidate_digest": candidate_digest,
        "candidate_count": sum(len(shard) for shard in candidate_shards),
        "skip_counts": skip_counts,
        "source_parquet_hashes": source_hashes,
        "shards": shard_entries,
    }
    plan_path = root / "candidate_plan.json"
    if plan_path.exists():
        if json.loads(plan_path.read_text(encoding="utf-8")) != plan:
            raise AxBenchBaseError(
                f"Candidate plan mismatch at {plan_path}. {_FRESH_RUN_TAG}"
            )
    else:
        atomic_json_dump(plan, plan_path)
    return expected_paths


def _hash_tensor(digest: Any, tensor: t.Tensor) -> None:
    value = tensor.detach().reshape(-1)
    digest.update(str(tensor.dtype).encode("ascii"))
    digest.update(canonical_json(list(tensor.shape)).encode("ascii"))
    chunk_items = max(1, (4 * 1024 * 1024) // max(1, tensor.element_size()))
    for start in range(0, value.numel(), chunk_items):
        chunk = value[start : start + chunk_items].to("cpu").contiguous()
        digest.update(chunk.view(t.uint8).numpy().tobytes())


def _module_state_hash(module: Any) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(module.state_dict().items()):
        digest.update(name.encode("utf-8"))
        _hash_tensor(digest, tensor)
    return digest.hexdigest()


def _forward_model(base_model: Any) -> tuple[Any, str]:
    native = getattr(base_model, "native_model", None)
    if native is not None:
        native_forward = getattr(native, "_base", None)
        if native_forward is None:
            raise AxBenchBaseError(
                "Loaded native extraction backend does not expose its actual "
                "forward module (_base) for parameter hashing"
            )
        return native_forward, "native"
    nnterp = getattr(base_model, "nnterp_model", None)
    if nnterp is not None:
        return getattr(nnterp, "model", nnterp), "nnterp"
    return base_model, "transformerlens"


def _runtime_identity(
    *,
    cfg: AxBenchBaseConfig,
    base_model: Any,
    base_sae: Any,
    extraction_tokenizer: Any,
) -> tuple[dict[str, Any], dict[str, str]]:
    forward_model, backend = _forward_model(base_model)
    if not hasattr(forward_model, "state_dict") or not hasattr(
        forward_model, "parameters"
    ):
        raise AxBenchBaseError(
            f"Resolved {backend} forward object does not expose module tensor state"
        )
    parameter_hashes = {
        "forward_model_sha256": _module_state_hash(forward_model),
        "sae_sha256": _module_state_hash(base_sae),
    }
    release, activation_type, sae_layer, width, l0_tier = (
        parse_neuronpedia_source(cfg.neuronpedia_source)
    )
    resolved_sae = resolve_sae(
        model=effective_extraction_model_id(cfg),
        release=release,
        activation_type=activation_type,
        layer=sae_layer,
        width=width,
        l0=l0_tier,
    )
    loaded_hook = _sae_hook_name(base_sae)
    if loaded_hook != resolved_sae.hook_name:
        raise AxBenchBaseError(
            "Loaded SAE hook does not match the configured resolver identity: "
            f"loaded={loaded_hook!r}, resolved={resolved_sae.hook_name!r}"
        )
    cuda_index = t.cuda.current_device() if t.cuda.is_available() else None
    def _package_version(name: str) -> str | None:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return None

    runtime = {
        "backend": backend,
        # Device ordinals legitimately differ by DDP rank.  The execution
        # identity is semantic only at the device-type / hardware level so it
        # remains identical for one-rank and multi-rank runs on equal GPUs.
        "device_type": next(base_sae.parameters()).device.type,
        "model_dtype": str(next(forward_model.parameters()).dtype),
        "sae_dtype": str(next(base_sae.parameters()).dtype),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": t.__version__,
        "cuda_runtime": t.version.cuda,
        "cudnn": t.backends.cudnn.version() if t.backends.cudnn.is_available() else None,
        "gpu_name": t.cuda.get_device_name(cuda_index) if cuda_index is not None else None,
        "gpu_capability": list(t.cuda.get_device_capability(cuda_index)) if cuda_index is not None else None,
        "pyarrow": pa.__version__,
        "safetensors": _package_version("safetensors"),
        "transformers": _package_version("transformers"),
        "transformer_lens": _package_version("transformer-lens"),
        "sae_lens": _package_version("sae-lens"),
        "nnterp": _package_version("nnterp"),
        "extraction_tokenizer_checksum": compute_tokenizer_checksum(extraction_tokenizer),
        "configured_extraction_model": cfg.feature_extraction_model_id or cfg.neuronpedia_model_id,
        "resolved_extraction_model": str(
            getattr(getattr(base_model, "cfg", None), "model_name", "")
        ),
        "resolved_sae_release": resolved_sae.release,
        "resolved_sae_id": resolved_sae.sae_id,
        "resolved_sae_hook": loaded_hook,
        "resolved_sae_l0_tier": l0_tier.value if l0_tier is not None else None,
        "resolved_sae_class": (
            f"{type(base_sae).__module__}.{type(base_sae).__qualname__}"
        ),
    }
    return runtime, parameter_hashes


def _run_l0_tier(cfg: AxBenchBaseConfig) -> L0Tier | None:
    _release, _activation, _layer, _width, l0 = parse_neuronpedia_source(
        cfg.neuronpedia_source
    )
    return l0 if isinstance(l0, L0Tier) else None


def _extract_candidate_shard(
    *,
    cfg: AxBenchBaseConfig,
    candidate_path: Path,
    output_dir: Path,
    base_model: Any,
    base_sae: Any,
) -> Path:
    parsed = parse_feature_dir(Path(cfg.feature_dir))
    if parsed.segment_mode is not SegmentMode.fixed_tokens:
        raise AxBenchBaseError(
            f"AxBench seed-pool extraction supports fixed_tokens feature directories only (segment_mode={parsed.segment_mode.value!r})"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    result = process_shard(
        str(candidate_path),
        base_model,
        base_sae,
        layer=parsed.layer,
        output_dir=str(output_dir),
        context_window=min(cfg.extraction_max_tokens, int(base_model.cfg.n_ctx)),
        pooling=parsed.pooling,
        gpu_batch_size=cfg.seed_extraction_gpu_batch_size,
        cpu_batch_size=cfg.seed_extraction_cpu_batch_size,
        nnterp_model=getattr(base_model, "nnterp_model", None),
        native_model=getattr(base_model, "native_model", None),
        segment_mode=SegmentMode.fixed_tokens,
        segment_size=parsed.segment_size,
        hook_name=_sae_hook_name(base_sae),
        activation_type=parsed.activation_type,
        dataset_phase=parsed.dataset_phase.value,
        context_independent_features=parsed.context_independent_features,
        ignore_bos_features=parsed.ignore_bos_features,
        run_l0_tier=_run_l0_tier(cfg),
    )
    return Path(result.output_path)


def _canonical_feature_digest(path: Path) -> tuple[str, int, list[int]]:
    features = load_features(path)
    digest = hashlib.sha256()
    # Historical shards already carried this header, but load_features ignored
    # it. Preserve their canonical sidecar digest when the reader exposes it.
    for key in sorted(features.keys() - {MetadataKey.N_SAE_FEATURES}):
        digest.update(key.encode("utf-8"))
        value = features[key]
        if isinstance(value, t.Tensor):
            _hash_tensor(digest, value)
        else:
            digest.update(canonical_json(value).encode("utf-8"))
    doc_counts = [int(value) for value in features["doc_token_counts"].tolist()]
    return digest.hexdigest(), int(features["n_docs"]), doc_counts


def _feature_sidecar_path(feature_path: Path) -> Path:
    return feature_path.parent / f"{feature_path.name}.seed-sidecar.json"


def _process_or_resume_feature_shard(
    *,
    cfg: AxBenchBaseConfig,
    candidate_path: Path,
    output_dir: Path,
    base_model: Any,
    base_sae: Any,
    extraction_fingerprint: str,
    runtime_identity: dict[str, Any],
) -> dict[str, Any]:
    expected_feature_path = feature_path(output_dir, candidate_path.stem)
    committed_feature_path = (
        expected_feature_path if expected_feature_path.is_file() else None
    )
    expected_sidecar_path = _feature_sidecar_path(expected_feature_path)
    candidate_sha = sha256_file(candidate_path)
    expected_docs = pq.read_metadata(candidate_path).num_rows
    if committed_feature_path is not None and not expected_sidecar_path.exists():
        logger.warning(
            "Stage 0b resume is discarding an uncommitted feature body without "
            "its commit sidecar before recomputing: {}",
            committed_feature_path,
        )
        committed_feature_path.unlink()
        committed_feature_path = None
    if committed_feature_path is not None or expected_sidecar_path.exists():
        if committed_feature_path is None or not expected_sidecar_path.is_file():
            raise AxBenchBaseError(
                f"Incomplete committed Stage 0b feature shard for {candidate_path}. "
                f"{_FRESH_RUN_TAG}"
            )
        sidecar = json.loads(expected_sidecar_path.read_text(encoding="utf-8"))
        digest, n_docs, doc_counts = _canonical_feature_digest(
            committed_feature_path
        )
        expected = {
            "schema_version": SEED_SCHEMA_VERSION,
            "candidate_parquet_sha256": candidate_sha,
            "extraction_fingerprint": extraction_fingerprint,
            "document_count": expected_docs,
            "canonical_output_sha256": digest,
            "runtime_identity": runtime_identity,
        }
        if any(sidecar.get(key) != value for key, value in expected.items()):
            raise AxBenchBaseError(
                f"Committed Stage 0b feature shard failed sidecar validation: "
                f"{committed_feature_path}. {_FRESH_RUN_TAG}"
            )
        if n_docs != expected_docs:
            raise AxBenchBaseError(
                "Stage 0b feature shard document count mismatch for "
                f"{committed_feature_path}"
            )
        return {
            "feature_path": str(committed_feature_path),
            **expected,
            "doc_token_counts": doc_counts,
        }

    committed_feature_path = _extract_candidate_shard(
        cfg=cfg,
        candidate_path=candidate_path,
        output_dir=output_dir,
        base_model=base_model,
        base_sae=base_sae,
    )
    if committed_feature_path != expected_feature_path:
        raise AxBenchBaseError(
            "Stage 0b process_shard wrote an unexpected feature path: "
            f"expected {expected_feature_path}, got {committed_feature_path}"
        )
    digest, n_docs, doc_counts = _canonical_feature_digest(
        committed_feature_path
    )
    if n_docs != expected_docs:
        raise AxBenchBaseError(
            f"Stage 0b extraction returned {n_docs} docs for {expected_docs} candidates"
        )
    sidecar = {
        "schema_version": SEED_SCHEMA_VERSION,
        "candidate_parquet_sha256": candidate_sha,
        "extraction_fingerprint": extraction_fingerprint,
        "document_count": expected_docs,
        "canonical_output_sha256": digest,
        "runtime_identity": runtime_identity,
    }
    atomic_json_dump(sidecar, expected_sidecar_path)
    return {
        "feature_path": str(committed_feature_path),
        **sidecar,
        "doc_token_counts": doc_counts,
    }


def _run_canary(
    *,
    cfg: AxBenchBaseConfig,
    candidates: list[SeedCandidate],
    root: Path,
    rank: int,
    world_size: int,
    base_model: Any,
    base_sae: Any,
) -> None:
    def _extract_twice() -> str:
        digests: list[str] = []
        for repetition in range(2):
            canary_root = root / f".canary_rank_{rank:05d}_{repetition}"
            shutil.rmtree(canary_root, ignore_errors=True)
            canary_root.mkdir(parents=True)
            canary_path = canary_root / "canary.parquet"
            atomic_write_parquet(
                pa.Table.from_pylist(
                    [candidate.parquet_row() for candidate in candidates[:2]],
                    schema=_CANDIDATE_SCHEMA,
                ),
                canary_path,
            )
            output_path = _extract_candidate_shard(
                cfg=cfg,
                candidate_path=canary_path,
                output_dir=canary_root,
                base_model=base_model,
                base_sae=base_sae,
            )
            digest, n_docs, doc_counts = _canonical_feature_digest(output_path)
            expected_counts = [candidate.extraction_token_count for candidate in candidates[:2]]
            if n_docs != len(expected_counts) or doc_counts != expected_counts:
                raise AxBenchBaseError(
                    "Stage 0b canary extraction truncated or reordered its input: "
                    f"n_docs={n_docs}, counts={doc_counts}, expected={expected_counts}"
                )
            digests.append(digest)
            shutil.rmtree(canary_root, ignore_errors=True)
        if digests[0] != digests[1]:
            raise AxBenchBaseError(
                "Stage 0b canary is not repeatable within rank "
                f"{rank}: {digests}"
            )
        return digests[0]

    canary_digest = _collective_call(
        "canary extraction",
        _extract_twice,
        rank=rank,
        world_size=world_size,
    )
    _require_rank_agreement(
        "canary output digest", canary_digest, world_size=world_size
    )


def _collect_feature_shards(
    *,
    cfg: AxBenchBaseConfig,
    candidate_paths: list[Path],
    root: Path,
    rank: int,
    world_size: int,
    base_model: Any,
    base_sae: Any,
    extraction_fingerprint: str,
    runtime_identity: dict[str, Any],
) -> None:
    output_dir = root / "features"
    for wave_start in range(0, len(candidate_paths), world_size):
        shard_index = wave_start + rank
        local_record: dict[str, Any]
        if shard_index >= len(candidate_paths):
            local_record = {"rank": rank, "status": "idle"}
        else:
            try:
                result = _process_or_resume_feature_shard(
                    cfg=cfg,
                    candidate_path=candidate_paths[shard_index],
                    output_dir=output_dir,
                    base_model=base_model,
                    base_sae=base_sae,
                    extraction_fingerprint=extraction_fingerprint,
                    runtime_identity=runtime_identity,
                )
                local_record = {
                    "rank": rank,
                    "status": "ok",
                    "shard_index": shard_index,
                    "canonical_output_sha256": result["canonical_output_sha256"],
                }
            except BaseException as exc:
                local_record = {
                    "rank": rank,
                    "status": "error",
                    "shard_index": shard_index,
                    "error": f"{type(exc).__name__}: {exc}",
                }
        records = [local_record]
        if world_size > 1:
            gathered: list[Any] = [None] * world_size
            dist.all_gather_object(gathered, local_record)
            records = [item for item in gathered if isinstance(item, dict)]
            if len(records) != world_size:
                raise AxBenchBaseError(
                    "Stage 0b collective wave returned malformed rank records"
                )
        errors = [record for record in records if record["status"] == "error"]
        if errors:
            raise AxBenchBaseError(
                "Stage 0b collective wave failed: " + canonical_json(errors)
            )


def _candidate_from_row(row: dict[str, Any]) -> SeedCandidate:
    """Load one committed candidate row without coercing malformed values."""

    def _integer(name: str, *, positive: bool = False) -> int:
        value = row.get(name)
        minimum = 1 if positive else 0
        if (
            not isinstance(value, int)
            or isinstance(value, bool)
            or value < minimum
        ):
            qualifier = "positive" if positive else "non-negative"
            raise AxBenchBaseError(
                f"Stage 0b candidate {name} must be a {qualifier} integer, "
                f"got {value!r}."
            )
        return value

    def _text(name: str, *, nonempty: bool) -> str:
        value = row.get(name)
        if not isinstance(value, str) or (nonempty and not value):
            qualifier = "non-empty " if nonempty else ""
            raise AxBenchBaseError(
                f"Stage 0b candidate {name} must be a {qualifier}string, "
                f"got {value!r}."
            )
        return value

    def _tokens(name: str) -> list[int]:
        value = row.get(name)
        if not isinstance(value, list) or not value:
            raise AxBenchBaseError(
                f"Stage 0b candidate {name} must be a non-empty token list, "
                f"got {value!r}."
            )
        if any(
            not isinstance(token, int) or isinstance(token, bool) or token < 0
            for token in value
        ):
            raise AxBenchBaseError(
                f"Stage 0b candidate {name} must contain only non-negative "
                f"integer token ids, got {value!r}."
            )
        return value

    candidate = SeedCandidate(
        candidate_index=_integer("candidate_index"),
        text=_text("text", nonempty=True),
        shard_idx=_integer("shard_idx"),
        doc_idx=_integer("doc_idx"),
        source_nanochat_token_count=_integer(
            "source_nanochat_token_count", positive=True
        ),
        prefix_source_token_start=_integer("prefix_source_token_start"),
        prefix_source_token_end=_integer("prefix_source_token_end", positive=True),
        prefix_content_nanochat_tokens=_tokens(
            "prefix_content_nanochat_tokens"
        ),
        # A non-empty raw token slice can legitimately decode to empty text;
        # exact source-token preservation takes precedence over rejecting it.
        continuation_text=_text("continuation_text", nonempty=False),
        continuation_content_nanochat_tokens=_tokens(
            "continuation_content_nanochat_tokens"
        ),
        extraction_token_count=_integer("extraction_token_count", positive=True),
    )
    if not (
        candidate.prefix_source_token_start
        < candidate.prefix_source_token_end
        <= candidate.source_nanochat_token_count
    ):
        raise AxBenchBaseError(
            "Stage 0b candidate prefix bounds are invalid: "
            f"start={candidate.prefix_source_token_start}, "
            f"end={candidate.prefix_source_token_end}, "
            f"source_count={candidate.source_nanochat_token_count}."
        )
    if len(candidate.prefix_content_nanochat_tokens) != (
        candidate.prefix_source_token_end - candidate.prefix_source_token_start
    ):
        raise AxBenchBaseError(
            "Stage 0b candidate prefix token length does not match its source "
            "bounds."
        )
    if (
        candidate.prefix_source_token_end
        + len(candidate.continuation_content_nanochat_tokens)
        != candidate.source_nanochat_token_count
    ):
        raise AxBenchBaseError(
            "Stage 0b candidate continuation token length does not reach the "
            "source-document end."
        )
    return candidate


def _merge_candidate_features(
    *,
    candidate_paths: list[Path],
    feature_root: Path,
    expectation: SeedPoolExpectation,
    nanochat_tokenizer: Any,
    extraction_fingerprint: str,
    runtime_identity: dict[str, Any],
) -> list[SeedRow]:
    rows_by_index: dict[int, SeedRow] = {}
    seen_provenance: set[tuple[int, int]] = set()
    for candidate_path in candidate_paths:
        candidates = [
            _candidate_from_row(row)
            for row in pq.read_table(candidate_path).to_pylist()
        ]
        for candidate in candidates:
            provenance = (candidate.shard_idx, candidate.doc_idx)
            if provenance in seen_provenance:
                raise AxBenchBaseError(
                    "Stage 0b candidate shards contain duplicate source "
                    f"provenance {provenance}"
                )
            seen_provenance.add(provenance)
        committed_feature_path = feature_path(feature_root, candidate_path.stem)
        if not committed_feature_path.is_file():
            raise AxBenchBaseError(
                f"Missing feature shard for {candidate_path}: {committed_feature_path}"
            )
        sidecar_path = _feature_sidecar_path(committed_feature_path)
        if not sidecar_path.is_file():
            raise AxBenchBaseError(f"Missing committed feature sidecar: {sidecar_path}")
        try:
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise AxBenchBaseError(
                f"Unreadable committed feature sidecar: {sidecar_path}"
            ) from exc
        features = load_features(committed_feature_path)
        doc_counts = [int(value) for value in features["doc_token_counts"].tolist()]
        expected_counts = [candidate.extraction_token_count for candidate in candidates]
        canonical_digest, n_docs, canonical_counts = _canonical_feature_digest(
            committed_feature_path
        )
        expected_sidecar = {
            "schema_version": SEED_SCHEMA_VERSION,
            "candidate_parquet_sha256": sha256_file(candidate_path),
            "extraction_fingerprint": extraction_fingerprint,
            "document_count": len(candidates),
            "canonical_output_sha256": canonical_digest,
            "runtime_identity": runtime_identity,
        }
        if any(sidecar.get(key) != value for key, value in expected_sidecar.items()):
            raise AxBenchBaseError(
                f"Committed Stage 0b sidecar no longer validates: {sidecar_path}"
            )
        if n_docs != len(candidates) or canonical_counts != doc_counts:
            raise AxBenchBaseError(
                f"Stage 0b feature document accounting changed for {candidate_path}"
            )
        if doc_counts != expected_counts:
            raise AxBenchBaseError(
                f"Stage 0b extraction truncated or reordered {candidate_path}: "
                f"expected token counts {expected_counts}, got {doc_counts}"
            )
        doc_ids = features["doc_ids"]
        observed_doc_ids = sorted({int(value) for value in doc_ids.tolist()})
        expected_doc_ids = list(range(len(candidates)))
        if observed_doc_ids != expected_doc_ids:
            raise AxBenchBaseError(
                "Stage 0b feature shard does not cover every candidate exactly "
                f"as a document: observed_doc_ids={observed_doc_ids}, "
                f"expected={expected_doc_ids}, path={committed_feature_path}"
            )
        all_segment_indices = features["segment_indices"]
        offsets = features["offsets"]
        all_ids = features["feature_ids"]
        all_mags = features["feature_mags"]
        for doc_index, candidate in enumerate(candidates):
            segment_indices = t.nonzero(doc_ids == doc_index, as_tuple=False).flatten().tolist()
            expected_segment_indices = list(range(len(segment_indices)))
            actual_segment_indices = [
                int(all_segment_indices[index].item()) for index in segment_indices
            ]
            if actual_segment_indices != expected_segment_indices:
                raise AxBenchBaseError(
                    "Stage 0b feature shard has missing, duplicate, or reordered "
                    f"segments for candidate {candidate.candidate_index}: "
                    f"got={actual_segment_indices}, "
                    f"expected={expected_segment_indices}"
                )
            feature_to_max: dict[int, float] = {}
            for segment_index in segment_indices:
                start = int(offsets[segment_index].item())
                end = int(offsets[segment_index + 1].item())
                for raw_feature_id, raw_magnitude in zip(
                    all_ids[start:end].tolist(),
                    all_mags[start:end].tolist(),
                    strict=True,
                ):
                    feature_id = int(raw_feature_id)
                    magnitude = float(raw_magnitude)
                    feature_to_max[feature_id] = max(
                        magnitude, feature_to_max.get(feature_id, magnitude)
                    )
            feature_ids = sorted(feature_to_max)
            row = SeedRow(
                shard_idx=candidate.shard_idx,
                doc_idx=candidate.doc_idx,
                source_nanochat_token_count=candidate.source_nanochat_token_count,
                prefix_source_token_start=candidate.prefix_source_token_start,
                prefix_source_token_end=candidate.prefix_source_token_end,
                prefix_text=candidate.text,
                prefix_content_nanochat_tokens=candidate.prefix_content_nanochat_tokens,
                sae_feature_ids=feature_ids,
                sae_feature_mags=[feature_to_max[feature_id] for feature_id in feature_ids],
                continuation_text=candidate.continuation_text,
                continuation_content_nanochat_tokens=(
                    candidate.continuation_content_nanochat_tokens
                ),
            )
            validate_seed_row(row, expectation=expectation, tokenizer=nanochat_tokenizer)
            if candidate.candidate_index in rows_by_index:
                raise AxBenchBaseError(
                    f"Duplicate Stage 0b candidate index {candidate.candidate_index}"
                )
            rows_by_index[candidate.candidate_index] = row
    expected_indices = list(range(len(rows_by_index)))
    if sorted(rows_by_index) != expected_indices:
        raise AxBenchBaseError(
            "Stage 0b candidate indices are missing or non-contiguous: "
            f"got {sorted(rows_by_index)}, expected {expected_indices}"
        )
    return [rows_by_index[index] for index in expected_indices]


def _existing_final_pair(
    *,
    cfg: AxBenchBaseConfig,
    tokenizer: Any,
) -> tuple[Path, Path] | None:
    assert cfg.run_tag is not None
    train_path = paths.AxBenchBasePaths.from_config(cfg).seeds_train()
    eval_path = paths.AxBenchBasePaths.from_config(cfg).seeds_eval()
    manifest_path = paths.AxBenchBasePaths.from_config(cfg).seed_pool_manifest()
    existing = (train_path.exists(), eval_path.exists(), manifest_path.exists())
    if not any(existing):
        return None
    if not all(existing):
        raise AxBenchBaseError(
            f"Torn Stage 0b output (train/eval/manifest existence={existing}). "
            f"{_FRESH_RUN_TAG}"
        )
    expectation = expectation_from_config(cfg, tokenizer)
    load_seed_pool(
        train_path,
        expected_split="train",
        expectation=expectation,
        tokenizer=tokenizer,
    )
    load_seed_pool(
        eval_path,
        expected_split="eval",
        expectation=expectation,
        tokenizer=tokenizer,
    )
    logger.info("Stage 0b committed seed artifacts already validate; reusing them")
    return train_path, eval_path


def _finalize_seed_pools(
    *,
    cfg: AxBenchBaseConfig,
    candidates: list[SeedCandidate],
    candidate_paths: list[Path],
    feature_root: Path,
    nanochat_tokenizer: Any,
    candidate_digest: str,
    skip_counts: dict[str, int],
    source_hashes: dict[str, str],
    extraction_fingerprint: str,
    extraction_settings: dict[str, Any],
    runtime_identity: dict[str, Any],
    parameter_hashes: dict[str, str],
    train_path: Path,
    eval_path: Path,
) -> None:
    """Merge every validated feature shard and atomically commit the pair."""

    assert cfg.run_tag is not None
    expectation = expectation_from_config(cfg, nanochat_tokenizer)
    rows = _merge_candidate_features(
        candidate_paths=candidate_paths,
        feature_root=feature_root,
        expectation=expectation,
        nanochat_tokenizer=nanochat_tokenizer,
        extraction_fingerprint=extraction_fingerprint,
        runtime_identity=runtime_identity,
    )
    if len(rows) != len(candidates):
        raise AxBenchBaseError(
            f"Stage 0b merge produced {len(rows)} rows for {len(candidates)} candidates"
        )
    provenance = [(row.shard_idx, row.doc_idx) for row in rows]
    if len(provenance) != len(set(provenance)):
        raise AxBenchBaseError(
            "Stage 0b accepted the same source document more than once; "
            "validation_shards and candidate provenance must be unique"
        )
    random.Random(cfg.seed_pool_seed).shuffle(rows)
    n_train = int(len(rows) * cfg.seed_pool_train_ratio)
    train_rows = rows[:n_train]
    eval_rows = rows[n_train:]
    if not train_rows or not eval_rows:
        raise AxBenchBaseError(
            "Stage 0b requires at least one row in both train and eval splits; "
            f"got train={len(train_rows)}, eval={len(eval_rows)}"
        )
    seed_fingerprint = compute_seed_fingerprint(
        candidate_digest=candidate_digest,
        config_fingerprint=seed_config_fingerprint(cfg),
        extraction_fingerprint=extraction_fingerprint,
        nanochat_tokenizer_checksum=expectation.nanochat_tokenizer_checksum,
        source_parquet_hashes=source_hashes,
    )
    common_metadata = {
        "schema_version": str(SEED_SCHEMA_VERSION),
        "seed_fingerprint": seed_fingerprint,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "nanochat_tokenizer_checksum": expectation.nanochat_tokenizer_checksum,
        "extraction_tokenizer_checksum": runtime_identity[
            "extraction_tokenizer_checksum"
        ],
        "config_fingerprint": expectation.config_fingerprint,
        "seed_config_json": canonical_json(seed_config_payload(cfg)),
        "source_parquet_hashes_json": canonical_json(source_hashes),
        "parameter_hashes_json": canonical_json(parameter_hashes),
        "runtime_identity_json": canonical_json(runtime_identity),
        "extraction_settings_json": canonical_json(extraction_settings),
    }
    save_seed_pool(
        train_path,
        train_rows,
        metadata={**common_metadata, "split": "train"},
    )
    save_seed_pool(
        eval_path,
        eval_rows,
        metadata={**common_metadata, "split": "eval"},
    )
    manifest = {
        "schema_version": SEED_SCHEMA_VERSION,
        "seed_fingerprint": seed_fingerprint,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "candidate_digest": candidate_digest,
        "candidate_count": len(candidates),
        "skip_counts": skip_counts,
        "source_parquet_hashes": source_hashes,
        "config_fingerprint": expectation.config_fingerprint,
        "extraction_fingerprint": extraction_fingerprint,
        "nanochat_tokenizer_checksum": expectation.nanochat_tokenizer_checksum,
        "extraction_tokenizer_checksum": runtime_identity[
            "extraction_tokenizer_checksum"
        ],
        "parameter_hashes": parameter_hashes,
        "runtime_identity": runtime_identity,
        "extraction_settings": extraction_settings,
        "files": {
            train_path.name: {
                "sha256": sha256_file(train_path),
                "row_count": len(train_rows),
                "split": "train",
            },
            eval_path.name: {
                "sha256": sha256_file(eval_path),
                "row_count": len(eval_rows),
                "split": "eval",
            },
        },
    }
    atomic_json_dump(manifest, paths.AxBenchBasePaths.from_config(cfg).seed_pool_manifest())
    logger.info(
        "Stage 0b committed schema-v2 seed pools: total={}, train={}, eval={}, skips={}",
        len(rows),
        len(train_rows),
        len(eval_rows),
        skip_counts,
    )


def prepare_seed_pools(
    cfg: AxBenchBaseConfig,
    *,
    base_model: Any,
    base_sae: Any,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[Path, Path]:
    """Build schema-v2 seed pools with production prefix feature extraction."""

    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before Stage 0b")
    if world_size <= 0 or rank < 0 or rank >= world_size:
        raise AxBenchBaseError(
            f"Invalid Stage 0b DDP coordinates rank={rank}, world_size={world_size}"
        )
    if world_size > 1 and (
        not dist.is_available() or not dist.is_initialized()
    ):
        raise AxBenchBaseError(
            f"Stage 0b world_size={world_size} requires an initialized process group"
        )

    # Stage 0b persists fixed-window extraction artifacts. Reject a
    # turn-delimited source before inspecting or creating any run output; the
    # extraction guard in steering/extract_features.py remains as defense in
    # depth. Path resolution can observe rank-local filesystem state, so the
    # rejection propagates collectively before any peer enters the resume
    # collectives (``_collective_call`` skips the gather when world_size == 1).
    _collective_call(
        "segment-mode preflight",
        lambda: require_fixed_feature_dir_segment_mode(
            cfg.feature_dir,
            surface="AxBench-Base seed-pool extraction",
        ),
        rank=rank,
        world_size=world_size,
    )
    nanochat_tokenizer = get_tokenizer()
    existing = _collective_call(
        "committed-artifact validation",
        lambda: _existing_final_pair(cfg=cfg, tokenizer=nanochat_tokenizer),
        rank=rank,
        world_size=world_size,
    )
    _require_rank_agreement(
        "committed-artifact resume decision",
        existing is not None,
        world_size=world_size,
    )
    if existing is not None:
        return existing
    extraction_tokenizer = base_model.tokenizer
    extraction_limit = min(cfg.extraction_max_tokens, int(base_model.cfg.n_ctx))
    candidates, skip_counts, source_hashes = _collective_call(
        "candidate streaming",
        lambda: _stream_candidates(
            cfg=cfg,
            nanochat_tokenizer=nanochat_tokenizer,
            extraction_tokenizer=extraction_tokenizer,
            extraction_limit=extraction_limit,
        ),
        rank=rank,
        world_size=world_size,
    )
    _require_rank_agreement(
        "candidate count", len(candidates), world_size=world_size
    )
    if not candidates:
        raise AxBenchBaseError(
            "Stage 0b produced no candidates after raw-token and extraction "
            f"preflight; skip_counts={skip_counts}"
        )
    digest = _candidate_digest(candidates, skip_counts, source_hashes)
    _require_rank_agreement("candidate digest", digest, world_size=world_size)
    _require_rank_agreement("candidate skip counts", skip_counts, world_size=world_size)
    _require_rank_agreement("source parquet hashes", source_hashes, world_size=world_size)

    runtime_identity, parameter_hashes = _collective_call(
        "model/SAE identity hashing",
        lambda: _runtime_identity(
            cfg=cfg,
            base_model=base_model,
            base_sae=base_sae,
            extraction_tokenizer=extraction_tokenizer,
        ),
        rank=rank,
        world_size=world_size,
    )
    _require_rank_agreement("model and SAE parameter hashes", parameter_hashes, world_size=world_size)
    _require_rank_agreement("runtime identity", runtime_identity, world_size=world_size)
    resolved_l0_tier = _run_l0_tier(cfg)
    extraction_settings = _collective_call(
        "extraction-setting resolution",
        lambda: {
            "context_window": extraction_limit,
            "layer": parse_feature_dir(Path(cfg.feature_dir)).layer,
            "hook_name": _sae_hook_name(base_sae),
            "l0_tier": (
                resolved_l0_tier.value if resolved_l0_tier is not None else None
            ),
            "feature_dir": seed_config_payload(cfg)["feature_directory"],
            "candidate_shard_size": cfg.seed_extraction_candidate_shard_size,
            "gpu_batch_size": cfg.seed_extraction_gpu_batch_size,
            "cpu_batch_size": cfg.seed_extraction_cpu_batch_size,
        },
        rank=rank,
        world_size=world_size,
    )
    _require_rank_agreement("extraction settings", extraction_settings, world_size=world_size)
    extraction_fingerprint = compute_extraction_fingerprint(
        extraction_settings=extraction_settings,
        runtime_identity=runtime_identity,
        parameter_hashes=parameter_hashes,
    )
    _require_rank_agreement("extraction fingerprint", extraction_fingerprint, world_size=world_size)

    root = paths.AxBenchBasePaths.from_config(cfg).seed_extraction_root()
    shards = _candidate_shards(candidates, shard_size=cfg.seed_extraction_candidate_shard_size)
    candidate_paths: list[Path] = []
    rank_zero_paths = _collective_call(
        "candidate-shard commit",
        lambda: (
            _write_or_validate_candidate_shards(
                root=root,
                candidate_shards=shards,
                candidate_digest=digest,
                skip_counts=skip_counts,
                source_hashes=source_hashes,
            )
            if rank == 0
            else None
        ),
        rank=rank,
        world_size=world_size,
    )
    if rank == 0:
        candidate_paths = rank_zero_paths
    if rank != 0:
        candidate_paths = [
            root / "candidates" / f"candidate_{index:05d}.parquet"
            for index in range(len(shards))
        ]
    candidate_shards_complete = all(path.is_file() for path in candidate_paths)
    _require_rank_agreement(
        "candidate-shard visibility",
        candidate_shards_complete,
        world_size=world_size,
    )
    if not candidate_shards_complete:
        raise AxBenchBaseError("Stage 0b candidate shards are incomplete after rank rendezvous")

    _run_canary(
        cfg=cfg,
        candidates=candidates,
        root=root,
        rank=rank,
        world_size=world_size,
        base_model=base_model,
        base_sae=base_sae,
    )
    _collect_feature_shards(
        cfg=cfg,
        candidate_paths=candidate_paths,
        root=root,
        rank=rank,
        world_size=world_size,
        base_model=base_model,
        base_sae=base_sae,
        extraction_fingerprint=extraction_fingerprint,
        runtime_identity=runtime_identity,
    )
    _barrier(world_size)

    train_path = paths.AxBenchBasePaths.from_config(cfg).seeds_train()
    eval_path = paths.AxBenchBasePaths.from_config(cfg).seeds_eval()
    _collective_call(
        "final merge and artifact commit",
        lambda: (
            _finalize_seed_pools(
                cfg=cfg,
                candidates=candidates,
                candidate_paths=candidate_paths,
                feature_root=root / "features",
                nanochat_tokenizer=nanochat_tokenizer,
                candidate_digest=digest,
                skip_counts=skip_counts,
                source_hashes=source_hashes,
                extraction_fingerprint=extraction_fingerprint,
                extraction_settings=extraction_settings,
                runtime_identity=runtime_identity,
                parameter_hashes=parameter_hashes,
                train_path=train_path,
                eval_path=eval_path,
            )
            if rank == 0
            else None
        ),
        rank=rank,
        world_size=world_size,
    )
    return train_path, eval_path


__all__ = [
    "_CANDIDATE_SCHEMA",
    "SeedCandidate",
    "_build_structural_candidate",
    "load_seed_pool",
    "prepare_seed_pools",
]
