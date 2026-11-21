"""Run corrected real-prompt SelfConditioning on an AxBench input frame."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import subprocess
from collections.abc import Mapping, Sequence
from copy import deepcopy
from pathlib import Path
from statistics import median
from typing import Any, cast

import pandas as pd
import pyarrow as pa
import torch as t
import torch.distributed as dist
from loguru import logger
from omegaconf import DictConfig, OmegaConf

from nanochat.atomic_io import atomic_write_bytes, atomic_write_parquet, atomic_write_text
from nanochat.common import get_base_dir
from nanochat.tokenizer import get_tokenizer
from scripts.eval.axbench_upstream.merge_methods import (
    INPUT_COLUMNS,
    MAX_BLANK_GENERATION_FRACTION,
    METHOD_FACTORS,
    MIN_MEDIAN_GENERATION_CHARS,
    NL_PROJECT_METHODS,
    SUPPORTED_METHOD_FACTORS,
    method_columns,
    resolve_concept_ids,
    validate_steering_frame,
)
from scripts.eval.axbench_upstream.run_baselines import (
    AXBENCH_UPSTREAM_COMMIT,
    _git_worktree_state,
)
from steering._common.ddp import rank_synchronized_stage
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
    SelfCondScorePolicy,
)
from steering.eval.axbench_base.profiler import profile_region
from steering.eval.axbench_upstream.cell_store import CellResultStore, CellStoreError
from steering.eval.axbench_upstream.data import select_training_texts
from steering.eval.axbench_upstream.methods import (
    SELF_CONDITIONING_FACTORS,
    SELF_CONDITIONING_MASKED_POLICY,
    SELF_CONDITIONING_POLICY,
    TEMPERATURE,
    TOP_K,
    MethodSetup,
    PreparedPromptContext,
    SelfConditioningGenerationRequest,
    assert_masked_diffmean_feasible,
    assert_prepared_prompt_matches,
    build_self_conditioning_masked_policy,
    fit_concept_dense_diffmeans,
    generate_self_conditioning,
    generate_self_conditioning_batch,
    load_extraction_stack,
    load_method_setup,
    load_target_model,
    prepare_prompt_context,
    resolve_raw_feature_selection,
    resolve_raw_negative_selection,
    self_conditioning_masked_pool_limit,
    self_conditioning_weight,
)
from steering.eval.axbench_upstream.nl_target_scores import (
    NL_RETRIEVAL_ALGORITHM,
    NL_STEERING_POLICY,
    build_retrieval_target_scores,
    retrieval_feature_identity,
)
from steering.eval.axbench_upstream.nl_target_source import (
    NLTargetSourceReceipt,
    load_nl_target_source_receipt,
    retrieval_targets_sha256,
)
from steering.eval.axbench_upstream.profiling import cell_trace_requested, trace_cell
from steering.eval.axbench_upstream.project_input import load_project_input_artifact
from steering.eval.axbench_upstream.prompt_prep import build_rendered_prompt_plan
from steering.eval.axbench_upstream.seeds import MAX_SIGNED_63_BIT as _MAX_SIGNED_63_BIT
from steering.eval.axbench_upstream.seeds import (
    SEED_MODULUS,
    derive_generation_seed,
    derive_selection_seed,
)
from steering.eval.axbench_upstream.work_grid import (
    PROJECT_METHODS,
    CellKey,
    build_project_cell_grid,
    owned_project_cells,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.target_conditioning import validate_feature_dict
from steering.immutable_io import (
    ImmutableArtifactError,
    install_bytes_once,
    stream_hash_file_descriptor_bound,
)
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    load_safetensors_file_snapshot,
)
from steering.segmentation import SegmentMode
from steering.stats_io import content_sha256

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_COLUMNS = (
    "SelfConditioning_steered_generation",
    "SelfConditioning_strength",
    "SelfConditioningMasked_steered_generation",
    "SelfConditioningMasked_strength",
)
NL_PROJECT_COLUMNS = method_columns(NL_PROJECT_METHODS)
_CONFIG_KEYS = frozenset(
    {
        "main_path",
        "prepared_dir",
        "output_path",
        "manifest_path",
        "target_meta_path",
        "feature_statistics_path",
        "max_concepts",
        "device",
        "seed",
        "max_new_tokens",
        "extraction_batch_size",
        "self_conditioning_policy",
        "self_conditioning_masked_policy",
        "force",
        "pretrain_meta_path",
        "require_exact_statistics_identity",
    }
)
_OPTIONAL_CONFIG_KEYS = frozenset(
    {"project_input_path", "project_input_manifest_path", "nl_target_source_manifest_path", "concept_ids"}
)
_PROJECT_FIT_FORMAT = "axbench-upstream-project-dense-diffmean-v1"
_PROJECT_FIT_IDENTITY_NAME = "identity.json"
_PROJECT_FIT_PATTERN = "dense_diffmean_{concept_id:05d}.safetensors"


def _sha256(path: Path) -> str:
    return stream_hash_file_descriptor_bound(path).sha256


def _file_identity(path: Path) -> dict[str, int | str]:
    path = Path(path)
    digest = stream_hash_file_descriptor_bound(path)
    return {
        "name": path.name,
        "byte_length": digest.byte_length,
        "sha256": digest.sha256,
    }


def _tokenizer_artifact_identity() -> dict[str, int | str]:
    """Identify the exact RustBPE artifact used for prompt rendering and decode."""

    return _file_identity(Path(get_base_dir()) / "tokenizer" / "tokenizer.pkl")


def _repository_source_sha256() -> str:
    """Bind resume state to dirty content, not only dirty path names."""

    digest = hashlib.sha256()
    tracked_diff = subprocess.check_output(
        ["git", "diff", "--binary", "--no-ext-diff", "HEAD", "--"],
        cwd=REPO_ROOT,
    )
    digest.update(b"tracked-diff\0")
    digest.update(len(tracked_diff).to_bytes(8, "big"))
    digest.update(tracked_diff)
    untracked_output = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard", "-z"],
        cwd=REPO_ROOT,
    )
    untracked_paths = sorted(path for path in untracked_output.split(b"\0") if path)
    for encoded_relative in untracked_paths:
        relative = Path(os.fsdecode(encoded_relative))
        path = REPO_ROOT / relative
        digest.update(b"untracked\0")
        digest.update(len(encoded_relative).to_bytes(8, "big"))
        digest.update(encoded_relative)
        if path.is_symlink():
            target = os.fsencode(path.readlink())
            digest.update(b"symlink\0")
            digest.update(len(target).to_bytes(8, "big"))
            digest.update(target)
            continue
        if not path.is_file():
            raise ValueError(f"untracked repository entry is not a regular file: {path}")
        digest.update(b"file\0")
        file_stat = path.stat()
        digest.update(file_stat.st_mode.to_bytes(8, "big"))
        digest.update(file_stat.st_size.to_bytes(8, "big"))
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
    return digest.hexdigest()


def _strict_json_bytes(payload: Any, *, label: str) -> bytes:
    try:
        return (
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            ).encode("utf-8")
            + b"\n"
        )
    except (TypeError, ValueError) as exc:
        raise CellStoreError(f"{label} must be strict JSON: {exc}") from exc


def _project_fit_path(fit_dir: Path, concept_id: int) -> Path:
    return fit_dir / _PROJECT_FIT_PATTERN.format(concept_id=concept_id)


def _prepare_project_fit_dir(
    *,
    fit_dir: Path,
    semantic_fingerprint: str,
    concept_ids: tuple[int, ...],
) -> None:
    """Bind a partial project-fit directory before any artifact is reused."""

    fit_dir.mkdir(parents=True, exist_ok=True)
    identity = {
        "schema_version": 1,
        "format": _PROJECT_FIT_FORMAT,
        "semantic_fingerprint": semantic_fingerprint,
        "concept_ids": list(concept_ids),
    }
    try:
        install_bytes_once(
            _strict_json_bytes(identity, label="project fit identity"),
            fit_dir / _PROJECT_FIT_IDENTITY_NAME,
            surface="AxBench-Upstream project fit identity",
        )
    except ImmutableArtifactError as exc:
        raise CellStoreError(str(exc)) from exc

    allowed = {
        _PROJECT_FIT_IDENTITY_NAME,
        *(_project_fit_path(fit_dir, concept_id).name for concept_id in concept_ids),
    }
    for path in fit_dir.iterdir():
        if path.is_symlink():
            raise CellStoreError(f"project fit directory contains a symlink: {path}")
        if not path.is_file() or path.name not in allowed:
            raise CellStoreError(f"project fit directory contains an unknown entry: {path}")


def _save_project_fit(
    path: Path,
    scores: t.Tensor,
    *,
    concept_id: int,
    semantic_fingerprint: str,
) -> None:
    value = scores.detach().to(dtype=t.float32, device="cpu").contiguous()
    if value.ndim != 1 or value.numel() == 0 or not bool(t.isfinite(value).all()):
        raise CellStoreError(
            f"project dense DiffMean concept_id={concept_id} must be a finite nonempty vector"
        )
    atomic_write_bytes(
        deterministic_safetensors_bytes(
            {"dense_diffmean": value},
            metadata={
                "format": _PROJECT_FIT_FORMAT,
                "concept_id": str(concept_id),
                "semantic_fingerprint": semantic_fingerprint,
            },
        ),
        path,
    )


def _load_project_fit(
    path: Path,
    *,
    concept_id: int,
    semantic_fingerprint: str,
    expected_width: int,
) -> t.Tensor:
    try:
        tensors, loaded_metadata = load_safetensors_file_snapshot(path)
    except Exception as exc:
        raise CellStoreError(f"could not read project fit artifact {path}: {exc}") from exc
    metadata = loaded_metadata or {}
    expected_metadata = {
        "format": _PROJECT_FIT_FORMAT,
        "concept_id": str(concept_id),
        "semantic_fingerprint": semantic_fingerprint,
    }
    if metadata != expected_metadata:
        raise CellStoreError(f"project fit artifact metadata differs from its run identity: {path}")
    if set(tensors) != {"dense_diffmean"}:
        raise CellStoreError(f"project fit artifact must contain only 'dense_diffmean': {path}")
    value = tensors["dense_diffmean"]
    if (
        value.dtype is not t.float32
        or value.device.type != "cpu"
        or value.shape != (expected_width,)
        or not value.is_contiguous()
        or not bool(t.isfinite(value).all())
    ):
        raise CellStoreError(
            f"project fit artifact has an invalid tensor contract: {path}: "
            f"dtype={value.dtype}, shape={tuple(value.shape)}, device={value.device}, "
            f"contiguous={value.is_contiguous()}"
        )
    return value


def _install_project_fit_receipt(
    *,
    resume_root: Path,
    fit_dir: Path,
    semantic_fingerprint: str,
    concept_ids: tuple[int, ...],
) -> None:
    files = [
        {
            "concept_id": concept_id,
            "path": _project_fit_path(fit_dir, concept_id).name,
            "sha256": _sha256(_project_fit_path(fit_dir, concept_id)),
        }
        for concept_id in concept_ids
    ]
    receipt = {
        "schema_version": 1,
        "format": _PROJECT_FIT_FORMAT,
        "semantic_fingerprint": semantic_fingerprint,
        "files": files,
    }
    try:
        install_bytes_once(
            _strict_json_bytes(receipt, label="project fit receipt"),
            resume_root / "fit_artifacts.json",
            surface="AxBench-Upstream project fit-artifact receipt",
        )
    except ImmutableArtifactError as exc:
        raise CellStoreError(str(exc)) from exc


def _json_object(path: Path) -> dict[str, Any]:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value!r}")

    def reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    try:
        loaded = json.loads(
            path.read_text(encoding="utf-8"),
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"expected a readable strict JSON object at {path}: {exc}") from exc
    if not isinstance(loaded, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return loaded


def _reject_forced_threshold_era_output(manifest_path: Path) -> None:
    """Keep ``force`` from deleting an identifiable threshold-era artifact."""

    try:
        manifest = _json_object(manifest_path)
    except ValueError:
        # Preserve the established force behavior for an unrelated malformed
        # output. Threshold-era project runs are identifiable strict manifests.
        return
    methods = manifest.get("methods")
    masked = methods.get("SelfConditioningMasked") if isinstance(methods, dict) else None
    config = manifest.get("configuration")
    is_threshold_era = (
        manifest.get("schema_version") == 2
        or (isinstance(masked, dict) and "masked_threshold_scale" in masked)
        or (isinstance(config, dict) and "masked_threshold_scale" in config)
    )
    if is_threshold_era:
        raise ValueError(
            "force=true refuses to replace a threshold-era SelfConditioningMasked "
            "project artifact; manually clear the exact output after inspection or "
            "select a new output root for a rank-era run"
        )


def _target_checkpoint_identity(target_meta_path: Path) -> dict[str, int | str]:
    meta = _json_object(target_meta_path)
    step = meta.get("step")
    if type(step) is not int or step < 0:
        raise ValueError(
            f"target metadata step must be a nonnegative int excluding bool: {target_meta_path}"
        )
    return _file_identity(target_meta_path.parent / f"model_{step:06d}.pt")


def _git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def _validate_prepared_manifest(prepared_manifest: dict[str, Any]) -> str:
    try:
        json.dumps(prepared_manifest, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"prepared manifest is not strict JSON: {exc}") from exc
    if type(prepared_manifest.get("schema_version")) is not int or (
        prepared_manifest["schema_version"] != 1
    ):
        raise ValueError("prepared manifest schema_version must equal 1")
    benchmark = prepared_manifest.get("benchmark")
    dataset_revision = benchmark.get("revision") if isinstance(benchmark, dict) else None
    if not isinstance(dataset_revision, str) or not dataset_revision.strip():
        raise ValueError("prepared manifest has no benchmark revision")
    if prepared_manifest.get("axbench_upstream_commit") != AXBENCH_UPSTREAM_COMMIT:
        raise ValueError("prepared manifest AxBench revision does not match the vendored snapshot")
    return dataset_revision


def _rejected_output_path(output_path: Path) -> Path:
    return output_path.with_suffix(".rejected.parquet")


def _assert_disjoint_paths(
    read_paths: Mapping[str, Path],
    write_paths: Mapping[str, Path],
) -> None:
    for write_name, write_path in write_paths.items():
        for read_name, read_path in read_paths.items():
            if write_path == read_path:
                raise ValueError(f"{write_name} must not alias {read_name}: {write_path}")
        for other_name, other_path in write_paths.items():
            if write_name != other_name and write_path == other_path:
                raise ValueError(f"{write_name} must not alias {other_name}: {write_path}")


def _validate_artifact_paths(
    *,
    input_path: Path,
    input_manifest_path: Path | None,
    prepared_dir: Path,
    output_path: Path,
    manifest_path: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    pretrain_meta_path: Path | None = None,
    nl_target_source_manifest_path: Path | None = None,
) -> Path:
    rejected_path = _rejected_output_path(output_path)
    read_paths = {
        "input_path": input_path,
        "prepared_dir": prepared_dir,
        "prepare_manifest": prepared_dir / "prepare_manifest.json",
        "prepared_metadata": prepared_dir / "generate" / "metadata.jsonl",
        "prepared_training_data": prepared_dir / "generate" / "train_data.parquet",
        "target_meta_path": target_meta_path,
        "feature_statistics_path": feature_statistics_path,
    }
    if input_manifest_path is not None:
        read_paths["input_manifest_path"] = input_manifest_path
    if pretrain_meta_path is not None:
        read_paths["pretrain_meta_path"] = pretrain_meta_path
    if nl_target_source_manifest_path is not None:
        read_paths["nl_target_source_manifest_path"] = nl_target_source_manifest_path
    write_paths = {
        "output_path": output_path,
        "manifest_path": manifest_path,
        "rejected output path": rejected_path,
    }
    _assert_disjoint_paths(read_paths, write_paths)
    return rejected_path


def _validate_nl_artifact_paths(
    receipt: NLTargetSourceReceipt,
    *,
    output_path: Path,
    manifest_path: Path,
    rejected_path: Path,
    resume_root: Path,
    fit_dir: Path,
) -> None:
    """Keep immutable NL inputs outside every project output and resume root."""

    inputs = (receipt.path, receipt.description_embeddings.path, receipt.concept_embeddings.path)
    outputs = (output_path, manifest_path, rejected_path, resume_root, fit_dir)
    for input_path in inputs:
        for output in outputs:
            same_existing_file = (
                input_path.exists() and output.exists() and input_path.samefile(output)
            )
            if (
                input_path == output
                or input_path.is_relative_to(output)
                or output.is_relative_to(input_path)
                or same_existing_file
            ):
                raise ValueError(
                    "NL target inputs, project outputs, and resume stores must be path-disjoint: "
                    f"{input_path} conflicts with {output}"
                )


def _resolve_input_source(
    main_path: Path | None,
    project_input_path: Path | None,
    project_input_manifest_path: Path | None,
) -> tuple[Path, str, Path | None]:
    """Resolve exactly one legacy-main or model-free project input source."""

    if (main_path is None) == (project_input_path is None):
        raise ValueError("exactly one of main_path or project_input_path must be configured")
    if project_input_path is not None:
        if project_input_manifest_path is None:
            raise ValueError(
                "project_input_manifest_path is required with project_input_path"
            )
        return (
            Path(project_input_path).expanduser().resolve(),
            "project_input",
            Path(project_input_manifest_path).expanduser().resolve(),
        )
    if project_input_manifest_path is not None:
        raise ValueError("project_input_manifest_path requires project_input_path")
    assert main_path is not None
    return Path(main_path).expanduser().resolve(), "main", None


def _validate_common_scalar_config(
    *,
    max_concepts: int,
    device: str,
    seed: int,
    max_new_tokens: int,
) -> None:
    # ``force`` is deliberately NOT validated here: both drivers check it
    # LAST, preserving each driver's original multi-defect error precedence.
    if type(max_concepts) is not int or max_concepts <= 0:
        raise ValueError("max_concepts must be a positive integer")
    if not isinstance(device, str) or not device:
        raise ValueError("device must be a nonempty string")
    if type(seed) is not int or not 0 <= seed <= _MAX_SIGNED_63_BIT:
        # Enforce the seed-derivation domain (run_methods._seed_identity_integer)
        # UP FRONT: a seed accepted here but rejected at per-cell derivation
        # would fail only after fits and LoRA training wrote durable artifacts.
        raise ValueError("seed must be a nonnegative integer in [0, 2^63 - 1]")
    if type(max_new_tokens) is not int or max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be a positive integer")


def _validate_scalar_config(
    *,
    max_concepts: int,
    device: str,
    seed: int,
    max_new_tokens: int,
    extraction_batch_size: int,
    self_conditioning_policy: str,
    self_conditioning_masked_policy: str,
    force: bool,
    require_exact_statistics_identity: bool,
) -> None:
    _validate_common_scalar_config(
        max_concepts=max_concepts,
        device=device,
        seed=seed,
        max_new_tokens=max_new_tokens,
    )
    if type(extraction_batch_size) is not int or extraction_batch_size <= 0:
        raise ValueError("extraction_batch_size must be a positive integer")
    if self_conditioning_policy != SELF_CONDITIONING_POLICY:
        raise ValueError(
            f"self_conditioning_policy must explicitly equal {SELF_CONDITIONING_POLICY!r}"
        )
    if self_conditioning_masked_policy != SELF_CONDITIONING_MASKED_POLICY:
        raise ValueError(
            "self_conditioning_masked_policy must explicitly equal "
            f"{SELF_CONDITIONING_MASKED_POLICY!r}"
        )
    if type(force) is not bool:
        raise ValueError("force must be a boolean")
    if type(require_exact_statistics_identity) is not bool:
        raise ValueError("require_exact_statistics_identity must be a boolean")


def _manifest_header() -> dict[str, Any]:
    return {
        "repository_git_commit": _git_commit(),
        "repository_git_worktree": _git_worktree_state(),
        "vendored_axbench_upstream_commit": AXBENCH_UPSTREAM_COMMIT,
        "package_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
    }


def _distributed_coordinates() -> tuple[int, int]:
    if not dist.is_initialized():
        return 0, 1
    return dist.get_rank(), dist.get_world_size()


def _runtime_identity(device: str) -> dict[str, Any]:
    identity: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": str(t.__version__),
        "cuda_build": str(t.version.cuda),
        "cudnn": t.backends.cudnn.version(),
        "device_type": device.split(":", 1)[0],
        "float32_matmul_precision": t.get_float32_matmul_precision(),
        "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
        "deterministic_warn_only": t.is_deterministic_algorithms_warn_only_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "torch_allow_tf32_cublas_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
    }
    if device.startswith("cuda") and t.cuda.is_available():
        try:
            configured_index = t.device(device).index
            device_index = t.cuda.current_device() if configured_index is None else configured_index
            identity["gpu_name"] = t.cuda.get_device_name(device_index)
            identity["gpu_capability"] = list(t.cuda.get_device_capability(device_index))
        except (AssertionError, RuntimeError):
            # Unit tests replace the CUDA execution surface with CPU fakes.
            # Production still fails when the driver later loads its CUDA model,
            # so this sentinel cannot make an unavailable CUDA device usable.
            identity["gpu_name"] = "unavailable-test-double"
            identity["gpu_capability"] = None
        identity.update(
            {
                "cudnn_enabled": t.backends.cudnn.enabled,
                "cudnn_benchmark": t.backends.cudnn.benchmark,
                "cudnn_deterministic": t.backends.cudnn.deterministic,
                "cudnn_allow_tf32": t.backends.cudnn.allow_tf32,
                "flash_sdp_enabled": t.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": t.backends.cuda.mem_efficient_sdp_enabled(),
                "math_sdp_enabled": t.backends.cuda.math_sdp_enabled(),
                "cudnn_sdp_enabled": t.backends.cuda.cudnn_sdp_enabled(),
                "fp16_bf16_reduction_math_sdp_allowed": (
                    t.backends.cuda.fp16_bf16_reduction_math_sdp_allowed()
                ),
                "preferred_blas_library": str(t.backends.cuda.preferred_blas_library()),
                "matmul_allow_tf32": t.backends.cuda.matmul.allow_tf32,
                "matmul_allow_fp16_reduced_precision_reduction": (
                    t.backends.cuda.matmul.allow_fp16_reduced_precision_reduction
                ),
                "matmul_allow_bf16_reduced_precision_reduction": (
                    t.backends.cuda.matmul.allow_bf16_reduced_precision_reduction
                ),
                "matmul_allow_fp16_accumulation": (t.backends.cuda.matmul.allow_fp16_accumulation),
            }
        )
    return identity


def _project_resume_identity(
    *,
    input_source: Mapping[str, str],
    prepared_dir: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    pretrain_meta_path: Path | None,
    tokenizer_identity: Mapping[str, int | str],
    max_concepts: int,
    device: str,
    seed: int,
    max_new_tokens: int,
    extraction_batch_size: int,
    self_conditioning_policy: str,
    self_conditioning_masked_policy: str,
    require_exact_statistics_identity: bool,
) -> dict[str, Any]:
    if input_source["kind"] == "main":
        source_identity = {"main_sha256": input_source["sha256"]}
    else:
        source_identity = {
            "source_kind": input_source["kind"],
            "source_sha256": input_source["sha256"],
            "source_manifest_sha256": input_source["manifest_sha256"],
        }
    return {
        "driver": "project",
        "schema_version": 1 if input_source["kind"] == "main" else 2,
        "repository_git_commit": _git_commit(),
        "repository_git_worktree": _git_worktree_state(),
        "repository_source_sha256": _repository_source_sha256(),
        "package_lock_sha256": _sha256(REPO_ROOT / "uv.lock"),
        "inputs": {
            **source_identity,
            "prepared_manifest_sha256": _sha256(prepared_dir / "prepare_manifest.json"),
            "prepared_metadata": _file_identity(prepared_dir / "generate" / "metadata.jsonl"),
            "prepared_training_data": _file_identity(
                prepared_dir / "generate" / "train_data.parquet"
            ),
            "target_meta_sha256": _sha256(target_meta_path),
            "target_model": _target_checkpoint_identity(target_meta_path),
            "pretrain_meta_sha256": (
                None if pretrain_meta_path is None else _sha256(pretrain_meta_path)
            ),
            "feature_statistics_sha256": _sha256(feature_statistics_path),
            "tokenizer": dict(tokenizer_identity),
        },
        "configuration": {
            "max_concepts": max_concepts,
            "device_type": device.split(":", 1)[0],
            "seed": seed,
            "max_new_tokens": max_new_tokens,
            "extraction_batch_size": extraction_batch_size,
            "self_conditioning_policy": self_conditioning_policy,
            "self_conditioning_masked_policy": self_conditioning_masked_policy,
            "require_exact_statistics_identity": require_exact_statistics_identity,
            "temperature_hex": float(TEMPERATURE).hex(),
            "top_k": TOP_K,
        },
        "runtime": _runtime_identity(device),
    }


def _nl_target_source(
    receipt: NLTargetSourceReceipt,
    *,
    setup: MethodSetup,
    concepts: Mapping[int, str],
    manifest_path: Path,
) -> dict[str, Any]:
    return {
        "algorithm": NL_RETRIEVAL_ALGORITHM,
        "feature_contract": retrieval_feature_identity(setup),
        "concepts": [
            {"concept_id": concept_id, "text": concepts[concept_id]}
            for concept_id in sorted(concepts)
        ],
        "description_embeddings": receipt.description_embeddings.manifest_record(
            relative_to=manifest_path.parent
        ),
        "concept_embeddings": receipt.concept_embeddings.manifest_record(
            relative_to=manifest_path.parent
        ),
    }


def _require_cross_rank_semantic_identity(identity: Mapping[str, Any]) -> None:
    """Reject divergent semantic bytes before any resume store is opened."""

    payload = json.dumps(
        identity,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    local_sha256 = hashlib.sha256(payload).hexdigest()
    if not dist.is_initialized() or dist.get_world_size() <= 1:
        return
    gathered: list[str | None] = [None] * dist.get_world_size()
    dist.all_gather_object(gathered, local_sha256)
    if any(item != local_sha256 for item in gathered):
        raise ValueError(
            "project semantic identity differs across distributed ranks: "
            f"{gathered!r}"
        )


def _project_score_policy(
    cell: CellKey,
    *,
    setup: MethodSetup,
    diffmean: t.Tensor,
    factor: float,
    nl_target: t.Tensor | None = None,
) -> SelfCondScorePolicy:
    """Build one cell policy through the shared scalar/batched dispatch."""

    if cell.method in ("SelfConditioning", "NLSteering"):
        target_scores = diffmean
        if cell.method == "NLSteering":
            if nl_target is None:
                raise ValueError("NLSteering cell has no validated retrieval target")
            target_scores = nl_target
        return SelfCondScoreInterpolation(
            target_scores=target_scores,
            negative_selection=setup.output_negative_selection,
            weight=self_conditioning_weight(factor),
        )
    if cell.method == "SelfConditioningMasked":
        return build_self_conditioning_masked_policy(
            setup,
            diffmean,
            factor,
        )
    raise AssertionError(f"unsupported project method in cell grid: {cell.method}")


def _generate_project_cell(
    cell: CellKey,
    *,
    rows: Sequence[Any],
    rendered_tokens_by_pair: Mapping[tuple[int, int], tuple[int, ...]],
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    dense_diffmeans: Mapping[int, t.Tensor],
    nl_targets: Mapping[int, t.Tensor] | None = None,
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> str:
    concept_id, input_id, factor = cell.validated_source_values(rows)
    exact_tokens = rendered_tokens_by_pair[(concept_id, input_id)]
    score = _project_score_policy(
        cell,
        setup=setup,
        diffmean=dense_diffmeans[concept_id],
        factor=factor,
        nl_target=None if nl_targets is None else nl_targets.get(concept_id),
    )
    with trace_cell(cell):
        with profile_region(
            "axbench_upstream.cell",
            driver="project",
            method=cell.method,
            concept_id=concept_id,
            input_id=input_id,
            factor_hex=cell.factor_hex,
            source_row_position=cell.source_row_position,
        ):
            return generate_self_conditioning(
                setup,
                model,
                tokenizer,
                prepared_by_tokens[exact_tokens],
                score,
                selection_seed=cell.selection_seed,
                seed=cell.token_seed,
                max_new_tokens=max_new_tokens,
            )


def _generate_project_cells(
    cells: Sequence[CellKey],
    *,
    rows: Sequence[Any],
    rendered_tokens_by_pair: Mapping[tuple[int, int], tuple[int, ...]],
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    dense_diffmeans: Mapping[int, t.Tensor],
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> tuple[str, ...]:
    """Generate one same-prompt group of untraced project cells together."""

    grouped = tuple(cells)
    if len(grouped) < 2:
        raise ValueError("project-cell batching requires at least two cells")
    if any(cell.method == "NLSteering" for cell in grouped):
        raise ValueError("NLSteering cells use the scalar generation path")
    pair = (grouped[0].concept_id, grouped[0].input_id)
    exact_tokens = rendered_tokens_by_pair[pair]
    requests: list[SelfConditioningGenerationRequest] = []
    for cell in grouped:
        if (cell.concept_id, cell.input_id) != pair:
            raise ValueError(
                "project-cell batch must share one concept/input pair; "
                f"expected {pair}, got {(cell.concept_id, cell.input_id)}"
            )
        concept_id, _input_id, factor = cell.validated_source_values(rows)
        policy = _project_score_policy(
            cell,
            setup=setup,
            diffmean=dense_diffmeans[concept_id],
            factor=factor,
        )
        requests.append(
            SelfConditioningGenerationRequest(
                policy=policy,
                selection_seed=cell.selection_seed,
                token_seed=cell.token_seed,
            )
        )

    with profile_region(
        "axbench_upstream.cell_batch",
        driver="project",
        concept_id=pair[0],
        input_id=pair[1],
        physical_batch_size=len(grouped),
    ):
        return generate_self_conditioning_batch(
            setup,
            model,
            tokenizer,
            prepared_by_tokens[exact_tokens],
            requests,
            max_new_tokens=max_new_tokens,
        )


def _append_missing_project_cells(
    store: CellResultStore,
    cells: Sequence[CellKey],
    *,
    rows: Sequence[Any],
    rendered_tokens_by_pair: Mapping[tuple[int, int], tuple[int, ...]],
    prepared_by_tokens: Mapping[tuple[int, ...], PreparedPromptContext],
    dense_diffmeans: Mapping[int, t.Tensor],
    nl_targets: Mapping[int, t.Tensor] | None = None,
    setup: MethodSetup,
    model: Any,
    tokenizer: Any,
    max_new_tokens: int,
) -> None:
    """Append missing cells, keeping unsafe or traced prompt groups scalar."""

    missing = store.missing(cells)
    boundary_only_batch_safe = setup.segment_mode is SegmentMode.fixed_tokens
    groups: dict[tuple[int, int], list[CellKey]] = {}
    for cell in missing:
        groups.setdefault((cell.concept_id, cell.input_id), []).append(cell)
    traced_pairs = {
        (cell.concept_id, cell.input_id)
        for cell in missing
        if cell.method in PROJECT_METHODS and cell_trace_requested(cell)
    }

    completed: set[CellKey] = set()
    for cell in missing:
        if cell in completed:
            continue
        pair = (cell.concept_id, cell.input_id)
        old_group = [member for member in groups[pair] if member.method in PROJECT_METHODS]
        # The optimized batch decoder is valid only when the completion stays
        # before fixed mode's next refresh boundary. Turn mode can refresh
        # repeatedly during one completion, so retain its audited scalar
        # generate_stream path, including its structural-stop union.
        if (
            cell.method in PROJECT_METHODS
            and boundary_only_batch_safe
            and pair not in traced_pairs
            and len(old_group) >= 2
        ):
            generations = _generate_project_cells(
                old_group,
                rows=rows,
                rendered_tokens_by_pair=rendered_tokens_by_pair,
                prepared_by_tokens=prepared_by_tokens,
                dense_diffmeans=dense_diffmeans,
                setup=setup,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            )
            if len(generations) != len(old_group):
                raise RuntimeError(
                    "project batch returned the wrong number of generations: "
                    f"expected {len(old_group)}, got {len(generations)}"
                )
            for grouped_cell, generation in zip(old_group, generations, strict=True):
                store.append(grouped_cell, generation)
            completed.update(old_group)
            continue
        store.append(
            cell,
            _generate_project_cell(
                cell,
                rows=rows,
                rendered_tokens_by_pair=rendered_tokens_by_pair,
                prepared_by_tokens=prepared_by_tokens,
                dense_diffmeans=dense_diffmeans,
                nl_targets=nl_targets,
                setup=setup,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=max_new_tokens,
            ),
        )
        completed.add(cell)


def _turn_self_conditioning_segment_geometry(setup: MethodSetup) -> dict[str, Any] | None:
    """Return turn-only generation geometry without changing legacy manifests."""

    segment_mode = setup.segment_mode
    if segment_mode is SegmentMode.fixed_tokens:
        return None
    if segment_mode is not SegmentMode.turn_delimited_fixed_tokens:
        raise ValueError(f"unsupported AxBench prompt segment mode: {segment_mode.value!r}")
    context = setup.prompt_selection_context
    if context is None or context.sampling_bundle is None:
        raise ValueError("turn-delimited AxBench setup has no prompt selector to publish")
    if setup.prompt_statistics_raw_sha256 is None:
        raise ValueError("turn-delimited AxBench setup has no raw statistics identity to publish")

    return {
        "prompt_segment_mode": segment_mode.value,
        "prompt_sae_model": setup.sae_model,
        "prompt_sae_layer": setup.sae_layer,
        "prompt_segment_size_gemma_tokens": setup.prompt_segment_size,
        "prompt_context_window_gemma_tokens": setup.prompt_context_window,
        "generated_cadence_rustbpe_tokens": setup.turn_self_cond_segment_size_rustbpe,
        # The lineage-derived selector state the prompts were actually
        # conditioned with. Two checkpoints whose lineages differ only in
        # missing-percentile policy or selection seed condition their prompts
        # differently and must not publish identical provenance.
        "prompt_missing_percentile_policy": context.missing_percentile.policy.value,
        "prompt_selection_seed": context.sampling_bundle.rng.initial_seed(),
        "prompt_statistics_raw_sha256": setup.prompt_statistics_raw_sha256,
    }


def _manifest(
    *,
    setup: MethodSetup,
    prepared_manifest: dict[str, Any],
    config: dict[str, Any],
    input_source: Mapping[str, str],
    feature_statistics_path: Path,
    sample_diffmean: t.Tensor,
    nl_target_source: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    dataset_revision = _validate_prepared_manifest(prepared_manifest)
    factor_zero_policy = build_self_conditioning_masked_policy(
        setup,
        sample_diffmean,
        0.0,
    )
    eligible_head_rows = int(factor_zero_policy.positive_pool_limit)

    user_config = setup.meta.get("user_config")
    model_tag = user_config.get("model_tag") if isinstance(user_config, dict) else None
    if input_source["kind"] == "main":
        source_identity = {"main_steering_parquet_sha256": input_source["sha256"]}
    else:
        source_identity = {
            "project_source_kind": input_source["kind"],
            "project_source_parquet_sha256": input_source["sha256"],
            "project_source_manifest_sha256": input_source["manifest_sha256"],
        }
    turn_geometry = _turn_self_conditioning_segment_geometry(setup)
    prompt_features = (
        "real context-independent fixed-128 Gemma l22 SAE active/"
        "true-inactive features from nested checkpoint conditioning"
    )
    completion_slots = "one_predicted_slot_for_complete_continuation"
    if turn_geometry is not None:
        prompt_features = (
            f"real context-independent turn-delimited {setup.sae_model} "
            f"l{setup.sae_layer} SAE active/true-inactive features from "
            "checkpoint-recorded online-SAE lineage"
        )
        completion_slots = (
            "marker_anchored_predicted_slots_refreshed_after_each_complete_"
            "checkpoint_rustbpe_cadence_window"
        )
    project_methods = PROJECT_METHODS if nl_target_source is None else NL_PROJECT_METHODS

    return {
        "schema_version": 3 if input_source["kind"] == "main" else 4,
        **_manifest_header(),
        "target": {
            "checkpoint_meta": str(setup.target_meta_path),
            "model_tag": model_tag,
            "step": setup.meta.get("step"),
        },
        "dataset_revision": dataset_revision,
        "prepared_manifest": prepared_manifest,
        "configuration": config,
        "generation": {
            "temperature": TEMPERATURE,
            "top_k": TOP_K,
        },
        **({} if turn_geometry is None else {"self_conditioning_segment_geometry": turn_geometry}),
        "inputs": {
            **source_identity,
            "feature_statistics_sha256": content_sha256(feature_statistics_path),
        },
        "method_order": [
            "DiffMean",
            "LinearProbe",
            "PromptSteering",
            "LoRA",
            *project_methods,
        ],
        "methods": {
            "SelfConditioning": {
                "factors": list(METHOD_FACTORS["SelfConditioning"]),
                "strength_semantics": "alpha_equals_strength_div_6",
                "policy": SELF_CONDITIONING_POLICY,
                "score_formula": (
                    "alpha=0 ranks last_prompt_segment_max_pooled_fp_logits; "
                    "0<alpha<1 ranks lerp(2*sigmoid(logits)-1,dense_diffmean,alpha); "
                    "alpha=1 ranks dense_diffmean"
                ),
                "dense_diffmean_units": (
                    "(mean_positive_percentile_rank-mean_negative_percentile_rank)/100"
                ),
                "factor_0": "pure_prompt_prediction_logit_ranking",
                "factor_6": "pure_dense_feature_space_diffmean",
                "prompt_features": prompt_features,
                "output_counts": {
                    "positive": resolve_raw_feature_selection(
                        setup.meta, setup.meta["user_config"]
                    ),
                    "negative": resolve_raw_negative_selection(
                        setup.meta, setup.meta["user_config"]
                    ),
                },
                "inactive_rule": "lowest_remaining_interpolated_scores",
                "completion_slots": completion_slots,
                "selection_seed": ("sha256(domain,base_seed,concept_id,input_id); factor excluded"),
                "token_seed": ("sha256(base_seed,method,concept_id,input_id,float_factor_hex)"),
            },
            "SelfConditioningMasked": {
                "factors": list(METHOD_FACTORS["SelfConditioningMasked"]),
                "policy": SELF_CONDITIONING_MASKED_POLICY,
                "strength_semantics": "linear_rank_pool_limit_from_N_to_2x_side_mean",
                "rounding": "nearest_ties_to_even",
                "clamp": {"minimum": 1, "maximum": "eligible_head_rows"},
                "eligible_head_rows": eligible_head_rows,
                "positive_mean": float(setup.output_selection.mean),
                "negative_mean": float(setup.output_negative_selection.mean),
                "pool_limits": [
                    {
                        "factor": factor,
                        "positive": self_conditioning_masked_pool_limit(
                            factor,
                            n_candidates=eligible_head_rows,
                            mean=float(setup.output_selection.mean),
                        ),
                        "negative": self_conditioning_masked_pool_limit(
                            factor,
                            n_candidates=eligible_head_rows,
                            mean=float(setup.output_negative_selection.mean),
                        ),
                    }
                    for factor in SELF_CONDITIONING_FACTORS
                ],
                "candidate_eligibility": (
                    "post_feature_mask_sign_consistent_stable_diffmean_rank"
                ),
                "final_ranking": "last_prompt_segment_max_pooled_float32_model_logits",
                "exact_zero_diffmean_eligible": False,
                "selection_seed": ("sha256(domain,base_seed,concept_id,input_id); factor excluded"),
                "token_seed": ("sha256(base_seed,method,concept_id,input_id,float_factor_hex)"),
            },
            **(
                {}
                if nl_target_source is None
                else {
                    "NLSteering": {
                        "factors": list(SUPPORTED_METHOD_FACTORS["NLSteering"]),
                        "policy": NL_STEERING_POLICY,
                        "target_source": dict(nl_target_source),
                    }
                }
            ),
        },
    }


#: Absolute floor on the blank COUNT before the EARLY (first-concept) check is
#: allowed to fire on the blank rate. The final, authoritative check --
#: ``validate_steering_frame(project_frame, source="project", ...)`` at the
#: end of :func:`run_project_methods` -- has no floor and is unchanged.
#:
#: Why a floor is required here, and why 5. The early check sees 120 declared
#: cells (one concept x 10 inputs x 12 factors), against a bound that is a
#: RATE (:data:`~scripts.eval.axbench_upstream.merge_methods.MAX_BLANK_GENERATION_FRACTION`,
#: 1%). At the measured honest blank rate of 0.25% that population expects
#: lambda = 0.3 blanks, and TWO blanks -- an ordinary Poisson fluctuation --
#: already reads as 1.67% and trips the bound. P(>=2 | lambda=0.3) = 3.67% per
#: method.
#:
#: A false stop here is UNRECOVERABLE by rerunning: generation seeds are
#: derived from the cell key (:func:`derive_generation_seed`), so a rerun
#: regenerates the identical concept and this check fires again, forever. The
#: operator's only exit is to delete the partial output and change the plan,
#: or to edit this guard -- i.e. a false positive is a hard block on a
#: healthy run, not "one concept of wasted work".
#:
#: With the floor at 5 blanks: P(>=5 | lambda=0.3) = 1.6e-5, so a false stop
#: becomes effectively impossible. Detection of what
#: this tripwire is FOR is barely touched: at a 10% blank rate (lambda = 12)
#: P(>=5) = 0.992, and a total collapse (every cell blank, 120 of 120) fires
#: with certainty. What the floor cedes is the marginal 1-4% regime on a
#: single concept -- exactly the regime the sample is too small to call --
#: and that regime remains fully covered by the authoritative full-frame
#: check, which sees every concept's cells and keeps no floor at all.
EARLY_BLANK_MIN_COUNT = 5


def _assert_first_concept_not_collapsed(
    generations_by_method: Mapping[str, Sequence[str]],
    concept_id: int,
    *,
    label: str = "project method",
) -> None:
    """Run both collapse tripwires against the FIRST concept's generations.

    The authoritative check is still the full-frame
    ``validate_steering_frame`` call at the end of
    :func:`run_project_methods`. This one exists because that check only
    speaks after every concept has been generated: a conditioning regression
    that empties every generation would otherwise burn the entire run's GPU
    budget before anything noticed, and the operator would pay for every
    remaining concept of blank output to be told what the first concept
    already showed.

    Population and constants, explicitly. The blank check applies the SAME
    rate bound
    (:data:`~scripts.eval.axbench_upstream.merge_methods.MAX_BLANK_GENERATION_FRACTION`)
    to THIS CONCEPT's generations only -- roughly 1/N of the rows the final
    check will see -- plus an absolute floor of
    :data:`EARLY_BLANK_MIN_COUNT` blanks. See that constant for the full
    arithmetic: a small sample makes the RATE alone dominated by Poisson
    noise, and derived seeds make a false stop unrecoverable by rerunning.
    The two BLANK conditions are ANDed, so the floor can only ever QUIETEN
    that check. The independent median-length check uses the same
    :data:`~scripts.eval.axbench_upstream.merge_methods.MIN_MEDIAN_GENERATION_CHARS`
    floor as final validation with no count floor: a concept full of one- or
    two-character nonblank output is already enough evidence of the distinct
    decode-collapse failure it detects.

    Called BEFORE the remaining concepts are generated, so a collapse can
    never burn budget past the first concept.
    """
    for method, generations in generations_by_method.items():
        lengths = [len(str(value).strip()) for value in generations]
        cell_count = len(lengths)
        blank_count = sum(1 for length in lengths if length == 0)
        blank_fraction = (blank_count / cell_count) if cell_count else 0.0
        if blank_count >= EARLY_BLANK_MIN_COUNT and blank_fraction > MAX_BLANK_GENERATION_FRACTION:
            raise ValueError(
                f"{label} {method!r} concept_id={concept_id} has "
                f"{blank_count} of {cell_count} blank generations after stripping "
                f"(fraction {blank_fraction:.4f}) in its first concept alone -- "
                "collapse detected before generating the remaining concepts"
            )

        median_length = float(median(lengths)) if cell_count else 0.0
        if median_length < MIN_MEDIAN_GENERATION_CHARS:
            raise ValueError(
                f"{label} {method!r} concept_id={concept_id} has collapsed output; "
                f"median stripped length is {median_length:g}, "
                f"minimum is {MIN_MEDIAN_GENERATION_CHARS} in its first concept alone -- "
                "collapse detected before generating the remaining concepts"
            )


def run_project_methods(
    *,
    main_path: Path | None,
    prepared_dir: Path,
    output_path: Path,
    manifest_path: Path,
    target_meta_path: Path,
    feature_statistics_path: Path,
    max_concepts: int,
    device: str,
    seed: int,
    max_new_tokens: int,
    extraction_batch_size: int,
    self_conditioning_policy: str,
    self_conditioning_masked_policy: str,
    force: bool = False,
    pretrain_meta_path: Path | None = None,
    require_exact_statistics_identity: bool = True,
    project_input_path: Path | None = None,
    project_input_manifest_path: Path | None = None,
    nl_target_source_manifest_path: Path | None = None,
    concept_ids: list[int] | tuple[int, ...] | None = None,
) -> pd.DataFrame:
    """Fit dense targets, prepare real prompts, and generate selected project methods."""

    _validate_scalar_config(
        max_concepts=max_concepts,
        device=device,
        seed=seed,
        max_new_tokens=max_new_tokens,
        extraction_batch_size=extraction_batch_size,
        self_conditioning_policy=self_conditioning_policy,
        self_conditioning_masked_policy=self_conditioning_masked_policy,
        force=force,
        require_exact_statistics_identity=require_exact_statistics_identity,
    )
    selected_ids = resolve_concept_ids(max_concepts, concept_ids)
    if concept_ids is not None and nl_target_source_manifest_path is not None:
        raise ValueError("concept_ids is supported only for the two default project methods")
    rank, world_size = _distributed_coordinates()
    stage_device = device if device.startswith("cuda") else "cpu"
    with rank_synchronized_stage("project input preflight", device=stage_device):
        input_path, input_source_kind, project_input_manifest_path = _resolve_input_source(
            main_path,
            project_input_path,
            project_input_manifest_path,
        )
        prepared_dir = Path(prepared_dir).expanduser().resolve()
        output_path = Path(output_path).expanduser().resolve()
        manifest_path = Path(manifest_path).expanduser().resolve()
        target_meta_path = Path(target_meta_path).expanduser().resolve()
        feature_statistics_path = Path(feature_statistics_path).expanduser().resolve()
        pretrain_meta_path = (
            None if pretrain_meta_path is None else Path(pretrain_meta_path).expanduser().resolve()
        )
        nl_target_source_manifest_path = (
            None
            if nl_target_source_manifest_path is None
            else Path(nl_target_source_manifest_path).expanduser().resolve()
        )
        project_methods = (
            PROJECT_METHODS
            if nl_target_source_manifest_path is None
            else NL_PROJECT_METHODS
        )
        rejected_path = _validate_artifact_paths(
            input_path=input_path,
            input_manifest_path=project_input_manifest_path,
            prepared_dir=prepared_dir,
            output_path=output_path,
            manifest_path=manifest_path,
            target_meta_path=target_meta_path,
            feature_statistics_path=feature_statistics_path,
            pretrain_meta_path=pretrain_meta_path,
            nl_target_source_manifest_path=nl_target_source_manifest_path,
        )
        existing = [path for path in (output_path, manifest_path) if path.exists()]
        if existing and not force:
            raise FileExistsError(
                f"project-method output already exists: {existing}; pass force=true to rebuild"
            )
        if force and manifest_path.is_file():
            _reject_forced_threshold_era_output(manifest_path)

        prepared_manifest = _json_object(prepared_dir / "prepare_manifest.json")
        _validate_prepared_manifest(prepared_manifest)
        if input_source_kind == "main":
            source_frame = pd.read_parquet(input_path)
            validate_steering_frame(
                source_frame,
                source="main",
                max_concepts=max_concepts,
                check_method_values=False,
            )
            project_frame = source_frame.loc[:, list(INPUT_COLUMNS)].copy()
            input_source = {"kind": "main", "sha256": _sha256(input_path)}
        else:
            assert project_input_manifest_path is not None
            source_frame, source_sha256, source_manifest_sha256 = load_project_input_artifact(
                input_path=input_path,
                manifest_path=project_input_manifest_path,
                prepared_dir=prepared_dir,
                max_concepts=max_concepts,
            )
            project_frame = source_frame.copy()
            input_source = {
                "kind": "project_input",
                "sha256": source_sha256,
                "manifest_sha256": source_manifest_sha256,
            }
        training = select_training_texts(prepared_dir, max_concepts=max_concepts)
        main_concepts = (
            project_frame.groupby("concept_id", sort=False)["input_concept"].first().to_dict()
        )
        for concept_id, examples in training.items():
            if main_concepts.get(concept_id) != examples.concept:
                raise ValueError(
                    "prepared metadata concept does not match the input frame: "
                    f"concept_id={concept_id}, prepared={examples.concept!r}, "
                    f"input={main_concepts.get(concept_id)!r}"
                )
        nanochat_tokenizer = get_tokenizer()
        tokenizer_identity = _tokenizer_artifact_identity()
        # Pass the renderer by name so the existing module-level test seam
        # (monkeypatching this module's ``render_prompt_for_chat_completion``)
        # still governs prompt rendering.
        prompt_plan = build_rendered_prompt_plan(
            project_frame,
            nanochat_tokenizer,
            declared_factors=SELF_CONDITIONING_FACTORS,
            render=render_prompt_for_chat_completion,
        )
        # Replay the complete source prompt plan: preparation consumes shared
        # checkpoint RNG in canonical order. Filtering it would change contexts.
        project_frame = project_frame.loc[project_frame.concept_id.isin(selected_ids)].copy()
        rows = tuple(project_frame.itertuples(index=False))
        rendered_tokens_by_pair = prompt_plan.rendered_tokens_by_pair
        prompt_by_tokens = prompt_plan.prompt_by_tokens

        first_concept_id = int(rows[0].concept_id)
        cells = build_project_cell_grid(project_frame, seed=seed, methods=project_methods)
        first_cells = tuple(cell for cell in cells if cell.concept_id == first_concept_id)
        resume_root = manifest_path.parent / "_project_cell_resume_v1"
        semantic_identity = _project_resume_identity(
            input_source=input_source,
            prepared_dir=prepared_dir,
            target_meta_path=target_meta_path,
            feature_statistics_path=feature_statistics_path,
            pretrain_meta_path=pretrain_meta_path,
            tokenizer_identity=tokenizer_identity,
            max_concepts=max_concepts,
            device=device,
            seed=seed,
            max_new_tokens=max_new_tokens,
            extraction_batch_size=extraction_batch_size,
            self_conditioning_policy=self_conditioning_policy,
            self_conditioning_masked_policy=self_conditioning_masked_policy,
            require_exact_statistics_identity=require_exact_statistics_identity,
        )
        if concept_ids is not None:
            semantic_identity["configuration"]["concept_ids"] = list(selected_ids)
    with rank_synchronized_stage(
        "project checkpoint geometry validation",
        device=stage_device,
    ):
        # Resolve every checkpoint witness, including turn cadence, before the
        # resume store creates directories or publishes identity.json.
        setup = load_method_setup(
            target_meta_path,
            feature_statistics_path,
            pretrain_meta_path=pretrain_meta_path,
            require_exact_statistics_identity=require_exact_statistics_identity,
        )

    nl_targets: dict[int, t.Tensor] = {}
    nl_target_source: dict[str, Any] | None = None
    nl_receipt: NLTargetSourceReceipt | None = None
    if nl_target_source_manifest_path is not None:
        with rank_synchronized_stage("project NL target preflight", device=stage_device):
            nl_receipt = load_nl_target_source_receipt(nl_target_source_manifest_path)
            _validate_nl_artifact_paths(
                nl_receipt,
                output_path=output_path,
                manifest_path=manifest_path,
                rejected_path=rejected_path,
                resume_root=resume_root,
                fit_dir=manifest_path.parent / "_project_fit_resume_v1",
            )
            concepts = {
                concept_id: training[concept_id].concept for concept_id in selected_ids
            }
            nl_targets = build_retrieval_target_scores(
                setup=setup,
                description_embeddings_path=nl_receipt.description_embeddings.path,
                concept_embeddings_path=nl_receipt.concept_embeddings.path,
                concepts=concepts,
                expected_hashes=nl_receipt.expected_hashes,
            )
            nl_target_source = _nl_target_source(
                nl_receipt,
                setup=setup,
                concepts=concepts,
                manifest_path=manifest_path,
            )
            # The identity is local to this run; only the published source needs
            # a separate copy when excluding operational paths from resume identity.
            semantic_target_source = deepcopy(nl_target_source)
            for artifact_name in ("description_embeddings", "concept_embeddings"):
                semantic_target_source[artifact_name].pop("path")
            semantic_identity["configuration"]["project_methods"] = list(project_methods)
            semantic_identity["inputs"]["nl_target_source"] = {
                "receipt_sha256": nl_receipt.sha256,
                "target_scores_sha256": retrieval_targets_sha256(nl_targets),
                **semantic_target_source,
            }
        assert nl_receipt is not None
        with rank_synchronized_stage("project NL rank identity", device=stage_device):
            _require_cross_rank_semantic_identity(semantic_identity)

    store: CellResultStore | None = None
    with rank_synchronized_stage("project resume initialization", device=stage_device):
        if rank == 0:
            store = CellResultStore(
                root=resume_root,
                cells=cells,
                tripwire_cells=first_cells,
                semantic_identity=semantic_identity,
                rank=rank,
                world_size=world_size,
            )
    with rank_synchronized_stage("project resume validation", device=stage_device):
        if rank != 0:
            store = CellResultStore(
                root=resume_root,
                cells=cells,
                tripwire_cells=first_cells,
                semantic_identity=semantic_identity,
                rank=rank,
                world_size=world_size,
            )
    assert store is not None
    prepared_by_tokens: dict[tuple[int, ...], PreparedPromptContext] = {}
    dense_diffmeans: dict[int, t.Tensor] = {}
    extraction_model: Any | None = None
    sae: Any | None = None
    try:
        with rank_synchronized_stage("project conditioning preparation", device=stage_device):
            logger.info(
                "Preparing {} unique real prompt contexts and fitting {} dense DiffMean vectors",
                len(prompt_by_tokens),
                len(selected_ids),
            )
            extraction_model, sae = load_extraction_stack(setup, device)
            for exact_tokens, (_prompt, assistant_end_id) in prompt_by_tokens.items():
                prepared_by_tokens[exact_tokens] = prepare_prompt_context(
                    setup,
                    extraction_model,
                    sae,
                    nanochat_tokenizer,
                    token_ids=exact_tokens,
                    assistant_end_id=assistant_end_id,
                )
        fit_dir = manifest_path.parent / "_project_fit_resume_v1"
        with rank_synchronized_stage(
            "project dense DiffMean fit initialization",
            device=stage_device,
        ):
            if rank == 0:
                _prepare_project_fit_dir(
                    fit_dir=fit_dir,
                    semantic_fingerprint=store.semantic_fingerprint,
                    concept_ids=selected_ids,
                )
        with rank_synchronized_stage(
            "project dense DiffMean fit validation",
            device=stage_device,
        ):
            if rank != 0:
                _prepare_project_fit_dir(
                    fit_dir=fit_dir,
                    semantic_fingerprint=store.semantic_fingerprint,
                    concept_ids=selected_ids,
                )
        with rank_synchronized_stage("project dense DiffMean fits", device=stage_device):
            assert extraction_model is not None
            assert sae is not None
            missing_owned = {
                concept_id: training[concept_id]
                for concept_id in selected_ids
                if concept_id % world_size == rank
                and not _project_fit_path(fit_dir, concept_id).is_file()
            }
            if missing_owned:
                with profile_region(
                    "axbench_upstream.project.concept_fits",
                    owned_concepts=len(missing_owned),
                ):
                    newly_fitted = fit_concept_dense_diffmeans(
                        setup,
                        extraction_model,
                        sae,
                        missing_owned,
                        extraction_batch_size=extraction_batch_size,
                    )
                if set(newly_fitted) != set(missing_owned):
                    raise CellStoreError(
                        "project dense DiffMean fitter returned the wrong concept set: "
                        f"expected={sorted(missing_owned)}, actual={sorted(newly_fitted)}"
                    )
                for concept_id, scores in newly_fitted.items():
                    _save_project_fit(
                        _project_fit_path(fit_dir, concept_id),
                        scores,
                        concept_id=concept_id,
                        semantic_fingerprint=store.semantic_fingerprint,
                    )

        expected_fit_width = (
            setup.n_sae_features if setup.feature_vocab is None else setup.feature_vocab.size
        )
        with rank_synchronized_stage("project dense DiffMean validation", device=stage_device):
            dense_diffmeans = {
                concept_id: _load_project_fit(
                    _project_fit_path(fit_dir, concept_id),
                    concept_id=concept_id,
                    semantic_fingerprint=store.semantic_fingerprint,
                    expected_width=expected_fit_width,
                )
                for concept_id in selected_ids
            }
            if rank == 0:
                _install_project_fit_receipt(
                    resume_root=resume_root,
                    fit_dir=fit_dir,
                    semantic_fingerprint=store.semantic_fingerprint,
                    concept_ids=selected_ids,
                )
        with rank_synchronized_stage("project masked-pool preflight", device=stage_device):
            # Pool feasibility is fully determined here; fail a doomed masked
            # sweep before the target model is loaded and generation budget spent.
            for preflight_concept_id, concept_diffmean in dense_diffmeans.items():
                assert_masked_diffmean_feasible(
                    setup,
                    concept_diffmean,
                    concept_id=int(preflight_concept_id),
                )
    finally:
        extraction_model = None
        sae = None
        gc.collect()
        if device.startswith("cuda") and t.cuda.is_available():
            t.cuda.empty_cache()

    with rank_synchronized_stage("project target-model validation", device=stage_device):
        logger.info("Loading target checkpoint after releasing the extraction stack")
        model, tokenizer = load_target_model(setup, device)
        for exact_tokens, prepared in prepared_by_tokens.items():
            prompt, _assistant_end_id = prompt_by_tokens[exact_tokens]
            assert_prepared_prompt_matches(tokenizer, prompt, prepared)
            if len(prepared.token_ids) + max_new_tokens > int(model.dec_config.sequence_len):
                raise ValueError(
                    "prepared prompt plus generation budget exceeds decoder sequence length"
                )
            if not prepared.segments:
                raise ValueError("prepared prompt has no real feature segments")
            for segment_index, segment in enumerate(prepared.segments):
                selected = {
                    int(feature_id): int(bucket)
                    for feature_id, bucket in zip(
                        segment.feature_ids,
                        segment.buckets,
                        strict=True,
                    )
                }
                if len(selected) != len(segment.feature_ids):
                    raise ValueError(
                        f"prepared prompt segment {segment_index} contains duplicate feature IDs"
                    )
                validate_feature_dict(
                    selected,
                    setup.n_sae_features,
                    2,
                    name=f"prepared prompt segment {segment_index}",
                    feature_vocab=setup.feature_vocab,
                )

    rank_cells = owned_project_cells(
        cells,
        rank=rank,
        world_size=world_size,
        methods=project_methods,
    )
    owned_first_cells = tuple(cell for cell in rank_cells if cell.concept_id == first_concept_id)
    owned_bulk_cells = tuple(cell for cell in rank_cells if cell.concept_id != first_concept_id)

    # The first-seen concept remains a complete phase even when source rows
    # are interleaved. Prompt-group striping changes placement only; CellKey
    # seeds and canonical output positions remain placement-free.
    with rank_synchronized_stage("project first-concept generation", device=stage_device):
        _append_missing_project_cells(
            store,
            owned_first_cells,
            rows=rows,
            rendered_tokens_by_pair=rendered_tokens_by_pair,
            prepared_by_tokens=prepared_by_tokens,
            dense_diffmeans=dense_diffmeans,
            nl_targets=nl_targets,
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        store.flush()

    with rank_synchronized_stage("project first-concept tripwire", device=stage_device):
        if rank == 0:
            with CellResultStore(
                root=resume_root,
                cells=cells,
                tripwire_cells=first_cells,
                semantic_identity=semantic_identity,
                rank=rank,
                world_size=world_size,
            ) as tripwire_store:
                first_results = tripwire_store.results_for(first_cells)
                _assert_first_concept_not_collapsed(
                    {
                        method: [
                            first_results[cell] for cell in first_cells if cell.method == method
                        ]
                        for method in project_methods
                    },
                    first_concept_id,
                )
                tripwire_store.mark_tripwire_passed()
    with rank_synchronized_stage("project tripwire receipt", device=stage_device):
        store.require_tripwire_passed()

    with rank_synchronized_stage("project bulk generation", device=stage_device):
        _append_missing_project_cells(
            store,
            owned_bulk_cells,
            rows=rows,
            rendered_tokens_by_pair=rendered_tokens_by_pair,
            prepared_by_tokens=prepared_by_tokens,
            dense_diffmeans=dense_diffmeans,
            nl_targets=nl_targets,
            setup=setup,
            model=model,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        store.flush()
    store.close()

    with rank_synchronized_stage("project result reconstruction", device=stage_device):
        with CellResultStore(
            root=resume_root,
            cells=cells,
            tripwire_cells=first_cells,
            semantic_identity=semantic_identity,
            rank=rank,
            world_size=world_size,
        ) as completed_store:
            generations = completed_store.complete_results()

    strengths = [float(value) for value in project_frame["factor"]]
    for method_offset, method in enumerate(project_methods):
        method_generations = [
            generations[cells[position * len(project_methods) + method_offset]]
            for position in range(len(rows))
        ]
        project_frame[f"{method}_steered_generation"] = pd.Series(
            method_generations,
            index=project_frame.index,
            dtype="object",
        )
        project_frame[f"{method}_strength"] = pd.Series(
            strengths,
            index=project_frame.index,
            dtype="float64",
        )
    project_columns = method_columns(project_methods)
    project_frame = project_frame.loc[:, [*INPUT_COLUMNS, *project_columns]]
    # Pinned exception notes differ from encdec_baselines/run.py; keep this writer local.
    with rank_synchronized_stage("project final validation", device=stage_device):
        try:
            validate_steering_frame(
                project_frame,
                source="project",
                max_concepts=max_concepts,
                concept_ids=selected_ids,
                methods=project_methods,
            )
        except ValueError as exc:
            if rank == 0:
                try:
                    atomic_write_parquet(
                        pa.Table.from_pandas(project_frame, preserve_index=False),
                        rejected_path,
                    )
                except BaseException as rejected_exc:
                    exc.add_note(
                        f"failed to write rejected project frame to {rejected_path}: "
                        f"{type(rejected_exc).__name__}: {rejected_exc}"
                    )
                    raise exc from rejected_exc
                exc.add_note(f"complete rejected project frame written to {rejected_path}")
            raise

    config = {
        "main_path": None if main_path is None else str(Path(main_path).expanduser().resolve()),
        "prepared_dir": str(prepared_dir),
        "output_path": str(output_path),
        "manifest_path": str(manifest_path),
        "target_meta_path": str(target_meta_path),
        "feature_statistics_path": str(feature_statistics_path),
        "max_concepts": max_concepts,
        "device": device,
        "seed": seed,
        "max_new_tokens": max_new_tokens,
        "extraction_batch_size": extraction_batch_size,
        "self_conditioning_policy": self_conditioning_policy,
        "self_conditioning_masked_policy": self_conditioning_masked_policy,
        "force": force,
        "pretrain_meta_path": None if pretrain_meta_path is None else str(pretrain_meta_path),
        "require_exact_statistics_identity": require_exact_statistics_identity,
    }
    if concept_ids is not None:
        config["concept_ids"] = list(selected_ids)
    if input_source_kind != "main":
        config["project_input_path"] = str(input_path)
        config["project_input_manifest_path"] = str(project_input_manifest_path)
    if nl_target_source_manifest_path is not None:
        config["nl_target_source_manifest_path"] = str(nl_target_source_manifest_path)
    if nl_receipt is not None:
        # Align every rank's validation before rank 0 can publish either artifact.
        with rank_synchronized_stage("project NL source revalidation", device=stage_device):
            current_receipt = load_nl_target_source_receipt(nl_receipt.path)
            if current_receipt != nl_receipt:
                raise ValueError("NL target receipt identity changed during generation")
    with rank_synchronized_stage("project canonical publication", device=stage_device):
        if rank == 0:
            manifest = _manifest(
                setup=setup,
                prepared_manifest=prepared_manifest,
                config=config,
                input_source=input_source,
                feature_statistics_path=feature_statistics_path,
                sample_diffmean=dense_diffmeans[min(dense_diffmeans)],
                nl_target_source=nl_target_source,
            )
            manifest_payload = (
                json.dumps(manifest, indent=2, sort_keys=True, allow_nan=False) + "\n"
            )
            if force and manifest_path.exists():
                manifest_path.unlink()
            atomic_write_parquet(
                pa.Table.from_pandas(project_frame, preserve_index=False),
                output_path,
            )
            atomic_write_text(manifest_payload, manifest_path)
            logger.info("Wrote {} project-method rows to {}", len(project_frame), output_path)
    return project_frame


def run_project_methods_from_dictconfig(cfg: DictConfig) -> pd.DataFrame:
    """Convert the thin Hydra config and call :func:`run_project_methods`."""

    raw = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(raw, Mapping):
        raise ValueError("AxBench project-method config must be a mapping")
    if not all(isinstance(key, str) for key in raw):
        raise ValueError("AxBench project-method config keys must be strings")
    config = cast(Mapping[str, Any], raw)
    missing = sorted(_CONFIG_KEYS - set(config))
    unknown = sorted(set(config) - _CONFIG_KEYS - _OPTIONAL_CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(
            f"AxBench project-method config keys do not match; missing={missing}, unknown={unknown}"
        )
    return run_project_methods(
        main_path=(None if config["main_path"] is None else Path(str(config["main_path"]))),
        prepared_dir=Path(str(config["prepared_dir"])),
        output_path=Path(str(config["output_path"])),
        manifest_path=Path(str(config["manifest_path"])),
        target_meta_path=Path(str(config["target_meta_path"])),
        feature_statistics_path=Path(str(config["feature_statistics_path"])),
        max_concepts=config["max_concepts"],
        concept_ids=config.get("concept_ids"),
        device=config["device"],
        seed=config["seed"],
        max_new_tokens=config["max_new_tokens"],
        extraction_batch_size=config["extraction_batch_size"],
        self_conditioning_policy=config["self_conditioning_policy"],
        self_conditioning_masked_policy=config["self_conditioning_masked_policy"],
        force=config["force"],
        pretrain_meta_path=(
            None
            if config["pretrain_meta_path"] is None
            else Path(str(config["pretrain_meta_path"]))
        ),
        require_exact_statistics_identity=config["require_exact_statistics_identity"],
        project_input_path=(
            None
            if config.get("project_input_path") is None
            else Path(str(config["project_input_path"]))
        ),
        project_input_manifest_path=(
            None
            if config.get("project_input_manifest_path") is None
            else Path(str(config["project_input_manifest_path"]))
        ),
        nl_target_source_manifest_path=(
            None
            if config.get("nl_target_source_manifest_path") is None
            else Path(str(config["nl_target_source_manifest_path"]))
        ),
    )


__all__ = [
    "EARLY_BLANK_MIN_COUNT",
    "PROJECT_COLUMNS",
    "NL_PROJECT_COLUMNS",
    "SEED_MODULUS",
    "derive_generation_seed",
    "derive_selection_seed",
    "run_project_methods",
    "run_project_methods_from_dictconfig",
]
