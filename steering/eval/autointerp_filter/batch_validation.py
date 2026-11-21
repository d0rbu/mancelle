"""Strict numeric validation for fixed-block token-example materialization."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import subprocess
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import torch as t
import transformers
from beartype import beartype

from steering.eval.autointerp_filter.types import AutoInterpBatchValidationError
from steering.refined_types import (
    AtLeastTwoInt,
    AtLeastTwoIntLike,
    LowercaseSha256,
    LowercaseSha256Like,
    NonEmptyStr,
    NonEmptyStrLike,
    NonNegativeFiniteNumber,
    NonNegativeFiniteNumberLike,
    NonNegativeInt,
    NonNegativeIntLike,
    PositiveInt,
    PositiveIntLike,
    ZeroInt,
    ZeroIntLike,
)

BATCH_VALIDATION_PROFILE_VERSION = "autointerp_batch_tolerance_v2"

class BatchSurfaceKind(StrEnum):
    """Numeric surfaces covered by the batch-validation profile."""

    SELECTED_SAE = "selected_sae"
    RAW_RESIDUAL = "raw_residual"

    @classmethod
    def coerce(cls, value: BatchSurfaceKind | str) -> BatchSurfaceKind:
        try:
            return cls(value)
        except ValueError as exc:
            raise AutoInterpBatchValidationError(
                f"unknown batch validation surface kind {value!r}"
            ) from exc


@dataclass(frozen=True, init=False)
class NumericTolerance:
    """Independent finite error bounds for one numeric comparison surface."""

    max_abs: NonNegativeFiniteNumber
    p99_abs: NonNegativeFiniteNumber
    mean_abs: NonNegativeFiniteNumber

    @beartype
    def __init__(
        self,
        max_abs: NonNegativeFiniteNumberLike,
        p99_abs: NonNegativeFiniteNumberLike,
        mean_abs: NonNegativeFiniteNumberLike,
    ) -> None:
        object.__setattr__(self, "max_abs", NonNegativeFiniteNumber.parse(max_abs))
        object.__setattr__(self, "p99_abs", NonNegativeFiniteNumber.parse(p99_abs))
        object.__setattr__(self, "mean_abs", NonNegativeFiniteNumber.parse(mean_abs))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.p99_abs > self.max_abs or self.mean_abs > self.max_abs:
            raise AutoInterpBatchValidationError(
                "numeric tolerance p99_abs and mean_abs must not exceed max_abs"
            )


@dataclass(frozen=True, init=False)
class BatchCalibrationContract:
    """Exact model/SAE/input surface on which a tolerance was calibrated."""

    model_name: NonEmptyStr
    model_dtype: NonEmptyStr
    activation_type: NonEmptyStr
    layer: NonNegativeInt
    segment_size: PositiveInt
    sae_release: NonEmptyStr
    sae_id: NonEmptyStr
    hook_name: NonEmptyStr

    @beartype
    def __init__(
        self,
        model_name: NonEmptyStrLike,
        model_dtype: NonEmptyStrLike,
        activation_type: NonEmptyStrLike,
        layer: NonNegativeIntLike,
        segment_size: PositiveIntLike,
        sae_release: NonEmptyStrLike,
        sae_id: NonEmptyStrLike,
        hook_name: NonEmptyStrLike,
    ) -> None:
        object.__setattr__(self, "model_name", NonEmptyStr.parse(model_name))
        object.__setattr__(self, "model_dtype", NonEmptyStr.parse(model_dtype))
        object.__setattr__(self, "activation_type", NonEmptyStr.parse(activation_type))
        object.__setattr__(self, "layer", NonNegativeInt.parse(layer))
        object.__setattr__(self, "segment_size", PositiveInt.parse(segment_size))
        object.__setattr__(self, "sae_release", NonEmptyStr.parse(sae_release))
        object.__setattr__(self, "sae_id", NonEmptyStr.parse(sae_id))
        object.__setattr__(self, "hook_name", NonEmptyStr.parse(hook_name))

    def identity(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, init=False)
class BatchToleranceProfile:
    """Frozen acceptance envelope established by calibration then held out."""

    name: NonEmptyStr
    batch_size: AtLeastTwoInt
    selected_sae: NumericTolerance
    raw_residual: NumericTolerance
    calibration_contract: BatchCalibrationContract
    max_support_mismatches: ZeroInt = ZeroInt.parse(0)
    runtime_fingerprint_sha256: LowercaseSha256 | None = None
    calibration_observations_sha256: LowercaseSha256 | None = None

    @beartype
    def __init__(
        self,
        name: NonEmptyStrLike,
        batch_size: AtLeastTwoIntLike,
        selected_sae: NumericTolerance,
        raw_residual: NumericTolerance,
        calibration_contract: BatchCalibrationContract,
        max_support_mismatches: ZeroIntLike = 0,
        runtime_fingerprint_sha256: LowercaseSha256Like | None = None,
        calibration_observations_sha256: LowercaseSha256Like | None = None,
    ) -> None:
        object.__setattr__(self, "name", NonEmptyStr.parse(name))
        object.__setattr__(self, "batch_size", AtLeastTwoInt.parse(batch_size))
        object.__setattr__(self, "selected_sae", selected_sae)
        object.__setattr__(self, "raw_residual", raw_residual)
        object.__setattr__(self, "calibration_contract", calibration_contract)
        object.__setattr__(
            self,
            "max_support_mismatches",
            ZeroInt.parse(max_support_mismatches),
        )
        object.__setattr__(
            self,
            "runtime_fingerprint_sha256",
            None
            if runtime_fingerprint_sha256 is None
            else LowercaseSha256.parse(runtime_fingerprint_sha256),
        )
        object.__setattr__(
            self,
            "calibration_observations_sha256",
            None
            if calibration_observations_sha256 is None
            else LowercaseSha256.parse(calibration_observations_sha256),
        )

    def identity(self) -> dict[str, Any]:
        return {
            "version": BATCH_VALIDATION_PROFILE_VERSION,
            **asdict(self),
        }

    def tolerance_for(self, surface_kind: BatchSurfaceKind) -> NumericTolerance:
        if surface_kind is BatchSurfaceKind.SELECTED_SAE:
            return self.selected_sae
        if surface_kind is BatchSurfaceKind.RAW_RESIDUAL:
            return self.raw_residual
        raise AutoInterpBatchValidationError(
            f"unknown batch validation surface kind {surface_kind!r}"
        )


@dataclass(frozen=True, init=False)
class BatchDifferenceMetrics:
    """Complete scalar surface used by the fail-loud tolerance gate."""

    count: PositiveInt
    support_mismatches: NonNegativeInt
    max_abs: NonNegativeFiniteNumber
    p99_abs: NonNegativeFiniteNumber
    mean_abs: NonNegativeFiniteNumber

    @beartype
    def __init__(
        self,
        count: PositiveIntLike,
        support_mismatches: NonNegativeIntLike,
        max_abs: NonNegativeFiniteNumberLike,
        p99_abs: NonNegativeFiniteNumberLike,
        mean_abs: NonNegativeFiniteNumberLike,
    ) -> None:
        object.__setattr__(self, "count", PositiveInt.parse(count))
        object.__setattr__(
            self,
            "support_mismatches",
            NonNegativeInt.parse(support_mismatches),
        )
        object.__setattr__(self, "max_abs", NonNegativeFiniteNumber.parse(max_abs))
        object.__setattr__(self, "p99_abs", NonNegativeFiniteNumber.parse(p99_abs))
        object.__setattr__(self, "mean_abs", NonNegativeFiniteNumber.parse(mean_abs))
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.support_mismatches > self.count:
            raise AutoInterpBatchValidationError(
                "batch difference metric support_mismatches must be a strict int "
                "between zero and count"
            )
        if self.p99_abs > self.max_abs or self.mean_abs > self.max_abs:
            raise AutoInterpBatchValidationError(
                "batch difference metric p99_abs and mean_abs must not exceed max_abs"
            )


# This profile is valid only for the exact production block/concurrency and
# runtime contract below. The frozen 60-anchor real-model calibration observed
# zero drift, so math.nextafter(0.0, +inf) is the strictest nonzero envelope.
# Tests exercise explicit profile injection for other block sizes.
PRODUCTION_BATCH_128_CALIBRATION_CONTRACT = BatchCalibrationContract(
    model_name="gemma-3-4b-pt",
    model_dtype="bfloat16",
    activation_type="residual",
    layer=22,
    segment_size=128,
    sae_release="gemma-scope-2-4b-pt-res",
    sae_id="layer_22_width_65k_l0_medium",
    hook_name="blocks.22.hook_resid_post",
)


PRODUCTION_BATCH_128_PROFILE = BatchToleranceProfile(
    name="issue696_gemma3_4bpt_block128_stream2_calibrated_v1",
    batch_size=128,
    selected_sae=NumericTolerance(max_abs=5e-324, p99_abs=5e-324, mean_abs=5e-324),
    raw_residual=NumericTolerance(max_abs=5e-324, p99_abs=5e-324, mean_abs=5e-324),
    calibration_contract=PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
    max_support_mismatches=0,
    runtime_fingerprint_sha256=("6443efc27a7953dfa88fb8dd1bd6043bffcb2508667e893da3f2362c5c76a7d9"),
    calibration_observations_sha256=(
        "9ac3b4e4dbac99b586b9d0ddbe5dd6b44ab7f6c8c2bc183c759ab5ffe3b1e0d9"
    ),
)


def _nvidia_driver_version(device_index: int) -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                f"--id={device_index}",
                "--query-gpu=driver_version",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    values = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return values[0] if result.returncode == 0 and len(values) == 1 else None


def runtime_numeric_fingerprint(
    device: str | t.device | None = None,
) -> dict[str, Any]:
    """Return runtime facts capable of changing CUDA numeric behavior."""
    cuda_available = t.cuda.is_available()
    payload: dict[str, Any] = {
        "torch": t.__version__,
        "transformers": transformers.__version__,
        "sae_lens": importlib.metadata.version("sae-lens"),
        "cuda_runtime": t.version.cuda,
        "cuda_available": cuda_available,
        "float32_matmul_precision": t.get_float32_matmul_precision(),
        "deterministic_algorithms": t.are_deterministic_algorithms_enabled(),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "nvidia_tf32_override": os.environ.get("NVIDIA_TF32_OVERRIDE"),
        "torch_allow_tf32_cublas_override": os.environ.get("TORCH_ALLOW_TF32_CUBLAS_OVERRIDE"),
    }
    if cuda_available:
        resolved_device = t.device("cuda" if device is None else device)
        if resolved_device.type != "cuda":
            raise AutoInterpBatchValidationError(
                f"numeric CUDA fingerprint received non-CUDA device {resolved_device}"
            )
        device_index = (
            t.cuda.current_device() if resolved_device.index is None else resolved_device.index
        )
        try:
            properties = t.cuda.get_device_properties(device_index)
        except (AssertionError, RuntimeError, ValueError) as exc:
            raise AutoInterpBatchValidationError(
                f"cannot fingerprint CUDA device index {device_index}: {exc}"
            ) from exc
        payload.update(
            {
                "gpu_name": properties.name,
                "compute_capability": [properties.major, properties.minor],
                "gpu_total_memory": properties.total_memory,
                "gpu_multiprocessor_count": properties.multi_processor_count,
                "nvidia_driver_version": _nvidia_driver_version(device_index),
                "cudnn_version": t.backends.cudnn.version(),
                "cudnn_enabled": t.backends.cudnn.enabled,
                "cudnn_benchmark": t.backends.cudnn.benchmark,
                "cudnn_deterministic": t.backends.cudnn.deterministic,
                "flash_sdp_enabled": t.backends.cuda.flash_sdp_enabled(),
                "mem_efficient_sdp_enabled": (t.backends.cuda.mem_efficient_sdp_enabled()),
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
                "cudnn_allow_tf32": t.backends.cudnn.allow_tf32,
            }
        )
    return payload


def runtime_numeric_fingerprint_sha256(
    device: str | t.device | None = None,
) -> str:
    encoded = json.dumps(
        runtime_numeric_fingerprint(device),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def assert_runtime_matches_profile(
    profile: BatchToleranceProfile,
    *,
    device: str | t.device | None = None,
) -> None:
    expected = profile.runtime_fingerprint_sha256
    if expected is None:
        return
    actual = runtime_numeric_fingerprint_sha256(device)
    if actual != expected:
        raise AutoInterpBatchValidationError(
            "batched token-example runtime is not covered by its tolerance profile: "
            f"profile={profile.name} expected={expected} actual={actual} "
            f"runtime={runtime_numeric_fingerprint(device)}"
        )


def measure_batch_difference(
    reference: t.Tensor,
    candidate: t.Tensor,
) -> BatchDifferenceMetrics:
    """Measure finite float32 tensors after exact structural validation."""
    if reference.shape != candidate.shape:
        raise AutoInterpBatchValidationError(
            f"batch validation shape mismatch: {reference.shape} != {candidate.shape}"
        )
    if reference.dtype != t.float32 or candidate.dtype != t.float32:
        raise AutoInterpBatchValidationError(
            "batch validation requires CPU/GPU-independent float32 comparison "
            f"surfaces, got {reference.dtype} and {candidate.dtype}"
        )
    if reference.device.type != "cpu" or candidate.device.type != "cpu":
        raise AutoInterpBatchValidationError("batch validation comparison surfaces must be on CPU")
    if reference.numel() == 0:
        raise AutoInterpBatchValidationError("batch validation surface is empty")
    if not bool(t.isfinite(reference).all()) or not bool(t.isfinite(candidate).all()):
        raise AutoInterpBatchValidationError("batch validation surface contains NaN or infinity")
    absolute = (reference - candidate).abs().reshape(-1)
    support_mismatches = int(((reference > 0.0) != (candidate > 0.0)).sum().item())
    return BatchDifferenceMetrics(
        count=int(absolute.numel()),
        support_mismatches=support_mismatches,
        max_abs=float(absolute.max().item()),
        p99_abs=float(t.quantile(absolute, 0.99).item()),
        mean_abs=float(absolute.mean().item()),
    )


def assert_batch_difference_within_profile(
    reference: t.Tensor,
    candidate: t.Tensor,
    *,
    profile: BatchToleranceProfile,
    context: str,
    surface_kind: BatchSurfaceKind | str,
    require_support_bound: bool = True,
) -> BatchDifferenceMetrics:
    """Fail loudly when any independently bounded metric exceeds its profile."""
    metrics = measure_batch_difference(reference, candidate)
    assert_batch_metrics_within_profile(
        metrics,
        profile=profile,
        context=context,
        surface_kind=surface_kind,
        require_support_bound=require_support_bound,
    )
    return metrics


def assert_batch_metrics_within_profile(
    metrics: BatchDifferenceMetrics,
    *,
    profile: BatchToleranceProfile,
    context: str,
    surface_kind: BatchSurfaceKind | str,
    require_support_bound: bool = True,
) -> None:
    """Validate already-measured metrics through the same production gate."""
    if not isinstance(metrics, BatchDifferenceMetrics):
        raise AutoInterpBatchValidationError(
            "batch validation metrics must be BatchDifferenceMetrics"
        )
    surface_kind = BatchSurfaceKind.coerce(surface_kind)
    expected_support_bound = surface_kind is BatchSurfaceKind.SELECTED_SAE
    if require_support_bound is not expected_support_bound:
        raise AutoInterpBatchValidationError(
            f"batch validation surface_kind={surface_kind.value!r} requires "
            f"require_support_bound={expected_support_bound}, got "
            f"{require_support_bound}"
        )
    tolerance = profile.tolerance_for(surface_kind)
    failures: list[str] = []
    if require_support_bound and metrics.support_mismatches > profile.max_support_mismatches:
        failures.append(
            f"support_mismatches={metrics.support_mismatches}>{profile.max_support_mismatches}"
        )
    for field_name in ("max_abs", "p99_abs", "mean_abs"):
        measured = getattr(metrics, field_name)
        allowed = getattr(tolerance, field_name)
        if measured > allowed:
            failures.append(f"{field_name}={measured:.9g}>{allowed:.9g}")
    if failures:
        raise AutoInterpBatchValidationError(
            f"batched token-example numeric validation failed ({context}; "
            f"profile={profile.name}): {', '.join(failures)}"
        )


def assert_candidate_repeat_exact(
    first: t.Tensor,
    second: t.Tensor,
    *,
    context: str,
) -> None:
    if first.shape != second.shape or first.dtype != second.dtype or not t.equal(first, second):
        max_abs = None
        if first.shape == second.shape and first.numel() and second.numel():
            max_abs = float((first.to(t.float32) - second.to(t.float32)).abs().max().item())
        raise AutoInterpBatchValidationError(
            "batched token-example candidate is not repeat-exact "
            f"({context}; first={first.shape}/{first.dtype}, "
            f"second={second.shape}/{second.dtype}, max_abs={max_abs})"
        )
