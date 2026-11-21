"""Test-only calibration of production batch-tolerance profiles."""

import hashlib
import json
import math
from typing import Any

import torch as t

from steering.eval.autointerp_filter.batch_validation import (
    BatchCalibrationContract,
    BatchSurfaceKind,
    BatchToleranceProfile,
    NumericTolerance,
    runtime_numeric_fingerprint_sha256,
)
from steering.eval.autointerp_filter.types import AutoInterpBatchValidationError


def derive_batch_tolerance_profile(
    observations: list[dict[str, Any]],
    *,
    name: str,
    batch_size: int,
    calibration_contract: BatchCalibrationContract,
    device: str | t.device | None = None,
    margin: float = 1.25,
    absolute_floor: float = 0.0,
) -> BatchToleranceProfile:
    """Freeze a profile from calibration observations, never held-out values."""
    if not observations:
        raise AutoInterpBatchValidationError(
            "cannot derive a batch tolerance profile from zero observations"
        )
    if (
        isinstance(margin, bool)
        or not isinstance(margin, (int, float))
        or not math.isfinite(margin)
        or margin < 1.0
    ):
        raise AutoInterpBatchValidationError("calibration margin must be finite and >= 1")
    if (
        isinstance(absolute_floor, bool)
        or not isinstance(absolute_floor, (int, float))
        or not math.isfinite(absolute_floor)
        or absolute_floor < 0.0
    ):
        raise AutoInterpBatchValidationError("calibration absolute_floor must be finite and >= 0")
    required = {
        "target_key",
        "surface",
        "surface_kind",
        "context",
        "require_support_bound",
        "count",
        "support_mismatches",
        "max_abs",
        "p99_abs",
        "mean_abs",
    }
    normalized: list[dict[str, Any]] = []
    saw_selected = False
    saw_residual = False
    surface_kinds_by_target: dict[str, set[BatchSurfaceKind]] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, dict):
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} is not an object"
            )
        missing = sorted(required - set(observation))
        if missing:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} missing fields: {missing}"
            )
        target_key = observation["target_key"]
        surface = observation["surface"]
        raw_surface_kind = observation["surface_kind"]
        context = observation["context"]
        require_support = observation["require_support_bound"]
        count = observation["count"]
        support_mismatches = observation["support_mismatches"]
        if not isinstance(target_key, str) or not target_key:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has invalid target_key"
            )
        if not isinstance(surface, str) or not surface:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has invalid surface"
            )
        try:
            surface_kind = BatchSurfaceKind(raw_surface_kind)
        except (TypeError, ValueError):
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has invalid surface_kind"
            ) from None
        if not surface.startswith(surface_kind):
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} surface={surface!r} does not "
                f"match surface_kind={surface_kind.value!r}"
            )
        if not isinstance(context, str) or not context:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has invalid context"
            )
        if not isinstance(require_support, bool):
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} require_support_bound is not bool"
            )
        expected_support_bound = surface_kind is BatchSurfaceKind.SELECTED_SAE
        if require_support is not expected_support_bound:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has require_support_bound="
                f"{require_support}; expected {expected_support_bound} for "
                f"surface_kind={surface_kind.value!r}"
            )
        if isinstance(count, bool) or not isinstance(count, int) or count < 1:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has invalid count"
            )
        if (
            isinstance(support_mismatches, bool)
            or not isinstance(support_mismatches, int)
            or support_mismatches < 0
            or support_mismatches > count
        ):
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} has invalid support_mismatches"
            )
        if require_support and support_mismatches:
            raise AutoInterpBatchValidationError(
                "calibration observed an active-support mismatch on a support-bound "
                f"surface ({context}): {support_mismatches}"
            )
        numeric: dict[str, float] = {}
        for field_name in ("max_abs", "p99_abs", "mean_abs"):
            value = observation[field_name]
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0.0
            ):
                raise AutoInterpBatchValidationError(
                    f"calibration observation {index} has invalid {field_name}"
                )
            numeric[field_name] = float(value)
        if numeric["p99_abs"] > numeric["max_abs"] or numeric["mean_abs"] > numeric["max_abs"]:
            raise AutoInterpBatchValidationError(
                f"calibration observation {index} p99_abs/mean_abs exceeds max_abs"
            )
        saw_selected |= surface_kind is BatchSurfaceKind.SELECTED_SAE
        saw_residual |= surface_kind is BatchSurfaceKind.RAW_RESIDUAL
        surface_kinds_by_target.setdefault(target_key, set()).add(surface_kind)
        normalized.append(
            {
                "target_key": target_key,
                "surface": surface,
                "surface_kind": surface_kind,
                "context": context,
                "require_support_bound": require_support,
                "count": count,
                "support_mismatches": support_mismatches,
                **numeric,
            }
        )
    if not saw_selected or not saw_residual:
        raise AutoInterpBatchValidationError(
            "calibration requires both selected-SAE and raw-residual observations"
        )
    required_surface_kinds = set(BatchSurfaceKind)
    incomplete_targets = {
        target_key: sorted(required_surface_kinds - surface_kinds)
        for target_key, surface_kinds in sorted(surface_kinds_by_target.items())
        if surface_kinds != required_surface_kinds
    }
    if incomplete_targets:
        raise AutoInterpBatchValidationError(
            "calibration requires selected-SAE and raw-residual observations for "
            f"every target; missing={incomplete_targets}"
        )

    def bound(surface_kind: BatchSurfaceKind, field_name: str) -> float:
        observed = max(
            float(row[field_name]) for row in normalized if row["surface_kind"] == surface_kind
        )
        return math.nextafter(max(float(absolute_floor), observed * float(margin)), math.inf)

    def tolerance(surface_kind: BatchSurfaceKind) -> NumericTolerance:
        return NumericTolerance(
            max_abs=bound(surface_kind, "max_abs"),
            p99_abs=bound(surface_kind, "p99_abs"),
            mean_abs=bound(surface_kind, "mean_abs"),
        )

    encoded = json.dumps(normalized, sort_keys=True, separators=(",", ":"))
    return BatchToleranceProfile(
        name=name,
        batch_size=batch_size,
        selected_sae=tolerance(BatchSurfaceKind.SELECTED_SAE),
        raw_residual=tolerance(BatchSurfaceKind.RAW_RESIDUAL),
        calibration_contract=calibration_contract,
        max_support_mismatches=0,
        runtime_fingerprint_sha256=runtime_numeric_fingerprint_sha256(device),
        calibration_observations_sha256=hashlib.sha256(encoded.encode("utf-8")).hexdigest(),
    )
