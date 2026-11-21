"""DiffMean and project-convention linear-probe direction fits."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression

from nanochat.atomic_io import atomic_write_bytes
from steering.eval.axbench_base.direction_scale import (
    compute_direction_max_activation,
    compute_signed_direction_max_activation,
)
from steering.safetensors_io import (
    deterministic_safetensors_bytes,
    load_safetensors_file_snapshot,
)

UNIT_NORM_EPS = float(t.finfo(t.float32).eps)

PROBE_C = 1.0
PROBE_MAX_ITER = 500
PROBE_SEED_DOMAIN = "axbench-upstream-encdec-probe-v1"
PROBE_SOLVER = "liblinear"
_PROBE_L1_RATIO = 1.0
_SEED_MODULUS = 2**31 - 1
_DIRECTION_FORMAT = "axbench-encdec-direction-v1"


@dataclass(frozen=True)
class FittedDirection:
    """One fitted fp32 steering direction and its fit-row calibration."""

    direction: t.Tensor
    max_activation: float
    metadata: dict[str, Any]


def unit_norm_vendored(raw: t.Tensor) -> t.Tensor:
    """Apply AxBench's additive-epsilon unit normalization exactly."""

    return raw / (raw.norm() + UNIT_NORM_EPS)


def _validated_rows(
    positive: Sequence[t.Tensor],
    negative: Sequence[t.Tensor],
    *,
    surface: str,
) -> tuple[list[t.Tensor], list[t.Tensor]]:
    if not positive:
        raise ValueError(f"{surface}: positive activations must not be empty")
    if not negative:
        raise ValueError(f"{surface}: negative activations must not be empty")

    width: int | None = None
    validated: list[list[t.Tensor]] = []
    for class_name, rows in (("positive", positive), ("negative", negative)):
        class_rows: list[t.Tensor] = []
        for row_index, row in enumerate(rows):
            if row.ndim != 2:
                raise ValueError(
                    f"{surface}: {class_name} row {row_index} must be a matrix, "
                    f"got shape {tuple(row.shape)}"
                )
            if row.shape[0] == 0:
                raise ValueError(
                    f"{surface}: {class_name} row {row_index} has no token activations"
                )
            if row.shape[1] == 0:
                raise ValueError(f"{surface}: {class_name} row {row_index} has zero width")
            if width is None:
                width = int(row.shape[1])
            elif row.shape[1] != width:
                raise ValueError(
                    f"{surface}: {class_name} row {row_index} width {row.shape[1]} "
                    f"does not match width {width}"
                )
            row_f32 = row.detach().to(dtype=t.float32, device="cpu")
            if not bool(t.isfinite(row_f32).all().item()):
                raise ValueError(
                    f"{surface}: {class_name} row {row_index} contains non-finite activations"
                )
            class_rows.append(row_f32)
        validated.append(class_rows)
    return validated[0], validated[1]


def fit_diffmean_direction(
    pos_acts: Sequence[t.Tensor],
    neg_acts: Sequence[t.Tensor],
    *,
    intervention_layer: int,
    surface: str,
    calibrate: bool = True,
    calibration_only: bool = False,
) -> FittedDirection:
    """Fit the token-weighted vendored DiffMean direction in fp32.

    ``calibration_only`` records a signed statistic for raw counters; the
    default requires a valid positive scale for nonzero directions.
    """

    positive, negative = _validated_rows(pos_acts, neg_acts, surface=surface)
    raw = t.cat(positive, dim=0).mean(dim=0) - t.cat(negative, dim=0).mean(dim=0)
    direction = unit_norm_vendored(raw)
    # Raw additive counters record calibration but never multiply by it.
    max_activation = (
        (compute_signed_direction_max_activation if calibration_only else compute_direction_max_activation)(
            [*positive, *negative], direction, surface=surface,
        )
        if calibrate else 0.0
    )
    return FittedDirection(
        direction=direction,
        max_activation=max_activation,
        metadata={
            "method": "diffmean",
            "intervention_layer": intervention_layer,
            "n_positive_rows": len(positive),
            "n_negative_rows": len(negative),
        },
    )


def _derive_probe_seed(seed: int, concept_id: int) -> int:
    fields = (PROBE_SEED_DOMAIN, str(seed), str(concept_id))
    digest = hashlib.sha256("\x1f".join(fields).encode()).digest()
    return int.from_bytes(digest[:8], "big") % _SEED_MODULUS


def fit_probe_direction(
    pos_acts: Sequence[t.Tensor],
    neg_acts: Sequence[t.Tensor],
    *,
    intervention_layer: int,
    seed: int,
    surface: str,
    concept_id: int = 0,
    calibrate: bool = True,
    calibration_only: bool = False,
) -> FittedDirection:
    """Fit the project row-mean liblinear probe on cached activations.

    ``calibration_only`` has the same raw-counter meaning as in DiffMean.
    """

    positive, negative = _validated_rows(pos_acts, neg_acts, surface=surface)
    rows = [*positive, *negative]
    row_means = t.stack([row.mean(dim=0) for row in rows])
    labels = t.tensor(
        [1] * len(positive) + [0] * len(negative),
        dtype=t.int64,
    )
    derived_seed = _derive_probe_seed(seed, concept_id)
    estimator = LogisticRegression(
        # Match axbench_base's _PROBE_L1_RATIO convention for true L1 coefficients.
        l1_ratio=_PROBE_L1_RATIO,
        C=PROBE_C,
        max_iter=PROBE_MAX_ITER,
        solver=PROBE_SOLVER,
        random_state=derived_seed,
        class_weight=None,
    ).fit(row_means.numpy(), labels.numpy())
    coefficient = t.as_tensor(estimator.coef_[0], dtype=t.float32)
    direction = F.normalize(coefficient, dim=0)
    # Raw additive counters record calibration but never multiply by it.
    max_activation = (
        (compute_signed_direction_max_activation if calibration_only else compute_direction_max_activation)(
            rows, direction, surface=surface,
        )
        if calibrate else 0.0
    )
    return FittedDirection(
        direction=direction,
        max_activation=max_activation,
        metadata={
            "method": "probe",
            "intervention_layer": intervention_layer,
            "n_positive_rows": len(positive),
            "n_negative_rows": len(negative),
            "seed": seed,
            "concept_id": concept_id,
            "seed_domain": PROBE_SEED_DOMAIN,
            "estimator": {
                "name": "sklearn.linear_model.LogisticRegression",
                "pooling": "per_row_mean_post_prefix_activations",
                "solver": PROBE_SOLVER,
                "l1_ratio": "1.0 (L1)",
                "C": PROBE_C,
                "max_iter": PROBE_MAX_ITER,
                "random_state": derived_seed,
                "class_weight": None,
                "direction": "F.normalize(coef_[0])",
            },
        },
    )


def save_direction(path: Path, fitted: FittedDirection) -> None:
    """Save one fitted direction with exact-round-trip text metadata."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    direction = fitted.direction.detach().to(dtype=t.float32, device="cpu").contiguous()
    metadata = {
        "format": _DIRECTION_FORMAT,
        "max_activation": repr(float(fitted.max_activation)),
        "metadata_json": json.dumps(
            fitted.metadata,
            sort_keys=True,
            separators=(",", ":"),
        ),
    }
    atomic_write_bytes(
        deterministic_safetensors_bytes({"direction": direction}, metadata=metadata),
        path,
    )


def load_direction(path: Path) -> FittedDirection:
    """Load one direction artifact written by :func:`save_direction`."""

    path = Path(path)
    tensors, loaded_metadata = load_safetensors_file_snapshot(path)
    metadata = loaded_metadata or {}
    if metadata.get("format") != _DIRECTION_FORMAT:
        raise ValueError(f"{path}: invalid or missing direction artifact format")
    if "max_activation" not in metadata:
        raise ValueError(f"{path}: missing max_activation metadata")
    if "metadata_json" not in metadata:
        raise ValueError(f"{path}: missing metadata_json metadata")
    expected_metadata_keys = {"format", "max_activation", "metadata_json"}
    unexpected_metadata = sorted(set(metadata) - expected_metadata_keys)
    if unexpected_metadata:
        raise ValueError(f"{path}: unexpected direction metadata keys: {unexpected_metadata}")
    if set(tensors) != {"direction"}:
        raise ValueError(f"{path}: direction artifact must contain only 'direction'")
    direction = tensors["direction"]
    if (
        direction.dtype is not t.float32
        or direction.device.type != "cpu"
        or direction.ndim != 1
        or direction.numel() == 0
        or not direction.is_contiguous()
        or not bool(t.isfinite(direction).all())
    ):
        raise ValueError(
            f"{path}: direction must be a finite, nonempty, contiguous CPU float32 vector"
        )
    try:
        max_activation = float(metadata["max_activation"])
    except ValueError as exc:
        raise ValueError(f"{path}: max_activation metadata is not a float") from exc
    if not math.isfinite(max_activation) or max_activation < 0.0:
        raise ValueError(f"{path}: max_activation must be finite and nonnegative")

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
        fit_metadata = json.loads(
            metadata["metadata_json"],
            parse_constant=reject_constant,
            object_pairs_hook=reject_duplicate_keys,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise ValueError(f"{path}: metadata_json is not strict JSON: {exc}") from exc
    if not isinstance(fit_metadata, dict):
        raise ValueError(f"{path}: metadata_json must contain an object")
    return FittedDirection(
        direction=direction,
        max_activation=max_activation,
        metadata=fit_metadata,
    )


__all__ = (
    "PROBE_C",
    "PROBE_MAX_ITER",
    "PROBE_SEED_DOMAIN",
    "PROBE_SOLVER",
    "UNIT_NORM_EPS",
    "FittedDirection",
    "fit_diffmean_direction",
    "fit_probe_direction",
    "load_direction",
    "save_direction",
    "unit_norm_vendored",
)
