"""Versioned Stage 1b positive-corpus QC provenance and invariants."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq

from steering._common.strict_json import strict_json_object
from steering.eval.axbench_base.percentile_units import (
    PERCENTILE_POINTS_PER_FRACTION,
    percentile_delta_meets_or_exceeds,
    percentile_fraction_to_points,
)
from steering.eval.axbench_base.positive_selection_contract import (
    schema_with_default_positive_selection_contract,
)
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
)

POSITIVE_QC_CONTRACT_METADATA_KEY = "axbench_stage1b_positive_qc_contract"
POSITIVE_QC_CONTRACT_SCHEMA_VERSION = 1
POSITIVE_QC_PERCENTILE_UNIT = (
    "raw_unquantized_percentile_points_zero_or_0_5_to_99_5"
)
POSITIVE_QC_COMPARISON = "output_minus_seed_greater_than_or_equal"
POSITIVE_QC_ALGORITHM = "stage1b_delta_ge_fraction_times_100_v1"

_CONTRACT_KEYS = frozenset(
    {
        "schema_version",
        "accept_threshold_fraction",
        "accept_threshold_points",
        "percentile_unit",
        "comparison",
        "algorithm",
    }
)


@dataclass(frozen=True)
class PositiveQcContract:
    """Parsed and validated positive-corpus QC contract."""

    schema_version: int
    accept_threshold_fraction: float
    accept_threshold_points: float
    percentile_unit: str
    comparison: str
    algorithm: str


def _contract_for_threshold(accept_threshold: object) -> PositiveQcContract:
    threshold_points = percentile_fraction_to_points(
        accept_threshold,
        field_name="accept_threshold",
    )
    return PositiveQcContract(
        schema_version=POSITIVE_QC_CONTRACT_SCHEMA_VERSION,
        accept_threshold_fraction=(
            threshold_points / PERCENTILE_POINTS_PER_FRACTION
        ),
        accept_threshold_points=threshold_points,
        percentile_unit=POSITIVE_QC_PERCENTILE_UNIT,
        comparison=POSITIVE_QC_COMPARISON,
        algorithm=POSITIVE_QC_ALGORITHM,
    )


def _contract_payload(contract: PositiveQcContract) -> dict[str, object]:
    return {
        "schema_version": contract.schema_version,
        "accept_threshold_fraction": contract.accept_threshold_fraction,
        "accept_threshold_points": contract.accept_threshold_points,
        "percentile_unit": contract.percentile_unit,
        "comparison": contract.comparison,
        "algorithm": contract.algorithm,
    }


def schema_with_positive_qc_contract(
    schema: pa.Schema,
    *,
    accept_threshold: object,
) -> pa.Schema:
    """Return ``schema`` with a canonical Stage 1b positive-QC contract."""

    contract = _contract_for_threshold(accept_threshold)
    encoded = json.dumps(
        _contract_payload(contract),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    metadata = dict(schema.metadata or {})
    metadata[POSITIVE_QC_CONTRACT_METADATA_KEY.encode("utf-8")] = encoded
    schema = schema.with_metadata(metadata)
    return schema_with_default_positive_selection_contract(
        schema,
        accept_threshold=accept_threshold,
    )


def _contract_error(*, consumer: str, path: Path, detail: str) -> AxBenchBaseError:
    return AxBenchBaseError(
        f"{consumer}: positive synth corpus at {path} has an invalid or legacy "
        f"Stage 1b QC contract: {detail}. Regenerate Stage 1b under a fresh "
        "run_tag with the current code."
    )


def _metadata_text(raw: object, *, consumer: str, path: Path) -> str:
    if isinstance(raw, bytes):
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise _contract_error(
                consumer=consumer,
                path=path,
                detail="contract metadata is not valid UTF-8",
            ) from exc
    if isinstance(raw, str):
        return raw
    raise _contract_error(
        consumer=consumer,
        path=path,
        detail=f"contract metadata has unsupported type {type(raw).__name__}",
    )


def _parse_contract(
    raw: object,
    *,
    consumer: str,
    path: Path,
) -> PositiveQcContract:
    try:
        payload = json.loads(
            _metadata_text(raw, consumer=consumer, path=path),
            object_pairs_hook=strict_json_object,
        )
    except (json.JSONDecodeError, ValueError) as exc:
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=f"contract metadata is not valid JSON ({exc})",
        ) from exc
    if not isinstance(payload, dict):
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail="contract JSON must be an object",
        )
    payload = cast(dict[str, Any], payload)
    keys = set(payload)
    if keys != _CONTRACT_KEYS:
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=(
                f"contract fields differ: missing={sorted(_CONTRACT_KEYS - keys)}, "
                f"unexpected={sorted(keys - _CONTRACT_KEYS)}"
            ),
        )
    schema_version = payload["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != POSITIVE_QC_CONTRACT_SCHEMA_VERSION
    ):
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=(
                f"schema_version must be the strict integer "
                f"{POSITIVE_QC_CONTRACT_SCHEMA_VERSION}, got {schema_version!r}"
            ),
        )

    fraction = payload["accept_threshold_fraction"]
    try:
        resolved_points = percentile_fraction_to_points(
            fraction,
            field_name="accept_threshold_fraction",
        )
    except AxBenchBaseError as exc:
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=str(exc),
        ) from exc
    points = payload["accept_threshold_points"]
    if isinstance(points, bool) or not isinstance(points, Real):
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=f"accept_threshold_points must be a finite real, got {points!r}",
        )
    points_float = float(points)
    if not math.isfinite(points_float) or points_float != resolved_points:
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=(
                "accept_threshold_fraction and accept_threshold_points disagree: "
                f"{fraction!r} resolves to {resolved_points!r}, got {points!r}"
            ),
        )
    for key, expected in (
        ("percentile_unit", POSITIVE_QC_PERCENTILE_UNIT),
        ("comparison", POSITIVE_QC_COMPARISON),
        ("algorithm", POSITIVE_QC_ALGORITHM),
    ):
        if payload[key] != expected:
            raise _contract_error(
                consumer=consumer,
                path=path,
                detail=f"{key} must be {expected!r}, got {payload[key]!r}",
            )
    return PositiveQcContract(
        schema_version=schema_version,
        accept_threshold_fraction=float(fraction),
        accept_threshold_points=points_float,
        percentile_unit=POSITIVE_QC_PERCENTILE_UNIT,
        comparison=POSITIVE_QC_COMPARISON,
        algorithm=POSITIVE_QC_ALGORITHM,
    )


def validate_positive_qc_contract(
    path: Path,
    *,
    consumer: str,
    schema: pa.Schema | None = None,
    expected_accept_threshold: object | None = None,
) -> PositiveQcContract:
    """Validate a positive synth Parquet's versioned QC contract."""

    if schema is None:
        try:
            parquet_schema = pq.read_schema(path)
        except (OSError, pa.ArrowException) as exc:
            raise _contract_error(
                consumer=consumer,
                path=path,
                detail=f"Parquet schema cannot be read ({exc})",
            ) from exc
    else:
        parquet_schema = schema
    metadata = parquet_schema.metadata or {}
    raw = metadata.get(POSITIVE_QC_CONTRACT_METADATA_KEY.encode("utf-8"))
    if raw is None:
        raise _contract_error(
            consumer=consumer,
            path=path,
            detail=f"missing {POSITIVE_QC_CONTRACT_METADATA_KEY!r} metadata",
        )
    contract = _parse_contract(raw, consumer=consumer, path=path)
    if expected_accept_threshold is not None:
        expected = _contract_for_threshold(expected_accept_threshold)
        if (
            contract.accept_threshold_fraction
            != expected.accept_threshold_fraction
            or contract.accept_threshold_points != expected.accept_threshold_points
        ):
            raise _contract_error(
                consumer=consumer,
                path=path,
                detail=(
                    "artifact threshold does not match the current config: "
                    f"artifact fraction={contract.accept_threshold_fraction!r} "
                    f"({contract.accept_threshold_points!r} points), current "
                    f"fraction={expected.accept_threshold_fraction!r} "
                    f"({expected.accept_threshold_points!r} points)"
                ),
            )
    return contract


def _finite_row_number(
    row: Mapping[str, Any],
    key: str,
    *,
    row_index: int,
) -> float:
    value = row.get(key)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise AxBenchBaseError(
            f"Stage 1b positive row {row_index} field {key!r} must be a finite "
            f"real number, got {value!r}"
        )
    result = float(value)
    if not math.isfinite(result):
        raise AxBenchBaseError(
            f"Stage 1b positive row {row_index} field {key!r} must be finite, "
            f"got {value!r}"
        )
    return result


def _raw_qc_percentile(
    row: Mapping[str, Any],
    key: str,
    *,
    row_index: int,
) -> float:
    value = _finite_row_number(row, key, row_index=row_index)
    if value != 0.0 and not 0.5 <= value <= 99.5:
        raise AxBenchBaseError(
            f"Stage 1b positive row {row_index} field {key!r} must be the "
            "strict-zero sentinel 0.0 or a raw unquantized percentile in "
            f"[0.5, 99.5], got {value!r}"
        )
    return value


def validate_positive_qc_rows(
    rows: list[dict[str, Any]],
    *,
    accept_threshold_points: object,
) -> None:
    """Validate full-precision Stage 1b rows before Arrow float32 casts."""

    if (
        isinstance(accept_threshold_points, bool)
        or not isinstance(accept_threshold_points, Real)
    ):
        raise AxBenchBaseError(
            "Stage 1b positive writer received a resolved threshold that is "
            f"not a real number: {accept_threshold_points!r}"
        )
    resolved_threshold_points = float(accept_threshold_points)
    if not math.isfinite(resolved_threshold_points):
        raise AxBenchBaseError(
            "Stage 1b positive writer received a non-finite resolved threshold"
        )
    if not 0.0 <= resolved_threshold_points <= PERCENTILE_POINTS_PER_FRACTION:
        raise AxBenchBaseError(
            "Stage 1b positive writer received a resolved threshold outside "
            f"[0, {PERCENTILE_POINTS_PER_FRACTION:g}]: "
            f"{resolved_threshold_points!r}"
        )
    for row_index, row in enumerate(rows):
        content_class = row.get("content_class")
        accepted = row.get("qc_accepted")
        if not isinstance(accepted, bool):
            raise AxBenchBaseError(
                f"Stage 1b positive row {row_index} qc_accepted must be a bool, "
                f"got {accepted!r}"
            )
        if content_class == CONCEPT_RICH_LABEL:
            seed = _raw_qc_percentile(
                row,
                "qc_seed_percentile",
                row_index=row_index,
            )
            output = _raw_qc_percentile(
                row,
                "qc_output_percentile",
                row_index=row_index,
            )
            delta = _finite_row_number(row, "qc_delta", row_index=row_index)
            expected_delta = output - seed
            if delta != expected_delta:
                raise AxBenchBaseError(
                    f"Stage 1b positive row {row_index} qc_delta={delta!r} does "
                    f"not exactly equal qc_output_percentile-qc_seed_percentile="
                    f"{expected_delta!r}"
                )
            passes = percentile_delta_meets_or_exceeds(
                delta,
                resolved_threshold_points,
            )
            if accepted != passes:
                raise AxBenchBaseError(
                    f"Stage 1b positive row {row_index} qc_accepted={accepted!r} "
                    f"disagrees with delta={delta!r} >= "
                    f"accept_threshold_points={resolved_threshold_points!r}"
                )
            if row.get("qc_zero_activation_verified") is not False:
                raise AxBenchBaseError(
                    f"Stage 1b positive rich row {row_index} must set "
                    "qc_zero_activation_verified=False"
                )
        elif content_class == CONCEPT_AVOIDING_LABEL:
            for key in (
                "qc_seed_percentile",
                "qc_output_percentile",
                "qc_delta",
            ):
                value = row.get(key)
                if (
                    isinstance(value, bool)
                    or not isinstance(value, Real)
                    or not math.isnan(float(value))
                ):
                    raise AxBenchBaseError(
                        f"Stage 1b positive avoiding row {row_index} field "
                        f"{key!r} must be NaN, got {value!r}"
                    )
            if not accepted or row.get("qc_zero_activation_verified") is not True:
                raise AxBenchBaseError(
                    f"Stage 1b positive avoiding row {row_index} must set "
                    "qc_accepted=True and qc_zero_activation_verified=True"
                )
        else:
            raise AxBenchBaseError(
                f"Stage 1b positive row {row_index} has unknown content_class "
                f"{content_class!r}"
            )


__all__ = [
    "POSITIVE_QC_ALGORITHM",
    "POSITIVE_QC_COMPARISON",
    "POSITIVE_QC_CONTRACT_METADATA_KEY",
    "POSITIVE_QC_CONTRACT_SCHEMA_VERSION",
    "POSITIVE_QC_PERCENTILE_UNIT",
    "PositiveQcContract",
    "schema_with_positive_qc_contract",
    "validate_positive_qc_contract",
    "validate_positive_qc_rows",
]
