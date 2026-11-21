"""Fail-closed Stage 1b positive-corpus contract tests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from steering.eval.axbench_base.positive_qc_contract import (
    POSITIVE_QC_ALGORITHM,
    POSITIVE_QC_COMPARISON,
    POSITIVE_QC_CONTRACT_METADATA_KEY,
    POSITIVE_QC_CONTRACT_SCHEMA_VERSION,
    POSITIVE_QC_PERCENTILE_UNIT,
    schema_with_positive_qc_contract,
    validate_positive_qc_contract,
    validate_positive_qc_rows,
)
from steering.eval.axbench_base.positive_selection_contract import (
    validate_positive_selection_contract,
)
from steering.eval.axbench_base.synth import positive_mode
from steering.eval.axbench_base.synth_continuation_contract import (
    validate_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    CONCEPT_AVOIDING_LABEL,
    CONCEPT_RICH_LABEL,
    AxBenchBaseError,
)
from tests.axbench_base.unit.test_config import make_valid_config

_PATH = Path("/scientific/source/synth_training_positive.parquet")
_KEY = POSITIVE_QC_CONTRACT_METADATA_KEY.encode("utf-8")


def _payload(**overrides: object) -> dict[str, object]:
    result: dict[str, object] = {
        "schema_version": POSITIVE_QC_CONTRACT_SCHEMA_VERSION,
        "accept_threshold_fraction": 0.3,
        "accept_threshold_points": 30.0,
        "percentile_unit": POSITIVE_QC_PERCENTILE_UNIT,
        "comparison": POSITIVE_QC_COMPARISON,
        "algorithm": POSITIVE_QC_ALGORITHM,
    }
    result.update(overrides)
    return result


def _schema_with_raw_metadata(raw: object) -> pa.Schema:
    return pa.schema(
        [("value", pa.int32())],
        metadata={
            b"unrelated": b"preserved",
            _KEY: raw,
        },
    )


def _schema_with_payload(**overrides: object) -> pa.Schema:
    return _schema_with_raw_metadata(
        json.dumps(
            _payload(**overrides),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=True,
        ).encode("utf-8")
    )


def _rich_row(
    *,
    seed: object = 10.0,
    output: object = 40.0,
    delta: object = 30.0,
    accepted: object = True,
    zero_verified: object = False,
) -> dict[str, Any]:
    return {
        "content_class": CONCEPT_RICH_LABEL,
        "qc_seed_percentile": seed,
        "qc_output_percentile": output,
        "qc_delta": delta,
        "qc_accepted": accepted,
        "qc_zero_activation_verified": zero_verified,
    }


def _avoiding_row(
    *,
    seed: object = math.nan,
    output: object = math.nan,
    delta: object = math.nan,
    accepted: object = True,
    zero_verified: object = True,
) -> dict[str, Any]:
    return {
        "content_class": CONCEPT_AVOIDING_LABEL,
        "qc_seed_percentile": seed,
        "qc_output_percentile": output,
        "qc_delta": delta,
        "qc_accepted": accepted,
        "qc_zero_activation_verified": zero_verified,
    }


def _full_training_rich_row(
    *,
    output: float = 40.0,
    delta: float = 30.0,
    accepted: bool = True,
) -> dict[str, Any]:
    return {
        "concept_id": 7,
        "category": "positive",
        "content_class": CONCEPT_RICH_LABEL,
        "prefix_text": "prefix",
        "output_text": "continuation",
        "output_concept": "concept",
        "n_prefix_nanochat_tokens": 8,
        "gen_source": "test",
        "gen_seed": 11,
        "qc_feature_id": 42,
        "qc_seed_percentile": 10.0,
        "qc_output_percentile": output,
        "qc_delta": delta,
        "qc_attempts": 1,
        "qc_accepted": accepted,
        "qc_zero_activation_verified": False,
    }


def test_schema_writer_emits_canonical_exact_contract_and_preserves_metadata() -> None:
    base = pa.schema(
        [("value", pa.int32())],
        metadata={b"unrelated": b"preserved"},
    )
    schema = schema_with_positive_qc_contract(
        base,
        accept_threshold=0.3,
    )
    assert schema.metadata is not None
    assert schema.metadata[b"unrelated"] == b"preserved"
    assert schema.metadata[_KEY] == json.dumps(
        _payload(),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def test_valid_contract_round_trips_from_schema() -> None:
    contract = validate_positive_qc_contract(
        _PATH,
        consumer="unit consumer",
        schema=_schema_with_payload(),
        expected_accept_threshold=0.3,
    )
    assert contract.schema_version == 1
    assert contract.accept_threshold_fraction == 0.3
    assert contract.accept_threshold_points == 30.0
    assert contract.percentile_unit == POSITIVE_QC_PERCENTILE_UNIT
    assert contract.comparison == POSITIVE_QC_COMPARISON
    assert contract.algorithm == POSITIVE_QC_ALGORITHM


def test_valid_contract_accepts_string_metadata_without_config_comparison() -> None:
    class _Schema:
        metadata = {
            _KEY: json.dumps(
                _payload(),
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        }

    contract = validate_positive_qc_contract(
        _PATH,
        consumer="in-memory schema consumer",
        schema=_Schema(),  # type: ignore[arg-type]
    )

    assert contract.accept_threshold_fraction == 0.3
    assert contract.accept_threshold_points == 30.0


def test_valid_contract_round_trips_from_real_parquet(tmp_path: Path) -> None:
    path = tmp_path / "positive.parquet"
    schema = schema_with_positive_qc_contract(
        pa.schema([("value", pa.int32())]),
        accept_threshold=0.3,
    )
    pq.write_table(pa.Table.from_pylist([{"value": 1}], schema=schema), path)
    contract = validate_positive_qc_contract(
        path,
        consumer="real parquet test",
        expected_accept_threshold=0.3,
    )
    assert contract.accept_threshold_points == 30.0
    assert pq.read_schema(path).metadata == schema.metadata


def test_production_writer_stamps_contract_and_preserves_float32_schema(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positive.parquet"
    cfg = make_valid_config(accept_threshold=0.3)
    positive_mode._write_training_parquet(
        [_full_training_rich_row()],
        path,
        tokenizer_id="test-tokenizer",
        cfg=cfg,
    )

    contract = validate_positive_qc_contract(
        path,
        consumer="writer round trip",
        expected_accept_threshold=0.3,
    )
    assert contract.accept_threshold_fraction == 0.3
    assert contract.accept_threshold_points == 30.0
    schema = pq.read_schema(path)
    assert schema.field("qc_seed_percentile").type == pa.float32()
    assert schema.field("qc_output_percentile").type == pa.float32()
    assert schema.field("qc_delta").type == pa.float32()
    validate_positive_selection_contract(
        path,
        consumer="writer round trip",
        expected_cfg=cfg,
    )
    validate_stage1_continuation_contract(
        path,
        consumer="writer round trip",
        expected_role="stage1b_positive",
    )


def test_production_writer_rejects_bad_decision_before_creating_parquet(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positive.parquet"
    with pytest.raises(
        AxBenchBaseError,
        match=r"qc_accepted=True disagrees with delta=0.5 >=.*30.0",
    ):
        positive_mode._write_training_parquet(
            [
                _full_training_rich_row(
                    output=10.5,
                    delta=0.5,
                    accepted=True,
                )
            ],
            path,
            tokenizer_id="test-tokenizer",
            cfg=make_valid_config(accept_threshold=0.3),
        )
    assert not path.exists()


def test_unreadable_parquet_rejects_with_contract_specific_error(
    tmp_path: Path,
) -> None:
    path = tmp_path / "positive.parquet"
    path.write_bytes(b"not a parquet file")
    with pytest.raises(
        AxBenchBaseError,
        match=r"real consumer.*Parquet schema cannot be read.*fresh run_tag",
    ):
        validate_positive_qc_contract(
            path,
            consumer="real consumer",
        )


def test_missing_contract_rejects_legacy_positive_corpus() -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=(
            r"unit consumer.*invalid or legacy.*missing "
            r"'axbench_stage1b_positive_qc_contract'.*fresh run_tag"
        ),
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=pa.schema([("value", pa.int32())]),
        )


@pytest.mark.parametrize(
    "raw",
    [
        b"\xff",
        b"{not-json",
        b"[]",
        b"null",
        b'"string"',
        b"7",
    ],
)
def test_contract_rejects_noncanonical_metadata_envelopes(raw: bytes) -> None:
    with pytest.raises(AxBenchBaseError, match="invalid or legacy"):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_schema_with_raw_metadata(raw),
        )


@pytest.mark.parametrize(
    "duplicate_field",
    [
        '"schema_version":0,"schema_version":1',
        '"accept_threshold_fraction":0.5,"accept_threshold_fraction":0.3',
        '"algorithm":"legacy","algorithm":"stage1b_delta_ge_fraction_times_100_v1"',
    ],
)
def test_contract_rejects_duplicate_json_object_keys(
    duplicate_field: str,
) -> None:
    other_fields = {
        key: value
        for key, value in _payload().items()
        if f'"{key}"' not in duplicate_field
    }
    raw = (
        "{"
        + duplicate_field
        + ","
        + json.dumps(
            other_fields,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )[1:]
    ).encode("utf-8")

    with pytest.raises(
        AxBenchBaseError,
        match=r"not valid JSON.*duplicate JSON object key",
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_schema_with_raw_metadata(raw),
        )


def test_contract_rejects_unsupported_metadata_runtime_type() -> None:
    class _Schema:
        metadata = {_KEY: object()}

    with pytest.raises(
        AxBenchBaseError,
        match="unsupported type object",
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_Schema(),  # type: ignore[arg-type]
        )


@pytest.mark.parametrize(
    "mutator",
    [
        lambda payload: payload.pop("algorithm"),
        lambda payload: payload.__setitem__("unexpected", "field"),
    ],
)
def test_contract_rejects_missing_or_unexpected_fields(mutator) -> None:
    payload = _payload()
    mutator(payload)
    schema = _schema_with_raw_metadata(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    )
    with pytest.raises(AxBenchBaseError, match="contract fields differ"):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=schema,
        )


@pytest.mark.parametrize("version", [True, False, 0, 2, 1.0, "1", None])
def test_contract_rejects_nonexact_schema_version(version: object) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"schema_version must be the strict integer 1",
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_schema_with_payload(schema_version=version),
        )


@pytest.mark.parametrize(
    "fraction",
    [
        True,
        False,
        None,
        "0.3",
        math.nan,
        math.inf,
        -math.inf,
        -0.1,
        1.1,
    ],
)
def test_contract_rejects_invalid_fraction(fraction: object) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"accept_threshold_fraction.*finite real percentile fraction",
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_schema_with_payload(
                accept_threshold_fraction=fraction,
            ),
        )


@pytest.mark.parametrize(
    "points",
    [
        True,
        False,
        None,
        "30",
        math.nan,
        math.inf,
        -math.inf,
        29.999,
        30.001,
        -30.0,
    ],
)
def test_contract_rejects_invalid_or_inconsistent_points(points: object) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=(
            "accept_threshold_points must be a finite real"
            "|accept_threshold_fraction and accept_threshold_points disagree"
        ),
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_schema_with_payload(accept_threshold_points=points),
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    [
        ("percentile_unit", "normalized_0_1"),
        ("percentile_unit", None),
        ("comparison", "greater_than"),
        ("comparison", None),
        ("algorithm", "legacy_unscaled"),
        ("algorithm", None),
    ],
)
def test_contract_rejects_semantic_identity_drift(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=rf"{field_name} must be",
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="unit consumer",
            schema=_schema_with_payload(**{field_name: bad_value}),
        )


def test_contract_rejects_current_config_threshold_mismatch() -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=(
            r"artifact threshold does not match the current config.*"
            r"artifact fraction=0.3.*current fraction=0.5"
        ),
    ):
        validate_positive_qc_contract(
            _PATH,
            consumer="Stage 3",
            schema=_schema_with_payload(),
            expected_accept_threshold=0.5,
        )


@pytest.mark.parametrize(
    ("delta", "accepted"),
    [
        (29.999999, False),
        (30.0, True),
        (30.000001, True),
    ],
)
def test_full_precision_rich_row_boundary_contract(
    delta: float,
    accepted: bool,
) -> None:
    validate_positive_qc_rows(
        [
            _rich_row(
                seed=0.0,
                output=delta,
                delta=delta,
                accepted=accepted,
            )
        ],
        accept_threshold_points=30.0,
    )


def test_half_point_accepted_row_is_rejected_before_arrow_cast() -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"qc_accepted=True disagrees with delta=0.5 >=.*30.0",
    ):
        validate_positive_qc_rows(
            [
                _rich_row(
                    seed=10.0,
                    output=10.5,
                    delta=0.5,
                    accepted=True,
                )
            ],
            accept_threshold_points=30.0,
        )


def test_passing_delta_marked_rejected_is_rejected_before_arrow_cast() -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"qc_accepted=False disagrees with delta=30.0 >=.*30.0",
    ):
        validate_positive_qc_rows(
            [_rich_row(accepted=False)],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize("accepted", [None, 0, 1, "true", [], {}])
def test_row_contract_requires_strict_boolean_qc_accepted(
    accepted: object,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="qc_accepted must be a bool",
    ):
        validate_positive_qc_rows(
            [_rich_row(accepted=accepted)],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("qc_seed_percentile", math.nan),
        ("qc_seed_percentile", math.inf),
        ("qc_seed_percentile", True),
        ("qc_seed_percentile", "10"),
        ("qc_output_percentile", math.nan),
        ("qc_output_percentile", -math.inf),
        ("qc_output_percentile", None),
        ("qc_delta", math.nan),
        ("qc_delta", math.inf),
        ("qc_delta", False),
        ("qc_delta", []),
    ],
)
def test_rich_row_contract_rejects_nonfinite_or_nonreal_qc_values(
    field_name: str,
    value: object,
) -> None:
    row = _rich_row()
    row[field_name] = value
    with pytest.raises(AxBenchBaseError, match=field_name):
        validate_positive_qc_rows(
            [row],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("qc_seed_percentile", -1e-12),
        ("qc_seed_percentile", 0.499999),
        ("qc_seed_percentile", 99.500001),
        ("qc_seed_percentile", 100.0),
        ("qc_output_percentile", -1.0),
        ("qc_output_percentile", 0.25),
        ("qc_output_percentile", 99.500001),
        ("qc_output_percentile", 100.0),
    ],
)
def test_rich_row_contract_rejects_values_outside_exact_qc_currency(
    field_name: str,
    value: float,
) -> None:
    row = _rich_row()
    row[field_name] = value
    with pytest.raises(
        AxBenchBaseError,
        match=rf"{field_name!r}.*0\.0 or.*\[0\.5, 99\.5\]",
    ):
        validate_positive_qc_rows(
            [row],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize("value", [0.0, 0.5, 99.5])
def test_rich_row_contract_accepts_exact_qc_currency_endpoints(
    value: float,
) -> None:
    validate_positive_qc_rows(
        [
            _rich_row(
                seed=value,
                output=value,
                delta=0.0,
                accepted=False,
            )
        ],
        accept_threshold_points=30.0,
    )


def test_rich_row_contract_requires_exact_delta_identity() -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=(
            r"qc_delta=29.999999.*does not exactly equal "
            r"qc_output_percentile-qc_seed_percentile=30.0"
        ),
    ):
        validate_positive_qc_rows(
            [_rich_row(delta=29.999999, accepted=False)],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize("zero_verified", [True, None, 0, 1, "false"])
def test_rich_row_contract_requires_strict_false_zero_verification(
    zero_verified: object,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="must set qc_zero_activation_verified=False",
    ):
        validate_positive_qc_rows(
            [_rich_row(zero_verified=zero_verified)],
            accept_threshold_points=30.0,
        )


def test_avoiding_row_nan_strict_zero_contract_is_valid() -> None:
    validate_positive_qc_rows(
        [_avoiding_row()],
        accept_threshold_points=30.0,
    )


@pytest.mark.parametrize(
    ("field_name", "value"),
    [
        ("qc_seed_percentile", 0.0),
        ("qc_seed_percentile", None),
        ("qc_seed_percentile", "nan"),
        ("qc_output_percentile", 0.0),
        ("qc_output_percentile", math.inf),
        ("qc_output_percentile", True),
        ("qc_delta", 0.0),
        ("qc_delta", -math.inf),
        ("qc_delta", []),
    ],
)
def test_avoiding_row_contract_requires_nan_percentile_fields(
    field_name: str,
    value: object,
) -> None:
    row = _avoiding_row()
    row[field_name] = value
    with pytest.raises(AxBenchBaseError, match=rf"{field_name!r} must be NaN"):
        validate_positive_qc_rows(
            [row],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize(
    ("accepted", "zero_verified"),
    [
        (False, True),
        (True, False),
        (False, False),
        (1, True),
        (True, 1),
    ],
)
def test_avoiding_row_contract_requires_accepted_strict_zero(
    accepted: object,
    zero_verified: object,
) -> None:
    expected = (
        "qc_accepted must be a bool"
        if not isinstance(accepted, bool)
        else "must set qc_accepted=True"
    )
    with pytest.raises(AxBenchBaseError, match=expected):
        validate_positive_qc_rows(
            [
                _avoiding_row(
                    accepted=accepted,
                    zero_verified=zero_verified,
                )
            ],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize("content_class", [None, "", "other", 1, True])
def test_row_contract_rejects_unknown_content_class(
    content_class: object,
) -> None:
    row = _rich_row()
    row["content_class"] = content_class
    with pytest.raises(AxBenchBaseError, match="unknown content_class"):
        validate_positive_qc_rows(
            [row],
            accept_threshold_points=30.0,
        )


@pytest.mark.parametrize("threshold", [math.nan, math.inf, -math.inf])
def test_row_contract_rejects_nonfinite_resolved_threshold(
    threshold: float,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="non-finite resolved threshold",
    ):
        validate_positive_qc_rows(
            [_rich_row()],
            accept_threshold_points=threshold,
        )


@pytest.mark.parametrize("threshold", [True, False, None, "30", [], {}])
def test_row_contract_rejects_nonreal_resolved_threshold(
    threshold: object,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match="resolved threshold that is not a real number",
    ):
        validate_positive_qc_rows(
            [_rich_row()],
            accept_threshold_points=threshold,
        )


@pytest.mark.parametrize("threshold", [-1e-12, -1.0, 100.0000000001, 101.0])
def test_row_contract_rejects_resolved_threshold_outside_percentile_space(
    threshold: float,
) -> None:
    with pytest.raises(
        AxBenchBaseError,
        match=r"resolved threshold outside \[0, 100\]",
    ):
        validate_positive_qc_rows(
            [_rich_row()],
            accept_threshold_points=threshold,
        )


def test_mixed_rich_best_fallback_and_avoiding_rows_validate_together() -> None:
    validate_positive_qc_rows(
        [
            _rich_row(),
            _avoiding_row(),
            _rich_row(
                seed=10.0,
                output=39.0,
                delta=29.0,
                accepted=False,
            ),
            _avoiding_row(),
        ],
        accept_threshold_points=30.0,
    )
