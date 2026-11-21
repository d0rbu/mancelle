from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.benchmark.compare_roundtrip_score_tables as compare_cli
from scripts.benchmark.compare_roundtrip_score_tables import (
    _DRIFT_COLUMNS,
    _EXACT_COLUMNS,
    _RESEEDING_DRIFT_COLUMNS,
    _SCIENTIFIC_DRIFT_COLUMNS,
    RoundtripScoreComparisonError,
    compare_roundtrip_score_tables,
    main,
)
from steering.eval.axbench_bpb_adv.schemas import ROUNDTRIP_SCORE_SCHEMA


def _default_value(data_type: pa.DataType) -> Any:
    if pa.types.is_string(data_type):
        return "value"
    if pa.types.is_boolean(data_type):
        return False
    if pa.types.is_integer(data_type):
        return 1
    if pa.types.is_floating(data_type):
        return 1.0
    if pa.types.is_list(data_type):
        return []
    raise AssertionError(f"unhandled test data type: {data_type}")


def _row(*, candidate_id: str = "candidate-a", sample_index: int = 0) -> dict[str, Any]:
    row = {
        field.name: _default_value(field.type)
        for field in ROUNDTRIP_SCORE_SCHEMA
    }
    row.update(
        {
            "candidate_id": candidate_id,
            "sample_index": sample_index,
            "generation_seed": 101,
            "control_generation_seed": 202,
            "roundtrip_prompt_token_digest": "roundtrip-digest",
            "control_roundtrip_prompt_token_digest": "control-digest",
            "reference_segment_token_count": 8,
            "generated_text": "generated",
            "control_generated_text": "control",
        }
    )
    return row


def _write(path: Path, rows: list[dict[str, Any]]) -> None:
    pq.write_table(pa.Table.from_pylist(rows, schema=ROUNDTRIP_SCORE_SCHEMA), path)


def _mutated_value(column: str, value: Any) -> Any:
    data_type = ROUNDTRIP_SCORE_SCHEMA.field(column).type
    if pa.types.is_string(data_type):
        return f"{value}-mutated"
    if pa.types.is_boolean(data_type):
        return not value
    if pa.types.is_integer(data_type):
        return int(value) + 1
    if pa.types.is_floating(data_type):
        return float(value) + 1.0
    if pa.types.is_list(data_type):
        return [*value, 999]
    raise AssertionError(f"unhandled mutation data type: {data_type}")


def test_comparison_column_partition_is_fail_closed() -> None:
    assert _DRIFT_COLUMNS == {
        "generated_text",
        "n_generated_tokens",
        "stop_reason",
        "control_generated_text",
        "control_n_generated_tokens",
        "control_stop_reason",
        "roundtrip_pre_act",
        "roundtrip_post_act",
        "roundtrip_percentile",
        "control_roundtrip_pre_act",
        "control_roundtrip_post_act",
        "control_roundtrip_percentile",
    }
    assert set(_EXACT_COLUMNS) == set(ROUNDTRIP_SCORE_SCHEMA.names) - _DRIFT_COLUMNS
    assert "loss_concept_rich" in _EXACT_COLUMNS
    assert "loss_concept_avoiding" in _EXACT_COLUMNS
    assert "n_roundtrip_features" in _EXACT_COLUMNS
    assert "control_n_roundtrip_features" in _EXACT_COLUMNS


def test_compare_identical_tables_reports_zero_drift_and_writes_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    json_path = tmp_path / "report.json"
    rows = [_row(), _row(candidate_id="candidate-b", sample_index=1)]
    _write(path_a, rows)
    _write(path_b, rows)

    assert main([str(path_a), str(path_b), "--json", str(json_path)]) == 0

    stdout_report = json.loads(capsys.readouterr().out)
    file_report = json.loads(json_path.read_text(encoding="utf-8"))
    assert stdout_report == file_report
    assert stdout_report["exact_contract_equal"] is True
    assert stdout_report["n_identical_generated_text"] == 2
    assert stdout_report["n_identical_control_text"] == 2
    assert stdout_report["row_order_equal"] is True
    assert stdout_report["row_order_violation"] is None
    assert all(
        drift == {"max_abs_delta": 0.0, "mean_abs_delta": 0.0}
        for drift in stdout_report["numeric_drift"].values()
    )
    assert all(
        stats["n_different"] == 0
        for stats in stdout_report["allowlisted_drift"].values()
    )
    assert all(
        means == {"mean_before": 1.0, "mean_after": 1.0, "delta": 0.0}
        for means in stdout_report["aggregate_means"].values()
    )


def test_compare_reports_text_drift_without_failing(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b["generated_text"] = "drifted"
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    report = compare_roundtrip_score_tables(path_a, path_b)

    assert report["exact_contract_equal"] is True
    assert report["n_identical_generated_text"] == 0
    assert report["generated_text_identity_rate"] == 0.0
    assert report["allowlisted_drift"]["generated_text"]["n_different"] == 1
    assert main([str(path_a), str(path_b)]) == 0


def test_compare_returns_one_and_reports_first_row_order_violation(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row(candidate_id="candidate-a", sample_index=0)
    row_b = _row(candidate_id="candidate-b", sample_index=1)
    _write(path_a, [row_a, row_b])
    _write(path_b, [row_b, row_a])

    assert main([str(path_a), str(path_b)]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["exact_contract_equal"] is False
    assert report["row_order_equal"] is False
    assert report["row_order_violation"] == {
        "first_differing_position": 0,
        "table_a_key": ["candidate-a", 0],
        "table_b_key": ["candidate-b", 1],
    }


def test_cli_reports_strict_prefix_ordered_key_length_mismatch(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    rows_a = [
        _row(candidate_id="candidate-a", sample_index=0),
        _row(candidate_id="candidate-b", sample_index=1),
        _row(candidate_id="candidate-c", sample_index=2),
    ]
    _write(path_a, rows_a)
    _write(path_b, rows_a[:-1])
    original_load_rows = compare_cli._load_rows

    def _load_rows_with_complete_comparison_map(path: Path):
        ordered_keys, rows = original_load_rows(path)
        if path == path_b:
            trailing_key = ("candidate-c", 2)
            rows[trailing_key] = rows_a[-1]
        return ordered_keys, rows

    monkeypatch.setattr(compare_cli, "_load_rows", _load_rows_with_complete_comparison_map)

    assert main([str(path_a), str(path_b)]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["exact_contract_equal"] is False
    assert report["row_order_equal"] is False
    assert report["row_order_violation"] == {
        "first_differing_position": 2,
        "table_a_key": ["candidate-c", 2],
        "table_b_key": None,
    }


@pytest.mark.parametrize("column", _EXACT_COLUMNS)
def test_every_non_drift_column_is_exact(
    column: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[column] = _mutated_value(column, row_b[column])
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    assert main([str(path_a), str(path_b)]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["exact_contract_equal"] is False
    if column not in {"candidate_id", "sample_index"}:
        assert report["exact_columns"][column]["n_mismatched"] == 1


@pytest.mark.parametrize("column", sorted(_DRIFT_COLUMNS))
def test_every_allowlisted_drift_column_is_reported_without_failing(
    column: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[column] = _mutated_value(column, row_b[column])
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    assert main([str(path_a), str(path_b)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["exact_contract_equal"] is True
    assert report["allowlisted_drift"][column] == {
        "n_equal": 0,
        "n_different": 1,
        "identity_rate": 0.0,
    }


def test_compare_reports_scientific_aggregate_mean_deltas(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    rows_a = [_row(), _row(candidate_id="candidate-b", sample_index=1)]
    rows_b = [dict(row) for row in rows_a]
    for column in _SCIENTIFIC_DRIFT_COLUMNS:
        rows_a[0][column] = 1.0
        rows_a[1][column] = 3.0
        rows_b[0][column] = 2.0
        rows_b[1][column] = 6.0
    _write(path_a, rows_a)
    _write(path_b, rows_b)

    report = compare_roundtrip_score_tables(path_a, path_b)

    assert report["exact_contract_equal"] is True
    for column in _SCIENTIFIC_DRIFT_COLUMNS:
        assert report["aggregate_means"][column] == {
            "mean_before": 2.0,
            "mean_after": 4.0,
            "delta": 2.0,
        }


def test_compare_returns_one_for_seed_mismatch(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b["generation_seed"] = 999
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    report = compare_roundtrip_score_tables(path_a, path_b)

    assert report["exact_contract_equal"] is False
    assert report["exact_columns"]["generation_seed"]["n_mismatched"] == 1
    assert main([str(path_a), str(path_b)]) == 1


def test_expect_control_drift_allows_control_seed_and_text_only(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    rows_a = [_row(), _row(candidate_id="candidate-b", sample_index=1)]
    rows_b = [dict(row) for row in rows_a]
    rows_a[0]["control_roundtrip_percentile"] = 1.0
    rows_a[1]["control_roundtrip_percentile"] = 3.0
    rows_b[0]["control_roundtrip_percentile"] = 2.0
    rows_b[1]["control_roundtrip_percentile"] = 6.0
    rows_b[0]["control_generation_seed"] = 999
    rows_b[0]["control_generated_text"] = "drifted control"
    _write(path_a, rows_a)
    _write(path_b, rows_b)

    assert main([str(path_a), str(path_b)]) == 1
    capsys.readouterr()
    assert main([str(path_a), str(path_b), "--expect-control-drift"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["exact_contract_equal"] is True
    assert report["expect_control_drift"] is True
    assert report["control_drift"]["control_generation_seed"] == {
        "n_equal": 1,
        "n_different": 1,
        "identity_rate": 0.5,
    }
    assert report["control_drift"]["control_generated_text"] == {
        "n_equal": 1,
        "n_different": 1,
        "identity_rate": 0.5,
    }
    assert report["control_drift"]["control_roundtrip_percentile_distribution"] == {
        "table_a": {"mean": 2.0, "std": 1.0},
        "table_b": {"mean": 4.0, "std": 2.0},
        "max_empirical_cdf_gap": 0.5,
    }
    assert report["allowlisted_drift"]["control_generation_seed"][
        "n_different"
    ] == 1


def test_expect_control_drift_still_rejects_steered_seed_mismatch(
    tmp_path: Path,
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b["generation_seed"] = 999
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    report = compare_roundtrip_score_tables(
        path_a,
        path_b,
        expect_control_drift=True,
    )

    assert report["exact_contract_equal"] is False
    assert report["exact_columns"]["generation_seed"]["n_mismatched"] == 1
    assert main([str(path_a), str(path_b), "--expect-control-drift"]) == 1


@pytest.mark.parametrize(
    "column",
    [
        "generated_text",
        "n_generated_tokens",
        "stop_reason",
        "roundtrip_pre_act",
        "roundtrip_post_act",
        "roundtrip_percentile",
        "n_roundtrip_features",
    ],
)
def test_expect_control_drift_rejects_each_steered_output_and_probe_column(
    column: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[column] = _mutated_value(column, row_b[column])
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    assert main([str(path_a), str(path_b), "--expect-control-drift"]) == 1

    report = json.loads(capsys.readouterr().out)
    assert report["exact_columns"][column]["n_mismatched"] == 1


@pytest.mark.parametrize(
    "column",
    [
        "control_generated_text",
        "control_n_generated_tokens",
        "control_stop_reason",
        "control_generation_seed",
        "control_roundtrip_pre_act",
        "control_roundtrip_post_act",
        "control_roundtrip_percentile",
        "control_n_roundtrip_features",
    ],
)
def test_expect_control_drift_allows_each_control_output_and_probe_column(
    column: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[column] = _mutated_value(column, row_b[column])
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    assert main([str(path_a), str(path_b), "--expect-control-drift"]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["allowlisted_drift"][column] == {
        "n_equal": 0,
        "n_different": 1,
        "identity_rate": 0.0,
    }


@pytest.mark.parametrize("column", sorted(_RESEEDING_DRIFT_COLUMNS))
def test_expect_reseeding_tolerates_every_generation_and_probe_drift_column(
    column: str,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[column] = _mutated_value(column, row_b[column])
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    assert main([str(path_a), str(path_b), "--expect-reseeding"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["expect_reseeding"] is True
    assert report["allowlisted_drift"][column]["n_different"] == 1


@pytest.mark.parametrize(
    "column",
    sorted(
        set(ROUNDTRIP_SCORE_SCHEMA.names)
        - _RESEEDING_DRIFT_COLUMNS
        - {"candidate_id", "sample_index"}
    ),
)
def test_expect_reseeding_still_gates_every_surviving_exact_column(
    column: str,
    tmp_path: Path,
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[column] = _mutated_value(column, row_b[column])
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    report = compare_roundtrip_score_tables(
        path_a,
        path_b,
        expect_reseeding=True,
    )
    assert report["exact_contract_equal"] is False
    assert main([str(path_a), str(path_b), "--expect-reseeding"]) == 1


@pytest.mark.parametrize("loss_column", ["loss_concept_rich", "loss_concept_avoiding"])
def test_expect_reseeding_hard_gates_teacher_forced_losses(
    loss_column: str,
    tmp_path: Path,
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_b = dict(row_a)
    row_b[loss_column] = 9.0
    _write(path_a, [row_a])
    _write(path_b, [row_b])

    assert main([str(path_a), str(path_b), "--expect-reseeding"]) == 1


def test_drift_flags_are_mutually_exclusive(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _write(path_a, [_row()])
    _write(path_b, [_row()])

    with pytest.raises(
        RoundtripScoreComparisonError,
        match="mutually exclusive",
    ):
        compare_roundtrip_score_tables(
            path_a,
            path_b,
            expect_control_drift=True,
            expect_reseeding=True,
        )
    with pytest.raises(SystemExit, match="2"):
        main(
            [
                str(path_a),
                str(path_b),
                "--expect-control-drift",
                "--expect-reseeding",
            ]
        )


def test_compare_rejects_non_finite_drift_value_before_summaries(
    tmp_path: Path,
) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    row_a = _row()
    row_a["control_roundtrip_percentile"] = math.nan
    _write(path_a, [row_a])
    _write(path_b, [_row()])

    with pytest.raises(
        RoundtripScoreComparisonError,
        match=(
            r"non-finite numeric drift value in table A, "
            r"row key \('candidate-a', 0\), column 'control_roundtrip_percentile'"
        ),
    ):
        compare_roundtrip_score_tables(path_a, path_b)


def test_compare_rejects_missing_keys_with_examples(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _write(path_a, [_row(candidate_id="only-a")])
    _write(path_b, [_row(candidate_id="only-b")])

    with pytest.raises(
        RoundtripScoreComparisonError,
        match=r"key sets differ.*only-b.*only-a",
    ):
        compare_roundtrip_score_tables(path_a, path_b)


def test_compare_rejects_duplicate_keys(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    duplicate = _row()
    _write(path_a, [duplicate, duplicate])
    _write(path_b, [duplicate])

    with pytest.raises(RoundtripScoreComparisonError, match="duplicate.*candidate-a"):
        compare_roundtrip_score_tables(path_a, path_b)


def test_compare_rejects_schema_mismatch(tmp_path: Path) -> None:
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _write(path_a, [_row()])
    pq.write_table(pa.table({"candidate_id": ["candidate-a"]}), path_b)

    with pytest.raises(RoundtripScoreComparisonError, match="schema mismatch"):
        compare_roundtrip_score_tables(path_a, path_b)


def test_compare_accepts_empty_tables_with_zero_stats(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Zero-row tables are a degenerate but valid comparison: exit 0, ordered
    keys trivially equal, and every drift summary empty-safe (no NaN, no
    crash in the ECDF merge)."""
    path_a = tmp_path / "a.parquet"
    path_b = tmp_path / "b.parquet"
    _write(path_a, [])
    _write(path_b, [])

    assert main([str(path_a), str(path_b)]) == 0

    report = json.loads(capsys.readouterr().out)
    assert report["exact_contract_equal"] is True
    assert report["row_order_equal"] is True
    assert report["row_order_violation"] is None
    assert all(
        drift == {"max_abs_delta": 0.0, "mean_abs_delta": 0.0}
        for drift in report["numeric_drift"].values()
    )
