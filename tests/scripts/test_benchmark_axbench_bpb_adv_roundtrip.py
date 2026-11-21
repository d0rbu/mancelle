"""Tests for the issue #1257 roundtrip benchmark harness."""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip as benchmark_module
from scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip import (
    _assert_subset_schema_fidelity,
    _counter_totals,
    _hardlink_copy_run_dir,
    _largest_remainder_allocation,
    _load_profile_summary,
    _parse_args,
    _positive_int,
    _reject_benchmark_dir_inside_source,
    _reject_reserved_overrides,
    _run_metric,
    _score_table_summary,
    _write_subset_table_atomic,
    candidate_fingerprint,
    compare_roundtrip_baselines,
    run_benchmark,
    sample_candidates,
    validate_regression_sentinel,
)
from steering.eval.axbench_bpb_adv.artifact_identity import (
    ARTIFACT_FAMILY_METADATA_KEY,
    FEATURE_CONTRACT_METADATA_KEY,
    PRECOMPUTED_BAG_SET_METADATA_KEY,
    SOURCE_CONTRACT_METADATA_KEY,
    restamp_candidates_bag_set,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
    SCORE_SCHEMA,
)


def _default_for_type(field: pa.Field) -> object:
    if pa.types.is_list(field.type):
        return []
    if pa.types.is_boolean(field.type):
        return False
    if pa.types.is_integer(field.type):
        return 0
    if pa.types.is_floating(field.type):
        return 0.0
    if pa.types.is_string(field.type):
        return ""
    raise AssertionError(f"test helper lacks a default for {field.type}")


def _score_row(schema: pa.Schema, candidate_id: str, rich: float, avoid: float) -> dict[str, object]:
    row = {field.name: _default_for_type(field) for field in schema}
    row.update(
        {
            "candidate_id": candidate_id,
            "loss_concept_rich": rich,
            "loss_concept_avoiding": avoid,
        }
    )
    return row


def _write_table(path: Path, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _candidate_table(
    strata: list[tuple[str, str, str, int]],
    *,
    metadata: dict[bytes, bytes] | None = None,
) -> pa.Table:
    rows = []
    candidate_index = 0
    for method_name, target_model, mode, count in strata:
        for _ in range(count):
            rows.append(
                {
                    "candidate_id": f"candidate-{candidate_index:04d}",
                    "method_name": method_name,
                    "target_model": target_model,
                    "mode": mode,
                    "original_index": candidate_index,
                }
            )
            candidate_index += 1
    table = pa.Table.from_pylist(rows)
    return table.replace_schema_metadata(metadata)


def _candidate_schema_row(candidate_id: str) -> dict[str, object]:
    row = {field.name: _default_for_type(field) for field in CANDIDATES_SCHEMA}
    row.update(
        {
            "candidate_id": candidate_id,
            "attack_row_id": f"attack-{candidate_id}",
            "core_prefix_id": f"prefix-{candidate_id}",
            "mode": "positive",
            "method_name": "non_bag_backed_fixture_method",
            "target_model": "encdec",
        }
    )
    return row


def _stamped_clustered_candidate_table(
    paths: AxBenchBpbAdvPaths,
    *,
    rows: int = 4,
) -> pa.Table:
    metadata = {
        FEATURE_CONTRACT_METADATA_KEY: b"feature-contract-stamp",
        SOURCE_CONTRACT_METADATA_KEY: b"source-contract-stamp",
        ARTIFACT_FAMILY_METADATA_KEY: b"candidates",
        PRECOMPUTED_BAG_SET_METADATA_KEY: b"placeholder",
    }
    table = pa.Table.from_pylist(
        [_candidate_schema_row(f"candidate-{index}") for index in range(rows)],
        schema=CANDIDATES_SCHEMA,
    ).replace_schema_metadata(metadata)
    return restamp_candidates_bag_set(table, paths=paths)


def _benchmark_args(
    *,
    output_root: Path,
    run_tag: str,
    benchmark_dir: Path,
    metrics: str = "both",
    sample_size: int | None = None,
    sample_seed: int = 0,
) -> argparse.Namespace:
    return argparse.Namespace(
        output_root=output_root,
        run_tag=run_tag,
        benchmark_dir=benchmark_dir,
        python="python",
        metrics=metrics,
        sample_size=sample_size,
        sample_seed=sample_seed,
    )


def _install_fake_metric_outputs(command: list[str]) -> None:
    metric = next(
        value.split("=", 1)[1]
        for value in command
        if value.startswith("steering_metric=")
    )
    score_path = Path(
        next(
            value.split("=", 1)[1]
            for value in command
            if value.startswith("scores_path=")
        )
    )
    profile_dir = Path(
        next(
            value.split("=", 1)[1]
            for value in command
            if value.startswith("+profile.output_dir=")
        )
    )
    schema = ROUNDTRIP_SCORE_SCHEMA if metric == "roundtrip" else SCORE_SCHEMA
    row = _score_row(schema, "candidate-a", 1.0, 2.0)
    if metric == "roundtrip":
        row["n_generated_tokens"] = 2
        row["control_n_generated_tokens"] = 3
    _write_table(score_path, schema, [row])
    profile_dir.mkdir(parents=True)
    (profile_dir / "summary.yaml").write_text(
        json.dumps({"elapsed_s": 1.0, "counters": []}),
        encoding="utf-8",
    )


def test_candidate_fingerprint_changes_when_bytes_change(tmp_path: Path) -> None:
    path = tmp_path / "candidates.parquet"
    pq.write_table(pa.table({"candidate_id": ["a"], "value": [1]}), path)
    first = candidate_fingerprint(path)

    pq.write_table(pa.table({"candidate_id": ["a"], "value": [2]}), path)
    second = candidate_fingerprint(path)

    assert first.rows == second.rows == 1
    assert first.sha256 != second.sha256
    assert first.candidate_ids_sha256 == second.candidate_ids_sha256


def test_benchmark_rejects_overrides_owned_by_harness() -> None:
    owned = (
        "stage=score",
        "steering_metric=roundtrip",
        "run_tag=other",
        "output_root=/tmp/output",
        "scores_path=/tmp/scores.parquet",
        "profile.enabled=false",
        "profile.output_dir=/tmp/profile",
    )
    for override in owned:
        with pytest.raises(ValueError, match=override.split("=", 1)[0]):
            _reject_reserved_overrides([override])
    with pytest.raises(ValueError, match=r"\+steering_metric=roundtrip"):
        _reject_reserved_overrides(["+steering_metric=roundtrip"])

    _reject_reserved_overrides(["source_run_dir=/tmp/source", "device=cpu"])


def test_benchmark_dir_must_not_equal_or_descend_from_source_run(tmp_path: Path) -> None:
    source = (tmp_path / "eval" / "run").resolve()
    for benchmark_dir in (source, source / "benchmarks" / "bounded"):
        with pytest.raises(
            ValueError,
            match=(
                rf"benchmark_dir={benchmark_dir}.*source_run_dir={source}"
            ),
        ):
            _reject_benchmark_dir_inside_source(
                benchmark_dir=benchmark_dir,
                source_run_dir=source,
            )

    _reject_benchmark_dir_inside_source(
        benchmark_dir=(tmp_path / "eval" / "sibling").resolve(),
        source_run_dir=source,
    )


def test_nested_benchmark_rejection_happens_before_creation(tmp_path: Path) -> None:
    output_root = tmp_path / "eval"
    run_tag = "run"
    source = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag).root
    source.mkdir(parents=True)
    benchmark_dir = source / "nested" / "benchmark"

    with pytest.raises(ValueError, match="must not equal or descend"):
        run_benchmark(
            _benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=benchmark_dir,
                metrics="roundtrip",
            ),
            (),
        )

    assert not benchmark_dir.exists()


def test_sampler_is_deterministic_in_ids_and_parquet_bytes(tmp_path: Path) -> None:
    table = _candidate_table([("method", "target", "mode", 40)])
    first = sample_candidates(table, sample_size=8, sample_seed=17)
    second = sample_candidates(table, sample_size=8, sample_seed=17)
    different = sample_candidates(table, sample_size=8, sample_seed=18)

    first_ids = first.table.column("candidate_id").to_pylist()
    second_ids = second.table.column("candidate_id").to_pylist()
    different_ids = different.table.column("candidate_id").to_pylist()
    assert first_ids == second_ids
    assert first_ids == [
        "candidate-0005",
        "candidate-0008",
        "candidate-0013",
        "candidate-0021",
        "candidate-0025",
        "candidate-0032",
        "candidate-0034",
        "candidate-0036",
    ]
    assert first_ids != different_ids

    first_path = tmp_path / "first" / "candidates.parquet"
    second_path = tmp_path / "second" / "candidates.parquet"
    _write_subset_table_atomic(first.table, first_path, source_schema=table.schema)
    _write_subset_table_atomic(second.table, second_path, source_schema=table.schema)
    assert hashlib.sha256(first_path.read_bytes()).hexdigest() == hashlib.sha256(
        second_path.read_bytes()
    ).hexdigest()


def test_sampler_uses_proportional_largest_remainder_allocation() -> None:
    table = _candidate_table(
        [
            ("large", "target", "mode", 20),
            ("medium", "target", "mode", 10),
            ("small", "target", "mode", 10),
        ]
    )

    sample = sample_candidates(table, sample_size=8, sample_seed=0)
    minimum = sample_candidates(table, sample_size=3, sample_seed=0)

    assert sample.strata_counts == {
        '["large","target","mode"]': 4,
        '["medium","target","mode"]': 2,
        '["small","target","mode"]': 2,
    }
    assert all(count >= 1 for count in minimum.strata_counts.values())


def test_allocation_correction_loops_cover_overshoot_and_undershoot() -> None:
    overshoot = {
        ("large", "target", "mode"): list(range(8)),
        ("small-a", "target", "mode"): [8],
        ("small-b", "target", "mode"): [9],
        ("small-c", "target", "mode"): [10],
        ("small-d", "target", "mode"): [11],
    }
    assert _largest_remainder_allocation(overshoot, 6, 12) == {
        ("large", "target", "mode"): 2,
        ("small-a", "target", "mode"): 1,
        ("small-b", "target", "mode"): 1,
        ("small-c", "target", "mode"): 1,
        ("small-d", "target", "mode"): 1,
    }

    undershoot = {
        ("a", "target", "mode"): list(range(5)),
        ("b", "target", "mode"): list(range(5, 8)),
        ("c", "target", "mode"): list(range(8, 10)),
    }
    assert _largest_remainder_allocation(undershoot, 6, 10) == {
        ("a", "target", "mode"): 3,
        ("b", "target", "mode"): 2,
        ("c", "target", "mode"): 1,
    }


def test_sampler_rejects_missing_sampling_columns_and_zero_rows() -> None:
    with pytest.raises(ValueError, match="candidate table lacks sampling columns"):
        sample_candidates(pa.table({"candidate_id": ["candidate"]}), sample_size=1, sample_seed=0)

    empty = _candidate_table([("method", "target", "mode", 1)]).slice(0, 0)
    with pytest.raises(
        ValueError,
        match=r"sample_size=1 cannot sample table\.num_rows=0",
    ):
        sample_candidates(empty, sample_size=1, sample_seed=0)


def test_stratum_seed_and_report_keys_have_no_delimiter_collision() -> None:
    table = _candidate_table(
        [
            ("a|b", "c", "d", 6),
            ("a", "b|c", "d", 6),
        ]
    )

    sample = sample_candidates(table, sample_size=4, sample_seed=7)

    assert sample.strata_counts == {
        '["a","b|c","d"]': 2,
        '["a|b","c","d"]': 2,
    }
    assert len(sample.strata_counts) == 2


def test_sampler_rejects_out_of_bounds_sizes() -> None:
    table = _candidate_table(
        [
            ("a", "target", "mode", 2),
            ("b", "target", "mode", 2),
            ("c", "target", "mode", 2),
        ]
    )

    with pytest.raises(
        ValueError,
        match=r"sample_size=7.*table\.num_rows=6",
    ):
        sample_candidates(table, sample_size=7, sample_seed=0)
    with pytest.raises(ValueError, match=r"sample_size=2.*n_strata=3"):
        sample_candidates(table, sample_size=2, sample_seed=0)


def test_sample_size_must_be_positive_at_argparse_boundary(tmp_path: Path) -> None:
    base = [
        "--output-root",
        str(tmp_path / "output"),
        "--run-tag",
        "run",
        "--benchmark-dir",
        str(tmp_path / "benchmark"),
    ]
    with pytest.raises(SystemExit):
        _parse_args([*base, "--sample-size", "0"])
    with pytest.raises(SystemExit):
        _parse_args([*base, "--sample-size", "-1"])


def test_sampler_preserves_original_table_order() -> None:
    table = _candidate_table([("method", "target", "mode", 20)])
    reverse_indices = pa.array(list(reversed(range(table.num_rows))), type=pa.int64())
    table = table.take(reverse_indices)

    sample = sample_candidates(table, sample_size=8, sample_seed=9)

    observed = sample.table.column("original_index").to_pylist()
    original_order = table.column("original_index").to_pylist()
    positions = [original_order.index(index) for index in observed]
    assert positions == sorted(positions)


def test_subset_preserves_schema_metadata_and_guards_written_schema(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _candidate_table(
        [("method", "target", "mode", 10)],
        metadata={b"axbench.artifact": b"stamp"},
    )
    sample = sample_candidates(table, sample_size=4, sample_seed=0)
    destination = tmp_path / "good" / "candidates.parquet"

    _write_subset_table_atomic(
        sample.table,
        destination,
        source_schema=table.schema,
    )

    written_schema = pq.read_schema(destination)
    assert sample.table.schema.metadata == table.schema.metadata
    assert written_schema.metadata == table.schema.metadata
    assert (
        written_schema.serialize().to_pybytes()
        == table.schema.serialize().to_pybytes()
    )

    real_write_table = pq.write_table

    def drop_metadata(value: pa.Table, path: Path) -> None:
        real_write_table(value.replace_schema_metadata(None), path)

    monkeypatch.setattr(
        "scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip.pq.write_table",
        drop_metadata,
    )
    with pytest.raises(RuntimeError, match="schema drifted during parquet write"):
        _write_subset_table_atomic(
            sample.table,
            tmp_path / "mutated" / "candidates.parquet",
            source_schema=table.schema,
        )

    with pytest.raises(RuntimeError, match="schema drifted before write"):
        _write_subset_table_atomic(
            sample.table.replace_schema_metadata(None),
            tmp_path / "prewrite-mutated" / "candidates.parquet",
            source_schema=table.schema,
        )


def test_derived_run_hardlinks_inputs_but_not_candidates(tmp_path: Path) -> None:
    source = tmp_path / "source" / "run"
    source.mkdir(parents=True)
    included = source / "source_contract.json"
    included.write_text("{}\n", encoding="utf-8")
    nested = source / "context_cache" / "nested" / "cache.bin"
    nested.parent.mkdir(parents=True)
    nested.write_bytes(b"nested-input")
    candidate_path = source / "candidates" / "candidates.parquet"
    candidate_path.parent.mkdir()
    table = _candidate_table([("method", "target", "mode", 10)])
    pq.write_table(table, candidate_path)
    for relative in (
        "scores.parquet",
        "roundtrip_scores.parquet",
        "general_val_bpb.parquet",
        "aggregates/summary.json",
        "roundtrip_aggregates/summary.json",
        "plots/plot.png",
        "roundtrip_plots/plot.png",
        "steering_strength_plots/plot.png",
        "samples/sample.yaml",
        "profile/summary.yaml",
        "universal_triggers.parquet",
        "candidates/universal_gcg_checkpoint/manifest.json",
    ):
        path = source / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"excluded")
    destination = tmp_path / "derived" / "run"

    _hardlink_copy_run_dir(source, destination)
    sample = sample_candidates(table, sample_size=4, sample_seed=0)
    derived_candidate_path = destination / "candidates" / "candidates.parquet"
    _write_subset_table_atomic(
        sample.table,
        derived_candidate_path,
        source_schema=table.schema,
    )

    assert os.stat(included).st_ino == os.stat(
        destination / "source_contract.json"
    ).st_ino
    assert (
        os.stat(nested).st_ino
        == os.stat(destination / "context_cache" / "nested" / "cache.bin").st_ino
    )
    assert os.stat(candidate_path).st_ino != os.stat(derived_candidate_path).st_ino
    assert not (destination / "scores.parquet").exists()
    assert not (destination / "roundtrip_scores.parquet").exists()
    assert not (destination / "general_val_bpb.parquet").exists()
    assert not (destination / "aggregates").exists()
    assert not (destination / "roundtrip_aggregates").exists()
    assert not (destination / "plots").exists()
    assert not (destination / "roundtrip_plots").exists()
    assert not (destination / "steering_strength_plots").exists()
    assert not (destination / "samples").exists()
    assert not (destination / "profile").exists()
    assert not (destination / "universal_triggers.parquet").exists()
    assert not (destination / "candidates/universal_gcg_checkpoint").exists()


def test_hardlink_copy_guards_invalid_source_entries(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError, match="source run directory does not exist"):
        _hardlink_copy_run_dir(missing, tmp_path / "missing-destination")

    source = tmp_path / "source"
    source.mkdir()
    destination = tmp_path / "destination"
    destination.mkdir()
    with pytest.raises(FileExistsError, match="derived run directory already exists"):
        _hardlink_copy_run_dir(source, destination)

    symlink_source = tmp_path / "symlink-source"
    symlink_source.mkdir()
    (symlink_source / "link").symlink_to(tmp_path / "target")
    with pytest.raises(ValueError, match="source run directory contains forbidden symlink"):
        _hardlink_copy_run_dir(symlink_source, tmp_path / "symlink-destination")

    fifo_source = tmp_path / "fifo-source"
    fifo_source.mkdir()
    os.mkfifo(fifo_source / "fifo")
    with pytest.raises(ValueError, match="source run directory contains unsupported entry"):
        _hardlink_copy_run_dir(fifo_source, tmp_path / "fifo-destination")


def test_every_zero_arg_path_accessor_is_classified_for_derived_runs(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    discovered = {
        name
        for name, function in inspect.getmembers(AxBenchBpbAdvPaths, inspect.isfunction)
        if name != "__init__"
        and len(inspect.signature(function).parameters) == 1
        and isinstance(getattr(paths, name)(), Path)
        and getattr(paths, name)().is_relative_to(paths.root)
    }
    excluded = benchmark_module._DERIVED_RUN_EXCLUDED_PATH_ACCESSORS
    required = benchmark_module._DERIVED_RUN_REQUIRED_INPUT_ACCESSORS

    assert excluded.isdisjoint(required)
    assert discovered == excluded | required


def test_roundtrip_baselines_are_compared_with_tolerance(tmp_path: Path) -> None:
    class_path = tmp_path / "class.parquet"
    roundtrip_path = tmp_path / "roundtrip.parquet"
    _write_table(
        class_path,
        SCORE_SCHEMA,
        [_score_row(SCORE_SCHEMA, "candidate-a", 1.0, 2.0)],
    )
    _write_table(
        roundtrip_path,
        ROUNDTRIP_SCORE_SCHEMA,
        [_score_row(ROUNDTRIP_SCORE_SCHEMA, "candidate-a", 1.0 + 1e-8, 2.0)],
    )

    comparison = compare_roundtrip_baselines(class_path, roundtrip_path)

    assert comparison["within_tolerance"] is True
    assert comparison["mismatch_count"] == 0
    assert comparison["roundtrip_missing_candidates"] == 0


def test_roundtrip_baseline_mismatch_is_reported(tmp_path: Path) -> None:
    class_path = tmp_path / "class.parquet"
    roundtrip_path = tmp_path / "roundtrip.parquet"
    _write_table(
        class_path,
        SCORE_SCHEMA,
        [_score_row(SCORE_SCHEMA, "candidate-a", 1.0, 2.0)],
    )
    _write_table(
        roundtrip_path,
        ROUNDTRIP_SCORE_SCHEMA,
        [_score_row(ROUNDTRIP_SCORE_SCHEMA, "candidate-a", 1.1, 2.0)],
    )

    comparison = compare_roundtrip_baselines(class_path, roundtrip_path)

    assert comparison["within_tolerance"] is False
    assert comparison["mismatch_count"] == 1
    assert comparison["max_abs_loss_delta"] == pytest.approx(0.1)


def test_score_table_summary_counts_roundtrip_tokens(tmp_path: Path) -> None:
    path = tmp_path / "roundtrip.parquet"
    row = _score_row(ROUNDTRIP_SCORE_SCHEMA, "candidate-a", 1.0, 2.0)
    row.update(
        {
            "n_generated_tokens": 3,
            "control_n_generated_tokens": 4,
        }
    )
    _write_table(path, ROUNDTRIP_SCORE_SCHEMA, [row])

    summary = _score_table_summary("roundtrip", path)

    assert summary["rows"] == 1
    assert summary["candidate_ids"] == 1
    assert summary["generated_tokens"] == 7


def test_regression_sentinel_requires_strict_counter_increase() -> None:
    result = validate_regression_sentinel(
        {"engine_constructions": 2, "prefill_tokens": 100},
        {"engine_constructions": 8, "prefill_tokens": 400},
        required_increases=("engine_constructions", "prefill_tokens"),
    )
    assert result["passed"] is True

    with pytest.raises(AssertionError, match="did not increase"):
        validate_regression_sentinel(
            {"engine_constructions": 2},
            {"engine_constructions": 2},
            required_increases=("engine_constructions",),
        )


def test_run_metric_uses_fresh_production_cli_and_separate_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "existing"
    candidate_path = output_root / run_tag / "candidates" / "candidates.parquet"
    candidate_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"candidate_id": ["candidate-a"]}), candidate_path)
    candidate = candidate_fingerprint(candidate_path)
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()
    args = argparse.Namespace(
        output_root=output_root,
        run_tag=run_tag,
        benchmark_dir=benchmark_dir,
        python="python",
    )

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        assert command[:3] == ["python", "-m", "scripts.eval.axbench_bpb_adv_eval"]
        assert "stage=score" in command
        score_path = Path(next(value.split("=", 1)[1] for value in command if value.startswith("scores_path=")))
        profile_dir = Path(next(value.split("=", 1)[1] for value in command if value.startswith("+profile.output_dir=")))
        row = _score_row(ROUNDTRIP_SCORE_SCHEMA, "candidate-a", 1.0, 2.0)
        row["n_generated_tokens"] = 2
        row["control_n_generated_tokens"] = 3
        _write_table(score_path, ROUNDTRIP_SCORE_SCHEMA, [row])
        profile_dir.mkdir(parents=True)
        (profile_dir / "summary.yaml").write_text(
            json.dumps({"elapsed_s": 1.0, "counters": []}),
            encoding="utf-8",
        )

    monkeypatch.setattr(
        "scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip.subprocess.run",
        fake_run,
    )

    result = _run_metric(
        metric="roundtrip",
        args=args,
        overrides=("device=cpu",),
        candidate=candidate,
    )

    assert result["metric"] == "roundtrip"
    assert result["score"]["generated_tokens"] == 5
    assert "steering_metric=roundtrip" in result["command"]
    assert "+profile.enabled=true" in result["command"]
    assert any(value.startswith("+profile.output_dir=") for value in result["command"])


def test_metric_selection_controls_subprocesses_and_comparison(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.append(command)
        _install_fake_metric_outputs(command)

    monkeypatch.setattr(
        "scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip.subprocess.run",
        fake_run,
    )
    output_root = tmp_path / "eval"
    run_tag = "existing"
    candidate_path = output_root / run_tag / "candidates" / "candidates.parquet"
    candidate_path.parent.mkdir(parents=True)
    pq.write_table(pa.table({"candidate_id": ["candidate-a"]}), candidate_path)

    single_dir = tmp_path / "single"
    single = run_benchmark(
        _benchmark_args(
            output_root=output_root,
            run_tag=run_tag,
            benchmark_dir=single_dir,
            metrics="roundtrip",
        ),
        (),
    )

    assert len(captured) == 1
    assert "steering_metric=roundtrip" in captured[0]
    assert single["baseline_comparison"] == {"skipped": "single-metric run"}
    assert "sample" not in single
    assert "run_dir" not in single["metrics"][0]
    assert json.loads((single_dir / "benchmark.json").read_text(encoding="utf-8"))[
        "baseline_comparison"
    ] == {"skipped": "single-metric run"}

    captured.clear()
    both = run_benchmark(
        _benchmark_args(
            output_root=output_root,
            run_tag=run_tag,
            benchmark_dir=tmp_path / "both",
            metrics="both",
        ),
        (),
    )

    assert len(captured) == 2
    assert [
        next(value for value in command if value.startswith("steering_metric="))
        for command in captured
    ] == ["steering_metric=bpb_class_margin", "steering_metric=roundtrip"]
    assert both["baseline_comparison"]["within_tolerance"] is True


def test_sampling_routes_metric_to_derived_output_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: list[list[str]] = []

    def fake_run(command: list[str], *, check: bool) -> None:
        assert check is True
        captured.append(command)
        _install_fake_metric_outputs(command)

    monkeypatch.setattr(
        "scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip.subprocess.run",
        fake_run,
    )
    output_root = tmp_path / "eval"
    run_tag = "existing"
    candidate_path = output_root / run_tag / "candidates" / "candidates.parquet"
    candidate_path.parent.mkdir(parents=True)
    table = _candidate_table([("method", "target", "mode", 10)])
    pq.write_table(table, candidate_path)
    (output_root / run_tag / "source_contract.json").write_text(
        "{}\n",
        encoding="utf-8",
    )
    benchmark_dir = tmp_path / "benchmark"

    report = run_benchmark(
        _benchmark_args(
            output_root=output_root,
            run_tag=run_tag,
            benchmark_dir=benchmark_dir,
            metrics="roundtrip",
            sample_size=4,
            sample_seed=3,
        ),
        ("--config-name=foo", "device=cpu"),
    )

    assert len(captured) == 1
    expected_output_root = benchmark_dir.resolve() / "derived" / "roundtrip"
    assert f"output_root={expected_output_root}" in captured[0]
    config_index = captured[0].index("--config-name=foo")
    device_index = captured[0].index("device=cpu")
    stage_index = captured[0].index("stage=score")
    output_index = captured[0].index(f"output_root={expected_output_root}")
    assert config_index < device_index < stage_index < output_index
    derived_candidate = expected_output_root / run_tag / "candidates" / "candidates.parquet"
    assert pq.read_table(derived_candidate).num_rows == 4
    assert report["metrics"][0]["run_dir"] == str(
        (expected_output_root / run_tag).resolve()
    )
    assert report["sample"]["sample_size"] == 4
    assert report["sample"]["numpy_version"] == benchmark_module.np.__version__
    assert report["sample"]["pyarrow_version"] == pa.__version__
    assert report["sample"]["subset_sha256"] == hashlib.sha256(
        derived_candidate.read_bytes()
    ).hexdigest()


def test_sampling_detects_source_candidate_mutation_during_subprocess(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "existing"
    candidate_path = output_root / run_tag / "candidates" / "candidates.parquet"
    candidate_path.parent.mkdir(parents=True)
    table = _candidate_table([("method", "target", "mode", 10)])
    pq.write_table(table, candidate_path)

    def mutate_source(command: list[str], *, check: bool) -> None:
        assert command
        assert check is True
        mutated = table.set_column(
            table.schema.get_field_index("candidate_id"),
            "candidate_id",
            pa.array([f"mutated-{index}" for index in range(table.num_rows)]),
        )
        pq.write_table(mutated, candidate_path)

    monkeypatch.setattr(
        "scripts.benchmark.benchmark_axbench_bpb_adv_roundtrip.subprocess.run",
        mutate_source,
    )

    with pytest.raises(RuntimeError, match="production state was mutated"):
        run_benchmark(
            _benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=tmp_path / "benchmark",
                metrics="roundtrip",
                sample_size=4,
            ),
            (),
        )


def test_subprocess_failure_propagates_when_fingerprints_are_clean(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "existing"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    pq.write_table(pa.table({"candidate_id": ["candidate-a"]}), paths.candidates_path())
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    def fail(command: list[str], *, check: bool) -> None:
        assert check is True
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(benchmark_module.subprocess, "run", fail)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_metric(
            metric="roundtrip",
            args=_benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=benchmark_dir,
                metrics="roundtrip",
            ),
            overrides=(),
            candidate=candidate_fingerprint(paths.candidates_path()),
        )

    assert exc_info.value.returncode == 17
    assert getattr(exc_info.value, "__notes__", []) == []


def test_subprocess_failure_remains_primary_when_source_is_mutated(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "existing"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    pq.write_table(pa.table({"candidate_id": ["candidate-a"]}), paths.candidates_path())
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    def mutate_and_fail(command: list[str], *, check: bool) -> None:
        assert check is True
        pq.write_table(
            pa.table({"candidate_id": ["mutated-candidate"]}),
            paths.candidates_path(),
        )
        raise subprocess.CalledProcessError(23, command)

    monkeypatch.setattr(benchmark_module.subprocess, "run", mutate_and_fail)

    with pytest.raises(subprocess.CalledProcessError) as exc_info:
        _run_metric(
            metric="roundtrip",
            args=_benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=benchmark_dir,
                metrics="roundtrip",
            ),
            overrides=(),
            candidate=candidate_fingerprint(paths.candidates_path()),
        )

    assert exc_info.value.returncode == 23
    notes = getattr(exc_info.value, "__notes__", [])
    assert any("production state was mutated" in note for note in notes)


@pytest.mark.parametrize(
    ("guard", "message"),
    [
        ("inode", "derived candidates table unexpectedly shares its inode"),
        ("row-count", "derived candidates row count mismatch"),
        ("source-before-run", "source candidate table changed while constructing derived run"),
    ],
)
def test_sampled_derived_run_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard: str,
    message: str,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "existing"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    table = _candidate_table([("method", "target", "mode", 10)])
    pq.write_table(table, paths.candidates_path())
    paths.source_contract().write_text("{}\n", encoding="utf-8")
    real_write = benchmark_module._write_subset_table_atomic

    def guarded_write(
        subset: pa.Table,
        destination: Path,
        *,
        source_schema: pa.Schema,
    ) -> None:
        if guard == "inode":
            destination.parent.mkdir(parents=True, exist_ok=True)
            os.link(paths.candidates_path(), destination)
            return
        if guard == "row-count":
            destination.parent.mkdir(parents=True, exist_ok=True)
            pq.write_table(subset.slice(0, 1), destination)
            return
        real_write(subset, destination, source_schema=source_schema)
        mutated = table.set_column(
            table.schema.get_field_index("candidate_id"),
            "candidate_id",
            pa.array([f"changed-{index}" for index in range(table.num_rows)]),
        )
        pq.write_table(mutated, paths.candidates_path())

    monkeypatch.setattr(benchmark_module, "_write_subset_table_atomic", guarded_write)

    with pytest.raises(RuntimeError, match=message):
        run_benchmark(
            _benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=tmp_path / "benchmark",
                metrics="roundtrip",
                sample_size=4,
            ),
            (),
        )


def test_cross_metric_subset_hash_guard_detects_rewrite_between_metrics(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "existing"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    pq.write_table(
        _candidate_table([("method", "target", "mode", 10)]),
        paths.candidates_path(),
    )
    paths.source_contract().write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module.subprocess,
        "run",
        lambda command, *, check: _install_fake_metric_outputs(command),
    )
    real_run_metric = benchmark_module._run_metric

    def rewrite_first_derived_subset(**kwargs: Any) -> dict[str, Any]:
        result = real_run_metric(**kwargs)
        if kwargs["metric"] == "bpb_class_margin":
            derived_path = Path(str(result["candidate_fingerprint"]["path"]))
            derived = pq.read_table(derived_path)
            candidate_ids = derived.column("candidate_id").to_pylist()
            rewritten = derived.set_column(
                derived.schema.get_field_index("candidate_id"),
                "candidate_id",
                pa.array([f"rewritten-{value}" for value in candidate_ids]),
            )
            pq.write_table(rewritten, derived_path)
            result["candidate_fingerprint"] = candidate_fingerprint(derived_path).as_dict()
        return result

    monkeypatch.setattr(benchmark_module, "_run_metric", rewrite_first_derived_subset)

    with pytest.raises(
        RuntimeError,
        match="derived candidate subset parquet bytes differ across metrics",
    ):
        run_benchmark(
            _benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=tmp_path / "benchmark",
                metrics="both",
                sample_size=4,
            ),
            (),
        )


def test_candidate_fingerprint_rejects_missing_path(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="candidate table does not exist"):
        candidate_fingerprint(tmp_path / "missing.parquet")


def test_candidate_fingerprint_requires_candidate_id(tmp_path: Path) -> None:
    path = tmp_path / "missing-id.parquet"
    pq.write_table(pa.table({"other": ["value"]}), path)

    with pytest.raises(ValueError, match="candidate table lacks candidate_id"):
        candidate_fingerprint(path)


def test_allocation_reports_inconsistent_capacity_exhaustion() -> None:
    strata = {
        ("a", "target", "mode"): [0],
        ("b", "target", "mode"): [1],
    }

    with pytest.raises(
        AssertionError,
        match="largest-remainder allocation exhausted stratum capacity",
    ):
        _largest_remainder_allocation(strata, sample_size=3, table_rows=100)


def test_sampler_detects_schema_drift_before_write() -> None:
    table = _candidate_table(
        [("method", "target", "mode", 4)],
        metadata={b"fixture": b"preserve"},
    )

    class MetadataStrippingTable:
        def __init__(self, wrapped: pa.Table) -> None:
            self._wrapped = wrapped

        @property
        def column_names(self) -> list[str]:
            return self._wrapped.column_names

        @property
        def num_rows(self) -> int:
            return self._wrapped.num_rows

        @property
        def schema(self) -> pa.Schema:
            return self._wrapped.schema

        def column(self, name: str) -> pa.ChunkedArray:
            return self._wrapped.column(name)

        def take(self, indices: pa.Array) -> pa.Table:
            return self._wrapped.take(indices).replace_schema_metadata(None)

    with pytest.raises(RuntimeError, match="schema drifted before write"):
        sample_candidates(
            MetadataStrippingTable(table),  # type: ignore[arg-type]
            sample_size=2,
            sample_seed=0,
        )


def test_subset_write_detects_post_write_metadata_drift(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    metadata = {
        FEATURE_CONTRACT_METADATA_KEY: b"feature",
        SOURCE_CONTRACT_METADATA_KEY: b"source",
        ARTIFACT_FAMILY_METADATA_KEY: b"candidates",
        PRECOMPUTED_BAG_SET_METADATA_KEY: b"before",
    }
    table = _candidate_table(
        [("method", "target", "mode", 2)],
        metadata=metadata,
    )
    observed_metadata = dict(metadata)
    observed_metadata[PRECOMPUTED_BAG_SET_METADATA_KEY] = b"after"
    monkeypatch.setattr(
        benchmark_module.pq,
        "read_schema",
        lambda _path: table.schema.with_metadata(observed_metadata),
    )

    with pytest.raises(RuntimeError, match="schema drifted during parquet write"):
        _write_subset_table_atomic(
            table,
            tmp_path / "subset.parquet",
            source_schema=table.schema,
        )


def test_subset_schema_fidelity_detects_column_drift() -> None:
    source = pa.schema([("candidate_id", pa.string())])
    observed = pa.schema([("renamed_candidate_id", pa.string())])

    with pytest.raises(RuntimeError, match="including columns"):
        _assert_subset_schema_fidelity(
            observed,
            source_schema=source,
            surface="in fixture",
        )


def test_clustered_sample_restamps_subset_bag_set(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "clustered"
    source_paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    source_paths.candidates_path().parent.mkdir(parents=True)
    source = _stamped_clustered_candidate_table(source_paths)
    pq.write_table(source, source_paths.candidates_path())
    source_paths.source_contract().write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr(
        benchmark_module.subprocess,
        "run",
        lambda command, *, check: _install_fake_metric_outputs(command),
    )
    benchmark_dir = tmp_path / "benchmark"

    report = run_benchmark(
        _benchmark_args(
            output_root=output_root,
            run_tag=run_tag,
            benchmark_dir=benchmark_dir,
            metrics="roundtrip",
            sample_size=2,
        ),
        (),
    )

    derived_paths = AxBenchBpbAdvPaths(
        output_root=benchmark_dir.resolve() / "derived" / "roundtrip",
        run_tag=run_tag,
    )
    derived = pq.read_table(derived_paths.candidates_path())
    source_metadata = dict(source.schema.metadata or {})
    derived_metadata = dict(derived.schema.metadata or {})
    recomputed = restamp_candidates_bag_set(derived, paths=derived_paths)
    recomputed_metadata = dict(recomputed.schema.metadata or {})
    for key in (
        FEATURE_CONTRACT_METADATA_KEY,
        SOURCE_CONTRACT_METADATA_KEY,
        ARTIFACT_FAMILY_METADATA_KEY,
    ):
        assert derived_metadata[key] == source_metadata[key]
    assert derived_metadata[PRECOMPUTED_BAG_SET_METADATA_KEY]
    assert (
        derived_metadata[PRECOMPUTED_BAG_SET_METADATA_KEY]
        == recomputed_metadata[PRECOMPUTED_BAG_SET_METADATA_KEY]
    )
    assert report["sample"]["sample_size"] == 2


@pytest.mark.parametrize(
    "mutation",
    ["missing-reserved-key", "empty-bag-set", "content-independent-mismatch"],
)
def test_clustered_subset_fidelity_guard_rejects_restamp_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
) -> None:
    output_root = tmp_path / mutation / "eval"
    run_tag = "clustered"
    source_paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    source_paths.candidates_path().parent.mkdir(parents=True)
    source = _stamped_clustered_candidate_table(source_paths)
    pq.write_table(source, source_paths.candidates_path())
    source_paths.source_contract().write_text("{}\n", encoding="utf-8")
    real_restamp = benchmark_module.restamp_candidates_bag_set

    def mutated_restamp(
        table: pa.Table,
        *,
        paths: AxBenchBpbAdvPaths,
    ) -> pa.Table:
        restamped = real_restamp(table, paths=paths)
        metadata = dict(restamped.schema.metadata or {})
        if mutation == "missing-reserved-key":
            metadata.pop(FEATURE_CONTRACT_METADATA_KEY)
        elif mutation == "empty-bag-set":
            metadata[PRECOMPUTED_BAG_SET_METADATA_KEY] = b""
        else:
            metadata[SOURCE_CONTRACT_METADATA_KEY] = b"mutated"
        return restamped.replace_schema_metadata(metadata)

    monkeypatch.setattr(
        benchmark_module,
        "restamp_candidates_bag_set",
        mutated_restamp,
    )

    with pytest.raises(RuntimeError, match="schema drifted before write.*metadata"):
        run_benchmark(
            _benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=tmp_path / mutation / "benchmark",
                metrics="roundtrip",
                sample_size=2,
            ),
            (),
        )


def test_profile_counter_guards() -> None:
    assert _counter_totals(
        {"counters": [{"name": "prefill_tokens", "total": "3.5"}]}
    ) == {"prefill_tokens": 3.5}
    with pytest.raises(ValueError, match="profile summary counters must be a list"):
        _counter_totals({"counters": {"name": "counter", "total": 1}})
    with pytest.raises(ValueError, match="malformed profile counter row"):
        _counter_totals({"counters": [{"name": "counter"}]})


def test_regression_sentinel_reports_missing_counters() -> None:
    with pytest.raises(ValueError, match="counters are missing.*prefill_tokens"):
        validate_regression_sentinel(
            {"engine_constructions": 1},
            {"engine_constructions": 2},
            required_increases=("engine_constructions", "prefill_tokens"),
        )


def test_load_profile_summary_guards(tmp_path: Path) -> None:
    profile_dir = tmp_path / "profile"
    with pytest.raises(FileNotFoundError, match="profile summary does not exist"):
        _load_profile_summary(profile_dir)

    profile_dir.mkdir()
    (profile_dir / "summary.yaml").write_text("[]\n", encoding="utf-8")
    with pytest.raises(ValueError, match="profile summary is not a mapping"):
        _load_profile_summary(profile_dir)


@pytest.mark.parametrize("metric", ["bpb_class_margin", "roundtrip"])
def test_score_table_summary_rejects_wrong_schema(
    tmp_path: Path,
    metric: str,
) -> None:
    path = tmp_path / f"{metric}.parquet"
    pq.write_table(pa.table({"wrong": [1]}), path)

    with pytest.raises(ValueError, match=rf"{metric} score schema mismatch"):
        _score_table_summary(metric, path)


@pytest.mark.parametrize("wrong_table", ["class-margin", "roundtrip"])
def test_compare_roundtrip_baselines_rejects_wrong_schemas(
    tmp_path: Path,
    wrong_table: str,
) -> None:
    class_path = tmp_path / "class.parquet"
    roundtrip_path = tmp_path / "roundtrip.parquet"
    if wrong_table == "class-margin":
        pq.write_table(pa.table({"wrong": [1]}), class_path)
        _write_table(
            roundtrip_path,
            ROUNDTRIP_SCORE_SCHEMA,
            [_score_row(ROUNDTRIP_SCORE_SCHEMA, "candidate", 1.0, 2.0)],
        )
        expected = "class-margin benchmark output has the wrong schema"
    else:
        _write_table(
            class_path,
            SCORE_SCHEMA,
            [_score_row(SCORE_SCHEMA, "candidate", 1.0, 2.0)],
        )
        pq.write_table(pa.table({"wrong": [1]}), roundtrip_path)
        expected = "roundtrip benchmark output has the wrong schema"

    with pytest.raises(ValueError, match=expected):
        compare_roundtrip_baselines(class_path, roundtrip_path)


def test_roundtrip_candidate_absent_from_class_table_is_a_mismatch(
    tmp_path: Path,
) -> None:
    class_path = tmp_path / "class.parquet"
    roundtrip_path = tmp_path / "roundtrip.parquet"
    _write_table(
        class_path,
        SCORE_SCHEMA,
        [_score_row(SCORE_SCHEMA, "class-only", 1.0, 2.0)],
    )
    _write_table(
        roundtrip_path,
        ROUNDTRIP_SCORE_SCHEMA,
        [_score_row(ROUNDTRIP_SCORE_SCHEMA, "roundtrip-only", 1.0, 2.0)],
    )

    comparison = compare_roundtrip_baselines(class_path, roundtrip_path)

    assert comparison["within_tolerance"] is False
    assert comparison["mismatches"] == ["roundtrip-only"]
    assert comparison["mismatch_count"] == 1
    assert comparison["roundtrip_missing_candidates"] == 1


def test_subprocess_failure_carries_both_integrity_notes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "run"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    table = _candidate_table([("method", "target", "mode", 4)])
    pq.write_table(table, paths.candidates_path())
    paths.source_contract().write_text("{}\n", encoding="utf-8")
    candidate = candidate_fingerprint(paths.candidates_path())
    sample = sample_candidates(table, sample_size=2, sample_seed=0)
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    def tamper_both_and_fail(command: list[str], *, check: bool) -> None:
        assert check is True
        run_output_root = Path(
            next(value.split("=", 1)[1] for value in command if value.startswith("output_root="))
        )
        derived = AxBenchBpbAdvPaths(
            output_root=run_output_root,
            run_tag=run_tag,
        ).candidates_path()
        pq.write_table(pq.read_table(derived).slice(0, 1), derived)
        pq.write_table(table.slice(0, 1), paths.candidates_path())
        raise subprocess.CalledProcessError(31, command)

    monkeypatch.setattr(benchmark_module.subprocess, "run", tamper_both_and_fail)

    with pytest.raises(
        subprocess.CalledProcessError,
        match="returned non-zero exit status 31",
    ) as exc_info:
        _run_metric(
            metric="roundtrip",
            args=_benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=benchmark_dir,
                metrics="roundtrip",
            ),
            overrides=(),
            candidate=candidate,
            sample=sample,
        )

    notes = getattr(exc_info.value, "__notes__", [])
    assert len(notes) == 2
    assert any("candidate table changed during benchmark" in note for note in notes)
    assert any("production state was mutated" in note for note in notes)


def test_two_integrity_failures_attach_the_second_to_the_first(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "run"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    table = _candidate_table([("method", "target", "mode", 4)])
    pq.write_table(table, paths.candidates_path())
    paths.source_contract().write_text("{}\n", encoding="utf-8")
    candidate = candidate_fingerprint(paths.candidates_path())
    sample = sample_candidates(table, sample_size=2, sample_seed=0)
    benchmark_dir = tmp_path / "benchmark"
    benchmark_dir.mkdir()

    def tamper_both(command: list[str], *, check: bool) -> None:
        assert check is True
        run_output_root = Path(
            next(value.split("=", 1)[1] for value in command if value.startswith("output_root="))
        )
        derived = AxBenchBpbAdvPaths(
            output_root=run_output_root,
            run_tag=run_tag,
        ).candidates_path()
        pq.write_table(pq.read_table(derived).slice(0, 1), derived)
        pq.write_table(table.slice(0, 1), paths.candidates_path())

    monkeypatch.setattr(benchmark_module.subprocess, "run", tamper_both)

    with pytest.raises(
        RuntimeError,
        match="candidate table changed during benchmark",
    ) as exc_info:
        _run_metric(
            metric="roundtrip",
            args=_benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=benchmark_dir,
                metrics="roundtrip",
            ),
            overrides=(),
            candidate=candidate,
            sample=sample,
        )

    notes = getattr(exc_info.value, "__notes__", [])
    assert len(notes) == 1
    assert "production state was mutated" in notes[0]


def test_sampling_detects_source_change_during_initial_read(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "run"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    table = _candidate_table([("method", "target", "mode", 4)])
    pq.write_table(table, paths.candidates_path())
    real_read_table = benchmark_module.pq.read_table
    reads = 0

    def rewrite_after_first_read(path: Path, *args: Any, **kwargs: Any) -> pa.Table:
        nonlocal reads
        observed = real_read_table(path, *args, **kwargs)
        reads += 1
        if reads == 1:
            mutated = table.set_column(
                table.schema.get_field_index("candidate_id"),
                "candidate_id",
                pa.array([f"mutated-{index}" for index in range(table.num_rows)]),
            )
            pq.write_table(mutated, paths.candidates_path())
        return observed

    monkeypatch.setattr(benchmark_module.pq, "read_table", rewrite_after_first_read)

    with pytest.raises(RuntimeError, match="changed while resolving benchmark sample"):
        run_benchmark(
            _benchmark_args(
                output_root=output_root,
                run_tag=run_tag,
                benchmark_dir=tmp_path / "benchmark",
                metrics="roundtrip",
                sample_size=2,
            ),
            (),
        )


@pytest.mark.parametrize("value", ["0", "-3"])
def test_positive_int_rejects_nonpositive_values(value: str) -> None:
    with pytest.raises(argparse.ArgumentTypeError, match="greater than zero"):
        _positive_int(value)


def test_positive_int_accepts_positive_value() -> None:
    assert _positive_int("7") == 7


def test_main_prints_sorted_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    report = {"z": 1, "a": {"value": 2}}
    monkeypatch.setattr(benchmark_module, "run_benchmark", lambda _args, _overrides: report)

    benchmark_module.main(
        [
            "--output-root",
            str(tmp_path / "eval"),
            "--run-tag",
            "run",
            "--benchmark-dir",
            str(tmp_path / "benchmark"),
        ]
    )

    assert capsys.readouterr().out == json.dumps(report, indent=2, sort_keys=True) + "\n"


def test_real_subprocess_runs_hermetic_production_cli_stub(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = tmp_path / "eval"
    run_tag = "run"
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag=run_tag)
    paths.candidates_path().parent.mkdir(parents=True)
    pq.write_table(pa.table({"candidate_id": ["candidate-a"]}), paths.candidates_path())
    stub = tmp_path / "production_cli_stub.py"
    stub.write_text(
        f"#!{sys.executable}\n"
        "import json\n"
        "import sys\n"
        "from pathlib import Path\n"
        "import pyarrow as pa\n"
        "import pyarrow.parquet as pq\n"
        "from steering.eval.axbench_bpb_adv.schemas import "
        "ROUNDTRIP_SCORE_SCHEMA, SCORE_SCHEMA\n"
        "argv = sys.argv[1:]\n"
        "assert argv[:2] == ['-m', 'scripts.eval.axbench_bpb_adv_eval']\n"
        "assert 'functional_override=sentinel' in argv\n"
        "def value(prefix):\n"
        "    return next(item.split('=', 1)[1] for item in argv if item.startswith(prefix))\n"
        "metric = value('steering_metric=')\n"
        "schema = ROUNDTRIP_SCORE_SCHEMA if metric == 'roundtrip' else SCORE_SCHEMA\n"
        "row = {}\n"
        "for field in schema:\n"
        "    if pa.types.is_list(field.type): row[field.name] = []\n"
        "    elif pa.types.is_boolean(field.type): row[field.name] = False\n"
        "    elif pa.types.is_integer(field.type): row[field.name] = 0\n"
        "    elif pa.types.is_floating(field.type): row[field.name] = 0.0\n"
        "    elif pa.types.is_string(field.type): row[field.name] = ''\n"
        "row['candidate_id'] = 'candidate-a'\n"
        "row['loss_concept_rich'] = 1.0\n"
        "row['loss_concept_avoiding'] = 2.0\n"
        "score_path = Path(value('scores_path='))\n"
        "score_path.parent.mkdir(parents=True, exist_ok=True)\n"
        "pq.write_table(pa.Table.from_pylist([row], schema=schema), score_path)\n"
        "profile_dir = Path(value('+profile.output_dir='))\n"
        "profile_dir.mkdir(parents=True, exist_ok=True)\n"
        "(profile_dir / 'summary.yaml').write_text("
        "json.dumps({'elapsed_s': 0.5, 'counters': []}), encoding='utf-8')\n",
        encoding="utf-8",
    )
    stub.chmod(0o755)
    monkeypatch.setenv("PYTHONPATH", str(Path.cwd()))
    benchmark_dir = tmp_path / "benchmark"
    args = _benchmark_args(
        output_root=output_root,
        run_tag=run_tag,
        benchmark_dir=benchmark_dir,
        metrics="both",
    )
    args.python = str(stub)

    report = run_benchmark(args, ("functional_override=sentinel",))

    persisted = json.loads((benchmark_dir / "benchmark.json").read_text(encoding="utf-8"))
    assert persisted == report
    assert report["candidate"]["rows"] == 1
    assert len(report["candidate"]["sha256"]) == 64
    assert [result["metric"] for result in report["metrics"]] == [
        "bpb_class_margin",
        "roundtrip",
    ]
    assert all(result["score"]["rows"] == 1 for result in report["metrics"])
    assert report["baseline_comparison"]["within_tolerance"] is True
    assert report["metrics"][0]["command"][:3] == [
        str(stub),
        "-m",
        "scripts.eval.axbench_bpb_adv_eval",
    ]
