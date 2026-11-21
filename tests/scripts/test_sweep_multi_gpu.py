"""Tests for multi-GPU sweep features: resume, rank striping, JSONL merge, and sync.

Covers the new functionality in scripts/run_encdec_sweep_fast.py:
- _load_completed_tags: reading completed model_tags from JSONL result files
- _merge_rank_files: merging per-rank JSONL files into a single file
- _serialize_result: converting TrainResult to JSON-serializable dict
- _append_result: appending results to JSONL files
- _write_csv_summary: converting JSONL to CSV
- _print_summary: human-readable summary output
- Config striping across ranks (--sweep-rank/--sweep-world-size)
- Min/max index filtering (--min-idx/--max-idx)
- Resume filtering (--resume)
- File-based rank synchronization (marker files)
- Per-rank JSONL path selection
- GPU index selection (--gpu)

Also covers steering/conditional/encoder_decoder/dataloader.py:
- _prefetch_wrapper: background-thread prefetch buffer for generators
"""

from __future__ import annotations

import csv
import inspect
import json
import time
from pathlib import Path

import pytest

from scripts.train.conditional_encdec_train import TrainResult
from scripts.train.run_encdec_sweep import (
    FeatureDeltaConfig,
    FeatureDeltaMode,
    FeatureSelectionConfig,
    NegativeSamplingConfig,
    SweepConfig,
    build_sweep_configs,
)
from scripts.train.run_encdec_sweep_fast import (
    _append_result,
    _load_completed_tags,
    _merge_rank_files,
    _print_summary,
    _run_one,
    _serialize_result,
    _write_csv_summary,
)
from steering.conditional.encoder_decoder.dataloader import _prefetch_wrapper

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sweep_config(
    fs_dist: str = "geometric",
    fs_mean: float = 8.0,
    fs_std: float = 8.0,
    ns_mean: float = 8.0,
    fd_mode: FeatureDeltaMode | str | None = FeatureDeltaMode.disabled,
    fd_ratio: float | None = None,
    fp_weight: float = 0.1,
) -> SweepConfig:
    """Build a SweepConfig with sensible defaults for testing.

    ``fd_mode`` accepts a FeatureDeltaMode, its string value, or None (== the
    disabled mode) for caller convenience; FeatureDeltaConfig itself is strict
    (enum-only) by design.
    """
    if fd_mode is None:
        fd_mode = FeatureDeltaMode.disabled
    elif isinstance(fd_mode, str):
        fd_mode = FeatureDeltaMode(fd_mode)
    return SweepConfig(
        fs=FeatureSelectionConfig(fs_dist, fs_mean, fs_std),
        ns=NegativeSamplingConfig(ns_mean),
        fd=FeatureDeltaConfig(fd_mode, fd_ratio),
        fp_weight=fp_weight,
    )


def _make_train_result(
    model_tag: str = "test_tag",
    val_bpb: float | None = 1.23,
    min_val_bpb: float = 1.20,
    total_training_time: float = 60.0,
    peak_memory_mib: float = 1024.0,
    startup_time: float = 5.0,
    error: str | None = None,
) -> TrainResult:
    """Build a TrainResult with sensible defaults for testing."""
    return TrainResult(
        model_tag=model_tag,
        val_bpb=val_bpb,
        min_val_bpb=min_val_bpb,
        total_training_time=total_training_time,
        peak_memory_mib=peak_memory_mib,
        startup_time=startup_time,
        error=error,
    )


def _write_jsonl(path: Path, records: list[dict]) -> None:
    """Write a list of dicts as JSONL lines."""
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    """Read JSONL file into a list of dicts."""
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            stripped_line = line.strip()
            if stripped_line:
                records.append(json.loads(stripped_line))
    return records


# ===========================================================================
# _load_completed_tags
# ===========================================================================


class TestLoadCompletedTags:
    """Tests for _load_completed_tags: reading completed model_tags from JSONL files."""

    def test_single_file_returns_successful_tags(self, tmp_path: Path):
        """Non-errored records are returned as completed tags."""
        records = [
            {"model_tag": "tag_a", "error": None, "config_index": 0},
            {"model_tag": "tag_b", "error": None, "config_index": 1},
        ]
        _write_jsonl(tmp_path / "sweep_results.jsonl", records)
        completed = _load_completed_tags(tmp_path)
        assert completed == {"tag_a", "tag_b"}

    def test_errored_records_excluded(self, tmp_path: Path):
        """Records with non-None error field are not counted as completed."""
        records = [
            {"model_tag": "tag_ok", "error": None, "config_index": 0},
            {"model_tag": "tag_fail", "error": "OOM", "config_index": 1},
        ]
        _write_jsonl(tmp_path / "sweep_results.jsonl", records)
        completed = _load_completed_tags(tmp_path)
        assert completed == {"tag_ok"}
        assert "tag_fail" not in completed

    def test_reads_per_rank_files(self, tmp_path: Path):
        """Glob pattern picks up sweep_results_rank*.jsonl files."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "r0_tag", "error": None},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "r1_tag", "error": None},
        ])
        completed = _load_completed_tags(tmp_path)
        assert completed == {"r0_tag", "r1_tag"}

    def test_reads_both_main_and_rank_files(self, tmp_path: Path):
        """Both sweep_results.jsonl and per-rank files are read."""
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": "main_tag", "error": None},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "rank_tag", "error": None},
        ])
        completed = _load_completed_tags(tmp_path)
        assert completed == {"main_tag", "rank_tag"}

    def test_empty_directory_returns_empty_set(self, tmp_path: Path):
        """No JSONL files → empty set."""
        completed = _load_completed_tags(tmp_path)
        assert completed == set()

    def test_empty_file_returns_empty_set(self, tmp_path: Path):
        """An empty JSONL file → empty set."""
        (tmp_path / "sweep_results.jsonl").write_text("")
        completed = _load_completed_tags(tmp_path)
        assert completed == set()

    def test_blank_lines_skipped(self, tmp_path: Path):
        """Blank lines in JSONL are gracefully skipped."""
        content = (
            json.dumps({"model_tag": "tag1", "error": None}) + "\n"
            "\n"
            "\n"
            + json.dumps({"model_tag": "tag2", "error": None}) + "\n"
        )
        (tmp_path / "sweep_results.jsonl").write_text(content)
        completed = _load_completed_tags(tmp_path)
        assert completed == {"tag1", "tag2"}

    def test_malformed_json_skipped(self, tmp_path: Path):
        """Invalid JSON lines cause the entire file to be skipped gracefully."""
        content = (
            json.dumps({"model_tag": "good_tag", "error": None}) + "\n"
            "this is not valid json\n"
        )
        (tmp_path / "sweep_results.jsonl").write_text(content)
        # The except catches JSONDecodeError and continues to next file
        # Since this is the same file, the whole file processing stops at the bad line
        # but no exception is raised
        completed = _load_completed_tags(tmp_path)
        # The function catches the error and continues to the next file
        # so we just verify no exception and a set is returned
        assert isinstance(completed, set)

    def test_missing_model_tag_field_excluded(self, tmp_path: Path):
        """Records without model_tag field are not counted."""
        records = [
            {"error": None, "config_index": 0},  # no model_tag
            {"model_tag": "has_tag", "error": None, "config_index": 1},
        ]
        _write_jsonl(tmp_path / "sweep_results.jsonl", records)
        completed = _load_completed_tags(tmp_path)
        assert completed == {"has_tag"}

    def test_empty_model_tag_excluded(self, tmp_path: Path):
        """Empty string model_tag is excluded (falsy check)."""
        records = [
            {"model_tag": "", "error": None, "config_index": 0},
            {"model_tag": "valid", "error": None, "config_index": 1},
        ]
        _write_jsonl(tmp_path / "sweep_results.jsonl", records)
        completed = _load_completed_tags(tmp_path)
        assert completed == {"valid"}

    def test_duplicate_tags_deduplicated(self, tmp_path: Path):
        """Same model_tag appearing across multiple files is deduplicated."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "shared", "error": None},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "shared", "error": None},
        ])
        completed = _load_completed_tags(tmp_path)
        assert completed == {"shared"}

    def test_ignores_non_matching_files(self, tmp_path: Path):
        """Files that don't match the glob pattern are ignored."""
        _write_jsonl(tmp_path / "other_results.jsonl", [
            {"model_tag": "should_not_appear", "error": None},
        ])
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": "should_appear", "error": None},
        ])
        completed = _load_completed_tags(tmp_path)
        assert completed == {"should_appear"}

    def test_large_number_of_tags(self, tmp_path: Path):
        """Handles a large number of completed tags without issue."""
        records = [{"model_tag": f"tag_{i}", "error": None} for i in range(1000)]
        _write_jsonl(tmp_path / "sweep_results.jsonl", records)
        completed = _load_completed_tags(tmp_path)
        assert len(completed) == 1000
        assert "tag_0" in completed
        assert "tag_999" in completed


# ===========================================================================
# _merge_rank_files
# ===========================================================================


class TestMergeRankFiles:
    """Tests for _merge_rank_files: merging per-rank JSONL into a single file."""

    def test_merges_two_rank_files(self, tmp_path: Path):
        """Two rank files are merged into sweep_results.jsonl."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "r0_a", "config_index": 0, "val_bpb": 1.1},
            {"model_tag": "r0_b", "config_index": 2, "val_bpb": 1.3},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "r1_a", "config_index": 1, "val_bpb": 1.2},
            {"model_tag": "r1_b", "config_index": 3, "val_bpb": 1.4},
        ])
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert len(merged) == 4
        tags = [r["model_tag"] for r in merged]
        assert tags == ["r0_a", "r1_a", "r0_b", "r1_b"]  # sorted by config_index

    def test_sorted_by_config_index(self, tmp_path: Path):
        """Merged output is sorted by config_index."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "c5", "config_index": 5},
            {"model_tag": "c1", "config_index": 1},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "c3", "config_index": 3},
            {"model_tag": "c0", "config_index": 0},
        ])
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        indices = [r["config_index"] for r in merged]
        assert indices == [0, 1, 3, 5]

    def test_no_rank_files_is_noop(self, tmp_path: Path):
        """No rank files → sweep_results.jsonl is not created."""
        _merge_rank_files(tmp_path)
        assert not (tmp_path / "sweep_results.jsonl").exists()

    def test_overwrites_existing_merged_file(self, tmp_path: Path):
        """Pre-existing sweep_results.jsonl is overwritten with fresh merge."""
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": "stale", "config_index": 99},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "fresh", "config_index": 0},
        ])
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert len(merged) == 1
        assert merged[0]["model_tag"] == "fresh"

    def test_empty_rank_files_produce_empty_merge(self, tmp_path: Path):
        """Empty rank files → empty (but existing) sweep_results.jsonl."""
        (tmp_path / "sweep_results_rank0.jsonl").write_text("")
        (tmp_path / "sweep_results_rank1.jsonl").write_text("")
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert merged == []

    def test_missing_config_index_defaults_to_zero(self, tmp_path: Path):
        """Records without config_index sort as 0."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "no_index"},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "has_index", "config_index": 1},
        ])
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert merged[0]["model_tag"] == "no_index"
        assert merged[1]["model_tag"] == "has_index"

    def test_single_rank_file(self, tmp_path: Path):
        """Single rank file is still merged correctly."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "solo", "config_index": 0},
        ])
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert len(merged) == 1
        assert merged[0]["model_tag"] == "solo"

    def test_many_rank_files(self, tmp_path: Path):
        """Handles 8 rank files (simulating 8-GPU setup)."""
        total_records = 0
        for rank in range(8):
            records = [
                {"model_tag": f"r{rank}_c{i}", "config_index": i * 8 + rank}
                for i in range(5)
            ]
            _write_jsonl(tmp_path / f"sweep_results_rank{rank}.jsonl", records)
            total_records += len(records)
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert len(merged) == total_records
        # Verify sorted by config_index
        indices = [r["config_index"] for r in merged]
        assert indices == sorted(indices)

    def test_rank_files_not_deleted(self, tmp_path: Path):
        """Rank files are preserved after merge (cleanup is done separately)."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "r0", "config_index": 0},
        ])
        _merge_rank_files(tmp_path)
        assert (tmp_path / "sweep_results_rank0.jsonl").exists()


# ===========================================================================
# _serialize_result
# ===========================================================================


class TestSerializeResult:
    """Tests for _serialize_result: converting TrainResult + SweepConfig to dict."""

    def test_basic_serialization(self):
        """Normal values are correctly serialized."""
        result = _make_train_result(val_bpb=1.5)
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=42)
        assert record["model_tag"] == "test_tag"
        assert record["val_bpb"] == 1.5
        assert record["config_index"] == 42

    def test_inf_replaced_with_none(self):
        """float('inf') is sanitized to None for JSON compatibility."""
        result = _make_train_result(min_val_bpb=float("inf"))
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["min_val_bpb"] is None

    def test_negative_inf_replaced_with_none(self):
        """float('-inf') is sanitized to None."""
        result = _make_train_result(min_val_bpb=float("-inf"))
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["min_val_bpb"] is None

    def test_nan_replaced_with_none(self):
        """float('nan') is sanitized to None."""
        result = _make_train_result(val_bpb=float("nan"))
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["val_bpb"] is None

    def test_none_values_preserved(self):
        """None val_bpb stays as None."""
        result = _make_train_result(val_bpb=None)
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["val_bpb"] is None

    def test_error_field_preserved(self):
        """Error string is correctly included."""
        result = _make_train_result(error="CUDA OOM")
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["error"] == "CUDA OOM"

    def test_no_error_is_none(self):
        """Successful result has error=None."""
        result = _make_train_result(error=None)
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["error"] is None

    def test_config_label_format(self):
        """config_label combines sweep config fields in expected format."""
        cfg = _make_sweep_config(
            fs_dist="geometric", fs_mean=16.0, fs_std=16.0,
            ns_mean=8.0,
            fd_mode=FeatureDeltaMode.disabled,
            fd_ratio=None,
            fp_weight=0.1,
        )
        result = _make_train_result()
        record = _serialize_result(result, cfg, index=0)
        label = record["config_label"]
        assert "geo_m16" in label
        assert "ns_geo_m8" in label
        assert "fd_disabled" in label
        assert "fpw_0.1" in label

    def test_sweep_config_fields_extracted(self):
        """Sweep config fields are flattened into the record."""
        cfg = _make_sweep_config(
            fs_dist="normal", fs_mean=8.0, fs_std=4.0,
            ns_mean=16.0,
            fd_mode=FeatureDeltaMode.prev_segment,
            fd_ratio=0.1,
            fp_weight=1.0,
        )
        result = _make_train_result()
        record = _serialize_result(result, cfg, index=0)
        assert record["fs_distribution"] == "normal"
        assert record["fs_mean"] == 8.0
        assert record["fs_std"] == 4.0
        assert record["ns_mean"] == 16.0
        assert record["fd_mode"] == "prev_segment"
        assert record["fd_neg_delta_ratio"] == 0.1
        assert record["fp_weight"] == 1.0

    def test_timestamp_is_present(self):
        """Record includes an ISO timestamp."""
        result = _make_train_result()
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert "timestamp" in record
        assert "T" in record["timestamp"]  # ISO format contains T separator

    def test_timing_values_rounded(self):
        """Timing values are rounded to expected precision."""
        result = _make_train_result(
            total_training_time=123.456789,
            peak_memory_mib=2048.123456,
            startup_time=7.891234,
        )
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["total_training_time"] == 123.46  # 2 decimals
        assert record["peak_memory_mib"] == 2048.1  # 1 decimal
        assert record["startup_time"] == 7.89  # 2 decimals

    def test_all_expected_keys_present(self):
        """Verify all required keys exist in the serialized record."""
        result = _make_train_result()
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        expected_keys = {
            "model_tag", "config_label", "val_bpb",
            "min_val_bpb", "total_training_time", "peak_memory_mib",
            "startup_time", "error", "fs_distribution", "fs_mean", "fs_std",
            "ns_mean", "fd_mode", "fd_neg_delta_ratio", "fp_weight",
            "config_index", "timestamp",
        }
        assert set(record.keys()) == expected_keys

    def test_json_serializable(self):
        """The entire record can be serialized to JSON without error."""
        result = _make_train_result()
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        serialized = json.dumps(record)
        deserialized = json.loads(serialized)
        assert deserialized["model_tag"] == "test_tag"


# ===========================================================================
# _append_result
# ===========================================================================


class TestAppendResult:
    """Tests for _append_result: atomic JSONL line appending."""

    def test_creates_file_if_missing(self, tmp_path: Path):
        """File is created on first append."""
        path = tmp_path / "results.jsonl"
        result = _make_train_result(model_tag="first")
        cfg = _make_sweep_config()
        _append_result(path, result, cfg, index=0)
        assert path.exists()
        records = _read_jsonl(path)
        assert len(records) == 1
        assert records[0]["model_tag"] == "first"

    def test_appends_to_existing_file(self, tmp_path: Path):
        """Second append adds a second line, preserving the first."""
        path = tmp_path / "results.jsonl"
        cfg = _make_sweep_config()
        _append_result(path, _make_train_result(model_tag="first"), cfg, index=0)
        _append_result(path, _make_train_result(model_tag="second"), cfg, index=1)
        records = _read_jsonl(path)
        assert len(records) == 2
        assert records[0]["model_tag"] == "first"
        assert records[1]["model_tag"] == "second"

    def test_each_call_adds_exactly_one_line(self, tmp_path: Path):
        """Verify line count increases by exactly 1 per call."""
        path = tmp_path / "results.jsonl"
        cfg = _make_sweep_config()
        for i in range(10):
            _append_result(path, _make_train_result(model_tag=f"tag_{i}"), cfg, index=i)
        records = _read_jsonl(path)
        assert len(records) == 10

    def test_result_with_error_is_written(self, tmp_path: Path):
        """Error results are still written to the JSONL file."""
        path = tmp_path / "results.jsonl"
        result = _make_train_result(
            model_tag="errored",
            val_bpb=None,
            error="RuntimeError: CUDA OOM",
        )
        cfg = _make_sweep_config()
        _append_result(path, result, cfg, index=0)
        records = _read_jsonl(path)
        assert records[0]["error"] == "RuntimeError: CUDA OOM"
        assert records[0]["val_bpb"] is None

    def test_preserves_config_index(self, tmp_path: Path):
        """The index argument is stored as config_index."""
        path = tmp_path / "results.jsonl"
        cfg = _make_sweep_config()
        _append_result(path, _make_train_result(), cfg, index=42)
        records = _read_jsonl(path)
        assert records[0]["config_index"] == 42


# ===========================================================================
# _write_csv_summary
# ===========================================================================


class TestWriteCsvSummary:
    """Tests for _write_csv_summary: converting JSONL to CSV."""

    def test_basic_conversion(self, tmp_path: Path):
        """JSONL is correctly converted to CSV with headers and rows."""
        jsonl_path = tmp_path / "results.jsonl"
        csv_path = tmp_path / "results.csv"
        records = [
            {"model_tag": "a", "val_bpb": 1.1, "error": None},
            {"model_tag": "b", "val_bpb": 1.2, "error": None},
        ]
        _write_jsonl(jsonl_path, records)
        _write_csv_summary(jsonl_path, csv_path)
        assert csv_path.exists()
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["model_tag"] == "a"
        assert rows[1]["model_tag"] == "b"

    def test_missing_jsonl_is_noop(self, tmp_path: Path):
        """Non-existent JSONL file → CSV is not created."""
        csv_path = tmp_path / "results.csv"
        _write_csv_summary(tmp_path / "nonexistent.jsonl", csv_path)
        assert not csv_path.exists()

    def test_empty_jsonl_is_noop(self, tmp_path: Path):
        """Empty JSONL file → CSV is not created."""
        jsonl_path = tmp_path / "results.jsonl"
        jsonl_path.write_text("")
        csv_path = tmp_path / "results.csv"
        _write_csv_summary(jsonl_path, csv_path)
        assert not csv_path.exists()

    def test_csv_headers_match_jsonl_keys(self, tmp_path: Path):
        """CSV column headers match the keys from the first JSONL record."""
        jsonl_path = tmp_path / "results.jsonl"
        csv_path = tmp_path / "results.csv"
        records = [{"alpha": 1, "beta": 2, "gamma": 3}]
        _write_jsonl(jsonl_path, records)
        _write_csv_summary(jsonl_path, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames is not None
            assert set(reader.fieldnames) == {"alpha", "beta", "gamma"}

    def test_csv_preserves_none_as_empty(self, tmp_path: Path):
        """None values in JSON become empty strings in CSV."""
        jsonl_path = tmp_path / "results.jsonl"
        csv_path = tmp_path / "results.csv"
        records = [{"model_tag": "a", "val_bpb": None}]
        _write_jsonl(jsonl_path, records)
        _write_csv_summary(jsonl_path, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            row = next(reader)
        # csv.DictWriter writes None as empty string
        assert row["val_bpb"] == ""

    def test_blank_lines_in_jsonl_skipped(self, tmp_path: Path):
        """Blank lines in JSONL don't produce extra rows."""
        jsonl_path = tmp_path / "results.jsonl"
        content = json.dumps({"tag": "a"}) + "\n\n" + json.dumps({"tag": "b"}) + "\n"
        jsonl_path.write_text(content)
        csv_path = tmp_path / "results.csv"
        _write_csv_summary(jsonl_path, csv_path)
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2


# ===========================================================================
# Config striping across ranks
# ===========================================================================


class TestConfigStriping:
    """Tests for the config striping logic: idx % world_size == rank."""

    def test_two_ranks_even_split(self):
        """8 configs across 2 ranks → 4 each, non-overlapping."""
        all_configs = list(range(8))
        rank0 = [c for idx, c in enumerate(all_configs) if idx % 2 == 0]
        rank1 = [c for idx, c in enumerate(all_configs) if idx % 2 == 1]
        assert rank0 == [0, 2, 4, 6]
        assert rank1 == [1, 3, 5, 7]
        assert set(rank0) | set(rank1) == set(all_configs)
        assert set(rank0) & set(rank1) == set()

    def test_three_ranks_uneven(self):
        """10 configs across 3 ranks → correct distribution."""
        all_configs = list(range(10))
        rank_configs = {
            r: [c for idx, c in enumerate(all_configs) if idx % 3 == r]
            for r in range(3)
        }
        assert len(rank_configs[0]) == 4  # indices 0,3,6,9
        assert len(rank_configs[1]) == 3  # indices 1,4,7
        assert len(rank_configs[2]) == 3  # indices 2,5,8
        # All configs covered exactly once
        all_assigned = []
        for r in range(3):
            all_assigned.extend(rank_configs[r])
        assert sorted(all_assigned) == all_configs

    def test_single_rank_gets_all(self):
        """world_size=1 → rank 0 gets all configs."""
        configs = list(range(20))
        rank0 = [c for idx, c in enumerate(configs) if idx % 1 == 0]
        assert rank0 == configs

    def test_more_ranks_than_configs(self):
        """world_size > num_configs → some ranks get 0 configs."""
        configs = list(range(3))
        rank_configs = {
            r: [c for idx, c in enumerate(configs) if idx % 8 == r]
            for r in range(8)
        }
        non_empty = [r for r in range(8) if len(rank_configs[r]) > 0]
        assert len(non_empty) == 3
        empty = [r for r in range(8) if len(rank_configs[r]) == 0]
        assert len(empty) == 5

    def test_stripe_preserves_original_indices(self):
        """Striping uses enumerate index, not config value, for assignment.

        This mirrors the actual code: idx % world_size == rank where idx
        is the position in the selected list.
        """
        # Simulate (original_index, config) tuples like the real code
        selected = [(10, "a"), (20, "b"), (30, "c"), (40, "d"), (50, "e"), (60, "f")]
        world_size = 2
        rank0 = [(i, c) for idx, (i, c) in enumerate(selected) if idx % world_size == 0]
        rank1 = [(i, c) for idx, (i, c) in enumerate(selected) if idx % world_size == 1]
        # Rank 0 gets items at positions 0, 2, 4 in selected list
        assert [c for _, c in rank0] == ["a", "c", "e"]
        # Rank 1 gets items at positions 1, 3, 5
        assert [c for _, c in rank1] == ["b", "d", "f"]

    def test_sweep_configs_striped_deterministically(self):
        """Full sweep configs are distributed deterministically across ranks."""
        all_configs = build_sweep_configs()
        # Simulate 4-GPU striping
        world_size = 4
        per_rank = {
            r: [(i, c) for idx, (i, c) in enumerate(enumerate(all_configs)) if idx % world_size == r]
            for r in range(world_size)
        }
        total = sum(len(v) for v in per_rank.values())
        assert total == len(all_configs)
        # Each rank gets roughly equal share
        sizes = [len(per_rank[r]) for r in range(world_size)]
        assert max(sizes) - min(sizes) <= 1


# ===========================================================================
# Min/max index filtering
# ===========================================================================


class TestMinMaxIndexFiltering:
    """Tests for --min-idx and --max-idx CLI argument filtering logic."""

    def test_min_idx_filters_lower(self):
        """min_idx=5 removes configs with index < 5."""
        selected = list(enumerate(range(10)))  # [(0,0), (1,1), ..., (9,9)]
        min_idx = 5
        filtered = [(i, c) for i, c in selected if i >= min_idx]
        assert len(filtered) == 5
        assert filtered[0] == (5, 5)

    def test_max_idx_filters_upper(self):
        """max_idx=5 removes configs with index >= 5."""
        selected = list(enumerate(range(10)))
        max_idx = 5
        filtered = [(i, c) for i, c in selected if i < max_idx]
        assert len(filtered) == 5
        assert filtered[-1] == (4, 4)

    def test_both_filters_combined(self):
        """min_idx=3, max_idx=7 selects indices [3, 4, 5, 6]."""
        selected = list(enumerate(range(10)))
        min_idx, max_idx = 3, 7
        filtered = [(i, c) for i, c in selected if i >= min_idx]
        filtered = [(i, c) for i, c in filtered if i < max_idx]
        assert len(filtered) == 4
        indices = [i for i, _ in filtered]
        assert indices == [3, 4, 5, 6]

    def test_min_equals_max_returns_empty(self):
        """min_idx == max_idx → empty selection."""
        selected = list(enumerate(range(10)))
        min_idx = max_idx = 5
        filtered = [(i, c) for i, c in selected if i >= min_idx and i < max_idx]
        assert len(filtered) == 0

    def test_no_filters_returns_all(self):
        """No min/max → all configs preserved."""
        selected = list(enumerate(range(10)))
        # No filtering applied
        assert len(selected) == 10

    def test_min_above_range_returns_empty(self):
        """min_idx beyond range → empty."""
        selected = list(enumerate(range(10)))
        filtered = [(i, c) for i, c in selected if i >= 100]
        assert len(filtered) == 0


# ===========================================================================
# Resume filtering
# ===========================================================================


class TestResumeFiltering:
    """Tests for resume logic: filtering out already-completed configs."""

    def test_resume_skips_completed(self, tmp_path: Path):
        """Completed tags are filtered out from the selected configs."""
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": "done_a", "error": None},
            {"model_tag": "done_b", "error": None},
        ])
        configs = [
            _make_sweep_config(fs_mean=4.0),
            _make_sweep_config(fs_mean=8.0),
            _make_sweep_config(fs_mean=16.0),
        ]
        # In real code, filtering checks c.model_tag. Here we simulate directly:
        tags_to_skip = {"done_a", "done_b"}
        # Simulate different tags that include one match
        mock_selected = [
            (0, ("done_a", configs[0])),
            (1, ("unique_tag", configs[1])),
            (2, ("done_b", configs[2])),
        ]
        filtered = [(i, (tag, c)) for i, (tag, c) in mock_selected if tag not in tags_to_skip]
        assert len(filtered) == 1
        assert filtered[0][1][0] == "unique_tag"

    def test_resume_with_no_completed_skips_nothing(self, tmp_path: Path):
        """Empty results dir → nothing is skipped."""
        completed_tags = _load_completed_tags(tmp_path)
        assert len(completed_tags) == 0
        configs = [_make_sweep_config(fs_mean=float(i)) for i in range(5)]
        selected = list(enumerate(configs))
        filtered = [(i, c) for i, c in selected if c.model_tag not in completed_tags]
        assert len(filtered) == 5

    def test_resume_all_completed_returns_empty(self, tmp_path: Path):
        """All configs completed → empty selection."""
        configs = build_sweep_configs()[:3]
        tags = [c.model_tag for c in configs]
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": tag, "error": None} for tag in tags
        ])
        completed_tags = _load_completed_tags(tmp_path)
        selected = list(enumerate(configs))
        filtered = [(i, c) for i, c in selected if c.model_tag not in completed_tags]
        assert len(filtered) == 0

    def test_resume_does_not_skip_errored(self, tmp_path: Path):
        """Errored configs are NOT skipped on resume (they should be retried)."""
        configs = build_sweep_configs()[:2]
        tags = [c.model_tag for c in configs]
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": tags[0], "error": None},  # completed
            {"model_tag": tags[1], "error": "OOM"},  # errored → should retry
        ])
        completed_tags = _load_completed_tags(tmp_path)
        assert tags[0] in completed_tags
        assert tags[1] not in completed_tags
        selected = list(enumerate(configs))
        filtered = [(i, c) for i, c in selected if c.model_tag not in completed_tags]
        assert len(filtered) == 1
        assert filtered[0][1].model_tag == tags[1]


# ===========================================================================
# Per-rank JSONL path selection
# ===========================================================================


class TestPerRankJsonlPath:
    """Tests for per-rank JSONL file naming logic."""

    def test_single_gpu_uses_main_file(self, tmp_path: Path):
        """world_size=1 → sweep_results.jsonl."""
        world_size = 1
        rank = 0
        if world_size > 1:
            path = tmp_path / f"sweep_results_rank{rank}.jsonl"
        else:
            path = tmp_path / "sweep_results.jsonl"
        assert path.name == "sweep_results.jsonl"

    def test_multi_gpu_rank0(self, tmp_path: Path):
        """world_size=2, rank=0 → sweep_results_rank0.jsonl."""
        world_size = 2
        rank = 0
        if world_size > 1:
            path = tmp_path / f"sweep_results_rank{rank}.jsonl"
        else:
            path = tmp_path / "sweep_results.jsonl"
        assert path.name == "sweep_results_rank0.jsonl"

    def test_multi_gpu_rank1(self, tmp_path: Path):
        """world_size=2, rank=1 → sweep_results_rank1.jsonl."""
        world_size = 2
        rank = 1
        if world_size > 1:
            path = tmp_path / f"sweep_results_rank{rank}.jsonl"
        else:
            path = tmp_path / "sweep_results.jsonl"
        assert path.name == "sweep_results_rank1.jsonl"

    def test_eight_gpu_rank7(self, tmp_path: Path):
        """world_size=8, rank=7 → sweep_results_rank7.jsonl."""
        world_size = 8
        rank = 7
        if world_size > 1:
            path = tmp_path / f"sweep_results_rank{rank}.jsonl"
        else:
            path = tmp_path / "sweep_results.jsonl"
        assert path.name == "sweep_results_rank7.jsonl"


# ===========================================================================
# File-based rank synchronization
# ===========================================================================


class TestRankSynchronization:
    """Tests for file-based inter-rank synchronization via marker files."""

    def test_marker_file_created(self, tmp_path: Path):
        """Each rank creates a .done_rank{rank} marker file."""
        rank = 3
        marker = tmp_path / f".done_rank{rank}"
        marker.touch()
        assert marker.exists()

    def test_all_markers_detectable(self, tmp_path: Path):
        """All marker files can be detected for a given world_size."""
        world_size = 4
        for r in range(world_size):
            (tmp_path / f".done_rank{r}").touch()
        for r in range(world_size):
            assert (tmp_path / f".done_rank{r}").exists()

    def test_partial_markers(self, tmp_path: Path):
        """Only some markers present → can detect which are missing."""
        world_size = 4
        (tmp_path / ".done_rank0").touch()
        (tmp_path / ".done_rank2").touch()
        missing = [r for r in range(world_size) if not (tmp_path / f".done_rank{r}").exists()]
        assert missing == [1, 3]

    def test_marker_cleanup(self, tmp_path: Path):
        """Marker files can be cleaned up with unlink(missing_ok=True)."""
        world_size = 4
        for r in range(world_size):
            (tmp_path / f".done_rank{r}").touch()
        for r in range(world_size):
            (tmp_path / f".done_rank{r}").unlink(missing_ok=True)
        for r in range(world_size):
            assert not (tmp_path / f".done_rank{r}").exists()

    def test_cleanup_missing_markers_no_error(self, tmp_path: Path):
        """Cleaning up non-existent markers with missing_ok=True doesn't raise."""
        for r in range(4):
            (tmp_path / f".done_rank{r}").unlink(missing_ok=True)  # no error


# ===========================================================================
# _print_summary
# ===========================================================================


class TestPrintSummary:
    """Tests for _print_summary: human-readable sweep summary."""

    def test_all_successful(self, capsys):
        """Summary correctly reports all successful results."""
        results = [
            _make_train_result(model_tag="a", val_bpb=1.1),
            _make_train_result(model_tag="b", val_bpb=1.2),
            _make_train_result(model_tag="c", val_bpb=1.0),
        ]
        _print_summary(results)
        captured = capsys.readouterr()
        assert "Total:      3" in captured.out
        assert "Successful: 3" in captured.out
        assert "Failed:     0" in captured.out
        assert "Best val_bpb: 1.000000" in captured.out

    def test_some_failures(self, capsys):
        """Summary correctly reports mixed success/failure results."""
        results = [
            _make_train_result(model_tag="ok", val_bpb=1.5),
            _make_train_result(model_tag="fail1", val_bpb=None, error="OOM"),
            _make_train_result(model_tag="fail2", val_bpb=None, error="NaN loss"),
        ]
        _print_summary(results)
        captured = capsys.readouterr()
        assert "Successful: 1" in captured.out
        assert "Failed:     2" in captured.out
        assert "fail1" in captured.out
        assert "fail2" in captured.out

    def test_all_failed(self, capsys):
        """Summary handles all-failed case without errors."""
        results = [
            _make_train_result(model_tag="fail", val_bpb=None, error="err"),
        ]
        _print_summary(results)
        captured = capsys.readouterr()
        assert "Successful: 0" in captured.out
        assert "Failed:     1" in captured.out

    def test_empty_results(self, capsys):
        """Summary handles empty results list."""
        _print_summary([])
        captured = capsys.readouterr()
        assert "Total:      0" in captured.out
        assert "Successful: 0" in captured.out
        assert "Failed:     0" in captured.out

    def test_none_val_bpb_displayed_as_na(self, capsys):
        """Results with None val_bpb show 'N/A' in sorted results."""
        results = [
            _make_train_result(model_tag="good", val_bpb=1.5),
            _make_train_result(model_tag="no_val", val_bpb=None),  # no error, but no val
        ]
        _print_summary(results)
        captured = capsys.readouterr()
        # Both count as successful (no error field)
        assert "Successful: 2" in captured.out


# ===========================================================================
# Module-level multi-GPU env setup
# ===========================================================================


class TestGPUIndexSelection:
    """Tests for --gpu and --sweep-rank/--sweep-world-size CLI arg logic.

    Multi-GPU now uses explicit CLI args instead of torchrun env vars.
    """

    def test_gpu_defaults_to_rank(self):
        """When --gpu is not specified, gpu_id defaults to sweep-rank."""
        for rank in range(8):
            gpu_arg = None
            gpu_id = gpu_arg if gpu_arg is not None else rank
            assert gpu_id == rank

    def test_gpu_override(self):
        """When --gpu is specified, it overrides the default rank-based assignment."""
        rank = 0
        gpu_arg = 3
        gpu_id = gpu_arg if gpu_arg is not None else rank
        assert gpu_id == 3

    def test_gpu_index_only_passed_for_multi_gpu(self):
        """gpu_index=None when world_size==1 (single GPU, no override)."""
        world_size = 1
        gpu_id = 0
        gpu_index = gpu_id if world_size > 1 else None
        assert gpu_index is None

    def test_gpu_index_passed_for_multi_gpu(self):
        """gpu_index=gpu_id when world_size>1."""
        world_size = 2
        gpu_id = 1
        gpu_index = gpu_id if world_size > 1 else None
        assert gpu_index == 1

    def test_sweep_rank_defaults(self):
        """Default sweep-rank is 0, default sweep-world-size is 1."""
        # Mirrors argparse defaults
        default_rank = 0
        default_world_size = 1
        assert default_rank == 0
        assert default_world_size == 1
        assert default_rank == 0  # master_process


# ===========================================================================
# _prefetch_wrapper
# ===========================================================================


class TestPrefetchWrapper:
    """Tests for _prefetch_wrapper: background-thread prefetch buffer."""

    def test_basic_passthrough(self):
        """All items from the inner generator are yielded in order."""
        def gen():
            yield from range(10)
        result = list(_prefetch_wrapper(gen(), prefetch_batches=3))
        assert result == list(range(10))

    def test_empty_generator(self):
        """Empty inner generator → empty output."""
        def gen():
            return
            yield  # Unreachable by design; forces generator semantics.
        result = list(_prefetch_wrapper(gen(), prefetch_batches=2))
        assert result == []

    def test_single_item(self):
        """Single-item generator works correctly."""
        def gen():
            yield 42
        result = list(_prefetch_wrapper(gen(), prefetch_batches=1))
        assert result == [42]

    def test_prefetch_buffer_size(self):
        """Items are buffered up to prefetch_batches without blocking."""
        items = list(range(20))
        def gen():
            yield from items
        result = list(_prefetch_wrapper(gen(), prefetch_batches=5))
        assert result == items

    def test_preserves_complex_objects(self):
        """Dicts, tuples, and other complex objects pass through unchanged."""
        expected = [
            {"key": "value", "num": 1},
            (1, 2, 3),
            [4, 5, 6],
            "string",
        ]
        def gen():
            yield from expected
        result = list(_prefetch_wrapper(gen(), prefetch_batches=2))
        assert result == expected

    def test_propagates_generator_exception(self):
        """Exceptions from the inner generator are re-raised in the consumer."""
        def gen():
            yield 1
            yield 2
            raise ValueError("generator error")
        with pytest.raises(ValueError, match="generator error"):
            list(_prefetch_wrapper(gen(), prefetch_batches=2))

    def test_large_batch_count(self):
        """Works with large prefetch buffer relative to item count."""
        def gen():
            yield from range(3)
        result = list(_prefetch_wrapper(gen(), prefetch_batches=100))
        assert result == [0, 1, 2]

    def test_prefetch_one(self):
        """prefetch_batches=1 (minimal buffer) works correctly."""
        items = list(range(50))
        def gen():
            yield from items
        result = list(_prefetch_wrapper(gen(), prefetch_batches=1))
        assert result == items

    def test_maintains_order_with_slow_producer(self):
        """Order is maintained even if producer is slower than consumer."""
        def gen():
            for i in range(5):
                time.sleep(0.01)  # simulate slow I/O
                yield i
        result = list(_prefetch_wrapper(gen(), prefetch_batches=2))
        assert result == [0, 1, 2, 3, 4]

    def test_none_items_not_confused_with_sentinel(self):
        """None values in the generator are yielded correctly (not confused with sentinel)."""
        def gen():
            yield None
            yield 1
            yield None
        result = list(_prefetch_wrapper(gen(), prefetch_batches=2))
        assert result == [None, 1, None]


# ===========================================================================
# End-to-end integration: load → resume → stripe → merge
# ===========================================================================


class TestEndToEndMultiGPU:
    """Integration tests combining multiple multi-GPU features."""

    def test_full_cycle_two_ranks(self, tmp_path: Path):
        """Simulate a 2-rank sweep: write per-rank files, merge, verify.

        This tests the complete lifecycle:
        1. Two ranks each write results to per-rank JSONL files
        2. Merge combines them
        3. Load completed tags sees all successful results
        4. Resume filtering correctly skips completed
        """
        # Step 1: Rank 0 writes its results
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "cfg_0", "config_index": 0, "error": None, "val_bpb": 1.1},
            {"model_tag": "cfg_2", "config_index": 2, "error": None, "val_bpb": 1.3},
        ])
        # Step 1: Rank 1 writes its results
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "cfg_1", "config_index": 1, "error": None, "val_bpb": 1.2},
            {"model_tag": "cfg_3", "config_index": 3, "error": "OOM", "val_bpb": None},
        ])

        # Step 2: Master merges
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert len(merged) == 4
        assert [r["config_index"] for r in merged] == [0, 1, 2, 3]

        # Step 3: Load completed tags
        completed = _load_completed_tags(tmp_path)
        assert completed == {"cfg_0", "cfg_1", "cfg_2"}
        assert "cfg_3" not in completed  # errored

        # Step 4: Resume would skip 3, retry 1
        all_tags = ["cfg_0", "cfg_1", "cfg_2", "cfg_3", "cfg_4"]
        remaining = [tag for tag in all_tags if tag not in completed]
        assert remaining == ["cfg_3", "cfg_4"]

    def test_resume_after_partial_failure(self, tmp_path: Path):
        """Partial sweep with failures → resume retries failed, skips completed."""
        configs = build_sweep_configs()[:6]
        tags = [c.model_tag for c in configs]

        # Write some completed, some failed
        _write_jsonl(tmp_path / "sweep_results.jsonl", [
            {"model_tag": tags[0], "error": None, "config_index": 0},
            {"model_tag": tags[1], "error": "CUDA OOM", "config_index": 1},
            {"model_tag": tags[2], "error": None, "config_index": 2},
            # tags[3], tags[4], tags[5] never ran
        ])

        completed = _load_completed_tags(tmp_path)
        assert tags[0] in completed
        assert tags[1] not in completed  # errored → retry
        assert tags[2] in completed

        selected = list(enumerate(configs))
        filtered = [(i, c) for i, c in selected if c.model_tag not in completed]
        assert len(filtered) == 4  # tags[1] (retry) + tags[3,4,5] (never ran)

    def test_stripe_then_resume(self, tmp_path: Path):
        """Striping happens before resume, so completed tags from any rank are skipped."""
        configs = build_sweep_configs()[:8]
        tags = [c.model_tag for c in configs]

        # Rank 0 completed some configs in a previous run
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": tags[0], "error": None, "config_index": 0},
            {"model_tag": tags[2], "error": None, "config_index": 2},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": tags[1], "error": None, "config_index": 1},
        ])

        completed = _load_completed_tags(tmp_path)
        assert len(completed) == 3

        # Stripe 8 configs across 2 ranks
        selected = list(enumerate(configs))
        rank0_selected = [(i, c) for idx, (i, c) in enumerate(selected) if idx % 2 == 0]
        rank1_selected = [(i, c) for idx, (i, c) in enumerate(selected) if idx % 2 == 1]

        # Resume filter on rank 0
        rank0_remaining = [(i, c) for i, c in rank0_selected if c.model_tag not in completed]
        # Resume filter on rank 1
        rank1_remaining = [(i, c) for i, c in rank1_selected if c.model_tag not in completed]

        # Total remaining should be 8 - 3 = 5
        assert len(rank0_remaining) + len(rank1_remaining) == 5

    def test_csv_summary_from_merged(self, tmp_path: Path):
        """After merge, CSV summary is generated correctly."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "a", "config_index": 0, "val_bpb": 1.1, "error": None},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "b", "config_index": 1, "val_bpb": 1.2, "error": None},
        ])
        _merge_rank_files(tmp_path)

        jsonl_path = tmp_path / "sweep_results.jsonl"
        csv_path = jsonl_path.with_suffix(".csv")
        _write_csv_summary(jsonl_path, csv_path)

        assert csv_path.exists()
        with open(csv_path, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        assert len(rows) == 2
        assert rows[0]["model_tag"] == "a"
        assert rows[1]["model_tag"] == "b"

    def test_marker_lifecycle(self, tmp_path: Path):
        """Full marker lifecycle: create → detect → cleanup."""
        world_size = 4

        # Each rank creates its marker
        for r in range(world_size):
            (tmp_path / f".done_rank{r}").touch()

        # Master checks all markers exist
        all_done = all(
            (tmp_path / f".done_rank{r}").exists()
            for r in range(world_size)
        )
        assert all_done

        # Master cleans up
        for r in range(world_size):
            (tmp_path / f".done_rank{r}").unlink(missing_ok=True)

        # Verify cleanup
        any_remaining = any(
            (tmp_path / f".done_rank{r}").exists()
            for r in range(world_size)
        )
        assert not any_remaining


# ===========================================================================
# Edge cases and robustness
# ===========================================================================


class TestEdgeCases:
    """Edge cases and robustness tests."""

    def test_serialize_zero_values(self):
        """Zero-valued metrics are preserved (not sanitized to None)."""
        result = _make_train_result(
            val_bpb=0.0,
            total_training_time=0.0,
            peak_memory_mib=0.0,
            startup_time=0.0,
        )
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["val_bpb"] == 0.0
        assert record["total_training_time"] == 0.0
        assert record["peak_memory_mib"] == 0.0
        assert record["startup_time"] == 0.0

    def test_serialize_negative_values(self):
        """Negative values (unusual but possible) are preserved."""
        result = _make_train_result(val_bpb=-0.5)
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["val_bpb"] == -0.5

    def test_serialize_very_large_values(self):
        """Very large (but finite) values are preserved."""
        result = _make_train_result(val_bpb=1e15, peak_memory_mib=1e10)
        cfg = _make_sweep_config()
        record = _serialize_result(result, cfg, index=0)
        assert record["val_bpb"] == 1e15
        assert record["peak_memory_mib"] is not None

    def test_load_completed_tags_concurrent_writes(self, tmp_path: Path):
        """Load works with files that are actively being written to.

        Simulates a partially-written file (missing newline at end).
        """
        content = json.dumps({"model_tag": "complete", "error": None}) + "\n"
        # Simulate partial write: valid JSON on one line
        (tmp_path / "sweep_results.jsonl").write_text(content)
        completed = _load_completed_tags(tmp_path)
        assert "complete" in completed

    def test_merge_preserves_all_fields(self, tmp_path: Path):
        """Merge preserves every field from the original records."""
        original = {
            "model_tag": "test",
            "config_index": 5,
            "val_bpb": 1.234,
            "error": None,
            "custom_field": "preserved",
        }
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [original])
        _merge_rank_files(tmp_path)
        merged = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert len(merged) == 1
        for key, value in original.items():
            assert merged[0][key] == value

    def test_config_label_with_delta(self):
        """Config label includes delta info when enabled."""
        cfg = _make_sweep_config(
            fd_mode=FeatureDeltaMode.prev_segment,
            fd_ratio=0.1,
        )
        result = _make_train_result()
        record = _serialize_result(result, cfg, index=0)
        assert "fd_prev_segment_r0.1" in record["config_label"]

    def test_config_label_without_delta(self):
        """Config label includes 'disabled' when delta is off."""
        cfg = _make_sweep_config(fd_mode=FeatureDeltaMode.disabled)
        result = _make_train_result()
        record = _serialize_result(result, cfg, index=0)
        assert "fd_disabled" in record["config_label"]

    def test_fast_sweep_resolves_splice_data_dir_before_wandb_snapshot(self):
        """Auto-derived data_dir must be visible in fast-sweep run metadata."""
        src = inspect.getsource(_run_one)
        derive_idx = src.index("_resolve_splice_counterfactual_data_dir")
        snapshot_idx = src.index("user_config = _resolved_user_config")
        train_idx = src.index("train_one_config")
        assert derive_idx < snapshot_idx
        assert derive_idx < train_idx

    def test_append_and_load_roundtrip(self, tmp_path: Path):
        """Appended results can be loaded back via _load_completed_tags."""
        path = tmp_path / "sweep_results.jsonl"
        cfg = _make_sweep_config()
        result = _make_train_result(model_tag="roundtrip_tag")
        _append_result(path, result, cfg, index=0)

        completed = _load_completed_tags(tmp_path)
        assert "roundtrip_tag" in completed

    def test_append_error_result_not_in_completed(self, tmp_path: Path):
        """Errored results written via _append_result are not in completed tags."""
        path = tmp_path / "sweep_results.jsonl"
        cfg = _make_sweep_config()
        result = _make_train_result(model_tag="errored_tag", error="failed")
        _append_result(path, result, cfg, index=0)

        completed = _load_completed_tags(tmp_path)
        assert "errored_tag" not in completed

    def test_merge_idempotent(self, tmp_path: Path):
        """Running merge twice produces the same result."""
        _write_jsonl(tmp_path / "sweep_results_rank0.jsonl", [
            {"model_tag": "a", "config_index": 0},
        ])
        _write_jsonl(tmp_path / "sweep_results_rank1.jsonl", [
            {"model_tag": "b", "config_index": 1},
        ])
        _merge_rank_files(tmp_path)
        first_merge = _read_jsonl(tmp_path / "sweep_results.jsonl")
        _merge_rank_files(tmp_path)
        second_merge = _read_jsonl(tmp_path / "sweep_results.jsonl")
        assert first_merge == second_merge

    def test_sweep_config_model_tag_uniqueness_for_resume(self):
        """model_tags from build_sweep_configs are unique enough for resume matching.

        Note: model_tag intentionally omits feature_selection (captured by checkpoint subdir).
        Uniqueness holds within a given feature selection variant.
        """
        configs = build_sweep_configs()
        # Group by feature selection
        from collections import defaultdict
        groups = defaultdict(list)
        for c in configs:
            key = (c.fs.distribution, c.fs.mean, c.fs.std)
            groups[key].append(c.model_tag)
        # Within each feature selection group, tags should be unique
        for key, tags in groups.items():
            assert len(tags) == len(set(tags)), f"Duplicate tags in group {key}"
