"""Unit tests for the packed-rows profiling sidecar dataclasses (issue #531)."""

import json

import pytest

from scripts.preprocess._packed_rows_profile import PackedRowsProfile, ShardSaveTiming


def test_add_output_shard_and_write(tmp_path):
    p = PackedRowsProfile(config={"context_len": 2048})
    p.stage_totals["csr_build"] = 1.5
    p.add_output_shard(
        ShardSaveTiming(
            shard_idx=0,
            n_rows=3,
            n_segments=9,
            n_features=40,
            assemble_seconds=0.1,
            save_seconds=2.0,
            worker_seconds=2.2,
            queue_wait_seconds=0.25,
            save_bytes=123,
        )
    )
    p.totals = {"wall_seconds": 5.0, "total_rows": 3, "rows_per_sec": 0.6}
    out = tmp_path / "prof.json"
    p.write(out)

    d = json.loads(out.read_text())
    assert d["config"]["context_len"] == 2048
    assert d["stage_totals"]["csr_build"] == 1.5
    assert len(d["per_output_shard"]) == 1
    assert d["per_output_shard"][0]["save_bytes"] == 123
    assert d["per_output_shard"][0]["n_features"] == 40
    assert d["per_output_shard"][0]["queue_wait_seconds"] == 0.25
    assert d["per_output_shard"][0]["worker_seconds"] == 2.2
    assert d["totals"]["total_rows"] == 3


def test_write_asserts_shard_count_consistency(tmp_path):
    p = PackedRowsProfile()
    p.add_output_shard(
        ShardSaveTiming(
            shard_idx=0,
            n_rows=1,
            n_segments=1,
            n_features=1,
            assemble_seconds=0.0,
            save_seconds=0.0,
            save_bytes=0,
        )
    )
    # n_shards_written disagrees with the number of recorded shards -> assert.
    p.totals = {
        "wall_seconds": 1.0,
        "total_rows": 1,
        "rows_per_sec": 1.0,
        "n_shards_written": 2,
    }
    with pytest.raises(AssertionError):
        p.write(tmp_path / "bad.json")


def test_add_output_shard_rejects_negative_invariants():
    """add_output_shard's invariant asserts (shard_idx >= 0, n_rows >= 0) have teeth (#540 N3)."""
    p = PackedRowsProfile()
    with pytest.raises(AssertionError):
        p.add_output_shard(ShardSaveTiming(shard_idx=-1, n_rows=1, n_segments=1, n_features=1))
    with pytest.raises(AssertionError):
        p.add_output_shard(ShardSaveTiming(shard_idx=0, n_rows=-1, n_segments=1, n_features=1))
    assert p.per_output_shard == []  # neither bad record was appended


def test_default_schema_has_expected_sections():
    d = PackedRowsProfile().to_dict()
    assert set(d) == {
        "config",
        "manifest_pre_pass",
        "stage_totals",
        "row_build_detail",
        "base_shard_loads",
        "startup",
        "feature_index_loads",
        "compression",
        "per_output_shard",
        "resume",
        "totals",
    }
    assert set(d["stage_totals"]) == {
        "parquet",
        "lookup",
        "tokenize",
        "seg_to_raw",
        "truncate_assert_append",
        "pack",
        "csr_build",
    }
    assert d["resume"]["mode"] == "none"
    assert set(d["resume"]) == {
        "mode",
        "replay_seconds",
        "rows_replayed",
        "checkpoint_seek_seconds",
        "checkpoint_write_seconds",
    }
    assert d["row_build_detail"] == {
        "row_to_csr_seconds": 0.0,
        "row_provenance_seconds": 0.0,
        "build_segments_seconds": 0.0,
        "segment_boundaries_seconds": 0.0,
        "feature_concat_seconds": 0.0,
        "feature_ids_concat_seconds": 0.0,
        "percentiles_concat_seconds": 0.0,
        "feature_counts_seconds": 0.0,
        "row_tensor_finalize_seconds": 0.0,
    }
    assert d["startup"]["feature_index_scan_seconds"] == 0.0
    assert d["feature_index_loads"] == []
    assert d["compression"] == {
        "backpressure_seconds": 0.0,
        "max_inflight_shards": 0,
    }
