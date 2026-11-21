from __future__ import annotations

import json

import pytest

from scripts.preprocess._splice_profile import (
    DETAIL_KEYS,
    PHASE_KEYS,
    SpliceRunProfile,
    SpliceShardProfile,
)


def test_splice_profile_schema_totals_and_atomic_write(tmp_path):
    base = tmp_path / "shard_00000_features.safetensors.zst"
    parquet = tmp_path / "shard_00000.parquet"
    base.write_bytes(b"feature")
    parquet.write_bytes(b"parquet")

    shard = SpliceShardProfile.from_paths(
        shard_stem="shard_00000",
        shard_index=0,
        base_shard_path=base,
        parquet_path=parquet,
    )
    shard.start_total()
    with shard.time_phase("load_features"):
        pass
    with shard.time_detail("source.truncate_tokenize"):
        pass
    with pytest.raises(KeyError, match="unknown splice profile detail"):
        with shard.time_detail("not.a.detail"):
            pass
    shard.finish(status="failed", error_type="AlignmentError", error_message="bad align")
    record = shard.to_dict()
    assert set(record["phase_seconds"]) == set(PHASE_KEYS)
    assert set(record["detail_seconds"]) == set(DETAIL_KEYS)
    assert record["detail_seconds"]["source.truncate_tokenize"] >= 0.0
    assert record["status"] == "failed"
    assert record["error_type"] == "AlignmentError"
    assert record["input_feature_bytes"] == len(b"feature")
    assert record["input_parquet_bytes"] == len(b"parquet")

    run = SpliceRunProfile(
        config={"feature_dir": "base", "output_dir": "out"},
        rank=1,
        world_size=2,
        shard_parallelism=1,
        n_skipped=3,
    )
    run.start()
    run.add_shard(record)
    path = tmp_path / "profile.json"
    run.write(path)

    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == {"config", "totals", "per_shard"}
    assert data["config"]["rank"] == 1
    assert data["config"]["world_size"] == 2
    assert data["totals"]["n_failed"] == 1
    assert data["totals"]["n_skipped"] == 3
    assert set(data["totals"]["phase_seconds"]) == set(PHASE_KEYS)
    assert set(data["totals"]["detail_seconds"]) == set(DETAIL_KEYS)
