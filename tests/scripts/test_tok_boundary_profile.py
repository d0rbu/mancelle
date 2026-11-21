"""Unit tests for the token-boundary profiling sidecar (issue #529)."""
from __future__ import annotations

import json

import pytest

from scripts.preprocess._tok_boundary_profile import ShardTiming, TokenBoundaryProfile


def test_profile_roundtrip(tmp_path):
    prof = TokenBoundaryProfile(config={"workers": 1})
    prof.add_shard(
        ShardTiming(
            stem="shard_00000",
            n_segments=10,
            load_features_seconds=1.0,
            load_bytes=100,
            tokenize_seconds=2.0,
            overlay_save_seconds=0.5,
            overlay_bytes=42,
        )
    )
    prof.totals["n_processed"] = 1
    out = tmp_path / "p.json"
    prof.write(out)

    data = json.loads(out.read_text())
    assert set(data) == {"config", "manifest_pre_pass", "worker_pool", "per_shard", "totals"}
    assert data["config"]["workers"] == 1
    assert len(data["per_shard"]) == 1
    sh = data["per_shard"][0]
    assert sh["stem"] == "shard_00000"
    assert sh["n_segments"] == 10
    assert sh["load_bytes"] == 100
    assert sh["overlay_bytes"] == 42
    # default sub-dicts present and typed
    assert data["manifest_pre_pass"]["cache_hit"] is False
    assert data["worker_pool"]["spawn_count"] == 0


def test_profile_write_asserts_consistency(tmp_path):
    prof = TokenBoundaryProfile()
    prof.add_shard(ShardTiming(stem="s", n_segments=1))
    prof.totals["n_processed"] = 2  # deliberate mismatch
    with pytest.raises(AssertionError, match="per_shard"):
        prof.write(tmp_path / "p.json")


def test_add_shard_rejects_empty_stem():
    prof = TokenBoundaryProfile()
    with pytest.raises(AssertionError, match="stem"):
        prof.add_shard(ShardTiming(stem="", n_segments=1))
