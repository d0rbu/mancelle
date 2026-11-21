"""Tests for PR-4 per-rank checkpoint paths."""

from __future__ import annotations

from tests.axbench_base import paths as test_paths


def test_stage0c_partial_rank_json_format(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    p = test_paths.default_paths("test-tag").stage0c_partial_rank_json(rank=2)
    assert p.name == "stage0c_partial_rank_2.json"


def test_stage0c_partial_rank_parquet_format(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    p = test_paths.default_paths("test-tag").stage0c_partial_rank_parquet(rank=3)
    assert p.name == "stage0c_partial_rank_3.parquet"


def test_stage0c_partial_rank_paths_share_run_dir(tmp_path, monkeypatch):
    """Per-rank files live in the run output directory."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    rank_path = test_paths.default_paths("test-tag").stage0c_partial_rank_json(rank=0)
    run_root = test_paths.default_paths("test-tag").run_output_root()
    assert rank_path.parent == run_root


def test_stage0c_partial_rank_paths_distinct_per_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    p0 = test_paths.default_paths("test-tag").stage0c_partial_rank_json(rank=0)
    p1 = test_paths.default_paths("test-tag").stage0c_partial_rank_json(rank=1)
    assert p0 != p1
