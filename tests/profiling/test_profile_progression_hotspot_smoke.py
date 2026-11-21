"""Smoke test for scripts.profile_progression run_hotspot (CPU-only, hermetic)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from scripts import profile_progression_microbench
from scripts.profile_progression import run_hotspot
from scripts.profile_progression_microbench import register

try:
    @register("_stub_microbench_smoke")
    def _stub(fixture_path, device_str, n_iters):
        # Ignore the fixture entirely — we're smoke-testing the dispatcher, not the bench.
        return {
            "name": "_stub_microbench_smoke",
            "median_ms": 0.42,
            "ci_lower_ms": 0.40,
            "ci_upper_ms": 0.44,
            "n": n_iters,
            "config_hash": "stubhash",
            "fixture_path": str(fixture_path),
            "git_sha": "stubsha",
        }
except ValueError:
    # Module was already imported (e.g. pytest-xdist worker re-import); skip re-registration.
    pass


try:
    @register("_stub_microbench_multirow_smoke")
    def _stub_multirow(fixture_path, device_str, n_iters):
        return [
            {
                "name": "_stub_microbench_multirow_smoke",
                "context_len_tokens": 128,
                "median_ms": 0.10,
                "ci_lower_ms": 0.09,
                "ci_upper_ms": 0.11,
                "n": n_iters,
                "config_hash": "stubhash",
                "fixture_path": str(fixture_path),
                "git_sha": "stubsha",
            },
            {
                "name": "_stub_microbench_multirow_smoke",
                "context_len_tokens": 256,
                "median_ms": 0.20,
                "ci_lower_ms": 0.19,
                "ci_upper_ms": 0.21,
                "n": n_iters,
                "config_hash": "stubhash",
                "fixture_path": str(fixture_path),
                "git_sha": "stubsha",
            },
        ]
except ValueError:
    # Module was already imported (e.g. pytest-xdist worker re-import); skip re-registration.
    pass


@pytest.fixture
def hotspot_cfg(tmp_path):
    return OmegaConf.create(
        {
            "level": "hotspot",
            "path": "C",
            "hotspot_name": "_stub_microbench_smoke",
            "hotspot_fixture": str(tmp_path / "unused.pt"),
            "hotspot_n_iters": 1,
            "out_root": str(tmp_path / "profiling"),
        }
    )


def test_hotspot_writes_jsonl_schema(hotspot_cfg, tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_hotspot(hotspot_cfg, run_dir)

    jsonl = run_dir / "microbench_results.jsonl"
    assert jsonl.exists()
    row = json.loads(jsonl.read_text().strip())
    for key in (
        "name",
        "median_ms",
        "ci_lower_ms",
        "ci_upper_ms",
        "n",
        "config_hash",
        "fixture_path",
        "git_sha",
    ):
        assert key in row, f"missing key {key}"
    assert isinstance(row["median_ms"], (int, float))

    baselines = Path(hotspot_cfg.out_root) / "summary" / "baselines.jsonl"
    assert baselines.exists()
    assert baselines.read_text().strip() == jsonl.read_text().strip()


def test_hotspot_writes_multirow_jsonl_schema(tmp_path):
    cfg = OmegaConf.create(
        {
            "level": "hotspot",
            "path": "C",
            "hotspot_name": "_stub_microbench_multirow_smoke",
            "hotspot_fixture": str(tmp_path / "unused.pt"),
            "hotspot_n_iters": 1,
            "out_root": str(tmp_path / "profiling"),
        }
    )
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    run_hotspot(cfg, run_dir)

    jsonl = run_dir / "microbench_results.jsonl"
    rows = [json.loads(line) for line in jsonl.read_text().splitlines()]
    assert [row["context_len_tokens"] for row in rows] == [128, 256]
    for row in rows:
        for key in (
            "name",
            "median_ms",
            "ci_lower_ms",
            "ci_upper_ms",
            "n",
            "config_hash",
            "fixture_path",
            "git_sha",
        ):
            assert key in row, f"missing key {key}"

    baselines = Path(cfg.out_root) / "summary" / "baselines.jsonl"
    assert baselines.read_text().strip() == jsonl.read_text().strip()


def test_registered_names_lists_builtins_without_importing(monkeypatch):
    monkeypatch.setitem(
        profile_progression_microbench._BUILTINS,
        "_lazy_builtin_that_should_not_import",
        "module.that.does.not.exist",
    )

    names = profile_progression_microbench.registered_names()
    assert "_lazy_builtin_that_should_not_import" in names
