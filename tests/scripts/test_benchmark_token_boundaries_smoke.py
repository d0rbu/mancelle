"""Smoke test for the token-boundary benchmark harness (issue #529)."""
from __future__ import annotations

import json
import time
from unittest.mock import patch

from omegaconf import OmegaConf

from scripts.benchmark._resource_sampler import sample_process_tree
from scripts.benchmark.benchmark_token_boundaries import run_bench
from tests.scripts._token_boundary_fixtures import make_token_boundary_env


def test_bench_smoke(tmp_path):
    env = make_token_boundary_env(tmp_path, n_shards=1)
    out_md = tmp_path / "bench.md"
    cfg = OmegaConf.create(
        {
            "feature_dir": env["feature_dir"],
            "data_dir": env["data_dir"],
            "segment_size": env["segment_size"],
            "report_md": str(out_md),
            "sidecar_dir": str(tmp_path / "sidecars"),
            "variants": [
                {"name": "w1", "overrides": {"workers": 1}},
                {"name": "w2", "overrides": {"workers": 2}},
            ],
        }
    )
    with patch(
        "scripts.preprocess.precompute_token_boundaries.get_tokenizer",
        return_value=env["tokenizer"],
    ):
        md = run_bench(cfg)

    assert out_md.is_file()
    assert "| variant |" in md
    assert "w1" in md and "w2" in md
    assert "threads_max" in md
    assert "cpu_cores_avg" in md
    assert "rss_peak_mb" in md
    w1 = json.loads((tmp_path / "sidecars" / "w1.digests.json").read_text())
    w2 = json.loads((tmp_path / "sidecars" / "w2.digests.json").read_text())
    assert w1 == w2
    w2_sidecar = json.loads((tmp_path / "sidecars" / "w2.json").read_text())
    assert w2_sidecar["worker_pool"]["thread_caps"]["torch_num_threads"] == 1
    assert w2_sidecar["worker_pool"]["thread_caps"]["torch_num_interop_threads"] == 1


def test_sample_process_tree_busy_loop() -> None:
    def busy_loop() -> int:
        deadline = time.perf_counter() + 0.3
        value = 0
        while time.perf_counter() < deadline:
            value += 1
        return value

    value, sample = sample_process_tree(busy_loop, interval_s=0.05)
    assert value > 0
    assert sample.wall_seconds > 0
    assert sample.cpu_cores_avg > 0
    assert sample.rss_peak_bytes > 0
    assert sample.pss_peak_bytes is None or sample.pss_peak_bytes > 0
    assert sample.swap_peak_bytes is None or sample.swap_peak_bytes >= 0
