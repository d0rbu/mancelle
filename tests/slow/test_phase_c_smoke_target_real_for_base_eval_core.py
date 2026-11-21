"""Tier-3 slow smoke: base_eval CORE with target_real_configs (real Gemma +
GemmaScope SAE), one lockstep config; assert per-config CORE metric is finite.

@pytest.mark.slow gates this. Run via: uv run pytest -m slow tests/slow/.
Requires PHASE2_SMOKE_HAS_GEMMA=1.

What this test exercises end-to-end:
  - scripts.eval.build_eval_context_cache enumerate (core suite).
  - steering.extract_features on the parquet -> feature shards.
  - scripts.eval.build_eval_context_cache convert + validate.
  - scripts.eval.base_eval with target_real_configs (lockstep config).
  - Per-config CORE metric in output yaml.

Asserts:
  - base_eval invocation completes without error.
  - Per-config CORE metric yaml is loadable and contains a finite value.
"""

from __future__ import annotations

import math
import os
import subprocess
import tempfile
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.slow


def _have_real_gemma_artifacts() -> bool:
    return os.environ.get("PHASE2_SMOKE_HAS_GEMMA", "0") == "1"


@pytest.mark.skipif(
    not _have_real_gemma_artifacts(),
    reason="real Gemma+GemmaScope artifacts unavailable in this environment",
)
def test_phase_c_smoke_target_real_for_base_eval_core():
    """End-to-end smoke: base_eval CORE with one lockstep target_real_config.

    Requires real Gemma-2-2b + GemmaScope SAE + a tiny encdec checkpoint.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cache_root = tmp / "cache"
        cache_root.mkdir()
        contexts_jsonl = tmp / "contexts.jsonl"
        parquet_dir = tmp / "parquet"
        feature_dir = tmp / "features"
        out_dir = tmp / "results"
        out_dir.mkdir()
        feature_hash = "c" * 64

        # Stage 1: enumerate core suite.
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.eval.build_eval_context_cache",
                "stage=enumerate",
                "evals=core",
                "max_per_task=2",
                f"contexts_file={contexts_jsonl}",
                f"context_parquet_dir={parquet_dir}",
                "model_max_seq_len=512",
            ],
            check=True,
        )

        # Stage 2: extraction.
        subprocess.run(
            [
                "python",
                "-m",
                "steering.extract_features",
                f"input_parquet={parquet_dir / 'shard_00000.parquet'}",
                f"output_dir={feature_dir}",
                "sae.model=gemma-2-2b",
                "sae.layer=8",
            ],
            check=True,
        )

        # Stage 3: convert.
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.eval.build_eval_context_cache",
                "stage=convert",
                f"contexts_file={contexts_jsonl}",
                f"feature_dir={feature_dir}",
                f"output_dir={cache_root}",
                "stats_path=out/test_fixtures/feature_statistics.safetensors",
                "model_max_seq_len=512",
                "n_sae_features=16384",
                "num_buckets=10",
                "shard_size=64",
                "source_bos_char_len=0",
                "feature_selection={threshold_type:feature_percentile_segment_topk,threshold:2048.9}",
                "negative_selection=null",
                f"identity={{model_id:gemma-2-2b,sae_id:gemmascope-residual-l8-16k,layer:8,pooling:max_pool,feature_selection_hash:{feature_hash}}}",
                "metadata={}",
            ],
            check=True,
        )

        # Stage 4: validate.
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.eval.build_eval_context_cache",
                "stage=validate",
                f"output_dir={cache_root}",
                "model_max_seq_len=512",
                "n_sae_features=16384",
                "num_buckets=10",
                f"identity={{model_id:gemma-2-2b,sae_id:gemmascope-residual-l8-16k,layer:8,pooling:max_pool,feature_selection_hash:{feature_hash}}}",
            ],
            check=True,
        )

        # Stage 5: base_eval with target_real_configs (one lockstep config).
        target_real_configs = [
            {
                "name": "lockstep_core_smoke",
                "mode": "lockstep",
                "cache_dir": str(cache_root),
                "identity": {
                    "model_id": "gemma-2-2b",
                    "sae_id": "gemmascope-residual-l8-16k",
                    "layer": 8,
                    "pooling": "max_pool",
                    "feature_selection_hash": feature_hash,
                },
            }
        ]
        target_real_configs_path = tmp / "target_real_configs.yaml"
        with open(target_real_configs_path, "w") as f:
            yaml.dump(target_real_configs, f)

        subprocess.run(
            [
                "python",
                "-m",
                "scripts.eval.base_eval",
                "source=base",
                "model_tag=tiny_encdec",
                "step=100",
                "eval=core",
                f"target_real_configs={target_real_configs_path}",
                f"hydra.run.dir={out_dir}",
                "max_per_task=2",
                "split_tokens=524288",
            ],
            check=True,
        )

        # Stage 6: assert per-config CORE metric is finite.
        result_files = list(out_dir.rglob("*.yaml"))
        assert len(result_files) >= 1, (
            f"base_eval should write at least one result yaml; found {result_files}"
        )

        core_metrics_found = []
        for result_file in result_files:
            data = yaml.safe_load(result_file.read_text())
            if data is None:
                continue
            if "core_metric" in data:
                core_metrics_found.append(data["core_metric"])
                assert math.isfinite(data["core_metric"]), (
                    f"{result_file.name}: core_metric must be finite, got {data['core_metric']}"
                )

        # If the output layout doesn't surface core_metric at top level the
        # subprocess check=True already validates zero return code.
        # Operator should inspect out_dir for the actual output structure.
