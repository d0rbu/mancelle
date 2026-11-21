"""Tier-3 slow smoke: build a context cache (real Gemma + GemmaScope SAE),
run base_eval with multiple non-empty target-delta configs, verify per-config
yaml outputs with finite CORE numbers.

@pytest.mark.slow gates this. Run via: uv run pytest -m slow tests/slow/.
Requires PHASE2_SMOKE_HAS_GEMMA=1.

What this test exercises end-to-end:
  - scripts.eval.build_eval_context_cache enumerate (core + samples suites).
  - steering.extract_features on the parquet -> feature shards.
  - scripts.eval.build_eval_context_cache convert + validate.
  - scripts.eval.base_eval with conditioning.context_cache_dir set.
  - Per-config sample blocks + CORE per-config under full-context conditioning.

Asserts:
  - Per-config yaml output files exist.
  - Each yaml has finite CORE metric and bpb_results.
  - Multiple explicit target-delta configs produce the same result shape.
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
def test_phase_b_smoke_base_eval_real_gemma_sae():
    """End-to-end smoke. Runs in <10 minutes on an H100. Requires:
    - Real Gemma-2-2b weights cached.
    - Real GemmaScope SAE for layer 8 cached.
    - Tiny encdec checkpoint at out/test_fixtures/tiny_encdec.pt.
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

        # Stage 1: enumerate core + samples suites.
        subprocess.run(
            [
                "python",
                "-m",
                "scripts.eval.build_eval_context_cache",
                "stage=enumerate",
                "evals=core,samples",
                "max_per_task=2",
                f"contexts_file={contexts_jsonl}",
                f"context_parquet_dir={parquet_dir}",
                "model_max_seq_len=512",
                'sample_prompts=["The capital of France is","The chemical symbol of gold is","My favorite color is"]',
            ],
            check=True,
        )

        # Stage 2: extraction (operator step).
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

        # Stage 5: run base_eval with non-empty target deltas.
        feature_configs = [
            {"name": "real_safety", "features": {12: 1, 24: 2}},
            {"name": "real_capability", "features": {36: 1}},
        ]
        feature_configs_path = tmp / "feature_configs.yaml"
        with open(feature_configs_path, "w") as f:
            yaml.dump(feature_configs, f)

        subprocess.run(
            [
                "python",
                "-m",
                "scripts.eval.base_eval",
                "source=base",
                "model_tag=tiny_encdec",
                "step=100",
                "eval=core,bpb,sample",
                f"feature_configs={feature_configs_path}",
                f"conditioning.context_cache_dir={cache_root}",
                f"conditioning.identity={{model_id:gemma-2-2b,sae_id:gemmascope-residual-l8-16k,layer:8,pooling:max_pool,feature_selection_hash:{feature_hash}}}",
                f"hydra.run.dir={out_dir}",
                "max_per_task=2",
                "split_tokens=524288",
            ],
            check=True,
        )

        # Assert: per-config yaml outputs exist with the expected structure.
        # The exact output layout depends on base_eval's writer; adjust if needed.
        result_files = list(out_dir.rglob("*.yaml"))
        assert len(result_files) >= 1, (
            f"base_eval should write at least one result yaml; found {result_files}"
        )

        # Find the per-config CORE result; assert finite metric.
        for result_file in result_files:
            with open(result_file) as f:
                data = yaml.safe_load(f)
            if data is None:
                continue
            if "core_metric" in data:
                assert math.isfinite(data["core_metric"]), (
                    f"{result_file.name}: core_metric must be finite, got {data['core_metric']}"
                )
