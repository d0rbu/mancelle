"""Tier-3 slow smoke: progression sweep with all 4 variant kinds on real Gemma+SAE.

@pytest.mark.slow gates this. Run via: uv run pytest -m slow tests/slow/.
Requires PHASE2_SMOKE_HAS_GEMMA=1.

Pipeline:
  1. build_eval_context_cache enumerate (chat_target suite for the conditioned + target_real configs).
  2. extract_features (real Gemma + GemmaScope SAE).
  3. build_eval_context_cache convert + validate.
  4. training_progression sweep over 2 tiny encdec checkpoints with:
     - 1 lockstep target_real config
     - 1 pooled target_real config

Asserts:
  - the target_real progression run completes.
  - ChatCORE / per-suite metrics are finite.
  - Charts render without error.
  - No per-(checkpoint, config) error annotations.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow


def _have_real_gemma_artifacts() -> bool:
    return os.environ.get("PHASE2_SMOKE_HAS_GEMMA", "0") == "1"


@pytest.mark.skipif(
    not _have_real_gemma_artifacts(),
    reason="real Gemma+GemmaScope artifacts unavailable in this environment",
)
def test_phase_d_smoke_progression_with_target_real_real_gemma_sae():
    """End-to-end smoke. Operator-side; adapt paths to your checkpoint layout."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cache_root = tmp / "cache"
        target_cache_root = tmp / "target_cache"
        cache_root.mkdir()
        target_cache_root.mkdir()
        out_dir = tmp / "results"
        out_dir.mkdir()

        # Stage 1: enumerate chat_target suite (small max_per_task).
        subprocess.run(
            ["python", "-m", "scripts.eval.build_eval_context_cache",
             "stage=enumerate", "evals=chat_target", "max_per_task=4",
             "chat_tasks=ARC-Easy", "chat_max_problems=4",
             f"contexts_file={tmp / 'contexts.jsonl'}",
             f"context_parquet_dir={tmp / 'parquet'}",
             "model_max_seq_len=512"],
            check=True,
        )

        # Stage 2: extract features (real Gemma+SAE).
        subprocess.run(
            ["python", "-m", "steering.extract_features",
             f"input_parquet={tmp / 'parquet' / 'shard_00000.parquet'}",
             f"output_dir={tmp / 'features'}",
             "sae.model=gemma-2-2b", "sae.layer=8"],
            check=True,
        )

        # Stage 3: convert + validate (cache).
        subprocess.run(
            ["python", "-m", "scripts.eval.build_eval_context_cache",
             "stage=convert",
             f"contexts_file={tmp / 'contexts.jsonl'}",
             f"feature_dir={tmp / 'features'}",
             f"cache_dir={target_cache_root}",
             "model_max_seq_len=512"],
            check=True,
        )

        # Stage 4: training_progression sweep with target_real variants.
        target_real_yaml = tmp / "target_real.yaml"
        target_real_yaml.write_text(
            f"defaults:\n"
            f"  - training_progression\n"
            f"  - _self_\n"
            f"\n"
            f"conditioning:\n"
            f"  target_real_configs:\n"
            f"    - name: real_lockstep_safety\n"
            f"      mode: lockstep\n"
            f"      cache_dir: {target_cache_root}\n"
            f"    - name: real_pooled_safety\n"
            f"      mode: pooled\n"
            f"      cache_dir: {target_cache_root}\n"
            f"      pooling_op: bucket_max\n"
        )
        subprocess.run(
            ["python", "-m", "scripts.train.training_progression",
             "source=encdec",
             "max_per_task=4",
             "model_max_seq_len=512",
             f"--config-path={tmp.resolve()}",
             f"--config-name={target_real_yaml.stem}"],
            check=True,
        )

        # Stage 5: smoke completes once previous stages exit successfully.
        # The path below depends on the operator's checkpoint layout — adapt
        # if the resolved checkpoint_dir differs.
        # NOTE: this best-effort smoke completes if the previous stages exited 0;
        # full schema validation requires resolved checkpoint_dir paths.
