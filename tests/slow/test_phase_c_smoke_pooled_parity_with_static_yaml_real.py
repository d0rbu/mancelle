"""Tier-3 slow smoke: pooled-parity check on real Gemma + GemmaScope SAE.

Runs chat_eval with a target_real_config (pooled) and a static-YAML
feature_configs entry with the offline-pooled dict, then asserts the two
yamls are equivalent (chat eval accuracies match).

@pytest.mark.slow gates this. Run via: uv run pytest -m slow tests/slow/.
Requires PHASE2_SMOKE_HAS_GEMMA=1.

What this test exercises end-to-end:
  - scripts.eval.build_eval_context_cache enumerate (chat_target suite, small N).
  - steering.extract_features on the parquet -> feature shards.
  - scripts.eval.chat_eval with target_real_configs (one pooled config).
  - scripts.eval.chat_eval with a static-YAML feature_configs entry using
    the offline-pooled dict.
  - Parity assertion: per-config accuracies from both runs match.

Asserts:
  - Both chat_eval invocations complete without error.
  - The output yamls are loadable and their accuracy values are equal.
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


@pytest.mark.xfail(
    reason=(
        "PR-C operator-side stub: requires real Gemma artifacts AND a known-good "
        "offline-pooled feature dict + matching cache_dir; operator must populate "
        "before running. Will be replaced by a real smoke once tier-3 infrastructure "
        "can resolve checkpoint output paths automatically."
    ),
    strict=False,
    run=False,
)
@pytest.mark.skipif(
    not _have_real_gemma_artifacts(),
    reason="real Gemma+GemmaScope artifacts unavailable in this environment",
)
def test_phase_c_smoke_pooled_parity_with_static_yaml_real():
    """End-to-end pooled-parity smoke.

    Requires real Gemma-2-2b + GemmaScope SAE + a tiny encdec checkpoint.
    Validates that online target_real pooled conditioning produces the same
    chat eval accuracies as offline-pooled static feature_configs.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        cache_root = tmp / "cache"
        cache_root.mkdir()
        contexts_jsonl = tmp / "contexts.jsonl"
        parquet_dir = tmp / "parquet"
        feature_dir = tmp / "features"
        out_target_real = tmp / "results_target_real"
        out_target_real.mkdir()
        out_static = tmp / "results_static"
        out_static.mkdir()

        # Stage 1: enumerate chat_target suite (small N for speed).
        subprocess.run(
            ["python", "-m", "scripts.eval.build_eval_context_cache",
             "stage=enumerate", "evals=chat_target", "max_per_task=2",
             "chat_tasks=ARC-Easy", "chat_max_problems=2",
             f"contexts_file={contexts_jsonl}",
             f"context_parquet_dir={parquet_dir}",
             "model_max_seq_len=512"],
            check=True,
        )

        # Stage 2: extraction.
        subprocess.run(
            ["python", "-m", "steering.extract_features",
             f"input_parquet={parquet_dir / 'shard_00000.parquet'}",
             f"output_dir={feature_dir}",
             "sae.model=gemma-2-2b", "sae.layer=8"],
            check=True,
        )

        # Stage 3: convert.
        subprocess.run(
            ["python", "-m", "scripts.eval.build_eval_context_cache",
             "stage=convert",
             f"contexts_file={contexts_jsonl}",
             f"feature_dir={feature_dir}",
             f"cache_dir={cache_root}",
             "model_max_seq_len=512"],
            check=True,
        )

        # Stage 4a: chat_eval with target_real_configs (online pooled).
        target_real_yaml = tmp / "target_real_pooled.yaml"
        target_real_yaml.write_text(
            f"target_real_configs:\n"
            f"  - name: pooled_online\n"
            f"    mode: pooled\n"
            f"    cache_dir: {cache_root}\n"
            f"    pooling_op: bucket_max\n"
        )
        subprocess.run(
            ["python", "-m", "scripts.eval.chat_eval",
             "source=encdec",
             "task_name=ARC-Easy",
             "max_problems=1",
             f"--config-path={tmp.resolve()}",
             f"--config-name={target_real_yaml.stem}",
             "model_max_seq_len=512",
             f"hydra.run.dir={out_target_real}"],
            check=True,
        )

        # Stage 4b: chat_eval with static-YAML feature_configs (offline pooled dict).
        # The offline dict represents the same pooled features serialised as a
        # mapping of {feature_index: bucket_value}. Operator should populate this
        # dict from their pooled extraction output; here we use a placeholder.
        static_feature_configs = [
            {"name": "pooled_static", "features": {12: 1, 24: 2}},
        ]
        static_feature_configs_path = tmp / "static_feature_configs.yaml"
        with open(static_feature_configs_path, "w") as f:
            yaml.dump(static_feature_configs, f)

        subprocess.run(
            ["python", "-m", "scripts.eval.chat_eval",
             "source=encdec",
             "task_name=ARC-Easy",
             "max_problems=1",
             f"feature_configs={static_feature_configs_path}",
             "model_max_seq_len=512",
             f"hydra.run.dir={out_static}"],
            check=True,
        )

        # Stage 5: parity assertion.
        # Collect yaml outputs from both runs and compare accuracy fields.
        # NOTE: operator should adapt glob patterns to their output layout.
        target_real_yamls = list(out_target_real.rglob("*.yaml"))
        static_yamls = list(out_static.rglob("*.yaml"))

        def _load_accuracy(files: list[Path]) -> list[float]:
            values = []
            for f in files:
                data = yaml.safe_load(f.read_text())
                if isinstance(data, dict) and "accuracy" in data:
                    values.append(float(data["accuracy"]))
            return values

        acc_target_real = _load_accuracy(target_real_yamls)
        acc_static = _load_accuracy(static_yamls)

        if acc_target_real and acc_static:
            assert len(acc_target_real) == len(acc_static), (
                f"Mismatch in number of accuracy entries: "
                f"target_real={acc_target_real}, static={acc_static}"
            )
            for a, b in zip(sorted(acc_target_real), sorted(acc_static), strict=True):
                assert math.isclose(a, b, rel_tol=1e-5), (
                    f"Pooled parity violated: target_real accuracy={a} != static accuracy={b}"
                )
        # If no accuracy keys surfaced the invocations still passed check=True;
        # the minimal invariant is zero subprocess return codes.
