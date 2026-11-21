"""End-to-end CLI invocation on the mini_axbench_base_run fixture.

Exercises the full pipeline: prepare_context_cache → context_only_reference →
bpb_score → bpb_aggregate → leaderboard.csv.

DEFENSIVE TEST: in CI we don't have real Gemma + nanochat checkpoints,
so the CLI cannot run all stages end-to-end. The meaningful CI signal is
that the CLI parses config + validates fixtures cleanly when invoked.
The full pipeline run is gated by a separate @pytest.mark.slow test
(Task 15.1, deferred — see deferred-agent-prompt §3).
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest


def test_cli_dev_tier_all_stages_end_to_end(tmp_path):
    """Invoke the CLI and verify it either:
      - succeeds end-to-end (full pipeline runs; leaderboard.csv produced), OR
      - fails with a recognizable post-config-validation error (no Gemma,
        no parquet shards, etc.) — meaning config parsing + fixture
        validation passed.

    This is the gate the test enforces: the CLI must not crash on config
    parsing or `validate_stage_0_3_artifacts`. Anything past that is OK
    in CI without real models.
    """
    repo_root = Path(__file__).resolve().parents[3]
    fixture_root = repo_root / "tests" / "axbench_bpb" / "fixtures"
    mini_run = fixture_root / "mini_axbench_base_run"
    tiny_dec = fixture_root / "tiny_decoder_only"

    if not mini_run.exists() or not tiny_dec.exists():
        pytest.skip("mini fixture or tiny checkpoint not present")

    # Use stage=bpb_aggregate to skip the model-loading stages we can't
    # exercise without real Gemma + nanochat checkpoints. This is enough
    # to verify the CLI parses config + validates fixtures cleanly.
    # The aggregation stage will then fail on missing parquet shards (no
    # score data on a fresh tmp_path), which is the documented
    # acceptable failure mode for this smoke test.
    cmd = [
        sys.executable, "-m", "scripts.eval.axbench_bpb_eval",
        "stage=bpb_aggregate",
        f"source_run_dir={mini_run}",
        f"output_root={tmp_path}",
        "feature_dir=/tmp/feat_unused",
        "feature_statistics_path=/tmp/stats_unused.safetensors",
        "decoder_only.source=base",
        "decoder_only.model_tag=tiny",
        "decoder_only.step=0",
        "encdec.source=encdec",
        "encdec.model_tag=tiny",
        "encdec.step=0",
        "diffmean_and_probe_layer=1",
        "device=cpu",
    ]

    result = subprocess.run(
        cmd, cwd=repo_root, capture_output=True, text=True, timeout=300,
        env={**os.environ, "PYTHONPATH": str(repo_root)},
        check=False,
    )

    if result.returncode == 0:
        # Full pipeline succeeded — verify leaderboard
        leaderboards = list(Path(tmp_path).rglob("leaderboard.csv"))
        assert len(leaderboards) == 1, (
            f"Expected exactly one leaderboard.csv; found {len(leaderboards)}"
        )
        return

    # Acceptable failure path: stage=bpb_aggregate alone has no parquets
    # under tmp_path to read, so it should fail with "No parquet shards
    # found under ...". The crucial gates this test enforces:
    stderr = result.stderr or ""
    stdout = result.stdout or ""
    combined = stderr + "\n" + stdout

    # Gate 1: config parsing must not have failed
    assert "AxBenchBpbConfig" not in stderr or "Unexpected keyword" not in stderr, (
        f"CLI failed during config parsing:\n{stderr[:2000]}"
    )

    # Gate 2: validate_stage_0_3_artifacts must not have failed (mini
    # fixture is well-formed; this is the regression we want to catch
    # if the schema ever drifts)
    assert "missing required Stage 0-3" not in stderr, (
        f"CLI failed validating mini fixture (Stage 0-3 artifacts):\n{stderr[:2000]}"
    )

    # Gate 3: the failure must be the SPECIFIC documented post-validation
    # error — bpb_aggregate cannot find any parquet shards under tmp_path
    # (no score data was generated for a fresh tmp_path). Any
    # other failure signature means config parsing / artifact validation
    # has regressed in a way the previous loose marker list missed.
    #
    # We require an AxBenchBpbError AND the specific "no shards found"
    # diagnostic — accepting bare ``FileNotFoundError`` or any "leaderboard"
    # mention let unrelated failures pass.
    assert "AxBenchBpbError" in combined, (
        f"expected AxBenchBpbError in CLI failure output:\n"
        f"stderr (first 2000 chars):\n{stderr[:2000]}\n"
        f"stdout (first 1000 chars):\n{stdout[:1000]}"
    )
    assert "No parquet shards found" in combined, (
        f"expected the documented 'No parquet shards found' diagnostic "
        f"from bpb_aggregate:\n"
        f"stderr (first 2000 chars):\n{stderr[:2000]}\n"
        f"stdout (first 1000 chars):\n{stdout[:1000]}"
    )
