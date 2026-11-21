"""Slow / GPU-gated tests for P0 inline steering eval caching.

These tests run a real (short) training loop to verify the caching
interacts correctly with the actual DDP/FP8/torch.compile path.

Gated on torch.cuda.is_available() — skipped on CPU-only machines.
Marked @pytest.mark.slow at the module level — deselected by default
(see pyproject.toml's `addopts = "-m 'not slow'"`).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for full train-loop test")
def test_p0_full_train_loop_steering_eval_caching(tmp_path):
    """Run train_one_config with steering_eval_every=2 for ~6 steps.

    Asserts (when implemented):
    - training runs to completion without errors
    - the first eval wall-clock is higher than subsequent evals
      (first eval pays cache-build cost; subsequent evals are cache hits)
    - all final metrics are finite
    """
    pytest.skip(
        "Placeholder for the real end-to-end train loop test. To implement: "
        "(1) build a tiny EncoderDecoderGPT via tests/encdec_test_helpers.py, "
        "(2) call train_one_config with cfg.num_iterations=6 and cfg.steering_eval_every=2, "
        "(3) capture steering/eval_time_seconds from wandb logs for steps 2/4/6, "
        "(4) assert eval_times[0] > eval_times[-1] + epsilon (first pays cache-build cost), "
        "(5) assert all metrics at step 6 are finite. "
        "Defer until the slow test harness is ready to reuse without duplication."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_p0_sweep_two_configs_cache_independence():
    """Run train_one_config twice back-to-back with different feature_selection.threshold.

    Asserts (when implemented): both runs complete, both produce finite
    metrics, and the second run's cache is independently built (does not
    reuse the first's).
    """
    pytest.skip(
        "Placeholder for the sweep-harness test. Requires a multi-config "
        "sweep fixture. Defer until P1 or later when the sweep-runner test "
        "infrastructure is mature enough to exercise without duplication."
    )
