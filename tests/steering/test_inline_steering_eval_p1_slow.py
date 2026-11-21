"""Slow / GPU-gated tests for P1 inline steering eval perf.

These tests run a real (short) training loop to verify the amortized
generation and batched probe interact correctly with the actual
DDP / FP8 / torch.compile path.

Gated on torch.cuda.is_available() — skipped on CPU-only machines.
Marked @pytest.mark.slow at the module level — deselected by default
(see pyproject.toml's `addopts = "-m 'not slow'"`).
"""

from __future__ import annotations

import pytest
import torch

pytestmark = pytest.mark.slow


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required for full train-loop test")
def test_p1_full_train_loop_generation_caching(tmp_path):
    """Run train_one_config with steering_eval_every=2 for ~6 steps and
    verify the P1 amortized generation produces correct metrics and a
    measurable wall-clock improvement vs. the pre-P1 baseline.

    Asserts (when implemented):
    - training runs to completion without errors
    - all eval metrics are finite at step 6
    - steering/eval_time_seconds at step 6 is measurably lower than a
      reference run that monkeypatches generate_samples_for_condition
      to call generate_completion per sample
    """
    pytest.skip(
        "Placeholder for the real end-to-end P1 train loop test. To implement: "
        "(1) build a tiny EncoderDecoderGPT via tests/encdec_test_helpers.py, "
        "(2) call train_one_config with cfg.num_iterations=6 and "
        "cfg.steering_eval_every=2, "
        "(3) record eval_time_seconds for step 6, "
        "(4) re-run with generate_samples_for_condition monkeypatched to the "
        "pre-P1 per-sample loop and record the reference eval_time_seconds, "
        "(5) assert P1 eval_time_seconds < reference - epsilon. "
        "Defer until the slow test harness is ready to reuse without duplication."
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_p1_numerical_stability_large_n_gt_features():
    """Stress-test the batched matmul with n_gt_features=256 on real-scale
    SAE weights. Asserts no NaN/Inf propagation in the final metrics."""
    pytest.skip(
        "Placeholder for numerical stability stress test. Requires a real "
        "SAE (or a larger synthetic SAE) with d_model=2304 and 16k features. "
        "Defer until the GPU test harness is mature."
    )
