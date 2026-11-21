"""Slow full-tier smoke: run axbench-bpb end-to-end on a real axbench-base run.

Gated behind both ``@pytest.mark.slow`` and ``@pytest.mark.xfail(run=False)``
(pending real axbench-base fixture + GPU). Unskipped once
``build_mini_axbench_base_run`` (Task 13.1) produces a real Stage 0-3 tree
and the model-loader wire-up in ``RuntimeState`` (Task 11.4 scaffold →
Task 13.x real) lands.

Slice 17 audit (2026-05-05): converted from ``@pytest.mark.skip`` to
``@pytest.mark.xfail(strict=True, run=False)`` so a future "ready" body
that accidentally returns without asserting is caught by xfail-strict
rather than silently passing.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    run=False,
    reason=(
        "Task 15.1 placeholder: slow full-tier run against real axbench-base "
        "run + GPU. Requires mini_axbench_base_run fixture (Task 13.1) and model-"
        "loader wire-up (Task 11.4 → 13.x). Run via: "
        "`uv run pytest -m slow tests/axbench_bpb/slow/ --gpu`. "
        "Tracked in audit findings.md slice-17 X2 §2.1."
    ),
)
def test_full_tier_produces_leaderboard(tmp_path):
    """Run axbench-bpb all stages on a 20-concept axbench-base run.

    Assertions once unskipped:
      - leaderboard.csv produced under ``tmp_path / "bpb_aggregate"``
      - row count == n_methods × n_modes × n_targets (25 × 2 × 2 = 100 upper bound,
        fewer after truncation-rate drops)
      - primary_metric column present; values descending
    """
    pytest.fail(
        "placeholder should never execute while @xfail(run=False) is in place"
    )
