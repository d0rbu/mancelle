"""Slow: Spearman rank correlation between axbench-bpb and axbench-base leaderboards.

Research gate: the BPB leaderboard should agree with axbench-base's LLM-judge
leaderboard on the relative ordering of ours-family methods. Formal criterion
is Spearman ρ ≥ 0.3 on the ours-* subset (per spec §15). Lower correlation
would suggest BPB and judge scores measure genuinely different things, which
is a flag for researchers but not a v1-ship blocker.
"""
from __future__ import annotations

import pytest


@pytest.mark.slow
@pytest.mark.xfail(
    strict=True,
    run=False,
    reason=(
        "Task 15.2 placeholder: requires both a completed axbench-base run "
        "(with judge leaderboard) AND an axbench-bpb run on the same source. "
        "Unskipped once both pipelines produce outputs side-by-side. "
        "Tracked in audit findings.md slice-17 X2 §2.1."
    ),
)
def test_ours_methods_spearman_correlation_positive(tmp_path):
    """Spearman ρ between the two leaderboards' ours_* method rankings ≥ 0.3.

    Implementation sketch once unskipped:
      1. Load bpb leaderboard: ``pd.read_csv(bpb_dir / "leaderboard.csv")``
      2. Load axbench-base leaderboard: ``pd.read_csv(base_dir / "leaderboard.csv")``
      3. Join on method_name; filter to ours_* rows
      4. ``scipy.stats.spearmanr(bpb_scores, judge_scores)``
      5. ``assert rho >= 0.3``

    A negative / near-zero ρ is a scientific finding, not a code bug: it means
    BPB and judge-based metrics disagree on which ours-family method works
    best. The gate catches implementation bugs (e.g., reversed sign in the
    conditioned margin, wrong best_coef selection) that would manifest as
    negative rank correlation.
    """
    pytest.fail("placeholder should never execute while @skip is in place")
