"""Pin the total==0 behavior of _compute_level_stats (map B-9).

When a magnitude level has no judgments, ``total`` is 0 and the
implementation falls back to ``win_rate = 0.0``. The plan flagged this as
"conservative-by-design" — the docstring justifies it for INVALID/INCONSISTENT
verdicts but not for the empty case. This test pins the current behavior so
any future change is intentional.

This test also covers the broader contract that a single-judgment WIN
yields win_rate=1.0, and that INCONSISTENT/INVALID counts toward the
denominator (both reduce win_rate).
"""

from __future__ import annotations

from steering.eval.steering_eval import (
    LevelStats,
    RawJudgment,
    RawMagnitudeData,
    Verdict,
    _compute_level_stats,
)


def _md(verdicts: list[Verdict]) -> RawMagnitudeData:
    """Build a minimal RawMagnitudeData with the given verdicts."""
    judgments = [
        RawJudgment(
            pair_index=i,
            conditioned_text="",
            baseline_text="",
            ab_response="",
            ba_response="",
            verdict=v,
        )
        for i, v in enumerate(verdicts)
    ]
    return RawMagnitudeData(
        level=1,
        conditioned_samples=[],
        baseline_samples=[],
        judgments=judgments,
    )


class TestTotalZero:
    def test_no_judgments_yields_zero_win_rate(self) -> None:
        feat = {1: _md([])}
        rates, lo, hi, stats = _compute_level_stats(feat, mag_levels=[1])
        assert rates == [0.0]
        assert stats[0] == LevelStats(0, 0, 0, 0, 0, 0)

    def test_single_win(self) -> None:
        feat = {1: _md([Verdict.WIN])}
        rates, lo, hi, stats = _compute_level_stats(feat, mag_levels=[1])
        assert rates == [1.0]

    def test_inconsistent_counts_as_loss(self) -> None:
        """1 WIN + 1 INCONSISTENT → win_rate = 0.5 (not 1.0)."""
        feat = {1: _md([Verdict.WIN, Verdict.INCONSISTENT])}
        rates, _, _, stats = _compute_level_stats(feat, mag_levels=[1])
        assert rates[0] == 0.5
        assert stats[0].n_inconsistent_verdicts == 1

    def test_invalid_counts_as_loss(self) -> None:
        """2 WIN + 2 INVALID → win_rate = 0.5."""
        feat = {1: _md([Verdict.WIN, Verdict.WIN, Verdict.INVALID, Verdict.INVALID])}
        rates, _, _, stats = _compute_level_stats(feat, mag_levels=[1])
        assert rates[0] == 0.5
        assert stats[0].n_invalid_verdicts == 2

    def test_tie_does_not_count_as_win(self) -> None:
        """TIE → counts toward total but adds no wins."""
        feat = {1: _md([Verdict.WIN, Verdict.TIE])}
        rates, _, _, stats = _compute_level_stats(feat, mag_levels=[1])
        assert rates[0] == 0.5
        assert stats[0].n_ties == 1
