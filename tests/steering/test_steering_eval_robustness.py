"""Tests for steering-eval robustness: judge retry + invalid-verdict accounting.

1. `PairwiseJudge.judge()` must retry on transient HTTP 429/5xx (like
   `JudgeClient.score_pair` does), with 3 attempts and exponential backoff. A
   single network hiccup must NOT kill the whole evaluation run.

2. `_compute_level_stats` must NOT silently exclude INVALID / INCONSISTENT
   verdicts from the win-rate denominator. The conservative choice is to count
   them as losses for the method being evaluated, while additionally emitting
   telemetry counters (`n_invalid_verdicts`, `n_inconsistent_verdicts`) so
   downstream consumers can see judge flakiness.
"""

from __future__ import annotations

import logging
from unittest.mock import patch

import pytest
import requests

from steering.eval.harmfulness_judge import JudgeClient
from steering.eval.steering_eval import (
    JudgeChoice,
    LevelStats,
    PairwiseJudge,
    RawJudgment,
    RawMagnitudeData,
    Verdict,
    _compute_level_stats,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_client(tmp_path) -> JudgeClient:
    """Build a JudgeClient with a temp SQLite cache."""
    return JudgeClient(
        model="test-model",
        base_url="https://fake.api",
        api_key="sk-test",
        cache_path=str(tmp_path / "judge.db"),
        requests_per_minute=9999,
    )


def _http_error(status: int) -> requests.HTTPError:
    """Return a requests.HTTPError carrying the given status code."""
    resp = requests.Response()
    resp.status_code = status
    err = requests.HTTPError(f"{status} HTTP error", response=resp)
    return err


def _make_judgment(pair_index: int, verdict: Verdict) -> RawJudgment:
    """Build a RawJudgment record with the given verdict."""
    if verdict == Verdict.WIN:
        ab, ba = JudgeChoice.A, JudgeChoice.B
    elif verdict == Verdict.LOSS:
        ab, ba = JudgeChoice.B, JudgeChoice.A
    elif verdict == Verdict.TIE:
        ab, ba = JudgeChoice.TIE, JudgeChoice.TIE
    elif verdict == Verdict.INCONSISTENT:
        ab, ba = JudgeChoice.A, JudgeChoice.A
    elif verdict == Verdict.INVALID:
        ab, ba = "GARBAGE", "NOPE"  # type: ignore[assignment]
    else:
        raise AssertionError(f"unknown verdict {verdict}")
    return RawJudgment(
        pair_index=pair_index,
        conditioned_text=f"cond_{pair_index}",
        baseline_text=f"base_{pair_index}",
        ab_response=ab,
        ba_response=ba,
        verdict=verdict,
    )


# ---------------------------------------------------------------------------
# P0 fix #1: PairwiseJudge retries on transient HTTP errors
# ---------------------------------------------------------------------------


class TestPairwiseJudgeRetry:
    """PairwiseJudge must wrap the API call with retry+backoff like score_pair."""

    def test_pairwise_judge_retries_on_429(self, tmp_path) -> None:
        """2x 429 then success: PairwiseJudge.judge returns the success response."""
        client = _make_client(tmp_path)
        pj = PairwiseJudge(client)

        call_count = {"n": 0}
        sleep_calls: list[float] = []

        def fake_call(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] <= 2:
                raise _http_error(429)
            return "A"

        with (
            patch.object(client, "_call_api", side_effect=fake_call),
            patch(
                "steering.eval.steering_eval.time.sleep",
                side_effect=lambda d: sleep_calls.append(d),
            ),
        ):
            result = pj.judge("some prompt")

        assert result == "A"
        assert call_count["n"] == 3
        # The rate limiter's `time.sleep` calls (if any) are sub-second and
        # < 0.5s. The retry-backoff sleeps live in `steering_eval`, base 1s
        # then 2s with ±20% jitter — first ≥0.8s, second ≥1.6s. Keep only
        # those for the assertion.
        backoff_sleeps = [d for d in sleep_calls if d >= 0.5]
        assert len(backoff_sleeps) == 2
        assert 0.8 <= backoff_sleeps[0] <= 1.2
        assert 1.6 <= backoff_sleeps[1] <= 2.4

    def test_pairwise_judge_retries_on_500(self, tmp_path) -> None:
        """1x 500 then success: PairwiseJudge.judge returns the success response."""
        client = _make_client(tmp_path)
        pj = PairwiseJudge(client)

        call_count = {"n": 0}

        def fake_call(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise _http_error(500)
            return "TIE"

        with (
            patch.object(client, "_call_api", side_effect=fake_call),
            patch("steering.eval.steering_eval.time.sleep"),
        ):
            result = pj.judge("prompt")

        assert result == "TIE"
        assert call_count["n"] == 2

    def test_pairwise_judge_retries_on_connection_error(self, tmp_path) -> None:
        """Connection errors are also retryable."""
        client = _make_client(tmp_path)
        pj = PairwiseJudge(client)

        call_count = {"n": 0}

        def fake_call(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            if call_count["n"] == 1:
                raise requests.ConnectionError("connection refused")
            return "B"

        with (
            patch.object(client, "_call_api", side_effect=fake_call),
            patch("steering.eval.steering_eval.time.sleep"),
        ):
            result = pj.judge("prompt")

        assert result == "B"
        assert call_count["n"] == 2

    def test_pairwise_judge_fails_after_max_retries(self, tmp_path) -> None:
        """4x 429 exhausts retries: PairwiseJudge.judge raises."""
        client = _make_client(tmp_path)
        pj = PairwiseJudge(client)

        call_count = {"n": 0}

        def fake_call(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            raise _http_error(429)

        with (
            patch.object(client, "_call_api", side_effect=fake_call),
            patch("steering.eval.steering_eval.time.sleep"),
            pytest.raises(requests.HTTPError),
        ):
            pj.judge("prompt")

        # Max 3 attempts.
        assert call_count["n"] == 3

    def test_pairwise_judge_non_retryable_http_raises_immediately(
        self, tmp_path
    ) -> None:
        """4xx (not 429) is NOT retryable — raise immediately."""
        client = _make_client(tmp_path)
        pj = PairwiseJudge(client)

        call_count = {"n": 0}

        def fake_call(messages: list[dict[str, str]]) -> str:
            call_count["n"] += 1
            raise _http_error(401)

        with (
            patch.object(client, "_call_api", side_effect=fake_call),
            patch("steering.eval.steering_eval.time.sleep"),
            pytest.raises(requests.HTTPError),
        ):
            pj.judge("prompt")

        assert call_count["n"] == 1

    def test_pairwise_judge_success_first_attempt_no_sleep(self, tmp_path) -> None:
        """Happy path: no retries, no sleeps."""
        client = _make_client(tmp_path)
        pj = PairwiseJudge(client)

        sleep_calls: list[float] = []

        with (
            patch.object(client, "_call_api", return_value="A") as mock_call,
            patch(
                "steering.eval.steering_eval.time.sleep",
                side_effect=lambda d: sleep_calls.append(d),
            ),
        ):
            result = pj.judge("prompt")

        assert result == "A"
        assert mock_call.call_count == 1
        assert sleep_calls == []


# ---------------------------------------------------------------------------
# P0 fix #2: INVALID / INCONSISTENT verdicts count as losses in denominator
# ---------------------------------------------------------------------------


class TestInvalidVerdictAccounting:
    """`_compute_level_stats` must include INVALID/INCONSISTENT in the denominator."""

    def test_invalid_verdicts_count_as_losses(self) -> None:
        """wins=5, invalid=5 → denominator=10, numerator=5 → win_rate=0.5 (NOT 1.0)."""
        judgments = [_make_judgment(i, Verdict.WIN) for i in range(5)] + [
            _make_judgment(i + 5, Verdict.INVALID) for i in range(5)
        ]
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=judgments,
            )
        }
        level_win_rates, _, _, stats = _compute_level_stats(feat_mag_data, [0])

        assert level_win_rates[0] == pytest.approx(0.5)
        assert isinstance(stats[0], LevelStats)
        assert stats[0].n_invalid_verdicts == 5
        assert stats[0].n_inconsistent_verdicts == 0
        assert stats[0].n_total == 10

    def test_inconsistent_verdicts_count_as_losses(self) -> None:
        """wins=3, inconsistent=7 → denominator=10, numerator=3 → win_rate=0.3."""
        judgments = [_make_judgment(i, Verdict.WIN) for i in range(3)] + [
            _make_judgment(i + 3, Verdict.INCONSISTENT) for i in range(7)
        ]
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=judgments,
            )
        }
        level_win_rates, _, _, stats = _compute_level_stats(feat_mag_data, [0])

        assert level_win_rates[0] == pytest.approx(0.3)
        assert stats[0].n_invalid_verdicts == 0
        assert stats[0].n_inconsistent_verdicts == 7
        assert stats[0].n_total == 10

    def test_mixed_verdicts_denominator_includes_all(self) -> None:
        """Mixed mix: wins=2, losses=1, ties=1, inconsistent=3, invalid=3 → total=10."""
        judgments = (
            [_make_judgment(0, Verdict.WIN), _make_judgment(1, Verdict.WIN)]
            + [_make_judgment(2, Verdict.LOSS)]
            + [_make_judgment(3, Verdict.TIE)]
            + [_make_judgment(4 + i, Verdict.INCONSISTENT) for i in range(3)]
            + [_make_judgment(7 + i, Verdict.INVALID) for i in range(3)]
        )
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=judgments,
            )
        }
        level_win_rates, _, _, stats = _compute_level_stats(feat_mag_data, [0])

        assert level_win_rates[0] == pytest.approx(2 / 10)
        assert stats[0].n_wins == 2
        assert stats[0].n_losses == 1
        assert stats[0].n_ties == 1
        assert stats[0].n_inconsistent_verdicts == 3
        assert stats[0].n_invalid_verdicts == 3
        assert stats[0].n_total == 10

    def test_no_invalid_verdicts_preserves_standard_win_rate(self) -> None:
        """When there are zero invalid/inconsistent judgments, win rate is unchanged."""
        judgments = (
            [_make_judgment(i, Verdict.WIN) for i in range(4)]
            + [_make_judgment(4, Verdict.LOSS)]
            + [_make_judgment(5, Verdict.TIE)]
        )
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=judgments,
            )
        }
        level_win_rates, _, _, stats = _compute_level_stats(feat_mag_data, [0])

        # 4 wins / (4 + 1 + 1 + 0 + 0) = 4/6 ≈ 0.667
        assert level_win_rates[0] == pytest.approx(4 / 6)
        assert stats[0].n_invalid_verdicts == 0
        assert stats[0].n_inconsistent_verdicts == 0
        assert stats[0].n_total == 6

    def test_empty_judgments_list_produces_zero(self) -> None:
        """No judgments → win_rate=0.0 and counters all zero."""
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=[],
            )
        }
        level_win_rates, _, _, stats = _compute_level_stats(feat_mag_data, [0])

        assert level_win_rates[0] == 0.0
        assert stats[0].n_total == 0
        assert stats[0].n_invalid_verdicts == 0
        assert stats[0].n_inconsistent_verdicts == 0

    def test_invalid_rate_over_5pct_logs_warning(self, caplog) -> None:
        """When invalid+inconsistent rate exceeds 5% of judgments, log a warning."""
        # 10% invalid: 1 invalid, 9 wins → rate 10%.
        judgments = [_make_judgment(i, Verdict.WIN) for i in range(9)] + [
            _make_judgment(9, Verdict.INVALID)
        ]
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=judgments,
            )
        }

        # loguru sinks logs to a pytest caplog bridge.
        from loguru import logger

        caplog.set_level(logging.WARNING)
        # Forward loguru logs to caplog.
        handler_id = logger.add(
            lambda message: caplog.handler.emit(
                logging.LogRecord(
                    name="loguru",
                    level=logging.WARNING,
                    pathname="",
                    lineno=0,
                    msg=str(message),
                    args=(),
                    exc_info=None,
                )
            ),
            level="WARNING",
        )
        try:
            _compute_level_stats(feat_mag_data, [0])
        finally:
            logger.remove(handler_id)

        warning_text = caplog.text.lower()
        assert "invalid" in warning_text
        # Either a float "0.10" or an int "10%" rendering is acceptable.
        assert "10" in warning_text or "0.1" in warning_text

    def test_invalid_rate_at_5pct_does_not_log_warning(self, caplog) -> None:
        """Exactly 5% invalid does not trigger the warning (threshold is > 5%)."""
        # 1 invalid out of 20 = 5%.
        judgments = [_make_judgment(i, Verdict.WIN) for i in range(19)] + [
            _make_judgment(19, Verdict.INVALID)
        ]
        feat_mag_data = {
            0: RawMagnitudeData(
                level=0,
                conditioned_samples=[],
                baseline_samples=[],
                judgments=judgments,
            )
        }

        from loguru import logger

        captured: list[str] = []
        handler_id = logger.add(lambda message: captured.append(str(message)), level="WARNING")
        try:
            _compute_level_stats(feat_mag_data, [0])
        finally:
            logger.remove(handler_id)

        # No warning about high invalid rate.
        high_invalid_warnings = [
            msg for msg in captured if "invalid" in msg.lower() and "rate" in msg.lower()
        ]
        assert high_invalid_warnings == []
