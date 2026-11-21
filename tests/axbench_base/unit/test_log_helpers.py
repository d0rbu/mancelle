"""Tests for the rank-aware logging helpers used by Stage 0c
instrumentation (PR-1).

Covers:

* ``is_log_rank()`` — defensive RANK env-var gating used by all new
  PR-1 ``logger.info`` calls so that non-zero ranks stay silent even
  when the standard ``nanochat.common`` loguru sink configuration was
  not loaded (e.g., when Stage 0c is invoked off the standard CLI
  path).
* ``StepwiseTimer`` — extracts the manual
  ``_t_start = time.monotonic()`` / ``elapsed_ms = int(... * 1000)``
  dance previously crowding ``_load_feature_extraction_pair`` into a
  small reusable helper. Caller passes a per-step log message; the
  helper appends ``[elapsed_ms=X]`` for the just-completed step.
"""

from __future__ import annotations

import time

import pytest

from steering.eval.axbench_base import _log_helpers as helpers
from steering.eval.axbench_base._log_helpers import (
    StepwiseTimer,
    is_log_rank,
)
from tests.axbench_base._loguru_capture import capture_loguru_info


def test_is_log_rank_returns_true_when_rank_unset(monkeypatch):
    """Default (single-process) invocations have RANK unset and must log."""
    monkeypatch.delenv("RANK", raising=False)
    assert is_log_rank() is True


def test_is_log_rank_returns_true_when_rank_zero(monkeypatch):
    """RANK=0 in a torchrun launch must log normally (rank 0 is the chatty rank)."""
    monkeypatch.setenv("RANK", "0")
    assert is_log_rank() is True


@pytest.mark.parametrize("rank_value", ["1", "2", "3", "7"])
def test_is_log_rank_returns_false_for_nonzero_ranks(monkeypatch, rank_value):
    """Non-zero ranks must stay silent for the new heartbeat / progress
    logs even if the loguru sink was not rank-aware-configured."""
    monkeypatch.setenv("RANK", rank_value)
    assert is_log_rank() is False


def test_is_log_rank_handles_malformed_rank_env(monkeypatch):
    """A malformed RANK value (e.g. ``""`` or ``"oops"``) must not crash
    the gating helper. The helper should treat unparseable values as
    "we cannot prove this is rank 0" and stay silent — better safe than
    spam."""
    monkeypatch.setenv("RANK", "")
    assert is_log_rank() is False
    monkeypatch.setenv("RANK", "oops")
    assert is_log_rank() is False


def test_stepwise_timer_step_emits_log_with_elapsed_ms(monkeypatch):
    """Each ``step()`` call appends ``[elapsed_ms=X]`` and emits the
    composed message at INFO level on rank 0."""
    monkeypatch.delenv("RANK", raising=False)
    timer = StepwiseTimer()
    with capture_loguru_info() as captured:
        time.sleep(0.03)
        timer.step("[runtime_state] step A done")
        time.sleep(0.02)
        timer.step("[runtime_state] step B done")

    assert len(captured) == 2, f"expected 2 log lines, got {captured!r}"
    for line in captured:
        assert "[elapsed_ms=" in line
    # First log line should reflect ~30ms; second ~20ms.
    first_ms = int(captured[0].rsplit("=", 1)[1].rstrip("]"))
    assert 20 <= first_ms <= 500, (
        f"first step elapsed_ms={first_ms}; expected ~30ms ± slack"
    )


def test_stepwise_timer_silent_on_non_zero_rank(monkeypatch):
    """When RANK != 0, the timer emits nothing — but still updates its
    internal mark so total_ms() reflects wall time."""
    monkeypatch.setenv("RANK", "1")
    timer = StepwiseTimer()
    with capture_loguru_info() as captured:
        time.sleep(0.02)
        timer.step("[runtime_state] should be suppressed")
    assert captured == [], (
        f"non-zero ranks must not emit step logs; got {captured!r}"
    )
    total = timer.total_ms()
    assert total >= 10, f"total_ms still tracks real time; got {total}"


def test_stepwise_timer_total_ms_measures_from_construction(monkeypatch):
    """``total_ms()`` returns wall-clock since construction, regardless
    of how many ``step()`` calls have been made."""
    monkeypatch.delenv("RANK", raising=False)
    timer = StepwiseTimer()
    time.sleep(0.04)
    total = timer.total_ms()
    assert 30 <= total <= 5000


def test_stepwise_timer_step_forwards_kwargs_into_message(monkeypatch):
    """Caller-supplied kwargs are forwarded into the loguru
    substitution so messages can use ``{device}`` etc. like the rest
    of the codebase."""
    monkeypatch.delenv("RANK", raising=False)
    timer = StepwiseTimer()
    with capture_loguru_info() as captured:
        timer.step(
            "[runtime_state] loaded model on {device}",
            device="cuda:0",
        )
    assert any("loaded model on cuda:0" in line for line in captured), (
        f"kwargs not interpolated; got {captured!r}"
    )
    assert any("[elapsed_ms=" in line for line in captured)


def test_helpers_module_exports_expected_names():
    """The public surface of the helpers module is small and stable.
    Tests pin the public names to catch accidental renames."""
    assert hasattr(helpers, "is_log_rank")
    assert hasattr(helpers, "StepwiseTimer")
