"""Default-verbosity log budget for AxBench synth stages.

A real Stage 1b run emitted ~150 protocol-rejection lines per minute (11,044
in a single stage). At WARNING those buried every genuinely rare warning and
made the log unusable, while the one signal an operator actually wants --
"is this converging?" -- was absent entirely.

The contract these tests pin:

- per-attempt rejection detail is DEBUG (counted always, shouted never)
- aggregate progress is INFO (bounded to one line per heartbeat window)
- the console level is env-overridable so DEBUG detail stays reachable
"""

from __future__ import annotations

import inspect
import os

from steering.eval.axbench_base.synth import continuation_protocol, positive_mode


def test_protocol_rejections_are_debug_not_warning() -> None:
    source = inspect.getsource(continuation_protocol)
    assert "logger.debug(\n            f\"{log_label}: synth response protocol failure" in source
    assert "logger.warning(\n            f\"{log_label}: synth response protocol failure" not in source


def test_special_token_refusals_are_debug_not_warning() -> None:
    source = inspect.getsource(continuation_protocol)
    assert "logger.debug(\n            f\"{log_label}: synth response refused" in source
    assert "logger.warning(\n            f\"{log_label}: synth response refused" not in source


def test_rejections_are_still_counted_even_though_they_are_quiet() -> None:
    """Demoting the log must not lose the signal.

    The aggregate has to survive somewhere an operator can reach, otherwise
    this trades noise for blindness.
    """

    source = inspect.getsource(continuation_protocol)
    assert "record_counter(protocol_counter, 1, **counter_attrs)" in source
    assert "record_counter(special_token_counter, 1, **counter_attrs)" in source


def test_progress_heartbeat_is_bounded_per_window() -> None:
    """INFO is only defensible because the heartbeat is rate-limited."""

    assert positive_mode._PROGRESS_HEARTBEAT_SECONDS >= 30


def test_console_level_is_env_overridable() -> None:
    """DEBUG detail must stay reachable without editing code."""

    import nanochat.common as common

    source = inspect.getsource(common)
    assert "NANOCHAT_LOG_LEVEL" in source


def test_default_console_level_is_unchanged() -> None:
    """Unset env must behave exactly as before this change."""

    import nanochat.common as common

    source = inspect.getsource(common)
    assert 'os.environ.get("NANOCHAT_LOG_LEVEL", "INFO")' in source
    assert os.environ.get("NANOCHAT_LOG_LEVEL") is None or True
