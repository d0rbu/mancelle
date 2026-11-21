"""SFT extraction window advisory (#1634)."""

from __future__ import annotations

import pytest
from loguru import logger

from steering import extract_features as ef


@pytest.fixture
def captured():
    records: list[tuple[str, str]] = []
    handle = logger.add(lambda msg: records.append((msg.record["level"].name, msg.record["message"])))
    try:
        yield records
    finally:
        logger.remove(handle)


@pytest.mark.parametrize("phase", [None, "base"])
def test_base_phase_says_nothing(captured, phase):
    ef._log_sft_context_window_advice(512, phase)
    assert captured == []


@pytest.mark.parametrize("window", [128, 1024, ef.SFT_RECOMMENDED_MIN_CONTEXT_WINDOW - 1])
def test_small_sft_window_warns_with_the_rule(captured, window):
    ef._log_sft_context_window_advice(window, "standard_steered_sft")
    assert len(captured) == 1
    level, message = captured[0]
    assert level == "WARNING"
    assert f"context_window={window}" in message
    assert "2 x the trainer's max_seq_len" in message
    assert "#1634" in message


@pytest.mark.parametrize("window", [ef.SFT_RECOMMENDED_MIN_CONTEXT_WINDOW, 4096])
def test_adequate_sft_window_logs_info_only(captured, window):
    ef._log_sft_context_window_advice(window, "standard_sft_val")
    assert [level for level, _ in captured] == ["INFO"]
    assert f"context_window={window}" in captured[0][1]


def test_recommended_minimum_matches_production_sft_shards():
    # Production SFT shards (l22 f65k gemma-3-4b-pt) were extracted at 2048 for
    # the trainer's default max_seq_len=1024; the advisory threshold is that value.
    assert ef.SFT_RECOMMENDED_MIN_CONTEXT_WINDOW == 2048


def test_main_calls_the_advisory_after_phase_validation():
    import inspect

    src = inspect.getsource(ef.main)
    assert "_validate_segment_mode_dataset_phase(segment_mode, dataset_phase)" in src
    after = src.split("_validate_segment_mode_dataset_phase(segment_mode, dataset_phase)")[1]
    assert "_log_sft_context_window_advice(context_window, dataset_phase)" in after
