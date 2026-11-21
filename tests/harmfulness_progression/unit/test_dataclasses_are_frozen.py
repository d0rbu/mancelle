"""Dataclasses must be frozen so that records can be hashed / set-membered."""

from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

from steering.eval.harmfulness_progression import (
    HarmfulnessMetrics,
    HarmfulnessRecord,
    ProgressionPoint,
    SignalStyle,
)


def test_harmfulness_metrics_is_frozen() -> None:
    m = HarmfulnessMetrics(asr=0.1, mean_score=2.0, safe_rate=0.8)
    with pytest.raises(FrozenInstanceError):
        setattr(m, "asr", 0.2)  # noqa: B010 - mutation is under test


def test_harmfulness_record_is_frozen() -> None:
    r = HarmfulnessRecord(
        model_tag="d12",
        source="base",
        signal="Safety delta-base",
        full_conditioning=False,
        metrics=HarmfulnessMetrics(asr=0.0, mean_score=1.0, safe_rate=1.0),
    )
    with pytest.raises(FrozenInstanceError):
        setattr(r, "model_tag", "d13")  # noqa: B010 - mutation is under test


def test_progression_point_is_frozen() -> None:
    p = ProgressionPoint(
        step=1000,
        phase="pretrain",
        asr=0.0,
        safe_rate=1.0,
        mean_score=1.0,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(p, "step", 2000)  # noqa: B010 - mutation is under test


def test_signal_style_is_frozen() -> None:
    s = SignalStyle(
        label="X",
        color="#000",
        linestyle="-",
        linewidth=1.0,
        marker="o",
        markersize=5.0,
    )
    with pytest.raises(FrozenInstanceError):
        setattr(s, "color", "#fff")  # noqa: B010 - mutation is under test
