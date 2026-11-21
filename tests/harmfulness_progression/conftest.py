"""Shared fixtures for harmfulness_progression test suite."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

# Make project root importable.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from steering.eval.harmfulness_progression import (
    HarmfulnessMetrics,
    HarmfulnessRecord,
)


@pytest.fixture(autouse=True)
def _close_figures():
    """Ensure matplotlib state doesn't leak between tests."""
    yield
    plt.close("all")


@pytest.fixture
def harmfulness_record_factory():
    """Build HarmfulnessRecord instances inline without YAML files.

    Usage:
        rec = harmfulness_record_factory(
            model_tag="d12_baseline",
            source="base/safety_cond",
            signal="Safety delta-base",
            asr=0.3,
            safe_rate=0.6,
            mean_score=2.5,
            full_conditioning=False,
        )
    """

    def _factory(
        *,
        model_tag: str = "d12_baseline",
        source: str = "base",
        signal: str = "Safety delta-base",
        asr: float = 0.0,
        safe_rate: float = 1.0,
        mean_score: float = 1.0,
        compliance_rate: float = 1.0,
        full_conditioning: bool = False,
    ) -> HarmfulnessRecord:
        return HarmfulnessRecord(
            model_tag=model_tag,
            source=source,
            signal=signal,
            full_conditioning=full_conditioning,
            metrics=HarmfulnessMetrics(
                asr=asr,
                mean_score=mean_score,
                safe_rate=safe_rate,
                compliance_rate=compliance_rate,
            ),
        )

    return _factory
