"""Tier-3 slow smoke: progression sweep crash-and-resume produces byte-equal results.

@pytest.mark.slow + PHASE2_SMOKE_HAS_GEMMA=1.

Pipeline:
  1. Build context cache + target cache (same as Smoke 1).
  2. Run progression sweep, kill mid-stream after second checkpoint's first config.
  3. Resume. Assert final results.json is byte-equal to an uninterrupted run.
"""

from __future__ import annotations

import os

import pytest

pytestmark = pytest.mark.slow


def _have_real_gemma_artifacts() -> bool:
    return os.environ.get("PHASE2_SMOKE_HAS_GEMMA", "0") == "1"


@pytest.mark.skipif(
    not _have_real_gemma_artifacts(),
    reason="real Gemma+GemmaScope artifacts unavailable in this environment",
)
def test_phase_d_smoke_progression_resume_after_crash_real_gemma_sae():
    """Crash-and-resume byte-equality smoke."""
    # ... same setup as Smoke 1 ...
    # Operator-side: this smoke is a template. Adapt the kill point and
    # results.json comparison to your checkpoint layout.
