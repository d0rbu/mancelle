"""render_standalone_progression: cross-lineage chart structural tests."""

from __future__ import annotations

import numpy as np
import pytest

from steering.eval.harmfulness_progression import (
    DEFAULT_SFT_STEPS,
    DEFAULT_SIGNAL_ORDER,
    DEFAULT_SIGNAL_STYLES,
    HarmfulnessMetric,
    ProgressionPoint,
    render_standalone_progression,
)

_LAST_PRETRAIN_STEP = 54_600


def _d13_progression() -> list[ProgressionPoint]:
    return [
        ProgressionPoint(step=53000, phase="pretrain", asr=0.20, safe_rate=0.80, mean_score=2.5),
        ProgressionPoint(step=54600, phase="pretrain", asr=0.18, safe_rate=0.82, mean_score=2.4),
        ProgressionPoint(step=900, phase="safety_sft", asr=0.05, safe_rate=0.95, mean_score=1.5),
        ProgressionPoint(
            step=1140, phase="capabilities_sft", asr=0.07, safe_rate=0.93, mean_score=1.6
        ),
    ]


def _d12_progression() -> list[ProgressionPoint]:
    return [
        ProgressionPoint(step=53000, phase="pretrain", asr=0.15, safe_rate=0.85, mean_score=2.2),
        ProgressionPoint(step=54600, phase="pretrain", asr=0.13, safe_rate=0.87, mean_score=2.1),
    ]


def _sft_records(harmfulness_record_factory):
    return [
        harmfulness_record_factory(
            source="base/safety_cond",
            signal="Safety delta-base",
            asr=0.06,
            safe_rate=0.94,
            mean_score=1.6,
        ),
        harmfulness_record_factory(
            source="base/safety_cond/capabilities_cond",
            signal="Safety delta-base",
            asr=0.04,
            safe_rate=0.96,
            mean_score=1.4,
        ),
    ]


def test_d13_line_includes_pretrain_plus_sft_branch(
    harmfulness_record_factory,
) -> None:
    fig = render_standalone_progression(
        _sft_records(harmfulness_record_factory),
        _d13_progression(),
        _d12_progression(),
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        last_pretrain_step=_LAST_PRETRAIN_STEP,
        sft_steps=DEFAULT_SFT_STEPS,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
    )
    ax = fig.axes[0]
    # The d13 line covers 4 points: 2 pretrain + 1 safety_sft + 1 capabilities_sft.
    d13_lines = [line for line in ax.lines if str(line.get_label()).startswith("d13_base")]
    assert len(d13_lines) == 1
    xs = np.asarray(d13_lines[0].get_xdata()).tolist()
    assert len(xs) == 4
    assert xs[0] == 53000
    assert xs[1] == 54600
    assert xs[2] == _LAST_PRETRAIN_STEP + DEFAULT_SFT_STEPS["d13_base"][0]
    assert xs[3] == (
        _LAST_PRETRAIN_STEP + DEFAULT_SFT_STEPS["d13_base"][0] + DEFAULT_SFT_STEPS["d13_base"][1]
    )


def test_d12_baseline_progression_covers_pretrain_only(
    harmfulness_record_factory,
) -> None:
    fig = render_standalone_progression(
        _sft_records(harmfulness_record_factory),
        _d13_progression(),
        _d12_progression(),
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        last_pretrain_step=_LAST_PRETRAIN_STEP,
        sft_steps=DEFAULT_SFT_STEPS,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
    )
    ax = fig.axes[0]
    d12_lines = [line for line in ax.lines if str(line.get_label()).startswith("d12_baseline")]
    assert len(d12_lines) == 1
    xs = np.asarray(d12_lines[0].get_xdata()).tolist()
    assert xs == [53000, 54600]


def test_sft_branches_anchor_at_last_pretrain_step(
    harmfulness_record_factory,
) -> None:
    fig = render_standalone_progression(
        _sft_records(harmfulness_record_factory),
        _d13_progression(),
        _d12_progression(),
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        last_pretrain_step=_LAST_PRETRAIN_STEP,
        sft_steps=DEFAULT_SFT_STEPS,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
    )
    ax = fig.axes[0]
    # SFT branch lines are conditioned-only and labeled "cond SFT — ...".
    sft_lines = [line for line in ax.lines if " SFT — " in str(line.get_label())]
    assert len(sft_lines) >= 1
    # Each SFT line's first x must be last_pretrain_step (the anchor).
    for line in sft_lines:
        xs = np.asarray(line.get_xdata()).tolist()
        assert xs[0] == _LAST_PRETRAIN_STEP


def test_signal_with_only_safety_val_and_no_pretrain_anchor_skipped(
    harmfulness_record_factory,
) -> None:
    """Covers harmfulness_progression.py:778-779 — when a signal has only
    safety_val (no cap_val) AND there is no pretrain_anchor (empty d12
    progression), `len(xs) < 2` triggers `continue` and no line is drawn.
    """
    # Only safety record exists for "Safety delta-base"; capabilities is
    # missing; no pretrain anchor (empty d12).
    sft_records = [
        harmfulness_record_factory(
            source="base/safety_cond",
            signal="Safety delta-base",
            asr=0.06,
            safe_rate=0.94,
            mean_score=1.6,
        ),
    ]
    fig = render_standalone_progression(
        sft_records,
        _d13_progression(),
        d12_progression=[],  # empty → pretrain_anchor is None
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        last_pretrain_step=_LAST_PRETRAIN_STEP,
        sft_steps=DEFAULT_SFT_STEPS,
        signal_order=["Safety delta-base"],
        signal_styles=DEFAULT_SIGNAL_STYLES,
    )
    ax = fig.axes[0]
    # No "cond SFT — Safety delta-base" line should be drawn (xs has length 1).
    sft_lines = [line for line in ax.lines if " SFT — Safety delta-base" in str(line.get_label())]
    assert sft_lines == []


def test_signal_with_only_cap_val_and_pretrain_anchor_drawn(
    harmfulness_record_factory,
) -> None:
    """Covers harmfulness_progression.py:771-779 — a signal with only
    cap_val (no safety_val) AND a pretrain_anchor produces exactly 2
    points (anchor + cap), and the line IS drawn.
    """
    # Only capabilities record exists for "Safety delta-base".
    sft_records = [
        harmfulness_record_factory(
            source="base/safety_cond/capabilities_cond",
            signal="Safety delta-base",
            asr=0.04,
            safe_rate=0.96,
            mean_score=1.4,
        ),
    ]
    fig = render_standalone_progression(
        sft_records,
        _d13_progression(),
        _d12_progression(),  # non-empty → pretrain_anchor is set
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        last_pretrain_step=_LAST_PRETRAIN_STEP,
        sft_steps=DEFAULT_SFT_STEPS,
        signal_order=["Safety delta-base"],
        signal_styles=DEFAULT_SIGNAL_STYLES,
    )
    ax = fig.axes[0]
    sft_lines = [line for line in ax.lines if " SFT — Safety delta-base" in str(line.get_label())]
    assert len(sft_lines) == 1
    xs = np.asarray(sft_lines[0].get_xdata()).tolist()
    ys = np.asarray(sft_lines[0].get_ydata()).tolist()
    # Exactly 2 points: pretrain anchor (last_pretrain_step) + cap step.
    assert len(xs) == 2
    assert xs[0] == _LAST_PRETRAIN_STEP
    expected_cap_x = (
        _LAST_PRETRAIN_STEP + DEFAULT_SFT_STEPS["cond"][0] + DEFAULT_SFT_STEPS["cond"][1]
    )
    assert xs[1] == expected_cap_x
    # The y at the cap step should match the recorded cap_val.
    assert ys[1] == pytest.approx(0.04)
