"""Functional tests of render_progression_chart: structural assertions on
the returned Figure (no golden-image diffing)."""

from __future__ import annotations

import matplotlib
import numpy as np

matplotlib.use("Agg")

from matplotlib.colors import to_rgba
from matplotlib.figure import Figure

from steering.eval.harmfulness_progression import (
    DEFAULT_COND_SOURCE_MAP,
    DEFAULT_SIGNAL_ORDER,
    DEFAULT_SIGNAL_STYLES,
    DEFAULT_STAGES,
    HarmfulnessMetric,
    render_progression_chart,
)

STAGES = list(DEFAULT_STAGES)


def _three_records(harmfulness_record_factory):
    return [
        harmfulness_record_factory(
            source="base",
            signal="Safety delta-base",
            asr=0.10,
            safe_rate=0.9,
        ),
        harmfulness_record_factory(
            source="base/safety_cond",
            signal="Safety delta-base",
            asr=0.05,
            safe_rate=0.95,
        ),
        harmfulness_record_factory(
            source="base/safety_cond/capabilities_cond",
            signal="Safety delta-base",
            asr=0.02,
            safe_rate=0.98,
        ),
    ]


def test_returns_figure_object(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
    )
    assert isinstance(fig, Figure)


def test_one_line_per_signal_with_data(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
    )
    ax = fig.axes[0]
    # Only Safety delta-base has data. Exclude axvlines (constant x).
    data_lines = [line for line in ax.lines if len(set(np.asarray(line.get_xdata()).tolist())) > 1]
    assert len(data_lines) == 1


def test_signals_with_no_data_omitted_from_legend(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,  # 9 signals; only 1 has data
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
    )
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    legend_labels = [t.get_text() for t in legend.get_texts()]
    assert legend_labels == ["Safety delta-base"]


def test_xticks_match_stage_labels(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
    )
    ax = fig.axes[0]
    xticklabels = [t.get_text() for t in ax.get_xticklabels()]
    assert xticklabels == STAGES


def test_phase_transition_vlines_drawn(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
    )
    ax = fig.axes[0]
    # 3 stages → 2 transition lines (between 0-1 and 1-2).
    # ax.get_lines() includes data lines too; instead, check ax.collections /
    # ax.axvline instances. matplotlib stores axvline as Line2D with x = const.
    n_vlines = 0
    for line in ax.get_lines():
        if line.get_xdata() is None:
            continue
        xs = np.asarray(line.get_xdata()).tolist()
        n_vlines += len(xs) == 2 and xs[0] == xs[1]
    assert n_vlines == 2


def test_yaxis_inverted_when_invert_y_true(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        invert_y=True,
    )
    ax = fig.axes[0]
    ymin, ymax = ax.get_ylim()
    assert ymin > ymax  # inverted


def test_endpoint_annotations_match_format_string(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
        fmt=".1%",
    )
    ax = fig.axes[0]
    annotation_texts = [child.get_text() for child in ax.texts]
    # Three values: 0.10, 0.05, 0.02 → "10.0%", "5.0%", "2.0%"
    assert "10.0%" in annotation_texts
    assert "5.0%" in annotation_texts
    assert "2.0%" in annotation_texts


def test_line_colors_match_signal_styles(harmfulness_record_factory) -> None:
    fig = render_progression_chart(
        _three_records(harmfulness_record_factory),
        source_to_stage=DEFAULT_COND_SOURCE_MAP,
        stage_labels=STAGES,
        signal_order=DEFAULT_SIGNAL_ORDER,
        signal_styles=DEFAULT_SIGNAL_STYLES,
        metric=HarmfulnessMetric.ASR,
        ylabel="ASR",
        title="Test",
    )
    ax = fig.axes[0]
    line = ax.lines[0]
    expected_color = DEFAULT_SIGNAL_STYLES["Safety delta-base"].color
    # Compare via to_rgba for normalized representation. The left-hand exact
    # string match was redundant — `to_rgba(c) == to_rgba(c)` is true for any
    # color expressible both ways, so the disjunction was a dead branch.
    assert to_rgba(line.get_color()) == to_rgba(expected_color), (
        f"line color {line.get_color()!r} != expected {expected_color!r}"
    )
