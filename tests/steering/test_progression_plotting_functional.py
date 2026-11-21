"""Functional tests that render MultiSeriesSpec via matplotlib and introspect Figure.

No PNGs are written; these tests walk Line2D / PolyCollection objects directly.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pytest

from steering.eval.progression_plotting import (
    BASELINE_COLOR,
    ChartSeries,
    MultiSeriesSpec,
    PhaseBoundary,
    render_multi_series_chart,
)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _series(
    label: str,
    color: str,
    linestyle: str,
    is_baseline: bool,
    stds=(0.01, 0.01),
) -> ChartSeries:
    return ChartSeries(
        label=label,
        steps=(500, 1000),
        phases=("pretrain", "pretrain"),
        means=(0.05, 0.10),
        stds=stds,
        color=color,
        linestyle=linestyle,
        is_baseline=is_baseline,
    )


def test_render_returns_figure():
    spec = MultiSeriesSpec(
        title="d27 — CORE Metric",
        metric_name="CORE Metric",
        model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    assert isinstance(fig, plt.Figure)


def test_render_n_lines_matches_series_count():
    spec = MultiSeriesSpec(
        title="t",
        metric_name="CORE Metric",
        model_tag="d27",
        series=(
            _series("Reference", BASELINE_COLOR, "solid", True),
            _series("conditioned: a", "#1f77b4", "dashed", False),
            _series("conditioned: b", "#ff7f0e", "dashed", False),
        ),
        phase_boundaries=(),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    # Count only lines that are series (2-point-or-more, not axvline which has 2 identical x's).
    data_lines = [
        line for line in ax.lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] != line.get_xdata()[1]
    ]
    assert len(data_lines) == 3


def test_render_baseline_is_solid_steelblue():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    baseline_line = fig.axes[0].lines[0]
    assert baseline_line.get_linestyle() == "-"
    assert baseline_line.get_color() == BASELINE_COLOR


def test_render_conditioned_is_dashed():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(_series("conditioned: a", "#1f77b4", "dashed", False),),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    line = fig.axes[0].lines[0]
    assert line.get_linestyle() == "--"


def test_render_std_shading_drawn_only_for_nonzero_series():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(
            _series("Reference", BASELINE_COLOR, "solid", True, stds=(0.01, 0.02)),
            _series("conditioned: a", "#1f77b4", "dashed", False, stds=(0.0, 0.0)),
            _series("conditioned: b", "#ff7f0e", "dashed", False, stds=(0.05, 0.03)),
        ),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    polys = [c for c in fig.axes[0].collections if hasattr(c, "get_paths")]
    # Only the two series with nonzero stds get shading.
    assert len(polys) == 2


def test_render_phase_boundaries_drawn():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(
            PhaseBoundary(x=1000.0, from_phase="pretrain", to_phase="sft", color="red"),
            PhaseBoundary(x=1500.0, from_phase="sft", to_phase="capabilities_sft", color="orange"),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert "Pretrain / SFT" in texts
    assert "SFT / Capabilities SFT" in texts


def test_render_legend_inside_for_small_n():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(
            _series("Reference", BASELINE_COLOR, "solid", True),
            _series("conditioned: a", "#1f77b4", "dashed", False),
        ),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    bbox = legend.get_window_extent()
    ax_bbox = fig.axes[0].get_window_extent()
    # Legend x-center is inside the axes rect.
    assert bbox.x0 >= ax_bbox.x0 - 5
    assert bbox.x1 <= ax_bbox.x1 + 5


def test_render_legend_outside_for_large_n():
    series = [_series("Reference", BASELINE_COLOR, "solid", True)]
    for i in range(5):
        series.append(_series(f"conditioned: c{i}", "#1f77b4", "dashed", False))
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=tuple(series),
        phase_boundaries=(), legend_outside=True,
    )
    fig = render_multi_series_chart(spec)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    bbox = legend.get_window_extent()
    ax_bbox = fig.axes[0].get_window_extent()
    # Legend starts to the right of the axes.
    assert bbox.x0 >= ax_bbox.x1 - 5


def test_render_title_and_labels():
    spec = MultiSeriesSpec(
        title="d27 — CORE Metric", metric_name="CORE Metric", model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    assert ax.get_title() == "d27 — CORE Metric"
    assert ax.get_xlabel() == "Training Step"
    assert ax.get_ylabel() == "CORE Metric"


def test_render_xticks_visible_with_auto_xaxis():
    """Ticks are shown while matplotlib focuses on the plotted data range.

    Prior behaviour hid every tick (``set_xticks([])``), which masked the bug
    where matplotlib auto-scaled to ``[first_x, last_x]`` and pushed a late
    pretrain checkpoint to the left edge.
    """
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    assert 0.0 < lo < 500.0
    assert hi > 1000.0
    visible = [t for t in ax.get_xticks() if lo <= t <= hi]
    assert len(visible) >= 1


def test_render_single_series_no_legend_unless_boundaries():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    assert fig.axes[0].get_legend() is None


def test_render_single_series_legend_only_has_boundary_labels():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(_series("Reference", BASELINE_COLOR, "solid", True),),
        phase_boundaries=(
            PhaseBoundary(x=100.0, from_phase="pretrain", to_phase="sft", color="red"),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts == ["Pretrain / SFT"]


def test_render_skips_empty_series():
    """A series with no data points adds no line; other series render normally."""
    empty = ChartSeries(
        label="conditioned: empty",
        steps=(),
        phases=(),
        means=(),
        stds=(),
        color="#ff7f0e",
        linestyle="dashed",
        is_baseline=False,
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(
            _series("Reference", BASELINE_COLOR, "solid", True),
            empty,
            _series("conditioned: real", "#1f77b4", "dashed", False),
        ),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    data_lines = [
        line for line in fig.axes[0].lines
        if len(line.get_xdata()) == 2
        and line.get_xdata()[0] != line.get_xdata()[1]
    ]
    # Only the 2 non-empty series contribute Line2D objects with data.
    assert len(data_lines) == 2


def test_render_duplicate_boundary_labels_dedup_in_legend():
    """Two boundaries sharing from/to phase produce only one legend entry."""
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(
            _series("Reference", BASELINE_COLOR, "solid", True),
            _series("conditioned: a", "#1f77b4", "dashed", False),
        ),
        phase_boundaries=(
            PhaseBoundary(x=100.0, from_phase="pretrain", to_phase="sft", color="red"),
            PhaseBoundary(x=200.0, from_phase="pretrain", to_phase="sft", color="orange"),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    legend = fig.axes[0].get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert texts.count("Pretrain / SFT") == 1


def test_render_deterministic_dump():
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(
            _series("Reference", BASELINE_COLOR, "solid", True),
            _series("conditioned: a", "#1f77b4", "dashed", False),
        ),
        phase_boundaries=(), legend_outside=False,
    )
    fig_a = render_multi_series_chart(spec)
    fig_b = render_multi_series_chart(spec)
    # Render both to the same figure manager and compare RGBA buffers.
    fig_a.canvas.draw()
    fig_b.canvas.draw()
    buf_a = bytes(fig_a.canvas.buffer_rgba())
    buf_b = bytes(fig_b.canvas.buffer_rgba())
    assert buf_a == buf_b
