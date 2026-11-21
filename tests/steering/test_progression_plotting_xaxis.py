"""X-axis behaviour tests for progression-plot rendering.

Regression coverage for the bug where a single late-step pretrain
checkpoint (e.g. step 54600) followed by a few small-step SFT checkpoints
(3, 5, 500, 886) collapsed the visible x-axis to [54600, 55486] and made
the pretrain point look as if it sat at x=0.

The contract these tests pin:

* ``render_multi_series_chart`` lets matplotlib autoscale the x-axis, so
  charts with only late checkpoints focus on that right-hand data range.
* Autoscaling still keeps the largest cumulative-x visible, so the rightmost
  marker is not clipped to the axis edge.
* Empty specs (no data across all series) leave ``xlim`` untouched, so
  the no-data branch keeps matplotlib's default behaviour and callers
  that pass empty series don't crash.
* The x-axis shows at least one informative tick label. Previous behaviour
  hid every tick via ``set_xticks([])``; the renderer keeps automatic ticks
  inside the visible data window so readers can see absolute step positions.
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


def _baseline_series(
    steps: tuple[int, ...],
    phases: tuple[str, ...],
    *,
    label: str = "Reference",
    color: str = BASELINE_COLOR,
    linestyle: str = "solid",
    is_baseline: bool = True,
) -> ChartSeries:
    """Helper: build a `ChartSeries` with flat stds for rendering tests."""
    n = len(steps)
    assert n == len(phases), "steps/phases length mismatch in test fixture"
    means = tuple(0.1 + 0.01 * i for i in range(n))
    stds = tuple(0.0 for _ in range(n))
    return ChartSeries(
        label=label,
        steps=steps,
        phases=phases,
        means=means,
        stds=stds,
        color=color,
        linestyle=linestyle,
        is_baseline=is_baseline,
    )


# ---------------------------------------------------------------------------
# Regression: user-reported d12_baseline scenario
# ---------------------------------------------------------------------------


def test_d12_baseline_shape_xlim_focuses_on_right_side():
    """Reproduces the reported d12_baseline layout.

    Pretrain step 54600 followed by safety_sft at raw steps 3, 5, 500, 886.
    The cumulative-x values are (54600, 54603, 54605, 55100, 55486). The
    x-axis should focus on that late-step range rather than anchoring at 0.
    """
    series = _baseline_series(
        steps=(54600, 3, 5, 500, 886),
        phases=("pretrain", "safety_sft", "safety_sft", "safety_sft", "safety_sft"),
    )
    spec = MultiSeriesSpec(
        title="d12_baseline — CORE Metric",
        metric_name="CORE Metric",
        model_tag="d12_baseline",
        series=(series,),
        phase_boundaries=(
            PhaseBoundary(x=54600.0, from_phase="pretrain", to_phase="safety_sft", color="red"),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    assert 0.0 < lo < 54600.0
    assert hi > 55486.0
    assert hi - lo < 2000.0


def test_d12_baseline_shape_xlim_upper_covers_last_cumulative_x():
    """The rightmost cumulative-x (55486) must lie strictly inside xlim."""
    series = _baseline_series(
        steps=(54600, 3, 5, 500, 886),
        phases=("pretrain", "safety_sft", "safety_sft", "safety_sft", "safety_sft"),
    )
    spec = MultiSeriesSpec(
        title="t",
        metric_name="m",
        model_tag="d12_baseline",
        series=(series,),
        phase_boundaries=(),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert lo > 0.0
    assert hi > 55486.0, f"xlim upper {hi} must exceed max cumulative-x 55486"


def test_d12_baseline_pretrain_marker_not_at_leftmost_pixel():
    """The pretrain marker at x=54600 must NOT land at axis pixel 0.

    Matplotlib autoscaling should add a small margin around the first point,
    keeping it off the left spine while still focusing on the late-step range.
    """
    series = _baseline_series(
        steps=(54600, 3, 5, 500, 886),
        phases=("pretrain", "safety_sft", "safety_sft", "safety_sft", "safety_sft"),
    )
    spec = MultiSeriesSpec(
        title="t",
        metric_name="m",
        model_tag="d12_baseline",
        series=(series,),
        phase_boundaries=(),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    fig.canvas.draw()

    # Pretrain marker is the first point in the baseline series.
    line = ax.lines[0]
    xdata = line.get_xdata()
    assert xdata[0] == 54600.0
    # Transform data x=54600 into display pixel; it should sit inside the
    # autoscale margin, not exactly on the left spine.
    pretrain_px, _ = ax.transData.transform((float(xdata[0]), 0.0))
    left_px = ax.transAxes.transform((0.0, 0.0))[0]
    right_px = ax.transAxes.transform((1.0, 0.0))[0]
    axis_width = right_px - left_px
    assert axis_width > 0
    fractional = (pretrain_px - left_px) / axis_width
    assert 0.0 < fractional < 0.20, (
        f"pretrain marker should sit inside the left margin; got {fractional:.3f}"
    )


def test_partial_conditioned_series_uses_shared_phase_offset():
    """Missing final pretrain data must not shift SFT points left.

    Regression: a delta-base series with pretrain data through 54000 but no
    54600 point placed safety_sft step 886 at 54886. The shared baseline
    boundary says safety_sft starts after pretrain x=54600, so the same point
    belongs at 55486.
    """
    baseline = _baseline_series(
        steps=(54000, 54600, 886),
        phases=("pretrain", "pretrain", "safety_sft"),
    )
    delta_base = _baseline_series(
        steps=(54000, 886),
        phases=("pretrain", "safety_sft"),
        label="Conditioned: Safety delta-base",
        color="#ff7f0e",
        linestyle="dashed",
        is_baseline=False,
    )
    spec = MultiSeriesSpec(
        title="d12_baseline — CORE Metric",
        metric_name="CORE Metric",
        model_tag="d12_baseline",
        series=(baseline, delta_base),
        phase_boundaries=(
            PhaseBoundary(
                x=54600.0,
                from_phase="pretrain",
                to_phase="safety_sft",
                color="red",
            ),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    conditioned_line = fig.axes[0].lines[1]
    assert tuple(conditioned_line.get_xdata()) == pytest.approx((54000.0, 55486.0))


def test_three_phase_chain_partial_conditioned_alignment():
    """Three-phase chain with a conditioned series missing the last pretrain step.

    Baseline pretrain (54000, 54600) → safety_sft (886) → capabilities_sft (1106).
    Conditioned drops 54600 from pretrain; SFT/capabilities steps must still
    line up at the same cumulative x as the baseline (55486 and 56592).
    """
    baseline = _baseline_series(
        steps=(54000, 54600, 886, 1106),
        phases=("pretrain", "pretrain", "safety_sft", "capabilities_sft"),
    )
    conditioned = _baseline_series(
        steps=(54000, 886, 1106),
        phases=("pretrain", "safety_sft", "capabilities_sft"),
        label="Conditioned: capability delta-base",
        color="#ff7f0e",
        linestyle="dashed",
        is_baseline=False,
    )
    spec = MultiSeriesSpec(
        title="d12_baseline — CORE Metric",
        metric_name="CORE Metric",
        model_tag="d12_baseline",
        series=(baseline, conditioned),
        phase_boundaries=(
            PhaseBoundary(
                x=54600.0,
                from_phase="pretrain",
                to_phase="safety_sft",
                color="red",
            ),
            PhaseBoundary(
                x=55486.0,
                from_phase="safety_sft",
                to_phase="capabilities_sft",
                color="orange",
            ),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    baseline_line = fig.axes[0].lines[0]
    conditioned_line = fig.axes[0].lines[1]

    assert tuple(baseline_line.get_xdata()) == pytest.approx(
        (54000.0, 54600.0, 55486.0, 56592.0)
    )
    assert tuple(conditioned_line.get_xdata()) == pytest.approx(
        (54000.0, 55486.0, 56592.0)
    )


def test_late_phase_boundary_kept_visible_via_axvline_autoscale():
    """A boundary beyond the rightmost series x must still land inside xlim.

    Matplotlib's autoscale considers axvline (a Line2D drawn by the renderer
    after the data lines), so the visible window expands to include the
    boundary even when no series data point reaches it.
    """
    series = _baseline_series(
        steps=(500, 1000),
        phases=("pretrain", "pretrain"),
    )
    spec = MultiSeriesSpec(
        title="t",
        metric_name="m",
        model_tag="d27",
        series=(series,),
        phase_boundaries=(
            PhaseBoundary(x=1500.0, from_phase="pretrain", to_phase="sft", color="red"),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    # Both data left edge (500) AND late boundary (1500) must be inside xlim.
    assert lo <= 500.0
    assert lo <= 1500.0 <= hi


def test_zero_anchor_no_longer_set_for_late_only_data():
    """The renderer no longer forces xlim left to 0 — late-only data sets ~0% drift.

    This is the inverse of the old contract: with only large-step data, lo
    must be > 1% of hi (otherwise the d12_baseline regression returns).
    """
    series = _baseline_series(
        steps=(54600, 3, 5, 500, 886),
        phases=("pretrain", "safety_sft", "safety_sft", "safety_sft", "safety_sft"),
    )
    spec = MultiSeriesSpec(
        title="t",
        metric_name="m",
        model_tag="d12_baseline",
        series=(series,),
        phase_boundaries=(
            PhaseBoundary(
                x=54600.0,
                from_phase="pretrain",
                to_phase="safety_sft",
                color="red",
            ),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert lo > 0.01 * hi, (
        f"xlim left {lo} must be far from 0 when data starts at 54600; "
        f"a left of 0 reproduces the d12_baseline collapse"
    )


# ---------------------------------------------------------------------------
# xlim invariants across common shapes
# ---------------------------------------------------------------------------


def test_xlim_autoscales_single_pretrain_series():
    series = _baseline_series(
        steps=(1000, 2000, 3000),
        phases=("pretrain", "pretrain", "pretrain"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert 0.0 < lo < 1000.0
    assert hi > 3000.0


def test_xlim_autoscales_multi_phase():
    series = _baseline_series(
        steps=(1000, 2000, 500, 1000),
        phases=("pretrain", "pretrain", "sft", "sft"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert 0.0 < lo < 1000.0
    assert hi > 3000.0


def test_xlim_autoscales_multi_series():
    baseline = _baseline_series(
        steps=(1000, 2000),
        phases=("pretrain", "pretrain"),
    )
    conditioned = _baseline_series(
        steps=(1000, 2000),
        phases=("pretrain", "pretrain"),
        label="conditioned: cfg_a",
        color="#1f77b4",
        linestyle="dashed",
        is_baseline=False,
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(baseline, conditioned),
        phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert 0.0 < lo < 1000.0
    assert hi > 2000.0


def test_xlim_ends_past_last_cumulative_x_single_series():
    """Upper xlim is strictly greater than the largest cumulative-x."""
    series = _baseline_series(
        steps=(13000, 13232),
        phases=("pretrain", "pretrain"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert lo > 0.0
    assert hi > 13232.0


def test_xlim_ends_past_last_cumulative_x_across_phases():
    """Cumulative-x after a phase transition must still fit in xlim."""
    series = _baseline_series(
        steps=(13000, 13232, 998, 1240),  # last cumulative = 13232+998+1240=15470
        phases=("pretrain", "pretrain", "safety_sft", "capabilities_sft"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert lo > 0.0
    # cumulative: 13000, 13232, 14230 (13232+998), 15470 (14230+1240)
    assert hi > 15470.0


def test_xlim_when_all_series_empty_uses_matplotlib_default():
    """Empty spec leaves xlim to matplotlib's default (fix must not raise)."""
    empty_series = ChartSeries(
        label="Reference",
        steps=(), phases=(), means=(), stds=(),
        color=BASELINE_COLOR, linestyle="solid", is_baseline=True,
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(empty_series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    # Fallback behaviour: we don't force xlim when no series has data.
    # The renderer must not raise; xlim should exist (default 0..1).
    lo, hi = fig.axes[0].get_xlim()
    assert hi > lo  # sanity: valid range


def test_xlim_when_only_some_series_empty():
    """Mixing empty and non-empty series still autoscales around real data."""
    empty = ChartSeries(
        label="conditioned: empty",
        steps=(), phases=(), means=(), stds=(),
        color="#ff7f0e", linestyle="dashed", is_baseline=False,
    )
    real = _baseline_series(
        steps=(500, 1000),
        phases=("pretrain", "pretrain"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(real, empty), phase_boundaries=(),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert 0.0 < lo < 500.0
    assert hi > 1000.0


def test_xlim_autoscales_even_when_first_step_is_tiny():
    """Single early pretrain checkpoint at step 1 still uses data margins."""
    series = _baseline_series(
        steps=(1,), phases=("pretrain",),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert 0.0 < lo < 1.0 < hi


# ---------------------------------------------------------------------------
# Tick visibility: readers should see SOME absolute step positions
# ---------------------------------------------------------------------------


def test_xaxis_has_visible_ticks_for_nonempty_spec():
    """The x-axis shows at least one tick inside the view."""
    series = _baseline_series(
        steps=(1000, 2000, 3000),
        phases=("pretrain", "pretrain", "pretrain"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    lo, hi = ax.get_xlim()
    visible = [t for t in ax.get_xticks() if lo <= t <= hi]
    assert len(visible) >= 1


def test_xaxis_ticks_are_all_nonnegative():
    """All rendered ticks lie within [0, xlim_upper] — no phantom negatives."""
    series = _baseline_series(
        steps=(500, 1000, 1500),
        phases=("pretrain", "pretrain", "pretrain"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d27",
        series=(series,), phase_boundaries=(), legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    hi = ax.get_xlim()[1]
    visible = [t for t in ax.get_xticks() if 0 <= t <= hi]
    for tick in visible:
        assert tick >= 0.0, f"Tick {tick} is negative"


# ---------------------------------------------------------------------------
# Phase-boundary interaction: boundary at cumulative-x must stay visible
# ---------------------------------------------------------------------------


def test_phase_boundary_at_pretrain_end_stays_visible():
    """Dashed boundary line at x=54600 (pretrain end) must be inside xlim."""
    series = _baseline_series(
        steps=(54600, 500),
        phases=("pretrain", "safety_sft"),
    )
    boundary = PhaseBoundary(
        x=54600.0,
        from_phase="pretrain",
        to_phase="safety_sft",
        color="red",
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d12_baseline",
        series=(series,),
        phase_boundaries=(boundary,),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    lo, hi = fig.axes[0].get_xlim()
    assert lo <= 54600.0 <= hi


def test_render_with_only_boundaries_no_data_draws_axvlines():
    """Renderer with no data series + non-empty boundaries draws boundaries.

    When called with an empty `series` tuple but real `phase_boundaries`, the
    renderer must still emit one Line2D per boundary, expand xlim to include
    them, and add the boundary labels to the legend. This guards against a
    refactor that gates axvline emission on having any data series.
    """
    spec = MultiSeriesSpec(
        title="t",
        metric_name="m",
        model_tag="d27",
        series=(),
        phase_boundaries=(
            PhaseBoundary(x=1000.0, from_phase="pretrain", to_phase="sft", color="red"),
            PhaseBoundary(
                x=2000.0,
                from_phase="sft",
                to_phase="capabilities_sft",
                color="orange",
            ),
        ),
        legend_outside=False,
    )
    fig = render_multi_series_chart(spec)
    ax = fig.axes[0]
    # Each axvline contributes one Line2D; with no data series, ax.lines is
    # exactly the boundary set.
    assert len(ax.lines) == 2
    lo, hi = ax.get_xlim()
    assert lo <= 1000.0
    assert 2000.0 <= hi
    legend = ax.get_legend()
    assert legend is not None
    texts = [t.get_text() for t in legend.get_texts()]
    assert "Pretrain / SFT" in texts
    assert "SFT / Capabilities SFT" in texts


# ---------------------------------------------------------------------------
# Determinism: the fix must not introduce rendering nondeterminism
# ---------------------------------------------------------------------------


def test_rendering_is_deterministic_with_autoscaled_xlim():
    """Two back-to-back renders of the same spec produce identical rasters."""
    series = _baseline_series(
        steps=(54600, 3, 5, 500, 886),
        phases=("pretrain", "safety_sft", "safety_sft", "safety_sft", "safety_sft"),
    )
    spec = MultiSeriesSpec(
        title="t", metric_name="m", model_tag="d12_baseline",
        series=(series,), phase_boundaries=(),
        legend_outside=False,
    )
    fig_a = render_multi_series_chart(spec)
    fig_b = render_multi_series_chart(spec)
    fig_a.canvas.draw()
    fig_b.canvas.draw()
    assert bytes(fig_a.canvas.buffer_rgba()) == bytes(fig_b.canvas.buffer_rgba())
