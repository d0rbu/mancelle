"""Pure-data + matplotlib rendering for training-progression multi-series charts.

This module is intentionally split into two layers:

- Extraction (``build_multi_series_spec``) walks a ``results.json``-shaped dict
  and produces a ``MultiSeriesSpec`` — a frozen dataclass of primitive types.
  No matplotlib imports are required to exercise this layer.
- Rendering (``render_multi_series_chart``, added in a follow-up task) turns
  a ``MultiSeriesSpec`` into a ``matplotlib.figure.Figure``. No business
  logic — just drawing.

The split keeps the interesting decisions (color assignment, legend placement,
phase-boundary anchoring) in a pure-Python module that unit tests can cover
exhaustively without needing to render images.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from loguru import logger


@dataclass(frozen=True)
class ChartSeries:
    """One progression line on the chart."""

    label: str
    steps: tuple[int, ...]
    phases: tuple[str, ...]
    means: tuple[float, ...]
    stds: tuple[float, ...]
    color: str
    linestyle: str
    is_baseline: bool


@dataclass(frozen=True)
class PhaseBoundary:
    """A dashed vertical line marking a phase transition."""

    x: float
    from_phase: str
    to_phase: str
    color: str


@dataclass(frozen=True)
class MultiSeriesSpec:
    """Everything a renderer needs to draw one progression chart."""

    title: str
    metric_name: str
    model_tag: str
    series: tuple[ChartSeries, ...]
    phase_boundaries: tuple[PhaseBoundary, ...]
    legend_outside: bool


BASELINE_COLOR = "steelblue"
BASELINE_LINESTYLE = "solid"
CONDITIONED_LINESTYLE = "dashed"


def _extract_seeds_for_path(
    seeds_data: dict[str, Any],
    tail_keys: list[str],
) -> list[float]:
    """Pull the metric value from each seed dict, skipping NaN/Inf."""
    values: list[float] = []
    for seed_result in seeds_data.values():
        val: Any = seed_result
        for key in tail_keys:
            if isinstance(val, dict):
                val = val.get(key)
            else:
                val = None
                break
        if val is None or not isinstance(val, (int, float)):
            continue
        fval = float(val)
        if not math.isfinite(fval):
            continue
        values.append(fval)
    return values


def _series_for_conditioned_config(
    checkpoints: list[dict],
    eval_name: str,
    tail_keys: list[str],
    config_name: str,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    steps: list[int] = []
    phases: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for ckpt in checkpoints:
        eval_data = ckpt.get("evals", {}).get(eval_name, {})
        seeds_data = (
            eval_data.get("conditioned", {})
            .get("configs", {})
            .get(config_name, {})
            .get("seeds", {})
        )
        if not seeds_data:
            continue
        values = _extract_seeds_for_path(seeds_data, tail_keys)
        if not values:
            continue
        steps.append(int(ckpt["step"]))
        phases.append(str(ckpt["phase"]))
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)) if len(values) > 1 else 0.0)

    return tuple(steps), tuple(phases), tuple(means), tuple(stds)


def _series_for_target_real(
    checkpoints: list[dict],
    eval_name: str,
    tail_keys: list[str],
    trc_name: str,
) -> tuple[tuple[int, ...], tuple[str, ...], tuple[float, ...], tuple[float, ...]]:
    """Extract steps/means/stds from evals[eval_name]['target_real']['configs'][trc_name]."""
    steps: list[int] = []
    phases: list[str] = []
    means: list[float] = []
    stds: list[float] = []

    for ckpt in checkpoints:
        seeds_data = (
            ckpt.get("evals", {})
            .get(eval_name, {})
            .get("target_real", {})
            .get("configs", {})
            .get(trc_name, {})
            .get("seeds", {})
        )
        if not seeds_data:
            continue
        values = _extract_seeds_for_path(seeds_data, tail_keys)
        if not values:
            continue
        steps.append(int(ckpt["step"]))
        phases.append(str(ckpt["phase"]))
        means.append(float(np.mean(values)))
        stds.append(float(np.std(values)) if len(values) > 1 else 0.0)

    return tuple(steps), tuple(phases), tuple(means), tuple(stds)


TAB10_COLORS: tuple[str, ...] = (
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
)
TAB20_COLORS: tuple[str, ...] = (
    "#1f77b4", "#aec7e8", "#ff7f0e", "#ffbb78", "#2ca02c",
    "#98df8a", "#d62728", "#ff9896", "#9467bd", "#c5b0d5",
    "#8c564b", "#c49c94", "#e377c2", "#f7b6d2", "#7f7f7f",
    "#c7c7c7", "#bcbd22", "#dbdb8d", "#17becf", "#9edae5",
)
_BOUNDARY_COLORS: tuple[str, ...] = ("red", "orange", "purple", "green")


def _conditioned_config_names(
    checkpoints: list[dict], eval_name: str
) -> list[str]:
    names: set[str] = set()
    for ckpt in checkpoints:
        names.update(
            ckpt.get("evals", {})
            .get(eval_name, {})
            .get("conditioned", {})
            .get("configs", {})
            .keys()
        )
    return sorted(names)


def _target_real_config_names(
    checkpoints: list[dict], eval_name: str
) -> list[tuple[str, dict[str, Any]]]:
    """Walk evals[eval_name]['target_real']['configs'][...] across checkpoints.

    Returns a stable sorted list of ``(name, first_config_entry_dict)`` tuples.
    The entry dict carries ``kind`` and other metadata so that
    ``_get_linestyle_for_variant`` / ``linestyle_resolver`` can dispatch
    without a separate lookup.
    """
    names_seen: dict[str, dict[str, Any]] = {}
    for ckpt in checkpoints:
        configs = (
            ckpt.get("evals", {})
            .get(eval_name, {})
            .get("target_real", {})
            .get("configs", {})
        )
        for name, entry in configs.items():
            if name not in names_seen and entry:
                names_seen[name] = dict(entry)
    # Stable sort by name; preserve first-seen entry for metadata.
    return [(name, names_seen[name]) for name in sorted(names_seen)]


def _conditioned_display_name(
    checkpoints: list[dict], eval_name: str, config_name: str
) -> str:
    """Resolve the human-readable legend name for a conditioned config.

    Walks every checkpoint looking for a `display_name` on the matching
    config entry; falls back to the sanitized `config_name` (backwards
    compatible with results.json files written before this field existed).
    """
    for ckpt in checkpoints:
        entry = (
            ckpt.get("evals", {})
            .get(eval_name, {})
            .get("conditioned", {})
            .get("configs", {})
            .get(config_name, {})
        )
        display = entry.get("display_name")
        if isinstance(display, str) and display:
            return display
    return config_name


def _first_config_entry(
    checkpoints: list[dict], eval_name: str, config_name: str
) -> dict[str, Any]:
    """Return the first non-empty config entry dict for a conditioned config.

    Used to extract kind/cache_dir/etc. for variant-aware linestyle dispatch.
    Falls back to an empty dict when no entry is found (caller treats as legacy).
    """
    for ckpt in checkpoints:
        entry = (
            ckpt.get("evals", {})
            .get(eval_name, {})
            .get("conditioned", {})
            .get("configs", {})
            .get(config_name, {})
        )
        if entry:
            return dict(entry)
    return {}


def _assign_conditioned_color(index: int, n_configs: int) -> str:
    if n_configs <= len(TAB10_COLORS):
        palette = TAB10_COLORS
    else:
        palette = TAB20_COLORS
        if n_configs > len(TAB20_COLORS) and index == 0:
            logger.warning(
                "Progression plot has {n} conditioned configs; palette cycling "
                "after the 20th (tab20 exhausted). Consider consolidating configs "
                "or grouping by prefix.",
                n=n_configs,
            )
    return palette[index % len(palette)]


def _cumulative_x(
    steps: tuple[int, ...],
    phases: tuple[str, ...],
    phase_offsets: dict[str, float] | None = None,
) -> tuple[float, ...]:
    """Build a proportional x-axis — same policy as scripts.train.training_progression."""
    phase_offsets = phase_offsets or {}
    x: list[float] = []
    phase_offset = 0.0
    prev_phase: str | None = None
    prev_last_x = 0.0
    for step, phase in zip(steps, phases, strict=True):
        if phase in phase_offsets:
            cur_x = phase_offsets[phase] + float(step)
        else:
            if phase != prev_phase and prev_phase is not None:
                phase_offset = prev_last_x
            cur_x = phase_offset + float(step)
        x.append(cur_x)
        prev_phase = phase
        prev_last_x = cur_x
    return tuple(x)


def _reference_series_for_boundaries(
    series: list[ChartSeries],
) -> ChartSeries | None:
    if not series:
        return None
    # Prefer the baseline when present; otherwise the conditioned series with
    # the most checkpoints, tie-broken on lexicographically smallest label.
    baseline = next((s for s in series if s.is_baseline), None)
    if baseline is not None:
        return baseline
    # When series is non-empty and baseline is None, at least one element must
    # be non-baseline, so the ``conditioned`` list is guaranteed non-empty.
    conditioned = [s for s in series if not s.is_baseline]
    ranked = sorted(conditioned, key=lambda s: (-len(s.steps), s.label))
    return ranked[0]


def _phase_boundaries_from_series(
    reference: ChartSeries,
) -> tuple[PhaseBoundary, ...]:
    steps = reference.steps
    phases = reference.phases
    if len(steps) < 2:
        return ()
    x = _cumulative_x(steps, phases)
    boundaries: list[PhaseBoundary] = []
    for i in range(1, len(phases)):
        if phases[i] != phases[i - 1]:
            color = _BOUNDARY_COLORS[len(boundaries) % len(_BOUNDARY_COLORS)]
            boundaries.append(
                PhaseBoundary(
                    x=x[i - 1],
                    from_phase=phases[i - 1],
                    to_phase=phases[i],
                    color=color,
                )
            )
    return tuple(boundaries)


def _phase_offsets_from_boundaries(
    boundaries: tuple[PhaseBoundary, ...],
) -> dict[str, float]:
    """Map each post-transition phase to its shared cumulative x offset."""
    return {boundary.to_phase: float(boundary.x) for boundary in boundaries}


def build_multi_series_spec(
    checkpoints: list[dict],
    metric_path: list[str],
    display_name: str,
    model_tag: str,
    phase_order: dict[str, int],
    linestyle_resolver: Callable[[str, dict[str, Any]], tuple[str, str]] | None = None,
) -> MultiSeriesSpec | None:
    """Extract a MultiSeriesSpec for one metric from a results-JSON checkpoint list.

    Produces one series per distinct conditioned config and target-real config,
    sorted alphabetically by name. Conditioned series get tab10 colors (tab20
    when >10 configs; cycling with a loguru warning when >20).

    Phase boundaries anchor to the longest series (lexicographic label
    tie-break). The legend is configured to render outside the axes when there
    are more than two conditioned/target-real series.

    Returns ``None`` when no series yields any data for the requested metric —
    callers typically log and skip the chart in that case.

    Args:
        linestyle_resolver: optional callable ``(config_name, config_entry_dict)
            -> (linestyle, color)`` for conditioned series. When ``None``, the
            legacy tab10/tab20 color palette and "dashed" linestyle are used.
            PR-D callers pass ``_variant_linestyle_resolver`` from
            ``scripts.train.training_progression`` which dispatches through
            ``_get_linestyle_for_variant``.
    """
    eval_name = metric_path[0]
    tail_keys = list(metric_path[1:])

    def _sort_key(ckpt: dict) -> tuple[int, int]:
        return (phase_order.get(ckpt.get("phase", ""), 10_000), int(ckpt.get("step", 0)))

    checkpoints = sorted(checkpoints, key=_sort_key)

    series: list[ChartSeries] = []

    config_names = _conditioned_config_names(checkpoints, eval_name)
    for idx, name in enumerate(config_names):
        c_steps, c_phases, c_means, c_stds = _series_for_conditioned_config(
            checkpoints,
            eval_name,
            tail_keys,
            config_name=name,
        )
        if not c_steps:
            continue
        display = _conditioned_display_name(checkpoints, eval_name, name)
        if linestyle_resolver is not None:
            config_entry = _first_config_entry(checkpoints, eval_name, name)
            c_linestyle, c_color = linestyle_resolver(name, config_entry)
        else:
            c_linestyle = CONDITIONED_LINESTYLE
            c_color = _assign_conditioned_color(idx, len(config_names))
        series.append(
            ChartSeries(
                label=f"Conditioned: {display}",
                steps=c_steps,
                phases=c_phases,
                means=c_means,
                stds=c_stds,
                color=c_color,
                linestyle=c_linestyle,
                is_baseline=False,
            )
        )

    tr_config_names = _target_real_config_names(checkpoints, eval_name)
    for trc_name, trc_entry in tr_config_names:
        tr_steps, tr_phases, tr_means, tr_stds = _series_for_target_real(
            checkpoints, eval_name, tail_keys, trc_name
        )
        if not tr_steps:
            continue
        if linestyle_resolver is not None:
            tr_linestyle, tr_color = linestyle_resolver(trc_name, trc_entry)
        else:
            tr_linestyle = CONDITIONED_LINESTYLE
            tr_color = _assign_conditioned_color(
                len(config_names) + [n for n, _ in tr_config_names].index(trc_name),
                len(config_names) + len(tr_config_names),
            )
        series.append(
            ChartSeries(
                label=f"TargetReal: {trc_name}",
                steps=tr_steps,
                phases=tr_phases,
                means=tr_means,
                stds=tr_stds,
                color=tr_color,
                linestyle=tr_linestyle,
                is_baseline=False,
            )
        )

    if not series:
        return None

    reference = _reference_series_for_boundaries(series)
    boundaries = (
        _phase_boundaries_from_series(reference) if reference is not None else ()
    )

    n_conditioned = sum(1 for s in series if not s.is_baseline)
    return MultiSeriesSpec(
        title=f"{model_tag} — {display_name}",
        metric_name=display_name,
        model_tag=model_tag,
        series=tuple(series),
        phase_boundaries=boundaries,
        legend_outside=n_conditioned > 2,
    )


_PHASE_DISPLAY_NAMES: dict[str, str] = {
    "pretrain": "Pretrain",
    "sft": "SFT",
    "safety_sft": "Safety SFT",
    "capabilities_sft": "Capabilities SFT",
    "standard_sft": "Standard SFT",
}


def _phase_label(phase: str) -> str:
    return _PHASE_DISPLAY_NAMES.get(phase, phase.replace("_", " ").title())


def render_multi_series_chart(spec: MultiSeriesSpec):
    """Render a MultiSeriesSpec to a matplotlib Figure."""
    # matplotlib import is local so pure-data callers don't pay for it;
    # numpy is already imported at module level for the extraction layer.
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(10, 5))
    phase_offsets = _phase_offsets_from_boundaries(spec.phase_boundaries)

    for series in spec.series:
        if not series.steps:
            continue
        x = np.array(
            _cumulative_x(series.steps, series.phases, phase_offsets),
            dtype=float,
        )
        means = np.array(series.means, dtype=float)
        stds = np.array(series.stds, dtype=float)
        # Single-series charts omit a redundant line legend unless phase
        # boundaries add labels of their own.
        effective_label = series.label if len(spec.series) > 1 else None
        ax.plot(
            x,
            means,
            color=series.color,
            linestyle={"solid": "-", "dashed": "--"}.get(series.linestyle, series.linestyle),
            linewidth=1.5,
            marker="o",
            markersize=3,
            label=effective_label,
        )
        if stds.size and stds.max() > 0:
            ax.fill_between(
                x,
                means - stds,
                means + stds,
                color=series.color,
                alpha=0.2,
            )

    seen: set[str] = set()
    for boundary in spec.phase_boundaries:
        label = f"{_phase_label(boundary.from_phase)} / {_phase_label(boundary.to_phase)}"
        if label in seen:
            label = None
        else:
            seen.add(label)
        ax.axvline(
            x=boundary.x,
            color=boundary.color,
            linestyle="--",
            alpha=0.7,
            label=label,
        )

    if ax.get_legend_handles_labels()[0]:
        if spec.legend_outside:
            ax.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                fontsize=8,
            )
        else:
            ax.legend(loc="upper left", fontsize=8)

    ax.set_xlabel("Training Step")
    ax.set_ylabel(spec.metric_name)
    ax.set_title(spec.title)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig
