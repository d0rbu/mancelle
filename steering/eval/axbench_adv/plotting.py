"""Plot concept-vs-fluency trajectories over GEPA proposal counts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import pyarrow as pa  # noqa: E402
import pyarrow.parquet as pq  # noqa: E402

from scripts.eval.axbench_adv.pressures import Pressure  # noqa: E402
from steering.eval.axbench_adv.aggregation import (  # noqa: E402
    validate_aggregate,
    validate_counter_aggregate,
    validate_gcg_aggregate,
)
from steering.eval.axbench_adv.errors import AxBenchAdvError  # noqa: E402
from steering.eval.axbench_bpb_adv.aggregation import (  # noqa: E402
    strength_color_for_rank,
)

CONCEPT_AXIS_LABEL = "Concept judge score"
FLUENCY_AXIS_LABEL = "Fluency judge score"
FOOTER = (
    "0 is the regenerated empty-suffix control; 1/2/4/8 are independent "
    "fixed-seed GEPA runs, and larger proposal counts are darker."
)

_METHOD_BASE_COLORS: dict[str, str] = {
    "DiffMean": "#1f77b4",
    "LinearProbe": "#ff7f0e",
    "LoRA": "#2ca02c",
    "PromptSteering": "#9467bd",
    "SelfConditioning": "#d62728",
    "SelfConditioningMasked": "#17becf",
    "NLSteering": "#8c564b",
}


@dataclass(frozen=True)
class TrajectoryPoint:
    """One plotted aggregate point."""

    proposal_count: float
    fluency_x: float
    concept_y: float
    fluency_sem: float
    concept_sem: float
    color: tuple[float, float, float]


def _load_table(aggregate: pa.Table | Path | str) -> pa.Table:
    table = aggregate if isinstance(aggregate, pa.Table) else pq.read_table(Path(aggregate))
    if "token_count" in table.column_names:
        validate_gcg_aggregate(table)
    elif "magnitude" in table.column_names:
        validate_counter_aggregate(table)
    else:
        validate_aggregate(table)
    return table


def method_trajectory(
    aggregate: pa.Table | Path | str,
    method: str,
    *,
    base_color: str | None = None,
) -> list[TrajectoryPoint]:
    """Return a method's control/proposal points ordered light to dark."""

    rows = [row for row in _load_table(aggregate).to_pylist() if row["method"] == method]
    if not rows:
        raise AxBenchAdvError(f"aggregate has no rows for method {method!r}")
    if "token_count" in rows[0]:
        coordinate = "token_count"
    elif "magnitude" in rows[0]:
        coordinate = "magnitude"
    else:
        coordinate = "proposal_count"
    rows.sort(key=lambda row: row[coordinate])
    color = base_color or _METHOD_BASE_COLORS.get(method, "#000000")
    return [
        TrajectoryPoint(
            proposal_count=row[coordinate],
            fluency_x=float(row["fluency_score"]),
            concept_y=float(row["concept_score"]),
            fluency_sem=float(row["fluency_score_sem"]),
            concept_sem=float(row["concept_score_sem"]),
            color=tuple(strength_color_for_rank(color, rank=rank, n_ranks=len(rows))),
        )
        for rank, row in enumerate(rows)
    ]


def _draw_method(
    ax: Any,
    points: Sequence[TrajectoryPoint],
    *,
    method: str,
    base_color: str,
) -> None:
    ax.plot(
        [point.fluency_x for point in points],
        [point.concept_y for point in points],
        color=base_color,
        linewidth=1.0,
        alpha=0.6,
        zorder=1,
        label=method,
    )
    for point in points:
        ax.errorbar(
            point.fluency_x,
            point.concept_y,
            xerr=point.fluency_sem,
            yerr=point.concept_sem,
            fmt="o",
            markersize=7,
            color=point.color,
            markeredgecolor=base_color,
            ecolor=base_color,
            elinewidth=0.6,
            capsize=2,
            zorder=2,
        )


def render_axes(
    ax: Any,
    aggregate: pa.Table | Path | str,
    *,
    methods: Sequence[str] | None = None,
    base_colors: Mapping[str, str] | None = None,
) -> None:
    """Draw all requested method trajectories on an existing axis."""

    table = _load_table(aggregate)
    palette = {**_METHOD_BASE_COLORS, **(dict(base_colors) if base_colors else {})}
    present = list(dict.fromkeys(str(row["method"]) for row in table.to_pylist()))
    ordered = list(methods) if methods is not None else present
    for method in ordered:
        color = palette.get(method, "#000000")
        _draw_method(
            ax,
            method_trajectory(table, method, base_color=color),
            method=method,
            base_color=color,
        )
    ax.set_xlabel(FLUENCY_AXIS_LABEL)
    ax.set_ylabel(CONCEPT_AXIS_LABEL)
    ax.legend(loc="best", fontsize="small")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.5)


def plot_concept_vs_fluency(
    aggregate: pa.Table | Path | str,
    output_dir: Path,
    *,
    base_colors: Mapping[str, str] | None = None,
    pressure: Pressure = Pressure.GEPA,
) -> list[Path]:
    """Write one combined plot and one plot per method."""

    if pressure == Pressure.GEPA:
        footer = FOOTER
    elif pressure in (Pressure.LINEAR_PROBE_COUNTER, Pressure.DIFFMEAN_COUNTER):
        footer = (
            "0 is the shared no-adversary control; darker points subtract larger "
            "multiples of the saved direction."
        )
    elif pressure == Pressure.GCG:
        footer = (
            "0 is the shared no-adversary control; darker points use retained GCG "
            "suffixes with more editable tokens."
        )
    elif pressure == Pressure.SOFT_PROMPT:
        footer = (
            "0 is the shared no-adversary control; darker points use retained soft "
            "suffixes with more embedding positions."
        )
    else:
        raise AxBenchAdvError(f"unsupported plot pressure {pressure!r}")

    table = _load_table(aggregate)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    methods = list(dict.fromkeys(str(row["method"]) for row in table.to_pylist()))
    written: list[Path] = []

    figure, ax = plt.subplots(figsize=(7.0, 6.0))
    try:
        render_axes(ax, table, methods=methods, base_colors=base_colors)
        title = (
            "AxBench-Adv: concept vs fluency under a GEPA suffix"
            if pressure == Pressure.GEPA
            else f"AxBench-Adv: {pressure}"
        )
        ax.set_title(title)
        figure.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=6, wrap=True)
        figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
        combined = output_dir / "concept_vs_fluency.png"
        figure.savefig(combined, dpi=150)
        written.append(combined)
    finally:
        plt.close(figure)

    for method in methods:
        figure, ax = plt.subplots(figsize=(6.0, 5.0))
        try:
            render_axes(ax, table, methods=[method], base_colors=base_colors)
            ax.set_title(f"AxBench-Adv: {method}")
            figure.text(0.5, 0.01, footer, ha="center", va="bottom", fontsize=6, wrap=True)
            figure.tight_layout(rect=(0.0, 0.06, 1.0, 1.0))
            path = output_dir / f"concept_vs_fluency_{method}.png"
            figure.savefig(path, dpi=150)
            written.append(path)
        finally:
            plt.close(figure)
    return written


__all__ = [
    "CONCEPT_AXIS_LABEL",
    "FLUENCY_AXIS_LABEL",
    "FOOTER",
    "TrajectoryPoint",
    "method_trajectory",
    "plot_concept_vs_fluency",
    "render_axes",
]
