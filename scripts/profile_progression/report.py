"""Markdown formatters for progressive-eval profiling summary artifacts."""

from __future__ import annotations

import csv
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from steering.profiling.statistics import bootstrap_ci


@dataclass
class _SuiteTiming:
    warmup: float | None = None
    timed: list[float] = field(default_factory=list)


def _load_macro_rows(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _row_variant(row: dict[str, str]) -> str:
    return row.get("variant") or row.get("eval_variant") or ""


def _row_config(row: dict[str, str]) -> str:
    return row.get("conditioned_config") or ""


def _render_suite_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[tuple[str, str, str, str], _SuiteTiming] = defaultdict(_SuiteTiming)
    for row in rows:
        if row.get("stage") != "suite_total":
            continue
        key = (
            row.get("path", ""),
            row.get("suite", ""),
            _row_variant(row),
            _row_config(row),
        )
        secs = float(row["wall_seconds"])
        if row.get("warmup", "").lower() == "true":
            grouped[key].warmup = secs
        else:
            grouped[key].timed.append(secs)

    lines = [
        "| Path | Suite | Variant | Conditioned Config | Median (s) | 95% CI (s) | N iters | Warmup (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for key in sorted(grouped):
        path, suite, variant, config = key
        info = grouped[key]
        timed = info.timed
        if timed:
            lo, hi, stat = bootstrap_ci(timed)
        else:
            lo = hi = stat = float("nan")
        warmup = info.warmup
        warmup_str = f"{warmup:.1f}" if isinstance(warmup, float) else "-"
        lines.append(
            f"| {path} | {suite} | {variant or '-'} | {config or '-'} | "
            f"{stat:.1f} | [{lo:.1f}, {hi:.1f}] | {len(timed)} | {warmup_str} |"
        )
    return "\n".join(lines)


def _render_stage_table(rows: list[dict[str, str]]) -> str:
    stage_rows = [
        row
        for row in rows
        if row.get("stage") not in {"suite_total", "run_total"}
    ]
    lines = [
        "## Stage Rows",
        "",
        "| Path | Suite | Variant | Conditioned Config | Stage | Iter | Warmup | Wall (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in stage_rows:
        lines.append(
            f"| {row.get('path', '')} | {row.get('suite', '') or '-'} | "
            f"{_row_variant(row) or '-'} | {_row_config(row) or '-'} | "
            f"{row.get('stage', '')} | {row.get('iter', '')} | "
            f"{row.get('warmup', '')} | {float(row['wall_seconds']):.2f} |"
        )
    return "\n".join(lines)


def render_macro_markdown(csv_path: Path) -> str:
    rows = _load_macro_rows(csv_path)
    sections = [
        "# Macro wall-time summary",
        "",
        "## Suite Passes",
        "",
        _render_suite_table(rows),
    ]
    if any(row.get("stage") not in {"suite_total", "run_total"} for row in rows):
        sections.extend(["", _render_stage_table(rows)])
    return "\n".join(sections).rstrip() + "\n"
