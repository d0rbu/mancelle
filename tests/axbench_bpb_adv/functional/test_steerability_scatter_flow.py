"""Functional tests: ``plot_bpb_adv_summaries`` emits the steerability-vs-baseline
scatter PNG ALONGSIDE the line plots, and registers it everywhere the recursive
plot counter looks (spec 2C, Feature 5).

The PR#503/#504 plot-counter undercount lesson: every emitted scatter PNG MUST
appear in the returned path list, in ``plot_index.csv``, and the series legend
``plot_series_legend.csv`` so ``_stage_plot``'s recursive ``rglob('*.png')`` count
includes them.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    _filter_rows_to_representative_coefficient,
    _scatter_points,
    plot_bpb_adv_summaries,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import MACRO_AGG_SCHEMA

_SCATTER_NAME = "steerability_vs_baseline.png"


def _macro_row(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    coefficient_str: str = "1.0",
    adversarial_strength: float = 0.0,
    strength_index: int = 0,
    mean_loss_concept_rich: float = 1.0,
    mean_loss_concept_avoiding: float = 3.0,
    primary_margin_y: float = 2.0,
    baseline_primary_margin: float | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": coefficient_str,
        "adversarial_method": "faster_gcg",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": "fixed_context",
        "mean_loss_concept_rich": mean_loss_concept_rich,
        "mean_loss_concept_avoiding": mean_loss_concept_avoiding,
        "primary_margin_y": primary_margin_y,
        "baseline_primary_margin": (
            primary_margin_y
            if baseline_primary_margin is None and adversarial_strength == 0.0
            else baseline_primary_margin
        ),
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": adversarial_strength == 0.0,
    }


def _coeff_family(
    *,
    method_name: str,
    coefficient_str: str,
    control_margin: float,
    n_strengths: int = 3,
) -> list[dict[str, Any]]:
    rich = 1.0
    avoid = 1.0 + control_margin
    rows = [
        _macro_row(
            method_name=method_name,
            coefficient_str=coefficient_str,
            adversarial_strength=0.0,
            strength_index=0,
            mean_loss_concept_rich=rich,
            mean_loss_concept_avoiding=avoid,
            primary_margin_y=control_margin,
            baseline_primary_margin=control_margin,
        )
    ]
    for strength_index in range(1, n_strengths):
        rows.append(
            _macro_row(
                method_name=method_name,
                coefficient_str=coefficient_str,
                adversarial_strength=float(strength_index),
                strength_index=strength_index,
                mean_loss_concept_rich=rich + float(strength_index),
                mean_loss_concept_avoiding=avoid,
                primary_margin_y=control_margin - float(strength_index),
                baseline_primary_margin=control_margin,
            )
        )
    return rows


def _macro_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=MACRO_AGG_SCHEMA)


def _full_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_name, best in (
        ("diffmean_decoder_only", "4.0"),
        ("linear_probe", "2.0"),
    ):
        for coeff in ("1.0", "2.0", "4.0"):
            rows.extend(
                _coeff_family(
                    method_name=method_name,
                    coefficient_str=coeff,
                    control_margin=3.0 if coeff == best else 0.5,
                )
            )
    return rows


def _feature_diffmean_target_row(
    *,
    mode: str = "positive",
    score: float = 3.0,
) -> dict[str, Any]:
    method_name = "feature_diffmean_encdec" if mode == "positive" else "neg_feature_diffmean_encdec"
    row = _macro_row(
        mode=mode,
        method_name=method_name,
        coefficient_str="None",
        primary_margin_y=score,
        baseline_primary_margin=score,
    )
    row["target_model"] = "encdec"
    return row


def _with_feature_diffmean_targets(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    modes = {str(row["mode"]) for row in rows}
    targets: list[dict[str, Any]] = []
    if "positive" in modes:
        targets.append(_feature_diffmean_target_row())
    if "negative" in modes:
        targets.append(_feature_diffmean_target_row(mode="negative"))
    return [*rows, *targets]


def _plot_index_rows(plots_dir: Path) -> list[dict[str, str]]:
    with (plots_dir / "plot_index.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def _legend_method_names(plots_dir: Path) -> set[str]:
    with (plots_dir / "plot_series_legend.csv").open(encoding="utf-8") as f:
        return {row["method_name"] for row in csv.DictReader(f)}


def test_scatter_png_emitted_and_in_written_list(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    written = plot_bpb_adv_summaries(
        _macro_table(_with_feature_diffmean_targets(_full_rows())),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )

    scatter = plots_dir / "positive" / _SCATTER_NAME
    assert scatter.exists(), "scatter PNG must be written alongside the line plots"
    assert scatter in written, "scatter PNG must be in the returned path list"

    # Line plots are STILL emitted (scatter is additive, not a replacement).
    line = plots_dir / "positive" / "primary_margin_vs_strength.png"
    assert line.exists()
    assert line in written


def test_scatter_png_registered_in_plot_index(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    rows = _full_rows()
    rows_with_target = _with_feature_diffmean_targets(rows)
    plot_bpb_adv_summaries(
        _macro_table(rows_with_target),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )
    index = _plot_index_rows(plots_dir)
    scatter_rows = [r for r in index if r["plot_name"] == _SCATTER_NAME]
    assert scatter_rows, "scatter PNG must be registered in plot_index.csv"
    # The relative path in the index points at the emitted file under alpha/mode.
    rel = scatter_rows[0]["path"]
    assert (plots_dir / rel).exists()

    # n_rows must count the ACTUALLY-plotted points (representative coefficient
    # per family, swept over counter-steering strength), NOT the pre-reduction
    # coefficient*strength grid. _full_rows = 2 methods x 3 coeffs x 3 strengths;
    # reduction collapses to the best coefficient per method -> 2 families x 3
    # strengths = 6 points (pre-reduction would be 18).
    expected = len(
        _scatter_points(
            _filter_rows_to_representative_coefficient(
                rows,
                baseline_rows=rows_with_target,
            ),
            scatter_baseline_x="baseline_loss",
        )
    )
    assert expected == 6
    assert int(scatter_rows[0]["n_rows"]) == expected


def test_scatter_png_in_legend_csv(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    plot_bpb_adv_summaries(
        _macro_table(_with_feature_diffmean_targets(_full_rows())),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )
    legend_path = plots_dir / "plot_series_legend.csv"
    assert legend_path.exists()
    with legend_path.open(encoding="utf-8") as f:
        legend_rows = list(csv.DictReader(f))
    # The scatter PNG is referenced in the legend (every emitted PNG is encoded
    # there so the recursive plot count and the human-readable index agree).
    referenced = {row.get("plot_path", row.get("path", "")) for row in legend_rows}
    assert any(_SCATTER_NAME in value for value in referenced), (
        "scatter PNG must be referenced in plot_series_legend.csv; "
        f"legend referenced: {sorted(referenced)}"
    )


def test_stage_plot_recursive_count_includes_scatter(tmp_path: Path) -> None:
    # Emulate _stage_plot's recursive count: every PNG under plots_dir is counted.
    plots_dir = tmp_path / "plots"
    written = plot_bpb_adv_summaries(
        _macro_table(_with_feature_diffmean_targets(_full_rows())),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )
    on_disk_pngs = sorted(plots_dir.rglob("*.png"))
    written_pngs = sorted(p for p in written if p.suffix == ".png")
    # Recursive glob (what _stage_plot uses) must equal the returned PNG list:
    # no scatter PNG is silently uncounted.
    assert written_pngs == on_disk_pngs
    # And the scatter is among them.
    assert any(p.name == _SCATTER_NAME for p in on_disk_pngs)


def test_plot_methods_filters_every_plot_to_subset(tmp_path: Path) -> None:
    # plot_methods restricts ALL plots (scatter + line plots + legend + index) to
    # the named subset, plotting straight from the existing aggregates -- no
    # re-scoring. _full_rows() carries diffmean_decoder_only + linear_probe.
    plots_dir = tmp_path / "plots"
    written = plot_bpb_adv_summaries(
        _macro_table(_with_feature_diffmean_targets(_full_rows())),
        plots_dir,
        plot_methods=["diffmean_decoder_only"],
    )
    assert _legend_method_names(plots_dir) == {"diffmean_decoder_only"}
    # Plots are still emitted for the surviving family.
    assert any(p.name == _SCATTER_NAME for p in written)
    assert any(p.name == "primary_margin_vs_strength.png" for p in written)


def test_plot_methods_none_includes_all_methods(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    plot_bpb_adv_summaries(
        _macro_table(_with_feature_diffmean_targets(_full_rows())),
        plots_dir,
        plot_methods=None,
    )
    assert _legend_method_names(plots_dir) == {
        "diffmean_decoder_only",
        "feature_diffmean_encdec",
        "linear_probe",
    }


def test_plot_methods_unknown_raises_and_does_not_wipe_existing(
    tmp_path: Path,
) -> None:
    # A typo in plot_methods must fail LOUDLY (not silently drop everything) AND
    # must not clobber a prior plot run: validation happens before the clear.
    plots_dir = tmp_path / "plots"
    rows_with_target = _with_feature_diffmean_targets(_full_rows())
    plot_bpb_adv_summaries(
        _macro_table(rows_with_target),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )
    before = sorted(plots_dir.rglob("*.png"))
    assert before
    with pytest.raises(AxBenchBpbAdvError, match="plot_methods"):
        plot_bpb_adv_summaries(
            _macro_table(rows_with_target), plots_dir, plot_methods=["typo_method"]
        )
    assert sorted(plots_dir.rglob("*.png")) == before


def test_end_to_end_line_and_scatter_idempotent(tmp_path: Path) -> None:
    # End-to-end (both modes): line plots AND scatter, and a re-run is idempotent
    # (no duplicate PNGs, same path set, stale outputs cleared).
    positive_rows = _full_rows()
    negative_rows = [r | {"mode": "negative"} for r in _full_rows()]
    rows = _with_feature_diffmean_targets([*positive_rows, *negative_rows])
    plots_dir = tmp_path / "plots"

    first = plot_bpb_adv_summaries(
        _macro_table(rows),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )
    first_pngs = sorted(p for p in first if p.suffix == ".png")
    # Both line and scatter plots, in BOTH modes.
    for mode in ("positive", "negative"):
        assert (plots_dir / mode / _SCATTER_NAME).exists()
        assert (plots_dir / mode / "primary_margin_vs_strength.png").exists()

    second = plot_bpb_adv_summaries(
        _macro_table(rows),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )
    second_pngs = sorted(p for p in second if p.suffix == ".png")
    # Idempotent: identical PNG path set across runs, and the on-disk set matches.
    assert first_pngs == second_pngs
    assert sorted(plots_dir.rglob("*.png")) == second_pngs
