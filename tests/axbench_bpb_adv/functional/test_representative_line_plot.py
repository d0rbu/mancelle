"""Functional tests: the regular-steering LINE plot shows only the
representative coefficient per family while keeping the full adversarial-strength
sweep (spec 2D, Feature 4)."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    _representative_family_key,
    _select_representative_coefficient,
    plot_bpb_adv_summaries,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import MACRO_AGG_SCHEMA


def _macro_row(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    coefficient_str: str = "1.0",
    adversarial_strength: float = 0.0,
    strength_index: int = 0,
    primary_margin_y: float = 0.0,
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
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 1.0 + primary_margin_y,
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
    rows = [
        _macro_row(
            method_name=method_name,
            coefficient_str=coefficient_str,
            adversarial_strength=0.0,
            strength_index=0,
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
                primary_margin_y=control_margin - float(strength_index),
                baseline_primary_margin=control_margin,
            )
        )
    return rows


def _feature_diffmean_target(score: float = 3.0) -> dict[str, Any]:
    row = _macro_row(
        method_name="feature_diffmean_encdec",
        coefficient_str="None",
        primary_margin_y=score,
        baseline_primary_margin=score,
    )
    row["target_model"] = "encdec"
    return row


def _with_feature_diffmean_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*rows, _feature_diffmean_target()]


def _macro_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=MACRO_AGG_SCHEMA)


def _plot_index_rows(plots_dir: Path) -> list[dict[str, str]]:
    with (plots_dir / "plot_index.csv").open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_primary_line_plot_emitted_and_indexed(tmp_path: Path) -> None:
    # 2 methods x 3 coefficients x 3 strengths.
    rows: list[dict[str, Any]] = []
    for method_name, best_coeff in (
        ("diffmean_decoder_only", "4.0"),
        ("linear_probe", "2.0"),
    ):
        for coeff in ("1.0", "2.0", "4.0"):
            control = 3.0 if coeff == best_coeff else 0.5
            rows.extend(
                _coeff_family(
                    method_name=method_name,
                    coefficient_str=coeff,
                    control_margin=control,
                )
    )

    plots_dir = tmp_path / "plots"
    written = plot_bpb_adv_summaries(
        _macro_table(_with_feature_diffmean_target(rows)),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )

    primary = plots_dir / "positive" / "primary_margin_vs_strength.png"
    assert primary.exists()
    assert primary in written
    index = _plot_index_rows(plots_dir)
    primary_index = [r for r in index if r["plot_name"] == "primary_margin_vs_strength.png"]
    assert primary_index, "primary plot must be registered in plot_index.csv"


def test_missing_target_fails_before_previous_plots_are_cleared(tmp_path: Path) -> None:
    rows = _coeff_family(
        method_name="diffmean_decoder_only",
        coefficient_str="1.0",
        control_margin=1.0,
    )
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    sentinel = plots_dir / "old-plot.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="missing fair feature-DiffMean"):
        plot_bpb_adv_summaries(
            _macro_table(rows),
            plots_dir,
            plot_methods=["diffmean_decoder_only"],
        )

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_null_target_metric_fails_before_previous_plots_are_cleared(tmp_path: Path) -> None:
    rows = _coeff_family(
        method_name="diffmean_decoder_only",
        coefficient_str="1.0",
        control_margin=1.0,
    )
    target = _feature_diffmean_target()
    target["baseline_primary_margin"] = None
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    sentinel = plots_dir / "old-plot.txt"
    sentinel.write_text("keep me", encoding="utf-8")

    with pytest.raises(AxBenchBpbAdvError, match="missing fair feature-DiffMean"):
        plot_bpb_adv_summaries(
            _macro_table([*rows, target]),
            plots_dir,
            plot_methods=["diffmean_decoder_only"],
        )

    assert sentinel.read_text(encoding="utf-8") == "keep me"


def test_primary_line_plot_has_at_most_n_methods_representative_lines(
    tmp_path: Path,
) -> None:
    # 2 methods x 3 coefficients. Each method's representative is ONE coefficient,
    # so the primary line plot must collapse to <= n_methods families.
    n_methods = 2
    rows: list[dict[str, Any]] = []
    for method_name, best_coeff in (
        ("diffmean_decoder_only", "4.0"),
        ("linear_probe", "2.0"),
    ):
        for coeff in ("1.0", "2.0", "4.0"):
            control = 3.0 if coeff == best_coeff else 0.5
            rows.extend(
                _coeff_family(
                    method_name=method_name,
                    coefficient_str=coeff,
                    control_margin=control,
                )
            )

    plots_dir = tmp_path / "plots"
    rows_with_target = _with_feature_diffmean_target(rows)
    plot_bpb_adv_summaries(
        _macro_table(rows_with_target),
        plots_dir,
        plot_methods=["diffmean_decoder_only", "linear_probe"],
    )

    # The primary plot's n_rows must equal the representative-filtered subset
    # (one coefficient per family, full strength sweep), NOT all coefficients.
    index = _plot_index_rows(plots_dir)
    primary_index = next(r for r in index if r["plot_name"] == "primary_margin_vs_strength.png")
    representative = _select_representative_coefficient(
        rows,
        baseline_rows=rows_with_target,
    )
    # One family per method here -> n_methods representative families.
    families = set(representative)
    assert len(families) <= n_methods
    # Each representative family keeps its full 3-strength sweep.
    expected_primary_rows = n_methods * 3
    assert int(primary_index["n_rows"]) == expected_primary_rows
    # Strictly fewer than the full (n_methods * 3 coeffs * 3 strengths) row count.
    assert expected_primary_rows < len(rows)


def test_primary_line_plot_keeps_full_strength_sweep_of_representative(
    tmp_path: Path,
) -> None:
    # One method, two coefficients, 4 strengths each. The representative
    # coefficient's full strength sweep must drive the primary line plot.
    rows = [
        *_coeff_family(
            method_name="diffmean_decoder_only",
            coefficient_str="1.0",
            control_margin=0.5,
            n_strengths=4,
        ),
        *_coeff_family(
            method_name="diffmean_decoder_only",
            coefficient_str="4.0",
            control_margin=3.0,
            n_strengths=4,
        ),
    ]
    plots_dir = tmp_path / "plots"
    rows_with_target = _with_feature_diffmean_target(rows)
    plot_bpb_adv_summaries(
        _macro_table(rows_with_target),
        plots_dir,
        plot_methods=["diffmean_decoder_only"],
    )

    index = _plot_index_rows(plots_dir)
    primary_index = next(r for r in index if r["plot_name"] == "primary_margin_vs_strength.png")
    # Representative "4.0" keeps all 4 strengths; "1.0" is dropped from the line.
    assert int(primary_index["n_rows"]) == 4

    # Diagnostics still show ALL coefficients (full grid, both families).
    avoid_index = next(
        r for r in index if r["plot_name"] == "mean_loss_concept_avoiding_vs_strength.png"
    )
    assert int(avoid_index["n_rows"]) == 8


def test_representative_family_key_excludes_coefficient(tmp_path: Path) -> None:
    a = _macro_row(coefficient_str="1.0")
    b = _macro_row(coefficient_str="4.0")
    # Same family despite different coefficient.
    assert _representative_family_key(a) == _representative_family_key(b)
