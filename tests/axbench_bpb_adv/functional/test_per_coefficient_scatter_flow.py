"""Functional tests: ``plot_bpb_adv_summaries`` emits the per-coefficient
steerability scatters under ``by_coefficient/`` ALONGSIDE the default
representative-coefficient scatter (Feature B).

The default ``steerability_vs_baseline.png`` (representative coefficient) must
stay byte-identical; the new ``by_coefficient/coeff_<slug>/...`` scatters fix the
magnitude instead. The top-level ``plot_index.csv`` must NOT reference the
per-coefficient files (they are self-contained under ``by_coefficient/``), and a
re-run must be idempotent (no stale per-coefficient PNGs left behind).
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pyarrow as pa

from steering.eval.axbench_bpb_adv.aggregation import (
    _fixed_coefficient_selector,
    _scatter_points,
    plot_bpb_adv_summaries,
)
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
    coefficient_str: str,
    control_margin: float,
    n_strengths: int = 3,
) -> list[dict[str, Any]]:
    rich = 1.0
    avoid = 1.0 + control_margin
    rows = [
        _macro_row(
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


def _macro_two_coeffs() -> list[dict[str, Any]]:
    """One positive-mode family, coefficient_str in {"0.2","1.0"}; "1.0" wins."""

    rows: list[dict[str, Any]] = []
    rows += _coeff_family(coefficient_str="0.2", control_margin=0.5)
    rows += _coeff_family(coefficient_str="1.0", control_margin=3.0)
    return rows


def _feature_diffmean_target_row(
    row_like: dict[str, Any],
    *,
    score: float = 3.0,
) -> dict[str, Any]:
    mode = str(row_like["mode"])
    method_name = "neg_feature_diffmean_encdec" if mode == "negative" else "feature_diffmean_encdec"
    row = _macro_row(
        mode=mode,
        method_name=method_name,
        coefficient_str="None",
        adversarial_strength=0.0,
        strength_index=0,
        primary_margin_y=score,
        baseline_primary_margin=score,
    )
    row["target_model"] = "encdec"
    row["adversarial_method"] = str(row_like["adversarial_method"])
    row["strength_unit"] = str(row_like["strength_unit"])
    row["context_signal_mode"] = str(row_like["context_signal_mode"])
    return row


def _with_feature_diffmean_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    target_methods = {"feature_diffmean_encdec", "neg_feature_diffmean_encdec"}
    existing = {
        (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        for row in rows
        if str(row["method_name"]) in target_methods
        and str(row["target_model"]) == "encdec"
    }
    targets: list[dict[str, Any]] = []
    seen = set(existing)
    for row in rows:
        key = (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        if key in seen:
            continue
        seen.add(key)
        targets.append(_feature_diffmean_target_row(row))
    return [*rows, *targets]


def _macro_table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(_with_feature_diffmean_target(rows), schema=MACRO_AGG_SCHEMA)


def _plot_index_paths(plots_dir: Path) -> list[str]:
    with (plots_dir / "plot_index.csv").open(encoding="utf-8") as f:
        return [row["path"] for row in csv.DictReader(f)]


def test_per_coefficient_scatters_emitted_under_by_coefficient(
    tmp_path: Path,
) -> None:
    plots_dir = tmp_path / "plots"
    written = plot_bpb_adv_summaries(_macro_table(_macro_two_coeffs()), plots_dir)

    subset = plots_dir / "positive"
    # Default representative-coefficient scatter still emitted alongside.
    assert (subset / _SCATTER_NAME).exists()
    # One per-coefficient scatter per distinct magnitude under by_coefficient/.
    coeff_02 = subset / "by_coefficient" / "coeff_0p2" / _SCATTER_NAME
    coeff_1 = subset / "by_coefficient" / "coeff_1" / _SCATTER_NAME
    assert coeff_02.exists()
    assert coeff_1.exists()
    # The self-contained by_coefficient index is written.
    assert (subset / "by_coefficient" / "plot_index.csv").exists()
    # The returned written list includes the by_coefficient paths.
    assert coeff_02 in written
    assert coeff_1 in written


def test_top_level_plot_index_excludes_by_coefficient(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    plot_bpb_adv_summaries(_macro_table(_macro_two_coeffs()), plots_dir)
    # The top-level plot_index references the default plots only -- the
    # per-coefficient scatters live in their OWN by_coefficient/plot_index.csv.
    top_level_paths = _plot_index_paths(plots_dir)
    assert top_level_paths  # not empty
    assert not any("by_coefficient" in path for path in top_level_paths)


def test_per_coefficient_scatters_idempotent(tmp_path: Path) -> None:
    plots_dir = tmp_path / "plots"
    rows = _macro_two_coeffs()

    first = plot_bpb_adv_summaries(_macro_table(rows), plots_dir)
    first_pngs = sorted(plots_dir.rglob("*.png"))
    # Both default scatter and the per-coefficient scatters are present.
    assert any(
        p.parent.name == "coeff_0p2" for p in first_pngs
    ), "per-coefficient scatter must be on disk"

    second = plot_bpb_adv_summaries(_macro_table(rows), plots_dir)
    second_pngs = sorted(plots_dir.rglob("*.png"))
    # Idempotent: identical on-disk PNG set across runs (stale per-coefficient
    # artifacts are cleared by _clear_previous_plot_outputs's recursive walk).
    assert first_pngs == second_pngs
    # And the returned PNG path sets match too.
    first_written_pngs = sorted(p for p in first if p.suffix == ".png")
    second_written_pngs = sorted(p for p in second if p.suffix == ".png")
    assert first_written_pngs == second_written_pngs


def test_shrinking_coefficient_set_leaves_no_stale_empty_coeff_dir(
    tmp_path: Path,
) -> None:
    # F16: when the coefficient set SHRINKS on a re-run, the now-unused
    # by_coefficient/coeff_<slug>/ directory must not survive as an empty husk --
    # _clear_previous_plot_outputs unlinks its PNG AND removes the emptied dir, so
    # the on-disk tree mirrors the CURRENT coefficient set.
    plots_dir = tmp_path / "plots"
    by_coeff = plots_dir / "positive" / "by_coefficient"

    # First run: two coefficients -> coeff_0p2/ and coeff_1/ both exist.
    plot_bpb_adv_summaries(_macro_table(_macro_two_coeffs()), plots_dir)
    assert (by_coeff / "coeff_0p2").is_dir()
    assert (by_coeff / "coeff_1").is_dir()

    # Re-run with ONLY the "1.0" coefficient -> coeff_0p2/ must be gone entirely
    # (not an empty leftover directory), coeff_1/ remains.
    plot_bpb_adv_summaries(
        _macro_table(_coeff_family(coefficient_str="1.0", control_margin=3.0)),
        plots_dir,
    )
    assert not (by_coeff / "coeff_0p2").exists(), (
        "the dropped coefficient's by_coefficient dir must be removed, not left "
        "as an empty husk"
    )
    assert (by_coeff / "coeff_1").is_dir()
    # No empty directories remain anywhere under plots_dir.
    empty_dirs = [
        p
        for p in plots_dir.rglob("*")
        if p.is_dir() and not any(p.iterdir())
    ]
    assert empty_dirs == [], f"stale empty dirs left behind: {empty_dirs}"


def test_by_coefficient_index_n_rows_matches_drawn_point_count(
    tmp_path: Path,
) -> None:
    # The index ``n_rows`` and the figure's drawn points are computed on SEPARATE
    # code paths (the index in ``_plot_per_coefficient_scatters``, the points via
    # the ``_plot_steerability_vs_baseline_scatter`` selector). Pin them in lockstep
    # so a future refactor of either path cannot silently let the count drift from
    # the figure.
    plots_dir = tmp_path / "plots"
    macro_rows = _with_feature_diffmean_target(_macro_two_coeffs())
    plot_bpb_adv_summaries(pa.Table.from_pylist(macro_rows, schema=MACRO_AGG_SCHEMA), plots_dir)

    index_path = (
        plots_dir / "positive" / "by_coefficient" / "plot_index.csv"
    )
    with index_path.open(encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    assert index_rows  # the fixture has two coefficients, both drawable

    for row in index_rows:
        expected = len(
            _scatter_points(
                _fixed_coefficient_selector(row["coefficient_str"])(macro_rows),
                scatter_baseline_x="baseline_loss",
            )
        )
        assert int(row["n_rows"]) == expected


def _no_steering_macro_row(
    *,
    method_name: str,
    adversarial_method: str,
    strength_index: int,
    context_signal_mode: str,
) -> dict[str, Any]:
    """A ``coefficient_str == "None"`` (no-steering) macro row.

    ``coefficient_str`` holds the STRING ``"None"`` (= ``str(None)``), NOT Python
    ``None``; this is exactly the case a reviewer feared ``_scatter_points`` would
    drop via its ``if coefficient is None`` guard. ``strength_index`` orders the
    counter-steering sweep; ``is_control`` is the k=0 control.
    """

    rich = 1.0 + float(strength_index)
    avoid = 3.0 + float(strength_index)
    return {
        "mode": "positive",
        "method_name": method_name,
        "target_model": "decoder_only",
        "coefficient_str": "None",
        "adversarial_method": adversarial_method,
        "adversarial_strength": float(strength_index),
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": context_signal_mode,
        "mean_loss_concept_rich": rich,
        "mean_loss_concept_avoiding": avoid,
        "primary_margin_y": avoid - rich,
        "baseline_primary_margin": (avoid - rich) if strength_index == 0 else None,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": strength_index == 0,
    }


def test_coeff_none_no_steering_scatter_is_emitted(tmp_path: Path) -> None:
    # USER REQUIREMENT: the "no steering" per-magnitude view must be emitted. "No
    # steering" families carry coefficient_str == "None" (the STRING, e.g.
    # control_decoder_only). A reviewer worried coeff_none "can essentially never
    # be emitted" because _scatter_points skips rows where coefficient IS None --
    # but the macro column holds str(None) == "None", which is NOT Python None, so
    # the rows survive and the coeff_none scatter is drawn.
    rows: list[dict[str, Any]] = [
        _no_steering_macro_row(
            method_name="control_decoder_only",
            adversarial_method="additive_direction",
            strength_index=strength_index,
            context_signal_mode="not_applicable",
        )
        for strength_index in range(3)
    ]
    # Alongside a numeric-coefficient family (the no-steering view must coexist).
    rows += _coeff_family(coefficient_str="1.0", control_margin=3.0)

    plots_dir = tmp_path / "plots"
    plot_bpb_adv_summaries(_macro_table(rows), plots_dir)

    coeff_none_png = (
        plots_dir
        / "positive"
        / "by_coefficient"
        / "coeff_none"
        / _SCATTER_NAME
    )
    assert coeff_none_png.exists(), (
        "the coefficient_str == 'None' (no-steering) scatter must be emitted; "
        "string 'None' must NOT be dropped by _scatter_points's `is None` guard"
    )

    index_path = (
        plots_dir / "positive" / "by_coefficient" / "plot_index.csv"
    )
    with index_path.open(encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    none_index_rows = [
        row for row in index_rows if row["coefficient_str"] == "None"
    ]
    assert len(none_index_rows) == 1, "exactly one coeff_none index row expected"
    assert int(none_index_rows[0]["n_rows"]) >= 1


def _negative_coeff_family(
    *, coefficient_str: str, control_margin: float, n_strengths: int = 3
) -> list[dict[str, Any]]:
    """A negative-mode drawable family (distinct (alpha, mode) subset).

    Under ``scatter_baseline_x="baseline_loss"`` + negative mode the x is read
    from ``mean_loss_concept_rich`` (kept non-null here), so the family draws.
    """

    rows: list[dict[str, Any]] = []
    for strength_index in range(n_strengths):
        rows.append(
            _macro_row(
                mode="negative",
                coefficient_str=coefficient_str,
                adversarial_strength=float(strength_index),
                strength_index=strength_index,
                mean_loss_concept_rich=2.0 + float(strength_index),
                mean_loss_concept_avoiding=1.0,
                primary_margin_y=control_margin - float(strength_index),
                baseline_primary_margin=(
                    control_margin if strength_index == 0 else None
                ),
            )
        )
    return rows


def test_orphaned_nested_plot_index_removed_when_subset_vanishes(
    tmp_path: Path,
) -> None:
    # A4 (distinct from test_shrinking_..._empty_coeff_dir, which keeps the
    # positive subset ALIVE at "1.0"): here the ENTIRE positive subset
    # vanishes on the re-run (only negative-mode rows remain), so its nested
    # by_coefficient/plot_index.csv is now orphaned. _clear_previous_plot_outputs'
    # recursive rglob("plot_index.csv") unlink must remove it (and the empty dirs).
    plots_dir = tmp_path / "plots"

    # First run: positive-mode, two coefficients -> the positive subset's nested
    # by_coefficient/plot_index.csv is written.
    plot_bpb_adv_summaries(_macro_table(_macro_two_coeffs()), plots_dir)
    positive_nested_index = (
        plots_dir / "positive" / "by_coefficient" / "plot_index.csv"
    )
    assert positive_nested_index.exists(), "first run must write the nested index"

    # Re-run with ONLY negative-mode rows -> the positive subset disappears.
    plot_bpb_adv_summaries(
        _macro_table(
            _negative_coeff_family(coefficient_str="1.0", control_margin=3.0)
        ),
        plots_dir,
    )

    # The orphaned positive-subset nested index (and its whole subtree) is gone.
    assert not positive_nested_index.exists(), (
        "the vanished subset's orphaned nested plot_index.csv must be removed"
    )
    assert not (plots_dir / "positive").exists()
    # The surviving negative subset DID write its own nested index.
    assert (
        plots_dir / "negative" / "by_coefficient" / "plot_index.csv"
    ).exists()
    # No stale nested plot_index.csv lingers anywhere outside the negative subtree.
    surviving = sorted(plots_dir.rglob("plot_index.csv"))
    assert all("negative" in str(p) or p.parent == plots_dir for p in surviving)
