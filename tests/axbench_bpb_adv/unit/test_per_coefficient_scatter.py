"""Unit tests for the per-coefficient steerability scatters (Feature B).

The default ``steerability_vs_baseline.png`` reduces each steering family to its
REPRESENTATIVE (best) coefficient. Feature B adds a ``by_coefficient/``
subdirectory with one scatter per distinct ``coefficient_str`` (the magnitude is
FIXED instead of chosen), while keeping the default plots byte-identical.

These tests pin the small filesystem-slug / sort / distinct helpers and the
``_plot_per_coefficient_scatters`` fanout.
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib.axes
import matplotlib.figure
import pyarrow as pa
import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    SCATTER_PLOT_NAME,
    _coefficient_slug,
    _coefficient_sort_key,
    _distinct_coefficients_sorted,
    _filter_rows_to_representative_coefficient,
    _fixed_coefficient_selector,
    _plot_per_coefficient_scatters,
    _plot_steerability_vs_baseline_scatter,
    _scatter_points,
    _scatter_trajectories,
    _signed_margin,
    plot_bpb_adv_summaries,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import MACRO_AGG_SCHEMA


def _macro_row(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    coefficient_str: str | None = "1.0",
    adversarial_method: str = "faster_gcg",
    adversarial_strength: float = 0.0,
    strength_index: int = 0,
    context_signal_mode: str = "fixed_context",
    mean_loss_concept_rich: float = 1.0,
    mean_loss_concept_avoiding: float = 3.0,
    baseline_primary_margin: float | None = None,
    general_val_bpb: float | None = None,
    general_val_corpus: str | None = None,
    is_control: bool | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": adversarial_method,
        "adversarial_strength": adversarial_strength,
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": context_signal_mode,
        "mean_loss_concept_rich": mean_loss_concept_rich,
        "mean_loss_concept_avoiding": mean_loss_concept_avoiding,
        "primary_margin_y": _signed_margin(
            mode=mode,
            mean_rich=mean_loss_concept_rich,
            mean_avoid=mean_loss_concept_avoiding,
        ),
        "baseline_primary_margin": baseline_primary_margin,
        "general_val_bpb": general_val_bpb,
        "general_val_corpus": general_val_corpus,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": (adversarial_strength == 0.0) if is_control is None else is_control,
    }


def _two_coefficient_family() -> list[dict[str, Any]]:
    """One positive-mode ``diffmean_decoder_only`` family at two coefficients.

    ``coefficient_str`` in {"0.2", "1.0"}, ``strength_index`` 0..2 each. The "1.0"
    family carries the LARGER control ``baseline_primary_margin`` so it wins the
    representative-coefficient selection used by the default scatter.
    """

    rows: list[dict[str, Any]] = []
    for coefficient_str, control_margin in (("0.2", 0.5), ("1.0", 3.0)):
        for strength_index in range(3):
            rows.append(
                _macro_row(
                    coefficient_str=coefficient_str,
                    adversarial_strength=float(strength_index),
                    strength_index=strength_index,
                    mean_loss_concept_rich=1.0 + float(strength_index),
                    mean_loss_concept_avoiding=3.0 + float(strength_index),
                    baseline_primary_margin=(control_margin if strength_index == 0 else None),
                )
            )
    return rows


def _feature_diffmean_target_row(score: float = 3.0) -> dict[str, Any]:
    return _macro_row(
        method_name="feature_diffmean_encdec",
        target_model="encdec",
        coefficient_str="None",
        mean_loss_concept_rich=1.0,
        mean_loss_concept_avoiding=1.0 + score,
        baseline_primary_margin=score,
    )


def _with_feature_diffmean_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [*rows, _feature_diffmean_target_row()]


def _select_with_feature_diffmean_target(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_with_feature_diffmean_target(rows),
    )


# ---------------------------------------------------------------------------
# _coefficient_slug
# ---------------------------------------------------------------------------


def test_coefficient_slug_none() -> None:
    assert _coefficient_slug("None") == "none"


def test_coefficient_slug_decimal() -> None:
    assert _coefficient_slug("0.2") == "0p2"


def test_coefficient_slug_trailing_zero_normalized() -> None:
    # "6.0" -> "6" (float %g drops the trailing .0), then "." -> "p" is a no-op.
    assert _coefficient_slug("6.0") == "6"


def test_coefficient_slug_one_point_five() -> None:
    assert _coefficient_slug("1.5") == "1p5"


def test_coefficient_slug_negative() -> None:
    assert _coefficient_slug("-1.5") == "neg_1p5"


def test_coefficient_slug_integer_string() -> None:
    assert _coefficient_slug("2") == "2"


# ---------------------------------------------------------------------------
# _coefficient_sort_key / _distinct_coefficients_sorted
# ---------------------------------------------------------------------------


def test_distinct_coefficients_sorted_none_first_then_ascending() -> None:
    rows = [
        _macro_row(coefficient_str="6.0"),
        _macro_row(coefficient_str="0.2"),
        _macro_row(coefficient_str="None"),
        _macro_row(coefficient_str="1.0"),
        _macro_row(coefficient_str="0.2"),
    ]
    assert _distinct_coefficients_sorted(rows) == ["None", "0.2", "1.0", "6.0"]


def test_coefficient_sort_key_orders_none_numeric_nonnumeric() -> None:
    keys = sorted(["3.0", "None", "weird", "0.5"], key=_coefficient_sort_key)
    # None first, then numeric ascending, then non-numeric lexicographic last.
    assert keys == ["None", "0.5", "3.0", "weird"]


# ---------------------------------------------------------------------------
# _fixed_coefficient_selector
# ---------------------------------------------------------------------------


def test_fixed_coefficient_selector_keeps_only_target() -> None:
    rows = _two_coefficient_family()
    kept = _fixed_coefficient_selector("0.2")(rows)
    assert {str(r["coefficient_str"]) for r in kept} == {"0.2"}
    assert len(kept) == 3  # all three strengths of the "0.2" family


# ---------------------------------------------------------------------------
# _plot_per_coefficient_scatters fanout
# ---------------------------------------------------------------------------


def test_per_coefficient_scatters_emit_one_png_per_distinct_coefficient(
    tmp_path: Path,
) -> None:
    rows = _two_coefficient_family()
    by_coeff_dir = tmp_path / "by_coefficient"
    written = _plot_per_coefficient_scatters(
        rows,
        by_coeff_dir,
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )

    coeff_02 = by_coeff_dir / "coeff_0p2" / SCATTER_PLOT_NAME
    coeff_1 = by_coeff_dir / "coeff_1" / SCATTER_PLOT_NAME
    assert coeff_02.exists()
    assert coeff_1.exists()
    written_paths = [path for path, _n in written]
    assert coeff_02 in written_paths
    assert coeff_1 in written_paths


def test_per_coefficient_scatters_write_plot_index(tmp_path: Path) -> None:
    rows = _two_coefficient_family()
    by_coeff_dir = tmp_path / "by_coefficient"
    _plot_per_coefficient_scatters(
        rows,
        by_coeff_dir,
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )
    index_path = by_coeff_dir / "plot_index.csv"
    assert index_path.exists()
    with index_path.open(encoding="utf-8") as f:
        index_rows = list(csv.DictReader(f))
    coefficients = {row["coefficient_str"] for row in index_rows}
    assert coefficients == {"0.2", "1.0"}
    # Each index row's relpath resolves under by_coefficient/.
    for row in index_rows:
        assert (by_coeff_dir / row["path"]).exists()
        assert row["plot_name"] == SCATTER_PLOT_NAME


def test_default_selector_is_byte_identical_representative_trajectory(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # Step 2 byte-identity guard: with DEFAULT args (no new kwargs) the
    # parametrized scatter still draws exactly the representative-coefficient
    # trajectories -- the same (xs, ys) ax.plot calls as before parametrization.
    captured: list[tuple[list[float], list[float]]] = []
    original_plot = matplotlib.axes.Axes.plot

    def _spy_plot(self: Any, *args: Any, **kwargs: Any) -> Any:
        if len(args) >= 2:
            captured.append((list(args[0]), list(args[1])))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

    rows = _two_coefficient_family()
    out_path = tmp_path / "positive" / SCATTER_PLOT_NAME
    result = _plot_steerability_vs_baseline_scatter(
        rows,
        out_path,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_target,
    )
    assert result == out_path

    expected = [
        ([p.x for p in trajectory], [p.y for p in trajectory])
        for trajectory in _scatter_trajectories(
            _scatter_points(
                _filter_rows_to_representative_coefficient(
                    rows,
                    baseline_rows=_with_feature_diffmean_target(rows),
                ),
                scatter_baseline_x="baseline_loss",
            )
        )
    ]
    assert captured == expected


def test_per_coefficient_fixes_magnitude_not_representative(tmp_path: Path) -> None:
    # The default scatter reduces to the representative ("1.0") coefficient, but
    # the by_coefficient subdir must ALSO draw the non-representative "0.2"
    # magnitude (the whole point of Feature B: fix the magnitude, don't pick best).
    rows = _two_coefficient_family()
    representative = {
        str(r["coefficient_str"])
        for r in _filter_rows_to_representative_coefficient(
            rows,
            baseline_rows=_with_feature_diffmean_target(rows),
        )
    }
    assert representative == {"1.0"}  # default scatter keeps only "1.0"

    by_coeff_dir = tmp_path / "by_coefficient"
    _plot_per_coefficient_scatters(
        rows,
        by_coeff_dir,
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )
    # by_coefficient still produced the NON-representative magnitude.
    assert (by_coeff_dir / "coeff_0p2" / SCATTER_PLOT_NAME).exists()


# ---------------------------------------------------------------------------
# F6: _coefficient_slug is non-injective ("6" and "6.0" both -> "coeff_6"), so a
# mixed int/float grid would overwrite a PNG. The fanout must DETECT the collision
# and raise a clear error naming both colliding coefficient_str values, rather than
# silently emit a single (wrong) figure.
# ---------------------------------------------------------------------------


def test_per_coefficient_slug_collision_raises(tmp_path: Path) -> None:
    # Sanity: the two distinct strings genuinely collide on the slug.
    assert _coefficient_slug("6") == _coefficient_slug("6.0") == "6"

    # One family at "6" and another at "6.0": distinct coefficient_str values that
    # collapse to the same coeff_6 directory -> a silent overwrite without the guard.
    rows = _coeff_family_at("6") + _coeff_family_at("6.0")
    by_coeff_dir = tmp_path / "by_coefficient"
    with pytest.raises(AxBenchBpbAdvError, match="slug collision"):
        _plot_per_coefficient_scatters(
            rows,
            by_coeff_dir,
            scatter_baseline_x="baseline_loss",
            mode="positive",
        )


def _coeff_family_at(coefficient_str: str) -> list[dict[str, Any]]:
    """A drawable 3-strength family at one coefficient (control margin present)."""

    rows: list[dict[str, Any]] = []
    for strength_index in range(3):
        rows.append(
            _macro_row(
                coefficient_str=coefficient_str,
                adversarial_strength=float(strength_index),
                strength_index=strength_index,
                mean_loss_concept_rich=1.0 + float(strength_index),
                mean_loss_concept_avoiding=3.0 + float(strength_index),
                baseline_primary_margin=(3.0 if strength_index == 0 else None),
            )
        )
    return rows


# ---------------------------------------------------------------------------
# F14: the per-coefficient scatter footer must NOT claim "representative
# coefficient" -- it fixes a magnitude. It says "at a fixed base coefficient
# ({coefficient})" instead, and differs from the default scatter's footer.
# ---------------------------------------------------------------------------


def test_per_coefficient_footer_differs_from_representative(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: list[str] = []
    original_text = matplotlib.figure.Figure.text

    def _spy_text(self: Any, x: float, y: float, s: str, *args: Any, **kwargs: Any) -> Any:
        # The footer is the bottom-left (0.01, 0.01) annotation.
        if (x, y) == (0.01, 0.01):
            captured.append(s)
        return original_text(self, x, y, s, *args, **kwargs)

    monkeypatch.setattr(matplotlib.figure.Figure, "text", _spy_text)

    rows = _two_coefficient_family()

    # Default (representative-coefficient) scatter footer.
    _plot_steerability_vs_baseline_scatter(
        rows,
        tmp_path / "default" / SCATTER_PLOT_NAME,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_target,
    )
    default_footer = captured[-1]
    assert "representative coefficient" in default_footer

    # Per-coefficient (fixed-magnitude) scatters.
    _plot_per_coefficient_scatters(
        rows,
        tmp_path / "by_coefficient",
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )
    fixed_footers = [s for s in captured if "fixed base coefficient" in s]
    # At least one fixed-magnitude footer was emitted, it names a concrete
    # coefficient, and it differs from the default representative footer.
    assert fixed_footers
    assert any("0.2" in s for s in fixed_footers)
    assert all("representative coefficient" not in s for s in fixed_footers)
    assert all(s != default_footer for s in fixed_footers)


# ---------------------------------------------------------------------------
# A1: the per-coefficient scatter's ``title_extra`` (", coefficient=<c>") is the
# only per-coefficient differentiator on the title; the default/representative
# scatter passes title_extra="" and so its title must NOT carry ", coefficient=".
# Spy ax.set_title (mirrors the footer-spy above).
# ---------------------------------------------------------------------------


def test_per_coefficient_title_names_coefficient_default_does_not(
    tmp_path: Path, monkeypatch: Any
) -> None:
    captured: list[str] = []
    original_set_title = matplotlib.axes.Axes.set_title

    def _spy_set_title(self: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        captured.append(label)
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", _spy_set_title)

    rows = _two_coefficient_family()

    # Default (representative) scatter title: no per-coefficient marker.
    _plot_steerability_vs_baseline_scatter(
        rows,
        tmp_path / "default" / SCATTER_PLOT_NAME,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_target,
    )
    default_title = captured[-1]
    assert ", coefficient=" not in default_title

    # Per-coefficient (fixed-magnitude) scatters: each title names its coefficient.
    _plot_per_coefficient_scatters(
        rows,
        tmp_path / "by_coefficient",
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )
    per_coeff_titles = [t for t in captured if ", coefficient=" in t]
    assert per_coeff_titles, "per-coefficient scatter must embed its coefficient"
    assert any(", coefficient=0.2" in t for t in per_coeff_titles)


# ---------------------------------------------------------------------------
# A2/A3: a coefficient whose rows ALL have a null x-baseline yields zero scatter
# points, so ``_plot_steerability_vs_baseline_scatter`` returns None and the
# fanout's ``path is None: continue`` skips it -- no PNG, not in the returned
# list, no index row. With ALL coefficients undrawable, no plot_index.csv at all.
# ---------------------------------------------------------------------------


def _null_x_family(coefficient_str: str) -> list[dict[str, Any]]:
    """A 3-strength family at one coefficient with a NULL baseline-x for every row.

    Under ``scatter_baseline_x="baseline_loss"`` + positive mode the x is read
    from ``mean_loss_concept_avoiding``; ``None`` there makes ``_scatter_points``
    yield zero points for this coefficient. ``primary_margin_y`` is set to None
    directly (the macro column is nullable) so we never call ``_signed_margin``
    with a None loss.
    """

    rows: list[dict[str, Any]] = []
    for strength_index in range(3):
        row = _macro_row(
            coefficient_str=coefficient_str,
            adversarial_strength=float(strength_index),
            strength_index=strength_index,
            mean_loss_concept_rich=1.0 + float(strength_index),
            mean_loss_concept_avoiding=3.0 + float(strength_index),
            baseline_primary_margin=(3.0 if strength_index == 0 else None),
        )
        row["mean_loss_concept_avoiding"] = None
        row["primary_margin_y"] = None
        rows.append(row)
    return rows


def test_per_coefficient_null_x_coefficient_is_skipped(tmp_path: Path) -> None:
    # "0.2" is all-null-x (undrawable); "1.0" is drawable.
    rows = _null_x_family("0.2") + _coeff_family_at("1.0")
    by_coeff_dir = tmp_path / "by_coefficient"
    written = _plot_per_coefficient_scatters(
        rows,
        by_coeff_dir,
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )

    null_png = by_coeff_dir / "coeff_0p2" / SCATTER_PLOT_NAME
    drawable_png = by_coeff_dir / "coeff_1" / SCATTER_PLOT_NAME
    # The undrawable coefficient: no PNG, not in the written list.
    assert not null_png.exists()
    written_paths = [path for path, _n in written]
    assert null_png not in written_paths
    # The drawable coefficient IS present.
    assert drawable_png.exists()
    assert drawable_png in written_paths

    # The undrawable coefficient has NO row in the by_coefficient plot_index.csv,
    # while the drawable one does.
    index_path = by_coeff_dir / "plot_index.csv"
    assert index_path.exists()
    with index_path.open(encoding="utf-8") as f:
        index_coefficients = {row["coefficient_str"] for row in csv.DictReader(f)}
    assert index_coefficients == {"1.0"}


def test_per_coefficient_all_undrawable_writes_no_index(tmp_path: Path) -> None:
    # ALL coefficients null-x -> empty index_rows -> the ``if index_rows`` branch is
    # false -> no plot_index.csv is written and the returned list is empty.
    rows = _null_x_family("0.2") + _null_x_family("1.0")
    by_coeff_dir = tmp_path / "by_coefficient"
    written = _plot_per_coefficient_scatters(
        rows,
        by_coeff_dir,
        scatter_baseline_x="baseline_loss",
        mode="positive",
    )

    assert written == []
    assert not (by_coeff_dir / "plot_index.csv").exists()


# ---------------------------------------------------------------------------
# A4 (cheap helpers + logger): _coefficient_slug non-numeric ValueError branch,
# _distinct_coefficients_sorted([]) == [], and the per-subset "emitted N
# per-coefficient scatters" log line via caplog.
# ---------------------------------------------------------------------------


def test_coefficient_slug_non_numeric_passthrough() -> None:
    # float("weird") raises ValueError -> the slug falls back to the raw string.
    assert _coefficient_slug("weird") == "weird"


def test_coefficient_slug_non_numeric_replaces_special_chars() -> None:
    # The ValueError fallback path still runs the .replace(): a non-numeric string
    # with a '-' and a '.' gets those characters slugged (teeth on the .replace).
    assert _coefficient_slug("a-b.c") == "aneg_bpc"


def test_distinct_coefficients_sorted_empty() -> None:
    assert _distinct_coefficients_sorted([]) == []


def test_per_coefficient_logs_emitted_count(tmp_path: Path, caplog: Any) -> None:
    # ``plot_bpb_adv_summaries`` logs "emitted N per-coefficient scatters" per
    # subset. The two-coefficient fixture is one (alpha=1.0, positive) subset with
    # two drawable coefficients, so the log must report exactly 2.
    table = pa.Table.from_pylist(
        _with_feature_diffmean_target(_two_coefficient_family()),
        schema=MACRO_AGG_SCHEMA,
    )
    with caplog.at_level(logging.INFO):
        plot_bpb_adv_summaries(
            table,
            tmp_path / "plots",
            plot_methods=["diffmean_decoder_only"],
        )

    assert "emitted 2 per-coefficient scatters" in caplog.text


# ---------------------------------------------------------------------------
# Issue #635: the single-corpus check guards the DRAWN rows -- it runs on the
# post-``plot_methods``-filter superset, per invocation. A family excluded by
# the whitelist cannot poison the invocation; the same table without the
# whitelist must raise.
# ---------------------------------------------------------------------------


def test_plot_bpb_adv_summaries_corpus_check_applies_to_filtered_rows(
    tmp_path: Path,
) -> None:
    mixed_corpus_rows = [
        _macro_row(
            method_name="linear_probe_decoder_only",
            general_val_bpb=9.0,
            general_val_corpus="nd=9;nb=99;h=bb",
        ),
        _macro_row(
            method_name="linear_probe_decoder_only",
            coefficient_str="2.0",
            general_val_bpb=9.5,
            general_val_corpus="nd=7;nb=77;h=cc",
        ),
    ]
    table = pa.Table.from_pylist(
        _with_feature_diffmean_target([*_two_coefficient_family(), *mixed_corpus_rows]),
        schema=MACRO_AGG_SCHEMA,
    )

    # Unfiltered: the mixed-corpus linear_probe family is drawn -> raise.
    with pytest.raises(AxBenchBpbAdvError, match="mixes general-val corpus signatures"):
        plot_bpb_adv_summaries(table, tmp_path / "unfiltered")

    # Whitelisting the clean family excludes the offender from the drawn rows:
    # the SAME table must plot cleanly (check applies post-filter).
    written = plot_bpb_adv_summaries(
        table,
        tmp_path / "filtered",
        plot_methods=["diffmean_decoder_only"],
    )
    assert written
