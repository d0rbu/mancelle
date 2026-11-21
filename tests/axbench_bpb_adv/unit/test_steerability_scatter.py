"""Unit tests for the steerability-vs-baseline scatter (spec 2C, Feature 5).

``_plot_steerability_vs_baseline_scatter`` plots, per method-family, the family's
**counter-steering-strength sweep**: it reduces each family to its representative
base coefficient (``_filter_rows_to_representative_coefficient``) and then emits
one point per ``adversarial_strength`` level, connected light->dark in strength
order. The swept/connected axis is the COUNTER-STEERING strength, NOT the
evaluated method's own base coefficient -- so a method with a single base config
(LoRA, a no-steering control, encdec-ours) still gets a full multi-point curve.

* y = ``primary_margin_y`` (steerability), recomputed via ``_signed_margin`` from
  the row's ``mean_loss_concept_rich`` / ``mean_loss_concept_avoiding`` -- the
  sign flip lives ONLY in ``_signed_margin`` and is never re-derived here.
* x = per ``scatter_baseline_x``: ``baseline_loss`` -> the natural (unsteered)
  completion BPB, mode-dependent (positive -> ``mean_loss_concept_avoiding``;
  negative -> ``mean_loss_concept_rich``); ``min`` -> ``min(rich, avoid)``.
* color = model/steering family; brightness = counter-steering strength rank;
  marker = method variant (regular circle, roofline star).

``_scatter_points`` itself is a PURE point builder -- one point per input row
(with a non-None coefficient and a non-None x); the representative-coefficient
reduction is applied by the plot function BEFORE calling it. Points with a
``None`` coefficient or a ``None`` baseline (x) are skipped.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pyarrow as pa
import pytest

import steering.eval.axbench_bpb_adv.aggregation as agg_mod
from steering.eval.axbench_bpb_adv.aggregation import (
    _add_style_legend,
    _context_linestyle,
    _filter_rows_to_representative_coefficient,
    _plot_steerability_vs_baseline_scatter,
    _scatter_points,
    _scatter_series_key,
    _scatter_trajectories,
    _scatter_xlabel,
    _ScatterPoint,
    _signed_margin,
    plot_bpb_adv_summaries,
    plot_roundtrip_adv_summaries,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
)


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
        "is_control": adversarial_strength == 0.0,
    }


def _feature_diffmean_target_row(
    *,
    score: float = 3.0,
    mode: str = "positive",
) -> dict[str, Any]:
    method_name = "feature_diffmean_encdec" if mode == "positive" else "neg_feature_diffmean_encdec"
    return _macro_row(
        mode=mode,
        method_name=method_name,
        target_model="encdec",
        coefficient_str="None",
        mean_loss_concept_rich=1.0,
        mean_loss_concept_avoiding=1.0 + score,
        baseline_primary_margin=score,
    )


def _with_feature_diffmean_targets(
    rows: list[dict[str, Any]],
    *,
    positive_score: float = 3.0,
    negative_score: float = 3.0,
) -> list[dict[str, Any]]:
    modes = {str(row["mode"]) for row in rows}
    targets: list[dict[str, Any]] = []
    if "positive" in modes:
        targets.append(_feature_diffmean_target_row(score=positive_score))
    if "negative" in modes:
        targets.append(_feature_diffmean_target_row(score=negative_score, mode="negative"))
    return [*rows, *targets]


def _select_with_feature_diffmean_targets(
    rows: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_with_feature_diffmean_targets(rows),
    )


def _strength_sweep(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    coefficient_str: str | None = "1.0",
    context_signal_mode: str = "fixed_context",
    n_strengths: int = 3,
    control_margin: float | None = 3.0,
    base_rich: float = 1.0,
    base_avoid: float = 3.0,
) -> list[dict[str, Any]]:
    """One family's full counter-steering sweep (k=0 control + adversarial rows).

    The control (strength 0) row carries ``baseline_primary_margin`` so the
    representative-coefficient selection can score it; each successive strength
    perturbs the losses so the points are distinct.
    """

    rows: list[dict[str, Any]] = []
    for strength_index in range(n_strengths):
        rows.append(
            _macro_row(
                mode=mode,
                method_name=method_name,
                coefficient_str=coefficient_str,
                adversarial_strength=float(strength_index),
                strength_index=strength_index,
                context_signal_mode=context_signal_mode,
                mean_loss_concept_rich=base_rich + float(strength_index),
                mean_loss_concept_avoiding=base_avoid + float(strength_index),
                baseline_primary_margin=control_margin if strength_index == 0 else None,
            )
        )
    return rows


# ---------------------------------------------------------------------------
# _scatter_points (pure point construction -- one point per row, ALL strengths)
# ---------------------------------------------------------------------------


def test_scatter_point_xy_baseline_loss() -> None:
    # One point per row (no control-row filter): a 3-strength sweep -> 3 points.
    rows = _strength_sweep(n_strengths=3)
    points = _scatter_points(rows, scatter_baseline_x="baseline_loss")
    assert len(points) == 3
    for row, point in zip(rows, points, strict=True):
        # y is the signed margin recomputed from the loss pair.
        assert point.y == _signed_margin(
            mode=point.mode,
            mean_rich=row["mean_loss_concept_rich"],
            mean_avoid=row["mean_loss_concept_avoiding"],
        )
        # positive mode: the unsteered/natural completion is concept-AVOIDING,
        # so baseline_loss x = mean_loss_concept_avoiding.
        assert point.x == row["mean_loss_concept_avoiding"]


def test_scatter_all_counter_steering_strengths_contribute() -> None:
    # THE behavior change: the scatter sweeps COUNTER-STEERING strength, so every
    # strength row contributes a point (the k=0 control AND each adversarial
    # strength), not only the control row.
    rows = [
        _macro_row(coefficient_str="2.0", adversarial_strength=0.0, strength_index=0),
        _macro_row(coefficient_str="2.0", adversarial_strength=1.0, strength_index=1),
        _macro_row(coefficient_str="2.0", adversarial_strength=2.0, strength_index=2),
    ]
    points = _scatter_points(rows, scatter_baseline_x="baseline_loss")
    assert len(points) == 3
    assert sorted(p.adversarial_strength for p in points) == [0.0, 1.0, 2.0]
    # strength_index is carried on the point (drives the brightness ranking).
    assert sorted(p.strength_index for p in points) == [0, 1, 2]


def test_scatter_baseline_loss_x_is_mode_dependent() -> None:
    """``baseline_loss`` x is the model's NATURAL (unsteered) completion BPB.

    * positive mode steers TOWARD the concept; the natural completion is
      concept-AVOIDING -> x = mean_loss_concept_avoiding.
    * negative mode steers AWAY from the concept; the natural completion is
      concept-RICH -> x = mean_loss_concept_rich (NOT avoiding).
    """

    positive_rows = _strength_sweep(mode="positive")
    negative_rows = _strength_sweep(mode="negative")
    positive_points = _scatter_points(positive_rows, scatter_baseline_x="baseline_loss")
    negative_points = _scatter_points(negative_rows, scatter_baseline_x="baseline_loss")
    assert len(positive_points) == 3
    assert len(negative_points) == 3
    for row, point in zip(positive_rows, positive_points, strict=True):
        assert point.x == row["mean_loss_concept_avoiding"]
    for row, point in zip(negative_rows, negative_points, strict=True):
        assert point.x == row["mean_loss_concept_rich"]
        # And explicitly NOT the avoiding side.
        assert point.x != row["mean_loss_concept_avoiding"]


def test_scatter_xlabel_baseline_loss_is_mode_dependent() -> None:
    assert _scatter_xlabel("baseline_loss", mode="positive") == ("baseline BPB (concept-avoiding)")
    assert _scatter_xlabel("baseline_loss", mode="negative") == ("baseline BPB (concept-rich)")
    # ``min`` is mode-independent (the smaller of the two sides either way).
    assert _scatter_xlabel("min", mode="positive") == _scatter_xlabel("min", mode="negative")
    assert "min" in _scatter_xlabel("min", mode="positive")


def test_scatter_point_y_equals_signed_margin_not_raw_primary() -> None:
    # primary_margin_y in the row is INTENTIONALLY wrong; the point must
    # recompute via _signed_margin and ignore the stored value.
    row = _macro_row(
        mode="positive",
        mean_loss_concept_rich=1.0,
        mean_loss_concept_avoiding=3.0,
    )
    row["primary_margin_y"] = -999.0
    points = _scatter_points([row], scatter_baseline_x="baseline_loss")
    assert len(points) == 1
    assert points[0].y == _signed_margin(mode="positive", mean_rich=1.0, mean_avoid=3.0)
    assert points[0].y == 2.0


def test_scatter_x_switches_baseline_loss_vs_min() -> None:
    # rich < avoid so min picks rich; baseline_loss picks avoid.
    row = _macro_row(
        mean_loss_concept_rich=1.5,
        mean_loss_concept_avoiding=4.0,
    )
    [baseline_point] = _scatter_points([row], scatter_baseline_x="baseline_loss")
    [min_point] = _scatter_points([row], scatter_baseline_x="min")
    assert baseline_point.x == 4.0
    assert min_point.x == 1.5


def test_scatter_min_picks_smaller_side_when_avoid_smaller() -> None:
    row = _macro_row(
        mean_loss_concept_rich=5.0,
        mean_loss_concept_avoiding=2.0,
    )
    [min_point] = _scatter_points([row], scatter_baseline_x="min")
    assert min_point.x == 2.0


def test_scatter_skips_none_coefficient() -> None:
    rows = [
        _macro_row(coefficient_str=None),
        _macro_row(coefficient_str="2.0"),
    ]
    points = _scatter_points(rows, scatter_baseline_x="baseline_loss")
    assert len(points) == 1
    assert points[0].coefficient_str == "2.0"


def test_scatter_skips_none_baseline() -> None:
    # A None x-baseline (e.g. missing avoiding loss) is skipped, no crash.
    good = _macro_row(coefficient_str="2.0")
    bad = _macro_row(coefficient_str="4.0", mean_loss_concept_avoiding=3.0)
    bad["mean_loss_concept_avoiding"] = None
    points = _scatter_points([good, bad], scatter_baseline_x="baseline_loss")
    assert len(points) == 1
    assert points[0].coefficient_str == "2.0"


def test_scatter_skips_none_baseline_under_min() -> None:
    # Under "min", a None rich OR avoid makes the min undefined -> skip.
    bad = _macro_row(coefficient_str="4.0")
    bad["mean_loss_concept_rich"] = None
    points = _scatter_points([bad], scatter_baseline_x="min")
    assert points == []


def test_scatter_negative_mode_higher_is_better_via_signed_margin() -> None:
    # Negative mode: steerability = BPB(rich) - BPB(avoid). A strongly steered
    # negative point (rich >> avoid) must have a HIGHER y than a weakly steered
    # one. The sign-flip is delegated to _signed_margin; a flipped sign fails.
    strong = _macro_row(
        mode="negative",
        adversarial_strength=0.0,
        strength_index=0,
        mean_loss_concept_rich=9.0,
        mean_loss_concept_avoiding=1.0,
    )
    weak = _macro_row(
        mode="negative",
        adversarial_strength=1.0,
        strength_index=1,
        mean_loss_concept_rich=3.0,
        mean_loss_concept_avoiding=1.0,
    )
    points = {
        p.strength_index: p
        for p in _scatter_points([strong, weak], scatter_baseline_x="baseline_loss")
    }
    # higher-is-better: the strongly steered negative point dominates.
    assert points[0].y > points[1].y
    # Exact signed-margin values (rich - avoid in negative mode).
    assert points[0].y == 8.0
    assert points[1].y == 2.0


# ---------------------------------------------------------------------------
# The fix: every family is swept by COUNTER-STEERING strength, including
# single-base-config methods (LoRA / no-steering controls / encdec-ours).
# ---------------------------------------------------------------------------


def test_single_config_family_swept_by_counter_steering_strength() -> None:
    # A method with a SINGLE base config (coefficient "None", e.g. LoRA, a
    # no-steering control, or encdec-ours) must still yield one point PER
    # counter-steering strength -- the whole point of the fix. The old scatter
    # collapsed it to a single control dot.
    rows = _strength_sweep(method_name="lora_decoder_only", coefficient_str="None", n_strengths=4)
    reduced = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_with_feature_diffmean_targets(rows),
    )
    points = _scatter_points(reduced, scatter_baseline_x="baseline_loss")
    assert len(points) == 4
    assert sorted(p.adversarial_strength for p in points) == [0.0, 1.0, 2.0, 3.0]
    # All one family (one method, one coefficient), swept by strength.
    assert len({p.family_key() for p in points}) == 1


def test_scatter_series_key_separates_coefficients_within_a_family() -> None:
    # The trajectory-grouping key (mirroring the strength plot's _series_key)
    # includes coefficient_str, so two points of the SAME family but different
    # coefficients are SEPARATE trajectories even though they share a family_key.
    a = _scatter_points([_macro_row(coefficient_str="1.0")], scatter_baseline_x="baseline_loss")[0]
    b = _scatter_points([_macro_row(coefficient_str="2.0")], scatter_baseline_x="baseline_loss")[0]
    assert a.family_key() == b.family_key()  # same model/steering family + context
    assert _scatter_series_key(a) != _scatter_series_key(b)  # but distinct trajectories
    assert "1.0" in _scatter_series_key(a)
    assert "2.0" in _scatter_series_key(b)


def test_unreduced_multi_coefficient_keeps_separate_strength_sweeps() -> None:
    # If representative selection CANNOT reduce a multi-coefficient family (all
    # control margins null -> kept wholesale), the plot must keep each
    # coefficient as its OWN counter-steering sweep, NOT merge them into one
    # family whose line zigzags across coefficients (family_key excludes the
    # coefficient; _scatter_series_key includes it -- this is the guard).
    #
    # Exercises the REAL grouping helper the plot uses (_scatter_trajectories),
    # not a reconstruction in the test body.
    rows: list[dict[str, Any]] = []
    rows += _strength_sweep(coefficient_str="1.0", control_margin=None, base_avoid=3.0)
    rows += _strength_sweep(coefficient_str="2.0", control_margin=None, base_avoid=5.0)
    reduced = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_with_feature_diffmean_targets(rows),
    )
    points = _scatter_points(reduced, scatter_baseline_x="baseline_loss")

    # No usable control margin -> reduction kept BOTH coefficients wholesale.
    assert {p.coefficient_str for p in points} == {"1.0", "2.0"}
    # family_key alone WOULD merge them (the latent bug):
    assert len({p.family_key() for p in points}) == 1
    # The plot's grouping keeps them as two separate strength sweeps:
    trajectories = _scatter_trajectories(points)
    assert len(trajectories) == 2
    for trajectory in trajectories:
        assert len(trajectory) == 3  # one point per strength
        assert len({p.coefficient_str for p in trajectory}) == 1  # single coefficient


def test_scatter_trajectories_one_per_series_sorted_ascending_by_strength() -> None:
    # _scatter_trajectories groups into one connected sweep per series and orders
    # each ASCENDING by strength_index (the connecting-line draw order AND the
    # light->dark brightness ranking). Feed strengths OUT of order to prove the
    # helper sorts (not the input order).
    rows = [
        _macro_row(adversarial_strength=4.0, strength_index=3),
        _macro_row(adversarial_strength=0.0, strength_index=0),
        _macro_row(adversarial_strength=2.0, strength_index=2),
        _macro_row(adversarial_strength=1.0, strength_index=1),
    ]
    points = _scatter_points(rows, scatter_baseline_x="baseline_loss")
    [trajectory] = _scatter_trajectories(points)
    assert [p.strength_index for p in trajectory] == [0, 1, 2, 3]


def test_scatter_trajectories_stable_order_across_series() -> None:
    # Trajectories are returned in a deterministic (series-key-sorted) order so
    # the plot output is reproducible.
    rows: list[dict[str, Any]] = []
    rows += _strength_sweep(coefficient_str="2.0", control_margin=None, base_avoid=5.0)
    rows += _strength_sweep(coefficient_str="1.0", control_margin=None, base_avoid=3.0)
    reduced = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_with_feature_diffmean_targets(rows),
    )
    points = _scatter_points(reduced, scatter_baseline_x="baseline_loss")
    trajectories = _scatter_trajectories(points)
    first_coeff_per_trajectory = [t[0].coefficient_str for t in trajectories]
    # Sorted by series key -> coefficient "1.0" trajectory precedes "2.0".
    assert first_coeff_per_trajectory == ["1.0", "2.0"]


def test_plot_scatter_draws_one_monotone_line_per_coefficient(
    tmp_path: Path, monkeypatch: Any
) -> None:
    # END-TO-END pin of the grouping (line 741) AND sort (line 750) THROUGH the
    # real plot fn: an UNREDUCED two-coefficient family must yield TWO separate
    # connected lines (one per coefficient), each MONOTONE in x (ascending
    # strength) -- not one zigzagging merged line, and not a reversed sweep.
    # Mirrors the PR#503/#504 lesson: pin plot-path topology end-to-end.
    import matplotlib.axes

    captured: list[tuple[list[float], list[float]]] = []
    original_plot = matplotlib.axes.Axes.plot

    def _spy_plot(self: Any, *args: Any, **kwargs: Any) -> Any:
        # The scatter draws exactly one ax.plot(xs, ys, ...) per trajectory.
        if len(args) >= 2:
            captured.append((list(args[0]), list(args[1])))
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", _spy_plot)

    rows: list[dict[str, Any]] = []
    rows += _strength_sweep(coefficient_str="1.0", control_margin=None, base_avoid=3.0)
    rows += _strength_sweep(coefficient_str="2.0", control_margin=None, base_avoid=20.0)
    out_path = tmp_path / "positive" / "steerability_vs_baseline.png"
    result = _plot_steerability_vs_baseline_scatter(
        rows,
        out_path,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_targets,
    )
    assert result == out_path

    # Exactly TWO connected trajectories were drawn (not one merged line).
    assert len(captured) == 2
    for xs, _ys in captured:
        # Each trajectory is monotone-ascending in x: catches BOTH a merged
        # zigzag (line 741) and a reversed strength sort (line 750).
        assert xs == sorted(xs)
        assert len(set(xs)) == len(xs)
    # The two trajectories occupy disjoint x-bands (coeff 1.0 -> ~3-5, 2.0 -> ~20-22).
    band_maxes = sorted(max(xs) for xs, _ in captured)
    band_mins = sorted(min(xs) for xs, _ in captured)
    assert band_maxes[0] < band_mins[1]  # no interleaving across coefficients


def test_multi_coefficient_family_reduced_to_representative_then_swept() -> None:
    # diffmean has a base-coefficient grid; the scatter reduces to the
    # representative coefficient closest to the fair Feature-DiffMean target, THEN
    # sweeps counter-steering strength -- NOT one point per coefficient.
    rows: list[dict[str, Any]] = []
    # coefficient "4.0" has the higher control margin -> it is representative.
    rows += _strength_sweep(coefficient_str="1.0", control_margin=0.5, base_avoid=1.5)
    rows += _strength_sweep(coefficient_str="4.0", control_margin=3.0, base_avoid=4.0)
    reduced = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_with_feature_diffmean_targets(rows),
    )
    points = _scatter_points(reduced, scatter_baseline_x="baseline_loss")
    # Only the representative coefficient survives, swept over its 3 strengths.
    assert len(points) == 3
    assert {p.coefficient_str for p in points} == {"4.0"}
    assert sorted(p.adversarial_strength for p in points) == [0.0, 1.0, 2.0]


# ---------------------------------------------------------------------------
# family_key context separation: each context_signal_mode is its OWN connected
# counter-steering sweep with its own light->dark brightness ranking.
# ---------------------------------------------------------------------------


def _two_context_strength_sweeps() -> list[dict[str, Any]]:
    """1 method x 1 mode x 2 contexts x 1 coefficient x 3 strengths.

    A single (alpha, mode, method) subset can carry rows under more than one
    context_signal_mode (``fixed_context`` for prompt-token attacks vs
    ``not_applicable`` for additive-direction rows); both contexts must form
    SEPARATE families.
    """

    rows: list[dict[str, Any]] = []
    for ctx_idx, context_signal_mode in enumerate(("fixed_context", "not_applicable")):
        rows += _strength_sweep(
            method_name="diffmean_decoder_only",
            coefficient_str="1.0",
            context_signal_mode=context_signal_mode,
            n_strengths=3,
            # Offset the two contexts so their points are distinguishable.
            base_rich=1.0 + 10.0 * ctx_idx,
            base_avoid=3.0 + 10.0 * ctx_idx,
        )
    return rows


def test_scatter_point_family_key_includes_context_signal_mode() -> None:
    # Two points identical except for context_signal_mode must NOT share a
    # family_key -- otherwise the two contexts' sweeps merge.
    fixed = _scatter_points(
        [_macro_row(context_signal_mode="fixed_context")],
        scatter_baseline_x="baseline_loss",
    )[0]
    not_applicable_point = _scatter_points(
        [_macro_row(context_signal_mode="not_applicable")],
        scatter_baseline_x="baseline_loss",
    )[0]
    assert fixed.context_signal_mode == "fixed_context"
    assert not_applicable_point.context_signal_mode == "not_applicable"
    assert fixed.family_key() != not_applicable_point.family_key()
    assert "fixed_context" in fixed.family_key()
    assert "not_applicable" in not_applicable_point.family_key()


def test_scatter_two_contexts_form_two_strength_swept_families() -> None:
    # Replicates the brightness-rank grouping inside the plot: points grouped by
    # family_key, then ranked 0..n-1 within each family by strength. With context
    # in the key there are TWO families of 3 strengths (ranks 0..2 each).
    rows = _two_context_strength_sweeps()
    points = _scatter_points(rows, scatter_baseline_x="baseline_loss")
    assert len(points) == 6

    by_family: dict[tuple[str, ...], list[_ScatterPoint]] = {}
    for point in points:
        by_family.setdefault(point.family_key(), []).append(point)

    # TWO separate families (one per context), each with 3 strength points.
    assert len(by_family) == 2
    contexts_seen = set()
    for family_points in by_family.values():
        assert len(family_points) == 3
        family_contexts = {p.context_signal_mode for p in family_points}
        assert len(family_contexts) == 1
        contexts_seen |= family_contexts
        # One point per counter-steering strength, distinct strength indices.
        assert sorted(p.strength_index for p in family_points) == [0, 1, 2]
    assert contexts_seen == {"fixed_context", "not_applicable"}


def test_scatter_line_sorted_by_strength_is_monotone_per_context() -> None:
    # Each family's connecting line is drawn in counter-steering-strength order.
    # Within one context the x baseline (mean_loss_concept_avoiding) increases
    # with strength, so a per-context sweep is monotone in x; a merged-context
    # family would interleave the contexts and zigzag. Exercises the REAL plot
    # grouping/sort helper (_scatter_trajectories), not a reconstruction.
    rows = _two_context_strength_sweeps()
    points = _scatter_points(rows, scatter_baseline_x="baseline_loss")

    trajectories = _scatter_trajectories(points)
    assert len(trajectories) == 2  # one per context

    for trajectory in trajectories:
        xs = [point.x for point in trajectory]
        # Strictly increasing within a single context -> no cross-context zigzag.
        assert xs == sorted(xs)
        assert len(set(xs)) == len(xs)
        # Single context per trajectory.
        assert len({p.context_signal_mode for p in trajectory}) == 1


# ---------------------------------------------------------------------------
# _plot_steerability_vs_baseline_scatter (PNG emission)
# ---------------------------------------------------------------------------


def test_plot_scatter_emits_png(tmp_path: Path) -> None:
    rows = _strength_sweep(mode="positive")
    out_path = tmp_path / "positive" / "steerability_vs_baseline.png"
    result = _plot_steerability_vs_baseline_scatter(
        rows,
        out_path,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_targets,
    )
    assert result == out_path
    assert out_path.exists()


def test_plot_scatter_emits_png_for_single_config_family(tmp_path: Path) -> None:
    # The fix end-to-end: a single-base-config family (coefficient "None") still
    # produces a scatter PNG (its counter-steering sweep is plotted).
    rows = _strength_sweep(method_name="lora_decoder_only", coefficient_str="None", n_strengths=4)
    out_path = tmp_path / "positive" / "steerability_vs_baseline.png"
    result = _plot_steerability_vs_baseline_scatter(
        rows,
        out_path,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_targets,
    )
    assert result == out_path
    assert out_path.exists()


def test_plot_scatter_returns_none_when_no_points(tmp_path: Path) -> None:
    # All rows skipped (None coefficient) -> no PNG, return None.
    rows = [_macro_row(coefficient_str=None)]
    out_path = tmp_path / "positive" / "steerability_vs_baseline.png"
    result = _plot_steerability_vs_baseline_scatter(
        rows,
        out_path,
        scatter_baseline_x="baseline_loss",
        mode="positive",
        coefficient_selector=_select_with_feature_diffmean_targets,
    )
    assert result is None
    assert not out_path.exists()


def test_plot_scatter_emits_png_under_min_baseline(tmp_path: Path) -> None:
    """`min` is exercised through the full PNG-emitting path, not just the
    point-construction unit (closes a coverage gap)."""
    rows = _strength_sweep(mode="negative")
    out_path = tmp_path / "negative" / "steerability_vs_baseline.png"
    result = _plot_steerability_vs_baseline_scatter(
        rows,
        out_path,
        scatter_baseline_x="min",
        mode="negative",
        coefficient_selector=_select_with_feature_diffmean_targets,
    )
    assert result == out_path
    assert out_path.exists()


def test_scatter_min_value_is_mode_independent() -> None:
    """`min` x is the symmetric smaller-of-the-two-classes -- identical value in
    either mode for the same loss pair (only `baseline_loss` is mode-dependent)."""
    pos = _macro_row(mode="positive", mean_loss_concept_rich=1.5, mean_loss_concept_avoiding=4.0)
    neg = _macro_row(mode="negative", mean_loss_concept_rich=1.5, mean_loss_concept_avoiding=4.0)
    [pos_point] = _scatter_points([pos], scatter_baseline_x="min")
    [neg_point] = _scatter_points([neg], scatter_baseline_x="min")
    assert pos_point.x == 1.5
    assert neg_point.x == 1.5  # same smaller side regardless of mode


def test_style_legend_omits_not_applicable_context() -> None:
    """``not_applicable`` is the additive rows' context_signal_mode (no prompt-token
    context); it is noise in the context-linestyle legend and must be omitted. Real
    context modes (e.g. ``fixed_context``) are still shown."""
    fig = agg_mod.plt.figure()
    try:
        _add_style_legend(
            fig,
            rows=[
                {"context_signal_mode": "fixed_context"},
                {"context_signal_mode": "not_applicable"},
            ],
            color_map={"decoder-only / diffmean": "tab:orange"},
        )
        labels = {text.get_text() for text in fig.legends[0].get_texts()}
    finally:
        agg_mod.plt.close(fig)

    assert "not_applicable" not in labels
    assert "fixed_context" in labels  # real context modes are still listed


def test_context_linestyle_not_applicable_is_solid() -> None:
    """The additive sweeps carry ``not_applicable`` context (omitted from the
    legend); they must draw as a SOLID line like the real ``fixed_context`` sweeps,
    not a dash-dot that has no legend key to explain it."""
    assert _context_linestyle("not_applicable") == "-"
    assert _context_linestyle("fixed_context") == "-"


def test_plot_bpb_adv_summaries_rejects_empty_plot_methods(tmp_path: Path) -> None:
    """An empty ``plot_methods`` list keeps NO methods -- it would clear every
    prior plot and emit nothing. Validate BEFORE clearing: raise loudly and leave
    existing plots untouched (None means 'plot all'; [] is almost surely a bug)."""
    plots_dir = tmp_path / "plots"
    plots_dir.mkdir()
    sentinel = plots_dir / "pre_existing.png"
    sentinel.write_bytes(b"keep me")
    macro = pa.table({"method_name": ["diffmean_decoder_only"]})

    with pytest.raises(AxBenchBpbAdvError, match="plot_methods"):
        plot_bpb_adv_summaries(macro, plots_dir, plot_methods=[])

    # The guard must fire before _clear_previous_plot_outputs wipes anything.
    assert sentinel.exists()


# ---------------------------------------------------------------------------
# Issue #635: every general_val_bpb drawn by ONE plot_* invocation must share a
# single held-out corpus signature -- BPB values scored on different corpora are
# not comparable on the shared general-val axes.
# ---------------------------------------------------------------------------


def test_plot_bpb_adv_summaries_rejects_mixed_corpus(tmp_path: Path) -> None:
    """Two plotted method families scored on DIFFERENT general-val corpora must
    raise, not silently draw incomparable general_val_bpb values on shared axes.

    The fair feature-DiffMean encdec target rows (null general_val) are included
    so representative-coefficient selection succeeds: the ONLY possible raise is
    the corpus check."""
    rows = _with_feature_diffmean_targets(
        [
            _macro_row(
                method_name="diffmean_decoder_only",
                baseline_primary_margin=2.0,
                general_val_bpb=3.0,
                general_val_corpus="nd=4;nb=2277;h=aa",
            ),
            _macro_row(
                method_name="linear_probe_decoder_only",
                baseline_primary_margin=2.0,
                general_val_bpb=3.2,
                general_val_corpus="nd=11;nb=6335;h=bb",
            ),
        ]
    )
    table = pa.Table.from_pylist(rows, schema=MACRO_AGG_SCHEMA)
    with pytest.raises(AxBenchBpbAdvError, match="mixes general-val corpus signatures"):
        plot_bpb_adv_summaries(table, tmp_path / "plots")


def _roundtrip_macro_row(
    *,
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    coefficient_str: str | None = "2.0",
    general_val_bpb: float | None = None,
    general_val_corpus: str | None = None,
) -> dict[str, Any]:
    """One ROUNDTRIP_MACRO_AGG_SCHEMA control row (mirrors the strength-plot
    fixtures): ``is_control=True`` plus a non-null ``primary_roundtrip_y`` so the
    fair feature-DiffMean representative selection can score every family."""
    return {
        "mode": "positive",
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": "additive_direction",
        "adversarial_strength": 0.0,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 3.0,
        "mean_roundtrip_pre_act": 0.5,
        "mean_roundtrip_post_act": 3.5,
        "mean_roundtrip_percentile": 99.0,
        "primary_roundtrip_y": 4.0,
        "mean_control_roundtrip_pre_act": 0.5,
        "mean_control_roundtrip_post_act": 0.5,
        "mean_control_roundtrip_percentile": 50.0,
        "control_primary_roundtrip_y": 0.0,
        "general_val_bpb": general_val_bpb,
        "general_val_corpus": general_val_corpus,
        "n_concepts": 1,
        "n_samples": 1,
        "n_generated_tokens": 8,
        "n_control_generated_tokens": 8,
        "max_tokens_rate": 0.0,
        "control_max_tokens_rate": 0.0,
        "is_control": True,
    }


def test_plot_roundtrip_adv_summaries_rejects_mixed_corpus(tmp_path: Path) -> None:
    """Same corpus-mixing guard on the roundtrip entry point (its scatter can
    draw general_val_bpb as the x baseline). The fair feature-DiffMean encdec
    target row (null general_val, same adversarial_method/context) is included
    so the corpus check is the only possible raise."""
    rows = [
        _roundtrip_macro_row(
            method_name="diffmean_decoder_only",
            general_val_bpb=5.0,
            general_val_corpus="nd=4;nb=2277;h=aa",
        ),
        _roundtrip_macro_row(
            method_name="linear_probe_decoder_only",
            general_val_bpb=5.2,
            general_val_corpus="nd=11;nb=6335;h=bb",
        ),
        _roundtrip_macro_row(
            method_name="feature_diffmean_encdec",
            target_model="encdec",
            coefficient_str="None",
        ),
    ]
    table = pa.Table.from_pylist(rows, schema=ROUNDTRIP_MACRO_AGG_SCHEMA)
    with pytest.raises(AxBenchBpbAdvError, match="mixes general-val corpus signatures"):
        plot_roundtrip_adv_summaries(table, tmp_path / "plots")
