"""Unit tests for representative-coefficient selection.

``_select_representative_coefficient`` picks, per steering family, the
``coefficient_str`` whose no-adversary control score is closest to the fair
encoder-decoder Feature-DiffMean baseline score for the same mode/adversary/
context tuple. Exact distance ties prefer the higher score, then smaller
``abs(float(coeff))``, then lexical coefficient. A ``None`` coefficient selects
``None``. The companion filter keeps the full adversarial-strength sweep of the
chosen coefficient per family.
"""

from __future__ import annotations

from typing import Any

import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    _filter_rows_to_representative_coefficient,
    _select_representative_coefficient,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError


def _macro_row(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    coefficient_str: str | None = "1.0",
    adversarial_method: str = "faster_gcg",
    adversarial_strength: float = 0.0,
    strength_unit: str = "optimized_tokens",
    strength_index: int = 0,
    context_signal_mode: str = "fixed_context",
    primary_margin_y: float = 0.0,
    baseline_primary_margin: float | None = None,
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": adversarial_method,
        "adversarial_strength": adversarial_strength,
        "strength_unit": strength_unit,
        "strength_index": strength_index,
        "context_signal_mode": context_signal_mode,
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
    coefficient_str: str,
    *,
    control_margin: float,
    n_strengths: int = 3,
) -> list[dict[str, Any]]:
    """One coefficient family: a k=0 control + an adversarial-strength sweep.

    The control row carries ``baseline_primary_margin == control_margin``; the
    adversarial rows carry the SAME (adversary-agnostic) baseline but degraded
    ``primary_margin_y`` so we can verify the full sweep survives filtering.
    """

    rows = [
        _macro_row(
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
                coefficient_str=coefficient_str,
                adversarial_strength=float(strength_index),
                strength_index=strength_index,
                primary_margin_y=control_margin - float(strength_index),
                baseline_primary_margin=control_margin,
            )
        )
    return rows


def _feature_diffmean_target(
    *,
    score: float,
    mode: str = "positive",
    roundtrip_score: float | None = None,
) -> dict[str, Any]:
    method = "feature_diffmean_encdec" if mode == "positive" else "neg_feature_diffmean_encdec"
    row = _macro_row(
        mode=mode,
        method_name=method,
        target_model="encdec",
        coefficient_str="None",
        adversarial_strength=0.0,
        strength_index=0,
        primary_margin_y=score,
        baseline_primary_margin=score,
    )
    if roundtrip_score is not None:
        row["primary_roundtrip_y"] = roundtrip_score
    return row


def _baseline_rows_with_targets(
    rows: list[dict[str, Any]],
    *,
    positive_score: float = 2.0,
    negative_score: float = 2.0,
    positive_roundtrip_score: float | None = None,
    negative_roundtrip_score: float | None = None,
) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    modes = {str(row["mode"]) for row in rows}
    if "positive" in modes:
        targets.append(
            _feature_diffmean_target(
                score=positive_score,
                mode="positive",
                roundtrip_score=positive_roundtrip_score,
            )
        )
    if "negative" in modes:
        targets.append(
            _feature_diffmean_target(
                score=negative_score,
                mode="negative",
                roundtrip_score=negative_roundtrip_score,
            )
        )
    return [*rows, *targets]


# ---------------------------------------------------------------------------
# _select_representative_coefficient
# ---------------------------------------------------------------------------


def test_select_picks_nearest_feature_diffmean_score() -> None:
    rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=2.0),
        *_coeff_family("8.0", control_margin=1.0),
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=1.8),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


def test_select_uses_only_control_rows_not_adversarial_rows() -> None:
    # Coefficient "8.0" has a much higher margin at an ADVERSARIAL strength, but a
    # weaker control margin. Selection must read the control rows only.
    rows = [
        *_coeff_family("4.0", control_margin=2.0),
        # "8.0": weak control, but a spuriously high adversarial-strength margin.
        _macro_row(
            coefficient_str="8.0",
            adversarial_strength=0.0,
            strength_index=0,
            primary_margin_y=0.1,
            baseline_primary_margin=0.1,
        ),
        _macro_row(
            coefficient_str="8.0",
            adversarial_strength=1.0,
            strength_index=1,
            primary_margin_y=99.0,
            baseline_primary_margin=0.1,
        ),
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=2.2),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


def test_select_closest_score_tie_prefers_higher_score() -> None:
    rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=1.0),
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=0.75),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


def test_select_score_tie_breaks_to_smallest_abs_coefficient() -> None:
    # All coefficients tie on distance and score; smaller |coeff| wins.
    rows = [
        *_coeff_family("8.0", control_margin=2.0),
        *_coeff_family("2.0", control_margin=2.0),
        *_coeff_family("-3.0", control_margin=2.0),
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=2.0),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    # |2.0| < |-3.0| < |8.0| -> "2.0" wins the tie.
    assert selected[key] == "2.0"


def test_select_tie_break_is_deterministic_run_twice() -> None:
    rows = [
        *_coeff_family("8.0", control_margin=2.0),
        *_coeff_family("2.0", control_margin=2.0),
        *_coeff_family("-2.0", control_margin=2.0),
    ]
    baseline_rows = _baseline_rows_with_targets(rows, positive_score=2.0)
    first = _select_representative_coefficient(rows, baseline_rows=baseline_rows)
    second = _select_representative_coefficient(
        list(reversed(rows)),
        baseline_rows=list(reversed(baseline_rows)),
    )
    # abs(2.0) == abs(-2.0); lexicographic string break -> "-2.0" < "2.0".
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert first[key] == second[key] == "-2.0"


def test_select_per_config_independent_selection() -> None:
    # Different mode AND different method pick independently.
    rows = [
        # positive / diffmean: "4.0" best
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=3.0),
        # negative / linear_probe: "1.0" best (distinct family)
        *[
            row | {"mode": "negative", "method_name": "linear_probe"}
            for row in _coeff_family("1.0", control_margin=5.0)
        ],
        *[
            row | {"mode": "negative", "method_name": "linear_probe"}
            for row in _coeff_family("4.0", control_margin=2.0)
        ],
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(
            rows,
            positive_score=2.6,
            negative_score=4.7,
        ),
    )
    pos_key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    neg_key = (
        "negative",
        "linear_probe",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[pos_key] == "4.0"
    assert selected[neg_key] == "1.0"


def test_select_none_coefficient_selects_none() -> None:
    rows = _coeff_family("1.0", control_margin=2.0)
    for row in rows:
        row["coefficient_str"] = None
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=2.0),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] is None


def test_select_skips_coefficients_with_null_baseline() -> None:
    # A coefficient family whose control baseline is entirely None must not be
    # selected; the one with a real control margin wins.
    rows = [
        # "9.0": no control row, baseline None everywhere.
        _macro_row(
            coefficient_str="9.0",
            adversarial_strength=1.0,
            strength_index=1,
            primary_margin_y=50.0,
            baseline_primary_margin=None,
        ),
        *_coeff_family("4.0", control_margin=1.0),
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=1.2),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


def test_select_group_with_only_null_baseline_yields_none() -> None:
    rows = [
        _macro_row(
            coefficient_str="9.0",
            adversarial_strength=1.0,
            strength_index=1,
            primary_margin_y=50.0,
            baseline_primary_margin=None,
        ),
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=1.2),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] is None


def test_select_ignores_strength_zero_row_marked_not_control() -> None:
    # F6: a strength-0 row whose persisted is_control flag is False (a mutated-
    # prompt surrogate) must NOT be treated as a control by rep-selection, even
    # though adversarial_strength == 0. Its spuriously high margin must lose to
    # the genuine control coefficient.
    rows = [
        # Genuine control at "4.0": strength 0, is_control True, margin 2.0.
        *_coeff_family("4.0", control_margin=2.0),
        # Strength-0 but is_control=False at "8.0" with a huge margin: must be
        # excluded from control-margin selection.
        _macro_row(
            coefficient_str="8.0",
            adversarial_strength=0.0,
            strength_index=0,
            primary_margin_y=99.0,
            baseline_primary_margin=99.0,
        )
        | {"is_control": False},
    ]
    selected = _select_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=2.2),
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


# ---------------------------------------------------------------------------
# _filter_rows_to_representative_coefficient
# ---------------------------------------------------------------------------


def test_filter_keeps_full_strength_sweep_of_representative_coefficient() -> None:
    rows = [
        *_coeff_family("1.0", control_margin=0.5, n_strengths=4),
        *_coeff_family("4.0", control_margin=2.0, n_strengths=4),
    ]
    filtered = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=1.8),
    )
    # Only the representative ("4.0") family survives, and ALL of its strengths.
    assert {row["coefficient_str"] for row in filtered} == {"4.0"}
    strength_indices = sorted(row["strength_index"] for row in filtered)
    assert strength_indices == [0, 1, 2, 3]
    assert len(filtered) == 4


def test_filter_keeps_independent_representative_per_config() -> None:
    rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=3.0),
        *[row | {"mode": "negative"} for row in _coeff_family("1.0", control_margin=5.0)],
        *[row | {"mode": "negative"} for row in _coeff_family("4.0", control_margin=2.0)],
    ]
    filtered = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(
            rows,
            positive_score=2.6,
            negative_score=4.7,
        ),
    )
    pos = {row["coefficient_str"] for row in filtered if row["mode"] == "positive"}
    neg = {row["coefficient_str"] for row in filtered if row["mode"] == "negative"}
    assert pos == {"4.0"}
    assert neg == {"1.0"}


def test_filter_keeps_none_coefficient_rows() -> None:
    rows = _coeff_family("1.0", control_margin=2.0)
    for row in rows:
        row["coefficient_str"] = None
    filtered = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=2.0),
    )
    # None-coefficient family selects None and is kept wholesale.
    assert len(filtered) == len(rows)
    assert all(row["coefficient_str"] is None for row in filtered)


def test_filter_keeps_null_baseline_group_when_representative_is_none() -> None:
    # A group whose representative resolves to None (no usable control margin)
    # must still be kept (no crash, not silently dropped).
    rows = [
        _macro_row(
            coefficient_str="9.0",
            adversarial_strength=1.0,
            strength_index=1,
            primary_margin_y=50.0,
            baseline_primary_margin=None,
        ),
    ]
    filtered = _filter_rows_to_representative_coefficient(
        rows,
        baseline_rows=_baseline_rows_with_targets(rows, positive_score=1.2),
    )
    assert filtered == rows


# ---------------------------------------------------------------------------
# Feature-DiffMean target behavior
# ---------------------------------------------------------------------------


def test_closest_to_feature_diffmean_selects_nearest_control_score() -> None:
    candidate_rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=1.1),
        *_coeff_family("8.0", control_margin=2.0),
    ]
    selected = _select_representative_coefficient(
        candidate_rows,
        baseline_rows=[*candidate_rows, _feature_diffmean_target(score=1.0)],
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


def test_closest_to_feature_diffmean_tie_prefers_higher_score() -> None:
    candidate_rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=1.0),
    ]
    selected = _select_representative_coefficient(
        candidate_rows,
        baseline_rows=[*candidate_rows, _feature_diffmean_target(score=0.75)],
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "4.0"


def test_closest_to_feature_diffmean_uses_hidden_baseline_rows() -> None:
    visible_rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=1.0),
    ]
    baseline_rows = [*visible_rows, _feature_diffmean_target(score=0.6)]
    filtered = _filter_rows_to_representative_coefficient(
        visible_rows,
        baseline_rows=baseline_rows,
    )
    assert {row["coefficient_str"] for row in filtered} == {"1.0"}


def test_closest_to_feature_diffmean_can_use_roundtrip_metric() -> None:
    candidate_rows = [
        row | {"primary_roundtrip_y": 70.0} for row in _coeff_family("1.0", control_margin=0.1)
    ]
    candidate_rows.extend(
        row | {"primary_roundtrip_y": 90.0} for row in _coeff_family("4.0", control_margin=0.2)
    )
    target = _feature_diffmean_target(score=0.0) | {"primary_roundtrip_y": 72.0}
    selected = _select_representative_coefficient(
        candidate_rows,
        baseline_metric="primary_roundtrip_y",
        baseline_rows=[*candidate_rows, target],
    )
    key = (
        "positive",
        "diffmean_decoder_only",
        "decoder_only",
        "faster_gcg",
        "fixed_context",
    )
    assert selected[key] == "1.0"


def test_closest_to_feature_diffmean_missing_target_raises() -> None:
    rows = [
        *_coeff_family("1.0", control_margin=0.5),
        *_coeff_family("4.0", control_margin=1.0),
    ]
    with pytest.raises(AxBenchBpbAdvError, match="missing fair feature-DiffMean"):
        _select_representative_coefficient(
            rows,
            baseline_rows=rows,
        )
