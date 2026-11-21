from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib.axes
import pyarrow as pa
import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    STEERING_STRENGTH_FAMILY_PLOT_NAME,
    _exclude_strength_outliers,
    _strength_family_points,
    plot_steering_strength_families,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    ROUNDTRIP_MACRO_AGG_SCHEMA,
)


def _bpb_row(
    *,
    mode: str = "positive",
    method_name: str = "diffmean_decoder_only",
    target_model: str = "decoder_only",
    coefficient_str: str | None = "1.0",
    adversarial_strength: float = 0.0,
    is_control: bool = True,
    primary_margin_y: float = 2.0,
    general_val_bpb: float | None = 3.0,
    general_val_corpus: str | None = "nd=1;nb=8;h=deadbeefdeadbeef",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": "additive_direction",
        "adversarial_strength": adversarial_strength,
        "strength_unit": "counter_direction_magnitude",
        "strength_index": 0,
        "context_signal_mode": "not_applicable",
        "mean_loss_concept_rich": 1.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": primary_margin_y,
        "baseline_primary_margin": primary_margin_y if is_control else None,
        "general_val_bpb": general_val_bpb,
        "general_val_corpus": general_val_corpus,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": is_control,
    }


def _roundtrip_row(
    *,
    mode: str = "positive",
    method_name: str = "linear_probe_decoder_only",
    coefficient_str: str | None = "2.0",
    primary_roundtrip_y: float = 4.0,
    general_val_bpb: float | None = 5.0,
    general_val_corpus: str | None = "nd=1;nb=8;h=deadbeefdeadbeef",
) -> dict[str, Any]:
    return {
        "mode": mode,
        "method_name": method_name,
        "target_model": "decoder_only",
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
        "primary_roundtrip_y": primary_roundtrip_y,
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


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f))


def test_strength_points_average_duplicate_control_rows() -> None:
    # Non-null rows (coefficient "1.0") share ONE corpus literal -- issue #635
    # forbids averaging general_val_bpb across mismatched held-out corpora.
    shared_corpus = "nd=1;nb=8;h=deadbeefdeadbeef"
    rows = [
        _bpb_row(
            coefficient_str="1.0",
            primary_margin_y=2.0,
            general_val_bpb=10.0,
            general_val_corpus=shared_corpus,
        ),
        _bpb_row(
            coefficient_str="1.0",
            primary_margin_y=4.0,
            general_val_bpb=14.0,
            general_val_corpus=shared_corpus,
        ),
        _bpb_row(
            coefficient_str="2.0",
            primary_margin_y=6.0,
            general_val_bpb=None,
            general_val_corpus=None,
        ),
        _bpb_row(
            coefficient_str="2.0",
            primary_margin_y=8.0,
            general_val_bpb=None,
            general_val_corpus=None,
        ),
    ]
    points = _strength_family_points(rows, metric_column="primary_margin_y")
    assert [(p.coefficient_str, p.metric_y, p.general_val_bpb, p.n_source_rows) for p in points] == [
        ("1.0", 3.0, 12.0, 2),
        ("2.0", 7.0, None, 2),
    ]


def test_strength_points_reject_partial_null_group() -> None:
    rows = [
        _bpb_row(
            coefficient_str="1.0",
            general_val_bpb=10.0,
            general_val_corpus="nd=2;nb=10;h=aa",
        ),
        _bpb_row(coefficient_str="1.0", general_val_bpb=None, general_val_corpus=None),
    ]
    with pytest.raises(AxBenchBpbAdvError, match="mixes null and non-null general_val_bpb"):
        _strength_family_points(rows, metric_column="primary_margin_y")


def test_bpb_strength_plot_writes_mode_png_index_and_legend(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            _bpb_row(coefficient_str="1.0", primary_margin_y=2.0),
            _bpb_row(coefficient_str="2.0", primary_margin_y=3.0),
            _bpb_row(
                method_name="control_decoder_only",
                coefficient_str=None,
                primary_margin_y=0.5,
            ),
            _bpb_row(
                mode="negative",
                method_name="neg_diffmean_decoder_only",
                coefficient_str="1.0",
                primary_margin_y=1.25,
            ),
            _bpb_row(
                coefficient_str="1.0",
                adversarial_strength=2.0,
                is_control=False,
                primary_margin_y=-99.0,
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )
    plots_dir = tmp_path / "strength"

    written = plot_steering_strength_families(
        table,
        plots_dir,
        metric_column="primary_margin_y",
        metric_label="primary margin",
    )

    assert plots_dir / "positive" / STEERING_STRENGTH_FAMILY_PLOT_NAME in written
    assert plots_dir / "negative" / STEERING_STRENGTH_FAMILY_PLOT_NAME in written
    assert (plots_dir / "positive" / STEERING_STRENGTH_FAMILY_PLOT_NAME).exists()
    assert (plots_dir / "negative" / STEERING_STRENGTH_FAMILY_PLOT_NAME).exists()
    index = _csv_rows(plots_dir / "plot_index.csv")
    assert {row["mode"] for row in index} == {"positive", "negative"}
    assert all(row["plot_name"] == STEERING_STRENGTH_FAMILY_PLOT_NAME for row in index)
    legend = _csv_rows(plots_dir / "plot_series_legend.csv")
    assert {row["method_name"] for row in legend} == {
        "diffmean_decoder_only",
        "control_decoder_only",
        "neg_diffmean_decoder_only",
    }
    control = next(row for row in legend if row["method_name"] == "control_decoder_only")
    assert control["coefficient_min"] == "None"
    assert control["coefficient_max"] == "None"


def test_strength_plot_emits_frontier_panel_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    titles: list[str] = []
    original_set_title = matplotlib.axes.Axes.set_title

    def _spy_set_title(self: Any, label: str, *args: Any, **kwargs: Any) -> Any:
        titles.append(str(label))
        return original_set_title(self, label, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "set_title", _spy_set_title)
    table = pa.Table.from_pylist(
        [
            _bpb_row(coefficient_str="1.0", primary_margin_y=2.0),
            _bpb_row(coefficient_str="2.0", primary_margin_y=3.0, general_val_bpb=4.0),
        ],
        schema=MACRO_AGG_SCHEMA,
    )

    plot_steering_strength_families(
        table,
        tmp_path / "plots",
        metric_column="primary_margin_y",
        metric_label="primary margin",
    )

    assert "steerability frontier" in titles
    assert "steering metric vs base strength" not in titles


def test_strength_plot_legend_marks_roofline_variants_with_stars(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            _bpb_row(
                method_name="feature_diffmean_encdec",
                target_model="encdec",
                coefficient_str=None,
            ),
            _bpb_row(
                method_name="feature_diffmean_roofline_encdec",
                target_model="encdec",
                coefficient_str=None,
            ),
            _bpb_row(
                mode="negative",
                method_name="neg_feature_diffmean_encdec",
                target_model="encdec",
                coefficient_str=None,
            ),
            _bpb_row(
                mode="negative",
                method_name="neg_feature_diffmean_roofline_encdec",
                target_model="encdec",
                coefficient_str=None,
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )

    plot_steering_strength_families(
        table,
        tmp_path / "plots",
        metric_column="primary_margin_y",
        metric_label="primary margin",
    )

    legend = _csv_rows(tmp_path / "plots" / "plot_series_legend.csv")
    marker_by_method = {row["method_name"]: row["marker"] for row in legend}
    assert marker_by_method == {
        "feature_diffmean_encdec": "o",
        "feature_diffmean_roofline_encdec": "*",
        "neg_feature_diffmean_encdec": "o",
        "neg_feature_diffmean_roofline_encdec": "*",
    }


def test_strength_plot_legend_swaps_feature_diffmean_and_gcg_colors(
    tmp_path: Path,
) -> None:
    table = pa.Table.from_pylist(
        [
            _bpb_row(
                method_name="feature_diffmean_encdec",
                target_model="encdec",
                coefficient_str=None,
            ),
            _bpb_row(
                method_name="feature_diffmean_roofline_encdec",
                target_model="encdec",
                coefficient_str=None,
            ),
            _bpb_row(
                method_name="ours_gcg_encdec",
                target_model="encdec",
                coefficient_str="0.0",
            ),
            _bpb_row(
                method_name="ours_gcg_roofline_encdec",
                target_model="encdec",
                coefficient_str="0.0",
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )

    plot_steering_strength_families(
        table,
        tmp_path / "plots",
        metric_column="primary_margin_y",
        metric_label="primary margin",
    )

    legend = _csv_rows(tmp_path / "plots" / "plot_series_legend.csv")
    color_by_method = {row["method_name"]: row["color"] for row in legend}
    assert color_by_method == {
        "feature_diffmean_encdec": "#9467bd",
        "feature_diffmean_roofline_encdec": "#9467bd",
        "ours_gcg_encdec": "#d62728",
        "ours_gcg_roofline_encdec": "#d62728",
    }


def test_strength_plot_raises_when_a_family_lacks_frontier_x(tmp_path: Path) -> None:
    # A family WITH a general_val_bpb x plus a (non-roofline) family WITHOUT one:
    # the frontier cannot place the second on the capability axis, so it raises
    # naming the offender instead of silently omitting it from the comparison.
    table = pa.Table.from_pylist(
        [
            _bpb_row(
                method_name="feature_diffmean_encdec",
                target_model="encdec",
                coefficient_str=None,
                general_val_bpb=3.0,
                general_val_corpus="nd=2;nb=10;h=aa",
            ),
            _bpb_row(
                method_name="diffmean_decoder_only",
                target_model="decoder_only",
                coefficient_str="1.0",
                general_val_bpb=None,
                general_val_corpus=None,
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="every plotted family must have a") as exc:
        plot_steering_strength_families(
            table, tmp_path / "p", metric_column="primary_margin_y", metric_label="m"
        )
    assert "diffmean_decoder_only" in str(exc.value)


def test_strength_plot_raises_for_roofline_without_frontier_x(tmp_path: Path) -> None:
    # Roofline is just one case of the general "every plotted family needs an x"
    # rule: a roofline family with no general_val_bpb (its general_val stage was
    # not run) raises like any other, not with a stale "not implemented" message.
    table = pa.Table.from_pylist(
        [
            _bpb_row(
                method_name="feature_diffmean_roofline_encdec",
                target_model="encdec",
                coefficient_str=None,
                general_val_bpb=None,
                general_val_corpus=None,
                primary_margin_y=3.0,
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="every plotted family must have a") as exc:
        plot_steering_strength_families(
            table,
            tmp_path / "plots",
            metric_column="primary_margin_y",
            metric_label="primary margin",
        )
    assert "feature_diffmean_roofline_encdec" in str(exc.value)


def test_strength_plot_excludes_frontier_outliers() -> None:
    rows = [
        _bpb_row(
            method_name="feature_diffmean_encdec",
            coefficient_str=None,
            general_val_bpb=4.42,
            primary_margin_y=-2.1,
        ),
        _bpb_row(
            method_name="ours_gcg_encdec",
            coefficient_str="0.0",
            general_val_bpb=4.43,
            primary_margin_y=-2.0,
        ),
        _bpb_row(
            method_name="control_decoder_only",
            general_val_bpb=4.37,
            primary_margin_y=-2.08,
        ),
        _bpb_row(
            method_name="diffmean_decoder_only",
            coefficient_str="0.2",
            general_val_bpb=4.36,
            primary_margin_y=-1.99,
        ),
        _bpb_row(
            method_name="diffmean_decoder_only",
            coefficient_str="0.4",
            general_val_bpb=4.38,
            primary_margin_y=-1.91,
        ),
        _bpb_row(
            method_name="diffmean_decoder_only",
            coefficient_str="6.0",
            general_val_bpb=8.07,
            primary_margin_y=0.60,
        ),
    ]
    points = _strength_family_points(rows, metric_column="primary_margin_y")
    kept = _exclude_strength_outliers(points)
    kept_methods_coeffs = {
        (point.method_name, point.coefficient_str) for point in kept
    }
    assert ("diffmean_decoder_only", "6.0") not in kept_methods_coeffs
    assert ("feature_diffmean_encdec", "None") in kept_methods_coeffs
    assert ("ours_gcg_encdec", "0.0") in kept_methods_coeffs


def test_roundtrip_strength_plot_uses_roundtrip_metric(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            _roundtrip_row(coefficient_str="1.0", primary_roundtrip_y=8.0),
            _roundtrip_row(coefficient_str="2.0", primary_roundtrip_y=9.0),
        ],
        schema=ROUNDTRIP_MACRO_AGG_SCHEMA,
    )

    written = plot_steering_strength_families(
        table,
        tmp_path / "roundtrip_strength",
        metric_column="primary_roundtrip_y",
        metric_label="primary roundtrip y",
    )

    assert any(path.name == STEERING_STRENGTH_FAMILY_PLOT_NAME for path in written)


def test_plot_methods_filters_strength_plot(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            _bpb_row(method_name="diffmean_decoder_only"),
            _bpb_row(method_name="linear_probe_decoder_only"),
        ],
        schema=MACRO_AGG_SCHEMA,
    )

    plot_steering_strength_families(
        table,
        tmp_path / "plots",
        metric_column="primary_margin_y",
        metric_label="primary margin",
        plot_methods=["linear_probe_decoder_only"],
    )

    legend = _csv_rows(tmp_path / "plots" / "plot_series_legend.csv")
    assert [row["method_name"] for row in legend] == ["linear_probe_decoder_only"]


def test_strength_plot_rejects_mixed_corpus(tmp_path: Path) -> None:
    table = pa.Table.from_pylist(
        [
            _bpb_row(
                method_name="diffmean_decoder_only",
                general_val_bpb=3.0,
                general_val_corpus="nd=4;nb=2277;h=aa",
            ),
            _bpb_row(
                method_name="linear_probe_decoder_only",
                general_val_bpb=3.2,
                general_val_corpus="nd=11;nb=6335;h=bb",
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="mixes .* corpus signatures"):
        plot_steering_strength_families(
            table, tmp_path / "p", metric_column="primary_margin_y", metric_label="m"
        )


_SHARED_CORPUS = "nd=11;nb=6335;h=1234567890abcdef"


def test_multi_method_shared_corpus_plots_all_families(tmp_path: Path) -> None:
    # The comparability HAPPY PATH: several distinct method families (across both
    # target models), all scored on ONE corpus, plot cleanly and every family
    # survives to the legend. This is the positive counterpart to the mixed-corpus
    # raise -- proof that "all methods on the same corpus" is what the plot accepts.
    methods = [
        ("diffmean_decoder_only", "decoder_only"),
        ("linear_probe_decoder_only", "decoder_only"),
        ("feature_diffmean_encdec", "encdec"),
        ("control_decoder_only", "decoder_only"),
    ]
    table = pa.Table.from_pylist(
        [
            _bpb_row(
                method_name=m,
                target_model=tgt,
                coefficient_str=None if m.startswith("control") else "1.0",
                general_val_bpb=3.0,
                general_val_corpus=_SHARED_CORPUS,
            )
            for m, tgt in methods
        ],
        schema=MACRO_AGG_SCHEMA,
    )

    plot_steering_strength_families(
        table, tmp_path / "plots", metric_column="primary_margin_y", metric_label="m"
    )

    legend = _csv_rows(tmp_path / "plots" / "plot_series_legend.csv")
    assert {row["method_name"] for row in legend} == {m for m, _ in methods}
    # Every plotted family reports gv present (all share the one corpus).
    assert all(row["has_general_val_bpb"] == "True" for row in legend)


def test_multi_method_one_drifted_corpus_raises_naming_it(tmp_path: Path) -> None:
    # With three families sharing a corpus and ONE drifted onto a different
    # corpus, the plot refuses and the message names the offending signatures/
    # families -- so a single mis-scored method cannot silently ride along.
    table = pa.Table.from_pylist(
        [
            _bpb_row(method_name="diffmean_decoder_only", general_val_corpus=_SHARED_CORPUS),
            _bpb_row(method_name="linear_probe_decoder_only", general_val_corpus=_SHARED_CORPUS),
            _bpb_row(
                method_name="feature_diffmean_encdec",
                target_model="encdec",
                general_val_corpus="nd=4;nb=2277;h=DRIFTED0DRIFTED0",
            ),
        ],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="mixes .* corpus signatures") as exc:
        plot_steering_strength_families(
            table, tmp_path / "p", metric_column="primary_margin_y", metric_label="m"
        )
    # the drifted family is named so the operator knows which run to re-score
    assert "feature_diffmean_encdec" in str(exc.value)
    assert "DRIFTED0DRIFTED0" in str(exc.value)


def test_strength_plot_rejects_value_with_empty_corpus(tmp_path: Path) -> None:
    # An empty-string corpus is as unusable as a null one: a SINGLE such row
    # must hit the missing-corpus raise, not slip through as its own "corpus"
    # (one row can never trip the >1-corpora mixing raise).
    table = pa.Table.from_pylist(
        [_bpb_row(general_val_bpb=3.0, general_val_corpus="")],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="without a corpus signature"):
        plot_steering_strength_families(
            table, tmp_path / "p", metric_column="primary_margin_y", metric_label="m"
        )


def test_strength_plot_rejects_zero_strength_non_control() -> None:
    table = pa.Table.from_pylist(
        [_bpb_row(adversarial_strength=0.0, is_control=False)],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="is_control=false"):
        plot_steering_strength_families(
            table,
            Path("/tmp/unused"),
            metric_column="primary_margin_y",
            metric_label="primary margin",
        )


def test_strength_plot_rejects_missing_metric_column() -> None:
    table = pa.Table.from_pylist([{"mode": "positive"}])
    with pytest.raises(AxBenchBpbAdvError, match="missing required column"):
        plot_steering_strength_families(
            table,
            Path("/tmp/unused"),
            metric_column="primary_margin_y",
            metric_label="primary margin",
        )


def test_strength_plot_rejects_non_numeric_coefficient() -> None:
    table = pa.Table.from_pylist(
        [_bpb_row(coefficient_str="large")],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="numeric coefficient_str"):
        plot_steering_strength_families(
            table,
            Path("/tmp/unused"),
            metric_column="primary_margin_y",
            metric_label="primary margin",
        )


def test_strength_plot_rejects_no_control_rows() -> None:
    table = pa.Table.from_pylist(
        [_bpb_row(adversarial_strength=1.0, is_control=False)],
        schema=MACRO_AGG_SCHEMA,
    )
    with pytest.raises(AxBenchBpbAdvError, match="no no-adversary control rows"):
        plot_steering_strength_families(
            table,
            Path("/tmp/unused"),
            metric_column="primary_margin_y",
            metric_label="primary margin",
        )
