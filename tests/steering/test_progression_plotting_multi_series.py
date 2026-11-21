"""Unit tests for progression_plotting data model (no matplotlib imports)."""

from __future__ import annotations

import dataclasses
import logging
from typing import Any

import pytest
from loguru import logger as _logger

from steering.eval.progression_plotting import (
    ChartSeries,
    MultiSeriesSpec,
    PhaseBoundary,
    _cumulative_x,
    _phase_boundaries_from_series,
    _phase_offsets_from_boundaries,
    build_multi_series_spec,
)


@pytest.fixture(autouse=True)
def _loguru_to_caplog(caplog):
    """Route loguru records into pytest's caplog so warnings can be asserted.

    Loguru accepts a stdlib logging.Handler directly; caplog.handler is one.
    """
    handler_id = _logger.add(caplog.handler, level="WARNING", format="{message}")
    caplog.set_level(logging.WARNING)
    yield
    _logger.remove(handler_id)


def test_chart_series_is_frozen():
    series = ChartSeries(
        label="baseline",
        steps=(100, 200),
        phases=("pretrain", "pretrain"),
        means=(0.05, 0.06),
        stds=(0.01, 0.01),
        color="steelblue",
        linestyle="solid",
        is_baseline=True,
    )
    # Slice 14 audit X2: narrowed from bare ``Exception``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        series.label = "other"  # frozen dataclass  # ty: ignore[invalid-assignment]


def test_phase_boundary_is_frozen():
    boundary = PhaseBoundary(
        x=1000.0, from_phase="pretrain", to_phase="sft", color="red"
    )
    # Slice 14 audit X2: narrowed from bare ``Exception``.
    with pytest.raises(dataclasses.FrozenInstanceError):
        boundary.x = 2000.0  # ty: ignore[invalid-assignment]


def test_multi_series_spec_holds_series_and_boundaries():
    series = ChartSeries(
        label="baseline",
        steps=(100,),
        phases=("pretrain",),
        means=(0.05,),
        stds=(0.0,),
        color="steelblue",
        linestyle="solid",
        is_baseline=True,
    )
    boundary = PhaseBoundary(x=100.0, from_phase="pretrain", to_phase="sft", color="red")
    spec = MultiSeriesSpec(
        title="d27 — CORE Metric",
        metric_name="CORE Metric",
        model_tag="d27",
        series=(series,),
        phase_boundaries=(boundary,),
        legend_outside=False,
    )
    assert spec.series[0].label == "baseline"
    assert spec.phase_boundaries[0].to_phase == "sft"
    assert spec.legend_outside is False


def _checkpoint(step: int, phase: str, seeds: dict[str, dict]) -> dict:
    return {
        "step": step,
        "phase": phase,
        "evals": {"core": seeds},
    }


def test_build_spec_ignores_top_level_seeds_key():
    checkpoints = [
        _checkpoint(
            500,
            "pretrain",
            {"seeds": {"1234": {"core_metric": 0.05}, "2234": {"core_metric": 0.07}}},
        ),
        _checkpoint(
            1000,
            "pretrain",
            {"seeds": {"1234": {"core_metric": 0.10}, "2234": {"core_metric": 0.12}}},
        ),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d27",
        phase_order={"pretrain": 0},
    )
    assert spec is None


def test_build_spec_ignores_unknown_top_level_eval_key():
    checkpoints = [
        _checkpoint(
            500,
            "pretrain",
            {
                "unsupported_legacy_key": {
                    "seeds": {
                        "1234": {"core_metric": 0.05},
                        "2234": {"core_metric": 0.07},
                    }
                }
            },
        ),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d27",
        phase_order={"pretrain": 0},
    )
    assert spec is None


def test_build_spec_returns_none_when_no_data():
    checkpoints = [_checkpoint(500, "pretrain", {"seeds": {}})]
    assert (
        build_multi_series_spec(
            checkpoints,
            metric_path=["core", "core_metric"],
            display_name="CORE Metric",
            model_tag="d27",
            phase_order={"pretrain": 0},
        )
        is None
    )


def test_build_spec_missing_metric_path_returns_none():
    checkpoints = [_checkpoint(500, "pretrain", {"seeds": {"1234": {"core_metric": 0.1}}})]
    assert (
        build_multi_series_spec(
            checkpoints,
            metric_path=["harmfulness", "asr"],
            display_name="ASR",
            model_tag="d27",
            phase_order={"pretrain": 0},
        )
        is None
    )


def test_build_spec_filters_nan_inf():
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {
                    "conditioned": {
                        "configs": {
                            "safety": {
                                "seeds": {
                                    "1234": {"core_metric": float("nan")},
                                    "2234": {"core_metric": 0.10},
                                    "3234": {"core_metric": float("inf")},
                                }
                            }
                        }
                    },
                }
            },
        },
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d27",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    assert spec.series[0].means == pytest.approx((0.10,))
    assert spec.series[0].stds == pytest.approx((0.0,))


def _encdec_checkpoint(
    step: int,
    phase: str,
    baseline_metric: float | None,
    configs: dict[str, float],
) -> dict:
    eval_body: dict[str, Any] = {}
    if baseline_metric is not None:
        eval_body["seeds"] = {"1234": {"core_metric": baseline_metric}}
    if configs:
        eval_body["conditioned"] = {
            "configs": {
                name: {
                    "target_features_file": f"/fake/{name}.yaml",
                    "context_cache_dir": "/fake/cache/c1024",
                    "seeds": {"1234": {"core_metric": val}},
                }
                for name, val in configs.items()
            }
        }
    return {"step": step, "phase": phase, "evals": {"core": eval_body}}


def test_build_spec_top_level_seeds_ignored_with_one_config():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"safety_top32": 0.08}),
        _encdec_checkpoint(1000, "pretrain", 0.06, {"safety_top32": 0.09}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    assert len(spec.series) == 1
    conditioned = spec.series[0]
    assert conditioned.is_baseline is False
    assert conditioned.label == "Conditioned: safety_top32"
    assert conditioned.linestyle == "dashed"
    # First conditioned color is the first entry of tab10
    assert conditioned.color == "#1f77b4"
    assert conditioned.means == pytest.approx((0.08, 0.09))


def test_build_spec_multiple_configs_sorted_and_distinct_colors():
    checkpoints = [
        _encdec_checkpoint(
            500, "pretrain", 0.05, {"zeta": 0.08, "alpha": 0.07, "mu": 0.09}
        ),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    assert [s.label for s in spec.series] == [
        "Conditioned: alpha",
        "Conditioned: mu",
        "Conditioned: zeta",
    ]
    cond_colors = [s.color for s in spec.series]
    assert len(set(cond_colors)) == 3


def test_build_spec_conditioned_only_when_no_baseline():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", None, {"safety": 0.10}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    assert len(spec.series) == 1
    assert spec.series[0].is_baseline is False
    assert spec.series[0].label == "Conditioned: safety"


def test_build_spec_palette_uses_tab20_when_above_ten_configs():
    configs = {f"cfg_{i:02d}": 0.1 + i * 0.01 for i in range(15)}
    checkpoints = [_encdec_checkpoint(500, "pretrain", 0.05, configs)]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    conditioned_colors = [s.color for s in spec.series if not s.is_baseline]
    assert len(conditioned_colors) == 15
    assert len(set(conditioned_colors)) == 15  # tab20 has enough slots


def test_build_spec_palette_cycles_with_warning_above_twenty(caplog):
    configs = {f"cfg_{i:02d}": 0.1 for i in range(22)}
    checkpoints = [_encdec_checkpoint(500, "pretrain", 0.05, configs)]
    with caplog.at_level(logging.WARNING):
        spec = build_multi_series_spec(
            checkpoints,
            metric_path=["core", "core_metric"],
            display_name="CORE Metric",
            model_tag="d12",
            phase_order={"pretrain": 0},
        )
    assert spec is not None
    conditioned_colors = [s.color for s in spec.series if not s.is_baseline]
    assert len(conditioned_colors) == 22
    # Some colors repeat beyond slot 20.
    assert len(set(conditioned_colors)) == 20
    assert any(
        "palette cycling" in rec.message.lower()
        or ("palette" in rec.message.lower() and "cycle" in rec.message.lower())
        for rec in caplog.records
    )


def test_no_conditioned_or_target_real_series_returns_none():
    checkpoints = [_encdec_checkpoint(500, "pretrain", 0.05, {})]
    spec = build_multi_series_spec(
        checkpoints, ["core", "core_metric"], "CORE Metric", "d12", {"pretrain": 0}
    )
    assert spec is None


def test_legend_outside_false_when_two_or_fewer_conditioned():
    for n in (1, 2):
        configs = {f"c{i}": 0.1 for i in range(n)}
        checkpoints = [_encdec_checkpoint(500, "pretrain", 0.05, configs)]
        spec = build_multi_series_spec(
            checkpoints, ["core", "core_metric"], "CORE Metric", "d12", {"pretrain": 0}
        )
        assert spec is not None, f"spec should exist for n={n}"
        assert spec.legend_outside is False, f"n={n} should be inside"


def test_legend_outside_true_when_three_or_more_conditioned():
    for n in (3, 5, 8):
        configs = {f"c{i}": 0.1 for i in range(n)}
        checkpoints = [_encdec_checkpoint(500, "pretrain", 0.05, configs)]
        spec = build_multi_series_spec(
            checkpoints, ["core", "core_metric"], "CORE Metric", "d12", {"pretrain": 0}
        )
        assert spec is not None, f"spec should exist for n={n}"
        assert spec.legend_outside is True, f"n={n} should be outside"


def test_phase_boundaries_single_phase_is_empty():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"safety": 0.08}),
        _encdec_checkpoint(1000, "pretrain", 0.06, {"safety": 0.09}),
    ]
    spec = build_multi_series_spec(
        checkpoints, ["core", "core_metric"], "CORE Metric", "d12", {"pretrain": 0}
    )
    assert spec is not None
    assert spec.phase_boundaries == ()


def test_phase_boundaries_pretrain_to_sft():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"safety": 0.08}),
        _encdec_checkpoint(1000, "pretrain", 0.06, {"safety": 0.09}),
        _encdec_checkpoint(200, "sft", 0.12, {"safety": 0.14}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        ["core", "core_metric"],
        "CORE Metric",
        "d12",
        {"pretrain": 0, "sft": 1},
    )
    assert spec is not None
    assert len(spec.phase_boundaries) == 1
    boundary = spec.phase_boundaries[0]
    assert boundary.from_phase == "pretrain"
    assert boundary.to_phase == "sft"
    assert boundary.x == pytest.approx(1000.0)
    assert boundary.color == "red"


def test_phase_boundaries_multi_phase_color_cycle():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"safety": 0.08}),
        _encdec_checkpoint(200, "safety_sft", 0.10, {"safety": 0.10}),
        _encdec_checkpoint(300, "capabilities_sft", 0.12, {"safety": 0.12}),
        _encdec_checkpoint(400, "standard_sft", 0.11, {"safety": 0.11}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        ["core", "core_metric"],
        "CORE Metric",
        "d12",
        {"pretrain": 0, "safety_sft": 1, "capabilities_sft": 2, "standard_sft": 3},
    )
    assert spec is not None
    colors = [b.color for b in spec.phase_boundaries]
    assert colors == ["red", "orange", "purple"]


def test_phase_boundaries_ignore_top_level_seeds_when_present():
    """Top-level seeds do not define phase boundaries for conditioned charts."""
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"safety": 0.08}),
        _encdec_checkpoint(1000, "pretrain", 0.06, {}),
        _encdec_checkpoint(200, "sft", 0.12, {"safety": 0.14}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        ["core", "core_metric"],
        "CORE Metric",
        "d12",
        {"pretrain": 0, "sft": 1},
    )
    assert spec is not None
    assert len(spec.phase_boundaries) == 1
    # The conditioned series has no 1000-step pretrain datapoint.
    assert spec.phase_boundaries[0].x == pytest.approx(500.0)


def test_phase_boundaries_fallback_to_longest_conditioned():
    """No baseline → use the conditioned series with the most checkpoints."""
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", None, {"short": 0.1, "long": 0.1}),
        _encdec_checkpoint(1000, "pretrain", None, {"long": 0.1}),
        _encdec_checkpoint(200, "sft", None, {"short": 0.12, "long": 0.12}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        ["core", "core_metric"],
        "CORE Metric",
        "d12",
        {"pretrain": 0, "sft": 1},
    )
    assert spec is not None
    assert len(spec.phase_boundaries) == 1
    # "long" series has 3 checkpoints across 2 phases, so boundary is at 1000.
    assert spec.phase_boundaries[0].x == pytest.approx(1000.0)


def test_build_spec_partial_checkpoint_coverage():
    """A conditioned series can be defined on a strict subset of checkpoints."""
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"safety": 0.08}),
        _encdec_checkpoint(1000, "pretrain", 0.06, {}),
        _encdec_checkpoint(1500, "pretrain", 0.07, {"safety": 0.10}),
    ]
    spec = build_multi_series_spec(
        checkpoints,
        ["core", "core_metric"],
        "CORE Metric",
        "d12",
        {"pretrain": 0},
    )
    assert spec is not None
    conditioned = spec.series[0]
    assert conditioned.steps == (500, 1500)


def test_build_spec_chat_metric_nested_path():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {}),
    ]
    checkpoints[0]["evals"]["chat"] = {
        "seeds": {"1234": {"MMLU": 0.3, "chatcore_metric": 0.25}},
        "conditioned": {
            "configs": {
                "safety": {
                    "seeds": {"1234": {"MMLU": 0.28, "chatcore_metric": 0.24}},
                }
            }
        },
    }
    spec = build_multi_series_spec(
        checkpoints,
        ["chat", "MMLU"],
        "MMLU",
        "d12",
        {"pretrain": 0},
    )
    assert spec is not None
    assert len(spec.series) == 1
    assert spec.series[0].means == pytest.approx((0.28,))


def test_build_spec_deterministic_two_runs():
    checkpoints = [
        _encdec_checkpoint(500, "pretrain", 0.05, {"a": 0.1, "b": 0.2}),
    ]
    spec_a = build_multi_series_spec(
        checkpoints, ["core", "core_metric"], "CORE Metric", "d12", {"pretrain": 0}
    )
    spec_b = build_multi_series_spec(
        checkpoints, ["core", "core_metric"], "CORE Metric", "d12", {"pretrain": 0}
    )
    assert spec_a == spec_b


def test_extract_seeds_ignores_non_dict_mid_path():
    """When descending a nested metric_path hits a non-dict, skip the seed."""
    from steering.eval.progression_plotting import _extract_seeds_for_path

    seeds_data = {
        "1234": {"core": {"per_task": 0.5}},   # non-dict where we expect dict
        "2234": {"core": {"per_task": {"task_a": 0.6}}},
    }
    values = _extract_seeds_for_path(seeds_data, ["core", "per_task", "task_a"])
    assert values == [0.6]


def test_series_for_conditioned_config_missing_data_returns_empty_series():
    from steering.eval.progression_plotting import _series_for_conditioned_config

    assert _series_for_conditioned_config([], "core", [], "missing") == ((), (), (), ())


def test_reference_series_empty_input_returns_none():
    from steering.eval.progression_plotting import _reference_series_for_boundaries

    assert _reference_series_for_boundaries([]) is None


def test_build_spec_color_index_reflects_canonical_config_order():
    """A config declared without data still consumes its palette slot.

    This locks the invariant that color assignments are driven by the sorted
    list of *all* declared configs, not just the subset that has data on a
    given checkpoint. Without it, a later run that adds data for a previously
    empty config would silently shuffle every existing config's color.
    """
    checkpoints = [
        {
            "step": 500,
            "phase": "pretrain",
            "evals": {
                "core": {
                    "seeds": {"1234": {"core_metric": 0.05}},
                    "conditioned": {
                        "configs": {
                            "alpha": {"seeds": {"1234": {"core_metric": 0.1}}},
                            # "beta" is declared but has no extractable metric.
                            "beta": {"seeds": {"1234": {}}},
                            "gamma": {"seeds": {"1234": {"core_metric": 0.2}}},
                        }
                    },
                }
            },
        }
    ]
    spec = build_multi_series_spec(
        checkpoints,
        ["core", "core_metric"],
        "CORE Metric",
        "d12",
        {"pretrain": 0},
    )
    assert spec is not None
    conditioned = {s.label: s for s in spec.series if not s.is_baseline}
    assert set(conditioned) == {"Conditioned: alpha", "Conditioned: gamma"}
    # alpha is idx 0 of 3 declared configs → tab10[0] = "#1f77b4"
    # gamma is idx 2 of 3 declared configs → tab10[2] = "#2ca02c"
    # (NOT tab10[1] — the empty "beta" slot is preserved, not collapsed.)
    assert conditioned["Conditioned: alpha"].color == "#1f77b4"
    assert conditioned["Conditioned: gamma"].color == "#2ca02c"


# ===================================================================
# Unit tests: _phase_offsets_from_boundaries
# ===================================================================


def test_phase_offsets_from_empty_boundaries():
    assert _phase_offsets_from_boundaries(()) == {}


def test_phase_offsets_from_single_boundary():
    boundary = PhaseBoundary(
        x=54600.0, from_phase="pretrain", to_phase="safety_sft", color="red"
    )
    assert _phase_offsets_from_boundaries((boundary,)) == {"safety_sft": 54600.0}


def test_phase_offsets_from_multiple_boundaries_keys_by_to_phase():
    boundaries = (
        PhaseBoundary(
            x=54600.0, from_phase="pretrain", to_phase="safety_sft", color="red"
        ),
        PhaseBoundary(
            x=55486.0,
            from_phase="safety_sft",
            to_phase="capabilities_sft",
            color="orange",
        ),
    )
    assert _phase_offsets_from_boundaries(boundaries) == {
        "safety_sft": 54600.0,
        "capabilities_sft": 55486.0,
    }


def test_phase_offsets_duplicate_to_phase_keeps_last():
    # Pathological but well-defined: dict semantics keep the last write.
    boundaries = (
        PhaseBoundary(x=10.0, from_phase="a", to_phase="b", color="red"),
        PhaseBoundary(x=20.0, from_phase="c", to_phase="b", color="orange"),
    )
    assert _phase_offsets_from_boundaries(boundaries) == {"b": 20.0}


def test_phase_offsets_returns_floats():
    # Even when a PhaseBoundary holds an integer-valued x.
    boundary = PhaseBoundary(x=54600, from_phase="pretrain", to_phase="sft", color="red")
    out = _phase_offsets_from_boundaries((boundary,))
    assert isinstance(out["sft"], float)
    assert out["sft"] == 54600.0


# ===================================================================
# Unit tests: _cumulative_x with explicit phase_offsets
# ===================================================================


def test_cumulative_x_no_offsets_matches_legacy_local_cumulative():
    """Empty offsets dict gives the original per-series cumulative behavior."""
    steps = (1000, 2000, 500, 1000)
    phases = ("pretrain", "pretrain", "sft", "sft")

    assert _cumulative_x(steps, phases) == _cumulative_x(steps, phases, {})
    assert _cumulative_x(steps, phases, None) == _cumulative_x(steps, phases, {})


def test_cumulative_x_legacy_local_cumulative_values():
    """Explicit values for the legacy local-cumulative path."""
    # pretrain 1000, 2000 -> 1000, 2000
    # sft 500, 1000 -> 2000+500=2500, 2000+1000=3000 (offset = last pretrain x)
    assert _cumulative_x(
        (1000, 2000, 500, 1000), ("pretrain", "pretrain", "sft", "sft")
    ) == (1000.0, 2000.0, 2500.0, 3000.0)


def test_cumulative_x_phase_in_offsets_uses_offset_plus_step():
    out = _cumulative_x(
        (54000, 886),
        ("pretrain", "safety_sft"),
        {"safety_sft": 54600.0},
    )
    # pretrain not in offsets -> local cumulative -> 54000
    # safety_sft in offsets -> 54600 + 886 = 55486
    assert out == (54000.0, 55486.0)


def test_cumulative_x_phase_not_in_offsets_uses_local_cumulative():
    """A phase missing from offsets falls through to the legacy branch."""
    # Only safety_sft is in offsets; capabilities_sft is not.
    out = _cumulative_x(
        (54600, 886, 1106),
        ("pretrain", "safety_sft", "capabilities_sft"),
        {"safety_sft": 54600.0},
    )
    # pretrain 54600 -> local 54600
    # safety_sft 886 -> offsets 54600+886 = 55486
    # capabilities_sft 1106 -> not in offsets; phase changed; phase_offset = prev_last_x = 55486; cur_x = 55486+1106 = 56592
    assert out == (54600.0, 55486.0, 56592.0)


def test_cumulative_x_three_phase_chain_all_in_offsets():
    out = _cumulative_x(
        (54000, 886, 1106),
        ("pretrain", "safety_sft", "capabilities_sft"),
        {"safety_sft": 54600.0, "capabilities_sft": 55486.0},
    )
    # pretrain not in offsets -> 54000
    # safety_sft -> 54600+886 = 55486
    # capabilities_sft -> 55486+1106 = 56592
    assert out == (54000.0, 55486.0, 56592.0)


def test_cumulative_x_first_phase_in_offsets():
    """A series that starts in a phase present in offsets."""
    out = _cumulative_x((886,), ("safety_sft",), {"safety_sft": 54600.0})
    assert out == (55486.0,)


def test_cumulative_x_empty_inputs():
    assert _cumulative_x((), (), {"safety_sft": 54600.0}) == ()
    assert _cumulative_x((), ()) == ()


def test_cumulative_x_round_trip_identity_for_reference_series():
    """Round-trip: derive boundaries from a series, feed back as offsets — identity.

    This is the single-series renderer invariant. The boundaries are computed via `_cumulative_x`
    *without* offsets, then `_phase_offsets_from_boundaries(boundaries)` is
    fed back into `_cumulative_x` and must produce the same x tuple.
    """
    steps = (54000, 54600, 886, 1106)
    phases = ("pretrain", "pretrain", "safety_sft", "capabilities_sft")
    no_offset = _cumulative_x(steps, phases)

    series = ChartSeries(
        label="ref",
        steps=steps,
        phases=phases,
        means=(0.0,) * len(steps),
        stds=(0.0,) * len(steps),
        color="steelblue",
        linestyle="solid",
        is_baseline=True,
    )
    boundaries = _phase_boundaries_from_series(series)
    offsets = _phase_offsets_from_boundaries(boundaries)
    with_offset = _cumulative_x(steps, phases, offsets)

    assert no_offset == with_offset


@pytest.mark.parametrize(
    "steps,phases",
    [
        ((1000,), ("pretrain",)),
        ((1000, 2000, 3000), ("pretrain", "pretrain", "pretrain")),
        ((1000, 500), ("pretrain", "sft")),
        ((1000, 2000, 500), ("pretrain", "pretrain", "sft")),
        ((1000, 500, 200), ("pretrain", "safety_sft", "capabilities_sft")),
        (
            (54000, 54600, 886, 1106),
            ("pretrain", "pretrain", "safety_sft", "capabilities_sft"),
        ),
    ],
)
def test_cumulative_x_round_trip_identity_for_various_shapes(steps, phases):
    """The round-trip identity is a property of the helper, not a single shape.

    For ANY reference series, boundaries derived from it and fed back as
    offsets must reproduce the same x tuple. This test exercises six
    representative shapes (single phase, multi-step single phase, two
    phases, multi-step then SFT, three-phase chain, d12_baseline).
    """
    no_offset = _cumulative_x(steps, phases)
    series = ChartSeries(
        label="ref",
        steps=steps,
        phases=phases,
        means=(0.0,) * len(steps),
        stds=(0.0,) * len(steps),
        color="steelblue",
        linestyle="solid",
        is_baseline=True,
    )
    boundaries = _phase_boundaries_from_series(series)
    offsets = _phase_offsets_from_boundaries(boundaries)
    with_offset = _cumulative_x(steps, phases, offsets)
    assert no_offset == with_offset


def test_cumulative_x_extra_offset_keys_are_ignored():
    """Phases not in the series ignore extraneous offset entries — no behavior change."""
    out = _cumulative_x(
        (1000, 2000),
        ("pretrain", "pretrain"),
        {"safety_sft": 99999.0, "capabilities_sft": 88888.0},
    )
    assert out == (1000.0, 2000.0)


def test_cumulative_x_phase_oscillation_with_offsets():
    """Phase oscillation with mixed offsets resumes local cumulative correctly.

    pretrain → sft → pretrain. The second pretrain entry is NOT in offsets,
    so it falls through to the local-cumulative branch using the previous
    cumulative x as the new local offset (which was set by the offset-mode
    sft entry).
    """
    out = _cumulative_x(
        (1000, 100, 500),
        ("pretrain", "sft", "pretrain"),
        {"sft": 1500.0},
    )
    # pretrain 1000 -> local 1000.
    # sft 100 -> offsets[sft] + 100 = 1600.
    # pretrain 500 -> not in offsets; phase != prev (sft); phase_offset = prev_last_x = 1600;
    #                cur_x = 1600 + 500 = 2100.
    assert out == (1000.0, 1600.0, 2100.0)
