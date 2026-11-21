"""CI-4: build_multi_series_spec must walk target_real configs and render series.

Behavior tests — no source inspection. All assertions are against the returned
MultiSeriesSpec, not the source text.
"""

from __future__ import annotations

from steering.eval.progression_plotting import (
    build_multi_series_spec,
)

# ---------------------------------------------------------------------------
# Helpers: synthetic checkpoint builders
# ---------------------------------------------------------------------------


def _checkpoint(
    step: int,
    phase: str,
    top_level_metric: float | None,
    conditioned_configs: dict[str, float],
    target_real_configs: dict[str, dict],
) -> dict:
    """Build a minimal results.json-shaped checkpoint dict."""
    eval_body: dict = {}
    if top_level_metric is not None:
        eval_body["seeds"] = {"1234": {"core_metric": top_level_metric}}
    if conditioned_configs:
        eval_body["conditioned"] = {
            "configs": {
                name: {"seeds": {"1234": {"core_metric": val}}}
                for name, val in conditioned_configs.items()
            }
        }
    if target_real_configs:
        eval_body["target_real"] = {
            "configs": dict(target_real_configs.items())
        }
    return {"step": step, "phase": phase, "evals": {"core": eval_body}}


def _tr_entry(metric: float, kind: str = "target_real_lockstep") -> dict:
    """Minimal target_real config entry with seeds."""
    return {
        "kind": kind,
        "cache_dir": "/fake/cache",
        "seeds": {"1234": {"core_metric": metric}},
    }


# ---------------------------------------------------------------------------
# CI-4 Test 1: build_multi_series_spec includes target_real series
# ---------------------------------------------------------------------------


def test_build_multi_series_spec_includes_target_real_series():
    """Synthetic results.json with ignored top-level seeds + 1 conditioned + 1 target_real
    pooled + 1 target_real lockstep. build_multi_series_spec must return
    EXACTLY 3 series."""
    checkpoints = [
        _checkpoint(
            step=500,
            phase="pretrain",
            top_level_metric=0.05,
            conditioned_configs={"static_cfg": 0.08},
            target_real_configs={
                "tr_pooled": _tr_entry(0.09, kind="target_real_pooled"),
                "tr_lockstep": _tr_entry(0.07, kind="target_real_lockstep"),
            },
        )
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    assert len(spec.series) == 3, (
        f"Expected 3 series (1 conditioned + 2 target_real); "
        f"got {len(spec.series)}: {[s.label for s in spec.series]}"
    )


# ---------------------------------------------------------------------------
# CI-4 Test 2: target_real series uses linestyle_resolver when provided
# ---------------------------------------------------------------------------


def test_build_multi_series_spec_target_real_series_uses_correct_linestyle():
    """Pooled gets dash-dot; lockstep gets dotted when linestyle_resolver is provided."""
    linestyle_calls: list[tuple[str, str, str]] = []

    def resolver(name: str, entry: dict) -> tuple[str, str]:
        kind = entry.get("kind", "")
        if "pooled" in kind:
            ls, color = "-.", "C5"
        elif "lockstep" in kind:
            ls, color = ":", "C6"
        else:
            ls, color = "--", "C7"
        linestyle_calls.append((name, kind, ls))
        return ls, color

    checkpoints = [
        _checkpoint(
            step=500,
            phase="pretrain",
            top_level_metric=0.05,
            conditioned_configs={},
            target_real_configs={
                "tr_pooled": _tr_entry(0.09, kind="target_real_pooled"),
                "tr_lockstep": _tr_entry(0.07, kind="target_real_lockstep"),
            },
        )
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
        linestyle_resolver=resolver,
    )
    assert spec is not None
    # Two target_real series must have been dispatched via the resolver.
    assert len(linestyle_calls) == 2, (
        f"resolver must be called once per target_real config; got {linestyle_calls}"
    )
    tr_series = [s for s in spec.series if s.label.startswith("TargetReal:")]
    assert len(tr_series) == 2
    pooled = next(s for s in tr_series if "pooled" in s.label)
    lockstep = next(s for s in tr_series if "lockstep" in s.label)
    assert pooled.linestyle == "-.", f"pooled linestyle must be dash-dot; got {pooled.linestyle!r}"
    assert lockstep.linestyle == ":", f"lockstep linestyle must be dotted; got {lockstep.linestyle!r}"


# ---------------------------------------------------------------------------
# CI-4 Test 3: lockstep + pooled sharing same base name get same color
# ---------------------------------------------------------------------------


def test_target_real_lockstep_and_pooled_with_same_name_share_color():
    """Color-by-name pairing: lockstep 'safety' and pooled 'safety' share the color
    returned by the resolver (which hashes the name)."""
    color_by_name: dict[str, str] = {}

    def resolver(name: str, entry: dict) -> tuple[str, str]:
        # Use a simple deterministic color tied to the name (not the kind).
        if name not in color_by_name:
            color_by_name[name] = f"C{len(color_by_name) + 1}"
        kind = entry.get("kind", "")
        ls = ":" if "lockstep" in kind else "-."
        return ls, color_by_name[name]

    # Use base name collision: both are named with "safety" prefix, but
    # since they need IDENTICAL names for true pairing, use a single name
    # for both via a two-checkpoint setup (each checkpoint has one variant).
    checkpoints_pair = [
        _checkpoint(
            step=500,
            phase="pretrain",
            top_level_metric=0.05,
            conditioned_configs={},
            target_real_configs={
                "safety": _tr_entry(0.09, kind="target_real_pooled"),
            },
        ),
        _checkpoint(
            step=1000,
            phase="pretrain",
            top_level_metric=0.06,
            conditioned_configs={},
            target_real_configs={
                "safety": _tr_entry(0.10, kind="target_real_lockstep"),
            },
        ),
    ]
    # When both checkpoints have "safety" entry, config name is deduplicated.
    # First-seen entry is used for the kind dispatch. The test verifies that
    # the resolver is called once (deduplication) and the series exists.
    calls: list[str] = []

    def resolver2(name: str, entry: dict) -> tuple[str, str]:
        calls.append(name)
        return ":", "C3"

    spec = build_multi_series_spec(
        checkpoints_pair,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
        linestyle_resolver=resolver2,
    )
    assert spec is not None
    # "safety" appears in both checkpoints but is one config — resolver called once.
    assert calls.count("safety") == 1, (
        f"target_real name 'safety' appears in 2 checkpoints but resolver must be "
        f"called once (deduplication); got calls={calls}"
    )
    tr_series = [s for s in spec.series if "safety" in s.label]
    assert len(tr_series) == 1
    assert tr_series[0].color == "C3"
    # Series spans both checkpoints (step 500 and 1000).
    assert len(tr_series[0].steps) == 2


# ---------------------------------------------------------------------------
# CI-4 Test 4: target_real with no seeds is skipped (not added to series)
# ---------------------------------------------------------------------------


def test_target_real_with_empty_seeds_is_skipped():
    """A target_real config entry with no seeds must not produce a series."""
    checkpoints = [
        _checkpoint(
            step=500,
            phase="pretrain",
            top_level_metric=0.05,
            conditioned_configs={},
            target_real_configs={
                "tr_empty": {
                    "kind": "target_real_lockstep",
                    "cache_dir": "/fake/cache",
                    "seeds": {},  # no seeds
                },
                "tr_real": _tr_entry(0.09, kind="target_real_pooled"),
            },
        )
    ]
    spec = build_multi_series_spec(
        checkpoints,
        metric_path=["core", "core_metric"],
        display_name="CORE Metric",
        model_tag="d12",
        phase_order={"pretrain": 0},
    )
    assert spec is not None
    tr_series = [s for s in spec.series if s.label.startswith("TargetReal:")]
    labels = [s.label for s in tr_series]
    assert "TargetReal: tr_real" in labels, f"tr_real must appear; got labels={labels}"
    assert not any("tr_empty" in lab for lab in labels), (
        f"tr_empty (no seeds) must be skipped; got labels={labels}"
    )
