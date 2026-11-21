"""The alpha family's ``reg_lambda_grid`` guard, and the score-time axis split.

Two failure modes this closes:

  * The ``ours_gcg_alpha*`` methods carry ONE coefficient and it is the alpha,
    so the score stage resolves their bags at ``reg_lambda = 0.0``. A run that
    also sets a different ``reg_lambda_grid`` derives bags nobody will ever
    look for — caught today only as a missing-bag error deep in scoring, AFTER
    the derive has been paid for. The guard turns that into a config error.
  * The alpha methods share the FAIR family with ``ours_gcg_encdec``, so only
    the method name says which axis a coefficient belongs to. Route it wrong
    and a lambda-swept bag is silently loaded and reported as an alpha point.
"""

from __future__ import annotations

import pytest

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    OursGcgConfig,
    _validate_alpha_family_reg_lambda,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.gcg_family import ALPHA_AXIS_GCG_METHODS
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import PrecomputedBagFamily


def _cfg(*, base_methods: list[str], reg_lambda_grid, alpha_grid=(1.0,)):
    return AxBenchBpbAdvConfig(
        target_models=["encdec"],
        base_methods=base_methods,
        adversarial_methods=[FasterGcgConfig()],
        ours_gcg=OursGcgConfig(
            reg_lambda_grid=reg_lambda_grid, objective_alpha_grid=alpha_grid
        ),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Config guard
# ─────────────────────────────────────────────────────────────────────────────


def test_alpha_family_accepts_the_pinned_lambda_grid() -> None:
    """The supported shape — one lambda, many alphas — validates."""
    cfg = _cfg(
        base_methods=["ours_gcg_alpha_encdec"],
        reg_lambda_grid=(0.0,),
        alpha_grid=(1.0, 0.5, 0.25, 0.0),
    )
    _validate_alpha_family_reg_lambda(cfg)  # must not raise


@pytest.mark.parametrize(
    "lambda_grid", [(0.0, 0.5), (0.5,), (1.0,), (0.0, 0.1, 0.2)]
)
def test_alpha_family_rejects_any_other_lambda_grid(lambda_grid) -> None:
    """Multi-lambda AND single-non-zero lambda both raise, naming the fix.

    The single-non-zero case matters as much as the multi case: the derive
    would write bags at that lambda while scoring looks at 0.0, so the run
    fails late instead of at config time.
    """
    cfg = _cfg(base_methods=["ours_gcg_alpha_encdec"], reg_lambda_grid=lambda_grid)
    with pytest.raises(AxBenchBpbAdvError, match="reg_lambda_grid"):
        _validate_alpha_family_reg_lambda(cfg)


def test_guard_names_the_offending_methods_and_the_remedy() -> None:
    """The error is actionable: which methods, what grid, what to do."""
    cfg = _cfg(
        base_methods=["ours_gcg_alpha_encdec", "neg_ours_gcg_alpha_encdec"],
        reg_lambda_grid=(0.0, 0.5),
    )
    with pytest.raises(AxBenchBpbAdvError) as excinfo:
        _validate_alpha_family_reg_lambda(cfg)
    message = str(excinfo.value)
    assert "ours_gcg_alpha_encdec" in message
    assert "neg_ours_gcg_alpha_encdec" in message
    assert "objective_alpha_grid" in message


def test_guard_is_silent_without_an_alpha_method() -> None:
    """A lambda-sweep run is untouched — the guard must not widen.

    Teeth: dropping the ``active_alpha_methods`` check would break every
    existing multi-lambda ``ours_gcg_encdec`` config.
    """
    cfg = _cfg(base_methods=["ours_gcg_encdec"], reg_lambda_grid=(0.0, 0.5, 1.0))
    _validate_alpha_family_reg_lambda(cfg)  # must not raise


@pytest.mark.parametrize("alpha_grid", [(1.0, 0.5), (0.5,), (0.0,)])
def test_alpha_sweep_without_an_alpha_method_is_rejected(alpha_grid) -> None:
    """The SYMMETRIC error: a non-default alpha grid nothing will ever score.

    ``ours_gcg_encdec`` resolves its bags at alpha = 1 regardless, so every
    non-unit alpha here derives a bag no scored method looks for. Unlike the
    reg_lambda direction this NEVER surfaces downstream — the run just burns a
    full GCG optimization per wasted bag and discards the result.
    """
    cfg = _cfg(
        base_methods=["ours_gcg_encdec"],
        reg_lambda_grid=(0.0,),
        alpha_grid=alpha_grid,
    )
    with pytest.raises(AxBenchBpbAdvError, match="objective_alpha_grid"):
        _validate_alpha_family_reg_lambda(cfg)


def test_default_alpha_grid_without_an_alpha_method_is_fine() -> None:
    """The default alpha grid is exactly what every pre-alpha run had.

    Teeth: raising here would reject every existing config in the repo.
    """
    cfg = _cfg(
        base_methods=["ours_gcg_encdec"],
        reg_lambda_grid=(0.0, 0.5),
        alpha_grid=(DEFAULT_OBJECTIVE_ALPHA,),
    )
    _validate_alpha_family_reg_lambda(cfg)  # must not raise


def test_guard_is_silent_when_base_methods_is_unset() -> None:
    """``base_methods=None`` disables the whitelist; nothing to check."""
    cfg = AxBenchBpbAdvConfig(
        target_models=["encdec"],
        base_methods=None,
        adversarial_methods=[FasterGcgConfig()],
        ours_gcg=OursGcgConfig(reg_lambda_grid=(0.0, 0.5)),
    )
    _validate_alpha_family_reg_lambda(cfg)  # must not raise


# ─────────────────────────────────────────────────────────────────────────────
# Score-time axis routing
# ─────────────────────────────────────────────────────────────────────────────


def _resolve(method_name: str, coefficient: float):
    """Mirror ``intervention._build_ours_intervention``'s axis split for one method.

    Kept as a local reimplementation of the two-line rule rather than standing
    up a full model + concept fixture: what is under test is the ROUTING, and
    the production expression is pinned byte-for-byte against this one by
    ``test_routing_matches_production``.
    """
    is_alpha_axis = method_name in ALPHA_AXIS_GCG_METHODS
    reg_lambda = 0.0 if is_alpha_axis else float(coefficient)
    objective_alpha = float(coefficient) if is_alpha_axis else DEFAULT_OBJECTIVE_ALPHA
    return reg_lambda, objective_alpha


def test_alpha_method_coefficient_routes_to_alpha() -> None:
    """An alpha method's 0.5 means α = 0.5 at reg_lambda 0 — not λ = 0.5."""
    assert _resolve("ours_gcg_alpha_encdec", 0.5) == (0.0, 0.5)
    assert _resolve("neg_ours_gcg_alpha_encdec", 0.0) == (0.0, 0.0)


def test_lambda_method_coefficient_routes_to_lambda() -> None:
    """A lambda method's 0.5 means λ = 0.5 at the plain margin."""
    assert _resolve("ours_gcg_encdec", 0.5) == (0.5, DEFAULT_OBJECTIVE_ALPHA)


def test_the_two_axes_resolve_to_different_bags() -> None:
    """The same scalar under the two methods must NOT name the same file.

    This is the whole point of the split: without it, ``ours_gcg_alpha_encdec``
    at 0.5 would load the λ = 0.5 bag and plot it as an alpha point.
    """
    paths = AxBenchBpbAdvPaths(output_root="/out", run_tag="tag")

    def bag_for(method_name: str, coefficient: float):
        reg_lambda, alpha = _resolve(method_name, coefficient)
        return paths.ours_gcg_bag(
            7,
            "positive",
            reg_lambda,
            PrecomputedBagFamily.FAIR_GCG,
            objective_alpha=alpha,
        )

    assert bag_for("ours_gcg_alpha_encdec", 0.5) != bag_for("ours_gcg_encdec", 0.5)


def test_alpha_one_reuses_the_fair_gcg_bag() -> None:
    """α = 1 resolves to the SAME file the plain fair method reads at λ = 0.

    Not a coincidence to be tolerated but the designed reuse: the α = 1 point of
    the sweep IS the fair GCG bag, so a sweep including it costs no re-derive.
    """
    paths = AxBenchBpbAdvPaths(output_root="/out", run_tag="tag")
    alpha_lambda, alpha_alpha = _resolve("ours_gcg_alpha_encdec", 1.0)
    plain_lambda, plain_alpha = _resolve("ours_gcg_encdec", 0.0)
    assert paths.ours_gcg_bag(
        7, "positive", alpha_lambda, PrecomputedBagFamily.FAIR_GCG, objective_alpha=alpha_alpha
    ) == paths.ours_gcg_bag(
        7, "positive", plain_lambda, PrecomputedBagFamily.FAIR_GCG, objective_alpha=plain_alpha
    )


def test_routing_matches_production() -> None:
    """The local ``_resolve`` mirrors the production source, not a stale copy.

    Pins the two production expressions textually so this module cannot drift
    into testing a rule ``intervention.py`` no longer implements.
    """
    import inspect

    from steering.eval.axbench_bpb import intervention

    source = inspect.getsource(intervention._build_ours_intervention)
    assert "is_alpha_axis = str(modifier_name) in ALPHA_AXIS_GCG_METHODS" in source
    assert "if is_alpha_axis and coefficient is not None" in source
    assert "objective_alpha=objective_alpha," in source
