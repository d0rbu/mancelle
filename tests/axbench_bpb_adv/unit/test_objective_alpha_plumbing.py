"""``objective_alpha`` plumbing: path, identity, coefficient axis, config guard.

The objective's arithmetic is pinned in
``tests/steering/feature_gcg/unit/test_objective_alpha.py``. This module pins
the surfaces that decide WHICH bag an alpha carries and whether an existing bag
still resolves:

  * strip-default — at α = 1 the bag path and the identity payload must be
    EXACTLY what they were before the knob existed. Every fair/roofline bag
    already on disk was derived under the plain margin and carries no alpha
    stamp, so a non-stripped default would fail all of them for a "missing
    identity key" and force a multi-day re-derive that could only reproduce
    the same bytes (the ``max_grad_batch`` failure mode, #1054).
  * separation — a non-default α must NOT collide with the plain-margin bag,
    in either the path or the identity, or a sweep would overwrite the very
    artifact it reuses at α = 1.
  * axis routing — the ``ours_gcg_alpha*`` methods share the FAIR family with
    ``ours_gcg_encdec``, so the only thing distinguishing their coefficient is
    the method name. Reading it on the wrong axis silently loads a
    lambda-swept bag and reports it under the alpha method.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from steering.conditional.feature_gcg.config import DEFAULT_OBJECTIVE_ALPHA
from steering.eval.axbench_base.coefficient_grids import (
    ours_gcg_alpha_grid,
    ours_gcg_lambda_grid,
)
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.types import SteeringMode, TargetModelMetadata
from steering.eval.axbench_bpb_adv.aggregation import _steering_family
from steering.eval.axbench_bpb_adv.gcg_family import (
    ALPHA_AXIS_GCG_METHODS,
    FAIR_GCG_METHOD_MODES,
)
from steering.eval.axbench_bpb_adv.paths import (
    AxBenchBpbAdvPaths,
    format_objective_alpha,
)
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    PrecomputedBagFamily,
    precomputed_bag_method_family,
)

_PATHS = AxBenchBpbAdvPaths(output_root="/out", run_tag="tag")
_TARGET_META = TargetModelMetadata(
    num_buckets=None,
    n_sae_features=None,
    negative_selection=None,
    feature_selection=None,
    per_feature_stats=None,
    device="cpu",
    sae_layer=8,
    segment_size=256,
)


# ─────────────────────────────────────────────────────────────────────────────
# Path: strip-default and separation
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "family", [PrecomputedBagFamily.FAIR_GCG, PrecomputedBagFamily.ROOFLINE_GCG]
)
def test_default_alpha_path_is_byte_identical_to_the_pre_alpha_path(family) -> None:
    """α = 1 (and the omitted argument) resolve to the legacy filename exactly.

    Teeth: appending an ``_alpha_1.0`` suffix unconditionally would orphan every
    bag on disk — this pin RED before any of them is re-derived.
    """
    kwargs = {"prefix_id": "p0"} if family is PrecomputedBagFamily.ROOFLINE_GCG else {}
    legacy = _PATHS.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.0, family, **kwargs)
    explicit = _PATHS.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        family,
        objective_alpha=DEFAULT_OBJECTIVE_ALPHA,
        **kwargs,
    )
    assert legacy == explicit
    assert legacy.name == "lambda_0.0.safetensors"


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 2.0])
def test_non_default_alpha_gets_its_own_path(alpha) -> None:
    """Each non-default α is a distinct sibling of the plain-margin bag."""
    plain = _PATHS.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.0, PrecomputedBagFamily.FAIR_GCG)
    alpha_path = _PATHS.ours_gcg_bag(
        7, SteeringMode.POSITIVE, 0.0, PrecomputedBagFamily.FAIR_GCG, objective_alpha=alpha
    )
    assert alpha_path != plain
    assert alpha_path.parent == plain.parent
    assert format_objective_alpha(alpha) in alpha_path.name


def test_distinct_alphas_never_collide() -> None:
    """A whole sweep maps to as many paths as it has α values."""
    alphas = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 2.0]
    paths = {
        _PATHS.ours_gcg_bag(
            7, SteeringMode.POSITIVE, 0.0, PrecomputedBagFamily.FAIR_GCG, objective_alpha=a
        )
        for a in alphas
    }
    assert len(paths) == len(alphas)


def test_alpha_slug_round_trips() -> None:
    """``float(format_objective_alpha(x)) == x`` — the loader reconstructs the path."""
    for alpha in (0.0, 0.25, 0.5, 1.0, 1e-3, 2.5):
        assert float(format_objective_alpha(alpha)) == alpha


# ─────────────────────────────────────────────────────────────────────────────
# Identity: strip-default and separation
# ─────────────────────────────────────────────────────────────────────────────


def test_identity_omits_alpha_at_the_default() -> None:
    """The plain-margin identity payload carries NO alpha key.

    This is what keeps every pre-alpha bag reusable: its stored metadata has no
    such key, and ``committed_bag_for_reuse`` compares the payloads.
    """
    from steering.eval.axbench_bpb_adv.gcg_bags import _stamp_objective_alpha

    identity: dict[str, str] = {}
    _stamp_objective_alpha(identity, DEFAULT_OBJECTIVE_ALPHA)
    assert identity == {}


@pytest.mark.parametrize("alpha", [0.0, 0.25, 0.5, 2.0])
def test_identity_stamps_every_non_default_alpha(alpha) -> None:
    """A non-default α IS stamped, so it can never be reused as the plain margin."""
    from steering.eval.axbench_bpb_adv.gcg_bags import _stamp_objective_alpha

    identity: dict[str, str] = {}
    _stamp_objective_alpha(identity, alpha)
    assert identity == {"objective_alpha": format_objective_alpha(alpha)}


def test_identity_distinguishes_two_non_default_alphas() -> None:
    """Two different α values stamp two different identities."""
    from steering.eval.axbench_bpb_adv.gcg_bags import _stamp_objective_alpha

    a, b = {}, {}
    _stamp_objective_alpha(a, 0.25)
    _stamp_objective_alpha(b, 0.5)
    assert a != b


# ─────────────────────────────────────────────────────────────────────────────
# Derive grid
# ─────────────────────────────────────────────────────────────────────────────


def test_default_alpha_grid_degenerates_to_the_lambda_sequence() -> None:
    """With the default α grid the derive walks EXACTLY the old lambda sequence.

    Same bags, same order, same progress total — so enabling this feature does
    not perturb a run that never asked for it.
    """
    from types import SimpleNamespace

    from steering.eval.axbench_bpb_adv.gcg_bags import _lambda_alpha_grid

    ours = SimpleNamespace(
        reg_lambda_grid=(0.0, 0.5, 1.0),
        objective_alpha_grid=(DEFAULT_OBJECTIVE_ALPHA,),
    )
    assert _lambda_alpha_grid(ours) == [(0.0, 1.0), (0.5, 1.0), (1.0, 1.0)]


def test_alpha_grid_is_swept_lambda_major() -> None:
    """The product is reg_lambda-major, so a pure α sweep is contiguous."""
    from types import SimpleNamespace

    from steering.eval.axbench_bpb_adv.gcg_bags import _lambda_alpha_grid

    ours = SimpleNamespace(reg_lambda_grid=(0.0,), objective_alpha_grid=(1.0, 0.5, 0.0))
    assert _lambda_alpha_grid(ours) == [(0.0, 1.0), (0.0, 0.5), (0.0, 0.0)]


# ─────────────────────────────────────────────────────────────────────────────
# Coefficient axis + registry
# ─────────────────────────────────────────────────────────────────────────────


def test_alpha_methods_are_registered_with_the_alpha_grid() -> None:
    """Both alpha methods exist and bind to ``ours_gcg_alpha_grid``, not the lambda one."""
    by_name = {m.name: m for m in ALL_METHODS}
    for name in ("ours_gcg_alpha_encdec", "neg_ours_gcg_alpha_encdec"):
        assert name in by_name, f"{name} is not registered"
        assert by_name[name].coefficient_grid_fn is ours_gcg_alpha_grid
        assert by_name[name].coefficient_grid_fn is not ours_gcg_lambda_grid


def test_alpha_methods_read_the_fair_family() -> None:
    """They share the FAIR bag family — the α = 1 point IS the fair GCG bag."""
    for name in ALPHA_AXIS_GCG_METHODS:
        assert precomputed_bag_method_family(name) is PrecomputedBagFamily.FAIR_GCG


def test_alpha_methods_trigger_the_fair_derive() -> None:
    """A run selecting ONLY the alpha methods must still derive fair bags.

    Teeth: omitting them from ``FAIR_GCG_METHOD_MODES`` would make such a run
    derive nothing and then fail at score time with a missing bag.
    """
    assert ALPHA_AXIS_GCG_METHODS <= set(FAIR_GCG_METHOD_MODES)
    assert FAIR_GCG_METHOD_MODES["ours_gcg_alpha_encdec"] is SteeringMode.POSITIVE
    assert FAIR_GCG_METHOD_MODES["neg_ours_gcg_alpha_encdec"] is SteeringMode.NEGATIVE


def test_lambda_methods_are_not_on_the_alpha_axis() -> None:
    """The lambda methods must stay on the lambda axis (no accidental widening)."""
    assert "ours_gcg_encdec" not in ALPHA_AXIS_GCG_METHODS
    assert "ours_gcg_roofline_encdec" not in ALPHA_AXIS_GCG_METHODS


def test_alpha_grid_factory_reads_the_alpha_grid() -> None:
    """The factory returns ``objective_alpha_grid``, never ``reg_lambda_grid``."""
    cfg = SimpleNamespace(
        ours_gcg=SimpleNamespace(
            reg_lambda_grid=(0.0, 9.0), objective_alpha_grid=(1.0, 0.5, 0.0)
        )
    )
    assert ours_gcg_alpha_grid(_TARGET_META, cfg) == [1.0, 0.5, 0.0]
    assert ours_gcg_lambda_grid(_TARGET_META, cfg) == [0.0, 9.0]


def test_alpha_methods_plot_as_their_own_family() -> None:
    """The alpha sweep is a SEPARATE plotted series from the lambda sweep.

    Their coefficients are different quantities, so one shared family would draw
    a single curve over two incomparable x-axes. Teeth: the generic ``ours_gcg``
    prefix would swallow ``ours_gcg_alpha`` if its branch came first.
    """
    alpha_family = _steering_family("ours_gcg_alpha_encdec")
    assert alpha_family == _steering_family("neg_ours_gcg_alpha_encdec")
    assert alpha_family != _steering_family("ours_gcg_encdec")
    assert alpha_family != _steering_family("feature_diffmean_encdec")
    # The lambda family must be left exactly as it was.
    assert _steering_family("ours_gcg_encdec") == "feature GCG"


def test_alpha_grid_factory_falls_back_to_the_plain_margin() -> None:
    """Without an ``ours_gcg`` capability the fallback is α = 1, NOT 0.

    α = 0 is a meaningful and very different objective, so falling back to it
    would silently redefine the method rather than degrade to the plain margin.
    """
    assert ours_gcg_alpha_grid(_TARGET_META, object()) == [
        DEFAULT_OBJECTIVE_ALPHA
    ]
