"""Unit tests for the 35-method registry (Task 30).

The registry encodes the AxBench-Base experiment matrix:
17 positive-mode methods + 18 negative-mode methods = 35 total.
Each entry is a frozen ``Method`` dataclass that pairs a baseline type
with a target model and the coefficient grid factory + partition
modifier the orchestrator should dispatch through at generation time.

These tests enforce the COUNT and CATEGORY invariants and pin the exact
35-method survivor set (name, mode, baseline, target, negative-sampling
flag, modifier, grid fn) so any change to the registry is caught. See
the comment at the top of ``method_registry.py`` for the enumeration.
"""

from __future__ import annotations

import dataclasses
import random
from collections.abc import Callable

import pytest

from steering.eval.axbench_base import method_registry
from steering.eval.axbench_base.coefficient_grids import (
    active_bucket_grid,
    diffmean_scalar_grid,
    ours_gcg_alpha_grid,
    ours_gcg_lambda_grid,
    single_condition_grid,
)
from steering.eval.axbench_base.encoder_input import (
    _MODIFIER_REGISTRY,
    PartitionModifier,
    build_mode_aware_encoder_input,
)
from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    BaselineKind,
    Method,
    filter_methods,
    get_method_by_name,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    SteeringMode,
    TargetModelMetadata,
)

_ALLOWED_BASELINES: set[BaselineKind] = set(BaselineKind)
_ALLOWED_TARGETS = {"decoder_only", "encdec"}
_ALLOWED_MODES: set[SteeringMode] = set(SteeringMode)
_ALLOWED_GRID_FNS = {
    diffmean_scalar_grid,
    single_condition_grid,
    active_bucket_grid,
    ours_gcg_lambda_grid,
    # Class-margin alpha sweep: a SEPARATE factory from the lambda one because a
    # method binds to exactly one coefficient axis.
    ours_gcg_alpha_grid,
}


# ---------------------------------------------------------------------------
# Cardinality / structural invariants
# ---------------------------------------------------------------------------


def test_all_35_methods_registered() -> None:
    """25 original + 4 feature-GCG + 2 class-margin-alpha + 4 feature-DiffMean = 35."""
    assert len(ALL_METHODS) == 35


def test_control_methods_registered() -> None:
    """The 4 no-steering control methods are first-class registry entries."""
    names = {m.name for m in ALL_METHODS}
    for name in (
        "control_decoder_only",
        "neg_control_decoder_only",
        "control_encdec",
        "neg_control_encdec",
    ):
        assert name in names
    controls = [m for m in ALL_METHODS if m.baseline == BaselineKind.CONTROL]
    assert len(controls) == 4
    assert all(m.partition_modifier_name is None for m in controls)
    assert all(not m.requires_negative_sampling for m in controls)
    assert {m.target_model for m in controls} == {"decoder_only", "encdec"}


def test_no_duplicate_names() -> None:
    names = [m.name for m in ALL_METHODS]
    assert len(names) == len(set(names))


def test_positive_methods_have_17_entries() -> None:
    """12 original + 2 feature-GCG + 1 alpha + 2 feature-DiffMean positive = 17."""
    positive = [m for m in ALL_METHODS if m.mode == "positive"]
    assert len(positive) == 17


def test_negative_methods_have_18_entries() -> None:
    """13 original + 2 feature-GCG + 1 alpha + 2 feature-DiffMean negative = 18."""
    negative = [m for m in ALL_METHODS if m.mode == "negative"]
    assert len(negative) == 18


def test_method_is_frozen_dataclass() -> None:
    assert dataclasses.is_dataclass(Method)
    params = getattr(Method, "__dataclass_params__", None)
    assert params is not None
    assert params.frozen is True


def test_method_frozen_assignment_raises() -> None:
    """A frozen dataclass should raise FrozenInstanceError on attribute set.

    We invoke ``__setattr__`` via the class (rather than ``m.name = "x"``)
    because ``ty`` correctly flags the latter as ``invalid-assignment``;
    this form is equivalent at runtime — ``dataclasses`` raises from its
    own ``__setattr__`` — without lying to the type checker.
    """
    m = ALL_METHODS[0]
    with pytest.raises(dataclasses.FrozenInstanceError):
        type(m).__setattr__(m, "name", "x")


def test_method_field_types() -> None:
    field_names = {f.name for f in dataclasses.fields(Method)}
    assert field_names == {
        "name",
        "mode",
        "baseline",
        "target_model",
        "requires_negative_sampling",
        "coefficient_grid_fn",
        "partition_modifier_name",
    }


def test_every_method_has_valid_field_values() -> None:
    for m in ALL_METHODS:
        assert m.mode in _ALLOWED_MODES, f"{m.name}: bad mode {m.mode!r}"
        assert m.baseline in _ALLOWED_BASELINES, f"{m.name}: bad baseline {m.baseline!r}"
        assert m.target_model in _ALLOWED_TARGETS, f"{m.name}: bad target_model {m.target_model!r}"
        assert isinstance(m.requires_negative_sampling, bool)
        assert callable(m.coefficient_grid_fn)
        assert m.partition_modifier_name is None or isinstance(
            m.partition_modifier_name,
            PartitionModifier,
        )


def test_method_registry_uses_enums() -> None:
    """Every ``ALL_METHODS`` entry must expose enum-typed fields for
    ``mode``, ``baseline``, and (when set) ``partition_modifier_name``.

    Why: StrEnum makes bare-string comparisons keep working, but the
    registry itself should be strict about emitting enum members so
    static analysis and ``isinstance`` checks catch drift.
    """
    for m in ALL_METHODS:
        assert isinstance(m.mode, SteeringMode), (
            f"{m.name}: mode is not a SteeringMode member ({type(m.mode).__name__})"
        )
        assert isinstance(m.baseline, BaselineKind), (
            f"{m.name}: baseline is not a BaselineKind member ({type(m.baseline).__name__})"
        )
        if m.partition_modifier_name is not None:
            assert isinstance(m.partition_modifier_name, PartitionModifier), (
                f"{m.name}: partition_modifier_name is not a PartitionModifier "
                f"member ({type(m.partition_modifier_name).__name__})"
            )


# ---------------------------------------------------------------------------
# Negative sampling flag invariants
# ---------------------------------------------------------------------------


def test_requires_negative_sampling_flag_count_is_three() -> None:
    flagged = [m for m in ALL_METHODS if m.requires_negative_sampling]
    assert len(flagged) == 3


def test_requires_negative_sampling_methods_are_negative_mode() -> None:
    for m in ALL_METHODS:
        if m.requires_negative_sampling:
            assert m.mode == "negative"
            assert m.target_model == "encdec"
            assert m.baseline == "ours"


def test_requires_negative_sampling_methods_use_negative_modifiers() -> None:
    """The 3 flagged methods all use negative-mode partition modifiers
    that touch the inactive partition (add/replace_random/replace_all).
    """
    expected_modifiers = {
        "neg_ours_add_to_negatives",
        "neg_ours_replace_random_negatives",
        "neg_ours_replace_all_negatives",
    }
    flagged = {m.partition_modifier_name for m in ALL_METHODS if m.requires_negative_sampling}
    assert flagged == expected_modifiers


# ---------------------------------------------------------------------------
# Coefficient grid dispatch
# ---------------------------------------------------------------------------


def test_method_coefficient_grid_fn_dispatch() -> None:
    """Every method's coefficient_grid_fn is one of the registered grids."""
    for m in ALL_METHODS:
        assert m.coefficient_grid_fn in _ALLOWED_GRID_FNS, (
            f"{m.name}: unknown grid fn {m.coefficient_grid_fn!r}"
        )


def test_diffmean_methods_use_diffmean_scalar_grid() -> None:
    for m in ALL_METHODS:
        if m.baseline == "diffmean":
            assert m.coefficient_grid_fn is diffmean_scalar_grid


def test_linear_probe_methods_use_diffmean_scalar_grid() -> None:
    for m in ALL_METHODS:
        if m.baseline == "linear_probe":
            assert m.coefficient_grid_fn is diffmean_scalar_grid


def test_lora_methods_use_single_condition_grid() -> None:
    for m in ALL_METHODS:
        if m.baseline == "lora":
            assert m.coefficient_grid_fn is single_condition_grid


def test_prompting_methods_use_single_condition_grid() -> None:
    for m in ALL_METHODS:
        if m.baseline == "prompting":
            assert m.coefficient_grid_fn is single_condition_grid


def test_active_bucket_grid_is_used_by_some_ours_methods() -> None:
    """At least one ``ours`` method should sweep over active buckets."""
    bucket_methods = [m for m in ALL_METHODS if m.coefficient_grid_fn is active_bucket_grid]
    assert len(bucket_methods) >= 1
    for m in bucket_methods:
        assert m.baseline == "ours"


# ---------------------------------------------------------------------------
# Partition modifier dispatch
# ---------------------------------------------------------------------------


def test_ours_methods_have_partition_modifier_name() -> None:
    for m in ALL_METHODS:
        if m.baseline == "ours":
            assert m.partition_modifier_name is not None, (
                f"{m.name}: ours method must declare a partition_modifier_name"
            )


def test_non_ours_methods_have_no_partition_modifier() -> None:
    for m in ALL_METHODS:
        if m.baseline != "ours":
            assert m.partition_modifier_name is None, (
                f"{m.name}: only ours methods should declare a "
                f"partition_modifier_name (got {m.partition_modifier_name!r})"
            )


def test_partition_modifier_name_matches_registry_for_ours_methods() -> None:
    """Every ours method's modifier name must be a key in _MODIFIER_REGISTRY."""
    for m in ALL_METHODS:
        if m.baseline == "ours":
            assert m.partition_modifier_name in _MODIFIER_REGISTRY, (
                f"{m.name}: partition_modifier_name {m.partition_modifier_name!r} is not registered"
            )


def test_every_modifier_in_registry_is_referenced_by_some_method() -> None:
    """Inverse of :func:`test_partition_modifier_name_matches_registry_for_ours_methods`.

    Every entry in ``_MODIFIER_REGISTRY`` should be referenced by at
    least one method in ``ALL_METHODS``. A modifier no method points at
    is dead code — refactors that rename or remove a method should also
    drop the orphaned modifier, and this test catches the drift.
    """
    referenced: set[str] = {
        m.partition_modifier_name for m in ALL_METHODS if m.partition_modifier_name is not None
    }
    orphans = set(_MODIFIER_REGISTRY.keys()) - referenced
    assert not orphans, (
        f"Modifiers in _MODIFIER_REGISTRY have no method referencing them: {sorted(orphans)}"
    )


def test_neg_prefix_implies_negative_mode() -> None:
    """Any method whose name starts with ``neg_`` MUST be negative mode.

    Registry construction derives names from mode-prefix so the
    invariant holds by construction today. A future hand-added entry
    could violate this silently; pinning the invariant here catches
    the drift before it reaches production.
    """
    for m in ALL_METHODS:
        if m.name.startswith("neg_"):
            assert m.mode == "negative", (
                f"{m.name!r} has ``neg_`` prefix but mode={m.mode!r}; negative-mode methods only"
            )


def test_positive_ours_methods_use_positive_modifiers() -> None:
    """Positive-mode ours methods must use a modifier whose key starts with
    ``ours_`` or a positive feature-DiffMean bag modifier.
    """
    positive_feature_diffmean_modifiers = {
        "feature_diffmean_encdec",
        "feature_diffmean_roofline_encdec",
    }
    for m in ALL_METHODS:
        if m.mode == "positive" and m.baseline == "ours":
            assert m.partition_modifier_name is not None
            modifier = str(m.partition_modifier_name)
            assert (
                modifier.startswith("ours_") or modifier in positive_feature_diffmean_modifiers
            ), (
                f"{m.name}: positive-mode ours method should use a positive-mode "
                f"modifier (got {m.partition_modifier_name!r})"
            )


def test_negative_ours_methods_use_appropriate_modifiers() -> None:
    """Negative-mode ours methods may use either a dedicated negative modifier
    (``neg_ours_*``) or one of the feature-GCG Phase-2 placeholder modifiers.
    """
    allowed_negative_modifiers = {
        "neg_ours_remove",
        "neg_ours_add_to_negatives",
        "neg_ours_replace_random_negatives",
        "neg_ours_replace_all_negatives",
        # feature-GCG Phase-2 (Task 1)
        "neg_ours_gcg_encdec",
        "neg_ours_gcg_roofline_encdec",
        # Same fair bags, class-margin alpha as the coefficient axis.
        "neg_ours_gcg_alpha_encdec",
        "neg_feature_diffmean_encdec",
        "neg_feature_diffmean_roofline_encdec",
    }
    for m in ALL_METHODS:
        if m.mode == "negative" and m.baseline == "ours":
            assert m.partition_modifier_name in allowed_negative_modifiers, (
                f"{m.name}: negative-mode ours method uses unexpected modifier "
                f"{m.partition_modifier_name!r}"
            )


# ---------------------------------------------------------------------------
# Target-model invariants
# ---------------------------------------------------------------------------


def test_ours_methods_are_encdec_only() -> None:
    """``ours`` baselines need an encoder; they cannot run on a decoder-only target."""
    for m in ALL_METHODS:
        if m.baseline == "ours":
            assert m.target_model == "encdec"


def test_lora_methods_are_decoder_only() -> None:
    """LoRA only writes ``baselines/lora/decoder_only/...`` so the
    encdec branch is not implemented.
    """
    for m in ALL_METHODS:
        if m.baseline == "lora":
            assert m.target_model == "decoder_only"


# ---------------------------------------------------------------------------
# get_method_by_name
# ---------------------------------------------------------------------------


def test_method_dispatch_by_name() -> None:
    """``get_method_by_name`` returns the unique method matching ``name``."""
    sample = ALL_METHODS[0]
    fetched = get_method_by_name(sample.name)
    assert fetched is sample


def test_get_method_by_name_finds_diffmean_decoder_only() -> None:
    m = get_method_by_name("diffmean_decoder_only")
    assert m.baseline == "diffmean"
    assert m.target_model == "decoder_only"
    assert m.mode == "positive"


def test_get_method_by_name_unknown_raises_axbench_base_error() -> None:
    with pytest.raises(AxBenchBaseError, match="unknown method"):
        get_method_by_name("totally_made_up_method")


def test_get_method_by_name_finds_every_registered_method() -> None:
    for m in ALL_METHODS:
        assert get_method_by_name(m.name) is m


# ---------------------------------------------------------------------------
# filter_methods
# ---------------------------------------------------------------------------


def test_filter_methods_no_filters_returns_all() -> None:
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=[],
        target_model_has_negative_sampling=True,
    )
    assert out == list(ALL_METHODS)


def test_enabled_methods_filter_keeps_only_named() -> None:
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=["diffmean_decoder_only"],
        disabled_methods=[],
        target_model_has_negative_sampling=True,
    )
    assert len(out) == 1
    assert out[0].name == "diffmean_decoder_only"


def test_enabled_methods_filter_with_multiple_names() -> None:
    enabled = ["diffmean_decoder_only", "linear_probe_encdec"]
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=enabled,
        disabled_methods=[],
        target_model_has_negative_sampling=True,
    )
    assert {m.name for m in out} == set(enabled)


def test_enabled_methods_filter_empty_list_keeps_nothing() -> None:
    """An explicit empty enabled list means: keep nothing."""
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=[],
        disabled_methods=[],
        target_model_has_negative_sampling=True,
    )
    assert out == []


def test_disabled_methods_filter_removes_named() -> None:
    excluded = "diffmean_decoder_only"
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=[excluded],
        target_model_has_negative_sampling=True,
    )
    assert all(m.name != excluded for m in out)
    assert len(out) == len(ALL_METHODS) - 1


def test_disabled_methods_filter_removes_multiple() -> None:
    excluded = [m.name for m in ALL_METHODS[:3]]
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=excluded,
        target_model_has_negative_sampling=True,
    )
    assert {m.name for m in out}.isdisjoint(excluded)
    assert len(out) == len(ALL_METHODS) - 3


def test_disabled_methods_filter_rejects_unknown_name() -> None:
    with pytest.raises(AxBenchBaseError, match="typo_method"):
        filter_methods(
            ALL_METHODS,
            enabled_methods=None,
            disabled_methods=["typo_method"],
            target_model_has_negative_sampling=True,
        )


def test_target_model_skips_negative_sampling_methods_d12() -> None:
    """When the target model lacks negative sampling, the 3 flagged
    methods must be removed.
    """
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=[],
        target_model_has_negative_sampling=False,
    )
    assert len(out) == len(ALL_METHODS) - 3
    for m in out:
        assert m.requires_negative_sampling is False


def test_target_model_keeps_negative_sampling_methods_d13() -> None:
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=[],
        target_model_has_negative_sampling=True,
    )
    flagged_in = [m for m in out if m.requires_negative_sampling]
    assert len(flagged_in) == 3


def test_filter_order_enabled_then_disabled_then_negative_sampling() -> None:
    """All three filters compose: enabled narrows to a set, disabled
    drops one, and negative_sampling=False removes any flagged remaining.
    """
    enabled = [m.name for m in ALL_METHODS]  # everything
    disabled = ["diffmean_decoder_only"]
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=enabled,
        disabled_methods=disabled,
        target_model_has_negative_sampling=False,
    )
    assert "diffmean_decoder_only" not in {m.name for m in out}
    for m in out:
        assert m.requires_negative_sampling is False
    assert len(out) == len(ALL_METHODS) - 1 - 3


def test_filter_returns_new_list_does_not_mutate_input() -> None:
    snapshot = list(ALL_METHODS)
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=["diffmean_decoder_only"],
        target_model_has_negative_sampling=False,
    )
    assert ALL_METHODS == snapshot
    assert out is not ALL_METHODS


def test_filter_preserves_method_order() -> None:
    out = filter_methods(
        ALL_METHODS,
        enabled_methods=None,
        disabled_methods=[],
        target_model_has_negative_sampling=True,
    )
    assert out == list(ALL_METHODS)


# ---------------------------------------------------------------------------
# Hashability / module-level constants
# ---------------------------------------------------------------------------


def test_methods_are_hashable() -> None:
    """A frozen dataclass with hashable fields should land in a set cleanly."""
    s = set(ALL_METHODS)
    assert len(s) == len(ALL_METHODS)


def test_all_methods_is_module_level_list() -> None:
    assert isinstance(ALL_METHODS, list)
    assert all(isinstance(m, Method) for m in ALL_METHODS)


def test_method_registry_is_pure_data_no_class_methods() -> None:
    """Helpers should be top-level functions, not methods of Method."""
    assert isinstance(filter_methods, Callable)
    assert isinstance(get_method_by_name, Callable)
    method_methods = {
        attr
        for attr in dir(Method)
        if not attr.startswith("_") and callable(getattr(Method, attr, None))
    }
    assert method_methods == set()


def test_module_exports_expected_public_names() -> None:
    """Sanity check that key names are reachable from the package."""
    assert hasattr(method_registry, "ALL_METHODS")
    assert hasattr(method_registry, "Method")
    assert hasattr(method_registry, "filter_methods")
    assert hasattr(method_registry, "get_method_by_name")


# ---------------------------------------------------------------------------
# Integration: Method dataclass dispatches through encoder_input orchestrator
# ---------------------------------------------------------------------------


def test_method_dispatches_via_partition_modifier_name() -> None:
    """A real Method (with ``name != partition_modifier_name``) should
    dispatch through the encoder_input orchestrator using
    ``partition_modifier_name``.

    This is the bridging contract between Task 30 and Task 29: the
    Method's ``name`` is a human-readable identifier (e.g.,
    ``ours_extract_and_add_encdec``) that's distinct from the
    modifier registry key (``ours_extract_and_add``). The
    orchestrator must look up by the modifier key, NOT the method name.
    """
    method = get_method_by_name("ours_extract_and_add_encdec")
    assert method.name != method.partition_modifier_name
    assert method.partition_modifier_name == "ours_extract_and_add"

    meta = TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 2: 3},
        inactive={},
        top_k_concept_fids=[],
        coefficient=0,
        target_model_meta=meta,
        inactive_sampler=None,
        rng=random.Random(0),
    )
    # With no top-k concepts and coefficient 0, ours_extract_and_add
    # passes the active partition through verbatim.
    assert out == {1: 2, 2: 3}


def test_baseline_method_returns_empty_encoder_input() -> None:
    """Baseline methods (no partition_modifier_name) should short-circuit
    to an empty encoder input, regardless of the active partition.
    """
    method = get_method_by_name("diffmean_decoder_only")
    assert method.partition_modifier_name is None

    meta = TargetModelMetadata(
        num_buckets=None,
        n_sae_features=None,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )
    out = build_mode_aware_encoder_input(
        method=method,
        active={1: 2, 2: 3},
        inactive={},
        top_k_concept_fids=[100],
        coefficient=4,
        target_model_meta=meta,
        inactive_sampler=None,
        rng=random.Random(0),
    )
    assert out == {}


# ---------------------------------------------------------------------------
# Exact survivor set (ETC removal, #601) + loud-failure guards
# ---------------------------------------------------------------------------

# (name, mode, baseline, target_model, requires_negative_sampling,
#  partition_modifier_name, coefficient_grid_fn.__name__)
_SURVIVOR_FINGERPRINT: frozenset[
    tuple[str, str, str, str, bool, str | None, str]
] = frozenset(
    {
        ("diffmean_decoder_only", "positive", "diffmean", "decoder_only", False, None, "diffmean_scalar_grid"),
        ("diffmean_encdec", "positive", "diffmean", "encdec", False, None, "diffmean_scalar_grid"),
        ("linear_probe_decoder_only", "positive", "linear_probe", "decoder_only", False, None, "diffmean_scalar_grid"),
        ("linear_probe_encdec", "positive", "linear_probe", "encdec", False, None, "diffmean_scalar_grid"),
        ("lora_decoder_only", "positive", "lora", "decoder_only", False, None, "single_condition_grid"),
        ("prompting_decoder_only", "positive", "prompting", "decoder_only", False, None, "single_condition_grid"),
        ("prompting_encdec", "positive", "prompting", "encdec", False, None, "single_condition_grid"),
        ("control_decoder_only", "positive", "control", "decoder_only", False, None, "single_condition_grid"),
        ("control_encdec", "positive", "control", "encdec", False, None, "single_condition_grid"),
        ("ours_extract_and_add_encdec", "positive", "ours", "encdec", False, "ours_extract_and_add", "active_bucket_grid"),
        ("ours_replace_random_encdec", "positive", "ours", "encdec", False, "ours_replace_random", "active_bucket_grid"),
        ("ours_replace_all_encdec", "positive", "ours", "encdec", False, "ours_replace_all", "active_bucket_grid"),
        ("ours_gcg_encdec", "positive", "ours", "encdec", False, "ours_gcg_encdec", "ours_gcg_lambda_grid"),
        ("ours_gcg_alpha_encdec", "positive", "ours", "encdec", False, "ours_gcg_alpha_encdec", "ours_gcg_alpha_grid"),
        ("ours_gcg_roofline_encdec", "positive", "ours", "encdec", False, "ours_gcg_roofline_encdec", "ours_gcg_lambda_grid"),
        ("feature_diffmean_encdec", "positive", "ours", "encdec", False, "feature_diffmean_encdec", "single_condition_grid"),
        ("feature_diffmean_roofline_encdec", "positive", "ours", "encdec", False, "feature_diffmean_roofline_encdec", "single_condition_grid"),
        ("neg_diffmean_decoder_only", "negative", "diffmean", "decoder_only", False, None, "diffmean_scalar_grid"),
        ("neg_diffmean_encdec", "negative", "diffmean", "encdec", False, None, "diffmean_scalar_grid"),
        ("neg_linear_probe_decoder_only", "negative", "linear_probe", "decoder_only", False, None, "diffmean_scalar_grid"),
        ("neg_linear_probe_encdec", "negative", "linear_probe", "encdec", False, None, "diffmean_scalar_grid"),
        ("neg_lora_decoder_only", "negative", "lora", "decoder_only", False, None, "single_condition_grid"),
        ("neg_prompting_decoder_only", "negative", "prompting", "decoder_only", False, None, "single_condition_grid"),
        ("neg_prompting_encdec", "negative", "prompting", "encdec", False, None, "single_condition_grid"),
        ("neg_control_decoder_only", "negative", "control", "decoder_only", False, None, "single_condition_grid"),
        ("neg_control_encdec", "negative", "control", "encdec", False, None, "single_condition_grid"),
        ("neg_ours_remove_encdec", "negative", "ours", "encdec", False, "neg_ours_remove", "single_condition_grid"),
        ("neg_ours_add_to_negatives_encdec", "negative", "ours", "encdec", True, "neg_ours_add_to_negatives", "single_condition_grid"),
        ("neg_ours_replace_random_negatives_encdec", "negative", "ours", "encdec", True, "neg_ours_replace_random_negatives", "single_condition_grid"),
        ("neg_ours_replace_all_negatives_encdec", "negative", "ours", "encdec", True, "neg_ours_replace_all_negatives", "single_condition_grid"),
        ("neg_ours_gcg_encdec", "negative", "ours", "encdec", False, "neg_ours_gcg_encdec", "ours_gcg_lambda_grid"),
        ("neg_ours_gcg_alpha_encdec", "negative", "ours", "encdec", False, "neg_ours_gcg_alpha_encdec", "ours_gcg_alpha_grid"),
        ("neg_ours_gcg_roofline_encdec", "negative", "ours", "encdec", False, "neg_ours_gcg_roofline_encdec", "ours_gcg_lambda_grid"),
        ("neg_feature_diffmean_encdec", "negative", "ours", "encdec", False, "neg_feature_diffmean_encdec", "single_condition_grid"),
        ("neg_feature_diffmean_roofline_encdec", "negative", "ours", "encdec", False, "neg_feature_diffmean_roofline_encdec", "single_condition_grid"),
    }
)


def _fingerprint(m: Method) -> tuple[str, str, str, str, bool, str | None, str]:
    modifier = m.partition_modifier_name
    modifier_value = modifier.value if hasattr(modifier, "value") else modifier
    grid_name = getattr(m.coefficient_grid_fn, "__name__", None)
    assert isinstance(grid_name, str)
    return (
        m.name,
        m.mode.value,
        m.baseline.value,
        m.target_model,
        m.requires_negative_sampling,
        modifier_value,
        grid_name,
    )


def test_registry_matches_survivor_golden_after_etc_removal() -> None:
    """The 35 registered methods are the post-#601 survivor set plus #593.

    Pins that removing extract-then-condition perturbed NOTHING about any
    other method's name, mode, baseline, target, negative-sampling flag,
    partition modifier, or coefficient grid, while allowing the four
    Feature-DiffMean methods added by #593.
    """
    assert frozenset(_fingerprint(m) for m in ALL_METHODS) == _SURVIVOR_FINGERPRINT
    assert len(ALL_METHODS) == 35


@pytest.mark.parametrize(
    "etc_name",
    [
        "ours_extract_then_condition_encdec",
        "neg_ours_extract_then_condition_encdec",
    ],
)
def test_removed_etc_names_are_unknown_methods(etc_name: str) -> None:
    """The removed ETC method names now fail loudly as unknown methods."""
    with pytest.raises(AxBenchBaseError, match="unknown method"):
        get_method_by_name(etc_name)


@pytest.mark.parametrize(
    "etc_name",
    [
        "ours_extract_then_condition_encdec",
        "neg_ours_extract_then_condition_encdec",
    ],
)
def test_filter_methods_rejects_removed_etc_in_disabled(etc_name: str) -> None:
    """Disabling a removed ETC name raises the unknown-disabled-method error
    (rather than silently succeeding)."""
    with pytest.raises(AxBenchBaseError, match="unknown disabled method"):
        filter_methods(
            ALL_METHODS,
            enabled_methods=None,
            disabled_methods=[etc_name],
            target_model_has_negative_sampling=True,
        )
