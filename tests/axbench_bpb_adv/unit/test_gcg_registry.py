"""Unit tests for Task 1: feature-GCG method registration and helpers.

Covers:
  - The 4 new ``ours_gcg*`` / ``neg_ours_gcg*`` methods in ``ALL_METHODS``.
  - The 4 new ``PartitionModifier`` members present in ``_MODIFIER_REGISTRY``.
  - ``assert_single_positive_bucket``: raises on ``num_buckets != 2``, returns
    ``(0, 1)`` on ``num_buckets == 2``.
  - ``resolve_k``: returns the metadata geometric means when ``top_k_pos`` /
    ``top_k_neg`` are ``None`` in config; respects explicit overrides.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.coefficient_grids import single_condition_grid
from steering.eval.axbench_base.encoder_input import (
    _MODIFIER_REGISTRY,
    PartitionModifier,
)
from steering.eval.axbench_base.method_registry import (
    ALL_METHODS,
    BaselineKind,
)
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.config import OursGcgConfig
from steering.eval.axbench_bpb_adv.gcg_config import (
    assert_single_positive_bucket,
    resolve_k,
)

# ---------------------------------------------------------------------------
# Helpers for building stub models
# ---------------------------------------------------------------------------


def _make_model(num_buckets: int, user_config: dict[str, Any] | None = None) -> Any:
    """Return a lightweight model stub with the minimum shape the helpers need."""
    meta: dict[str, Any] = {"user_config": user_config or {}}
    encoder_config = SimpleNamespace(num_buckets=num_buckets)
    encoder = SimpleNamespace(config=encoder_config)
    model = SimpleNamespace(encoder=encoder)
    model._loaded_meta_data = meta
    return model


def _geo_selection(mean: float) -> dict[str, Any]:
    """Minimal geometric feature_selection dict accepted by parse_feature_selection."""
    return {"distribution": "geometric", "mean": mean}


# ---------------------------------------------------------------------------
# New PartitionModifier members
# ---------------------------------------------------------------------------


_GCG_MODIFIERS = [
    PartitionModifier.OURS_GCG_ENCDEC,
    PartitionModifier.NEG_OURS_GCG_ENCDEC,
    PartitionModifier.OURS_GCG_ROOFLINE_ENCDEC,
    PartitionModifier.NEG_OURS_GCG_ROOFLINE_ENCDEC,
]


def test_gcg_partition_modifier_members_exist() -> None:
    """All 4 GCG PartitionModifier members are importable."""
    assert len(_GCG_MODIFIERS) == 4


def test_gcg_modifiers_are_in_registry() -> None:
    """All 4 new GCG modifiers are registered in _MODIFIER_REGISTRY."""
    for modifier in _GCG_MODIFIERS:
        assert modifier in _MODIFIER_REGISTRY, (
            f"GCG modifier {modifier!r} is missing from _MODIFIER_REGISTRY"
        )


def test_gcg_modifier_placeholders_raise_feature_gcg_error() -> None:
    """The placeholder modifier fns raise FeatureGcgError without a precomputed bag."""
    for modifier in _GCG_MODIFIERS:
        fn = _MODIFIER_REGISTRY[modifier]
        with pytest.raises(FeatureGcgError, match="precomputed conditioning bag"):
            fn({}, {})


def test_gcg_modifier_values_match_expected_strings() -> None:
    """StrEnum values follow the ours_gcg_* / neg_ours_gcg_* naming convention."""
    assert str(PartitionModifier.OURS_GCG_ENCDEC) == "ours_gcg_encdec"
    assert str(PartitionModifier.NEG_OURS_GCG_ENCDEC) == "neg_ours_gcg_encdec"
    assert str(PartitionModifier.OURS_GCG_ROOFLINE_ENCDEC) == "ours_gcg_roofline_encdec"
    assert str(PartitionModifier.NEG_OURS_GCG_ROOFLINE_ENCDEC) == "neg_ours_gcg_roofline_encdec"


def test_feature_diffmean_partition_modifiers_registered() -> None:
    modifiers = [
        PartitionModifier.FEATURE_DIFFMEAN_ENCDEC,
        PartitionModifier.NEG_FEATURE_DIFFMEAN_ENCDEC,
        PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
        PartitionModifier.NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
    ]
    for modifier in modifiers:
        assert modifier in _MODIFIER_REGISTRY
        with pytest.raises(FeatureGcgError, match="precomputed conditioning bag"):
            _MODIFIER_REGISTRY[modifier]({}, {})
    assert str(PartitionModifier.FEATURE_DIFFMEAN_ENCDEC) == "feature_diffmean_encdec"
    assert str(PartitionModifier.NEG_FEATURE_DIFFMEAN_ENCDEC) == "neg_feature_diffmean_encdec"
    assert (
        str(PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC)
        == "feature_diffmean_roofline_encdec"
    )
    assert (
        str(PartitionModifier.NEG_FEATURE_DIFFMEAN_ROOFLINE_ENCDEC)
        == "neg_feature_diffmean_roofline_encdec"
    )


@pytest.mark.parametrize(
    "modifier",
    [
        PartitionModifier.OURS_GCG_ENCDEC,
        PartitionModifier.FEATURE_DIFFMEAN_ENCDEC,
        PartitionModifier.FEATURE_DIFFMEAN_ROOFLINE_ENCDEC,
    ],
)
def test_precomputed_conditioning_modifier_uses_neutral_bag_arg(
    modifier: PartitionModifier,
) -> None:
    active, inactive = _MODIFIER_REGISTRY[modifier](
        {},
        {},
        precomputed_bag={7: 1, 11: 0},
    )

    assert active == {7: 1, 11: 0}
    assert inactive == {}


# ---------------------------------------------------------------------------
# New methods in ALL_METHODS
# ---------------------------------------------------------------------------


_EXPECTED_GCG_METHOD_NAMES = {
    "ours_gcg_encdec",
    "neg_ours_gcg_encdec",
    "ours_gcg_roofline_encdec",
    "neg_ours_gcg_roofline_encdec",
}


def test_gcg_methods_present_in_all_methods() -> None:
    """All 4 GCG method names appear in ALL_METHODS."""
    names = {m.name for m in ALL_METHODS}
    for name in _EXPECTED_GCG_METHOD_NAMES:
        assert name in names, f"GCG method {name!r} missing from ALL_METHODS"


def test_gcg_methods_have_ours_baseline() -> None:
    gcg_methods = [m for m in ALL_METHODS if m.name in _EXPECTED_GCG_METHOD_NAMES]
    assert len(gcg_methods) == 4
    for m in gcg_methods:
        assert m.baseline == BaselineKind.OURS, (
            f"{m.name}: expected baseline=OURS, got {m.baseline!r}"
        )


def test_gcg_methods_have_encdec_target() -> None:
    gcg_methods = [m for m in ALL_METHODS if m.name in _EXPECTED_GCG_METHOD_NAMES]
    for m in gcg_methods:
        assert m.target_model == "encdec", (
            f"{m.name}: expected target_model='encdec', got {m.target_model!r}"
        )


def test_gcg_methods_have_correct_modes() -> None:
    positive_gcg = {
        m.name
        for m in ALL_METHODS
        if m.name
        in {
            "ours_gcg_encdec",
            "ours_gcg_roofline_encdec",
        }
    }
    negative_gcg = {
        m.name
        for m in ALL_METHODS
        if m.name
        in {
            "neg_ours_gcg_encdec",
            "neg_ours_gcg_roofline_encdec",
        }
    }
    assert len(positive_gcg) == 2
    assert len(negative_gcg) == 2
    for m in ALL_METHODS:
        if m.name in positive_gcg:
            assert m.mode == SteeringMode.POSITIVE
        if m.name in negative_gcg:
            assert m.mode == SteeringMode.NEGATIVE


def test_gcg_methods_have_non_none_partition_modifier() -> None:
    gcg_methods = [m for m in ALL_METHODS if m.name in _EXPECTED_GCG_METHOD_NAMES]
    for m in gcg_methods:
        assert m.partition_modifier_name is not None, (
            f"{m.name}: partition_modifier_name must not be None"
        )
        assert isinstance(m.partition_modifier_name, PartitionModifier), (
            f"{m.name}: partition_modifier_name must be a PartitionModifier member"
        )


def test_gcg_methods_partition_modifier_in_registry() -> None:
    gcg_methods = [m for m in ALL_METHODS if m.name in _EXPECTED_GCG_METHOD_NAMES]
    for m in gcg_methods:
        assert m.partition_modifier_name in _MODIFIER_REGISTRY, (
            f"{m.name}: partition_modifier_name {m.partition_modifier_name!r} "
            "not in _MODIFIER_REGISTRY"
        )


def test_gcg_methods_do_not_require_negative_sampling() -> None:
    """GCG placeholder methods don't manipulate the inactive partition."""
    gcg_methods = [m for m in ALL_METHODS if m.name in _EXPECTED_GCG_METHOD_NAMES]
    for m in gcg_methods:
        assert not m.requires_negative_sampling, (
            f"{m.name}: requires_negative_sampling should be False"
        )


def test_feature_diffmean_methods_registered_as_single_condition_ours() -> None:
    methods = {
        method.name: method
        for method in ALL_METHODS
        if method.name
        in {
            "feature_diffmean_encdec",
            "neg_feature_diffmean_encdec",
            "feature_diffmean_roofline_encdec",
            "neg_feature_diffmean_roofline_encdec",
        }
    }
    assert set(methods) == {
        "feature_diffmean_encdec",
        "neg_feature_diffmean_encdec",
        "feature_diffmean_roofline_encdec",
        "neg_feature_diffmean_roofline_encdec",
    }
    assert methods["feature_diffmean_encdec"].mode is SteeringMode.POSITIVE
    assert methods["neg_feature_diffmean_encdec"].mode is SteeringMode.NEGATIVE
    assert methods["feature_diffmean_roofline_encdec"].mode is SteeringMode.POSITIVE
    assert methods["neg_feature_diffmean_roofline_encdec"].mode is SteeringMode.NEGATIVE
    for method in methods.values():
        assert method.baseline == BaselineKind.OURS
        assert method.target_model == "encdec"
        assert method.requires_negative_sampling is False
        assert method.coefficient_grid_fn is single_condition_grid


# ---------------------------------------------------------------------------
# assert_single_positive_bucket
# ---------------------------------------------------------------------------


def test_assert_single_positive_bucket_raises_on_num_buckets_3() -> None:
    """num_buckets=3 violates the single-positive-bucket assumption."""
    model = _make_model(num_buckets=3)
    with pytest.raises(FeatureGcgError):
        assert_single_positive_bucket(model)


def test_assert_single_positive_bucket_raises_on_num_buckets_1() -> None:
    """num_buckets=1 also violates (no room for a positive AND inactive bucket)."""
    model = _make_model(num_buckets=1)
    with pytest.raises(FeatureGcgError):
        assert_single_positive_bucket(model)


def test_assert_single_positive_bucket_returns_0_1_on_num_buckets_2() -> None:
    """num_buckets=2 satisfies the assumption: positive_bucket=0, negative_bucket=1."""
    model = _make_model(num_buckets=2)
    result = assert_single_positive_bucket(model)
    assert result == (0, 1)


def test_assert_single_positive_bucket_missing_meta_raises() -> None:
    """Model without _loaded_meta_data raises FeatureGcgError."""
    model = SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace(num_buckets=2)))
    with pytest.raises(FeatureGcgError):
        assert_single_positive_bucket(model)


def test_assert_single_positive_bucket_missing_encoder_raises() -> None:
    """Model without encoder.config.num_buckets raises FeatureGcgError."""
    model = SimpleNamespace(_loaded_meta_data={"user_config": {}})
    with pytest.raises(FeatureGcgError):
        assert_single_positive_bucket(model)


# ---------------------------------------------------------------------------
# resolve_k
# ---------------------------------------------------------------------------


def test_resolve_k_reads_metadata_means() -> None:
    """When top_k_pos/top_k_neg are None, resolve_k reads from metadata means."""
    user_config = {
        "feature_selection": _geo_selection(mean=10.0),
        "negative_selection": _geo_selection(mean=10.0),
    }
    model = _make_model(num_buckets=2, user_config=user_config)
    cfg = OursGcgConfig()  # top_k_pos=None, top_k_neg=None by default
    pos, neg = resolve_k(model, cfg)
    assert pos == 10
    assert neg == 10


def test_resolve_k_pos_override_respected() -> None:
    """top_k_pos=5 overrides the metadata mean for positive selection."""
    user_config = {
        "feature_selection": _geo_selection(mean=10.0),
        "negative_selection": _geo_selection(mean=10.0),
    }
    model = _make_model(num_buckets=2, user_config=user_config)
    cfg = OursGcgConfig(top_k_pos=5)
    pos, neg = resolve_k(model, cfg)
    assert pos == 5
    assert neg == 10


def test_resolve_k_neg_override_respected() -> None:
    """top_k_neg=3 overrides the metadata mean for negative selection."""
    user_config = {
        "feature_selection": _geo_selection(mean=10.0),
        "negative_selection": _geo_selection(mean=10.0),
    }
    model = _make_model(num_buckets=2, user_config=user_config)
    cfg = OursGcgConfig(top_k_neg=3)
    pos, neg = resolve_k(model, cfg)
    assert pos == 10
    assert neg == 3


def test_resolve_k_both_overrides_respected() -> None:
    """Both top_k_pos and top_k_neg can be overridden independently."""
    model = _make_model(num_buckets=2)
    cfg = OursGcgConfig(top_k_pos=7, top_k_neg=4)
    pos, neg = resolve_k(model, cfg)
    assert pos == 7
    assert neg == 4


def test_resolve_k_raises_when_metadata_missing() -> None:
    """No selection metadata (and no override) → resolve_k RAISES: k must come
    from the checkpoint's own training selection distribution, never a default."""
    model = _make_model(num_buckets=2, user_config={})
    cfg = OursGcgConfig()
    with pytest.raises(FeatureGcgError, match="feature_selection"):
        resolve_k(model, cfg)


def test_resolve_k_raises_when_negative_selection_absent() -> None:
    """feature_selection present but negative_selection absent → RAISES for the
    negative dimension (no silent default-to-10)."""
    user_config = {"feature_selection": _geo_selection(mean=8.0)}
    model = _make_model(num_buckets=2, user_config=user_config)
    cfg = OursGcgConfig()
    with pytest.raises(FeatureGcgError, match="negative_selection"):
        resolve_k(model, cfg)


def test_resolve_k_different_pos_neg_means() -> None:
    """Positive and negative means are read independently."""
    user_config = {
        "feature_selection": _geo_selection(mean=12.0),
        "negative_selection": _geo_selection(mean=6.0),
    }
    model = _make_model(num_buckets=2, user_config=user_config)
    cfg = OursGcgConfig()
    pos, neg = resolve_k(model, cfg)
    assert pos == 12
    assert neg == 6


def test_resolve_k_raises_when_no_loaded_meta_data() -> None:
    """Model without _loaded_meta_data → resolve_k RAISES (no silent default-10)."""
    model = SimpleNamespace()  # no _loaded_meta_data at all
    cfg = OursGcgConfig()
    with pytest.raises(FeatureGcgError, match="no _loaded_meta_data"):
        resolve_k(model, cfg)
