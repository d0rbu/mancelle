"""Unit tests for precomputed conditioning-bag method classification."""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.gcg_family import (
    FAIR_GCG_METHOD_MODES,
    ROOFLINE_GCG_METHOD_MODES,
    GcgFamily,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_family import (
    FAIR_FEATURE_DIFFMEAN_METHOD_MODES,
    PRECOMPUTED_BAG_METHOD_NAME_TO_FAMILY,
    ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES,
    ROOFLINE_PRECOMPUTED_BAG_FAMILIES,
    PrecomputedBagFamily,
    active_precomputed_bag_modes,
    coerce_precomputed_bag_family,
    precomputed_bag_method_family,
)


def test_precomputed_family_map_includes_gcg_and_feature_diffmean() -> None:
    expected = dict.fromkeys(FAIR_GCG_METHOD_MODES, PrecomputedBagFamily.FAIR_GCG)
    expected.update(dict.fromkeys(ROOFLINE_GCG_METHOD_MODES, PrecomputedBagFamily.ROOFLINE_GCG))
    expected.update(
        dict.fromkeys(FAIR_FEATURE_DIFFMEAN_METHOD_MODES, PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN)
    )
    expected.update(
        dict.fromkeys(ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES, PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN)
    )
    assert PRECOMPUTED_BAG_METHOD_NAME_TO_FAMILY == expected


def test_feature_diffmean_is_precomputed_bag() -> None:
    assert (
        precomputed_bag_method_family("feature_diffmean_encdec")
        is PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
    )
    assert (
        precomputed_bag_method_family("neg_feature_diffmean_encdec")
        is PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
    )
    assert (
        precomputed_bag_method_family("feature_diffmean_roofline_encdec")
        is PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN
    )
    assert (
        precomputed_bag_method_family("neg_feature_diffmean_roofline_encdec")
        is PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN
    )


def test_gcg_methods_remain_precomputed_bags() -> None:
    assert precomputed_bag_method_family("ours_gcg_encdec") is PrecomputedBagFamily.FAIR_GCG
    assert precomputed_bag_method_family("neg_ours_gcg_encdec") is PrecomputedBagFamily.FAIR_GCG
    assert (
        precomputed_bag_method_family("ours_gcg_roofline_encdec")
        is PrecomputedBagFamily.ROOFLINE_GCG
    )
    assert (
        precomputed_bag_method_family("neg_ours_gcg_roofline_encdec")
        is PrecomputedBagFamily.ROOFLINE_GCG
    )


def test_feature_diffmean_mode_map_and_activation() -> None:
    assert FAIR_FEATURE_DIFFMEAN_METHOD_MODES["feature_diffmean_encdec"] is SteeringMode.POSITIVE
    assert FAIR_FEATURE_DIFFMEAN_METHOD_MODES["neg_feature_diffmean_encdec"] is SteeringMode.NEGATIVE
    assert active_precomputed_bag_modes(None, FAIR_FEATURE_DIFFMEAN_METHOD_MODES) == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }
    assert active_precomputed_bag_modes(
        ["neg_feature_diffmean_encdec"], FAIR_FEATURE_DIFFMEAN_METHOD_MODES
    ) == {SteeringMode.NEGATIVE}
    assert active_precomputed_bag_modes(["ours_gcg_encdec"], FAIR_FEATURE_DIFFMEAN_METHOD_MODES) == set()
    assert (
        ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES["feature_diffmean_roofline_encdec"]
        is SteeringMode.POSITIVE
    )
    assert (
        ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES["neg_feature_diffmean_roofline_encdec"]
        is SteeringMode.NEGATIVE
    )
    assert active_precomputed_bag_modes(
        ["feature_diffmean_roofline_encdec"],
        ROOFLINE_FEATURE_DIFFMEAN_METHOD_MODES,
    ) == {SteeringMode.POSITIVE}


def test_roofline_precomputed_bag_method_names_include_both_families() -> None:
    assert ROOFLINE_PRECOMPUTED_BAG_FAMILIES == frozenset(
        {
            PrecomputedBagFamily.ROOFLINE_GCG,
            PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        }
    )
    # All four roofline method names classify into a roofline family.
    for name in (
        "ours_gcg_roofline_encdec",
        "neg_ours_gcg_roofline_encdec",
        "feature_diffmean_roofline_encdec",
        "neg_feature_diffmean_roofline_encdec",
    ):
        assert precomputed_bag_method_family(name) in ROOFLINE_PRECOMPUTED_BAG_FAMILIES


def test_family_values_and_gcg_coercion_are_historical_slugs() -> None:
    assert str(PrecomputedBagFamily.FAIR_GCG) == "fair"
    assert str(PrecomputedBagFamily.ROOFLINE_GCG) == "roofline"
    assert str(PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN) == "feature_diffmean"
    assert str(PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN) == "feature_diffmean_roofline"
    assert coerce_precomputed_bag_family(GcgFamily.FAIR) is PrecomputedBagFamily.FAIR_GCG
    assert coerce_precomputed_bag_family(GcgFamily.ROOFLINE) is PrecomputedBagFamily.ROOFLINE_GCG
    assert (
        coerce_precomputed_bag_family("feature_diffmean") is PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
    )
    assert (
        coerce_precomputed_bag_family("feature_diffmean_roofline")
        is PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN
    )


def test_feature_diffmean_path_slug_is_stable(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="slugs")
    enum_path = paths.ours_gcg_bag(
        7, SteeringMode.POSITIVE, 0.0, PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
    )
    str_path = paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.0, "feature_diffmean")
    assert enum_path == str_path
    assert "/feature_diffmean/" in str(enum_path)
    assert paths.ours_gcg_scored_prefixes(
        7, SteeringMode.POSITIVE, PrecomputedBagFamily.FAIR_FEATURE_DIFFMEAN
    ) == paths.ours_gcg_scored_prefixes(7, SteeringMode.POSITIVE, "feature_diffmean")


def test_roofline_feature_diffmean_path_slug_is_stable(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="slugs")
    enum_path = paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        PrecomputedBagFamily.ROOFLINE_FEATURE_DIFFMEAN,
        prefix_id="p1",
    )
    str_path = paths.ours_gcg_bag(
        7,
        SteeringMode.POSITIVE,
        0.0,
        "feature_diffmean_roofline",
        prefix_id="p1",
    )
    assert enum_path == str_path
    assert "/feature_diffmean_roofline/" in str(enum_path)


@pytest.mark.parametrize("bad", ["FeatureDiffMean", "diffmean", "gcg", ""])
def test_precomputed_family_rejects_unknown_slugs(bad: str) -> None:
    with pytest.raises(ValueError):
        coerce_precomputed_bag_family(bad)
