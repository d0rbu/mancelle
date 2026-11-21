"""Unit teeth for the live feature-GCG family and activation contracts.

Pins (1) the family-specific mode maps, (2) roofline methods with general_val
enabled, and (3) the ``paths`` seams' family coercion — enum and exact strings produce
byte-identical historical slugs; ANY other spelling raises instead of silently
minting a new directory the loader would later report as a missing artifact.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    GeneralValBpbConfig,
    validate_config,
)
from steering.eval.axbench_bpb_adv.gcg_family import (
    FAIR_GCG_METHOD_MODES,
    ROOFLINE_GCG_METHOD_MODES,
    GcgFamily,
    active_gcg_modes,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths

# ─────────────────────────────────────────────────────────────────────────────
# Family maps
# ─────────────────────────────────────────────────────────────────────────────


def test_mode_maps_carry_the_locked_steering_modes() -> None:
    """POSITIVE for the base methods, NEGATIVE for the neg_* variants."""
    assert not set(FAIR_GCG_METHOD_MODES) & set(ROOFLINE_GCG_METHOD_MODES)
    assert FAIR_GCG_METHOD_MODES["ours_gcg_encdec"] is SteeringMode.POSITIVE
    assert FAIR_GCG_METHOD_MODES["neg_ours_gcg_encdec"] is SteeringMode.NEGATIVE
    assert (
        ROOFLINE_GCG_METHOD_MODES["ours_gcg_roofline_encdec"]
        is SteeringMode.POSITIVE
    )
    assert (
        ROOFLINE_GCG_METHOD_MODES["neg_ours_gcg_roofline_encdec"]
        is SteeringMode.NEGATIVE
    )

def test_active_gcg_modes_none_is_all_methods() -> None:
    """base_methods=None ⇒ every mode of the given map is active (full run)."""
    assert active_gcg_modes(None, FAIR_GCG_METHOD_MODES) == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }
    assert active_gcg_modes(None, ROOFLINE_GCG_METHOD_MODES) == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }


def test_active_gcg_modes_filters_to_listed_names() -> None:
    """An explicit whitelist activates only the listed method names' modes."""
    assert active_gcg_modes(["ours_gcg_encdec"], FAIR_GCG_METHOD_MODES) == {
        SteeringMode.POSITIVE
    }
    assert active_gcg_modes(
        ["neg_ours_gcg_roofline_encdec", "diffmean_encdec"],
        ROOFLINE_GCG_METHOD_MODES,
    ) == {SteeringMode.NEGATIVE}
    # A whitelist naming no method from the map activates nothing.
    assert active_gcg_modes(["diffmean_encdec"], FAIR_GCG_METHOD_MODES) == set()
    assert active_gcg_modes([], ROOFLINE_GCG_METHOD_MODES) == set()


def test_family_values_are_the_historical_strings() -> None:
    """StrEnum values must stay byte-identical to the historical slugs."""
    assert str(GcgFamily.FAIR) == "fair"
    assert str(GcgFamily.ROOFLINE) == "roofline"
    assert GcgFamily("fair") is GcgFamily.FAIR
    assert GcgFamily("roofline") is GcgFamily.ROOFLINE


# ─────────────────────────────────────────────────────────────────────────────
# Roofline methods are valid with general_val enabled
# ─────────────────────────────────────────────────────────────────────────────


def _valid_config(**overrides: object) -> AxBenchBpbAdvConfig:
    base = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root="/tmp/out",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
    )
    for key, value in overrides.items():
        setattr(base, key, value)
    return base


@pytest.mark.parametrize(
    "base_methods",
    [
        ["ours_gcg_roofline_encdec"],
        ["neg_ours_gcg_roofline_encdec", "diffmean_encdec"],
    ],
)
def test_general_val_accepts_explicit_roofline_at_config_time(
    base_methods: list[str],
) -> None:
    cfg = _valid_config(
        base_methods=base_methods,
        general_val_bpb=GeneralValBpbConfig(enabled=True),
    )
    validate_config(cfg)


def test_general_val_allows_base_methods_none() -> None:
    """base_methods=None + general_val is valid."""
    validate_config(
        _valid_config(
            base_methods=None,
            general_val_bpb=GeneralValBpbConfig(enabled=True),
        )
    )


def test_general_val_allows_fair_gcg_methods() -> None:
    """The FAIR gcg methods (universal bag) remain valid with general_val."""
    validate_config(
        _valid_config(
            base_methods=["ours_gcg_encdec", "neg_ours_gcg_encdec"],
            general_val_bpb=GeneralValBpbConfig(enabled=True),
        )
    )


def test_general_val_allows_feature_diffmean_methods() -> None:
    """Feature-DiffMean uses universal bags, not prefix-specific roofline bags."""
    validate_config(
        _valid_config(
            base_methods=["feature_diffmean_encdec", "neg_feature_diffmean_encdec"],
            general_val_bpb=GeneralValBpbConfig(enabled=True),
        )
    )


def test_roofline_methods_fine_when_general_val_disabled() -> None:
    """The gate keys on the COMBINATION: roofline alone stays valid."""
    validate_config(_valid_config(base_methods=["ours_gcg_roofline_encdec"]))


# ─────────────────────────────────────────────────────────────────────────────
# Path-seam coercion + slug stability
# ─────────────────────────────────────────────────────────────────────────────


def test_bag_path_slugs_are_stable_for_enum_and_exact_string(tmp_path: Path) -> None:
    """Enum members and the exact historical strings produce IDENTICAL paths.

    This is the artifact-compatibility pin: bags derived before the enum-ification
    must resolve to the same files.
    """
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="slugs")
    for family, slug in (
        (GcgFamily.FAIR, "fair"),
        (GcgFamily.ROOFLINE, "roofline"),
    ):
        enum_path = paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.5, family)
        str_path = paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.5, slug)
        assert enum_path == str_path
        assert f"/{slug}/" in str(enum_path)
        assert (
            paths.ours_gcg_scored_prefixes(7, SteeringMode.POSITIVE, family)
            == paths.ours_gcg_scored_prefixes(7, SteeringMode.POSITIVE, slug)
        )


@pytest.mark.parametrize("bad", ["Fair", "ROOFLINE", "roof", "", "fair/"])
def test_bag_path_rejects_non_family_spellings(tmp_path: Path, bad: str) -> None:
    """Any spelling that is not a GcgFamily value RAISES instead of silently
    minting a new directory slug the loader would later report as a
    missing-artifact pipeline bug."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="slugs")
    with pytest.raises(ValueError):
        paths.ours_gcg_bag(7, SteeringMode.POSITIVE, 0.5, bad)
    with pytest.raises(ValueError):
        paths.ours_gcg_scored_prefixes(7, SteeringMode.POSITIVE, bad)
