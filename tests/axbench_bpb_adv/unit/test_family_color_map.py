"""Plot colour assignment: one distinguishable colour per steering family.

These figures are read SIDE BY SIDE across runs, so a family's colour has to be
a property of the family, not of which other methods happened to be in the run.
Two rules carry that:

  * a pinned family keeps its colour regardless of the population;
  * an index-assigned family never lands ON a pinned colour.

The second is not hypothetical. Before ``feature GCG (margin alpha)`` was
pinned, it took tab10 index 4 -- ``feature DiffMean``'s #9467bd -- in any run
without a decoder-only control, so two distinct families were drawn in exactly
the same purple in the same figure, with nothing to tell them apart.
"""

from __future__ import annotations

import matplotlib.colors as mcolors
import pytest

from steering.eval.axbench_bpb_adv.aggregation import (
    _FEATURE_FAMILY_COLOR_OVERRIDES,
    _family_color_map,
    _model_steering_family,
)

_ALPHA_FAMILY = "encdec / feature GCG (margin alpha)"
_GCG_FAMILY = "encdec / feature GCG"
_DIFFMEAN_FAMILY = "encdec / feature DiffMean"

# The alpha-sweep population: decoder baselines + control + the two encoder-space
# feature families + the alpha sweep.
_FULL = [
    "diffmean_decoder_only",
    "linear_probe_decoder_only",
    "control_decoder_only",
    "feature_diffmean_encdec",
    "ours_gcg_encdec",
    "ours_gcg_alpha_encdec",
]


def _rows(method_names: list[str]) -> list[dict[str, str]]:
    return [
        {
            "target_model": "decoder_only"
            if name.endswith("_decoder_only")
            else "encdec",
            "method_name": name,
        }
        for name in method_names
    ]


def _hexes(method_names: list[str]) -> dict[str, str]:
    return {
        family: mcolors.to_hex(color)
        for family, color in _family_color_map(_rows(method_names)).items()
    }


def test_alpha_family_is_its_own_plotted_family() -> None:
    """The alpha sweep is a separate family from the lambda sweep it shares bags with."""
    assert _model_steering_family(
        {"target_model": "encdec", "method_name": "ours_gcg_alpha_encdec"}
    ) == _ALPHA_FAMILY
    assert _ALPHA_FAMILY != _GCG_FAMILY


@pytest.mark.parametrize(
    "population",
    [
        _FULL,
        [m for m in _FULL if m != "control_decoder_only"],
        [m for m in _FULL if not m.endswith("_decoder_only")],
        _FULL + ["ours_extract_and_add_encdec"],
        # PREDATES the alpha family: with four decoder-only families sorting
        # ahead of it, `decoder-only / lora` takes tab10 index 3 -- which is
        # `feature GCG`'s pinned #d62728. This case is the one that exercises
        # the pin-before-cycle ORDERING itself; the alpha cases above are
        # satisfied by alpha's own pin and would pass under the old ordering.
        [
            "control_decoder_only",
            "diffmean_decoder_only",
            "linear_probe_decoder_only",
            "lora_decoder_only",
            "ours_gcg_encdec",
        ],
    ],
    ids=["full", "no-decoder-control", "encdec-only", "extra-family", "lora-vs-gcg"],
)
def test_no_two_families_share_a_colour(population) -> None:
    """Distinct families are drawn in distinct colours, for every population.

    Teeth: assigning the cycle BEFORE resolving pins (the original ordering)
    reds the ``lora-vs-gcg`` case, where an unpinned family lands exactly on a
    pinned one. Pinning alone does not fix that class -- only withholding the
    pinned colours from the cycle does.
    """
    hexes = _hexes(population)
    duplicates = {value for value in hexes.values() if list(hexes.values()).count(value) > 1}
    assert not duplicates, f"families share colours {duplicates} in {sorted(hexes)}"


@pytest.mark.parametrize("family", [_ALPHA_FAMILY, _GCG_FAMILY, _DIFFMEAN_FAMILY])
def test_pinned_families_keep_their_colour_across_populations(family) -> None:
    """A pinned family's colour does not depend on who else is in the run.

    Teeth: unpinning any of the three makes its colour move between these two
    populations, because removing a family that sorts earlier shifts every
    later index.
    """
    with_control = _hexes(_FULL)
    without_control = _hexes([m for m in _FULL if m != "control_decoder_only"])
    assert with_control[family] == without_control[family]


def test_alpha_family_colour_is_pinned_not_cycled() -> None:
    """The alpha family is declared in the overrides, not left to the cycle."""
    assert "feature GCG (margin alpha)" in _FEATURE_FAMILY_COLOR_OVERRIDES
    expected = mcolors.to_hex(
        _FEATURE_FAMILY_COLOR_OVERRIDES["feature GCG (margin alpha)"]
    )
    assert _hexes(_FULL)[_ALPHA_FAMILY] == expected


def test_alpha_is_visually_distinct_from_its_sibling_families() -> None:
    """Alpha must not be a near-shade of feature GCG or feature DiffMean.

    They are adjacent curves on the same axes, so 'related hue' would read as
    'same series'. Compared in RGB rather than by hex inequality, which a
    one-bit difference would satisfy while being invisible on the page.
    """
    hexes = _hexes(_FULL)

    def rgb(value: str) -> tuple[float, float, float]:
        return mcolors.to_rgb(value)

    def distance(a: str, b: str) -> float:
        return sum((x - y) ** 2 for x, y in zip(rgb(a), rgb(b), strict=True)) ** 0.5

    for other in (_GCG_FAMILY, _DIFFMEAN_FAMILY):
        assert distance(hexes[_ALPHA_FAMILY], hexes[other]) > 0.35, (
            f"alpha colour {hexes[_ALPHA_FAMILY]} is too close to {other} "
            f"{hexes[other]} to be told apart on the plot"
        )


def test_every_plotted_family_gets_a_colour() -> None:
    """No family is missing from the map -- the plotter indexes it directly."""
    families = {
        _model_steering_family(row) for row in _rows(_FULL + ["ours_extract_and_add_encdec"])
    }
    color_map = _family_color_map(_rows(_FULL + ["ours_extract_and_add_encdec"]))
    assert set(color_map) == families
