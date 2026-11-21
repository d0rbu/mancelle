"""Exact publication gate for the issue #1673 campaign matrix."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from scripts.eval.axbench_adv.merge_slices import (
    _ControlRecord,
    _load_run_slice,
    _RunSlice,
    _validate_across_pressures,
    _validate_complete_campaign,
)
from scripts.eval.axbench_adv.pressures import DEFAULT_MAGNITUDES, LEGACY_PRESSURES, Pressure
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.types import ALL_METHODS, PLOT_EFFORT_LEVELS

AUDITED_LEGACY_C0_2 = Path(
    "/tmp/issue-1673-audit.MHNIpX/legacy_full/sft_s32mask/c0_2"
)


def _slice(pressure: Pressure) -> _RunSlice:
    levels = (
        tuple(float(level) for level in PLOT_EFFORT_LEVELS)
        if pressure == Pressure.GEPA
        else (0.0, *DEFAULT_MAGNITUDES)
    )
    return _RunSlice(
        root=Path(f"/{pressure}"),
        manifest_sha256="0" * 64,
        manifest={"max_concepts": 50},
        methods=ALL_METHODS,
        concept_ids=tuple(range(50)),
        factors=dict.fromkeys(ALL_METHODS, 1.0),
        entries=dict.fromkeys(levels, ()),
    )


def test_complete_campaign_requires_exact_pressure_method_concept_and_level_product() -> None:
    complete = {pressure: (_slice(pressure),) for pressure in LEGACY_PRESSURES}
    _validate_complete_campaign(complete)

    with pytest.raises(AxBenchAdvError, match="all three pressure families"):
        _validate_complete_campaign({**complete, Pressure.GCG: (_slice(Pressure.GCG),)})

    missing_pressure = dict(complete)
    missing_pressure.pop(Pressure.GEPA)
    with pytest.raises(AxBenchAdvError, match="all three pressure families"):
        _validate_complete_campaign(missing_pressure)

    counter = complete[Pressure.DIFFMEAN_COUNTER][0]
    missing_nl = replace(counter, methods=ALL_METHODS[:-1])
    with pytest.raises(AxBenchAdvError, match="coverage differs"):
        _validate_complete_campaign({**complete, Pressure.DIFFMEAN_COUNTER: (missing_nl,)})

    wrong_grid = replace(counter, entries=dict.fromkeys((0.0, 0.25), ()))
    with pytest.raises(AxBenchAdvError, match="wrong five-level grid"):
        _validate_complete_campaign({**complete, Pressure.DIFFMEAN_COUNTER: (wrong_grid,)})

    with pytest.raises(AxBenchAdvError, match="repeats"):
        _validate_complete_campaign(
            {**complete, Pressure.DIFFMEAN_COUNTER: (counter, counter)}
        )


def test_nl_attachment_identity_is_scoped_but_exact_across_nl_slices() -> None:
    first = replace(
        _slice(Pressure.GEPA),
        methods=(ALL_METHODS[-1],),
        factors={ALL_METHODS[-1]: 1.0},
        manifest={"max_concepts": 50, "nl_source_attachment": {"identity": "a"}},
    )
    second = replace(
        first,
        root=Path("/counter"),
        manifest={"max_concepts": 50, "nl_source_attachment": {"identity": "b"}},
    )
    with pytest.raises(AxBenchAdvError, match="incompatible source provenance"):
        _validate_across_pressures(
            {Pressure.GEPA: (first,), Pressure.DIFFMEAN_COUNTER: (second,)}
        )


def test_shared_zero_control_mismatch_remains_rejected() -> None:
    coordinate = ("DiffMean", 0, 0)
    first = replace(
        _slice(Pressure.GEPA),
        methods=("DiffMean",),
        concept_ids=(0,),
        factors={"DiffMean": 1.0},
        controls={coordinate: _ControlRecord(prompt=("same",), judged=("first",))},
    )
    second = replace(
        first,
        root=Path("/counter"),
        controls={coordinate: _ControlRecord(prompt=("same",), judged=("second",))},
    )

    with pytest.raises(AxBenchAdvError, match="shared zero control differs"):
        _validate_across_pressures(
            {Pressure.GEPA: (first,), Pressure.DIFFMEAN_COUNTER: (second,)}
        )


@pytest.mark.skipif(
    not (AUDITED_LEGACY_C0_2 / "manifest.json").is_file(),
    reason="read-only audited issue #1673 artifact is unavailable",
)
def test_exact_audited_legacy_c0_2_remains_loadable() -> None:
    loaded = _load_run_slice(AUDITED_LEGACY_C0_2)

    assert loaded.methods == ALL_METHODS[:-1]
    assert loaded.concept_ids == (0, 1)
