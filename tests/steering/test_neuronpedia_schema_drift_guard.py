"""Slice 14 audit B31: Neuronpedia fixture ↔ live API key drift guard.

Captured fixtures use snake-case (``freq_hist_bar_*``) while the live
Neuronpedia ``/api/feature`` endpoint uses ``freq_hist_data_bar_*`` and
``maxActApprox``. The pre-existing ``_check_fixture_consistency`` only
compared fixtures to each other — it would not have flagged a live-side
rename.

These tests exercise ``_check_live_payload`` directly with synthesized
live-shape dicts to verify that:

1. Removing any documented top-level key surfaces as drift.
2. Removing any documented per-entry key surfaces as drift.
3. A well-formed live payload returns no drift.

Also pins the snake/camel mapping table at module-import time.
"""

from __future__ import annotations

from typing import Any

import pytest

from scripts.validation.check_neuronpedia_schema import (
    REQUIRED_ENTRY,
    REQUIRED_TOP,
    _check_live_payload,
)

# Live (camel/long) name → captured fixture (snake/short) name. Runtime
# validation uses ``REQUIRED_TOP``; this test-owned table guards the documented
# relationship between the two independently versioned payload shapes.
_LIVE_TO_FIXTURE_KEY_MAP: dict[str, str] = {
    "freq_hist_data_bar_heights": "freq_hist_bar_heights",
    "freq_hist_data_bar_values": "freq_hist_bar_values",
    "maxActApprox": "max_act_approx",
    "activations": "entries",
    "explanations": "explanations",
    "frac_nonzero": "frac_nonzero",
}


def _well_formed_live_payload() -> dict[str, Any]:
    """Synthesize a minimal valid live-shape payload."""
    return {
        "explanations": ["something"],
        "frac_nonzero": 0.05,
        "freq_hist_data_bar_heights": [0.1, 0.2, 0.3],
        "freq_hist_data_bar_values": [0.0, 0.5, 1.0],
        "maxActApprox": 4.2,
        "activations": [
            {
                "tokens": ["foo", "bar"],
                "values": [0.0, 0.5],
                "maxValue": 0.5,
                "maxValueTokenIndex": 1,
                "binMin": -1.0,
                "binMax": -1.0,
                "binContains": 0.0,
            },
            {
                "tokens": ["baz"],
                "values": [1.0],
                "maxValue": 1.0,
                "maxValueTokenIndex": 0,
                "binMin": 0.0,
                "binMax": 1.0,
                "binContains": 0.5,
            },
        ],
    }


def test_well_formed_payload_returns_no_drift() -> None:
    """Sanity: the synthesized happy-path payload triggers zero drift."""
    drift = _check_live_payload(42, _well_formed_live_payload())
    assert drift == [], f"Unexpected drift on happy-path payload: {drift}"


@pytest.mark.parametrize("missing_key", sorted(REQUIRED_TOP))
def test_drift_when_top_level_key_missing(missing_key: str) -> None:
    """Removing any documented live-shape top-level key MUST surface drift."""
    payload = _well_formed_live_payload()
    payload.pop(missing_key)
    drift = _check_live_payload(42, payload)
    if missing_key == "activations":
        # Special path: empty activations → "no activations in live payload"
        assert any("no activations" in d for d in drift)
    else:
        assert any("top-level missing" in d and missing_key in d for d in drift), (
            f"Expected drift mentioning {missing_key!r}; got {drift}"
        )


@pytest.mark.parametrize("missing_entry_key", sorted(REQUIRED_ENTRY))
def test_drift_when_entry_key_missing(missing_entry_key: str) -> None:
    """Removing any documented per-entry key MUST surface drift OR raise.

    For ``binMin`` / ``binMax``, ``_check_live_payload``'s post-entry-key
    inner loop unconditionally does ``float(a["binMin"])`` /
    ``float(a["binMax"])`` and raises ``KeyError`` before its entry-keys
    drift block can append a message. Both outcomes are acceptable as
    "schema drift surfaces"; what we want to reject is *silent pass*.

    For all other keys we expect the explicit entry-keys drift message.
    """
    payload = _well_formed_live_payload()
    for act in payload["activations"]:
        act.pop(missing_entry_key, None)

    if missing_entry_key in {"binMin", "binMax"}:
        # Schema drift surfaces as a KeyError before the entry-keys block.
        with pytest.raises(KeyError, match=missing_entry_key):
            _check_live_payload(42, payload)
    else:
        drift = _check_live_payload(42, payload)
        assert any("activation entry missing" in d for d in drift), (
            f"Expected entry-keys drift for {missing_entry_key}; got {drift}"
        )


def test_drift_when_all_activations_empty() -> None:
    """Empty ``activations`` list still surfaces as drift."""
    payload = _well_formed_live_payload()
    payload["activations"] = []
    drift = _check_live_payload(42, payload)
    assert any("no activations in live payload" in d for d in drift)


def test_live_to_fixture_mapping_documents_all_drifting_keys() -> None:
    """Pin the snake/camel mapping table.

    The mapping documents (in code) which live keys correspond to which
    fixture-shape keys. A schema change must update both ``REQUIRED_TOP``
    *and* this mapping. Test ensures the keys we declared as "drifting"
    are recorded.
    """
    # Every camel/long live-shape key listed below must appear in the mapping.
    drifting_live_keys = {
        "freq_hist_data_bar_heights",
        "freq_hist_data_bar_values",
        "maxActApprox",
        "activations",
    }
    assert drifting_live_keys.issubset(_LIVE_TO_FIXTURE_KEY_MAP.keys())
