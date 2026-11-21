from __future__ import annotations

import json
from typing import Any

import pytest
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_bpb_adv.candidate_metadata import (
    FasterGcgFinalMetadata,
    GepaFinalMetadata,
    NoAdversaryControlMetadata,
    candidate_metadata_to_dict,
    candidate_metadata_to_json,
    parse_candidate_metadata_json,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.refined_types import NonNegativeInt


def _counter(value: int) -> NonNegativeInt:
    return NonNegativeInt.parse(value)


_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)


def test_candidate_metadata_serializes_to_stable_legacy_json_shape() -> None:
    assert (
        candidate_metadata_to_json(NoAdversaryControlMetadata())
        == '{"candidate":"no_adversary_control"}'
    )
    assert (
        candidate_metadata_to_json(GepaFinalMetadata())
        == '{"candidate":"gepa_final"}'
    )
    assert (
        candidate_metadata_to_json(
            FasterGcgFinalMetadata(
                max_steps=_counter(4),
                accepted_steps=_counter(2),
                visited_sequences=_counter(9),
            )
        )
        == (
            '{"accepted_steps":2,"candidate":"faster_gcg_final",'
            '"max_steps":4,"visited_sequences":9}'
        )
    )


def test_candidate_metadata_roundtrips_known_payloads() -> None:
    metadata = FasterGcgFinalMetadata(
        max_steps=_counter(4),
        accepted_steps=_counter(2),
        visited_sequences=_counter(9),
    )

    assert (
        parse_candidate_metadata_json(candidate_metadata_to_json(metadata))
        == metadata
    )
    assert (
        parse_candidate_metadata_json('{"candidate":"gepa_final"}')
        == GepaFinalMetadata()
    )
    assert parse_candidate_metadata_json(
        '{"candidate":"no_adversary_control"}'
    ) == NoAdversaryControlMetadata()


def test_candidate_metadata_parser_raises_on_corrupt_payloads() -> None:
    for raw in (
        None,
        "",
        "{}",
        "not json",
        "[1, 2, 3]",
        '{"candidate":"faster_gcg_final","max_steps":true,'
        '"accepted_steps":2,"visited_sequences":9}',
        '{"candidate":"faster_gcg_final","max_steps":"4",'
        '"accepted_steps":2,"visited_sequences":9}',
        '{"candidate":"faster_gcg_final","max_steps":4.2,'
        '"accepted_steps":2,"visited_sequences":9}',
        '{"candidate":"faster_gcg_final","max_steps":-1,'
        '"accepted_steps":2,"visited_sequences":9}',
        '{"candidate":"faster_gcg_final","max_steps":4,'
        '"accepted_steps":5,"visited_sequences":9}',
        '{"candidate":"faster_gcg_final","max_steps":4,'
        '"accepted_steps":2,"visited_sequences":0}',
        '{"candidate":"faster_gcg_final","max_steps":4,'
        '"accepted_steps":2,"visited_sequences":2}',
        '{"candidate":"unknown"}',
    ):
        with pytest.raises(AxBenchBpbAdvError):
            parse_candidate_metadata_json(raw)


def test_candidate_metadata_parser_error_names_missing_or_empty_metadata() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="missing"):
        parse_candidate_metadata_json(None)
    with pytest.raises(AxBenchBpbAdvError, match="empty"):
        parse_candidate_metadata_json("")
    with pytest.raises(AxBenchBpbAdvError, match="missing required 'candidate'"):
        parse_candidate_metadata_json("{}")


def test_candidate_metadata_parser_error_names_bad_integer_field() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="'max_steps'"):
        parse_candidate_metadata_json(
            '{"candidate":"faster_gcg_final","max_steps":"4",'
            '"accepted_steps":2,"visited_sequences":9}'
        )


@settings(max_examples=500, derandomize=True)
@given(
    field=st.sampled_from(("max_steps", "accepted_steps", "visited_sequences")),
    surface=st.sampled_from(("dataclass", "json")),
    value=_INVALID_NON_NEGATIVE_INT,
)
def test_candidate_metadata_rejects_every_invalid_counter(
    field: str,
    surface: str,
    value: Any,
) -> None:
    fields: dict[str, Any] = {
        "max_steps": _counter(4),
        "accepted_steps": _counter(2),
        "visited_sequences": _counter(9),
    }
    fields[field] = value
    if surface == "dataclass":
        with pytest.raises(BeartypeCallHintParamViolation, match=field):
            FasterGcgFinalMetadata(**fields)
        return

    raw = json.dumps({"candidate": "faster_gcg_final", **fields})
    with pytest.raises(AxBenchBpbAdvError, match=field):
        parse_candidate_metadata_json(raw)


def test_candidate_metadata_parser_error_names_bad_counter_relation() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="accepted_steps=5 > max_steps=4"):
        parse_candidate_metadata_json(
            '{"candidate":"faster_gcg_final","max_steps":4,'
            '"accepted_steps":5,"visited_sequences":9}'
        )


def test_candidate_metadata_control_serializes_to_candidate_tag() -> None:
    assert candidate_metadata_to_dict(NoAdversaryControlMetadata()) == {
        "candidate": "no_adversary_control"
    }
