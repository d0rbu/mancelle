import json
from types import MappingProxyType
from typing import Any

import pytest

from steering._common.strict_json import (
    canonical_json,
    canonical_json_str_fallback,
    reject_nonstandard_json_constant,
    strict_json_object,
    string_keyed_mapping,
)


def test_canonical_json_contract() -> None:
    assert canonical_json({"b": 1, "a": "é"}) == '{"a":"é","b":1}'


def test_canonical_json_is_stable_across_insertion_order() -> None:
    forward = {"a": 1, "b": 2, "c": 3}
    reverse = {"c": 3, "b": 2, "a": 1}

    assert canonical_json(forward) == canonical_json(reverse)


def test_strict_json_object_builds_plain_dict() -> None:
    result = strict_json_object([("b", 2), ("a", 1)])

    assert result == {"b": 2, "a": 1}
    assert type(result) is dict


def test_strict_json_object_rejects_duplicate_keys() -> None:
    with pytest.raises(ValueError, match=r"duplicate JSON object key 'a'"):
        strict_json_object([("a", 1), ("a", 2)])


def test_strict_json_object_round_trip() -> None:
    payload = '{"outer":{"a":1},"items":[true,null]}'

    assert json.loads(payload, object_pairs_hook=strict_json_object) == {
        "outer": {"a": 1},
        "items": [True, None],
    }
    with pytest.raises(ValueError, match=r"duplicate JSON object key 'a'"):
        json.loads('{"a":1,"a":2}', object_pairs_hook=strict_json_object)


def test_string_keyed_mapping_normalizes_arbitrary_mapping() -> None:
    source = MappingProxyType({"b": object(), "a": 1})

    result = string_keyed_mapping(source)

    assert result == source
    assert type(result) is dict


@pytest.mark.parametrize("value", [None, [], {1: "integer key"}, {"ok": 1, 2: "mixed"}])
def test_string_keyed_mapping_rejects_non_string_keyed_values(value: object) -> None:
    assert string_keyed_mapping(value) is None


@pytest.mark.parametrize(
    "value",
    [
        pytest.param(
            {"nested": {"z": [None, True, 3, 1.5], "a": {"é": "☃"}}},
            id="nested-dict",
        ),
        pytest.param(["é", None, True, 1, 2.5], id="list"),
        pytest.param(None, id="none"),
        pytest.param(True, id="bool"),
        pytest.param(42, id="int"),
        pytest.param(3.25, id="float"),
        pytest.param("é", id="non-ascii-string"),
        pytest.param({}, id="empty-dict"),
        pytest.param({"z": 1, "m": 2, "a": 3}, id="reverse-sorted-keys"),
    ],
)
def test_shared_canonical_json_matches_every_replaced_body(value: Any) -> None:
    def _original(original_value: Any) -> str:
        return json.dumps(
            original_value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    assert canonical_json(value) == _original(value)


@pytest.mark.parametrize(
    "pairs",
    [
        pytest.param([], id="empty"),
        pytest.param([("a", 1)], id="single"),
        pytest.param(
            [("b", {"nested": [None, True]}), ("a", "é")],
            id="nested-values",
        ),
    ],
)
def test_shared_strict_json_object_matches_every_replaced_body(
    pairs: list[tuple[str, Any]],
) -> None:
    def _original(original_pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in original_pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    assert strict_json_object(pairs) == _original(pairs)


def test_shared_strict_json_object_duplicate_matches_every_replaced_body() -> None:
    def _original(original_pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in original_pairs:
            if key in result:
                raise ValueError(f"duplicate JSON object key {key!r}")
            result[key] = value
        return result

    pairs = [("a", 1), ("a", 2)]
    with pytest.raises(
        ValueError, match=r"duplicate JSON object key 'a'"
    ) as shared_error:
        strict_json_object(pairs)
    with pytest.raises(
        ValueError, match=r"duplicate JSON object key 'a'"
    ) as original_error:
        _original(pairs)

    assert str(shared_error.value) == str(original_error.value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param("NaN", id="nan"),
        pytest.param("Infinity", id="infinity"),
        pytest.param("-Infinity", id="negative-infinity"),
    ],
)
def test_shared_reject_nonstandard_json_constant_matches_original_body(value: str) -> None:
    def _original(original_value: str) -> None:
        raise ValueError(f"non-standard JSON numeric constant {original_value!r}")

    with pytest.raises(ValueError, match=rf"non-standard JSON numeric constant {value!r}") as shared_error:
        reject_nonstandard_json_constant(value)
    with pytest.raises(ValueError, match=rf"non-standard JSON numeric constant {value!r}") as original_error:
        _original(value)

    assert str(shared_error.value) == str(original_error.value)


@pytest.mark.parametrize(
    "value",
    [
        pytest.param({"nested": {"z": [None, True, 3, 1.5], "a": {"é": "☃"}}}, id="nested-dict"),
        pytest.param(["é", None, True, 1, 2.5], id="list"),
        pytest.param(None, id="none"),
        pytest.param(True, id="bool"),
        pytest.param(42, id="int"),
        pytest.param(3.25, id="float"),
        pytest.param("é", id="non-ascii-string"),
        pytest.param({}, id="empty-dict"),
        pytest.param({"z": 1, "m": 2, "a": 3}, id="reverse-sorted-keys"),
    ],
)
def test_shared_canonical_json_str_fallback_matches_original_body(value: Any) -> None:
    def _original(original_value: Any) -> str:
        return json.dumps(original_value, sort_keys=True, separators=(",", ":"), default=str)

    assert canonical_json_str_fallback(value) == _original(value)


def test_shared_reject_nonstandard_json_constant_round_trip() -> None:
    with pytest.raises(ValueError, match=r"non-standard JSON numeric constant 'NaN'"):
        json.loads("NaN", parse_constant=reject_nonstandard_json_constant)


class _StringifiedValue:
    def __str__(self) -> str:
        return "stringified-value"


def test_canonical_json_and_str_fallback_deliberately_disagree() -> None:
    assert canonical_json_str_fallback({"text": "é"}) == '{"text":"\\u00e9"}'
    assert canonical_json({"text": "é"}) == '{"text":"é"}'

    value = _StringifiedValue()
    assert canonical_json_str_fallback({"value": value}) == '{"value":"stringified-value"}'
    with pytest.raises(
        TypeError,
        match=r"Object of type _StringifiedValue is not JSON serializable",
    ):
        canonical_json({"value": value})
