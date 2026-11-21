"""PR-D results.json variant types and dispatch."""

from __future__ import annotations

import pytest

from steering.eval.progression_results import (
    StaticYamlSuiteResult,
    SuiteResultKind,
    TargetRealLockstepSuiteResult,
    TargetRealPooledSuiteResult,
    dump_suite_to_dict,
    load_suite_from_dict,
)


def test_static_yaml_result_with_kind():
    r = StaticYamlSuiteResult(
        name="safety_static",
        seeds={"1234": {"score": 0.5}},
    )
    assert r.kind == "static_yaml"


def test_target_real_lockstep_result_with_cache_dir():
    r = TargetRealLockstepSuiteResult(
        name="real_lockstep_safety",
        cache_dir="out/.../safety_cache_lockstep",
        seeds={"1234": {"score": 0.65}},
    )
    assert r.kind == "target_real_lockstep"


def test_target_real_pooled_result_with_pooling_op():
    r = TargetRealPooledSuiteResult(
        name="real_pooled_safety",
        cache_dir="out/.../safety_cache_pooled",
        pooling_op="bucket_max",
        seeds={"1234": {"score": 0.6}},
    )
    assert r.kind == "target_real_pooled"
    assert r.pooling_op == "bucket_max"


def test_dump_static_yaml_to_dict():
    r = StaticYamlSuiteResult(
        name="safety_static",
        seeds={"1234": {"score": 0.5}},
    )
    d = dump_suite_to_dict(r)
    assert d["kind"] == "static_yaml"
    assert d["target_conditioning_mode"] == "current"
    assert d["seeds"]["1234"]["score"] == 0.5


def test_dump_target_real_lockstep_to_dict():
    r = TargetRealLockstepSuiteResult(
        name="real_lockstep",
        cache_dir="/tmp/x", seeds={"1234": {"score": 0.5}},
    )
    d = dump_suite_to_dict(r)
    assert d["kind"] == "target_real_lockstep"
    assert d["cache_dir"] == "/tmp/x"
    assert d["target_conditioning_mode"] == "current"


def test_dump_target_real_pooled_to_dict():
    r = TargetRealPooledSuiteResult(
        name="real_pooled",
        cache_dir="/tmp/x", pooling_op="bucket_mean",
        seeds={"1234": {"score": 0.5}},
    )
    d = dump_suite_to_dict(r)
    assert d["kind"] == "target_real_pooled"
    assert d["pooling_op"] == "bucket_mean"
    assert d["target_conditioning_mode"] == "current"


def test_load_rejects_missing_kind_field():
    with pytest.raises(ValueError, match="missing required 'kind'"):
        load_suite_from_dict({"seeds": {"1234": {"score": 0.5}}}, name="legacy")


def test_load_dispatches_static_yaml_on_kind():
    r = load_suite_from_dict({
        "kind": "static_yaml",
        "target_conditioning_mode": "current",
        "seeds": {"1234": {"score": 0.5}},
    }, name="safety_static")
    assert isinstance(r, StaticYamlSuiteResult)


def test_load_dispatches_target_real_lockstep_on_kind():
    r = load_suite_from_dict({
        "kind": "target_real_lockstep",
        "target_conditioning_mode": "next",
        "cache_dir": "/tmp/x", "seeds": {"1234": {"score": 0.5}},
    }, name="real_lockstep")
    assert isinstance(r, TargetRealLockstepSuiteResult)
    assert r.target_conditioning_mode == "next"


def test_load_dispatches_target_real_pooled_on_kind():
    r = load_suite_from_dict({
        "kind": "target_real_pooled",
        "target_conditioning_mode": "current",
        "cache_dir": "/tmp/x", "pooling_op": "bucket_max",
        "seeds": {"1234": {"score": 0.5}},
    }, name="real_pooled")
    assert isinstance(r, TargetRealPooledSuiteResult)
    assert r.pooling_op == "bucket_max"


def test_load_unknown_kind_raises():
    with pytest.raises(ValueError, match="unknown kind"):
        load_suite_from_dict({"kind": "future_kind", "seeds": {}}, name="x")


def test_dump_then_load_round_trip_each_variant():
    cases = [
        StaticYamlSuiteResult(name="a", seeds={"1234": {"score": 0.5}}),
        TargetRealLockstepSuiteResult(name="b", cache_dir="/tmp/b", seeds={"1234": {"score": 0.6}}),
        TargetRealPooledSuiteResult(name="c", cache_dir="/tmp/c", pooling_op="bucket_max", seeds={"1234": {"score": 0.55}}),
    ]
    for original in cases:
        d = dump_suite_to_dict(original)
        loaded = load_suite_from_dict(
            d,
            name=original.name,
        )
        assert type(loaded) is type(original)
        assert dump_suite_to_dict(loaded) == d


# ---------------------------------------------------------------------------
# SuiteResultKind StrEnum tests (Fix #2)
# ---------------------------------------------------------------------------


def test_suite_result_kind_enum_values_match_variant_kind_strings():
    assert SuiteResultKind.STATIC_YAML.value == "static_yaml"
    assert SuiteResultKind.TARGET_REAL_LOCKSTEP.value == "target_real_lockstep"
    assert SuiteResultKind.TARGET_REAL_POOLED.value == "target_real_pooled"


def test_suite_result_kind_round_trips_with_dataclass_variants():
    """Each variant's `kind` field equals SuiteResultKind.<X>.value."""
    static = StaticYamlSuiteResult(name="x", seeds={})
    assert static.kind == SuiteResultKind.STATIC_YAML.value
    lockstep = TargetRealLockstepSuiteResult(name="x", cache_dir="/tmp", seeds={})
    assert lockstep.kind == SuiteResultKind.TARGET_REAL_LOCKSTEP.value
    pooled = TargetRealPooledSuiteResult(name="x", cache_dir="/tmp", pooling_op="bucket_max", seeds={})
    assert pooled.kind == SuiteResultKind.TARGET_REAL_POOLED.value


def test_suite_result_kind_str_comparison_works_as_strenum():
    """StrEnum members compare equal to their string values."""
    assert SuiteResultKind.STATIC_YAML == "static_yaml"
    assert SuiteResultKind.TARGET_REAL_LOCKSTEP == "target_real_lockstep"
    assert SuiteResultKind.TARGET_REAL_POOLED == "target_real_pooled"


def test_kind_field_is_a_property_not_a_settable_dataclass_field():
    """`kind` is a property derived from the class, not a settable field
    that could be set to a wrong value at construction."""
    r = TargetRealLockstepSuiteResult(
        name="x",
        cache_dir="/tmp", seeds={},
    )
    # The property still returns the right kind.
    assert r.kind == "target_real_lockstep"
    # But it cannot be set at construction (no kwarg accepted).
    with pytest.raises(TypeError, match=r"kind"):
        TargetRealLockstepSuiteResult(
            name="x",
            cache_dir="/tmp", seeds={},
            kind="WRONG",  # ty: ignore[unknown-argument]
        )
