"""PR-D: analyze_sweep filters by kind for the new target_real variants."""

from __future__ import annotations

import inspect

import pytest

import scripts.eval.analyze_sweep as mod
from scripts.eval.analyze_sweep import filter_by_kind, result_kind
from steering.eval.progression_results import SuiteResultKind


def test_analyze_sweep_uses_explicit_kind_dispatch():
    """analyze_sweep must not infer a kind for bare records."""
    src = inspect.getsource(mod)
    assert "SuiteResultKind" in src
    assert "load_suite_from_dict" not in src
    assert "PlainSuiteResult" not in src


def test_analyze_sweep_recognizes_target_real_kinds():
    """The filter knows about target_real_lockstep / target_real_pooled kinds."""
    src = inspect.getsource(mod)
    assert "target_real_lockstep" in src or "target_real" in src


# ---------------------------------------------------------------------------
# Fix #2: SuiteResultKind enum usage tests
# ---------------------------------------------------------------------------


def test_analyze_sweep_uses_suite_result_kind_enum():
    """analyze_sweep imports SuiteResultKind, not loose KIND_* constants."""
    src = inspect.getsource(mod)
    assert "SuiteResultKind" in src
    assert "KIND_UNCONDITIONED" not in src


def test_result_kind_rejects_bare_seeds_record():
    """Bare-seeds records are missing the required explicit kind."""
    record = {"seeds": {"1234": {"score": 0.5}}}
    with pytest.raises(ValueError, match="missing required 'kind'"):
        result_kind(record)


def test_result_kind_returns_suite_result_kind_for_explicit_kind_field():
    """result_kind honors the explicit kind field in a record."""
    record = {"kind": "target_real_lockstep", "cache_dir": "/tmp/x", "seeds": {}}
    kind = result_kind(record)
    assert kind == SuiteResultKind.TARGET_REAL_LOCKSTEP


def test_filter_by_kind_accepts_strenum_value():
    """filter_by_kind accepts SuiteResultKind members (StrEnum compares equal to str)."""
    records = [
        {"kind": "target_real_lockstep", "cache_dir": "/tmp", "seeds": {}},
        {"kind": "target_real_pooled", "cache_dir": "/tmp", "pooling_op": "bucket_max", "seeds": {}},
    ]
    pooled_only = filter_by_kind(records, SuiteResultKind.TARGET_REAL_POOLED)
    assert len(pooled_only) == 1
    assert pooled_only[0]["kind"] == "target_real_pooled"


def test_suite_result_kind_rejects_unknown_via_constructor():
    """SuiteResultKind('bogus') raises ValueError for validation in main()."""
    import pytest
    with pytest.raises(ValueError):
        SuiteResultKind("bogus_kind")
