"""load_suite_from_dict raises operator-actionable ValueError on missing fields."""
from __future__ import annotations

import pytest

from steering.eval.progression_results import load_suite_from_dict


def test_missing_cache_dir_raises_value_error():
    with pytest.raises(ValueError, match=r"missing.*cache_dir"):
        load_suite_from_dict({
            "kind": "target_real_lockstep",
            "target_conditioning_mode": "current",
        }, name="x")


def test_missing_pooling_op_raises_value_error():
    with pytest.raises(ValueError, match=r"missing.*pooling_op"):
        load_suite_from_dict({
            "kind": "target_real_pooled",
            "target_conditioning_mode": "current",
            "cache_dir": "/tmp",
        }, name="x")


def test_unknown_kind_still_raises_value_error():
    with pytest.raises(ValueError, match=r"unknown kind"):
        load_suite_from_dict({"kind": "future_kind", "seeds": {}}, name="x")


def test_no_kind_raises_value_error():
    with pytest.raises(ValueError, match=r"missing required 'kind'"):
        load_suite_from_dict({"seeds": {"1234": {}}}, name="legacy")
