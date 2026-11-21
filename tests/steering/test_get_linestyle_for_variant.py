"""PR-D: _get_linestyle_for_variant returns (linestyle, color) per variant kind."""

from __future__ import annotations

import pytest

from steering.eval.progression_results import (
    StaticYamlSuiteResult,
    TargetRealLockstepSuiteResult,
    TargetRealPooledSuiteResult,
)
from steering.eval.result_plot_dispatch import _get_linestyle_for_variant


def test_static_yaml_dashed():
    r = StaticYamlSuiteResult(name="x", seeds={})
    ls, _ = _get_linestyle_for_variant(r)
    assert ls == "--"


def test_target_real_lockstep_dotted():
    r = TargetRealLockstepSuiteResult(name="x", cache_dir="/", seeds={})
    ls, _ = _get_linestyle_for_variant(r)
    assert ls == ":"


def test_target_real_pooled_dash_dot():
    r = TargetRealPooledSuiteResult(name="x", cache_dir="/", pooling_op="bucket_max", seeds={})
    ls, _ = _get_linestyle_for_variant(r)
    assert ls == "-."


def test_color_by_name_pairing():
    """Lockstep + pooled at the same NAME share color (visual pairing)."""
    a = TargetRealLockstepSuiteResult(name="real_safety", cache_dir="/", seeds={})
    b = TargetRealPooledSuiteResult(name="real_safety", cache_dir="/", pooling_op="bucket_max", seeds={})
    _, ca = _get_linestyle_for_variant(a)
    _, cb = _get_linestyle_for_variant(b)
    assert ca == cb


def test_unknown_variant_raises():
    with pytest.raises(TypeError):
        _get_linestyle_for_variant(object())  # ty: ignore[invalid-argument-type]
