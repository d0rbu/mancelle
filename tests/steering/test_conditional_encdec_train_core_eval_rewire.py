"""Pin: CORE eval uses a single full-context cached wrapper.

The legacy eval_wrappers dict and no_features/unconditioned branch are gone.
"""

from __future__ import annotations

import inspect

import scripts.train.conditional_encdec_train as mod


def test_core_eval_loop_uses_cached_full_context_provider():
    src = inspect.getsource(mod)
    assert "core_metric_every" in src
    assert "core_metric/cached" in src
    assert '"no_features"' not in src and "'no_features'" not in src


def test_core_eval_loop_uses_build_wrapper_for_features():
    src = inspect.getsource(mod)
    assert "build_wrapper_for_features" in src, (
        "CORE eval rewire must construct wrappers via build_wrapper_for_features"
    )


def test_core_eval_no_longer_iterates_eval_wrappers_dict():
    src = inspect.getsource(mod)
    assert "for mode_name, wrapper in eval_wrappers.items()" not in src, (
        "Legacy eval_wrappers dict iteration must be removed"
    )


def test_core_eval_logs_metric_under_cached_key_not_common_features():
    src = inspect.getsource(mod)
    assert 'core_metric/cached' in src or '"core_metric/cached"' in src.replace("'", '"')
    assert 'core_metric/common_features' not in src


def test_core_eval_passes_disable_fp8_around_eval_call():
    """disable_fp8 around evaluate_core preserved per existing convention."""
    src = inspect.getsource(mod)
    assert "disable_fp8(orig_model)" in src
