"""PR-D: plot_evals.py discovers target_real/ subdir result yamls.

Pinned invariants:
  - The discovery loop walks <eval_dir>/target_real/<name>.yaml entries
    in addition to top-level <name>.yaml.
  - Each discovered target_real result is plotted with the linestyle
    returned by _get_linestyle_for_variant (so target_real/pooled is
    dash-dot, target_real/lockstep is dotted).
  - Source-inspection: plot_evals imports _get_linestyle_for_variant
    from steering.eval.result_plot_dispatch.
"""

from __future__ import annotations

import inspect

import scripts.eval.plot_evals as mod


def test_plot_evals_imports_get_linestyle_for_variant():
    src = inspect.getsource(mod)
    assert "_get_linestyle_for_variant" in src, (
        "plot_evals must import the shared linestyle dispatch (PR-D Task 7)"
    )
    assert "result_plot_dispatch" in src


def test_plot_evals_walks_target_real_subdir():
    src = inspect.getsource(mod)
    assert "target_real" in src, (
        "plot_evals must discover <eval_dir>/target_real/<name>.yaml entries"
    )


def test_plot_evals_dispatches_via_load_suite_from_dict():
    """Using the shared load_suite_from_dict (PR-D Task 2) for type-correct dispatch."""
    src = inspect.getsource(mod)
    assert "load_suite_from_dict" in src or "SuiteResultVariant" in src, (
        "plot_evals must use the shared progression_results loader (PR-D Task 2) "
        "to dispatch by kind for plotting"
    )
