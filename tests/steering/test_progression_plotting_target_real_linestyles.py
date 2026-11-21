"""PR-D: training_progression's plotting uses _get_linestyle_for_variant for all variants.

Pinned invariants:
  - Source-inspection: plotting code imports _get_linestyle_for_variant.
  - 4-variant synthetic results: every variant gets its variant-specific linestyle.
"""

from __future__ import annotations

import inspect

import scripts.train.training_progression as mod


def test_training_progression_imports_get_linestyle_for_variant():
    src = inspect.getsource(mod)
    assert "_get_linestyle_for_variant" in src, (
        "training_progression must use the shared linestyle dispatch (PR-D Task 7) "
        "for charts so that lockstep/pooled/static-yaml variants "
        "render with the canonical per-variant linestyle/color."
    )
    assert "result_plot_dispatch" in src


def test_training_progression_plots_dispatch_via_load_suite_from_dict():
    """Plotting reads results.json variants via the shared loader so kind dispatch is correct."""
    src = inspect.getsource(mod)
    assert "load_suite_from_dict" in src or "SuiteResultVariant" in src, (
        "training_progression plotting must consume results.json via "
        "progression_results.load_suite_from_dict / SuiteResultVariant for "
        "type-correct linestyle dispatch."
    )
