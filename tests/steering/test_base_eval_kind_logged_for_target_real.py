"""base_eval / dclm_eval pass kind to report.log via run_target_real_evals."""
from __future__ import annotations

import inspect


def test_base_eval_main_passes_checkpoint_dir_to_run_target_real_evals():
    """Regression: base_eval.main MUST pass checkpoint_dir + report_section_prefix
    so the kind discriminator reaches report.log (parity with chat_eval)."""
    import scripts.eval.base_eval as mod
    src = inspect.getsource(mod)
    assert "checkpoint_dir=" in src and "run_target_real_evals" in src, (
        "base_eval.main must pass checkpoint_dir= to run_target_real_evals so "
        "report.log receives the kind discriminator (parity with chat_eval)."
    )
    assert "report_section_prefix=" in src


def test_dclm_eval_main_passes_checkpoint_dir():
    import steering.eval.dclm_eval as mod
    src = inspect.getsource(mod)
    assert "checkpoint_dir=" in src and "run_target_real_evals" in src
    assert "report_section_prefix=" in src
