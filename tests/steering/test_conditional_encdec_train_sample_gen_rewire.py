"""Pin: sample generation builds a cached full-context engine per prompt.

No legacy eval_wrappers iteration or no_features/unconditioned branch.
No hardcoded prompt list — must read cfg.sample_prompts."""

from __future__ import annotations

import inspect

import scripts.train.conditional_encdec_train as mod


def test_sample_gen_uses_build_engine_for_features():
    src = inspect.getsource(mod)
    assert "build_engine_for_features" in src, (
        "Sample generation rewire must construct engines via "
        "build_engine_for_features"
    )


def test_sample_gen_uses_cfg_sample_prompts_not_hardcoded_list():
    src = inspect.getsource(mod)
    assert "cfg.sample_prompts" in src, (
        "Sample generation must read prompts from cfg.sample_prompts"
    )
    code_only = "\n".join(
        line for line in src.splitlines()
        if not line.strip().startswith("#")
    )
    assert '"The capital of France is"' not in code_only, (
        "Hardcoded sample prompts must move to configs/training/conditional_encdec_train.yaml"
    )


def test_sample_gen_uses_cached_full_context_provider():
    src = inspect.getsource(mod)
    anchor = "def _run_sample_generation"
    idx = src.find(anchor)
    assert idx != -1, f"Expected anchor {anchor!r} not found in source"
    sample_section = src[idx:]
    sample_section = sample_section[: sample_section.find("# ---------------------------------------------------------------------------", 1)]
    assert len(sample_section) > 0, "Sample-gen section is unexpectedly empty"
    assert "cached_context_provider" in sample_section
    assert "no_features" not in sample_section


def test_sample_gen_no_longer_iterates_eval_wrappers_dict():
    src = inspect.getsource(mod)
    assert "for mode_name, wrapper in eval_wrappers.items()" not in src
