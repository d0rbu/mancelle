"""Pin the new SharedState shape: no eval_wrappers; cached_context_provider exists."""

from __future__ import annotations

import dataclasses
import inspect

import scripts.train.conditional_encdec_train as mod


def test_shared_state_no_longer_has_eval_wrappers_field():
    fields = {f.name for f in dataclasses.fields(mod.SharedState)}
    assert "eval_wrappers" not in fields, (
        "Phase 2 PR-A removes eval_wrappers (legacy wrapper dispatch dict)"
    )


def test_shared_state_has_cached_context_provider_field():
    fields = {f.name for f in dataclasses.fields(mod.SharedState)}
    assert "cached_context_provider" in fields, (
        "Phase 2 PR-A introduces cached_context_provider for in-training "
        "CACHED-mode CORE eval and sample generation"
    )


def test_shared_state_no_longer_imports_legacy_wrapper_class():
    src = inspect.getsource(mod)
    parts = src.split("from steering.conditional.encoder_decoder.eval_adapter")
    # If the import block is entirely absent, the split produces one part — pass.
    # If it is present, verify EncoderDecoderEvalWrapper is not among the imported names.
    if len(parts) > 1:
        assert "EncoderDecoderEvalWrapper," not in parts[1].split(")")[0], (
            "EncoderDecoderEvalWrapper import must be removed"
        )
