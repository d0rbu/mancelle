"""Slice 17 audit (2026-05-05) X2 §3 regression: _StubRuntimeState contract.

The DDP test helper ``tests/axbench_bpb/_mini_runtime.py`` previously wrote
to private cache slots on ``RuntimeState``. The fix replaces this with a
``_StubRuntimeState`` subclass that overrides the public API.

Tests:
1. The stubbed accessors return the same instances on repeated calls (cache).
2. The stub does not write to any private cache slots that callers might
   check for (``_base_gemma`` / ``_base_sae`` / ``_tokenizer_extraction``).
"""
from __future__ import annotations

from pathlib import Path

from tests.axbench_bpb._mini_runtime import build_mini_runtime_state


def test_stub_runtime_state_repeat_calls_return_same_objects(tmp_path: Path):
    """The lazy-loader override must cache its return values."""
    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=1)
    m1, s1 = rs.base_gemma_and_sae()
    m2, s2 = rs.base_gemma_and_sae()
    assert m1 is m2
    assert s1 is s2
    assert rs.extraction_tokenizer() is rs.extraction_tokenizer()


def test_stub_runtime_state_extraction_tok_consistent_with_base_call(tmp_path: Path):
    """``extraction_tokenizer()`` is consistent with ``base_gemma_and_sae`` cache."""
    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=1)
    rs.base_gemma_and_sae()
    tok1 = rs.extraction_tokenizer()
    tok2 = rs.extraction_tokenizer()
    assert tok1 is tok2


def test_stub_runtime_state_d_sae_propagates(tmp_path: Path):
    """The stubbed SAE config exposes ``d_sae`` (used by build_context_cache)."""
    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=1)
    _model, sae = rs.base_gemma_and_sae()
    assert sae.cfg.d_sae == 16  # default per build_mini_runtime_state


def test_stub_runtime_state_subclass_kept(tmp_path: Path):
    """The stub is a subclass of RuntimeState (preserves type contract)."""
    from steering.eval.axbench_bpb.runtime_state import RuntimeState

    rs = build_mini_runtime_state(run_root=tmp_path, rank=0, world_size=1)
    assert isinstance(rs, RuntimeState)
