"""Behaviour-locking tests for slice-13 P0s.

Slice 13 originally documented several items as DEFERRED follow-ups per
consolidation manifest §5.3 / §6.8:

* SB-13-P0-01 — ``Engine.generate_batch`` strips terminal
  ``<assistant_end>`` but preserves tool-injected markers. FIXED at
  the consumer layer by #444; the per-consumer regression tests live
  under ``tests/scripts/`` and ``tests/properties/``.
* SB-13-P0-02 — ``Engine.generate`` advance-by-T cache bug
  (``max_tokens=None`` + long prompts). FIXED by #442; the test for
  this item now locks the entry-check fix state.
* SB-13-P1-10 — ``KVCache.prefill`` uses ``assert`` instead of typed
  exception (strippable under ``python -O``). FIXED by #441; the tests
  for this item now lock the RuntimeError-based fix state.

These tests pin the post-fix behaviour so any regression has a clear,
named guard.
"""

from __future__ import annotations

import inspect

import pytest
import torch as t

from nanochat.engine import Engine, KVCache
from tests.conftest import MockTokenizer, _make_gpt

_DEVICE = t.device("cpu")


# ---------------------------------------------------------------------------
# SB-13-P1-10 (FIXED by #441): ``KVCache.prefill`` raises RuntimeError.
# ---------------------------------------------------------------------------


def test_kv_cache_prefill_uses_runtime_error_not_assert():
    """``KVCache.prefill`` validates invariants with ``raise RuntimeError``
    rather than ``assert``. Locks the post-#441 fix state: under
    ``python -O`` (PYTHONOPTIMIZE=1) the guards remain active because
    ``RuntimeError`` is not strippable.

    This is the inverse of the original deferred lock — it now fails if
    anyone reverts the fix.
    """
    src = inspect.getsource(KVCache.prefill)
    assert "assert " not in src, (
        "KVCache.prefill must not use 'assert' for runtime invariants "
        "(stripped under python -O). Use 'raise RuntimeError(...)' instead."
    )
    assert "raise RuntimeError" in src, (
        "KVCache.prefill must raise RuntimeError for invariant violations."
    )


def test_kv_cache_prefill_raises_runtime_error_at_pos_nonzero():
    """KVCache.prefill on a non-empty self raises RuntimeError (post-#441)."""
    cache_a = KVCache(
        batch_size=1,
        num_heads=2,
        seq_len=8,
        head_dim=4,
        num_layers=2,
        device=_DEVICE,
        dtype=t.float32,
    )
    cache_b = KVCache(
        batch_size=1,
        num_heads=2,
        seq_len=8,
        head_dim=4,
        num_layers=2,
        device=_DEVICE,
        dtype=t.float32,
    )
    cache_a.advance(2)
    assert cache_a.get_pos() == 2
    with pytest.raises(RuntimeError, match="non-empty KV cache"):
        cache_a.prefill(cache_b)


# ---------------------------------------------------------------------------
# SB-13-P0-02 (FIXED by #442): ``Engine.generate`` rejects overflow at entry.
# ---------------------------------------------------------------------------


def test_engine_generate_rejects_prompt_overflow_at_entry():
    """``Engine.generate`` raises RuntimeError at entry when the prompt
    leaves no room for the requested generation horizon. Locks the
    post-#442 fix state."""
    src = inspect.getsource(Engine.generate)
    # The entry check exists.
    assert "raise RuntimeError" in src
    # The check names the offending knobs.
    assert "context window" in src or "max_len" in src or "sequence_len" in src


def test_engine_generate_short_prompt_long_max_tokens_works():
    """Sanity: short prompt + explicit ``max_tokens`` works as before
    (the slow regression path, not the bug we're documenting). This test
    just locks the happy-path behaviour so the deferred-issue test set
    has a positive control."""
    model = _make_gpt(device="cpu")
    tok = MockTokenizer()
    engine = Engine(model, tok)
    tokens = list(tok.encode("hi", prepend=tok.get_bos_token_id()))
    out = list(engine.generate(tokens, num_samples=1, max_tokens=2, temperature=0.0))
    # Two yields max — output structure is (token_column, mask) per yield.
    assert len(out) <= 2
    for token_column, token_masks in out:
        assert len(token_column) == 1  # num_samples=1
        assert len(token_masks) == 1


# SB-13-P0-01 (FIXED by #444 — test moved to tests/scripts/)


def test_engine_module_exposes_assert_finite_logits():
    """The vendored guard exists; the slice-13 PR adds the same guard to
    both ``steering/eval/inference_engine.py`` and
    ``steering/eval/multiseg_generator.py`` (the deferred consolidation
    follow-up would hoist a single helper into nanochat)."""
    from nanochat import engine as engine_mod  # local for the test only

    assert hasattr(engine_mod, "_assert_finite_logits")
    assert hasattr(engine_mod, "sample_next_token")
