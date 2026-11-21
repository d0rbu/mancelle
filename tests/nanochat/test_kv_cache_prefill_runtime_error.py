"""Per-invariant tests for KVCache.prefill's RuntimeError guards (issue #441).

Pre-fix: 3 `assert` statements validated the shape, position, and
sequence-length invariants. Under `python -O`, all three are stripped.

Post-fix: each invariant violation raises RuntimeError with a message
naming the offending attribute(s).
"""

from __future__ import annotations

import pytest
import torch as t

from nanochat.engine import KVCache

_DEVICE = t.device("cpu")
_DTYPE = t.float32


def _make_cache(
    *,
    batch_size: int = 1,
    num_heads: int = 2,
    seq_len: int = 8,
    head_dim: int = 4,
    num_layers: int = 2,
) -> KVCache:
    return KVCache(
        batch_size=batch_size,
        num_heads=num_heads,
        seq_len=seq_len,
        head_dim=head_dim,
        num_layers=num_layers,
        device=_DEVICE,
        dtype=_DTYPE,
    )


# ---------------------------------------------------------------------------
# Invariant 1: target cache (self) must be empty (pos == 0).
# ---------------------------------------------------------------------------


def test_prefill_raises_when_self_position_is_nonzero():
    self_cache = _make_cache()
    other_cache = _make_cache()
    self_cache.advance(3)
    with pytest.raises(RuntimeError) as excinfo:
        self_cache.prefill(other_cache)
    msg = str(excinfo.value)
    assert "non-empty KV cache" in msg
    # The message must name the offending position so triage is easy.
    assert "3" in msg


# ---------------------------------------------------------------------------
# Invariant 2: shape parity (n_layers, n_heads, head_dim).
# ---------------------------------------------------------------------------


def test_prefill_raises_on_n_layers_mismatch():
    self_cache = _make_cache(num_layers=2)
    other_cache = _make_cache(num_layers=4)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        self_cache.prefill(other_cache)


def test_prefill_raises_on_n_heads_mismatch():
    self_cache = _make_cache(num_heads=2)
    other_cache = _make_cache(num_heads=4)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        self_cache.prefill(other_cache)


def test_prefill_raises_on_head_dim_mismatch():
    self_cache = _make_cache(head_dim=4)
    other_cache = _make_cache(head_dim=8)
    with pytest.raises(RuntimeError, match="shape mismatch"):
        self_cache.prefill(other_cache)


def test_prefill_shape_mismatch_message_names_attributes():
    self_cache = _make_cache(num_heads=2, head_dim=4)
    other_cache = _make_cache(num_heads=3, head_dim=4)
    with pytest.raises(RuntimeError) as excinfo:
        self_cache.prefill(other_cache)
    msg = str(excinfo.value)
    assert "n_heads" in msg
    assert "2" in msg and "3" in msg


# ---------------------------------------------------------------------------
# Invariant 3: target seq_len must be at least source seq_len.
# ---------------------------------------------------------------------------


def test_prefill_raises_when_self_max_seq_len_too_small():
    self_cache = _make_cache(seq_len=4)
    other_cache = _make_cache(seq_len=8)
    with pytest.raises(RuntimeError) as excinfo:
        self_cache.prefill(other_cache)
    msg = str(excinfo.value)
    assert "max_seq_len" in msg
    assert "4" in msg and "8" in msg


# ---------------------------------------------------------------------------
# Happy path.
# ---------------------------------------------------------------------------


def test_prefill_succeeds_when_all_invariants_satisfied():
    self_cache = _make_cache(seq_len=16)
    other_cache = _make_cache(seq_len=8)
    # Fill `other_cache` with some content + advance its position.
    other_cache.k_cache[:, :, :3, :, :] = 1.0
    other_cache.v_cache[:, :, :3, :, :] = 2.0
    other_cache.advance(3)
    assert other_cache.get_pos() == 3
    # Now prefill self from other.
    self_cache.prefill(other_cache)
    assert self_cache.get_pos() == 3
    # Source slots are copied verbatim.
    assert t.all(self_cache.k_cache[:, :, :3, :, :] == 1.0)
    assert t.all(self_cache.v_cache[:, :, :3, :, :] == 2.0)
    # Slots beyond the copied range remain zeroed.
    assert t.all(self_cache.k_cache[:, :, 3:, :, :] == 0.0)


def test_prefill_succeeds_with_equal_seq_len():
    """Boundary case: self.max_seq_len == other.max_seq_len is allowed."""
    self_cache = _make_cache(seq_len=8)
    other_cache = _make_cache(seq_len=8)
    other_cache.advance(2)
    self_cache.prefill(other_cache)
    assert self_cache.get_pos() == 2


def test_prefill_succeeds_when_other_pos_is_zero():
    """Pre-condition: self.pos == 0 (the *target* must be empty).
    `other` is allowed to be empty too (degenerate but legal)."""
    self_cache = _make_cache()
    other_cache = _make_cache()
    self_cache.prefill(other_cache)
    assert self_cache.get_pos() == 0
