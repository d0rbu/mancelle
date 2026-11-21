"""Unit tests for steering/nnsight_compat.py — slice 01 Phase 1 audit.

Pins fix for plan item A-11: idempotency of unpatch_nnsight + ImportError path.
"""

import sys

from steering.nnsight_compat import unpatch_nnsight


def test_unpatch_when_nnsight_not_installed_returns_false(monkeypatch):
    """If nnsight cannot be imported, unpatch_nnsight returns False."""
    # Remove any cached nnsight import + simulate ImportError on `import nnsight`.
    monkeypatch.setitem(sys.modules, "nnsight", None)
    # If `nnsight` is mapped to None, `import nnsight` raises ImportError; the
    # function returns False.
    assert unpatch_nnsight() is False


def test_unpatch_double_call_is_safe(monkeypatch):
    """Calling unpatch_nnsight twice does not raise."""
    monkeypatch.setitem(sys.modules, "nnsight", None)
    # First call returns False (nnsight absent).
    result1 = unpatch_nnsight()
    # Second call still returns False; no exception.
    result2 = unpatch_nnsight()
    assert result1 is False
    assert result2 is False
