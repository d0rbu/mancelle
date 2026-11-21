"""Slice 16 audit — property tests for ``paths._sanitize``.

Pins map contract C58 (the path-traversal sanitization invariants)
and addresses map P2-2 (the silent-bounce concern). Three property
groups:

1. **No path-traversal byte ever survives.** For any input
   containing ``/``, ``\\``, or ``..``, the output is the
   hash-based ``traversal_<16hex>`` form and contains none of those
   bytes — a fundamental safety property protecting against directory
   escape.

2. **Output character set is constrained.** All non-traversal output
   matches ``[A-Za-z0-9_\\-.]`` (only the safe character class).

3. **Deterministic.** Same input → same output (hash is stable).
"""

from __future__ import annotations

import re

from hypothesis import given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.paths import _sanitize

_SAFE_OUTPUT_PATTERN = re.compile(r"^[A-Za-z0-9_\-.]+$")


@given(s=st.text(min_size=1, max_size=64))
@settings(max_examples=200, derandomize=True)
def test_output_never_contains_path_separators_or_traversal(s: str) -> None:
    """C58 part A: no ``/``, ``\\``, or ``..`` in output."""
    out = _sanitize(s)
    assert "/" not in out
    assert "\\" not in out
    assert ".." not in out
    # And output is non-empty.
    assert out


@given(s=st.text(min_size=1, max_size=64))
@settings(max_examples=200, derandomize=True)
def test_output_matches_safe_character_class(s: str) -> None:
    """C58 part B: output is always safe-character-class only."""
    out = _sanitize(s)
    # Either ``traversal_<hex>`` (which trivially matches the safe class)
    # or a substituted form where any non-safe char became ``_``.
    assert _SAFE_OUTPUT_PATTERN.match(out), (
        f"Sanitized output {out!r} (from input {s!r}) contains "
        f"unsafe characters"
    )


@given(s=st.text(min_size=1, max_size=64))
@settings(max_examples=200, derandomize=True)
def test_sanitize_is_deterministic(s: str) -> None:
    """Same input always produces the same output."""
    a = _sanitize(s)
    b = _sanitize(s)
    assert a == b


@given(
    s=st.text(
        alphabet=st.characters(
            blacklist_characters=["/", "\\"],
            blacklist_categories=("Cs",),
        ),
        min_size=1,
        max_size=64,
    ).filter(lambda x: ".." not in x),
)
@settings(max_examples=200, derandomize=True)
def test_non_traversal_inputs_avoid_hash_form(s: str) -> None:
    """If the input is free of traversal bytes, the output should NOT
    be the ``traversal_<hex>`` hash form (only suspicious inputs go
    through the hash)."""
    out = _sanitize(s)
    # Non-traversal inputs may still get bounced to ``_`` when they
    # collapse to dot-only or empty after substitution; that's fine.
    # The contract being pinned: hash form is reserved for actually-
    # suspicious inputs.
    assert not out.startswith("traversal_"), (
        f"Input {s!r} (no traversal bytes) was incorrectly bounced to "
        f"hash form {out!r}"
    )


def test_legitimate_typo_with_double_dot_bounced_to_hash() -> None:
    """Map P2-2: ``..-foo`` (operator typo) is silently bounced to
    ``traversal_<hex>``. This is the documented behaviour; the test
    pins it so a future relaxation surfaces as a test failure.
    """
    out = _sanitize("..-foo")
    assert out.startswith("traversal_"), (
        f"Map P2-2: ``..-foo`` should bounce to traversal hash; got {out!r}"
    )
    assert ".." not in out
    assert "foo" not in out


def test_dot_only_inputs_bounce_to_underscore() -> None:
    """Dot-only inputs (".", "..", ...) — though "." has no traversal
    bytes — should still not produce a leading-dot path.
    """
    # Pure "." has no "..", so it does not hit the traversal branch.
    # The dot-collapse path then returns "_".
    assert _sanitize(".") == "_"
    # ``"..."`` has ``..`` so hits the traversal branch.
    out = _sanitize("...")
    assert out.startswith("traversal_")


def test_normal_safe_run_tag_passes_through() -> None:
    """Sane run tags pass through with no substitution."""
    assert _sanitize("safe_tag_42") == "safe_tag_42"
    assert _sanitize("20260101-120000-abcdef12") == "20260101-120000-abcdef12"
