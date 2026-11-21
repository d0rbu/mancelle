"""Correctness tests for ``nanochat.engine.use_calculator``.

Manifest §7 ("ADD: ``use_calculator`` correctness tests"). The vendored
sandbox combines:
1. A pure-arithmetic allow-list (chars in ``"0123456789*+-/.() "``, no ``**``).
2. A string-op path keyed on ``.count(``, with a forbidden-pattern list.

Existing coverage in ``test_engine_robustness.py`` only confirms that
``eval_with_timeout`` swallows exceptions; the calculator's grammar wasn't
tested. We do NOT modify ``nanochat/engine.py`` (audit-only).
"""

from __future__ import annotations

import pytest

from nanochat.engine import use_calculator

# ---------------------------------------------------------------------------
# Pure-arithmetic happy path
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr,expected",
    [
        ("1+1", 2),
        ("2+3*4", 14),
        ("(2+3)*4", 20),
        ("100-1", 99),
        ("8/2", 4.0),
        ("7.5*2", 15.0),
        ("3.14+0.86", pytest.approx(4.0, rel=1e-9)),
        ("0", 0),
        ("(1)", 1),
    ],
)
def test_use_calculator_pure_arithmetic(expr, expected):
    """Pure-arithmetic expressions evaluate via ``eval_with_timeout``."""
    assert use_calculator(expr) == expected


def test_use_calculator_strips_thousands_commas():
    """The calculator removes commas before parsing — '1,000+1' → 1001."""
    assert use_calculator("1,000+1") == 1001


def test_use_calculator_rejects_power_operator():
    """``**`` is rejected even in the pure-arithmetic path."""
    assert use_calculator("2**3") is None


def test_use_calculator_passes_with_unary_minus():
    """Unary minus is allowed (- is in the char allow-list)."""
    assert use_calculator("-1+2") == 1


# ---------------------------------------------------------------------------
# String-op path: .count() is the only allowed method.
# ---------------------------------------------------------------------------


def test_use_calculator_count_basic():
    """``.count(`` is the only string-op gate; basic count works."""
    assert use_calculator("'banana'.count('a')") == 3


def test_use_calculator_count_double_quotes():
    """Both single and double quotes allowed."""
    assert use_calculator('"banana".count("a")') == 3


def test_use_calculator_count_substring():
    """Counting longer substrings works."""
    assert use_calculator("'aaaa'.count('aa')") == 2


def test_use_calculator_string_op_without_count_returns_none():
    """A string-op expression that does NOT contain ``.count(`` returns
    ``None`` even if the chars are in the allowed set."""
    # Method exists but isn't .count, so the gate rejects.
    assert use_calculator("'banana'.upper()") is None


# ---------------------------------------------------------------------------
# Dangerous-pattern allow-list (case-insensitive substring match).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "__import__('os').system('ls')",
        "import os",
        "exec('1+1')",
        "eval('1+1')",
        "compile('1+1', '', 'eval')",
        "open('foo')",
        "file('foo')",
        "input()",
        "raw_input()",
        "globals()",
        "locals()",
        "vars()",
        "dir()",
        "getattr(x, 'y')",
        "setattr(x, 'y', z)",
        "delattr(x, 'y')",
        "hasattr(x, 'y')",
    ],
)
def test_use_calculator_rejects_dangerous_patterns(expr):
    """Each pattern in the dangerous-patterns list returns ``None``.

    NOTE: most of these would be caught by the char-allowlist (e.g.
    ``[``, ``,``, ``;``, ``=``) before the dangerous-pattern check fires;
    the test still asserts the public contract (``None`` return) without
    distinguishing which gate caught it. That's intentional — the public
    interface is "returns None for anything dangerous"; the layered checks
    are an implementation detail.
    """
    assert use_calculator(expr) is None


@pytest.mark.parametrize(
    "expr",
    [
        # Within the allow-list character set but contains dangerous patterns.
        "'a'.count('exec')",  # contains 'exec'
        "'a'.count('eval')",  # contains 'eval'
        "'__'.count('a')",    # contains '__'
        "'a'.count('IMPORT')",  # case-insensitive
        "'a'.count('OPEN')",
    ],
)
def test_use_calculator_dangerous_pattern_in_count_arg(expr):
    """Even inside the allow-list character grammar, dangerous-pattern
    substrings (case-insensitive) shut down the eval. NOTE: this is
    arguably over-strict — counting the LITERAL string "exec" in another
    string is harmless — but it locks the documented behaviour."""
    assert use_calculator(expr) is None


# ---------------------------------------------------------------------------
# Char-allow-list rejects.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "expr",
    [
        "1+1;print(1)",   # semicolon
        "1+1\n",          # newline
        "1=1",            # equality op
        "1<2",            # comparison
        "x[0]",           # subscript
        "a@b",            # decorator
        "1#comment",      # comment
        "1+1\t",          # tab
    ],
)
def test_use_calculator_rejects_chars_outside_allowlist(expr):
    """Anything that isn't pure-arithmetic AND isn't an allowed string-op
    returns ``None``."""
    assert use_calculator(expr) is None


# ---------------------------------------------------------------------------
# Edge cases.
# ---------------------------------------------------------------------------


def test_use_calculator_empty_string():
    """Empty string is technically an empty char-allow-list match for
    arithmetic; ``eval('')`` → ``SyntaxError`` → ``None``."""
    assert use_calculator("") is None


def test_use_calculator_just_operators():
    """``+++`` evaluates to a SyntaxError → None."""
    assert use_calculator("+++") is None


def test_use_calculator_division_by_zero_returns_none():
    """Divide-by-zero → ``ZeroDivisionError`` → swallowed by
    ``eval_with_timeout`` → None."""
    assert use_calculator("1/0") is None


def test_use_calculator_letters_not_in_arithmetic_set():
    """Letters in the expression force the string-op path; without
    ``.count(``, return None."""
    assert use_calculator("abc") is None
