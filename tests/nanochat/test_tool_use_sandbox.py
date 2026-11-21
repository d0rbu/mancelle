"""Tool-use sandbox documentation tests.

Manifest §4.3 / §7 / map P3-01 (SB-13-P3-01):

    The contract advertised by ``docs/pipelines/serving.md:168`` previously
    claimed "(5s) and memory limits (256MB)"; the real defaults are 3s and
    no memory limit. ``use_calculator`` only invokes ``eval_with_timeout``
    — the heavier ``reliability_guard`` sandbox in
    ``nanochat/execution.py`` is NOT wired in. The doc was updated; this
    file pins both the doc and the code so future drift surfaces
    immediately.
"""

from __future__ import annotations

import inspect
import re
from pathlib import Path

from nanochat import engine as engine_mod
from nanochat.engine import eval_with_timeout, use_calculator

_ROOT = Path(__file__).resolve().parents[2]


def _read_serving_md() -> str:
    """Read the serving doc directly so the test pins the file content."""
    return (_ROOT / "docs" / "pipelines" / "serving.md").read_text()


# ---------------------------------------------------------------------------
# Code defaults — what the production sandbox actually does.
# ---------------------------------------------------------------------------


def test_eval_with_timeout_default_max_time_is_three():
    """``eval_with_timeout(formula, max_time=3)`` — default timeout is 3s,
    NOT 5s as the old doc claimed."""
    sig = inspect.signature(eval_with_timeout)
    assert sig.parameters["max_time"].default == 3


def test_use_calculator_does_not_call_reliability_guard():
    """The ``use_calculator`` body does NOT mention ``reliability_guard``;
    only ``eval_with_timeout`` is wired in."""
    src = inspect.getsource(use_calculator)
    assert "reliability_guard" not in src


def test_engine_module_does_not_import_execution_reliability_guard():
    """The engine module must not import ``reliability_guard`` from
    ``nanochat.execution`` — that sandbox is for ``execute_code``, not the
    inference engine."""
    src = inspect.getsource(engine_mod)
    # The engine module shouldn't pull in reliability_guard.
    assert "reliability_guard" not in src


# ---------------------------------------------------------------------------
# Doc content — pin the corrected text in serving.md so future drift surfaces.
# ---------------------------------------------------------------------------


def test_serving_doc_describes_3s_timeout():
    """``docs/pipelines/serving.md`` states 3s timeout (not 5s)."""
    content = _read_serving_md()
    assert "3s timeout" in content


def test_serving_doc_describes_no_memory_limit():
    """``docs/pipelines/serving.md`` states there is no memory limit (the
    old "(256MB)" claim was incorrect)."""
    content = _read_serving_md()
    assert "no memory limit" in content
    # The old wrong claim must NOT appear.
    assert "(256MB)" not in content
    assert "(5s)" not in content


def test_serving_doc_does_not_imply_reliability_guard_is_wired():
    """The doc must NOT imply ``reliability_guard`` is wired into the
    engine path. It explicitly notes the gap."""
    content = _read_serving_md()
    assert "reliability_guard" in content  # the doc explicitly disclaims it
    # Confirm the disclaim is in the right shape (negation).
    assert re.search(
        r"reliability_guard.*NOT.*wired", content, flags=re.IGNORECASE
    ), (
        "expected the serving doc to explicitly state that "
        "reliability_guard is NOT wired into the engine path"
    )


# ---------------------------------------------------------------------------
# Behavioural pins — confirm both the eval_with_timeout default and
# use_calculator wiring lines up with the doc.
# ---------------------------------------------------------------------------


def test_eval_with_timeout_swallows_exceptions_and_returns_none():
    """``eval_with_timeout`` returns ``None`` (never raises) for any
    failure — pinned in ``test_engine_robustness.py`` already; we re-pin
    here to make slice 13's contract self-contained."""
    assert eval_with_timeout("1/0") is None
    assert eval_with_timeout("import os") is None
    assert eval_with_timeout("foo()") is None


def test_use_calculator_returns_none_for_anything_dangerous():
    """High-level: any expression that escapes the allow-list returns
    ``None``. This is a low-grade smoke test; per-pattern behaviour is in
    ``test_use_calculator.py``."""
    assert use_calculator("__import__('os')") is None
    assert use_calculator("open('/etc/passwd')") is None
    assert use_calculator("eval('1+1')") is None
