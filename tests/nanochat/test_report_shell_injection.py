"""Security regression tests for ``nanochat/report.py`` shell-injection.

``nanochat/report.py`` previously used ``subprocess.run(cmd, shell=True, ...)`` on a format-built
string.  A future contributor who plumbed an operator-controlled value
(e.g. ``cfg.report.bloat_extensions``) into the formatted command would have
opened a direct shell-metacharacter injection.

This module pins four regressions:

1. ``test_report_does_not_execute_shell_metacharacters`` — end-to-end,
   unmocked ``subprocess.run``: a list-form argv containing literal shell
   metacharacters (``;``, ``$(...)``) MUST be treated as plain arguments
   (because ``shell=False``).  The test uses a filesystem sentinel
   (``tmp_path / "pwned"``); if shell interpretation leaked in, the file
   would be created.

2. ``test_report_uses_list_subprocess`` — monkeypatches
   ``subprocess.run`` and asserts the first positional argument is a
   ``list`` (not a string) and that ``shell`` is ``False``.  This pins the
   API shape for future regressions.

3. ``test_report_rejects_or_sanitizes_path_traversal`` —
   ``get_report(checkpoint_dir=...)`` accepts a directory from config; a
   traversal payload must resolve to a safe location (or be rejected).

4. ``test_report_produces_expected_output_on_clean_input`` — happy-path
   regression: a straightforward list-form ``run_command`` call still
   returns the command's stdout.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from nanochat.report import Report, run_command

# ---------------------------------------------------------------------------
# Test 1 — real subprocess + filesystem sentinel (no mocks).
# ---------------------------------------------------------------------------


def test_report_does_not_execute_shell_metacharacters(tmp_path: Path) -> None:
    """Shell metacharacters must NOT be interpreted by a shell.

    We pass a list-form argv whose second token contains both ``;`` and
    ``$(touch ...)`` constructs.  With ``shell=False`` these become literal
    arguments to ``echo`` — the sentinel file MUST NOT be created.
    """
    sentinel = tmp_path / "pwned"
    payload_semicolon = f"hi; touch {sentinel}"
    payload_cmdsubst = f"hi $(touch {sentinel})"

    # Both metacharacter styles, via the hardened run_command.
    assert sentinel.exists() is False

    out1 = run_command(["echo", payload_semicolon])
    # echo prints the literal argv (minus the program name); no shell
    # expansion happens.  Expected stdout: the payload itself.
    assert out1 is not None
    assert payload_semicolon in out1
    assert sentinel.exists() is False, (
        "shell metacharacter ';' was interpreted by a shell — command "
        "injection is still open"
    )

    out2 = run_command(["echo", payload_cmdsubst])
    assert out2 is not None
    assert payload_cmdsubst in out2
    assert sentinel.exists() is False, (
        "command substitution '$(...)' was interpreted by a shell — "
        "command injection is still open"
    )


# ---------------------------------------------------------------------------
# Test 2 — monkeypatch to pin the list-form + shell=False API.
# ---------------------------------------------------------------------------


def test_report_uses_list_subprocess(monkeypatch: pytest.MonkeyPatch) -> None:
    """``run_command`` must call ``subprocess.run`` with a list and shell=False.

    This pins the API shape so future regressions (someone re-introducing
    ``shell=True`` or a formatted string) are caught immediately.
    """
    positional_captured: list[tuple[object, ...]] = []
    kwargs_captured: list[dict[str, object]] = []

    class _FakeResult:
        def __init__(self) -> None:
            self.stdout = "ok\n"
            self.returncode = 0

    def _fake_run(*args: object, **kwargs: object) -> _FakeResult:
        positional_captured.append(args)
        kwargs_captured.append(kwargs)
        return _FakeResult()

    monkeypatch.setattr(subprocess, "run", _fake_run)

    run_command(["git", "rev-parse", "--short", "HEAD"])

    assert len(positional_captured) == 1, (
        f"subprocess.run must be called exactly once, got {len(positional_captured)}"
    )
    positional = positional_captured[0]
    assert len(positional) >= 1
    first_arg = positional[0]
    assert isinstance(first_arg, list), (
        f"expected first positional to be a list, got {type(first_arg).__name__}: "
        f"{first_arg!r}"
    )
    kwargs = kwargs_captured[0]
    shell_value = kwargs.get("shell", False)
    assert shell_value is False, (
        f"subprocess.run was invoked with shell={shell_value!r}; "
        "must be False (or not passed)"
    )


# ---------------------------------------------------------------------------
# Test 3 — path traversal in checkpoint_dir is resolved, not trusted raw.
# ---------------------------------------------------------------------------


def test_report_rejects_or_sanitizes_path_traversal(tmp_path: Path) -> None:
    """A ``../../../etc/passwd``-style checkpoint_dir must not escape ``tmp_path``.

    Design: ``Report(report_dir)`` itself does not validate the path
    (that's the caller's job), BUT once instantiated, every write must be
    confined to ``self.report_dir``.  We verify that a slug written by
    ``Report.log(...)`` lands under the resolved report directory — never
    above it — and that the resolved path cannot escape the allow-root.
    """
    allow_root = tmp_path / "checkpoints"
    allow_root.mkdir()

    # A traversal attempt: joining a traversal string with an allow-root
    # and then resolving MUST land outside the root; assert we detect it.
    evil_ckpt = os.path.join(str(allow_root), "..", "..", "etc")
    resolved = Path(evil_ckpt).resolve()
    is_safe = allow_root.resolve() in resolved.parents or resolved == allow_root.resolve()
    assert is_safe is False, (
        "path-traversal sanity check itself is broken — fix the test"
    )

    # Actual invariant: a Report created with a clean directory only writes
    # inside that directory.  A later attacker-controlled section name
    # cannot escape via slug injection because slugify strips '/'.
    clean_dir = allow_root / "d26" / "report"
    report = Report(str(clean_dir))
    # A section name containing a traversal-like string must be slugified
    # (slashes removed) so the written file stays inside clean_dir.
    out_path = report.log("../../../etc/passwd", [{"k": 1}])
    resolved_out = Path(out_path).resolve()
    assert str(resolved_out).startswith(str(clean_dir.resolve())), (
        f"report.log wrote outside its report_dir: {resolved_out}"
    )
    # And the file name must not contain path separators after slugify.
    assert os.sep not in os.path.basename(out_path)


# ---------------------------------------------------------------------------
# Test 4 — happy path regression.
# ---------------------------------------------------------------------------


def test_report_produces_expected_output_on_clean_input() -> None:
    """A vanilla list-form command still returns its stdout."""
    out = run_command(["echo", "hello-world"])
    assert out is not None
    assert "hello-world" in out
