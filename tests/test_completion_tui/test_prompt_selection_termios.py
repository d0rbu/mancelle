"""Real pty-based tests for ``prompt_selection``.

``scripts.completion_tui.discovery.prompt_selection`` reads user input
directly through ``termios``/``tty``/``select.select`` on ``sys.stdin``,
which means patching ``builtins.input`` (as the legacy test_discovery
tests did) exercises nothing.

These tests drive a real pseudo-terminal via :mod:`pty` and feed
keystrokes through the master end, giving coverage of the primary
interactive path. Skipped on Windows, which lacks ``termios``/``pty``.
"""

from __future__ import annotations

import importlib
import os
import sys
import threading
import time
from typing import Any
from unittest.mock import patch

import pytest

from scripts.completion_tui.discovery import TUIConfig, prompt_selection

pytestmark = pytest.mark.skipif(
    sys.platform == "win32",
    reason="termios/pty are not available on Windows",
)

pty: Any | None
try:
    pty = importlib.import_module("pty")
except ImportError:  # pragma: no cover - defensive: pty is stdlib on POSIX
    pty = None


def _configs() -> list[TUIConfig]:
    return [
        TUIConfig("d26", "/a", "s128", 100, "/f", "/p", 128, 11, False, "label1"),
        TUIConfig("d27", "/b", "s64", 200, "/g", "/q", 64, 11, False, "label2"),
    ]


def _run_prompt_on_pty(
    configs: list[TUIConfig],
    feed: bytes,
    *,
    feed_delay: float = 0.05,
) -> TUIConfig | BaseException:
    """Invoke ``prompt_selection`` with ``sys.stdin`` replaced by a real pty slave.

    Returns the selected config or the raised exception.
    """
    if pty is None:  # pragma: no cover
        pytest.skip("pty module unavailable")

    master_fd, slave_fd = pty.openpty()
    try:
        slave_reader = os.fdopen(slave_fd, "rb", buffering=0, closefd=False)

        # Redirect sys.stdin to the slave end of the pty so ``prompt_selection``
        # sees a real terminal (termios.tcgetattr requires one).
        class _SlaveStdin:
            def fileno(self) -> int:
                return slave_fd

            def read(self, n: int) -> str:
                data = slave_reader.read(n)
                return data.decode("utf-8", errors="replace") if data else ""

        result_box: dict[str, object] = {}

        def _target() -> None:
            try:
                result_box["value"] = prompt_selection(configs)
            except BaseException as exc:  # noqa: BLE001 - we propagate it
                result_box["error"] = exc

        with patch.object(sys, "stdin", _SlaveStdin()):
            worker = threading.Thread(target=_target, daemon=True)
            worker.start()

            # Give the prompt a moment to enter its read loop, then deliver keystrokes.
            time.sleep(feed_delay)
            os.write(master_fd, feed)

            worker.join(timeout=15.0)

        if "error" in result_box:
            err = result_box["error"]
            assert isinstance(err, BaseException)
            return err
        if "value" in result_box:
            val = result_box["value"]
            assert isinstance(val, TUIConfig)
            return val
        raise TimeoutError("prompt_selection did not return within 15s")
    finally:
        try:
            os.close(master_fd)
        except OSError:
            pass
        try:
            os.close(slave_fd)
        except OSError:
            pass


def test_default_selection_on_enter() -> None:
    """Pressing Enter with no digits returns the first config."""
    configs = _configs()
    result = _run_prompt_on_pty(configs, b"\r")
    assert result is configs[0]


def test_explicit_digit_selection() -> None:
    """Typing '2' then Enter returns the second config."""
    configs = _configs()
    result = _run_prompt_on_pty(configs, b"2\r")
    assert result is configs[1]


def test_ctrl_d_exits() -> None:
    """Ctrl-D (EOT) raises SystemExit."""
    configs = _configs()
    result = _run_prompt_on_pty(configs, b"\x04")
    assert isinstance(result, SystemExit)


def test_backspace_corrects_input() -> None:
    """Typing '12', backspace, '\\r' selects config 1 (first)."""
    configs = _configs()
    # '1' '2' BS BS '\r' -> buffer empty -> default (first config)
    result = _run_prompt_on_pty(configs, b"12\x7f\x7f\r")
    assert result is configs[0]
