"""Smoke import tests for the ``scripts.completion_tui`` package surface.

Slice 18 audit (surfaces): catches import-time regressions in the TUI
package — broken cross-slice imports, missing modules, accidental top-level
side effects that crash, etc. Full E2E behaviour is covered by the rest of
the ``tests/test_completion_tui/`` suite; this file specifically guards the
operator-facing ``python -m scripts.completion_tui`` entrypoint and the
documented sub-module surface.
"""

from __future__ import annotations

import importlib


def test_completion_tui_package_imports() -> None:
    """``scripts.completion_tui`` is a Textual TUI package; full E2E tests
    live in ``tests/test_completion_tui/``. Slice 18 catches import-time
    regressions in the package surface."""
    importlib.import_module("scripts.completion_tui")
    importlib.import_module("scripts.completion_tui.app")
    importlib.import_module("scripts.completion_tui.state")
    importlib.import_module("scripts.completion_tui.discovery")
    importlib.import_module("scripts.completion_tui.data_loader")
    importlib.import_module("scripts.completion_tui.generation")


def test_completion_tui_main_module_callable() -> None:
    """``python -m scripts.completion_tui`` resolves to ``__main__.main``;
    ensure the entrypoint exists."""
    main_mod = importlib.import_module("scripts.completion_tui.__main__")
    assert hasattr(main_mod, "main"), "scripts.completion_tui.__main__ missing main()"
