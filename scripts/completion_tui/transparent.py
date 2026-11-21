"""Transparent terminal background support.

Monkey-patches Rich's Color.get_ansi_codes() to emit the ANSI default
background escape (\\x1b[49m]) instead of explicit RGB for the Textual
theme's primary background color. This allows terminal emulators with
transparency (Ghostty, Alacritty, iTerm2) to show through.

Must be called after App construction but before app.run() — once
rendering starts, styles cache their ANSI output and the patch only
affects fresh style renders.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol
from unittest.mock import patch

from rich.color import Color as RichColor
from textual.color import Color as TextualColor
from textual.strip import Strip

if TYPE_CHECKING:
    from textual.app import App

_ANSI_DEFAULT_BG = ("49",)
# The original lru_cache-wrapped Color.get_ansi_codes, saved for restore.
_original_get_ansi_codes = None
_patch_handle: _PatchHandle | None = None


class _PatchHandle(Protocol):
    """Runtime patch lifecycle used for the Rich class method."""

    def start(self) -> object: ...

    def stop(self) -> object: ...


def enable_transparent_background(app: App) -> None:
    """Patch Rich Color to emit ANSI default bg for the theme background."""
    global _original_get_ansi_codes, _patch_handle
    if _original_get_ansi_codes is not None:
        return

    theme_bg_hex = app.theme_variables.get("background")
    if theme_bg_hex is None:
        return

    bg = TextualColor.parse(theme_bg_hex)
    target = (bg.r, bg.g, bg.b)

    original_get_ansi_codes = RichColor.get_ansi_codes

    def _patched_get_ansi_codes(self, foreground=True):
        if not foreground:
            triplet = self.get_truecolor()
            if (triplet.red, triplet.green, triplet.blue) == target:
                return _ANSI_DEFAULT_BG
        return original_get_ansi_codes(self, foreground)

    patch_handle = patch.object(
        RichColor, "get_ansi_codes", _patched_get_ansi_codes
    )
    patch_handle.start()
    _original_get_ansi_codes = original_get_ansi_codes
    _patch_handle = patch_handle
    Strip.render_ansi.cache_clear()


def disable_transparent_background() -> None:
    """Restore the original Color.get_ansi_codes."""
    global _original_get_ansi_codes, _patch_handle
    if _original_get_ansi_codes is None:
        return
    assert _patch_handle is not None
    _patch_handle.stop()
    _patch_handle = None
    _original_get_ansi_codes = None
    Strip.render_ansi.cache_clear()
