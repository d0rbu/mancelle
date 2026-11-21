"""Tests for transparent terminal background patch."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rich.color import Color as RichColor
from rich.color import ColorSystem, ColorType
from rich.color_triplet import ColorTriplet
from rich.console import Console
from rich.segment import Segment
from rich.style import Style
from textual.strip import Strip

from scripts.completion_tui.transparent import (
    disable_transparent_background,
    enable_transparent_background,
)


@pytest.fixture(autouse=True)
def _cleanup_patch():
    """Ensure patch is removed after every test."""
    yield
    disable_transparent_background()


def _make_app_mock(bg_hex: str = "#121212") -> MagicMock:
    """Create a mock App with theme_variables."""
    app = MagicMock()
    app.theme_variables = {"background": bg_hex}
    return app


def _make_truecolor(r: int, g: int, b: int) -> RichColor:
    """Create a TRUECOLOR Rich Color."""
    return RichColor(
        f"#{r:02x}{g:02x}{b:02x}",
        ColorType.TRUECOLOR,
        number=None,
        triplet=ColorTriplet(r, g, b),
    )


class TestMatchingBackground:
    def test_returns_ansi_default(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        color = _make_truecolor(18, 18, 18)
        assert color.get_ansi_codes(foreground=False) == ("49",)

    def test_foreground_unchanged(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        color = _make_truecolor(18, 18, 18)
        assert color.get_ansi_codes(foreground=True) == (
            "38", "2", "18", "18", "18",
        )


class TestNonMatchingBackground:
    def test_different_rgb_unchanged(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        color = _make_truecolor(255, 0, 0)
        assert color.get_ansi_codes(foreground=False) == (
            "48", "2", "255", "0", "0",
        )


class TestIdempotent:
    def test_double_enable_same_behavior(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        enable_transparent_background(app)
        color = _make_truecolor(18, 18, 18)
        assert color.get_ansi_codes(foreground=False) == ("49",)


class TestDisable:
    def test_restores_original_behavior(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        disable_transparent_background()
        color = _make_truecolor(18, 18, 18)
        assert color.get_ansi_codes(foreground=False) == (
            "48", "2", "18", "18", "18",
        )


class TestCacheCleared:
    def test_strip_cache_cleared_on_enable(self):
        app = _make_app_mock("#121212")
        # Prime the Strip.render_ansi cache
        style = Style(bgcolor=RichColor.parse("#121212"))
        Strip.render_ansi(style, ColorSystem.TRUECOLOR)
        assert Strip.render_ansi.cache_info().currsize > 0

        enable_transparent_background(app)
        assert Strip.render_ansi.cache_info().currsize == 0

    def test_strip_cache_cleared_on_disable(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)

        style = Style(bgcolor=RichColor.parse("#121212"))
        Strip.render_ansi(style, ColorSystem.TRUECOLOR)

        disable_transparent_background()
        assert Strip.render_ansi.cache_info().currsize == 0


class TestNoThemeBackground:
    def test_none_background_is_noop(self):
        app = MagicMock()
        app.theme_variables = {}
        enable_transparent_background(app)
        # Should not have patched — original behavior preserved
        color = _make_truecolor(18, 18, 18)
        assert color.get_ansi_codes(foreground=False) == (
            "48", "2", "18", "18", "18",
        )


class TestEightBitColors:
    """Textual downgrades TRUECOLOR to EIGHT_BIT before calling get_ansi_codes.
    The patch must match 8-bit colors whose palette RGB equals the target."""

    def test_eight_bit_matching_bg_returns_ansi_default(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        # 8-bit palette entry 233 = #121212 = (18, 18, 18)
        color = RichColor("color(233)", ColorType.EIGHT_BIT, number=233, triplet=None)
        assert color.get_ansi_codes(foreground=False) == ("49",)

    def test_eight_bit_non_matching_unchanged(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)
        # 8-bit palette entry 196 = bright red
        color = RichColor("color(196)", ColorType.EIGHT_BIT, number=196, triplet=None)
        result = color.get_ansi_codes(foreground=False)
        assert result == ("48", "5", "196")


class TestStripIntegration:
    def test_rendered_output_uses_ansi_default_bg_truecolor(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)

        style = Style(color="white", bgcolor=RichColor.parse("#121212"))
        strip = Strip([Segment("hello", style)])
        console = Console(color_system="truecolor")
        output = strip.render(console)

        assert ";49" in output
        assert "48;2;18;18;18" not in output

    def test_rendered_output_uses_ansi_default_bg_256color(self):
        """Textual typically uses 256-color system, not truecolor."""
        app = _make_app_mock("#121212")
        enable_transparent_background(app)

        style = Style(color="white", bgcolor=RichColor.parse("#121212"))
        strip = Strip([Segment("hello", style)])
        console = Console(color_system="256")
        output = strip.render(console)

        assert ";49" in output
        # 8-bit palette 233 = #121212; should NOT appear
        assert "48;5;233" not in output

    def test_non_matching_bg_renders_normally(self):
        app = _make_app_mock("#121212")
        enable_transparent_background(app)

        style = Style(color="white", bgcolor=RichColor.parse("#ff0000"))
        strip = Strip([Segment("hello", style)])
        console = Console(color_system="truecolor")
        output = strip.render(console)

        assert "48;2;255;0;0" in output
