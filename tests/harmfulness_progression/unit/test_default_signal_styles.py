"""DEFAULT_SIGNAL_STYLES must cover every signal in DEFAULT_SIGNAL_ORDER, and
its color groupings must match the documented intent (safety = cool,
capability = warm)."""

from __future__ import annotations

from steering.eval.harmfulness_progression import (
    DEFAULT_SIGNAL_ORDER,
    DEFAULT_SIGNAL_STYLES,
)


def test_every_default_signal_has_style() -> None:
    missing = [s for s in DEFAULT_SIGNAL_ORDER if s not in DEFAULT_SIGNAL_STYLES]
    assert not missing, f"signals without default style: {missing}"


def _hex_to_rgb(hex_str: str) -> tuple[int, int, int]:
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def test_safety_signals_use_cool_colors() -> None:
    """Safety signals: blues. Cool color := blue dominates over red."""
    for sig in DEFAULT_SIGNAL_ORDER:
        if not sig.startswith("Safety"):
            continue
        r, g, b = _hex_to_rgb(DEFAULT_SIGNAL_STYLES[sig].color)
        assert b >= r, (
            f"Safety signal {sig!r} expected cool color, got "
            f"{DEFAULT_SIGNAL_STYLES[sig].color} (r={r}, b={b})"
        )


def test_capability_signals_use_warm_colors() -> None:
    """Capability signals: reds. Warm color := red dominates over blue."""
    for sig in DEFAULT_SIGNAL_ORDER:
        if not sig.startswith("Capability"):
            continue
        r, g, b = _hex_to_rgb(DEFAULT_SIGNAL_STYLES[sig].color)
        assert r >= b, (
            f"Capability signal {sig!r} expected warm color, got "
            f"{DEFAULT_SIGNAL_STYLES[sig].color} (r={r}, b={b})"
        )
