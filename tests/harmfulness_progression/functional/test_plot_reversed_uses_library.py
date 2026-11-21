"""Source-inspection: the reverse-order plot script uses library constants."""

from __future__ import annotations

import inspect

from scripts.eval import plot_harmfulness_reversed_progression as plot_module


def test_script_imports_reversed_source_maps() -> None:
    """Source maps must be imported from the library, not defined inline.

    This guards against drift: the library has DEFAULT_REVERSED_* constants
    If the plot script defines its own copies, it bypasses the shared source
    mapping used by the renderer.
    """
    src = inspect.getsource(plot_module)
    assert "DEFAULT_REVERSED_COND_SOURCE_MAP" in src
    assert "DEFAULT_REVERSED_STAGES" in src


def test_script_calls_progression_renderer() -> None:
    """The output uses the shared progression renderer."""
    src = inspect.getsource(plot_module)
    assert "render_progression_chart" in src


def test_script_does_not_define_inline_reversed_constants() -> None:
    """Defensive: catch a future regression where someone re-adds the inline
    REVERSED_* constants instead of importing them from the library."""
    src = inspect.getsource(plot_module)
    # The inline definitions had these top-level assignments. Library import
    # uses DEFAULT_REVERSED_*, so plain "REVERSED_" should not appear at the
    # start of an assignment line in the script.
    for line in src.splitlines():
        stripped = line.lstrip()
        if stripped.startswith("REVERSED_"):
            raise AssertionError(
                f"Inline REVERSED_ constant detected: {line!r}. "
                "Use DEFAULT_REVERSED_* imported from the library instead."
            )
