"""Color palette and assignment for completion TUI segments.

Each segment gets a color from a fixed palette. The assignment logic
avoids neighbor collisions when possible.
"""

from __future__ import annotations

SEGMENT_COLORS = [
    "#e06c75",  # red
    "#98c379",  # green
    "#61afef",  # blue
    "#e5c07b",  # yellow
    "#c678dd",  # purple
    "#56b6c2",  # cyan
    "#d19a66",  # orange
    "#be5046",  # dark red
]

DEFAULT_COLOR = "#abb2bf"  # default/uncolored text

_PALETTE_SIZE = len(SEGMENT_COLORS)
_HIGHLIGHT_LEVEL_RATIOS = {
    0: 0.0,
    1: 0.55,
    2: 0.80,
    3: 1.0,
}


def next_color(
    existing_color_indices: list[int],
    left_neighbor_idx: int | None = None,
    right_neighbor_idx: int | None = None,
) -> int:
    """Pick the lowest unused color that also differs from neighbors.

    Falls back to any unused color if all non-neighbor colors are taken,
    then cycles forward from the left neighbor (or last existing color)
    through the full palette if every color is in use.
    """
    used = set(existing_color_indices)
    neighbors = {
        idx for idx in (left_neighbor_idx, right_neighbor_idx) if idx is not None
    }

    # First pass: lowest index not in use AND not a neighbor
    for i in range(_PALETTE_SIZE):
        if i not in used and i not in neighbors:
            return i

    # Second pass: any unused color (even if it matches a neighbor)
    for i in range(_PALETTE_SIZE):
        if i not in used:
            return i

    # All colors in use — continue around the palette, skipping neighbors.
    previous = left_neighbor_idx if left_neighbor_idx is not None else existing_color_indices[-1]
    for offset in range(1, _PALETTE_SIZE + 1):
        candidate = (previous + offset) % _PALETTE_SIZE
        if candidate not in neighbors:
            return candidate

    # Absolute fallback (only happens if palette_size <= 2 and both are neighbors)
    return 0


def color_for_idx(color_idx: int) -> str:
    """Return the hex color string for a palette index (wraps around)."""
    return SEGMENT_COLORS[color_idx % _PALETTE_SIZE]


def _parse_hex_color(color: str) -> tuple[int, int, int]:
    clean = color.strip()
    if len(clean) != 7 or not clean.startswith("#"):
        raise ValueError(f"Expected #rrggbb color, got {color!r}")
    return (
        int(clean[1:3], 16),
        int(clean[3:5], 16),
        int(clean[5:7], 16),
    )


def blend_hex_colors(start: str, end: str, ratio: float) -> str:
    """Linearly interpolate two ``#rrggbb`` colors."""
    clamped = max(0.0, min(1.0, ratio))
    start_rgb = _parse_hex_color(start)
    end_rgb = _parse_hex_color(end)
    blended = tuple(
        round(a + (b - a) * clamped) for a, b in zip(start_rgb, end_rgb, strict=True)
    )
    return f"#{blended[0]:02x}{blended[1]:02x}{blended[2]:02x}"


def feature_highlight_color(segment_color: str, level: int) -> str:
    """Return the display color for a live activation highlight level.

    Level 0 is gray. Levels 1 and 2 interpolate from gray to the segment's
    normal text color, biased far enough toward the segment color to stay
    visible on terminals where subtle truecolor differences are compressed.
    Level 3 uses the segment's normal text color exactly.
    """
    ratio = _HIGHLIGHT_LEVEL_RATIOS.get(max(0, min(3, level)), 0.0)
    return blend_hex_colors(DEFAULT_COLOR, segment_color, ratio)
