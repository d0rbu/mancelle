"""Tests for scripts/completion_tui/color_manager.py."""

from __future__ import annotations

from scripts.completion_tui.color_manager import (
    DEFAULT_COLOR,
    SEGMENT_COLORS,
    blend_hex_colors,
    color_for_idx,
    feature_highlight_color,
    next_color,
)

# ---------------------------------------------------------------------------
# next_color
# ---------------------------------------------------------------------------


class TestNextColor:
    def test_picks_unused_color(self) -> None:
        existing = [0, 1, 2]
        result = next_color(existing)
        assert result not in existing

    def test_picks_lowest_unused(self) -> None:
        # 0 and 1 are used, so lowest unused is 2
        assert next_color([0, 1]) == 2

    def test_avoids_left_neighbor(self) -> None:
        result = next_color([], left_neighbor_idx=0)
        assert result != 0

    def test_avoids_right_neighbor(self) -> None:
        result = next_color([], right_neighbor_idx=1)
        assert result != 1

    def test_avoids_both_neighbors(self) -> None:
        result = next_color([], left_neighbor_idx=0, right_neighbor_idx=1)
        assert result != 0
        assert result != 1

    def test_avoids_neighbors_and_existing(self) -> None:
        # Everything except index 5 is excluded
        n = len(SEGMENT_COLORS)
        existing = list(range(n))
        existing.remove(5)
        result = next_color(existing, left_neighbor_idx=5)
        # 5 is available but is the left neighbor, so it's avoided
        # All others are in existing. Falls back to any unused color.
        # Only 5 is unused, so it should return 5 (fallback)
        assert result == 5

    def test_all_used_cycles(self) -> None:
        # All palette colors are in use; should still return something
        existing = list(range(len(SEGMENT_COLORS)))
        result = next_color(existing)
        assert 0 <= result < len(SEGMENT_COLORS)

    def test_empty_existing_returns_zero(self) -> None:
        result = next_color([])
        assert result == 0

    def test_empty_existing_with_zero_as_neighbor(self) -> None:
        result = next_color([], left_neighbor_idx=0)
        assert result == 1  # 0 is neighbor, so pick 1


# ---------------------------------------------------------------------------
# color_for_idx
# ---------------------------------------------------------------------------


class TestColorForIdx:
    def test_returns_hex_string(self) -> None:
        color = color_for_idx(0)
        assert color.startswith("#")
        assert len(color) == 7

    def test_each_index_returns_palette_color(self) -> None:
        for i, expected in enumerate(SEGMENT_COLORS):
            assert color_for_idx(i) == expected

    def test_wraps_around(self) -> None:
        n = len(SEGMENT_COLORS)
        assert color_for_idx(n) == SEGMENT_COLORS[0]
        assert color_for_idx(n + 1) == SEGMENT_COLORS[1]

    def test_default_color_is_valid(self) -> None:
        assert DEFAULT_COLOR.startswith("#")
        assert len(DEFAULT_COLOR) == 7


# ---------------------------------------------------------------------------
# feature highlight palette
# ---------------------------------------------------------------------------


class TestFeatureHighlightColors:
    def test_blend_hex_colors_interpolates_channels(self) -> None:
        assert blend_hex_colors("#000000", "#ffffff", 0.5) == "#808080"

    def test_feature_highlight_level_zero_is_default_gray(self) -> None:
        assert feature_highlight_color("#ffffff", 0) == DEFAULT_COLOR

    def test_feature_highlight_level_three_is_segment_color(self) -> None:
        segment_color = color_for_idx(2)
        assert feature_highlight_color(segment_color, 3) == segment_color

    def test_feature_highlight_levels_one_and_two_are_between_gray_and_segment(self) -> None:
        segment_color = color_for_idx(0)
        level_1 = feature_highlight_color(segment_color, 1)
        level_2 = feature_highlight_color(segment_color, 2)

        assert level_1 not in {DEFAULT_COLOR, segment_color}
        assert level_2 not in {DEFAULT_COLOR, segment_color}
        assert level_1 != level_2

    def test_feature_highlight_positive_level_is_visible_for_all_palette_colors(self) -> None:
        def rgb(color: str) -> tuple[int, int, int]:
            red, green, blue = (int(color[i : i + 2], 16) for i in (1, 3, 5))
            return red, green, blue

        default_rgb = rgb(DEFAULT_COLOR)
        for segment_color in SEGMENT_COLORS:
            level_1_rgb = rgb(feature_highlight_color(segment_color, 1))
            channel_distance = sum(
                abs(a - b) for a, b in zip(default_rgb, level_1_rgb, strict=True)
            )
            assert channel_distance >= 50
