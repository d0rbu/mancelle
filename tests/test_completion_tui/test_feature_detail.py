"""Tests for the FeatureDetail widget.

Tests the feature detail overlay: description display, loading state,
scroll behavior, and message posting.
"""

from __future__ import annotations

from scripts.completion_tui.widgets.feature_detail import FeatureDetail
from tests.test_completion_tui._key_dispatch import dispatch_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_SAMPLE_DATA = {
    "explanations": [{"description": "Activates on references to scientific concepts"}],
    "activations": [
        {
            "tokens": ["The", " theory", " of", " relativity"],
            "values": [0.1, 0.8, 0.2, 0.9],
        },
        {
            "tokens": ["quantum", " mechanics", " predicts"],
            "values": [0.7, 0.6, 0.3],
        },
    ],
}


def _make_detail(
    feature_id: int = 42,
    description: str | None = "Scientific concept detector",
) -> FeatureDetail:
    return FeatureDetail(feature_id=feature_id, description=description)


# ---------------------------------------------------------------------------
# 1. Shows feature description
# ---------------------------------------------------------------------------


class TestDescription:
    def test_shows_description_after_data_loaded(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        output = detail.render()
        assert "Feature 42" in output
        assert "Scientific concept detector" in output

    def test_shows_explanation_from_data(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        output = detail.render()
        assert "Activates on references to scientific concepts" in output

    def test_shows_activations(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        output = detail.render()
        # Should show activation examples
        assert "relativity" in output
        assert "quantum" in output


# ---------------------------------------------------------------------------
# 2. Shows loading state before fetch completes
# ---------------------------------------------------------------------------


class TestLoadingState:
    def test_loading_by_default(self) -> None:
        detail = _make_detail()
        output = detail.render()
        assert "Loading" in output
        assert "42" in output

    def test_loading_cleared_after_data(self) -> None:
        detail = _make_detail()
        assert "Loading" in detail.render()
        detail.set_data(_SAMPLE_DATA)
        output = detail.render()
        assert "Loading" not in output


# ---------------------------------------------------------------------------
# Live highlight status
# ---------------------------------------------------------------------------


class TestLiveHighlightStatus:
    def test_loading_status_renders_while_feature_data_loads(self) -> None:
        detail = _make_detail()
        detail.set_highlight_status("loading", "computing token activations")

        output = detail.render()

        assert "Loading" in output
        assert "Live highlight: computing token activations" in output

    def test_ready_status_renders_after_data_loaded(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        detail.set_highlight_status("ready", "4 tokens, max p99.5")

        output = detail.render()

        assert "Live highlight: 4 tokens, max p99.5" in output

    def test_clear_status_removes_live_highlight_line(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        detail.set_highlight_status("ready", "4 tokens")
        detail.clear_highlight_status()

        output = detail.render()

        assert "Live highlight" not in output


# ---------------------------------------------------------------------------
# 3. Esc posts FeatureDetailClosed
# ---------------------------------------------------------------------------


class TestEscClose:
    def test_esc_posts_closed(self) -> None:
        detail = _make_detail()
        messages = dispatch_key(detail, "escape")
        assert len(messages) == 1
        assert isinstance(messages[0], FeatureDetail.FeatureDetailClosed)

    def test_other_keys_no_close_message(self) -> None:
        detail = _make_detail()
        messages = dispatch_key(detail, "enter")
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# 4. Content is scrollable
# ---------------------------------------------------------------------------


class TestScrolling:
    def test_scroll_down(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        assert detail._scroll_offset == 0
        detail.move_scroll_down()
        assert detail._scroll_offset == 1
        detail.move_scroll_down()
        assert detail._scroll_offset == 2

    def test_scroll_up(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        detail.move_scroll_down()
        detail.move_scroll_down()
        detail.move_scroll_up()
        assert detail._scroll_offset == 1

    def test_scroll_up_clamps_at_zero(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        detail.move_scroll_up()
        assert detail._scroll_offset == 0

    def test_scroll_affects_render(self) -> None:
        detail = _make_detail()
        detail.set_data(_SAMPLE_DATA)
        output_no_scroll = detail.render()
        detail.move_scroll_down()
        detail.move_scroll_down()
        detail.move_scroll_down()
        output_scrolled = detail.render()
        # Scrolled output should differ (first lines are hidden)
        assert output_scrolled != output_no_scroll


# ---------------------------------------------------------------------------
# Description fallback from data
# ---------------------------------------------------------------------------


class TestDescriptionFallback:
    def test_description_extracted_from_data_when_not_provided(self) -> None:
        detail = FeatureDetail(feature_id=42, description=None)
        detail.set_data(_SAMPLE_DATA)
        output = detail.render()
        assert "Activates on references to scientific concepts" in output
