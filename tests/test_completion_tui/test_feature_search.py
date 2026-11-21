"""Tests for the FeatureSearch widget.

Tests the feature search overlay: digit-based query filtering, result
selection, exclusion of already-present features, and message posting.
"""

from __future__ import annotations

from scripts.completion_tui.widgets.feature_search import (
    MAX_SEARCH_RESULTS,
    FeatureSearch,
)
from tests.test_completion_tui._key_dispatch import dispatch_key

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_available(ids: list[int]) -> dict[int, str]:
    """Create an available_features dict from a list of IDs."""
    return {fid: f"desc for {fid}" for fid in ids}


def _make_search(
    available_ids: list[int] | None = None,
    excluded_ids: set[int] | None = None,
) -> FeatureSearch:
    if available_ids is None:
        # Default: features 1..50 to give a good spread for digit search
        available_ids = list(range(1, 51))
    available = _make_available(available_ids)
    excluded = excluded_ids or set()
    return FeatureSearch(available_features=available, excluded_ids=excluded)


# ---------------------------------------------------------------------------
# 1. Typing "3" shows features starting with 3, sorted by ID
# ---------------------------------------------------------------------------


class TestDigitFilter:
    def test_typing_3_shows_features_starting_with_3(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        # Features whose str(id) starts with "3": 3, 30, 31, 32, ..., 39
        assert search.query == "3"
        assert all(str(fid).startswith("3") for fid in search.results)
        # Should be sorted ascending
        assert search.results == sorted(search.results)
        # Expect: 3, 30, 31, 32, 33, 34, 35, 36, 37, 38 (max 10)
        assert len(search.results) <= MAX_SEARCH_RESULTS


# ---------------------------------------------------------------------------
# 2. Typing "34" narrows results
# ---------------------------------------------------------------------------


class TestNarrowSearch:
    def test_typing_34_narrows_results(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        initial_count = len(search.results)
        search.handle_digit(4)
        assert search.query == "34"
        # Only feature 34 should match (starts with "34")
        assert search.results == [34]
        assert len(search.results) <= initial_count


# ---------------------------------------------------------------------------
# 3. Already-present features excluded (including disabled)
# ---------------------------------------------------------------------------


class TestExclusion:
    def test_existing_features_excluded(self) -> None:
        # Feature 3 is already in segment (enabled)
        search = _make_search(excluded_ids={3})
        search.handle_digit(3)
        assert 3 not in search.results
        # 30-39 should still appear
        assert 30 in search.results

    def test_disabled_features_also_excluded(self) -> None:
        # Feature 30 is in segment but disabled — still excluded
        search = _make_search(excluded_ids={30})
        search.handle_digit(3)
        assert 30 not in search.results


# ---------------------------------------------------------------------------
# 4. Max results capped at MAX_SEARCH_RESULTS
# ---------------------------------------------------------------------------


class TestMaxResults:
    def test_results_capped_at_max(self) -> None:
        # Create many features starting with "1": 1, 10-19, 100-199
        ids = [1] + list(range(10, 20)) + list(range(100, 200))
        search = _make_search(available_ids=ids)
        search.handle_digit(1)
        assert len(search.results) == MAX_SEARCH_RESULTS
        # Results should be sorted ascending (so first 10 matching)
        assert search.results == sorted(search.results)


# ---------------------------------------------------------------------------
# 5. Query update resets selection to first
# ---------------------------------------------------------------------------


class TestSelectionReset:
    def test_query_update_resets_selection(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        # Move selection down
        search.move_selection(1)
        search.move_selection(1)
        assert search.selected_idx == 2
        # Type another digit — selection should reset to 0
        search.handle_digit(4)
        assert search.selected_idx == 0


# ---------------------------------------------------------------------------
# 6. Down then type → selection resets
# ---------------------------------------------------------------------------


class TestDownThenType:
    def test_down_then_digit_resets_selection(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        # Navigate down to third result
        search.move_selection(1)
        search.move_selection(1)
        assert search.selected_idx == 2
        # Type another digit → selection resets to first
        search.handle_digit(0)
        assert search.selected_idx == 0
        assert search.query == "30"


# ---------------------------------------------------------------------------
# 7. Space/Enter adds feature at magnitude 0, posts FeatureAdded
# ---------------------------------------------------------------------------


class TestAddFeature:
    def test_space_adds_feature(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        selected = search.results[search.selected_idx]
        messages = dispatch_key(search, "space")
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, FeatureSearch.FeatureAdded)
        assert msg.feature_id == selected

    def test_enter_adds_feature(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        selected = search.results[search.selected_idx]
        messages = dispatch_key(search, "enter")
        assert len(messages) == 1
        msg = messages[0]
        assert isinstance(msg, FeatureSearch.FeatureAdded)
        assert msg.feature_id == selected

    def test_no_results_space_does_nothing(self) -> None:
        search = _make_search(available_ids=[1, 2])
        search.handle_digit(9)  # No features start with 9
        messages = dispatch_key(search, "space")
        assert len(messages) == 0


# ---------------------------------------------------------------------------
# 8. Esc closes without adding
# ---------------------------------------------------------------------------


class TestEscClose:
    def test_esc_posts_closed(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        messages = dispatch_key(search, "escape")
        assert len(messages) == 1
        assert isinstance(messages[0], FeatureSearch.FeatureSearchClosed)

    def test_esc_without_query_posts_closed(self) -> None:
        search = _make_search()
        messages = dispatch_key(search, "escape")
        assert len(messages) == 1
        assert isinstance(messages[0], FeatureSearch.FeatureSearchClosed)


# ---------------------------------------------------------------------------
# 9. Empty query shows no results
# ---------------------------------------------------------------------------


class TestEmptyQuery:
    def test_empty_query_no_results(self) -> None:
        search = _make_search()
        assert search.query == ""
        assert search.results == []

    def test_backspace_to_empty_clears_results(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        assert len(search.results) > 0
        search.handle_backspace()
        assert search.query == ""
        assert search.results == []


# ---------------------------------------------------------------------------
# 10. Backspace removes last query digit
# ---------------------------------------------------------------------------


class TestBackspace:
    def test_backspace_removes_last_digit(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        search.handle_digit(4)
        assert search.query == "34"
        search.handle_backspace()
        assert search.query == "3"
        # Results should match "3" again, not "34"
        assert len(search.results) > 1

    def test_backspace_on_empty_is_noop(self) -> None:
        search = _make_search()
        search.handle_backspace()
        assert search.query == ""
        assert search.results == []


# ---------------------------------------------------------------------------
# Navigation
# ---------------------------------------------------------------------------


class TestNavigation:
    def test_down_moves_selection(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        assert search.selected_idx == 0
        search.move_selection(1)
        assert search.selected_idx == 1

    def test_up_clamps_at_zero(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        search.move_selection(-1)
        assert search.selected_idx == 0

    def test_down_clamps_at_end(self) -> None:
        search = _make_search(available_ids=[31, 32])
        search.handle_digit(3)
        assert len(search.results) == 2
        search.move_selection(1)
        search.move_selection(1)  # past end
        assert search.selected_idx == 1


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------


class TestRendering:
    def test_render_shows_query(self) -> None:
        search = _make_search()
        search.handle_digit(3)
        output = search.render()
        assert "Search: 3" in output

    def test_render_shows_results_with_indicator(self) -> None:
        search = _make_search(available_ids=[31, 32])
        search.handle_digit(3)
        output = search.render()
        # First result should have indicator
        assert "▶" in output
        assert "31" in output
        assert "32" in output

    def test_render_no_matching_features(self) -> None:
        search = _make_search(available_ids=[1, 2])
        search.handle_digit(9)
        output = search.render()
        assert "No matching features" in output
