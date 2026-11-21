"""Feature detail overlay showing full Neuronpedia feature information.

Displays feature description, explanations, and max activating examples
fetched from Neuronpedia via SAEFeature.get().
"""

from __future__ import annotations

from typing import Any

from textual.events import Key
from textual.message import Message
from textual.widget import Widget


class FeatureDetail(Widget):
    """Detail overlay for viewing a single feature's Neuronpedia data.

    Shows loading state until data arrives, then renders the full
    explanation, activation examples, and statistics. Scrollable if
    content exceeds the visible area.
    """

    # ------------------------------------------------------------------
    # Messages
    # ------------------------------------------------------------------

    class FeatureDetailClosed(Message):
        """Posted when the user closes the detail overlay."""

    # ------------------------------------------------------------------
    # Constructor
    # ------------------------------------------------------------------

    def __init__(
        self,
        feature_id: int,
        description: str | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._feature_id = feature_id
        self._description = description
        self._loading: bool = True
        self._feature_data: dict[str, Any] | None = None
        self._highlight_status: str | None = None
        self._highlight_message: str | None = None
        self._scroll_offset: int = 0

    # ------------------------------------------------------------------
    # Public read-only properties
    # ------------------------------------------------------------------

    @property
    def feature_id(self) -> int:
        return self._feature_id

    # ------------------------------------------------------------------
    # Data management
    # ------------------------------------------------------------------

    def set_data(self, data: dict[str, Any]) -> None:
        """Set the fetched feature data, clearing loading state."""
        self._feature_data = data
        self._loading = False
        # Extract description from data if not already set
        explanations = data.get("explanations", [])
        if explanations and not self._description:
            self._description = explanations[0].get("description")

    def set_highlight_status(
        self,
        status: str,
        message: str | None = None,
    ) -> None:
        """Set live token-highlight status shown in the detail overlay."""
        self._highlight_status = status
        self._highlight_message = message
        self.refresh()

    def clear_highlight_status(self) -> None:
        """Remove the live token-highlight status line."""
        self._highlight_status = None
        self._highlight_message = None
        self.refresh()

    # ------------------------------------------------------------------
    # Scrolling
    # ------------------------------------------------------------------

    def move_scroll_down(self) -> None:
        """Scroll content down by one line, clamping at end of content."""
        max_offset = max(0, len(self._build_lines()) - 1)
        if self._scroll_offset < max_offset:
            self._scroll_offset += 1

    def move_scroll_up(self) -> None:
        """Scroll content up by one line, clamping at zero."""
        if self._scroll_offset > 0:
            self._scroll_offset -= 1

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------

    can_focus = True

    def _build_lines(self) -> list[str]:
        """Build the full content lines before scroll offset is applied."""
        lines: list[str] = []

        if self._loading:
            lines.append(f"Feature {self._feature_id}: Loading...")
            self._append_highlight_status(lines)
            return lines

        # Header
        desc = self._description or "(no description)"
        lines.append(f"Feature {self._feature_id}: {desc}")
        self._append_highlight_status(lines)
        lines.append("")

        if self._feature_data:
            # Explanations
            explanations = self._feature_data.get("explanations", [])
            if explanations:
                lines.append("Explanation:")
                for exp in explanations:
                    exp_desc = exp.get("description", "")
                    if exp_desc:
                        lines.append(f"  {exp_desc}")
                lines.append("")

            # Activations
            activations = self._feature_data.get("activations", [])
            if activations:
                lines.append("Max Activating Examples:")
                for i, act in enumerate(activations):
                    tokens = act.get("tokens", [])
                    values = act.get("values", [])
                    lines.append(f"  Example {i + 1}:")
                    token_str = "".join(tokens)
                    lines.append(f"    Text: {token_str}")
                    if values:
                        max_val = max(values)
                        lines.append(f"    Max activation: {max_val:.2f}")
                    lines.append("")

        return lines

    def _append_highlight_status(self, lines: list[str]) -> None:
        if self._highlight_status is None:
            return
        message = self._highlight_message or self._highlight_status
        lines.append(f"Live highlight: {message}")

    def render(self) -> str:
        """Render the detail overlay as a plain string."""
        lines = self._build_lines()
        # Apply scroll offset
        visible = lines[self._scroll_offset :]
        return "\n".join(visible)

    # ------------------------------------------------------------------
    # Key event dispatch
    # ------------------------------------------------------------------

    def on_key(self, event: Key) -> None:
        """Route key events to the appropriate handler."""
        key = event.key

        if key == "escape":
            self.post_message(self.FeatureDetailClosed())
        elif key == "up":
            self.move_scroll_up()
            self.refresh()
        elif key == "down":
            self.move_scroll_down()
            self.refresh()
        else:
            return

        event.prevent_default()
        event.stop()
