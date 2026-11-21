"""Prompt rendering for token-level intruder-detection examples."""

from __future__ import annotations

from itertools import groupby

from beartype import beartype

from steering.eval.autointerp_filter.types import (
    IntruderPromptError,
    RenderedExample,
    TokenWindowRef,
)
from steering.refined_types import NonEmptyNonNegativeIntTupleLike


def escape_prompt_markers(text: str) -> str:
    """Prevent source text from introducing fake highlight markers."""
    return text.replace("<<", "< <").replace(">>", "> >")


class TokenWindowRenderer:
    """Render token windows with explicit token-index highlights."""

    @beartype
    def render(
        self,
        ref: TokenWindowRef,
        *,
        activating: bool,
        highlighted_token_indices: NonEmptyNonNegativeIntTupleLike,
    ) -> RenderedExample:
        n_tokens = len(ref.token_strings)
        bad = [idx for idx in highlighted_token_indices if idx >= n_tokens]
        if bad:
            raise IntruderPromptError(
                f"token window {ref.stable_key()} highlight indices out of "
                f"range for {n_tokens} tokens: {bad}"
            )
        highlighted = set(highlighted_token_indices)
        parts: list[str] = []
        for should_highlight, token_group in groupby(
            enumerate(ref.token_strings),
            key=lambda item: item[0] in highlighted,
        ):
            tokens = [escape_prompt_markers(token) for _idx, token in token_group]
            if should_highlight:
                parts.append("<<")
                parts.extend(tokens)
                parts.append(">>")
            else:
                parts.extend(tokens)
        return RenderedExample(
            text="".join(parts),
            ref=ref,
            activating=activating,
            highlighted_token_indices=tuple(sorted(highlighted_token_indices)),
        )
