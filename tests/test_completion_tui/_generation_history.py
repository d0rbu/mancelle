"""Test-owned setup for completion-TUI generation-history scenarios."""

from __future__ import annotations

from scripts.completion_tui.state import GenerationSnapshot, SessionState


def push_generation(
    state: SessionState,
    generated_text: str,
    generated_tokens: list[int],
) -> None:
    """Construct the post-generation state needed by undo/redo tests."""
    snapshot = GenerationSnapshot(
        document_text=state.document_text,
        segments=state.segments,
        generated_text=generated_text,
        generated_token_ids=generated_tokens,
        seed=state.seed,
    )

    target_idx = next(
        (index for index, segment in enumerate(state.segments) if segment.is_generation_target),
        None,
    )
    if target_idx is None:
        raise ValueError("No generation target segment found")
    target = state.segments[target_idx]
    insert_pos = target.char_end
    state.document_text = (
        state.document_text[:insert_pos]
        + generated_text
        + state.document_text[insert_pos:]
    )
    target.char_end += len(generated_text)
    for segment in state.segments[target_idx + 1 :]:
        segment.char_start += len(generated_text)
        segment.char_end += len(generated_text)

    state.generation_history.append(snapshot)
    state.generation_redo_stack.clear()
