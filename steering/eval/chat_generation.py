"""Shared chat-completion rendering.

This module intentionally has no dependency on ``nanochat``.  It is the
project-owned seam for callers that need the RustBPE SFT chat template and
assistant-end stop-token validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ChatCompletionPrompt:
    """Rendered prompt and the assistant terminator used for generation."""

    token_ids: list[int]
    assistant_end_id: int


@dataclass(frozen=True)
class ChatGenerationResult:
    """Generated body plus conservative exhaustion evidence from the native engine."""

    text: str
    token_ids: tuple[int, ...]
    budget_exhausted: bool


def render_messages_for_chat_completion(
    tokenizer: Any, messages: Sequence[Mapping[str, str]],
) -> ChatCompletionPrompt:
    """Render a complete text history without the convenience renderer's length cap.

    Histories end in a user turn; no source assistant answer is removed or
    invented. A defensive copy preserves the caller's immutable source bytes.
    Context admission belongs to the target and must reject, never truncate.
    """
    if not messages:
        raise ValueError("chat history must contain at least one user turn")
    copied = []
    expected = "user"
    for index, message in enumerate(messages):
        if set(message) != {"role", "content"}:
            raise ValueError("text chat messages require exactly role and content")
        role, content = message["role"], message["content"]
        if not isinstance(content, str) or not content.strip():
            raise ValueError("chat message content must be nonblank text")
        if index == 0 and role == "system":
            copied.append(dict(message))
            continue
        if role != expected:
            raise ValueError("chat history must alternate user and assistant roles")
        expected = "assistant" if role == "user" else "user"
        copied.append(dict(message))
    if copied[-1]["role"] != "user":
        raise ValueError("chat completion history must end in a user turn")
    token_ids, _ = tokenizer.render_conversation({"messages": copied}, max_tokens=None)
    assistant_start_id = tokenizer.encode_special("<assistant_start>")
    assistant_end_id = tokenizer.encode_special("<assistant_end>")
    if type(assistant_start_id) is not int or type(assistant_end_id) is not int:
        raise ValueError("chat assistant boundaries must resolve to integer token IDs")
    return ChatCompletionPrompt([*token_ids, assistant_start_id], assistant_end_id)


def empty_assistant_conversation(prompt: str) -> dict[str, list[dict[str, str]]]:
    """Build the canonical one-user-turn, empty-assistant conversation.

    Returning a new nested object on every call is important: tokenizers are
    allowed to perform defensive conversation surgery, and callers must not
    share mutable message state between requests or samples.
    """
    if not isinstance(prompt, str):
        raise TypeError(f"chat completion prompt must be str, got {type(prompt).__name__}")
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": ""},
        ]
    }


def require_chat_render_support(tokenizer: Any) -> None:
    """Fail fast with the established error for non-chat tokenizers."""
    if not (
        hasattr(tokenizer, "render_for_completion")
        and hasattr(tokenizer, "encode_special")
    ):
        raise ValueError(
            "render_as_chat=True requires a tokenizer with "
            "render_for_completion and encode_special "
            f"(got {type(tokenizer).__name__}). Use a RustBPE-trained "
            "nanochat model, or drop render_as_chat for raw-prompt generation."
        )


def render_chat_completion(
    tokenizer: Any,
    conversation: object,
) -> ChatCompletionPrompt:
    """Render a conversation for assistant completion and resolve its stop ID.

    ``conversation`` may be a task conversation containing a populated final
    assistant answer (``render_for_completion`` removes it on its defensive
    copy) or the canonical empty-assistant conversation returned by
    :func:`empty_assistant_conversation`.  The tokenizer's rendered token
    sequence is returned unchanged apart from materializing it as a list.

    The exception classes and message/chaining behavior for invalid
    ``<assistant_end>`` resolution intentionally match the former
    ``InferenceEngine`` implementation.
    """
    require_chat_render_support(tokenizer)
    token_ids = list(tokenizer.render_for_completion(conversation))
    try:
        assistant_end_id = tokenizer.encode_special("<assistant_end>")
    except (ValueError, RuntimeError, TypeError, AttributeError, KeyError) as exc:
        raise ValueError(
            "render_as_chat=True requires tokenizer.encode_special("
            "'<assistant_end>') to return a valid token id "
            f"(got {type(tokenizer).__name__}: {type(exc).__name__}: {exc}). "
            "Use a tokenizer whose vocabulary contains the <assistant_end> "
            "special token, or drop render_as_chat."
        ) from exc
    if isinstance(assistant_end_id, bool) or not isinstance(assistant_end_id, int):
        raise ValueError(
            "render_as_chat=True requires tokenizer.encode_special("
            "'<assistant_end>') to return an int token id, got "
            f"{assistant_end_id!r} from {type(tokenizer).__name__}. "
            "A None return silently disables the assistant-end stop "
            "token and lets generation run to max_new_tokens."
        )
    return ChatCompletionPrompt(
        token_ids=token_ids,
        assistant_end_id=assistant_end_id,
    )


def render_prompt_for_chat_completion(
    tokenizer: Any,
    prompt: str,
) -> ChatCompletionPrompt:
    """Render a plain user prompt with an empty assistant turn."""
    return render_chat_completion(
        tokenizer,
        empty_assistant_conversation(prompt),
    )
