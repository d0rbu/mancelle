import math
import os
from itertools import pairwise
from pathlib import Path

import pytest

from nanochat.tokenizer import get_tokenizer
from scripts.chat_tui.session import (
    ChatSessionConfig,
    ConversationText,
    TextDelta,
    TurnState,
    segment_char_spans,
)


def _shared_nanochat_base_dir() -> Path:
    for parent in Path(__file__).resolve().parents:
        candidate = parent / "nanochat_cache"
        if (candidate / "tokenizer" / "tokenizer.pkl").is_file():
            return candidate
    return Path(__file__).resolve().parents[2] / "nanochat_cache"


@pytest.fixture(scope="module")
def tokenizer():
    previous = os.environ.get("NANOCHAT_BASE_DIR")
    os.environ["NANOCHAT_BASE_DIR"] = str(_shared_nanochat_base_dir())
    try:
        yield get_tokenizer()
    finally:
        if previous is None:
            os.environ.pop("NANOCHAT_BASE_DIR", None)
        else:
            os.environ["NANOCHAT_BASE_DIR"] = previous


def _messages(user: str = "Question", assistant: str = "Answer") -> list[dict[str, str]]:
    return [
        {"role": "user", "content": user},
        {"role": "assistant", "content": assistant},
    ]


def test_config_validates_positive_ints_and_finite_floats() -> None:
    valid = ChatSessionConfig(
        self_cond_segment_size=256,
        temperature=0.6,
        top_k=50,
        seed=42,
        max_new_tokens_per_turn=256,
        deterministic_rewind=False,
    )
    assert valid.self_cond_segment_size == 256

    replacements = [
        ("self_cond_segment_size", 0),
        ("top_k", 0),
        ("seed", 0),
        ("max_new_tokens_per_turn", -1),
        ("temperature", math.inf),
    ]
    for field, value in replacements:
        kwargs = {
            "self_cond_segment_size": 256,
            "temperature": 0.6,
            "top_k": 50,
            "seed": 42,
            "max_new_tokens_per_turn": 256,
            "deterministic_rewind": False,
        }
        kwargs[field] = value
        with pytest.raises(ValueError, match=field):
            ChatSessionConfig(**kwargs)


def test_load_prefix_enforces_strict_2048_render_guard(tokenizer) -> None:
    allowed_messages = _messages(user="x " * 2041, assistant="")
    refused_messages = _messages(user="x " * 2042, assistant="")
    allowed_ids, _ = tokenizer.render_conversation({"messages": allowed_messages}, max_tokens=2**31)
    refused_ids, _ = tokenizer.render_conversation({"messages": refused_messages}, max_tokens=2**31)
    assert len(allowed_ids) == 2047
    assert len(refused_ids) == 2048

    conversation = ConversationText(tokenizer, sequence_len=4096)
    conversation.load_prefix(allowed_messages)
    assert conversation.tokens == tuple(allowed_ids)
    with pytest.raises(ValueError, match="2048"):
        conversation.load_prefix(refused_messages)
    assert conversation.tokens == tuple(allowed_ids)


def test_empty_prefix_starts_at_bos_between_turns(tokenizer) -> None:
    conversation = ConversationText(tokenizer, sequence_len=64)
    conversation.load_prefix([])

    assert conversation.tokens == (tokenizer.get_bos_token_id(),)
    assert conversation.text == tokenizer.decode(list(conversation.tokens))
    assert conversation.turn_state() is TurnState.BETWEEN


def test_turn_state_machine_transitions_after_each_marker(tokenizer) -> None:
    conversation = ConversationText(tokenizer, sequence_len=64)
    bos = tokenizer.get_bos_token_id()
    user_start = tokenizer.encode_special("<user_start>")
    user_end = tokenizer.encode_special("<user_end>")
    assistant_start = tokenizer.encode_special("<assistant_start>")
    assistant_end = tokenizer.encode_special("<assistant_end>")

    assert conversation.tokens == (bos,)
    assert conversation.turn_state() is TurnState.BETWEEN
    conversation.append_tokens([user_start])
    assert conversation.tokens == (bos, user_start)
    assert conversation.turn_state() is TurnState.IN_USER
    conversation.append_tokens([user_end])
    assert conversation.tokens == (bos, user_start, user_end)
    assert conversation.turn_state() is TurnState.BETWEEN
    conversation.append_tokens([assistant_start])
    assert conversation.tokens == (bos, user_start, user_end, assistant_start)
    assert conversation.turn_state() is TurnState.IN_ASSISTANT
    conversation.append_tokens([assistant_end])
    assert conversation.tokens == (
        bos,
        user_start,
        user_end,
        assistant_start,
        assistant_end,
    )
    assert conversation.turn_state() is TurnState.BETWEEN


def test_turn_state_machine_on_rendered_two_turn_conversation(tokenizer) -> None:
    messages = [
        *_messages("First question", "First answer"),
        *_messages("Second question", "Second answer"),
    ]
    conversation = ConversationText(tokenizer, sequence_len=128)
    conversation.load_prefix(messages)
    rendered_ids, _ = tokenizer.render_conversation({"messages": messages}, max_tokens=2**31)

    assert conversation.tokens == tuple(rendered_ids)
    assert conversation.turn_state() is TurnState.BETWEEN
    user_start = tokenizer.encode_special("<user_start>")
    conversation.append_tokens([user_start])
    assert conversation.tokens == (*rendered_ids, user_start)
    assert conversation.turn_state() is TurnState.IN_USER


def test_forced_turn_tokens_wrap_or_resume_user_turn(tokenizer) -> None:
    text = "Typed café 🙂"
    encoded = tokenizer.encode(text)
    user_start = tokenizer.encode_special("<user_start>")
    user_end = tokenizer.encode_special("<user_end>")
    assistant_start = tokenizer.encode_special("<assistant_start>")

    between = ConversationText(tokenizer, sequence_len=64)
    forced = between.forced_turn_tokens(text)
    assert forced == [user_start, *encoded, user_end, assistant_start]
    assert tokenizer.decode(forced[1:-2]) == text
    assert between.tokens == (tokenizer.get_bos_token_id(),)

    in_user = ConversationText(tokenizer, sequence_len=64)
    in_user.append_tokens([user_start])
    resumed = in_user.forced_turn_tokens(text)
    assert in_user.turn_state() is TurnState.IN_USER
    assert resumed == [*encoded, user_end, assistant_start]
    assert tokenizer.decode(resumed[:-2]) == text
    assert in_user.tokens == (tokenizer.get_bos_token_id(), user_start)

    in_assistant = ConversationText(tokenizer, sequence_len=64)
    in_assistant.append_tokens([assistant_start])
    assert in_assistant.turn_state() is TurnState.IN_ASSISTANT
    with pytest.raises(ValueError, match="assistant turn is still open"):
        in_assistant.forced_turn_tokens(text)
    assert in_assistant.tokens == (tokenizer.get_bos_token_id(), assistant_start)


def test_atomic_user_turn_preflight_reserves_sample_and_close(tokenizer) -> None:
    conversation = ConversationText(tokenizer, sequence_len=64)
    original_tokens = conversation.tokens

    conversation.preflight_user_turn([7] * 61)
    assert conversation.tokens == original_tokens
    with pytest.raises(ValueError, match=r"65.*64"):
        conversation.preflight_user_turn([7] * 62)
    assert conversation.tokens == original_tokens


def test_sampling_cap_reserves_assistant_close_slot(tokenizer) -> None:
    roomy = ConversationText(tokenizer, sequence_len=64)
    assert roomy.sampling_cap(10) == 10
    assert roomy.sampling_cap(-1) == 0

    tight = ConversationText(tokenizer, sequence_len=5)
    assert len(tight.tokens) == 1
    assert tight.sampling_cap(10) == 3

    full = ConversationText(tokenizer, sequence_len=1)
    assert len(full.tokens) == 1
    assert full.sampling_cap(10) == 0


def test_progressive_decode_reports_utf8_boundary_rewrite(tokenizer) -> None:
    conversation = ConversationText(tokenizer, sequence_len=64)
    emoji_ids = tokenizer.encode("✅")
    assert len(emoji_ids) == 2
    assert all("\ufffd" in tokenizer.decode([token_id]) for token_id in emoji_ids)

    first = conversation.append_tokens(emoji_ids[:1])
    assert first == TextDelta(appended=tokenizer.decode(emoji_ids[:1]), rewritten_from=None)
    assert "\ufffd" in first.appended
    assert conversation.tokens == (tokenizer.get_bos_token_id(), emoji_ids[0])

    second = conversation.append_tokens(emoji_ids[1:])
    assert second.rewritten_from is not None
    assert second.appended == "✅"
    assert conversation.tokens == (tokenizer.get_bos_token_id(), *emoji_ids)
    assert conversation.text == tokenizer.decode(list(conversation.tokens))
    assert "\ufffd" not in conversation.text


def test_segment_char_spans_tile_progressively_decoded_text(tokenizer) -> None:
    messages = [
        *_messages("First 🙂 question", "First answer"),
        *_messages("Second question", "Second café answer"),
    ]
    conversation = ConversationText(tokenizer, sequence_len=128)
    conversation.load_prefix(messages)
    offsets = conversation.char_offsets()

    # token_to_char_offsets returns one boundary per token plus the final text boundary.
    assert len(offsets) == len(conversation.tokens) + 1
    assert offsets[0] == 0
    assert offsets[-1] == len(conversation.text)

    windows = []
    token_start = 0
    while token_start < len(conversation.tokens):
        token_end = min(token_start + 3, len(conversation.tokens))
        windows.append(segment_char_spans(conversation.tokens, offsets, token_start, token_end))
        token_start = token_end

    assert windows[0][0] == 0
    assert windows[-1][1] == len(conversation.text)
    assert all(left[1] == right[0] for left, right in pairwise(windows))
    assert "".join(conversation.text[start:end] for start, end in windows) == conversation.text


def test_can_commit_accepts_exact_boundary_only(tokenizer) -> None:
    conversation = ConversationText(tokenizer, sequence_len=64)
    assert len(conversation.tokens) == 1
    assert not conversation.can_commit(-1)
    assert conversation.can_commit(63)
    assert not conversation.can_commit(64)
