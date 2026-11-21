"""Exact-mapping tests for the pretty transcript renderer."""

from __future__ import annotations

import pytest

from scripts.chat_tui.render import (
    ASSISTANT_MARKER,
    USER_MARKER,
    build_special_display,
    pretty_view,
    stream_display_text,
)
from scripts.chat_tui.session import SegmentKind, TranscriptSpan
from steering.segmentation import token_to_char_offsets
from tests.conftest import MockTokenizer
from tests.test_chat_tui.test_session_tokens import _shared_nanochat_base_dir


def _span(char_start: int, char_end: int, color_idx: int, kind: SegmentKind, is_live: bool):
    return TranscriptSpan(char_start, char_end, color_idx, kind, is_live)


def _mock_conversation(tokenizer: MockTokenizer) -> tuple[tuple[int, ...], list[int]]:
    """BOS + one full user turn + one full assistant turn, with char offsets."""

    tokens = (
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<user_start>"),
        *tokenizer.encode("hi"),
        tokenizer.encode_special("<user_end>"),
        tokenizer.encode_special("<assistant_start>"),
        *tokenizer.encode("yo"),
        tokenizer.encode_special("<assistant_end>"),
    )
    return tokens, token_to_char_offsets(tokenizer, list(tokens))


def test_single_turn_text_replaces_markers_and_omits_bos_and_ends() -> None:
    tokenizer = MockTokenizer()
    tokens, _offsets = _mock_conversation(tokenizer)
    special = build_special_display(tokenizer)

    text, spans = pretty_view(tokenizer, tokens, (), special)

    assert text == f"{USER_MARKER}hi\n\n{ASSISTANT_MARKER}yo"
    assert spans == ()


def test_spans_cover_markers_of_the_tokens_they_replace() -> None:
    """The prefix span must end exactly where S0 begins, and S0's mapped range
    must cover the assistant marker standing in for <assistant_start>."""

    tokenizer = MockTokenizer()
    tokens, offsets = _mock_conversation(tokenizer)
    special = build_special_display(tokenizer)
    # The session's prefix ledger tiles [0, prefill_len - 1); S0 starts at the
    # <assistant_start> token (prefill_len - 1) and runs to the end.
    assistant_start_index = tokens.index(tokenizer.encode_special("<assistant_start>"))
    raw_spans = (
        _span(0, offsets[assistant_start_index], 2, SegmentKind.PREFIX_REAL, False),
        _span(offsets[assistant_start_index], offsets[-1], 4, SegmentKind.SELFCOND, True),
    )

    text, spans = pretty_view(tokenizer, tokens, raw_spans, special)

    user_part = f"{USER_MARKER}hi"
    assistant_part = f"\n\n{ASSISTANT_MARKER}yo"
    assert text == user_part + assistant_part
    assert spans[0].char_start == 0
    assert spans[0].char_end == len(user_part)
    assert spans[1].char_start == len(user_part)
    assert spans[1].char_end == len(text)
    assert text[spans[1].char_start : spans[1].char_end] == assistant_part
    assert spans[0].kind is SegmentKind.PREFIX_REAL
    assert spans[1].is_live


def test_second_user_turn_gets_a_blank_line_separator() -> None:
    tokenizer = MockTokenizer()
    first, _offsets = _mock_conversation(tokenizer)
    tokens = (
        *first,
        tokenizer.encode_special("<user_start>"),
        *tokenizer.encode("ok"),
        tokenizer.encode_special("<user_end>"),
        tokenizer.encode_special("<assistant_start>"),
    )
    special = build_special_display(tokenizer)

    text, _spans = pretty_view(tokenizer, tokens, (), special)

    assert text == (
        f"{USER_MARKER}hi\n\n{ASSISTANT_MARKER}yo\n\n{USER_MARKER}ok\n\n{ASSISTANT_MARKER}"
    )


def test_leading_marker_has_no_leading_newlines() -> None:
    tokenizer = MockTokenizer()
    tokens = (
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<user_start>"),
        *tokenizer.encode("q"),
    )
    special = build_special_display(tokenizer)

    text, _spans = pretty_view(tokenizer, tokens, (), special)

    assert text == f"{USER_MARKER}q"


def test_zero_width_bos_boundary_maps_span_start_exactly() -> None:
    """MockTokenizer's BOS decodes to nothing: the raw boundary it shares with
    the next token must still map a span starting at raw offset 0."""

    tokenizer = MockTokenizer()
    tokens, offsets = _mock_conversation(tokenizer)
    assert offsets[0] == offsets[1] == 0
    special = build_special_display(tokenizer)
    raw_spans = (_span(0, offsets[-1], 3, SegmentKind.PREFIX_REAL, False),)

    text, spans = pretty_view(tokenizer, tokens, raw_spans, special)

    assert spans[0].char_start == 0
    assert spans[0].char_end == len(text)


def test_empty_conversation_renders_empty_and_rejects_spans() -> None:
    tokenizer = MockTokenizer()
    special = build_special_display(tokenizer)

    assert pretty_view(tokenizer, (), (), special) == ("", ())
    with pytest.raises(ValueError, match="empty conversation"):
        pretty_view(
            tokenizer,
            (),
            (_span(0, 1, 0, SegmentKind.PREFIX_REAL, False),),
            special,
        )


def test_non_token_boundary_span_offset_fails_loud() -> None:
    tokenizer = MockTokenizer()
    tokens, offsets = _mock_conversation(tokenizer)
    special = build_special_display(tokenizer)
    inside_a_token = offsets[-1] + 7

    with pytest.raises(ValueError, match="not a token boundary"):
        pretty_view(
            tokenizer,
            tokens,
            (_span(0, inside_a_token, 0, SegmentKind.PREFIX_REAL, False),),
            special,
        )


def test_stream_display_text_substitutes_literals_and_strips_leading_newlines() -> None:
    tokenizer = MockTokenizer()
    special = build_special_display(tokenizer)
    user_start = tokenizer.decode([tokenizer.encode_special("<user_start>")])
    user_end = tokenizer.decode([tokenizer.encode_special("<user_end>")])
    assistant_start = tokenizer.decode([tokenizer.encode_special("<assistant_start>")])
    raw = f"{user_start}hi{user_end}{assistant_start}yo"

    assert stream_display_text(raw, special) == f"{USER_MARKER}hi\n\n{ASSISTANT_MARKER}yo"
    assert stream_display_text("plain text", special) == "plain text"


def test_mock_bos_decodes_empty_and_is_excluded_from_stream_literals() -> None:
    tokenizer = MockTokenizer()
    special = build_special_display(tokenizer)

    assert tokenizer.decode([special.bos_id]) == ""
    assert all(literal for literal, _replacement in special.literals)


def test_real_tokenizer_roundtrip_with_utf8_and_exact_offsets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real nanochat tokenizer: literals, UTF-8 widths, and span exactness."""

    from nanochat.tokenizer import get_tokenizer

    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(_shared_nanochat_base_dir()))
    tokenizer = get_tokenizer()
    special = build_special_display(tokenizer)
    body = tokenizer.encode("héllo ✅")
    tokens = (
        tokenizer.get_bos_token_id(),
        tokenizer.encode_special("<user_start>"),
        *body,
        tokenizer.encode_special("<user_end>"),
        tokenizer.encode_special("<assistant_start>"),
    )
    offsets = token_to_char_offsets(tokenizer, list(tokens))
    raw_spans = (
        _span(0, offsets[len(tokens) - 1], 1, SegmentKind.PREFIX_REAL, False),
        _span(offsets[len(tokens) - 1], offsets[-1], 5, SegmentKind.SELFCOND, True),
    )

    text, spans = pretty_view(tokenizer, tokens, raw_spans, special)

    assert text == f"{USER_MARKER}héllo ✅\n\n{ASSISTANT_MARKER}"
    assert spans[0].char_start == 0
    assert spans[0].char_end == spans[1].char_start
    assert text[spans[1].char_start :] == f"\n\n{ASSISTANT_MARKER}"

    raw_text = tokenizer.decode(list(tokens))
    assert stream_display_text(raw_text, special) == f"{USER_MARKER}héllo ✅\n\n{ASSISTANT_MARKER}"
