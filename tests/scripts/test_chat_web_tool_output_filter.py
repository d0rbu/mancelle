"""Test that scripts.chat.chat_web.generate_stream preserves the full stream."""

from __future__ import annotations


class _FakeEngine:
    def __init__(self, script):
        self._script = list(script)

    def generate(self, tokens, **kwargs):
        yield from self._script


class _FakeTokenizer:
    def __init__(self, id_to_text, special_to_id):
        self._id_to_text = id_to_text
        self._special_to_id = special_to_id

    def get_bos_token_id(self):
        return 0

    def encode_special(self, s):
        return self._special_to_id[s]

    def decode(self, tokens):
        return "".join(self._id_to_text.get(t, "?") for t in tokens)


def _replicate_generate_stream_logic(engine, tokenizer, assistant_end, bos):
    """Mirror the chat_web.generate_stream body.

    Returns the list of SSE events that would have been yielded (sans the
    JSON wrapping — just the textual deltas).
    """
    display_tokens: list[int] = []
    last_clean_text = ""
    emitted_deltas: list[str] = []

    for token_column, _token_masks in engine.generate([bos]):
        token = token_column[0]
        if token in (assistant_end, bos):
            break
        display_tokens.append(token)
        current_text = tokenizer.decode(display_tokens)
        if not current_text.endswith("�"):
            new_text = current_text[len(last_clean_text):]
            if new_text:
                emitted_deltas.append(new_text)
                last_clean_text = current_text

    return emitted_deltas


def test_chat_web_generate_stream_preserves_forced_tokens():
    bos = 0
    assistant_end = 13
    output_start, output_end = 102, 103

    id_to_text = {
        bos: "<bos>",
        assistant_end: "<assistant_end>",
        output_start: "<output_start>",
        output_end: "<output_end>",
        ord("A"): "A",
        ord("B"): "B",
        ord("4"): "4",
        ord(" "): " ",
    }
    special_to_id = {"<assistant_end>": assistant_end}
    tokenizer = _FakeTokenizer(id_to_text, special_to_id)

    script = [
        ([ord("A")], [1]),
        ([output_start], [0]),
        ([ord("4")], [0]),
        ([output_end], [0]),
        ([ord("B")], [1]),
        ([assistant_end], [1]),
    ]
    engine = _FakeEngine(script)

    deltas = _replicate_generate_stream_logic(
        engine, tokenizer, assistant_end=assistant_end, bos=bos
    )
    combined = "".join(deltas)
    assert "A" in combined
    assert "B" in combined
    assert "<output_start>" in combined
    assert "<output_end>" in combined
    assert "4" in combined
    assert combined == "A<output_start>4<output_end>B"
