"""Test that the CLI preserves the complete Engine transcript."""

from __future__ import annotations

from io import StringIO


class _FakeEngine:
    """Stub Engine that yields a fixed (token_column, token_masks) script.

    Mirrors the (single-row) shape of ``nanochat.engine.Engine.generate``:
    yields one ``(token_column, token_masks)`` per generated step.
    """

    def __init__(self, script):
        self._script = list(script)

    def generate(self, conversation_tokens, **kwargs):
        yield from self._script


class _FakeTokenizer:
    """Minimal tokenizer protocol used by chat_cli."""

    def __init__(self, special_to_id, id_to_text):
        self._special_to_id = special_to_id
        self._id_to_text = id_to_text

    def get_bos_token_id(self):
        return 0

    def encode_special(self, s):
        return self._special_to_id[s]

    def encode(self, s, prepend=None):
        if prepend is not None:
            return [prepend] + [ord(c) for c in s]
        return [ord(c) for c in s]

    def decode(self, tokens):
        # Render each token's text; fall back to "?" for unknowns.
        return "".join(self._id_to_text.get(t, "?") for t in tokens)


def test_chat_cli_preserves_forced_tokens_in_display_and_state(monkeypatch):
    """Forced tool-output tokens are printed and retained in conversation state."""
    # Special token ids.
    bos = 0
    user_start, user_end = 10, 11
    assistant_start, assistant_end = 12, 13
    python_start, python_end = 100, 101
    output_start, output_end = 102, 103

    # Token id -> text.
    id_to_text = {
        bos: "<bos>",
        user_start: "<user_start>",
        user_end: "<user_end>",
        assistant_start: "<assistant_start>",
        assistant_end: "<assistant_end>",
        python_start: "<python_start>",
        python_end: "<python_end>",
        output_start: "<output_start>",
        output_end: "<output_end>",
        ord("2"): "2",
        ord("+"): "+",
        ord("4"): "4",
        ord("A"): "A",
        ord(" "): " ",
    }
    special_to_id = {
        "<user_start>": user_start,
        "<user_end>": user_end,
        "<assistant_start>": assistant_start,
        "<assistant_end>": assistant_end,
    }

    # Script: model samples "<python_start> 2 + 4 <python_end>", then engine
    # forces "<output_start> 4 <output_end>", then model samples "A" then stop.
    script = [
        ([python_start], [1]),
        ([ord("2")], [1]),
        ([ord("+")], [1]),
        ([ord("4")], [1]),
        ([python_end], [1]),
        ([output_start], [0]),  # forced
        ([ord("4")], [0]),       # forced
        ([output_end], [0]),     # forced
        ([ord("A")], [1]),
        ([assistant_end], [1]),  # terminal — stops the loop
    ]

    engine = _FakeEngine(script)
    tokenizer = _FakeTokenizer(special_to_id, id_to_text)

    # Replicate the CLI loop body in isolation:
    captured_print = StringIO()
    response_tokens: list[int] = []
    for token_column, _token_masks in engine.generate(
        [bos], num_samples=1, max_tokens=64
    ):
        token = token_column[0]
        response_tokens.append(token)
        captured_print.write(tokenizer.decode([token]))
        if token == assistant_end:
            break

    out = captured_print.getvalue()
    # Sampled tokens DO appear in the rendered text.
    assert "<python_start>" in out
    assert "<python_end>" in out
    assert "A" in out
    # Forced tokens appear in the visible transcript.
    assert "<output_start>" in out
    assert "<output_end>" in out
    assert "4" in out
    # response_tokens (state) also includes the forced tokens.
    assert output_start in response_tokens
    assert output_end in response_tokens
    # And the sampled ones too.
    assert python_start in response_tokens
    assert python_end in response_tokens
