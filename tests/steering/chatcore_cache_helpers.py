"""Collision-free tokenizer and local task data for cache integration tests."""

from nanochat.tokenizer import RustBPETokenizer


class ChatTokenizer:
    """Collision-free ASCII vocabulary using the real legacy chat renderer."""

    _specials = {
        "<bos>": 0, "<user_start>": 1, "<user_end>": 2,
        "<assistant_start>": 3, "<assistant_end>": 4,
        "<python_start>": 5, "<python_end>": 6,
        "<output_start>": 7, "<output_end>": 8,
    }
    render_conversation = RustBPETokenizer.render_conversation
    render_for_completion = RustBPETokenizer.render_for_completion

    def get_bos_token_id(self):
        return 0

    def encode_special(self, text):
        return self._specials[text]

    def encode(self, text, **kwargs):
        ids = [16 + ord(char) for char in text]
        prepend = kwargs.get("prepend")
        return ([prepend] if prepend is not None else []) + ids

    def decode(self, ids):
        inverse = {value: key for key, value in self._specials.items()}
        return "".join(inverse[token] if token in inverse else chr(token - 16) for token in ids)


class ByteChatTokenizer(ChatTokenizer):
    """UTF-8 byte vocabulary, including zero-width decoder tokens, below ID 272."""

    def encode(self, text, **kwargs):
        ids = [byte + 16 for byte in text.encode("utf-8")]
        prepend = kwargs.get("prepend")
        return ([prepend] if prepend is not None else []) + ids

    def decode(self, ids):
        inverse = {value: key for key, value in self._specials.items()}
        parts = []
        buffer = bytearray()
        for token in ids:
            if 16 <= token < 272:
                buffer.append(token - 16)
            else:
                parts.append(buffer.decode("utf-8", errors="replace"))
                buffer.clear()
                parts.append(inverse.get(token, f"<unused{token}>"))
        parts.append(buffer.decode("utf-8", errors="replace"))
        return "".join(parts)
