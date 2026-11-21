"""Pin the ``tests/cross_tokenizer_helpers.SimpleTokenizer`` empty-string
contract (X2 §3 broken-fixture #6 + slice-02 plan Cluster D).

X2 flagged that ``SimpleTokenizer.encode("") -> []`` does NOT prepend a BOS
token, while the production ``RustBPETokenizer.encode("")`` may emit a BOS
or fail. This is intentional for cross-tokenizer geometry tests where char
positions are derived from raw ``ord(c)`` and BOS would inflate them.

The contract is documented in ``tests/cross_tokenizer_helpers.py``'s
docstring (slice 02 owns this file). This test pins the behavior so a
future contributor doesn't silently drift the helper into production-shaped
behavior.
"""

from __future__ import annotations

from tests.cross_tokenizer_helpers import SimpleTokenizer


class TestSimpleTokenizerEmptyStringContract:
    """SimpleTokenizer.encode/decode round-trip with empty strings."""

    def test_encode_empty_returns_empty_list(self) -> None:
        tok = SimpleTokenizer()
        assert tok.encode("") == [], (
            "SimpleTokenizer must NOT prepend BOS on empty encode — used by "
            "char-level cross-tokenizer geometry tests where BOS would "
            "inflate positions."
        )

    def test_decode_empty_returns_empty_string(self) -> None:
        tok = SimpleTokenizer()
        assert tok.decode([]) == ""

    def test_round_trip_empty_string(self) -> None:
        tok = SimpleTokenizer()
        ids = tok.encode("")
        assert tok.decode(ids) == ""

    def test_call_empty_string_yields_empty_input_ids(self) -> None:
        tok = SimpleTokenizer()
        out = tok("")
        assert out["input_ids"] == []

    def test_call_empty_list_yields_empty_input_ids_per_text(self) -> None:
        tok = SimpleTokenizer()
        out = tok([""])
        assert out["input_ids"] == [[]]


class TestSimpleTokenizerCharIdContract:
    """Per-char ID == ord(c) — the documented invariant in cross_tokenizer_helpers.py.

    Tests in axbench_bpb/, scripts/test_*, and steering/ all assume this
    one-to-one mapping. Any drift surfaces here.
    """

    def test_ascii_round_trip(self) -> None:
        tok = SimpleTokenizer()
        text = "Hello, world!"
        ids = tok.encode(text)
        assert ids == [ord(c) for c in text]
        assert tok.decode(ids) == text

    def test_get_bos_token_id_returns_zero(self) -> None:
        # Documented at cross_tokenizer_helpers.py:14 — fixed BOS=0.
        # axbench_bpb tests rely on this for seg_map prefix arithmetic.
        tok = SimpleTokenizer()
        assert tok.get_bos_token_id() == 0
