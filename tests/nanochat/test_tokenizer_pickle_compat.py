"""Compatibility tests for the RustBPE tokenizer cache format."""

from __future__ import annotations

import pickle

import tiktoken

from nanochat.tokenizer import RustBPETokenizer, SPECIAL_TOKENS


def _encoding_with_nanochat_specials() -> tiktoken.Encoding:
    base = tiktoken.get_encoding("gpt2")
    special_tokens = {
        token: len(base._mergeable_ranks) + i
        for i, token in enumerate(SPECIAL_TOKENS)
    }
    return tiktoken.Encoding(
        name="nanochat-pickle-compat",
        pat_str=base._pat_str,
        mergeable_ranks=base._mergeable_ranks,
        special_tokens=special_tokens,
    )


def test_rustbpe_save_load_uses_tokenizer_pkl(tmp_path):
    tokenizer = RustBPETokenizer(_encoding_with_nanochat_specials(), "<bos>")

    tokenizer.save(tmp_path)

    pickle_path = tmp_path / "tokenizer.pkl"
    assert pickle_path.exists()
    assert not (tmp_path / "tokenizer.json").exists()
    with pickle_path.open("rb") as f:
        assert isinstance(pickle.load(f), tiktoken.Encoding)

    reloaded = RustBPETokenizer.from_directory(tmp_path)
    text = "hello from the existing tokenizer cache format"
    assert reloaded.encode(text) == tokenizer.encode(text)
    assert reloaded.encode(text, prepend="<bos>", append="<assistant_end>") == (
        tokenizer.encode(text, prepend="<bos>", append="<assistant_end>")
    )
    assert reloaded.decode(tokenizer.encode(text)) == text
