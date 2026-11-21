"""Unit tests for tokenization layer (BOS handling, truncation)."""
from __future__ import annotations

import pytest

from steering.eval.axbench_bpb.tokenization import (
    assert_bos_lengths_match,
    assert_tokenizer_checksum_matches,
    encode_output,
    encode_prefix,
    truncate_prefix_left,
)
from steering.eval.axbench_bpb.types import AxBenchBpbError

# Reuse the SimpleTokenizer (one char = one token) from cross_tokenizer_helpers
from tests.cross_tokenizer_helpers import SimpleTokenizer


def test_encode_prefix_with_bos():
    tok = SimpleTokenizer()
    prefix_tokens = encode_prefix("hi", tok)
    # SimpleTokenizer BOS id = 0; 'h' = 104, 'i' = 105
    assert prefix_tokens == [0, 104, 105]


def test_encode_prefix_produces_exactly_one_bos_with_bpe_tokenizer():
    """BPETokenizer's encode() does not auto-prepend BOS; encode_prefix
    prepends exactly one BOS at position 0.

    SimpleTokenizer mirrors that contract — encode("text") returns ids
    only (no BOS). After encode_prefix, BOS appears exactly once at
    position 0.
    """
    tok = SimpleTokenizer()
    bos_id = tok.get_bos_token_id()
    tokens = encode_prefix("hello world", tok)
    assert tokens[0] == bos_id
    assert tokens.count(bos_id) == 1, (
        f"encode_prefix produced {tokens.count(bos_id)} BOS tokens; expected exactly 1"
    )


def test_encode_prefix_produces_exactly_one_bos_with_gemma_wrapper():
    """HFGemmaTokenizerWrapper._encode_one passes add_special_tokens=False
    so encode_prefix prepends exactly one BOS.

    Stub the wrapper's API (get_bos_token_id + encode that returns ids
    without BOS) and verify a single BOS at position 0.
    """
    class _StubGemmaWrapper:
        def get_bos_token_id(self) -> int:
            return 2

        def encode(self, text: str) -> list[int]:
            # Mirrors HFGemmaTokenizerWrapper._encode_one:
            # add_special_tokens=False → no auto-BOS.
            return [10, 11, 12]

    tokens = encode_prefix("anything", _StubGemmaWrapper())
    assert tokens == [2, 10, 11, 12]
    assert tokens.count(2) == 1


def test_encode_prefix_double_bos_with_raw_hf_tokenizer_is_a_caller_bug():
    """If a tokenizer's encode() ALREADY returns BOS at position 0,
    encode_prefix produces a double-BOS sequence — caller bug.

    Currently encode_prefix doesn't guard against this. The contract is:
    tokenizer.encode(text) MUST NOT include BOS. The failure mode is a
    silent double-BOS, which degrades model behavior.

    This test pins the (current) silent failure so a future fix that
    detects double-BOS will be intentional and tested. Raw HF
    AutoTokenizers default to add_special_tokens=True — wrap them
    (HFGemmaTokenizerWrapper) before passing to encode_prefix.
    """
    class _BadTokenizer:
        """Simulates a raw HF tokenizer that auto-prepends BOS."""

        def get_bos_token_id(self) -> int:
            return 2

        def encode(self, text: str) -> list[int]:
            return [2, 10, 11]  # BOS already included

    tokens = encode_prefix("anything", _BadTokenizer())
    # Document: this currently produces [2, 2, 10, 11] (double BOS) —
    # a known footgun for raw-HF-tokenizer callers.
    assert tokens == [2, 2, 10, 11]
    assert tokens.count(2) == 2  # double BOS — caller bug, not encode_prefix's fault


def test_encode_output_no_bos():
    tok = SimpleTokenizer()
    assert encode_output("ab", tok) == [97, 98]


def test_truncate_prefix_none_if_fits():
    trimmed, truncated = truncate_prefix_left(
        prefix=[1, 2, 3, 4, 5], n_output=3, max_seq_len=10,
    )
    assert trimmed == [1, 2, 3, 4, 5]
    assert truncated is False


def test_truncate_prefix_drops_leftmost():
    trimmed, truncated = truncate_prefix_left(
        prefix=[1, 2, 3, 4, 5], n_output=4, max_seq_len=7,
    )
    assert trimmed == [3, 4, 5]  # dropped leftmost
    assert len(trimmed) + 4 == 7
    assert truncated is True


def test_truncate_prefix_raises_when_output_larger_than_context():
    with pytest.raises(AxBenchBpbError, match="output"):
        truncate_prefix_left(prefix=[1, 2], n_output=10, max_seq_len=5)


def test_truncate_prefix_allowed_zero_returns_empty():
    """n_output == max_seq_len exactly: allowed_prefix = 0, return ([], True)."""
    trimmed, truncated = truncate_prefix_left(
        prefix=[1, 2, 3], n_output=5, max_seq_len=5,
    )
    assert trimmed == []
    assert truncated is True


def test_truncate_prefix_empty_prefix_fits():
    """Empty prefix + small output: no truncation, no errors."""
    trimmed, truncated = truncate_prefix_left(
        prefix=[], n_output=3, max_seq_len=10,
    )
    assert trimmed == []
    assert truncated is False


def test_assert_bos_lengths_match_when_equal():
    class TA:
        def get_bos_token_id(self): return 0
        def decode(self, ids): return "<bos>"
    class TB:
        def get_bos_token_id(self): return 2
        def decode(self, ids): return "<bos>"
    assert_bos_lengths_match(tokenizer_extraction=TA(), tokenizer_nanochat=TB())  # must not raise


def test_assert_bos_lengths_match_when_different():
    class TA:
        def get_bos_token_id(self): return 0
        def decode(self, ids): return "<bos>"     # 5 chars
    class TB:
        def get_bos_token_id(self): return 2
        def decode(self, ids): return "<BOS_TOKEN>"  # 11 chars
    with pytest.raises(AxBenchBpbError, match="BOS"):
        assert_bos_lengths_match(tokenizer_extraction=TA(), tokenizer_nanochat=TB())


def test_assert_tokenizer_checksum_mismatch_raises(tmp_path, monkeypatch):
    # Monkeypatch compute_tokenizer_checksum + load_tokenizer_checksum +
    # find_statistics_path (slice 17 audit P1.B fix updated the helper to
    # resolve via find_statistics_path).
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.compute_tokenizer_checksum",
        lambda _tok: "abc123",
    )
    fake_stats_path = tmp_path / "feature_statistics.safetensors"
    fake_stats_path.touch()
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.find_statistics_path",
        lambda _fd: fake_stats_path,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.load_tokenizer_checksum",
        lambda _fd: "xyz789",
    )
    with pytest.raises(AxBenchBpbError, match="checksum"):
        assert_tokenizer_checksum_matches(tokenizer=object(), feature_dir=tmp_path)


def test_assert_tokenizer_checksum_match_ok(tmp_path, monkeypatch):
    # Slice 17 audit P1.B: helper now resolves via find_statistics_path.
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.compute_tokenizer_checksum",
        lambda _tok: "abc123",
    )
    fake_stats_path = tmp_path / "feature_statistics.safetensors"
    fake_stats_path.touch()
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.find_statistics_path",
        lambda _fd: fake_stats_path,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_bpb.tokenization.load_tokenizer_checksum",
        lambda _fd: "abc123",
    )
    assert_tokenizer_checksum_matches(tokenizer=object(), feature_dir=tmp_path)


def test_assert_bos_lengths_match_hf_attribute_style_equal():
    """HF-style tokenizers expose ``bos_token_id`` as an ATTRIBUTE
    (Gemma's case). assert_bos_lengths_match must accept this contract
    via resolve_bos_token_id — pre-fix, it called the method form and
    crashed with AttributeError."""
    class _HFExtraction:
        bos_token_id = 2
        def decode(self, ids):
            return "<bos>"

    class _NanochatStyle:
        def get_bos_token_id(self): return 0
        def decode(self, ids): return "<bos>"

    # Must NOT raise.
    assert_bos_lengths_match(
        tokenizer_extraction=_HFExtraction(),
        tokenizer_nanochat=_NanochatStyle(),
    )


def test_assert_bos_lengths_match_hf_attribute_style_mismatch():
    """HF extraction tokenizer + nanochat scoring tokenizer that decode
    to different lengths must raise AxBenchBpbError."""
    class _HFExtraction:
        bos_token_id = 2
        def decode(self, ids):
            return "<bos>"  # 5 chars

    class _NanochatStyle:
        def get_bos_token_id(self): return 0
        def decode(self, ids):
            return "<BOS_TOKEN>"  # 11 chars

    with pytest.raises(AxBenchBpbError, match="BOS"):
        assert_bos_lengths_match(
            tokenizer_extraction=_HFExtraction(),
            tokenizer_nanochat=_NanochatStyle(),
        )


def test_assert_bos_lengths_match_both_hf_style():
    """Both tokenizers HF-style — uncommon in production but the helper
    should still resolve via resolve_bos_token_id on both args."""
    class _HFExtraction:
        bos_token_id = 2
        def decode(self, ids): return "<bos>"

    class _HFNanochat:
        bos_token_id = 5
        def decode(self, ids): return "<bos>"

    assert_bos_lengths_match(
        tokenizer_extraction=_HFExtraction(),
        tokenizer_nanochat=_HFNanochat(),
    )


def test_assert_bos_lengths_match_raises_when_bos_unresolvable():
    """If either tokenizer exposes NEITHER bos_token_id attribute nor
    get_bos_token_id method, raise AxBenchBpbError with a descriptive
    message so the operator can fix the tokenizer (not silently mismatch)."""
    class _NoBosContract:
        def decode(self, ids): return "<bos>"

    class _NanochatStyle:
        def get_bos_token_id(self): return 0
        def decode(self, ids): return "<bos>"

    with pytest.raises(AxBenchBpbError, match="Could not resolve BOS"):
        assert_bos_lengths_match(
            tokenizer_extraction=_NoBosContract(),
            tokenizer_nanochat=_NanochatStyle(),
        )
