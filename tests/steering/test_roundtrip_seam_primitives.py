"""Unit tests for the round-trip seam primitives (issue #590 feature-shard splice).

Covers the round-trip helpers ``to_text`` / ``is_roundtrip_stable`` that decide
whether a recombined token sequence re-tokenizes to itself under the SAE tokenizer.
"""

from __future__ import annotations

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from steering._roundtrip_seam import (
    is_roundtrip_stable,
    to_text,
)
from steering.segmentation import resolve_bos_token_id
from tests.cross_tokenizer_helpers import SimpleTokenizer

# ---------------------------------------------------------------------------
# to_text — BOS-free detokenization
# ---------------------------------------------------------------------------


def test_to_text_empty_is_empty_string():
    assert to_text((), SimpleTokenizer()) == ""


def test_to_text_without_leading_bos_decodes_all():
    tok = SimpleTokenizer()  # bos id == 0, but does not prepend BOS on encode
    ids = tuple(ord(c) for c in "hi")  # (104, 105) — no leading bos
    assert to_text(ids, tok) == "hi"


def test_to_text_strips_a_single_leading_bos():
    tok = SimpleTokenizer()  # get_bos_token_id() -> 0
    ids = (0, 104, 105)  # leading bos (id 0) then "hi"
    assert to_text(ids, tok) == "hi"


def test_to_text_strips_only_one_leading_bos():
    tok = SimpleTokenizer()
    ids = (0, 0, 104)  # two leading bos; only the first is the prepended bos
    # second 0 is content -> decodes to chr(0) + "h"
    assert to_text(ids, tok) == chr(0) + "h"


def _hf_tokenizer(tokenizer_class=PreTrainedTokenizerFast):
    vocab = {"[UNK]": 0, "<bos>": 1, "a . ": 2, "日": 3, "": 4,
             "<0xF0>": 5, "<0x9F>": 6, "<0x98>": 7, "<0x80>": 8}
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.decoder = decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()])
    return tokenizer_class(tokenizer_object=backend, unk_token="[UNK]", bos_token="<bos>")


def _legacy_to_text(ids, tokenizer):
    if not ids:
        return ""
    bos_id = resolve_bos_token_id(tokenizer)
    body = ids[1:] if bos_id is not None and ids[0] == bos_id else ids
    return tokenizer.decode(list(body))


@pytest.mark.parametrize("ids", [(), (1,), (1, 1), (3, 1, 2), (3, 2, 4),
                                  (1, 3, 2, 4), (1, 5, 6, 7, 8), (4,)])
def test_to_text_hf_matches_legacy_across_bos_unicode_and_cleanup(ids):
    tokenizer = _hf_tokenizer()
    for cleanup in (False, True):
        tokenizer.clean_up_tokenization_spaces = cleanup
        assert to_text(ids, tokenizer) == _legacy_to_text(ids, tokenizer)


def test_to_text_hf_array_becomes_python_list_before_backend_decode(monkeypatch):
    tokenizer = _hf_tokenizer()
    original_decode = PreTrainedTokenizerBase.decode
    original_backend = tokenizer._decode
    public_inputs, backend_calls = [], []

    def public_decode(self, token_ids, **kwargs):
        public_inputs.append(token_ids)
        return original_decode(self, token_ids, **kwargs)

    def backend_decode(token_ids, **kwargs):
        backend_calls.append((token_ids, kwargs))
        return original_backend(token_ids, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerBase, "decode", public_decode)
    monkeypatch.setattr(tokenizer, "_decode", backend_decode)
    assert to_text((1, 3, 1, 4), tokenizer) == "日<bos>"
    assert len(public_inputs) == 1
    assert isinstance(public_inputs[0], np.ndarray)
    assert public_inputs[0].dtype == np.dtype(object)
    assert public_inputs[0].shape == (3,)
    assert backend_calls == [([3, 1, 4], {"skip_special_tokens": False})]


class _ListOnlyHF(PreTrainedTokenizerFast):
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        assert isinstance(token_ids, list)
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        assert isinstance(decoded, str)
        return "custom:" + decoded


@pytest.mark.parametrize("instance_override", [False, True])
def test_to_text_hf_public_decode_overrides_keep_list_inputs(monkeypatch, instance_override):
    tokenizer = _hf_tokenizer(PreTrainedTokenizerFast if instance_override else _ListOnlyHF)
    if instance_override:
        original_decode = tokenizer.decode

        def decode(token_ids, **kwargs):
            assert isinstance(token_ids, list)
            return "custom:" + original_decode(token_ids, **kwargs)

        monkeypatch.setattr(tokenizer, "decode", decode)
    assert to_text((1, 3, 2), tokenizer) == _legacy_to_text((1, 3, 2), tokenizer)


@pytest.mark.parametrize("invalids", [(-1, 2**100), (2**100, -1)])
def test_to_text_hf_preserves_first_invalid_integer_error(invalids):
    tokenizer = _hf_tokenizer()
    ids = (1, 3, *invalids)
    with pytest.raises(OverflowError) as legacy:
        _legacy_to_text(ids, tokenizer)
    with pytest.raises(type(legacy.value)) as candidate:
        to_text(ids, tokenizer)
    assert str(candidate.value) == str(legacy.value)


@pytest.mark.parametrize("ids,expected", [((), ""), ((0,), ""), ((0, 0, 104), "\0h"), ((104,), "h")])
def test_to_text_generic_queries_bos_once_and_keeps_list_inputs(ids, expected):
    class RecordingTokenizer(SimpleTokenizer):
        bos_queries = 0

        def get_bos_token_id(self):
            self.bos_queries += 1
            return super().get_bos_token_id()

        def decode(self, token_ids):
            assert isinstance(token_ids, list)
            return super().decode(token_ids)

    tokenizer = RecordingTokenizer()
    assert to_text(ids, tokenizer) == expected
    assert tokenizer.bos_queries == int(bool(ids))


# ---------------------------------------------------------------------------
# is_roundtrip_stable
# ---------------------------------------------------------------------------


def test_roundtrip_stable_for_bos_free_simple_sequence():
    tok = SimpleTokenizer()  # per-char, no merges, no bos prepend
    ids = tuple(ord(c) for c in "stable text 123")
    assert is_roundtrip_stable(ids, tok) is True


def test_roundtrip_unstable_when_reencode_drops_leading_bos():
    # SimpleTokenizer does not re-prepend BOS, so a BOS-led sequence cannot
    # round-trip through it: to_text strips the 0, re-encode omits it.
    tok = SimpleTokenizer()
    ids = (0,) + tuple(ord(c) for c in "hi")
    assert is_roundtrip_stable(ids, tok) is False


def test_roundtrip_stable_on_real_gemma_sentence(gemma_tok):
    ids = tuple(gemma_tok("The quick brown fox jumps over the lazy dog.",
                          padding=False)["input_ids"])
    assert ids[0] == gemma_tok.bos_token_id  # gemma prepends BOS
    # The BOS-free decoded text re-tokenizes (with BOS) back to the same ids.
    assert is_roundtrip_stable(ids, gemma_tok) is True
    assert to_text(ids, gemma_tok) == gemma_tok.decode(list(ids[1:]))
