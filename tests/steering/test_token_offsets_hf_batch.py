"""Offline HF singleton batching parity against the existing frozen offset oracle."""

import itertools
import types

import numpy as np
import pytest
import torch as t
import transformers.tokenization_utils_base as hf_base
from tokenizers import Tokenizer, decoders, models
from transformers import (
    GemmaTokenizer,
    PreTrainedTokenizerBase,
    PreTrainedTokenizerFast,
    TokenizersBackend,
)

import steering.feature_shard_splice as splice
import steering.segmentation as segmentation
from steering.segmentation import get_path_stats, token_to_char_offsets
from tests.steering.test_token_offsets_tiktoken_bytes import (
    _reference_token_to_char_offsets,
    _reference_token_to_char_offsets_progressive,
)


def _tokenizer(decoder="fuse", cleanup=False, tokenizer_class=PreTrainedTokenizerFast):
    vocab = {
        "[UNK]": 0, "<bos>": 1, "x": 2, " a . ": 3, "日": 4, "": 5,
        "<0xF0>": 6, "<0x9F>": 7, "<0x98>": 8, "<0x80>": 9, "##ing": 10, ".": 11,
    }
    backend = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    backend.decoder = {
        "fuse": decoders.Fuse(),
        "wordpiece": decoders.WordPiece(prefix="##", cleanup=False),
        "bytes": decoders.Sequence([decoders.ByteFallback(), decoders.Fuse()]),
    }[decoder]
    return tokenizer_class(
        tokenizer_object=backend, unk_token="[UNK]", bos_token="<bos>",
        clean_up_tokenization_spaces=cleanup,
    )


@pytest.mark.parametrize("decoder", ["fuse", "wordpiece", "bytes"])
@pytest.mark.parametrize("cleanup", [False, True])
@pytest.mark.parametrize("size", [0, 1, 16, 17, 31, 32, 33, 70])
def test_offline_hf_matches_frozen_offsets_and_path(decoder, cleanup, size):
    tokenizer = _tokenizer(decoder, cleanup)
    ids = list(itertools.islice(itertools.cycle([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]), size))
    path, expected = _reference_token_to_char_offsets(tokenizer, ids)
    get_path_stats()
    assert token_to_char_offsets(tokenizer, ids) == expected
    stats = get_path_stats()
    assert sum(row["count"] for row in stats.values()) == (1 if ids else 0)
    if ids:
        assert stats[path]["count"] == 1


@pytest.mark.parametrize("decoder", ["fuse", "wordpiece", "bytes"])
@pytest.mark.parametrize("cleanup", [False, True])
def test_hf_shared_cache_cold_and_warm_match_frozen_offsets(decoder, cleanup):
    tokenizer = _tokenizer(decoder, cleanup)
    cache = {}
    for ids in ([1, 2, 3, 4, 5] * 7, [9, 6, 7, 8, 5] * 7, [1, 2, 3, 4, 5] * 7):
        expected = _reference_token_to_char_offsets(tokenizer, ids)[1]
        assert token_to_char_offsets(tokenizer, ids, _singleton_cache=cache) == expected
    assert cache[5] == ""
    assert cache == {tid: tokenizer.decode([tid]) for tid in cache}


def test_hf_shared_cache_misses_clear_and_oversized_bypass(monkeypatch):
    tokenizer = _tokenizer()
    cache = {}
    batches = []
    original_batch = PreTrainedTokenizerBase.batch_decode

    def recording_batch(self, rows, **kwargs):
        batches.append(rows.tolist())
        return original_batch(self, rows, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerBase, "batch_decode", recording_batch)
    monkeypatch.setattr(segmentation, "_SINGLETON_DECODE_CACHE_MAX_ENTRIES", 3)
    for unique in ([2, 5], [5, 2], [5, 4], [4, 3], [1, 2, 3, 4]):
        ids = list(unique) * 20
        assert token_to_char_offsets(tokenizer, ids, _singleton_cache=cache) == (
            _reference_token_to_char_offsets(tokenizer, ids)[1]
        )
        assert len(cache) <= 3
    assert batches == [[[2], [5]], [[4]], [[4], [3]], [[1], [2], [3], [4]]]
    assert cache == {tid: tokenizer.decode([tid]) for tid in (4, 3)}


def test_hf_batch_preserves_unique_order_specials_and_changed_cleanup(monkeypatch):
    tokenizer = _tokenizer()
    ids = [4, 5, 3, 2, 1] * 7
    original_batch = PreTrainedTokenizerBase.batch_decode
    calls = []

    def recording_batch(self, rows, **kwargs):
        calls.append(rows)
        return original_batch(self, rows, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerBase, "batch_decode", recording_batch)
    outputs = []
    for cleanup in (False, True):
        tokenizer.clean_up_tokenization_spaces = cleanup
        expected = _reference_token_to_char_offsets(tokenizer, ids)[1]
        outputs.append(token_to_char_offsets(tokenizer, ids))
        assert outputs[-1] == expected
    assert outputs[0] != outputs[1]
    assert "<bos>" in tokenizer.decode(ids)
    assert len(calls) == 2
    for rows in calls:
        assert rows.dtype == np.dtype(object)
        assert rows.tolist() == [[4], [5], [3], [2], [1]]


@pytest.mark.parametrize("size", [0, 1, 16])
def test_small_hf_inputs_do_not_batch(monkeypatch, size):
    tokenizer = _tokenizer()

    def unexpected_batch(*_args, **_kwargs):
        pytest.fail("small inputs must retain progressive decoding")

    monkeypatch.setattr(PreTrainedTokenizerBase, "batch_decode", unexpected_batch)
    ids = [2] * size
    cache = {}
    assert token_to_char_offsets(tokenizer, ids, _singleton_cache=cache) == (
        _reference_token_to_char_offsets(tokenizer, ids)[1]
    )
    assert cache == {}


class _DecodeOverride(PreTrainedTokenizerFast):
    def decode(self, token_ids, skip_special_tokens=False, **kwargs):
        assert not isinstance(token_ids, np.ndarray)
        decoded = super().decode(token_ids, skip_special_tokens=skip_special_tokens, **kwargs)
        assert isinstance(decoded, str)
        return "prefix" + decoded


class _BatchOverride(PreTrainedTokenizerFast):
    def batch_decode(self, *_args, **_kwargs):
        pytest.fail("custom batch behavior must retain singleton decoding")


class _PrivateDecodeOverride(PreTrainedTokenizerFast):
    def _decode(self, *args, **kwargs):
        return "prefix" + super()._decode(*args, **kwargs)


@pytest.mark.parametrize("tokenizer_class", [_DecodeOverride, _BatchOverride, _PrivateDecodeOverride])
def test_hf_subclass_public_overrides_keep_existing_path(tokenizer_class):
    tokenizer = _tokenizer(tokenizer_class=tokenizer_class)
    ids = [2, 3, 4] * 12
    cache = {}
    assert token_to_char_offsets(tokenizer, ids, _singleton_cache=cache) == (
        _reference_token_to_char_offsets(tokenizer, ids)[1]
    )
    assert cache == {}


@pytest.mark.parametrize("method", ["decode", "batch_decode", "_decode"])
def test_hf_instance_public_overrides_keep_existing_path(monkeypatch, method):
    tokenizer = _tokenizer()
    original_decode = tokenizer._decode if method == "_decode" else tokenizer.decode

    def custom_decode(token_ids, **kwargs):
        assert not isinstance(token_ids, np.ndarray)
        decoded = original_decode(token_ids, **kwargs)
        assert isinstance(decoded, str)
        return "prefix" + decoded

    def custom_batch(*_args, **_kwargs):
        pytest.fail("custom batch behavior must retain singleton decoding")

    monkeypatch.setattr(tokenizer, method, custom_batch if method == "batch_decode" else custom_decode)
    ids = [2, 3, 4] * 12
    cache = {}
    assert token_to_char_offsets(tokenizer, ids, _singleton_cache=cache) == (
        _reference_token_to_char_offsets(tokenizer, ids)[1]
    )
    assert cache == {}


@pytest.mark.parametrize("invalids", [(-1, 2**100), (2**100, -1)])
@pytest.mark.parametrize("warm", [False, True])
def test_hf_invalid_python_integer_keeps_decoder_error(monkeypatch, invalids, warm):
    tokenizer = _tokenizer()
    cache = {}
    if warm:
        monkeypatch.setattr(segmentation, "_SINGLETON_DECODE_CACHE_MAX_ENTRIES", 5)
        token_to_char_offsets(tokenizer, [2, 4, 5, 6, 7] * 5, _singleton_cache=cache)
    ids = [2, 4] * 10 + list(invalids) + [3, 2, 4] * 2
    with pytest.raises(OverflowError) as reference:
        _reference_token_to_char_offsets(tokenizer, ids)
    with pytest.raises(type(reference.value)) as candidate:
        token_to_char_offsets(tokenizer, ids, _singleton_cache=cache)
    assert str(candidate.value) == str(reference.value)


def test_derived_build_cache_is_shared_only_within_one_unchanged_tokenizer_build(monkeypatch):
    tokenizer = _tokenizer()
    ids = (1, 3, 4, 5, 2) * 7
    source = [[splice.SourceSegment(ids, t.tensor([2]), t.tensor([1.0]))]]
    result = splice.ShardSpliceResult(
        docs=(splice.DerivedDoc(((0, 0),), False),) * 2,
        n_cuts_final=0, n_redrawn=0, n_loops=0, n_splits=0,
    )
    meta = {
        "context_window": 128, "segment_mode": "fixed_tokens", "layer": 8,
        "pooling": "max_pool", "context_independent_features": False,
        "ignore_bos_features": True,
    }
    caches = []
    original_offsets = splice.token_to_char_offsets

    def recording_offsets(tok, token_ids, *, _singleton_cache=None):
        caches.append(_singleton_cache)
        return original_offsets(tok, token_ids, _singleton_cache=_singleton_cache)

    monkeypatch.setattr(splice, "token_to_char_offsets", recording_offsets)
    ends = []
    for cleanup, decoder in ((False, decoders.Fuse()), (True, decoders.Fuse()),
                             (True, decoders.WordPiece(prefix="##", cleanup=False))):
        tokenizer.clean_up_tokenization_spaces = cleanup
        tokenizer.backend_tokenizer.decoder = decoder
        expected = _reference_token_to_char_offsets(tokenizer, list(ids))[1][-1]
        shard, texts = splice.build_derived_shard(
            result, source, meta, tokenizer, segment_size=64,
        )
        assert shard["segment_char_ends"].tolist() == [expected, expected]
        assert texts == [tokenizer.decode(list(ids)[1:])] * 2
        ends.append(expected)
    assert len(set(ends)) == 3
    assert all(caches[i] is caches[i + 1] for i in (0, 2, 4))
    assert len({id(caches[i]) for i in (0, 2, 4)}) == 3


def test_hf_progressive_prefixes_use_public_conversion_and_frozen_oracle(monkeypatch):
    tokenizer = _tokenizer("bytes", cleanup=True)
    ids = [6, 7, 8, 9, 2] * 7
    expected = _reference_token_to_char_offsets_progressive(tokenizer, ids)
    original = hf_base.to_py_obj
    observed = []

    def recording_conversion(value):
        observed.append((type(value), len(value)))
        return original(value)

    monkeypatch.setattr(hf_base, "to_py_obj", recording_conversion)
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == expected
    assert observed == [(list if n <= 16 else np.ndarray, n) for n in range(1, 36)]
    assert ids == [6, 7, 8, 9, 2] * 7


@pytest.mark.parametrize("ids", [[True] * 17, [np.int64(2)] * 17, (2,) * 17])
def test_offset_decode_preserves_non_native_input_for_custom_callables(ids):
    class Decode:
        @property
        def __func__(self):
            pytest.fail("custom callable metadata must not be inspected")

        def __call__(self, received):
            assert received is ids
            return "unchanged"

    assert segmentation._decode_offset_ids(Decode(), ids) == "unchanged"


def test_hf_class_decode_patch_receives_original_list(monkeypatch):
    tokenizer = _tokenizer()
    calls = []
    original = PreTrainedTokenizerBase.decode

    def patched(self, ids, **kwargs):
        assert type(ids) is list
        calls.append(list(ids))
        return original(self, ids, **kwargs)

    monkeypatch.setattr(PreTrainedTokenizerBase, "decode", patched)
    ids = [2, 3] * 17
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == (
        _reference_token_to_char_offsets_progressive(tokenizer, ids)
    )
    assert ids in calls


def test_hf_private_decode_getter_and_copy_boundary_remain_public():
    class PrivateGetter(PreTrainedTokenizerFast):
        @property
        def _decode(self):
            if not getattr(self, "probe_active", False):
                return super()._decode
            self.probe_reads += 1
            if self.probe_reads > 20:
                raise RuntimeError(f"getter {self.probe_reads} after {self.probe_completed}")

            def decode(token_ids, **_kwargs):
                assert type(token_ids) is list
                self.probe_completed += 1
                token_ids.append(3)
                return "x" * len(token_ids)

            return decode

    for offsets in (_reference_token_to_char_offsets_progressive,
                    segmentation._token_to_char_offsets_progressive):
        tokenizer = _tokenizer(tokenizer_class=PrivateGetter)
        tokenizer.probe_active = True
        tokenizer.probe_reads = tokenizer.probe_completed = 0
        ids = [2] * 35
        with pytest.raises(RuntimeError, match="^getter 21 after 20$"):
            offsets(tokenizer, ids)
        assert ids == [2] * 35
        assert tokenizer.probe_completed == 20


def test_progressive_decode_lookup_precedes_slice_that_installs_override():
    tokenizer = _tokenizer()
    original = tokenizer.decode
    calls = []

    def changed(self, token_ids):
        calls.append(len(token_ids))
        return "p" + original(token_ids)

    class InstallOnSlice(list):
        def __getitem__(self, index):
            if isinstance(index, slice):
                tokenizer.decode = types.MethodType(changed, tokenizer)
            return super().__getitem__(index)

    ids = InstallOnSlice([2] * 20)
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == [0, 1, *range(3, 22)]
    assert calls == list(range(2, 21))


@pytest.mark.parametrize("ids", [[True] * 17, [np.int64(2)] * 17, (2,) * 17])
def test_hf_offset_decode_retains_non_native_normalizer_input(monkeypatch, ids):
    tokenizer = _tokenizer()
    original = hf_base.to_py_obj
    observed = []

    def recording_conversion(value):
        observed.append(value)
        return original(value)

    monkeypatch.setattr(hf_base, "to_py_obj", recording_conversion)
    expected = tokenizer.decode(ids)
    observed.clear()
    assert segmentation._decode_offset_ids(tokenizer.decode, ids) == expected
    assert observed[0] is ids


_GEMMA_PIECES = ["▁hello", "world", "", "�", "😀", "▁", ".",
                 "<0xC3>", "<0xA9>", "<0xFF>", "<0xE5>", "<0x8F>", "<0xAB>",
                 "<0xzz>", "<0x+0>", "<0x▁A>"]


def _gemma_tokenizer():
    vocab = {"<pad>": 0, "<eos>": 1, "<bos>": 2, "<unk>": 3, "<mask>": 4}
    vocab.update({piece: index + 5 for index, piece in enumerate(_GEMMA_PIECES)})
    return GemmaTokenizer(vocab=vocab, merges=[], clean_up_tokenization_spaces=False)


@pytest.mark.parametrize("pieces", [
    ["▁hello", "world", "😀"],
    ["<0xC3>", "<0xA9>", "<0xFF>"],
    ["<0xC3>", "<0xA9>", "<0xC3>", "<0xA9>"],
    ["<0xE5>", "<0x8F>", "<0xAB>"],
    ["<0xC3>", "", "<0xA9>"],
    ["�", "<0xC3>", "<0xA9>", "▁"],
    ["<0xzz>", "<0x+0>", "<0x▁A>", "<0xC3>", "<0xA9>"],
])
def test_gemma_native_run_lengths_match_every_public_prefix_and_frozen_offsets(pieces):
    tokenizer = _gemma_tokenizer()
    ids = tokenizer.convert_tokens_to_ids(pieces)
    before = tokenizer.backend_tokenizer.to_str()
    expected_lengths = [len(tokenizer.decode(ids[:end])) for end in range(1, len(ids) + 1)]
    assert segmentation._gemma_prefix_lengths(tokenizer, ids) == expected_lengths
    expected = _reference_token_to_char_offsets_progressive(tokenizer, ids)
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == expected
    assert token_to_char_offsets(tokenizer, ids) == _reference_token_to_char_offsets(tokenizer, ids)[1]
    assert tokenizer.backend_tokenizer.to_str() == before
    assert ids == tokenizer.convert_tokens_to_ids(pieces)


@pytest.mark.parametrize("change", ["cleanup", "decoder", "public_default", "private_default",
                                    "borrowed_public", "borrowed_private", "borrowed_converter"])
def test_gemma_changed_decode_contract_falls_back_to_frozen_path(monkeypatch, change):
    tokenizer = _gemma_tokenizer()
    ids = [2, 5, 10, 11, 1]
    if change == "cleanup":
        tokenizer.clean_up_tokenization_spaces = True
        tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True
    elif change == "decoder":
        tokenizer.backend_tokenizer.decoder = decoders.Sequence([decoders.Strip(" ", 1, 0), decoders.Fuse()])
    elif change == "public_default":
        monkeypatch.setattr(PreTrainedTokenizerBase.decode, "__defaults__", (True,))
    elif change == "private_default":
        monkeypatch.setattr(TokenizersBackend._decode, "__defaults__", (False, True))
        tokenizer.clean_up_tokenization_spaces_for_bpe_even_though_it_will_corrupt_output = True
    else:
        name = {"borrowed_public": "decode", "borrowed_private": "_decode",
                "borrowed_converter": "convert_ids_to_tokens"}[change]
        monkeypatch.setattr(tokenizer, name, getattr(_gemma_tokenizer(), name))
    assert segmentation._gemma_prefix_lengths(tokenizer, ids) is None
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == (
        _reference_token_to_char_offsets_progressive(tokenizer, ids)
    )


@pytest.mark.parametrize("invalid", [-1, 2**100])
@pytest.mark.parametrize("first", [False, True])
def test_gemma_invalid_ids_preserve_public_error_order(invalid, first):
    tokenizer = _gemma_tokenizer()
    ids = [invalid, 5] if first else [5, invalid]
    assert segmentation._gemma_prefix_lengths(tokenizer, ids) is None
    with pytest.raises(OverflowError) as reference:
        _reference_token_to_char_offsets_progressive(tokenizer, ids)
    with pytest.raises(type(reference.value)) as candidate:
        segmentation._token_to_char_offsets_progressive(tokenizer, ids)
    assert candidate.value.args == reference.value.args
    assert ids == ([invalid, 5] if first else [5, invalid])


@pytest.mark.parametrize("public_override", [False, True])
def test_gemma_static_guard_does_not_call_private_getter_before_public_decode(monkeypatch, public_override):
    tokenizer = _gemma_tokenizer()
    calls = []

    def private(self):
        calls.append("getter")
        return lambda token_ids, **kwargs: "x" * len(token_ids)

    monkeypatch.setattr(GemmaTokenizer, "_decode", property(private), raising=False)
    if public_override:
        monkeypatch.setattr(tokenizer, "decode", lambda ids: "public")
    ids = [5] * 20
    assert segmentation._gemma_prefix_lengths(tokenizer, ids) is None
    assert calls == []
    expected = _reference_token_to_char_offsets_progressive(tokenizer, ids)
    expected_calls = list(calls)
    calls.clear()
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == expected
    assert calls == expected_calls


def test_gemma_empty_uninitialized_and_unknown_id_keep_original_behavior():
    tokenizer = object.__new__(GemmaTokenizer)
    assert segmentation._token_to_char_offsets_progressive(tokenizer, []) == [0]
    tokenizer = _gemma_tokenizer()
    ids = [5, 999999, 6]
    assert segmentation._gemma_prefix_lengths(tokenizer, ids) is None
    assert segmentation._token_to_char_offsets_progressive(tokenizer, ids) == (
        _reference_token_to_char_offsets_progressive(tokenizer, ids)
    )
