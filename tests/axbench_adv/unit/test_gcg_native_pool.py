"""Native singleton decoding preserves the public tokenizer's pool contract."""

from __future__ import annotations

from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from transformers import PreTrainedTokenizerBase, PreTrainedTokenizerFast

from tests.axbench_adv.gcg_pool_reference import scalar_allowed_token_ids
from tests.axbench_adv.unit.test_gcg_token_pool import _WORKER, _EncodingString

_VOCAB = {
    "[UNK]": 0,
    "!": 1,
    "a": 2,
    " .": 3,
    "é": 4,
    "e\u0301": 5,
    "\x00": 6,
    "\n": 7,
    "\u200b": 8,
    "\ue000": 9,
    "\u0378": 10,
    "😀": 11,
}


def _tokenizer(cls: type[Any] = PreTrainedTokenizerFast) -> Any:
    tokenizer = cls(
        tokenizer_object=Tokenizer(WordLevel(_VOCAB, unk_token="[UNK]")),
        unk_token="[UNK]",
        clean_up_tokenization_spaces=True,
    )
    tokenizer.add_tokens(["added"])
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _pool(tokenizer: Any, *, reference: bool = False) -> tuple[int, ...]:
    runtime = SimpleNamespace(tokenizer=tokenizer)
    rows = len(tokenizer) + 3  # Include model embedding rows beyond tokenizer IDs.
    if reference:
        return scalar_allowed_token_ids(_WORKER, runtime, rows)
    return _WORKER._allowed_token_ids(runtime, rows)


def test_native_pool_matches_public_decode_and_unicode_policy() -> None:
    tokenizer = _tokenizer()
    assert type(tokenizer) is PreTrainedTokenizerFast
    assert _pool(tokenizer) == _pool(tokenizer, reference=True) == (1, 2, 3, 4, 9, 10, 11)


def test_native_pool_does_not_use_general_sequence_decoding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def general_decode(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("plain native pool IDs do not need general sequence conversion")

    monkeypatch.setattr(_WORKER, "_decode", general_decode)
    assert _pool(_tokenizer()) == (1, 2, 3, 4, 9, 10, 11)


def test_native_pool_preserves_keyword_only_decoder_and_explicit_flags(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tokenizer = _tokenizer()
    original = tokenizer._decode
    calls: list[tuple[int, ...]] = []

    def decode(
        *, token_ids: list[int], skip_special_tokens: bool, clean_up_tokenization_spaces: bool
    ) -> str:
        assert type(token_ids) is list
        assert len(token_ids) == 1 and type(token_ids[0]) is int
        assert skip_special_tokens is clean_up_tokenization_spaces is False
        calls.append(tuple(token_ids))
        return original(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
        )

    monkeypatch.setattr(tokenizer, "_decode", decode)
    actual = _pool(tokenizer)
    actual_calls = list(calls)
    calls.clear()
    assert _pool(tokenizer, reference=True) == actual
    assert calls == actual_calls
    assert calls == [(i,) for i in range(1, 12)] + [(14,), (15,), (16,)]


@pytest.mark.parametrize("owner", ["instance", "fast_class", "base_class"])
def test_native_pool_honors_public_decode_overrides(
    monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    tokenizer = _tokenizer()
    calls: list[tuple[int, ...]] = []

    def replacement(ids: list[int], **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        calls.append(tuple(ids))
        return "!"

    if owner == "instance":
        monkeypatch.setattr(tokenizer, "decode", replacement)
    else:
        target = PreTrainedTokenizerFast if owner == "fast_class" else PreTrainedTokenizerBase
        monkeypatch.setattr(target, "decode", lambda self, ids, **kw: replacement(ids, **kw))
    assert _pool(tokenizer) == (1,)
    actual_calls = list(calls)
    calls.clear()
    assert _pool(tokenizer, reference=True) == (1,)
    assert calls == actual_calls


class _CustomFastTokenizer(PreTrainedTokenizerFast):
    def decode(self, *_args: Any, **kwargs: Any) -> str:
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        return "!"


def test_native_pool_honors_subclass_decode() -> None:
    tokenizer = _tokenizer(_CustomFastTokenizer)
    assert _pool(tokenizer) == _pool(tokenizer, reference=True) == (1,)


@pytest.mark.parametrize("owner", ["instance", "class"])
def test_native_pool_rechecks_decoder_after_each_token(
    monkeypatch: pytest.MonkeyPatch, owner: str
) -> None:
    def run(reference: bool) -> tuple[int, ...]:
        tokenizer = _tokenizer()
        original = tokenizer._decode

        def replacement(ids: list[int], **_kwargs: Any) -> str:
            return "!"

        with monkeypatch.context() as changes:

            def decode(*, token_ids: list[int], **kwargs: Any) -> str:
                if token_ids == [1]:
                    if owner == "instance":
                        changes.setattr(tokenizer, "decode", replacement)
                    else:
                        changes.setattr(
                            PreTrainedTokenizerFast,
                            "decode",
                            lambda self, ids, **kw: replacement(ids, **kw),
                        )
                return original(token_ids=token_ids, **kwargs)

            changes.setattr(tokenizer, "_decode", decode)
            return _pool(tokenizer, reference=reference)

    assert run(False) == run(True) == (1,)


@pytest.mark.parametrize("error", [ValueError, TypeError, RuntimeError, UnicodeError, KeyError])
def test_native_pool_preserves_per_token_exception_boundary(
    monkeypatch: pytest.MonkeyPatch, error: type[Exception]
) -> None:
    tokenizer = _tokenizer()
    original = tokenizer._decode

    def decode(*, token_ids: list[int], **kwargs: Any) -> str:
        if token_ids == [2]:
            raise error("injected decoder failure")
        return original(token_ids=token_ids, **kwargs)

    monkeypatch.setattr(tokenizer, "_decode", decode)
    if error is KeyError:
        for reference in (False, True):
            with pytest.raises(KeyError, match="injected decoder failure"):
                _pool(tokenizer, reference=reference)
    else:
        assert _pool(tokenizer) == _pool(tokenizer, reference=True) == (1, 3, 4, 9, 10, 11)


@pytest.mark.parametrize(
    "text",
    [
        None,
        "\ud800",
        "\x00",
        "",
        "\n",
        "e\u0301",
        pytest.param(_EncodingString("a"), id="encoding-error"),
    ],
)
def test_native_pool_validates_low_level_decoder_output(
    monkeypatch: pytest.MonkeyPatch, text: object
) -> None:
    tokenizer = _tokenizer()
    original = tokenizer._decode

    def decode(*, token_ids: list[int], **kwargs: Any) -> object:
        return text if token_ids == [2] else original(token_ids=token_ids, **kwargs)

    monkeypatch.setattr(tokenizer, "_decode", decode)
    assert _pool(tokenizer) == _pool(tokenizer, reference=True) == (1, 3, 4, 9, 10, 11)


class _LookupTokenizer:
    all_special_ids: tuple[int, ...] = ()
    added_tokens_decoder: dict[int, object] = {}
    is_fast = False

    def __init__(self) -> None:
        self.lookups = 0

    def __len__(self) -> int:
        return 0

    @property
    def decode(self) -> Callable[..., str]:
        self.lookups += 1
        return lambda ids, **kwargs: ("!", "a", "b")[ids[0]]

    def __call__(self, text: str, **_kwargs: Any) -> dict[str, list[int]]:
        return {"input_ids": [("!", "a", "b").index(text)]}


def test_custom_pool_keeps_one_public_decoder_lookup_per_token() -> None:
    candidate, reference = _LookupTokenizer(), _LookupTokenizer()
    assert _pool(candidate) == _pool(reference, reference=True) == (0, 1, 2)
    assert candidate.lookups == reference.lookups == 3
