"""Ordered allowed-token pools remain exact across batching and failed encodes."""

from __future__ import annotations

import sys
from collections import UserDict
from collections.abc import Iterator
from itertools import product
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch as t

from tests.axbench_adv.gcg_pool_reference import original_ids, scalar_allowed_token_ids
from tests.axbench_adv.unit.test_gcg_vendor_worker import GCGSession

_WORKER = sys.modules[GCGSession.__module__]


class _StringMethodTrap(str):
    def isascii(self) -> bool:
        raise AssertionError("str subclasses must retain the original policy")

    def isprintable(self) -> bool:
        raise AssertionError("str subclasses must retain the original policy")


class _IterationString(str):
    def __iter__(self) -> Iterator[str]:
        return iter("\t")


class _EncodingString(str):
    def encode(self, encoding: str = "utf-8", errors: str = "strict") -> bytes:
        raise UnicodeEncodeError(encoding, self, 0, 1, "injected encoding failure")


# The project pool deliberately does not require NFC; preserve that distinction.
_TEXT_POLICY_CASES = [
    pytest.param("", False, False, id="empty"),
    pytest.param(" ", True, True, id="ascii-space"),
    pytest.param("~", True, True, id="last-printable-ascii"),
    pytest.param("\x00", False, False, id="nul"),
    pytest.param("\x1f", False, False, id="last-low-control"),
    pytest.param("\x7f", False, False, id="delete"),
    pytest.param("a\nb", False, False, id="embedded-newline"),
    pytest.param("\u0085", False, False, id="unicode-control"),
    pytest.param("\u00a0", True, True, id="nonbreaking-space"),
    pytest.param("\u2009", True, True, id="thin-space"),
    pytest.param("\u2028", True, True, id="line-separator"),
    pytest.param("\u2029", True, True, id="paragraph-separator"),
    pytest.param("a\u200bb", False, False, id="zero-width-space"),
    pytest.param("\u2060", False, False, id="word-joiner"),
    pytest.param("\ufeff", False, False, id="unicode-bom"),
    pytest.param("\ud800", False, False, id="surrogate"),
    pytest.param("\ue000", True, True, id="private-use"),
    pytest.param("\u0378", True, True, id="unassigned"),
    pytest.param("\ufffd", True, True, id="replacement-character"),
    pytest.param("\U0001f600", True, True, id="non-bmp"),
    pytest.param("\u0301", True, True, id="standalone-combining-mark"),
    pytest.param("\u00e9", True, True, id="composed-accent"),
    pytest.param("e\u0301", False, True, id="decomposed-accent"),
    pytest.param("\u212b", False, True, id="canonical-equivalent"),
    pytest.param("\ufb01", True, True, id="compatibility-ligature"),
    pytest.param("a" * 4096, True, True, id="long-ascii"),
    pytest.param(_StringMethodTrap("trap"), True, True, id="string-method-override"),
    pytest.param(_IterationString("iter"), False, False, id="string-iteration-override"),
    pytest.param(_EncodingString("encode"), False, False, id="string-encoding-failure"),
]


class _TextPolicyTokenizer:
    all_special_ids: tuple[int, ...] = ()
    added_tokens_decoder: dict[int, object] = {}

    def __init__(self, texts: tuple[str, ...], *, fast: bool = False) -> None:
        self.texts = texts
        self.is_fast = fast
        self.ids = {text: index for index, text in reversed(tuple(enumerate(texts)))}

    def decode(self, ids: list[int], **_kwargs: Any) -> str:
        [token_id] = ids
        return self.texts[token_id]

    def encode(self, text: str) -> list[int]:
        return [self.ids[text]]

    def __call__(self, text: str | list[str], **_kwargs: Any) -> dict[str, Any]:
        values = (
            [self.encode(item) for item in text] if isinstance(text, list) else self.encode(text)
        )
        return {"input_ids": values}

    def get_vocab_size(self) -> int:
        return len(self.texts)

    def get_bos_token_id(self) -> int:
        return len(self.texts)

    def encode_special(self, _name: str) -> int:
        return len(self.texts)


def _ascii_strings() -> tuple[str, ...]:
    characters = tuple(map(chr, range(128)))
    return ("", *characters, *("".join(pair) for pair in product(characters, repeat=2)))


@pytest.mark.parametrize("fast", [False, True])
def test_pool_preserves_every_ascii_string_up_to_two_characters(fast: bool) -> None:
    texts = _ascii_strings()
    tokenizer = _TextPolicyTokenizer(texts, fast=fast)
    runtime = SimpleNamespace(tokenizer=tokenizer)
    expected = tuple(
        index
        for index, text in enumerate(texts)
        if text and all(32 <= ord(character) <= 126 for character in text)
    )
    assert len(expected) == 9120
    assert _WORKER._allowed_token_ids(runtime, len(texts)) == expected
    assert scalar_allowed_token_ids(_WORKER, runtime, len(texts)) == expected


@pytest.mark.parametrize("text,baseline_allowed,project_allowed", _TEXT_POLICY_CASES)
def test_pool_preserves_unicode_and_string_subclass_policy(
    text: str, baseline_allowed: bool, project_allowed: bool
) -> None:
    tokenizer = _TextPolicyTokenizer(("!", text), fast=True)
    runtime = SimpleNamespace(tokenizer=tokenizer)
    expected = (0, 1) if baseline_allowed else (0,)
    assert _WORKER._allowed_token_ids(runtime, 2) == expected
    assert scalar_allowed_token_ids(_WORKER, runtime, 2) == expected


class _IndexingList(list[int]):
    def __getitem__(self, index: Any) -> Any:
        return 99 if index == 0 else super().__getitem__(index)


class _LengthList(list[int]):
    def __len__(self) -> int:
        return 1


def _ids_outcome(function: Any, value: object) -> object:
    try:
        return function(value, label="probe")
    except (ValueError, TypeError, AttributeError, RuntimeError) as exc:
        return type(exc), str(exc)


@pytest.mark.parametrize(
    "value,expected",
    [
        ([0], (0,)),
        ([-1], (-1,)),
        ([2**80], (2**80,)),
        ([3, 7], (3, 7)),
        ([], ()),
        ([[3]], (3,)),
        ([[]], ()),
        ({"input_ids": [3]}, (3,)),
        (UserDict(input_ids=[3]), (3,)),
        (_IndexingList([3]), (3,)),
        (_LengthList([3, 7]), (3, 7)),
        (t.tensor([3, 7]), (3, 7)),
        (t.tensor([[3, 7]]), (3, 7)),
        ({"input_ids": t.tensor([3])}, (3,)),
    ],
)
def test_token_id_normalization_matches_frozen_reference(
    value: object, expected: tuple[int, ...]
) -> None:
    assert _WORKER._ids(value, label="probe") == original_ids(value, label="probe") == expected


@pytest.mark.parametrize(
    "value",
    [
        [True],
        [False],
        [3.0],
        ["3"],
        [None],
        (3,),
        3,
        {},
        {"input_ids": None},
        [[3], [7]],
        [[[3]]],
        t.tensor(3),
        t.tensor([True]),
        t.tensor([3.0]),
    ],
)
def test_token_id_normalization_preserves_exact_errors(value: object) -> None:
    expected = _ids_outcome(original_ids, value)
    assert isinstance(expected, tuple) and expected[0] is ValueError
    assert _ids_outcome(_WORKER._ids, value) == expected


def test_token_id_normalization_matches_bounded_container_cross_product() -> None:
    atoms = (0, -1, True, 1.0, None, "x")
    for count in range(3):
        for items in product(atoms, repeat=count):
            row = list(items)
            for value in (row, [row], {"input_ids": row}, UserDict(input_ids=[row])):
                assert _ids_outcome(_WORKER._ids, value) == _ids_outcome(original_ids, value)


class _IntegerSubclass(int):
    pass


class _TypeComparisonTrapMeta(type):
    def __eq__(cls, _other: object) -> bool:
        raise AssertionError("ID validation must not compare types by equality")

    def __hash__(cls) -> int:
        raise AssertionError("ID validation must not hash types")


class _TypeComparisonTrap(metaclass=_TypeComparisonTrapMeta):
    pass


@pytest.mark.parametrize(
    "invalid_id",
    [_IntegerSubclass(3), np.int32(3), np.int64(3), _TypeComparisonTrap()],
    ids=["int-subclass", "numpy-int32", "numpy-int64", "type-comparison-trap"],
)
def test_token_id_normalization_requires_builtin_int_identity(invalid_id: object) -> None:
    value = [0, invalid_id]
    expected = (ValueError, "probe returned invalid input_ids")
    assert _ids_outcome(original_ids, value) == expected
    assert _ids_outcome(_WORKER._ids, value) == expected


class _ObservedIdList(list[object]):
    def __init__(self, values: list[object], events: list[int | str], fail_at: int | None) -> None:
        super().__init__(values)
        self.events = events
        self.fail_at = fail_at

    def __iter__(self) -> Iterator[object]:
        self.events.append("begin")
        try:
            for index, value in enumerate(super().__iter__()):
                self.events.append(index)
                if index == self.fail_at:
                    raise RuntimeError("token-ID iterator failed")
                yield value
        finally:
            self.events.append("end")


@pytest.mark.parametrize(
    "values,fail_at",
    [([], 0), ([3, 7], None), ([True, 7], 1), ([3, True, 7], 2), ([3, 7], 0), ([3, 7], 1)],
    ids=["empty", "valid", "invalid-first", "invalid-middle", "raise-first", "raise-last"],
)
def test_token_id_normalization_preserves_iteration_and_first_failure(
    values: list[object], fail_at: int | None
) -> None:
    expected_events: list[int | str] = []
    actual_events: list[int | str] = []
    expected = _ids_outcome(original_ids, _ObservedIdList(values, expected_events, fail_at))
    actual = _ids_outcome(_WORKER._ids, _ObservedIdList(values, actual_events, fail_at))
    assert actual == expected
    assert actual_events == expected_events


class _EphemeralInvalidId:
    def __init__(self, events: list[str]) -> None:
        self.events = events

    def __del__(self) -> None:
        self.events.append("drop-invalid-id")


class _EphemeralIdList(list[int]):
    def __init__(self, events: list[str]) -> None:
        super().__init__([0, 0])
        self.events = events

    def __iter__(self) -> Iterator[Any]:
        self.events.append("begin")
        try:
            yield _EphemeralInvalidId(self.events)
        finally:
            self.events.append("close-iterator")


def test_token_id_normalization_preserves_subclass_cleanup_order() -> None:
    expected_events: list[str] = []
    actual_events: list[str] = []
    expected = _ids_outcome(original_ids, _EphemeralIdList(expected_events))
    actual = _ids_outcome(_WORKER._ids, _EphemeralIdList(actual_events))
    assert actual == expected == (ValueError, "probe returned invalid input_ids")
    # CPython 3.12 and 3.13 release the temporary ID in different orders.
    # Preserve the frozen reference's order on the running interpreter.
    assert actual_events == expected_events
    assert actual_events[0] == "begin"
    assert sorted(actual_events[1:]) == ["close-iterator", "drop-invalid-id"]


class _PoolTokenizer:
    def __init__(self, *, fast: bool, rich: bool = True) -> None:
        self.is_fast = fast
        self.rich = rich
        self.all_special_ids = [3] if rich else []
        self.added_tokens_decoder = {7: object()} if rich else {}
        self.batch_fault: str | None = None
        self.encode_failure = False
        self.empty = False
        self.batch_sizes: list[int] = []
        self.decoded: list[int] = []
        self.encoded: list[str] = []

    def decode(self, token_ids: list[int], **kwargs: Any) -> Any:
        assert kwargs == {"skip_special_tokens": False, "clean_up_tokenization_spaces": False}
        [token_id] = token_ids
        self.decoded.append(token_id)
        if self.empty:
            return ""
        if self.rich and token_id == 15:
            raise ValueError("invalid decode")
        values = {
            0: "",
            1: "alpha",
            2: " ",
            3: "special",
            4: "a\x00b",
            5: "e\u0301",
            6: "\u00e9",
            7: "added",
            8: "\t",
            9: "a\u200bb",
            10: "\ud800",
            11: "alpha",
            12: "split",
            13: "error",
            14: "tail",
            16: 42,
        }
        return values.get(token_id, f"t{token_id}") if self.rich else f"t{token_id}"

    def _ids(self, text: str) -> list[int]:
        if text == "error" and self.encode_failure:
            raise ValueError("one bad scalar encoding")
        if text.startswith("t") and text[1:].isdigit():
            return [int(text[1:])]
        return {
            "alpha": [1],
            " ": [2],
            "special": [3],
            "a\x00b": [4],
            "e\u0301": [5],
            "\u00e9": [6],
            "added": [7],
            "\t": [8],
            "a\u200bb": [9],
            "split": [1, 14],
            "error": [13],
            "tail": [14],
        }[text]

    def __call__(self, text: str | list[str], **kwargs: Any) -> dict[str, Any]:
        assert kwargs["add_special_tokens"] is False
        assert kwargs["truncation"] is False
        if isinstance(text, list):
            assert self.is_fast
            assert kwargs["return_attention_mask"] is False
            assert kwargs["return_token_type_ids"] is False
            self.batch_sizes.append(len(text))
            if self.batch_fault == "raise":
                raise TypeError("batch API unsupported")
            rows = [self._ids(value) for value in text]
            if self.batch_fault == "short":
                rows.pop()
            if self.batch_fault == "bad_ids":
                rows[-1] = [True]
            return {"input_ids": rows}
        self.encoded.append(text)
        return {"input_ids": self._ids(text)}


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("encode_failure", [False, True])
def test_canonical_pool_matches_frozen_reference(fast: bool, encode_failure: bool) -> None:
    tokenizer = _PoolTokenizer(fast=fast)
    tokenizer.encode_failure = encode_failure
    runtime = SimpleNamespace(tokenizer=tokenizer)
    actual = _WORKER._allowed_token_ids(runtime, 21)
    assert 3 not in tokenizer.decoded and 7 not in tokenizer.decoded
    assert "e\u0301" not in tokenizer.encoded and "a\u200bb" not in tokenizer.encoded
    assert bool(tokenizer.batch_sizes) == fast
    expected = (1, 2, 6, *((13,) if not encode_failure else ()), 14, 17, 18, 19, 20)
    assert actual == expected
    assert actual == scalar_allowed_token_ids(_WORKER, runtime, 21)


@pytest.mark.parametrize("batch_fault", ["raise", "short", "bad_ids"])
@pytest.mark.parametrize("encode_failure", [False, True])
def test_pool_fallback_skips_only_the_failing_token(batch_fault: str, encode_failure: bool) -> None:
    tokenizer = _PoolTokenizer(fast=True)
    tokenizer.batch_fault = batch_fault
    tokenizer.encode_failure = encode_failure
    runtime = SimpleNamespace(tokenizer=tokenizer)
    actual = _WORKER._allowed_token_ids(runtime, 21)
    assert actual == scalar_allowed_token_ids(_WORKER, runtime, 21)
    assert actual[-1] == 20
    assert (13 in actual) != encode_failure


@pytest.mark.parametrize("vocab_size", [1, 513, 1041])
def test_pool_batch_boundaries_preserve_every_id_in_order(vocab_size: int) -> None:
    tokenizer = _PoolTokenizer(fast=True, rich=False)
    runtime = SimpleNamespace(tokenizer=tokenizer)
    actual = _WORKER._allowed_token_ids(runtime, vocab_size)
    assert actual == tuple(range(vocab_size))
    assert actual == scalar_allowed_token_ids(_WORKER, runtime, vocab_size)
    assert max(tokenizer.batch_sizes) <= 512
    assert sum(tokenizer.batch_sizes) == vocab_size


def test_failed_chunk_preserves_all_later_batches() -> None:
    tokenizer = _PoolTokenizer(fast=True)
    tokenizer.encode_failure = True
    runtime = SimpleNamespace(tokenizer=tokenizer)
    actual = _WORKER._allowed_token_ids(runtime, 1041)
    assert actual == scalar_allowed_token_ids(_WORKER, runtime, 1041)
    assert 13 not in actual and actual[-1] == 1040
    assert len(tokenizer.batch_sizes) == 3


@pytest.mark.parametrize("fast", [False, True])
def test_unhandled_scalar_exception_still_propagates(
    monkeypatch: pytest.MonkeyPatch, fast: bool
) -> None:
    tokenizer = _PoolTokenizer(fast=fast)
    original = tokenizer._ids

    def encode(text: str) -> list[int]:
        if text == "error":
            raise KeyError("unexpected scalar failure")
        return original(text)

    monkeypatch.setattr(tokenizer, "_ids", encode)
    runtime = SimpleNamespace(tokenizer=tokenizer)
    with pytest.raises(KeyError, match="unexpected scalar failure"):
        _WORKER._allowed_token_ids(runtime, 21)
    with pytest.raises(KeyError, match="unexpected scalar failure"):
        scalar_allowed_token_ids(_WORKER, runtime, 21)


def test_nonmapping_added_metadata_keeps_reference_behavior() -> None:
    tokenizer = _PoolTokenizer(fast=True)
    tokenizer.added_tokens_decoder = []  # ty: ignore[invalid-assignment]
    runtime = SimpleNamespace(tokenizer=tokenizer)
    actual = _WORKER._allowed_token_ids(runtime, 21)
    assert 7 in actual
    assert actual == scalar_allowed_token_ids(_WORKER, runtime, 21)


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("vocab_size", [0, 4])
def test_empty_canonical_pool_preserves_reference_failure(fast: bool, vocab_size: int) -> None:
    tokenizer = _PoolTokenizer(fast=fast)
    tokenizer.empty = True
    runtime = SimpleNamespace(tokenizer=tokenizer)
    message = "tokenizer has no canonical non-special GCG token ids"
    with pytest.raises(ValueError, match=message):
        _WORKER._allowed_token_ids(runtime, vocab_size)
    with pytest.raises(ValueError, match=message):
        scalar_allowed_token_ids(_WORKER, runtime, vocab_size)
    assert tokenizer.batch_sizes == []
