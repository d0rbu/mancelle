from __future__ import annotations

import pytest

from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.token_filter import (
    build_allowed_token_ids,
    decodes_to_control_only,
    default_special_token_ids,
)


class _FilterTokenizer:
    bos_token_id = 0
    eos_token_id = 1

    def get_pad_token_id(self) -> int:
        return 2

    def get_vocab_size(self) -> int:
        return 7

    def decode(self, ids: list[int]) -> str:
        token_id = int(ids[0])
        if token_id == 3:
            return "\n"
        if token_id == 4:
            return ""
        if token_id == 5:
            raise ValueError("bad token")
        return chr(ord("a") + token_id)


def test_allowed_token_filter_removes_special_control_empty_and_decode_failures() -> None:
    tokenizer = _FilterTokenizer()

    assert default_special_token_ids(tokenizer) == {0, 1, 2}
    assert decodes_to_control_only("\n") is True
    assert decodes_to_control_only("x") is False
    assert build_allowed_token_ids(tokenizer) == (6,)


def test_allowed_token_filter_reports_empty_vocab_and_missing_vocab_size() -> None:
    class EmptyVocab:
        vocab_size = 0

        def decode(self, ids: list[int]) -> str:
            return str(ids[0])

    class MissingVocab:
        pass

    with pytest.raises(AxBenchBpbAdvError, match="positive"):
        build_allowed_token_ids(EmptyVocab())
    with pytest.raises(AxBenchBpbAdvError, match="vocab"):
        build_allowed_token_ids(MissingVocab())


def test_allowed_token_filter_reports_when_every_token_is_removed() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="every token"):
        build_allowed_token_ids(_FilterTokenizer(), extra_disallowed_ids=[6])
