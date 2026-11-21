from __future__ import annotations

import random
import re
from types import SimpleNamespace
from typing import Any

import pytest

from nanochat.tokenizer import get_tokenizer
from steering.eval.axbench_upstream.encdec_baselines import rendering
from steering.eval.axbench_upstream.encdec_baselines.rendering import (
    RenderedFitRow,
    render_fit_row,
    targets_from_mask,
    template_prefix_length,
)
from steering.eval.chat_generation import render_prompt_for_chat_completion


@pytest.fixture(scope="module")
def tokenizer() -> Any:
    return get_tokenizer()


def _exact(message: str) -> str:
    return rf"^{re.escape(message)}$"


def test_render_fit_row_real_tokenizer_golden(tokenizer: Any) -> None:
    assert render_fit_row(
        tokenizer,
        input_text="Hi",
        output_text="OK",
        sequence_len=32,
        surface="golden row",
    ) == RenderedFitRow(
        ids=(32759, 32760, 29154, 32761, 32762, 14601, 32763),
        mask=(0, 0, 0, 0, 0, 1, 1),
        prompt_token_count=5,
    )


@pytest.mark.parametrize(
    "input_text",
    ("", "a", "two words", "line one\nline two", "Unicode: λ"),
)
def test_render_fit_row_prefix_equals_completion_prompt(
    tokenizer: Any,
    input_text: str,
) -> None:
    row = render_fit_row(
        tokenizer,
        input_text=input_text,
        output_text="OK",
        sequence_len=64,
        surface=f"input {input_text!r}",
    )

    assert row.ids[: row.prompt_token_count] == tuple(
        render_prompt_for_chat_completion(tokenizer, input_text).token_ids
    )


def test_template_prefix_length_real_tokenizer_contract(tokenizer: Any) -> None:
    ids = render_prompt_for_chat_completion(tokenizer, "1").token_ids

    assert template_prefix_length(tokenizer) == 2
    assert ids[0] == tokenizer.get_bos_token_id()
    assert ids[1] == tokenizer.encode_special("<user_start>")


def test_template_prefix_length_uses_shorter_probe_when_only_lengths_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProbeTokenizer:
        probes = {"1": [1, 2], "2": [1, 2, 3]}

    def render_probe(active_tokenizer: _ProbeTokenizer, prompt: str) -> Any:
        return SimpleNamespace(token_ids=active_tokenizer.probes[prompt])

    monkeypatch.setattr(rendering, "render_prompt_for_chat_completion", render_probe)

    assert template_prefix_length(_ProbeTokenizer()) == 2


def test_template_prefix_length_rejects_identical_probes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _ProbeTokenizer:
        probes = {"1": [1, 2, 3], "2": [1, 2, 3]}

    def render_probe(active_tokenizer: _ProbeTokenizer, prompt: str) -> Any:
        return SimpleNamespace(token_ids=active_tokenizer.probes[prompt])

    monkeypatch.setattr(rendering, "render_prompt_for_chat_completion", render_probe)

    with pytest.raises(
        RuntimeError,
        match=_exact("chat-completion probes have no differing token"),
    ):
        template_prefix_length(_ProbeTokenizer())


def test_render_fit_row_rejects_overflow_naming_surface(tokenizer: Any) -> None:
    with pytest.raises(
        ValueError,
        match=_exact(
            "concept 3 positive row 4: rendered fit row has 7 tokens, exceeding sequence_len=6"
        ),
    ):
        render_fit_row(
            tokenizer,
            input_text="Hi",
            output_text="OK",
            sequence_len=6,
            surface="concept 3 positive row 4",
        )


def test_render_fit_row_rejects_empty_output_naming_surface(tokenizer: Any) -> None:
    with pytest.raises(
        ValueError,
        match=_exact("concept 1 negative row 2: output_text must be nonempty"),
    ):
        render_fit_row(
            tokenizer,
            input_text="Hi",
            output_text="",
            sequence_len=32,
            surface="concept 1 negative row 2",
        )


def test_render_fit_row_rejects_render_without_supervised_tokens() -> None:
    class _AllPromptTokenizer:
        def render_conversation(
            self,
            _conversation: dict[str, Any],
            *,
            max_tokens: int,
        ) -> tuple[list[int], list[int]]:
            assert max_tokens == 33
            return [10, 11, 12], [0, 0, 0]

    with pytest.raises(
        RuntimeError,
        match=_exact("all-prompt render: rendered fit row has no supervised tokens"),
    ):
        render_fit_row(
            _AllPromptTokenizer(),
            input_text="Hi",
            output_text="OK",
            sequence_len=32,
            surface="all-prompt render",
        )


def test_render_fit_row_checks_structural_prompt_equality(
    tokenizer: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = tokenizer.render_conversation

    def mismatched_prompt(conversation: dict[str, Any], **kwargs: Any) -> Any:
        ids, mask = original(conversation, **kwargs)
        if len(conversation["messages"]) == 1:
            ids[-1] = tokenizer.get_bos_token_id()
        return ids, mask

    monkeypatch.setattr(tokenizer, "render_conversation", mismatched_prompt)

    with pytest.raises(
        RuntimeError,
        match=_exact("structural row: rendered prompt tokens do not match completion prompt"),
    ):
        render_fit_row(
            tokenizer,
            input_text="Hi",
            output_text="OK",
            sequence_len=32,
            surface="structural row",
        )


def test_render_fit_row_uses_model_context_beyond_completion_renderer_cap(tokenizer: Any) -> None:
    input_text = " word" * 2100
    prompt = [tokenizer.get_bos_token_id(), tokenizer.encode_special("<user_start>"),
              *tokenizer.encode(input_text), tokenizer.encode_special("<user_end>"),
              tokenizer.encode_special("<assistant_start>")]
    expected = (*prompt, *tokenizer.encode("OK"), tokenizer.encode_special("<assistant_end>"))
    assert len(prompt) > 2048
    row = render_fit_row(tokenizer, input_text=input_text, output_text="OK",
                         sequence_len=len(expected), surface="long fit row")
    assert row.ids == expected
    assert row.prompt_token_count == len(prompt)
    with pytest.raises(ValueError, match="exceeding sequence_len"):
        render_fit_row(tokenizer, input_text=input_text, output_text="OK",
                       sequence_len=len(expected) - 1, surface="long fit row")


def test_targets_from_mask_real_render_golden(tokenizer: Any) -> None:
    row = render_fit_row(
        tokenizer,
        input_text="Hi",
        output_text="OK",
        sequence_len=32,
        surface="target row",
    )

    targets = targets_from_mask(row.ids, row.mask)

    assert targets == [-1, -1, -1, -1, 14601, 32763, -1]
    assert targets[: row.prompt_token_count - 1] == [-1] * (row.prompt_token_count - 1)
    assert targets[row.prompt_token_count - 1 : -1] == [
        14601,
        tokenizer.encode_special("<assistant_end>"),
    ]
    assert targets[-1] == -1


def test_targets_from_mask_matches_old_inline_rule_on_random_cases() -> None:
    generator = random.Random(1729)
    cases = [
        ([10, 11, 12, 13], [0, 0, 0, 0]),
        ([10, 11, 12, 13], [0, 1, 1, 1]),
        ([10, 11, 12, 13], [0, 0, 1, 0]),
        ([10], [0]),
        ([10, 11, 12, 13, 14], [1, 0, 1, 1, 0]),
    ]
    for _ in range(45):
        length = generator.randint(1, 32)
        cases.append(
            (
                [generator.randrange(32768) for _ in range(length)],
                [generator.randrange(2) for _ in range(length)],
            )
        )

    for ids, mask in cases:
        old_inline = [
            ids[index + 1] if mask[index + 1] == 1 else -1 for index in range(len(ids) - 1)
        ] + [-1]
        assert targets_from_mask(ids, mask) == old_inline


def test_targets_from_mask_rejects_length_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match=_exact("ids and mask must have equal length, got 2 and 1"),
    ):
        targets_from_mask([10, 11], [0])


def test_targets_from_mask_rejects_empty_input() -> None:
    with pytest.raises(
        ValueError,
        match=_exact("ids and mask must be nonempty"),
    ):
        targets_from_mask([], [])
