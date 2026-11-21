"""Exact Stage 4 prompt ordering, identity, and description tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from steering.eval.axbench_base.stage4_prompts import (
    build_stage4_prompt_tokens,
    load_prompting_descriptions,
    prompt_token_key,
    prompting_prefix_text,
)
from steering.eval.axbench_base.types import AxBenchBaseError, SteeringMode
from tests.axbench_base import paths as test_paths
from tests.axbench_base.unit.test_config import make_valid_config


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 7

    def encode(self, text: str) -> list[int]:
        return [100 + ord(char) for char in text]


def test_ordinary_prompt_is_owned_bos_then_exact_seed_content() -> None:
    tokenizer = _Tokenizer()
    # A payload ID equal to BOS remains payload at position one.
    assert build_stage4_prompt_tokens(
        tokenizer=tokenizer,
        seed_content_tokens=[7, 20, 21],
    ) == [7, 7, 20, 21]


@pytest.mark.parametrize(
    ("mode", "expected_text"),
    [
        (SteeringMode.POSITIVE, "Related to dragons:\n\n"),
        (SteeringMode.NEGATIVE, "Not related to dragons:\n\n"),
    ],
)
def test_prompting_order_is_bos_then_prompt_then_seed(
    mode: SteeringMode,
    expected_text: str,
) -> None:
    tokenizer = _Tokenizer()
    prefix = prompting_prefix_text(mode, "dragons")
    assert prefix == expected_text
    tokens = build_stage4_prompt_tokens(
        tokenizer=tokenizer,
        prompting_prefix=prefix,
        seed_content_tokens=[30, 31],
    )
    assert tokens == [7, *tokenizer.encode(expected_text), 30, 31]


def test_prompt_identity_distinguishes_exact_token_ids() -> None:
    assert prompt_token_key([7, 10, 11]) != prompt_token_key([7, 10, 12])
    assert prompt_token_key([7, 10, 11]) == prompt_token_key([7, 10, 11])


def test_empty_seed_or_prompt_encoding_fails() -> None:
    tokenizer = _Tokenizer()
    with pytest.raises(AxBenchBaseError, match="seed content"):
        build_stage4_prompt_tokens(tokenizer=tokenizer, seed_content_tokens=[])

    class _EmptyTokenizer(_Tokenizer):
        def encode(self, text: str) -> list[int]:
            del text
            return []

    with pytest.raises(AxBenchBaseError, match="encoded to an empty"):
        build_stage4_prompt_tokens(
            tokenizer=_EmptyTokenizer(),
            seed_content_tokens=[1],
            prompting_prefix="prompt",
        )


def test_description_loader_is_strict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(run_tag="prompts")
    path = test_paths.default_paths("prompts").prompting_descriptions(SteeringMode.POSITIVE)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"0": "dragons", "2": "knights"}), encoding="utf-8")
    assert load_prompting_descriptions(cfg, SteeringMode.POSITIVE) == {
        0: "dragons",
        2: "knights",
    }

    path.write_text(json.dumps({"0": ""}), encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="Invalid prompting description"):
        load_prompting_descriptions(cfg, SteeringMode.POSITIVE)
