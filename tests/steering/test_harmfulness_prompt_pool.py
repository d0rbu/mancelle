from __future__ import annotations

import random
from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from omegaconf import OmegaConf

import scripts.eval.harmfulness_eval as harmfulness_eval
import steering.eval.harmfulness_judge as harmfulness_judge
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.eval.context_enumeration import enumerate_harmfulness_pool_contexts
from steering.eval.harmfulness_prompt_pool import (
    build_harmfulness_prompt_pool,
    resolve_harmfulness_prompt_pool,
)
from tests.conftest import get_tokenizer_or_skip

_HARMFUL_PROMPTS = (
    {"prompt": "harmful alpha", "category": "violence"},
    {"prompt": "harmful beta", "category": "fraud"},
    {"prompt": "harmful gamma", "category": "weapons"},
    {"prompt": "harmful delta", "category": "privacy"},
)
_BENIGN_PROMPTS = tuple(
    {"prompt": f"benign {index}", "category": "benign"} for index in range(8)
)


def _fake_load_harmful_prompts() -> list[dict[str, str]]:
    return [dict(row) for row in _HARMFUL_PROMPTS]


def _fake_load_benign_prompts(*, n: int, seed: int) -> list[dict[str, str]]:
    return [dict(row) for row in random.Random(seed).sample(_BENIGN_PROMPTS, n)]


def _patch_prompt_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[
    Callable[[], list[dict[str, str]]],
    Callable[..., list[dict[str, str]]],
]:
    monkeypatch.setattr(
        harmfulness_judge,
        "load_harmful_prompts",
        _fake_load_harmful_prompts,
    )
    monkeypatch.setattr(
        harmfulness_judge,
        "load_benign_prompts",
        _fake_load_benign_prompts,
    )
    return _fake_load_harmful_prompts, _fake_load_benign_prompts


@pytest.mark.parametrize("seed", [0, 42, 1234])
@pytest.mark.parametrize("num_benign", [0, 1, 5])
@pytest.mark.parametrize("max_harmful", [None, 0, 1, 3, 10_000])
def test_resolve_harmfulness_prompt_pool_matches_previous_inline_derivation(
    monkeypatch: pytest.MonkeyPatch,
    seed: int,
    num_benign: int,
    max_harmful: int | None,
) -> None:
    load_harmful_prompts, load_benign_prompts = _patch_prompt_loaders(monkeypatch)

    harmful = load_harmful_prompts()
    if max_harmful is not None and max_harmful > 0:
        rng = random.Random(seed)
        rng.shuffle(harmful)
        harmful = harmful[:max_harmful]
    benign = load_benign_prompts(n=num_benign, seed=seed)
    expected_rows = tuple({**p, "pool": "harmful"} for p in harmful) + tuple(
        {**p, "pool": "benign"} for p in benign
    )

    pool = resolve_harmfulness_prompt_pool(
        seed=seed,
        max_harmful=max_harmful,
        num_benign=num_benign,
    )

    assert pool.rows == expected_rows
    assert pool.harmful == expected_rows[: len(harmful)]
    assert pool.benign == expected_rows[len(harmful) :]
    assert pool.seed == seed
    assert pool.max_harmful == max_harmful
    assert pool.num_benign == num_benign


@pytest.mark.parametrize("max_harmful", [None, 2])
def test_harmfulness_entry_points_render_identical_real_token_sequences(
    monkeypatch: pytest.MonkeyPatch,
    max_harmful: int | None,
) -> None:
    seed = 42
    num_benign = 2
    load_harmful_prompts, load_benign_prompts = _patch_prompt_loaders(monkeypatch)
    tokenizer = get_tokenizer_or_skip()

    pool = resolve_harmfulness_prompt_pool(
        seed=seed,
        max_harmful=max_harmful,
        num_benign=num_benign,
    )
    records = enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool)
    evaluator_rows = harmfulness_eval._load_expected_prompt_pairs(
        OmegaConf.create(
            {
                "seed": seed,
                "num_harmful": max_harmful,
                "num_benign": num_benign,
            }
        )
    )

    identity_keys = ("prompt", "category", "pool")
    pool_identities = [tuple(row[key] for key in identity_keys) for row in pool.rows]
    record_identities = [
        (
            row["prompt"],
            row["category"],
            record.metadata["pool"],
        )
        for row, record in zip(pool.rows, records, strict=True)
    ]
    evaluator_identities = [
        tuple(row[key] for key in identity_keys) for row in evaluator_rows
    ]
    rendered_token_ids = [
        tuple(render_prompt_for_chat_completion(tokenizer, row["prompt"]).token_ids)
        for row in pool.rows
    ]

    assert pool_identities == record_identities == evaluator_identities
    assert rendered_token_ids == [record.token_ids for record in records]


def test_build_harmfulness_prompt_pool_does_not_mutate_harmful_input() -> None:
    harmful = [dict(row) for row in _HARMFUL_PROMPTS]
    original = [dict(row) for row in harmful]

    build_harmfulness_prompt_pool(
        harmful=harmful,
        benign=(),
        seed=42,
        max_harmful=2,
    )

    assert harmful == original


def test_build_harmfulness_prompt_pool_rejects_bool_seed() -> None:
    with pytest.raises(TypeError, match="seed must be an integer"):
        build_harmfulness_prompt_pool(
            harmful=(),
            benign=(),
            seed=True,
            max_harmful=None,
        )


def test_resolve_harmfulness_prompt_pool_rejects_bool_num_benign() -> None:
    with pytest.raises(TypeError, match="num_benign must be an integer"):
        resolve_harmfulness_prompt_pool(
            seed=42,
            max_harmful=None,
            num_benign=True,
        )


def test_resolve_harmfulness_prompt_pool_rejects_negative_num_benign() -> None:
    with pytest.raises(ValueError, match="num_benign must be non-negative"):
        resolve_harmfulness_prompt_pool(
            seed=42,
            max_harmful=None,
            num_benign=-1,
        )


@pytest.mark.parametrize("invalid_max_harmful", [True, 1.5])
def test_resolve_rejects_invalid_max_harmful_before_loading_sources(
    monkeypatch: pytest.MonkeyPatch,
    invalid_max_harmful: Any,
) -> None:
    load_harmful = MagicMock()
    load_benign = MagicMock()
    monkeypatch.setattr(
        harmfulness_judge,
        "load_harmful_prompts",
        load_harmful,
    )
    monkeypatch.setattr(
        harmfulness_judge,
        "load_benign_prompts",
        load_benign,
    )

    with pytest.raises(TypeError, match="max_harmful must be an integer or None"):
        resolve_harmfulness_prompt_pool(
            seed=42,
            max_harmful=invalid_max_harmful,
            num_benign=1,
        )

    load_harmful.assert_not_called()
    load_benign.assert_not_called()


@pytest.mark.parametrize(
    ("control", "invalid_value", "message"),
    [
        ("seed", True, "seed must be an integer"),
        ("num_harmful", False, "max_harmful must be an integer or None"),
        ("num_harmful", 1.5, "max_harmful must be an integer or None"),
        ("num_benign", False, "num_benign must be an integer"),
        ("num_benign", 1.5, "num_benign must be an integer"),
    ],
)
def test_harmfulness_eval_rejects_invalid_raw_pool_controls_before_loading(
    monkeypatch: pytest.MonkeyPatch,
    control: str,
    invalid_value: object,
    message: str,
) -> None:
    load_harmful = MagicMock()
    load_benign = MagicMock()
    monkeypatch.setattr(
        harmfulness_judge,
        "load_harmful_prompts",
        load_harmful,
    )
    monkeypatch.setattr(
        harmfulness_judge,
        "load_benign_prompts",
        load_benign,
    )
    controls: dict[str, object] = {
        "seed": 42,
        "num_harmful": None,
        "num_benign": 1,
    }
    controls[control] = invalid_value

    with pytest.raises(TypeError, match=message):
        harmfulness_eval._load_expected_prompt_pairs(OmegaConf.create(controls))

    load_harmful.assert_not_called()
    load_benign.assert_not_called()


def test_build_harmfulness_prompt_pool_rejects_bool_max_harmful() -> None:
    with pytest.raises(TypeError, match="max_harmful must be an integer or None"):
        build_harmfulness_prompt_pool(
            harmful=(),
            benign=(),
            seed=42,
            max_harmful=True,
        )
