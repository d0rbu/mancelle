"""Shared prompt-preparation contracts for AxBench upstream drivers."""

from __future__ import annotations

import re
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from steering.eval.axbench_upstream import prompt_prep


def _frame(rows: list[tuple[int, int, float, str]]) -> pd.DataFrame:
    return pd.DataFrame(
        rows,
        columns=("concept_id", "input_id", "factor", "original_prompt"),
    )


def test_build_rendered_prompt_plan_matches_golden_two_concept_frame() -> None:
    frame = _frame(
        [
            (1, 2, 0.0, "prompt 1/2"),
            (0, 3, 0.0, "prompt 0/3"),
            (1, 2, 1.0, "prompt 1/2"),
            (0, 3, 1.0, "prompt 0/3"),
        ]
    )

    def render(_tokenizer: Any, prompt: str) -> Any:
        concept_id, input_id = (int(value) for value in prompt.removeprefix("prompt ").split("/"))
        return SimpleNamespace(
            token_ids=[1, 10 + concept_id, 100 + input_id, 2],
            assistant_end_id=9 + concept_id,
        )

    plan = prompt_prep.build_rendered_prompt_plan(
        frame,
        object(),
        declared_factors=(0.0, 1.0),
        render=render,
    )

    expected_rows = tuple(frame.itertuples(index=False))
    assert plan.rows == expected_rows
    assert plan.rendered_tokens_by_pair == {
        (1, 2): (1, 11, 102, 2),
        (0, 3): (1, 10, 103, 2),
    }
    assert plan.rendered_prompt_by_pair == {
        (1, 2): ("prompt 1/2", 10),
        (0, 3): ("prompt 0/3", 9),
    }
    assert plan.prompt_by_tokens == {
        (1, 10, 103, 2): ("prompt 0/3", 9),
        (1, 11, 102, 2): ("prompt 1/2", 10),
    }


@pytest.mark.parametrize(
    ("rows", "rendered", "declared_factors", "message"),
    [
        (
            [(4, 5, 2.0, "prompt")],
            [SimpleNamespace(token_ids=[1, 2], assistant_end_id=9)],
            (0.0, 1.0),
            "SelfConditioning factor 2.0 is not in the declared factor grid",
        ),
        (
            [(4, 5, 0.0, "prompt")],
            [SimpleNamespace(token_ids=[], assistant_end_id=9)],
            (0.0,),
            "rendered prompt token tuple must be nonempty for concept_id=4, input_id=5",
        ),
        (
            [(4, 5, 0.0, "prompt"), (4, 5, 1.0, "prompt")],
            [
                SimpleNamespace(token_ids=[1, 2], assistant_end_id=9),
                SimpleNamespace(token_ids=[1, 3], assistant_end_id=9),
            ],
            (0.0, 1.0),
            "rendered prompt token tuple varies across factors for concept_id=4, input_id=5",
        ),
        (
            [(4, 5, 0.0, "first"), (4, 5, 1.0, "second")],
            [
                SimpleNamespace(token_ids=[1, 2], assistant_end_id=9),
                SimpleNamespace(token_ids=[1, 2], assistant_end_id=9),
            ],
            (0.0, 1.0),
            "rendered prompt or assistant-end ID varies across factors for concept_id=4, input_id=5",
        ),
    ],
)
def test_build_rendered_prompt_plan_preserves_exact_errors(
    rows: list[tuple[int, int, float, str]],
    rendered: list[SimpleNamespace],
    declared_factors: tuple[float, ...],
    message: str,
) -> None:
    rendered_iter = iter(rendered)

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        prompt_prep.build_rendered_prompt_plan(
            _frame(rows),
            object(),
            declared_factors=declared_factors,
            render=lambda *_args: next(rendered_iter),
        )


def test_first_defect_in_row_order_wins_when_multiple_defects_coexist() -> None:
    # Pins the preserved validation order: the declared-factor check fires
    # before rendering, so a frame whose FIRST row has an off-grid factor
    # raises that error even when a LATER row would fail the empty-token
    # check. (Order inherited byte-for-byte from the pre-extraction driver.)
    frame = _frame(
        [
            (4, 5, 2.0, "off-grid first"),
            (4, 6, 0.0, "empty-render second"),
        ]
    )

    def render(_tokenizer: Any, _prompt: str) -> Any:
        return SimpleNamespace(token_ids=[], assistant_end_id=9)

    message = "SelfConditioning factor 2.0 is not in the declared factor grid"
    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        prompt_prep.build_rendered_prompt_plan(
            frame,
            object(),
            declared_factors=(0.0,),
            render=render,
        )


def test_prompt_token_dedup_uses_canonical_pair_order() -> None:
    frame = _frame(
        [
            (2, 8, 1.0, "source-first"),
            (0, 4, 1.0, "canonical-first"),
            (1, 6, 1.0, "canonical-second"),
        ]
    )
    rendered = {
        "source-first": SimpleNamespace(token_ids=[7, 8], assistant_end_id=9),
        "canonical-first": SimpleNamespace(token_ids=[7, 8], assistant_end_id=9),
        "canonical-second": SimpleNamespace(token_ids=[5, 6], assistant_end_id=9),
    }

    plan = prompt_prep.build_rendered_prompt_plan(
        frame,
        object(),
        declared_factors=(1.0,),
        render=lambda _tokenizer, prompt: rendered[prompt],
    )

    assert tuple(plan.prompt_by_tokens) == ((7, 8), (5, 6))
    assert plan.prompt_by_tokens[(7, 8)] == ("canonical-first", 9)


def test_deduplicated_token_twins_require_one_assistant_end_id() -> None:
    frame = _frame(
        [
            (0, 0, 1.0, "first"),
            (0, 1, 1.0, "second"),
        ]
    )
    rendered = {
        "first": SimpleNamespace(token_ids=[1, 2], assistant_end_id=9),
        "second": SimpleNamespace(token_ids=[1, 2], assistant_end_id=10),
    }
    message = "deduplicated prompt token tuple has inconsistent assistant-end IDs"

    with pytest.raises(ValueError, match=f"^{re.escape(message)}$"):
        prompt_prep.build_rendered_prompt_plan(
            frame,
            object(),
            declared_factors=(1.0,),
            render=lambda _tokenizer, prompt: rendered[prompt],
        )
