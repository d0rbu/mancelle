"""RLVR rejects malformed calculator structure without changing native generation."""

from __future__ import annotations

import pytest

import steering.eval.chat_self_conditioned_generation as chat_generation
from steering.conditional.encoder_decoder.rlvr_data import (
    RLVRRow,
    grade_completion_result,
    rlvr_row_id,
)
from steering.conditional.encoder_decoder.rlvr_policy import collect_rlvr_group
from steering.conditional.encoder_decoder.sft_grpo_training import (
    configure_grpo_trainable_parameters,
)
from steering.eval.target_conditioning import token_context_key
from tests.steering.selfcond.rlvr_test_helpers import factory as _factory
from tests.steering.selfcond.rlvr_test_helpers import policy as _policy
from tests.steering.selfcond.rlvr_test_helpers import script as _script
from tests.steering.test_chat_self_conditioned_generation import make_fixture


def _gsm8k_row(prompt: tuple[int, ...]) -> RLVRRow:
    identity = {
        "split": "train",
        "task": "GSM8K",
        "source": "source",
        "subset": "main",
        "source_split": "train",
        "source_index": 0,
        "revision": "0" * 40,
        "prompt_token_ids": prompt,
    }
    return RLVRRow(
        **identity,
        row_id=rlvr_row_id(**identity),
        context_key=token_context_key(prompt),
        oracle={"kind": "gsm8k", "answer": "10"},
    )


def _collect(fixture, monkeypatch, script: list[int], *, budget: int):
    _script(monkeypatch, script)
    row = _gsm8k_row(tuple(fixture.prompt))
    grader_calls = []

    def grade(generation):
        grader_calls.append(generation)
        completion = fixture.tokenizer.decode(
            list(generation.token_ids[len(fixture.prompt) :])
        )
        return grade_completion_result(row, completion).reward

    group = collect_rlvr_group(
        fixture.model,
        _factory(fixture),
        fixture.prompt,
        grade,
        group_size=1,
        policy_config=_policy(),
        group_seed=41,
        max_tokens=budget,
        extraction_tokenizer=fixture.source,
    )
    return row, group.trajectories[0], grader_calls


@pytest.mark.parametrize("delimiter_case", ["orphan_end", "nested_start"])
@pytest.mark.parametrize("head_only", [False, True])
def test_invalid_python_delimiter_stops_only_traced_rlvr(
    tmp_path, monkeypatch: pytest.MonkeyPatch, delimiter_case: str, head_only: bool
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    configure_grpo_trainable_parameters(
        fixture.model, train_lm_head=False, train_feature_head_only=head_only
    )
    tok = fixture.tokenizer
    python_start = tok.encode_special("<python_start>")
    python_end = tok.encode_special("<python_end>")
    poison = tok.encode(" #### 10")
    script = (
        [python_end, *poison]
        if delimiter_case == "orphan_end"
        else [python_start, python_start, *poison]
    )
    invalid = python_end if delimiter_case == "orphan_end" else python_start

    _script(monkeypatch, script)
    native = chat_generation.generate_self_conditioned_chat(
        fixture.engine,
        fixture.prompt,
        max_tokens=len(script),
        extraction_tokenizer=fixture.source,
    )
    assert native.token_ids == (*fixture.prompt, *script)
    assert native.mask == (0,) * len(fixture.prompt) + (1,) * len(script)
    assert native.stop_reason == "budget"
    row = _gsm8k_row(tuple(fixture.prompt))
    assert grade_completion_result(row, tok.decode(script)).reward == 1.0

    _row, trajectory, grader_calls = _collect(
        fixture, monkeypatch, script, budget=len(script)
    )
    expected = script[:1] if delimiter_case == "orphan_end" else script[:2]
    assert trajectory.token_ids == (*fixture.prompt, *expected)
    assert trajectory.sampled_mask == (0,) * len(fixture.prompt) + (1,) * len(expected)
    assert trajectory.stop_reason == "invalid_structure"
    assert trajectory.structurally_valid is False
    assert trajectory.invalid_structural_token == invalid
    assert trajectory.reward == 0.0
    assert grader_calls == []


@pytest.mark.parametrize("ending", ["eos", "budget"])
def test_unclosed_python_block_is_zero_reward_without_changing_native_termination(
    tmp_path, monkeypatch: pytest.MonkeyPatch, ending: str
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    tok = fixture.tokenizer
    python_start = tok.encode_special("<python_start>")
    assistant_end = tok.encode_special("<assistant_end>")
    completion = [python_start, *tok.encode(" #### 10")]
    script = [*completion, assistant_end] if ending == "eos" else completion
    budget = len(script)

    _script(monkeypatch, script)
    native = chat_generation.generate_self_conditioned_chat(
        fixture.engine,
        fixture.prompt,
        max_tokens=budget,
        extraction_tokenizer=fixture.source,
    )
    expected_mask = (0,) * len(fixture.prompt) + (1,) * len(completion)
    assert native.token_ids == (*fixture.prompt, *completion)
    assert native.mask == expected_mask
    assert native.stop_reason == ending

    row, trajectory, grader_calls = _collect(
        fixture, monkeypatch, script, budget=budget
    )
    assert grade_completion_result(row, tok.decode(completion)).reward == 1.0
    assert trajectory.token_ids == native.token_ids
    assert trajectory.sampled_mask == native.mask
    assert trajectory.stop_reason == native.stop_reason
    assert trajectory.structurally_valid is False
    assert trajectory.invalid_structural_token == python_start
    assert trajectory.reward == 0.0
    assert grader_calls == []


@pytest.mark.parametrize("forced_tokens_kept", [0, 1, 2])
def test_each_budget_cut_inside_forced_output_is_invalid_but_preserves_native_result(
    tmp_path, monkeypatch: pytest.MonkeyPatch, forced_tokens_kept: int
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    tok = fixture.tokenizer
    output_start = tok.encode_special("<output_start>")
    sampled = [
        *tok.encode("#### 10 "),
        tok.encode_special("<python_start>"),
        *tok.encode("1+1"),
        tok.encode_special("<python_end>"),
    ]
    forced = [output_start, *tok.encode("2"), tok.encode_special("<output_end>")]
    script = [*sampled, *forced]
    budget = len(sampled) + forced_tokens_kept

    _script(monkeypatch, script)
    native = chat_generation.generate_self_conditioned_chat(
        fixture.engine,
        fixture.prompt,
        max_tokens=budget,
        extraction_tokenizer=fixture.source,
    )
    assert native.token_ids == (
        *fixture.prompt,
        *sampled,
        *forced[:forced_tokens_kept],
    )
    assert native.mask == (
        (0,) * len(fixture.prompt)
        + (1,) * len(sampled)
        + (0,) * forced_tokens_kept
    )
    assert native.stop_reason == "budget"

    row, trajectory, grader_calls = _collect(
        fixture, monkeypatch, script, budget=budget
    )
    assert grade_completion_result(
        row,
        tok.decode(list(trajectory.token_ids[len(fixture.prompt) :])),
    ).reward == 1.0
    assert trajectory.token_ids == native.token_ids
    assert trajectory.sampled_mask == native.mask
    assert trajectory.stop_reason == native.stop_reason
    assert trajectory.structurally_valid is False
    assert trajectory.invalid_structural_token == output_start
    assert trajectory.reward == 0.0
    assert grader_calls == []


@pytest.mark.parametrize("head_only", [False, True])
def test_complete_forced_output_at_budget_boundary_remains_valid(
    tmp_path, monkeypatch: pytest.MonkeyPatch, head_only: bool
) -> None:
    fixture = make_fixture(tmp_path, monkeypatch, K=2)
    configure_grpo_trainable_parameters(
        fixture.model, train_lm_head=False, train_feature_head_only=head_only
    )
    tok = fixture.tokenizer
    sampled = [
        *tok.encode("#### 10 "),
        tok.encode_special("<python_start>"),
        *tok.encode("1+1"),
        tok.encode_special("<python_end>"),
    ]
    forced = [
        tok.encode_special("<output_start>"),
        *tok.encode("2"),
        tok.encode_special("<output_end>"),
    ]
    script = [*sampled, *forced]
    budget = len(script)

    _script(monkeypatch, script)
    native = chat_generation.generate_self_conditioned_chat(
        fixture.engine,
        fixture.prompt,
        max_tokens=budget,
        extraction_tokenizer=fixture.source,
    )
    _row, trajectory, grader_calls = _collect(
        fixture, monkeypatch, script, budget=budget
    )
    assert trajectory.token_ids == native.token_ids == (*fixture.prompt, *script)
    assert trajectory.sampled_mask == native.mask
    assert trajectory.sampled_mask[-len(forced) :] == (0,) * len(forced)
    assert trajectory.stop_reason == native.stop_reason == "budget"
    assert trajectory.structurally_valid is True
    assert trajectory.invalid_structural_token is None
    assert trajectory.reward == 1.0
    assert len(grader_calls) == 1
