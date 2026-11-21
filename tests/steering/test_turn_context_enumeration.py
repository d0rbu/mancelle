"""Turn annotation, serialization and exact-prefix mutation guards."""

from dataclasses import replace

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

from scripts.eval import build_eval_context_cache as builder
from scripts.eval import validate_eval_context_cache as validator
from scripts.eval.context_cache_common import load_context_records
from steering.eval.context_enumeration import (
    annotate_turn_context_record,
    count_assistant_turns,
    dedupe_context_records,
    enumerate_chat_target_contexts,
    enumerate_harmfulness_pool_contexts,
    turn_supervision_interval,
    validate_turn_context_records,
)
from steering.eval.harmfulness_prompt_pool import build_harmfulness_prompt_pool
from steering.segmentation import SegmentMode, token_to_char_offsets
from steering.sft.turn_spans import (
    ChatSpecialTokens,
    legacy_turn_spans,
    rendered_text_and_turn_char_starts,
)
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer, ChatTokenizer

TURN = SegmentMode.turn_delimited_fixed_tokens


def turn_records(answer="answer", *, system=False, tokenizer=None):
    tokenizer = tokenizer or ChatTokenizer()
    messages = [{"role": "user", "content": "Qé🦋"}, {"role": "assistant", "content": answer}]
    if system:
        messages.insert(0, {"role": "system", "content": "system"})
    return enumerate_chat_target_contexts(
        tokenizer=tokenizer, task_object=[{"messages": messages}],
        max_problems=None, task_name="unit", segment_mode=TURN,
    )


@pytest.mark.parametrize("answer", ["", "é🦋", "a" * 2200, [
    {"type": "python", "text": "1+1"},
    {"type": "python_output", "text": "2"},
    {"type": "python_output", "text": "3"},
    {"type": "text", "text": "#### 2"},
]])
@pytest.mark.parametrize("system", [False, True])
def test_turn_render_and_roundtrip(tmp_path, answer, system):
    # kills: BOS text, mask-derived turns, marker split, dropped/reordered starts, truncation drift.
    tokenizer = ChatTokenizer()
    records = turn_records(answer, system=system, tokenizer=tokenizer)
    full, prefix = records
    assert full.turn_char_starts[:2] == prefix.turn_char_starts
    assert prefix.context_text[prefix.turn_char_starts[-1]:] == "<assistant_start>"
    assert not full.context_text.startswith("<bos>")
    assert len(full.token_ids) <= 2048
    if system:
        assert full.context_text.startswith("<user_start>system\n\nQ")
    for row in records:
        assert (row.context_text, row.turn_char_starts) == rendered_text_and_turn_char_starts(row.token_ids, tokenizer)
    path = builder.write_context_records(tmp_path / "rows.jsonl", records)
    assert load_context_records(path) == records
    table = pq.read_table(builder.write_context_parquet(tmp_path / "parquet", records))
    assert table.schema.field("turn_char_starts") == pa.field("turn_char_starts", pa.list_(pa.int64()), nullable=False)
    assert table["turn_char_starts"].to_pylist() == [list(row.turn_char_starts) for row in records]
    assert table["context_key"].to_pylist() == [row.key for row in records]


@pytest.mark.parametrize("starts", [(), (1,), (0, 0), (0, 99999), (False,), (0, 1.5)])
def test_invalid_starts(starts):
    # kills: accepting empty, non-integral, unordered or out-of-text turn boundaries.
    # All six inputs legitimately share the one ordering/bounds guard on the field.
    with pytest.raises(ValueError, match="turn_char_starts must start at zero, strictly increase, and lie within text"):
        replace(turn_records()[0], turn_char_starts=starts)


@pytest.mark.parametrize("mutation", ["text", "starts", "missing", "unpaired", "mixed"])
def test_annotation_rejections(tmp_path, mutation):
    # kills: silently keeping conflicting duplicates, missing pairs, null Arrow annotations.
    records = turn_records()
    if mutation in {"text", "starts"}:
        changed = replace(records[0], **({"context_text": "X" + records[0].context_text} if mutation == "text" else {"turn_char_starts": (0, 1)}))
        # `text` and `starts` legitimately share both guards: either edit makes the
        # annotation disagree with the exact tokens it is attached to.
        with pytest.raises(ValueError, match="Conflicting duplicate turn annotations for tokens_sha256:"):
            dedupe_context_records([records[0], changed])
        with pytest.raises(ValueError, match="Turn text/boundaries disagree with exact tokens"):
            validate_turn_context_records([changed, records[1]], ChatTokenizer())
    elif mutation == "mixed":
        with pytest.raises(ValueError, match="turn parquet requires non-null turn_char_starts on every row"):
            builder.write_context_parquet(tmp_path / "bad", [records[0], replace(records[1], turn_char_starts=None)])
        assert not (tmp_path / "bad").exists()
    else:
        with pytest.raises(ValueError, match={"missing": "Missing independent context_prefix for tokens_sha256:",
                                              "unpaired": "Missing matching full row for context_prefix"}[mutation]):
            validate_turn_context_records(records[:1] if mutation == "missing" else records[1:], ChatTokenizer())


def test_multi_assistant_turn_rejected_with_its_own_diagnostic():
    # kills: reporting a second round as a completion-prefix disagreement.
    tokenizer = ChatTokenizer()
    conversation = {"messages": [
        {"role": "user", "content": "Q1"}, {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"}, {"role": "assistant", "content": "A2"},
    ]}
    with pytest.raises(ValueError, match="single assistant turn; example 0 has 2"):
        enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[conversation],
                                       max_problems=None, task_name="rounds", segment_mode=TURN)
    # A tool-split single turn keeps its one round, so the count is not span-based.
    tool_turn = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": [
        {"type": "python", "text": "1+1"}, {"type": "python_output", "text": "2"},
        {"type": "text", "text": "A"}]}]}
    ids, _ = tokenizer.render_conversation(tool_turn)
    assert count_assistant_turns(ids, tokenizer) == 1
    assert len(legacy_turn_spans(ids, ChatSpecialTokens.from_tokenizer(tokenizer))) == 4
    assert enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[tool_turn],
                                          max_problems=None, task_name="tool", segment_mode=TURN)


def test_completion_prefix_disagreement():
    # kills: annotating a different prefix slice from the actual completion renderer.
    class WrongPrefix(ChatTokenizer):
        def render_for_completion(self, conversation):
            return super().render_for_completion(conversation)[:-1]

    with pytest.raises(ValueError, match="completion-prefix disagreement"):
        turn_records(tokenizer=WrongPrefix())


@pytest.mark.parametrize("suites", ["core_target", "chat_target,core_target", "chat", "samples"])
def test_turn_core_rejected_before_writing(tmp_path, monkeypatch, suites):
    # kills: wrapping CORE in tags or failing only after creating enumeration artifacts.
    monkeypatch.setattr(builder, "get_tokenizer", ChatTokenizer)
    cfg = OmegaConf.create({"segment_mode": TURN.value, "evals": suites,
                           "contexts_file": str(tmp_path / "rows.jsonl"),
                           "context_parquet_dir": str(tmp_path / "parquet")})
    # All four suite spellings hit the same unsupported-suite admission guard.
    with pytest.raises(
        ValueError,
        match="turn enumeration supports chat_target, harmfulness, and chat_rlvr only",
    ):
        builder.enumerate_contexts_from_config(cfg)
    assert list(tmp_path.iterdir()) == []


def test_harmfulness_turn_contexts_preserve_exact_tokens_and_independent_prefixes(tmp_path):
    tokenizer = ByteChatTokenizer()
    pool = build_harmfulness_prompt_pool(harmful=[{"prompt": "é🦋 harmful question"}],
        benign=[{"prompt": "benign question"}], seed=7, max_harmful=1)
    fixed = enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool)
    turns = enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool, segment_mode=TURN)
    assert len(turns) == 2
    assert [row.token_ids for row in turns] == [row.token_ids for row in fixed]
    assert [row.key for row in turns] == [row.key for row in fixed]
    assert all(row.turn_char_starts is None for row in fixed)
    for index, row in enumerate(turns):
        assert row.example_index == index and row.seed == 7
        assert (row.context_text, row.turn_char_starts) == rendered_text_and_turn_char_starts(row.token_ids, tokenizer)
        assert row.turn_char_starts is not None
        assert len(row.turn_char_starts) == 2
        assert row.context_text[row.turn_char_starts[-1]:] == "<assistant_start>"
        assert "target_real_role" not in row.metadata
    validate_turn_context_records(turns, tokenizer)
    path = builder.write_context_records(tmp_path / "harmfulness.jsonl", turns)
    assert load_context_records(path) == turns


@pytest.mark.parametrize("independent_suite", ["harmfulness", "chat_rlvr"])
@pytest.mark.parametrize("independent_first", [False, True])
def test_mixed_suite_equal_prefix_is_order_independent(independent_suite, independent_first):
    tokenizer = ChatTokenizer()
    full, prefix = turn_records(tokenizer=tokenizer)
    independent = replace(prefix, suite=independent_suite, metadata={})
    records = [independent, full, prefix] if independent_first else [full, prefix, independent]
    unique = dedupe_context_records(records)
    assert len(unique) == 2
    assert {record.token_ids for record in unique} == {full.token_ids, prefix.token_ids}
    validate_turn_context_records(unique, tokenizer)


def test_harmfulness_cli_enumeration_threads_turn_mode_and_seed(monkeypatch):
    tokenizer = ByteChatTokenizer()
    calls = []

    def pool(**kwargs):
        calls.append(kwargs)
        return build_harmfulness_prompt_pool(harmful=[{"prompt": "harmful"}], benign=[{"prompt": "benign"}],
            seed=kwargs["seed"], max_harmful=kwargs["max_harmful"])

    monkeypatch.setattr(validator, "resolve_harmfulness_prompt_pool", pool)
    cfg = OmegaConf.create({"segment_mode": TURN.value, "evals": "harmfulness", "base_seed": 42,
                           "num_seeds": 2, "harm_num_harmful": 1, "harm_num_benign": 1})
    records = validator.enumerate_expected_eval_contexts(cfg, tokenizer)
    assert calls == [{"seed": seed, "max_harmful": 1, "num_benign": 1} for seed in (42, 1042)]
    assert len(records) == 4 and all(record.turn_char_starts is not None for record in records)
    assert [record.seed for record in records] == [42, 42, 1042, 1042]
    validate_turn_context_records(records, tokenizer)


@pytest.mark.parametrize("suite", ["harmfulness", "chat_rlvr"])
@pytest.mark.parametrize("fault", ["reference_role", "completed_response", "second_assistant_turn"])
def test_independent_turn_cache_refuses_references_and_multi_turn_prefixes(fault, suite):
    tokenizer = ByteChatTokenizer()
    pool = build_harmfulness_prompt_pool(harmful=[{"prompt": "question"}], benign=[], seed=0, max_harmful=1)
    row = replace(enumerate_harmfulness_pool_contexts(tokenizer=tokenizer, pool=pool, segment_mode=TURN)[0], suite=suite)
    if fault == "reference_role":
        row = replace(row, metadata={**row.metadata, "target_real_role": "context_prefix"})
    else:
        messages = [{"role": "user", "content": "question"}, {"role": "assistant", "content": "answer"}]
        if fault == "second_assistant_turn":
            messages += [{"role": "user", "content": "next question"}, {"role": "assistant", "content": ""}]
            ids = tokenizer.render_for_completion({"messages": messages})
        else:
            ids, _ = tokenizer.render_conversation({"messages": messages})
        text, starts = rendered_text_and_turn_char_starts(ids, tokenizer)
        row = replace(row, token_ids=tuple(ids), context_text=text, turn_char_starts=starts)
    with pytest.raises(ValueError, match="independent marker-only assistant prefix"):
        validate_turn_context_records([row], tokenizer)


@pytest.mark.parametrize("mutation,message", [
    ("role", "Unknown turn target_real_role: 'unknown'"),
    # `key` and `full-role` legitimately share the pairing guard: one breaks the
    # prefix's pointer, the other leaves two prefixes and no full row to pair with.
    ("key", "Missing matching full row for context_prefix"),
    ("full-role", "Missing matching full row for context_prefix"),
    ("marker", "context_prefix must end with a marker-only assistant turn"),
    # zero/bool/upper are out-of-domain and share the target_start domain guard;
    # `interior` is in-domain, so only the renderer-interval comparison rejects it.
    ("zero", "turn full row has invalid target_start"),
    ("bool", "turn full row has invalid target_start"),
    ("upper", "turn full row has invalid target_start"),
    ("interior", "target_start disagrees with renderer supervision interval"),
])
def test_pair_roles_keys_marker_and_target_start(mutation, message):
    records = turn_records()
    full, prefix = records
    if mutation == "role":
        records = [replace(full, metadata={**full.metadata, "target_real_role": "unknown"})]
    elif mutation == "key":
        records[1] = replace(prefix, metadata={**prefix.metadata, "paired_target_key": "wrong"})
    elif mutation == "full-role":
        records[0] = replace(full, metadata={**full.metadata, "target_real_role": "context_prefix"})
    elif mutation == "marker":
        # Authentic supervision can start after an initial tool output. That
        # exact prefix ends in output_end, so only marker ownership rejects it.
        tokenizer = ChatTokenizer()
        records = [annotate_turn_context_record(row, tokenizer) for row in enumerate_chat_target_contexts(
            tokenizer=tokenizer, task_object=[{"messages": [
                {"role": "user", "content": "Q"}, {"role": "assistant", "content": [
                    {"type": "python_output", "text": "2"}, {"type": "text", "text": "AB"}]}]}],
            max_problems=1, task_name="invalid-prefix")]
    else:
        value = {"zero": 0, "bool": True, "upper": len(full.token_ids), "interior": len(prefix.token_ids) + 1}[mutation]
        records[0] = replace(full, metadata={**full.metadata, "target_start": value})
    with pytest.raises(ValueError, match=message):
        validate_turn_context_records(records, ChatTokenizer())


@pytest.mark.parametrize("cut", [6, 8, 10, 13, 15, 17, 100])
def test_supervision_interval_matches_truncated_tool_renderer(cut):
    tokenizer = ChatTokenizer()
    conversation = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": [
        {"type": "python", "text": "1+1"}, {"type": "python_output", "text": "2345"},
        {"type": "text", "text": "AB"}]}]}
    ids, mask = tokenizer.render_conversation(conversation, max_tokens=cut)
    supervised = [index for index, value in enumerate(mask) if value]
    assert turn_supervision_interval(ids, tokenizer) == (supervised[0], supervised[-1] + 1)
    offsets = token_to_char_offsets(tokenizer, ids)
    assert offsets[supervised[-1] + 1] <= offsets[-1]
