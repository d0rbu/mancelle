"""Benchmark tasks through structured conversion and GRPO admission, offline."""

import json
from collections import Counter
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from hydra import compose, initialize_config_dir

from errors import DataLoadError
from scripts.preprocess.convert_sft_to_parquet import main as convert_main
from steering.feature_paths import DatasetPhase, sft_phase_for
from steering.sft.structured_dataset import (
    StructuredEnvironmentCursor,
)
from steering.sft.structured_rollout import (
    EnvironmentFilterReason,
    SegmentationMode,
    StructuredRolloutError,
    build_conversation_environment,
)
from steering.sft_schema import SFTSplit
from tasks.arc import ARC
from tasks.common import TaskMixture
from tasks.gsm8k import GSM8K
from tasks.mixture_names import SFTMixture
from tasks.mixtures import build_task_mixtures, exclude_heldout_questions, question_key
from tasks.mmlu import MMLU
from tests.chatcore_fixtures import _SOURCE_SIZES, _converter_config, _load_artifact
from tests.chatcore_fixtures import benchmark_sources as benchmark_sources
from tests.chatcore_fixtures import converted_chatcore as converted_chatcore
from tests.chatcore_fixtures import rustbpe_tokenizer as rustbpe_tokenizer

_ROOT = Path(__file__).resolve().parents[2]


def test_chatcore_preset_admits_short_answers(converted_chatcore, rustbpe_tokenizer):
    _, train = converted_chatcore
    tokenizer = rustbpe_tokenizer
    with initialize_config_dir(config_dir=str(_ROOT / "configs/training"), version_base=None):
        cfg = compose(config_name="conditional_encdec_sft_grpo_chatcore")
    val = _load_artifact(train.path.with_name("chatcore_val"), tokenizer, "val")
    bos = tokenizer.get_bos_token_id()
    for row in (*train.rows, *val.rows):
        result = build_conversation_environment(
            row,
            bos_token_id=bos,
            context_length=1024,
            segment_length=cfg.segment_length,
            segmentation_mode=SegmentationMode(cfg.segmentation_mode),
            min_reward_tokens=cfg.min_reward_tokens,
        )
        env = result.environment
        assert env is not None
        assert env.targets == row.target_token_ids
        assert env.reward_mask == row.target_mask
        first_answer = row.target_mask.index(True)
        assert env.seg_map[first_answer] > 0
        if sum(row.target_mask) == 2:  # One-letter/numeric choice and assistant-end.
            assert build_conversation_environment(
                row, bos_token_id=bos, context_length=1024
            ).filter_reason is (EnvironmentFilterReason.AT_MOST_ONE_SEGMENT)
            assert (
                build_conversation_environment(
                    row,
                    bos_token_id=bos,
                    context_length=1024,
                    segmentation_mode=SegmentationMode(cfg.segmentation_mode),
                ).filter_reason
                is EnvironmentFilterReason.INSUFFICIENT_REWARD_TARGETS
            )
    assert sft_phase_for("chatcore", SFTSplit.TRAIN) is DatasetPhase.CHATCORE_SFT
    assert sft_phase_for("chatcore", SFTSplit.VAL) is DatasetPhase.CHATCORE_SFT_VAL


def _build(tokenizer, row, *, context_length=1024):
    return build_conversation_environment(
        row,
        bos_token_id=tokenizer.get_bos_token_id(),
        context_length=context_length,
        segment_length=256,
        segmentation_mode=SegmentationMode.TURN_DELIMITED_FIXED_LENGTH,
        min_reward_tokens=1,
    )


def test_real_sources_cover_each_row_once_and_rebuild_deterministically(benchmark_sources):
    calls, _ = benchmark_sources
    train, val = build_task_mixtures(SFTMixture.CHATCORE)
    rebuilt_train, rebuilt_val = build_task_mixtures(SFTMixture.CHATCORE)
    assert val is not None and rebuilt_val is not None
    expected_calls = {
        ("allenai/ai2_arc", subset, split)
        for subset in ("ARC-Easy", "ARC-Challenge")
        for split in ("train", "test")
    } | {
        ("cais/mmlu", "auxiliary_train", "train"),
        ("cais/mmlu", "all", "test"),
        ("openai/gsm8k", "main", "train"),
        ("openai/gsm8k", "main", "test"),
    }
    assert Counter(calls) == Counter(dict.fromkeys(expected_calls, 2))
    prompts = []
    for mixture, rebuilt, split in ((train, rebuilt_train, "train"), (val, rebuilt_val, "test")):
        conversations = [mixture[index] for index in range(len(mixture))]
        assert conversations == [rebuilt[index] for index in range(len(rebuilt))]
        actual = [conversation["messages"][0]["content"] for conversation in conversations]
        assert len(actual) == len(set(actual)) == 24
        subsets = (
            "ARC-Easy",
            "ARC-Challenge",
            "main",
            "auxiliary_train" if split == "train" else "all",
        )
        for subset in subsets:
            for index in range(_SOURCE_SIZES[subset]):
                assert (
                    sum(f"{subset}/{split}/{index} café 数学" in prompt for prompt in actual) == 1
                )
        prompts.append(set(actual))
    assert prompts[0].isdisjoint(prompts[1])


def test_heldout_questions_are_excluded_across_all_sources_without_reordering(benchmark_sources):
    _, sources = benchmark_sources
    # Build the oracle directly from source loaders: passing the baseline
    # through the filtering helper would hide a helper that reordered all rows.
    original_train = TaskMixture(
        [
            ARC(subset="ARC-Easy", split="train"),
            ARC(subset="ARC-Challenge", split="train"),
            MMLU(subset="auxiliary_train", split="train"),
            GSM8K(subset="main", split="train"),
        ]
    )
    original_val = TaskMixture(
        [
            ARC(subset="ARC-Easy", split="test"),
            ARC(subset="ARC-Challenge", split="test"),
            MMLU(subset="all", split="test"),
            GSM8K(subset="main", split="test"),
        ]
    )
    train_keys = [
        ("allenai/ai2_arc", "ARC-Easy", "train"),
        ("allenai/ai2_arc", "ARC-Challenge", "train"),
        ("cais/mmlu", "auxiliary_train", "train"),
        ("openai/gsm8k", "main", "train"),
    ]
    heldout_keys = [
        ("allenai/ai2_arc", "ARC-Easy", "test"),
        ("cais/mmlu", "all", "test"),
        ("openai/gsm8k", "main", "test"),
        ("allenai/ai2_arc", "ARC-Challenge", "test"),
    ]
    removed = []
    for index, (train_key, heldout_key) in enumerate(zip(train_keys, heldout_keys, strict=True)):
        row = sources[train_key][0]
        if train_key[1] == "auxiliary_train":
            row = row["train"]
        removed.append(row["question"])
        heldout_question = sources[heldout_key][0]["question"]
        if index == 0:
            # Equivalent text despite fullwidth characters, a decomposed accent,
            # case, and whitespace; source answers/options deliberately differ.
            heldout_question = (
                " \t"
                + heldout_question.upper().replace("CAFÉ", "ＣＡＦＥ\u0301").replace(" ", " \n\t ")
                + "\n"
            )
            row["choices"] = {"text": ["three", "two"], "label": ["A", "B"]}
            row["answerKey"] = "B"
        row["question"] = heldout_question
    train, val = build_task_mixtures(SFTMixture.CHATCORE)
    assert val is not None
    assert len(train) == 20
    assert len(val) == len(original_val) == 24
    for previous, current, removed_question in zip(
        original_train.tasks, train.tasks, removed, strict=True
    ):
        assert type(current) is type(previous)
        assert current.ds.to_list() == [
            row for row in previous.ds.to_list() if row["question"] != removed_question
        ]
    # The held-out split is authoritative and must remain byte-for-byte equal
    # at the conversation level, including its own answer labels and solutions.
    assert [val[index] for index in range(len(val))] == [
        original_val[index] for index in range(len(original_val))
    ]


@pytest.mark.parametrize(
    "question,expected",
    [
        (" \n CAFÉ\t数学  ", "café 数学"),
        ("ＣＡＦＥ\u0301　数学", "café 数学"),
        ("Straße\nSTRASSE", "strasse strasse"),
        ("What is 1+2?", "what is 1+2?"),
        ("What is 1-2?", "what is 1-2?"),
        ("", ""),
    ],
)
def test_question_identity_normalization_preserves_meaningful_symbols(question, expected):
    assert question_key(question) == expected
    assert question_key(expected) == expected


@pytest.mark.parametrize("remove_all", [False, True], ids=["one-source-empty", "all-sources-empty"])
def test_heldout_filter_handles_empty_training_sources(benchmark_sources, remove_all):
    _, sources = benchmark_sources
    build_task_mixtures(SFTMixture.CHATCORE)
    heldout_question = sources[("allenai/ai2_arc", "ARC-Easy", "test")][0]["question"]
    for (_, subset, split), rows in sources.items():
        if split != "train" or (not remove_all and subset != "ARC-Easy"):
            continue
        for row in rows:
            payload = row["train"] if subset == "auxiliary_train" else row
            payload["question"] = heldout_question
    if remove_all:
        with pytest.raises(ValueError, match="no training rows"):
            build_task_mixtures(SFTMixture.CHATCORE)
    else:
        train, val = build_task_mixtures(SFTMixture.CHATCORE)
        assert [len(task) for task in train.tasks] == [0, 5, 7, 9]
        assert len(train) == 21
        assert val is not None and len(val) == 24
        assert all(
            heldout_question not in train[index]["messages"][0]["content"]
            for index in range(len(train))
        )


def test_heldout_filter_empty_inputs_are_explicit(benchmark_sources):
    train, val = build_task_mixtures(SFTMixture.CHATCORE)
    assert val is not None
    before = [task.ds.to_list() for task in train.tasks]
    exclude_heldout_questions(train.tasks, [])
    assert [task.ds.to_list() for task in train.tasks] == before
    with pytest.raises(ValueError, match="no training rows"):
        exclude_heldout_questions([], val.tasks)
    with pytest.raises(ValueError, match="no training rows"):
        exclude_heldout_questions([], [])


@pytest.mark.parametrize("split", ["train", "val", "both"])
def test_converter_split_selection_is_exact(
    tmp_path, monkeypatch, benchmark_sources, rustbpe_tokenizer, split
):
    monkeypatch.setattr(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer", lambda: rustbpe_tokenizer
    )
    convert_main.__wrapped__(_converter_config(tmp_path / "chatcore", split))
    for name, selected in (("train", split != "val"), ("val", split != "train")):
        path = tmp_path / ("chatcore" if name == "train" else "chatcore_val")
        assert path.exists() == selected
        if selected:
            artifact = _load_artifact(path, rustbpe_tokenizer, name)
            assert len(artifact.rows) == 24
            assert [row.original_index for row in artifact.rows] == list(range(24))
            assert len(list(path.glob("*.parquet"))) == 5  # Includes a partial last shard.


def test_conversion_resume_is_byte_identical_and_audits_changed_source(
    converted_chatcore, benchmark_sources, rustbpe_tokenizer
):
    cfg, artifact = converted_chatcore
    before = {path: path.read_bytes() for path in artifact.path.iterdir() if path.is_file()}
    convert_main.__wrapped__(cfg)
    assert {path: path.read_bytes() for path in before} == before
    assert (
        _load_artifact(artifact.path, rustbpe_tokenizer).identity_sha256 == artifact.identity_sha256
    )
    _, sources = benchmark_sources
    sources[("allenai/ai2_arc", "ARC-Easy", "train")][0]["answerKey"] = "B"
    with pytest.raises(DataLoadError, match="row drift"):
        convert_main.__wrapped__(cfg)
    assert {path: path.read_bytes() for path in before} == before


def test_resume_rejects_new_heldout_overlap_before_reusing_train_rows(
    converted_chatcore, benchmark_sources
):
    cfg, artifact = converted_chatcore
    _, sources = benchmark_sources
    before = {path: path.read_bytes() for path in artifact.path.iterdir() if path.is_file()}
    # A held-out source revision can make a previously valid train row unsafe.
    # Re-running conversion must not silently retain the old training payload.
    train_question = sources[("allenai/ai2_arc", "ARC-Easy", "train")][0]["question"]
    sources[("cais/mmlu", "all", "test")][0]["question"] = train_question
    with pytest.raises(DataLoadError, match="identity mismatch"):
        convert_main.__wrapped__(cfg)
    assert {path: path.read_bytes() for path in before} == before


def test_filtered_sources_convert_to_eligible_artifacts_with_heldout_unchanged(
    tmp_path, converted_chatcore, benchmark_sources, rustbpe_tokenizer
):
    _, original_train = converted_chatcore
    original_val = _load_artifact(tmp_path / "chatcore_val", rustbpe_tokenizer, "val")
    _, sources = benchmark_sources
    heldout_question = sources[("allenai/ai2_arc", "ARC-Easy", "test")][0]["question"]
    sources[("cais/mmlu", "auxiliary_train", "train")][0]["train"]["question"] = heldout_question
    convert_main.__wrapped__(_converter_config(tmp_path / "filtered"))
    train = _load_artifact(tmp_path / "filtered", rustbpe_tokenizer)
    val = _load_artifact(tmp_path / "filtered_val", rustbpe_tokenizer, "val")
    assert len(train.rows) == len(original_train.rows) - 1 == 23
    assert [row.original_index for row in train.rows] == list(range(23))
    assert all(heldout_question not in row.text for row in train.rows)
    assert val.rows == original_val.rows
    assert val.identity_sha256 == original_val.identity_sha256
    cursor = StructuredEnvironmentCursor(train, rank=0, world_size=1, geometry_identity={})
    staged = cursor.stage(23, prefetch=0, build=lambda row: _build(rustbpe_tokenizer, row))
    assert staged.filter_counts == {}
    assert len(staged.environments) == 23


@pytest.mark.parametrize(
    "field,value", [("mixture", "standard"), ("split", "val"), ("tokenizer_sha256", "0" * 64)]
)
def test_conversion_and_loader_reject_tampered_identity(
    converted_chatcore, rustbpe_tokenizer, field, value
):
    cfg, artifact = converted_chatcore
    metadata_path = artifact.path / "conversion_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata[field] = value
    metadata_path.write_text(json.dumps(metadata))
    before = metadata_path.read_bytes()
    with pytest.raises(StructuredRolloutError, match=f"{field} mismatch"):
        _load_artifact(artifact.path, rustbpe_tokenizer)
    with pytest.raises(DataLoadError, match=f"{field} expected"):
        convert_main.__wrapped__(cfg)
    assert metadata_path.read_bytes() == before


def test_conversion_and_loader_audit_earlier_shards(converted_chatcore, rustbpe_tokenizer):
    cfg, artifact = converted_chatcore
    path = artifact.path / "shard_00000.parquet"
    table = pq.read_table(path)
    rows = table.to_pylist()
    rows[0]["original_index"] = 1000  # Still structurally valid; provenance is now wrong.
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), path)
    before = path.read_bytes()
    with pytest.raises(StructuredRolloutError, match="digest mismatch"):
        _load_artifact(artifact.path, rustbpe_tokenizer)
    with pytest.raises(DataLoadError, match="row drift"):
        convert_main.__wrapped__(cfg)
    assert path.read_bytes() == before


def test_all_ranks_replay_exact_partition_after_prefetch_resume(
    converted_chatcore, rustbpe_tokenizer
):
    _, artifact = converted_chatcore
    partitions = []
    for rank in range(3):
        geometry = {"context_length": 1024, "segmentation_mode": "turn_delimited_fixed_length"}
        cursor = StructuredEnvironmentCursor(
            artifact, rank=rank, world_size=3, geometry_identity=geometry
        )
        first = cursor.stage(2, prefetch=3, build=lambda row: _build(rustbpe_tokenizer, row))
        assert cursor.state.accepted == cursor.state.attempted == 0
        cursor.commit(first)
        state = json.loads(json.dumps(cursor.state.to_json_dict()))
        resumed = StructuredEnvironmentCursor(
            artifact, rank=rank, world_size=3, geometry_identity=geometry, state=state
        )
        rest = cursor.stage(6, prefetch=0, build=lambda row: _build(rustbpe_tokenizer, row))
        replayed = resumed.stage(6, prefetch=0, build=lambda row: _build(rustbpe_tokenizer, row))
        assert replayed == rest
        cursor.commit(rest)
        indices = [env.original_index for env in (*first.environments, *rest.environments)]
        assert indices == list(range(rank, 24, 3))
        partitions.append(set(indices))
        with pytest.raises(StructuredRolloutError, match="exhausted"):
            cursor.stage(1, prefetch=0, build=lambda row: _build(rustbpe_tokenizer, row))
        assert cursor.state.accepted == 8
    assert set.union(*partitions) == set(range(24))
    assert sum(len(partition) for partition in partitions) == 24


def test_real_token_clipping_and_calculator_reward_mask(converted_chatcore, rustbpe_tokenizer):
    _, artifact = converted_chatcore
    for row in artifact.rows:
        answer_start = next(index for index, scored in enumerate(row.target_mask) if scored)
        empty = _build(rustbpe_tokenizer, row, context_length=answer_start)
        assert empty.environment is None
        assert empty.filter_reason is EnvironmentFilterReason.INSUFFICIENT_REWARD_TARGETS
        clipped = _build(rustbpe_tokenizer, row, context_length=answer_start + 1)
        assert clipped.environment is not None
        assert clipped.environment.targets == row.target_token_ids[: answer_start + 1]
        assert sum(clipped.environment.reward_mask) == 1
        assert clipped.environment.seg_map[-1] > 0
        full = _build(rustbpe_tokenizer, row).environment
        assert full is not None
        assert full.inputs == (rustbpe_tokenizer.get_bos_token_id(), *full.targets[:-1])
        assert not any(full.reward_mask[:answer_start])
        assert full.reward_mask[-1]
        output_start = rustbpe_tokenizer.encode_special("<output_start>")
        if output_start in row.target_token_ids:
            start = row.target_token_ids.index(output_start)
            end = row.target_token_ids.index(rustbpe_tokenizer.encode_special("<output_end>"))
            assert not any(full.reward_mask[start : end + 1])
            assert full.reward_mask[start - 1]  # The model must produce python_end.
            assert any(full.reward_mask[end + 1 :])  # The final answer remains scored.
