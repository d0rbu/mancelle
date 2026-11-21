"""Focused contracts for question-only RLVR preparation and native grading."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from datasets import Dataset
from omegaconf import OmegaConf

import scripts.preprocess.prepare_encdec_rlvr as prepare_script
from steering.conditional.encoder_decoder.rlvr_data import (
    RLVR_DATASET_REVISIONS,
    RLVRBundle,
    RLVRDataError,
    RLVRRow,
    RLVRTaskSource,
    admit_rlvr_prompt_cache,
    bundle_manifest_sha256,
    grade_completion_result,
    jsonl_bytes,
    load_rlvr_bundle,
    load_rlvr_jsonl,
    prepare_rlvr_rows,
    rlvr_row_id,
)
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment, token_context_key
from steering.eval.turn_context_contract import canonical_turn_payload, turn_context_contract_hash
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from tasks.arc import ARC
from tasks.gsm8k import GSM8K
from tasks.mixtures import exclude_heldout_questions
from tasks.mmlu import MMLU
from tests.steering.chatcore_cache_helpers import ChatTokenizer
from tests.steering.test_context_cache_in_memory_conversion import _DifferentSourceTokenizer
from tests.steering.test_turn_context_contract import checkpoint_fixture, resolve


@pytest.mark.parametrize("nested_mmlu", [False, True])
def test_real_hf_shuffle_and_mmlu_flatten_preserve_raw_source_indices(nested_mmlu):
    raw = [{"question": f"source row {index}"} for index in range(37)]
    source_indices = np.random.default_rng(42).permutation(len(raw)).tolist()
    if nested_mmlu:
        dataset = Dataset.from_dict({"train": raw}).shuffle(seed=42)
        dataset = dataset.map(lambda row: row["train"], remove_columns=["train"])
    else:
        dataset = Dataset.from_list(raw).shuffle(seed=42)
    assert list(dataset["question"]) == [raw[index]["question"] for index in source_indices]
    kept = [index for index in range(len(raw)) if index not in {0, 3, 8}]
    filtered = dataset.select(kept)
    assert list(filtered["question"]) == [raw[source_indices[index]]["question"] for index in kept]

_REVISION = "a" * 40


class _Tokenizer:
    def encode(self, text):
        return list(text.encode())

    def render_for_completion(self, conversation):
        user = conversation["messages"][0]["content"]
        return [1, *user.encode(), 99]

    def encode_special(self, token):
        assert token == "<assistant_end>"
        return 100


class _Dataset:
    def __init__(self, rows):
        self.rows = list(rows)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        if isinstance(index, str):
            return [row[index] for row in self.rows]
        return self.rows[index]

    def select(self, indices):
        return _Dataset([self.rows[index] for index in indices])


class _Task:
    def __init__(self, rows, *, labels=("A", "B", "C", "D")):
        self.ds = _Dataset(rows)
        self.labels = labels

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        row = self.ds[index]
        if "answer" in row and "label" not in row:
            return {
                "messages": [
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["answer"]},
                ]
            }
        return {
            "messages": [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["label"]},
            ],
            "letters": list(self.labels),
        }


def test_pinned_sources_retain_exact_source_order(monkeypatch) -> None:
    calls = []

    def load_task(cls, subset, split, *, revision):
        calls.append((cls, subset, split, revision))
        return _Task([{"question": repr(calls[-1]), "label": "A"}])

    monkeypatch.setattr(prepare_script, "_load_task", load_task)
    train, validation, removed = prepare_script.load_pinned_sources()

    def identity(item):
        return item.task, item.source, item.subset, item.source_split, item.revision

    specs = (
        (ARC, "ARC-Easy", "allenai/ai2_arc", "ARC-Easy", "ARC-Easy", "arc"),
        (ARC, "ARC-Challenge", "allenai/ai2_arc", "ARC-Challenge", "ARC-Challenge", "arc"),
        (MMLU, "MMLU", "cais/mmlu", "auxiliary_train", "all", "mmlu"),
        (GSM8K, "GSM8K", "openai/gsm8k", "main", "main", "gsm8k"),
    )
    assert len(calls) == 2 * len(specs)
    assert len(train) == len(validation) == len(specs)
    for index, (cls, task, source, train_subset, validation_subset, revision_key) in enumerate(
        specs
    ):
        revision = RLVR_DATASET_REVISIONS[revision_key]
        assert calls[index] == (cls, train_subset, "train", revision)
        assert calls[index + len(specs)] == (cls, validation_subset, "test", revision)
        assert identity(train[index]) == (task, source, train_subset, "train", revision)
        assert identity(validation[index]) == (task, source, validation_subset, "test", revision)
    assert removed == dict.fromkeys((spec[1] for spec in specs), 0)


def _sources(*, duplicate_arc=False, bad_gsm=False):
    arc_rows = [{"question": "same", "label": "B"}]
    if duplicate_arc:
        arc_rows.append(dict(arc_rows[0]))
    return [
        RLVRTaskSource(
            "ARC-Easy",
            "arc",
            "easy",
            "train",
            _REVISION,
            _Task(arc_rows),
            tuple(10 + 2 * index for index in range(len(arc_rows))),
        ),
        RLVRTaskSource(
            "ARC-Challenge",
            "arc",
            "challenge",
            "train",
            _REVISION,
            _Task([{"question": "challenge", "label": "1"}], labels=("1", "2", "3")),
            (20,),
        ),
        RLVRTaskSource(
            "MMLU",
            "mmlu",
            "auxiliary_train",
            "train",
            _REVISION,
            _Task([{"question": "mmlu", "label": "D"}]),
            (30,),
        ),
        RLVRTaskSource(
            "GSM8K",
            "gsm8k",
            "main",
            "train",
            _REVISION,
            _Task(
                [
                    {"question": "math", "answer": "work #### 1,000"},
                    {"question": "bad math", "answer": "unparseable"},
                ]
                if bad_gsm
                else [{"question": "math", "answer": "work #### 1,000"}]
            ),
            (40, 41) if bad_gsm else (40,),
        ),
    ]


def _row(
    task: str = "GSM8K",
    *,
    answer: str = "10",
    labels: Sequence[str] | None = None,
    source_index: int = 0,
    prompt: tuple[int, ...] = (1, 2, 99),
):
    oracle = (
        {"kind": "gsm8k", "answer": answer}
        if task == "GSM8K"
        else {
            "kind": "multiple_choice",
            "answer": answer,
            "labels": list(labels or ()),
        }
    )
    kwargs = {
        "split": "train",
        "task": task,
        "source": "source",
        "subset": "subset",
        "source_split": "train",
        "source_index": source_index,
        "revision": _REVISION,
        "prompt_token_ids": prompt,
        "oracle": oracle,
    }
    return RLVRRow(
        **kwargs,
        row_id=rlvr_row_id(
            split="train",
            task=task,
            source="source",
            subset="subset",
            source_split="train",
            source_index=source_index,
            revision=_REVISION,
            prompt_token_ids=prompt,
        ),
        context_key=token_context_key(prompt),
    )


def test_preparation_is_question_only_retains_duplicates_and_applies_caps() -> None:
    tokenizer = _Tokenizer()
    rows = prepare_rlvr_rows(
        _sources(duplicate_arc=True),
        split="train",
        tokenizer=tokenizer,
        caps={"ARC-Easy": 2, "ARC-Challenge": 1, "MMLU": 1, "GSM8K": 1},
    )
    easy = [row for row in rows if row.task == "ARC-Easy"]
    assert len(easy) == 2
    assert easy[0].prompt_token_ids == easy[1].prompt_token_ids
    assert [row.source_index for row in easy] == [10, 12]
    assert easy[0].row_id != easy[1].row_id
    assert easy[0].oracle == {
        "kind": "multiple_choice",
        "answer": "B",
        "labels": ["A", "B", "C", "D"],
    }
    assert b"REFERENCE_MUST_NOT_AFFECT_PROMPT" not in jsonl_bytes(rows)


def test_reference_perturbation_cannot_change_prompt() -> None:
    tokenizer = _Tokenizer()
    first = prepare_rlvr_rows(_sources(), split="train", tokenizer=tokenizer)
    sources = _sources()
    sources[0].task_object.ds.rows[0]["label"] = "C"
    second = prepare_rlvr_rows(sources, split="train", tokenizer=tokenizer)
    assert first[0].prompt_token_ids == second[0].prompt_token_ids
    assert first[0].oracle != second[0].oracle
    assert first[0].row_id == second[0].row_id
    assert (
        hashlib.sha256(jsonl_bytes(first)).digest() != hashlib.sha256(jsonl_bytes(second)).digest()
    )


def test_unparseable_gsm_reference_fails_preparation() -> None:
    with pytest.raises(RLVRDataError, match="unparseable reference"):
        prepare_rlvr_rows(_sources(bad_gsm=True), split="train", tokenizer=_Tokenizer())


def test_qualification_cap_does_not_skip_full_source_admission() -> None:
    with pytest.raises(RLVRDataError, match="unparseable reference"):
        prepare_rlvr_rows(
            _sources(bad_gsm=True),
            split="train",
            tokenizer=_Tokenizer(),
            caps={"ARC-Easy": 1, "ARC-Challenge": 1, "MMLU": 1, "GSM8K": 1},
        )


def test_shared_heldout_filter_normalizes_and_preserves_duplicate_order() -> None:
    train = _Task(
        [
            {"question": " KEEP ", "label": "A"},
            {"question": "ＴｅＳｔ  Question", "label": "A"},
            {"question": " KEEP ", "label": "B"},
        ]
    )
    heldout = _Task([{"question": "test question", "label": "A"}])
    exclude_heldout_questions(cast(Any, [train]), cast(Any, [heldout]))
    assert train.ds["question"] == [" KEEP ", " KEEP "]


@pytest.mark.parametrize(
    "labels", [("A", "B", "C"), ("A", "B", "C", "D"), ("1", "2", "3", "4", "5")]
)
def test_multiple_choice_exhaustive_valid_label_parity(labels) -> None:
    for answer in labels:
        row = _row("ARC-Easy", answer=answer, labels=labels)
        native = object.__new__(ARC)
        conversation = {
            "letters": list(labels),
            "messages": [{"role": "assistant", "content": answer}],
        }
        for completion in labels:
            assert grade_completion_result(row, completion).reward == float(
                native.evaluate(conversation, completion)
            )
        assert grade_completion_result(row, f" {answer} ").reward == 1.0
        assert grade_completion_result(row, f"{answer} because").reward == 0.0
        assert grade_completion_result(row, answer.lower()).reward == float(answer.lower() == answer)


@pytest.mark.parametrize(
    ("completion", "reward", "parsed", "valid"),
    [
        ("#### 10", 1.0, "10", True),
        ("first #### 10 then #### 11", 1.0, "10", True),
        ("#### 1,0", 1.0, "10", True),
        ("#### 10.0", 0.0, "10.0", True),
        ("#### +10", 0.0, None, False),
        ("####  10", 0.0, None, False),
        ("10", 0.0, None, False),
    ],
)
def test_gsm8k_native_formatting_semantics(completion, reward, parsed, valid) -> None:
    result = grade_completion_result(_row(), completion)
    assert (result.reward, result.parsed_answer, result.valid_format) == (
        reward,
        parsed,
        valid,
    )
    native = object.__new__(GSM8K)
    conversation = {
        "messages": [
            {
                "role": "assistant",
                "content": [{"type": "text", "text": "work #### 10"}],
            }
        ]
    }
    assert result.reward == float(native.evaluate(conversation, completion))


def test_mmlu_valid_label_parity() -> None:
    native = object.__new__(MMLU)
    conversation = {"messages": [{"role": "assistant", "content": "C"}]}
    for completion in MMLU.letters:
        row = _row("MMLU", answer="C", labels=MMLU.letters)
        assert grade_completion_result(row, completion).reward == float(native.evaluate(conversation, completion))


def test_jsonl_rejects_revision_or_row_identity_mutation(tmp_path) -> None:
    row = _row()
    path = tmp_path / "train.jsonl"
    path.write_bytes(jsonl_bytes([row]))
    assert load_rlvr_jsonl(path, expected_split="train") == (row,)
    value = row.to_dict()
    value["revision"] = "b" * 40
    path.write_text(json.dumps(value) + "\n")
    with pytest.raises(RLVRDataError, match="row_id disagrees"):
        load_rlvr_jsonl(path, expected_split="train")


@pytest.mark.parametrize(
    ("malformed", "error_type"),
    [
        (jsonl_bytes([_row()]).rstrip(b"\n"), RLVRDataError),
        (b"\xff\n", UnicodeDecodeError),
    ],
)
def test_jsonl_authenticated_payload_preserves_file_errors(
    tmp_path, malformed, error_type
) -> None:
    path = tmp_path / "train.jsonl"
    crlf = jsonl_bytes([_row()]).replace(b"\n", b"\r\n")
    path.write_bytes(crlf)
    assert load_rlvr_jsonl(path, expected_split="train", payload=crlf) == load_rlvr_jsonl(
        path, expected_split="train"
    )

    path.write_bytes(malformed)
    with pytest.raises(error_type) as file_error:
        load_rlvr_jsonl(path, expected_split="train")
    with pytest.raises(error_type) as payload_error:
        load_rlvr_jsonl(path, expected_split="train", payload=malformed)
    assert str(payload_error.value) == str(file_error.value)


class _Provider:
    def __init__(self, path, row, *, cache, base):
        path.write_text("cache manifest")
        self.path = path
        self.manifest_path = path
        self._row = row
        self.manifest: dict[str, Any] = {
            "schema_version": 5,
            "source_segment_mode": "turn_delimited_fixed_tokens",
            "turn_contract": {"contract": "x"},
            "metadata": {
                "rlvr_data_sha256": cache["data_sha256"],
                "rlvr_base_model_sha256": base["model_sha256"],
                "rlvr_base_metadata_sha256": base["metadata_sha256"],
            },
            "shards": [{"file": "contexts_00000.safetensors.zst", "sha256": "c" * 64}],
        }

    def __len__(self):
        return 1

    def get(self, token_ids):
        return cast(Any, (object(),)) if tuple(token_ids) == self._row.prompt_token_ids else None


def test_prompt_cache_admission_binds_cache_and_base_authority(tmp_path) -> None:
    row = _row()
    turn_hash = hashlib.sha256(canonical_turn_payload({"contract": "x"})).hexdigest()
    base = {"model_sha256": "a" * 64, "metadata_sha256": "b" * 64}
    cache = {
        "turn_context_contract_sha256": turn_hash,
        "data_sha256": "d" * 64,
        "shards": [{"file": "contexts_00000.safetensors.zst", "sha256": "c" * 64}],
    }
    provider = _Provider(tmp_path / "cache.json", row, cache=cache, base=base)
    cache["manifest_sha256"] = hashlib.sha256(provider.path.read_bytes()).hexdigest()
    bundle = RLVRBundle(
        tmp_path,
        (row,),
        (),
        {"context_cache": cache, "base_checkpoint": base},
    )
    assert admit_rlvr_prompt_cache(bundle, cast(Any, provider)) is provider
    provider.manifest["metadata"]["rlvr_base_model_sha256"] = "f" * 64
    with pytest.raises(RLVRDataError, match="schema-5 turn cache"):
        admit_rlvr_prompt_cache(bundle, cast(Any, provider))


@pytest.mark.parametrize("population", ["exact", "missing", "extra", "substituted"])
def test_prompt_cache_admission_uses_real_schema5_provider(tmp_path, population) -> None:
    meta, _, _ = checkpoint_fixture(tmp_path)
    contract = resolve(meta)
    identity = cache_identity_from_config({}, turn_contract=contract)
    row = _row()
    other = _row(source_index=1, prompt=(1, 2, 100))
    bundle_rows = (row, other) if population == "missing" else (row,)
    cached_rows = (row, other) if population == "extra" else (
        (other,) if population == "substituted" else (row,)
    )
    data_sha = "d" * 64
    base = {"model_sha256": "a" * 64, "metadata_sha256": "b" * 64}
    cache_dir = tmp_path / "context_cache" / "c2048" / identity.feature_selection_hash[:16]
    manifest_path = save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                cached_row.prompt_token_ids,
                [ContextFeatureSegment((2,), (1,), 0, 37)],
                segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            )
            for cached_row in cached_rows
        ],
        identity=identity,
        n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(4),
        metadata={
            "rlvr_data_sha256": data_sha,
            "rlvr_base_model_sha256": base["model_sha256"],
            "rlvr_base_metadata_sha256": base["metadata_sha256"],
        },
        source_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        turn_contract=contract,
    )
    provider = SafetensorsContextFeatureProvider(
        cache_dir,
        identity=identity,
        expected_source_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        expected_turn_contract=contract,
    )
    cache = {
        "turn_context_contract_sha256": turn_context_contract_hash(contract),
        "data_sha256": data_sha,
        "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "shards": [
            {"file": shard["file"], "sha256": shard["sha256"]}
            for shard in provider.manifest["shards"]
        ],
    }
    bundle = RLVRBundle(
        tmp_path,
        bundle_rows,
        (),
        {"context_cache": cache, "base_checkpoint": base},
    )
    if population == "exact":
        assert admit_rlvr_prompt_cache(bundle, provider) is provider
    else:
        message = (
            f"missing prepared row {row.row_id}"
            if population == "substituted"
            else "exactly the bundle's distinct prompt population"
        )
        with pytest.raises(RLVRDataError, match=message):
            admit_rlvr_prompt_cache(bundle, provider)


@pytest.mark.parametrize(
    "failure",
    [None, "base_hash", "source_counts", "source_revision", "context_length",
     "prepared_body", "preparation_hash", "turn_contract", "row_order"],
)
def test_finalizer_authenticates_real_schema5_provider(tmp_path, monkeypatch, failure) -> None:
    _meta, meta_path, stats_path = checkpoint_fixture(tmp_path)
    contract = resolve(meta_path)
    identity = cache_identity_from_config({}, turn_contract=contract)
    output = tmp_path / "bundle"
    tokenizer = ChatTokenizer()
    extraction_tokenizer = _DifferentSourceTokenizer()
    model_path = tmp_path / "model.pt"
    model_path.write_bytes(b"immutable base")
    model_sha256 = hashlib.sha256(model_path.read_bytes()).hexdigest()
    train_sources = _sources()
    validation_sources = _sources()
    expected_counts = {source.task: 1 for source in train_sources}
    cfg = OmegaConf.create(
        {
            "output_dir": str(output),
            "checkpoint_meta_path": str(meta_path),
            "base_checkpoint_path": str(model_path),
            "base_checkpoint_sha256": model_sha256,
            "stats_path": str(stats_path),
            "model_max_seq_len": 2048,
            "revisions": dict(RLVR_DATASET_REVISIONS),
            "expected_train_counts": expected_counts,
            "expected_validation_counts": expected_counts,
            "expected_removed_counts": dict.fromkeys(expected_counts, 0),
            "train_caps": None,
            "validation_caps": None,
        }
    )
    monkeypatch.setattr(prepare_script, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        prepare_script,
        "turn_contract_from_config",
        lambda _cfg, _tokenizer: (contract, extraction_tokenizer),
    )
    monkeypatch.setattr(
        prepare_script,
        "load_pinned_sources",
        lambda: (train_sources, validation_sources, dict.fromkeys(expected_counts, 0)),
    )
    if failure in {"base_hash", "source_counts", "source_revision", "context_length"}:
        if failure == "base_hash":
            cfg.base_checkpoint_sha256 = "f" * 64
            message = "base checkpoint SHA-256 mismatch"
        elif failure == "source_counts":
            cfg.expected_train_counts["MMLU"] += 1
            message = "retained train counts disagree"
        elif failure == "source_revision":
            cfg.revisions.arc = "f" * 40
            message = "configured dataset revisions disagree"
        else:
            cfg.model_max_seq_len = 1024
            message = "configured model_max_seq_len=1024 disagrees"
        with pytest.raises(ValueError, match=message):
            prepare_script.prepare_data_from_config(cfg)
        assert not output.exists()
        return
    preparation_path = prepare_script.prepare_data_from_config(cfg)
    preparation_bytes = preparation_path.read_bytes()
    with pytest.raises(FileExistsError, match="preparation or bundle already exists"):
        prepare_script.prepare_data_from_config(cfg)
    assert preparation_path.read_bytes() == preparation_bytes
    preparation = json.loads(preparation_path.read_text())
    train = load_rlvr_jsonl(output / "train.jsonl", expected_split="train")
    validation = load_rlvr_jsonl(output / "validation.jsonl", expected_split="validation")
    cache_dir = output / preparation["context_cache"]["path"]
    unique_rows = {row.prompt_token_ids: row for row in (*train, *validation)}.values()
    cache_manifest = save_context_cache(
        cache_dir,
        [
            ContextCacheEntry.from_sequences(
                row.prompt_token_ids,
                [ContextFeatureSegment((2,), (1,), 0, 37)],
                segment_mode=SegmentMode.turn_delimited_fixed_tokens,
            )
            for row in unique_rows
        ],
        identity=identity,
        n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(4),
        metadata={
            "rlvr_data_sha256": preparation["bundle_sha256"],
            "rlvr_base_model_sha256": model_sha256,
            "rlvr_base_metadata_sha256": contract.provenance["checkpoint_metadata_sha256"],
        },
        source_segment_mode=SegmentMode.turn_delimited_fixed_tokens,
        turn_contract=contract,
    )
    train_path = output / "train.jsonl"
    train_bytes = train_path.read_bytes()
    if failure is not None:
        if failure == "prepared_body":
            train_path.write_bytes(train_bytes + b"\n")
            message = "prepared train artifact changed before finalization"
        elif failure == "preparation_hash":
            preparation["preparation"]["prepared_train_counts"]["MMLU"] += 1
            preparation_path.write_text(json.dumps(preparation))
            message = "preparation manifest checksum disagrees"
        else:
            key = "turn_context_contract_sha256" if failure == "turn_contract" else "ordered_row_ids_sha256"
            preparation["context_cache"][key] = "f" * 64
            preparation["bundle_sha256"] = bundle_manifest_sha256(preparation)
            preparation_path.write_text(json.dumps(preparation))
            # Keep the cache's producer-data pin valid so population order is
            # checked independently of metadata authentication.
            cache_value = json.loads(cache_manifest.read_text())
            cache_value["metadata"]["rlvr_data_sha256"] = preparation["bundle_sha256"]
            cache_manifest.write_text(json.dumps(cache_value))
            message = (
                "preparation turn contract disagrees during finalization"
                if failure == "turn_contract"
                else "ordered row population changed before finalization"
            )
        with pytest.raises(ValueError, match=message):
            prepare_script.finalize_from_config(cfg)
        assert not (output / "manifest.json").exists()
        return
    replacement_bytes = jsonl_bytes((*train[1:], train[0]))
    replacement_path = output / "replacement.jsonl"
    replacement_state = {"done": False}
    original_read_bytes = Path.read_bytes
    original_sha256_file = prepare_script._sha256_file

    def replace_train_once() -> None:
        if replacement_state["done"]:
            return
        replacement_path.write_bytes(replacement_bytes)
        replacement_path.replace(train_path)
        replacement_state["done"] = True

    def read_bytes_then_replace(path: Path) -> bytes:
        payload = original_read_bytes(path)
        if path == train_path:
            replace_train_once()
        return payload

    def hash_then_replace(path: Path) -> str:
        digest = original_sha256_file(path)
        if path == train_path:
            replace_train_once()
        return digest

    monkeypatch.setattr(Path, "read_bytes", read_bytes_then_replace)
    monkeypatch.setattr(prepare_script, "_sha256_file", hash_then_replace)
    final_path = prepare_script.finalize_from_config(cfg)
    final_manifest_bytes = final_path.read_bytes()
    with pytest.raises(FileExistsError, match="bundle already committed"):
        prepare_script.finalize_from_config(cfg)
    assert final_path.read_bytes() == final_manifest_bytes
    assert replacement_state["done"]
    train_path.write_bytes(train_bytes)
    final = json.loads(final_path.read_text())
    cache_manifest_value = json.loads(cache_manifest.read_text())
    expected_final = {
        **preparation,
        "context_cache": {
            **preparation["context_cache"],
            "data_sha256": preparation["bundle_sha256"],
            "manifest_sha256": hashlib.sha256(cache_manifest.read_bytes()).hexdigest(),
            "shards": [
                {"file": shard["file"], "sha256": shard["sha256"]}
                for shard in cache_manifest_value["shards"]
            ],
        },
    }
    expected_final["bundle_sha256"] = bundle_manifest_sha256(expected_final)
    assert final == expected_final
    replacement_state["done"] = False
    bundle = load_rlvr_bundle(
        output,
        tokenizer=tokenizer,
        expected_base_model_sha256=model_sha256,
        expected_turn_context_contract=contract,
    )
    assert replacement_state["done"]
    assert bundle.train == train
    train_path.write_bytes(train_bytes)
    provider = bundle.open_prompt_cache(expected_turn_context_contract=contract)
    assert bundle.identity_sha256 == final["bundle_sha256"]
    assert len(provider) == len({row.prompt_token_ids for row in bundle.rows})
    assert all(provider.get(row.prompt_token_ids) is not None for row in bundle.rows)

    final_bytes = final_path.read_bytes()
    final["base_checkpoint"]["model_sha256"] = "f" * 64
    final_path.write_text(json.dumps(final) + "\n")
    with pytest.raises(RLVRDataError, match="bundle_sha256 disagrees"):
        load_rlvr_bundle(
            output,
            tokenizer=tokenizer,
            expected_base_model_sha256=model_sha256,
            expected_turn_context_contract=contract,
        )
    final_path.write_bytes(final_bytes)

    train_path.write_bytes(train_bytes + b"\n")
    with pytest.raises(RLVRDataError, match="train JSONL digest disagrees"):
        load_rlvr_bundle(
            output,
            tokenizer=tokenizer,
            expected_base_model_sha256=model_sha256,
            expected_turn_context_contract=contract,
        )
    train_path.write_bytes(train_bytes)

    raw_cache_manifest = json.loads(cache_manifest.read_text())
    raw_cache_manifest["metadata"]["mutation"] = True
    cache_manifest.write_text(json.dumps(raw_cache_manifest) + "\n")
    with pytest.raises(RLVRDataError, match="bundle's schema-5 turn cache"):
        bundle.open_prompt_cache(expected_turn_context_contract=contract)
