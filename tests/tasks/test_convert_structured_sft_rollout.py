"""Artifact and resume tests for structured_nanochat_rollout_v1 conversion."""

from __future__ import annotations

import copy
import json
from unittest.mock import patch

import pyarrow.parquet as pq
import pytest

from errors import DataLoadError
from nanochat.tokenizer import RustBPETokenizer
from scripts.preprocess.convert_sft_to_parquet import (
    convert_mixture_to_structured_rollout,
)
from steering.sft.structured_dataset import (
    StructuredEnvironmentCursor,
    load_structured_rollout_artifact,
)
from steering.sft.structured_rollout import (
    STRUCTURED_ROLLOUT_ARROW_SCHEMA,
    StructuredRolloutError,
    build_conversation_environment,
    structured_table_to_rows,
)
from tasks.common import TaskMixture
from tasks.steered import SteeredConversations, SteeredTaskMixture


class _Task:
    def __init__(self, conversations: list[dict]) -> None:
        self.conversations = conversations

    def __len__(self) -> int:
        return len(self.conversations)

    def __getitem__(self, index: int) -> dict:
        return self.conversations[index]


class _Tokenizer:
    _specials = {
        "<bos>": 1,
        "<user_start>": 2,
        "<user_end>": 3,
        "<assistant_start>": 4,
        "<assistant_end>": 5,
        "<python_start>": 6,
        "<python_end>": 7,
        "<output_start>": 8,
        "<output_end>": 9,
    }
    _decoded_specials = {value: key for key, value in _specials.items()}
    structured_tokenizer_sha256 = "a" * 64
    render_structured_conversation = RustBPETokenizer.render_structured_conversation

    def get_bos_token_id(self) -> int:
        return 1

    def encode_special(self, text: str) -> int:
        return self._specials[text]

    def encode(self, text: str) -> list[int]:
        return [1000 + ord(character) for character in text]

    def decode(self, token_ids) -> str:
        return "".join(
            self._decoded_specials[token_id]
            if token_id in self._decoded_specials
            else chr(token_id - 1000)
            for token_id in token_ids
        )


def _complete(user: str, assistant: str) -> dict:
    return {
        "messages": [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant},
        ]
    }


def _mixture(conversations: list[dict]) -> TaskMixture:
    return TaskMixture([_Task(conversations)])


def _convert(mixture: TaskMixture, output_dir, **kwargs) -> None:
    with patch(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
        return_value=_Tokenizer(),
    ):
        convert_mixture_to_structured_rollout(
            mixture,
            output_dir,
            mixture_name="test",
            **kwargs,
        )


def _write_steered_jsonl(path, rows: int) -> str:
    path.write_text(
        "".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt {index}"},
                    {"role": "assistant", "content": f"steered answer {index}"},
                ]
            )
            + "\n"
            for index in range(rows)
        ),
        encoding="utf-8",
    )
    return str(path)


def test_structured_conversion_accepts_a_steered_mixture_without_stamping_steered_rows(
    tmp_path,
) -> None:
    """A ``SteeredTaskMixture`` converts; ``steered_rows`` is legacy-only.

    ``SFTConversationSource`` is a presence-only Protocol over
    ``__len__``/``__getitem__``, which ``SteeredTaskMixture`` satisfies through
    ``Task``. The steered row count is deliberately NOT part of the structured
    metadata schema, so this pins both halves of that claim.
    """
    steered = SteeredConversations(_write_steered_jsonl(tmp_path / "s.jsonl", 2))
    mixture = SteeredTaskMixture([_Task([_complete("u", "a")] * 4)], steered)
    output_dir = tmp_path / "structured"

    _convert(mixture, output_dir, conversations_per_shard=100)

    meta = json.loads((output_dir / "conversion_meta.json").read_text())
    assert "steered_rows" not in meta
    assert meta["format"] == "structured_nanochat_rollout_v1"
    assert meta["total_conversations"] == len(mixture) == 6
    # The identity recorded is the STEERED interleaving, not a plain mixture.
    assert meta["mixture_fingerprint"]["tasks"][-1] == {
        "class_name": "SteeredConversations",
        "length": 2,
    }
    rows = structured_table_to_rows(
        pq.read_table(output_dir / "shard_00000.parquet"), decode=_Tokenizer().decode
    )
    assert len(rows) == 6


def test_structured_conversion_writes_exact_schema_metadata_and_empty_shard(
    tmp_path,
) -> None:
    mixture = _mixture(
        [
            _complete("u", "a"),
            {"messages": [{"role": "user", "content": "trailing"}]},
        ]
    )
    _convert(mixture, tmp_path, conversations_per_shard=1)

    tables = [
        pq.read_table(tmp_path / f"shard_{shard_index:05d}.parquet")
        for shard_index in range(2)
    ]
    assert all(table.schema == STRUCTURED_ROLLOUT_ARROW_SCHEMA for table in tables)
    assert sorted(table.num_rows for table in tables) == [0, 1]
    populated = next(table for table in tables if table.num_rows == 1)
    rows = structured_table_to_rows(populated, decode=_Tokenizer().decode)
    assert len(rows) == 1
    assert rows[0].original_index in {0, 1}
    assert rows[0].text == _Tokenizer().decode(rows[0].target_token_ids)

    metadata = json.loads((tmp_path / "conversion_meta.json").read_text())
    assert metadata["format"] == "structured_nanochat_rollout_v1"
    assert metadata["schema_version"] == 1
    assert metadata["renderer_version"] == 1
    assert metadata["tokenizer_sha256"] == "a" * 64
    assert metadata["completed_shards"] == 2
    assert metadata["total_rows"] == 1
    assert metadata["skip_counters"] == {"no_completed_transaction": 1}
    assert len(metadata["shards"]) == 2
    assert all("row_content_sha256" in record for record in metadata["shards"])


def test_resume_rerenders_every_source_row_and_is_idempotent(tmp_path) -> None:
    conversations = [_complete("u0", "a0"), _complete("u1", "a1")]
    mixture = _mixture(conversations)
    _convert(mixture, tmp_path, conversations_per_shard=1)
    before = {
        path.name: path.read_bytes()
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    }

    _convert(mixture, tmp_path, conversations_per_shard=1)
    after = {
        path.name: path.read_bytes()
        for path in sorted(tmp_path.iterdir())
        if path.is_file()
    }
    assert after == before

    drifted = copy.deepcopy(conversations)
    drifted[0]["messages"][1]["content"] = "changed"
    with pytest.raises(DataLoadError, match="row drift|provenance drift"):
        _convert(_mixture(drifted), tmp_path, conversations_per_shard=1)


def test_structured_consumer_rejects_legacy_namespace_immediately(tmp_path) -> None:
    legacy_meta = {
        "mixture": "test",
        "split": "train",
        "format": "full_nanochat_text",
        "completed_shards": 0,
        "total_shards": 1,
        "total_conversations": 1,
    }
    (tmp_path / "conversion_meta.json").write_text(json.dumps(legacy_meta))
    with pytest.raises(DataLoadError, match="format"):
        _convert(_mixture([_complete("u", "a")]), tmp_path)


def test_atomic_parquet_failure_never_advertises_a_shard(tmp_path) -> None:
    mixture = _mixture([_complete("u", "a")])
    with (
        patch(
            "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
            return_value=_Tokenizer(),
        ),
        patch(
            "scripts.preprocess.convert_sft_to_parquet.atomic_write_parquet",
            side_effect=RuntimeError("crash"),
        ),
        pytest.raises(RuntimeError, match="crash"),
    ):
        convert_mixture_to_structured_rollout(
            mixture,
            tmp_path,
            mixture_name="test",
        )
    assert not (tmp_path / "shard_00000.parquet").exists()
    assert not (tmp_path / "conversion_meta.json").exists()


def test_strict_loader_audits_content_and_rejects_legacy(tmp_path) -> None:
    mixture = _mixture([_complete("u0", "a0"), _complete("u1", "a1")])
    _convert(mixture, tmp_path, conversations_per_shard=1)
    artifact = load_structured_rollout_artifact(
        tmp_path,
        decode=_Tokenizer().decode,
        tokenizer_sha256="a" * 64,
        expected_mixture="test",
        expected_split="train",
    )
    assert len(artifact.rows) == 2
    assert len(artifact.identity_sha256) == 64

    metadata_path = tmp_path / "conversion_meta.json"
    metadata = json.loads(metadata_path.read_text())
    metadata["format"] = "full_nanochat_text"
    metadata_path.write_text(json.dumps(metadata))
    with pytest.raises(StructuredRolloutError, match="structured rollout consumer"):
        load_structured_rollout_artifact(
            tmp_path,
            decode=_Tokenizer().decode,
            tokenizer_sha256="a" * 64,
        )


def test_rank_cursor_is_transactional_and_resumes_ordered_lookahead(tmp_path) -> None:
    mixture = _mixture([_complete(f"u{i}", f"a{i}") for i in range(8)])
    _convert(mixture, tmp_path, conversations_per_shard=3)
    tokenizer = _Tokenizer()
    artifact = load_structured_rollout_artifact(
        tmp_path,
        decode=tokenizer.decode,
        tokenizer_sha256="a" * 64,
    )
    geometry = {"context_length": 128, "segment_length": 4, "group_size": 4}

    def build(row):
        return build_conversation_environment(
            row,
            bos_token_id=tokenizer.get_bos_token_id(),
            context_length=128,
            segment_length=4,
            min_reward_tokens=1,
        )

    cursor = StructuredEnvironmentCursor(
        artifact,
        rank=0,
        world_size=2,
        geometry_identity=geometry,
    )
    staged = cursor.stage(1, prefetch=1, build=build)
    repeated = cursor.stage(1, prefetch=1, build=build)
    assert repeated.environments == staged.environments
    assert cursor.state.attempted == 0
    assert len(staged.next_state.buffered_original_indices) == 1

    cursor.commit(staged)
    assert cursor.state.accepted == 1
    with pytest.raises(StructuredRolloutError, match="stale|already"):
        cursor.commit(staged)
    resumed = StructuredEnvironmentCursor(
        artifact,
        rank=0,
        world_size=2,
        geometry_identity=geometry,
        state=cursor.state.to_json_dict(),
    )
    next_batch = resumed.stage(1, prefetch=0, build=build)
    assert next_batch.environments[0].original_index == (
        cursor.state.buffered_original_indices[0]
    )
    with pytest.raises(StructuredRolloutError, match="world_size"):
        StructuredEnvironmentCursor(
            artifact,
            rank=0,
            world_size=1,
            geometry_identity=geometry,
            state=cursor.state.to_json_dict(),
        )
