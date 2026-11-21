"""Tests for the full-sequence SFT parquet format."""

from pathlib import Path
from unittest.mock import patch

import pyarrow.parquet as pq

from scripts.preprocess.convert_sft_to_parquet import convert_mixture_to_parquet


class FakeMixture:
    """Mixture of conversations for testing."""

    def __init__(self, conversations):
        self.conversations = conversations

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):
        return self.conversations[idx]


class FakeTokenizer:
    _specials = {
        "<user_start>": 100_001,
        "<assistant_start>": 100_002,
        "<output_start>": 100_003,
        "<output_end>": 100_004,
    }

    def render_conversation(self, conversation):
        ids = [101]
        for message in conversation["messages"]:
            ids.append(self._specials[f"<{message['role']}_start>"])
            ids.extend(ord(ch) for ch in str(message["content"]))
        return ids, [0] * len(ids)

    def get_bos_token_id(self):
        return 101

    def encode_special(self, name):
        return self._specials[name]

    def decode(self, ids):
        decoded_specials = {
            self._specials["<user_start>"]: "<user>",
            self._specials["<assistant_start>"]: "<assistant>",
            self._specials["<output_start>"]: "<output>",
            self._specials["<output_end>"]: "</output>",
        }
        return "".join(decoded_specials.get(token_id, chr(token_id)) for token_id in ids)


def _convert(mixture, output_dir: Path, **kwargs) -> None:
    with patch(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
        return_value=FakeTokenizer(),
    ):
        convert_mixture_to_parquet(mixture, output_dir, **kwargs)


class TestSftParquetFormat:
    def test_parquet_has_only_full_sequence_columns(self, tmp_path):
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "What is 2+2?"},
                    {"role": "assistant", "content": "4"},
                ]
            },
        ]
        mixture = FakeMixture(convs)

        _convert(mixture, tmp_path, mixture_name="test")

        table = pq.read_table(tmp_path / "shard_00000.parquet")
        assert table.column_names == ["text", "original_index", "turn_char_starts"]
        assert "messages" not in table.column_names
        assert "response" not in table.column_names
        assert "assistant_char_offset" not in table.column_names
        assert "assistant_loss_mask" not in table.column_names
        assert table.column("original_index").to_pylist() == [0]
        assert table.column("text").to_pylist() == ["<user>What is 2+2?<assistant>4"]

    def test_multi_turn_text_is_rendered_as_single_training_sequence(self, tmp_path):
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello"},
                    {"role": "user", "content": "What?"},
                    {"role": "assistant", "content": "I said hello"},
                ]
            },
        ]
        mixture = FakeMixture(convs)

        _convert(mixture, tmp_path, mixture_name="test")

        text = pq.read_table(tmp_path / "shard_00000.parquet").column("text").to_pylist()[0]
        assert text == "<user>Hi<assistant>Hello<user>What?<assistant>I said hello"

    def test_trailing_user_message_is_truncated_before_rendering(self, tmp_path):
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Hi"},
                    {"role": "assistant", "content": "Hello!"},
                    {"role": "user", "content": "Thanks"},
                ]
            },
        ]
        mixture = FakeMixture(convs)

        _convert(mixture, tmp_path, mixture_name="test")

        text = pq.read_table(tmp_path / "shard_00000.parquet").column("text").to_pylist()[0]
        assert text == "<user>Hi<assistant>Hello!"

    def test_original_index_tracks_skipped_conversations(self, tmp_path):
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Q0"},
                    {"role": "assistant", "content": "A0"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": ""},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
        ]
        mixture = FakeMixture(convs)

        _convert(mixture, tmp_path, mixture_name="test")

        table = pq.read_table(tmp_path / "shard_00000.parquet")
        assert table.column("original_index").to_pylist() == [0, 2]
        assert table.column("text").to_pylist() == [
            "<user>Q0<assistant>A0",
            "<user>Q2<assistant>A2",
        ]

    def test_no_assistant_message_skipped(self, tmp_path):
        convs = [
            {
                "messages": [
                    {"role": "user", "content": "Q0"},
                    {"role": "assistant", "content": "A0"},
                ]
            },
            {"messages": [{"role": "user", "content": "Q1"}]},
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
        ]
        mixture = FakeMixture(convs)

        _convert(mixture, tmp_path, mixture_name="test")

        table = pq.read_table(tmp_path / "shard_00000.parquet")
        assert table.column("original_index").to_pylist() == [0, 2]
