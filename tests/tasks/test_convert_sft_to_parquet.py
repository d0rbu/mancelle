"""Tests for SFT to parquet conversion.

Run with:
    uv run pytest tests/test_convert_sft_to_parquet.py -v
"""

import json
import math
from unittest.mock import patch

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

from errors import DataLoadError
from scripts.preprocess.convert_sft_to_parquet import (
    convert_mixture_to_parquet,
    main,
    read_conversion_status,
    render_messages_to_full_nanochat_row,
    render_messages_to_full_nanochat_text,
)
from steering.segmentation import token_to_char_offsets
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft.turn_spans import ChatSpecialTokens, legacy_turn_spans
from steering.sft_schema import SFTSplit
from tasks.common import TaskMixture
from tasks.steered import SteeredConversations, SteeredTaskMixture
from tests.conftest import get_tokenizer_or_skip

# ---------------------------------------------------------------------------
# Fakes and helpers
# ---------------------------------------------------------------------------


class FakeTask:
    """Minimal Task implementation for testing without HuggingFace downloads."""

    def __init__(self, conversations):
        self.conversations = conversations
        self.start = 0
        self.stop = None
        self.step = 1

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, index):
        return self.conversations[index]

    def num_examples(self):
        return len(self.conversations)

    def get_example(self, index):
        return self.conversations[index]


class FakeTokenizer:
    """Tokenizer stub for full_nanochat conversion tests."""

    _user_text = "<user_start>Question<user_end>"
    _full_text = _user_text + "<assistant_start>Answer"

    def render_conversation(self, conversation):
        assert conversation["messages"][0]["role"] == "user"
        return [101, 102, 103], [0, 0, 1]

    def get_bos_token_id(self):
        return 101

    def encode_special(self, name):
        return {
            "<user_start>": 102,
            "<assistant_start>": 103,
            "<output_start>": 104,
            "<output_end>": 105,
        }[name]

    def decode(self, ids):
        return {
            (102,): self._user_text,
            (102, 103): self._full_text,
        }[tuple(ids)]


class _NoBosTokenizer(FakeTokenizer):
    def render_conversation(self, conversation):
        ids, mask = super().render_conversation(conversation)
        return ids[1:], mask[1:]


def _fake_conversations(n=20):
    """Generate n fake conversations."""
    return [
        {
            "messages": [
                {"role": "user", "content": f"Question {i}"},
                {"role": "assistant", "content": f"Answer {i}"},
            ]
        }
        for i in range(n)
    ]


def _make_fake_mixture(n=20):
    """Build a TaskMixture from fake conversations."""
    convos = _fake_conversations(n)
    task = FakeTask(convos)
    return TaskMixture([task])


def _convert_with_fake_tokenizer(*args, **kwargs):
    with patch(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
        return_value=FakeTokenizer(),
    ):
        return convert_mixture_to_parquet(*args, **kwargs)


_BASELINE_TRAIN_META = {
    "mixture": "standard",
    "format": "full_nanochat_text",
    "completed_shards": 2,
    "total_shards": 2,
    "total_conversations": 3,
}


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestConvertMixtureToParquet:
    """Core conversion tests."""

    def test_render_helpers_reject_missing_bos_identically(self):
        tokenizer = _NoBosTokenizer()
        messages = _fake_conversations(1)[0]["messages"]
        with pytest.raises(
            ValueError,
            match="BOS-inclusive token stream.*producer contract",
        ) as text_error:
            render_messages_to_full_nanochat_text(tokenizer, messages)
        with pytest.raises(
            ValueError,
            match="BOS-inclusive token stream.*producer contract",
        ) as row_error:
            render_messages_to_full_nanochat_row(tokenizer, messages)
        assert str(text_error.value) == str(row_error.value)

    def test_render_helpers_decode_same_text_with_one_render_each(self):
        tokenizer = FakeTokenizer()
        messages = _fake_conversations(1)[0]["messages"]
        with patch.object(
            tokenizer,
            "render_conversation",
            wraps=tokenizer.render_conversation,
        ) as render_conversation:
            text_only = render_messages_to_full_nanochat_text(tokenizer, messages)
            assert render_conversation.call_count == 1
            render_conversation.reset_mock()
            row_text, _ = render_messages_to_full_nanochat_row(tokenizer, messages)
            assert render_conversation.call_count == 1
        assert text_only == row_text

    def test_parquet_has_full_text_column(self, tmp_path):
        """Parquet files store full nanochat-rendered text."""
        mixture = _make_fake_mixture(n=10)

        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=100)

        parquet_files = sorted(tmp_path.glob("*.parquet"))
        assert len(parquet_files) >= 1

        table = pq.read_table(parquet_files[0])
        assert table.column_names == ["text", "original_index", "turn_char_starts"]
        assert (
            table.column("text").to_pylist()
            == ["<user_start>Question<user_end><assistant_start>Answer"] * 10
        )
        assert (
            table.column("turn_char_starts").to_pylist()
            == [[0, len(FakeTokenizer._user_text)]] * 10
        )
        assert table.schema.field("turn_char_starts").type == pa.list_(pa.int32())

    def test_train_conversion_matches_pre_split_axis_golden(self, tmp_path):
        """Preserve legacy text/index content while adding turn starts."""
        mixture = _make_fake_mixture(n=3)

        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=2,
            mixture_name="standard",
        )

        tables = [pq.read_table(path) for path in sorted(tmp_path.glob("shard_*.parquet"))]
        assert [table.column("text").to_pylist() for table in tables] == [
            [FakeTokenizer._full_text, FakeTokenizer._full_text],
            [FakeTokenizer._full_text],
        ]
        assert [table.column("original_index").to_pylist() for table in tables] == [
            [0, 1],
            [2],
        ]
        assert [table.column("turn_char_starts").to_pylist() for table in tables] == [
            [[0, len(FakeTokenizer._user_text)]] * 2,
            [[0, len(FakeTokenizer._user_text)]],
        ]
        meta = json.loads((tmp_path / "conversion_meta.json").read_text())
        assert meta.pop("split") == "train"
        assert meta.pop("mixture_fingerprint") == compute_mixture_fingerprint(
            mixture,
            mixture_name="standard",
            split=SFTSplit.TRAIN,
        )
        assert meta == {
            **_BASELINE_TRAIN_META,
            "max_conversations": None,
            "conversations_per_shard": 2,
            "written_conversations": 3,
            "skipped_conversations": 0,
        }

    def test_parquet_has_original_index(self, tmp_path):
        """Parquet files include original_index column tracking source indices."""
        mixture = _make_fake_mixture(n=5)

        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=100)

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        assert "original_index" in table.column_names
        indices = table.column("original_index").to_pylist()
        assert len(indices) == 5
        for idx in indices:
            assert isinstance(idx, int)
            assert 0 <= idx < 5

    def test_conversion_writes_full_nanochat_text_only(self, tmp_path):
        """SFT conversion stores pre-rendered text only."""
        mixture = _make_fake_mixture(n=2)

        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=100,
        )

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        assert table.column_names == ["text", "original_index", "turn_char_starts"]
        assert "messages" not in table.column_names
        assert "response" not in table.column_names
        assert "assistant_char_offset" not in table.column_names
        assert table.column("text").to_pylist() == [
            "<user_start>Question<user_end><assistant_start>Answer",
            "<user_start>Question<user_end><assistant_start>Answer",
        ]

        meta = json.loads((tmp_path / "conversion_meta.json").read_text())
        assert meta["format"] == "full_nanochat_text"

    def test_metadata_file_written(self, tmp_path):
        """conversion_meta.json exists with correct fields."""
        mixture = _make_fake_mixture(n=15)

        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=10,
            mixture_name="standard",
        )

        meta_path = tmp_path / "conversion_meta.json"
        assert meta_path.exists()

        meta = json.loads(meta_path.read_text())
        assert meta["mixture"] == "standard"
        assert meta["split"] == "train"
        assert meta["format"] == "full_nanochat_text"
        assert meta["total_conversations"] == 15
        # 15 conversations, 10 per shard => 2 shards
        assert meta["completed_shards"] == 2
        assert meta["total_shards"] == 2
        assert meta["mixture_fingerprint"] == compute_mixture_fingerprint(
            mixture,
            mixture_name="standard",
            split=SFTSplit.TRAIN,
        )

    def test_shard_count(self, tmp_path):
        """Correct number of shards based on conversations_per_shard."""
        n = 25
        per_shard = 10
        mixture = _make_fake_mixture(n=n)

        _convert_with_fake_tokenizer(mixture, tmp_path, conversations_per_shard=per_shard)

        parquet_files = sorted(tmp_path.glob("*.parquet"))
        expected_shards = math.ceil(n / per_shard)
        assert len(parquet_files) == expected_shards

    def test_max_conversations_limits_output(self, tmp_path):
        """max_conversations limits how many conversations are written."""
        mixture = _make_fake_mixture(n=50)

        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            max_conversations=15,
            conversations_per_shard=10,
        )

        # Read all parquet files and count total rows
        total_rows = 0
        for pf in tmp_path.glob("*.parquet"):
            table = pq.read_table(pf)
            total_rows += table.num_rows
        assert total_rows == 15

    def test_resumability(self, tmp_path):
        """Partial conversion can resume: only missing shards are recreated."""
        mixture = _make_fake_mixture(n=30)

        # First, do a full conversion with 10 per shard => 3 shards
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=10,
            mixture_name="standard",
        )

        parquet_files = sorted(tmp_path.glob("*.parquet"))
        assert len(parquet_files) == 3

        # Simulate partial state: remove last shard and update metadata
        last_shard = parquet_files[-1]
        last_shard.unlink()

        meta_path = tmp_path / "conversion_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["completed_shards"] = 2
        meta_path.write_text(json.dumps(meta))

        # Record modification times of the first two shards
        remaining = sorted(tmp_path.glob("*.parquet"))
        assert len(remaining) == 2
        mtimes_before = {f.name: f.stat().st_mtime for f in remaining}

        # Re-run conversion
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=10,
            mixture_name="standard",
        )

        # Should now have all 3 shards again
        parquet_files_after = sorted(tmp_path.glob("*.parquet"))
        assert len(parquet_files_after) == 3

        # First two shards should NOT have been rewritten
        for f in parquet_files_after[:2]:
            assert f.name in mtimes_before
            assert f.stat().st_mtime == mtimes_before[f.name]

        # Metadata should be complete
        meta = json.loads(meta_path.read_text())
        assert meta["completed_shards"] == 3

    @pytest.mark.parametrize(
        ("n_conversations", "total_shards", "conversations_per_shard"),
        [(1, 1, 100), (2, 2, 1)],
    )
    def test_resume_rejects_completed_shard_without_turn_boundary_column(
        self,
        tmp_path,
        n_conversations,
        total_shards,
        conversations_per_shard,
    ):
        mixture = _make_fake_mixture(n=n_conversations)
        pq.write_table(
            pa.table(
                {
                    "text": [FakeTokenizer._full_text],
                    "original_index": [0],
                }
            ),
            tmp_path / "shard_00000.parquet",
        )
        (tmp_path / "conversion_meta.json").write_text(
            json.dumps(
                {
                    "mixture": "standard",
                    "split": "train",
                    "format": "full_nanochat_text",
                    "completed_shards": 1,
                    "total_shards": total_shards,
                    "total_conversations": n_conversations,
                    "mixture_fingerprint": compute_mixture_fingerprint(
                        mixture,
                        mixture_name="standard",
                        split=SFTSplit.TRAIN,
                    ),
                }
            )
        )

        with pytest.raises(
            DataLoadError,
            match=(
                "predates the turn-boundary column.*must be regenerated into a fresh output_dir"
            ),
        ):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=conversations_per_shard,
                mixture_name="standard",
            )
        assert not (tmp_path / "shard_00001.parquet").exists()

    def test_partial_resume_validates_every_shard_below_the_cursor(self, tmp_path):
        mixture = _make_fake_mixture(n=3)
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=1,
            mixture_name="standard",
        )
        assert len(sorted(tmp_path.glob("*.parquet"))) == 3
        pq.write_table(
            pa.table({"text": ["x"], "original_index": [1]}),
            tmp_path / "shard_00001.parquet",
        )

        with pytest.raises(
            DataLoadError,
            match="shard_00001.*predates the turn-boundary column",
        ):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=1,
                mixture_name="standard",
            )

    def test_partial_resume_requires_every_shard_below_the_cursor(self, tmp_path):
        mixture = _make_fake_mixture(n=3)
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=1,
            mixture_name="standard",
        )
        (tmp_path / "shard_00001.parquet").unlink()

        with pytest.raises(
            DataLoadError,
            match="metadata records completed shard 1 but the file is missing",
        ):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=1,
                mixture_name="standard",
            )

    def test_complete_post_change_conversion_rerun_is_noop(self, tmp_path):
        mixture = _make_fake_mixture(n=1)
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=100,
            mixture_name="standard",
        )
        meta_path = tmp_path / "conversion_meta.json"
        shard_path = tmp_path / "shard_00000.parquet"
        mtime_before = shard_path.stat().st_mtime

        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=100,
            mixture_name="standard",
        )

        assert json.loads(meta_path.read_text())["completed_shards"] == 1
        assert shard_path.stat().st_mtime == mtime_before

    def test_resume_rejects_metadata_cursor_pointing_at_missing_file(self, tmp_path):
        mixture = _make_fake_mixture(n=2)
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=1,
            mixture_name="standard",
        )
        (tmp_path / "shard_00000.parquet").unlink()
        meta_path = tmp_path / "conversion_meta.json"
        meta = json.loads(meta_path.read_text())
        meta["completed_shards"] = 1
        meta_path.write_text(json.dumps(meta))

        with pytest.raises(
            DataLoadError,
            match=(
                "metadata records completed shard 0 but the file is missing; use a fresh output_dir"
            ),
        ):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=1,
                mixture_name="standard",
            )

    def test_real_tokenizer_conversion_writes_independent_turn_starts(self, tmp_path):
        tokenizer = get_tokenizer_or_skip()
        conversations = [
            {
                "messages": [
                    {"role": "user", "content": "plain question"},
                    {"role": "assistant", "content": "plain answer"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "first question"},
                    {"role": "assistant", "content": "first answer"},
                    {"role": "user", "content": "second question"},
                    {"role": "assistant", "content": "second answer"},
                ]
            },
            {
                "messages": [
                    {"role": "system", "content": "system instruction"},
                    {"role": "user", "content": "question"},
                    {"role": "assistant", "content": "answer"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "what is 🎉🔥 and 日本語テスト?"},
                    {"role": "assistant", "content": "café résumé 混合 mixed テキスト"},
                    {"role": "user", "content": "🔥🔥🔥"},
                    {"role": "assistant", "content": "混合"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "compute 日本語 🎉"},
                    {
                        "role": "assistant",
                        "content": [
                            {"type": "text", "text": "café"},
                            {"type": "python", "text": "1+1"},
                            {"type": "python_output", "text": "2"},
                            {"type": "text", "text": "混合 done"},
                        ],
                    },
                ]
            },
        ]
        mixture = TaskMixture([FakeTask(conversations)])

        with patch(
            "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
            return_value=tokenizer,
        ):
            convert_mixture_to_parquet(
                mixture,
                tmp_path,
                conversations_per_shard=100,
            )

        specials = ChatSpecialTokens.from_tokenizer(tokenizer)
        table = pq.read_table(tmp_path / "shard_00000.parquet")
        texts = table.column("text").to_pylist()
        written_starts = table.column("turn_char_starts").to_pylist()
        assert len(texts) == len(written_starts) == len(mixture)
        markers = {"user": "<user_start>", "assistant": "<assistant_start>"}
        expected = []
        for conversation_index in range(len(mixture)):
            conversation = mixture[conversation_index]
            ids, _ = tokenizer.render_conversation(conversation)
            spans = legacy_turn_spans(ids, specials)
            offsets = token_to_char_offsets(tokenizer, ids[1:])
            expected.append([offsets[span.start] for span in spans])
            # Independent oracle: every written start must index the same
            # decoded suffix, including marker-less post-tool assistant spans.
            text = texts[conversation_index]
            starts = written_starts[conversation_index]
            assert len(starts) == len(spans)
            assert starts[0] == 0
            for span, start in zip(spans, starts, strict=True):
                assert text[start:] == tokenizer.decode(ids[1 + span.start :]), (
                    conversation_index,
                    span,
                    text[start : start + 24],
                )
                if span.role in markers and text[start:].startswith("<"):
                    assert text[start:].startswith(markers[span.role])
        assert any(any(ord(char) > 127 for char in text) for text in texts)
        assert written_starts == expected

    def test_complete_legacy_conversion_cannot_be_upgraded_in_place(self, tmp_path):
        mixture = _make_fake_mixture(n=1)
        pq.write_table(
            pa.table(
                {
                    "text": ["legacy rendered row"],
                    "original_index": [0],
                }
            ),
            tmp_path / "shard_00000.parquet",
        )
        (tmp_path / "conversion_meta.json").write_text(
            json.dumps(
                {
                    "mixture": "standard",
                    "split": "train",
                    "format": "full_nanochat_text",
                    "completed_shards": 1,
                    "total_shards": 1,
                    "total_conversations": 1,
                }
            )
        )

        with pytest.raises(
            DataLoadError,
            match=(
                r"completed shards but no mixture_fingerprint.*cannot be upgraded "
                r"in place.*output_dir=/path/to/fresh/sft_data"
            ),
        ):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=100,
                mixture_name="standard",
            )

    def test_resume_rejects_existing_fingerprint_from_different_live_mixture(
        self,
        tmp_path,
    ):
        original = _make_fake_mixture(n=1)
        changed = _make_fake_mixture(n=2)
        _convert_with_fake_tokenizer(
            original,
            tmp_path,
            conversations_per_shard=100,
            mixture_name="standard",
        )

        with pytest.raises(
            DataLoadError,
            match=(
                r"mixture fingerprint that does not match the live TaskMixture.*"
                r"output_dir=/path/to/fresh/sft_data"
            ),
        ):
            _convert_with_fake_tokenizer(
                changed,
                tmp_path,
                conversations_per_shard=100,
                mixture_name="standard",
            )

    @pytest.mark.parametrize(
        "legacy_column",
        ["assistant_char_offset", "assistant_loss_mask"],
    )
    def test_existing_legacy_shard_without_meta_is_rejected(
        self,
        tmp_path,
        legacy_column,
    ):
        """Resume must not silently keep stale assistant-only SFT shards."""
        mixture = _make_fake_mixture(n=1)
        pq.write_table(
            pa.table(
                {
                    "text": ["legacy"],
                    "original_index": [0],
                    legacy_column: [0],
                }
            ),
            tmp_path / "shard_00000.parquet",
        )

        with pytest.raises(DataLoadError, match="not a full_nanochat SFT shard"):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=100,
            )

    def test_existing_split_mismatch_is_rejected(self, tmp_path):
        mixture = _make_fake_mixture(n=1)
        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=100,
            split=SFTSplit.TRAIN,
        )

        with pytest.raises(DataLoadError, match="uses split='train'.*split='val'"):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=100,
                split=SFTSplit.VAL,
            )

    def test_legacy_split_absent_metadata_is_accepted_for_train(self, tmp_path):
        mixture = _make_fake_mixture(n=1)
        legacy_meta = {
            "mixture": "standard",
            "format": "full_nanochat_text",
            "completed_shards": 0,
            "total_shards": 1,
            "total_conversations": 1,
        }
        (tmp_path / "conversion_meta.json").write_text(json.dumps(legacy_meta))

        _convert_with_fake_tokenizer(
            mixture,
            tmp_path,
            conversations_per_shard=100,
            mixture_name="standard",
            split=SFTSplit.TRAIN,
        )

        meta = json.loads((tmp_path / "conversion_meta.json").read_text())
        assert meta["split"] == "train"

    def test_legacy_split_absent_metadata_is_rejected_for_val(self, tmp_path):
        mixture = _make_fake_mixture(n=1)
        legacy_meta = {
            "mixture": "standard",
            "format": "full_nanochat_text",
            "completed_shards": 0,
            "total_shards": 1,
            "total_conversations": 1,
        }
        (tmp_path / "conversion_meta.json").write_text(json.dumps(legacy_meta))

        with pytest.raises(DataLoadError, match="no split metadata.*split='val'"):
            _convert_with_fake_tokenizer(
                mixture,
                tmp_path,
                conversations_per_shard=100,
                split=SFTSplit.VAL,
            )

    @pytest.mark.parametrize(
        "kwargs",
        [
            {"conversations_per_shard": 0},
            {"max_conversations": 0},
        ],
    )
    def test_numeric_contracts_reject_non_positive_values(self, tmp_path, kwargs):
        mixture = _make_fake_mixture(n=1)

        with pytest.raises(BeartypeCallHintParamViolation):
            _convert_with_fake_tokenizer(mixture, tmp_path, **kwargs)


def _main_cfg(output_dir, *, split: str, mixture: str = "standard", steered_jsonl=None):
    return OmegaConf.create(
        {
            "mixture": mixture,
            "split": split,
            "steered_jsonl": steered_jsonl,
            "max_conversations": None,
            "conversations_per_shard": 100,
            "output_dir": str(output_dir),
        }
    )


def _write_steered_jsonl(tmp_path, rows: int):
    path = tmp_path / "steered.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                [
                    {"role": "user", "content": f"Question {index}"},
                    {"role": "assistant", "content": f"Answer {index}"},
                ]
            )
            for index in range(rows)
        )
        + "\n",
        encoding="utf-8",
    )
    return path


class TestConverterSplitAxis:
    def test_split_val_writes_only_suffixed_val_dir(self, tmp_path):
        output_dir = tmp_path / "custom"
        train_task = FakeTask(_fake_conversations(2))
        val_task = FakeTask(_fake_conversations(1))
        with (
            patch(
                "tasks.mixtures.build_mixture",
                return_value=([train_task], [val_task]),
            ),
            patch(
                "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
                return_value=FakeTokenizer(),
            ),
        ):
            main.__wrapped__(_main_cfg(output_dir, split="val"))

        val_dir = tmp_path / "custom_val"
        assert not output_dir.exists()
        assert (val_dir / "shard_00000.parquet").exists()
        meta = json.loads((val_dir / "conversion_meta.json").read_text())
        assert meta["split"] == "val"

    def test_split_both_writes_train_and_val_dirs(self, tmp_path):
        output_dir = tmp_path / "custom"
        train_task = FakeTask(_fake_conversations(2))
        val_task = FakeTask(_fake_conversations(1))
        with (
            patch(
                "tasks.mixtures.build_mixture",
                return_value=([train_task], [val_task]),
            ),
            patch(
                "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
                return_value=FakeTokenizer(),
            ),
        ):
            main.__wrapped__(_main_cfg(output_dir, split="both"))

        train_meta = json.loads((output_dir / "conversion_meta.json").read_text())
        assert train_meta["split"] == "train"
        # A non-steered conversion must not carry the steered row count.
        assert "steered_rows" not in train_meta
        assert (
            json.loads((tmp_path / "custom_val" / "conversion_meta.json").read_text())["split"]
            == "val"
        )

    def test_empty_val_writes_no_dir_and_logs_info(self, tmp_path):
        output_dir = tmp_path / "custom"
        train_task = FakeTask(_fake_conversations(2))
        with (
            patch(
                "tasks.mixtures.build_mixture",
                return_value=([train_task], []),
            ),
            patch("scripts.preprocess.convert_sft_to_parquet.logger.info") as info,
        ):
            main.__wrapped__(_main_cfg(output_dir, split="val"))

        assert not output_dir.exists()
        assert not (tmp_path / "custom_val").exists()
        assert any(
            "'standard' has an empty val task list" in str(call.args[0])
            for call in info.call_args_list
        )


class TestConverterSteeredMixtures:
    """The steered mixtures convert train-only through the shared builder."""

    def test_steered_conversion_records_steered_rows(self, tmp_path):
        output_dir = tmp_path / "custom"
        steered_path = _write_steered_jsonl(tmp_path, 2)
        base_task = FakeTask(_fake_conversations(4))
        with (
            patch(
                "tasks.mixtures.build_mixture",
                return_value=([base_task, SteeredConversations(str(steered_path))], []),
            ),
            patch(
                "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
                return_value=FakeTokenizer(),
            ),
        ):
            main.__wrapped__(
                _main_cfg(
                    output_dir,
                    split="train",
                    mixture="standard_steered",
                    steered_jsonl=str(steered_path),
                )
            )

        meta = json.loads((output_dir / "conversion_meta.json").read_text())
        assert meta["steered_rows"] == 2
        assert meta["total_conversations"] == 6
        # The shared builder path put the steered task LAST and kept every base
        # task's own length entry in the fingerprint.
        fingerprint_tasks = meta["mixture_fingerprint"]["tasks"]
        assert [task["class_name"] for task in fingerprint_tasks] == [
            "FakeTask",
            "SteeredConversations",
        ]
        assert fingerprint_tasks[-1]["length"] == 2
        # The recorded identity is the STEERED construction, not a plain
        # TaskMixture over the same task list.
        steered_mixture = SteeredTaskMixture([base_task], SteeredConversations(str(steered_path)))
        assert meta["mixture_fingerprint"] == compute_mixture_fingerprint(
            steered_mixture,
            mixture_name="standard_steered",
            split=SFTSplit.TRAIN,
        )
        assert meta["mixture_fingerprint"] != compute_mixture_fingerprint(
            TaskMixture([base_task, SteeredConversations(str(steered_path))]),
            mixture_name="standard_steered",
            split=SFTSplit.TRAIN,
        )

    @pytest.mark.parametrize("split", ["val", "both"])
    def test_steered_conversion_rejects_val_splits(self, tmp_path, split):
        with pytest.raises(
            ValueError,
            match=r"val family for standard_steered is 'standard': run mixture=standard split=val",
        ):
            main.__wrapped__(
                _main_cfg(
                    tmp_path / "out",
                    split=split,
                    mixture="standard_steered",
                    steered_jsonl=str(_write_steered_jsonl(tmp_path, 2)),
                )
            )

    def test_steered_conversion_requires_steered_jsonl(self, tmp_path):
        with pytest.raises(
            ValueError,
            match=r"mixture 'standard_steered' requires steered_jsonl=<path>",
        ):
            main.__wrapped__(_main_cfg(tmp_path / "out", split="train", mixture="standard_steered"))

    def test_non_steered_conversion_rejects_steered_jsonl(self, tmp_path):
        with pytest.raises(
            ValueError,
            match=r"mixture 'standard' does not consume steered_jsonl=.*silently ignored",
        ):
            main.__wrapped__(
                _main_cfg(
                    tmp_path / "out",
                    split="train",
                    mixture="standard",
                    steered_jsonl=str(_write_steered_jsonl(tmp_path, 2)),
                )
            )


class TestReadConversionStatus:
    """Tests for read_conversion_status."""

    def test_returns_none_for_nonexistent_dir(self, tmp_path):
        """Returns None for a directory that doesn't exist."""
        result = read_conversion_status(tmp_path / "does_not_exist")
        assert result is None

    def test_returns_none_for_empty_dir(self, tmp_path):
        """Returns None for directory without conversion_meta.json."""
        result = read_conversion_status(tmp_path)
        assert result is None

    def test_returns_typed_metadata_when_meta_exists(self, tmp_path):
        """Returns parsed metadata when conversion_meta.json exists."""
        meta = {
            "mixture": "safety",
            "format": "full_nanochat_text",
            "completed_shards": 3,
            "total_shards": 5,
            "total_conversations": 200,
        }
        (tmp_path / "conversion_meta.json").write_text(json.dumps(meta))

        result = read_conversion_status(tmp_path)
        assert result is not None
        assert result.mixture == "safety"
        assert result.split is None
        assert result.to_json_dict() == meta

    def test_returns_typed_metadata_with_all_fields(self, tmp_path):
        """Returned metadata exposes all validated fields."""
        meta = {
            "mixture": "capabilities",
            "split": "val",
            "format": "full_nanochat_text",
            "completed_shards": 8,
            "total_shards": 12,
            "total_conversations": 462000,
        }
        (tmp_path / "conversion_meta.json").write_text(json.dumps(meta))

        result = read_conversion_status(tmp_path)
        assert result is not None
        assert result.mixture == "capabilities"
        assert result.split is SFTSplit.VAL
        assert result.completed_shards == 8
        assert result.total_shards == 12
        assert result.total_conversations == 462000
