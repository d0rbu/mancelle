"""Tests for shared decoder-only and encoder-decoder SFT mixtures.

Covers:
1. The exact ``SFTMixture`` selector surface and shared TaskMixture builder
2. Converter/encoder-decoder index-alignment parity
3. convert_mixture_to_parquet ValueError skip handling + original_index column
4. Checkpoint directory naming

Run with:
    uv run pytest tests/test_encdec_sft_mixtures.py -v
"""

import json
import re
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq
import pytest
from omegaconf import OmegaConf

import tasks.mixtures as mixtures_module
from errors import DataLoadError
from scripts.analysis.sft_parquet_alignment_audit import run_audit
from scripts.preprocess.convert_sft_to_parquet import (
    convert_mixture_to_parquet,
    render_conversation_to_full_nanochat_text,
)
from scripts.preprocess.convert_sft_to_parquet import (
    main as convert_main,
)
from scripts.train.conditional_encdec_sft import (
    _assert_sft_mixture_fingerprint_matches_task,
    _build_sft_checkpoint_dir,
)
from steering.sft.mixture_fingerprint import compute_mixture_fingerprint
from steering.sft_schema import SFTSplit
from tasks.common import Task, TaskMixture
from tasks.mixtures import (
    STEERED_MIXTURES,
    SFTMixture,
    build_task_mixtures,
    sft_val_family,
    steered_mixture_overrides,
)
from tasks.steered import SteeredTaskMixture
from tests.conftest import get_tokenizer_or_skip

_REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class FakeTask(Task):
    """Minimal Task for testing without HuggingFace downloads."""

    def __init__(self, conversations: list[dict[str, Any]]):
        super().__init__()
        self.conversations = conversations

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self) -> int:
        return len(self.conversations)

    def get_example(self, index: int):
        return self.conversations[index]


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


def _fake_conversations(prefix: str, count: int) -> list[dict[str, Any]]:
    return [
        {
            "messages": [
                {"role": "user", "content": f"{prefix} prompt {i}"},
                {"role": "assistant", "content": f"{prefix} answer {i}"},
            ]
        }
        for i in range(count)
    ]


def _fake_task(prefix: str, count: int = 2) -> FakeTask:
    return FakeTask(_fake_conversations(prefix, count))


@pytest.fixture(autouse=True)
def stub_converter_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep SFT parquet conversion tests independent of the nanochat tokenizer file."""
    monkeypatch.setattr(
        "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
        lambda: FakeTokenizer(),
    )


@pytest.fixture
def steered_jsonl(tmp_path: Path) -> str:
    """A small steered JSONL, sized to fit the stubbed standard mixture."""
    path = tmp_path / "steered.jsonl"
    path.write_text(
        "\n".join(
            json.dumps(
                [
                    {"role": "user", "content": f"steered prompt {index}"},
                    {"role": "assistant", "content": f"steered answer {index}"},
                ]
            )
            for index in range(4)
        )
        + "\n",
        encoding="utf-8",
    )
    return str(path)


def _convert_cfg(output_dir: Path, mixture: SFTMixture, steered_jsonl: str | None):
    """The converter CLI config for one train-split conversion."""
    return OmegaConf.create(
        {
            "mixture": mixture.value,
            "split": "train",
            "steered_jsonl": steered_jsonl,
            "max_conversations": None,
            "conversations_per_shard": 100,
            "output_dir": str(output_dir),
        }
    )


def _read_parquet_rows(output_dir: Path) -> dict[int, str]:
    """``original_index`` -> rendered ``text`` across every written shard."""
    rows: dict[int, str] = {}
    for shard in sorted(output_dir.glob("shard_*.parquet")):
        table = pq.read_table(shard)
        for index, text in zip(
            table.column("original_index").to_pylist(),
            table.column("text").to_pylist(),
            strict=True,
        ):
            assert index not in rows, f"duplicate original_index {index}"
            rows[index] = text
    return rows


class _FingerprintStore:
    """The two attributes the trainer's startup fingerprint gate reads."""

    def __init__(self, fingerprint: dict[str, Any]) -> None:
        self.parquet_mixture_fingerprint = fingerprint
        self.feature_mixture_fingerprint = fingerprint


def _overrides_for(mixture: SFTMixture, steered_jsonl: str) -> dict[str, Any]:
    """Feed the JSONL to the members that consume it, and only to those."""
    return steered_mixture_overrides(
        mixture, steered_jsonl if mixture in STEERED_MIXTURES else None
    )


@pytest.fixture
def stub_mixture_tasks(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep mixture-shape tests independent of external datasets and caches."""

    monkeypatch.setattr(mixtures_module, "exclude_heldout_questions", lambda train, val: None)

    task_names = (
        "ARC",
        "CoCoNot",
        "CustomJSON",
        "GSM8K",
        "MMLU",
        "MetaMathQA",
        "NemotronSafety",
        "OpenCoder",
        "OrcaMath",
        "SimpleSpelling",
        "SmolTalk",
        "SpellingBee",
        "WildGuardMix",
        "WildJailbreak",
    )

    for task_name in task_names:

        def fake_task_factory(*args: Any, _task_name: str = task_name, **kwargs: Any) -> FakeTask:
            del args
            split = kwargs.get("split", "default")
            return _fake_task(f"{_task_name}-{split}", count=2)

        monkeypatch.setattr(mixtures_module, task_name, fake_task_factory)
    monkeypatch.setattr(mixtures_module, "get_base_dir", lambda: "/fake-base")


# ===========================================================================
# 1. Shared SFT mixture selection and construction
# ===========================================================================


class TestSFTMixtureSelection:
    """Pin the exact shared mixture selector and TaskMixture result shape."""

    @pytest.fixture(autouse=True)
    def _stub_mixture_tasks(self, stub_mixture_tasks: None) -> None:
        pass

    def test_valid_set_is_exactly_the_shared_mixtures(self) -> None:
        assert {mixture.value for mixture in SFTMixture} == {
            "standard",
            "capabilities",
            "safety",
            "chatcore",
            "standard_steered",
            "standard_steered_balanced",
        }

    def test_val_family_maps_only_the_steered_members_to_standard(self) -> None:
        assert STEERED_MIXTURES == {
            SFTMixture.STANDARD_STEERED,
            SFTMixture.STANDARD_STEERED_BALANCED,
        }
        for mixture in SFTMixture:
            expected = SFTMixture.STANDARD if mixture in STEERED_MIXTURES else mixture
            assert sft_val_family(mixture) is expected

    def test_unknown_dataset_error_names_valid_mixtures(self, tmp_path) -> None:
        with pytest.raises(
            ValueError,
            match=r"Unknown dataset.*capabilities.*safety.*standard",
        ):
            run_audit(
                dataset="unknown",
                parquet_dir=tmp_path,
                max_examples=8,
            )

    @pytest.mark.parametrize("mixture", list(SFTMixture))
    def test_build_task_mixtures_returns_train_and_val_task_mixtures(
        self, mixture: SFTMixture, steered_jsonl: str
    ) -> None:
        train, val = build_task_mixtures(mixture, **_overrides_for(mixture, steered_jsonl))

        assert isinstance(train, TaskMixture)
        assert isinstance(train, SteeredTaskMixture) is (mixture in STEERED_MIXTURES)
        assert isinstance(val, TaskMixture)
        assert len(train) > 0
        assert len(val) > 0

    @pytest.mark.parametrize("mixture", list(SFTMixture))
    def test_converter_original_index_matches_the_trainer_mixture_row_for_row(
        self, tmp_path: Path, mixture: SFTMixture, steered_jsonl: str
    ) -> None:
        """Parity against what the converter REALLY writes, not against itself.

        The trainer's mixture is built here; the parquet is produced by the
        converter's own ``main()``. Every row is compared by content at its
        ``original_index``, and the recorded fingerprint must equal the one the
        trainer computes over its own task — so a converter that reverted to a
        second construction path (or reordered anything) fails for every
        member, steered and plain alike.
        """
        overrides = _overrides_for(mixture, steered_jsonl)
        trainer_train, _ = build_task_mixtures(mixture, **overrides)

        output_dir = tmp_path / mixture.value
        convert_main.__wrapped__(
            _convert_cfg(output_dir, mixture, overrides.get("steered_jsonl"))
        )

        rows = _read_parquet_rows(output_dir)
        assert sorted(rows) == list(range(len(trainer_train)))
        tokenizer = FakeTokenizer()
        for original_index, text in rows.items():
            expected_text, skip_reason = render_conversation_to_full_nanochat_text(
                tokenizer, trainer_train[original_index]
            )
            assert skip_reason is None
            assert text == expected_text

        meta = json.loads((output_dir / "conversion_meta.json").read_text())
        assert meta["mixture_fingerprint"] == compute_mixture_fingerprint(
            trainer_train,
            mixture_name=mixture.value,
            split=SFTSplit.TRAIN,
        )

    @pytest.mark.parametrize("mixture", sorted(STEERED_MIXTURES))
    def test_steered_parquet_fingerprint_is_accepted_by_the_trainer_gate(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        mixture: SFTMixture,
        steered_jsonl: str,
    ) -> None:
        """The REAL tokenizer path, end to end, through the trainer's gate.

        Converts a steered mixture with the production nanochat tokenizer, then
        rebuilds the same mixture the way the trainer does and hands the
        artifact to ``_assert_sft_mixture_fingerprint_matches_task`` — the
        trainer's own startup guard. Nothing else proves a steered parquet is
        loadable by the trainer that has to consume it.
        """
        tokenizer = get_tokenizer_or_skip()
        monkeypatch.setattr(
            "scripts.preprocess.convert_sft_to_parquet.get_tokenizer",
            lambda: tokenizer,
        )
        overrides = _overrides_for(mixture, steered_jsonl)
        output_dir = tmp_path / mixture.value
        convert_main.__wrapped__(
            _convert_cfg(output_dir, mixture, overrides["steered_jsonl"])
        )

        task, _ = build_task_mixtures(mixture, **overrides)
        assert isinstance(task, SteeredTaskMixture)
        rows = _read_parquet_rows(output_dir)
        assert sorted(rows) == list(range(len(task)))
        # original_index IS the generator position, and the steered rows sit
        # exactly on the evenly spaced positions the mixture computed.
        steered_rows = {
            index for index, text in rows.items() if "steered prompt" in text
        }
        assert steered_rows == set(task.steered_positions)
        assert len(steered_rows) == len(task.tasks[task.steered_task_index])

        meta = json.loads((output_dir / "conversion_meta.json").read_text())
        assert meta["steered_rows"] == len(steered_rows)
        store = _FingerprintStore(meta["mixture_fingerprint"])
        _assert_sft_mixture_fingerprint_matches_task(
            store=store,
            task=task,
            mixture_name=mixture.value,
            expected_split=SFTSplit.TRAIN,
            feature_dir=str(tmp_path / "features"),
            parquet_dir=str(output_dir),
            alignment_probe_samples=0,
        )

        # Non-vacuity: a DIFFERENT steered file of a different length is a
        # different mixture and the same gate must reject it.
        other_path = tmp_path / "other.jsonl"
        other_path.write_text(
            "".join(
                json.dumps(
                    [
                        {"role": "user", "content": f"steered prompt {index}"},
                        {"role": "assistant", "content": f"steered answer {index}"},
                    ]
                )
                + "\n"
                for index in range(5)
            ),
            encoding="utf-8",
        )
        other_task, _ = build_task_mixtures(mixture, steered_jsonl=str(other_path))
        assert len(other_task) != len(task)
        with pytest.raises(DataLoadError, match="fingerprint mismatch"):
            _assert_sft_mixture_fingerprint_matches_task(
                store=store,
                task=other_task,
                mixture_name=mixture.value,
                expected_split=SFTSplit.TRAIN,
                feature_dir=str(tmp_path / "features"),
                parquet_dir=str(output_dir),
                alignment_probe_samples=0,
            )

    def test_encdec_source_calls_build_task_mixtures_with_only_steered_overrides(self) -> None:
        """The steered JSONL is the ONE permitted override at this call site.

        Every mixture-sizing knob still has to stay out: honouring one would
        desynchronise ``original_index`` from the converted parquet. The steered
        path is exempt because the mixture cannot be built without its JSONL and
        the converter reads the same key from its own config.
        """
        source = (_REPO_ROOT / "scripts" / "train" / "conditional_encdec_sft.py").read_text()
        normalized = re.sub(r"\s+", "", source)
        assert normalized.count("build_task_mixtures(") == 1
        assert (
            "build_task_mixtures(dataset_mixture,"
            "**steered_mixture_overrides(dataset_mixture,cfg.steered_jsonl),)" in normalized
        )

    def test_converter_source_shares_the_build_task_mixtures_path(self) -> None:
        """Structural backstop for the behavioural parity test above.

        The behavioural proof is
        ``test_converter_original_index_matches_the_trainer_mixture_row_for_row``;
        this pin additionally forbids a SECOND construction path appearing
        elsewhere in the converter, which row-for-row equality of the one path
        that ``main()`` uses could not see.
        """
        source = (
            _REPO_ROOT / "scripts" / "preprocess" / "convert_sft_to_parquet.py"
        ).read_text()
        normalized = re.sub(r"\s+", "", source)
        assert normalized.count("build_task_mixtures(") == 1
        assert "steered_mixture_overrides(mixture_enum,cfg.steered_jsonl)" in normalized
        assert "TaskMixture(train_tasks)" not in normalized


# ===========================================================================
# 2. convert_mixture_to_parquet ValueError skip + original_index tests
# ===========================================================================


class TestParquetConversionSkipsBadConversations:
    """Test that convert_mixture_to_parquet skips conversations with no
    assistant message and writes correct original_index column tracking
    which mixture indices survived.

    The conversion stores full rendered text plus ``original_index``.
    Conversations with no assistant message or an empty final response are
    skipped.
    """

    def _make_mixture_with_bad_conversations(self, n_good, n_bad):
        """Create a mixture where some conversations have no assistant messages.

        Layout: [good_0..good_{n/2-1}, bad_0..bad_{n_bad-1}, good_{n/2}..good_{n-1}]
        """
        good_convos = [
            {
                "messages": [
                    {"role": "user", "content": f"Good question {i}"},
                    {"role": "assistant", "content": f"Good answer {i}"},
                ]
            }
            for i in range(n_good)
        ]
        # Bad conversations: no assistant message -> skipped
        bad_convos = [
            {
                "messages": [
                    {"role": "user", "content": f"Bad question {i} with no answer"},
                ]
            }
            for i in range(n_bad)
        ]
        # Interleave: put bad ones in the middle
        all_convos = good_convos[: n_good // 2] + bad_convos + good_convos[n_good // 2 :]
        task = FakeTask(all_convos)
        return TaskMixture([task])

    def test_bad_conversations_are_skipped(self, tmp_path):
        """Conversations with no assistant message are excluded from parquet output."""
        n_good, n_bad = 8, 3
        mixture = self._make_mixture_with_bad_conversations(n_good, n_bad)

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        assert table.num_rows == n_good, (
            f"Expected {n_good} rows (bad conversations skipped), got {table.num_rows}"
        )

    def test_good_conversations_intact_when_some_skipped(self, tmp_path):
        """Good conversations are rendered to full text even when others are skipped."""
        n_good, n_bad = 6, 2
        mixture = self._make_mixture_with_bad_conversations(n_good, n_bad)

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        texts = table.column("text").to_pylist()
        indices = table.column("original_index").to_pylist()

        assert len(texts) == n_good
        assert len(indices) == n_good

        for text, idx in zip(texts, indices, strict=True):
            conv = mixture[idx]
            assert any(m["role"] == "assistant" for m in conv["messages"])
            assert "<assistant>" in text
            assert len(text) > 0

    def test_all_bad_conversations_raises_data_load_error(self, tmp_path):
        """If every conversation is bad, DataLoadError is raised (>50% skip rate)."""
        from errors import DataLoadError

        bad_convos = [
            {"messages": [{"role": "user", "content": f"No answer {i}"}]} for i in range(5)
        ]
        task = FakeTask(bad_convos)
        mixture = TaskMixture([task])

        with pytest.raises(DataLoadError, match="More than 50%"):
            convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

    def test_no_good_conversations_still_writes_metadata(self, tmp_path):
        """A rejected conversion retains identity without publishing success.

        Main's audited-resume converter commits completion only after the
        skip-rate guard. Reusing its empty shard must not bypass that guard.
        """
        bad_convos = [{"messages": [{"role": "user", "content": f"Bad {i}"}]} for i in range(3)]
        task = FakeTask(bad_convos)
        mixture = TaskMixture([task])

        snapshots = []
        for _ in range(2):
            with pytest.raises(DataLoadError, match="More than 50%"):
                convert_mixture_to_parquet(
                    mixture,
                    tmp_path,
                    conversations_per_shard=100,
                    mixture_name="test",
                )
            meta_path = tmp_path / "conversion_meta.json"
            snapshots.append(meta_path.read_bytes())
            meta = json.loads(snapshots[-1])
            assert meta["completed_shards"] == 0
            assert meta["total_shards"] == 1
            assert meta["total_conversations"] == 3
            assert meta["written_conversations"] == 0
            assert pq.read_table(tmp_path / "shard_00000.parquet").num_rows == 0
        assert snapshots[0] == snapshots[1]

    def test_skip_across_multiple_shards(self, tmp_path):
        """Bad conversations spread across shards are all skipped correctly."""
        # 10 good + 4 bad = 14 total, 5 per shard = 3 shards
        mixture = self._make_mixture_with_bad_conversations(n_good=10, n_bad=4)

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=5)

        total_rows = 0
        for pf in tmp_path.glob("*.parquet"):
            table = pq.read_table(pf)
            total_rows += table.num_rows
        assert total_rows == 10, f"Expected 10 good rows across shards, got {total_rows}"

    def test_columns_consistent_length(self, tmp_path):
        """text and original_index columns always have same length."""
        mixture = self._make_mixture_with_bad_conversations(n_good=7, n_bad=3)

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        for pf in tmp_path.glob("*.parquet"):
            table = pq.read_table(pf)
            n = table.num_rows
            assert len(table.column("text")) == n
            assert len(table.column("original_index")) == n

    def test_original_index_column_exists(self, tmp_path):
        """Parquet output includes original_index column."""
        convos = [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ]
            }
            for i in range(5)
        ]
        task = FakeTask(convos)
        mixture = TaskMixture([task])

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        assert "original_index" in table.column_names

    def test_original_index_sequential_when_no_skips(self, tmp_path):
        """When no conversations are skipped, original_index is 0..n-1."""
        n = 8
        convos = [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ]
            }
            for i in range(n)
        ]
        task = FakeTask(convos)
        mixture = TaskMixture([task])

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        indices = table.column("original_index").to_pylist()
        assert indices == list(range(n)), (
            f"Expected sequential indices {list(range(n))}, got {indices}"
        )

    def test_original_index_skips_bad_conversations(self, tmp_path):
        """original_index skips indices of conversations with no assistant message.

        Verifies that every original_index points to a conversation that actually
        has assistant tokens, and that indices pointing to bad conversations are
        excluded. TaskMixture shuffles with seed=42, so the mapping from mixture
        index to underlying task index is deterministic but not identity.
        """
        # Build mixture manually for precise control
        convos = [
            {
                "messages": [
                    {"role": "user", "content": "Q0"},
                    {"role": "assistant", "content": "A0"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q1"},
                    {"role": "assistant", "content": "A1"},
                ]
            },
            {"messages": [{"role": "user", "content": "Bad0"}]},  # will be skipped
            {"messages": [{"role": "user", "content": "Bad1"}]},  # will be skipped
            {
                "messages": [
                    {"role": "user", "content": "Q2"},
                    {"role": "assistant", "content": "A2"},
                ]
            },
            {
                "messages": [
                    {"role": "user", "content": "Q3"},
                    {"role": "assistant", "content": "A3"},
                ]
            },
        ]
        task = FakeTask(convos)
        mixture = TaskMixture([task])

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        table = pq.read_table(sorted(tmp_path.glob("*.parquet"))[0])
        indices = table.column("original_index").to_pylist()
        assert table.num_rows == 4, f"Expected 4 rows, got {table.num_rows}"
        # Every original_index should point to a good conversation (one with assistant)
        for idx in indices:
            conv = mixture[idx]
            has_assistant = any(m["role"] == "assistant" for m in conv["messages"])
            assert has_assistant, (
                f"original_index {idx} points to a conversation without assistant tokens"
            )
        # The skipped mixture indices should NOT appear
        # Find which mixture indices map to bad conversations
        bad_mixture_indices = set()
        for i in range(len(mixture)):
            conv = mixture[i]
            if not any(m["role"] == "assistant" for m in conv["messages"]):
                bad_mixture_indices.add(i)
        assert len(bad_mixture_indices) == 2, f"Expected 2 bad indices, found {bad_mixture_indices}"
        for bad_idx in bad_mixture_indices:
            assert bad_idx not in indices, (
                f"Bad conversation at mixture index {bad_idx} should be excluded"
            )

    def test_original_index_across_shards_are_globally_correct(self, tmp_path):
        """original_index values reflect global mixture position, not shard-local position."""
        n = 10
        convos = [
            {
                "messages": [
                    {"role": "user", "content": f"Q{i}"},
                    {"role": "assistant", "content": f"A{i}"},
                ]
            }
            for i in range(n)
        ]
        task = FakeTask(convos)
        mixture = TaskMixture([task])

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=4)

        # Collect all original_indices across shards
        all_indices = []
        for pf in sorted(tmp_path.glob("*.parquet")):
            table = pq.read_table(pf)
            all_indices.extend(table.column("original_index").to_pylist())

        # Should have all n indices, each exactly once
        assert sorted(all_indices) == list(range(n)), (
            f"Expected indices 0..{n - 1} across all shards, got {sorted(all_indices)}"
        )

    def test_original_index_monotonically_increasing_within_shard(self, tmp_path):
        """Within each shard, original_index values are monotonically increasing
        (since we iterate through conversations in order and skip bad ones)."""
        mixture = self._make_mixture_with_bad_conversations(n_good=8, n_bad=3)

        convert_mixture_to_parquet(mixture, tmp_path, conversations_per_shard=100)

        for pf in tmp_path.glob("*.parquet"):
            table = pq.read_table(pf)
            indices = table.column("original_index").to_pylist()
            for i in range(1, len(indices)):
                assert indices[i] > indices[i - 1], (
                    f"original_index not monotonically increasing: {indices}"
                )


# ===========================================================================
# 3. Checkpoint directory naming tests
# ===========================================================================


class TestCheckpointDirectoryNaming:
    """Test the conditioned SFT checkpoint directory name."""

    def test_conditioned_suffix(self):
        dataset = "safety"
        checkpoint_dir_name = f"{dataset}_sft_conditioned_checkpoints"

        assert checkpoint_dir_name == "safety_sft_conditioned_checkpoints"

    @pytest.mark.parametrize("dataset", [mixture.value for mixture in SFTMixture])
    def test_naming_for_each_dataset(self, dataset):
        """Verify naming pattern works for all dataset types."""
        name = _build_sft_checkpoint_dir("/checkpoints", dataset)
        assert "_conditioned_" in name
        assert name.endswith(f"{dataset}_sft_conditioned_checkpoints")
        assert name.endswith("_checkpoints")

    def test_old_naming_not_used(self):
        """The old naming convention ({dataset}_sft_checkpoints) should not match
        either new convention."""
        dataset = "safety"
        old_name = f"{dataset}_sft_checkpoints"
        conditioned = f"{dataset}_sft_conditioned_checkpoints"
        assert old_name != conditioned


# ===========================================================================
# 4. build_task_mixtures helper tests
# ===========================================================================


class TestBuildSftMixture:
    """Test the shared build_task_mixtures helper function."""

    @pytest.fixture(autouse=True)
    def _stub_mixture_tasks(self, stub_mixture_tasks: None) -> None:
        pass

    def test_returns_task_mixture(self):
        """build_task_mixtures must return a (TaskMixture, ...) tuple."""
        result, _val = build_task_mixtures(SFTMixture.SAFETY)
        assert isinstance(result, TaskMixture)

    def test_empty_val_task_list_returns_none(self, monkeypatch: pytest.MonkeyPatch):
        monkeypatch.setattr(
            mixtures_module,
            "build_mixture",
            lambda _mixture: ([_fake_task("train")], []),
        )

        train, val = build_task_mixtures(SFTMixture.STANDARD)

        assert isinstance(train, TaskMixture)
        assert val is None

    def test_safety_mixture_non_empty(self):
        mixture, _val = build_task_mixtures(SFTMixture.SAFETY)
        assert len(mixture) > 0

    def test_capabilities_mixture_non_empty(self):
        mixture, _val = build_task_mixtures(SFTMixture.CAPABILITIES)
        assert len(mixture) > 0

    def test_safety_has_many_tasks(self):
        """Safety mixture should compose many sub-tasks (refusal, helpfulness, identity)."""
        mixture, _val = build_task_mixtures(SFTMixture.SAFETY)
        # The safety mixture includes 13 task copies (WildJailbreak*2,
        # WildGuardMix*3, CoCoNot*2, SmolTalk, NemotronSafety, Identity*2).
        assert len(mixture.tasks) >= 10, (
            f"Safety mixture should have many sub-tasks, got {len(mixture.tasks)}"
        )

    def test_capabilities_has_expected_components(self):
        """Capabilities mixture should include coding, math, knowledge components."""
        mixture, _val = build_task_mixtures(SFTMixture.CAPABILITIES)
        # Should have SmolTalk, Identity*2, MMLU*2, OpenCoder, OrcaMath,
        # MetaMath, spelling*2.
        assert len(mixture.tasks) >= 7, (
            f"Capabilities mixture should have several sub-tasks, got {len(mixture.tasks)}"
        )

    @pytest.mark.parametrize("mixture_type", list(SFTMixture))
    def test_can_index_into_mixture(self, mixture_type: SFTMixture, steered_jsonl: str):
        """Should be able to access examples by index."""
        mixture, _val = build_task_mixtures(
            mixture_type, **_overrides_for(mixture_type, steered_jsonl)
        )
        conv = mixture[0]
        assert "messages" in conv
        assert len(conv["messages"]) >= 2
        assert conv["messages"][0]["role"] == "user"
        assert any(message["role"] == "assistant" for message in conv["messages"])

    def test_deterministic_shuffle(self):
        """Two calls with same enum should produce identical ordering (seed=42)."""
        mix1, _val1 = build_task_mixtures(SFTMixture.SAFETY)
        mix2, _val2 = build_task_mixtures(SFTMixture.SAFETY)
        assert len(mix1) == len(mix2)
        # Check first 10 examples are identical
        for i in range(min(10, len(mix1))):
            conv1 = mix1[i]
            conv2 = mix2[i]
            assert conv1["messages"][0]["content"] == conv2["messages"][0]["content"]

    def test_standard_mixture_also_works(self):
        """Standard mixture should also work through the helper."""
        mixture, _val = build_task_mixtures(SFTMixture.STANDARD)
        assert isinstance(mixture, TaskMixture)
        assert len(mixture) > 0
