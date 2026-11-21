"""Tests for the SFT-pretraining mixing pipeline."""

import hashlib
import json
import os
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

import scripts.preprocess.mix_sft_into_pretraining as mix_sft_module
from scripts.preprocess.mix_sft_into_pretraining import (
    _default_output_name,
    discover_shards,
    distribute_uniformly,
    mix_sft_into_pretraining,
    render_sft_conversations,
)
from steering.models import resolve_tokenizer

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_sft_parquet(path: Path, conversations: list[dict]) -> None:
    """Create a fake full-rendered SFT parquet shard."""
    texts = []
    indices = []
    for i, conv in enumerate(conversations):
        if "text" in conv:
            texts.append(conv["text"])
        else:
            parts = [
                f"<{msg['role']}>{msg['content']}</{msg['role']}>"
                for msg in conv["messages"]
            ]
            texts.append("".join(parts))
        indices.append(i)
    table = pa.table(
        {
            "text": texts,
            "original_index": indices,
        }
    )
    pq.write_table(table, path)


def _make_pretraining_parquet(path: Path, texts: list[str]) -> None:
    """Create a fake pretraining parquet shard with a text column."""
    table = pa.table({"text": texts})
    pq.write_table(table, path)


def _assert_output_alias_rejected_without_mutation(
    *,
    base_data_dir: Path,
    sft_data_dir: Path,
    output_dir: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Assert an aliased output is rejected before any filesystem mutation."""
    input_paths = [
        *discover_shards(base_data_dir),
        *discover_shards(sft_data_dir / "safety"),
    ]
    hashes_before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths
    }
    mutation_calls: list[tuple[str, Path]] = []
    original_mkdir = Path.mkdir

    def record_mkdir(path: Path, *args, **kwargs) -> None:
        mutation_calls.append(("mkdir", path))
        original_mkdir(path, *args, **kwargs)

    def reject_write(_table, path: Path, *args, **kwargs) -> None:
        mutation_calls.append(("write", Path(path)))
        raise AssertionError(f"attempted to write aliased output: {path}")

    def reject_unlink(path: Path, *args, **kwargs) -> None:
        mutation_calls.append(("unlink", path))
        raise AssertionError(f"attempted to unlink aliased output: {path}")

    monkeypatch.setattr(Path, "mkdir", record_mkdir)
    monkeypatch.setattr(Path, "unlink", reject_unlink)
    monkeypatch.setattr(mix_sft_module.pq, "write_table", reject_write)

    try:
        with pytest.raises(ValueError, match="aliases an input"):
            mix_sft_into_pretraining(
                mixtures=["safety"],
                model_name="gemma-2-2b",
                base_data_dir=base_data_dir,
                sft_data_dir=sft_data_dir,
                output_dir=output_dir,
            )
    finally:
        hashes_after = {
            path: hashlib.sha256(path.read_bytes()).hexdigest() for path in input_paths
        }
        assert hashes_after == hashes_before
        assert mutation_calls == []


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDiscoverShards:
    def test_finds_existing_shards(self, tmp_path: Path) -> None:
        for i in range(3):
            _make_pretraining_parquet(tmp_path / f"shard_{i:05d}.parquet", [f"doc{i}"])
        shards = discover_shards(tmp_path)
        assert len(shards) == 3
        assert [s.name for s in shards] == [
            "shard_00000.parquet",
            "shard_00001.parquet",
            "shard_00002.parquet",
        ]

    def test_returns_sorted(self, tmp_path: Path) -> None:
        for i in [2, 0, 1]:
            _make_pretraining_parquet(tmp_path / f"shard_{i:05d}.parquet", [f"doc{i}"])
        shards = discover_shards(tmp_path)
        assert [s.name for s in shards] == [
            "shard_00000.parquet",
            "shard_00001.parquet",
            "shard_00002.parquet",
        ]

    def test_ignores_non_shard_files(self, tmp_path: Path) -> None:
        _make_pretraining_parquet(tmp_path / "shard_00000.parquet", ["doc0"])
        (tmp_path / "other_file.parquet").touch()
        (tmp_path / "conversion_meta.json").write_text("{}")
        (tmp_path / "mixing_meta.json").write_text("{}")
        shards = discover_shards(tmp_path)
        assert len(shards) == 1
        assert shards[0].name == "shard_00000.parquet"

    def test_empty_directory(self, tmp_path: Path) -> None:
        assert discover_shards(tmp_path) == []


class TestDistributeUniformly:
    def test_even_distribution(self) -> None:
        items = list(range(12))
        buckets = distribute_uniformly(items, 3, random.Random(0))
        assert all(len(b) == 4 for b in buckets)
        assert sorted(sum(buckets, [])) == items

    def test_uneven_distribution_within_one(self) -> None:
        items = list(range(10))
        buckets = distribute_uniformly(items, 3, random.Random(0))
        sizes = sorted(len(b) for b in buckets)
        assert sizes == [3, 3, 4]
        assert sorted(sum(buckets, [])) == items

    def test_deterministic_with_same_seed(self) -> None:
        items = list(range(20))
        b1 = distribute_uniformly(items, 4, random.Random(42))
        b2 = distribute_uniformly(items, 4, random.Random(42))
        assert b1 == b2

    def test_different_seeds_different_order(self) -> None:
        items = list(range(50))
        b1 = distribute_uniformly(items, 5, random.Random(0))
        b2 = distribute_uniformly(items, 5, random.Random(99))
        flat1 = sum(b1, [])
        flat2 = sum(b2, [])
        assert flat1 != flat2
        assert sorted(flat1) == sorted(flat2)

    def test_empty_items(self) -> None:
        buckets = distribute_uniformly([], 3, random.Random(0))
        assert len(buckets) == 3
        assert all(len(b) == 0 for b in buckets)

    def test_single_bucket(self) -> None:
        items = list(range(5))
        buckets = distribute_uniformly(items, 1, random.Random(0))
        assert len(buckets) == 1
        assert sorted(buckets[0]) == items

    def test_more_buckets_than_items(self) -> None:
        items = [1, 2]
        buckets = distribute_uniformly(items, 5, random.Random(0))
        assert len(buckets) == 5
        non_empty = [b for b in buckets if b]
        assert len(non_empty) == 2
        assert sorted(sum(buckets, [])) == [1, 2]


class TestRenderSftConversations:
    def test_single_conversation(self, tmp_path: Path) -> None:
        sft_dir = tmp_path / "safety"
        sft_dir.mkdir()
        conv = {
            "messages": [
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": "Hi there"},
            ],
            "response": "Hi there",
        }
        _make_sft_parquet(sft_dir / "shard_00000.parquet", [conv])
        texts = render_sft_conversations([sft_dir])

        assert len(texts) == 1
        assert "<user>Hello</user>" in texts[0]
        assert "<assistant>Hi there</assistant>" in texts[0]

    def test_multi_turn_conversation(self, tmp_path: Path) -> None:
        sft_dir = tmp_path / "standard"
        sft_dir.mkdir()
        conv = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hello!"},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I'm fine"},
            ],
            "response": "I'm fine",
        }
        _make_sft_parquet(sft_dir / "shard_00000.parquet", [conv])
        texts = render_sft_conversations([sft_dir])

        assert len(texts) == 1
        assert texts[0].count("<user>") == 2
        assert texts[0].count("<assistant>") == 2

    def test_uses_prerendered_text_without_tokenizer_bos_logic(self, tmp_path: Path) -> None:
        sft_dir = tmp_path / "safety"
        sft_dir.mkdir()
        conv = {
            "messages": [
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ],
            "response": "Hey",
        }
        _make_sft_parquet(sft_dir / "shard_00000.parquet", [conv])
        texts = render_sft_conversations([sft_dir])

        assert len(texts) == 1
        assert not texts[0].startswith("<BOS>")
        assert "<user>Hi</user>" in texts[0]

    def test_prerendered_text_keeps_system_content(
        self, tmp_path: Path
    ) -> None:
        sft_dir = tmp_path / "safety"
        sft_dir.mkdir()
        conv = {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ],
            "response": "Hey",
        }
        _make_sft_parquet(sft_dir / "shard_00000.parquet", [conv])
        texts = render_sft_conversations([sft_dir])

        assert len(texts) == 1
        assert "<system>Be helpful</system>" in texts[0]
        assert "Be helpful" in texts[0]
        assert "Hi" in texts[0]
        assert "<user>" in texts[0]

    def test_preserves_system_when_not_folding(self, tmp_path: Path) -> None:
        sft_dir = tmp_path / "standard"
        sft_dir.mkdir()
        conv = {
            "messages": [
                {"role": "system", "content": "Be helpful"},
                {"role": "user", "content": "Hi"},
                {"role": "assistant", "content": "Hey"},
            ],
            "response": "Hey",
        }
        _make_sft_parquet(sft_dir / "shard_00000.parquet", [conv])
        texts = render_sft_conversations([sft_dir])

        assert len(texts) == 1
        assert "<system>Be helpful</system>" in texts[0]

    def test_multiple_shards_and_dirs(self, tmp_path: Path) -> None:
        for mix_name in ["safety", "capabilities"]:
            d = tmp_path / mix_name
            d.mkdir()
            for shard_idx in range(2):
                conv = {
                    "messages": [
                        {"role": "user", "content": f"{mix_name} q{shard_idx}"},
                        {"role": "assistant", "content": f"{mix_name} a{shard_idx}"},
                    ],
                    "response": f"{mix_name} a{shard_idx}",
                }
                _make_sft_parquet(d / f"shard_{shard_idx:05d}.parquet", [conv])

        texts = render_sft_conversations(
            [tmp_path / "safety", tmp_path / "capabilities"]
        )

        assert len(texts) == 4

    def test_empty_sft_dir_warns(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()
        texts = render_sft_conversations([empty_dir])
        assert texts == []

    def test_multiple_conversations_per_shard(self, tmp_path: Path) -> None:
        sft_dir = tmp_path / "safety"
        sft_dir.mkdir()
        convs = [
            {
                "messages": [
                    {"role": "user", "content": f"Question {i}"},
                    {"role": "assistant", "content": f"Answer {i}"},
                ],
                "response": f"Answer {i}",
            }
            for i in range(5)
        ]
        _make_sft_parquet(sft_dir / "shard_00000.parquet", convs)
        texts = render_sft_conversations([sft_dir])

        assert len(texts) == 5
        for i in range(5):
            assert f"Question {i}" in texts[i]
            assert f"Answer {i}" in texts[i]

    @pytest.mark.parametrize(
        "legacy_column",
        ["assistant_char_offset", "assistant_loss_mask"],
    )
    def test_rejects_text_shard_with_legacy_sft_columns(
        self,
        tmp_path: Path,
        legacy_column: str,
    ) -> None:
        sft_dir = tmp_path / "safety"
        sft_dir.mkdir()
        pq.write_table(
            pa.table(
                {
                    "text": ["rendered"],
                    "original_index": [0],
                    legacy_column: [3],
                }
            ),
            sft_dir / "shard_00000.parquet",
        )

        with pytest.raises(ValueError, match="legacy schema"):
            render_sft_conversations([sft_dir])


class TestModelSupportsSystem:
    def test_unknown_model_raises_in_resolve(self) -> None:
        with pytest.raises(ValueError, match="not found in EXTRACTION_MODELS"):
            resolve_tokenizer("unknown-model-xyz")


class TestMixSftIntoPretraining:
    """Tests for the core mixing pipeline."""

    @pytest.fixture()
    def data_dirs(self, tmp_path: Path) -> tuple[Path, Path, Path]:
        """Create synthetic pretraining and SFT data for testing."""
        base_dir = tmp_path / "base_data"
        base_dir.mkdir()
        for i in range(3):
            texts = [f"pretrain_doc_{i}_{j}" for j in range(10)]
            _make_pretraining_parquet(base_dir / f"shard_{i:05d}.parquet", texts)

        sft_dir = tmp_path / "sft_data"
        safety_dir = sft_dir / "safety"
        safety_dir.mkdir(parents=True)
        convs = [
            {
                "messages": [
                    {"role": "user", "content": f"safety_q_{j}"},
                    {"role": "assistant", "content": f"safety_a_{j}"},
                ],
                "response": f"safety_a_{j}",
            }
            for j in range(6)
        ]
        _make_sft_parquet(safety_dir / "shard_00000.parquet", convs)

        output_dir = tmp_path / "mixed_data" / "safety"
        return base_dir, sft_dir, output_dir

    @pytest.mark.parametrize("source_kind", ["pretraining", "sft"])
    @pytest.mark.parametrize("alias_kind", ["exact", "normalized", "symlink"])
    def test_rejects_output_directory_alias_before_mutation(
        self,
        data_dirs: tuple[Path, Path, Path],
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        source_kind: str,
        alias_kind: str,
    ) -> None:
        base_dir, sft_dir, _ = data_dirs
        source_dir = base_dir if source_kind == "pretraining" else sft_dir / "safety"
        if alias_kind == "exact":
            output_dir = source_dir
        elif alias_kind == "normalized":
            output_dir = source_dir / ".." / source_dir.name
        else:
            output_dir = tmp_path / f"{source_kind}_alias"
            output_dir.symlink_to(source_dir, target_is_directory=True)

        _assert_output_alias_rejected_without_mutation(
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
            monkeypatch=monkeypatch,
        )

    def test_rejects_existing_output_shard_with_input_inode_before_mutation(
        self,
        data_dirs: tuple[Path, Path, Path],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        output_dir.mkdir(parents=True)
        source_shard = base_dir / "shard_00000.parquet"
        os.link(source_shard, output_dir / source_shard.name)

        _assert_output_alias_rejected_without_mutation(
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
            monkeypatch=monkeypatch,
        )

    def test_output_has_same_shard_names(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        output_shards = sorted(output_dir.glob("shard_*.parquet"))
        assert [s.name for s in output_shards] == [
            "shard_00000.parquet",
            "shard_00001.parquet",
            "shard_00002.parquet",
        ]

    def test_output_schema_is_text_only(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        for shard_path in output_dir.glob("shard_*.parquet"):
            table = pq.read_table(shard_path)
            assert table.column_names == ["text"]
            assert table.schema.field("text").type == pa.string()

    def test_total_docs_equals_pretrain_plus_sft(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        meta = mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        assert meta["total_pretraining_docs"] == 30
        assert meta["total_sft_docs"] == 6
        assert meta["total_docs"] == 36

        total_rows = 0
        for shard_path in sorted(output_dir.glob("shard_*.parquet")):
            table = pq.read_table(shard_path)
            total_rows += len(table)
        assert total_rows == 36

    def test_sft_docs_distributed_uniformly(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        for shard_path in sorted(output_dir.glob("shard_*.parquet")):
            table = pq.read_table(shard_path)
            texts = table.column("text").to_pylist()
            sft_count = sum(1 for t in texts if "safety_q_" in t)
            assert sft_count == 2, f"{shard_path.name}: expected 2 SFT, got {sft_count}"

    def test_within_shard_shuffle(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        any_not_at_end = False
        for shard_path in sorted(output_dir.glob("shard_*.parquet")):
            table = pq.read_table(shard_path)
            texts = table.column("text").to_pylist()
            sft_positions = [i for i, t in enumerate(texts) if "safety_q_" in t]
            if sft_positions and sft_positions[-1] < len(texts) - 1:
                any_not_at_end = True
                break
        assert any_not_at_end, "SFT docs should be shuffled within shards"

    def test_idempotent_same_seed(self, data_dirs: tuple, tmp_path: Path) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        output_dir_2 = tmp_path / "mixed_data_2" / "safety"

        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
            seed=0,
        )
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir_2,
            seed=0,
        )

        for shard_name in [
            "shard_00000.parquet",
            "shard_00001.parquet",
            "shard_00002.parquet",
        ]:
            t1 = pq.read_table(output_dir / shard_name).column("text").to_pylist()
            t2 = pq.read_table(output_dir_2 / shard_name).column("text").to_pylist()
            assert t1 == t2

    def test_different_seed_different_output(
        self, data_dirs: tuple, tmp_path: Path
    ) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        output_dir_2 = tmp_path / "mixed_data_2" / "safety"

        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
            seed=0,
        )
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir_2,
            seed=999,
        )

        any_different = False
        for shard_name in [
            "shard_00000.parquet",
            "shard_00001.parquet",
            "shard_00002.parquet",
        ]:
            t1 = pq.read_table(output_dir / shard_name).column("text").to_pylist()
            t2 = pq.read_table(output_dir_2 / shard_name).column("text").to_pylist()
            if t1 != t2:
                any_different = True
                break
        assert any_different

    def test_metadata_written(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        meta_path = output_dir / "mixing_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            saved_meta = json.load(f)
        assert saved_meta["mixtures"] == ["safety"]
        assert saved_meta["model"] == "gemma-2-2b"
        assert saved_meta["seed"] == 0
        assert saved_meta["total_pretraining_docs"] == 30
        assert saved_meta["total_sft_docs"] == 6
        assert saved_meta["n_shards"] == 3
        assert abs(saved_meta["sft_fraction"] - 6 / 36) < 1e-9

    def test_multi_mixture(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "base_data"
        base_dir.mkdir()
        _make_pretraining_parquet(
            base_dir / "shard_00000.parquet", [f"doc_{i}" for i in range(10)]
        )

        sft_dir = tmp_path / "sft_data"
        for mix_name in ["safety", "capabilities"]:
            d = sft_dir / mix_name
            d.mkdir(parents=True)
            convs = [
                {
                    "messages": [
                        {"role": "user", "content": f"{mix_name}_q_{j}"},
                        {"role": "assistant", "content": f"{mix_name}_a_{j}"},
                    ],
                    "response": f"{mix_name}_a_{j}",
                }
                for j in range(3)
            ]
            _make_sft_parquet(d / "shard_00000.parquet", convs)

        output_dir = tmp_path / "mixed_data" / "capabilities+safety"
        meta = mix_sft_into_pretraining(
            mixtures=["safety", "capabilities"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        assert meta["total_sft_docs"] == 6
        assert meta["total_pretraining_docs"] == 10

        table = pq.read_table(output_dir / "shard_00000.parquet")
        assert len(table) == 16

    def test_no_pretraining_shards_raises(self, tmp_path: Path) -> None:
        empty_base = tmp_path / "empty_base"
        empty_base.mkdir()
        with pytest.raises(ValueError, match="No pretraining shards"):
            mix_sft_into_pretraining(
                mixtures=["safety"],
                model_name="gemma-2-2b",
                base_data_dir=empty_base,
                sft_data_dir=tmp_path,
                output_dir=tmp_path / "out",
            )

    def test_missing_sft_dir_raises(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "base_data"
        base_dir.mkdir()
        _make_pretraining_parquet(base_dir / "shard_00000.parquet", ["doc"])
        with pytest.raises(ValueError, match="SFT data directory not found"):
            mix_sft_into_pretraining(
                mixtures=["nonexistent"],
                model_name="gemma-2-2b",
                base_data_dir=base_dir,
                sft_data_dir=tmp_path / "sft_data",
                output_dir=tmp_path / "out",
            )

    def test_empty_sft_writes_pretraining_only(self, tmp_path: Path) -> None:
        base_dir = tmp_path / "base_data"
        base_dir.mkdir()
        _make_pretraining_parquet(base_dir / "shard_00000.parquet", ["doc0", "doc1"])

        sft_dir = tmp_path / "sft_data"
        safety_dir = sft_dir / "safety"
        safety_dir.mkdir(parents=True)
        table = pa.table(
            {
                "text": pa.array([], type=pa.string()),
                "original_index": pa.array([], type=pa.int64()),
            }
        )
        pq.write_table(table, safety_dir / "shard_00000.parquet")

        output_dir = tmp_path / "mixed_data" / "safety"
        meta = mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        assert meta["total_sft_docs"] == 0
        assert meta["total_pretraining_docs"] == 2

        out_table = pq.read_table(output_dir / "shard_00000.parquet")
        assert len(out_table) == 2

    def test_pretraining_docs_preserved_exactly(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        source_docs = set()
        for shard_path in sorted(base_dir.glob("shard_*.parquet")):
            table = pq.read_table(shard_path)
            source_docs.update(table.column("text").to_pylist())

        output_pretrain_docs = set()
        for shard_path in sorted(output_dir.glob("shard_*.parquet")):
            table = pq.read_table(shard_path)
            for text in table.column("text").to_pylist():
                if text.startswith("pretrain_doc_"):
                    output_pretrain_docs.add(text)

        assert source_docs == output_pretrain_docs

    def test_sft_fraction_accurate(self, data_dirs: tuple) -> None:
        base_dir, sft_dir, output_dir = data_dirs
        meta = mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="gemma-2-2b",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=output_dir,
        )
        expected_fraction = 6 / 36
        assert abs(meta["sft_fraction"] - expected_fraction) < 1e-9


class TestDefaultOutputName:
    def test_single_mixture(self) -> None:
        assert _default_output_name(["safety"]) == "safety"

    def test_multiple_mixtures_sorted(self) -> None:
        assert _default_output_name(["safety", "capabilities"]) == "capabilities+safety"
        assert _default_output_name(["capabilities", "safety"]) == "capabilities+safety"

    def test_three_mixtures(self) -> None:
        assert (
            _default_output_name(["standard", "safety", "capabilities"])
            == "capabilities+safety+standard"
        )
