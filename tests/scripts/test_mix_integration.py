"""Integration test: run the mixing pipeline with capabilities, safety, and both.

Creates synthetic pretraining + SFT data, runs three mixing scenarios, and
verifies:
  1. Output shard count matches input.
  2. Total doc count = pretrain + SFT.
  3. SFT fraction in metadata matches actual frequency in data.
  4. SFT docs are present and recognisable in output shards.
  5. Both mixture sources appear when mixing two at once.
"""

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from scripts.preprocess.mix_sft_into_pretraining import (
    mix_sft_into_pretraining,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pretraining_shards(base_dir: Path, n_shards: int, docs_per_shard: int):
    """Create n_shards pretraining parquets with recognisable text."""
    base_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        texts = [f"pretrain_s{i}_d{j}" for j in range(docs_per_shard)]
        table = pa.table({"text": texts})
        pq.write_table(table, base_dir / f"shard_{i:05d}.parquet")


def _make_sft_shards(sft_dir: Path, mixture_name: str, n_convs: int):
    """Create SFT parquets with tagged conversation content."""
    mix_dir = sft_dir / mixture_name
    mix_dir.mkdir(parents=True, exist_ok=True)
    texts = []
    indices = []
    for j in range(n_convs):
        texts.append(
            f"<user>[{mixture_name}] question {j}</user>"
            f"<assistant>[{mixture_name}] answer {j}</assistant>"
        )
        indices.append(j)
    table = pa.table(
        {
            "text": texts,
            "original_index": indices,
        }
    )
    pq.write_table(table, mix_dir / "shard_00000.parquet")


def _count_docs_matching(output_dir: Path, pattern: str) -> int:
    """Count documents across all output shards that contain *pattern*."""
    count = 0
    for shard in sorted(output_dir.glob("shard_*.parquet")):
        texts = pq.read_table(shard).column("text").to_pylist()
        count += sum(1 for t in texts if pattern in t)
    return count


def _total_docs(output_dir: Path) -> int:
    total = 0
    for shard in sorted(output_dir.glob("shard_*.parquet")):
        total += len(pq.read_table(shard))
    return total


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_SHARDS = 4
DOCS_PER_SHARD = 50
N_CAP_CONVS = 20  # capabilities SFT conversations
N_SAF_CONVS = 12  # safety SFT conversations


@pytest.fixture()
def workspace(tmp_path: Path):
    base_dir = tmp_path / "base_data"
    sft_dir = tmp_path / "sft_data"
    _make_pretraining_shards(base_dir, N_SHARDS, DOCS_PER_SHARD)
    _make_sft_shards(sft_dir, "capabilities", N_CAP_CONVS)
    _make_sft_shards(sft_dir, "safety", N_SAF_CONVS)
    return base_dir, sft_dir, tmp_path


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestCapabilitiesOnly:
    def test_capabilities_mix(self, workspace):
        base_dir, sft_dir, root = workspace
        out = root / "mixed_data" / "capabilities"
        meta = mix_sft_into_pretraining(
            mixtures=["capabilities"],
            model_name="fake-model",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=out,
        )

        total_pretrain = N_SHARDS * DOCS_PER_SHARD  # 200
        total_sft = N_CAP_CONVS  # 20
        total = total_pretrain + total_sft  # 220

        # Metadata checks
        assert meta["total_pretraining_docs"] == total_pretrain
        assert meta["total_sft_docs"] == total_sft
        assert meta["total_docs"] == total
        assert meta["mixtures"] == ["capabilities"]

        # Shard count preserved
        assert len(list(out.glob("shard_*.parquet"))) == N_SHARDS

        # Doc count matches
        assert _total_docs(out) == total

        # SFT docs recognisable
        cap_count = _count_docs_matching(out, "[capabilities]")
        assert cap_count == N_CAP_CONVS

        # SFT fraction
        expected_frac = total_sft / total
        assert abs(meta["sft_fraction"] - expected_frac) < 1e-9

        # No safety docs leaked in
        saf_count = _count_docs_matching(out, "[safety]")
        assert saf_count == 0


class TestSafetyOnly:
    def test_safety_mix(self, workspace):
        base_dir, sft_dir, root = workspace
        out = root / "mixed_data" / "safety"
        meta = mix_sft_into_pretraining(
            mixtures=["safety"],
            model_name="fake-model",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=out,
        )

        total_pretrain = N_SHARDS * DOCS_PER_SHARD
        total_sft = N_SAF_CONVS
        total = total_pretrain + total_sft

        assert meta["total_pretraining_docs"] == total_pretrain
        assert meta["total_sft_docs"] == total_sft
        assert meta["total_docs"] == total

        assert _total_docs(out) == total
        assert _count_docs_matching(out, "[safety]") == N_SAF_CONVS
        assert _count_docs_matching(out, "[capabilities]") == 0

        expected_frac = total_sft / total
        assert abs(meta["sft_fraction"] - expected_frac) < 1e-9


class TestBothMixtures:
    def test_capabilities_plus_safety(self, workspace):
        base_dir, sft_dir, root = workspace
        out = root / "mixed_data" / "capabilities+safety"
        meta = mix_sft_into_pretraining(
            mixtures=["capabilities", "safety"],
            model_name="fake-model",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=out,
        )

        total_pretrain = N_SHARDS * DOCS_PER_SHARD  # 200
        total_sft = N_CAP_CONVS + N_SAF_CONVS  # 32
        total = total_pretrain + total_sft  # 232

        # Metadata
        assert meta["total_pretraining_docs"] == total_pretrain
        assert meta["total_sft_docs"] == total_sft
        assert meta["total_docs"] == total

        # Doc counts
        assert _total_docs(out) == total

        # Both mixture sources present
        cap_count = _count_docs_matching(out, "[capabilities]")
        saf_count = _count_docs_matching(out, "[safety]")
        assert cap_count == N_CAP_CONVS, (
            f"Expected {N_CAP_CONVS} capabilities, got {cap_count}"
        )
        assert saf_count == N_SAF_CONVS, (
            f"Expected {N_SAF_CONVS} safety, got {saf_count}"
        )

        # SFT fraction
        expected_frac = total_sft / total
        assert abs(meta["sft_fraction"] - expected_frac) < 1e-9

        # Verify shards
        assert len(list(out.glob("shard_*.parquet"))) == N_SHARDS

    def test_sft_distributed_across_shards(self, workspace):
        """SFT docs should be spread across shards, not clumped."""
        base_dir, sft_dir, root = workspace
        out = root / "mixed_data" / "dist_check"
        mix_sft_into_pretraining(
            mixtures=["capabilities", "safety"],
            model_name="fake-model",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=out,
        )

        total_sft = N_CAP_CONVS + N_SAF_CONVS  # 32
        expected_per_shard = total_sft / N_SHARDS  # 8.0

        sft_per_shard = []
        for shard in sorted(out.glob("shard_*.parquet")):
            texts = pq.read_table(shard).column("text").to_pylist()
            sft_count = sum(
                1 for t in texts if "[capabilities]" in t or "[safety]" in t
            )
            sft_per_shard.append(sft_count)

        # Each shard should have within ±1 of the expected count
        for i, count in enumerate(sft_per_shard):
            assert abs(count - expected_per_shard) <= 1, (
                f"Shard {i}: expected ~{expected_per_shard} SFT docs, got {count}"
            )

    def test_metadata_file_written(self, workspace):
        base_dir, sft_dir, root = workspace
        out = root / "mixed_data" / "meta_check"
        mix_sft_into_pretraining(
            mixtures=["capabilities", "safety"],
            model_name="fake-model",
            base_data_dir=base_dir,
            sft_data_dir=sft_dir,
            output_dir=out,
        )

        meta_path = out / "mixing_meta.json"
        assert meta_path.exists()
        with open(meta_path) as f:
            saved = json.load(f)
        assert set(saved["mixtures"]) == {"capabilities", "safety"}
        assert saved["n_shards"] == N_SHARDS
        assert "timestamp" in saved
        assert "sft_fraction" in saved
