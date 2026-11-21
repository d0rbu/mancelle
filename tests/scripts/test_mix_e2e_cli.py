"""End-to-end CLI test: exercises the Hydra entry point with synthetic data.

Creates fake pretraining + SFT parquets, invokes the script via subprocess,
and verifies the output parquets and metadata are correct.
"""

import json
import subprocess
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _make_pretraining_shards(base_dir: Path, n_shards: int, docs_per_shard: int):
    base_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        texts = [f"pretrain_s{i}_d{j}" for j in range(docs_per_shard)]
        table = pa.table({"text": texts})
        pq.write_table(table, base_dir / f"shard_{i:05d}.parquet")


def _make_sft_shards(sft_dir: Path, mixture_name: str, n_convs: int):
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


N_SHARDS = 3
DOCS_PER_SHARD = 20
N_CAP = 10
N_SAF = 8


@pytest.fixture()
def cli_workspace(tmp_path: Path):
    base_dir = tmp_path / "base_data"
    sft_dir = tmp_path / "sft_data"
    _make_pretraining_shards(base_dir, N_SHARDS, DOCS_PER_SHARD)
    _make_sft_shards(sft_dir, "capabilities", N_CAP)
    _make_sft_shards(sft_dir, "safety", N_SAF)
    return base_dir, sft_dir, tmp_path


def _run_cli(base_dir, sft_dir, output_dir, mixtures_str, model="gemma-2-2b"):
    """Run the mixing script via subprocess and return (returncode, stdout, stderr)."""
    cmd = [
        sys.executable,
        "-m",
        "scripts.preprocess.mix_sft_into_pretraining",
        f"mixtures={mixtures_str}",
        f"model={model}",
        f"base_data_dir={base_dir}",
        f"sft_data_dir={sft_dir}",
        f"output_dir={output_dir}",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=120, check=False)
    return result


class TestCLICapabilities:
    def test_cli_capabilities_only(self, cli_workspace):
        base_dir, sft_dir, root = cli_workspace
        out = root / "out_cap"
        result = _run_cli(base_dir, sft_dir, out, "[capabilities]")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        # Verify output
        assert (out / "mixing_meta.json").exists()
        meta = json.loads((out / "mixing_meta.json").read_text())
        assert meta["mixtures"] == ["capabilities"]
        assert meta["total_sft_docs"] == N_CAP
        assert meta["total_pretraining_docs"] == N_SHARDS * DOCS_PER_SHARD

        # Count docs in output
        total = 0
        cap_count = 0
        for shard in sorted(out.glob("shard_*.parquet")):
            texts = pq.read_table(shard).column("text").to_pylist()
            total += len(texts)
            cap_count += sum(1 for t in texts if "[capabilities]" in t)

        assert total == N_SHARDS * DOCS_PER_SHARD + N_CAP
        assert cap_count == N_CAP


class TestCLISafety:
    def test_cli_safety_only(self, cli_workspace):
        base_dir, sft_dir, root = cli_workspace
        out = root / "out_saf"
        result = _run_cli(base_dir, sft_dir, out, "[safety]")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        meta = json.loads((out / "mixing_meta.json").read_text())
        assert meta["mixtures"] == ["safety"]
        assert meta["total_sft_docs"] == N_SAF

        total = 0
        saf_count = 0
        for shard in sorted(out.glob("shard_*.parquet")):
            texts = pq.read_table(shard).column("text").to_pylist()
            total += len(texts)
            saf_count += sum(1 for t in texts if "[safety]" in t)

        assert total == N_SHARDS * DOCS_PER_SHARD + N_SAF
        assert saf_count == N_SAF


class TestCLIBoth:
    def test_cli_both_mixtures(self, cli_workspace):
        base_dir, sft_dir, root = cli_workspace
        out = root / "out_both"
        result = _run_cli(base_dir, sft_dir, out, "[capabilities,safety]")
        assert result.returncode == 0, f"Script failed:\n{result.stderr}"

        meta = json.loads((out / "mixing_meta.json").read_text())
        assert set(meta["mixtures"]) == {"capabilities", "safety"}
        assert meta["total_sft_docs"] == N_CAP + N_SAF

        total = 0
        cap_count = 0
        saf_count = 0
        for shard in sorted(out.glob("shard_*.parquet")):
            texts = pq.read_table(shard).column("text").to_pylist()
            total += len(texts)
            cap_count += sum(1 for t in texts if "[capabilities]" in t)
            saf_count += sum(1 for t in texts if "[safety]" in t)

        assert total == N_SHARDS * DOCS_PER_SHARD + N_CAP + N_SAF
        assert cap_count == N_CAP
        assert saf_count == N_SAF

        # Verify SFT fraction
        expected_frac = (N_CAP + N_SAF) / total
        assert abs(meta["sft_fraction"] - expected_frac) < 1e-9
