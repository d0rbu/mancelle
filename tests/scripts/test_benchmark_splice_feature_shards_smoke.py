from __future__ import annotations

import json

import pyarrow as pa
import pytest
from omegaconf import OmegaConf

from nanochat.atomic_io import atomic_write_parquet
from scripts.benchmark import benchmark_splice_feature_shards as bench
from scripts.preprocess import splice_feature_shards as sfs
from steering.feature_io import save_features
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

TOK = CharMergeTokenizer(bos_id=500)


@pytest.fixture(autouse=True)
def _patch_model(monkeypatch):
    def raw_tokenizer(_model, *, load_chat_template):
        assert load_chat_template is False
        return TOK

    monkeypatch.setattr(sfs, "resolve_tokenizer", raw_tokenizer)
    monkeypatch.setattr(sfs, "resolve_sae_max_context", lambda model: 8192)


def _make_base_dirs(tmp_path):
    feature_dir = tmp_path / "base"
    parquet_dir = tmp_path / "parquets"
    feature_dir.mkdir()
    parquet_dir.mkdir()
    texts_by_idx = {
        0: ["alpha beta", "gamma delta epsilon", "zeta eta"],
        1: ["one two three", "four five", "six seven eight nine"],
    }
    for idx, texts in texts_by_idx.items():
        save_features(make_base_shard(texts, TOK, 2), feature_dir / f"shard_{idx:05d}")
        atomic_write_parquet(pa.table({"text": texts}), parquet_dir / f"shard_{idx:05d}.parquet")
    return feature_dir, parquet_dir


def test_benchmark_splice_feature_shards_smoke(tmp_path):
    feature_dir, parquet_dir = _make_base_dirs(tmp_path)

    sidecar_dir = tmp_path / "sidecars"
    report_md = tmp_path / "report.md"
    report = bench.run_bench(OmegaConf.create({
        "feature_dir": str(feature_dir),
        "data_dir": str(parquet_dir),
        "max_shards": 1,
        "model": "gemma-2-2b",
        "segment_size": 2,
        "context_window": 2048,
        "cut_fraction": 0.6,
        "sidecar_dir": str(sidecar_dir),
        "report_md": str(report_md),
        "bench_root": str(tmp_path / "bench_root"),
    }))

    assert "Feature-Shard Splice Benchmark" in report
    assert "## Detail Timings" in report
    assert "| profile | save.zstd_compress |" in report
    assert report_md.read_text(encoding="utf-8") == report
    sidecar = sidecar_dir / "profile.json"
    assert sidecar.exists()
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    assert data["totals"]["n_ok"] == 1
    assert data["totals"]["n_failed"] == 0
    assert data["totals"]["detail_seconds"]["source.truncate_tokenize"] > 0.0


def test_benchmark_rejects_variant_output_escape(tmp_path):
    feature_dir, parquet_dir = _make_base_dirs(tmp_path)

    with pytest.raises(ValueError, match="output_dir"):
        bench.run_bench(OmegaConf.create({
            "feature_dir": str(feature_dir),
            "data_dir": str(parquet_dir),
            "max_shards": 1,
            "model": "gemma-2-2b",
            "segment_size": 2,
            "context_window": 2048,
            "bench_root": str(tmp_path / "bench_root"),
            "variants": [
                {
                    "name": "escape",
                    "overrides": {"output_dir": str(feature_dir)},
                }
            ],
        }))
