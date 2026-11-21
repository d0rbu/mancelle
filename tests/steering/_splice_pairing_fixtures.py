"""Disk-backed two-feature corpus for independent splice pairing integration tests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pyarrow as pa
import torch as t
from omegaconf import DictConfig

from nanochat.atomic_io import atomic_write_parquet
from scripts.preprocess import splice_feature_shards as sfs
from steering.compute_statistics import compute_statistics
from steering.feature_io import save_features
from steering.stats_types import ExtractionStatistics
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer

PAIRING_TOKENIZER = CharMergeTokenizer(bos_id=500)
PAIRING_TEXTS = [char * 7 for char in "abcdefgh"]
PAIRING_WIDTH = 8


@dataclass(frozen=True)
class PairingCorpus:
    root: Path
    feature_dir: Path
    data_dir: Path
    statistics: ExtractionStatistics | None


def build_pairing_corpus(root: Path, *, statistics: bool = True) -> PairingCorpus:
    """Eight two-segment docs per shard; each doc's two segments share a feature.

    Four docs use feature 0, four use feature 1. Pairing across families is
    strictly more distant than pairing within one family, and all token seams
    are stable. Three shard copies exercise worker/rank scheduling and the
    normal excluded-last-shard statistics convention.
    """
    feature_dir = root / "features" / "base"
    data_dir = root / "parquets"
    feature_dir.mkdir(parents=True)
    data_dir.mkdir()
    for index in range(3):
        shard = make_base_shard(
            PAIRING_TEXTS, PAIRING_TOKENIZER, 4, context_window=32, n_features=PAIRING_WIDTH
        )
        assert int(shard["n_segments"]) == 16
        shard["offsets"] = t.arange(17, dtype=t.int64)
        shard["feature_ids"] = t.tensor(
            [doc % 2 for doc in range(8) for _ in range(2)], dtype=t.int32
        )
        shard["feature_mags"] = t.tensor(
            [1.0 + doc // 2 for doc in range(8) for _ in range(2)], dtype=t.float32
        )
        save_features(shard, feature_dir / f"shard_{index:05d}")
        atomic_write_parquet(
            pa.table({"text": PAIRING_TEXTS}), data_dir / f"shard_{index:05d}.parquet"
        )
    stats = (
        compute_statistics(
            str(feature_dir),
            str(feature_dir / "feature_statistics.safetensors"),
            n_sae_features=PAIRING_WIDTH,
            exclude_last_shard=True,
            device="cpu",
            allow_degenerate_statistics=True,
        )
        if statistics
        else None
    )
    return PairingCorpus(root, feature_dir, data_dir, stats)


def pairing_run_config(corpus: PairingCorpus, output_dir: Path | None, **overrides) -> DictConfig:
    return sfs.load_splice_config(
        {
            "feature_dir": str(corpus.feature_dir),
            "data_dir": str(corpus.data_dir),
            "output_dir": None if output_dir is None else str(output_dir),
            "model": "gemma-2-2b",
            "segment_size": 4,
            "context_window": 32,
            "cut_fraction": 1.0,
            "pairing_mode": "opposite_percentile",
            "pairing_candidate_pool_size": 8,
            **overrides,
        }
    )


def snapshot_files(directory: Path) -> dict[str, bytes]:
    return {
        str(path.relative_to(directory)): path.read_bytes()
        for path in directory.rglob("*")
        if path.is_file()
    }
