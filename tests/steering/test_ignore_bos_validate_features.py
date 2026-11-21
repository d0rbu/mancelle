"""validate_shard_activations forwards ignore_bos_features to re-extraction."""

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

import steering.validate_features as vf
from steering.extract_features import PoolingMode, process_shard
from steering.feature_io import load_features, save_features
from steering.validate_features import validate_feature_directory, validate_shard_activations
from tests.steering.test_context_independent_extraction import (
    _make_parquet,
    _tiny_model_sae,
)


def _save_minimal_shard(path, *, ignore_bos_features: bool):
    return save_features({
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
        "n_docs": 1,
        "n_segments": 1,
        "layer": 8,
        "context_window": 2048,
        "pooling": "max_pool",
        "segment_mode": "fixed_tokens",
        "segment_size": 128,
        "ignore_bos_features": ignore_bos_features,
    }, path)


def test_validate_feature_directory_rejects_mixed_ignore_bos(tmp_path):
    _save_minimal_shard(tmp_path / "shard_00000_features", ignore_bos_features=True)
    _save_minimal_shard(tmp_path / "shard_00001_features", ignore_bos_features=False)

    result = validate_feature_directory(tmp_path)

    assert not result.valid
    assert any("ignore_bos_features" in v for v in result.violations)


def test_ignore_bos_flag_forwarded(monkeypatch, tmp_path):
    pq.write_table(pa.table({"text": ["hello world this is a doc"]}),
                   tmp_path / "shard_00000.parquet")

    captured = {}

    def fake_extract(model, sae, text, **kw):
        captured["ignore_bos_features"] = kw.get("ignore_bos_features")
        return (
            [t.tensor([0], dtype=t.int32)], [t.tensor([1.0])],
            [0], [0], [1], 1,
        )

    monkeypatch.setattr(vf, "extract_document_features", fake_extract)

    shard_data = {
        "n_docs": 1,
        "context_independent_features": False,
        "ignore_bos_features": True,
        "doc_token_counts": t.tensor([5], dtype=t.int32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([0], dtype=t.int32),
        "feature_mags": t.tensor([1.0], dtype=t.float32),
    }
    vf.validate_shard_activations(
        shard_data, tmp_path / "shard_00000.parquet",
        model=object(), sae=object(), layer=8, context_window=2048,
        segment_size=128, pooling="max_pool", n_spot_check=1,
    )
    assert captured["ignore_bos_features"] is True


def test_validate_shard_activations_real_stub_ignore_bos(tmp_path):
    """Real-stub end-to-end: process_shard with ignore_bos_features=True produces
    a shard that validate_shard_activations re-extracts identically (self-consistent).

    This proves the flag is not merely forwarded but produces genuine self-consistent
    re-extraction: the stored features match a fresh extraction with the same flag.
    """
    model, sae = _tiny_model_sae()
    parquet_path = str(tmp_path / "shard.parquet")
    _make_parquet(parquet_path, ["abcdefgh", "ijklmnop"])
    result = process_shard(
        parquet_path, model, sae,
        layer=8, output_dir=str(tmp_path),
        context_window=128, pooling=PoolingMode.max_pool,
        segment_size=4, gpu_batch_size=2, cpu_batch_size=16,
        ignore_bos_features=True,
    )
    shard_data = load_features(result.output_path)
    assert shard_data["ignore_bos_features"] is True
    vr = validate_shard_activations(
        shard_data,
        parquet_path=parquet_path,
        model=model, sae=sae,
        layer=shard_data["layer"],
        context_window=shard_data["context_window"],
        segment_size=shard_data["segment_size"],
        pooling=shard_data["pooling"],
        n_spot_check=2,
    )
    assert vr.valid, vr.violations
