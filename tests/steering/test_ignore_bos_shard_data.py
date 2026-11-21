"""ShardData carries ignore_bos_features and round-trips via to_dict/from_dict."""

import torch as t

from steering.extract_features import ShardData


def _kw(**extra):
    kw = {
        "offsets": t.tensor([0, 1], dtype=t.int64),
        "feature_ids": t.tensor([3], dtype=t.int32),
        "feature_mags": t.tensor([1.5], dtype=t.float32),
        "doc_ids": t.tensor([0], dtype=t.int32),
        "segment_indices": t.tensor([0], dtype=t.int32),
        "segment_positions": t.tensor([0], dtype=t.int32),
        "doc_token_counts": t.tensor([1], dtype=t.int32),
        "n_docs": 1, "n_segments": 1, "layer": 8, "context_window": 2048,
        "pooling": "max_pool", "segment_size": 128,
        "segment_char_starts": t.tensor([0], dtype=t.int32),
        "segment_char_ends": t.tensor([1], dtype=t.int32),
    }
    kw.update(extra)
    return kw


def test_default_false():
    assert ShardData(**_kw()).ignore_bos_features is False


def test_roundtrip_true():
    sd = ShardData(**_kw(ignore_bos_features=True))
    d = sd.to_dict()
    assert d["ignore_bos_features"] is True
    assert ShardData.from_dict(d).ignore_bos_features is True
