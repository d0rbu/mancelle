"""Issue #521: conditioned SFT subscripts the feature store (no silent .get())."""

from __future__ import annotations

import pathlib
from typing import cast

import pytest

from steering.conditional.encoder_decoder.sft_features import SFTFeatureStore
from steering.conditional.feature_index import FeatureIndex


class _StubIndex:
    def lookup(self, shard_stem, doc_idx):
        raise KeyError(f"Document {doc_idx} not found in shard {shard_stem!r} (shard has 0 docs)")


def _corrupt_store():
    store = SFTFeatureStore.__new__(SFTFeatureStore)
    store._feature_index = cast(FeatureIndex, _StubIndex())
    store._locations = {5: ("shard_00000", 2)}
    return store


def test_subscript_access_is_loud_but_get_swallows():
    """store[idx] raises on a corrupt row; store.get(idx) would hide it as None."""
    store = _corrupt_store()
    with pytest.raises(KeyError, match="Document 2 not found"):
        _ = store[5]
    assert store.get(5) is None  # proves .get() would have hidden the corruption


def test_sft_runtime_uses_subscript_not_get():
    """Source guard: the conditioned-SFT runtime must NOT use .get() for the
    required per-example feature row (Mapping.get swallows KeyError)."""
    src = pathlib.Path("scripts/train/conditional_encdec_sft.py").read_text()
    assert "sft_features[example_idx]" in src
    assert "sft_features.get(example_idx)" not in src
