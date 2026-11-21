"""T-S4: packed-row/selection discovery extraction parity."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch as t

from scripts.train.conditional_encdec_train import (
    _load_packed_row_checksums,
    discover_packed_selection,
)
from steering.packed_batch_io import (
    compute_selection_hash,
    save_packed_row_shard,
    save_selection_shard,
)
from steering.stats_io import save_percentile_checksum
from tests.encdec_test_helpers import _make_identity_selection, _make_test_shard
from tests.scripts.test_checksum_pipeline_wiring import _make_stats_file

_PERCENTILE_CHECKSUM = "1" * 64
_TOKENIZER_CHECKSUM = "2" * 64
_FEATURE_SELECTION = {"threshold_type": "segment_topk", "threshold": 3}
_NUM_BUCKETS = 4
_SEED = 17
_CONTEXT_LEN = 64
_SEGMENT_SIZE = 16
_N_SAE_FEATURES = 200


def _write_smallest_discovery_tree(
    feature_dir: Path,
    *,
    include_manifest: bool,
) -> tuple[Path, Path, str]:
    packed_dir = (
        feature_dir
        / "pct"
        / _PERCENTILE_CHECKSUM[:16]
        / "tok"
        / _TOKENIZER_CHECKSUM[:16]
        / f"c{_CONTEXT_LEN}"
        / "sall"
    )
    shard = _make_test_shard(
        n_rows=1,
        context_len=_CONTEXT_LEN,
        segs_per_row=2,
        feats_per_seg=3,
        n_features=_N_SAE_FEATURES,
        seed=4,
    )
    assert shard.tokens.dtype is t.uint16
    save_packed_row_shard(shard, packed_dir / "shard_00000")
    if include_manifest:
        (packed_dir / "manifest.json").write_text(
            json.dumps(
                {
                    "checksum": "packed-test-checksum",
                    "metadata": {"max_seg_size": 37},
                }
            )
        )

    selection_hash = compute_selection_hash(
        positive_selection=_FEATURE_SELECTION,
        negative_selection=None,
        delta_config=None,
        num_buckets=_NUM_BUCKETS,
        seed=_SEED,
        ddp_rank=0,
        feature_mask=None,
        feature_clusters=None,
    )
    selection_dir = packed_dir / "sel" / selection_hash
    save_selection_shard(
        _make_identity_selection(shard),
        selection_dir / "shard_00000",
    )
    return packed_dir, selection_dir, selection_hash


def _discover(feature_dir: Path):
    return discover_packed_selection(
        feature_dir=feature_dir,
        original_val_feature_dir=None,
        feature_selection=_FEATURE_SELECTION,
        negative_selection=None,
        feature_delta=None,
        num_buckets=_NUM_BUCKETS,
        seed=_SEED,
        feature_mask=None,
        feature_clusters=None,
        n_sae_features=_N_SAE_FEATURES,
        use_precomputed_batches=True,
        percentile_checksum=_PERCENTILE_CHECKSUM,
        tokenizer_checksum=_TOKENIZER_CHECKSUM,
        context_len=_CONTEXT_LEN,
        segment_size=_SEGMENT_SIZE,
        max_train_shards=None,
        compile_static=False,
        fp8=False,
        eval_every=0,
        original_val_data_dir=None,
        boundary_fingerprint=None,
    )


def test_discovery_returns_every_inline_result_for_smallest_valid_tree(
    tmp_path: Path,
) -> None:
    feature_dir = tmp_path / "features"
    packed_dir, selection_dir, selection_hash = _write_smallest_discovery_tree(
        feature_dir,
        include_manifest=True,
    )

    result = _discover(feature_dir)

    assert result.train_packed_dir == packed_dir
    assert result.train_selection_dir == selection_dir
    assert result.val_packed_dir is None
    assert result.val_selection_dir is None
    assert result.original_val_feature_dir is None
    assert result.original_val_packed_dir is None
    assert result.original_val_selection_dir is None
    assert result.original_val_feature_index is None
    assert result.spliced_val_feature_index is None
    assert result.max_seg_size == 37
    assert result.pad_n_seg is None
    assert result.pad_f is None
    assert result.selection_hash == selection_hash


def test_missing_manifest_preserves_inline_none_tolerance(tmp_path: Path) -> None:
    feature_dir = tmp_path / "features"
    packed_dir, selection_dir, selection_hash = _write_smallest_discovery_tree(
        feature_dir,
        include_manifest=False,
    )

    result = _discover(feature_dir)

    assert result.train_packed_dir == packed_dir
    assert result.train_selection_dir == selection_dir
    assert result.val_packed_dir is None
    assert result.val_selection_dir is None
    assert result.original_val_feature_dir is None
    assert result.original_val_packed_dir is None
    assert result.original_val_selection_dir is None
    assert result.original_val_feature_index is None
    assert result.spliced_val_feature_index is None
    assert result.max_seg_size is None
    assert result.pad_n_seg is None
    assert result.pad_f is None
    assert result.selection_hash == selection_hash


def test_checksum_loader_fails_loud_when_tokenizer_unresolvable(tmp_path: Path) -> None:
    """The extracted helper raises a typed, actionable error (not a bare assert)
    when no tokenizer checksum can be resolved — pinning the fail-loud branch
    that the discovery extraction rewrote (was `assert tokenizer is not None`)."""
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    stats_path = _make_stats_file(feature_dir)
    save_percentile_checksum(stats_path, _PERCENTILE_CHECKSUM)
    # No tokenizer checksum saved and no fallback ⇒ unresolvable.
    with pytest.raises(ValueError, match="tokenizer"):
        _load_packed_row_checksums(
            feature_dir,
            fallback_tokenizer_checksum=None,
        )
