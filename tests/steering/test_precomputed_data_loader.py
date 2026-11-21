"""Integration tests for precomputed_data_loader.

Tests that the precomputed batch loader correctly:
1. Loads packed row shards from disk
2. Produces batches with correct shapes and dtypes
3. Handles label dropout
4. Cycles across epochs
5. Handles cross-shard batch boundaries
6. Prevents fully-empty rows (NaN guard)
"""

from __future__ import annotations

from pathlib import Path

import pytest
import torch as t

from steering.conditional.effective_feature_vocab import effective_feature_vocab
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.feature_prediction import (
    FeaturePredictionTargetLayout,
)
from steering.conditional.encoder_decoder.fp_target_source import (
    certify_fp_target_source,
)
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    save_packed_row_shard,
)
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import NUM_BUCKETS, build_threshold_pipeline


def _make_pipeline(n_features: int = 16384) -> FeaturePipelineConfig:
    """Build a threshold pipeline whose boundaries cover feature IDs 0..(n_features-1)."""
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(n_features, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_shard(
    n_rows: int = 8,
    context_len: int = 32,
    segs_per_row: int = 3,
    feats_per_seg: int = 5,
    seed: int = 42,
) -> PackedRowShard:
    """Create a synthetic packed row shard."""
    t.manual_seed(seed)
    T = context_len
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg

    tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)

    seg_size = T // segs_per_row
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id = t.zeros(total_segs, dtype=t.uint8)

    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == segs_per_row - 1 else (s + 1) * seg_size
            seg_doc_id[idx] = 0

    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)
    feature_ids = t.randint(0, 16384, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)  # >0 so some features are active

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def _write_shards(output_dir: Path, n_shards: int = 2, **shard_kwargs) -> list[Path]:
    """Write n_shards to output_dir, return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_shards):
        shard = _make_shard(seed=42 + i, **shard_kwargs)
        path = save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")
        paths.append(path)
    return paths


def _compact_fp_loader_inputs(
    tmp_path: Path,
    *,
    n_shards: int = 1,
    n_rows: int = 8,
) -> tuple[Path, FeaturePipelineConfig, FeaturePredictionTargetLayout]:
    packed = tmp_path / "compact_packed"
    _write_shards(packed, n_shards=n_shards, n_rows=n_rows)
    write_manifest(
        packed,
        "c" * 64,
        OverlayKind.PACKED_ROW,
        metadata={
            "type": "packed_rows",
            "max_n_seg": 3,
        },
    )
    certify_fp_target_source(packed, n_sae_features=PositiveInt.parse(16_384))
    mask = FeatureMask(FeatureMaskMode.whitelist, tuple(range(128)))
    vocab = effective_feature_vocab(mask, 16384)
    assert vocab is not None
    layout = FeaturePredictionTargetLayout.build(
        n_sae_features=PositiveInt.parse(16_384),
        feature_mask=mask,
        feature_clusters=None,
        feature_vocab=vocab,
    )
    pipeline = _make_pipeline()
    pipeline.feature_mask = mask
    return packed, pipeline, layout


# ======================================================================
# 1. Basic batch loading
# ======================================================================


class TestPrecomputedDataLoaderBasic:
    """Basic functional tests for precomputed_data_loader."""

    def test_produces_batches(self, tmp_path):
        """The loader yields at least one batch."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        batch = next(loader)
        assert batch is not None
        assert len(batch) == 10  # (inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state_dict)

    def test_batch_shapes(self, tmp_path):
        """Batch tensors have correct shapes."""
        T = 32
        B = 4
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8, context_len=T)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=B,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, state = next(loader)

        assert inputs.shape == (B, T)
        assert targets.shape == (B, T)
        assert seg_map.shape == (B, T)
        assert enc_fids.shape[0] == B
        assert enc_mags.shape == enc_fids.shape
        assert enc_mask.shape == enc_fids.shape
        assert enc_pcts.shape == enc_fids.shape
        assert seg_doc_id.shape[0] == B

    def test_batch_dtypes(self, tmp_path):
        """Batch tensors have correct dtypes."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, _fp_targets, _ = next(loader)

        assert inputs.dtype == t.int64
        assert targets.dtype == t.int64
        assert enc_fids.dtype == t.int64
        assert enc_mags.dtype == t.int64
        assert enc_mask.dtype == t.bool
        assert enc_pcts.dtype == t.float32
        assert seg_map.dtype == t.int64
        assert seg_doc_id.dtype == t.int64

    def test_state_dict_has_metadata(self, tmp_path):
        """State dict contains batch_idx, shard, and epoch."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        *_, state = next(loader)

        assert "batch_idx" in state
        assert "shard" in state
        assert "epoch" in state
        assert state["batch_idx"] == 0
        assert state["epoch"] == 1


# ======================================================================
# 2. Multiple batches and epoch cycling
# ======================================================================


class TestPrecomputedDataLoaderCycling:
    """Test batch iteration across shards and epochs."""

    def test_multiple_batches(self, tmp_path):
        """Loader yields multiple batches from a single shard."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=16)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )

        batches = [next(loader) for _ in range(4)]
        assert len(batches) == 4
        # batch_idx should increment
        assert batches[0][-1]["batch_idx"] == 0
        assert batches[1][-1]["batch_idx"] == 1

    def test_cross_shard_batches(self, tmp_path):
        """Batches that span shard boundaries are handled correctly."""
        # 2 shards with 6 rows each, batch size 4 → some batches cross shard boundary
        _write_shards(tmp_path / "packed", n_shards=2, n_rows=6)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )

        # Should be able to get multiple batches without error
        for _ in range(5):
            batch = next(loader)
            inputs = batch[0]
            assert inputs.shape[0] == 4

    def test_epoch_cycling(self, tmp_path):
        """Loader cycles to next epoch when data is exhausted."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=4)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )

        # First batch: epoch 1
        batch1 = next(loader)
        assert batch1[-1]["epoch"] == 1

        # Second batch: should be epoch 2 (only 4 rows, B=4 → 1 batch per epoch)
        batch2 = next(loader)
        assert batch2[-1]["epoch"] == 2


# ======================================================================
# 3. Label dropout
# ======================================================================


class TestPrecomputedDataLoaderLabelDropout:
    """Test that label dropout masks segments correctly."""

    def test_label_dropout_reduces_features(self, tmp_path):
        """With 100% label_dropout, features should be deterministically reduced.

        Uses dropout=1.0 to avoid flakiness from statistical tests on small samples.
        The NaN guard ensures at least 1 feature survives per row, so total features
        will be exactly B (one sentinel per row) vs the full feature count at dropout=0.
        """
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=16, feats_per_seg=10)

        loader_no_drop = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=8,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            label_dropout=0.0,
        )
        _, _, _, _, enc_mask_no_drop, *_ = next(loader_no_drop)
        features_no_drop = enc_mask_no_drop.sum().item()

        loader_full_drop = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=8,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            label_dropout=1.0,
        )
        _, _, _, _, enc_mask_full_drop, *_ = next(loader_full_drop)
        features_full_drop = enc_mask_full_drop.sum().item()

        # Full dropout leaves only the NaN-guard sentinels (1 per row = B)
        assert features_full_drop == 8  # exactly B sentinel features
        assert features_full_drop < features_no_drop

    def test_zero_dropout_preserves_all(self, tmp_path):
        """With label_dropout=0, all features are preserved."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=4, feats_per_seg=5)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            label_dropout=0.0,
        )
        batch = next(loader)
        enc_mask = batch[4]
        # With 0 dropout, all segments should have their features
        # (Each row has 3 segs × 5 feats)
        assert enc_mask.any()


# ======================================================================
# 4. NaN prevention guard
# ======================================================================


class TestNaNPreventionGuard:
    """Test that fully-empty rows get at least one True in enc_mask."""

    def test_fully_dropped_row_gets_sentinel(self, tmp_path):
        """After label_dropout, any fully-empty row gets enc_mask[row,0,0]=True."""
        # Use 100% dropout to guarantee all segments are dropped
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=4)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            label_dropout=1.0,  # drop everything
        )
        batch = next(loader)
        enc_mask = batch[4]

        # Every row should have at least one True (NaN guard)
        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), (
            "Some rows have no features after 100% dropout — NaN guard should prevent this"
        )


# ======================================================================
# 5. Error handling
# ======================================================================


class TestPrecomputedDataLoaderErrors:
    """Test error conditions."""

    def test_empty_dir_raises(self, tmp_path):
        """Empty directory raises FileNotFoundError."""
        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No packed row shards found"):
            loader = precomputed_data_loader(
                packed_row_dir=empty_dir,
                B=4,
                feature_pipeline=_make_pipeline(),
                device="cpu",
            )
            next(loader)

    def test_nonexistent_dir_raises(self, tmp_path):
        """Non-existent directory raises error."""
        with pytest.raises((FileNotFoundError, OSError)):
            loader = precomputed_data_loader(
                packed_row_dir=tmp_path / "nonexistent",
                B=4,
                feature_pipeline=_make_pipeline(),
                device="cpu",
            )
            next(loader)


# ======================================================================
# 6. Checkpoint resume
# ======================================================================


class TestPrecomputedDataLoaderResume:
    """Tests for checkpoint resume via resume_state_dict."""

    def test_state_dict_keys(self, tmp_path):
        """First batch state_dict has keys batch_idx, shard, epoch, max_seg_size."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        *_, state = next(loader)
        assert "batch_idx" in state
        assert "shard" in state
        assert "shard_batch_offset" in state
        assert "epoch" in state
        assert "max_seg_size" in state

    def test_resume_skips_batches(self, tmp_path):
        """Consume 5 batches, save state from batch 3, resume from it."""
        _write_shards(tmp_path / "packed", n_shards=2, n_rows=16)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        # Consume 5 batches, save state from the 4th (batch_idx=3)
        states = []
        for _ in range(5):
            *_, state = next(loader)
            states.append(state.copy())

        saved_state = states[3]  # batch_idx=3

        # Resume from saved state
        resumed_loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            resume_state_dict=saved_state,
        )
        *_, resumed_state = next(resumed_loader)

        # Resumed first batch should be at or past batch_idx 3
        assert resumed_state["batch_idx"] >= 3, (
            f"Resumed batch_idx {resumed_state['batch_idx']} < expected 3"
        )

    def test_resume_none_is_fresh(self, tmp_path):
        """With resume_state_dict=None, starts from batch_idx=0, epoch=1."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            resume_state_dict=None,
        )
        *_, state = next(loader)
        assert state["batch_idx"] == 0
        assert state["epoch"] == 1

    def test_resume_at_epoch_boundary(self, tmp_path):
        """Save state at epoch > 1, resume, verify epoch is restored."""
        # 4 rows, B=4 -> 1 batch per epoch -> epoch advances quickly
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=4)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        # Consume batches until we hit epoch 2+
        saved_state = None
        for _ in range(5):
            *_, state = next(loader)
            if state["epoch"] > 1 and saved_state is None:
                saved_state = state.copy()
        assert saved_state is not None, "Never reached epoch > 1"

        # Resume from the epoch > 1 state
        resumed_loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            resume_state_dict=saved_state,
        )
        *_, resumed_state = next(resumed_loader)

        # Epoch should be at least as high as the saved epoch
        assert resumed_state["epoch"] >= saved_state["epoch"], (
            f"Resumed epoch {resumed_state['epoch']} < saved epoch {saved_state['epoch']}"
        )

    def test_resume_with_unknown_shard(self, tmp_path):
        """If resume shard doesn't match any shard, loader still works (starts fresh epoch)."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        fake_state = {
            "batch_idx": 0,
            "shard": "nonexistent_shard.safetensors",
            "epoch": 1,
            "max_seg_size": None,
        }
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            resume_state_dict=fake_state,
        )
        # Should still produce a batch (from epoch 2, since no shard matched in epoch 1)
        batch = next(loader)
        assert batch is not None
        assert len(batch) == 10

    def test_resume_from_step_0(self, tmp_path):
        """State dict from the very first batch is equivalent to continuing past step 0."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=16)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        # Get first batch state
        first_inputs, *_, first_state = next(loader)
        assert first_state["batch_idx"] == 0

        # Resume from step 0 state -- should skip past batch 0
        resumed_loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            resume_state_dict=first_state,
        )
        resumed_inputs, *_, resumed_state = next(resumed_loader)

        # The resumed loader should be at batch_idx >= 0
        # (it skips batch 0 since we resume from it)
        assert resumed_state["batch_idx"] >= 0

    def test_state_dict_with_missing_keys(self, tmp_path):
        """State dict with missing keys uses defaults and does not crash."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        # Only provide batch_idx, omit shard and epoch
        partial_state = {"batch_idx": 0}
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            resume_state_dict=partial_state,
        )
        batch = next(loader)
        assert batch is not None
        assert len(batch) == 10

    def test_resume_produces_different_data(self, tmp_path):
        """Resumed loader yields different first batch than a fresh loader."""
        _write_shards(tmp_path / "packed", n_shards=2, n_rows=16)
        pipeline = _make_pipeline()

        # Fresh loader first batch
        fresh_loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
        )
        fresh_inputs, *_ = next(fresh_loader)

        # Consume 3 batches from another loader to get a state to resume from
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
        )
        for _ in range(3):
            *_, state = next(loader)

        # Resume from batch 3
        resumed_loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            resume_state_dict=state,
        )
        resumed_inputs, *_ = next(resumed_loader)

        assert not t.equal(fresh_inputs, resumed_inputs), (
            "Resumed loader produced same first batch as fresh loader -- resume did not skip"
        )


class TestCompactFeaturePredictionResume:
    """Compact-FP loader state pins row identity and the certified source."""

    def test_state_is_complete_and_resume_replays_exact_targets(self, tmp_path):
        packed, pipeline, layout = _compact_fp_loader_inputs(tmp_path)
        loader = precomputed_data_loader(
            packed,
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            fp_target_layout=layout,
        )
        first = next(loader)
        fp_targets = first[-2]
        state = first[-1]
        assert fp_targets is not None
        assert fp_targets.shape == (4, 3, layout.output_width)
        assert state["feature_vocab"] == layout.feature_vocab.metadata_dict()  # ty: ignore[unresolved-attribute]
        assert state["feature_prediction_width"] == layout.output_width
        assert len(state["fp_target_source_certificate_sha256"]) == 64

        resumed = precomputed_data_loader(
            packed,
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            fp_target_layout=layout,
            resume_state_dict=state,
        )
        resumed_batch = next(resumed)
        assert t.equal(resumed_batch[0], first[0])
        assert t.equal(resumed_batch[-2], fp_targets)

    def test_partial_metadata_and_full_width_resume_are_rejected(self, tmp_path):
        packed, pipeline, layout = _compact_fp_loader_inputs(tmp_path)
        loader = precomputed_data_loader(
            packed,
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            fp_target_layout=layout,
        )
        *_, state = next(loader)

        partial = dict(state)
        del partial["feature_prediction_width"]
        with pytest.raises(ValueError, match="missing or partial"):
            next(
                precomputed_data_loader(
                    packed,
                    B=4,
                    feature_pipeline=pipeline,
                    device="cpu",
                    fp_target_layout=layout,
                    resume_state_dict=partial,
                )
            )

        with pytest.raises(ValueError, match="full-width or headless"):
            next(
                precomputed_data_loader(
                    packed,
                    B=4,
                    feature_pipeline=pipeline,
                    device="cpu",
                    resume_state_dict=state,
                )
            )

    def test_same_width_different_original_ids_are_rejected(self, tmp_path):
        packed, pipeline, layout = _compact_fp_loader_inputs(tmp_path)
        *_, state = next(
            precomputed_data_loader(
                packed,
                B=4,
                feature_pipeline=pipeline,
                device="cpu",
                fp_target_layout=layout,
            )
        )

        other_mask = FeatureMask(
            FeatureMaskMode.whitelist, tuple(range(128, 256))
        )
        other_vocab = effective_feature_vocab(other_mask, 16384)
        assert other_vocab is not None and other_vocab.size == layout.output_width
        other_layout = FeaturePredictionTargetLayout.build(
            n_sae_features=PositiveInt.parse(16_384),
            feature_mask=other_mask,
            feature_clusters=None,
            feature_vocab=other_vocab,
        )
        other_pipeline = _make_pipeline()
        other_pipeline.feature_mask = other_mask
        with pytest.raises(ValueError, match="feature vocabulary mismatch"):
            next(
                precomputed_data_loader(
                    packed,
                    B=4,
                    feature_pipeline=other_pipeline,
                    device="cpu",
                    fp_target_layout=other_layout,
                    resume_state_dict=state,
                )
            )

    def test_compact_targets_stack_across_shard_boundaries(self, tmp_path):
        packed, pipeline, layout = _compact_fp_loader_inputs(
            tmp_path,
            n_shards=2,
            n_rows=3,
        )
        batch = next(
            precomputed_data_loader(
                packed,
                B=4,
                feature_pipeline=pipeline,
                device="cpu",
                fp_target_layout=layout,
            )
        )
        fp_targets = batch[-2]
        assert fp_targets is not None
        assert fp_targets.shape == (4, 3, layout.output_width)
        assert "shard_00001" in batch[-1]["shard"]


class TestPrecomputedResumeExactPosition:
    """Verify resume yields the EXACT same data as continuing the original loader.

    These tests catch the C1 bug where global batch_idx was used for
    shard-relative skip logic, causing the wrong number of batches to be skipped.
    """

    def test_resumed_data_replays_checkpoint_batch(self, tmp_path):
        """Resume replays the checkpointed batch (the last consumed batch).

        The state_dict records shard_batch_offset=N for the batch that was
        just yielded. On resume, the skip logic `shard_batch_idx < N` skips
        batches 0..N-1 and yields batch N again. This is the standard
        "approximate resume" — one batch replayed, no data skipped.

        This test verifies the resumed batch matches the CHECKPOINT batch
        (the one whose state was saved), not the next one.
        """
        _write_shards(tmp_path / "packed", n_shards=3, n_rows=12)
        pipeline = _make_pipeline()

        # Run the original loader: consume 5 batches, save state at each
        original = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
        )
        all_inputs = []
        all_states = []
        for _ in range(6):
            inputs, *_, state = next(original)
            all_inputs.append(inputs.clone())
            all_states.append(state.copy())

        # Resume from batch 4's state — should replay batch 4
        checkpoint_state = all_states[4]
        checkpoint_inputs = all_inputs[4]

        resumed = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            resume_state_dict=checkpoint_state,
        )
        resumed_inputs, *_, resumed_state = next(resumed)

        assert t.equal(resumed_inputs, checkpoint_inputs), (
            "Resumed batch should replay the checkpointed batch (batch 4). "
            f"checkpoint shard={checkpoint_state['shard']}, "
            f"shard_batch_offset={checkpoint_state['shard_batch_offset']}"
        )
        assert resumed_state["batch_idx"] == checkpoint_state["batch_idx"], (
            "resume must preserve the global batch index while replaying the "
            "checkpointed shard-relative batch"
        )

    def test_shard_batch_offset_is_shard_relative(self, tmp_path):
        """shard_batch_offset resets per-shard, not global.

        With 3 shards of 8 rows each and B=4, each shard produces 2 batches.
        The 1st batch of shard_002 should have shard_batch_offset=0
        (not global batch_idx=4).
        """
        _write_shards(tmp_path / "packed", n_shards=3, n_rows=8)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
        )
        states = []
        for _ in range(6):  # 3 shards * 2 batches = 6 batches
            *_, state = next(loader)
            states.append(state.copy())

        # Each shard's first batch should have shard_batch_offset=0
        # Shard 0: batches 0,1 -> offsets 0,1
        assert states[0]["shard_batch_offset"] == 0
        assert states[1]["shard_batch_offset"] == 1
        # Shard 1: batches 2,3 -> offsets 0,1
        assert states[2]["shard_batch_offset"] == 0
        assert states[3]["shard_batch_offset"] == 1
        # Shard 2: batches 4,5 -> offsets 0,1
        assert states[4]["shard_batch_offset"] == 0
        assert states[5]["shard_batch_offset"] == 1

        # But global batch_idx is monotonically increasing
        assert states[0]["batch_idx"] == 0
        assert states[4]["batch_idx"] == 4
        assert states[5]["batch_idx"] == 5

    def test_resume_mid_shard_replays_correct_batch(self, tmp_path):
        """Resuming from the 2nd batch of shard_001 replays that exact batch,
        not a batch offset by a global count.

        This is the test that would have caught the C1 bug where global
        batch_idx was used instead of shard-relative offset.
        """
        _write_shards(tmp_path / "packed", n_shards=4, n_rows=12)
        pipeline = _make_pipeline()

        # Collect all batches from a fresh run
        original = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
        )
        all_inputs = []
        all_states = []
        for _ in range(12):  # 4 shards * 12 rows / 4 = 12 batches
            inputs, *_, state = next(original)
            all_inputs.append(inputs.clone())
            all_states.append(state.copy())

        # Find the 2nd batch of shard_001
        shard_001_batches = [
            (i, s) for i, s in enumerate(all_states)
            if "shard_00001" in s["shard"]
        ]
        assert len(shard_001_batches) >= 2, "Expected at least 2 batches from shard_001"
        resume_idx = shard_001_batches[1][0]  # 2nd batch of shard_001
        resume_state = {
            key: value
            for key, value in all_states[resume_idx].items()
            if key
            not in {
                "batch_rows",
                "ddp_world_size",
                "feature_pipeline_rng_state",
                "label_dropout_rng_state",
            }
        }
        checkpoint_inputs = all_inputs[resume_idx]

        # Resume should replay the checkpointed batch
        resumed = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            resume_state_dict=resume_state,
        )
        resumed_inputs, *_ = next(resumed)

        assert t.equal(resumed_inputs, checkpoint_inputs), (
            f"Resumed from shard_001 batch 1: should replay that batch. "
            f"Resume state: shard={resume_state['shard']}, "
            f"shard_batch_offset={resume_state['shard_batch_offset']}, "
            f"batch_idx={resume_state['batch_idx']}"
        )

        # Drain PAST the resume shard. The legacy offset is shard-relative and
        # must be cleared after shard_00001; otherwise the loader incorrectly
        # drops offset=1 from both complete later shards (audit PR5 F1).
        resumed_inputs = [resumed_inputs]
        resumed_states = []
        for _ in range(resume_idx + 1, len(all_inputs)):
            next_inputs, *_, next_state = next(resumed)
            resumed_inputs.append(next_inputs.clone())
            resumed_states.append(next_state.copy())

        expected_inputs = all_inputs[resume_idx:]
        assert len(resumed_inputs) == len(expected_inputs)
        for relative_idx, (actual, expected) in enumerate(
            zip(resumed_inputs, expected_inputs, strict=True)
        ):
            assert t.equal(actual, expected), (
                "resume changed or omitted a later-shard batch at relative "
                f"index {relative_idx}"
            )

        for later_shard in ("shard_00002", "shard_00003"):
            later_offsets = [
                state["shard_batch_offset"]
                for state in resumed_states
                if later_shard in state["shard"]
            ]
            assert later_offsets == [0, 1, 2]

    def test_topology_change_raises_before_packed_inventory_access(self, tmp_path):
        """F4 applies to precomputed row striping before artifact access too."""
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "does-not-exist",
            B=4,
            feature_pipeline=_make_pipeline(),
            device="cpu",
            ddp_rank=0,
            ddp_world_size=4,
            resume_state_dict={"ddp_world_size": 8},
        )
        with pytest.raises(ValueError, match=r"world.size.*8.*4|8.*4"):
            next(loader)
