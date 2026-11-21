"""Tests for fp_targets building in the selection shard path.

Verifies that the build_fp_targets_from_shard helper correctly builds dense
feature prediction targets from a PackedRowShard's CSR data, and that
precomputed_data_loader yields non-None fp_targets when selection shards are
present (the bug fix).

Test categories:
1. Helper correctness — build_fp_targets_from_shard matches inline code
2. Selection shard path — fp_targets built in single-shard fast path
3. Cross-shard boundary — fp_targets built across shard boundaries
4. Integration — dataloader yields fp_targets, model computes fp_loss
5. Edge cases — empty segments, variable feature counts, padding
"""

from __future__ import annotations

import json
from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    build_fp_targets_from_shard,
    precomputed_data_loader,
    reconstruct_and_select,
)
from steering.conditional.feature_pipeline import (
    FeaturePipelineConfig,
)
from steering.conditional.sampling import (
    Distribution,
    NegativeBundle,
    NegativeSamplingSelection,
    SamplingBundle,
    SamplingSelection,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    _build_batch_index,
    save_packed_row_shard,
    save_selection_shard,
)
from tests.encdec_test_helpers import NUM_BUCKETS, build_threshold_pipeline

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

N_SAE_FEATURES = 1024


def _make_pipeline(n_features):
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0).expand(n_features, -1).contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _make_shard(
    n_rows: int = 8,
    context_len: int = 32,
    segs_per_row: int = 3,
    feats_per_seg: int = 20,
    n_features: int = N_SAE_FEATURES,
    seed: int = 42,
) -> PackedRowShard:
    t.manual_seed(seed)
    T = context_len
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg

    tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)
    row_seg_offsets = t.arange(0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32)

    seg_size = T // segs_per_row
    seg_token_start = t.zeros(total_segs, dtype=t.uint16)
    seg_token_end = t.zeros(total_segs, dtype=t.uint16)
    seg_doc_id_arr = t.zeros(total_segs, dtype=t.uint8)

    for r in range(n_rows):
        for s in range(segs_per_row):
            idx = r * segs_per_row + s
            seg_token_start[idx] = s * seg_size
            seg_token_end[idx] = T if s == segs_per_row - 1 else (s + 1) * seg_size
            seg_doc_id_arr[idx] = 0

    seg_feat_offsets = t.arange(0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32)
    feature_ids = t.randint(0, n_features, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id_arr,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def _make_variable_shard(seed: int = 42) -> PackedRowShard:
    """Shard with variable segments per row and features per segment."""
    t.manual_seed(seed)
    n_rows, T = 3, 32
    tokens = t.randint(1, 32000, (n_rows, T + 1), dtype=t.uint16)

    # Row 0: 4 segments (100, 50, 3, 200 features)
    # Row 1: 2 segments (80, 0 features)
    # Row 2: 3 segments (1, 150, 60 features)
    row_seg_offsets = t.tensor([0, 4, 6, 9], dtype=t.int32)

    seg_token_start = t.tensor([0, 8, 16, 24, 0, 16, 0, 10, 20], dtype=t.uint16)
    seg_token_end = t.tensor([8, 16, 24, 32, 16, 32, 10, 20, 32], dtype=t.uint16)
    seg_doc_id_arr = t.tensor([0, 0, 1, 1, 0, 0, 0, 0, 1], dtype=t.uint8)

    feat_counts = [100, 50, 3, 200, 80, 0, 1, 150, 60]
    total_feats = sum(feat_counts)
    offsets = [0]
    for c in feat_counts:
        offsets.append(offsets[-1] + c)

    seg_feat_offsets = t.tensor(offsets, dtype=t.int32)
    feature_ids = t.randint(0, N_SAE_FEATURES, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

    return PackedRowShard(
        tokens=tokens,
        row_seg_offsets=row_seg_offsets,
        seg_token_start=seg_token_start,
        seg_token_end=seg_token_end,
        seg_doc_id=seg_doc_id_arr,
        seg_feat_offsets=seg_feat_offsets,
        feature_ids=feature_ids,
        feature_pcts=feature_pcts,
    )


def _make_selection_shard_for(
    base: PackedRowShard,
    sel_per_seg: int = 5,
    neg_per_seg: int = 2,
    seed: int = 99,
) -> SelectionShard:
    """Create a SelectionShard aligned with a base PackedRowShard."""
    t.manual_seed(seed)
    total_segs = base.total_segments

    total_sel = total_segs * sel_per_seg
    total_neg = total_segs * neg_per_seg

    sel_offsets = t.arange(0, (total_segs + 1) * sel_per_seg, sel_per_seg, dtype=t.int32)
    sel_fids = t.randint(0, N_SAE_FEATURES, (total_sel,), dtype=t.uint16)
    sel_buckets = t.randint(0, 9, (total_sel,), dtype=t.uint8)
    sel_pcts = t.randint(1, 255, (total_sel,), dtype=t.uint8)

    neg_offsets = t.arange(0, (total_segs + 1) * neg_per_seg, neg_per_seg, dtype=t.int32)
    neg_fids = t.randint(0, N_SAE_FEATURES, (total_neg,), dtype=t.uint16)

    return SelectionShard(
        sel_offsets=sel_offsets,
        sel_fids=sel_fids,
        sel_buckets=sel_buckets,
        sel_pcts=sel_pcts,
        neg_offsets=neg_offsets,
        neg_fids=neg_fids,
    )


def _make_sampling_pipeline(
    n_features: int = N_SAE_FEATURES,
    n_buckets: int = 10,
    with_negatives: bool = False,
    mean: float = 3.0,
    seed: int = 42,
) -> FeaturePipelineConfig:
    rng = t.Generator().manual_seed(seed)
    selection = SamplingSelection(
        distribution=Distribution.geometric,
        mean=mean, std=mean, min_k=1, max_k=None,
    )
    pct_table = t.linspace(0.01, 0.99, 99, dtype=t.float64).unsqueeze(0).expand(n_features, -1).contiguous()
    sampling_bundle = SamplingBundle(
        selection=selection, percentile_table=pct_table, rng=rng,
    )
    effective_buckets = n_buckets - 1 if with_negatives else n_buckets
    n_boundaries = effective_buckets - 1
    b = t.linspace(1 / effective_buckets, 1.0 - 1 / effective_buckets, n_boundaries, dtype=t.float64)
    boundaries = b.unsqueeze(0).expand(n_features, -1).contiguous()

    negative_bundle = None
    if with_negatives:
        neg_rng = t.Generator().manual_seed(seed + 1)
        negative_bundle = NegativeBundle(
            selection=NegativeSamplingSelection(
                distribution=Distribution.geometric,
                mean=2.0, std=2.0, min_k=1,
            ),
            n_sae_features=n_features, rng=neg_rng,
        )

    return FeaturePipelineConfig(
        selection_config=sampling_bundle,
        boundaries=boundaries,
        negative_bundle=negative_bundle,
        effective_buckets=effective_buckets,
    )


def _write_packed_shard(output_dir: Path, shard: PackedRowShard, name: str = "shard_00000"):
    """Save a shard to disk with a minimal manifest."""
    output_dir.mkdir(parents=True, exist_ok=True)
    save_packed_row_shard(shard, output_dir / name)
    # Write a minimal manifest (read_manifest returns None if missing, which is fine,
    # but we provide one for max_n_seg so static shapes work)
    segs_per_row = []
    offsets = shard.row_seg_offsets.tolist()
    for i in range(shard.n_rows):
        segs_per_row.append(offsets[i + 1] - offsets[i])
    max_n_seg = max(segs_per_row) if segs_per_row else 1
    manifest = {
        "checksum": "test",
        "kind": "packed_rows",
        "metadata": {"max_n_seg": max_n_seg},
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest))


def _write_selection_manifest(sel_dir: Path, packed_dir: Path) -> None:
    write_manifest(
        sel_dir,
        "test-selection",
        OverlayKind.PACKED_ROW,
        metadata={
            "packed_rows": {
                "packed_row_fingerprint": packed_row_fingerprint(packed_dir),
            },
        },
    )


def _build_fp_targets_reference(
    shard: PackedRowShard,
    row_indices: list[int],
    n_sae_features: int,
    pad_n_seg: int | None = None,
) -> t.Tensor:
    """Reference implementation: build fp_targets using explicit loop over segments."""
    idx = _build_batch_index(shard, row_indices, pad_n_seg=pad_n_seg)
    B = idx.B
    batch_n_seg = idx.batch_n_seg

    fp_targets = t.zeros(B, batch_n_seg, n_sae_features, dtype=t.float32)

    for i in range(idx.total_segs):
        b = int(idx.b_flat[i].item())
        local = int(idx.local_flat[i].item())
        global_seg = int(idx.global_flat[i].item())

        feat_start = int(shard.seg_feat_offsets[global_seg].item())
        feat_end = int(shard.seg_feat_offsets[global_seg + 1].item())

        for f in range(feat_start, feat_end):
            fid = int(shard.feature_ids[f].item())
            pct = float(shard.feature_pcts[f].float().item()) / 255.0
            fp_targets[b, local, fid] = pct

    return fp_targets


# ======================================================================
# 1. Helper correctness tests
# ======================================================================


class TestBuildFpTargetsHelper:
    def test_helper_matches_reference_loop(self):
        """build_fp_targets_from_shard produces identical output to reference loop."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        row_indices = [0, 1, 2, 3]
        idx = _build_batch_index(shard, row_indices)

        result = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)
        reference = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert t.allclose(result, reference, atol=1e-7), "Helper must match reference loop"

    def test_helper_matches_reconstruct_and_select(self):
        """build_fp_targets_from_shard produces IDENTICAL output to reconstruct_and_select."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        row_indices = [0, 1, 2, 3]
        pipeline = _make_sampling_pipeline()

        _, fused_fp = reconstruct_and_select(
            shard, row_indices, pipeline, n_sae_features=N_SAE_FEATURES,
        )

        idx = _build_batch_index(shard, row_indices)
        helper_fp = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert fused_fp is not None
        assert t.equal(helper_fp, fused_fp), (
            "Helper and fused path must produce identical fp_targets"
        )

    def test_helper_with_empty_segments(self):
        """Segments with 0 features get all-zero target rows."""
        shard = _make_variable_shard()
        # Row 1, seg 1 has 0 features
        row_indices = [1]
        idx = _build_batch_index(shard, row_indices)

        result = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # The second segment (local index 1) of row 1 has 0 features
        assert (result[0, 1] == 0.0).all(), "Empty segment should have all-zero targets"

    def test_helper_with_variable_feature_counts(self):
        """Segments with different feature counts (1, 50, 100, 150, 200) all correct."""
        shard = _make_variable_shard()
        row_indices = [0, 2]
        idx = _build_batch_index(shard, row_indices)

        result = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)
        reference = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert t.allclose(result, reference, atol=1e-7)

    def test_helper_with_pad_n_seg(self):
        """Padded segments beyond real count have all-zero targets."""
        shard = _make_shard(n_rows=4, segs_per_row=3)
        row_indices = [0, 1]
        pad_n_seg = 8  # much larger than actual 3 segs per row

        idx = _build_batch_index(shard, row_indices, pad_n_seg=pad_n_seg)
        result = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert result.shape == (2, pad_n_seg, N_SAE_FEATURES)
        # Segments 3..7 should be all zeros (only 3 real segments)
        assert (result[:, 3:, :] == 0.0).all(), "Padded segments should be all zeros"

        # Real segments should match reference
        reference = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES, pad_n_seg=pad_n_seg)
        assert t.allclose(result, reference, atol=1e-7)


# ======================================================================
# 2. Core correctness tests (selection shard path)
# ======================================================================


class TestFpTargetsSelectionShardPath:
    def test_fp_targets_not_none_with_selection_shards(self, tmp_path):
        """When selection shards exist and n_sae_features is set,
        fp_targets must NOT be None. This is the bug we're fixing."""
        shard = _make_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard_for(shard)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_packed_shard(packed_dir, shard)
        sel_dir.mkdir(parents=True, exist_ok=True)
        save_selection_shard(sel, sel_dir / "shard_00000")
        _write_selection_manifest(sel_dir, packed_dir)

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            n_sae_features=N_SAE_FEATURES,
        )
        batch = next(loader)
        fp_targets = batch[8]

        assert fp_targets is not None, "fp_targets must NOT be None with selection shards"

    def test_fp_targets_match_fused_path(self):
        """fp_targets from the selection shard path must be IDENTICAL to
        fp_targets from the fused path (reconstruct_and_select) for the
        same base shard and row indices."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        row_indices = [0, 1, 2, 3]
        pipeline = _make_sampling_pipeline()

        # Fused path fp_targets
        _, fused_fp = reconstruct_and_select(
            shard, row_indices, pipeline, n_sae_features=N_SAE_FEATURES,
        )

        # Selection shard path fp_targets (via helper)
        idx = _build_batch_index(shard, row_indices)
        sel_fp = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert fused_fp is not None
        assert t.equal(sel_fp, fused_fp)

    def test_fp_targets_have_all_extracted_features(self):
        """fp_targets should have many non-zero entries per segment (all extracted
        features), NOT just the ~5 selected subset in the selection shard."""
        shard = _make_shard(n_rows=4, segs_per_row=3, feats_per_seg=20)
        row_indices = [0, 1, 2, 3]
        idx = _build_batch_index(shard, row_indices)

        fp_targets = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Each segment has 20 extracted features. Count non-zeros per segment.
        for b in range(4):
            for s in range(3):
                nonzero_count = (fp_targets[b, s] > 0).sum().item()
                # Some feature IDs may collide (rand), so count could be < 20
                # but should be significantly more than ~5 selected features
                assert nonzero_count >= 10, (
                    f"Segment [{b},{s}] has only {nonzero_count} non-zero entries, "
                    f"expected close to 20 (all extracted features)"
                )

    def test_fp_targets_values_match_csr(self):
        """Every non-zero value in fp_targets matches the corresponding
        feature_pcts / 255.0 from the base shard's CSR data.

        When duplicate feature IDs exist within a segment, both the vectorized
        scatter and this check use last-write-wins semantics. We build an
        expected dict per segment (last write wins) then compare.
        """
        shard = _make_shard(n_rows=2, segs_per_row=2, feats_per_seg=10)
        row_indices = [0, 1]
        idx = _build_batch_index(shard, row_indices)

        fp_targets = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        # Verify each segment's values match the CSR data (last-write-wins safe)
        for i in range(idx.total_segs):
            b = int(idx.b_flat[i].item())
            local = int(idx.local_flat[i].item())
            global_seg = int(idx.global_flat[i].item())

            feat_start = int(shard.seg_feat_offsets[global_seg].item())
            feat_end = int(shard.seg_feat_offsets[global_seg + 1].item())

            # Build expected values: last occurrence wins (matches scatter semantics)
            expected: dict[int, float] = {}
            for f_idx in range(feat_start, feat_end):
                fid = int(shard.feature_ids[f_idx].item())
                expected[fid] = float(
                    shard.feature_pcts[f_idx].float().item()
                ) / 255.0

            for fid, expected_pct in expected.items():
                actual = fp_targets[b, local, fid].item()
                assert abs(actual - expected_pct) < 1e-6, (
                    f"Value mismatch at seg ({b},{local}) feature {fid}: "
                    f"expected {expected_pct:.6f}, got {actual:.6f}"
                )

    def test_fp_targets_shape(self):
        """fp_targets shape is [B, n_seg, n_sae_features]."""
        shard = _make_shard(n_rows=4, segs_per_row=3, feats_per_seg=20)
        row_indices = [0, 1, 2, 3]
        idx = _build_batch_index(shard, row_indices)

        fp_targets = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert fp_targets.shape == (4, 3, N_SAE_FEATURES)

    def test_fp_targets_values_in_zero_one(self):
        """All values in [0.0, 1.0]."""
        shard = _make_shard(n_rows=4, feats_per_seg=50)
        row_indices = [0, 1, 2, 3]
        idx = _build_batch_index(shard, row_indices)

        fp_targets = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert fp_targets.min() >= 0.0
        assert fp_targets.max() <= 1.0

    def test_fp_targets_dtype(self):
        """fp_targets must be float32."""
        shard = _make_shard(n_rows=2)
        row_indices = [0, 1]
        idx = _build_batch_index(shard, row_indices)

        fp_targets = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)

        assert fp_targets.dtype == t.float32


# ======================================================================
# 3. Cross-shard boundary tests
# ======================================================================


class TestFpTargetsCrossShardBoundary:
    def test_cross_shard_fp_targets_not_none(self, tmp_path):
        """fp_targets is built correctly even when a batch spans two shards."""
        shard1 = _make_shard(n_rows=3, segs_per_row=2, feats_per_seg=15, seed=42)
        shard2 = _make_shard(n_rows=3, segs_per_row=2, feats_per_seg=15, seed=99)
        sel1 = _make_selection_shard_for(shard1, seed=100)
        sel2 = _make_selection_shard_for(shard2, seed=200)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        packed_dir.mkdir(parents=True, exist_ok=True)
        sel_dir.mkdir(parents=True, exist_ok=True)

        save_packed_row_shard(shard1, packed_dir / "shard_00000")
        save_packed_row_shard(shard2, packed_dir / "shard_00001")
        save_selection_shard(sel1, sel_dir / "shard_00000")
        save_selection_shard(sel2, sel_dir / "shard_00001")
        _write_selection_manifest(sel_dir, packed_dir)

        # Write manifest with max_n_seg=2
        manifest = {
            "checksum": "test",
            "kind": "packed_rows",
            "metadata": {"max_n_seg": 2},
        }
        (packed_dir / "manifest.json").write_text(json.dumps(manifest))

        # B=4 with 3 rows per shard means the second batch will span shards
        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            n_sae_features=N_SAE_FEATURES,
        )

        # First batch (rows 0-3: 3 from shard1, 1 from shard2 — cross-shard!)
        # Actually with B=4 and 3 rows per shard:
        # pending accumulates: shard1[0], shard1[1], shard1[2] — not yet B=4
        # then shard2[0] — now B=4, cross-shard boundary!
        batch = next(loader)
        fp_targets = batch[8]
        assert fp_targets is not None, "fp_targets must not be None for cross-shard batch"
        assert fp_targets.shape[0] == 4

    def test_cross_shard_fp_targets_match_per_row(self, tmp_path):
        """Rows from different shards each get fp_targets matching their
        respective base shard's CSR data."""
        shard1 = _make_shard(n_rows=3, segs_per_row=2, feats_per_seg=10, seed=42)
        shard2 = _make_shard(n_rows=3, segs_per_row=2, feats_per_seg=10, seed=99)
        sel1 = _make_selection_shard_for(shard1, seed=100)
        sel2 = _make_selection_shard_for(shard2, seed=200)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        packed_dir.mkdir(parents=True, exist_ok=True)
        sel_dir.mkdir(parents=True, exist_ok=True)

        save_packed_row_shard(shard1, packed_dir / "shard_00000")
        save_packed_row_shard(shard2, packed_dir / "shard_00001")
        save_selection_shard(sel1, sel_dir / "shard_00000")
        save_selection_shard(sel2, sel_dir / "shard_00001")
        _write_selection_manifest(sel_dir, packed_dir)

        manifest = {
            "checksum": "test",
            "kind": "packed_rows",
            "metadata": {"max_n_seg": 2},
        }
        (packed_dir / "manifest.json").write_text(json.dumps(manifest))

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            n_sae_features=N_SAE_FEATURES,
        )

        batch = next(loader)
        fp_targets = batch[8]
        assert fp_targets is not None

        # First 3 rows from shard1, last 1 from shard2
        ref_shard1 = _build_fp_targets_reference(shard1, [0, 1, 2], N_SAE_FEATURES, pad_n_seg=2)
        ref_shard2 = _build_fp_targets_reference(shard2, [0], N_SAE_FEATURES, pad_n_seg=2)

        # Cross-shard path builds fp_targets per-row then concatenates
        assert t.allclose(fp_targets[:3], ref_shard1, atol=1e-6), (
            "First 3 rows should match shard1 reference"
        )
        assert t.allclose(fp_targets[3:], ref_shard2, atol=1e-6), (
            "Last row should match shard2 reference"
        )


# ======================================================================
# 4. Integration tests
# ======================================================================


class TestFpTargetsIntegration:
    def test_dataloader_yields_fp_targets_with_selection_shards(self, tmp_path):
        """precomputed_data_loader yields non-None fp_targets in the
        selection shard path for every batch."""
        shard = _make_shard(n_rows=12, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard_for(shard)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_packed_shard(packed_dir, shard)
        sel_dir.mkdir(parents=True, exist_ok=True)
        save_selection_shard(sel, sel_dir / "shard_00000")
        _write_selection_manifest(sel_dir, packed_dir)

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            n_sae_features=N_SAE_FEATURES,
        )

        # Check 3 batches (12 rows / 4 = 3 batches)
        for i in range(3):
            batch = next(loader)
            fp_targets = batch[8]
            assert fp_targets is not None, f"Batch {i}: fp_targets must not be None"
            assert fp_targets.shape == (4, 3, N_SAE_FEATURES), (
                f"Batch {i}: shape mismatch {fp_targets.shape}"
            )
            assert fp_targets.max() > 0, f"Batch {i}: fp_targets should have non-zero values"

    def test_dataloader_fp_targets_none_without_n_sae_features(self, tmp_path):
        """When n_sae_features is not set, fp_targets should be None even with selection shards."""
        shard = _make_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard_for(shard)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_packed_shard(packed_dir, shard)
        sel_dir.mkdir(parents=True, exist_ok=True)
        save_selection_shard(sel, sel_dir / "shard_00000")
        _write_selection_manifest(sel_dir, packed_dir)

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            # n_sae_features NOT set
        )

        batch = next(loader)
        fp_targets = batch[8]
        assert fp_targets is None

    def test_fp_loss_not_silently_skipped(self, tmp_path):
        """With feature_pred_config set and selection shards present,
        fp_targets is non-None (the model would compute fp_loss)."""
        shard = _make_shard(n_rows=8, segs_per_row=3, feats_per_seg=20)
        sel = _make_selection_shard_for(shard)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_packed_shard(packed_dir, shard)
        sel_dir.mkdir(parents=True, exist_ok=True)
        save_selection_shard(sel, sel_dir / "shard_00000")
        _write_selection_manifest(sel_dir, packed_dir)

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            n_sae_features=N_SAE_FEATURES,
        )

        batch = next(loader)
        fp_targets = batch[8]

        # The critical check: fp_targets is not None, so the model WILL
        # compute fp_loss instead of silently skipping it
        assert fp_targets is not None
        assert fp_targets.shape[2] == N_SAE_FEATURES

    def test_fused_path_still_works(self):
        """Verify the fused path (reconstruct_and_select) still produces correct fp_targets
        after refactoring to use the helper."""
        shard = _make_shard(n_rows=4, feats_per_seg=30)
        pipeline = _make_sampling_pipeline()

        _, fp_targets = reconstruct_and_select(
            shard, [0, 1, 2, 3], pipeline, n_sae_features=N_SAE_FEATURES,
        )

        assert fp_targets is not None
        assert fp_targets.shape == (4, 3, N_SAE_FEATURES)
        assert fp_targets.max() > 0
        assert fp_targets.min() >= 0
        assert fp_targets.max() <= 1.0

        # Verify against reference
        reference = _build_fp_targets_reference(shard, [0, 1, 2, 3], N_SAE_FEATURES)
        assert t.allclose(fp_targets, reference, atol=1e-7)

    def test_fused_path_without_n_sae_features(self):
        """Fused path without n_sae_features returns None fp_targets."""
        shard = _make_shard(n_rows=4)
        pipeline = _make_sampling_pipeline()

        _, fp_targets = reconstruct_and_select(
            shard, [0, 1, 2, 3], pipeline,
        )

        assert fp_targets is None


# ======================================================================
# 5. Edge case tests
# ======================================================================


class TestFpTargetsEdgeCases:
    def test_single_row(self):
        """Works with a single row."""
        shard = _make_shard(n_rows=4, feats_per_seg=20)
        row_indices = [2]
        idx = _build_batch_index(shard, row_indices)

        result = build_fp_targets_from_shard(shard, idx, N_SAE_FEATURES)
        reference = _build_fp_targets_reference(shard, row_indices, N_SAE_FEATURES)

        assert result.shape[0] == 1
        assert t.allclose(result, reference, atol=1e-7)

    def test_large_n_sae_features(self):
        """Works with very large n_sae_features (sparse targets)."""
        shard = _make_shard(n_rows=2, feats_per_seg=5, n_features=512)
        row_indices = [0, 1]
        idx = _build_batch_index(shard, row_indices)

        # Use n_sae_features much larger than max feature ID
        fp = build_fp_targets_from_shard(shard, idx, n_sae_features=16384)

        assert fp.shape == (2, 3, 16384)
        # Should be very sparse
        nonzero_frac = (fp > 0).float().mean().item()
        assert nonzero_frac < 0.01

    def test_duplicate_feature_ids_last_write_wins(self):
        """When a segment has duplicate feature IDs, the last value wins (scatter semantics)."""
        T = 16
        tokens = t.randint(1, 1000, (1, T + 1), dtype=t.uint16)
        row_seg_offsets = t.tensor([0, 1], dtype=t.int32)
        seg_token_start = t.tensor([0], dtype=t.uint16)
        seg_token_end = t.tensor([T], dtype=t.uint16)
        seg_doc_id_arr = t.tensor([0], dtype=t.uint8)

        # Two features with same ID but different pcts
        seg_feat_offsets = t.tensor([0, 2], dtype=t.int32)
        feature_ids = t.tensor([5, 5], dtype=t.uint16)
        feature_pcts = t.tensor([100, 200], dtype=t.uint8)

        shard = PackedRowShard(
            tokens=tokens,
            row_seg_offsets=row_seg_offsets,
            seg_token_start=seg_token_start,
            seg_token_end=seg_token_end,
            seg_doc_id=seg_doc_id_arr,
            seg_feat_offsets=seg_feat_offsets,
            feature_ids=feature_ids,
            feature_pcts=feature_pcts,
        )

        idx = _build_batch_index(shard, [0])
        fp = build_fp_targets_from_shard(shard, idx, 1024)

        # The last write should win (200 / 255.0)
        assert abs(fp[0, 0, 5].item() - 200 / 255.0) < 1e-6

    def test_no_features_at_all(self):
        """Shard where every segment has 0 features."""
        T = 16
        n_rows = 2
        tokens = t.randint(1, 1000, (n_rows, T + 1), dtype=t.uint16)
        row_seg_offsets = t.tensor([0, 1, 2], dtype=t.int32)
        seg_token_start = t.tensor([0, 0], dtype=t.uint16)
        seg_token_end = t.tensor([T, T], dtype=t.uint16)
        seg_doc_id_arr = t.tensor([0, 0], dtype=t.uint8)
        seg_feat_offsets = t.tensor([0, 0, 0], dtype=t.int32)  # 0 features each
        feature_ids = t.tensor([], dtype=t.uint16)
        feature_pcts = t.tensor([], dtype=t.uint8)

        shard = PackedRowShard(
            tokens=tokens,
            row_seg_offsets=row_seg_offsets,
            seg_token_start=seg_token_start,
            seg_token_end=seg_token_end,
            seg_doc_id=seg_doc_id_arr,
            seg_feat_offsets=seg_feat_offsets,
            feature_ids=feature_ids,
            feature_pcts=feature_pcts,
        )

        idx = _build_batch_index(shard, [0, 1])
        fp = build_fp_targets_from_shard(shard, idx, 512)

        assert fp.shape == (2, 1, 512)
        assert (fp == 0.0).all()

    def test_selection_shard_path_dataloader_correct_values(self, tmp_path):
        """End-to-end: values in fp_targets from dataloader match reference
        when using selection shards."""
        shard = _make_shard(n_rows=4, segs_per_row=2, feats_per_seg=15)
        sel = _make_selection_shard_for(shard, sel_per_seg=4, neg_per_seg=1)

        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_packed_shard(packed_dir, shard)
        sel_dir.mkdir(parents=True, exist_ok=True)
        save_selection_shard(sel, sel_dir / "shard_00000")
        _write_selection_manifest(sel_dir, packed_dir)

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(N_SAE_FEATURES),
            device="cpu",
            selection_dir=sel_dir,
            effective_buckets=10,
            n_sae_features=N_SAE_FEATURES,
        )

        batch = next(loader)
        fp_targets = batch[8]

        # Build reference from same shard + rows
        reference = _build_fp_targets_reference(shard, [0, 1, 2, 3], N_SAE_FEATURES, pad_n_seg=2)

        assert fp_targets is not None
        assert t.allclose(fp_targets, reference, atol=1e-6)
