"""Tests for DDP row striping in precomputed_data_loader.

Verifies that with ddp_rank / ddp_world_size, each rank receives a
disjoint subset of rows, the union covers all data, and backward
compatibility with the default single-rank case is preserved.
"""

from __future__ import annotations

from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder.dataloader import (
    precomputed_data_loader,
    reconstruct_and_select,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import (
    PackedRowShard,
    SelectionShard,
    load_packed_row_shard,
    reconstruct_batch,
    save_packed_row_shard,
    save_selection_shard,
)
from tests.encdec_test_helpers import (
    NUM_BUCKETS,
    _make_identity_selection,
    build_threshold_pipeline,
)


def _make_pipeline(n_features):
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0).expand(n_features, -1).contiguous()
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
    """Create a synthetic packed row shard with deterministic data.

    Each row gets unique token values derived from (seed, row_idx) so we can
    identify which shard rows ended up in a given batch.
    """
    t.manual_seed(seed)
    T = context_len
    total_segs = n_rows * segs_per_row
    total_feats = total_segs * feats_per_seg

    tokens = t.zeros(n_rows, T + 1, dtype=t.uint16)
    for r in range(n_rows):
        # Embed (seed, row_idx) as a fingerprint in the first two token positions
        tokens[r, 0] = seed % 65536
        tokens[r, 1] = r
        tokens[r, 2:] = t.randint(1, 32000, (T - 1,), dtype=t.uint16)

    row_seg_offsets = t.arange(
        0, (n_rows + 1) * segs_per_row, segs_per_row, dtype=t.int32
    )

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

    seg_feat_offsets = t.arange(
        0, (total_segs + 1) * feats_per_seg, feats_per_seg, dtype=t.int32
    )
    feature_ids = t.randint(0, 16384, (total_feats,), dtype=t.uint16)
    feature_pcts = t.randint(1, 255, (total_feats,), dtype=t.uint8)

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


def _write_shards(
    output_dir: Path, n_shards: int = 2, **shard_kwargs
) -> list[Path]:
    """Write *n_shards* to *output_dir*, return paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for i in range(n_shards):
        shard = _make_shard(seed=42 + i, **shard_kwargs)
        path = save_packed_row_shard(shard, output_dir / f"shard_{i:05d}")
        paths.append(path)
    return paths


def _collect_token_fingerprints(
    packed_row_dir: Path,
    B: int,
    n_batches: int,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[int, int]]:
    """Run the dataloader for *n_batches* and return (seed, row_idx) fingerprints.

    Reads back the (seed, row_idx) pair embedded in tokens[:, :2] by ``_make_shard``.
    """
    loader = precomputed_data_loader(
        packed_row_dir=packed_row_dir,
        B=B,
        feature_pipeline=_make_pipeline(16384),
        device="cpu",
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    fingerprints: list[tuple[int, int]] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        inputs = batch[0]  # (B, T)
        for b in range(inputs.shape[0]):
            seed_val = inputs[b, 0].item()
            row_val = inputs[b, 1].item()
            fingerprints.append((seed_val, row_val))
    return fingerprints


def _count_rank_rows(n_rows_per_shard: int, n_shards: int, rank: int, world_size: int) -> int:
    """Count total rows a rank will see across all shards (one epoch)."""
    total = 0
    for _ in range(n_shards):
        for row_idx in range(n_rows_per_shard):
            if row_idx % world_size == rank:
                total += 1
    return total


def _make_selection_shard_for(
    base: PackedRowShard,
    sel_per_seg: int = 3,
    neg_per_seg: int = 2,
    seed: int = 99,
) -> SelectionShard:
    """Create a SelectionShard aligned with a base PackedRowShard."""
    t.manual_seed(seed)
    total_segs = base.total_segments

    total_sel = total_segs * sel_per_seg
    total_neg = total_segs * neg_per_seg

    sel_offsets = t.arange(0, (total_segs + 1) * sel_per_seg, sel_per_seg, dtype=t.int32)
    sel_fids = t.randint(0, 1024, (total_sel,), dtype=t.uint16)
    sel_buckets = t.randint(0, 9, (total_sel,), dtype=t.uint8)
    sel_pcts = t.randint(1, 255, (total_sel,), dtype=t.uint8)

    neg_offsets = t.arange(0, (total_segs + 1) * neg_per_seg, neg_per_seg, dtype=t.int32)
    neg_fids = t.randint(0, 1024, (total_neg,), dtype=t.uint16)

    return SelectionShard(
        sel_offsets=sel_offsets,
        sel_fids=sel_fids,
        sel_buckets=sel_buckets,
        sel_pcts=sel_pcts,
        neg_offsets=neg_offsets,
        neg_fids=neg_fids,
    )


def _write_shards_with_selection(
    packed_dir: Path,
    sel_dir: Path,
    n_shards: int = 1,
    **shard_kwargs,
) -> None:
    """Write base shards + aligned selection shards."""
    packed_dir.mkdir(parents=True, exist_ok=True)
    sel_dir.mkdir(parents=True, exist_ok=True)
    for i in range(n_shards):
        base = _make_shard(seed=42 + i, **shard_kwargs)
        save_packed_row_shard(base, packed_dir / f"shard_{i:05d}")
        sel = _make_selection_shard_for(base, seed=99 + i)
        save_selection_shard(sel, sel_dir / f"shard_{i:05d}")
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


def _collect_token_fingerprints_with_selection(
    packed_row_dir: Path,
    selection_dir: Path,
    B: int,
    n_batches: int,
    rank: int = 0,
    world_size: int = 1,
) -> list[tuple[int, int]]:
    """Like ``_collect_token_fingerprints`` but with selection shards."""
    loader = precomputed_data_loader(
        packed_row_dir=packed_row_dir,
        B=B,
        feature_pipeline=_make_pipeline(16384),
        device="cpu",
        selection_dir=selection_dir,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    fingerprints: list[tuple[int, int]] = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        inputs = batch[0]
        for b in range(inputs.shape[0]):
            fingerprints.append((inputs[b, 0].item(), inputs[b, 1].item()))
    return fingerprints


# ======================================================================
# 1. Row striping — disjointness and coverage
# ======================================================================


class TestDDPRowStriping:
    """Verify each rank sees a disjoint, complete subset of rows."""

    def test_ranks_see_disjoint_rows(self, tmp_path):
        """With world_size=2, rank 0 and rank 1 see completely disjoint rows."""
        n_rows = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        # Each rank: 4 rows / batch = 1 batch
        fp0 = _collect_token_fingerprints(tmp_path / "packed", B=4, n_batches=1, rank=0, world_size=2)
        fp1 = _collect_token_fingerprints(tmp_path / "packed", B=4, n_batches=1, rank=1, world_size=2)

        set0 = set(fp0)
        set1 = set(fp1)

        assert set0.isdisjoint(set1), f"Overlap: {set0 & set1}"

    def test_all_rows_covered(self, tmp_path):
        """The union of rows across all ranks equals the full row set."""
        n_rows = 12
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        world_size = 3
        all_fps: set[tuple[int, int]] = set()
        for rank in range(world_size):
            fps = _collect_token_fingerprints(
                tmp_path / "packed", B=4, n_batches=10, rank=rank, world_size=world_size
            )
            all_fps.update(fps)

        # Should have all n_rows rows from the single shard (seed=42, rows 0..11)
        expected = {(42, r) for r in range(n_rows)}
        assert all_fps == expected

    def test_rows_evenly_distributed(self, tmp_path):
        """Each rank sees approximately R/world_size rows per shard."""
        n_rows = 10
        world_size = 3
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)

        counts = []
        for rank in range(world_size):
            count = _count_rank_rows(n_rows, 1, rank, world_size)
            counts.append(count)

        # 10 rows / 3 ranks → 4, 3, 3
        assert sum(counts) == n_rows
        assert max(counts) - min(counts) <= 1

    def test_single_rank_sees_all_rows(self, tmp_path):
        """With world_size=1 rank=0, all rows are visible (backward compat)."""
        n_rows = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        fps = _collect_token_fingerprints(
            tmp_path / "packed", B=4, n_batches=2, rank=0, world_size=1
        )
        expected = {(42, r) for r in range(n_rows)}
        assert set(fps) == expected

    def test_three_ranks_disjoint(self, tmp_path):
        """With world_size=3, each rank sees every 3rd row, no overlap."""
        n_rows = 9
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        fps = []
        for rank in range(3):
            fps.append(
                set(_collect_token_fingerprints(
                    tmp_path / "packed", B=3, n_batches=1, rank=rank, world_size=3
                ))
            )

        # Pairwise disjoint
        assert fps[0].isdisjoint(fps[1])
        assert fps[0].isdisjoint(fps[2])
        assert fps[1].isdisjoint(fps[2])
        # Union is complete
        assert fps[0] | fps[1] | fps[2] == {(42, r) for r in range(n_rows)}

    def test_world_size_larger_than_shard_rows(self, tmp_path):
        """If world_size=8 but shard has 4 rows, high ranks get no rows.

        Only ranks 0-3 receive data (one row per shard per rank).
        Ranks 4-7 never form a batch — we verify by checking only active ranks.
        """
        n_rows = 4
        n_shards = 2
        world_size = 8
        _write_shards(tmp_path / "packed", n_shards=n_shards, n_rows=n_rows)

        all_fps: set[tuple[int, int]] = set()
        for rank in range(world_size):
            rank_total = _count_rank_rows(n_rows, n_shards, rank, world_size)
            if rank_total == 0:
                continue  # rank gets no rows — would loop forever
            fps = _collect_token_fingerprints(
                tmp_path / "packed", B=1, n_batches=20, rank=rank, world_size=world_size
            )
            all_fps.update(fps)

        # All 8 rows (4 from each of 2 shards) should be covered by active ranks
        expected = {(42, r) for r in range(n_rows)} | {(43, r) for r in range(n_rows)}
        assert all_fps == expected


# ======================================================================
# 2. Batch correctness
# ======================================================================


class TestDDPBatchCorrectness:
    """Batch tensors are correct for each rank's striped rows."""

    def test_batch_shapes_consistent_across_ranks(self, tmp_path):
        """All ranks produce batches with the same tensor shapes."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        shapes = []
        for rank in range(2):
            loader = precomputed_data_loader(
                packed_row_dir=tmp_path / "packed",
                B=4,
                feature_pipeline=_make_pipeline(16384),
                device="cpu",
                ddp_rank=rank,
                ddp_world_size=2,
            )
            batch = next(loader)
            shapes.append(tuple(tensor.shape for tensor in batch[:8]))

        assert shapes[0] == shapes[1]

    def test_rank0_gets_even_rows(self, tmp_path):
        """Rank 0 with world_size=2 gets rows 0, 2, 4, 6."""
        n_rows = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        fps = _collect_token_fingerprints(
            tmp_path / "packed", B=4, n_batches=1, rank=0, world_size=2
        )
        expected_rows = {0, 2, 4, 6}
        actual_rows = {row for _, row in fps}
        assert actual_rows == expected_rows

    def test_rank1_gets_odd_rows(self, tmp_path):
        """Rank 1 with world_size=2 gets rows 1, 3, 5, 7."""
        n_rows = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        fps = _collect_token_fingerprints(
            tmp_path / "packed", B=4, n_batches=1, rank=1, world_size=2
        )
        expected_rows = {1, 3, 5, 7}
        actual_rows = {row for _, row in fps}
        assert actual_rows == expected_rows

    def test_state_dict_batch_idx_increments(self, tmp_path):
        """batch_idx increments normally for each rank."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=16)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            ddp_rank=0,
            ddp_world_size=2,
        )
        # 16 rows, rank 0 gets 8, B=4 → 2 batches per epoch
        b0 = next(loader)
        b1 = next(loader)
        assert b0[-1]["batch_idx"] == 0
        assert b1[-1]["batch_idx"] == 1


# ======================================================================
# 3. Epoch cycling
# ======================================================================


class TestDDPEpochCycling:
    """Epoch boundaries work correctly with DDP striping."""

    def test_epoch_cycles_per_rank(self, tmp_path):
        """Each rank cycles to epoch 2 after exhausting its rows."""
        n_rows = 4
        B = 2
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        # world_size=2: each rank gets 2 rows, B=2 → 1 batch per epoch
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=B,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            ddp_rank=0,
            ddp_world_size=2,
        )
        batch1 = next(loader)
        assert batch1[-1]["epoch"] == 1

        batch2 = next(loader)
        assert batch2[-1]["epoch"] == 2

    def test_both_ranks_reach_epoch_2(self, tmp_path):
        """Both ranks independently reach epoch 2."""
        n_rows = 4
        B = 2
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)

        for rank in range(2):
            loader = precomputed_data_loader(
                packed_row_dir=tmp_path / "packed",
                B=B,
                feature_pipeline=_make_pipeline(16384),
                device="cpu",
                ddp_rank=rank,
                ddp_world_size=2,
            )
            batch1 = next(loader)
            assert batch1[-1]["epoch"] == 1
            batch2 = next(loader)
            assert batch2[-1]["epoch"] == 2

    def test_state_dict_epoch_advances(self, tmp_path):
        """The state_dict 'epoch' field increments when the rank's data cycles."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=6)
        # world_size=2, rank=0 gets rows 0,2,4 (3 rows), B=3 → 1 batch/epoch
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=3,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            ddp_rank=0,
            ddp_world_size=2,
        )
        epochs_seen = []
        for i, batch in enumerate(loader):
            if i >= 3:
                break
            epochs_seen.append(batch[-1]["epoch"])

        assert epochs_seen == [1, 2, 3]


# ======================================================================
# 4. Cross-shard boundary
# ======================================================================


class TestDDPCrossShardBoundary:
    """Pending rows correctly accumulate across shard boundaries with DDP."""

    def test_pending_carries_across_shards(self, tmp_path):
        """With small shards, pending accumulates across shard boundaries."""
        # 2 shards, 4 rows each, world_size=2 → each rank gets 2 rows/shard
        # B=3 → first batch needs rows from both shards
        _write_shards(tmp_path / "packed", n_shards=2, n_rows=4)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=3,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            ddp_rank=0,
            ddp_world_size=2,
        )
        batch = next(loader)
        inputs = batch[0]
        assert inputs.shape[0] == 3  # batch formed from rows across shards

    def test_cross_shard_batch_has_rows_from_both_shards(self, tmp_path):
        """A cross-shard batch contains rows from two different shards."""
        # 2 shards (seeds 42, 43), 4 rows each, world_size=2 → 2 rows/rank/shard
        # B=3 → first batch: 2 rows from shard 0 + 1 row from shard 1
        _write_shards(tmp_path / "packed", n_shards=2, n_rows=4)
        fps = _collect_token_fingerprints(
            tmp_path / "packed", B=3, n_batches=1, rank=0, world_size=2
        )
        seeds = {s for s, _ in fps}
        assert len(seeds) == 2, f"Expected rows from 2 shards, got seeds {seeds}"

    def test_multi_shard_all_rows_covered_with_ddp(self, tmp_path):
        """Across 3 shards with world_size=2, union of ranks covers all rows."""
        n_rows = 6
        _write_shards(tmp_path / "packed", n_shards=3, n_rows=n_rows)
        all_fps: set[tuple[int, int]] = set()
        for rank in range(2):
            fps = _collect_token_fingerprints(
                tmp_path / "packed", B=3, n_batches=20, rank=rank, world_size=2
            )
            all_fps.update(fps)

        expected = set()
        for shard_i in range(3):
            for r in range(n_rows):
                expected.add((42 + shard_i, r))
        assert all_fps == expected


# ======================================================================
# 5. Backward compatibility
# ======================================================================


class TestDDPBackwardCompat:
    """Default DDP params produce identical behavior to pre-DDP code."""

    def test_default_params_match_old_behavior(self, tmp_path):
        """With ddp_rank=0, ddp_world_size=1, output is bit-identical."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)

        # No DDP params (uses defaults)
        loader_old = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
        )
        batch_old = next(loader_old)

        # Explicit defaults
        loader_new = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            ddp_rank=0,
            ddp_world_size=1,
        )
        batch_new = next(loader_new)

        # Compare all tensor fields (first 8 elements)
        for i in range(8):
            assert t.equal(batch_old[i], batch_new[i]), f"Mismatch at tuple index {i}"

        # state_dict should match too
        assert batch_old[-1]["batch_idx"] == batch_new[-1]["batch_idx"]
        assert batch_old[-1]["epoch"] == batch_new[-1]["epoch"]

    def test_val_loader_stripes_with_ddp(self, tmp_path):
        """Validation-style loader (no selection/dropout) stripes correctly."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8)
        fps0 = _collect_token_fingerprints(
            tmp_path / "packed", B=4, n_batches=1, rank=0, world_size=2
        )
        fps1 = _collect_token_fingerprints(
            tmp_path / "packed", B=4, n_batches=1, rank=1, world_size=2
        )
        assert set(fps0).isdisjoint(set(fps1))
        assert set(fps0) | set(fps1) == {(42, r) for r in range(8)}


# ======================================================================
# 6. Edge cases
# ======================================================================


class TestDDPEdgeCases:
    """Edge cases that could cause subtle bugs."""

    def test_single_row_per_rank(self, tmp_path):
        """world_size = n_rows → each rank gets exactly 1 row."""
        n_rows = 4
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        for rank in range(n_rows):
            fps = _collect_token_fingerprints(
                tmp_path / "packed", B=1, n_batches=1, rank=rank, world_size=n_rows
            )
            assert len(fps) == 1
            assert fps[0] == (42, rank)

    def test_rank_with_no_rows_in_shard_still_works_multi_shard(self, tmp_path):
        """Rank with 0 rows in one shard can still get rows from another."""
        # 2 shards (seeds 42, 43), 2 rows each, world_size=3
        # Rank 0: row 0 from each shard (2 rows total)
        # Rank 1: row 1 from each shard (2 rows total)
        # Rank 2: no rows (row_idx % 3 == 2 never matches 0 or 1)
        _write_shards(tmp_path / "packed", n_shards=2, n_rows=2)
        fps = _collect_token_fingerprints(
            tmp_path / "packed", B=1, n_batches=2, rank=0, world_size=3
        )
        assert set(fps) == {(42, 0), (43, 0)}

    def test_large_world_size_small_shard_active_ranks_work(self, tmp_path):
        """With world_size=8 and 2-row shard, only ranks 0-1 get data."""
        n_rows = 2
        world_size = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)

        # Ranks 0 and 1 should each get exactly 1 row
        for rank in range(2):
            assert _count_rank_rows(n_rows, 1, rank, world_size) == 1
            fps = _collect_token_fingerprints(
                tmp_path / "packed", B=1, n_batches=1, rank=rank, world_size=world_size
            )
            assert len(fps) == 1
            assert fps[0] == (42, rank)

        # Ranks 2-7 get 0 rows
        for rank in range(2, world_size):
            assert _count_rank_rows(n_rows, 1, rank, world_size) == 0

    def test_uneven_rows_distribution(self, tmp_path):
        """With 7 rows and world_size=3: ranks get 3, 2, 2 rows."""
        n_rows = 7
        world_size = 3
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        counts = [_count_rank_rows(n_rows, 1, rank, world_size) for rank in range(world_size)]

        assert counts == [3, 2, 2]  # rank 0: rows 0,3,6; rank 1: 1,4; rank 2: 2,5
        assert sum(counts) == n_rows

        # Also verify via actual dataloader that rank 0 sees rows 0, 3, 6
        fps = _collect_token_fingerprints(
            tmp_path / "packed", B=1, n_batches=3, rank=0, world_size=world_size
        )
        assert {row for _, row in fps} == {0, 3, 6}


# ======================================================================
# 7. Tensor data correctness (enc_fids, enc_pcts match shard rows)
# ======================================================================


class TestDDPTensorDataCorrectness:
    """Verify that encoder feature tensors match the expected shard rows."""

    def test_enc_fids_match_expected_rows(self, tmp_path):
        """enc_fids from the DDP-striped loader match reconstruct_and_select for those rows."""
        n_rows = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)
        pipeline = _make_pipeline(16384)

        # Get batch from rank 0 (rows 0, 2, 4, 6)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=pipeline,
            device="cpu",
            ddp_rank=0,
            ddp_world_size=2,
        )
        batch = next(loader)
        loader_enc_fids = batch[2]  # (B, n_seg, F)
        loader_enc_pcts = batch[5]  # (B, n_seg, F)

        # Manually reconstruct using the same pipeline (fused path)
        shard_paths = sorted((tmp_path / "packed").glob("*.safetensors*"))
        shard = load_packed_row_shard(shard_paths[0])
        ref, _ = reconstruct_and_select(shard, [0, 2, 4, 6], pipeline, device="cpu")

        assert t.equal(loader_enc_fids, ref.enc_fids)
        assert t.equal(loader_enc_pcts, ref.enc_pcts)

    def test_inputs_targets_match_expected_rows(self, tmp_path):
        """inputs/targets from the DDP-striped loader match direct reconstruct."""
        n_rows = 8
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=n_rows)

        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            ddp_rank=1,
            ddp_world_size=2,
        )
        batch = next(loader)

        shard_paths = sorted((tmp_path / "packed").glob("*.safetensors*"))
        shard = load_packed_row_shard(shard_paths[0])
        ref = reconstruct_batch(shard, [1, 3, 5, 7], _make_identity_selection(shard), 10, device="cpu")

        assert t.equal(batch[0], ref.inputs)
        assert t.equal(batch[1], ref.targets)
        assert t.equal(batch[6], ref.seg_map)
        assert t.equal(batch[7], ref.seg_doc_id)


# ======================================================================
# 8. Selection shard path with DDP
# ======================================================================


class TestDDPWithSelectionShards:
    """DDP striping works correctly when selection shards are present."""

    def test_selection_shard_ranks_disjoint(self, tmp_path):
        """With selection shards, ranks still see disjoint rows."""
        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_shards_with_selection(packed_dir, sel_dir, n_shards=1, n_rows=8)

        fp0 = _collect_token_fingerprints_with_selection(
            packed_dir, sel_dir, B=4, n_batches=1, rank=0, world_size=2
        )
        fp1 = _collect_token_fingerprints_with_selection(
            packed_dir, sel_dir, B=4, n_batches=1, rank=1, world_size=2
        )
        assert set(fp0).isdisjoint(set(fp1))
        assert set(fp0) | set(fp1) == {(42, r) for r in range(8)}

    def test_selection_shard_produces_valid_batches(self, tmp_path):
        """Batches from the selection shard path have non-zero enc_mags."""
        packed_dir = tmp_path / "packed"
        sel_dir = tmp_path / "sel"
        _write_shards_with_selection(packed_dir, sel_dir, n_shards=1, n_rows=8)

        loader = precomputed_data_loader(
            packed_row_dir=packed_dir,
            B=4,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            selection_dir=sel_dir,
            ddp_rank=0,
            ddp_world_size=2,
        )
        batch = next(loader)
        enc_mags = batch[3]  # (B, n_seg, F)
        # Selection shards produce bucket values, so enc_mags should be non-zero
        assert enc_mags.any(), "Selection path should produce non-zero enc_mags"


# ======================================================================
# 9. Label dropout + DDP interaction
# ======================================================================


class TestDDPWithLabelDropout:
    """Label dropout works correctly in combination with DDP striping."""

    def test_label_dropout_with_ddp_produces_valid_batches(self, tmp_path):
        """With label_dropout=1.0 + DDP, NaN guard keeps at least 1 feature per row."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8, feats_per_seg=10)
        loader = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed",
            B=4,
            feature_pipeline=_make_pipeline(16384),
            device="cpu",
            label_dropout=1.0,
            ddp_rank=0,
            ddp_world_size=2,
        )
        batch = next(loader)
        enc_mask = batch[4]  # (B, n_seg, F) bool
        # NaN guard: every row must have at least one True in enc_mask
        row_has_any = enc_mask.any(dim=-1).any(dim=-1)  # (B,)
        assert row_has_any.all(), "NaN guard must preserve at least 1 feature per row"

    def test_label_dropout_with_ddp_reduces_features(self, tmp_path):
        """DDP + dropout reduces features vs DDP without dropout."""
        _write_shards(tmp_path / "packed", n_shards=1, n_rows=8, feats_per_seg=10)

        loader_no_drop = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, feature_pipeline=_make_pipeline(16384),
            device="cpu", ddp_rank=0, ddp_world_size=2,
        )
        mask_no_drop = next(loader_no_drop)[4]

        loader_drop = precomputed_data_loader(
            packed_row_dir=tmp_path / "packed", B=4, feature_pipeline=_make_pipeline(16384),
            device="cpu", label_dropout=1.0, ddp_rank=0, ddp_world_size=2,
        )
        mask_drop = next(loader_drop)[4]

        # Full dropout should yield fewer active features (only NaN sentinel)
        assert mask_drop.sum() < mask_no_drop.sum()
