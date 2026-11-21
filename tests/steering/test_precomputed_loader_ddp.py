"""Tests for DDP row striping in precomputed_data_loader (fused path).

Verifies that ddp_rank / ddp_world_size correctly partition rows when using
the fused feature_pipeline path (FeaturePipelineConfig with threshold-based
selection).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch as t

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.packed_batch_io import save_packed_row_shard
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    _make_test_shard,
    build_threshold_pipeline,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write_shard_dir(tmp_path: Path, n_rows: int = 8, seed: int = 42) -> Path:
    """Write a single shard + manifest, return directory."""
    shard_dir = tmp_path / "packed"
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard = _make_test_shard(n_rows=n_rows, seed=seed)
    save_packed_row_shard(shard, shard_dir / "shard_00000")
    (shard_dir / "manifest.json").write_text(json.dumps({"metadata": {}}))
    return shard_dir


def _make_pipeline():
    """Build a FeaturePipelineConfig for the fused path."""
    boundaries = (
        t.linspace(1 / NUM_BUCKETS, 1.0 - 1 / NUM_BUCKETS, NUM_BUCKETS - 1, dtype=t.float64)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def _collect_inputs(shard_dir: Path, B: int, n_batches: int, rank: int, world_size: int) -> list[t.Tensor]:
    """Run loader for n_batches, return list of input tensors (B, T) per batch."""
    pipeline = _make_pipeline()
    loader = precomputed_data_loader(
        packed_row_dir=shard_dir,
        B=B,
        device="cpu",
        feature_pipeline=pipeline,
        n_sae_features=N_SAE_FEATURES,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    batches = []
    for i, batch in enumerate(loader):
        if i >= n_batches:
            break
        batches.append(batch[0].clone())  # inputs (B, T)
    return batches


# ---------------------------------------------------------------------------
# TestDDPRowStriping
# ---------------------------------------------------------------------------


class TestDDPRowStriping:
    """Verify DDP row striping partitions rows correctly with the fused pipeline."""

    def test_rank0_and_rank1_see_disjoint_batches(self, tmp_path):
        """Create 1 shard with 8 rows. rank=0,world=2 and rank=1,world=2 (B=2 each).
        First batches should have different inputs."""
        shard_dir = _write_shard_dir(tmp_path, n_rows=8)

        batches0 = _collect_inputs(shard_dir, B=2, n_batches=1, rank=0, world_size=2)
        batches1 = _collect_inputs(shard_dir, B=2, n_batches=1, rank=1, world_size=2)

        assert len(batches0) == 1
        assert len(batches1) == 1

        inputs0 = batches0[0]  # (2, T)
        inputs1 = batches1[0]  # (2, T)

        # The two ranks should not have the same inputs
        assert not t.equal(inputs0, inputs1), "Rank 0 and rank 1 first batches should differ"

        # More precisely: neither row of rank0 should appear in rank1
        for i in range(inputs0.shape[0]):
            for j in range(inputs1.shape[0]):
                assert not t.equal(inputs0[i], inputs1[j]), (
                    f"Row {i} from rank 0 matches row {j} from rank 1 — ranks are not disjoint"
                )

    def test_all_ranks_cover_all_rows(self, tmp_path):
        """6 rows, 3 ranks (B=2 each). Union of all rank batches should have 6 unique rows."""
        n_rows = 6
        shard_dir = _write_shard_dir(tmp_path, n_rows=n_rows)

        all_rows: list[t.Tensor] = []
        for rank in range(3):
            batches = _collect_inputs(shard_dir, B=2, n_batches=1, rank=rank, world_size=3)
            for batch in batches:
                for i in range(batch.shape[0]):
                    all_rows.append(batch[i])

        # Should have all 6 rows
        assert len(all_rows) == n_rows, f"Expected {n_rows} rows total, got {len(all_rows)}"

        # All rows should be unique (no duplicates across ranks)
        for i in range(len(all_rows)):
            for j in range(i + 1, len(all_rows)):
                assert not t.equal(all_rows[i], all_rows[j]), (
                    f"Rows {i} and {j} are identical — duplicate row across ranks"
                )

    def test_single_rank_sees_all_rows(self, tmp_path):
        """world_size=1, rank=0 gets all 4 rows in one batch."""
        n_rows = 4
        shard_dir = _write_shard_dir(tmp_path, n_rows=n_rows)

        batches = _collect_inputs(shard_dir, B=4, n_batches=1, rank=0, world_size=1)

        assert len(batches) == 1
        assert batches[0].shape[0] == n_rows, (
            f"Expected batch size {n_rows}, got {batches[0].shape[0]}"
        )
