"""Tests for epoch cycling and state_dict tracking in precomputed_data_loader.

Verifies that:
- batch_idx increments correctly across batches
- epoch increments when all rows are consumed and cycling occurs
- state_dict contains a "shard" key with a string value
- the loader never crashes over many draws
- epoch cycling produces identical data each cycle
"""

from __future__ import annotations

import torch as t

from tests.encdec_test_helpers import make_precomputed_dataloader


class TestEpochCycling:
    """Tests for epoch boundary detection and state_dict tracking."""

    def test_state_dict_batch_idx_increments(self, tmp_path):
        """batch_idx in state_dict should be 0, 1, 2 for first three batches."""
        loader = make_precomputed_dataloader(tmp_path, B=2, n_rows=8)

        batch_indices = []
        for _ in range(3):
            batch = next(loader)
            state = batch[9]
            batch_indices.append(state["batch_idx"])

        assert batch_indices == [0, 1, 2], (
            f"Expected batch_idx to increment as [0, 1, 2], got {batch_indices}"
        )

    def test_state_dict_epoch_increments(self, tmp_path):
        """With B=4 and n_rows=4, first batch is epoch 1 and second is epoch 2."""
        loader = make_precomputed_dataloader(tmp_path, B=4, n_rows=4)

        batch1 = next(loader)
        epoch1 = batch1[9]["epoch"]

        batch2 = next(loader)
        epoch2 = batch2[9]["epoch"]

        assert epoch1 == 1, f"First batch should have epoch=1, got {epoch1}"
        assert epoch2 == 2, (
            f"Second batch (after cycling) should have epoch=2, got {epoch2}"
        )

    def test_state_dict_shard_name(self, tmp_path):
        """state_dict must have a 'shard' key whose value is a string."""
        loader = make_precomputed_dataloader(tmp_path, B=2, n_rows=8)

        batch = next(loader)
        state = batch[9]

        assert "shard" in state, "state_dict must contain 'shard' key"
        assert isinstance(state["shard"], str), (
            f"state_dict['shard'] must be a str, got {type(state['shard'])}"
        )

    def test_many_batches_no_crash(self, tmp_path):
        """Drawing 20 batches should not crash and all must have shape[0] == B."""
        B = 2
        loader = make_precomputed_dataloader(tmp_path, B=B, n_rows=8)

        for i in range(20):
            batch = next(loader)
            inputs = batch[0]
            assert inputs.shape[0] == B, (
                f"Batch {i}: expected inputs.shape[0]={B}, got {inputs.shape[0]}"
            )

    def test_epoch_cycling_produces_same_data(self, tmp_path):
        """With B=4 and n_rows=4, epoch 1 and epoch 2 inputs must be identical."""
        loader = make_precomputed_dataloader(tmp_path, B=4, n_rows=4)

        batch1 = next(loader)
        inputs_epoch1 = batch1[0]
        assert batch1[9]["epoch"] == 1

        batch2 = next(loader)
        inputs_epoch2 = batch2[0]
        assert batch2[9]["epoch"] == 2

        assert t.equal(inputs_epoch1, inputs_epoch2), (
            "Epoch 1 and epoch 2 inputs differ — cycling should reproduce the same data"
        )
