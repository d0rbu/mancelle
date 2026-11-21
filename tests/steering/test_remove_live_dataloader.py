"""Tests for the live dataloader removal migration."""

import torch as t

from tests.encdec_test_helpers import (
    SEQ_LEN,
    make_precomputed_dataloader,
)


class TestMakePrecomputedDataloader:
    """The new test helper yields valid batches from temp packed row shards."""

    def test_yields_10_tuple(self, tmp_path):
        dl = make_precomputed_dataloader(tmp_path, B=2)
        batch = next(dl)
        assert len(batch) == 10

    def test_batch_shapes(self, tmp_path):
        B, T = 2, SEQ_LEN
        dl = make_precomputed_dataloader(tmp_path, B=B)
        inputs, targets, enc_fids, enc_mags, enc_mask, enc_pcts, seg_map, seg_doc_id, fp_targets, state = next(dl)
        assert inputs.shape == (B, T)
        assert targets.shape == (B, T)
        assert seg_map.shape == (B, T)
        assert enc_fids.shape[0] == B
        assert enc_mask.dtype == t.bool
        assert enc_mask.any(), "At least some features should be active"

    def test_fp_targets_populated(self, tmp_path):
        """fp_targets should have non-zero entries (all extracted features)."""
        dl = make_precomputed_dataloader(tmp_path, B=2)
        batch = next(dl)
        fp_targets = batch[8]
        assert fp_targets is not None
        assert fp_targets.dtype == t.float32
        assert (fp_targets > 0).any()

    def test_multiple_batches(self, tmp_path):
        """Can yield multiple batches without error."""
        dl = make_precomputed_dataloader(tmp_path, B=2, n_rows=8)
        for _ in range(3):
            batch = next(dl)
            assert len(batch) == 10

    def test_state_dict_present(self, tmp_path):
        dl = make_precomputed_dataloader(tmp_path, B=2)
        batch = next(dl)
        state = batch[9]
        assert isinstance(state, dict)
        assert "batch_idx" in state

    def test_label_dropout(self, tmp_path):
        """Label dropout should reduce active features."""
        dl_no_drop = make_precomputed_dataloader(tmp_path / "a", B=2, label_dropout=0.0)
        _, _, _, _, mask_no_drop, *_ = next(dl_no_drop)

        dl_drop = make_precomputed_dataloader(tmp_path / "b", B=2, label_dropout=1.0, seed=99)
        _, _, _, _, mask_drop, *_ = next(dl_drop)

        # Full dropout leaves only NaN-guard sentinels (1 per row = B)
        assert mask_drop.sum().item() <= 2  # B sentinel features
        assert mask_drop.sum().item() < mask_no_drop.sum().item()
