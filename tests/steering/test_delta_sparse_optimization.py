"""Tests for the live document-batched delta implementation.

``compute_doc_deltas`` is checked against a frozen sequential implementation
that predates the document-batched optimization.
"""

import pytest
import torch as t

from steering.conditional.delta import (
    DeltaMode,
    compute_doc_deltas,
)

# ---------------------------------------------------------------------------
# Reference implementation (exact copy of the original compute_and_update)
# ---------------------------------------------------------------------------
from steering.conditional.percentile_table import magnitude_to_percentile


class _SequentialDeltaReference:
    """Frozen sequential oracle for the live document-batched implementation."""

    def __init__(self, mode: DeltaMode, n_sae_features: int) -> None:
        self.mode = mode
        self._ref = t.zeros(n_sae_features, dtype=t.float32)
        self._ref_active_fids: t.Tensor | None = None
        self._mask_buf = t.zeros(n_sae_features, dtype=t.bool)

    def compute_and_update(
        self,
        feature_ids: t.Tensor,
        feature_mags: t.Tensor,
        percentile_table: t.Tensor,
        precomputed_percentiles: t.Tensor | None = None,
    ) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor]:
        n = len(feature_ids)

        if n > 0:
            if precomputed_percentiles is not None:
                assert len(precomputed_percentiles) == n
                current_pcts = precomputed_percentiles.float()
            else:
                raw_pcts = magnitude_to_percentile(
                    feature_ids, feature_mags, percentile_table,
                )
                current_pcts = (raw_pcts / 100.0).float()
        else:
            current_pcts = t.tensor([], dtype=t.float32)

        if n > 0:
            fids_long = feature_ids if feature_ids.dtype == t.int64 else feature_ids.long()
            ref_for_current = self._ref[fids_long]
            current_deltas = current_pcts - ref_for_current
        else:
            fids_long = None
            current_deltas = t.tensor([], dtype=t.float32)

        prev_fids = self._ref_active_fids
        if prev_fids is None:
            nonzero = t.nonzero(self._ref > 0, as_tuple=True)[0]
            prev_fids = nonzero if len(nonzero) > 0 else None
        if prev_fids is not None and len(prev_fids) > 0:
            if n > 0:
                self._mask_buf.zero_()
                self._mask_buf[fids_long] = True
                ref_only_mask = ~self._mask_buf[prev_fids]
                ref_only_fids = prev_fids[ref_only_mask]
            else:
                ref_only_fids = prev_fids
        else:
            ref_only_fids = t.tensor([], dtype=t.int64)

        neg_parts_fids: list[t.Tensor] = []
        neg_parts_weights: list[t.Tensor] = []

        if len(ref_only_fids) > 0:
            ref_only_vals = self._ref[ref_only_fids]
            neg_parts_fids.append(ref_only_fids.to(t.int64))
            neg_parts_weights.append(ref_only_vals.float())

        if n > 0:
            neg_mask = current_deltas < 0
            if neg_mask.any():
                neg_parts_fids.append(feature_ids[neg_mask].to(t.int64))
                neg_parts_weights.append((-current_deltas[neg_mask]).float())

        if neg_parts_fids:
            neg_delta_fids = t.cat(neg_parts_fids)
            neg_delta_weights = t.cat(neg_parts_weights)
        else:
            neg_delta_fids = t.tensor([], dtype=t.int64)
            neg_delta_weights = t.tensor([], dtype=t.float32)

        self._mask_buf.zero_()
        if n > 0:
            self._mask_buf[fids_long] = True
        if len(ref_only_fids) > 0:
            self._mask_buf[ref_only_fids] = True
        ref_mask = self._mask_buf.clone()

        if self.mode is DeltaMode.prev_segment:
            self._ref.zero_()
            if n > 0:
                self._ref[fids_long] = current_pcts
                self._ref_active_fids = fids_long.clone()  # ty: ignore[unresolved-attribute]
            else:
                self._ref_active_fids = None
        else:  # cum_max
            if n > 0:
                self._ref[fids_long] = t.maximum(
                    self._ref[fids_long], current_pcts,
                )
                if prev_fids is not None and len(prev_fids) > 0:
                    combined = t.cat([prev_fids, fids_long])  # ty: ignore[invalid-argument-type]
                    self._ref_active_fids = combined.unique()
                else:
                    self._ref_active_fids = fids_long.clone()  # ty: ignore[unresolved-attribute]

        return current_deltas, neg_delta_fids, neg_delta_weights, ref_mask


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_percentile_table(n_sae: int = 16384) -> t.Tensor:
    """Create a simple percentile table for testing."""
    return t.linspace(0.0, 10.0, n_sae * 99).reshape(n_sae, 99).double()


def _make_segment(n_features: int, n_sae: int, seed: int) -> tuple[t.Tensor, t.Tensor, t.Tensor]:
    """Create random feature_ids, mags, and precomputed_pcts for a segment."""
    rng = t.Generator().manual_seed(seed)
    fids = t.randperm(n_sae, generator=rng)[:n_features].to(t.int64)
    mags = t.rand(n_features, generator=rng) * 5.0
    pcts = t.rand(n_features, generator=rng)  # normalized [0, 1]
    return fids, mags, pcts


# ---------------------------------------------------------------------------
# Tests for compute_doc_deltas (batched)
# ---------------------------------------------------------------------------


def _run_batch_vs_sequential(
    mode: DeltaMode,
    n_sae: int,
    segments: list[tuple[t.Tensor, t.Tensor, t.Tensor]],
    pct_table: t.Tensor,
):
    """Compare the live batched path against the frozen sequential oracle."""
    state = _SequentialDeltaReference(mode, n_sae)
    seq_results = []
    for fids, mags, pcts in segments:
        deltas, neg_fids, neg_w, ref_mask = state.compute_and_update(
            fids, mags, pct_table, precomputed_percentiles=pcts,
        )
        if len(fids) > 0:
            positive = deltas > 0
            f_fids = fids[positive]
            f_mags = mags[positive]
            f_deltas = deltas[positive]
        else:
            f_fids, f_mags, f_deltas = fids, mags, None
        seq_results.append((f_fids, f_mags, f_deltas, neg_fids, neg_w, ref_mask))

    # Batched
    batch_results = compute_doc_deltas(
        [s[0] for s in segments],
        [s[1] for s in segments],
        [s[2] for s in segments],
        n_sae, mode, pct_table,
    )

    assert len(batch_results) == len(seq_results), (
        f"Result count mismatch: {len(batch_results)} vs {len(seq_results)}"
    )

    for s_idx, (seq, batch) in enumerate(zip(seq_results, batch_results, strict=True)):
        label = f"seg {s_idx}"
        # filtered_fids
        assert t.equal(seq[0], batch[0]), f"{label} filtered_fids mismatch"
        # filtered_fmags
        assert t.equal(seq[1], batch[1]), f"{label} filtered_fmags mismatch"
        assert seq[2] is not None
        assert batch[2] is not None
        assert t.allclose(seq[2], batch[2], atol=1e-7), (
            f"{label} delta_values mismatch: {seq[2]} vs {batch[2]}"
        )
        # neg_delta_fids
        assert t.equal(seq[3], batch[3]), (
            f"{label} neg_delta_fids mismatch:\n  seq: {seq[3]}\n  batch: {batch[3]}"
        )
        # neg_delta_weights
        assert t.allclose(seq[4], batch[4], atol=1e-7), (
            f"{label} neg_delta_weights mismatch:\n  seq: {seq[4]}\n  batch: {batch[4]}"
        )
        # ref_mask
        assert t.equal(seq[5], batch[5]), f"{label} ref_mask mismatch"


class TestComputeDocDeltasBatch:
    """The batched implementation matches the frozen sequential oracle."""

    N_SAE = 16384

    def test_single_segment(self):
        pct_table = _make_percentile_table(self.N_SAE)
        fids, mags, pcts = _make_segment(10, self.N_SAE, seed=42)
        _run_batch_vs_sequential(
            DeltaMode.prev_segment, self.N_SAE, [(fids, mags, pcts)], pct_table,
        )

    def test_two_segments_partial_overlap(self):
        pct_table = _make_percentile_table(self.N_SAE)
        fids1 = t.tensor([0, 1, 2, 3, 4], dtype=t.int64)
        fids2 = t.tensor([3, 4, 5, 6, 7], dtype=t.int64)
        pcts1 = t.tensor([0.5, 0.6, 0.7, 0.8, 0.9])
        pcts2 = t.tensor([0.4, 0.3, 0.5, 0.6, 0.7])
        mags = t.ones(5)
        _run_batch_vs_sequential(
            DeltaMode.prev_segment, self.N_SAE,
            [(fids1, mags, pcts1), (fids2, mags, pcts2)], pct_table,
        )

    def test_many_segments_varied_sizes(self):
        pct_table = _make_percentile_table(self.N_SAE)
        segments = []
        for i in range(7):
            n_feat = 5 + i * 3
            fids, mags, pcts = _make_segment(n_feat, self.N_SAE, seed=100 + i)
            segments.append((fids, mags, pcts))
        _run_batch_vs_sequential(
            DeltaMode.prev_segment, self.N_SAE, segments, pct_table,
        )

    def test_empty_segment_in_middle_raises(self):
        pct_table = _make_percentile_table(self.N_SAE)
        fids1, mags1, pcts1 = _make_segment(8, self.N_SAE, seed=1)
        empty = (t.tensor([], dtype=t.int64), t.tensor([], dtype=t.float32),
                 t.tensor([], dtype=t.float32))
        fids3, mags3, pcts3 = _make_segment(8, self.N_SAE, seed=3)
        with pytest.raises(ValueError, match="zero feature ids"):
            compute_doc_deltas(
                [fids1, empty[0], fids3],
                [mags1, empty[1], mags3],
                [pcts1, empty[2], pcts3],
                self.N_SAE,
                DeltaMode.prev_segment,
                pct_table,
            )

    def test_all_negative_deltas(self):
        pct_table = _make_percentile_table(self.N_SAE)
        fids = t.arange(0, 5, dtype=t.int64)
        mags = t.ones(5)
        _run_batch_vs_sequential(
            DeltaMode.prev_segment, self.N_SAE,
            [(fids, mags, t.full((5,), 0.9)), (fids, mags, t.full((5,), 0.1))],
            pct_table,
        )

    def test_int32_fids(self):
        pct_table = _make_percentile_table(self.N_SAE)
        fids1 = t.tensor([100, 200, 300], dtype=t.int32)
        fids2 = t.tensor([200, 300, 400], dtype=t.int32)
        mags = t.ones(3)
        _run_batch_vs_sequential(
            DeltaMode.prev_segment, self.N_SAE,
            [(fids1, mags, t.tensor([0.5, 0.6, 0.7])),
             (fids2, mags, t.tensor([0.4, 0.8, 0.5]))],
            pct_table,
        )

    def test_cum_max_mode(self):
        pct_table = _make_percentile_table(self.N_SAE)
        fids = t.arange(0, 5, dtype=t.int64)
        mags = t.ones(5)
        _run_batch_vs_sequential(
            DeltaMode.cum_max, self.N_SAE,
            [
                (fids, mags, t.tensor([0.3, 0.4, 0.5, 0.6, 0.7])),
                (fids, mags, t.tensor([0.5, 0.2, 0.8, 0.4, 0.9])),
                (fids, mags, t.tensor([0.1, 0.1, 0.1, 0.1, 0.1])),
            ],
            pct_table,
        )

    def test_many_docs_independent(self):
        """10 independent docs — ensures no state leaks between calls."""
        pct_table = _make_percentile_table(self.N_SAE)
        for doc in range(10):
            segments = []
            for seg in range(6):
                fids, mags, pcts = _make_segment(
                    8 + seg, self.N_SAE, seed=doc * 100 + seg
                )
                segments.append((fids, mags, pcts))
            _run_batch_vs_sequential(
                DeltaMode.prev_segment, self.N_SAE, segments, pct_table,
            )
