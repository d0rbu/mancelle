"""Tests for feature_mask wiring into EncoderDecoderGPT FP-loss objective.

Verifies:
- test_model_feature_mask_default_none: freshly constructed model has feature_mask=None
- test_forward_passes_feature_mask_to_fp_loss: setting model.feature_mask causes the
  mask to genuinely reach the loss (different loss, zero grad on masked dim K)
- test_no_mask_forward_byte_identical: feature_mask=None (default) gives byte-identical
  loss to the same forward with no explicit mask set (I5 invariant)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT, ForwardResult
from steering.conditional.feature_mask import (
    FeatureMask,
    FeatureMaskMode,
    canonical_dict,
)
from steering.conditional.splice_counterfactual import packed_row_fingerprint
from steering.overlay_io import OverlayKind, write_manifest
from steering.packed_batch_io import SelectionShard, save_packed_row_shard, save_selection_shard
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    NUM_BUCKETS,
    SEQ_LEN,
    _make_test_shard,
    build_model_with_feature_prediction,
    build_threshold_pipeline,
)
from tests.feature_prediction_helpers import dense_targets_from_sparse

# ---------------------------------------------------------------------------
# Module-level fixture: force SDPA backend (FlexAttention doesn't support
# backward on CPU, which is needed for the gradient zero-check test).
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _force_sdpa():
    """Force SDPA backend for all tests in this module."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_fp_batch(
    B: int = 1,
    T: int = SEQ_LEN,
    n_sae: int = N_SAE_FEATURES,
    n_seg: int = 4,
    F: int = 5,
    num_buckets: int = NUM_BUCKETS,
    seed: int = 0,
):
    """Create a minimal synthetic batch suitable for a forward with fp_targets.

    Returns idx, targets, enc_fids, enc_mags, enc_mask, seg_map,
            seg_doc_id, fp_targets.
    """
    torch.manual_seed(seed)
    idx = torch.randint(0, 100, (B, T))
    targets = torch.randint(0, 100, (B, T))
    # Use small IDs so feature K can be planted later at a known position
    enc_fids = torch.randint(0, n_sae, (B, n_seg, F), dtype=torch.long)
    enc_mags = torch.randint(0, num_buckets, (B, n_seg, F), dtype=torch.long)
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    enc_pcts = torch.rand(B, n_seg, F)
    seg_size = T // n_seg
    seg_map = torch.arange(T).unsqueeze(0).expand(B, -1) // seg_size
    seg_map = seg_map.clamp(max=n_seg - 1)
    seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)
    # Build dense fp_targets from sparse encoder inputs
    fp_targets = dense_targets_from_sparse(
        enc_fids.flatten(0, 1),
        enc_pcts.flatten(0, 1),
        enc_mask.flatten(0, 1),
        n_sae,
    ).view(B, n_seg, n_sae)
    return idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets


def _run_forward(
    model: EncoderDecoderGPT,
    idx,
    targets,
    enc_fids,
    enc_mags,
    enc_mask,
    seg_map,
    seg_doc_id,
    fp_targets,
) -> ForwardResult:
    """Run a forward that triggers the FP-loss branch and return ForwardResult."""
    result = model(
        idx,
        targets,
        enc_fids,
        enc_mags,
        enc_mask,
        seg_map,
        seg_doc_id=seg_doc_id,
        fp_targets=fp_targets,
    )
    assert isinstance(result, ForwardResult), (
        f"Expected ForwardResult (model has feature_pred_config), got {type(result)}"
    )
    return result


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestModelFeatureMaskDefaultNone:
    """Freshly constructed EncoderDecoderGPT must have feature_mask=None."""

    def test_model_feature_mask_default_none(self):
        """build_model_with_feature_prediction() ⇒ feature_mask is None."""
        model = build_model_with_feature_prediction()
        assert hasattr(model, "feature_mask"), (
            "EncoderDecoderGPT must have a 'feature_mask' attribute"
        )
        assert model.feature_mask is None, (
            f"Expected feature_mask=None on fresh model, got {model.feature_mask!r}"
        )

    def test_attribute_settable(self):
        """feature_mask can be set to a FeatureMask without error."""
        model = build_model_with_feature_prediction()
        mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(0, 1, 2))
        model.feature_mask = mask
        assert model.feature_mask is mask

    def test_attribute_settable_back_to_none(self):
        """feature_mask can be set back to None after being set."""
        model = build_model_with_feature_prediction()
        model.feature_mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(5,))
        model.feature_mask = None
        assert model.feature_mask is None


def _make_fp_targets_only_column_k(
    B: int,
    n_seg: int,
    n_sae: int,
    K: int,
    pct_value: float = 0.9,
) -> torch.Tensor:
    """Return fp_targets with only column K set to pct_value, all others zero."""
    fp_targets = torch.zeros(B, n_seg, n_sae)
    fp_targets[:, :, K] = pct_value
    return fp_targets


class TestForwardPassesFeatureMaskToFpLoss:
    """Setting model.feature_mask genuinely reaches the FP-loss computation.

    Strategy:
    1. Pick a specific feature column K.
    2. Build fp_targets where ONLY column K is nonzero (plant K as an active
       feature), so the FP loss is dominated by that column.
    3. Run forward with model.feature_mask=None ⇒ baseline loss L_baseline.
    4. Run forward with model.feature_mask=blacklist({K}) ⇒ masked loss L_masked.
    5. Assert L_masked != L_baseline (mask reached loss) AND that perturbing
       only logit column K leaves L_masked unchanged (K excluded from reduction).
    """

    def test_mask_changes_fp_loss(self):
        """Blacklisting feature K changes the fp_loss scalar when K was the only target."""
        torch.manual_seed(42)
        K = 7  # arbitrary feature dim to mask
        B, T, n_seg = 1, SEQ_LEN, 4
        n_sae = N_SAE_FEATURES

        model = build_model_with_feature_prediction(n_sae_features=n_sae)
        model.eval()

        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, _ = _make_fp_batch(
            B=B, T=T, n_sae=n_sae, n_seg=n_seg, seed=42
        )
        # Plant: only column K carries a nonzero target
        fp_targets = _make_fp_targets_only_column_k(B, n_seg, n_sae, K)

        with torch.no_grad():
            # Baseline: no mask
            model.feature_mask = None
            result_no_mask = _run_forward(
                model,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_targets,
            )

            # Masked: blacklist K → column K excluded from loss mean
            model.feature_mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(K,))
            result_masked = _run_forward(
                model,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_targets,
            )

        fp_no_mask = result_no_mask.feature_pred_loss.item()
        fp_masked = result_masked.feature_pred_loss.item()

        assert fp_no_mask != fp_masked, (
            f"feature_pred_loss identical with and without mask on column K={K}: "
            f"{fp_no_mask}. The mask did not reach the loss."
        )

    def test_perturbing_masked_column_leaves_loss_unchanged(self):
        """After masking column K, perturbing fp_targets column K has no effect on fp_loss."""
        torch.manual_seed(99)
        K = 3  # feature dim to mask
        B, T, n_seg = 1, SEQ_LEN, 4
        n_sae = N_SAE_FEATURES

        model = build_model_with_feature_prediction(n_sae_features=n_sae)
        model.eval()
        model.feature_mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(K,))

        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, _ = _make_fp_batch(
            B=B, T=T, n_sae=n_sae, n_seg=n_seg, seed=99
        )
        fp_base = _make_fp_targets_only_column_k(B, n_seg, n_sae, K, pct_value=0.5)
        fp_perturbed = fp_base.clone()
        fp_perturbed[:, :, K] = 1.0  # change only the masked column

        with torch.no_grad():
            r_base = _run_forward(
                model,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_base,
            )
            r_pert = _run_forward(
                model,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_perturbed,
            )

        assert r_base.feature_pred_loss.item() == r_pert.feature_pred_loss.item(), (
            f"Perturbing only masked column K={K} changed fp_loss: "
            f"{r_base.feature_pred_loss.item()} vs {r_pert.feature_pred_loss.item()}. "
            "Column K was not excluded from the loss reduction."
        )

    def test_grad_on_masked_column_is_zero(self):
        """Gradient on fp_logit column K is zero when K is blacklisted."""
        torch.manual_seed(77)
        K = 5
        B, T, n_seg = 1, SEQ_LEN, 4
        n_sae = N_SAE_FEATURES

        model = build_model_with_feature_prediction(n_sae_features=n_sae)
        model.train()
        model.feature_mask = FeatureMask(mode=FeatureMaskMode.blacklist, ids=(K,))

        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, _ = _make_fp_batch(
            B=B, T=T, n_sae=n_sae, n_seg=n_seg, seed=77
        )
        # All features active so there is a meaningful loss on other dims
        fp_targets = torch.rand(B, n_seg, n_sae)
        fp_targets[:, :, K] = 0.9  # nonzero on K — should still produce zero grad

        result = _run_forward(
            model,
            idx,
            targets,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id,
            fp_targets,
        )
        result.loss.backward()

        # Inspect gradients on feature_pred_head weight rows corresponding to K
        # The head is Linear(n_embd → n_sae); weight row K is fp_head.weight[K, :]
        fp_head = model.feature_pred_head
        assert fp_head is not None
        grad_row_k = fp_head.weight.grad[K]  # gradient for output dim K
        assert grad_row_k.abs().max().item() == 0.0, (
            f"fp_head.weight.grad[{K}] has non-zero entries after backward with "
            f"feature_mask=blacklist({{{K}}}). Max abs grad: "
            f"{grad_row_k.abs().max().item()}. Mask did not reach the gradient."
        )


class TestNoMaskForwardByteIdentical:
    """I5: feature_mask=None (default) produces byte-identical loss vs explicit None."""

    def test_none_mask_forward_byte_identical_loss(self):
        """Two forwards, both with feature_mask=None, give identical combined loss."""
        torch.manual_seed(11)
        B, T, n_seg = 1, SEQ_LEN, 4
        n_sae = N_SAE_FEATURES

        model = build_model_with_feature_prediction(n_sae_features=n_sae)
        model.eval()

        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = (
            _make_fp_batch(B=B, T=T, n_sae=n_sae, n_seg=n_seg, seed=11)
        )

        # Both passes: feature_mask stays None (default)
        model.feature_mask = None
        with torch.no_grad():
            r1 = _run_forward(
                model,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_targets,
            )
            r2 = _run_forward(
                model,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_targets,
            )

        assert r1.loss.item() == r2.loss.item(), (
            f"Byte-identity violated: {r1.loss.item()} vs {r2.loss.item()}"
        )
        assert r1.feature_pred_loss.item() == r2.feature_pred_loss.item(), (
            f"FP loss differs: {r1.feature_pred_loss.item()} vs {r2.feature_pred_loss.item()}"
        )

    def test_none_mask_vs_no_mask_attribute_identical(self):
        """feature_mask=None (set explicitly) vs fresh model (attribute never set)
        produce identical fp_loss — verifies the default=None path is the same code path."""
        torch.manual_seed(22)
        B, T, n_seg = 1, SEQ_LEN, 4
        n_sae = N_SAE_FEATURES

        # Model A: feature_mask left at default (None)
        model_a = build_model_with_feature_prediction(n_sae_features=n_sae)
        model_a.eval()
        assert model_a.feature_mask is None

        # Model B: explicitly set to None (should be the same code path)
        model_b = build_model_with_feature_prediction(n_sae_features=n_sae)
        model_b.eval()
        model_b.feature_mask = None

        # Copy weights from A to B so forward is numerically identical
        model_b.load_state_dict(model_a.state_dict())

        idx, targets, enc_fids, enc_mags, enc_mask, seg_map, seg_doc_id, fp_targets = (
            _make_fp_batch(B=B, T=T, n_sae=n_sae, n_seg=n_seg, seed=22)
        )

        with torch.no_grad():
            r_a = _run_forward(
                model_a,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_targets,
            )
            r_b = _run_forward(
                model_b,
                idx,
                targets,
                enc_fids,
                enc_mags,
                enc_mask,
                seg_map,
                seg_doc_id,
                fp_targets,
            )

        assert r_a.loss.item() == r_b.loss.item(), (
            f"I5 violated: model_a loss {r_a.loss.item()} != model_b loss "
            f"{r_b.loss.item()} even though both have feature_mask=None."
        )
        assert r_a.feature_pred_loss.item() == r_b.feature_pred_loss.item(), (
            f"I5 violated: fp_loss differs: {r_a.feature_pred_loss.item()} vs "
            f"{r_b.feature_pred_loss.item()}"
        )


# ---------------------------------------------------------------------------
# Selection-shard call-site wiring: Task 13b / §5.4 consistency fix
# ---------------------------------------------------------------------------


def _make_mask_filtered_selection(shard, excluded_ids: frozenset[int]) -> SelectionShard:
    """Create a SelectionShard that mirrors what real precompute produces under a mask.

    Real ``precompute_selected_features`` calls ``feature_mask.apply_to_positive``
    (scripts/preprocess/precompute_selected_features.py:L116) to strip masked ids
    from the positive feature set BEFORE selection, then writes only the filtered
    ids into the sel/<hash>/ shard.  The BASE shard's CSR data is untouched.

    This helper replicates that behaviour: for each segment it copies the base
    shard's feature_ids EXCEPT those in ``excluded_ids``, building consistent
    CSR offsets.  No neg features are included (negative_bundle is None in the
    test pipeline).

    This is what the c177b5db guard (dataloader.py:2421-2422) expects:
    enc_fids (read from sel_shard.sel_fids via reconstruct_batch) must never
    contain masked ids — guaranteed by real precompute and now correctly modelled
    by this fixture.
    """
    n_segs = int(shard.seg_feat_offsets.numel() - 1)
    feat_offsets = shard.seg_feat_offsets.long().tolist()

    out_fids: list[int] = []
    out_pcts: list[int] = []
    out_offsets: list[int] = [0]

    for seg in range(n_segs):
        f_start, f_end = feat_offsets[seg], feat_offsets[seg + 1]
        seg_fids = shard.feature_ids[f_start:f_end].to(torch.int64)
        seg_pcts = shard.feature_pcts[f_start:f_end]
        keep = torch.tensor([int(f.item()) not in excluded_ids for f in seg_fids], dtype=torch.bool)
        kept_fids = seg_fids[keep].tolist()
        kept_pcts = seg_pcts[keep].tolist()
        out_fids.extend(kept_fids)
        out_pcts.extend(kept_pcts)
        out_offsets.append(len(out_fids))

    sel_fids_t = (
        torch.tensor(out_fids, dtype=torch.uint16)
        if out_fids
        else torch.tensor([], dtype=torch.uint16)
    )
    sel_pcts_t = (
        torch.tensor(out_pcts, dtype=torch.uint8)
        if out_pcts
        else torch.tensor([], dtype=torch.uint8)
    )
    sel_buckets_t = torch.ones(len(out_fids), dtype=torch.uint8)

    return SelectionShard(
        sel_offsets=torch.tensor(out_offsets, dtype=torch.int32),
        sel_fids=sel_fids_t,
        sel_buckets=sel_buckets_t,
        sel_pcts=sel_pcts_t,
        neg_offsets=torch.zeros(n_segs + 1, dtype=torch.int32),
        neg_fids=torch.tensor([], dtype=torch.uint16),
    )


def _make_shard_with_feature_k(
    tmp_path: Path,
    n_sae: int,
    K: int,
    n_rows: int = 4,
    segs_per_row: int = 2,
    seed: int = 7,
) -> tuple[Path, Path]:
    """Create a packed-row shard and a matching REALISTIC selection shard.

    The BASE shard's CSR data contains feature K (so ``build_fp_targets_from_shard``
    scatters it into the dense fp_targets tensor — the thing under test is that the
    mask then zeroes that column).

    The SELECTION shard's sel_fids have K excluded, mirroring what real
    ``precompute_selected_features`` produces when run with a blacklist{K}:
      - scripts/preprocess/precompute_selected_features.py:L116 calls
        ``feature_mask.apply_to_positive(fids, pcts)`` before selection,
        stripping K from the per-segment feature list before any
        sel/<hash>/ shard entry is written.
      - L188 then ``assert_no_masked(selected_fids)`` verifies the write.
    A real masked run therefore never produces a sel/<hash>/ shard with K in
    sel_fids, so the c177b5db guard (dataloader.py:2421-2422) that asserts
    enc_fids[enc_mask] has no masked ids is consistent with real precompute.

    Returns (shard_dir, sel_dir).
    """
    shard_dir = tmp_path / "packed"
    shard_dir.mkdir(parents=True)
    sel_dir = tmp_path / "sel"
    sel_dir.mkdir(parents=True)

    shard = _make_test_shard(
        n_rows=n_rows,
        segs_per_row=segs_per_row,
        n_features=n_sae,
        seed=seed,
    )

    # Ensure feature K appears at least once in the BASE shard's CSR data so
    # build_fp_targets_from_shard has a nonzero value at column K to zero out.
    # _make_test_shard uses randint(0, n_features) for feature_ids, but K may or
    # may not be present; force it to guarantee the assertion is meaningful.
    if shard.feature_ids.numel() > 0:
        shard.feature_ids[0] = K
        # Give it a nonzero pct so the dense target is nonzero (not 0/255 = 0.0)
        shard.feature_pcts[0] = 200  # 200/255 ≈ 0.784

    save_packed_row_shard(shard, shard_dir / "shard_00000")

    # Build a REALISTIC selection shard: K excluded from sel_fids (mask-filtered),
    # just as real precompute would have produced under a blacklist{K}.
    sel = _make_mask_filtered_selection(shard, excluded_ids=frozenset({K}))
    save_selection_shard(sel, sel_dir / "shard_00000")

    # Write a minimal manifest so the loader's read_manifest doesn't raise
    manifest_path = shard_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"metadata": {}}))

    return shard_dir, sel_dir


def _write_selection_manifest(
    sel_dir: Path,
    shard_dir: Path,
    feature_mask: FeatureMask | None,
) -> None:
    metadata = {
        "packed_rows": {
            "packed_row_fingerprint": packed_row_fingerprint(shard_dir),
        },
    }
    mask_meta = canonical_dict(feature_mask)
    if mask_meta is not None:
        metadata["feature_mask"] = mask_meta
    write_manifest(
        sel_dir,
        "test-selection",
        OverlayKind.PACKED_ROW,
        metadata=metadata,
    )


class TestSelectionShardCallSiteMaskWiring:
    """Feature mask reaches build_fp_targets_from_shard in the selection-shard branch.

    This is the consistency/defense-in-depth fix from Task 13b §5.4. The two
    call sites at lines ~2163 and ~2194 of dataloader.py (within the
    `sel0 is not None` branches of precomputed_data_loader) must forward
    feature_pipeline.feature_mask so that masked columns are zeroed in fp_targets.

    The model-level gate from Task 13b still excludes masked dims from the loss,
    but these call-site fixes make ALL build_fp_targets_from_shard invocations
    uniformly mask-gated.
    """

    def test_single_shard_selection_branch_masks_column_k(self, tmp_path):
        """Same-shard batch (sel0 is not None path) zeros masked column K in fp_targets.

        This exercises the single-shard fast path (~line 2163): all rows come from
        the same shard and a selection shard is present.
        """
        K = 10  # feature to blacklist
        n_sae = N_SAE_FEATURES
        B = 2

        shard_dir, sel_dir = _make_shard_with_feature_k(tmp_path, n_sae=n_sae, K=K, n_rows=B * 2)

        boundaries = (
            torch.linspace(
                1 / NUM_BUCKETS,
                1.0 - 1 / NUM_BUCKETS,
                NUM_BUCKETS - 1,
                dtype=torch.float64,
            )
            .unsqueeze(0)
            .expand(n_sae, -1)
            .contiguous()
        )
        pipeline_masked = build_threshold_pipeline(boundaries)
        pipeline_masked.feature_mask = FeatureMask(
            mode=FeatureMaskMode.blacklist,
            ids=(K,),
        )
        _write_selection_manifest(sel_dir, shard_dir, pipeline_masked.feature_mask)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=B,
            feature_pipeline=pipeline_masked,
            device="cpu",
            selection_dir=sel_dir,
            n_sae_features=n_sae,
        )

        # fp_targets is the 9th element (index 8) of the yielded 10-tuple
        batch = next(loader)
        fp_targets = batch[8]  # shape [B, n_seg, n_sae] or None

        assert fp_targets is not None, (
            "fp_targets is None — n_sae_features was not threaded through the loader"
        )
        assert fp_targets.shape[-1] == n_sae, (
            f"fp_targets last dim {fp_targets.shape[-1]} != n_sae {n_sae}"
        )
        assert fp_targets[:, :, K].abs().max().item() == 0.0, (
            f"Selection-shard branch did NOT zero column K={K} in fp_targets. "
            f"Max abs value at col K: {fp_targets[:, :, K].abs().max().item():.4f}. "
            "feature_mask was not forwarded to build_fp_targets_from_shard at ~line 2163."
        )

    def test_single_shard_selection_branch_unmasked_column_nonzero(self, tmp_path):
        """Baseline: without a mask, column K is nonzero (confirms the assertion above
        is meaningful — K actually appears in the shard data)."""
        K = 10
        n_sae = N_SAE_FEATURES
        B = 2

        shard_dir, sel_dir = _make_shard_with_feature_k(tmp_path, n_sae=n_sae, K=K, n_rows=B * 2)

        boundaries = (
            torch.linspace(
                1 / NUM_BUCKETS,
                1.0 - 1 / NUM_BUCKETS,
                NUM_BUCKETS - 1,
                dtype=torch.float64,
            )
            .unsqueeze(0)
            .expand(n_sae, -1)
            .contiguous()
        )
        pipeline_no_mask = build_threshold_pipeline(boundaries)
        # feature_mask stays None (default)
        _write_selection_manifest(sel_dir, shard_dir, pipeline_no_mask.feature_mask)

        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=B,
            feature_pipeline=pipeline_no_mask,
            device="cpu",
            selection_dir=sel_dir,
            n_sae_features=n_sae,
        )

        batch = next(loader)
        fp_targets = batch[8]

        assert fp_targets is not None
        # K was planted with pct=200/255≈0.784, so at least one element must be nonzero
        assert fp_targets[:, :, K].abs().max().item() > 0.0, (
            f"Baseline check failed: column K={K} is zero even without a mask. "
            "Feature K was not planted in the shard — the masked test above would "
            "be vacuously true and not catch a regression."
        )
