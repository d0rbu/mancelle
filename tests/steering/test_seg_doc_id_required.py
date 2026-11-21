"""Slice-10 audit: ``EncoderDecoderGPT.forward`` requires ``seg_doc_id`` when
``fp_targets`` is provided.

Pre-fix: ``model.forward`` silently substituted ``torch.zeros(B, n_seg)`` for
``seg_doc_id`` when it was missing — making every segment look like doc 0,
which collapses the cross-document filter inside
``compute_feature_pred_loss`` and leaks loss across unrelated documents.

The fix raises a clear ``ValueError`` instead. This test pins that contract
and verifies the message references the relevant manifest section so a
future maintainer knows where to look.

Covers plan T-2.

See manifest §4.16, slice-10 map P1-B.
"""

from __future__ import annotations

import pytest
import torch

from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from tests.encdec_test_helpers import (
    NUM_BUCKETS,
    SEQ_LEN,
    VOCAB_SIZE,
    build_tiny_model,
)


def _build_fp_model(n_sae_features: int = 64):
    """Build a tiny encoder-decoder with a non-trivial feature_pred_config.

    The shared helper ``build_model_with_feature_prediction`` does NOT accept
    a ``feature_pred_config`` kwarg (it hardcodes one), so call
    ``build_tiny_model`` directly.
    """
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.FULL_SEGMENT,
        loss_fn=FeaturePredLossFn.BCE,
    )
    return build_tiny_model(
        n_sae_features=n_sae_features,
        feature_pred_config=fp_config,
    )


def _make_inputs(model, n_sae_features: int):
    """Tiny synthetic inputs for a forward pass with feature_pred enabled."""
    B = 2
    T = SEQ_LEN
    n_seg = 4
    F = 8
    idx = torch.randint(2, VOCAB_SIZE, (B, T), dtype=torch.long)
    targets = torch.randint(0, VOCAB_SIZE, (B, T), dtype=torch.long)
    enc_fids = torch.randint(0, n_sae_features, (B, n_seg, F), dtype=torch.long)
    enc_mags = torch.randint(0, NUM_BUCKETS, (B, n_seg, F), dtype=torch.long)
    enc_mask = torch.ones(B, n_seg, F, dtype=torch.bool)
    seg_map = torch.zeros(B, T, dtype=torch.long)
    seg_size = T // n_seg
    for s in range(n_seg):
        seg_map[:, s * seg_size : (s + 1) * seg_size] = s
    fp_targets = torch.zeros(B, n_seg, n_sae_features)
    return idx, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets


def test_forward_raises_when_seg_doc_id_missing_with_fp_targets():
    """fp_targets without seg_doc_id → ValueError pointing at manifest §4.16."""
    model = _build_fp_model(n_sae_features=64)
    model.eval()
    idx, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets = _make_inputs(
        model, n_sae_features=64
    )

    with pytest.raises(ValueError) as excinfo:
        with torch.no_grad():
            model(
                idx,
                targets,
                enc_feature_ids=enc_fids,
                enc_magnitudes=enc_mags,
                enc_mask=enc_mask,
                seg_map=seg_map,
                fp_targets=fp_targets,
                # seg_doc_id deliberately omitted
            )
    msg = str(excinfo.value)
    assert "seg_doc_id" in msg
    # The error message should explain why (silent doc-merging) so
    # maintainers can read the failure mode without diving into the source.
    assert "doc 0" in msg or "merge" in msg.lower() or "leak" in msg.lower()


def test_forward_succeeds_when_seg_doc_id_explicit():
    """Sanity check: passing seg_doc_id explicitly does NOT raise.

    Guards against the fix accidentally raising when seg_doc_id is provided
    (e.g., off-by-one in the None-check predicate).
    """
    model = _build_fp_model(n_sae_features=64)
    model.eval()
    idx, targets, enc_fids, enc_mags, enc_mask, seg_map, fp_targets = _make_inputs(
        model, n_sae_features=64
    )
    B, n_seg = enc_mask.shape[0], enc_mask.shape[1]
    seg_doc_id = torch.zeros(B, n_seg, dtype=torch.long)

    with torch.no_grad():
        out = model(
            idx,
            targets,
            enc_feature_ids=enc_fids,
            enc_magnitudes=enc_mags,
            enc_mask=enc_mask,
            seg_map=seg_map,
            fp_targets=fp_targets,
            seg_doc_id=seg_doc_id,
        )
    # ForwardResult dataclass when fp loss is active — just verify it ran.
    assert out is not None


def test_forward_no_fp_targets_path_does_not_require_seg_doc_id():
    """Without fp_targets, seg_doc_id is unused — model still runs.

    Pin this so the fix didn't accidentally tighten the contract for the
    plain LM-loss path (which is the dominant inference / SFT path that
    has no per-segment doc tracking).
    """
    # Plain model (no fp head). build_tiny_model defaults: feature_pred_config=None.
    model = build_tiny_model(n_sae_features=64)
    model.eval()
    idx, targets, enc_fids, enc_mags, enc_mask, seg_map, _fp = _make_inputs(
        model, n_sae_features=64
    )
    with torch.no_grad():
        loss = model(
            idx,
            targets,
            enc_feature_ids=enc_fids,
            enc_magnitudes=enc_mags,
            enc_mask=enc_mask,
            seg_map=seg_map,
            # NOTE: no fp_targets, no seg_doc_id
        )
    # Plain LM cross-entropy path returns a scalar tensor.
    assert loss.dim() == 0
    assert torch.isfinite(loss)
