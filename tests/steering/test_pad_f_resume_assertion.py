"""Slice-10 audit: ``precomputed_data_loader`` resume-time ``pad_F`` assertion.

Pre-fix: a checkpoint saved with one shard set, when resumed against a
different shard set with a different feature distribution, would silently
desync from the static-compile graph (which was traced for the old pad_F).
The mismatch never surfaces — the model just fails or produces wrong logits.

Fix: ``precomputed_data_loader`` now persists ``pad_F`` in its yielded
``state_dict``, and at resume time compares the saved value to the
recomputed value. Mismatch → ``ValueError``.

Covers plan T-4.

See manifest §4.15, slice-10 map P1-E.
"""

from __future__ import annotations

import json

import pytest
import torch

from steering.conditional.encoder_decoder.dataloader import precomputed_data_loader
from tests.encdec_test_helpers import (
    N_SAE_FEATURES,
    build_threshold_pipeline,
    make_precomputed_dataloader,
)


def _make_pipeline():
    boundaries = (
        torch.linspace(0.25, 0.75, 3, dtype=torch.float64)
        .unsqueeze(0)
        .expand(N_SAE_FEATURES, -1)
        .contiguous()
    )
    return build_threshold_pipeline(boundaries)


def test_resume_with_correct_pad_f_does_not_raise(tmp_path):
    """Saving + resuming with the same shard set + matching pad_F: no raise."""
    # First pass: get the natural pad_F by grabbing a batch and reading state.
    loader = make_precomputed_dataloader(
        tmp_path,
        B=2,
        n_rows=4,
        segs_per_row=2,
        feats_per_seg=10,
        n_sae_features=N_SAE_FEATURES,
        seed=0,
    )
    batch = next(loader)
    saved_state = batch[-1]
    saved_pad_F = saved_state.get("pad_F")
    # The fused path has caller-provided pad_F = None (helper does not pass it),
    # so saved_pad_F may be None — that's fine, the assertion is conditional.

    pipeline = _make_pipeline()
    shard_dir = tmp_path / "packed"
    resume_state = {
        "shard": "shard_00000",
        "shard_batch_offset": 0,
        "epoch": 1,
        "batch_idx": 0,
        "pad_F": saved_pad_F,  # matching value (or None) — should not raise
    }
    loader3 = precomputed_data_loader(
        packed_row_dir=shard_dir,
        B=2,
        device="cpu",
        feature_pipeline=pipeline,
        n_sae_features=N_SAE_FEATURES,
        resume_state_dict=resume_state,
    )
    # Generator — calling next must succeed (or yield batches without raising
    # the pad_F assertion). We just take one batch.
    out = next(loader3)
    assert out is not None


def test_resume_with_mismatched_pad_f_raises(tmp_path):
    """A divergent saved pad_F triggers the resume assertion."""
    # Set up the same shard env as in the fixture.
    make_precomputed_dataloader(
        tmp_path,
        B=2,
        n_rows=4,
        segs_per_row=2,
        feats_per_seg=10,
        n_sae_features=N_SAE_FEATURES,
        seed=0,
    )
    pipeline = _make_pipeline()
    shard_dir = tmp_path / "packed"

    # Caller-provided pad_F = 50 forces fused-path pad_F to 50; saved value is
    # 999, which mismatches → ValueError.
    resume_state = {
        "shard": "shard_00000",
        "shard_batch_offset": 0,
        "epoch": 1,
        "batch_idx": 0,
        "pad_F": 999,  # deliberately wrong
    }

    with pytest.raises(ValueError) as excinfo:
        # The generator is constructed but the assertion is reached eagerly
        # in the front-matter (before yield), so we have to step it.
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
            pad_F=50,
            resume_state_dict=resume_state,
        )
        # Drive the generator to past the front-matter resume block.
        next(loader)
    msg = str(excinfo.value)
    assert "pad_F" in msg
    assert "999" in msg
    assert "50" in msg


def test_resume_with_mismatched_max_seg_size_raises(tmp_path):
    """Bonus regression: max_seg_size mismatch is also caught."""
    make_precomputed_dataloader(
        tmp_path,
        B=2,
        n_rows=4,
        segs_per_row=2,
        feats_per_seg=10,
        n_sae_features=N_SAE_FEATURES,
        seed=0,
    )
    # Inject max_seg_size into the manifest so the recomputed value is non-None.
    # ``read_manifest`` requires a ``checksum`` key — without it, the function
    # returns None and the metadata is silently dropped.
    shard_dir = tmp_path / "packed"
    manifest = {
        "checksum": "test-checksum",
        "metadata": {"max_n_seg": 4, "max_seg_size": 32},
    }
    (shard_dir / "manifest.json").write_text(json.dumps(manifest))

    pipeline = _make_pipeline()

    resume_state = {
        "shard": "shard_00000",
        "shard_batch_offset": 0,
        "epoch": 1,
        "batch_idx": 0,
        "max_seg_size": 999,  # diverges
    }
    with pytest.raises(ValueError) as excinfo:
        loader = precomputed_data_loader(
            packed_row_dir=shard_dir,
            B=2,
            device="cpu",
            feature_pipeline=pipeline,
            n_sae_features=N_SAE_FEATURES,
            pad_F=50,
            resume_state_dict=resume_state,
        )
        next(loader)
    assert "max_seg_size" in str(excinfo.value)


def test_state_dict_persists_pad_f_at_yield(tmp_path):
    """The yielded state_dict must include ``pad_F`` so a future resume has
    something to compare against. Pin the contract.
    """
    loader = make_precomputed_dataloader(
        tmp_path,
        B=2,
        n_rows=4,
        segs_per_row=2,
        feats_per_seg=10,
        n_sae_features=N_SAE_FEATURES,
        seed=0,
    )
    batch = next(loader)
    state = batch[-1]
    assert "pad_F" in state, (
        f"yielded state_dict must include pad_F for resume validation; "
        f"got keys: {sorted(state.keys())}"
    )
    # Value may be None when sel_shard_paths is empty AND no caller pad_F.
    # That's fine — the assertion only fires when both saved and recomputed
    # are non-None.
    assert state["pad_F"] is None or isinstance(state["pad_F"], int)
