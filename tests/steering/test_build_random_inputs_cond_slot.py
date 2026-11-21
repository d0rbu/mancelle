"""Pin the _build_random_inputs direct target-slot contract.

Encoder slot ``target_seg_idx`` must hold the random donor's features after
the swap. All other encoder slots are unchanged from the real-condition inputs
(i.e. only one slot is mutated per call).

These tests construct synthetic ``MultisegInputs`` directly so that we
test the helper in isolation from ``build_multiseg_inputs`` (slice 13's
own contract test covers that side).
"""

from __future__ import annotations

import pytest
import torch

from steering.conditional.feature_index import SegmentFeatures
from steering.eval.inline_steering_eval import _build_random_inputs
from steering.eval.multiseg_generator import MultisegInputs


def _make_real_inputs(
    n_encoder_seg: int,
    F_max: int = 4,
    T: int = 8,
    real_fid: int = 7,
    real_mag: int = 3,
) -> MultisegInputs:
    """Create a MultisegInputs with ``real_fid`` populating every encoder slot."""
    fids = torch.full((1, n_encoder_seg, F_max), 0, dtype=torch.int64)
    mags = torch.full((1, n_encoder_seg, F_max), 0, dtype=torch.int64)
    mask = torch.zeros((1, n_encoder_seg, F_max), dtype=torch.bool)
    fids[0, :, 0] = real_fid
    mags[0, :, 0] = real_mag
    mask[0, :, 0] = True
    seg_map = torch.zeros((1, T), dtype=torch.int64)
    return MultisegInputs(
        enc_feature_ids=fids,
        enc_magnitudes=mags,
        enc_mask=mask,
        seg_map=seg_map,
    )


def _make_donor_seg(
    fids: list[int], mags: list[int]
) -> SegmentFeatures:
    return SegmentFeatures(
        segment_idx=99,
        position=0,
        feature_ids=torch.tensor(fids, dtype=torch.int64),
        feature_mags=torch.tensor(mags, dtype=torch.int64),
        char_start=0,
        char_end=10,
        token_start=0,
        token_end=10,
    )


class TestCondSlot:
    def test_target_seg_idx_2_writes_to_slot_2(self) -> None:
        real = _make_real_inputs(n_encoder_seg=4)
        donor = _make_donor_seg([42, 43, 44], [9, 8, 7])
        out = _build_random_inputs(real, donor, target_seg_idx=2)
        # Slot 2 carries donor data
        assert out.enc_feature_ids[0, 2, 0].item() == 42
        assert out.enc_feature_ids[0, 2, 1].item() == 43
        assert out.enc_feature_ids[0, 2, 2].item() == 44
        assert out.enc_magnitudes[0, 2, 0].item() == 9
        assert bool(out.enc_mask[0, 2, 0].item()) is True
        assert bool(out.enc_mask[0, 2, 2].item()) is True
        # Slots 0/1/3 untouched (still the real fid)
        for s in (0, 1, 3):
            assert out.enc_feature_ids[0, s, 0].item() == 7

    def test_seg_map_preserved(self) -> None:
        """seg_map must be passed through verbatim — the random condition
        only swaps payload, never positions."""
        real = _make_real_inputs(n_encoder_seg=3, T=5)
        real.seg_map[0, :] = torch.tensor([0, 0, 1, 2, 2], dtype=torch.int64)
        donor = _make_donor_seg([1], [1])
        out = _build_random_inputs(real, donor, target_seg_idx=1)
        assert torch.equal(out.seg_map, real.seg_map)


class TestCondSlotBoundsChecking:
    def test_target_seg_idx_too_large_raises(self) -> None:
        """target_seg_idx beyond n_encoder_seg should raise."""
        real = _make_real_inputs(n_encoder_seg=2)
        donor = _make_donor_seg([1], [1])
        with pytest.raises(ValueError, match=r"out of range for encoder"):
            _build_random_inputs(real, donor, target_seg_idx=5)


# ---------------------------------------------------------------------------
# Truncation to F_max
# ---------------------------------------------------------------------------


class TestDonorTruncation:
    def test_donor_truncated_to_F_max(self) -> None:
        """If donor has more features than F_max, only F_max copy across."""
        real = _make_real_inputs(n_encoder_seg=3, F_max=2)
        donor = _make_donor_seg([10, 11, 12, 13], [1, 2, 3, 4])
        out = _build_random_inputs(real, donor, target_seg_idx=1)
        # Only first F_max=2 entries copied
        assert out.enc_feature_ids[0, 1, 0].item() == 10
        assert out.enc_feature_ids[0, 1, 1].item() == 11
        assert bool(out.enc_mask[0, 1, 0].item()) is True
        assert bool(out.enc_mask[0, 1, 1].item()) is True

    def test_empty_donor_zeros_slot(self) -> None:
        """Donor with 0 features results in cleared (all-zero, all-False) slot."""
        real = _make_real_inputs(n_encoder_seg=3)
        donor = _make_donor_seg([], [])
        out = _build_random_inputs(real, donor, target_seg_idx=1)
        assert out.enc_feature_ids[0, 1].sum().item() == 0
        assert out.enc_mask[0, 1].any().item() is False
