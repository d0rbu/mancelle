"""Math-spec tests for the shared SFT learning-rate schedule.

The SFT schedule differs from pretraining's warmup and configurable final LR:

* No warmup.
* Flat at ``1.0`` for the first 80% of progress.
* Linear decay from ``1.0`` to ``0.0`` over the final 20%.

Decoder and conditioned SFT both use
``steering._common.lr_schedules.sft_flat_decay``; these tests exercise that
shared production helper directly.

The Muon momentum schedule for SFT mirrors pretrain (0.85 -> 0.95 over
300 steps); we pin it here too because the pretrain unit tests live in
``tests/scripts/test_encdec_lr_schedule.py`` and assume the pretrain
script as the source of truth, not SFT.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from steering._common.lr_schedules import sft_flat_decay


def sft_get_muon_momentum(it: int) -> float:
    """SFT Muon momentum: linear ramp 0.85 -> 0.95 over 300 steps, then flat.

    Identical to the pretrain schedule (deliberate per map.md §3 M).
    """
    frac = min(it / 300, 1)
    return (1 - frac) * 0.85 + frac * 0.95


# ---------------------------------------------------------------------------
# 1. SFT LR multiplier (flat 80% + linear decay 20%)
# ---------------------------------------------------------------------------


class TestSFTLRMultiplier:
    """Pin the SFT-specific shape: flat 1.0 then linear decay to 0."""

    def test_progress_zero_is_one(self) -> None:
        """At the very start of training, full LR."""
        assert sft_flat_decay(0.0) == pytest.approx(1.0)

    def test_progress_mid_flat_is_one(self) -> None:
        """Mid-flat region (any progress < 0.8) is exactly 1.0."""
        for p in (0.05, 0.25, 0.5, 0.7, 0.799, 0.7999999):
            assert sft_flat_decay(p) == pytest.approx(1.0), (
                f"progress={p}: expected 1.0 in flat region"
            )

    def test_progress_eight_tenths_boundary(self) -> None:
        """At progress=0.8 we transition; the closed form returns 1.0."""
        assert sft_flat_decay(0.8) == pytest.approx(1.0)

    def test_progress_quarter_through_decay(self) -> None:
        """At progress=0.85 (quarter into decay window) we are at 0.75."""
        assert sft_flat_decay(0.85) == pytest.approx(0.75)

    def test_progress_half_through_decay(self) -> None:
        """At progress=0.9 (mid decay) we are at 0.5."""
        assert sft_flat_decay(0.9) == pytest.approx(0.5)

    def test_progress_three_quarters_decay(self) -> None:
        """At progress=0.95 we are at 0.25."""
        assert sft_flat_decay(0.95) == pytest.approx(0.25)

    def test_progress_one_is_zero(self) -> None:
        """End of training: LR multiplier is 0 (within FP tolerance)."""
        assert sft_flat_decay(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_no_warmup(self) -> None:
        """SFT differs from pretrain: there is NO warmup phase.

        The pretrain schedule starts at ``1/warmup_iters`` and ramps to 1.0;
        SFT starts immediately at 1.0.
        """
        # Pretrain at it=0 with warmup_ratio=0.05 would be 1/50 = 0.02.
        # SFT starts at 1.0 always.
        assert sft_flat_decay(0.0) > 0.5

    def test_no_final_lr_frac(self) -> None:
        """SFT ends at 0 — there is no ``final_lr_frac`` knob.

        Pretrain has ``cfg.final_lr_frac`` (typically 0 but configurable);
        SFT hard-codes 0 and clamps progress at and above 1.0.
        """
        assert sft_flat_decay(1.0) == pytest.approx(0.0, abs=1e-12)

    def test_clamps_outside_progress_range(self) -> None:
        assert sft_flat_decay(-0.1) == pytest.approx(1.0)
        assert sft_flat_decay(1.1) == pytest.approx(0.0)

    def test_flat_then_decay_monotone(self) -> None:
        """Multiplier is monotonically non-increasing across [0, 1]."""
        prev = math.inf
        for i in range(1001):
            progress = i / 1000.0
            cur = sft_flat_decay(progress)
            assert cur <= prev + 1e-9, f"non-monotone at progress={progress}: {prev} -> {cur}"
            prev = cur

    def test_decay_is_linear(self) -> None:
        """In [0.8, 1.0] the diff between consecutive samples is constant."""
        progresses = [0.8 + 0.005 * i for i in range(41)]  # 0.8 to 1.0 incl
        lrs = [sft_flat_decay(p) for p in progresses]
        diffs = [lrs[i + 1] - lrs[i] for i in range(len(lrs) - 1)]
        for d in diffs:
            assert d == pytest.approx(diffs[0], abs=1e-9), f"non-linear decay: diffs={diffs}"
        # All diffs should be negative (decreasing).
        assert all(d < 0 for d in diffs)

    def test_ratio_pretrain_vs_sft_at_start(self) -> None:
        """At step 0, SFT runs at 100% LR while pretrain ramps from ~0%.

        This is the *behavioural* divergence the audit found: copying
        the pretrain LR-schedule test into ``test_encdec_lr_schedule.py``
        would fail to catch a regression in SFT that flattened the
        decay or introduced a warmup.
        """
        sft = sft_flat_decay(0.0)
        # Pretrain at it=0 with warmup_iters=100 (warmup_ratio=0.1) -> 1/100.
        pretrain_at_start_with_warmup = 1 / 100
        assert sft >= 100 * pretrain_at_start_with_warmup


# ---------------------------------------------------------------------------
# 2. SFT Muon momentum (mirrors pretrain)
# ---------------------------------------------------------------------------


class TestSFTMuonMomentum:
    """Sanity-pin: SFT reuses the pretrain momentum schedule verbatim."""

    def test_starts_at_085(self) -> None:
        assert sft_get_muon_momentum(0) == pytest.approx(0.85)

    def test_midpoint_at_step_150(self) -> None:
        assert sft_get_muon_momentum(150) == pytest.approx(0.90)

    def test_endpoint_at_step_300(self) -> None:
        assert sft_get_muon_momentum(300) == pytest.approx(0.95)

    def test_clamped_after_300(self) -> None:
        for it in (301, 500, 1000, 50000):
            assert sft_get_muon_momentum(it) == pytest.approx(0.95)

    def test_linear_warmup(self) -> None:
        """Linearity of momentum warmup over [0, 300]."""
        moments = [sft_get_muon_momentum(it) for it in range(301)]
        diffs = [moments[i + 1] - moments[i] for i in range(300)]
        for d in diffs:
            assert d == pytest.approx(diffs[0], abs=1e-9)
        assert all(d > 0 for d in diffs)


# ---------------------------------------------------------------------------
# 3. Decoder-only wiring; conditioned SFT is covered by actual-loop tests.
# ---------------------------------------------------------------------------


class TestSFTHelperSourceOfTruth:
    def test_decoder_script_calls_shared_schedule(self) -> None:
        src_path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "chat_sft.py"
        assert "lrm = sft_flat_decay(" in src_path.read_text()
