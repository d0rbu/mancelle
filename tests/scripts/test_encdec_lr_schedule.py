"""Pure unit tests for the encoder-decoder **pretraining** LR, momentum, and weight decay schedules.

The schedule functions are copied from
``scripts/train/conditional_encdec_train.py:1250-1267`` so that tests run
without any model, CUDA, or data-loading dependencies.

NOTE — Slice 12 audit (2026-05-05) finding P1 #1: this file used to claim
to cover "the encoder-decoder training LR schedule" without distinguishing
pretrain (linear warmup + flat + linear warmdown to ``final_lr_frac``)
from SFT (flat-80% then linear decay to 0).  The two schedules are
*different* shapes and SFT was therefore untested.  The SFT formula
(``scripts/train/conditional_encdec_sft.py:2180-2181``) is now pinned in
``tests/audit_2026_05/slice_12_encdec_training/test_sft_lr_schedule.py``.
"""

import pytest

# ---------------------------------------------------------------------------
# Pretraining schedule functions extracted from
# scripts/train/conditional_encdec_train.py:1250-1267 — these match the
# formulas inside ``train_one_config()``.  See
# tests/audit_2026_05/slice_12_encdec_training/test_sft_lr_schedule.py
# for the SFT (flat-80% + linear decay to 0) variant.
# ---------------------------------------------------------------------------

def get_lr_multiplier(it, num_iterations, warmup_ratio, warmdown_ratio, final_lr_frac):
    """LR multiplier: linear warmup -> constant -> linear warmdown."""
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    if it <= num_iterations - warmdown_iters:
        return 1.0
    progress = (num_iterations - it) / warmdown_iters
    return progress * 1.0 + (1 - progress) * final_lr_frac


def get_muon_momentum(it):
    """Muon momentum: linear warmup from 0.85 to 0.95 over 300 steps."""
    frac = min(it / 300, 1)
    momentum = (1 - frac) * 0.85 + frac * 0.95
    return momentum


def get_weight_decay(it, weight_decay_scaled, num_iterations):
    """Weight decay: linear decay from weight_decay_scaled to 0."""
    return weight_decay_scaled * (1 - it / num_iterations)


# ---------------------------------------------------------------------------
# 1. LR multiplier tests
# ---------------------------------------------------------------------------

class TestLRMultiplier:
    """Tests for the warmup / constant / warmdown learning rate schedule."""

    def test_warmup_starts_near_zero(self):
        """At it=0 with warmup, LR should be 1/warmup_iters (not 0)."""
        # 10% warmup of 1000 iterations -> warmup_iters = 100
        lr = get_lr_multiplier(0, num_iterations=1000, warmup_ratio=0.1,
                               warmdown_ratio=0.0, final_lr_frac=0.0)
        assert lr == pytest.approx(1 / 100)

    def test_warmup_ends_at_one(self):
        """At the last warmup step, LR should be 1.0."""
        # warmup_iters = round(0.1 * 1000) = 100, last warmup step is it=99
        lr = get_lr_multiplier(99, num_iterations=1000, warmup_ratio=0.1,
                               warmdown_ratio=0.0, final_lr_frac=0.0)
        assert lr == pytest.approx(1.0)

    def test_warmup_linear(self):
        """LR increases linearly during warmup."""
        num_iterations = 1000
        warmup_ratio = 0.1
        warmup_iters = round(warmup_ratio * num_iterations)  # 100

        lrs = [get_lr_multiplier(it, num_iterations, warmup_ratio,
                                 warmdown_ratio=0.0, final_lr_frac=0.0)
               for it in range(warmup_iters)]

        # Check that consecutive differences are constant
        diffs = [lrs[i + 1] - lrs[i] for i in range(len(lrs) - 1)]
        for d in diffs:
            assert d == pytest.approx(diffs[0])

    def test_constant_phase(self):
        """Between warmup and warmdown, LR is exactly 1.0."""
        num_iterations = 1000
        warmup_ratio = 0.1
        warmdown_ratio = 0.2
        warmup_iters = round(warmup_ratio * num_iterations)    # 100
        warmdown_iters = round(warmdown_ratio * num_iterations)  # 200

        for it in range(warmup_iters, num_iterations - warmdown_iters + 1):
            lr = get_lr_multiplier(it, num_iterations, warmup_ratio,
                                   warmdown_ratio, final_lr_frac=0.0)
            assert lr == pytest.approx(1.0), f"Expected 1.0 at step {it}, got {lr}"

    def test_warmdown_starts_at_one(self):
        """First warmdown step should be approximately 1.0."""
        num_iterations = 1000
        warmdown_ratio = 0.5
        warmdown_iters = round(warmdown_ratio * num_iterations)  # 500
        first_warmdown_step = num_iterations - warmdown_iters + 1  # 501

        lr = get_lr_multiplier(first_warmdown_step, num_iterations,
                               warmup_ratio=0.0, warmdown_ratio=warmdown_ratio,
                               final_lr_frac=0.0)
        # progress = (1000 - 501) / 500 = 0.998, so lr ~ 0.998
        assert lr == pytest.approx(1.0, abs=0.01)

    def test_warmdown_ends_at_final_lr_frac(self):
        """At the last iteration (num_iterations-1), LR should be close to final_lr_frac."""
        for final_lr_frac in [0.0, 0.05, 0.1, 0.5]:
            lr = get_lr_multiplier(999, num_iterations=1000,
                                   warmup_ratio=0.0, warmdown_ratio=0.5,
                                   final_lr_frac=final_lr_frac)
            # progress = (1000 - 999) / 500 = 0.002
            expected = 0.002 * 1.0 + (1 - 0.002) * final_lr_frac
            assert lr == pytest.approx(expected), (
                f"final_lr_frac={final_lr_frac}: expected {expected}, got {lr}"
            )

    def test_warmdown_linear_to_zero(self):
        """With final_lr_frac=0, LR linearly goes to 0 during warmdown."""
        num_iterations = 1000
        warmdown_ratio = 0.5
        warmdown_iters = round(warmdown_ratio * num_iterations)  # 500
        warmdown_start = num_iterations - warmdown_iters  # 500

        lrs = [get_lr_multiplier(it, num_iterations, warmup_ratio=0.0,
                                 warmdown_ratio=warmdown_ratio, final_lr_frac=0.0)
               for it in range(warmdown_start + 1, num_iterations)]

        # Consecutive differences should be constant (linear)
        diffs = [lrs[i + 1] - lrs[i] for i in range(len(lrs) - 1)]
        for d in diffs:
            assert d == pytest.approx(diffs[0])
        # All diffs should be negative (decreasing)
        assert all(d < 0 for d in diffs)

    def test_warmdown_linear_to_nonzero(self):
        """With final_lr_frac=0.1, LR linearly goes to 0.1 during warmdown."""
        num_iterations = 1000
        warmdown_ratio = 0.5
        warmdown_iters = round(warmdown_ratio * num_iterations)  # 500

        # Check that the last step approaches final_lr_frac
        lr_last = get_lr_multiplier(num_iterations - 1, num_iterations,
                                    warmup_ratio=0.0, warmdown_ratio=warmdown_ratio,
                                    final_lr_frac=0.1)
        # progress = 1/500 = 0.002, expected = 0.002 * 1.0 + 0.998 * 0.1 = 0.1018
        assert lr_last == pytest.approx(0.1, abs=0.01)

        # Check linearity in the warmdown region
        warmdown_start = num_iterations - warmdown_iters  # 500
        lrs = [get_lr_multiplier(it, num_iterations, warmup_ratio=0.0,
                                 warmdown_ratio=warmdown_ratio, final_lr_frac=0.1)
               for it in range(warmdown_start + 1, num_iterations)]
        diffs = [lrs[i + 1] - lrs[i] for i in range(len(lrs) - 1)]
        for d in diffs:
            assert d == pytest.approx(diffs[0])

    def test_no_warmup_default(self):
        """With warmup_ratio=0.0, first step should have LR=1.0 (no warmup)."""
        lr = get_lr_multiplier(0, num_iterations=1000, warmup_ratio=0.0,
                               warmdown_ratio=0.5, final_lr_frac=0.0)
        assert lr == pytest.approx(1.0)

    def test_no_warmdown(self):
        """With warmdown_ratio=0.0, LR stays at 1.0 for all steps."""
        num_iterations = 100
        for it in range(num_iterations):
            lr = get_lr_multiplier(it, num_iterations, warmup_ratio=0.0,
                                   warmdown_ratio=0.0, final_lr_frac=0.0)
            assert lr == pytest.approx(1.0), f"Step {it}: expected 1.0, got {lr}"

    def test_full_warmdown(self):
        """With warmdown_ratio=1.0, entire training is warmdown."""
        num_iterations = 100
        for it in range(num_iterations):
            lr = get_lr_multiplier(it, num_iterations, warmup_ratio=0.0,
                                   warmdown_ratio=1.0, final_lr_frac=0.0)
            expected = (num_iterations - it) / num_iterations
            assert lr == pytest.approx(expected), f"Step {it}: expected {expected}, got {lr}"

    def test_default_config(self):
        """With default config (warmup=0, warmdown=0.5, final_lr_frac=0), verify schedule shape.

        First half at 1.0, second half linear decay to 0.
        """
        num_iterations = 1000
        warmup_ratio = 0.0
        warmdown_ratio = 0.5
        final_lr_frac = 0.0
        warmdown_iters = round(warmdown_ratio * num_iterations)  # 500

        # First half: constant at 1.0
        for it in range(num_iterations - warmdown_iters + 1):
            lr = get_lr_multiplier(it, num_iterations, warmup_ratio,
                                   warmdown_ratio, final_lr_frac)
            assert lr == pytest.approx(1.0), f"Step {it}: expected 1.0 in constant phase"

        # Second half: linear decay toward 0
        for it in range(num_iterations - warmdown_iters + 1, num_iterations):
            lr = get_lr_multiplier(it, num_iterations, warmup_ratio,
                                   warmdown_ratio, final_lr_frac)
            expected = (num_iterations - it) / warmdown_iters
            assert lr == pytest.approx(expected), f"Step {it}: expected {expected}, got {lr}"
            assert lr < 1.0


# ---------------------------------------------------------------------------
# 2. Muon momentum schedule tests
# ---------------------------------------------------------------------------

class TestMuonMomentumSchedule:
    """Tests for the Muon momentum warmup schedule."""

    def test_momentum_starts_at_085(self):
        """At it=0, momentum is 0.85."""
        assert get_muon_momentum(0) == pytest.approx(0.85)

    def test_momentum_at_150(self):
        """At it=150, momentum is midpoint (0.90)."""
        assert get_muon_momentum(150) == pytest.approx(0.90)

    def test_momentum_at_300(self):
        """At it=300, momentum is 0.95."""
        assert get_muon_momentum(300) == pytest.approx(0.95)

    def test_momentum_after_300(self):
        """At it>300, momentum stays at 0.95."""
        for it in [301, 500, 1000, 10000]:
            assert get_muon_momentum(it) == pytest.approx(0.95), (
                f"Expected 0.95 at step {it}"
            )

    def test_momentum_linear_warmup(self):
        """Momentum increases linearly from 0.85 to 0.95 over first 300 steps."""
        momenta = [get_muon_momentum(it) for it in range(301)]

        # Check endpoints
        assert momenta[0] == pytest.approx(0.85)
        assert momenta[300] == pytest.approx(0.95)

        # Check linearity: consecutive differences should be constant
        diffs = [momenta[i + 1] - momenta[i] for i in range(300)]
        for d in diffs:
            assert d == pytest.approx(diffs[0])
        assert all(d > 0 for d in diffs)


# ---------------------------------------------------------------------------
# 3. Weight decay schedule tests
# ---------------------------------------------------------------------------

class TestWeightDecaySchedule:
    """Tests for the linearly decaying weight decay schedule."""

    def test_starts_at_weight_decay_scaled(self):
        """At it=0, weight decay equals weight_decay_scaled."""
        for wds in [0.1, 0.5, 1.0, 0.002]:
            wd = get_weight_decay(0, weight_decay_scaled=wds, num_iterations=1000)
            assert wd == pytest.approx(wds)

    def test_ends_at_zero(self):
        """At it=num_iterations, weight decay is 0."""
        for num_iters in [100, 1000, 10000]:
            wd = get_weight_decay(num_iters, weight_decay_scaled=0.5,
                                  num_iterations=num_iters)
            assert wd == pytest.approx(0.0)

    def test_linear_decay(self):
        """Weight decay decreases linearly."""
        num_iterations = 1000
        weight_decay_scaled = 0.5

        wds = [get_weight_decay(it, weight_decay_scaled, num_iterations)
               for it in range(num_iterations + 1)]

        # Check that consecutive differences are constant
        diffs = [wds[i + 1] - wds[i] for i in range(len(wds) - 1)]
        for d in diffs:
            assert d == pytest.approx(diffs[0])
        # All diffs should be negative (decreasing)
        assert all(d < 0 for d in diffs)

    def test_midpoint(self):
        """At it=num_iterations/2, weight decay is half of initial."""
        for num_iters in [100, 1000, 10000]:
            wds = 0.5
            wd = get_weight_decay(num_iters // 2, weight_decay_scaled=wds,
                                  num_iterations=num_iters)
            assert wd == pytest.approx(wds / 2)


# ---------------------------------------------------------------------------
# 4. Schedule integration tests (optimizer param_group interaction)
# ---------------------------------------------------------------------------

class TestScheduleIntegration:
    """Verify how schedules interact with optimizer param groups.

    Uses plain dicts to simulate optimizer.param_groups -- no real optimizer needed.
    """

    @staticmethod
    def _make_param_groups():
        """Create fake param groups mimicking the real training setup."""
        return [
            {
                "kind": "adamw",
                "lr": 0.001,
                "initial_lr": 0.001,
                "weight_decay": 0.0,
                "betas": (0.8, 0.95),
            },
            {
                "kind": "muon",
                "lr": 0.02,
                "initial_lr": 0.02,
                "momentum": 0.85,
                "weight_decay": 0.1,
            },
            {
                "kind": "adamw",
                "lr": 0.3,
                "initial_lr": 0.3,
                "weight_decay": 0.0,
                "betas": (0.8, 0.95),
            },
            {
                "kind": "muon",
                "lr": 0.04,
                "initial_lr": 0.04,
                "momentum": 0.85,
                "weight_decay": 0.1,
            },
        ]

    @staticmethod
    def _apply_schedules(param_groups, step, num_iterations, warmup_ratio,
                         warmdown_ratio, final_lr_frac, weight_decay_scaled):
        """Replicate the training loop schedule application logic."""
        lrm = get_lr_multiplier(step, num_iterations, warmup_ratio,
                                warmdown_ratio, final_lr_frac)
        muon_momentum = get_muon_momentum(step)
        muon_weight_decay = get_weight_decay(step, weight_decay_scaled, num_iterations)
        for group in param_groups:
            group["lr"] = group["initial_lr"] * lrm
            if group["kind"] == "muon":
                group["momentum"] = muon_momentum
                group["weight_decay"] = muon_weight_decay

    def test_all_groups_get_same_lr_multiplier(self):
        """All param groups (both AdamW and Muon) get lr = initial_lr * lrm."""
        groups = self._make_param_groups()
        step = 50
        num_iterations = 1000
        lrm = get_lr_multiplier(step, num_iterations, warmup_ratio=0.1,
                                warmdown_ratio=0.5, final_lr_frac=0.0)

        self._apply_schedules(groups, step, num_iterations,
                              warmup_ratio=0.1, warmdown_ratio=0.5,
                              final_lr_frac=0.0, weight_decay_scaled=0.1)

        for group in groups:
            expected_lr = group["initial_lr"] * lrm
            assert group["lr"] == pytest.approx(expected_lr), (
                f"{group['kind']} group: expected lr={expected_lr}, got {group['lr']}"
            )

    def test_only_muon_groups_get_momentum_update(self):
        """AdamW groups should NOT have their momentum changed."""
        groups = self._make_param_groups()
        self._apply_schedules(groups, step=200, num_iterations=1000,
                              warmup_ratio=0.0, warmdown_ratio=0.5,
                              final_lr_frac=0.0, weight_decay_scaled=0.1)

        expected_momentum = get_muon_momentum(200)
        for group in groups:
            if group["kind"] == "muon":
                assert group["momentum"] == pytest.approx(expected_momentum)
            else:
                # AdamW groups should not have a "momentum" key added
                assert "momentum" not in group

    def test_only_muon_groups_get_weight_decay_update(self):
        """AdamW groups keep weight_decay=0.0."""
        groups = self._make_param_groups()
        self._apply_schedules(groups, step=500, num_iterations=1000,
                              warmup_ratio=0.0, warmdown_ratio=0.5,
                              final_lr_frac=0.0, weight_decay_scaled=0.2)

        expected_wd = get_weight_decay(500, weight_decay_scaled=0.2, num_iterations=1000)
        for group in groups:
            if group["kind"] == "muon":
                assert group["weight_decay"] == pytest.approx(expected_wd)
            else:
                assert group["weight_decay"] == pytest.approx(0.0)

    def test_initial_lr_preserved(self):
        """After LR update, group['initial_lr'] should not change."""
        groups = self._make_param_groups()
        original_initial_lrs = [g["initial_lr"] for g in groups]

        for step in [0, 100, 500, 999]:
            self._apply_schedules(groups, step, num_iterations=1000,
                                  warmup_ratio=0.1, warmdown_ratio=0.5,
                                  final_lr_frac=0.0, weight_decay_scaled=0.1)

        for group, orig_lr in zip(groups, original_initial_lrs, strict=True):
            assert group["initial_lr"] == pytest.approx(orig_lr), (
                f"{group['kind']} group: initial_lr changed from {orig_lr} to {group['initial_lr']}"
            )


# ---------------------------------------------------------------------------
# 5. Source-of-truth pin: confirm fixture matches production formula text.
# Slice-12 audit P1 #1: this file was historically miss-titled and tested
# the pretrain schedule under the name of "encoder-decoder training LR".
# This pin makes drift between the fixture and the pretrain script audible.
# The SFT formula is pinned separately in tests/audit_2026_05/.
# ---------------------------------------------------------------------------


class TestPretrainScheduleSourceOfTruth:
    """If the pretrain script's LR formula changes, fail loudly here."""

    def test_pretrain_lr_source_contains_warmup_and_warmdown_branches(self) -> None:
        from pathlib import Path

        src_path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "conditional_encdec_train.py"
        src = src_path.read_text()
        # Three production identifiers; if any drift this test fires.
        assert "warmup_iters = round(cfg.warmup_ratio * num_iterations)" in src
        assert "warmdown_iters = round(cfg.warmdown_ratio * num_iterations)" in src
        assert "progress = (num_iterations - it) / warmdown_iters" in src
        # Belt-and-suspenders: ensure the pretrain script does NOT contain
        # the SFT-style closed-form expression — they have different shapes.
        assert (
            "1.0 if progress < 0.8 else 1.0 - (progress - 0.8) / 0.2"
            not in src
        ), (
            "Pretrain script should NOT contain the SFT flat-80% formula. "
            "If this fires the two schedules have been homogenized — update tests."
        )
