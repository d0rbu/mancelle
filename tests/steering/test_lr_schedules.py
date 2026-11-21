"""Math-spec tests for ``steering._common.lr_schedules``.

The three schedules are pure functions deduplicated from the inline
``get_lr_multiplier`` helpers in ``base_train.py`` / ``chat_sft.py`` /
``chat_rl.py`` (slice 11 audit / consolidation manifest section 1.8).

For each schedule we assert:
* boundary values match the spec,
* monotonicity in the obvious direction(s),
* output range is within [0, 1] (or [final_lr_frac, 1] for warmdown),
* extreme inputs (zero warmup, zero warmdown, etc.) degrade gracefully.

A separate ``test_chat_rl_lr_schedule.py`` adds a chat_rl-specific
end-to-end assertion that 1.0 -> 0.0 over the run.
"""

from __future__ import annotations

import math

import pytest

from steering._common.lr_schedules import (
    pretrain_warmup_flat_warmdown,
    rl_linear_rampdown,
    sft_flat_decay,
)


class TestPretrainWarmupFlatWarmdown:
    NUM_ITERATIONS = 1000

    def _lr(self, it: int, *, warmup_ratio=0.05, warmdown_ratio=0.5, final_lr_frac=0.0):
        return pretrain_warmup_flat_warmdown(
            it,
            num_iterations=self.NUM_ITERATIONS,
            warmup_ratio=warmup_ratio,
            warmdown_ratio=warmdown_ratio,
            final_lr_frac=final_lr_frac,
        )

    def test_warmup_starts_above_zero(self) -> None:
        """At ``it == 0`` we should see a positive multiplier (1/warmup_iters)."""
        warmup_iters = round(0.05 * self.NUM_ITERATIONS)
        assert self._lr(0) == pytest.approx(1 / warmup_iters)

    def test_warmup_endpoint_is_one(self) -> None:
        """At the last warmup step the multiplier reaches 1.0."""
        warmup_iters = round(0.05 * self.NUM_ITERATIONS)
        assert self._lr(warmup_iters) == pytest.approx(1.0)

    def test_warmup_is_monotonically_increasing(self) -> None:
        warmup_iters = round(0.05 * self.NUM_ITERATIONS)
        prev = -math.inf
        for it in range(warmup_iters):
            cur = self._lr(it)
            assert cur > prev
            prev = cur

    def test_flat_region_is_one(self) -> None:
        warmup_iters = round(0.05 * self.NUM_ITERATIONS)
        warmdown_start = self.NUM_ITERATIONS - round(0.5 * self.NUM_ITERATIONS)
        # Pick an iteration squarely inside the flat region.
        flat_it = (warmup_iters + warmdown_start) // 2
        assert self._lr(flat_it) == pytest.approx(1.0)

    def test_warmdown_is_monotonically_decreasing(self) -> None:
        warmdown_start = self.NUM_ITERATIONS - round(0.5 * self.NUM_ITERATIONS)
        prev = math.inf
        for it in range(warmdown_start + 1, self.NUM_ITERATIONS + 1):
            cur = self._lr(it, final_lr_frac=0.1)
            assert cur < prev
            prev = cur

    def test_final_step_returns_final_lr_frac(self) -> None:
        assert self._lr(self.NUM_ITERATIONS, final_lr_frac=0.1) == pytest.approx(0.1)
        assert self._lr(self.NUM_ITERATIONS, final_lr_frac=0.0) == pytest.approx(0.0)

    def test_zero_warmup_starts_at_one(self) -> None:
        """``warmup_ratio=0`` means no warmup; first step is in the flat region."""
        assert pretrain_warmup_flat_warmdown(
            0,
            num_iterations=100,
            warmup_ratio=0.0,
            warmdown_ratio=0.5,
            final_lr_frac=0.0,
        ) == pytest.approx(1.0)

    def test_zero_warmdown_returns_one_until_final_step(self) -> None:
        """``warmdown_ratio=0`` collapses the warmdown region; multiplier
        stays at 1 except possibly at the boundary."""
        for it in [0, 50, 99]:
            assert pretrain_warmup_flat_warmdown(
                it,
                num_iterations=100,
                warmup_ratio=0.05,
                warmdown_ratio=0.0,
                final_lr_frac=0.0,
            ) == pytest.approx(1.0 if it >= round(0.05 * 100) else (it + 1) / round(0.05 * 100))


class TestSftFlatDecay:
    def test_below_eighty_pct_returns_one(self) -> None:
        for p in [0.0, 0.1, 0.4, 0.5, 0.79, 0.799999]:
            assert sft_flat_decay(p) == pytest.approx(1.0)

    def test_at_eighty_pct_starts_decay(self) -> None:
        assert sft_flat_decay(0.8) == pytest.approx(1.0)

    def test_at_ninety_pct_is_half(self) -> None:
        assert sft_flat_decay(0.9) == pytest.approx(0.5)

    def test_at_full_progress_returns_zero(self) -> None:
        assert sft_flat_decay(1.0) == pytest.approx(0.0)

    def test_overshoot_is_clamped_to_zero(self) -> None:
        """Out-of-range inputs (>1.0) should not produce negative LR."""
        assert sft_flat_decay(1.5) == pytest.approx(0.0)
        assert sft_flat_decay(2.0) == pytest.approx(0.0)

    def test_undershoot_returns_one(self) -> None:
        assert sft_flat_decay(-0.5) == pytest.approx(1.0)


class TestRlLinearRampdown:
    def test_at_zero_returns_one(self) -> None:
        assert rl_linear_rampdown(0, num_steps=100) == pytest.approx(1.0)

    def test_at_last_step_minus_one_is_close_to_zero(self) -> None:
        # 1.0 - 99/100 = 0.01
        assert rl_linear_rampdown(99, num_steps=100) == pytest.approx(0.01)

    def test_at_num_steps_returns_zero(self) -> None:
        assert rl_linear_rampdown(100, num_steps=100) == pytest.approx(0.0)

    def test_monotonically_decreasing(self) -> None:
        prev = math.inf
        for it in range(101):
            cur = rl_linear_rampdown(it, num_steps=100)
            assert cur < prev
            prev = cur

    def test_zero_num_steps_raises(self) -> None:
        """Invalid ``num_steps == 0`` must raise rather than divide by zero."""
        with pytest.raises(ValueError, match="num_steps > 0"):
            rl_linear_rampdown(0, num_steps=0)

    def test_negative_num_steps_raises(self) -> None:
        with pytest.raises(ValueError):
            rl_linear_rampdown(0, num_steps=-1)
