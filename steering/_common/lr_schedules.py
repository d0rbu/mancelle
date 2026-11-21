"""Pure-function LR-multiplier schedules used by training scripts.

The schedules below are the canonical implementations consumed
directly by:

* ``scripts/train/base_train.py`` (``pretrain_warmup_flat_warmdown``)
  -- linear warmup -> flat -> linear warmdown to ``final_lr_frac``
* ``scripts/train/chat_sft.py`` (``sft_flat_decay``)
  -- flat-80% -> linear decay to 0
* ``scripts/train/chat_rl.py`` (``rl_linear_rampdown``)
  -- linear ramp-down to 0

Encoder-decoder pretraining (``conditional_encdec_train.py``) still inlines
its own schedule. Conditioned SFT composes ``sft_flat_decay`` with an integer
optimizer-update warmup via ``sft_warmup_flat_decay``.
"""

from __future__ import annotations


def pretrain_warmup_flat_warmdown(
    it: int,
    *,
    num_iterations: int,
    warmup_ratio: float,
    warmdown_ratio: float,
    final_lr_frac: float,
) -> float:
    """Linear warmup -> flat -> linear warmdown to ``final_lr_frac``.

    Used by ``scripts/train/base_train.py``.

    Returns 1.0 in the flat region, ramps from 0 -> 1 over the first
    ``round(warmup_ratio * num_iterations)`` steps, and ramps from
    1 -> ``final_lr_frac`` over the last ``round(warmdown_ratio * num_iterations)``
    steps.

    Boundary semantics (matched to the inline base_train implementation):

    * ``it == 0`` returns ``1 / max(warmup_iters, 1)`` if ``warmup_iters > 0``,
      else 1.0 (flat from the start).
    * ``it == num_iterations`` returns ``final_lr_frac`` exactly.
    * ``it < 0`` is not validated; pass non-negative iteration counts.
    """
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if warmup_iters > 0 and it < warmup_iters:
        return (it + 1) / warmup_iters
    if it <= num_iterations - warmdown_iters:
        return 1.0
    if warmdown_iters <= 0:
        return 1.0
    progress = (num_iterations - it) / warmdown_iters
    return progress * 1.0 + (1 - progress) * final_lr_frac


def flat_then_linear_decay(progress: float, *, decay_start: float) -> float:
    """Flat 1.0 until ``decay_start``, then linear to 0.0 at ``progress == 1``.

    ``decay_start`` is a fraction of the run, so it says *when decay begins*
    rather than scaling a horizon -- which is the question an operator actually
    asks. ``decay_start=0.8`` is the historical shape; ``0.0`` decays linearly
    across the whole run; ``1.0`` never decays within it.

    Values outside [0, 1] are clamped at the edges (1.0 below 0, 0.0 above 1) so
    a slightly-overshooting generator cannot yield a negative multiplier.
    """
    if not 0.0 <= decay_start <= 1.0:
        raise ValueError(f"decay_start must lie in [0, 1], got {decay_start}")
    if progress < decay_start:
        return 1.0
    if progress >= 1.0:
        return 0.0
    return 1 - (progress - decay_start) / (1 - decay_start)


def sft_flat_decay(progress: float) -> float:
    """Flat-80% then linear decay to 0.

    Used by decoder and conditioned SFT. ``progress`` is derived from
    successfully completed optimizer steps or committed dataset consumption.

    * ``progress in [0, 0.8)``: returns 1.0
    * ``progress in [0.8, 1.0]``: returns linearly-decayed multiplier
      hitting 0.0 at ``progress == 1.0``.

    Values outside [0, 1] are clamped at the edges (return 1.0 below
    0, return 0.0 above 1) so a slightly-overshooting SFT generator
    can't yield negative multipliers.

    This is exactly ``flat_then_linear_decay(progress, decay_start=0.8)``; the
    80/20 split that name refers to is the only thing it fixes.
    """
    return flat_then_linear_decay(progress, decay_start=0.8)


def sft_warmup_flat_decay(
    progress: float,
    *,
    completed_steps: int,
    warmup_steps: int,
) -> float:
    """Scale the unchanged SFT decay axis by an optimizer-update warmup.

    The caller validates the nonnegative integer window and rejects overlap
    with decay. Before update ``s``, the ramp is ``(s + 1) / W`` for ``s < W``.
    Zero bypasses the ramp exactly, including floating-point boundary values.
    """
    if warmup_steps == 0 or completed_steps >= warmup_steps:
        return sft_flat_decay(progress)
    return sft_flat_decay(progress) * ((completed_steps + 1) / warmup_steps)


def rl_linear_rampdown(it: int, *, num_steps: int) -> float:
    """Linear ramp-down from 1.0 -> 0.0 over ``num_steps`` steps.

    Used by ``scripts/train/chat_rl.py``. Mirrors the inline ``1.0 -
    it/num_steps`` form. ``num_steps`` is taken from chat_rl's
    ``(len(train_task) // examples_per_step) * num_epochs`` derivation.

    * ``it == 0`` returns 1.0
    * ``it == num_steps - 1`` returns ``1 / num_steps`` (the last
      training step still gets a small but non-zero LR)
    * ``it == num_steps`` returns 0.0 — but RL training stops before
      this (range loop terminates at ``num_steps - 1``).
    * ``num_steps <= 0`` is invalid (would divide by zero); raise
      ValueError to surface upstream config bugs.
    """
    if num_steps <= 0:
        raise ValueError(
            f"rl_linear_rampdown requires num_steps > 0, got {num_steps}. "
            "This usually indicates a chat_rl cfg bug "
            "(examples_per_step too large vs len(train_task), or num_epochs <= 0)."
        )
    return 1.0 - it / num_steps
