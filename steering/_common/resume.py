"""Shared ``resume_from_step`` validation for encdec pretrain + SFT.

Slice-12 audit map P3 #20: pretrain (``conditional_encdec_train.py``)
and SFT (``conditional_encdec_sft.py``) both validated
``cfg.resume_from_step`` independently — pretrain via inline assert,
SFT via a script-private function.  This module consolidates the two
into one helper so the contract cannot drift.

Contract:

* ``-1`` means fresh start; we return ``None``.  Inherited as the
  default from ``base_train.yaml``.
* Any positive integer is returned as-is and means "resume from that
  step number".
* ``0`` is rejected (no step-0 checkpoint to resume from; step 0 is
  pre-training-loop state).
* Any other negative integer is rejected — there is no meaningful
  resume behaviour to associate with them.
"""

from __future__ import annotations


def resolve_resume_step(resume_from_step: int) -> int | None:
    """Validate ``cfg.resume_from_step`` and return the resume target.

    Args:
        resume_from_step: Value from the training config.  Should be
            ``-1`` (fresh start) or a positive integer (resume from
            that step).  Other values are rejected loudly.

    Returns:
        ``None`` if ``resume_from_step == -1`` (fresh start), or the
        positive step number to resume from.

    Raises:
        AssertionError: ``resume_from_step`` is ``0`` or negative
            other than ``-1``.
    """
    if resume_from_step == -1:
        return None
    assert resume_from_step > 0, (
        f"resume_from_step must be > 0 to resume (got {resume_from_step}). "
        f"Use -1 for a fresh start."
    )
    return resume_from_step
