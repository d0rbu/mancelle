"""Pin the unified ``resolve_resume_step`` semantics across pretrain + SFT.

Pretrain (``conditional_encdec_train.py``) and SFT
(``conditional_encdec_sft.py``) both validate ``cfg.resume_from_step``
and historically did so in different ways (SFT via a script-private
function, pretrain via an inline assert).  This file pins the shared
helper at ``steering._common.resume.resolve_resume_step`` so both
scripts cannot drift.

Contract:

* ``-1`` (the inherited default from ``base_train.yaml``) -> ``None``
  (fresh start; no resume).
* Any positive integer -> the same integer (resume from that step).
* ``0`` -> ``AssertionError`` (there is no "step 0" checkpoint to
  resume from; step 0 is pre-training-loop state).
* Any negative integer other than ``-1`` -> ``AssertionError``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from steering._common.resume import resolve_resume_step


class TestResolveResumeStep:
    def test_minus_one_means_fresh_start(self) -> None:
        assert resolve_resume_step(-1) is None

    def test_positive_int_returned_as_is(self) -> None:
        for n in (1, 100, 1_000, 54_600):
            assert resolve_resume_step(n) == n

    def test_zero_is_rejected(self) -> None:
        with pytest.raises(AssertionError, match=r"resume_from_step"):
            resolve_resume_step(0)

    def test_negative_other_than_minus_one_rejected(self) -> None:
        for n in (-2, -5, -100):
            with pytest.raises(AssertionError, match=r"resume_from_step"):
                resolve_resume_step(n)

    def test_string_int_is_rejected(self) -> None:
        """Operator must pass an int — strings reject loudly.

        Hydra parses the YAML default ``-1`` as a Python int so this is
        defensive: a stringified override (``resume_from_step="100"``)
        would slip past ``== -1`` and ``> 0``, and we want a loud TypeError
        (or AssertionError) rather than silent acceptance.
        """
        # Either TypeError (from comparison) or AssertionError is acceptable.
        with pytest.raises((TypeError, AssertionError)):
            resolve_resume_step("100")  # ty: ignore[invalid-argument-type]


class TestSFTSemantics:
    """Pin the resume-step semantics that SFT relies on.

    SFT historically wrapped this rule in a script-private function;
    the wrapper has been removed in favour of calling the canonical
    helper directly, but the contract still needs pinning so the SFT
    resume path keeps the same accept/reject behaviour.
    """

    def test_sft_resume_contract(self) -> None:
        assert resolve_resume_step(-1) is None
        assert resolve_resume_step(100) == 100
        with pytest.raises(AssertionError):
            resolve_resume_step(0)


class TestSourceOfTruth:
    """Pin that pretrain + SFT both consume the shared helper.

    A regression where either script grew its own inline assert would
    let the two contracts drift again.  This test reads the source of
    both scripts and fires loudly if the canonical import disappears.
    """

    def test_pretrain_script_imports_resolve_resume_step(self) -> None:
        src_path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "conditional_encdec_train.py"
        src = src_path.read_text()
        assert "from steering._common.resume import resolve_resume_step" in src, (
            "Pretrain script should consume the shared resolve_resume_step "
            "helper from steering._common.resume."
        )

    def test_sft_script_imports_resolve_resume_step(self) -> None:
        src_path = Path(__file__).resolve().parents[2] / "scripts" / "train" / "conditional_encdec_sft.py"
        src = src_path.read_text()
        assert "from steering._common.resume import resolve_resume_step" in src, (
            "SFT script should consume the shared resolve_resume_step "
            "helper from steering._common.resume."
        )
