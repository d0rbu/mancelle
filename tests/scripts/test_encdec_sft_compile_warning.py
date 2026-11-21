"""The force-dynamic + FA4 combination warns loudly at resolve time.

``compile_static=false`` (whole-graph ``dynamic=True``) crashes the first
compiled forward when the flex FLASH (CuteDSL) decoder backend is active on
torch>=2.13 ("NYI: score_mod or mask_mod captures a dynamic scalar").  The
escape hatch stays usable under Triton/SDPA, so the guard is a warning gated
on the effective backend, not an error.
"""

from __future__ import annotations

import logging

import pytest

from nanochat import flash_attention
from scripts.train.conditional_encdec_sft import _warn_if_forced_dynamic_breaks_fa4

_MARKER = "NYI: score_mod or mask_mod captures a dynamic scalar"


class TestForcedDynamicFa4Warning:
    def test_warns_when_flex_flash_active(self, monkeypatch, caplog) -> None:
        monkeypatch.setattr(flash_attention, "_use_flex", lambda: True)
        monkeypatch.setattr(flash_attention, "has_fa4_backend", lambda: True)
        with caplog.at_level(logging.WARNING):
            _warn_if_forced_dynamic_breaks_fa4(True)
        assert any(_MARKER in record.message for record in caplog.records), (
            "force-dynamic under FLASH must warn with the exact upstream error text"
        )

    @pytest.mark.parametrize("compile_dynamic", [None, False])
    def test_silent_for_non_forced_dynamic(
        self, compile_dynamic, monkeypatch, caplog
    ) -> None:
        monkeypatch.setattr(flash_attention, "_use_flex", lambda: True)
        monkeypatch.setattr(flash_attention, "has_fa4_backend", lambda: True)
        with caplog.at_level(logging.WARNING):
            _warn_if_forced_dynamic_breaks_fa4(compile_dynamic)
        assert not [r for r in caplog.records if _MARKER in r.message]

    def test_silent_when_flex_inactive(self, monkeypatch, caplog) -> None:
        """SDPA/FA3 overrides keep the escape hatch quiet and never probe FA4."""

        def _fail() -> bool:
            raise AssertionError("has_fa4_backend must not be probed when flex is off")

        monkeypatch.setattr(flash_attention, "_use_flex", lambda: False)
        monkeypatch.setattr(flash_attention, "has_fa4_backend", _fail)
        with caplog.at_level(logging.WARNING):
            _warn_if_forced_dynamic_breaks_fa4(True)
        assert not [r for r in caplog.records if _MARKER in r.message]

    def test_warning_text_matches_helper_docstring_contract(
        self, monkeypatch, caplog
    ) -> None:
        """The warning must name both the cause and the two remedies."""
        monkeypatch.setattr(flash_attention, "_use_flex", lambda: True)
        monkeypatch.setattr(flash_attention, "has_fa4_backend", lambda: True)
        with caplog.at_level(logging.WARNING):
            _warn_if_forced_dynamic_breaks_fa4(True)
        matching = [r.message for r in caplog.records if _MARKER in r.message]
        assert matching
        assert "compile_static=auto" in matching[0]
        assert "NANOCHAT_ATTN_IMPL" in matching[0]
