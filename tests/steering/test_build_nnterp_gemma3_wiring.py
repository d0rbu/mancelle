"""Wiring tests for _build_nnterp_gemma3's gemma-3 load path (issue #549).

CPU-only: monkeypatches load_gemma3_text_decoder — no model weights loaded.
"""

import pytest

import steering.extract_features as ef
from errors import ModelWeightLoadError


def test_build_nnterp_gemma3_uses_loader(monkeypatch):
    calls = {}

    def fake_loader(hf_id, *, dtype, attn_implementation):
        calls["hf_id"] = hf_id
        raise ModelWeightLoadError("boom")

    monkeypatch.setattr(ef, "load_gemma3_text_decoder", fake_loader)
    with pytest.raises(ModelWeightLoadError):
        ef._build_nnterp_gemma3(
            model_name="gemma-3-4b-pt", tl_model=None, device="cpu", model_dtype=None,
        )
    assert calls["hf_id"] == "google/gemma-3-4b-pt"


def test_build_nnterp_model_propagates_model_weight_load_error(monkeypatch):
    """ModelWeightLoadError from the loader must NOT be swallowed by the
    build_nnterp_model dispatcher's broad ``except Exception → return None``
    handler (IMPORTANT #2 — issue #549 review finding)."""

    def raising_loader(hf_id, *, dtype, attn_implementation):
        raise ModelWeightLoadError("missing weights")

    monkeypatch.setattr(ef, "load_gemma3_text_decoder", raising_loader)
    with pytest.raises(ModelWeightLoadError):
        ef.build_nnterp_model(
            model_name="gemma-3-4b-pt",
            tl_model=None,
            device="cpu",
            model_dtype=None,
        )


def test_build_nnterp_gemma3_docstring_has_no_false_claim():
    # The original docstring falsely claimed Gemma3ForCausalLM "transparently
    # strips the vision tower" on from_pretrained — it does NOT (issue #549).
    # Check the key phrase regardless of where line-breaks fall.
    doc = (ef._build_nnterp_gemma3.__doc__ or "").replace("\n", " ")
    assert "transparently strips the vision tower" not in doc


# ---------------------------------------------------------------------------
# Coverage-gap test (issue #549 hardening audit)
# ---------------------------------------------------------------------------


def test_build_nnterp_model_gemma2_returns_none_on_runtimeerror(monkeypatch):
    """build_nnterp_model's gemma-2 path catches RuntimeError from
    _build_nnterp_gemma2 and returns None (so Backend.auto can fall through
    to TL).

    Non-vacuity: if the ``except RuntimeError`` were removed and the error
    propagated instead, the call would raise RuntimeError instead of returning
    None, and this test would fail.
    """
    class _FakeTLModel:
        """Minimal tl_model: satisfies the non-None tl_model check."""

    monkeypatch.setattr(
        ef, "_build_nnterp_gemma2",
        lambda tl_model, device, model_dtype: (_ for _ in ()).throw(
            RuntimeError("synthetic nnterp gemma-2 build failure")
        ),
    )
    result = ef.build_nnterp_model(
        model_name="gemma-2-2b",
        tl_model=_FakeTLModel(),  # ty: ignore[invalid-argument-type]
        device="cpu",
        model_dtype=None,
    )
    assert result is None, f"Expected None, got {result!r}"
