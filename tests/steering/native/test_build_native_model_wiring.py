"""Wiring tests for build_native_model's gemma-3 load path (issue #549).

CPU-only: monkeypatches load_gemma3_text_decoder — no model weights loaded.
"""

import pytest
import torch as t

import steering.native_collector as nc
from errors import ModelWeightLoadError


def test_build_native_model_uses_loader_for_gemma3(monkeypatch):
    calls = {}

    def fake_loader(hf_id, *, dtype, attn_implementation):
        calls["hf_id"] = hf_id
        calls["attn"] = attn_implementation
        raise RuntimeError("stop after loader call")

    monkeypatch.setattr(nc, "load_gemma3_text_decoder", fake_loader)
    # gemma-3-1b-pt is a registered gemma-3 EXTRACTION_MODELS entry
    result = nc.build_native_model("gemma-3-1b-pt", "cpu", t.float32, 7)
    assert result is None  # generic load failure → None (fallback preserved)
    assert calls["hf_id"] == "google/gemma-3-1b-pt"
    assert calls["attn"] == "sdpa"


def test_build_native_model_reraises_weight_load_error(monkeypatch):
    def fake_loader(hf_id, *, dtype, attn_implementation):
        raise ModelWeightLoadError("random weights!")

    monkeypatch.setattr(nc, "load_gemma3_text_decoder", fake_loader)
    with pytest.raises(ModelWeightLoadError):
        nc.build_native_model("gemma-3-1b-pt", "cpu", t.float32, 7)
