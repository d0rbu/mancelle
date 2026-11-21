"""Wiring tests for build_gradient_model's gemma-3 load path (issue #549).

CPU-only: monkeypatches load_gemma3_text_decoder — no model weights loaded.

Note: max_position_embeddings was relocated from
scripts.preprocess.precompute_gradient_attribution to steering.models (Fix 2);
the docstring check now reads from the canonical location.
"""

import pytest
import torch as t

import steering.gradient_attribution as ga
import steering.models as steering_models
from errors import GradientAttributionError, ModelWeightLoadError


def test_gradient_attribution_uses_loader_for_gemma3(monkeypatch):
    calls = {}

    def fake_loader(hf_id, *, dtype, attn_implementation):
        calls["hf_id"] = hf_id
        raise ModelWeightLoadError("boom")

    monkeypatch.setattr(ga, "load_gemma3_text_decoder", fake_loader)
    with pytest.raises(ModelWeightLoadError):
        ga.build_gradient_model("gemma-3-4b-pt", "cpu", t.float32)
    assert calls["hf_id"] == "google/gemma-3-4b-pt"


def test_max_position_embeddings_docstring_has_no_false_claim():
    # The original docstring falsely claimed that Gemma3ForCausalLM.from_pretrained
    # "remaps the weights transparently" — it does NOT (issue #549).
    # max_position_embeddings was relocated to steering.models (Fix 2).
    doc = (steering_models.max_position_embeddings.__doc__ or "").replace("\n", " ")
    assert "transparently" not in doc


# ---------------------------------------------------------------------------
# Coverage-gap tests (issue #549 hardening audit)
# ---------------------------------------------------------------------------


def test_build_gradient_model_rejects_unknown_model():
    """Non-Gemma model raises GradientAttributionError mentioning Gemma-2/Gemma-3.

    Non-vacuity: removing the guard at the top of build_gradient_model makes it
    proceed to load "gpt2" from HF (or fail with a different error), so this
    test fails.
    """
    with pytest.raises(GradientAttributionError, match="Gemma-2/Gemma-3"):
        ga.build_gradient_model("gpt2", "cpu", t.float32)


def test_build_gradient_model_gemma2_uses_auto_model(monkeypatch):
    """For a gemma-2 model, build_gradient_model uses AutoModelForCausalLM (NOT
    load_gemma3_text_decoder).

    Non-vacuity: if the gemma-2/gemma-3 routing were swapped, the fake auto
    loader would NOT be called and ``auto_called`` would be False, failing the
    assertion; meanwhile load_gemma3_text_decoder would be called and raise.
    """
    auto_called = []

    class _FakeModel:
        """Minimal model object: supports .to() and .eval() and .parameters().

        Includes a minimal ``.config`` so ``from_pretrained_checked`` can read
        ``tie_word_embeddings`` from the model after unpacking the load-info tuple.
        """

        class config:  # noqa: N801
            tie_word_embeddings = False

        def to(self, *a, **k):
            return self

        def eval(self):
            return self

        def parameters(self):
            return iter([])

    _info_ok: dict[str, list] = {"missing_keys": [], "unexpected_keys": [], "mismatched_keys": []}

    def fake_auto(hf_id, **kw):  # noqa: ARG001
        auto_called.append(hf_id)
        return _FakeModel(), _info_ok

    def fail_gemma3(hf_id, **kw):  # noqa: ARG001
        raise AssertionError(
            "gemma-2 build_gradient_model must NOT call load_gemma3_text_decoder"
        )

    monkeypatch.setattr(ga.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(fake_auto))
    monkeypatch.setattr(ga, "load_gemma3_text_decoder", fail_gemma3)

    ga.build_gradient_model("gemma-2-2b", "cpu", t.float32)
    assert len(auto_called) == 1, f"AutoModelForCausalLM not called: {auto_called}"


def test_build_gradient_model_gemma2_strict_guard_propagates(monkeypatch):
    # gradient gemma-2 now routes through from_pretrained_checked → a missing-key
    # load must raise ModelWeightLoadError (previously silently random-initialised).
    def fake(hf_id, **kw):
        class _Cfg:
            tie_word_embeddings = True

        class _M:
            config = _Cfg()

        return _M(), {"missing_keys": ["model.embed_tokens.weight"], "unexpected_keys": set(), "mismatched_keys": []}
    monkeypatch.setattr(ga.AutoModelForCausalLM, "from_pretrained", staticmethod(fake))
    with pytest.raises(ModelWeightLoadError):
        ga.build_gradient_model("gemma-2-2b", "cpu", t.float32)
