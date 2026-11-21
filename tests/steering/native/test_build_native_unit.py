"""build_native_model loader-contract unit tests (offline).

Re-pointed for Phase-2 PR-1: ``build_native_model`` now REQUIRES a ``layer``
arg and performs structural truncation on the loaded HF model
(slice ``.model.layers[:layer+1]``, ``.norm`` → ``Identity``). These tests
still pin the still-valid pre-truncation contract — the FIRST guard
(unsupported model → ``None``), the loader routing (gemma-2 →
``AutoModelForCausalLM``, gemma-3 → ``Gemma3ForCausalLM``) and the pinned
``attn_implementation='sdpa'`` / ``torch_dtype=float32`` kwargs — but the
monkeypatched ``from_pretrained`` now returns a model with a real sliceable
``.model.layers`` ModuleList so the truncation step succeeds and a
``NativeCollector`` is actually returned (asserting the loader path end to
end, not just that it was reached).
"""

from __future__ import annotations

import torch as t

import steering.native_collector as nc
from steering.native_collector import NativeCollector, build_native_model

_LAYER = 1  # 0 <= 1 < _N_LAYERS (3): a real interior truncation.
_N_LAYERS = 3


class _Cfg:
    """Minimal HF-ish config: truncation reads/writes these attributes."""

    num_attention_heads = 4
    num_hidden_layers = _N_LAYERS
    layer_types = ["full_attention"] * _N_LAYERS


class _Block(t.nn.Module):
    def forward(self, x):  # noqa: ARG002
        return (x,)


class _Inner(t.nn.Module):
    """Stand-in inner decoder stack: has ``.layers`` / ``.norm`` / ``.config``
    exactly as the truncation build asserts."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = t.nn.ModuleList([_Block() for _ in range(_N_LAYERS)])
        self.norm = t.nn.Identity()
        self.config = _Cfg()


class _FakeHF:
    """Fake HF causal-LM whose ``.model`` is a sliceable decoder stack.

    Records ``.to`` / ``.eval`` so the existing device/eval contract is still
    observable; exposes ``.lm_head`` so the free-``>L`` branch runs.
    """

    config = _Cfg()

    def __init__(self) -> None:
        self.model = _Inner()
        self.lm_head = t.nn.Identity()

    def to(self, *a, **k):  # noqa: ARG002
        return self

    def eval(self):
        return self

    def parameters(self):
        yield t.zeros(1)


def test_unsupported_model_returns_none():
    # "gpt2" is not a key in EXTRACTION_MODELS, so the FIRST guard
    # (model_name not in EXTRACTION_MODELS) returns None — Backend.auto can
    # then fall through to nnterp/TL. (gpt2 is out of native v1 scope.)
    # The layer arg is still required by the new signature; the guard fires
    # before it is ever used.
    assert build_native_model("gpt2", device="cpu",
                               model_dtype=t.float32, layer=_LAYER) is None


def test_pins_attn_implementation_gemma2(monkeypatch):
    # gemma-2 now uses output_loading_info=True; fake must return (model, info).
    _info_stub: dict[str, list[str]] = {
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [],
    }
    seen: dict = {}

    def fake_from_pretrained(name, **kw):  # noqa: ARG001
        seen.update(kw)
        return (_FakeHF(), _info_stub)

    monkeypatch.setattr(
        nc.AutoModelForCausalLM, "from_pretrained",
        staticmethod(fake_from_pretrained),
    )
    col = build_native_model("gemma-2-2b", device="cpu",
                             model_dtype=t.float32, layer=_LAYER)
    assert isinstance(col, NativeCollector)
    assert seen.get("attn_implementation") == "sdpa"
    assert seen.get("torch_dtype") == t.float32
    # truncation actually ran: collector records the layer + is residual-only.
    assert col.truncated_layer == _LAYER
    assert col.residual_only is True


def test_gemma3_routes_to_gemma3_loader(monkeypatch):
    """gemma-3* must load via load_gemma3_text_decoder, NOT AutoModelForCausalLM."""
    seen: dict = {}

    def fake_loader(hf_id, *, dtype, attn_implementation):  # noqa: ARG001
        seen["loader"] = "gemma3_loader"
        seen["attn_implementation"] = attn_implementation
        seen["torch_dtype"] = dtype
        return _FakeHF()

    def fail_auto(name, **kw):  # noqa: ARG001
        raise AssertionError(
            "gemma-3 must not use AutoModelForCausalLM.from_pretrained"
        )

    monkeypatch.setattr(nc, "load_gemma3_text_decoder", fake_loader)
    monkeypatch.setattr(
        nc.AutoModelForCausalLM, "from_pretrained", staticmethod(fail_auto)
    )
    col = build_native_model("gemma-3-1b-pt", device="cpu",
                             model_dtype=t.float32, layer=_LAYER)
    assert isinstance(col, NativeCollector)
    assert seen.get("loader") == "gemma3_loader"
    assert seen.get("attn_implementation") == "sdpa"
    assert seen.get("torch_dtype") == t.float32
    assert col.truncated_layer == _LAYER
    assert col.residual_only is True
