"""Exhaustive NON-VACUOUS line+branch coverage for steering.native_collector.

Re-baselined for Phase-2 PR-1 (spec §6, Risk #7). The Phase-1
full-forward + forward-hook mechanism (``resolve_capture_target`` /
``CaptureKind`` / ``_fwd_hook`` / ``_pre_hook`` / ``_reshape_preproj_z``) was
REMOVED from production in T4; tests that asserted those *removed symbols*
were obsolete (testing nothing) and were deleted. The hook capture *behaviour*
is preserved verbatim in ``tests/steering/native/_hook_reference.py`` and
gated by ``test_hook_reference.py`` + the byte-identical assertions in
``test_truncated_collector.py`` — so no coverage intent is lost, it moved.

PR-1 load/transform split: ``build_native_model`` is now I/O-only (loader
routing + dtype cast + None-fallthrough) and delegates the PURE transform to
the ``NativeCollector.truncated`` classmethod (structure-assert, layer-OOB,
slice ``.layers``/``config``, Identity norm, ALWAYS-free ``>L``+``lm_head``).
The ``_tiny_model``/``_free`` test seams were REMOVED — transform branches are
now driven through the public ``NativeCollector.truncated`` boundary on a
conftest-built tiny model; loader branches stay monkeypatch-driven on
``build_native_model``.

This file drives BOTH to 100% line+branch with H2-style non-vacuous tests
(exact-value asserts / ``pytest.raises(..., match=...)``):

* delegation (``parameters``/``to``/``device``) — targets the truncated
  ``self._base`` (NOT ``hf_model``); asserted explicitly.
* every ``build_native_model`` (I/O) branch: unsupported model, in-EXTRACTION
  but not gemma, gemma-2 vs gemma-3 loader routing + pinned kwargs, loader
  raises → None, fp32 (no extra cast) vs non-fp32 (extra cast), delegation to
  ``NativeCollector.truncated``.
* every ``NativeCollector.truncated`` (transform) branch: unexpected inner
  structure → typed ExtractionError, layer-out-of-range, ``layer_types``
  present (sliced) vs absent (skipped), ALWAYS-free executes, ``lm_head``
  present vs absent, ``empty_cache`` gated on ``cuda.is_available()``.
* every ``capture`` branch: non-residual typed error, wrong-layer typed
  error, non-resid_post hook_name typed error, the truncated-forward
  except-wrap (ExtractionError re-raised as-is vs a plain Exception wrapped).
* ``_truncated_forward`` happy path (exact bytes vs the hook reference) is in
  ``test_truncated_collector.py``; the real-Gemma dispatch path is
  ``test_dispatch_e2e.py`` + the slow Layer-4 tests. This file is fast and
  offline (synthetic ``nn.Module`` stand-ins) except where a real seeded twin
  is needed for the byte-identical delegation/capture checks.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import steering.native_collector as nc
from errors import ExtractionError, ModelWeightLoadError
from steering.models import EXTRACTION_MODELS, ExtractionModelConfig
from steering.native_collector import NativeCollector, build_native_model
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import (
    TINY_LAYERS,
    _build_tiny_gemma2,
)

_LAYER = 2  # 0 <= 2 < TINY_LAYERS (4)
_RESID_HOOK = f"blocks.{_LAYER}.hook_resid_post"
_IDS = t.tensor([[3, 9, 1, 7, 2]], dtype=t.long)


def _built_g2(**kw):
    """A real truncated collector from a fresh seeded gemma-2 twin.

    Calls the genuine public transform boundary ``NativeCollector.truncated``
    directly on a conftest-built tiny HF model (NO ``_tiny_model`` seam —
    that test backdoor was removed in the load/transform split)."""
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER, **kw,
    )
    assert col is not None
    return col


# --------------------------------------------------------------------------- #
# Delegation — parameters() / to() / device now target the TRUNCATED _base.    #
# (spec §3.2: "parameters()/to()/device keep delegating (now to the truncated  #
# base)").                                                                     #
# --------------------------------------------------------------------------- #
def test_parameters_delegates_to_truncated_base():
    """``parameters()`` yields the truncated inner model's params — and ONLY
    layers ``0..L`` (the freed ``lm_head`` / ``>L`` params are NOT yielded,
    proving delegation targets ``_base``, not the outer hf_model)."""
    col = _built_g2()
    params = list(col.parameters())
    base_params = list(col._base.parameters())
    assert len(params) == len(base_params) and len(params) > 0
    assert all(p is q for p, q in zip(params, base_params, strict=True))
    # _base.layers was sliced to [:_LAYER+1]; no parameter belongs to a
    # dropped (>L) decoder layer.
    assert len(col._base.layers) == _LAYER + 1


def test_to_delegates_to_base_and_returns_self():
    """``.to(...)`` forwards to ``self._base`` and returns the collector.

    Asserted by moving to the SAME device and confirming a real param moved
    (synthetic ``_base`` records the call AND the collector identity holds)."""
    col = _built_g2()
    recorded: list[tuple] = []
    real_base = col._base

    class _SpyBase:
        def to(self, *a, **k):
            recorded.append((a, k))
            return real_base

        def parameters(self):
            return real_base.parameters()

    col._base = _SpyBase()  # type: ignore[assignment]
    ret = col.to("cpu", non_blocking=True)
    assert ret is col
    assert recorded == [(("cpu",), {"non_blocking": True})]


def test_device_property_reflects_truncated_base_param_device():
    """``device`` == device of the first *truncated-base* parameter."""
    col = _built_g2()
    assert col.device == next(col._base.parameters()).device
    assert col.device == t.device("cpu")


# --------------------------------------------------------------------------- #
# build_native_model — guard branches.                                         #
# --------------------------------------------------------------------------- #
def test_build_unsupported_model_returns_none(caplog):
    """model_name NOT in EXTRACTION_MODELS -> None (+ warning). FIRST guard,
    fires before ``layer`` is used."""
    with caplog.at_level("WARNING"):
        assert build_native_model("gpt2", "cpu", t.float32, _LAYER) is None


def test_build_in_extraction_models_but_not_gemma_returns_none(monkeypatch):
    """A key present in EXTRACTION_MODELS that is NOT gemma-2*/gemma-3* ->
    the second guard returns None. No real key is non-gemma, so monkeypatch a
    synthetic non-gemma entry; the loader must NOT be hit."""
    monkeypatch.setitem(
        EXTRACTION_MODELS,
        "llama-fake-7b",
        ExtractionModelConfig(hf_model_id="meta/llama-fake-7b"),
    )

    def _must_not_load(*a, **k):
        raise AssertionError("loader must NOT be called for non-gemma model")

    monkeypatch.setattr(nc.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(_must_not_load))
    monkeypatch.setattr(nc, "load_gemma3_text_decoder", _must_not_load)
    assert build_native_model("llama-fake-7b", "cpu", t.float32,
                              _LAYER) is None


def test_build_loader_raises_returns_none(monkeypatch):
    """loader.from_pretrained raising -> None (Backend.auto can then fall
    through to nnterp/TL)."""

    def boom(*a, **k):
        raise OSError("offline / gated repo")

    monkeypatch.setattr(nc.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(boom))
    assert build_native_model("gemma-2-2b", "cpu", t.float32,
                              _LAYER) is None


# --------------------------------------------------------------------------- #
# build_native_model — loader routing + pinned kwargs + dtype cast branch.     #
# A real sliceable .model.layers is required so truncation succeeds and a      #
# NativeCollector is actually returned (asserts the full loader→truncate path).#
# --------------------------------------------------------------------------- #
_N = 3


class _Cfg:
    num_attention_heads = 4
    num_hidden_layers = _N
    layer_types = ["full_attention"] * _N


class _Block(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = t.nn.Parameter(t.ones(1))

    def forward(self, x):  # noqa: ARG002
        return (x,)


class _Inner(t.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = t.nn.ModuleList([_Block() for _ in range(_N)])
        self.norm = t.nn.Identity()
        self.config = _Cfg()


class _FakeLoaded:
    """Fake HF model with a real sliceable ``.model``; records ordered
    ``.to()`` calls and that ``.eval()`` was invoked."""

    def __init__(self, tag: str) -> None:
        self.tag = tag
        self.to_calls: list[tuple] = []
        self.eval_called = False
        self.model = _Inner()
        self.lm_head = t.nn.Identity()
        # Real HF causal-LMs always expose a top-level ``.config`` carrying
        # ``tie_word_embeddings``; the gemma-2 load guard reads it.
        self.config = SimpleNamespace(tie_word_embeddings=False)

    def to(self, *args, **kwargs):
        self.to_calls.append((args, kwargs))
        return self

    def eval(self):
        self.eval_called = True
        return self

    def parameters(self):
        yield t.zeros(1)


def test_build_gemma2_uses_automodel_loader(monkeypatch):
    """gemma-2 -> AutoModelForCausalLM loader (output_loading_info=True), pinned kwargs, truncation runs."""
    seen: dict = {}
    _info_stub: dict[str, list[str]] = {
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [],
    }

    def fake_auto(name, **kw):
        seen["loader"] = "auto"
        seen["hf_id"] = name
        seen.update(kw)
        return (_FakeLoaded("auto"), _info_stub)

    def fail_g3(*a, **k):
        raise AssertionError("gemma-2 must NOT use load_gemma3_text_decoder")

    monkeypatch.setattr(nc.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(fake_auto))
    monkeypatch.setattr(nc, "load_gemma3_text_decoder", fail_g3)
    col = build_native_model("gemma-2-2b", "cpu", t.float32, _LAYER)
    assert isinstance(col, NativeCollector)
    assert col.model_name == "gemma-2-2b"
    assert col.truncated_layer == _LAYER and col.residual_only is True
    assert seen["loader"] == "auto"
    assert seen["hf_id"] == EXTRACTION_MODELS["gemma-2-2b"].hf_model_id
    assert seen["attn_implementation"] == "sdpa"
    assert seen["torch_dtype"] == t.float32


def test_build_gemma3_uses_gemma3_loader(monkeypatch):
    """gemma-3 -> load_gemma3_text_decoder, NOT AutoModelForCausalLM."""
    seen: dict = {}

    def fake_loader(hf_id, *, dtype, attn_implementation):
        seen["loader"] = "gemma3_loader"
        seen["hf_id"] = hf_id
        seen["attn_implementation"] = attn_implementation
        seen["torch_dtype"] = dtype
        return _FakeLoaded("g3")

    def fail_auto(*a, **k):
        raise AssertionError("gemma-3 must NOT use AutoModelForCausalLM")

    monkeypatch.setattr(nc, "load_gemma3_text_decoder", fake_loader)
    monkeypatch.setattr(nc.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(fail_auto))
    col = build_native_model("gemma-3-1b-pt", "cpu", t.float32, _LAYER)
    assert isinstance(col, NativeCollector)
    assert col.model_name == "gemma-3-1b-pt"
    assert col.truncated_layer == _LAYER and col.residual_only is True
    assert seen["loader"] == "gemma3_loader"
    assert seen["hf_id"] == EXTRACTION_MODELS["gemma-3-1b-pt"].hf_model_id
    assert seen["attn_implementation"] == "sdpa"
    assert seen["torch_dtype"] == t.float32


def test_build_fp32_no_dtype_cast(monkeypatch):
    """model_dtype == float32: full HF wrapper is never moved/cast,
    .eval() is invoked, and the truncated collector is placed on device."""
    fake = _FakeLoaded("auto")
    _info_stub: dict[str, list[str]] = {
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [],
    }
    monkeypatch.setattr(
        nc.AutoModelForCausalLM, "from_pretrained",
        staticmethod(lambda name, **kw: (fake, _info_stub)),
    )
    col = build_native_model("gemma-2-2b", "cpu", t.float32, _LAYER)
    assert isinstance(col, NativeCollector)
    assert fake.eval_called is True
    # The full loaded HF wrapper must stay on CPU; only the truncated collector
    # is moved/cast, avoiding a full-model CUDA peak.
    assert fake.to_calls == [], fake.to_calls
    param = next(col.parameters())
    assert param.device.type == "cpu"
    assert param.dtype == t.float32


def test_build_non_fp32_extra_dtype_cast(monkeypatch):
    """model_dtype != float32 casts the truncated collector, not the full HF
    wrapper."""
    fake = _FakeLoaded("auto")
    _info_stub: dict[str, list[str]] = {
        "missing_keys": [], "unexpected_keys": [], "mismatched_keys": [],
    }
    monkeypatch.setattr(
        nc.AutoModelForCausalLM, "from_pretrained",
        staticmethod(lambda name, **kw: (fake, _info_stub)),
    )
    col = build_native_model("gemma-2-2b", "cpu", t.bfloat16, _LAYER)
    assert isinstance(col, NativeCollector)
    assert col.model_name == "gemma-2-2b"
    assert fake.eval_called is True
    assert fake.to_calls == [], fake.to_calls
    param = next(col.parameters())
    assert param.device.type == "cpu"
    assert param.dtype == t.bfloat16


# --------------------------------------------------------------------------- #
# NativeCollector.truncated — pure-transform structural branches.              #
# (PR-1 split: structure-assert / layer-OOB moved out of build_native_model    #
# into the pure classmethod; driven directly via the public boundary, no       #
# loader monkeypatch needed.)                                                  #
# --------------------------------------------------------------------------- #
class _NoModelHF:
    """hf_model whose ``.model`` is None — triggers the structure guard."""

    model = None


class _BadLayersHF:
    """hf_model whose ``.model.layers`` is NOT a ModuleList — structure guard."""

    class _M:
        layers = [1, 2, 3]  # a plain list, not nn.ModuleList
        norm = t.nn.Identity()

        class config:  # noqa: N801
            num_hidden_layers = 3

    model = _M()


@pytest.mark.parametrize(
    "bad_model",
    [_NoModelHF(), _BadLayersHF()],
    ids=["model_is_None", "layers_not_ModuleList"],
)
def test_truncated_unexpected_inner_structure_raises_typed(bad_model):
    """An inner model that is not the expected Gemma decoder stack (no
    ``.model``, or ``.layers`` not a ModuleList) -> typed ExtractionError, so
    a HF-version structural drift fails LOUDLY (never silently truncates the
    wrong thing and corrupts every activation). The structure-assert lives in
    the pure ``NativeCollector.truncated`` transform (no I/O / loader)."""
    with pytest.raises(
        ExtractionError,
        match="native: unexpected inner model structure for gemma-2-2b",
    ):
        NativeCollector.truncated(
            bad_model, model_name="gemma-2-2b", layer=_LAYER,
        )


# G6 — structure-assert clauses 3 & 4: missing `.norm` / missing `.config`.
#
# The existing parametrize above covers clause 1 (`base is None`) and clause 2
# (`.layers` not ModuleList). Clauses 3 (`not hasattr(base, "norm")`) and 4
# (`not hasattr(base, "config")`) are never hit: a model whose `base` HAS a
# valid ModuleList `.layers` but is MISSING `.norm` (clause 3) or MISSING
# `.config` (clause 4) has no existing test. Add two fake-HF models matching
# the `_NoModelHF` / `_BadLayersHF` pattern: both have `base` with a real
# `nn.ModuleList` `.layers` (so clauses 1+2 pass), but one lacks `.norm`
# (clause 3 fires) and the other lacks `.config` (clause 4 fires).

class _NoNormInner(t.nn.Module):
    """Inner model with valid ``.layers`` (ModuleList) + ``.config`` but NO ``.norm``.

    Clause 3 of the structure guard: ``not hasattr(base, "norm")``. A real
    ``nn.Module`` so ``hasattr`` is genuinely False (no stray attribute; nn.Module
    defines only its own attrs, and ``norm`` is NOT one of them unless assigned).
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = t.nn.ModuleList([_Block() for _ in range(_N)])
        # deliberately NO .norm

        class _C:
            num_hidden_layers = _N

        self.config = _C()


class _NoNormHF(t.nn.Module):
    """hf_model whose ``.model`` has a valid ModuleList ``.layers`` + ``.config``
    but NO ``.norm`` — triggers clause 3 of the structure guard."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _NoNormInner()


class _NoConfigInner(t.nn.Module):
    """Inner model with valid ``.layers`` (ModuleList) + ``.norm`` but NO ``.config``.

    Clause 4 of the structure guard: ``not hasattr(base, "config")``. A real
    ``nn.Module`` so ``hasattr`` is genuinely False (``config`` is NOT a default
    ``nn.Module`` attr unless assigned; this never assigns it).
    """

    def __init__(self) -> None:
        super().__init__()
        self.layers = t.nn.ModuleList([_Block() for _ in range(_N)])
        self.norm = t.nn.Identity()
        # deliberately NO .config


class _NoConfigHF(t.nn.Module):
    """hf_model whose ``.model`` has a valid ModuleList ``.layers`` + ``.norm``
    but NO ``.config`` — triggers clause 4 of the structure guard."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _NoConfigInner()


@pytest.mark.parametrize(
    "bad_model",
    [_NoNormHF(), _NoConfigHF()],
    ids=["inner_missing_norm", "inner_missing_config"],
)
def test_truncated_structure_guard_clauses_3_and_4(bad_model):
    """Structure guard clauses 3 (missing `.norm`) and 4 (missing `.config`).

    G6 gap: the existing test only covers clause 1 (`base is None`) and clause 2
    (`.layers` not ModuleList). Clauses 3 and 4 are never hit by any test. Each
    fake-HF model here has a VALID `base` and VALID `ModuleList` `.layers` (so
    clauses 1+2 are satisfied) but is missing `.norm` (clause 3) or `.config`
    (clause 4). The guard must still raise a typed `ExtractionError` with the
    'unexpected inner model structure' message, ensuring structural drift in
    either attr fails loudly.

    Non-vacuity: if you add a valid `.norm` / `.config` to the fake, the guard
    no longer fires and ``NativeCollector.truncated`` succeeds instead of
    raising, so this test FAILS (no error is raised).
    """
    with pytest.raises(
        ExtractionError,
        match="native: unexpected inner model structure for gemma-2-2b",
    ):
        NativeCollector.truncated(
            bad_model, model_name="gemma-2-2b", layer=_LAYER,
        )




@pytest.mark.parametrize("bad_layer", [TINY_LAYERS, TINY_LAYERS + 9, -1, -50])
def test_truncated_layer_out_of_range_raises_typed(bad_layer):
    """``layer >= n_layers`` or ``layer < 0`` -> typed ExtractionError with
    the documented range message (a subtly-wrong layer must never silently
    truncate at the wrong depth)."""
    with pytest.raises(ExtractionError, match="native: layer .* out of range"):
        NativeCollector.truncated(
            _build_tiny_gemma2(), model_name="gemma-2-2b", layer=bad_layer,
        )


class _NoLayerTypesInner(t.nn.Module):
    """Inner stack whose ``config`` has NO ``layer_types`` (gemma-2-like
    minimal config) — exercises the ``layer_types is None`` skip."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = t.nn.ModuleList([_Block() for _ in range(_N)])
        self.norm = t.nn.Identity()

        class _C:
            num_hidden_layers = _N
            # NOTE: deliberately NO ``layer_types`` attribute.

        self.config = _C()


class _NoLayerTypesNoLMHeadHF(t.nn.Module):
    """hf_model whose inner config lacks ``layer_types`` AND which has NO
    ``lm_head`` — covers BOTH the ``layer_types is None`` skip and the
    ``not hasattr(hf_model, 'lm_head')`` skip in the ALWAYS-free block of
    ``NativeCollector.truncated`` in one transform. A real ``nn.Module`` so
    ``hasattr(self, 'lm_head')`` is genuinely False (no stray attr)."""

    def __init__(self) -> None:
        super().__init__()
        self.model = _NoLayerTypesInner()


def test_truncated_no_layer_types_and_no_lm_head_branches():
    """Config without ``layer_types`` → slice skipped, AND hf_model without
    ``lm_head`` → lm_head-free skipped — the pure transform must still succeed
    and produce a residual-only truncated collector. Driven directly through
    the public ``NativeCollector.truncated`` boundary (no loader / I/O)."""
    col = NativeCollector.truncated(
        _NoLayerTypesNoLMHeadHF(), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert isinstance(col, NativeCollector)
    assert col.truncated_layer == _LAYER and col.residual_only is True
    # layers actually sliced to [:_LAYER+1] despite the missing config bits.
    assert len(col._base.layers) == _LAYER + 1
    assert isinstance(col._base.norm, t.nn.Identity)
    assert not hasattr(col.hf_model, "lm_head")  # the skipped branch held


@pytest.mark.parametrize("cuda_avail", [True, False])
def test_truncated_empty_cache_gated_on_cuda_available(monkeypatch,
                                                       cuda_avail):
    """Both edges of ``if t.cuda.is_available():`` in the ALWAYS-free block of
    ``NativeCollector.truncated``: when CUDA is present the transform calls
    ``t.cuda.empty_cache()`` once after ``gc.collect()`` to release the freed
    ``>L``/``lm_head`` VRAM; when absent it must NOT call it (a CPU-only host
    must not touch the CUDA allocator). This env actually HAS a GPU, so the
    ``False`` edge is only reachable by patching ``is_available`` -> False;
    both edges are pinned with an exact call-count assertion."""
    calls: list[int] = []
    monkeypatch.setattr(nc.t.cuda, "is_available", lambda: cuda_avail)
    monkeypatch.setattr(nc.t.cuda, "empty_cache", lambda: calls.append(1))
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert isinstance(col, NativeCollector)
    expected = [1] if cuda_avail else []
    assert calls == expected, (
        f"empty_cache call-count wrong for cuda_avail={cuda_avail}: "
        f"got {calls}, expected {expected}"
    )


def test_truncated_always_frees_and_builds_working_collector():
    """The single ALWAYS-free path (production has NO no-free toggle): the
    transform drops >L / orig-norm / lm_head + gc, slices to ``_LAYER+1``, and
    yields a working residual-only collector whose truncated forward runs.
    (Byte-identity to the hook reference is proven in
    ``test_truncated_collector.py``; the "no >L pin" reclamation invariant is
    ``test_truncated_structurally_reclaims_gt_L_layers``.)"""
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert isinstance(col, NativeCollector)
    assert col.residual_only is True
    assert len(col._base.layers) == _LAYER + 1
    # lm_head was present on a real Gemma2 model → freed to Identity.
    assert isinstance(col.hf_model.lm_head, t.nn.Identity)
    out = col._truncated_forward(_IDS)
    assert out.shape == (1, _IDS.shape[1], col._base.config.hidden_size)
    assert out.requires_grad is False


# --------------------------------------------------------------------------- #
# NativeCollector.capture — every guard + the except-wrap.                      #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    ("activation_type", "bad_hook"),
    [
        (ActivationType.mlp, f"blocks.{_LAYER}.hook_mlp_out"),
        (ActivationType.attention, f"blocks.{_LAYER}.attn.hook_z"),
        (ActivationType.attention, f"blocks.{_LAYER}.hook_attn_out"),
    ],
)
def test_capture_non_residual_raises_typed(activation_type, bad_hook):
    """Non-residual -> typed ExtractionError ('residual-only'); the truncated
    collector is residual-only by design (spec §3.2)."""
    col = _built_g2()
    with pytest.raises(ExtractionError, match="residual-only"):
        col.capture(_IDS, _LAYER, hook_name=bad_hook,
                    activation_type=activation_type)


def test_capture_wrong_layer_raises_typed():
    """``layer != truncated_layer`` -> typed ExtractionError; the model is
    structurally truncated for ONE fixed layer."""
    col = _built_g2()
    wrong = _LAYER + 1
    with pytest.raises(
        ExtractionError, match=f"truncated for layer {_LAYER}, got {wrong}"
    ):
        col.capture(_IDS, wrong, hook_name=f"blocks.{wrong}.hook_resid_post",
                    activation_type=ActivationType.residual)


@pytest.mark.parametrize(
    "bad_hook",
    [
        f"blocks.{_LAYER}.hook_resid_pre",
        f"blocks.{_LAYER}.hook_resid_mid",
        "totally_wrong",
        "",
    ],
)
def test_capture_non_resid_post_hook_name_raises_typed(bad_hook):
    """residual activation_type but a hook_name that is NOT the
    ``hook_resid_post`` site -> typed ExtractionError. Defensive check
    mirroring the removed Phase-1 ``resolve_capture_target`` residual branch
    (the truncated forward serves ONLY resid_post)."""
    col = _built_g2()
    with pytest.raises(
        ExtractionError, match="requires the resid_post site"
    ):
        col.capture(_IDS, _LAYER, hook_name=bad_hook,
                    activation_type=ActivationType.residual)


def test_capture_extraction_error_reraised_as_is():
    """an ExtractionError raised *inside* the truncated forward propagates
    AS-IS — NOT double-wrapped with the 'truncated forward failed' prefix."""
    col = _built_g2()
    inner = ExtractionError("inner-sentinel-boom")

    def _boom(_ids):
        raise inner

    col._truncated_forward = _boom  # type: ignore[method-assign]
    with pytest.raises(ExtractionError) as ei:
        col.capture(_IDS, _LAYER, hook_name=_RESID_HOOK,
                    activation_type=ActivationType.residual)
    msg = str(ei.value)
    assert msg == "inner-sentinel-boom"
    assert "truncated forward failed" not in msg, (
        f"ExtractionError must re-raise as-is, not re-wrap: {msg!r}"
    )
    assert ei.value is inner


def test_capture_plain_exception_wrapped_as_extraction_error():
    """a non-ExtractionError raised in the truncated forward is wrapped as
    ExtractionError('... truncated forward failed for ...') with the original
    chained (so the dispatch fallback degrades cleanly and the root cause is
    preserved)."""
    col = _built_g2()
    original = RuntimeError("kaboom-in-truncated-forward")

    def _boom(_ids):
        raise original

    col._truncated_forward = _boom  # type: ignore[method-assign]
    with pytest.raises(ExtractionError) as ei:
        col.capture(_IDS, _LAYER, hook_name=_RESID_HOOK,
                    activation_type=ActivationType.residual)
    msg = str(ei.value)
    assert "native: truncated forward failed for" in msg
    assert f"{_RESID_HOOK!r}@{_LAYER}" in msg
    assert "RuntimeError: kaboom-in-truncated-forward" in msg
    assert ei.value.__cause__ is original  # original chained via `from e`


# --------------------------------------------------------------------------- #
# Constructor default state (pre-build) — the un-truncated sentinel values.     #
# --------------------------------------------------------------------------- #
class _CtorHF:
    """Minimal hf_model exposing ``.model`` for the constructor."""

    model = "inner-sentinel"


def test_constructor_defaults_before_build():
    """A bare ``NativeCollector(hf)`` (pre-truncation) records the
    un-truncated sentinels (-1 / False) and ``_base = hf_model.model``;
    ``build_native_model`` overwrites these. Pins the documented init
    contract so a not-yet-truncated collector can never be mistaken for a
    valid layer-0 truncated one."""
    hf: Any = _CtorHF()
    col = NativeCollector(hf, model_name="gemma-2-2b")
    assert col.truncated_layer == -1
    assert col.residual_only is False
    assert col._base == "inner-sentinel"
    assert col.model_name == "gemma-2-2b"
    assert col.hf_model is hf


# --------------------------------------------------------------------------- #
# Coverage-gap: gemma-2 weight-guard fires and PROPAGATES (not swallowed).    #
# --------------------------------------------------------------------------- #
def test_build_native_model_gemma2_reraises_weight_load_error(monkeypatch):
    """build_native_model for gemma-2 calls assert_weights_materialized on the
    loading info dict. When the info contains a non-lm_head missing key the
    guard raises ModelWeightLoadError — and that MUST propagate, not be swallowed
    into ``return None`` (the explicit ``except ModelWeightLoadError: raise``
    in the production code).

    Non-vacuity: if the ``except ModelWeightLoadError: raise`` is removed and
    replaced with a fallthrough to ``return None``, this test fails because
    ``build_native_model`` returns None instead of raising.

    We do NOT monkeypatch assert_weights_materialized — the real guard runs on
    the deliberately-bad info dict to prove the full wiring end-to-end.
    """
    _info_bad: dict[str, list[str]] = {
        "missing_keys": ["model.embed_tokens.weight"],
        "unexpected_keys": [],
        "mismatched_keys": [],
    }

    def fake_from_pretrained(name, **kw):  # noqa: ARG001
        return (_FakeLoaded("auto"), _info_bad)

    monkeypatch.setattr(nc.AutoModelForCausalLM, "from_pretrained",
                        staticmethod(fake_from_pretrained))
    with pytest.raises(ModelWeightLoadError):
        build_native_model("gemma-2-2b", "cpu", t.float32, _LAYER)
