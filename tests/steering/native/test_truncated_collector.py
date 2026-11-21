"""Phase-2 PR-1 — structural-truncation transform correctness gates (§4.1/§4.4).

`NativeCollector.truncated(hf_model, ..., layer)` is the PURE transform: it
slices the inner Gemma decoder stack to `[:layer+1]`, replaces `.norm` with
`nn.Identity()`, and ALWAYS frees layers `>layer` / `lm_head` / orig-norm.
Correctness invariant (spec §4 / Risk #1): the truncated forward
`last_hidden_state` MUST be `torch.equal` to the Phase-1 forward-hook
residual capture (`hook_reference_residual`) for the SAME seeded weights —
`resid_post(L)` is the layer-`L` output *before* the final norm, and the
installed `Gemma2Model.forward` / `Gemma3TextModel.forward` apply *only*
`self.norm` between the last executed decoder layer and `last_hidden_state`
(verified by re-reading the installed source), so an Identity norm makes them
bit-identical.

The transform MUTATES `.layers`/`.norm`/`config` IN PLACE, so every collector
is built from a FRESH seeded twin (the conftest tiny fixtures are
session-scoped and must not be mutated). The twins replicate the conftest
construction verbatim (same config constants, same seed, same fork_rng).
`NativeCollector.truncated` is the genuine public load/transform boundary —
tests call it directly on a conftest-built tiny HF model (NO `_tiny_model`
seam); `build_native_model` is the I/O-only loader that delegates to it.
"""

from __future__ import annotations

import gc
import weakref

import pytest
import torch as t
from transformers.models.gemma2.modeling_gemma2 import Gemma2DecoderLayer
from transformers.models.gemma3.modeling_gemma3 import Gemma3DecoderLayer

from errors import ExtractionError
from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native._hook_reference import hook_reference_residual
from tests.steering.native.conftest import (
    TINY_LAYERS,
    TINY_MIXED_FIRST_FULL_IDX,
    TINY_MIXED_LAYERS,
    TINY_MIXED_SLIDING_WINDOW,
    _build_tiny_gemma2,
    _build_tiny_gemma3,
    _build_tiny_gemma3_mixed,
)

_LAYER = 2  # 0 <= 2 < TINY_LAYERS (4): a real interior truncation.

# Variable-length right-padded batch (row 1 has 2 pad tokens) AND a single
# doc — exercises the documented maskless contract on a ragged batch.
_IDS_BATCH = t.tensor([[3, 9, 1, 7, 2], [5, 5, 8, 0, 0]], dtype=t.long)
_IDS_SINGLE = t.tensor([[3, 9, 1, 7, 2]], dtype=t.long)

# Shared seeded-twin factories live in conftest (single source of truth — the
# re-pointed hook tests reuse the same factories rather than copy-pasting the
# construction). Re-export under the historical local names so the existing
# test bodies/parametrize lists are untouched.
_fresh_tiny_gemma2 = _build_tiny_gemma2
_fresh_tiny_gemma3 = _build_tiny_gemma3
# Mixed (INTERLEAVED full/sliding `layer_types`, distinct per-type rope,
# small sliding_window) tiny gemma-3 — gemma-3's production reality the
# uniform all-`sliding_attention` tiny model never exercised.
_fresh_tiny_gemma3_mixed = _build_tiny_gemma3_mixed

# Structural truncation (slice `.layers`/`config`, Identity norm) is
# dtype-agnostic ⇒ the truncated forward must be `torch.equal` to the
# hook-reference twin AT THE SAME DTYPE even at bf16 (NOT merely allclose). If
# any bf16 case is not exact that is a REAL finding (do not weaken to
# allclose). Both sides are built `.to(dtype)` from byte-identical fp32
# weights, so the only thing under test is the structural transform.
_DTYPES = [
    pytest.param(t.float32, id="fp32"),
    pytest.param(t.bfloat16, id="bf16"),
]


def _twin(fresh_factory, dtype: t.dtype):
    """A fresh seeded twin cast to `dtype` (fp32 weights → `.to(dtype)`)."""
    return fresh_factory().to(dtype)


def _assert_truncated_eq_reference(
    model_name: str, fresh_factory, dtype: t.dtype, layer: int = _LAYER,
) -> None:
    """truncated `_truncated_forward` == `hook_reference_residual`, max_abs==0.

    Two INDEPENDENT fresh seeded twins (same weights, BOTH cast to `dtype`):
    one is truncated and fed through the collector, the other runs the
    verbatim Phase-1 hook. They must be byte-identical (`torch.equal`, NOT
    allclose) for both the ragged batch and the single doc — structural
    truncation is dtype-agnostic, so this holds exactly even at bf16.
    """
    for ids in (_IDS_BATCH, _IDS_SINGLE):
        nc = NativeCollector.truncated(
            _twin(fresh_factory, dtype), model_name=model_name, layer=layer,
        )
        assert nc is not None, f"{model_name}: build returned None"
        got = nc._truncated_forward(ids)
        expected = hook_reference_residual(
            _twin(fresh_factory, dtype), ids, layer,
        )
        assert got.shape == expected.shape, (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: shape "
            f"{got.shape} != {expected.shape}"
        )
        assert got.dtype == expected.dtype == dtype, (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: dtype "
            f"{got.dtype}/{expected.dtype} != {dtype}"
        )
        assert t.equal(got, expected), (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: truncated "
            f"!= hook_reference "
            f"(max_abs={(got - expected).abs().max().item()!r})"
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_truncated_eq_hook_reference_g2(dtype):
    """gemma-2: truncated forward byte-identical to Phase-1 hook (batch+single).

    Parametrized over fp32 AND bf16 — structural truncation is dtype-agnostic
    so `torch.equal` must hold exactly at both.
    """
    _assert_truncated_eq_reference("gemma-2-2b", _fresh_tiny_gemma2, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_truncated_eq_hook_reference_g3(dtype):
    """gemma-3 (uniform): truncated forward byte-identical to Phase-1 hook.

    gemma-3's `Gemma3TextModel.forward` differs (per-layer-type rotary
    embeddings, `layer_types`-indexed mask) but truncation is arch-agnostic;
    this must pass identically to gemma-2, at fp32 AND bf16. (The INTERLEAVED
    full/sliding `layer_types` reality is the dedicated mixed gate below; this
    uniform-all-sliding fixture is retained for parity with gemma-2.)
    """
    _assert_truncated_eq_reference("gemma-3-1b-pt", _fresh_tiny_gemma3, dtype)


# --------------------------------------------------------------------------- #
# REALISM gate: INTERLEAVED-`layer_types` gemma-3 truncation == hook reference. #
#                                                                              #
# The uniform tiny gemma-3 above is all-`sliding_attention` (verified in       #
# conftest), so NO fast test exercised gemma-3's production reality:           #
# INTERLEAVED full/sliding `layer_types` (full ONLY at index 5 for this        #
# fixture) + DISTINCT per-type rope (full base 1e6 vs sliding base 1e4) + a    #
# real small `sliding_window` (4). The installed `Gemma3TextModel.forward`     #
# builds `position_embeddings` per `set(self.config.layer_types)` and indexes  #
# BOTH the per-type rotary and the per-type causal mask by                     #
# `config.layer_types[i]`. `NativeCollector.truncated` slices                  #
# `config.layer_types` to `[:layer+1]` in lockstep with `.layers` /            #
# `num_hidden_layers`; this gate proves that slice is correct across the full  #
# boundary in BOTH directions:                                                 #
#                                                                              #
#   * layer < first_full_idx → kept `set(layer_types[:kept])` = {sliding}      #
#     ONLY (every full layer dropped). A bug that fails to slice               #
#     `config.layer_types` leaves the len-N list → IndexError / wrong per-type #
#     rope+mask (proven: an off-by-one `layer_types` slice raises IndexError). #
#   * layer >= first_full_idx → kept slice MIXES sliding + the full layer      #
#     → both rope tables + both masks exercised; a full-vs-sliding mix-up      #
#     changes the output (proven non-vacuous: full@5 vs sliding@5 differ by    #
#     max_abs≈0.34 at seq>sliding_window).                                     #
#                                                                              #
# Ragged right-padded batch + single, seq length 7 > sliding_window 4 so the   #
# sliding-window mask genuinely differs from the full causal mask. fp32 AND    #
# bf16 (structural truncation is dtype-agnostic ⇒ exact at both).              #
# --------------------------------------------------------------------------- #

# seq length 7 > TINY_MIXED_SLIDING_WINDOW (4) so sliding ≠ full actually
# matters; row 1 right-padded (2 trailing pads) → ragged ⇒ exercises the
# maskless contract on a variable-length batch for the mixed stack too.
assert 7 > TINY_MIXED_SLIDING_WINDOW, "mixed batch seq must exceed sliding_window"
_MIXED_IDS_BATCH = t.tensor(
    [[3, 9, 1, 7, 2, 4, 6], [5, 5, 8, 1, 2, 0, 0]], dtype=t.long
)
_MIXED_IDS_SINGLE = t.tensor([[3, 9, 1, 7, 2, 4, 6]], dtype=t.long)

# Layers crossing the single-full-layer boundary BOTH ways. With
# TINY_MIXED_FIRST_FULL_IDX == 5 and TINY_MIXED_LAYERS == 8:
#   L in {3,4} → kept slice [:L+1] is sliding-ONLY (all full dropped);
#   L in {5,7} → kept slice still contains full@5 (sliding+full mix), with
#                L=5 the exact first-full boundary and L=7 the deepest layer.
assert TINY_MIXED_FIRST_FULL_IDX == 5 and TINY_MIXED_LAYERS == 8, (
    "mixed-layer_types layer parametrization assumes full@5 of 8 layers"
)
_MIXED_LAYERS_DROP_ALL_FULL = [3, 4]  # < first_full_idx → set() == {sliding}
_MIXED_LAYERS_KEEP_FULL = [5, 7]  # >= first_full_idx → set() == {sliding,full}


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "layer", _MIXED_LAYERS_DROP_ALL_FULL + _MIXED_LAYERS_KEEP_FULL
)
def test_truncated_eq_hook_reference_g3_mixed_layer_types(layer, dtype):
    """gemma-3 INTERLEAVED full/sliding `layer_types`: truncated == hook ref.

    The realism gate the pre-merge review found missing. For each parametrized
    truncation layer (some BELOW the single full layer's index → kept
    `set(layer_types)` collapses to sliding-only with every full layer
    dropped; some AT/ABOVE it → kept slice mixes sliding + full) the
    structurally-truncated forward MUST be `torch.equal` (max_abs==0, NOT
    allclose) to the verbatim Phase-1 hook on a byte-identical twin — for a
    ragged right-padded batch AND a single doc, at fp32 AND bf16, with seq
    length (7) > the fixture's `sliding_window` (4) so the sliding-window mask
    genuinely differs from the full causal mask. A `config.layer_types`
    slice-misalignment or a full-vs-sliding rope/mask mix-up changes the
    output and fails here.
    """
    for ids in (_MIXED_IDS_BATCH, _MIXED_IDS_SINGLE):
        nc = NativeCollector.truncated(
            _twin(_fresh_tiny_gemma3_mixed, dtype),
            model_name="gemma-3-1b-pt",
            layer=layer,
        )
        assert nc is not None, f"mixed g3 L={layer}: build returned None"
        # The kept slice's layer-type set must match the boundary side this
        # case is asserting (pins that the parametrization is actually
        # exercising both the drop-all-full and keep-full paths).
        kept_types = set(nc._base.config.layer_types)
        if layer < TINY_MIXED_FIRST_FULL_IDX:
            assert kept_types == {"sliding_attention"}, (
                f"mixed g3 L={layer}: expected sliding-only kept set "
                f"(all full dropped), got {kept_types}"
            )
        else:
            assert kept_types == {"sliding_attention", "full_attention"}, (
                f"mixed g3 L={layer}: expected mixed kept set "
                f"(sliding + full), got {kept_types}"
            )
        got = nc._truncated_forward(ids)
        expected = hook_reference_residual(
            _twin(_fresh_tiny_gemma3_mixed, dtype), ids, layer,
        )
        assert got.shape == expected.shape, (
            f"mixed g3 {dtype} L={layer} ids{tuple(ids.shape)}: shape "
            f"{got.shape} != {expected.shape}"
        )
        assert got.dtype == expected.dtype == dtype, (
            f"mixed g3 {dtype} L={layer} ids{tuple(ids.shape)}: dtype "
            f"{got.dtype}/{expected.dtype} != {dtype}"
        )
        assert t.equal(got, expected), (
            f"mixed g3 {dtype} L={layer} ids{tuple(ids.shape)}: truncated != "
            f"hook_reference "
            f"(max_abs={(got - expected).abs().max().item()!r})"
        )


_DECODER_LAYER_CLS = {
    "gemma-2-2b": Gemma2DecoderLayer,
    "gemma-3-1b-pt": Gemma3DecoderLayer,
}


@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
def test_truncated_structurally_reclaims_gt_L_layers(model_name, fresh_factory):
    """`NativeCollector.truncated` does NOT retain the `>L` decoder layers.

    Replaces the moot `_free`-vs-no-free byte-identity gate (production ALWAYS
    frees; output correctness is covered by truncated == hook_reference). This
    pins the user-meaningful "the collector does not pin the big `>L` weights"
    invariant (the gc-pinning bug being fixed), structurally + deterministically
    (no GPU, no flaky transient-VRAM probe — the transient-peak aspect is the
    slow GPU gate H-R3):

    (a) the truncated `_base.layers` has EXACTLY `layer+1` modules — a refactor
        that forgot to slice would fail here (distinguishes the bug);
    (b) NONE of the original `>L` ``*DecoderLayer`` object ids are reachable
        from the returned collector's module tree
        (``collector._base.modules()`` ∪ ``collector.hf_model.modules()``) —
        a refactor that sliced ``base.layers`` but left the original full
        ModuleList reachable elsewhere (e.g. the OLD code's stale ``layers``
        local pinning it, had it been stored) would fail here;
    (c) a ``weakref`` to a captured ``>L`` layer is DEAD after ``gc.collect()``
        while the collector is still alive — i.e. the in-function reclaim
        actually frees the ``>L`` weights once external strong refs are
        dropped (the OLD code pinned the original ModuleList via a stale local
        across its in-function ``gc.collect()``; this asserts no such pin
        survives into the returned object's reachable graph).
    """
    decoder_cls = _DECODER_LAYER_CLS[model_name]
    hf = fresh_factory()
    n_full = len(hf.model.layers)
    assert n_full == TINY_LAYERS and _LAYER + 1 < n_full, (
        "fixture must have real >L layers for this test to be non-vacuous"
    )

    # Capture the original >L decoder layers + a weakref to one of them
    # BEFORE the transform. Keep ids (cheap ints) for reachability; keep one
    # strong ref only long enough to make the weakref, then drop it for (c).
    gt_l_layers = list(hf.model.layers)[_LAYER + 1:]
    assert gt_l_layers and all(
        isinstance(m, decoder_cls) for m in gt_l_layers
    )
    gt_l_ids = {id(m) for m in gt_l_layers}
    probe_ref = weakref.ref(gt_l_layers[-1])  # the deepest (>L) layer

    nc = NativeCollector.truncated(
        hf, model_name=model_name, layer=_LAYER,
    )
    assert nc is not None

    # (a) sliced to exactly layer+1.
    assert len(nc._base.layers) == _LAYER + 1, (
        f"{model_name}: _base.layers has {len(nc._base.layers)} modules, "
        f"expected {_LAYER + 1} (transform forgot to slice .layers)"
    )

    # (b) no >L DecoderLayer object id reachable from the collector's tree.
    reachable_ids = {
        id(m)
        for m in (*nc._base.modules(), *nc.hf_model.modules())
    }
    leaked = gt_l_ids & reachable_ids
    assert not leaked, (
        f"{model_name}: {len(leaked)} of the original >L decoder layer(s) "
        f"are still reachable from the returned collector's module tree "
        f"(collector pins the big >L weights — gc-pinning regressed)"
    )

    # (c) once our external strong refs are dropped, gc reclaims the >L
    # layer while the collector is alive (it holds no surviving pin to it).
    del gt_l_layers
    gc.collect()
    assert probe_ref() is None, (
        f"{model_name}: a >L decoder layer survived gc.collect() while the "
        f"collector is alive — something still strong-refs it (the gc-pinning "
        f"bug: the original full ModuleList must not be retained)"
    )
    # The collector itself must remain fully alive/usable after the gc.
    assert len(nc._base.layers) == _LAYER + 1
    _ = nc._truncated_forward(_IDS_SINGLE)


@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
@pytest.mark.parametrize("bad_layer", [TINY_LAYERS, TINY_LAYERS + 5, -1, -100])
def test_truncated_validates_layer_oob(model_name, fresh_factory, bad_layer):
    """`layer >= n_layers` or `layer < 0` → typed ExtractionError.

    A subtly-wrong / out-of-range layer must fail loudly (never silently
    truncate at the wrong depth and corrupt every extracted activation).
    """
    with pytest.raises(ExtractionError, match="native: layer .* out of range"):
        NativeCollector.truncated(
            fresh_factory(), model_name=model_name, layer=bad_layer,
        )


# --------------------------------------------------------------------------- #
# Production NativeCollector.capture() — the T4 residual-only truncated path.   #
#                                                                              #
# The AUTHORITATIVE correctness gate (spec §4.1, "CRITICAL spec nuance"):       #
# production capture(residual) returns the UNSTRIPPED [B,S,d] tensor           #
# byte-identical to `hook_reference_residual` (which is itself spec-verified    #
# unstripped — the Phase-1 CALLER, not capture, strips padding). NO padding     #
# strip is added; the gate is `torch.equal(capture(...), hook_reference(...))`. #
# --------------------------------------------------------------------------- #


def _assert_capture_residual_eq_reference(
    model_name: str,
    fresh_factory,
    dtype: t.dtype,
    layer: int = _LAYER,
    ids_pair=(_IDS_BATCH, _IDS_SINGLE),
) -> None:
    """Production `capture(residual)` == `hook_reference_residual`, max_abs==0.

    Two INDEPENDENT fresh seeded twins (same weights, BOTH cast to `dtype`):
    one is truncated and driven through the *production* ``capture()``
    entrypoint (NOT the bare ``_truncated_forward`` helper — this gates the
    public API), the other runs the verbatim Phase-1 hook. UNSTRIPPED
    [B,S,d]; byte-identical (`torch.equal`, NOT allclose — exact even at
    bf16) for the ragged batch and the single doc. Also pins dtype/shape and
    that the returned tensor carries no autograd graph
    (``requires_grad is False``).
    """
    resid_hook = f"blocks.{layer}.hook_resid_post"
    for ids in ids_pair:
        nc = NativeCollector.truncated(
            _twin(fresh_factory, dtype), model_name=model_name, layer=layer,
        )
        assert nc is not None, f"{model_name}: build returned None"
        got = nc.capture(
            ids,
            layer,
            hook_name=resid_hook,
            activation_type=ActivationType.residual,
        )
        expected = hook_reference_residual(
            _twin(fresh_factory, dtype), ids, layer,
        )
        assert got.shape == expected.shape, (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: shape "
            f"{got.shape} != {expected.shape}"
        )
        assert got.dtype == expected.dtype == dtype, (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: dtype "
            f"{got.dtype}/{expected.dtype} != {dtype}"
        )
        assert got.requires_grad is False, (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: capture() "
            f"leaked an autograd graph (requires_grad must be False)"
        )
        # UNSTRIPPED: capture must return the full [B, S, d] padded tensor —
        # S == the padded input length, NOT a per-row true length.
        assert got.shape[0] == ids.shape[0] and got.shape[1] == ids.shape[1], (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: capture() "
            f"returned {tuple(got.shape)[:2]}; expected unstripped "
            f"[B={ids.shape[0]}, S={ids.shape[1]}, d] (no padding strip — "
            f"caller strips)"
        )
        assert t.equal(got, expected), (
            f"{model_name} {dtype} L={layer} ids{tuple(ids.shape)}: "
            f"capture(residual) != hook_reference (max_abs="
            f"{(got - expected).abs().max().item()!r})"
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_capture_residual_eq_hook_reference_g2(dtype):
    """gemma-2: production capture(residual) byte-identical to Phase-1 hook.

    Batch + single doc; UNSTRIPPED [B,S,d]; the authoritative spec §4.1 gate,
    at fp32 AND bf16.
    """
    _assert_capture_residual_eq_reference("gemma-2-2b", _fresh_tiny_gemma2, dtype)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_capture_residual_eq_hook_reference_g3(dtype):
    """gemma-3 (uniform): production capture(residual) == Phase-1 hook.

    gemma-3's forward differs (per-layer-type rotary, ``layer_types`` mask)
    but the truncated capture is arch-agnostic — must match identically at
    fp32 AND bf16. (The INTERLEAVED-`layer_types` production reality is the
    dedicated mixed capture gate below.)
    """
    _assert_capture_residual_eq_reference(
        "gemma-3-1b-pt", _fresh_tiny_gemma3, dtype
    )


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    "layer", _MIXED_LAYERS_DROP_ALL_FULL + _MIXED_LAYERS_KEEP_FULL
)
def test_capture_residual_eq_hook_reference_g3_mixed(layer, dtype):
    """gemma-3 INTERLEAVED `layer_types`: production capture == Phase-1 hook.

    The mixed-stack realism gate at the PRODUCTION ``capture()`` boundary
    (the dedicated _truncated_forward variant is above). Same layers crossing
    the single-full-layer boundary both ways (drop-all-full vs keep-full),
    ragged batch + single, seq length 7 > sliding_window 4, fp32 AND bf16 —
    `torch.equal` (max_abs==0) required.
    """
    _assert_capture_residual_eq_reference(
        "gemma-3-1b-pt",
        _fresh_tiny_gemma3_mixed,
        dtype,
        layer=layer,
        ids_pair=(_MIXED_IDS_BATCH, _MIXED_IDS_SINGLE),
    )


@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
@pytest.mark.parametrize(
    ("activation_type", "bad_hook"),
    [
        (ActivationType.attention, f"blocks.{_LAYER}.attn.hook_z"),
        (ActivationType.mlp, f"blocks.{_LAYER}.hook_mlp_out"),
    ],
)
def test_capture_non_residual_raises_typed(
    model_name, fresh_factory, activation_type, bad_hook
):
    """capture(mlp|attention) → typed ExtractionError ('residual-only').

    The Phase-2 truncated collector is residual-only by design (spec §3.2);
    mlp/attention must fail loudly with a typed ExtractionError so the
    dispatch fallback chain degrades exactly as for any other
    ExtractionError (never silently return a wrong/residual tensor).
    """
    nc = NativeCollector.truncated(
        fresh_factory(), model_name=model_name, layer=_LAYER,
    )
    assert nc is not None
    with pytest.raises(ExtractionError, match="residual-only"):
        nc.capture(
            _IDS_BATCH,
            _LAYER,
            hook_name=bad_hook,
            activation_type=activation_type,
        )


@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
def test_capture_wrong_layer_raises_typed(model_name, fresh_factory):
    """capture(layer != truncated_layer) → typed ExtractionError.

    The model is structurally truncated for ONE fixed layer; requesting a
    different layer must fail loudly (it cannot be served and must never
    silently return resid_post of the truncation layer instead).
    """
    nc = NativeCollector.truncated(
        fresh_factory(), model_name=model_name, layer=_LAYER,
    )
    assert nc is not None
    wrong_layer = 5  # != _LAYER (2); collector truncated at layer 2.
    with pytest.raises(
        ExtractionError, match=f"truncated for layer {_LAYER}"
    ):
        nc.capture(
            _IDS_BATCH,
            wrong_layer,
            hook_name=f"blocks.{wrong_layer}.hook_resid_post",
            activation_type=ActivationType.residual,
        )


# --------------------------------------------------------------------------- #
# G2 — final-layer (n-1) byte-identity for plain gemma-2 / uniform gemma-3.   #
#                                                                              #
# The existing deterministic gates (test_capture_residual_eq_hook_reference_*) #
# fix `layer=2` (an interior truncation). The `test_native_property` test      #
# EXCLUDES `layer == TINY_LAYERS-1`. No gate verifies the final layer n-1 is   #
# bit-identical to `hook_reference_residual` — that's the gap. Add a           #
# `torch.equal` test for `layer = TINY_LAYERS-1` for g2 AND uniform g3, at    #
# fp32 AND bf16 (matching the existing gate's `_DTYPES` parametrization).       #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
def test_capture_residual_eq_hook_reference_final_layer(model_name, fresh_factory, dtype):
    """capture(layer=TINY_LAYERS-1) byte-identical to Phase-1 hook at final layer.

    G2 gap: the existing gates pin an interior layer (L=2). No test verifies
    the final layer (n-1 = 3 for the tiny fixture). This must be `torch.equal`
    (NOT allclose — structural truncation is dtype-agnostic, exact even at
    bf16) at fp32 AND bf16 for g2 AND uniform g3.

    Non-vacuity: if you compare against `layer=TINY_LAYERS-2` reference (the
    penultimate layer), the tensors differ (max_abs >> 0) and the test FAILS.
    """
    final_layer = TINY_LAYERS - 1
    _assert_capture_residual_eq_reference(model_name, fresh_factory, dtype, layer=final_layer)


# --------------------------------------------------------------------------- #
# G3 — layer=0 boundary: single-layer truncation byte-identity.                #
#                                                                              #
# No deterministic `torch.equal`-vs-hook-reference test pins `layer=0`. A     #
# single-layer truncation (`.layers[:1]`, `norm=Identity`) is the minimal case  #
# and a potential edge in the slice logic. Test g2 AND uniform g3 at fp32 AND  #
# bf16 (matching existing gate's dtype pattern).                               #
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
def test_capture_residual_eq_hook_reference_layer_zero(model_name, fresh_factory, dtype):
    """capture(layer=0) byte-identical to Phase-1 hook at layer 0.

    G3 gap: no deterministic gate pins `layer=0` (single-layer truncation:
    `.layers[:1]`, `norm=Identity`). `torch.equal` required at fp32 AND bf16
    for g2 AND uniform g3.

    Non-vacuity: comparing against `layer=1` reference yields a different
    tensor (max_abs >> 0) and the test FAILS.
    """
    _assert_capture_residual_eq_reference(model_name, fresh_factory, dtype, layer=0)


# --------------------------------------------------------------------------- #
# G4 — single-token S=1: capture on B=1, S=1 input is bit-identical to hook   #
# reference, and .view(-1, d) succeeds with shape [1, d].                      #
# --------------------------------------------------------------------------- #

# S=1 input (B=1, S=1) — the minimal seq-length case.
_IDS_SINGLE_TOKEN = t.tensor([[7]], dtype=t.long)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize(
    ("model_name", "fresh_factory"),
    [
        ("gemma-2-2b", _fresh_tiny_gemma2),
        ("gemma-3-1b-pt", _fresh_tiny_gemma3),
    ],
)
def test_capture_residual_single_token_eq_hook_reference(model_name, fresh_factory, dtype):
    """capture on a single-token input (B=1, S=1) is byte-identical to hook ref.

    G4 gap: all existing batch/single tests use sequences of length >= 3. This
    tests S=1 — the minimal seq-len edge: single-layer case exercises that
    `.view(-1, d)` gives shape `[1, d]` (the consumer's first op after capture)
    and that the truncated forward produces a byte-identical result to the
    Phase-1 hook reference even for the edge input.

    Non-vacuity: if you take the output[0, 0] (one token) and compare against
    a WRONG slice (e.g. output[0, 1] if S>1, or a zeroed tensor), it FAILS.
    """
    nc = NativeCollector.truncated(
        _twin(fresh_factory, dtype), model_name=model_name, layer=_LAYER,
    )
    assert nc is not None
    got = nc.capture(
        _IDS_SINGLE_TOKEN,
        _LAYER,
        hook_name=f"blocks.{_LAYER}.hook_resid_post",
        activation_type=ActivationType.residual,
    )
    expected = hook_reference_residual(
        _twin(fresh_factory, dtype), _IDS_SINGLE_TOKEN, _LAYER,
    )
    # shape checks
    B, S = _IDS_SINGLE_TOKEN.shape
    d = got.shape[-1]
    assert got.shape == (B, S, d), (
        f"{model_name} {dtype} S=1: capture shape {got.shape} != ({B}, {S}, {d})"
    )
    assert got.dtype == dtype
    assert got.requires_grad is False
    # byte-identical to hook reference
    assert t.equal(got, expected), (
        f"{model_name} {dtype} S=1: capture != hook_reference "
        f"(max_abs={(got - expected).abs().max().item()!r})"
    )
    # consumer contract: .view(-1, d) must succeed with shape [B*S, d] = [1, d]
    flat = got.view(-1, d)
    assert flat.shape == (B * S, d), (
        f"{model_name} {dtype} S=1: .view(-1, d) gave {flat.shape}, expected ({B*S}, {d})"
    )
