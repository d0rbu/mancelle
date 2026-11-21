"""Home-grown SAE activation collector — raw HF + structural truncation.

Additive alternative to transformer_lens / nnterp for the
steering.extract_features forward pass; see the native-backend section of
docs/pipelines/feature-extraction.md.

No production forward hook: the inner Gemma{2,3} decoder stack is sliced to
layers 0..L and `.norm` is replaced with Identity, so the truncated
forward output IS resid_post(L). Forward-only and residual-only by design —
mlp / attention, non-truncated layers, GPT-2, and generation are out of
scope and raise a typed ExtractionError.
"""

from __future__ import annotations

import gc
from typing import Any

import torch as t
import torch._inductor.config as _inductor_config
from loguru import logger
from transformers import AutoModelForCausalLM

from errors import ExtractionError, ModelWeightLoadError
from steering.gemma3_loading import (
    from_pretrained_checked,
    load_gemma3_text_decoder,
)
from steering.models import EXTRACTION_MODELS
from steering.sae_resolver import ActivationType

# The TL-style residual-post hook site (`blocks.{L}.hook_resid_post`) — the
# ONLY site this residual-only collector serves. Kept as the defensive
# substring check from the removed Phase-1 `resolve_capture_target` residual
# branch (spec §3.2: validate the hook_name is the resid_post site even though
# the truncated forward registers no hook).
_RESID_POST_HOOK_SUBSTR = "hook_resid_post"

# Pinned to match the reference backends' attention kernel so SDPA-vs-eager
# numeric drift cannot silently break feature-equivalence. _build_nnterp_gemma3
# pins "sdpa"; gemma-2 nnterp uses the HF default — we pin it explicitly.
# Module-level (referenced by both NativeCollector.truncated's log line and
# build_native_model's loader) — imports/constants at top (HARD rule).
_NATIVE_ATTN_IMPLEMENTATION = "sdpa"


class NativeCollector:
    """Residual-only collector backed by a structurally-truncated HF model.

    Phase 2 (PR-1): the Phase-1 full-forward + forward-hook capture is REMOVED
    from production. `build_native_model` slices the inner Gemma{2,3} decoder
    stack to `[:truncated_layer+1]` and replaces `.norm` with `nn.Identity()`;
    `capture()` runs that truncated model directly (no hook, maskless) and its
    `last_hidden_state` IS `resid_post(truncated_layer)`, byte-identical to the
    Phase-1 forward-hook capture (spec §3.1/§4.1; gated by
    tests/steering/native/test_truncated_collector.py against the verbatim
    Phase-1 hook reference `tests/steering/native/_hook_reference.py`).

    Residual-only by design (spec §3.2): mlp/attention and a layer other than
    `truncated_layer` raise a typed `ExtractionError`, caught by the existing
    dispatch fallback chain identically to any other `ExtractionError` (no
    dispatch behavior change). Returns `[B, S, d]` (UNSTRIPPED — the caller
    `_get_sae_activations_batched` strips padding per true length, exactly as
    in Phase-1), same shape/dtype/device semantics as
    `_get_activations_nnterp` / `_get_activations_tl`.
    """

    def __init__(self, hf_model, *, model_name: str):
        self.hf_model = hf_model
        self.model_name = model_name  # threaded into truncated()'s log line + ExtractionError messages
        # Phase-2 truncation state — populated by NativeCollector.truncated
        # after it slices the inner decoder stack to [:L+1] and replaces .norm
        # with Identity. self._base is the truncated inner Gemma{2,3} model;
        # running it directly yields resid_post(self.truncated_layer) with no
        # hook (spec §3.1). delegation (parameters/to/device) targets
        # self._base so the freed lm_head / sliced-off layers are not pulled
        # onto the device.
        self._base: Any = hf_model.model
        self.truncated_layer: int = -1
        self.residual_only: bool = False
        # Opt-in compiled mechanism (S1). Default INERT: a built collector is
        # byte-identical to the eager PR-1 path until `enable_compiled()` ARMS
        # it AND the runtime gate (a later task) flips `_use_compiled` via
        # `set_compiled_active`. `_compiled_base` is the (lazily-armed)
        # `torch.compile(self._base)`; both stay default so even a collector
        # built by a path other than `truncated` is inert.
        self._use_compiled: bool = False
        self._compiled_base: Any = None
        # S2 row-packing state (A1). Default INERT: `set_rowpack_verified` is
        # called by the S2 runtime gate (a later task) after the packed forward
        # passes feature-equivalence verification; until then the dispatch layer
        # (a later task) will NOT route batches through `capture_packed`.
        self._rowpack_verified: bool = False

    @classmethod
    def truncated(
        cls,
        hf_model,
        *,
        model_name: str,
        layer: int,
    ) -> NativeCollector:
        """Pure structural-truncation transform of an already-loaded HF model.

        The genuine load/transform boundary (spec §3.1): NO I/O — the caller
        (`build_native_model`, tests) supplies an already-constructed
        Gemma{2,3}ForCausalLM. This classmethod performs the entire in-place
        transform that makes the model's plain forward output exactly
        `resid_post(layer)`:

        1. Assert the expected inner Gemma decoder structure
           (`hf_model.model` with a `.layers` ModuleList, `.norm`, `.config`)
           — a HF structural drift raises a typed `ExtractionError` (fails
           LOUDLY, never silently truncates the wrong thing).
        2. Validate `0 <= layer < n_layers` (a subtly-wrong / out-of-range
           layer must never silently truncate at the wrong depth) → typed
           `ExtractionError`.
        3. Slice `base.layers` / `config.num_hidden_layers` / (gemma-3 only)
           `config.layer_types` to `[:layer+1]` so EXACTLY layers `0..layer`
           execute. The installed `Gemma2Model.forward` /
           `Gemma3TextModel.forward` iterate
           `self.layers[: self.config.num_hidden_layers]` and index
           `config.layer_types[i]` (gemma-3 additionally indexes per-layer-type
           rotary embeddings) — all three must agree.
        4. Replace `base.norm` with `nn.Identity()`. The ONLY op applied to
           `hidden_states` between the last executed decoder layer and
           `last_hidden_state` is `self.norm`, so an Identity norm makes the
           output exactly `resid_post(layer)`, byte-identical to the Phase-1
           forward-hook capture (spec §3.1/§4.1).
        5. ALWAYS free the now-unreachable `>layer` decoder layers + the
           original `.norm` + `lm_head` so the truncated model's peak VRAM is
           ~`(layer+1)/n_layers` of the full model (the documented headroom
           intent). Production ALWAYS frees — there is no `free` toggle.

        gc-pinning correctness: the original (full) `base.layers` ModuleList
        is rebound to the sliced one and is NEVER retained by any local that
        survives to the `gc.collect()`; the only transient holding the `>L`
        layer objects (`_dropped`) is `del`-ed BEFORE `gc.collect()` runs, so
        the in-function collect actually reclaims the `>L` weights (the prior
        implementation pinned the original full ModuleList via a stale local
        `layers`, defeating the in-function reclaim — fixed here).
        """
        base = getattr(hf_model, "model", None)
        layers = getattr(base, "layers", None)
        if (
            base is None
            or not isinstance(layers, t.nn.ModuleList)
            or not hasattr(base, "norm")
            or not hasattr(base, "config")
        ):
            raise ExtractionError(
                f"native: unexpected inner model structure for {model_name}"
            )

        n_layers = len(layers)
        if not (0 <= layer < n_layers):
            raise ExtractionError(
                f"native: layer {layer} out of range for {model_name} "
                f"with {n_layers} layers (need 0 <= layer < {n_layers})"
            )

        # Slice all three so EXACTLY layers 0..layer execute. `layers` (the
        # original full ModuleList) MUST NOT survive to gc.collect() — capture
        # it ONLY transiently via `_dropped` (the >L tail) and `del` that
        # before collecting. After `base.layers` is rebound below, the local
        # `layers` is the LAST strong ref to the original full ModuleList; we
        # immediately overwrite it (`layers = None`) so nothing pins it.
        kept = layer + 1
        _dropped = list(layers)[kept:]  # transient: the >L decoder layers
        base.layers = t.nn.ModuleList(list(layers)[:kept])
        layers = None  # drop the last ref to the ORIGINAL full ModuleList
        base.config.num_hidden_layers = kept
        layer_types = getattr(base.config, "layer_types", None)
        if layer_types is not None:
            base.config.layer_types = list(layer_types)[:kept]
        base.norm = t.nn.Identity()

        # ALWAYS free (production has no no-free path). Layers 0..L and
        # lm_head/orig-norm are independent module objects; dropping refs to
        # >L / lm_head / orig-norm cannot perturb 0..L (output proven
        # byte-identical to the hook reference). No local binds the original
        # full ModuleList here, so `gc.collect()` actually reclaims the >L
        # weights in-function (the documented peak-VRAM headroom intent).
        del _dropped  # the only transient holding the >L layers — drop pre-gc
        if hasattr(hf_model, "lm_head"):
            hf_model.lm_head = t.nn.Identity()
        gc.collect()
        if t.cuda.is_available():
            t.cuda.empty_cache()

        collector = cls(hf_model, model_name=model_name)
        collector._base = base
        collector.truncated_layer = layer
        collector.residual_only = True
        logger.info(
            f"native: built truncated NativeCollector for {model_name} "
            f"(layers[:{kept}], norm=Identity, freed >{layer}+lm_head, "
            f"attn={_NATIVE_ATTN_IMPLEMENTATION})"
        )
        return collector

    def parameters(self):
        return self._base.parameters()

    def to(self, *args, **kwargs):
        self._base.to(*args, **kwargs)
        return self

    @property
    def device(self):
        return next(self._base.parameters()).device

    def enable_compiled(self) -> None:
        """ARM the opt-in compiled forward (does NOT activate it).

        Sets the inductor config the gate's empirical evidence (4090,
        FineWeb-Edu shard 00388, gemma-3-1b-pt @7 bf16, jaccard vs fp32-eager
        reference) selected — ``torch._inductor.config.
        emulate_precision_casts = False`` (the DEFAULT inductor value) — and
        lazily builds ``self._compiled_base = torch.compile(self._base,
        backend="inductor", dynamic=True)`` IF it is not already armed.
        Idempotent: a second call is a no-op (it does NOT recompile / replace
        an existing ``_compiled_base``).

        Why ``emulate_precision_casts=False`` (the DEFAULT) is the right
        choice for this gate: jaccard-vs-fp32-eager measurements on
        gemma-3-1b-pt @7 bf16 (the historically hardest cell, RTX 4090):
            emF dyn    = 0.990183   ← shipped default (closer to ideal-math)
            emT dyn    = 0.988893   (prior shipped — REJECTED here)
            bf16-eager = 0.988895   (baseline reference)
        emF is the EMPIRICAL winner against ideal-math: its compiled forward
        diverges from fp32-eager by LESS than bf16-eager itself diverges. The
        ideal-math-relative gate (the production decision predicate) PASSES
        this config; the previous absolute-threshold gate spuriously rejected
        it on a bf16-vs-bf16 noise blip.

        Whether the compiled graph is also *faster* is per-(model, layer,
        dtype) and is decided at runtime by the gate — this is NOT a universal
        speedup. The ideal-math-relative gate now ACTIVATES compiled on every
        currently-tested cell on the 4090; a future cell whose compiled forward
        is materially worse than bf16-eager vs the ideal-math reference (more
        than ``BASELINE_RELATIVE_SLACK``) is rejected, exactly as designed.

        This ONLY arms — it deliberately does NOT touch ``self._use_compiled``.
        The runtime gate (``_native_compile_gate`` in extract_features.py)
        decides ACTIVATION by calling ``set_compiled_active(True)`` once the
        compiled forward has passed its ideal-math-relative probe; until then
        the eager path (byte-identical to PR-1) keeps serving every activation.
        """
        # Process-global by necessity and DELIBERATELY never restored: the
        # armed graph is compiled under this flag and every compiled call
        # depends on it for numeric correctness for the whole run — restoring
        # it would corrupt the compiled path. Blast radius is empty (native is
        # the only torch.compile consumer); the value here is the inductor
        # DEFAULT, so it cannot corrupt any other path even if one existed.
        _inductor_config.emulate_precision_casts = False
        if self._compiled_base is None:
            self._compiled_base = t.compile(
                self._base, backend="inductor", dynamic=True
            )

    def set_compiled_active(self, active: bool) -> None:
        """Flip the runtime switch routing `_truncated_forward` through the
        armed compiled base (`True`) or back to the eager path (`False`).

        Called by the runtime gate (a later task). A no-op safety-wise unless
        `enable_compiled()` has armed `_compiled_base`: `_truncated_forward`
        only takes the compiled branch when BOTH this flag is True AND
        `_compiled_base is not None`.
        """
        self._use_compiled = bool(active)

    def _truncated_forward(self, input_ids: t.Tensor) -> t.Tensor:
        """resid_post(truncated_layer) via the structurally-truncated model.

        Runs the inner Gemma{2,3} model whose `.layers` is sliced to
        `[:truncated_layer+1]` and whose `.norm` is `nn.Identity()`. The
        installed `Gemma2Model.forward` / `Gemma3TextModel.forward` apply
        ONLY `self.norm` to `hidden_states` between the last executed decoder
        layer and `last_hidden_state`, so with an Identity norm this is exactly
        the Phase-1 forward-hook residual capture (byte-identical; spec
        §3.1/§4.1). MASKLESS (`attention_mask=None`) — preserves the documented
        `_get_activations_native` contract so native ≡ nnterp stays
        bit-identical. Returns `[B, S, d]`, detached, no padding stripped (the
        caller strips per true length, as in Phase-1).

        Opt-in compiled path: ONLY when `enable_compiled()` has armed
        `_compiled_base` AND the gate has called `set_compiled_active(True)`,
        the IDENTICAL contract runs through the compiled base. Otherwise the
        EXISTING eager body runs verbatim, byte-identical to PR-1 (the trusted
        anchor — the whole sub-project's safety rests on it being unchanged).
        """
        if self._use_compiled and self._compiled_base is not None:
            with t.no_grad():
                return self._compiled_base(
                    input_ids=input_ids, attention_mask=None
                ).last_hidden_state.detach()
        with t.no_grad():
            out = self._base(input_ids=input_ids, attention_mask=None)
        return out.last_hidden_state.detach()

    def capture(
        self,
        input_ids: t.Tensor,
        layer: int,
        *,
        hook_name: str,
        activation_type: ActivationType,
        attention_mask: t.Tensor | None = None,
    ) -> t.Tensor:
        """resid_post(`truncated_layer`) via the truncated model. Residual-only.

        Spec §3.2. The Phase-1 full-forward + forward-hook path is REMOVED;
        this runs the structurally-truncated inner model directly (no hook).

        - `activation_type != residual` → typed `ExtractionError`
          (residual-only by design; mlp/attention unsupported).
        - `layer != self.truncated_layer` → typed `ExtractionError` (the model
          is structurally truncated for ONE fixed layer; extraction uses a
          fixed `cfg.layer`).
        - `hook_name` is defensively validated to be the `hook_resid_post`
          site (the same substring check as the removed Phase-1
          `resolve_capture_target` residual branch) then otherwise unused —
          the truncated forward registers no hook.

        `attention_mask` is accepted ONLY to preserve the adapter-contract
        signature (`_get_activations_native` never passes it). It is
        DELIBERATELY UNUSED: the truncated forward is MASKLESS by design
        (`_truncated_forward` calls `self._base(..., attention_mask=None)`) —
        a causal decoder-only LM with right-padding is invariant to trailing
        pads at real-token positions, and a maskless forward is required to
        stay byte-identical to the maskless nnterp backend (the Phase-1
        `_get_activations_native` §1-4 maskless contract, preserved verbatim).

        Returns `[B, S, d]`, detached, `requires_grad=False`, UNSTRIPPED —
        byte-identical to `hook_reference_residual` (the §4.1 gate); the caller
        `_get_sae_activations_batched` strips padding per true length, exactly
        as in Phase-1.
        """
        if activation_type != ActivationType.residual:
            raise ExtractionError(
                "native (Phase-2 truncated collector) is residual-only; "
                "mlp/attention not supported — use backend=auto/nnterp"
            )
        if layer != self.truncated_layer:
            raise ExtractionError(
                f"native collector is truncated for layer "
                f"{self.truncated_layer}, got {layer}"
            )
        if _RESID_POST_HOOK_SUBSTR not in hook_name:
            raise ExtractionError(
                f"native: residual capture requires the resid_post site; "
                f"hook_name {hook_name!r} is not a "
                f"{_RESID_POST_HOOK_SUBSTR!r} site"
            )
        try:
            return self._truncated_forward(input_ids)
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"native: truncated forward failed for {hook_name!r}@{layer}: "
                f"{type(e).__name__}: {e}"
            ) from e

    # ---------------------------------------------------------------------- #
    # S2 additions (A1) — capture_packed + supporting helpers                 #
    # ---------------------------------------------------------------------- #

    def _validate_residual_request(
        self,
        layer: int,
        hook_name: str | None,
        activation_type: ActivationType,
    ) -> None:
        """The residual-only guard shared by `capture_packed` (and any future
        callers that must not inline/fork the typed-error messages).

        Mirrors the inlined guard inside `capture` exactly (same messages, same
        order) so the typed `ExtractionError` contract is a single source of
        truth for S2 callers:

        - `activation_type != residual` → ExtractionError
        - `layer != self.truncated_layer` → ExtractionError
        - `hook_name` lacking the `hook_resid_post` substring → ExtractionError
        """
        if activation_type != ActivationType.residual:
            raise ExtractionError(
                "native (Phase-2 truncated collector) is residual-only; "
                "mlp/attention not supported — use backend=auto/nnterp"
            )
        if layer != self.truncated_layer:
            raise ExtractionError(
                f"native collector is truncated for layer "
                f"{self.truncated_layer}, got {layer}"
            )
        if hook_name is None or _RESID_POST_HOOK_SUBSTR not in hook_name:
            raise ExtractionError(
                f"native: residual capture requires the resid_post site; "
                f"hook_name {hook_name!r} is not a "
                f"{_RESID_POST_HOOK_SUBSTR!r} site"
            )

    def set_rowpack_verified(self, verified: bool) -> None:
        """Flip the S2 runtime switch marking the packed forward as
        feature-equivalence-verified.

        Called by the S2 runtime gate (a later task) once `capture_packed` has
        passed its SAE feature-equivalence probe. INERT by itself: `capture_packed`
        always runs the same packed forward; this flag is the gate's record that
        packing was certified safe for this (model, layer, dtype) — the
        caller-side dispatch (a later task) consults it to decide whether to
        route a batch through `capture_packed`. Mirrors `set_compiled_active`'s
        arming/activation split so the additive mechanism stays default-inert on
        every build.
        """
        self._rowpack_verified = bool(verified)

    def capture_packed(
        self,
        token_id_lists: list[list[int]],
        layer: int,
        *,
        hook_name: str | None = None,
        activation_type: ActivationType = ActivationType.residual,
    ) -> list[t.Tensor]:
        """Row-packed resid_post(`truncated_layer`): per-doc `[Lᵢ, d]` slices.

        S2 packing core (A1). Concatenates the ragged docs into ONE `[1, ΣLᵢ]`
        packed `input_ids` (NO padding) with a per-doc-RESET `position_ids`
        (`arange(Lᵢ)` concatenated per doc) and runs the structurally-truncated
        inner Gemma{2,3} model with `attention_mask=None` and a HARD-WIRED
        `use_cache=False`.

        With exactly those inputs HF's `Gemma2Model.forward` /
        `Gemma3TextModel.forward` autodetect packing (`position_ids is not None
        and attention_mask is None and past_key_values is None`), build the
        block-diagonal causal document mask, and apply per-doc RoPE — so each
        doc's residual equals the padded path's per-doc stripped slice.

        `use_cache=False` is MANDATORY and a hard-wired literal (NOT a caller
        knob): the default `use_cache=True` builds a `DynamicCache`, the
        packing-autodetect guard `past_key_values is None` becomes False →
        plain full-causal attention across the WHOLE pack → cross-document
        leakage.  This is invariant I3 (spec §2); it is teeth-guarded by a
        non-vacuous test that forces `use_cache=True` and asserts the output
        then DIFFERS from the per-doc reference.

        Residual-only typed-error contract identical to `capture` (the same
        error messages via `_validate_residual_request`).

        Returns: `list[t.Tensor]` — per-doc `[Lᵢ, d]` slices, each detached,
        `requires_grad=False`.  Empty `token_id_lists` → `[]`.
        """
        self._validate_residual_request(layer, hook_name, activation_type)
        if not token_id_lists:
            return []
        device = next(self._base.parameters()).device
        lengths = [len(ids) for ids in token_id_lists]
        flat = [tok for ids in token_id_lists for tok in ids]
        input_ids = t.tensor(flat, dtype=t.long, device=device).unsqueeze(0)
        position_ids = t.cat(
            [t.arange(L, dtype=t.long, device=device) for L in lengths]
        ).unsqueeze(0)
        # Route through the S1 compiled base when armed+active, so packing
        # COMPOSES with S1's torch.compile (else packing would silently drop
        # S1's 1.1-1.2x compile win): the whole truncated forward — incl. the
        # attention (FlexAttention fuses into the graph; SDPA composes too) —
        # runs as one compiled graph. Falls back to eager `_base` when compile
        # is not active. Mirrors `_truncated_forward`'s compiled routing, so
        # `capture` (baseline) and `capture_packed` (candidate) share the SAME
        # compile state inside the rowpack gate (clean packing-vs-unpacked
        # comparison, no compile/packing confound).
        base = (
            self._compiled_base
            if self._use_compiled and self._compiled_base is not None
            else self._base
        )
        try:
            with t.no_grad():
                out = base(
                    input_ids=input_ids,
                    position_ids=position_ids,
                    attention_mask=None,
                    use_cache=False,  # MANDATORY hard-wired literal — see docstring
                )
        except ExtractionError:
            raise
        except Exception as e:
            raise ExtractionError(
                f"native: packed truncated forward failed for "
                f"{hook_name!r}@{layer}: {type(e).__name__}: {e}"
            ) from e
        resid = out.last_hidden_state[0].detach()
        slices: list[t.Tensor] = []
        offset = 0
        for L in lengths:
            slices.append(resid[offset : offset + L, :])
            offset += L
        return slices


def build_native_model(
    model_name: str,
    device: str | t.device,
    model_dtype: t.dtype,
    layer: int,
) -> NativeCollector | None:
    """Load a Gemma{2,3} model and return its truncated NativeCollector, or None.

    I/O ONLY (the load half of the load/transform split, spec §3.1): pin
    `_NATIVE_ATTN_IMPLEMENTATION` ("sdpa") — the S2 row-packing path runs the
    SAME sdpa forward, just on packed inputs (FlexAttention was evaluated and
    dropped: slower end-to-end on every tested GPU incl. H100, and less
    robust to compile) — `from_pretrained` fp32, `.eval()`, structurally
    truncate on CPU, then convert/move only the truncated collector to
    `model_dtype`/`device`. The entire PURE structural-truncation transform
    (structure-assert, `0<=layer<n_layers`, slice `.layers`/`config`, Identity
    norm, ALWAYS-free `>layer`+`lm_head`+orig-norm with the gc-pinning fixed)
    lives in `NativeCollector.truncated`, the genuine public load/transform
    boundary — this just delegates to it after the load.

    Returns None for out-of-v1-scope architectures (e.g. gpt2) or a load
    failure so Backend.auto can fall through to nnterp/TL, matching
    `build_nnterp_model`'s contract. A loaded-but-structurally-unexpected
    model raises a typed `ExtractionError` from `NativeCollector.truncated`
    (fails loudly — never silently truncates the wrong thing).
    """
    if model_name not in EXTRACTION_MODELS:
        logger.warning(f"native: {model_name!r} not in EXTRACTION_MODELS")
        return None
    if not model_name.startswith(("gemma-2", "gemma-3")):
        logger.warning(f"native: no v1 support for {model_name!r} (Gemma only)")
        return None

    hf_id = EXTRACTION_MODELS[model_name].hf_model_id
    try:
        if model_name.startswith("gemma-3"):
            _loaded: Any = load_gemma3_text_decoder(
                hf_id,
                dtype=t.float32,  # load fp32, convert below (cf. nnterp)
                attn_implementation=_NATIVE_ATTN_IMPLEMENTATION,
            )
        else:
            _loaded = from_pretrained_checked(
                AutoModelForCausalLM, hf_id,
                dtype=t.float32, attn_implementation=_NATIVE_ATTN_IMPLEMENTATION,
            )
    except ModelWeightLoadError:
        raise  # FATAL — never silently fall back to another (also-broken) backend
    except Exception as e:
        logger.warning(f"native: failed to load {model_name} ({hf_id}): {e!r}")
        return None
    # widen HF union so .to(str|device) / overload-wrapped calls type-check
    # (mirrors the Phase-1 `hf_model: Any = _loaded` idiom)
    hf_model: Any = _loaded
    hf_model = hf_model.eval()
    collector = NativeCollector.truncated(
        hf_model, model_name=model_name, layer=layer
    )
    if model_dtype != t.float32:
        collector.to(model_dtype)
        logger.info(f"native: converted {model_name} to {model_dtype}")
    collector.to(device)

    return collector
