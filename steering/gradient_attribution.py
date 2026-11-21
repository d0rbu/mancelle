"""Gradient-attribution scoring of SAE features (offline producer core).

For every (segment, stored feature) of a base extraction shard this collector
computes a non-negative gradient **sensitivity**: ``max over the window's read
rows of ReLU(-dL/df)`` (gated by feature activity ``encode(resid) > 0``), where
``dL/df`` is the loss gradient at the SAE's residual hook pushed through the SAE
decoder (``grad @ W_dec.T``). The OFFLINE PRODUCER
(:mod:`scripts.preprocess.precompute_gradient_attribution`) then multiplies this
sensitivity by the feature's activation PERCENTILE to form the stored
``pct · sensitivity`` attribution — the percentile (each feature vs its own
distribution) keeps a barely-active feature with a large per-unit gradient from
dominating, without over-weighting globally high-firing features the way raw
magnitude would. The collector here is the pure gradient half; the percentile
join lives in the producer.

Mechanism
---------

* The FULL Gemma model runs eagerly (SDPA, no compile, ``use_cache=False``)
  on right-padded **maskless** batches — the same pad-invariance contract as
  the nnterp/native extraction backends (causal masking makes real-token
  rows invariant to trailing pads; loss positions are restricted to real
  ranges; pad-row gradients are never read).
* All parameters are frozen (``requires_grad_(False)``). A forward hook on
  ``model.model.layers[L]`` flips ``requires_grad_(True)`` on the layer
  output (``capt``), making it a LEAF autograd root: no graph below layer L
  and no parameter gradients, so the only live graph is the TRUNK
  ``capt -> hidden`` (the post-final-norm hidden state).
* Per-position CE losses are produced lazily through a chunked lm_head tail
  (``lm_head(hidden[:, lo:hi])`` + Gemma-2's tanh softcap when configured,
  CE in fp32) built off a DETACHED copy of ``hidden``, then attributed in two
  VJP steps per schedule unit (see ``score_documents``):
  (1) a BRANCH backward ``loss -> hidden_leaf`` whose graph is freed
  immediately (default ``retain_graph=False``), and (2) a TRUNK
  ``torch.autograd.grad(hidden, capt, grad_outputs=g_hidden)`` that rides the
  one retained constant-size trunk. The per-unit results accumulate into an
  explicit ``capt_grad`` sum. Backwarding the WHOLE graph with
  ``retain_graph=True`` instead kept every unit's 262k-vocab logit branch
  alive at once -> unbounded VRAM; the two-step split frees each branch while
  keeping the trunk, and is bit-exact with the fused backward by linearity.

The spec quantity
-----------------

The gradient a segment is scored with is the **ordered sum of per-unit
gradients in exactly the schedule order** produced by
:func:`build_backward_schedule` (mathematically equal to the suffix/self
loss gradient; numerically pinned by the ordering so a reference oracle that
sums independent ``torch.autograd.grad`` calls per unit in the same order
matches bit-exactly on CPU fp32). Summing several docs' unit losses into one
backward is exact even in floating point: the backward of
``stack(losses).sum()`` delivers cotangent 1.0 to every per-doc subgraph
independently, and attention never crosses batch rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, cast

import torch as t
import torch.nn.functional as F
from loguru import logger
from transformers import AutoModelForCausalLM

from errors import GradientAttributionError
from steering.feature_io import FP16_SAFE_MAX
from steering.gemma3_loading import from_pretrained_checked, load_gemma3_text_decoder
from steering.gradient_overlay import (
    MIN_SCOREABLE_TOKENS,
    GradientAttributionMode,
)
from steering.models import EXTRACTION_MODELS

# ---------------------------------------------------------------------------
# Backward schedule — THE SPEC, as data produced by a pure function
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SegmentRead:
    """Read segment ``seg_idx``'s scores from gradient rows ``[row_lo, row_hi)``.

    The rows are the attribution unit's read window — under a short-final
    merge BOTH merged segments carry the same (unit-wide) row window. Windows
    are never empty: every segment scored by gradient attribution has a
    readable window by construction (the producer enforces doc_len >= 2,
    segment_size >= 2, and min_final_segment_tokens >= 2 — see
    :func:`build_backward_schedule`), and the schedule RAISES rather than
    emit an empty read or a zero-signal placeholder.
    """

    seg_idx: int
    row_lo: int
    row_hi: int


@dataclass(frozen=True)
class BackwardUnit:
    """One backward pass over loss positions ``[loss_lo, loss_hi)`` followed by
    the segment reads it FINALIZES.

    Why ``reads`` is a tuple attached to backward units (reverse
    accumulation): every segment's suffix loss is different, but suffixes
    nest — ``suffix(m-1) = chunk(m) + suffix(m)``. Instead of one backward
    per segment (O(S²) backward cost), the schedule walks chunks from the
    doc end DOWNWARD, accumulating each chunk's gradient into the running
    ``capt_grad`` sum (see ``score_documents``); after folding in chunk ``m``,
    the accumulated gradient IS segment ``m-1``'s full suffix gradient, so
    segment ``m-1`` is read exactly then.
    A unit's ``reads`` are the segments whose accumulation completes at that
    point — usually one; several only for the doc-final unit (a merged short
    final reads both slots, and the below-segment read rides on phase 0 when
    phase 1 has no loss to backward).

    ``key`` identifies the unit's global phase so the batched executor can
    merge corresponding units of different documents into a single backward:
    ``("final", 0)`` / ``("final", 1)`` / ``("chunk", m)`` for the future
    modes, ``("self", m)`` for self mode. ``reset_grad_before`` is True for
    every self-mode unit (per-chunk gradient isolation).

    Loss position ``j`` is the prediction of token ``j + 1``. Units with an
    empty loss range are never emitted (the collector asserts).
    """

    key: tuple
    loss_lo: int
    loss_hi: int
    reads: tuple[SegmentRead, ...]
    reset_grad_before: bool = False


def _pred_range(a: int, b: int) -> tuple[int, int]:
    """Loss positions predicting tokens ``[a, b)``: ``j + 1 in [a, b), j >= 0``.

    Token 0 is never predicted (it is the input start / BOS), hence the
    ``max(a, 1)`` clip — the one legitimate adjustment. A malformed span
    (negative start, empty, or backwards) raises instead of clipping. The
    RESULT is empty ONLY for the span ``[0, 1)`` (token 0 has no predictor);
    callers that could construct that span guard it explicitly and raise.
    """
    if not (0 <= a < b):
        raise GradientAttributionError(
            f"malformed token span [{a}, {b}) in the backward schedule — "
            f"empty/backwards spans must never be constructed."
        )
    return (max(a, 1) - 1, b - 1)


def _final_unit_bounds(
    segments: list[tuple[int, int]],
    min_final_segment_tokens: int,
) -> tuple[int, list[int]]:
    """Resolve the doc-final attribution unit.

    Returns ``(u, unit_seg_indices)``: the unit spans ``[u, e)`` where ``e``
    is the doc end, and ``unit_seg_indices`` are the segments scored from the
    unit's read window. A doc-final segment shorter than
    ``min_final_segment_tokens`` (>= 2, enforced upstream) is merged with the
    second-to-last segment (one merge, never cascaded — non-final segments
    are always full ``segment_size`` under fixed tiling), guaranteeing every
    final unit is >= 2 tokens. A single-segment doc has nothing to merge with
    and is its own unit; with ``doc_len >= 2`` it is already >= 2 tokens.
    """
    n_segs = len(segments)
    b_final, e_final = segments[-1]
    # `n_segs >= 2` is a segment COUNT (a second-to-last must exist to merge
    # into) — a different quantity from MIN_SCOREABLE_TOKENS, so it stays a
    # literal. The length comparison is against the configured merge threshold.
    if n_segs >= 2 and (e_final - b_final) < min_final_segment_tokens:
        return segments[-2][0], [n_segs - 2, n_segs - 1]
    return b_final, [n_segs - 1]


def build_backward_schedule(
    segments: list[tuple[int, int]],
    doc_len: int,
    mode: GradientAttributionMode,
    min_final_segment_tokens: int,
) -> tuple[BackwardUnit, ...]:
    """Build one document's backward/read schedule — the attribution spec.

    Every gradient-attributable segment has a readable window by
    construction; this function RAISES (never returns a zero-signal
    placeholder) when a structurally unscoreable input slips through the
    producer's upstream guards (``doc_len >= 2``, ``segment_size >= 2``,
    ``min_final_segment_tokens >= 2``):

      * ``doc_len < 2`` — the doc has no LM loss at all (token 0 is never
        predicted);
      * a self-mode first segment ``[0, 1)`` — a 1-token segment has no own
        predictions (only reachable with ``segment_size == 1``);
      * a doc-final unit of 1 token — an unscoreable final that escaped
        merging (only reachable with ``min_final_segment_tokens < 2``).

    Invariants (pinned by tests):
      * every segment appears in EXACTLY ONE unit's reads;
      * every emitted unit has a non-empty loss range and only non-empty
        read windows;
      * future modes: a segment's accumulated gradient at read time is the
        ordered sum of the doc's prior units' gradients = the gradient of
        its post-segment suffix loss (the doc-final unit replaces the final
        — and, when merged, the second-to-last — segment's treatment);
      * self mode: every unit resets accumulation first, so reads see only
        their own unit's gradient.
    """
    if not segments:
        raise ValueError("build_backward_schedule requires >= 1 segment")
    if segments[-1][1] != doc_len:
        raise ValueError(
            f"segments must tile the document: last segment ends at "
            f"{segments[-1][1]} but doc_len={doc_len}"
        )
    if doc_len < MIN_SCOREABLE_TOKENS:
        raise GradientAttributionError(
            f"doc_len={doc_len} has no LM loss — a gradient-attributable doc "
            f"needs >= {MIN_SCOREABLE_TOKENS} tokens (token 0 is never "
            f"predicted). Reject short docs upstream."
        )
    e_doc = doc_len
    u, unit_segs = _final_unit_bounds(segments, min_final_segment_tokens)
    first_unit_seg = unit_segs[0]
    unit_len = e_doc - u
    if unit_len < MIN_SCOREABLE_TOKENS:
        raise GradientAttributionError(
            f"doc-final attribution unit [{u}, {e_doc}) is {unit_len} token(s) "
            f"— a 1-token final has no readable signal. With "
            f"min_final_segment_tokens >= {MIN_SCOREABLE_TOKENS} a short "
            f"final always merges; got "
            f"min_final_segment_tokens={min_final_segment_tokens}."
        )

    if mode is GradientAttributionMode.self_:
        units: list[BackwardUnit] = []
        for m in range(first_unit_seg):
            b_m, e_m = segments[m]
            lo, hi = _pred_range(b_m, e_m)
            if hi <= lo:
                # Only segment [0, 1): its single token has no predictor —
                # reachable only with segment_size == 1.
                raise GradientAttributionError(
                    f"self-mode segment {m} [{b_m}, {e_m}) has no own "
                    f"predictions (a 1-token segment at the doc start). Use "
                    f"segment_size >= 2."
                )
            units.append(
                BackwardUnit(
                    key=("self", m),
                    loss_lo=lo,
                    loss_hi=hi,
                    reads=(SegmentRead(m, b_m, e_m),),
                    reset_grad_before=True,
                )
            )
        # The final unit (merged or single-segment alike; unit_len >= 2).
        lo, hi = _pred_range(u, e_doc)
        units.append(
            BackwardUnit(
                key=("self", first_unit_seg),
                loss_lo=lo,
                loss_hi=hi,
                reads=tuple(SegmentRead(s, u, e_doc) for s in unit_segs),
                reset_grad_before=True,
            )
        )
        return tuple(units)

    # Only future_half_split / future_self_full reach here: `self` returned
    # above and GradientAttributionMode is an exhaustive 3-value enum.
    below_read: tuple[SegmentRead, ...] = ()
    if first_unit_seg >= 1:
        b_p, e_p = segments[first_unit_seg - 1]
        below_read = (SegmentRead(first_unit_seg - 1, b_p, e_p),)

    # unit_len >= 2, so half_split always splits into a proper source/read.
    if mode is GradientAttributionMode.future_half_split:
        src_lo = u + unit_len // 2
        read_hi = src_lo
    else:  # future_self_full
        src_lo = u
        read_hi = e_doc
    final_reads = tuple(SegmentRead(s, u, read_hi) for s in unit_segs)

    units = []
    lo0, hi0 = _pred_range(src_lo, e_doc)
    # Phase 1 backwards the unit's remaining predictions [u, src_lo) so the
    # below-segment's suffix is complete at its read. It is emitted only when
    # that loss is non-empty AND something below depends on it; otherwise the
    # below-segment read rides on phase 0 (the accumulated gradient is
    # identical — an unemitted phase 1 contributes nothing anyone reads).
    lo1, hi1 = _pred_range(u, src_lo) if src_lo > u else (0, 0)
    if hi1 > lo1 and below_read:
        units.append(
            BackwardUnit(key=("final", 0), loss_lo=lo0, loss_hi=hi0, reads=final_reads)
        )
        units.append(
            BackwardUnit(key=("final", 1), loss_lo=lo1, loss_hi=hi1, reads=below_read)
        )
    else:
        units.append(
            BackwardUnit(
                key=("final", 0),
                loss_lo=lo0,
                loss_hi=hi0,
                reads=final_reads + below_read,
            )
        )
    for m in range(first_unit_seg - 1, 0, -1):
        b_m, e_m = segments[m]
        lo_m, hi_m = _pred_range(b_m, e_m)  # b_m >= 1 ⇒ never empty
        b_r, e_r = segments[m - 1]
        units.append(
            BackwardUnit(
                key=("chunk", m),
                loss_lo=lo_m,
                loss_hi=hi_m,
                reads=(SegmentRead(m - 1, b_r, e_r),),
            )
        )
    return tuple(units)


def _global_key_order(key: tuple) -> tuple:
    """Canonical execution order for merged per-doc unit keys.

    Future modes: ``("final", 0)`` then ``("final", 1)`` then ``("chunk", m)``
    with m DESCENDING (each doc's own units thereby execute in its per-doc
    schedule order). Self mode: ``("self", m)`` ascending. The two key
    families never mix within a run.
    """
    kind = key[0]
    if kind == "final":
        return (0, key[1])
    if kind == "chunk":
        return (1, -key[1])
    if kind == "self":
        return (0, key[1])
    raise ValueError(f"Unknown schedule key: {key!r}")


def group_units_by_key(
    schedules: list[tuple[BackwardUnit, ...]],
) -> list[tuple[tuple, list[tuple[int, BackwardUnit]]]]:
    """Merge per-doc schedules into globally ordered ``(key, [(doc, unit)])``.

    Documents only contribute units for keys they have; per-doc relative
    order is preserved by the canonical key order (verified in tests).
    """
    by_key: dict[tuple, list[tuple[int, BackwardUnit]]] = {}
    for d, units in enumerate(schedules):
        for unit in units:
            by_key.setdefault(unit.key, []).append((d, unit))
    return sorted(by_key.items(), key=lambda kv: _global_key_order(kv[0]))


# ---------------------------------------------------------------------------
# SAE decoder-linearity gate
# ---------------------------------------------------------------------------

_SUPPORTED_SAE_ARCHITECTURES = frozenset({"standard", "jumprelu"})


def _sae_architecture(sae: Any) -> str | None:
    """Read the SAE architecture defensively (sae_lens exposes it as a cfg
    attribute in 5.x; tolerate a callable for forward-compat)."""
    arch = getattr(sae.cfg, "architecture", None)
    if callable(arch):
        arch = arch()
    return arch


def decoder_linearity_gate(
    sae: Any, *, expected_hook_name: str | None = None
) -> None:
    """Hard gate: the explicit ``grad @ W_dec.T`` decoder VJP must be EXACTLY
    what autograd computes through ``sae.decode`` for this SAE instance.

    Config guards first (fail with the reason), then an exactness probe
    (``torch.equal``) on random vectors. No silent fallback — mirroring the
    extraction compile/rowpack gate culture.

    Raises:
        GradientAttributionError: unsupported architecture, runtime input
            normalization, a finetuning scaling factor, or a probe mismatch.
    """
    arch = _sae_architecture(sae)
    if arch not in _SUPPORTED_SAE_ARCHITECTURES:
        raise GradientAttributionError(
            f"gradient attribution supports SAE architectures "
            f"{sorted(_SUPPORTED_SAE_ARCHITECTURES)}, got {arch!r}; the "
            f"decoder VJP is only proven for the plain affine decoder."
        )
    normalize = getattr(sae.cfg, "normalize_activations", "none") or "none"
    if normalize != "none":
        raise GradientAttributionError(
            f"gradient attribution requires normalize_activations='none' "
            f"(stateless affine decode); got {normalize!r}. Runtime "
            f"normalization makes standalone decode stateful and the "
            f"W_dec^T shortcut wrong."
        )
    if bool(getattr(sae.cfg, "finetuning_scaling_factor", False)):
        raise GradientAttributionError(
            "gradient attribution does not support finetuning_scaling_factor "
            "SAEs (decode is no longer the plain affine `acts @ W_dec + b_dec`)."
        )
    # hook_name lives on cfg directly in sae_lens 5.x and under cfg.metadata
    # in 6.x. The producer supplies the resolver's exact identity; standalone
    # gate callers retain the legacy metadata-optional behavior.
    direct_hook = getattr(sae.cfg, "hook_name", None)
    meta = getattr(sae.cfg, "metadata", None)
    metadata_hook = getattr(meta, "hook_name", None) if meta is not None else None
    if expected_hook_name is not None:
        expected_parts = expected_hook_name.split(".")
        expected_is_resid_post = (
            len(expected_parts) == 3
            and expected_parts[0] == "blocks"
            and expected_parts[1].isdigit()
            and expected_parts[2] == "hook_resid_post"
        )
        if not expected_is_resid_post:
            raise GradientAttributionError(
                f"resolved SAE hook {expected_hook_name!r} is not an exact "
                f"blocks.<layer>.hook_resid_post identity."
            )
        if direct_hook is None and metadata_hook is None:
            raise GradientAttributionError(
                f"loaded SAE hook metadata is missing; expected exact "
                f"hook_name={expected_hook_name!r}."
            )
        if (
            direct_hook is not None
            and metadata_hook is not None
            and direct_hook != metadata_hook
        ):
            raise GradientAttributionError(
                f"loaded SAE has conflicting hook metadata: "
                f"cfg.hook_name={direct_hook!r}, "
                f"cfg.metadata.hook_name={metadata_hook!r}; expected "
                f"{expected_hook_name!r}."
            )
        loaded_hook = direct_hook if direct_hook is not None else metadata_hook
        if loaded_hook != expected_hook_name:
            raise GradientAttributionError(
                f"loaded SAE hook_name={loaded_hook!r}, expected exact "
                f"resolver hook {expected_hook_name!r}."
            )
    else:
        loaded_hook = direct_hook if direct_hook is not None else metadata_hook
    if loaded_hook and "hook_resid" not in loaded_hook:
        raise GradientAttributionError(
            f"gradient attribution is residual-only; SAE hook_name="
            f"{loaded_hook!r} is not a residual-stream site."
        )

    device = sae.W_dec.device
    dtype = sae.W_dec.dtype
    gen = t.Generator(device="cpu").manual_seed(0)
    n_probe = 4
    acts = t.randn(
        n_probe, sae.W_dec.shape[0], generator=gen, dtype=t.float32
    ).to(device=device, dtype=dtype)
    cotangent = t.randn(
        n_probe, sae.W_dec.shape[1], generator=gen, dtype=t.float32
    ).to(device=device, dtype=dtype)
    acts_leaf = acts.detach().requires_grad_(True)
    (autograd_vjp,) = t.autograd.grad(
        (sae.decode(acts_leaf) * cotangent).sum(), acts_leaf
    )
    explicit_vjp = cotangent @ sae.W_dec.T
    if not t.equal(autograd_vjp, explicit_vjp):
        max_abs = (autograd_vjp - explicit_vjp).abs().max().item()
        raise GradientAttributionError(
            f"decoder-linearity gate FAILED: autograd VJP through sae.decode "
            f"differs from `cotangent @ W_dec.T` (max_abs={max_abs:.3e}). "
            f"This SAE's decode is not the plain affine map the gradient "
            f"attribution shortcut assumes; refusing to emit scores."
        )


# ---------------------------------------------------------------------------
# Model build
# ---------------------------------------------------------------------------


def build_gradient_model(
    model_name: str,
    device: str | t.device,
    model_dtype: t.dtype,
):
    """Load the FULL-depth HF model for gradient attribution.

    Mirrors the nnterp builders' load path (same HF ids via
    ``EXTRACTION_MODELS``, ``attn_implementation="sdpa"``) but bare HF — no
    nnsight wrapping, no truncation (the loss needs the lm_head). Eval mode,
    all parameters frozen.

    Raises:
        GradientAttributionError: for non-Gemma models (the collector's
            ``model.model.layers`` / softcap handling is Gemma-2/3 only,
            mirroring the native collector's scope).
        ModelWeightLoadError: if the Gemma-3 loader detects a VLM checkpoint
            with randomly-initialised text weights (issue #549 fail-fast).
    """
    if model_name not in EXTRACTION_MODELS or not model_name.startswith(
        ("gemma-2", "gemma-3")
    ):
        raise GradientAttributionError(
            f"gradient attribution supports Gemma-2/Gemma-3 extraction models "
            f"only, got {model_name!r} (registered: "
            f"{sorted(EXTRACTION_MODELS)})."
        )
    hf_id = EXTRACTION_MODELS[model_name].hf_model_id
    if model_name.startswith("gemma-3"):
        model = load_gemma3_text_decoder(
            hf_id, dtype=model_dtype, attn_implementation="sdpa"
        )
    else:
        model = from_pretrained_checked(
            AutoModelForCausalLM, hf_id, dtype=model_dtype, attn_implementation="sdpa"
        )
    # cast: transformers' wrapped `.to` overloads confuse the type checker.
    model = cast(Any, model).to(t.device(device))
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    logger.info(
        f"Built gradient-attribution model {model_name} ({hf_id}) "
        f"dtype={model_dtype} on {device} (frozen, eager SDPA, full depth)"
    )
    return model


# ---------------------------------------------------------------------------
# The collector
# ---------------------------------------------------------------------------


class GradientAttributionCollector:
    """Scores documents' stored SAE features by gradient attribution.

    Args:
        model: A full-depth HF Gemma-2/Gemma-3 causal LM in eval mode. All
            parameters are (re-)frozen here — the leaf-root hook trick
            requires it.
        sae: A sae_lens SAE on the same device/dtype (callers run
            :func:`decoder_linearity_gate` once at build).
        layer: The SAE's residual layer L; gradients are read at the output
            of ``model.model.layers[layer]`` (== ``hook_resid_post(L)``).
        mode: Attribution mode (three-state).
        min_final_segment_tokens: Short-final merge threshold (see
            :func:`build_backward_schedule`).
        bos_token_id / ignore_bos_features: when the base shards were
            extracted with ``ignore_bos_features=true``, BOS rows are
            excluded from every read window's max-reduction (keep-mask).
    """

    def __init__(
        self,
        model: Any,
        sae: Any,
        *,
        layer: int,
        mode: GradientAttributionMode,
        min_final_segment_tokens: int,
        bos_token_id: int | None = None,
        ignore_bos_features: bool = False,
    ):
        if ignore_bos_features and bos_token_id is None:
            raise GradientAttributionError(
                "ignore_bos_features=True requires a bos_token_id"
            )
        n_layers = len(model.model.layers)
        if not (0 <= layer < n_layers):
            raise GradientAttributionError(
                f"layer {layer} out of range for model with {n_layers} layers"
            )
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        # Freeze the SAE too: this is a pure offline SCORER, never trained. If
        # W_dec carries requires_grad, `_score_read`'s decoder pull-back
        # (`grad_rows @ W_dec[fids].T`) builds an autograd graph and the
        # returned score row escapes with a grad_fn pinning its saved tensors —
        # a per-call retention the frozen-SAE test fixtures never exercised.
        # Forward values are identical either way (bit-exactness preserved).
        for p in sae.parameters():
            p.requires_grad_(False)
        self.model = model
        self.sae = sae
        self.layer = layer
        self.mode = mode
        self.min_final_segment_tokens = min_final_segment_tokens
        self.bos_token_id = bos_token_id
        self.ignore_bos_features = ignore_bos_features
        self.device = next(model.parameters()).device
        self._final_logit_softcapping = getattr(
            model.config, "final_logit_softcapping", None
        )

    # -- loss tail ---------------------------------------------------------

    def _chunk_loss(
        self, hidden: t.Tensor, input_ids: t.Tensor, d: int, lo: int, hi: int
    ) -> t.Tensor:
        """Sum of fp32 CE losses for doc ``d``'s positions ``[lo, hi)``.

        Reproduces the HF causal-LM tail exactly: lm_head on the post-norm
        hidden states, Gemma-2's tanh softcap when configured, CE on
        fp32-upcast logits. ``reduction="sum"`` — the schedule's spec
        quantity sums position losses within a unit.
        """
        logits = self.model.lm_head(hidden[d, lo:hi])
        cap = self._final_logit_softcapping
        if cap is not None:
            logits = t.tanh(logits / cap) * cap
        targets = input_ids[d, lo + 1 : hi + 1]
        return F.cross_entropy(logits.float(), targets, reduction="sum")

    # -- per-read scoring ----------------------------------------------------

    def _score_read(
        self,
        capt: t.Tensor,
        capt_grad: t.Tensor | None,
        input_ids: t.Tensor,
        d: int,
        read: SegmentRead,
        fids: t.Tensor,
    ) -> t.Tensor:
        """One segment's fp16 score row from the accumulated gradient.

        ``capt`` carries the residual-stream VALUES (for the activity gate);
        ``capt_grad`` is the schedule-accumulated loss gradient AT capt for
        this read (the suffix/self gradient — see ``score_documents``).

        The five steps (one numbered block each below):

        1. PULL BACK — the accumulated loss gradient at the residual stream,
           ``capt_grad[d, rows]`` of shape ``[R, d_model]``, is mapped into
           feature space through the decoder: ``g = grad @ W_dec[fids].T``
           (``[R, F]``). Decode is affine (``decoder_linearity_gate`` proves
           it for this SAE), so ``g[r, f]`` is EXACTLY ``dL/d(activation of
           feature f at row r)``.
        2. GATE — a feature only counts where it actually fired:
           ``encode(resid) > 0`` per (row, feature); inactive cells drop to 0.
        3. BOS MASK — when the base shards were extracted with
           ``ignore_bos_features``, BOS rows are excluded the same way.
        4. REDUCE — ``score = max over rows of ReLU(-g)``: the gradient
           SENSITIVITY, how strongly INCREASING the feature would have
           DECREASED the loss. Non-negative. The producer then multiplies this
           by the feature's activation PERCENTILE to form the stored
           ``pct · sensitivity`` score (see ``precompute_gradient_attribution``)
           — the collector itself stays a pure gradient tool.
        5. PERSIST GUARD — refuse non-finite scores AND scores at or above
           ``FP16_SAFE_MAX`` (fp16 is the storage dtype; a finite fp32 value
           >= 65520 would silently become +inf — the repo-wide
           ``steering.feature_io`` contract).

        Raises on structurally impossible reads (empty window, missing
        accumulated gradient, a segment with no stored features): the
        schedule never emits them.
        """
        if read.row_hi <= read.row_lo or capt_grad is None or fids.numel() == 0:
            raise GradientAttributionError(
                f"invalid read for doc {d} segment {read.seg_idx}: rows "
                f"[{read.row_lo}, {read.row_hi}), accumulated grad "
                f"{'missing' if capt_grad is None else 'present'}, "
                f"{int(fids.numel())} stored features — the schedule must "
                f"never emit empty reads."
            )
        rows = slice(read.row_lo, read.row_hi)
        fids_dev = fids.to(device=self.device, dtype=t.int64)

        # 1. Pull the residual-stream gradient back through the decoder.
        grad_rows = capt_grad[d, rows].to(t.float32)  # [R, d_model]
        w_rows = self.sae.W_dec.index_select(0, fids_dev).to(t.float32)
        g = grad_rows @ w_rows.T  # [R, F]

        # 2. Activity gate — only rows where the feature fired contribute.
        with t.no_grad():
            acts = self.sae.encode(capt[d, rows].detach())
        gate = acts.index_select(1, fids_dev) > 0  # [R, F]
        vals = t.relu(-g).masked_fill(~gate, 0.0)

        # 3. BOS keep-mask (matches the extraction-side exclusion).
        if self.ignore_bos_features:
            assert self.bos_token_id is not None  # enforced in __init__
            keep = input_ids[d, rows] != int(self.bos_token_id)
            vals = vals.masked_fill(~keep.unsqueeze(1), 0.0)

        # 4. max over the window's rows of ReLU(-g) (gradient sensitivity).
        score = vals.amax(dim=0)

        # 5. Refuse anything the fp16 overlay cannot represent faithfully.
        if not bool(t.isfinite(score).all()) or bool(
            (score >= FP16_SAFE_MAX).any()
        ):
            raise GradientAttributionError(
                f"gradient-attribution score for doc {d}, segment "
                f"{read.seg_idx} (rows [{read.row_lo}, {read.row_hi})) is "
                f"non-finite or >= {FP16_SAFE_MAX:.0f} and would corrupt the "
                f"fp16 overlay; refusing to persist."
            )
        return score.to(t.float16).cpu()

    # -- main entry ----------------------------------------------------------

    def score_documents(
        self,
        token_id_lists: list[list[int]],
        doc_segments: list[list[tuple[int, int]]],
        doc_stored_fids: list[list[t.Tensor]],
    ) -> list[list[t.Tensor]]:
        """Score a batch of documents.

        Args:
            token_id_lists: per-doc SAE-tokenizer ids (BOS included,
                head-truncated — exactly the extraction-time token stream).
            doc_segments: per-doc fixed-tiling ``(start, end)`` segment spans.
            doc_stored_fids: per-doc per-segment stored feature-id tensors
                (ascending-by-id, as loaded from the base shard).

        Returns:
            Per-doc lists of fp16 CPU score rows aligned 1:1 with
            ``doc_stored_fids``.
        """
        assert len(token_id_lists) == len(doc_segments) == len(doc_stored_fids)
        assert all(len(ids) > 0 for ids in token_id_lists)
        for segs, fid_rows in zip(doc_segments, doc_stored_fids, strict=True):
            assert len(segs) == len(fid_rows), (
                f"{len(segs)} segments but {len(fid_rows)} fid rows"
            )

        n_docs = len(token_id_lists)
        lengths = [len(ids) for ids in token_id_lists]
        max_seq = max(lengths)
        input_ids = t.zeros(n_docs, max_seq, dtype=t.long, device=self.device)
        for i, ids in enumerate(token_id_lists):
            input_ids[i, : len(ids)] = t.tensor(
                ids, dtype=t.long, device=self.device
            )

        # The schedule RAISES on any structurally unscoreable doc (doc_len < 2,
        # a 1-token self segment, a 1-token final) — never a zero placeholder.
        schedules: list[tuple[BackwardUnit, ...]] = [
            build_backward_schedule(
                doc_segments[d], lengths[d], self.mode,
                self.min_final_segment_tokens,
            )
            for d in range(n_docs)
        ]
        results: list[list[t.Tensor | None]] = [
            [None] * len(segs) for segs in doc_segments
        ]
        grouped = group_units_by_key(schedules)

        captured: list[t.Tensor] = []

        def _hook(_module, _args, output):
            h = output[0] if isinstance(output, tuple) else output
            if h.grad_fn is not None:
                raise GradientAttributionError(
                    "layer output already carries a graph — model parameters "
                    "must be frozen for the leaf-root attribution hook."
                )
            h.requires_grad_(True)
            captured.append(h)

        handle = self.model.model.layers[self.layer].register_forward_hook(_hook)
        try:
            out = self.model.model(
                input_ids=input_ids, attention_mask=None, use_cache=False
            )
            hidden = out.last_hidden_state
            if len(captured) != 1:
                raise GradientAttributionError(
                    f"expected exactly one layer-{self.layer} capture, got "
                    f"{len(captured)}"
                )
            capt = captured[0]
            # The lm_head loss tail is built off a DETACHED copy of hidden so
            # that each unit-group's (262k-vocab, multi-token) logit branch is
            # freed right after its backward — only the TRUNK (capt -> hidden)
            # is retained across groups, via the explicit trunk VJP below.
            # Backwarding the whole graph with retain_graph=True instead kept
            # EVERY group's logit branch alive at once -> unbounded VRAM.
            hidden_leaf = hidden.detach().requires_grad_(True)
            # The schedule-accumulated loss gradient AT capt (the suffix/self
            # gradient), grown group by group; reads see the running sum. By
            # linearity, sum_g trunk_VJP(g_hidden) == trunk_VJP(sum_g g_hidden)
            # == the gradient a single fused backward would deposit on capt.
            capt_grad: t.Tensor | None = None

            n_groups = len(grouped)
            for gi, (_key, doc_units) in enumerate(grouped):
                if doc_units[0][1].reset_grad_before:
                    capt_grad = None
                loss_terms: list[t.Tensor] = []
                for d, unit in doc_units:
                    if unit.loss_hi <= unit.loss_lo:
                        raise GradientAttributionError(
                            f"unit {unit.key} for doc {d} has an empty loss "
                            f"range [{unit.loss_lo}, {unit.loss_hi}) — the "
                            f"schedule must never emit empty backwards."
                        )
                    loss_terms.append(
                        self._chunk_loss(
                            hidden_leaf, input_ids, d, unit.loss_lo, unit.loss_hi
                        )
                    )
                total = (
                    loss_terms[0]
                    if len(loss_terms) == 1
                    else t.stack(loss_terms).sum()
                )
                # Branch VJP (loss -> hidden): frees the logit branch (default
                # retain_graph=False); hidden_leaf.grad is THIS group's only.
                hidden_leaf.grad = None
                total.backward()
                del loss_terms, total
                g_hidden = hidden_leaf.grad
                assert g_hidden is not None
                # Trunk VJP (hidden -> capt). retain_graph keeps the ONE shared
                # trunk graph alive for the NEXT group's VJP; the LAST group
                # frees it (retain_graph=False) so NO forward graph survives the
                # call. Keeping it retained leaks across calls: autograd graphs
                # are cyclic, so a retained graph is reclaimed only by Python's
                # cyclic GC — between GC passes every score_documents call (one
                # per producer _flush batch) leaves its full layer-L->output
                # graph resident, climbing monotonically to OOM.
                (g_capt,) = t.autograd.grad(
                    hidden, capt, grad_outputs=g_hidden,
                    retain_graph=(gi < n_groups - 1),
                )
                capt_grad = g_capt if capt_grad is None else capt_grad + g_capt
                for d, unit in doc_units:
                    for read in unit.reads:
                        results[d][read.seg_idx] = self._score_read(
                            capt, capt_grad, input_ids, d, read,
                            doc_stored_fids[d][read.seg_idx],
                        )
        finally:
            handle.remove()
            # Drop the hook's strong ref to capt so the forward graph cannot
            # outlive the call via the closure (belt-and-suspenders alongside
            # the last-group retain_graph=False above).
            captured.clear()

        final: list[list[t.Tensor]] = []
        for d, rows in enumerate(results):
            doc_rows: list[t.Tensor] = []
            for s, row in enumerate(rows):
                if row is None:  # pragma: no cover - schedule invariant
                    raise GradientAttributionError(
                        f"schedule never read doc {d} segment {s}"
                    )
                doc_rows.append(row)
            final.append(doc_rows)
        return final
