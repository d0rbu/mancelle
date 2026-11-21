"""Identity, hashing, and layout helpers for gradient-attribution overlays.

Deliberately lightweight (no torch-model / transformers imports) so the
selection pipeline, training discovery, and eval guards can import it without
pulling in the GPU extraction stack. The heavy computation lives in
:mod:`steering.gradient_attribution`; the producer scripts live under
``scripts/preprocess/``.

Two artifacts share the vocabulary defined here:

* **Base-aligned attribution overlay** — ``{feature_dir}/grad/<checksum16>/``
  with one shard per base feature shard. Each grad shard is self-contained
  for joins: ``grad_scores`` (fp16, 1:1 with the base shard's CSR
  ``feature_ids``), plus verified copies of the base ``offsets`` and a
  per-doc segment-count cumsum ``doc_seg_offsets`` so consumers never load
  base shards.
* **Packed-segment provenance** — ``{packed_row_dir}/provenance/`` with one
  shard per packed-row shard mapping every packed segment back to
  ``(base_shard_idx, doc_idx, seg_ordinal)``. Config-independent and written
  by the packer itself (``scripts.preprocess.precompute_packed_rows``) in the
  same pass as the packed shards; ``resume=true`` heals it for dirs that
  predate the integration.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from errors import GradientAttributionError

# Code-versioned semantics constants. Both enter the overlay checksum AND the
# selection hash, so any algorithm revision re-addresses every derived
# artifact instead of silently reusing stale ones.
#
# Score-reduction history (the per-(segment, feature) overlay score):
#   relu_neg_max     — max_r [gate * ReLU(-dL/df)]: the bare loss SENSITIVITY
#                      (activation only a binary gate). A barely-active feature
#                      with a large per-unit gradient could dominate.
#   act_relu_neg_max — max_r [a * ReLU(-dL/df)]: first-order attribution
#                      weighting the per-row gradient by the per-row activation.
#                      Still over-weighted globally high-firing features.
#   pct_relu_neg_max — percentile(a) * max_r [gate * ReLU(-dL/df)]: the gradient
#                      SENSITIVITY scaled by the feature's activation PERCENTILE
#                      (each feature vs its OWN distribution). The collector
#                      computes the sensitivity; the producer multiplies by the
#                      percentile (from the pct overlay) — so this overlay's
#                      identity also depends on the pct overlay it was built
#                      against (recorded + validated via the manifest). Different
#                      scores => a NEW content hash => old overlays/sel dirs are
#                      never silently reused.
#
# Split-policy history (the schedule/unit geometry — distinct from the score):
#   v1 — short single-segment docs were silently zero-scored.
#   v2 — single-segment docs are their own final unit; every segment is
#        scored, with structurally unscoreable inputs (1-token docs, 1-token
#        segments/finals) made impossible UPSTREAM (the producer enforces
#        doc_len >= 2, segment_size >= 2, min_final_segment_tokens >= 2) and
#        the schedule RAISING rather than emitting any zero placeholder. The
#        schedule is UNCHANGED by the score-reduction switch, so this stays 2.
GRAD_SCORE_REDUCTION = "pct_relu_neg_max"
GRAD_SPLIT_POLICY_VERSION = 2

# Tensor keys for the base-aligned grad overlay shards.
GRAD_SCORES_KEY = "grad_scores"
GRAD_OFFSETS_KEY = "offsets"
GRAD_DOC_SEG_OFFSETS_KEY = "doc_seg_offsets"

# Tensor keys for the packed-segment provenance shards. Every packed segment
# maps to a real base segment — there is no sentinel/filler value; the packer
# raises on a featureless or zero-segment doc rather than emit one.
PROVENANCE_BASE_SHARD_IDX_KEY = "base_shard_idx"
PROVENANCE_DOC_IDX_KEY = "doc_idx"
PROVENANCE_SEG_ORDINAL_KEY = "seg_ordinal"

# Directory leaf for the provenance sibling (no hash component — the artifact
# is config-independent, written by the packer in the same pass as the packed
# shards).
PROVENANCE_DIRNAME = "provenance"

# The minimum tokens any span needs to carry gradient signal: next-token
# prediction means a length-L span has exactly L-1 loss positions (token 0 is
# never predicted), so a 1-token span has NO readable gradient — nothing
# follows it, and its single row precedes its only (nonexistent) loss position.
# Single source for the "2" across every scoreability floor: a whole doc
# (`doc_len`, `len(kept_ids)`), a segment tiling (`segment_size`), the realized
# doc-final attribution unit (`unit_len`), and the hard floor on
# `min_final_segment_tokens` (so a short final ALWAYS merges into a unit of
# >= 2 tokens — merging is never disengageable). NOT the same quantity as the
# ">= 2 segments needed to have a second-to-last to merge with", which is a
# segment COUNT and stays a literal where it appears.
MIN_SCOREABLE_TOKENS = 2

# Default minimum doc-final segment length before it is merged with the
# second-to-last segment into one attribution unit.
DEFAULT_MIN_FINAL_SEGMENT_TOKENS = 8

_VALID_MODEL_DTYPES = ("bfloat16", "float32")


class GradientAttributionMode(StrEnum):
    """Which loss window each segment's gradient attribution uses.

    Three states so invalid combinations are unrepresentable (the doc-final
    treatment is part of the mode, not a separate knob):

    * ``future_half_split`` — non-final segments attribute the post-segment
      suffix loss; the doc-final attribution unit ``[u, e)`` is split at
      ``mid = u + (e - u) // 2``: backprop predictions of ``[mid, e)``, read
      gradients at rows ``[u, mid)``.
    * ``future_self_full`` — non-final segments identical; the doc-final unit
      attributes its own full window (backprop all of the unit's own
      predictions, read all of the unit's rows).
    * ``self`` — every segment attributes its own chunk's loss, with
      gradients isolated per chunk.
    """

    future_half_split = "future_half_split"
    future_self_full = "future_self_full"
    self_ = "self"


@dataclass(frozen=True)
class GradientRankingSpec:
    """Parsed ``feature_selection.gradient`` block.

    Presence of the block in a feature-selection config IS the gradient-mode
    switch (there is no separate enable key); ``None`` everywhere means
    legacy percentile ranking.
    """

    mode: GradientAttributionMode
    model_dtype: str = "bfloat16"
    min_final_segment_tokens: int = DEFAULT_MIN_FINAL_SEGMENT_TOKENS


def _validate_model_dtype(model_dtype: str) -> str:
    if model_dtype not in _VALID_MODEL_DTYPES:
        raise ValueError(
            f"gradient.model_dtype must be one of {list(_VALID_MODEL_DTYPES)}, "
            f"got {model_dtype!r}"
        )
    return model_dtype


def parse_gradient_ranking_spec(raw: dict) -> GradientRankingSpec:
    """Parse and validate a raw ``gradient`` config mapping.

    Raises:
        ValueError: unknown/missing mode, invalid model_dtype, a
            min_final_segment_tokens that is non-int or < 2, or unknown keys
            (loud rather than silently ignoring a typo'd knob).
    """
    if not isinstance(raw, dict):
        raise ValueError(
            f"feature_selection.gradient must be a mapping, got {type(raw).__name__}"
        )
    known = {"mode", "model_dtype", "min_final_segment_tokens"}
    unknown = set(raw) - known
    if unknown:
        raise ValueError(
            f"feature_selection.gradient has unknown keys {sorted(unknown)}; "
            f"known keys: {sorted(known)}"
        )
    if "mode" not in raw:
        raise ValueError(
            "feature_selection.gradient requires 'mode' "
            f"(one of {[m.value for m in GradientAttributionMode]})"
        )
    try:
        mode = GradientAttributionMode(raw["mode"])
    except ValueError:
        raise ValueError(
            f"Unsupported gradient attribution mode {raw['mode']!r}. "
            f"Supported: {[m.value for m in GradientAttributionMode]}."
        ) from None
    model_dtype = _validate_model_dtype(str(raw.get("model_dtype", "bfloat16")))
    min_final_raw = raw.get(
        "min_final_segment_tokens", DEFAULT_MIN_FINAL_SEGMENT_TOKENS
    )
    if isinstance(min_final_raw, bool) or not isinstance(min_final_raw, int):
        raise ValueError(
            f"gradient.min_final_segment_tokens must be an int "
            f">= {MIN_SCOREABLE_TOKENS}, got {min_final_raw!r}"
        )
    if min_final_raw < MIN_SCOREABLE_TOKENS:
        raise ValueError(
            f"gradient.min_final_segment_tokens must be >= "
            f"{MIN_SCOREABLE_TOKENS} (a 1-token final is unscoreable, so "
            f"it must always merge — merging is never disabled); got "
            f"{min_final_raw}. Use {MIN_SCOREABLE_TOKENS} to merge only "
            f"1-token finals."
        )
    return GradientRankingSpec(
        mode=mode,
        model_dtype=model_dtype,
        min_final_segment_tokens=min_final_raw,
    )


def resolve_gradient_spec_from_feature_selection(
    fs_raw: dict | None,
) -> GradientRankingSpec | None:
    """Resolve the optional gradient block from a raw feature_selection dict.

    Returns ``None`` when the key is absent or explicitly null (legacy
    percentile ranking — byte-identical hash payload). A present block is
    parsed strictly.
    """
    if fs_raw is None:
        return None
    raw = fs_raw.get("gradient")
    if raw is None:
        return None
    return parse_gradient_ranking_spec(raw)


def canonical_gradient_selection_dict(spec: GradientRankingSpec) -> dict:
    """The canonical gradient block for ``compute_selection_hash``.

    Every knob that changes scores or selection semantics is included, so a
    ``sel/<hash>/`` dir can never be silently reused across algorithm or
    config revisions. Keys not implied by the feature/packed dir identity
    only (mirrors how feature_mask hashes its resolved ids, not its source
    path).
    """
    return {
        "mode": spec.mode.value,
        "model_dtype": spec.model_dtype,
        "min_final_segment_tokens": spec.min_final_segment_tokens,
        "score_reduction": GRAD_SCORE_REDUCTION,
        "split_policy_version": GRAD_SPLIT_POLICY_VERSION,
    }


def _normalize_sae_l0(sae_l0: str | int | None) -> str | int | None:
    """Normalize the l0 knob before it enters the checksum payload.

    The producer and the consumer derive l0 from DIFFERENT config files
    (extraction vs training), and OmegaConf/CLI overrides can deliver the
    same value as ``68``, ``"68"``, or ``null`` — which ``json.dumps``
    distinguishes, silently re-addressing a byte-valid overlay. Int-like
    values normalize to int; ``None`` stays ``None``; tier strings (e.g.
    ``"medium"``) stay strings.
    """
    if sae_l0 is None:
        return None
    if isinstance(sae_l0, bool):  # bool is an int subclass — never an l0
        return str(sae_l0)
    if isinstance(sae_l0, int):
        return sae_l0
    s = str(sae_l0).strip()
    try:
        return int(s)
    except ValueError:
        return s


def canonical_gradient_descriptor(
    *,
    attribution_mode: str,
    model: str,
    model_dtype: str,
    layer: int,
    sae_release: str,
    sae_width: str,
    sae_l0: str | int | None,
    activation_type: str,
    ignore_bos_features: bool,
    min_final_segment_tokens: int,
    percentile_checksum: str,
) -> dict:
    """Full identity of a base-aligned grad overlay (the checksum payload).

    Includes everything that changes the overlay's score bytes: the
    attribution mode + merge threshold, the model/SAE identity, dtype, the
    code-versioned reduction/split-policy constants, AND the percentile-overlay
    checksum — the stored score is ``pct * sensitivity``, so a different
    percentile overlay produces different scores and MUST re-address the
    grad/<hash>/ dir (otherwise a force-rebuild against a new percentile overlay
    could silently overwrite scores in place at the same address). Every field
    is normalized here so producer and consumer cannot disagree on spelling.
    """
    mode = GradientAttributionMode(attribution_mode)  # validate
    return {
        "attribution_mode": mode.value,
        "score_reduction": GRAD_SCORE_REDUCTION,
        "split_policy_version": GRAD_SPLIT_POLICY_VERSION,
        "min_final_segment_tokens": int(min_final_segment_tokens),
        "model": str(model),
        "model_dtype": _validate_model_dtype(model_dtype),
        "layer": int(layer),
        "sae_release": str(sae_release),
        "sae_width": str(sae_width),
        "sae_l0": _normalize_sae_l0(sae_l0),
        "activation_type": str(activation_type),
        "ignore_bos_features": bool(ignore_bos_features),
        "percentile_checksum": str(percentile_checksum),
    }


def compute_gradient_overlay_checksum(descriptor: dict) -> str:
    """Deterministic SHA-256 of the canonical descriptor (64 hex chars)."""
    canonical = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode()).hexdigest()


def provenance_dir(packed_row_dir: str | Path) -> Path:
    """The packed-segment provenance sibling dir for a packed-row dir."""
    return Path(packed_row_dir) / PROVENANCE_DIRNAME


def assert_resume_gradient_matches(
    configured_spec: GradientRankingSpec | None,
    checkpoint_meta: dict | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Verify a resumed checkpoint was trained with the SAME canonical
    gradient block (the feature_mask resume-assert twin).

    The canonical dict carries the code-versioned score_reduction /
    split_policy_version constants, which the checkpoint-dir leaf does NOT —
    a version bump re-addresses sel/ data while leaving the checkpoint path
    unchanged, and without this assert a v1-scored lineage would silently
    continue on v2-scored selections. Absent on both sides (legacy / no
    gradient) passes byte-identically.
    """
    configured = (
        canonical_gradient_selection_dict(configured_spec)
        if configured_spec is not None
        else None
    )
    stored = (checkpoint_meta or {}).get("gradient_attribution_canonical")
    if configured != stored:
        raise GradientAttributionError(
            f"resume mismatch at {checkpoint_dir}: the checkpoint records "
            f"gradient_attribution_canonical={stored!r} but the current "
            f"config resolves to {configured!r}. A differing "
            f"score_reduction/split_policy_version means the sel/ data was "
            f"re-scored since this lineage started — continuing would "
            f"silently mix score semantics. Start a fresh run (or restore "
            f"the original code version)."
        )


def assert_descriptor_matches(
    expected: dict, actual: dict, *, artifact: str
) -> None:
    """I4-style guard: a consumed overlay's manifest descriptor must equal the
    descriptor the consumer derived from its own config.

    Raises:
        GradientAttributionError: naming every differing key (all problems in
            one report, mirroring verify_feature_dir's accumulate style).
    """
    keys = sorted(set(expected) | set(actual))
    diffs = [
        f"  - {k}: expected {expected.get(k, '<absent>')!r}, "
        f"found {actual.get(k, '<absent>')!r}"
        for k in keys
        if (k in expected) != (k in actual)
        or (k in expected and k in actual and expected[k] != actual[k])
    ]
    if diffs:
        raise GradientAttributionError(
            f"{artifact} descriptor mismatch (stale or foreign artifact; "
            f"re-run its producer):\n" + "\n".join(diffs)
        )
