"""Feature blacklist/whitelist primitive (the single source of truth).

A FeatureMask is resolved once from config and threaded through the
selection pipeline. It is enforced at every site that chooses positive,
negative, or feature-prediction-target features so a masked feature can
never appear as a conditioning signal.

The whole module is a strict no-op when no mask is configured (callers
pass / receive None), preserving bit-identical behavior for legacy runs.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar, cast

import torch as t
from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from omegaconf import OmegaConf

from errors import FeatureMaskError, ManualFeatureSelectionError
from steering.manual_feature_ids import (
    parse_feature_ids_txt,
    parse_manual_feature_yaml_ids,
)
from steering.refined_types import (
    NonEmptyNonNegativeIntTuple,
    NonEmptyNonNegativeIntTupleLike,
    PositiveIntLike,
    UniqueNonEmptyNonNegativeIntListLike,
)


class FeatureMaskMode(StrEnum):
    blacklist = "blacklist"
    whitelist = "whitelist"


class _MaskMemo:
    """Per-mask cache of derived constant tensors (issue #740 perf).

    Every entry is a PURE function of the frozen ``(mode, ids)`` [+ ``n`` /
    ``device``], so caching is byte-identical — the same values, materialized
    once instead of on every one of the 45k+ per-segment ``feature_mask`` calls
    in the splice preselection hot loop (where ``t.tensor(self.ids)`` +
    full-width membership scatter dominated the profile). Held on the mask via
    ``object.__setattr__`` (NOT a dataclass field) so it is invisible to
    ``==``/hash/repr AND ``dataclasses.asdict``, and is rebuilt lazily after
    unpickle — pickling (e.g. multiprocessing) ships only ``(mode, ids)``.
    """

    __slots__ = (
        "allowed_by_n",
        "allowed_ids_by_n",
        "ids_by_device",
        "member_by_device",
        "membership_by_n",
    )

    def __init__(self) -> None:
        self.allowed_by_n: dict[int, t.Tensor] = {}
        self.allowed_ids_by_n: dict[int, t.Tensor] = {}
        self.ids_by_device: dict[str, t.Tensor] = {}
        self.member_by_device: dict[str, t.Tensor] = {}
        self.membership_by_n: dict[int, t.Tensor] = {}


@dataclass(frozen=True, init=False)
class FeatureMask:
    """A resolved, validated, canonical feature mask.

    ``ids`` is stored sorted + de-duplicated so two logically-identical
    masks (any source order/format) are ``==`` and hash identically.
    """

    mode: FeatureMaskMode
    ids: NonEmptyNonNegativeIntTuple
    # Derived-tensor memo + precomputed max id (issue #740 perf). Declared
    # ClassVar so @dataclass does NOT treat them as fields (excluded from the
    # constructor, ``==``/hash/repr, and ``dataclasses.asdict``/``fields()``);
    # each instance gets its OWN value via ``object.__setattr__`` below.
    _memo: ClassVar[_MaskMemo]
    _max_id: ClassVar[int]

    @beartype
    def __init__(
        self,
        mode: FeatureMaskMode | str,
        ids: NonEmptyNonNegativeIntTupleLike,
    ) -> None:
        try:
            normalized_mode = FeatureMaskMode(mode)
        except (ValueError, KeyError, TypeError) as exc:
            raise FeatureMaskError(
                f"feature_mask.mode must be one of {list(FeatureMaskMode)}, got {mode!r}"
            ) from exc
        canon = NonEmptyNonNegativeIntTuple.parse(tuple(sorted(set(ids))))
        object.__setattr__(self, "mode", normalized_mode)
        object.__setattr__(self, "ids", canon)
        # canon is sorted + non-empty ⇒ canon[-1] is the max id, as a plain int:
        # no tensor / no ``.item()`` sync, and available BEFORE any cache entry,
        # which keeps the OOB guard allocation-free and the cold cache race-free.
        object.__setattr__(self, "_max_id", canon[-1])
        object.__setattr__(self, "_memo", _MaskMemo())

    def __getstate__(self) -> dict[str, object]:
        # Ship only the identity fields; _max_id + the derived-tensor memo are
        # rebuilt by __setstate__ so pickling (e.g. multiprocessing) never
        # serializes cached tensors.
        return {"mode": self.mode, "ids": self.ids}

    def __setstate__(self, state: dict[str, object]) -> None:
        object.__setattr__(self, "mode", state["mode"])
        object.__setattr__(self, "ids", state["ids"])
        object.__setattr__(self, "_max_id", self.ids[-1])
        object.__setattr__(self, "_memo", _MaskMemo())

    def _cpu_ids(self) -> t.Tensor:
        """Cached CPU int64 ``ids`` tensor (used by the membership path). Read-only."""
        memo = self._memo
        ids = memo.ids_by_device.get("cpu")
        if ids is None:
            ids = t.tensor(self.ids, dtype=t.int64)
            memo.ids_by_device["cpu"] = ids
        return ids

    def _ids_member(self, device: t.device) -> tuple[int, t.Tensor]:
        """Cached ``(max_id, member bool[max_id+1])`` for ``device``.

        Matches the pre-memo ``keep_mask`` prelude exactly (same per-device
        allocation), just built once. ``member`` is published LAST so any reader
        that sees it also sees ``ids``; ``_max_id`` is precomputed so no reader
        can observe a half-initialised cache. Read-only — never mutate these.
        """
        memo = self._memo
        key = str(device)
        member = memo.member_by_device.get(key)
        if member is not None:
            return self._max_id, member
        ids = t.tensor(self.ids, dtype=t.int64, device=device)
        member = t.zeros(self._max_id + 1, dtype=t.bool, device=device)
        member[ids] = True
        memo.ids_by_device[key] = ids
        memo.member_by_device[key] = member  # publish last
        return self._max_id, member

    def _membership_cached(self, n_sae_features: int) -> t.Tensor:
        """Cached ``bool[n_sae_features]`` membership. Read-only.

        Byte-identical to the pre-memo per-call membership construction with
        the pre-memo failure modes preserved:
        - the public refined-type boundary rejects non-int ``n`` before this
          cache lookup (so ``True`` or ``1.0`` cannot alias a cached int key);
        - raise the SAME OOB ``FeatureMaskError`` BEFORE any allocation via the
          precomputed ``_max_id`` (pre-memo allocated ``member`` first, so a
          corrupt/huge id could OOM before the guard fired);
        - cache ONLY the CPU default-device path (the producer/hot path); under
          any other ambient ``torch`` default device reproduce the pre-memo
          build fresh so the result tracks the ambient device (a CPU-cached
          tensor is never returned for a CUDA-ambient call). meta/fake ambient
          devices are out of contract.
        """
        if self._max_id >= n_sae_features:
            raise FeatureMaskError(
                f"feature_mask id {self._max_id} >= n_sae_features {n_sae_features}"
            )
        if t.get_default_device().type != "cpu":
            # Non-CPU ambient default device: reproduce the pre-memo build fresh
            # on that device (no alloc-free probe cost on the cached CPU path).
            mem = t.zeros(n_sae_features, dtype=t.bool)
            mem[t.tensor(self.ids, dtype=t.int64)] = True
            return mem
        memo = self._memo
        mem = memo.membership_by_n.get(n_sae_features)
        if mem is None:
            mem = t.zeros(n_sae_features, dtype=t.bool)
            mem[self._cpu_ids()] = True
            memo.membership_by_n[n_sae_features] = mem
        return mem

    def _allowed_universe_cached(self, n_sae_features: int) -> t.Tensor:
        """Cached allowed-universe tensor for internal read-only consumers.

        Public callers receive a defensive copy from
        :meth:`allowed_universe_mask`. The splice-selection hot path only reads
        this tensor, so retaining the derived complement avoids allocating and
        filling a full SAE-width tensor once per segment. As with
        ``_membership_cached``, only the CPU-default path is memoized; other
        ambient default devices are rebuilt so device semantics cannot drift.
        """
        member = self._membership_cached(n_sae_features)
        if t.get_default_device().type != "cpu":
            return ~member if self.mode is FeatureMaskMode.blacklist else member
        memo = self._memo
        allowed = memo.allowed_by_n.get(n_sae_features)
        if allowed is None:
            allowed = ~member if self.mode is FeatureMaskMode.blacklist else member
            memo.allowed_by_n[n_sae_features] = allowed
        return allowed

    def _allowed_feature_ids_cached(self, n_sae_features: int) -> t.Tensor:
        """Sorted allowed IDs for internal read-only CPU sampling paths."""
        allowed = self._allowed_universe_cached(n_sae_features)
        if t.get_default_device().type != "cpu":
            return t.nonzero(allowed, as_tuple=False).squeeze(1)
        memo = self._memo
        ids = memo.allowed_ids_by_n.get(n_sae_features)
        if ids is None:
            ids = t.nonzero(allowed, as_tuple=False).squeeze(1)
            memo.allowed_ids_by_n[n_sae_features] = ids
        return ids

    def canonical_dict(self) -> dict:
        """Canonical hash/metadata form."""
        # Emit .value (plain str, not a FeatureMaskMode instance) so JSON
        # serialization and compute_selection_hash are byte-identical to pre-H0a
        # (I1 frozen golden preserved).
        return {"mode": self.mode.value, "ids": list(self.ids)}

    def dirname(self) -> str:
        """Checkpoint-folder leaf component: ``{bl|wl}_{16hex}``.

        16 hex digits = 64 bits of SHA-256 prefix.  Birthday collision is
        ~50% at ~5e9 distinct masks — negligible at any realistic scale,
        eliminating the silent-clobber risk present with 8-hex (32-bit).

        The hash source (canonical_dict JSON) is unchanged; only the
        truncation length widens from 8 → 16.  This is a separate concern
        from compute_selection_hash / cache-identity hashes (goldens I1/I2
        unaffected).
        """
        payload = json.dumps(
            self.canonical_dict(), sort_keys=True, separators=(",", ":")
        )
        h = hashlib.sha256(payload.encode()).hexdigest()[:16]
        prefix = "bl" if self.mode is FeatureMaskMode.blacklist else "wl"
        return f"{prefix}_{h}"

    # ``FeatureMask`` is stored as a plain Python attribute on the compiled
    # encoder-decoder model.  Keep this small object boundary eager: beartype's
    # generated wrapper and Python memo are intentionally not
    # TorchDynamo-traceable. The surrounding tensor graph resumes after this
    # eager boundary. Eager callers still receive the full refined-type check.
    @t.compiler.disable
    @beartype
    def allowed_universe_mask(self, n_sae_features: PositiveIntLike) -> t.Tensor:
        """bool[n_sae_features] — True where a feature MAY be a negative.

        blacklist ⇒ everything except masked; whitelist ⇒ only whitelisted.
        """
        # Always return a fresh tensor: a caller that mutates the result in
        # place must not corrupt either the membership cache or the internal
        # read-only allowed-universe cache.
        return self._allowed_universe_cached(n_sae_features).clone()

    def keep_mask(self, feature_ids: t.Tensor) -> t.Tensor:
        """bool[len(feature_ids)] — True for ids that survive the mask."""
        if feature_ids.numel() == 0:
            return t.zeros(0, dtype=t.bool, device=feature_ids.device)
        # Cached constant prelude (ids tensor + full-width membership); the
        # returned ``is_member`` below is freshly allocated per call.
        max_id, member = self._ids_member(feature_ids.device)
        fids_long = feature_ids.long()
        # Negative or OOB ids cannot be a member of self.ids (validated non-negative
        # ints); treat them deterministically as not-a-member without indexing member
        # with a negative/OOB value (which would wrap or raise).
        safe = (fids_long >= 0) & (fids_long <= max_id)
        is_member = t.zeros(fids_long.shape, dtype=t.bool, device=feature_ids.device)
        is_member[safe] = member[fids_long[safe]]
        return ~is_member if self.mode is FeatureMaskMode.blacklist else is_member

    def apply_to_positive(
        self, feature_ids: t.Tensor, *parallel: t.Tensor
    ) -> tuple[t.Tensor, ...]:
        """Filter a segment's feature_ids (+ co-aligned parallel arrays).

        blacklist ⇒ drop masked ids; whitelist ⇒ keep only whitelisted ids.
        Pure: callers must only invoke this when the mask is set (I5 —
        never call in the no-mask path).
        """
        keep = self.keep_mask(feature_ids)
        return (feature_ids[keep], *(p[keep] for p in parallel))

    def assert_no_masked(self, feature_ids: t.Tensor) -> None:
        """Runtime invariant (I4): no masked id present (blacklist) /
        only whitelisted ids present (whitelist).

        Raises FeatureMaskError unconditionally on violation so the guard
        cannot be stripped by ``python -O`` (unlike a bare ``assert``).

        Negative feature ids are definitionally invalid (real SAE feature ids
        are always >= 0); any negative id triggers an immediate loud raise.
        """
        if feature_ids.numel() == 0:
            return
        if bool((feature_ids < 0).any()):
            neg_val = int(feature_ids[feature_ids < 0][0].item())
            raise FeatureMaskError(
                f"feature_mask guard received a negative feature id ({neg_val}); "
                "SAE feature ids are always >= 0 — a negative value indicates "
                "a sentinel or uninitialised tensor that should have been filtered "
                "before reaching this guard"
            )
        if not bool(self.keep_mask(feature_ids).all()):
            raise FeatureMaskError(
                f"feature_mask violated: mode={self.mode} produced disallowed ids"
            )


def canonical_dict(mask: FeatureMask | None) -> dict | None:
    """Canonical hash/metadata form. ``None`` ⇒ ``None`` (preserves I1)."""
    return None if mask is None else mask.canonical_dict()


def assert_resume_feature_mask_matches(
    configured_mask: FeatureMask | None,
    checkpoint_meta: dict | None,
    *,
    checkpoint_dir: str,
) -> None:
    """Verify a resumed checkpoint was written with ``configured_mask``.

    No configured mask means there is nothing to check. A configured mask
    requires the checkpoint to carry the resolved canonical metadata written
    by the current training scripts; raw config is not re-resolved at resume
    time.
    """
    if configured_mask is None:
        return

    configured_canonical = canonical_dict(configured_mask)
    if checkpoint_meta is None or "feature_mask_canonical" not in checkpoint_meta:
        raise FeatureMaskError(
            f"feature_mask resume from {checkpoint_dir} requires checkpoint "
            "metadata to include feature_mask_canonical. Start fresh or resume "
            "from a checkpoint written by the current masked-training path."
        )

    checkpoint_canonical = checkpoint_meta["feature_mask_canonical"]
    if checkpoint_canonical != configured_canonical:
        raise FeatureMaskError(
            f"feature_mask mismatch on resume from {checkpoint_dir}: "
            f"checkpoint records {checkpoint_canonical!r}, "
            f"configured {configured_canonical!r}. "
            "Resuming with a different mask than the original run would "
            "silently corrupt the checkpoint. "
            "Use the same feature_mask as the original run, or start fresh."
        )


@beartype
def _canonicalize_ids(
    raw: UniqueNonEmptyNonNegativeIntListLike,
) -> tuple[int, ...]:
    """Convert a validated feature-ID list to its canonical tuple form."""
    return tuple(sorted(raw))


def _canonicalize_config_ids(raw: list) -> tuple[int, ...]:
    """Canonicalize config IDs while preserving the public config error type.

    The refined collection type owns strict-int, non-negative, non-empty, and
    uniqueness validation. This adapter only translates its beartype violation
    into the stable :class:`FeatureMaskError` expected at config/file ingress.
    The SAE-width relationship remains in :func:`resolve_feature_mask`.
    """
    try:
        return _canonicalize_ids(raw)
    except BeartypeCallHintParamViolation as exc:
        raise FeatureMaskError(
            "feature_mask ids must be a non-empty unique list of strict non-negative ints"
        ) from exc


def _coerce_block(block: Mapping[str, Any] | None) -> dict | None:
    """Normalize a mapping config block to a plain dict or ``None``."""
    if block is None:
        return None
    if not isinstance(block, Mapping):
        raise TypeError(
            f"feature mask config must be a mapping, got {type(block).__name__}"
        )
    fm = block.get("feature_mask")
    if fm is None:
        return None
    if OmegaConf.is_config(fm):
        fm = OmegaConf.to_container(fm, resolve=True)
    if not isinstance(fm, dict):
        raise FeatureMaskError(
            f"feature_mask must be a mapping, got {type(fm).__name__}"
        )
    return fm


def _parse_ids_file(path: str) -> list[int]:
    p = Path(path)
    if not p.is_file():
        raise FeatureMaskError(f"feature_mask.path not found: {path}")
    try:
        if p.suffix.lower() in (".yaml", ".yml"):
            return parse_manual_feature_yaml_ids(p)
        return parse_feature_ids_txt(p)
    except ManualFeatureSelectionError as exc:
        raise FeatureMaskError(
            f"feature_mask file {path} is invalid: {exc}"
        ) from exc


@beartype
def resolve_feature_mask(
    cfg_block: Mapping[str, Any] | None,
    n_sae_features: PositiveIntLike,
) -> FeatureMask | None:
    """Resolve + validate a feature_mask block. ``None`` ⇒ disabled.

    Raises FeatureMaskError on any invalid config (fail fast).
    """
    fm = _coerce_block(cfg_block)
    if fm is None:
        return None

    mode = fm.get("mode")
    has_ids = fm.get("ids") is not None
    has_path = fm.get("path") is not None
    if has_ids == has_path:
        raise FeatureMaskError(
            "feature_mask requires exactly one of 'ids' or 'path'"
        )

    raw = list(fm["ids"]) if has_ids else _parse_ids_file(str(fm["path"]))
    ids = _canonicalize_config_ids(raw)

    for i in ids:
        if i < 0 or i >= n_sae_features:
            raise FeatureMaskError(
                f"feature_mask id {i} out of range [0, {n_sae_features})"
            )
    try:
        mode_enum = FeatureMaskMode(str(mode))
    except (ValueError, KeyError, TypeError) as exc:
        raise FeatureMaskError(
            f"feature_mask.mode must be one of {list(FeatureMaskMode)}, got {mode!r}"
        ) from exc
    return FeatureMask(mode=mode_enum, ids=ids)


def feature_mask_dirname(mask: FeatureMask | None) -> str:
    """Checkpoint-folder leaf component. ``None`` ⇒ ``""`` (preserves I2:
    callers append only when non-empty).

    Enabled ⇒ ``{bl|wl}_{16hex}`` where the hex is a 64-bit prefix of the
    sha256 of the canonical mask JSON (mode included, ids sorted+unique).
    64 bits eliminates the silent-clobber birthday-collision risk present
    with the former 32-bit (8-hex) prefix.
    """
    return "" if mask is None else mask.dirname()


def append_mask_leaf(path: str, mask: FeatureMask | None) -> str:
    """os.path.join(path, feature_mask_dirname(mask)) — but ONLY when a mask
    is set. mask=None ⇒ returns ``path`` unchanged (preserves I2: legacy
    checkpoint paths are byte-identical)."""
    leaf = feature_mask_dirname(mask)
    return os.path.join(path, leaf) if leaf else path


def feature_mask_leaf_from_cfg(cfg_block: Mapping[str, Any] | None) -> str:
    """Compute feature_mask_dirname from a raw cfg block without the
    n_sae_features upper-bound validation. Used by orchestrators that don't have
    n_sae_features available (e.g. run_encdec_sft_pipeline.py) — the dirname
    hash depends only on mode + sorted ids, not on n_sae_features. Negative IDs
    remain invalid even without that upper bound.

    Returns ``""`` when no mask is configured (preserves I2).
    """
    # Canonicalization (sorted+unique ids) is guaranteed by FeatureMask.__post_init__, not by any pre-sort here — so feature_mask_leaf_from_cfg(cfg) == feature_mask_dirname(resolve_feature_mask(cfg, n)) for every VALID cfg (resolve_feature_mask additionally rejects dup/out-of-range loudly). See test_leaf_from_cfg_matches_resolve_then_dirname.
    fm = _coerce_block(cfg_block)
    if fm is None:
        return ""
    mode = fm.get("mode")
    # Validity is enforced by FeatureMask.__post_init__ (raises FeatureMaskError);
    # no separate _VALID_MODES check needed.
    has_ids = fm.get("ids") is not None
    has_path = fm.get("path") is not None
    if has_ids == has_path:
        raise FeatureMaskError(
            "feature_mask requires exactly one of 'ids' or 'path'"
        )
    raw = list(fm["ids"]) if has_ids else _parse_ids_file(str(fm["path"]))
    ids = _canonicalize_config_ids(raw)
    # Construct FeatureMask bypassing the n_sae_features range check
    try:
        mode_enum = FeatureMaskMode(str(mode))
    except (ValueError, KeyError, TypeError) as exc:
        raise FeatureMaskError(
            f"feature_mask.mode must be one of {list(FeatureMaskMode)}, got {mode!r}"
        ) from exc
    mask = FeatureMask(mode=mode_enum, ids=ids)
    return feature_mask_dirname(mask)


def feature_mask_from_checkpoint_meta(
    meta: Mapping[str, Any] | None,
) -> FeatureMask | None:
    """Rebuild the canonical feature mask stored in checkpoint metadata.

    Absent metadata, a missing ``feature_mask_canonical`` key, or a ``None``
    value means that no mask was used.  A present value must be a mapping with
    exactly ``mode`` and ``ids``; ``ids`` must be a list of strict,
    non-negative ints in strictly increasing canonical order.  Every malformed
    value raises :class:`FeatureMaskError` instead of being re-canonicalized.
    """
    if meta is None or "feature_mask_canonical" not in meta:
        return None
    raw = meta["feature_mask_canonical"]
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        raise FeatureMaskError("feature_mask_canonical must be a mapping with exactly mode and ids")
    expected_keys = {"mode", "ids"}
    keys = set(raw)
    if keys != expected_keys:
        raise FeatureMaskError(
            "feature_mask_canonical must contain exactly mode and ids; "
            f"missing={sorted(expected_keys - keys)}, "
            f"extra={sorted(keys - expected_keys, key=str)}"
        )
    ids = raw["ids"]
    if not isinstance(ids, list):
        raise FeatureMaskError("feature_mask_canonical.ids must be a list")
    if any(type(feature_id) is not int for feature_id in ids):
        raise FeatureMaskError("feature_mask_canonical.ids must contain only strict ints")
    strict_ids = cast(list[int], ids)
    if any(feature_id < 0 for feature_id in strict_ids):
        raise FeatureMaskError("feature_mask_canonical.ids must contain only non-negative ids")
    if any(left >= right for left, right in itertools.pairwise(strict_ids)):
        raise FeatureMaskError(
            "feature_mask_canonical.ids must be in strictly increasing canonical order"
        )
    try:
        mode = FeatureMaskMode(raw["mode"])
    except (ValueError, KeyError, TypeError):
        raise FeatureMaskError(
            "feature_mask_canonical.mode must be one of "
            f"{list(FeatureMaskMode)}, got {raw['mode']!r}"
        ) from None
    try:
        return FeatureMask(mode=mode, ids=tuple(strict_ids))
    except BeartypeCallHintParamViolation as exc:
        raise FeatureMaskError("feature_mask resolved to an empty set") from exc
