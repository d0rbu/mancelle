"""Activation-frequency feature blacklist generator (single source of truth).

Producer for PR 471's ``feature_mask``: reads the precomputed
``feature_statistics.safetensors`` for a feature directory, blacklists every
SAE feature whose per-segment activation frequency lies OUTSIDE the
configured ``(freq_min, freq_max]`` interval, and writes a file PR 471
consumes verbatim via ``feature_mask.path``.

The metric is fraction of POST-POOLING SEGMENTS active (extraction pools
``segment_size`` tokens per segment before SAE encoding) — not fraction of
tokens. See ``docs/reference/configuration.md``.

Bounds semantics (both independently nullable; both null ⇒ feature disabled,
byte-identical legacy behavior preserving PR 471's I1/I2/I5):

* ``freq_min`` (lower bound, INCLUSIVE) — blacklist if
  ``count <= freq_min * n_segments``. Default ``1e-5`` ⇒ kills dead features
  (``count == 0``) and other features too rare for the autointerpretability
  judge to score meaningfully (``count <= 1e-5 * n_segments`` — e.g. up to
  ~1,305 activations on a 130M-segment corpus). Pass ``0.0`` explicitly to
  kill dead features (``count == 0``, absent from ``per_feature_stats``)
  only, matching pre-#702 behavior exactly.
* ``freq_max`` (upper bound, EXCLUSIVE) — blacklist if
  ``count >  freq_max * n_segments``. Default ``0.5``.

Since issue #695 the resolved mask ALSO sizes the encoder's feature-indexed embedding
tables, so these bounds decide parameter count as well as the selection universe. That
does NOT give #695 a vote on the band: the remap costs a single gather, so compaction
pays at ANY band and has no business narrowing the research universe to justify itself.
The band is chosen on SELECTION grounds alone (#702: kill dead features, those too rare
for the autointerp judge to score, and the near-ubiquitous ones).

Equivalently: a feature is KEPT iff ``min < freq <= max`` (open below, closed
above). The exact integer ``count`` and ``n_segments`` are used for the
boundary, never the lossily-rounded ``pct_segments_active``.

Source = the statistics artifact, not a fresh shard scan: ``compute_statistics``
already aggregates per-feature segment-fire counts and persists ``n_segments``,
excluding the held-out validation shard by default. The blacklist's train/val
split is therefore exactly the one ``compute_statistics`` was run with.

No ``torch.distributed`` import: DDP coordination is injected (``barrier_fn``)
so this module is fully unit-testable in a single process.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from beartype import beartype
from beartype.roar import BeartypeCallHintParamViolation
from loguru import logger
from omegaconf import OmegaConf

from errors import FeatureMaskError, ManualFeatureSelectionError
from nanochat.atomic_io import atomic_write_text
from steering.manual_feature_ids import parse_feature_ids_txt
from steering.refined_types import (
    NonNegativeInt,
    NonNegativeIntLike,
    NonNegativeIntTuple,
    NonNegativeIntTupleLike,
    PositiveInt,
    PositiveIntLike,
    UnitIntervalOpenLower,
    UnitIntervalOpenLowerLike,
    UnitIntervalOpenUpper,
    UnitIntervalOpenUpperLike,
)
from steering.stats_io import find_statistics_path, load_statistics
from steering.stats_types import ExtractionStatistics

DEFAULT_FREQ_MIN: float = 1e-5
DEFAULT_FREQ_MAX: float = 0.5
# Artifact format version, embedded in the .txt header + meta.json. Bump in a
# future change that alters the generation logic or file format so an existing
# cached artifact is regenerated (the header content changes ⇒ the
# content-compare rewrite in generate_or_load_blacklist fires). Starts at 1 —
# this is the first version of the on-disk format.
GENERATOR_VERSION = 1
BLACKLIST_SUBDIR = "blacklist"


def canonical_bound(x: float) -> str:
    """Stable string for a bound value. Bit-equal floats ⇒ identical string.

    ``repr(float)`` of any value in ``[0, 1]`` only ever yields characters
    from ``[0-9.e-]`` — all filesystem-safe — so it doubles as a filename
    leaf component.
    """
    return repr(float(x))


def _bound_to_str(x: float | None) -> str:
    """Canonical string for a bound; ``None`` ⇒ ``"null"``. The single source
    of the on-disk bound spelling (used by both filenames and headers)."""
    return "null" if x is None else canonical_bound(x)


def _bound_from_str(value: str) -> float | None:
    """Inverse of ``_bound_to_str`` (``meta.json`` round-trip)."""
    return None if value == "null" else float(value)


@dataclass(frozen=True, init=False)
class FrequencyBounds:
    """Resolved ``(freq_min, freq_max)`` bounds (each independently nullable).

    ``min`` is INCLUSIVE (``count <= min * n_segments`` ⇒ kill); ``max`` is
    EXCLUSIVE (``count > max * n_segments`` ⇒ kill). KEEP iff
    ``min < freq <= max``.
    """

    min: UnitIntervalOpenUpper | None
    max: UnitIntervalOpenLower | None

    @beartype
    def __init__(
        self,
        min: UnitIntervalOpenUpperLike | None,
        max: UnitIntervalOpenLowerLike | None,
    ) -> None:
        normalized_min = None if min is None else UnitIntervalOpenUpper.parse(float(min))
        normalized_max = None if max is None else UnitIntervalOpenLower.parse(float(max))
        if (
            normalized_min is not None
            and normalized_max is not None
            and normalized_min >= normalized_max
        ):
            raise FeatureMaskError(
                f"feature_mask_freq_min ({normalized_min}) must be strictly less than "
                f"feature_mask_freq_max ({normalized_max}); the kept interval "
                f"(min, max] would otherwise be empty"
            )
        object.__setattr__(self, "min", normalized_min)
        object.__setattr__(self, "max", normalized_max)


@beartype
def _normalize_freq_min(
    raw: UnitIntervalOpenUpperLike | None,
) -> float | None:
    return None if raw is None else float(raw)


@beartype
def _normalize_freq_max(
    raw: UnitIntervalOpenLowerLike | None,
) -> float | None:
    return None if raw is None else float(raw)


def _normalize_config_bound(raw, *, name: str, is_min: bool) -> float | None:
    """Apply a refined interval and translate only the config error surface."""
    normalize = _normalize_freq_min if is_min else _normalize_freq_max
    interval = "[0.0, 1.0)" if is_min else "(0.0, 1.0]"
    try:
        return normalize(raw)
    except BeartypeCallHintParamViolation as exc:
        raise FeatureMaskError(
            f"{name} must be a float in {interval} or null, got {type(raw).__name__} {raw!r}"
        ) from exc


def resolve_blacklist_bounds(cfg) -> FrequencyBounds | None:
    """Read+validate ``feature_mask_freq_min`` / ``feature_mask_freq_max``.

    Returns ``None`` when BOTH bounds are explicitly ``null`` (fully disabled
    — caller must leave ``cfg`` untouched to preserve PR 471's I1/I2/I5
    byte-identical no-op). Otherwise returns ``FrequencyBounds(min, max)``
    where each component is independently a float (active bound) or ``None``
    (skip that bound).

    Defaults: ``freq_min = 1e-5`` (kills dead features and other features
    too rare to be autointerp-scoreable), ``freq_max = 0.5`` (kills features
    active in more than half of all segments).

    Raises ``FeatureMaskError`` on any invalid value.
    """
    raw_min = cfg.get("feature_mask_freq_min", DEFAULT_FREQ_MIN)
    raw_max = cfg.get("feature_mask_freq_max", DEFAULT_FREQ_MAX)
    # freq_min: [0, 1)  — 0 valid (kill dead); 1 forbidden (would kill all).
    fmin = _normalize_config_bound(
        raw_min,
        name="feature_mask_freq_min",
        is_min=True,
    )
    # freq_max: (0, 1]  — 1 valid (kill nothing); 0 forbidden (would kill all).
    fmax = _normalize_config_bound(
        raw_max,
        name="feature_mask_freq_max",
        is_min=False,
    )
    if fmin is None and fmax is None:
        return None
    return FrequencyBounds(min=fmin, max=fmax)


@beartype
def blacklist_paths(
    feature_dir: str | Path,
    freq_min: UnitIntervalOpenUpperLike | None,
    freq_max: UnitIntervalOpenLowerLike | None,
) -> tuple[Path, Path]:
    """``(blacklist.txt, meta.json)`` for ``feature_dir`` × bounds.

    Filename = ``freq_min{m}_max{M}.{txt,meta.json}`` where ``m, M`` are
    canonical (round-trip) representations or ``"null"`` if that bound is
    skipped.  Bit-equal bounds yield identical filenames.
    """
    base = Path(feature_dir) / BLACKLIST_SUBDIR
    leaf = f"freq_min{_bound_to_str(freq_min)}_max{_bound_to_str(freq_max)}"
    return base / f"{leaf}.txt", base / f"{leaf}.meta.json"


@dataclass(frozen=True, init=False)
class BlacklistResult:
    """Resolved blacklist + provenance. ``ids`` is canonicalized to a
    sorted-unique tuple at the typed constructor boundary."""

    ids: NonNegativeIntTuple
    n_sae_features: PositiveInt
    n_segments: NonNegativeInt
    n_active_features: NonNegativeInt
    freq_min: UnitIntervalOpenUpper | None
    freq_max: UnitIntervalOpenLower | None
    stats_generated_at: str

    @beartype
    def __init__(
        self,
        ids: NonNegativeIntTupleLike,
        n_sae_features: PositiveIntLike,
        n_segments: NonNegativeIntLike,
        n_active_features: NonNegativeIntLike,
        freq_min: UnitIntervalOpenUpperLike | None,
        freq_max: UnitIntervalOpenLowerLike | None,
        stats_generated_at: str,
    ) -> None:
        object.__setattr__(
            self,
            "ids",
            NonNegativeIntTuple.parse(tuple(sorted(set(ids)))),
        )
        object.__setattr__(
            self,
            "n_sae_features",
            PositiveInt.parse(n_sae_features),
        )
        object.__setattr__(
            self,
            "n_segments",
            NonNegativeInt.parse(n_segments),
        )
        object.__setattr__(
            self,
            "n_active_features",
            NonNegativeInt.parse(n_active_features),
        )
        object.__setattr__(
            self,
            "freq_min",
            None if freq_min is None else UnitIntervalOpenUpper.parse(float(freq_min)),
        )
        object.__setattr__(
            self,
            "freq_max",
            None if freq_max is None else UnitIntervalOpenLower.parse(float(freq_max)),
        )
        object.__setattr__(self, "stats_generated_at", stats_generated_at)

    @property
    def is_empty(self) -> bool:
        return len(self.ids) == 0


@beartype
def compute_activation_frequency_blacklist(
    feature_dir: str | Path,
    n_sae_features: PositiveIntLike,
    *,
    freq_min: UnitIntervalOpenUpperLike | None,
    freq_max: UnitIntervalOpenLowerLike | None,
    _statistics: ExtractionStatistics | None = None,
) -> BlacklistResult:
    """Blacklist features whose per-segment activation frequency falls outside
    ``(freq_min, freq_max]``. KEEP iff ``min < freq <= max``; ``None`` skips
    that bound.

    Uses exact integer ``count`` from ``per_feature_stats`` and exact integer
    ``n_segments`` (never the lossily-rounded ``pct_segments_active``), so
    boundary behavior is exact. Iterates the full ``[0, n_sae_features)`` id
    range so DEAD features (``count == 0``, absent from ``per_feature_stats``)
    can be killed by the lower bound when ``freq_min`` is set (any
    ``freq_min >= 0`` kills dead features since ``0 <= freq_min * n_segments``).

    Pure / RNG-free / deterministic.
    """
    stats_path = find_statistics_path(feature_dir)
    if stats_path is None:
        raise FeatureMaskError(
            f"feature_statistics.safetensors not found in {feature_dir}; run "
            f"feature statistics computation (compute_statistics) first, or "
            f"set feature_mask_freq_min: null AND feature_mask_freq_max: null "
            f"to disable the auto activation-frequency blacklist."
        )
    stats = load_statistics(stats_path) if _statistics is None else _statistics
    stats_width = int(stats.metadata.n_sae_features)
    if stats_width != int(n_sae_features):
        raise FeatureMaskError(
            f"feature_statistics width ({stats_width}) at {stats_path} "
            f"disagrees with the configured n_sae_features "
            f"({int(n_sae_features)}). The statistics were computed for a "
            f"DIFFERENT SAE. Point feature_dir at the statistics for THIS SAE, or "
            f"correct sae.width — those are the only two fixes.\n"
            f"Do NOT null feature_mask_freq_min/feature_mask_freq_max to get past this. "
            f"Unlike the missing-statistics case above, the statistics here exist and are "
            f"simply the wrong ones: nulling the band would not read them, it would drop "
            f"the mask entirely, so the encoder is built full-width (#695), the checkpoint "
            f"lands in a directory with no mask leaf, and the run silently stops being the "
            f"one you configured."
        )
    n_segments = int(stats.features.n_segments)

    # Dense count vector — 0 for features absent from per_feature_stats (dead).
    counts = [0] * int(n_sae_features)
    for fid_str, pfs in stats.per_feature_stats.items():
        fid = int(fid_str)
        if fid < 0 or fid >= n_sae_features:
            raise FeatureMaskError(
                f"feature id {fid} in {stats_path} is out of range "
                f"[0, {n_sae_features}) (feature_statistics width "
                f"{stats_width})."
            )
        counts[fid] = int(pfs.count)

    if n_segments <= 0:
        # Degenerate corpus: frequency undefined; blacklist nothing.
        ids: tuple[int, ...] = ()
    else:
        lo_cut = None if freq_min is None else float(freq_min) * float(n_segments)
        hi_cut = None if freq_max is None else float(freq_max) * float(n_segments)
        ids_list: list[int] = []
        for fid in range(int(n_sae_features)):
            c = float(counts[fid])
            kill_lo = lo_cut is not None and c <= lo_cut
            kill_hi = hi_cut is not None and c > hi_cut
            if kill_lo or kill_hi:
                ids_list.append(fid)
        ids = tuple(ids_list)
    return BlacklistResult(
        ids=ids,
        n_sae_features=int(n_sae_features),
        n_segments=n_segments,
        n_active_features=len(stats.per_feature_stats),
        freq_min=freq_min,
        freq_max=freq_max,
        stats_generated_at=str(stats.metadata.generated_at),
    )


_TXT_HEADER = (
    "# activation-frequency blacklist (generator v{v}, source: "
    "feature_statistics.safetensors)\n"
    "# freq_min: {fmin}  (blacklist iff count <= freq_min * n_segments)\n"
    "# freq_max: {fmax}  (blacklist iff count >  freq_max * n_segments)\n"
    "# n_sae_features: {n}\n"
    "# n_segments: {seg}\n"
    "# n_active_features: {act}\n"
    "# stats_generated_at: {sga}\n"
    "# n_blacklisted: {nb}\n"
)


def _render_txt(result: BlacklistResult) -> str:
    """The deterministic ``blacklist.txt`` content — used both for writing
    and for the rewrite-skip comparison key. No timestamp ⇒ golden-able."""
    header = _TXT_HEADER.format(
        v=GENERATOR_VERSION,
        fmin=_bound_to_str(result.freq_min),
        fmax=_bound_to_str(result.freq_max),
        n=result.n_sae_features,
        seg=result.n_segments,
        act=result.n_active_features,
        sga=result.stats_generated_at,
        nb=len(result.ids),
    )
    body = "".join(f"{i}\n" for i in result.ids)
    return header + body


def _parse_ids_file(path: str | Path) -> list[int]:
    """Parse persisted IDs with the same canonical parser as feature masks."""
    return parse_feature_ids_txt(path)


def _metadata_payload(result: BlacklistResult) -> dict[str, object]:
    """Deterministic ``meta.json`` fields for freshness checks and writes."""
    return {
        "generator_version": GENERATOR_VERSION,
        "source": "feature_statistics.safetensors",
        "freq_min": _bound_to_str(result.freq_min),
        "freq_max": _bound_to_str(result.freq_max),
        "n_sae_features": result.n_sae_features,
        "n_segments": result.n_segments,
        "n_active_features": result.n_active_features,
        "stats_generated_at": result.stats_generated_at,
        "n_blacklisted": len(result.ids),
        "empty": result.is_empty,
    }


def _metadata_matches(meta_path: str | Path, result: BlacklistResult) -> bool:
    """Whether persisted metadata exactly matches every deterministic field."""
    try:
        meta = json.loads(Path(meta_path).read_text())
    except (ValueError, OSError, TypeError):
        return False
    if type(meta) is not dict or type(meta.get("generated_at")) is not str:
        return False
    expected = _metadata_payload(result)
    return all(
        key in meta
        and type(meta[key]) is type(expected_value)
        and meta[key] == expected_value
        for key, expected_value in expected.items()
    )


def write_blacklist(txt_path: str | Path, meta_path: str | Path, result: BlacklistResult) -> None:
    """Atomically write ``blacklist.txt`` (+ ``meta.json``). RANK-0 ONLY."""
    txt_path = Path(txt_path)
    meta_path = Path(meta_path)
    atomic_write_text(_render_txt(result), txt_path)
    meta = _metadata_payload(result)
    meta["generated_at"] = datetime.now(UTC).isoformat()
    atomic_write_text(json.dumps(meta, sort_keys=True, indent=2), meta_path)


def _load_blacklist(txt_path: Path, meta_path: Path) -> BlacklistResult:
    """Reconstruct a present persisted pair, propagating the actual parse cause."""
    meta = json.loads(meta_path.read_text())
    return BlacklistResult(
        ids=tuple(sorted(_parse_ids_file(txt_path))),
        n_sae_features=int(meta["n_sae_features"]),
        n_segments=int(meta["n_segments"]),
        n_active_features=int(meta["n_active_features"]),
        freq_min=_bound_from_str(meta["freq_min"]),
        freq_max=_bound_from_str(meta["freq_max"]),
        stats_generated_at=str(meta["stats_generated_at"]),
    )


@beartype
def generate_or_load_blacklist(
    feature_dir: str | Path,
    n_sae_features: PositiveIntLike,
    *,
    freq_min: UnitIntervalOpenUpperLike | None,
    freq_max: UnitIntervalOpenLowerLike | None,
    ddp_rank: NonNegativeIntLike = 0,
    barrier_fn: Callable[[], None] | None = None,
    force: bool = False,
    _statistics: ExtractionStatistics | None = None,
) -> tuple[Path, BlacklistResult] | None:
    """DDP-safe generate-if-stale. Returns ``(txt_path, result)`` or ``None``
    when the computed blacklist is empty.

    Rank 0 reads statistics + (re)writes the artifact iff the file/meta is
    missing, ``force``, or either the freshly-rendered deterministic ``.txt``
    content or the deterministic metadata projection differs from disk (so
    stats recompute / bound change / generator-version bump / a torn write
    auto-corrects; identical re-runs never rewrite).
    All ranks then read the (atomically written) artifact so they agree on
    the exact bytes. Rank 0's work is wrapped so the barrier is reached on
    every rank even if it raises.
    """
    txt_path, meta_path = blacklist_paths(feature_dir, freq_min, freq_max)
    rank0_error: BaseException | None = None
    if ddp_rank == 0:
        try:
            if _statistics is None:
                result = compute_activation_frequency_blacklist(
                    feature_dir,
                    n_sae_features,
                    freq_min=freq_min,
                    freq_max=freq_max,
                )
            else:
                result = compute_activation_frequency_blacklist(
                    feature_dir,
                    n_sae_features,
                    freq_min=freq_min,
                    freq_max=freq_max,
                    _statistics=_statistics,
                )
            intended = _render_txt(result)
            current = txt_path.read_text() if txt_path.is_file() else None
            metadata_matches = _metadata_matches(meta_path, result)
            if force or current != intended or not metadata_matches:
                write_blacklist(txt_path, meta_path, result)
        except BaseException as exc:  # noqa: BLE001 — must release the barrier
            rank0_error = exc
    if barrier_fn is not None:
        barrier_fn()
    if rank0_error is not None:
        raise rank0_error
    if not (txt_path.is_file() and meta_path.is_file()):
        raise FeatureMaskError(
            f"freq-blacklist artifact missing after generation at {txt_path} "
            f"(ddp_rank={ddp_rank}); rank 0 likely failed to write it (check "
            f"rank-0 logs for the real error), a non-zero rank read before "
            f"rank-0 wrote (missing barrier), or the cache dir is on a "
            f"rank-divergent filesystem."
        )
    try:
        result = _load_blacklist(txt_path, meta_path)
    except (
        ValueError,
        OSError,
        KeyError,
        TypeError,
        ManualFeatureSelectionError,
    ) as exc:
        raise FeatureMaskError(
            f"freq-blacklist artifact is corrupt after generation at "
            f"{txt_path} / {meta_path} (ddp_rank={ddp_rank}): {exc}"
        ) from exc
    if result.is_empty:
        return None
    return txt_path, result


class FrequencyBlacklistStatus(StrEnum):
    DISABLED = "disabled"
    SKIPPED_USER_MASK = "skipped_user_mask"
    SKIPPED_NO_FEATURE_DIR = "skipped_no_feature_dir"
    EMPTY_NOOP = "empty_noop"
    APPLIED = "applied"


@dataclass(frozen=True, init=False)
class FrequencyBlacklistOutcome:
    status: FrequencyBlacklistStatus
    path: str | None = None
    n_blacklisted: NonNegativeInt = NonNegativeInt.parse(0)

    @beartype
    def __init__(
        self,
        status: FrequencyBlacklistStatus,
        path: str | None = None,
        n_blacklisted: NonNegativeIntLike = 0,
    ) -> None:
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "path", path)
        object.__setattr__(
            self,
            "n_blacklisted",
            NonNegativeInt.parse(n_blacklisted),
        )


@beartype
def maybe_apply_frequency_blacklist(
    cfg,
    feature_dir: str | None,
    n_sae_features: PositiveIntLike,
    *,
    ddp_rank: NonNegativeIntLike = 0,
    barrier_fn: Callable[[], None] | None = None,
) -> FrequencyBlacklistOutcome:
    """Single training entry point. Mutates ``cfg.feature_mask`` to
    ``{mode: blacklist, path: <generated>}`` only when a non-empty blacklist
    is produced AND no explicit ``feature_mask`` is configured; otherwise
    leaves ``cfg`` byte-identically untouched (preserving PR 471 I1/I2/I5)."""
    bounds = resolve_blacklist_bounds(cfg)
    if bounds is None:
        logger.info(
            "freq-blacklist: disabled (both feature_mask_freq_min and "
            "feature_mask_freq_max are null)"
        )
        return FrequencyBlacklistOutcome(FrequencyBlacklistStatus.DISABLED)
    if cfg.get("feature_mask") is not None:
        logger.warning(
            "freq-blacklist: feature_mask_freq_min/max ignored — an explicit "
            "feature_mask is configured (user mask wins)."
        )
        return FrequencyBlacklistOutcome(FrequencyBlacklistStatus.SKIPPED_USER_MASK)
    if feature_dir is None:
        logger.warning(
            "freq-blacklist: feature_mask_freq_min/max set but feature_dir is "
            "None — skipping the auto activation-frequency blacklist. Set "
            "feature_dir, or set both bounds to null, to silence this warning."
        )
        return FrequencyBlacklistOutcome(FrequencyBlacklistStatus.SKIPPED_NO_FEATURE_DIR)
    out = generate_or_load_blacklist(
        feature_dir,
        n_sae_features,
        freq_min=bounds.min,
        freq_max=bounds.max,
        ddp_rank=ddp_rank,
        barrier_fn=barrier_fn,
    )
    if out is None:
        logger.info(
            "freq-blacklist: bounds (min={}, max={}) produced an EMPTY "
            "blacklist — no-op (feature_mask left unset; byte-identical to "
            "baseline).",
            bounds.min,
            bounds.max,
        )
        return FrequencyBlacklistOutcome(FrequencyBlacklistStatus.EMPTY_NOOP)
    txt, result = out
    txt_str = str(txt)
    if "${" in txt_str:
        raise FeatureMaskError(
            f"generated blacklist path contains '${{' which OmegaConf would "
            f"misinterpret as an interpolation: {txt_str!r}. Move feature_dir "
            f"to a path without '${{'."
        )
    OmegaConf.update(
        cfg,
        "feature_mask",
        {"mode": "blacklist", "path": txt_str},
        merge=False,
        force_add=True,
    )
    logger.info(
        "freq-blacklist: APPLIED — {} features (out of {}) blacklisted with "
        "min={}, max={} over {} segments. Path: {}",
        len(result.ids),
        result.n_sae_features,
        bounds.min,
        bounds.max,
        result.n_segments,
        txt_str,
    )
    return FrequencyBlacklistOutcome(
        FrequencyBlacklistStatus.APPLIED,
        path=txt_str,
        n_blacklisted=len(result.ids),
    )
