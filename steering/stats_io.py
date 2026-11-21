"""I/O utilities for feature statistics files (safetensors format).

Write path:
    - ``save_statistics`` writes ``.safetensors`` with per-feature data stored
      as dense tensors instead of nested dicts, for ~10x faster load times.

Read path:
    - ``load_statistics`` loads ``.safetensors`` and returns an
      ``ExtractionStatistics`` dataclass — the same type ``compute_statistics``
      produces — so all code uses typed attribute access.
"""

from __future__ import annotations

import hashlib
import json
import struct
from collections.abc import Mapping
from pathlib import Path

import torch as t
from loguru import logger
from safetensors.torch import load as _safetensors_load_bytes
from safetensors.torch import load_file

from errors import DataLoadError
from nanochat.atomic_io import atomic_safetensors_save
from steering.sae_resolver import L0Tier
from steering.safetensors_io import read_safetensors_metadata_from_bytes
from steering.segmentation import SegmentMode
from steering.stats_types import (
    CorpusStats,
    ExtractionStatistics,
    Extrapolation,
    MetadataStats,
    PerFeatureStat,
    SegmentFeatureStats,
    StorageStats,
)

SAFETENSORS_EXT = ".safetensors"
STATS_STEM = "feature_statistics"
_TENSOR_TOLIST = t.Tensor.tolist


def _parse_l0_tier(raw: str | None) -> L0Tier | None:
    """Parse the ``l0_tier`` safetensors-metadata value into an ``L0Tier``.

    Returns ``None`` for absent (v1 stats files). Raises ``ValueError`` with
    the valid tiers on a present-but-unrecognized value, mirroring
    ``steering.feature_io.read_l0_tier_from_shard``'s contract.
    """
    if raw is None:
        return None
    try:
        return L0Tier(raw)
    except ValueError as exc:
        raise ValueError(
            f"Unknown L0Tier {raw!r} in stats file metadata. "
            f"Valid: {[t.value for t in L0Tier]}."
        ) from exc


def _parse_bool_metadata(raw: str) -> bool:
    """Parse a strict bool stats-metadata value.

    ``"true"``→``True``; ``"false"``→``False``; any other value →
    ``ValueError`` (loud, mirroring :func:`_parse_l0_tier`). Callers pass
    ``metadata.get(key, "false")`` so an absent key (a stats file written
    before the field existed) deterministically yields ``False``.
    """
    if raw == "true":
        return True
    if raw == "false":
        return False
    raise ValueError(
        f"Invalid bool stats metadata {raw!r}; expected 'true' or 'false'."
    )


def save_statistics(
    stats,
    path: str | Path,
    *,
    extra_metadata: Mapping[str, str] | None = None,
) -> Path:
    """Save an ExtractionStatistics dataclass to safetensors.

    Per-feature data is stored as dense tensors indexed by feature ID
    (sorted by count descending). Scalar metadata is stored in the
    safetensors header as strings.

    Args:
        stats: An ExtractionStatistics instance.
        path: Output path. Extension is forced to ``.safetensors``.
        extra_metadata: Optional producer provenance to add to the existing
            safetensors metadata surface. Keys may not collide with statistics
            fields. ``None`` preserves the legacy serialized bytes.

    Returns:
        The actual path written.
    """
    path = Path(path).with_suffix(SAFETENSORS_EXT)

    tensors: dict[str, t.Tensor] = {}
    metadata: dict[str, str] = {}

    pfs = stats.per_feature_stats  # dict[str, PerFeatureStat]

    # Feature IDs in count-descending order (already sorted by compute_statistics)
    fids = list(pfs.keys())
    n_active = len(fids)

    if n_active > 0:
        feature_stats = list(pfs.values())
        tensors["feature_ids"] = t.tensor([int(f) for f in fids], dtype=t.int32)

        first_stat = feature_stats[0]
        pct_keys = sorted(first_stat.percentiles.keys())
        n_pct = len(pct_keys)
        metadata["percentile_names"] = ",".join(pct_keys)

        tensors["counts"] = t.tensor(
            [fstat.count for fstat in feature_stats], dtype=t.int64
        )
        tensors["pct_segments_active"] = t.tensor(
            [fstat.pct_segments_active for fstat in feature_stats],
            dtype=t.float32,
        )
        tensors["mean_mag"] = t.tensor(
            [fstat.mean_mag for fstat in feature_stats], dtype=t.float64
        )
        tensors["mean_mag_incl_zeros"] = t.tensor(
            [fstat.mean_mag_incl_zeros for fstat in feature_stats], dtype=t.float64
        )
        tensors["std_mag"] = t.tensor(
            [fstat.std_mag for fstat in feature_stats], dtype=t.float64
        )
        tensors["min_mag"] = t.tensor(
            [fstat.min_mag for fstat in feature_stats], dtype=t.float64
        )
        tensors["max_mag"] = t.tensor(
            [fstat.max_mag for fstat in feature_stats], dtype=t.float64
        )
        tensors["median_mag"] = t.tensor(
            [fstat.median_mag for fstat in feature_stats], dtype=t.float64
        )
        tensors["percentile_values"] = t.tensor(
            [
                [fstat.percentiles.get(pk, 0.0) for pk in pct_keys]
                for fstat in feature_stats
            ],
            dtype=t.float64,
        ).reshape(n_active, n_pct)

        # The first present histogram defines the shared row width. Preserve
        # the legacy empty-first behavior: zero bins suppress the tensor even
        # if a later feature has a non-empty histogram.
        hist_n_bins: int | None = None
        for fstat in feature_stats:
            if fstat.histogram and "counts" in fstat.histogram:
                hist_n_bins = len(fstat.histogram["counts"])
                break

        if hist_n_bins:
            histogram_rows = []
            for fstat in feature_stats:
                histogram = fstat.histogram
                histogram_rows.append(
                    histogram["counts"]
                    if histogram and "counts" in histogram
                    else [0] * hist_n_bins
                )
            tensors["histogram_counts"] = t.tensor(
                histogram_rows, dtype=t.int64
            )

    # Shared histogram bin edges
    if stats.features.mag_histogram_bin_edges is not None:
        tensors["mag_histogram_bin_edges"] = t.tensor(
            stats.features.mag_histogram_bin_edges, dtype=t.float64
        )

    # Corpus-level histograms
    _save_histogram(
        tensors,
        "active_per_segment",
        stats.features.active_per_segment_histogram,
    )
    _save_histogram(
        tensors,
        "segment_length",
        stats.features.segment_length_histogram,
    )
    _save_histogram(
        tensors,
        "features_per_token",
        stats.features.features_per_token_histogram,
    )

    # Corpus-level percentiles
    _save_percentile_dict(
        tensors,
        metadata,
        "active_per_segment_pct",
        stats.features.active_per_segment_percentiles,
    )
    _save_percentile_dict(
        tensors,
        metadata,
        "segment_length_pct",
        stats.features.segment_length_percentiles,
    )
    _save_percentile_dict(
        tensors,
        metadata,
        "features_per_token_pct",
        stats.features.features_per_token_percentiles,
    )

    # Count percentiles
    _save_percentile_dict(
        tensors,
        metadata,
        "count_pct",
        stats.features.count_percentiles,
    )

    # Magnitude stats (nested dict -> JSON string)
    if stats.features.magnitude_stats:
        metadata["magnitude_stats"] = json.dumps(stats.features.magnitude_stats)

    # Scalar metadata
    meta = stats.metadata
    metadata["pooling_mode"] = meta.pooling_mode
    metadata["segment_mode"] = meta.segment_mode
    metadata["segment_size"] = str(meta.segment_size)
    metadata["context_window"] = str(meta.context_window)
    metadata["layer"] = str(meta.layer)
    metadata["n_sae_features"] = str(meta.n_sae_features)
    metadata["generated_at"] = meta.generated_at
    metadata["activation_percentile"] = str(meta.activation_percentile)

    # gemma-scope-2 L0 tier. Written only when present so v1 stats files (and
    # any pre-existing file) stay byte-identical; absent stays absent on load.
    if stats.l0_tier is not None:
        # safetensors metadata must be dict[str, str]; serialize the enum's
        # string value explicitly rather than relying on StrEnum coercion.
        metadata["l0_tier"] = stats.l0_tier.value

    # ignore_bos_features. Always written (plain bool, like every other
    # scalar metadata). A stats file written before this field existed lacks
    # the key and loads as False via the metadata.get(..., "false") default.
    metadata["ignore_bos_features"] = (
        "true" if stats.ignore_bos_features else "false"
    )

    # Corpus stats
    corpus = stats.corpus
    metadata["n_shards_processed"] = str(corpus.n_shards_processed)
    metadata["n_docs"] = str(corpus.n_docs)
    metadata["n_tokens_gpt2"] = str(corpus.n_tokens_gpt2)
    metadata["mean_tokens_per_doc"] = str(corpus.mean_tokens_per_doc)
    metadata["median_tokens_per_doc"] = str(corpus.median_tokens_per_doc)

    # Feature aggregate stats
    features = stats.features
    metadata["n_segments"] = str(features.n_segments)
    metadata["n_features_seen"] = str(features.n_features_seen)
    metadata["mean_active_per_segment"] = str(features.mean_active_per_segment)
    metadata["median_active_per_segment"] = str(features.median_active_per_segment)
    metadata["p90_active_per_segment"] = str(features.p90_active_per_segment)
    metadata["max_active_per_segment"] = str(features.max_active_per_segment)
    metadata["min_active_per_segment"] = str(features.min_active_per_segment)
    metadata["n_total_sae_features"] = str(features.n_total_sae_features)
    metadata["pct_features_active"] = str(features.pct_features_active)

    # Storage stats
    storage = stats.storage
    metadata["total_bytes"] = str(storage.total_bytes)
    metadata["mean_bytes_per_doc"] = str(storage.mean_bytes_per_doc)
    metadata["mean_bytes_per_token"] = str(storage.mean_bytes_per_token)

    # Extrapolation
    extrap = stats.extrapolation_10b
    metadata["extrap_target_tokens"] = str(extrap.target_tokens)
    metadata["extrap_estimated_docs"] = str(extrap.estimated_docs)
    metadata["extrap_estimated_storage_gb"] = str(extrap.estimated_storage_gb)
    metadata["extrap_estimated_wall_time_hours"] = str(extrap.estimated_wall_time_hours)

    # Per-feature thresholds
    if stats.per_feature_thresholds is not None:
        fids_thresh = sorted(stats.per_feature_thresholds.keys(), key=int)
        if fids_thresh:
            thresh_fids = t.tensor([int(f) for f in fids_thresh], dtype=t.int32)
            thresh_vals = t.tensor(
                [stats.per_feature_thresholds[f] for f in fids_thresh],
                dtype=t.float64,
            )
            tensors["threshold_feature_ids"] = thresh_fids
            tensors["threshold_values"] = thresh_vals

    # Health / diagnostic metrics (additive; namespaced so they cannot collide
    # with reserved keys; round-tripped by _load_safetensors_stats).
    if stats.health_metrics:
        for k, v in stats.health_metrics.items():
            metadata[f"health.{k}"] = v

    if extra_metadata is not None:
        for key, value in extra_metadata.items():
            if not isinstance(key, str) or not isinstance(value, str):
                raise TypeError(
                    "save_statistics extra_metadata must be a str-to-str mapping, "
                    f"got {key!r}: {value!r}"
                )
            if key in metadata:
                raise ValueError(
                    f"save_statistics extra_metadata key {key!r} collides with "
                    "reserved statistics metadata"
                )
            metadata[key] = value

    # Atomic write: tmp + fsync + os.replace via shared helper. Prevents a
    # crashed compute_statistics run from leaving a torn file that downstream
    # loaders would silently consume.
    atomic_safetensors_save(tensors, path, metadata=metadata)
    return path


def load_statistics(path: str | Path) -> ExtractionStatistics:
    """Load feature statistics from a safetensors file.

    Returns:
        An ``ExtractionStatistics`` dataclass with typed attribute access.
    """
    path = Path(path)

    if path.suffix != SAFETENSORS_EXT:
        raise ValueError(
            f"Unknown statistics file extension: {path.suffix!r} "
            f"(expected {SAFETENSORS_EXT})"
        )

    return _load_safetensors_stats(path)


def content_sha256(path: str | Path) -> str:
    """Hash the decoded content of a statistics safetensors file.

    ``atomic_safetensors_save`` (``nanochat/atomic_io.py``) writes the file's
    header via the ``safetensors`` crate, which serializes its metadata
    ``HashMap`` in a randomized key order on every call. Decoded tensors and
    the parsed metadata dict are bit-identical across independent saves of
    the same data, but the raw file bytes (and thus a plain file-level
    SHA-256) are not. Hash the decoded, canonically-ordered content instead
    so identity checks are reproducible across re-saves.
    """
    path = Path(path)
    tensors = load_file(str(path))
    metadata = _read_metadata(path)

    digest = hashlib.sha256()
    for name in sorted(tensors):
        tensor = tensors[name].cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("utf-8"))
        digest.update(repr(tuple(tensor.shape)).encode("utf-8"))
        digest.update(tensor.numpy().tobytes())
    digest.update(json.dumps(dict(metadata), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def verify_statistics_sha256(
    path: str | Path,
    expected: str,
    *,
    surface: str,
) -> str:
    """Verify a current content hash or an exact legacy raw-file hash.

    Statistics identities created before :func:`content_sha256` recorded the
    SHA-256 of the safetensors file bytes. Those identities cannot be
    recomputed after an equivalent re-save because safetensors metadata key
    order is unstable, but they can still be verified against the exact
    original file. Prefer the canonical decoded-content hash; only compute the
    raw digest as a compatibility fallback when that comparison fails.

    Returns the caller-supplied digest so legacy identities remain unchanged
    in downstream cache paths and manifests.
    """
    path = Path(path)
    actual_content_sha256 = content_sha256(path)
    if expected == actual_content_sha256:
        return expected

    raw_digest = hashlib.sha256()
    with path.open("rb") as statistics_file:
        while chunk := statistics_file.read(8 * 1024 * 1024):
            raw_digest.update(chunk)
    actual_file_sha256 = raw_digest.hexdigest()
    if expected == actual_file_sha256:
        logger.warning(
            "{} uses a verified legacy raw-file statistics SHA-256 for {}; "
            "the identity is exact for these bytes but is not stable across "
            "equivalent safetensors re-saves",
            surface,
            path,
        )
        return expected

    raise ValueError(
        f"{surface} stats_source_sha256 does not match stats_path: "
        f"configured={expected!r}, actual_content={actual_content_sha256!r}, "
        f"actual_file={actual_file_sha256!r}"
    )


def statistics_path(directory: str | Path) -> Path:
    """Return the default statistics file path (safetensors format)."""
    return Path(directory) / f"{STATS_STEM}{SAFETENSORS_EXT}"


def find_statistics_path(directory: str | Path) -> Path | None:
    """Find the statistics file in a directory."""
    sf = Path(directory) / f"{STATS_STEM}{SAFETENSORS_EXT}"
    return sf if sf.is_file() else None


def glob_statistics(directory: str | Path) -> list[Path]:
    """Recursively find all statistics files under directory."""
    d = Path(directory)
    return sorted(d.rglob(f"{STATS_STEM}{SAFETENSORS_EXT}"))


def load_statistics_metadata(path: str | Path) -> dict[str, str]:
    """Read a statistics safetensors header without loading tensor bodies."""
    return _read_metadata(Path(path))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _save_histogram(
    tensors: dict[str, t.Tensor],
    prefix: str,
    histogram: dict | None,
) -> None:
    """Save a histogram dict (with bin_edges and counts) as tensors."""
    if histogram is None or not histogram:
        return
    edges = histogram.get("bin_edges", [])
    counts = histogram.get("counts", [])
    if edges and counts:
        tensors[f"{prefix}_hist_edges"] = t.tensor(edges, dtype=t.float64)
        tensors[f"{prefix}_hist_counts"] = t.tensor(counts, dtype=t.int64)


def _save_percentile_dict(
    tensors: dict[str, t.Tensor],
    metadata: dict[str, str],
    prefix: str,
    pct_dict: dict[str, float] | None,
) -> None:
    """Save a percentile dict as a tensor + keys in metadata.

    .. warning::

        Save/load are NOT precision-symmetric: this writer stores values as
        ``float64`` exactly as supplied, while ``_load_percentile_dict`` rounds
        each value to 4 decimal digits. ``compute_statistics`` typically pre-
        rounds values via ``_percentile_dict(round_digits=…)`` so the asymmetry
        is invisible at the orchestrator boundary, but a caller passing
        higher-precision values (e.g. ``p50 = 1.234567``) WILL observe the
        rounding on read. This is intentional load-time normalisation; tighten
        save+load together if a future pipeline needs sub-1e-4 precision.
        Map P3 #14 / consolidation manifest §2.4.
    """
    if pct_dict is None or not pct_dict:
        return
    keys = sorted(pct_dict.keys())
    metadata[f"{prefix}_keys"] = ",".join(keys)
    vals = [pct_dict[k] for k in keys]
    tensors[f"{prefix}_values"] = t.tensor(vals, dtype=t.float64)


def _load_safetensors_stats(path: Path) -> ExtractionStatistics:
    """Load safetensors statistics and reconstruct typed dataclasses."""
    tensors = load_file(str(path))
    metadata = _read_metadata(path)
    return _reconstruct_statistics(tensors, metadata)


def _can_release_percentile_rows(tensors):
    if type(tensors) is not dict or not all(type(key) is str for key in tensors):
        return False
    value = tensors.get("percentile_values")
    return (
        type(value) is t.Tensor
        and not t.overrides.has_torch_function((value,))
        and not dict.__len__(vars(value))
        and t.Tensor.tolist is _TENSOR_TOLIST
    )


def _reconstruct_statistics(
    tensors: dict[str, t.Tensor], metadata: dict[str, str]
) -> ExtractionStatistics:
    """Rebuild the typed ``ExtractionStatistics`` dataclasses from already-
    parsed tensors + safetensors metadata header. Split out of
    ``_load_safetensors_stats`` so a caller that already has both in memory
    (e.g. from one already-verified in-memory snapshot; see
    ``load_statistics_and_checksum_from_bytes``) does not need a second,
    independent filesystem read to get the same reconstruction.
    """
    meta = MetadataStats(
        pooling_mode=metadata.get("pooling_mode", ""),
        segment_mode=SegmentMode(metadata["segment_mode"])
        if metadata.get("segment_mode")
        else SegmentMode.fixed_tokens,
        segment_size=int(metadata.get("segment_size", "0")),
        context_window=int(metadata.get("context_window", "0")),
        layer=int(metadata.get("layer", "0")),
        n_sae_features=int(metadata.get("n_sae_features", "0")),
        generated_at=metadata.get("generated_at", ""),
        activation_percentile=float(metadata.get("activation_percentile", "0.0")),
    )

    corpus = CorpusStats(
        n_shards_processed=int(metadata.get("n_shards_processed", "0")),
        n_docs=int(metadata.get("n_docs", "0")),
        n_tokens_gpt2=int(metadata.get("n_tokens_gpt2", "0")),
        mean_tokens_per_doc=float(metadata.get("mean_tokens_per_doc", "0.0")),
        median_tokens_per_doc=int(metadata.get("median_tokens_per_doc", "0")),
    )

    storage = StorageStats(
        total_bytes=int(metadata.get("total_bytes", "0")),
        mean_bytes_per_doc=float(metadata.get("mean_bytes_per_doc", "0.0")),
        mean_bytes_per_token=float(metadata.get("mean_bytes_per_token", "0.0")),
    )

    extrapolation = Extrapolation(
        target_tokens=int(metadata.get("extrap_target_tokens", "0")),
        estimated_docs=int(metadata.get("extrap_estimated_docs", "0")),
        estimated_storage_gb=float(metadata.get("extrap_estimated_storage_gb", "0.0")),
        estimated_wall_time_hours=float(
            metadata.get("extrap_estimated_wall_time_hours", "0.0")
        ),
    )

    # Count percentiles (stored as float64 tensors, need int cast)
    count_pct_raw = _load_percentile_dict(tensors, metadata, "count_pct")
    count_percentiles: dict[str, float | int] = {
        k: int(v) for k, v in count_pct_raw.items()
    }

    # Magnitude stats
    mag_stats_str = metadata.get("magnitude_stats")
    magnitude_stats = json.loads(mag_stats_str) if mag_stats_str else {}

    # Shared mag histogram bin edges
    mag_histogram_bin_edges = None
    if "mag_histogram_bin_edges" in tensors:
        mag_histogram_bin_edges = [
            round(v, 6) for v in tensors["mag_histogram_bin_edges"].tolist()
        ]

    features = SegmentFeatureStats(
        n_segments=int(metadata.get("n_segments", "0")),
        n_features_seen=int(metadata.get("n_features_seen", "0")),
        mean_active_per_segment=float(metadata.get("mean_active_per_segment", "0.0")),
        median_active_per_segment=int(metadata.get("median_active_per_segment", "0")),
        p90_active_per_segment=int(metadata.get("p90_active_per_segment", "0")),
        max_active_per_segment=int(metadata.get("max_active_per_segment", "0")),
        min_active_per_segment=int(metadata.get("min_active_per_segment", "0")),
        n_total_sae_features=int(metadata.get("n_total_sae_features", "0")),
        pct_features_active=float(metadata.get("pct_features_active", "0.0")),
        count_percentiles=count_percentiles,
        magnitude_stats=magnitude_stats,
        active_per_segment_histogram=_load_histogram(tensors, "active_per_segment"),
        active_per_segment_percentiles=_load_percentile_dict(
            tensors, metadata, "active_per_segment_pct"
        ),
        mag_histogram_bin_edges=mag_histogram_bin_edges,
        segment_length_histogram=_load_histogram(tensors, "segment_length"),
        segment_length_percentiles=_load_percentile_dict(
            tensors, metadata, "segment_length_pct"
        ),
        features_per_token_histogram=_load_histogram(tensors, "features_per_token"),
        features_per_token_percentiles=_load_percentile_dict(
            tensors, metadata, "features_per_token_pct"
        ),
    )

    # Per-feature stats
    per_feature_stats: dict[str, PerFeatureStat] = {}

    if "feature_ids" in tensors:
        fids = tensors["feature_ids"].tolist()
        counts = tensors["counts"].tolist()
        pct_seg_active = tensors["pct_segments_active"].tolist()
        mean_mag = tensors["mean_mag"].tolist()
        mean_mag_iz = tensors["mean_mag_incl_zeros"].tolist()
        std_mag = tensors["std_mag"].tolist()
        min_mag = tensors["min_mag"].tolist()
        max_mag = tensors["max_mag"].tolist()
        median_mag = tensors["median_mag"].tolist()
        release_percentile_rows = _can_release_percentile_rows(tensors)
        percentile_values = tensors["percentile_values"].tolist()
        release_percentile_rows = release_percentile_rows and type(percentile_values) is list

        pct_names_str = metadata.get("percentile_names", "")
        pct_names = pct_names_str.split(",") if pct_names_str else []

        has_hist = "histogram_counts" in tensors
        hist_counts_t = tensors.get("histogram_counts")

        for i, fid in enumerate(fids):
            pcts = {}
            for j, pk in enumerate(pct_names):
                pcts[pk] = round(percentile_values[i][j], 6)
            if release_percentile_rows and i < len(percentile_values):
                percentile_values[i] = None

            hist_data: dict[str, object] | None = None
            if has_hist and hist_counts_t is not None:
                hist_data = {"counts": hist_counts_t[i].tolist()}

            per_feature_stats[str(fid)] = PerFeatureStat(
                count=int(counts[i]),
                pct_segments_active=round(pct_seg_active[i], 4),
                mean_mag=round(mean_mag[i], 6),
                mean_mag_incl_zeros=round(mean_mag_iz[i], 6),
                std_mag=round(std_mag[i], 6),
                min_mag=round(min_mag[i], 6),
                max_mag=round(max_mag[i], 6),
                median_mag=round(median_mag[i], 6),
                percentiles=pcts,
                histogram=hist_data,
            )

    # Per-feature thresholds
    per_feature_thresholds = None
    if "threshold_feature_ids" in tensors:
        thresh_fids = tensors["threshold_feature_ids"].tolist()
        thresh_vals = tensors["threshold_values"].tolist()
        per_feature_thresholds = {
            str(fid): round(val, 6) for fid, val in zip(thresh_fids, thresh_vals, strict=True)
        }

    health = {k[len("health."):]: v for k, v in metadata.items() if k.startswith("health.")}

    return ExtractionStatistics(
        metadata=meta,
        corpus=corpus,
        features=features,
        storage=storage,
        per_feature_stats=per_feature_stats,
        extrapolation_10b=extrapolation,
        per_feature_thresholds=per_feature_thresholds,
        l0_tier=_parse_l0_tier(metadata.get("l0_tier")),
        ignore_bos_features=_parse_bool_metadata(
            metadata.get("ignore_bos_features", "false")
        ),
        health_metrics=health or None,
    )


def _load_histogram(tensors: dict[str, t.Tensor], prefix: str) -> dict[str, object]:
    """Load a histogram from tensors.

    Raises DataLoadError if expected keys are missing.
    """
    edges_key = f"{prefix}_hist_edges"
    counts_key = f"{prefix}_hist_counts"
    if edges_key not in tensors and counts_key not in tensors:
        # Histogram was not saved (empty dict at write time) — return empty.
        return {}
    if edges_key not in tensors:
        raise DataLoadError(f"Missing histogram edges key {edges_key!r} in statistics file")
    if counts_key not in tensors:
        raise DataLoadError(f"Missing histogram counts key {counts_key!r} in statistics file")
    return {
        "bin_edges": [round(v, 4) for v in tensors[edges_key].tolist()],
        "counts": tensors[counts_key].tolist(),
    }


def _load_percentile_dict(
    tensors: dict[str, t.Tensor],
    metadata: dict[str, str],
    prefix: str,
) -> dict[str, float]:
    """Load a percentile dict from tensors + metadata keys.

    Raises DataLoadError if expected keys are missing.

    .. note::

        Float values are rounded to 4 decimal digits on load. The save side
        (:func:`_save_percentile_dict`) writes the float as-is, so the
        save+load round-trip is NOT bitwise identity for values that exceed
        4-digit precision. Pinned by the slice 04 regression test
        ``test_save_percentile_dict_roundtrip.py`` so a future precision
        change is detected. Map P3 #14 / consolidation manifest §2.4.
    """
    keys_str = metadata.get(f"{prefix}_keys", "")
    values_key = f"{prefix}_values"
    if not keys_str:
        # Percentile data was not saved (empty dict at write time) — return empty.
        if values_key not in tensors:
            return {}
        raise DataLoadError(f"Missing percentile keys metadata for prefix {prefix!r} in statistics file")
    if values_key not in tensors:
        raise DataLoadError(f"Missing percentile values key {values_key!r} in statistics file")
    keys = keys_str.split(",")
    vals = tensors[values_key].tolist()
    return {k: round(float(v), 4) for k, v in zip(keys, vals, strict=True)}


PERCENTILE_CHECKSUM_KEY = "percentile_checksum"


def save_percentile_checksum(stats_path: str | Path, checksum: str) -> None:
    """Write a percentile checksum into the statistics file's metadata.

    Adds or updates the ``percentile_checksum`` metadata when needed. An
    already-current value is a byte-preserving no-op so repeated precompute
    validation cannot churn the statistics source identity.
    """
    stats_path = Path(stats_path)
    metadata = _read_metadata(stats_path)
    # safetensors metadata serialization does not preserve a stable byte
    # ordering across rewrites. Re-saving an already-current checksum would
    # therefore change the exact statistics source SHA-256 even though every
    # tensor and metadata value is unchanged, invalidating caches keyed by that
    # source identity. The completed-overlay path must be a literal no-op.
    if metadata.get(PERCENTILE_CHECKSUM_KEY) == checksum:
        return
    tensors = load_file(str(stats_path))
    metadata[PERCENTILE_CHECKSUM_KEY] = checksum
    # Atomic rewrite: the old file stays intact until the tmp file is fully
    # written + fsynced + renamed on top. A crash here used to blank the stats.
    atomic_safetensors_save(tensors, stats_path, metadata=metadata)


def load_percentile_checksum(stats_path: str | Path) -> str | None:
    """Read the percentile checksum from a statistics file, or None if absent."""
    metadata = _read_metadata(Path(stats_path))
    return metadata.get(PERCENTILE_CHECKSUM_KEY)


def load_statistics_and_checksum_from_bytes(
    payload: bytes,
) -> tuple[ExtractionStatistics, str | None, t.Tensor | None]:
    """Reconstruct statistics, the stored percentile checksum, AND the raw
    (pre-collapse) ``feature_ids`` tensor from ONE already-verified
    in-memory snapshot -- no filesystem access.

    A caller that has independently digest-verified ``payload`` (e.g. via a
    fd-bound stream hash followed by a content-hash-checked read, mirroring
    ``steering.eval.cluster_checkpoint_pin._load_json_mapping``'s fused
    design) gets all three values from that single verified snapshot,
    instead of the shape ``load_statistics``/``load_percentile_checksum``
    have when called separately against a live path (each independently
    reopens and re-reads the file).

    ``feature_ids`` is returned pre-collapse (Codex fix batch finding 6):
    the raw safetensors tensor, before ``_reconstruct_statistics`` folds it
    into a ``dict[str, PerFeatureStat]`` keyed by ``str(feature_id)``,
    where a later, larger row silently wins any tensor-level duplicate. A
    caller that needs to validate dtype/range/uniqueness on the ORIGINAL
    per-row data must read it from here -- by the time ``ExtractionStatistics
    .per_feature_stats`` exists, that evidence is already gone. ``None``
    when the file has no active features (mirrors ``_reconstruct_
    statistics``'s own ``if "feature_ids" in tensors`` guard).
    """
    tensors = _safetensors_load_bytes(payload)
    # The canonical byte reader returns ``None`` for absent metadata, while
    # statistics reconstruction has always consumed an empty mapping.
    metadata = read_safetensors_metadata_from_bytes(payload) or {}
    stats = _reconstruct_statistics(tensors, metadata)
    return stats, metadata.get(PERCENTILE_CHECKSUM_KEY), tensors.get("feature_ids")


TOKENIZER_CHECKSUM_KEY = "tokenizer_checksum"


def save_tokenizer_checksum(stats_path: str | Path, checksum: str) -> None:
    """Write a tokenizer checksum into the statistics file's metadata.

    Adds or updates the ``tokenizer_checksum`` metadata when needed. An
    already-current value is a byte-preserving no-op so repeated precompute
    validation cannot churn the statistics source identity.
    """
    stats_path = Path(stats_path)
    metadata = _read_metadata(stats_path)
    # See save_percentile_checksum: an already-current metadata write must not
    # churn the exact statistics file identity used by clustered eval caches.
    if metadata.get(TOKENIZER_CHECKSUM_KEY) == checksum:
        return
    tensors = load_file(str(stats_path))
    metadata[TOKENIZER_CHECKSUM_KEY] = checksum
    # Atomic rewrite — see save_percentile_checksum for rationale.
    atomic_safetensors_save(tensors, stats_path, metadata=metadata)


def load_tokenizer_checksum(stats_path: str | Path) -> str | None:
    """Read the tokenizer checksum from a statistics file, or None if absent."""
    metadata = _read_metadata(Path(stats_path))
    return metadata.get(TOKENIZER_CHECKSUM_KEY)


def _read_metadata(path: Path) -> dict[str, str]:
    """Read metadata header from a safetensors file without loading tensors."""
    with open(path, "rb") as f:
        header_size = struct.unpack("<Q", f.read(8))[0]
        header_bytes = f.read(header_size)

    header = json.loads(header_bytes)
    return header.get("__metadata__", {})
