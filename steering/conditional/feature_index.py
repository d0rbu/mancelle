"""Index for O(1) lookup of pre-extracted SAE features by parquet shard and document.

Holds a bounded number of shards' per-doc indices resident at once (a
capacity-``N`` LRU keyed by shard stem, MRU at the end). Construct with
``max_resident_shards`` to raise the working set above the default of 1 shard
(issue #740); the default reproduces the original LRU-1 cache byte-for-byte:
exactly one shard resident, switching shards evicts the old one and loads the
new one.

When ``prefetch=True``, a background daemon thread loads the predicted next
shard (sorted stem order) during the current shard's processing.  At most
``max_resident_shards + 1`` shards are held in memory simultaneously (the
resident set plus the single prefetch slot: one in-flight load or one
published-but-unconsumed result, never both).

Shards can be *pinned* via ``pin_shard()`` to keep them in memory permanently
(bypassing the LRU cache).  This is used for the validation shard so that
periodic eval doesn't evict the current training shard.

``retain()``/``ensure_resident()``/``drop_except()`` give callers explicit
control over the resident working set (e.g., pre-warming the shards a
splice/eval pass is about to touch). ``cache_stats`` exposes declared
best-effort diagnostic counters — never a correctness signal.
"""

from __future__ import annotations

import os
import stat
import threading
import time as _time
from collections import OrderedDict
from collections.abc import Collection, Iterable, Iterator, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import torch as t
from loguru import logger

from steering.conditional.shared_stem_store import SharedStemStore
from steering.feature_io import (
    CorruptFeatureShardError,
    MetadataKey,
    RegularFileIdentity,
    empty_segment_indices,
    glob_features,
    load_features,
    load_features_stable,
    read_shard_header,
    read_shard_header_stable,
    shard_stem_from_path,
)
from steering.overlay_io import (
    OverlayIncompleteError,
    OverlayKind,
    discover_overlay,
    expected_overlay_shard_stems,
    overlay_shard_path,
    validate_percentile_overlay_shard,
    validate_percentile_overlay_shard_stable,
    validate_token_boundary_overlay_shard,
)

# Subset of base-shard tensor fields the packed-rows / boundary consumers need when they do
# NOT use feature magnitudes (issue #531 O4). Skips loading + upcasting + splitting the large
# ``feature_mags`` tensor. ``feature_ids`` is delta-encoded against ``offsets`` so both must be
# present (load_features enforces this).
_NO_MAGS_SHARD_KEYS: frozenset[str] = frozenset(
    {
        "offsets",
        "feature_ids",
        "doc_ids",
        "segment_indices",
        "segment_positions",
        "segment_char_starts",
        "segment_char_ends",
    }
)
# ID-only subset for the startup percentile-coverage scan (issue #1495):
# document ids, CSR offsets, and (delta-decoded, requires offsets) feature
# ids. No mags, no char/token boundaries, no percentile overlay.
_ID_SCAN_SHARD_KEYS: frozenset[str] = frozenset({"offsets", "feature_ids", "doc_ids"})
# Read-only placeholder for SegmentFeatures.feature_mags when mags are not loaded. The
# packed-rows path (``_seg_features_to_packed``) never reads mags; sharing one empty tensor avoids
# allocating n_segments empties. NEVER mutate.
_EMPTY_MAGS: t.Tensor = t.empty(0, dtype=t.float32)
_CATALOG_SCHEMA_VERSION = 1
_SHARED_DOC_ORDER_KEY = "__feature_index_doc_order"
_SHARED_SORTED_DOC_IDS_KEY = "__feature_index_sorted_doc_ids"
_SHARED_DOC_COUNT_KEY = "__feature_index_doc_count"


class FeatureIndexCatalogError(RuntimeError):
    """An immutable parent-validated FeatureIndex catalog drifted or is malformed."""


@dataclass(frozen=True)
class MaterializedFeatureShard:
    """Exact transient raw/percentile leaves supplied by a transport."""

    raw_path: Path
    raw_identity: RegularFileIdentity
    percentile_path: Path
    percentile_identity: RegularFileIdentity


class FeatureShardMaterializer(Protocol):
    def materialize(self, stem: str) -> AbstractContextManager[MaterializedFeatureShard]: ...


@dataclass(frozen=True)
class TransportFeatureIndexCatalog:
    """Authenticated logical catalog whose bodies are absent until materialization."""

    feature_dir: str
    percentile_overlay_dir: str
    entries: tuple[tuple[str, str, str], ...]
    segment_size: int
    percentile_checksum: str
    validated_content_fingerprint: str

    def validate(self) -> None:
        feature_dir = Path(self.feature_dir)
        percentile_dir = Path(self.percentile_overlay_dir)
        if not feature_dir.is_absolute() or str(feature_dir) != self.feature_dir:
            raise FeatureIndexCatalogError("transport FeatureIndex feature_dir must be absolute")
        if not percentile_dir.is_absolute() or str(percentile_dir) != self.percentile_overlay_dir:
            raise FeatureIndexCatalogError(
                "transport FeatureIndex percentile_overlay_dir must be absolute"
            )
        if (
            isinstance(self.segment_size, bool)
            or not isinstance(self.segment_size, int)
            or self.segment_size < 1
        ):
            raise FeatureIndexCatalogError(
                "transport FeatureIndex segment_size must be a strict int >= 1"
            )
        for label, value in (
            ("percentile_checksum", self.percentile_checksum),
            ("validated_content_fingerprint", self.validated_content_fingerprint),
        ):
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(char not in "0123456789abcdef" for char in value)
            ):
                raise FeatureIndexCatalogError(
                    f"transport FeatureIndex {label} must be lowercase SHA-256"
                )
        stems = [stem for stem, _raw, _pct in self.entries]
        if not stems or stems != sorted(stems) or len(stems) != len(set(stems)):
            raise FeatureIndexCatalogError(
                "transport FeatureIndex entries must have unique sorted stems"
            )
        for stem, raw_value, pct_value in self.entries:
            raw_path = Path(raw_value)
            pct_path = Path(pct_value)
            if (
                not raw_path.is_absolute()
                or str(raw_path) != raw_value
                or raw_path.parent != feature_dir
                or shard_stem_from_path(raw_path) != stem
            ):
                raise FeatureIndexCatalogError(
                    f"transport FeatureIndex raw path is noncanonical for {stem!r}"
                )
            if (
                not pct_path.is_absolute()
                or str(pct_path) != pct_value
                or pct_path != overlay_shard_path(percentile_dir, stem)
            ):
                raise FeatureIndexCatalogError(
                    f"transport FeatureIndex percentile path is noncanonical for {stem!r}"
                )


@dataclass(frozen=True)
class ArtifactSignature:
    """Exact path, link, target, and metadata identity without reading a body."""

    path: str
    lstat_fields: tuple[int, int, int, int, int, int]
    stat_fields: tuple[int, int, int, int, int, int]

    @staticmethod
    def _fields(result: os.stat_result) -> tuple[int, int, int, int, int, int]:
        return (
            int(result.st_mode),
            int(result.st_dev),
            int(result.st_ino),
            int(result.st_size),
            int(result.st_mtime_ns),
            int(result.st_ctime_ns),
        )

    @classmethod
    def capture(cls, path: Path) -> ArtifactSignature:
        absolute = path.absolute()
        try:
            link_result = absolute.lstat()
            target_result = absolute.stat()
        except OSError as exc:
            raise FeatureIndexCatalogError(
                f"cannot capture FeatureIndex catalog signature for {absolute}: {exc}"
            ) from exc
        return cls(
            path=str(absolute),
            lstat_fields=cls._fields(link_result),
            stat_fields=cls._fields(target_result),
        )

    def validate(self) -> None:
        try:
            current = ArtifactSignature.capture(Path(self.path))
        except FeatureIndexCatalogError as exc:
            raise FeatureIndexCatalogError(
                f"FeatureIndex catalog input disappeared or became unreadable: {self.path}"
            ) from exc
        if current != self:
            raise FeatureIndexCatalogError(
                f"FeatureIndex catalog input changed after parent validation: {self.path}; "
                f"expected lstat={self.lstat_fields}, stat={self.stat_fields}, "
                f"got lstat={current.lstat_fields}, stat={current.stat_fields}"
            )


@dataclass(frozen=True)
class FeatureIndexCatalog:
    """Immutable result of one full parent scan, safe to pass to spawn workers.

    The catalog records filesystem signatures rather than trusting paths. A
    worker validates every recorded base/overlay file and containing directory
    before using it. The parent still performs the existing full first-shard
    load and overlay validation, so catalog construction cannot weaken corrupt
    input detection.
    """

    schema_version: int
    feature_dir: str
    shard_entries: tuple[tuple[str, str], ...]
    segment_size: int | None
    percentile_checksum: str | None
    percentile_overlay_dir: str | None
    percentile_overlay_entries: tuple[tuple[str, str], ...]
    tokenizer_checksum: str | None
    token_overlay_dir: str | None
    token_overlay_entries: tuple[tuple[str, str], ...]
    signatures: tuple[ArtifactSignature, ...]
    validated_content_fingerprint: str | None = None
    # A transport-managed staging directory may legitimately gain and lose a
    # one-body lookahead while the exact staged raw leaf remains identity-bound.
    # Ordinary scans and certified inline evaluation retain the legacy default.
    feature_dir_signature_required: bool = True
    # The same rule applies to a separately streamed percentile leaf. The
    # checksum and canonical leaf path remain in the catalog; only the mutable
    # transport-owned parent directory and absent local manifest are excluded.
    percentile_overlay_dir_signature_required: bool = True

    def validate(self) -> None:
        if self.schema_version != _CATALOG_SCHEMA_VERSION:
            raise FeatureIndexCatalogError(
                f"FeatureIndex catalog schema_version={self.schema_version}, "
                f"expected {_CATALOG_SCHEMA_VERSION}"
            )
        feature_dir = Path(self.feature_dir)
        if not feature_dir.is_absolute() or str(feature_dir) != self.feature_dir:
            raise FeatureIndexCatalogError(
                f"FeatureIndex catalog feature_dir must be absolute: {self.feature_dir!r}"
            )
        stems = [stem for stem, _path in self.shard_entries]
        if stems != sorted(stems) or len(stems) != len(set(stems)):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog shard entries must have unique sorted stems"
            )
        if self.shard_entries:
            if (
                isinstance(self.segment_size, bool)
                or not isinstance(self.segment_size, int)
                or self.segment_size < 1
            ):
                raise FeatureIndexCatalogError(
                    "FeatureIndex catalog with shards requires segment_size int >= 1"
                )
        elif self.segment_size is not None:
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog without shards requires segment_size=None"
            )
        for stem, path_str in self.shard_entries:
            path = Path(path_str)
            if not path.is_absolute() or str(path) != path_str:
                raise FeatureIndexCatalogError(
                    f"FeatureIndex catalog shard path must be absolute: {path_str!r}"
                )
            if path.parent != feature_dir or shard_stem_from_path(path) != stem:
                raise FeatureIndexCatalogError(
                    "FeatureIndex catalog shard entry does not match its "
                    f"direct-child path: stem={stem!r}, path={path_str!r}"
                )
        if bool(self.percentile_checksum) != bool(self.percentile_overlay_dir):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog percentile checksum/overlay presence mismatch"
            )
        if bool(self.tokenizer_checksum) != bool(self.token_overlay_dir):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog tokenizer checksum/overlay presence mismatch"
            )
        if not isinstance(self.feature_dir_signature_required, bool):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog feature_dir_signature_required must be a bool"
            )
        if not isinstance(self.percentile_overlay_dir_signature_required, bool):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog percentile_overlay_dir_signature_required must be a bool"
            )
        overlay_specs = (
            (
                "percentile",
                self.percentile_overlay_dir,
                self.percentile_overlay_entries,
                OverlayKind.PERCENTILE,
            ),
            (
                "token",
                self.token_overlay_dir,
                self.token_overlay_entries,
                OverlayKind.SEG_BOUNDARY,
            ),
        )
        overlay_expected_paths: set[str] = set()
        for label, directory_str, entries, kind in overlay_specs:
            entry_stems = [stem for stem, _path in entries]
            if entry_stems != sorted(entry_stems) or len(entry_stems) != len(set(entry_stems)):
                raise FeatureIndexCatalogError(
                    f"FeatureIndex catalog {label} overlay entries must have unique sorted stems"
                )
            if directory_str is None:
                if entries:
                    raise FeatureIndexCatalogError(
                        f"FeatureIndex catalog {label} overlay entries require a directory"
                    )
                continue
            directory = Path(directory_str)
            if not directory.is_absolute() or str(directory) != directory_str:
                raise FeatureIndexCatalogError(
                    f"FeatureIndex catalog {label} overlay directory must be "
                    f"absolute: {directory_str!r}"
                )
            if kind is OverlayKind.PERCENTILE and entry_stems != stems:
                raise FeatureIndexCatalogError(
                    "FeatureIndex catalog percentile overlay must cover every base shard"
                )
            if kind is OverlayKind.SEG_BOUNDARY and not set(entry_stems).issubset(stems):
                raise FeatureIndexCatalogError(
                    "FeatureIndex catalog token overlay contains a non-base shard stem"
                )
            if kind is not OverlayKind.PERCENTILE or self.percentile_overlay_dir_signature_required:
                overlay_expected_paths.add(directory_str)
                overlay_expected_paths.add(str(directory / "manifest.json"))
            for stem, path_str in entries:
                expected_path = str(overlay_shard_path(directory, stem).absolute())
                if path_str != expected_path:
                    raise FeatureIndexCatalogError(
                        f"FeatureIndex catalog {label} overlay entry has noncanonical "
                        f"path for {stem!r}: {path_str!r} != {expected_path!r}"
                    )
                overlay_expected_paths.add(path_str)
        fingerprint = self.validated_content_fingerprint
        if fingerprint is not None and (
            len(fingerprint) != 64 or any(char not in "0123456789abcdef" for char in fingerprint)
        ):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog validated_content_fingerprint must be a "
                "lowercase SHA-256 hex digest"
            )
        signature_paths = [signature.path for signature in self.signatures]
        if signature_paths != sorted(signature_paths):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog artifact signatures must be path-sorted"
            )
        if len(signature_paths) != len(set(signature_paths)):
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog contains duplicate artifact signatures"
            )
        expected_paths = {self.feature_dir} if self.feature_dir_signature_required else set()
        expected_paths.update(path for _stem, path in self.shard_entries)
        expected_paths.update(overlay_expected_paths)
        actual_paths = set(signature_paths)
        missing = sorted(expected_paths - actual_paths)
        if missing:
            raise FeatureIndexCatalogError(
                f"FeatureIndex catalog is missing required signatures: {missing}"
            )
        unexpected = sorted(actual_paths - expected_paths)
        if unexpected:
            raise FeatureIndexCatalogError(
                f"FeatureIndex catalog contains unexpected signatures: {unexpected}"
            )
        # Validate files before their containing directories. Removing a file mutates the
        # directory too; checking the leaf first reports the exact missing input rather than
        # only the coarser directory-metadata drift.
        ordered_signatures = sorted(
            self.signatures,
            key=lambda signature: (
                stat.S_ISDIR(signature.stat_fields[0]),
                signature.path,
            ),
        )
        for signature in ordered_signatures:
            signature.validate()


@dataclass(frozen=True)
class SegmentFeatures:
    """Features for a single segment within a document."""

    segment_idx: int
    position: int  # token position of segment start
    feature_ids: t.Tensor  # [n_features] int32
    feature_mags: t.Tensor  # [n_features] float32
    char_start: int  # character start offset
    char_end: int  # character end offset
    feature_percentiles: t.Tensor | None = None  # [n_features] uint8, precomputed
    token_start: int | None = None  # nanochat token index where segment starts (inclusive)
    token_end: int | None = None  # nanochat token index where segment ends (exclusive)


class _DocSpanIndex(Mapping[int, list[SegmentFeatures]]):
    """Document lookup over the stable ``(doc_id, segment_idx)`` row order.

    Both lazy facades below binary-search the same pair of tensors — the row
    order ``_build_docs`` sorts into and the correspondingly sorted int64
    document ids — and differ only in how they materialize a document's
    segments.  The span arithmetic lives here rather than in each subclass
    because the duplicated copies had already drifted apart once (the eager
    token-bounds conjunction was mirrored in one facade and not the other).

    Subclasses set ``_order``, ``_sorted_doc_ids`` and ``_doc_count``, and
    define ``__getitem__``.
    """

    _order: t.Tensor
    _sorted_doc_ids: t.Tensor
    _doc_count: int

    def __len__(self) -> int:
        return self._doc_count

    def __iter__(self) -> Iterator[int]:
        # Ascending document id — the order _build_docs' dict was populated in.
        if self._sorted_doc_ids.numel() == 0:
            return
        for doc_id in t.unique_consecutive(self._sorted_doc_ids).tolist():
            yield int(doc_id)

    def _bounds(self, doc_idx: object) -> tuple[int, int] | None:
        """Half-open row span of ``doc_idx``, or None when it is absent.

        Non-``int`` keys and out-of-int64 ints return None rather than reaching
        ``searchsorted``: the eager dict accepts them only as misses, and an
        overflowing int64 scalar cannot be constructed for one.  Screening them
        HERE (rather than in ``__contains__`` alone) is what keeps
        ``key in mapping`` and ``mapping[key]`` in agreement.  Both facades
        deliberately differ from the eager dict for a float key equal to a
        present int (``docs[3.0]``): unreachable, since every caller keys by the
        int document ids stored in the shard.
        """
        if not isinstance(doc_idx, int) or not -(1 << 63) <= doc_idx < (1 << 63):
            return None
        left = int(t.searchsorted(self._sorted_doc_ids, doc_idx, right=False).item())
        right = int(t.searchsorted(self._sorted_doc_ids, doc_idx, right=True).item())
        if left == right:
            return None
        return left, right

    def __contains__(self, key: object) -> bool:
        return self._bounds(key) is not None


class _SharedLazyDocs(_DocSpanIndex):
    """Read-only per-document facade over one shared stem's contiguous tensors.

    Eager ``FeatureIndex`` shards create a ``SegmentFeatures`` object and tensor
    slice for every segment up front.  That private object graph is appropriate
    for the normal cache, but rebuilding it whenever an LRU-1 shared worker
    switches stems turns interleaved packed provenance into pathological CPU
    thrash.  A shared-store materialization therefore publishes two additional
    immutable tensors: the exact stable row order used by ``_build_docs`` and
    the corresponding sorted int64 document IDs.  This facade binary-searches
    those tensors and constructs only the requested document's segment views.

    It deliberately does not memoize returned lists.  The only caller enabling
    this mode is offline parallel splice preselection, which consumes a document
    synchronously and never observes object identity.  Values, dtype, shape,
    segment order, and storage aliasing match the eager shared-store path.
    """

    def __init__(self, data: dict) -> None:
        n_segments = int(data["n_segments"])
        order = data.get(_SHARED_DOC_ORDER_KEY)
        sorted_doc_ids = data.get(_SHARED_SORTED_DOC_IDS_KEY)
        doc_count = data.get(_SHARED_DOC_COUNT_KEY)
        if not isinstance(order, t.Tensor) or order.dtype is not t.int64:
            raise RuntimeError(
                "shared lazy FeatureIndex stem is missing its int64 document-order tensor"
            )
        if order.shape != (n_segments,):
            raise RuntimeError(
                "shared lazy FeatureIndex document-order shape mismatch: "
                f"{tuple(order.shape)} != {(n_segments,)}"
            )
        if not isinstance(sorted_doc_ids, t.Tensor) or sorted_doc_ids.dtype is not t.int64:
            raise RuntimeError(
                "shared lazy FeatureIndex stem is missing its int64 sorted-doc tensor"
            )
        if sorted_doc_ids.shape != (n_segments,):
            raise RuntimeError(
                "shared lazy FeatureIndex sorted-doc shape mismatch: "
                f"{tuple(sorted_doc_ids.shape)} != {(n_segments,)}"
            )
        if isinstance(doc_count, bool) or not isinstance(doc_count, int):
            raise RuntimeError(
                "shared lazy FeatureIndex stem is missing its integer document count"
            )
        if not 0 <= doc_count <= n_segments:
            raise RuntimeError(
                "shared lazy FeatureIndex document count is out of range: "
                f"{doc_count} for {n_segments} segments"
            )

        self._data = data
        self._order = order
        self._sorted_doc_ids = sorted_doc_ids
        self._doc_count = doc_count

    def __getitem__(self, doc_idx: int) -> list[SegmentFeatures]:
        bounds = self._bounds(doc_idx)
        if bounds is None:
            raise KeyError(doc_idx)
        left, right = bounds
        rows: list[int] = self._order[left:right].tolist()
        data = self._data
        offsets = data["offsets"]
        feature_ids = data["feature_ids"]
        feature_mags = data.get("feature_mags")
        feature_percentiles = data.get("feature_percentiles")
        seg_token_starts = data.get("seg_token_starts")
        seg_token_ends = data.get("seg_token_ends")

        segments: list[SegmentFeatures] = []
        for row in rows:
            feat_start = int(offsets[row].item())
            feat_end = int(offsets[row + 1].item())
            segments.append(
                SegmentFeatures(
                    segment_idx=int(data["segment_indices"][row].item()),
                    position=int(data["segment_positions"][row].item()),
                    feature_ids=feature_ids[feat_start:feat_end],
                    feature_mags=(
                        _EMPTY_MAGS if feature_mags is None else feature_mags[feat_start:feat_end]
                    ),
                    char_start=int(data["segment_char_starts"][row].item()),
                    char_end=int(data["segment_char_ends"][row].item()),
                    feature_percentiles=(
                        None
                        if feature_percentiles is None
                        else feature_percentiles[feat_start:feat_end]
                    ),
                    token_start=(
                        None if seg_token_starts is None else int(seg_token_starts[row].item())
                    ),
                    token_end=(None if seg_token_ends is None else int(seg_token_ends[row].item())),
                )
            )
        return segments


class _LazyShardDocs(_DocSpanIndex):
    """Per-document facade over ONE private shard's contiguous arrays (issue #1504).

    Eager ``_build_docs`` constructs a ``SegmentFeatures`` and a tensor slice for
    every segment of a shard up front — at production geometry ~1.8 M objects per
    load, a large GIL-holding phase that sits directly on the conditioned-SFT
    critical path and makes the background prefetch lose its race.  This mapping
    replaces it with one vectorized ``(doc_id, segment_idx)`` argsort at load
    time and per-document materialization on access.

    Like :class:`_SharedLazyDocs` it deliberately does **not** memoize returned
    lists: memoizing re-accumulates the entire eager object graph on top of the
    retained contiguous arrays, which measured 1.55x the eager benchmark's RSS
    delta and failed this task's 1.05x acceptance gate.  Its only caller
    (``SFTFeatureStore``, the live conditioned-SFT index) looks each document up
    once per epoch and never observes object identity; values, dtypes, storage
    independence, and segment order all still match the eager path exactly.

    Unlike :class:`_SharedLazyDocs`, the per-segment tensors are **clones**, not
    views: ``all_feature_ids`` escapes into the SFT conversation buffer for up to
    ~100 conversations, so views would pin whole evicted shards' arrays.
    """

    def __init__(self, data: dict) -> None:
        self._data = data
        n_segments = int(data["n_segments"])
        if n_segments == 0:
            # Parity with _build_docs' zero-segment early return, which precedes
            # every structural check (and any max() over an empty index tensor).
            self._order = t.empty(0, dtype=t.int64)
            self._sorted_doc_ids = t.empty(0, dtype=t.int64)
            self._doc_count = 0
            return

        # Eager loading enforces whole-shard consistency implicitly:
        # split_with_sizes(_copy) requires the per-segment sizes to sum EXACTLY to
        # each backing tensor's length, while _validate_feature_csr only requires
        # offsets[-1] <= len(feature_ids). A per-doc slicer would otherwise
        # silently accept a corrupt surplus that eager loading rejects today, and
        # for offsets[0] != 0 it would read a different span than eager does
        # (eager splits by relative sizes from index 0). Fail before publishing.
        offsets = data["offsets"]
        feature_ids = data["feature_ids"]
        csr_start = int(offsets[0].item())
        csr_end = int(offsets[-1].item())
        if csr_start != 0 or csr_end != len(feature_ids):
            raise RuntimeError(
                f"lazy FeatureIndex shard CSR span [{csr_start}, {csr_end}] does not "
                f"cover its {len(feature_ids)} feature_ids"
            )
        feature_mags = data.get("feature_mags")
        if feature_mags is not None and len(feature_mags) != len(feature_ids):
            raise RuntimeError(
                f"lazy FeatureIndex shard has {len(feature_mags)} feature_mags for "
                f"{len(feature_ids)} feature_ids"
            )
        feature_percentiles = data.get("feature_percentiles")
        if feature_percentiles is not None and len(feature_percentiles) != len(feature_ids):
            raise RuntimeError(
                f"lazy FeatureIndex shard has {len(feature_percentiles)} "
                f"feature_percentiles for {len(feature_ids)} feature_ids"
            )
        # Eager _build_docs indexes bulk .tolist() copies of the char bounds by
        # row, so a short array is an IndexError DURING the build. Nothing else
        # validates their length (_materialize_data_local checks segment_indices
        # and segment_positions, _validate_shard_lengths checks offsets/doc_ids),
        # and a per-doc gather would only fail on the unlucky document — so this
        # is the check that keeps the lazy path failing at load time too.
        segment_char_starts = data["segment_char_starts"]
        segment_char_ends = data["segment_char_ends"]
        if len(segment_char_starts) != n_segments or len(segment_char_ends) != n_segments:
            raise RuntimeError(
                f"lazy FeatureIndex shard has {len(segment_char_starts)} "
                f"segment_char_starts and {len(segment_char_ends)} segment_char_ends "
                f"for {n_segments} segments"
            )

        # Exactly _build_docs' compound sort key and stable argsort, so the
        # per-document segment order is provably identical.
        doc_ids_i64 = data["doc_ids"].to(t.int64)
        segment_indices_i64 = data["segment_indices"].to(t.int64)
        max_seg_idx = int(segment_indices_i64.max().item()) + 1
        sort_key = doc_ids_i64 * max_seg_idx + segment_indices_i64
        order = sort_key.argsort(stable=True)
        sorted_doc_ids = doc_ids_i64[order]
        self._order = order
        self._sorted_doc_ids = sorted_doc_ids
        self._doc_count = 1 + int(t.count_nonzero(sorted_doc_ids[1:] != sorted_doc_ids[:-1]).item())

    def __getitem__(self, doc_idx: int) -> list[SegmentFeatures]:
        bounds = self._bounds(doc_idx)
        if bounds is None:
            raise KeyError(doc_idx)
        left, right = bounds
        rows = self._order[left:right]
        data = self._data
        offsets = data["offsets"]
        feature_ids = data["feature_ids"]
        feature_mags = data.get("feature_mags")
        feature_percentiles = data.get("feature_percentiles")
        seg_token_starts = data.get("seg_token_starts")
        seg_token_ends = data.get("seg_token_ends")
        # Conjunction, exactly as eager _build_docs decides it: a half-present
        # overlay yields None on BOTH ends, never one real bound and one None.
        use_tok_bounds = seg_token_starts is not None and seg_token_ends is not None

        # One gather + one C++->Python crossing per metadata field per document.
        starts: list[int] = offsets[rows].tolist()
        ends: list[int] = offsets[rows + 1].tolist()
        seg_indices: list[int] = data["segment_indices"][rows].tolist()
        positions: list[int] = data["segment_positions"][rows].tolist()
        char_starts: list[int] = data["segment_char_starts"][rows].tolist()
        char_ends: list[int] = data["segment_char_ends"][rows].tolist()
        token_starts: list[int | None]
        token_ends: list[int | None]
        if use_tok_bounds:
            assert seg_token_starts is not None and seg_token_ends is not None
            token_starts = seg_token_starts[rows].tolist()
            token_ends = seg_token_ends[rows].tolist()
        else:
            token_starts = [None] * len(starts)
            token_ends = [None] * len(starts)

        # ``.to(int64, copy=True)`` is the eager whole-array upcast followed by
        # split_with_sizes_copy: identical values and dtype, independent storage.
        segments = [
            SegmentFeatures(
                segment_idx=seg_indices[i],
                position=positions[i],
                feature_ids=feature_ids[starts[i] : ends[i]].to(t.int64, copy=True),
                feature_mags=(
                    _EMPTY_MAGS
                    if feature_mags is None
                    else feature_mags[starts[i] : ends[i]].clone()
                ),
                char_start=char_starts[i],
                char_end=char_ends[i],
                feature_percentiles=(
                    None
                    if feature_percentiles is None
                    else feature_percentiles[starts[i] : ends[i]].clone()
                ),
                token_start=token_starts[i],
                token_end=token_ends[i],
            )
            for i in range(len(starts))
        ]
        return segments


@dataclass
class CacheStats:
    """Best-effort diagnostic counters for a FeatureIndex or SpliceProvenanceIndex cache.

    Under legacy background prefetch, ``loads`` may include worker-thread loads;
    counters are diagnostic (never a correctness signal), so minor cross-thread
    races on increments are acceptable. ``load_seconds`` and ``bytes_read`` are
    populated by both indices from their respective source files. Shared-store
    fields remain zero for a non-shared index and for
    ``SpliceProvenanceIndex``.
    """

    loads: int = 0
    hits: int = 0
    evictions: int = 0
    retain_calls: int = 0
    load_seconds: float = 0.0
    bytes_read: int = 0
    shared_hits: int = 0
    shared_misses: int = 0
    shared_get_races: int = 0
    materializations: int = 0
    materialize_seconds: float = 0.0
    materialize_lock_wait_seconds: float = 0.0
    publications: int = 0
    publish_declines: int = 0
    private_fallbacks: int = 0
    shared_mapped_bytes: int = 0

    def snapshot(self) -> dict[str, int | float]:
        """Return a stable, JSON-safe copy for per-task diagnostic deltas."""
        return {
            "loads": self.loads,
            "hits": self.hits,
            "evictions": self.evictions,
            "retain_calls": self.retain_calls,
            "load_seconds": self.load_seconds,
            "bytes_read": self.bytes_read,
            "shared_hits": self.shared_hits,
            "shared_misses": self.shared_misses,
            "shared_get_races": self.shared_get_races,
            "materializations": self.materializations,
            "materialize_seconds": self.materialize_seconds,
            "materialize_lock_wait_seconds": self.materialize_lock_wait_seconds,
            "publications": self.publications,
            "publish_declines": self.publish_declines,
            "private_fallbacks": self.private_fallbacks,
            "shared_mapped_bytes": self.shared_mapped_bytes,
        }

    def delta(self, earlier: dict[str, int | float]) -> dict[str, int | float]:
        """Subtract a prior :meth:`snapshot`, rejecting incomplete schemas."""
        current = self.snapshot()
        if set(earlier) != set(current):
            missing = sorted(set(current) - set(earlier))
            extra = sorted(set(earlier) - set(current))
            raise ValueError(f"CacheStats delta schema mismatch: missing={missing}, extra={extra}")
        return {key: current[key] - earlier[key] for key in current}


def _validate_shard_segment_size(
    shard_stem: str, shard_seg_size: int, expected: int | None
) -> None:
    """Cross-shard ``segment_size`` consistency, shared by both loaders.

    Message strings are pinned by tests through BOTH consumers — do not reword.
    """
    assert shard_seg_size == expected, (
        f"Segment size mismatch: shard {shard_stem} has segment_size="
        f"{shard_seg_size}, expected {expected}"
    )


def _validate_shard_lengths(offsets: t.Tensor, doc_ids_len: int, n_segments: int) -> None:
    """CSR offsets / doc-id array length consistency, shared by both loaders."""
    assert len(offsets) == n_segments + 1, (
        f"offsets length {len(offsets)} != n_segments + 1 ({n_segments + 1})"
    )
    assert doc_ids_len == n_segments


def _validate_feature_csr(shard_stem: str, offsets: t.Tensor, feature_ids_len: int) -> None:
    """Structural CSR validation shared by the full loader and the ID-only scan.

    A zero-segment shard (``offsets == [0]``) passes vacuously. Message
    strings are pinned by tests through BOTH consumers — do not reword.
    """
    assert (offsets[:-1] <= offsets[1:]).all().item(), "Non-monotonic offsets detected"
    empties = empty_segment_indices(offsets)
    if empties:
        i = empties[0]
        raise CorruptFeatureShardError(
            f"FeatureIndex: shard {shard_stem!r} segment {i} has zero "
            f"extracted features (offsets[{i}]==offsets[{i + 1}]="
            f"{int(offsets[i].item())}); base extracted shards must be "
            f"non-empty (issue #522). Re-extract this shard."
        )
    assert offsets[-1].item() <= feature_ids_len, (
        f"Offset end {offsets[-1].item()} exceeds feature_ids length {feature_ids_len}"
    )


class FeatureIndex:
    """Lazy, bounded-LRU-cached index of pre-extracted SAE features.

    On construction the directory is scanned for feature shard files but
    only the *first* shard is briefly loaded to read ``segment_size``, then
    immediately discarded.  Subsequent ``lookup()`` calls load (and cache) up
    to ``max_resident_shards`` shards at a time (default 1: exactly one shard
    resident, matching the original LRU-1 behavior byte-for-byte).

    ``lookup()`` raises ``KeyError`` for an unknown shard or an absent document
    (fail-fast; it never returns ``None``).
    """

    def __init__(
        self,
        feature_dir: str,
        prefetch: bool = False,
        percentile_checksum: str | None = None,
        tokenizer_checksum: str | None = None,
        skip_feature_mags: bool = False,
        compact_feature_ids: bool = False,
        max_resident_shards: int = 1,
        shared_store: SharedStemStore | None = None,
        lazy_shared_docs: bool = False,
        lazy_docs: bool = False,
        required_token_boundary_stems: Collection[str] | None = None,
        profile_loads: bool = False,
    ) -> None:
        self._initialize_empty(
            prefetch=prefetch,
            percentile_checksum=percentile_checksum,
            tokenizer_checksum=tokenizer_checksum,
            skip_feature_mags=skip_feature_mags,
            compact_feature_ids=compact_feature_ids,
            max_resident_shards=max_resident_shards,
            shared_store=shared_store,
            lazy_shared_docs=lazy_shared_docs,
            lazy_docs=lazy_docs,
            required_token_boundary_stems=required_token_boundary_stems,
            profile_loads=profile_loads,
        )
        self._scan(feature_dir)

    def _initialize_empty(
        self,
        *,
        prefetch: bool,
        percentile_checksum: str | None,
        tokenizer_checksum: str | None,
        skip_feature_mags: bool,
        compact_feature_ids: bool,
        max_resident_shards: int,
        shared_store: SharedStemStore | None,
        lazy_shared_docs: bool,
        required_token_boundary_stems: Collection[str] | None,
        profile_loads: bool = False,
        lazy_docs: bool = False,
    ) -> None:
        if (
            isinstance(max_resident_shards, bool)
            or not isinstance(max_resident_shards, int)
            or max_resident_shards < 1
        ):
            raise ValueError(
                f"max_resident_shards must be an int >= 1, got {max_resident_shards!r}"
            )
        self._max_resident_shards: int = max_resident_shards
        self._stats = CacheStats()
        if not isinstance(profile_loads, bool):
            raise TypeError(f"profile_loads must be a bool, got {profile_loads!r}")
        self._profile_loads = profile_loads
        self._load_profiles: list[dict] = []

        if not isinstance(lazy_shared_docs, bool):
            raise ValueError(f"lazy_shared_docs must be a bool, got {lazy_shared_docs!r}")
        if lazy_shared_docs and shared_store is None:
            raise ValueError("lazy_shared_docs requires a shared_store")
        if lazy_shared_docs and not compact_feature_ids:
            raise ValueError("lazy_shared_docs requires compact_feature_ids=True")
        if lazy_shared_docs and prefetch:
            raise ValueError("lazy_shared_docs does not support background prefetch")

        # Issue #1504: per-document materialization for the live conditioned-SFT
        # index. It supports exactly that configuration — a private, int64-resident
        # index — so every other combination is rejected loudly rather than
        # silently taking a different code path.
        if not isinstance(lazy_docs, bool):
            raise ValueError(f"lazy_docs must be a bool, got {lazy_docs!r}")
        if lazy_docs and lazy_shared_docs:
            raise ValueError("lazy_docs and lazy_shared_docs are mutually exclusive")
        if lazy_docs and shared_store is not None:
            raise ValueError("lazy_docs does not support a shared_store")
        if lazy_docs and compact_feature_ids:
            raise ValueError("lazy_docs does not support compact_feature_ids")
        self._lazy_docs: bool = lazy_docs

        # Issue #531 O4: when True, skip loading/upcasting/splitting feature_mags (the
        # packed-rows + boundary consumers never read them). SegmentFeatures.feature_mags is
        # an empty tensor on this index. Default False ⇒ the live dataloader path (which DOES
        # use mags for runtime selection) is byte-identical / unchanged.
        self._skip_feature_mags: bool = skip_feature_mags

        # Issue #740 PR-E2: when True, keep the resident feature_ids at their parquet int32
        # dtype instead of upcasting to int64 (~halves the biggest resident tensor). Byte-
        # identical for consumers that convert on use (the splice preselection path always
        # does seg.feature_ids.to(int64)/.long()); default False ⇒ int64-resident, unchanged.
        self._compact_feature_ids: bool = compact_feature_ids

        # Issue #740 PR-F: optional shared-memory source store. When set, _load_shard maps each
        # stem's materialized arrays once on /dev/shm and builds zero-copy VIEWS into that shared
        # buffer (instead of independent per-worker copies), so N parallel workers reference one
        # physical copy per stem. Default None ⇒ each index owns its copies (unchanged).
        self._shared_store: SharedStemStore | None = shared_store

        # Issue #758: parallel splice workers keep an LRU-1 mapping but must not
        # rebuild a complete per-segment Python object graph on every interleaved
        # stem switch.  The opt-in shared-only facade constructs one requested
        # document at a time from a compact locator published with the stem.
        self._lazy_shared_docs: bool = lazy_shared_docs

        if required_token_boundary_stems is None:
            self._required_token_boundary_stems: frozenset[str] | None = None
        else:
            if isinstance(required_token_boundary_stems, (str, bytes)):
                raise TypeError(
                    "required_token_boundary_stems must be a collection of shard "
                    "stem strings, not one string"
                )
            normalized = frozenset(required_token_boundary_stems)
            if not normalized:
                raise ValueError("required_token_boundary_stems must contain at least one stem")
            invalid = sorted(
                repr(stem) for stem in normalized if not isinstance(stem, str) or not stem
            )
            if invalid:
                raise TypeError(f"required_token_boundary_stems contains invalid stems: {invalid}")
            self._required_token_boundary_stems = normalized

        # {shard_stem: Path} — discovered shards, never eagerly loaded
        self._shard_paths: dict[str, Path] = {}

        # Bounded LRU-N resident map (MRU at end). Capacity 1 == the legacy
        # single-slot cache: same load/evict sequence, byte-for-byte.
        self._resident: OrderedDict[str, Mapping[int, list[SegmentFeatures]]] = OrderedDict()

        # Pinned shard cache: shards added here are never evicted (for val shard)
        self._pinned: dict[str, Mapping[int, list[SegmentFeatures]]] = {}

        self._segment_size: int | None = None

        # Precomputed percentile checksum for validation
        self._percentile_checksum: str | None = percentile_checksum
        self._has_precomputed_percentiles: bool = False
        self._feature_dir: Path | None = None
        # Discovered overlay directory (set in _scan)
        self._pct_overlay_dir: Path | None = None

        # Precomputed token boundary checksum for validation
        self._tokenizer_checksum: str | None = tokenizer_checksum
        self._has_precomputed_token_boundaries: bool = False
        self._tok_overlay_dir: Path | None = None

        # Single-slot prefetch state (issue #1504). One worker at a time holding
        # one shard map: the slot is guarded by ``_prefetch_lock``, and
        # ``_prefetch_closed`` makes a post-close publish a discard rather than a
        # leak. ``_prefetch_lock`` is assigned last, so ``close()`` can probe it
        # to tell a fully initialized instance from one whose ``__init__``
        # raised.
        self._prefetch = prefetch
        self._prefetch_thread: threading.Thread | None = None
        self._prefetch_result: Mapping[int, list[SegmentFeatures]] | None = None
        self._prefetch_stem: str | None = None
        self._prefetch_closed: bool = False
        # Diagnostic counters + load profiles are written by the prefetch worker
        # concurrently with the consumer thread; serialize the increments.
        self._stats_lock = threading.Lock()
        self._prefetch_lock = threading.Lock()

        # Sorted stem list for O(1) next-shard prediction (populated in _scan)
        self._sorted_stems: list[str] = []
        self._stem_to_idx: dict[str, int] = {}
        # Immutable filesystem snapshot captured by the parent's validating scan.
        # ``export_catalog`` refuses to rebase onto a later filesystem state.
        self._scan_artifact_signatures: dict[str, ArtifactSignature] = {}
        # Narrow certified sparse sources can bind a staged raw pathname to the
        # exact regular-file identity published by an external transport. Normal
        # scans never populate this map and retain the legacy load_features path.
        self._expected_shard_identities: dict[str, RegularFileIdentity] = {}
        self._expected_percentile_identities: dict[str, RegularFileIdentity] = {}
        self._transport_materializer: FeatureShardMaterializer | None = None

    @classmethod
    def from_catalog(
        cls,
        catalog: FeatureIndexCatalog,
        *,
        prefetch: bool = False,
        skip_feature_mags: bool = False,
        compact_feature_ids: bool = False,
        max_resident_shards: int = 1,
        shared_store: SharedStemStore | None = None,
        lazy_shared_docs: bool = False,
        profile_loads: bool = False,
    ) -> FeatureIndex:
        """Build empty worker caches from one parent scan without loading a shard."""
        catalog.validate()
        index = cls.__new__(cls)
        index._initialize_empty(
            prefetch=prefetch,
            percentile_checksum=catalog.percentile_checksum,
            tokenizer_checksum=catalog.tokenizer_checksum,
            skip_feature_mags=skip_feature_mags,
            compact_feature_ids=compact_feature_ids,
            max_resident_shards=max_resident_shards,
            shared_store=shared_store,
            lazy_shared_docs=lazy_shared_docs,
            required_token_boundary_stems=None,
            profile_loads=profile_loads,
        )
        index._feature_dir = Path(catalog.feature_dir)
        index._shard_paths = {stem: Path(path) for stem, path in catalog.shard_entries}
        index._segment_size = catalog.segment_size
        index._pct_overlay_dir = (
            Path(catalog.percentile_overlay_dir)
            if catalog.percentile_overlay_dir is not None
            else None
        )
        index._has_precomputed_percentiles = index._pct_overlay_dir is not None
        index._tok_overlay_dir = (
            Path(catalog.token_overlay_dir) if catalog.token_overlay_dir is not None else None
        )
        index._has_precomputed_token_boundaries = index._tok_overlay_dir is not None
        index._sorted_stems = [stem for stem, _path in catalog.shard_entries]
        index._stem_to_idx = {stem: i for i, stem in enumerate(index._sorted_stems)}
        index._scan_artifact_signatures = {
            signature.path: signature for signature in catalog.signatures
        }
        logger.info(
            f"FeatureIndex restored validated catalog with {len(index._shard_paths)} "
            f"shards in {catalog.feature_dir} (segment_size={catalog.segment_size})"
        )
        return index

    @classmethod
    def from_transport_catalog(
        cls,
        catalog: TransportFeatureIndexCatalog,
        *,
        materializer: FeatureShardMaterializer,
        skip_feature_mags: bool = False,
        compact_feature_ids: bool = False,
        max_resident_shards: int = 1,
        shared_store: SharedStemStore | None = None,
        lazy_shared_docs: bool = False,
        profile_loads: bool = False,
    ) -> FeatureIndex:
        """Build an index whose exact compressed leaves are supplied on cache miss."""
        catalog.validate()
        index = cls.__new__(cls)
        index._initialize_empty(
            prefetch=False,
            percentile_checksum=catalog.percentile_checksum,
            tokenizer_checksum=None,
            skip_feature_mags=skip_feature_mags,
            compact_feature_ids=compact_feature_ids,
            max_resident_shards=max_resident_shards,
            shared_store=shared_store,
            lazy_shared_docs=lazy_shared_docs,
            required_token_boundary_stems=None,
            profile_loads=profile_loads,
        )
        index._feature_dir = Path(catalog.feature_dir)
        index._pct_overlay_dir = Path(catalog.percentile_overlay_dir)
        index._has_precomputed_percentiles = True
        index._shard_paths = {stem: Path(raw_path) for stem, raw_path, _pct_path in catalog.entries}
        index._segment_size = catalog.segment_size
        index._sorted_stems = [stem for stem, _raw, _pct in catalog.entries]
        index._stem_to_idx = {stem: position for position, stem in enumerate(index._sorted_stems)}
        index._transport_materializer = materializer
        logger.info(
            "FeatureIndex restored authenticated transport catalog with {} shards "
            "(segment_size={}, fingerprint={}...)",
            len(index._shard_paths),
            catalog.segment_size,
            catalog.validated_content_fingerprint[:16],
        )
        return index

    @classmethod
    def from_certified_sparse_source(
        cls,
        *,
        feature_dir: Path,
        shard_path: Path,
        token_overlay_dir: Path,
        tokenizer_checksum: str,
        segment_size: int,
        validated_content_fingerprint: str,
        percentile_overlay_dir: Path | None = None,
        percentile_checksum: str | None = None,
        token_overlay_required: bool = True,
        skip_feature_mags: bool = False,
        expected_shard_identity: RegularFileIdentity | None = None,
        expected_percentile_identity: RegularFileIdentity | None = None,
        profile_loads: bool = False,
    ) -> FeatureIndex:
        """Build a one-shard index from an externally authenticated source.

        This constructor deliberately does not call :meth:`_scan`: a schema-v2
        token overlay is bound to its complete producer-side raw family, so an
        intentionally sparse remote staging directory cannot be revalidated by
        hashing only its local subset.  The caller must instead authenticate the
        exact raw shard and token leaf, then pass the certificate digest here.

        Ordinary ``FeatureIndex(...)`` construction and its complete-family
        overlay validation are unchanged.  This narrow constructor records the
        same immutable path signatures as a parent scan and reuses the normal
        materialization/lookup implementation byte-for-byte.
        """
        feature_dir = Path(feature_dir).absolute()
        shard_path = Path(shard_path).absolute()
        token_overlay_dir = Path(token_overlay_dir).absolute()
        if (percentile_overlay_dir is None) != (percentile_checksum is None):
            raise FeatureIndexCatalogError(
                "certified sparse FeatureIndex percentile_overlay_dir and "
                "percentile_checksum must be both present or both absent"
            )
        if expected_percentile_identity is not None and percentile_overlay_dir is None:
            raise FeatureIndexCatalogError(
                "certified sparse FeatureIndex expected_percentile_identity requires "
                "a percentile overlay"
            )
        if not isinstance(token_overlay_required, bool):
            raise TypeError("certified sparse FeatureIndex token_overlay_required must be a bool")
        if shard_path.parent != feature_dir:
            raise FeatureIndexCatalogError(
                "certified sparse FeatureIndex shard must be a direct child of "
                f"feature_dir: shard={shard_path}, feature_dir={feature_dir}"
            )
        shard_stem = shard_stem_from_path(shard_path)
        token_leaf = overlay_shard_path(token_overlay_dir, shard_stem).absolute()
        token_manifest_path = (token_overlay_dir / "manifest.json").absolute()
        if not token_overlay_required:
            header = (
                read_shard_header_stable(
                    shard_path,
                    expected_identity=expected_shard_identity,
                )
                if expected_shard_identity is not None
                else read_shard_header(shard_path)
            )
            raw_n_segments = header["__metadata__"].get(MetadataKey.N_SEGMENTS)
            try:
                n_segments = int(raw_n_segments) if raw_n_segments is not None else None
            except ValueError as exc:
                raise FeatureIndexCatalogError(
                    f"certified sparse FeatureIndex shard has invalid "
                    f"n_segments={raw_n_segments!r}: {shard_path}"
                ) from exc
            if n_segments != 0:
                raise FeatureIndexCatalogError(
                    "certified sparse FeatureIndex may omit its token leaf only "
                    f"for a zero-segment shard; got n_segments={n_segments!r}"
                )

        percentile_dir = (
            Path(percentile_overlay_dir).absolute() if percentile_overlay_dir is not None else None
        )
        percentile_leaf = (
            overlay_shard_path(percentile_dir, shard_stem).absolute()
            if percentile_dir is not None
            else None
        )
        paths = {shard_path, token_overlay_dir, token_manifest_path}
        feature_dir_signature_required = expected_shard_identity is None
        if feature_dir_signature_required:
            paths.add(feature_dir)
        if token_overlay_required:
            paths.add(token_leaf)
        if percentile_dir is not None:
            assert percentile_leaf is not None
            if expected_percentile_identity is None:
                paths.add(percentile_dir)
                paths.add((percentile_dir / "manifest.json").absolute())
            paths.add(percentile_leaf)
        catalog = FeatureIndexCatalog(
            schema_version=_CATALOG_SCHEMA_VERSION,
            feature_dir=str(feature_dir),
            shard_entries=((shard_stem, str(shard_path)),),
            segment_size=segment_size,
            percentile_checksum=percentile_checksum,
            percentile_overlay_dir=(str(percentile_dir) if percentile_dir is not None else None),
            percentile_overlay_entries=(
                ((shard_stem, str(percentile_leaf)),) if percentile_leaf is not None else ()
            ),
            tokenizer_checksum=tokenizer_checksum,
            token_overlay_dir=str(token_overlay_dir),
            token_overlay_entries=(
                ((shard_stem, str(token_leaf)),) if token_overlay_required else ()
            ),
            signatures=tuple(ArtifactSignature.capture(path) for path in sorted(paths, key=str)),
            validated_content_fingerprint=validated_content_fingerprint,
            feature_dir_signature_required=feature_dir_signature_required,
            percentile_overlay_dir_signature_required=expected_percentile_identity is None,
        )
        index = cls.from_catalog(
            catalog,
            skip_feature_mags=skip_feature_mags,
            profile_loads=profile_loads,
        )
        if expected_shard_identity is not None:
            index._expected_shard_identities[shard_stem] = expected_shard_identity
        if expected_percentile_identity is not None:
            index._expected_percentile_identities[shard_stem] = expected_percentile_identity
        return index

    def export_catalog(
        self,
        *,
        validated_content_fingerprint: str | None = None,
    ) -> FeatureIndexCatalog:
        """Freeze this fully scanned index for worker construction."""
        if self._feature_dir is None:
            raise FeatureIndexCatalogError("cannot export catalog before FeatureIndex scan")
        if self._required_token_boundary_stems is not None:
            raise FeatureIndexCatalogError(
                "cannot export a FeatureIndex catalog whose token-boundary "
                "overlay intentionally covers only required shard stems"
            )
        shard_entries = tuple(
            (stem, str(path.absolute())) for stem, path in sorted(self._shard_paths.items())
        )
        paths: dict[str, Path] = {str(self._feature_dir.absolute()): self._feature_dir.absolute()}
        for _stem, path_str in shard_entries:
            paths[path_str] = Path(path_str)
        all_files = [Path(path) for _stem, path in shard_entries]
        percentile_entries = self._catalog_overlay_entries(
            self._pct_overlay_dir, all_files, OverlayKind.PERCENTILE
        )
        token_entries = self._catalog_overlay_entries(
            self._tok_overlay_dir, all_files, OverlayKind.SEG_BOUNDARY
        )
        for overlay_dir, entries in (
            (self._pct_overlay_dir, percentile_entries),
            (self._tok_overlay_dir, token_entries),
        ):
            if overlay_dir is None:
                continue
            absolute_dir = overlay_dir.absolute()
            paths[str(absolute_dir)] = absolute_dir
            manifest_path = absolute_dir / "manifest.json"
            paths[str(manifest_path)] = manifest_path
            for _stem, overlay_path_str in entries:
                paths[overlay_path_str] = Path(overlay_path_str)
        expected_path_set = set(paths)
        scanned_path_set = set(self._scan_artifact_signatures)
        if expected_path_set != scanned_path_set:
            raise FeatureIndexCatalogError(
                "FeatureIndex catalog artifact inventory changed after parent scan: "
                f"missing={sorted(scanned_path_set - expected_path_set)}, "
                f"new={sorted(expected_path_set - scanned_path_set)}"
            )
        for signature in self._scan_artifact_signatures.values():
            signature.validate()
        catalog = FeatureIndexCatalog(
            schema_version=_CATALOG_SCHEMA_VERSION,
            feature_dir=str(self._feature_dir.absolute()),
            shard_entries=shard_entries,
            segment_size=self._segment_size,
            percentile_checksum=self._percentile_checksum,
            percentile_overlay_dir=(
                str(self._pct_overlay_dir.absolute()) if self._pct_overlay_dir is not None else None
            ),
            percentile_overlay_entries=percentile_entries,
            tokenizer_checksum=self._tokenizer_checksum,
            token_overlay_dir=(
                str(self._tok_overlay_dir.absolute()) if self._tok_overlay_dir is not None else None
            ),
            token_overlay_entries=token_entries,
            signatures=tuple(self._scan_artifact_signatures[path] for path in sorted(paths)),
            validated_content_fingerprint=validated_content_fingerprint,
        )
        catalog.validate()
        return catalog

    @staticmethod
    def _catalog_overlay_entries(
        overlay_dir: Path | None,
        base_paths: list[Path],
        kind: OverlayKind,
    ) -> tuple[tuple[str, str], ...]:
        if overlay_dir is None:
            return ()
        absolute_dir = overlay_dir.absolute()
        return tuple(
            (stem, str(overlay_shard_path(absolute_dir, stem).absolute()))
            for stem in expected_overlay_shard_stems(base_paths, kind)
        )

    # ------------------------------------------------------------------
    # Pickling support (for multiprocessing with spawn context)
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        # threading objects are not picklable — drop them. The in-flight thread
        # and anything it published go with them: the unpickled index sync-loads
        # or re-prefetches instead.
        state["_prefetch_thread"] = None
        state["_prefetch_result"] = None
        state["_prefetch_stem"] = None
        state["_prefetch_lock"] = None
        state["_stats_lock"] = None
        # The unpickled index is a fresh, open slot. Carrying a True flag across
        # would silently produce an index that can never prefetch again.
        state["_prefetch_closed"] = False
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        self.__dict__.update(state)
        self._prefetch_lock = threading.Lock()
        self._stats_lock = threading.Lock()

    # ------------------------------------------------------------------
    # Public properties
    # ------------------------------------------------------------------

    @property
    def n_shards(self) -> int:
        return len(self._shard_paths)

    @property
    def feature_dir(self) -> Path | None:
        """Directory containing the discovered feature shards.

        Public read-only accessor for the underlying ``_feature_dir`` —
        consumers in other slices (e.g., ``encoder_decoder/dataloader.py``
        for cross-stage segment_size verification) should read this property
        rather than reaching into the private attribute. Returns ``None`` if
        no shards have been scanned yet.

        See slice-10 map §6 P2-A and consolidation manifest §7 slice-10.
        """
        return self._feature_dir

    @property
    def segment_size(self) -> int | None:
        """Segment size from extraction config, or None if no shards found."""
        return self._segment_size

    @property
    def has_char_boundaries(self) -> bool:
        """Whether feature shards include character-level segment boundaries (always True)."""
        return True

    @property
    def has_precomputed_percentiles(self) -> bool:
        """Whether the loaded shard has valid precomputed percentiles.

        Set to True after the first shard load if:
        1. A percentile_checksum was provided at construction.
        2. The shard's stored checksum matches the expected one.
        3. The shard contains a feature_percentiles tensor.
        """
        return self._has_precomputed_percentiles

    @property
    def has_precomputed_token_boundaries(self) -> bool:
        """Whether the loaded shard has valid precomputed token boundaries.

        Set to True after the first shard load if:
        1. A tokenizer_checksum was provided at construction.
        2. The shard's stored checksum matches the expected one.
        3. The shard contains segment_token_starts/segment_token_ends tensors.
        """
        return self._has_precomputed_token_boundaries

    @property
    def shard_stems(self) -> list[str]:
        """Return sorted list of discovered shard stem strings."""
        return list(self._sorted_stems)

    @property
    def max_resident_shards(self) -> int:
        """Resident-set capacity (issue #740). Default 1 == legacy LRU-1."""
        return self._max_resident_shards

    @property
    def cache_stats(self) -> CacheStats:
        """Best-effort diagnostic counters (never a correctness signal)."""
        return self._stats

    @property
    def load_profiles(self) -> list[dict]:
        """Completed diagnostic load records; empty unless ``profile_loads=True``."""
        return [
            {
                **record,
                "detail_seconds": dict(record.get("detail_seconds", {})),
                "detail_cpu_seconds": dict(record.get("detail_cpu_seconds", {})),
            }
            for record in self._load_profiles
        ]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def pin_shard(self, shard_stem: str) -> None:
        """Load a shard and pin it in memory permanently (never evicted by LRU).

        Use this for the validation shard to avoid evicting training shards
        during periodic validation.
        """
        if shard_stem in self._pinned:
            return  # already pinned
        if shard_stem not in self._shard_paths:
            raise KeyError(f"Unknown shard stem: {shard_stem!r}")
        logger.info(f"Pinning shard {shard_stem} in memory")
        self._pinned[shard_stem] = self._load_shard(shard_stem)

    def doc_ids_for_shard(self, shard_stem: str) -> list[int]:
        """Load the shard and return sorted list of doc_ids present in it.

        Raises KeyError if shard_stem is not known.
        """
        if shard_stem not in self._shard_paths:
            raise KeyError(f"Unknown shard stem: {shard_stem!r}")
        docs = self._resident.get(shard_stem)
        if docs is not None:
            self._resident.move_to_end(shard_stem)
            self._stats.hits += 1
        else:
            # Plain sync load after joining in-flight prefetch — preserves the
            # legacy contract (no prefetch consume, no prefetch start).
            self._join_inflight()
            docs = self._load_shard(shard_stem)
            self._admit(shard_stem, docs)
        return sorted(docs)

    def load_id_scan_arrays(
        self,
        shard_stem: str,
        *,
        profile: dict[str, float | int] | None = None,
    ) -> dict:
        """Load only the arrays the startup percentile-ID scan needs.

        Returns ``offsets`` (int64), ``doc_ids`` (int32), delta-decoded
        ``feature_ids`` (int32) plus the ``segment_size``/``n_segments``
        scalars for one shard — no overlay merge, no feature_mags upcast, no
        ``SegmentFeatures`` construction (issue #1495). Structural validation
        matches :meth:`_materialize_data` for the fields read here; corruption
        confined to unread fields is still caught by the unchanged full loader
        on that shard's first access (which may be after optimizer steps).
        """
        if shard_stem not in self._shard_paths:
            raise KeyError(f"Unknown shard stem: {shard_stem!r}")
        if (
            self._transport_materializer is not None
            or self._expected_shard_identities
            or self._required_token_boundary_stems is not None
        ):
            raise RuntimeError(
                "load_id_scan_arrays supports only locally scanned full-family "
                "feature directories; transport/certified/required-stem-scoped "
                "indices must go through lookup()"
            )
        raw = load_features(
            self._shard_paths[shard_stem],
            keys=set(_ID_SCAN_SHARD_KEYS),
            profile=profile,
        )
        _validate_shard_segment_size(shard_stem, int(raw["segment_size"]), self._segment_size)
        offsets = raw["offsets"]
        _validate_shard_lengths(offsets, len(raw["doc_ids"]), int(raw["n_segments"]))
        _validate_feature_csr(shard_stem, offsets, len(raw["feature_ids"]))
        return raw

    def lookup(
        self,
        shard_stem: str,
        doc_idx: int,
    ) -> list[SegmentFeatures]:
        """Look up features for a specific document.

        Raises:
            KeyError: if ``shard_stem`` is not a discovered shard, or if
                ``doc_idx`` is absent from that shard. A present document always
                has >= 1 segment (the extraction pipeline rejects zero-token and
                zero-segment docs), so a missing doc indicates corruption or a
                stale/misaligned parquet shard — never a normal control-flow state.
        """
        if shard_stem not in self._shard_paths:
            raise KeyError(f"Unknown shard stem: {shard_stem!r}")

        # Resolve the per-doc index for this shard (pinned / cache-hit / load).
        if shard_stem in self._pinned:
            docs = self._pinned[shard_stem]
        else:
            docs = self._resident.get(shard_stem)
            if docs is not None:
                self._resident.move_to_end(shard_stem)
                self._stats.hits += 1
            else:
                docs = self._insert_resident(shard_stem)

        # One membership resolution, not two: for the eager dict and for both
        # lazy facades a KeyError out of __getitem__ means exactly "absent", so
        # catching it is equivalent to the former `not in` pre-check while
        # halving the lazy path's searchsorted round-trips on this hot call.
        try:
            return docs[doc_idx]
        except KeyError:
            raise KeyError(
                f"Document {doc_idx} not found in shard {shard_stem!r} (shard has {len(docs)} docs)"
            ) from None

    def ensure_resident(self, shard_stem: str) -> None:
        """Load-if-absent and hold resident (pinned shards are already held).

        Raises KeyError if shard_stem is not known.
        """
        if shard_stem not in self._shard_paths:
            raise KeyError(f"Unknown shard stem: {shard_stem!r}")
        if shard_stem in self._pinned:
            return
        if shard_stem in self._resident:
            self._resident.move_to_end(shard_stem)
            self._stats.hits += 1
            return
        self._insert_resident(shard_stem)

    def drop_except(self, stems: Iterable[str]) -> None:
        """Evict resident entries not in ``stems`` (pinned entries are untouched)."""
        keep = set(stems)
        for stem in list(self._resident):
            if stem not in keep:
                del self._resident[stem]
                self._stats.evictions += 1

    def retain(self, stems: Sequence[str]) -> None:
        """Drop non-listed residents, then seed the first ``capacity`` stems.

        Intended for pre-warming the working set a caller is about to touch
        (e.g., a splice or eval pass over a known shard subset). If ``stems``
        exceeds ``max_resident_shards``, the overflow is not pre-loaded here —
        it will load lazily (with LRU eviction) on first access, which is
        slower but still byte-identical.

        Duplicate stems are deduped (order-preserving) before slicing/counting
        against ``max_resident_shards`` — otherwise a duplicate-laden call
        (e.g. ``["A", "A", "B"]`` at capacity 2) would seed ``A`` twice, never
        seed ``B``, and compare the RAW length against capacity to fire a
        false overflow warning despite only 2 distinct shards (issue #740
        hardening).
        """
        self._stats.retain_calls += 1
        deduped = list(dict.fromkeys(stems))
        self.drop_except(deduped)
        for stem in deduped[: self._max_resident_shards]:
            self.ensure_resident(stem)
        if len(deduped) > self._max_resident_shards:
            logger.warning(
                f"FeatureIndex.retain: working set of {len(deduped)} shards exceeds "
                f"max_resident_shards={self._max_resident_shards}; overflow loads "
                f"lazily with LRU eviction (slower, still byte-identical)."
            )

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _scan(self, feature_dir: str) -> None:
        """Discover shard files, overlays, and read segment_size from the first shard."""
        self._feature_dir = Path(feature_dir)
        all_files = glob_features(feature_dir)
        if not all_files:
            signature = ArtifactSignature.capture(self._feature_dir)
            self._scan_artifact_signatures[signature.path] = signature
            logger.warning(f"No feature shard files found in {feature_dir}")
            return

        for path in all_files:
            shard_stem = shard_stem_from_path(path)
            self._shard_paths[shard_stem] = path

        # Load the first shard briefly to extract segment_size and validate format
        first_path = all_files[0]
        first_signature = ArtifactSignature.capture(first_path)
        scan_started = _time.monotonic() if self._profile_loads else 0.0
        scan_detail: dict[str, float | int] | None = {} if self._profile_loads else None
        scan_cpu_detail: dict[str, float] | None = {} if self._profile_loads else None
        data = (
            load_features(
                first_path,
                profile=scan_detail,
                profile_cpu=scan_cpu_detail,
            )
            if scan_detail is not None
            else load_features(first_path)
        )
        first_signature_after = ArtifactSignature.capture(first_path)
        if first_signature_after != first_signature:
            raise FeatureIndexCatalogError(
                f"FeatureIndex first shard changed while the parent validated it: {first_path}"
            )
        self._segment_size = int(data["segment_size"])
        if scan_detail is not None:
            self._load_profiles.append(
                {
                    "phase": "scan_validation",
                    "shard_stem": shard_stem_from_path(first_path),
                    "total_seconds": _time.monotonic() - scan_started,
                    "compressed_bytes": int(scan_detail.pop("compressed_bytes", 0)),
                    "decompressed_bytes": int(scan_detail.pop("decompressed_bytes", 0)),
                    "n_segments": int(data["n_segments"]),
                    "n_features": int(data["offsets"][-1].item()),
                    "detail_seconds": scan_detail,
                    "detail_cpu_seconds": scan_cpu_detail,
                }
            )

        has_char_boundaries = "segment_char_starts" in data and "segment_char_ends" in data
        assert has_char_boundaries, (
            f"Legacy feature format in {first_path.name}: "
            f"missing segment_char_starts/segment_char_ends. "
            f"Re-extract features with the current pipeline."
        )

        del data  # free immediately

        # Discover percentile overlay
        if self._percentile_checksum is not None:
            self._pct_overlay_dir = discover_overlay(
                self._feature_dir,
                OverlayKind.PERCENTILE,
                self._percentile_checksum,
                base_shard_paths=all_files,
                cache_dir=self._feature_dir,
            )
            if self._pct_overlay_dir is None:
                raise OverlayIncompleteError(
                    "Requested percentile overlay is absent or has no readable "
                    f"matching manifest for checksum {self._percentile_checksum!r}."
                )
            expected_pct_stems = expected_overlay_shard_stems(all_files, OverlayKind.PERCENTILE)
            missing_pct_stems = [
                stem
                for stem in expected_pct_stems
                if not overlay_shard_path(self._pct_overlay_dir, stem).is_file()
            ]
            if missing_pct_stems:
                raise OverlayIncompleteError(
                    f"Requested percentile overlay {self._pct_overlay_dir} is "
                    f"incomplete: missing {len(missing_pct_stems)} expected shard(s)."
                )
            self._has_precomputed_percentiles = True
            logger.info(f"Found percentile overlay: {self._pct_overlay_dir}")

        # Discover token boundary overlay
        if self._tokenizer_checksum is not None:
            self._tok_overlay_dir = discover_overlay(
                self._feature_dir,
                OverlayKind.SEG_BOUNDARY,
                self._tokenizer_checksum,
                base_shard_paths=all_files,
                cache_dir=self._feature_dir,
            )
            if self._tok_overlay_dir is None:
                raise OverlayIncompleteError(
                    "Requested token-boundary overlay is absent or has no readable "
                    f"matching manifest for checksum {self._tokenizer_checksum!r}."
                )
            expected_tok_stems = expected_overlay_shard_stems(all_files, OverlayKind.SEG_BOUNDARY)
            required_tok_stems = self._required_token_boundary_stems
            if required_tok_stems is None:
                stems_to_require = expected_tok_stems
            else:
                unknown_stems = sorted(required_tok_stems - set(expected_tok_stems))
                if unknown_stems:
                    raise OverlayIncompleteError(
                        "Requested token-boundary shard stems are absent from the "
                        f"feature family: {unknown_stems}"
                    )
                stems_to_require = sorted(required_tok_stems)
            missing_tok_stems = [
                stem
                for stem in stems_to_require
                if not overlay_shard_path(self._tok_overlay_dir, stem).is_file()
            ]
            if missing_tok_stems:
                raise OverlayIncompleteError(
                    f"Requested token-boundary overlay {self._tok_overlay_dir} is "
                    f"incomplete: missing {len(missing_tok_stems)} expected shard(s)."
                )
            self._has_precomputed_token_boundaries = True
            coverage = (
                "all feature shards"
                if required_tok_stems is None
                else f"required stems {sorted(required_tok_stems)}"
            )
            logger.info(
                f"Found token boundary overlay: {self._tok_overlay_dir} (validated {coverage})"
            )

        logger.info(
            f"FeatureIndex scanned {len(self._shard_paths)} shards "
            f"in {feature_dir} (segment_size={self._segment_size})"
        )

        # Build sorted stem index for prefetch prediction
        self._sorted_stems = sorted(self._shard_paths.keys())
        self._stem_to_idx = {stem: i for i, stem in enumerate(self._sorted_stems)}

        # Freeze every artifact whose inventory or metadata the scan accepted. The first
        # shard's pre/post guard above binds its full-body validation to this snapshot.
        catalog_paths: set[Path] = {self._feature_dir.absolute()}
        catalog_paths.update(path.absolute() for path in all_files)
        for overlay_dir, kind in (
            (self._pct_overlay_dir, OverlayKind.PERCENTILE),
            (self._tok_overlay_dir, OverlayKind.SEG_BOUNDARY),
        ):
            if overlay_dir is None:
                continue
            absolute_dir = overlay_dir.absolute()
            catalog_paths.add(absolute_dir)
            catalog_paths.add(absolute_dir / "manifest.json")
            overlay_stems = expected_overlay_shard_stems(all_files, kind)
            if kind is OverlayKind.SEG_BOUNDARY and self._required_token_boundary_stems is not None:
                overlay_stems = sorted(self._required_token_boundary_stems)
            for stem in overlay_stems:
                catalog_paths.add(overlay_shard_path(absolute_dir, stem).absolute())
        signatures = {
            str(path): ArtifactSignature.capture(path) for path in sorted(catalog_paths, key=str)
        }
        if signatures[str(first_path.absolute())] != first_signature:
            raise FeatureIndexCatalogError(
                f"FeatureIndex first shard changed after the parent validated it: {first_path}"
            )
        self._scan_artifact_signatures = signatures

    def _admit(
        self,
        shard_stem: str,
        docs: Mapping[int, list[SegmentFeatures]],
    ) -> None:
        """Insert as MRU, evicting the LRU resident while over capacity."""
        self._resident[shard_stem] = docs
        self._resident.move_to_end(shard_stem)
        while len(self._resident) > self._max_resident_shards:
            self._resident.popitem(last=False)
            self._stats.evictions += 1

    def _insert_resident(self, shard_stem: str) -> Mapping[int, list[SegmentFeatures]]:
        """Miss path: prefetch-consume or sync-load, admit, refill the pipeline."""
        if self._prefetch:
            docs = self._resolve_prefetch_or_load(shard_stem)
        else:
            docs = self._load_shard(shard_stem)
        self._admit(shard_stem, docs)
        if self._prefetch:
            self._schedule_prefetch(shard_stem)
        return docs

    def _load_shard(self, shard_stem: str) -> Mapping[int, list[SegmentFeatures]]:
        """Load a single shard and build its per-doc index (vectorized).

        Two phases: :meth:`_materialize_data` decompresses + overlay-merges + validates the raw
        contiguous arrays; :meth:`_build_docs` slices them into per-segment ``SegmentFeatures``.

        With a shared source store (issue #740 PR-F) each stem's materialized arrays live once
        on ``/dev/shm`` and are mapped shared across workers, and the per-segment features are
        zero-copy VIEWS into that shared buffer instead of independent copies — byte-identical
        (same bytes), so all workers reference one physical copy per stem. Without a store the
        per-segment slices are cloned so the contiguous arrays can be freed (unchanged path).
        """
        required_tok_stems = self._required_token_boundary_stems
        if (
            self._tok_overlay_dir is not None
            and required_tok_stems is not None
            and shard_stem not in required_tok_stems
        ):
            raise OverlayIncompleteError(
                "This FeatureIndex intentionally validated token boundaries only "
                f"for required stems {sorted(required_tok_stems)}; refusing to "
                f"load out-of-scope shard {shard_stem!r}. Construct a full-coverage "
                "index for any caller that needs other shards."
            )
        _t0 = _time.monotonic()
        detail: dict[str, float | int] | None = {} if self._profile_loads else None
        detail_cpu: dict[str, float] | None = {} if self._profile_loads else None
        store = self._shared_store
        if store is None:
            materialize_started = _time.monotonic() if detail is not None else 0.0
            materialize_cpu_started = _time.thread_time() if detail_cpu is not None else 0.0
            data = self._materialize_data(
                shard_stem,
                profile_detail=detail,
                profile_cpu_detail=detail_cpu,
            )
            if detail is not None:
                detail["materialize_total"] = _time.monotonic() - materialize_started
            if detail_cpu is not None:
                detail_cpu["materialize_total"] = _time.thread_time() - materialize_cpu_started
            doc_index_started = _time.monotonic() if detail is not None else 0.0
            doc_index_cpu_started = _time.thread_time() if detail_cpu is not None else 0.0
            docs = _LazyShardDocs(data) if self._lazy_docs else self._build_docs(data, copy=True)
            if detail is not None:
                detail["doc_index"] = _time.monotonic() - doc_index_started
            if detail_cpu is not None:
                detail_cpu["doc_index"] = _time.thread_time() - doc_index_cpu_started
        else:
            data = self._shared_get_or_materialize(store, shard_stem)
            doc_index_started = _time.monotonic() if detail is not None else 0.0
            doc_index_cpu_started = _time.thread_time() if detail_cpu is not None else 0.0
            docs = (
                _SharedLazyDocs(data)
                if self._lazy_shared_docs
                else self._build_docs(data, copy=False)
            )
            if detail is not None:
                detail["shared_materialize_and_doc_index"] = _time.monotonic() - doc_index_started
            if detail_cpu is not None:
                detail_cpu["shared_materialize_and_doc_index"] = (
                    _time.thread_time() - doc_index_cpu_started
                )
        total_seconds = _time.monotonic() - _t0
        with self._stats_lock:
            self._stats.load_seconds += total_seconds
        if detail is not None:
            record = {
                "phase": "resident_load",
                "shard_stem": shard_stem,
                "total_seconds": total_seconds,
                "compressed_bytes": int(detail.pop("compressed_bytes", 0)),
                "decompressed_bytes": int(detail.pop("decompressed_bytes", 0)),
                "n_segments": int(data["n_segments"]),
                "n_features": int(data["offsets"][-1].item()),
                "n_docs": len(docs),
                "detail_seconds": detail,
                "detail_cpu_seconds": detail_cpu,
            }
            with self._stats_lock:
                self._load_profiles.append(record)
        return docs

    def _shared_get_or_materialize(self, store: SharedStemStore, shard_stem: str) -> dict:
        """Return a stem's arrays from the shared store, materializing on a miss. Robust to the
        eviction race: a stem present at ``has()`` can be evicted before ``get()`` reads it (only
        below a pathologically small capacity, since active stems are MRU) — on that KeyError we
        re-materialize. Falls back to a private (unshared) materialize if the store keeps evicting.
        """
        # Fast path: shared hit.
        if store.has(shard_stem):
            try:
                data = store.get(shard_stem)
                with self._stats_lock:
                    self._stats.shared_hits += 1
                    self._stats.shared_mapped_bytes += sum(
                        value.numel() * value.element_size()
                        for value in data.values()
                        if isinstance(value, t.Tensor)
                    )
                return data
            except KeyError:
                with self._stats_lock:
                    self._stats.shared_get_races += 1
                # Evicted between has() and get() — materialize below.
        with self._stats_lock:
            self._stats.shared_misses += 1
        # Miss: every possible materialization, INCLUDING the peer-publication race fallback,
        # stays under the cross-process lock. A peer may publish while we wait, so first attempt
        # to map its copy inside the lease. If that copy races with eviction, build exactly one
        # private fallback while the lease is still held. This makes "one source decompression
        # transient at a time" a real resource bound rather than a best-effort common case.
        lock_wait_started = _time.monotonic()
        with store.materialize_lock():
            with self._stats_lock:
                self._stats.materialize_lock_wait_seconds += _time.monotonic() - lock_wait_started
            if store.has(shard_stem):
                try:
                    data = store.get(shard_stem)
                    with self._stats_lock:
                        self._stats.shared_hits += 1
                        self._stats.shared_mapped_bytes += sum(
                            value.numel() * value.element_size()
                            for value in data.values()
                            if isinstance(value, t.Tensor)
                        )
                    return data
                except KeyError:
                    with self._stats_lock:
                        self._stats.shared_get_races += 1
                    # The peer's copy vanished despite the inside-lock recheck
                    # (e.g. an explicit external evict). Keep the fallback
                    # private rather than attempting to replace an ambiguous
                    # store entry; most importantly, build it under this lock.
                    materialize_started = _time.monotonic()
                    materialized = self._materialize_data(shard_stem)
                    self._attach_shared_doc_index(materialized)
                    with self._stats_lock:
                        self._stats.materializations += 1
                        self._stats.materialize_seconds += _time.monotonic() - materialize_started
                        self._stats.private_fallbacks += 1
                    return materialized

            materialize_started = _time.monotonic()
            materialized = self._materialize_data(shard_stem)
            self._attach_shared_doc_index(materialized)
            with self._stats_lock:
                self._stats.materializations += 1
                self._stats.materialize_seconds += _time.monotonic() - materialize_started
            tensors = {k: v for k, v in materialized.items() if isinstance(v, t.Tensor)}
            scalars = {k: v for k, v in materialized.items() if isinstance(v, int)}
            if not store.put(shard_stem, tensors, scalars):
                with self._stats_lock:
                    self._stats.publish_declines += 1
                    self._stats.private_fallbacks += 1
                return materialized  # space guard declined ⇒ private copy, no re-materialize
            with self._stats_lock:
                self._stats.publications += 1
            # Map before releasing the materialization lease. Every production
            # eviction is driven by a put under this same lease, so the newly
            # published stem cannot be evicted between publication and mapping.
            try:
                data = store.get(shard_stem)
            except KeyError:
                with self._stats_lock:
                    self._stats.shared_get_races += 1
                    self._stats.private_fallbacks += 1
                return materialized
            with self._stats_lock:
                self._stats.shared_hits += 1
                self._stats.shared_mapped_bytes += sum(
                    value.numel() * value.element_size()
                    for value in data.values()
                    if isinstance(value, t.Tensor)
                )
            return data

    def _attach_shared_doc_index(self, data: dict) -> None:
        """Add the exact eager document order as fixed-size shared tensors.

        The locator is built only by the store materializer.  Store hits map
        the published tensors and therefore never repeat the full-stem sort.
        Its two tensors each have exactly ``n_segments`` int64 elements, which
        gives resource admission a header-derived fixed payload bound.
        """
        if not self._lazy_shared_docs:
            return
        locator_keys = {
            _SHARED_DOC_ORDER_KEY,
            _SHARED_SORTED_DOC_IDS_KEY,
            _SHARED_DOC_COUNT_KEY,
        }
        present = locator_keys.intersection(data)
        if present:
            if present != locator_keys:
                raise RuntimeError(
                    f"shared lazy FeatureIndex locator is only partially present: {sorted(present)}"
                )
            return

        n_segments = int(data["n_segments"])
        if n_segments == 0:
            data[_SHARED_DOC_ORDER_KEY] = t.empty(0, dtype=t.int64)
            data[_SHARED_SORTED_DOC_IDS_KEY] = t.empty(0, dtype=t.int64)
            data[_SHARED_DOC_COUNT_KEY] = 0
            return

        doc_ids_i64 = data["doc_ids"].to(t.int64)
        segment_indices_i64 = data["segment_indices"].to(t.int64)
        max_seg_idx = int(segment_indices_i64.max().item()) + 1
        sort_key = doc_ids_i64 * max_seg_idx + segment_indices_i64
        del segment_indices_i64
        order = sort_key.argsort(stable=True)
        del sort_key
        sorted_doc_ids = doc_ids_i64[order]
        del doc_ids_i64
        doc_count = 1 + int(t.count_nonzero(sorted_doc_ids[1:] != sorted_doc_ids[:-1]).item())
        data[_SHARED_DOC_ORDER_KEY] = order
        data[_SHARED_SORTED_DOC_IDS_KEY] = sorted_doc_ids
        data[_SHARED_DOC_COUNT_KEY] = doc_count

    def _materialize_data(
        self,
        shard_stem: str,
        *,
        profile_detail: dict[str, float | int] | None = None,
        profile_cpu_detail: dict[str, float] | None = None,
    ) -> dict:
        materializer = self._transport_materializer
        if materializer is None:
            return self._materialize_data_local(
                shard_stem,
                profile_detail=profile_detail,
                profile_cpu_detail=profile_cpu_detail,
            )
        expected_raw = self._shard_paths[shard_stem]
        assert self._pct_overlay_dir is not None
        expected_pct = overlay_shard_path(self._pct_overlay_dir, shard_stem)
        with materializer.materialize(shard_stem) as staged:
            if staged.raw_path != expected_raw or staged.percentile_path != expected_pct:
                raise FeatureIndexCatalogError(
                    f"FeatureIndex materializer returned noncanonical paths for {shard_stem!r}"
                )
            self._expected_shard_identities[shard_stem] = staged.raw_identity
            self._expected_percentile_identities[shard_stem] = staged.percentile_identity
            try:
                return self._materialize_data_local(
                    shard_stem,
                    profile_detail=profile_detail,
                    profile_cpu_detail=profile_cpu_detail,
                )
            finally:
                self._expected_shard_identities.pop(shard_stem, None)
                self._expected_percentile_identities.pop(shard_stem, None)

    def _materialize_data_local(
        self,
        shard_stem: str,
        *,
        profile_detail: dict[str, float | int] | None = None,
        profile_cpu_detail: dict[str, float] | None = None,
    ) -> dict:
        """Decompress + overlay-merge + validate one shard's raw arrays (the load phase).

        Returns a dict of contiguous tensors plus scalar metadata (``segment_size``,
        ``n_segments``). Overlay presence is encoded by key presence: ``feature_percentiles``
        (pct overlay) and ``seg_token_starts``/``seg_token_ends`` (token overlay) appear only
        when discovered, and ``feature_mags`` is omitted on a ``skip_feature_mags`` index. Shared
        by the normal and shared-store load paths, so both build from identical arrays.
        """
        with self._stats_lock:
            self._stats.loads += 1
        shard_path = self._shard_paths[shard_stem]
        try:
            shard_bytes = shard_path.stat().st_size
        except OSError:
            pass
        else:
            with self._stats_lock:
                self._stats.bytes_read += shard_bytes
        skip_feature_mags = self._skip_feature_mags
        # O4 (issue #531): on a skip_feature_mags index, materialize only the segmentation +
        # feature_ids fields, skipping the large feature_mags upcast + split (the packed-rows
        # consumer never reads mags). keys=None on the live path => unchanged full load.
        load_keys = set(_NO_MAGS_SHARD_KEYS) if skip_feature_mags else None
        expected_identity = self._expected_shard_identities.get(shard_stem)
        if expected_identity is not None:
            raw = (
                load_features_stable(
                    shard_path,
                    expected_identity=expected_identity,
                    keys=load_keys,
                    profile=profile_detail,
                    profile_cpu=profile_cpu_detail,
                )
                if profile_detail is not None or profile_cpu_detail is not None
                else load_features_stable(
                    shard_path,
                    expected_identity=expected_identity,
                    keys=load_keys,
                )
            )
        else:
            raw = (
                load_features(
                    shard_path,
                    keys=load_keys,
                    profile=profile_detail,
                    profile_cpu=profile_cpu_detail,
                )
                if profile_detail is not None or profile_cpu_detail is not None
                else load_features(shard_path, keys=load_keys)
            )

        # Validate segment_size matches the one from the first shard
        postprocess_started = _time.monotonic() if profile_detail is not None else 0.0
        postprocess_cpu_started = _time.thread_time() if profile_cpu_detail is not None else 0.0
        shard_seg_size = int(raw["segment_size"])
        _validate_shard_segment_size(shard_stem, shard_seg_size, self._segment_size)
        n_segments = int(raw["n_segments"])
        # feature_ids arrive int32 (parquet, delta-decoded). On a compact index, narrow to the
        # smallest LOSSLESS dtype (issue #740 PR-E2 int32 → PR-H uint16): a 65k SAE's ids are all
        # in [0, 65535] and fit uint16, halving the biggest resident tensor again (3.69 → 1.85 GB
        # /stem), which lets the shared /dev/shm store hold ~2× the stems (no eviction thrash at
        # higher worker counts). Guard by the ACTUAL max so a wider SAE (>65536 features) safely
        # stays int32. Byte-identical: the splice path converts feature_ids to int64 on use.
        feature_ids = raw["feature_ids"]
        if self._compact_feature_ids:
            if (
                feature_ids.numel()
                and int(feature_ids.min()) >= 0
                and int(feature_ids.max()) < 65536
            ):
                feature_ids = feature_ids.to(t.uint16)
            # else: negative/out-of-range IDs retain the legacy compact int32
            # representation; never silently wrap corrupt or wider-SAE values.
        data: dict = {
            "offsets": raw["offsets"],
            "doc_ids": raw["doc_ids"],
            "segment_indices": raw["segment_indices"],
            "segment_positions": raw["segment_positions"],
            "feature_ids": feature_ids,
            "segment_char_starts": raw["segment_char_starts"],
            "segment_char_ends": raw["segment_char_ends"],
            "segment_size": shard_seg_size,
            "n_segments": n_segments,
        }
        if not skip_feature_mags:
            data["feature_mags"] = raw["feature_mags"]

        offsets = data["offsets"]
        _validate_shard_lengths(offsets, len(data["doc_ids"]), n_segments)
        assert len(data["segment_indices"]) == n_segments
        assert len(data["segment_positions"]) == n_segments
        if profile_detail is not None:
            profile_detail["base_postprocess"] = _time.monotonic() - postprocess_started
        if profile_cpu_detail is not None:
            profile_cpu_detail["base_postprocess"] = _time.thread_time() - postprocess_cpu_started

        # Percentile coverage includes zero-segment shards, so validate it
        # before the early return below. A requested overlay never falls back
        # shard-by-shard after scan-time coverage succeeded.
        if self._pct_overlay_dir is not None:
            percentile_started = _time.monotonic() if profile_detail is not None else 0.0
            percentile_cpu_started = _time.thread_time() if profile_cpu_detail is not None else 0.0
            percentile_path = overlay_shard_path(self._pct_overlay_dir, shard_stem)
            percentile_identity = self._expected_percentile_identities.get(shard_stem)
            data["feature_percentiles"] = (
                validate_percentile_overlay_shard_stable(
                    percentile_path,
                    expected_numel=len(data["feature_ids"]),
                    expected_identity=percentile_identity,
                )
                if percentile_identity is not None
                else validate_percentile_overlay_shard(
                    percentile_path,
                    expected_numel=len(data["feature_ids"]),
                )
            )
            if profile_detail is not None:
                profile_detail["percentile_overlay"] = _time.monotonic() - percentile_started
            if profile_cpu_detail is not None:
                profile_cpu_detail["percentile_overlay"] = (
                    _time.thread_time() - percentile_cpu_started
                )

        if n_segments == 0:
            return data

        # Vectorized offset validation (replaces per-element assertions)
        validation_started = _time.monotonic() if profile_detail is not None else 0.0
        validation_cpu_started = _time.thread_time() if profile_cpu_detail is not None else 0.0
        _validate_feature_csr(shard_stem, offsets, len(data["feature_ids"]))
        if profile_detail is not None:
            profile_detail["array_validation"] = _time.monotonic() - validation_started
        if profile_cpu_detail is not None:
            profile_cpu_detail["array_validation"] = _time.thread_time() - validation_cpu_started

        # Load precomputed token boundaries from overlay (if discovered during _scan)
        if self._tok_overlay_dir is not None:
            token_started = _time.monotonic() if profile_detail is not None else 0.0
            token_cpu_started = _time.thread_time() if profile_cpu_detail is not None else 0.0
            starts, ends = validate_token_boundary_overlay_shard(
                overlay_shard_path(self._tok_overlay_dir, shard_stem),
                expected_n_segments=n_segments,
            )
            data["seg_token_starts"] = starts
            data["seg_token_ends"] = ends
            if profile_detail is not None:
                profile_detail["token_boundary_overlay"] = _time.monotonic() - token_started
            if profile_cpu_detail is not None:
                profile_cpu_detail["token_boundary_overlay"] = (
                    _time.thread_time() - token_cpu_started
                )
        return data

    def _build_docs(self, data: dict, *, copy: bool) -> dict[int, list[SegmentFeatures]]:
        """Slice materialized arrays into per-doc ``SegmentFeatures`` (vectorized).

        ``copy=True`` clones each per-segment slice into independent storage (the normal path, so
        the contiguous arrays can be freed afterwards). ``copy=False`` returns zero-copy VIEWS into
        ``data``'s arrays — used with the shared ``/dev/shm`` store so all workers alias one
        physical copy per stem; the views keep the shared buffer alive while the shard is resident.
        """
        n_segments = int(data["n_segments"])
        if n_segments == 0:
            return {}
        offsets = data["offsets"]
        doc_ids = data["doc_ids"]
        segment_indices = data["segment_indices"]
        segment_positions = data["segment_positions"]
        feature_ids = data["feature_ids"]
        feature_mags = data.get("feature_mags")
        segment_char_starts = data["segment_char_starts"]
        segment_char_ends = data["segment_char_ends"]
        feature_percentiles = data.get("feature_percentiles")
        use_pcts = feature_percentiles is not None
        seg_token_starts = data.get("seg_token_starts")
        seg_token_ends = data.get("seg_token_ends")
        use_tok_bounds = seg_token_starts is not None and seg_token_ends is not None

        # Bulk-convert metadata to Python lists — one C++→Python crossing each,
        # instead of n_segments × 7 individual .item() calls.
        doc_ids_list: list[int] = doc_ids.tolist()
        seg_indices_list: list[int] = segment_indices.tolist()
        seg_positions_list: list[int] = segment_positions.tolist()
        offsets_list: list[int] = offsets.tolist()
        char_starts_list: list[int] = segment_char_starts.tolist()
        char_ends_list: list[int] = segment_char_ends.tolist()
        tok_starts_list: list[int | None]
        tok_ends_list: list[int | None]
        if use_tok_bounds:
            assert seg_token_starts is not None and seg_token_ends is not None
            tok_starts_list = seg_token_starts.tolist()
            tok_ends_list = seg_token_ends.tolist()
        else:
            tok_starts_list = [None] * n_segments
            tok_ends_list = [None] * n_segments

        # Sort by (doc_id, segment_idx) via compound argsort — builds lists
        # in final order, eliminating the per-doc .sort() pass.
        max_seg_idx = max(seg_indices_list) + 1
        sort_key = doc_ids.to(t.int64) * max_seg_idx + segment_indices.to(t.int64)
        sorted_order: list[int] = sort_key.argsort(stable=True).tolist()

        # Split each contiguous array into per-segment pieces. copy=True ⇒ independent storage
        # (split_with_sizes_copy, so the source can be freed); copy=False ⇒ zero-copy views
        # (split_with_sizes) that alias the shared /dev/shm buffer. Issue #740: a compact index
        # already narrowed feature_ids to uint16/int32 in _materialize_data (the splice path
        # converts on use), so here we only upcast to int64 for a NON-compact index — avoiding the
        # int64 doubling. split_with_sizes_copy has an out= overload so its stub return is
        # tuple|None; assert-narrow it (out is never passed ⇒ always a sequence).
        sizes: list[int] = [offsets_list[i + 1] - offsets_list[i] for i in range(n_segments)]
        compact_feature_ids = self._compact_feature_ids
        fid_source = feature_ids if compact_feature_ids else feature_ids.to(t.int64)
        split = t.split_with_sizes_copy if copy else t.split_with_sizes
        fid_parts = split(fid_source, sizes)
        assert fid_parts is not None
        mag_parts: Sequence[t.Tensor] | None
        if feature_mags is None:
            mag_parts = None
        else:
            mag_parts = split(feature_mags, sizes)
        pct_parts: Sequence[t.Tensor] | None = None
        if use_pcts:
            assert feature_percentiles is not None
            pct_parts = split(feature_percentiles, sizes)

        shard_docs: dict[int, list[SegmentFeatures]] = {}
        for seg_i in sorted_order:
            doc_id = doc_ids_list[seg_i]
            seg_feat = SegmentFeatures(
                segment_idx=seg_indices_list[seg_i],
                position=seg_positions_list[seg_i],
                feature_ids=fid_parts[seg_i],
                feature_mags=_EMPTY_MAGS if mag_parts is None else mag_parts[seg_i],
                char_start=char_starts_list[seg_i],
                char_end=char_ends_list[seg_i],
                feature_percentiles=pct_parts[seg_i] if pct_parts is not None else None,
                token_start=tok_starts_list[seg_i],
                token_end=tok_ends_list[seg_i],
            )
            if doc_id not in shard_docs:
                shard_docs[doc_id] = []
            shard_docs[doc_id].append(seg_feat)
        return shard_docs

    # ------------------------------------------------------------------
    # Single-slot background prefetch.
    #
    # Single-consumer contract: ``_schedule_prefetch``, ``_resolve_prefetch_or_load``
    # and ``_admit`` all run on the one thread that calls ``lookup()`` /
    # ``ensure_resident()`` / ``doc_ids_for_shard()``, and each of them joins the
    # previous worker before touching the slot, so at most one worker exists at a
    # time. The worker only loads and then publishes under the lock.
    # ------------------------------------------------------------------

    def _resolve_prefetch_or_load(self, shard_stem: str) -> Mapping[int, list[SegmentFeatures]]:
        """Consume the prefetch slot if it holds this stem, otherwise sync load."""
        self._join_inflight()
        with self._prefetch_lock:
            hit = self._prefetch_stem == shard_stem and self._prefetch_result is not None
            result = self._prefetch_result if hit else None
            stale_stem = None if hit else self._prefetch_stem
            # Empty the slot either way. On a miss its map is about to become
            # unreachable anyway (this stem is admitted next), and releasing it
            # *before* the sync load is what keeps the documented
            # "max_resident_shards + 1" bound true across the transient in which
            # the sync load's own map also exists.
            self._prefetch_result = None
            self._prefetch_stem = None

        if result is not None:
            logger.debug(f"Prefetch hit for {shard_stem}")
            return result
        if stale_stem is not None:
            logger.debug(f"Prefetch miss: had {stale_stem}, needed {shard_stem}")
        else:
            logger.debug(f"Prefetch miss: no slot for {shard_stem}")
        # A failed background load published nothing; this sync load re-raises
        # its real error on the consumer thread.
        return self._load_shard(shard_stem)

    def _schedule_prefetch(self, current_stem: str) -> None:
        """Start the background load of the shard after ``current_stem``."""
        self._join_inflight()  # One worker at a time; never leak a thread.
        idx = self._stem_to_idx.get(current_stem)
        if idx is None or idx + 1 >= len(self._sorted_stems):
            return  # Unknown stem / zero-shard scan / no next shard.
        next_stem = self._sorted_stems[idx + 1]
        if next_stem in self._resident or next_stem in self._pinned:
            return  # Already held by the cache.
        with self._prefetch_lock:
            if self._prefetch_closed:
                return
            # Spawn under the lock so a concurrent ``close()`` either refuses
            # this worker or is guaranteed to see (and join) its thread handle.
            thread = threading.Thread(
                target=self._prefetch_worker,
                args=(next_stem,),
                daemon=True,
                name=f"FeatureIndex-prefetch-{next_stem}",
            )
            self._prefetch_thread = thread
            thread.start()

    def _prefetch_worker(self, shard_stem: str) -> None:
        """Load one stem in the background and publish it into the slot."""
        try:
            result = self._load_shard(shard_stem)
        except Exception:
            # Log at failure time on the worker: a stem that is never consumed
            # (last shard, teardown, reposition) must still report loudly. The
            # consumer's sync load re-raises the real error on its own thread.
            logger.opt(exception=True).warning(f"Prefetch of {shard_stem} failed")
            return
        with self._prefetch_lock:
            if not self._prefetch_closed:
                self._prefetch_result = result
                self._prefetch_stem = shard_stem

    def _join_inflight(self) -> None:
        """Join the in-flight worker, leaving a published result intact.

        ``doc_ids_for_shard`` relies on exactly this: no prefetch consume, no
        prefetch start.
        """
        thread = self._prefetch_thread
        if thread is not None:
            if thread is not threading.current_thread():
                thread.join()
            self._prefetch_thread = None

    def _shutdown_prefetch(self) -> None:
        """Close the slot: refuse further publishes, join, clear. Idempotent.

        May join one mid-decompress load (bounded seconds).
        """
        with self._prefetch_lock:
            self._prefetch_closed = True
        self._join_inflight()
        with self._prefetch_lock:
            self._prefetch_result = None
            self._prefetch_stem = None

    def close(self) -> None:
        """Shut down background prefetch. Safe to call multiple times."""
        # ``__del__`` also runs on an instance whose ``__init__`` raised before
        # the lock existed (e.g. a rejected constructor argument), so probe the
        # lock rather than assuming a fully initialized object.
        if getattr(self, "_prefetch_lock", None) is None:
            return
        self._shutdown_prefetch()

    def __del__(self) -> None:
        self.close()
