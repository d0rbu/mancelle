"""Reader for the splice `_splice_provenance` sidecar (issue #596).

Provides the training-time consumer of schema-v2 sidecars emitted by
``steering.feature_shard_splice.build_provenance``: per derived doc, the
splice positions and the derived location of each would-have-followed
segment. Every inconsistency raises :class:`SpliceProvenanceError` (D1/D11 —
no silent fallback anywhere).
"""

from __future__ import annotations

import json
import time as _time
from array import array
from collections import OrderedDict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

from beartype import beartype
from loguru import logger

from steering.conditional.feature_index import CacheStats
from steering.feature_shard_splice import PROVENANCE_DIRNAME
from steering.refined_types import PositiveIntLike

_RESPLICE_HINT = (
    "Re-run scripts.preprocess.splice_feature_shards on the base feature dir "
    "with skip_existing=false to regenerate schema-v2 sidecars, and point "
    "feature_dir at a spliced derived directory."
)


class SpliceProvenanceError(RuntimeError):
    """A _splice_provenance sidecar is missing, outdated, or inconsistent."""


@dataclass(frozen=True)
class SpliceRef:
    """One splice position: derived position + would-have-followed identities."""

    pos: int
    whf_base: tuple[int, int, int]  # [base_shard_idx, doc_idx, seg_ordinal]
    whf_derived: tuple[int, int]    # [derived_doc_idx, derived_position]


@dataclass(frozen=True)
class _ShardSplices:
    n_docs: int
    n_splice_positions: int
    n_segments_by_doc: tuple[int, ...]
    splices_by_doc: dict[int, dict[int, SpliceRef]]  # only docs with splices


@dataclass
class _CompactShardSplices:
    """Array-backed sidecar plus lazy, API-compatible per-document maps.

    The reference JSON parser remains the validation oracle. This representation
    only compacts its already-validated result, so validation order and messages
    cannot diverge. A requested non-empty doc map is materialized once and then
    returned by identity, preserving the reference index's mutation semantics.
    """

    n_docs: int
    n_splice_positions: int
    n_segments_by_doc: array[int]
    doc_splice_offsets: array[int]
    positions: array[int]
    whf_base_shard: array[int]
    whf_base_doc: array[int]
    whf_base_ordinal: array[int]
    whf_derived_doc: array[int]
    whf_derived_position: array[int]
    materialized_docs: dict[int, dict[int, SpliceRef]]

    @classmethod
    def from_reference(cls, shard: _ShardSplices) -> _CompactShardSplices:
        doc_splice_offsets = array("q", [0])
        positions = array("q")
        whf_base_shard = array("q")
        whf_base_doc = array("q")
        whf_base_ordinal = array("q")
        whf_derived_doc = array("q")
        whf_derived_position = array("q")
        for doc_idx in range(shard.n_docs):
            for ref in shard.splices_by_doc.get(doc_idx, {}).values():
                positions.append(ref.pos)
                whf_base_shard.append(ref.whf_base[0])
                whf_base_doc.append(ref.whf_base[1])
                whf_base_ordinal.append(ref.whf_base[2])
                whf_derived_doc.append(ref.whf_derived[0])
                whf_derived_position.append(ref.whf_derived[1])
            doc_splice_offsets.append(len(positions))
        if len(positions) != shard.n_splice_positions:
            raise AssertionError(
                "compact sidecar conversion count diverged from validated reference: "
                f"{len(positions)} != {shard.n_splice_positions}"
            )
        return cls(
            n_docs=shard.n_docs,
            n_splice_positions=shard.n_splice_positions,
            n_segments_by_doc=array("q", shard.n_segments_by_doc),
            doc_splice_offsets=doc_splice_offsets,
            positions=positions,
            whf_base_shard=whf_base_shard,
            whf_base_doc=whf_base_doc,
            whf_base_ordinal=whf_base_ordinal,
            whf_derived_doc=whf_derived_doc,
            whf_derived_position=whf_derived_position,
            materialized_docs={},
        )

    def doc_splices(self, doc_idx: int) -> dict[int, SpliceRef]:
        start = self.doc_splice_offsets[doc_idx]
        end = self.doc_splice_offsets[doc_idx + 1]
        if start == end:
            # The reference ``dict.get(doc_idx, {})`` returns a fresh empty mapping.
            return {}
        cached = self.materialized_docs.get(doc_idx)
        if cached is not None:
            return cached
        result: dict[int, SpliceRef] = {}
        for i in range(start, end):
            pos = self.positions[i]
            result[pos] = SpliceRef(
                pos=pos,
                whf_base=(
                    self.whf_base_shard[i],
                    self.whf_base_doc[i],
                    self.whf_base_ordinal[i],
                ),
                whf_derived=(
                    self.whf_derived_doc[i],
                    self.whf_derived_position[i],
                ),
            )
        self.materialized_docs[doc_idx] = result
        return result


class SpliceProvenanceIndex:
    """Lazy, bounded-LRU-cached reader of schema-v2 splice sidecars.

    Holds a bounded number of parsed per-shard sidecars resident at once (a
    capacity-``N`` LRU keyed by shard stem, MRU at the end). Construct with
    ``max_resident_shards`` to raise the working set above the default of 1
    shard (issue #740); the default reproduces the original LRU-1 cache
    byte-for-byte: exactly one shard resident, switching shards evicts the
    old one and reparses the new one. Unlike ``FeatureIndex`` this class has
    no threads, no pickling support, and no pinning — every load is
    synchronous and the resident set is a plain ``OrderedDict``.

    ``retain()``/``ensure_resident()``/``drop_except()`` give callers
    explicit control over the resident working set, mirroring
    ``FeatureIndex``. ``cache_stats`` exposes best-effort diagnostic counters
    (loads, hits, evictions, retain_calls) — never a correctness signal.
    """

    @beartype
    def __init__(
        self,
        feature_dir: str | Path,
        *,
        max_resident_shards: PositiveIntLike = 1,
        compact: bool = False,
    ) -> None:
        self._dir = Path(feature_dir) / PROVENANCE_DIRNAME
        if not self._dir.is_dir():
            raise SpliceProvenanceError(
                f"splice provenance dir not found: {self._dir}. "
                f"feature_delta.mode=splice_counterfactual requires a spliced "
                f"derived feature directory. {_RESPLICE_HINT}"
            )
        self._max_resident_shards = max_resident_shards
        self._compact = compact
        self._resident: OrderedDict[
            str, _ShardSplices | _CompactShardSplices
        ] = OrderedDict()
        self._stats = CacheStats()

    # -- properties -----------------------------------------------------------

    @property
    def max_resident_shards(self) -> int:
        """Resident-set capacity (issue #740). Default 1 == legacy LRU-1."""
        return self._max_resident_shards

    @property
    def cache_stats(self) -> CacheStats:
        """Best-effort diagnostic counters (never a correctness signal)."""
        return self._stats

    # -- public API ---------------------------------------------------------

    def shard_n_docs(self, shard_stem: str) -> int:
        return self._load(shard_stem).n_docs

    def shard_n_splice_positions(self, shard_stem: str) -> int:
        return self._load(shard_stem).n_splice_positions

    def doc_n_segments(self, shard_stem: str, doc_idx: int) -> int:
        shard = self._load(shard_stem)
        if not (0 <= doc_idx < shard.n_docs):
            raise SpliceProvenanceError(
                f"{shard_stem}: doc {doc_idx} out of range (n_docs={shard.n_docs})."
            )
        return int(shard.n_segments_by_doc[doc_idx])

    def doc_splices(self, shard_stem: str, doc_idx: int) -> dict[int, SpliceRef]:
        shard = self._load(shard_stem)
        if not (0 <= doc_idx < shard.n_docs):
            raise SpliceProvenanceError(
                f"{shard_stem}: doc {doc_idx} out of range (n_docs={shard.n_docs})."
            )
        if isinstance(shard, _CompactShardSplices):
            return shard.doc_splices(doc_idx)
        return shard.splices_by_doc.get(doc_idx, {})

    def ensure_resident(self, shard_stem: str) -> None:
        """Load-if-absent and hold resident."""
        self._load(shard_stem)

    def drop_except(self, stems: Iterable[str]) -> None:
        """Evict resident entries not in ``stems``."""
        keep = set(stems)
        for stem in list(self._resident):
            if stem not in keep:
                del self._resident[stem]
                self._stats.evictions += 1

    def retain(self, stems: Sequence[str]) -> None:
        """Drop non-listed residents, then seed the first ``capacity`` stems.

        Intended for pre-warming the working set a caller is about to touch.
        If ``stems`` exceeds ``max_resident_shards``, the overflow is not
        pre-loaded here — it will load lazily (with LRU eviction) on first
        access, which is slower but still byte-identical.

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
                f"SpliceProvenanceIndex.retain: working set of {len(deduped)} sidecars "
                f"exceeds max_resident_shards={self._max_resident_shards}; overflow "
                f"reparses lazily (still byte-identical)."
            )

    # -- internals ----------------------------------------------------------

    def _fail(self, shard_stem: str, why: str) -> SpliceProvenanceError:
        return SpliceProvenanceError(
            f"{self._dir / (shard_stem + '.json')}: {why} {_RESPLICE_HINT}"
        )

    def _load(self, shard_stem: str) -> _ShardSplices | _CompactShardSplices:
        cached = self._resident.get(shard_stem)
        if cached is not None:
            self._resident.move_to_end(shard_stem)
            self._stats.hits += 1
            return cached
        path = self._dir / f"{shard_stem}.json"
        try:
            self._stats.bytes_read += path.stat().st_size
        except OSError:
            pass
        started = _time.monotonic()
        try:
            shard = self._parse_shard(shard_stem)
        finally:
            self._stats.load_seconds += _time.monotonic() - started
        if self._compact:
            try:
                shard = _CompactShardSplices.from_reference(shard)
            except OverflowError:
                # The legacy schema accepts arbitrary-size Python integers for
                # whf_base values. Real artifacts use ordinary non-negative IDs,
                # but retaining the reference object here preserves that broader
                # public behavior exactly instead of narrowing it to int64.
                logger.warning(
                    "{}: sidecar values exceed compact int64 storage; retaining "
                    "the validated reference representation",
                    shard_stem,
                )
        self._admit(shard_stem, shard)
        return shard

    def _admit(
        self,
        shard_stem: str,
        shard: _ShardSplices | _CompactShardSplices,
    ) -> None:
        """Insert as MRU, evicting the LRU resident while over capacity."""
        self._stats.loads += 1
        self._resident[shard_stem] = shard
        self._resident.move_to_end(shard_stem)
        while len(self._resident) > self._max_resident_shards:
            self._resident.popitem(last=False)
            self._stats.evictions += 1

    def _parse_shard(self, shard_stem: str) -> _ShardSplices:
        path = self._dir / f"{shard_stem}.json"
        if not path.is_file():
            raise self._fail(shard_stem, "sidecar file missing.")
        raw = json.loads(path.read_text())
        version = raw.get("schema_version")
        if not isinstance(version, int) or version < 2:
            raise self._fail(
                shard_stem,
                f"schema_version={version!r} but splice_counterfactual needs >= 2 "
                f"(would_have_followed records). Regenerate with skip_existing=false.",
            )
        docs = raw.get("docs")
        if not isinstance(docs, list):
            raise self._fail(shard_stem, f"'docs' must be a list, got {type(docs).__name__}.")
        n_docs = raw.get("n_docs")
        if not isinstance(n_docs, int) or n_docs != len(docs):
            raise self._fail(shard_stem, f"n_docs={n_docs!r} != len(docs)={len(docs)}.")
        n_segments_by_doc: list[int] = []
        for i, rec in enumerate(docs):
            if not isinstance(rec, dict):
                raise self._fail(shard_stem, f"docs[{i}] is not an object: {rec!r}.")
            if rec.get("doc_id") != i:
                raise self._fail(shard_stem, f"docs[{i}].doc_id={rec.get('doc_id')!r} != {i}.")
            n_segs = rec.get("n_segments")
            if not isinstance(n_segs, int) or n_segs < 1:
                raise self._fail(shard_stem, f"docs[{i}].n_segments={n_segs!r} invalid.")
            n_segments_by_doc.append(n_segs)
        splices_by_doc: dict[int, dict[int, SpliceRef]] = {}
        n_total = 0
        for i, rec in enumerate(docs):
            if not isinstance(rec, dict):
                raise self._fail(shard_stem, f"docs[{i}] is not an object: {rec!r}.")
            splices = rec.get("splices")
            if not isinstance(splices, list):
                raise self._fail(
                    shard_stem,
                    f"docs[{i}] has no 'splices' list — pre-#596 sidecar.",
                )
            if not splices:
                continue
            doc_map: dict[int, SpliceRef] = {}
            for j, s in enumerate(splices):
                if not isinstance(s, dict):
                    raise self._fail(
                        shard_stem, f"docs[{i}].splices[{j}] is not an object: {s!r}."
                    )
                pos = s.get("pos")
                whf_base = s.get("whf_base")
                whf_derived = s.get("whf_derived")
                malformed = self._fail(
                    shard_stem, f"docs[{i}].splices[{j}] malformed: {s!r}."
                )
                if not isinstance(pos, int):
                    raise malformed
                if not isinstance(whf_base, list) or len(whf_base) != 3:
                    raise malformed
                wb0, wb1, wb2 = whf_base
                if not (
                    isinstance(wb0, int) and isinstance(wb1, int) and isinstance(wb2, int)
                ):
                    raise malformed
                if not isinstance(whf_derived, list) or len(whf_derived) != 2:
                    raise malformed
                ref_doc, ref_pos = whf_derived
                if not (isinstance(ref_doc, int) and isinstance(ref_pos, int)):
                    raise malformed
                if not (1 <= pos < n_segments_by_doc[i]):
                    raise self._fail(
                        shard_stem,
                        f"docs[{i}].splices[{j}].pos={pos} out of range "
                        f"[1, {n_segments_by_doc[i]}).",
                    )
                if pos in doc_map:
                    raise self._fail(
                        shard_stem,
                        f"docs[{i}].splices[{j}].pos={pos} duplicates an earlier "
                        "splice position in the same derived doc.",
                    )
                if not (0 <= ref_doc < n_docs) or not (
                    0 <= ref_pos < n_segments_by_doc[ref_doc]
                ):
                    raise self._fail(
                        shard_stem,
                        f"docs[{i}].splices[{j}].whf_derived={whf_derived} out of "
                        f"range for this shard.",
                    )
                doc_map[pos] = SpliceRef(
                    pos=pos,
                    whf_base=(wb0, wb1, wb2),
                    whf_derived=(ref_doc, ref_pos),
                )
            splices_by_doc[i] = doc_map
            n_total += len(doc_map)
        params_obj = raw.get("params", {})
        if not isinstance(params_obj, dict):
            raise self._fail(
                shard_stem, f"params is not a dict: {type(params_obj).__name__}."
            )
        recorded_total = params_obj.get("n_splice_positions")
        if recorded_total != n_total:
            raise self._fail(
                shard_stem,
                f"params.n_splice_positions={recorded_total!r} != counted {n_total}.",
            )
        logger.info(
            "{}: {} splice position(s) across {} derived doc(s)",
            shard_stem, n_total, n_docs,
        )
        if n_total == 0:
            logger.error(
                "{}: sidecar records ZERO splice positions — this shard trains "
                "identically to absolute-percentile conditioning under "
                "splice_counterfactual (identity splice / cut_fraction too low?)",
                shard_stem,
            )
        return _ShardSplices(
            n_docs=n_docs,
            n_splice_positions=n_total,
            n_segments_by_doc=tuple(n_segments_by_doc),
            splices_by_doc=splices_by_doc,
        )
