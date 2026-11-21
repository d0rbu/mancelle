"""Join packed conditioning segments to splice junctions without reading packed bodies.

The packed loader supplies its already-loaded row shard. Only one provenance
overlay and the bounded splice-sidecar index are retained here.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch as t

from steering.conditional.splice_counterfactual import (
    _file_fingerprint,
    _FileSignature,
    _fingerprint_files_cached,
    splice_sidecar_fingerprint,
)
from steering.conditional.splice_provenance import SpliceProvenanceError, SpliceProvenanceIndex
from steering.gradient_overlay import (
    PROVENANCE_BASE_SHARD_IDX_KEY,
    PROVENANCE_DOC_IDX_KEY,
    PROVENANCE_SEG_ORDINAL_KEY,
    provenance_dir,
)
from steering.overlay_io import (
    MANIFEST_FILENAME,
    glob_overlay_shards,
    load_overlay_shard,
    read_manifest,
)
from steering.packed_batch_io import PackedRowShard


def validate_splice_identity_resume(
    resume_state: Mapping[str, Any] | None,
    identity: Mapping[str, str] | None,
) -> None:
    """Allow a legacy starting checkpoint, but never silently change active inputs."""
    key = "conditioning_splice_identity"
    if resume_state is not None and key in resume_state and resume_state[key] != identity:
        raise SpliceProvenanceError(
            "conditioning cooldown splice provenance changed since the saved loader state"
        )


class ConditioningSpliceSidecars(SpliceProvenanceIndex):
    """Bind lazy sidecar loads to the same immutable bytes as the saved identity."""

    def __init__(self, feature_dir: Path, *, cache_dir: Path | None = None) -> None:
        super().__init__(feature_dir)
        paths = sorted((feature_dir / "_splice_provenance").glob("*.json"))
        self._signatures = {path.stem: _FileSignature.capture(path) for path in paths}
        self.fingerprint = splice_sidecar_fingerprint(feature_dir, cache_dir=cache_dir)
        if paths != sorted((feature_dir / "_splice_provenance").glob("*.json")) or any(
            _FileSignature.capture(path) != self._signatures[path.stem] for path in paths
        ):
            raise SpliceProvenanceError(
                "splice sidecars changed during cooldown identity validation"
            )

    def _parse_shard(self, shard_stem: str):
        path = self._dir / f"{shard_stem}.json"
        expected = self._signatures.get(shard_stem)
        if expected is None or _FileSignature.capture(path) != expected:
            raise SpliceProvenanceError(
                f"{path}: splice sidecar changed since cooldown identity validation"
            )
        result = super()._parse_shard(shard_stem)
        if _FileSignature.capture(path) != expected:
            raise SpliceProvenanceError(f"{path}: splice sidecar changed during cooldown load")
        return result


class ConditioningSpliceIndex:
    """Validated junction flags for real packed segments, with a one-shard cache."""

    def __init__(
        self,
        packed_dir: Path,
        feature_dir: Path,
        *,
        packed_shard_names: list[str],
    ) -> None:
        self._packed_dir = packed_dir
        self._sidecars = ConditioningSpliceSidecars(feature_dir, cache_dir=packed_dir)
        manifest_path = packed_dir / MANIFEST_FILENAME
        manifest_signature = _FileSignature.capture(manifest_path)
        manifest_fingerprint = _file_fingerprint(manifest_path)
        manifest = read_manifest(packed_dir)
        metadata = manifest.get("metadata") if manifest is not None else None
        stems = metadata.get("base_shard_stems") if isinstance(metadata, dict) else None
        if (
            not isinstance(stems, list)
            or not stems
            or not all(isinstance(stem, str) and stem for stem in stems)
            or len(set(stems)) != len(stems)
        ):
            raise SpliceProvenanceError(
                "conditioning cooldown requires packed metadata.base_shard_stems"
            )
        self._stems: list[str] = stems
        paths = glob_overlay_shards(provenance_dir(packed_dir))
        if (
            sorted(path.name for path in paths) != sorted(packed_shard_names)
            or not paths
            or not isinstance(metadata, dict)
            or type(metadata.get("provenance_shards")) is not int
            or metadata["provenance_shards"] != len(paths)
        ):
            raise SpliceProvenanceError(
                "conditioning cooldown requires one provenance shard per packed shard "
                "and matching metadata.provenance_shards; rerun precompute_packed_rows"
            )
        self._provenance_signatures = {path.name: _FileSignature.capture(path) for path in paths}
        self.identity = {
            "manifest": manifest_fingerprint,
            "provenance": _fingerprint_files_cached(
                paths,
                cache_dir=packed_dir,
                cache_filename=".packed-provenance-fingerprint-cache.json",
                namespace="packed_provenance",
            ),
            "sidecars": self._sidecars.fingerprint,
        }
        if paths != glob_overlay_shards(provenance_dir(packed_dir)) or any(
            _FileSignature.capture(path) != self._provenance_signatures[path.name] for path in paths
        ):
            raise SpliceProvenanceError(
                "packed provenance changed during cooldown identity validation"
            )
        if (
            _file_fingerprint(manifest_path) != manifest_fingerprint
            or _FileSignature.capture(manifest_path) != manifest_signature
        ):
            raise SpliceProvenanceError(
                "packed manifest changed during cooldown provenance validation"
            )
        self._cached_name: str | None = None
        self._cached_triples: tuple[t.Tensor, t.Tensor, t.Tensor] | None = None

    def row_flags(self, shard_name: str, base: PackedRowShard, row_idx: int) -> list[bool]:
        if shard_name != self._cached_name:
            path = provenance_dir(self._packed_dir) / shard_name
            expected_signature = self._provenance_signatures.get(shard_name)
            if expected_signature is None or _FileSignature.capture(path) != expected_signature:
                raise SpliceProvenanceError(
                    f"{path}: packed provenance changed since cooldown identity validation"
                )
            raw = load_overlay_shard(path)
            if _FileSignature.capture(path) != expected_signature:
                raise SpliceProvenanceError(
                    f"{path}: packed provenance changed during cooldown load"
                )
            keys = (
                PROVENANCE_BASE_SHARD_IDX_KEY,
                PROVENANCE_DOC_IDX_KEY,
                PROVENANCE_SEG_ORDINAL_KEY,
            )
            if set(raw) != set(keys):
                raise SpliceProvenanceError(f"{path}: invalid packed provenance fields")
            triples = tuple(raw[key] for key in keys)
            for values in triples:
                if (
                    values.dtype != t.int32
                    or values.ndim != 1
                    or len(values) != base.total_segments
                ):
                    raise SpliceProvenanceError(
                        f"{path}: provenance must be int32 vectors matching packed segment count"
                    )
            self._cached_name = shard_name
            self._cached_triples = (triples[0], triples[1], triples[2])
        assert self._cached_triples is not None
        start, end = base.row_seg_offsets[row_idx : row_idx + 2].tolist()
        base_ids, doc_ids, ordinals = (
            values[start:end].tolist() for values in self._cached_triples
        )
        packed_doc_ids = base.seg_doc_id[start:end].tolist()
        flags: list[bool] = []
        documents: dict[int, tuple[int, int, int]] = {}
        for packed_doc, base_id, doc_id, ordinal in zip(
            packed_doc_ids, base_ids, doc_ids, ordinals, strict=True
        ):
            if not 0 <= base_id < len(self._stems) or doc_id < 0 or ordinal < 0:
                raise SpliceProvenanceError(f"{shard_name}: out-of-bounds packed provenance")
            previous = documents.get(packed_doc)
            expected_ordinal = 0 if previous is None else previous[2] + 1
            if ordinal != expected_ordinal or (
                previous is not None and (base_id, doc_id) != previous[:2]
            ):
                raise SpliceProvenanceError(
                    f"{shard_name}: packed document provenance must be one source document "
                    "with prefix segment ordinals"
                )
            documents[packed_doc] = (base_id, doc_id, ordinal)
            stem = self._stems[base_id]
            if ordinal >= self._sidecars.doc_n_segments(stem, doc_id):
                raise SpliceProvenanceError(
                    f"{shard_name}: segment ordinal exceeds sidecar document"
                )
            flags.append(ordinal in self._sidecars.doc_splices(stem, doc_id))
        return flags
