"""Bounded raw-feature lookup for HF-streamed packed-row production.

This module deliberately owns transport and source-identity coordination only.
Document tokenization, best-fit packing, CSR construction, compression,
provenance, and manifest publication remain in the existing packed-row
producer.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from steering.conditional.feature_index import FeatureIndex, SegmentFeatures
from steering.feature_io import (
    MetadataKey,
    read_shard_header_stable,
    shard_stem_from_path,
)
from steering.remote_feature_inventory import RemoteFeatureShard
from steering.token_boundary_remote import (
    _CAPACITY_RESERVE_BYTES,
    required_staging_capacity_bytes,
)
from steering.token_boundary_remote import (
    PreparedTokenBoundaryShard as _PreparedPackedRowsShard,
)
from steering.token_boundary_remote import (
    RemoteTokenBoundaryTransport as _RemotePackedRowsTransport,
)


class RemotePackedRowsError(RuntimeError):
    """Remote packed-row source or lifecycle invariants were violated."""


@dataclass(frozen=True)
class SelectedParquet:
    """One selected local parquet and its semantic row count."""

    stem: str
    path: Path
    n_rows: int


@dataclass(frozen=True)
class _RawShardMetadata:
    n_docs: int
    n_segments: int


def required_packed_rows_staging_capacity_bytes(
    raw_inventory: Sequence[RemoteFeatureShard],
    percentile_inventory: Sequence[RemoteFeatureShard],
) -> int:
    """Return both streams' two-body lookahead envelope plus 1 GiB."""

    return (
        required_staging_capacity_bytes(raw_inventory)
        + required_staging_capacity_bytes(percentile_inventory)
        - _CAPACITY_RESERVE_BYTES
    )


def _strict_nonnegative_metadata_int(
    metadata: Mapping[str, str],
    key: MetadataKey,
    *,
    shard_path: Path,
) -> int:
    raw = metadata.get(key)
    if raw is None:
        raise RemotePackedRowsError(f"remote feature shard {shard_path} lacks {key.value} metadata")
    try:
        parsed = int(raw)
    except ValueError as exc:
        raise RemotePackedRowsError(
            f"remote feature shard {shard_path} has invalid {key.value}={raw!r}"
        ) from exc
    if parsed < 0:
        raise RemotePackedRowsError(
            f"remote feature shard {shard_path} has negative {key.value}={parsed}"
        )
    return parsed


class RemotePackedRowsFeatureLookup:
    """Monotonic one-shard feature lookup over the complete sealed HF family."""

    def __init__(
        self,
        *,
        transport: _RemotePackedRowsTransport,
        percentile_transport: _RemotePackedRowsTransport,
        segment_size: int,
        percentile_checksum: str,
        token_overlay_dir: Path,
        tokenizer_checksum: str,
        selected_parquets: Sequence[SelectedParquet],
        profile_loads: bool = False,
        start_position: int = 0,
    ) -> None:
        if isinstance(segment_size, bool) or not isinstance(segment_size, int) or segment_size < 1:
            raise ValueError(f"segment_size must be a positive integer, got {segment_size!r}")
        self._transport = transport
        self._percentile_transport = percentile_transport
        if not isinstance(profile_loads, bool):
            raise TypeError(f"profile_loads must be a bool, got {profile_loads!r}")
        self._profile_loads = profile_loads
        self._segment_size = segment_size
        self._percentile_checksum = percentile_checksum
        self._token_overlay_dir = Path(token_overlay_dir).absolute()
        self._tokenizer_checksum = tokenizer_checksum
        self._inventory = tuple(transport.inventory)
        self._percentile_inventory = tuple(percentile_transport.inventory)
        self._shard_stems = [
            shard_stem_from_path(Path(entry.info.path)) for entry in self._inventory
        ]
        # Both transports were sealed by snapshot(): the family grammar plus the
        # exact contiguous 0..N-1 domain (with the percentile transport given
        # expected_shard_count = len(raw inventory)) make equal counts, index
        # alignment, and stem alignment theorems of the transport contract, and
        # stream() yields those sealed positions in order — no consumer-side
        # re-validation is needed.
        if [entry.index for entry in self._percentile_inventory] != [
            entry.index for entry in self._inventory
        ]:
            raise RemotePackedRowsError(
                "raw and percentile transport inventories must align position-for-position: "
                f"raw_count={len(self._inventory)}, "
                f"percentile_count={len(self._percentile_inventory)}"
            )
        self._stem_to_position = {stem: position for position, stem in enumerate(self._shard_stems)}
        self._selected = {entry.stem: entry for entry in selected_parquets}
        if (
            not isinstance(start_position, int)
            or isinstance(start_position, bool)
            or not (0 <= start_position <= len(self._inventory))
        ):
            raise ValueError(
                "start_position must index the sealed remote inventory: "
                f"got {start_position!r} for {len(self._inventory)} bodies"
            )
        # Stream only through the last SELECTED body; the sealed inventory (and
        # therefore positions, profile records, and the final fence) stays
        # whole-family. With no selected stems (test harnesses only — production
        # always selects >= 2 parquets) the stream stays unbounded.
        selected_positions = [self._stem_to_position[stem] for stem in self._selected]
        stream_stop = max(selected_positions) + 1 if selected_positions else len(self._inventory)
        self._stream = (
            (prepared for prepared in ())
            if start_position >= stream_stop
            else transport.stream(
                start_position=start_position,
                stop_position=stream_stop,
            )
        )
        self._percentile_stream = (
            (prepared for prepared in ())
            if start_position >= stream_stop
            else percentile_transport.stream(
                start_position=start_position,
                stop_position=stream_stop,
            )
        )
        self._current_position: int | None = None
        self._current_prepared: _PreparedPackedRowsShard | None = None
        self._current_percentile_prepared: _PreparedPackedRowsShard | None = None
        self._current_raw: _RawShardMetadata | None = None
        self._current_index: FeatureIndex | None = None
        self._current_was_activated = False
        self._activated_stems = {
            entry.stem
            for entry in selected_parquets
            if entry.n_rows == 0 or self._stem_to_position[entry.stem] < start_position
        }
        self._start_position = start_position
        self._consumed_bodies = start_position
        self._header_bytes_processed = 0
        self._decoded_selected_shards = 0
        self._finished = False
        self._body_profiles: dict[int, dict[str, object]] = {}
        self._feature_index_loads: list[dict] = []

    @property
    def segment_size(self) -> int:
        return self._segment_size

    @property
    def shard_stems(self) -> list[str]:
        return list(self._shard_stems)

    def seed_activated_before(self, position: int) -> None:
        """Restore selected-stem completion proven by a packing checkpoint."""
        if not (0 <= position <= len(self._inventory)):
            raise ValueError(f"checkpoint position is outside the sealed inventory: {position}")
        self._activated_stems.update(
            stem for stem in self._selected if self._stem_to_position[stem] < position
        )

    def _validate_departing_current(self) -> None:
        position = self._current_position
        if position is None or self._current_was_activated:
            return
        stem = self._shard_stems[position]
        selected = self._selected.get(stem)
        if selected is None:
            return
        assert self._current_raw is not None
        raw = self._current_raw
        if (selected.n_rows, raw.n_docs, raw.n_segments) != (0, 0, 0):
            raise RemotePackedRowsError(
                "packed-row lookup skipped a selected source shard that was not "
                f"completely empty: stem={stem}, parquet_rows={selected.n_rows}, "
                f"raw_n_docs={raw.n_docs}, raw_n_segments={raw.n_segments}"
            )
        self._activated_stems.add(stem)

    def _consume_next(self) -> None:
        self._validate_departing_current()
        self._current_index = None
        transport_started = time.perf_counter() if self._profile_loads else 0.0
        # Both streams iterate the same sealed [start, stop) positions of two
        # snapshot-validated inventories with equal index domains, and each
        # transport verifies every yielded body against its own inventory
        # entry, so yields arrive in lockstep by construction: a raw
        # StopIteration implies the percentile stream is exhausted too.
        try:
            prepared = next(self._stream)
        except StopIteration:
            self._current_position = None
            self._current_prepared = None
            self._current_percentile_prepared = None
            self._current_raw = None
            raise
        try:
            percentile_prepared = next(self._percentile_stream)
        except StopIteration as exc:
            raise RemotePackedRowsError(
                "percentile transport ended before its lockstep raw body"
            ) from exc
        transport_wait_seconds = (
            time.perf_counter() - transport_started if self._profile_loads else 0.0
        )
        position = self._consumed_bodies
        # Tripwires for a future transport refactor (e.g. a multi-worker pool):
        # the yield-order theorem above assumes single-outstanding submission.
        assert prepared.index == self._inventory[position].index
        assert percentile_prepared.index == self._percentile_inventory[position].index
        header_started = time.perf_counter() if self._profile_loads else 0.0
        header_cpu_started = time.thread_time() if self._profile_loads else 0.0
        header = read_shard_header_stable(
            prepared.local_path,
            expected_identity=prepared.file_identity,
        )
        header_seconds = time.perf_counter() - header_started if self._profile_loads else 0.0
        header_cpu_seconds = time.thread_time() - header_cpu_started if self._profile_loads else 0.0
        metadata = header["__metadata__"]
        n_docs = _strict_nonnegative_metadata_int(
            metadata,
            MetadataKey.N_DOCS,
            shard_path=prepared.local_path,
        )
        n_segments = _strict_nonnegative_metadata_int(
            metadata,
            MetadataKey.N_SEGMENTS,
            shard_path=prepared.local_path,
        )
        raw_segment_size = _strict_nonnegative_metadata_int(
            metadata,
            MetadataKey.SEGMENT_SIZE,
            shard_path=prepared.local_path,
        )
        if raw_segment_size != self._segment_size:
            raise RemotePackedRowsError(
                "remote packed-row segment_size mismatch: "
                f"shard={prepared.local_path}, header={raw_segment_size}, "
                f"configured/path={self._segment_size}"
            )
        stem = shard_stem_from_path(prepared.local_path)
        self._header_bytes_processed += len(
            json.dumps(header, sort_keys=True, default=sorted).encode("utf-8")
        )
        raw = _RawShardMetadata(
            n_docs=n_docs,
            n_segments=n_segments,
        )
        self._current_position = position
        self._current_prepared = prepared
        self._current_percentile_prepared = percentile_prepared
        self._current_raw = raw
        self._current_was_activated = False
        self._consumed_bodies += 1
        if self._profile_loads:
            self._body_profiles[position] = {
                "position": position,
                "shard_stem": stem,
                "selected": stem in self._selected,
                "compressed_bytes": prepared.file_identity.size,
                "transport_wait_seconds": transport_wait_seconds,
                "header_seconds": header_seconds,
                "header_cpu_seconds": header_cpu_seconds,
                "activate_seconds": 0.0,
                "activate_cpu_seconds": 0.0,
            }

    def _activate(self, stem: str) -> None:
        activate_started = time.perf_counter() if self._profile_loads else 0.0
        activate_cpu_started = time.thread_time() if self._profile_loads else 0.0
        selected = self._selected.get(stem)
        if selected is None:
            raise RemotePackedRowsError(
                f"packed-row lookup requested an unselected parquet stem {stem!r}"
            )
        raw = self._current_raw
        assert raw is not None
        if raw.n_docs != selected.n_rows:
            raise RemotePackedRowsError(
                "remote raw/parquet document-count mismatch: "
                f"stem={stem}, raw_n_docs={raw.n_docs}, parquet_rows={selected.n_rows}"
            )
        if selected.n_rows > 0 and raw.n_segments == 0:
            raise RemotePackedRowsError(
                f"nonempty selected parquet {selected.path} has zero raw feature segments"
            )
        prepared = self._current_prepared
        assert prepared is not None
        percentile_prepared = self._current_percentile_prepared
        assert percentile_prepared is not None
        index = FeatureIndex.from_certified_sparse_source(
            feature_dir=prepared.local_path.parent,
            shard_path=prepared.local_path,
            token_overlay_dir=self._token_overlay_dir,
            tokenizer_checksum=self._tokenizer_checksum,
            segment_size=self._segment_size,
            validated_content_fingerprint=self._transport.inventory_sha256,
            percentile_overlay_dir=percentile_prepared.local_path.parent,
            percentile_checksum=self._percentile_checksum,
            token_overlay_required=raw.n_segments > 0,
            skip_feature_mags=True,
            expected_shard_identity=prepared.file_identity,
            expected_percentile_identity=percentile_prepared.file_identity,
            profile_loads=self._profile_loads,
        )
        index.ensure_resident(stem)
        if self._profile_loads:
            self._feature_index_loads.extend(index.load_profiles)
        self._current_index = index
        self._current_was_activated = True
        self._activated_stems.add(stem)
        self._decoded_selected_shards += 1
        if self._profile_loads:
            position = self._current_position
            assert position is not None and position in self._body_profiles
            self._body_profiles[position]["activate_seconds"] = (
                time.perf_counter() - activate_started
            )
            self._body_profiles[position]["activate_cpu_seconds"] = (
                time.thread_time() - activate_cpu_started
            )

    def lookup(self, shard_stem: str, doc_idx: int) -> list[SegmentFeatures]:
        if self._finished:
            raise RemotePackedRowsError("remote packed-row lookup is already finished")
        requested_position = self._stem_to_position.get(shard_stem)
        if requested_position is None:
            raise KeyError(f"Unknown remote shard stem: {shard_stem!r}")
        if requested_position < self._consumed_bodies and self._current_position is None:
            raise RemotePackedRowsError(
                "remote packed-row lookup cannot request a stem before its seek cursor: "
                f"start_position={self._consumed_bodies}, requested={shard_stem!r}"
            )
        current = self._current_position
        if current is not None and requested_position < current:
            raise RemotePackedRowsError(
                "remote packed-row lookup cannot move backwards: "
                f"current={self._shard_stems[current]!r}, requested={shard_stem!r}"
            )
        while self._current_position is None or self._current_position < requested_position:
            try:
                self._consume_next()
            except StopIteration as exc:
                raise RemotePackedRowsError(
                    f"remote packed-row stream ended before requested stem {shard_stem!r}"
                ) from exc
        if self._current_index is None:
            self._activate(shard_stem)
        assert self._current_index is not None
        return self._current_index.lookup(shard_stem, doc_idx)

    def finish(self) -> None:
        """Verify that packing activated every selected source shard."""
        if self._finished:
            return
        self._validate_departing_current()
        missing = sorted(set(self._selected) - self._activated_stems)
        if missing:
            raise RemotePackedRowsError(
                "remote packed-row lookup did not activate every selected parquet stem: "
                f"missing={missing}"
            )
        # Close the streams HERE, not at context exit: closing releases the
        # final staged bodies, and their release fences must fire before
        # manifests publish and the resume state is completion-removed. If the
        # first close raises, the enclosing `with transport,
        # percentile_transport:` scope still tears the other stream down.
        self._stream.close()
        self._percentile_stream.close()
        self._finished = True

    def profile_payload(self) -> dict[str, object]:
        payload: dict[str, object] = {
            "bodies_processed": self._consumed_bodies - self._start_position,
            "cursor_position": self._consumed_bodies,
            "header_bytes_processed": self._header_bytes_processed,
            "decoded_selected_shards": self._decoded_selected_shards,
        }
        if self._profile_loads:
            payload["per_body"] = [
                self._body_profiles[position] for position in sorted(self._body_profiles)
            ]
        return payload

    @property
    def feature_index_loads(self) -> list[dict]:
        return list(self._feature_index_loads)


__all__ = [
    "RemotePackedRowsError",
    "RemotePackedRowsFeatureLookup",
    "SelectedParquet",
    "required_packed_rows_staging_capacity_bytes",
]
