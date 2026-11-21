"""Audited replay-boundary dataloader migration across DDP topologies.

The precomputed loader stripes rows independently inside every packed shard, so
rank-local checkpoint frontiers do not collapse to one exact global cursor when
the world size changes.  This module deliberately does *not* try to manufacture
an exact continuation.  Instead it chooses the start of the shard containing the
earliest rank-local frontier and synthesizes ordinary target-topology
``batch_rows`` from there.  The resulting continuation may replay rows, but it
provably skips none.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch as t

from steering.conditional.splice_counterfactual import resolve_packed_row_fingerprint_strict
from steering.conditional.splice_resource_admission import validate_packed_layout
from steering.overlay_io import glob_overlay_shards, read_manifest
from steering.safetensors_layout import read_compressed_safetensors_layout


@dataclass(frozen=True, order=True)
class RowReference:
    """One row in the loader's total ``(epoch, shard, row)`` order."""

    epoch: int
    shard_index: int
    row_index: int


@dataclass(frozen=True)
class PackedShardInfo:
    """Header-only identity of one ordered packed-row shard."""

    name: str
    rows: int
    context_len: int
    compressed_bytes: int


@dataclass(frozen=True)
class ReplayBoundaryPlan:
    """Audit record for a replay-only topology handoff."""

    boundary: tuple[int, str, int]
    replay_rows: int
    replay_tokens: int
    replay_rows_by_source_rank: tuple[int, ...]
    source_frontiers: tuple[tuple[int, str, int], ...]
    inventory_sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "policy": "replay_from_previous_shard_boundary",
            "boundary": list(self.boundary),
            "replay_rows": self.replay_rows,
            "replay_tokens": self.replay_tokens,
            "skipped_rows": 0,
            "replay_rows_by_source_rank": list(self.replay_rows_by_source_rank),
            "source_frontiers": [list(frontier) for frontier in self.source_frontiers],
            "packed_inventory_sha256": self.inventory_sha256,
        }


class PackedRowInventory:
    """Validated, sorted packed-row shapes without decompressing tensor bodies."""

    def __init__(self, shards: list[PackedShardInfo]) -> None:
        if not shards:
            raise ValueError("packed-row inventory must contain at least one shard")
        names = [shard.name for shard in shards]
        if names != sorted(names) or len(names) != len(set(names)):
            raise ValueError("packed-row shard names must be unique and sorted")
        if any(shard.rows <= 0 for shard in shards):
            raise ValueError("packed-row shards must contain at least one row")
        context_lengths = {shard.context_len for shard in shards}
        if len(context_lengths) != 1:
            raise ValueError(
                f"packed-row shards disagree on context length: {sorted(context_lengths)}"
            )
        self.shards = tuple(shards)
        self._index_by_name = {shard.name: index for index, shard in enumerate(shards)}
        self.rows_per_epoch = sum(shard.rows for shard in shards)
        self.context_len = shards[0].context_len
        canonical = [
            {
                "name": shard.name,
                "rows": shard.rows,
                "context_len": shard.context_len,
                "compressed_bytes": shard.compressed_bytes,
            }
            for shard in shards
        ]
        payload = json.dumps(canonical, sort_keys=True, separators=(",", ":")).encode()
        self.sha256 = hashlib.sha256(payload).hexdigest()

    @classmethod
    def from_directory(cls, packed_row_dir: str | Path) -> PackedRowInventory:
        directory = Path(packed_row_dir)
        paths = glob_overlay_shards(directory)
        if not paths:
            raise FileNotFoundError(f"no packed-row shards found in {directory}")
        shards: list[PackedShardInfo] = []
        for path in paths:
            layout = read_compressed_safetensors_layout(path)
            validate_packed_layout(layout)
            tokens = layout.tensor("tokens")
            shards.append(
                PackedShardInfo(
                    name=path.name,
                    rows=tokens.shape[0],
                    context_len=tokens.shape[1] - 1,
                    compressed_bytes=path.stat().st_size,
                )
            )
        return cls(shards)

    def to_tuple(self, reference: RowReference) -> tuple[int, str, int]:
        self.validate(reference)
        return (
            reference.epoch,
            self.shards[reference.shard_index].name,
            reference.row_index,
        )

    def parse(self, raw: Any, *, context: str) -> RowReference:
        if not isinstance(raw, (list, tuple)) or len(raw) != 3:
            raise ValueError(f"{context} must be [epoch, shard_name, row_idx], got {raw!r}")
        epoch, shard_name, row_index = raw
        if type(epoch) is not int or epoch < 1:
            raise ValueError(f"{context} epoch must be a positive int, got {epoch!r}")
        if not isinstance(shard_name, str) or shard_name not in self._index_by_name:
            raise ValueError(f"{context} names unknown packed shard {shard_name!r}")
        if type(row_index) is not int:
            raise ValueError(f"{context} row index must be an int, got {row_index!r}")
        reference = RowReference(epoch, self._index_by_name[shard_name], row_index)
        self.validate(reference)
        return reference

    def validate(self, reference: RowReference) -> None:
        if reference.epoch < 1:
            raise ValueError(f"row-reference epoch must be positive, got {reference.epoch}")
        if not 0 <= reference.shard_index < len(self.shards):
            raise ValueError(f"row-reference shard index is out of range: {reference}")
        rows = self.shards[reference.shard_index].rows
        if not 0 <= reference.row_index < rows:
            raise ValueError(
                f"row-reference index {reference.row_index} is out of range for "
                f"{self.shards[reference.shard_index].name} with {rows} rows"
            )

    def ordinal(self, reference: RowReference) -> int:
        self.validate(reference)
        prior_shards = sum(shard.rows for shard in self.shards[: reference.shard_index])
        return (reference.epoch - 1) * self.rows_per_epoch + prior_shards + reference.row_index

    def next_owned(self, reference: RowReference, *, rank: int, world_size: int) -> RowReference:
        """Return the first later row owned by ``rank`` under shard-local striping."""

        self.validate(reference)
        if not 0 <= rank < world_size:
            raise ValueError(f"rank must be in [0, {world_size}), got {rank}")
        epoch = reference.epoch
        shard_index = reference.shard_index
        row_index = reference.row_index + 1
        while True:
            shard = self.shards[shard_index]
            remainder = row_index % world_size
            candidate = row_index + ((rank - remainder) % world_size)
            if candidate < shard.rows:
                return RowReference(epoch, shard_index, candidate)
            shard_index += 1
            row_index = 0
            if shard_index == len(self.shards):
                shard_index = 0
                epoch += 1
            if epoch > reference.epoch + 1 and not any(
                rank < candidate_shard.rows for candidate_shard in self.shards
            ):
                raise ValueError(
                    f"rank {rank} owns no rows in the packed inventory at world size {world_size}"
                )

    def first_owned_at_or_after(
        self,
        boundary: RowReference,
        *,
        rank: int,
        world_size: int,
    ) -> RowReference:
        """Return the first rank-owned row at or after an inclusive boundary."""

        self.validate(boundary)
        shard = self.shards[boundary.shard_index]
        remainder = boundary.row_index % world_size
        candidate = boundary.row_index + ((rank - remainder) % world_size)
        if candidate < shard.rows:
            return RowReference(boundary.epoch, boundary.shard_index, candidate)
        previous = RowReference(boundary.epoch, boundary.shard_index, shard.rows - 1)
        return self.next_owned(previous, rank=rank, world_size=world_size)

    def consecutive_owned(
        self,
        start: RowReference,
        *,
        rank: int,
        world_size: int,
        count: int,
    ) -> list[RowReference]:
        if type(count) is not int or count <= 0:
            raise ValueError(f"row-reference count must be positive, got {count!r}")
        if start.row_index % world_size != rank:
            raise ValueError(f"start reference {start} is not owned by rank {rank}")
        rows = [start]
        while len(rows) < count:
            rows.append(self.next_owned(rows[-1], rank=rank, world_size=world_size))
        return rows

    def count_owned_between(
        self,
        start: RowReference,
        end: RowReference,
        *,
        rank: int,
        world_size: int,
    ) -> int:
        """Count rank-owned rows in the half-open total-order range ``[start, end)``."""

        start_ordinal = self.ordinal(start)
        end_ordinal = self.ordinal(end)
        if end_ordinal < start_ordinal:
            raise ValueError(f"row range ends before it starts: {start} -> {end}")
        count = 0
        cursor = start
        while self.ordinal(cursor) < end_ordinal:
            shard_end_ordinal = (cursor.epoch - 1) * self.rows_per_epoch + sum(
                item.rows for item in self.shards[: cursor.shard_index + 1]
            )
            stop_ordinal = min(shard_end_ordinal, end_ordinal)
            stop_row = cursor.row_index + (stop_ordinal - self.ordinal(cursor))
            first = cursor.row_index + ((rank - cursor.row_index % world_size) % world_size)
            if first < stop_row:
                count += 1 + (stop_row - 1 - first) // world_size
            if stop_ordinal == end_ordinal:
                break
            next_shard = cursor.shard_index + 1
            next_epoch = cursor.epoch
            if next_shard == len(self.shards):
                next_shard = 0
                next_epoch += 1
            cursor = RowReference(next_epoch, next_shard, 0)
        return count


_PER_RANK_STATE_KEYS = frozenset(
    {
        "batch_idx",
        "batch_rows",
        "ddp_world_size",
        "device_batch_size",
        "epoch",
        "feature_pipeline_rng_state",
        "label_dropout_draw_batch_size",
        "label_dropout_draw_offset",
        "label_dropout_rng_state",
        "shard",
        "shard_batch_offset",
    }
)
_LABEL_DROPOUT_STATE_KEYS = frozenset(
    {
        "label_dropout_draw_batch_size",
        "label_dropout_draw_offset",
        "label_dropout_rng_state",
    }
)


def validate_selection_inventory(
    packed_row_dir: str | Path,
    selection_dir: str | Path,
) -> str:
    """Require a manifest-bound selection shard for every packed-row shard."""

    packed_directory = Path(packed_row_dir).resolve()
    selection_directory = Path(selection_dir).resolve()
    if (
        selection_directory.parent.name != "sel"
        or selection_directory.parent.parent != packed_directory
    ):
        raise ValueError(
            "precomputed selection directory must be the canonical "
            f"<packed_row_dir>/sel/<selection-hash> leaf: {selection_directory}"
        )

    packed_stems = {
        path.name.removesuffix(".safetensors.zst")
        for path in glob_overlay_shards(packed_directory)
    }
    selection_stems = {
        path.name.removesuffix(".safetensors.zst")
        for path in glob_overlay_shards(selection_directory)
    }
    if packed_stems != selection_stems:
        missing = sorted(packed_stems - selection_stems)
        extra = sorted(selection_stems - packed_stems)
        raise ValueError(
            "precomputed selection inventory does not exactly match packed rows: "
            f"missing={missing[:8]}, extra={extra[:8]}"
        )
    manifest = read_manifest(selection_directory)
    if manifest is None:
        raise ValueError("precomputed selection directory has no valid manifest.json")
    metadata = manifest.get("metadata")
    if not isinstance(metadata, Mapping):
        raise ValueError("precomputed selection manifest metadata must be a mapping")
    if manifest.get("checksum") != selection_directory.name:
        raise ValueError(
            "precomputed selection manifest checksum does not match its directory leaf: "
            f"manifest={manifest.get('checksum')!r}, leaf={selection_directory.name!r}"
        )
    packed_metadata = metadata.get("packed_rows")
    recorded_fingerprint = (
        packed_metadata.get("packed_row_fingerprint")
        if isinstance(packed_metadata, Mapping)
        else None
    )
    actual_fingerprint = resolve_packed_row_fingerprint_strict(
        packed_directory,
        allow_cache_write=False,
    ).aggregate_sha256
    if recorded_fingerprint != actual_fingerprint:
        raise ValueError(
            "precomputed selection manifest does not bind the supplied packed rows: "
            f"manifest={recorded_fingerprint!r}, actual={actual_fingerprint!r}"
        )
    return actual_fingerprint


def _rng_roles(snapshot: Any, *, context: str) -> tuple[tuple[str, ...], ...]:
    if not isinstance(snapshot, list):
        raise ValueError(f"{context} feature_pipeline_rng_state must be a list")
    roles: list[tuple[str, ...]] = []
    flattened_roles: list[str] = []
    for index, entry in enumerate(snapshot):
        if not isinstance(entry, Mapping):
            raise ValueError(f"{context} RNG entry {index} must be a mapping")
        if set(entry) != {"roles", "device_type", "device_index", "state_hex"}:
            raise ValueError(f"{context} RNG entry {index} has unexpected keys")
        raw_roles = entry["roles"]
        if (
            not isinstance(raw_roles, list)
            or not raw_roles
            or not all(isinstance(role, str) for role in raw_roles)
        ):
            raise ValueError(f"{context} RNG entry {index} roles must be a non-empty string list")
        if entry["device_type"] != "cpu" or entry["device_index"] is not None:
            raise ValueError(f"{context} RNG entry {index} must describe a CPU generator")
        state_hex = entry["state_hex"]
        if not isinstance(state_hex, str) or not state_hex or len(state_hex) % 2:
            raise ValueError(f"{context} RNG entry {index} has malformed state_hex")
        try:
            raw_state = bytes.fromhex(state_hex)
        except ValueError as exc:
            raise ValueError(f"{context} RNG entry {index} has invalid state_hex") from exc
        try:
            t.Generator(device="cpu").set_state(t.tensor(list(raw_state), dtype=t.uint8))
        except RuntimeError as exc:
            raise ValueError(f"{context} RNG entry {index} has invalid generator state") from exc
        roles.append(tuple(raw_roles))
        flattened_roles.extend(raw_roles)
    canonical_roles = [role for role in ("selection", "negative") if role in flattened_roles]
    if flattened_roles != canonical_roles:
        raise ValueError(
            f"{context} feature RNG roles must be unique and canonical, got {flattened_roles}"
        )
    return tuple(roles)


def _fresh_pipeline_rng_state(
    *, seed: int, rank: int, roles: tuple[tuple[str, ...], ...]
) -> list[dict[str, Any]]:
    rank_seed = (seed * 1_000_003 + rank) % (2**63)
    state = t.Generator(device="cpu").manual_seed(rank_seed).get_state()
    state_hex = bytes(state.tolist()).hex()
    return [
        {
            "roles": list(role_group),
            "device_type": "cpu",
            "device_index": None,
            "state_hex": state_hex,
        }
        for role_group in roles
    ]


def build_replay_boundary_envelope(
    saved_envelope: Mapping[str, Any],
    *,
    inventory: PackedRowInventory,
    source_world_size: int,
    destination_world_size: int,
    device_batch_size: int,
    seed: int,
) -> tuple[dict[str, Any], ReplayBoundaryPlan]:
    """Synthesize normal target states at a no-skip replay boundary."""

    if saved_envelope.get("ddp_world_size") != source_world_size:
        raise ValueError(
            "dataloader envelope world size does not match optimizer shards: "
            f"{saved_envelope.get('ddp_world_size')!r} != {source_world_size}"
        )
    raw_states = saved_envelope.get("rank_states")
    if not isinstance(raw_states, list) or len(raw_states) != source_world_size:
        raise ValueError("dataloader envelope must contain exactly one rank state per source rank")
    if type(device_batch_size) is not int or device_batch_size <= 0:
        raise ValueError("device_batch_size must be a positive strict int")

    states: list[dict[str, Any]] = []
    frontiers: list[RowReference] = []
    common_state: dict[str, Any] | None = None
    common_batch_idx: int | None = None
    common_roles: tuple[tuple[str, ...], ...] | None = None
    for rank, raw_state in enumerate(raw_states):
        if not isinstance(raw_state, Mapping):
            raise ValueError(f"dataloader rank {rank} state must be a mapping")
        state = dict(raw_state)
        if state.get("ddp_world_size") != source_world_size:
            raise ValueError(f"dataloader rank {rank} records the wrong source world size")
        present_dropout_state = sorted(_LABEL_DROPOUT_STATE_KEYS.intersection(state))
        if present_dropout_state:
            raise ValueError(
                "topology migration requires label_dropout=0 checkpoint state; "
                f"found={present_dropout_state}"
            )
        raw_rows = state.get("batch_rows")
        if not isinstance(raw_rows, list) or len(raw_rows) != device_batch_size:
            raise ValueError(
                f"dataloader rank {rank} must contain exactly B={device_batch_size} batch_rows"
            )
        rows = [
            inventory.parse(raw, context=f"dataloader rank {rank} batch_rows[{index}]")
            for index, raw in enumerate(raw_rows)
        ]
        for row in rows:
            if row.row_index % source_world_size != rank:
                raise ValueError(f"dataloader rank {rank} contains a row owned by another rank")
        expected = inventory.consecutive_owned(
            rows[0],
            rank=rank,
            world_size=source_world_size,
            count=device_batch_size,
        )
        if rows != expected:
            raise ValueError(f"dataloader rank {rank} batch_rows are not consecutive")
        if (state.get("epoch"), state.get("shard")) != (
            rows[-1].epoch,
            inventory.shards[rows[-1].shard_index].name,
        ):
            raise ValueError(f"dataloader rank {rank} epoch/shard does not match batch_rows")
        batch_idx = state.get("batch_idx")
        if type(batch_idx) is not int or batch_idx < 0:
            raise ValueError(f"dataloader rank {rank} batch_idx must be non-negative")
        if common_batch_idx is None:
            common_batch_idx = batch_idx
        elif batch_idx != common_batch_idx:
            raise ValueError("source dataloader ranks disagree on batch_idx")
        shard_batch_offset = state.get("shard_batch_offset")
        if type(shard_batch_offset) is not int or shard_batch_offset < 0:
            raise ValueError(f"dataloader rank {rank} shard_batch_offset must be non-negative")
        saved_device_batch_size = state.get("device_batch_size")
        if saved_device_batch_size is not None and saved_device_batch_size != device_batch_size:
            raise ValueError(
                f"dataloader rank {rank} records device_batch_size="
                f"{saved_device_batch_size!r}, expected {device_batch_size}"
            )
        role_layout = _rng_roles(
            state.get("feature_pipeline_rng_state"),
            context=f"dataloader rank {rank}",
        )
        if common_roles is None:
            common_roles = role_layout
        elif role_layout != common_roles:
            raise ValueError("source dataloader ranks disagree on feature RNG roles")
        semantic = {key: value for key, value in state.items() if key not in _PER_RANK_STATE_KEYS}
        if common_state is None:
            common_state = semantic
        elif semantic != common_state:
            raise ValueError("source dataloader ranks disagree on semantic state fields")
        states.append(state)
        frontiers.append(rows[0])

    assert common_state is not None
    assert common_batch_idx is not None
    assert common_roles is not None
    earliest = min(frontiers, key=inventory.ordinal)
    boundary = RowReference(earliest.epoch, earliest.shard_index, 0)
    replay_by_rank = tuple(
        inventory.count_owned_between(
            boundary,
            frontier,
            rank=rank,
            world_size=source_world_size,
        )
        for rank, frontier in enumerate(frontiers)
    )

    target_states: list[dict[str, Any]] = []
    for rank in range(destination_world_size):
        first = inventory.first_owned_at_or_after(
            boundary,
            rank=rank,
            world_size=destination_world_size,
        )
        rows = inventory.consecutive_owned(
            first,
            rank=rank,
            world_size=destination_world_size,
            count=device_batch_size,
        )
        last = rows[-1]
        target_states.append(
            {
                **common_state,
                "batch_idx": common_batch_idx,
                "shard": inventory.shards[last.shard_index].name,
                "shard_batch_offset": 0,
                "epoch": last.epoch,
                "ddp_world_size": destination_world_size,
                "device_batch_size": device_batch_size,
                "batch_rows": [list(inventory.to_tuple(row)) for row in rows],
                "feature_pipeline_rng_state": _fresh_pipeline_rng_state(
                    seed=seed,
                    rank=rank,
                    roles=common_roles,
                ),
            }
        )

    replay_rows = sum(replay_by_rank)
    plan = ReplayBoundaryPlan(
        boundary=inventory.to_tuple(boundary),
        replay_rows=replay_rows,
        replay_tokens=replay_rows * inventory.context_len,
        replay_rows_by_source_rank=replay_by_rank,
        source_frontiers=tuple(inventory.to_tuple(frontier) for frontier in frontiers),
        inventory_sha256=inventory.sha256,
    )
    return {
        "ddp_world_size": destination_world_size,
        "rank_states": target_states,
    }, plan
