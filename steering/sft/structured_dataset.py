"""Strict structured-rollout loading and transactional rank-local cursoring."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import pyarrow as pa
import pyarrow.parquet as pq

from steering.sft.structured_rollout import (
    STRUCTURED_RENDERER_VERSION,
    STRUCTURED_ROLLOUT_SCHEMA_VERSION,
    ConversationEnvironment,
    EnvironmentBuildResult,
    StructuredRolloutError,
    StructuredRolloutRow,
    assert_rollout_artifact_format,
    structured_row_digest,
    structured_table_to_rows,
)


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _aggregate_digests(digests: Sequence[str]) -> str:
    return hashlib.sha256("".join(f"{digest}\n" for digest in digests).encode()).hexdigest()


@dataclass(frozen=True)
class StructuredRolloutArtifact:
    """Fully audited immutable rows and their canonical artifact identity."""

    path: Path
    rows: tuple[StructuredRolloutRow, ...]
    metadata: Mapping[str, object]
    identity_sha256: str


def load_structured_rollout_artifact(
    artifact_dir: Path | str,
    *,
    decode: Callable[[Sequence[int]], str],
    tokenizer_sha256: str,
    expected_mixture: str | None = None,
    expected_split: str | None = None,
) -> StructuredRolloutArtifact:
    """Load only a complete, exact ``structured_nanochat_rollout_v1`` artifact."""
    path = Path(artifact_dir)
    meta_path = path / "conversion_meta.json"
    try:
        raw = json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StructuredRolloutError(
            f"cannot read structured rollout metadata {meta_path}: {error}"
        ) from error
    if not isinstance(raw, dict):
        raise StructuredRolloutError("structured rollout metadata must be a JSON object")
    assert_rollout_artifact_format(raw, structured=True)
    expected_versions = {
        "format_version": 1,
        "schema_version": STRUCTURED_ROLLOUT_SCHEMA_VERSION,
        "renderer_version": STRUCTURED_RENDERER_VERSION,
        "tokenizer_sha256": tokenizer_sha256,
    }
    for key, expected in expected_versions.items():
        if raw.get(key) != expected:
            raise StructuredRolloutError(
                f"structured rollout {key} mismatch: expected {expected!r}, "
                f"got {raw.get(key)!r}"
            )
    for key, expected in (("mixture", expected_mixture), ("split", expected_split)):
        if expected is not None and raw.get(key) != expected:
            raise StructuredRolloutError(
                f"structured rollout {key} mismatch: expected {expected!r}, "
                f"got {raw.get(key)!r}"
            )
    completed = raw.get("completed_shards")
    total = raw.get("total_shards")
    if (
        isinstance(completed, bool)
        or not isinstance(completed, int)
        or isinstance(total, bool)
        or not isinstance(total, int)
        or completed != total
        or total < 0
    ):
        raise StructuredRolloutError(
            "structured rollout must have completed_shards == total_shards >= 0"
        )
    records = raw.get("shards")
    if not isinstance(records, list) or len(records) != total:
        raise StructuredRolloutError(
            "structured rollout must record exactly one entry per shard"
        )
    expected_files = {f"shard_{index:05d}.parquet" for index in range(total)}
    actual_files = {candidate.name for candidate in path.glob("*.parquet")}
    if actual_files != expected_files:
        raise StructuredRolloutError(
            "structured rollout shard set mismatch: "
            f"missing={sorted(expected_files - actual_files)}, "
            f"extra={sorted(actual_files - expected_files)}"
        )

    rows: list[StructuredRolloutRow] = []
    shard_row_digests: list[str] = []
    for shard_index, record in enumerate(records):
        if not isinstance(record, Mapping) or record.get("shard_index") != shard_index:
            raise StructuredRolloutError(
                f"structured rollout shard record {shard_index} is malformed or out of order"
            )
        shard_path = path / f"shard_{shard_index:05d}.parquet"
        try:
            table = pq.read_table(shard_path)
            shard_rows = structured_table_to_rows(table, decode=decode)
        except (OSError, pa.ArrowException, StructuredRolloutError) as error:
            raise StructuredRolloutError(
                f"invalid structured rollout shard {shard_path}: {error}"
            ) from error
        if record.get("written_row_count") != len(shard_rows):
            raise StructuredRolloutError(
                f"structured rollout shard {shard_index} row count disagrees with metadata"
            )
        shard_digest = _aggregate_digests(
            [structured_row_digest(row) for row in shard_rows]
        )
        if record.get("row_content_sha256") != shard_digest:
            raise StructuredRolloutError(
                f"structured rollout shard {shard_index} content digest mismatch"
            )
        rows.extend(shard_rows)
        shard_row_digests.append(shard_digest)
    original_indices = [row.original_index for row in rows]
    if len(set(original_indices)) != len(original_indices):
        raise StructuredRolloutError("structured rollout original_index values must be unique")
    if raw.get("total_rows") != len(rows):
        raise StructuredRolloutError("structured rollout total_rows disagrees with shards")
    if raw.get("row_content_sha256") != _aggregate_digests(shard_row_digests):
        raise StructuredRolloutError("structured rollout aggregate row digest mismatch")
    return StructuredRolloutArtifact(
        path=path,
        rows=tuple(rows),
        metadata=raw,
        identity_sha256=_canonical_sha256(raw),
    )


@dataclass(frozen=True)
class RankCursorState:
    """Committed attempted/accepted state plus ordered eligible lookahead."""

    artifact_sha256: str
    geometry_sha256: str
    rank: int
    world_size: int
    attempted: int
    accepted: int
    buffered_original_indices: tuple[int, ...]

    def to_json_dict(self) -> dict[str, object]:
        return {
            "version": 1,
            "artifact_sha256": self.artifact_sha256,
            "geometry_sha256": self.geometry_sha256,
            "rank": self.rank,
            "world_size": self.world_size,
            "attempted": self.attempted,
            "accepted": self.accepted,
            "buffered_original_indices": list(self.buffered_original_indices),
        }


@dataclass(frozen=True)
class StagedEnvironmentBatch:
    """A provisional rank-local batch; cursor state changes only on commit."""

    environments: tuple[ConversationEnvironment, ...]
    next_state: RankCursorState
    filter_counts: Mapping[str, int]
    base_state_sha256: str


class StructuredEnvironmentCursor:
    """Deterministic rank-strided filtering/refill with transactional commits."""

    def __init__(
        self,
        artifact: StructuredRolloutArtifact,
        *,
        rank: int,
        world_size: int,
        geometry_identity: Mapping[str, object],
        state: Mapping[str, object] | None = None,
    ) -> None:
        if (
            isinstance(rank, bool)
            or not isinstance(rank, int)
            or isinstance(world_size, bool)
            or not isinstance(world_size, int)
            or world_size <= 0
            or not 0 <= rank < world_size
        ):
            raise StructuredRolloutError("cursor rank/world_size topology is invalid")
        self._artifact = artifact
        self._rank_rows = artifact.rows[rank::world_size]
        self._row_by_original = {row.original_index: row for row in self._rank_rows}
        self._geometry_sha256 = _canonical_sha256(dict(geometry_identity))
        self._state = self._parse_state(state, rank=rank, world_size=world_size)

    def _parse_state(
        self,
        raw: Mapping[str, object] | None,
        *,
        rank: int,
        world_size: int,
    ) -> RankCursorState:
        if raw is None:
            return RankCursorState(
                artifact_sha256=self._artifact.identity_sha256,
                geometry_sha256=self._geometry_sha256,
                rank=rank,
                world_size=world_size,
                attempted=0,
                accepted=0,
                buffered_original_indices=(),
            )
        expected_fields = {
            "version",
            "artifact_sha256",
            "geometry_sha256",
            "rank",
            "world_size",
            "attempted",
            "accepted",
            "buffered_original_indices",
        }
        if set(raw) != expected_fields or raw.get("version") != 1:
            raise StructuredRolloutError("cursor checkpoint schema/version mismatch")
        expected_identity = {
            "artifact_sha256": self._artifact.identity_sha256,
            "geometry_sha256": self._geometry_sha256,
            "rank": rank,
            "world_size": world_size,
        }
        for key, expected in expected_identity.items():
            if raw.get(key) != expected:
                raise StructuredRolloutError(
                    f"cursor resume identity mismatch for {key}: "
                    f"expected {expected!r}, got {raw.get(key)!r}"
                )
        attempted = raw.get("attempted")
        accepted = raw.get("accepted")
        buffered = raw.get("buffered_original_indices")
        if (
            isinstance(attempted, bool)
            or not isinstance(attempted, int)
            or not 0 <= attempted <= len(self._rank_rows)
            or isinstance(accepted, bool)
            or not isinstance(accepted, int)
            or accepted < 0
            or not isinstance(buffered, list)
            or any(
                isinstance(index, bool) or not isinstance(index, int)
                for index in buffered
            )
        ):
            raise StructuredRolloutError("cursor attempted/accepted/buffer state is invalid")
        if len(set(buffered)) != len(buffered) or any(
            index not in self._row_by_original for index in buffered
        ):
            raise StructuredRolloutError("cursor buffer contains duplicate or foreign rows")
        buffered_indices = cast(list[int], buffered)
        return RankCursorState(
            artifact_sha256=self._artifact.identity_sha256,
            geometry_sha256=self._geometry_sha256,
            rank=rank,
            world_size=world_size,
            attempted=attempted,
            accepted=accepted,
            buffered_original_indices=tuple(buffered_indices),
        )

    @property
    def state(self) -> RankCursorState:
        return self._state

    def stage(
        self,
        count: int,
        *,
        prefetch: int,
        build: Callable[[StructuredRolloutRow], EnvironmentBuildResult],
    ) -> StagedEnvironmentBatch:
        """Build exactly ``count`` environments and retain ordered lookahead."""
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(prefetch, bool)
            or not isinstance(prefetch, int)
            or prefetch < 0
        ):
            raise StructuredRolloutError("cursor count must be positive and prefetch non-negative")
        desired = count + prefetch
        buffered_rows = [
            self._row_by_original[index]
            for index in self._state.buffered_original_indices
        ]
        environments: list[ConversationEnvironment] = []
        next_buffer: list[int] = []
        filter_counts: dict[str, int] = {}

        def admit(row: StructuredRolloutRow) -> None:
            result = build(row)
            if result.environment is None:
                reason = (
                    "unknown"
                    if result.filter_reason is None
                    else result.filter_reason.value
                )
                filter_counts[reason] = filter_counts.get(reason, 0) + 1
                return
            if len(environments) < desired:
                environments.append(result.environment)
            else:
                next_buffer.append(row.original_index)

        for row in buffered_rows:
            admit(row)
        attempted = self._state.attempted
        while len(environments) < desired and attempted < len(self._rank_rows):
            row = self._rank_rows[attempted]
            attempted += 1
            admit(row)
        if len(environments) < count:
            raise StructuredRolloutError(
                "structured rollout exhausted before every rank could refill its "
                f"required group: requested={count}, eligible={len(environments)}"
            )
        selected = environments[:count]
        # Lookahead environments are reconstructed from authoritative rows on
        # resume, so only their ordered original IDs enter checkpoint state.
        next_buffer = [
            environment.original_index for environment in environments[count:]
        ] + next_buffer
        next_state = RankCursorState(
            artifact_sha256=self._state.artifact_sha256,
            geometry_sha256=self._state.geometry_sha256,
            rank=self._state.rank,
            world_size=self._state.world_size,
            attempted=attempted,
            accepted=self._state.accepted + count,
            buffered_original_indices=tuple(next_buffer),
        )
        return StagedEnvironmentBatch(
            environments=tuple(selected),
            next_state=next_state,
            filter_counts=filter_counts,
            base_state_sha256=_canonical_sha256(self._state.to_json_dict()),
        )

    def commit(self, staged: StagedEnvironmentBatch) -> None:
        """Commit once, after a successful optimizer step."""
        current = _canonical_sha256(self._state.to_json_dict())
        if staged.base_state_sha256 != current:
            raise StructuredRolloutError(
                "staged cursor batch is stale or has already been committed"
            )
        self._state = staged.next_state


__all__ = [
    "RankCursorState",
    "StagedEnvironmentBatch",
    "StructuredEnvironmentCursor",
    "StructuredRolloutArtifact",
    "load_structured_rollout_artifact",
]
