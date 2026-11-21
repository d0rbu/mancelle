"""Memory-lifetime regressions for Stage 0c candidate lookup."""

from __future__ import annotations

import gc
import json
import subprocess
import sys
import tracemalloc
import weakref
from types import MethodType, SimpleNamespace
from typing import Any, cast

import pytest

from steering.eval.axbench_base import stage0c_runner as runner_mod
from steering.eval.axbench_base._stage0c_internals import (
    _CandidatePrefixIndex,
    _PrefixCandidate,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.cluster_feature_contract import ClusterFeatureContract


def _synthetic_index(
    *,
    payload_chars: int = 128 * 1024,
    candidates_per_feature: int = 4,
) -> _CandidatePrefixIndex:
    cfg = cast(
        AxBenchBaseConfig,
        SimpleNamespace(validation_shards=[0]),
    )
    index = _CandidatePrefixIndex(
        cfg,
        parquets=[],
        feature_shards=[],
        candidate_feature_ids=set(),
        base_context_window=2048,
    )

    def load_shard(self: _CandidatePrefixIndex, shard_idx: int) -> object:
        del self, shard_idx
        return object()

    def materialize(
        self: _CandidatePrefixIndex,
        shard: object,
        feature_id: int,
    ) -> list[_PrefixCandidate]:
        del self, shard
        return [
            _PrefixCandidate(
                shard_idx=candidate_idx % 2,
                doc_idx=feature_id * 100 + candidate_idx,
                seg_idx=candidate_idx,
                prefix_text=(
                    f"{feature_id}:{candidate_idx}:"
                    + ("x" * payload_chars)
                ),
                prefix_gemma_tokens=[feature_id, candidate_idx],
                source_bridge_text=f"bridge-{feature_id}-{candidate_idx}",
            )
            for candidate_idx in range(candidates_per_feature)
        ]

    index._load_shard = MethodType(load_shard, index)  # ty: ignore[invalid-assignment]
    index._materialize_shard_candidates = MethodType(  # ty: ignore[invalid-assignment]
        materialize,
        index,
    )
    return index


def test_candidate_lists_are_fresh_equal_and_collectible() -> None:
    index = _synthetic_index(payload_chars=1024)

    first = index.candidates_for_feature(7)
    second = index.candidates_for_feature(7)

    assert first == second
    assert first is not second
    assert all(
        left is not right
        for left, right in zip(first, second, strict=True)
    )
    first_refs = [weakref.ref(candidate) for candidate in first]
    second_refs = [weakref.ref(candidate) for candidate in second]
    del first
    del second
    gc.collect()

    assert all(reference() is None for reference in first_refs + second_refs)
    assert not hasattr(index, "_candidates_by_feature")


def test_many_unique_feature_lookups_leave_no_python_retention_slope() -> None:
    index = _synthetic_index()
    references: list[weakref.ReferenceType[_PrefixCandidate]] = []

    tracemalloc.start()
    try:
        one = index.candidates_for_feature(0)
        references.extend(weakref.ref(candidate) for candidate in one)
        del one
        gc.collect()
        after_one, _ = tracemalloc.get_traced_memory()

        for feature_id in range(1, 13):
            candidates = index.candidates_for_feature(feature_id)
            references.extend(
                weakref.ref(candidate) for candidate in candidates
            )
            del candidates
            gc.collect()
        after_many, _ = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    assert all(reference() is None for reference in references)
    assert after_many <= after_one + 256 * 1024
    assert set(vars(index)) == {
        "_cfg",
        "_parquets",
        "_feature_shards",
        "_candidate_feature_ids",
        "_base_context_window",
        "_loaded_shards",
        "_tokenizer",
        "_heartbeat_segments_scanned",
        "_heartbeat_emitted_count",
        "_heartbeat_last_emit_at",
        "_load_shard",
        "_materialize_shard_candidates",
    }


class _SyntheticClusterContract:
    def target_for_representative(self, representative_id: int) -> Any:
        return SimpleNamespace(
            allowed_member_ids=(
                representative_id,
                representative_id + 1,
                representative_id + 2,
            )
        )


def test_cluster_member_lists_are_released_with_active_concept() -> None:
    inner = _synthetic_index(payload_chars=1024)
    index = runner_mod._ClusterCandidatePrefixIndex(
        inner,
        cast(ClusterFeatureContract, _SyntheticClusterContract()),
    )

    candidates = index.candidates_for_feature(30)
    assert len(candidates) == 12
    references = [weakref.ref(candidate) for candidate in candidates]
    del candidates
    gc.collect()

    assert all(reference() is None for reference in references)
    assert not hasattr(inner, "_candidates_by_feature")


def test_cluster_union_releases_noncanonical_overlapping_member_candidate() -> None:
    references: list[weakref.ReferenceType[_PrefixCandidate]] = []

    class _OverlappingInner:
        def candidates_for_feature(
            self,
            feature_id: int,
        ) -> list[_PrefixCandidate]:
            candidate = _PrefixCandidate(
                shard_idx=0,
                doc_idx=4,
                seg_idx=5,
                prefix_text=f"member-{feature_id}",
                prefix_gemma_tokens=[feature_id],
                source_bridge_text=f"bridge-{feature_id}",
            )
            references.append(weakref.ref(candidate))
            return [candidate]

    class _OverlappingContract:
        def target_for_representative(self, representative_id: int) -> Any:
            assert representative_id == 10
            return SimpleNamespace(allowed_member_ids=(10, 11))

    index = runner_mod._ClusterCandidatePrefixIndex(
        cast(Any, _OverlappingInner()),
        cast(ClusterFeatureContract, _OverlappingContract()),
    )

    candidates = index.candidates_for_feature(10)
    gc.collect()

    assert [candidate.prefix_text for candidate in candidates] == ["member-10"]
    assert references[0]() is candidates[0]
    assert references[1]() is None

    del candidates
    gc.collect()
    assert all(reference() is None for reference in references)


def _run_rss_worker(mode: str, concepts: int) -> dict[str, int]:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "tests.axbench_base._stage0c_memory_worker",
            "--mode",
            mode,
            "--concepts",
            str(concepts),
        ],
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    return cast(dict[str, int], json.loads(completed.stdout))


@pytest.mark.parametrize("mode", ["ordinary", "clustered"])
def test_subprocess_rss_does_not_scale_with_visited_concepts(
    mode: str,
) -> None:
    one_concept = _run_rss_worker(mode, 1)
    many_concepts = _run_rss_worker(mode, 12)
    tolerance_bytes = 32 * 1024 * 1024

    assert (
        many_concepts["max_active_rss"]
        <= one_concept["max_active_rss"] + tolerance_bytes
    )
    assert (
        many_concepts["max_after_drop_rss"]
        <= one_concept["max_after_drop_rss"] + tolerance_bytes
    )
