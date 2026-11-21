"""Fresh-process RSS probe for Stage 0c candidate ownership tests."""

from __future__ import annotations

import argparse
import gc
import json
from types import MethodType, SimpleNamespace
from typing import Any, cast

import psutil

from steering.eval.axbench_base import stage0c_runner as runner_mod
from steering.eval.axbench_base._stage0c_internals import (
    _CandidatePrefixIndex,
    _PrefixCandidate,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.cluster_feature_contract import ClusterFeatureContract

_CANDIDATES_PER_FEATURE = 8
_PREFIX_BYTES = 1024 * 1024


def _synthetic_index() -> _CandidatePrefixIndex:
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
                shard_idx=0,
                doc_idx=feature_id * 100 + candidate_idx,
                seg_idx=candidate_idx,
                prefix_text=(
                    f"{feature_id}:{candidate_idx}:"
                    + ("x" * _PREFIX_BYTES)
                ),
                prefix_gemma_tokens=[feature_id, candidate_idx],
                source_bridge_text=f"bridge-{feature_id}-{candidate_idx}",
            )
            for candidate_idx in range(_CANDIDATES_PER_FEATURE)
        ]

    index._load_shard = MethodType(load_shard, index)  # ty: ignore[invalid-assignment]
    index._materialize_shard_candidates = MethodType(  # ty: ignore[invalid-assignment]
        materialize,
        index,
    )
    return index


class _SyntheticClusterContract:
    def target_for_representative(self, representative_id: int) -> Any:
        return SimpleNamespace(
            allowed_member_ids=(
                representative_id,
                representative_id + 1,
                representative_id + 2,
            )
        )


def _run(mode: str, n_concepts: int) -> dict[str, int]:
    process = psutil.Process()
    index = _synthetic_index()
    lookup: Any = index
    if mode == "clustered":
        lookup = runner_mod._ClusterCandidatePrefixIndex(
            index,
            cast(ClusterFeatureContract, _SyntheticClusterContract()),
        )

    gc.collect()
    baseline_rss = int(process.memory_info().rss)
    max_active_rss = baseline_rss
    max_after_drop_rss = baseline_rss
    for concept_idx in range(n_concepts):
        feature_id = concept_idx if mode == "ordinary" else concept_idx * 3
        candidates = lookup.candidates_for_feature(feature_id)
        max_active_rss = max(
            max_active_rss,
            int(process.memory_info().rss),
        )
        del candidates
        gc.collect()
        max_after_drop_rss = max(
            max_after_drop_rss,
            int(process.memory_info().rss),
        )

    return {
        "baseline_rss": baseline_rss,
        "max_active_rss": max_active_rss,
        "max_after_drop_rss": max_after_drop_rss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode",
        choices=("ordinary", "clustered"),
        required=True,
    )
    parser.add_argument("--concepts", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(_run(args.mode, args.concepts), sort_keys=True))


if __name__ == "__main__":
    main()
