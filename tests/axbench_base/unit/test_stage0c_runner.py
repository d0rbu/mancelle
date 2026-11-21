"""Unit tests for the per-rank Stage 0c runner (PR-4 Tasks 5+6).

Covers:
* world_size=1 byte-identity to legacy single-rank loop
* multi-rank stripe-partitioning + accept disjointness (gloo)
* block-barrier termination once global accepts hit max_concepts (gloo)
* resume after a mid-loop crash
* run_merge canonical concept_id remapping
* run_merge truncation at max_concepts
* run_merge fingerprint validation across ranks
* dispatcher world_size=1 short-circuits dist.all_gather_object
* dispatcher invokes compute_init with cfg.device
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from steering.conditional.percentile_table import (
    build_percentile_table,
    compute_percentile_checksum,
)
from steering.eval.axbench_base import _stage0c_internals as si_mod
from steering.eval.axbench_base import cluster_artifacts as cluster_artifacts_mod
from steering.eval.axbench_base import extraction, paths
from steering.eval.axbench_base import stage0c_partition as partition_mod
from steering.eval.axbench_base import stage0c_runner as runner_mod
from steering.eval.axbench_base._stage0c_internals import (
    _compute_checkpoint_fingerprint,
    _fingerprint_with_pool,
    _VerifiedNegativePrefix,
    candidate_pool_identity,
)
from steering.eval.axbench_base.concept_examples_qc import QCFeatureScore
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.stage0c_partition import (
    _RANK_CHECKPOINT_SCHEMA_VERSION,
    write_per_rank_checkpoint,
)
from steering.eval.axbench_base.stage0c_runner import (
    _DEFAULT_BLOCK_SIZE,
    _accept_dict_for_concept,
    _block_size,
    _build_payload,
    _cleanup_per_rank_partial_checkpoints,
    _qc_resume_start_at,
    _reconcile_resumed_state_for_rank,
    _row_dicts_for_accept,
    _try_resume_per_rank,
    run_merge,
    run_single_rank,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    QcOutcome,
    RejectedConceptReason,
)
from steering.eval.cluster_checkpoint_pin import _statistics_identity
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureTarget,
)
from steering.feature_paths import DatasetPhase, PoolingMode, build_feature_dir
from steering.sae_resolver import ActivationType, SAEWidth
from steering.stats_io import load_statistics, save_percentile_checksum
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.steering_stats import write_steering_stats

# ---------------------------------------------------------------------------
# Test scaffolding
# ---------------------------------------------------------------------------


def _write_minimal_stats(path: Path, n_sae_features: int = 16384) -> Path:
    """Write a minimal feature_statistics safetensors with the expected
    n_sae_features header — enough for read_n_sae_features_from_stats."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )
    return path


def _rt(cfg: AxBenchBaseConfig) -> str:
    """Narrow ``cfg.run_tag`` to ``str`` for paths.* calls."""
    assert cfg.run_tag is not None
    return cfg.run_tag


def _build_cfg(
    *,
    run_root: Path,
    run_tag: str = "test-runner",
    max_concepts: int = 4,
    enable_qc: bool = False,
    target_surviving: int | None = None,
    block_size: int | None = None,
    n_verified_target: int = 2,
    extraction_tokenizer_id: str = "google/gemma-2-2b",
) -> AxBenchBaseConfig:
    """Build a deterministic AxBenchBaseConfig for runner tests."""
    stats_path = _write_minimal_stats(run_root / "stats.safetensors")
    cfg = AxBenchBaseConfig(
        feature_dir=str(run_root / "features" / "s128"),
        feature_statistics_path=str(stats_path),
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=None,
        run_tag=run_tag,
        max_concepts=max_concepts,
        enable_concept_examples_qc=enable_qc,
        concept_examples_qc_target_surviving=target_surviving,
        n_verified_prefixes_per_concept_target=n_verified_target,
        n_verification_seeds=2,
        min_consistency_passes=1,
        verification_percentile_threshold=0.5,
        verification_length=4,
        min_prefix_chars=1,
        extraction_tokenizer_id=extraction_tokenizer_id,
        verification_api_model=None,
    )
    if block_size is not None:
        cfg = cfg.replace(stage0c_multi_gpu_block_size=block_size)
    return cfg


def _build_candidates(n: int) -> list[CandidateConcept]:
    """Build N deterministic CandidateConcept entries."""
    return [
        CandidateConcept(
            feature_index=100 + i,
            layer=8,
            sae_source="8-gemmascope-res-16k",
            description=f"feature_{i}",
            corpus_count=10 * (i + 1),
            description_source="neuronpedia",
        )
        for i in range(n)
    ]


def _write_live_candidate_pool(
    cfg: AxBenchBaseConfig,
    candidates: list[CandidateConcept],
) -> Path:
    path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([asdict(candidate) for candidate in candidates]),
        encoding="utf-8",
    )
    return path


@dataclass
class _StubVerifyPlan:
    """Plan for the stub _verify_concept: returns N rows for each candidate
    indexed by concept.feature_index. Use n=0 to model a verification
    rejection."""

    rows_by_feature_index: dict[int, int]

    def __call__(
        self,
        cfg: Any, concept: Any, parquets: Any, feature_shards: Any,
        base_model: Any, base_sae: Any, percentile_table: Any,
        *, sae_layer: int, base_context_window: int, candidate_index: Any,
        **_kwargs: Any,
    ) -> list[_VerifiedNegativePrefix]:
        n_rows = self.rows_by_feature_index.get(concept.feature_index, 0)
        return [
            _VerifiedNegativePrefix(
                concept_id=concept.concept_id,
                prefix_text=f"prefix-{concept.feature_index}-{i}",
                prefix_source_shard=0,
                prefix_source_doc=i,
                prefix_activating_seg_idx=i,
                verification_continuations=[f"cont-{i}"],
                verification_seeds=[1000 + i],
                verification_percentiles=[80.0],
            )
            for i in range(n_rows)
        ]


def _patch_runner_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    *,
    rows_by_feature_index: dict[int, int],
) -> _StubVerifyPlan:
    """Patch _verify_concept used by the runner to return canned rows."""
    plan = _StubVerifyPlan(rows_by_feature_index=rows_by_feature_index)
    monkeypatch.setattr(si_mod, "_verify_concept", plan)
    return plan


def _read_rank_payload(run_root: Path, run_tag: str | None, rank: int) -> dict[str, Any]:
    assert run_tag is not None
    json_path = test_paths.default_paths(run_tag).stage0c_partial_rank_json(rank)
    return json.loads(json_path.read_text(encoding="utf-8"))


def _read_rank_rows(run_root: Path, run_tag: str | None, rank: int) -> list[dict[str, Any]]:
    assert run_tag is not None
    parquet_path = test_paths.default_paths(run_tag).stage0c_partial_rank_parquet(rank)
    if not parquet_path.exists():
        return []
    return pq.read_table(parquet_path).to_pylist()


# ---------------------------------------------------------------------------
# Tests for the small helpers
# ---------------------------------------------------------------------------


def test_block_size_default_when_unset(tmp_path):
    cfg = _build_cfg(run_root=tmp_path)
    assert _block_size(cfg) == _DEFAULT_BLOCK_SIZE


def test_block_size_uses_cfg_when_set(tmp_path):
    cfg = _build_cfg(run_root=tmp_path, block_size=4)
    assert _block_size(cfg) == 4


def test_block_size_adapts_to_max_concepts_under_ddp(tmp_path):
    """With multi-rank DDP and a small ``max_concepts``, the configured
    block size is capped at ``ceil(max_concepts * 2 / world_size)`` so
    one global block doesn't overshoot the target by an order of
    magnitude. Observed 2026-05-15: a ``max_concepts=10`` / ``ws=4`` /
    ``block=16`` run had rank 0 finish its 16 candidates with 13
    accepts then spin 90 min on NCCL waiting for the other ranks to
    finish bonus work before the global ``>= max_concepts`` check
    could fire.
    """
    # max_concepts=10, ws=4 → per_rank_quota = ceil(20/4) = 5.
    # Configured 16 gets capped at 5. Block spans 5*4 = 20 candidates
    # globally — enough to fill 10 accepts with typical pass rates,
    # not the 64-candidate overshoot.
    cfg = _build_cfg(run_root=tmp_path, max_concepts=10, block_size=16)
    assert _block_size(cfg, world_size=4) == 5

    # max_concepts=50, ws=4 → per_rank_quota = 25, capped by configured 16.
    cfg_standard = _build_cfg(run_root=tmp_path, max_concepts=50, block_size=16)
    assert _block_size(cfg_standard, world_size=4) == 16

    # max_concepts=500, ws=8 (full tier) → per_rank_quota = 125, cap=16.
    cfg_full = _build_cfg(run_root=tmp_path, max_concepts=500, block_size=16)
    assert _block_size(cfg_full, world_size=8) == 16

    # max_concepts=10, ws=8 → per_rank_quota = ceil(20/8) = 3. Cap kicks in.
    cfg_small = _build_cfg(run_root=tmp_path, max_concepts=10, block_size=16)
    assert _block_size(cfg_small, world_size=8) == 3


def test_block_size_world_size_1_returns_configured(tmp_path):
    """At single-rank (no DDP), the adaptive cap is a no-op — the
    block-end rendezvous is local, so over-processing doesn't cost
    cross-rank wait time. Preserves byte-identical output to the
    legacy single-rank loop.
    """
    cfg = _build_cfg(run_root=tmp_path, max_concepts=10, block_size=16)
    # Default world_size=1 path returns configured value.
    assert _block_size(cfg) == 16
    assert _block_size(cfg, world_size=1) == 16


def test_block_size_clamps_at_one_for_tiny_target(tmp_path):
    """``max_concepts=1`` with any DDP topology still returns >= 1.
    Otherwise the runner would compute ``range(0, 0)`` and never visit
    a single candidate. Pinning the floor keeps the runner correct in
    pathological tier configurations.
    """
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=16)
    assert _block_size(cfg, world_size=4) == 1
    # max_concepts=0 (edge case): per_rank_quota = ceil(0/4) = 0; clamped
    # to 1 by the ``max(1, ...)`` guard.
    cfg_zero = _build_cfg(run_root=tmp_path, max_concepts=0, block_size=16)
    # max_concepts=0 may be rejected upstream by validate_config; we only
    # pin that the helper itself doesn't return 0.
    assert _block_size(cfg_zero, world_size=4) >= 1


def test_block_size_never_returns_below_one_even_if_configured_zero(tmp_path):
    """D1: effective B must be >= 1 unconditionally. ``configured == 0``
    is reachable because unit tests construct AxBenchBaseConfig directly
    (validate_config bypassed). Both the ws<=1 passthrough and the
    multi-rank cap path must clamp to 1, never 0 (B=0 → ValueError in
    my_candidate_block and covers==0 in the termination predicate).
    """
    cfg0 = _build_cfg(run_root=tmp_path, max_concepts=10, block_size=0)
    assert _block_size(cfg0, world_size=1) == 1   # ws<=1 path clamps
    assert _block_size(cfg0, world_size=4) == 1   # min(0,quota)→0→clamp
    # max_concepts negative (also unvalidated at construction): clamp.
    cfg_neg = _build_cfg(run_root=tmp_path, max_concepts=-3, block_size=16)
    assert _block_size(cfg_neg, world_size=4) == 1


@pytest.mark.parametrize(
    "configured,max_concepts,world_size,expected",
    [
        (16, 10, 1, 16),
        (16, 10, 0, 16),
        (4, 500, 1, 4),
        (64, 10, 1, 64),
        (0, 10, 1, 1),
        (16, 50, 4, 16),
        (16, 500, 8, 16),
        (4, 100, 2, 4),
        (16, 10, 4, 5),
        (16, 10, 8, 3),
        (16, 1, 4, 1),
        (16, 2, 4, 1),
        (8, 3, 4, 2),
        (16, 10, 64, 1),
        (0, 10, 4, 1),
        (16, 0, 4, 1),
        (16, -3, 4, 1),
    ],
)
def test_block_size_table(tmp_path, configured, max_concepts, world_size, expected):
    """Exhaustive contract table for _block_size. Subsumes the 3 branch
    tests; covers ws<=1 passthrough, cap-inert, cap-binds, ceil
    boundaries, and the D1 clamp. Formula: ws<=1 → max(1, configured);
    else max(1, min(configured, ceil(2*max_concepts/ws))).
    """
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=max_concepts, block_size=configured,
    )
    b = _block_size(cfg, world_size=world_size)
    assert b == expected
    assert b >= 1
    if configured >= 1:
        assert b <= configured
    if world_size > 1 and configured >= 1 and max_concepts >= 1:
        cap = math.ceil(max_concepts * 2 / world_size)
        assert b == max(1, min(configured, cap))


def test_accept_dict_round_trip(tmp_path):
    cands = _build_candidates(1)
    accept = _accept_dict_for_concept(
        cands[0],
        candidate_index=7, qc_score=2, n_verified_rows=3,
    )
    assert accept["candidate_index"] == 7
    assert accept["feature_index"] == 100
    assert accept["qc_score"] == 2
    assert accept["verified_rows_count"] == 3
    # Carries every Concept-essential field.
    for k in (
        "feature_index", "layer", "sae_source",
        "description", "corpus_count", "description_source",
    ):
        assert k in accept


def test_row_dicts_for_accept_assigns_placeholder_concept_id(tmp_path):
    rows = [
        _VerifiedNegativePrefix(
            concept_id=99, prefix_text="x", prefix_source_shard=0,
            prefix_source_doc=0, prefix_activating_seg_idx=0,
            verification_continuations=["c"], verification_seeds=[1],
            verification_percentiles=[80.0],
        ),
    ]
    out = _row_dicts_for_accept(rows, candidate_index=5, placeholder_concept_id=2)
    assert out[0]["concept_id"] == 2  # placeholder, not the original 99
    assert out[0]["candidate_index"] == 5


def test_build_payload_includes_fingerprint(tmp_path):
    cfg = _build_cfg(run_root=tmp_path)
    payload = _build_payload(
        cfg, rank=0, world_size=2, n_candidates=10,
        next_block_index=0, visited_candidate_indices=[],
        rank_accepts=[], rank_rejects=[], rank_qc_per_feature=[],
        candidate_pool_identity="pool-id",
        precomputed_clustered_fields={},
    )
    assert payload["fingerprint"] == _fingerprint_with_pool(
        cfg, candidate_pool_identity="pool-id",
    )
    assert payload["rank"] == 0
    assert payload["world_size"] == 2


def test_build_payload_embeds_candidate_pool_identity(tmp_path):
    """``_build_payload`` threads the candidate-pool identity into the
    checkpoint fingerprint under the ``candidate_pool_identity`` key so a
    regenerated pool invalidates resume."""
    cfg = _build_cfg(run_root=tmp_path)
    payload = _build_payload(
        cfg, rank=0, world_size=1, n_candidates=3,
        next_block_index=0, visited_candidate_indices=[],
        rank_accepts=[], rank_rejects=[], rank_qc_per_feature=[],
        candidate_pool_identity="X",
        precomputed_clustered_fields={},
    )
    assert payload["fingerprint"]["candidate_pool_identity"] == "X"


def test_fingerprint_includes_effective_feature_extraction_model_id(tmp_path):
    cfg = _build_cfg(run_root=tmp_path).replace(
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-65k",
    )
    fingerprint = _compute_checkpoint_fingerprint(cfg)
    assert fingerprint["neuronpedia_model_id"] == "gemma-3-4b"
    assert fingerprint["feature_extraction_model_id"] == "gemma-3-4b-pt"
    assert fingerprint["neuronpedia_source"] == "22-gemmascope-2-res-65k"


def test_cleanup_per_rank_partials_idempotent(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    # No-op when nothing exists.
    _cleanup_per_rank_partial_checkpoints(cfg)
    # Idempotent: a second call after explicit cleanup is fine.
    _cleanup_per_rank_partial_checkpoints(cfg)


def test_cleanup_requires_run_tag(tmp_path):
    cfg = _build_cfg(run_root=tmp_path)
    cfg = cfg.replace(run_tag=None)
    with pytest.raises(AxBenchBaseError, match="run_tag"):
        _cleanup_per_rank_partial_checkpoints(cfg)


def test_checkpoint_fingerprint_omits_cluster_fields_when_unclustered(tmp_path):
    """U-invariant (PR #739 C4): an unclustered cfg's checkpoint fingerprint
    carries none of the clustered-only identity fields.
    ``clustered_stage0c_fingerprint_fields`` (PR #739 C1, folded into
    ``_stage0c_extractor_identity`` in ``_stage0c_internals.py`` -- untouched
    by this port) returns ``{}`` for ``cfg.conditioning=None``, so this
    runner's checkpoint/extractor identity is byte-unchanged for the
    non-clustered path every existing production run still uses.
    """
    cfg = _build_cfg(run_root=tmp_path)
    assert cfg.conditioning is None
    fingerprint = _compute_checkpoint_fingerprint(cfg)
    cluster_only_keys = {
        "feature_contract_fingerprint",
        "candidate_cluster_provenance_sha256",
        "cluster_concept_projection_version",
    }
    assert not cluster_only_keys & set(fingerprint)


# ---------------------------------------------------------------------------
# Tests for _ClusterCandidatePrefixIndex (PR #739 C4)
# ---------------------------------------------------------------------------


class _FakeInnerCandidateIndex:
    """Records lookups and returns canned candidates per feature_id."""

    def __init__(self, candidates_by_feature: dict[int, list[Any]]) -> None:
        self._by_feature = candidates_by_feature
        self.calls: list[int] = []

    def candidates_for_feature(self, feature_id: int) -> list[Any]:
        self.calls.append(feature_id)
        return list(self._by_feature.get(feature_id, []))


class _FakeClusterContract:
    """Minimal duck-typed stand-in for ``ClusterFeatureContract`` --
    ``_ClusterCandidatePrefixIndex`` only ever calls
    ``target_for_representative``, so a full contract (SAE/statistics/
    checkpoint identity) is unnecessary machinery for this unit. Mirrors the
    existing ``ClusterFeatureTarget``-only pattern already used by
    ``test_clustered_candidate_scan_visits_every_member_uncapped`` in
    ``test_negative_prefixes.py``.
    """

    def __init__(self, targets_by_representative: dict[int, ClusterFeatureTarget]) -> None:
        self._targets = targets_by_representative

    def target_for_representative(self, representative_id: int) -> ClusterFeatureTarget:
        return self._targets[representative_id]


def test_cluster_candidate_prefix_index_unions_all_member_occurrences_and_dedupes():
    """``candidates_for_feature`` must: (1) resolve the representative's
    ``ClusterFeatureTarget``, (2) query the inner index for EVERY allowed
    member id (not only the representative), and (3) run the combined pool
    through ``_canonical_cluster_candidate_union`` so a location rediscovered
    by two different members collapses to one row in canonical
    ``(shard, doc, seg)`` order.
    """
    rep_only = si_mod._PrefixCandidate(
        shard_idx=0, doc_idx=0, seg_idx=0,
        prefix_text="rep-only", prefix_gemma_tokens=[1],
    )
    shared_from_rep = si_mod._PrefixCandidate(
        shard_idx=0, doc_idx=1, seg_idx=0,
        prefix_text="rep-sees-shared", prefix_gemma_tokens=[2],
    )
    shared_from_member = si_mod._PrefixCandidate(
        shard_idx=0, doc_idx=1, seg_idx=0,
        prefix_text="member-sees-shared", prefix_gemma_tokens=[3],
    )
    member_only = si_mod._PrefixCandidate(
        shard_idx=1, doc_idx=0, seg_idx=0,
        prefix_text="member-only", prefix_gemma_tokens=[4],
    )
    inner = _FakeInnerCandidateIndex({
        3: [rep_only, shared_from_rep],
        9: [shared_from_member, member_only],
    })
    target = ClusterFeatureTarget(
        representative_id=3,
        allowed_member_ids=(3, 9),
        projecting_source_ids=(3, 9),
        feature_contract_fingerprint="a" * 64,
    )
    contract = _FakeClusterContract({3: target})
    index = runner_mod._ClusterCandidatePrefixIndex(
        inner, cast(ClusterFeatureContract, contract)
    )

    result = index.candidates_for_feature(3)

    # Queried the inner index for BOTH members, not only the representative.
    assert sorted(inner.calls) == [3, 9]
    # Union dedupes the shared (shard=0, doc=1, seg=0) location -- first-seen
    # (member 3's scan is unioned before member 9's) wins -- and results come
    # back in canonical (shard, doc, seg) order, not scan order.
    assert result == [rep_only, shared_from_rep, member_only]


def test_cluster_candidate_prefix_index_applies_configured_scan_limit_post_union():
    """The configured ``candidate_scan_limit`` is threaded through to
    ``_canonical_cluster_candidate_union`` as a SHARED budget across the
    whole cluster, applied after cross-member dedup -- not a per-member
    budget (which would let a 2-member cluster through with 2x candidates).
    """
    member_a_candidates = [
        si_mod._PrefixCandidate(
            shard_idx=0, doc_idx=i, seg_idx=0,
            prefix_text=f"a{i}", prefix_gemma_tokens=[i],
        )
        for i in range(3)
    ]
    member_b_candidates = [
        si_mod._PrefixCandidate(
            shard_idx=0, doc_idx=i + 10, seg_idx=0,
            prefix_text=f"b{i}", prefix_gemma_tokens=[i],
        )
        for i in range(3)
    ]
    inner = _FakeInnerCandidateIndex({5: member_a_candidates, 6: member_b_candidates})
    target = ClusterFeatureTarget(
        representative_id=5,
        allowed_member_ids=(5, 6),
        projecting_source_ids=(5, 6),
        feature_contract_fingerprint="b" * 64,
    )
    contract = _FakeClusterContract({5: target})
    index = runner_mod._ClusterCandidatePrefixIndex(
        inner,
        cast(ClusterFeatureContract, contract),
        candidate_scan_limit=4,
    )

    result = index.candidates_for_feature(5)

    # 6 distinct locations across both members in one shard, capped at 4.
    assert len(result) == 4


def test_cluster_candidate_prefix_index_uncapped_when_scan_limit_is_none():
    candidates = [
        si_mod._PrefixCandidate(
            shard_idx=0, doc_idx=i, seg_idx=0,
            prefix_text=f"c{i}", prefix_gemma_tokens=[i],
        )
        for i in range(5)
    ]
    inner = _FakeInnerCandidateIndex({7: candidates})
    target = ClusterFeatureTarget(
        representative_id=7,
        allowed_member_ids=(7,),
        projecting_source_ids=(7,),
        feature_contract_fingerprint="c" * 64,
    )
    contract = _FakeClusterContract({7: target})
    index = runner_mod._ClusterCandidatePrefixIndex(
        inner, cast(ClusterFeatureContract, contract)
    )

    assert len(index.candidates_for_feature(7)) == 5


# ---------------------------------------------------------------------------
# Tests for run_single_rank — single rank (world_size=1)
# ---------------------------------------------------------------------------


def test_run_single_rank_world_size_1_processes_all_candidates_writes_per_rank(
    tmp_path, monkeypatch,
):
    """At world_size=1, the runner visits every candidate in
    candidate_concepts order and writes a per-rank checkpoint file."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2, block_size=2)
    cands = _build_candidates(4)
    # Make every candidate accept.
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    del plan

    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )

    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    rows = _read_rank_rows(tmp_path, cfg.run_tag, rank=0)
    # max_concepts=2 means the loop short-circuits after 2 accepts at ws=1.
    assert len(payload["rank_accepts"]) == 2
    assert payload["rank_accepts"][0]["candidate_index"] == 0
    assert payload["rank_accepts"][1]["candidate_index"] == 1
    # 2 rows per accept.
    assert len(rows) == 4
    # Every row has a candidate_index.
    assert all("candidate_index" in r for r in rows)
    rank_parquet = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    assert pq.read_schema(rank_parquet).metadata == {
        b"extraction_tokenizer_id": cfg.extraction_tokenizer_id.encode("utf-8")
    }


def test_run_single_rank_world_size_1_byte_identical_to_legacy_logic(
    tmp_path, monkeypatch,
):
    """Stricter byte-identity contract: per-rank rows match the order +
    content of the legacy loop, when stripped of the placeholder
    concept_id (which the merge will overwrite anyway)."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=3, block_size=2)
    cands = _build_candidates(3)
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    del plan

    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    rows = _read_rank_rows(tmp_path, cfg.run_tag, rank=0)
    accepted = payload["rank_accepts"]
    assert [a["candidate_index"] for a in accepted] == [0, 1, 2]
    # Rows preserve candidate-index order (legacy parity).
    assert [r["candidate_index"] for r in rows] == [0, 0, 1, 1, 2, 2]
    # Rows preserve their per-prefix doc_idx ordering.
    assert [r["prefix_source_doc"] for r in rows] == [0, 1, 0, 1, 0, 1]


def test_run_single_rank_world_size_1_verification_rejection(
    tmp_path, monkeypatch,
):
    """Candidates that yield 0 verified rows go to rank_rejects, not
    rank_accepts."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2, block_size=2)
    cands = _build_candidates(3)
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={
            cands[0].feature_index: 0,
            cands[1].feature_index: 2,
            cands[2].feature_index: 2,
        },
    )
    del plan

    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    accepts = [a["candidate_index"] for a in payload["rank_accepts"]]
    rejects = [
        r["candidate_index"] for r in payload["rank_rejects"]
    ]
    assert accepts == [1, 2]
    assert rejects == [0]
    assert payload["rank_rejects"][0]["reason"] == (
        RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value
    )
    assert payload["rank_rejects"][0]["n_verified_found"] == 0


def test_run_single_rank_releases_cuda_after_each_verified_candidate(
    tmp_path,
    monkeypatch,
):
    """Stage 0c releases cached CUDA allocations after every verification."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    calls: list[str] = []
    monkeypatch.setattr(
        runner_mod,
        "release_cuda_iteration_memory",
        lambda: calls.append("release"),
    )
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2, block_size=2)
    cands = _build_candidates(3)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={
            cands[0].feature_index: 0,
            cands[1].feature_index: 2,
            cands[2].feature_index: 2,
        },
    )

    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )

    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert [a["candidate_index"] for a in payload["rank_accepts"]] == [1, 2]
    assert [r["candidate_index"] for r in payload["rank_rejects"]] == [0]
    assert calls == ["release", "release", "release"]


@pytest.mark.parametrize("verified_rows", [0, 2])
def test_run_single_rank_releases_api_concept_state_before_cuda(
    tmp_path,
    monkeypatch,
    verified_rows,
):
    """Definition cleanup covers both verification accept and rejection."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    events: list[str] = []

    class _ApiGenerator:
        def _release_concept_state(self) -> None:
            events.append("definitions")

    monkeypatch.setattr(
        runner_mod,
        "release_cuda_iteration_memory",
        lambda: events.append("cuda"),
    )
    monkeypatch.setattr(
        runner_mod._si,
        "_log_candidate_completion",
        lambda **_kwargs: events.append("heartbeat"),
    )
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=1)
    candidate = _build_candidates(1)[0]
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={candidate.feature_index: verified_rows},
    )

    run_single_rank(
        cfg,
        [candidate],
        rank=0,
        world_size=1,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[],
        feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        verification_api_generator=cast(Any, _ApiGenerator()),
        precomputed_clustered_fields={},
    )

    assert events == ["definitions", "cuda", "heartbeat"]


def test_run_single_rank_releases_api_concept_state_on_verification_exception(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    events: list[str] = []

    class _ApiGenerator:
        def _release_concept_state(self) -> None:
            events.append("definitions")

    def raise_during_verification(*args, **kwargs):
        del args, kwargs
        raise RuntimeError("verification failed")

    monkeypatch.setattr(si_mod, "_verify_concept", raise_during_verification)
    monkeypatch.setattr(
        runner_mod,
        "release_cuda_iteration_memory",
        lambda: events.append("cuda"),
    )
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=1)

    with pytest.raises(RuntimeError, match="verification failed"):
        run_single_rank(
            cfg,
            _build_candidates(1),
            rank=0,
            world_size=1,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[],
            feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            verification_api_generator=cast(Any, _ApiGenerator()),
            precomputed_clustered_fields={},
        )

    assert events == ["definitions", "cuda"]


def test_run_single_rank_releases_cuda_if_api_concept_cleanup_raises(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    events: list[str] = []

    class _ApiGenerator:
        def _release_concept_state(self) -> None:
            events.append("definitions")
            raise RuntimeError("definition cleanup failed")

    monkeypatch.setattr(
        runner_mod,
        "release_cuda_iteration_memory",
        lambda: events.append("cuda"),
    )
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=1)
    candidate = _build_candidates(1)[0]
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={candidate.feature_index: 2},
    )

    with pytest.raises(RuntimeError, match="definition cleanup failed"):
        run_single_rank(
            cfg,
            [candidate],
            rank=0,
            world_size=1,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[],
            feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            verification_api_generator=cast(Any, _ApiGenerator()),
            precomputed_clustered_fields={},
        )

    assert events == ["definitions", "cuda"]


def test_run_single_rank_with_qc_buffer_rejected(tmp_path, monkeypatch):
    """When the QC client rejects a candidate, the runner records a
    rank_reject and skips verification for it."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2,
                     enable_qc=True)
    cands = _build_candidates(2)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    # Stub QC client. .filter returns (survivors, report) where
    # report.per_feature contains QCFeatureScore for each item.
    class _StubQcClient:
        def filter(self, batch, target_surviving=None):
            del target_surviving
            scores = []
            for fi, _desc in batch:
                outcome = (
                    QcOutcome.ACCEPTED
                    if fi == cands[1].feature_index
                    else QcOutcome.REJECTED
                )
                scores.append(QCFeatureScore(
                    feature_index=fi, description="x",
                    outcome=outcome,
                    score=2 if outcome is QcOutcome.ACCEPTED else 0,
                    n_examples=1, error_message=None,
                ))
            report = MagicMock(per_feature=scores)
            return [], report

    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=_StubQcClient(),
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert [a["candidate_index"] for a in payload["rank_accepts"]] == [1]
    assert [r["candidate_index"] for r in payload["rank_rejects"]] == [0]
    assert payload["rank_rejects"][0]["reason"] == (
        RejectedConceptReason.LLM_JUDGE_BELOW_MIN_SCORE.value
    )
    assert payload["rank_qc_per_feature"]
    assert "next_local_qc_concept_id" not in payload


def test_run_single_rank_resume_no_op_if_already_complete(
    tmp_path, monkeypatch,
):
    """A second invocation after a complete first run is a no-op:
    visited_candidate_indices is already populated, and resume reads
    the same state back."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cands = _build_candidates(2)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    for _ in range(2):
        run_single_rank(
            cfg, cands,
            rank=0, world_size=1,
            base_model=MagicMock(), base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[], feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            precomputed_clustered_fields={},
        )
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    # max_concepts=1 means we accept exactly 1, even after rerunning.
    assert len(payload["rank_accepts"]) == 1
    rows = _read_rank_rows(tmp_path, cfg.run_tag, rank=0)
    assert len(rows) == 2  # n_verified_target=2, 1 accepted candidate


def test_run_single_rank_resume_invalid_fingerprint_raises(
    tmp_path, monkeypatch,
):
    """Resume from a per-rank checkpoint with mismatched fingerprint
    raises AxBenchBaseError."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    # Hand-craft a corrupted per-rank JSON.
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    bad_payload = {
        "schema_version": _RANK_CHECKPOINT_SCHEMA_VERSION,
        "rank": 0, "world_size": 1, "n_candidates": 4,
        "fingerprint": {"feature_dir": "/bogus"},  # wrong field set
        "next_block_index": 0,
        "visited_candidate_indices": [],
        "rank_accepts": [], "rank_rejects": [],
        "rank_qc_per_feature": [],
    }
    json_path.write_text(json.dumps(bad_payload), encoding="utf-8")

    cands = _build_candidates(4)
    with pytest.raises(AxBenchBaseError, match="fingerprint"):
        run_single_rank(
            cfg, cands,
            rank=0, world_size=1,
            base_model=MagicMock(), base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[], feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            precomputed_clustered_fields={},
        )


# ---------------------------------------------------------------------------
# Tests for run_merge
# ---------------------------------------------------------------------------


def _seed_runner_state_for_merge(
    cfg: AxBenchBaseConfig,
    *,
    rank: int,
    world_size: int,
    candidates_owned: list[int],
    accept_indices: list[int],
    reject_indices: list[int],
    n_rows_per_accept: int = 2,
    n_pool: int | None = None,
) -> None:
    """Directly synthesize a per-rank checkpoint for merge tests.

    ``n_pool`` fixes the shared candidate-pool size so every rank in a
    multi-rank merge computes the SAME ``candidate_pool_identity`` (all
    ranks see one identical candidate pool in production). When ``None``
    the pool is sized to this rank's own max index — fine for single-rank
    merges, but multi-rank merges MUST pass a shared ``n_pool`` or the
    pooled fingerprints legitimately differ and merge rejects them.
    """
    cands = _build_candidates(
        n_pool
        if n_pool is not None
        else max(candidates_owned + accept_indices + reject_indices) + 1
    )
    _write_live_candidate_pool(cfg, cands)
    accepts = []
    rows = []
    for placeholder, idx in enumerate(accept_indices):
        c = cands[idx]
        accepts.append(_accept_dict_for_concept(
            c,
            candidate_index=idx,
            qc_score=None,
            n_verified_rows=n_rows_per_accept,
        ))
        sample_rows = [
            _VerifiedNegativePrefix(
                concept_id=placeholder,
                prefix_text=f"r{idx}_{i}",
                prefix_source_shard=0,
                prefix_source_doc=i,
                prefix_activating_seg_idx=i,
                verification_continuations=["c"],
                verification_seeds=[1000 + i],
                verification_percentiles=[80.0],
            )
            for i in range(n_rows_per_accept)
        ]
        rows.extend(_row_dicts_for_accept(
            sample_rows, candidate_index=idx, placeholder_concept_id=placeholder,
        ))
    rejects = [
        {
            "candidate_index": idx,
            "feature_index": cands[idx].feature_index,
            "description": cands[idx].description,
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": None,
        }
        for idx in reject_indices
    ]
    # Resolve exactly like the production dispatcher does once at startup
    # (`si_mod.clustered_stage0c_fingerprint_fields`, a live read): {} for
    # every unclustered test cfg here, or the test's stubbed clustered
    # fields when `_stub_cluster_fingerprint_fields` is active. Mirrors
    # what a real `run_single_rank`'s threaded value would have been.
    precomputed_clustered_fields = si_mod.clustered_stage0c_fingerprint_fields(cfg)
    payload = _build_payload(
        cfg, rank=rank, world_size=world_size, n_candidates=len(cands),
        next_block_index=10,  # arbitrary
        visited_candidate_indices=candidates_owned,
        rank_accepts=accepts, rank_rejects=rejects,
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(cands),
        precomputed_clustered_fields=precomputed_clustered_fields,
    )
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(rank)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(rank)
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        payload,
        rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )


def test_run_merge_canonical_concept_id_remapping(tmp_path, monkeypatch):
    """Accepts spread across ranks merge to dense concept_ids 0..N-1
    sorted by candidate_index. Per-rank parquet rows are remapped to
    canonical concept_ids."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=4)
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=2,
        candidates_owned=[0, 2, 4],
        accept_indices=[0, 4],
        reject_indices=[2],
        n_pool=6,
    )
    _seed_runner_state_for_merge(
        cfg, rank=1, world_size=2,
        candidates_owned=[1, 3, 5],
        accept_indices=[1, 3],
        reject_indices=[5],
        n_pool=6,
    )

    out_path = run_merge(cfg, world_size=2)
    assert out_path.exists()

    accepted = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text(encoding="utf-8"),
    )
    assert [c["concept_id"] for c in accepted] == [0, 1, 2, 3]
    # Sorted by candidate_index 0, 1, 3, 4 → feature_index 100, 101, 103, 104
    assert [c["feature_index"] for c in accepted] == [100, 101, 103, 104]

    rejected = json.loads(
        paths.AxBenchBasePaths.from_config(cfg)
        .rejected_concepts_json()
        .read_text(encoding="utf-8"),
    )
    # Per the multi-GPU reject-truncation contract (see
    # ``merge_per_rank_checkpoints``): when the run saturates
    # ``max_concepts``, only rejects with candidate_index <= the
    # last canonical accept's candidate_index appear in the merged
    # rejected_concepts.json. Here truncated_accepts = [0, 1, 3, 4],
    # so max_accept_candidate_index = 4, and reject 5 (candidate_index=5)
    # is filtered out — matching what ws=1 would produce because at
    # ws=1 the loop would have stopped after the 4th accept (candidate
    # 4) and never visited candidate 5.
    assert [r["candidate_index"] for r in rejected] == [2]


def test_run_merge_truncates_to_max_concepts(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=2,
        candidates_owned=[0, 2, 4],
        accept_indices=[0, 2, 4],
        reject_indices=[],
        n_pool=6,
    )
    _seed_runner_state_for_merge(
        cfg, rank=1, world_size=2,
        candidates_owned=[1, 3, 5],
        accept_indices=[1, 3],
        reject_indices=[5],
        n_pool=6,
    )
    run_merge(cfg, world_size=2)
    accepted = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text(encoding="utf-8"),
    )
    # max_concepts=2 → first 2 in sorted candidate_index order: 0, 1.
    assert [c["concept_id"] for c in accepted] == [0, 1]
    assert [c["feature_index"] for c in accepted] == [100, 101]


def test_run_merge_validates_fingerprint_consistency_across_ranks(
    tmp_path, monkeypatch,
):
    """When ranks committed under different cfg fingerprints, merge raises."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=2,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    cfg_b = cfg.replace(max_concepts=999)  # changes fingerprint
    _seed_runner_state_for_merge(
        cfg_b, rank=1, world_size=2,
        candidates_owned=[1],
        accept_indices=[1],
        reject_indices=[],
    )
    with pytest.raises(AxBenchBaseError, match="fingerprint"):
        run_merge(cfg, world_size=2)


def test_run_merge_reloads_live_candidate_pool_before_any_write(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    changed_pool = _build_candidates(1)
    changed_pool[0] = replace(changed_pool[0], description="changed live pool")
    _write_live_candidate_pool(cfg, changed_pool)

    with pytest.raises(AxBenchBaseError, match="candidate pool changed"):
        run_merge(cfg, world_size=1)

    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0).exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes().exists()


def test_run_merge_rejects_denormalized_candidate_field_drift(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    rank_json = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    payload = json.loads(rank_json.read_text(encoding="utf-8"))
    payload["rank_accepts"][0]["description"] = "corrupt checkpoint copy"
    rank_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(
        AxBenchBaseError,
        match="accepted checkpoint state disagrees.*description",
    ):
        run_merge(cfg, world_size=1)

    assert rank_json.exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes().exists()


def test_run_merge_rejects_live_candidate_source_mismatch(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    changed_pool = [replace(_build_candidates(1)[0], sae_source="wrong-sae-source")]
    _write_live_candidate_pool(cfg, changed_pool)
    rank_json = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    payload = json.loads(rank_json.read_text(encoding="utf-8"))
    payload["rank_accepts"][0]["sae_source"] = "wrong-sae-source"
    payload["fingerprint"]["candidate_pool_identity"] = candidate_pool_identity(changed_pool)
    rank_json.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="uses SAE source"):
        run_merge(cfg, world_size=1)

    assert rank_json.exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes().exists()


def test_run_merge_raises_if_no_accepts(tmp_path, monkeypatch):
    """When the accept tally falls short of max_concepts, merge raises
    with an actionable error."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=1,
        candidates_owned=[0],
        accept_indices=[],
        reject_indices=[0],
    )
    with pytest.raises(AxBenchBaseError, match="exhausted"):
        run_merge(cfg, world_size=1)


def test_run_merge_cleans_up_partials_on_success(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=2,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
        n_pool=2,
    )
    _seed_runner_state_for_merge(
        cfg, rank=1, world_size=2,
        candidates_owned=[1],
        accept_indices=[1],
        reject_indices=[],
        n_pool=2,
    )
    out_path = run_merge(cfg, world_size=2)
    metadata = pq.read_schema(out_path).metadata or {}
    assert metadata[b"candidate_pool_identity"] == candidate_pool_identity(
        _build_candidates(2)
    ).encode("utf-8")
    # Per-rank files removed after canonical write.
    assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0).exists()
    assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0).exists()
    assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(1).exists()
    assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(1).exists()


def test_run_merge_world_size_1_byte_identical_to_legacy_artifacts(
    tmp_path, monkeypatch,
):
    """Single-rank merge produces concepts.json + rejected_concepts.json
    + negative_mode_prefixes.parquet that match what the legacy loop
    would have written."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    cands = _build_candidates(3)
    _write_live_candidate_pool(cfg, cands)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    out_path = run_merge(cfg, world_size=1)
    accepted = json.loads(
        paths.AxBenchBasePaths.from_config(cfg).concepts_json().read_text(encoding="utf-8"),
    )
    assert [c["concept_id"] for c in accepted] == [0, 1]
    assert [c["feature_index"] for c in accepted] == [100, 101]
    rows = pq.read_table(out_path).to_pylist()
    # 2 accepts × 2 rows each → 4 rows.
    assert len(rows) == 4
    # Rows mapped to canonical concept_id and stripped of candidate_index.
    assert sorted({r["concept_id"] for r in rows}) == [0, 1]
    assert all("candidate_index" not in r for r in rows)
    metadata = pq.read_schema(out_path).metadata or {}
    expected_base_metadata = {
        b"extraction_tokenizer_id": cfg.extraction_tokenizer_id.encode("utf-8"),
        b"neuronpedia_model_id": cfg.neuronpedia_model_id.encode("utf-8"),
        b"feature_extraction_model_id": cfg.neuronpedia_model_id.encode("utf-8"),
        b"neuronpedia_source": cfg.neuronpedia_source.encode("utf-8"),
        b"candidate_pool_identity": candidate_pool_identity(cands).encode("utf-8"),
    }
    assert {
        key: metadata[key] for key in expected_base_metadata
    } == expected_base_metadata
    for identity_key in (
        b"concepts_identity",
        b"rejected_concepts_identity",
        b"stage0c_fingerprint_identity",
    ):
        identity = metadata[identity_key].decode("ascii")
        assert len(identity) == 64
        assert set(identity) <= set("0123456789abcdef")


# ---------------------------------------------------------------------------
# Multi-rank tests via gloo subprocess
# ---------------------------------------------------------------------------


def _multirank_target(
    rank: int, world_size: int,
    tmp_path_str: str, run_tag: str,
    n_candidates: int, max_concepts: int, block_size: int,
):
    """Worker entry for the multi-rank gloo tests.

    Imported at module scope by ``torch.multiprocessing.spawn`` (via
    ``run_gloo_multiprocess``). Returns a dict with the per-rank state
    after running ``run_single_rank``; the parent then merges them.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)

    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag=run_tag,
        max_concepts=max_concepts,
        block_size=block_size,
    )
    cands = _build_candidates(n_candidates)
    # Make every candidate produce enough rows to be accepted.
    plan = _StubVerifyPlan(
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    # Rebind the canonical negative_prefixes symbol through an Any-typed
    # module view: this deliberately installs a callable test double whose
    # runtime behavior matches the function it replaces.
    # The production symbol itself remains the canonical reference.
    si_module: Any = si_mod
    si_module._verify_concept = plan

    run_single_rank(
        cfg, cands,
        rank=rank, world_size=world_size,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    payload = json.loads(
        test_paths.default_paths(run_tag).stage0c_partial_rank_json(rank).read_text(encoding="utf-8"),
    )
    return {
        "rank_accepts": payload["rank_accepts"],
        "rank_rejects": payload["rank_rejects"],
    }


def _divergent_candidate_pool_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
) -> str:
    candidates = _build_candidates(4 if rank == 0 else 2)
    # The replacement deliberately has a narrower test-only signature.
    runner_module: Any = runner_mod
    runner_module.load_and_validate_candidate_concepts = (
        lambda path, *, n_sae_features: candidates
    )
    try:
        runner_mod._load_candidate_pool_with_collective_identity_agreement(
            Path(tmp_path_str) / "candidate_concepts.json",
            n_sae_features=16384,
            world_size=world_size,
        )
    except AxBenchBaseError as exc:
        return str(exc)
    return "unexpected success"


def test_candidate_pool_divergence_fails_collectively_under_real_gloo(
    tmp_path,
    gloo_multiprocess,
):
    results = gloo_multiprocess(
        2,
        _divergent_candidate_pool_target,
        str(tmp_path),
    )

    assert all(
        "different ordered candidate pools" in result
        for result in results
    )
    assert results[0] == results[1]


def test_run_single_rank_world_size_2_accepts_disjoint(
    tmp_path, gloo_multiprocess,
):
    """Both ranks process disjoint stripes; their accepts cover the
    expected candidate-index modulo partition."""
    results = gloo_multiprocess(
        2, _multirank_target,
        str(tmp_path), "wsmpr2-disjoint", 4, 4, 2,
    )
    # Guard: this test asserts partition correctness, which is
    # block-size-invariant. Pin the effective B so a future cap change
    # cannot silently alter what this exercises (here: cap inert, B==2).
    cfg = _build_cfg(run_root=tmp_path, max_concepts=4, block_size=2)
    assert _block_size(cfg, world_size=2) == 2
    rank0_accepts = [a["candidate_index"] for a in results[0]["rank_accepts"]]
    rank1_accepts = [a["candidate_index"] for a in results[1]["rank_accepts"]]
    # Modulo partition: rank 0 owns even, rank 1 owns odd.
    assert all(i % 2 == 0 for i in rank0_accepts)
    assert all(i % 2 == 1 for i in rank1_accepts)
    # Combined coverage equals all 4 candidates.
    assert sorted(rank0_accepts + rank1_accepts) == [0, 1, 2, 3]


def test_run_single_rank_world_size_2_block_barrier_terminates(
    tmp_path, gloo_multiprocess,
):
    """When max_concepts=4 and all 8 candidates would accept, the loop
    terminates after enough blocks to satisfy the global cap rather
    than visiting every candidate."""
    results = gloo_multiprocess(
        2, _multirank_target,
        str(tmp_path), "wsmpr2-cap", 8, 4, 2,
    )
    cfg = _build_cfg(run_root=tmp_path, max_concepts=4, block_size=2)
    assert _block_size(cfg, world_size=2) == 2  # cap inert here
    rank0_accepts = [a["candidate_index"] for a in results[0]["rank_accepts"]]
    rank1_accepts = [a["candidate_index"] for a in results[1]["rank_accepts"]]
    total_accepts = len(rank0_accepts) + len(rank1_accepts)
    # The combined accepts must cover at least max_concepts=4 (so the
    # merge can still pick 4 winners) but must not visit the entire
    # 8-candidate pool — that's the termination win.
    assert total_accepts >= 4
    assert total_accepts < 8


@pytest.mark.parametrize(
    "n_candidates,max_concepts,configured_block,world_size",
    [
        (24, 2, 16, 4),
        (32, 4, 32, 4),
    ],
)
def test_run_single_rank_over_accept_budget_uses_effective_block_size(
    tmp_path, gloo_multiprocess,
    n_candidates, max_concepts, configured_block, world_size,
):
    """When the adaptive cap binds, the runner's over-accept budget is
    governed by the EFFECTIVE block size (_block_size(cfg, ws)), not the
    configured cfg field. Proves the cap actually constrains the
    multi-rank loop, and that a configured-B bound is vacuously loose.
    """
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=max_concepts,
        block_size=configured_block,
    )
    effective_b = _block_size(cfg, world_size=world_size)
    assert effective_b < configured_block, (
        f"scenario must exercise the cap: effective {effective_b} "
        f">= configured {configured_block}"
    )
    results = gloo_multiprocess(
        world_size, _multirank_target,
        str(tmp_path),
        f"effbudget-n{n_candidates}-mc{max_concepts}-ws{world_size}",
        n_candidates, max_concepts, configured_block,
    )
    total_accepts = sum(len(r["rank_accepts"]) for r in results)
    effective_budget = effective_b * world_size - 1
    configured_budget = configured_block * world_size - 1
    assert max_concepts <= total_accepts <= max_concepts + effective_budget, (
        f"total_accepts={total_accepts} outside effective-B bound "
        f"[{max_concepts}, {max_concepts + effective_budget}] "
        f"(effective_b={effective_b})"
    )
    assert effective_budget < configured_budget


# ---------------------------------------------------------------------------
# Multi-rank resume across-block test (single-process simulation)
# ---------------------------------------------------------------------------


def test_run_single_rank_resume_across_block_boundary(tmp_path, monkeypatch):
    """Crash mid-loop: visit only block 0, then restart and resume from
    block 1. The combined effect equals the no-crash full run."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=4, block_size=2)
    cands = _build_candidates(4)
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    # Wrap _verify_concept so that on the third call it raises — simulating
    # a mid-block crash partway through the second block.
    n_calls = {"n": 0}

    def _exploding_verify(*args, **kwargs):
        n_calls["n"] += 1
        if n_calls["n"] == 3:
            raise RuntimeError("crash!")
        return plan(*args, **kwargs)

    monkeypatch.setattr(si_mod, "_verify_concept", _exploding_verify)

    with pytest.raises(RuntimeError, match="crash"):
        run_single_rank(
            cfg, cands,
            rank=0, world_size=1,
            base_model=MagicMock(), base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[], feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            precomputed_clustered_fields={},
        )

    # First block (idx 0, 1) committed; we expect 2 accepts in the partial
    # checkpoint, plus visited_candidate_indices = [0, 1].
    payload_after_crash = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert len(payload_after_crash["rank_accepts"]) == 2
    assert payload_after_crash["next_block_index"] == 1
    assert sorted(payload_after_crash["visited_candidate_indices"]) == [0, 1]

    # Second invocation — resume. Restore the working stub.
    n_calls["n"] = 0
    monkeypatch.setattr(si_mod, "_verify_concept", plan)
    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    final_payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    # max_concepts=4, all four accept → final state has 4 accepts.
    assert len(final_payload["rank_accepts"]) == 4
    assert sorted(
        a["candidate_index"] for a in final_payload["rank_accepts"]
    ) == [0, 1, 2, 3]


# ---------------------------------------------------------------------------
# Dispatcher tests — _all_gather_state short-circuits at world_size=1
# ---------------------------------------------------------------------------


def test_all_gather_state_world_size_1_no_dist_call(monkeypatch):
    """At world_size=1, the runner short-circuits dist.all_gather_object
    so single-GPU paths don't require an initialized process group."""
    sentinel = (1, 2, 3, 4)

    # Make dist.all_gather_object explode if called — proving the
    # short-circuit really fires.
    def _explode(*args, **kwargs):
        raise RuntimeError("dist.all_gather_object should not be called at ws=1")

    monkeypatch.setattr(runner_mod.dist, "all_gather_object", _explode)
    out = runner_mod._all_gather_state(sentinel, world_size=1)
    assert out == [sentinel]


def test_candidate_pool_collective_rejects_divergent_rank_views(
    tmp_path,
    monkeypatch,
):
    candidates = _build_candidates(2)
    local_identity = candidate_pool_identity(candidates)
    monkeypatch.setattr(runner_mod.dist, "is_available", lambda: True)
    monkeypatch.setattr(runner_mod.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(
        runner_mod,
        "load_and_validate_candidate_concepts",
        lambda path, *, n_sae_features: candidates,
    )

    def _divergent_gather(out, local_status):
        out[0] = local_status
        out[1] = {
            "ok": True,
            "candidate_pool_identity": "f" * 64,
            "n_candidates": 1,
            "error_type": None,
            "error": None,
        }

    monkeypatch.setattr(runner_mod.dist, "all_gather_object", _divergent_gather)

    with pytest.raises(
        AxBenchBaseError,
        match="different ordered candidate pools.*divergent collectives",
    ):
        runner_mod._load_candidate_pool_with_collective_identity_agreement(
            tmp_path / "candidate_concepts.json",
            n_sae_features=16384,
            world_size=2,
        )

    assert local_identity != "f" * 64


def test_dispatcher_invokes_run_single_rank_and_run_merge_at_ws1(
    tmp_path, monkeypatch,
):
    """The dispatcher must call run_single_rank then run_merge at ws=1.

    Verifies the runner is the sole code path at every world_size and rank 0
    also performs the merge at world_size=1.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cands = _build_candidates(2)

    # Build a candidate_concepts.json that the dispatcher loads.
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text(
        json.dumps([
            {
                "feature_index": c.feature_index,
                "layer": c.layer,
                "sae_source": c.sae_source,
                "description": c.description,
                "corpus_count": c.corpus_count,
                "description_source": c.description_source,
            }
            for c in cands
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: cands)
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir",
                        lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(
        extraction, "build_percentile_table_from_stats",
        lambda p, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(
        si_mod, "_CandidatePrefixIndex",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        si_mod, "_base_context_window", lambda m: 2048,
    )
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    del plan

    n_run_single_calls = {"n": 0}
    n_run_merge_calls = {"n": 0}
    real_run_single = runner_mod.run_single_rank
    real_run_merge = runner_mod.run_merge

    def _spy_run_single(*args, **kwargs):
        n_run_single_calls["n"] += 1
        return real_run_single(*args, **kwargs)

    def _spy_run_merge(*args, **kwargs):
        n_run_merge_calls["n"] += 1
        return real_run_merge(*args, **kwargs)

    monkeypatch.setattr(runner_mod, "run_single_rank", _spy_run_single)
    monkeypatch.setattr(runner_mod, "run_merge", _spy_run_merge)

    out = runner_mod.dispatch_prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=0,
        world_size=1,
    )
    assert out == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    assert n_run_single_calls["n"] == 1
    assert n_run_merge_calls["n"] == 1


@pytest.mark.parametrize("api_model", [None, "openai/gpt-5-nano"])
def test_dispatcher_conditionally_constructs_and_routes_api_generator(
    tmp_path,
    monkeypatch,
    api_model,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1).replace(
        verification_api_model=api_model,
    )
    candidates = _build_candidates(1)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    response_cache = MagicMock()
    constructed: list[tuple[AxBenchBaseConfig, Any]] = []
    rank_kwargs: list[dict[str, Any]] = []
    sentinel_generator = object()

    class _FakeGenerator:
        def __new__(cls, cfg_arg, *, response_cache):
            del cls
            constructed.append((cfg_arg, response_cache))
            return sentinel_generator

    monkeypatch.setattr(runner_mod, "Stage0cApiGenerator", _FakeGenerator)
    monkeypatch.setattr(
        runner_mod,
        "load_and_validate_candidate_concepts",
        lambda path, n_sae_features: candidates,
    )
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir", lambda cfg: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda cfg: [])
    monkeypatch.setattr(
        extraction,
        "build_percentile_table_from_stats",
        lambda path, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(
        si_mod,
        "_CandidatePrefixIndex",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(si_mod, "_base_context_window", lambda model: 2048)

    def _capture_rank(*args, **kwargs):
        del args
        rank_kwargs.append(kwargs)

    monkeypatch.setattr(runner_mod, "run_single_rank", _capture_rank)
    monkeypatch.setattr(
        runner_mod,
        "run_merge",
        lambda *args, **kwargs: paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes(),
    )

    runner_mod.dispatch_prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=response_cache,
        rank=0,
        world_size=1,
    )

    assert len(rank_kwargs) == 1
    if api_model is None:
        assert constructed == []
        assert "verification_api_generator" not in rank_kwargs[0]
    else:
        assert constructed == [(cfg, response_cache)]
        assert rank_kwargs[0]["verification_api_generator"] is sentinel_generator


def test_dispatcher_api_mode_rejects_missing_response_cache(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1).replace(
        verification_api_model="openai/gpt-5-nano",
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        runner_mod,
        "load_and_validate_candidate_concepts",
        lambda path, n_sae_features: _build_candidates(1),
    )
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir", lambda cfg: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda cfg: [])
    monkeypatch.setattr(
        extraction,
        "build_percentile_table_from_stats",
        lambda path, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(
        si_mod,
        "_CandidatePrefixIndex",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(si_mod, "_base_context_window", lambda model: 2048)

    with pytest.raises(AxBenchBaseError, match="open synth response cache"):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg,
            concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0,
            world_size=1,
        )


def test_dispatcher_world_size_1_no_dist_barrier(tmp_path, monkeypatch):
    """At world_size=1 the dispatcher must NOT call dist.barrier()."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cands = _build_candidates(2)

    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: cands)
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir",
                        lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(
        extraction, "build_percentile_table_from_stats",
        lambda p, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(
        si_mod, "_CandidatePrefixIndex",
        lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(
        si_mod, "_base_context_window", lambda m: 2048,
    )
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    def _explode(*a, **kw):
        raise RuntimeError("dist.barrier should not fire at world_size=1")

    monkeypatch.setattr(runner_mod.dist, "barrier", _explode)

    out = runner_mod.dispatch_prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=0,
        world_size=1,
    )
    assert out == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()


def test_dispatcher_emits_setup_logs(tmp_path, monkeypatch):
    """The dispatcher emits the legacy [stage0c] setup log lines so
    test_init_setup_emits_progress_logs (and operators reading logs)
    don't lose visibility into the pre-loop pipeline state."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: _build_candidates(1))
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir",
                        lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(
        extraction, "build_percentile_table_from_stats",
        lambda p, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(si_mod, "_CandidatePrefixIndex",
                        lambda *a, **kw: MagicMock())
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={100: 2},
    )

    out = runner_mod.dispatch_prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=0,
        world_size=1,
    )
    assert out == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()


# ---------------------------------------------------------------------------
# Tests for _try_resume_per_rank mid-write crash recovery
# (PR-4 review hardening, Critical Fix #4)
# ---------------------------------------------------------------------------


def _seed_per_rank_partial_for_resume(
    cfg: AxBenchBaseConfig,
    *,
    rank: int,
    n_accepts_in_json: int,
    n_extra_rows_only_in_parquet: int = 0,
    n_rows_per_accept: int = 2,
    rows_in_parquet_minus: int = 0,
) -> tuple[Path, Path, str]:
    """Seed a per-rank checkpoint with a controlled accept/parquet-row mismatch.

    - The JSON records exactly ``n_accepts_in_json`` accepts (each with
      ``n_rows_per_accept`` verified_rows_count, summing to the
      "expected row count").
    - The parquet, however, holds those rows PLUS
      ``n_extra_rows_only_in_parquet`` trailing rows belonging to a
      hypothetical (K+1)-th accept whose JSON entry never landed.
    - When ``rows_in_parquet_minus > 0``, the parquet is instead missing
      that many rows from the legitimate set (data-loss path).

    Returns (json_path, parquet_path, candidate_pool_identity). The pool
    identity is the one baked into the written checkpoint's fingerprint;
    resume callers must pass it back to ``_try_resume_per_rank`` so the
    pooled-fingerprint validation matches.
    """
    n_total = max(
        n_accepts_in_json + max(1, n_extra_rows_only_in_parquet // n_rows_per_accept),
        n_accepts_in_json + 1,
    )
    cands = _build_candidates(n_total)
    accepts = []
    rows = []
    for placeholder, idx in enumerate(range(n_accepts_in_json)):
        c = cands[idx]
        accepts.append(_accept_dict_for_concept(
            c, candidate_index=idx, qc_score=2,
            n_verified_rows=n_rows_per_accept,
        ))
        sample_rows = [
            _VerifiedNegativePrefix(
                concept_id=placeholder,
                prefix_text=f"r{idx}_{i}",
                prefix_source_shard=0,
                prefix_source_doc=i,
                prefix_activating_seg_idx=i,
                verification_continuations=["c"],
                verification_seeds=[1000 + i],
                verification_percentiles=[80.0],
            )
            for i in range(n_rows_per_accept)
        ]
        rows.extend(_row_dicts_for_accept(
            sample_rows, candidate_index=idx, placeholder_concept_id=placeholder,
        ))
    # Add trailing rows that would belong to the (K+1)-th accept whose
    # JSON entry never made it to disk (mid-write crash signature).
    if n_extra_rows_only_in_parquet > 0:
        ghost_idx = n_accepts_in_json
        ghost_rows = [
            _VerifiedNegativePrefix(
                concept_id=n_accepts_in_json,
                prefix_text=f"ghost_{ghost_idx}_{i}",
                prefix_source_shard=0,
                prefix_source_doc=i,
                prefix_activating_seg_idx=i,
                verification_continuations=["c"],
                verification_seeds=[2000 + i],
                verification_percentiles=[90.0],
            )
            for i in range(n_extra_rows_only_in_parquet)
        ]
        rows.extend(_row_dicts_for_accept(
            ghost_rows, candidate_index=ghost_idx,
            placeholder_concept_id=n_accepts_in_json,
        ))
    # Truncate the parquet (data-loss path).
    if rows_in_parquet_minus > 0:
        rows = rows[:-rows_in_parquet_minus]
    pool_id = candidate_pool_identity(cands)
    # See _seed_runner_state_for_merge's identical comment: reproduces
    # whatever a real threaded `precomputed_clustered_fields` would have
    # been for this cfg (empty for unclustered, the stubbed clustered
    # dict when `_stub_cluster_fingerprint_fields` is active).
    precomputed_clustered_fields = si_mod.clustered_stage0c_fingerprint_fields(cfg)
    payload = _build_payload(
        cfg, rank=rank, world_size=1, n_candidates=len(cands),
        next_block_index=0,
        visited_candidate_indices=list(range(n_accepts_in_json)),
        rank_accepts=accepts, rank_rejects=[],
        rank_qc_per_feature=[],
        candidate_pool_identity=pool_id,
        precomputed_clustered_fields=precomputed_clustered_fields,
    )
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(rank)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(rank)
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        payload,
        rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    return json_path, parquet_path, pool_id


def test_resume_mid_write_crash_no_duplicate_rows(tmp_path, monkeypatch):
    """Mid-write crash: parquet has rows for the (K+1)-th accept but
    the JSON only counts K. _try_resume_per_rank slices off the trailing
    rows so resume re-derives them from candidate K — NO duplicates."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path, n_verified_target=2, max_concepts=4,
    )
    n_accepts = 2
    rows_per = 2
    extra_rows = rows_per  # trailing (K+1)-th accept's rows
    _, _, pool_id = _seed_per_rank_partial_for_resume(
        cfg, rank=0,
        n_accepts_in_json=n_accepts,
        n_extra_rows_only_in_parquet=extra_rows,
        n_rows_per_accept=rows_per,
    )

    payload = _try_resume_per_rank(
        cfg, rank=0, candidate_pool_identity=pool_id,
        n_sae_features=16384, precomputed_clustered_fields={},
    )
    assert payload is not None, "fresh-start path should not fire here"
    restored_rows = payload["_resumed_rows"]
    # Expected count: n_accepts * rows_per. The trailing extra_rows are
    # sliced off so resume re-derives them deterministically.
    expected = n_accepts * rows_per
    assert len(restored_rows) == expected, (
        f"expected {expected} rows after slice, got {len(restored_rows)} "
        f"(would be {expected + extra_rows} with the bug)"
    )
    # The restored rows belong to candidate 0 and 1 only — no ghost
    # candidate index sneaks through.
    candidate_indices = {r["candidate_index"] for r in restored_rows}
    assert candidate_indices == {0, 1}


def test_resume_mid_write_crash_data_loss_raises(tmp_path, monkeypatch):
    """Negative path: when the parquet has FEWER rows than the JSON
    accept tally implies, the data is genuinely lost. Surface a clear
    error rather than silently truncating."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path, n_verified_target=2, max_concepts=4,
    )
    _, _, pool_id = _seed_per_rank_partial_for_resume(
        cfg, rank=0,
        n_accepts_in_json=2,
        n_extra_rows_only_in_parquet=0,
        n_rows_per_accept=2,
        rows_in_parquet_minus=1,  # parquet is short by 1 row
    )

    with pytest.raises(AxBenchBaseError, match="data loss"):
        _try_resume_per_rank(
            cfg, rank=0, candidate_pool_identity=pool_id,
            n_sae_features=16384, precomputed_clustered_fields={},
        )


def test_resume_clean_no_op_when_counts_match(tmp_path, monkeypatch):
    """Sanity: a clean per-rank checkpoint where the parquet row count
    matches the JSON accept tally exactly resumes without any slicing
    or error."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path, n_verified_target=2, max_concepts=4,
    )
    _, _, pool_id = _seed_per_rank_partial_for_resume(
        cfg, rank=0,
        n_accepts_in_json=3,
        n_extra_rows_only_in_parquet=0,
        n_rows_per_accept=2,
    )
    payload = _try_resume_per_rank(
        cfg, rank=0, candidate_pool_identity=pool_id,
        n_sae_features=16384, precomputed_clustered_fields={},
    )
    assert payload is not None
    restored_rows = payload["_resumed_rows"]
    assert len(restored_rows) == 3 * 2
    assert {r["candidate_index"] for r in restored_rows} == {0, 1, 2}


def test_resume_rejects_changed_candidate_pool(tmp_path, monkeypatch):
    """A per-rank checkpoint written under candidate pool P1 must refuse
    to resume once the pool is regenerated (P2, same pinned run_tag): the
    pooled fingerprint's ``candidate_pool_identity`` field mismatches and
    ``_try_resume_per_rank`` raises ``AxBenchBaseError``.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    pool_p1 = _build_candidates(3)
    pool_p2 = _build_candidates(4)  # different pool -> different identity
    id1 = candidate_pool_identity(pool_p1)
    id2 = candidate_pool_identity(pool_p2)
    assert id1 != id2
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=len(pool_p1),
        next_block_index=0,
        visited_candidate_indices=[],
        rank_accepts=[],
        rank_rejects=[],
        rank_qc_per_feature=[],
        candidate_pool_identity=id1,
        precomputed_clustered_fields={},
    )
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        payload,
        [],
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    with pytest.raises(AxBenchBaseError, match="candidate_pool_identity"):
        _try_resume_per_rank(
            cfg,
            rank=0,
            candidate_pool_identity=id2,
            n_sae_features=16384,
            precomputed_clustered_fields={},
        )


def test_resume_accepts_same_candidate_pool(tmp_path, monkeypatch):
    """The same-pool resume succeeds: the stored pooled fingerprint's
    ``candidate_pool_identity`` matches the live identity so
    ``_try_resume_per_rank`` returns the payload without raising."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    pool = _build_candidates(3)
    pool_id = candidate_pool_identity(pool)
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=len(pool),
        next_block_index=0,
        visited_candidate_indices=[],
        rank_accepts=[],
        rank_rejects=[],
        rank_qc_per_feature=[],
        candidate_pool_identity=pool_id,
        precomputed_clustered_fields={},
    )
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        payload,
        [],
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    resumed = _try_resume_per_rank(
        cfg,
        rank=0,
        candidate_pool_identity=pool_id,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    assert resumed is not None
    assert resumed["fingerprint"]["candidate_pool_identity"] == pool_id


def test_reconcile_scale_up_filters_every_correlated_state_collection(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=4,
        enable_qc=True,
        n_verified_target=1,
    )
    candidates = _build_candidates(4)
    accepts = [
        _accept_dict_for_concept(
            candidates[idx],
            candidate_index=idx,
            qc_score=2,
            n_verified_rows=1,
        )
        for idx in (0, 1)
    ]
    rejects = [
        {
            "candidate_index": idx,
            "feature_index": candidates[idx].feature_index,
            "description": candidates[idx].description,
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": 2,
        }
        for idx in (2, 3)
    ]
    rows = [
        _row_dicts_for_accept(
            [
                _VerifiedNegativePrefix(
                    concept_id=idx,
                    prefix_text=f"prefix-{idx}",
                    prefix_source_shard=0,
                    prefix_source_doc=idx,
                    prefix_activating_seg_idx=idx,
                    verification_continuations=["continuation"],
                    verification_seeds=[idx],
                    verification_percentiles=[80.0],
                )
            ],
            candidate_index=idx,
            placeholder_concept_id=idx,
        )[0]
        for idx in (0, 1)
    ]
    qc_scores = [
        QCFeatureScore(
            feature_index=candidate.feature_index,
            description=candidate.description,
            outcome=QcOutcome.ACCEPTED,
            score=2,
            n_examples=1,
            error_message=None,
        )
        for candidate in candidates
    ]
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=len(candidates),
        next_block_index=7,
        visited_candidate_indices=[0, 1, 2, 3],
        rank_accepts=accepts,
        rank_rejects=rejects,
        rank_qc_per_feature=qc_scores,
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    payload["_resumed_rows"] = rows

    reconciled, changed = _reconcile_resumed_state_for_rank(
        cfg,
        payload,
        candidates,
        rank=0,
        world_size=2,
    )

    assert changed is True
    assert reconciled["world_size"] == 2
    assert reconciled["_prior_world_size"] == 1
    assert reconciled["next_block_index"] == 0
    assert reconciled["visited_candidate_indices"] == [0, 2]
    assert [r["candidate_index"] for r in reconciled["rank_accepts"]] == [0]
    assert [r["candidate_index"] for r in reconciled["rank_rejects"]] == [2]
    assert [r["candidate_index"] for r in reconciled["_resumed_rows"]] == [0]
    assert [
        r["feature_index"] for r in reconciled["rank_qc_per_feature"]
    ] == [
        candidates[0].feature_index,
        candidates[2].feature_index,
    ]


def test_reconcile_non_divisible_topology_change_filters_moved_indices(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=8)
    candidates = _build_candidates(8)
    rejects = [
        {
            "candidate_index": idx,
            "feature_index": candidates[idx].feature_index,
            "description": candidates[idx].description,
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": None,
        }
        for idx in (1, 4, 7)
    ]
    payload = _build_payload(
        cfg,
        rank=1,
        world_size=3,
        n_candidates=len(candidates),
        next_block_index=3,
        visited_candidate_indices=[1, 4, 7],
        rank_accepts=[],
        rank_rejects=rejects,
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    payload["_resumed_rows"] = []

    reconciled, changed = _reconcile_resumed_state_for_rank(
        cfg,
        payload,
        candidates,
        rank=1,
        world_size=2,
    )

    assert changed is True
    assert reconciled["visited_candidate_indices"] == [1, 7]
    assert [r["candidate_index"] for r in reconciled["rank_rejects"]] == [1, 7]


def test_reconcile_same_world_size_returns_state_unchanged(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    candidates = _build_candidates(2)
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=2,
        n_candidates=len(candidates),
        next_block_index=4,
        visited_candidate_indices=[0],
        rank_accepts=[],
        rank_rejects=[{
            "candidate_index": 0,
            "feature_index": candidates[0].feature_index,
            "description": candidates[0].description,
            "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
            "n_verified_found": 0,
            "llm_judge_score": None,
        }],
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    payload["_resumed_rows"] = []

    reconciled, changed = _reconcile_resumed_state_for_rank(
        cfg,
        payload,
        candidates,
        rank=0,
        world_size=2,
    )

    assert changed is False
    assert reconciled is payload
    assert reconciled["next_block_index"] == 4


def test_reconcile_same_topology_rewinds_to_earliest_unvisited_hole(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=8, block_size=2)
    candidates = _build_candidates(8)
    visited = [0, 4, 6]
    rejects = [{
        "candidate_index": idx,
        "feature_index": candidates[idx].feature_index,
        "description": candidates[idx].description,
        "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
        "n_verified_found": 0,
        "llm_judge_score": None,
    } for idx in visited]
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=2,
        n_candidates=len(candidates),
        next_block_index=2,
        visited_candidate_indices=visited,
        rank_accepts=[],
        rank_rejects=rejects,
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    payload["_resumed_rows"] = []

    reconciled, changed = _reconcile_resumed_state_for_rank(
        cfg,
        payload,
        candidates,
        rank=0,
        world_size=2,
    )

    assert changed is True
    assert reconciled["next_block_index"] == 0
    assert reconciled["visited_candidate_indices"] == visited


def test_qc_resume_start_uses_lower_newly_owned_hole() -> None:
    assert _qc_resume_start_at(
        n_candidates=8,
        visited_candidate_indices=[1, 7],
        rank=1,
        world_size=2,
    ) == 3
    assert _qc_resume_start_at(
        n_candidates=8,
        visited_candidate_indices=[1, 3, 5, 7],
        rank=1,
        world_size=2,
    ) == 8


# ---------------------------------------------------------------------------
# PR-4 second-round review: Tier 2 high-value tests
# ---------------------------------------------------------------------------


def test_run_single_rank_resume_world_size_change_resets_next_block_index(
    tmp_path, monkeypatch,
):
    """Bug surface: when world_size changes between phases (e.g., ws=4
    partial → ws=2 resume), the saved ``next_block_index`` is invalid
    because the candidate range covered by each block depends on
    block_size * world_size. The runner must reset
    ``next_block_index=0`` on world_size change AND retain
    ``visited_candidate_indices`` so already-completed candidates are
    skipped via the ``initial_visited_set``.

    Crash-free assertion: the restored runner re-iterates from block 0
    (so it covers all owned candidates under the new partition) but
    never re-processes any candidate already in
    ``visited_candidate_indices``.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=10, block_size=2)
    cands = _build_candidates(8)
    # Hand-craft a per-rank checkpoint that claims this rank ran under
    # ws_initial=4: visited a few candidates, advanced next_block_index.
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    payload_initial = _build_payload(
        cfg,
        rank=0,
        world_size=4,
        n_candidates=8,
        next_block_index=2,  # under ws=4 with B=2, this means 2 blocks done
        visited_candidate_indices=[0, 4],  # rank-0 owned under ws=4
        rank_accepts=[],
        rank_rejects=[
            {
                "candidate_index": idx,
                "feature_index": cands[idx].feature_index,
                "description": cands[idx].description,
                "reason": (
                    RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value
                ),
                "n_verified_found": 0,
                "llm_judge_score": None,
            }
            for idx in (0, 4)
        ],
        rank_qc_per_feature=[],
        # Same candidate pool as the ws=2 resume below, so the pooled
        # fingerprint matches and resume validates on pool identity.
        candidate_pool_identity=candidate_pool_identity(cands),
        precomputed_clustered_fields={},
    )
    write_per_rank_checkpoint(
        json_path, parquet_path, payload_initial, [],
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    # Now resume at world_size=2. Track _verify_concept calls to confirm
    # already-visited indices are skipped, but block 0 IS re-iterated.
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    visited_calls: list[int] = []
    real_plan = plan

    def _spy_verify(*args, **kwargs):
        # Recover the candidate position from the concept's feature_index
        # (deterministic builder: 100 + i).
        concept = args[1]
        candidate_pos = int(concept.feature_index) - 100
        visited_calls.append(candidate_pos)
        return real_plan(*args, **kwargs)

    monkeypatch.setattr(si_mod, "_verify_concept", _spy_verify)

    # Stub the all_gather rendezvous so this single-process test can
    # exercise the world_size > 1 code path without an initialised
    # process group. The stub returns the rank's own state for every
    # peer slot — equivalent to a degenerate ws=1 collective. The loop
    # terminates on pool-exhaustion, which is the relevant signal.
    def _fake_gather(state, world_size):
        return [state] * world_size

    monkeypatch.setattr(runner_mod, "_all_gather_state", _fake_gather)

    run_single_rank(
        cfg, cands,
        rank=0, world_size=2,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    # _verify_concept must NOT have been called for indices 0 or 4 —
    # those were preserved from the prior checkpoint.
    assert 0 not in visited_calls, (
        "index 0 was already visited under ws=4; should not re-verify"
    )
    assert 4 not in visited_calls, (
        "index 4 was already visited under ws=4; should not re-verify"
    )
    # Final visited set must include both prior + newly visited owned
    # under ws=2 (rank 0 owns even indices: 0, 2, 4, 6).
    final_payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert sorted(final_payload["visited_candidate_indices"]) == [0, 2, 4, 6]


def test_topology_reconcile_invalidates_old_json_before_smaller_parquet(
    tmp_path,
    monkeypatch,
):
    """A crash between the first reconciled parquet/JSON writes is resumable."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=3,
        n_verified_target=1,
    )
    candidates = _build_candidates(2)
    accepts = [
        _accept_dict_for_concept(
            candidates[idx],
            candidate_index=idx,
            qc_score=None,
            n_verified_rows=1,
        )
        for idx in (0, 1)
    ]
    rows = [
        _row_dicts_for_accept(
            [
                _VerifiedNegativePrefix(
                    concept_id=idx,
                    prefix_text=f"prefix-{idx}",
                    prefix_source_shard=0,
                    prefix_source_doc=idx,
                    prefix_activating_seg_idx=idx,
                    verification_continuations=["continuation"],
                    verification_seeds=[1000 + idx],
                    verification_percentiles=[80.0],
                )
            ],
            candidate_index=idx,
            placeholder_concept_id=idx,
        )[0]
        for idx in (0, 1)
    ]
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=len(candidates),
        next_block_index=1,
        visited_candidate_indices=[0, 1],
        rank_accepts=accepts,
        rank_rejects=[],
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        payload,
        rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )

    monkeypatch.setattr(
        runner_mod,
        "_all_gather_state",
        lambda state, *, world_size: [state, (0, 1, -1, 1)],
    )

    def _crash_before_json_commit(data, path):
        del data, path
        raise OSError("simulated reconciled JSON commit crash")

    monkeypatch.setattr(
        partition_mod,
        "atomic_write_bytes",
        _crash_before_json_commit,
    )
    with pytest.raises(OSError, match="reconciled JSON commit crash"):
        run_single_rank(
            cfg,
            candidates,
            rank=0,
            world_size=2,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[],
            feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            precomputed_clustered_fields={},
        )

    assert not json_path.exists()
    assert pq.read_table(parquet_path).num_rows == 1
    assert _try_resume_per_rank(
        cfg,
        rank=0,
        candidate_pool_identity=candidate_pool_identity(candidates),
        n_sae_features=16384, precomputed_clustered_fields={},
    ) is None


def test_reconcile_topology_change_rejects_qc_survivor_cap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=4,
        enable_qc=True,
        target_surviving=2,
    )
    candidates = _build_candidates(4)
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=len(candidates),
        next_block_index=0,
        visited_candidate_indices=[],
        rank_accepts=[],
        rank_rejects=[],
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    payload["_resumed_rows"] = []

    with pytest.raises(
        AxBenchBaseError,
        match="world_size change.*concept_examples_qc_target_surviving",
    ):
        _reconcile_resumed_state_for_rank(
            cfg,
            payload,
            candidates,
            rank=0,
            world_size=2,
        )


def test_reconcile_topology_change_ignores_dormant_qc_survivor_cap(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=4,
        enable_qc=False,
        target_surviving=2,
    )
    candidates = _build_candidates(4)
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=len(candidates),
        next_block_index=0,
        visited_candidate_indices=[],
        rank_accepts=[],
        rank_rejects=[],
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    payload["_resumed_rows"] = []

    reconciled, changed = _reconcile_resumed_state_for_rank(
        cfg,
        payload,
        candidates,
        rank=0,
        world_size=2,
    )

    assert changed is True
    assert reconciled["world_size"] == 2
    assert reconciled["_prior_world_size"] == 1


def test_run_single_rank_preserves_resumed_qc_cap_stop_flag(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=2,
        enable_qc=True,
        target_surviving=1,
        n_verified_target=1,
    )
    candidates = _build_candidates(1)
    accept = _accept_dict_for_concept(
        candidates[0],
        candidate_index=0,
        qc_score=2,
        n_verified_rows=1,
    )
    rows = _row_dicts_for_accept(
        [_VerifiedNegativePrefix(
            concept_id=0,
            prefix_text="prefix-0",
            prefix_source_shard=0,
            prefix_source_doc=0,
            prefix_activating_seg_idx=0,
            verification_continuations=["continuation"],
            verification_seeds=[1000],
            verification_percentiles=[80.0],
        )],
        candidate_index=0,
        placeholder_concept_id=0,
    )
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=1,
        next_block_index=1,
        visited_candidate_indices=[0],
        rank_accepts=[accept],
        rank_rejects=[],
        rank_qc_per_feature=[QCFeatureScore(
            feature_index=candidates[0].feature_index,
            description=candidates[0].description,
            outcome=QcOutcome.ACCEPTED,
            score=2,
            n_examples=1,
            error_message=None,
        )],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
        stopped_by_qc_survivor_cap=True,
    )
    write_per_rank_checkpoint(
        test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0),
        test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0),
        payload,
        rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )

    run_single_rank(
        cfg,
        candidates,
        rank=0,
        world_size=1,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[],
        feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )

    final_payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert final_payload["stopped_by_qc_survivor_cap"] is True


def test_ahead_rank_keeps_gathering_until_lagging_rank_passes_pool(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=10,
        block_size=1,
    )
    candidates = _build_candidates(4)
    rejects = [{
        "candidate_index": idx,
        "feature_index": candidates[idx].feature_index,
        "description": candidates[idx].description,
        "reason": RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES.value,
        "n_verified_found": 0,
        "llm_judge_score": None,
    } for idx in (0, 2)]
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=2,
        n_candidates=4,
        next_block_index=3,
        visited_candidate_indices=[0, 2],
        rank_accepts=[],
        rank_rejects=rejects,
        rank_qc_per_feature=[],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    write_per_rank_checkpoint(
        test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0),
        test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0),
        payload,
        [],
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    gather_calls = {"n": 0}

    def _skewed_gather(state, *, world_size):
        assert world_size == 2
        gather_calls["n"] += 1
        if gather_calls["n"] == 1:
            return [state, (0, 1, -1, 1)]
        return [state, (0, 2, -1, 2)]

    monkeypatch.setattr(runner_mod, "_all_gather_state", _skewed_gather)
    monkeypatch.setattr(
        si_mod,
        "_verify_concept",
        lambda *args, **kwargs: pytest.fail("ahead rank must have no work"),
    )

    run_single_rank(
        cfg,
        candidates,
        rank=0,
        world_size=2,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[],
        feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )

    assert gather_calls["n"] == 2


def test_run_merge_propagates_stopped_by_qc_survivor_cap_from_rank_payload(
    tmp_path, monkeypatch,
):
    """When a per-rank payload sets ``stopped_by_qc_survivor_cap=True``
    AND fewer than max_concepts accepts land overall, ``run_merge``
    raises with the QC-cap-specific error pointing at
    ``concept_examples_qc_target_surviving``."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=4,
        enable_qc=True,
        target_surviving=1,
    )
    # Hand-write a per-rank file with the cap flag set + 1 accept only.
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    cands = _build_candidates(4)
    _write_live_candidate_pool(cfg, cands)
    accepts = [
        _accept_dict_for_concept(
            cands[0],
            candidate_index=0,
            qc_score=2,
            n_verified_rows=2,
        )
    ]
    rows = _row_dicts_for_accept(
        [
            _VerifiedNegativePrefix(
                concept_id=0,
                prefix_text="r0",
                prefix_source_shard=0,
                prefix_source_doc=0,
                prefix_activating_seg_idx=0,
                verification_continuations=["c"],
                verification_seeds=[1000],
                verification_percentiles=[80.0],
            ),
            _VerifiedNegativePrefix(
                concept_id=0,
                prefix_text="r1",
                prefix_source_shard=0,
                prefix_source_doc=1,
                prefix_activating_seg_idx=1,
                verification_continuations=["c"],
                verification_seeds=[1001],
                verification_percentiles=[80.0],
            ),
        ],
        candidate_index=0,
        placeholder_concept_id=0,
    )
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=4,
        next_block_index=0,
        visited_candidate_indices=[0],
        rank_accepts=accepts,
        rank_rejects=[],
        rank_qc_per_feature=[
            QCFeatureScore(
                feature_index=cands[0].feature_index,
                description=cands[0].description,
                outcome=QcOutcome.ACCEPTED,
                score=2,
                n_examples=1,
                error_message=None,
            )
        ],
        candidate_pool_identity=candidate_pool_identity(cands),
        precomputed_clustered_fields={},
        stopped_by_qc_survivor_cap=True,
    )
    write_per_rank_checkpoint(
        json_path,
        parquet_path,
        payload,
        rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )

    with pytest.raises(AxBenchBaseError, match="qc_target_surviving"):
        run_merge(cfg, world_size=1)


def test_run_merge_writes_qc_report_when_qc_enabled(tmp_path, monkeypatch):
    """When ``cfg.enable_concept_examples_qc=True`` and the per-rank
    payload contains ``rank_qc_per_feature`` entries, run_merge writes
    ``concept_examples_qc_report.json`` containing those scores."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=2, enable_qc=True,
    )
    cands = _build_candidates(2)
    _write_live_candidate_pool(cfg, cands)
    # Seed two accepts WITH QC per_feature entries so the report has
    # the expected per-feature scores.
    json_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    parquet_path = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    accepts = [
        _accept_dict_for_concept(
            cands[0],
            candidate_index=0,
            qc_score=2,
            n_verified_rows=2,
        ),
        _accept_dict_for_concept(
            cands[1],
            candidate_index=1,
            qc_score=2,
            n_verified_rows=2,
        ),
    ]
    rows = []
    for placeholder, c in enumerate(cands):
        sample = [
            _VerifiedNegativePrefix(
                concept_id=placeholder,
                prefix_text=f"r{c.feature_index}_{i}",
                prefix_source_shard=0, prefix_source_doc=i,
                prefix_activating_seg_idx=i,
                verification_continuations=["c"],
                verification_seeds=[1000 + i],
                verification_percentiles=[80.0],
            )
            for i in range(2)
        ]
        rows.extend(_row_dicts_for_accept(
            sample, candidate_index=placeholder,
            placeholder_concept_id=placeholder,
        ))
    qc_per_feature = [
        QCFeatureScore(
            feature_index=c.feature_index, description=c.description,
            outcome=QcOutcome.ACCEPTED, score=2, n_examples=1, error_message=None,
        ) for c in cands
    ]
    payload = _build_payload(
        cfg, rank=0, world_size=1, n_candidates=2,
        next_block_index=0,
        visited_candidate_indices=[0, 1],
        rank_accepts=accepts, rank_rejects=[],
        rank_qc_per_feature=qc_per_feature,
        candidate_pool_identity=candidate_pool_identity(cands),
        precomputed_clustered_fields={},
    )
    write_per_rank_checkpoint(
        json_path, parquet_path, payload, rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    run_merge(cfg, world_size=1, n_candidates=2)

    qc_report_path = paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report()
    assert qc_report_path.exists()
    report = json.loads(qc_report_path.read_text(encoding="utf-8"))
    fis = [pf["feature_index"] for pf in report["per_feature"]]
    assert sorted(fis) == sorted(c.feature_index for c in cands)


def test_dispatcher_non_rank_0_skips_merge_and_returns_canonical_path(
    tmp_path, monkeypatch,
):
    """Bug surface: at ws=2, rank 1 must NOT call ``run_merge`` and
    must return the canonical parquet path. A regression that drops
    the rank gate would silently corrupt the canonical artifact via a
    racing rank-1 write."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cands = _build_candidates(2)

    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: cands)
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir",
                        lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(
        extraction, "build_percentile_table_from_stats",
        lambda p, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(si_mod, "_CandidatePrefixIndex",
                        lambda *a, **kw: MagicMock())
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    # Stub run_single_rank so the rank-1 path doesn't actually need
    # peer ranks for its all_gather calls.
    monkeypatch.setattr(runner_mod, "run_single_rank",
                        lambda *a, **kw: None)
    # Stub the merge barrier (no peers to rendezvous with).
    monkeypatch.setattr(
        runner_mod.dist, "barrier",
        lambda *a, **kw: None,
    )
    monkeypatch.setattr(
        runner_mod.dist,
        "broadcast_object_list",
        lambda payload, src=0: payload.__setitem__(
            0,
            {
                "ok": True,
                "path": str(paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()),
            },
        ),
    )
    # Spy on run_merge — rank 1 must not invoke it.
    n_merge_calls = {"n": 0}

    def _spy_merge(*a, **kw):
        n_merge_calls["n"] += 1
        return paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()

    monkeypatch.setattr(runner_mod, "run_merge", _spy_merge)

    out = runner_mod.dispatch_prepare_negative_prefixes(
        cfg,
        concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=1,
        world_size=2,
    )
    assert out == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    assert n_merge_calls["n"] == 0, (
        f"rank 1 must not call run_merge; got {n_merge_calls['n']} calls"
    )


def _patch_dispatch_through_merge(
    monkeypatch: pytest.MonkeyPatch,
    *,
    candidates: list[CandidateConcept],
) -> None:
    monkeypatch.setattr(
        runner_mod,
        "load_and_validate_candidate_concepts",
        lambda path, n_sae_features: candidates,
    )
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir", lambda cfg: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda cfg: [])
    monkeypatch.setattr(
        extraction,
        "build_percentile_table_from_stats",
        lambda path, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(
        si_mod,
        "_CandidatePrefixIndex",
        lambda *args, **kwargs: MagicMock(),
    )
    monkeypatch.setattr(si_mod, "_base_context_window", lambda model: 2048)
    monkeypatch.setattr(runner_mod, "run_single_rank", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner_mod.dist, "barrier", lambda: None)


def test_dispatcher_rank1_propagates_rank0_merge_error(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    candidates = _build_candidates(2)
    _patch_dispatch_through_merge(monkeypatch, candidates=candidates)
    monkeypatch.setattr(
        runner_mod.dist,
        "broadcast_object_list",
        lambda payload, src=0: payload.__setitem__(
            0,
            {
                "ok": False,
                "error_type": "ValueError",
                "error": "forced merge failure",
            },
        ),
    )
    monkeypatch.setattr(
        runner_mod,
        "run_merge",
        lambda *args, **kwargs: pytest.fail("rank 1 must not merge"),
    )

    with pytest.raises(
        AxBenchBaseError,
        match="rank-0 merge failed.*forced merge failure",
    ):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg,
            tmp_path / "candidate_concepts.json",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16,
            response_cache=None,
            rank=1,
            world_size=2,
        )


def test_dispatcher_rank0_broadcasts_then_reraises_merge_error(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    candidates = _build_candidates(2)
    _patch_dispatch_through_merge(monkeypatch, candidates=candidates)
    error = ValueError("forced rank0 merge failure")
    monkeypatch.setattr(
        runner_mod,
        "run_merge",
        lambda *args, **kwargs: (_ for _ in ()).throw(error),
    )
    broadcasts: list[dict[str, Any] | None] = []

    def _capture_broadcast(payload, src=0):
        assert src == 0
        broadcasts.append(payload[0])

    monkeypatch.setattr(
        runner_mod.dist,
        "broadcast_object_list",
        _capture_broadcast,
    )

    with pytest.raises(ValueError, match="forced rank0 merge failure") as exc_info:
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg,
            tmp_path / "candidate_concepts.json",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16,
            response_cache=None,
            rank=0,
            world_size=2,
        )

    assert exc_info.value is error
    assert broadcasts == [
        {
            "ok": False,
            "error_type": "ValueError",
            "error": "forced rank0 merge failure",
        }
    ]


def test_dispatcher_raises_when_run_tag_unset(tmp_path, monkeypatch):
    """Defensive: dispatcher must validate ``cfg.run_tag is not None``
    before any expensive work."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cfg = cfg.replace(run_tag=None)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")

    with pytest.raises(AxBenchBaseError, match="run_tag"):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
        )


def test_dispatcher_requires_cluster_contract_together_with_clustered_config(
    tmp_path, monkeypatch,
):
    """Fail-loud pairing (PR #739 C4): a clustered config demands a
    validated ``ClusterFeatureContract`` be threaded through explicitly.
    Fires as the second check in the dispatcher (right after the run_tag
    guard), before any candidate-pool or model work, so dummy args suffice.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    with pytest.raises(
        AxBenchBaseError,
        match="requires config and validated feature-contract state",
    ):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, Path("/unused/concepts.json"),
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
        )


def test_dispatcher_rejects_cluster_contract_for_unclustered_config(
    tmp_path, monkeypatch,
):
    """The converse pairing failure: an unclustered config must not silently
    accept a caller-supplied contract."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    assert cfg.conditioning is None
    with pytest.raises(
        AxBenchBaseError,
        match="requires config and validated feature-contract state",
    ):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, Path("/unused/concepts.json"),
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
            cluster_contract=cast(Any, object()),
        )


def test_dispatcher_clustered_routes_through_collective_agreement_wiring(
    tmp_path, monkeypatch,
):
    """Wiring proof (PR #739 FU-D-stage0c fix batch): the dispatcher's
    clustered branch must call ``cluster_artifacts.validate_cluster_
    candidate_identity_with_collective_agreement`` -- not the bare, pre-fix
    ``load_cluster_candidate_identity`` -- with exactly the contract/
    candidate_path/n_sae_features/world_size this call site has in
    scope, and must propagate whatever it raises without swallowing it.
    Fires as the very next check after the run_tag/pairing guards, before
    any candidate-pool or model work, so dummy args suffice.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    contract = cast(Any, SimpleNamespace(feature_contract_fingerprint="a" * 64))
    concepts_path = Path("/unused/concepts.json")
    seen: dict[str, Any] = {}

    def _fake_validate(cfg_arg, contract_arg, *, candidate_path, n_sae_features, world_size):
        seen["cfg"] = cfg_arg
        seen["contract"] = contract_arg
        seen["candidate_path"] = candidate_path
        seen["n_sae_features"] = n_sae_features
        seen["world_size"] = world_size
        raise AxBenchBaseError("sentinel from the collective-agreement wrapper")

    monkeypatch.setattr(
        runner_mod, "validate_cluster_candidate_identity_with_collective_agreement",
        _fake_validate,
    )
    # stage0c_runner.py no longer imports the bare `load_cluster_candidate_
    # identity` name at all (only the collective-agreement wrapper above) --
    # confirmed by grep; there is no leftover call site left to patch away.
    assert not hasattr(runner_mod, "load_cluster_candidate_identity")

    with pytest.raises(AxBenchBaseError, match="sentinel from the collective-agreement wrapper"):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=1, world_size=3,
            cluster_contract=contract,
        )

    assert seen == {
        "cfg": cfg,
        "contract": contract,
        "candidate_path": concepts_path,
        "n_sae_features": 16384,
        "world_size": 3,
    }


def test_dispatcher_raises_when_n_ctx_too_small(tmp_path, monkeypatch):
    """Bug surface: when the base model's context window is not strictly
    greater than ``cfg.verification_length``, Stage 0c cannot generate
    even one prefix token. The dispatcher must surface this with a
    clear error rather than running into a silent truncation later."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    # Force base_context_window == cfg.verification_length (boundary
    # case: must be strictly greater).
    monkeypatch.setattr(
        si_mod, "_base_context_window", lambda m: cfg.verification_length,
    )
    with pytest.raises(AxBenchBaseError, match="n_ctx"):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
        )


def test_dispatcher_raises_on_empty_candidate_list(tmp_path, monkeypatch):
    """Defensive: an empty candidate_concepts.json signals upstream Stage
    0a failure or stale artifacts. The dispatcher must raise rather
    than treating "0 candidates" as "0 accepts → Stage 0c done"."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(
        runner_mod, "load_and_validate_candidate_concepts",
        lambda p, n_sae_features: [],
    )
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)
    with pytest.raises(AxBenchBaseError, match="empty"):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
        )


def test_run_single_rank_empty_pool_writes_empty_payload(
    tmp_path, monkeypatch,
):
    """``run_single_rank`` with ``[]`` candidates writes a per-rank
    JSON with empty accept/reject/visited lists but a valid fingerprint
    so the merge step can still proceed (e.g., observe the empty state
    for diagnostics)."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    _patch_runner_dependencies(monkeypatch, rows_by_feature_index={})

    run_single_rank(
        cfg, [],
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert payload["rank_accepts"] == []
    assert payload["rank_rejects"] == []
    assert payload["visited_candidate_indices"] == []
    # Fingerprint validity: matches the pooled computation for the
    # current cfg and the (empty) candidate pool.
    assert payload["fingerprint"] == _fingerprint_with_pool(
        cfg, candidate_pool_identity=candidate_pool_identity([]),
    )


def test_run_single_rank_block_past_pool_end_breaks_safely(
    tmp_path, monkeypatch,
):
    """When ``block_size`` is larger than the candidate pool, the
    runner's ``block_start >= n_candidates`` guard must fire to
    prevent infinite iteration."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=10, block_size=64,
    )
    cands = _build_candidates(3)  # tiny pool, huge block
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    # The loop terminates cleanly; all 3 candidates accepted (max
    # is 10 so we don't truncate).
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert sorted(
        a["candidate_index"] for a in payload["rank_accepts"]
    ) == [0, 1, 2]


@pytest.mark.parametrize(
    "n_candidates,max_concepts,configured_block,world_size",
    [
        (8, 4, 2, 2),
        (24, 4, 16, 4),
    ],
)
def test_run_single_rank_block_barrier_pins_exact_count(
    tmp_path, gloo_multiprocess,
    n_candidates, max_concepts, configured_block, world_size,
):
    """Pin the EXACT total accept count + over-accept budget, computed
    from the EFFECTIVE block size ``_block_size(cfg, ws)`` so the bound
    stays tight even when the adaptive cap binds. (A configured-B bound
    goes vacuous under capping — the G2 regression class.)

    Over-accept budget contract: at most ``effective_B*W - 1`` extra
    accepts beyond ``max_concepts`` can leak through the all_gather
    rendezvous (each rank decides independently within a block; the loop
    fires at end-of-block so up to ``effective_B*W - 1`` accepts land
    before the next gather).
    """
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=max_concepts,
        block_size=configured_block,
    )
    effective_b = _block_size(cfg, world_size=world_size)
    results = gloo_multiprocess(
        world_size, _multirank_target,
        str(tmp_path),
        f"wsmpr-pin-n{n_candidates}-mc{max_concepts}-ws{world_size}",
        n_candidates, max_concepts, configured_block,
    )
    rank_accepts = [
        a["candidate_index"] for r in results for a in r["rank_accepts"]
    ]
    total_accepts = len(rank_accepts)
    over_accept_budget = effective_b * world_size - 1
    assert max_concepts <= total_accepts <= max_concepts + over_accept_budget, (
        f"total_accepts={total_accepts} outside expected bound "
        f"[{max_concepts}, {max_concepts + over_accept_budget}] "
        f"(effective_b={effective_b}, configured={configured_block})"
    )


def test_resume_clean_no_op_when_counts_match_no_re_verify(
    tmp_path, monkeypatch,
):
    """Sanity assertion strengthening: a complete first run + a second
    invocation must NOT call ``_verify_concept`` again. The counts-
    match resume short-circuit is observable by counting calls."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cands = _build_candidates(2)
    plan = _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    n_calls_before = {"n": 0}
    n_calls_after = {"n": 0}

    def _phase1_verify(*args, **kwargs):
        n_calls_before["n"] += 1
        return plan(*args, **kwargs)

    def _phase2_verify(*args, **kwargs):
        n_calls_after["n"] += 1
        return plan(*args, **kwargs)

    # Phase 1: full run.
    monkeypatch.setattr(si_mod, "_verify_concept", _phase1_verify)
    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    # Phase 2: re-run; resume short-circuit must skip the verification
    # for already-visited candidates.
    monkeypatch.setattr(si_mod, "_verify_concept", _phase2_verify)
    run_single_rank(
        cfg, cands,
        rank=0, world_size=1,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        qc_client=None,
        parquets=[], feature_shards=[],
        percentile_table=t.zeros(16384),
        candidate_index=MagicMock(),
        base_context_window=2048,
        n_sae_features=16384,
        precomputed_clustered_fields={},
    )
    # In phase 1 we accept exactly 1 (max_concepts=1); the loop calls
    # verify exactly once. Phase 2 must not invoke verify at all.
    assert n_calls_before["n"] >= 1
    assert n_calls_after["n"] == 0, (
        f"resume should skip _verify_concept; got {n_calls_after['n']} "
        "extra calls"
    )


def test_run_single_rank_qc_returns_none_at_world_size_2_raises(
    tmp_path, monkeypatch,
):
    """Bug C-3 regression: when QcBuffer.next returns None at
    world_size > 1, the runner MUST raise rather than silently popping
    visited_locally and continuing. ``target_surviving=None`` is wired
    in for ws>1, so a None score there indicates a regression in
    QcBuffer batching."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=4, block_size=2, enable_qc=True,
    )
    cands = _build_candidates(4)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )

    class _BadQcClient:
        """Returns nothing — every batch produces an empty per_feature
        list, simulating a regression where QcBuffer.next yields None."""
        def filter(self, batch, target_surviving=None):
            del batch, target_surviving
            return [], MagicMock(per_feature=[])

    # Stub _all_gather_state so the test runs on a single process —
    # we only need to provoke the None-score path within run_single_rank
    # before any rendezvous.
    monkeypatch.setattr(
        runner_mod, "_all_gather_state",
        lambda state, world_size: [state] * world_size,
    )

    with pytest.raises(AxBenchBaseError, match="multi-rank invariant"):
        run_single_rank(
            cfg, cands,
            rank=0, world_size=2,
            base_model=MagicMock(), base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=_BadQcClient(),
            parquets=[], feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            precomputed_clustered_fields={},
        )


def test_run_merge_atomic_writes_no_partial_parquet_on_crash(
    tmp_path, monkeypatch,
):
    """Bug I-1 regression: when ``atomic_write_parquet`` raises mid-write,
    NO partial ``negative_mode_prefixes.parquet`` file is left on disk.

    The atomic helper writes to a uuid-suffixed tmp path and only
    ``os.replace``s into the final name on success — so a
    pre-os.replace failure is fully reverted. Patches the helper at
    its import site (``stage0c_runner``) since the consolidation move
    routed callers through the canonical ``nanochat.atomic_io`` symbol.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    # Seed a valid per-rank state so merge gets past validation.
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=1,
        candidates_owned=[0, 1],
        accept_indices=[0, 1],
        reject_indices=[],
    )

    # Patch the atomic-parquet helper to raise mid-write — simulating
    # a disk-full / IO error during the canonical artifact's parquet
    # commit. Canonical signature is (table, final_path).
    def _exploding_atomic_parquet(table, path):
        del table
        # Simulate a disk-full error AFTER the parent dir was created
        # but BEFORE the tmp file gets atomically renamed.
        raise OSError(f"simulated mid-write failure for {path}")

    from steering.eval.axbench_base import stage0c_runner as _runner_mod
    monkeypatch.setattr(
        _runner_mod, "atomic_write_parquet", _exploding_atomic_parquet,
    )

    with pytest.raises(OSError, match="simulated mid-write failure"):
        run_merge(cfg, world_size=1)

    # Critical assertion: NO canonical parquet on disk.
    canonical = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    assert not canonical.exists(), (
        f"negative_mode_prefixes.parquet should not exist after a "
        f"mid-write crash; found at {canonical}"
    )


def test_run_merge_invalidates_old_commit_before_replacing_sidecars(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    canonical = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_bytes(b"old completed commit marker")
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()

    def _fail_first_sidecar(data, path):
        del data
        assert Path(path) == rejected_path
        assert not canonical.exists()
        raise OSError("simulated sidecar write failure")

    monkeypatch.setattr(runner_mod, "atomic_write_bytes", _fail_first_sidecar)
    with pytest.raises(OSError, match="sidecar write failure"):
        run_merge(cfg, world_size=1)

    assert not canonical.exists()
    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0).exists()
    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0).exists()


def _stub_cluster_fingerprint_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub the C1-owned Stage 0a identity chain with a canned, internally
    consistent fingerprint so runner-level tests can exercise the clustered
    branch without installing a full ``feature_contract.json`` +
    provenance-sidecar fixture. ``clustered_stage0c_fingerprint_fields``
    itself (real Stage 0a marker validation, byte-hashing, semantic
    cross-checks) is C1's own turf and already covered end to end by
    ``tests/axbench_base/unit/test_cluster_artifacts_functional.py`` against
    REAL ``ClusterFeatureContract`` objects -- re-deriving that fixture here
    would test C1's code, not this file's. Patched at ``_stage0c_internals``
    (the name-binding import site ``_stage0c_extractor_identity`` actually
    calls), so every fingerprint computation in a test -- per-rank payload
    AND the merge-time re-check -- observes the same canned value.
    """
    monkeypatch.setattr(
        si_mod,
        "clustered_stage0c_fingerprint_fields",
        lambda cfg: {
            "feature_contract_fingerprint": "a" * 64,
            "candidate_cluster_provenance_sha256": "b" * 64,
            "cluster_concept_projection_version": "v1",
        },
    )


def test_run_merge_clustered_refuses_to_replace_existing_canonical_artifact(
    tmp_path, monkeypatch,
):
    """Clustered Stage 0c artifacts are immutable commit markers (PR #739
    C4): unlike the unclustered path above (which invalidates an older
    parquet marker before a forced rerun), a clustered merge must refuse
    outright when ANY of the four canonical files already exists under this
    run_tag -- a torn or foreign pre-existing commit requires a fresh
    run_tag, never a same-run_tag overwrite.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_path.write_bytes(b"pre-existing canonical concepts")
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()

    with pytest.raises(
        AxBenchBaseError,
        match="refuses to replace an existing canonical artifact",
    ):
        run_merge(cfg, world_size=1)

    # The pre-existing canonical file is untouched, and no other canonical
    # artifact (e.g. rejected_concepts.json) was created by the refused run.
    assert concepts_path.read_bytes() == b"pre-existing canonical concepts"
    assert not rejected_path.exists()
    # Per-rank partials retained for investigation, exactly like the
    # unclustered refusal paths above.
    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0).exists()
    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0).exists()


def test_run_merge_clustered_installs_canonical_artifacts_when_fresh(
    tmp_path, monkeypatch,
):
    """Positive path: a clustered run_tag with no pre-existing canonical
    artifacts installs all three through the clustered install/write
    helpers (``install_clustered_bytes_once`` / ``write_clustered_parquet_
    once``) -- proving the clustered branch actually produces a complete,
    valid canonical artifact set, not merely a guard that always refuses.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=1,
        candidates_owned=[0], accept_indices=[0], reject_indices=[],
    )

    out_path = run_merge(cfg, world_size=1)

    assert out_path == paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    assert out_path.exists()
    assert paths.AxBenchBasePaths.from_config(cfg).concepts_json().exists()
    assert paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json().exists()
    # QC disabled by default (``_build_cfg``'s ``enable_qc=False``): no
    # report should be installed.
    assert not paths.AxBenchBasePaths.from_config(cfg).concept_examples_qc_report().exists()
    # Per-rank partials cleaned up after a successful canonical commit,
    # exactly like the unclustered success path.
    assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0).exists()


def test_run_merge_rejects_wrong_accept_target_before_canonical_mutation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=1,
        n_verified_target=2,
    )
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    rank_json = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    rank_parquet = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    payload = json.loads(rank_json.read_text(encoding="utf-8"))
    payload["rank_accepts"][0]["verified_rows_count"] = 1
    rank_json.write_text(json.dumps(payload), encoding="utf-8")
    table = pq.read_table(rank_parquet)
    pq.write_table(table.slice(0, 1), rank_parquet)

    canonical = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    old_marker = b"old completed commit marker"
    canonical.write_bytes(old_marker)

    with pytest.raises(
        AxBenchBaseError,
        match="verified_rows_count=.*must equal fingerprinted",
    ):
        run_merge(cfg, world_size=1)

    assert canonical.read_bytes() == old_marker


def test_run_merge_rejects_invalid_qc_semantics_before_canonical_mutation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        max_concepts=1,
        enable_qc=True,
        n_verified_target=2,
    )
    candidates = _build_candidates(1)
    _write_live_candidate_pool(cfg, candidates)
    accept = _accept_dict_for_concept(
        candidates[0],
        candidate_index=0,
        qc_score=0,
        n_verified_rows=2,
    )
    verified = [
        _VerifiedNegativePrefix(
            concept_id=0,
            prefix_text=f"prefix-{row_idx}",
            prefix_source_shard=0,
            prefix_source_doc=row_idx,
            prefix_activating_seg_idx=row_idx,
            verification_continuations=["continuation"],
            verification_seeds=[1000 + row_idx],
            verification_percentiles=[80.0],
        )
        for row_idx in range(2)
    ]
    rows = _row_dicts_for_accept(
        verified,
        candidate_index=0,
        placeholder_concept_id=0,
    )
    payload = _build_payload(
        cfg,
        rank=0,
        world_size=1,
        n_candidates=1,
        next_block_index=1,
        visited_candidate_indices=[0],
        rank_accepts=[accept],
        rank_rejects=[],
        rank_qc_per_feature=[
            QCFeatureScore(
                feature_index=candidates[0].feature_index,
                description=candidates[0].description,
                outcome=QcOutcome.ACCEPTED,
                score=0,
                n_examples=1,
                error_message=None,
            )
        ],
        candidate_pool_identity=candidate_pool_identity(candidates),
        precomputed_clustered_fields={},
    )
    write_per_rank_checkpoint(
        test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0),
        test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0),
        payload,
        rows,
        extraction_tokenizer_id=cfg.extraction_tokenizer_id,
    )
    canonical = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    old_marker = b"old completed commit marker"
    canonical.write_bytes(old_marker)

    with pytest.raises(AxBenchBaseError, match="inconsistent with outcome"):
        run_merge(cfg, world_size=1)

    assert canonical.read_bytes() == old_marker


def test_run_merge_uses_strict_fingerprint_types_before_canonical_mutation(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    rank_json = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    payload = json.loads(rank_json.read_text(encoding="utf-8"))
    payload["fingerprint"]["verification_length"] = True
    rank_json.write_text(json.dumps(payload), encoding="utf-8")
    canonical = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    canonical.parent.mkdir(parents=True, exist_ok=True)
    old_marker = b"old completed commit marker"
    canonical.write_bytes(old_marker)

    with pytest.raises(
        AxBenchBaseError,
        match="mismatched=.*verification_length",
    ):
        run_merge(cfg, world_size=1)

    assert canonical.read_bytes() == old_marker


def test_run_merge_post_write_validation_preserves_partials_on_bad_row(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0],
        accept_indices=[0],
        reject_indices=[],
    )
    rank_json = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0)
    rank_parquet = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0)
    table = pq.read_table(rank_parquet)
    rows = table.to_pylist()
    rows[0]["prefix_text"] = ""
    pq.write_table(pa.Table.from_pylist(rows, schema=table.schema), rank_parquet)

    with pytest.raises(AxBenchBaseError, match="prefix_text is empty"):
        run_merge(cfg, world_size=1)

    assert rank_json.exists()
    assert rank_parquet.exists()
    assert not paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes().exists()


def test_run_merge_validates_persisted_rejected_json_before_cleanup(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2)
    _seed_runner_state_for_merge(
        cfg,
        rank=0,
        world_size=1,
        candidates_owned=[0, 1, 2],
        accept_indices=[0, 2],
        reject_indices=[1],
    )
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()
    real_atomic_write_bytes = runner_mod.atomic_write_bytes

    def _corrupt_rejected_sidecar(data, path):
        if Path(path) == rejected_path:
            data = b'[{"unexpected": true}]'
        real_atomic_write_bytes(data, path)

    monkeypatch.setattr(
        runner_mod,
        "atomic_write_bytes",
        _corrupt_rejected_sidecar,
    )
    with pytest.raises(AxBenchBaseError, match="has keys"):
        run_merge(cfg, world_size=1)

    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(0).exists()
    assert test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(0).exists()


def test_cleanup_per_rank_partials_globs_orphans_from_larger_world_size(
    tmp_path, monkeypatch,
):
    """Bug I-2 regression: the cleanup must remove orphaned per-rank
    files left behind by a prior LARGER world_size.

    Scenario: ws_initial=4 left rank 0,1,2,3 files. Now we resume at
    ws_resume=2. The cleanup call passes world_size=2 but must still
    delete rank 2 and rank 3 files (which would otherwise mislead a
    future merge into reading stale data).
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    # Seed per-rank files for ranks 0, 1, 2, 3 (a prior ws=4 run).
    for r in range(4):
        json_p = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(r)
        parquet_p = test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(r)
        json_p.parent.mkdir(parents=True, exist_ok=True)
        json_p.write_text("{}", encoding="utf-8")
        parquet_p.write_bytes(b"placeholder")

    # Cleanup with the CURRENT (smaller) world_size.
    _cleanup_per_rank_partial_checkpoints(cfg)

    # ALL FOUR rank files (not just 0, 1) must be gone.
    for r in range(4):
        assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_json(r).exists(), (
            f"rank {r} JSON file was not cleaned up"
        )
        assert not test_paths.default_paths(_rt(cfg)).stage0c_partial_rank_parquet(r).exists(), (
            f"rank {r} parquet file was not cleaned up"
        )


# ---------------------------------------------------------------------------
# Regression test: dist.monitored_barrier is GLOO-only (NCCL incompatible)
# ---------------------------------------------------------------------------


def test_runner_source_does_not_call_monitored_barrier():
    """Production multi-GPU runs use NCCL backend (per nanochat.common.compute_init).

    PyTorch's ``dist.monitored_barrier`` is GLOO-only — it raises
    ``ValueError("monitored_barrier is only implemented for GLOO backend.")``
    when the active backend is NCCL. The runner must NOT call
    ``monitored_barrier``; only ``dist.barrier()`` (backend-agnostic) or
    ``dist.broadcast_object_list`` / ``dist.all_gather_object`` (also
    backend-agnostic) are safe at the merge rendezvous.

    Regression test for the bug where ``dispatch_prepare_negative_prefixes``
    used ``dist.monitored_barrier(timeout=...)`` at the merge rendezvous,
    causing every ``torchrun --nproc_per_node=N>1`` with ``device=cuda``
    to crash with ``ValueError("monitored_barrier is only implemented
    for GLOO backend.")``.

    The test uses static analysis on the runner source to catch any
    re-introduction of a ``dist.monitored_barrier(...)`` call. We grep
    for the call pattern (parens at the end) rather than the bare name,
    so that the docstring/comments can still mention ``monitored_barrier``
    when explaining why we deliberately avoid it.
    """
    from steering.eval.axbench_base import stage0c_runner
    src = Path(stage0c_runner.__file__).read_text(encoding="utf-8")
    # Look for an actual call to monitored_barrier — not just the name
    # (the file's docstring/comments DO reference it in the "why we
    # avoid this" explanation).
    assert "dist.monitored_barrier(" not in src, (
        "dist.monitored_barrier(...) is GLOO-only and crashes production "
        "NCCL runs with ValueError. Use dist.barrier() (backend-agnostic) "
        "instead. The PG default timeout (30 min, set in init_process_group) "
        "bounds rank-failure hangs."
    )
    # Defense-in-depth: also assert no plain `monitored_barrier(` call
    # under any other prefix (e.g. an aliased import).
    assert "monitored_barrier(" not in src, (
        "monitored_barrier(...) (any binding) appears in the runner. "
        "It is GLOO-only and crashes production NCCL runs. Use "
        "dist.barrier() instead."
    )


def test_dispatch_rejects_qc_survivor_cap_under_multi_rank(tmp_path, monkeypatch):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        enable_qc=True,
        target_surviving=1,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="concept_examples_qc_target_surviving.*world_size=1",
    ):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg,
            tmp_path / "candidate_concepts.json",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16,
            response_cache=None,
            rank=0,
            world_size=2,
        )


def test_dispatch_ignores_dormant_qc_survivor_cap_when_qc_disabled(
    tmp_path, monkeypatch,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(
        run_root=tmp_path,
        enable_qc=False,
        target_surviving=1,
    )

    class _ReachedSetup(RuntimeError):
        pass

    def _reached_setup(model):
        del model
        raise _ReachedSetup

    monkeypatch.setattr(si_mod, "_base_context_window", _reached_setup)
    with pytest.raises(_ReachedSetup):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg,
            tmp_path / "candidate_concepts.json",
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16,
            response_cache=None,
            rank=0,
            world_size=2,
        )


def test_dispatcher_does_not_call_gloo_only_monitored_barrier(
    tmp_path, monkeypatch,
):
    """Runtime regression test (companion to the static check).

    Wires a spy onto ``dist.monitored_barrier`` that records calls and
    raises the same ``ValueError`` PyTorch raises when the active
    backend is NCCL. Drives the dispatcher through the world_size>1
    rendezvous path (rank 1 of ws=2). If the runner ever re-introduces
    a ``monitored_barrier`` call, the spy fires and the assertion
    catches it — even if the spy itself is silenced (the call list
    is checked unconditionally).
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1, block_size=2)
    cands = _build_candidates(2)

    monitored_calls: list = []

    def _spy_monitored_barrier(*args, **kwargs):
        monitored_calls.append((args, kwargs))
        raise ValueError(
            "monitored_barrier is only implemented for GLOO backend."
        )

    monkeypatch.setattr(
        runner_mod.dist, "monitored_barrier", _spy_monitored_barrier,
    )

    # Wire up the dependency stubs identically to other dispatcher tests.
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: cands)
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir",
                        lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(
        extraction, "build_percentile_table_from_stats",
        lambda p, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(si_mod, "_CandidatePrefixIndex",
                        lambda *a, **kw: MagicMock())
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)
    _patch_runner_dependencies(
        monkeypatch,
        rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    # Stub run_single_rank so the rank-1 path doesn't actually need
    # peer ranks for its all_gather calls.
    monkeypatch.setattr(runner_mod, "run_single_rank",
                        lambda *a, **kw: None)
    # Stub the merge-rendezvous barrier so the test runs single-process.
    monkeypatch.setattr(runner_mod.dist, "barrier", lambda *a, **kw: None)
    monkeypatch.setattr(
        runner_mod.dist,
        "broadcast_object_list",
        lambda payload, src=0: payload.__setitem__(
            0,
            {
                "ok": True,
                "path": str(paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()),
            },
        ),
    )
    # Stub run_merge so rank 0 (if it were called) would be a no-op.
    monkeypatch.setattr(
        runner_mod,
        "run_merge",
        lambda *a, **kw: paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes(),
    )

    # Drive both rank=0 and rank=1 paths with world_size=2 — both
    # exercise the merge-rendezvous code path (rank 0 still hits
    # ``dist.barrier()``).
    runner_mod.dispatch_prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=1, world_size=2,
    )
    runner_mod.dispatch_prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=0, world_size=2,
    )

    assert monitored_calls == [], (
        f"Production code must not call dist.monitored_barrier "
        f"(GLOO-only, crashes NCCL launches). Found "
        f"{len(monitored_calls)} call(s): {monitored_calls!r}"
    )


# ---------------------------------------------------------------------------
# Clustered partial-resume live-statistics re-validation (PR #739
# FU-D-stage0c fix batch, Codex cross-review Bug 1 HIGH): ``_try_resume_
# per_rank``'s ``_fingerprint_with_pool`` records only ``cfg.feature_
# statistics_path`` -- the path STRING, never file content -- so a
# statistics file replaced at the same path went undetected. When
# clustering is enabled, ``_try_resume_per_rank`` must now re-validate the
# LIVE on-disk statistics identity against the contract's pinned identity
# BEFORE any partial rows are accepted.
# ---------------------------------------------------------------------------


def _stats_feature_dir(tmp_path: Path, *, name: str = "clustered-features") -> Path:
    feature_dir = build_feature_dir(
        tmp_path / name,
        segment_size=128,
        layer=8,
        activation_type=ActivationType.residual,
        pooling=PoolingMode.max_pool,
        width=SAEWidth.W_16K,
        model="gemma-2-2b",
        dataset_phase=DatasetPhase.BASE,
        ignore_bos_features=False,
    )
    feature_dir.mkdir(parents=True, exist_ok=True)
    return feature_dir


def _write_versioned_stats(
    feature_dir: Path,
    *,
    n_sae_features: int = 16_384,
    feature_ids: tuple[int, ...] = (0, 1, 2),
    max_mags: dict[int, float],
) -> Path:
    """Write a real, self-consistent statistics file under ``feature_dir``.
    Calling this twice at the same ``feature_dir`` overwrites the same
    final path in place (``atomic_safetensors_save`` is an unconditional
    ``os.replace``) -- exactly the plan's attack scenario: same path, same
    width/domain, different bytes.
    """
    statistics_path = write_steering_stats(
        feature_dir / "feature_statistics",
        n_sae_features=n_sae_features,
        max_mags=dict(max_mags),
        absent=[i for i in range(n_sae_features) if i not in feature_ids],
    )
    stats = load_statistics(statistics_path)
    checksum = compute_percentile_checksum(
        build_percentile_table(stats.per_feature_stats, n_sae_features)
    )
    save_percentile_checksum(statistics_path, checksum)
    return statistics_path


def _clustered_cfg_with_stats(
    tmp_path: Path, *, feature_dir: Path, statistics_path: Path, **kwargs: Any,
) -> AxBenchBaseConfig:
    cfg = _build_cfg(run_root=tmp_path, **kwargs)
    return cfg.replace(
        feature_dir=str(feature_dir),
        feature_statistics_path=str(statistics_path),
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )


def _fake_cluster_contract(statistics_dict: dict[str, Any]) -> Any:
    return cast(
        Any,
        SimpleNamespace(
            feature_contract_fingerprint="a" * 64,
            statistics=SimpleNamespace(canonical_dict=lambda: dict(statistics_dict)),
        ),
    )


def test_try_resume_per_rank_clustered_noop_when_statistics_unchanged(
    tmp_path, monkeypatch,
):
    """Normal-path proof: a genuine resume where the statistics file is
    untouched must succeed exactly as before -- no false raise."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir, statistics_path=statistics_path,
        run_tag="clustered-resume-unchanged", n_verified_target=2, max_concepts=4,
    )
    _, _, pool_id = _seed_per_rank_partial_for_resume(
        cfg, rank=0, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    contract = _fake_cluster_contract(pinned.canonical_dict())

    payload = _try_resume_per_rank(
        cfg, rank=0, candidate_pool_identity=pool_id,
        cluster_contract=contract, n_sae_features=16_384,
        precomputed_clustered_fields=si_mod.clustered_stage0c_fingerprint_fields(cfg),
    )

    assert payload is not None
    assert len(payload["_resumed_rows"]) == 2


def test_try_resume_per_rank_clustered_raises_on_swapped_statistics(
    tmp_path, monkeypatch,
):
    """RED/GREEN core proof (Bug 1 HIGH): commit a per-rank partial under
    statistics A, then swap the statistics file in place to B (same
    width/domain, different bytes). Resuming must raise loud -- NOT
    silently accept the old partial rows against a percentile table that
    would later be rebuilt from B.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir, statistics_path=statistics_path,
        run_tag="clustered-resume-swapped", n_verified_target=2, max_concepts=4,
    )
    _, _, pool_id = _seed_per_rank_partial_for_resume(
        cfg, rank=0, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    contract = _fake_cluster_contract(pinned_a.canonical_dict())

    # Swap in place: same path, same n_sae_features/feature_ids domain,
    # different max_mag bytes.
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        _try_resume_per_rank(
            cfg, rank=0, candidate_pool_identity=pool_id,
            cluster_contract=contract, n_sae_features=16_384,
            precomputed_clustered_fields=si_mod.clustered_stage0c_fingerprint_fields(cfg),
        )


def test_try_resume_per_rank_requires_n_sae_features_and_precomputed_clustered_fields(
    tmp_path, monkeypatch,
):
    """Regression (owner review of PR #884: "what's the point of making a
    parameter nullable if we're just going to raise when it's None?").
    ``n_sae_features`` and ``precomputed_clustered_fields`` are now
    unconditionally required keyword-only parameters on
    ``_try_resume_per_rank`` -- no more Optional-then-raise. This
    supersedes the old conditional-on-``cluster_contract``
    ``AxBenchBaseError("... requires n_sae_features ...")`` check this
    test used to pin (deleted by the cleanup: the sole production caller,
    ``run_single_rank``, always threads real values for both, clustered
    or not). Pins the successor contract instead: omitting either kwarg
    now fails loud via Python's own ``TypeError`` for a missing required
    keyword-only argument, unconditionally -- not only under a clustered
    ``cluster_contract`` -- so this can't silently regress back to
    Optional. Deliberately tiny (a plain unclustered cfg): the TypeError
    fires at the call boundary before ``cluster_contract`` is ever
    consulted, so no clustered statistics/contract fixture is needed.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path)
    resume = cast(Callable[..., Any], _try_resume_per_rank)

    with pytest.raises(TypeError, match="precomputed_clustered_fields"):
        resume(
            cfg, rank=0, candidate_pool_identity="pool-id",
            n_sae_features=16384,
        )
    with pytest.raises(TypeError, match="n_sae_features"):
        resume(
            cfg, rank=0, candidate_pool_identity="pool-id",
            precomputed_clustered_fields={},
        )


def test_run_single_rank_clustered_resume_raises_on_swapped_statistics(
    tmp_path, monkeypatch,
):
    """End-to-end wiring proof: ``run_single_rank`` (not just ``_try_
    resume_per_rank`` directly) threads ``cluster_contract``/``n_sae_
    features`` through to the resume check."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir, statistics_path=statistics_path,
        run_tag="run-single-rank-clustered-swapped", n_verified_target=2, max_concepts=4,
    )
    # n_accepts_in_json=1, n_rows_per_accept=2 -> _seed_per_rank_partial_
    # for_resume's own n_total = max(1 + max(1, 0 // 2), 1 + 1) = 2; the
    # live candidate pool handed to run_single_rank must match that exact
    # count so its pool-identity fingerprint field agrees with the seeded
    # checkpoint's.
    cands = _build_candidates(2)
    _, _, pool_id = _seed_per_rank_partial_for_resume(
        cfg, rank=0, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    assert candidate_pool_identity(cands) == pool_id
    contract = _fake_cluster_contract(pinned_a.canonical_dict())
    _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        run_single_rank(
            cfg, cands,
            rank=0, world_size=1,
            base_model=MagicMock(), base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            qc_client=None,
            parquets=[], feature_shards=[],
            percentile_table=t.zeros(16384),
            candidate_index=MagicMock(),
            base_context_window=2048,
            n_sae_features=16384,
            precomputed_clustered_fields=si_mod.clustered_stage0c_fingerprint_fields(cfg),
            cluster_contract=contract,
        )


# ---------------------------------------------------------------------------
# Real-gloo RESUME-path collective-agreement proof (LAST known deadlock
# analog in the FU-D-stage0c fix batch): ``_try_resume_per_rank``'s own
# live-statistics re-validation used to be a bare per-rank local raise,
# reached only when THIS rank's own partial checkpoint exists. Both ranks
# below have a REAL partial checkpoint on disk; rank 1's cfg points at a
# statistics file that is genuinely different from what the shared
# contract pinned -- the same "same width/domain, different content" swap
# scenario used throughout this fix batch, modeled as two distinct on-disk
# files (rather than an in-place overwrite) so each rank's process
# observes a fixed, deterministic view instead of depending on inter-
# process filesystem-write timing. Before this fix, rank 1 raised alone at
# ``stage0c_runner.py:418`` while rank 0 (nothing to catch the mismatch)
# proceeded into ``run_single_rank``'s later ``_all_gather_state``
# collective and hung forever; the test itself completing (rather than
# hanging until ``gloo_multiprocess``'s own ``_RESULT_TIMEOUT_S`` bound)
# is the no-hang proof.
# ---------------------------------------------------------------------------


def _resume_stats_collective_gloo_target(
    rank: int, world_size: int,
    tmp_path_str: str,
    run_tag: str,
    matching_feature_dir_str: str,
    matching_statistics_path_str: str,
    divergent_feature_dir_str: str,
    divergent_statistics_path_str: str,
    pinned_dict: dict[str, Any],
    pool_id: str,
) -> str:
    """Worker for the resume-path collective no-hang proofs below. Rank 0
    resumes against a statistics file that matches ``pinned_dict``; rank 1
    resumes against a genuinely different one written under its own
    ``feature_dir`` (same canonical component shape, different bytes).

    No local imports (matching this file's other gloo-target worker
    functions, e.g. ``_multirank_target``): ``os``/``Path``/``SimpleNamespace``/
    ``cast``/``Any``/``si_mod``/``runner_mod``/``AxBenchBaseError``/
    ``_clustered_cfg_with_stats`` are all already-imported or already-
    defined module globals by the time a freshly-spawned child process
    (``mp.get_context("spawn")``) re-executes this module's top level and
    locates this function by qualified name.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)
    feature_dir = Path(
        matching_feature_dir_str if rank == 0 else divergent_feature_dir_str
    )
    statistics_path = Path(
        matching_statistics_path_str if rank == 0 else divergent_statistics_path_str
    )
    # No monkeypatch fixture exists in a spawned child process, so use an
    # Any-typed module view to stub the C1-owned Stage 0a identity chain.
    # This ensures
    # ``_fingerprint_with_pool`` (called AFTER the stats-identity
    # collective, only reached when this rank has a local checkpoint) sees
    # the same canned fingerprint fields the parent process used to seed
    # the on-disk checkpoints -- mirrors ``_stub_cluster_fingerprint_
    # fields``'s monkeypatch version, which cannot cross a process boundary.
    si_module: Any = si_mod
    si_module.clustered_stage0c_fingerprint_fields = lambda cfg: {
        "feature_contract_fingerprint": "a" * 64,
        "candidate_cluster_provenance_sha256": "b" * 64,
        "cluster_concept_projection_version": "v1",
    }
    cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir, statistics_path=statistics_path,
        run_tag=run_tag, n_verified_target=2, max_concepts=4,
    )
    contract = cast(
        Any,
        SimpleNamespace(
            feature_contract_fingerprint="a" * 64,
            statistics=SimpleNamespace(canonical_dict=lambda: dict(pinned_dict)),
        ),
    )
    try:
        resumed = runner_mod._try_resume_per_rank(
            cfg, rank=rank, candidate_pool_identity=pool_id,
            cluster_contract=contract, n_sae_features=16_384,
            world_size=world_size,
            precomputed_clustered_fields=si_mod.clustered_stage0c_fingerprint_fields(cfg),
        )
    except AxBenchBaseError as exc:
        return f"raised:{exc}"
    if resumed is None:
        return "resumed-none"
    return f"ok:{len(cast(list[Any], resumed['_resumed_rows']))}"


def test_try_resume_per_rank_collective_raises_together_when_a_peer_sees_swapped_statistics_under_real_gloo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gloo_multiprocess: Any,
) -> None:
    """RED/GREEN core proof: a real 2-process gloo group where BOTH ranks
    have a genuine partial checkpoint on disk, but rank 1's own live
    statistics view disagrees with the pinned contract. Before the fix
    (bare per-rank local ``assert_live_statistics_identity`` call at
    ``stage0c_runner.py:418``) this scenario hangs: rank 1 raises alone
    and dies, rank 0 proceeds with no way to learn about it. GREEN means
    both ranks raise the SAME collective failure summary and the test
    completes well inside ``gloo_multiprocess``'s timeout bound.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "resume-collective-swap"
    feature_dir_a = _stats_feature_dir(tmp_path, name="resume-features-a")
    statistics_path_a = _write_versioned_stats(
        feature_dir_a, max_mags={0: 1.0, 1: 2.0, 2: 3.0},
    )
    pinned, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir_a, statistics_path=statistics_path_a, n_sae_features=16_384,
    )
    feature_dir_b = _stats_feature_dir(tmp_path, name="resume-features-b")
    statistics_path_b = _write_versioned_stats(
        feature_dir_b, max_mags={0: 9.0, 1: 8.0, 2: 7.0},
    )

    _stub_cluster_fingerprint_fields(monkeypatch)
    seed_cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir_a, statistics_path=statistics_path_a,
        run_tag=run_tag, n_verified_target=2, max_concepts=4,
    )
    _, _, pool_id_0 = _seed_per_rank_partial_for_resume(
        seed_cfg, rank=0, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    _, _, pool_id_1 = _seed_per_rank_partial_for_resume(
        seed_cfg, rank=1, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    assert pool_id_0 == pool_id_1

    results = gloo_multiprocess(
        2, _resume_stats_collective_gloo_target,
        str(tmp_path), run_tag,
        str(feature_dir_a), str(statistics_path_a),
        str(feature_dir_b), str(statistics_path_b),
        pinned.canonical_dict(), pool_id_0,
    )

    assert all(result.startswith("raised:") for result in results), results
    assert all(
        "failed collectively before rank work" in result for result in results
    ), results
    assert all(
        "no longer match the identity pinned" in result for result in results
    ), results
    assert results[0] == results[1]


def test_try_resume_per_rank_collective_all_ranks_agree_resumes_cleanly_under_real_gloo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gloo_multiprocess: Any,
) -> None:
    """Normal-path companion proof: a real 2-process gloo group where both
    ranks' statistics genuinely match the pinned contract. No false raise,
    AND each rank actually resumes its own real checkpoint content (not
    just "did not crash") -- the collective wrapper's no-op-on-agreement
    path must not corrupt or block the ordinary resume."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "resume-collective-agree"
    feature_dir_a = _stats_feature_dir(tmp_path, name="resume-features-agree")
    statistics_path_a = _write_versioned_stats(
        feature_dir_a, max_mags={0: 1.0, 1: 2.0, 2: 3.0},
    )
    pinned, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir_a, statistics_path=statistics_path_a, n_sae_features=16_384,
    )

    _stub_cluster_fingerprint_fields(monkeypatch)
    seed_cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir_a, statistics_path=statistics_path_a,
        run_tag=run_tag, n_verified_target=2, max_concepts=4,
    )
    _, _, pool_id_0 = _seed_per_rank_partial_for_resume(
        seed_cfg, rank=0, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    _, _, pool_id_1 = _seed_per_rank_partial_for_resume(
        seed_cfg, rank=1, n_accepts_in_json=1, n_rows_per_accept=2,
    )
    assert pool_id_0 == pool_id_1

    # Both ranks resolve to the SAME (matching) feature_dir/statistics_path:
    # pass feature_dir_a/statistics_path_a for both the "matching" and
    # "divergent" slots so rank 1 also observes the genuinely-unchanged file.
    results = gloo_multiprocess(
        2, _resume_stats_collective_gloo_target,
        str(tmp_path), run_tag,
        str(feature_dir_a), str(statistics_path_a),
        str(feature_dir_a), str(statistics_path_a),
        pinned.canonical_dict(), pool_id_0,
    )

    assert results == ["ok:2", "ok:2"], results


def test_try_resume_per_rank_collective_rank_with_no_checkpoint_still_raises_with_swapped_peer_under_real_gloo(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, gloo_multiprocess: Any,
) -> None:
    """The critical edge case that makes gating the collective call on
    ``has_local_checkpoint`` unsafe: rank 0 has NO local checkpoint at all
    (a fresh start for this rank -- pre-fix, its copy of the stats check
    never even ran), rank 1's checkpoint IS present and its statistics
    view is swapped. Both ranks must still raise together -- rank 0 must
    not be able to sail past this rendezvous just because it personally
    had nothing to validate, or it would proceed into ``run_single_rank``'s
    later collective while rank 1 is dead, reproducing the exact deadlock.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    run_tag = "resume-collective-nocheckpoint"
    feature_dir_a = _stats_feature_dir(tmp_path, name="resume-features-nc-a")
    statistics_path_a = _write_versioned_stats(
        feature_dir_a,
        max_mags={0: 1.0, 1: 2.0, 2: 3.0},
    )
    pinned, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir_a,
        statistics_path=statistics_path_a,
        n_sae_features=16_384,
    )
    feature_dir_b = _stats_feature_dir(tmp_path, name="resume-features-nc-b")
    statistics_path_b = _write_versioned_stats(
        feature_dir_b,
        max_mags={0: 9.0, 1: 8.0, 2: 7.0},
    )

    _stub_cluster_fingerprint_fields(monkeypatch)
    seed_cfg = _clustered_cfg_with_stats(
        tmp_path,
        feature_dir=feature_dir_a,
        statistics_path=statistics_path_a,
        run_tag=run_tag,
        n_verified_target=2,
        max_concepts=4,
    )
    # Only rank 1 gets a seeded checkpoint; rank 0's json_path never exists.
    _, _, pool_id_1 = _seed_per_rank_partial_for_resume(
        seed_cfg,
        rank=1,
        n_accepts_in_json=1,
        n_rows_per_accept=2,
    )
    assert not test_paths.default_paths(run_tag).stage0c_partial_rank_json(0).exists()

    results = gloo_multiprocess(
        2,
        _resume_stats_collective_gloo_target,
        str(tmp_path),
        run_tag,
        str(feature_dir_a),
        str(statistics_path_a),
        str(feature_dir_b),
        str(statistics_path_b),
        pinned.canonical_dict(),
        pool_id_1,
    )

    assert all(result.startswith("raised:") for result in results), results
    assert all("failed collectively before rank work" in result for result in results), results
    assert results[0] == results[1]


# ---------------------------------------------------------------------------
# FRESH-run percentile-table build binding (Codex cross-review follow-up to
# FU-D-stage0c Bug 1 HIGH). A fresh clustered run never reaches ``_try_
# resume_per_rank``'s per-rank identity guard, so the percentile table built
# in ``dispatch_prepare_negative_prefixes`` (and consumed by ``_verify_
# concept``/merge) must itself be bound to the pinned statistics identity --
# otherwise an A->B statistics swap after the collective candidate-pool check
# but before the table build slips through undetected.
# ---------------------------------------------------------------------------


class _ReachedCandidatePrefixIndex(Exception):
    """Sentinel raised by a monkeypatched ``_CandidatePrefixIndex`` so a
    dispatcher test can stop execution EXACTLY one step past the percentile-
    table build, proving whether the build itself raised or was cleanly
    passed."""


def _fake_cluster_contract_with_targets(statistics_dict: dict[str, Any]) -> Any:
    """Like :func:`_fake_cluster_contract`, but also exposes the ``target_
    for_representative`` method the dispatcher calls right after the
    percentile-table build, so a no-false-raise test can proceed from the
    build to the ``_CandidatePrefixIndex`` sentinel."""
    return cast(
        Any,
        SimpleNamespace(
            feature_contract_fingerprint="a" * 64,
            statistics=SimpleNamespace(canonical_dict=lambda: dict(statistics_dict)),
            target_for_representative=lambda feature_id: SimpleNamespace(
                allowed_member_ids=(feature_id,),
            ),
        ),
    )


def _patch_dispatcher_preloop_to_reach_table_build(
    monkeypatch: pytest.MonkeyPatch, *, candidates: list[CandidateConcept],
) -> None:
    """Monkeypatch the dispatcher's pre-loop pipeline so execution reaches
    the percentile-table build with the REAL clustered statistics on disk,
    then stops one step later at the ``_CandidatePrefixIndex`` sentinel.

    Deliberately does NOT patch ``build_percentile_table_from_stats``: the
    clustered path must exercise the real fused identity-bound builder (and,
    without the fix, the real cached fallback reading whatever bytes are on
    disk), so the swap actually reaches a real read.
    """
    monkeypatch.setattr(
        runner_mod, "load_and_validate_candidate_concepts",
        lambda p, n_sae_features: candidates,
    )
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir", lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)

    def _raise_reached(*a: Any, **kw: Any) -> Any:
        raise _ReachedCandidatePrefixIndex

    monkeypatch.setattr(si_mod, "_CandidatePrefixIndex", _raise_reached)


def test_dispatch_fresh_clustered_raises_when_statistics_swapped_before_table_build(
    tmp_path, monkeypatch,
):
    """RED/GREEN core proof for the FRESH-run table-build hole. A fresh
    clustered Stage 0c run (NO partial checkpoint -> ``_try_resume_per_rank``
    is never reached) passes the collective candidate-pool check under
    statistics A, then the file is swapped A->B in the window before the
    percentile-table build. The build MUST re-validate against the pinned
    identity and raise loud, not silently build the consumed table from B.
    Without the fix the swap slips through to the ``_CandidatePrefixIndex``
    sentinel instead of raising the identity-mismatch error.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir, statistics_path=statistics_path,
        run_tag="dispatch-fresh-clustered-swapped", max_concepts=1, block_size=2,
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    contract = _fake_cluster_contract_with_targets(pinned_a.canonical_dict())
    _patch_dispatcher_preloop_to_reach_table_build(
        monkeypatch, candidates=_build_candidates(1),
    )

    def _collective_passes_a_then_swaps_to_b(
        cfg_arg, contract_arg, *, candidate_path, n_sae_features, world_size,
    ):
        # The collective genuinely validates statistics A, exactly as it
        # would on a real fresh run ...
        live, _dir, _stats = _statistics_identity(
            feature_dir=feature_dir, statistics_path=statistics_path,
            n_sae_features=16_384,
        )
        assert live.canonical_dict() == pinned_a.canonical_dict()
        # ... then, in the TOCTOU window before the table build, the on-disk
        # file is swapped in place to B (same width/domain, different bytes).
        _write_versioned_stats(feature_dir, max_mags={0: 9.0, 1: 8.0, 2: 7.0})
        # Real production contract: this collective vote never returns
        # None (see validate_cluster_candidate_identity_with_collective_
        # agreement's own explicit raise on that structurally-unreachable
        # case) -- the dispatcher immediately feeds the return value to
        # the now-non-Optional _clustered_fingerprint_fields_from_identity.
        # Match that contract here rather than the stub's prior implicit
        # `None` (which only worked because that helper used to tolerate
        # None -- owner review of #884).
        return cluster_artifacts_mod.ClusterCandidateIdentity(
            feature_contract_fingerprint="a" * 64,
            candidate_concepts_sha256="c" * 64,
            candidate_provenance_sha256="b" * 64,
            concept_projection_version="v1",
        )

    monkeypatch.setattr(
        runner_mod, "validate_cluster_candidate_identity_with_collective_agreement",
        _collective_passes_a_then_swaps_to_b,
    )

    with pytest.raises(AxBenchBaseError, match="no longer match the identity pinned"):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
            cluster_contract=contract,
        )


def test_dispatch_fresh_clustered_builds_table_when_statistics_unchanged(
    tmp_path, monkeypatch,
):
    """Normal-path no-false-raise proof: a fresh clustered run whose
    statistics are untouched must pass the identity-bound build and proceed
    -- reaching the ``_CandidatePrefixIndex`` sentinel one step past the
    build. Proves the new binding does not spuriously reject the honest
    fresh-run path.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    _stub_cluster_fingerprint_fields(monkeypatch)
    feature_dir = _stats_feature_dir(tmp_path)
    statistics_path = _write_versioned_stats(feature_dir, max_mags={0: 1.0, 1: 2.0, 2: 3.0})
    pinned_a, _dir_identity, _stats = _statistics_identity(
        feature_dir=feature_dir, statistics_path=statistics_path, n_sae_features=16_384,
    )
    cfg = _clustered_cfg_with_stats(
        tmp_path, feature_dir=feature_dir, statistics_path=statistics_path,
        run_tag="dispatch-fresh-clustered-unchanged", max_concepts=1, block_size=2,
    )
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text("[]", encoding="utf-8")
    contract = _fake_cluster_contract_with_targets(pinned_a.canonical_dict())
    _patch_dispatcher_preloop_to_reach_table_build(
        monkeypatch, candidates=_build_candidates(1),
    )
    # Collective passes; NO swap -- statistics stay A through the build.
    # Real return value (never None -- see the sibling swap test's comment
    # for why the stub must match that production contract).
    monkeypatch.setattr(
        runner_mod, "validate_cluster_candidate_identity_with_collective_agreement",
        lambda *a, **kw: cluster_artifacts_mod.ClusterCandidateIdentity(
            feature_contract_fingerprint="a" * 64,
            candidate_concepts_sha256="c" * 64,
            candidate_provenance_sha256="b" * 64,
            concept_projection_version="v1",
        ),
    )

    with pytest.raises(_ReachedCandidatePrefixIndex):
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=0, world_size=1,
            cluster_contract=contract,
        )


# ---------------------------------------------------------------------------
# PR #739 FU-D-2 (owner-trimmed to its simplification core, 2026-07-26):
# resolve-once Stage-0c identity cache.
#
# ``validate_cluster_candidate_identity_with_collective_agreement`` now
# resolves the candidate identity ONCE at startup and returns it; the
# dispatcher derives ``precomputed_clustered_fields`` from that return value
# and threads it through ``run_single_rank`` -> ``_build_payload`` /
# ``_try_resume_per_rank`` instead of letting each independently re-read the
# candidate/provenance sidecar. The one remaining per-rank comparison raise
# (``validate_per_rank_fingerprint``, on the resume path) is left as an
# ACCEPTED-RESIDUAL bare per-rank raise by owner decision -- the DDP vote
# that would have collectivized it was descoped as not worth further vote
# machinery for a rare, abnormal-state-only, bounded failure mode; see the
# PR #739 campaign ledger.
# ---------------------------------------------------------------------------


def test_build_payload_unclustered_precomputed_empty_dict_matches_fresh_read_fingerprint(
    tmp_path, monkeypatch,
):
    """U-invariant, updated for the owner's required-parameter cleanup
    (review of PR #884): ``_build_payload``'s ``precomputed_clustered_
    fields`` is now a required kwarg -- there is no more None-default
    fresh-read fallback AT THIS LAYER, so this test can no longer compare
    two ``_build_payload`` calls (an omitted-kwarg baseline is not
    expressible anymore, and reusing ``{}`` on both sides would be a
    vacuous self-equality tautology with zero regression value).

    Anchors the comparison one layer down instead, at the lower
    ``_fingerprint_with_pool`` helper, which deliberately KEPT its
    ``precomputed_clustered_fields: ... | None = None`` Optional default
    (see its own docstring). For an unclustered cfg, threading the
    dispatcher-resolved ``{}`` through the now-required ``_build_payload``
    layer must still produce a fingerprint byte-identical to what the
    still-Optional layer computes via its OWN fresh-read default -- i.e.
    the required-vs-Optional boundary is transparent to the checkpoint
    fingerprint content.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = _build_cfg(run_root=tmp_path, max_concepts=2, block_size=2)
    assert cfg.conditioning is None

    cached_payload = _build_payload(
        cfg, rank=0, world_size=1, n_candidates=2, next_block_index=1,
        visited_candidate_indices=[0], rank_accepts=[], rank_rejects=[],
        rank_qc_per_feature=[], candidate_pool_identity="p" * 64,
        precomputed_clustered_fields={},
    )
    fresh_read_fingerprint = _fingerprint_with_pool(
        cfg, candidate_pool_identity="p" * 64,
    )

    assert cached_payload["fingerprint"] == fresh_read_fingerprint


def test_fingerprint_with_pool_precomputed_clustered_fields_used_verbatim_no_read(
    tmp_path, monkeypatch,
):
    """Threading proof at the lowest layer: when ``precomputed_clustered_
    fields`` is supplied, ``_fingerprint_with_pool`` never calls
    ``clustered_stage0c_fingerprint_fields`` (and, transitively,
    ``load_cluster_candidate_identity``) at all -- proven by making that
    fresh-read path explode if reached."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    def _explode(cfg_arg):
        del cfg_arg
        raise AssertionError(
            "clustered_stage0c_fingerprint_fields must not be called when "
            "precomputed_clustered_fields is supplied"
        )

    monkeypatch.setattr(si_mod, "clustered_stage0c_fingerprint_fields", _explode)
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )

    fp = _fingerprint_with_pool(
        cfg,
        candidate_pool_identity="p" * 64,
        precomputed_clustered_fields={
            "feature_contract_fingerprint": "a" * 64,
            "candidate_cluster_provenance_sha256": "b" * 64,
            "cluster_concept_projection_version": "v1",
        },
    )

    assert fp["feature_contract_fingerprint"] == "a" * 64
    assert fp["candidate_cluster_provenance_sha256"] == "b" * 64
    assert fp["cluster_concept_projection_version"] == "v1"


# ---- Read-count proof (dispatcher-level, multi-block) ----


def test_dispatch_clustered_reads_candidate_identity_exactly_once_at_startup_across_multiple_blocks(
    tmp_path, monkeypatch,
):
    """Read-count proof (PR #739 FU-D-2): a clustered run spanning multiple
    block boundaries must call ``load_cluster_candidate_identity`` EXACTLY
    ONCE per rank for the startup-through-``run_single_rank`` phase (the
    startup collective vote), not once per block. Before this fix, every
    ``_build_payload`` call at a block boundary independently re-read the
    candidate-identity sidecar via ``clustered_stage0c_fingerprint_
    fields`` -> ``load_cluster_candidate_identity``, so the count grew
    with the number of blocks -- empirically verified RED against the
    pre-FU-D-2 code (this exact test body run against origin/main @
    329c8309): ``assert 5 == 1`` (1 for the dispatcher's own vote + 4 for
    the 3 in-loop block commits plus the trailing insurance commit, this
    scenario's exact 3-candidate/``block_size=1`` shape). GREEN (this fix)
    is exactly 1.

    ``run_merge`` is stubbed out (returning a fixed path) so this count
    isolates the hot/per-rank phase from ``run_merge``'s OWN INTENTIONALLY
    live re-reads (the separate staleness-at-merge story, proven by
    ``test_run_merge_catches_sidecar_mutation_after_startup_via_live_
    reread`` below) -- counting those together would conflate two
    different, deliberately different caching behaviors.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    read_calls = {"n": 0}

    def _counting_load(
        cfg_arg: Any, *, candidate_path: Any = None, expected_fingerprint: Any = None,
    ) -> Any:
        del cfg_arg, candidate_path, expected_fingerprint
        read_calls["n"] += 1
        return cluster_artifacts_mod.ClusterCandidateIdentity(
            feature_contract_fingerprint="a" * 64,
            candidate_concepts_sha256="c" * 64,
            candidate_provenance_sha256="b" * 64,
            concept_projection_version="v1",
        )

    monkeypatch.setattr(
        cluster_artifacts_mod, "load_cluster_candidate_identity", _counting_load,
    )
    monkeypatch.setattr(
        cluster_artifacts_mod, "assert_live_statistics_identity",
        lambda *a, **k: None,
    )
    # Percentile-table build is orthogonal to candidate-identity read
    # counting (a separate, already-collectivized statistics concern) --
    # stub it out entirely so this test needs no real statistics files.
    monkeypatch.setattr(
        runner_mod, "build_verified_percentile_table_with_collective_agreement",
        lambda *a, **kw: t.zeros(16384),
    )
    # run_merge is stubbed to a no-op fixed-path return -- see the
    # docstring above for why its own (intentionally live) re-reads must
    # not be conflated with this proof.
    fixed_merge_path = tmp_path / "stub_merge_output.parquet"
    monkeypatch.setattr(runner_mod, "run_merge", lambda *a, **kw: fixed_merge_path)
    cfg = _build_cfg(
        run_root=tmp_path, max_concepts=3, block_size=1,
    ).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    cands = _build_candidates(3)
    concepts_path = tmp_path / "candidate_concepts.json"
    concepts_path.write_text(
        json.dumps([
            {
                "feature_index": c.feature_index, "layer": c.layer,
                "sae_source": c.sae_source, "description": c.description,
                "corpus_count": c.corpus_count,
                "description_source": c.description_source,
            }
            for c in cands
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: cands)
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir", lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda c: [])
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)
    monkeypatch.setattr(si_mod, "_CandidatePrefixIndex", lambda *a, **kw: MagicMock())
    plan = _patch_runner_dependencies(
        monkeypatch, rows_by_feature_index={c.feature_index: 2 for c in cands},
    )
    del plan
    contract = _fake_cluster_contract_with_targets({"anything": "ok"})

    out = runner_mod.dispatch_prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=MagicMock(), base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384, response_cache=None,
        rank=0, world_size=1, cluster_contract=contract,
    )

    assert out == fixed_merge_path
    payload = _read_rank_payload(tmp_path, cfg.run_tag, rank=0)
    assert len(payload["rank_accepts"]) == 3
    # 3 candidates at block_size=1 -> at least 3 distinct block-boundary
    # commits plus the trailing insurance commit -- every one of them
    # reuses the SAME cached identity: exactly 1 read total (the
    # dispatcher's own startup vote).
    assert read_calls["n"] == 1


# ---- Staleness-at-merge ----


def test_run_merge_catches_sidecar_mutation_after_startup_via_live_reread(
    tmp_path, monkeypatch,
):
    """Staleness-at-merge story (PR #739 FU-D-2): identity is pinned ONCE
    at startup and threaded to every per-block checkpoint commit -- a
    sidecar mutation AFTER startup is invisible to those commits (they
    never re-read; the read-count test above proves that half). ``run_
    merge``'s OWN live re-read (deliberately left unthreaded -- see
    ``_stage0c_internals._schema_with_completed_negative_prefixes_
    metadata``'s own FU-D-2 comment) still catches the swap at merge time.
    Every rank raising together via the existing broadcast machinery is
    already proven by ``test_dispatcher_rank0_broadcasts_then_reraises_
    merge_error`` / ``test_dispatcher_rank1_propagates_rank0_merge_error``
    -- this test isolates the DETECTION mechanism itself: identity pinned
    at startup, mid-run mutation invisible to block commits, caught at
    merge by a genuinely fresh read.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    current_fields = {
        "feature_contract_fingerprint": "a" * 64,
        "candidate_cluster_provenance_sha256": "b" * 64,
        "cluster_concept_projection_version": "v1",
    }
    monkeypatch.setattr(
        si_mod, "clustered_stage0c_fingerprint_fields",
        lambda cfg: dict(current_fields),
    )
    cfg = _build_cfg(run_root=tmp_path, max_concepts=1).replace(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        }
    )
    # Seed the per-rank checkpoint under the startup-pinned identity (a
    # fresh read, at seed time, exactly as run_single_rank's own cached
    # commits would have used the vote's derived fields).
    _seed_runner_state_for_merge(
        cfg, rank=0, world_size=1,
        candidates_owned=[0], accept_indices=[0], reject_indices=[],
    )

    # Mutate the sidecar's logical identity AFTER the per-rank checkpoint
    # was committed -- simulating a mid-run Stage 0a regeneration the
    # startup vote could not have observed (the in-memory pool IS the
    # startup pool for every block-boundary commit; only run_merge's own
    # deliberately-live re-read can see this).
    current_fields["candidate_cluster_provenance_sha256"] = "c" * 64

    with pytest.raises(
        AxBenchBaseError, match="merged per-rank fingerprint does not match",
    ):
        run_merge(cfg, world_size=1)
