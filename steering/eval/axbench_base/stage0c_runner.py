"""Per-rank Stage 0c runner (PR-4 Tasks 5+6).

Implements the multi-GPU work-distribution loop for Stage 0c. The
single-rank Stage 0c loop has been split into
two cooperating routines:

* :func:`run_single_rank` — one rank's stripe of the candidate pool. It
  visits the indices owned by ``(rank, world_size)`` modulo partition,
  runs LLM-QC + N-seed verification per candidate, and commits a
  per-rank checkpoint pair (``stage0c_partial_rank_{rank}.{json,parquet}``)
  before each block-barrier step. After every block of ``B*world_size``
  candidates the ranks rendezvous via ``dist.all_gather_object`` so the
  global accept tally can drive a deterministic termination condition.
* :func:`run_merge` — rank-0 final merge. Reads every per-rank
  checkpoint, sorts accepts by candidate_index, truncates to
  ``cfg.max_concepts``, assigns the canonical dense ``concept_id`` per
  the merge function in :mod:`stage0c_partition`, and writes the three
  Stage 0c artifacts (``concepts.json``, ``rejected_concepts.json``,
  ``negative_mode_prefixes.parquet``).

Single-/multi-rank equivalence (completed artifacts)
----------------------------------------------------
At ``world_size == 1`` the block-barrier loop reduces to iterating
candidates in ``range(n_candidates)``: every candidate index is owned by
rank 0, the all_gather rendezvous is short-circuited, and the canonical
``concept_id`` assignment in :func:`run_merge` is the index among accepted
candidates in candidate order. Merge applies the same candidate-index cutoff
and ordering to rejected records and QC records, so rank-stripe over-visits do
not enter any completed sidecar. With deterministic verification and identical
QC responses/cache state, single- and multi-rank runs therefore produce the
same JSON bytes and logical parquet schema/rows/metadata. Physical parquet
bytes are not an API because PyArrow may vary encoding. When examples QC is
enabled, the canonical parquet also pins the semantic identity of
``concept_examples_qc_report.json``; disabled runs pin a ``None`` sentinel and
require that no stale report exists.

Multi-GPU termination
---------------------
Each rank tracks its local ``n_accepted`` and ``n_visited`` counters and
posts them to the all_gather. The loop terminates as soon as
``sum(n_accepted) >= cfg.max_concepts`` AND every rank has visited at
least up to ``max(candidate_index for accept in global_accepts)``, or
``sum(n_visited) == n_candidates`` (full pool exhausted).

The global accept count may exceed ``max_concepts`` by up to
``world_size`` accepts because all ranks decide independently within a
block; the merge truncates back to ``max_concepts`` deterministically by
candidate-index sort. This is the over-accept budget in the design.

QC target_surviving and multi-rank
----------------------------------
Under multi-rank mode each rank's :class:`QcBuffer` is constructed with
``target_surviving=None`` so per-rank API batching cannot impose a different
cap on each stripe. An exact global live-QC survivor cap would require another
coordinated state machine, so ``concept_examples_qc_target_surviving`` is
rejected when ``world_size > 1`` instead of being silently ignored or
topology-dependently overrun. Production runs use its default ``None``.

At ``world_size == 1`` the runner passes
``cfg.concept_examples_qc_target_surviving`` through unchanged so the
single-GPU behavior is preserved.

Rank failure
------------
On any uncaught exception in one rank's loop, the surviving ranks block
on ``dist.all_gather_object`` (or the post-loop ``dist.barrier()``)
until the process-group collective timeout expires. PyTorch's
``init_process_group`` defaults (~10 min for NCCL / 30 min for GLOO)
are preserved globally — ``nanochat.common.compute_init`` only passes
an explicit ``timeout=`` when ``NCCL_TIMEOUT_SEC`` is set, so it is
NOT a global default change. The axbench-base Stage 0c launcher
(``scripts/eval/axbench_base_eval``) defaults ``NCCL_TIMEOUT_SEC`` to
1800s (30 min) for its long rank-0-only Stage-0a phase, so for Stage
0c runs the hang bound is 1800s, while other DDP jobs keep PyTorch's
default. The torchrun launcher kills the entire job on first rank
failure, so in practice the hang is bounded by the launcher's exit
hook, not by the PG timeout.

We deliberately use ``dist.barrier()`` (backend-agnostic) at the merge
rendezvous instead of ``dist.monitored_barrier`` because the latter is
GLOO-only and crashes the production NCCL launch with
``ValueError("monitored_barrier is only implemented for GLOO backend.")``.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from dataclasses import asdict, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import torch.distributed as dist
from loguru import logger

from nanochat.atomic_io import atomic_write_bytes, atomic_write_parquet
from steering.eval.axbench_base import _stage0c_internals as _si
from steering.eval.axbench_base import concept_list as _concept_list
from steering.eval.axbench_base import extraction as _extraction
from steering.eval.axbench_base import negative_prefixes as _np
from steering.eval.axbench_base import paths
from steering.eval.axbench_base._stats_reader import (
    read_n_sae_features_from_stats,
)
from steering.eval.axbench_base.cluster_artifacts import (
    _clustered_fingerprint_fields_from_identity,
    build_verified_percentile_table_with_collective_agreement,
    clustered_source_enabled,
    install_clustered_bytes_once,
    validate_cluster_candidate_identity_with_collective_agreement,
    validate_resume_statistics_identity_with_collective_agreement,
    write_clustered_parquet_once,
)
from steering.eval.axbench_base.concept_examples_qc import QCFeatureScore
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.cuda_memory import release_cuda_iteration_memory
from steering.eval.axbench_base.qc_buffer import QcBuffer
from steering.eval.axbench_base.stage0c_api_generation import Stage0cApiGenerator
from steering.eval.axbench_base.stage0c_partition import (
    _validate_rank_checkpoint_state_for_merge,
    is_my_candidate,
    merge_per_rank_checkpoints,
    my_candidate_block,
    read_per_rank_checkpoint,
    validate_per_rank_fingerprint,
    write_per_rank_checkpoint,
)
from steering.eval.axbench_base.synth.llm_response_cache import SynthResponseCache
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    QcOutcome,
    RejectedConceptReason,
    TargetModelMetadata,
)
from steering.eval.axbench_base.validate import (
    load_and_validate_candidate_concepts,
    load_and_validate_concepts,
    validate_concepts,
)
from steering.feature_io import glob_features
from steering.refined_types import NonNegativeInt, PositiveInt

if TYPE_CHECKING:
    from steering.eval.cluster_feature_contract import ClusterFeatureContract


# Default block size (B). Mirrors the dataclass default for
# ``cfg.stage0c_multi_gpu_block_size`` so callers (and tests) can refer
# to a stable constant without instantiating a config. Task 8 wired
# this through the canonical cfg field; the runner reads from the cfg
# directly (no ``getattr`` fallback) and the fingerprint includes it
# so a block-size change rejects stale checkpoints.
_DEFAULT_BLOCK_SIZE = 16


class _ClusterCandidatePrefixIndex:
    """Build one concept-owned member union in canonical location order.

    The inner index returns fresh member lists and retains none of them.
    ``member_candidates`` therefore bounds clustered candidate memory to the
    active concept while preserving member traversal, location deduplication,
    the shared per-shard cap, and final sort.
    """

    def __init__(
        self,
        inner: _si._CandidatePrefixLookup,
        contract: ClusterFeatureContract,
        *,
        candidate_scan_limit: int | None = None,
    ) -> None:
        self._inner = inner
        self._contract = contract
        self._candidate_scan_limit = candidate_scan_limit

    def candidates_for_feature(self, feature_id: int) -> list[_si._PrefixCandidate]:
        target = self._contract.target_for_representative(feature_id)
        member_candidates: list[_si._PrefixCandidate] = []
        for member_id in target.allowed_member_ids:
            member_candidates.extend(self._inner.candidates_for_feature(member_id))
        return _si._canonical_cluster_candidate_union(
            member_candidates,
            per_shard_limit=self._candidate_scan_limit,
        )


def _block_size(cfg: AxBenchBaseConfig, world_size: int = 1) -> int:
    """Resolve the effective block size B from cfg, capped by max_concepts.

    PR-4 Task 8 made ``stage0c_multi_gpu_block_size`` a real
    :class:`AxBenchBaseConfig` field with a default of
    :data:`_DEFAULT_BLOCK_SIZE`. For small-target runs, the configured
    value can be far too coarse: with ``world_size=4`` and the default
    ``block_size=16``, each rendezvous covers 64 candidates — overshooting
    a ``max_concepts=10`` target by 6× before the global ``accepts >=
    max_concepts`` check can fire. Observed 2026-05-15 on a d12_l12
    10-concept run: rank 0 reached its block-0 cap with 13 accepts and
    then spun on NCCL for 30+ min while ranks 1-3 ground through their
    bonus candidates, with no way to early-exit until each rank
    independently finished all 16 of its block-0 candidates.

    The cap is ``ceil(max_concepts * 2 / world_size)`` — enough headroom
    for typical 50-70% accept rates to fill ``max_concepts`` within one
    or two blocks, without the runaway over-processing. A run with
    ``max_concepts >= block_size * world_size`` sees no change (the
    configured value is already smaller than the cap), so the standard
    (50-concept) and full (500-concept) tiers are unaffected.

    The return value is clamped to ``>= 1`` unconditionally: a 0 or
    negative configured/``max_concepts`` (reachable only when
    ``validate_config`` is bypassed, e.g. unit tests) would otherwise
    yield ``B == 0`` and crash ``my_candidate_block`` / zero out the
    ``covers`` term in the termination predicate.
    """
    configured = int(cfg.stage0c_multi_gpu_block_size)
    if world_size <= 1:
        # ws<=1: configured value verbatim (preserves byte-identity with
        # the legacy single-rank loop), but never below 1 — a 0/negative
        # configured (reachable when validate_config is bypassed) would
        # make my_candidate_block raise and covers==0.
        return max(1, configured)
    # Two-blocks-worth of headroom for the per-rank accept quota; ceil
    # so we never go below 1 even when ``max_concepts < world_size``.
    per_rank_quota = max(
        1, (cfg.max_concepts * 2 + world_size - 1) // world_size,
    )
    # Outer ``max(1, …)``: ``min`` can still be 0 if ``configured == 0``.
    # Effective B >= 1 is a hard precondition of the block loop.
    return max(1, min(configured, per_rank_quota))


def _accept_dict_for_concept(
    candidate: CandidateConcept,
    *,
    candidate_index: int,
    qc_score: int | None,
    n_verified_rows: int,
) -> dict[str, Any]:
    """Build the rank_accepts payload entry for one accepted candidate.

    The dict carries every field
    :func:`merge_per_rank_checkpoints` reads when reconstructing the
    canonical :class:`Concept` (sans ``concept_id``, which is assigned at
    merge time). ``qc_score`` and ``verified_rows_count`` are extra
    diagnostic fields useful for reject-vs-accept post-mortems but are
    not consumed by the merge.
    """
    return {
        "candidate_index": int(candidate_index),
        "feature_index": int(candidate.feature_index),
        "layer": int(candidate.layer),
        "sae_source": str(candidate.sae_source),
        "description": str(candidate.description),
        "corpus_count": int(candidate.corpus_count),
        "description_source": str(candidate.description_source),
        "qc_score": qc_score,
        "verified_rows_count": int(n_verified_rows),
    }


def _row_dicts_for_accept(
    rows: list[_si._VerifiedNegativePrefix],
    *,
    candidate_index: int,
    placeholder_concept_id: int,
) -> list[dict[str, Any]]:
    """Convert verified-prefix dataclasses into per-rank parquet row dicts.

    The ``concept_id`` here is a rank-local placeholder; the merge step
    overwrites it with the canonical dense concept_id keyed by
    ``candidate_index``. ``candidate_index`` is the cross-rank-stable
    field the merge sorts by — see :func:`merge_per_rank_checkpoints`.
    """
    out: list[dict[str, Any]] = []
    for row in rows:
        d = asdict(replace(row, concept_id=int(placeholder_concept_id)))
        d["candidate_index"] = int(candidate_index)
        out.append(d)
    return out


def _build_payload(
    cfg: AxBenchBaseConfig,
    *,
    rank: int,
    world_size: int,
    n_candidates: int,
    next_block_index: int,
    visited_candidate_indices: list[int],
    rank_accepts: list[dict[str, Any]],
    rank_rejects: list[dict[str, Any]],
    rank_qc_per_feature: list[QCFeatureScore],
    candidate_pool_identity: str,
    precomputed_clustered_fields: Mapping[str, str],
    stopped_by_qc_survivor_cap: bool = False,
) -> dict[str, Any]:
    """Build the payload dict written to ``stage0c_partial_rank_{rank}.json``.

    The payload schema MUST satisfy the contract in
    :func:`merge_per_rank_checkpoints` (which reads ``fingerprint``,
    ``rank_accepts``, ``rank_rejects``, ``rank_qc_per_feature``) AND the
    contract in :func:`validate_per_rank_fingerprint` (fingerprint dict
    must equal the one computed from the current ``cfg``).

    ``candidate_pool_identity`` is a content hash of the ordered candidate
    pool (see :func:`_stage0c_internals.candidate_pool_identity`); it is
    folded into the fingerprint so a regenerated pool under the same pinned
    ``run_tag`` refuses to resume against stale accepted concepts.

    ``precomputed_clustered_fields`` (PR #739 FU-D-2): forwarded to
    :func:`_stage0c_internals._fingerprint_with_pool` -- called once per
    block boundary (every ``_commit_partial`` in :func:`run_single_rank`),
    so without this the clustered candidate-identity sidecar would
    otherwise be re-read from disk at every commit. Required (owner review
    of PR #884: no more nullable-then-fresh-read default at this layer) --
    every caller threads the dispatcher-resolved value, ``{}`` for an
    unclustered ``cfg``. ``_fingerprint_with_pool`` below still accepts
    ``None`` for its own genuinely-live (not cached) callers; see its
    docstring.
    """
    return {
        "rank": int(rank),
        "world_size": int(world_size),
        "n_candidates": int(n_candidates),
        "fingerprint": _si._fingerprint_with_pool(
            cfg,
            candidate_pool_identity=candidate_pool_identity,
            precomputed_clustered_fields=precomputed_clustered_fields,
        ),
        "next_block_index": int(next_block_index),
        "visited_candidate_indices": list(visited_candidate_indices),
        "rank_accepts": list(rank_accepts),
        "rank_rejects": list(rank_rejects),
        "rank_qc_per_feature": [asdict(q) for q in rank_qc_per_feature],
        "stopped_by_qc_survivor_cap": bool(stopped_by_qc_survivor_cap),
    }


def _expected_per_rank_row_count(
    cfg: AxBenchBaseConfig,
    rank_accepts: list[dict[str, Any]],
) -> int:
    """Return the exact parquet row count implied by ``rank_accepts``.

    Each accept records the per-candidate ``verified_rows_count`` (the
    number of verified prefixes its successful verification produced;
    always ``>= cfg.n_verified_prefixes_per_concept_target``). Summing
    them gives the byte-exact row count the parquet must hold.

    Older / external accept dicts that lack ``verified_rows_count`` fall
    back to ``cfg.n_verified_prefixes_per_concept_target * len(accepts)``,
    which is a lower bound (the verifier short-circuits at target, so
    the actual count is exactly this except in degenerate paths).
    """
    if all("verified_rows_count" in a for a in rank_accepts):
        return sum(int(a["verified_rows_count"]) for a in rank_accepts)
    return len(rank_accepts) * int(cfg.n_verified_prefixes_per_concept_target)


def _try_resume_per_rank(
    cfg: AxBenchBaseConfig,
    *,
    rank: int,
    candidate_pool_identity: str,
    n_sae_features: int,
    precomputed_clustered_fields: Mapping[str, str],
    cluster_contract: ClusterFeatureContract | None = None,
    world_size: int = 1,
) -> dict[str, Any] | None:
    """Validate and return the per-rank resume payload, or ``None``.

    Returns ``None`` if the rank's JSON does not exist (fresh start). On
    fingerprint mismatch raises :class:`AxBenchBaseError` (matching the
    legacy single-rank checkpoint resume contract).

    ``candidate_pool_identity`` is the content hash of the current
    (live) candidate pool. It is folded into the expected fingerprint so
    a resume against a regenerated pool (same pinned ``run_tag``, changed
    ``concept_feature_whitelist_path`` / ``min_corpus_count`` / descriptions)
    is rejected on the ``candidate_pool_identity`` fingerprint field
    rather than silently reusing stale accepted concepts.

    ``cluster_contract`` / ``n_sae_features`` / ``world_size`` (Codex
    cross-review fix batch, FU-D-stage0c, Bug 1 HIGH + LAST-known-analog
    follow-up): ``_fingerprint_with_pool`` below records only ``cfg.
    feature_statistics_path`` — the path STRING, never the file's actual
    bytes — so it cannot detect a statistics file replaced at the same
    path. When clustering is enabled, re-validate the LIVE on-disk
    statistics identity against what the contract pinned BEFORE any
    partial rows below are accepted, via :func:`cluster_artifacts.
    validate_resume_statistics_identity_with_collective_agreement`.

    This check is invoked UNCONDITIONALLY whenever ``cluster_contract`` is
    set — even when THIS rank's own ``json_path`` does not exist (fresh
    start for this rank) — because it must call the same collective
    ``all_gather_object`` rendezvous every peer calls at this exact point.
    An earlier revision only ran this as a bare per-rank local call gated
    on ``json_path.exists()``, on the theory that
    :func:`dispatch_prepare_negative_prefixes` already ran an equivalent
    check collectively (via ``validate_cluster_candidate_identity_with_
    collective_agreement``, and later also ``build_verified_percentile_
    table_with_collective_agreement``) for every rank before any rank
    reaches :func:`run_single_rank`. That reasoning does not hold: both of
    those collectives complete BEFORE ``run_single_rank`` is called, and
    non-trivial per-rank-local work happens in between (building the
    candidate prefix index from the live parquet/feature-shard files, and
    optionally constructing a QC client / API generator) — a statistics
    swap landing in that window could still be observed asymmetrically by
    only one rank's resume check, which would then raise alone while its
    peers (including a peer with no local checkpoint, whose copy of this
    check previously never ran at all) proceeded into ``run_single_rank``'s
    block-barrier ``_all_gather_state`` collective and hung forever. The
    collective wrapper closes that window the same way its siblings do:
    every rank posts its local outcome (a trivial "ok" when it has no
    local checkpoint to validate) and all ranks raise together on any
    disagreement. No-op when ``cluster_contract`` is ``None`` (unclustered
    path, U-invariant); at ``world_size <= 1`` this is exactly the wrapped
    local call with no ``torch.distributed`` usage, so the pre-fix
    single-rank behavior is unchanged.

    ``precomputed_clustered_fields`` (PR #739 FU-D-2): forwarded to
    :func:`_stage0c_internals._fingerprint_with_pool` when computing
    ``expected_fp`` below, so this no longer re-reads the clustered
    candidate/provenance sidecar from disk on every resume. The checkpoint
    FINGERPRINT COMPARISON itself (this rank's stored fingerprint vs.
    ``expected_fp``) remains the original bare per-rank
    :func:`validate_per_rank_fingerprint` call below, byte-identical to
    pre-FU-D-2 behavior for every ``cluster_contract`` value (U-invariant).
    A per-rank raise there on an abnormal mixed-state resume is an
    ACCEPTED-RESIDUAL failure mode by owner decision (PR #739 FU-D-2
    descope, 2026-07-26): see the comment at the raise site below and the
    PR #739 campaign ledger.

    Mid-write crash recovery (PR-4 review hardening): per-rank checkpoint
    writes order parquet first, JSON last (see
    :func:`stage0c_partition.write_per_rank_checkpoint`). If a rank
    crashes between those two writes the parquet on disk has rows for
    the (K+1)-th accept that the JSON does not yet count. On resume,
    the runner would re-process candidate K and re-append duplicate
    rows. To prevent that, slice the trailing rows off so the resume
    re-derives them deterministically from candidate K. Mirrors the
    legacy single-rank guard at
    :mod:`steering.eval.axbench_base._stage0c_internals` (search for
    ``actual_rows`` — the single-rank loop has the same slice-or-error
    branch).

    Negative path: if the JSON records strictly more accepts than the
    parquet has rows, the parquet was truncated; no safe recovery is
    possible because the per-accept rows from earlier candidates are
    gone. Raise :class:`AxBenchBaseError` with a clear "data loss"
    diagnostic so the operator can delete the partial files and
    restart.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before calling _try_resume_per_rank")
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    json_path = output_paths.stage0c_partial_rank_json(rank)
    parquet_path = output_paths.stage0c_partial_rank_parquet(rank)
    has_local_checkpoint = json_path.exists()
    if cluster_contract is not None:
        # Called unconditionally (regardless of has_local_checkpoint) so
        # every rank in the process group participates in the SAME
        # all_gather_object rendezvous -- see the docstring above.
        validate_resume_statistics_identity_with_collective_agreement(
            cfg,
            cluster_contract,
            has_local_checkpoint=has_local_checkpoint,
            n_sae_features=n_sae_features,
            world_size=world_size,
            surface="Clustered Stage 0c partial-checkpoint resume",
        )
    if not has_local_checkpoint:
        return None
    expected_fp = _si._fingerprint_with_pool(
        cfg,
        candidate_pool_identity=candidate_pool_identity,
        precomputed_clustered_fields=precomputed_clustered_fields,
    )
    # ACCEPTED-RESIDUAL (PR #739 FU-D-2 descope, owner decision 2026-07-26):
    # a per-rank fingerprint-comparison raise here on an abnormal
    # mixed-state resume (e.g. a stale-pool checkpoint written before
    # Stage 0a regenerated the candidate/provenance sidecar under the same
    # pinned run_tag) is a bounded, loud failure -- this rank exits with
    # the real error; peers are torn down by torchrun's supervisor or the
    # NCCL collective timeout, not left silently corrupted. Deliberately
    # NOT wrapped in collective-agreement machinery (the sibling vote this
    # descope removed); see the PR #739 FU-D-2 campaign ledger.
    validate_per_rank_fingerprint(
        json_path,
        expected_fp,
        expected_schema_version=_si._stage0c_checkpoint_schema_version(cfg),
    )
    payload, rows = read_per_rank_checkpoint(
        json_path,
        parquet_path,
        expected_schema_version=_si._stage0c_checkpoint_schema_version(cfg),
    )

    rank_accepts = list(payload.get("rank_accepts", []))
    expected_rows = _expected_per_rank_row_count(cfg, rank_accepts)
    actual_rows = len(rows)
    if actual_rows > expected_rows:
        # Parquet has rows for the (K+1)-th accept whose JSON entry
        # never made it to disk. Slice off the trailing rows so the
        # rank re-processes candidate K from scratch and re-derives the
        # exact same rows deterministically. The on-disk parquet still
        # carries the stale rows; the next ``write_per_rank_checkpoint``
        # rewrites it from the in-memory accumulator.
        logger.info(
            "[stage0c] rank {r} mid-write-crash recovery: parquet has "
            "{a} rows but JSON tally implies {e}; slicing trailing "
            "{n} row(s) so resume re-derives them.",
            r=rank, a=actual_rows, e=expected_rows,
            n=actual_rows - expected_rows,
        )
        rows = rows[:expected_rows]
    elif actual_rows < expected_rows:
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {parquet_path} has "
            f"{actual_rows} rows but the JSON at {json_path} records "
            f"{len(rank_accepts)} accepted prefixes implying "
            f"{expected_rows} rows. This is data loss (parquet was "
            "truncated). Delete the per-rank stage0c_partial_rank_*.json/"
            "parquet files for this run_tag to restart Stage 0c from scratch."
        )
    payload["_resumed_rows"] = rows
    return payload


def _reconcile_resumed_state_for_rank(
    cfg: AxBenchBaseConfig,
    payload: dict[str, Any],
    candidate_concepts: list[CandidateConcept],
    *,
    rank: int,
    world_size: int,
) -> tuple[dict[str, Any], bool]:
    """Reconcile one saved rank payload with the current DDP topology.

    A rank-local checkpoint may be reused across world-size changes, but only
    work whose candidate remains owned by the same rank is visible to the new
    owner. Retaining moved accepts/rejects/rows would make the new owner process
    the same candidate again and feed duplicate or contradictory state to
    merge. Validate the old state under its original topology, filter every
    correlated collection together, then validate the result under the current
    topology before the runner continues.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before reconciling Stage 0c resume state"
        )
    prior_world_size = payload.get("world_size")
    if (
        not isinstance(prior_world_size, int)
        or isinstance(prior_world_size, bool)
        or prior_world_size <= 0
    ):
        raise AxBenchBaseError(
            "Per-rank Stage 0c checkpoint has invalid world_size="
            f"{prior_world_size!r}"
        )
    if (
        prior_world_size != world_size
        and cfg.enable_concept_examples_qc
        and cfg.concept_examples_qc_target_surviving is not None
    ):
        raise AxBenchBaseError(
            "Cannot resume Stage 0c across a world_size change while "
            "concept_examples_qc_target_surviving is set: world_size=1 "
            "enforces this early-stop cap, while multi-rank execution "
            "intentionally disables it. Use the original world_size or delete "
            "the per-rank partial files and restart under the new topology."
        )
    if payload.get("n_candidates") != len(candidate_concepts):
        raise AxBenchBaseError(
            "Per-rank Stage 0c checkpoint candidate count "
            f"{payload.get('n_candidates')!r} does not match the current "
            f"candidate pool length {len(candidate_concepts)}"
        )

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    json_path = output_paths.stage0c_partial_rank_json(rank)
    parquet_path = output_paths.stage0c_partial_rank_parquet(rank)
    rows = payload.get("_resumed_rows", [])
    if not isinstance(rows, list):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has non-list "
            "resumed parquet rows"
        )
    _validate_rank_checkpoint_state_for_merge(
        payload,
        rows,
        rank=rank,
        world_size=prior_world_size,
        json_path=json_path,
        parquet_path=parquet_path,
    )

    topology_changed = prior_world_size != world_size
    visited = list(payload.get("visited_candidate_indices", []))
    typed_rank = NonNegativeInt.parse(rank)
    typed_world_size = PositiveInt.parse(world_size)
    retained_visited = [
        candidate_idx
        for candidate_idx in visited
        if is_my_candidate(
            NonNegativeInt.parse(candidate_idx),
            rank=typed_rank,
            world_size=typed_world_size,
        )
    ]
    retained_set = set(retained_visited)

    def _retain_records(field: str) -> list[dict[str, Any]]:
        return [
            record
            for record in payload.get(field, [])
            if record["candidate_index"] in retained_set
        ]

    retained_accepts = _retain_records("rank_accepts")
    retained_rejects = _retain_records("rank_rejects")
    retained_accept_indices = {
        record["candidate_index"] for record in retained_accepts
    }
    retained_rows = [
        row
        for row in rows
        if row["candidate_index"] in retained_accept_indices
    ]

    feature_to_candidate_index = {
        candidate.feature_index: candidate_idx
        for candidate_idx, candidate in enumerate(candidate_concepts)
    }
    raw_qc = payload.get("rank_qc_per_feature", [])
    if not isinstance(raw_qc, list):
        raise AxBenchBaseError(
            f"Per-rank Stage 0c checkpoint at {json_path} has non-list "
            "rank_qc_per_feature"
        )
    retained_qc: list[dict[str, Any]] = []
    seen_qc_features: set[int] = set()
    visited_set = set(visited)
    for qc_idx, qc_record in enumerate(raw_qc):
        if not isinstance(qc_record, dict):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} "
                f"rank_qc_per_feature[{qc_idx}] must be an object"
            )
        qc_record = cast(dict[str, Any], qc_record)
        feature_index = qc_record.get("feature_index")
        if (
            not isinstance(feature_index, int)
            or isinstance(feature_index, bool)
            or feature_index in seen_qc_features
        ):
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has invalid or "
                f"duplicate QC feature_index={feature_index!r}"
            )
        seen_qc_features.add(feature_index)
        candidate_idx = feature_to_candidate_index.get(feature_index)
        if candidate_idx is None or candidate_idx not in visited_set:
            raise AxBenchBaseError(
                f"Per-rank Stage 0c checkpoint at {json_path} has QC state "
                f"for feature_index={feature_index!r} without a visited "
                "candidate"
            )
        if candidate_idx in retained_set:
            retained_qc.append(qc_record)

    if not topology_changed:
        saved_next_block = int(payload["next_block_index"])
        block_span = _block_size(cfg, world_size) * world_size
        completed_boundary = min(
            len(candidate_concepts),
            saved_next_block * block_span,
        )
        visited_set = set(visited)
        missing_before_boundary = [
            candidate_idx
            for candidate_idx in range(rank, completed_boundary, world_size)
            if candidate_idx not in visited_set
        ]
        if not missing_before_boundary:
            return payload, False
        earliest_missing = missing_before_boundary[0]
        rewound_block = earliest_missing // block_span
        rewound = {**payload, "next_block_index": rewound_block}
        logger.warning(
            "[stage0c] rank {rank} checkpoint claimed completed blocks "
            "through {saved}, but candidate {missing} was unvisited; rewinding "
            "to block {rewound}",
            rank=rank,
            saved=saved_next_block,
            missing=earliest_missing,
            rewound=rewound_block,
        )
        return rewound, True

    reconciled = {
        **payload,
        "_prior_world_size": prior_world_size,
        "world_size": world_size,
        "next_block_index": 0,
        "visited_candidate_indices": retained_visited,
        "rank_accepts": retained_accepts,
        "rank_rejects": retained_rejects,
        "rank_qc_per_feature": retained_qc,
        "_resumed_rows": retained_rows,
    }
    _validate_rank_checkpoint_state_for_merge(
        reconciled,
        retained_rows,
        rank=rank,
        world_size=world_size,
        json_path=json_path,
        parquet_path=parquet_path,
    )
    logger.info(
        "[stage0c] rank {rank} reconciled checkpoint topology "
        "{prior_ws}->{current_ws}: retained {retained}/{visited} visited "
        "candidate(s), {accepts} accept(s), {rejects} reject(s), and "
        "{rows} row(s)",
        rank=rank,
        prior_ws=prior_world_size,
        current_ws=world_size,
        retained=len(retained_visited),
        visited=len(visited),
        accepts=len(retained_accepts),
        rejects=len(retained_rejects),
        rows=len(retained_rows),
    )
    return reconciled, True


def _qc_resume_start_at(
    *,
    n_candidates: int,
    visited_candidate_indices: list[int],
    rank: int,
    world_size: int,
) -> int:
    """Return the first unvisited candidate currently owned by this rank."""
    visited = set(visited_candidate_indices)
    return next(
        (
            candidate_idx
            for candidate_idx in range(rank, n_candidates, world_size)
            if candidate_idx not in visited
        ),
        n_candidates,
    )


def _all_gather_state(
    state: tuple[int, int, int, int],
    *,
    world_size: int,
) -> list[tuple[int, int, int, int]]:
    """Gather ``(n_accepted, n_visited, max_accepted_idx, next_block_index)``
    from all ranks.

    Returns a list indexed by rank. When ``world_size == 1`` the function
    short-circuits without touching :mod:`torch.distributed` so single-GPU
    runs exit cleanly even when the process group is not initialized.
    """
    if world_size <= 1:
        return [state]
    out_list: list[tuple[int, int, int, int]] = [(-1, -1, -1, -1)] * world_size
    dist.all_gather_object(out_list, state)
    return out_list


def _load_candidate_pool_with_collective_identity_agreement(
    path: Path,
    *,
    n_sae_features: int,
    world_size: int,
) -> list[CandidateConcept]:
    """Load locally, then make every distributed rank agree before work.

    Filesystem cache lag or a concurrent Stage 0a replacement can otherwise
    give ranks different pool lengths. Their pool-exhaustion branches would
    then execute different collective sequences and deadlock before merge has
    a chance to compare checkpoint fingerprints. Local load failures are also
    gathered so one malformed view cannot strand healthy peers.
    """
    if world_size <= 1 or not (dist.is_available() and dist.is_initialized()):
        return load_and_validate_candidate_concepts(
            path,
            n_sae_features=n_sae_features,
        )

    candidates: list[CandidateConcept] | None = None
    try:
        candidates = load_and_validate_candidate_concepts(
            path,
            n_sae_features=n_sae_features,
        )
        local_status: dict[str, Any] = {
            "ok": True,
            "candidate_pool_identity": _si.candidate_pool_identity(candidates),
            "n_candidates": len(candidates),
            "error_type": None,
            "error": None,
        }
    except Exception as exc:
        local_status = {
            "ok": False,
            "candidate_pool_identity": None,
            "n_candidates": None,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }

    gathered: list[dict[str, Any] | None] = [None] * world_size
    dist.all_gather_object(gathered, local_status)
    malformed_ranks = [
        gathered_rank
        for gathered_rank, status in enumerate(gathered)
        if not isinstance(status, dict)
    ]
    if malformed_ranks:
        raise AxBenchBaseError(
            "Stage 0c candidate-pool identity collective returned malformed "
            f"state for ranks {malformed_ranks!r}"
        )
    statuses = cast(list[dict[str, Any]], gathered)
    failed = [
        (gathered_rank, status.get("error_type"), status.get("error"))
        for gathered_rank, status in enumerate(statuses)
        if status.get("ok") is not True
    ]
    if failed:
        raise AxBenchBaseError(
            "Stage 0c candidate-pool load failed collectively before rank "
            f"work: {failed!r}"
        )
    identities = {
        (status.get("candidate_pool_identity"), status.get("n_candidates"))
        for status in statuses
    }
    if len(identities) != 1:
        per_rank = [
            (
                gathered_rank,
                status.get("candidate_pool_identity"),
                status.get("n_candidates"),
            )
            for gathered_rank, status in enumerate(statuses)
        ]
        raise AxBenchBaseError(
            "Stage 0c ranks loaded different ordered candidate pools before "
            f"rank work; per-rank identity/count={per_rank!r}. Refusing to "
            "enter divergent collectives. Stabilize candidate_concepts.json "
            "and restart Stage 0c."
        )
    assert candidates is not None
    return candidates


def _cleanup_per_rank_partial_checkpoints(cfg: AxBenchBaseConfig) -> None:
    """Delete every per-rank stage0c partial file (idempotent).

    Globs ``stage0c_partial_rank_*.{json,parquet}`` so orphans from a
    prior LARGER world_size are also cleaned. Without the glob, a
    ws_initial=4 → ws_resume=2 sequence would leave rank 2 / rank 3
    files behind, and the next run_merge call would still read them
    even though the current ranks aren't writing them.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before _cleanup_per_rank_partial_checkpoints"
        )
    run_dir = paths.AxBenchBasePaths.from_config(cfg).run_output_root()
    if not run_dir.exists():
        return
    for partial in run_dir.glob("stage0c_partial_rank_*.json"):
        partial.unlink(missing_ok=True)
    for partial in run_dir.glob("stage0c_partial_rank_*.parquet"):
        partial.unlink(missing_ok=True)


def run_single_rank(
    cfg: AxBenchBaseConfig,
    candidate_concepts: list[CandidateConcept],
    *,
    rank: int,
    world_size: int,
    base_model: Any,
    base_sae: Any,
    target_model_meta: TargetModelMetadata,
    qc_client: Any | None,
    parquets: list[Path],
    feature_shards: list[Path],
    percentile_table: t.Tensor,
    candidate_index: _si._CandidatePrefixLookup,
    base_context_window: int,
    n_sae_features: int,
    verification_api_generator: Stage0cApiGenerator | None = None,
    precomputed_clustered_fields: Mapping[str, str],
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    """Process this rank's stripe of candidates.

    Visits the indices owned by ``(rank, world_size)`` (modulo partition,
    block-iterated in chunks of size ``B*world_size``), runs LLM-QC + N-seed
    verification per visited candidate, and commits a per-rank checkpoint
    pair (``stage0c_partial_rank_{rank}.{json,parquet}``) at every block
    boundary.

    Multi-GPU coordination uses block-barrier termination: every block of
    ``B*world_size`` candidates, all ranks rendezvous via
    ``dist.all_gather_object`` to share their
    ``(n_accepted_so_far, n_visited_so_far, max_accepted_candidate_index)``.
    The loop terminates once ``sum(n_accepted) >= cfg.max_concepts`` AND
    every rank has visited at least up to the maximum accepted
    candidate_index, OR the candidate pool is exhausted.

    Single-GPU equivalence: at ``world_size == 1``, this function
    produces byte-identical per-rank output to what the legacy
    single-rank loop produced (verified by the tests in
    ``tests/axbench_base/unit/test_stage0c_runner.py``).

    Rank-failure semantics: an uncaught exception inside the loop will
    leave the surviving ranks blocked on the next ``all_gather_object``
    call until the process-group collective timeout expires. PyTorch's
    backend default (~10 min NCCL / 30 min GLOO) is preserved unless
    ``NCCL_TIMEOUT_SEC`` is set — ``compute_init`` does not impose a
    global default. The Stage 0c launcher
    (``scripts/eval/axbench_base_eval``) defaults ``NCCL_TIMEOUT_SEC``
    to 1800s, so for Stage 0c runs the bound is 1800s (other DDP jobs
    keep PyTorch's default). The torchrun launcher kills the entire job
    on first rank failure, so the hang is bounded by the launcher's
    exit hook, not by the collective timeout.

    ``precomputed_clustered_fields`` (PR #739 FU-D-2): the clustered
    Stage 0c fingerprint-fields dict already resolved ONCE by the
    dispatcher's startup collective vote (``cluster_artifacts.validate_
    cluster_candidate_identity_with_collective_agreement``), threaded
    verbatim to every per-block checkpoint commit (``_build_payload``) and
    the resume check (``_try_resume_per_rank``) instead of letting each
    independently re-read the candidate/provenance sidecar from disk.
    Required (owner review of PR #884: no more nullable-then-fresh-read
    default at this layer) -- the dispatcher always threads a resolved
    value, ``{}`` for an unclustered ``cfg``.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling run_single_rank"
        )
    if (
        world_size > 1
        and cfg.enable_concept_examples_qc
        and cfg.concept_examples_qc_target_surviving is not None
    ):
        raise AxBenchBaseError(
            "concept_examples_qc_target_surviving is supported only at "
            "world_size=1. Multi-rank Stage 0c cannot enforce that global "
            "live-QC survivor cap without topology-dependent overrun; set it "
            "to null or launch a single rank."
        )
    n_candidates = len(candidate_concepts)
    # Content hash of the ordered candidate pool: threaded into the
    # per-rank checkpoint fingerprint so a regenerated pool refuses to
    # resume against this rank's stale accepted concepts.
    pool_identity = _si.candidate_pool_identity(candidate_concepts)
    target = cfg.n_verified_prefixes_per_concept_target
    block_size_b = _block_size(cfg, world_size)

    # ----------------------------- Resume -------------------------------
    accepted_locally: list[dict[str, Any]] = []
    rejected_locally: list[dict[str, Any]] = []
    qc_per_feature_locally: list[QCFeatureScore] = []
    rows_locally: list[dict[str, Any]] = []
    visited_locally: list[int] = []
    next_block_index = 0
    stopped_by_qc_survivor_cap = False
    invalidate_old_json_before_next_commit = False

    resumed = _try_resume_per_rank(
        cfg, rank=rank, candidate_pool_identity=pool_identity,
        cluster_contract=cluster_contract, n_sae_features=n_sae_features,
        world_size=world_size,
        precomputed_clustered_fields=precomputed_clustered_fields,
    )
    if resumed is not None:
        resumed, _ = _reconcile_resumed_state_for_rank(
            cfg,
            resumed,
            candidate_concepts,
            rank=rank,
            world_size=world_size,
        )
        accepted_locally = list(resumed.get("rank_accepts", []))
        rejected_locally = list(resumed.get("rank_rejects", []))
        qc_per_feature_locally = [
            QCFeatureScore(**q) for q in resumed.get("rank_qc_per_feature", [])
        ]
        rows_locally = list(resumed.get("_resumed_rows", []))
        visited_locally = list(resumed.get("visited_candidate_indices", []))
        stopped_by_qc_survivor_cap = resumed[
            "stopped_by_qc_survivor_cap"
        ]
        # Reconciliation resets next_block_index to zero on a topology
        # change and retains only candidates still owned by this rank.
        prior_world_size = int(
            resumed.get("_prior_world_size", resumed.get("world_size", world_size))
        )
        invalidate_old_json_before_next_commit = prior_world_size != world_size
        next_block_index = int(resumed.get("next_block_index", 0))
        rank_tag = "" if world_size == 1 else f" (rank {rank})"
        logger.info(
            "[stage0c]{tag} resumed: next_block_index={b}, "
            "accepted={a}, rejected={rj}, qc={q}, visited={v} "
            "(prior_world_size={pw}, current world_size={cw})",
            tag=rank_tag, b=next_block_index, a=len(accepted_locally),
            rj=len(rejected_locally), q=len(qc_per_feature_locally),
            v=len(visited_locally), pw=prior_world_size, cw=world_size,
        )

    # ----------------------------- QC buffer ----------------------------
    # Start at the first CURRENTLY owned, unvisited candidate. On a topology
    # change, newly owned holes may be lower than max(retained_visited), so
    # max(visited)+1 would make QcBuffer return None and silently skip work.
    qc_start_at = _qc_resume_start_at(
        n_candidates=n_candidates,
        visited_candidate_indices=visited_locally,
        rank=rank,
        world_size=world_size,
    )

    qc_buffer: QcBuffer | None = None
    if qc_client is not None:
        # Under multi-rank, target_surviving=None to ensure each rank can
        # visit any of its owned candidates without a global cap leaking
        # into per-rank QC API fanout. The global cap is enforced via the
        # all_gather rendezvous below (and via the legacy single-rank
        # cap path when world_size == 1).
        ts_for_buffer: int | None
        if world_size == 1:
            ts_for_buffer = cfg.concept_examples_qc_target_surviving
        else:
            ts_for_buffer = None
        qc_buffer = QcBuffer(
            qc_client=qc_client,
            candidates=candidate_concepts,
            qc_batch_size=cfg.qc_batch_size,
            target_surviving=ts_for_buffer,
            start_at=qc_start_at,
            pipeline=cfg.qc_pipeline,
        )

    def _commit_partial() -> None:
        """Atomically write this rank's checkpoint pair."""
        nonlocal invalidate_old_json_before_next_commit
        output_paths = paths.AxBenchBasePaths.from_config(cfg)
        json_path = output_paths.stage0c_partial_rank_json(rank)
        parquet_path = output_paths.stage0c_partial_rank_parquet(rank)
        payload = _build_payload(
            cfg,
            rank=rank,
            world_size=world_size,
            n_candidates=n_candidates,
            next_block_index=next_block_index,
            visited_candidate_indices=visited_locally,
            rank_accepts=accepted_locally,
            rank_rejects=rejected_locally,
            rank_qc_per_feature=qc_per_feature_locally,
            candidate_pool_identity=pool_identity,
            stopped_by_qc_survivor_cap=stopped_by_qc_survivor_cap,
            precomputed_clustered_fields=precomputed_clustered_fields,
        )
        if invalidate_old_json_before_next_commit:
            # Ordinary commits only append/sustain rows, so parquet-first and
            # JSON-last lets resume discard a parquet tail after a crash.
            # Topology reconciliation can *shrink* the retained row set. An
            # old JSON commit marker paired with that smaller parquet would
            # instead look like irrecoverable data loss. Invalidate the old
            # marker first: a crash anywhere in this one transition then
            # leaves no committed pair, and the rank safely restarts rather
            # than interpreting two transaction generations as one.
            json_path.unlink(missing_ok=True)
        write_per_rank_checkpoint(
            json_path,
            parquet_path,
            payload,
            rows_locally,
            extraction_tokenizer_id=cfg.extraction_tokenizer_id,
            schema_version=_si._stage0c_checkpoint_schema_version(cfg),
        )
        invalidate_old_json_before_next_commit = False

    # If we resumed and have nothing left to do, still write our payload
    # once (so the merge has a file to read), then return.
    initial_visited_set = set(visited_locally)

    # The legacy loop's stage_started_at clock is per-rank in the multi-
    # rank case; that's fine because _log_candidate_completion is purely
    # informational.
    stage_started_at = time.monotonic()

    def _emit_completion_log(concept_idx: int) -> None:
        """Emit the legacy ``[Stage 0c] Completed candidate ...`` heartbeat.

        Reconstructs the typed Concept / _RejectedConcept / _VerifiedNegativePrefix
        instances from the rank-local dict accumulators so the legacy
        :func:`_log_candidate_completion` helper sees exactly the same
        structure it did under the single-rank loop. Cheap — runs once per
        candidate.
        """
        accepted_concept_objs = [
            Concept(
                concept_id=i,
                feature_index=a["feature_index"],
                layer=a["layer"],
                sae_source=a["sae_source"],
                description=a["description"],
                corpus_count=a["corpus_count"],
                description_source=a["description_source"],
            )
            for i, a in enumerate(accepted_locally)
        ]
        rejected_concept_objs = [
            _si._rejected_concept_from_json(r) for r in rejected_locally
        ]
        all_row_objs = [
            _si._VerifiedNegativePrefix(
                concept_id=int(row["concept_id"]),
                prefix_text=str(row["prefix_text"]),
                prefix_source_shard=int(row["prefix_source_shard"]),
                prefix_source_doc=int(row["prefix_source_doc"]),
                prefix_activating_seg_idx=int(row["prefix_activating_seg_idx"]),
                verification_continuations=[
                    str(s) for s in row["verification_continuations"]
                ],
                verification_seeds=[
                    int(s) for s in row["verification_seeds"]
                ],
                verification_percentiles=[
                    float(p) for p in row["verification_percentiles"]
                ],
            )
            for row in rows_locally
        ]
        _si._log_candidate_completion(
            stage_started_at=stage_started_at,
            concept_idx=concept_idx,
            n_candidates=n_candidates,
            accepted_concepts=accepted_concept_objs,
            rejected_concepts=rejected_concept_objs,
            all_rows=all_row_objs,
            max_concepts=cfg.max_concepts,
        )

    try:
        # Accumulators for the all_gather payload.
        block_idx = next_block_index
        typed_block_size = PositiveInt.parse(block_size_b)
        typed_rank = NonNegativeInt.parse(rank)
        typed_world_size = PositiveInt.parse(world_size)
        typed_n_candidates = NonNegativeInt.parse(n_candidates)
        # Continue iterating blocks until the global termination condition
        # fires or the candidate pool is exhausted.
        while True:
            # Determine this rank's owned indices in this block.
            owned = [
                idx for idx in my_candidate_block(
                    NonNegativeInt.parse(block_idx),
                    typed_block_size,
                    rank=typed_rank,
                    world_size=typed_world_size,
                    n_candidates=typed_n_candidates,
                )
                if idx not in initial_visited_set
            ]
            # Visit each owned candidate.
            for concept_idx in owned:
                # Per-rank local short-circuits matching the legacy
                # behavior. The "global accepts >= max_concepts" check is
                # delegated to the all_gather rendezvous below; we keep a
                # per-rank short-circuit so that a rank that has already
                # contributed enough accepts doesn't continue verifying
                # locally beyond what's needed when running solo.
                if (
                    world_size == 1
                    and len(accepted_locally) >= cfg.max_concepts
                ):
                    break
                if (
                    qc_client is not None
                    and cfg.concept_examples_qc_target_surviving is not None
                    and world_size == 1
                    and (
                        len(accepted_locally)
                        + sum(
                            reject.get("reason")
                            == RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES
                            for reject in rejected_locally
                        )
                    ) >= cfg.concept_examples_qc_target_surviving
                ):
                    stopped_by_qc_survivor_cap = True
                    logger.info(
                        "[Stage 0c] Reached concept_examples_qc_target_surviving={c}; "
                        "stopping candidate verification early",
                        c=cfg.concept_examples_qc_target_surviving,
                    )
                    break
                candidate = candidate_concepts[concept_idx]
                visited_locally.append(int(concept_idx))

                # ---- LLM-QC -------------------------------------------------
                llm_judge_score: int | None = None
                if qc_buffer is not None:
                    qc_score = qc_buffer.next(concept_idx)
                    if qc_score is None:
                        # Buffer fill skipped this candidate (QC client hit
                        # target_surviving early termination, only possible
                        # at world_size == 1). Treat as not-visited and
                        # continue; legacy semantics.
                        #
                        # Multi-rank invariant: we construct QcBuffer with
                        # ``target_surviving=None`` whenever world_size>1
                        # (see qc_buffer setup above), so QcBuffer.next
                        # MUST return a score for every owned candidate.
                        # A None at world_size>1 is a regression in
                        # QcBuffer batching that would silently corrupt
                        # the per-rank visited bookkeeping; raise loudly
                        # so the failure is unambiguous.
                        if world_size > 1:
                            raise AxBenchBaseError(
                                f"QcBuffer.next returned None for candidate "
                                f"{concept_idx} at world_size={world_size}. "
                                "This violates the multi-rank invariant: "
                                "target_surviving=None should ensure every "
                                "owned candidate gets a score. This typically "
                                "indicates a regression in QcBuffer batching."
                            )
                        # Pop from visited_locally because the legacy loop
                        # doesn't increment its concept_idx tally for this
                        # case and we want byte-identity at world_size=1.
                        visited_locally.pop()
                        continue
                    qc_per_feature_locally.append(qc_score)
                    llm_judge_score = qc_score.score
                    if qc_score.outcome != QcOutcome.ACCEPTED:
                        reason = RejectedConceptReason.from_qc_outcome(
                            qc_score.outcome
                        )
                        rej = _si._RejectedConcept(
                            candidate_index=concept_idx,
                            feature_index=candidate.feature_index,
                            description=candidate.description,
                            reason=reason,
                            n_verified_found=None,
                            llm_judge_score=llm_judge_score,
                        )
                        rejected_locally.append(_si._rejected_concept_to_json(rej))
                        rank_tag = "" if world_size == 1 else f" (rank {rank})"
                        logger.info(
                            "[Stage 0c]{tag} Rejected candidate {i}: "
                            "{reason} (feature {f}, llm_score={s})",
                            tag=rank_tag, i=concept_idx, reason=reason,
                            f=candidate.feature_index, s=llm_judge_score,
                        )
                        _emit_completion_log(concept_idx)
                        continue

                # ---- Verification ------------------------------------------
                # Candidate index is the only provisional identity stable
                # across world sizes and rank ownership. Verification seeds
                # and candidate shuffling consume ``concept.concept_id``;
                # rank-local QC-survivor ordinals would therefore make ws=1
                # and ws=N evaluate the same candidate with different random
                # streams. Final dense concept IDs are still assigned by merge.
                concept = _si._concept_from_candidate(
                    candidate, concept_id=concept_idx,
                )
                # Log line format mirrors the legacy single-rank loop so
                # tests that grep ``[Stage 0c]`` capture both single- and
                # multi-rank runs. The rank tag is appended only when
                # ``world_size > 1`` to keep single-GPU output identical.
                rank_tag = "" if world_size == 1 else f" (rank {rank})"
                logger.info(
                    "[Stage 0c]{tag} Verifying negative prefixes for "
                    "concept {cid} ({i}/{n}) (feature {f}, description={d!r})",
                    tag=rank_tag, cid=concept.concept_id, i=concept_idx + 1,
                    n=n_candidates, f=concept.feature_index,
                    d=concept.description,
                )
                try:
                    verify_kwargs: dict[str, Any] = {}
                    if verification_api_generator is not None:
                        verify_kwargs["api_generator"] = verification_api_generator
                    if cluster_contract is not None:
                        verify_kwargs["cluster_target"] = (
                            cluster_contract.target_for_representative(
                                concept.feature_index
                            )
                        )
                    rows = _si._verify_concept(
                        cfg, concept, parquets, feature_shards,
                        base_model, base_sae, percentile_table,
                        sae_layer=target_model_meta.sae_layer,
                        base_context_window=base_context_window,
                        candidate_index=candidate_index,
                        **verify_kwargs,
                    )
                finally:
                    try:
                        if verification_api_generator is not None:
                            verification_api_generator._release_concept_state()
                    finally:
                        release_cuda_iteration_memory()

                if len(rows) >= target:
                    placeholder_cid = len(accepted_locally)
                    accepted_locally.append(
                        _accept_dict_for_concept(
                            candidate,
                            candidate_index=concept_idx,
                            qc_score=llm_judge_score,
                            n_verified_rows=len(rows),
                        )
                    )
                    rows_locally.extend(
                        _row_dicts_for_accept(
                            rows,
                            candidate_index=concept_idx,
                            placeholder_concept_id=placeholder_cid,
                        )
                    )
                    rank_tag = "" if world_size == 1 else f" (rank {rank})"
                    logger.info(
                        "[Stage 0c]{tag} Accepted candidate {i} "
                        "(feature {f}) with {n} verified prefixes",
                        tag=rank_tag, i=concept_idx, f=candidate.feature_index,
                        n=len(rows),
                    )
                else:
                    reason = RejectedConceptReason.NOT_ENOUGH_VERIFIED_PREFIXES
                    rej = _si._RejectedConcept(
                        candidate_index=concept_idx,
                        feature_index=candidate.feature_index,
                        description=candidate.description,
                        reason=reason,
                        n_verified_found=len(rows),
                        llm_judge_score=llm_judge_score,
                    )
                    rejected_locally.append(_si._rejected_concept_to_json(rej))
                    rank_tag = "" if world_size == 1 else f" (rank {rank})"
                    logger.info(
                        "[Stage 0c]{tag} Rejected candidate {i}: "
                        "{reason} ({n}/{t} verified prefixes)",
                        tag=rank_tag, i=concept_idx, reason=reason,
                        n=len(rows), t=target,
                    )
                _emit_completion_log(concept_idx)

            # ---- End of block: commit then barrier --------------------
            next_block_index = block_idx + 1
            _commit_partial()
            block_idx = next_block_index

            # Determine global state for termination.
            local_max_accepted_idx = max(
                (a["candidate_index"] for a in accepted_locally), default=-1,
            )
            my_state = (
                len(accepted_locally),
                len(visited_locally),
                int(local_max_accepted_idx),
                int(next_block_index),
            )
            gathered = _all_gather_state(my_state, world_size=world_size)
            global_n_accepted = sum(g[0] for g in gathered)
            global_n_visited = sum(g[1] for g in gathered)
            global_max_accepted_idx = max(g[2] for g in gathered)

            # Terminate if global accepts hit the cap AND every rank has
            # already finished a block whose end-of-range covers
            # global_max_accepted_idx. After rank R finishes its share of
            # block N (i.e., next_block_index == N+1), it has visited
            # every owned candidate index in ``[0, (N+1)*B*W)``. So the
            # invariant ``next_block_index * B * W > global_max_accepted_idx``
            # tells us rank R has covered every index ≤ global_max_accepted_idx
            # that it owns — there is no straggler accept in candidate-index
            # sorted order.
            covers = block_size_b * world_size
            all_caught_up = all(
                g[3] * covers > global_max_accepted_idx
                for g in gathered
            )
            if (
                global_n_accepted >= cfg.max_concepts
                and global_max_accepted_idx >= 0
                and all_caught_up
            ):
                rank_tag = "" if world_size == 1 else f" (rank {rank})"
                logger.info(
                    "[Stage 0c]{tag} Block-barrier termination: "
                    "global_accepts={ga}/{cap}, max_accept_idx={mi}, "
                    "global_visited={gv}/{n}",
                    tag=rank_tag, ga=global_n_accepted, cap=cfg.max_concepts,
                    mi=global_max_accepted_idx, gv=global_n_visited,
                    n=n_candidates,
                )
                break

            # Pool exhausted — every owned index has been visited globally.
            if global_n_visited >= n_candidates:
                break

            # Single-rank early breaks above — also exit the outer loop.
            if world_size == 1 and (
                len(accepted_locally) >= cfg.max_concepts
                or stopped_by_qc_survivor_cap
            ):
                break

            # Ranks may resume with divergent block indices because the JSON
            # commit happens before the barrier. Never let one ahead rank exit
            # while a lagging peer will call another collective: only a
            # gathered, uniform all-past-pool decision can terminate here.
            all_past_pool = all(
                gathered_next_block * covers >= n_candidates
                for _, _, _, gathered_next_block in gathered
            )
            if all_past_pool:
                raise AxBenchBaseError(
                    "All Stage 0c ranks advanced past the candidate pool, but "
                    f"the gathered visited count is {global_n_visited}/"
                    f"{n_candidates} and no accept/cap termination applied. "
                    "Per-rank checkpoint progress is inconsistent; retain "
                    "partials for investigation."
                )

    finally:
        if qc_buffer is not None:
            qc_buffer.close()

    # Final per-rank commit (covers the case where the last visited
    # candidate didn't land on a block boundary — which can't happen with
    # the current design since we commit AT every block boundary, but
    # this is cheap insurance).
    _commit_partial()


def run_merge(
    cfg: AxBenchBaseConfig,
    *,
    world_size: int,
    n_sae_features: int | None = None,
    n_candidates: int | None = None,
    candidate_concepts_path: Path | None = None,
) -> Path:
    """Rank-0 final merge.

    Calls :func:`merge_per_rank_checkpoints`, writes the canonical
    Stage 0c artifacts (``concepts.json``, ``rejected_concepts.json``,
    ``negative_mode_prefixes.parquet``), validates them, and returns the
    parquet path.

    Cleans up the per-rank partial files only AFTER the canonical
    artifacts are durably on disk.

    ``n_sae_features``: passed by the dispatcher so we don't re-open the
    feature_statistics safetensors. When ``None`` (e.g., direct
    test calls) we read it from disk.

    Live-statistics residual (FU-D-stage0c): this ``__all__``-exported merge
    trusts that its caller already validated the pinned statistics identity
    (both production callers run it inside
    :func:`dispatch_prepare_negative_prefixes`, strictly after that check +
    the identity-bound percentile-table build); a future direct caller would
    bypass it -- an accepted, named residual, not a regression introduced here.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError("cfg.run_tag must be set before calling run_merge")
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    run_dir = output_paths.run_output_root()
    merged = merge_per_rank_checkpoints(
        run_dir=run_dir,
        world_size=world_size,
        max_concepts=cfg.max_concepts,
        checkpoint_schema_version=_si._stage0c_checkpoint_schema_version(cfg),
    )
    merged_fingerprint = merged.get("fingerprint")
    if not isinstance(merged_fingerprint, dict):
        raise AxBenchBaseError(
            "Stage 0c merge produced no canonical fingerprint; refusing to "
            "write completed artifacts without candidate-pool provenance"
        )
    pool_identity = _si._validate_candidate_pool_identity_digest(
        merged_fingerprint.get("candidate_pool_identity"),
        consumer="Stage 0c merged checkpoint fingerprint",
    )
    if n_sae_features is None:
        n_sae_features = read_n_sae_features_from_stats(
            Path(cfg.feature_statistics_path),
        )
    live_candidate_path = (
        candidate_concepts_path
        if candidate_concepts_path is not None
        else output_paths.candidate_concepts_json()
    )
    live_candidates = load_and_validate_candidate_concepts(
        live_candidate_path,
        n_sae_features,
    )
    live_pool_identity = _si.candidate_pool_identity(live_candidates)
    if live_pool_identity != pool_identity:
        raise AxBenchBaseError(
            "Stage 0c merge candidate pool changed after per-rank work: "
            f"merged checkpoints carry {pool_identity!r}, but the live "
            f"candidate pool at {live_candidate_path} has "
            f"{live_pool_identity!r}. Refusing mixed-pool canonical writes; "
            "retain the partials for investigation and restart Stage 0c."
        )
    live_sources = {candidate.sae_source for candidate in live_candidates}
    if live_sources != {cfg.neuronpedia_source}:
        raise AxBenchBaseError(
            f"Stage 0c merge live candidate pool at {live_candidate_path} "
            f"uses SAE source(s) {sorted(live_sources)!r}, but the current "
            f"config requires neuronpedia_source={cfg.neuronpedia_source!r}. "
            "Refusing canonical writes and cleanup."
        )
    merged_n_candidates = merged.get("n_candidates")
    if merged_n_candidates != len(live_candidates):
        raise AxBenchBaseError(
            "Stage 0c merged checkpoints declare n_candidates="
            f"{merged_n_candidates!r}, but the live candidate pool at "
            f"{live_candidate_path} has {len(live_candidates)} entries"
        )
    if n_candidates is not None and n_candidates != len(live_candidates):
        raise AxBenchBaseError(
            f"Stage 0c dispatcher passed n_candidates={n_candidates}, but "
            f"the merge-time candidate pool at {live_candidate_path} has "
            f"{len(live_candidates)} entries"
        )
    accepted_records = merged.get("checkpoint_accept_records")
    if not isinstance(accepted_records, list):
        raise AxBenchBaseError(
            "Stage 0c merge omitted accepted checkpoint records required for "
            "live candidate-pool field validation"
        )
    candidate_fields = (
        "feature_index",
        "layer",
        "sae_source",
        "description",
        "corpus_count",
        "description_source",
    )
    for record in accepted_records:
        candidate_idx = record["candidate_index"]
        live_candidate = live_candidates[candidate_idx]
        mismatched_fields = [
            field
            for field in candidate_fields
            if record.get(field) != getattr(live_candidate, field)
        ]
        if mismatched_fields:
            raise AxBenchBaseError(
                "Stage 0c accepted checkpoint state disagrees with live "
                f"candidate_concepts[{candidate_idx}] on fields "
                f"{mismatched_fields!r}; refusing canonical writes"
            )
    rejected_records = merged.get("checkpoint_reject_records")
    if not isinstance(rejected_records, list):
        raise AxBenchBaseError(
            "Stage 0c merge omitted rejected checkpoint records required for "
            "live candidate-pool field validation"
        )
    for record in rejected_records:
        candidate_idx = record["candidate_index"]
        live_candidate = live_candidates[candidate_idx]
        mismatched_fields = [
            field
            for field in ("feature_index", "description")
            if record.get(field) != getattr(live_candidate, field)
        ]
        if mismatched_fields:
            raise AxBenchBaseError(
                "Stage 0c rejected checkpoint state disagrees with live "
                f"candidate_concepts[{candidate_idx}] on fields "
                f"{mismatched_fields!r}; refusing canonical writes"
            )
    expected_fingerprint = _si._fingerprint_with_pool(
        cfg,
        candidate_pool_identity=pool_identity,
    )
    if _si._semantic_json_identity(merged_fingerprint) != (
        _si._semantic_json_identity(expected_fingerprint)
    ):
        actual_keys = set(merged_fingerprint)
        expected_keys = set(expected_fingerprint)
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys, key=str)
        mismatched = sorted(
            key
            for key in expected_fingerprint
            if key in merged_fingerprint
            if _si._semantic_json_identity(merged_fingerprint[key])
            != _si._semantic_json_identity(expected_fingerprint[key])
        )
        raise AxBenchBaseError(
            "Stage 0c merged per-rank fingerprint does not match the current "
            f"config (missing={missing}, extra={extra}, mismatched="
            f"{mismatched}). Refusing to write or clean up a consistently "
            "stale checkpoint set. Delete per-rank partial files and restart."
        )
    accepted_concepts: list[Concept] = merged["accepted_concepts"]
    rejected_concepts_payload: list[dict[str, Any]] = merged["rejected_concepts"]
    rows_payload: list[dict[str, Any]] = merged["rows"]
    qc_payload: list[dict[str, Any]] = merged["qc_per_feature"]
    qc_report = None
    qc_report_payload: dict[str, Any] | None = None
    if cfg.enable_concept_examples_qc:
        qc_report = _si._build_qc_report(
            n_total=len(live_candidates),
            per_feature=[QCFeatureScore(**q) for q in qc_payload],
            judge_min_score=cfg.activation_example_judge_min_score,
        )
        # Match the exact JSON-domain representation written by
        # ``_write_qc_report`` (notably score_histogram's integer keys become
        # JSON object strings). This is the payload whose semantic identity is
        # pinned in parquet metadata and later reloaded on completed resume.
        json_domain_qc_report = json.loads(
            json.dumps(asdict(qc_report), ensure_ascii=False)
        )
        qc_report_payload = _np._validate_concept_examples_qc_report_payload(
            json_domain_qc_report,
            source="in-memory merged Stage 0c QC report",
            expected_candidates=live_candidates,
            expected_judge_min_score=cfg.activation_example_judge_min_score,
        )
    _np._validate_completed_sidecars_against_candidate_pool(
        candidate_concepts=live_candidates,
        concepts=accepted_concepts,
        rejected_concepts=rejected_concepts_payload,
        concept_examples_qc_report=qc_report_payload,
        cfg=cfg,
    )
    out_path = output_paths.negative_mode_prefixes()
    canonical_table: pa.Table | None = None
    if len(accepted_concepts) >= cfg.max_concepts:
        validate_concepts(accepted_concepts, n_sae_features=n_sae_features)
        canonical_schema = _si._schema_with_completed_negative_prefixes_metadata(
            cfg,
            candidate_pool_identity=pool_identity,
        )
        canonical_schema = _si._schema_with_completed_artifact_set_identities(
            canonical_schema,
            concepts=accepted_concepts,
            rejected_concepts=rejected_concepts_payload,
            concept_examples_qc_report=qc_report_payload,
            stage0c_fingerprint=merged_fingerprint,
        )
        canonical_table = pa.Table.from_pylist(
            rows_payload,
            schema=canonical_schema,
        )
        _np._validate_negative_prefixes_table(
            canonical_table,
            path=out_path,
            concepts=accepted_concepts,
            cfg=cfg,
        )

    qc_report_path = output_paths.concept_examples_qc_report()
    clustered = clustered_source_enabled(cfg)
    concepts_path = output_paths.concepts_json()
    rejected_path = output_paths.rejected_concepts_json()
    if clustered:
        existing_canonical = [
            path
            for path in (
                out_path,
                concepts_path,
                rejected_path,
                qc_report_path,
            )
            if path.exists()
        ]
        if existing_canonical:
            raise AxBenchBaseError(
                "Clustered Stage 0c refuses to replace an existing canonical "
                f"artifact during merge: {existing_canonical}. Use a fresh "
                "run tag."
            )
    else:
        # The canonical parquet is the commit marker for the completed
        # artifact set. Invalidate an older marker before replacing either JSON
        # sidecar, so a crash during a forced rerun can expose only an
        # incomplete set, never an old parquet alongside new concepts or
        # rejections. Per-rank partials remain untouched until the new set has
        # passed post-write validation.
        if out_path.exists():
            out_path.unlink()
        if not cfg.enable_concept_examples_qc:
            # A disabled-QC completion has an explicit ``None`` report
            # identity. Remove diagnostics left by an earlier failed/enabled
            # attempt only after invalidating the old parquet commit marker.
            qc_report_path.unlink(missing_ok=True)

    # Read the per-rank "stopped_by_qc_survivor_cap" flag — any rank
    # may have set it (in single-rank-cap mode only one rank does, but
    # for forwards-compat we OR across all ranks).
    stopped_by_qc_cap = False
    for r in range(world_size):
        json_path = output_paths.stage0c_partial_rank_json(r)
        if not json_path.exists():
            continue
        rank_payload = json.loads(json_path.read_text(encoding="utf-8"))
        if rank_payload.get("stopped_by_qc_survivor_cap", False):
            stopped_by_qc_cap = True
            break

    # Write rejected_concepts.json FIRST — the legacy code wrote it
    # before the exhaustion check so operators inspecting a failed run
    # could still see why each candidate was rejected.
    #
    # Atomic write: ``atomic_write_bytes`` (uuid-suffixed tmp + fsync +
    # os.replace) ensures a crash mid-write never leaves a partial file
    # for ``_resume_if_artifacts_complete`` to misinterpret as complete.
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_bytes = json.dumps(
        rejected_concepts_payload, indent=2, ensure_ascii=False,
    ).encode("utf-8")
    if clustered:
        install_clustered_bytes_once(
            rejected_bytes,
            rejected_path,
            surface="Stage 0c rejected-concepts sidecar",
        )
    else:
        atomic_write_bytes(rejected_bytes, rejected_path)
    logger.info(
        "Wrote {n} rejected concept records to {p}",
        n=len(rejected_concepts_payload), p=str(rejected_path),
    )

    # ---- Write QC report (when QC was enabled) ----
    # Also written before the exhaustion check so the legacy
    # ``test_stage_0c_writes_qc_report_for_judged_candidates`` semantics
    # are preserved: every QC-judged feature appears in the report even
    # if the run later raises max_concepts exhaustion. ``n_total`` is the
    # whole candidate pool size (NOT the number of QC-processed ones) —
    # that's the legacy report contract.
    if cfg.enable_concept_examples_qc:
        assert qc_report is not None
        if clustered:
            install_clustered_bytes_once(
                json.dumps(
                    asdict(qc_report),
                    indent=2,
                    ensure_ascii=False,
                ).encode("utf-8"),
                qc_report_path,
                surface="Stage 0c concept-examples QC report",
            )
        else:
            _si._write_qc_report(output_paths, qc_report)
        qc_report_payload = _np._validate_concept_examples_qc_report_json(
            qc_report_path,
            expected_candidates=live_candidates,
            expected_judge_min_score=cfg.activation_example_judge_min_score,
        )

    # Surface the same exhaustion error the legacy single-rank loop did.
    # When n_candidates was passed in we report it for byte-identical
    # error message formatting; otherwise we fall back to the count of
    # accepts + rejects (sufficient diagnostic info for direct callers).
    if len(accepted_concepts) < cfg.max_concepts:
        nc = (
            n_candidates
            if n_candidates is not None
            else len(accepted_concepts) + len(rejected_concepts_payload)
        )
        if stopped_by_qc_cap:
            raise AxBenchBaseError(
                "Stage 0c stopped because "
                "concept_examples_qc_target_surviving="
                f"{cfg.concept_examples_qc_target_surviving} was reached "
                f"before accepting cfg.max_concepts={cfg.max_concepts} "
                f"concepts (accepted {len(accepted_concepts)} from "
                f"{nc} candidates; need {cfg.max_concepts}). "
                "Raise concept_examples_qc_target_surviving or lower "
                "max_concepts for this capped dev/profiling run."
            )
        raise AxBenchBaseError(
            f"Stage 0c exhausted candidate_concepts.json without accepting "
            f"cfg.max_concepts={cfg.max_concepts} concepts "
            f"(accepted {len(accepted_concepts)} from {nc} candidates; "
            f"need {cfg.max_concepts}). Consider lowering "
            "verification_percentile_threshold, raising min_corpus_count to "
            "change the candidate pool, or the base SAE LM is genuinely too "
            "weak for this feature set."
        )

    # ---- Write concepts.json ----
    # Atomic write so a crash mid-write does not leave a partial JSON
    # for the resume-short-circuit check to misinterpret as complete.
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_bytes = json.dumps(
        [asdict(c) for c in accepted_concepts],
        indent=2,
        ensure_ascii=False,
    ).encode("utf-8")
    if clustered:
        install_clustered_bytes_once(
            concepts_bytes,
            concepts_path,
            surface="Stage 0c final concepts",
        )
    else:
        atomic_write_bytes(concepts_bytes, concepts_path)
    logger.info(
        "Wrote {n} final concepts to {p}",
        n=len(accepted_concepts), p=str(concepts_path),
    )

    persisted_concepts = load_and_validate_concepts(
        concepts_path,
        n_sae_features,
    )
    if persisted_concepts != accepted_concepts:
        raise AxBenchBaseError(
            f"Persisted Stage 0c concepts at {concepts_path} differ from the "
            "validated merge result; retaining per-rank checkpoints"
        )
    persisted_rejected = _np._validate_rejected_concepts_json(rejected_path)
    if persisted_rejected != rejected_concepts_payload:
        raise AxBenchBaseError(
            f"Persisted Stage 0c rejections at {rejected_path} differ from "
            "the validated merge result; retaining per-rank checkpoints"
        )
    if cfg.enable_concept_examples_qc:
        persisted_qc_report = _np._validate_concept_examples_qc_report_json(
            qc_report_path,
            expected_candidates=live_candidates,
            expected_judge_min_score=cfg.activation_example_judge_min_score,
        )
        if persisted_qc_report != qc_report_payload:
            raise AxBenchBaseError(
                f"Persisted Stage 0c QC report at {qc_report_path} differs "
                "from the validated merge result; retaining per-rank "
                "checkpoints"
            )
    else:
        if qc_report_path.exists():
            raise AxBenchBaseError(
                f"Stage 0c QC is disabled, but stale report state reappeared "
                f"at {qc_report_path}; retaining per-rank checkpoints"
            )
        persisted_qc_report = None

    # ---- Install negative_mode_prefixes.parquet last ----
    # This file is the completed-set commit marker. All in-memory row semantics
    # and persisted JSON sidecars have already validated, so no known-invalid
    # state can become visible to downstream stages through this marker.
    assert canonical_table is not None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if clustered:
        write_clustered_parquet_once(
            canonical_table,
            out_path,
            surface="Stage 0c negative-mode prefixes commit marker",
        )
    else:
        atomic_write_parquet(canonical_table, out_path)
    logger.info(
        "Wrote {n} verified negative-mode prefixes across {c} concepts to {p}",
        n=len(rows_payload), c=len(accepted_concepts), p=str(out_path),
    )

    # The canonical parquet is the completed-artifact commit marker. Validate
    # the exact bytes just persisted before deleting the only resumable rank
    # state: schema/tokenizer/extractor metadata, every row invariant, pool
    # identity, and the semantic identities binding both JSON sidecars plus
    # the full agreed Stage 0c fingerprint. A writer regression or corrupt
    # checkpoint row must leave the rank checkpoints intact for investigation.
    try:
        _np._validate_negative_prefixes_parquet(
            out_path,
            concepts=accepted_concepts,
            cfg=cfg,
        )
        persisted_schema = pq.read_schema(out_path)
        _si.validate_negative_prefixes_candidate_pool_identity_metadata(
            out_path,
            consumer="Stage 0c merge post-write validation",
            expected_candidate_pool_identity=pool_identity,
            schema=persisted_schema,
        )
        _si.validate_negative_prefixes_completed_artifact_identities(
            out_path,
            consumer="Stage 0c merge post-write validation",
            expected_concepts=persisted_concepts,
            expected_rejected_concepts=persisted_rejected,
            expected_concept_examples_qc_report=persisted_qc_report,
            expected_stage0c_fingerprint=merged_fingerprint,
            schema=persisted_schema,
        )
    except Exception:
        out_path.unlink(missing_ok=True)
        raise

    # ---- Cleanup per-rank partials ----
    _cleanup_per_rank_partial_checkpoints(cfg)

    return out_path


def dispatch_prepare_negative_prefixes(
    cfg: AxBenchBaseConfig,
    concepts_path: Path,
    base_model: Any,
    base_sae: Any,
    target_model_meta: TargetModelMetadata,
    *,
    n_sae_features: int,
    response_cache: SynthResponseCache | None,
    rank: int = 0,
    world_size: int = 1,
    cluster_contract: ClusterFeatureContract | None = None,
) -> Path:
    """Run Stage 0c verification and merge its per-rank results.

    Pre-loop setup (load candidate_concepts, parquets, feature_shards,
    percentile_table, candidate prefix index, base_context_window) runs
    on every rank — each rank needs the same view of the candidate pool
    to apply its modulo partition deterministically.

    Each rank then enters :func:`run_single_rank` and writes its per-rank
    checkpoint pair. After all ranks return, rank 0 calls
    :func:`run_merge` to assemble the final canonical Stage 0c artifacts.

    The single-GPU path (``world_size == 1``) skips the
    :func:`torch.distributed.barrier` call entirely so callers do not
    need an initialized process group.
    """
    if cfg.run_tag is None:
        raise AxBenchBaseError(
            "cfg.run_tag must be set before calling Stage 0c dispatcher"
        )
    cluster_enabled = clustered_source_enabled(cfg)
    if cluster_enabled != (cluster_contract is not None):
        raise AxBenchBaseError(
            "Clustered Stage 0c requires config and validated feature-contract state "
            "to be present together"
        )
    # PR #739 FU-D-2: resolve the candidate identity ONCE here, inside the
    # existing collective vote, and derive the clustered fingerprint fields
    # from the RETURNED identity -- not a fresh disk read. Every hot/
    # per-rank consumer downstream (``run_single_rank`` -> ``_build_
    # payload`` / ``_try_resume_per_rank`` -> ``_stage0c_internals.
    # _fingerprint_with_pool``) receives this dict via explicit parameter
    # threading (no module-global memo cache) and never calls
    # ``clustered_stage0c_fingerprint_fields``/``load_cluster_candidate_
    # identity`` again for the rest of this rank's run. Unclustered runs
    # thread an explicit ``{}`` so ``_stage0c_extractor_identity`` never
    # even attempts the (harmlessly empty) fresh read -- byte-identical
    # U-invariant, zero reads.
    precomputed_clustered_fields: dict[str, str] = {}
    if cluster_contract is not None:
        # Codex cross-review fix batch (FU-D-stage0c, Bug 1 HIGH + Bug 2
        # MEDIUM): this used to be a bare ``load_cluster_candidate_
        # identity`` call with no live-statistics re-check and no
        # collective failure-agreement -- see ``cluster_artifacts.
        # validate_cluster_candidate_identity_with_collective_agreement``'s
        # own docstring for the full rationale (stale-NFS-view / genuine
        # statistics-swap deadlock this closes).
        validated_identity = validate_cluster_candidate_identity_with_collective_agreement(
            cfg,
            cluster_contract,
            candidate_path=concepts_path,
            n_sae_features=n_sae_features,
            world_size=world_size,
        )
        precomputed_clustered_fields = _clustered_fingerprint_fields_from_identity(
            validated_identity
        )
    if (
        world_size > 1
        and cfg.enable_concept_examples_qc
        and cfg.concept_examples_qc_target_surviving is not None
    ):
        raise AxBenchBaseError(
            "concept_examples_qc_target_surviving is supported only at "
            "world_size=1. Multi-rank Stage 0c cannot enforce that global "
            "live-QC survivor cap without topology-dependent overrun; set it "
            "to null or launch a single rank."
        )

    # ---- Pre-loop setup (every rank) ----
    base_context_window = _si._base_context_window(base_model)
    if base_context_window <= cfg.verification_length:
        raise AxBenchBaseError(
            f"base Gemma n_ctx ({base_context_window}) must be greater "
            f"than cfg.verification_length ({cfg.verification_length}) "
            "so Stage 0c has room for at least one prefix token before "
            "verification generation"
        )
    logger.info(
        "[stage0c] loading candidate concepts from {path}",
        path=str(concepts_path),
    )
    candidate_concepts = _load_candidate_pool_with_collective_identity_agreement(
        concepts_path,
        n_sae_features=n_sae_features,
        world_size=world_size,
    )
    logger.info(
        "[stage0c] loaded {n} candidate concepts", n=len(candidate_concepts),
    )
    if not candidate_concepts:
        raise AxBenchBaseError(
            f"Candidate concept list at {concepts_path} is empty; "
            "Stage 0a must run first"
        )
    candidate_sources = {concept.sae_source for concept in candidate_concepts}
    if candidate_sources != {cfg.neuronpedia_source}:
        raise AxBenchBaseError(
            f"Candidate concept list at {concepts_path} has SAE source(s) "
            f"{sorted(candidate_sources)!r}, but the current config uses "
            f"neuronpedia_source={cfg.neuronpedia_source!r}. Refusing to run "
            "Stage 0c with a candidate pool from a different extractor "
            "identity. Use a new `run_tag`, or rerun Stage 0a before Stage 0c."
        )
    runtime_sae_layer = target_model_meta.sae_layer
    if (
        not isinstance(runtime_sae_layer, int)
        or isinstance(runtime_sae_layer, bool)
        or runtime_sae_layer < 0
    ):
        raise AxBenchBaseError(
            "Stage 0c target_model_meta.sae_layer must be a non-negative "
            f"int, got {runtime_sae_layer!r}"
        )
    candidate_layers = {concept.layer for concept in candidate_concepts}
    if candidate_layers != {runtime_sae_layer}:
        raise AxBenchBaseError(
            f"Candidate concept list at {concepts_path} has SAE layer(s) "
            f"{sorted(candidate_layers)!r}, but Stage 0c runtime metadata "
            f"requires sae_layer={runtime_sae_layer!r}. Refusing to "
            "measure verification activations at a layer different from the "
            "candidate pool."
        )

    logger.info(
        "[stage0c] listing parquets for feature_dir={fd}", fd=str(cfg.feature_dir),
    )
    parquets = _si._list_parquets_for_feature_dir(cfg)
    logger.info("[stage0c] listed {n} parquet shard(s)", n=len(parquets))
    logger.info(
        "[stage0c] globbing feature shards for feature_dir={fd}",
        fd=str(cfg.feature_dir),
    )
    feature_shards = glob_features(Path(cfg.feature_dir))
    logger.info("[stage0c] found {n} feature shard(s)", n=len(feature_shards))
    logger.info(
        "[stage0c] building percentile table from {path} (n_sae_features={n})",
        path=str(cfg.feature_statistics_path), n=n_sae_features,
    )
    # Codex cross-review fix batch (FU-D-stage0c, Bug 1 HIGH follow-up +
    # Codex RE-CHECK MEDIUM): the table CONSUMED downstream (``_verify_
    # concept`` below and merge) must be provably built from the statistics
    # identity pinned by the clustered feature contract -- not from whatever
    # bytes happen to be at ``cfg.feature_statistics_path`` at this later
    # moment. A fresh clustered run (no partial checkpoint) never reaches
    # ``_try_resume_per_rank``'s per-rank identity guard, so before the HIGH
    # fix the collective candidate-pool check above was the ONLY live-
    # statistics check, leaving an A->B swap in the window before this build
    # undetected. The fused helper reads the statistics ONCE, verifies the
    # identity against the pinned snapshot, and builds the table FROM those
    # same verified bytes (no separate reopen -> no hash-then-reopen
    # TOCTOU) -- but that fused read+verify+build is itself PER-RANK LOCAL,
    # so a rank that observes a stale/swapped view here would previously
    # raise alone while its peer(s) proceeded into the dispatcher's next
    # real collective and blocked forever: the exact one-rank-raises/peer-
    # enters-collective deadlock ``validate_cluster_candidate_identity_
    # with_collective_agreement`` (above) exists to prevent, reintroduced
    # at this new call site. ``build_verified_percentile_table_with_
    # collective_agreement`` closes that gap by wrapping the fused call so
    # every rank agrees on success/failure before any rank proceeds --
    # raises collectively, never returns a table built from an unverified
    # or rank-divergent view. It returns ``None`` for unclustered runs
    # (never entered here -- guarded by ``cluster_contract is not None``),
    # which fall back to the byte-identical cached builder (U-invariant).
    percentile_table: t.Tensor | None = None
    if cluster_contract is not None:
        percentile_table = build_verified_percentile_table_with_collective_agreement(
            cfg,
            cluster_contract,
            n_sae_features=n_sae_features,
            surface="Clustered Stage 0c percentile-table build",
            world_size=world_size,
        )
    if percentile_table is None:
        percentile_table = _extraction.build_percentile_table_from_stats(
            cfg.feature_statistics_path, n_sae_features=n_sae_features,
        )
    logger.info("[stage0c] percentile table built")
    candidate_feature_ids = {c.feature_index for c in candidate_concepts}
    if cluster_contract is not None:
        for feature_id in candidate_feature_ids:
            cluster_contract.target_for_representative(feature_id)
        candidate_feature_ids = {
            member_id
            for feature_id in candidate_feature_ids
            for member_id in cluster_contract.target_for_representative(
                feature_id
            ).allowed_member_ids
        }
    logger.info(
        "[stage0c] initialising candidate prefix index "
        "(candidate_features={n})",
        n=len(candidate_feature_ids),
    )
    candidate_index_cfg = (
        cfg
        if cluster_contract is None or cfg.verification_candidate_scan_limit is None
        else cfg.replace(verification_candidate_scan_limit=None)
    )
    cand_index = _si._CandidatePrefixIndex(
        candidate_index_cfg, parquets, feature_shards,
        candidate_feature_ids=candidate_feature_ids,
        base_context_window=base_context_window,
    )
    candidate_lookup: _si._CandidatePrefixLookup = cand_index
    if cluster_contract is not None:
        candidate_lookup = _ClusterCandidatePrefixIndex(
            cand_index,
            cluster_contract,
            candidate_scan_limit=cfg.verification_candidate_scan_limit,
        )

    qc_client = (
        _concept_list._build_concept_examples_qc_client(
            cfg, response_cache=response_cache,
        )
        if cfg.enable_concept_examples_qc else None
    )
    verification_api_generator: Stage0cApiGenerator | None = None
    if cfg.verification_api_model is not None:
        if response_cache is None:
            raise AxBenchBaseError(
                "Stage 0c API verification requires an open synth response "
                "cache; call prepare_negative_prefixes so it can manage the "
                "cache lifetime"
            )
        verification_api_generator = Stage0cApiGenerator(
            cfg,
            response_cache=response_cache,
        )

    # ---- Per-rank stripe ----
    rank_kwargs: dict[str, Any] = {}
    if verification_api_generator is not None:
        rank_kwargs["verification_api_generator"] = verification_api_generator
    if cluster_contract is not None:
        rank_kwargs["cluster_contract"] = cluster_contract
    run_single_rank(
        cfg, candidate_concepts,
        rank=rank, world_size=world_size,
        base_model=base_model, base_sae=base_sae,
        target_model_meta=target_model_meta,
        qc_client=qc_client,
        parquets=parquets, feature_shards=feature_shards,
        percentile_table=percentile_table,
        candidate_index=candidate_lookup,
        base_context_window=base_context_window,
        n_sae_features=n_sae_features,
        precomputed_clustered_fields=precomputed_clustered_fields,
        **rank_kwargs,
    )

    # ---- Merge ----
    if world_size > 1:
        # All ranks rendezvous so the merge sees every rank's per-rank
        # commit. ``dist.barrier()`` is backend-agnostic (works for both
        # NCCL and GLOO); the rank-failure hang is bounded by the
        # process-group collective timeout — PyTorch's backend default
        # (~10 min NCCL / 30 min GLOO) is preserved unless
        # ``NCCL_TIMEOUT_SEC`` is set (``compute_init`` does NOT change
        # the global default). The Stage 0c launcher
        # (``scripts/eval/axbench_base_eval``) defaults
        # ``NCCL_TIMEOUT_SEC`` to 1800s, so for Stage 0c the bound is
        # 1800s rather than a per-call timeout.
        # ``dist.monitored_barrier`` is intentionally NOT used because
        # PyTorch raises ``ValueError("monitored_barrier is only
        # implemented for GLOO backend.")`` whenever the active backend
        # is NCCL — which is the production launcher's choice
        # (``nanochat/common.py`` initializes the PG with
        # ``backend="nccl"`` for any CUDA run). torchrun's launcher
        # also kills the entire job on first rank failure, so in
        # practice the hang is bounded by the launcher's exit hook.
        dist.barrier()
        merge_result: list[dict[str, Any] | None] = [None]
        rank0_merge_error: Exception | None = None
        if rank == 0:
            try:
                merged_path = run_merge(
                    cfg,
                    world_size=world_size,
                    n_sae_features=n_sae_features,
                    n_candidates=len(candidate_concepts),
                    candidate_concepts_path=concepts_path,
                )
                merge_result[0] = {
                    "ok": True,
                    "path": str(merged_path),
                }
            except Exception as exc:
                rank0_merge_error = exc
                merge_result[0] = {
                    "ok": False,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
        dist.broadcast_object_list(merge_result, src=0)
        result = merge_result[0]
        if not isinstance(result, dict) or not result.get("ok"):
            if rank == 0 and rank0_merge_error is not None:
                raise rank0_merge_error
            error_type = (
                result.get("error_type", "UnknownError")
                if isinstance(result, dict)
                else "InvalidMergeBroadcast"
            )
            error = (
                result.get("error", repr(result))
                if isinstance(result, dict)
                else repr(result)
            )
            raise AxBenchBaseError(
                f"Stage 0c rank-0 merge failed ({error_type}): {error}"
            )
        path_value = result.get("path")
        if not isinstance(path_value, str):
            raise AxBenchBaseError(
                "Stage 0c rank-0 merge broadcast omitted its canonical path"
            )
        return Path(path_value)

    return run_merge(
        cfg,
        world_size=world_size,
        n_sae_features=n_sae_features,
        n_candidates=len(candidate_concepts),
        candidate_concepts_path=concepts_path,
    )


__all__ = [
    "_DEFAULT_BLOCK_SIZE",
    "_block_size",
    "_cleanup_per_rank_partial_checkpoints",
    "_load_candidate_pool_with_collective_identity_agreement",
    "dispatch_prepare_negative_prefixes",
    "run_merge",
    "run_single_rank",
]
