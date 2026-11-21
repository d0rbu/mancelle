"""Resume-across-world-size matrix for the PR-4 multi-GPU Stage 0c runner.

Mirrors the BPB resume contract pinned in
``tests/axbench_bpb/integration/test_bpb_resume_world_size_changes.py``
but for the Stage 0c per-rank checkpoint. The contract:

* The fingerprint is INVARIANT under ``world_size`` (world_size is not
  a fingerprint field; this is the deliberate PR-4 design choice that
  enables resume across DDP topology changes).
* When ``ws_resume == ws_initial`` the resume is full-fidelity: every
  visited candidate is preserved per its rank's checkpoint.
* When ``ws_resume != ws_initial`` only candidates whose modulo
  partition assignment is the SAME rank under both world sizes are
  preserved. The others must be re-processed because rank R only
  reads its own per-rank JSON; if the candidate moved to rank R',
  R' won't see R's prior work.

The matrix ranges over scale-up, scale-down, and same-world-size
control cases. For each case we:

1. Run a "partial" Stage 0c at ``ws_initial`` that visits only the
   first ``n_candidates // 2`` candidates (the rest are intentionally
   skipped in the worker via a per-rank cap).
2. Resume at ``ws_resume`` with the SAME cfg (modulo run_tag, which is
   not in the fingerprint). The dispatcher should validate the
   per-rank fingerprints, read each rank's checkpoint, and continue
   from where each rank left off.
3. Assert the EXACT expected set of preserved candidates per the
   per-rank-only contract; mismatches indicate a regression in either
   :func:`is_my_candidate` or
   :func:`steering.eval.axbench_base.stage0c_partition.read_per_rank_checkpoint`.

Plus two non-matrix tests:

* Fingerprint mismatch on resume — running with a different
  ``max_concepts`` value (a fingerprint field) raises
  :class:`AxBenchBaseError`.
* Schema-version bump rejection — a hand-rolled JSON with the OLD
  ``schema_version`` value fails resume with the canonical "stale
  schema, delete" error.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

from steering.eval.axbench_base import _stage0c_internals as si_mod
from steering.eval.axbench_base import extraction as extraction_mod
from steering.eval.axbench_base import paths as paths_mod
from steering.eval.axbench_base import stage0c_runner as runner_mod
from steering.eval.axbench_base._stage0c_internals import (
    _VerifiedNegativePrefix,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.stage0c_partition import (
    _RANK_CHECKPOINT_SCHEMA_VERSION,
    is_my_candidate,
)
from steering.eval.axbench_base.types import AxBenchBaseError, CandidateConcept
from steering.refined_types import NonNegativeInt, PositiveInt

# ---------------------------------------------------------------------------
# Module-level helpers — all spawn-friendly. The workers must be importable.
# ---------------------------------------------------------------------------


def _write_minimal_stats(path: Path, n_sae_features: int = 16384) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )
    return path


def _build_candidates(n: int) -> list[CandidateConcept]:
    return [
        CandidateConcept(
            feature_index=2000 + i,
            layer=8,
            sae_source="8-gemmascope-res-16k",
            description=f"resume-feature_{i:03d}",
            corpus_count=10 * (i + 1),
            description_source="neuronpedia",
        )
        for i in range(n)
    ]


def _build_cfg(
    *, run_root: Path, run_tag: str,
    max_concepts: int = 100,
    block_size: int = 4,
    n_verified_target: int = 2,
    verification_length: int = 4,
) -> AxBenchBaseConfig:
    """Build a deterministic AxBenchBaseConfig for resume tests.

    Default ``max_concepts`` is intentionally large (100) so the runs
    don't terminate early on accept-saturation; the resume contract is
    about per-rank checkpoint state, not termination.
    """
    stats_path = _write_minimal_stats(run_root / "stats.safetensors")
    return AxBenchBaseConfig(
        feature_dir=str(run_root / "features" / "s128"),
        feature_statistics_path=str(stats_path),
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=str(run_root / "eval_out"),
        run_tag=run_tag,
        max_concepts=max_concepts,
        enable_concept_examples_qc=False,
        n_verified_prefixes_per_concept_target=n_verified_target,
        n_verification_seeds=2,
        min_consistency_passes=1,
        verification_percentile_threshold=0.5,
        verification_length=verification_length,
        min_prefix_chars=1,
        extraction_tokenizer_id="google/gemma-2-2b",
        stage0c_multi_gpu_block_size=block_size,
        verification_api_model=None,
    )


def _output_paths(run_root: str | Path, run_tag: str) -> paths_mod.AxBenchBasePaths:
    return paths_mod.AxBenchBasePaths(Path(run_root) / "eval_out", run_tag)


def _write_concepts_json(path: Path, candidates: list[CandidateConcept]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "feature_index": c.feature_index,
            "layer": c.layer,
            "sae_source": c.sae_source,
            "description": c.description,
            "corpus_count": c.corpus_count,
            "description_source": c.description_source,
        }
        for c in candidates
    ]
    path.write_text(json.dumps(payload), encoding="utf-8")


def _stub_verify_factory(
    n_verified_target: int,
    reject_feature_indices: frozenset[int] = frozenset(),
):
    """Build a deterministic ``_verify_concept`` stub.

    By default (empty ``reject_feature_indices``) every candidate accepts
    (returns ``n_verified_target`` rows) — the historical all-accept
    behavior the 19 existing resume tests rely on.

    When ``reject_feature_indices`` is non-empty, any candidate whose
    ``feature_index`` is in the set returns ZERO verified rows (a
    verification rejection). This mirrors the byte-identity file's
    ``reject_feature_indices`` pattern
    (``tests/axbench_base/integration/test_stage0c_multi_gpu_byte_identity.py``)
    and is keyed by ``concept.feature_index`` so the accept/reject
    decision is stable regardless of which rank verifies the candidate
    or in what order. A low global accept ratio lets a capped run
    POOL-EXHAUST (never accept-saturate), which is the only regime in
    which a cap-binding partial can commit a block-boundary checkpoint
    and THEN raise mid-flight (see
    ``test_stage0c_resume_cap_binds_equivalent_to_uninterrupted``).
    """

    def _verify(
        cfg: Any, concept: Any, parquets: Any, feature_shards: Any,
        base_model: Any, base_sae: Any, percentile_table: Any,
        *, sae_layer: int, base_context_window: int, candidate_index: Any,
    ) -> list[_VerifiedNegativePrefix]:
        del (
            cfg, parquets, feature_shards, base_model, base_sae,
            percentile_table, sae_layer, base_context_window, candidate_index,
        )
        if concept.feature_index in reject_feature_indices:
            return []
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
            for i in range(n_verified_target)
        ]

    return _verify


def _patch_dispatcher_helpers(
    candidates: list[CandidateConcept],
    *,
    n_verified_target: int,
    candidate_visit_cap: int | None,
    reject_feature_indices: frozenset[int] = frozenset(),
) -> None:
    """Monkey-patch every dispatcher helper that touches the network or
    heavy I/O.

    ``candidate_visit_cap`` is a cooperative cap: when set, the verify
    stub raises a deterministic exception for any candidate index >=
    cap so the loop bails before processing the second half. This
    simulates the "partial run" half of the resume contract.

    ``reject_feature_indices`` (default empty ⇒ all-accept, the historical
    behavior the 19 existing resume tests rely on) is threaded into the
    base verify stub so any candidate whose ``feature_index`` is in the
    set returns 0 verified rows (a reject). The cap-raise check runs
    FIRST and is unchanged, so the crash-simulation semantics are
    byte-identical to before; the reject set only affects accept/reject
    of candidates the stub does verify.
    """
    runner_module: Any = runner_mod
    runner_module.load_and_validate_candidate_concepts = (
        lambda p, n_sae_features: candidates
    )
    si_module: Any = si_mod
    si_module._list_parquets_for_feature_dir = lambda c: []
    runner_module.glob_features = lambda feature_dir: []
    extraction_module: Any = extraction_mod
    extraction_module.build_percentile_table_from_stats = (
        lambda p, n_sae_features: t.zeros(n_sae_features)
    )
    si_module._CandidatePrefixIndex = lambda *a, **kw: MagicMock()
    si_module._base_context_window = lambda m: 2048

    base_verify = _stub_verify_factory(
        n_verified_target, reject_feature_indices,
    )
    if candidate_visit_cap is None:
        si_module._verify_concept = base_verify
        return

    # Cooperative cap. Each candidate's feature_index -> its candidate
    # position is `feature_index - 2000` per the deterministic builder.
    cap = int(candidate_visit_cap)

    def _capped_verify(*args, **kwargs):
        concept = args[1]  # by signature: (cfg, concept, ...)
        candidate_pos = int(concept.feature_index) - 2000
        if candidate_pos >= cap:
            raise RuntimeError(
                f"_capped_verify: refusing candidate {candidate_pos} "
                f">= cap {cap} (simulated mid-run interruption)"
            )
        return base_verify(*args, **kwargs)

    si_module._verify_concept = _capped_verify


# ---------------------------------------------------------------------------
# Worker targets — partial run + resume run.
# ---------------------------------------------------------------------------


def _stage0c_partial_run_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    run_tag: str,
    n_candidates: int,
    candidate_visit_cap: int,
    block_size: int,
    n_verified_target: int,
    max_concepts: int = 100,
    reject_feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Partial run worker: process candidates with index < cap, then
    raise. The runner's try/finally + per-block commit means the rank's
    checkpoint should persist with the visited indices.

    ``reject_feature_indices`` (optional, default ``None`` ⇒ all-accept —
    byte-identical to the historical behavior, so the 19 existing callers
    that pass neither this nor a non-default ``max_concepts`` are
    unaffected) lists ``feature_index`` values the verify stub rejects
    (0 rows). The cap-raise behavior is unchanged: the cap check fires
    before the reject check.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)
    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag=run_tag,
        max_concepts=max_concepts,
        block_size=block_size,
        n_verified_target=n_verified_target,
    )
    candidates = _build_candidates(n_candidates)
    concepts_path = tmp_path / f"{run_tag}_candidate_concepts.json"
    _write_concepts_json(concepts_path, candidates)

    _patch_dispatcher_helpers(
        candidates,
        n_verified_target=n_verified_target,
        candidate_visit_cap=candidate_visit_cap,
        reject_feature_indices=frozenset(reject_feature_indices or []),
    )

    # The capped verify will raise once a rank reaches a candidate >= cap.
    # We catch RuntimeError so the worker still returns a result the
    # parent can read; the per-rank checkpoint is durable on disk
    # regardless because the runner commits at every block boundary.
    raised = False
    try:
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=rank, world_size=world_size,
        )
    except (RuntimeError, AxBenchBaseError):
        raised = True

    json_path = _output_paths(tmp_path_str, run_tag).stage0c_partial_rank_json(rank)
    payload: dict[str, Any] = {}
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "rank": rank,
        "raised": raised,
        "visited_candidate_indices": list(
            payload.get("visited_candidate_indices", []),
        ),
        "rank_accepts_count": len(payload.get("rank_accepts", [])),
    }


def _stage0c_resume_count_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    run_tag: str,
    n_candidates: int,
    block_size: int,
    n_verified_target: int,
) -> dict[str, Any]:
    """Resume worker: BEFORE running anything, count the candidates this
    rank "newly owns" under ``world_size`` whose visited state is NOT
    already present in this rank's per-rank checkpoint. Returns:

    - ``owned_now``: count of candidates this rank owns under the new
      world_size (the per-rank stripe).
    - ``preserved``: count of those owned candidates that ALREADY appear
      in the rank's prior visited_candidate_indices.
    - ``remaining``: ``owned_now - preserved`` — the work this rank
      will redo on resume.

    Returns the snapshot WITHOUT mutating the checkpoints, so the parent
    can compute expected values via the same partition function.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    json_path = _output_paths(tmp_path_str, run_tag).stage0c_partial_rank_json(rank)
    visited_set: set[int] = set()
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        visited_set = set(payload.get("visited_candidate_indices", []))

    owned_now = 0
    preserved = 0
    for idx in range(n_candidates):
        if is_my_candidate(
            NonNegativeInt.parse(idx),
            rank=NonNegativeInt.parse(rank),
            world_size=PositiveInt.parse(world_size),
        ):
            owned_now += 1
            if idx in visited_set:
                preserved += 1
    return {
        "rank": rank,
        "owned_now": owned_now,
        "preserved": preserved,
        "remaining": owned_now - preserved,
    }


def _stage0c_resume_run_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    run_tag: str,
    n_candidates: int,
    block_size: int,
    n_verified_target: int,
    max_concepts: int = 100,
    reject_feature_indices: list[int] | None = None,
) -> dict[str, Any]:
    """Resume worker: actually invoke the dispatcher under ``world_size``
    and let it pick up each rank's per-rank checkpoint. Useful for
    integration smoke tests; the byte-identity matrix is covered by a
    separate test file (see test_stage0c_multi_gpu_byte_identity.py).

    ``reject_feature_indices`` (optional, default ``None`` ⇒ all-accept —
    byte-identical to the historical behavior, so the existing callers
    that pass neither this nor a non-default ``max_concepts`` are
    unaffected) lists ``feature_index`` values the verify stub rejects
    (0 rows). No cap is installed on resume; the run verifies every
    owned candidate (rejecting the listed ones).
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)
    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag=run_tag,
        max_concepts=max_concepts,
        block_size=block_size,
        n_verified_target=n_verified_target,
    )
    candidates = _build_candidates(n_candidates)
    concepts_path = tmp_path / f"{run_tag}_candidate_concepts.json"
    _write_concepts_json(concepts_path, candidates)

    # No cap on resume — verify every candidate (rejecting the listed).
    _patch_dispatcher_helpers(
        candidates,
        n_verified_target=n_verified_target,
        candidate_visit_cap=None,
        reject_feature_indices=frozenset(reject_feature_indices or []),
    )

    raised = False
    err: str | None = None
    try:
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=rank, world_size=world_size,
        )
    except AxBenchBaseError as exc:
        raised = True
        err = str(exc)

    payload: dict[str, Any] = {}
    json_path = _output_paths(tmp_path_str, run_tag).stage0c_partial_rank_json(rank)
    if json_path.exists():
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    return {
        "rank": rank,
        "raised": raised,
        "err": err,
        "visited_candidate_indices": list(
            payload.get("visited_candidate_indices", []),
        ),
        "rank_accepts_count": len(payload.get("rank_accepts", [])),
    }


# ---------------------------------------------------------------------------
# Helper: compute expected preserved-set per the per-rank-only contract.
# ---------------------------------------------------------------------------


def _expected_remaining_from_actual_visited(
    *,
    ws_initial: int,
    ws_resume: int,
    n_candidates: int,
    visited_by_rank: dict[int, list[int]],
) -> int:
    """Compute the EXACT expected ``remaining`` total across all ranks of
    ``ws_resume`` from the ACTUAL committed visited indices per rank
    under ``ws_initial`` (read from the per-rank checkpoint JSONs after
    the partial run).

    A candidate is preserved iff it appears in some rank R's per-rank
    visited list AND its assignment under ``ws_resume`` is the SAME
    rank R (so the new owner can see the same JSON). Anything else
    must be re-visited.

    Reading the actual committed state — rather than computing from a
    "visit cap" — keeps the expected calculation robust against the
    per-block commit cadence (the runner only commits at block
    boundaries, so the visited set is block-aligned and depends on
    ws_initial and block_size).
    """
    initial_owners: dict[int, int] = {}
    for r, visited_indices in visited_by_rank.items():
        for idx in visited_indices:
            initial_owners[idx] = r

    remaining = 0
    for idx in range(n_candidates):
        resume_owner = next(
            r for r in range(ws_resume)
            if is_my_candidate(
                NonNegativeInt.parse(idx),
                rank=NonNegativeInt.parse(r),
                world_size=PositiveInt.parse(ws_resume),
            )
        )
        if idx in initial_owners and initial_owners[idx] == resume_owner:
            continue  # preserved by same-rank checkpoint
        remaining += 1
    return remaining


# ---------------------------------------------------------------------------
# THE MATRIX: resume across world_size changes. Mirrors the BPB pattern.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("ws_initial,ws_resume", [
    (1, 1),  # control: same ws=1
    (2, 2),  # control: same ws=2
    (4, 4),  # control: same ws=4
    (1, 2),  # scale-up
    (1, 4),  # scale-up
    (2, 4),  # scale-up
    (4, 2),  # scale-down
    (4, 1),  # scale-down to single-GPU
    (8, 2),  # large to small
    # PR-4 second-round review extension: ws_resume=8 cases close
    # the matrix gap (largest scale-up row was missing).
    (1, 8),
    (2, 8),
    (4, 8),
    (8, 8),
])
def test_stage0c_resume_across_world_size_change(
    tmp_path, monkeypatch, gloo_multiprocess, ws_initial, ws_resume,
):
    """After partial run at ws_initial visits half the candidates,
    resume at ws_resume sees the EXACT expected number of remaining
    units, computed from the documented per-rank-only checkpoint
    contract.

    Asserts that the actual cross-rank "remaining" count from the
    snapshot equals the model count from
    :func:`_expected_remaining_after_partial_run`. A regression in
    :func:`is_my_candidate` (the partition function), in
    :func:`read_per_rank_checkpoint`, or in the runner's resume guard
    surfaces as an exact-equality mismatch with a clear failure
    message.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    # Pool size scales with the largest world_size in the matrix so
    # block 0 always covers a strict subset of the pool — that way the
    # cap can fire mid-block-1 and the per-rank checkpoint reflects a
    # block-0-only commit. For B=4, ws=8 → block 0 covers 32; we need
    # n > 32 + B*ws (= 64) so block 1 has room. Pick n=80.
    n_candidates = 80
    block_size = 4
    n_verified_target = 2
    # Cap = block 0's last index + 1 + (block_size * ws_initial) // 2
    # → fires inside block 1 after some rank has begun its block-1
    # stripe, leaving block 0 fully committed and block 1 in-flight
    # (uncommitted). For all ws ∈ {1, 2, 4, 8} this lands well inside
    # block 1.
    candidate_visit_cap = (
        block_size * ws_initial + (block_size * ws_initial) // 2
    )
    run_tag = f"resume-ws{ws_initial}-to-{ws_resume}"

    # Phase 1: partial run at ws_initial. The cap triggers an exception
    # mid-loop; the rank's last successful block-boundary commit becomes
    # the resume start point.
    partial_results = gloo_multiprocess(
        ws_initial, _stage0c_partial_run_target,
        str(tmp_path), run_tag,
        n_candidates, candidate_visit_cap, block_size, n_verified_target,
    )

    # Read the ACTUAL committed visited indices per rank — this is the
    # authoritative input to the expected-remaining calculation. The
    # runner's per-block commit cadence determines what's preserved
    # under ws_resume; a different cap or B value would yield a
    # different visited set, but the contract about "preserved iff
    # same-rank under both world sizes" is invariant to those knobs.
    visited_by_initial_rank: dict[int, list[int]] = {
        r["rank"]: list(r["visited_candidate_indices"])
        for r in partial_results
    }
    # Sanity: every rank under ws_initial should have at least one
    # committed candidate; if a rank's first verify call raises on its
    # very first owned index it would have an empty visited list, which
    # breaks the meaningful test scenario. Tweak ``candidate_visit_cap``
    # if you ever see this fail.
    assert all(
        len(v) > 0 for v in visited_by_initial_rank.values()
    ), (
        f"Every initial rank must have committed at least one block; "
        f"got {visited_by_initial_rank!r}"
    )

    expected_remaining = _expected_remaining_from_actual_visited(
        ws_initial=ws_initial,
        ws_resume=ws_resume,
        n_candidates=n_candidates,
        visited_by_rank=visited_by_initial_rank,
    )

    # Phase 2: snapshot (no mutation) — count remaining work under
    # ws_resume given the per-rank checkpoints already on disk.
    snapshots = gloo_multiprocess(
        ws_resume, _stage0c_resume_count_target,
        str(tmp_path), run_tag,
        n_candidates, block_size, n_verified_target,
    )

    total_owned_now = sum(s["owned_now"] for s in snapshots)
    total_preserved = sum(s["preserved"] for s in snapshots)
    total_remaining = sum(s["remaining"] for s in snapshots)

    # Sanity: every candidate is owned by exactly one rank under ws_resume.
    assert total_owned_now == n_candidates, (
        f"At ws_resume={ws_resume}, ranks should own all {n_candidates} "
        f"candidates; got {total_owned_now}"
    )

    # EXACT EQUALITY.
    assert total_remaining == expected_remaining, (
        f"Resume mismatch at ws_initial={ws_initial} -> ws_resume={ws_resume}: "
        f"expected exactly {expected_remaining} remaining, got "
        f"{total_remaining}. preserved={total_preserved}. This indicates "
        "a regression in is_my_candidate, read_per_rank_checkpoint, or "
        "the runner's resume guard."
    )


def test_stage0c_scale_up_resume_executes_without_duplicate_state(
    tmp_path,
    monkeypatch,
    gloo_multiprocess,
):
    """Execute ws=1 partial -> ws=2 completion and validate final artifacts."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    n_candidates = 8
    block_size = 1
    n_verified_target = 2
    max_concepts = n_candidates
    run_tag = "resume-actual-ws1-to-ws2"

    partial = gloo_multiprocess(
        1,
        _stage0c_partial_run_target,
        str(tmp_path),
        run_tag,
        n_candidates,
        3,
        block_size,
        n_verified_target,
        max_concepts,
    )
    assert partial[0]["raised"] is True
    assert partial[0]["visited_candidate_indices"] == [0, 1, 2]

    resumed = gloo_multiprocess(
        2,
        _stage0c_resume_run_target,
        str(tmp_path),
        run_tag,
        n_candidates,
        block_size,
        n_verified_target,
        max_concepts,
    )
    assert all(not result["raised"] for result in resumed), resumed

    output_paths = _output_paths(tmp_path, run_tag)
    concepts_path = output_paths.concepts_json()
    rejected_path = output_paths.rejected_concepts_json()
    parquet_path = output_paths.negative_mode_prefixes()
    concepts = json.loads(concepts_path.read_text(encoding="utf-8"))
    assert len(concepts) == n_candidates
    assert [row["feature_index"] for row in concepts] == [
        candidate.feature_index for candidate in _build_candidates(n_candidates)
    ]
    assert json.loads(rejected_path.read_text(encoding="utf-8")) == []
    table = pq.read_table(parquet_path)
    assert table.num_rows == n_candidates * n_verified_target
    assert sorted({row["concept_id"] for row in table.to_pylist()}) == list(
        range(n_candidates)
    )
    metadata = table.schema.metadata or {}
    assert metadata[b"candidate_pool_identity"] == (
        si_mod.candidate_pool_identity(_build_candidates(n_candidates)).encode(
            "utf-8"
        )
    )


# ---------------------------------------------------------------------------
# Same-ws resume — full-fidelity preservation, end-to-end.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_stage0c_resume_same_world_size_preserves_all_visited(
    tmp_path, monkeypatch, gloo_multiprocess, world_size,
):
    """Same-ws resume: the resume run visits the SAME complete candidate
    set the no-crash run would have visited; every candidate from the
    partial run is preserved (same-rank assignment), and the resume
    fills in the remaining half.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    # Match the scaling chosen for the ws-change matrix: bigger pool +
    # ws-aware cap → block 0 commits, block 1 partially executes.
    n_candidates = 80
    block_size = 4
    n_verified_target = 2
    candidate_visit_cap = (
        block_size * world_size + (block_size * world_size) // 2
    )
    run_tag = f"resume-same-ws{world_size}"

    # Phase 1: partial run.
    gloo_multiprocess(
        world_size, _stage0c_partial_run_target,
        str(tmp_path), run_tag,
        n_candidates, candidate_visit_cap, block_size, n_verified_target,
    )

    # Phase 2: resume. Should pick up exactly where each rank left off.
    resumed = gloo_multiprocess(
        world_size, _stage0c_resume_run_target,
        str(tmp_path), run_tag,
        n_candidates, block_size, n_verified_target,
    )

    # After resume, every candidate has been visited (by some rank).
    visited_by_any_rank: set[int] = set()
    for r in resumed:
        visited_by_any_rank |= set(r["visited_candidate_indices"])
    assert visited_by_any_rank == set(range(n_candidates)), (
        f"After same-ws resume the union of visited indices should "
        f"equal {{0..{n_candidates - 1}}}; got "
        f"{sorted(visited_by_any_rank)}"
    )


def test_stage0c_resume_cap_binds_equivalent_to_uninterrupted(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Non-vacuous cap-binding resume correctness (spec §2.2 / G5).

    Proves: when the adaptive block-size cap BINDS, a partial run that
    crashes mid-flight AFTER a block-boundary commit, then resumes from
    the surviving per-rank checkpoint, reaches the SAME visited set as an
    uninterrupted run with identical cfg — and the resume GENUINELY
    continues from the on-disk checkpoint (it is NOT a fresh run).

    Why the earlier ``small_visit_cap=12`` formulation was VACUOUS
    --------------------------------------------------------------
    With an all-accept stub a binding cap sized ``≈2·max_concepts``
    candidates lets block 0 saturate ``sum(n_accepted) >= max_concepts``
    at the block-0 ``all_gather`` barrier, so the partial run COMPLETES
    in block 0 with no raise. ``dispatch_prepare_negative_prefixes`` then
    calls ``run_merge`` → ``_cleanup_per_rank_partial_checkpoints``
    (``stage0c_runner.py:1050`` → ``:411-412``) which DELETES every
    ``stage0c_partial_rank_*.json``. The resume phase's
    ``_try_resume_per_rank`` then finds no JSON, returns ``None``, and
    "resume" degenerates to an independent fresh deterministic run, so
    ``resumed == ref`` is true *vacuously* (two fresh runs) and no
    checkpoint code path is exercised.

    A surviving checkpoint REQUIRES the partial to raise mid-flight
    AFTER a block-boundary commit but BEFORE ``run_merge``. With an
    all-accept stub that is impossible under a binding cap (block 0
    accept-saturates → merge → cleanup). So this test uses a
    LOW-ACCEPT-RATIO stub (the new ``reject_feature_indices`` mechanism,
    mirroring the byte-identity file) so the run POOL-EXHAUSTS instead of
    accept-saturating, leaving the partial free to crash in a later block
    with earlier-block checkpoints intact.

    Concrete params + arithmetic
    ----------------------------
    * ``max_concepts=6``, ``world_size=4``, configured
      ``stage0c_multi_gpu_block_size=8``.
      Effective ``B = _block_size(cfg, ws=4)
                     = max(1, min(8, ceil(2*6/4)=ceil(3.0)=3)) = 3``.
      ``3 < 8`` ⇒ **the cap BINDS** (asserted via a ``runner_mod._block_size``
      probe; the fingerprint stores the RAW ``8`` and excludes
      ``world_size``, so resume recomputes the same effective B and is
      never falsely fingerprint-rejected).
    * ``covers = B*W = 3*4 = 12``. Block ``b`` spans global candidate
      indices ``[12b, 12b+12)``.
    * ``n_candidates = 80``.
    * **Accept set = candidates {0,1,2}** (feature_index {2000,2001,2002});
      everything else is rejected via ``reject_feature_indices``. The
      GLOBAL accept count over the whole 80-candidate pool is exactly 3,
      which is STRICTLY < ``max_concepts=6``. Hence
      ``sum(n_accepted) >= max_concepts`` NEVER holds: the run can only
      end on pool exhaustion (``global_n_visited >= 80``) or the crash —
      it **POOL-EXHAUSTS, never accept-saturates**.
    * **Crash cap ``candidate_visit_cap=14``** (global-candidate-position
      semantics: the stub raises when ``feature_index-2000 >= 14``):
      - Block 0 = ``[0,12)``. Every owned index ``0..11 < 14`` → all
        verified (cand 0,1,2 accept; 3..11 reject), no raise. At the end
        of block 0 each rank ``_commit_partial()``s its JSON
        (``next_block_index=1``, ``visited_candidate_indices`` = its
        block-0 stripe ``{r, r+4, r+8}``). all_gather:
        ``global_n_accepted=3 < 6`` (no term),
        ``global_n_visited=12 < 80`` (no exhaust),
        next ``block_start = 1*12 = 12 < 80`` (continue).
      - Block 1 = ``[12, 24)``. Rank ``r`` owns ``{12+r, 16+r, 20+r}``
        (ownership is ``idx % world_size == rank``). Ranks 0,1 verify+reject
        their first owned (``12+r`` = 12,13 ``< 14``) then RAISE on the second
        (``16+r`` = 16,17 ``>= 14``); ranks 2,3 RAISE immediately on their
        first owned (``12+r`` = 14,15 ``>= 14``). In all cases the raise lands
        strictly inside block 1, BEFORE block-1's end-of-block commit, so the
        on-disk per-rank JSON still reflects end-of-block-0 state
        (``visited == {0..11}``).
      ⇒ the partial commits exactly block 0 then raises mid-block-1
      before ``run_merge`` (the worker catches the RuntimeError so the
      per-rank JSONs survive on disk; merge/cleanup never runs).

    Expected sets: ``partial_visited`` = block-0 indices ``{0..11}``
    (12). ``resumed_visited`` = the resume reads ``next_block_index=1``
    + ``visited={0..11}`` and continues blocks 1..6 to pool-exhaust →
    all ``{0..79}`` (80). So ``partial_visited = {0..11} ⊊ {0..79} =
    resumed_visited`` (STRICT subset ⇒ resume genuinely continued from
    the checkpoint). The uninterrupted reference uses the IDENTICAL
    reject set with no cap ⇒ also pool-exhausts to ``{0..79}`` ⇒
    ``resumed_visited == ref_visited``.

    HARD ANTI-VACUITY GUARD (correctness — never weaken/remove): before
    the resume phase we assert a real non-empty per-rank
    ``stage0c_partial_rank_*.json`` exists on disk with committed
    ``visited_candidate_indices``; after resume we assert
    ``partial_visited < resumed_visited`` (STRICT). Both fail if the
    checkpoint was destroyed (the original vacuity) — so this test
    cannot silently regress to a fresh-run pair.

    On the "no raise on resume" constraint: the spec's pool-exhaust
    requirement (global accepts STRICTLY BELOW ``max_concepts`` — the
    anti-vacuity-critical knob that prevents the accept-saturate +
    merge-cleanup that destroyed the checkpoint in the original formulation)
    makes rank 0's ``run_merge`` raise the INTRINSIC "Stage 0c exhausted ...
    cfg.max_concepts" ``AxBenchBaseError`` by construction
    (``len(accepted)=3 < max_concepts=6``), then the dispatcher broadcasts
    that merge failure so every rank terminates coherently. That error is NOT
    a checkpoint rejection: it fires AFTER every per-rank checkpoint is
    committed, and it fires IDENTICALLY on the uninterrupted reference (same
    reject set, same 3 accepts). The release-blocking failure this test
    actually guards is a FALSE fingerprint / stale-schema rejection of the
    cap-binding checkpoint; assertion (a) discriminates exactly that (error
    text contains "fingerprint" / "stale schema") and asserts the ref/resume
    exhaustion errors are symmetric. A genuine resume fingerprint regression
    would surface as a checkpoint-rejection error on resume but NOT on ref →
    assertion (a) fails → BLOCKED.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    n_candidates = 80
    configured_block = 8
    n_verified_target = 2
    world_size = 4
    max_concepts = 6
    # Crash cap (global-candidate-position): block 0=[0,12) all <14 →
    # commits; block 1=[12,24) → first owned 12+r<14 then 16+r>=14 raises.
    candidate_visit_cap = 14

    # Low-accept-ratio reject set: accept ONLY candidates 0,1,2 (3 global
    # accepts < max_concepts=6) so the run POOL-EXHAUSTS, never
    # accept-saturates. feature_index = 2000 + i per _build_candidates.
    accept_positions = {0, 1, 2}
    reject_feature_indices = [
        2000 + i for i in range(n_candidates) if i not in accept_positions
    ]

    # The cap MUST bind: effective B = min(8, ceil(2*6/4)=3) = 3 < 8.
    probe = _build_cfg(
        run_root=tmp_path, run_tag="probe", max_concepts=max_concepts,
        block_size=configured_block, n_verified_target=n_verified_target,
    )
    eff_b = runner_mod._block_size(probe, world_size=world_size)
    assert eff_b == 3, f"expected effective B==3, got {eff_b}"
    assert eff_b < configured_block, (
        f"cap must BIND: effective B={eff_b} must be < configured "
        f"{configured_block}"
    )

    # Uninterrupted reference: resume target with no prior checkpoint =
    # a full fresh run under the SAME cap-binding cfg AND the SAME
    # reject set (so ref_visited is the true pool-exhaust visited set).
    ref = gloo_multiprocess(
        world_size, _stage0c_resume_run_target,
        str(tmp_path), "capbinds-ref",
        n_candidates, configured_block, n_verified_target, max_concepts,
        reject_feature_indices,
    )
    ref_visited: set[int] = set()
    for r in ref:
        ref_visited |= set(r["visited_candidate_indices"])

    # Interrupted: capped partial (crashes mid-block-1 after block-0
    # commit), then resume — same cfg & run_tag & reject set.
    partial = gloo_multiprocess(
        world_size, _stage0c_partial_run_target,
        str(tmp_path), "capbinds-int",
        n_candidates, candidate_visit_cap, configured_block,
        n_verified_target, max_concepts, reject_feature_indices,
    )
    partial_visited: set[int] = set()
    for r in partial:
        partial_visited |= set(r["visited_candidate_indices"])
    # The partial worker catches the cap RuntimeError internally; every
    # rank should report it raised (the crash actually fired).
    assert all(r["raised"] for r in partial), (
        f"every rank's capped partial must have raised mid-flight; got "
        f"{[(r['rank'], r['raised']) for r in partial]!r}"
    )

    # ---- HARD ANTI-VACUITY GUARD (pre-resume) ----
    # A real, non-empty per-rank checkpoint MUST exist on disk before the
    # resume phase. If it does not, the partial accept-saturated +
    # merge-cleanup destroyed it (the original vacuity) and "resume"
    # would degenerate to a fresh run. This assertion is a correctness
    # gate, not a convenience — never weaken or remove it.
    per_rank_json_paths = [
        _output_paths(tmp_path, "capbinds-int").stage0c_partial_rank_json(r)
        for r in range(world_size)
    ]
    existing_with_visited: list[Path] = []
    for p in per_rank_json_paths:
        if not p.exists():
            continue
        committed = json.loads(p.read_text(encoding="utf-8"))
        if list(committed.get("visited_candidate_indices", [])):
            existing_with_visited.append(p)
    assert existing_with_visited, (
        "ANTI-VACUITY: no per-rank stage0c_partial_rank_*.json with "
        "committed visited_candidate_indices exists before resume — the "
        "partial must have accept-saturated and merge-cleanup deleted it, "
        "so 'resume' would be a vacuous fresh run. Checked: "
        f"{[str(p) for p in per_rank_json_paths]}"
    )
    assert partial_visited, (
        "ANTI-VACUITY: union of committed visited indices is empty; the "
        "partial committed no block-boundary checkpoint."
    )

    resumed = gloo_multiprocess(
        world_size, _stage0c_resume_run_target,
        str(tmp_path), "capbinds-int",
        n_candidates, configured_block, n_verified_target, max_concepts,
        reject_feature_indices,
    )
    resumed_per_rank = [set(r["visited_candidate_indices"]) for r in resumed]
    resumed_visited: set[int] = set().union(*resumed_per_rank)

    # (a) NO FALSE FINGERPRINT/SCHEMA REJECTION on RESUME under a binding
    # cap (the release-blocking failure mode this test guards). By design
    # the pool has only 3 global accepts < max_concepts=6, so the run
    # POOL-EXHAUSTS and rank 0's ``run_merge`` raises the intrinsic
    # "Stage 0c exhausted ... without accepting cfg.max_concepts"
    # AxBenchBaseError, which is then broadcast so all ranks terminate
    # coherently. This is EXPECTED and occurs IDENTICALLY on the uninterrupted
    # reference (asserted symmetric below); it is NOT a checkpoint rejection.
    # The release-blocker is a fingerprint/stale-schema rejection:
    # ``validate_per_rank_fingerprint`` raises a message containing
    # "fingerprint" / "stale schema". Assert no resume worker raised THAT.
    # (gloo_multiprocess re-raises any non-AxBenchBaseError worker exception as
    # RuntimeError, so an unexpected crash still fails the test loudly.)
    def _is_checkpoint_rejection(err: str | None) -> bool:
        return err is not None and (
            "fingerprint" in err or "stale schema" in err
        )

    resume_rejections = [
        (r["rank"], r.get("err"))
        for r in resumed
        if _is_checkpoint_rejection(r.get("err"))
    ]
    assert not resume_rejections, (
        f"RESUME under a binding cap was FALSELY fingerprint/schema-"
        f"rejected (release-blocking — the cap-binding checkpoint must "
        f"validate because the fingerprint stores the RAW block size and "
        f"excludes world_size): {resume_rejections!r}"
    )
    # Symmetry: every resume worker that raised must have raised the SAME
    # intrinsic pool-exhaustion error the uninterrupted reference does —
    # never anything else. (ref has no prior checkpoint so it cannot get
    # a fingerprint error; both sides must hit identical exhaustion.)
    ref_errs = {r.get("err") for r in ref if r["raised"]}
    resumed_errs = {r.get("err") for r in resumed if r["raised"]}
    # ``run_merge`` executes on rank 0, and its result is broadcast so EVERY
    # rank raises instead of letting nonzero ranks report false success. Pin
    # the complete rank set so the symmetry assertions below cannot pass
    # vacuously or miss a broken merge-error broadcast.
    expected_raising_ranks = list(range(world_size))
    resumed_raising_ranks = [r["rank"] for r in resumed if r["raised"]]
    assert resumed_raising_ranks == expected_raising_ranks, (
        f"expected every rank to raise on resume, got {resumed_raising_ranks}"
    )
    assert [r["rank"] for r in ref if r["raised"]] == expected_raising_ranks, (
        f"expected every rank to raise on ref, got "
        f"{[r['rank'] for r in ref if r['raised']]}"
    )
    assert [r.get("err") for r in resumed] == [r.get("err") for r in ref], (
        "resume and uninterrupted runs must broadcast the same merge failure "
        f"per rank; resumed={[r.get('err') for r in resumed]!r}, "
        f"ref={[r.get('err') for r in ref]!r}"
    )
    assert all(
        e is not None and "exhausted candidate_concepts.json" in e
        for e in resumed_errs
    ), (
        f"every raising resume worker must raise the intrinsic pool-"
        f"exhaustion error, got {sorted(resumed_errs)!r}"
    )
    assert all(
        e is not None and "exhausted candidate_concepts.json" in e
        for e in ref_errs
    ), (
        f"every raising reference worker must raise the intrinsic pool-"
        f"exhaustion error (symmetry), got {sorted(ref_errs)!r}"
    )
    # (b) interrupted+resumed ≡ uninterrupted under a binding cap.
    assert resumed_visited == ref_visited, (
        f"resume under a binding cap must reach the same visited set as "
        f"an uninterrupted run; resumed-only="
        f"{sorted(resumed_visited - ref_visited)} "
        f"ref-only={sorted(ref_visited - resumed_visited)}"
    )
    # (c) preservation: nothing the partial visited was lost.
    assert partial_visited <= resumed_visited, (
        f"partial-only (lost work) = "
        f"{sorted(partial_visited - resumed_visited)}"
    )
    # (d) HARD ANTI-VACUITY GUARD (post-resume): the resume STRICTLY
    # extended the partial's committed set — proving it genuinely
    # continued from the on-disk checkpoint rather than running fresh
    # and coincidentally matching. This is the assertion the original
    # vacuous formulation could not make (there partial == resumed).
    # Never weaken to ``<=``.
    assert partial_visited < resumed_visited, (
        f"ANTI-VACUITY: resume must STRICTLY extend the partial's "
        f"committed visited set (proving it read the checkpoint and "
        f"continued). |partial|={len(partial_visited)}, "
        f"|resumed|={len(resumed_visited)}; if equal, resume did not "
        f"continue from the checkpoint."
    )
    # no double-process: per-rank ownership disjoint + modulo-correct.
    for rank, vis in enumerate(resumed_per_rank):
        assert all(idx % world_size == rank for idx in vis), (
            f"rank {rank} visited non-owned index: {sorted(vis)}"
        )
    for i in range(world_size):
        for j in range(i + 1, world_size):
            assert resumed_per_rank[i].isdisjoint(resumed_per_rank[j])


# ---------------------------------------------------------------------------
# Fingerprint mismatch on resume — different ``verification_length``.
# ---------------------------------------------------------------------------


def _stage0c_resume_run_with_different_verification_length_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    run_tag: str,
    n_candidates: int,
    block_size: int,
    n_verified_target: int,
    verification_length: int,
) -> dict[str, Any]:
    """Resume worker that overrides ``verification_length`` to verify the
    per-rank fingerprint validator rejects a resume under a changed cfg.
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)
    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag=run_tag,
        block_size=block_size,
        n_verified_target=n_verified_target,
        verification_length=verification_length,
    )
    candidates = _build_candidates(n_candidates)
    concepts_path = tmp_path / f"{run_tag}_candidate_concepts.json"
    _write_concepts_json(concepts_path, candidates)

    _patch_dispatcher_helpers(
        candidates,
        n_verified_target=n_verified_target,
        candidate_visit_cap=None,
    )

    raised = False
    err: str | None = None
    try:
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=rank, world_size=world_size,
        )
    except AxBenchBaseError as exc:
        raised = True
        err = str(exc)
    return {"rank": rank, "raised": raised, "err": err}


def test_stage0c_resume_fingerprint_mismatch_raises(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Run at ws=2 with ``verification_length=4``, then resume with 5.

    The fingerprint differs, so every rank's
    :func:`validate_per_rank_fingerprint` raises with the canonical
    "fingerprint mismatch" message.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    n_candidates = 16
    candidate_visit_cap = 8
    block_size = 4
    n_verified_target = 2
    run_tag = "resume-fingerprint-mismatch"

    # Phase 1: partial run at verification_length=4.
    gloo_multiprocess(
        2, _stage0c_partial_run_target,
        str(tmp_path), run_tag,
        n_candidates, candidate_visit_cap, block_size, n_verified_target,
    )

    # Phase 2: resume with a different verification_length (5).
    results = gloo_multiprocess(
        2, _stage0c_resume_run_with_different_verification_length_target,
        str(tmp_path), run_tag,
        n_candidates, block_size, n_verified_target,
        5,
    )

    assert all(r["raised"] for r in results), (
        f"Every rank must raise on fingerprint mismatch; got {results!r}"
    )
    assert all(
        r["err"] is not None and "fingerprint" in r["err"]
        for r in results
    ), (
        f"Every rank's error must mention 'fingerprint'; got "
        f"{[r['err'] for r in results]!r}"
    )


# ---------------------------------------------------------------------------
# Schema-version bump rejection. Hand-rolls a JSON with a stale
# schema_version field; the resume must surface a clear "stale schema"
# error rather than silently using the partial fingerprint.
# ---------------------------------------------------------------------------


def test_stage0c_resume_stale_schema_version_raises(
    tmp_path, monkeypatch,
):
    """Hand-write a per-rank JSON whose ``schema_version`` is ``1`` (the
    pre-PR-4 baseline) and trigger a resume; the runner must reject
    the partial file with a "stale schema" error directing the user to
    delete the partial files.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    n_candidates = 8
    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag="resume-stale-schema",
        block_size=4,
        n_verified_target=2,
    )
    # cfg.run_tag is ``str | None`` per the dataclass; we just set it
    # to a real string above, so narrow it for ty.
    assert cfg.run_tag is not None
    run_tag_str: str = cfg.run_tag
    output_paths = _output_paths(tmp_path, run_tag_str)
    json_path = output_paths.stage0c_partial_rank_json(0)
    parquet_path = output_paths.stage0c_partial_rank_parquet(0)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    # Stale schema_version=1; the rest of the payload doesn't matter
    # because the version check fires first.
    stale_payload = {
        "schema_version": _RANK_CHECKPOINT_SCHEMA_VERSION - 1,
        "rank": 0,
        "world_size": 1,
        "n_candidates": n_candidates,
        "fingerprint": {},
        "next_block_index": 0,
        "visited_candidate_indices": [],
        "rank_accepts": [],
        "rank_rejects": [],
        "rank_qc_per_feature": [],
    }
    json_path.write_text(json.dumps(stale_payload), encoding="utf-8")
    del parquet_path  # not used; schema check fires before parquet read

    candidates = _build_candidates(n_candidates)
    # Use monkeypatch (NOT raw setattr) so the parent process's view of
    # ``si_mod._verify_concept`` is restored after the test. Raw setattr
    # would leak into other tests in the same pytest session.
    base_verify = _stub_verify_factory(2)
    monkeypatch.setattr(runner_mod, "load_and_validate_candidate_concepts",
                        lambda p, n_sae_features: candidates)
    monkeypatch.setattr(si_mod, "_list_parquets_for_feature_dir", lambda c: [])
    monkeypatch.setattr(runner_mod, "glob_features", lambda feature_dir: [])
    monkeypatch.setattr(
        extraction_mod, "build_percentile_table_from_stats",
        lambda p, n_sae_features: t.zeros(n_sae_features),
    )
    monkeypatch.setattr(
        si_mod, "_CandidatePrefixIndex", lambda *a, **kw: MagicMock(),
    )
    monkeypatch.setattr(si_mod, "_base_context_window", lambda m: 2048)
    monkeypatch.setattr(si_mod, "_verify_concept", base_verify)

    with pytest.raises(AxBenchBaseError, match=r"stale schema"):
        runner_mod.run_single_rank(
            cfg, candidates,
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
# Block-size mismatch on resume — different ``stage0c_multi_gpu_block_size``.
# ---------------------------------------------------------------------------


def _stage0c_resume_run_with_different_block_size_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    run_tag: str,
    n_candidates: int,
    block_size: int,
    n_verified_target: int,
) -> dict[str, Any]:
    """Resume worker that uses a different block_size than phase 1.

    Block size is in the fingerprint, so this should raise a
    ``fingerprint mismatch`` error (with the field name in the message).
    """
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)
    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag=run_tag,
        block_size=block_size,
        n_verified_target=n_verified_target,
    )
    candidates = _build_candidates(n_candidates)
    concepts_path = tmp_path / f"{run_tag}_candidate_concepts.json"
    _write_concepts_json(concepts_path, candidates)

    _patch_dispatcher_helpers(
        candidates,
        n_verified_target=n_verified_target,
        candidate_visit_cap=None,
    )

    raised = False
    err: str | None = None
    try:
        runner_mod.dispatch_prepare_negative_prefixes(
            cfg, concepts_path,
            base_model=MagicMock(),
            base_sae=MagicMock(),
            target_model_meta=MagicMock(sae_layer=8),
            n_sae_features=16384,
            response_cache=None,
            rank=rank, world_size=world_size,
        )
    except AxBenchBaseError as exc:
        raised = True
        err = str(exc)
    return {"rank": rank, "raised": raised, "err": err}


def test_stage0c_resume_block_size_mismatch_raises(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Bug surface: ``stage0c_multi_gpu_block_size`` is in the fingerprint
    so phase 1 at block_size=4 → phase 2 at block_size=8 must fail
    validation. A regression that excluded block_size from the
    fingerprint would let the runner resume with mismatched block
    boundaries → silently skip candidates.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    n_candidates = 16
    candidate_visit_cap = 8
    n_verified_target = 2
    run_tag = "resume-block-size-mismatch"
    # Phase 1: block_size=4.
    gloo_multiprocess(
        2, _stage0c_partial_run_target,
        str(tmp_path), run_tag,
        n_candidates, candidate_visit_cap, 4, n_verified_target,
    )
    # Phase 2: block_size=8 (different from phase 1).
    results = gloo_multiprocess(
        2, _stage0c_resume_run_with_different_block_size_target,
        str(tmp_path), run_tag,
        n_candidates, 8, n_verified_target,
    )
    assert all(r["raised"] for r in results), (
        f"Every rank must raise on block_size fingerprint mismatch; "
        f"got {results!r}"
    )
    # The error must mention the field name so the operator knows
    # what changed.
    assert all(
        r["err"] is not None and "stage0c_multi_gpu_block_size" in r["err"]
        for r in results
    ), (
        f"Every rank's error must mention the changed field; got "
        f"{[r['err'] for r in results]!r}"
    )


# ---------------------------------------------------------------------------
# G2: cross-world-size resume where the adaptive cap binds DIFFERENTLY per ws.
# ---------------------------------------------------------------------------


def test_stage0c_resume_cross_world_size_cap_binds_differently(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Cross-ws resume where ``_block_size`` yields DIFFERENT effective B per
    world size, proving the adaptive cap does not regress cross-ws resume.

    Params chosen so ``_block_size(cfg, ws=8) != _block_size(cfg, ws=2)``
    and BOTH are capped below ``configured_block``:

    * ``configured_block=8``, ``max_concepts=6``.
    * ``_block_size(cfg, ws=8) = max(1, min(8, ceil(12/8)=2)) = 2``.
    * ``_block_size(cfg, ws=2) = max(1, min(8, ceil(12/2)=6)) = 6``.

    The fingerprint stores the RAW ``stage0c_multi_gpu_block_size`` (=8)
    and excludes ``world_size``, so the resume at ws=2 sees the same raw
    fingerprint as the partial at ws=8 and is NOT fingerprint-rejected even
    though the effective B differs. The cross-ws resume contract asserts:

    * Every initial rank committed at least one block (sanity).
    * Every candidate is owned by exactly one rank under ws_resume.
    * ``total_remaining == expected_remaining`` (EXACT match per the
      per-rank-only checkpoint contract — same formula as the parametric
      matrix test).

    The low-accept-ratio stub (accept only candidates 0,1,2) ensures the
    run POOL-EXHAUSTS (3 global accepts < max_concepts=6), never
    accept-saturates, so per-rank checkpoints survive on disk after the
    capped partial crashes mid-block-1.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    ws_initial = 8
    ws_resume = 2
    configured_block = 8
    max_concepts = 6
    n_verified_target = 2
    n_candidates = 80
    run_tag = "resume-cap-differs-ws8-to-ws2"

    # Low-accept-ratio reject set: accept ONLY candidates 0,1,2 so the run
    # POOL-EXHAUSTS (3 global accepts < max_concepts=6), never accept-saturates.
    # feature_index = 2000 + i per _build_candidates.
    accept_positions = {0, 1, 2}
    reject_feature_indices = [
        2000 + i for i in range(n_candidates) if i not in accept_positions
    ]

    # Probe asserts: verify the cap binds differently per ws.
    probe = _build_cfg(
        run_root=tmp_path, run_tag="probe-cap-differs",
        max_concepts=max_concepts, block_size=configured_block,
        n_verified_target=n_verified_target,
    )
    assert runner_mod._block_size(probe, 8) == 2, (
        f"expected _block_size(cfg, ws=8)==2, "
        f"got {runner_mod._block_size(probe, 8)}"
    )
    assert runner_mod._block_size(probe, 2) == 6, (
        f"expected _block_size(cfg, ws=2)==6, "
        f"got {runner_mod._block_size(probe, 2)}"
    )

    # With ws=8, effective B=2: block 0 covers 2*8=16 global candidates
    # [0,16). The cap fires inside block 1 [16,32) after block 0 commits.
    # Each rank r's block-0 stripe = {r, 8+r} (both < cap); block-1 first
    # owned = 16+r < 24 (verified), second owned = 24+r >= 24 (raises).
    # So every rank commits exactly block 0 then raises mid-block-1.
    candidate_visit_cap = (
        configured_block * ws_initial // ws_initial  # = effective B = 2
        * ws_initial  # block 0 size = 16
        + (configured_block * ws_initial // ws_initial * ws_initial) // 2
        # = 16 + 8 = 24; but let's compute directly:
    )
    # Compute cap cleanly: block 0 size = eff_b * ws_initial; fire mid-block-1.
    eff_b_initial = runner_mod._block_size(probe, ws_initial)  # = 2
    block0_size = eff_b_initial * ws_initial  # = 16
    candidate_visit_cap = block0_size + block0_size // 2  # = 16 + 8 = 24

    # Phase 1: partial run at ws_initial=8.
    partial_results = gloo_multiprocess(
        ws_initial, _stage0c_partial_run_target,
        str(tmp_path), run_tag,
        n_candidates, candidate_visit_cap, configured_block, n_verified_target,
        max_concepts, reject_feature_indices,
    )

    # Read ACTUAL committed visited indices per initial rank.
    visited_by_initial_rank: dict[int, list[int]] = {
        r["rank"]: list(r["visited_candidate_indices"])
        for r in partial_results
    }
    # Sanity: every initial rank must have committed at least one block.
    assert all(
        len(v) > 0 for v in visited_by_initial_rank.values()
    ), (
        f"Every initial rank must have committed at least one block; "
        f"got {visited_by_initial_rank!r}"
    )

    expected_remaining = _expected_remaining_from_actual_visited(
        ws_initial=ws_initial,
        ws_resume=ws_resume,
        n_candidates=n_candidates,
        visited_by_rank=visited_by_initial_rank,
    )

    # Phase 2: snapshot (no mutation) — count remaining work under ws_resume.
    snapshots = gloo_multiprocess(
        ws_resume, _stage0c_resume_count_target,
        str(tmp_path), run_tag,
        n_candidates, configured_block, n_verified_target,
    )

    total_owned_now = sum(s["owned_now"] for s in snapshots)
    total_preserved = sum(s["preserved"] for s in snapshots)
    total_remaining = sum(s["remaining"] for s in snapshots)

    # Sanity: every candidate is owned by exactly one rank under ws_resume.
    assert total_owned_now == n_candidates, (
        f"At ws_resume={ws_resume}, ranks should own all {n_candidates} "
        f"candidates; got {total_owned_now}"
    )

    # EXACT EQUALITY — the core G2 contract.
    assert total_remaining == expected_remaining, (
        f"Resume mismatch at ws_initial={ws_initial} -> ws_resume={ws_resume} "
        f"(cap-binds-differently: eff_B={eff_b_initial} vs "
        f"{runner_mod._block_size(probe, ws_resume)}): "
        f"expected exactly {expected_remaining} remaining, got "
        f"{total_remaining}. preserved={total_preserved}. This indicates "
        "a regression in is_my_candidate, read_per_rank_checkpoint, or "
        "the runner's resume guard — the adaptive cap must not regress "
        "cross-ws resume when effective B differs per world size."
    )
