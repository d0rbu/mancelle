"""Byte-identity gate for the PR-4 multi-GPU Stage 0c runner.

This is THE GATE for PR-4 correctness. It pins the contract:

    For a fixed cfg + fixed candidate pool + deterministic stubs of
    ``_verify_concept`` (and the prefix-index/parquet/feature-shard
    loaders), a multi-GPU Stage 0c run at world_size N produces
    byte-identical Stage 0c artifacts to the single-GPU run at
    world_size 1.

The three artifacts checked:

* ``concepts.json`` — final accepted concepts in canonical order
* ``rejected_concepts.json`` — every rejected candidate, sorted by
  candidate_index
* ``negative_mode_prefixes.parquet`` — verified negative prefixes,
  rows indexed by canonical concept_id

The contract is enforced via raw ``bytes`` equality on the JSON files
and via row-by-row pylist equality on the parquet (the parquet metadata
inherits the tokenizer id from cfg, which is identical across the two
runs, so the binary form should also match — but the row comparison is
the load-bearing one because pyarrow does not guarantee byte-equal
output for identical input across releases).

Test design
-----------
Each scenario:

1. Run the dispatcher at world_size=1 → reference artifacts.
2. Run the dispatcher at world_size=N (N in {2, 4, 8}) under
   ``gloo_multiprocess`` → multi-GPU artifacts.
3. Compare the three artifacts byte-for-byte.

The test scenarios cover the matrix specified in the PR-4 plan:

* default block size (16) at ws ∈ {2, 4, 8}
* small block size (4) at ws=4 (tighter loop)
* large block size (64) at ws=4 with a larger pool (fewer barriers)
* "easy" pool (N >> max_concepts*ws — every rank has plenty of work)
* "tight" pool (N == max_concepts — loop exhausts pool before max)
* low-accept-ratio (only 3 accepts in 24, max_concepts=2 — early term)

Workers monkeypatch the heavy helpers (verify, prefix index, parquet
listing, feature shard glob, percentile table) so the dispatcher exercises
PURE python control flow without touching the network or the filesystem-
heavy paths. The deterministic stub for ``_verify_concept`` returns a
canned per-candidate row count keyed by the candidate's feature_index;
this is the same pattern used in the existing unit tests
(``tests/axbench_base/unit/test_stage0c_runner.py``).

If any byte-identity assertion fails this test must NOT be papered over
with a "looser" comparison: a regression here is a real correctness bug
in the multi-GPU runner.
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
from steering.eval.axbench_base import concept_list as concept_list_mod
from steering.eval.axbench_base import extraction as extraction_mod
from steering.eval.axbench_base import paths as paths_mod
from steering.eval.axbench_base import stage0c_runner as runner_mod
from steering.eval.axbench_base._stage0c_internals import (
    _VerifiedNegativePrefix,
)
from steering.eval.axbench_base.concept_examples_qc import QCFeatureScore, QCReport
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.types import CandidateConcept, QcOutcome

# ---------------------------------------------------------------------------
# Module-level helpers used both inside and outside the spawned workers.
# Worker targets must be importable, so EVERY function the workers reach
# must live at module scope.
# ---------------------------------------------------------------------------


def _write_minimal_stats(path: Path, n_sae_features: int = 16384) -> Path:
    """Write a minimal feature-statistics safetensors.

    The dispatcher reads ``n_sae_features`` from the safetensors metadata.
    Tests don't need real per-feature percentiles because we stub the
    ``build_percentile_table_from_stats`` helper to return an all-zero
    tensor of the correct shape.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )
    return path


def _build_candidates(n: int) -> list[CandidateConcept]:
    """Build N deterministic CandidateConcept entries.

    feature_index = 1000 + i (so feature_index uniquely identifies the
    candidate across both ws=1 and ws=N runs). The other fields are
    stable strings derived from i.
    """
    return [
        CandidateConcept(
            feature_index=1000 + i,
            layer=8,
            sae_source="8-gemmascope-res-16k",
            description=f"feature_{i:03d}",
            corpus_count=10 * (i + 1),
            description_source="neuronpedia",
        )
        for i in range(n)
    ]


def _build_cfg(
    *,
    run_root: Path,
    run_tag: str,
    max_concepts: int,
    block_size: int,
    n_verified_target: int,
    enable_qc: bool = False,
) -> AxBenchBaseConfig:
    """Build a deterministic AxBenchBaseConfig for byte-identity tests.

    The dispatcher reads several cfg fields beyond what
    ``run_single_rank`` directly uses (validate_concepts requires
    n_sae_features, etc.). We pre-write the stats file so the
    dispatcher's "n_sae_features=...." log line resolves consistently
    across ranks.
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
        enable_concept_examples_qc=enable_qc,
        n_verified_prefixes_per_concept_target=n_verified_target,
        n_verification_seeds=2,
        min_consistency_passes=1,
        verification_percentile_threshold=0.5,
        verification_length=4,
        min_prefix_chars=1,
        extraction_tokenizer_id="google/gemma-2-2b",
        stage0c_multi_gpu_block_size=block_size,
        verification_api_model=None,
    )


def _write_concepts_json(path: Path, candidates: list[CandidateConcept]) -> None:
    """Persist the candidate concept list under ``path`` (the dispatcher
    reads it via ``load_and_validate_candidate_concepts``)."""
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


# Deterministic accept/reject pattern — given a per-test "rejects" set,
# every candidate not in the set produces ``n_verified_target`` rows;
# every candidate IN the set produces ZERO rows (verification rejection).
def _stub_verify_factory(
    n_verified_target: int, reject_feature_indices: frozenset[int],
):
    """Return a callable matching the ``_verify_concept`` signature.

    The verification stub is keyed by ``concept.feature_index`` so accept/
    reject decisions are stable regardless of which rank calls
    ``_verify_concept`` and in what order.

    Each row has stable per-prefix fields derived from ``feature_index`` and
    the row index. Crucially, ``verification_seeds`` also depend on the
    provisional ``concept.concept_id`` just like production verification.
    This makes the gate fail if a topology assigns rank-local ordinals instead
    of the stable candidate index, even when accept/reject decisions match.
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
                verification_seeds=[
                    1000 + 10 * int(concept.concept_id) + i
                ],
                verification_percentiles=[80.0],
            )
            for i in range(n_verified_target)
        ]

    return _verify


def _patch_dispatcher_helpers(
    candidates: list[CandidateConcept],
    *,
    n_verified_target: int,
    reject_feature_indices: frozenset[int],
    enable_qc: bool = False,
) -> None:
    """Monkeypatch every dispatcher helper that touches the filesystem,
    network, or heavy compute path. Stays at module scope so workers can
    call it after spawn (each worker has its own monkeypatch-equivalent
    on the imported module).
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
    si_module._verify_concept = _stub_verify_factory(
        n_verified_target, reject_feature_indices
    )
    if enable_qc:
        concept_list_module: Any = concept_list_mod
        concept_list_module._build_concept_examples_qc_client = (
            lambda cfg, response_cache=None: _DeterministicQcClient()
        )


class _DeterministicQcClient:
    """Accept every requested candidate with one stable QC record."""

    def filter(self, candidates, *, target_surviving=None):
        del target_surviving
        per_feature = [
            QCFeatureScore(
                feature_index=feature_id,
                description=description,
                outcome=QcOutcome.ACCEPTED,
                score=2,
                n_examples=1,
                error_message=None,
            )
            for feature_id, description in candidates
        ]
        return list(candidates), QCReport(
            n_total=len(candidates),
            n_accepted=len(candidates),
            n_rejected=0,
            n_fetch_errors=0,
            n_judge_errors=0,
            n_empty_examples=0,
            score_histogram={2: len(candidates)} if candidates else {},
            per_feature=per_feature,
            judge_min_score=1,
            n_processed=len(candidates),
        )


# ---------------------------------------------------------------------------
# Worker target — one rank's invocation of the Stage 0c dispatcher under
# ``gloo_multiprocess``. The parent passes a list of plain-tuple candidates
# (CandidateConcept is a regular dataclass and pickles fine; we keep the
# tuple form for clarity in the worker signature).
# ---------------------------------------------------------------------------


def _stage0c_run_target(
    rank: int,
    world_size: int,
    tmp_path_str: str,
    run_tag: str,
    n_candidates: int,
    max_concepts: int,
    block_size: int,
    n_verified_target: int,
    reject_feature_indices: list[int],
    enable_qc: bool = False,
) -> dict[str, Any]:
    """Worker entrypoint: spawn-friendly, returns a small dict.

    Every parameter is JSON/pickle-friendly. The worker rebuilds the
    candidate pool from ``n_candidates`` (deterministic by index) and
    monkeypatches the dispatcher helpers BEFORE calling the dispatcher.
    Returns a dict with the absolute path of each Stage 0c artifact so
    the parent can do byte equality.
    """
    # Same NANOCHAT_BASE_DIR semantics as the unit tests — the dispatcher
    # (via paths_mod.run_output_root) honours this env var.
    os.environ["NANOCHAT_BASE_DIR"] = tmp_path_str
    tmp_path = Path(tmp_path_str)
    cfg = _build_cfg(
        run_root=tmp_path,
        run_tag=run_tag,
        max_concepts=max_concepts,
        block_size=block_size,
        n_verified_target=n_verified_target,
        enable_qc=enable_qc,
    )
    candidates = _build_candidates(n_candidates)
    concepts_path = tmp_path / f"{run_tag}_candidate_concepts.json"
    _write_concepts_json(concepts_path, candidates)

    _patch_dispatcher_helpers(
        candidates,
        n_verified_target=n_verified_target,
        reject_feature_indices=frozenset(reject_feature_indices),
        enable_qc=enable_qc,
    )

    # Each rank invokes the dispatcher; rank 0 returns the canonical
    # parquet path after performing the merge, the others return the
    # same path string (after ``dist.barrier()``).
    out_path = runner_mod.dispatch_prepare_negative_prefixes(
        cfg, concepts_path,
        base_model=MagicMock(),
        base_sae=MagicMock(),
        target_model_meta=MagicMock(sae_layer=8),
        n_sae_features=16384,
        response_cache=None,
        rank=rank, world_size=world_size,
    )

    # Every rank reports the canonical paths. Rank 0 owns the actual
    # files; the others are no-ops on the filesystem.
    output_paths = paths_mod.AxBenchBasePaths.from_config(cfg)
    return {
        "rank": rank,
        "concepts_json": str(output_paths.concepts_json()),
        "rejected_json": str(output_paths.rejected_concepts_json()),
        "prefixes_parquet": str(out_path),
        "qc_report": str(output_paths.concept_examples_qc_report()),
    }


# ---------------------------------------------------------------------------
# Byte-identity helpers. Each artifact has its own comparator so failure
# messages localise the regression to the offending file.
# ---------------------------------------------------------------------------


def _assert_json_bytes_identical(
    reference: Path, candidate: Path, *, label: str,
) -> None:
    ref_bytes = reference.read_bytes()
    cand_bytes = candidate.read_bytes()
    assert ref_bytes == cand_bytes, (
        f"{label}: {candidate} differs from reference {reference}.\n"
        f"  reference len={len(ref_bytes)}, candidate len={len(cand_bytes)}.\n"
        f"  Diff: ws=1 wrote {ref_bytes!r}\n  ws=N wrote {cand_bytes!r}"
    )


def _assert_parquet_rows_identical(
    reference: Path, candidate: Path, *, label: str,
) -> None:
    """Compare two parquet files by reading the rows back to pylists.

    pyarrow doesn't guarantee byte-identical output for identical input
    (compression metadata, page boundaries can shift across pyarrow
    versions even with the same schema), so we equate at the row level.
    Schema column order + types are also equated to surface any
    accidental schema drift.
    """
    ref_table = pq.read_table(reference)
    cand_table = pq.read_table(candidate)
    # Schema equality: column names and types match. Metadata may differ
    # (Arrow writes a build identifier into the schema sometimes), so
    # we compare names + types explicitly rather than ``schema.equals``.
    ref_names = ref_table.schema.names
    cand_names = cand_table.schema.names
    assert ref_names == cand_names, (
        f"{label}: column-name order differs.\n"
        f"  reference: {ref_names}\n  candidate: {cand_names}"
    )
    ref_types = [str(ref_table.schema.field(n).type) for n in ref_names]
    cand_types = [str(cand_table.schema.field(n).type) for n in cand_names]
    assert ref_types == cand_types, (
        f"{label}: column types differ.\n"
        f"  reference: {ref_types}\n  candidate: {cand_types}"
    )
    ref_rows = ref_table.to_pylist()
    cand_rows = cand_table.to_pylist()
    assert len(ref_rows) == len(cand_rows), (
        f"{label}: row count differs. reference={len(ref_rows)}, "
        f"candidate={len(cand_rows)}"
    )
    for i, (rr, cr) in enumerate(zip(ref_rows, cand_rows, strict=True)):
        assert rr == cr, (
            f"{label}: row {i} differs.\n"
            f"  reference: {rr}\n  candidate: {cr}"
        )


def _assert_three_artifacts_identical(
    *,
    output_root: Path,
    ref_run_tag: str,
    cand_run_tag: str,
    ws: int,
    enable_qc: bool = False,
) -> None:
    """Compare the three Stage 0c artifacts between two run_tag dirs."""
    ref_paths = paths_mod.AxBenchBasePaths(output_root, ref_run_tag)
    cand_paths = paths_mod.AxBenchBasePaths(output_root, cand_run_tag)
    ref_concepts = ref_paths.concepts_json()
    cand_concepts = cand_paths.concepts_json()
    ref_rejected = ref_paths.rejected_concepts_json()
    cand_rejected = cand_paths.rejected_concepts_json()
    ref_parquet = ref_paths.negative_mode_prefixes()
    cand_parquet = cand_paths.negative_mode_prefixes()

    _assert_json_bytes_identical(
        ref_concepts, cand_concepts,
        label=f"concepts.json byte-identity (ws=1 vs ws={ws})",
    )
    _assert_json_bytes_identical(
        ref_rejected, cand_rejected,
        label=f"rejected_concepts.json byte-identity (ws=1 vs ws={ws})",
    )
    _assert_parquet_rows_identical(
        ref_parquet, cand_parquet,
        label=f"negative_mode_prefixes.parquet (ws=1 vs ws={ws})",
    )
    if enable_qc:
        _assert_json_bytes_identical(
            ref_paths.concept_examples_qc_report(),
            cand_paths.concept_examples_qc_report(),
            label=f"concept_examples_qc_report.json (ws=1 vs ws={ws})",
        )
        assert pq.read_schema(ref_parquet).metadata == (
            pq.read_schema(cand_parquet).metadata
        )


# ---------------------------------------------------------------------------
# Scenario builder — captures the three control variables for a
# byte-identity test (pool size, max_concepts, block_size, accept set).
# ---------------------------------------------------------------------------


def _run_reference_and_compare(
    *,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    gloo_multiprocess,
    ws_resume: int,
    n_candidates: int,
    max_concepts: int,
    block_size: int,
    n_verified_target: int,
    reject_feature_indices: list[int],
    label: str,
    enable_qc: bool = False,
) -> None:
    """Run ws=1 reference + ws=N multi-GPU and compare artifacts.

    Each invocation uses a distinct ``run_tag`` so the two artifact
    directories are isolated within the same ``tmp_path``. Both runs
    use IDENTICAL cfg modulo run_tag (which is not in the fingerprint).

    The parent process sets ``NANOCHAT_BASE_DIR`` so the post-run
    artifact-path lookups via :func:`paths_mod.concepts_json` resolve
    to the same directory the workers wrote to. Workers also set the
    env var inside their process; this fixture only ensures the parent
    sees the same view of the filesystem.
    """
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    ref_tag = f"{label}-ws1"
    cand_tag = f"{label}-ws{ws_resume}"

    # Reference run — single-GPU. world_size=1 takes the no-dist path
    # in the dispatcher.
    gloo_multiprocess(
        1, _stage0c_run_target,
        str(tmp_path), ref_tag,
        n_candidates, max_concepts, block_size, n_verified_target,
        list(reject_feature_indices), enable_qc,
    )

    # Multi-GPU run.
    gloo_multiprocess(
        ws_resume, _stage0c_run_target,
        str(tmp_path), cand_tag,
        n_candidates, max_concepts, block_size, n_verified_target,
        list(reject_feature_indices), enable_qc,
    )

    _assert_three_artifacts_identical(
        output_root=tmp_path / "eval_out",
        ref_run_tag=ref_tag,
        cand_run_tag=cand_tag,
        ws=ws_resume,
        enable_qc=enable_qc,
    )


# ---------------------------------------------------------------------------
# THE GATE: the byte-identity matrix. Every test below runs ws=1 +
# ws=N and asserts byte-identity of the three Stage 0c artifacts.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_stage0c_byte_identity_default_block_size(
    tmp_path, monkeypatch, gloo_multiprocess, world_size,
):
    """Default block size (16). 24 candidates, 8 max_concepts; reject
    every other candidate so the loop has both accept and reject paths
    interleaved with the block boundaries.

    THIS IS THE PRIMARY GATE. Run with -v to see the parametric matrix:
    ws=2, ws=4, ws=8. A failure at any world_size indicates a real
    correctness bug in the multi-GPU runner.
    """
    n_candidates = 24
    candidates = _build_candidates(n_candidates)
    # Reject every other candidate (12 rejects, 12 accepts in candidate
    # order). max_concepts=8 → first 8 accepts in candidate-index order,
    # which (with this reject pattern) is candidates 0, 2, 4, 6, 8, 10,
    # 12, 14.
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i % 2 == 1
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=world_size,
        n_candidates=n_candidates,
        max_concepts=8,
        block_size=16,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="default-block",
    )


def test_stage0c_qc_report_and_bound_identity_match_ws1_ws2(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """QC diagnostics are canonical, not rank-stripe/overvisit ordered."""
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=2,
        n_candidates=8,
        max_concepts=3,
        block_size=4,
        n_verified_target=2,
        reject_feature_indices=[],
        label="qc-report-identity",
        enable_qc=True,
    )


def test_stage0c_byte_identity_small_block_size_4(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Small block size (4), ws=4, 24 candidates. Tighter loop means more
    all_gather rendezvous points; this surfaces any bug in the
    block-barrier termination heuristic that doesn't hit at the default
    block size.
    """
    n_candidates = 24
    candidates = _build_candidates(n_candidates)
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i % 3 == 0
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=4,
        n_candidates=n_candidates,
        max_concepts=8,
        block_size=4,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="small-block",
    )


def test_stage0c_byte_identity_large_block_size_64(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Large block size (64), ws=4, 32 candidates. Single block covers
    the entire pool; only one all_gather happens. This is the "fewest
    barriers" stress case — a bug in the per-block commit ordering that
    only surfaces at a tight block size would NOT show here, hence the
    matrix.
    """
    n_candidates = 32
    candidates = _build_candidates(n_candidates)
    # Reject every fifth.
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i % 5 == 0
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=4,
        n_candidates=n_candidates,
        max_concepts=8,
        block_size=64,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="large-block",
    )


def test_stage0c_byte_identity_pool_larger_than_max_concepts(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Pool size 32 >> max_concepts*world_size = 8*4 = 32. The "easy"
    case: every rank has plenty of work in its first block; termination
    is driven by the global accept count, not pool exhaustion.
    """
    n_candidates = 32
    # All accept — no rejects. Termination is purely on max_concepts.
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=4,
        n_candidates=n_candidates,
        max_concepts=8,
        block_size=8,
        n_verified_target=2,
        reject_feature_indices=[],
        label="easy-pool",
    )


def test_stage0c_byte_identity_pool_smaller_than_max_concepts_times_ws(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Pool size 12 < max_concepts*world_size = 8*4 = 32. The "tight"
    case: the loop exhausts the pool before global accepts hit
    max_concepts; the run_merge "exhausted" error path applies BUT only
    when accepted < max_concepts. We make max_concepts=6 so 12-pool
    yields exactly 6 accepts (no exhaustion error).
    """
    n_candidates = 12
    candidates = _build_candidates(n_candidates)
    # Reject 6 / accept 6 → exactly hits max_concepts=6.
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i >= 6
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=4,
        n_candidates=n_candidates,
        max_concepts=6,
        block_size=4,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="tight-pool",
    )


def test_stage0c_byte_identity_low_accept_ratio(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Low accept ratio: only 3 accepts in N=24, max_concepts=2 →
    loop terminates as soon as 2 accepts land in candidate-sorted order.
    Every rank that owns one of the rejecting candidates contributes a
    rank_reject; the cross-rank reject-list merge is the load-bearing
    behaviour here.
    """
    n_candidates = 24
    candidates = _build_candidates(n_candidates)
    # Accept ONLY candidates 5, 11, 17. Everything else rejects.
    accept_set = {5, 11, 17}
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i not in accept_set
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=4,
        n_candidates=n_candidates,
        max_concepts=2,
        block_size=4,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="low-accept-ratio",
    )


def test_stage0c_byte_identity_adaptive_cap_binds_hard(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """Hard-invariant gate (spec decision §3.4): with the adaptive cap
    bound to its FLOOR (effective B == 1 at ws=8 while the ws=1
    reference uses the uncapped configured block of 32), the three
    production artifacts must be byte-identical. Sparse accept set so
    the cross-rank reject merge is also exercised under B==1
    (rendezvous every candidate).
    """
    n_candidates = 24
    max_concepts = 4
    configured_block = 32
    ws = 8
    # Sanity: the cap MUST bind to 1 for this to test what it claims.
    # ws=1 reference uses 32; ws=8 candidate uses min(32, ceil(8/8))=1.
    probe = _build_cfg(
        run_root=tmp_path, run_tag="probe",
        max_concepts=max_concepts, block_size=configured_block,
        n_verified_target=2,
    )
    assert runner_mod._block_size(probe, world_size=1) == configured_block
    assert runner_mod._block_size(probe, world_size=ws) == 1

    candidates = _build_candidates(n_candidates)
    accept_set = {2, 9, 16, 21}
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i not in accept_set
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=ws,
        n_candidates=n_candidates,
        max_concepts=max_concepts,
        block_size=configured_block,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="adaptive-cap-binds-hard",
    )


@pytest.mark.parametrize(
    "n_candidates,world_size",
    [
        (23, 4),  # n_candidates indivisible by world_size — uneven stripe
        (3, 4),   # pool smaller than world_size — most ranks own 0 candidates
        (1, 4),   # single candidate — only one rank owns it
    ],
)
def test_stage0c_byte_identity_uneven_stripe(
    tmp_path, monkeypatch, gloo_multiprocess, n_candidates, world_size,
):
    """Uneven stripes (pool size NOT divisible by world_size).

    The modulo partition assigns idx → rank = idx % world_size. When
    n_candidates is not a multiple of world_size, some ranks have
    fewer owned candidates than others. The byte-identity contract
    must hold across these cases — a regression in the merge step that
    assumed even striping would silently corrupt artifacts here.

    Cases:
    - (23, 4): rank 0 owns 6, ranks 1-2 own 6, rank 3 owns 5.
    - (3, 4): only ranks 0-2 have an owned candidate; rank 3 is empty.
    - (1, 4): only rank 0 has work.
    """
    # All accept — the test is about partitioning, not reject paths.
    reject_feature_indices: list[int] = []
    # max_concepts = whatever fits, but stay <= n_candidates so the
    # test exercises pool exhaustion (empty rejects, all accepts).
    max_concepts = min(n_candidates, 8)
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=world_size,
        n_candidates=n_candidates,
        max_concepts=max_concepts,
        block_size=2,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label=f"uneven-{n_candidates}-{world_size}",
    )


def test_stage0c_byte_identity_block_size_1(
    tmp_path, monkeypatch, gloo_multiprocess,
):
    """block_size=1 stresses the all_gather rendezvous frequency.

    Every candidate triggers a block-barrier, exposing any bug in
    early-termination heuristics that assumes amortisation across a
    block.
    """
    n_candidates = 12
    candidates = _build_candidates(n_candidates)
    reject_feature_indices = [
        c.feature_index for i, c in enumerate(candidates) if i % 4 == 0
    ]
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=2,
        n_candidates=n_candidates,
        max_concepts=4,
        block_size=1,
        n_verified_target=2,
        reject_feature_indices=reject_feature_indices,
        label="block-size-1",
    )


@pytest.mark.parametrize("world_size", [2, 4, 8])
def test_stage0c_byte_identity_all_accept(
    tmp_path, monkeypatch, gloo_multiprocess, world_size,
):
    """No verification rejects at all. ALL candidates are accept
    candidates; the rejected_concepts.json should be empty (just `[]`)
    and the parquet has max_concepts*n_verified_target rows.
    """
    n_candidates = 32
    _run_reference_and_compare(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        gloo_multiprocess=gloo_multiprocess,
        ws_resume=world_size,
        n_candidates=n_candidates,
        max_concepts=8,
        block_size=8,
        n_verified_target=3,
        reject_feature_indices=[],
        label="all-accept",
    )
