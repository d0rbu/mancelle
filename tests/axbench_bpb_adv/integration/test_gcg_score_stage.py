"""Task-10 score-stage integration for the Phase-2 feature-GCG methods.

The final whole-branch review found a CRITICAL: the four ``ours_gcg_*`` methods
hard-require ``adv_paths`` but it was wired into ONLY the class-margin scorer, so
enabling a gcg method crashed the DEFAULT ``steering_metric=roundtrip`` score
stage; and no test ran a real score STAGE with a gcg method. The human's design:

  1. a score-time coverage guard — the fair gcg methods score ONLY their
     HELD-OUT prefixes, and the roofline gcg methods score the SAME HELD-OUT
     prefixes with per-prefix oracle bags (the scored-prefix set the derive
     persists), and
  2. RAISE on any missing bag or missing scored-prefix coverage; once a row
     reaches scoring, its bag MUST exist.

This module pins, end to end:

  * the scored-prefix artifact I/O + path accessor (Section A);
  * the guard accepts complete persisted scored-prefix coverage, raises on
    incomplete coverage, is identity for non-gcg tables, and RAISES on a missing
    artifact — with a RED-proof that a LEAKED optimize prefix would pass
    (Section B);
  * the REAL derive (Tasks 5/6) persists EXACTLY the held-out prefixes for both
    fair and roofline, and the guard then accepts exactly those — the optimize
    prefixes produce NO scored rows (Section C);
  * the DEFAULT roundtrip score stage threads ``adv_paths`` to the scorer (no
    crash) AND checks coverage before scoring (Section D);
  * a SCORED prefix whose bag is missing RAISES through the real scorer seam
    (Section D).
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
import torch as t
from safetensors.torch import save_file

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.method_registry import get_method_by_name
from steering.eval.axbench_base.types import SteeringMode, TargetModelMetadata
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import AxBenchBpbError, InterventionKind
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import write_evaluation_population
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    read_ours_gcg_scored_prefixes,
    write_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.prefix_split import split_prefixes_for_concept
from steering.eval.axbench_bpb_adv.roundtrip_scoring import (
    run_model_backed_roundtrip_score_stage,
)
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA, ROUNDTRIP_SCORE_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import (
    ModelBackedDualCompletionScorer,
    require_precomputed_bag_candidates_cover_scored_prefixes,
    run_model_backed_bpb_adv_score_stage,
)

# Reuse the proven Task-5/9 tiny-encdec derive harness (real GCG optimize on a
# CPU/float32 SDPA encdec; the warm-start is a deterministic sub-optimal stub).
from tests.axbench_bpb_adv.integration.test_gcg_bags_fair import (
    _NUM_BUCKETS,
    _build_model,
)
from tests.axbench_bpb_adv.integration.test_gcg_end_to_end import (
    _CONCEPT_ID,
    _FAIR_METHOD,
    _ROOFLINE_METHOD,
    _distinct_prefixes,
    _make_cfg,
    _make_concept,
    _run_fair,
    _run_roofline,
)

_POS = SteeringMode.POSITIVE


@pytest.fixture(autouse=True)
def _force_sdpa(monkeypatch: pytest.MonkeyPatch) -> None:
    """FlexAttention lacks a CPU backward; force SDPA for the whole module."""
    monkeypatch.setattr(fa_module, "_override_impl", FlashAttnImpl.SDPA)


# ─────────────────────────────────────────────────────────────────────────────
# Candidate-row builder (a SCHEMA-valid row; the guard reads only method_name /
# concept_id / mode / core_prefix_id, so the token fields are trivial fillers)
# ─────────────────────────────────────────────────────────────────────────────

_SOURCE = [97, 98]
_RICH = [99, 100]
_AVOID = [101, 102]


def _candidate(
    *,
    method_name: str,
    core_prefix_id: str,
    candidate_id: str,
    mode: str = "positive",
    concept_id: int = _CONCEPT_ID,
    target_model: str = "encdec",
    coefficient_str: str = "0.0",
) -> dict[str, Any]:
    return {
        "candidate_id": candidate_id,
        "attack_row_id": f"ar_{candidate_id}",
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": "rich0",
        "concept_avoiding_sample_id": "avoid0",
        "concept_id": concept_id,
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": "faster_gcg",  # non-additive => no counter add-on
        "adversarial_strength": 0.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 0,
        "context_signal_mode": "fixed_context",
        "prefix_text": "ab",
        "prefix_hash": f"h_{core_prefix_id}",
        "source_prompt_tokens": list(_SOURCE),
        "source_prompt_token_digest": token_sequence_digest(_SOURCE),
        "attacked_prompt_text": "ab",
        "attacked_prompt_tokens": list(_SOURCE),
        "attacked_prompt_token_digest": token_sequence_digest(_SOURCE),
        "concept_rich_output_tokens": list(_RICH),
        "concept_avoiding_output_tokens": list(_AVOID),
        "concept_rich_output_digest": token_sequence_digest(_RICH),
        "concept_avoiding_output_digest": token_sequence_digest(_AVOID),
        "editable_token_start": len(_SOURCE),
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": None,
        "candidate_metadata_json": "{}",
    }


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def _write_bag(path: Path, bag: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fids = sorted(bag)
    save_file(
        {
            "feature_ids": t.tensor(fids, dtype=t.int64),
            "buckets": t.tensor([bag[fid] for fid in fids], dtype=t.int64),
        },
        str(path),
    )


def _survivors(table: pa.Table, method_name: str) -> set[str]:
    rows = table.to_pylist()
    return {str(row["core_prefix_id"]) for row in rows if str(row["method_name"]) == method_name}


# ─────────────────────────────────────────────────────────────────────────────
# Section A — scored-prefix artifact I/O + path accessor
# ─────────────────────────────────────────────────────────────────────────────


def test_scored_prefixes_path_layout(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    path = paths.ours_gcg_scored_prefixes(7, _POS, "fair")
    bag = paths.ours_gcg_bag(7, _POS, 0.0, "fair")
    # The artifact is a sibling of the bag tree, one per (concept, mode, family).
    assert path.parent == bag.parent
    assert path.name == "scored_prefixes.json"
    assert path.parent.name == "c7"


def test_scored_prefixes_write_read_roundtrip(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    path = paths.ours_gcg_scored_prefixes(7, _POS, "fair")
    # Duplicates + unsorted input => stored sorted + de-duped (a pure function).
    write_ours_gcg_scored_prefixes(path, ["p2", "p1", "p2"], concept_id=7, mode=_POS, family="fair")
    assert read_ours_gcg_scored_prefixes(path) == ["p1", "p2"]


def test_read_missing_scored_prefixes_raises(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    with pytest.raises(AxBenchBpbAdvError, match="missing precomputed-bag scored-prefix"):
        read_ours_gcg_scored_prefixes(paths.ours_gcg_scored_prefixes(7, _POS, "fair"))


def test_read_corrupt_scored_prefixes_raises(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    path = paths.ours_gcg_scored_prefixes(7, _POS, "fair")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{ this is not valid json")
    with pytest.raises(AxBenchBpbAdvError, match="corrupt or unreadable"):
        read_ours_gcg_scored_prefixes(path)


# ─────────────────────────────────────────────────────────────────────────────
# Section B — the coverage guard (pure logic; the artifact is written directly)
# ─────────────────────────────────────────────────────────────────────────────


def _fair_candidates(prefix_ids: list[str]) -> pa.Table:
    return _table(
        [
            _candidate(
                method_name="ours_gcg_encdec",
                core_prefix_id=pid,
                candidate_id=f"c_{pid}",
            )
            for pid in prefix_ids
        ]
    )


def test_guard_accepts_complete_fair_prefix_coverage(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1", "held2"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    # Candidates for held-out prefixes + an unrelated non-gcg row.
    candidates = _table(
        [
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c2"),
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held2", candidate_id="c4"),
            _candidate(
                method_name="diffmean_decoder_only",
                core_prefix_id="opt1",
                candidate_id="c5",
                target_model="decoder_only",
            ),
        ]
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert out is candidates


def test_guard_raises_for_uncovered_fair_prefix(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    candidates = _table(
        [
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c1"),
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="opt1", candidate_id="c2"),
        ]
    )
    with pytest.raises(AxBenchBpbAdvError, match="precomputed-bag coverage guard"):
        require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)


def test_guard_accepts_complete_roofline_prefix_coverage(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "roofline"),
        ["sub1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="roofline",
    )
    candidates = _table(
        [
            _candidate(
                method_name="ours_gcg_roofline_encdec",
                core_prefix_id="sub1",
                candidate_id="c1",
            ),
        ]
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert out is candidates


def test_guard_accepts_complete_feature_diffmean_prefix_coverage(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "feature_diffmean"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="feature_diffmean",
    )
    candidates = _table(
        [
            _candidate(
                method_name="feature_diffmean_encdec",
                core_prefix_id="held1",
                candidate_id="c1",
            ),
        ]
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert out is candidates


def test_guard_accepts_complete_roofline_feature_diffmean_prefix_coverage(
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(
            _CONCEPT_ID,
            _POS,
            "feature_diffmean_roofline",
        ),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="feature_diffmean_roofline",
    )
    candidates = _table(
        [
            _candidate(
                method_name="feature_diffmean_roofline_encdec",
                core_prefix_id="held1",
                candidate_id="c1",
            ),
        ]
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert out is candidates


def test_guard_red_leaked_optimize_prefix_is_accepted(tmp_path: Path) -> None:
    """RED-proof: the scored set drives the guard, so leaked optimize coverage passes.

    The correct artifact excludes optimize prefixes, so an optimize row raises.
    Re-writing the SAME artifact to also include an optimize prefix makes the
    guard pass, proving the guard is genuinely keyed on the persisted scored set.
    """
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    scored_path = paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair")
    candidates = _fair_candidates(["opt1", "held1"])

    write_ours_gcg_scored_prefixes(
        scored_path, ["held1"], concept_id=_CONCEPT_ID, mode=_POS, family="fair"
    )
    with pytest.raises(AxBenchBpbAdvError, match="precomputed-bag coverage guard"):
        require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)

    # Leak: the optimize prefix sneaks into the scored set.
    write_ours_gcg_scored_prefixes(
        scored_path,
        ["held1", "opt1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert out is candidates


def test_guard_no_bag_methods_is_identity(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    candidates = _table(
        [
            _candidate(
                method_name="diffmean_decoder_only",
                core_prefix_id="p1",
                candidate_id="c1",
                target_model="decoder_only",
            )
        ]
    )
    guarded = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    # No precomputed-bag rows => returned unchanged, no artifact read attempted.
    assert guarded is candidates


def test_guard_missing_artifact_raises(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    candidates = _fair_candidates(["held1"])  # bag row, but no artifact derived
    with pytest.raises(AxBenchBpbAdvError, match="missing precomputed-bag scored-prefix"):
        require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)


def test_guard_logs_kept_and_dropped_counts_before_raise(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    stats: dict[str, int] = {}
    with pytest.raises(AxBenchBpbAdvError, match="precomputed-bag coverage guard"):
        require_precomputed_bag_candidates_cover_scored_prefixes(
            _fair_candidates(["opt1", "held1"]), paths=paths, stats=stats
        )
    assert stats == {
        "precomputed_bag_coverage_kept": 1,
        "precomputed_bag_coverage_dropped": 1,
    }
    messages = " ".join(r.message for r in caplog.records)
    assert "precomputed-bag coverage guard" in messages
    assert "kept 1 dropped 1" in messages


# ─────────────────────────────────────────────────────────────────────────────
# Section C — the REAL derive persists the scored set; the guard accepts exactly it
# ─────────────────────────────────────────────────────────────────────────────


def test_fair_derive_persists_held_out_and_guard_accepts_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL fair derive persists the held-out prefixes; the guard accepts them.

    Drives the real fair derive, then asserts (i) the persisted scored set EQUALS
    the Task-3 held-out split (disjoint from optimize), and (ii) candidates built
    only for held-out prefixes pass the coverage guard.
    """
    model = _build_model(monkeypatch)
    prefixes = _distinct_prefixes(_POS)
    cfg = _make_cfg(reg_lambda_grid=(0.0,))
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="fair")
    _run_fair(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )
    held_out_ids = {p.core_prefix_id for p in held_out_prefixes}
    optimize_ids = {p.core_prefix_id for p in optimize_prefixes}
    assert held_out_ids and optimize_ids and held_out_ids.isdisjoint(optimize_ids)

    persisted = set(
        read_ours_gcg_scored_prefixes(paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"))
    )
    assert persisted == held_out_ids

    candidates = _table(
        [
            _candidate(
                method_name=_FAIR_METHOD[_POS],
                core_prefix_id=p.core_prefix_id,
                candidate_id=f"c_{p.core_prefix_id}",
            )
            for p in held_out_prefixes
        ]
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert _survivors(out, _FAIR_METHOD[_POS]) == held_out_ids


def test_roofline_derive_persists_held_out_and_guard_accepts_them(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The REAL roofline derive persists the fair held-out prefixes; the guard accepts them."""
    model = _build_model(monkeypatch)
    prefixes = _distinct_prefixes(_POS, n=5)
    cfg = _make_cfg(reg_lambda_grid=(0.0,), split_ratio=0.6)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="roof")
    _run_roofline(monkeypatch, model=model, prefixes=prefixes, cfg=cfg, paths=paths)

    optimize_prefixes, held_out_prefixes = split_prefixes_for_concept(
        prefixes, ratio=cfg.ours_gcg.split_ratio, seed=cfg.ours_gcg.split_seed
    )
    held_out_ids = {p.core_prefix_id for p in held_out_prefixes}
    optimize_ids = {p.core_prefix_id for p in optimize_prefixes}
    assert held_out_ids and optimize_ids and held_out_ids.isdisjoint(optimize_ids)

    persisted = set(
        read_ours_gcg_scored_prefixes(paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "roofline"))
    )
    assert persisted == held_out_ids

    candidates = _table(
        [
            _candidate(
                method_name=_ROOFLINE_METHOD[_POS],
                core_prefix_id=p.core_prefix_id,
                candidate_id=f"c_{p.core_prefix_id}",
            )
            for p in held_out_prefixes
        ]
    )
    out = require_precomputed_bag_candidates_cover_scored_prefixes(candidates, paths=paths)
    assert _survivors(out, _ROOFLINE_METHOD[_POS]) == held_out_ids


# ─────────────────────────────────────────────────────────────────────────────
# Section D — the DEFAULT roundtrip score stage threads adv_paths and coverage
# ─────────────────────────────────────────────────────────────────────────────


def test_roundtrip_score_stage_threads_adv_paths_and_checks_coverage(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The default roundtrip score stage reaches the scorer with ``adv_paths`` and
    feeds it the universal-population candidates after checking bag coverage.
    """
    captured: dict[str, Any] = {}
    paths = AxBenchBpbAdvPaths(output_root=str(tmp_path / "out"), run_tag="run")
    # The derive's scored set covers the universal held-out row.
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    candidates = _table(
        [
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c2")
        ]
    )
    write_evaluation_population(
        paths.evaluation_population(),
        candidates.to_pylist(),
        ours=AxBenchBpbAdvConfig().ours_gcg,
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        target_models=["encdec"],
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        offload_between_stages=False,
    )

    class _FakeRuntimeState:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            del args, kwargs

        def encdec(self) -> Any:
            return SimpleNamespace(model=object(), tokenizer=object(), max_seq_len=128)

        def inactive_sampler(self) -> Any:
            return None

        def target_device(self) -> str:
            return "cpu"

        def offload_models(self) -> None:
            pass

    def _fake_scorer(**kwargs: Any) -> Any:
        captured["adv_paths"] = kwargs.get("adv_paths")
        return SimpleNamespace(concepts_by_id={_CONCEPT_ID: _make_concept()})

    def _fake_bpb_scores(**kwargs: Any) -> pa.Table:
        captured["scored_candidates"] = kwargs["candidates"]
        return pa.table(
            {
                "candidate_id": ["c2"],
                "loss_concept_rich": [1.0],
                "loss_concept_avoiding": [2.0],
            }
        )

    def _patch(name: str, value: Any) -> None:
        monkeypatch.setattr(f"steering.eval.axbench_bpb_adv.roundtrip_scoring.{name}", value)

    _patch("RuntimeState", _FakeRuntimeState)
    _patch("_read_concepts", lambda _sp: {_CONCEPT_ID: _make_concept()})
    _patch("_build_token_bytes", lambda _tok: {})
    _patch("load_context_segments_by_prefix_hash", lambda _paths: {})
    _patch("get_extraction_tokenizer", lambda _id: object())
    _patch("detect_target_model_metadata", lambda _m, _c: _target_meta())
    _patch("_validate_roundtrip_probe_matches_concepts", lambda **_k: None)
    _patch("ModelBackedDualCompletionScorer", _fake_scorer)
    _patch("build_bpb_adv_score_table", _fake_bpb_scores)
    _patch("build_roundtrip_generation_rows", lambda **_k: [{"candidate_id": "c2"}])
    _patch("_validate_roundtrip_stats_match_probe", lambda **_k: None)
    _patch("load_percentile_table_for_roundtrip", lambda _sp: (None, "checksum"))
    _patch("load_probe_model_for_roundtrip", lambda _p: "probe")
    _patch(
        "score_roundtrip_generation_rows",
        lambda **_k: pa.Table.from_pylist([], schema=ROUNDTRIP_SCORE_SCHEMA),
    )

    out = run_model_backed_roundtrip_score_stage(cfg=cfg, paths=paths, candidates=candidates)
    assert out.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    # (i) the scorer was constructed WITH adv_paths (the CRITICAL fix).
    assert captured["adv_paths"] is paths
    # (ii) the universal held-out gcg row reached scoring.
    scored = captured["scored_candidates"].to_pylist()
    assert [str(r["core_prefix_id"]) for r in scored] == ["held1"]


def test_roundtrip_score_stage_raises_when_bag_artifact_misses_universal_prefix(
    tmp_path: Path,
) -> None:
    """After universal checks, bag-backed family artifacts must cover all rows."""
    paths = AxBenchBpbAdvPaths(output_root=str(tmp_path / "out"), run_tag="run")
    candidates = _table(
        [
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c1"),
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held2", candidate_id="c2"),
        ]
    )
    write_evaluation_population(
        paths.evaluation_population(),
        candidates.to_pylist(),
        ours=AxBenchBpbAdvConfig().ours_gcg,
    )
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
    )

    with pytest.raises(AxBenchBpbAdvError, match="precomputed-bag coverage guard"):
        run_model_backed_roundtrip_score_stage(cfg=cfg, paths=paths, candidates=candidates)


def test_class_margin_score_stage_raises_when_decoder_only_row_is_outside_universal_population(
    tmp_path: Path,
) -> None:
    """The universal population is fail-loud for decoder-only baselines too."""
    paths = AxBenchBpbAdvPaths(output_root=str(tmp_path / "out"), run_tag="run")
    candidates = _table(
        [
            _candidate(
                method_name="control_decoder_only",
                core_prefix_id="opt1",
                candidate_id="c1",
                target_model="decoder_only",
            ),
            _candidate(
                method_name="control_decoder_only",
                core_prefix_id="held1",
                candidate_id="c2",
                target_model="decoder_only",
            ),
        ]
    )
    write_evaluation_population(
        paths.evaluation_population(),
        [row for row in candidates.to_pylist() if str(row["core_prefix_id"]) == "held1"],
        ours=AxBenchBpbAdvConfig().ours_gcg,
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        target_models=["decoder_only"],
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        offload_between_stages=False,
    )

    with pytest.raises(AxBenchBpbAdvError, match="absent from the exact held-out pair keys"):
        run_model_backed_bpb_adv_score_stage(cfg=cfg, paths=paths, candidates=candidates)


def test_class_margin_score_stage_raises_when_bag_artifact_misses_universal_prefix(
    tmp_path: Path,
) -> None:
    """Class-margin scoring refuses to score a stale bag-backed subset."""
    paths = AxBenchBpbAdvPaths(output_root=str(tmp_path / "out"), run_tag="run")
    candidates = _table(
        [
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c1"),
            _candidate(method_name="ours_gcg_encdec", core_prefix_id="held2", candidate_id="c2"),
        ]
    )
    write_evaluation_population(
        paths.evaluation_population(),
        candidates.to_pylist(),
        ours=AxBenchBpbAdvConfig().ours_gcg,
    )
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
    )

    with pytest.raises(AxBenchBpbAdvError, match="precomputed-bag coverage guard"):
        run_model_backed_bpb_adv_score_stage(cfg=cfg, paths=paths, candidates=candidates)


def _target_meta() -> Any:
    return TargetModelMetadata(
        num_buckets=_NUM_BUCKETS,
        n_sae_features=None,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=256,
    )


def _make_scorer(paths: AxBenchBpbAdvPaths, tmp_path: Path) -> ModelBackedDualCompletionScorer:
    """A scorer wired with ``adv_paths`` (only the gcg intervention seam is exercised)."""
    return ModelBackedDualCompletionScorer(
        # Legacy (uncompacted) model stand-in for the scorer's checkpoint-V verify
        # (issue #695 PR-1 follow-up); every bag here is the legacy (unstamped)
        # shape via _write_bag, so this keeps the tests byte-identical to
        # pre-checkpoint-V-stamping behavior.
        target_bundles={
            "encdec": TargetModelBundle(
                model=SimpleNamespace(feature_vocab=None),
                tokenizer=SimpleNamespace(),
                max_seq_len=128,
                target_name="encdec",
            )
        },
        target_model_metas={
            "encdec": SimpleNamespace(negative_selection=None, num_buckets=_NUM_BUCKETS)
        },
        concepts_by_id={_CONCEPT_ID: _make_concept()},
        source_paths=SourcePaths(source_run_dir=tmp_path),
        tokenizer=SimpleNamespace(),
        device=t.device("cpu"),
        diffmean_layer=8,
        inactive_sampler=None,
        adv_paths=paths,
        apply_max_act_scale=False,
    )


def test_score_stage_missing_bag_for_scored_prefix_raises(tmp_path: Path) -> None:
    """A SCORED prefix whose bag is missing RAISES through the real scorer seam.

    With the guard, only prefixes-with-bags reach the scorer; a present bag
    builds the steered intervention (no crash), and DELETING that bag —
    simulating a stale derive — makes the scorer RAISE loudly.
    """
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    scorer = _make_scorer(paths, tmp_path)
    method = get_method_by_name("ours_gcg_encdec")
    concept = _make_concept()
    row = _candidate(method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c1")

    bag_path = paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair")
    _write_bag(bag_path, {42: 0})
    intervention = scorer._steered_intervention(
        method=method, concept=concept, coefficient=0.0, candidate_row=row
    )
    assert intervention.kind == InterventionKind.OURS
    assert intervention.feature_conditions == {42: 0}

    bag_path.unlink()
    with pytest.raises(AxBenchBpbError, match="precomputed bag not found"):
        scorer._steered_intervention(
            method=method, concept=concept, coefficient=0.0, candidate_row=row
        )


def test_score_stage_scored_prefix_with_bag_builds_intervention(tmp_path: Path) -> None:
    """A scored prefix WITH its bag builds Intervention.ours (adv_paths reaches the
    scorer; the no-crash half of the CRITICAL)."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    scorer = _make_scorer(paths, tmp_path)
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {7: 1, 9: 0})
    intervention = scorer._steered_intervention(
        method=get_method_by_name("ours_gcg_encdec"),
        concept=_make_concept(),
        coefficient=0.0,
        candidate_row=_candidate(
            method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c1"
        ),
    )
    assert intervention.kind == InterventionKind.OURS
    assert intervention.feature_conditions == {7: 1, 9: 0}


def test_scorer_without_adv_paths_crashes_on_gcg_method(tmp_path: Path) -> None:
    """Regression for the CRITICAL: a scorer built WITHOUT adv_paths (the pre-fix
    roundtrip path) crashes on a gcg method — pinning why adv_paths must be threaded."""
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {42: 0})
    scorer = ModelBackedDualCompletionScorer(
        # Legacy model stand-in (see _make_scorer above) -- this test's RED-proof
        # is specifically about adv_paths=None, not about the target bundle, so it
        # must not trip the (unrelated) missing-target-bundle guard first.
        target_bundles={
            "encdec": TargetModelBundle(
                model=SimpleNamespace(feature_vocab=None),
                tokenizer=SimpleNamespace(),
                max_seq_len=128,
                target_name="encdec",
            )
        },
        target_model_metas={
            "encdec": SimpleNamespace(negative_selection=None, num_buckets=_NUM_BUCKETS)
        },
        concepts_by_id={_CONCEPT_ID: _make_concept()},
        source_paths=SourcePaths(source_run_dir=tmp_path),
        tokenizer=SimpleNamespace(),
        device=t.device("cpu"),
        diffmean_layer=8,
        inactive_sampler=None,
        adv_paths=None,  # the bug the CRITICAL was about
        apply_max_act_scale=False,
    )
    with pytest.raises(AxBenchBpbError, match="adv_paths"):
        scorer._steered_intervention(
            method=get_method_by_name("ours_gcg_encdec"),
            concept=_make_concept(),
            coefficient=0.0,
            candidate_row=_candidate(
                method_name="ours_gcg_encdec", core_prefix_id="held1", candidate_id="c1"
            ),
        )
