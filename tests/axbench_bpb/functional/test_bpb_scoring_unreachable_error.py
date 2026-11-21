"""bpb_scoring.py:194 raises RuntimeError if enumerate_work_units yields
a unit for a sample_id absent from samples_by_concept_mode. The branch
is defensive ("should be unreachable") but coverage matters — if a
future schema change makes this reachable in production, the test
fires and forces an explicit re-derivation.

Approach: monkeypatch enumerate_work_units to yield a synthetic orphan
WorkUnit referencing a sample_id not present in samples_by_concept_mode.
is_mine is also stubbed to True so the orphan reaches the lookup. The
real BpbCache is used (its has() returns False on a fresh cache, so
control proceeds past the cache-skip and hits the orphan check).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from steering.eval.axbench_bpb import bpb_scoring as scoring_module
from steering.eval.axbench_bpb.bpb_scoring import run_bpb_score_stage
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import SteeringMode, WorkUnit


@dataclass(frozen=True)
class _FakeMethod:
    name: str
    mode: SteeringMode
    target_model: str


@dataclass(frozen=True)
class _FakeConcept:
    concept_id: int


@dataclass
class _FakeOutputPaths:
    root: Path

    def bpb_scores_rank_dir(self, rank: int) -> Path:
        d = self.root / "bpb_scores" / f"rank_{rank:04d}"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def bpb_cache_path(self, rank: int) -> Path:
        return self.root / "bpb_scores" / f"rank_{rank:04d}.sqlite"


@dataclass
class _Cfg:
    log_progress_every: int = 1000
    feature_statistics_path: str = "unused.safetensors"


def test_run_bpb_score_stage_raises_runtime_error_on_orphan_workunit(
    tmp_path, monkeypatch,
):
    """Orphan WorkUnit (sample_id not in samples_by_concept_mode) → RuntimeError.

    enumerate_work_units is monkeypatched to yield a WorkUnit whose
    sample_id ("missing") is not in the (empty) samples list for the
    same (concept_id, mode) key. The check at bpb_scoring.py:192-198
    catches this and raises RuntimeError with a descriptive message.
    """
    method = _FakeMethod(
        name="m",
        mode=SteeringMode.POSITIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=0)
    # Empty list means next(...) returns None → orphan check fires.
    samples_by_concept_mode = {(0, SteeringMode.POSITIVE): []}
    coefficient_grids = {"m": ["1"]}

    def _yield_orphan(**kwargs):
        yield WorkUnit(
            method_name="m",
            concept_id=0,
            sample_id="missing",
            coefficient_str="1",
        )

    monkeypatch.setattr(scoring_module, "enumerate_work_units", _yield_orphan)
    monkeypatch.setattr(
        scoring_module, "is_mine", lambda unit, **kw: True,
    )
    monkeypatch.setattr(
        scoring_module, "reconcile_cache", lambda **kw: None,
    )

    output_paths = _FakeOutputPaths(root=tmp_path)
    # RuntimeError fires before any model/tokenizer field is touched, so
    # placeholder None values are fine. TargetModelBundle is a frozen
    # dataclass so a real instance satisfies the type annotation.
    bundle = TargetModelBundle(
        model=None, tokenizer=None, max_seq_len=8, target_name="decoder_only",
    )

    with pytest.raises(RuntimeError, match=r"work unit.*has no matching sample"):
        run_bpb_score_stage(
            cfg=_Cfg(),
            output_paths=output_paths,
            source_paths=None,
            methods=[method],
            concepts=[concept],
            samples_by_concept_mode=samples_by_concept_mode,
            coefficient_grids=coefficient_grids,
            decoder_only_bundle=bundle,
            encdec_bundle=bundle,
            target_model_metas={"decoder_only": None, "encdec": None},
            tokenizer_nanochat=None,
            device=torch.device("cpu"),
            ddp_rank=0,
            ddp_world_size=1,
            diffmean_layer=4,
        )


def test_run_bpb_score_stage_orphan_message_includes_concept_and_mode(
    tmp_path, monkeypatch,
):
    """The error message identifies which (concept_id, mode) pair was
    missing the sample — operator diagnosability matters when the
    "unreachable" branch ever fires in production.
    """
    method = _FakeMethod(
        name="m",
        mode=SteeringMode.NEGATIVE,
        target_model="decoder_only",
    )
    concept = _FakeConcept(concept_id=42)
    samples_by_concept_mode = {(42, SteeringMode.NEGATIVE): []}
    coefficient_grids = {"m": ["1"]}

    def _yield_orphan(**kwargs):
        yield WorkUnit(
            method_name="m",
            concept_id=42,
            sample_id="ghost",
            coefficient_str="1",
        )

    monkeypatch.setattr(scoring_module, "enumerate_work_units", _yield_orphan)
    monkeypatch.setattr(
        scoring_module, "is_mine", lambda unit, **kw: True,
    )
    monkeypatch.setattr(
        scoring_module, "reconcile_cache", lambda **kw: None,
    )

    output_paths = _FakeOutputPaths(root=tmp_path)
    bundle = TargetModelBundle(
        model=None, tokenizer=None, max_seq_len=8, target_name="decoder_only",
    )

    with pytest.raises(RuntimeError) as exc_info:
        run_bpb_score_stage(
            cfg=_Cfg(),
            output_paths=output_paths,
            source_paths=None,
            methods=[method],
            concepts=[concept],
            samples_by_concept_mode=samples_by_concept_mode,
            coefficient_grids=coefficient_grids,
            decoder_only_bundle=bundle,
            encdec_bundle=bundle,
            target_model_metas={"decoder_only": None, "encdec": None},
            tokenizer_nanochat=None,
            device=torch.device("cpu"),
            ddp_rank=0,
            ddp_world_size=1,
            diffmean_layer=4,
        )
    msg = str(exc_info.value)
    # Concept_id and the mode string both surface — operator can grep
    # logs to localize the orphan source.
    assert "42" in msg
    assert "negative" in msg.lower()
