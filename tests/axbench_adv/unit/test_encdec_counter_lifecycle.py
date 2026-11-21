"""Public routing, provenance and independent score oracles for the EncDec lane."""

from __future__ import annotations

import json
import runpy
import subprocess
from pathlib import Path
from typing import Any

import hydra
import pytest
from omegaconf import DictConfig, OmegaConf

from steering.eval.axbench_adv import encdec_baseline_counter as campaign
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gepa_optimization import RawEvaluation
from steering.eval.axbench_adv.native_mailbox import read_json, write_json
from steering.eval.axbench_adv.types import ENCDEC_BASELINE_METHODS
from steering.eval.axbench_upstream import run_methods
from steering.immutable_io import ImmutableArtifactError
from tests.axbench_adv.unit.test_encdec_baseline_counter import (
    artifact_campaign as artifact_campaign,
)

ROOT = Path(__file__).resolve().parents[3]


@pytest.fixture
def routed(artifact_campaign: Any, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Any:
    run = artifact_campaign
    run.cfg = OmegaConf.merge(
        OmegaConf.load(ROOT / "configs/eval/axbench_adv/encdec_baseline_counter.yaml"),
        run.cfg,
        {"output_dir": str(tmp_path / "routed"), "concept_ids": [0], "counter_magnitudes": [256]},
    )
    monkeypatch.setattr(campaign, "validate_encdec_sources", lambda **_k: run.source)
    monkeypatch.setattr(campaign, "_counter_sources", lambda *_a: run.counter)
    return run


def test_preflight_does_not_require_gpu_or_write_output(
    routed: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(campaign, "_runtime", lambda *_a: pytest.fail("preflight queried GPU"))
    result = campaign.run_encdec_baseline_counter(routed.cfg)
    assert result == {
        "phase": "preflight_complete",
        "source": routed.source.source_digest,
        "counter_manifest_sha256": routed.counter.manifest_sha256,
        "best_factors": routed.source.best_factors,
    }
    assert not Path(routed.cfg.output_dir).exists()


@pytest.mark.parametrize("phase", ["pilot", "generate"])
def test_generation_router_installs_identity_before_dispatch(
    routed: Any, monkeypatch: pytest.MonkeyPatch, phase: str,
) -> None:
    routed.cfg.phase = phase
    routed.cfg.pretrain_meta_path = "pretrain.json"
    runtime = {"device_type": "cpu", "version": "qualified"}
    monkeypatch.setattr(campaign, "_runtime", lambda *_a: runtime)

    def dispatch(cfg: DictConfig, source: Any, counter: Any, output: Path,
                 identity: dict[str, Any]) -> dict[str, Any]:
        assert cfg is routed.cfg and source is routed.source and counter is routed.counter
        assert read_json(output / "identity.json") == identity
        assert identity["phase"] == phase
        assert identity["runtime"] == runtime
        assert identity["concept_ids"] == [0] and identity["magnitudes"] == [256.0]
        return {"phase": "dispatched"}

    monkeypatch.setattr(campaign, "generate", dispatch)
    assert campaign.run_encdec_baseline_counter(routed.cfg) == {"phase": "dispatched"}


def test_router_rejects_unknown_phase_and_multi_concept_pilot(routed: Any) -> None:
    routed.cfg.phase = "typo"
    with pytest.raises(AxBenchAdvError, match="phase must be"):
        campaign.run_encdec_baseline_counter(routed.cfg)
    routed.cfg.phase = "pilot"
    routed.cfg.concept_ids = None
    with pytest.raises(AxBenchAdvError, match="pilot requires one concept"):
        campaign.run_encdec_baseline_counter(routed.cfg)
    assert not Path(routed.cfg.output_dir).exists()


def test_cpu_judge_and_finalize_retain_generation_runtime_and_resume_without_calls(
    routed: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    root, _ = routed.finish([0], "finished")
    routed.cfg.output_dir = str(root)
    routed.cfg.device = "cuda:unavailable-on-judge-host"
    monkeypatch.setattr(campaign, "_runtime", lambda *_a: pytest.fail("judge queried GPU"))
    calls = list(routed.calls)
    for phase in ("judge", "finalize"):
        routed.cfg.phase = phase
        result = campaign.run_encdec_baseline_counter(routed.cfg)
        assert result.get("status") == "complete" or result.get("completed") == 120
        assert routed.calls == calls
    identity = read_json(root / "identity.json")
    del identity["runtime"]
    write_json(root / "identity.json", identity)
    with pytest.raises(AxBenchAdvError, match="no bound GPU runtime"):
        campaign.run_encdec_baseline_counter(routed.cfg)


def test_router_rejects_changed_scope_before_judging(routed: Any) -> None:
    root, _ = routed.finish([0], "finished")
    routed.cfg.output_dir = str(root)
    routed.cfg.phase = "judge"
    routed.cfg.concept_ids = [1]
    calls = list(routed.calls)
    with pytest.raises(ImmutableArtifactError):
        campaign.run_encdec_baseline_counter(routed.cfg)
    assert routed.calls == calls


def test_merge_router_uses_validated_sources_without_gpu(routed: Any,
                                                       monkeypatch: pytest.MonkeyPatch) -> None:
    routed.cfg.phase = "merge"
    monkeypatch.setattr(campaign, "_runtime", lambda *_a: pytest.fail("merge queried GPU"))

    def merge(cfg: Any, sources: Any, counter: Any) -> dict[str, Any]:
        assert (cfg, sources, counter) == (routed.cfg, routed.source, routed.counter)
        return {"status": "merged"}

    monkeypatch.setattr(campaign, "merge", merge)
    assert campaign.run_encdec_baseline_counter(routed.cfg) == {"status": "merged"}


def test_cli_passes_hydra_config_and_prints_strict_json(routed: Any,
                                                    monkeypatch: pytest.MonkeyPatch,
                                                    capsys: pytest.CaptureFixture[str]) -> None:
    # Exercise the executable guard with Hydra's argument parsing isolated; the
    # config itself is the shipped YAML and the driver is the actual preflight.
    monkeypatch.setattr(hydra, "main", lambda **_k: lambda fn: lambda: fn(routed.cfg))
    imported = runpy.run_module("scripts.eval.axbench_adv.run_encdec_baseline_counter")
    assert callable(imported["main"])
    assert capsys.readouterr().out == ""
    runpy.run_module("scripts.eval.axbench_adv.run_encdec_baseline_counter", run_name="__main__")
    assert json.loads(capsys.readouterr().out)["phase"] == "preflight_complete"


def test_resume_identity_binds_transitive_dirty_and_committed_code(
    artifact_campaign: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    repo = tmp_path / "repository"
    repo.mkdir()

    def git(*args: str) -> str:
        return subprocess.check_output(["git", *args], cwd=repo, text=True).strip()

    git("init", "--quiet")
    helper = repo / "prompt_prep.py"
    helper.write_text("selection = 1\n")
    git("add", "prompt_prep.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "initial")
    monkeypatch.setattr(run_methods, "REPO_ROOT", repo)
    monkeypatch.setattr(campaign, "_git_commit", run_methods._git_commit)
    monkeypatch.setattr(campaign, "_repository_source_sha256", run_methods._repository_source_sha256)
    run = artifact_campaign

    def identity() -> dict[str, Any]:
        return campaign._identity(cfg=run.cfg, sources=run.source, counter=run.counter,
                                  concepts=[0], magnitudes=[256.0], phase="generate",
                                  runtime={"device_type": "cpu"})

    output = tmp_path / "output"
    original = identity()
    campaign._install_identity(output, original)
    campaign._install_identity(output, identity())
    helper.write_text("selection = 2\n")
    dirty = identity()
    assert dirty != original
    with pytest.raises(ImmutableArtifactError):
        campaign._install_identity(output, dirty)
    git("add", "prompt_prep.py")
    git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "--quiet", "-m", "changed")
    committed = identity()
    assert committed not in (original, dirty)
    with pytest.raises(ImmutableArtifactError):
        campaign._install_identity(output, committed)
    new_helper = repo / "sampling.py"
    new_helper.write_text("temperature = 1\n")
    untracked = identity()
    new_helper.write_text("temperature = 2\n")
    assert identity() != untracked != committed


@pytest.mark.parametrize("phase", ["judge", "finalize"])
@pytest.mark.parametrize("field,value", [("allow_live_provider", "false"),
                                        ("allow_live_provider", 1),
                                        ("cache_only", "false"), ("cache_only", None)])
def test_provider_policy_requires_literal_booleans(
    artifact_campaign: Any, phase: str, field: str, value: Any,
) -> None:
    run = artifact_campaign
    root, identity = run.finish([0], "finished")
    (root / "manifest.json").unlink()
    run.cfg.cache_only = False
    run.cfg.allow_live_provider = True
    run.cfg[field] = value
    calls = list(run.calls)
    with pytest.raises(AxBenchAdvError, match="Boolean"):
        if phase == "judge":
            campaign.judge(run.cfg, run.source, root, identity)
        else:
            campaign.finalize(run.cfg, run.source, run.counter, root, identity)
    assert run.calls == calls


@pytest.mark.parametrize("cache_only,allow_live", [(True, False), (True, True), (False, True), (False, False)])
def test_provider_policy_authorizes_only_explicit_live_opt_in(cache_only: bool, allow_live: bool) -> None:
    cfg = OmegaConf.create({"cache_only": cache_only, "allow_live_provider": allow_live})
    if not cache_only and not allow_live:
        with pytest.raises(AxBenchAdvError, match="requires allow_live_provider=true"):
            campaign._judge_cache_only(cfg)
    else:
        assert campaign._judge_cache_only(cfg) is cache_only


def test_incomplete_output_and_pilot_cannot_be_final_results(artifact_campaign: Any, tmp_path: Path) -> None:
    output = tmp_path / "unidentified"
    output.mkdir()
    (output / "generation.json").write_text("{}")
    with pytest.raises(AxBenchAdvError, match="without a run identity"):
        campaign._install_identity(output, {"phase": "generate"})
    assert not (output / "identity.json").exists()
    run = artifact_campaign
    with pytest.raises(AxBenchAdvError, match="not a one-input pilot"):
        campaign.finalize(run.cfg, run.source, run.counter, output, {"phase": "pilot"})
    assert not (output / "manifest.json").exists()
    assert run.calls == []


@pytest.mark.parametrize("all_zero", [False, True])
def test_score_tables_follow_literal_macro_sem_harmonic_and_fitness_oracles(
    tmp_path: Path, all_zero: bool,
) -> None:
    factors = dict.fromkeys(ENCDEC_BASELINE_METHODS, 1.0)
    triples = [(0.0, 2.0, 1.0), (2.0, 1.0, 0.5)] if not all_zero else [(0.0, 0.0, 0.0)]
    entries, emissions = [], {}
    for cid, (concept, fluency, instruction) in enumerate(triples):
        entries.append({"concept_id": cid, "results": {"LMJudgeEvaluator": {
            method: {"factor": [1.0], "relevance_concept_ratings": [concept],
                     "fluency_ratings": [fluency], "relevance_instruction_ratings": [instruction]}
            for method in factors
        }}})
        for method in factors:
            for iid in range(10):
                emissions[(method, cid, iid)] = RawEvaluation(
                    "A complete substantive response.", concept, fluency, "unused for fitness", 0,
                )
    path = tmp_path / "steering.jsonl"
    path.write_text("".join(json.dumps(entry) + "\n" for entry in entries))
    rows, per_concept = campaign._point_scores(
        path, emissions=emissions, factors=factors, concepts=tuple(range(len(triples))),
        pressure="diffmean_counter", magnitude=256.0,
    )
    assert len(rows) == 4 and len(per_concept) == 4 * len(triples)
    for row in rows:
        expected = (0.0,) * 9 if all_zero else (1.0, 1.5, 0.75, 1.0, 0.5, 0.25, 1.2, 1.0, 0.5)
        assert tuple(row[key] for key in (
            "concept_score", "fluency_score", "instruction_score", "concept_score_sem",
            "fluency_score_sem", "instruction_score_sem", "harmonic_quality_2",
            "harmonic_quality_3", "mean_fitness",
        )) == pytest.approx(expected)
        assert row["n_inputs"] == 10 * len(triples)
    for row in per_concept:
        assert row["harmonic_quality_2"] == pytest.approx(0.0 if row["concept_id"] == 0 else 4 / 3)
        assert row["harmonic_quality_3"] == pytest.approx(0.0 if row["concept_id"] == 0 else 6 / 7)
