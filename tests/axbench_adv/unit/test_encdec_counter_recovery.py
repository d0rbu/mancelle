"""Failure and recovery contracts for durable encoder-decoder counter artifacts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow.parquet as pq
import pytest
from omegaconf import DictConfig

from steering.eval.axbench_adv import encdec_baseline_counter as campaign
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.native_mailbox import read_json, write_json
from tests.axbench_adv.unit.test_encdec_baseline_counter import _source
from tests.axbench_adv.unit.test_encdec_baseline_counter import (
    artifact_campaign as artifact_campaign,
)


@pytest.mark.parametrize("filename", ["aggregate.parquet", "per_concept.parquet"])
def test_finalization_interruption_resumes_without_repeating_judges(
    artifact_campaign: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, filename: str,
) -> None:
    run = artifact_campaign
    root = tmp_path / "shard"
    write = campaign.atomic_write_parquet

    def interrupted_write(table: Any, path: Path) -> None:
        write(table, path)
        if path == root / filename:
            raise OSError("process interrupted after atomic parquet publication")

    with monkeypatch.context() as patch:
        patch.setattr(campaign, "atomic_write_parquet", interrupted_write)
        with pytest.raises(OSError, match="process interrupted"):
            run.finish([0], "shard")
    assert (root / filename).is_file()
    assert not (root / "manifest.json").exists()
    assert len([call for call in run.calls if call[0] == "final"]) == 3
    committed_points = {path: path.read_bytes() for path in (root / "points").rglob("*") if path.is_file()}
    calls = list(run.calls)
    identity = read_json(root / "identity.json")
    campaign.judge(run.cfg, run.source, root, identity)
    result = campaign.finalize(run.cfg, run.source, run.counter, root, identity)
    assert result["status"] == "complete"
    assert run.calls == calls
    assert committed_points == {path: path.read_bytes() for path in committed_points}
    assert len(pq.read_table(root / "aggregate.parquet")) == 12
    assert len(pq.read_table(root / "per_concept.parquet")) == 12
    assert campaign.finalize(run.cfg, run.source, run.counter, root, identity) == result
    assert run.calls == calls


@pytest.mark.parametrize("field,value", [
    ("paid_judge_cache_misses", -1),
    ("paid_judge_cache_misses", 1),
    ("paid_judge_cache_misses", False),
    ("paid_judge_cache_misses", 0.0),
    ("actual_evaluator_calls", 120.0),
])
def test_unfinished_finalization_revalidates_receipt_call_accounting(
    artifact_campaign: Any, field: str, value: Any,
) -> None:
    run = artifact_campaign
    root, identity = run.finish([0], "shard")
    (root / "manifest.json").unlink()
    judged = campaign._point_path(root, "shared_zero", 0.0) / "judged"
    assert read_json(judged / "judge_call_accounting.json") == {
        "total_calls": 120, "total_cache_hits": 120,
    }
    path = judged / "manifest.json"
    manifest = read_json(path)
    manifest["call_accounting"][field] = value
    write_json(path, manifest)
    calls = list(run.calls)
    with pytest.raises(AxBenchAdvError, match="final judge manifest differs"):
        campaign.finalize(run.cfg, run.source, run.counter, root, identity)
    assert not (root / "manifest.json").exists()
    assert run.calls == calls


@pytest.mark.parametrize("mutation", [
    "input", "manifest", "unfinished_manifest", "judgment", "judged_symlink", "runtime",
])
def test_finalization_rejects_malformed_dependencies_without_rejudging(
    artifact_campaign: Any, tmp_path: Path, mutation: str,
) -> None:
    run = artifact_campaign
    root, identity = run.finish([0], "shard")
    point = campaign._point_path(root, "shared_zero", 0.0)
    expected = {
        "input": "judged point input differs",
        "manifest": "final judge manifest differs",
        "unfinished_manifest": "final judge manifest differs",
        "judgment": "judgment record population",
        "judged_symlink": "unsupported entry",
        "runtime": "shard has no runtime identity",
    }[mutation]
    if mutation == "input":
        path = point / "input/steering_data.parquet"
        frame = pq.read_table(path).to_pandas()
        frame.loc[0, "EncDecLoRA_steered_generation"] = "changed prompt result"
        frame.to_parquet(path, index=False)
    elif mutation in {"manifest", "unfinished_manifest"}:
        path = point / "judged/manifest.json"
        manifest = read_json(path)
        manifest["call_accounting"]["actual_evaluator_calls"] += 1
        write_json(path, manifest)
        if mutation == "unfinished_manifest":
            (root / "manifest.json").unlink()
    elif mutation == "judgment":
        (root / "judgments/extra.json").write_text("{}")
    elif mutation == "judged_symlink":
        target = tmp_path / "other-plot.pdf"
        path = point / "judged/plot.pdf"
        path.rename(target)
        path.symlink_to(target)
    else:
        retained = read_json(root / "identity.json")
        retained.pop("runtime")
        write_json(root / "identity.json", retained)
    calls = list(run.calls)
    with pytest.raises(AxBenchAdvError, match=expected):
        campaign.finalize(run.cfg, run.source, run.counter, root, identity)
    assert run.calls == calls


@pytest.mark.parametrize("tree", [
    "generations", "judgments", "points", "pressure", "point", "input", "input_file", "judged",
])
def test_finished_shard_rejects_relocated_tree_symlink(
    artifact_campaign: Any, tmp_path: Path, tree: str,
) -> None:
    run = artifact_campaign
    root, identity = run.finish([0], "shard")
    point = campaign._point_path(root, "shared_zero", 0.0)
    path = {
        "generations": root / "generations",
        "judgments": root / "judgments",
        "points": root / "points",
        "pressure": point.parent,
        "point": point,
        "input": point / "input",
        "input_file": point / "input/steering_data.parquet",
        "judged": point / "judged",
    }[tree]
    target = tmp_path / "relocated-tree"
    path.rename(target)
    path.symlink_to(target, target_is_directory=target.is_dir())
    calls = list(run.calls)
    with pytest.raises(AxBenchAdvError, match="symlink|unsupported entry"):
        campaign.finalize(run.cfg, run.source, run.counter, root, identity)
    assert run.calls == calls


def test_artifact_tree_requires_real_directory_and_regular_descendants(tmp_path: Path) -> None:
    root = tmp_path / "missing"
    with pytest.raises(AxBenchAdvError, match="missing artifact tree"):
        campaign._files(root)
    root.mkdir()
    target = tmp_path / "target"
    target.write_bytes(b"same content does not make a symlink a committed record")
    (root / "record.json").symlink_to(target)
    with pytest.raises(AxBenchAdvError, match="unsupported entry"):
        campaign._files(root)


@pytest.mark.parametrize("value", [None, 7, ["a generation"]])
def test_nontext_generation_cannot_commit_a_cell(tmp_path: Path, value: Any) -> None:
    _, cells = campaign._cells(_source(), [0], pilot=True)
    with pytest.raises(AxBenchAdvError, match="generation must be text"):
        campaign._record(
            output=tmp_path, identity_sha="a" * 64, cell=cells[0], pressure="shared_zero",
            magnitude=0.0, generate=lambda: value,
        )
    assert not (tmp_path / "generations").exists()


@pytest.mark.parametrize("seed", [43, True, 42.0])
def test_campaign_counter_fit_seed_is_exact_integer_42(seed: Any) -> None:
    cfg = cast(DictConfig, SimpleNamespace(counter_layer=9, counter_fit_seed=seed))
    with pytest.raises(AxBenchAdvError, match="counter fit seed must be 42"):
        campaign._counter_sources(cfg, _source())


def test_merge_requires_completed_shards(artifact_campaign: Any) -> None:
    run = artifact_campaign
    run.cfg.shard_roots = []
    with pytest.raises(AxBenchAdvError, match="at least one completed shard"):
        campaign.merge(run.cfg, run.source, run.counter)
    assert not run.calls


def test_merge_rejects_overlapping_or_occupied_destination(
    artifact_campaign: Any, tmp_path: Path,
) -> None:
    run = artifact_campaign
    root, _ = run.finish(list(range(50)), "shard")
    run.cfg.shard_roots = [str(root)]
    calls = list(run.calls)
    marker = (root / "manifest.json").read_bytes()
    for output in (root, root / "merged", tmp_path):
        run.cfg.output_dir = str(output)
        with pytest.raises(AxBenchAdvError, match="disjoint from every shard"):
            campaign.merge(run.cfg, run.source, run.counter)
    occupied = tmp_path / "occupied"
    occupied.mkdir()
    sentinel = occupied / "unrelated-file"
    sentinel.write_bytes(b"preserve me")
    run.cfg.output_dir = str(occupied)
    with pytest.raises(AxBenchAdvError, match="merge output already exists"):
        campaign.merge(run.cfg, run.source, run.counter)
    assert sentinel.read_bytes() == b"preserve me"
    assert not (occupied / "manifest.json").exists()
    assert (root / "manifest.json").read_bytes() == marker
    assert run.calls == calls


def test_merge_rejects_individually_valid_shards_with_different_runtimes(
    artifact_campaign: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    run = artifact_campaign
    left, _ = run.finish([0], "left")
    identity = campaign._identity

    def other_runtime(**kwargs: Any) -> dict[str, Any]:
        kwargs["runtime"] = {"device_type": "cuda", "gpu_name": "other fixture hardware"}
        return identity(**kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(campaign, "_identity", other_runtime)
        right, _ = run.finish(list(range(1, 50)), "right")
    # Both manifests, records, source bindings and derived scores are valid;
    # combining their runtime-specific positive controls is the rejected act.
    run.cfg.shard_roots = [str(left), str(right)]
    run.cfg.output_dir = str(tmp_path / "merged")
    calls = list(run.calls)
    with pytest.raises(AxBenchAdvError, match="incompatible identity or hardware"):
        campaign.merge(run.cfg, run.source, run.counter)
    assert not Path(run.cfg.output_dir).exists()
    assert run.calls == calls
