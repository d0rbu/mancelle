"""Completed soft-run boundaries: zero-only controls and incompatible slices."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq
import pytest

from scripts.eval.axbench_adv.merge_slices import merge_slices
from steering.eval.axbench_adv.errors import AxBenchAdvError
from steering.eval.axbench_adv.gcg_config import GcgObjective
from tests.axbench_adv.functional.test_gcg_contract_review import _write_training_source
from tests.axbench_adv.functional.test_soft_prompt_pipeline import _run_soft, _SoftMixin


def test_zero_only_l2_completed_slices_publish_and_merge_without_soft_runtime(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    prepared = tmp_path / "prepared"
    _write_training_source(prepared)

    def unexpected_prepare(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("zero-only control attempted soft preparation")

    monkeypatch.setattr(_SoftMixin, "prepare_soft_prompt", unexpected_prepare)
    roots = [
        _run_soft(
            tmp_path,
            monkeypatch,
            prepared,
            f"zero-{index}",
            concepts=(index, index + 1),
            counts=(0,),
            objective=GcgObjective.ACTIVATION_L2,
        )
        for index in range(2)
    ]
    for root in roots:
        manifest = json.loads((root / "manifest.json").read_text())
        assert manifest["soft_prompt"]["optimized_token_counts"] == [0]
        accounting = manifest["call_accounting"]
        assert accounting["control_generation_calls"] == 20
        assert accounting["actual_generation_calls"] == 0
        assert accounting["actual_evaluator_calls"] == 0
        assert not list(root.rglob("*.safetensors"))
        assert not (root / "soft_prompt/points").exists()
        receipts = list(root.glob("soft_prompt/*/c*/k0/search.json"))
        assert len(receipts) == 2
        for path in receipts:
            assert json.loads(path.read_text())["diagnostics"] == {
                "control": True,
                "updates": 0,
                "best_step": 0,
                "full_population_scores": 0,
            }
    merged = merge_slices(roots, tmp_path / "merged")
    aggregate = pq.read_table(merged.parent / "soft_prompt/aggregate.parquet").to_pylist()
    assert len(aggregate) == 2
    assert {row["method"] for row in aggregate} == {"DiffMean", "SelfConditioning"}
    assert all(row["token_count"] == 0 and row["n_concepts"] == 2 for row in aggregate)
    assert not list(merged.parent.rglob("*.safetensors"))


@pytest.mark.parametrize("difference", ["learning_rate", "training_source"])
def test_merge_rejects_individually_valid_slices_from_different_searches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, difference: str
) -> None:
    prepared = tmp_path / "prepared"
    _write_training_source(prepared)
    first = _run_soft(
        tmp_path,
        monkeypatch,
        prepared,
        "first",
        methods=("DiffMean",),
        concepts=(0, 1),
        counts=(0, 1),
    )
    if difference == "training_source":
        # Both completed runs are legitimate; the source population changed
        # before the second one, including a concept outside its own slice.
        path = prepared / "generate/train_data.parquet"
        frame = pd.read_parquet(path)
        frame.loc[0, "output"] = "new training response"
        frame.to_parquet(path, index=False)
    second = _run_soft(
        tmp_path,
        monkeypatch,
        prepared,
        "second",
        methods=("DiffMean",),
        concepts=(1, 2),
        counts=(0, 1),
        learning_rate=0.02 if difference == "learning_rate" else 0.01,
    )
    assert (first / "manifest.json").is_file() and (second / "manifest.json").is_file()
    expected = (
        "incompatible effective configs"
        if difference == "learning_rate"
        else "incompatible training source identities"
    )
    with pytest.raises(AxBenchAdvError, match=expected):
        merge_slices((first, second), tmp_path / "merged")
    assert not (tmp_path / "merged/manifest.json").exists()
    assert not list((tmp_path / "merged").rglob("aggregate.parquet"))
