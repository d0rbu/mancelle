"""Real-gloo tests for NL preflight identity and producer publication."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from steering._common.ddp import AlignedRankFailure, rank_synchronized_stage
from steering.eval.axbench_upstream import run_methods as driver
from steering.eval.axbench_upstream.run_methods import (
    _require_cross_rank_semantic_identity,
)
from tests.scripts.eval.test_axbench_project_methods_driver import (
    _enable_nl_receipt,
    _install_fakes,
    _write_inputs,
)


def _identity_worker(rank: int, _world_size: int, divergent: bool) -> str:
    identity = {"nl_target_scores_sha256": str(rank) if divergent else "same"}
    try:
        with rank_synchronized_stage("project NL rank identity", device="cpu"):
            _require_cross_rank_semantic_identity(identity)
    except AlignedRankFailure as exc:
        return str(exc)
    return "ok"


def _preflight_failure_worker(rank: int, world_size: int, failing_rank: int) -> str:
    try:
        with rank_synchronized_stage("project NL target preflight", device="cpu"):
            if rank == failing_rank:
                raise ValueError("synthetic rank-local NL receipt failure")
    except AlignedRankFailure as exc:
        return str(exc)
    return "unexpected-success"


@pytest.mark.parametrize("world_size", [1, 2, 4])
def test_nl_semantic_identity_agrees_in_real_gloo(
    gloo_multiprocess: Any,
    world_size: int,
) -> None:
    assert gloo_multiprocess(
        world_size,
        _identity_worker,
        False,
    ) == ["ok"] * world_size


@pytest.mark.parametrize("world_size", [2, 4])
def test_nl_semantic_identity_mismatch_aborts_every_real_gloo_rank(
    gloo_multiprocess: Any,
    world_size: int,
) -> None:
    messages = gloo_multiprocess(
        world_size,
        _identity_worker,
        True,
    )

    assert all("project NL rank identity" in message for message in messages)
    assert all("differs across distributed ranks" in message for message in messages)


@pytest.mark.parametrize("world_size", [2, 4])
def test_one_rank_nl_preflight_failure_aborts_every_real_gloo_rank(
    gloo_multiprocess: Any,
    world_size: int,
) -> None:
    messages = gloo_multiprocess(
        world_size,
        _preflight_failure_worker,
        world_size - 1,
    )

    assert all("project NL target preflight" in message for message in messages)
    assert all(f"rank {world_size - 1}" in message for message in messages)
    assert all("synthetic rank-local NL receipt failure" in message for message in messages)


def _driver_worker(
    rank: int,
    _world_size: int,
    kwargs: dict[str, Any],
    failing_rank: int | None,
) -> dict[str, Any]:
    """Use real scoring, stores, collectives and publication with cheap model doubles."""
    receipt_reads = 0
    real_load = driver.load_nl_target_source_receipt

    def load_receipt(path: Path) -> Any:
        nonlocal receipt_reads
        receipt_reads += 1
        if rank == failing_rank and receipt_reads == 2:
            raise OSError("rank-local receipt read failed before publication")
        return real_load(path)

    with pytest.MonkeyPatch.context() as monkeypatch:
        observed = _install_fakes(monkeypatch, kwargs)
        fake_fit = driver.fit_concept_dense_diffmeans

        def fit_owned(setup: Any, model: Any, sae: Any, training: Any, **named: Any) -> Any:
            fitted = fake_fit(setup, model, sae, training, **named)
            return {concept_id: fitted[concept_id] for concept_id in training}

        monkeypatch.setattr(driver, "fit_concept_dense_diffmeans", fit_owned)
        monkeypatch.setattr(driver, "load_nl_target_source_receipt", load_receipt)
        try:
            frame = driver.run_project_methods(**kwargs)
        except AlignedRankFailure as exc:
            return {"error": str(exc), "receipt_reads": receipt_reads}
    generated_cells = len(observed["scalar_calls"]) + sum(
        len(policies) for _pair, policies in observed["batch_calls"]
    )
    return {
        "rows": len(frame),
        "receipt_reads": receipt_reads,
        "generated_cells": generated_cells,
    }


@pytest.mark.parametrize("failing_rank", [0, 1])
def test_late_nl_rank_failure_prevents_publication_and_preserves_resumability(
    tmp_path: Path,
    gloo_multiprocess: Any,
    failing_rank: int,
) -> None:
    kwargs = _write_inputs(tmp_path)
    _enable_nl_receipt(kwargs)

    failed = gloo_multiprocess(2, _driver_worker, kwargs, failing_rank)

    assert all(result["receipt_reads"] == 2 for result in failed), failed
    assert all("rank-local receipt read failed" in result["error"] for result in failed)
    assert all(f"rank {failing_rank}" in result["error"] for result in failed)
    assert not Path(kwargs["output_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()

    # All generation cells were persisted before this late failure. A clean retry
    # must publish the same complete frame as a serial run in an independent store.
    resumed = gloo_multiprocess(2, _driver_worker, kwargs, None)
    assert resumed == [{"rows": 240, "receipt_reads": 2, "generated_cells": 0}] * 2
    serial_kwargs = _write_inputs(tmp_path / "serial")
    _enable_nl_receipt(serial_kwargs)
    assert gloo_multiprocess(1, _driver_worker, serial_kwargs, None) == [
        {"rows": 240, "receipt_reads": 2, "generated_cells": 720}
    ]
    pd.testing.assert_frame_equal(
        pd.read_parquet(kwargs["output_path"]),
        pd.read_parquet(serial_kwargs["output_path"]),
        check_exact=True,
    )
