"""Benchmark, selected-cell profiler, and project-fit resume contracts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pandas as pd
import pytest
import torch as t
from omegaconf import OmegaConf
from safetensors.torch import save_file

from scripts.benchmark import benchmark_axbench_upstream as benchmark
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS
from steering.eval.axbench_upstream.comparison import BundleKind
from steering.eval.axbench_upstream.profiling import (
    CellTraceController,
    CellTraceSelector,
    activate_cell_tracing,
    cell_trace_requested,
    trace_cell,
)
from steering.eval.axbench_upstream.run_methods import (
    _install_project_fit_receipt,
    _load_project_fit,
    _prepare_project_fit_dir,
    _project_fit_path,
    _save_project_fit,
)
from steering.eval.axbench_upstream.work_grid import (
    CellKey,
    build_encdec_cell_grid,
    build_project_cell_grid,
    owned_encdec_cells,
    owned_project_cells,
)
from steering.profiling.markers import timed_region


def _cell(*, factor: float = 6.0, method: str = "SelfConditioning") -> CellKey:
    cells = build_project_cell_grid(
        pd.DataFrame([{"concept_id": 0, "input_id": 0, "factor": factor}]),
        seed=42,
    )
    return next(cell for cell in cells if cell.method == method)


def _selector(cell: CellKey) -> CellTraceSelector:
    return CellTraceSelector(
        driver=cell.driver,
        method=cell.method,
        concept_id=cell.concept_id,
        input_id=cell.input_id,
        factor_hex=cell.factor_hex,
    )


def _samples() -> dict[str, Any]:
    return {
        "schema_version": 1,
        "arms": {
            "project": [
                {
                    "concepts": 2,
                    "generation_cells": 480,
                    "fit_seconds": fit,
                    "generation_seconds": generation,
                    "other_seconds": other,
                    "actual_output_tokens": tokens,
                }
                for fit, generation, other, tokens in (
                    (20.0, 480.0, 4.0, 48_000.0),
                    (24.0, 528.0, 5.0, 52_800.0),
                    (16.0, 432.0, 6.0, 43_200.0),
                )
            ],
            "encdec": [
                {
                    "concepts": 2,
                    "generation_cells": 520,
                    "fit_seconds": fit,
                    "generation_seconds": generation,
                    "other_seconds": other,
                    "actual_output_tokens": tokens,
                }
                for fit, generation, other, tokens in (
                    (40.0, 1_040.0, 8.0, 104_000.0),
                    (48.0, 1_144.0, 10.0, 114_400.0),
                    (32.0, 936.0, 12.0, 93_600.0),
                )
            ],
        },
    }


def _project_gloo_worker(
    rank: int,
    world_size: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete fake project driver under a real process group."""

    del world_size
    from steering.eval.axbench_upstream import run_methods as driver
    from tests.scripts.eval.test_axbench_project_methods_driver import (
        _install_fakes,
    )

    patcher = pytest.MonkeyPatch()
    observed = _install_fakes(patcher, kwargs)

    try:
        frame = driver.run_project_methods(**kwargs)
        generation_count = sum(
            len(values)
            for bucket in (
                observed["token_seeds"],
                observed["masked_token_seeds"],
            )
            for values in bucket.values()
        )
        return {
            "rank": rank,
            "rows": len(frame),
            "generation_count": generation_count,
            "batch_pairs": [pair for pair, _requests in observed["batch_calls"]],
            "batch_sizes": [len(requests) for _pair, requests in observed["batch_calls"]],
            "fit_events": [f"fit-rank-{rank}" for _ in observed["fitted_concepts"]],
        }
    finally:
        patcher.undo()


def _encdec_gloo_worker(
    rank: int,
    world_size: int,
    kwargs: dict[str, Any],
) -> dict[str, Any]:
    """Run the complete fake encoder-decoder driver under real Gloo."""

    del world_size
    from steering.eval.axbench_upstream.encdec_baselines import run as driver
    from steering.eval.axbench_upstream.encdec_baselines.directions import FittedDirection
    from tests.scripts.eval.test_axbench_encdec_driver import _install_fakes

    patcher = pytest.MonkeyPatch()
    observed = _install_fakes(patcher, kwargs)

    def load_direction(path: Path) -> FittedDirection:
        maximum = float(path.read_bytes().decode().split(":", 1)[1])
        concept_id = int(path.name.split("_")[1])
        first = 10.0 + concept_id if "linear_probe" in path.name else 1.0 + concept_id
        return FittedDirection(
            direction=t.tensor([first, 0.0, 0.0]),
            max_activation=maximum,
            metadata={
                "method": "probe" if "linear_probe" in path.name else "diffmean",
                "intervention_layer": 9,
            },
        )

    def load_lora(path: Path) -> tuple[dict[str, t.Tensor], dict[str, str]]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        return {"concept": t.tensor([payload["concept"]])}, payload["metadata"]

    generated: list[int] = []
    original_generate = driver.generate_encdec_baseline
    original_generate_batch = driver.generate_encdec_direction_batch

    def counted_generate(*args: Any, **generate_kwargs: Any) -> str:
        generated.append(1)
        return original_generate(*args, **generate_kwargs)

    def counted_generate_batch(*args: Any, **generate_kwargs: Any) -> tuple[str, ...]:
        result = original_generate_batch(*args, **generate_kwargs)
        generated.extend([1] * len(result))
        return result

    patcher.setattr(driver, "load_direction", load_direction)
    patcher.setattr(driver, "load_lora", load_lora)
    patcher.setattr(driver, "generate_encdec_baseline", counted_generate)
    patcher.setattr(driver, "generate_encdec_direction_batch", counted_generate_batch)
    try:
        main, aux = driver.run_encdec_baselines(**kwargs)
        return {
            "rank": rank,
            "main_rows": len(main),
            "aux_rows": len(aux),
            "generation_count": len(generated),
            "direction_batch_pairs": [
                (concept_id, input_id)
                for _methods, concept_id, input_id, _batch_size in observed["batches"]
            ],
            "direction_batch_sizes": [
                batch_size for _methods, _concept_id, _input_id, batch_size in observed["batches"]
            ],
            "fit_events": sorted(
                event
                for event in observed["events"]
                if event.startswith(("fit-diffmean", "fit-probe", "fit-lora"))
            ),
        }
    finally:
        patcher.undo()


def _project_gloo_rank_failure_worker(
    rank: int,
    world_size: int,
    kwargs: dict[str, Any],
) -> str:
    del world_size
    from steering._common.ddp import AlignedRankFailure
    from steering.eval.axbench_upstream import run_methods as driver
    from tests.scripts.eval.test_axbench_project_methods_driver import _install_fakes

    patcher = pytest.MonkeyPatch()
    _install_fakes(patcher, kwargs)
    original_prepare = driver.prepare_prompt_context

    def prepare(*args: Any, **prepare_kwargs: Any) -> Any:
        if rank == 1:
            raise OSError("rank-one synthetic conditioning failure")
        return original_prepare(*args, **prepare_kwargs)

    patcher.setattr(driver, "prepare_prompt_context", prepare)
    try:
        with pytest.raises(AlignedRankFailure) as caught:
            driver.run_project_methods(**kwargs)
        return str(caught.value)
    finally:
        patcher.undo()


def test_selector_matches_exactly_the_declared_semantic_dimensions() -> None:
    cell = _cell()
    selector = _selector(cell)

    assert selector.matches(cell)
    assert selector.to_dict() == {
        "driver": "project",
        "method": "SelfConditioning",
        "concept_id": 0,
        "input_id": 0,
        "factor_hex": "0x1.8000000000000p+2",
    }
    for replacement in (
        replace(cell, method="SelfConditioningMasked"),
        replace(cell, concept_id=1),
        replace(cell, input_id=1),
        replace(cell, factor_hex=(1.0).hex()),
    ):
        assert not selector.matches(replacement)


def test_selected_cell_profiler_writes_real_cpu_trace_and_exact_ledger(tmp_path: Path) -> None:
    cell = _cell()
    controller = CellTraceController(tmp_path / "traces", (_selector(cell),), row_limit=100)

    with activate_cell_tracing(controller):
        with trace_cell(cell):
            with timed_region("generation/decode_model_forward", sync_cuda=False):
                left = t.arange(64, dtype=t.float32).reshape(8, 8)
                result = left @ left
                assert result.shape == (8, 8)
    controller.require_complete()

    trace_dirs = list((tmp_path / "traces").iterdir())
    assert len(trace_dirs) == 1
    trace_dir = trace_dirs[0]
    assert (trace_dir / "trace.json").stat().st_size > 0
    payload = json.loads((trace_dir / "key_averages.json").read_text(encoding="utf-8"))
    assert payload["schema_version"] == 2
    assert payload["selector"] == _selector(cell).to_dict()
    assert payload["cell"] == cell.to_dict()
    assert payload["activities"][0] == "CPU"
    assert [row["key"] for row in payload["phase_events"]] == [
        "axbench_upstream.selected_cell",
        "generation/decode_model_forward",
    ]
    phase_by_key = {row["key"]: row for row in payload["phase_events"]}
    assert phase_by_key["generation/decode_model_forward"]["count"] == 1
    assert phase_by_key["generation/decode_model_forward"]["cpu_time_total_us"] > 0
    assert 1 <= len(payload["top_events"]) <= 100
    assert any(row["key"] == "aten::arange" for row in payload["top_events"])


def test_unselected_cell_is_a_true_noop_and_completion_remains_missing(tmp_path: Path) -> None:
    selected = _cell()
    unselected = _cell(factor=1.0)
    controller = CellTraceController(tmp_path / "traces", (_selector(selected),))

    with activate_cell_tracing(controller):
        assert cell_trace_requested(selected)
        assert not cell_trace_requested(unselected)
        with trace_cell(unselected):
            assert int(t.tensor(3).item()) == 3
        assert cell_trace_requested(selected)

    assert not (tmp_path / "traces").exists()
    with pytest.raises(RuntimeError, match="were not observed"):
        controller.require_complete()


def test_trace_controller_rejects_duplicate_selectors_and_invalid_row_limits(
    tmp_path: Path,
) -> None:
    selector = _selector(_cell())
    with pytest.raises(ValueError, match="must be unique"):
        CellTraceController(tmp_path, (selector, selector))
    for bad in (0, -1, True, 1.5):
        with pytest.raises(ValueError, match="row_limit must be a positive integer"):
            CellTraceController(tmp_path, (selector,), row_limit=cast(Any, bad))


def test_nested_trace_bindings_restore_the_outer_controller(tmp_path: Path) -> None:
    outer_cell = _cell(factor=6.0)
    inner_cell = _cell(factor=1.0)
    outer = CellTraceController(tmp_path / "outer", (_selector(outer_cell),))
    inner = CellTraceController(tmp_path / "inner", (_selector(inner_cell),))

    with activate_cell_tracing(outer):
        with activate_cell_tracing(inner):
            with trace_cell(inner_cell):
                _ = t.ones(2) + 1
        with trace_cell(outer_cell):
            _ = t.ones(2) + 2

    outer.require_complete()
    inner.require_complete()
    assert len(list((tmp_path / "outer").iterdir())) == 1
    assert len(list((tmp_path / "inner").iterdir())) == 1


def test_default_representative_selectors_cover_three_distinct_cells() -> None:
    project = benchmark._default_trace_selectors("project")
    encdec = benchmark._default_trace_selectors("encdec")

    assert len(project) == 1
    assert len(encdec) == 2
    assert {selector.method for selector in (*project, *encdec)} == {
        "SelfConditioning",
        "EncDecDiffMean",
        "EncDecLoRA",
    }
    assert len({*project, *encdec}) == 3


@pytest.mark.parametrize(("concept_ids", "owner_rank"), [(None, 1), ([1], 0)])
def test_project_owned_trace_selectors_follow_prompt_group_owner(
    tmp_path: Path,
    concept_ids: list[int] | None,
    owner_rank: int,
) -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 9, "input_id": 4, "factor": 0.0},
            {"concept_id": 1, "input_id": 0, "factor": 6.0},
            {"concept_id": 1, "input_id": 0, "factor": 1.0},
        ]
    )
    path = tmp_path / "main.parquet"
    frame.to_parquet(path, index=False)
    cfg = OmegaConf.create(
        {"main_path": str(path), "seed": 42, "max_concepts": 10, "concept_ids": concept_ids}
    )

    selectors_by_rank = [
        benchmark._owned_trace_selectors("project", cfg, rank=rank, world_size=3)
        for rank in range(3)
    ]

    assert [len(selectors) for selectors in selectors_by_rank] == [
        int(rank == owner_rank) for rank in range(3)
    ]
    if concept_ids is not None:
        frame = frame.loc[frame.concept_id.isin(concept_ids)]
    cells = build_project_cell_grid(frame, seed=42)
    assert benchmark._cell_grid("project", cfg) == cells
    rank_cells = owned_project_cells(cells, rank=owner_rank, world_size=3)
    selector = selectors_by_rank[owner_rank][0]
    assert selector.concept_id == 1
    assert sum(selector.matches(cell) for cell in rank_cells) == 1

    if concept_ids is None:
        selected_index = next(index for index, cell in enumerate(cells) if selector.matches(cell))
        assert selected_index % 3 == 2


def test_encdec_owned_trace_selectors_follow_prompt_group_owner(
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": 9, "input_id": 4, "factor": 0.0},
            {"concept_id": 0, "input_id": 0, "factor": 6.0},
            {"concept_id": 0, "input_id": 0, "factor": 1.0},
        ]
    )
    path = tmp_path / "main.parquet"
    frame.to_parquet(path, index=False)
    cfg = OmegaConf.create({"main_path": str(path), "seed": 42})

    selectors_by_rank = [
        benchmark._owned_trace_selectors("encdec", cfg, rank=rank, world_size=3)
        for rank in range(3)
    ]

    assert [len(selectors) for selectors in selectors_by_rank] == [0, 2, 0]
    cells = build_encdec_cell_grid(frame, seed=42)
    rank_one_cells = owned_encdec_cells(cells, rank=1, world_size=3)
    assert all(
        sum(selector.matches(cell) for cell in rank_one_cells) == 1
        for selector in selectors_by_rank[1]
    )


def test_owned_trace_selector_requires_one_exact_cell(tmp_path: Path) -> None:
    frame = pd.DataFrame([{"concept_id": 0, "input_id": 0, "factor": 1.0}])
    path = tmp_path / "main.parquet"
    frame.to_parquet(path, index=False)
    cfg = OmegaConf.create({"main_path": str(path), "seed": 42})

    with pytest.raises(ValueError, match="match exactly one logical cell"):
        benchmark._owned_trace_selectors("project", cfg, rank=0, world_size=1)


@pytest.mark.parametrize("concept_ids", [None, [1]])
def test_project_cell_grid_loads_the_bound_model_free_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    concept_ids: list[int] | None,
) -> None:
    frame = pd.DataFrame(
        [
            {"concept_id": concept_id, "input_id": 0, "factor": factor}
            for concept_id in range(2)
            for factor in range(7)
        ]
    )
    observed: dict[str, object] = {}

    def load(**kwargs: object) -> tuple[pd.DataFrame, str, str]:
        observed.update(kwargs)
        return frame, "a" * 64, "b" * 64

    monkeypatch.setattr(benchmark, "load_project_input_artifact", load)
    cfg = OmegaConf.create(
        {
            "main_path": None,
            "project_input_path": str(tmp_path / "input.parquet"),
            "project_input_manifest_path": str(tmp_path / "manifest.json"),
            "prepared_dir": str(tmp_path / "prepared"),
            "max_concepts": 2,
            "concept_ids": concept_ids,
            "seed": 42,
        }
    )

    cells = benchmark._cell_grid("project", cfg)

    assert cells == build_project_cell_grid(
        frame if concept_ids is None else frame.loc[frame.concept_id.isin(concept_ids)],
        seed=42,
    )
    assert observed == {
        "input_path": Path(cfg.project_input_path),
        "manifest_path": Path(cfg.project_input_manifest_path),
        "prepared_dir": Path(cfg.prepared_dir),
        "max_concepts": 2,
    }


@pytest.mark.parametrize(
    ("main_path", "project_path", "manifest_path", "expected"),
    [
        (None, None, None, "exactly one"),
        ("main", "project", "manifest", "exactly one"),
        ("main", None, "manifest", "requires project_input_path"),
        (None, "project", None, "manifest_path is required"),
    ],
)
def test_project_profiler_rejects_ambiguous_or_unbound_sources(
    main_path: str | None,
    project_path: str | None,
    manifest_path: str | None,
    expected: str,
) -> None:
    cfg = OmegaConf.create(
        {
            "main_path": main_path,
            "project_input_path": project_path,
            "project_input_manifest_path": manifest_path,
            "prepared_dir": "prepared",
            "max_concepts": 1,
            "seed": 42,
        }
    )

    with pytest.raises(ValueError, match=expected):
        benchmark._cell_grid("project", cfg)


def test_profile_completion_failure_is_recorded_inside_profiler_region(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed_exit_types: list[type[BaseException] | None] = []
    cleanup_calls: list[None] = []

    class FakeProfiler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeProfiler:
            return self

        def __exit__(
            self,
            exc_type: type[BaseException] | None,
            _exc: BaseException | None,
            _traceback: Any,
        ) -> None:
            observed_exit_types.append(exc_type)

    monkeypatch.setattr(benchmark, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: False)
    monkeypatch.setattr(benchmark, "compute_init", lambda _device: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(benchmark, "compute_cleanup", lambda: cleanup_calls.append(None))
    monkeypatch.setattr(benchmark, "_runtime_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        benchmark,
        "_owned_trace_selectors",
        lambda *_args, **_kwargs: (_selector(_cell()),),
    )
    monkeypatch.setattr(benchmark, "run_project_methods_from_dictconfig", lambda _cfg: None)

    with pytest.raises(RuntimeError, match="were not observed"):
        benchmark.run_profile(
            kind="project",
            output_dir=tmp_path / "profile",
            config_name="methods",
            overrides=("device=cpu",),
            torch_traces=True,
        )

    assert observed_exit_types == [RuntimeError]
    assert cleanup_calls == []


def test_direct_profile_never_initializes_ddp_or_mutates_numeric_or_rng_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProfiler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeProfiler:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

    monkeypatch.setattr(benchmark, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: False)
    monkeypatch.setattr(
        benchmark,
        "compute_init",
        lambda _device: (_ for _ in ()).throw(AssertionError("direct run called compute_init")),
    )
    monkeypatch.setattr(
        benchmark,
        "compute_cleanup",
        lambda: (_ for _ in ()).throw(AssertionError("direct run called compute_cleanup")),
    )
    monkeypatch.setattr(benchmark, "_runtime_metadata", lambda *_args, **_kwargs: {})
    observed: list[tuple[str, t.Tensor]] = []

    def run(_cfg: Any) -> None:
        observed.append((t.get_float32_matmul_precision(), t.get_rng_state().clone()))

    monkeypatch.setattr(benchmark, "run_project_methods_from_dictconfig", run)
    original_precision = t.get_float32_matmul_precision()
    original_rng = t.get_rng_state().clone()
    try:
        t.set_float32_matmul_precision("highest")
        t.random.default_generator.manual_seed(9127)
        expected_rng = t.get_rng_state().clone()
        benchmark.run_profile(
            kind="project",
            output_dir=tmp_path / "profile",
            config_name="methods",
            overrides=("device=cpu",),
            torch_traces=False,
        )
        assert observed[0][0] == "highest"
        assert t.equal(observed[0][1], expected_rng)
        assert t.get_float32_matmul_precision() == "highest"
        assert t.equal(t.get_rng_state(), expected_rng)
    finally:
        t.set_float32_matmul_precision(original_precision)
        t.set_rng_state(original_rng)


def test_torchrun_profile_restores_precision_before_producer_and_cleans_up(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProfiler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeProfiler:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

    cleanup_calls: list[None] = []

    def initialize(_device: str) -> tuple[bool, int, int, int, str]:
        t.set_float32_matmul_precision("high")
        return True, 0, 0, 1, "cuda:0"

    monkeypatch.setattr(benchmark, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: True)
    monkeypatch.setattr(benchmark, "compute_init", initialize)
    monkeypatch.setattr(benchmark, "compute_cleanup", lambda: cleanup_calls.append(None))
    monkeypatch.setattr(benchmark, "_runtime_metadata", lambda *_args, **_kwargs: {})
    observed_precision: list[str] = []
    monkeypatch.setattr(
        benchmark,
        "run_project_methods_from_dictconfig",
        lambda _cfg: observed_precision.append(t.get_float32_matmul_precision()),
    )
    original_precision = t.get_float32_matmul_precision()
    try:
        t.set_float32_matmul_precision("highest")
        benchmark.run_profile(
            kind="project",
            output_dir=tmp_path / "profile",
            config_name="methods",
            overrides=("device=cuda",),
            torch_traces=False,
        )
        assert observed_precision == ["highest"]
        assert t.get_float32_matmul_precision() == "highest"
        assert cleanup_calls == [None]
    finally:
        t.set_float32_matmul_precision(original_precision)


def test_torchrun_profile_init_failure_restores_precision_and_cleans_partial_state(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleanup_calls: list[None] = []

    def fail_after_mutation(_device: str) -> None:
        t.set_float32_matmul_precision("high")
        raise KeyboardInterrupt("synthetic init failure")

    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: True)
    monkeypatch.setattr(benchmark, "compute_init", fail_after_mutation)
    monkeypatch.setattr(benchmark, "compute_cleanup", lambda: cleanup_calls.append(None))
    original_precision = t.get_float32_matmul_precision()
    try:
        t.set_float32_matmul_precision("highest")
        with pytest.raises(KeyboardInterrupt, match="synthetic init failure"):
            benchmark.run_profile(
                kind="project",
                output_dir=tmp_path / "profile",
                config_name="methods",
                overrides=("device=cuda",),
                torch_traces=False,
            )
        assert t.get_float32_matmul_precision() == "highest"
        assert cleanup_calls == [None]
    finally:
        t.set_float32_matmul_precision(original_precision)


def test_torchrun_profile_cleans_up_on_producer_base_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeProfiler:
        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            pass

        def __enter__(self) -> FakeProfiler:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

    cleanup_calls: list[None] = []
    monkeypatch.setattr(benchmark, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: True)
    monkeypatch.setattr(
        benchmark,
        "compute_init",
        lambda _device: (True, 0, 0, 1, "cuda:0"),
    )
    monkeypatch.setattr(benchmark, "compute_cleanup", lambda: cleanup_calls.append(None))
    monkeypatch.setattr(benchmark, "_runtime_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        benchmark,
        "run_project_methods_from_dictconfig",
        lambda _cfg: (_ for _ in ()).throw(KeyboardInterrupt("synthetic producer failure")),
    )

    with pytest.raises(KeyboardInterrupt, match="synthetic producer failure"):
        benchmark.run_profile(
            kind="project",
            output_dir=tmp_path / "profile",
            config_name="methods",
            overrides=("device=cuda",),
            torch_traces=False,
        )

    assert cleanup_calls == [None]


def test_profile_samples_explicit_single_process_cuda_device(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    sampled_indices: list[int] = []

    class FakeProfiler:
        def __init__(self, *_args: Any, gpu_device_index: int, **_kwargs: Any) -> None:
            sampled_indices.append(gpu_device_index)

        def __enter__(self) -> FakeProfiler:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

    monkeypatch.setattr(benchmark, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: False)
    monkeypatch.setattr(
        benchmark,
        "compute_init",
        lambda _device: (False, 0, 0, 1, "cuda:3"),
    )
    monkeypatch.setattr(benchmark, "compute_cleanup", lambda: None)
    monkeypatch.setattr(benchmark, "_runtime_metadata", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(benchmark, "run_project_methods_from_dictconfig", lambda _cfg: None)

    benchmark.run_profile(
        kind="project",
        output_dir=tmp_path / "profile",
        config_name="methods",
        overrides=("device=cuda:3",),
        torch_traces=False,
    )

    assert sampled_indices == [3]


def test_throughput_cli_disables_selected_traces_and_cuda_timing_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, Any]] = []

    def capture(**kwargs: Any) -> None:
        observed.append(kwargs)

    monkeypatch.setattr(benchmark, "run_profile", capture)
    benchmark.main(
        [
            "throughput",
            "--kind",
            "encdec",
            "--output-dir",
            str(tmp_path / "sample"),
            "--override",
            "device=cuda",
        ]
    )

    assert observed == [
        {
            "kind": "encdec",
            "output_dir": tmp_path / "sample",
            "config_name": "encdec_baselines",
            "overrides": ["device=cuda"],
            "torch_traces": False,
            "synchronize_cuda": False,
            "measurement_mode": "throughput",
            "cuda_event_regions": (),
        }
    ]


def test_throughput_cli_enables_deferred_cuda_timing_for_exact_regions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[dict[str, Any]] = []
    monkeypatch.setattr(benchmark, "run_profile", lambda **kwargs: observed.append(kwargs))

    benchmark.main(
        [
            "throughput",
            "--kind",
            "project",
            "--output-dir",
            str(tmp_path / "sample"),
            "--cuda-event-region",
            "axbench_upstream.generation.batch_prefills",
            "--cuda-event-region",
            "axbench_upstream.generation.batch_stream",
            "--override",
            "device=cuda",
        ]
    )

    assert observed == [
        {
            "kind": "project",
            "output_dir": tmp_path / "sample",
            "config_name": "methods",
            "overrides": ["device=cuda"],
            "torch_traces": False,
            "synchronize_cuda": False,
            "measurement_mode": "throughput",
            "cuda_event_regions": (
                "axbench_upstream.generation.batch_prefills",
                "axbench_upstream.generation.batch_stream",
            ),
        }
    ]


def test_throughput_metadata_and_profiler_disable_cuda_synchronization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profiler_options: list[dict[str, Any]] = []
    completion_checks: list[None] = []

    class FakeProfiler:
        def __init__(self, *_args: Any, **kwargs: Any) -> None:
            profiler_options.append(kwargs)

        def __enter__(self) -> FakeProfiler:
            return self

        def __exit__(self, *_args: Any) -> None:
            pass

        def require_cuda_event_regions_complete(self) -> None:
            completion_checks.append(None)

    monkeypatch.setattr(benchmark, "AxBenchProfiler", FakeProfiler)
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: False)
    monkeypatch.setattr(benchmark, "compute_init", lambda _device: (False, 0, 0, 1, "cpu"))
    monkeypatch.setattr(benchmark, "compute_cleanup", lambda: None)
    runtime = {"python": "3.13.test"}
    monkeypatch.setattr(benchmark, "_runtime_metadata", lambda *_args, **_kwargs: runtime)
    monkeypatch.setattr(benchmark, "run_project_methods_from_dictconfig", lambda _cfg: None)

    benchmark.run_profile(
        kind="project",
        output_dir=tmp_path / "sample",
        config_name="methods",
        overrides=("device=cuda:3",),
        torch_traces=False,
        synchronize_cuda=False,
        measurement_mode="throughput",
        cuda_event_regions=("measured.region",),
    )

    assert profiler_options[0]["synchronize_cuda"] is False
    assert profiler_options[0]["cuda_event_regions"] == ("measured.region",)
    assert profiler_options[0]["metadata"]["measurement_mode"] == "throughput"
    assert profiler_options[0]["metadata"]["python"] == "3.13.test"
    assert profiler_options[0]["metadata"]["cuda_event_regions"] == ["measured.region"]
    assert completion_checks == [None]


def test_deferred_cuda_region_timing_rejects_non_cuda_profile_before_producer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: False)
    monkeypatch.setattr(
        benchmark,
        "run_project_methods_from_dictconfig",
        lambda _cfg: pytest.fail("invalid timing request must not run the producer"),
    )

    with pytest.raises(ValueError, match="device=cuda"):
        benchmark.run_profile(
            kind="project",
            output_dir=tmp_path / "sample",
            config_name="methods",
            overrides=("device=cpu",),
            torch_traces=False,
            synchronize_cuda=False,
            measurement_mode="throughput",
            cuda_event_regions=("measured.region",),
        )


def test_deferred_cuda_region_timing_rejects_per_region_synchronization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark, "is_ddp_requested", lambda: False)
    monkeypatch.setattr(
        benchmark,
        "run_project_methods_from_dictconfig",
        lambda _cfg: pytest.fail("invalid timing request must not run the producer"),
    )

    with pytest.raises(ValueError, match="synchronize_cuda=False"):
        benchmark.run_profile(
            kind="project",
            output_dir=tmp_path / "sample",
            config_name="methods",
            overrides=("device=cuda",),
            torch_traces=False,
            synchronize_cuda=True,
            cuda_event_regions=("measured.region",),
        )


def test_runtime_metadata_records_exact_python_version(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(benchmark.platform, "python_version", lambda: "3.13.exact")
    monkeypatch.setattr(benchmark.t.cuda, "is_available", lambda: False)

    metadata = benchmark._runtime_metadata(
        OmegaConf.create({"device": "cpu"}),
        rank=0,
        local_rank=0,
        world_size=1,
    )

    assert metadata["python"] == "3.13.exact"


def test_runtime_metadata_uses_configured_cuda_device_and_omits_gpu_for_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(benchmark.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark.t.cuda, "current_device", lambda: 7)
    monkeypatch.setattr(benchmark.t.cuda, "get_device_name", lambda index: f"gpu-{index}")
    monkeypatch.setattr(benchmark.t.cuda, "get_device_capability", lambda index: (9, index))

    cuda_metadata = benchmark._runtime_metadata(
        OmegaConf.create({"device": "cuda:3"}),
        rank=0,
        local_rank=0,
        world_size=1,
    )
    cpu_metadata = benchmark._runtime_metadata(
        OmegaConf.create({"device": "cpu"}),
        rank=0,
        local_rank=0,
        world_size=1,
    )

    assert cuda_metadata["gpu_name"] == "gpu-3"
    assert cuda_metadata["gpu_capability"] == [9, 3]
    assert "gpu_name" not in cpu_metadata
    assert "gpu_capability" not in cpu_metadata


def test_attention_delta_reports_exact_and_known_numeric_differences() -> None:
    reference = t.tensor([[1.0, -2.0], [0.5, 4.0]], dtype=t.float32)

    assert benchmark._attention_delta(reference, reference.clone()) == {
        "bitwise_equal": True,
        "max_absolute": 0.0,
        "mean_absolute": 0.0,
        "max_relative": 0.0,
    }
    changed = reference.clone()
    changed[0, 0] = 1.25
    assert benchmark._attention_delta(reference, changed) == {
        "bitwise_equal": False,
        "max_absolute": 0.25,
        "mean_absolute": 0.0625,
        "max_relative": 0.25,
    }


def test_attention_delta_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError, match="identical shapes"):
        benchmark._attention_delta(t.ones(2), t.ones(1, 2))


@pytest.mark.parametrize("packed_count", [2, 128, 384, 1024])
def test_attention_value_views_are_logically_exact_but_have_the_expected_strides(
    packed_count: int,
) -> None:
    width = benchmark._ATTENTION_WIDTH
    projected = t.arange(packed_count * 2 * width, dtype=t.float32).view(
        packed_count,
        2 * width,
    )
    _key, value = projected.split([width, width], dim=-1)

    historical = benchmark._historical_attention_value_view(
        value,
        packed_count=packed_count,
    )
    candidate = benchmark._extent_invariant_attention_value_view(
        value,
        packed_count=packed_count,
    )

    assert t.equal(historical, candidate)
    assert historical.shape == (1, 6, packed_count, 128)
    assert candidate.shape == historical.shape
    assert historical.stride() == (packed_count * 1536, 128, 1536, 1)
    assert candidate.stride() == (768, 128, 1536, 1)
    assert historical.storage_offset() == 768
    assert candidate.storage_offset() == historical.storage_offset()


@pytest.mark.parametrize(
    ("field", "bad"),
    [
        ("sequence_length", 0),
        ("packed_count", -1),
        ("warmups", True),
        ("iterations", 1.5),
    ],
)
def test_attention_benchmark_rejects_invalid_domains_before_cuda(
    field: str,
    bad: Any,
) -> None:
    kwargs: dict[str, Any] = {
        "sequence_length": 127,
        "packed_count": 1024,
        "warmups": 5,
        "iterations": 20,
    }
    kwargs[field] = bad
    with pytest.raises(ValueError, match="positive integer"):
        benchmark.run_attention_microbenchmark(**kwargs)


def test_attention_cli_serializes_the_measured_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "attention.json"
    observed: list[dict[str, int]] = []

    def run(**kwargs: int) -> dict[str, Any]:
        observed.append(kwargs)
        return {"schema_version": 1, "benchmark": "attention", "measured": True}

    monkeypatch.setattr(benchmark, "run_attention_microbenchmark", run)
    benchmark.main(
        [
            "attention",
            "--sequence-length",
            "113",
            "--packed-count",
            "384",
            "--warmups",
            "3",
            "--iterations",
            "7",
            "--output-json",
            str(output),
        ]
    )

    assert observed == [
        {
            "sequence_length": 113,
            "packed_count": 384,
            "warmups": 3,
            "iterations": 7,
        }
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "schema_version": 1,
        "benchmark": "attention",
        "measured": True,
    }


def test_cost_model_uses_medians_and_labels_every_result_as_projection() -> None:
    modeled = benchmark.build_cost_model(_samples(), workers=4)

    assert modeled["result_kind"] == "projection_not_measurement"
    assert modeled["workers"] == 4
    project = modeled["arms"]["project"]
    assert project["normalized_samples"]["generation_seconds_per_cell"] == {
        "sample_count": 3,
        "median": 1.0,
        "p95": 1.1,
        "min": 0.9,
        "max": 1.1,
    }
    c2 = project["projections"]["c2"]
    assert c2 == {
        "kind": "projection",
        "concepts": 2,
        "generation_cells": 480,
        "workers": 4,
        "fit_seconds": 10.0,
        "generation_seconds": 144.0,
        "generation_seconds_arbitrary_order_upper_bound": 240.0,
        "other_seconds": 5.0,
        "total_seconds": 159.0,
        "total_seconds_arbitrary_order_upper_bound": 255.0,
        "projected_output_tokens": 48_000.0,
    }
    c500 = modeled["arms"]["encdec"]["projections"]["c500"]
    assert c500["generation_cells"] == 130_000
    assert c500["fit_seconds"] == 20.0 * 125
    assert c500["generation_seconds"] == 2.0 * 32_526
    assert c500["generation_seconds_arbitrary_order_upper_bound"] == 2.0 * 32_760


@pytest.mark.parametrize(
    (
        "workers",
        "concepts",
        "project_cells",
        "encdec_cells",
        "project_upper_bound",
        "encdec_upper_bound",
    ),
    [
        (4, 2, 144, 156, 240, 260),
        (8, 2, 96, 104, 144, 156),
        (4, 500, 30_024, 32_526, 30_240, 32_760),
        (8, 500, 15_024, 16_276, 15_240, 16_510),
    ],
)
def test_cost_model_charges_each_synchronized_generation_stage_to_slowest_worker(
    workers: int,
    concepts: int,
    project_cells: int,
    encdec_cells: int,
    project_upper_bound: int,
    encdec_upper_bound: int,
) -> None:
    modeled = benchmark.build_cost_model(_samples(), workers=workers)

    assert (
        benchmark._canonical_generation_cells_per_worker(
            "project", concepts=concepts, workers=workers
        )
        == project_cells
    )
    assert (
        benchmark._canonical_generation_cells_per_worker(
            "encdec", concepts=concepts, workers=workers
        )
        == encdec_cells
    )
    assert (
        benchmark._arbitrary_order_generation_cells_upper_bound(
            "project", concepts=concepts, workers=workers
        )
        == project_upper_bound
    )
    assert (
        benchmark._arbitrary_order_generation_cells_upper_bound(
            "encdec", concepts=concepts, workers=workers
        )
        == encdec_upper_bound
    )
    projection = f"c{concepts}"
    assert (
        modeled["arms"]["project"]["projections"][projection]["generation_seconds"]
        == 1.0 * project_cells
    )
    assert (
        modeled["arms"]["encdec"]["projections"][projection]["generation_seconds"]
        == 2.0 * encdec_cells
    )
    assert (
        modeled["arms"]["project"]["projections"][projection][
            "generation_seconds_arbitrary_order_upper_bound"
        ]
        == 1.0 * project_upper_bound
    )
    assert (
        modeled["arms"]["encdec"]["projections"][projection][
            "generation_seconds_arbitrary_order_upper_bound"
        ]
        == 2.0 * encdec_upper_bound
    )


def test_arbitrary_order_bound_covers_executable_c2_w4_stage_skew() -> None:
    first_pairs = [(0, input_id) for input_id in range(10)]
    bulk_pairs = [(1, input_id) for input_id in range(10)]
    pair_order: list[tuple[int, int] | None] = [None] * 20
    for index, pair in zip((0, 1, 4, 5, 8, 9, 12, 13, 16, 17), first_pairs, strict=True):
        pair_order[index] = pair
    for index, pair in zip((2, 3, 6, 7, 10, 11, 14, 15, 18, 19), bulk_pairs, strict=True):
        pair_order[index] = pair
    assert all(pair is not None for pair in pair_order)
    frame = pd.DataFrame(
        [
            {"concept_id": pair[0], "input_id": pair[1], "factor": factor}
            for pair in pair_order
            if pair is not None
            for factor in FULL_FACTORS
        ]
    )

    for kind, cells, owner, expected in (
        ("project", build_project_cell_grid(frame, seed=42), owned_project_cells, 240),
        ("encdec", build_encdec_cell_grid(frame, seed=42), owned_encdec_cells, 260),
    ):
        stripes = [owner(cells, rank=rank, world_size=4) for rank in range(4)]
        first_max = max(sum(cell.concept_id == 0 for cell in stripe) for stripe in stripes)
        bulk_max = max(sum(cell.concept_id != 0 for cell in stripe) for stripe in stripes)
        assert first_max + bulk_max == expected
        assert (
            benchmark._arbitrary_order_generation_cells_upper_bound(kind, concepts=2, workers=4)
            == expected
        )


@pytest.mark.parametrize(("kind", "expected"), [("project", 96), ("encdec", 104)])
def test_arbitrary_order_bound_is_deliberately_conservative_for_c2_w19(
    kind: BundleKind,
    expected: int,
) -> None:
    # Twenty groups over nineteen workers leave exactly one worker with two
    # groups. The stage-wise bound may charge that sole two-group capacity to
    # both stages, even though one worker cannot hold two groups from each.
    assert (
        benchmark._arbitrary_order_generation_cells_upper_bound(
            kind,
            concepts=2,
            workers=19,
        )
        == expected
    )


@pytest.mark.parametrize("workers", [0, -1, True, 1.5, "2", None])
def test_cost_model_rejects_invalid_worker_counts(workers: Any) -> None:
    with pytest.raises(ValueError, match="workers must be a positive integer"):
        benchmark.build_cost_model(_samples(), workers=workers)


def test_cost_model_rejects_fewer_than_three_samples() -> None:
    samples = _samples()
    samples["arms"]["project"] = samples["arms"]["project"][:2]

    with pytest.raises(ValueError, match="at least three samples"):
        benchmark.build_cost_model(samples, workers=1)


@pytest.mark.parametrize(
    ("field", "bad", "message"),
    [
        ("concepts", True, "positive integer"),
        ("generation_cells", 479, "must equal 480"),
        ("fit_seconds", -1.0, "finite nonnegative"),
        ("generation_seconds", float("nan"), "finite nonnegative"),
        ("other_seconds", float("inf"), "finite nonnegative"),
        ("actual_output_tokens", "100", "finite nonnegative"),
    ],
)
def test_cost_model_rejects_each_malformed_sample_field(
    field: str,
    bad: Any,
    message: str,
) -> None:
    samples = _samples()
    samples["arms"]["project"][0][field] = bad

    with pytest.raises(ValueError, match=message):
        benchmark.build_cost_model(samples, workers=1)


def test_cost_model_rejects_missing_or_extra_sample_fields() -> None:
    for mutate in (
        lambda sample: sample.pop("other_seconds"),
        lambda sample: sample.__setitem__("unknown", 1),
    ):
        samples = _samples()
        mutate(samples["arms"]["project"][0])
        with pytest.raises(ValueError, match="missing or unexpected fields"):
            benchmark.build_cost_model(samples, workers=1)


def test_strict_json_loader_rejects_duplicate_keys_and_nonfinite_values(tmp_path: Path) -> None:
    duplicate = tmp_path / "duplicate.json"
    duplicate.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
    nonfinite = tmp_path / "nonfinite.json"
    nonfinite.write_text('{"value":NaN}', encoding="utf-8")

    with pytest.raises(ValueError, match="duplicate key"):
        benchmark._strict_json_object(duplicate, label="samples")
    with pytest.raises(ValueError, match="non-finite constant"):
        benchmark._strict_json_object(nonfinite, label="samples")


@pytest.mark.parametrize("include_nl_steering", (False, True))
def test_compare_cli_serializes_the_machine_readable_report(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    include_nl_steering: bool,
) -> None:
    output = tmp_path / "comparison.json"
    observed: list[tuple[Any, ...]] = []

    def compare(
        kind: str, reference: Path, candidate: Path, *, include_nl_steering: bool
    ) -> Any:
        observed.append((kind, reference, candidate, include_nl_steering))
        return SimpleNamespace(to_dict=lambda: {"kind": kind, "cell_count": 480})

    monkeypatch.setattr(benchmark, "compare_bundle", compare)
    benchmark.main(
        [
            "compare",
            "--kind",
            "project",
            "--reference-manifest",
            "reference.json",
            "--candidate-manifest",
            "candidate.json",
            "--output-json",
            str(output),
            *(["--include-nl-steering"] if include_nl_steering else []),
        ]
    )

    assert observed == [
        ("project", Path("reference.json"), Path("candidate.json"), include_nl_steering)
    ]
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "kind": "project",
        "cell_count": 480,
    }


def test_cost_model_cli_round_trips_strict_json(tmp_path: Path) -> None:
    input_path = tmp_path / "samples.json"
    output_path = tmp_path / "model.json"
    input_path.write_text(json.dumps(_samples(), allow_nan=False), encoding="utf-8")

    benchmark.main(
        [
            "cost-model",
            "--samples-json",
            str(input_path),
            "--workers",
            "2",
            "--output-json",
            str(output_path),
        ]
    )

    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["workers"] == 2
    assert payload["result_kind"] == "projection_not_measurement"


def test_project_fit_artifact_exact_round_trip_and_receipt(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fits"
    resume_root = tmp_path / "resume"
    resume_root.mkdir()
    fingerprint = "a" * 64
    _prepare_project_fit_dir(
        fit_dir=fit_dir,
        semantic_fingerprint=fingerprint,
        concept_ids=(0, 1),
    )
    expected = {
        0: t.linspace(-0.9, 0.9, 8, dtype=t.float32),
        1: t.linspace(0.8, -0.8, 8, dtype=t.float32),
    }
    for concept_id, scores in expected.items():
        _save_project_fit(
            _project_fit_path(fit_dir, concept_id),
            scores,
            concept_id=concept_id,
            semantic_fingerprint=fingerprint,
        )

    actual = {
        concept_id: _load_project_fit(
            _project_fit_path(fit_dir, concept_id),
            concept_id=concept_id,
            semantic_fingerprint=fingerprint,
            expected_width=8,
        )
        for concept_id in range(2)
    }
    _install_project_fit_receipt(
        resume_root=resume_root,
        fit_dir=fit_dir,
        semantic_fingerprint=fingerprint,
        concept_ids=(0, 1),
    )
    first_receipt = (resume_root / "fit_artifacts.json").read_bytes()
    _install_project_fit_receipt(
        resume_root=resume_root,
        fit_dir=fit_dir,
        semantic_fingerprint=fingerprint,
        concept_ids=(0, 1),
    )

    assert all(t.equal(actual[concept_id], expected[concept_id]) for concept_id in expected)
    assert (resume_root / "fit_artifacts.json").read_bytes() == first_receipt
    assert not list(fit_dir.glob("*.tmp"))


def test_project_fit_serialization_is_byte_identical_across_independent_paths(
    tmp_path: Path,
) -> None:
    scores = t.linspace(-1.0, 1.0, 17, dtype=t.float32)
    first = tmp_path / "first.safetensors"
    second = tmp_path / "second.safetensors"

    for path in (first, second):
        _save_project_fit(
            path,
            scores,
            concept_id=7,
            semantic_fingerprint="f" * 64,
        )

    assert first.read_bytes() == second.read_bytes()


def test_project_fit_identity_rejects_changed_semantics(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fits"
    _prepare_project_fit_dir(
        fit_dir=fit_dir,
        semantic_fingerprint="a" * 64,
        concept_ids=(0, 1),
    )

    with pytest.raises(RuntimeError, match="project fit identity"):
        _prepare_project_fit_dir(
            fit_dir=fit_dir,
            semantic_fingerprint="b" * 64,
            concept_ids=(0, 1),
        )
    with pytest.raises(RuntimeError, match="project fit identity"):
        _prepare_project_fit_dir(
            fit_dir=fit_dir,
            semantic_fingerprint="a" * 64,
            concept_ids=(0, 1, 2),
        )


def test_project_fit_directory_rejects_unknown_files_directories_and_symlinks(
    tmp_path: Path,
) -> None:
    for kind in ("file", "directory", "symlink"):
        fit_dir = tmp_path / kind
        _prepare_project_fit_dir(
            fit_dir=fit_dir,
            semantic_fingerprint="a" * 64,
            concept_ids=(0,),
        )
        unknown = fit_dir / "unknown"
        if kind == "file":
            unknown.write_text("x", encoding="utf-8")
        elif kind == "directory":
            unknown.mkdir()
        else:
            unknown.symlink_to(fit_dir / "identity.json")
        with pytest.raises(RuntimeError, match="unknown entry|symlink"):
            _prepare_project_fit_dir(
                fit_dir=fit_dir,
                semantic_fingerprint="a" * 64,
                concept_ids=(0,),
            )


@pytest.mark.parametrize(
    ("tensor", "message"),
    [
        (t.tensor([], dtype=t.float32), "finite nonempty vector"),
        (t.ones(2, 2), "finite nonempty vector"),
        (t.tensor([float("nan")]), "finite nonempty vector"),
    ],
)
def test_project_fit_save_rejects_invalid_tensors(
    tmp_path: Path,
    tensor: t.Tensor,
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _save_project_fit(
            tmp_path / "fit.safetensors",
            tensor,
            concept_id=0,
            semantic_fingerprint="a" * 64,
        )


@pytest.mark.parametrize(
    ("tensor", "metadata", "expected_width", "message"),
    [
        (
            {"dense_diffmean": t.ones(8, dtype=t.float64)},
            {
                "format": "axbench-upstream-project-dense-diffmean-v1",
                "concept_id": "0",
                "semantic_fingerprint": "a" * 64,
            },
            8,
            "invalid tensor contract",
        ),
        (
            {"wrong": t.ones(8)},
            {
                "format": "axbench-upstream-project-dense-diffmean-v1",
                "concept_id": "0",
                "semantic_fingerprint": "a" * 64,
            },
            8,
            "must contain only",
        ),
        (
            {"dense_diffmean": t.ones(8)},
            {"format": "wrong"},
            8,
            "metadata differs",
        ),
        (
            {"dense_diffmean": t.ones(7)},
            {
                "format": "axbench-upstream-project-dense-diffmean-v1",
                "concept_id": "0",
                "semantic_fingerprint": "a" * 64,
            },
            8,
            "invalid tensor contract",
        ),
    ],
)
def test_project_fit_load_falsifies_tensor_and_metadata_contracts(
    tmp_path: Path,
    tensor: dict[str, t.Tensor],
    metadata: dict[str, str],
    expected_width: int,
    message: str,
) -> None:
    path = tmp_path / "fit.safetensors"
    save_file(tensor, path, metadata=metadata)

    with pytest.raises(RuntimeError, match=message):
        _load_project_fit(
            path,
            concept_id=0,
            semantic_fingerprint="a" * 64,
            expected_width=expected_width,
        )


def test_project_fit_receipt_rejects_post_receipt_mutation(tmp_path: Path) -> None:
    fit_dir = tmp_path / "fits"
    resume_root = tmp_path / "resume"
    resume_root.mkdir()
    fingerprint = "a" * 64
    _prepare_project_fit_dir(
        fit_dir=fit_dir,
        semantic_fingerprint=fingerprint,
        concept_ids=(0,),
    )
    path = _project_fit_path(fit_dir, 0)
    _save_project_fit(
        path,
        t.ones(8),
        concept_id=0,
        semantic_fingerprint=fingerprint,
    )
    _install_project_fit_receipt(
        resume_root=resume_root,
        fit_dir=fit_dir,
        semantic_fingerprint=fingerprint,
        concept_ids=(0,),
    )
    _save_project_fit(
        path,
        t.zeros(8),
        concept_id=0,
        semantic_fingerprint=fingerprint,
    )

    with pytest.raises(RuntimeError, match="fit-artifact receipt"):
        _install_project_fit_receipt(
            resume_root=resume_root,
            fit_dir=fit_dir,
            semantic_fingerprint=fingerprint,
            concept_ids=(0,),
        )


def test_project_driver_real_gloo_sharding_and_changed_world_resume_are_byte_exact(
    tmp_path: Path,
    gloo_multiprocess: Any,
) -> None:
    from tests.scripts.eval.test_axbench_project_methods_driver import _write_inputs

    kwargs = _write_inputs(tmp_path)
    first = gloo_multiprocess(4, _project_gloo_worker, kwargs)

    output_path = Path(kwargs["output_path"])
    manifest_path = Path(kwargs["manifest_path"])
    first_output = output_path.read_bytes()
    first_manifest = manifest_path.read_bytes()
    first_output_sha = hashlib.sha256(first_output).hexdigest()
    first_manifest_sha = hashlib.sha256(first_manifest).hexdigest()

    assert [result["rows"] for result in first] == [240, 240, 240, 240]
    assert sum(result["generation_count"] for result in first) == 480
    assert [result["generation_count"] for result in first] == [120, 120, 120, 120]
    assert [result["batch_sizes"] for result in first] == [[24] * 5] * 4
    batch_pairs = [set(result["batch_pairs"]) for result in first]
    assert all(len(pairs) == 5 for pairs in batch_pairs)
    assert all(
        left.isdisjoint(right)
        for left_index, left in enumerate(batch_pairs)
        for right in batch_pairs[left_index + 1 :]
    )
    assert set().union(*batch_pairs) == {
        (concept_id, input_id) for concept_id in range(2) for input_id in range(10)
    }
    assert [result["fit_events"] for result in first] == [
        ["fit-rank-0"],
        ["fit-rank-1"],
        [],
        [],
    ]
    fit_dir = manifest_path.parent / "_project_fit_resume_v1"
    assert sorted(path.name for path in fit_dir.glob("*.safetensors")) == [
        "dense_diffmean_00000.safetensors",
        "dense_diffmean_00001.safetensors",
    ]

    output_path.unlink()
    manifest_path.unlink()
    resumed = gloo_multiprocess(3, _project_gloo_worker, kwargs)

    assert [result["rows"] for result in resumed] == [240, 240, 240]
    assert [result["generation_count"] for result in resumed] == [0, 0, 0]
    assert [result["batch_pairs"] for result in resumed] == [[], [], []]
    assert [result["batch_sizes"] for result in resumed] == [[], [], []]
    assert [result["fit_events"] for result in resumed] == [[], [], []]
    assert hashlib.sha256(output_path.read_bytes()).hexdigest() == first_output_sha
    assert hashlib.sha256(manifest_path.read_bytes()).hexdigest() == first_manifest_sha
    assert output_path.read_bytes() == first_output
    assert manifest_path.read_bytes() == first_manifest


def test_encdec_driver_real_gloo_sharding_and_changed_world_resume_are_byte_exact(
    tmp_path: Path,
    gloo_multiprocess: Any,
) -> None:
    from tests.scripts.eval.test_axbench_encdec_driver import _write_inputs

    kwargs = _write_inputs(tmp_path)
    first = gloo_multiprocess(2, _encdec_gloo_worker, kwargs)

    main_path = Path(kwargs["output_main_path"])
    aux_path = Path(kwargs["output_aux_path"])
    manifest_path = Path(kwargs["manifest_path"])
    first_main = main_path.read_bytes()
    first_aux = aux_path.read_bytes()
    first_manifest = manifest_path.read_bytes()

    assert [result["main_rows"] for result in first] == [240, 240]
    assert [result["aux_rows"] for result in first] == [20, 20]
    assert sum(result["generation_count"] for result in first) == 520
    assert all(result["generation_count"] > 0 for result in first)
    assert [result["direction_batch_sizes"] for result in first] == [[24] * 10, [24] * 10]
    first_batch_pairs = [set(result["direction_batch_pairs"]) for result in first]
    assert first_batch_pairs[0].isdisjoint(first_batch_pairs[1])
    assert first_batch_pairs[0] | first_batch_pairs[1] == {
        (concept_id, input_id) for concept_id in range(2) for input_id in range(10)
    }
    assert first[0]["fit_events"] == ["fit-diffmean:0", "fit-lora:0", "fit-probe:0"]
    assert first[1]["fit_events"] == ["fit-diffmean:1", "fit-lora:1", "fit-probe:1"]
    assert sorted(path.name for path in Path(kwargs["train_dir"]).glob("*.safetensors")) == [
        "concept_00000_diffmean.safetensors",
        "concept_00000_linear_probe.safetensors",
        "concept_00000_lora.safetensors",
        "concept_00001_diffmean.safetensors",
        "concept_00001_linear_probe.safetensors",
        "concept_00001_lora.safetensors",
    ]

    main_path.unlink()
    aux_path.unlink()
    manifest_path.unlink()
    resumed = gloo_multiprocess(3, _encdec_gloo_worker, kwargs)

    assert [result["main_rows"] for result in resumed] == [240, 240, 240]
    assert [result["aux_rows"] for result in resumed] == [20, 20, 20]
    assert [result["generation_count"] for result in resumed] == [0, 0, 0]
    assert [result["direction_batch_sizes"] for result in resumed] == [[], [], []]
    assert [result["direction_batch_pairs"] for result in resumed] == [[], [], []]
    assert [result["fit_events"] for result in resumed] == [[], [], []]
    assert main_path.read_bytes() == first_main
    assert aux_path.read_bytes() == first_aux
    assert manifest_path.read_bytes() == first_manifest


def test_project_driver_rank_local_preparation_failure_aborts_both_real_gloo_ranks(
    tmp_path: Path,
    gloo_multiprocess: Any,
) -> None:
    from tests.scripts.eval.test_axbench_project_methods_driver import _write_inputs

    kwargs = _write_inputs(tmp_path)
    messages = gloo_multiprocess(2, _project_gloo_rank_failure_worker, kwargs)

    assert all("project conditioning preparation" in message for message in messages)
    assert all("rank 1" in message for message in messages)
    assert all("rank-one synthetic conditioning failure" in message for message in messages)
    assert not Path(kwargs["output_path"]).exists()
    assert not Path(kwargs["manifest_path"]).exists()
    resume_root = Path(kwargs["manifest_path"]).parent / "_project_cell_resume_v1"
    assert not list(resume_root.glob("rank_*/batch_*.parquet"))
