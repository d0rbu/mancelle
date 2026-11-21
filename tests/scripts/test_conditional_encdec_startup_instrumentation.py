"""Route and source-contract tests for issue #1481 startup instrumentation."""

from __future__ import annotations

import ast
import inspect
from collections.abc import Callable
from pathlib import Path

import pytest

from scripts.train.conditional_encdec_train import (
    _describe_effective_data_path,
    _startup_phase,
    _startup_skip,
    build_shared_state,
    train_one_config,
)
from steering.profiling import StartupTrace


@pytest.mark.parametrize(
    ("packed", "selection", "ignored", "use_precomputed", "remote", "expected"),
    [
        (
            Path("packed"),
            Path("selection"),
            False,
            True,
            False,
            (True, "precomputed", "selection_shards", "local", "selected"),
        ),
        (
            Path("packed"),
            None,
            False,
            True,
            False,
            (True, "precomputed", "fused_base_csr", "local", "selected"),
        ),
        (
            None,
            None,
            False,
            True,
            False,
            (False, "assembled", "live_feature_pipeline", "local", "packed_rows_unavailable"),
        ),
        (
            None,
            None,
            False,
            False,
            False,
            (False, "assembled", "live_feature_pipeline", "local", "precomputed_disabled"),
        ),
        (
            None,
            None,
            True,
            True,
            False,
            (False, "assembled", "live_feature_pipeline", "local", "feature_delta"),
        ),
        (
            Path("packed"),
            Path("selection"),
            False,
            True,
            True,
            (True, "precomputed", "selection_shards", "remote_certified", "selected"),
        ),
    ],
)
def test_effective_route_taxonomy_is_exact(
    packed: Path | None,
    selection: Path | None,
    ignored: bool,
    use_precomputed: bool,
    remote: bool,
    expected: tuple[object, ...],
) -> None:
    route = _describe_effective_data_path(
        packed_row_dir=packed,
        selection_dir=selection,
        ignored_packed_rows_for_delta=ignored,
        use_precomputed_batches=use_precomputed,
        remote_certified=remote,
    )
    assert (
        route.packed_rows,
        route.route,
        route.selection,
        route.source,
        route.reason,
    ) == expected


def test_route_event_preserves_explicit_selection_and_resolution_fields() -> None:
    lines: list[str] = []
    trace = StartupTrace(clock=lambda: 10.0, emit=lines.append)
    route = _describe_effective_data_path(
        packed_row_dir=Path("packed"),
        selection_dir=None,
        ignored_packed_rows_for_delta=False,
        use_precomputed_batches=True,
        remote_certified=False,
    )
    trace.event(
        "encdec_startup_data_path",
        packed_rows=route.packed_rows,
        route=route.route,
        selection=route.selection,
        source=route.source,
        resolution_s=1.5,
        decision_s=0.25,
    )
    assert lines == [
        "encdec_startup_data_path packed_rows=yes route=precomputed "
        "selection=fused_base_csr source=local resolution_s=1.500000 decision_s=0.250000"
    ]


def test_disabled_trace_context_runs_body_and_emits_nothing() -> None:
    ran: list[str] = []
    with _startup_phase(None, "phase"):
        ran.append("phase")
    _startup_skip(None, "optional")
    assert ran == ["phase"]


def test_trainer_signatures_keep_trace_keyword_only_and_legacy_arguments() -> None:
    shared_params = inspect.signature(build_shared_state).parameters
    train_params = inspect.signature(train_one_config).parameters
    assert shared_params["gpu_index"].kind is inspect.Parameter.KEYWORD_ONLY
    assert shared_params["startup_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert train_params["startup_trace"].kind is inspect.Parameter.KEYWORD_ONLY
    assert list(train_params)[:3] == ["cfg", "shared", "wandb_run"]


def _function_tree(function: Callable[..., object]) -> ast.FunctionDef:
    source = inspect.getsource(function)
    tree = ast.parse(source)
    node = tree.body[0]
    assert isinstance(node, ast.FunctionDef)
    return node


def test_build_and_train_source_contains_all_required_phase_names() -> None:
    names = {
        node.args[1].value
        for function in (build_shared_state, train_one_config)
        for node in ast.walk(_function_tree(function))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "_startup_phase"
        and len(node.args) > 1
        and isinstance(node.args[1], ast.Constant)
        and isinstance(node.args[1].value, str)
    }
    assert names == {
        "startup_validation",
        "device_ddp_init",
        "attention_backend",
        "tokenizer",
        "feature_source_resolution",
        "statistics_checksums",
        "feature_index_overlays",
        "boundaries_and_feature_identity",
        "model_materialization",
        "packed_selection_artifacts",
        "compile_wrapper",
        "eval_contexts",
        "config_identity",
        "checkpoint_resume",
        "feature_pipeline_setup",
        "optimizer_setup",
        "effective_data_path",
        "dataloader_build",
        "first_batch_fetch",
        "pre_first_update_control",
    }


def test_existing_step_timer_still_has_one_post_optimizer_sync() -> None:
    source = inspect.getsource(train_one_config)
    tree = ast.parse(source)
    sync_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "synchronize"
    ]
    assert len(sync_calls) == 1
    assert "startup_trace.observe_step" in source
    assert "dt = t1 - t0" in source


def test_fast_sweep_keeps_legacy_call_and_startup_time_serialization() -> None:
    sweep_path = Path(__file__).resolve().parents[2] / "scripts/train/run_encdec_sweep_fast.py"
    source = sweep_path.read_text()
    assert "train_one_config(per_cfg, shared, wandb_run)" in source
    assert "startup_trace=" not in source
    assert '"startup_time": _sanitize(round(result.startup_time, 2))' in source
