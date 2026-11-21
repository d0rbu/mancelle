"""Slow, model-backed parity checks for the startup trace seam.

The repository's complete trainer has many production-only dependencies (Hydra,
W&B, feature-index overlays, and checkpoint stores). This test deliberately
holds those inputs constant at the existing precomputed-dataloader seam and
executes real forward/backward/optimizer work. It proves that observing the
same post-step boundary cannot perturb losses, gradients, or optimizer state.
The opt-in production artifact gate is documented separately and is never
replaced by this small deterministic fixture.
"""

from __future__ import annotations

import copy
import time
from pathlib import Path

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.profiling import StartupTrace
from tests.encdec_test_helpers import (
    ForwardResult,
    build_tiny_model,
    make_precomputed_dataloader,
    setup_optimizer_simple,
)

pytestmark = pytest.mark.slow


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _run_steps(
    root: Path,
    *,
    trace_enabled: bool,
    start_step: int = 0,
) -> tuple[list[float], dict[str, t.Tensor], dict, list[str]]:
    t.manual_seed(42)
    model = build_tiny_model()
    optimizer = setup_optimizer_simple(model)
    dataloader = make_precomputed_dataloader(root, B=2, n_rows=32)
    lines: list[str] = []
    trace = (
        StartupTrace(clock=time.perf_counter, emit=lines.append)
        if trace_enabled
        else None
    )
    losses: list[float] = []
    model.train()
    for step in range(start_step, start_step + 12):
        optimizer.zero_grad(set_to_none=True)
        t0 = time.perf_counter()
        (
            x,
            y,
            enc_fids,
            enc_mags,
            enc_mask,
            _enc_pcts,
            seg_map,
            seg_doc_id,
            _fp_targets,
            _state,
        ) = next(dataloader)
        result = model(
            x,
            y,
            enc_fids,
            enc_mags,
            enc_mask,
            seg_map,
            seg_doc_id=seg_doc_id,
        )
        loss = result.loss if isinstance(result, ForwardResult) else result
        loss.backward()
        optimizer.step()
        t1 = time.perf_counter()
        losses.append(float(loss.detach().item()))
        if trace is not None:
            trace.observe_step(
                update_step=step,
                step_s=t1 - t0,
                compile_configured=False,
            )
    summary = (
        dict(trace.finish(status="complete"))
        if trace is not None
        else {}
    )
    state = {
        key: value.detach().clone()
        for key, value in model.state_dict().items()
        if t.is_tensor(value)
    }
    return losses, state, summary, lines


def _assert_optimizer_state_equal(left: dict, right: dict) -> None:
    assert left.keys() == right.keys()
    assert left["param_groups"] == right["param_groups"]
    assert left["state"].keys() == right["state"].keys()
    for key in left["state"]:
        left_state = left["state"][key]
        right_state = right["state"][key]
        assert left_state.keys() == right_state.keys()
        for field in left_state:
            left_value = left_state[field]
            right_value = right_state[field]
            if t.is_tensor(left_value):
                assert t.equal(left_value, right_value)
            else:
                assert left_value == right_value


def test_trace_observation_preserves_real_model_losses_and_weights(tmp_path: Path) -> None:
    baseline_losses, baseline_state, _baseline_summary, _baseline_lines = _run_steps(
        tmp_path / "baseline",
        trace_enabled=False,
    )
    traced_losses, traced_state, summary, lines = _run_steps(
        tmp_path / "traced",
        trace_enabled=True,
    )

    assert traced_losses == baseline_losses
    assert traced_state.keys() == baseline_state.keys()
    for key in baseline_state:
        assert t.equal(traced_state[key], baseline_state[key]), key
    assert summary["first_update_step"] == 0
    assert summary["steady_state_steps"] == 1
    assert summary["first_step_s"] >= 0.0
    assert any(line.startswith("encdec_startup_first_step ") for line in lines)
    assert lines[-1].startswith("encdec_startup_summary status=complete")


def test_trace_resume_accounting_uses_nonzero_first_step(tmp_path: Path) -> None:
    _losses, _state, summary, lines = _run_steps(
        tmp_path / "resumed",
        trace_enabled=True,
        start_step=500,
    )

    assert summary["first_update_step"] == 500
    assert summary["steady_state_steps"] == 11
    assert summary["steady_state_step_s"] is not None
    first_line = next(line for line in lines if line.startswith("encdec_startup_first_step "))
    assert "update_step=500" in first_line
    assert "compile_configured=no" in first_line


def test_trace_does_not_change_optimizer_state_when_compared_directly(
    tmp_path: Path,
) -> None:
    """Compare the complete optimizer state, not only the model parameters."""
    t.manual_seed(42)
    model_a = build_tiny_model()
    optimizer_a = setup_optimizer_simple(model_a)
    dataloader_a = make_precomputed_dataloader(tmp_path / "a", B=2, n_rows=8)

    t.manual_seed(42)
    model_b = build_tiny_model()
    optimizer_b = setup_optimizer_simple(model_b)
    dataloader_b = make_precomputed_dataloader(tmp_path / "b", B=2, n_rows=8)
    trace = StartupTrace(clock=time.perf_counter, emit=lambda _line: None)

    for step in range(2):
        for model, optimizer, dataloader in (
            (model_a, optimizer_a, dataloader_a),
            (model_b, optimizer_b, dataloader_b),
        ):
            optimizer.zero_grad(set_to_none=True)
            batch = next(dataloader)
            result = model(
                batch[0],
                batch[1],
                batch[2],
                batch[3],
                batch[4],
                batch[6],
                seg_doc_id=batch[7],
            )
            loss = result.loss if isinstance(result, ForwardResult) else result
            loss.backward()
            optimizer.step()
        trace.observe_step(update_step=step, step_s=1.0, compile_configured=False)

    _assert_optimizer_state_equal(
        copy.deepcopy(optimizer_a.state_dict()),
        copy.deepcopy(optimizer_b.state_dict()),
    )
    for key, value in model_a.state_dict().items():
        assert t.equal(value, model_b.state_dict()[key]), key
