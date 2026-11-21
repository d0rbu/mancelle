"""Strict unit contracts for issue-#1479 startup and step timing."""

from __future__ import annotations

import inspect
import json

import pytest

import scripts.train.conditional_encdec_selfcond as selfcond
import scripts.train.conditional_encdec_sft as sft_script


class _Clock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def _payloads(monkeypatch: pytest.MonkeyPatch) -> list[dict]:
    payloads: list[dict] = []
    monkeypatch.setattr(
        selfcond.logger,
        "info",
        lambda message: payloads.append(json.loads(message.removeprefix("selfcond_startup "))),
    )
    return payloads


def test_trace_buffers_pre_ddp_phases_and_flushes_rank_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=clock)

    with trace.phase("hydra", config_name="conditional_encdec_selfcond"):
        clock.advance(2.0)
    assert payloads == []

    trace.set_master_process(True)
    assert [payload["phase"] for payload in payloads] == ["hydra"]
    assert payloads[0]["elapsed_s"] == 2.0
    assert payloads[0]["cumulative_s"] == 2.0

    with trace.phase("checkpoint"):
        clock.advance(3.5)
    assert [payload["phase"] for payload in payloads] == ["hydra", "checkpoint"]
    assert payloads[1]["elapsed_s"] == 3.5
    assert payloads[1]["cumulative_s"] == 5.5


def test_trace_summary_freezes_time_to_first_step_and_reports_steady_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=clock)
    trace.set_master_process(True)

    with trace.phase("setup"):
        clock.advance(4.0)
    clock.advance(1.0)
    trace.mark_first_step()
    clock.advance(100.0)
    trace.complete(first_step_s=7.0, steady_state_s=[2.0, 4.0])

    summary = payloads[-1]
    assert summary == {
        "first_step_s": 7.0,
        "kind": "summary",
        "named_phase_s": 4.0,
        "scope": "main_entry_to_first_completed_step",
        "status": "ok",
        "steady_state_mean_s": 3.0,
        "steady_state_step_count": 2,
        "total_s": 5.0,
        "unattributed_s": 1.0,
    }
    trace.complete(first_step_s=99.0, steady_state_s=[])
    assert payloads[-1] == summary


def test_trace_summary_deduplicates_nested_phase_intervals(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=clock)
    trace.set_master_process(True)

    with trace.phase("outer"):
        clock.advance(2.0)
        with trace.phase("inner"):
            clock.advance(3.0)
        clock.advance(1.0)

    trace.complete(first_step_s=None, steady_state_s=[])

    assert payloads[-1]["total_s"] == 6.0
    assert payloads[-1]["named_phase_s"] == 6.0
    assert payloads[-1]["unattributed_s"] == 0.0


def test_trace_accepts_shared_sft_leaf_phase_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=clock)
    trace.set_master_process(True)

    with trace.phase("sft_artifact_identity", extra=lambda: {"documents": 4}):
        clock.advance(1.0)
    clock.advance(2.0)
    trace.record(
        "sft_initial_batch",
        2_000_000_000,
        extra={"batch_rows": 1},
    )

    assert payloads[0]["documents"] == 4
    assert payloads[1]["phase"] == "sft_initial_batch"
    assert payloads[1]["batch_rows"] == 1
    assert payloads[1]["status"] == "ok"


def test_trace_failure_preserves_error_type_and_emits_phase_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=clock)

    with pytest.raises(RuntimeError, match="sentinel"):
        with trace.phase("model_restore", checkpoint="base"):
            clock.advance(1.25)
            raise RuntimeError("sentinel")

    trace.failed(RuntimeError("sentinel"))
    trace.ensure_failure_visible()
    assert payloads[0]["kind"] == "phase"
    assert payloads[0]["status"] == "error"
    assert payloads[0]["error_type"] == "RuntimeError"
    assert payloads[-1] == {
        "error_type": "RuntimeError",
        "kind": "summary",
        "named_phase_s": 1.25,
        "scope": "main_entry_to_failure",
        "status": "error",
        "total_s": 1.25,
        "unattributed_s": 0.0,
    }


def test_trace_non_master_does_not_emit_info(monkeypatch: pytest.MonkeyPatch) -> None:
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=_Clock())
    trace.set_master_process(False)
    with trace.phase("rank_local"):
        pass
    trace.complete(first_step_s=None, steady_state_s=[])
    assert payloads == []


def test_pre_ddp_failure_visibility_uses_launcher_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads(monkeypatch)
    monkeypatch.setenv("RANK", "1")
    trace = selfcond._SelfCondStartupTrace(clock=_Clock())
    with trace.phase("early_failure"):
        pass
    trace.failed(RuntimeError("sentinel"))
    trace.ensure_failure_visible()
    assert payloads == []


def test_trace_summary_labels_zero_step_run_without_claiming_a_first_step(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = _Clock()
    payloads = _payloads(monkeypatch)
    trace = selfcond._SelfCondStartupTrace(clock=clock)
    trace.set_master_process(True)
    clock.advance(2.5)

    trace.complete(first_step_s=None, steady_state_s=[])

    assert payloads[-1]["scope"] == "main_entry_to_training_end_without_completed_step"
    assert payloads[-1]["first_step_s"] is None
    assert payloads[-1]["steady_state_step_count"] == 0


def test_startup_phase_is_a_no_op_without_a_trace() -> None:
    with selfcond._startup_phase(None, "not_traced"):
        pass


def test_public_sft_generator_forwards_startup_observers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def fake_impl(**kwargs: object) -> object:
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(sft_script, "_make_sft_data_generator_impl", fake_impl)

    startup_trace = selfcond._SelfCondStartupTrace(clock=_Clock())

    result = sft_script.make_sft_data_generator(
        task=[],
        tokenizer=object(),
        B=1,
        T=1,
        device="cpu",
        sft_state=sft_script.SFTState(),
        startup_trace=startup_trace,
    )

    assert result is not None
    assert captured["startup_trace"] is startup_trace


def test_main_declares_issue_1479_phase_boundaries_without_cuda_sync() -> None:
    source = inspect.getsource(selfcond.main)
    helper_source = "\n".join(
        (
            inspect.getsource(selfcond.build_selfcond_sft_data),
            inspect.getsource(sft_script._make_sft_data_generator_impl),
        )
    )
    all_source = f"{source}\n{helper_source}"
    for phase in (
        "config_and_hydra_resolution",
        "checkpoint_resolution_and_metadata_load",
        "model_construction_and_checkpoint_restore",
        "sft_feature_mask_reconstruction",
        "sft_cluster_consolidation",
        "sft_effective_feature_vocab",
        "sft_percentile_table_load",
        "sft_artifact_identity",
        "sft_document_coverage",
        "sft_feature_row_coverage",
        "sft_alignment_probe",
        "sft_initial_batch_materialization",
        "first_optimizer_step",
    ):
        assert phase in all_source
    assert "compile_and_graph_warmup" in source
    assert "torch.cuda.synchronize" not in source


def test_main_flushes_a_pre_ddp_configuration_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payloads = _payloads(monkeypatch)
    monkeypatch.delenv("RANK", raising=False)

    def fail_config(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("config sentinel")

    monkeypatch.setattr(selfcond, "parse_config_name_and_overrides", fail_config)
    monkeypatch.setattr(selfcond.sys, "argv", ["conditional_encdec_selfcond"])

    with pytest.raises(RuntimeError, match="config sentinel"):
        selfcond.main()

    assert payloads[0]["kind"] == "phase"
    assert payloads[0]["phase"] == "config_and_hydra_resolution"
    assert payloads[0]["status"] == "error"
    assert payloads[0]["error_type"] == "RuntimeError"
    assert payloads[-1]["kind"] == "summary"
    assert payloads[-1]["scope"] == "main_entry_to_failure"
