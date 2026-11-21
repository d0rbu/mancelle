"""Strict startup-telemetry contracts for conditioned encoder-decoder SFT.

These tests deliberately exercise the new ledger independently from optional
live-loop profiling and then exercise it through the real lazy SFT generator.
The generator tests use the same production-shaped tensor/artifact helpers as
the conditioned-SFT integration suite and compare telemetry-on/off output
exactly.  Full-size GPU execution is an explicit opt-in gate documented at the
bottom of this module; it is never replaced by a toy model claim.
"""

from __future__ import annotations

import inspect
import os
import re
import shlex
import subprocess
from pathlib import Path
from unittest.mock import patch

import pytest
import torch as t

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from scripts.preprocess.convert_sft_to_parquet import (
    render_conversation_to_full_nanochat_text,
)
from scripts.train import conditional_encdec_sft as sft
from steering.conditional.encoder_decoder.sft_features import (
    SFTFeatureLoadReport,
    load_sft_features,
)
from steering.sft_schema import SFTSplit
from tests.encdec_test_helpers import N_SAE_FEATURES, NUM_BUCKETS, build_tiny_model
from tests.steering.test_encdec_sft_conditioned_integration import (
    SFTMockTokenizer,
    SingleTurnTask,
    build_synthetic_features,
    patch_load_sft_features,
)
from tests.steering.test_sft_features import _write_parquet, _write_shard


def _trace_records(
    trace: sft._SFTStartupTrace,
) -> tuple[sft._SFTStartupPhaseRecord, ...]:
    return tuple(trace._records)


def _record_map(trace: sft._SFTStartupTrace) -> dict[str, sft._SFTStartupPhaseRecord]:
    """Index unique phase records and fail loudly on accidental duplicates."""
    records = _trace_records(trace)
    names = [record.name for record in records]
    assert len(names) == len(set(names)), f"duplicate phases: {names}"
    return {record.name: record for record in records}


@pytest.fixture(autouse=True)
def _force_sdpa_backend():
    """Keep the CPU backward parity gate on the supported attention backend."""
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


class TestSftStartupTraceUnit:
    """The monotonic ledger itself has a complete, deterministic contract."""

    def test_phase_uses_monotonic_clock_and_sorts_metadata(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        now = iter([100, 175, 425])
        monkeypatch.setattr(sft.time, "perf_counter_ns", lambda: next(now))
        lines: list[str] = []
        monkeypatch.setattr(sft.logger, "info", lines.append)

        trace = sft._SFTStartupTrace()
        with trace.phase("mixture_build", extra={"zeta": True, "alpha": 3}):
            pass

        assert trace.start_ns == 100
        assert trace.phase_sum_ns == 250
        assert _trace_records(trace) == (
            sft._SFTStartupPhaseRecord(
                name="mixture_build",
                elapsed_ns=250,
                status="completed",
                extra=(("alpha", 3), ("zeta", True)),
            ),
        )
        assert lines == [
            "SFT startup phase=mixture_build status=completed "
            "elapsed_s=0.000000 alpha=3 zeta=true"
        ]

    def test_dynamic_metadata_is_evaluated_after_the_phase(self) -> None:
        lines: list[str] = []
        with patch.object(sft.logger, "info", lines.append):
            trace = sft._SFTStartupTrace(start_ns=0)
            observed: dict[str, object] = {"documents": 0}
            with trace.phase("sft_document_coverage", extra=lambda: observed):
                observed["documents"] = 123

        assert _trace_records(trace)[0].extra == (("documents", 123),)
        assert "documents=123" in lines[0]

    @pytest.mark.parametrize(
        ("name", "elapsed_ns", "status", "match"),
        [
            ("", 1, "completed", "one token"),
            ("has space", 1, "completed", "one token"),
            ("phase", -1, "completed", "non-negative"),
            ("phase", 1, "unknown", "unsupported"),
        ],
    )
    def test_record_rejects_malformed_contract(
        self,
        name: str,
        elapsed_ns: int,
        status: str,
        match: str,
    ) -> None:
        trace = sft._SFTStartupTrace(start_ns=0)
        with pytest.raises(ValueError, match=match):
            trace.record(name, elapsed_ns, status=status)

    def test_failed_phase_logs_and_preserves_original_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lines: list[str] = []
        monkeypatch.setattr(sft.logger, "info", lines.append)
        trace = sft._SFTStartupTrace(start_ns=0)

        with pytest.raises(RuntimeError, match="original failure"):
            with trace.phase("checkpoint_load"):
                raise RuntimeError("original failure")

        assert _trace_records(trace)[0].status == "failed"
        assert "status=failed" in lines[0]

    def test_overlapping_phases_are_rejected_without_masking_outer_failure(self) -> None:
        trace = sft._SFTStartupTrace(start_ns=0)
        with trace.phase("outer"):
            with pytest.raises(RuntimeError, match="must not overlap"):
                with trace.phase("inner"):
                    pass
        assert [record.name for record in _trace_records(trace)] == ["outer"]

    def test_excluded_phase_is_visible_but_not_in_breakdown(self) -> None:
        trace = sft._SFTStartupTrace(start_ns=0)
        trace.record("included", 100)
        trace.record("context_only", 900, include_in_sum=False)
        assert trace.phase_sum_ns == 100
        assert [record.name for record in _trace_records(trace)] == ["included", "context_only"]

    def test_first_step_summary_is_monotonic_and_rank_annotated(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lines: list[str] = []
        monkeypatch.setattr(sft.logger, "info", lines.append)
        trace = sft._SFTStartupTrace(start_ns=100)
        trace.set_ddp(rank=0, world_size=4)
        trace.record("startup", 200)

        trace.record_first_step(completed_ns=1_100, step_elapsed_ns=700, step=0)

        assert trace.first_step_elapsed_ns == 700
        assert trace.phase_sum_ns == 200
        assert _trace_records(trace)[-1] == sft._SFTStartupPhaseRecord(
            name="first_optimizer_step",
            elapsed_ns=700,
            status="completed",
            extra=(("step", 0),),
            include_in_sum=False,
        )
        assert "time_to_first_step_s=0.000001" in lines[-1]
        assert "startup_phase_sum_s=0.000000" in lines[-1]
        assert "first_step_warmup_s=0.000001" in lines[-1]
        assert "post_first_step_steps=0" in lines[-1]
        assert "ddp_rank=0" in lines[-1]
        assert "ddp_world_size=4" in lines[-1]

    def test_steady_state_summary_excludes_post_startup_validation_phases(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lines: list[str] = []
        monkeypatch.setattr(sft.logger, "info", lines.append)
        trace = sft._SFTStartupTrace(start_ns=0)
        trace.record("startup", 100)
        trace.record_first_step(completed_ns=1_000, step_elapsed_ns=700, step=3)
        # Validation happens after the first step and must not alter the startup
        # phase sum used by the closing breakdown.
        trace.record("validation_feature_store", 50)
        trace.log_steady_state(
            time_to_first_step_ns=1_000,
            post_first_step_total_ns=600,
            post_first_step_steps=2,
        )

        assert "startup_phase_sum_s=0.000000" in lines[-1]
        assert "first_step_warmup_s=0.000001" in lines[-1]
        assert "post_first_step_mean_s=0.000000" in lines[-1]
        assert "post_first_step_steps=2" in lines[-1]
        assert trace.phase_sum_ns == 150

    def test_no_step_summary_is_explicitly_not_reached(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        lines: list[str] = []
        monkeypatch.setattr(sft.logger, "info", lines.append)
        trace = sft._SFTStartupTrace(start_ns=100)
        trace.record("config", 25)
        trace.log_no_step_summary(completed_ns=500)

        assert "status=not_reached" in lines[-1]
        assert "first_step_warmup_s=null" in lines[-1]
        assert "post_first_step_mean_s=null" in lines[-1]

    def test_first_step_cannot_be_recorded_twice(self) -> None:
        trace = sft._SFTStartupTrace(start_ns=0)
        trace.record_first_step(completed_ns=10, step_elapsed_ns=5)
        with pytest.raises(RuntimeError, match="recorded twice"):
            trace.record_first_step(completed_ns=20, step_elapsed_ns=5)

    def test_first_step_completion_cannot_precede_start(self) -> None:
        trace = sft._SFTStartupTrace(start_ns=10)
        with pytest.raises(ValueError, match="precedes startup start"):
            trace.record_first_step(completed_ns=9, step_elapsed_ns=1)

    def test_ddp_metadata_is_not_added_to_early_phase_lines(self) -> None:
        lines: list[str] = []
        with patch.object(sft.logger, "info", lines.append):
            trace = sft._SFTStartupTrace(start_ns=0)
            trace.record("hydra_config", 1)
            trace.set_ddp(rank=0, world_size=2)
            trace.record_first_step(completed_ns=10, step_elapsed_ns=5)

        assert "ddp_rank" not in lines[0]
        assert "ddp_rank=0" in lines[-1]

    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            (None, "null"),
            (True, "true"),
            (False, "false"),
            (1.23456789, "1.234568"),
            (17, "17"),
            ("train", "train"),
        ],
    )
    def test_machine_value_formatter_is_unambiguous(self, value: object, expected: str) -> None:
        assert sft._SFTStartupTrace._format_value(value) == expected

    def test_skipped_status_is_valid_and_excluded_when_requested(self) -> None:
        lines: list[str] = []
        with patch.object(sft.logger, "info", lines.append):
            trace = sft._SFTStartupTrace(start_ns=0)
            record = trace.record(
                "optional_phase",
                42,
                status="skipped",
                extra={"reason": "disabled"},
                include_in_sum=False,
            )

        assert record.status == "skipped"
        assert record.include_in_sum is False
        assert trace.phase_sum_ns == 0
        assert lines == [
            "SFT startup phase=optional_phase status=skipped "
            "elapsed_s=0.000000 reason=disabled"
        ]

    def test_no_step_summary_is_idempotent(self, monkeypatch: pytest.MonkeyPatch) -> None:
        lines: list[str] = []
        monkeypatch.setattr(sft.logger, "info", lines.append)
        trace = sft._SFTStartupTrace(start_ns=10)
        trace.log_no_step_summary(completed_ns=20)
        trace.log_no_step_summary(completed_ns=30)

        assert len(lines) == 1
        assert "time_to_first_step_s=0.000000" in lines[0]

    def test_steady_state_without_first_step_remains_not_reached(self) -> None:
        lines: list[str] = []
        with patch.object(sft.logger, "info", lines.append):
            trace = sft._SFTStartupTrace(start_ns=0)
            trace.record("config", 10)
            trace.log_steady_state(
                time_to_first_step_ns=50,
                post_first_step_total_ns=0,
                post_first_step_steps=0,
            )

        assert "summary=steady_state" in lines[-1]
        assert "status=not_reached" in lines[-1]
        assert "post_first_step_mean_s=null" in lines[-1]

    def test_log_no_step_after_first_step_does_not_emit_duplicate_summary(self) -> None:
        lines: list[str] = []
        with patch.object(sft.logger, "info", lines.append):
            trace = sft._SFTStartupTrace(start_ns=0)
            trace.record_first_step(completed_ns=20, step_elapsed_ns=5)
            before = len(lines)
            trace.log_no_step_summary(completed_ns=30)

        assert len(lines) == before == 2
        assert "phase=first_optimizer_step" in lines[0]
        assert "summary=time_to_first_step" in lines[1]

    def test_main_source_keeps_startup_trace_independent_of_live_profiling(self) -> None:
        source = inspect.getsource(sft.main)
        for phase_name in (
            "hydra_config",
            "compute_init",
            "base_checkpoint_load",
            "checkpoint_identity",
            "selection_identity",
            "tokenizer_load",
            "mixture_build",
            "model_construct",
            "model_materialize",
            "model_state_load",
            "compile_setup",
            "statistics_load",
            "feature_pipeline_setup",
        ):
            assert phase_name in source
        # The two percentile phases moved into the extracted preflight helper
        # (issue #1495): main must still call it, and the helper must own both
        # phases — asserted against the helper's OWN source so the phases
        # cannot silently detach from the startup path.
        assert "_percentile_coverage_preflight(" in source
        helper_source = inspect.getsource(sft._percentile_coverage_preflight)
        for phase_name in ("sft_percentile_id_scan", "sft_percentile_coverage"):
            assert phase_name in helper_source
        assert "_SFTStartupTrace()" in source
        assert "time.perf_counter_ns()" in source
        assert "record_first_step" in source
        assert "profile_timing" in source


class TestSftFeatureLoadReport:
    """The optional report observes the real feature/parquet loader."""

    def test_real_loader_reports_miss_then_weak_cache_hit(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 1])
        _write_parquet(parquet_dir, "shard_00000", [100, 200])

        first = SFTFeatureLoadReport()
        store = load_sft_features(str(feature_dir), str(parquet_dir), report=first)
        second = SFTFeatureLoadReport()
        cached = load_sft_features(str(feature_dir), str(parquet_dir), report=second)

        assert first.cache_hit is False
        assert first.feature_shards == 1
        assert first.parquet_shards == 1
        assert first.documents == 2
        assert second.cache_hit is True
        assert second.feature_shards == 1
        assert second.parquet_shards == 1
        assert second.documents == 2
        assert cached is store
        assert store.feature_shard_count == 1
        assert store.parquet_shard_count == 1

    def test_report_is_optional_and_does_not_change_mapping_contract(
        self,
        tmp_path: Path,
    ) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", [7])

        store = load_sft_features(str(feature_dir), str(parquet_dir))

        assert list(store) == [7]
        assert len(store) == 1
        assert store[7][0].feature_ids.tolist() == [10, 11]

    def test_report_extra_is_bounded_and_uses_source_label(self) -> None:
        report = SFTFeatureLoadReport(
            cache_hit=False,
            feature_shards=22,
            parquet_shards=22,
            documents=849_656,
        )
        assert sft._sft_feature_load_report_extra(report, source="train") == {
            "source": "train",
            "cache": "miss",
            "feature_shards": 22,
            "parquet_shards": 22,
            "documents": 849_656,
        }

    @pytest.mark.parametrize(
        ("report", "expected"),
        [
            (SFTFeatureLoadReport(), {"source": "val"}),
            (
                SFTFeatureLoadReport(cache_hit=True, documents=4),
                {"source": "val", "cache": "hit", "documents": 4},
            ),
            (
                SFTFeatureLoadReport(cache_hit=False, feature_shards=2),
                {"source": "val", "cache": "miss", "feature_shards": 2},
            ),
        ],
    )
    def test_report_extra_omits_unknown_counts_without_fabrication(
        self,
        report: SFTFeatureLoadReport,
        expected: dict[str, object],
    ) -> None:
        assert sft._sft_feature_load_report_extra(report, source="val") == expected

    def test_real_loader_reports_all_shards_and_documents(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0, 1])
        _write_shard(feature_dir, "shard_00001", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", [10, 11])
        _write_parquet(parquet_dir, "shard_00001", [12])

        report = SFTFeatureLoadReport()
        store = load_sft_features(str(feature_dir), str(parquet_dir), report=report)

        assert report == SFTFeatureLoadReport(
            cache_hit=False,
            feature_shards=2,
            parquet_shards=2,
            documents=3,
        )
        assert sorted(store) == [10, 11, 12]

    def test_report_remains_unmodified_when_loader_fails(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "missing-parquet"
        feature_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        report = SFTFeatureLoadReport()

        with pytest.raises(sft.DataLoadError, match="does not exist"):
            load_sft_features(str(feature_dir), str(parquet_dir), report=report)

        assert report == SFTFeatureLoadReport()

    def test_reused_report_is_cleared_before_a_failed_attempt(self, tmp_path: Path) -> None:
        feature_dir = tmp_path / "features"
        parquet_dir = tmp_path / "parquet"
        feature_dir.mkdir()
        parquet_dir.mkdir()
        _write_shard(feature_dir, "shard_00000", doc_ids=[0])
        _write_parquet(parquet_dir, "shard_00000", [9])
        report = SFTFeatureLoadReport()
        load_sft_features(str(feature_dir), str(parquet_dir), report=report)
        assert report.cache_hit is False

        with pytest.raises(sft.DataLoadError, match="does not exist"):
            load_sft_features(
                str(feature_dir), str(tmp_path / "removed-parquet"), report=report
            )

        assert report == SFTFeatureLoadReport()

    def test_startup_loader_keeps_legacy_call_when_trace_is_disabled(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[tuple[str, str]] = []
        sentinel = object()

        def fake_loader(feature_dir: str, parquet_dir: str):
            calls.append((feature_dir, parquet_dir))
            return sentinel

        monkeypatch.setattr(sft, "load_sft_features", fake_loader)

        result = sft._load_sft_features_for_startup(
            "features",
            "parquet",
            startup_trace=None,
            source="train",
        )

        assert result is sentinel
        assert calls == [("features", "parquet")]

    def test_startup_loader_records_real_report_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        sentinel = object()

        def fake_loader(
            feature_dir: str,
            parquet_dir: str,
            *,
            report: SFTFeatureLoadReport,
        ):
            assert (feature_dir, parquet_dir) == ("features", "parquet")
            report.cache_hit = True
            report.feature_shards = 22
            report.parquet_shards = 22
            report.documents = 849_656
            return sentinel

        monkeypatch.setattr(sft, "load_sft_features", fake_loader)
        trace = sft._SFTStartupTrace(start_ns=0)
        result = sft._load_sft_features_for_startup(
            "features",
            "parquet",
            startup_trace=trace,
            source="train",
        )

        assert result is sentinel
        record = _trace_records(trace)[0]
        assert record.name == "sft_feature_store_load"
        assert dict(record.extra) == {
            "cache": "hit",
            "documents": 849_656,
            "feature_shards": 22,
            "parquet_shards": 22,
            "source": "train",
        }

    def test_startup_loader_failure_is_reported_without_changing_exception(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def failing_loader(
            feature_dir: str,
            parquet_dir: str,
            *,
            report: SFTFeatureLoadReport,
        ):
            raise sft.DataLoadError(f"bad artifacts: {feature_dir}/{parquet_dir}")

        monkeypatch.setattr(sft, "load_sft_features", failing_loader)
        trace = sft._SFTStartupTrace(start_ns=0)

        with pytest.raises(sft.DataLoadError, match="bad artifacts"):
            sft._load_sft_features_for_startup(
                "features",
                "parquet",
                startup_trace=trace,
                source="val",
            )

        assert _trace_records(trace)[0].status == "failed"
        assert dict(_trace_records(trace)[0].extra) == {"source": "val"}

    def test_maybe_startup_phase_is_a_true_noop_without_trace(self) -> None:
        marker: list[str] = []
        with sft._maybe_sft_startup_phase(None, "not_recorded"):
            marker.append("body")
        assert marker == ["body"]

    def test_maybe_startup_phase_records_dynamic_metadata_with_trace(self) -> None:
        trace = sft._SFTStartupTrace(start_ns=0)
        values: dict[str, object] = {"rows": 0}
        with sft._maybe_sft_startup_phase(
            trace,
            "batch_build",
            extra=lambda: values,
        ):
            values["rows"] = 32
        assert _trace_records(trace) == (
            sft._SFTStartupPhaseRecord(
                name="batch_build",
                elapsed_ns=_trace_records(trace)[0].elapsed_ns,
                status="completed",
                extra=(("rows", 32),),
            ),
        )


class TestSftGeneratorTelemetryParity:
    """Telemetry-on and telemetry-off must emit identical training data."""

    def _build_batch(self, *, trace: sft._SFTStartupTrace | None):
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=5, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        with patch_load_sft_features(features):
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=sft.SFTState(),
                ddp_rank=0,
                ddp_world_size=1,
                buffer_size=100,
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(
                    sft.ThresholdType.segment_topk,
                    1000,
                ),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                min_loss_tokens=1,
                alignment_probe_samples=0,
                mixture_name="capabilities",
                expected_split=SFTSplit.TRAIN,
                startup_trace=trace,
            )
            return next(generator)

    def test_trace_disabled_and_enabled_batches_are_exactly_equal(self) -> None:
        baseline = self._build_batch(trace=None)
        trace = sft._SFTStartupTrace(start_ns=0)
        observed = self._build_batch(trace=trace)

        assert len(observed) == len(baseline)
        for expected, actual in zip(baseline, observed, strict=True):
            if expected is None:
                assert actual is None
            else:
                assert actual is not None
                assert t.equal(expected, actual)

        names = [record.name for record in _trace_records(trace)]
        assert names == [
            "sft_feature_store_load",
            "sft_artifact_identity",
            "sft_document_coverage",
            "sft_feature_row_coverage",
            "sft_alignment_probe",
            "sft_initial_batch",
        ]
        records = _record_map(trace)
        assert dict(records["sft_alignment_probe"].extra) == {
            "candidates": 5,
            "matched": 0,
            "requested": 0,
            "sampled": 0,
            "source": "train",
        }
        assert dict(records["sft_document_coverage"].extra)["documents"] == 5
        assert dict(records["sft_feature_row_coverage"].extra)["missing_documents"] == 0
        assert dict(records["sft_initial_batch"].extra)["batch_rows"] == 2

    def test_alignment_phase_reports_exact_sample_count(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=8, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        texts: dict[int, str] = {}
        for index in range(len(task)):
            text, skip_reason = render_conversation_to_full_nanochat_text(
                tokenizer, task[index]
            )
            assert text is not None, skip_reason
            texts[index] = text
        trace = sft._SFTStartupTrace(start_ns=0)
        with patch_load_sft_features(features, texts=texts):
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=sft.SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(sft.ThresholdType.segment_topk, 1000),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                min_loss_tokens=1,
                alignment_probe_samples=3,
                startup_trace=trace,
            )
            next(generator)
        alignment = [record for record in _trace_records(trace) if record.name == "sft_alignment_probe"]
        assert len(alignment) == 1
        assert dict(alignment[0].extra)["candidates"] == 8
        assert dict(alignment[0].extra) == {
            "candidates": 8,
            "matched": 3,
            "requested": 3,
            "sampled": 3,
            "source": "train",
        }

    def test_validation_trace_is_lazy_and_uses_val_source(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=4, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        trace = sft._SFTStartupTrace(start_ns=0)
        with patch_load_sft_features(features, split=SFTSplit.VAL):
            loader = sft._build_sft_val_loader(
                val_task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                ddp_rank=0,
                ddp_world_size=1,
                feature_dir="/dummy/features",
                expected_feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(sft.ThresholdType.segment_topk, 1000),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                precomputed_thresholds=None,
                sampling_bundle=None,
                loss_mask_mode=sft.LossMaskMode.FULL_SEQUENCE,
                min_loss_tokens=1,
                alignment_probe_samples=0,
                startup_trace=trace,
            )
            assert _trace_records(trace) == ()
            next(loader)

        sources = {
            dict(record.extra).get("source")
            for record in _trace_records(trace)
            if "source" in dict(record.extra)
        }
        assert "val" in sources

    def test_initial_batch_phase_is_emitted_once_across_multiple_batches(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=7, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        trace = sft._SFTStartupTrace(start_ns=0)
        with patch_load_sft_features(features):
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=sft.SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(sft.ThresholdType.segment_topk, 1000),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                min_loss_tokens=1,
                alignment_probe_samples=0,
                startup_trace=trace,
            )
            next(generator)
            next(generator)

        initial = [record for record in _trace_records(trace) if record.name == "sft_initial_batch"]
        assert len(initial) == 1
        assert dict(initial[0].extra)["documents"] == 7

    def test_missing_feature_row_is_counted_before_generator_failure(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=5, user_chars=3, asst_chars=10)
        features = build_synthetic_features(
            task,
            tokenizer,
            skip_indices=frozenset({2}),
        )
        trace = sft._SFTStartupTrace(start_ns=0)
        with patch_load_sft_features(features):
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=2,
                T=64,
                device="cpu",
                sft_state=sft.SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(sft.ThresholdType.segment_topk, 1000),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                min_loss_tokens=1,
                alignment_probe_samples=0,
                startup_trace=trace,
            )
            with pytest.raises(ValueError, match=r"1/5.*missing features"):
                next(generator)

        coverage = [
            record for record in _trace_records(trace) if record.name == "sft_feature_row_coverage"
        ]
        assert len(coverage) == 1
        assert coverage[0].status == "completed"
        assert dict(coverage[0].extra)["missing_documents"] == 1
        assert not any(record.name == "sft_alignment_probe" for record in _trace_records(trace))

    def test_artifact_identity_failure_is_timed_and_preserves_error(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=2, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        trace = sft._SFTStartupTrace(start_ns=0)
        with patch_load_sft_features(features, split=SFTSplit.VAL):
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=1,
                T=64,
                device="cpu",
                sft_state=sft.SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(sft.ThresholdType.segment_topk, 1000),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                min_loss_tokens=1,
                alignment_probe_samples=0,
                startup_trace=trace,
                expected_split=SFTSplit.TRAIN,
            )
            with pytest.raises(sft.DataLoadError, match="split mismatch"):
                next(generator)

        identity = [
            record for record in _trace_records(trace) if record.name == "sft_artifact_identity"
        ]
        assert len(identity) == 1
        assert identity[0].status == "failed"
        assert dict(identity[0].extra) == {"source": "train"}

    def test_alignment_failure_is_timed_and_never_emits_initial_batch(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=5, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)
        texts: dict[int, str] = {}
        for index in range(len(task)):
            text, skip_reason = render_conversation_to_full_nanochat_text(
                tokenizer, task[index]
            )
            assert text is not None, skip_reason
            texts[index] = text
        texts[0] += "tampered"
        trace = sft._SFTStartupTrace(start_ns=0)
        with patch_load_sft_features(features, texts=texts):
            generator = sft.make_sft_data_generator(
                task=task,
                tokenizer=tokenizer,
                B=1,
                T=64,
                device="cpu",
                sft_state=sft.SFTState(),
                feature_dir="/dummy/features",
                parquet_dir="/dummy/parquet",
                threshold_cfg=sft.ThresholdConfig(sft.ThresholdType.segment_topk, 1000),
                boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                min_loss_tokens=1,
                alignment_probe_samples=1,
                startup_trace=trace,
            )
            with pytest.raises(sft.DataLoadError, match="alignment probe mismatch"):
                next(generator)

        alignment = [
            record for record in _trace_records(trace) if record.name == "sft_alignment_probe"
        ]
        assert len(alignment) == 1
        assert alignment[0].status == "failed"
        assert dict(alignment[0].extra) == {
            "candidates": 5,
            "matched": 0,
            "requested": 1,
            "sampled": 1,
            "source": "train",
        }
        assert not any(record.name == "sft_initial_batch" for record in _trace_records(trace))

    def test_full_generator_stream_matches_without_telemetry(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=9, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        def collect(trace: sft._SFTStartupTrace | None):
            state = sft.SFTState()
            with patch_load_sft_features(features):
                generator = sft.make_sft_data_generator(
                    task=task,
                    tokenizer=tokenizer,
                    B=2,
                    T=64,
                    device="cpu",
                    sft_state=state,
                    feature_dir="/dummy/features",
                    parquet_dir="/dummy/parquet",
                    threshold_cfg=sft.ThresholdConfig(
                        sft.ThresholdType.segment_topk,
                        1000,
                    ),
                    boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                    min_loss_tokens=1,
                    alignment_probe_samples=0,
                    startup_trace=trace,
                )
                batches = [next(generator) for _ in range(3)]
            return batches, state

        t.manual_seed(1480)
        baseline_rng_before = t.get_rng_state().clone()
        baseline, baseline_state = collect(None)
        baseline_rng_after = t.get_rng_state().clone()
        trace = sft._SFTStartupTrace(start_ns=0)
        t.manual_seed(1480)
        observed_rng_before = t.get_rng_state().clone()
        observed, observed_state = collect(trace)
        observed_rng_after = t.get_rng_state().clone()
        assert t.equal(observed_rng_before, baseline_rng_before)
        assert t.equal(observed_rng_after, baseline_rng_after)
        for expected_batch, actual_batch in zip(baseline, observed, strict=True):
            for expected, actual in zip(expected_batch, actual_batch, strict=True):
                if expected is None:
                    assert actual is None
                else:
                    assert actual is not None
                    assert t.equal(expected, actual)
        assert observed_state.consumed == baseline_state.consumed
        assert observed_state.dataloader_state == baseline_state.dataloader_state
        assert len([r for r in _trace_records(trace) if r.name == "sft_initial_batch"]) == 1

    def test_telemetry_parity_reaches_identical_forward_and_gradients(self) -> None:
        tokenizer = SFTMockTokenizer()
        task = SingleTurnTask(n=6, user_chars=3, asst_chars=10)
        features = build_synthetic_features(task, tokenizer)

        def collect_batch(trace: sft._SFTStartupTrace | None):
            with patch_load_sft_features(features):
                generator = sft.make_sft_data_generator(
                    task=task,
                    tokenizer=tokenizer,
                    B=2,
                    T=64,
                    device="cpu",
                    sft_state=sft.SFTState(),
                    feature_dir="/dummy/features",
                    parquet_dir="/dummy/parquet",
                    threshold_cfg=sft.ThresholdConfig(
                        sft.ThresholdType.segment_topk,
                        1000,
                    ),
                    boundaries=t.ones((N_SAE_FEATURES, NUM_BUCKETS - 1)),
                    min_loss_tokens=1,
                    alignment_probe_samples=0,
                    startup_trace=trace,
                )
                return next(generator)

        baseline_batch = collect_batch(None)
        trace = sft._SFTStartupTrace(start_ns=0)
        traced_batch = collect_batch(trace)

        def run(batch):
            t.manual_seed(777)
            model = build_tiny_model(
                n_sae_features=N_SAE_FEATURES,
                num_buckets=NUM_BUCKETS,
                n_layer=2,
                n_embd=64,
                seq_len=64,
                xattn_frequency=1,
            )
            model.train()
            optimizer = t.optim.AdamW(model.parameters(), lr=1e-3)
            inputs, targets, enc_fids, enc_mags, enc_mask, seg_map, _ = batch
            result = model(inputs, targets, enc_fids, enc_mags, enc_mask, seg_map)
            loss = result.loss if hasattr(result, "loss") else result
            loss.backward()
            gradients = {
                name: parameter.grad.detach().clone()
                for name, parameter in model.named_parameters()
                if parameter.grad is not None
            }
            optimizer.step()
            optimizer_state = optimizer.state_dict()
            return loss.detach().clone(), gradients, optimizer_state

        baseline_loss, baseline_gradients, baseline_optimizer_state = run(baseline_batch)
        traced_loss, traced_gradients, traced_optimizer_state = run(traced_batch)
        assert t.equal(traced_loss, baseline_loss)
        assert baseline_gradients.keys() == traced_gradients.keys()
        for name in baseline_gradients:
            assert t.equal(traced_gradients[name], baseline_gradients[name]), name
        assert baseline_optimizer_state.keys() == traced_optimizer_state.keys()
        assert baseline_optimizer_state["param_groups"] == traced_optimizer_state["param_groups"]
        for parameter_id, baseline_state in baseline_optimizer_state["state"].items():
            traced_state = traced_optimizer_state["state"][parameter_id]
            assert baseline_state.keys() == traced_state.keys()
            for key in baseline_state:
                assert t.equal(traced_state[key], baseline_state[key]), (parameter_id, key)


@pytest.mark.very_slow
def test_issue_1480_real_size_gpu_gate() -> None:
    """Run an operator-supplied production-shaped command without downgrading it.

    Set ``NANOCHAT_ISSUE_1480_REAL_COMMAND`` to a fully resolved, bounded
    ``torchrun``/SFT command when a real GPU allocation and artifact family are
    available. The command must select a short run (for example one optimizer
    step) against the actual checkpoint and corpus; this test never invents
    paths, downloads artifacts, or substitutes a toy model. Captured output is
    checked for the telemetry contract and non-zero corpus scale.
    """
    command_text = os.environ.get("NANOCHAT_ISSUE_1480_REAL_COMMAND")
    if not command_text:
        pytest.skip(
            "set NANOCHAT_ISSUE_1480_REAL_COMMAND to an operator-resolved, "
            "bounded full-size GPU SFT command"
        )
    command = shlex.split(command_text)
    assert command, "NANOCHAT_ISSUE_1480_REAL_COMMAND must not be empty"
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=int(os.environ.get("NANOCHAT_ISSUE_1480_REAL_TIMEOUT_S", "3600")),
    )
    output = f"{result.stdout}\n{result.stderr}"
    assert result.returncode == 0, output
    assert "SFT startup phase=first_optimizer_step status=completed" in output
    assert "SFT startup summary" in output
    assert "summary=steady_state" in output
    feature_load = re.search(
        r"SFT startup phase=sft_feature_store_load .*"
        r"documents=([1-9][0-9]*) .*feature_shards=([1-9][0-9]*)",
        output,
    )
    assert feature_load is not None, output
    assert int(feature_load.group(1)) > 1_000
    assert int(feature_load.group(2)) > 1
