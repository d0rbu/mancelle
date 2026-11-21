"""Opt-in real Gemma 3 4B / Gemma Scope 2 / base-corpus batch gates.

These tests intentionally do not auto-run merely because CUDA exists. They
require explicit environment authorization and refuse a non-quiet GPU; the
serial reference producer is a long-running shared-machine job and must never
be contended by a routine pytest invocation.
"""

from __future__ import annotations

import hashlib
import json
import os
import sqlite3
import subprocess
import sys
import threading
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

import pynvml
import pytest
import torch as t

import steering.eval.autointerp_filter.token_examples as token_examples_module
from scripts.benchmark.benchmark_autointerp_token_examples import (
    compose_benchmark_config,
    run_benchmark,
)
from steering.eval.autointerp_filter.batch_validation import (
    PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
    PRODUCTION_BATCH_128_PROFILE,
    BatchToleranceProfile,
    NumericTolerance,
)
from steering.eval.autointerp_filter.parity import sqlite_ro_uri
from steering.eval.autointerp_filter.targets import resolve_targets
from steering.eval.autointerp_filter.token_examples import infer_n_sae_features
from tests.steering.autointerp_filter._batch_acceptance import (
    ValidationTargetPlan,
    evaluate_batch_performance_replicates,
    feature_activation_counts,
    select_validation_target_plan,
)
from tests.steering.autointerp_filter._batch_calibration import (
    derive_batch_tolerance_profile,
)

_ROOT = Path(
    os.environ.get(
        "ISSUE696_ASSET_ROOT",
        "/home/d0rb/Documents/Github/nano-gpt-pretrain-steer",
    )
)
_FEATURE_DIR = _ROOT / (
    "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos"
)
_DATA_DIR = _ROOT / "nanochat_cache/base_data"
_INDEX_DIR = _ROOT / (
    "out/autointerp_filter/gemma3_4bpt_l22_f65k_ci_nobos_freqmask_root_20260710_1002"
)
_BLACKLIST = _FEATURE_DIR / "blacklist/freq_min0.0001_max0.5.txt"
_SERIAL_RUN_DIR = _ROOT / (
    "out/autointerp_filter/gemma3_4bpt_l22_f65k_ci_nobos_freqmask_n40x40_root_20260710_145543"
)
_REFERENCE_CACHE = Path(
    os.environ.get(
        "ISSUE696_SERIAL_REFERENCE_CACHE",
        str(_SERIAL_RUN_DIR / "token_example_cache.sqlite"),
    )
)
_VALIDATION_PLAN_ENV = "ISSUE696_VALIDATION_PLAN"
_ACCEPTED_VALIDATION_PLAN_SHA256 = (
    "23dd7c7f59d1f6654e07efc7975d25a48adf215967a396e2aad013f1ea9c9853"
)


def _write_json_artifact(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        f"{json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False)}\n",
        encoding="utf-8",
    )


def _nvidia_compute_pids() -> frozenset[int]:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid",
            "--format=csv,noheader,nounits",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "continuous GPU exclusivity monitor could not query nvidia-smi: "
            f"returncode={result.returncode} stderr={result.stderr!r}"
        )
    pids: set[int] = set()
    for line_number, line in enumerate(result.stdout.splitlines(), start=1):
        value = line.strip()
        if not value:
            continue
        if not value.isdigit() or int(value) <= 0:
            raise AssertionError(
                "continuous GPU exclusivity monitor received an invalid PID "
                f"at line {line_number}: {value!r}"
            )
        pid = int(value)
        if pid in pids:
            raise AssertionError(f"continuous GPU exclusivity monitor received duplicate PID {pid}")
        pids.add(pid)
    return frozenset(pids)


def _nvml_compute_pids(handles: tuple[object, ...]) -> frozenset[int]:
    pids: set[int] = set()
    for handle in handles:
        for process in pynvml.nvmlDeviceGetComputeRunningProcesses(handle):
            pid = int(process.pid)
            if pid <= 0:
                raise AssertionError(f"NVML returned an invalid compute PID: {pid}")
            pids.add(pid)
    return frozenset(pids)


class _ContinuousGpuExclusivity:
    """Sample the full CUDA process table throughout one real-GPU gate."""

    def __init__(
        self,
        allowed_pids: Callable[[], set[int]],
        *,
        interval_seconds: float = 0.25,
        pid_reader: Callable[[], frozenset[int]] | None = None,
    ) -> None:
        if not callable(allowed_pids):
            raise TypeError("allowed_pids must be callable")
        if not isinstance(interval_seconds, float) or interval_seconds <= 0.0:
            raise ValueError("interval_seconds must be a positive float")
        self._allowed_pids = allowed_pids
        self._interval_seconds = interval_seconds
        self._pid_reader = pid_reader
        self._owns_nvml = pid_reader is None
        self._nvml_started = False
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.sample_count = 0
        self.unexpected_observations: list[tuple[int, ...]] = []
        self.query_errors: list[str] = []

    def _sample_once(self) -> None:
        try:
            if self._pid_reader is None:
                raise AssertionError("continuous GPU exclusivity monitor has no PID reader")
            observed = self._pid_reader()
            allowed = self._allowed_pids()
            if not isinstance(allowed, set) or any(
                isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0 for pid in allowed
            ):
                raise AssertionError(
                    f"continuous GPU exclusivity monitor has invalid allowed PIDs: {allowed!r}"
                )
        except BaseException as exc:
            self.query_errors.append(f"{type(exc).__name__}: {exc}")
        else:
            unexpected = tuple(sorted(observed - allowed))
            if unexpected:
                self.unexpected_observations.append(unexpected)
        finally:
            self.sample_count += 1

    def _run(self) -> None:
        while not self._stop.wait(self._interval_seconds):
            self._sample_once()

    def __enter__(self) -> _ContinuousGpuExclusivity:
        if self._thread is not None:
            raise RuntimeError("continuous GPU exclusivity monitor cannot be reused")
        if self._owns_nvml:
            try:
                pynvml.nvmlInit()
                self._nvml_started = True
                handles = tuple(
                    pynvml.nvmlDeviceGetHandleByIndex(index)
                    for index in range(pynvml.nvmlDeviceGetCount())
                )
                if not handles:
                    raise AssertionError("NVML reported zero GPU devices")
                self._pid_reader = lambda: _nvml_compute_pids(handles)
            except BaseException as exc:
                self.query_errors.append(f"{type(exc).__name__}: {exc}")
        self._sample_once()
        self._thread = threading.Thread(
            target=self._run,
            name="issue696-gpu-exclusivity",
            daemon=True,
        )
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self._stop.set()
        assert self._thread is not None
        self._thread.join(timeout=max(5.0, 4.0 * self._interval_seconds))
        if self._thread.is_alive():
            self.query_errors.append("monitor thread did not stop")
        self._sample_once()
        if self._nvml_started:
            try:
                pynvml.nvmlShutdown()
            except BaseException as exc:
                self.query_errors.append(f"{type(exc).__name__}: {exc}")
            self._nvml_started = False
        if exc_type is None:
            self.assert_exclusive()

    def assert_exclusive(self) -> None:
        if self.sample_count < 2:
            raise AssertionError("continuous GPU exclusivity monitor took fewer than two samples")
        if self.query_errors or self.unexpected_observations:
            raise AssertionError(
                "continuous GPU exclusivity monitor failed: "
                f"samples={self.sample_count} query_errors={self.query_errors} "
                f"unexpected_compute_pids={self.unexpected_observations}"
            )


def _assert_exclusive_clean_real_environment(report: dict) -> None:
    expected_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[3],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()
    assert report["git"] == {"sha": expected_sha, "dirty": False}
    environment = report["environment"]
    assert environment["gpu_name"] == "NVIDIA GeForce RTX 4090"
    assert environment["nvidia_compute_process_count"] == 1
    preconditions = report["preconditions"]
    assert preconditions["source_index_required"] is True
    assert preconditions["index_stat_unchanged"] is True
    assert preconditions["manifest_stat_unchanged"] is True
    assert report["cuda_max_memory_allocated_bytes"] > 0
    assert report["cuda_max_memory_reserved_bytes"] >= report["cuda_max_memory_allocated_bytes"]


def _assert_exact_materialization_plan(
    report: dict,
    *,
    expected: tuple[int, ...],
    forbidden: tuple[int, ...],
) -> None:
    actual = tuple(row["feature_ids"][0] for row in report["materialization_targets"])
    assert actual == expected
    assert not set(actual) & set(forbidden)
    assert report["benchmark"]["materialization_target_count"] == len(expected)


def _require_real_gpu_authorization(env_name: str) -> None:
    if os.environ.get(env_name) != "1":
        pytest.skip(f"set {env_name}=1 to authorize the real issue #696 GPU gate")
    if not t.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    required = (
        _FEATURE_DIR,
        _DATA_DIR,
        _INDEX_DIR / "source_doc_index.sqlite",
        _INDEX_DIR / "source_doc_index.manifest.json",
        _BLACKLIST,
        _REFERENCE_CACHE,
    )
    for path in required:
        if not path.exists():
            pytest.skip(f"issue #696 production artifact is missing: {path}")
    try:
        active = sorted(_nvidia_compute_pids())
    except AssertionError:
        pytest.skip("cannot prove the GPU is quiet with nvidia-smi")
    if active:
        pytest.skip(f"GPU has active compute processes; refusing contention: {active}")


def _config(output_dir: Path, *, batch_size: int):
    return compose_benchmark_config(
        "precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos",
        [
            f"output_dir={output_dir}",
            "neuronpedia.enabled=false",
            f"feature_dir={_FEATURE_DIR}",
            f"data_dir={_DATA_DIR}",
            "source_index.enabled=true",
            f"source_index.path={_INDEX_DIR / 'source_doc_index.sqlite'}",
            (f"source_index.manifest_path={_INDEX_DIR / 'source_doc_index.manifest.json'}"),
            f"feature_mask={{mode:blacklist,path:{_BLACKLIST}}}",
            "token_examples.max_positive_windows_per_target=40",
            "token_examples.max_non_activating_windows_per_target=40",
            f"token_examples.materialization_batch_size={batch_size}",
        ],
    )


def _run_performance_variant_subprocess(
    *,
    output_dir: Path,
    workload: list[int],
    batch_size: int,
    prefetch: bool,
) -> dict:
    report_path = output_dir.with_suffix(".json")
    command = [
        sys.executable,
        "-m",
        "scripts.benchmark.benchmark_autointerp_token_examples",
        "--config-name",
        "precompute_autointerp_filter_gemma3_4bpt_l22_f65k_ci_nobos",
        "--output-dir",
        str(output_dir),
        "--report",
        str(report_path),
        "--feature-dir",
        str(_FEATURE_DIR),
        "--data-dir",
        str(_DATA_DIR),
        "--source-index-path",
        str(_INDEX_DIR / "source_doc_index.sqlite"),
        "--source-index-manifest-path",
        str(_INDEX_DIR / "source_doc_index.manifest.json"),
        "--device",
        "cuda",
        "--activation-backend",
        "native",
        "--model-dtype",
        "bfloat16",
        "--targets",
        ",".join(str(feature_id) for feature_id in workload),
        "--instrumentation",
        "off",
        "--materialization-prefetch",
        "on" if prefetch else "off",
        "--extra-override",
        f"feature_mask={{mode:blacklist,path:{_BLACKLIST}}}",
        "--extra-override",
        "token_examples.max_positive_windows_per_target=40",
        "--extra-override",
        "token_examples.max_non_activating_windows_per_target=40",
        "--extra-override",
        f"token_examples.materialization_batch_size={batch_size}",
    ]
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[3],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        with _ContinuousGpuExclusivity(lambda: {process.pid}):
            stdout, stderr = process.communicate()
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()
    assert process.returncode == 0, (
        f"benchmark subprocess failed with {process.returncode}\n"
        f"stdout tail:\n{stdout[-4000:]}\n"
        f"stderr tail:\n{stderr[-4000:]}"
    )
    assert report_path.is_file()
    return json.loads(report_path.read_text())


def _reference_feature_ids() -> set[int]:
    with sqlite3.connect(sqlite_ro_uri(_REFERENCE_CACHE), uri=True) as conn:
        rows = conn.execute("SELECT DISTINCT target_key FROM token_example_sets").fetchall()
    result: set[int] = set()
    for (target_key,) in rows:
        prefix, separator, raw_id = str(target_key).partition(":")
        if prefix == "feature" and separator and raw_id.isdigit():
            result.add(int(raw_id))
    return result


def _resolved_allowed_ids(cfg) -> set[int]:
    feature_dir = Path(cfg.feature_dir)
    n_features = infer_n_sae_features(feature_dir, cfg.n_sae_features)
    return {
        target.feature_ids[0]
        for target in resolve_targets(
            cfg,
            n_sae_features=n_features,
            feature_dir=feature_dir,
        )
    }


def _single_reference_block(cfg, *, anchor_count: int = 8) -> tuple[list[int], list[int]]:
    allowed = _resolved_allowed_ids(cfg)
    reference = _reference_feature_ids() & allowed
    by_block: dict[int, list[int]] = defaultdict(list)
    for feature_id in sorted(reference):
        by_block[feature_id // 128].append(feature_id)
    eligible = [
        (block_id, ids) for block_id, ids in sorted(by_block.items()) if len(ids) >= anchor_count
    ]
    if not eligible:
        pytest.skip(
            f"serial reference cache has no batch block with {anchor_count} completed anchors"
        )
    block_id, reference_ids = eligible[0]
    anchors = reference_ids[:anchor_count]
    materialization = sorted(fid for fid in allowed if fid // 128 == block_id)
    assert set(anchors) <= set(materialization)
    return anchors, materialization


def _validation_plan(cfg, *, require_frozen: bool) -> ValidationTargetPlan:
    frozen_path_raw = os.environ.get(_VALIDATION_PLAN_ENV)
    if require_frozen and frozen_path_raw is None:
        raise AssertionError(
            f"set {_VALIDATION_PLAN_ENV} to the accepted out-of-tree issue #696 validation plan"
        )
    allowed = _resolved_allowed_ids(cfg)
    reference_ids = _reference_feature_ids() & allowed
    if frozen_path_raw is not None:
        frozen_path = Path(frozen_path_raw).expanduser()
        if not frozen_path.is_file():
            raise AssertionError(f"{_VALIDATION_PLAN_ENV} does not name a file: {frozen_path}")
        frozen_bytes = frozen_path.read_bytes()
        frozen_sha256 = hashlib.sha256(frozen_bytes).hexdigest()
        assert frozen_sha256 == _ACCEPTED_VALIDATION_PLAN_SHA256, (
            f"{_VALIDATION_PLAN_ENV} SHA-256 mismatch: expected "
            f"{_ACCEPTED_VALIDATION_PLAN_SHA256}, got {frozen_sha256}"
        )
        plan = ValidationTargetPlan.from_json(json.loads(frozen_bytes))
        assert plan.batch_size == 128
        assert set(plan.anchor_target_ids) <= reference_ids
        assert set(plan.materialization_target_ids) <= allowed
        return plan
    return select_validation_target_plan(
        feature_activation_counts(Path(cfg.feature_dir)),
        allowed=allowed,
        anchor_allowed=reference_ids,
    )


def _assert_strict_real_report(
    report: dict,
    *,
    anchor_count: int,
    profile: BatchToleranceProfile = PRODUCTION_BATCH_128_PROFILE,
) -> None:
    _assert_exclusive_clean_real_environment(report)
    assert report["benchmark"]["target_count"] == anchor_count
    assert report["benchmark"]["materialization_target_count"] >= anchor_count
    assert report["benchmark"]["unvalidated_candidate"] is False
    assert report["batch_validation"]["profile"] == profile.identity()
    observations = report["batch_validation"]["observations"]
    assert observations
    assert all(
        observation["support_mismatches"] == 0
        for observation in observations
        if observation["require_support_bound"]
    )
    assert report["parity"] is not None
    for parity in report["parity"]["per_target"].values():
        positives = parity["examples"]["positives"]
        non_activating = parity["examples"]["non_activating"]
        assert positives["identity_set_equal"] is True
        assert non_activating["identity_set_equal"] is True
        assert positives["same_identity_order_count"] == positives["reference_count"]
        assert non_activating["same_identity_order_count"] == (non_activating["reference_count"])
        assert positives["highlight_exact_count"] == positives["matched_window_count"]
        assert positives["quantile_exact_count"] == positives["matched_window_count"]
        assert non_activating["quantile_exact_count"] == (non_activating["matched_window_count"])
        for numeric_surface in (
            positives["activation"],
            positives["max_activation"],
            non_activating["activation"],
            non_activating["max_activation"],
        ):
            assert numeric_surface["support_mismatches"] == 0
            assert numeric_surface["max_abs"] <= profile.selected_sae.max_abs
            assert numeric_surface["p99_abs"] <= profile.selected_sae.p99_abs
            assert numeric_surface["mean_abs"] <= profile.selected_sae.mean_abs
        assert non_activating["activation"]["max_abs"] == 0.0
        assert (
            parity["trials"]["same_prompt_surface_count"] == (parity["trials"]["reference_count"])
        )
        assert parity["rendered_prompts"]["all_rendered_examples_exact"] is True
        assert parity["rendered_prompts"]["all_message_bytes_exact"] is True


def _assert_exact_serial_oracle_report(report: dict, *, anchor_count: int) -> None:
    _assert_exclusive_clean_real_environment(report)
    assert report["benchmark"]["target_count"] == anchor_count
    assert report["config"]["token_examples"]["materialization_batch_size"] == 1
    assert report["batch_validation"]["profile"] is None
    assert report["parity"] is not None
    for parity in report["parity"]["per_target"].values():
        for group_name in ("positives", "non_activating"):
            group = parity["examples"][group_name]
            assert group["identity_set_equal"] is True
            assert group["candidate_count"] == group["reference_count"]
            assert group["same_identity_order_count"] == group["reference_count"]
            assert group["same_stable_key_order_count"] == group["reference_count"]
            assert group["quantile_exact_count"] == group["reference_count"]
            assert group["highlight_exact_count"] == group["reference_count"]
            for metric_name in ("activation", "max_activation"):
                metric = group[metric_name]
                assert metric["support_mismatches"] == 0
                assert metric["max_abs"] == 0.0
                assert metric["p99_abs"] == 0.0
                assert metric["mean_abs"] == 0.0
        trials = parity["trials"]
        assert trials["candidate_count"] == trials["reference_count"]
        assert trials["same_prompt_surface_count"] == trials["reference_count"]
        assert trials["same_trial_key_count"] == trials["reference_count"]
        assert trials["same_intruder_window_count"] == trials["reference_count"]
        assert parity["rendered_prompts"]["all_rendered_examples_exact"] is True
        assert parity["rendered_prompts"]["all_message_bytes_exact"] is True


@pytest.mark.parametrize(
    ("stdout", "returncode", "expected", "error"),
    [
        ("", 0, frozenset(), None),
        ("12\n34\n", 0, frozenset({12, 34}), None),
        ("not-a-pid\n", 0, None, "invalid PID"),
        ("12\n12\n", 0, None, "duplicate PID"),
        ("", 1, None, "could not query"),
    ],
)
def test_nvidia_compute_pid_query_is_strict(
    monkeypatch: pytest.MonkeyPatch,
    stdout: str,
    returncode: int,
    expected: frozenset[int] | None,
    error: str | None,
) -> None:
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args=args,
            returncode=returncode,
            stdout=stdout,
            stderr="query failed" if returncode else "",
        ),
    )
    if error is None:
        assert _nvidia_compute_pids() == expected
    else:
        with pytest.raises(AssertionError, match=error):
            _nvidia_compute_pids()


def test_continuous_gpu_monitor_accepts_only_declared_pids() -> None:
    monitor = _ContinuousGpuExclusivity(
        lambda: {11, 17},
        interval_seconds=0.01,
        pid_reader=lambda: frozenset({11, 17}),
    )
    with monitor:
        pass
    assert monitor.sample_count >= 2
    assert monitor.query_errors == []
    assert monitor.unexpected_observations == []


def test_continuous_gpu_monitor_rejects_any_transient_external_pid() -> None:
    observations = iter(
        (
            frozenset({11}),
            frozenset({11, 99}),
        )
    )
    with pytest.raises(AssertionError, match=r"unexpected_compute_pids=\[\(99,\)\]"):
        with _ContinuousGpuExclusivity(
            lambda: {11},
            interval_seconds=60.0,
            pid_reader=lambda: next(observations),
        ):
            pass


def test_continuous_gpu_monitor_records_query_and_allowed_pid_failures() -> None:
    with pytest.raises(AssertionError, match="invalid allowed PIDs"):
        with _ContinuousGpuExclusivity(
            lambda: {True},
            interval_seconds=60.0,
            pid_reader=frozenset,
        ):
            pass


def test_continuous_gpu_monitor_does_not_mask_primary_body_failure() -> None:
    monitor = _ContinuousGpuExclusivity(
        lambda: {11},
        interval_seconds=60.0,
        pid_reader=lambda: frozenset({99}),
    )
    with pytest.raises(RuntimeError, match="primary failure"):
        with monitor:
            raise RuntimeError("primary failure")
    assert monitor.unexpected_observations == [(99,), (99,)]


def test_required_frozen_validation_plan_fails_before_artifact_resolution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv(_VALIDATION_PLAN_ENV, raising=False)
    with pytest.raises(AssertionError, match=rf"set {_VALIDATION_PLAN_ENV}"):
        _validation_plan(object(), require_frozen=True)


def test_frozen_validation_plan_loader_rejects_missing_and_wrong_bytes_then_loads_exact_plan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = select_validation_target_plan(
        [(0, 1), (1, 2)],
        allowed={0, 1},
        batch_size=128,
        n_calibration=1,
        n_heldout=1,
    )
    this_module = sys.modules[__name__]
    monkeypatch.setattr(this_module, "_resolved_allowed_ids", lambda _cfg: {0, 1})
    monkeypatch.setattr(this_module, "_reference_feature_ids", lambda: {0, 1})

    missing = tmp_path / "missing-plan.json"
    monkeypatch.setenv(_VALIDATION_PLAN_ENV, str(missing))
    with pytest.raises(AssertionError, match="does not name a file"):
        _validation_plan(object(), require_frozen=True)

    frozen = tmp_path / "frozen-plan.json"
    frozen.write_text(json.dumps(plan.to_json(), sort_keys=True), encoding="utf-8")
    monkeypatch.setenv(_VALIDATION_PLAN_ENV, str(frozen))
    with pytest.raises(AssertionError, match="SHA-256 mismatch"):
        _validation_plan(object(), require_frozen=True)

    accepted_sha256 = hashlib.sha256(frozen.read_bytes()).hexdigest()
    monkeypatch.setattr(this_module, "_ACCEPTED_VALIDATION_PLAN_SHA256", accepted_sha256)
    assert _validation_plan(object(), require_frozen=True) == plan


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_real_candidate_serial_path_is_byte_exact_to_untouched_reference_for_100_anchors(
    tmp_path: Path,
) -> None:
    _require_real_gpu_authorization("RUN_ISSUE696_SERIAL_ORACLE")
    cfg = _config(tmp_path / "candidate-serial", batch_size=1)
    plan = _validation_plan(cfg, require_frozen=True)

    with _ContinuousGpuExclusivity(lambda: {os.getpid()}):
        report = run_benchmark(
            cfg,
            plan.anchor_target_ids,
            tier=1,
            reference_cache_path=_REFERENCE_CACHE,
            trial_n_positive=3,
            trial_count=40,
            trial_quantiles=5,
            instrument=False,
        )

    _write_json_artifact(tmp_path / "serial-oracle-report.json", report)
    _assert_exact_serial_oracle_report(report, anchor_count=100)


@pytest.mark.slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_real_batch128_single_full_block_matches_serial_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_real_gpu_authorization("RUN_ISSUE696_REAL_GPU")
    cfg = _config(tmp_path / "batch128", batch_size=128)
    anchors, materialization = _single_reference_block(cfg)
    original_get_activations = token_examples_module._get_activations_native
    observation_lock = threading.Lock()
    worker_forwards: list[tuple[str, int, tuple[int, ...]]] = []

    def _record_worker_forward(*args, **kwargs):
        tokens = args[1]
        thread_name = threading.current_thread().name
        if thread_name.startswith("autointerp-token-gpu"):
            stream_id = int(t.cuda.current_stream(device=tokens.device).cuda_stream)
            with observation_lock:
                worker_forwards.append((thread_name, stream_id, tuple(tokens.shape)))
        return original_get_activations(*args, **kwargs)

    monkeypatch.setattr(
        token_examples_module,
        "_get_activations_native",
        _record_worker_forward,
    )

    with _ContinuousGpuExclusivity(lambda: {os.getpid()}):
        report = run_benchmark(
            cfg,
            anchors,
            materialization_target_ids=materialization,
            tier=1,
            reference_cache_path=_REFERENCE_CACHE,
            trial_n_positive=3,
            trial_count=40,
            trial_quantiles=5,
            instrument=False,
            materialization_prefetch=True,
        )

    _write_json_artifact(tmp_path / "single-full-block-report.json", report)
    _assert_strict_real_report(report, anchor_count=len(anchors))
    assert worker_forwards
    assert all(shape[0] == 1 and 1 <= shape[1] <= 129 for _name, _stream, shape in worker_forwards)
    streams_by_thread: dict[str, set[int]] = defaultdict(set)
    for thread_name, stream_id, _shape in worker_forwards:
        streams_by_thread[thread_name].add(stream_id)
    assert set(streams_by_thread) == {
        "autointerp-token-gpu_0",
        "autointerp-token-gpu_1",
    }
    assert all(len(stream_ids) == 1 for stream_ids in streams_by_thread.values())
    assert len(set().union(*streams_by_thread.values())) == 2


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_real_100_anchor_calibration_is_preselected_and_emits_frozen_profile_input(
    tmp_path: Path,
) -> None:
    _require_real_gpu_authorization("RUN_ISSUE696_FULL_CALIBRATION")
    cfg = _config(tmp_path / "calibration", batch_size=128)
    plan = _validation_plan(cfg, require_frozen=False)
    assert len(plan.calibration_target_ids) == 60
    assert len(plan.heldout_target_ids) == 40
    permissive = BatchToleranceProfile(
        name="issue696_unvalidated_calibration_only",
        batch_size=128,
        selected_sae=NumericTolerance(
            max_abs=1_000_000.0,
            p99_abs=1_000_000.0,
            mean_abs=1_000_000.0,
        ),
        raw_residual=NumericTolerance(
            max_abs=1_000_000.0,
            p99_abs=1_000_000.0,
            mean_abs=1_000_000.0,
        ),
        calibration_contract=PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
        max_support_mismatches=0,
    )

    with _ContinuousGpuExclusivity(lambda: {os.getpid()}):
        report = run_benchmark(
            cfg,
            plan.calibration_target_ids,
            materialization_target_ids=plan.calibration_materialization_target_ids,
            tier=1,
            reference_cache_path=_REFERENCE_CACHE,
            trial_n_positive=3,
            trial_count=40,
            trial_quantiles=5,
            instrument=False,
            materialization_prefetch=True,
            calibration_mode=True,
            calibration_profile=permissive,
        )
    _assert_exclusive_clean_real_environment(report)
    _assert_exact_materialization_plan(
        report,
        expected=plan.calibration_materialization_target_ids,
        forbidden=plan.heldout_target_ids,
    )
    assert report["benchmark"]["unvalidated_candidate"] is True
    assert report["benchmark"]["full_anchor_validation"] is True
    profile_observations = report["batch_validation"]["profile_observations"]
    assert profile_observations
    profile = derive_batch_tolerance_profile(
        profile_observations,
        name="issue696_gemma3_4bpt_block128_stream2_calibrated_v1",
        batch_size=128,
        calibration_contract=PRODUCTION_BATCH_128_CALIBRATION_CONTRACT,
        device=cfg.token_examples.device,
    )
    _write_json_artifact(tmp_path / "calibration-report.json", report)
    _write_json_artifact(tmp_path / "derived-profile.json", profile.identity())
    _write_json_artifact(tmp_path / "frozen-validation-plan.json", plan.to_json())
    assert profile.max_support_mismatches == 0
    assert profile.calibration_observations_sha256 is not None


@pytest.mark.very_slow
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_real_frozen_profile_passes_all_40_preselected_heldout_anchors(
    tmp_path: Path,
) -> None:
    _require_real_gpu_authorization("RUN_ISSUE696_HELDOUT")
    cfg = _config(tmp_path / "heldout", batch_size=128)
    plan = _validation_plan(cfg, require_frozen=True)

    with _ContinuousGpuExclusivity(lambda: {os.getpid()}):
        report = run_benchmark(
            cfg,
            plan.heldout_target_ids,
            materialization_target_ids=plan.heldout_materialization_target_ids,
            tier=1,
            reference_cache_path=_REFERENCE_CACHE,
            trial_n_positive=3,
            trial_count=40,
            trial_quantiles=5,
            instrument=False,
            materialization_prefetch=True,
            validate_all_anchor_surfaces=True,
        )

    _write_json_artifact(tmp_path / "heldout-report.json", report)
    _assert_exact_materialization_plan(
        report,
        expected=plan.heldout_materialization_target_ids,
        forbidden=plan.calibration_target_ids,
    )
    assert report["benchmark"]["full_anchor_validation"] is True
    assert report["benchmark"]["unvalidated_candidate"] is False
    assert report["batch_validation"]["profile_observations"]
    _assert_strict_real_report(report, anchor_count=40)


@pytest.mark.very_slow
@pytest.mark.benchmark
@pytest.mark.requires_feature_data
@pytest.mark.skipif(not t.cuda.is_available(), reason="requires CUDA")
def test_real_dense_1000_target_batch_and_prefetch_performance_acceptance(
    tmp_path: Path,
) -> None:
    _require_real_gpu_authorization("RUN_ISSUE696_PERFORMANCE")
    probe_cfg = _config(tmp_path / "probe", batch_size=128)
    allowed = _resolved_allowed_ids(probe_cfg)
    by_block: dict[int, list[int]] = defaultdict(list)
    for feature_id in sorted(allowed):
        by_block[feature_id // 128].append(feature_id)
    workload: list[int] = []
    for block_id in sorted(by_block):
        workload.extend(by_block[block_id])
        if len(workload) >= 1000:
            break
    if len(workload) < 1000:
        pytest.skip("resolved production target set cannot supply 1,000 dense targets")
    workload = workload[:1000]

    serial_seconds: list[float] = []
    batch_seconds: list[float] = []
    prefetch_seconds: list[float] = []
    counterbalanced_orders = (
        ("serial", "batch", "prefetch"),
        ("batch", "prefetch", "serial"),
        ("prefetch", "serial", "batch"),
    )
    for repeat, order in enumerate(counterbalanced_orders):
        reports: dict[str, dict] = {}
        for variant in order:
            _require_real_gpu_authorization("RUN_ISSUE696_PERFORMANCE")
            reports[variant] = _run_performance_variant_subprocess(
                output_dir=tmp_path / f"{variant}-{repeat}",
                workload=workload,
                batch_size=1 if variant == "serial" else 128,
                prefetch=variant == "prefetch",
            )
        serial = reports["serial"]
        batch = reports["batch"]
        prefetch = reports["prefetch"]
        for report in reports.values():
            _assert_exclusive_clean_real_environment(report)
        batch_payloads = {
            row["target_key"]: (row["payload_sha256"], row["trial_keys_sha256"])
            for row in batch["per_target"]
        }
        prefetch_payloads = {
            row["target_key"]: (row["payload_sha256"], row["trial_keys_sha256"])
            for row in prefetch["per_target"]
        }
        assert batch_payloads == prefetch_payloads
        steady = [report["benchmark"]["steady_state"] for report in (serial, batch, prefetch)]
        assert all(measurement["available"] is True for measurement in steady)
        assert len({measurement["target_count"] for measurement in steady}) == 1
        assert (
            steady[0]["measured_target_keys_sha256"]
            == (steady[1]["measured_target_keys_sha256"])
            == steady[2]["measured_target_keys_sha256"]
        )
        serial_seconds.append(float(steady[0]["elapsed_s"]))
        batch_seconds.append(float(steady[1]["elapsed_s"]))
        prefetch_seconds.append(float(steady[2]["elapsed_s"]))

    acceptance = evaluate_batch_performance_replicates(
        serial_seconds=serial_seconds,
        batched_seconds=batch_seconds,
        prefetched_seconds=prefetch_seconds,
    )
    _write_json_artifact(tmp_path / "performance-acceptance.json", acceptance)
    assert acceptance["batch_pass"] is True
    assert acceptance["prefetch_pass"] is True
    assert acceptance["retain_prefetch"] is True
