from __future__ import annotations

import hashlib
import importlib.metadata
import json
import math
from dataclasses import replace
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import steering.eval.autointerp_filter.batch_validation as batch_validation_module
from steering.eval.autointerp_filter.batch_validation import (
    BATCH_VALIDATION_PROFILE_VERSION,
    PRODUCTION_BATCH_128_PROFILE,
    BatchCalibrationContract,
    BatchDifferenceMetrics,
    BatchToleranceProfile,
    NumericTolerance,
    assert_batch_difference_within_profile,
    assert_batch_metrics_within_profile,
    assert_candidate_repeat_exact,
    assert_runtime_matches_profile,
    measure_batch_difference,
    runtime_numeric_fingerprint,
    runtime_numeric_fingerprint_sha256,
)
from steering.eval.autointerp_filter.types import AutoInterpBatchValidationError
from tests.steering.autointerp_filter._batch_calibration import (
    derive_batch_tolerance_profile,
)


def _contract(**updates: Any) -> BatchCalibrationContract:
    values: dict[str, Any] = {
        "model_name": "gemma-3-test",
        "model_dtype": "bfloat16",
        "activation_type": "residual",
        "layer": 7,
        "segment_size": 128,
        "sae_release": "test-release",
        "sae_id": "test-sae-id",
        "hook_name": "blocks.7.hook_resid_post",
    }
    values.update(updates)
    return BatchCalibrationContract(**values)


def _profile(**updates: Any) -> BatchToleranceProfile:
    max_abs = updates.pop("max_abs", 0.2)
    p99_abs = updates.pop("p99_abs", 0.1)
    mean_abs = updates.pop("mean_abs", 0.05)
    tolerance = NumericTolerance(
        max_abs=max_abs,
        p99_abs=p99_abs,
        mean_abs=mean_abs,
    )
    values: dict[str, Any] = {
        "name": "unit-test-profile",
        "batch_size": 128,
        "selected_sae": tolerance,
        "raw_residual": tolerance,
        "calibration_contract": _contract(),
        "max_support_mismatches": 0,
        "runtime_fingerprint_sha256": None,
        "calibration_observations_sha256": None,
    }
    values.update(updates)
    return BatchToleranceProfile(**values)


def test_tolerance_profile_identity_is_complete_and_json_stable() -> None:
    profile = _profile()
    identity = profile.identity()
    assert identity == {
        "version": BATCH_VALIDATION_PROFILE_VERSION,
        "name": "unit-test-profile",
        "batch_size": 128,
        "selected_sae": {"max_abs": 0.2, "p99_abs": 0.1, "mean_abs": 0.05},
        "raw_residual": {"max_abs": 0.2, "p99_abs": 0.1, "mean_abs": 0.05},
        "calibration_contract": _contract().identity(),
        "max_support_mismatches": 0,
        "runtime_fingerprint_sha256": None,
        "calibration_observations_sha256": None,
    }
    assert json.loads(json.dumps(identity, sort_keys=True)) == identity


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"name": ""}, "name is empty"),
        ({"batch_size": 1}, "batch_size.*>= 2"),
        ({"batch_size": 0}, "batch_size.*>= 2"),
        ({"batch_size": True}, "batch_size.*strict int"),
        ({"batch_size": 2.0}, "batch_size.*strict int"),
        ({"max_abs": -1.0}, "max_abs must be finite"),
        ({"p99_abs": -1.0}, "p99_abs must be finite"),
        ({"mean_abs": -1.0}, "mean_abs must be finite"),
        ({"max_abs": math.inf}, "max_abs must be finite"),
        ({"max_abs": True}, "max_abs must be finite"),
        ({"p99_abs": math.nan}, "p99_abs must be finite"),
        ({"max_abs": 0.05, "p99_abs": 0.1}, "must not exceed max_abs"),
        ({"max_abs": 0.04, "mean_abs": 0.05}, "must not exceed max_abs"),
        ({"max_support_mismatches": -1}, "max_support_mismatches"),
        ({"max_support_mismatches": 1}, "max_support_mismatches"),
        ({"max_support_mismatches": True}, "max_support_mismatches"),
    ],
)
def test_tolerance_profile_rejects_invalid_acceptance_envelopes(
    updates: dict[str, float | int | str],
    message: str,
) -> None:
    relationship_error = len(updates) > 1
    error_type = (
        AutoInterpBatchValidationError if relationship_error else BeartypeCallHintParamViolation
    )
    match = message if relationship_error else next(iter(updates))
    with pytest.raises(error_type, match=match):
        _profile(**updates)


@pytest.mark.parametrize("field_name", ["selected_sae", "raw_residual"])
def test_tolerance_profile_rejects_unvalidated_nested_tolerance_objects(
    field_name: str,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=field_name):
        _profile(**{field_name: {"max_abs": 1.0}})


def test_tolerance_profile_rejects_unknown_surface_kind() -> None:
    with pytest.raises(AutoInterpBatchValidationError, match="unknown.*surface kind"):
        _profile().tolerance_for(cast(Any, "other"))


@pytest.mark.parametrize(
    ("updates", "message"),
    [
        ({"model_name": ""}, "model_name.*non-empty"),
        ({"model_dtype": None}, "model_dtype.*non-empty"),
        ({"activation_type": ""}, "activation_type.*non-empty"),
        ({"sae_release": ""}, "sae_release.*non-empty"),
        ({"sae_id": 1}, "sae_id.*non-empty"),
        ({"hook_name": ""}, "hook_name.*non-empty"),
        ({"layer": -1}, "layer.*strict int"),
        ({"layer": True}, "layer.*strict int"),
        ({"segment_size": 0}, "segment_size.*strict int"),
        ({"segment_size": 128.0}, "segment_size.*strict int"),
    ],
)
def test_calibration_contract_rejects_incomplete_or_non_strict_fields(
    updates: dict[str, Any],
    message: str,
) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match=next(iter(updates))):
        _contract(**updates)


def test_tolerance_profile_rejects_unvalidated_calibration_contract_object() -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="calibration_contract"):
        _profile(calibration_contract={"model_name": "gemma"})


def test_runtime_numeric_fingerprint_is_complete_stable_and_hash_bound() -> None:
    first = runtime_numeric_fingerprint()
    second = runtime_numeric_fingerprint()
    assert first == second
    assert {
        "torch",
        "transformers",
        "sae_lens",
        "cuda_runtime",
        "cuda_available",
        "float32_matmul_precision",
        "deterministic_algorithms",
        "cublas_workspace_config",
        "nvidia_tf32_override",
        "torch_allow_tf32_cublas_override",
    } <= set(first)
    assert first["sae_lens"] == importlib.metadata.version("sae-lens")
    if first["cuda_available"]:
        assert {
            "gpu_name",
            "compute_capability",
            "flash_sdp_enabled",
            "mem_efficient_sdp_enabled",
            "math_sdp_enabled",
            "matmul_allow_tf32",
            "cudnn_allow_tf32",
            "cudnn_version",
            "nvidia_driver_version",
            "matmul_allow_bf16_reduced_precision_reduction",
            "preferred_blas_library",
        } <= set(first)
    encoded = json.dumps(first, sort_keys=True, separators=(",", ":"))
    assert (
        runtime_numeric_fingerprint_sha256() == hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    )


def test_checked_in_production_profile_exactly_matches_frozen_real_calibration() -> None:
    smallest_positive_float = math.nextafter(0.0, math.inf)
    assert smallest_positive_float == 5e-324
    assert PRODUCTION_BATCH_128_PROFILE.identity() == {
        "version": BATCH_VALIDATION_PROFILE_VERSION,
        "name": "issue696_gemma3_4bpt_block128_stream2_calibrated_v1",
        "batch_size": 128,
        "selected_sae": {
            "max_abs": smallest_positive_float,
            "p99_abs": smallest_positive_float,
            "mean_abs": smallest_positive_float,
        },
        "raw_residual": {
            "max_abs": smallest_positive_float,
            "p99_abs": smallest_positive_float,
            "mean_abs": smallest_positive_float,
        },
        "calibration_contract": {
            "model_name": "gemma-3-4b-pt",
            "model_dtype": "bfloat16",
            "activation_type": "residual",
            "layer": 22,
            "segment_size": 128,
            "sae_release": "gemma-scope-2-4b-pt-res",
            "sae_id": "layer_22_width_65k_l0_medium",
            "hook_name": "blocks.22.hook_resid_post",
        },
        "max_support_mismatches": 0,
        "runtime_fingerprint_sha256": (
            "6443efc27a7953dfa88fb8dd1bd6043bffcb2508667e893da3f2362c5c76a7d9"
        ),
        "calibration_observations_sha256": (
            "9ac3b4e4dbac99b586b9d0ddbe5dd6b44ab7f6c8c2bc183c759ab5ffe3b1e0d9"
        ),
    }


def test_runtime_numeric_fingerprint_without_cuda_has_no_gpu_claims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    payload = runtime_numeric_fingerprint()
    assert payload["cuda_available"] is False
    assert "gpu_name" not in payload
    assert "nvidia_driver_version" not in payload


def test_runtime_numeric_fingerprint_rejects_non_cuda_or_unresolvable_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    with pytest.raises(AutoInterpBatchValidationError, match="non-CUDA device"):
        runtime_numeric_fingerprint("cpu")
    monkeypatch.setattr(
        t.cuda,
        "get_device_properties",
        lambda _index: (_ for _ in ()).throw(RuntimeError("missing device")),
    )
    with pytest.raises(AutoInterpBatchValidationError, match="cannot fingerprint.*7"):
        runtime_numeric_fingerprint("cuda:7")


def test_runtime_numeric_fingerprint_records_resolved_cuda_hardware(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    properties = SimpleNamespace(
        name="Synthetic RTX",
        major=9,
        minor=1,
        total_memory=24_000_000_000,
        multi_processor_count=128,
    )
    monkeypatch.setattr(t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(t.cuda, "current_device", lambda: 3)
    monkeypatch.setattr(t.cuda, "get_device_properties", lambda index: properties)
    monkeypatch.setattr(t.backends.cudnn, "version", lambda: 90100)
    monkeypatch.setattr(
        batch_validation_module,
        "_nvidia_driver_version",
        lambda index: f"driver-for-{index}",
    )

    payload = runtime_numeric_fingerprint()

    assert payload["cuda_available"] is True
    assert payload["gpu_name"] == "Synthetic RTX"
    assert payload["compute_capability"] == [9, 1]
    assert payload["gpu_total_memory"] == 24_000_000_000
    assert payload["gpu_multiprocessor_count"] == 128
    assert payload["nvidia_driver_version"] == "driver-for-3"
    for field_name in (
        "cudnn_version",
        "cudnn_enabled",
        "cudnn_benchmark",
        "cudnn_deterministic",
        "flash_sdp_enabled",
        "mem_efficient_sdp_enabled",
        "math_sdp_enabled",
        "cudnn_sdp_enabled",
        "fp16_bf16_reduction_math_sdp_allowed",
        "preferred_blas_library",
        "matmul_allow_tf32",
        "matmul_allow_fp16_reduced_precision_reduction",
        "matmul_allow_bf16_reduced_precision_reduction",
        "matmul_allow_fp16_accumulation",
        "cudnn_allow_tf32",
    ):
        assert field_name in payload


def test_nvidia_driver_fingerprint_requires_one_successful_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _result(*, returncode: int, stdout: str):
        return batch_validation_module.subprocess.CompletedProcess(
            args=["nvidia-smi"],
            returncode=returncode,
            stdout=stdout,
            stderr="",
        )

    monkeypatch.setattr(
        batch_validation_module.subprocess,
        "run",
        lambda *_args, **_kwargs: _result(returncode=0, stdout="580.82.09\n"),
    )
    assert batch_validation_module._nvidia_driver_version(0) == "580.82.09"
    monkeypatch.setattr(
        batch_validation_module.subprocess,
        "run",
        lambda *_args, **_kwargs: _result(returncode=1, stdout=""),
    )
    assert batch_validation_module._nvidia_driver_version(0) is None
    monkeypatch.setattr(
        batch_validation_module.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("missing")),
    )
    assert batch_validation_module._nvidia_driver_version(0) is None


def test_runtime_profile_gate_accepts_exact_fingerprint_and_rejects_stale_profile() -> None:
    assert_runtime_matches_profile(_profile())
    actual = runtime_numeric_fingerprint_sha256()
    assert_runtime_matches_profile(_profile(runtime_fingerprint_sha256=actual))
    with pytest.raises(AutoInterpBatchValidationError, match="not covered.*expected=.*actual="):
        assert_runtime_matches_profile(_profile(runtime_fingerprint_sha256="0" * 64))


@pytest.mark.parametrize(
    "bad",
    ["", "a" * 63, "a" * 65, "A" * 64, "g" * 64],
)
def test_runtime_profile_fingerprint_must_be_lowercase_sha256(bad: str) -> None:
    with pytest.raises(BeartypeCallHintParamViolation, match="runtime_fingerprint_sha256"):
        _profile(runtime_fingerprint_sha256=bad)


def _observations() -> list[dict]:
    return [
        {
            "target_key": "feature:1",
            "surface": "selected_sae_startup",
            "surface_kind": "selected_sae",
            "context": "feature:1",
            "require_support_bound": True,
            "count": 128,
            "support_mismatches": 0,
            "max_abs": 0.1,
            "p99_abs": 0.05,
            "mean_abs": 0.01,
        },
        {
            "target_key": "feature:1",
            "surface": "raw_residual_startup",
            "surface_kind": "raw_residual",
            "context": "feature:2",
            "require_support_bound": False,
            "count": 32768,
            "support_mismatches": 7,
            "max_abs": 0.2,
            "p99_abs": 0.04,
            "mean_abs": 0.02,
        },
        {
            "target_key": "feature:1",
            "surface": "selected_sae_periodic_shadow",
            "surface_kind": "selected_sae",
            "context": "feature:3",
            "require_support_bound": True,
            "count": 128,
            "support_mismatches": 0,
            "max_abs": 0.15,
            "p99_abs": 0.06,
            "mean_abs": 0.03,
        },
    ]


def test_derive_profile_uses_calibration_maxima_margin_runtime_and_observation_hash() -> None:
    profile = derive_batch_tolerance_profile(
        _observations(),
        name="derived-v1",
        batch_size=128,
        calibration_contract=_contract(),
        margin=1.25,
    )
    assert profile.name == "derived-v1"
    assert profile.batch_size == 128
    assert profile.selected_sae.max_abs == math.nextafter(0.15 * 1.25, math.inf)
    assert profile.selected_sae.p99_abs == math.nextafter(0.06 * 1.25, math.inf)
    assert profile.selected_sae.mean_abs == math.nextafter(0.03 * 1.25, math.inf)
    assert profile.raw_residual.max_abs == math.nextafter(0.2 * 1.25, math.inf)
    assert profile.raw_residual.p99_abs == math.nextafter(0.04 * 1.25, math.inf)
    assert profile.raw_residual.mean_abs == math.nextafter(0.02 * 1.25, math.inf)
    assert profile.max_support_mismatches == 0
    assert profile.runtime_fingerprint_sha256 == runtime_numeric_fingerprint_sha256()
    assert profile.calibration_observations_sha256 is not None
    assert len(profile.calibration_observations_sha256) == 64


def test_derive_profile_is_order_sensitive_and_absolute_floor_is_explicit() -> None:
    observations = _observations()
    first = derive_batch_tolerance_profile(
        observations,
        name="derived-v1",
        batch_size=128,
        calibration_contract=_contract(),
        absolute_floor=0.5,
    )
    reordered = derive_batch_tolerance_profile(
        list(reversed(observations)),
        name="derived-v1",
        batch_size=128,
        calibration_contract=_contract(),
        absolute_floor=0.5,
    )
    for tolerance in (first.selected_sae, first.raw_residual):
        assert tolerance.max_abs == math.nextafter(0.5, math.inf)
        assert tolerance.p99_abs == math.nextafter(0.5, math.inf)
        assert tolerance.mean_abs == math.nextafter(0.5, math.inf)
    assert first.calibration_observations_sha256 != reordered.calibration_observations_sha256


def test_derive_profile_rejects_active_support_mismatch_instead_of_widening_profile() -> None:
    observations = _observations()
    observations[0] = {**observations[0], "support_mismatches": 1}
    with pytest.raises(AutoInterpBatchValidationError, match="active-support mismatch"):
        derive_batch_tolerance_profile(
            observations,
            name="bad",
            batch_size=128,
            calibration_contract=_contract(),
        )


def test_derive_profile_requires_both_numeric_surfaces_for_every_target() -> None:
    observations = _observations()
    observations[2] = {**observations[2], "target_key": "feature:2"}
    with pytest.raises(
        AutoInterpBatchValidationError,
        match="every target.*feature:2.*raw_residual",
    ):
        derive_batch_tolerance_profile(
            observations,
            name="incomplete-target",
            batch_size=128,
            calibration_contract=_contract(),
        )


@pytest.mark.parametrize(
    ("mutate", "fragment"),
    [
        (lambda rows: [], "zero observations"),
        (lambda rows: [42], "not an object"),
        (
            lambda rows: [{key: value for key, value in rows[0].items() if key != "count"}],
            "missing fields",
        ),
        (lambda rows: [{**rows[0], "target_key": ""}, *rows[1:]], "invalid target_key"),
        (lambda rows: [{**rows[0], "surface": ""}, *rows[1:]], "invalid surface"),
        (
            lambda rows: [{**rows[0], "surface_kind": "other"}, *rows[1:]],
            "invalid surface_kind",
        ),
        (
            lambda rows: [{**rows[0], "surface_kind": "raw_residual"}, *rows[1:]],
            "does not match surface_kind",
        ),
        (
            lambda rows: [{**rows[0], "require_support_bound": False}, *rows[1:]],
            "expected True for surface_kind",
        ),
        (lambda rows: [{**rows[0], "context": ""}, *rows[1:]], "invalid context"),
        (lambda rows: [{**rows[0], "require_support_bound": 1}, *rows[1:]], "not bool"),
        (lambda rows: [{**rows[0], "count": 0}, *rows[1:]], "invalid count"),
        (lambda rows: [{**rows[0], "support_mismatches": -1}, *rows[1:]], "invalid support"),
        (
            lambda rows: [{**rows[0], "support_mismatches": 129}, *rows[1:]],
            "invalid support",
        ),
        (lambda rows: [{**rows[0], "max_abs": math.inf}, *rows[1:]], "invalid max_abs"),
        (
            lambda rows: [{**rows[0], "p99_abs": 0.2}, *rows[1:]],
            "p99_abs/mean_abs exceeds max_abs",
        ),
        (
            lambda rows: [
                {**rows[0], "surface_kind": "selected_sae"},
                {
                    **rows[1],
                    "surface_kind": "selected_sae",
                    "surface": "selected_sae_x",
                    "require_support_bound": True,
                    "support_mismatches": 0,
                },
            ],
            "requires both selected-SAE and raw-residual",
        ),
    ],
)
def test_derive_profile_rejects_malformed_or_incomplete_calibration_observations(
    mutate,
    fragment: str,
) -> None:
    with pytest.raises(AutoInterpBatchValidationError, match=fragment):
        derive_batch_tolerance_profile(
            mutate(_observations()),
            name="bad",
            batch_size=128,
            calibration_contract=_contract(),
        )


@pytest.mark.parametrize(
    ("kwargs", "fragment"),
    [
        ({"margin": 0.99}, "margin"),
        ({"margin": True}, "margin"),
        ({"margin": math.inf}, "margin"),
        ({"absolute_floor": -1.0}, "absolute_floor"),
        ({"absolute_floor": True}, "absolute_floor"),
        ({"absolute_floor": math.nan}, "absolute_floor"),
    ],
)
def test_derive_profile_rejects_invalid_margin_and_floor(kwargs: dict, fragment: str) -> None:
    with pytest.raises(AutoInterpBatchValidationError, match=fragment):
        derive_batch_tolerance_profile(
            _observations(),
            name="bad",
            batch_size=128,
            calibration_contract=_contract(),
            **kwargs,
        )


def test_measure_batch_difference_reports_exact_support_and_error_surface() -> None:
    reference = t.tensor([0.0, 1.0, -2.0, 4.0], dtype=t.float32)
    candidate = t.tensor([0.0, 1.1, 0.5, 3.8], dtype=t.float32)

    metrics = measure_batch_difference(reference, candidate)

    assert metrics.count == 4
    assert metrics.support_mismatches == 1
    assert metrics.max_abs == pytest.approx(2.5)
    assert metrics.p99_abs == pytest.approx(float(t.quantile(t.tensor([0.0, 0.1, 2.5, 0.2]), 0.99)))
    assert metrics.mean_abs == pytest.approx(0.7)


def test_measure_batch_difference_exact_tensors_have_zero_metrics() -> None:
    tensor = t.linspace(-1.0, 1.0, 257, dtype=t.float32)
    metrics = measure_batch_difference(tensor, tensor.clone())
    assert metrics.count == 257
    assert metrics.support_mismatches == 0
    assert metrics.max_abs == 0.0
    assert metrics.p99_abs == 0.0
    assert metrics.mean_abs == 0.0


def test_measure_batch_difference_rejects_shape_dtype_empty_and_nonfinite() -> None:
    with pytest.raises(AutoInterpBatchValidationError, match="shape mismatch"):
        measure_batch_difference(t.zeros(2), t.zeros(3))
    with pytest.raises(AutoInterpBatchValidationError, match="requires.*float32"):
        measure_batch_difference(t.zeros(2, dtype=t.float64), t.zeros(2, dtype=t.float64))
    with pytest.raises(AutoInterpBatchValidationError, match="surface is empty"):
        measure_batch_difference(t.empty(0, dtype=t.float32), t.empty(0, dtype=t.float32))
    for nonfinite in (math.nan, math.inf, -math.inf):
        with pytest.raises(AutoInterpBatchValidationError, match="NaN or infinity"):
            measure_batch_difference(
                t.tensor([0.0, nonfinite], dtype=t.float32),
                t.zeros(2, dtype=t.float32),
            )
        with pytest.raises(AutoInterpBatchValidationError, match="NaN or infinity"):
            measure_batch_difference(
                t.zeros(2, dtype=t.float32),
                t.tensor([0.0, nonfinite], dtype=t.float32),
            )


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA device check requires CUDA")
def test_measure_batch_difference_rejects_cuda_comparison_surface() -> None:
    reference = t.zeros(2, dtype=t.float32, device="cuda")
    candidate = t.zeros(2, dtype=t.float32, device="cuda")
    with pytest.raises(AutoInterpBatchValidationError, match="must be on CPU"):
        measure_batch_difference(reference, candidate)


def test_measure_batch_difference_rejects_non_cpu_surface_without_cuda() -> None:
    reference = t.zeros(2, dtype=t.float32, device="meta")
    candidate = t.zeros(2, dtype=t.float32, device="meta")
    with pytest.raises(AutoInterpBatchValidationError, match="must be on CPU"):
        measure_batch_difference(reference, candidate)


def test_profile_gate_accepts_values_exactly_on_every_boundary() -> None:
    reference = t.ones(100, dtype=t.float32)
    candidate = t.full((100,), 1.05, dtype=t.float32)
    measured = measure_batch_difference(reference, candidate)
    profile = _profile(
        max_abs=measured.max_abs,
        p99_abs=measured.p99_abs,
        mean_abs=measured.mean_abs,
    )

    metrics = assert_batch_difference_within_profile(
        reference,
        candidate,
        profile=profile,
        context="boundary",
        surface_kind="selected_sae",
    )

    assert metrics.max_abs == pytest.approx(0.05)
    assert metrics.p99_abs == pytest.approx(0.05)
    assert metrics.mean_abs == pytest.approx(0.05)


@pytest.mark.parametrize("metric", ["max_abs", "p99_abs", "mean_abs"])
def test_profile_gate_rejects_one_nextafter_beyond_each_numeric_boundary(
    metric: str,
) -> None:
    reference = t.ones(100, dtype=t.float32)
    candidate = reference + t.arange(1, 101, dtype=t.float32) / 100.0
    metrics = measure_batch_difference(reference, candidate)
    measured = getattr(metrics, metric)
    profile = _profile(
        max_abs=metrics.max_abs,
        p99_abs=metrics.p99_abs,
        mean_abs=metrics.mean_abs,
    )
    profile = replace(
        profile,
        selected_sae=replace(
            profile.selected_sae,
            **{metric: math.nextafter(measured, -math.inf)},
        ),
    )

    with pytest.raises(AutoInterpBatchValidationError, match=metric):
        assert_batch_difference_within_profile(
            reference,
            candidate,
            profile=profile,
            context=f"nextafter-{metric}",
            surface_kind="selected_sae",
        )


def test_profile_gate_rejects_support_flip_even_when_numeric_error_is_tiny() -> None:
    tiny = t.finfo(t.float32).tiny
    reference = t.tensor([0.0, tiny], dtype=t.float32)
    candidate = t.tensor([tiny, tiny], dtype=t.float32)
    with pytest.raises(AutoInterpBatchValidationError, match="support_mismatches=1>0"):
        assert_batch_difference_within_profile(
            reference,
            candidate,
            profile=_profile(max_abs=1.0, p99_abs=1.0, mean_abs=1.0),
            context="support-flip",
            surface_kind="selected_sae",
        )


def test_profile_gate_can_explicitly_ignore_sign_support_for_raw_residuals() -> None:
    tiny = t.finfo(t.float32).tiny
    reference = t.tensor([-tiny, tiny], dtype=t.float32)
    candidate = t.tensor([tiny, -tiny], dtype=t.float32)
    metrics = assert_batch_difference_within_profile(
        reference,
        candidate,
        profile=_profile(max_abs=1.0, p99_abs=1.0, mean_abs=1.0),
        context="raw-residual-signs",
        surface_kind="raw_residual",
        require_support_bound=False,
    )
    assert metrics.support_mismatches == 2


def test_selected_sae_and_raw_residual_numeric_envelopes_are_independent() -> None:
    profile = _profile(
        selected_sae=NumericTolerance(max_abs=0.01, p99_abs=0.01, mean_abs=0.01),
        raw_residual=NumericTolerance(max_abs=1.0, p99_abs=1.0, mean_abs=1.0),
    )
    reference = t.ones(100, dtype=t.float32)
    candidate = t.full((100,), 1.5, dtype=t.float32)

    assert_batch_difference_within_profile(
        reference,
        candidate,
        profile=profile,
        context="residual-independent",
        surface_kind="raw_residual",
        require_support_bound=False,
    )
    with pytest.raises(AutoInterpBatchValidationError, match="max_abs"):
        assert_batch_difference_within_profile(
            reference,
            candidate,
            profile=profile,
            context="sae-independent",
            surface_kind="selected_sae",
        )


@pytest.mark.parametrize(
    ("surface_kind", "require_support_bound"),
    [("selected_sae", False), ("raw_residual", True)],
)
def test_profile_gate_rejects_surface_support_policy_mismatch(
    surface_kind: str,
    require_support_bound: bool,
) -> None:
    with pytest.raises(AutoInterpBatchValidationError, match="requires require_support_bound"):
        assert_batch_difference_within_profile(
            t.ones(2, dtype=t.float32),
            t.ones(2, dtype=t.float32),
            profile=_profile(),
            context="wrong-policy",
            surface_kind=cast(Any, surface_kind),
            require_support_bound=require_support_bound,
        )


def test_profile_gate_reports_all_simultaneous_failures_and_context() -> None:
    reference = t.zeros(10, dtype=t.float32)
    candidate = t.ones(10, dtype=t.float32)
    with pytest.raises(AutoInterpBatchValidationError) as caught:
        assert_batch_difference_within_profile(
            reference,
            candidate,
            profile=_profile(max_abs=0.1, p99_abs=0.1, mean_abs=0.1),
            context="feature=123 block=0 slot=123",
            surface_kind="selected_sae",
        )
    message = str(caught.value)
    assert "feature=123 block=0 slot=123" in message
    assert "profile=unit-test-profile" in message
    assert "support_mismatches" in message
    assert "max_abs" in message
    assert "p99_abs" in message
    assert "mean_abs" in message


def test_profile_gate_catches_scale_bug_that_preserves_support() -> None:
    reference = t.linspace(0.1, 10.0, 1000, dtype=t.float32)
    candidate = reference * 0.5
    with pytest.raises(AutoInterpBatchValidationError, match="max_abs"):
        assert_batch_difference_within_profile(
            reference,
            candidate,
            profile=_profile(max_abs=0.01, p99_abs=0.01, mean_abs=0.01),
            context="injected-half-scale",
            surface_kind="selected_sae",
        )


def test_candidate_repeat_gate_accepts_exact_clone() -> None:
    first = t.randn(17, 31, generator=t.Generator().manual_seed(7))
    assert_candidate_repeat_exact(first, first.clone(), context="exact")


@pytest.mark.parametrize(
    ("second", "fragment"),
    [
        (t.zeros(3, dtype=t.float32), "max_abs=None"),
        (t.zeros(2, dtype=t.float64), "float64"),
        (t.tensor([0.0, 1.0], dtype=t.float32), "max_abs=1.0"),
    ],
)
def test_candidate_repeat_gate_rejects_shape_dtype_and_value_drift(
    second: t.Tensor,
    fragment: str,
) -> None:
    first = t.zeros(2, dtype=t.float32)
    with pytest.raises(AutoInterpBatchValidationError) as caught:
        assert_candidate_repeat_exact(first, second, context="repeat-regression")
    assert "repeat-regression" in str(caught.value)
    assert fragment in str(caught.value)


@pytest.mark.parametrize(
    ("updates", "fragment"),
    [
        ({"count": 0}, "count"),
        ({"count": True}, "count"),
        ({"support_mismatches": -1}, "support_mismatches"),
        ({"support_mismatches": 3}, "support_mismatches"),
        ({"support_mismatches": True}, "support_mismatches"),
        ({"max_abs": float("nan")}, "max_abs"),
        ({"p99_abs": float("inf")}, "p99_abs"),
        ({"mean_abs": -0.1}, "mean_abs"),
        ({"max_abs": 0.1, "p99_abs": 0.2}, "must not exceed"),
        ({"max_abs": 0.1, "mean_abs": 0.2}, "must not exceed"),
    ],
)
def test_precomputed_difference_metrics_reject_every_malformed_scalar(
    updates: dict[str, Any],
    fragment: str,
) -> None:
    values: dict[str, Any] = {
        "count": 2,
        "support_mismatches": 0,
        "max_abs": 0.1,
        "p99_abs": 0.05,
        "mean_abs": 0.02,
    }
    values.update(updates)
    relationship_error = len(updates) > 1 or updates.get("support_mismatches") == 3
    error_type = (
        AutoInterpBatchValidationError if relationship_error else BeartypeCallHintParamViolation
    )
    match = fragment if relationship_error else next(iter(updates))
    with pytest.raises(error_type, match=match):
        BatchDifferenceMetrics(**values)


def test_precomputed_metric_gate_is_identical_to_tensor_gate() -> None:
    profile = _profile(max_abs=0.2, p99_abs=0.2, mean_abs=0.2)
    reference = t.tensor([0.0, 1.0, 2.0], dtype=t.float32)
    candidate = t.tensor([0.0, 1.1, 1.9], dtype=t.float32)
    measured = assert_batch_difference_within_profile(
        reference,
        candidate,
        profile=profile,
        context="tensor",
        surface_kind="selected_sae",
    )

    assert_batch_metrics_within_profile(
        measured,
        profile=profile,
        context="precomputed",
        surface_kind="selected_sae",
    )

    with pytest.raises(AutoInterpBatchValidationError, match="metrics must be"):
        assert_batch_metrics_within_profile(
            cast(Any, {"max_abs": 0.0}),
            profile=profile,
            context="wrong-type",
            surface_kind="selected_sae",
        )
    with pytest.raises(AutoInterpBatchValidationError, match="max_abs"):
        assert_batch_metrics_within_profile(
            replace(measured, max_abs=0.3),
            profile=profile,
            context="too-large",
            surface_kind="selected_sae",
        )
