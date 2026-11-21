"""Unit tests for _select_spot_check_shards (slice-3 P1-4 fix).

The original ``_validate_percentile_checksums`` only ever inspected the
middle shard (``shard_paths[len(shard_paths) // 2]``); per the slice-3
plan we now spot-check first / middle / last by default. These unit
tests pin that contract and the API surface used by the verification
test harness.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch as t

import steering.validate_features as validate_module
from steering.feature_io import TensorKey, save_features
from steering.overlay_io import save_overlay_shard
from steering.validate_features import _select_spot_check_shards
from tests.steering._helpers import make_valid_shard_data


def _shard_paths(n: int) -> list[Path]:
    return [Path(f"/tmp/shard_{i:04d}.safetensors.zst") for i in range(n)]


def test_zero_shards_returns_empty() -> None:
    assert _select_spot_check_shards([], n_spot_check_shards=3) == []


def test_n_zero_returns_empty() -> None:
    paths = _shard_paths(5)
    assert _select_spot_check_shards(paths, n_spot_check_shards=0) == []


def test_n_negative_returns_empty() -> None:
    paths = _shard_paths(5)
    assert _select_spot_check_shards(paths, n_spot_check_shards=-1) == []


def test_one_shard_returns_that_shard() -> None:
    paths = _shard_paths(1)
    assert _select_spot_check_shards(paths, n_spot_check_shards=3) == paths


def test_two_shards_with_n_three_returns_both() -> None:
    paths = _shard_paths(2)
    assert _select_spot_check_shards(paths, n_spot_check_shards=3) == paths


def test_three_shards_with_n_three_returns_first_middle_last() -> None:
    paths = _shard_paths(3)
    selected = _select_spot_check_shards(paths, n_spot_check_shards=3)

    assert len(selected) == 3
    assert paths[0] in selected
    assert paths[1] in selected
    assert paths[2] in selected


def test_five_shards_with_n_three_includes_first_middle_last() -> None:
    """The original middle-only selection (P1-4) MISSED first/last; this
    test pins that the new selection covers both endpoints PLUS the middle."""
    paths = _shard_paths(5)
    selected = _select_spot_check_shards(paths, n_spot_check_shards=3)

    assert paths[0] in selected, f"Expected first shard {paths[0]} in {selected}"
    assert paths[2] in selected, f"Expected middle shard {paths[2]} in {selected}"
    assert paths[4] in selected, f"Expected last shard {paths[4]} in {selected}"


def test_returns_paths_in_sorted_index_order() -> None:
    paths = _shard_paths(7)
    selected = _select_spot_check_shards(paths, n_spot_check_shards=3)
    indices = [paths.index(p) for p in selected]
    assert indices == sorted(indices)


def test_two_shards_n_two_returns_first_and_last() -> None:
    paths = _shard_paths(2)
    selected = _select_spot_check_shards(paths, n_spot_check_shards=2)
    assert selected == [paths[0], paths[1]]


def test_n_greater_than_shard_count_returns_all() -> None:
    paths = _shard_paths(4)
    selected = _select_spot_check_shards(paths, n_spot_check_shards=10)
    assert selected == paths


def test_n_one_returns_only_first() -> None:
    paths = _shard_paths(5)
    selected = _select_spot_check_shards(paths, n_spot_check_shards=1)
    assert selected == [paths[0]]


def _cuda_producer_manifest(**updates: object) -> dict:
    producer = {
        "device_type": "cuda",
        "torch_version": "different-torch",
        "cuda_runtime": "different-cuda",
        "gpu_name": "different-gpu",
        "gpu_capability": [8, 9],
    }
    producer.update(updates)
    return {"metadata": {"producer": producer}}


def test_cuda_manifest_uses_documented_cpu_one_bin_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(t.cuda, "is_available", lambda: False)
    violations: list[str] = []
    resolved = validate_module._percentile_validation_producer(
        _cuda_producer_manifest(),
        violations,
    )
    assert resolved == (t.device("cpu"), 1)
    assert violations == []


@pytest.mark.parametrize(
    "producer",
    [
        "not-a-mapping",
        {"device_type": "tpu"},
        {
            "device_type": "cuda",
            "torch_version": t.__version__,
            "cuda_runtime": None,
            "gpu_name": None,
            "gpu_capability": None,
        },
        {
            "device_type": "cpu",
            "torch_version": "",
            "cuda_runtime": None,
            "gpu_name": None,
            "gpu_capability": None,
        },
        {
            "device_type": "cpu",
            "torch_version": t.__version__,
            "cuda_runtime": "12.8",
            "gpu_name": None,
            "gpu_capability": None,
        },
        {
            "device_type": "cuda",
            "torch_version": t.__version__,
            "cuda_runtime": "",
            "gpu_name": "gpu",
            "gpu_capability": [8, 9],
        },
        {
            "device_type": "cuda",
            "torch_version": t.__version__,
            "cuda_runtime": "12.8",
            "gpu_name": "",
            "gpu_capability": [8, 9],
        },
        {
            "device_type": "cuda",
            "torch_version": t.__version__,
            "cuda_runtime": "12.8",
            "gpu_name": "gpu",
            "gpu_capability": [-1, 9],
        },
    ],
)
def test_malformed_percentile_producer_contract_fails(producer: object) -> None:
    violations: list[str] = []
    assert validate_module._percentile_validation_producer(
        {"metadata": {"producer": producer}},
        violations,
    ) is None
    assert violations and "malformed" in violations[0]


def test_spot_check_accepts_only_documented_cross_device_one_bin_difference(
    tmp_path: Path,
) -> None:
    data = make_valid_shard_data(n_docs=1, segs_per_doc=1, feats_per_seg=1)
    data["feature_ids"] = t.tensor([0], dtype=t.int32)
    data["feature_mags"] = t.tensor([1.0], dtype=t.float32)
    base_path = save_features(data, tmp_path / "shard_00000_features")
    overlay_path = save_overlay_shard(
        {TensorKey.FEATURE_PERCENTILES.value: t.tensor([25], dtype=t.uint8)},
        tmp_path / "shard_00000",
    )
    table = t.arange(1, 100, dtype=t.float64).unsqueeze(0) * 0.1

    exact_violations: list[str] = []
    validate_module._spot_check_one_shard(
        base_path,
        overlay_path,
        table,
        exact_violations,
        1,
        random.Random(42),
        producer_device=t.device("cpu"),
        max_abs_bin_difference=0,
    )
    assert exact_violations

    fallback_violations: list[str] = []
    validate_module._spot_check_one_shard(
        base_path,
        overlay_path,
        table,
        fallback_violations,
        1,
        random.Random(42),
        producer_device=t.device("cpu"),
        max_abs_bin_difference=1,
    )
    assert fallback_violations == []
