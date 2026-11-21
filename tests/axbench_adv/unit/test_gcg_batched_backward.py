"""Independent rowwise parity and batch ownership for one-backward GCG."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest
import torch as t
from safetensors import safe_open

from tests.axbench_adv.gcg_gradient_reference import rowwise_vjp
from tests.axbench_adv.unit.test_gcg_vendor_worker import (
    GCGSession,
    _runtime,
    _write_activation_artifacts,
    _write_manifest,
)


@pytest.mark.parametrize("objective", ["response_margin", "activation_l2"])
@pytest.mark.parametrize("batch_size", [1, 2, 4, 8])
@pytest.mark.parametrize("alpha", [0.0, 0.01, 1.0])
def test_vendor_joint_vjp_preserves_rowwise_pair_arithmetic(
    tmp_path: Path, objective: str, batch_size: int, alpha: float
) -> None:
    _write_manifest(tmp_path)
    _write_activation_artifacts(tmp_path)
    runtime = _runtime(tmp_path)
    session = GCGSession(runtime)
    prepared = session.prepare(
        {
            "objective": objective,
            "alpha": alpha,
            "device_batch_size": batch_size,
            "separator": "\n\n",
            "suffix_token_count": 3,
            "rows": [
                {
                    "pair_id": index,
                    "positive": {
                        "sample_id": 2 * index,
                        "prompt": "P" * (index + 1),
                        "response": "AB",
                    },
                    "negative": {
                        "sample_id": 2 * index + 1,
                        "prompt": "N",
                        "response": "C" * (index + 1),
                    },
                }
                for index in range(3)
            ],
        }
    )
    payload = {"token_ids": prepared["seed_token_ids"], "member_indices": [2, 0, 1]}
    path = runtime._gcg_sidecar_dir / "gradients.safetensors"
    original = t.autograd.grad
    counts: list[int] = []

    def observed(outputs: Any, inputs: Any, **kwargs: Any) -> Any:
        counts.append(1 if isinstance(outputs, t.Tensor) else len(outputs))
        return original(outputs, inputs, **kwargs)

    with patch.object(t.autograd, "grad", observed):
        result = session.gradient(payload)
    with safe_open(path, framework="pt", device="cpu") as handle:
        actual = handle.get_tensor("gradients").clone()
    with rowwise_vjp() as reference_counts:
        reference_result = session.gradient(payload)
    with safe_open(path, framework="pt", device="cpu") as handle:
        expected = handle.get_tensor("gradients").clone()
    assert result["member_indices"] == reference_result["member_indices"] == [2, 0, 1]
    assert len(counts) == math.ceil(6 / batch_size)
    assert sum(counts) == 6
    assert counts == reference_counts
    assert t.equal(actual, expected)
    # A separate forward for each class row also checks the objective scale:
    # the joint backward must not average or preweight the scalar losses.
    session.device_batch_size = 1
    session.gradient(payload)
    with safe_open(path, framework="pt", device="cpu") as handle:
        scalar_forward = handle.get_tensor("gradients")
    t.testing.assert_close(actual, scalar_forward, rtol=1e-5, atol=1e-6)
    session.close()


def test_rowwise_oracle_rejects_cross_row_coupling() -> None:
    inputs = t.tensor([[1.0, 2.0], [3.0, 5.0]], requires_grad=True)
    losses = tuple((inputs[index] + inputs.mean(dim=0)).square().sum() for index in range(2))
    with rowwise_vjp(), pytest.raises(AssertionError, match="rows are not independent"):
        t.autograd.grad(losses, inputs)
