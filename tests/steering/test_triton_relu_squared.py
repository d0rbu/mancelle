from __future__ import annotations

import sys

import pytest
import torch as t
import torch.nn.functional as F

if sys.platform == "linux":
    from steering.eval.triton_relu_squared import triton_relu_squared_
else:
    triton_relu_squared_ = None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("dtype", [t.bfloat16, t.float16])
def test_triton_relu_squared_matches_torch_in_place(dtype: t.dtype) -> None:
    assert triton_relu_squared_ is not None
    device = t.device("cuda")
    generator = t.Generator(device=device).manual_seed(2828)
    source = t.randn(
        73,
        1,
        6656,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    reference = F.relu(source).square()
    candidate = source.clone()
    pointer = candidate.data_ptr()

    returned = triton_relu_squared_(candidate)
    t.cuda.synchronize()

    assert returned.data_ptr() == pointer
    assert returned is candidate
    assert t.equal(candidate, reference)
