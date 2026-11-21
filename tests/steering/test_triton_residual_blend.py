from __future__ import annotations

import sys
from typing import cast

import pytest
import torch as t

if sys.platform == "linux":
    from steering.eval.triton_residual_blend import triton_residual_blend
else:
    triton_residual_blend = None

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required"),
]


@pytest.mark.parametrize("output_dtype", [t.bfloat16, t.float16])
@pytest.mark.parametrize("first_layer", [False, True])
@pytest.mark.parametrize("shape", [(1, 1, 1), (3, 1, 257), (73, 1, 1664)])
def test_residual_blend_matches_eager_rounding_and_preserves_sources(
    output_dtype: t.dtype,
    first_layer: bool,
    shape: tuple[int, ...],
) -> None:
    assert triton_residual_blend is not None
    device = t.device("cuda")
    generator = t.Generator(device=device).manual_seed(75491)
    x0 = t.randn(shape, device=device, generator=generator)
    x = x0 if first_layer else t.randn(shape, device=device, generator=generator).to(output_dtype)
    # Three trained checkpoint pairs, nonrepresentable lower-precision scales,
    # and zero/unit coefficients exercise both products and their rounding.
    resid_lambdas = t.tensor(
        [1.1625638008117676, 0.1281435340642929, 0.7509695291519165, 0.701231, 0.0, 1.0],
        device=device,
    )
    x0_lambdas = t.tensor(
        [19.66294288635254, 48.840946197509766, -5.8309149742126465, -0.303791, 1.0, 0.0],
        device=device,
    )
    sources = (x, x0, resid_lambdas, x0_lambdas)
    snapshots = tuple(source.clone() for source in sources)

    with t.inference_mode(), t.autocast("cuda", dtype=output_dtype):
        for layer_idx in range(resid_lambdas.numel()):
            reference = (resid_lambdas[layer_idx] * x + x0_lambdas[layer_idx] * x0).to(output_dtype)
            candidate = triton_residual_blend(
                x,
                x0,
                resid_lambdas,
                x0_lambdas,
                layer_idx,
                output_dtype=output_dtype,
            )
            assert candidate.shape == x.shape
            assert candidate.dtype == output_dtype
            assert candidate.is_contiguous()
            assert candidate.data_ptr() not in (x.data_ptr(), x0.data_ptr())
            assert t.equal(candidate, reference)

    for source, snapshot in zip(sources, snapshots, strict=True):
        assert t.equal(source, snapshot)


@pytest.mark.parametrize("output_dtype", [t.bfloat16, t.float16])
def test_residual_blend_preserves_rounding_near_cancellation(output_dtype: t.dtype) -> None:
    assert triton_residual_blend is not None
    x = t.linspace(-4, 4, 2049, device="cuda").to(output_dtype)
    resid_lambdas = t.tensor([0.701231], device="cuda")
    x0_lambdas = t.tensor([-0.303791], device="cuda")
    x0 = -(resid_lambdas[0].to(output_dtype).float() * x.float()) / x0_lambdas[0]

    with t.inference_mode(), t.autocast("cuda", dtype=output_dtype):
        reference = (resid_lambdas[0] * x + x0_lambdas[0] * x0).to(output_dtype)
        naive = (resid_lambdas[0] * x.float() + x0_lambdas[0] * x0).to(output_dtype)
        candidate = triton_residual_blend(
            x,
            x0,
            resid_lambdas,
            x0_lambdas,
            0,
            output_dtype=output_dtype,
        )

    assert not t.equal(reference, naive)
    assert t.equal(candidate, reference)


@pytest.mark.parametrize("output_dtype", [t.bfloat16, t.float16])
def test_residual_blend_does_not_contract_fp32_product_and_sum(output_dtype: t.dtype) -> None:
    assert triton_residual_blend is not None
    # 4097**2 rounds from 16785409 to 16785408 in FP32. Contracting the
    # multiplication and addition would incorrectly leave 1 instead of 0.
    x = t.full((3, 1, 257), 4097.0, device="cuda")
    x0 = t.full_like(x, -16785408.0)
    resid_lambdas = t.tensor([4097.0], device="cuda")
    x0_lambdas = t.tensor([1.0], device="cuda")

    with t.inference_mode(), t.autocast("cuda", dtype=output_dtype):
        reference = (resid_lambdas[0] * x + x0_lambdas[0] * x0).to(output_dtype)
        candidate = triton_residual_blend(
            x,
            x0,
            resid_lambdas,
            x0_lambdas,
            0,
            output_dtype=output_dtype,
        )

    assert t.count_nonzero(reference).item() == 0
    assert t.equal(candidate, reference)


@pytest.mark.parametrize("output_dtype", [t.bfloat16, t.float16])
def test_residual_blend_reads_mutated_strided_parameter_arrays(output_dtype: t.dtype) -> None:
    assert triton_residual_blend is not None
    x = t.linspace(-3, 3, 259, device="cuda").to(output_dtype)
    x0 = t.linspace(1, 2, 259, device="cuda")
    resid_storage = t.tensor([99.0, 1.0, 99.0, 1.0, 99.0, 1.0], device="cuda")
    x0_storage = t.tensor([99.0, 0.0, 99.0, 99.0, 0.0, 99.0, 99.0, 0.0, 99.0], device="cuda")
    resid_lambdas = resid_storage[1::2]
    x0_lambdas = x0_storage[1::3]
    layer_idx = 1

    with t.inference_mode(), t.autocast("cuda", dtype=output_dtype):
        before = triton_residual_blend(
            x,
            x0,
            resid_lambdas,
            x0_lambdas,
            layer_idx,
            output_dtype=output_dtype,
        )
        resid_lambdas[layer_idx] = -0.701231
        x0_lambdas[layer_idx] = 0.303791
        snapshots = (resid_storage.clone(), x0_storage.clone())
        reference = (resid_lambdas[layer_idx] * x + x0_lambdas[layer_idx] * x0).to(output_dtype)
        after = triton_residual_blend(
            x,
            x0,
            resid_lambdas,
            x0_lambdas,
            layer_idx,
            output_dtype=output_dtype,
        )

    assert not t.equal(before, after)
    assert t.equal(before, x)
    assert t.equal(after, reference)
    assert t.equal(resid_storage, snapshots[0])
    assert t.equal(x0_storage, snapshots[1])


@pytest.mark.parametrize("output_dtype", [t.bfloat16, t.float16])
def test_residual_blend_empty_activations(output_dtype: t.dtype) -> None:
    assert triton_residual_blend is not None
    x = t.empty((0, 1, 1664), device="cuda", dtype=output_dtype)
    x0 = t.empty_like(x, dtype=t.float32)
    scalars = t.ones(2, device="cuda")

    output = triton_residual_blend(x, x0, scalars, scalars, 1, output_dtype=output_dtype)

    assert output.shape == x.shape
    assert output.dtype == output_dtype
    assert output is not x


@pytest.mark.parametrize(
    ("layer_idx", "error"),
    [(-1, IndexError), (2, IndexError), (True, TypeError), (0.0, TypeError)],
)
def test_residual_blend_rejects_invalid_layer_index(
    layer_idx: object,
    error: type[Exception],
) -> None:
    assert triton_residual_blend is not None
    x = t.ones((2, 1, 3), device="cuda")
    scalars = t.ones(2, device="cuda")

    with pytest.raises(error):
        triton_residual_blend(x, x, scalars, scalars, cast(int, layer_idx), output_dtype=t.bfloat16)


@pytest.mark.parametrize(
    ("case", "error"),
    [
        ("cpu_x", ValueError),
        ("cpu_x0", ValueError),
        ("cpu_resid", ValueError),
        ("cpu_x0_scalars", ValueError),
        ("x_dtype", TypeError),
        ("x0_dtype", TypeError),
        ("scalar_dtype", TypeError),
        ("scalar_shape", ValueError),
        ("activation_shape", ValueError),
        ("strided_x", ValueError),
        ("strided_x0", ValueError),
        ("output_dtype", TypeError),
    ],
)
def test_residual_blend_rejects_unsupported_inputs(case: str, error: type[Exception]) -> None:
    assert triton_residual_blend is not None
    x = t.ones((2, 1, 3), device="cuda")
    x0 = t.ones_like(x)
    resid_lambdas = t.ones(2, device="cuda")
    x0_lambdas = t.ones(2, device="cuda")
    output_dtype = t.bfloat16
    if case == "cpu_x":
        x = x.cpu()
    elif case == "cpu_x0":
        x0 = x0.cpu()
    elif case == "cpu_resid":
        resid_lambdas = resid_lambdas.cpu()
    elif case == "cpu_x0_scalars":
        x0_lambdas = x0_lambdas.cpu()
    elif case == "x_dtype":
        x = x.double()
    elif case == "x0_dtype":
        x0 = x0.bfloat16()
    elif case == "scalar_dtype":
        resid_lambdas = resid_lambdas.bfloat16()
    elif case == "scalar_shape":
        x0_lambdas = x0_lambdas.unsqueeze(0)
    elif case == "activation_shape":
        x0 = x0.flatten()
    elif case == "strided_x":
        x = t.ones((2, 1, 6), device="cuda")[..., ::2]
    elif case == "strided_x0":
        x0 = t.ones((2, 1, 6), device="cuda")[..., ::2]
    elif case == "output_dtype":
        output_dtype = t.float32

    with pytest.raises(error):
        triton_residual_blend(x, x0, resid_lambdas, x0_lambdas, 0, output_dtype=output_dtype)
