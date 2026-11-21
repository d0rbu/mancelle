"""Run the production Triton recurrence on CPU, in an isolated interpreter."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
import torch.nn.functional as F

if sys.platform == "linux":
    from steering.eval.triton_ragged_attention import _ragged_decode_attention_kernel


@pytest.mark.skipif(sys.platform != "linux", reason="Triton requires Linux")
@pytest.mark.parametrize("window_left", [512, 0, -1])
def test_ragged_softmax_empty_tiles_cpu(window_left: int) -> None:
    # The interpreter flag must precede import of @triton.jit functions. Keep
    # it out of the pytest process so GPU tests retain the compiled kernel.
    repo = Path(__file__).resolve().parents[2]
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), str(window_left)],
        cwd=repo,
        env={
            **os.environ,
            "PYTHONPATH": str(repo),
            "CUDA_VISIBLE_DEVICES": "",
            "TRITON_INTERPRET": "1",
        },
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def _check_interpreted_kernel(window_left: int) -> None:
    t.manual_seed(42)
    # D14 has a 512-token left window. Position 575 leaves key 63 in the
    # first tile; position 576 masks that entire tile for the first time.
    # Later rows exercise multiple empty tiles and the final partial tile.
    positions = t.tensor([575, 576, 640, 990])
    batch, cache_end, q_heads, kv_heads, head_dim = 4, 1001, 4, 2, 128
    q = t.randn(batch, q_heads, head_dim)
    k = t.randn(batch, cache_end, kv_heads, head_dim)
    v = t.randn_like(k)
    keys = t.arange(cache_end)[None, :]
    mask = keys <= positions[:, None]
    if window_left >= 0:
        mask &= keys >= positions[:, None] - window_left
    reference = F.scaled_dot_product_attention(
        q.unsqueeze(2),
        k.transpose(1, 2),
        v.transpose(1, 2),
        attn_mask=mask[:, None, None, :],
        enable_gqa=True,
    ).squeeze(2)
    # Masked storage must not be loaded, whether it is a past window tile or
    # unused future cache. This also rules out reliance on zero allocation.
    k.masked_fill_(~mask[:, :, None, None], t.nan)
    v.masked_fill_(~mask[:, :, None, None], t.nan)
    actual = t.empty_like(q)
    kernel = cast(Any, _ragged_decode_attention_kernel)
    kernel[(batch, q_heads)](
        actual,
        q,
        k,
        v,
        mask,
        q.stride(0),
        q.stride(1),
        k.stride(0),
        k.stride(1),
        k.stride(2),
        mask.stride(0),
        actual.stride(0),
        actual.stride(1),
        cache_end,
        q_heads // kv_heads,
        head_dim,
        head_dim**-0.5,
        block_n=64,
        block_d=head_dim,
        num_warps=4,
    )
    assert t.isfinite(actual).all(), (window_left, positions, actual)
    t.testing.assert_close(actual, reference, atol=2e-6, rtol=2e-5)


if __name__ == "__main__":
    _check_interpreted_kernel(int(sys.argv[1]))
