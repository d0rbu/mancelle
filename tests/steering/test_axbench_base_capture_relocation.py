"""Pin deterministic residual capture for the decoder-only path."""

from __future__ import annotations

import torch as t

from steering.eval.axbench_base.baselines._capture import (
    _ResidualCaptureHook,
    _run_capture_forward,
)
from tests.conftest import build_tiny_decoder


def test_capture_residual_tensor_byte_parity():
    """End-to-end: tiny model + the residual-capture hook produces a
    deterministic residual tensor."""

    model = build_tiny_decoder()
    model.eval()
    idx = t.tensor([[1, 2, 3, 4]], dtype=t.long)
    device = t.device("cpu")
    with t.no_grad():
        with _ResidualCaptureHook(model=model, layer=0) as hook:
            _run_capture_forward(model, idx, device)
            out = hook.last
        assert out is not None
        assert out.shape == (1, 4, model.config.n_embd)
        assert out.dtype == t.float32
        # Determinism: same inputs -> same outputs across reruns.
        with _ResidualCaptureHook(model=model, layer=0) as hook:
            _run_capture_forward(model, idx, device)
            out2 = hook.last
        assert t.allclose(out, out2, atol=0.0)  # ty: ignore[invalid-argument-type]
