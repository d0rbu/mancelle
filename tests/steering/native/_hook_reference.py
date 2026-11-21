"""TEST-ONLY reference.

Verbatim copy of the Phase-1 `NativeCollector.capture` residual hook
mechanism, which PR-1 removes from production (spec §3.2). Gates §4.1/§4.6
assert the truncated collector is byte-identical to this. NOT imported by
production code.
"""

from __future__ import annotations

import torch as t

from errors import ExtractionError


def hook_reference_residual(
    hf_model,
    input_ids: t.Tensor,
    layer: int,
) -> t.Tensor:
    """Phase-1 residual capture: forward hook on decoder layer `layer`.

    Lifted VERBATIM from `steering.native_collector.NativeCollector.capture`
    (residual path): the resolved residual target is the decoder layer
    `hf_model.model.layers[layer]` (`resolve_capture_target` →
    `CaptureKind.layer_output`); a `register_forward_hook` unwraps a
    tuple/list output via `[0]` and stores `.detach().clone()`; the full
    forward runs under `torch.no_grad()` and is MASKLESS (no
    `attention_mask`); the hook handle is removed in a `finally:`. Returns
    the layer-`L` residual `[B, S, d]` (resid_post(L)); padding is NOT
    stripped here (the Phase-1 caller does that).
    """
    module = hf_model.model.layers[layer]

    captured: dict[str, t.Tensor] = {}

    def _fwd_hook(_mod, _inp, out):
        tensor = out[0] if isinstance(out, (tuple, list)) else out
        if not isinstance(tensor, t.Tensor):
            raise ExtractionError(
                f"native: hooked module produced non-tensor {type(tensor)}"
            )
        captured["v"] = tensor.detach().clone()

    handle = module.register_forward_hook(_fwd_hook)

    try:
        with t.no_grad():
            hf_model(input_ids=input_ids, attention_mask=None)
    finally:
        handle.remove()

    return captured["v"]
