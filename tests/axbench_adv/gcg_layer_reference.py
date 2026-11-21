"""Frozen full-depth capture from 700f84f6c5629d53d4063693cd645c548d191e92.

Only the name/signature and module-helper bindings differ from the parent.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any
from unittest.mock import patch

import torch as t


@contextmanager
def full_decoder_depth(worker: Any) -> Iterator[None]:
    """Collect the target activation using the frozen full decoder forward."""

    def capture(session: Any, embeds: t.Tensor, attention: t.Tensor) -> t.Tensor:
        return full_depth_capture(worker, session, embeds, attention)

    with patch.object(worker.GCGSession, "_capture_hidden", capture):
        yield


def full_depth_capture(worker: Any, self: Any, embeds: t.Tensor, attention: t.Tensor) -> t.Tensor:
    _ADDITIVE_METHODS = worker._ADDITIVE_METHODS
    _subspace = worker._subspace
    decoder_layers = worker.decoder_layers
    _forward = worker._forward
    # The collector needs every hidden state, but no vocabulary logits. HF
    # slices only the LM-head input, leaving the layer output and VJP intact.
    if self.runtime.method in _ADDITIVE_METHODS:
        if self._activation_model is None:
            raise ValueError("activation collector is unavailable")
        (_base, collected), _output = self._activation_model(
            {"inputs_embeds": embeds, "attention_mask": attention, "logits_to_keep": 1},
            unit_locations=None,
            subspaces=[_subspace(self.runtime, embeds.shape[0]), None],
            use_cache=False,
        )
        if not isinstance(collected, list) or len(collected) != 1:
            raise RuntimeError("activation collector did not return exactly one block output")
        return collected[0]
    layer = decoder_layers(self.runtime.benchmark_model.model)[self.runtime.layer]
    captured: list[t.Tensor] = []

    def capture(_module: Any, _inputs: Any, output: Any) -> None:
        captured.append(output[0] if isinstance(output, (tuple, list)) else output)

    handle = layer.register_forward_hook(capture, always_call=True)
    try:
        _forward(self.runtime, embeds=embeds, attention_mask=attention, logits_to_keep=1)
    finally:
        handle.remove()
    if len(captured) != 1:
        raise RuntimeError("activation_l2 did not capture exactly one decoder-layer output")
    return captured[0]
