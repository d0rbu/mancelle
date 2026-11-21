"""#1249: `_run_capture_forward` must never run an encdec forward unconditioned.

The old code fed `build_no_feature_inputs` -- an all-masked bag whose
cross-attention NaNs were folded to zero, "effectively a decoder-only residual
trajectory on the encdec backbone". These tests pin the replacement contract:

* encdec WITHOUT conditioning -> raise (no fallback of any kind);
* encdec WITH conditioning -> the forward receives exactly the bag's tensors;
* decoder-only WITH conditioning -> raise (the caller has the targets crossed);
* decoder-only without -> plain ``model(idx)``, unchanged.
"""

from __future__ import annotations

import pytest
import torch as t
import torch.nn as nn

from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.eval.axbench_base.baselines._capture import _run_capture_forward
from steering.eval.axbench_base.baselines.conditioning import EncoderConditioning
from steering.eval.axbench_base.types import AxBenchBaseError

_COND = EncoderConditioning(entries=((3, 1), (8, 0)))


class _EncDecStub(EncoderDecoderGPT):
    """An EncoderDecoderGPT by isinstance, recording its forward kwargs.

    ``__init__`` skips the real constructor deliberately: the guard under test
    dispatches on ``isinstance`` and the forward signature, neither of which
    needs real weights.
    """

    def __init__(self) -> None:  # noqa: D107 -- see class docstring
        nn.Module.__init__(self)
        self.calls: list[dict[str, t.Tensor | None]] = []

    def forward(self, idx, targets=None, **kwargs):  # ty: ignore[invalid-method-override]
        self.calls.append({"idx": idx, "targets": targets, **kwargs})
        return None


class _DecoderStub(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.calls: list[t.Tensor] = []

    def forward(self, idx):  # type: ignore[override]
        self.calls.append(idx)
        return None


def test_encdec_without_conditioning_raises_and_never_forwards() -> None:
    model = _EncDecStub()
    with pytest.raises(AxBenchBaseError, match="1249"):
        _run_capture_forward(model, t.zeros((1, 4), dtype=t.long), t.device("cpu"))
    assert model.calls == [], "the guard must fire BEFORE any forward runs"


def test_encdec_with_conditioning_forwards_the_bag_tensors() -> None:
    model = _EncDecStub()
    idx = t.zeros((1, 5), dtype=t.long)
    _run_capture_forward(
        model, idx, t.device("cpu"), encoder_conditioning=_COND
    )
    assert len(model.calls) == 1
    call = model.calls[0]
    want_fids, want_mags, want_mask = _COND.as_tensors(t.device("cpu"))
    enc_feature_ids = call["enc_feature_ids"]
    enc_magnitudes = call["enc_magnitudes"]
    enc_mask = call["enc_mask"]
    assert isinstance(enc_feature_ids, t.Tensor) and t.equal(enc_feature_ids, want_fids)
    assert isinstance(enc_magnitudes, t.Tensor) and t.equal(enc_magnitudes, want_mags)
    assert isinstance(enc_mask, t.Tensor) and t.equal(enc_mask, want_mask)
    assert call["targets"] is None
    seg_map = call["seg_map"]
    assert isinstance(seg_map, t.Tensor)
    assert seg_map.shape == idx.shape and int(seg_map.abs().sum()) == 0, (
        "one bag = one segment: the whole continuation maps to segment 0"
    )


def test_decoder_only_with_conditioning_raises_crossed_targets() -> None:
    model = _DecoderStub()
    with pytest.raises(AxBenchBaseError, match="crossed"):
        _run_capture_forward(
            model, t.zeros((1, 4), dtype=t.long), t.device("cpu"),
            encoder_conditioning=_COND,
        )
    assert model.calls == []


def test_decoder_only_path_is_unchanged() -> None:
    model = _DecoderStub()
    idx = t.zeros((1, 4), dtype=t.long)
    _run_capture_forward(model, idx, t.device("cpu"))
    assert len(model.calls) == 1 and model.calls[0] is idx


def test_build_no_feature_inputs_is_gone_from_production() -> None:
    """The defect's mechanism must not be reachable from the capture module."""
    import steering.eval.axbench_base.baselines._capture as capture

    assert not hasattr(capture, "build_no_feature_inputs"), (
        "build_no_feature_inputs must not be importable from the capture "
        "module; its presence is how the unconditioned forward comes back"
    )
