"""chat_eval rejects standalone target-only encoder-decoder conditioning."""

from __future__ import annotations

import pytest

from scripts.eval.chat_eval import prepare_model_for_chat_eval
from steering.eval.target_conditioning import SingleTargetSegment
from tests.conftest import build_tiny_decoder, build_tiny_encdec


def test_prepare_rejects_standalone_target_only_encdec():
    model = build_tiny_encdec()
    target = SingleTargetSegment(features={42: 3})
    with pytest.raises(ValueError, match="target-only"):
        prepare_model_for_chat_eval(model, target)


def test_prepare_returns_decoder_unchanged_for_context_only_target():
    """Decoder-only models stay unwrapped when there is no feature target."""
    model = build_tiny_decoder()
    prep = prepare_model_for_chat_eval(
        model,
        SingleTargetSegment(features={}),
    )
    assert prep is model


def test_prepare_rejects_features_for_decoder_only():
    model = build_tiny_decoder()
    with pytest.raises(ValueError, match="Decoder-only"):
        prepare_model_for_chat_eval(
            model,
            SingleTargetSegment(features={42: 3}),
        )
