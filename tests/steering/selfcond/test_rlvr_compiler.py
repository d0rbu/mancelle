"""Torch compiler-budget contract for canonical RLVR prefix replay."""

from __future__ import annotations

from contextvars import ContextVar
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t
from torch._dynamo.exc import FailOnRecompileLimitHit

import steering.conditional.encoder_decoder.rlvr_policy as rlvr_policy
from steering.conditional.encoder_decoder.rlvr_runtime import (
    configure_rlvr_compiler,
)
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError

_CONFIG_KEYS = (
    "recompile_limit",
    "accumulated_recompile_limit",
    "fail_on_recompile_limit_hit",
    "suppress_errors",
)


def _set_config(name: str, value: int | bool) -> None:
    setattr(t._dynamo.config, name, value)


@pytest.fixture(autouse=True)
def _restore_dynamo_config():
    original = {key: getattr(t._dynamo.config, key) for key in _CONFIG_KEYS}
    t._dynamo.reset()
    try:
        yield
    finally:
        t._dynamo.reset()
        for key, value in original.items():
            _set_config(key, value)


def _model(*, sequence_len: int = 11):
    return SimpleNamespace(
        dec_config=SimpleNamespace(sequence_len=sequence_len),
        window_sizes=[(3, 0), (7, 0), (3, 0)],
    )


def _checkpointed_test_logits(
    monkeypatch: pytest.MonkeyPatch,
    model: t.nn.Module,
    canonical: Any,
    *,
    train_feature_head_only: bool = False,
) -> t.Tensor:
    monkeypatch.setattr(rlvr_policy, "canonical_fp_logits", canonical)
    action = SimpleNamespace(
        committed_prefix=(),
        outgoing_seg_map=(),
        pool_positions=(),
        feature_history=(),
    )
    return rlvr_policy._checkpointed_canonical_fp_logits(
        model,
        cast(rlvr_policy.RLVRPromptReplay, SimpleNamespace()),
        cast(rlvr_policy.RLVRActionTrace, action),
        train_feature_head_only=train_feature_head_only,
    )


def test_configure_rlvr_compiler_is_derived_idempotent_and_preserves_larger_limits() -> None:
    _set_config("recompile_limit", 3)
    _set_config("accumulated_recompile_limit", 5)
    _set_config("fail_on_recompile_limit_hit", False)
    _set_config("suppress_errors", False)

    expected_budget = 1 + 2 * 2 * 11
    configured = configure_rlvr_compiler(_model())
    assert configured == {
        "derived_minimum": expected_budget,
        "window_variants": 2,
        "sequence_len": 11,
        "recompile_limit": expected_budget,
        "accumulated_recompile_limit": expected_budget,
        "fail_on_recompile_limit_hit": True,
        "suppress_errors": False,
    }
    assert configure_rlvr_compiler(_model()) == configured

    _set_config("recompile_limit", expected_budget + 7)
    _set_config("accumulated_recompile_limit", expected_budget + 9)
    larger = configure_rlvr_compiler(_model())
    assert larger["recompile_limit"] == expected_budget + 7
    assert larger["accumulated_recompile_limit"] == expected_budget + 9
    assert larger["derived_minimum"] == expected_budget


def test_configure_rlvr_compiler_rejects_suppressed_compile_failures_atomically() -> None:
    _set_config("recompile_limit", 8)
    _set_config("accumulated_recompile_limit", 256)
    _set_config("fail_on_recompile_limit_hit", False)
    _set_config("suppress_errors", True)

    with pytest.raises(SFTGRPOError, match="suppress_errors=False"):
        configure_rlvr_compiler(_model())
    assert t._dynamo.config.recompile_limit == 8
    assert t._dynamo.config.accumulated_recompile_limit == 256
    assert t._dynamo.config.fail_on_recompile_limit_hit is False


def test_real_torch_compile_limit_exhaustion_raises_instead_of_falling_back() -> None:
    _set_config("recompile_limit", 1)
    _set_config("accumulated_recompile_limit", 1)
    _set_config("fail_on_recompile_limit_hit", False)
    _set_config("suppress_errors", False)

    @t.compile(dynamic=False, backend="eager")
    def compiled(value: t.Tensor) -> t.Tensor:
        return value.sin() + 1

    # The startup compatibility probe compiles before RLVR enables fail-on-limit.
    assert compiled(t.ones(1)).shape == (1,)
    _set_config("fail_on_recompile_limit_hit", True)
    with pytest.raises(
        FailOnRecompileLimitHit,
        match="fail_on_recompile_limit_hit",
    ):
        compiled(t.ones(2))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required"),
        ),
    ],
)
def test_checkpoint_recompute_preserves_rlvr_compiler_configuration(
    monkeypatch: pytest.MonkeyPatch,
    device: str,
) -> None:
    configured = configure_rlvr_compiler(_model(sequence_len=2048))
    assert configured["recompile_limit"] == 8193
    assert configured["fail_on_recompile_limit_hit"] is True
    marker = ContextVar("rlvr_checkpoint_test_marker", default="default")
    marker.set("caller")
    seen: list[tuple[int, int, bool, str]] = []
    model = t.nn.Linear(1, 2, bias=False, device=device)

    def canonical(model: t.nn.Module, *_args, **_kwargs) -> t.Tensor:
        seen.append(
            (
                t._dynamo.config.recompile_limit,
                t._dynamo.config.accumulated_recompile_limit,
                t._dynamo.config.fail_on_recompile_limit_hit,
                marker.get(),
            )
        )
        marker.set("replay")
        return t.square(cast(t.Tensor, model.weight)).flatten()

    logits = _checkpointed_test_logits(monkeypatch, model, canonical)
    logits.sum().backward()
    assert seen == [
        (8193, 8193, True, "caller"),
        (8193, 8193, True, "caller"),
    ]
    assert marker.get() == "caller"
    t.testing.assert_close(model.weight.grad, 2 * model.weight.detach())


def test_head_only_replay_uses_caller_compiler_configuration_without_checkpoint(monkeypatch):
    configured = configure_rlvr_compiler(_model(sequence_len=2048))
    model = t.nn.Linear(1, 2, bias=False)
    seen = []

    def canonical(model, *_args, **kwargs):
        seen.append(
            (t.is_grad_enabled(), t._dynamo.config.recompile_limit, kwargs["train_feature_head_only"])
        )
        return model.weight.square().flatten()

    def unexpected_checkpoint(*_args, **_kwargs):
        pytest.fail("head-only replay must not checkpoint the frozen trunk")

    monkeypatch.setattr(rlvr_policy, "checkpoint", unexpected_checkpoint)
    logits = _checkpointed_test_logits(monkeypatch, model, canonical, train_feature_head_only=True)
    logits.sum().backward()
    assert seen == [(True, configured["recompile_limit"], True)]
    t.testing.assert_close(model.weight.grad, 2 * model.weight.detach(), rtol=0, atol=0)


@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_checkpoint_recompute_raises_on_compiler_limit_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_config("recompile_limit", 1)
    _set_config("accumulated_recompile_limit", 1)
    _set_config("fail_on_recompile_limit_hit", True)
    _set_config("suppress_errors", False)

    @t.compile(dynamic=False, backend="eager")
    def compiled(value: t.Tensor) -> t.Tensor:
        return value.sin() + 1

    assert compiled(t.ones(1, device="cuda")).shape == (1,)
    model = t.nn.Linear(1, 2, bias=False, device="cuda")

    def canonical(model: t.nn.Module, *_args, **_kwargs) -> t.Tensor:
        if t.is_grad_enabled():
            compiled(t.ones(2, device="cuda"))
        return t.square(cast(t.Tensor, model.weight)).flatten()

    logits = _checkpointed_test_logits(monkeypatch, model, canonical)
    with pytest.raises(FailOnRecompileLimitHit, match="fail_on_recompile_limit_hit"):
        logits.sum().backward()
