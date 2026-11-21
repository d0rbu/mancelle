"""Decoder attention benchmark routing and CLI contracts."""

from __future__ import annotations

import json
from contextlib import nullcontext
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t

import nanochat.flash_attention as fa_module
from nanochat.flash_attention import FlashAttnImpl
from scripts.benchmark import benchmark_attention


@pytest.fixture(autouse=True)
def _restore_attention_override() -> Any:
    original = fa_module._override_impl
    try:
        yield
    finally:
        fa_module._override_impl = original


def test_parse_args_accepts_explicit_flex_alias_and_preserves_hydra_overrides() -> None:
    impl, overrides = benchmark_attention._parse_args(
        ["impl=flex", "depth=12", "device_type=cuda"]
    )

    assert impl == "flex"
    assert overrides == ["depth=12", "device_type=cuda"]


def test_select_flex_uses_compiled_flex_route_without_claiming_fa4(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fa_module, "HAS_FLEX_ATTENTION", True)
    monkeypatch.setattr(fa_module, "HAS_FA4_BACKEND", False)

    benchmark_attention._select_backend("flex")

    assert fa_module._override_impl is FlashAttnImpl.FA4
    assert benchmark_attention._backend_routes() == {
        "configured": "fa4",
        "non_cached": "flex_triton",
        "kv_cache": "sdpa" if not fa_module.HAS_FA3 else "fa3",
    }


def test_select_fa4_rejects_flex_triton_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fa_module, "HAS_FLEX_ATTENTION", True)
    monkeypatch.setattr(fa_module, "HAS_FA4_BACKEND", False)

    with pytest.raises(RuntimeError, match="FA4 FLASH backend is not available"):
        benchmark_attention._select_backend("fa4")


def test_select_flex_rejects_missing_flex_attention(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(fa_module, "HAS_FLEX_ATTENTION", False)

    with pytest.raises(RuntimeError, match="FlexAttention is not available"):
        benchmark_attention._select_backend("flex")


def test_select_fa3_rejects_missing_backend(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(fa_module, "HAS_FA3", False)

    with pytest.raises(RuntimeError, match="FA3 is not available"):
        benchmark_attention._select_backend("fa3")


@pytest.mark.parametrize(
    (
        "has_flex",
        "has_fa4",
        "has_fa3",
        "override",
        "expected",
    ),
    [
        (
            True,
            True,
            True,
            None,
            {"configured": "fa4", "non_cached": "flex_flash", "kv_cache": "fa3"},
        ),
        (
            True,
            False,
            False,
            None,
            {"configured": "fa4", "non_cached": "flex_triton", "kv_cache": "sdpa"},
        ),
        (
            False,
            False,
            True,
            None,
            {"configured": "fa3", "non_cached": "fa3", "kv_cache": "fa3"},
        ),
        (
            True,
            False,
            True,
            None,
            {"configured": "fa3", "non_cached": "sdpa", "kv_cache": "fa3"},
        ),
        (
            True,
            False,
            True,
            FlashAttnImpl.SDPA,
            {"configured": "sdpa", "non_cached": "sdpa", "kv_cache": "sdpa"},
        ),
        (
            True,
            False,
            True,
            FlashAttnImpl.FA4,
            {"configured": "fa4", "non_cached": "flex_triton", "kv_cache": "fa3"},
        ),
    ],
)
def test_backend_routes_report_executed_branches_not_only_configured_label(
    monkeypatch: pytest.MonkeyPatch,
    has_flex: bool,
    has_fa4: bool,
    has_fa3: bool,
    override: FlashAttnImpl | None,
    expected: dict[str, str],
) -> None:
    monkeypatch.setattr(fa_module, "HAS_FLEX_ATTENTION", has_flex)
    monkeypatch.setattr(fa_module, "HAS_FA4_BACKEND", has_fa4)
    monkeypatch.setattr(fa_module, "HAS_FA3", has_fa3)
    fa_module._override_impl = override

    assert benchmark_attention._backend_routes() == expected


def test_main_writes_the_executed_decoder_routes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    class _Tokenizer:
        def get_vocab_size(self) -> int:
            return 32

    class _Model(t.nn.Module):
        def __init__(self, _config) -> None:
            super().__init__()
            self.weight = t.nn.Parameter(t.ones(()))

        def init_weights(self) -> None:
            with t.no_grad():
                self.weight.fill_(1.0)

        def forward(self, _x, _y):
            return self.weight.square()

    def _loader(*_args, **_kwargs):
        x = t.zeros((1, 2), dtype=t.long)
        while True:
            yield x, x, None

    cleaned: list[bool] = []
    cfg = SimpleNamespace(device_type="cpu", aspect_ratio=64, head_dim=8)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        benchmark_attention,
        "initialize_config_dir",
        lambda **_kwargs: nullcontext(),
    )
    monkeypatch.setattr(benchmark_attention, "compose", lambda **_kwargs: cfg)
    monkeypatch.setattr(
        benchmark_attention,
        "compute_init",
        lambda _device_type: (False, 0, 0, 1, t.device("cpu")),
    )
    monkeypatch.setattr(
        benchmark_attention,
        "compute_cleanup",
        lambda: cleaned.append(True),
    )
    monkeypatch.setattr(benchmark_attention, "get_tokenizer", _Tokenizer)
    monkeypatch.setattr(
        benchmark_attention,
        "resolve_gpt_dimensions",
        lambda **_kwargs: (16, 2),
    )
    monkeypatch.setattr(benchmark_attention, "GPT", _Model)
    monkeypatch.setattr(benchmark_attention.torch, "compile", lambda model, **_kwargs: model)
    monkeypatch.setattr(
        benchmark_attention,
        "tokenizing_distributed_data_loader_with_state_bos_bestfit",
        _loader,
    )
    monkeypatch.setattr(benchmark_attention, "NUM_STEPS", 11)
    monkeypatch.setattr(benchmark_attention, "BENCH_DEPTH", 1)
    monkeypatch.setattr(benchmark_attention, "BENCH_SEQ_LEN", 2)
    monkeypatch.setattr(benchmark_attention, "BENCH_BATCH_SIZE", 1)

    benchmark_attention.main(["impl=sdpa"])

    payload = json.loads((tmp_path / "benchmark_sdpa.json").read_text(encoding="utf-8"))
    assert payload["impl"] == "sdpa"
    assert payload["requested_impl"] == "sdpa"
    assert payload["attention_routes"] == {
        "configured": "sdpa",
        "non_cached": "sdpa",
        "kv_cache": "sdpa",
    }
    assert cleaned == [True]
