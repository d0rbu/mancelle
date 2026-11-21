"""Torchrun/Hydra ingress tests for both root-environment producers."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.eval.axbench_upstream import run_encdec_baselines as encdec_entry
from scripts.eval.axbench_upstream import run_methods as project_entry


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
@pytest.mark.parametrize(
    ("configured", "expected"),
    [("cpu", "cpu"), ("cuda", "cuda"), ("cuda:7", "cuda"), ("mps", "mps")],
)
def test_device_type_accepts_supported_device_spellings(
    module: ModuleType,
    configured: str,
    expected: str,
) -> None:
    assert module._device_type(configured) == expected  # type: ignore[attr-defined]


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
@pytest.mark.parametrize("configured", [None, "", 1, True, [], {}])
def test_device_type_rejects_nonempty_string_contract(
    module: ModuleType,
    configured: object,
) -> None:
    with pytest.raises(ValueError, match="device must be a nonempty string"):
        module._device_type(configured)  # type: ignore[attr-defined]


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
@pytest.mark.parametrize("configured", ["xpu", "CUDA", "gpu:0", "cudaa"])
def test_device_type_rejects_unknown_backend(module: ModuleType, configured: str) -> None:
    with pytest.raises(ValueError, match="unsupported AxBench"):
        module._device_type(configured)  # type: ignore[attr-defined]


def _runner_name(module: ModuleType) -> str:
    return (
        "run_project_methods_from_dictconfig"
        if module is project_entry
        else "run_encdec_baselines_from_dictconfig"
    )


def _install_ingress_fakes(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    *,
    configured_device: str = "cuda",
    ddp: bool = False,
    runner_error: BaseException | None = None,
) -> tuple[list[Any], list[Any]]:
    events: list[Any] = []
    compose_calls: list[Any] = []

    @contextmanager
    def initialize(*, config_dir: str, version_base: object) -> Iterator[None]:
        events.append(("initialize-enter", config_dir, version_base))
        yield
        events.append("initialize-exit")

    def compose(*, config_name: str, overrides: list[str]) -> Any:
        compose_calls.append((config_name, overrides))
        events.append("compose")
        return OmegaConf.create({"device": configured_device, "seed": 42})

    def compute_init(device_type: str) -> tuple[bool, int, int, int, str]:
        events.append(("compute-init", device_type))
        return ddp, 0, 0, 2 if ddp else 1, configured_device

    def run(cfg: Any) -> None:
        events.append(("run", OmegaConf.to_container(cfg, resolve=True)))
        if runner_error is not None:
            raise runner_error

    monkeypatch.setattr(module, "initialize_config_dir", initialize)
    monkeypatch.setattr(module, "compose", compose)
    monkeypatch.setattr(module, "is_ddp_requested", lambda: ddp)
    monkeypatch.setattr(module, "compute_init", compute_init)
    monkeypatch.setattr(module, "compute_cleanup", lambda: events.append("cleanup"))
    monkeypatch.setattr(module, _runner_name(module), run)
    return events, compose_calls


@pytest.mark.parametrize(
    ("module", "default_name"),
    [(project_entry, "methods"), (encdec_entry, "encdec_baselines")],
)
def test_direct_main_composes_default_config_without_training_initialization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    default_name: str,
) -> None:
    del tmp_path
    events, compose_calls = _install_ingress_fakes(monkeypatch, module)

    module.main(["seed=91", "+extra=value", "--local-rank=0", "ignored"])

    assert compose_calls == [(default_name, ["seed=91", "+extra=value"])]
    assert events == [
        ("initialize-enter", str(module._CONFIG_DIR), None),  # type: ignore[attr-defined]
        "compose",
        "initialize-exit",
        ("run", {"device": "cuda", "seed": 42}),
    ]


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
def test_main_last_config_name_wins_and_is_not_forwarded(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    _events, compose_calls = _install_ingress_fakes(monkeypatch, module)

    module.main(["config_name=first", "seed=7", "config_name=second", "~obsolete"])

    assert compose_calls == [("second", ["seed=7", "~obsolete"])]


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
def test_main_always_cleans_up_after_runner_exception(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    error = RuntimeError("runner failed")
    events, _calls = _install_ingress_fakes(
        monkeypatch,
        module,
        ddp=True,
        runner_error=error,
    )

    with pytest.raises(RuntimeError, match="runner failed") as caught:
        module.main([])

    assert caught.value is error
    assert events[-1] == "cleanup"


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
def test_main_always_cleans_up_after_base_exception(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    events, _calls = _install_ingress_fakes(
        monkeypatch,
        module,
        ddp=True,
        runner_error=KeyboardInterrupt(),
    )

    with pytest.raises(KeyboardInterrupt):
        module.main([])

    assert events[-1] == "cleanup"


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
def test_compute_init_failure_restores_precision_and_cleans_partial_state(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    events, _calls = _install_ingress_fakes(monkeypatch, module, ddp=True)

    original_precision = t.get_float32_matmul_precision()

    def fail_init(_device_type: str) -> Any:
        events.append("init-failed")
        t.set_float32_matmul_precision("high")
        raise RuntimeError("init failed")

    monkeypatch.setattr(module, "compute_init", fail_init)

    try:
        t.set_float32_matmul_precision("highest")
        with pytest.raises(RuntimeError, match="init failed"):
            module.main([])
        assert t.get_float32_matmul_precision() == "highest"
    finally:
        t.set_float32_matmul_precision(original_precision)

    assert events[-2:] == ["init-failed", "cleanup"]


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
def test_torchrun_restores_matmul_precision_before_driver(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    events, _calls = _install_ingress_fakes(monkeypatch, module, ddp=True)
    observed: list[str] = []

    def compute_init(device_type: str) -> tuple[bool, int, int, int, str]:
        events.append(("compute-init", device_type))
        t.set_float32_matmul_precision("high")
        return True, 0, 0, 2, "cuda"

    def run(_cfg: Any) -> None:
        observed.append(t.get_float32_matmul_precision())

    monkeypatch.setattr(module, "compute_init", compute_init)
    monkeypatch.setattr(module, _runner_name(module), run)
    original_precision = t.get_float32_matmul_precision()
    try:
        t.set_float32_matmul_precision("highest")
        module.main([])
        assert observed == ["highest"]
        assert t.get_float32_matmul_precision() == "highest"
    finally:
        t.set_float32_matmul_precision(original_precision)

    assert events[-1] == "cleanup"


@pytest.mark.parametrize(
    ("module", "message"),
    [
        (project_entry, "torchrun AxBench project methods require device=cuda"),
        (
            encdec_entry,
            "torchrun AxBench encoder-decoder baselines require device=cuda",
        ),
    ],
)
@pytest.mark.parametrize("configured_device", ["cpu", "mps", "cuda:1"])
def test_torchrun_rejects_noncanonical_device_before_driver(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
    message: str,
    configured_device: str,
) -> None:
    events, _calls = _install_ingress_fakes(
        monkeypatch,
        module,
        configured_device=configured_device,
        ddp=True,
    )

    with pytest.raises(ValueError, match=message):
        module.main([])

    assert not any(isinstance(event, tuple) and event[0] == "run" for event in events)
    assert events[-1] == "cleanup"


@pytest.mark.parametrize("module", [project_entry, encdec_entry])
def test_torchrun_accepts_canonical_cuda_device(
    monkeypatch: pytest.MonkeyPatch,
    module: ModuleType,
) -> None:
    events, _calls = _install_ingress_fakes(
        monkeypatch,
        module,
        configured_device="cuda",
        ddp=True,
    )

    module.main([])

    assert any(isinstance(event, tuple) and event[0] == "run" for event in events)
    assert events[-1] == "cleanup"


def test_real_project_default_config_is_composable() -> None:
    with project_entry.initialize_config_dir(
        config_dir=str(project_entry._CONFIG_DIR),
        version_base=None,
    ):
        cfg = project_entry.compose(config_name="methods")

    assert cfg.device == "cuda"
    assert cfg.max_concepts == 2


def test_real_encdec_default_config_is_composable() -> None:
    with encdec_entry.initialize_config_dir(
        config_dir=str(encdec_entry._CONFIG_DIR),
        version_base=None,
    ):
        cfg = encdec_entry.compose(config_name="encdec_baselines")

    assert cfg.device == "cuda"
    assert cfg.max_concepts == 2
