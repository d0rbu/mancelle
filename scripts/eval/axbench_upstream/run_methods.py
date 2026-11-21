"""Torchrun-compatible entry point for the selected project AxBench methods."""

from __future__ import annotations

import sys
from pathlib import Path

import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig

from nanochat.common import compute_cleanup, compute_init, is_ddp_requested
from steering._common.hydra_cli import parse_config_name_and_overrides
from steering.eval.axbench_upstream.run_methods import run_project_methods_from_dictconfig

_CONFIG_DIR = Path(__file__).resolve().parents[3] / "configs" / "eval" / "axbench_upstream"


def _device_type(device: object) -> str:
    if not isinstance(device, str) or not device:
        raise ValueError("AxBench project-method device must be a nonempty string")
    device_type = device.split(":", 1)[0]
    if device_type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"unsupported AxBench project-method device type: {device_type!r}")
    return device_type


def _run_config(cfg: DictConfig) -> None:
    """Preserve the historical Hydra-decorated test seam."""

    run_project_methods_from_dictconfig(cfg)


def main(argv: list[str] | None = None) -> None:
    """Compose config, initialize optional torchrun state, and execute."""

    config_name, overrides = parse_config_name_and_overrides(
        sys.argv[1:] if argv is None else argv,
        default_config_name="methods",
    )
    with initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base=None):
        cfg = compose(config_name=config_name, overrides=overrides)
    configured_device = cfg.get("device")
    device_type = _device_type(configured_device)
    if not is_ddp_requested():
        _run_config(cfg)
        return

    matmul_precision = t.get_float32_matmul_precision()
    try:
        try:
            ddp, _rank, _local_rank, _world_size, _device = compute_init(device_type)
        finally:
            # compute_init selects TF32 for training. The historical AxBench
            # producer did not, and its exact serial oracle was generated with
            # the caller's preexisting mode. Keep process-group setup without
            # changing producer math.
            t.set_float32_matmul_precision(matmul_precision)
    except BaseException:
        # compute_init can fail after process-group creation (for example at its
        # barrier), so its idempotent companion must inspect and clean partial
        # state before the original failure propagates.
        compute_cleanup()
        raise
    try:
        if ddp and configured_device != "cuda":
            raise ValueError("torchrun AxBench project methods require device=cuda")
        _run_config(cfg)
    finally:
        compute_cleanup()


main.__wrapped__ = _run_config  # ty: ignore[unresolved-attribute]


if __name__ == "__main__":
    main()
