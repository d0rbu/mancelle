"""On-box schema check for real post-#1228 direction artifacts."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from safetensors import safe_open

from steering.eval.axbench_base.direction_scale import parse_max_activation
from steering.eval.axbench_base.types import AxBenchBaseError

pytestmark = pytest.mark.requires_feature_data


def _direction_artifacts() -> list[Path]:
    raw = os.environ.get("AXBENCH_DIRECTION_RUN_DIR")
    if raw is None:
        pytest.skip("set AXBENCH_DIRECTION_RUN_DIR to a real post-#1228 Stage-3 run")
    run_dir = Path(raw)
    if not run_dir.is_dir():
        pytest.skip(f"AXBENCH_DIRECTION_RUN_DIR is absent: {run_dir}")
    artifacts = sorted(
        path
        for family in ("diffmean", "linear_probe")
        for path in (run_dir / "baselines" / family).rglob("*.safetensors")
    )
    if not artifacts:
        pytest.skip(f"no direction artifacts found under {run_dir / 'baselines'}")
    return artifacts


def test_real_direction_artifacts_carry_valid_max_activation() -> None:
    """Every real Stage-3 direction binds a valid direction-owned scale.

    Ratio-band calibration remains the on-box A2 acceptance gate; this portable
    check only enforces the artifact schema and zero-direction consistency.
    """
    offenders: list[str] = []
    for path in _direction_artifacts():
        try:
            with safe_open(str(path), framework="pt", device="cpu") as handle:
                metadata = handle.metadata()
                direction = handle.get_tensor("direction")
            value = parse_max_activation(metadata, surface=str(path))
            direction_is_zero = bool((direction == 0).all().item())
            if value < 0.0 or (value == 0.0) != direction_is_zero:
                offenders.append(
                    f"{path}: max_activation={value}, direction_zero={direction_is_zero}"
                )
        except (AxBenchBaseError, KeyError, OSError) as exc:
            offenders.append(f"{path}: {exc}")

    assert not offenders, "invalid direction artifact(s):\n" + "\n".join(offenders)
