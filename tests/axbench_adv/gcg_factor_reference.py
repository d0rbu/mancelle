"""Independent loss oracle taking a coefficient derived from source calibration."""

from __future__ import annotations

import hashlib
from types import ModuleType
from typing import Any

import torch as t


def capture_search_progress(driver: ModuleType, monkeypatch: Any) -> list[dict[str, Any]]:
    original = driver.run_gcg
    events: list[dict[str, Any]] = []

    def run(**kwargs: Any) -> Any:
        return original(**kwargs, progress_callback=lambda event: events.append(dict(event)))

    monkeypatch.setattr(driver, "run_gcg", run)
    return events


def tensor_digest(value: t.Tensor) -> str:
    return hashlib.sha256(
        value.detach().cpu().contiguous().view(t.uint8).numpy().tobytes()
    ).hexdigest()


def l2_with_expected_offset(
    attacked: t.Tensor, clean: t.Tensor, observed: t.Tensor, *, expected: t.Tensor
) -> t.Tensor:
    assert observed.dtype == expected.dtype
    assert t.equal(observed, expected.to(device=observed.device))
    return t.linalg.vector_norm(
        attacked.float()
        + expected.to(device=attacked.device, dtype=t.float32)
        - clean.detach().to(device=attacked.device, dtype=t.float32),
        dim=-1,
    ).mean()
