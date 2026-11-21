"""Device-placement lifetime tests for alternating DiffMean and feature-GCG."""

from __future__ import annotations

from typing import Any

import pytest
import torch as t

import steering.eval.axbench_bpb_adv.gcg_bags as gcg_bags


class _Movable:
    def __init__(self, name: str, events: list[tuple[str, str]]) -> None:
        self.name = name
        self.events = events

    def to(self, device: Any) -> None:
        self.events.append((self.name, str(device)))


@pytest.mark.parametrize(
    ("prepare_name", "expected"),
    (
        (
            "_prepare_for_diffmean_forward",
            [
                ("target", "cpu"),
                ("release", "cuda"),
                ("base", "cuda"),
                ("sae", "cuda"),
            ],
        ),
        (
            "_prepare_for_gcg_forward",
            [
                ("base", "cpu"),
                ("sae", "cpu"),
                ("release", "cuda"),
                ("target", "cuda"),
            ],
        ),
    ),
)
def test_model_swap_collects_before_upload(
    monkeypatch: pytest.MonkeyPatch,
    prepare_name: str,
    expected: list[tuple[str, str]],
) -> None:
    """The outgoing model is offloaded, cycles collected, then the peer uploads."""
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(gcg_bags.t.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        gcg_bags,
        "release_cuda_iteration_memory",
        lambda: events.append(("release", "cuda")),
    )
    prepare = getattr(gcg_bags, prepare_name)

    prepare(
        model=_Movable("target", events),
        base_model=_Movable("base", events),
        sae=_Movable("sae", events),
        device=t.device("cuda"),
    )

    assert events == expected


def test_model_swap_is_exact_noop_on_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, str]] = []
    monkeypatch.setattr(
        gcg_bags,
        "release_cuda_iteration_memory",
        lambda: events.append(("release", "cuda")),
    )

    for prepare in (
        gcg_bags._prepare_for_diffmean_forward,
        gcg_bags._prepare_for_gcg_forward,
    ):
        prepare(
            model=_Movable("target", events),
            base_model=_Movable("base", events),
            sae=_Movable("sae", events),
            device=t.device("cpu"),
        )

    assert events == []
