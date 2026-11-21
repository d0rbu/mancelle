"""Slice 16 audit — pin ``_broadcast_resume_decision`` contract.

Map P0-2 / P3-12 (manifest §7 Slice 16) reclassed this duplicate
implementation between ``negative_prefixes._broadcast_resume_decision``
and ``scripts/eval/axbench_base_eval.py:541-555`` as P1. The fix
delegates the CLI to the helper. This test pins:

1. The single-rank short-circuit returns the input unchanged.
2. The multi-rank pre-init short-circuit returns the input unchanged.
3. The CLI no longer calls ``dist.broadcast_object_list`` directly
   (i.e., does not re-implement the helper) — verified by
   importing the CLI module and asserting it now imports the
   helper symbol.
"""

from __future__ import annotations

from pathlib import Path

import pytest


def test_single_rank_returns_input_unchanged() -> None:
    """world_size <= 1: helper short-circuits and returns the input verbatim."""
    from steering.eval.axbench_base.negative_prefixes import (
        _broadcast_resume_decision,
    )

    p = Path("/tmp/some-resume.json")
    out = _broadcast_resume_decision(p, rank=0, world_size=1)
    assert out is p

    out_none = _broadcast_resume_decision(None, rank=0, world_size=1)
    assert out_none is None


def test_single_rank_propagates_rank0_resume_error() -> None:
    """Fail-loud provenance errors must surface before model loading."""
    from steering.eval.axbench_base.negative_prefixes import (
        _broadcast_resume_decision,
    )
    from steering.eval.axbench_base.types import AxBenchBaseError

    with pytest.raises(AxBenchBaseError, match="stale extractor identity"):
        _broadcast_resume_decision(
            None,
            rank=0,
            world_size=1,
            resume_error="stale extractor identity",
        )


def test_nonzero_rank_receives_rank0_resume_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The error decision is broadcast so no rank enters Stage 0c alone."""
    from steering.eval.axbench_base import negative_prefixes
    from steering.eval.axbench_base.types import AxBenchBaseError

    monkeypatch.setattr(negative_prefixes.dist, "is_available", lambda: True)
    monkeypatch.setattr(negative_prefixes.dist, "is_initialized", lambda: True)

    def _fake_broadcast(payload, src):
        assert src == 0
        payload[0] = {
            "resume_path": None,
            "resume_error": "rank-zero identity mismatch",
        }

    monkeypatch.setattr(
        negative_prefixes.dist,
        "broadcast_object_list",
        _fake_broadcast,
    )

    with pytest.raises(AxBenchBaseError, match="rank-zero identity mismatch"):
        negative_prefixes._broadcast_resume_decision(
            None,
            rank=1,
            world_size=2,
        )


def test_multi_rank_pre_init_returns_input_unchanged() -> None:
    """world_size > 1 but PG not initialized: helper does not call dist."""
    import torch.distributed as dist

    from steering.eval.axbench_base.negative_prefixes import (
        _broadcast_resume_decision,
    )

    # In a CPU test process, ``dist.is_initialized()`` is False, so the
    # helper falls back to single-rank semantics.
    assert not dist.is_initialized()
    out = _broadcast_resume_decision(
        Path("/tmp/r"), rank=0, world_size=2,
    )
    assert out == Path("/tmp/r")


def test_cli_imports_helper_and_does_not_reimplement_broadcast() -> None:
    """Audit fix: CLI delegates to ``_broadcast_resume_decision``.

    Pins the slice-16 fix so a future refactor that drops the helper
    delegation in the CLI is caught immediately. We assert two
    properties of the CLI module's source:

    - It imports ``_broadcast_resume_decision`` from
      ``negative_prefixes``.
    - It does NOT call ``dist.broadcast_object_list`` directly inside
      the Stage 0c block (the dedicated helper handles that
      precisely-once).
    """
    import inspect

    import scripts.eval.axbench_base_eval as cli

    # The helper must be present in the CLI's import surface.
    assert hasattr(cli, "_broadcast_resume_decision"), (
        "CLI must import _broadcast_resume_decision from negative_prefixes"
    )

    # And the Stage 0c dispatch MUST NOT call
    # ``dist.broadcast_object_list`` directly. Other lifecycle helpers in the
    # CLI have separate broadcast contracts.
    source = inspect.getsource(cli._run_single_stage)
    assert "dist.broadcast_object_list" not in source, (
        "CLI must delegate to _broadcast_resume_decision rather than "
        "re-implementing the broadcast"
    )


def test_helper_payload_shape_via_signature() -> None:
    """Pin the helper's keyword-only signature so consumers can rely on it.

    The CLI passes ``rank=`` / ``world_size=`` by keyword. A future
    change that promotes them to positional would silently let bad
    callers slip through; this test pins the kw-only contract.
    """
    import inspect

    from steering.eval.axbench_base.negative_prefixes import (
        _broadcast_resume_decision,
    )

    sig = inspect.signature(_broadcast_resume_decision)
    params = sig.parameters
    assert "rank" in params
    assert "world_size" in params
    assert params["rank"].kind is inspect.Parameter.KEYWORD_ONLY
    assert params["world_size"].kind is inspect.Parameter.KEYWORD_ONLY


@pytest.mark.parametrize("world_size", [1, 2, 4, 8])
def test_helper_handles_world_sizes_in_pre_init_mode(world_size: int) -> None:
    """No matter the world size, pre-init falls back to identity passthrough."""
    from steering.eval.axbench_base.negative_prefixes import (
        _broadcast_resume_decision,
    )

    out = _broadcast_resume_decision(
        Path("/x.json"), rank=0, world_size=world_size,
    )
    assert out == Path("/x.json")
