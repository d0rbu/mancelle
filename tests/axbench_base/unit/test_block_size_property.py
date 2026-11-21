"""Property-based invariants for stage0c _block_size (spec G-coverage).

Pins the contract against future refactors without overlapping the
explicit table in test_stage0c_runner.py::test_block_size_table.
"""

from __future__ import annotations

import math
from pathlib import Path

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.stage0c_runner import _block_size


def _cfg(tmp: Path, *, max_concepts: int, block_size: int) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir=str(tmp / "f"),
        feature_statistics_path=str(tmp / "s.safetensors"),
        decoder_only={"source": "base", "model_tag": "d13", "step": 1},
        encdec={"source": "encdec", "model_tag": "d13", "step": 1},
        device="cpu",
        output_root=str(tmp / "o"),
        run_tag="prop",
        max_concepts=max_concepts,
        stage0c_multi_gpu_block_size=block_size,
    )


# tmp_path reused across examples; safe — _cfg builds path strings only, no I/O.
@settings(max_examples=300, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture], derandomize=True)
@given(
    configured=st.integers(min_value=1, max_value=256),
    max_concepts=st.integers(min_value=1, max_value=2000),
    world_size=st.integers(min_value=1, max_value=128),
)
def test_block_size_invariants(tmp_path, configured, max_concepts, world_size):
    b = _block_size(_cfg(tmp_path, max_concepts=max_concepts, block_size=configured), world_size)
    assert 1 <= b <= configured
    if world_size <= 1:
        assert b == configured
    else:
        cap = math.ceil(max_concepts * 2 / world_size)
        assert b == max(1, min(configured, cap))


# tmp_path reused across examples; safe — _cfg builds path strings only, no I/O.
@settings(max_examples=100, deadline=None, suppress_health_check=[HealthCheck.function_scoped_fixture], derandomize=True)
@given(
    configured=st.integers(min_value=-5, max_value=256),
    max_concepts=st.integers(min_value=-5, max_value=2000),
    world_size=st.integers(min_value=0, max_value=128),
)
def test_block_size_never_below_one_for_any_input(tmp_path, configured, max_concepts, world_size):
    """D1: even pathological (<=0) inputs never yield B < 1."""
    assert _block_size(
        _cfg(tmp_path, max_concepts=max_concepts, block_size=configured),
        world_size,
    ) >= 1


def test_block_size_ceil_exhaustive_tie_points(tmp_path):
    """Deterministic exhaustive check of the integer-ceil cap at every
    (world_size, max_concepts) residue. configured is set huge so the cap
    always binds, isolating the ``ceil(2*max_concepts/world_size)``
    arithmetic — including exact tie residues 2*mc % ws in {0,1,ws-1}.
    Complements the randomized property test with a reproducible matrix.
    """
    import math
    for ws in range(2, 65):              # ws<=1 is the passthrough branch
        for mc in range(1, 65):
            cfg = _cfg(tmp_path, max_concepts=mc, block_size=10**6)
            assert _block_size(cfg, ws) == max(1, math.ceil(2 * mc / ws))
