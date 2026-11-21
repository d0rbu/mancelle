"""Assert checked-in microbench fixtures reflect the live context cache.

Skipped automatically if the live cache's parquet is absent (e.g. CI without
the extracted cache)."""

from __future__ import annotations

import statistics
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch

from scripts.profile_progression_microbench.encdec_build_conditioning import (
    _FIXTURE_CONFIG,
    _target_mode_for_checkpoint,
    config_hash,
)
from steering.eval.target_conditioning import TargetConditioningMode

LIVE_CACHE_PARQUET = Path(
    "out/data/eval_contexts/progression_profiling/parquet/shard_00000.parquet"
)


def _quantile(values, q):
    """Simple linear-interpolation quantile over sorted values."""
    v = sorted(float(x) for x in values)
    if not v:
        raise ValueError("empty")
    if len(v) == 1:
        return v[0]
    idx = q * (len(v) - 1)
    lo_i = int(idx)
    hi_i = min(lo_i + 1, len(v) - 1)
    frac = idx - lo_i
    return v[lo_i] * (1 - frac) + v[hi_i] * frac


@pytest.mark.skipif(
    not LIVE_CACHE_PARQUET.exists(),
    reason=f"live cache parquet {LIVE_CACHE_PARQUET} not present",
)
def test_encdec_build_conditioning_fixture_freshness():
    """Fixture distribution should remain consistent with the live cache it was built from."""
    tbl = pq.read_table(LIVE_CACHE_PARQUET)
    counts = [int(c) for c in tbl.column("token_count").to_pylist()]
    live_median = statistics.median(counts)
    live_p10 = _quantile(counts, 0.10)
    live_p90 = _quantile(counts, 0.90)

    fixture_path = Path("tests/fixtures/profile_progression/encdec_build_conditioning.pt")
    assert fixture_path.exists(), "fixture missing; run build_fixture"

    fixture = torch.load(fixture_path, map_location="cpu", weights_only=True)
    fixture_seq_lens = [int(t.shape[0]) for t in fixture["input_ids"]]
    fixture_median = statistics.median(fixture_seq_lens)

    # Fixture median within ±5% of live median
    tol = 0.05
    rel_delta = abs(fixture_median - live_median) / max(1, live_median)
    assert rel_delta <= tol, (
        f"fixture median {fixture_median} deviates from live median {live_median} "
        f"by {100 * rel_delta:.1f}% (tolerance {100 * tol:.0f}%)"
    )

    # The _FIXTURE_CONFIG seq_len_distribution must cover the live p10/p90 within ±10%
    dist_lo = min(_FIXTURE_CONFIG.seq_len_distribution)
    dist_hi = max(_FIXTURE_CONFIG.seq_len_distribution)
    assert dist_lo <= live_p10 * 1.10, (
        f"_FIXTURE_CONFIG min {dist_lo} > live_p10 * 1.10 ({live_p10 * 1.10:.0f})"
    )
    assert dist_hi >= live_p90 * 0.90, (
        f"_FIXTURE_CONFIG max {dist_hi} < live_p90 * 0.90 ({live_p90 * 0.90:.0f})"
    )

    # Distribution must be non-degenerate (≥5 samples)
    assert len(_FIXTURE_CONFIG.seq_len_distribution) >= 5, (
        "_FIXTURE_CONFIG seq_len_distribution must have at least 5 samples"
    )

    # Fixture must have at least one context row
    assert len(fixture_seq_lens) >= 1


def test_fixture_config_hash_is_deterministic():
    """config_hash must be stable across calls (no randomness)."""
    h1 = config_hash()
    h2 = config_hash()
    assert h1 == h2 == "781bee0986c67f70"
    assert len(h1) == 16


@pytest.mark.parametrize(
    ("next_segment_offset", "expected"),
    [
        (False, TargetConditioningMode.CURRENT),
        (True, TargetConditioningMode.NEXT),
    ],
)
def test_microbench_uses_checkpoint_target_mode(
    next_segment_offset: bool,
    expected: TargetConditioningMode,
) -> None:
    meta = {
        "user_config": {
            "next_segment_conditioning_offset": next_segment_offset,
        }
    }
    assert _target_mode_for_checkpoint(meta) is expected
