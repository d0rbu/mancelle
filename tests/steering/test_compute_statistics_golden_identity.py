"""Output-identity guards for issue #563's compute_statistics refactor.

The #563 fix is a pure behavior-preserving re-indent (binding the streaming sort
temp dir's lifetime to one ``contextlib.ExitStack``). These tests pin that the
refactor changed NO computed output, two ways:

1. Characterization golden — the EXACT serialized output captured from the
   PRE-change (origin/main) implementation, committed alongside the exact fixture
   shard bytes (so ``storage.total_bytes`` is reproducible despite
   ``save_features`` being byte-non-deterministic across writes — decoded tensors
   are stable, file sizes are not). Post-change output must match it exactly.
   The comparison is a LITERAL string equality of the canonical JSON
   (``sort_keys=False``), so it is sensitive to value, scalar type (int vs
   float), and ORDER (e.g. the count-descending ``per_feature_stats`` ordering) —
   stricter than a dict ``==``, which would mask a reorder or an int/float drift.
2. Determinism — serializing a config twice yields the identical string, the
   property that makes the golden / the one-time real-data A/B valid.

``metadata.generated_at`` (wall-clock) and
``extrapolation_10b.estimated_wall_time_hours`` (``time.monotonic``-derived) are
excluded: inherently runtime-dependent, unaffected by the refactor's logic.

Coverage: the six CPU config branches below — streaming + eager, the
``skip_per_feature_detail`` Pass-2-skip branch, the ``exclude_last_shard`` paths
(streaming + eager), and the eager ``activation_percentile`` filtering branch.
The GPU Pass-2 path (``vectorized_pass2_percentiles``) is not exercised here
(no CUDA in default CI); it is covered structurally (the whitespace-normalized
diff shows no value statement changed in any branch) and by the deferred
real-data GPU A/B.

On THIS small fixture streaming and eager outputs coincide exactly, so they share
``golden_default.json`` (and the two exclude_last variants share
``golden_exclude_last.json``); the shared-golden test still pins each config's
output AND the coincidence. The ~1e-4 cross-mode divergence is a large-corpus
property tested in ``test_compute_statistics.py::TestStreamingMatchesEager``.
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import pytest

from steering.compute_statistics import compute_statistics

_FIXTURE_DIR = Path(__file__).parent / "fixtures" / "issue563"

# Config name -> (compute_statistics kwargs, golden output-class key). The golden
# file is golden_<key>.json. Configs whose outputs coincide on this fixture share
# a golden — streaming == eager, and the two exclude_last variants — so each
# DISTINCT output is committed exactly once and a shared-golden test still pins
# the coincidence (a future divergence makes the paired config fail). Goldens are
# regenerated from the pristine pre-change code.
_CONFIGS: dict[str, tuple[dict, str]] = {
    "streaming": ({"streaming": True}, "default"),
    "eager": ({"streaming": False}, "default"),
    "streaming_skip_detail": (
        {"streaming": True, "skip_per_feature_detail": True},
        "skip_detail",
    ),
    "streaming_exclude_last": (
        {"streaming": True, "exclude_last_shard": True},
        "exclude_last",
    ),
    "eager_exclude_last": (
        {"streaming": False, "exclude_last_shard": True},
        "exclude_last",
    ),
    "eager_actpct50": ({"streaming": False, "activation_percentile": 0.5}, "actpct50"),
}


def _canonical(stats) -> str:
    """Canonical, order-/type-sensitive serialization of an ExtractionStatistics,
    with the two inherently runtime-dependent fields removed."""
    d = dataclasses.asdict(stats)
    d["metadata"].pop("generated_at", None)
    d["extrapolation_10b"].pop("estimated_wall_time_hours", None)
    return json.dumps(d, indent=2, sort_keys=False)


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_matches_prechange_golden(tmp_path, name):
    kwargs, golden_key = _CONFIGS[name]
    golden = (_FIXTURE_DIR / f"golden_{golden_key}.json").read_text()
    stats = compute_statistics(
        feature_dir=str(_FIXTURE_DIR),
        output_path=str(tmp_path / "stats"),
        **kwargs,
    )
    assert _canonical(stats) == golden, (
        f"{name}: post-change output differs from the pre-change golden "
        f"golden_{golden_key}.json (literal, order- and type-sensitive) — the "
        "refactor changed a computed value (or a shared-golden coincidence broke)"
    )


@pytest.mark.parametrize("name", sorted(_CONFIGS))
def test_recompute_is_deterministic(tmp_path, name):
    kwargs, _ = _CONFIGS[name]
    a = _canonical(
        compute_statistics(
            feature_dir=str(_FIXTURE_DIR),
            output_path=str(tmp_path / "a"),
            **kwargs,
        )
    )
    b = _canonical(
        compute_statistics(
            feature_dir=str(_FIXTURE_DIR),
            output_path=str(tmp_path / "b"),
            **kwargs,
        )
    )
    assert a == b, f"{name}: recompute is non-deterministic"
