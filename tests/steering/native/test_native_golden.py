"""Layer 5: pin truncated native capture output across future code changes.

Re-pointed for Phase-2 PR-1: the collector is the PRODUCTION truncated build
(``build_native_model(..., layer)``) from a fresh seeded gemma-2 twin (shared
conftest factory reused; truncation mutates in place). The golden hash is
re-baselined on the truncation mechanism (spec §6 / Risk #7: coverage and
goldens are re-baselined on the new mechanism, not assumed to carry over) and
asserted byte-stable thereafter.
"""

import hashlib

import pytest
import torch as t

from steering.native_collector import NativeCollector
from steering.sae_resolver import ActivationType
from tests.steering.native.conftest import _build_tiny_gemma2

_LAYER = 2

# Regenerate ONLY intentionally: set GOLDEN_SHA256 back to the
# "AUTO-GENERATED-ON-FIRST-RUN" sentinel, run this test (it fails and prints
# the freshly computed hash), then paste that hash here.
#
# NOTE: this hash is byte-IDENTICAL to the Phase-1 hook-path golden — the
# spec §4.1 invariant (truncated capture == full-forward-hook capture,
# max_abs == 0) holds for this fixed input, re-confirmed at re-baseline.
GOLDEN_SHA256 = "722e70dca8149da966407401ccdedb502f0cc714e34cc8009b68937773b42875"


def _capture_hash() -> str:
    col = NativeCollector.truncated(
        _build_tiny_gemma2(), model_name="gemma-2-2b", layer=_LAYER,
    )
    assert col is not None
    ids = t.arange(1, 13, dtype=t.long).reshape(2, 6)  # fixed input
    out = col.capture(ids, _LAYER, hook_name=f"blocks.{_LAYER}.hook_resid_post",
                      activation_type=ActivationType.residual)
    return hashlib.sha256(out.float().contiguous().numpy().tobytes()).hexdigest()


def test_native_capture_is_byte_stable():
    h = _capture_hash()
    if GOLDEN_SHA256 == "AUTO-GENERATED-ON-FIRST-RUN":
        pytest.fail(
            f"Golden not pinned. Set GOLDEN_SHA256 = {h!r} in this file "
            f"and re-run to lock the regression."
        )
    assert h == GOLDEN_SHA256, (
        "truncated native capture changed — investigate before updating the "
        "golden"
    )
