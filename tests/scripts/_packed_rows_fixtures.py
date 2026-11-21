"""Shared synthetic-env builders for the packed-rows golden + profiling tests (issue #531).

Re-exports the deterministic env/runner helpers that already live in the packed-rows
integration test, so the golden anchor and the profiling tests reuse ONE builder (DRY)
without duplicating ~200 lines. The integration test remains the single source of these
helpers.

NOTE on determinism: ``_make_precompute_env`` draws the percentile-overlay values with an
*unseeded* ``torch.randint``. Callers that need a reproducible env (the golden anchor) must
``torch.manual_seed(...)`` immediately before calling it — see ``GOLDEN_SEED`` and its use in
``test_packed_rows_golden.py``. The feature tensors, token boundaries, and (with a fixed seed)
the percentile overlay are otherwise fully deterministic, and packed-row shards are written via
``save_overlay_shard`` (empty safetensors metadata ⇒ reproducible ``.zst`` bytes).
"""
from tests.steering.test_precompute_packed_rows_integration import (  # noqa: F401
    N_DOCS,
    N_FEATS_PER_SEG,
    N_SAE,
    SEGMENT_SIZE,
    SimpleTokenizer,
    _copy_shards,
    _make_feature_index,
    _make_precompute_env,
    _run_precompute,
    _shard_bytes,
)

# Fixed seed for the committed golden anchor (see test_packed_rows_golden.py).
GOLDEN_SEED = 20260615
