"""Identity contract for splice-selection multi-segment batching.

The batching mode changes how each shard-local RNG is consumed: cardinalities
for several ordinary (non-delta) segments are drawn together before their
per-segment ranking/sampling draws. The resulting selections retain the same
eligibility and distribution contracts, but are not byte-identical to the
legacy one-segment execution. Consequently this is a semantic artifact mode,
not a producer-only resource knob, and must be represented in every selection
identity and manifest that enables it.

``None`` and ``1`` are the legacy mode. Their canonical descriptor is absent,
which preserves every existing ``sel/<hash>/`` identity byte-for-byte.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

LEGACY_SPLICE_SELECTION_BATCH_SIZE = 1
SPLICE_SELECTION_BATCHING_SCHEMA_VERSION = 1
SPLICE_SELECTION_BATCHING_ALGORITHM = "ordinary_execution_runs_v1"


def resolve_splice_selection_batch_size(value: object) -> int:
    """Resolve ``None`` to legacy mode and strictly validate explicit values."""
    if value is None:
        return LEGACY_SPLICE_SELECTION_BATCH_SIZE
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"selection_splice_batch_size must be null or an int >= 1, got {value!r}")
    return value


def canonical_splice_selection_batching(
    delta_config: Mapping[str, Any] | None,
    batch_size: object,
) -> dict[str, Any] | None:
    """Return the conditional hash/manifest descriptor for the active mode.

    Batching is implemented only for ``splice_counterfactual`` selection. A
    configured value on any other pipeline is inert and therefore omitted from
    identity. Legacy splice mode (``None`` or ``1``) is also omitted so old
    hashes and manifests remain valid without migration.
    """
    resolved = resolve_splice_selection_batch_size(batch_size)
    if (
        resolved == LEGACY_SPLICE_SELECTION_BATCH_SIZE
        or delta_config is None
        or delta_config.get("mode") != "splice_counterfactual"
    ):
        return None
    return {
        "schema_version": SPLICE_SELECTION_BATCHING_SCHEMA_VERSION,
        "algorithm": SPLICE_SELECTION_BATCHING_ALGORITHM,
        "batch_size": resolved,
    }
