"""Test-only readers for clustered artifact metadata."""

from __future__ import annotations

import pyarrow as pa

from steering.eval.axbench_bpb_adv.artifact_identity import (
    ARTIFACT_FAMILY_METADATA_KEY,
    FEATURE_CONTRACT_METADATA_KEY,
    PRECOMPUTED_BAG_SET_METADATA_KEY,
    SOURCE_CONTRACT_METADATA_KEY,
    UPSTREAM_CANDIDATES_METADATA_KEY,
)

_RESERVED_KEYS = frozenset(
    {
        ARTIFACT_FAMILY_METADATA_KEY,
        FEATURE_CONTRACT_METADATA_KEY,
        PRECOMPUTED_BAG_SET_METADATA_KEY,
        SOURCE_CONTRACT_METADATA_KEY,
        UPSTREAM_CANDIDATES_METADATA_KEY,
    }
)


def artifact_identity_metadata(table: pa.Table) -> dict[str, str]:
    """Decode the reserved metadata fields asserted by artifact tests."""
    metadata = dict(table.schema.metadata or {})
    return {
        key.decode("ascii"): metadata[key].decode("utf-8")
        for key in sorted(_RESERVED_KEYS)
        if key in metadata
    }
