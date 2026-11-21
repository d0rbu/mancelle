"""Domain errors for AxBench-BPB-Adv."""

from __future__ import annotations

from steering.conditional.feature_gcg.errors import FeatureGcgError


class AxBenchBpbAdvError(FeatureGcgError):
    """AxBench-BPB-Adv error (a FeatureGcgError so the shared primitives' base is caught)."""
