"""Cache-key and manifest-serialization contracts for the autointerp filter."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict

from steering.eval.autointerp_filter.config import AutoInterpFilterConfig

INTRUDER_TRIAL_METHOD = "delphi_intruder_token_v3"


def intruder_trial_key(
    *,
    target_hash: str,
    trial_index: int,
    positive_ref_keys: list[str],
    intruder_ref_key: str,
    intruder_index: int,
    quantile: int | None,
    intruder_highlight_indices: list[int],
    n_positive_examples: int,
    n_quantiles: int,
    example_ctx_len: int,
    activation_threshold_ratio: float,
) -> str:
    """Return the byte-stable legacy trial cache key payload hash."""
    payload = {
        "method": INTRUDER_TRIAL_METHOD,
        "target_hash": target_hash,
        "trial_index": trial_index,
        "positive_refs": positive_ref_keys,
        "intruder_ref": intruder_ref_key,
        "intruder_index": intruder_index,
        "quantile": quantile,
        "intruder_highlight_indices": intruder_highlight_indices,
        "n_positive_examples": n_positive_examples,
        "n_quantiles": n_quantiles,
        "example_ctx_len": example_ctx_len,
        "activation_threshold_ratio": activation_threshold_ratio,
        "highlight_policy": "delphi_false_positive_token_v1",
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def runner_manifest_source_index(
    cfg: AutoInterpFilterConfig,
) -> dict[str, object]:
    """Persist the serial runner's source-index configuration."""
    return asdict(cfg.source_index)


def runner_manifest_token_examples(
    cfg: AutoInterpFilterConfig,
    *,
    bulk_materialization: bool = True,
) -> dict[str, object]:
    """Serialize token-example config without invalidating legacy serial runs.

    ``materialization_batch_size=1`` selects the pre-batching implementation.
    Omitting the new scheduling field in that mode preserves the exact manifest
    comparison payload produced before the field existed.
    """
    token_examples = asdict(cfg.token_examples)
    if cfg.token_examples.materialization_batch_size == 1 or not bulk_materialization:
        token_examples.pop("materialization_batch_size")
    return token_examples
