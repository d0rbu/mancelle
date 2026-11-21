"""Bin-stratified sampling over Neuronpedia activation entries.

No randomness — sampling is bit-deterministic.
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any

from steering.eval.feature_filter.neuronpedia_activations import (
    ActivationEntry,
    FeatureActivationPayload,
)
from steering.eval.feature_filter.types import (
    ActivatingExample,
    BinSample,
    InvalidHistogramError,
    NoNonzeroActivationsError,
)

__all__ = ["sample_by_bins"]


_BAND_LABELS: dict[int, str] = {
    4: "band 5/5 (peak firing)",
    3: "band 4/5 (very high firing)",
    2: "band 3/5 (high firing)",
    1: "band 2/5 (moderate firing)",
    0: "band 1/5 (weak firing)",
}
_BAND_LABEL_ABSENT = "band 1/5 (absent — feature did not fire)"


def _band_label(bin_index: int, is_absent: bool) -> str:
    if bin_index == 0 and is_absent:
        return _BAND_LABEL_ABSENT
    return _BAND_LABELS[bin_index]


def _to_example(
    entry: ActivationEntry, tokenizer: Any,
) -> ActivatingExample:
    text = tokenizer.convert_tokens_to_string(entry.tokens)
    idx = entry.max_value_token_index
    if idx == 0:
        peak_char = 0
    else:
        peak_char = len(tokenizer.convert_tokens_to_string(entry.tokens[:idx]))
    return ActivatingExample(
        tokens=entry.tokens,
        text=text,
        max_value=entry.max_value,
        max_value_token_index=idx,
        peak_char_offset=peak_char,
    )


def sample_by_bins(
    payload: FeatureActivationPayload,
    *,
    samples_per_bin: int,
    tokenizer: Any,
) -> list[BinSample]:
    """Return up to `5 * samples_per_bin` BinSamples, peak-first order.

    Requires exactly 5 non-sentinel bins; otherwise raises
    `InvalidHistogramError` (fail fast on Neuronpedia drift). If every
    bin is empty (after filtering sentinel `bin_min == -1` entries), raises
    `NoNonzeroActivationsError`.
    """
    non_sentinel = [e for e in payload.entries if e.bin_min != -1.0]
    if not non_sentinel:
        raise NoNonzeroActivationsError(
            f"feature {payload.feature_id}: no non-sentinel activation entries"
        )

    groups: dict[tuple[float, float], list[tuple[int, ActivationEntry]]] = defaultdict(list)
    for pos, e in enumerate(non_sentinel):
        groups[(e.bin_min, e.bin_max)].append((pos, e))

    sorted_keys = sorted(groups.keys(), key=lambda k: k[0])
    if len(sorted_keys) != 5:
        raise InvalidHistogramError(
            f"feature {payload.feature_id}: expected 5 non-sentinel bins, got "
            f"{len(sorted_keys)}. Neuronpedia's bin structure may have changed; "
            f"re-run scripts/validation/check_neuronpedia_schema.py."
        )

    results: list[BinSample] = []
    for bin_index, key in enumerate(sorted_keys):
        bin_entries = groups[key]
        bin_entries.sort(key=lambda pe: (-pe[1].max_value, pe[0]))
        take = min(samples_per_bin, len(bin_entries))
        for _, entry in bin_entries[:take]:
            is_absent = entry.max_value == 0.0
            label = _band_label(bin_index, is_absent)
            example = _to_example(entry, tokenizer)
            results.append(BinSample(
                bin_index=bin_index,
                band_label=label,
                is_absent=is_absent,
                max_value=entry.max_value,
                example=example,
            ))

    # Final sort is peak-first. timsort stability + the fact that entries
    # were appended in bin_index-asc order (and within each bin in desc
    # max_value with pos tie-break) means within-bin ordering is preserved
    # by this sort. DO NOT reorder pre-sort entries without updating this.
    results.sort(key=lambda s: (-s.bin_index, -s.max_value))
    return results
