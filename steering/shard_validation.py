"""Core shard validation primitives.

Leaf module with no heavy dependencies — safe to import from both
``extract_features`` and ``validate_features`` without circular imports.
"""

from dataclasses import dataclass, field

import torch as t

from steering.feature_io import (
    MetadataKey,
    TensorKey,
    empty_segment_indices,
    load_features,
)

REQUIRED_FIELDS = [
    "offsets",
    "feature_ids",
    "feature_mags",
    "doc_ids",
    "segment_indices",
    "segment_positions",
    "segment_char_starts",
    "segment_char_ends",
    "doc_token_counts",
    "n_docs",
    "n_segments",
    "layer",
    "context_window",
    "pooling",
    "segment_mode",
    "segment_size",
]


# Expected dtypes — must match extraction-time dtypes (see
# ``steering.extract_features.process_shard``) and the post-load contract
# enforced by ``feature_io._upcast`` (see comment ``C17`` in slice-3 map).
_EXPECTED_DTYPES: dict[str, t.dtype] = {
    "offsets": t.int64,
    "feature_ids": t.int32,
    "feature_mags": t.float32,
    "doc_ids": t.int32,
    "segment_indices": t.int32,
    "segment_positions": t.int32,
    "segment_char_starts": t.int32,
    "segment_char_ends": t.int32,
    "doc_token_counts": t.int32,
}


@dataclass
class ValidationResult:
    valid: bool
    violations: list[str] = field(default_factory=list)


def _check_csr_shapes(data: dict, violations: list[str]) -> None:
    """Mutate ``violations`` with shape / offset / mag / doc-id / id-range issues.

    Shared by both ``validate_shard_data`` (in-memory) and ``validate_shard``
    (file-based). Both validators must already have verified ``REQUIRED_FIELDS``
    presence before calling this helper.

    The helper does NOT enforce dtypes (that's the file-based / in-memory
    contract; both validators perform the dtype check independently to make
    the violation messages self-explanatory) and does NOT perform the
    optional ``n_sae_features`` range check (caller-controlled parameter).
    """
    offsets = data["offsets"]
    feat_ids = data["feature_ids"]
    feat_mags = data["feature_mags"]
    doc_ids = data["doc_ids"]
    seg_indices = data["segment_indices"]
    seg_positions = data["segment_positions"]
    seg_char_starts = data["segment_char_starts"]
    seg_char_ends = data["segment_char_ends"]
    n_docs = data["n_docs"]
    n_segments = data["n_segments"]

    # Shape checks
    if len(offsets) != n_segments + 1:
        violations.append(f"len(offsets) {len(offsets)} != n_segments+1 {n_segments + 1}")
    if len(doc_ids) != n_segments:
        violations.append(f"len(doc_ids) {len(doc_ids)} != n_segments {n_segments}")
    if len(seg_indices) != n_segments:
        violations.append(f"len(segment_indices) {len(seg_indices)} != n_segments {n_segments}")
    if len(seg_positions) != n_segments:
        violations.append(f"len(segment_positions) {len(seg_positions)} != n_segments {n_segments}")
    if len(seg_char_starts) != n_segments:
        violations.append(
            f"len(segment_char_starts) {len(seg_char_starts)} != n_segments {n_segments}"
        )
    if len(seg_char_ends) != n_segments:
        violations.append(f"len(segment_char_ends) {len(seg_char_ends)} != n_segments {n_segments}")
    if len(data["doc_token_counts"]) != n_docs:
        violations.append(
            f"len(doc_token_counts) {len(data['doc_token_counts'])} != n_docs {n_docs}"
        )

    # Offset invariants
    if len(offsets) > 0 and offsets[0].item() != 0:
        violations.append(f"offsets[0] = {offsets[0].item()}, expected 0")
    if len(offsets) > 0 and offsets[-1].item() != len(feat_ids):
        violations.append(f"offsets[-1] {offsets[-1].item()} != len(feature_ids) {len(feat_ids)}")
    if len(feat_ids) != len(feat_mags):
        violations.append(f"len(feature_ids) {len(feat_ids)} != len(feature_mags) {len(feat_mags)}")
    if len(offsets) > 1:
        diffs = offsets[1:] - offsets[:-1]
        if not (diffs >= 0).all():
            violations.append("offsets not monotonically non-decreasing")

    # Base shards must never contain an empty segment (issue #522). These
    # validators are only ever applied to base-extraction data, so this is
    # unconditional — there is no legitimate caller that needs to relax it.
    empties = empty_segment_indices(offsets)
    if empties:
        violations.append(
            f"segment(s) {empties[:8]} have zero features "
            f"(offsets[i]==offsets[i+1]); base shards must be non-empty (issue #522)"
        )

    # Magnitude checks
    if len(feat_mags) > 0 and not (feat_mags > 0).all():
        violations.append("Some feature magnitudes are <= 0")
    if len(feat_mags) > 0 and not bool(t.isfinite(feat_mags).all()):
        violations.append("Some feature magnitudes are not finite (inf/nan)")

    # doc_ids monotonically non-decreasing
    if len(doc_ids) > 1:
        doc_diffs = doc_ids[1:].to(t.int64) - doc_ids[:-1].to(t.int64)
        if not (doc_diffs >= 0).all():
            violations.append("doc_ids not monotonically non-decreasing")


def _check_dtypes(data: dict, violations: list[str]) -> None:
    """Mutate ``violations`` with disk-canonical dtype mismatches.

    The contract is the same in-memory (extraction-side) and after load
    (``feature_io._upcast`` normalizes back to int64/int32/float32/int32).
    Both ``validate_shard`` and ``validate_shard_data`` enforce this so the
    two paths report identical violations on a malformed shard.
    """
    for key, expected in _EXPECTED_DTYPES.items():
        if key in data and data[key].dtype != expected:
            violations.append(f"{key} dtype {data[key].dtype} != {expected}")


def _check_feature_id_range(data: dict, n_sae_features: int | None, violations: list[str]) -> None:
    """Mutate ``violations`` with feature-ID range issues if ``n_sae_features`` is set."""
    feat_ids = data["feature_ids"]
    if n_sae_features is not None and len(feat_ids) > 0:
        if feat_ids.min().item() < 0:
            violations.append(f"Negative feature_id: {feat_ids.min().item()}")
        if feat_ids.max().item() >= n_sae_features:
            violations.append(
                f"feature_id {feat_ids.max().item()} >= n_sae_features {n_sae_features}"
            )


def _check_bool_flag(data: dict, key: str, violations: list[str]) -> None:
    """If ``key`` is present in ``data`` it must be a Python bool."""
    if key in data and not isinstance(data[key], bool):
        violations.append(f"{key} must be bool, got {type(data[key]).__name__}: {data[key]!r}")


def validate_shard_data(
    data: dict,
    n_sae_features: int | None = None,
) -> ValidationResult:
    """Validate a shard data dict in-memory (no disk I/O).

    Performs the same checks as validate_shard but operates on the in-memory
    dict, avoiding the save->load round-trip. This is the performance-critical
    path used during extraction.

    Args:
        data: Shard dict with standard keys (offsets, feature_ids, etc.)
            and scalar metadata (n_docs, n_segments, etc.).
        n_sae_features: If provided, check feature IDs are in valid range.

    Returns:
        ValidationResult with valid=True if all checks pass.
    """
    violations: list[str] = []

    # Check required fields
    for f in REQUIRED_FIELDS:
        if f not in data:
            violations.append(f"Missing required field: {f}")
    if violations:
        return ValidationResult(valid=False, violations=violations)

    # Dtype enforcement — must match the on-disk contract because
    # ``feature_io._upcast`` normalizes back to these dtypes after load.
    # Production extraction (``ShardData.to_dict()``) emits these dtypes
    # directly, so this check guards against extraction-side regressions.
    _check_dtypes(data, violations)

    # Shape, offset, magnitude, doc-id checks (shared with validate_shard)
    _check_csr_shapes(data, violations)

    # Optional feature-ID range check
    _check_feature_id_range(data, n_sae_features, violations)

    # Precomputed percentile checks (optional — only if present)
    _validate_percentiles(data, violations)

    # Optional extraction flags — when present, each must be a bool.
    _check_bool_flag(data, "context_independent_features", violations)
    _check_bool_flag(data, "ignore_bos_features", violations)

    return ValidationResult(valid=len(violations) == 0, violations=violations)


def _validate_percentiles(data: dict, violations: list[str]) -> None:
    """Validate precomputed percentile data if present in the shard."""
    pct_key = TensorKey.FEATURE_PERCENTILES
    checksum_key = MetadataKey.PERCENTILE_CHECKSUM

    has_pcts = pct_key in data and isinstance(data[pct_key], t.Tensor)
    has_checksum = checksum_key in data

    if not has_pcts and not has_checksum:
        return  # No precomputed data — nothing to validate

    if has_pcts and not has_checksum:
        violations.append("feature_percentiles present but percentile_checksum missing")

    if has_checksum and not has_pcts:
        violations.append("percentile_checksum present but feature_percentiles missing")

    if has_pcts:
        pcts = data[pct_key]
        feat_ids = data["feature_ids"]

        if pcts.dtype != t.uint8:
            violations.append(f"feature_percentiles dtype {pcts.dtype}, expected uint8")
        if len(pcts) != len(feat_ids):
            violations.append(
                f"feature_percentiles length {len(pcts)} != feature_ids length {len(feat_ids)}"
            )


def validate_shard(
    path: str,
    n_sae_features: int | None = None,
) -> ValidationResult:
    """Validate a feature shard file.

    Args:
        path: Path to .safetensors.zst shard file.
        n_sae_features: If provided, check feature IDs are in valid range.

    Returns:
        ValidationResult with valid=True if all checks pass, else violations list.
    """
    violations: list[str] = []
    data = load_features(path)

    # Check required fields
    for f in REQUIRED_FIELDS:
        if f not in data:
            violations.append(f"Missing required field: {f}")
    if violations:
        return ValidationResult(valid=False, violations=violations)

    # Dtype enforcement (same as validate_shard_data; symmetry per slice-3 P1-3).
    _check_dtypes(data, violations)

    # Shape, offset, magnitude, doc-id checks (shared with validate_shard_data)
    _check_csr_shapes(data, violations)

    # Optional feature-ID range check
    _check_feature_id_range(data, n_sae_features, violations)

    # Precomputed percentile checks (optional — only if present)
    _validate_percentiles(data, violations)

    # Optional extraction flags — each must be a bool when present.
    # Symmetry with validate_shard_data (slice-3 P1-2).
    _check_bool_flag(data, "context_independent_features", violations)
    _check_bool_flag(data, "ignore_bos_features", violations)

    return ValidationResult(valid=len(violations) == 0, violations=violations)


def assert_csr_invariants(
    all_feat_ids: list[t.Tensor],
    all_feat_mags: list[t.Tensor],
    n_features: int | None = None,
) -> None:
    """Lightweight invariant check for in-memory extraction results.

    TEST/DEV sanity helper ONLY. This is the lone ``assert``-based guard in
    this module, so every check here is stripped under ``python -O`` — do NOT
    rely on it as a production guard. Production code uses the typed ``raise``
    guards (``validate_shard``/``validate_shard_data`` and the
    ``EmptyExtractionSegmentError``/``CorruptFeatureShardError`` raises), which
    survive ``-O``. Use this only for fast sanity checks in tests and ad-hoc
    dev scripts.

    Raises AssertionError if any invariant is violated (when assertions are on).
    """
    assert len(all_feat_ids) == len(all_feat_mags), (
        f"Length mismatch: {len(all_feat_ids)} ids vs {len(all_feat_mags)} mags"
    )
    for i, (ids, mags) in enumerate(zip(all_feat_ids, all_feat_mags, strict=True)):
        # Base shards must never contain an empty segment (issue #522);
        # unconditional because this helper only checks base-extraction results.
        assert len(ids) >= 1, (
            f"Segment {i}: empty (zero features); base shards must be non-empty (issue #522)"
        )
        assert ids.dtype == t.int32, f"Segment {i}: ids dtype {ids.dtype}, expected int32"
        assert mags.dtype == t.float32, f"Segment {i}: mags dtype {mags.dtype}, expected float32"
        assert len(ids) == len(mags), f"Segment {i}: len mismatch {len(ids)} vs {len(mags)}"
        if len(mags) > 0:
            assert (mags > 0).all(), f"Segment {i}: non-positive magnitudes"
        if len(ids) > 1:
            assert (ids[:-1] <= ids[1:]).all(), f"Segment {i}: feature IDs not sorted ascending"
        if n_features is not None and len(ids) > 0:
            assert ids.max().item() < n_features, (
                f"Segment {i}: feature ID {ids.max().item()} >= n_features {n_features}"
            )
            assert ids.min().item() >= 0, f"Segment {i}: negative feature ID"
