"""Hypothesis-based property tests for the shard validators (slice-3 §B6).

Pins three properties:

1. ``validate_shard_data`` and ``validate_shard`` agree on round-tripped
   valid shards (the symmetry contract closed by P1-2 + P1-3).
2. ``assert_csr_invariants`` always names the offending invariant.
3. ``_select_spot_check_docs`` always includes the first and last document
   when ``n_spot_check >= 2``.

Hypothesis settings (per design spec §7 + project memory):
``derandomize=True`` so failures reproduce locally, ``max_examples=200``.
"""

from __future__ import annotations

import pytest
import torch as t
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from steering.shard_validation import (
    assert_csr_invariants,
    validate_shard,
    validate_shard_data,
)
from steering.validate_features import _select_spot_check_docs
from tests.steering._helpers import (
    make_valid_shard_data,
    save_valid_shard,
)

_HYPOTHESIS_SETTINGS = settings(
    derandomize=True,
    max_examples=200,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
    deadline=None,
)


# ---------------------------------------------------------------------------
# Property 1 — validators agree on valid round-tripped shards.
# ---------------------------------------------------------------------------


@_HYPOTHESIS_SETTINGS
@given(
    n_docs=st.integers(min_value=1, max_value=4),
    segs_per_doc=st.integers(min_value=1, max_value=3),
    feats_per_seg=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_validators_agree_on_valid_shards(
    tmp_path_factory: pytest.TempPathFactory,
    n_docs: int,
    segs_per_doc: int,
    feats_per_seg: int,
    seed: int,
) -> None:
    """For any valid shard, both validators report ``valid=True``."""
    tmp_path = tmp_path_factory.mktemp("prop_agree")
    path, data = save_valid_shard(
        tmp_path,
        n_docs=n_docs,
        segs_per_doc=segs_per_doc,
        feats_per_seg=feats_per_seg,
        seed=seed,
    )

    in_memory = validate_shard_data(data)
    on_disk = validate_shard(path)

    assert in_memory.valid == on_disk.valid, (
        f"in_memory={in_memory.violations}; on_disk={on_disk.violations}"
    )
    assert in_memory.valid, in_memory.violations


# ---------------------------------------------------------------------------
# Property 2 — assert_csr_invariants always names the failing invariant.
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=200, deadline=None)
@given(
    n_segments=st.integers(min_value=1, max_value=5),
    seg_len=st.integers(min_value=2, max_value=6),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_assert_csr_invariants_descending_ids_named(
    n_segments: int, seg_len: int, seed: int
) -> None:
    """Force a descending-ids violation; the message must say the segment index.

    For each segment, we either keep ascending IDs (valid) or strictly
    descend (invalid). We pick a single random segment to corrupt and
    expect the AssertionError message to name that segment.
    """
    g = t.Generator().manual_seed(seed)
    bad_seg = int(t.randint(0, n_segments, (1,), generator=g).item())

    ids: list[t.Tensor] = []
    mags: list[t.Tensor] = []
    for i in range(n_segments):
        if i == bad_seg:
            # Strictly descending: e.g. [seg_len, seg_len-1, ..., 1].
            ids.append(
                t.arange(seg_len, 0, -1, dtype=t.int32)
            )
        else:
            ids.append(t.arange(seg_len, dtype=t.int32))
        mags.append(t.full((seg_len,), 0.1, dtype=t.float32))

    with pytest.raises(AssertionError) as exc_info:
        assert_csr_invariants(ids, mags)
    msg = str(exc_info.value)
    assert f"Segment {bad_seg}" in msg, msg
    assert "feature IDs not sorted ascending" in msg, msg


# ---------------------------------------------------------------------------
# Property 3 — _select_spot_check_docs always includes first + last endpoints
# when n_spot_check >= 2.
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=200, deadline=None)
@given(
    n_docs=st.integers(min_value=2, max_value=20),
    n_spot_check=st.integers(min_value=2, max_value=20),
    seed=st.integers(min_value=0, max_value=2**16 - 1),
)
def test_property_select_spot_check_docs_first_and_last_always_picked(
    n_docs: int, n_spot_check: int, seed: int
) -> None:
    g = t.Generator().manual_seed(seed)
    counts = t.randint(1, 1000, (n_docs,), generator=g, dtype=t.int32)

    selected = _select_spot_check_docs(counts, n_docs, n_spot_check)

    # Endpoints always present
    assert 0 in selected, selected
    assert (n_docs - 1) in selected, selected
    # No duplicates
    assert len(selected) == len(set(selected)), selected
    # Sorted
    assert selected == sorted(selected), selected
    # Bounded by both n_docs and n_spot_check
    assert len(selected) <= n_docs
    assert len(selected) <= n_spot_check


# ---------------------------------------------------------------------------
# Property 4 — validate_shard_data dtype check is order-invariant
# (any one wrong dtype produces at least one dtype violation).
# ---------------------------------------------------------------------------


@settings(derandomize=True, max_examples=50, deadline=None)
@given(
    field_choice=st.sampled_from(
        ["offsets", "feature_ids", "feature_mags", "doc_ids"]
    ),
    bad_dtype_choice=st.sampled_from(["int8", "int16", "int64", "float64"]),
)
def test_property_dtype_violation_always_reported(
    field_choice: str, bad_dtype_choice: str
) -> None:
    """Any single wrong-dtype field produces at least one dtype violation."""
    bad_dtype = {
        "int8": t.int8,
        "int16": t.int16,
        "int64": t.int64,
        "float64": t.float64,
    }[bad_dtype_choice]

    data = make_valid_shard_data()
    if data[field_choice].dtype == bad_dtype:
        # Skip the no-op corruption case.
        return
    data[field_choice] = data[field_choice].to(bad_dtype)

    result = validate_shard_data(data)

    assert not result.valid
    assert any(f"{field_choice} dtype" in v for v in result.violations), (
        result.violations
    )
