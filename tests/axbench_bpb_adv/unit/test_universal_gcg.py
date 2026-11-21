"""Unit tests for the model-free universal-GCG bookkeeping core.

These tests exercise pure functions over attack-row dicts and bare gradient
tensors only: NO model and NO torch forward. Tensors here are hand-built data,
not produced by a model. The attack rows are hand-built ATTACK_ROWS_SCHEMA-shaped
dicts (only the columns the universal-GCG core reads are populated).
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, assert_type, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

from nanochat.atomic_io import atomic_write_parquet
from steering.conditional.feature_gcg.aggregation import aggregate_objective
from steering.conditional.feature_gcg.aggregation import (
    minibatch_for_step as _typed_minibatch_for_step,
)
from steering.conditional.feature_gcg.candidate_search import (
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.edit_primitives import EditableSpan
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_bpb_adv.editable_tokens import select_final_k_editable_span
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.optimize import (
    _padded_trigger_tokens,
    _strip_padded_trigger_tokens,
    _trigger_row,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import UNIVERSAL_TRIGGERS_SCHEMA
from steering.eval.axbench_bpb_adv.universal_gcg import (
    ConceptGroup,
    ConceptGroupKey,
    aggregate_trigger_gradient,
    concept_groups,
    group_key,
    init_trigger,
    rank_trigger_replacements,
)
from steering.refined_types import NonEmptyStr, NonNegativeInt, PositiveInt


def minibatch_for_step(
    n_pairs: int,
    *,
    step: int,
    minibatch_size: int,
    group_seed: str,
) -> list[int]:
    """Build the refined scalar inputs owned by these behavioral tests."""
    return _typed_minibatch_for_step(
        NonNegativeInt.parse(n_pairs),
        step=NonNegativeInt.parse(step),
        minibatch_size=PositiveInt.parse(minibatch_size),
        group_seed=group_seed,
    )


# ---------------------------------------------------------------------------
# attack-row builder
# ---------------------------------------------------------------------------


def _attack_row(
    *,
    concept_id: int,
    mode: str = "positive",
    method_name: str = "diff_mean",
    target_model: str = "d12_base",
    coefficient_str: str = "1.0",
    adversarial_method: str = "faster_gcg",
    strength_index: int = 0,
    editable_token_count: int = 3,
    core_prefix_id: str = "coreprefix000001",
    concept_rich_sample_id: str = "rich-0",
    concept_avoiding_sample_id: str = "avoid-0",
    source_prompt_tokens: list[int] | None = None,
    editable_token_start: int | None = None,
) -> dict[str, Any]:
    """Emit one ATTACK_ROWS_SCHEMA-shaped dict (relevant columns only).

    The editable span is the FINAL ``editable_token_count`` non-BOS tokens, so
    ``editable_token_start`` defaults to ``len(tokens) - k`` (matching how
    ``prepare_bpb_attack_rows`` builds the span). BOS is token 0.
    """

    if source_prompt_tokens is None:
        source_prompt_tokens = [0, 11, 12, 13, 14, 15]
    if editable_token_start is None:
        editable_token_start = (
            len(source_prompt_tokens)
            if editable_token_count == 0
            else len(source_prompt_tokens) - editable_token_count
        )
    return {
        "concept_id": int(concept_id),
        "mode": str(mode),
        "method_name": str(method_name),
        "target_model": str(target_model),
        "coefficient_str": str(coefficient_str),
        "adversarial_method": str(adversarial_method),
        "strength_index": int(strength_index),
        "editable_token_count": int(editable_token_count),
        "editable_token_start": int(editable_token_start),
        "core_prefix_id": str(core_prefix_id),
        "concept_rich_sample_id": str(concept_rich_sample_id),
        "concept_avoiding_sample_id": str(concept_avoiding_sample_id),
        "source_prompt_tokens": [int(token_id) for token_id in source_prompt_tokens],
    }


def _expected_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        int(row["concept_id"]),
        str(row["mode"]),
        str(row["method_name"]),
        str(row["target_model"]),
        str(row["coefficient_str"]),
        str(row["adversarial_method"]),
        int(row["strength_index"]),
    )


# ---------------------------------------------------------------------------
# concept_groups
# ---------------------------------------------------------------------------


def _grid_rows() -> list[dict[str, Any]]:
    """2 concepts x 2 coeffs x strength_index{0,1}, 3 prefixes/pairs.

    Each (concept, coeff, strength_index) cell is one concept group; each group
    carries several core prefixes (and several pairs per prefix), so the
    sampler/grouping cross-products are non-trivial.
    """

    rows: list[dict[str, Any]] = []
    for concept_id in (1, 2):
        for coefficient_str in ("1.0", "2.0"):
            for strength_index in (0, 1):
                for prefix_idx in range(3):
                    core_prefix = f"coreprefix{prefix_idx:010d}"
                    for pair_idx in range(2):
                        rows.append(
                            _attack_row(
                                concept_id=concept_id,
                                coefficient_str=coefficient_str,
                                strength_index=strength_index,
                                core_prefix_id=core_prefix,
                                concept_rich_sample_id=f"rich-{pair_idx}",
                                concept_avoiding_sample_id=f"avoid-{pair_idx}",
                            )
                        )
    return rows


def test_concept_groups_partition_by_seven_tuple_every_row_in_one_group() -> None:
    rows = _grid_rows()
    groups = concept_groups(rows)

    # 2 concepts x 2 coeffs x 2 strengths = 8 groups.
    assert len(groups) == 8
    assert all(isinstance(group, ConceptGroup) for group in groups)

    # Every input row lands in exactly one group; union == input (by identity of
    # the populated fields), no row duplicated or dropped.
    total_member_rows = sum(len(group.rows) for group in groups)
    assert total_member_rows == len(rows)

    # Each group is internally homogeneous on the 7-tuple.
    for group in groups:
        for row in group.rows:
            assert _expected_key(row) == group.key

    # Keys are unique across groups and each covers 3 prefixes x 2 pairs = 6 rows.
    seen_keys = [group.key for group in groups]
    assert len(seen_keys) == len(set(seen_keys))
    for group in groups:
        assert len(group.rows) == 6


def test_concept_groups_only_faster_gcg_rows_survive() -> None:
    rows = [
        _attack_row(concept_id=1, adversarial_method="faster_gcg"),
        _attack_row(concept_id=1, adversarial_method="gepa", editable_token_count=2),
        _attack_row(
            concept_id=1,
            adversarial_method="additive_direction",
            editable_token_count=0,
        ),
    ]
    groups = concept_groups(rows)

    assert len(groups) == 1
    assert len(groups[0].rows) == 1
    assert all(
        row["adversarial_method"] == "faster_gcg" for row in groups[0].rows
    )


def test_concept_groups_key_str_is_deterministic_pipe_join() -> None:
    rows = [_attack_row(concept_id=7, coefficient_str="3.5", strength_index=0)]
    (group,) = concept_groups(rows)

    assert group.key_str == "|".join(str(part) for part in group.key)
    assert group.key_str == "7|positive|diff_mean|d12_base|3.5|faster_gcg|0"


def test_concept_groups_editable_token_count_recorded_as_k() -> None:
    rows = [
        _attack_row(concept_id=1, editable_token_count=4),
        _attack_row(
            concept_id=1, editable_token_count=4, core_prefix_id="coreprefix000002"
        ),
    ]
    (group,) = concept_groups(rows)

    assert group.editable_token_count == 4
    assert group.is_control is False


def test_concept_groups_control_group_k0_is_flagged() -> None:
    rows = [
        _attack_row(
            concept_id=1,
            adversarial_method="faster_gcg",
            editable_token_count=0,
        )
    ]
    (group,) = concept_groups(rows)

    assert group.editable_token_count == 0
    assert group.is_control is True


def test_concept_groups_nonconstant_editable_count_raises() -> None:
    # Two rows that share the 8-tuple but disagree on editable_token_count: the
    # group has no well-defined k.
    rows = [
        _attack_row(concept_id=1, editable_token_count=3),
        _attack_row(
            concept_id=1,
            editable_token_count=2,
            core_prefix_id="coreprefix000002",
        ),
    ]
    with pytest.raises(AxBenchBpbAdvError, match="editable_token_count"):
        concept_groups(rows)


def test_concept_groups_row_editable_count_must_equal_group_k() -> None:
    # The cross-check the design review flagged: k is anchored on the first
    # member row, and EVERY member row must equal it. The offending row is named
    # so the failure is actionable.
    rows = [
        _attack_row(concept_id=9, editable_token_count=3),
        _attack_row(
            concept_id=9,
            editable_token_count=5,
            core_prefix_id="coreprefix000002",
        ),
    ]
    with pytest.raises(AxBenchBpbAdvError) as excinfo:
        concept_groups(rows)
    message = str(excinfo.value)
    # The error names the offending group (concept_id=9 in the key_str) AND the
    # specific offending row (its core_prefix_id and its mismatching value).
    assert "9" in message
    assert "coreprefix000002" in message
    assert "5" in message
    assert "k=3" in message


def test_concept_groups_deterministic_order_by_key() -> None:
    rows = _grid_rows()
    groups_a = concept_groups(rows)
    groups_b = concept_groups(list(reversed(rows)))

    keys_a = [group.key for group in groups_a]
    keys_b = [group.key for group in groups_b]
    assert keys_a == keys_b
    assert keys_a == sorted(keys_a)


def test_concept_groups_rows_sorted_within_group_deterministically() -> None:
    # Rows handed in a scrambled order must come back in the canonical
    # (core_prefix_id, concept_rich_sample_id, concept_avoiding_sample_id) order.
    scrambled = [
        _attack_row(
            concept_id=1,
            core_prefix_id="coreprefix000002",
            concept_rich_sample_id="rich-1",
            concept_avoiding_sample_id="avoid-1",
        ),
        _attack_row(
            concept_id=1,
            core_prefix_id="coreprefix000001",
            concept_rich_sample_id="rich-1",
            concept_avoiding_sample_id="avoid-0",
        ),
        _attack_row(
            concept_id=1,
            core_prefix_id="coreprefix000001",
            concept_rich_sample_id="rich-0",
            concept_avoiding_sample_id="avoid-9",
        ),
        _attack_row(
            concept_id=1,
            core_prefix_id="coreprefix000001",
            concept_rich_sample_id="rich-0",
            concept_avoiding_sample_id="avoid-0",
        ),
    ]
    (group,) = concept_groups(scrambled)
    ordering = [
        (
            row["core_prefix_id"],
            row["concept_rich_sample_id"],
            row["concept_avoiding_sample_id"],
        )
        for row in group.rows
    ]
    assert ordering == sorted(ordering)
    assert ordering == [
        ("coreprefix000001", "rich-0", "avoid-0"),
        ("coreprefix000001", "rich-0", "avoid-9"),
        ("coreprefix000001", "rich-1", "avoid-0"),
        ("coreprefix000002", "rich-1", "avoid-1"),
    ]


def test_concept_groups_empty_input_returns_empty() -> None:
    assert concept_groups([]) == []


def test_concept_groups_all_non_gcg_returns_empty() -> None:
    rows = [
        _attack_row(
            concept_id=1, adversarial_method="gepa", editable_token_count=2
        ),
        _attack_row(
            concept_id=1,
            adversarial_method="additive_direction",
            editable_token_count=0,
        ),
    ]
    assert concept_groups(rows) == []


# ---------------------------------------------------------------------------
# minibatch_for_step
# ---------------------------------------------------------------------------


def test_minibatch_for_step_deterministic() -> None:
    first = minibatch_for_step(7, step=4, minibatch_size=3, group_seed="seed-A")
    second = minibatch_for_step(7, step=4, minibatch_size=3, group_seed="seed-A")
    assert first == second


def test_minibatch_for_step_covers_every_index_once_per_epoch() -> None:
    n_pairs = 7
    minibatch_size = 3
    epoch_len = math.ceil(n_pairs / minibatch_size)  # 3 steps cover the epoch

    for epoch in range(3):
        collected: list[int] = []
        for sub_step in range(epoch_len):
            step = epoch * epoch_len + sub_step
            collected.extend(
                minibatch_for_step(
                    n_pairs,
                    step=step,
                    minibatch_size=minibatch_size,
                    group_seed="seed-cov",
                )
            )
        # Full coverage: union over one epoch is exactly range(n_pairs), no repeat.
        assert sorted(collected) == list(range(n_pairs))
        assert len(collected) == n_pairs
        assert Counter(collected) == Counter(range(n_pairs))


def test_minibatch_for_step_reshuffles_between_epochs() -> None:
    n_pairs = 6
    minibatch_size = 2
    epoch_len = math.ceil(n_pairs / minibatch_size)  # 3

    def epoch_order(epoch: int) -> list[int]:
        out: list[int] = []
        for sub_step in range(epoch_len):
            step = epoch * epoch_len + sub_step
            out.extend(
                minibatch_for_step(
                    n_pairs,
                    step=step,
                    minibatch_size=minibatch_size,
                    group_seed="seed-reshuffle",
                )
            )
        return out

    order_epoch0 = epoch_order(0)
    order_epoch1 = epoch_order(1)

    # Both epochs are full permutations of range(n_pairs)...
    assert sorted(order_epoch0) == list(range(n_pairs))
    assert sorted(order_epoch1) == list(range(n_pairs))
    # ...but the permutation differs between epochs (a reshuffle happened).
    assert order_epoch0 != order_epoch1
    # ...and each epoch order is itself deterministic.
    assert epoch_order(0) == order_epoch0
    assert epoch_order(1) == order_epoch1


def test_minibatch_for_step_size_geq_npairs_returns_all_every_step() -> None:
    n_pairs = 4
    for step in range(10):
        batch = minibatch_for_step(
            n_pairs, step=step, minibatch_size=10, group_seed="seed-all"
        )
        assert sorted(batch) == list(range(n_pairs))


def test_minibatch_for_step_size_eq_npairs_returns_all_every_step() -> None:
    n_pairs = 5
    for step in range(7):
        batch = minibatch_for_step(
            n_pairs, step=step, minibatch_size=5, group_seed="seed-eq"
        )
        assert sorted(batch) == list(range(n_pairs))


def test_minibatch_for_step_zero_pairs_returns_empty() -> None:
    assert minibatch_for_step(0, step=0, minibatch_size=3, group_seed="seed") == []
    assert minibatch_for_step(0, step=5, minibatch_size=1, group_seed="seed") == []


_INVALID_NON_NEGATIVE_INT = st.one_of(
    st.integers(max_value=-1),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)
_INVALID_POSITIVE_INT = st.one_of(
    st.integers(max_value=0),
    st.booleans(),
    st.floats(allow_nan=True, allow_infinity=True),
    st.text(),
    st.none(),
)


@settings(max_examples=500, derandomize=True)
@given(
    concept_id=st.integers(min_value=0),
    mode=st.text(min_size=1),
    method_name=st.text(min_size=1),
    target_model=st.text(min_size=1),
    coefficient_str=st.text(min_size=1),
    adversarial_method=st.text(min_size=1),
    strength_index=st.integers(min_value=0),
)
def test_group_key_normalizes_every_valid_field_to_nominal_domains(
    concept_id: int,
    mode: str,
    method_name: str,
    target_model: str,
    coefficient_str: str,
    adversarial_method: str,
    strength_index: int,
) -> None:
    key = group_key(
        {
            "concept_id": concept_id,
            "mode": mode,
            "method_name": method_name,
            "target_model": target_model,
            "coefficient_str": coefficient_str,
            "adversarial_method": adversarial_method,
            "strength_index": strength_index,
        }
    )

    assert_type(key, ConceptGroupKey)
    assert_type(key[0], NonNegativeInt)
    assert_type(key[1], NonEmptyStr)
    assert_type(key[6], NonNegativeInt)
    assert key == (
        concept_id,
        mode,
        method_name,
        target_model,
        coefficient_str,
        adversarial_method,
        strength_index,
    )


@settings(max_examples=500, derandomize=True)
@given(
    field=st.sampled_from(("concept_id", "strength_index")),
    value=_INVALID_NON_NEGATIVE_INT,
)
def test_group_key_rejects_every_invalid_integer_field_contextually(
    field: str,
    value: object,
) -> None:
    row = _attack_row(concept_id=1)
    row[field] = value
    with pytest.raises(AxBenchBpbAdvError, match=field):
        group_key(row)


@settings(max_examples=500, derandomize=True)
@given(
    field=st.sampled_from(
        (
            "mode",
            "method_name",
            "target_model",
            "coefficient_str",
            "adversarial_method",
        )
    ),
    value=st.one_of(st.just(""), st.integers(), st.booleans(), st.none()),
)
def test_group_key_rejects_every_invalid_string_field_contextually(
    field: str,
    value: object,
) -> None:
    row = _attack_row(concept_id=1)
    row[field] = value
    with pytest.raises(AxBenchBpbAdvError, match=field):
        group_key(row)


@settings(max_examples=500, derandomize=True)
@given(value=_INVALID_NON_NEGATIVE_INT)
def test_concept_groups_rejects_every_invalid_editable_count_contextually(
    value: object,
) -> None:
    row = _attack_row(concept_id=1)
    row["editable_token_count"] = value
    with pytest.raises(AxBenchBpbAdvError, match="editable_token_count"):
        concept_groups([row])


@settings(max_examples=500, derandomize=True)
@given(
    st.one_of(
        st.tuples(st.just("n_pairs"), _INVALID_NON_NEGATIVE_INT),
        st.tuples(st.just("step"), _INVALID_NON_NEGATIVE_INT),
        st.tuples(st.just("minibatch_size"), _INVALID_POSITIVE_INT),
    )
)
def test_minibatch_for_step_rejects_every_invalid_scalar(
    invalid: tuple[str, object],
) -> None:
    field, value = invalid
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        _typed_minibatch_for_step(
            cast(Any, value) if field == "n_pairs" else NonNegativeInt.parse(5),
            step=(
                cast(Any, value)
                if field == "step"
                else NonNegativeInt.parse(0)
            ),
            minibatch_size=(
                cast(Any, value)
                if field == "minibatch_size"
                else PositiveInt.parse(3)
            ),
            group_seed="seed",
        )


def test_minibatch_for_step_seed_changes_order() -> None:
    n_pairs = 8
    minibatch_size = 8  # whole epoch in one step -> permutation visible directly
    order_a = minibatch_for_step(
        n_pairs, step=0, minibatch_size=minibatch_size, group_seed="seed-X"
    )
    order_b = minibatch_for_step(
        n_pairs, step=0, minibatch_size=minibatch_size, group_seed="seed-Y"
    )
    assert sorted(order_a) == sorted(order_b) == list(range(n_pairs))
    assert order_a != order_b  # different seed -> different shuffle


def test_minibatch_for_step_indices_within_bounds_and_unique_per_step() -> None:
    n_pairs = 10
    minibatch_size = 4
    epoch_len = math.ceil(n_pairs / minibatch_size)  # 3
    for step in range(20):
        batch = minibatch_for_step(
            n_pairs, step=step, minibatch_size=minibatch_size, group_seed="seed-bnd"
        )
        assert all(0 <= index < n_pairs for index in batch)
        assert len(batch) == len(set(batch))  # no repeat within a single minibatch
        # Non-final steps deal a full minibatch; the final step of an epoch deals
        # the (possibly ragged) remainder. Either way it is non-empty and never
        # exceeds minibatch_size.
        offset = (step % epoch_len) * minibatch_size
        expected = min(minibatch_size, n_pairs - offset)
        assert len(batch) == expected
        assert 1 <= len(batch) <= minibatch_size


# ---------------------------------------------------------------------------
# init_trigger
# ---------------------------------------------------------------------------


def test_init_trigger_returns_last_k_of_smallest_core_prefix() -> None:
    k = 3
    # The smallest core_prefix_id (lexicographically) owns these tokens; its
    # last-k span is the trigger seed.
    small_tokens = [0, 21, 22, 23, 24, 25]
    big_tokens = [0, 91, 92, 93, 94, 95]
    rows = [
        _attack_row(
            concept_id=1,
            editable_token_count=k,
            core_prefix_id="coreprefixZZZZZZZZ",
            source_prompt_tokens=big_tokens,
        ),
        _attack_row(
            concept_id=1,
            editable_token_count=k,
            core_prefix_id="coreprefixAAAAAAAA",
            source_prompt_tokens=small_tokens,
        ),
    ]
    (group,) = concept_groups(rows)

    trigger = init_trigger(group)

    span = select_final_k_editable_span(small_tokens, k=k)
    assert trigger == small_tokens[span.start : span.stop]
    assert trigger == [23, 24, 25]
    assert len(trigger) == k


def test_init_trigger_is_deterministic() -> None:
    rows = _grid_rows()
    groups = concept_groups(rows)
    # Pick a representative non-control group.
    group = groups[0]
    assert not group.is_control
    assert init_trigger(group) == init_trigger(group)


def test_init_trigger_single_row_equals_that_rows_last_k_span() -> None:
    # 1-pair anchor for the future per-prefix-equivalence claim: with exactly one
    # row, the universal trigger seed IS that row's last-k source span.
    k = 2
    tokens = [0, 31, 32, 33, 34]
    rows = [
        _attack_row(
            concept_id=3,
            editable_token_count=k,
            core_prefix_id="coreprefix000042",
            source_prompt_tokens=tokens,
        )
    ]
    (group,) = concept_groups(rows)
    assert len(group.rows) == 1

    span = select_final_k_editable_span(tokens, k=k)
    assert init_trigger(group) == tokens[span.start : span.stop]
    assert init_trigger(group) == [33, 34]


def test_init_trigger_control_group_raises() -> None:
    rows = [
        _attack_row(
            concept_id=1,
            adversarial_method="faster_gcg",
            editable_token_count=0,
        )
    ]
    (group,) = concept_groups(rows)
    assert group.is_control

    with pytest.raises(AxBenchBpbAdvError, match="control"):
        init_trigger(group)


def test_init_trigger_length_matches_k() -> None:
    k = 4
    tokens = [0, 41, 42, 43, 44, 45, 46]
    rows = [
        _attack_row(
            concept_id=5,
            editable_token_count=k,
            source_prompt_tokens=tokens,
            core_prefix_id="coreprefix000007",
        )
    ]
    (group,) = concept_groups(rows)
    assert len(init_trigger(group)) == k


# ---------------------------------------------------------------------------
# aggregate_trigger_gradient
# ---------------------------------------------------------------------------


def test_aggregate_trigger_gradient_means_last_k_slices() -> None:
    """Three per-pair grads of DIFFERENT prompt_len, each aligned by its OWN
    last-k span (NOT an absolute index), then mean over the per-pair (k,D) blocks.

    Lengths 5, 7, 9 with k=2, D=3. Each row's value at position p is encoded as
    p so the slices are trivially hand-checkable, then offset per pair so the mean
    is a non-degenerate arithmetic average (not three identical blocks).
    """

    k = 2
    d = 3

    def ramp(prompt_len: int, *, offset: float) -> t.Tensor:
        # grad[p, :] = p + offset (broadcast across D) -> last-k rows are the two
        # largest positions, distinct per length.
        base = t.arange(prompt_len, dtype=t.float64).reshape(prompt_len, 1)
        return (base + offset).expand(prompt_len, d).contiguous()

    grad_a = ramp(5, offset=0.0)  # last-2 rows: positions 3,4 -> 3,4
    grad_b = ramp(7, offset=10.0)  # last-2 rows: positions 5,6 -> 15,16
    grad_c = ramp(9, offset=100.0)  # last-2 rows: positions 7,8 -> 107,108

    span_a = select_final_k_editable_span([0] * 5, k=k)  # start=3, stop=5
    span_b = select_final_k_editable_span([0] * 7, k=k)  # start=5, stop=7
    span_c = select_final_k_editable_span([0] * 9, k=k)  # start=7, stop=9

    out = aggregate_trigger_gradient(
        [(grad_a, span_a), (grad_b, span_b), (grad_c, span_c)]
    )

    # Hand-computed mean of the three (2,3) last-k slices:
    #   row0 = mean(3, 15, 107) = 125/3
    #   row1 = mean(4, 16, 108) = 128/3
    expected = t.tensor(
        [[125.0 / 3.0] * d, [128.0 / 3.0] * d],
        dtype=t.float64,
    )
    assert out.shape == (k, d)
    assert t.allclose(out, expected)


def test_aggregate_trigger_gradient_single_pair_is_identity() -> None:
    """One pair -> exactly its own last-k slice (the 1-pair equivalence anchor)."""

    k = 2
    d = 4
    grad = t.arange(7 * d, dtype=t.float64).reshape(7, d)  # prompt_len 7
    span = select_final_k_editable_span([0] * 7, k=k)  # start=5, stop=7

    out = aggregate_trigger_gradient([(grad, span)])

    expected = grad[span.start : span.stop]
    assert out.shape == (k, d)
    assert t.equal(out, expected)


def test_aggregate_trigger_gradient_empty_raises() -> None:
    with pytest.raises(AxBenchBpbAdvError):
        aggregate_trigger_gradient([])


def test_aggregate_trigger_gradient_inconsistent_k_raises() -> None:
    # Two pairs whose last-k slices disagree in width cannot be stacked/meaned.
    grad_a = t.zeros(5, 3, dtype=t.float64)
    grad_b = t.zeros(6, 3, dtype=t.float64)
    span_a = EditableSpan(start=3, count=2)
    span_b = EditableSpan(start=3, count=3)
    with pytest.raises(AxBenchBpbAdvError):
        aggregate_trigger_gradient([(grad_a, span_a), (grad_b, span_b)])


def test_full_batch_float32_order_can_change_the_top_ranked_token() -> None:
    """Freeze the counterexample that makes fixed-pool reuse a new algorithm.

    A full-batch deal contains the same six members at every step, but
    ``minibatch_for_step`` reshuffles their order and float32 ``stack().mean()``
    is order-sensitive under cancellation. The resulting gradient changes sign
    between steps 0 and 12, which flips the top-ranked replacement. Therefore
    holding the first pool fixed is deliberate replacement semantics, not a
    legacy-output-preserving early exit.
    """

    values = [1.0e20, 3.0, -1.0e20, 1.0, -1.0, -3.0]
    gradients = [t.tensor([[value]], dtype=t.float32) for value in values]
    span = EditableSpan(start=0, count=1)

    def aggregate_at(step: int) -> t.Tensor:
        order = minibatch_for_step(
            6,
            step=step,
            minibatch_size=16,
            group_seed="counterexample|seed0",
        )
        assert sorted(order) == list(range(6))
        return aggregate_trigger_gradient([(gradients[index], span) for index in order])

    aggregate_0 = aggregate_at(0)
    aggregate_12 = aggregate_at(12)
    assert aggregate_0.item() > 0.0
    assert aggregate_12.item() < 0.0

    embedding = t.tensor([[0.0], [-1.0], [1.0], [0.0]], dtype=t.float32)
    ranked_0 = rank_trigger_replacements(
        trigger_gradient=aggregate_0,
        embedding_weight=embedding,
        current_trigger=[3],
        allowed_token_ids=[1, 2],
        top_k=1,
        distance_weight=0.0,
        bos_token_id=0,
    )
    ranked_12 = rank_trigger_replacements(
        trigger_gradient=aggregate_12,
        embedding_weight=embedding,
        current_trigger=[3],
        allowed_token_ids=[1, 2],
        top_k=1,
        distance_weight=0.0,
        bos_token_id=0,
    )
    assert ranked_0[0][0].token_id == 1
    assert ranked_12[0][0].token_id == 2


# ---------------------------------------------------------------------------
# rank_trigger_replacements (BOS-padded view; BOS carries no signal)
# ---------------------------------------------------------------------------


def _toy_embedding(vocab: int, d: int) -> t.Tensor:
    """A deterministic, well-spread (V, D) embedding table for ranking math."""

    return t.arange(vocab * d, dtype=t.float64).reshape(vocab, d) * 0.1 - 3.0


def test_rank_trigger_replacements_matches_full_prompt_ranking() -> None:
    """Trigger-local ranking (BOS-padded) == full-prompt ranking at the absolute
    last-k positions, for a single prefix. This is the math underlying the 1-pair
    equivalence: same TokenReplacement token_ids AND scores, only re-indexed.
    """

    d = 3
    vocab = 9
    k = 2
    bos_token_id = 0
    top_k = 4
    distance_weight = 1.0
    allowed_token_ids = [1, 2, 3, 4, 5, 6, 7, 8]

    embedding = _toy_embedding(vocab, d)
    trigger = [5, 6]

    # Build a full BOS-prefixed prompt whose LAST-k tokens are exactly the trigger
    # and whose last-k gradient rows are exactly the trigger gradient. The prefix
    # (positions 1..start-1) holds arbitrary distinct tokens/gradient that must NOT
    # affect the last-k ranking.
    full_tokens = [bos_token_id, 1, 2, *trigger]  # prompt_len 5, last-2 == trigger
    start = len(full_tokens) - k  # 3
    full_grad = t.tensor(
        [
            [9.0, 9.0, 9.0],  # BOS row (ignored by guard)
            [8.0, -8.0, 8.0],  # prefix row (ignored for last-k ranking)
            [7.0, 7.0, -7.0],  # prefix row
            [0.5, -1.5, 2.0],  # trigger position 0
            [-2.0, 3.0, 1.0],  # trigger position 1
        ],
        dtype=t.float64,
    )
    trigger_grad = full_grad[start : start + k].clone()

    full_ranked = rank_replacements_from_gradients(
        embedding_gradient=full_grad,
        embedding_weight=embedding,
        editable_positions=list(range(start, start + k)),
        allowed_token_ids=allowed_token_ids,
        current_tokens=full_tokens,
        top_k=top_k,
        distance_weight=distance_weight,
    )

    trigger_ranked = rank_trigger_replacements(
        trigger_gradient=trigger_grad,
        embedding_weight=embedding,
        current_trigger=trigger,
        allowed_token_ids=allowed_token_ids,
        top_k=top_k,
        distance_weight=distance_weight,
        bos_token_id=bos_token_id,
    )

    # Returned dict is keyed by TRIGGER-LOCAL positions 0..k-1.
    assert sorted(trigger_ranked.keys()) == list(range(k))

    for local_pos in range(k):
        absolute_pos = start + local_pos
        local_repls = trigger_ranked[local_pos]
        full_repls = full_ranked[absolute_pos]
        # Same token ids AND same scores, in the same order.
        assert [r.token_id for r in local_repls] == [
            r.token_id for r in full_repls
        ]
        local_scores = [r.score for r in local_repls]
        full_scores = [r.score for r in full_repls]
        assert local_scores == pytest.approx(full_scores, abs=0.0, rel=0.0)
        # The TokenReplacement.position is re-indexed to the trigger-local coord.
        assert all(r.position == local_pos for r in local_repls)


def test_rank_trigger_bos_carries_no_signal() -> None:
    """The padded BOS row/position is never ranked or proposed as a replacement.

    Make the BOS gradient row enormous; if it leaked into ranking it would show
    up as a position-0 entry or dominate. Only trigger-local positions 0..k-1 are
    returned, and the BOS token id is never proposed (it is not editable).
    """

    d = 3
    vocab = 7
    bos_token_id = 0
    embedding = _toy_embedding(vocab, d)
    trigger = [4, 5, 6]
    trigger_grad = t.tensor(
        [
            [0.1, -0.2, 0.3],
            [-0.4, 0.5, -0.6],
            [0.7, -0.8, 0.9],
        ],
        dtype=t.float64,
    )

    ranked = rank_trigger_replacements(
        trigger_gradient=trigger_grad,
        embedding_weight=embedding,
        current_trigger=trigger,
        allowed_token_ids=[1, 2, 3, 4, 5, 6],
        top_k=5,
        distance_weight=0.0,
        bos_token_id=bos_token_id,
    )

    # No BOS position (the design's "BOS row is never ranked").
    assert set(ranked.keys()) == {0, 1, 2}
    assert -1 not in ranked
    # The result is invariant to whatever the BOS gradient row would have been:
    # build an explicitly different padded BOS row and confirm the trigger-local
    # ranking is unchanged (BOS truly carries no signal). We do this by ranking a
    # second time -- the function pads zeros internally, so the only way the BOS
    # row could matter is if it were read, which the position guard forbids.
    ranked_again = rank_trigger_replacements(
        trigger_gradient=trigger_grad.clone(),
        embedding_weight=embedding,
        current_trigger=list(trigger),
        allowed_token_ids=[1, 2, 3, 4, 5, 6],
        top_k=5,
        distance_weight=0.0,
        bos_token_id=bos_token_id,
    )
    assert ranked == ranked_again


def test_rank_trigger_replacements_score_scale_invariant_to_minibatch_size() -> None:
    """MEAN aggregation keeps the proposal-score scale comparable across minibatch
    sizes for a FIXED pair set: averaging N identical per-pair (k,D) gradients
    yields the same trigger gradient as N=1, so the ranked scores are identical
    regardless of how many (identical) pairs the mean was taken over.

    This is the documented rationale for MEAN (not SUM): a sum would scale scores
    by N and make the sampling softmax `temperature` N-dependent.
    """

    d = 3
    vocab = 8
    k = 2
    bos_token_id = 0
    embedding = _toy_embedding(vocab, d)
    trigger = [3, 4]
    base_grad = t.tensor([[0.5, -1.0, 2.0], [-2.0, 1.5, 0.25]], dtype=t.float64)
    span = EditableSpan(start=1, count=k)  # BOS-padded view positions 1..k

    def grad_full(offset: int) -> t.Tensor:
        # A (1+k, D) tensor whose last-k slice is base_grad; the BOS pad slot is
        # absorbed by aggregate via the span, so reuse a (k,D)-bearing full tensor.
        full = t.zeros(1 + k, d, dtype=t.float64)
        full[1 : 1 + k] = base_grad
        return full

    # Aggregate the SAME pair gradient over minibatches of size 1 and size 4.
    grad_small = aggregate_trigger_gradient([(grad_full(0), span)])
    grad_large = aggregate_trigger_gradient(
        [(grad_full(0), span) for _ in range(4)]
    )
    assert t.allclose(grad_small, grad_large)

    ranked_small = rank_trigger_replacements(
        trigger_gradient=grad_small,
        embedding_weight=embedding,
        current_trigger=trigger,
        allowed_token_ids=[1, 2, 3, 4, 5, 6, 7],
        top_k=4,
        distance_weight=0.0,
        bos_token_id=bos_token_id,
    )
    ranked_large = rank_trigger_replacements(
        trigger_gradient=grad_large,
        embedding_weight=embedding,
        current_trigger=trigger,
        allowed_token_ids=[1, 2, 3, 4, 5, 6, 7],
        top_k=4,
        distance_weight=0.0,
        bos_token_id=bos_token_id,
    )
    # Identical token ids AND identical scores -> identical softmax weights.
    assert ranked_small == ranked_large


def test_rank_trigger_replacements_gradient_must_be_2d() -> None:
    embedding = _toy_embedding(6, 3)
    with pytest.raises(AxBenchBpbAdvError):
        rank_trigger_replacements(
            trigger_gradient=t.zeros(2, 3, 1, dtype=t.float64),
            embedding_weight=embedding,
            current_trigger=[1, 2],
            allowed_token_ids=[1, 2, 3],
            top_k=2,
            distance_weight=0.0,
            bos_token_id=0,
        )


def test_rank_trigger_replacements_length_mismatch_raises() -> None:
    embedding = _toy_embedding(6, 3)
    with pytest.raises(AxBenchBpbAdvError):
        rank_trigger_replacements(
            trigger_gradient=t.zeros(3, 3, dtype=t.float64),  # k=3
            embedding_weight=embedding,
            current_trigger=[1, 2],  # k=2 -> mismatch
            allowed_token_ids=[1, 2, 3],
            top_k=2,
            distance_weight=0.0,
            bos_token_id=0,
        )


def test_rank_trigger_replacements_bos_must_be_in_vocab() -> None:
    embedding = _toy_embedding(6, 3)
    with pytest.raises(AxBenchBpbAdvError):
        rank_trigger_replacements(
            trigger_gradient=t.zeros(2, 3, dtype=t.float64),
            embedding_weight=embedding,
            current_trigger=[1, 2],
            allowed_token_ids=[1, 2, 3],
            top_k=2,
            distance_weight=0.0,
            bos_token_id=999,  # outside [0, vocab)
        )


def test_rank_trigger_replacements_rejects_double_bos_trigger_view() -> None:
    embedding = _toy_embedding(6, 3)
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        rank_trigger_replacements(
            trigger_gradient=t.zeros(2, 3, dtype=t.float64),
            embedding_weight=embedding,
            current_trigger=[0, 2],
            allowed_token_ids=[1, 2, 3],
            top_k=2,
            distance_weight=0.0,
            bos_token_id=0,
        )


def test_rank_trigger_replacements_rejects_non_adjacent_trigger_bos() -> None:
    embedding = _toy_embedding(6, 3)
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        rank_trigger_replacements(
            trigger_gradient=t.zeros(2, 3, dtype=t.float64),
            embedding_weight=embedding,
            current_trigger=[2, 0],
            allowed_token_ids=[1, 2, 3],
            top_k=2,
            distance_weight=0.0,
            bos_token_id=0,
        )


def test_padded_trigger_tokens_rejects_double_bos() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        _padded_trigger_tokens(
            trigger=[0, 4],
            bos_token_id=0,
            context="test trigger",
        )


def test_padded_trigger_tokens_rejects_non_adjacent_bos() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        _padded_trigger_tokens(
            trigger=[4, 0],
            bos_token_id=0,
            context="test trigger",
        )


def test_strip_padded_trigger_tokens_rejects_double_bos_candidate_view() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        _strip_padded_trigger_tokens(
            [0, 0, 4],
            bos_token_id=0,
            context="test candidate",
        )


def test_strip_padded_trigger_tokens_rejects_non_adjacent_bos() -> None:
    with pytest.raises(AxBenchBpbAdvError, match="double BOS"):
        _strip_padded_trigger_tokens(
            [0, 4, 0],
            bos_token_id=0,
            context="test candidate",
        )


# ---------------------------------------------------------------------------
# aggregate_objective
# ---------------------------------------------------------------------------


def test_aggregate_objective_is_mean_of_pair_g() -> None:
    assert aggregate_objective([1.0, 2.0, 3.0]) == pytest.approx(2.0)
    assert aggregate_objective([-1.0, 1.0]) == pytest.approx(0.0)
    assert aggregate_objective([0.5, 0.25, 0.25, 1.0]) == pytest.approx(0.5)


def test_aggregate_objective_single_pair_is_identity() -> None:
    # 1-pair anchor: the aggregate IS that pair's objective (the accept compare
    # then reduces to the per-prefix choose_best rule).
    assert aggregate_objective([0.123456789]) == 0.123456789
    assert aggregate_objective([-42.0]) == -42.0


def test_aggregate_objective_empty_raises() -> None:
    with pytest.raises(FeatureGcgError):
        aggregate_objective([])


def test_aggregate_objective_non_finite_propagates_positive_inf() -> None:
    """Finiteness guard (v2/m10): a non-finite g must yield a well-defined result
    that the accept compare (best_obj < cur_obj) REJECTS. We propagate +inf so a
    NaN/inf/-inf in any pair makes the aggregate +inf (never accepted), rather
    than letting a NaN poison the mean into a NaN that compares False both ways.
    """

    assert aggregate_objective([1.0, math.nan, 2.0]) == math.inf
    assert aggregate_objective([1.0, math.inf, 2.0]) == math.inf
    # -inf is also non-finite and must be rejected (not treated as "best").
    assert aggregate_objective([1.0, -math.inf, 2.0]) == math.inf
    # A +inf aggregate compares False under `<`, i.e. never accepted.
    assert not (aggregate_objective([math.nan]) < 0.0)
    assert not (aggregate_objective([math.nan]) < math.inf)


# ---------------------------------------------------------------------------
# Phase 4: universal_triggers artifact (schema + _trigger_row + path)
# ---------------------------------------------------------------------------


class _DecodeTokenizer:
    """Tiny tokenizer: decode == ascii-join, BOS == 0 (the rank/decode pad)."""

    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids)


def _trigger_group(
    *,
    concept_id: int = 7,
    mode: str = "positive",
    method_name: str = "diff_mean",
    target_model: str = "d12_base",
    coefficient_str: str = "8.0",
    strength_index: int = 2,
    n_pairs: int = 3,
) -> ConceptGroup:
    """A concept group of ``n_pairs`` faster_gcg rows sharing the 7-tuple.

    Built through ``concept_groups`` (the production grouper) so the group's
    ``key``/``key_str``/``editable_token_count`` are exactly what the optimizer
    sees. ``strength_index == k == editable_token_count`` (the attack-row
    convention), so the group's adversarial_strength is ``k``.
    """

    rows = [
        _attack_row(
            concept_id=concept_id,
            mode=mode,
            method_name=method_name,
            target_model=target_model,
            coefficient_str=coefficient_str,
            strength_index=strength_index,
            editable_token_count=strength_index,
            core_prefix_id=f"coreprefix{pair_idx:010d}",
            concept_rich_sample_id=f"rich-{pair_idx}",
            concept_avoiding_sample_id=f"avoid-{pair_idx}",
        )
        for pair_idx in range(n_pairs)
    ]
    # Attack rows carry no adversarial_strength column in the test fixture; the
    # _trigger_row builder reads it from the row, so stamp it = k on each.
    for row in rows:
        row["adversarial_strength"] = float(strength_index)
    groups = concept_groups(rows)
    assert len(groups) == 1
    return groups[0]


def test_trigger_row_matches_universal_triggers_schema_columns() -> None:
    """``_trigger_row`` emits exactly UNIVERSAL_TRIGGERS_SCHEMA's columns/types.

    A single dict built from a real concept group must round-trip through
    ``pa.Table.from_pylist(..., schema=UNIVERSAL_TRIGGERS_SCHEMA)`` with no missing
    / extra / mistyped field (from_pylist raises on a schema mismatch).
    """

    group = _trigger_group(strength_index=2, n_pairs=3)
    row = _trigger_row(
        group=group,
        tokenizer=_DecodeTokenizer(),
        trigger=[65, 66],
        objective_value=0.5,
        max_steps=4,
        accepted_steps=2,
        visited_sequences=5,
    )
    # Exactly the schema's field names, in any order.
    assert set(row) == set(UNIVERSAL_TRIGGERS_SCHEMA.names)
    # Round-trips through the schema (raises on a column/type mismatch).
    table = pa.Table.from_pylist([row], schema=UNIVERSAL_TRIGGERS_SCHEMA)
    assert table.schema.equals(UNIVERSAL_TRIGGERS_SCHEMA)
    assert table.num_rows == 1
    back = table.to_pylist()[0]
    # The group identity, trigger, and metadata are carried verbatim.
    assert back["concept_id"] == 7
    assert back["mode"] == "positive"
    assert back["method_name"] == "diff_mean"
    assert back["target_model"] == "d12_base"
    assert back["coefficient_str"] == "8.0"
    assert back["adversarial_method"] == "faster_gcg"
    assert back["strength_index"] == 2
    assert back["adversarial_strength"] == 2.0
    assert back["trigger_tokens"] == [65, 66]
    # trigger_text == tokenizer.decode(trigger_tokens).
    assert back["trigger_text"] == "AB"
    assert back["objective_value"] == 0.5
    assert back["max_steps"] == 4
    assert back["accepted_steps"] == 2
    assert back["visited_sequences"] == 5
    assert back["n_pairs"] == 3


def test_trigger_row_text_is_tokenizer_decode_of_trigger_tokens() -> None:
    """trigger_text is the decode of trigger_tokens (not the source / replacement
    text) — this is what general_val / the samples YAML surface as the human view.
    """

    group = _trigger_group(strength_index=3, n_pairs=1)
    row = _trigger_row(
        group=group,
        tokenizer=_DecodeTokenizer(),
        trigger=[72, 73, 33],  # "HI!"
        objective_value=1.0,
        max_steps=2,
        accepted_steps=0,
        visited_sequences=1,
    )
    assert row["trigger_tokens"] == [72, 73, 33]
    assert row["trigger_text"] == "HI!"
    assert row["n_pairs"] == 1


def test_universal_triggers_schema_roundtrip_atomic_write(tmp_path) -> None:
    """A multi-group trigger table (one row per group) survives an atomic
    parquet write + read-back byte-for-byte equal, with the canonical schema.
    """

    groups = [
        _trigger_group(concept_id=1, strength_index=1, n_pairs=2),
        _trigger_group(concept_id=2, strength_index=2, n_pairs=3),
    ]
    rows = [
        _trigger_row(
            group=group,
            tokenizer=_DecodeTokenizer(),
            trigger=[65] * group.editable_token_count,
            objective_value=float(index),
            max_steps=8,
            accepted_steps=index,
            visited_sequences=index + 1,
        )
        for index, group in enumerate(groups)
    ]
    table = pa.Table.from_pylist(rows, schema=UNIVERSAL_TRIGGERS_SCHEMA)
    out = tmp_path / "universal_triggers.parquet"
    atomic_write_parquet(table, out)
    assert out.exists()
    read_back = pq.read_table(out)
    assert read_back.schema.equals(UNIVERSAL_TRIGGERS_SCHEMA)
    assert read_back.equals(table)
    # One row per group, concept ids preserved.
    assert read_back.column("concept_id").to_pylist() == [1, 2]
    assert read_back.column("n_pairs").to_pylist() == [2, 3]


def test_empty_universal_triggers_serializes_schema_stable(tmp_path) -> None:
    """A zero-row trigger set (no non-control faster_gcg groups) still writes a
    schema-stable table (the candidate stage always writes the artifact).
    """

    table = pa.Table.from_pylist([], schema=UNIVERSAL_TRIGGERS_SCHEMA)
    out = tmp_path / "universal_triggers.parquet"
    atomic_write_parquet(table, out)
    read_back = pq.read_table(out)
    assert read_back.schema.equals(UNIVERSAL_TRIGGERS_SCHEMA)
    assert read_back.num_rows == 0


def test_universal_triggers_path_is_under_root() -> None:
    """``paths.universal_triggers_path()`` is ``<root>/universal_triggers.parquet``."""

    paths = AxBenchBpbAdvPaths(output_root="/tmp/out", run_tag="run")
    assert (
        paths.universal_triggers_path()
        == paths.root / "universal_triggers.parquet"
    )
    assert not (aggregate_objective([math.nan]) < math.inf)
