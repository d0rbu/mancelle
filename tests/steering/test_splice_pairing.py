"""Independent numerical and graph contracts for bounded opposite pairing."""

from __future__ import annotations

import math
import random
from bisect import bisect_right
from dataclasses import replace

import numpy as np
import pytest
import torch as t
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.conditional.percentile_table as percentile_mod
import steering.feature_shard_pairing as pairing_mod
from steering._roundtrip_seam import is_roundtrip_stable
from steering.conditional.percentile_table import compute_percentile_checksum
from steering.feature_shard_pairing import (
    OppositePercentilePairing,
    _percentile_rows,
    _shared_minimum_matrix,
)
from steering.feature_shard_splice import (
    SourceSegment,
    _draw_derangement,
    _piece_ids,
    load_source_segments,
    recombine_segments,
)
from tests.steering._splice_fixtures import make_base_shard
from tests.steering._splice_test_tokenizers import CharMergeTokenizer


def _table(width: int = 8) -> t.Tensor:
    return t.arange(1, 100, dtype=t.float64).repeat(width, 1)


def _segment(values: dict[int, float], token: int = 97) -> SourceSegment:
    ids = sorted(values)
    return SourceSegment(
        (token, token),
        t.tensor(ids, dtype=t.int32),
        t.tensor([values[fid] for fid in ids], dtype=t.float64),
    )


def _source_docs(prefixes: list[dict[int, float]], suffixes: list[dict[int, float]]):
    return [
        [_segment(prefix), _segment(suffix)]
        for prefix, suffix in zip(prefixes, suffixes, strict=True)
    ]


def _percentile_scalar(magnitude: float, thresholds: list[float]) -> float:
    """Independent Python bisect oracle, without the production tensor lookup."""
    index = bisect_right(thresholds, magnitude)
    if index == 0:
        return 0.5
    if index == 99:
        return 99.5
    return index + (magnitude - thresholds[index - 1]) / (thresholds[index] - thresholds[index - 1])


def _dense(segment: SourceSegment, table: t.Tensor) -> np.ndarray:
    result = np.zeros(len(table), dtype=np.float64)
    for fid, magnitude in zip(
        segment.feature_ids.tolist(), segment.feature_mags.tolist(), strict=True
    ):
        result[fid] = _percentile_scalar(magnitude, table[fid].tolist())
    return result


def _l1(left: np.ndarray, right: np.ndarray) -> float:
    return math.fsum(abs(float(a) - float(b)) for a, b in zip(left, right, strict=True))


def _oracle(source, cuts, sigma, table, pool_size):
    """Scalar L1 swap search, independent of the sparse overlap algorithm."""
    result = list(sigma)
    for start in range(0, len(cuts), pool_size):
        stop = min(start + pool_size, len(cuts))
        for source_slot in range(start, stop):
            source_ref = cuts[source_slot]
            left = _dense(source[source_ref[0]][source_ref[1] + 1], table)
            target_ref = cuts[result[source_slot]]
            right = _dense(source[target_ref[0]][target_ref[1] + 1], table)
            best_gain = 0.0
            best = None
            for partner in range(start, stop):
                if (
                    partner == source_slot
                    or result[partner] == source_slot
                    or result[source_slot] == partner
                ):
                    continue
                partner_ref = cuts[partner]
                partner_left = _dense(source[partner_ref[0]][partner_ref[1] + 1], table)
                partner_target = cuts[result[partner]]
                partner_right = _dense(source[partner_target[0]][partner_target[1] + 1], table)
                gain = (
                    _l1(left, partner_right)
                    + _l1(partner_left, right)
                    - (_l1(left, right) + _l1(partner_left, partner_right))
                )
                if gain > best_gain:
                    best_gain, best = gain, partner
            if best is not None:
                result[source_slot], result[best] = result[best], result[source_slot]
    return result


def _total_l1(source, cuts, sigma, table):
    return math.fsum(
        _l1(
            _dense(source[d][b + 1], table),
            _dense(source[cuts[sigma[i]][0]][cuts[sigma[i]][1] + 1], table),
        )
        for i, (d, b) in enumerate(cuts)
    )


def test_sparse_overlap_reconstructs_dense_l1_and_clears_workspace():
    table = _table()
    table[1] = t.arange(1, 34, dtype=t.float64).repeat_interleave(3)
    prefixes = [_segment({0: 0.25, 1: 5.25}), _segment({1: 4.3, 7: 100.0})]
    suffixes = [_segment({0: 50.0, 1: 7.0}), _segment({4: 1.1}), _segment({1: 4.3, 7: 100.0})]
    left = _percentile_rows(prefixes, table)
    right = _percentile_rows(suffixes, table)
    scratch = np.zeros(len(table))
    overlap = _shared_minimum_matrix(left, right, scratch)
    for i, prefix in enumerate(prefixes):
        a = _dense(prefix, table)
        for j, suffix in enumerate(suffixes):
            b = _dense(suffix, table)
            actual = a.sum() + b.sum() - 2 * overlap[i, j]
            assert actual == pytest.approx(_l1(a, b), abs=1e-12)
    np.testing.assert_array_equal(scratch, np.zeros(len(table)))


def test_uses_displaced_continuation_not_identical_left_of_cut_segments():
    # Adjacent-segment scoring cannot improve this assignment: every prefix has
    # the same vector. The displaced continuations form two distinct groups.
    source = _source_docs([{2: 50.0}] * 4, [{0: 50.0}, {0: 50.0}, {1: 50.0}, {1: 50.0}])
    cuts = [(i, 0) for i in range(4)]
    initial = [1, 0, 3, 2]
    table = _table()
    result = OppositePercentilePairing(table, 4).improve_pairing(source, cuts, initial)
    assert result == [3, 2, 1, 0]
    assert _total_l1(source, cuts, initial, table) == 0.0
    assert _total_l1(source, cuts, result, table) == 400.0
    assert initial == [1, 0, 3, 2]
    assert cuts == [(i, 0) for i in range(4)]


@given(
    seed=st.integers(min_value=0, max_value=2**32 - 1),
    size=st.integers(min_value=2, max_value=14),
    pool_size=st.integers(min_value=2, max_value=16),
)
@settings(max_examples=60, deadline=None)
def test_exact_scalar_l1_oracle_and_bijection(seed, size, pool_size):
    rng = random.Random(seed)
    table = _table()

    def row():
        return {
            fid: rng.choice([0.25, 25.0, 50.0, 75.0, 100.0])
            for fid in rng.sample(range(8), rng.randint(1, 8))
        }

    source = _source_docs([row() for _ in range(size)], [row() for _ in range(size)])
    cuts = [(i, 0) for i in range(size)]
    sigma = _draw_derangement(rng, size)
    pairing = OppositePercentilePairing(table, pool_size)
    actual = pairing.improve_pairing(source, cuts, sigma)
    expected = _oracle(source, cuts, sigma, table, pool_size)
    assert actual == expected
    assert sorted(actual) == list(range(size))
    assert all(i != j for i, j in enumerate(actual))
    assert _total_l1(source, cuts, actual, table) >= _total_l1(source, cuts, sigma, table)
    assert actual == pairing.improve_pairing(source, cuts, sigma)
    for start in range(0, size, pool_size):
        assert sorted(actual[start : start + pool_size]) == sorted(sigma[start : start + pool_size])


@pytest.mark.parametrize("sigma", [[1, 0], [1, 2, 0]])
def test_two_and_three_cut_derangements_cannot_be_changed_by_a_valid_swap(sigma):
    source = _source_docs([{0: 50.0}] * len(sigma), [{1: 50.0}] * len(sigma))
    cuts = [(i, 0) for i in range(len(sigma))]
    assert OppositePercentilePairing(_table()).improve_pairing(source, cuts, sigma) == sigma


@pytest.mark.parametrize("same_features", [True, False])
def test_equal_scores_leave_assignments_unchanged(same_features):
    source = _source_docs([{0: 50.0}] * 7, [{0 if same_features else 1: 50.0}] * 7)
    sigma = [1, 2, 3, 4, 5, 6, 0]
    assert (
        OppositePercentilePairing(_table(), 3).improve_pairing(
            source, [(i, 0) for i in range(7)], sigma
        )
        == sigma
    )


def test_continuous_zero_gain_ties_preserve_original_derangement():
    # Each feature has the same positive value in exactly three of four docs.
    # In EVERY derangement, its sole absent doc has one incoming and one
    # outgoing edge to present docs. Its total L1 contribution is exactly 2p,
    # independent of the assignment; no swap can improve the objective.
    rng = np.random.default_rng(2)
    weights = rng.uniform(1, 99, 100)
    missing = rng.integers(0, 4, 100)
    continuations = [
        {int(fid): float(weights[fid]) for fid in np.flatnonzero(missing != doc)}
        for doc in range(4)
    ]
    source = _source_docs([{0: 50.0}] * 4, continuations)
    sigma = [1, 0, 3, 2]
    actual = OppositePercentilePairing(_table(100), 4).improve_pairing(
        source, [(i, 0) for i in range(4)], sigma
    )
    assert actual == sigma


def test_continuous_equal_positive_gains_choose_first_candidate():
    # The first 100 features have assignment-independent total L1, as above.
    # Feature 100 alone rewards crossing {0, 1} with {2, 3}: either candidate
    # improves equally. The first candidate (slot 2) must win at the first row,
    # even if reductions of the first 100 features differ by a few ULPs.
    rng = np.random.default_rng(2)
    weights = rng.uniform(1, 99, 100)
    missing = rng.integers(0, 4, 100)
    continuations = [
        {int(fid): float(weights[fid]) for fid in np.flatnonzero(missing != doc)}
        for doc in range(4)
    ]
    continuations[0][100] = 50.0
    continuations[1][100] = 50.0
    source = _source_docs([{0: 50.0}] * 4, continuations)
    cuts = [(i, 0) for i in range(4)]
    sigma = [1, 0, 3, 2]
    table = _table(101)
    actual = OppositePercentilePairing(table, 4).improve_pairing(source, cuts, sigma)
    assert actual == [3, 2, 1, 0]
    assert _total_l1(source, cuts, actual, table) - _total_l1(
        source, cuts, sigma, table
    ) == pytest.approx(200.0)


def test_partial_groups_are_bounded_and_singleton_is_untouched(monkeypatch):
    source = _source_docs([{i % 2: 50.0} for i in range(9)], [{i % 2: 50.0} for i in range(9)])
    sigma = [1, 2, 3, 4, 5, 6, 7, 8, 0]
    observed = []
    original = pairing_mod._shared_minimum_matrix

    def recording(left, right, scratch):
        result = original(left, right, scratch)
        observed.append(result.shape)
        return result

    monkeypatch.setattr(pairing_mod, "_shared_minimum_matrix", recording)
    actual = OppositePercentilePairing(_table(), 4).improve_pairing(
        source, [(i, 0) for i in range(9)], sigma
    )
    assert observed == [(4, 4), (4, 4)]
    assert actual[-1] == sigma[-1]


@pytest.mark.parametrize("trusted", [False, True])
def test_lookup_conversion_is_chunk_bounded(monkeypatch, trusted):
    table = _table()
    segments = [_segment({0: 25.0, 1: 75.0}), _segment({2: 50.0, 3: 100.0})]
    expected = _percentile_rows(segments, table)
    original = pairing_mod._magnitude_to_percentile
    calls = []

    def recording(ids, mags, thresholds, *, table_is_finite=False):
        calls.append((len(ids), table_is_finite))
        return original(ids, mags, thresholds, table_is_finite=table_is_finite)

    monkeypatch.setattr(pairing_mod, "_PERCENTILE_LOOKUP_CHUNK_SIZE", 3)
    monkeypatch.setattr(pairing_mod, "_magnitude_to_percentile", recording)
    kwargs = {"table_is_finite": True} if trusted else {}
    actual = _percentile_rows(segments, table, **kwargs)
    assert calls == [(3, trusted), (1, trusted)]
    np.testing.assert_array_equal(actual.values, expected.values)


@pytest.mark.parametrize("pool_size", [True, False, None, "64", 2.0, 0, 1, -3])
def test_invalid_pool_sizes_raise(pool_size):
    with pytest.raises(ValueError, match="integer >= 2"):
        OppositePercentilePairing(_table(), pool_size)


@pytest.mark.parametrize(
    "kind",
    ["rank", "columns", "empty", "dtype", "device", "nan", "infinity", "negative", "descending"],
)
def test_invalid_tables_raise(kind):
    table = _table()
    if kind == "rank":
        table = table[0]
    elif kind == "columns":
        table = table[:, :98]
    elif kind == "empty":
        table = table[:0]
    elif kind == "dtype":
        table = table.float()
    elif kind == "device":
        table = t.empty((8, 99), dtype=t.float64, device="meta")
    elif kind == "nan":
        table[0, 2] = math.nan
    elif kind == "infinity":
        table[0, -1] = math.inf
    elif kind == "negative":
        table[0, 0] = -1
    else:
        table[0, 3] = 0
    with pytest.raises(ValueError, match="percentile table"):
        OppositePercentilePairing(table)


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "shape",
        "float_ids",
        "integer_mags",
        "duplicate",
        "descending",
        "negative_id",
        "large_id",
        "nan",
        "infinity",
        "zero",
        "negative_mag",
        "device",
    ],
)
def test_invalid_queried_segments_raise(kind):
    segment = _segment({0: 25.0, 1: 75.0})
    if kind == "empty":
        segment = _segment({})
    elif kind == "shape":
        segment = replace(segment, feature_mags=segment.feature_mags[:1])
    elif kind == "float_ids":
        segment = replace(segment, feature_ids=segment.feature_ids.float())
    elif kind == "integer_mags":
        segment = replace(segment, feature_mags=segment.feature_mags.long())
    elif kind == "duplicate":
        segment = replace(segment, feature_ids=t.tensor([0, 0]))
    elif kind == "descending":
        segment = replace(segment, feature_ids=t.tensor([1, 0]))
    elif kind == "negative_id":
        segment = replace(segment, feature_ids=t.tensor([-1, 0]))
    elif kind == "large_id":
        segment = replace(segment, feature_ids=t.tensor([0, 8]))
    elif kind == "device":
        segment = replace(segment, feature_ids=t.empty(2, dtype=t.int64, device="meta"))
    else:
        bad = {"nan": math.nan, "infinity": math.inf, "zero": 0.0, "negative_mag": -1.0}[kind]
        segment = replace(segment, feature_mags=t.tensor([25.0, bad]))
    with pytest.raises(ValueError, match="splice pairing"):
        _percentile_rows([segment], _table())


def test_missing_active_statistics_raise_but_unused_zero_rows_are_legal():
    table = _table()
    table[1] = 0
    pairing = OppositePercentilePairing(table)
    good = _segment({0: 50.0})
    _percentile_rows([good], pairing.percentile_table)
    with pytest.raises(ValueError, match="statistics are missing"):
        _percentile_rows([_segment({1: 50.0})], pairing.percentile_table)


@pytest.mark.parametrize("layout", ["contiguous", "transposed", "strided"])
def test_owned_table_and_provenance_pin_actual_statistics(layout):
    table = _table()
    if layout == "transposed":
        table = table.t().contiguous().t()
    elif layout == "strided":
        table = _table(16)[::2]
    assert table.is_contiguous() == (layout == "contiguous")
    expected = compute_percentile_checksum(table)
    pairing = OppositePercentilePairing(table, 5)
    assert pairing._table_is_finite
    table[:] = 0
    assert pairing.percentile_table.is_contiguous()
    assert t.equal(pairing.percentile_table, _table())
    assert pairing.provenance_params() == {
        "pairing_mode": "opposite_percentile",
        "pairing_candidate_pool_size": 5,
        "pairing_percentile_checksum": expected,
        "pairing_algorithm": "bounded_swap_v1",
    }


class _NativeNaNClone(t.Tensor):
    def detach(self):
        return self

    def clone(self, *args, **kwargs):
        owned = self.as_subclass(t.Tensor).clone(*args, **kwargs)
        owned[0, 49] = math.nan
        return owned


class _FeatureTensorSubclass(t.Tensor):
    pass


def test_subclass_clone_cannot_supply_native_finite_evidence():
    pairing = OppositePercentilePairing(_table().as_subclass(_NativeNaNClone))
    assert type(pairing.percentile_table) is t.Tensor
    assert not pairing._table_is_finite
    source = _source_docs([{0: 1.0}] * 2, [{0: 40.0}] * 2)
    with pytest.raises(ValueError, match="contains NaN"):
        pairing.improve_pairing(source, [(0, 0), (1, 0)], [1, 0])


@pytest.mark.parametrize("attribute", ["feature_ids", "feature_mags"])
def test_subclass_group_keeps_checked_lookup(monkeypatch, attribute):
    table = _table()
    segment = _segment({0: 25.0, 1: 75.0})
    segment = replace(
        segment, **{attribute: getattr(segment, attribute).as_subclass(_FeatureTensorSubclass)}
    )
    expected = _percentile_rows([segment], table)
    original = pairing_mod._magnitude_to_percentile
    evidence = []

    def recording(ids, mags, thresholds, *, table_is_finite=False):
        evidence.append(table_is_finite)
        return original(ids, mags, thresholds, table_is_finite=table_is_finite)

    monkeypatch.setattr(pairing_mod, "_magnitude_to_percentile", recording)
    actual = _percentile_rows([segment], table, table_is_finite=True)
    assert evidence == [False]
    np.testing.assert_array_equal(actual.values, expected.values)


@pytest.mark.parametrize("trusted", [False, True])
def test_missing_diagnostic_survives_qualified_chunk_lookup(monkeypatch, trusted):
    table = _table()
    table[1] = 0
    pairing = OppositePercentilePairing(table)
    assert pairing._table_is_finite
    monkeypatch.setattr(pairing_mod, "_PERCENTILE_LOOKUP_CHUNK_SIZE", 1)
    with pytest.raises(ValueError, match=r"statistics are missing for 1 distinct.*\[1\]"):
        _percentile_rows(
            [_segment({0: 25.0, 1: 75.0})],
            pairing.percentile_table,
            table_is_finite=trusted,
        )


@pytest.mark.parametrize("column", [0, 49, 98])
def test_public_lookup_retains_probed_nan_checks(column):
    table = _table()
    table[0, column] = math.nan
    with pytest.raises(ValueError, match="contains NaN"):
        percentile_mod.magnitude_to_percentile(t.tensor([0]), t.tensor([40.0]), table)


@pytest.mark.parametrize("sigma", [[0, 1], [1, 1], [1], [2, 0]])
def test_invalid_derangements_raise(sigma):
    source = _source_docs([{0: 50.0}] * 2, [{1: 50.0}] * 2)
    with pytest.raises(ValueError, match="derangement"):
        OppositePercentilePairing(_table()).improve_pairing(source, [(0, 0), (1, 0)], sigma)


@pytest.mark.parametrize(
    "cuts", [[(0, 0), (0, 0)], [(0, -1), (1, 0)], [(0, 1), (1, 0)], [(0, 0), (2, 0)]]
)
def test_invalid_cuts_raise(cuts):
    source = _source_docs([{0: 50.0}] * 2, [{1: 50.0}] * 2)
    with pytest.raises(ValueError, match="cut"):
        OppositePercentilePairing(_table()).improve_pairing(source, cuts, [1, 0])


def test_recombination_preserves_segments_and_full_tokenizer_oracle():
    tok = CharMergeTokenizer(bos_id=500)
    texts = ["abcdefghijklm", "nopqrstuvwxy", "ABCDEFGHIJK", "LMNOPQRSTU"]
    shard = make_base_shard(texts, tok, 2, context_window=32, n_features=128)
    source, _ = load_source_segments(
        shard,
        texts,
        tok,
        segment_size=2,
        context_window=32,
        sae_max_context=None,
        shard_name="pairing_test",
    )
    pairing = OppositePercentilePairing(_table(128), 8)
    seen_split = seen_loop = False
    for seed in range(12):
        result = recombine_segments(
            source,
            tok,
            cut_fraction=0.7,
            shard_seed=seed,
            shard_name="pairing_test",
            limit=8,
            pairing=pairing,
        )
        placed = [ref for doc in result.docs for ref in doc.segment_refs]
        expected = [(d, j) for d, doc in enumerate(source) for j in range(len(doc))]
        assert sorted(placed) == expected
        seg_tokens = [[seg.tokens for seg in doc] for doc in source]
        for doc in result.docs:
            ids = _piece_ids(list(doc.segment_refs), doc.new_started, seg_tokens, 500)
            assert len(ids) <= 8
            assert is_roundtrip_stable(ids, tok)
            assert doc.new_started == (doc.segment_refs[0][1] != 0)
        seen_split |= result.n_splits > 0
        seen_loop |= result.n_loops > 0
    assert seen_split and seen_loop


def test_identity_path_does_not_query_feature_statistics(monkeypatch):
    def unexpected(*args, **kwargs):
        raise AssertionError("identity recombination must not score partners")

    monkeypatch.setattr(OppositePercentilePairing, "improve_pairing", unexpected)
    source = _source_docs([{0: 50.0}] * 2, [{1: 50.0}] * 2)
    result = recombine_segments(
        source,
        CharMergeTokenizer(),
        cut_fraction=0.0,
        shard_seed=42,
        shard_name="identity",
        limit=8,
        pairing=OppositePercentilePairing(_table()),
    )
    assert result.n_cuts_final == 0
    assert [doc.segment_refs for doc in result.docs] == [((0, 0), (0, 1)), ((1, 0), (1, 1))]
