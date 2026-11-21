"""Exactness and lifecycle guards for the issue-#712 embedding-pool cache."""

from __future__ import annotations

import gc
import weakref
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
import torch as t
import yaml
from beartype.roar import BeartypeCallHintParamViolation
from hypothesis import given, settings
from hypothesis import strategies as st

import steering.conditional.feature_gcg.candidate_search as candidate_search
from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.edit_primitives import TokenReplacement
from steering.conditional.feature_gcg.errors import FeatureGcgError
from steering.eval.axbench_base.profiler import AxBenchProfiler


def _legacy_rank_oracle(
    *,
    embedding_gradient: t.Tensor,
    embedding_weight: t.Tensor,
    editable_positions: Sequence[int],
    allowed_token_ids: Sequence[int],
    current_tokens: Sequence[int],
    top_k: int,
    distance_weight: float,
) -> dict[int, tuple[TokenReplacement, ...]]:
    """Literal valid-input copy of the pre-#712 ranking body.

    This intentionally does not call the refactored materialization helper. It
    is the independent executable oracle for operation order and exact outputs.
    """

    allowed = t.tensor(
        list(allowed_token_ids),
        dtype=t.long,
        device=embedding_weight.device,
    )
    allowed_embeddings = embedding_weight.index_select(0, allowed)
    k_eff = min(int(top_k), len(allowed_token_ids))
    out: dict[int, tuple[TokenReplacement, ...]] = {}
    for position in editable_positions:
        grad = embedding_gradient[int(position)].to(allowed_embeddings.device)
        current_token = int(current_tokens[int(position)])
        current_embedding = embedding_weight[current_token].to(allowed_embeddings.device)
        scores = (allowed_embeddings - current_embedding) @ grad
        if distance_weight:
            same = allowed == int(current_tokens[int(position)])
            scores = scores + same.to(scores.dtype) * float(distance_weight)
        values, indices = t.topk(scores, k=k_eff, largest=False)
        out[int(position)] = tuple(
            TokenReplacement(
                position=int(position),
                token_id=int(allowed[int(idx)].item()),
                score=float(value.item()),
            )
            for value, idx in zip(values, indices, strict=True)
        )
    return out


def _rank(
    *,
    weight: t.Tensor,
    gradient: t.Tensor,
    allowed: Sequence[int],
    current: Sequence[int],
    positions: Sequence[int],
    top_k: int,
    distance_weight: float,
    cache: AllowedEmbeddingCache | None = None,
) -> dict[int, tuple[TokenReplacement, ...]]:
    return rank_replacements_from_gradients(
        embedding_gradient=gradient,
        embedding_weight=weight,
        editable_positions=positions,
        allowed_token_ids=allowed,
        current_tokens=current,
        top_k=top_k,
        distance_weight=distance_weight,
        allowed_embedding_cache=cache,
    )


def assert_ranking_matches_oracle(
    actual: dict[int, tuple[TokenReplacement, ...]],
    expected: dict[int, tuple[TokenReplacement, ...]],
    *,
    what: str = "ranking",
) -> None:
    """Compare a ranking to the legacy oracle: SELECTION exact, scores close.

    These assertions used to be plain ``actual == expected``, which pinned the
    exact float produced by the per-position form
    ``(allowed_embeddings - current_embedding) @ grad``.

    ``rank_replacements_from_gradients`` now distributes that dot product --
    ``(A - c) . g == A . g - (c . g)`` -- so the whole allowed-embedding pool is
    read ONCE for all positions instead of being copied per position. That line
    was 60.3% of in-loop GCG time (py-spy, live BPB-Adv run) and the rewrite is
    ~65x faster on it, but summing in a different order changes the last bits.

    What must NOT change is the SELECTION: which token ids are proposed, in
    which order, for which position. That is what GCG consumes. So selection is
    still compared exactly, and only the score float is given a tolerance.
    """
    assert actual.keys() == expected.keys(), f"{what}: position sets differ"
    for position, expected_row in expected.items():
        actual_row = actual[position]
        assert [r.position for r in actual_row] == [r.position for r in expected_row], (
            f"{what}: position field differs at {position}"
        )
        # Exact: the ordered proposal list is the semantic contract.
        assert [r.token_id for r in actual_row] == [r.token_id for r in expected_row], (
            f"{what}: proposed token ids differ at position {position}"
        )
        for actual_rep, expected_rep in zip(actual_row, expected_row, strict=True):
            assert actual_rep.score == pytest.approx(expected_rep.score, rel=1e-5, abs=1e-5), (
                f"{what}: score drift beyond rounding at position {position}"
            )


@pytest.mark.parametrize("dtype", [t.float32, t.float64])
@pytest.mark.parametrize("noncontiguous", [False, True])
def test_cached_and_raw_ranking_exactly_match_legacy_oracle(
    dtype: t.dtype,
    noncontiguous: bool,
) -> None:
    generator = t.Generator().manual_seed(712)
    vocab, dim = 31, 13
    if noncontiguous:
        weight = t.randn(dim, vocab, dtype=dtype, generator=generator).t()
        assert not weight.is_contiguous()
    else:
        weight = t.randn(vocab, dim, dtype=dtype, generator=generator)
    gradient = t.randn(6, dim, dtype=dtype, generator=generator)
    allowed = (17, 3, 17, 8, 29, 2, 11)
    current = (0, 17, 4, 8, 30, 2)
    positions = (1, 2, 4, 5)
    expected = _legacy_rank_oracle(
        embedding_gradient=gradient,
        embedding_weight=weight,
        editable_positions=positions,
        allowed_token_ids=allowed,
        current_tokens=current,
        top_k=12,
        distance_weight=7.25,
    )
    raw = _rank(
        weight=weight,
        gradient=gradient,
        allowed=allowed,
        current=current,
        positions=positions,
        top_k=12,
        distance_weight=7.25,
    )
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=allowed,
    )
    cached_first = _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=current,
        positions=positions,
        top_k=12,
        distance_weight=7.25,
        cache=cache,
    )
    cached_second = _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=current,
        positions=positions,
        top_k=12,
        distance_weight=7.25,
        cache=cache,
    )

    assert_ranking_matches_oracle(raw, expected, what="raw")
    assert_ranking_matches_oracle(cached_first, expected, what="cached_first")
    assert_ranking_matches_oracle(cached_second, expected, what="cached_second")
    # The cache must be self-consistent EXACTLY: same inputs, same path, twice.
    assert cached_first == cached_second
    assert cache.is_materialized


@settings(max_examples=80, deadline=None)
@given(
    seed=st.integers(min_value=0, max_value=2**31 - 1),
    dim=st.integers(min_value=1, max_value=17),
    allowed_list=st.lists(
        st.integers(min_value=0, max_value=11),
        min_size=1,
        max_size=18,
    ),
    top_k=st.integers(min_value=1, max_value=20),
    distance_weight=st.sampled_from([0.0, 0.125, 1.0, 10.0]),
)
def test_cached_ranking_property_matches_legacy_oracle(
    seed: int,
    dim: int,
    allowed_list: list[int],
    top_k: int,
    distance_weight: float,
) -> None:
    generator = t.Generator().manual_seed(seed)
    weight = t.randn(12, dim, dtype=t.float64, generator=generator)
    gradient = t.randn(5, dim, dtype=t.float64, generator=generator)
    allowed = tuple(allowed_list)
    current = (0, 1, 5, 7, 11)
    positions = (1, 2, 3, 4)
    expected = _legacy_rank_oracle(
        embedding_gradient=gradient,
        embedding_weight=weight,
        editable_positions=positions,
        allowed_token_ids=allowed,
        current_tokens=current,
        top_k=top_k,
        distance_weight=distance_weight,
    )
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=allowed,
    )
    actual = _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=current,
        positions=positions,
        top_k=top_k,
        distance_weight=distance_weight,
        cache=cache,
    )
    assert_ranking_matches_oracle(actual, expected)


def test_realistic_blas_shape_is_exact_not_tolerance_based() -> None:
    generator = t.Generator().manual_seed(17)
    vocab, dim = 4099, 769
    weight = t.randn(vocab, dim, generator=generator)
    gradient = t.randn(3, dim, generator=generator)
    allowed = tuple(range(1, vocab))
    current = (0, 3, vocab - 1)
    positions = (1, 2)
    expected = _legacy_rank_oracle(
        embedding_gradient=gradient,
        embedding_weight=weight,
        editable_positions=positions,
        allowed_token_ids=allowed,
        current_tokens=current,
        top_k=32,
        distance_weight=10.0,
    )
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=allowed,
    )
    assert_ranking_matches_oracle(
        _rank(
            weight=weight,
            gradient=gradient,
            allowed=cache.allowed_token_ids,
            current=current,
            positions=positions,
            top_k=32,
            distance_weight=10.0,
            cache=cache,
        ),
        expected,
        what="realistic BLAS shape",
    )


def test_numpy_integer_ids_preserve_order_duplicates_and_exact_scores() -> None:
    weight = t.arange(40, dtype=t.float64).reshape(10, 4)
    gradient = t.tensor([[0.0] * 4, [1.0, -2.0, 3.0, -4.0]], dtype=t.float64)
    allowed = cast(
        Sequence[int],
        (np.int64(7), np.int32(2), np.int64(7), np.int32(9)),
    )
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=allowed,
    )
    expected = _legacy_rank_oracle(
        embedding_gradient=gradient,
        embedding_weight=weight,
        editable_positions=(1,),
        allowed_token_ids=allowed,
        current_tokens=(0, 2),
        top_k=4,
        distance_weight=3.0,
    )
    actual = _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=(0, 2),
        positions=(1,),
        top_k=4,
        distance_weight=3.0,
        cache=cache,
    )
    assert_ranking_matches_oracle(actual, expected)


def test_cache_materializes_once_and_never_mutates_pool(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = t.randn(19, 7, generator=t.Generator().manual_seed(2))
    gradient = t.randn(4, 7, generator=t.Generator().manual_seed(3))
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 4, 9, 12, 18),
    )
    original = candidate_search._materialize_allowed_embedding_pool
    calls = 0

    def counted_materialize(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        counted_materialize,
    )
    assert not cache.is_materialized
    _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=(0, 1, 2, 3),
        positions=(1, 2, 3),
        top_k=3,
        distance_weight=0.0,
        cache=cache,
    )
    assert cache._allowed is not None
    assert cache._allowed_embeddings is not None
    allowed_before = cache._allowed.clone()
    embeddings_before = cache._allowed_embeddings.clone()
    allowed_object = cache._allowed
    embeddings_object = cache._allowed_embeddings

    _rank(
        weight=weight,
        gradient=-gradient,
        allowed=cache.allowed_token_ids,
        current=(0, 5, 6, 7),
        positions=(1, 3),
        top_k=5,
        distance_weight=2.0,
        cache=cache,
    )

    assert calls == 1
    assert cache._allowed is allowed_object
    assert cache._allowed_embeddings is embeddings_object
    assert cache._allowed is not None
    assert cache._allowed_embeddings is not None
    assert t.equal(cache._allowed, allowed_before)
    assert t.equal(cache._allowed_embeddings, embeddings_before)


@pytest.mark.parametrize(
    ("invalid", "expected"),
    [
        ({"positions": (0,)}, "editable position 0 outside non-BOS prompt range"),
        (
            {"current": (0, 99, 2)},
            "current token id 99 at position 1 is outside embedding vocab",
        ),
    ],
)
def test_post_materialization_ranking_error_keeps_cache_reusable(
    monkeypatch: pytest.MonkeyPatch,
    invalid: dict[str, Any],
    expected: str,
) -> None:
    """A downstream ranking error must not poison or rematerialize a valid pool."""

    weight = t.randn(8, 3, generator=t.Generator().manual_seed(91))
    gradient = t.randn(3, 3, generator=t.Generator().manual_seed(92))
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )
    original = candidate_search._materialize_allowed_embedding_pool
    calls = 0

    def counted_materialize(**kwargs):
        nonlocal calls
        calls += 1
        return original(**kwargs)

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        counted_materialize,
    )
    common: dict[str, Any] = {
        "weight": weight,
        "gradient": gradient,
        "allowed": cache.allowed_token_ids,
        "current": (0, 1, 2),
        "positions": (1,),
        "top_k": 2,
        "distance_weight": 1.0,
        "cache": cache,
    }
    with pytest.raises(FeatureGcgError, match=expected):
        _rank(**(common | invalid))

    assert cache.is_materialized
    assert calls == 1
    actual = _rank(**common)
    oracle = _legacy_rank_oracle(
        embedding_gradient=gradient,
        embedding_weight=weight,
        editable_positions=(1,),
        allowed_token_ids=cache.allowed_token_ids,
        current_tokens=(0, 1, 2),
        top_k=2,
        distance_weight=1.0,
    )
    assert actual == oracle
    assert calls == 1


@pytest.mark.parametrize(
    ("mutate", "expected"),
    [
        (
            lambda kwargs: kwargs.update(embedding_gradient=t.zeros(3, 4, 1)),
            "embedding_gradient must be 2-D (T, D), got torch.Size([3, 4, 1])",
        ),
        (
            lambda kwargs: kwargs.update(embedding_weight=t.zeros(5)),
            "embedding_weight must be 2-D (V, D), got torch.Size([5])",
        ),
        (
            lambda kwargs: kwargs.update(embedding_gradient=t.zeros(3, 3)),
            "gradient/embedding dim mismatch: 3 vs 4",
        ),
        (
            lambda kwargs: kwargs.update(allowed_token_ids=()),
            "allowed_token_ids must be non-empty",
        ),
        (
            lambda kwargs: kwargs.update(allowed_token_ids=(1, 9)),
            "allowed_token_ids contain ids outside embedding vocab",
        ),
        (
            lambda kwargs: kwargs.update(editable_positions=(0,)),
            "editable position 0 outside non-BOS prompt range",
        ),
        (
            lambda kwargs: kwargs.update(current_tokens=(0, 9, 2)),
            "current token id 9 at position 1 is outside embedding vocab",
        ),
    ],
)
def test_cache_preserves_legacy_error_text_and_precedence(mutate, expected: str) -> None:
    kwargs: dict[str, Any] = {
        "embedding_gradient": t.zeros(3, 4),
        "embedding_weight": t.zeros(5, 4),
        "editable_positions": (1,),
        "allowed_token_ids": (1, 2),
        "current_tokens": (0, 1, 2),
        "top_k": 1,
        "distance_weight": 0.0,
    }
    mutate(kwargs)
    with pytest.raises(FeatureGcgError) as raw_exc:
        rank_replacements_from_gradients(**kwargs)
    assert str(raw_exc.value) == expected

    weight = kwargs["embedding_weight"]
    allowed = tuple(kwargs["allowed_token_ids"])
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=allowed,
    )
    kwargs["allowed_token_ids"] = cache.allowed_token_ids
    with pytest.raises(FeatureGcgError) as cached_exc:
        rank_replacements_from_gradients(
            **kwargs,
            allowed_embedding_cache=cache,
        )
    assert str(cached_exc.value) == expected


@pytest.mark.parametrize(
    ("field", "value"),
    [("top_k", 0), ("distance_weight", -1.0)],
)
def test_ranking_unary_domains_precede_cache_materialization(
    field: str,
    value: int | float,
) -> None:
    weight = t.zeros(5, 4)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2),
    )
    kwargs: dict[str, Any] = {
        "embedding_gradient": t.zeros(3, 4),
        "embedding_weight": weight,
        "editable_positions": (1,),
        "allowed_token_ids": cache.allowed_token_ids,
        "current_tokens": (0, 1, 2),
        "top_k": 1,
        "distance_weight": 0.0,
        "allowed_embedding_cache": cache,
    }
    kwargs[field] = value
    with pytest.raises(BeartypeCallHintParamViolation, match=field):
        rank_replacements_from_gradients(**kwargs)
    assert not cache.is_materialized


def test_cache_rejects_wrong_sources_autograd_and_structural_change() -> None:
    weight = t.randn(8, 3)
    gradient = t.randn(3, 3)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )
    with pytest.raises(FeatureGcgError, match="different embedding_weight"):
        _rank(
            weight=weight.clone(),
            gradient=gradient,
            allowed=cache.allowed_token_ids,
            current=(0, 1, 2),
            positions=(1,),
            top_k=1,
            distance_weight=0.0,
            cache=cache,
        )
    with pytest.raises(FeatureGcgError, match="different allowed_token_ids"):
        _rank(
            weight=weight,
            gradient=gradient,
            allowed=tuple(list(cache.allowed_token_ids)),  # noqa: C414
            current=(0, 1, 2),
            positions=(1,),
            top_k=1,
            distance_weight=0.0,
            cache=cache,
        )

    weight.requires_grad_(True)
    raw = _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=(0, 1, 2),
        positions=(1,),
        top_k=1,
        distance_weight=0.0,
    )
    assert raw
    with pytest.raises(FeatureGcgError, match="requires_grad=False"):
        _rank(
            weight=weight,
            gradient=gradient,
            allowed=cache.allowed_token_ids,
            current=(0, 1, 2),
            positions=(1,),
            top_k=1,
            distance_weight=0.0,
            cache=cache,
        )
    weight.requires_grad_(False)
    _rank(
        weight=weight,
        gradient=gradient,
        allowed=cache.allowed_token_ids,
        current=(0, 1, 2),
        positions=(1,),
        top_k=1,
        distance_weight=0.0,
        cache=cache,
    )
    replacement_storage = weight.clone()
    weight.set_(
        replacement_storage.untyped_storage(),
        0,
        replacement_storage.size(),
        replacement_storage.stride(),
    )
    with pytest.raises(FeatureGcgError, match="changed after"):
        _rank(
            weight=weight,
            gradient=gradient,
            allowed=cache.allowed_token_ids,
            current=(0, 1, 2),
            positions=(1,),
            top_k=1,
            distance_weight=0.0,
            cache=cache,
        )


def test_failed_materialization_does_not_commit_partial_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = t.zeros(8, 3)
    gradient = t.zeros(3, 3)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )

    def fail_materialization(**_kwargs):
        raise RuntimeError("synthetic gather failure")

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        fail_materialization,
    )
    with pytest.raises(RuntimeError, match="synthetic gather failure"):
        _rank(
            weight=weight,
            gradient=gradient,
            allowed=cache.allowed_token_ids,
            current=(0, 1, 2),
            positions=(1,),
            top_k=1,
            distance_weight=0.0,
            cache=cache,
        )
    assert not cache.is_materialized
    assert cache._allowed is None
    assert cache._allowed_embeddings is None
    assert cache._source_signature is None


def test_cache_rejects_partial_internal_state() -> None:
    cache = AllowedEmbeddingCache(
        embedding_weight=t.zeros(8, 3),
        allowed_token_ids=(1, 2, 3),
    )
    cache._allowed = t.tensor([1, 2, 3])
    with pytest.raises(FeatureGcgError, match="partially initialized"):
        _ = cache.is_materialized


def test_source_change_during_materialization_is_not_committed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = t.zeros(8, 3)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )
    original = candidate_search._materialize_allowed_embedding_pool

    def materialize_then_swap(**kwargs):
        result = original(**kwargs)
        replacement = weight.clone()
        weight.set_(
            replacement.untyped_storage(),
            0,
            replacement.size(),
            replacement.stride(),
        )
        return result

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        materialize_then_swap,
    )
    with pytest.raises(FeatureGcgError, match="changed while materializing"):
        _rank(
            weight=weight,
            gradient=t.zeros(3, 3),
            allowed=cache.allowed_token_ids,
            current=(0, 1, 2),
            positions=(1,),
            top_k=1,
            distance_weight=0.0,
            cache=cache,
        )
    assert cache._allowed is None
    assert cache._allowed_embeddings is None
    assert cache._source_signature is None


@pytest.mark.parametrize("corrupt", ["ids", "embeddings"])
def test_cache_rejects_corrupted_cached_tensor_structure(corrupt: str) -> None:
    weight = t.zeros(8, 3)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )
    common: dict[str, Any] = {
        "weight": weight,
        "gradient": t.zeros(3, 3),
        "allowed": cache.allowed_token_ids,
        "current": (0, 1, 2),
        "positions": (1,),
        "top_k": 1,
        "distance_weight": 0.0,
        "cache": cache,
    }
    _rank(**common)
    if corrupt == "ids":
        cache._allowed = t.zeros(3, dtype=t.float32)
        expected = "id tensor has inconsistent structure"
    else:
        cache._allowed_embeddings = t.zeros(3, 2)
        expected = "embedding tensor has inconsistent structure"
    with pytest.raises(FeatureGcgError, match=expected):
        _rank(**common)


def test_cache_detects_partial_state_created_between_hit_guards(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    weight = t.zeros(8, 3)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )
    common: dict[str, Any] = {
        "weight": weight,
        "gradient": t.zeros(3, 3),
        "allowed": cache.allowed_token_ids,
        "current": (0, 1, 2),
        "positions": (1,),
        "top_k": 1,
        "distance_weight": 0.0,
        "cache": cache,
    }
    _rank(**common)
    original = candidate_search._embedding_weight_signature

    def signature_then_corrupt(source: t.Tensor):
        signature = original(source)
        cache._allowed = None
        return signature

    monkeypatch.setattr(
        candidate_search,
        "_embedding_weight_signature",
        signature_then_corrupt,
    )
    with pytest.raises(FeatureGcgError, match="partially initialized"):
        _rank(**common)


def test_cache_has_no_global_tensor_retention() -> None:
    weight = t.zeros(8, 3)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 2, 3),
    )
    _rank(
        weight=weight,
        gradient=t.zeros(3, 3),
        allowed=cache.allowed_token_ids,
        current=(0, 1, 2),
        positions=(1,),
        top_k=1,
        distance_weight=0.0,
        cache=cache,
    )
    pool_ref = weakref.ref(cache._allowed_embeddings)
    del cache
    gc.collect()
    assert pool_ref() is None


def test_profile_records_one_materialization_and_exact_bytes(tmp_path: Path) -> None:
    weight = t.zeros(19, 7, dtype=t.float32)
    cache = AllowedEmbeddingCache(
        embedding_weight=weight,
        allowed_token_ids=(1, 4, 9, 12, 18),
    )
    kwargs = {
        "weight": weight,
        "gradient": t.zeros(4, 7),
        "allowed": cache.allowed_token_ids,
        "current": (0, 1, 2, 3),
        "positions": (1, 2, 3),
        "top_k": 3,
        "distance_weight": 0.0,
        "cache": cache,
    }
    profile_dir = tmp_path / "profile"
    with AxBenchProfiler(profile_dir, metadata={"run_tag": "cache-unit"}):
        _rank(**kwargs)
        _rank(**kwargs)

    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text("utf-8"))
    regions = {row["name"]: row for row in summary["regions"]}
    counters = {row["name"]: row for row in summary["counters"]}
    assert regions["feature_gcg.rank.materialize_allowed_embedding_pool"]["count"] == 1
    assert counters["feature_gcg.rank.allowed_embedding_pool_materializations"]["total"] == 1
    expected_bytes = 5 * t.tensor([], dtype=t.long).element_size() + 5 * 7 * 4
    assert counters["feature_gcg.rank.allowed_embedding_pool_bytes"]["total"] == expected_bytes


def test_raw_profile_records_every_materialization_and_exact_bytes(tmp_path: Path) -> None:
    weight = t.zeros(19, 7, dtype=t.float32)
    kwargs: dict[str, Any] = {
        "weight": weight,
        "gradient": t.zeros(4, 7),
        "allowed": (1, 4, 9, 12, 18),
        "current": (0, 1, 2, 3),
        "positions": (1, 2, 3),
        "top_k": 3,
        "distance_weight": 0.0,
    }
    profile_dir = tmp_path / "raw-profile"
    with AxBenchProfiler(profile_dir, metadata={"run_tag": "raw-unit"}):
        _rank(**kwargs)
        _rank(**kwargs)

    summary = yaml.safe_load((profile_dir / "summary.yaml").read_text("utf-8"))
    regions = {row["name"]: row for row in summary["regions"]}
    counters = {row["name"]: row for row in summary["counters"]}
    assert regions["feature_gcg.rank.materialize_allowed_embedding_pool"]["count"] == 2
    assert counters["feature_gcg.rank.allowed_embedding_pool_materializations"]["total"] == 2
    expected_bytes = 5 * t.tensor([], dtype=t.long).element_size() + 5 * 7 * 4
    assert counters["feature_gcg.rank.allowed_embedding_pool_bytes"]["total"] == 2 * expected_bytes
