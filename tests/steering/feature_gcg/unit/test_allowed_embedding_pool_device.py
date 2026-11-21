"""Device placement for the allowed-embedding pool.

The BPB-Adv gradient path forces its surrogate embedding snapshot to CPU
(``surrogate.py:112,114`` mirror the CPU gradient capture at ``:788``). The
Faster-GCG ranking inherited that placement, so the scoring matmul
``(P, D) @ (D, V)`` -- a 0.66 GiB pool at the production shapes
(V=32734, D=5376) -- ran on the host. A profiler-attributed run measured it at
**14.9% of build_candidates, 55.2 ms/call**; on GPU the same matmul is ~0.9 ms
(140x at P=32).

The pool can therefore live on a DIFFERENT device from its source. These tests
pin that the split placement is honored, that the cache's structural guards
validate against the DECLARED pool device rather than the source device (they
previously compared to ``embedding_weight.device`` and would reject their own
pool), and that the default is unchanged.
"""

from __future__ import annotations

import pytest
import torch as t

from steering.conditional.feature_gcg.candidate_search import (
    AllowedEmbeddingCache,
    rank_replacements_from_gradients,
)
from steering.conditional.feature_gcg.errors import FeatureGcgError

_CUDA = pytest.mark.skipif(not t.cuda.is_available(), reason="needs CUDA")


def _pool(vocab: int = 40, dim: int = 8, seed: int = 11) -> t.Tensor:
    generator = t.Generator().manual_seed(seed)
    return t.randn(vocab, dim, generator=generator)


def test_default_device_is_the_source_device() -> None:
    """Omitting `device` must reproduce the pre-existing behavior exactly."""
    weight = _pool()
    cache = AllowedEmbeddingCache(embedding_weight=weight, allowed_token_ids=(1, 2, 3))
    assert cache.device == weight.device
    allowed, allowed_embeddings = cache.resolve(
        embedding_weight=weight, allowed_token_ids=cache.allowed_token_ids
    )
    assert allowed.device == weight.device
    assert allowed_embeddings.device == weight.device


def test_explicit_cpu_device_is_accepted_and_inert() -> None:
    """An explicit same-device request must behave like the default."""
    weight = _pool()
    cache = AllowedEmbeddingCache(
        embedding_weight=weight, allowed_token_ids=(1, 2, 3), device="cpu"
    )
    allowed, allowed_embeddings = cache.resolve(
        embedding_weight=weight, allowed_token_ids=cache.allowed_token_ids
    )
    assert allowed.device.type == "cpu"
    assert allowed_embeddings.device.type == "cpu"


def test_out_of_range_ids_still_raise_before_any_relocation() -> None:
    """The range check must be unaffected by the device request.

    The relocation deliberately happens AFTER the gather and range checks, so a
    bad id raises the same legacy message regardless of target device.
    """
    weight = _pool(vocab=10)
    cache = AllowedEmbeddingCache(embedding_weight=weight, allowed_token_ids=(1, 999), device="cpu")
    with pytest.raises(
        FeatureGcgError, match="allowed_token_ids contain ids outside embedding vocab"
    ):
        cache.resolve(embedding_weight=weight, allowed_token_ids=cache.allowed_token_ids)


@_CUDA
def test_pool_lands_on_cuda_while_source_stays_on_cpu() -> None:
    """The split placement this change exists to enable."""
    weight = _pool()  # CPU source, as the gradient path produces
    cache = AllowedEmbeddingCache(
        embedding_weight=weight, allowed_token_ids=(1, 2, 3, 4), device="cuda"
    )
    allowed, allowed_embeddings = cache.resolve(
        embedding_weight=weight, allowed_token_ids=cache.allowed_token_ids
    )
    assert weight.device.type == "cpu", "source must not be relocated"
    assert allowed.device.type == "cuda"
    assert allowed_embeddings.device.type == "cuda"


@_CUDA
def test_cached_hit_revalidates_against_declared_device_not_source() -> None:
    """The regression this change had to avoid.

    ``resolve`` re-validates the cached tensors' structure on every hit. Those
    checks compared ``allowed.device != embedding_weight.device``; with a CUDA
    pool over a CPU source that guard would fire on the cache's OWN correct
    output. The second resolve exercises the hit path.
    """
    weight = _pool()
    cache = AllowedEmbeddingCache(
        embedding_weight=weight, allowed_token_ids=(1, 2, 3, 4), device="cuda"
    )
    first = cache.resolve(embedding_weight=weight, allowed_token_ids=cache.allowed_token_ids)
    assert cache.is_materialized
    second = cache.resolve(  # hit path -- must not raise
        embedding_weight=weight, allowed_token_ids=cache.allowed_token_ids
    )
    assert first[0] is second[0] and first[1] is second[1]


@_CUDA
def test_ranking_selection_matches_cpu_pool_exactly() -> None:
    """Selection parity across devices, which is the semantic contract.

    Scores differ in the last bits (CPU and CUDA accumulate a 5376-length dot
    product in different orders), so the ordered token-id list -- what GCG
    actually consumes -- is what must match.
    """
    generator = t.Generator().manual_seed(99)
    vocab, dim, n_tokens = 512, 64, 12
    weight = t.randn(vocab, dim, generator=generator)
    gradient = t.randn(n_tokens, dim, generator=generator)
    allowed_ids = tuple(range(1, vocab))
    current = (0, *[int(x) for x in t.randint(1, vocab, (n_tokens - 1,)).tolist()])
    positions = (1, 3, 5, 7)

    def rank(device: str | None) -> dict[int, tuple[int, ...]]:
        cache = AllowedEmbeddingCache(
            embedding_weight=weight,
            allowed_token_ids=allowed_ids,
            device=device,
        )
        ranking = rank_replacements_from_gradients(
            embedding_gradient=gradient,
            embedding_weight=weight,
            editable_positions=positions,
            allowed_token_ids=cache.allowed_token_ids,
            current_tokens=current,
            top_k=16,
            distance_weight=7.0,
            allowed_embedding_cache=cache,
        )
        return {p: tuple(r.token_id for r in reps) for p, reps in ranking.items()}

    assert rank(None) == rank("cuda"), (
        "GPU pool must propose the same token ids, in the same order, as the "
        "CPU pool; only the score float may differ"
    )


@_CUDA
def test_ranking_scores_are_close_across_devices() -> None:
    """Bound the numerical drift the device move introduces."""
    generator = t.Generator().manual_seed(5)
    vocab, dim = 256, 32
    weight = t.randn(vocab, dim, generator=generator)
    gradient = t.randn(4, dim, generator=generator)
    allowed_ids = tuple(range(1, vocab))
    current = (0, 5, 9, 11)
    positions = (1, 2)

    def scores(device: str | None) -> list[float]:
        cache = AllowedEmbeddingCache(
            embedding_weight=weight, allowed_token_ids=allowed_ids, device=device
        )
        ranking = rank_replacements_from_gradients(
            embedding_gradient=gradient,
            embedding_weight=weight,
            editable_positions=positions,
            allowed_token_ids=cache.allowed_token_ids,
            current_tokens=current,
            top_k=8,
            distance_weight=0.0,
            allowed_embedding_cache=cache,
        )
        return [r.score for p in sorted(ranking) for r in ranking[p]]

    cpu_scores, gpu_scores = scores(None), scores("cuda")
    assert len(cpu_scores) == len(gpu_scores)
    for cpu_score, gpu_score in zip(cpu_scores, gpu_scores, strict=True):
        assert gpu_score == pytest.approx(cpu_score, rel=1e-4, abs=1e-4)
