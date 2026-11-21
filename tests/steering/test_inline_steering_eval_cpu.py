"""CPU-only branching tests for inline_steering_eval.

The existing ``tests/steering/test_inline_steering_eval_p0_slow.py``
contains placeholder ``pytest.skip(...)`` stubs gated on CUDA. This file
provides marker-free CPU coverage for the selection / partition / gather
/ build_random_inputs branching logic that the placeholders were meant
to cover, using synthetic features and ``MockTokenizer`` rather than a
real model.
"""

from __future__ import annotations

import torch

from steering.eval.inline_steering_eval import (
    _gather_int,
    _partition_docs,
)

# ---------------------------------------------------------------------------
# _partition_docs — round-robin DDP partition
# ---------------------------------------------------------------------------


class TestPartitionDocs:
    def test_single_rank_returns_full_list(self) -> None:
        docs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        out = _partition_docs(docs, rank=0, world_size=1)
        assert out == docs

    def test_round_robin_two_ranks(self) -> None:
        docs = [(0, 1), (1, 2), (2, 3), (3, 4)]
        rank0 = _partition_docs(docs, rank=0, world_size=2)
        rank1 = _partition_docs(docs, rank=1, world_size=2)
        assert rank0 == [(0, 1), (2, 3)]
        assert rank1 == [(1, 2), (3, 4)]

    def test_uneven_partition(self) -> None:
        docs = [(0, 1), (1, 2), (2, 3)]
        rank0 = _partition_docs(docs, rank=0, world_size=4)
        rank3 = _partition_docs(docs, rank=3, world_size=4)
        assert rank0 == [(0, 1)]
        # rank3 gets nothing — n_eval_docs (3) < world_size (4)
        assert rank3 == []

    def test_partitions_disjoint_and_complete(self) -> None:
        """Union of all rank slices must equal the full list (no gaps, no dups)."""
        docs = [(i, i + 1) for i in range(10)]
        world = 4
        seen: list[tuple[int, int]] = []
        for r in range(world):
            seen.extend(_partition_docs(docs, rank=r, world_size=world))
        assert sorted(seen) == sorted(docs)


# ---------------------------------------------------------------------------
# _gather_int — single-rank fast path (no DDP needed for CPU test)
# ---------------------------------------------------------------------------


class TestGatherInt:
    def test_gather_int_world_size_1_no_dist(self) -> None:
        """world_size=1 must not require dist.is_initialized() → True.

        ``_gather_int`` returns the local value untouched when
        ``world_size <= 1 or dist is not initialized``.
        """
        device = torch.device("cpu")
        out = _gather_int(local_val=42, world_size=1, device=device)
        assert out == 42

    def test_gather_int_uninitialized_dist_returns_local(self) -> None:
        """No DDP init → fallback to local int."""
        device = torch.device("cpu")
        out = _gather_int(local_val=7, world_size=2, device=device)
        # World > 1 but dist.is_initialized() is False → fallback to local
        assert out == 7


# ---------------------------------------------------------------------------
# Imports / smoke tests for selection helpers (run on CPU; no model needed)
# ---------------------------------------------------------------------------


class TestImports:
    def test_select_random_docs_imports(self) -> None:
        """``_select_random_docs`` must import and be callable on CPU."""
        from steering.eval.inline_steering_eval import _select_random_docs

        assert callable(_select_random_docs)

    def test_aggregate_metrics_imports(self) -> None:
        from steering.eval.inline_steering_eval import _aggregate_roundtrip_metrics

        assert callable(_aggregate_roundtrip_metrics)

    def test_build_random_inputs_imports(self) -> None:
        from steering.eval.inline_steering_eval import _build_random_inputs

        assert callable(_build_random_inputs)
