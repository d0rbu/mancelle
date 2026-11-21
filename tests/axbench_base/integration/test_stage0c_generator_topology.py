"""Generator-aware Stage 0c topology and resume identity checks.

The larger Stage 0c distributed gates stub ``_verify_concept`` so they can
exercise orchestration cheaply.  This companion keeps the fixture small but
runs the real cross-prefix generator with a public selectable cache.  It pins
same-mode identity across world sizes and simulated checkpoint boundaries.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pyarrow as pa
import pytest
import torch as t

from steering.eval.axbench_base.batched_generation import (
    cross_prefix_batched_generate_seeds,
)
from steering.eval.axbench_base.config import VerificationPrefillMode

_N_CANDIDATES = 9
_INTERRUPT_AFTER = 4
_NEGATIVE_PREFIX_SCHEMA = pa.schema(
    [
        pa.field("candidate_index", pa.int64(), nullable=False),
        pa.field("prefix_index", pa.int64(), nullable=False),
        pa.field("seed", pa.int64(), nullable=False),
        pa.field("continuation", pa.list_(pa.int64()), nullable=False),
    ]
)


class _TopologyCache:
    """Tiny cache whose row state is expanded through the public API."""

    def __init__(self, prefix_codes: t.Tensor) -> None:
        self.prefix_codes = prefix_codes.clone()
        self.decode_step = 0

    def batch_select_indices(self, indices: t.Tensor) -> None:
        self.prefix_codes = self.prefix_codes.index_select(0, indices).clone()


class _TopologyHFModel:
    """Deterministic, batch-shape-sensitive HF-style causal LM."""

    def __init__(self) -> None:
        self.config = SimpleNamespace(pad_token_id=0, eos_token_id=31)
        self._anchor = t.nn.Parameter(t.zeros(()), requires_grad=False)

    def parameters(self):
        return iter([self._anchor])

    def __call__(
        self,
        input_ids: t.Tensor,
        *,
        past_key_values: _TopologyCache | None = None,
        use_cache: bool = False,
        **kwargs: Any,
    ) -> SimpleNamespace:
        del use_cache
        batch_size = int(input_ids.shape[0])
        if past_key_values is None:
            attention_mask = kwargs.get("attention_mask")
            assert isinstance(attention_mask, t.Tensor)
            prefix_codes = (input_ids * attention_mask).sum(dim=1) % 23
            cache = _TopologyCache(prefix_codes)
            centers = (prefix_codes + batch_size) % 29
        else:
            cache = past_key_values
            assert cache.prefix_codes.shape[0] == batch_size
            cache.decode_step += 1
            centers = (cache.prefix_codes + input_ids[:, -1] + cache.decode_step) % 29

        vocab = t.arange(32, dtype=t.float32, device=input_ids.device)
        logits = -(vocab.unsqueeze(0) - centers.unsqueeze(1)).abs() / 2.0
        return SimpleNamespace(
            logits=logits.unsqueeze(1),
            past_key_values=cache,
        )


def _base_model() -> SimpleNamespace:
    hf_model = _TopologyHFModel()
    nnterp_model = SimpleNamespace(
        _model=hf_model,
    )
    return SimpleNamespace(nnterp_model=nnterp_model)


def _candidate_inputs(
    candidate_index: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Return uneven prompts and stable candidate-identity-derived seeds."""
    token_rows = [
        [1, 3 + candidate_index, 5],
        [2, 7 + candidate_index, 11, 13 + candidate_index],
    ]
    seed_rows = [
        [candidate_index * 10_000 + 101, candidate_index * 10_000 + 103],
        [
            candidate_index * 10_000 + 211,
            candidate_index * 10_000 + 223,
            candidate_index * 10_000 + 227,
        ],
    ]
    return token_rows, seed_rows


def _generator_topology_target(
    rank: int,
    world_size: int,
    mode: str,
    candidate_indices: list[int],
) -> list[dict[str, Any]]:
    """Run this rank's modulo stripe through the real generator."""
    rows: list[dict[str, Any]] = []
    for candidate_index in candidate_indices:
        if candidate_index % world_size != rank:
            continue
        token_rows, seed_rows = _candidate_inputs(candidate_index)
        continuations = cross_prefix_batched_generate_seeds(
            _base_model(),
            token_rows,
            seed_rows=seed_rows,
            max_new_tokens=5,
            temperature=0.7,
            top_k=8,
            stop_token_ids=set(),
            prefill_mode=mode,
        )
        token_sum = sum(
            token
            for prefix_rows in continuations
            for continuation in prefix_rows
            for token in continuation
        )
        rows.append(
            {
                "candidate_index": candidate_index,
                "accepted": (token_sum + candidate_index) % 3 != 0,
                "seed_rows": seed_rows,
                "continuations": continuations,
            }
        )
    return rows


def _merge_rank_rows(rank_rows: list[list[dict[str, Any]]]) -> list[dict[str, Any]]:
    merged = [row for rows in rank_rows for row in rows]
    return sorted(merged, key=lambda row: int(row["candidate_index"]))


def _artifact_views(
    rows: list[dict[str, Any]],
) -> tuple[bytes, bytes, pa.Table]:
    accepted = [
        {
            "candidate_index": row["candidate_index"],
            "continuations": row["continuations"],
        }
        for row in rows
        if row["accepted"]
    ]
    rejected = [{"candidate_index": row["candidate_index"]} for row in rows if not row["accepted"]]
    concepts_bytes = (json.dumps(accepted, sort_keys=True, separators=(",", ":")) + "\n").encode()
    rejected_bytes = (json.dumps(rejected, sort_keys=True, separators=(",", ":")) + "\n").encode()

    prefix_rows: list[dict[str, Any]] = []
    for row in rows:
        if not row["accepted"]:
            continue
        for prefix_index, (seeds, continuations) in enumerate(
            zip(row["seed_rows"], row["continuations"], strict=True)
        ):
            for seed, continuation in zip(seeds, continuations, strict=True):
                prefix_rows.append(
                    {
                        "candidate_index": row["candidate_index"],
                        "prefix_index": prefix_index,
                        "seed": seed,
                        "continuation": continuation,
                    }
                )
    table = pa.Table.from_pylist(prefix_rows, schema=_NEGATIVE_PREFIX_SCHEMA)
    return concepts_bytes, rejected_bytes, table


def _assert_artifacts_equal(
    expected: tuple[bytes, bytes, pa.Table],
    actual: tuple[bytes, bytes, pa.Table],
) -> None:
    expected_concepts, expected_rejected, expected_table = expected
    actual_concepts, actual_rejected, actual_table = actual
    assert actual_concepts == expected_concepts
    assert actual_rejected == expected_rejected
    assert actual_table.schema == expected_table.schema
    assert actual_table.to_pylist() == expected_table.to_pylist()


@pytest.mark.parametrize("mode", list(VerificationPrefillMode))
def test_real_generator_is_identical_at_world_sizes_1_2_and_4(
    gloo_multiprocess: Any,
    mode: VerificationPrefillMode,
) -> None:
    candidate_indices = list(range(_N_CANDIDATES))
    reference_rows = _merge_rank_rows(
        gloo_multiprocess(
            1,
            _generator_topology_target,
            mode.value,
            candidate_indices,
        )
    )
    reference = _artifact_views(reference_rows)

    for world_size in (2, 4):
        distributed_rows = _merge_rank_rows(
            gloo_multiprocess(
                world_size,
                _generator_topology_target,
                mode.value,
                candidate_indices,
            )
        )
        _assert_artifacts_equal(reference, _artifact_views(distributed_rows))


@pytest.mark.parametrize("mode", list(VerificationPrefillMode))
@pytest.mark.parametrize("resume_world_size", [2, 4])
def test_real_generator_resume_and_topology_change_are_identical(
    gloo_multiprocess: Any,
    mode: VerificationPrefillMode,
    resume_world_size: int,
) -> None:
    candidate_indices = list(range(_N_CANDIDATES))
    reference = _artifact_views(
        _merge_rank_rows(
            gloo_multiprocess(
                1,
                _generator_topology_target,
                mode.value,
                candidate_indices,
            )
        )
    )

    completed_indices = candidate_indices[:_INTERRUPT_AFTER]
    remaining_indices = candidate_indices[_INTERRUPT_AFTER:]
    initial_rows = _merge_rank_rows(
        gloo_multiprocess(
            2,
            _generator_topology_target,
            mode.value,
            completed_indices,
        )
    )
    resumed_rows = _merge_rank_rows(
        gloo_multiprocess(
            resume_world_size,
            _generator_topology_target,
            mode.value,
            remaining_indices,
        )
    )
    resumed = _artifact_views(
        sorted(
            initial_rows + resumed_rows,
            key=lambda row: int(row["candidate_index"]),
        )
    )
    _assert_artifacts_equal(reference, resumed)
