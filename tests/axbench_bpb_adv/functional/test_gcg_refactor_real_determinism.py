"""INTEGRATION: real-model run-to-run determinism gate for the refactored GCG loop.

Proves that ``build_universal_gcg_candidates`` is bit-for-bit deterministic on
the real d12_base model: two runs with IDENTICAL inputs (same cfg + seed + rows)
yield byte-identical candidate tables. This is the Phase-0 real-model guard: the
``run_gcg`` refactor could have broken determinism (e.g., by introducing
non-seeded sampling, accumulating mutable state across calls, or leaking RNG
state from a helper).

Why run-to-run determinism (not per-prefix-vs-universal byte-equality on the
real model): the per-prefix path evaluates candidates one-at-a-time
(batch-of-1) while the universal path evaluates them batched (batch-of-N), and
bf16 cuBLAS batch-shape rounding differs between those two calling conventions
-- so they are NOT byte-equal on real bf16 hardware. That 1-pair byte-equality
is already proven by the committed TOY golden (``test_universal_gcg_equiv_golden.py``
and ``test_universal_gcg_flow.py``). The real-model gate must therefore check a
different, achievable property: run-to-run determinism within the universal
path, which is the property the refactor could have broken without the toy
golden detecting it.

Marked ``slow`` (real-model, GPU): deselected by the default ``-m 'not slow ...'``.
Falls back to skip (not fail) when d12_base / CUDA are unavailable.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, NoReturn

import pyarrow as pa
import pytest
import torch as t

import steering.conditional.feature_gcg.candidate_search as candidate_search
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.model_optimize import BpbObjectiveCallbacks
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import _build_token_bytes
from steering.eval.axbench_bpb_adv.surrogate import bpb_surrogate_embedding_weight
from steering.eval.axbench_bpb_adv.token_filter import build_allowed_token_ids
from steering.eval.axbench_bpb_adv.universal_gcg import concept_groups
from tests.axbench_bpb_adv.functional._d12_base_harness import (
    D12_DIR,
    D12_STEP,
    D12_TAG,
    load_d12_base,
    make_objective,
    single_pair_attack_rows,
)
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import table_to_parquet_bytes


def _cfg() -> FasterGcgConfig:
    """Minimal multi-k config, kept small enough for the real-model gate."""
    return FasterGcgConfig(
        optimized_token_counts=(1, 2),
        max_steps=1,
        candidate_batch_size=8,
        top_k=16,
        distance_weight=10.0,
        temperature=0.1,
        master_seed=0,
    )


def _multi_pair_multi_k_attack_rows(prompt_tokens: list[int]) -> pa.Table:
    rows: list[dict[str, Any]] = []
    for editable_count in (1, 2):
        base = single_pair_attack_rows(
            prompt_tokens,
            editable_start=len(prompt_tokens) - editable_count,
            editable_count=editable_count,
        ).to_pylist()[0]
        base["strength_index"] = editable_count
        for pair_index in range(2):
            row = dict(base)
            suffix = f"k{editable_count}-p{pair_index}"
            row["attack_row_id"] = f"integ-{suffix}"
            row["core_prefix_id"] = f"core-{suffix}"
            row["concept_rich_sample_id"] = f"rich-{suffix}"
            row["concept_avoiding_sample_id"] = f"avoid-{suffix}"
            rows.append(row)
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)


@pytest.mark.slow
def test_universal_gcg_run_to_run_determinism_on_real_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """build_universal_gcg_candidates is byte-identical across two runs on d12_base.

    Two independent runs with FRESH callbacks (empty eval cache each time) must
    produce bit-for-bit identical candidate tables. The comparison uses
    ``table_to_parquet_bytes`` -- the same serialisation the Phase-0 golden test
    uses -- for exact byte-level equality (not just Python-value equality).

    Using fresh callbacks (rather than replaying a warm cache) is deliberate: the
    recompute must be deterministic through the REAL bf16 model, not just through
    Python dict equality. Any non-seeded sampling, mutable shared state, or RNG
    leak introduced by the ``run_gcg`` refactor would manifest as a difference
    between the two runs.
    """
    if not t.cuda.is_available():
        pytest.skip("requires CUDA")
    if not (D12_DIR / D12_TAG / f"model_{D12_STEP:06d}.pt").exists():
        pytest.skip("requires d12_base checkpoint")

    bundle = load_d12_base()
    device = t.device("cuda")
    token_bytes = _build_token_bytes(bundle.tokenizer)
    prompt_tokens = [
        int(x) for x in bundle.tokenizer.encode("the quick brown fox jumps over")
    ][:8]
    attack_rows = _multi_pair_multi_k_attack_rows(prompt_tokens)
    cfg = _cfg()
    embedding_weight = bpb_surrogate_embedding_weight(bundle.model)
    allowed = [
        int(tid)
        for tid in build_allowed_token_ids(bundle.tokenizer)
        if 0 <= int(tid) < int(embedding_weight.shape[0])
    ]

    def _objective_tripwire(
        _items: Sequence[tuple[dict[str, Any], Sequence[int]]],
    ) -> NoReturn:
        pytest.fail("an all-restored real-model run called the objective or gradient")

    def _gradient_tripwire(
        _items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        _step: int,
    ) -> NoReturn:
        pytest.fail("an all-restored real-model run called the objective or gradient")

    def _run(
        *,
        restored: dict[str, dict[str, object]] | None = None,
        fail_model_calls: bool = False,
        reject_strength_index: int | None = None,
    ) -> tuple[pa.Table, list[dict[str, object]]]:
        # Fresh callbacks each run: empty eval cache forces a real recompute,
        # so determinism comes from the seeded model path, not from cache hits.
        obj: BpbObjectiveCallbacks = make_objective(bundle, token_bytes, device)

        def _objective_checked(
            items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        ) -> list[int | float]:
            if reject_strength_index is not None:
                assert all(
                    int(row["strength_index"]) != reject_strength_index
                    for row, _tokens in items
                ), "a restored real-model group reached the objective"
            return obj.objective_batch(items)

        def _gradient_checked(
            items: Sequence[tuple[dict[str, Any], Sequence[int]]],
            step: int,
        ) -> list[t.Tensor]:
            if reject_strength_index is not None:
                assert all(
                    int(row["strength_index"]) != reject_strength_index
                    for row, _tokens in items
                ), "a restored real-model group reached the gradient"
            return obj.gradient_provider_batch(items, step)

        return build_universal_gcg_candidates(
            attack_rows=attack_rows,
            tokenizer=bundle.tokenizer,
            cfg=cfg,
            embedding_weight=obj.embedding_weight(),
            allowed_token_ids=allowed,
            objective=_objective_tripwire if fail_model_calls else _objective_checked,
            gradient_provider_batch=(
                _gradient_tripwire if fail_model_calls else _gradient_checked
            ),
            restored_triggers=restored,
        )

    first, first_triggers = _run()
    second, second_triggers = _run()

    assert table_to_parquet_bytes(first) == table_to_parquet_bytes(second), (
        "build_universal_gcg_candidates is NOT deterministic on the real model: "
        "two runs with identical inputs produced different candidate tables."
    )
    assert first_triggers == second_triggers

    original_materialize = candidate_search._materialize_allowed_embedding_pool
    pool_materializations = 0

    def _count_materialization(
        *,
        embedding_weight: t.Tensor,
        allowed_token_ids: Sequence[int],
        id_to_row: Callable[[t.Tensor], t.Tensor] | None = None,
        device: t.device | None = None,
    ) -> tuple[t.Tensor, t.Tensor]:
        nonlocal pool_materializations
        pool_materializations += 1
        return original_materialize(
            embedding_weight=embedding_weight,
            allowed_token_ids=allowed_token_ids,
            id_to_row=id_to_row,
            device=device,
        )

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        _count_materialization,
    )
    groups = concept_groups(attack_rows.to_pylist())
    restored_one = {str(groups[0].key_str): first_triggers[0]}
    mixed, mixed_triggers = _run(
        restored=restored_one,
        reject_strength_index=int(groups[0].rows[0]["strength_index"]),
    )
    assert table_to_parquet_bytes(first) == table_to_parquet_bytes(mixed)
    assert first_triggers == mixed_triggers
    assert pool_materializations == 1

    restored_all = {
        str(group.key_str): trigger
        for group, trigger in zip(groups, first_triggers, strict=True)
    }
    restored, restored_triggers = _run(
        restored=restored_all,
        fail_model_calls=True,
    )
    assert table_to_parquet_bytes(first) == table_to_parquet_bytes(restored)
    assert first_triggers == restored_triggers
    assert pool_materializations == 1
