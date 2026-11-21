"""Shared fixture + synthetic closures for the universal-GCG equivalence oracle.

This module is the SINGLE SOURCE OF TRUTH for the per-prefix-vs-universal
byte-identity proof. Both the Phase-0.2 golden-capture/smoke test and the
Phase-3.4 universal-optimizer equivalence test import the IDENTICAL fixture,
synthetic objective/gradient closures, and ``FasterGcgConfig`` from here. That
shared provenance is what makes the byte-identity *provable*: the only thing
that changes between the two tests is which optimizer (the legacy per-prefix
``build_faster_gcg_candidates`` vs the new ``build_universal_gcg_candidates``)
consumes them.

Determinism regime (v2 finding M1 + the seed note). The legacy per-prefix path
seeds candidate sampling on ``attack_row_id`` while the universal path seeds on
the concept-group key; these differ even for a single-pair group. Byte-identity
is therefore only achievable when candidate sampling is RNG-INDEPENDENT. We
force that here with ``top_k=1`` (a SINGLETON replacement pool per editable
position ⇒ ``sample_candidate_sequences`` draws the only element regardless of
the RNG seed) and ``candidate_batch_size=1``, exactly as
``tests/axbench_bpb_adv/unit/test_optimize.py::
test_faster_gcg_builds_final_candidate_and_not_saved_steps`` does. For a
non-deterministic sampling regime (``top_k>1``) the universal optimizer is a
strict generalisation but the exact candidate stream differs by seed -- that is
expected, not a regression.

Fixture shape (the backward-compat oracle is a DEGENERATE single-pair group).
Two concepts, each with exactly ONE core-prefix carrying exactly ONE
``concept_rich`` x ONE ``concept_avoiding`` pair, so every concept group holds
exactly one pair (minibatch == {that pair} every step). Rows span
``optimized_token_counts=(0, 1, 2)`` at a single coefficient ⇒ 2 concepts x 3
k-values == 6 attack rows, each its own concept group.

Synthetic objective/gradient (a REAL optimizer run that changes tokens). Modeled
on ``test_optimize.py:166-205``: the embedding places the lone allowed token
``_TARGET_TOKEN`` at ``[-1.0, 0.0]`` and every editable position's gradient is
``[1.0, 0.0]``, so the first-order GCG score ``(e_target - e_cur) . grad`` is
minimised by ``_TARGET_TOKEN`` at each editable position. The objective counts
editable positions NOT yet equal to ``_TARGET_TOKEN`` (lower is better; 0 when
the whole span is the target), so the optimizer monotonically flips every
editable position to the target over the steps. The closures depend ONLY on the
row's editable span + token contents -- never on ``concept_id``
-- so the same closures are reused verbatim by the universal path.
"""

from __future__ import annotations

import io
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t

from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    token_sequence_digest,
)
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.editable_tokens import select_final_k_editable_span
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA

# Embedding dimensionality of the synthetic surrogate.
_EMBED_DIM = 2
# A vocabulary large enough to cover BOS (0), the prompt tokens, the rich/avoid
# output tokens, and the lone adversarial target token.
_VOCAB_SIZE = 200
# The single allowed replacement token. With ``top_k=1`` this makes every
# position's replacement pool a singleton ⇒ sampling is RNG-independent.
_TARGET_TOKEN = 120
# Editable-span widths (the k grid) the oracle spans.
_OPTIMIZED_TOKEN_COUNTS: tuple[int, ...] = (0, 1, 2)
# The single coefficient the oracle uses.
_COEFFICIENT_STR = "8.0"
# Where the committed golden parquet lives (relative to the repo root).
GOLDEN_PARQUET_RELPATH = (
    "tests/axbench_bpb_adv/fixtures/golden/universal_gcg_equivalence/"
    "per_prefix_one_pair_candidates.parquet"
)


def golden_parquet_path() -> Path:
    """Absolute path to the committed per-prefix one-pair-group golden parquet."""

    # tests/axbench_bpb_adv/functional/_universal_gcg_equiv.py -> repo root is
    # four parents up.
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / GOLDEN_PARQUET_RELPATH


class _EquivTokenizer:
    """Minimal tokenizer: BOS == 0, decode drops BOS and maps ids to chars."""

    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def equiv_tokenizer() -> _EquivTokenizer:
    return _EquivTokenizer()


def equiv_embedding_weight() -> t.Tensor:
    """``(V, D)`` embedding placing the target token at ``[-1.0, 0.0]``.

    Every other row is zero, so ``(e_target - e_cur) . [1, 0] == -1`` for any
    non-target current token (strictly preferred) and ``0`` for the target
    itself -- the same construction as ``test_optimize.py``.
    """

    weight = t.zeros(_VOCAB_SIZE, _EMBED_DIM, dtype=t.float32)
    weight[_TARGET_TOKEN] = t.tensor([-1.0, 0.0])
    return weight


def equiv_allowed_token_ids() -> list[int]:
    """The singleton allowed-replacement pool (⇒ RNG-independent sampling)."""

    return [_TARGET_TOKEN]


def equiv_config() -> FasterGcgConfig:
    """The deterministic-sampling ``FasterGcgConfig`` shared by both tests.

    ``top_k=1`` + ``candidate_batch_size=1`` make candidate sampling
    RNG-independent; ``max_steps=2`` is enough to flip both editable positions
    (k<=2) to the target. The grid is pinned so the fixture builder and the
    universal optimizer agree on the k sweep.
    """

    return FasterGcgConfig(
        optimized_token_counts=_OPTIMIZED_TOKEN_COUNTS,
        max_steps=2,
        candidate_batch_size=1,
        top_k=1,
        distance_weight=0.0,
        temperature=1.0,
        master_seed=0,
    )


def _span_from_row(row: dict[str, Any]) -> EditableSpan:
    return EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
    )


def equiv_objective(_row: dict[str, Any], tokens: Sequence[int]) -> float:
    """Count editable positions NOT yet equal to the target token (minimize).

    Lower is better; ``0.0`` once the whole editable span is the target. The
    span is read from the row, so the objective is correct for any k. The
    closure ignores ``concept_id`` -- only the editable
    token contents matter -- so it is reusable verbatim by the universal path.
    """

    span = _span_from_row(_row)
    return float(
        sum(
            1
            for position in span.positions()
            if int(tokens[int(position)]) != _TARGET_TOKEN
        )
    )


def as_batched_objective(
    single: Callable[[dict[str, Any], Sequence[int]], float],
) -> Callable[[Sequence[tuple[dict[str, Any], Sequence[int]]]], list[float]]:
    """Wrap a single ``(row, tokens) -> float`` objective into the batched
    ``ObjectiveBatchFn`` the optimizer now consumes. A pure-Python synthetic
    objective is exact (no bf16 batch-shape rounding), so the batched path yields
    byte-identical values to evaluating each item singly -- which is why the
    1-pair equivalence golden still holds under batching for the synthetic oracle
    (only the real model's batched forward introduces the accepted ~bf16 drift).
    """

    def batched(items: Sequence[tuple[dict[str, Any], Sequence[int]]]) -> list[float]:
        return [float(single(row, tokens)) for row, tokens in items]

    return batched


equiv_objective_batch = as_batched_objective(equiv_objective)


def equiv_gradient_provider(
    row: dict[str, Any],
    tokens: Sequence[int],
    _step: int,
) -> t.Tensor:
    """``(len(tokens), D)`` gradient: ``[1.0, 0.0]`` at each editable position.

    Combined with ``equiv_embedding_weight`` this makes the lone allowed token
    the first-order-best replacement at every editable position, mirroring
    ``test_optimize.py:_gradient_provider`` generalised from the last position
    to the whole editable span.
    """

    span = _span_from_row(row)
    gradient = t.zeros(len(tokens), _EMBED_DIM, dtype=t.float32)
    for position in span.positions():
        gradient[int(position)] = t.tensor([1.0, 0.0])
    return gradient


def as_batched_gradient(
    single: Callable[[dict[str, Any], Sequence[int], int], t.Tensor],
) -> Callable[[Sequence[tuple[dict[str, Any], Sequence[int]]], int], list[t.Tensor]]:
    """Wrap a single ``(row, tokens, step) -> Tensor`` gradient provider into the
    batched ``GradientBatchProvider`` the universal optimizer now consumes. A
    pure-Python synthetic gradient is exact, so looping it is byte-identical to the
    real batched fwd+bwd would be on an exact model -- which is why the 1-pair
    equivalence golden still holds under gradient batching for the synthetic oracle
    (only the real model's batched backward introduces the accepted ~bf16 drift).
    """

    def batched(
        items: Sequence[tuple[dict[str, Any], Sequence[int]]], step: int
    ) -> list[t.Tensor]:
        return [single(row, tokens, step) for row, tokens in items]

    return batched


equiv_gradient_provider_batch = as_batched_gradient(equiv_gradient_provider)


def _attack_row(
    *,
    concept_id: int,
    prompt_tokens: list[int],
    rich_output: list[int],
    avoid_output: list[int],
    rich_sample_id: str,
    avoid_sample_id: str,
    core_prefix_id: str,
    k: int,
) -> dict[str, Any]:
    span = select_final_k_editable_span(prompt_tokens, k=k)
    original_editable = [int(tok) for tok in prompt_tokens[span.start : span.stop]]
    attack_row_id = (
        f"c{concept_id}|{core_prefix_id}|{rich_sample_id}|{avoid_sample_id}"
        f"|k{k}"
    )
    return {
        "attack_row_id": attack_row_id,
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": rich_sample_id,
        "concept_avoiding_sample_id": avoid_sample_id,
        "concept_id": int(concept_id),
        "mode": "positive",
        "method_name": "diffmean_encdec",
        "target_model": "encdec",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(k),
        "strength_unit": "optimized_tokens",
        "strength_index": int(k),
        "context_signal_mode": "fixed_context",
        "prefix_text": "".join(
            chr(int(tok)) for tok in prompt_tokens if int(tok) != 0
        ),
        "prefix_hash": f"hash-{core_prefix_id}",
        "source_prompt_tokens": [int(tok) for tok in prompt_tokens],
        "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
        "concept_rich_output_tokens": [int(tok) for tok in rich_output],
        "concept_avoiding_output_tokens": [int(tok) for tok in avoid_output],
        "concept_rich_output_digest": token_sequence_digest(rich_output),
        "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
        "editable_token_start": span.start,
        "editable_token_count": span.count,
        "original_editable_tokens": original_editable,
    }


# Two concepts, each with EXACTLY ONE core-prefix / ONE (rich, avoid) pair, so a
# concept group holds exactly one pair. Prompts are BOS-prefixed with >= 2
# non-BOS tokens so the k=2 span never collides with BOS. Per-concept distinct
# tokens keep digests / candidate_ids collision-free.
_CONCEPT_SPECS: tuple[dict[str, Any], ...] = (
    {
        "concept_id": 1,
        "prompt_tokens": [0, 91, 92, 93],
        "rich_output": [100],
        "avoid_output": [101],
        "rich_sample_id": "c1-rich",
        "avoid_sample_id": "c1-avoid",
        "core_prefix_id": "coreprefix000001",
    },
    {
        "concept_id": 2,
        "prompt_tokens": [0, 81, 82, 83],
        "rich_output": [110],
        "avoid_output": [111],
        "rich_sample_id": "c2-rich",
        "avoid_sample_id": "c2-avoid",
        "core_prefix_id": "coreprefix000002",
    },
)


def equiv_attack_rows() -> pa.Table:
    """The 6-row one-pair-per-group oracle fixture (ATTACK_ROWS_SCHEMA).

    2 concepts x ``optimized_token_counts=(0,1,2)``. Every row is its OWN
    concept group (single pair), so the universal minibatch is always exactly
    that pair -- the degenerate regime in which a shared trigger must equal the
    per-prefix bespoke trigger byte-for-byte.
    """

    rows: list[dict[str, Any]] = []
    for spec in _CONCEPT_SPECS:
        for k in _OPTIMIZED_TOKEN_COUNTS:
            rows.append(
                _attack_row(
                    concept_id=int(spec["concept_id"]),
                    prompt_tokens=list(spec["prompt_tokens"]),
                    rich_output=list(spec["rich_output"]),
                    avoid_output=list(spec["avoid_output"]),
                    rich_sample_id=str(spec["rich_sample_id"]),
                    avoid_sample_id=str(spec["avoid_sample_id"]),
                    core_prefix_id=str(spec["core_prefix_id"]),
                    k=int(k),
                )
            )
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)


def table_to_parquet_bytes(table: pa.Table) -> bytes:
    """Serialize a table to parquet bytes deterministically.

    ``pyarrow.parquet.write_table`` is byte-deterministic for a fixed table (no
    timestamps / random ids in the container), so re-running the build and
    re-serializing yields identical bytes -- the basis of the smoke test and the
    Phase-3.4 byte-identity assertion.
    """

    buffer = io.BytesIO()
    pq.write_table(table, buffer)
    return buffer.getvalue()
