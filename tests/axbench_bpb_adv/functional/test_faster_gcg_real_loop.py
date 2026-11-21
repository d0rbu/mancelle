"""Gap E: a SLOW-marked real-loop test for the Faster-GCG optimizer.

The unit tests in ``tests/axbench_bpb_adv/unit/test_optimize.py`` drive
``build_faster_gcg_candidates`` with a constant/stub objective and gradient, so
they never exercise the loop as a genuine *optimization* (gradient -> ranking ->
sampling -> accept/reject across multiple steps). This functional test closes
that gap with the timing-probe knobs (``max_steps=2, candidate_batch_size=4,
top_k=8``) and a real, gradient-consistent objective over a tiny embedding:

* the objective is the embedding value of the single editable token (lower is
  better), so minimizing the search objective genuinely descends toward the
  lowest-value allowed token, and
* the gradient provider returns the exact analytic gradient of that objective,
  so ``rank_replacements_from_gradients`` proposes the descending tokens.

It asserts the production invariants the live faster-GCG run depends on:

* the final candidate's ``objective_value`` is non-null on an optimizable row,
* it is monotone-improved (``<=`` the step-0 objective), and
* it is bit-reproducible under a fixed ``master_seed`` (run twice -> identical).

Marked ``slow`` (deselected by the default ``-m 'not slow ...'``) because it
runs the full multi-step GCG accept/reject loop rather than a single stubbed
step.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import pytest
import torch as t

from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from tests.axbench_bpb_adv.faster_gcg_oracle import build_faster_gcg_candidates

# Vocabulary layout. Token 0 is BOS (stripped by the tokenizer). The editable
# position starts at token 99 (high embedding value) and the allowed set spans a
# descending ladder so the loop has real headroom to improve over two steps.
_BOS = 0
_PREFIX_TOKENS = [_BOS, 97, 98, 99]
_EDITABLE_POSITION = len(_PREFIX_TOKENS) - 1
_START_EDITABLE_TOKEN = 99
# Allowed replacements (id -> embedding value). Lower value == better objective.
_ALLOWED_TOKEN_VALUES: dict[int, float] = {
    99: 9.0,  # the starting token (highest value -> worst objective)
    50: 5.0,
    30: 3.0,
    10: 1.0,  # the global minimum -> best objective
}
_VOCAB_SIZE = max(_ALLOWED_TOKEN_VALUES) + 1


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return _BOS

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != _BOS)


def _embedding_weight() -> t.Tensor:
    weight = t.zeros(_VOCAB_SIZE, 1, dtype=t.float32)
    for token_id, value in _ALLOWED_TOKEN_VALUES.items():
        weight[token_id, 0] = value
    return weight


def _objective(_row: dict[str, Any], tokens: Sequence[int]) -> float:
    """Minimize the embedding value of the editable token (lower is better)."""

    token_id = int(tokens[_EDITABLE_POSITION])
    return float(_ALLOWED_TOKEN_VALUES.get(token_id, 99.0))


def _gradient_provider(
    _row: dict[str, Any],
    tokens: Sequence[int],
    _step: int,
) -> t.Tensor:
    # d objective / d embedding == 1.0 at the editable position (the objective is
    # the embedding value there), 0 elsewhere. This is the exact analytic
    # gradient, so the GCG ranking descends toward lower-value tokens.
    gradient = t.zeros(len(tokens), 1, dtype=t.float32)
    gradient[_EDITABLE_POSITION, 0] = 1.0
    return gradient


def _optimizable_attack_rows() -> pa.Table:
    rich_output = [100]
    avoid_output = [101]
    return pa.Table.from_pylist(
        [
            {
                "attack_row_id": "real-loop-row",
                "core_prefix_id": "coreprefix000001",
                "concept_rich_sample_id": "rich-sample",
                "concept_avoiding_sample_id": "avoid-sample",
                "concept_id": 1,
                "mode": "positive",
                "method_name": "diffmean_decoder_only",
                "target_model": "decoder_only",
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": 1.0,
                "strength_unit": "optimized_tokens",
                "strength_index": 1,
                "context_signal_mode": "fixed_context",
                "prefix_text": "abc",
                "prefix_hash": "hash",
                "source_prompt_tokens": list(_PREFIX_TOKENS),
                "source_prompt_token_digest": token_sequence_digest(_PREFIX_TOKENS),
                "concept_rich_output_tokens": rich_output,
                "concept_avoiding_output_tokens": avoid_output,
                "concept_rich_output_digest": token_sequence_digest(rich_output),
                "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
                "editable_token_start": _EDITABLE_POSITION,
                "editable_token_count": 1,
                "original_editable_tokens": [_START_EDITABLE_TOKEN],
            }
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )


def _timing_probe_cfg() -> FasterGcgConfig:
    # The committed timing-probe knobs (configs/.../axbench_bpb_adv_timing_probe.yaml).
    return FasterGcgConfig(
        optimized_token_counts=(0, 1, 2),
        max_steps=2,
        candidate_batch_size=4,
        top_k=8,
        distance_weight=10.0,
        temperature=0.1,
        master_seed=0,
    )


def _run_real_loop(cfg: FasterGcgConfig) -> dict[str, Any]:
    candidates = build_faster_gcg_candidates(
        attack_rows=_optimizable_attack_rows(),
        tokenizer=_TinyTokenizer(),
        cfg=cfg,
        embedding_weight=_embedding_weight(),
        allowed_token_ids=sorted(_ALLOWED_TOKEN_VALUES),
        objective=_objective,
        gradient_provider=_gradient_provider,
    )
    rows = candidates.to_pylist()
    assert len(rows) == 1
    return rows[0]


@pytest.mark.slow
def test_faster_gcg_real_loop_monotone_improves_objective() -> None:
    cfg = _timing_probe_cfg()
    step_zero_objective = _objective({}, _PREFIX_TOKENS)

    row = _run_real_loop(cfg)

    # Optimizable row -> non-null objective that improved (non-increasing) vs step 0.
    assert row["objective_value"] is not None
    assert row["objective_value"] <= step_zero_objective
    # The gradient-guided loop must make real progress on this descending ladder,
    # not merely hold at the start token.
    assert row["objective_value"] < step_zero_objective
    assert row["attacked_prompt_tokens"][_EDITABLE_POSITION] != _START_EDITABLE_TOKEN


@pytest.mark.slow
def test_faster_gcg_real_loop_deterministic_under_fixed_master_seed() -> None:
    cfg = _timing_probe_cfg()

    first = _run_real_loop(cfg)
    second = _run_real_loop(cfg)

    assert first["objective_value"] == second["objective_value"]
    assert first["attacked_prompt_tokens"] == second["attacked_prompt_tokens"]
    assert first["candidate_id"] == second["candidate_id"]
