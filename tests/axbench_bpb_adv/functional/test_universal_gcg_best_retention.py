"""Universal-GCG optimizer: best-so-far retention + multi-pair determinism.

These cover two PR #546-review gaps the single-pair / convergence-to-target
oracle (``_universal_gcg_equiv.py``) cannot exercise:

* #3 -- best-so-far retention. The accept gate runs on the per-step MINIBATCH,
  so for ``n_pairs > universal_minibatch_size`` a trigger that wins on its
  minibatch can be WORSE over the full pair set. The optimizer must emit the
  trigger with the minimum FULL-batch objective it actually achieved, NOT the
  last-accepted one (which can be non-minimal / non-monotone).

* #11 -- multi-pair determinism. The byte-identity golden is scoped to
  single-pair groups at ``top_k=1`` (RNG-independent by construction). This
  pins that the multi-pair minibatch path (``top_k>1``, ``candidate_batch>1``,
  ``n_pairs>minibatch`` -- where the seeded RNG is live) is byte-reproducible
  across re-runs, so a future iteration-order / unseeded-RNG regression fails
  loudly here.

Both use a tiny synthetic distance objective over a small token space with a
per-pair "favorite" token. Each pair's gradient points the shared trigger
toward its own favorite, so under ``minibatch_size=1`` the trigger chases
whichever pair the step sampled -- a setup that deterministically produces a
non-monotone full-batch trajectory (the ``_CONCEPT_ID`` below was chosen
empirically for a clean down-then-up path).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import torch as t

from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    token_sequence_digest,
)
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.editable_tokens import select_final_k_editable_span
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import as_batched_gradient

# Token layout: a contiguous allowed span, an init token OUTSIDE it (so the
# initial trigger is never already optimal), embedding = [token_value, 1.0].
_ALLOWED: list[int] = list(range(10, 30))
_INIT_TOKEN = 40
_EMBED_DIM = 2
# Three pairs with distinct favorite tokens spanning the allowed range, so the
# full-batch optimum is interior and the per-pair pulls genuinely conflict.
_FAVORITES: tuple[int, ...] = (12, 20, 28)
# Concept id chosen empirically (its seeded minibatch deal at master_seed=3) for
# a clean down-then-up full-batch trajectory (full obj: 20.0 init -> 9.0 ->
# 7.667 -> 8.0): the LAST accept is strictly WORSE than an earlier one, so
# retaining the best vs the last is observable.
_NONMONOTONE_CONCEPT_ID = 17
# A concept id whose seeded deal makes master_seed flip the optimized trigger
# (ms0 -> [27], ms1 -> [28]); used to prove the deal now varies with master_seed.
_MASTER_SEED_SENSITIVE_CONCEPT_ID = 4


class _Tokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(i)) for i in ids if int(i) != 0)


def _embedding_weight() -> t.Tensor:
    weight = t.zeros(64, _EMBED_DIM, dtype=t.float32)
    for token in [*_ALLOWED, _INIT_TOKEN]:
        weight[token] = t.tensor([float(token), 1.0])
    return weight


def _span(row: dict[str, Any]) -> EditableSpan:
    return EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
    )


def _favorite(row: dict[str, Any], favorites: dict[str, int]) -> int:
    return favorites[str(row["core_prefix_id"])]


def _make_objective(favorites: dict[str, int]):
    """Batched ObjectiveBatchFn: |trigger_token - favorite| per item (exact, so the
    batched optimizer path matches a single-prompt eval bit-for-bit here)."""

    def objective_batch(
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
    ) -> list[float]:
        out: list[float] = []
        for row, tokens in items:
            position = next(iter(_span(row).positions()))
            out.append(float(abs(int(tokens[position]) - _favorite(row, favorites))))
        return out

    return objective_batch


def _make_gradient(favorites: dict[str, int]):
    def gradient_provider(
        row: dict[str, Any], tokens: Sequence[int], _step: int
    ) -> t.Tensor:
        position = next(iter(_span(row).positions()))
        current = int(tokens[position])
        # Rank minimizes gradient . embedding[candidate]; embedding[token] =
        # [token, 1]. Push toward the favorite: if the trigger is above the
        # favorite, prefer SMALLER tokens (+1 on the value axis) and vice versa.
        direction = 1.0 if current > _favorite(row, favorites) else -1.0
        grad = t.zeros(len(tokens), _EMBED_DIM, dtype=t.float32)
        grad[position] = t.tensor([direction, 0.0])
        return grad

    return gradient_provider


def _attack_row(concept_id: int, core_prefix_id: str) -> dict[str, Any]:
    prompt = [0, 90, 91, _INIT_TOKEN]  # last token (init) is the lone editable
    span = select_final_k_editable_span(prompt, k=1)
    return {
        "attack_row_id": f"c{concept_id}|{core_prefix_id}",
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": f"{core_prefix_id}-r",
        "concept_avoiding_sample_id": f"{core_prefix_id}-a",
        "concept_id": concept_id,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 1.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 1,
        "context_signal_mode": "fixed_context",
        "prefix_text": "x",
        "prefix_hash": f"h{core_prefix_id}",
        "source_prompt_tokens": prompt,
        "source_prompt_token_digest": token_sequence_digest(prompt),
        "concept_rich_output_tokens": [100],
        "concept_avoiding_output_tokens": [101],
        "concept_rich_output_digest": token_sequence_digest([100]),
        "concept_avoiding_output_digest": token_sequence_digest([101]),
        "editable_token_start": span.start,
        "editable_token_count": span.count,
        "original_editable_tokens": [_INIT_TOKEN],
    }


def _multi_pair_group(concept_id: int) -> tuple[pa.Table, dict[str, int]]:
    """One concept group of 3 conflicting pairs (favorites span the range)."""

    favorites: dict[str, int] = {}
    rows: list[dict[str, Any]] = []
    for index, favorite in enumerate(_FAVORITES):
        core_prefix_id = f"cp{concept_id:03d}{index}"
        favorites[core_prefix_id] = favorite
        rows.append(_attack_row(concept_id, core_prefix_id))
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA), favorites


def _config(master_seed: int = 0) -> FasterGcgConfig:
    # minibatch_size=1 (each step chases one pair) and candidate_batch_size <<
    # token space so the trajectory unfolds over many steps instead of
    # exhausting the candidate space in one.
    return FasterGcgConfig(
        optimized_token_counts=(1,),
        max_steps=12,
        candidate_batch_size=2,
        top_k=3,
        universal_minibatch_size=1,
        master_seed=master_seed,
        distance_weight=0.0,
        temperature=1.0,
    )


def _run(
    concept_id: int, *, master_seed: int = 0
) -> tuple[pa.Table, list[dict[str, Any]], list[float]]:
    table, favorites = _multi_pair_group(concept_id)
    full_batch_objectives: list[float] = []

    def progress_callback(event: dict[str, Any]) -> None:
        if event.get("event") in ("group_start", "step_done"):
            value = event.get("objective_value")
            if value is not None:
                full_batch_objectives.append(float(value))

    candidates, triggers = build_universal_gcg_candidates(
        attack_rows=table,
        tokenizer=_Tokenizer(),
        cfg=_config(master_seed),
        embedding_weight=_embedding_weight(),
        allowed_token_ids=_ALLOWED,
        objective=_make_objective(favorites),
        gradient_provider_batch=as_batched_gradient(_make_gradient(favorites)),
        progress_callback=progress_callback,
    )
    return candidates, triggers, full_batch_objectives


def test_universal_optimizer_retains_best_not_last_trigger() -> None:
    # master_seed pins a seeded deal whose full-batch trajectory is non-monotone
    # (an early step reaches a strictly better objective than the last-accepted
    # one), giving the best-retention guard below its teeth. The in-test
    # non-monotone assertion fails loud if this seed ever stops exhibiting it.
    candidates, triggers, objectives = _run(_NONMONOTONE_CONCEPT_ID, master_seed=3)
    assert len(triggers) == 1
    emitted = float(triggers[0]["objective_value"])

    # The scenario MUST be non-monotone, else this test has no teeth: the last
    # full-batch objective is strictly worse than an earlier one.
    best_seen = min(objectives)
    assert best_seen < objectives[-1] - 1e-9, (
        f"scenario not non-monotone (seq={objectives}); the best-retention "
        "guard would be untested"
    )

    # The fix: emit the BEST full-batch objective achieved, not the last.
    assert emitted == best_seen
    # ... and the emitted trigger is the one that achieved it (the better token),
    # not the last-accepted (worse) one.
    assert emitted < objectives[-1] - 1e-9
    # Every emitted candidate carries the same retained objective + trigger.
    for objective_value in candidates.column("objective_value").to_pylist():
        assert float(objective_value) == best_seen


def test_universal_multi_pair_build_is_byte_reproducible() -> None:
    # The multi-pair minibatch path runs the seeded RNG live (top_k=3,
    # candidate_batch_size=2, n_pairs=3 > minibatch=1). Two runs at the SAME
    # master_seed must be byte-identical -- the determinism the single-pair
    # top_k=1 golden cannot prove.
    candidates_a, triggers_a, _ = _run(_NONMONOTONE_CONCEPT_ID)
    candidates_b, triggers_b, _ = _run(_NONMONOTONE_CONCEPT_ID)
    assert candidates_a.equals(candidates_b)
    assert triggers_a == triggers_b
    # And the run actually exercised the RNG-live regime (multiple candidates
    # sampled per step at some point), not a degenerate top_k=1 path.
    assert int(triggers_a[0]["visited_sequences"]) > 1


def test_universal_minibatch_deal_varies_with_master_seed() -> None:
    # #546 review #8: the per-step minibatch deal now folds in master_seed, so a
    # different master_seed yields a different optimization trajectory (not just
    # different candidate sampling). For n_pairs > minibatch the deal genuinely
    # differs -> the optimized trigger differs.
    _, triggers_seed0, _ = _run(_MASTER_SEED_SENSITIVE_CONCEPT_ID, master_seed=0)
    _, triggers_seed1, _ = _run(_MASTER_SEED_SENSITIVE_CONCEPT_ID, master_seed=1)
    assert triggers_seed0[0]["trigger_tokens"] != triggers_seed1[0]["trigger_tokens"]
    # Each master_seed is still internally deterministic (re-run is identical).
    _, triggers_seed0_again, _ = _run(
        _MASTER_SEED_SENSITIVE_CONCEPT_ID, master_seed=0
    )
    assert triggers_seed0 == triggers_seed0_again
