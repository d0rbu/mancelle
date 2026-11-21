"""Functional tests for the universal per-concept GCG optimizer (Phase 3).

Two pillars:

* a MULTI-PAIR group produces ONE shared trigger substituted into every member
  prefix's own last-k span (one candidate per pair, distinct ``candidate_id``s,
  length-preserving), with k=0 controls left intact; and
* THE CROWN JEWEL: on the SHARED single-pair-per-group fixture + synthetic
  closures + config from ``_universal_gcg_equiv`` (the same provenance the
  Phase-0.2 golden was captured from), ``build_universal_gcg_candidates``'
  candidates table is BYTE-IDENTICAL to the committed per-prefix golden parquet.
  That proves the universalization is a strict generalisation: a single-pair
  group reproduces today's per-prefix Faster-GCG bit-for-bit.

Everything here is model-free (synthetic objective/gradient closures), so the
loop logic + aggregation + ranking + emission are exercised without loading a
target model.
"""

from __future__ import annotations

import gc
import weakref
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

import steering.conditional.feature_gcg.candidate_search as candidate_search
import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
from steering.conditional.feature_gcg.core import run_gcg
from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    token_sequence_digest,
)
from steering.conditional.feature_gcg.edit_primitives import (
    token_sequence_digest as _bare_token_sequence_digest,
)
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv.candidate_metadata import (
    FasterGcgFinalMetadata,
    parse_candidate_metadata_json,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, FasterGcgConfig
from steering.eval.axbench_bpb_adv.editable_tokens import select_final_k_editable_span
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import (
    write_evaluation_population,
)
from steering.eval.axbench_bpb_adv.model_optimize import (
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.optimize import (
    TokenTriggerEditSpace,
    build_universal_gcg_candidates,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.universal_gcg import concept_groups
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import (
    as_batched_gradient,
    as_batched_objective,
    equiv_allowed_token_ids,
    equiv_attack_rows,
    equiv_config,
    equiv_embedding_weight,
    equiv_gradient_provider_batch,
    equiv_objective_batch,
    equiv_tokenizer,
    golden_parquet_path,
    table_to_parquet_bytes,
)
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)

# ---------------------------------------------------------------------------
# Multi-pair fixture (one concept group holding several prefixes / pairs).
# ---------------------------------------------------------------------------

_TARGET_TOKEN = 120
_EMBED_DIM = 2
_VOCAB_SIZE = 200


class _CharTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _embedding_weight() -> t.Tensor:
    weight = t.zeros(_VOCAB_SIZE, _EMBED_DIM, dtype=t.float32)
    weight[_TARGET_TOKEN] = t.tensor([-1.0, 0.0])
    return weight


def _write_population(
    *,
    paths: AxBenchBpbAdvPaths,
    rows: pa.Table,
    cfg: AxBenchBpbAdvConfig,
) -> None:
    write_evaluation_population(
        paths.evaluation_population(),
        rows.to_pylist(),
        ours=cfg.ours_gcg,
    )


def _objective(_row: dict[str, Any], tokens: Sequence[int]) -> float:
    """Count editable positions NOT yet the target (minimize; 0 = fully flipped)."""

    span = EditableSpan(
        start=int(_row["editable_token_start"]),
        count=int(_row["editable_token_count"]),
    )
    return float(
        sum(
            1
            for position in span.positions()
            if int(tokens[int(position)]) != _TARGET_TOKEN
        )
    )


def _gradient_provider(
    row: dict[str, Any],
    tokens: Sequence[int],
    _step: int,
) -> t.Tensor:
    span = EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
    )
    gradient = t.zeros(len(tokens), _EMBED_DIM, dtype=t.float32)
    for position in span.positions():
        gradient[int(position)] = t.tensor([1.0, 0.0])
    return gradient


def _multi_pair_row(
    *,
    concept_id: int,
    k: int,
    prompt_tokens: list[int],
    core_prefix_id: str,
    rich_sample_id: str,
    avoid_sample_id: str,
    rich_output: list[int],
    avoid_output: list[int],
    editable_count: int | None = None,
) -> dict[str, Any]:
    span = select_final_k_editable_span(prompt_tokens, k=k)
    count = span.count if editable_count is None else editable_count
    start = span.start
    if count == 0:
        start = len(prompt_tokens)
    attack_row_id = (
        f"c{concept_id}|{core_prefix_id}|{rich_sample_id}|{avoid_sample_id}|k{count}"
    )
    return {
        "attack_row_id": attack_row_id,
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": rich_sample_id,
        "concept_avoiding_sample_id": avoid_sample_id,
        "concept_id": int(concept_id),
        "mode": "positive",
        "method_name": "diffmean_encdec",
        "target_model": "decoder_only",
        "coefficient_str": "8.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(count),
        "strength_unit": "optimized_tokens",
        "strength_index": int(count),
        "context_signal_mode": "not_applicable",
        "prefix_text": "".join(chr(int(tok)) for tok in prompt_tokens if int(tok) != 0),
        "prefix_hash": f"hash-{core_prefix_id}",
        "source_prompt_tokens": [int(tok) for tok in prompt_tokens],
        "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
        "concept_rich_output_tokens": [int(tok) for tok in rich_output],
        "concept_avoiding_output_tokens": [int(tok) for tok in avoid_output],
        "concept_rich_output_digest": token_sequence_digest(rich_output),
        "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
        "editable_token_start": start,
        "editable_token_count": count,
        "original_editable_tokens": [
            int(tok) for tok in prompt_tokens[start : start + count]
        ],
    }


def _multi_pair_attack_rows() -> pa.Table:
    """One k=2 concept group with THREE distinct prefixes (3 pairs) + a k=0 control.

    The three prefixes have DIFFERENT prompt lengths so the shared trigger lands
    at a different absolute ``editable_token_start`` per prefix, and the k=0
    control is a separate (control) group that must pass through untouched.
    """

    k = 2
    rows: list[dict[str, Any]] = []
    prefixes = [
        ("coreprefix000001", [0, 41, 42, 43, 44]),  # start = 3
        ("coreprefix000002", [0, 51, 52, 53, 54, 55, 56]),  # start = 5
        ("coreprefix000003", [0, 61, 62, 63]),  # start = 2
    ]
    for index, (core_prefix_id, prompt_tokens) in enumerate(prefixes):
        rows.append(
            _multi_pair_row(
                concept_id=7,
                k=k,
                prompt_tokens=prompt_tokens,
                core_prefix_id=core_prefix_id,
                rich_sample_id=f"rich-{index}",
                avoid_sample_id=f"avoid-{index}",
                rich_output=[100 + index],
                avoid_output=[150 + index],
            )
        )
    # A k=0 control for the SAME concept (its own control group).
    rows.append(
        _multi_pair_row(
            concept_id=7,
            k=0,
            prompt_tokens=[0, 71, 72, 73],
            core_prefix_id="coreprefix000009",
            rich_sample_id="rich-ctl",
            avoid_sample_id="avoid-ctl",
            rich_output=[190],
            avoid_output=[191],
            editable_count=0,
        )
    )
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)


def _multi_pair_config() -> FasterGcgConfig:
    return FasterGcgConfig(
        optimized_token_counts=(0, 2),
        max_steps=4,
        candidate_batch_size=1,
        top_k=1,  # singleton pool -> deterministic, target-token-only flips
        distance_weight=0.0,
        temperature=1.0,
        universal_minibatch_size=16,
        master_seed=0,
    )


# ---------------------------------------------------------------------------
# (a) multi-pair group => one shared trigger, one candidate per pair
# ---------------------------------------------------------------------------


def test_universal_multi_pair_shares_one_trigger_and_emits_per_pair_candidates() -> None:
    attack_rows = _multi_pair_attack_rows()
    # Pre-assert the fixture is the intended shape: one 3-pair non-control group.
    groups = concept_groups(attack_rows.to_pylist())
    non_control = [group for group in groups if not group.is_control]
    control = [group for group in groups if group.is_control]
    assert len(non_control) == 1
    assert len(non_control[0].rows) == 3
    assert len(control) == 1

    candidates, trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_CharTokenizer(),
        cfg=_multi_pair_config(),
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
        objective=as_batched_objective(_objective),
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
    )

    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    rows = candidates.to_pylist()
    # 3 optimized pairs + 1 control == 4 candidate rows.
    assert len(rows) == 4

    non_control_rows = [row for row in rows if int(row["editable_token_count"]) > 0]
    control_rows = [row for row in rows if int(row["editable_token_count"]) == 0]
    assert len(non_control_rows) == 3
    assert len(control_rows) == 1

    # ONE shared trigger: the replacement (last-k) of every optimized candidate is
    # the SAME two tokens, and the optimizer drove them to the target.
    replacements = {tuple(row["replacement_tokens"]) for row in non_control_rows}
    assert replacements == {(_TARGET_TOKEN, _TARGET_TOKEN)}

    # Exactly ONE trigger row for the non-control group, carrying that trigger +
    # the optimization metadata, and n_pairs == 3.
    assert len(trigger_rows) == 1
    trigger_row = trigger_rows[0]
    assert trigger_row["trigger_tokens"] == [_TARGET_TOKEN, _TARGET_TOKEN]
    assert trigger_row["n_pairs"] == 3
    assert trigger_row["max_steps"] == 4
    assert trigger_row["accepted_steps"] >= 1  # the optimizer changed the trigger
    assert trigger_row["visited_sequences"] >= 1
    assert trigger_row["objective_value"] == 0.0  # whole span flipped to target

    # One candidate per pair with DISTINCT candidate_ids (digests differ by
    # core_prefix / output digests), and every candidate is length-preserving:
    # the shared trigger sits at each prefix's OWN last-k span.
    ids = [row["candidate_id"] for row in non_control_rows]
    assert len(set(ids)) == 3
    for row in non_control_rows:
        assert len(row["attacked_prompt_tokens"]) == len(row["source_prompt_tokens"])
        start = int(row["editable_token_start"])
        count = int(row["editable_token_count"])
        assert row["attacked_prompt_tokens"][start : start + count] == [
            _TARGET_TOKEN,
            _TARGET_TOKEN,
        ]
        # The prefix (everything before the span) is unchanged.
        assert (
            row["attacked_prompt_tokens"][:start]
            == row["source_prompt_tokens"][:start]
        )

    # The k=0 control is byte-identical to the no-adversary control: attacked ==
    # source, no objective, empty replacement.
    control_row = control_rows[0]
    assert control_row["attacked_prompt_tokens"] == control_row["source_prompt_tokens"]
    assert control_row["replacement_tokens"] == []
    assert control_row["objective_value"] is None


def test_token_trigger_builder_stops_after_exact_full_pool_exhaustion() -> None:
    """Exercise the production adapter, sampler, padded digests, and row builders."""

    source_tokens = [0, 41]
    attack_rows = pa.Table.from_pylist(
        [
            _multi_pair_row(
                concept_id=17,
                k=1,
                prompt_tokens=source_tokens,
                core_prefix_id="coreprefix000017",
                rich_sample_id="rich-exhausted",
                avoid_sample_id="avoid-exhausted",
                rich_output=[100],
                avoid_output=[150],
            )
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )
    cfg = FasterGcgConfig(
        optimized_token_counts=(1,),
        max_steps=10,
        candidate_batch_size=8,
        top_k=1,
        distance_weight=0.0,
        temperature=1.0,
        universal_minibatch_size=16,
        master_seed=0,
    )
    events: list[dict[str, Any]] = []
    gradient_calls = 0

    def counted_gradient(
        items: Sequence[tuple[dict[str, Any], Sequence[int]]],
        step: int,
    ) -> list[t.Tensor]:
        nonlocal gradient_calls
        gradient_calls += 1
        return zero_gradient_batch(items, step)

    candidates, trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_CharTokenizer(),
        cfg=cfg,
        embedding_weight=t.zeros(100, 1, dtype=t.float32),
        allowed_token_ids=[41],
        objective=zero_objective,
        gradient_provider_batch=counted_gradient,
        progress_callback=events.append,
    )

    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    assert candidates.num_rows == 1
    assert pa.Table.from_pylist(
        trigger_rows, schema=UNIVERSAL_TRIGGERS_SCHEMA
    ).schema.equals(UNIVERSAL_TRIGGERS_SCHEMA)
    assert gradient_calls == 1
    assert [event["event"] for event in events] == [
        "group_start",
        "step_done",
        "group_done",
    ]
    assert events[1]["step_index"] == 0
    assert events[1]["candidate_count"] == 0

    candidate = candidates.to_pylist()[0]
    assert candidate["attacked_prompt_tokens"] == source_tokens
    assert candidate["replacement_tokens"] == [41]
    metadata = parse_candidate_metadata_json(candidate["candidate_metadata_json"])
    assert isinstance(metadata, FasterGcgFinalMetadata)
    assert int(metadata.max_steps) == 10
    assert int(metadata.accepted_steps) == 0
    assert int(metadata.visited_sequences) == 1

    assert len(trigger_rows) == 1
    trigger = trigger_rows[0]
    assert trigger["trigger_tokens"] == [41]
    assert trigger["max_steps"] == 10
    assert trigger["accepted_steps"] == 0
    assert trigger["visited_sequences"] == 1


def test_universal_strict_subset_minibatch_still_converges_shared_trigger() -> None:
    """With ``universal_minibatch_size`` < ``n_pairs`` the optimizer deals a STRICT
    subset of the group's pairs each step (deal-without-replacement), yet still
    drives the single shared trigger to the target and emits one candidate per
    pair. Exercises the subset-minibatch path through the full optimizer (the
    size>=n_pairs test cannot).
    """

    # FIVE prefixes (5 pairs), minibatch of 2 => strict subsets, multi-epoch.
    k = 2
    prefixes = [
        ("coreprefix000001", [0, 41, 42, 43, 44]),
        ("coreprefix000002", [0, 51, 52, 53, 54]),
        ("coreprefix000003", [0, 61, 62, 63, 64]),
        ("coreprefix000004", [0, 71, 72, 73, 74]),
        ("coreprefix000005", [0, 81, 82, 83, 84]),
    ]
    rows = [
        _multi_pair_row(
            concept_id=11,
            k=k,
            prompt_tokens=prompt_tokens,
            core_prefix_id=core_prefix_id,
            rich_sample_id=f"rich-{index}",
            avoid_sample_id=f"avoid-{index}",
            rich_output=[100 + index],
            avoid_output=[150 + index],
        )
        for index, (core_prefix_id, prompt_tokens) in enumerate(prefixes)
    ]
    attack_rows = pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)
    cfg = FasterGcgConfig(
        optimized_token_counts=(2,),
        max_steps=8,  # enough epochs (ceil(5/2)=3 steps/epoch) to cover all pairs
        candidate_batch_size=1,
        top_k=1,
        distance_weight=0.0,
        temperature=1.0,
        universal_minibatch_size=2,  # STRICT subset of the 5 pairs
        master_seed=0,
    )

    candidates, trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_CharTokenizer(),
        cfg=cfg,
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
        objective=as_batched_objective(_objective),
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
    )

    rows_out = candidates.to_pylist()
    assert len(rows_out) == 5
    assert {tuple(row["replacement_tokens"]) for row in rows_out} == {
        (_TARGET_TOKEN, _TARGET_TOKEN)
    }
    assert len({row["candidate_id"] for row in rows_out}) == 5
    assert len(trigger_rows) == 1
    assert trigger_rows[0]["n_pairs"] == 5
    # The fixed eval batch (all 5 pairs) reports the converged group objective.
    assert trigger_rows[0]["objective_value"] == 0.0


def test_universal_multi_pair_shared_trigger_substituted_at_each_prefix_span() -> None:
    """The shared trigger is the SAME content at every prefix, but its absolute
    position differs because the prefixes have different lengths (last-k of a
    longer prompt starts later). This pins the production-shape contract (v2 C1).
    """

    candidates, _trigger_rows = build_universal_gcg_candidates(
        attack_rows=_multi_pair_attack_rows(),
        tokenizer=_CharTokenizer(),
        cfg=_multi_pair_config(),
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
        objective=as_batched_objective(_objective),
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
    )
    rows = [
        row
        for row in candidates.to_pylist()
        if int(row["editable_token_count"]) > 0
    ]
    starts = {int(row["editable_token_start"]) for row in rows}
    # Three prefixes of lengths 5, 7, 4 with k=2 => starts 3, 5, 2 (all distinct).
    assert starts == {3, 5, 2}
    # ...yet the substituted content is identical across them.
    contents = {
        tuple(
            row["attacked_prompt_tokens"][
                int(row["editable_token_start"]) : int(row["editable_token_start"])
                + int(row["editable_token_count"])
            ]
        )
        for row in rows
    }
    assert contents == {(_TARGET_TOKEN, _TARGET_TOKEN)}


# ---------------------------------------------------------------------------
# (b) THE CROWN JEWEL: 1-pair groups == the per-prefix golden, byte-for-byte
# ---------------------------------------------------------------------------


def test_universal_one_pair_group_candidates_match_per_prefix_golden_bytes() -> None:
    # Provenance: the SAME fixture + synthetic closures + config the Phase-0.2
    # golden was captured from. The only difference vs the golden capture is which
    # optimizer consumes them — so byte-identity here proves the universal path is
    # a strict generalisation (a single-pair group reproduces per-prefix GCG).
    attack_rows = equiv_attack_rows()

    # Guard the regime the byte-identity depends on: EVERY concept group must hold
    # exactly one pair (so the minibatch == {that pair} every step). If this ever
    # broadens to a multi-prefix group the equivalence claim is no longer valid.
    groups = concept_groups(attack_rows.to_pylist())
    assert groups, "expected at least one faster_gcg concept group"
    for group in groups:
        assert len(group.rows) == 1, (
            f"group {group.key_str} holds {len(group.rows)} rows; the equivalence "
            "oracle requires exactly one pair per group"
        )

    candidates, _trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=equiv_tokenizer(),
        cfg=equiv_config(),
        embedding_weight=equiv_embedding_weight(),
        allowed_token_ids=equiv_allowed_token_ids(),
        objective=equiv_objective_batch,
        gradient_provider_batch=equiv_gradient_provider_batch,
    )

    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    # 2 concepts x optimized_token_counts(0,1,2).
    assert candidates.num_rows == 6

    golden_path = golden_parquet_path()
    assert golden_path.exists(), (
        f"golden parquet missing at {golden_path}; (re)generate it with "
        "`PYTHONPATH=. uv run python "
        "tests/axbench_bpb_adv/functional/_capture_universal_gcg_golden.py`"
    )
    # FULL-TABLE byte-identity: every column, including candidate_metadata_json
    # (the universal candidate rows deliberately reuse the per-prefix
    # `faster_gcg_final` metadata; the universal-trigger identity lives in the
    # separate trigger_rows, so NO column is narrowed or dropped here).
    assert table_to_parquet_bytes(candidates) == golden_path.read_bytes()


def test_universal_one_pair_group_candidate_metadata_equals_per_prefix() -> None:
    """Explicit teeth on the metadata column: the 1-pair universal candidates
    carry the IDENTICAL ``candidate_metadata_json`` the per-prefix path emits
    (``faster_gcg_final`` + max_steps/accepted_steps/visited_sequences). This is
    what lets the full-table byte-identity above hold without narrowing.
    """

    candidates, _trigger_rows = build_universal_gcg_candidates(
        attack_rows=equiv_attack_rows(),
        tokenizer=equiv_tokenizer(),
        cfg=equiv_config(),
        embedding_weight=equiv_embedding_weight(),
        allowed_token_ids=equiv_allowed_token_ids(),
        objective=equiv_objective_batch,
        gradient_provider_batch=equiv_gradient_provider_batch,
    )
    metadata_values = {row["candidate_metadata_json"] for row in candidates.to_pylist()}
    # Every k>0 candidate uses the faster_gcg_final metadata; k=0 uses the control
    # metadata. No universal-only key leaks into the candidate metadata column.
    assert any('"candidate":"faster_gcg_final"' in value for value in metadata_values)
    assert any('"candidate":"no_adversary_control"' in value for value in metadata_values)
    for value in metadata_values:
        assert "trigger" not in value
        assert "n_pairs" not in value


def test_universal_k0_only_groups_are_byte_identical_controls() -> None:
    """A table of ONLY k=0 faster_gcg rows yields exactly the control candidates
    (attacked == source, objective None) and NO trigger rows.
    """

    rows = [
        row
        for row in equiv_attack_rows().to_pylist()
        if int(row["editable_token_count"]) == 0
    ]
    attack_rows = pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)

    candidates, trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=equiv_tokenizer(),
        cfg=equiv_config(),
        embedding_weight=equiv_embedding_weight(),
        allowed_token_ids=equiv_allowed_token_ids(),
        objective=as_batched_objective(lambda _row, _tokens: pytest_fail_k0()),
        gradient_provider_batch=lambda _items, _step: pytest_fail_k0(),
    )

    assert trigger_rows == []
    out_rows = candidates.to_pylist()
    assert len(out_rows) == len(rows)
    for row in out_rows:
        assert row["attacked_prompt_tokens"] == row["source_prompt_tokens"]
        assert row["replacement_tokens"] == []
        assert row["objective_value"] is None


def pytest_fail_k0() -> NoReturn:
    raise AssertionError("k=0 control groups must not invoke the objective/gradient")


def _run_multi_pair_space(
    *,
    use_cache: bool,
) -> tuple[list[Any], pa.Table, pa.Table]:
    """Run the same universal flow with the production or raw rank path."""

    attack_rows = _multi_pair_attack_rows()
    cfg = _multi_pair_config()
    space = TokenTriggerEditSpace(
        tokenizer=_CharTokenizer(),
        cfg=cfg,
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
    )
    if not use_cache:
        space._allowed_embedding_cache = None
    results = run_gcg(
        edit_space=space,
        groups=concept_groups(attack_rows.to_pylist()),
        objective_batch=as_batched_objective(_objective),
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
        cfg=cfg,
    )
    candidate_rows: list[dict[str, Any]] = []
    trigger_rows: list[dict[str, Any]] = []
    for result in results:
        candidate_rows.extend(result.payload["candidates"])
        if result.payload["trigger"] is not None:
            trigger_rows.append(result.payload["trigger"])
    return (
        results,
        pa.Table.from_pylist(candidate_rows, schema=CANDIDATES_SCHEMA),
        pa.Table.from_pylist(trigger_rows, schema=UNIVERSAL_TRIGGERS_SCHEMA),
    )


def test_cached_universal_flow_is_byte_identical_to_raw_rank_path() -> None:
    raw_results, raw_candidates, raw_triggers = _run_multi_pair_space(
        use_cache=False
    )
    cached_results, cached_candidates, cached_triggers = _run_multi_pair_space(
        use_cache=True
    )

    assert cached_results == raw_results
    assert table_to_parquet_bytes(cached_candidates) == table_to_parquet_bytes(
        raw_candidates
    )
    assert table_to_parquet_bytes(cached_triggers) == table_to_parquet_bytes(
        raw_triggers
    )


def test_allowed_embedding_pool_materializes_once_across_groups_and_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = _multi_pair_attack_rows().to_pylist()
    duplicated: list[dict[str, Any]] = []
    for row in rows:
        duplicated.append(row)
        if int(row["editable_token_count"]) == 0:
            continue
        copy = dict(row)
        copy["concept_id"] = 8
        copy["attack_row_id"] = f"{row['attack_row_id']}|copy"
        copy["core_prefix_id"] = f"{row['core_prefix_id']}-copy"
        copy["concept_rich_sample_id"] = f"{row['concept_rich_sample_id']}-copy"
        copy["concept_avoiding_sample_id"] = (
            f"{row['concept_avoiding_sample_id']}-copy"
        )
        duplicated.append(copy)
    attack_rows = pa.Table.from_pylist(duplicated, schema=ATTACK_ROWS_SCHEMA)
    assert len([group for group in concept_groups(duplicated) if not group.is_control]) == 2

    original = candidate_search._materialize_allowed_embedding_pool
    calls = 0
    pool_refs: list[weakref.ReferenceType[t.Tensor]] = []

    def counted_materialize(**kwargs):
        nonlocal calls
        calls += 1
        result = original(**kwargs)
        pool_refs.append(weakref.ref(result[1]))
        return result

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        counted_materialize,
    )
    build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_CharTokenizer(),
        cfg=_multi_pair_config(),
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
        objective=as_batched_objective(_objective),
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
    )
    assert calls == 1
    assert len(pool_refs) == 1
    gc.collect()
    assert pool_refs[0]() is None, (
        "the completed builder must release its search cache"
    )


@pytest.mark.parametrize("control_only", [True, False])
def test_control_and_zero_step_builds_do_not_materialize_allowed_pool(
    monkeypatch: pytest.MonkeyPatch,
    control_only: bool,
) -> None:
    rows = _multi_pair_attack_rows().to_pylist()
    if control_only:
        rows = [row for row in rows if int(row["editable_token_count"]) == 0]
        cfg = _multi_pair_config()
        objective = as_batched_objective(
            lambda _row, _tokens: pytest_fail_k0()
        )
    else:
        rows = [row for row in rows if int(row["editable_token_count"]) > 0]
        cfg = replace(_multi_pair_config(), max_steps=0)
        objective = as_batched_objective(_objective)
    attack_rows = pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)

    def fail_materialization(**_kwargs):
        raise AssertionError("zero-work build must not materialize allowed pool")

    monkeypatch.setattr(
        candidate_search,
        "_materialize_allowed_embedding_pool",
        fail_materialization,
    )
    candidates, triggers = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_CharTokenizer(),
        cfg=cfg,
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[-1, _VOCAB_SIZE + 1],
        objective=objective,
        gradient_provider_batch=lambda _items, _step: pytest_fail_k0(),
    )
    assert candidates.num_rows == len(rows)
    if control_only:
        assert triggers == []
    else:
        assert len(triggers) == 1


# ---------------------------------------------------------------------------
# Task 3.3: wire run_model_backed_bpb_candidate_stage to the universal builder
# ---------------------------------------------------------------------------


def test_candidate_stage_writes_shared_trigger_candidates_and_universal_triggers(
    tmp_path,
    monkeypatch,
) -> None:
    """The candidate stage (test embedding path) routes faster_gcg through the
    universal builder: candidates.parquet carries ONE shared trigger per concept
    group (one row per pair), and the stage writes the CANONICAL
    ``universal_triggers.parquet`` (UNIVERSAL_TRIGGERS_SCHEMA) — exactly one row
    per non-control group. The Phase-3 ``.pending.parquet`` stash is gone.

    The embedding-weight smoke path wires a ZERO objective/gradient (D=1), so no
    step is accepted and the shared trigger stays at its deterministic init (the
    smallest-core_prefix's last-k source tokens). That is enough to prove the
    WIRING: the stage shares one trigger across all member prefixes, emits one
    candidate per pair, and persists the canonical trigger table. The REAL
    optimization wiring is covered by the model-free functional tests above and
    the Phase-7 real-tiny-model integration test.
    """

    cfg = AxBenchBpbAdvConfig(
        source_run_dir="/tmp/source",
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path="/tmp/stats",
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(2,),
                max_steps=1,
                candidate_batch_size=1,
                top_k=1,
                distance_weight=0.0,
                temperature=1.0,
                universal_minibatch_size=16,
            )
        ],
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.attack_rows_dir().mkdir(parents=True)
    # A 3-pair k=2 concept group (decoder_only) — the universal builder must share
    # one trigger across all three prefixes.
    attack_rows = _multi_pair_attack_rows()
    optimized = pa.Table.from_pylist(
        [
            row
            for row in attack_rows.to_pylist()
            if int(row["editable_token_count"]) > 0
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )
    pq.write_table(optimized, paths.attack_rows_path())
    _write_population(paths=paths, rows=optimized, cfg=cfg)
    monkeypatch.setattr(
        model_optimize, "build_allowed_token_ids", lambda _tokenizer: [_TARGET_TOKEN]
    )

    written = run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        # The smoke path uses zero_gradient_batch (D=1), so the embedding must be D=1.
        embedding_weight=t.zeros(200, 1),
        test_objective=zero_objective,
        test_gradient_provider_batch=zero_gradient_batch,
    )

    assert written == 3  # one candidate per pair
    candidates = pq.read_table(paths.candidates_path())
    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    rows = candidates.to_pylist()
    assert len(rows) == 3
    # All three pairs share ONE trigger (same replacement content) at their OWN
    # last-k spans (length-preserving). Zero objective never accepts, so the
    # shared trigger is the deterministic init = smallest core_prefix's last-2
    # source tokens (coreprefix000001 -> [43, 44]).
    replacements = {tuple(row["replacement_tokens"]) for row in rows}
    assert replacements == {(43, 44)}
    for row in rows:
        assert len(row["attacked_prompt_tokens"]) == len(row["source_prompt_tokens"])
        start = int(row["editable_token_start"])
        assert row["attacked_prompt_tokens"][start : start + 2] == [43, 44]

    # The canonical universal_triggers artifact: exactly one row for the single
    # non-control group, carrying the shared trigger + n_pairs, under the
    # UNIVERSAL_TRIGGERS_SCHEMA at paths.universal_triggers_path().
    triggers_path = paths.universal_triggers_path()
    assert triggers_path.exists()
    triggers_table = pq.read_table(triggers_path)
    assert triggers_table.schema.equals(UNIVERSAL_TRIGGERS_SCHEMA)
    triggers = triggers_table.to_pylist()
    assert len(triggers) == 1
    assert triggers[0]["n_pairs"] == 3
    assert triggers[0]["concept_id"] == 7
    assert triggers[0]["trigger_tokens"] == [43, 44]
    # The Phase-3 .pending stash is no longer written.
    assert not (paths.candidates_dir() / "universal_triggers.pending.parquet").exists()


# ---------------------------------------------------------------------------
# Task 8: BOS-padded visited-digest contract (teeth test)
# ---------------------------------------------------------------------------
#
# The ``visited`` set in ``run_gcg`` is seeded with ``edit_space.digest(target)``
# where ``target`` is the init trigger.  ``TokenTriggerEditSpace.digest`` returns
# the BOS-PADDED digest ``token_sequence_digest([BOS, *trigger])``.
# ``sample_candidate_sequences`` operates on the BOS-padded view and adds
# BOS-padded digests to ``visited``.  Seeding with the padded digest means a
# step-0 candidate that RE-CREATES the init trigger is already in ``visited``
# and is skipped.  If we instead seed with the BARE digest (the bug), the
# init-recreating candidate is NOT in ``visited`` and is sampled as an
# additional candidate, incrementing ``visited_sequences`` in the final metadata
# and changing the parquet bytes.
#
# To guarantee the init-recreating candidate is reachable at step 0:
#  - ``_TEETH_INIT_TOKEN`` is in ``allowed_token_ids`` (the pool can propose it)
#  - ``top_k=2`` puts BOTH tokens (target + init) in proposals
#  - ``candidate_batch_size=2`` tries to sample BOTH, so the init-recreating
#    candidate is always drawn (not just with some RNG probability)
#  - ``distance_weight=0.0`` removes the regularizer that penalises same-token
#    replacements, so ``_TEETH_INIT_TOKEN`` scores 0 (vs -1 for target) and
#    lands in top-2
#  - ``max_steps=1`` (one step is sufficient to expose the difference)

def test_candidate_stage_preflights_invalid_direction_scale(
    tmp_path, monkeypatch
) -> None:
    """The production candidate branch rejects a bad direction artifact up front."""
    source_run_dir = tmp_path / "source"
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(source_run_dir),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path=str(
            tmp_path / "unused-feature-statistics.safetensors"
        ),
        adversarial_methods=[_multi_pair_config()],
    )
    direction_path = SourcePaths(source_run_dir).diffmean_direction(
        "decoder_only", "positive", 7
    )
    direction_path.parent.mkdir(parents=True)
    save_file(
        {"direction": t.ones(4)},
        str(direction_path),
        metadata={
            "concept_id": "7",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": str(cfg.diffmean_and_probe_layer),
            "max_activation": "0.0",
        },
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.attack_rows_dir().mkdir(parents=True)
    # _multi_pair_attack_rows() carries faster_gcg diffmean rows for concept_id=7.
    attack_rows = _multi_pair_attack_rows()
    pq.write_table(attack_rows, paths.attack_rows_path())
    _write_population(paths=paths, rows=attack_rows, cfg=cfg)

    class _StubBundle:
        model = object()
        tokenizer = _CharTokenizer()

    class _StubState:
        def decoder_only(self) -> _StubBundle:
            return _StubBundle()

        def offload_models(self) -> None:  # called in the stage's finally block
            return None

    monkeypatch.setattr(model_optimize, "RuntimeState", lambda **_kw: _StubState())
    monkeypatch.setattr(
        model_optimize, "detect_target_model_metadata", lambda *_a, **_k: None
    )
    monkeypatch.setattr(model_optimize, "_build_token_bytes", lambda *_a, **_k: {})
    monkeypatch.setattr(
        model_optimize,
        "_read_concepts",
        lambda *_a, **_k: {7: SimpleNamespace(concept_id=7, feature_index=5)},
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="zero-direction max_activation mismatch",
    ):
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=_CharTokenizer(),
            embedding_weight=None,  # -> production elif branch (not the test path)
        )


_TEETH_INIT_TOKEN = 50   # init trigger token; in allowed_token_ids
_TEETH_TARGET_TOKEN = 121  # gradient drives toward this; distinct from _TARGET_TOKEN=120
_TEETH_EMBED_DIM = 2
_TEETH_VOCAB_SIZE = 200


def _teeth_embedding_weight() -> t.Tensor:
    """Place ``_TEETH_TARGET_TOKEN`` at ``[-1.0, 0.0]``; all others at zero.

    Combined with a gradient of ``[1.0, 0.0]`` at every editable position, the
    first-order GCG score for ``_TEETH_TARGET_TOKEN`` is -1 and for
    ``_TEETH_INIT_TOKEN`` (embedding = [0, 0]) is 0.  Both land in top-2 with
    ``distance_weight=0.0``.
    """
    weight = t.zeros(_TEETH_VOCAB_SIZE, _TEETH_EMBED_DIM, dtype=t.float32)
    weight[_TEETH_TARGET_TOKEN] = t.tensor([-1.0, 0.0])
    return weight


def _teeth_objective(row: dict[str, Any], tokens: Sequence[int]) -> float:
    """Count editable positions NOT yet equal to ``_TEETH_TARGET_TOKEN`` (minimize)."""
    span = EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
    )
    return float(
        sum(
            1
            for position in span.positions()
            if int(tokens[int(position)]) != _TEETH_TARGET_TOKEN
        )
    )


def _teeth_gradient_provider(
    row: dict[str, Any],
    tokens: Sequence[int],
    _step: int,
) -> t.Tensor:
    """``[1.0, 0.0]`` gradient at each editable position."""
    span = EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
    )
    gradient = t.zeros(len(tokens), _TEETH_EMBED_DIM, dtype=t.float32)
    for position in span.positions():
        gradient[int(position)] = t.tensor([1.0, 0.0])
    return gradient


def _teeth_attack_row() -> pa.Table:
    """A single k=1 pair whose last-prompt-token IS ``_TEETH_INIT_TOKEN``.

    Prompt: ``[BOS=0, 51, 50]`` → k=1 editable span starts at position 2,
    so the init trigger is exactly ``[_TEETH_INIT_TOKEN]``.  This is the token
    that step-0 can re-create (the no-op replacement), which is the candidate
    the padded-vs-bare seed difference gates.
    """
    prompt_tokens = [0, 51, _TEETH_INIT_TOKEN]  # BOS, filler, init_token
    k = 1
    span = select_final_k_editable_span(prompt_tokens, k=k)
    row: dict[str, Any] = {
        "attack_row_id": f"c99|cpteeth|rich-t|avoid-t|k{k}",
        "core_prefix_id": "cpteeth",
        "concept_rich_sample_id": "rich-t",
        "concept_avoiding_sample_id": "avoid-t",
        "concept_id": 99,
        "mode": "positive",
        "method_name": "diffmean",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(k),
        "strength_unit": "optimized_tokens",
        "strength_index": int(k),
        "context_signal_mode": "not_applicable",
        "prefix_text": "".join(
            chr(int(tok)) for tok in prompt_tokens if int(tok) != 0
        ),
        "prefix_hash": "hash-cpteeth",
        "source_prompt_tokens": [int(tok) for tok in prompt_tokens],
        "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
        "concept_rich_output_tokens": [170],
        "concept_avoiding_output_tokens": [171],
        "concept_rich_output_digest": token_sequence_digest([170]),
        "concept_avoiding_output_digest": token_sequence_digest([171]),
        "editable_token_start": span.start,
        "editable_token_count": span.count,
        "original_editable_tokens": [
            int(tok) for tok in prompt_tokens[span.start : span.start + span.count]
        ],
    }
    return pa.Table.from_pylist([row], schema=ATTACK_ROWS_SCHEMA)


def _teeth_config() -> FasterGcgConfig:
    """Config that guarantees the init-recreating candidate is reachable at step 0.

    ``top_k=2`` → both ``_TEETH_TARGET_TOKEN`` (score -1) and ``_TEETH_INIT_TOKEN``
    (score 0) are in proposals.  ``candidate_batch_size=2`` → the sampling loop
    always draws both distinct candidates (2 slots, 2 non-visited options in the
    bare case; only 1 in the padded case because init is pre-excluded).
    ``distance_weight=0.0`` → no penalty for the same-token replacement.
    ``max_steps=1`` → one step is sufficient to expose the visited-count difference.
    """
    return FasterGcgConfig(
        optimized_token_counts=(1,),
        max_steps=1,
        candidate_batch_size=2,
        top_k=2,
        distance_weight=0.0,
        temperature=1.0,
        master_seed=0,
    )


def _build_universal_one_pair_init_in_allowed() -> (
    tuple[pa.Table, list[dict[str, Any]]]
):
    """Build universal GCG candidates for the padded-visited-seed teeth fixture."""
    return build_universal_gcg_candidates(
        attack_rows=_teeth_attack_row(),
        tokenizer=_CharTokenizer(),
        cfg=_teeth_config(),
        embedding_weight=_teeth_embedding_weight(),
        allowed_token_ids=[_TEETH_INIT_TOKEN, _TEETH_TARGET_TOKEN],
        objective=as_batched_objective(_teeth_objective),
        gradient_provider_batch=as_batched_gradient(_teeth_gradient_provider),
    )


def _table_bytes(table: pa.Table) -> bytes:
    return table_to_parquet_bytes(table)


def test_padded_visited_seed_is_load_bearing(monkeypatch: Any) -> None:
    """Prove ``TokenTriggerEditSpace.digest`` returning the BOS-padded digest is
    byte-identity-critical.

    Two invariants:

    1. DETERMINISM: two identical runs produce identical parquet bytes.
    2. LOAD-BEARING: patching ``digest`` to return the BARE trigger digest
       (the wrong digest space) changes the parquet bytes.

    Mechanism: with the CORRECT padded seed, the init-recreating step-0
    candidate has digest ``token_sequence_digest([BOS, init_token])`` which
    MATCHES the seed in ``visited`` → it is filtered out → only 1 candidate
    is sampled → ``visited_sequences=2`` in the output metadata.

    With the bare patch, the seed is ``token_sequence_digest([init_token])``
    which does NOT match the BOS-padded digest of the init-recreating candidate
    → that candidate is NOT filtered → both candidates are sampled →
    ``visited_sequences=3`` → different parquet bytes.
    """
    # -- Part 1: determinism ------------------------------------------------
    table1, _ = _build_universal_one_pair_init_in_allowed()
    table2, _ = _build_universal_one_pair_init_in_allowed()
    assert _table_bytes(table1) == _table_bytes(table2), (
        "build_universal_gcg_candidates is not deterministic for the teeth fixture"
    )

    # -- Part 2: padded seed is load-bearing --------------------------------
    monkeypatch.setattr(
        TokenTriggerEditSpace,
        "digest",
        lambda self, trigger: _bare_token_sequence_digest(trigger),
    )
    bared, _ = _build_universal_one_pair_init_in_allowed()
    assert _table_bytes(bared) != _table_bytes(table1), (
        "patching digest to bare did NOT change the output — the init-recreating "
        "candidate may not be reachable at step 0; check that _TEETH_INIT_TOKEN "
        "is in allowed_token_ids, top_k >= 2, and candidate_batch_size >= 2"
    )


# ---------------------------------------------------------------------------
# Merge-regression: the hoisted max_act preflight must stay LAZY
# ---------------------------------------------------------------------------


def test_candidate_stage_with_no_optimizable_rows_never_reads_concepts(
    tmp_path: Path,
) -> None:
    """Zero optimizable prompt rows ⇒ the noop candidates build WITHOUT source I/O.

    The #594-merge hoisted the max_act preflight out of the ``elif prompt_rows:``
    gate, making ``_read_concepts(source_run_dir)`` run eagerly on every real-path
    candidate build — a run whose rows are all k=0 controls (nothing to optimize)
    on a source without ``concepts.json`` crashed before building its noop table.
    This pins the restored laziness: the source_run_dir here does NOT exist at
    all, so any concepts/stats read raises — the stage must not perform one.
    """
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "does-not-exist"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/tmp/features",
        feature_statistics_path=str(tmp_path / "no-stats.safetensors"),
        base_methods=["diffmean_decoder_only"],  # no gcg methods -> no derive
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    paths.attack_rows_dir().mkdir(parents=True)
    control_only = pa.Table.from_pylist(
        [
            _multi_pair_row(
                concept_id=7,
                k=0,
                prompt_tokens=[0, 71, 72, 73],
                core_prefix_id="coreprefix000009",
                rich_sample_id="rich-ctl",
                avoid_sample_id="avoid-ctl",
                rich_output=[190],
                avoid_output=[191],
                editable_count=0,
            )
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )
    pq.write_table(control_only, paths.attack_rows_path())
    _write_population(paths=paths, rows=control_only, cfg=cfg)

    written = run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        embedding_weight=None,  # the REAL path — the one the hoist regressed
    )

    assert written == 1  # the k=0 control noop candidate
    candidates = pq.read_table(paths.candidates_path())
    assert candidates.num_rows == 1
    assert candidates.schema.equals(CANDIDATES_SCHEMA)
