"""The candidate progress logger must speak the universal GROUP vocabulary (M6).

A concept group spans many prefixes, so universal-GCG events have no single
``attack_row_id``; they carry ``group_key_str`` / ``n_pairs`` /
``editable_token_count`` (the group vocabulary) and ``group_start`` /
``group_done`` event names. ``_candidate_progress_logger`` reads events with
``event.get(...)`` so it must not ``KeyError`` on the group-shaped events — and
it must still emit informative lines. We drive the REAL events produced by
``build_universal_gcg_candidates`` (model-free synthetic closures) through the
REAL logger and assert it logs without raising.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import torch as t
from loguru import logger

from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    token_sequence_digest,
)
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.editable_tokens import select_final_k_editable_span
from steering.eval.axbench_bpb_adv.model_optimize import _candidate_progress_logger
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import as_batched_gradient

_TARGET_TOKEN = 120
_EMBED_DIM = 2


class _CharTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _embedding_weight() -> t.Tensor:
    weight = t.zeros(200, _EMBED_DIM, dtype=t.float32)
    weight[_TARGET_TOKEN] = t.tensor([-1.0, 0.0])
    return weight


def _objective(row: dict[str, Any], tokens: Sequence[int]) -> float:
    span = EditableSpan(
        start=int(row["editable_token_start"]),
        count=int(row["editable_token_count"]),
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


def _row(
    *,
    k: int,
    prompt_tokens: list[int],
    core_prefix_id: str,
    rich_sample_id: str,
    avoid_sample_id: str,
    rich_output: list[int],
    avoid_output: list[int],
) -> dict[str, Any]:
    span = select_final_k_editable_span(prompt_tokens, k=k)
    start = span.start if k > 0 else len(prompt_tokens)
    return {
        "attack_row_id": f"{core_prefix_id}|{rich_sample_id}|k{k}",
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": rich_sample_id,
        "concept_avoiding_sample_id": avoid_sample_id,
        "concept_id": 3,
        "mode": "positive",
        "method_name": "diffmean_encdec",
        "target_model": "decoder_only",
        "coefficient_str": "8.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(k),
        "strength_unit": "optimized_tokens",
        "strength_index": int(k),
        "context_signal_mode": "not_applicable",
        "prefix_text": "p",
        "prefix_hash": f"hash-{core_prefix_id}",
        "source_prompt_tokens": [int(tok) for tok in prompt_tokens],
        "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
        "concept_rich_output_tokens": [int(tok) for tok in rich_output],
        "concept_avoiding_output_tokens": [int(tok) for tok in avoid_output],
        "concept_rich_output_digest": token_sequence_digest(rich_output),
        "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
        "editable_token_start": start,
        "editable_token_count": int(k) if k > 0 else 0,
        "original_editable_tokens": [
            int(tok) for tok in prompt_tokens[start : start + (k if k > 0 else 0)]
        ],
    }


def _attack_rows() -> pa.Table:
    rows = [
        _row(
            k=2,
            prompt_tokens=[0, 41, 42, 43, 44],
            core_prefix_id="coreprefix000001",
            rich_sample_id="rich-0",
            avoid_sample_id="avoid-0",
            rich_output=[100],
            avoid_output=[150],
        ),
        _row(
            k=2,
            prompt_tokens=[0, 51, 52, 53, 54],
            core_prefix_id="coreprefix000002",
            rich_sample_id="rich-1",
            avoid_sample_id="avoid-1",
            rich_output=[101],
            avoid_output=[151],
        ),
        _row(
            k=0,
            prompt_tokens=[0, 61, 62, 63],
            core_prefix_id="coreprefix000003",
            rich_sample_id="rich-c",
            avoid_sample_id="avoid-c",
            rich_output=[190],
            avoid_output=[191],
        ),
    ]
    return pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)


def test_progress_logger_handles_universal_group_events_without_keyerror() -> None:
    # Capture the REAL events, then feed every one through the REAL logger.
    captured: list[dict[str, Any]] = []
    build_universal_gcg_candidates(
        attack_rows=_attack_rows(),
        tokenizer=_CharTokenizer(),
        cfg=FasterGcgConfig(
            optimized_token_counts=(0, 2),
            max_steps=3,
            candidate_batch_size=1,
            top_k=1,
            distance_weight=0.0,
            temperature=1.0,
            universal_minibatch_size=16,
        ),
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
        objective=lambda _items: [_objective(_r, _t) for _r, _t in _items],
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
        progress_callback=captured.append,
    )

    # The universal builder speaks the group vocabulary: group_start / step_done /
    # group_done, with group_key_str + n_pairs (NOT attack_row_id).
    event_names = {event["event"] for event in captured}
    assert event_names == {"group_start", "step_done", "group_done"}
    assert all("group_key_str" in event for event in captured)
    assert all("attack_row_id" not in event for event in captured)

    # The control group emits a group_done with control=True (no group_start/step).
    control_done = [
        event
        for event in captured
        if event["event"] == "group_done" and event.get("control")
    ]
    assert len(control_done) == 1

    # Drive ALL captured events through the real logger; it must not raise AND it
    # must emit informative lines for the group events (M6: the logger learns the
    # group vocabulary instead of silently skipping group_start/group_done). Use
    # log_every=1 so every event passes the throttle and the formatter runs.
    logger_callback = _candidate_progress_logger(
        family="faster_gcg",
        target_name="decoder_only",
        log_every=1,
    )
    lines: list[str] = []
    sink_id = logger.add(lambda message: lines.append(str(message)), level="INFO")
    try:
        for event in captured:
            logger_callback(event)
    finally:
        logger.remove(sink_id)

    text = "\n".join(lines)
    # The group identity (key_str) and n_pairs surface in the log output.
    group_key_strs = {event["group_key_str"] for event in captured}
    for key_str in group_key_strs:
        assert key_str in text
    # group_start and group_done both produce a line (not silently dropped).
    assert "start" in text
    assert "done" in text
    # The non-control group reports a non-zero pair population somewhere.
    assert "n_pairs=2" in text or "pairs=2" in text


def test_step_done_key_set_is_narrow_for_token_trigger_path() -> None:
    """step_done events carry only the narrow step_event_fields (group_key_str + n_pairs).

    Pins that the TokenTriggerEditSpace.step_event_fields seam restores the
    original inline-loop field set: step_done must NOT include the full 8-field
    group identity (method_name, target_model, etc.), only the two group-identity
    fields group_key_str and n_pairs, plus the fixed loop fields.
    """

    _EXPECTED_STEP_DONE_KEYS = {
        "event",
        "group_index",
        "total_groups",
        "group_key_str",
        "n_pairs",
        "step_index",
        "max_steps",
        "candidate_count",
        "accepted",
        "objective_value",
        "best_value",
        "visited_sequences",
    }

    captured: list[dict[str, Any]] = []
    build_universal_gcg_candidates(
        attack_rows=_attack_rows(),
        tokenizer=_CharTokenizer(),
        cfg=FasterGcgConfig(
            optimized_token_counts=(0, 2),
            max_steps=3,
            candidate_batch_size=1,
            top_k=1,
            distance_weight=0.0,
            temperature=1.0,
            universal_minibatch_size=16,
        ),
        embedding_weight=_embedding_weight(),
        allowed_token_ids=[_TARGET_TOKEN],
        objective=lambda _items: [_objective(_r, _t) for _r, _t in _items],
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
        progress_callback=captured.append,
    )

    step_done_events = [e for e in captured if e["event"] == "step_done"]
    assert step_done_events, "expected at least one step_done event from the non-control group"
    for event in step_done_events:
        assert set(event) == _EXPECTED_STEP_DONE_KEYS, (
            f"step_done event has unexpected keys: "
            f"extra={set(event) - _EXPECTED_STEP_DONE_KEYS!r}, "
            f"missing={_EXPECTED_STEP_DONE_KEYS - set(event)!r}"
        )
