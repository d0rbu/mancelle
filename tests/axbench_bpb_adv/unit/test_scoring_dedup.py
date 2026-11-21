"""Issue #511: build_bpb_adv_score_table memoizes the dual-completion forward
passes so twin candidates (identical except candidate_id / attack_row_id) are
scored ONCE, byte-identically."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pyarrow as pa
import pytest
import torch as t

import steering.eval.axbench_bpb_adv.scoring as scoring
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb.types import BpbScoreRow, ScoringInputs
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA, SCORE_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import (
    DualCompletionScoreFn,
    _scoring_identity_key,
    build_bpb_adv_score_table,
)


class _TinyTokenizer:
    def get_vocab_size(self) -> int:
        return 256

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


_SOURCE = [0, 97, 98]
_RICH = [99, 100]
_AVOID = [101, 102, 103]


def _k0_control(*, candidate_id: str, attack_row_id: str) -> dict[str, Any]:
    """A k=0 no-adversary control candidate (empty span, attacked == source)."""
    return {
        "candidate_id": candidate_id,
        "attack_row_id": attack_row_id,
        "core_prefix_id": "core0",
        "concept_rich_sample_id": "rich_sample",
        "concept_avoiding_sample_id": "avoid_sample",
        "concept_id": 1,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 0.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 0,
        "context_signal_mode": "fixed_context",
        "prefix_text": "ab",
        "prefix_hash": "hash",
        "source_prompt_tokens": list(_SOURCE),
        "source_prompt_token_digest": token_sequence_digest(_SOURCE),
        "attacked_prompt_text": "ab",
        "attacked_prompt_tokens": list(_SOURCE),
        "attacked_prompt_token_digest": token_sequence_digest(_SOURCE),
        "concept_rich_output_tokens": list(_RICH),
        "concept_avoiding_output_tokens": list(_AVOID),
        "concept_rich_output_digest": token_sequence_digest(_RICH),
        "concept_avoiding_output_digest": token_sequence_digest(_AVOID),
        "editable_token_start": len(_SOURCE),
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": None,
        "candidate_metadata_json": '{"candidate":"no_adversary_control"}',
    }


def _k1_attacked(*, candidate_id: str, attack_row_id: str) -> dict[str, Any]:
    """A k=1 adversarial candidate: attacked tokens differ from source."""
    attacked = [0, 97, 120]
    row = _k0_control(
        candidate_id=candidate_id,
        attack_row_id=attack_row_id,
    )
    row.update(
        {
            "adversarial_strength": 1.0,
            "strength_index": 1,
            "attacked_prompt_text": "ax",
            "attacked_prompt_tokens": attacked,
            "attacked_prompt_token_digest": token_sequence_digest(attacked),
            "editable_token_start": 2,
            "editable_token_count": 1,
            "replacement_tokens": [120],
            "replacement_text": "x",
            "objective_value": 0.5,
            "candidate_metadata_json": (
                '{"accepted_steps":0,"candidate":"faster_gcg_final",'
                '"max_steps":1,"visited_sequences":1}'
            ),
        }
    )
    return row


def _table(rows: list[dict[str, Any]]) -> pa.Table:
    return pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)


def _make_score_row(
    candidate_row: Mapping[str, Any], si: ScoringInputs, *, bpb: float, n_bytes: int = 2
) -> BpbScoreRow:
    """Build a BpbScoreRow whose validated fields mirror the candidate row.

    Shared by every fake score_pair in this module so the 11-field BpbScoreRow
    constructor lives in one place.
    """
    return BpbScoreRow(
        method_name=str(candidate_row["method_name"]),
        concept_id=int(candidate_row["concept_id"]),
        mode=str(candidate_row["mode"]),
        content_class=si.content_class,
        target=str(candidate_row["target_model"]),
        coefficient_str=str(candidate_row["coefficient_str"]),
        sample_id=si.sample_id,
        bpb_intervened=bpb,
        n_output_tokens=si.n_output,
        n_output_bytes=n_bytes,
        truncated=False,
    )


def _counting_score_pair() -> tuple[
    DualCompletionScoreFn,
    list[str],
]:
    """A deterministic score_pair (loss from adversarial_strength) that records the
    candidate_id of every REAL invocation, so call count == unique scorings."""
    calls: list[str] = []

    def _pair(candidate_row: Mapping[str, Any], rich_input: ScoringInputs, avoid_input: ScoringInputs):
        calls.append(str(candidate_row["candidate_id"]))
        strength = float(candidate_row["adversarial_strength"])
        return (
            _make_score_row(candidate_row, rich_input, bpb=1.0 + strength, n_bytes=2),
            _make_score_row(candidate_row, avoid_input, bpb=3.0, n_bytes=5),
        )

    return _pair, calls


def _build(table: pa.Table, score_pair: DualCompletionScoreFn) -> pa.Table:
    return build_bpb_adv_score_table(
        candidates=table,
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=score_pair,
    )


def test_k0_control_twins_scored_once() -> None:
    # 4 twin k=0 controls (identical except candidate_id/attack_row_id)
    # + 1 distinct k=1 candidate => 2 unique scorings, not 5.
    rows = [
        _k0_control(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(4)
    ]
    rows.append(_k1_attacked(candidate_id="c_adv", attack_row_id="r_adv"))
    score_pair, calls = _counting_score_pair()

    scores = _build(_table(rows), score_pair)

    assert scores.num_rows == 5  # one score row per candidate (no rows dropped)
    assert len(calls) == 2  # 1 for the k=0 twin group + 1 for the distinct k=1


def test_k0_twins_reuse_first_computation_not_recompute() -> None:
    # TEETH: a score_pair returning a DIFFERENT value each call. If memoization
    # broke, the twins would diverge; with reuse they all carry the FIRST value.
    state = {"n": 0}

    def _pair(candidate_row, rich_input, avoid_input):
        state["n"] += 1
        v = float(state["n"])
        return (
            _make_score_row(candidate_row, rich_input, bpb=v),
            _make_score_row(candidate_row, avoid_input, bpb=v + 100.0),
        )

    rows = [
        _k0_control(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(4)
    ]
    scores = _build(_table(rows), _pair).to_pylist()

    assert {row["loss_concept_rich"] for row in scores} == {1.0}
    assert {row["loss_concept_avoiding"] for row in scores} == {101.0}
    assert state["n"] == 1  # the forward pass ran exactly once


def test_output_is_exact_per_row_independent_scoring() -> None:
    # Exact-equality vs independent reconstruction: each output row's losses equal
    # the deterministic fake applied to its own candidate, twins share the score,
    # and the id passthroughs stay distinct. Row order preserved.
    rows = [
        _k0_control(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(4)
    ]
    rows.append(_k1_attacked(candidate_id="c_adv", attack_row_id="r_adv"))
    score_pair, _calls = _counting_score_pair()

    scores = _build(_table(rows), score_pair).to_pylist()

    assert [r["candidate_id"] for r in scores] == ["c0", "c1", "c2", "c3", "c_adv"]
    for r in scores:
        strength = float(r["adversarial_strength"])
        assert r["loss_concept_rich"] == pytest.approx(1.0 + strength)
        assert r["loss_concept_avoiding"] == pytest.approx(3.0)
    # k=0 twins all share identical scoring outputs.
    k0 = [r for r in scores if r["editable_token_count"] == 0]
    assert {r["loss_concept_rich"] for r in k0} == {1.0}
    assert all(r["is_control"] is True for r in k0)


def test_scores_schema_preserved() -> None:
    rows = [_k0_control(candidate_id="c0", attack_row_id="r0")]
    score_pair, _calls = _counting_score_pair()
    scores = _build(_table(rows), score_pair)
    assert scores.schema.equals(SCORE_SCHEMA)


# --- Key COMPLETENESS + EXCLUSION (pure key, no scoring path) ------------------


def _mutate(value: Any) -> Any:
    """A guaranteed-different, type-preserving value for any candidate cell."""
    if isinstance(value, bool):
        return not value
    if isinstance(value, int):
        return value + 1
    if isinstance(value, float):
        return value + 1.0
    if isinstance(value, str):
        return value + "_x"
    if isinstance(value, list):
        return [*value, 0]
    if value is None:
        return 0.0
    return value


def test_scoring_identity_key_covers_every_column_except_the_two_excluded() -> None:
    # Completeness AND exclusion in one sweep: mutating ANY column changes the key
    # iff that column is part of the scoring identity. Exactly candidate_id /
    # attack_row_id are excluded (so mutating them leaves the key unchanged); every
    # other column is in the key. Robust to schema growth.
    base = _k0_control(candidate_id="c0", attack_row_id="r0")
    excluded = {"candidate_id", "attack_row_id"}
    base_key = _scoring_identity_key(base)
    for name in CANDIDATES_SCHEMA.names:
        other = dict(base)
        other[name] = _mutate(base[name])
        if name in excluded:
            assert _scoring_identity_key(other) == base_key, (
                f"{name} must be EXCLUDED from the scoring identity key"
            )
        else:
            assert _scoring_identity_key(other) != base_key, (
                f"{name} must be PART of the scoring identity key"
            )


# --- The key is actually WIRED into the dedup loop ----------------------------


@pytest.mark.parametrize("field", ["candidate_id", "attack_row_id"])
def test_dedups_when_only_excluded_field_differs(field: str) -> None:
    base = _k0_control(candidate_id="c0", attack_row_id="r0")
    twin = dict(base)
    # candidate_id must always differ (the dup-id guard requires distinct ids).
    twin["candidate_id"] = "c1"
    if field == "attack_row_id":
        twin["attack_row_id"] = "r1"
    score_pair, calls = _counting_score_pair()
    scores = _build(_table([base, twin]), score_pair)
    assert scores.num_rows == 2
    assert len(calls) == 1  # deduped: one forward-pass evaluation


def test_does_not_dedup_when_attacked_tokens_differ() -> None:
    # An in-key field (attacked prompt) splits the cache in the real loop.
    base = _k0_control(candidate_id="c0", attack_row_id="r0")
    other = _k1_attacked(candidate_id="c1", attack_row_id="r1")
    score_pair, calls = _counting_score_pair()
    _build(_table([base, other]), score_pair)
    assert len(calls) == 2


def test_does_not_dedup_when_coefficient_differs() -> None:
    base = _k0_control(candidate_id="c0", attack_row_id="r0")
    other = dict(base)
    other["candidate_id"] = "c1"
    other["coefficient_str"] = "2.0"
    score_pair, calls = _counting_score_pair()
    _build(_table([base, other]), score_pair)
    assert len(calls) == 2


# --- Validation runs per candidate, even after a cache-populating candidate ----


def test_digest_validation_runs_for_candidate_after_warm_cache() -> None:
    # good_a populates the cache; good_b is a genuine cache HIT (same scoring
    # identity, differs only in ids); `bad` is a DISTINCT (cache-miss) candidate
    # processed third. Per-candidate digest validation
    # (candidate_row_to_scoring_inputs -> _validate_candidate_digests) runs for
    # EVERY row BEFORE the cache lookup, so it must still fire for `bad` even
    # though the cache is already warm. `bad`'s replacement_tokens [121] do not
    # match its attacked span [120], so it raises. (The POST-cache
    # _validate_side_row path cannot FAIL on a hit -- every field it checks is in
    # the scoring-identity key, so a hit's cached row always matches -- so it is
    # covered by a call-count spy instead; see the next test.)
    good_a = _k0_control(candidate_id="c0", attack_row_id="r0")
    good_b = _k0_control(candidate_id="c1", attack_row_id="r1")
    bad = _k1_attacked(candidate_id="c2", attack_row_id="r2")
    bad["replacement_tokens"] = [121]  # attacked span has [120], not [121]
    score_pair, _calls = _counting_score_pair()
    with pytest.raises(AxBenchBpbAdvError, match="replacement_tokens"):
        _build(_table([good_a, good_b, bad]), score_pair)


def test_side_row_validation_runs_per_candidate_not_per_unique(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Byte-identity invariant: per-candidate validation is NOT gated by the score
    # cache. _validate_side_row must run for EVERY candidate (both sides),
    # including cache hits -- not once per UNIQUE scoring identity. Spy it: with 4
    # k=0 twins (1 unique scoring, 3 hits) + 1 distinct k=1, score_pair runs
    # twice, but _validate_side_row must run 2x per candidate = 10. A regression
    # that moved validation into the cache-miss branch would drop this to 2x2 = 4.
    real = scoring._validate_side_row
    validate_calls = {"n": 0}

    def _spy(**kwargs: Any) -> None:
        validate_calls["n"] += 1
        real(**kwargs)

    monkeypatch.setattr(scoring, "_validate_side_row", _spy)

    rows = [
        _k0_control(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(4)
    ]
    rows.append(_k1_attacked(candidate_id="c_adv", attack_row_id="r_adv"))
    score_pair, calls = _counting_score_pair()

    _build(_table(rows), score_pair)

    assert len(calls) == 2  # scoring deduped to 2 unique evaluations
    assert validate_calls["n"] == 2 * len(rows)  # but validation ran per candidate (10)


def test_duplicate_candidate_id_still_raises() -> None:
    dup = _k0_control(candidate_id="dup", attack_row_id="r0")
    dup2 = _k0_control(candidate_id="dup", attack_row_id="r1")
    score_pair, _calls = _counting_score_pair()
    with pytest.raises(AxBenchBpbAdvError, match="duplicate candidate_id"):
        _build(_table([dup, dup2]), score_pair)


def test_stats_dict_reports_dedup_counts() -> None:
    rows = [
        _k0_control(candidate_id=f"c{i}", attack_row_id=f"r{i}")
        for i in range(4)
    ]
    rows.append(_k1_attacked(candidate_id="c_adv", attack_row_id="r_adv"))
    score_pair, _calls = _counting_score_pair()
    stats: dict[str, int] = {}
    build_bpb_adv_score_table(
        candidates=_table(rows),
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(256, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": 32},
        score_pair=score_pair,
        stats=stats,
    )
    assert stats == {"scored_unique": 2, "cache_hits": 3}
