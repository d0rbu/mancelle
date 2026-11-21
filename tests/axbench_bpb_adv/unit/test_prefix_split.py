"""Unit tests for split_prefixes_for_concept (prefix_split.py).

Invariants verified:
1. Determinism — same prefixes + seed → identical partition across two calls.
2. Disjoint + complete — optimize and held_out together cover all prefixes exactly.
3. Sizes — n_optimize == int(len(prefixes) * ratio).
4. Per-group independence — two different (concept_id, mode) groups produce
   different orderings/partitions even with the same external seed.
5. Degenerate < 2 prefixes — split RAISES (cannot hold out from < 2), while
   order_prefixes_for_concept tolerates the 0/1-element case.
"""

from __future__ import annotations

import pytest

from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackPrefix
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.prefix_split import (
    order_prefixes_for_concept,
    split_prefixes_for_concept,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_prefix(
    *,
    concept_id: int = 1,
    mode: SteeringMode = SteeringMode.POSITIVE,
    prefix_text: str = "prefix",
    idx: int = 0,
) -> BpbAttackPrefix:
    """Build a minimal BpbAttackPrefix for testing purposes."""
    unique_text = f"{prefix_text}_{idx}"
    return BpbAttackPrefix(
        core_prefix_id=f"cpid_{concept_id}_{mode}_{idx}",
        concept_id=concept_id,
        mode=mode,
        prefix_text=unique_text,
        prefix_hash=f"hash_{idx}",
        prefix_tokens=[0, 1, idx],
        concept_rich_sample_ids=[f"rich_{idx}"],
        concept_avoiding_sample_ids=[f"avoid_{idx}"],
        concept_rich_output_tokens=[[10, idx]],
        concept_avoiding_output_tokens=[[20, idx]],
    )


def _make_prefixes(
    n: int,
    *,
    concept_id: int = 1,
    mode: SteeringMode = SteeringMode.POSITIVE,
) -> list[BpbAttackPrefix]:
    return [
        _make_prefix(concept_id=concept_id, mode=mode, idx=i) for i in range(n)
    ]


# ---------------------------------------------------------------------------
# Test 1: Determinism
# ---------------------------------------------------------------------------


def test_determinism_same_seed_same_partition() -> None:
    """Calling split_prefixes_for_concept twice with identical args yields
    identical (optimize, held_out) partitions."""
    prefixes = _make_prefixes(10)
    opt1, held1 = split_prefixes_for_concept(prefixes, ratio=0.7, seed=42)
    opt2, held2 = split_prefixes_for_concept(prefixes, ratio=0.7, seed=42)
    assert opt1 == opt2
    assert held1 == held2


def test_determinism_golden_partition() -> None:
    """Cross-process determinism: the partition for a fixed input must equal a
    hard-coded golden value computed in a separate process.

    If the implementation were changed to use Python's builtin hash() (which is
    per-process randomised via PYTHONHASHSEED), this test would fail in CI
    because the golden was captured in a different process — the two processes
    would produce different shuffles and the asserted ids would not match.
    """
    prefixes = _make_prefixes(10, concept_id=1, mode=SteeringMode.POSITIVE)
    opt, held = split_prefixes_for_concept(prefixes, ratio=0.7, seed=42)

    # Golden values captured from a separate interpreter invocation.
    expected_optimize_ids = [
        "cpid_1_positive_5",
        "cpid_1_positive_0",
        "cpid_1_positive_1",
        "cpid_1_positive_2",
        "cpid_1_positive_7",
        "cpid_1_positive_6",
        "cpid_1_positive_9",
    ]
    expected_held_ids = [
        "cpid_1_positive_3",
        "cpid_1_positive_8",
        "cpid_1_positive_4",
    ]

    assert [p.core_prefix_id for p in opt] == expected_optimize_ids
    assert [p.core_prefix_id for p in held] == expected_held_ids


def test_different_seeds_typically_differ() -> None:
    """Different seeds should (almost always) produce a different shuffle for
    sufficiently large prefix lists."""
    prefixes = _make_prefixes(12)
    opt1, _ = split_prefixes_for_concept(prefixes, ratio=0.5, seed=0)
    opt2, _ = split_prefixes_for_concept(prefixes, ratio=0.5, seed=1)
    # With 12 elements the probability of a random collision is negligible.
    assert opt1 != opt2


# ---------------------------------------------------------------------------
# Test 2: Disjoint and complete coverage
# ---------------------------------------------------------------------------


def test_disjoint_and_complete_coverage() -> None:
    """optimize and held_out are disjoint and together cover all prefixes."""
    prefixes = _make_prefixes(8)
    opt, held = split_prefixes_for_concept(prefixes, ratio=0.75, seed=7)
    all_returned = opt + held
    # Same elements (as a set) even if order differs between the two parts.
    assert {p.core_prefix_id for p in all_returned} == {
        p.core_prefix_id for p in prefixes
    }
    # No overlap between optimize and held_out.
    opt_ids = {p.core_prefix_id for p in opt}
    held_ids = {p.core_prefix_id for p in held}
    assert opt_ids.isdisjoint(held_ids)
    # Together they account for every input prefix.
    assert len(all_returned) == len(prefixes)


# ---------------------------------------------------------------------------
# Test 3: Sizes match ratio
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("n", "ratio"),
    [
        (10, 0.7),
        (7, 0.5),
        (20, 0.8),
        (3, 0.6),
        (5, 0.0),
        (5, 1.0),
    ],
)
def test_sizes_match_ratio(n: int, ratio: float) -> None:
    """n_optimize == int(len(prefixes) * ratio), held_out gets the rest."""
    prefixes = _make_prefixes(n)
    opt, held = split_prefixes_for_concept(prefixes, ratio=ratio, seed=99)
    expected_n_optimize = int(n * ratio)
    assert len(opt) == expected_n_optimize
    assert len(held) == n - expected_n_optimize


# ---------------------------------------------------------------------------
# Test 4: Per-group independence
# ---------------------------------------------------------------------------


def test_different_concept_ids_split_independently() -> None:
    """Two groups with different concept_ids produce different shuffles even
    with the same external seed — the per-group seed encodes concept_id."""
    prefixes_a = _make_prefixes(10, concept_id=1)
    prefixes_b = _make_prefixes(10, concept_id=2)
    opt_a, _ = split_prefixes_for_concept(prefixes_a, ratio=0.5, seed=42)
    opt_b, _ = split_prefixes_for_concept(prefixes_b, ratio=0.5, seed=42)
    # The prefix objects differ by definition (concept_id encoded in ids), so
    # we compare the *index* positions selected — if the same positional indices
    # were chosen the shuffle would be effectively identical.
    positions_a = {p.core_prefix_id.rsplit("_", 1)[-1] for p in opt_a}
    positions_b = {p.core_prefix_id.rsplit("_", 1)[-1] for p in opt_b}
    assert positions_a != positions_b, (
        "Different concept_ids should produce different shuffles"
    )


def test_different_modes_split_independently() -> None:
    """Two groups that differ only by mode produce different shuffles."""
    prefixes_pos = _make_prefixes(10, mode=SteeringMode.POSITIVE)
    prefixes_neg = _make_prefixes(10, mode=SteeringMode.NEGATIVE)
    opt_pos, _ = split_prefixes_for_concept(prefixes_pos, ratio=0.5, seed=42)
    opt_neg, _ = split_prefixes_for_concept(prefixes_neg, ratio=0.5, seed=42)
    positions_pos = {p.core_prefix_id.rsplit("_", 1)[-1] for p in opt_pos}
    positions_neg = {p.core_prefix_id.rsplit("_", 1)[-1] for p in opt_neg}
    assert positions_pos != positions_neg, (
        "Different modes should produce different shuffles"
    )


# ---------------------------------------------------------------------------
# Test 5: Degenerate case (< 2 prefixes) — split RAISES loudly
# ---------------------------------------------------------------------------


def test_degenerate_zero_prefixes_raises() -> None:
    """Empty input cannot be split into optimize/held-out → loud raise."""
    with pytest.raises(AxBenchBpbAdvError, match=">= 2 prefixes"):
        split_prefixes_for_concept([], ratio=0.7, seed=0)


def test_degenerate_one_prefix_raises() -> None:
    """A single prefix cannot be held out from itself → loud raise naming concept+mode."""
    prefixes = _make_prefixes(1, concept_id=42, mode=SteeringMode.POSITIVE)
    with pytest.raises(AxBenchBpbAdvError, match=">= 2 prefixes") as exc_info:
        split_prefixes_for_concept(prefixes, ratio=0.7, seed=0)
    msg = str(exc_info.value)
    assert "concept_id=42" in msg, f"message should name the concept; got: {msg}"
    assert "mode=positive" in msg, f"message should name the mode; got: {msg}"


# ---------------------------------------------------------------------------
# Test 6: order_prefixes_for_concept — deterministic ordering primitive
# ---------------------------------------------------------------------------


def test_order_empty_passes_through() -> None:
    """Zero prefixes: the ordering is the empty list (no raise)."""
    assert order_prefixes_for_concept([], seed=0) == []


def test_order_single_prefix_passes_through() -> None:
    """One prefix: the ordering is that single prefix (roofline's minimal unit)."""
    prefixes = _make_prefixes(1)
    assert order_prefixes_for_concept(prefixes, seed=0) == prefixes


def test_order_matches_split_ratio_one_ordering() -> None:
    """For >= 2 prefixes the ordering is byte-identical to the old
    split(ratio=1.0) optimize partition — the roofline derive's ordering is
    unchanged by the extraction."""
    prefixes = _make_prefixes(6)
    ordered = order_prefixes_for_concept(prefixes, seed=42)
    opt, held = split_prefixes_for_concept(prefixes, ratio=1.0, seed=42)
    assert held == []
    assert [p.core_prefix_id for p in ordered] == [p.core_prefix_id for p in opt]


def test_order_is_deterministic() -> None:
    """Same prefixes + seed → identical ordering across two calls."""
    prefixes = _make_prefixes(8)
    assert [p.core_prefix_id for p in order_prefixes_for_concept(prefixes, seed=7)] == [
        p.core_prefix_id for p in order_prefixes_for_concept(prefixes, seed=7)
    ]
