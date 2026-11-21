"""Phase-0.2: freeze the per-prefix GCG candidates as the 1-pair oracle.

This captures the backward-compat oracle for the universal-GCG work: running
the CURRENT ``build_faster_gcg_candidates`` on the SHARED single-pair-per-group
fixture (with the SHARED synthetic closures + ``FasterGcgConfig`` from
``_universal_gcg_equiv``) in a deterministic-sampling regime, and pinning the
result to a committed golden parquet. Phase 3.4's
``build_universal_gcg_candidates`` must reproduce these bytes exactly on the
same fixture, proving the universalization is a strict generalisation (a
single-pair group ⇒ today's per-prefix behavior).

The smoke tests here assert:
  * the build is REPRODUCIBLE (re-running ⇒ byte-identical parquet bytes),
  * the freshly-built candidates equal the committed golden bytes,
  * the fixture is genuinely the degenerate one-pair-per-group regime (v2 C1).
"""

from __future__ import annotations

from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA
from steering.eval.axbench_bpb_adv.universal_gcg import concept_groups
from tests.axbench_bpb_adv.faster_gcg_oracle import build_faster_gcg_candidates
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import (
    equiv_allowed_token_ids,
    equiv_attack_rows,
    equiv_config,
    equiv_embedding_weight,
    equiv_gradient_provider,
    equiv_objective,
    equiv_tokenizer,
    golden_parquet_path,
    table_to_parquet_bytes,
)


def _build_per_prefix_candidates():
    cfg = equiv_config()
    return build_faster_gcg_candidates(
        attack_rows=equiv_attack_rows(),
        tokenizer=equiv_tokenizer(),
        cfg=cfg,
        embedding_weight=equiv_embedding_weight(),
        allowed_token_ids=equiv_allowed_token_ids(),
        objective=equiv_objective,
        gradient_provider=equiv_gradient_provider,
    )


def test_oracle_fixture_is_one_pair_per_concept_group() -> None:
    # The backward-compat oracle is a DEGENERATE single-prefix/single-pair group
    # (v2 finding C1): every non-control concept group must hold EXACTLY one row
    # so the universal minibatch == {that pair} every step. Pin it so the
    # equivalence claim cannot silently broaden to a real multi-prefix group.
    groups = concept_groups(equiv_attack_rows().to_pylist())
    assert groups, "expected at least one faster_gcg concept group"
    non_control = [group for group in groups if not group.is_control]
    control = [group for group in groups if group.is_control]
    # 2 concepts x {k=1, k=2} == 4 non-control groups.
    assert len(non_control) == 4
    # 2 concepts x {k=0} == 2 control groups.
    assert len(control) == 2
    for group in groups:
        assert len(group.rows) == 1, (
            f"group {group.key_str} holds {len(group.rows)} rows; the oracle "
            "requires exactly one pair per group"
        )


def test_per_prefix_build_is_reproducible() -> None:
    # pq.write_table is byte-deterministic for a fixed table, and the optimizer
    # is fully deterministic in this regime (top_k=1 ⇒ RNG-independent sampling),
    # so two independent builds must serialize to identical bytes. This is the
    # reproducibility contract the golden depends on.
    first = table_to_parquet_bytes(_build_per_prefix_candidates())
    second = table_to_parquet_bytes(_build_per_prefix_candidates())
    assert first == second


def test_per_prefix_candidates_match_committed_golden() -> None:
    # The crown-jewel oracle: the freshly-built per-prefix candidates must equal
    # the committed golden parquet byte-for-byte. Phase 3.4's universal optimizer
    # is then asserted against this SAME golden, proving 1-pair equivalence.
    candidates = _build_per_prefix_candidates()
    assert candidates.schema.equals(CANDIDATES_SCHEMA)
    # One candidate per attack row (6 rows: 2 concepts x 3 k).
    assert candidates.num_rows == 6

    golden_path = golden_parquet_path()
    assert golden_path.exists(), (
        f"golden parquet missing at {golden_path}; (re)generate it with "
        "`PYTHONPATH=. uv run python "
        "tests/axbench_bpb_adv/functional/_capture_universal_gcg_golden.py`"
    )
    assert table_to_parquet_bytes(candidates) == golden_path.read_bytes()
