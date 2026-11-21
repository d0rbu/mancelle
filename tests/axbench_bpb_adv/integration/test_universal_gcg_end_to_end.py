"""Phase 7.1 — the real tiny-model END-TO-END universal-GCG integration test.

This is the fast, deterministic full-pyramid gate: it proves the WHOLE pipeline
runs on the tiny decoder model with the UNIVERSAL optimizer and produces a
NON-FLAT per-k attacked ``general_val`` — the entire point of the
universal-trigger attack.

The flow exercised here, end to end:

  1. Build REAL attack rows for a MULTI-pair concept (2 rich x 1 avoid = 2 pairs
     per prefix) across ``optimized_token_counts=(0, 1, 2)``, one coefficient,
     decoder-only.
  2. Run the REAL ``build_universal_gcg_candidates`` (the production builder, real
     optimizer loop) with the SAME synthetic-objective/gradient/embedding pattern
     as ``tests/.../functional/test_generate_samples_real_gcg.py`` (no target
     model is loaded for the optimizer). It returns ``(candidates, trigger_rows)``.
     We persist ``candidates.parquet`` + the canonical ``universal_triggers.parquet``
     (``UNIVERSAL_TRIGGERS_SCHEMA``) under a tmp run dir via ``AxBenchBpbAdvPaths``.
     ASSERTS: one shared trigger per concept group; a DISTINCT trigger per
     k; candidates length-preserving; the k=0 control intact.
  3. Drive the attacked ``run_general_val_bpb_stage`` on the tiny model: the
     generic val docs are tiny-vocab strings the ``_TinyTokenizer`` can encode,
     scored by the REAL tiny GPT under ``Intervention.none`` (a no-op steered
     intervention — we are measuring the TRIGGER's effect on the docs, not a
     steering direction). The model-bound boundaries (RuntimeState, concepts,
     method, the steered-intervention builder) are stubbed; the load-bearing
     ATTACKED-PREFIX path (``_distinct_combos`` -> ``_intervention_signature`` ->
     ``_build_trigger_index`` -> per-k ``_samples_for_combo`` ->
     ``_attack_doc`` last-k substitution -> ``score_triple`` on the real model)
     runs FOR REAL. ASSERTS: per-k ``general_val_bpb`` values; the k>0
     attacked values DIFFER from the k=0 clean value (NON-FLAT).
  4. Aggregate: ``build_bpb_adv_score_table`` -> ``aggregate_bpb_adv_scores``
     (which folds general_val via ``_join_general_val_bpb``) and assert the
     per_candidate/macro ``general_val_bpb`` is joined PER k — NOT
     broadcast flat across the GCG axis — and that the line plots build.

Every assertion is NON-VACUOUS: values are read back and compared (k>0 != k=0,
the joined per_candidate values equal the scored general_val table per k);
presence alone is never asserted.

The model + the mini fixture are loaded LOCALLY here (the conftest caveat),
mirroring the sibling tiny-model flow tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle, score_triple
from steering.eval.axbench_bpb.types import BpbScoreRow, Intervention, SteeringMode
from steering.eval.axbench_bpb_adv import general_val
from steering.eval.axbench_bpb_adv.aggregation import (
    aggregate_bpb_adv_scores,
    plot_bpb_adv_summaries,
)
from steering.eval.axbench_bpb_adv.attack_rows import (
    BpbAttackPrefix,
    BpbAttackWorkSpec,
    core_prefix_id,
    prepare_bpb_attack_rows,
)
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
)
from steering.eval.axbench_bpb_adv.evaluation_population import write_evaluation_population
from steering.eval.axbench_bpb_adv.general_val import run_general_val_bpb_stage
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    GENERAL_VAL_BPB_SCHEMA,
    SCORE_SCHEMA,
    UNIVERSAL_TRIGGERS_SCHEMA,
)
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table
from tests.axbench_base.fixtures.steering_stats import write_steering_stats
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import as_batched_gradient

_FIXTURES = Path(__file__).resolve().parents[2] / "axbench_bpb" / "fixtures"
_TINY_CKPT = _FIXTURES / "tiny_decoder_only" / "ckpt.pt"
_MINI_FIXTURE = _FIXTURES / "mini_axbench_base_run"

_VOCAB = 128
_CONCEPT_ID = 1  # "dogs" in the mini fixture's concepts.json
_PREFIX_TEXT = "abc"
_PREFIX_TOKENS = [0, 97, 98, 99]
_COEFFICIENT_STR = "4.0"
_METHOD = "diffmean_decoder_only"
_DIFFMEAN_LAYER = 1  # valid block index for the 2-layer tiny model
_MAX_SEQ_LEN = 64

# The optimizer installs this single target token into every editable position,
# so each concept group's trigger is that token repeated k times. It is the only
# allowed replacement token.
_TARGET_TOKEN = 121
_TARGET_TOKENS = [_TARGET_TOKEN]


# --------------------------------------------------------------------------- #
# tiny model + tokenizer                                                       #
# --------------------------------------------------------------------------- #
class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return _VOCAB

    def encode(self, text: str) -> list[int]:
        # Raw (no-BOS) encoding: one token per char (ascii code, clamped into the
        # tiny vocab, never the BOS id). general_val.build_general_val_docs adds the
        # single leading BOS additively itself.
        return [max(1, ord(ch) % _VOCAB) for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _tiny_model() -> GPT:
    state = t.load(_TINY_CKPT, map_location="cpu", weights_only=False)
    cfg = GPTConfig(**state["config"])
    with t.device("meta"):
        model = GPT(cfg, pad_vocab_size_to=state["pad_vocab_size_to"])
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(state["model"], strict=True, assign=True)
    model.eval()
    return model


# --------------------------------------------------------------------------- #
# attack-row fixture (one MULTI-pair concept)                                  #
# --------------------------------------------------------------------------- #
def _multi_pair_prefix(
    *, rich_outputs: list[list[int]], avoid_outputs: list[list[int]]
) -> BpbAttackPrefix:
    rich_ids = [
        make_sample_id(
            concept_id=_CONCEPT_ID,
            mode=SteeringMode.POSITIVE,
            content_class="concept_rich",
            prefix_text=_PREFIX_TEXT,
            output_text=f"rich-{idx}",
        )
        for idx in range(len(rich_outputs))
    ]
    avoid_ids = [
        make_sample_id(
            concept_id=_CONCEPT_ID,
            mode=SteeringMode.POSITIVE,
            content_class="concept_avoiding",
            prefix_text=_PREFIX_TEXT,
            output_text=f"avoid-{idx}",
        )
        for idx in range(len(avoid_outputs))
    ]
    return BpbAttackPrefix(
        core_prefix_id=core_prefix_id(
            concept_id=_CONCEPT_ID, mode=SteeringMode.POSITIVE, prefix_text=_PREFIX_TEXT
        ),
        concept_id=_CONCEPT_ID,
        mode=SteeringMode.POSITIVE,
        prefix_text=_PREFIX_TEXT,
        prefix_hash=make_prefix_hash(_PREFIX_TEXT),
        prefix_tokens=list(_PREFIX_TOKENS),
        concept_rich_sample_ids=rich_ids,
        concept_avoiding_sample_ids=avoid_ids,
        concept_rich_output_tokens=[list(out) for out in rich_outputs],
        concept_avoiding_output_tokens=[list(out) for out in avoid_outputs],
    )


# --------------------------------------------------------------------------- #
# synthetic optimizer closures (drive every group toward the single target)    #
# --------------------------------------------------------------------------- #
def _embedding_weight() -> t.Tensor:
    # One-dim surrogate embedding. The single target token is favourable along
    # dim 0, so the gradient (below) drives every editable position toward it.
    weight = t.zeros(_VOCAB, 1, dtype=t.float32)
    weight[_TARGET_TOKEN] = t.tensor([-1.0])
    return weight


def _gradient_provider(
    row: dict[str, Any], tokens: Sequence[int], _step: int
) -> t.Tensor:
    # Push every editable position toward the single target token.
    gradient = t.zeros(len(tokens), 1, dtype=t.float32)
    start = int(row["editable_token_start"])
    stop = start + int(row["editable_token_count"])
    for position in range(start, stop):
        gradient[position][0] = 1.0
    return gradient


def _objective(row: dict[str, Any], tokens: Sequence[int]) -> float:
    # Pair-specific baseline (the rich output's first token differs per pair) MINUS
    # a reward per editable position set to the single target. The optimizer
    # minimizes -> installs the target (changing the prompt); the per-pair baseline
    # makes each pair's final objective distinct.
    pair_base = float(int(row["concept_rich_output_tokens"][0]))
    start = int(row["editable_token_start"])
    edited = sum(1 for token in tokens[start:] if int(token) == _TARGET_TOKEN)
    return pair_base - 10.0 * edited


def _objective_batch(
    items: Sequence[tuple[dict[str, Any], Sequence[int]]],
) -> list[float]:
    # Batched ObjectiveBatchFn (exact synthetic objective -> batched == single).
    return [_objective(row, tokens) for row, tokens in items]


# --------------------------------------------------------------------------- #
# build the real candidates + universal triggers                              #
# --------------------------------------------------------------------------- #
def _faster_gcg_cfg() -> FasterGcgConfig:
    return FasterGcgConfig(
        optimized_token_counts=(0, 1, 2),
        max_steps=4,
        candidate_batch_size=4,
        top_k=2,
        distance_weight=0.0,
        temperature=1.0,
        universal_minibatch_size=16,
        master_seed=0,
    )


def _build_run(output_root: Path) -> tuple[AxBenchBpbAdvPaths, pa.Table, list[dict[str, Any]]]:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(_MINI_FIXTURE),
        output_root=str(output_root),
        run_tag="run",
        feature_dir="/unused/features",
        feature_statistics_path="/unused/stats",
        target_models=["decoder_only"],
        pair_subsample=0,  # keep ALL rich x avoid pairs -> >1 candidate per datapoint
        adversarial_methods=[_faster_gcg_cfg()],
    )
    attack_rows = prepare_bpb_attack_rows(
        prefixes=[
            _multi_pair_prefix(
                rich_outputs=[[100], [105]],  # 2 rich
                avoid_outputs=[[101]],  # x 1 avoid = 2 pairs / datapoint
            )
        ],
        work_specs=[
            BpbAttackWorkSpec(
                method_name=_METHOD,
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str=_COEFFICIENT_STR,
                baseline_family="diffmean",
            )
        ],
        cfg=cfg,
    )
    faster_gcg = cfg.adversarial_methods[0]
    assert isinstance(faster_gcg, FasterGcgConfig)
    candidates, trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_TinyTokenizer(),
        cfg=faster_gcg,
        embedding_weight=_embedding_weight(),
        allowed_token_ids=_TARGET_TOKENS,
        objective=_objective_batch,
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
    )

    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(candidates, paths.candidates_path())
    write_evaluation_population(
        paths.evaluation_population(),
        candidates.to_pylist(),
        ours=cfg.ours_gcg,
    )
    # Persist the canonical universal_triggers.parquet exactly as the candidate
    # stage does (one row per non-control group, UNIVERSAL_TRIGGERS_SCHEMA).
    triggers_table = pa.Table.from_pylist(trigger_rows, schema=UNIVERSAL_TRIGGERS_SCHEMA)
    pq.write_table(triggers_table, paths.universal_triggers_path())
    return paths, candidates, trigger_rows


# --------------------------------------------------------------------------- #
# attacked general_val: a deterministic tiny-model scorer driver               #
# --------------------------------------------------------------------------- #
# Tiny-vocab generic-val docs (printable ascii) the _TinyTokenizer can encode.
# Long enough to slice prefix_tokens(3) + output_tokens(2) after BOS-prefixing.
_GENERAL_VAL_TEXTS = [
    "abcdefghij",
    "klmnopqrst",
    "uvwxyzabcd",
]


def _decoder_only_bundle() -> TargetModelBundle:
    return TargetModelBundle(
        model=_tiny_model(),
        tokenizer=_TinyTokenizer(),
        max_seq_len=_MAX_SEQ_LEN,
        target_name="decoder_only",
    )


class _FakeRuntimeState:
    """Minimal RuntimeState stand-in: a real tiny decoder-only bundle on CPU.

    Only the methods ``run_general_val_bpb_stage`` calls for a decoder-only run
    are implemented. The base-Gemma / encdec / inactive-sampler paths are never
    reached (decoder-only scope), so they fail loud if a refactor starts using
    them.
    """

    def __init__(self, **_kwargs: Any) -> None:
        self._bundle = _decoder_only_bundle()

    def decoder_only(self) -> TargetModelBundle:
        return self._bundle

    def target_device(self) -> str:
        return "cpu"

    def offload_models(self) -> None:
        return None


def _install_general_val_stubs(monkeypatch: pytest.MonkeyPatch) -> None:
    """Stub ONLY the model-bound boundaries; keep the attacked-prefix path real.

    Real (un-stubbed): ``_distinct_combos``, ``_intervention_signature``,
    ``_build_trigger_index`` / ``_lookup_universal_trigger``, ``_samples_for_combo``
    -> ``_attack_doc`` (the per-k last-k substitution), the per-signature
    broadcast, ``_byte_weighted_bpb``, the table write, AND ``score_triple`` (the
    real tiny GPT forward). Stubbed: the FineWeb val texts, ``RuntimeState`` (real
    tiny bundle), the concept reader, the method lookup, the target-meta
    detection, and the steered-intervention builder (a no-op ``Intervention.none``
    so no steering-direction artifacts are needed -- the docs measure the TRIGGER's
    effect, which is the whole point).
    """

    monkeypatch.setattr(general_val, "iter_general_val_texts", lambda: iter(_GENERAL_VAL_TEXTS))
    monkeypatch.setattr(general_val, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(
        general_val,
        "_read_concepts",
        lambda _source_paths: {
            _CONCEPT_ID: Concept(
                concept_id=_CONCEPT_ID,
                feature_index=_CONCEPT_ID,
                layer=_DIFFMEAN_LAYER,
                sae_source="tiny-fixture-sae",
                description="dogs",
                corpus_count=0,
                description_source="fixture",
            )
        },
    )
    monkeypatch.setattr(general_val, "get_method_by_name", lambda _name: object())
    monkeypatch.setattr(
        general_val, "detect_target_model_metadata", lambda _model, _cfg: None
    )
    # The STEERED intervention is a no-op (Intervention.none): we are scoring the
    # tiny model's general_val BPB under the substituted TRIGGER, not a steering
    # direction (which would need diffmean artifacts the mini fixture lacks for
    # this method). The trigger substitution -- the thing under test -- is fully
    # exercised; the steered intervention is orthogonal and deferred to the
    # steered-direction unit tests.
    monkeypatch.setattr(
        general_val,
        "_build_combo_intervention",
        lambda **_kwargs: Intervention.none(),
    )


def test_universal_gcg_end_to_end_non_flat_general_val(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths, candidates, trigger_rows = _build_run(tmp_path / "out")

    # ----- Step 2 asserts: candidates + universal triggers shape -----
    candidate_rows = candidates.to_pylist()
    by_strength: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_strength.setdefault(int(row["strength_index"]), []).append(row)
    assert set(by_strength) == {0, 1, 2}
    # 2 rich x 1 avoid = 2 candidates per strength (no alpha fan-out).
    assert all(len(rows) == 2 for rows in by_strength.values())

    # k=0 control intact: attacked == source, no objective.
    for row in by_strength[0]:
        assert row["attacked_prompt_tokens"] == _PREFIX_TOKENS
        assert row["objective_value"] is None

    # k>0: the optimizer changed the prompt; the two pairs of a k datapoint share
    # ONE trigger (identical attacked tokens / replacement / objective) with
    # DISTINCT candidate_ids; length-preserving.
    for k in (1, 2):
        rows = by_strength[k]
        assert len(rows) == 2  # the 2 pairs of this k datapoint
        assert len({tuple(r["attacked_prompt_tokens"]) for r in rows}) == 1
        assert len({r["replacement_text"] for r in rows}) == 1
        assert len({r["objective_value"] for r in rows}) == 1
        assert len({r["candidate_id"] for r in rows}) == 2  # distinct ids
        for r in rows:
            assert r["attacked_prompt_tokens"] != _PREFIX_TOKENS
            assert len(r["attacked_prompt_tokens"]) == len(r["source_prompt_tokens"])
            assert int(r["editable_token_count"]) == k
            # The shared trigger == the single installed target token x k.
            assert r["replacement_tokens"] == [_TARGET_TOKEN] * k

    # ONE shared trigger per concept GROUP == one trigger row per k (2 non-control
    # k's), each carrying that group's trigger.
    assert len(trigger_rows) == 2
    trigger_by_k = {int(r["strength_index"]): r for r in trigger_rows}
    assert set(trigger_by_k) == {1, 2}
    for k, row in trigger_by_k.items():
        assert row["concept_id"] == _CONCEPT_ID
        assert row["n_pairs"] == 2
        assert row["trigger_tokens"] == [_TARGET_TOKEN] * k
    # DISTINCT trigger per k: the two trigger sequences (lengths 1 and 2) are unique.
    trigger_seqs = {tuple(r["trigger_tokens"]) for r in trigger_rows}
    assert len(trigger_seqs) == 2

    # ----- Step 3: drive the ATTACKED general_val on the real tiny model -----
    # General-val context extraction still needs feature statistics for bucket
    # boundaries; direction scaling itself reads only direction artifacts.
    stats_path = tmp_path / "stats.safetensors"
    write_steering_stats(stats_path, n_sae_features=16, max_mags={_CONCEPT_ID: 5.0})
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(_MINI_FIXTURE),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        feature_dir="/unused/features",
        feature_statistics_path=str(stats_path),
        target_models=["decoder_only"],
        device="cpu",
        device_batch_size=4,
        diffmean_and_probe_layer=_DIFFMEAN_LAYER,
        offload_between_stages=False,
        adversarial_methods=[_faster_gcg_cfg()],
        general_val_bpb=GeneralValBpbConfig(
            enabled=True,
            n_documents=len(_GENERAL_VAL_TEXTS),
            prefix_tokens=3,
            output_tokens=2,
        ),
    )

    _install_general_val_stubs(monkeypatch)

    general_val_table = run_general_val_bpb_stage(cfg=cfg, paths=paths)
    assert general_val_table.schema.equals(GENERAL_VAL_BPB_SCHEMA)
    # The on-disk artifact equals the returned table.
    assert pq.read_table(paths.general_val_bpb_path()).equals(general_val_table)

    gv_rows = general_val_table.to_pylist()
    # Every faster_gcg combo present in the candidates gets a general_val row
    # (broadcast per intervention signature): 3 k's -> 3 rows, all decoder-only /
    # this concept / this coefficient.
    assert len(gv_rows) == 3
    for row in gv_rows:
        assert row["target_model"] == "decoder_only"
        assert row["concept_id"] == _CONCEPT_ID
        assert row["adversarial_method"] == "faster_gcg"
        assert row["general_val_bpb"] is not None and row["general_val_bpb"] > 0.0

    # COMPARABILITY (issue #635): the corpus signature is doc-identity only, so
    # every row of this single-target run -- across every k, and (in a real run)
    # across every method/concept scored on this target -- shares ONE
    # general_val_corpus. This is the broadcast that lets the frontier compare
    # methods on the same validation corpus; a per-k / per-method corpus would
    # break it. n_documents/n_val_bytes are likewise uniform and agree with the
    # signature's nd/nb fields.
    corpora = {row["general_val_corpus"] for row in gv_rows}
    assert len(corpora) == 1, f"single-target run must share one corpus, got {corpora}"
    (corpus,) = corpora
    assert corpus and corpus.startswith("nd=")
    n_docs = {int(row["n_documents"]) for row in gv_rows}
    n_bytes = {int(row["n_val_bytes"]) for row in gv_rows}
    assert len(n_docs) == 1 and len(n_bytes) == 1
    assert corpus.startswith(f"nd={n_docs.pop()};nb={n_bytes.pop()};h=")

    # Index general_val_bpb by k. The faster_gcg adversarial_strength == k, so it
    # is the k axis.
    gv_by_k = {
        float(row["adversarial_strength"]): float(row["general_val_bpb"])
        for row in gv_rows
    }
    assert set(gv_by_k) == {0.0, 1.0, 2.0}

    # (A) NON-FLAT across the GCG axis: the k>0 ATTACKED values DIFFER from the
    # k=0 CLEAN value (the whole point of the universal-trigger attack). Pinned so
    # a flat broadcast (the old behaviour) fails loudly.
    clean = gv_by_k[0.0]
    for k in (1.0, 2.0):
        attacked = gv_by_k[k]
        assert attacked != pytest.approx(clean), (
            f"k={k}: attacked general_val_bpb {attacked} equals "
            f"the k=0 clean value {clean} -- general_val is FLAT (the attack "
            "did not transfer into the val docs)"
        )

    # (A') The k>0 stage value is EXACTLY the BPB of the trigger-substituted docs.
    # Independently reconstruct each k attacked-doc score from scratch (clean docs
    # -> _attack_doc last-k substitution with that group's trigger -> the SAME tiny
    # model under Intervention.none) and assert the stage produced that value. This
    # proves the per-k value came from the REAL attacked-prefix path (not the clean
    # path).
    independent = _independent_attacked_general_val(trigger_rows)
    for key, scored_value in gv_by_k.items():
        if key == 0.0:
            continue  # k=0 is the clean path, reconstructed below
        assert scored_value == pytest.approx(independent[key]), (
            f"stage general_val_bpb at k={key} ({scored_value}) does not "
            f"match an independent re-score of the trigger-substituted docs "
            f"({independent[key]}); the substitution path did not drive the value"
        )

    # (B) The independent clean re-score (no trigger) must match the stage's k=0
    # value -- non-vacuous teeth on the clean baseline, confirming the non-flatness
    # above is the TRIGGER's doing.
    clean_independent = _independent_clean_general_val()
    assert gv_by_k[0.0] == pytest.approx(clean_independent)

    # ----- Step 4: score -> aggregate -> join, asserting the per-k join -----
    scores = build_bpb_adv_score_table(
        candidates=candidates,
        context_segments_by_prefix_hash={},
        tokenizer=_TinyTokenizer(),
        token_bytes=t.ones(_VOCAB, dtype=t.int64),
        max_seq_len_by_target={"decoder_only": _MAX_SEQ_LEN},
        score_pair=_synthetic_score_pair,
    )
    scores = _with_fair_feature_diffmean_score_targets(scores)
    scores_path = tmp_path / "scores.parquet"
    pq.write_table(scores, scores_path)
    aggregates_dir = tmp_path / "aggregate"
    macro = aggregate_bpb_adv_scores(
        scores_path,
        aggregates_dir,
        general_val_path=paths.general_val_bpb_path(),
    )

    # The macro general_val_bpb must be joined PER k -- NOT broadcast flat. Read it
    # back keyed by (adversarial_strength == k) and compare to the scored
    # general_val table value for that exact k: a k-agnostic join would
    # cross-contaminate and these would mismatch.
    macro_rows = macro.to_pylist()
    macro_gv_by_k = {
        float(row["adversarial_strength"]): row["general_val_bpb"]
        for row in macro_rows
        if row["adversarial_method"] == "faster_gcg"
        and row["target_model"] == "decoder_only"
        and row["method_name"] == _METHOD
    }
    assert set(macro_gv_by_k) == set(gv_by_k)
    for key, scored_value in gv_by_k.items():
        assert macro_gv_by_k[key] is not None
        assert float(macro_gv_by_k[key]) == pytest.approx(scored_value), (
            f"macro general_val_bpb at k={key} is {macro_gv_by_k[key]} "
            f"but the scored general_val table has {scored_value}; the join did "
            "NOT land per-k"
        )

    # per_candidate carries the same per-k join (one concept here, so the macro
    # mean over concepts is the identity).
    per_candidate = pq.read_table(aggregates_dir / "per_candidate.parquet").to_pylist()
    pc_gv_by_k = {
        float(row["adversarial_strength"]): row["general_val_bpb"]
        for row in per_candidate
        if row["adversarial_method"] == "faster_gcg"
        and row["target_model"] == "decoder_only"
        and row["method_name"] == _METHOD
    }
    for key, scored_value in gv_by_k.items():
        assert pc_gv_by_k[key] is not None
        assert float(pc_gv_by_k[key]) == pytest.approx(scored_value)

    # Independent NON-FLAT teeth at the aggregate level: the macro general_val_bpb
    # still varies across the GCG axis after the join (the join did not flatten it).
    assert float(macro_gv_by_k[1.0]) != pytest.approx(float(macro_gv_by_k[0.0]))

    # The line plots build (general_val_bpb_vs_strength among them, since the
    # joined macro now carries non-null general_val_bpb across the GCG axis).
    plot_paths = plot_bpb_adv_summaries(
        macro, tmp_path / "plots", scatter_baseline_x="general_val_bpb"
    )
    assert plot_paths
    assert all(path.exists() for path in plot_paths)
    assert any(
        path.name == "general_val_bpb_vs_strength.png" for path in plot_paths
    ), "the attacked general_val should be non-null -> its vs-strength plot must build"


# --------------------------------------------------------------------------- #
# synthetic class-margin scorer for step 4 (model-free; the margin values are   #
# immaterial to the general_val JOIN claims -- only the join keys matter)        #
# --------------------------------------------------------------------------- #
def _synthetic_score_pair(
    candidate_row: Any, rich_input: Any, avoid_input: Any
) -> tuple[BpbScoreRow, BpbScoreRow]:
    # Distinct, finite per-side losses so the aggregation produces well-formed
    # macro rows; rich loss rises with k so the margin is non-degenerate. These
    # values do NOT touch general_val_bpb (joined separately); the test asserts
    # the JOIN keys, not these numbers.
    strength = float(candidate_row["adversarial_strength"])
    rich = BpbScoreRow(
        method_name=str(candidate_row["method_name"]),
        concept_id=int(candidate_row["concept_id"]),
        mode=str(candidate_row["mode"]),
        content_class=rich_input.content_class,
        target=str(candidate_row["target_model"]),
        coefficient_str=str(candidate_row["coefficient_str"]),
        sample_id=rich_input.sample_id,
        bpb_intervened=1.0 + strength,
        n_output_tokens=rich_input.n_output,
        n_output_bytes=2,
        truncated=False,
    )
    avoid = BpbScoreRow(
        method_name=str(candidate_row["method_name"]),
        concept_id=int(candidate_row["concept_id"]),
        mode=str(candidate_row["mode"]),
        content_class=avoid_input.content_class,
        target=str(candidate_row["target_model"]),
        coefficient_str=str(candidate_row["coefficient_str"]),
        sample_id=avoid_input.sample_id,
        bpb_intervened=3.0,
        n_output_tokens=avoid_input.n_output,
        n_output_bytes=4,
        truncated=False,
    )
    return rich, avoid


def _with_fair_feature_diffmean_score_targets(scores: pa.Table) -> pa.Table:
    rows = scores.to_pylist()
    existing = {
        (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        for row in rows
        if str(row["target_model"]) == "encdec"
        and str(row["method_name"])
        in {"feature_diffmean_encdec", "neg_feature_diffmean_encdec"}
        and bool(row["is_control"])
    }
    targets: list[dict[str, Any]] = []
    seen = set(existing)
    for row in rows:
        if not bool(row["is_control"]):
            continue
        key = (
            str(row["mode"]),
            str(row["adversarial_method"]),
            str(row["context_signal_mode"]),
        )
        if key in seen:
            continue
        seen.add(key)
        mode = str(row["mode"])
        method_name = (
            "neg_feature_diffmean_encdec"
            if mode == "negative"
            else "feature_diffmean_encdec"
        )
        targets.append(
            {
                **row,
                "candidate_id": f"fair-feature-diffmean-target-{mode}",
                "method_name": method_name,
                "target_model": "encdec",
                "coefficient_str": "None",
                "n_rich_output_tokens": 0,
                "n_avoid_output_tokens": 0,
                "n_rich_output_bytes": 0,
                "n_avoid_output_bytes": 0,
            }
        )
    return pa.Table.from_pylist([*rows, *targets], schema=SCORE_SCHEMA)


# --------------------------------------------------------------------------- #
# independent ground-truth re-scoring of the (attacked / clean) val docs        #
#                                                                             #
# These rebuild the EXACT production scoring path (build_general_val_docs ->    #
# _attack_doc last-k substitution -> _doc_to_scoring_input -> score_triple on   #
# the real tiny model under Intervention.none -> _byte_weighted_bpb) OUTSIDE    #
# run_general_val_bpb_stage, so the stage's per-k value can be checked          #
# against an independent re-score. A bug shared by both sides cannot hide: the  #
# stage routes through _samples_for_combo's signature/lookup machinery, while   #
# the reconstruction substitutes the trigger directly, so they only agree when  #
# the stage genuinely substituted that group's trigger into each doc's last-k.  #
# --------------------------------------------------------------------------- #
def _clean_general_val_docs() -> list[general_val._GeneralValDoc]:
    tokenizer = _TinyTokenizer()
    return general_val.build_general_val_docs(
        texts=iter(_GENERAL_VAL_TEXTS),
        tokenizer=tokenizer,
        token_bytes=t.ones(_VOCAB, dtype=t.int64),
        n_documents=len(_GENERAL_VAL_TEXTS),
        prefix_tokens=3,
        output_tokens=2,
    )


def _score_docs_bpb(docs: list[general_val._GeneralValDoc]) -> float:
    # Score the given docs with the real tiny model under a NO-OP intervention,
    # exactly as run_general_val_bpb_stage does for a decoder-only combo (whose
    # steered intervention the stub also makes a no-op), then corpus-BPB them.
    bundle = _decoder_only_bundle()
    samples = [
        general_val._doc_to_scoring_input(
            doc, segments=(), max_seq_len=bundle.max_seq_len
        )
        for doc in docs
    ]
    rows = score_triple(
        samples=samples,
        target_bundle=bundle,
        intervention=Intervention.none(),
        device=t.device("cpu"),
        coefficient_str=_COEFFICIENT_STR,
        method_name=_METHOD,
        device_batch_size=4,
    )
    bpb, _total_bytes = general_val._byte_weighted_bpb(rows)
    return bpb


def _independent_clean_general_val() -> float:
    return _score_docs_bpb(_clean_general_val_docs())


def _independent_attacked_general_val(
    trigger_rows: list[dict[str, Any]],
) -> dict[float, float]:
    """Ground-truth attacked BPB per k, reconstructed independently.

    For each non-control trigger row, substitute its trigger into every clean
    doc's last-k (the production ``_attack_doc``) and re-score on the tiny model.
    Keyed by (adversarial_strength == k) to match the stage's general_val rows.
    """

    tokenizer = _TinyTokenizer()
    clean_docs = _clean_general_val_docs()
    out: dict[float, float] = {}
    for row in trigger_rows:
        trigger = [int(token_id) for token_id in row["trigger_tokens"]]
        attacked_docs = [
            general_val._attack_doc(doc, trigger=trigger, tokenizer=tokenizer)
            for doc in clean_docs
        ]
        out[float(row["adversarial_strength"])] = _score_docs_bpb(attacked_docs)
    return out
