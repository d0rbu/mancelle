"""Realistic universal Faster-GCG end-to-end through the samples generator.

Unlike ``test_generate_samples_flow`` (which hand-crafts candidate rows), this test
runs the REAL universal Faster-GCG optimizer (``build_universal_gcg_candidates``)
over real attack rows so the candidates carry genuinely-optimized,
per-strength-distinct ``attacked_prompt_tokens`` and a real ``objective_value`` --
the regime the generator must surface and select on.

Under the UNIVERSAL optimizer a concept's prefixes share ONE trigger, optimised
over a minibatch of the concept's (rich, avoid) pairs. So a ``_multi_pair_prefix``
still puts MORE THAN ONE candidate on each ``(concept, prefix, strength)``
datapoint (one per rich x avoid pair, distinct ``candidate_id``s), but those
candidates now share the SAME substituted trigger ⇒ IDENTICAL
``attacked_prompt_tokens`` / ``replacement_text`` / ``objective_value`` (the
concept-level group-aggregated ``g``). The representative selection therefore
reduces to ``min(candidate_id)`` WITHIN a datapoint (every candidate gives the
same prompt/objective, so the choice is well-defined regardless of tie-break),
while the displayed prompt still VARIES across strengths ``k`` (a distinct trigger
per ``k``).

The optimizer is real; the per-row objective/gradient are synthetic surrogates
(no target model is loaded -- coherent full-size completions are validated
separately on d12_base). The generator then samples with the tiny real GPT.

The minimal per_candidate/macro aggregates are synthesized to carry the SAME
datapoint join keys as the real candidates (the generator only left-joins on those
keys; the aggregate values are immaterial to the issue-#543 claims). Assertions
read the real candidates back and compare, so they never depend on the optimizer's
exact token choices -- only on the generator faithfully reflecting them.

The model + the mini fixture are loaded LOCALLY here (the conftest caveat),
mirroring the sibling generator-flow tests.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import torch as t
import yaml

from nanochat.gpt import GPT, GPTConfig
from steering.eval.axbench_bpb.sample_loader import make_prefix_hash, make_sample_id
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import (
    BpbAttackPrefix,
    BpbAttackWorkSpec,
    core_prefix_id,
    prepare_bpb_attack_rows,
)
from steering.eval.axbench_bpb_adv.config import AxBenchBpbAdvConfig, FasterGcgConfig
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.sample_generation import (
    GenerationConfig,
    representative_attack_for_datapoint,
    run_generate_samples_core,
)
from steering.eval.axbench_bpb_adv.schemas import (
    MACRO_AGG_SCHEMA,
    PER_CANDIDATE_AGG_SCHEMA,
)
from tests.axbench_bpb_adv.functional._universal_gcg_equiv import as_batched_gradient

_FIXTURES = Path(__file__).resolve().parents[2] / "axbench_bpb" / "fixtures"
_TINY_CKPT = _FIXTURES / "tiny_decoder_only" / "ckpt.pt"
_MINI_FIXTURE = _FIXTURES / "mini_axbench_base_run"

_VOCAB = 128
_CONCEPT_ID = 1
_PREFIX_TEXT = "abc"
_PREFIX_TOKENS = [0, 97, 98, 99]
_COEFFICIENT_STR = "4.0"
_METHOD = "diffmean_decoder_only"
_DIFFMEAN_LAYER = 1  # valid block index for the 2-layer tiny model
# Allowed replacement tokens the optimizer may install into the editable span.
_TARGET_TOKENS = [120, 121]


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return _VOCAB

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


def _embedding_weight() -> t.Tensor:
    # Two-dim surrogate embedding: the target tokens are gradient-favorable so the
    # optimizer reliably installs them into the editable span (mirrors the proven
    # setup in tests/.../unit/test_optimize.py).
    weight = t.zeros(_VOCAB, 2, dtype=t.float32)
    for token_id in _TARGET_TOKENS:
        weight[token_id] = t.tensor([-1.0, 0.0])
    return weight


def _gradient_provider(
    row: dict[str, Any], tokens: Sequence[int], _step: int
) -> t.Tensor:
    # Push every editable position toward the target tokens.
    gradient = t.zeros(len(tokens), 2, dtype=t.float32)
    start = int(row["editable_token_start"])
    stop = start + int(row["editable_token_count"])
    for position in range(start, stop):
        gradient[position] = t.tensor([1.0, 0.0])
    return gradient


def _objective(row: dict[str, Any], tokens: Sequence[int]) -> float:
    # Pair-specific baseline (the rich output's first token differs per pair) MINUS
    # a reward per editable position set to an allowed target. The optimizer
    # minimizes -> installs targets (changing the prompt); the per-pair baseline
    # makes each pair's final objective_value distinct, so the representative
    # selection (min objective_value) is non-trivial.
    pair_base = float(int(row["concept_rich_output_tokens"][0]))
    start = int(row["editable_token_start"])
    edited = sum(1 for token in tokens[start:] if int(token) in _TARGET_TOKENS)
    return pair_base - 10.0 * edited


def _build_real_gcg_candidates(cfg: AxBenchBpbAdvConfig) -> pa.Table:
    attack_rows = prepare_bpb_attack_rows(
        prefixes=[
            _multi_pair_prefix(
                rich_outputs=[[100], [105]],  # 2 rich
                avoid_outputs=[[101]],  # x 1 avoid = 2 pairs/datapoint
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
    # The REAL production builder is now the UNIVERSAL optimizer: one shared trigger
    # per concept group, substituted into every member prefix's last-k span. The
    # second return value (the trigger artifact rows) is exercised separately; the
    # samples generator reads only the candidates table.
    candidates, _trigger_rows = build_universal_gcg_candidates(
        attack_rows=attack_rows,
        tokenizer=_TinyTokenizer(),
        cfg=faster_gcg,
        embedding_weight=_embedding_weight(),
        allowed_token_ids=_TARGET_TOKENS,
        objective=lambda _items: [_objective(_r, _t) for _r, _t in _items],
        gradient_provider_batch=as_batched_gradient(_gradient_provider),
    )
    return candidates


def _per_candidate_row(*, strength_index: int, is_control: bool) -> dict[str, Any]:
    return {
        "concept_id": _CONCEPT_ID,
        "mode": "positive",
        "method_name": _METHOD,
        "target_model": "decoder_only",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(strength_index),
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": "fixed_context",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0 - 0.1 * strength_index,
        "baseline_primary_margin": 0.9 if is_control else None,
        "general_val_bpb": None,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "n_rich_output_tokens": 1,
        "n_avoid_output_tokens": 1,
        "n_rich_output_bytes": 1,
        "n_avoid_output_bytes": 1,
        "is_control": is_control,
    }


def _macro_row(*, strength_index: int, is_control: bool) -> dict[str, Any]:
    return {
        "mode": "positive",
        "method_name": _METHOD,
        "target_model": "decoder_only",
        "coefficient_str": _COEFFICIENT_STR,
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(strength_index),
        "strength_unit": "optimized_tokens",
        "strength_index": strength_index,
        "context_signal_mode": "fixed_context",
        "mean_loss_concept_rich": 2.0,
        "mean_loss_concept_avoiding": 3.0,
        "primary_margin_y": 1.0 - 0.1 * strength_index,
        "baseline_primary_margin": 0.9 if is_control else None,
        "general_val_bpb": None,
        "n_concepts": 1,
        "n_samples": 1,
        "truncation_rate": 0.0,
        "is_control": is_control,
    }


def _feature_diffmean_macro_target() -> dict[str, Any]:
    row = _macro_row(strength_index=0, is_control=True)
    row["method_name"] = "feature_diffmean_encdec"
    row["target_model"] = "encdec"
    row["coefficient_str"] = "None"
    row["primary_margin_y"] = 0.9
    row["baseline_primary_margin"] = 0.9
    return row


def _write_real_gcg_run(output_root: Path) -> tuple[AxBenchBpbAdvPaths, pa.Table]:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(_MINI_FIXTURE),
        output_root=str(output_root),
        run_tag="run",
        feature_dir="/unused/features",
        feature_statistics_path="/unused/stats",
        target_models=["decoder_only"],
        pair_subsample=0,  # keep ALL rich x avoid pairs -> >1 candidate per datapoint
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0, 1, 2),
                max_steps=2,
                candidate_batch_size=4,
                top_k=2,
                distance_weight=0.0,
                temperature=1.0,
            )
        ],
    )
    candidates = _build_real_gcg_candidates(cfg)
    paths = AxBenchBpbAdvPaths(output_root=output_root, run_tag="run")
    paths.candidates_dir().mkdir(parents=True, exist_ok=True)
    paths.aggregates_dir().mkdir(parents=True, exist_ok=True)
    pq.write_table(candidates, paths.candidates_path())
    per_candidate = [
        _per_candidate_row(strength_index=k, is_control=(k == 0)) for k in (0, 1, 2)
    ]
    macro = [
        *[_macro_row(strength_index=k, is_control=(k == 0)) for k in (0, 1, 2)],
        _feature_diffmean_macro_target(),
    ]
    pq.write_table(
        pa.Table.from_pylist(per_candidate, schema=PER_CANDIDATE_AGG_SCHEMA),
        paths.aggregates_dir() / "per_candidate.parquet",
    )
    pq.write_table(
        pa.Table.from_pylist(macro, schema=MACRO_AGG_SCHEMA),
        paths.aggregates_dir() / "macro.parquet",
    )
    return paths, candidates


def test_generate_samples_real_tiny_gcg_end_to_end(tmp_path: Path) -> None:
    paths, candidates = _write_real_gcg_run(tmp_path / "out")
    candidate_rows = candidates.to_pylist()

    # Sanity on the REAL optimizer output before driving the generator.
    by_strength: dict[int, list[dict[str, Any]]] = {}
    for row in candidate_rows:
        by_strength.setdefault(int(row["strength_index"]), []).append(row)
    assert set(by_strength) == {0, 1, 2}
    # Multi-candidate datapoints: 2 rich x 1 avoid = 2 candidates per strength.
    assert all(len(rows) == 2 for rows in by_strength.values())
    # k=0 is the no-op control (attacked == source, no objective).
    for row in by_strength[0]:
        assert row["attacked_prompt_tokens"] == _PREFIX_TOKENS
        assert row["objective_value"] is None
    # k>=1: the optimizer ACTUALLY changed the prompt and recorded an objective.
    for k in (1, 2):
        for row in by_strength[k]:
            assert row["objective_value"] is not None
            assert row["attacked_prompt_tokens"] != _PREFIX_TOKENS
        # UNIVERSAL trigger: the two pairs of one datapoint share ONE trigger, so
        # their attacked prompt / replacement / objective are IDENTICAL while their
        # candidate_id (which folds in the rich/avoid output digests) stays DISTINCT.
        datapoint_cands = by_strength[k]
        assert (
            len({tuple(row["attacked_prompt_tokens"]) for row in datapoint_cands}) == 1
        )
        assert len({row["replacement_text"] for row in datapoint_cands}) == 1
        assert len({row["objective_value"] for row in datapoint_cands}) == 1
        assert len({row["candidate_id"] for row in datapoint_cands}) == len(
            datapoint_cands
        )

    core_prefix = str(candidate_rows[0]["core_prefix_id"])

    run_generate_samples_core(
        paths=paths,
        source_run_dir=_MINI_FIXTURE,
        model=_tiny_model(),
        tokenizer=_TinyTokenizer(),
        device=t.device("cpu"),
        target_model="decoder_only",
        target_model_meta=None,
        inactive_sampler=None,
        diffmean_layer=_DIFFMEAN_LAYER,
        counter_direction_family="diffmean",
        scatter_baseline_x="baseline_loss",
        steering_metric="bpb_class_margin",
        generation=GenerationConfig(
            n_concepts=5,
            n_prefixes_per_concept=1,
            n_completions=2,
            temperature=0.8,
            top_k=None,
            max_new_tokens=4,
            base_seed=0,
            methods=None,
        ),
        apply_max_act_scale=False,
    )

    out_path = (
        paths.samples_dir() / "positive" / f"{_METHOD}__faster_gcg.yaml"
    )
    assert out_path.exists()
    doc = yaml.safe_load(out_path.read_text())
    assert doc["adversarial_method"] == "faster_gcg"
    assert doc["strength_unit"] == "optimized_tokens"
    datapoints = doc["concepts"][0]["prefixes"][0]["datapoints"]
    assert [dp["strength_index"] for dp in datapoints] == [0, 1, 2]

    texts_by_strength = {dp["strength_index"]: dp["attacked_prompt_text"] for dp in datapoints}
    # Per-strength prompts are DISTINCT (real optimizer): k=0 source, k>=1 differ.
    assert texts_by_strength[1] != texts_by_strength[0]
    assert texts_by_strength[2] != texts_by_strength[1]

    # The generator surfaces EXACTLY the representative the selector computes from
    # the real candidates -- proving the YAML's prompt/replacement/objective match
    # the chosen candidate. Under universal triggers every candidate of a datapoint
    # shares one prompt/objective, so the representative is well-defined (the sort
    # reduces to the candidate_id tie-break, which is total and deterministic).
    for dp in datapoints:
        strength_index = int(dp["strength_index"])
        expected = representative_attack_for_datapoint(
            candidate_rows,
            concept_id=_CONCEPT_ID,
            core_prefix_id=core_prefix,
            method_name=_METHOD,
            adversarial_method="faster_gcg",
            coefficient_str=_COEFFICIENT_STR,
            strength_index=strength_index,
            mode="positive",
        )
        assert dp["attacked_prompt_text"] == expected.attacked_prompt_text
        assert dp["replacement_text"] == expected.replacement_text
        assert dp["objective_value"] == expected.objective_value
        assert len(dp["completions"]) == 2
        assert all(isinstance(c, str) for c in dp["completions"])
        # INDEPENDENT of the selector: the surfaced fields must come from the
        # min-objective / candidate_id-tie-break candidate, derived directly from
        # candidates.parquet rather than via representative_attack_for_datapoint
        # (so a bug shared by both sides cannot hide). With a shared trigger the
        # objectives tie, so candidate_id decides -- still a total, well-defined
        # choice -- and the prompt/replacement are identical across the datapoint's
        # candidates regardless.
        if strength_index >= 1:
            cands_here = by_strength[strength_index]
            chosen = min(
                cands_here, key=lambda r: (r["objective_value"], r["candidate_id"])
            )
            assert dp["objective_value"] == chosen["objective_value"]
            assert dp["replacement_text"] == chosen["replacement_text"]
            assert dp["attacked_prompt_text"] == chosen["attacked_prompt_text"]
