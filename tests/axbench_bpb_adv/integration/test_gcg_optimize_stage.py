"""Task-11 candidate-OPTIMIZE-stage integration for the Phase-2 feature-GCG methods.

The Task-10 review found a 6th reachable caller of the
``ModelBackedDualCompletionScorer`` still unwired: the candidate-**OPTIMIZE** stage
(``run_model_backed_bpb_candidate_stage``). A gcg steering method crossed with the
``faster_gcg`` prompt-token attack lands in the optimize stage's ``prompt_rows`` and
its per-pair steered intervention reaches ``build_intervention`` — which RAISES on
``adv_paths=None`` for the four ``ours_gcg_*`` methods. The human's decision: the
feature-GCG methods SHOULD be prompt-attacked (full adversarial integration), so we

  * Part A — thread ``adv_paths`` into the optimize-stage scorer (no crash), and
  * Part B — require each gcg ``prompt_rows`` entry to be covered by its method's
    SCORED prefix set (fair and roofline → held-out) BEFORE optimization,
    reusing the SAME scored-prefix keying as the Task-10 score-stage candidate guard.

This module pins:

  * the OPTIMIZE-stage objective seam builds the gcg intervention WITH ``adv_paths``
    and RAISES without it — the exact CRITICAL crash + its fix (Section A);
  * ``run_model_backed_bpb_candidate_stage`` completes on a gcg method + faster_gcg
    ``k>0`` rows, with the stage's scorer reaching ``build_intervention`` WITH
    ``adv_paths`` (no crash) — RED-proved against the ``adv_paths=None`` base
    behavior (Section B);
  * the optimize stage produces gcg prompt-attack candidates ONLY when every row
    is covered by the scored prefixes (a leaked non-scored prefix RED-proof),
    non-gcg rows pass through
    untouched, and the kept/dropped counts are logged (Section C).
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from safetensors.torch import save_file

import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
import steering.eval.axbench_bpb_adv.surrogate as surrogate
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.eval.axbench_base.types import SteeringMode
from steering.eval.axbench_bpb.config import AxBenchBpbConfig
from steering.eval.axbench_bpb.runtime_state import RuntimeState as BpbRuntimeState
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import (
    AxBenchBpbError,
    Intervention,
    InterventionKind,
)
from steering.eval.axbench_bpb_adv.artifact_identity import (
    ClusteredCandidateSnapshot,
    bind_clustered_table,
    resolve_clustered_source_identity,
)
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
)
from steering.eval.axbench_bpb_adv.editable_tokens import select_final_k_editable_span
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.evaluation_population import write_evaluation_population
from steering.eval.axbench_bpb_adv.gcg_family import GcgFamily
from steering.eval.axbench_bpb_adv.model_optimize import (
    BpbObjectiveCallbacks,
    run_model_backed_bpb_candidate_stage,
)
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.precomputed_bag_artifacts import (
    write_ours_gcg_scored_prefixes,
)
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA, CANDIDATES_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import ModelBackedDualCompletionScorer
from steering.eval.axbench_bpb_adv.source_validation import (
    CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
    BpbAdvSourceContract,
)
from steering.eval.axbench_bpb_adv.universal_gcg import concept_groups
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from tests.axbench_bpb_adv.artifact_helpers import artifact_identity_metadata
from tests.axbench_bpb_adv.full_sequence_reference import (
    _decoder_microbatch_bpb_reference,
)
from tests.axbench_bpb_adv.functional.test_surrogate_parity import _ToyDecoder
from tests.axbench_bpb_adv.integration.test_gcg_end_to_end import (
    _CONCEPT_ID,
    _make_concept,
)
from tests.axbench_bpb_adv.source_run_builders import (
    both_class_rows,
    one_sided_rows,
    write_stage_0_3_source_run,
)
from tests.axbench_bpb_adv.unit.test_feature_diffmean import (
    _linear_percentile_table,
    _real_cluster_contract,
)
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)

_POS = SteeringMode.POSITIVE
_FAIR_METHOD = "ours_gcg_encdec"
_NON_GCG_METHOD = "diffmean_encdec"  # not precomputed-bag-backed -> no bag guard
_TARGET_TOKEN = 120
_VOCAB_SIZE = 200


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures: a char tokenizer + ATTACK_ROWS_SCHEMA rows + a bag/scored-prefix writer
# ─────────────────────────────────────────────────────────────────────────────


class _CharTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: Any) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _attack_row(
    *,
    method_name: str,
    core_prefix_id: str,
    prompt_tokens: list[int],
    target_model: str = "decoder_only",
    concept_id: int = _CONCEPT_ID,
    mode: str = "positive",
    k: int = 2,
    coefficient_str: str = "0.0",
    rich_sample_id: str = "rich",
    avoid_sample_id: str = "avoid",
    rich_output: tuple[int, ...] = (100,),
    avoid_output: tuple[int, ...] = (150,),
) -> dict[str, Any]:
    """One ATTACK_ROWS_SCHEMA row (a faster_gcg ``k>0`` prompt-attack row).

    Mirrors the proven ``test_universal_gcg_flow._multi_pair_row`` shape but lets
    the caller choose ``method_name`` / ``core_prefix_id`` / ``target_model`` so a
    gcg vs non-gcg method and scored vs non-scored prefix can be exercised.
    """
    span = select_final_k_editable_span(prompt_tokens, k=k)
    count = span.count
    start = span.start if count > 0 else len(prompt_tokens)
    rich = [int(token_id) for token_id in rich_output]
    avoid = [int(token_id) for token_id in avoid_output]
    return {
        "attack_row_id": f"{method_name}|c{concept_id}|{core_prefix_id}|k{count}",
        "core_prefix_id": core_prefix_id,
        "concept_rich_sample_id": rich_sample_id,
        "concept_avoiding_sample_id": avoid_sample_id,
        "concept_id": int(concept_id),
        "mode": mode,
        "method_name": method_name,
        "target_model": target_model,
        "coefficient_str": coefficient_str,
        "adversarial_method": "faster_gcg",
        "adversarial_strength": float(count),
        "strength_unit": "optimized_tokens",
        "strength_index": int(count),
        "context_signal_mode": "not_applicable",
        "prefix_text": "".join(chr(int(tok)) for tok in prompt_tokens if int(tok) != 0),
        "prefix_hash": f"hash-{core_prefix_id}",
        "source_prompt_tokens": [int(tok) for tok in prompt_tokens],
        "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
        "concept_rich_output_tokens": rich,
        "concept_avoiding_output_tokens": avoid,
        "concept_rich_output_digest": token_sequence_digest(rich),
        "concept_avoiding_output_digest": token_sequence_digest(avoid),
        "editable_token_start": start,
        "editable_token_count": count,
        "original_editable_tokens": [int(tok) for tok in prompt_tokens[start : start + count]],
    }


def _write_attack_rows(
    paths: AxBenchBpbAdvPaths,
    rows: list[dict[str, Any]],
    *,
    evaluation_rows: list[dict[str, Any]] | None = None,
    cluster_contract: ClusterFeatureContract | None = None,
) -> None:
    paths.attack_rows_dir().mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows, schema=ATTACK_ROWS_SCHEMA)
    table = bind_clustered_table(
        table,
        paths=paths,
        contract=cluster_contract,
        artifact_family="attack_rows",
    )
    pq.write_table(
        table,
        paths.attack_rows_path(),
    )
    write_evaluation_population(
        paths.evaluation_population(),
        evaluation_rows if evaluation_rows is not None else rows,
        ours=AxBenchBpbAdvConfig().ours_gcg,
        feature_contract_fingerprint=(
            None
            if cluster_contract is None
            else cluster_contract.feature_contract_fingerprint
        ),
    )


def _write_bag(path: Path, bag: dict[int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fids = sorted(bag)
    save_file(
        {
            "feature_ids": t.tensor(fids, dtype=t.int64),
            "buckets": t.tensor([bag[fid] for fid in fids], dtype=t.int64),
        },
        str(path),
    )


def _cfg(tmp_path: Path, *, base_methods: list[str] | None = None) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats"),
        base_methods=base_methods,
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


def _paths(tmp_path: Path) -> AxBenchBpbAdvPaths:
    return AxBenchBpbAdvPaths(output_root=str(tmp_path / "out"), run_tag="run")


def _real_contract(*, alternate_mask: bool = False) -> ClusterFeatureContract:
    """Construct a validated clustered contract with real mask/consolidation state."""

    mask_ids = (0, 1, 3) if alternate_mask else (0, 1, 2)
    return _real_cluster_contract(
        percentile_table=_linear_percentile_table(4),
        mask=FeatureMask(FeatureMaskMode.whitelist, mask_ids),
    )


def _write_clustered_source_contract(
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
) -> None:
    paths.root.mkdir(parents=True, exist_ok=True)
    fingerprint = contract.feature_contract_fingerprint
    payload = BpbAdvSourceContract(
        seed_fingerprint="1" * 64,
        schema_version=CLUSTERED_SOURCE_CONTRACT_SCHEMA_VERSION,
        source_run_dir="/fixture/g6b-model-optimize",
        direction_training_layer=6,
        feature_contract_schema_version=contract.schema_version,
        feature_contract_fingerprint=fingerprint,
        candidate_provenance_sha256="2" * 64,
        stage2_feature_contract_fingerprints={
            "concept_features.safetensors": fingerprint,
            "concept_features_neg.safetensors": fingerprint,
        },
        baseline_training_corpus_sha256={
            "positive": "3" * 64,
            "negative": "4" * 64,
        },
    ).to_json_dict()
    paths.source_contract().write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _clustered_source_args(
    monkeypatch: pytest.MonkeyPatch,
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
) -> dict[str, Any]:
    """Install a focused paired-source boundary for stage integration tests."""

    validation = SimpleNamespace(
        concept_entries=lambda: [asdict(_make_concept())],
        concept_ids=frozenset({_CONCEPT_ID}),
        positive_rows=list,
        negative_rows=list,
    )
    stage_source = SimpleNamespace(
        validation=validation,
        source_identity=resolve_clustered_source_identity(
            paths=paths,
            contract=contract,
        ),
    )
    monkeypatch.setattr(
        model_optimize,
        "require_clustered_stage_source",
        lambda **_kwargs: stage_source,
    )
    return {"stage_source": stage_source}


def _table_payload_digest(path: Path) -> str:
    """Digest parsed parquet payload, excluding serializer-level nondeterminism."""

    payload = pq.read_table(path).to_pylist()
    rendered = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(rendered).hexdigest()


def _table_schema_digest(path: Path) -> str:
    """Digest the read-back Arrow schema, including table/field metadata."""

    schema_bytes = pq.read_table(path).schema.serialize().to_pybytes()
    return hashlib.sha256(schema_bytes).hexdigest()


def _candidate_survivors(paths: AxBenchBpbAdvPaths, method_name: str) -> set[str]:
    table = pq.read_table(paths.candidates_path())
    return {
        str(row["core_prefix_id"])
        for row in table.to_pylist()
        if str(row["method_name"]) == method_name
    }


# ─────────────────────────────────────────────────────────────────────────────
# Section A — the OPTIMIZE-stage objective seam (the exact crash + its fix)
# ─────────────────────────────────────────────────────────────────────────────


def _objective_with_scorer(
    monkeypatch: pytest.MonkeyPatch,
    *,
    adv_paths: AxBenchBpbAdvPaths | None,
) -> BpbObjectiveCallbacks:
    """A ``BpbObjectiveCallbacks`` whose ``pair_scorer`` carries (or omits) ``adv_paths``.

    The surrogate-embedding build (``__post_init__``) is the only model touch, so it
    is stubbed; the gcg ``_steered_intervention_for`` seam needs no model forward
    (it loads the precomputed bag from ``adv_paths``).
    """
    monkeypatch.setattr(
        model_optimize, "bpb_surrogate_embedding_weight", lambda _model: t.zeros(_VOCAB_SIZE, 2)
    )
    scorer = ModelBackedDualCompletionScorer(
        # A legacy (uncompacted, feature_vocab=None) model stand-in: the scorer's
        # checkpoint-V verify (issue #695 PR-1 follow-up) needs a real .model to
        # fingerprint, and every bag this helper's tests write is the legacy
        # (unstamped) shape via _write_bag, so a legacy model keeps them
        # byte-identical to pre-checkpoint-V-stamping behavior.
        target_bundles={
            "encdec": TargetModelBundle(
                model=SimpleNamespace(feature_vocab=None),
                tokenizer=SimpleNamespace(),
                max_seq_len=128,
                target_name="encdec",
            )
        },
        target_model_metas={"encdec": SimpleNamespace(negative_selection=None, num_buckets=4)},
        concepts_by_id={_CONCEPT_ID: _make_concept()},
        source_paths=SourcePaths(source_run_dir=Path("/unused")),
        tokenizer=SimpleNamespace(),
        device=t.device("cpu"),
        diffmean_layer=8,
        inactive_sampler=None,
        adv_paths=adv_paths,
        apply_max_act_scale=False,
    )
    return BpbObjectiveCallbacks(
        target_bundle=TargetModelBundle(
            model=object(),
            tokenizer=SimpleNamespace(),
            max_seq_len=128,
            target_name="encdec",
        ),
        target_model_meta=SimpleNamespace(),
        tokenizer=SimpleNamespace(),
        token_bytes=t.zeros(1),
        source_paths=SourcePaths(source_run_dir=Path("/unused")),
        pair_scorer=scorer,
        context_segments_by_prefix_hash={},
        device=t.device("cpu"),
    )


def test_optimize_objective_seam_builds_gcg_intervention_with_adv_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optimize stage's per-pair seam builds the gcg intervention WITH adv_paths.

    ``BpbObjectiveCallbacks._steered_intervention_for`` (model_optimize.py:481) is
    the exact crash site: it calls ``pair_scorer._base_steered_intervention`` →
    ``build_intervention`` for the gcg method. With ``adv_paths`` threaded and the
    bag present, it returns ``Intervention.ours(bag)`` — the no-crash half of the
    CRITICAL fix.
    """
    paths = _paths(tmp_path)
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {42: 0})
    objective = _objective_with_scorer(monkeypatch, adv_paths=paths)
    row = _attack_row(
        method_name=_FAIR_METHOD,
        core_prefix_id="held1",
        prompt_tokens=[0, 41, 42, 43, 44],
        target_model="encdec",
    )
    intervention = objective._steered_intervention_for(row)
    assert intervention.kind == InterventionKind.OURS
    assert intervention.feature_conditions == {42: 0}


def test_optimize_objective_seam_raises_without_adv_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """RED-proof for Part A: the un-threaded seam RAISES on a gcg method.

    This is the exact CRITICAL the task closes — a scorer built WITHOUT ``adv_paths``
    (the pre-fix optimize-stage construction) crashes when the per-pair intervention
    reaches ``build_intervention`` for a gcg method. Pins WHY Part A is load-bearing.
    """
    paths = _paths(tmp_path)
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {42: 0})
    objective = _objective_with_scorer(monkeypatch, adv_paths=None)
    row = _attack_row(
        method_name=_FAIR_METHOD,
        core_prefix_id="held1",
        prompt_tokens=[0, 41, 42, 43, 44],
        target_model="encdec",
    )
    with pytest.raises(AxBenchBpbError, match="adv_paths"):
        objective._steered_intervention_for(row)


# ─────────────────────────────────────────────────────────────────────────────
# Section B — run the REAL optimize stage: the scorer reaches build_intervention
#             WITH adv_paths (no crash)
# ─────────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("recompile_limit", "accumulated_limit", "expected"),
    [
        (8, 256, (256, 2_048)),
        (512, 4_096, (512, 4_096)),
    ],
)
def test_dynamo_limits_are_fixed_minima(
    monkeypatch: pytest.MonkeyPatch,
    recompile_limit: int,
    accumulated_limit: int,
    expected: tuple[int, int],
) -> None:
    cfg = t._dynamo.config
    monkeypatch.setattr(cfg, "recompile_limit", recompile_limit)
    monkeypatch.setattr(cfg, "accumulated_recompile_limit", accumulated_limit)
    model_optimize._raise_dynamo_recompile_limit()
    assert (cfg.recompile_limit, cfg.accumulated_recompile_limit) == expected


def test_clustered_candidate_stage_requires_source_before_any_stage_work(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(feature_contract_fingerprint="f" * 64),
    )
    monkeypatch.setattr(
        model_optimize,
        "_raise_dynamo_recompile_limit",
        lambda: pytest.fail(
            "stage work must not start before clustered source ownership"
        ),
    )

    with pytest.raises(
        AxBenchBpbAdvError,
        match="clustered candidate stage requires one resolved stage source",
    ):
        run_model_backed_bpb_candidate_stage(
            cfg=_cfg(tmp_path),
            paths=paths,
            tokenizer=_CharTokenizer(),
            cluster_contract=contract,
        )


class _FakeRuntimeState:
    """A model-free RuntimeState for the optimize stage's real (non-embedding) path."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def _bundle(self) -> Any:
        # feature_vocab=None (legacy/uncompacted) rather than a bare object(): the
        # scorer's checkpoint-V verify (issue #695 PR-1 follow-up) dereferences
        # model.feature_vocab to fingerprint the loaded checkpoint.
        return SimpleNamespace(
            model=SimpleNamespace(feature_vocab=None),
            tokenizer=_CharTokenizer(),
            max_seq_len=128,
        )

    def encdec(self) -> Any:
        return self._bundle()

    def decoder_only(self) -> Any:
        return self._bundle()

    def inactive_sampler(self) -> Any:
        return None

    def target_device(self) -> str:
        return "cpu"

    def loaded_target_metadata(self, _target_name: str) -> dict[str, Any]:
        return {}

    def offload_models(self) -> None:
        pass

    def release_base_gemma_and_sae(self) -> None:
        """#1245: the clustered path releases the extraction pair mid-stage.

        Counted rather than ignored so a test can assert the release actually
        happened; a silent ``pass`` would let the release be dropped entirely
        without any stage test noticing.
        """
        self.released_base_pair = getattr(self, "released_base_pair", 0) + 1


class _ModelFreeCheckpointStore:
    observed_runtime_hashes: frozenset[str] = frozenset()

    @classmethod
    def prepare(cls, **_kwargs: Any) -> _ModelFreeCheckpointStore:
        return cls()

    def load_triggers(self, *_args: Any, **_kwargs: Any) -> dict[str, dict[str, Any]]:
        return {}

    def commit_trigger(self, *_args: Any, **_kwargs: Any) -> bool:
        return True


@pytest.fixture(autouse=True)
def _model_free_checkpoint_plumbing(monkeypatch: pytest.MonkeyPatch) -> None:
    """This module tests model-backed orchestration with model-free target fakes.

    Checkpoint filesystem/identity behavior has its own focused tests; keep these
    older seams independent of real nanochat checkpoint and tokenizer trees.
    """

    monkeypatch.setattr(model_optimize, "resolve_and_pin_target_checkpoints", lambda *_a: {})
    monkeypatch.setattr(model_optimize, "assert_target_checkpoints_unchanged", lambda *_a: None)
    monkeypatch.setattr(model_optimize, "build_checkpoint_fingerprint", lambda **_kw: {})
    monkeypatch.setattr(
        model_optimize,
        "current_runtime_fingerprint",
        lambda _device: ({"device_type": "cpu"}, "0" * 64),
    )
    monkeypatch.setattr(
        model_optimize,
        "UniversalGcgCheckpointStore",
        _ModelFreeCheckpointStore,
    )
    monkeypatch.setattr(model_optimize, "warn_on_runtime_mismatch", lambda *_a: None)


def test_candidate_stage_gcg_method_completes_with_adv_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """``run_model_backed_bpb_candidate_stage`` completes on a gcg + faster_gcg k>0 run.

    Drives the REAL (non-embedding) optimize path with model-free fakes so the stage
    builds the REAL ``ModelBackedDualCompletionScorer`` (line 1349) and the REAL
    ``BpbObjectiveCallbacks``. A stub for ``build_universal_gcg_candidates`` captures
    the scorer's ``adv_paths`` AND drives the per-pair seam
    (``_steered_intervention_for``) over the gcg row — proving the stage's scorer
    reaches ``build_intervention`` WITH ``adv_paths`` (Intervention.ours), not None.

    RED-proof: reverting Part A (scorer built without ``adv_paths``) makes
    ``_steered_intervention_for`` RAISE the CRITICAL "adv_paths …" error THROUGH the
    stage here, and ``captured["adv_paths"] is paths`` fails — exactly the crash this
    task closes. The fair/roofline derive is stubbed (covered end to end by Task
    5/6/9); the scored-prefix + bag artifacts it would persist are pre-written so the
    optimize stage's scorer + coverage guard are exercised in isolation.
    """
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)  # base_methods=None -> all bag derives active; stubbed below.
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_FAIR_METHOD,
                core_prefix_id="held1",
                prompt_tokens=[0, 41, 42, 43, 44],
                target_model="encdec",
            )
        ],
    )
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {42: 0})

    captured: dict[str, Any] = {}

    def _stub_builder(**kwargs: Any) -> tuple[pa.Table, list[dict[str, Any]]]:
        callbacks = kwargs["objective"].__self__
        captured["adv_paths"] = callbacks.pair_scorer.adv_paths
        # Drive the exact per-pair seam the universal optimizer would: with adv_paths
        # threaded + the bag present this returns Intervention.ours; without it RAISES.
        captured["intervention"] = callbacks._steered_intervention_for(
            kwargs["attack_rows"].to_pylist()[0]
        )
        return pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA), []

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", lambda **_kw: None)
    monkeypatch.setattr(model_optimize, "_build_fair_gcg_bags", lambda **_kw: None)
    monkeypatch.setattr(model_optimize, "_build_roofline_gcg_bags", lambda **_kw: None)
    monkeypatch.setattr(model_optimize, "_build_fair_feature_diffmean_bags", lambda **_kw: None)
    monkeypatch.setattr(model_optimize, "_build_roofline_feature_diffmean_bags", lambda **_kw: None)
    monkeypatch.setattr(model_optimize, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(
        model_optimize, "detect_target_model_metadata", lambda _m, _c: SimpleNamespace()
    )
    monkeypatch.setattr(model_optimize, "_build_token_bytes", lambda _tok: t.zeros(1))
    monkeypatch.setattr(
        model_optimize, "_read_concepts", lambda _sp: {_CONCEPT_ID: _make_concept()}
    )
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", lambda _paths: {})
    monkeypatch.setattr(
        model_optimize,
        "bpb_surrogate_embedding_weight",
        lambda _model: t.zeros(_VOCAB_SIZE, 2),
    )
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tok: [1, 2, 3])
    monkeypatch.setattr(model_optimize, "build_universal_gcg_candidates", _stub_builder)

    written = run_model_backed_bpb_candidate_stage(cfg=cfg, paths=paths, tokenizer=_CharTokenizer())

    # The stage completed (no crash) and emitted the (empty, stubbed) candidate set.
    assert written == 0
    # (i) the stage's scorer was constructed WITH adv_paths (Part A wiring).
    assert captured["adv_paths"] is paths
    # (ii) the per-pair seam reached build_intervention WITH adv_paths → Intervention.ours.
    assert captured["intervention"].kind == InterventionKind.OURS
    assert captured["intervention"].feature_conditions == {42: 0}


# ─────────────────────────────────────────────────────────────────────────────
# Section C — the prefix coverage guard (test embedding path builds REAL candidates)
# ─────────────────────────────────────────────────────────────────────────────


def _run_embedding_stage(
    monkeypatch: pytest.MonkeyPatch, paths: AxBenchBpbAdvPaths, cfg: AxBenchBpbAdvConfig
) -> int:
    """Run the candidate stage through the model-free embedding test path.

    The four gcg methods are encdec in production; the guard keys ONLY on
    ``method_name`` / ``concept_id`` / ``mode`` / ``core_prefix_id`` (target-agnostic),
    so the proven ``decoder_only`` embedding-weight smoke path exercises it on CPU.
    A zero objective/gradient never accepts a step, so the per-pair candidate is the
    deterministic init trigger — enough to observe which covered prefixes are optimized.
    """
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tok: [_TARGET_TOKEN])
    return run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        embedding_weight=t.zeros(_VOCAB_SIZE, 1),  # D=1 matches zero_gradient_batch
        test_objective=zero_objective,
        test_gradient_provider_batch=zero_gradient_batch,
    )


def _two_gcg_prefix_rows(method_name: str) -> list[dict[str, Any]]:
    """Two same-concept gcg prefixes (one group, two pairs): ``held1`` and ``opt1``."""
    return [
        _attack_row(
            method_name=method_name,
            core_prefix_id="held1",
            prompt_tokens=[0, 41, 42, 43, 44],
            rich_sample_id="rich-h",
            avoid_sample_id="avoid-h",
            rich_output=(100,),
            avoid_output=(150,),
        ),
        _attack_row(
            method_name=method_name,
            core_prefix_id="opt1",
            prompt_tokens=[0, 51, 52, 53, 54, 55],
            rich_sample_id="rich-o",
            avoid_sample_id="avoid-o",
            rich_output=(101,),
            avoid_output=(151,),
        ),
    ]


def test_optimize_stage_attacks_only_scored_prefixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The optimize stage builds gcg prompt-attack candidates only for universal rows."""
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)
    rows = _two_gcg_prefix_rows(_FAIR_METHOD)
    _write_attack_rows(paths, [rows[0]])
    scored_path = paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair")

    write_ours_gcg_scored_prefixes(
        scored_path, ["held1"], concept_id=_CONCEPT_ID, mode=_POS, family="fair"
    )
    _run_embedding_stage(monkeypatch, paths, cfg)
    assert _candidate_survivors(paths, _FAIR_METHOD) == {"held1"}


def test_optimize_stage_raises_when_bag_artifact_misses_universal_prefix(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A stale family artifact cannot silently narrow bag-backed prompt rows."""
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)
    rows = _two_gcg_prefix_rows(_FAIR_METHOD)
    _write_attack_rows(paths, rows)
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )

    with pytest.raises(AxBenchBpbAdvError, match="precomputed-bag prompt coverage guard"):
        _run_embedding_stage(monkeypatch, paths, cfg)


def test_optimize_stage_non_gcg_rows_pass_through_without_artifact(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A non-gcg method's prompt_rows are unaffected by the bag guard.

    No scored-prefix artifact is written; if the guard incorrectly tried to read one
    for a non-gcg method it would RAISE (missing artifact). The stage completing with
    BOTH prefixes surviving proves the precomputed-bag guard early-returns
    (identity) for the common non-gcg path. The universal evaluation-population
    check is separately pinned below.
    """
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)
    _write_attack_rows(paths, _two_gcg_prefix_rows(_NON_GCG_METHOD))
    # Deliberately NO write_ours_gcg_scored_prefixes(...) here.
    _run_embedding_stage(monkeypatch, paths, cfg)
    assert _candidate_survivors(paths, _NON_GCG_METHOD) == {"held1", "opt1"}


def test_optimize_stage_raises_when_decoder_only_row_is_outside_universal_population(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The universal evaluation population is fail-loud for decoder-only rows too."""
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path, base_methods=["diffmean_decoder_only"])
    rows = [
        _attack_row(
            method_name="diffmean_decoder_only",
            target_model="decoder_only",
            core_prefix_id="held1",
            prompt_tokens=[0, 41, 42, 43, 44],
            rich_sample_id="rich-h",
            avoid_sample_id="avoid-h",
            rich_output=(100,),
            avoid_output=(150,),
        ),
        _attack_row(
            method_name="diffmean_decoder_only",
            target_model="decoder_only",
            core_prefix_id="opt1",
            prompt_tokens=[0, 51, 52, 53, 54, 55],
            rich_sample_id="rich-o",
            avoid_sample_id="avoid-o",
            rich_output=(101,),
            avoid_output=(151,),
        ),
    ]
    _write_attack_rows(paths, rows, evaluation_rows=[rows[0]])

    with pytest.raises(AxBenchBpbAdvError, match="absent from the exact held-out pair keys"):
        _run_embedding_stage(monkeypatch, paths, cfg)


def test_optimize_stage_logs_kept_and_dropped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """The per-method kept/dropped counts are logged (never a silent drop)."""
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)
    rows = _two_gcg_prefix_rows(_FAIR_METHOD)
    _write_attack_rows(paths, [rows[0]])
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    _run_embedding_stage(monkeypatch, paths, cfg)
    messages = " ".join(record.message for record in caplog.records)
    assert "precomputed-bag prompt coverage guard" in messages
    assert "kept 1 dropped 0" in messages


# ─────────────────────────────────────────────────────────────────────────────
# #14 — an active gcg method with zero class-margin prefixes RAISES (was warn+return)
# ─────────────────────────────────────────────────────────────────────────────


def test_model_backed_gcg_derive_no_prefixes_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A gcg method active for a mode but with zero class-margin prefixes RAISES.

    ``_build_model_backed_gcg_bags`` is reached only when the family's gcg method is
    active for >= 1 mode (the ``if not modes: return`` above handles the inactive
    case), so zero prefixes for the active modes is a real data/config error, not a
    routine skip. The check runs BEFORE the encdec/SAE load, so stubbing the sample
    loader + pairing to empty drives the raise without loading a model. Replaces
    the old warn + return that silently produced no bags.
    """
    monkeypatch.setattr(model_optimize, "load_raw_synth_rows", lambda *a, **k: [])
    monkeypatch.setattr(
        model_optimize,
        "_read_source_concept_ids",
        lambda _concepts_path: {1},
    )
    monkeypatch.setattr(model_optimize, "pair_samples_into_prefixes", lambda *a, **k: [])
    cfg = _cfg(tmp_path)
    paths = _paths(tmp_path)
    with pytest.raises(AxBenchBpbAdvError, match="no class-margin prefixes"):
        model_optimize._build_model_backed_gcg_bags(
            cfg=cfg,
            paths=paths,
            runtime_state=cast(
                model_optimize.RuntimeState,
                SimpleNamespace(cluster_contract=None),
            ),
            tokenizer=SimpleNamespace(),
            modes={SteeringMode.POSITIVE},
            family=GcgFamily.FAIR,
        )


def test_feature_diffmean_derive_does_not_route_through_gcg_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Feature-DiffMean uses the neutral precomputed-bag setup, not the GCG wrapper."""
    captured: dict[str, Any] = {}

    def _raise_if_called(**_kwargs: Any) -> None:
        raise AssertionError("_build_model_backed_gcg_bags must not be called")

    def _capture_precomputed(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(model_optimize, "_build_model_backed_gcg_bags", _raise_if_called)
    monkeypatch.setattr(
        model_optimize,
        "_build_model_backed_precomputed_bags",
        _capture_precomputed,
    )
    cfg = _cfg(tmp_path, base_methods=["feature_diffmean_encdec"])
    paths = _paths(tmp_path)

    model_optimize._build_fair_feature_diffmean_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=cast(model_optimize.RuntimeState, SimpleNamespace()),
        tokenizer=SimpleNamespace(),
    )

    assert captured["cfg"] is cfg
    assert captured["paths"] is paths
    assert captured["modes"] == {SteeringMode.POSITIVE}
    assert captured["derive"] is model_optimize.build_feature_diffmean_bags_fair
    assert captured["label"] == "feature-DiffMean"


def test_roofline_feature_diffmean_derive_does_not_route_through_gcg_builder(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Roofline Feature-DiffMean also uses the neutral setup, not the GCG wrapper."""
    captured: dict[str, Any] = {}

    def _raise_if_called(**_kwargs: Any) -> None:
        raise AssertionError("_build_model_backed_gcg_bags must not be called")

    def _capture_precomputed(**kwargs: Any) -> None:
        captured.update(kwargs)

    monkeypatch.setattr(model_optimize, "_build_model_backed_gcg_bags", _raise_if_called)
    monkeypatch.setattr(
        model_optimize,
        "_build_model_backed_precomputed_bags",
        _capture_precomputed,
    )
    cfg = _cfg(tmp_path, base_methods=["feature_diffmean_roofline_encdec"])
    paths = _paths(tmp_path)

    model_optimize._build_roofline_feature_diffmean_bags(
        cfg=cfg,
        paths=paths,
        runtime_state=cast(model_optimize.RuntimeState, SimpleNamespace()),
        tokenizer=SimpleNamespace(),
    )

    assert captured["cfg"] is cfg
    assert captured["paths"] is paths
    assert captured["modes"] == {SteeringMode.POSITIVE}
    assert captured["derive"] is model_optimize.build_feature_diffmean_bags_roofline
    assert captured["label"] == "roofline feature-DiffMean"


# ─────────────────────────────────────────────────────────────────────────────
# Section F — shared precomputed-bag derive setup (#7)
#
# All four bag families (fair/roofline feature-GCG + fair/roofline feature-
# DiffMean) share one source pairing, encdec/SAE, percentile table, and context
# cache. It is built ONCE over the union of active modes and reused; each family
# filters the shared prefixes down to its own modes. These tests pin every part
# of that contract: the setup builder, the per-family dispatcher (reuse / filter
# / label / no-op), union-then-filter equivalence, and the candidate-stage
# orchestration that must build the setup exactly once and hand it to every
# active family.
# ─────────────────────────────────────────────────────────────────────────────

_NEG = SteeringMode.NEGATIVE


def _fx_prefix(prefix_id: str, mode: SteeringMode) -> SimpleNamespace:
    """A stand-in prefix carrying only the ``mode`` the setup filter reads."""
    return SimpleNamespace(mode=mode, core_prefix_id=prefix_id)


class _SetupRuntime:
    """Model-free RuntimeState for _prepare_precomputed_bag_setup; counts loads."""

    def __init__(self) -> None:
        self.encdec_calls = 0
        self.sae_calls = 0
        self.offload_calls = 0
        self.events: list[str] = []
        self.cluster_contract: ClusterFeatureContract | None = None

    def offload_models(self) -> None:
        self.offload_calls += 1
        self.events.append("offload")

    def release_base_gemma_and_sae(self) -> None:
        """#1245: the clustered path drops the extraction pair mid-stage.

        Recorded in ``events`` (not a silent ``pass``) so its ORDER relative to
        the warm-start build and the derives is observable -- releasing before
        the warm-start cache exists would hand the derives a released model.
        """
        self.release_base_pair_calls = getattr(self, "release_base_pair_calls", 0) + 1
        self.events.append("release_base_pair")

    def encdec(self) -> Any:
        self.encdec_calls += 1
        self.events.append("encdec")
        return SimpleNamespace(
            model=SimpleNamespace(encoder=SimpleNamespace(config=SimpleNamespace(n_sae_features=8)))
        )

    def base_gemma_and_sae(self) -> tuple[str, str]:
        self.sae_calls += 1
        self.events.append("sae")
        return ("BASE_MODEL", "SAE")

    def target_device(self) -> str:
        return "cpu"


def _stub_prepare_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prefixes: list[SimpleNamespace],
) -> dict[str, Any]:
    """Stub every _prepare_precomputed_bag_setup dependency; return a call record.

    ``pair_samples_into_prefixes`` is stubbed to return *prefixes* verbatim so the
    mode filter inside the setup is exercised deterministically, independent of the
    real pairing. The record captures which synth parquet paths were read and how
    many times the context cache was loaded.
    """
    record: dict[str, Any] = {
        "load_paths": [],
        "positive_contract_flags": [],
        "context_loads": 0,
    }

    def _load(
        path: Any,
        *,
        filter_unaccepted: bool,
        require_positive_qc_contract: bool = False,
        expected_continuation_role: str,
    ) -> list[dict[str, Any]]:
        record["load_paths"].append(str(path))
        record["positive_contract_flags"].append(
            require_positive_qc_contract
        )
        return [{"concept_id": 1, "path": str(path)}]

    def _load_ctx(_paths: Any) -> dict[str, Any]:
        record["context_loads"] += 1
        return {"ctx": True}

    monkeypatch.setattr(model_optimize, "load_raw_synth_rows", _load)
    monkeypatch.setattr(
        model_optimize,
        "_read_source_concept_ids",
        lambda _concepts_path: {1},
    )
    monkeypatch.setattr(
        model_optimize,
        "bpb_attack_sample_from_synth_row",
        lambda row, *, mode, tokenizer: SimpleNamespace(row=row, mode=mode),
    )
    monkeypatch.setattr(
        model_optimize,
        "pair_samples_into_prefixes",
        lambda samples: list(prefixes),
    )
    monkeypatch.setattr(model_optimize, "assert_sae_matches_encdec", lambda _cfg, _model: None)
    monkeypatch.setattr(
        model_optimize, "build_percentile_table_from_stats", lambda _path, _n: "PCT_TABLE"
    )
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", _load_ctx)
    return record


# --- the setup builder in isolation ------------------------------------------


def test_prepare_setup_returns_shared_objects_over_union(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """_prepare_precomputed_bag_setup returns the union prefixes + shared objects."""
    prefixes = [_fx_prefix("pos1", _POS), _fx_prefix("neg1", _NEG)]
    _stub_prepare_deps(monkeypatch, prefixes=prefixes)
    runtime = _SetupRuntime()
    monkeypatch.setattr(
        model_optimize,
        "release_cuda_iteration_memory",
        lambda: runtime.events.append("release"),
    )

    setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, runtime),
        tokenizer=_CharTokenizer(),
        modes={_POS, _NEG},
    )

    assert [p.core_prefix_id for p in setup.prefixes] == ["pos1", "neg1"]
    assert setup.model.encoder.config.n_sae_features == 8
    assert setup.base_model == "BASE_MODEL"
    assert setup.sae == "SAE"
    assert setup.percentile_table == "PCT_TABLE"
    assert setup.context_segments_by_prefix_hash == {"ctx": True}
    assert str(setup.device) == "cpu"
    # D2 safety conjunct: every cached target is offloaded and allocator state
    # released before either large phase owner is restored.
    assert runtime.events == ["offload", "release", "encdec", "sae"]
    assert runtime.offload_calls == 1
    assert runtime.encdec_calls == 1
    assert runtime.sae_calls == 1


def test_prepare_setup_preserves_no_offload_configuration(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D2 U-conjunct: disabling offload preserves the legacy loader order."""

    _stub_prepare_deps(monkeypatch, prefixes=[_fx_prefix("pos1", _POS)])
    runtime = _SetupRuntime()
    monkeypatch.setattr(
        model_optimize,
        "release_cuda_iteration_memory",
        lambda: runtime.events.append("release"),
    )
    cfg = _cfg(tmp_path)
    cfg.offload_between_stages = False

    model_optimize._prepare_precomputed_bag_setup(
        cfg=cfg,
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, runtime),
        tokenizer=_CharTokenizer(),
        modes={_POS},
    )

    assert runtime.events == ["encdec", "sae"]
    assert runtime.offload_calls == 0


def test_prepare_setup_keeps_only_requested_mode_prefixes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The setup drops prefixes whose mode is not in the requested set."""
    prefixes = [_fx_prefix("pos1", _POS), _fx_prefix("neg1", _NEG), _fx_prefix("pos2", _POS)]
    _stub_prepare_deps(monkeypatch, prefixes=prefixes)

    setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS},
    )

    assert [p.core_prefix_id for p in setup.prefixes] == ["pos1", "pos2"]


def test_prepare_setup_filters_out_of_concepts_rows_before_pairing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Precomputed-bag setup must pair the same source population as validation.

    Validation and attack-row preparation intentionally ignore synth rows whose
    ``concept_id`` is not present in ``concepts.json``. If this setup pairs the
    broader raw parquet population, a valid source can fail here on an incomplete
    foreign-concept row after validation already succeeded.
    """
    source_root = tmp_path / "source"
    write_stage_0_3_source_run(
        source_root,
        concepts=[{"concept_id": _CONCEPT_ID, "description": "kept"}],
        positive_rows=both_class_rows("kept", concept_id=_CONCEPT_ID)
        + one_sided_rows(
            "foreign",
            content_class="concept_rich",
            concept_id=_CONCEPT_ID + 999,
        ),
        negative_rows=both_class_rows("negative", concept_id=_CONCEPT_ID),
    )
    monkeypatch.setattr(model_optimize, "assert_sae_matches_encdec", lambda _cfg, _model: None)
    monkeypatch.setattr(
        model_optimize,
        "build_percentile_table_from_stats",
        lambda _path, _n_sae: "PCT_TABLE",
    )
    monkeypatch.setattr(
        model_optimize,
        "load_context_segments_by_prefix_hash",
        lambda _paths_arg: {"ctx": True},
    )

    setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS},
    )

    assert len(setup.prefixes) == 1
    assert setup.prefixes[0].concept_id == _CONCEPT_ID
    assert setup.prefixes[0].prefix_text == "kept"


def test_prepare_setup_still_raises_for_in_concepts_incomplete_rows(tmp_path: Path) -> None:
    """The concept filter must not hide incomplete prefixes in the source contract."""
    source_root = tmp_path / "source"
    write_stage_0_3_source_run(
        source_root,
        concepts=[{"concept_id": _CONCEPT_ID, "description": "kept"}],
        positive_rows=both_class_rows("kept", concept_id=_CONCEPT_ID)
        + one_sided_rows(
            "bad",
            content_class="concept_rich",
            concept_id=_CONCEPT_ID,
        ),
        negative_rows=both_class_rows("negative", concept_id=_CONCEPT_ID),
    )

    class _BoomRuntime:
        cluster_contract = None

        def encdec(self) -> Any:
            raise AssertionError("model must not load before pairing raises")

        def base_gemma_and_sae(self) -> Any:
            raise AssertionError("SAE must not load before pairing raises")

        def target_device(self) -> str:
            return "cpu"

    with pytest.raises(AxBenchBpbAdvError, match="bad"):
        model_optimize._prepare_precomputed_bag_setup(
            cfg=_cfg(tmp_path),
            paths=_paths(tmp_path),
            runtime_state=cast(model_optimize.RuntimeState, _BoomRuntime()),
            tokenizer=_CharTokenizer(),
            modes={_POS},
        )


def test_prepare_setup_reads_only_active_mode_source_parquets(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """POSITIVE-only modes read only the positive parquet; both modes read both."""
    source_paths = SourcePaths(Path(_cfg(tmp_path).source_run_dir))

    record_pos = _stub_prepare_deps(monkeypatch, prefixes=[_fx_prefix("pos1", _POS)])
    model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS},
    )
    assert record_pos["load_paths"] == [str(source_paths.synth_training_positive())]
    assert record_pos["positive_contract_flags"] == [True]

    record_both = _stub_prepare_deps(
        monkeypatch, prefixes=[_fx_prefix("pos1", _POS), _fx_prefix("neg1", _NEG)]
    )
    model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS, _NEG},
    )
    assert record_both["load_paths"] == [
        str(source_paths.synth_training_positive()),
        str(source_paths.synth_training_negative()),
    ]
    assert record_both["positive_contract_flags"] == [True, False]


def test_clustered_prepare_setup_consumes_validation_rows_without_path_reads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prefix = _fx_prefix("validated-a", _POS)
    _stub_prepare_deps(monkeypatch, prefixes=[prefix])
    captured_samples: list[Any] = []
    validation = SimpleNamespace(
        concept_ids=frozenset({_CONCEPT_ID}),
        positive_rows=lambda: [
            {
                "concept_id": _CONCEPT_ID,
                "prefix_text": "validated-a",
                "source": "A",
            }
        ],
        negative_rows=lambda: pytest.fail(
            "positive-only setup must not request negative rows"
        ),
    )
    monkeypatch.setattr(
        model_optimize,
        "load_raw_synth_rows",
        lambda *_args, **_kwargs: pytest.fail(
            "clustered setup must not reopen synth paths"
        ),
    )
    monkeypatch.setattr(
        model_optimize,
        "_read_source_concept_ids",
        lambda *_args, **_kwargs: pytest.fail(
            "clustered setup must not reopen concepts.json"
        ),
    )

    def _pair(samples: list[Any]) -> list[SimpleNamespace]:
        captured_samples.extend(samples)
        return [prefix]

    monkeypatch.setattr(model_optimize, "pair_samples_into_prefixes", _pair)

    setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=_SetupRuntime(),  # ty: ignore[invalid-argument-type]
        tokenizer=_CharTokenizer(),
        modes={_POS},
        source_validation=validation,  # ty: ignore[invalid-argument-type]
    )

    assert [sample.row for sample in captured_samples] == [
        {
            "concept_id": _CONCEPT_ID,
            "prefix_text": "validated-a",
            "source": "A",
        }
    ]
    assert setup.prefixes == [prefix]


def test_prepare_setup_empty_prefixes_raises_before_touching_model(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Zero class-margin prefixes raise loudly BEFORE the encdec/SAE is loaded."""
    _stub_prepare_deps(monkeypatch, prefixes=[])

    class _BoomRuntime:
        cluster_contract = None

        def encdec(self) -> Any:
            raise AssertionError("model must not load when there are no prefixes")

        def base_gemma_and_sae(self) -> Any:
            raise AssertionError("SAE must not load when there are no prefixes")

        def target_device(self) -> str:
            return "cpu"

    with pytest.raises(AxBenchBpbAdvError, match="no class-margin prefixes"):
        model_optimize._prepare_precomputed_bag_setup(
            cfg=_cfg(tmp_path),
            paths=_paths(tmp_path),
            runtime_state=cast(model_optimize.RuntimeState, _BoomRuntime()),
            tokenizer=_CharTokenizer(),
            modes={_POS},
        )


# --- the per-family dispatcher ------------------------------------------------


def _shared_setup(prefixes: list[SimpleNamespace]) -> Any:
    return model_optimize._PrecomputedBagSetup(
        prefixes=prefixes,
        model="MODEL",
        base_model="BASE_MODEL",
        sae="SAE",
        percentile_table="PCT",
        context_segments_by_prefix_hash={"ctx": True},
        device="cpu",
    )


def test_build_with_shared_reuses_setup_and_never_reprepares(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Given a shared setup, the family derive reuses it and never re-prepares."""

    def _boom_prepare(**_kwargs: Any) -> None:
        raise AssertionError("_prepare_precomputed_bag_setup must not run when shared is passed")

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", _boom_prepare)
    captured: dict[str, Any] = {}
    shared = _shared_setup([_fx_prefix("pos1", _POS), _fx_prefix("neg1", _NEG)])

    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(
            model_optimize.RuntimeState,
            SimpleNamespace(cluster_contract=None),
        ),
        tokenizer=_CharTokenizer(),
        modes={_POS},
        derive=lambda **kwargs: captured.update(kwargs),
        label="fam",
        shared=shared,
    )

    assert [p.core_prefix_id for p in captured["prefixes"]] == ["pos1"]  # filtered to POS
    assert captured["model"] == "MODEL"
    assert captured["base_model"] == "BASE_MODEL"
    assert captured["sae"] == "SAE"
    assert captured["percentile_table"] == "PCT"
    assert captured["context_segments_by_prefix_hash"] == {"ctx": True}
    assert captured["device"] == "cpu"


def test_build_without_shared_prepares_setup_once(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With no shared setup, the dispatcher prepares one itself (backward compat)."""
    prepare_calls: list[set[SteeringMode]] = []

    def _prep(*, modes: set[SteeringMode], **_kwargs: Any) -> Any:
        prepare_calls.append(set(modes))
        return _shared_setup([_fx_prefix("pos1", _POS)])

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", _prep)
    captured: dict[str, Any] = {}

    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(
            model_optimize.RuntimeState,
            SimpleNamespace(cluster_contract=None),
        ),
        tokenizer=_CharTokenizer(),
        modes={_POS},
        derive=lambda **kwargs: captured.update(kwargs),
        label="fam",
    )

    assert prepare_calls == [{_POS}]
    assert [p.core_prefix_id for p in captured["prefixes"]] == ["pos1"]


def test_build_family_with_zero_prefixes_raises_with_label(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A family whose modes match none of the shared prefixes raises WITH its label."""
    shared = _shared_setup([_fx_prefix("neg1", _NEG)])  # union has only NEG

    def _derive(**_kwargs: Any) -> None:
        raise AssertionError("derive must not run with zero prefixes")

    with pytest.raises(
        AxBenchBpbAdvError, match=r"my-family-label derive: no class-margin prefixes"
    ):
        model_optimize._build_model_backed_precomputed_bags(
            cfg=_cfg(tmp_path),
            paths=_paths(tmp_path),
            runtime_state=cast(
                model_optimize.RuntimeState,
                SimpleNamespace(cluster_contract=None),
            ),
            tokenizer=_CharTokenizer(),
            modes={_POS},  # no POS prefix in the shared union
            derive=_derive,
            label="my-family-label",
            shared=shared,
        )


def test_build_empty_modes_is_noop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Inactive family (empty modes) derives nothing and never prepares a setup."""

    def _boom_prepare(**_kwargs: Any) -> None:
        raise AssertionError("_prepare_precomputed_bag_setup must not run for empty modes")

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", _boom_prepare)

    def _derive(**_kwargs: Any) -> None:
        raise AssertionError("derive must not run for empty modes")

    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, SimpleNamespace()),
        tokenizer=_CharTokenizer(),
        modes=set(),
        derive=_derive,
        label="fam",
        shared=None,
    )  # no raise, nothing derived


def test_union_then_filter_prefixes_match_per_family_alone(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Behaviour-preserving: filtering a union setup to a family's modes equals
    building that family's setup alone."""
    prefixes = [_fx_prefix("pos1", _POS), _fx_prefix("pos2", _POS), _fx_prefix("neg1", _NEG)]
    _stub_prepare_deps(monkeypatch, prefixes=prefixes)

    per_family: dict[str, Any] = {}
    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS},
        derive=lambda **kwargs: per_family.update(kwargs),
        label="fam",
    )

    union_setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS, _NEG},
    )
    shared_family: dict[str, Any] = {}
    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, _SetupRuntime()),
        tokenizer=_CharTokenizer(),
        modes={_POS},
        derive=lambda **kwargs: shared_family.update(kwargs),
        label="fam",
        shared=union_setup,
    )

    assert (
        [p.core_prefix_id for p in per_family["prefixes"]]
        == [p.core_prefix_id for p in shared_family["prefixes"]]
        == ["pos1", "pos2"]
    )


# --- candidate-stage orchestration: built ONCE, handed to every active family --


def _stub_candidate_stage_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Model-free stubs so run_model_backed_bpb_candidate_stage completes on CPU."""
    monkeypatch.setattr(model_optimize, "RuntimeState", _FakeRuntimeState)
    monkeypatch.setattr(
        model_optimize, "detect_target_model_metadata", lambda _m, _c: SimpleNamespace()
    )
    monkeypatch.setattr(model_optimize, "_build_token_bytes", lambda _tok: t.zeros(1))
    monkeypatch.setattr(
        model_optimize, "_read_concepts", lambda _sp: {_CONCEPT_ID: _make_concept()}
    )
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", lambda _paths: {})
    monkeypatch.setattr(
        model_optimize, "bpb_surrogate_embedding_weight", lambda _model: t.zeros(_VOCAB_SIZE, 2)
    )
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tok: [1, 2, 3])
    monkeypatch.setattr(
        model_optimize,
        "build_universal_gcg_candidates",
        lambda **_kw: (pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA), []),
    )


def test_candidate_stage_builds_shared_setup_once_for_all_families(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With every bag family active, the shared setup is built EXACTLY ONCE (over
    the union of active modes) and the same object is handed to every family.

    RED-proof: reverting #7 so each ``_build_*`` prepares its own setup would leave
    ``_prepare_precomputed_bag_setup`` uncalled by the stage (prepare_calls == [])
    and every family's ``shared`` kwarg None -- both assertions below fail.
    """
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)  # base_methods=None -> all four families active over {pos, neg}
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_FAIR_METHOD,
                core_prefix_id="held1",
                prompt_tokens=[0, 41, 42, 43, 44],
                target_model="encdec",
            )
        ],
    )
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {42: 0})

    prepare_calls: list[set[SteeringMode]] = []
    sentinel = object()

    def _prep(*, modes: set[SteeringMode], **_kwargs: Any) -> Any:
        prepare_calls.append(set(modes))
        return sentinel

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", _prep)

    shared_seen: dict[str, Any] = {}

    def _recorder(name: str) -> Any:
        def _f(**kwargs: Any) -> None:
            shared_seen[name] = kwargs.get("shared")

        return _f

    monkeypatch.setattr(model_optimize, "_build_fair_gcg_bags", _recorder("fair_gcg"))
    monkeypatch.setattr(model_optimize, "_build_roofline_gcg_bags", _recorder("roofline_gcg"))
    monkeypatch.setattr(
        model_optimize, "_build_fair_feature_diffmean_bags", _recorder("fair_fd")
    )
    monkeypatch.setattr(
        model_optimize, "_build_roofline_feature_diffmean_bags", _recorder("roofline_fd")
    )
    _stub_candidate_stage_env(monkeypatch)

    run_model_backed_bpb_candidate_stage(cfg=cfg, paths=paths, tokenizer=_CharTokenizer())

    # Built exactly once, over the union of every active family's modes.
    assert prepare_calls == [{_POS, _NEG}]
    # The single setup object reached every active family.
    assert shared_seen == {
        "fair_gcg": sentinel,
        "roofline_gcg": sentinel,
        "fair_fd": sentinel,
        "roofline_fd": sentinel,
    }


def test_candidate_stage_skips_setup_when_no_bag_method_active(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """With only a non-bag method active, the shared setup is never built."""
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path, base_methods=[_NON_GCG_METHOD])
    # k=0 control row: no prompt-optimization rows, so the run needs no stats /
    # steering-scale preflight -- keeps the test focused on the skip-setup guard.
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_NON_GCG_METHOD,
                core_prefix_id="p1",
                prompt_tokens=[0, 41, 42, 43, 44],
                target_model="encdec",
                k=0,
            )
        ],
    )

    def _boom_prepare(**_kwargs: Any) -> None:
        raise AssertionError("no bag family active -> setup must not be built")

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", _boom_prepare)
    _stub_candidate_stage_env(monkeypatch)

    run_model_backed_bpb_candidate_stage(cfg=cfg, paths=paths, tokenizer=_CharTokenizer())


def test_candidate_stage_embedding_path_skips_setup(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The model-free embedding (test-surrogate) path never builds the derive setup."""
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)  # base_methods=None -> would activate every family on the real path
    _write_attack_rows(
        paths, [_attack_row(method_name=_FAIR_METHOD, core_prefix_id="held1", prompt_tokens=[0, 41, 42, 43, 44])]
    )
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        ["held1"],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    _write_bag(paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"), {42: 0})

    def _boom_prepare(**_kwargs: Any) -> None:
        raise AssertionError("embedding path must not build the derive setup")

    monkeypatch.setattr(model_optimize, "_prepare_precomputed_bag_setup", _boom_prepare)
    monkeypatch.setattr(model_optimize, "_read_concepts", lambda _sp: {_CONCEPT_ID: _make_concept()})

    _run_embedding_stage(monkeypatch, paths, cfg)


def test_candidate_stage_embedding_path_never_prepares_checkpoint_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path, base_methods=[_NON_GCG_METHOD])
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_NON_GCG_METHOD,
                core_prefix_id="held1",
                prompt_tokens=[0, 41, 42, 43, 44],
            )
        ],
    )

    class _ForbiddenStore:
        @classmethod
        def prepare(cls, **_kwargs: Any) -> Any:
            pytest.fail("the embedding_weight test path created checkpoint state")

    monkeypatch.setattr(model_optimize, "UniversalGcgCheckpointStore", _ForbiddenStore)
    monkeypatch.setattr(
        model_optimize,
        "resolve_and_pin_target_checkpoints",
        lambda *_args: pytest.fail("the embedding path resolved production checkpoints"),
    )
    monkeypatch.setattr(
        model_optimize,
        "_read_concepts",
        lambda _source_paths: {_CONCEPT_ID: _make_concept()},
    )
    _run_embedding_stage(monkeypatch, paths, cfg)
    assert not paths.universal_gcg_checkpoint_dir().exists()


def test_candidate_stage_checkpoint_orchestration_filters_targets_and_commits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cfg = _cfg(
        tmp_path,
        base_methods=["prompting_decoder_only", "prompting_encdec"],
    )
    rows = [
        _attack_row(
            method_name="prompting_decoder_only",
            core_prefix_id="decoder",
            prompt_tokens=[0, 41, 42, 43],
            target_model="decoder_only",
        ),
        _attack_row(
            method_name="prompting_encdec",
            core_prefix_id="encdec",
            prompt_tokens=[0, 51, 52, 53],
            target_model="encdec",
        ),
    ]
    _write_attack_rows(paths, rows)
    groups = concept_groups(rows)
    restored: dict[str, dict[str, Any]] = {
        str(group.key_str): {"target_model": str(group.rows[0]["target_model"])}
        for group in groups
    }
    trace: list[str] = []
    prepared: list[dict[str, Any]] = []
    commits: list[tuple[str, Mapping[str, Any], str]] = []
    builds: list[dict[str, Any]] = []

    class _Runtime(_FakeRuntimeState):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            trace.append("runtime")
            super().__init__(*args, **kwargs)

        def loaded_target_metadata(self, _target_name: str) -> dict[str, Any]:
            trace.append(f"metadata:{_target_name}")
            return {"target": _target_name}

    class _CaptureStore:
        observed_runtime_hashes = frozenset({"9" * 64})

        @classmethod
        def prepare(cls, **kwargs: Any) -> Any:
            trace.append("prepare")
            prepared.append(dict(kwargs))
            return cls()

        def load_triggers(self) -> dict[str, dict[str, Any]]:
            trace.append("load")
            return deepcopy(restored)

        def commit_trigger(
            self,
            key: str,
            row: Mapping[str, Any],
            runtime_hash: str,
        ) -> bool:
            commits.append((key, dict(row), runtime_hash))
            return True

    def _builder(**kwargs: Any) -> tuple[pa.Table, list[dict[str, Any]]]:
        target = str(kwargs["attack_rows"].to_pylist()[0]["target_model"])
        builds.append(dict(kwargs))
        expected_keys = {
            group.key_str
            for group in groups
            if str(group.rows[0]["target_model"]) == target
        }
        assert set(kwargs["restored_triggers"]) == expected_keys
        assert kwargs["pool_device"] == "cpu"
        callback = kwargs["on_trigger_complete"]
        assert callback is not None
        callback(next(iter(expected_keys)), {"target_model": target})
        return pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA), []

    monkeypatch.setattr(model_optimize, "RuntimeState", _Runtime)
    monkeypatch.setattr(
        model_optimize,
        "resolve_and_pin_target_checkpoints",
        lambda bpb_cfg, targets: trace.append("resolve")
        or {target: SimpleNamespace(target_name=target) for target in targets},
    )
    monkeypatch.setattr(
        model_optimize,
        "assert_target_checkpoints_unchanged",
        lambda _resolved: trace.append("rehash"),
    )
    monkeypatch.setattr(
        model_optimize,
        "build_checkpoint_fingerprint",
        lambda **_kwargs: trace.append("fingerprint") or {"semantic": "v1"},
    )
    monkeypatch.setattr(
        model_optimize,
        "current_runtime_fingerprint",
        lambda _device: trace.append("runtime_fingerprint")
        or ({"device_type": "cpu"}, "8" * 64),
    )
    monkeypatch.setattr(model_optimize, "UniversalGcgCheckpointStore", _CaptureStore)
    monkeypatch.setattr(
        model_optimize,
        "warn_on_runtime_mismatch",
        lambda _store, _runtime: trace.append("warn"),
    )
    monkeypatch.setattr(
        model_optimize, "detect_target_model_metadata", lambda _model, _cfg: SimpleNamespace()
    )
    monkeypatch.setattr(model_optimize, "_build_token_bytes", lambda _tok: t.zeros(1))
    monkeypatch.setattr(
        model_optimize,
        "_read_concepts",
        lambda _source_paths: {_CONCEPT_ID: _make_concept()},
    )
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", lambda _p: {})
    monkeypatch.setattr(
        model_optimize,
        "bpb_surrogate_embedding_weight",
        lambda _model: t.zeros(_VOCAB_SIZE, 2),
    )
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tok: [1, 2, 3])
    monkeypatch.setattr(model_optimize, "build_universal_gcg_candidates", _builder)

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
    )

    assert len(prepared) == 1
    assert prepared[0]["checkpoint_dir"] == paths.universal_gcg_checkpoint_dir()
    assert [group.key_str for group in prepared[0]["expected_groups"]] == [
        group.key_str for group in groups
    ]
    assert prepared[0]["max_steps"] == 1
    assert len(builds) == 2
    assert [build["attack_rows"].to_pylist()[0]["target_model"] for build in builds] == [
        "decoder_only",
        "encdec",
    ]
    assert {key for key, _row, _runtime in commits} == set(restored)
    assert {runtime for _key, _row, runtime in commits} == {"8" * 64}
    assert trace.index("resolve") < trace.index("runtime")
    assert trace.index("rehash") < trace.index("fingerprint") < trace.index("prepare")
    assert trace.index("prepare") < trace.index("load") < trace.index("warn")


def test_checkpoint_prepare_failure_precedes_any_faster_gcg_builder_call(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path, base_methods=["prompting_decoder_only"])
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name="prompting_decoder_only",
                core_prefix_id="p1",
                prompt_tokens=[0, 41, 42, 43],
            )
        ],
    )
    _stub_candidate_stage_env(monkeypatch)

    class _MismatchedStore:
        @classmethod
        def prepare(cls, **_kwargs: Any) -> Any:
            raise AxBenchBpbAdvError("manifest semantic mismatch")

    monkeypatch.setattr(model_optimize, "UniversalGcgCheckpointStore", _MismatchedStore)
    monkeypatch.setattr(
        model_optimize,
        "build_universal_gcg_candidates",
        lambda **_kwargs: pytest.fail("GCG builder ran after manifest mismatch"),
    )
    with pytest.raises(AxBenchBpbAdvError, match="manifest semantic mismatch"):
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=_CharTokenizer(),
        )


# ─────────────────────────────────────────────────────────────────────────────
# G6b safety argument — one independently defeatable criterion per conjunct
# ─────────────────────────────────────────────────────────────────────────────


def _write_clustered_bag_dependencies(
    paths: AxBenchBpbAdvPaths,
    *,
    prefix_id: str = "held1",
) -> None:
    write_ours_gcg_scored_prefixes(
        paths.ours_gcg_scored_prefixes(_CONCEPT_ID, _POS, "fair"),
        [prefix_id],
        concept_id=_CONCEPT_ID,
        mode=_POS,
        family="fair",
    )
    _write_bag(
        paths.ours_gcg_bag(_CONCEPT_ID, _POS, 0.0, "fair"),
        {0: 0},
    )


def _clustered_k0_fixture(
    tmp_path: Path,
    *,
    contract: ClusterFeatureContract,
) -> tuple[AxBenchBpbAdvConfig, AxBenchBpbAdvPaths]:
    paths = _paths(tmp_path)
    _write_clustered_source_contract(paths, contract)
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_FAIR_METHOD,
                core_prefix_id="held1",
                prompt_tokens=[0, 41, 42],
                target_model="encdec",
                k=0,
            )
        ],
        cluster_contract=contract,
    )
    _write_clustered_bag_dependencies(paths)
    return _cfg(tmp_path), paths


def _install_clustered_stage_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    shared: model_optimize._PrecomputedBagSetup,
    cache: object,
    family_seen: dict[str, Any],
    cache_calls: list[dict[str, Any]],
    runtime_kwargs: list[dict[str, Any]],
) -> None:
    class _Runtime(_FakeRuntimeState):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            runtime_kwargs.append(dict(kwargs))
            self.cluster_contract = kwargs.get("cluster_contract")
            del args

    monkeypatch.setattr(model_optimize, "RuntimeState", _Runtime)
    monkeypatch.setattr(
        model_optimize,
        "_prepare_precomputed_bag_setup",
        lambda **_kwargs: shared,
    )

    def _build_cache(**kwargs: Any) -> object:
        cache_calls.append(dict(kwargs))
        return cache

    monkeypatch.setattr(
        model_optimize,
        "build_clustered_warm_start_cache",
        _build_cache,
    )

    def _derive(name: str) -> Any:
        def _record(**kwargs: Any) -> None:
            family_seen[name] = kwargs["clustered_warm_starts"]

        return _record

    monkeypatch.setitem(
        model_optimize._GCG_DERIVE_BY_FAMILY,
        GcgFamily.FAIR,
        _derive("fair_gcg"),
    )
    monkeypatch.setitem(
        model_optimize._GCG_DERIVE_BY_FAMILY,
        GcgFamily.ROOFLINE,
        _derive("roofline_gcg"),
    )
    monkeypatch.setattr(
        model_optimize,
        "build_feature_diffmean_bags_fair",
        _derive("fair_diffmean"),
    )
    monkeypatch.setattr(
        model_optimize,
        "build_feature_diffmean_bags_roofline",
        _derive("roofline_diffmean"),
    )


def test_unclustered_stage_matches_main_payload_digests_and_legacy_writers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T1.1/T1.3: legacy payloads stay fixed and only legacy writers run."""

    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path, base_methods=[_NON_GCG_METHOD])
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_NON_GCG_METHOD,
                core_prefix_id="legacy",
                prompt_tokens=[0, 41, 42],
                target_model="encdec",
                k=0,
            )
        ],
    )
    legacy_writer = model_optimize.atomic_write_parquet
    writes: list[str] = []

    def _legacy_write(table: pa.Table, path: Path) -> None:
        writes.append(path.name)
        legacy_writer(table, path)

    monkeypatch.setattr(model_optimize, "atomic_write_parquet", _legacy_write)
    monkeypatch.setattr(
        model_optimize,
        "write_clustered_table_atomic",
        lambda *_args, **_kwargs: pytest.fail(
            "unclustered stage reached clustered writer"
        ),
        raising=False,
    )

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
    )

    assert writes == ["candidates.parquet", "universal_triggers.parquet"]
    assert (
        _table_schema_digest(paths.candidates_path()),
        _table_schema_digest(paths.universal_triggers_path()),
    ) == (
        "ae70a6072b9f55ef92ae74509c7ac23e01cf3f55f77f14dc06cfc93897576ecb",
        "fab2652ac74133a916a86c37baffcf59c582aea011ab710d0c7d606951bb5fb5",
    )
    assert (
        _table_payload_digest(paths.candidates_path()),
        _table_payload_digest(paths.universal_triggers_path()),
    ) == (
        "220a756335faaeb06f9a9413675f022313a2aed75d14645cf19d63bc289ed20a",
        "4f53cda18c2baa0c0354bb5f9a3ecbe5ed12ab4d8e11ba873c2f11161202b945",
    )


def test_unclustered_stage_omits_every_threaded_cluster_kwarg(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T1.2/D1/D5: legacy callers receive absent kwargs, never explicit None."""

    paths = _paths(tmp_path)
    cfg = _cfg(
        tmp_path,
        base_methods=["prompting_decoder_only", "prompting_encdec"],
    )
    cfg.adversarial_methods.append(
        AdditiveDirectionConfig(counter_direction_family="linear_probe")
    )
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name="prompting_decoder_only",
                core_prefix_id="decoder",
                prompt_tokens=[0, 41, 42],
                target_model="decoder_only",
            ),
            _attack_row(
                method_name="prompting_encdec",
                core_prefix_id="encdec",
                prompt_tokens=[0, 51, 52],
                target_model="encdec",
            ),
        ],
    )
    observed: dict[str, Any] = {
        "runtime": [],
        "population": [],
        "coverage": [],
        "detect": [],
        "context": [],
        "scorer": [],
    }

    class _Runtime(_FakeRuntimeState):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            observed["runtime"].append(dict(kwargs))
            del args

    original_population = model_optimize.read_evaluation_population
    original_coverage = (
        model_optimize.require_precomputed_bag_prompt_rows_cover_scored_prefixes
    )

    def _population(path: Path, **kwargs: Any) -> Any:
        observed["population"].append(dict(kwargs))
        return original_population(path, **kwargs)

    def _coverage(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
        observed["coverage"].append(dict(kwargs))
        return original_coverage(rows, **kwargs)

    def _detect(_model: Any, _cfg_arg: Any, **kwargs: Any) -> Any:
        observed["detect"].append(dict(kwargs))
        return SimpleNamespace()

    def _context(_paths_arg: Any, **kwargs: Any) -> dict[str, Any]:
        observed["context"].append(dict(kwargs))
        return {}

    def _scorer(**kwargs: Any) -> Any:
        observed["scorer"].append(dict(kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(model_optimize, "RuntimeState", _Runtime)
    monkeypatch.setattr(model_optimize, "read_evaluation_population", _population)
    monkeypatch.setattr(
        model_optimize,
        "require_precomputed_bag_prompt_rows_cover_scored_prefixes",
        _coverage,
    )
    monkeypatch.setattr(model_optimize, "detect_target_model_metadata", _detect)
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", _context)
    monkeypatch.setattr(model_optimize, "ModelBackedDualCompletionScorer", _scorer)
    monkeypatch.setattr(model_optimize, "_build_token_bytes", lambda _tok: t.ones(256))
    monkeypatch.setattr(
        model_optimize,
        "_read_concepts",
        lambda _paths_arg: {_CONCEPT_ID: _make_concept()},
    )
    monkeypatch.setattr(
        model_optimize,
        "bpb_surrogate_embedding_weight",
        lambda _model: t.zeros(_VOCAB_SIZE, 1),
    )
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tok: [1])
    monkeypatch.setattr(
        model_optimize,
        "build_universal_gcg_candidates",
        lambda **_kwargs: (pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA), []),
    )

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
    )

    assert observed["runtime"] == [
        {"cfg": observed["runtime"][0]["cfg"], "ddp_rank": 0, "ddp_world_size": 1}
    ]
    assert observed["population"] == [{}]
    assert observed["detect"] == [{}, {}]
    assert observed["context"] == [{}]
    assert len(observed["coverage"]) == 1
    assert "feature_contract_fingerprint" not in observed["coverage"][0]
    assert "cluster_contract" not in observed["scorer"][0]
    assert observed["scorer"][0]["counter_direction_family"] == "linear_probe"


def test_unclustered_precomputed_setup_and_derive_omit_cluster_kwargs(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T1.2/D1: setup guards and derives keep their legacy call signatures."""

    _stub_prepare_deps(monkeypatch, prefixes=[_fx_prefix("pos", _POS)])
    runtime = _SetupRuntime()
    seen: dict[str, list[dict[str, Any]]] = {
        "sae": [],
        "context": [],
        "derive": [],
    }

    def _sae(_cfg_arg: Any, _model: Any, **kwargs: Any) -> None:
        seen["sae"].append(dict(kwargs))

    def _context(_paths_arg: Any, **kwargs: Any) -> dict[str, Any]:
        seen["context"].append(dict(kwargs))
        return {}

    def _derive(**kwargs: Any) -> None:
        seen["derive"].append(dict(kwargs))

    monkeypatch.setattr(model_optimize, "assert_sae_matches_encdec", _sae)
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", _context)
    setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, runtime),
        tokenizer=_CharTokenizer(),
        modes={_POS},
    )
    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, runtime),
        tokenizer=_CharTokenizer(),
        modes={_POS},
        derive=_derive,
        label="legacy",
        shared=setup,
    )

    assert seen["sae"] == [{}]
    assert seen["context"] == [{}]
    assert "cluster_contract" not in seen["derive"][0]
    assert "clustered_warm_starts" not in seen["derive"][0]


def test_clustered_precomputed_setup_and_derive_receive_exact_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D1 clustered conjunct: setup guards and derive receive the same contract."""

    _stub_prepare_deps(monkeypatch, prefixes=[_fx_prefix("pos", _POS)])
    contract = _real_contract()
    runtime = _SetupRuntime()
    runtime.cluster_contract = contract
    seen: dict[str, Any] = {}

    def _sae(
        _cfg_arg: Any,
        _model: Any,
        *,
        cluster_contract: ClusterFeatureContract,
    ) -> None:
        seen["sae"] = cluster_contract

    def _context(
        _paths_arg: Any,
        *,
        cluster_contract: ClusterFeatureContract,
    ) -> dict[str, Any]:
        seen["context"] = cluster_contract
        return {}

    def _derive(
        *,
        cluster_contract: ClusterFeatureContract,
        **_kwargs: Any,
    ) -> None:
        seen["derive"] = cluster_contract

    monkeypatch.setattr(model_optimize, "assert_sae_matches_encdec", _sae)
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", _context)
    setup = model_optimize._prepare_precomputed_bag_setup(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, runtime),
        tokenizer=_CharTokenizer(),
        modes={_POS},
    )
    model_optimize._build_model_backed_precomputed_bags(
        cfg=_cfg(tmp_path),
        paths=_paths(tmp_path),
        runtime_state=cast(model_optimize.RuntimeState, runtime),
        tokenizer=_CharTokenizer(),
        modes={_POS},
        derive=_derive,
        label="clustered",
        shared=setup,
    )

    assert seen == {
        "sae": contract,
        "context": contract,
        "derive": contract,
    }


def test_clustered_prompt_stage_threads_contract_to_every_model_callee(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """D1/D5 clustered conjunct: runtime, metadata, context and scorer agree."""

    contract = _real_contract()
    paths = _paths(tmp_path)
    _write_clustered_source_contract(paths, contract)
    cfg = _cfg(
        tmp_path,
        base_methods=["prompting_decoder_only", "prompting_encdec"],
    )
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name="prompting_decoder_only",
                core_prefix_id="decoder",
                prompt_tokens=[0, 41, 42],
                target_model="decoder_only",
            ),
            _attack_row(
                method_name="prompting_encdec",
                core_prefix_id="encdec",
                prompt_tokens=[0, 51, 52],
                target_model="encdec",
            ),
        ],
        cluster_contract=contract,
    )
    observed: dict[str, Any] = {
        "runtime": [],
        "coverage": [],
        "detect": [],
        "context": [],
        "scorer": [],
    }

    class _Runtime(_FakeRuntimeState):
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            observed["runtime"].append(dict(kwargs))
            self.cluster_contract = kwargs.get("cluster_contract")
            del args

    def _coverage(rows: list[dict[str, Any]], **kwargs: Any) -> Any:
        observed["coverage"].append(dict(kwargs))
        return rows

    def _detect(_model: Any, _cfg_arg: Any, **kwargs: Any) -> Any:
        observed["detect"].append(dict(kwargs))
        return SimpleNamespace()

    def _context(_paths_arg: Any, **kwargs: Any) -> dict[str, Any]:
        observed["context"].append(dict(kwargs))
        return {}

    def _scorer(**kwargs: Any) -> Any:
        observed["scorer"].append(dict(kwargs))
        return SimpleNamespace()

    monkeypatch.setattr(model_optimize, "RuntimeState", _Runtime)
    monkeypatch.setattr(
        model_optimize,
        "require_precomputed_bag_prompt_rows_cover_scored_prefixes",
        _coverage,
    )
    monkeypatch.setattr(model_optimize, "detect_target_model_metadata", _detect)
    monkeypatch.setattr(model_optimize, "load_context_segments_by_prefix_hash", _context)
    monkeypatch.setattr(model_optimize, "ModelBackedDualCompletionScorer", _scorer)
    monkeypatch.setattr(model_optimize, "_build_token_bytes", lambda _tok: t.ones(256))
    monkeypatch.setattr(
        model_optimize,
        "_read_concepts",
        lambda _paths_arg: pytest.fail(
            "clustered candidate stage must not reopen concepts.json"
        ),
    )
    monkeypatch.setattr(
        model_optimize,
        "bpb_surrogate_embedding_weight",
        lambda _model: t.zeros(_VOCAB_SIZE, 1),
    )
    monkeypatch.setattr(model_optimize, "build_allowed_token_ids", lambda _tok: [1])
    monkeypatch.setattr(
        model_optimize,
        "build_universal_gcg_candidates",
        lambda **_kwargs: (pa.Table.from_pylist([], schema=CANDIDATES_SCHEMA), []),
    )

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        cluster_contract=contract,
        **_clustered_source_args(monkeypatch, paths, contract),
    )

    assert observed["runtime"][0]["cluster_contract"] is contract
    assert observed["coverage"][0]["feature_contract_fingerprint"] == (
        contract.feature_contract_fingerprint
    )
    assert observed["detect"] == [
        {"cluster_contract": contract},
        {"cluster_contract": contract},
    ]
    assert observed["context"] == [{"cluster_contract": contract}]
    assert observed["scorer"][0]["cluster_contract"] is contract
    assert observed["scorer"][0]["source_validation"].concept_ids == frozenset(
        {_CONCEPT_ID}
    )
    assert observed["scorer"][0]["concepts_by_id"] == {
        _CONCEPT_ID: _make_concept()
    }


def test_clustered_stage_builds_one_cache_and_shares_exact_object(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T2.4/D3: one cache reaches all four derives through the real bridge."""

    contract = _real_contract()
    cfg, paths = _clustered_k0_fixture(tmp_path, contract=contract)
    shared = _shared_setup([_fx_prefix("pos", _POS), _fx_prefix("neg", _NEG)])
    cache = object()
    cache_calls: list[dict[str, Any]] = []
    family_seen: dict[str, Any] = {}
    runtime_kwargs: list[dict[str, Any]] = []
    _install_clustered_stage_stubs(
        monkeypatch,
        shared=shared,
        cache=cache,
        family_seen=family_seen,
        cache_calls=cache_calls,
        runtime_kwargs=runtime_kwargs,
    )

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        cluster_contract=contract,
        **_clustered_source_args(monkeypatch, paths, contract),
    )

    assert len(cache_calls) == 1
    assert cache_calls[0]["prefixes"] is shared.prefixes
    assert cache_calls[0]["fair_modes"] == {_POS, SteeringMode.NEGATIVE}
    assert cache_calls[0]["roofline_modes"] == {_POS, SteeringMode.NEGATIVE}
    assert cache_calls[0]["cluster_contract"] is contract
    assert set(family_seen) == {
        "fair_gcg",
        "roofline_gcg",
        "fair_diffmean",
        "roofline_diffmean",
    }
    assert all(seen is cache for seen in family_seen.values())
    assert runtime_kwargs[0]["cluster_contract"] is contract


def test_unclustered_stage_never_builds_or_threads_clustered_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T2.5/D3: absent contract means no cache construction and no family kwarg."""

    paths = _paths(tmp_path)
    cfg = _cfg(tmp_path)
    _write_attack_rows(
        paths,
        [
            _attack_row(
                method_name=_FAIR_METHOD,
                core_prefix_id="held1",
                prompt_tokens=[0, 41, 42],
                target_model="encdec",
                k=0,
            )
        ],
    )
    _write_clustered_bag_dependencies(paths)
    shared = _shared_setup([_fx_prefix("pos", _POS), _fx_prefix("neg", _NEG)])
    monkeypatch.setattr(
        model_optimize,
        "_prepare_precomputed_bag_setup",
        lambda **_kwargs: shared,
    )
    monkeypatch.setattr(
        model_optimize,
        "build_clustered_warm_start_cache",
        lambda **_kwargs: pytest.fail("legacy stage built clustered cache"),
    )
    family_kwargs: list[dict[str, Any]] = []

    def _family(**kwargs: Any) -> None:
        family_kwargs.append(dict(kwargs))

    for name in (
        "_build_fair_gcg_bags",
        "_build_roofline_gcg_bags",
        "_build_fair_feature_diffmean_bags",
        "_build_roofline_feature_diffmean_bags",
    ):
        monkeypatch.setattr(model_optimize, name, _family)
    monkeypatch.setattr(model_optimize, "RuntimeState", _FakeRuntimeState)

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
    )

    assert len(family_kwargs) == 4
    assert all("clustered_warm_starts" not in kwargs for kwargs in family_kwargs)


def test_clustered_stage_routes_and_stamps_both_artifacts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T2.1/D4: both output families use clustered atomic writes with identity."""

    contract = _real_contract()
    cfg, paths = _clustered_k0_fixture(tmp_path, contract=contract)
    shared = _shared_setup([_fx_prefix("pos", _POS), _fx_prefix("neg", _NEG)])
    _install_clustered_stage_stubs(
        monkeypatch,
        shared=shared,
        cache=object(),
        family_seen={},
        cache_calls=[],
        runtime_kwargs=[],
    )
    real_writer = model_optimize.write_clustered_table_atomic
    writes: list[tuple[str, ClusteredCandidateSnapshot | None]] = []

    def _clustered_write(
        table: pa.Table,
        path: Path,
        **kwargs: Any,
    ) -> ClusteredCandidateSnapshot | None:
        result = real_writer(table, path, **kwargs)
        writes.append((kwargs["artifact_family"], kwargs.get("candidate_snapshot")))
        return result

    monkeypatch.setattr(
        model_optimize,
        "write_clustered_table_atomic",
        _clustered_write,
    )
    monkeypatch.setattr(
        model_optimize,
        "atomic_write_parquet",
        lambda *_args, **_kwargs: pytest.fail(
            "clustered stage reached legacy parquet writer"
        ),
    )

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        cluster_contract=contract,
        **_clustered_source_args(monkeypatch, paths, contract),
    )

    assert [family for family, _snapshot in writes] == [
        "candidates",
        "universal_triggers",
    ]
    universal_snapshot = writes[1][1]
    assert type(universal_snapshot) is ClusteredCandidateSnapshot
    assert universal_snapshot.payload == paths.candidates_path().read_bytes()
    assert universal_snapshot.table.equals(
        pq.read_table(paths.candidates_path()),
        check_metadata=True,
    )
    attack_identity = artifact_identity_metadata(pq.read_table(paths.attack_rows_path()))
    for path, family in (
        (paths.candidates_path(), "candidates"),
        (paths.universal_triggers_path(), "universal_triggers"),
    ):
        metadata = artifact_identity_metadata(pq.read_table(path))
        assert metadata["clustered_artifact_family"] == family
        assert (
            metadata["feature_contract_fingerprint"]
            == contract.feature_contract_fingerprint
        )
        assert metadata["source_contract_hash"] == attack_identity["source_contract_hash"]


def test_candidate_stage_rejects_attack_rows_from_other_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """T2.2/D4, clustered direction: a different real contract is rejected."""

    written_contract = _real_contract()
    requested_contract = _real_contract(alternate_mask=True)
    cfg, paths = _clustered_k0_fixture(tmp_path, contract=written_contract)
    monkeypatch.setattr(
        model_optimize,
        "read_evaluation_population",
        lambda *_args, **_kwargs: pytest.fail(
            "attack-row identity must reject before the population guard"
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="fingerprint|identity"):
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=_CharTokenizer(),
            cluster_contract=requested_contract,
            **_clustered_source_args(monkeypatch, paths, written_contract),
        )


def test_candidate_stage_unclustered_rejects_cluster_stamped_attack_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """T2.2/D4, legacy direction: stamped rows cannot cross into unclustered."""

    contract = _real_contract()
    cfg, paths = _clustered_k0_fixture(tmp_path, contract=contract)
    monkeypatch.setattr(
        model_optimize,
        "read_evaluation_population",
        lambda *_args, **_kwargs: pytest.fail(
            "attack-row identity must reject before the population guard"
        ),
    )

    with pytest.raises(AxBenchBpbAdvError, match="unclustered reader rejects"):
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=_CharTokenizer(),
        )


class _StageSetupRuntime(_SetupRuntime):
    instances: list[_StageSetupRuntime] = []

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__()
        self.cluster_contract = kwargs.get("cluster_contract")
        del args
        self.instances.append(self)


def test_clustered_offload_toggle_is_artifact_output_inert(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """T2.3/D2: on/off runs differ in lifetime events but not parsed artifacts."""

    monkeypatch.setattr(model_optimize, "RuntimeState", _StageSetupRuntime)
    _stub_prepare_deps(
        monkeypatch,
        prefixes=[_fx_prefix("pos", _POS), _fx_prefix("neg", SteeringMode.NEGATIVE)],
    )
    monkeypatch.setattr(
        model_optimize,
        "assert_sae_matches_encdec",
        lambda _cfg_arg, _model, **_kwargs: None,
    )
    monkeypatch.setattr(
        model_optimize,
        "load_context_segments_by_prefix_hash",
        lambda _paths_arg, **_kwargs: {},
    )
    monkeypatch.setattr(
        model_optimize,
        "build_clustered_warm_start_cache",
        lambda **_kwargs: object(),
    )
    for name in (
        "_build_fair_gcg_bags",
        "_build_roofline_gcg_bags",
        "_build_fair_feature_diffmean_bags",
        "_build_roofline_feature_diffmean_bags",
    ):
        monkeypatch.setattr(model_optimize, name, lambda **_kwargs: None)

    digests: list[tuple[str, str]] = []
    offload_events: list[list[str]] = []
    for enabled, dirname in ((False, "off"), (True, "on")):
        contract = _real_contract()
        cfg, paths = _clustered_k0_fixture(
            tmp_path / dirname,
            contract=contract,
        )
        cfg.offload_between_stages = enabled
        before = len(_StageSetupRuntime.instances)
        run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=_CharTokenizer(),
            cluster_contract=contract,
            **_clustered_source_args(monkeypatch, paths, contract),
        )
        [instance] = _StageSetupRuntime.instances[before:]
        digests.append(
            (
                _table_payload_digest(paths.candidates_path()),
                _table_payload_digest(paths.universal_triggers_path()),
            )
        )
        offload_events.append(list(instance.events))

    # #1245 adds `release_base_pair`: the extraction pair is dropped once the
    # clustered warm-start cache exists. It must land AFTER "sae" -- releasing
    # before the cache is built would hand the derives a released model -- and
    # it fires on BOTH runs, since it is not the offload toggle's business.
    assert offload_events[0] == ["encdec", "sae", "release_base_pair"]
    assert offload_events[1] == [
        "offload",
        "encdec",
        "sae",
        "release_base_pair",
        "offload",
    ]
    for events in offload_events:
        assert events.index("release_base_pair") > events.index("sae")
    assert digests[0] == digests[1]


def test_clustered_offload_uses_real_runtime_state_cpu_path(tmp_path: Path) -> None:
    """T2.3/D2 CPU guard: execute the real ``RuntimeState.offload_models``.

    This proves both cached target modules receive the real ``.to("cpu")`` call.
    Without a GPU it does not prove real CUDA round-trip artifact parity,
    allocator-reserved state, the actual 24-GiB peak, or the temporary overlap
    after encdec restore but before Gemma+SAE load
    (``model_optimize.py:1099-1113``). Those remain explicit H3 GPU acceptance.
    """

    class _CpuModule(t.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = t.nn.Parameter(t.ones(1))
            self.to_calls: list[str] = []

        def to(self, device: Any, *args: Any, **kwargs: Any) -> _CpuModule:
            self.to_calls.append(str(device))
            return cast(_CpuModule, super().to(device, *args, **kwargs))

    cfg = _cfg(tmp_path)
    cfg.offload_between_stages = True
    runtime_state = BpbRuntimeState(
        cfg=AxBenchBpbConfig(
            source_run_dir=cfg.source_run_dir,
            output_root=cfg.output_root,
            feature_dir=cfg.feature_dir,
            feature_statistics_path=cfg.feature_statistics_path,
            device="cpu",
        ),
        ddp_rank=0,
        ddp_world_size=1,
    )
    decoder_only = _CpuModule()
    encdec = _CpuModule()
    runtime_state._decoder_only = cast(
        TargetModelBundle,
        SimpleNamespace(model=decoder_only),
    )
    runtime_state._encdec = cast(
        TargetModelBundle,
        SimpleNamespace(model=encdec),
    )

    runtime_state.offload_models()

    assert decoder_only.to_calls == ["cpu"]
    assert encdec.to_calls == ["cpu"]


class _OrderedForwardModel:
    def __init__(self, events: list[str], *, encdec: bool = False) -> None:
        self.events = events
        if encdec:
            self.encoder = SimpleNamespace()

    def __call__(self, *_args: Any, **_kwargs: Any) -> t.Tensor:
        assert self.events and self.events[-1] == "guard"
        self.events.append("forward")
        return t.ones((1, 2), dtype=t.float32)


def _ordered_bundle(
    *,
    target_name: str,
    model: Any,
    events: list[str],
    cluster_contract: ClusterFeatureContract | None = None,
) -> Any:
    return SimpleNamespace(
        target_name=target_name,
        model=model,
        cluster_contract=cluster_contract,
        ensure_ready_for_forward=lambda: events.append("guard"),
    )


def test_surrogate_single_encdec_forward_runs_guard_immediately_before_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T2.6/D6 site 1: encdec single-forward order is guard then model."""

    events: list[str] = []
    bundle = _ordered_bundle(
        target_name="encdec",
        model=_OrderedForwardModel(events, encdec=True),
        events=events,
    )
    monkeypatch.setattr(surrogate, "assert_enc_ids_in_vocab", lambda *_args: None)
    encoder = surrogate._EncoderTensors(
        feature_ids=t.zeros((1, 1, 1), dtype=t.int64),
        magnitudes=t.zeros((1, 1, 1)),
        mask=t.ones((1, 1, 1), dtype=t.bool),
        seg_map=t.zeros((1, 2), dtype=t.int64),
    )

    surrogate._teacher_forced_model_forward(
        target_bundle=bundle,
        dec_input=t.zeros((1, 2), dtype=t.int64),
        dec_target=t.zeros((1, 2), dtype=t.int64),
        encoder=encoder,
    )

    assert events == ["guard", "forward"]


def test_surrogate_single_decoder_forward_runs_guard_immediately_before_model() -> None:
    """T2.6/D6 site 2: decoder single-forward order is guard then model."""

    events: list[str] = []
    bundle = _ordered_bundle(
        target_name="decoder_only",
        model=_OrderedForwardModel(events),
        events=events,
    )

    surrogate._teacher_forced_model_forward(
        target_bundle=bundle,
        dec_input=t.zeros((1, 2), dtype=t.int64),
        dec_target=t.zeros((1, 2), dtype=t.int64),
        encoder=surrogate._EncoderTensors(),
    )

    assert events == ["guard", "forward"]


def _microbatch_layout() -> surrogate._TeacherForcedLayout:
    return surrogate._TeacherForcedLayout(
        prompt=[1],
        output=[2],
        prompt_prefix_tokens=[],
        full_prefix=[1],
        tokens_dropped_total=0,
        n_full_prefix=1,
        truncated_prefix_segments=(),
        n_prompt_surviving=0,
    )


def test_surrogate_decoder_loss_microbatch_guards_immediately_before_model() -> None:
    """T2.6/D6 site 3: batched loss cannot forward before the bundle guard."""

    events: list[str] = []
    bundle = _ordered_bundle(
        target_name="decoder_only",
        model=_OrderedForwardModel(events),
        events=events,
    )

    _decoder_microbatch_bpb_reference(
        layouts=[_microbatch_layout()],
        target_bundle=bundle,
        output_token_bytes=t.ones(8, dtype=t.int64),
        device=t.device("cpu"),
        t_pad=3,
    )

    assert events == ["guard", "forward"]


def test_surrogate_decoder_gradient_microbatch_guards_immediately_before_model() -> None:
    """T2.6/D6 site 4: batched gradient cannot forward before the bundle guard."""

    events: list[str] = []

    class _GuardedToyDecoder(_ToyDecoder):
        def forward(self, *args: Any, **kwargs: Any) -> t.Tensor:
            assert events and events[-1] == "guard"
            events.append("forward")
            return super().forward(*args, **kwargs)

    bundle = _ordered_bundle(
        target_name="decoder_only",
        model=_GuardedToyDecoder(),
        events=events,
    )

    surrogate._decoder_microbatch_gradient(
        layouts=[_microbatch_layout()],
        target_bundle=bundle,
        output_token_bytes=t.ones(32, dtype=t.int64),
        device=t.device("cpu"),
        t_pad=3,
        embedding_weight=None,
    )

    assert events == ["guard", "forward"]


def test_surrogate_cluster_guard_rejects_before_forward_and_legacy_skips() -> None:
    """T2.7/D6: real contract rejects bad ids; no-contract path reaches forward."""

    contract = _real_contract()
    bad_encoder = surrogate._EncoderTensors(
        feature_ids=t.tensor([[[1]]], dtype=t.int64),
        magnitudes=t.ones((1, 1, 1)),
        mask=t.ones((1, 1, 1), dtype=t.bool),
        seg_map=t.zeros((1, 2), dtype=t.int64),
        allowed_ids=None,
    )

    class _MustNotForward:
        def __call__(self, *_args: Any, **_kwargs: Any) -> t.Tensor:
            raise AssertionError("cluster guard must reject before model forward")

    clustered = TargetModelBundle(
        model=_MustNotForward(),
        tokenizer=_CharTokenizer(),
        max_seq_len=8,
        target_name="encdec",
        cluster_contract=contract,
        cluster_pre_forward_guard=lambda: None,
    )
    with pytest.raises(AxBenchBpbError, match="pre-forward packed IDs"):
        surrogate._teacher_forced_model_forward(
            target_bundle=clustered,
            dec_input=t.zeros((1, 2), dtype=t.int64),
            dec_target=t.zeros((1, 2), dtype=t.int64),
            encoder=bad_encoder,
        )

    events: list[str] = []
    legacy = _ordered_bundle(
        target_name="encdec",
        model=_OrderedForwardModel(events, encdec=True),
        events=events,
    )
    surrogate._teacher_forced_model_forward(
        target_bundle=legacy,
        dec_input=t.zeros((1, 2), dtype=t.int64),
        dec_target=t.zeros((1, 2), dtype=t.int64),
        encoder=bad_encoder,
    )
    assert events == ["guard", "forward"]


def test_surrogate_encoder_input_builder_receives_exact_real_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """D6 packing conjunct: production passes the real contract into the builder."""

    contract = _real_contract()
    assert contract.checkpoint.feature_vocab is not None
    allowed = t.tensor(
        contract.checkpoint.feature_vocab.original_ids,
        dtype=t.int64,
    )
    model = SimpleNamespace(
        feature_vocab=contract.checkpoint.feature_vocab,
        encoder=SimpleNamespace(feature_vocab_ids=allowed),
    )
    bundle = TargetModelBundle(
        model=model,
        tokenizer=_CharTokenizer(),
        max_seq_len=8,
        target_name="encdec",
        cluster_contract=contract,
        cluster_pre_forward_guard=lambda: None,
    )
    seen: list[ClusterFeatureContract | None] = []

    def _build(**kwargs: Any) -> tuple[t.Tensor, t.Tensor, t.Tensor, t.Tensor, int]:
        seen.append(kwargs.get("cluster_contract"))
        return (
            t.zeros((1, 1, 1), dtype=t.int64),
            t.zeros((1, 1, 1)),
            t.ones((1, 1, 1), dtype=t.bool),
            t.zeros((1, 3), dtype=t.int64),
            0,
        )

    monkeypatch.setattr(surrogate, "_build_encdec_seg_map_and_encoder", _build)
    layout = surrogate._TeacherForcedLayout(
        prompt=[1],
        output=[2],
        prompt_prefix_tokens=[],
        full_prefix=[1],
        tokens_dropped_total=0,
        n_full_prefix=1,
        truncated_prefix_segments=(),
        n_prompt_surviving=0,
    )

    surrogate._build_teacher_forced_forward_inputs(
        target_bundle=bundle,
        layout=layout,
        intervention=Intervention.ours(feature_conditions={}),
        device=t.device("cpu"),
    )

    assert seen == [contract]


def test_clustered_stage_resolves_source_identity_once_and_threads_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """#936: one paired source identity reaches every stage consumer.

    The unit tests in ``test_clustered_artifact_identity.py`` prove the binding
    *mechanism* works when an identity is supplied. They say nothing about whether
    production supplies it — removing every ``source_identity=`` argument from this
    module left those tests fully green. This asserts at the layer where the defect
    lives: the resolver runs exactly once per stage, and the object it returns reaches
    every identity consumer, so no two stamps can come from different reads.
    """

    contract = _real_contract()
    cfg, paths = _clustered_k0_fixture(tmp_path, contract=contract)
    shared = _shared_setup([_fx_prefix("pos", _POS), _fx_prefix("neg", _NEG)])
    _install_clustered_stage_stubs(
        monkeypatch,
        shared=shared,
        cache=object(),
        family_seen={},
        cache_calls=[],
        runtime_kwargs=[],
    )

    source_args = _clustered_source_args(monkeypatch, paths, contract)
    identity = source_args["stage_source"].source_identity

    consumers: list[tuple[str, Any]] = []
    for name in (
        "assert_table_cluster_identity",
        "bind_clustered_table",
        "write_clustered_table_atomic",
    ):
        real_fn = getattr(model_optimize, name)

        def _spy(*args: Any, _name: str = name, _real: Any = real_fn, **kwargs: Any) -> Any:
            consumers.append((_name, kwargs.get("source_identity")))
            return _real(*args, **kwargs)

        monkeypatch.setattr(model_optimize, name, _spy)

    run_model_backed_bpb_candidate_stage(
        cfg=cfg,
        paths=paths,
        tokenizer=_CharTokenizer(),
        cluster_contract=contract,
        **source_args,
    )

    assert consumers, "no clustered identity consumer ran"
    for name, passed in consumers:
        assert passed is identity, (
            f"{name} received {passed!r} instead of the stage's single resolved "
            "source identity (#936)"
        )
