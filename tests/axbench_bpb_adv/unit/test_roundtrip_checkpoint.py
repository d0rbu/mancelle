from __future__ import annotations

import hashlib
import json
import multiprocessing
import os
import signal
import time
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from huggingface_hub import _CACHED_NO_EXIST

import steering.eval.axbench_bpb_adv.roundtrip_checkpoint as roundtrip_checkpoint
import steering.eval.axbench_bpb_adv.roundtrip_scoring as roundtrip_scoring
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_base.config import GenerationConfig
from steering.eval.axbench_base.types import Concept
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.config import (
    AdditiveDirectionConfig,
    AxBenchBpbAdvConfig,
    RoundtripGenerationConfig,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.roundtrip_checkpoint import (
    GENERATION_ROW_SCHEMA,
    CheckpointFamily,
    RoundtripCheckpointStore,
    build_roundtrip_checkpoint_fingerprint,
    candidates_payload_sha256,
    warn_on_roundtrip_runtime_mismatch,
)
from steering.eval.axbench_bpb_adv.roundtrip_scoring import (
    RoundtripGeneratedCompletion,
    RoundtripGenerationInputs,
    build_roundtrip_generation_rows,
    run_model_backed_roundtrip_score_stage,
    score_roundtrip_generation_rows,
)
from steering.eval.axbench_bpb_adv.schemas import (
    CANDIDATES_SCHEMA,
    ROUNDTRIP_SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.universal_gcg_checkpoint import (
    ResolvedTargetCheckpoint,
    canonical_sha256,
)
from steering.eval.roundtrip_steering import (
    RoundtripFeatureScores,
    RoundtripProbeConfig,
)
from steering.sae_resolver import L0Tier

_RUNTIME = "1" * 64


class _Tokenizer:
    def encode(self, text: str) -> list[int]:
        return [ord(char) for char in text]

    def decode(self, ids: Sequence[int]) -> str:
        return "".join(chr(int(token)) for token in ids if int(token) != 0)

    def __call__(
        self,
        text: str,
        *,
        padding: bool = False,
        return_offsets_mapping: bool = False,
    ) -> dict[str, Any]:
        del padding
        ids = self.encode(text)
        result: dict[str, Any] = {"input_ids": ids}
        if return_offsets_mapping:
            result["offset_mapping"] = [(index, index + 1) for index in range(len(ids))]
        return result


def _candidate(candidate_id: str, attack_row_id: str) -> dict[str, Any]:
    source = [97, 98]
    rich = [99, 100]
    avoid = [101, 102]
    return {
        "candidate_id": candidate_id,
        "attack_row_id": attack_row_id,
        "core_prefix_id": "core0",
        "concept_rich_sample_id": "rich0",
        "concept_avoiding_sample_id": "avoid0",
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
        "prefix_hash": "hash0",
        "source_prompt_tokens": source,
        "source_prompt_token_digest": token_sequence_digest(source),
        "attacked_prompt_text": "ab",
        "attacked_prompt_tokens": source,
        "attacked_prompt_token_digest": token_sequence_digest(source),
        "concept_rich_output_tokens": rich,
        "concept_avoiding_output_tokens": avoid,
        "concept_rich_output_digest": token_sequence_digest(rich),
        "concept_avoiding_output_digest": token_sequence_digest(avoid),
        "editable_token_start": 2,
        "editable_token_count": 0,
        "replacement_tokens": [],
        "replacement_text": "",
        "objective_value": None,
        "candidate_metadata_json": "{}",
    }


def _candidates(ids: Sequence[str] = ("c0",)) -> pa.Table:
    return pa.Table.from_pylist(
        [_candidate(candidate_id, f"row-{index}") for index, candidate_id in enumerate(ids)],
        schema=CANDIDATES_SCHEMA,
    )


def _concepts() -> dict[int, Concept]:
    return {
        1: Concept(
            concept_id=1,
            feature_index=42,
            layer=8,
            sae_source="8-gemmascope-res-16k",
            description="feature",
            corpus_count=1,
            description_source="fixture",
        )
    }


def _generate_rows(
    candidates: pa.Table,
    generate: Any,
    *,
    store: RoundtripCheckpointStore | None = None,
) -> list[dict[str, Any]]:
    return build_roundtrip_generation_rows(
        candidates=candidates,
        context_segments_by_prefix_hash={},
        tokenizer=_Tokenizer(),
        extraction_tokenizer=_Tokenizer(),
        concepts_by_id=_concepts(),
        generation_cfg=GenerationConfig(max_new_tokens=2),
        reference_segment_size_by_target={"decoder_only": 256},
        samples_per_candidate=2,
        master_seed=123,
        baseline_losses_by_candidate_id={
            str(row["candidate_id"]): (1.0, 3.0) for row in candidates.to_pylist()
        },
        build_intervention_for_row=lambda _row: Intervention.none(),
        build_control_intervention_for_row=lambda _row: Intervention.none(),
        generate_completions=generate,
        checkpoint_store=store,
        runtime_fingerprint=None if store is None else _RUNTIME,
    )


def _generator(calls: list[str]) -> Any:
    def _generate(
        *,
        candidate_row: Mapping[str, Any],
        intervention: Intervention,
        inputs: RoundtripGenerationInputs,
        generation_cfg: GenerationConfig,
        sample_seeds: Sequence[tuple[int, int]],
    ) -> list[RoundtripGeneratedCompletion]:
        del intervention, inputs, generation_cfg
        calls.append(str(candidate_row["candidate_id"]))
        return [
            RoundtripGeneratedCompletion(
                text=f"completion-{sample_index}",
                token_count=2,
                stop_reason="max_tokens",
            )
            for sample_index, _seed in sample_seeds
        ]

    return _generate


def _throwing_generator(**_kwargs: Any) -> NoReturn:
    pytest.fail("restored generation reached the generator")


def _fingerprint() -> dict[str, Any]:
    return {
        "candidates_sha256": "a" * 64,
        "evaluation_population_sha256": "b" * 64,
        "source_contract_sha256": "c" * 64,
        "cluster_feature_contract_fingerprint": None,
        "targets": [{"model_sha256": "d" * 64}],
        "tokenizer_tree_sha256": "e" * 64,
        "generation": {
            "samples_per_candidate": 2,
            "master_seed": 123,
            "temperature": 0.7,
            "top_k_sampling": 50,
            "reference_segment_size_by_target": {"decoder_only": 256},
            "extraction_tokenizer_id": "gemma",
        },
        "probe": {
            "config": {"model_name": "gemma"},
            "feature_statistics_sha256": "f" * 64,
            "feature_chunk_size": 512,
        },
        "expected_candidate_ids": {"count": 2, "sha256": "0" * 64},
    }


def _store(
    root: Path,
    *,
    ids: Sequence[str] = ("c0", "c1"),
    fingerprint: Mapping[str, Any] | None = None,
) -> RoundtripCheckpointStore:
    return RoundtripCheckpointStore.prepare(
        checkpoint_dir=root / "roundtrip_checkpoint",
        fingerprint=_fingerprint() if fingerprint is None else fingerprint,
        expected_candidate_rows=[
            _candidate(candidate_id, f"row-{index}") for index, candidate_id in enumerate(ids)
        ],
        samples_per_candidate=2,
    )


def _probe_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            **dict(row),
            "n_roundtrip_features": 1,
            "roundtrip_pre_act": 1.0,
            "roundtrip_post_act": 2.0,
            "roundtrip_percentile": 3.0,
            "control_n_roundtrip_features": 1,
            "control_roundtrip_pre_act": 4.0,
            "control_roundtrip_post_act": 5.0,
            "control_roundtrip_percentile": 6.0,
        }
        for row in rows
    ]


def _parquet_bytes(table: pa.Table) -> bytes:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return sink.getvalue().to_pybytes()


def _replace_shard_table(shard: Path, table: pa.Table) -> Path:
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    payload = sink.getvalue().to_pybytes()
    key_hash = shard.stem.split("_")[1]
    replacement = shard.with_name(f"cand_{key_hash}_{hashlib.sha256(payload).hexdigest()}.parquet")
    shard.unlink()
    replacement.write_bytes(payload)
    return replacement


def test_generation_and_probe_shards_roundtrip_exact_rows(tmp_path: Path) -> None:
    candidates = _candidates(("c0",))
    rows = _generate_rows(candidates, _generator([]))
    store = _store(tmp_path, ids=("c0",))
    assert store.commit_generation_rows("c0", rows, _RUNTIME)
    assert store.commit_probe_rows("c0", _probe_rows(rows), _RUNTIME)

    restored = _store(tmp_path, ids=("c0",))
    assert restored.load_generation_rows() == {"c0": rows}
    assert restored.load_probe_rows() == {"c0": _probe_rows(rows)}


def test_fingerprint_builder_binds_every_semantic_component(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    hashes = {
        str(paths.evaluation_population()): "1" * 64,
        str(paths.source_contract()): "2" * 64,
        str(tmp_path / "stats.safetensors"): "3" * 64,
        str(tmp_path / "probe-stats.safetensors"): "7" * 64,
    }
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_hash_file",
        lambda path, **_kwargs: hashes[str(path)],
    )
    monkeypatch.setattr(roundtrip_checkpoint, "get_base_dir", lambda: str(tmp_path))
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "tree_sha256",
        lambda path: "4" * 64 if path == tmp_path / "tokenizer" else pytest.fail(str(path)),
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_extraction_tokenizer_content_identity",
        lambda _tokenizer_id: {"snapshot_revision": "extract-rev"},
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_probe_artifact_identity",
        lambda _probe: {"base_model": {"snapshot_revision": "probe-rev"}},
    )
    target = ResolvedTargetCheckpoint(
        target_name="decoder_only",
        source="base",
        model_tag="d12",
        step=7,
        checkpoint_dir=tmp_path / "checkpoint",
        model_path=tmp_path / "checkpoint/model_000007.pt",
        metadata_path=tmp_path / "checkpoint/meta_000007.json",
        model_sha256="5" * 64,
        metadata_sha256="6" * 64,
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path),
        run_tag="run",
        device="cpu",
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        extraction_tokenizer_id="extractor",
        lora_adapter={"rank": 2},
        diffmean_and_probe_layer=7,
    )
    cfg.roundtrip_steering = replace(
        cfg.roundtrip_steering,
        probe=RoundtripProbeConfig(model_name="gemma", layer=8),
        feature_chunk_size=512,
    )
    fingerprint = build_roundtrip_checkpoint_fingerprint(
        cfg=cfg,
        paths=paths,
        candidates_sha256="a" * 64,
        candidate_rows=_candidates(("c1", "c0")).to_pylist(),
        cluster_feature_contract_fingerprint="f" * 64,
        resolved_targets={"decoder_only": target},
        reference_segment_size_by_target={"decoder_only": 256},
        probe_statistics_path=tmp_path / "probe-stats.safetensors",
    )
    assert fingerprint["store_algorithm_version"].endswith("checkpoint_v3")
    assert fingerprint["candidates_sha256"] == "a" * 64
    assert fingerprint["evaluation_population_sha256"] == "1" * 64
    assert fingerprint["source_contract_sha256"] == "2" * 64
    assert fingerprint["cluster_feature_contract_fingerprint"] == "f" * 64
    assert fingerprint["tokenizer_tree_sha256"] == "4" * 64
    assert fingerprint["targets"] == [
        {
            "target_name": "decoder_only",
            "source": "base",
            "model_tag": "d12",
            "step": 7,
            "model_sha256": "5" * 64,
            "metadata_sha256": "6" * 64,
        }
    ]
    assert fingerprint["generation_semantics"] == {
        "lora_adapter": {"rank": 2},
        "diffmean_and_probe_layer": 7,
        "counter_direction_family": "diffmean",
        "context_cache_tree_sha256": "unused",
        "unclustered_precomputed_bags_tree_sha256": "unused",
        "feature_statistics_sha256": "3" * 64,
        "extraction_tokenizer": {"snapshot_revision": "extract-rev"},
        "roundtrip_generation": {
            "samples_per_candidate": 4,
            "temperature": 0.7,
            "top_k_sampling": 50,
            "master_seed": 42,
        },
        "reference_segment_size_by_target": {"decoder_only": 256},
    }
    assert fingerprint["probe"] == {
        "config": {
            "model_name": "gemma",
            "sae_release": "gemmascope",
            "activation_type": "residual",
            "layer": 8,
            "sae_width": "16k",
            "l0": None,
            "device": "auto",
        },
        "artifacts": {"base_model": {"snapshot_revision": "probe-rev"}},
        "feature_statistics_sha256": "7" * 64,
        "feature_chunk_size": 512,
    }
    assert fingerprint["expected_candidate_ids"]["count"] == 2


@pytest.mark.parametrize(
    "semantic_field",
    [
        "lora_adapter",
        "diffmean_and_probe_layer",
        "counter_direction_family",
        "feature_statistics_path",
        "extraction_tokenizer_id",
        "samples_per_candidate",
        "master_seed",
        "temperature",
        "top_k_sampling",
        "probe_config",
        "feature_chunk_size",
    ],
)
def test_fingerprint_includes_semantics_and_excludes_operations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    semantic_field: str,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_hash_file",
        lambda path, **_kwargs: canonical_sha256({"file": str(path)}),
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "tree_sha256",
        lambda path: canonical_sha256({"tree": str(path)}),
    )
    monkeypatch.setattr(roundtrip_checkpoint, "get_base_dir", lambda: str(tmp_path / "base"))
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_extraction_tokenizer_content_identity",
        lambda tokenizer_id: {"tokenizer_id": tokenizer_id},
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_probe_artifact_identity",
        lambda probe: {
            "base_model": probe.model_name,
            "sae": [
                probe.sae_release,
                probe.activation_type,
                probe.layer,
                probe.sae_width,
                probe.l0,
            ],
        },
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path),
        run_tag="run",
        device="cpu",
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
        extraction_tokenizer_id="extractor",
        diffmean_and_probe_layer=6,
        lora_adapter={"rank": 2},
        adversarial_methods=[AdditiveDirectionConfig(counter_direction_family="diffmean")],
    )
    cfg.roundtrip_generation = RoundtripGenerationConfig(
        samples_per_candidate=2,
        temperature=0.7,
        top_k_sampling=50,
        master_seed=42,
    )
    candidate = _candidate("c0", "row-0")
    candidate["target_model"] = "encdec"
    candidate["method_name"] = "ours_gcg_encdec"
    target = ResolvedTargetCheckpoint(
        target_name="encdec",
        source="encdec",
        model_tag="d12",
        step=7,
        checkpoint_dir=tmp_path / "checkpoint",
        model_path=tmp_path / "checkpoint/model_000007.pt",
        metadata_path=tmp_path / "checkpoint/meta_000007.json",
        model_sha256="5" * 64,
        metadata_sha256="6" * 64,
    )

    def _build(
        current_cfg: AxBenchBpbAdvConfig,
        *,
        current_candidate: Mapping[str, Any] = candidate,
        segment_size: int = 256,
        cluster_fingerprint: str | None = None,
        current_target: ResolvedTargetCheckpoint = target,
    ) -> dict[str, Any]:
        return build_roundtrip_checkpoint_fingerprint(
            cfg=current_cfg,
            paths=paths,
            candidates_sha256=canonical_sha256(current_candidate),
            candidate_rows=[current_candidate],
            cluster_feature_contract_fingerprint=cluster_fingerprint,
            resolved_targets={"encdec": current_target},
            reference_segment_size_by_target={"encdec": segment_size},
            probe_statistics_path=tmp_path / "probe-stats.safetensors",
        )

    baseline = _build(cfg)
    operational = deepcopy(cfg)
    operational.log_progress_every = 7
    operational.offload_between_stages = False
    operational.run_tag = "other"
    operational.output_root = str(tmp_path / "other")
    operational.device = "mps"
    operational.profile = replace(operational.profile, enabled=True)
    operational.plot_methods = ["anything"]
    operational.scatter_baseline_x = "min"
    operational.scores_path = str(tmp_path / "scores.parquet")
    assert _build(operational) == baseline

    variants: dict[str, AxBenchBpbAdvConfig] = {}
    for name, mutation in (
        ("lora_adapter", lambda changed: setattr(changed, "lora_adapter", {"rank": 3})),
        (
            "diffmean_and_probe_layer",
            lambda changed: setattr(changed, "diffmean_and_probe_layer", 7),
        ),
        (
            "counter_direction_family",
            lambda changed: setattr(
                changed,
                "adversarial_methods",
                [AdditiveDirectionConfig(counter_direction_family="linear_probe")],
            ),
        ),
        (
            "feature_statistics_path",
            lambda changed: setattr(changed, "feature_statistics_path", "other-stats"),
        ),
        (
            "extraction_tokenizer_id",
            lambda changed: setattr(changed, "extraction_tokenizer_id", "other-tokenizer"),
        ),
        (
            "samples_per_candidate",
            lambda changed: setattr(
                changed,
                "roundtrip_generation",
                replace(changed.roundtrip_generation, samples_per_candidate=3),
            ),
        ),
        (
            "master_seed",
            lambda changed: setattr(
                changed,
                "roundtrip_generation",
                replace(changed.roundtrip_generation, master_seed=43),
            ),
        ),
        (
            "temperature",
            lambda changed: setattr(
                changed,
                "roundtrip_generation",
                replace(changed.roundtrip_generation, temperature=0.8),
            ),
        ),
        (
            "top_k_sampling",
            lambda changed: setattr(
                changed,
                "roundtrip_generation",
                replace(changed.roundtrip_generation, top_k_sampling=51),
            ),
        ),
        (
            "probe_config",
            lambda changed: setattr(
                changed,
                "roundtrip_steering",
                replace(
                    changed.roundtrip_steering,
                    probe=replace(changed.roundtrip_steering.probe, layer=9),
                ),
            ),
        ),
        (
            "feature_chunk_size",
            lambda changed: setattr(
                changed,
                "roundtrip_steering",
                replace(changed.roundtrip_steering, feature_chunk_size=513),
            ),
        ),
    ):
        changed = deepcopy(cfg)
        mutation(changed)
        variants[name] = changed
    assert _build(variants[semantic_field]) != baseline
    assert _build(cfg, segment_size=257) != baseline
    assert _build(cfg, cluster_fingerprint="7" * 64) != baseline
    assert _build(cfg, current_target=replace(target, model_sha256="8" * 64)) != baseline


@pytest.mark.parametrize("mutated_file", ["sae", "model_config"])
def test_probe_artifact_identity_is_content_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutated_file: str,
) -> None:
    sae_snapshot = tmp_path / "models--sae" / "snapshots" / "sae-revision"
    model_snapshot = tmp_path / "models--model" / "snapshots" / "model-revision"
    sae_weight = sae_snapshot / "layer_8/width_16k/average_l0_71/params.npz"
    model_config = model_snapshot / "config.json"
    sae_weight.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)
    sae_weight.write_bytes(b"sae-v1")
    model_config.write_bytes(b"config-v1")

    def _cached(_repo_id: str, filename: str, *, label: str) -> Path:
        del label
        return model_config if filename == "config.json" else sae_weight

    monkeypatch.setattr(roundtrip_checkpoint, "_cached_hf_file", _cached)
    probe = RoundtripProbeConfig(model_name="gemma-2-2b", layer=8)
    baseline = roundtrip_checkpoint.resolve_probe_artifact_identity(probe)
    (sae_weight if mutated_file == "sae" else model_config).write_bytes(b"mutated")
    assert roundtrip_checkpoint.resolve_probe_artifact_identity(probe) != baseline


def test_extraction_tokenizer_local_content_is_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    tokenizer_dir = tmp_path / "tokenizer"
    tokenizer_dir.mkdir()
    tokenizer_file = tokenizer_dir / "tokenizer.json"
    tokenizer_file.write_bytes(b"v1")
    monkeypatch.setattr(
        roundtrip_checkpoint, "get_extraction_tokenizer", lambda _tokenizer_id: object()
    )
    baseline = roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity(
        str(tokenizer_dir)
    )
    tokenizer_file.write_bytes(b"v2")
    assert (
        roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity(str(tokenizer_dir))
        != baseline
    )


@pytest.mark.parametrize(
    "artifact",
    [
        "evaluation_population",
        "source_contract",
        "main_statistics",
        "probe_statistics",
        "nanochat_tokenizer",
        "context_cache",
        "precomputed_bags",
        "extraction_tokenizer",
        "probe_artifacts",
    ],
)
def test_every_semantic_artifact_content_independently_changes_fingerprint(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    artifact: str,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path),
        run_tag="run",
        device="cpu",
        feature_statistics_path=str(tmp_path / "main-stats.safetensors"),
    )
    candidate = _candidate("c0", "row-0")
    candidate["target_model"] = "encdec"
    candidate["method_name"] = "ours_gcg_encdec"
    target = ResolvedTargetCheckpoint(
        target_name="encdec",
        source="encdec",
        model_tag="d12",
        step=7,
        checkpoint_dir=tmp_path / "checkpoint",
        model_path=tmp_path / "checkpoint/model.pt",
        metadata_path=tmp_path / "checkpoint/meta.json",
        model_sha256="5" * 64,
        metadata_sha256="6" * 64,
    )
    salts = dict.fromkeys(
        (
            "evaluation_population",
            "source_contract",
            "main_statistics",
            "probe_statistics",
            "nanochat_tokenizer",
            "context_cache",
            "precomputed_bags",
            "extraction_tokenizer",
            "probe_artifacts",
        ),
        "v1",
    )
    file_categories = {
        str(paths.evaluation_population()): "evaluation_population",
        str(paths.source_contract()): "source_contract",
        str(tmp_path / "main-stats.safetensors"): "main_statistics",
        str(tmp_path / "probe-stats.safetensors"): "probe_statistics",
    }
    tree_categories = {
        str(tmp_path / "base/tokenizer"): "nanochat_tokenizer",
        str(paths.context_cache_dir()): "context_cache",
        str(paths.ours_gcg_bags_dir()): "precomputed_bags",
    }
    monkeypatch.setattr(roundtrip_checkpoint, "get_base_dir", lambda: str(tmp_path / "base"))
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_hash_file",
        lambda path, **_kwargs: canonical_sha256(
            {"category": file_categories[str(path)], "salt": salts[file_categories[str(path)]]}
        ),
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "tree_sha256",
        lambda path: canonical_sha256(
            {"category": tree_categories[str(path)], "salt": salts[tree_categories[str(path)]]}
        ),
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_extraction_tokenizer_content_identity",
        lambda _tokenizer_id: {"pin": salts["extraction_tokenizer"]},
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_probe_artifact_identity",
        lambda _probe: {"pin": salts["probe_artifacts"]},
    )

    def _build() -> dict[str, Any]:
        return build_roundtrip_checkpoint_fingerprint(
            cfg=cfg,
            paths=paths,
            candidates_sha256="a" * 64,
            candidate_rows=[candidate],
            cluster_feature_contract_fingerprint=None,
            resolved_targets={"encdec": target},
            reference_segment_size_by_target={"encdec": 256},
            probe_statistics_path=tmp_path / "probe-stats.safetensors",
        )

    baseline = _build()
    salts[artifact] = "v2"
    assert _build() != baseline


def test_candidate_payload_hash_binds_exact_consumed_table(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    table = _candidates(("c0",))
    paths.candidates_dir().mkdir(parents=True)
    pq.write_table(table, paths.candidates_path())
    payload = paths.candidates_path().read_bytes()
    digest = hashlib.sha256(payload).hexdigest()
    assert candidates_payload_sha256(candidates=table, paths=paths) == digest
    assert (
        candidates_payload_sha256(
            candidates=table,
            paths=paths,
            snapshot_payload=payload,
            snapshot_sha256=digest,
        )
        == digest
    )
    with pytest.raises(AxBenchBpbAdvError, match="exact consumed candidate table"):
        candidates_payload_sha256(candidates=_candidates(("other",)), paths=paths)


@pytest.mark.parametrize(
    ("mutation", "field"),
    [
        (lambda fp: fp.__setitem__("candidates_sha256", "9" * 64), "candidates_sha256"),
        (
            lambda fp: fp.__setitem__("evaluation_population_sha256", "9" * 64),
            "evaluation_population_sha256",
        ),
        (lambda fp: fp["generation"].__setitem__("master_seed", 999), "master_seed"),
        (
            lambda fp: fp["generation"].__setitem__("samples_per_candidate", 3),
            "samples_per_candidate",
        ),
        (lambda fp: fp["generation"].__setitem__("temperature", 0.8), "temperature"),
        (lambda fp: fp["generation"].__setitem__("top_k_sampling", 51), "top_k_sampling"),
        (lambda fp: fp["probe"]["config"].__setitem__("model_name", "other"), "model_name"),
        (
            lambda fp: fp["probe"].__setitem__("feature_statistics_sha256", "9" * 64),
            "feature_statistics_sha256",
        ),
        (
            lambda fp: fp["expected_candidate_ids"].__setitem__("sha256", "9" * 64),
            "expected_candidate_ids.sha256",
        ),
    ],
)
def test_manifest_identity_fences_every_component(
    tmp_path: Path,
    mutation: Any,
    field: str,
) -> None:
    _store(tmp_path)
    changed = deepcopy(_fingerprint())
    mutation(changed)
    with pytest.raises(AxBenchBpbAdvError, match=rf"{field}.*fresh run_tag"):
        _store(tmp_path, fingerprint=changed)


def test_manifest_rejects_pre_pr7_store_algorithm_version(tmp_path: Path) -> None:
    old = deepcopy(_fingerprint())
    old["store_algorithm_version"] = (
        "axbench_bpb_adv_roundtrip_candidate_checkpoint_v2"
    )
    _store(tmp_path, fingerprint=old)
    current = deepcopy(_fingerprint())
    current["store_algorithm_version"] = roundtrip_checkpoint.STORE_ALGORITHM_VERSION

    with pytest.raises(
        AxBenchBpbAdvError,
        match=r"store_algorithm_version.*fresh run_tag",
    ):
        _store(tmp_path, fingerprint=current)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("truncated", "payload digest differs"),
        ("payload_name", "payload digest differs"),
        ("schema", "logical schema differs"),
        ("duplicate", "multiple committed payloads"),
        ("unknown", "unknown candidate-key hash"),
        ("missing_sample", "sample_index coverage differs"),
        ("bucket", "bucket differs"),
        ("metadata_key", "metadata keys differ"),
        ("metadata_value", "metadata roundtrip_checkpoint_family differs"),
        ("list_child_null", "contains null values"),
        ("filename", "unknown committed checkpoint filename"),
        ("unreadable", "cannot read shard"),
        ("retained_none", "read retained no payload"),
        ("invalid_parquet", "not valid Parquet"),
        ("runtime_nonascii", "runtime hash is not ASCII"),
        ("runtime_invalid", "runtime hash is invalid"),
        ("scalar_null", "contains null values"),
        ("empty_table", "sample_index coverage differs"),
        ("duplicate_sample", "duplicate sample_index values"),
        ("wrong_candidate", "candidate_id differs"),
        ("symlink", "not a real regular file"),
    ],
)
def test_generation_shard_corruption_fails_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: str,
    error: str,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    shard = next((store.checkpoint_dir / "generation" / "groups").glob("*/*.parquet"))
    if mutation == "truncated":
        shard.write_bytes(shard.read_bytes()[:16])
    elif mutation == "payload_name":
        shard.rename(shard.with_name(f"{shard.stem.rsplit('_', 1)[0]}_{'9' * 64}.parquet"))
    elif mutation == "schema":
        table = pq.read_table(shard).append_column("extra", pa.array([1, 2]))
        _replace_shard_table(shard, table)
    elif mutation == "duplicate":
        table = pq.read_table(shard)
        metadata = dict(table.schema.metadata or {})
        metadata[b"runtime_numeric_fingerprint_sha256"] = b"2" * 64
        sink = pa.BufferOutputStream()
        pq.write_table(table.replace_schema_metadata(metadata), sink)
        payload = sink.getvalue().to_pybytes()
        shard.with_name(
            f"cand_{shard.stem.split('_')[1]}_{hashlib.sha256(payload).hexdigest()}.parquet"
        ).write_bytes(payload)
    elif mutation == "unknown":
        payload_hash = shard.stem.split("_")[2]
        unknown = "9" * 64
        bucket = store.checkpoint_dir / "generation" / "groups" / "99"
        bucket.mkdir()
        shard.rename(bucket / f"cand_{unknown}_{payload_hash}.parquet")
    elif mutation == "missing_sample":
        _replace_shard_table(shard, pq.read_table(shard).slice(0, 1))
    elif mutation == "bucket":
        wrong_name = "ff" if shard.parent.name != "ff" else "00"
        wrong_bucket = shard.parent.parent / wrong_name
        wrong_bucket.mkdir()
        shard.rename(wrong_bucket / shard.name)
    elif mutation in {"metadata_key", "metadata_value"}:
        table = pq.read_table(shard)
        metadata = dict(table.schema.metadata or {})
        if mutation == "metadata_key":
            metadata.pop(b"roundtrip_checkpoint_family")
        else:
            metadata[b"roundtrip_checkpoint_family"] = b"probe"
        _replace_shard_table(shard, table.replace_schema_metadata(metadata))
    elif mutation == "list_child_null":
        table = pq.read_table(shard)
        tampered = table.to_pylist()
        tampered[0]["source_prompt_tokens"] = [97, None]
        _replace_shard_table(
            shard,
            pa.Table.from_pylist(tampered, schema=GENERATION_ROW_SCHEMA).replace_schema_metadata(
                table.schema.metadata
            ),
        )
    elif mutation == "filename":
        shard.rename(shard.with_name("not-a-shard.parquet"))
    elif mutation in {"unreadable", "retained_none"}:
        original_hash = roundtrip_checkpoint.stream_hash_file_descriptor_bound

        def _hash(path: Path, **kwargs: Any) -> Any:
            if path == shard:
                if mutation == "unreadable":
                    raise OSError("cannot read shard")
                return SimpleNamespace(
                    sha256=shard.stem.split("_")[2],
                    payload=None,
                )
            return original_hash(path, **kwargs)

        monkeypatch.setattr(
            roundtrip_checkpoint,
            "stream_hash_file_descriptor_bound",
            _hash,
        )
    elif mutation == "invalid_parquet":
        payload = b"not parquet"
        key_hash = shard.stem.split("_")[1]
        replacement = shard.with_name(
            f"cand_{key_hash}_{hashlib.sha256(payload).hexdigest()}.parquet"
        )
        shard.unlink()
        replacement.write_bytes(payload)
    elif mutation in {"runtime_nonascii", "runtime_invalid"}:
        table = pq.read_table(shard)
        metadata = dict(table.schema.metadata or {})
        metadata[b"runtime_numeric_fingerprint_sha256"] = (
            b"\xff" if mutation == "runtime_nonascii" else b"z" * 64
        )
        _replace_shard_table(shard, table.replace_schema_metadata(metadata))
    elif mutation == "scalar_null":
        table = pq.read_table(shard)
        tampered = table.to_pylist()
        tampered[0]["generated_text"] = None
        _replace_shard_table(
            shard,
            pa.Table.from_pylist(tampered, schema=GENERATION_ROW_SCHEMA).replace_schema_metadata(
                table.schema.metadata
            ),
        )
    elif mutation == "empty_table":
        _replace_shard_table(shard, pq.read_table(shard).slice(0, 0))
    elif mutation == "duplicate_sample":
        table = pq.read_table(shard)
        tampered = table.to_pylist()
        tampered[1]["sample_index"] = 0
        _replace_shard_table(
            shard,
            pa.Table.from_pylist(tampered, schema=GENERATION_ROW_SCHEMA).replace_schema_metadata(
                table.schema.metadata
            ),
        )
    elif mutation == "wrong_candidate":
        table = pq.read_table(shard)
        tampered = table.to_pylist()
        tampered[0]["candidate_id"] = "c1"
        _replace_shard_table(
            shard,
            pa.Table.from_pylist(tampered, schema=GENERATION_ROW_SCHEMA).replace_schema_metadata(
                table.schema.metadata
            ),
        )
    elif mutation == "symlink":
        link = shard.with_name(f"cand_{'8' * 64}_{'8' * 64}.parquet")
        link.symlink_to(shard)
    else:
        pytest.fail(mutation)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        _store(tmp_path, ids=("c0",))


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("without_generation", "has no generation shard"),
        ("shared_disagreement", "generation payload differs"),
        ("incomplete_coverage", "sample_index coverage differs"),
    ],
)
def test_probe_shard_cross_family_corruption_fails_loud(
    tmp_path: Path, mutation: str, error: str
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    store.commit_probe_rows("c0", _probe_rows(rows), _RUNTIME)
    generation_shard = next((store.checkpoint_dir / "generation" / "groups").glob("*/*.parquet"))
    probe_shard = next((store.checkpoint_dir / "probe" / "groups").glob("*/*.parquet"))
    if mutation == "without_generation":
        generation_shard.unlink()
    else:
        table = pq.read_table(probe_shard)
        if mutation == "shared_disagreement":
            tampered = table.to_pylist()
            tampered[0]["loss_concept_rich"] = 99.0
            replacement = pa.Table.from_pylist(
                tampered, schema=ROUNDTRIP_SCORE_SCHEMA
            ).replace_schema_metadata(table.schema.metadata)
        elif mutation == "incomplete_coverage":
            replacement = table.slice(0, 1)
        else:
            pytest.fail(mutation)
        _replace_shard_table(probe_shard, replacement)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        _store(tmp_path, ids=("c0",))


@pytest.mark.parametrize(
    ("family", "field", "value", "error"),
    [
        (CheckpointFamily.GENERATION, "n_generated_tokens", -1, "must be in"),
        (CheckpointFamily.GENERATION, "n_generated_tokens", 3, "must be in"),
        (CheckpointFamily.GENERATION, "control_n_generated_tokens", -1, "must be in"),
        (CheckpointFamily.GENERATION, "control_n_generated_tokens", 3, "must be in"),
        (CheckpointFamily.GENERATION, "stop_reason", "eos", "stop_reason is invalid"),
        (
            CheckpointFamily.GENERATION,
            "control_stop_reason",
            "eos",
            "control_stop_reason is invalid",
        ),
        (CheckpointFamily.GENERATION, "generated_text", "", "text/token-count consistency"),
        (CheckpointFamily.GENERATION, "n_generated_tokens", 0, "text/token-count consistency"),
        (
            CheckpointFamily.GENERATION,
            "control_generated_text",
            "",
            "control_generated_text/token-count consistency",
        ),
        (CheckpointFamily.GENERATION, "concept_id", 99, "static candidate projection differs"),
        (
            CheckpointFamily.GENERATION,
            "reference_segment_token_count",
            0,
            "reference_segment_token_count must be >= 1",
        ),
        (CheckpointFamily.PROBE, "roundtrip_pre_act", float("nan"), "must be finite"),
        (CheckpointFamily.PROBE, "roundtrip_post_act", float("-inf"), "must be finite"),
        (CheckpointFamily.PROBE, "roundtrip_percentile", float("inf"), "must be finite"),
        (CheckpointFamily.PROBE, "control_roundtrip_pre_act", float("nan"), "must be finite"),
        (CheckpointFamily.PROBE, "control_roundtrip_post_act", float("-inf"), "must be finite"),
        (
            CheckpointFamily.PROBE,
            "control_roundtrip_percentile",
            float("inf"),
            "must be finite",
        ),
    ],
)
def test_row_domain_guards_reject_each_invalid_mutation(
    tmp_path: Path,
    family: CheckpointFamily,
    field: str,
    value: Any,
    error: str,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    if family == CheckpointFamily.PROBE:
        store.commit_generation_rows("c0", rows, _RUNTIME)
        payload = _probe_rows(rows)
    else:
        payload = deepcopy(rows)
    payload[0][field] = value
    commit = store.commit_probe_rows if family == CheckpointFamily.PROBE else store.commit_generation_rows
    with pytest.raises(AxBenchBpbAdvError, match=error):
        commit("c0", payload, _RUNTIME)


def test_manifest_schema_version_rejects_bool(tmp_path: Path) -> None:
    store = _store(tmp_path)
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    manifest["schema_version"] = True
    store.manifest_path.write_bytes(roundtrip_checkpoint.canonical_json_bytes(manifest))
    with pytest.raises(AxBenchBpbAdvError, match="invalid type/value"):
        _store(tmp_path)


def test_runtime_fingerprint_requires_string(tmp_path: Path) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    with pytest.raises(AxBenchBpbAdvError, match="runtime fingerprint is invalid"):
        store.commit_generation_rows("c0", rows, 1)  # ty: ignore[invalid-argument-type]


def test_idempotent_recommit_and_divergent_payload(tmp_path: Path) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    assert store.commit_generation_rows("c0", rows, _RUNTIME)
    assert not store.commit_generation_rows("c0", rows, "2" * 64)
    changed = deepcopy(rows)
    changed[0]["generated_text"] = "different"
    with pytest.raises(AxBenchBpbAdvError, match="differ from committed candidate"):
        store.commit_generation_rows("c0", changed, _RUNTIME)


@pytest.mark.parametrize("family", [CheckpointFamily.GENERATION, CheckpointFamily.PROBE])
def test_idempotent_recommit_compares_float_bit_patterns(
    tmp_path: Path,
    family: CheckpointFamily,
) -> None:
    generation = _generate_rows(_candidates(("c0",)), _generator([]))
    generation[0]["loss_concept_rich"] = 0.0
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", generation, _RUNTIME)
    if family == CheckpointFamily.PROBE:
        payload = _probe_rows(generation)
        payload[0]["roundtrip_pre_act"] = 0.0
        store.commit_probe_rows("c0", payload, _RUNTIME)
        changed = deepcopy(payload)
        changed[0]["roundtrip_pre_act"] = -0.0
        commit = store.commit_probe_rows
    else:
        changed = deepcopy(generation)
        changed[0]["loss_concept_rich"] = -0.0
        commit = store.commit_generation_rows
    with pytest.raises(AxBenchBpbAdvError, match="differ from committed candidate"):
        commit("c0", changed, _RUNTIME)


def test_restored_probe_memo_conflict_compares_float_bit_patterns(
    tmp_path: Path,
) -> None:
    generation = _generate_rows(_candidates(("c0", "c1")), _generator([]))
    store = _store(tmp_path)
    for index, candidate_id in enumerate(("c0", "c1")):
        candidate_generation = generation[index * 2 : (index + 1) * 2]
        store.commit_generation_rows(candidate_id, candidate_generation, _RUNTIME)
        probe_rows = _probe_rows(candidate_generation)
        probe_rows[0]["roundtrip_pre_act"] = 0.0 if index == 0 else -0.0
        store.commit_probe_rows(candidate_id, probe_rows, _RUNTIME)
    restored = _store(tmp_path)
    with pytest.raises(AxBenchBpbAdvError, match="restored probe score conflicts"):
        score_roundtrip_generation_rows(
            generation_rows=generation,
            probe_model=object(),
            percentile_table=None,
            feature_chunk_size=1,
            probe_text_fn=_throwing_probe,
            checkpoint_store=restored,
            runtime_fingerprint=_RUNTIME,
        )


@pytest.mark.parametrize("family", [CheckpointFamily.GENERATION, CheckpointFamily.PROBE])
def test_runtime_mismatch_warns_but_does_not_invalidate(
    tmp_path: Path,
    family: CheckpointFamily,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    if family == CheckpointFamily.PROBE:
        store.commit_probe_rows("c0", _probe_rows(rows), _RUNTIME)
    restored = _store(tmp_path, ids=("c0",))
    warnings: list[tuple[str, tuple[Any, ...]]] = []
    original = roundtrip_checkpoint.logger.warning
    roundtrip_checkpoint.logger.warning = (  # ty: ignore[invalid-assignment]
        lambda message, *args: warnings.append((message, args))
    )
    try:
        warn_on_roundtrip_runtime_mismatch(restored, family, "2" * 64)
    finally:
        roundtrip_checkpoint.logger.warning = original  # ty: ignore[invalid-assignment]
    assert warnings and warnings[0][1] == (
        family.upper(),
        "2" * 64,
        [_RUNTIME],
    )
    assert restored.load_generation_rows() == {"c0": rows}


def test_generation_restore_skips_calls_and_preseeds_fresh_twin(tmp_path: Path) -> None:
    candidates = _candidates(("c0", "c1"))
    reference = _generate_rows(candidates, _generator([]))
    partial = _store(tmp_path)
    partial.commit_generation_rows("c0", reference[:2], _RUNTIME)

    restored = _store(tmp_path)
    calls: list[str] = []
    actual = _generate_rows(candidates, _generator(calls), store=restored)
    assert calls == []
    assert actual == reference
    assert _parquet_bytes(pa.Table.from_pylist(actual, schema=GENERATION_ROW_SCHEMA)) == (
        _parquet_bytes(pa.Table.from_pylist(reference, schema=GENERATION_ROW_SCHEMA))
    )


def test_restored_prompt_digest_mismatch_fails_before_generation(tmp_path: Path) -> None:
    candidates = _candidates(("c0",))
    rows = _generate_rows(candidates, _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    shard = next((store.checkpoint_dir / "generation" / "groups").glob("*/*.parquet"))
    table = pq.read_table(shard)
    tampered = table.to_pylist()
    tampered[0]["roundtrip_prompt_tokens"] = [120]
    _replace_shard_table(
        shard,
        pa.Table.from_pylist(tampered, schema=GENERATION_ROW_SCHEMA).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    restored = _store(tmp_path, ids=("c0",))
    with pytest.raises(AxBenchBpbAdvError, match="restored generation prompt"):
        _generate_rows(candidates, _throwing_generator, store=restored)


def test_interrupted_subset_reports_counts_and_completes_with_parity(tmp_path: Path) -> None:
    candidates = _candidates(("c0", "c1"))
    reference = _generate_rows(candidates, _generator([]))
    partial = _store(tmp_path)
    partial.commit_generation_rows("c0", reference[:2], _RUNTIME)
    resumed = _store(tmp_path)
    assert resumed.expected_count == 2
    assert resumed.restored_generation_count == 1
    assert resumed.restored_probe_count == 0
    assert resumed.pending_count == 2
    actual = _generate_rows(candidates, _generator([]), store=resumed)
    assert actual == reference
    assert _store(tmp_path).restored_generation_count == 2


def _probe_fn(calls: list[str]) -> Any:
    def _probe(**kwargs: Any) -> RoundtripFeatureScores:
        calls.append(str(kwargs["response_text"]))
        return RoundtripFeatureScores(
            n_features=1,
            pre_act_mean=1.0,
            post_act_mean=2.0,
            percentile_mean=3.0,
        )

    return _probe


def _throwing_probe(**_kwargs: Any) -> NoReturn:
    pytest.fail("restored probe row reached probe_text_fn")


def _score_generation(
    generation: Sequence[Mapping[str, Any]],
    *,
    store: RoundtripCheckpointStore | None = None,
    probe_fn: Any | None = None,
) -> pa.Table:
    return score_roundtrip_generation_rows(
        generation_rows=generation,
        probe_model=object(),
        percentile_table=None,
        feature_chunk_size=1,
        probe_text_fn=_probe_fn([]) if probe_fn is None else probe_fn,
        checkpoint_store=store,
        runtime_fingerprint=None if store is None else _RUNTIME,
    )


def test_probe_restore_skips_probe_and_partial_resume_preserves_table(tmp_path: Path) -> None:
    generation = _generate_rows(_candidates(("c0", "c1")), _generator([]))
    store = _store(tmp_path)
    store.commit_generation_rows("c0", generation[:2], _RUNTIME)
    store.commit_generation_rows("c1", generation[2:], _RUNTIME)
    calls: list[str] = []
    reference = score_roundtrip_generation_rows(
        generation_rows=generation,
        probe_model=object(),
        percentile_table=None,
        feature_chunk_size=1,
        probe_text_fn=_probe_fn(calls),
    )
    store.commit_probe_rows("c0", reference.slice(0, 2).to_pylist(), _RUNTIME)
    resumed = _store(tmp_path)
    fresh_calls: list[str] = []
    actual = score_roundtrip_generation_rows(
        generation_rows=generation,
        probe_model=object(),
        percentile_table=None,
        feature_chunk_size=1,
        probe_text_fn=_probe_fn(fresh_calls),
        checkpoint_store=resumed,
        runtime_fingerprint=_RUNTIME,
    )
    assert actual.equals(reference)
    assert fresh_calls == []
    all_restored = _store(tmp_path)
    replay = score_roundtrip_generation_rows(
        generation_rows=generation,
        probe_model=object(),
        percentile_table=None,
        feature_chunk_size=1,
        probe_text_fn=_throwing_probe,
        checkpoint_store=all_restored,
        runtime_fingerprint=_RUNTIME,
    )
    assert _parquet_bytes(replay) == _parquet_bytes(reference)


@pytest.mark.parametrize("window", ["after_generation_k", "after_generation_all_probe_j"])
def test_stage_level_resume_windows_preserve_final_parquet_bytes(
    tmp_path: Path, window: str
) -> None:
    candidates = _candidates(("c0", "c1"))
    reference_generation = _generate_rows(candidates, _generator([]))
    reference = _score_generation(reference_generation)
    partial = _store(tmp_path)
    if window == "after_generation_k":
        partial.commit_generation_rows("c0", reference_generation[:2], _RUNTIME)
    else:
        _generate_rows(candidates, _generator([]), store=partial)
        partial.commit_probe_rows("c0", reference.slice(0, 2).to_pylist(), _RUNTIME)
    resumed = _store(tmp_path)
    resumed_generation = _generate_rows(candidates, _generator([]), store=resumed)
    actual = _score_generation(resumed_generation, store=resumed)
    assert _parquet_bytes(actual) == _parquet_bytes(reference)


@pytest.mark.parametrize("candidate_ids", [("c0", "c1"), ("c1", "c0")])
def test_restored_fresh_twin_orderings_preserve_final_parquet_bytes(
    tmp_path: Path, candidate_ids: tuple[str, str]
) -> None:
    candidates = _candidates(candidate_ids)
    reference_generation = _generate_rows(candidates, _generator([]))
    reference = _score_generation(reference_generation)
    partial = _store(tmp_path, ids=candidate_ids)
    partial.commit_generation_rows(candidate_ids[0], reference_generation[:2], _RUNTIME)
    partial.commit_probe_rows(candidate_ids[0], reference.slice(0, 2).to_pylist(), _RUNTIME)
    resumed = _store(tmp_path, ids=candidate_ids)
    calls: list[str] = []
    resumed_generation = _generate_rows(candidates, _generator(calls), store=resumed)
    actual = _score_generation(resumed_generation, store=resumed, probe_fn=_probe_fn(calls))
    assert calls == []
    assert _parquet_bytes(actual) == _parquet_bytes(reference)


def test_empty_completion_crossing_restore_boundary_preserves_final_bytes(
    tmp_path: Path,
) -> None:
    candidates = _candidates(("c0", "c1"))

    def _empty(**kwargs: Any) -> list[RoundtripGeneratedCompletion]:
        return [
            RoundtripGeneratedCompletion(text="", token_count=0, stop_reason="bos_token")
            for _ in kwargs["sample_seeds"]
        ]

    reference_generation = _generate_rows(candidates, _empty)
    reference = _score_generation(reference_generation)
    partial = _store(tmp_path)
    partial.commit_generation_rows("c0", reference_generation[:2], _RUNTIME)
    resumed = _store(tmp_path)
    resumed_generation = _generate_rows(candidates, _empty, store=resumed)
    actual = _score_generation(resumed_generation, store=resumed)
    assert actual.num_rows == 0
    assert _parquet_bytes(actual) == _parquet_bytes(reference)


def _sigkill_child(root_str: str) -> None:
    root = Path(root_str)
    candidates = _candidates(("c0", "c1"))
    store = _store(root)
    original_commit = store.commit_generation_rows

    def _commit_then_block(
        candidate_id: str,
        rows: Sequence[Mapping[str, Any]],
        runtime_fingerprint: str,
    ) -> bool:
        committed = original_commit(candidate_id, rows, runtime_fingerprint)
        if candidate_id == "c0":
            (root / "committed").write_text("yes", encoding="utf-8")
            while True:
                time.sleep(0.05)
        return committed

    store.commit_generation_rows = _commit_then_block  # ty: ignore[invalid-assignment]
    _generate_rows(candidates, _generator([]), store=store)


def test_sigkill_inside_generation_run_resumes_to_final_table_byte_parity(
    tmp_path: Path,
) -> None:
    resumed_root = tmp_path / "resumed"
    context = multiprocessing.get_context("spawn")
    interrupted = context.Process(target=_sigkill_child, args=(str(resumed_root),))
    interrupted.start()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline and not (resumed_root / "committed").exists():
        if interrupted.exitcode is not None:
            break
        time.sleep(0.05)
    assert (resumed_root / "committed").exists()
    assert interrupted.pid is not None
    os.kill(interrupted.pid, signal.SIGKILL)
    interrupted.join(timeout=10)
    assert interrupted.exitcode == -signal.SIGKILL

    restored = _store(resumed_root)
    assert restored.restored_generation_count == 1
    candidates = _candidates(("c0", "c1"))
    reference_generation = _generate_rows(candidates, _generator([]))
    reference = _score_generation(reference_generation)
    resumed_generation = _generate_rows(candidates, _generator([]), store=restored)
    actual = _score_generation(resumed_generation, store=restored)
    assert _parquet_bytes(actual) == _parquet_bytes(reference)


@pytest.mark.parametrize("clustered", [False, True])
def test_stage_prepares_and_logs_checkpoint_before_target_load(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    clustered: bool,
) -> None:
    events: list[str] = []
    messages: list[str] = []
    runtime_devices: list[str] = []
    candidates = _candidates(("c0",))
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag="run",
        device="cpu",
        target_models=["decoder_only"],
        feature_dir=str(tmp_path / "features"),
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
    )
    cfg.roundtrip_steering = replace(
        cfg.roundtrip_steering,
        probe=replace(cfg.roundtrip_steering.probe, device="mps"),
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="run")
    contract = SimpleNamespace(feature_contract_fingerprint="f" * 64) if clustered else None
    stage_source = (
        SimpleNamespace(
            validation=SimpleNamespace(
                concept_entries=lambda: [vars(next(iter(_concepts().values())))]
            ),
            source_identity=object(),
        )
        if clustered
        else None
    )
    snapshot = SimpleNamespace(payload=b"candidate", sha256="a" * 64) if clustered else None
    store = SimpleNamespace(
        expected_count=1,
        restored_generation_count=1,
        restored_probe_count=0,
        pending_count=1,
    )

    monkeypatch.setattr(
        roundtrip_scoring,
        "require_clustered_stage_source",
        lambda **_kwargs: stage_source,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_clustered_candidate_snapshot",
        lambda _snapshot, **_kwargs: snapshot,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "read_evaluation_population",
        lambda _path, **_kwargs: object(),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: table,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_precomputed_bag_candidates_cover_scored_prefixes",
        lambda table, **_kwargs: table,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "resolve_and_pin_target_checkpoints",
        lambda _cfg, _targets: {},
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "parse_feature_dir",
        lambda _path: SimpleNamespace(segment_size=256),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "candidates_payload_sha256",
        lambda **_kwargs: "a" * 64,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "build_roundtrip_checkpoint_fingerprint",
        lambda **_kwargs: {"identity": "test"},
    )
    monkeypatch.setattr(
        roundtrip_scoring.RoundtripCheckpointStore,
        "prepare",
        lambda **_kwargs: events.append("prepare") or store,
    )

    class _Runtime:
        def __init__(self, **_kwargs: Any) -> None:
            pass

        def target_device(self) -> str:
            return "cpu"

        def decoder_only(self) -> Any:
            events.append("load_target")
            return SimpleNamespace(model=object(), tokenizer=_Tokenizer(), max_seq_len=128)

        def offload_models(self) -> None:
            events.append("offload")

    monkeypatch.setattr(roundtrip_scoring, "RuntimeState", _Runtime)
    monkeypatch.setattr(
        roundtrip_scoring,
        "current_runtime_fingerprint",
        lambda device: runtime_devices.append(str(device)) or ({"runtime": str(device)}, _RUNTIME),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "warn_on_roundtrip_runtime_mismatch",
        lambda _store, _family, _runtime: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring.logger,
        "info",
        lambda message, *_args: messages.append(message),
    )
    monkeypatch.setattr(roundtrip_scoring, "record_counter", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(roundtrip_scoring, "_read_concepts", lambda _paths: _concepts())
    monkeypatch.setattr(
        roundtrip_scoring,
        "_concepts_from_entries",
        lambda _entries: _concepts(),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_validate_roundtrip_probe_matches_concepts",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "assert_target_checkpoints_unchanged",
        lambda _targets: None,
    )
    monkeypatch.setattr(roundtrip_scoring, "_build_token_bytes", lambda _tokenizer: {})
    monkeypatch.setattr(
        roundtrip_scoring,
        "get_extraction_tokenizer",
        lambda _model_id: _Tokenizer(),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "detect_target_model_metadata",
        lambda *_args, **_kwargs: SimpleNamespace(segment_size=256),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_preflight_direction_scales_for_rows",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "ModelBackedDualCompletionScorer",
        lambda **_kwargs: SimpleNamespace(concepts_by_id=_concepts()),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "build_bpb_adv_score_table",
        lambda **_kwargs: pa.table(
            {
                "candidate_id": ["c0"],
                "loss_concept_rich": [1.0],
                "loss_concept_avoiding": [2.0],
            }
        ),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "build_roundtrip_generation_rows",
        lambda **_kwargs: [{"candidate_id": "c0"}],
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "_validate_roundtrip_stats_match_probe",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "load_percentile_table_for_roundtrip",
        lambda _path: (None, "checksum"),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "load_probe_model_for_roundtrip",
        lambda _probe: object(),
    )
    expected = pa.Table.from_pylist([], schema=ROUNDTRIP_SCORE_SCHEMA)
    monkeypatch.setattr(
        roundtrip_scoring,
        "score_roundtrip_generation_rows",
        lambda **_kwargs: expected,
    )
    stats: dict[str, int] = {}

    actual = run_model_backed_roundtrip_score_stage(
        cfg=cfg,
        paths=paths,
        candidates=candidates,
        stats=stats,
        cluster_contract=contract,  # ty: ignore[invalid-argument-type]
        stage_source=stage_source,  # ty: ignore[invalid-argument-type]
        candidate_snapshot=snapshot,  # ty: ignore[invalid-argument-type]
    )

    assert actual is expected
    assert events.index("prepare") < events.index("load_target")
    assert runtime_devices == ["cpu", "mps"]
    assert any("roundtrip checkpoint summary:" in message for message in messages)
    assert stats["checkpoint_expected_candidates"] == 1
    assert stats["checkpoint_restored_generation_candidates"] == 1
    assert stats["checkpoint_restored_probe_candidates"] == 0
    assert stats["checkpoint_pending_candidates"] == 1


def test_probe_base_model_pin_resolves_bare_ids_to_hf_repo_ids(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A bare gemma-scope-2 probe id must pin the SAME org-prefixed HF cache
    snapshot the loader consumes (models--google--...), not a nonexistent
    bare-id cache entry — the first live run failed exactly there."""

    sae_snapshot = tmp_path / "models--sae" / "snapshots" / "sae-revision"
    model_snapshot = tmp_path / "models--google--gemma-3-4b-pt" / "snapshots" / "model-revision"
    sae_weight = sae_snapshot / "layer_22/width_65k/l0_medium/params.safetensors"
    model_config = model_snapshot / "config.json"
    sae_weight.parent.mkdir(parents=True)
    model_config.parent.mkdir(parents=True)
    sae_weight.write_bytes(b"sae-v2")
    model_config.write_bytes(b"config-v2")

    requested_repo_ids: list[str] = []

    def _cached(repo_id: str, filename: str, *, label: str) -> Path:
        del label
        requested_repo_ids.append(repo_id)
        return model_config if filename == "config.json" else sae_weight

    monkeypatch.setattr(roundtrip_checkpoint, "_cached_hf_file", _cached)
    probe = RoundtripProbeConfig(
        model_name="gemma-3-4b-pt",
        sae_release="gemmascope_2",
        activation_type="residual",
        layer=22,
        sae_width="65k",
        l0=L0Tier.medium,
    )
    identity = roundtrip_checkpoint.resolve_probe_artifact_identity(probe)

    assert "google/gemma-3-4b-pt" in requested_repo_ids
    assert identity["base_model"]["repo_id"] == "google/gemma-3-4b-pt"
    assert identity["base_model"]["configured_model_name"] == "gemma-3-4b-pt"

    prefixed = roundtrip_checkpoint.resolve_probe_artifact_identity(
        replace(probe, model_name="google/gemma-3-4b-pt")
    )
    assert prefixed["base_model"]["repo_id"] == "google/gemma-3-4b-pt"


def test_cached_hf_file_accepts_only_an_existing_regular_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cached = tmp_path / "tokenizer.json"
    cached.write_bytes(b"tokenizer")
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "try_to_load_from_cache",
        lambda repo_id, filename: str(cached),
    )
    assert (
        roundtrip_checkpoint._cached_hf_file(
            "org/repo", "tokenizer.json", label="extraction tokenizer"
        )
        == cached
    )

    cached.unlink()
    with pytest.raises(AxBenchBpbAdvError, match="does not exist"):
        roundtrip_checkpoint._cached_hf_file(
            "org/repo", "tokenizer.json", label="extraction tokenizer"
        )


@pytest.mark.parametrize(
    "cached",
    [None, _CACHED_NO_EXIST],
    ids=["not-cached", "known-missing-sentinel"],
)
def test_cached_hf_file_rejects_both_hf_missing_results(
    monkeypatch: pytest.MonkeyPatch, cached: object
) -> None:
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "try_to_load_from_cache",
        lambda repo_id, filename: cached,
    )
    with pytest.raises(
        AxBenchBpbAdvError,
        match="not present in the local Hugging Face cache.*will not fetch",
    ):
        roundtrip_checkpoint._cached_hf_file(
            "org/repo", "tokenizer.json", label="extraction tokenizer"
        )


def test_cached_hf_file_wraps_cache_lookup_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(_repo_id: str, _filename: str) -> NoReturn:
        raise RuntimeError("cache exploded")

    monkeypatch.setattr(roundtrip_checkpoint, "try_to_load_from_cache", _fail)
    with pytest.raises(AxBenchBpbAdvError, match="could not resolve.*cache exploded"):
        roundtrip_checkpoint._cached_hf_file(
            "org/repo", "tokenizer.json", label="extraction tokenizer"
        )


def test_snapshot_identity_rejects_missing_snapshot_and_non_snapshot_file(tmp_path: Path) -> None:
    missing = tmp_path / "models--repo" / "snapshots" / "revision" / "tokenizer.json"
    with pytest.raises(AxBenchBpbAdvError, match="snapshot directory.*does not exist"):
        roundtrip_checkpoint._snapshot_identity_from_file(missing, label="extraction tokenizer")

    outside = tmp_path / "tokenizer.json"
    outside.write_bytes(b"tokenizer")
    with pytest.raises(AxBenchBpbAdvError, match="not under snapshots/<revision>"):
        roundtrip_checkpoint._snapshot_identity_from_file(outside, label="extraction tokenizer")


@pytest.mark.parametrize("commit", [None, "revision"], ids=["no-loader-commit", "matching-commit"])
def test_extraction_tokenizer_hf_snapshot_identity_is_content_pinned(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    commit: str | None,
) -> None:
    snapshot = tmp_path / "models--org--tokenizer" / "snapshots" / "revision"
    snapshot.mkdir(parents=True)
    files = {
        "tokenizer_config.json": b"config",
        "tokenizer.json": b"tokenizer",
        "vocab.model": b"vocab",
    }
    for filename, payload in files.items():
        (snapshot / filename).write_bytes(payload)
    tokenizer = SimpleNamespace(
        init_kwargs={"_commit_hash": commit},
        vocab_files_names={"vocab_file": "vocab.model", "unused": None},
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: tokenizer,
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_cached_hf_file",
        lambda *_args, **_kwargs: snapshot / "tokenizer_config.json",
    )

    identity = roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity("org/tokenizer")
    descriptors = [
        {"path": filename, "sha256": hashlib.sha256(payload).hexdigest()}
        for filename, payload in sorted(files.items())
    ]
    assert identity == {
        "tokenizer_id": "org/tokenizer",
        "source": "hf_snapshot",
        "snapshot_revision": "revision",
        "resolved_tokenizer_files_sha256": canonical_sha256(descriptors),
    }


def test_extraction_tokenizer_rejects_local_file_and_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    local_file = tmp_path / "tokenizer.json"
    local_file.write_bytes(b"tokenizer")
    with pytest.raises(AxBenchBpbAdvError, match="must be a real directory"):
        roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity(str(local_file))

    target = tmp_path / "real-tokenizer"
    target.mkdir()
    symlink = tmp_path / "tokenizer-link"
    symlink.symlink_to(target, target_is_directory=True)
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "get_extraction_tokenizer",
        lambda _tokenizer_id: pytest.fail("invalid local path reached loader"),
    )
    with pytest.raises(AxBenchBpbAdvError, match="must be a real directory"):
        roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity(str(symlink))


def test_extraction_tokenizer_rejects_commit_snapshot_disagreement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    snapshot = tmp_path / "models--repo" / "snapshots" / "snapshot-rev"
    snapshot.mkdir(parents=True)
    config = snapshot / "tokenizer_config.json"
    config.write_bytes(b"config")
    tokenizer = SimpleNamespace(init_kwargs={"_commit_hash": "loader-rev"}, vocab_files_names={})
    monkeypatch.setattr(
        roundtrip_checkpoint, "get_extraction_tokenizer", lambda _tokenizer_id: tokenizer
    )
    monkeypatch.setattr(roundtrip_checkpoint, "_cached_hf_file", lambda *_args, **_kwargs: config)
    with pytest.raises(AxBenchBpbAdvError, match="resolved commit differs.*loader-rev"):
        roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity("org/tokenizer")


def test_extraction_tokenizer_rejects_non_snapshot_cache_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = tmp_path / "tokenizer_config.json"
    config.write_bytes(b"config")
    tokenizer = SimpleNamespace(init_kwargs={}, vocab_files_names={})
    monkeypatch.setattr(
        roundtrip_checkpoint, "get_extraction_tokenizer", lambda _tokenizer_id: tokenizer
    )
    monkeypatch.setattr(roundtrip_checkpoint, "_cached_hf_file", lambda *_args, **_kwargs: config)
    with pytest.raises(AxBenchBpbAdvError, match="not under snapshots/<revision>"):
        roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity("org/tokenizer")


def test_extraction_tokenizer_rejects_snapshot_without_resolvable_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    empty_snapshot = tmp_path / "empty-snapshot"
    empty_snapshot.mkdir()
    tokenizer = SimpleNamespace(init_kwargs={}, vocab_files_names={"vocab": "missing.model"})
    monkeypatch.setattr(
        roundtrip_checkpoint, "get_extraction_tokenizer", lambda _tokenizer_id: tokenizer
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_cached_hf_file",
        lambda *_args, **_kwargs: tmp_path / "cached-tokenizer-config.json",
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_snapshot_identity_from_file",
        lambda *_args, **_kwargs: (empty_snapshot, "revision"),
    )
    with pytest.raises(AxBenchBpbAdvError, match="no tokenizer files were resolvable"):
        roundtrip_checkpoint.resolve_extraction_tokenizer_content_identity("org/tokenizer")


def test_candidates_payload_requires_snapshot_pair_and_valid_digest(tmp_path: Path) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    table = _candidates(("c0",))
    with pytest.raises(AxBenchBpbAdvError, match="payload and sha256 must be supplied together"):
        candidates_payload_sha256(candidates=table, paths=paths, snapshot_payload=b"payload")
    with pytest.raises(AxBenchBpbAdvError, match="payload digest differs"):
        candidates_payload_sha256(
            candidates=table,
            paths=paths,
            snapshot_payload=b"payload",
            snapshot_sha256="0" * 64,
        )


def test_candidates_payload_rejects_unreadable_invalid_and_mismatched_payloads(
    tmp_path: Path,
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    table = _candidates(("c0",))
    with pytest.raises(AxBenchBpbAdvError, match="could not stably read candidates.parquet"):
        candidates_payload_sha256(candidates=table, paths=paths)

    invalid = b"not parquet"
    with pytest.raises(AxBenchBpbAdvError, match="payload is not valid Parquet"):
        candidates_payload_sha256(
            candidates=table,
            paths=paths,
            snapshot_payload=invalid,
            snapshot_sha256=hashlib.sha256(invalid).hexdigest(),
        )

    other = _candidates(("other",))
    payload = _parquet_bytes(other)
    with pytest.raises(AxBenchBpbAdvError, match="exact consumed candidate table"):
        candidates_payload_sha256(
            candidates=table,
            paths=paths,
            snapshot_payload=payload,
            snapshot_sha256=hashlib.sha256(payload).hexdigest(),
        )


def test_candidates_payload_rejects_missing_retained_payload(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "stream_hash_file_descriptor_bound",
        lambda *_args, **_kwargs: SimpleNamespace(payload=None, sha256="0" * 64),
    )
    with pytest.raises(AxBenchBpbAdvError, match="read retained no payload"):
        candidates_payload_sha256(candidates=_candidates(("c0",)), paths=paths)


def test_fingerprint_unused_and_active_conditional_content_fences_differ(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "_hash_file",
        lambda path, **_kwargs: canonical_sha256({"file": str(path)}),
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "tree_sha256",
        lambda path: canonical_sha256({"tree": str(path)}),
    )
    monkeypatch.setattr(roundtrip_checkpoint, "get_base_dir", lambda: str(tmp_path / "base"))
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_extraction_tokenizer_content_identity",
        lambda _tokenizer_id: {"tokenizer": "pinned"},
    )
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "resolve_probe_artifact_identity",
        lambda _probe: {"probe": "pinned"},
    )
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path),
        run_tag="run",
        device="cpu",
        feature_statistics_path=str(tmp_path / "stats.safetensors"),
    )
    target = ResolvedTargetCheckpoint(
        target_name="decoder_only",
        source="base",
        model_tag="d12",
        step=1,
        checkpoint_dir=tmp_path,
        model_path=tmp_path / "model.pt",
        metadata_path=tmp_path / "meta.json",
        model_sha256="1" * 64,
        metadata_sha256="2" * 64,
    )

    def _build(
        candidate: Mapping[str, Any], cluster_fingerprint: str | None = None
    ) -> dict[str, Any]:
        return build_roundtrip_checkpoint_fingerprint(
            cfg=cfg,
            paths=paths,
            candidates_sha256=canonical_sha256(candidate),
            candidate_rows=[candidate],
            cluster_feature_contract_fingerprint=cluster_fingerprint,
            resolved_targets={str(candidate["target_model"]): target},
            reference_segment_size_by_target={str(candidate["target_model"]): 256},
            probe_statistics_path=tmp_path / "probe-stats.safetensors",
        )

    unused = _build(_candidate("c0", "row-0"))
    assert unused["generation_semantics"]["context_cache_tree_sha256"] == "unused"
    assert unused["generation_semantics"]["unclustered_precomputed_bags_tree_sha256"] == "unused"

    active_candidate = _candidate("c0", "row-0")
    active_candidate["target_model"] = "encdec"
    active_candidate["method_name"] = "ours_gcg_encdec"
    active = _build(active_candidate)
    assert active["generation_semantics"]["context_cache_tree_sha256"] != "unused"
    assert active["generation_semantics"]["unclustered_precomputed_bags_tree_sha256"] != "unused"
    assert active != unused

    clustered = _build(active_candidate, "f" * 64)
    assert (
        clustered["generation_semantics"]["unclustered_precomputed_bags_tree_sha256"]
        == "covered_by_clustered_candidate_metadata"
    )
    assert clustered != active


@pytest.mark.parametrize(
    ("expected", "observed", "difference"),
    [
        ({"a": 1}, [1], ("root", {"a": 1}, [1])),
        ({"a": 1}, {"b": 1}, ("root.keys", ["a"], ["b"])),
        ({"a": [1]}, {"a": [1, 2]}, ("root.a.length", 1, 2)),
        ({"a": [{"b": 1}]}, {"a": [{"c": 1}]}, ("root.a[0].keys", ["b"], ["c"])),
        ({"a": [1]}, {"a": [2]}, ("root.a[0]", 1, 2)),
    ],
)
def test_first_difference_names_nested_shape_and_leaf_changes(
    expected: Any, observed: Any, difference: tuple[str, Any, Any]
) -> None:
    assert roundtrip_checkpoint._first_difference(expected, observed, "root") == difference


def test_first_difference_returns_none_for_equal_nested_payload() -> None:
    payload = {"a": [{"b": [1, 2]}]}
    assert roundtrip_checkpoint._first_difference(payload, deepcopy(payload)) is None


def test_bitwise_value_comparison_covers_nan_signed_zero_nested_and_passthrough() -> None:
    nan = float("nan")
    assert roundtrip_checkpoint._float_bits_equal(nan, nan)
    assert not roundtrip_checkpoint._float_bits_equal(0.0, -0.0)
    assert roundtrip_checkpoint._values_bits_equal(
        {"nested": [nan, {"value": 1}]},
        {"nested": [nan, {"value": 1}]},
    )
    assert not roundtrip_checkpoint._values_bits_equal([0.0], [-0.0])
    assert roundtrip_checkpoint._values_bits_equal("same", "same")
    assert not roundtrip_checkpoint._values_bits_equal(1, True)
    assert roundtrip_checkpoint.rows_bits_equal([{"value": nan}], [{"value": nan}])
    assert not roundtrip_checkpoint.rows_bits_equal([{"value": 1}], [])


def test_installer_temp_matcher_accepts_regular_files_and_rejects_lookalikes(
    tmp_path: Path,
) -> None:
    manifest_temp = tmp_path / ".manifest.json.tmp.123.abcdef12"
    manifest_temp.write_bytes(b"temp")
    shard_temp = tmp_path / f".cand_{'1' * 64}_{'2' * 64}.parquet.tmp.9.1234abcd"
    shard_temp.write_bytes(b"temp")
    assert roundtrip_checkpoint._is_known_installer_temp(manifest_temp)
    assert roundtrip_checkpoint._is_known_installer_temp(shard_temp)
    assert not roundtrip_checkpoint._is_known_installer_temp(tmp_path / "manifest.json.tmp")


@pytest.mark.parametrize("entry_kind", ["directory", "symlink"])
def test_installer_temp_matcher_rejects_non_regular_entries(
    tmp_path: Path, entry_kind: str
) -> None:
    entry = tmp_path / ".manifest.json.tmp.123.abcdef12"
    if entry_kind == "directory":
        entry.mkdir()
    else:
        target = tmp_path / "target"
        target.write_bytes(b"target")
        entry.symlink_to(target)
    with pytest.raises(AxBenchBpbAdvError, match="temporary entry is not a real regular file"):
        roundtrip_checkpoint._is_known_installer_temp(entry)


@pytest.mark.parametrize("samples_per_candidate", [True, 0, -1])
def test_store_rejects_invalid_samples_per_candidate(
    tmp_path: Path, samples_per_candidate: Any
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="must be a positive integer"):
        RoundtripCheckpointStore(
            checkpoint_dir=tmp_path / "checkpoint",
            fingerprint=_fingerprint(),
            expected_candidate_rows=[_candidate("c0", "row-0")],
            samples_per_candidate=samples_per_candidate,
        )


def test_store_rejects_duplicate_candidate_ids_and_hash_collisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="candidate_id values are not unique"):
        RoundtripCheckpointStore(
            checkpoint_dir=tmp_path / "duplicates",
            fingerprint=_fingerprint(),
            expected_candidate_rows=[
                _candidate("c0", "row-0"),
                _candidate("c0", "row-1"),
            ],
            samples_per_candidate=2,
        )

    monkeypatch.setattr(
        roundtrip_checkpoint,
        "hashlib",
        SimpleNamespace(sha256=lambda _payload: SimpleNamespace(hexdigest=lambda: "a" * 64)),
    )
    with pytest.raises(AxBenchBpbAdvError, match="SHA-256 collision between candidate ids"):
        RoundtripCheckpointStore(
            checkpoint_dir=tmp_path / "collision",
            fingerprint=_fingerprint(),
            expected_candidate_rows=[
                _candidate("c0", "row-0"),
                _candidate("c1", "row-1"),
            ],
            samples_per_candidate=2,
        )


@pytest.mark.parametrize("root_kind", ["file", "symlink"])
def test_prepare_rejects_non_directory_checkpoint_root(tmp_path: Path, root_kind: str) -> None:
    checkpoint = tmp_path / "roundtrip_checkpoint"
    if root_kind == "file":
        checkpoint.write_bytes(b"not a directory")
    else:
        target = tmp_path / "target"
        target.mkdir()
        checkpoint.symlink_to(target, target_is_directory=True)
    with pytest.raises(AxBenchBpbAdvError, match="root is not a real directory"):
        _store(tmp_path, ids=("c0",))


def test_prepare_rejects_committed_state_without_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "roundtrip_checkpoint"
    checkpoint.mkdir()
    (checkpoint / "committed").write_bytes(b"state")
    with pytest.raises(AxBenchBpbAdvError, match="committed state but no manifest"):
        _store(tmp_path, ids=("c0",))


def test_prepare_installs_manifest_around_only_known_installer_temp(tmp_path: Path) -> None:
    checkpoint = tmp_path / "roundtrip_checkpoint"
    checkpoint.mkdir()
    temp = checkpoint / ".manifest.json.tmp.123.abcdef12"
    temp.write_bytes(b"stale temp")
    store = _store(tmp_path, ids=("c0",))
    assert store.manifest_path.is_file()
    assert temp.is_file()
    assert _store(tmp_path, ids=("c0",)).fingerprint == store.fingerprint


def _rewrite_roundtrip_manifest(store: RoundtripCheckpointStore, payload: bytes) -> None:
    store.manifest_path.write_bytes(payload)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("invalid_utf8", "manifest is malformed JSON"),
        ("invalid_json", "manifest is malformed JSON"),
        ("non_object", "manifest root must be an object"),
        ("missing_schema", "manifest keys differ"),
        ("schema_string", "schema_version has invalid type/value"),
        ("schema_wrong", "schema_version differs"),
        ("store_kind", "store_kind differs"),
        ("fingerprint_sha_type", "fingerprint_sha256 is invalid"),
        ("fingerprint_sha_regex", "fingerprint_sha256 is invalid"),
        ("fingerprint_non_object", "fingerprint must be an object"),
        ("fingerprint_digest", "differs from canonical fingerprint"),
        ("fingerprint_field", "fingerprint.candidates_sha256 differs"),
        ("noncanonical", "payload is not canonical JSON"),
    ],
)
def test_manifest_structural_guards_fail_loud(tmp_path: Path, mutation: str, error: str) -> None:
    store = _store(tmp_path, ids=("c0",))
    manifest = json.loads(store.manifest_path.read_text(encoding="utf-8"))
    if mutation == "invalid_utf8":
        payload = b"\xff"
    elif mutation == "invalid_json":
        payload = b"{"
    elif mutation == "non_object":
        payload = roundtrip_checkpoint.canonical_json_bytes(["manifest"])
    else:
        if mutation == "missing_schema":
            manifest.pop("schema_version")
        elif mutation == "schema_string":
            manifest["schema_version"] = "1"
        elif mutation == "schema_wrong":
            manifest["schema_version"] = 2
        elif mutation == "store_kind":
            manifest["store_kind"] = "other"
        elif mutation == "fingerprint_sha_type":
            manifest["fingerprint_sha256"] = True
        elif mutation == "fingerprint_sha_regex":
            manifest["fingerprint_sha256"] = "A" * 64
        elif mutation == "fingerprint_non_object":
            manifest["fingerprint"] = []
        elif mutation == "fingerprint_digest":
            manifest["fingerprint"]["candidates_sha256"] = "9" * 64
        elif mutation == "fingerprint_field":
            manifest["fingerprint"]["candidates_sha256"] = "9" * 64
            manifest["fingerprint_sha256"] = canonical_sha256(manifest["fingerprint"])
        elif mutation != "noncanonical":
            pytest.fail(mutation)
        payload = (
            json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")
            if mutation == "noncanonical"
            else roundtrip_checkpoint.canonical_json_bytes(manifest)
        )
    _rewrite_roundtrip_manifest(store, payload)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        _store(tmp_path, ids=("c0",))


@pytest.mark.parametrize(
    ("failure", "error"),
    [("read", "cannot read manifest"), ("payload", "manifest read retained no payload")],
)
def test_manifest_defensive_read_guards_fail_loud(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    error: str,
) -> None:
    store = _store(tmp_path, ids=("c0",))
    expected = store.manifest_path.read_bytes()
    if failure == "read":

        def _read_fail(*_args: Any, **_kwargs: Any) -> NoReturn:
            raise OSError("cannot read manifest")

        replacement = _read_fail
    else:

        def _missing_payload(*_args: Any, **_kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(payload=None)

        replacement = _missing_payload
    monkeypatch.setattr(
        roundtrip_checkpoint,
        "stream_hash_file_descriptor_bound",
        replacement,
    )
    with pytest.raises(AxBenchBpbAdvError, match=error):
        store._validate_manifest(expected)


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("manifest_directory", "manifest is not a real regular file"),
        ("manifest_symlink", "manifest is not a real regular file"),
        ("family_file", "generation entry is not a real directory"),
        ("family_symlink", "generation entry is not a real directory"),
        ("unknown_root", "unknown checkpoint entry"),
        ("temp_directory", "temporary entry is not a real regular file"),
        ("unexpected_family_entry", "unexpected generation family entry"),
        ("groups_file", "unexpected generation family entry"),
        ("invalid_bucket", "unexpected checkpoint bucket entry"),
        ("bucket_file", "unexpected checkpoint bucket entry"),
        ("bucket_symlink", "unexpected checkpoint bucket entry"),
        ("invalid_shard_name", "unknown committed checkpoint filename"),
    ],
)
def test_checkpoint_tree_rejects_unexpected_entry_at_every_level(
    tmp_path: Path, mutation: str, error: str
) -> None:
    store = _store(tmp_path, ids=("c0",))
    root = store.checkpoint_dir
    generation = root / CheckpointFamily.GENERATION
    groups = generation / "groups"
    if mutation == "manifest_directory":
        store.manifest_path.unlink()
        store.manifest_path.mkdir()
    elif mutation == "manifest_symlink":
        payload = store.manifest_path.read_bytes()
        target = tmp_path / "manifest-target"
        target.write_bytes(payload)
        store.manifest_path.unlink()
        store.manifest_path.symlink_to(target)
    elif mutation == "family_file":
        generation.write_bytes(b"family")
    elif mutation == "family_symlink":
        target = tmp_path / "family-target"
        target.mkdir()
        generation.symlink_to(target, target_is_directory=True)
    elif mutation == "unknown_root":
        (root / "unknown").write_bytes(b"unknown")
    elif mutation == "temp_directory":
        (root / ".manifest.json.tmp.123.abcdef12").mkdir()
    elif mutation == "unexpected_family_entry":
        generation.mkdir()
        (generation / "unexpected").write_bytes(b"unexpected")
    elif mutation == "groups_file":
        generation.mkdir()
        groups.write_bytes(b"groups")
    elif mutation in {"invalid_bucket", "bucket_file", "bucket_symlink"}:
        groups.mkdir(parents=True)
        bucket = groups / ("zz" if mutation == "invalid_bucket" else "aa")
        if mutation == "invalid_bucket":
            bucket.mkdir()
        elif mutation == "bucket_file":
            bucket.write_bytes(b"bucket")
        else:
            target = tmp_path / "bucket-target"
            target.mkdir()
            bucket.symlink_to(target, target_is_directory=True)
    elif mutation == "invalid_shard_name":
        bucket = groups / "aa"
        bucket.mkdir(parents=True)
        (bucket / "not-a-shard.parquet").write_bytes(b"payload")
    else:
        pytest.fail(mutation)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        store._scan_all_shards()


@pytest.mark.parametrize(
    ("surface", "error"),
    [
        ("prepare_root", "cannot enumerate checkpoint root"),
        ("scan_root", "cannot enumerate checkpoint root"),
        ("family", "cannot enumerate generation family"),
        ("groups", "cannot enumerate generation groups"),
        ("bucket", "cannot enumerate checkpoint bucket"),
    ],
)
def test_checkpoint_enumeration_failures_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    surface: str,
    error: str,
) -> None:
    if surface == "prepare_root":
        checkpoint = tmp_path / "roundtrip_checkpoint"
        checkpoint.mkdir()
        store = RoundtripCheckpointStore(
            checkpoint_dir=checkpoint,
            fingerprint=_fingerprint(),
            expected_candidate_rows=[_candidate("c0", "row-0")],
            samples_per_candidate=2,
        )
        target = checkpoint
    else:
        store = _store(tmp_path, ids=("c0",))
        generation = store.checkpoint_dir / CheckpointFamily.GENERATION
        groups = generation / "groups"
        if surface in {"family", "groups", "bucket"}:
            groups.mkdir(parents=True)
        if surface == "scan_root":
            target = store.checkpoint_dir
        elif surface == "family":
            target = generation
        elif surface == "groups":
            target = groups
        else:
            target = groups / "aa"
            target.mkdir()
    original_iterdir = Path.iterdir

    def _iterdir(path: Path):
        if path == target:
            raise OSError("enumeration exploded")
        return original_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", _iterdir)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        if surface == "prepare_root":
            store._prepare()
        elif surface == "scan_root":
            store._scan_all_shards()
        else:
            store._scan_family(CheckpointFamily.GENERATION)


def test_hash_file_wraps_descriptor_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "artifact"
    path.write_bytes(b"artifact")

    def _fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise OSError("descriptor exploded")

    monkeypatch.setattr(
        roundtrip_checkpoint,
        "stream_hash_file_descriptor_bound",
        _fail,
    )
    with pytest.raises(AxBenchBpbAdvError, match="could not stably hash artifact.*exploded"):
        roundtrip_checkpoint._hash_file(path, label="artifact")


def test_manifest_install_failure_is_normalized(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = RoundtripCheckpointStore(
        checkpoint_dir=tmp_path / "checkpoint",
        fingerprint=_fingerprint(),
        expected_candidate_rows=[_candidate("c0", "row-0")],
        samples_per_candidate=2,
    )

    def _fail(*_args: Any, **_kwargs: Any) -> NoReturn:
        raise OSError("manifest install exploded")

    monkeypatch.setattr(roundtrip_checkpoint, "install_bytes_once", _fail)
    with pytest.raises(AxBenchBpbAdvError, match="manifest install exploded"):
        store._prepare()


def test_commit_rejects_unknown_candidate_invalid_runtime_columns_and_arrow_values(
    tmp_path: Path,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    with pytest.raises(AxBenchBpbAdvError, match="cannot commit unknown candidate_id"):
        store.commit_generation_rows("unknown", rows, _RUNTIME)
    with pytest.raises(AxBenchBpbAdvError, match="runtime fingerprint is invalid"):
        store.commit_generation_rows("c0", rows, "A" * 64)

    missing_column = deepcopy(rows)
    missing_column[0].pop("generated_text")
    with pytest.raises(AxBenchBpbAdvError, match="generation row columns differ"):
        store.commit_generation_rows("c0", missing_column, _RUNTIME)

    wrong_type = deepcopy(rows)
    wrong_type[0]["sample_index"] = "not-an-integer"
    with pytest.raises(AxBenchBpbAdvError, match="rows do not conform to their schema"):
        store.commit_generation_rows("c0", wrong_type, _RUNTIME)


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        ("serialize", "could not serialize generation rows as Parquet"),
        ("install", "shard install exploded"),
        ("enumerate", "cannot enumerate checkpoint bucket"),
    ],
)
def test_commit_io_failures_are_normalized(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
    error: str,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    if failure == "serialize":

        def _write_fail(*_args: Any, **_kwargs: Any) -> NoReturn:
            raise pa.ArrowInvalid("serialization exploded")

        monkeypatch.setattr(roundtrip_checkpoint.pq, "write_table", _write_fail)
    elif failure == "install":

        def _install_fail(*_args: Any, **_kwargs: Any) -> NoReturn:
            raise OSError("shard install exploded")

        monkeypatch.setattr(roundtrip_checkpoint, "install_bytes_once", _install_fail)
    else:
        key_hash = hashlib.sha256(b"c0").hexdigest()
        bucket = store.checkpoint_dir / CheckpointFamily.GENERATION / "groups" / key_hash[:2]
        original_iterdir = Path.iterdir

        def _iterdir(path: Path):
            if path == bucket:
                raise OSError("bucket enumeration exploded")
            return original_iterdir(path)

        monkeypatch.setattr(Path, "iterdir", _iterdir)
    with pytest.raises(AxBenchBpbAdvError, match=error):
        store.commit_generation_rows("c0", rows, _RUNTIME)


def test_same_payload_concurrent_stores_converge_and_divergent_stores_fail(
    tmp_path: Path,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    first = _store(tmp_path, ids=("c0",))
    concurrent_same = _store(tmp_path, ids=("c0",))
    concurrent_divergent = _store(tmp_path, ids=("c0",))

    assert first.commit_generation_rows("c0", rows, _RUNTIME)
    assert not concurrent_same.commit_generation_rows("c0", rows, _RUNTIME)
    assert concurrent_same.load_generation_rows() == {"c0": rows}

    changed = deepcopy(rows)
    changed[0]["generated_text"] = "different"
    with pytest.raises(
        AxBenchBpbAdvError, match="expected exactly one committed generation payload"
    ):
        concurrent_divergent.commit_generation_rows("c0", changed, _RUNTIME)


def test_commit_post_install_rescan_rejects_unrelated_own_prefix(tmp_path: Path) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    key_hash = hashlib.sha256(b"c0").hexdigest()
    bucket = store.checkpoint_dir / CheckpointFamily.GENERATION / "groups" / key_hash[:2]
    bucket.mkdir(parents=True)
    (bucket / f"cand_{key_hash}_not-a-payload").write_bytes(b"junk")
    with pytest.raises(
        AxBenchBpbAdvError, match="expected exactly one committed generation payload"
    ):
        store.commit_generation_rows("c0", rows, _RUNTIME)


def test_runtime_warning_reports_families_independently_and_matching_hashes_are_quiet(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    store.commit_probe_rows("c0", _probe_rows(rows), "2" * 64)
    restored = _store(tmp_path, ids=("c0",))
    handler_id = roundtrip_checkpoint.logger.add(caplog.handler, format="{message}")
    try:
        warn_on_roundtrip_runtime_mismatch(restored, CheckpointFamily.GENERATION, _RUNTIME)
        warn_on_roundtrip_runtime_mismatch(restored, CheckpointFamily.PROBE, "2" * 64)
        assert not caplog.records

        warn_on_roundtrip_runtime_mismatch(restored, CheckpointFamily.GENERATION, "3" * 64)
        generation_messages = [record.message for record in caplog.records]
        assert generation_messages
        assert all("ROUNDTRIP GENERATION RESUME" in message for message in generation_messages)

        caplog.clear()
        warn_on_roundtrip_runtime_mismatch(restored, CheckpointFamily.PROBE, "3" * 64)
        probe_messages = [record.message for record in caplog.records]
        assert probe_messages
        assert all("ROUNDTRIP PROBE RESUME" in message for message in probe_messages)
    finally:
        roundtrip_checkpoint.logger.remove(handler_id)


@pytest.mark.parametrize(
    ("candidate_rows", "digest", "error"),
    [
        (
            [_candidate("c0", "row-0"), _candidate("c0", "row-1")],
            "a" * 64,
            "candidate_id values must be unique",
        ),
        (
            [_candidate("c0", "row-0")],
            "A" * 64,
            "candidates_sha256 must be a lowercase SHA-256 digest",
        ),
    ],
)
def test_fingerprint_rejects_duplicate_candidate_ids_and_invalid_candidate_digest(
    tmp_path: Path,
    candidate_rows: list[dict[str, Any]],
    digest: str,
    error: str,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path),
        run_tag="run",
        device="cpu",
    )
    paths = AxBenchBpbAdvPaths(output_root=tmp_path, run_tag="run")
    with pytest.raises(AxBenchBpbAdvError, match=error):
        build_roundtrip_checkpoint_fingerprint(
            cfg=cfg,
            paths=paths,
            candidates_sha256=digest,
            candidate_rows=candidate_rows,
            cluster_feature_contract_fingerprint=None,
            resolved_targets={},
            reference_segment_size_by_target={},
            probe_statistics_path=tmp_path / "probe-stats",
        )


def test_observed_runtime_hashes_unions_families_and_bucket_temps_are_ignored(
    tmp_path: Path,
) -> None:
    rows = _generate_rows(_candidates(("c0",)), _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    store.commit_probe_rows("c0", _probe_rows(rows), "2" * 64)
    generation_shard = next(
        (store.checkpoint_dir / CheckpointFamily.GENERATION / "groups").glob("*/*.parquet")
    )
    bucket_temp = generation_shard.parent / (f".{generation_shard.name}.tmp.123.abcdef12")
    bucket_temp.write_bytes(b"stale temp")

    restored = _store(tmp_path, ids=("c0",))
    assert restored.observed_runtime_hashes == frozenset({_RUNTIME, "2" * 64})
    assert restored.load_generation_rows() == {"c0": rows}


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("roundtrip_prompt_token_digest", "0" * 64, "restored generation prompt"),
        ("roundtrip_prompt_text", "wrong", "restored generation prompt"),
        ("control_roundtrip_prompt_tokens", [120], "restored generation prompt"),
        ("control_roundtrip_prompt_token_digest", "0" * 64, "restored generation prompt"),
        ("control_roundtrip_prompt_text", "wrong", "restored generation prompt"),
        ("generation_seed", 0, "restored generation_seed differs"),
        ("control_generation_seed", 0, "restored control_generation_seed differs"),
    ],
)
def test_restored_generation_cache_rejects_each_prompt_and_seed_mismatch(
    tmp_path: Path,
    field: str,
    value: Any,
    error: str,
) -> None:
    candidates = _candidates(("c0",))
    rows = _generate_rows(candidates, _generator([]))
    store = _store(tmp_path, ids=("c0",))
    store.commit_generation_rows("c0", rows, _RUNTIME)
    shard = next((store.checkpoint_dir / CheckpointFamily.GENERATION / "groups").glob("*/*.parquet"))
    table = pq.read_table(shard)
    tampered = table.to_pylist()
    tampered[0][field] = value
    _replace_shard_table(
        shard,
        pa.Table.from_pylist(tampered, schema=GENERATION_ROW_SCHEMA).replace_schema_metadata(
            table.schema.metadata
        ),
    )
    restored = _store(tmp_path, ids=("c0",))
    with pytest.raises(AxBenchBpbAdvError, match=error):
        _generate_rows(candidates, _throwing_generator, store=restored)


def test_restored_generation_cache_rejects_conflicting_semantic_twins(tmp_path: Path) -> None:
    candidates = _candidates(("c0", "c1"))
    rows = _generate_rows(candidates, _generator([]))
    store = _store(tmp_path)
    store.commit_generation_rows("c0", rows[:2], _RUNTIME)
    conflicting = deepcopy(rows[2:])
    conflicting[0]["generated_text"] = "conflicting completion"
    store.commit_generation_rows("c1", conflicting, _RUNTIME)

    restored = _store(tmp_path)
    with pytest.raises(
        AxBenchBpbAdvError, match="conflicts with another restored semantic cache entry"
    ):
        _generate_rows(candidates, _throwing_generator, store=restored)


@pytest.mark.parametrize(
    ("guard", "error"),
    [
        ("run_tag", "cfg.run_tag must be set before scoring"),
        ("empty_candidates", "no candidates to score"),
        ("disabled", "requires roundtrip_steering.enabled"),
        ("statistics", "requires feature_statistics_path"),
        ("unknown_target", "unsupported target_model values"),
    ],
)
def test_stage_integration_fail_loud_guards(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guard: str,
    error: str,
) -> None:
    cfg = AxBenchBpbAdvConfig(
        source_run_dir=str(tmp_path / "source"),
        output_root=str(tmp_path / "out"),
        run_tag=None if guard == "run_tag" else "run",
        device="cpu",
        feature_statistics_path=("" if guard == "statistics" else str(tmp_path / "stats")),
    )
    if guard == "disabled":
        cfg.roundtrip_steering = replace(cfg.roundtrip_steering, enabled=False)
    rows = [] if guard == "empty_candidates" else [_candidate("c0", "row-0")]
    if guard == "unknown_target":
        rows[0]["target_model"] = "unknown"
    candidates = pa.Table.from_pylist(rows, schema=CANDIDATES_SCHEMA)
    paths = AxBenchBpbAdvPaths(output_root=tmp_path / "out", run_tag="run")
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_clustered_stage_source",
        lambda **_kwargs: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_clustered_candidate_snapshot",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "read_evaluation_population",
        lambda *_args, **_kwargs: object(),
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_table_in_evaluation_population",
        lambda table, **_kwargs: table,
    )
    monkeypatch.setattr(
        roundtrip_scoring,
        "require_precomputed_bag_candidates_cover_scored_prefixes",
        lambda table, **_kwargs: table,
    )
    with pytest.raises(AxBenchBpbAdvError, match=error):
        run_model_backed_roundtrip_score_stage(
            cfg=cfg,
            paths=paths,
            candidates=candidates,
        )
