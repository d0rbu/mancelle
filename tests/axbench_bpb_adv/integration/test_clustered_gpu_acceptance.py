"""H3 real-checkpoint clustered Stage 0a -> BPB GPU acceptance.

This is a deliberately manual, very-slow acceptance. It binds two projected
representatives to the real l22/f65k clustered contract, materializes a small
deterministic Stage 0b-3 source around the real Stage 0a artifacts, and then
drives the production BPB dispatcher through all fourteen clustered methods.

Run the small checkpoint first, then the final l22 target, exactly as documented
in ``docs/evaluation/clustered-gpu-acceptance.md``.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
import yaml
from huggingface_hub import hf_hub_download
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import DictConfig
from sae_lens.loading import pretrained_sae_loaders
from safetensors import safe_open
from safetensors.torch import save_file

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
import steering.eval.axbench_bpb_adv.source_validation as source_validation
from nanochat.atomic_io import atomic_json_dump
from nanochat.tokenizer import get_tokenizer
from steering._common.strict_json import canonical_json
from steering.eval.axbench_base import paths as axbench_base_paths
from steering.eval.axbench_base.concept_list import prepare_concept_list
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.positive_selection_contract import (
    schema_with_default_positive_selection_contract,
)
from steering.eval.axbench_base.seed_contract import (
    SEED_SCHEMA_VERSION,
    SEED_TOKEN_SEMANTICS,
    compute_extraction_fingerprint,
    compute_seed_fingerprint,
    save_seed_pool,
    sha256_bytes,
    sha256_file,
)
from steering.eval.axbench_base.types import Concept, SeedRow
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import (
    ATTACK_ROWS_SCHEMA,
    CANDIDATES_SCHEMA,
    SCORE_SCHEMA,
)
from steering.eval.axbench_bpb_adv.source_validation import (
    validate_stage_0_3_source_run,
)
from steering.eval.cluster_contract_identity import stream_hash_file_stable
from steering.eval.cluster_feature_contract import (
    ClusterFeatureContract,
    ClusterFeatureContractError,
    ClusterFeatureContractRequest,
    ensure_feature_contract_run_marker,
    resolve_cluster_feature_contract,
)
from steering.segmentation import compute_tokenizer_checksum
from tests.axbench_bpb_adv.artifact_helpers import artifact_identity_metadata

_REPO = Path(__file__).resolve().parents[3]
_CONFIG_DIR = _REPO / "configs" / "eval" / "axbench_bpb_adv"
_FEATURE_DIR = (
    _REPO / "out/data/features/s128/l22/a_res/p_max_pool/f65k/"
    "m_gemma-3-4b-pt/base/context_independent/no_bos"
)
_STATISTICS = _FEATURE_DIR / "feature_statistics.safetensors"
_PROVENANCE = _FEATURE_DIR / "preselection_provenance/l22_clustered_splice_cf"
_MASK = _PROVENANCE / "manual_feature_filter_with_descriptions.yaml"
_CLUSTERS = _PROVENANCE / "feature_clusters_two_stage_schema_v1.yaml"
_SELECTION_MANIFEST = (
    _FEATURE_DIR / "splice/cut_frac0.15/seed42/pct/623c372d3f121b18/"
    "tok/dc6308e6116cbfe0/c1024/val_s2/"
    "sel/e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3/"
    "manifest.json"
)
_DESCRIPTION_CACHE_ROOT = _REPO / "out/eval/axbench_base_l22_f65k"
_EXPECTED_FEATURE_CONTRACTS = {
    (
        "i768_pr3_real_cpu_smoke_20260723",
        1,
    ): "3b91132728f46ebe06a3d5600c0a7f89a2017503063b978f9fb3069f0a16afda",
    (
        "l22_main_h100_8rank_db16_v1",
        54600,
    ): "979a087474e0a1b5d44ddb650be732d3754f7efc10d218d4f880113e465fa538",
}
_EXPECTED_CLUSTER_DESCRIPTOR = "5738db84ba00c8b7a84cd28e7a6b5ebbd4d8385bc81262cb63d904265ec95039"
_EXPECTED_CLUSTER_CONTENT = "8f4d1f3288aec0a59b7126ea5d300060ac1a88b41f365cfb85c780af7c15640b"
_EXPECTED_SELECTION_HASH = "e9e395739f1ac4c9498f0935e2e48a407ae9eb13b0134904d70f7ec64a3d5be3"
# Must equal the composed config's diffmean_and_probe_layer.
# Stage 1b positive-QC accept threshold stamped into the fixture's positive corpus
# (#1053). Consumers validate the contract's STRUCTURE, not this value against a
# config, so any well-formed threshold serves -- but it must be present.
_POSITIVE_QC_ACCEPT_THRESHOLD = 0.3
_DIFFMEAN_LAYER = 6
_SOURCE_IDS = (344, 1026)
_EXPECTED_PROJECTION = ((344, 344), (1026, 89))
_EXPECTED_REPRESENTATIVES = {89, 344}
_EXPECTED_METHODS = (
    "diffmean_decoder_only",
    "neg_diffmean_decoder_only",
    "linear_probe_decoder_only",
    "neg_linear_probe_decoder_only",
    "control_decoder_only",
    "neg_control_decoder_only",
    "feature_diffmean_encdec",
    "neg_feature_diffmean_encdec",
    "feature_diffmean_roofline_encdec",
    "neg_feature_diffmean_roofline_encdec",
    "ours_gcg_encdec",
    "neg_ours_gcg_encdec",
    "ours_gcg_roofline_encdec",
    "neg_ours_gcg_roofline_encdec",
)
_SYNTH_SCHEMA = pa.schema(
    [
        ("concept_id", pa.int32()),
        ("content_class", pa.string()),
        ("prefix_text", pa.string()),
        ("output_text", pa.string()),
        ("qc_accepted", pa.bool_()),
        ("prefix_source_shard", pa.int32()),
        ("prefix_source_doc", pa.int32()),
    ]
)

pytestmark = pytest.mark.very_slow


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _require_assets() -> None:
    paths = (
        _FEATURE_DIR,
        _STATISTICS,
        _MASK,
        _CLUSTERS,
        _SELECTION_MANIFEST,
        _DESCRIPTION_CACHE_ROOT / "neuronpedia_cache.yaml",
        _REPO / "nanochat_cache/base_checkpoints/d12_base/model_054600.pt",
    )
    missing = [str(path) for path in paths if not path.exists()]
    if missing:
        pytest.fail(f"H3 clustered acceptance is missing real assets: {missing}")


def _install_local_safetensors_header_reader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep SAELens' shape-only probe offline when the full file is cached."""

    def _local_shapes(repo_id: str, filename: str) -> dict[str, list[int]]:
        local_path = hf_hub_download(
            repo_id,
            filename,
            local_files_only=True,
        )
        with safe_open(local_path, framework="pt", device="cpu") as handle:
            return {
                name: list(handle.get_slice(name).get_shape())
                for name in handle.keys()
            }

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setattr(
        pretrained_sae_loaders,
        "get_safetensors_tensor_shapes",
        _local_shapes,
    )


def _checkpoint_target() -> tuple[str, int]:
    return (
        os.environ.get(
            "PR739_H3_ENCDEC_TAG",
            "l22_main_h100_8rank_db16_v1",
        ),
        int(os.environ.get("PR739_H3_ENCDEC_STEP", "54600")),
    )


def _contract_request(
    *,
    whitelist_path: Path,
    cluster_path: Path = _CLUSTERS,
    model_tag: str,
    step: int,
) -> ClusterFeatureContractRequest:
    return ClusterFeatureContractRequest(
        feature_dir=_FEATURE_DIR,
        feature_statistics_path=_STATISTICS,
        feature_mask={"mode": "whitelist", "path": str(_MASK)},
        feature_clusters={"path": str(cluster_path)},
        checkpoint_source="encdec",
        checkpoint_model_tag=model_tag,
        checkpoint_step=step,
        selection_manifest_path=_SELECTION_MANIFEST,
        concept_feature_whitelist_path=whitelist_path,
        selection_splice_batch_size=64,
        enabled_methods=_EXPECTED_METHODS,
    )


def _write_concept_whitelist(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("344\n1026\n", encoding="utf-8")


def _mutated_cluster_copy(work_root: Path) -> Path:
    output = work_root / "mutations" / "clusters-content-mutated.yaml"
    output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(_CLUSTERS, output)
    payload = yaml.safe_load(output.read_text(encoding="utf-8"))
    clusters = payload["clusters"]
    assert isinstance(clusters, list)
    assert isinstance(clusters[0], list)
    assert len(clusters[0]) > 2
    clusters[0] = clusters[0][:-1]
    output.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return output


def _assert_cluster_mutation_invalidates(
    *,
    contract: ClusterFeatureContract,
    request: ClusterFeatureContractRequest,
    work_root: Path,
) -> dict[str, str]:
    mutated_path = _mutated_cluster_copy(work_root)
    mutated_request = dataclasses.replace(
        request,
        feature_clusters={"path": str(mutated_path)},
    )
    with pytest.raises(
        ClusterFeatureContractError,
        match="FeatureVocab|feature.cluster|cluster.*match|configured.*cluster",
    ) as caught:
        resolve_cluster_feature_contract(mutated_request)
    return {
        "canonical_cluster_sha256": _sha256(_CLUSTERS),
        "mutated_cluster_sha256": _sha256(mutated_path),
        "rejection": str(caught.value),
        "unchanged_contract_fingerprint": contract.feature_contract_fingerprint,
    }


def _stage0a_config(
    *,
    whitelist_path: Path,
    model_tag: str,
    step: int,
) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir=str(_FEATURE_DIR),
        feature_statistics_path=str(_STATISTICS),
        decoder_only={
            "source": "base",
            "model_tag": "d12_base",
            "step": 54600,
        },
        encdec={
            "source": "encdec",
            "model_tag": model_tag,
            "step": step,
        },
        device="cuda",
        output_root=str(_DESCRIPTION_CACHE_ROOT),
        run_tag="pr739-h3-source",
        neuronpedia_source="22-gemmascope-2-res-65k",
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="google/gemma-3-4b-pt",
        extraction_tokenizer_id="google/gemma-3-4b-pt",
        concept_feature_whitelist_path=str(whitelist_path),
        conditioning={
            "feature_mask": {
                "mode": "whitelist",
                "path": str(_MASK),
            },
            "feature_clusters": {"path": str(_CLUSTERS)},
        },
        stage="0a",
        enabled_methods=list(_EXPECTED_METHODS[:6]),
        selection_splice_batch_size=64,
        selection_manifest_path=str(_SELECTION_MANIFEST),
        concept_descriptions_cache_only=True,
        max_concepts=2,
        min_corpus_count=1,
        min_description_length=10,
    )


def _long_prefix(concept_id: int, mode: str, index: int) -> str:
    topic = (
        "A digital archive records URLs, catalog titles, and citation paths."
        if concept_id == 1
        else "A chronology records dates, years, and historical milestones."
    )
    polarity = (
        "The analyst carefully follows the requested theme."
        if mode == "positive"
        else "The analyst deliberately avoids relying on the requested theme."
    )
    sentence = f"{topic} {polarity} Example {index}. "
    return sentence * 18


def _synth_rows(concepts: list[Concept], mode: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        for prefix_index in range(2):
            prefix = _long_prefix(concept.concept_id, mode, prefix_index)
            for content_class in ("concept_rich", "concept_avoiding"):
                output = (
                    "The completion explicitly uses the described pattern with a concrete example."
                    if content_class == "concept_rich"
                    else "The completion stays generic and discusses an unrelated "
                    "ordinary activity."
                )
                rows.append(
                    {
                        "concept_id": concept.concept_id,
                        "content_class": content_class,
                        "prefix_text": prefix,
                        "output_text": output,
                        "qc_accepted": True,
                        "prefix_source_shard": prefix_index,
                        "prefix_source_doc": concept.concept_id,
                    }
                )
    return rows


def _write_parquet(
    path: Path,
    rows: list[dict[str, Any]],
    *,
    metadata: dict[str, str],
    positive_qc_accept_threshold: float | None = None,
) -> None:
    """Write one synth corpus.

    ``positive_qc_accept_threshold`` stamps the Stage 1b positive-QC contract that
    ``validate_positive_qc_contract`` requires of the POSITIVE corpus (#1053). The
    negative corpus carries no such contract, so the parameter is opt-in rather than
    stamped unconditionally -- writing it on both would fabricate a contract for a
    corpus the production writer never stamps.
    """
    encoded = {key.encode("utf-8"): value.encode("utf-8") for key, value in metadata.items()}
    schema = _SYNTH_SCHEMA.with_metadata(encoded)
    if positive_qc_accept_threshold is not None:
        schema = schema_with_positive_qc_contract(
            schema, accept_threshold=positive_qc_accept_threshold
        )
        # #1055 added a SECOND required record: the positive-selection contract.
        # The default variant exists for exactly this case -- its docstring names
        # synthetic/fixture corpora -- so the fixture tracks the shipped selection
        # defaults instead of hand-rolling a payload that would drift.
        schema = schema_with_default_positive_selection_contract(
            schema, accept_threshold=positive_qc_accept_threshold
        )
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def _write_seed_contract(root: Path, contract: ClusterFeatureContract) -> None:
    tokenizer = get_tokenizer()
    prefix_text = "A reproducible clustered acceptance prefix."
    continuation_text = " A deterministic continuation."
    prefix_tokens = list(tokenizer.encode(prefix_text))
    continuation_tokens = list(tokenizer.encode(continuation_text))
    if len(prefix_tokens) < len(continuation_tokens):
        prefix_text = prefix_text * 4
        prefix_tokens = list(tokenizer.encode(prefix_text))
    row = SeedRow(
        shard_idx=0,
        doc_idx=0,
        source_nanochat_token_count=len(prefix_tokens) + len(continuation_tokens),
        prefix_source_token_start=0,
        prefix_source_token_end=len(prefix_tokens),
        prefix_text=prefix_text,
        prefix_content_nanochat_tokens=prefix_tokens,
        sae_feature_ids=[344],
        sae_feature_mags=[1.0],
        continuation_text=continuation_text,
        continuation_content_nanochat_tokens=continuation_tokens,
    )
    config_json = canonical_json(
        {
            "feature_directory": contract.feature_directory.canonical_dict(),
            "seed_continuation_nanochat_tokens": len(continuation_tokens),
            "seed_prefix_max_nanochat_tokens": len(prefix_tokens),
        }
    )
    config_fingerprint = sha256_bytes(config_json.encode("utf-8"))
    source_hashes = {"h3://deterministic-source": "3" * 64}
    parameter_hashes = {
        "encdec_checkpoint": contract.checkpoint.model_file.sha256,
        "feature_statistics": contract.statistics.file_sha256,
    }
    runtime_identity = {
        "backend": "h3-deterministic-source",
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
    }
    extraction_settings = {
        "layer": contract.feature_directory.layer,
        "pooling": contract.feature_directory.pooling,
    }
    tokenizer_checksum = compute_tokenizer_checksum(tokenizer)
    extraction_fingerprint = compute_extraction_fingerprint(
        extraction_settings=extraction_settings,
        runtime_identity=runtime_identity,
        parameter_hashes=parameter_hashes,
    )
    candidate_digest = sha256_file(root / "candidate_concepts.json")
    seed_fingerprint = compute_seed_fingerprint(
        candidate_digest=candidate_digest,
        config_fingerprint=config_fingerprint,
        extraction_fingerprint=extraction_fingerprint,
        nanochat_tokenizer_checksum=tokenizer_checksum,
        source_parquet_hashes=source_hashes,
    )
    common = {
        "schema_version": str(SEED_SCHEMA_VERSION),
        "seed_fingerprint": seed_fingerprint,
        "token_semantics": SEED_TOKEN_SEMANTICS,
        "nanochat_tokenizer_checksum": tokenizer_checksum,
        "extraction_tokenizer_checksum": "2" * 64,
        "config_fingerprint": config_fingerprint,
        "seed_config_json": config_json,
        "source_parquet_hashes_json": canonical_json(source_hashes),
        "parameter_hashes_json": canonical_json(parameter_hashes),
        "runtime_identity_json": canonical_json(runtime_identity),
        "extraction_settings_json": canonical_json(extraction_settings),
    }
    train_path = root / "seeds_train.safetensors"
    eval_path = root / "seeds_eval.safetensors"
    save_seed_pool(train_path, [row], metadata={**common, "split": "train"})
    save_seed_pool(eval_path, [row], metadata={**common, "split": "eval"})
    atomic_json_dump(
        {
            "schema_version": SEED_SCHEMA_VERSION,
            "seed_fingerprint": seed_fingerprint,
            "token_semantics": SEED_TOKEN_SEMANTICS,
            "candidate_digest": candidate_digest,
            "candidate_count": 2,
            "skip_counts": {},
            "source_parquet_hashes": source_hashes,
            "config_fingerprint": config_fingerprint,
            "extraction_fingerprint": extraction_fingerprint,
            "nanochat_tokenizer_checksum": tokenizer_checksum,
            "extraction_tokenizer_checksum": "2" * 64,
            "parameter_hashes": parameter_hashes,
            "runtime_identity": runtime_identity,
            "extraction_settings": extraction_settings,
            "files": {
                train_path.name: {
                    "sha256": sha256_file(train_path),
                    "row_count": 1,
                    "split": "train",
                },
                eval_path.name: {
                    "sha256": sha256_file(eval_path),
                    "row_count": 1,
                    "split": "eval",
                },
            },
        },
        root / "seed_pool_manifest.json",
    )


def _materialize_source(
    *,
    source_root: Path,
    whitelist_path: Path,
    contract: ClusterFeatureContract,
    model_tag: str,
    step: int,
) -> dict[str, Any]:
    ensure_feature_contract_run_marker(source_root, contract)
    stage0a_cfg = _stage0a_config(
        whitelist_path=whitelist_path,
        model_tag=model_tag,
        step=step,
    )
    candidate_path = prepare_concept_list(
        stage0a_cfg,
        cluster_contract=contract,
    )
    assert candidate_path == source_root / "candidate_concepts.json"
    candidates = json.loads(candidate_path.read_text(encoding="utf-8"))
    assert {int(row["feature_index"]) for row in candidates} == (_EXPECTED_REPRESENTATIVES)

    if (source_root / "concepts_stage1c.json").exists():
        validation = validate_stage_0_3_source_run(source_root)
        return {
            "candidate_sha256": _sha256(candidate_path),
            "candidate_provenance_sha256": _sha256(
                source_root / "candidate_concepts.cluster_provenance.json"
            ),
            "source_contract_hash": validation.contract_hash,
        }

    concepts = [
        Concept(concept_id=index, **candidate)
        for index, candidate in enumerate(
            sorted(candidates, key=lambda row: int(row["feature_index"])),
            start=1,
        )
    ]
    concepts_payload = json.dumps(
        [dataclasses.asdict(concept) for concept in concepts],
        indent=2,
    )
    # BPB resolves its concept population through SourcePaths.concepts_json(), which
    # now points at the Stage-1c artifact (#1060). Write BOTH: concepts.json remains a
    # distinct earlier DAG artifact, and the fixture stands in for a completed Stage
    # 0-3 run, so it must satisfy every consumer rather than only the one that used to
    # be checked.
    (source_root / "concepts_stage1c.json").write_text(
        concepts_payload, encoding="utf-8"
    )
    (source_root / "concepts.json").write_text(
        concepts_payload,
        encoding="utf-8",
    )
    target_members = json.dumps(
        {
            str(concept.feature_index): list(
                contract.target_for_representative(concept.feature_index).allowed_member_ids
            )
            for concept in concepts
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    synth_metadata = {
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "cluster_qc_target_algorithm": (
            contract.semantic_dict()["algorithms"]["qc_target_reduction"]
        ),
        "cluster_target_members_json": target_members,
    }
    _write_parquet(
        source_root / "synth_training_positive.parquet",
        _synth_rows(concepts, "positive"),
        metadata=synth_metadata,
        positive_qc_accept_threshold=_POSITIVE_QC_ACCEPT_THRESHOLD,
    )
    _write_parquet(
        source_root / "synth_training_negative.parquet",
        _synth_rows(concepts, "negative"),
        metadata=synth_metadata,
    )
    provenance_sha = _sha256(source_root / "candidate_concepts.cluster_provenance.json")
    stage0c_schema = pa.schema([("concept_id", pa.int32())]).with_metadata(
        {
            b"feature_contract_fingerprint": (
                contract.feature_contract_fingerprint.encode("ascii")
            ),
            b"candidate_cluster_provenance_sha256": provenance_sha.encode("ascii"),
        }
    )
    pq.write_table(
        pa.Table.from_pylist(
            [{"concept_id": concept.concept_id} for concept in concepts],
            schema=stage0c_schema,
        ),
        source_root / "negative_mode_prefixes.parquet",
    )
    # Stage 2 artifacts are INDEXED BY concept_id, not a dense list in concept order:
    # source_validation reads feature_ids[concept_id, 0] and
    # canonical_feature_id[concept_id], and requires max(concept_id) + 1 rows. Concept
    # ids start at 1 here, so row 0 exists and stays sentinel-filled. The dense layout
    # this replaces was wrong all along -- #1060's concept-yield guard is simply the
    # first thing to check it.
    stage2_rows = max(concept.concept_id for concept in concepts) + 1
    stage2_feature_ids = t.full((stage2_rows, 2), -1, dtype=t.int32)
    stage2_canonical = t.full((stage2_rows,), -1, dtype=t.int32)
    for concept in concepts:
        stage2_feature_ids[concept.concept_id, 0] = concept.feature_index
        stage2_canonical[concept.concept_id] = concept.feature_index
    stage2 = {
        "feature_ids": stage2_feature_ids,
        "canonical_feature_id": stage2_canonical,
    }
    for filename in (
        "concept_features.safetensors",
        "concept_features_neg.safetensors",
    ):
        save_file(
            stage2,
            str(source_root / filename),
            metadata={"feature_contract_fingerprint": (contract.feature_contract_fingerprint)},
        )
    # Hash via the SAME helper source_validation uses to derive its expectation
    # (_baseline_training_corpus_sha256's no-provenance fallback). A hand-rolled
    # sha256 happens to agree today, but nothing would keep the two in step.
    corpus_hashes = {
        "positive": stream_hash_file_stable(
            source_root / "synth_training_positive.parquet"
        ).sha256,
        "negative": stream_hash_file_stable(
            source_root / "synth_training_negative.parquet"
        ).sha256,
    }
    for family in ("diffmean", "linear_probe"):
        for mode in ("positive", "negative"):
            for concept in concepts:
                direction_path = (
                    source_root
                    / "baselines"
                    / family
                    / "decoder_only"
                    / mode
                    / f"concept_{concept.concept_id:04d}.safetensors"
                )
                direction_path.parent.mkdir(parents=True, exist_ok=True)
                direction = t.zeros(768)
                direction[concept.concept_id - 1] = 1.0 if mode == "positive" else -1.0
                save_file(
                    {"direction": direction},
                    str(direction_path),
                    metadata={
                        "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
                        "source_corpus_sha256": corpus_hashes[mode],
                        "target_model": "decoder_only",
                        "mode": mode,
                        # Required by source_validation and compared again at
                        # consumption time by _build_direction_intervention, which
                        # matches metadata["layer"] against cfg's
                        # diffmean_and_probe_layer (6). "layer" must round-trip
                        # through str() exactly -- no padding.
                        "concept_id": str(concept.concept_id),
                        "layer": str(_DIFFMEAN_LAYER),
                    },
                )
    _write_seed_contract(source_root, contract)
    validation = validate_stage_0_3_source_run(source_root)
    assert validation.n_concepts == 2
    assert validation.n_positive_prefixes == 4
    assert validation.n_negative_prefixes == 4
    return {
        "candidate_sha256": _sha256(candidate_path),
        "candidate_provenance_sha256": provenance_sha,
        "source_contract_hash": validation.contract_hash,
    }


def _compose_bpb_config(
    *,
    source_root: Path,
    output_root: Path,
    model_tag: str,
    step: int,
) -> DictConfig:
    GlobalHydra.instance().clear()
    with initialize_config_dir(
        config_dir=str(_CONFIG_DIR),
        version_base=None,
    ):
        return compose(
            config_name="axbench_bpb_adv_clustered_gpu_acceptance",
            overrides=[
                f"source_run_dir={source_root}",
                "run_tag=h3-clustered-gpu",
                f"output_root={output_root}",
                f"feature_dir={_FEATURE_DIR}",
                f"feature_statistics_path={_STATISTICS}",
                f"feature_mask.path={_MASK}",
                f"feature_clusters.path={_CLUSTERS}",
                f"selection_manifest_path={_SELECTION_MANIFEST}",
                f"encdec_target.model_tag={model_tag}",
                f"encdec_target.step={step}",
            ],
        )


def _tree_snapshot(root: Path) -> dict[str, str]:
    excluded_roots = {"profile"}
    return {
        str(path.relative_to(root)): _sha256(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and not set(path.relative_to(root).parts) & excluded_roots
    }


def _nvidia_snapshot() -> str:
    command = [
        "nvidia-smi",
        "--query-gpu=timestamp,name,uuid,memory.used,memory.total,utilization.gpu,temperature.gpu",
        "--format=csv,noheader",
    ]
    try:
        result = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        output = getattr(exc, "stdout", None) or getattr(exc, "output", None)
        pytest.fail(
            "H3 requires a successful nvidia-smi immediately before every GPU "
            f"run; command failed before model execution: {output or exc}"
        )
    return result.stdout.strip()


def _assert_no_competing_gpu_process() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    rows = [row.strip() for row in result.stdout.splitlines() if row.strip()]
    competitors = [row for row in rows if row.split(",", maxsplit=1)[0].strip() != str(os.getpid())]
    if competitors:
        pytest.fail(f"H3 refuses to contend for a busy GPU; competing processes: {competitors}")
    return "\n".join(rows)


def _assert_final_tables(
    paths: AxBenchBpbAdvPaths,
    contract: ClusterFeatureContract,
) -> dict[str, Any]:
    attack = pq.read_table(paths.attack_rows_path())
    candidates = pq.read_table(paths.candidates_path())
    scores = pq.read_table(paths.scores_path())
    assert attack.schema.remove_metadata().equals(ATTACK_ROWS_SCHEMA)
    assert candidates.schema.remove_metadata().equals(CANDIDATES_SCHEMA)
    assert scores.schema.remove_metadata().equals(SCORE_SCHEMA)
    assert set(attack.column("method_name").to_pylist()) == set(_EXPECTED_METHODS)
    assert set(candidates.column("method_name").to_pylist()) == set(_EXPECTED_METHODS)
    assert set(scores.column("method_name").to_pylist()) == set(_EXPECTED_METHODS)
    assert set(attack.column("concept_id").to_pylist()) == {1, 2}
    assert candidates.num_rows == attack.num_rows
    assert scores.num_rows == candidates.num_rows
    for table, family in (
        (attack, "attack_rows"),
        (candidates, "candidates"),
        (scores, "class_margin_scores"),
    ):
        identity = artifact_identity_metadata(table)
        assert identity["feature_contract_fingerprint"] == (contract.feature_contract_fingerprint)
        assert identity["clustered_artifact_family"] == family
    return {
        "attack_rows": attack.num_rows,
        "candidates": candidates.num_rows,
        "scores": scores.num_rows,
        "methods": sorted(set(scores.column("method_name").to_pylist())),
    }


def test_real_clustered_two_representative_stage0a_to_bpb_acceptance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _require_assets()
    _install_local_safetensors_header_reader(monkeypatch)
    # The BPB-Adv unit-test subtree installs a fake seed tokenizer autouse.
    # This real acceptance must explicitly restore the production tokenizer.
    monkeypatch.setattr(source_validation, "get_tokenizer", get_tokenizer)
    model_tag, step = _checkpoint_target()
    work_root = Path(os.environ.get("PR739_H3_WORK_ROOT", str(tmp_path / "h3"))).resolve()
    source_root = work_root / "source"
    output_root = work_root / "bpb"
    whitelist_path = work_root / "concept-sources.txt"
    _write_concept_whitelist(whitelist_path)

    request = _contract_request(
        whitelist_path=whitelist_path,
        model_tag=model_tag,
        step=step,
    )
    contract = resolve_cluster_feature_contract(request)
    expected_feature_contract = _EXPECTED_FEATURE_CONTRACTS.get((model_tag, step))
    assert expected_feature_contract is not None, (
        f"unreviewed H3 checkpoint target: {(model_tag, step)!r}"
    )
    assert contract.feature_contract_fingerprint == expected_feature_contract
    assert contract.feature_clusters.descriptor.identity_sha256() == (_EXPECTED_CLUSTER_DESCRIPTOR)
    assert contract.feature_clusters.descriptor.content_sha256 == (_EXPECTED_CLUSTER_CONTENT)
    assert contract.selection.selection_hash == _EXPECTED_SELECTION_HASH
    assert contract.source_to_representative == _EXPECTED_PROJECTION
    # main pruned representative_for_source as a test-only convenience
    # (383cb613a); source_to_representative is the surviving public surface.
    projection = dict(contract.source_to_representative)
    assert projection[1026] == 89, "sibling evidence must project to its representative"
    assert 1026 in contract.target_for_representative(89).allowed_member_ids
    assert projection[344] == 344, "a representative must project to itself"

    mutation_evidence = _assert_cluster_mutation_invalidates(
        contract=contract,
        request=request,
        work_root=work_root,
    )
    # run_output_root is a METHOD on AxBenchBasePaths, not a module-level
    # function -- patching the module silently targeted an attribute that never
    # existed, so this only surfaced once the call was actually reached.
    monkeypatch.setattr(
        axbench_base_paths.AxBenchBasePaths,
        "run_output_root",
        lambda _self: source_root,
    )
    source_evidence = _materialize_source(
        source_root=source_root,
        whitelist_path=whitelist_path,
        contract=contract,
        model_tag=model_tag,
        step=step,
    )
    semantic_contract = contract.semantic_dict()
    preflight_evidence = {
        "checkpoint": {
            "model_tag": model_tag,
            "step": step,
            "model_sha256": contract.checkpoint.model_file.sha256,
            "metadata_sha256": contract.checkpoint.metadata_file.sha256,
        },
        "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        "cluster_descriptor_fingerprint": (contract.feature_clusters.descriptor.identity_sha256()),
        "cluster_content_sha256": (contract.feature_clusters.descriptor.content_sha256),
        "feature_stack": {
            "directory": semantic_contract["feature_directory"],
            "sae": semantic_contract["sae"],
            "mask_file_sha256": _sha256(_MASK),
            "mask_allowed_ids_sha256": semantic_contract["feature_mask"]["allowed_ids_sha256"],
            "mask_allowed_count": semantic_contract["feature_mask"]["allowed_count"],
        },
        "selection_hash": contract.selection.selection_hash,
        "selection_manifest_file_sha256": (contract.selection.selection_manifest_file_sha256),
        "statistics_sha256": contract.statistics.file_sha256,
        "source_projection": list(contract.source_to_representative),
        "source": source_evidence,
        "mutation": mutation_evidence,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "preflight-evidence.json").write_text(
        json.dumps(preflight_evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    cfg = _compose_bpb_config(
        source_root=source_root,
        output_root=output_root,
        model_tag=model_tag,
        step=step,
    )
    paths = AxBenchBpbAdvPaths(
        output_root=output_root,
        run_tag="h3-clustered-gpu",
    )

    stage_events: dict[str, list[dict[str, Any]]] = {
        "initial": [],
        "resume": [],
    }
    phase = "initial"
    original_run_stage = bpb_adv_cli._run_stage

    def _captured_stage(stage: Any, *args: Any, **kwargs: Any) -> None:
        before = _tree_snapshot(paths.root)
        started = time.perf_counter()
        original_run_stage(stage, *args, **kwargs)
        if t.cuda.is_available():
            t.cuda.synchronize()
        elapsed = time.perf_counter() - started
        after = _tree_snapshot(paths.root)
        stage_events[phase].append(
            {
                "stage": stage.value,
                "wall_seconds": elapsed,
                "before": before,
                "after": after,
            }
        )
        if phase == "resume":
            assert after == before, f"resume stage {stage.value} changed committed artifact bytes"

    monkeypatch.setattr(bpb_adv_cli, "_run_stage", _captured_stage)
    gpu_before_initial = _nvidia_snapshot()
    competing_before_initial = _assert_no_competing_gpu_process()
    initial_started = time.perf_counter()
    bpb_adv_cli._run(cfg)
    initial_wall = time.perf_counter() - initial_started
    gpu_after_initial = _nvidia_snapshot()
    first_final_snapshot = _tree_snapshot(paths.root)

    phase = "resume"
    gpu_before_resume = _nvidia_snapshot()
    competing_before_resume = _assert_no_competing_gpu_process()
    resume_started = time.perf_counter()
    bpb_adv_cli._run(cfg)
    resume_wall = time.perf_counter() - resume_started
    gpu_after_resume = _nvidia_snapshot()
    assert _tree_snapshot(paths.root) == first_final_snapshot
    table_evidence = _assert_final_tables(paths, contract)

    evidence = {
        "schema_version": 1,
        "checkpoint": {
            "model_tag": model_tag,
            "step": step,
            "model_sha256": contract.checkpoint.model_file.sha256,
            "metadata_sha256": contract.checkpoint.metadata_file.sha256,
        },
        "contract": {
            "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
            "cluster_descriptor_fingerprint": (
                contract.feature_clusters.descriptor.identity_sha256()
            ),
            "cluster_content_sha256": (contract.feature_clusters.descriptor.content_sha256),
            "selection_hash": contract.selection.selection_hash,
            "selection_manifest_sha256": (contract.selection.selection_manifest_file_sha256),
            "statistics_sha256": contract.statistics.file_sha256,
            "source_projection": list(contract.source_to_representative),
            "feature_stack": preflight_evidence["feature_stack"],
        },
        "source": source_evidence,
        "mutation": mutation_evidence,
        "tables": table_evidence,
        "wall_clock": {
            "initial_seconds": initial_wall,
            "resume_seconds": resume_wall,
        },
        "nvidia_smi": {
            "before_initial": gpu_before_initial,
            "competing_before_initial": competing_before_initial,
            "after_initial": gpu_after_initial,
            "before_resume": gpu_before_resume,
            "competing_before_resume": competing_before_resume,
            "after_resume": gpu_after_resume,
        },
        "per_stage": stage_events,
        "final_artifact_sha256": first_final_snapshot,
    }
    work_root.mkdir(parents=True, exist_ok=True)
    (work_root / "acceptance-evidence.json").write_text(
        json.dumps(evidence, indent=2, sort_keys=True),
        encoding="utf-8",
    )
