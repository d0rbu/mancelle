"""Unit tests for Stage 4 ``generate_all`` orchestration (Task 38).

Covers :func:`steering.eval.axbench_base.generation.generate_all`, which
iterates every ``(method, concept, eval_seed, coefficient)`` combination
in the registry and emits one row per combination to the generations
directory-dataset.

All tests mock :func:`run_intervention` via monkeypatch so the heavy
dispatch / model / hook machinery is never exercised here. Each test
pins specific expected call counts, schema contents, or resume
behaviour, isolating ``generate_all``'s outer-loop logic.
"""

from __future__ import annotations

import hashlib
import inspect
import itertools
import json
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pyarrow as pa
import pytest
import torch as t
import torch.nn as nn
from beartype.roar import BeartypeCallHintParamViolation
from safetensors.torch import save_file

from errors import FeatureClustersError
from steering.eval.axbench_base import generation as generation_mod
from steering.eval.axbench_base import paths
from steering.eval.axbench_base import runtime_state as runtime_state_mod
from steering.eval.axbench_base import seed_contract as seed_contract_mod
from steering.eval.axbench_base.config import AxBenchBaseConfig, GenerationConfig
from steering.eval.axbench_base.generation import (
    GENERATIONS_SCHEMA,
    Stage4MethodPlan,
    _authoritative_concepts_by_mode,
    _validate_stage4_context_cache,
    build_generation_key,
)
from steering.eval.axbench_base.generation import (
    generate_all as _generate_all,
)
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.seed_contract import load_seed_pool_fingerprint
from steering.eval.axbench_base.stage4_prompts import prompt_token_key
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    SteeringMode,
    TargetBundle,
    TargetModelMetadata,
)
from steering.eval.cluster_checkpoint_pin import (
    ClusterFeatureContractError,
    assert_checkpoint_files_pin_unchanged,
    resolve_checkpoint_files_pin,
)
from steering.eval.cluster_contract_identity import _sha256_json
from steering.eval.target_conditioning import ContextFeatureSegment
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.seed_contract import write_committed_seed_pair
from tests.axbench_base.unit.test_config import make_valid_config

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def generate_all(*args: Any, apply_max_act_scale: bool = False, **kwargs: Any) -> Path:
    """Keep orchestration-only tests explicit about the legacy off regime."""
    return _generate_all(
        *args,
        apply_max_act_scale=apply_max_act_scale,
        **kwargs,
    )


def _concept(concept_id: int, description: str = "the letter e") -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id * 11,
        layer=8,
        sae_source="stub",
        description=description,
        corpus_count=100,
        description_source="stub",
    )


def _seed_row(doc_idx: int = 0, text: str = "prefix") -> SeedRow:
    """Construct a minimal SeedRow with a non-empty continuation.

    ``__post_init__`` rejects empty continuation content tokens, so we
    always supply a one-token continuation list. The other fields are
    set to empty-ish defaults since the outer loop only reads
    ``prefix_content_nanochat_tokens`` and ``prefix_text`` from seed rows.
    """
    return SeedRow(
        shard_idx=0,
        doc_idx=doc_idx,
        source_nanochat_token_count=4,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text=text,
        prefix_content_nanochat_tokens=[1, 2, 3],
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="cont",
        continuation_content_nanochat_tokens=[7],
    )


def _meta(
    *,
    num_buckets: int | None = 4,
    negative_selection: dict[str, Any] | None = None,
) -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=num_buckets,
        n_sae_features=16384,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


class _TargetModule(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(sequence_len=128)


def _make_target_module() -> _TargetModule:
    """Return a minimal nn.Module suitable as a ``target_model`` arg.

    The outer loop never calls forward on this module; it's only
    threaded through to ``run_intervention``, which is mocked.
    """
    return _TargetModule()


def _model_sequence_len(model: nn.Module) -> int:
    config = getattr(model, "config", None)
    sequence_len = getattr(config, "sequence_len", None)
    assert isinstance(sequence_len, int)
    return sequence_len


def _target_bundle(
    *,
    name: str,
    context_provider=None,
    cluster_contract=None,
) -> TargetBundle:
    return TargetBundle(
        name=name,
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=context_provider,
        cluster_contract=cluster_contract,
    )


def _make_cfg(
    tmp_path: Path,
    *,
    run_tag: str | None = "rt",
    enabled_methods: list[str] | None = None,
    disabled_methods: list[str] | None = None,
) -> AxBenchBaseConfig:
    overrides: dict[str, Any] = {
        "run_tag": run_tag,
        "output_root": str(tmp_path / "eval_out"),
        "generation": GenerationConfig(
            max_new_tokens=4,
            temperature=0.7,
            top_k_sampling=50,
            master_seed=13,
        ),
    }
    if enabled_methods is not None:
        overrides["enabled_methods"] = enabled_methods
    if disabled_methods is not None:
        overrides["disabled_methods"] = disabled_methods
    cfg = make_valid_config(**overrides)
    selected_names = (
        {method.name for method in ALL_METHODS}
        if cfg.enabled_methods is None
        else set(cfg.enabled_methods)
    ) - set(cfg.disabled_methods)
    scalar_selected = any(
        method.name in selected_names and method.baseline.value in {"diffmean", "linear_probe"}
        for method in ALL_METHODS
    )
    if cfg.run_tag is not None and scalar_selected:
        output_paths = paths.AxBenchBasePaths.from_config(cfg)
        for family in ("diffmean", "linear_probe"):
            path_fn = (
                output_paths.diffmean_direction
                if family == "diffmean"
                else output_paths.linear_probe_direction
            )
            for target in ("decoder_only", "encdec"):
                for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
                    for concept_id in (0, 1):
                        artifact = path_fn(target, mode, concept_id)
                        artifact.parent.mkdir(parents=True, exist_ok=True)
                        save_file(
                            {"direction": t.zeros(8, dtype=t.float32)},
                            str(artifact),
                            metadata={
                                "concept_id": str(concept_id),
                                "layer": str(cfg.layer_hooks.diffmean_and_probe_layer),
                                "mode": mode.value,
                                "target_model": target,
                            },
                        )
    return cfg


@pytest.mark.parametrize("run_negative_mode", [False, True])
def test_authoritative_concepts_are_shared_only_with_enabled_modes(
    tmp_path: Path,
    run_negative_mode: bool,
) -> None:
    concepts = [_concept(0), _concept(1)]
    cfg = _make_cfg(tmp_path).replace(
        run_negative_mode=run_negative_mode,
    )

    by_mode = _authoritative_concepts_by_mode(cfg, concepts)

    assert set(by_mode) == {
        SteeringMode.POSITIVE,
        SteeringMode.NEGATIVE,
    }
    assert by_mode[SteeringMode.POSITIVE] is concepts
    if run_negative_mode:
        assert by_mode[SteeringMode.NEGATIVE] is concepts
    else:
        assert by_mode[SteeringMode.NEGATIVE] == []


def _set_base_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))


class _Tokenizer:
    """Minimal tokenizer stub (never actually touched by the outer loop)."""

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        del prepend
        return [ord(c) % 256 for c in text]

    def get_bos_token_id(self) -> int:
        return 1

    def decode(self, ids: list[int]) -> str:
        return f"dec:{','.join(str(i) for i in ids)}"


class _ArtifactTokenizer:
    """Small reversible tokenizer for committed seed-artifact regressions."""

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        encoded = [1_000 + ord(char) for char in text]
        return encoded if prepend is None else [prepend, *encoded]

    def get_bos_token_id(self) -> int:
        return 1

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(token - 1_000) for token in ids)


def _artifact_seed_row(*, doc_idx: int, prefix: str) -> SeedRow:
    tokenizer = _ArtifactTokenizer()
    continuation = "cont"
    prefix_tokens = tokenizer.encode(prefix)
    continuation_tokens = tokenizer.encode(continuation)
    return SeedRow(
        shard_idx=0,
        doc_idx=doc_idx,
        source_nanochat_token_count=len(prefix_tokens) + len(continuation_tokens),
        prefix_source_token_start=0,
        prefix_source_token_end=len(prefix_tokens),
        prefix_text=prefix,
        prefix_content_nanochat_tokens=prefix_tokens,
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text=continuation,
        continuation_content_nanochat_tokens=continuation_tokens,
    )


def test_target_bundle_context_provider_argument_is_required() -> None:
    param = inspect.signature(TargetBundle).parameters["context_provider"]
    assert param.default is inspect.Parameter.empty


def test_target_sequence_len_supports_raw_encdec_dec_config() -> None:
    model = SimpleNamespace(dec_config=SimpleNamespace(sequence_len=257))
    assert (
        generation_mod._target_sequence_len(
            model,
            target_name="encdec",
        )
        == 257
    )


def test_prompt_length_preflight_fails_before_generation_write(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_decoder_only"])
    model = _make_target_module()
    model.config.sequence_len = 7
    mock = _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="prompt preflight exceeds"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[
                TargetBundle(
                    name="decoder_only",
                    model=model,
                    meta=_meta(),
                    tokenizer=_Tokenizer(),
                    context_provider=None,
                )
            ],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )

    assert mock.call_count == 0
    assert cfg.run_tag is not None
    assert not paths.AxBenchBasePaths.from_config(cfg).generations_dir().exists()


def test_clustered_decoder_only_stage4_rejects_mismatched_stage3e_contract(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Context ownership is checked even when no encdec method reads rows."""

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "contexts_fingerprint": "c" * 64,
            "feature_contract_fingerprint": "b" * 64,
        },
    )
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=cast(Any, _FakeClusterContract()),
    )

    with pytest.raises(AxBenchBaseError, match="do not match.*Stage 4 target"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[target],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )


def test_stage4_rejects_mixed_cluster_contract_state_across_targets() -> None:
    targets = [
        _target_bundle(
            name="decoder_only",
            cluster_contract=cast(Any, _FakeClusterContract()),
        ),
        _target_bundle(name="encdec"),
    ]

    with pytest.raises(AxBenchBaseError, match="mix clustered and unclustered"):
        generation_mod._cluster_contract_for_targets(targets)


def test_stage4_rejects_different_cluster_contracts_across_targets() -> None:
    targets = [
        _target_bundle(
            name="decoder_only",
            cluster_contract=cast(
                Any,
                SimpleNamespace(feature_contract_fingerprint="a" * 64),
            ),
        ),
        _target_bundle(
            name="encdec",
            cluster_contract=cast(
                Any,
                SimpleNamespace(feature_contract_fingerprint="b" * 64),
            ),
        ),
    ]

    with pytest.raises(AxBenchBaseError, match="different feature contracts"):
        generation_mod._cluster_contract_for_targets(targets)


def test_cluster_artifact_hash_wraps_stable_hash_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        generation_mod,
        "stream_hash_file_stable",
        lambda _path: (_ for _ in ()).throw(OSError("unreadable")),
    )
    with pytest.raises(AxBenchBaseError, match="could not stably hash final concepts"):
        generation_mod._stable_cluster_artifact_sha256(
            tmp_path / "concepts.json",
            surface="final concepts",
        )


def test_cluster_context_manifest_requires_provider_and_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = cast(Any, _FakeClusterContract())
    with pytest.raises(AxBenchBaseError, match="has no context provider"):
        generation_mod._context_cache_manifest_sha256(
            _target_bundle(
                name="encdec",
                context_provider=None,
                cluster_contract=contract,
            ),
        )

    malformed = tmp_path / "manifest.json"
    malformed.write_text("{malformed", encoding="utf-8")
    with pytest.raises(AxBenchBaseError, match="could not freeze.*manifest"):
        generation_mod._context_cache_manifest_sha256(
            _target_bundle(
                name="encdec",
                context_provider=cast(
                    Any,
                    SimpleNamespace(manifest_path=malformed, manifest={}),
                ),
                cluster_contract=contract,
            ),
        )

    with pytest.raises(AxBenchBaseError, match="no exact manifest identity"):
        generation_mod._context_cache_manifest_sha256(
            _target_bundle(
                name="encdec",
                context_provider=cast(Any, SimpleNamespace()),
                cluster_contract=contract,
            ),
        )


def test_unclustered_decoder_only_stage4_never_reads_stale_stage3e_manifest(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """U-invariant: legacy decoder-only reads match the pinned base exactly."""

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    assert cfg.run_tag is not None
    manifest_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_manifest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("{stale malformed Stage3e manifest", encoding="utf-8")
    original_is_file = Path.is_file

    def discriminating_is_file(path: Path) -> bool:
        if path == manifest_path:
            raise AssertionError("unclustered Stage 4 probed the Stage 3e manifest")
        return original_is_file(path)

    monkeypatch.setattr(Path, "is_file", discriminating_is_file)
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: pytest.fail("unclustered Stage 4 read the Stage 3e manifest"),
    )
    marker_checks: list[tuple[Any, Any, dict[str, Any]]] = []
    real_validate = generation_mod.validate_cluster_generation_resume

    def validate_marker(*args: Any, **kwargs: Any) -> None:
        marker_checks.append((args[0], args[1], kwargs))
        real_validate(*args, **kwargs)

    monkeypatch.setattr(
        generation_mod,
        "validate_cluster_generation_resume",
        validate_marker,
    )
    monkeypatch.setattr(
        generation_mod,
        "commit_cluster_generation_marker",
        lambda *_args, **_kwargs: pytest.fail(
            "unclustered Stage 4 probed or wrote the cluster resume marker"
        ),
    )
    intervention = _patch_run_intervention(monkeypatch)
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
    )

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=[target],
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[],
    )

    assert intervention.call_count == 1
    assert marker_checks == [
        (
            cfg,
            None,
            {"contexts_fingerprint": None},
        )
    ]


def test_unclustered_stage4_rejects_committed_cluster_generation_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A clustered crash marker makes the same run_tag unavailable to U."""

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    assert cfg.run_tag is not None
    marker = paths.AxBenchBasePaths.from_config(cfg).generations_feature_contract_json()
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("{}", encoding="utf-8")
    intervention = _patch_run_intervention(monkeypatch)
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="Unclustered Stage 4 rejects cluster-stamped generations",
    ):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[target],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )

    intervention.assert_not_called()


def test_clustered_generate_all_commits_matching_real_marker_and_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Compose real generation, marker commit/resume, and parquet effects."""

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["control_decoder_only"],
    ).replace(
        conditioning={
            "feature_mask": {
                "mode": "whitelist",
                "path": str(tmp_path / "mask.txt"),
            },
            "feature_clusters": {"path": str(tmp_path / "clusters.yaml")},
        },
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)

    semantic = {
        "algorithms": {
            "concept_projection": "mask_first_min_allowed_representative_v1",
        },
        "feature_clusters": {
            "representative_universe": [1, 5],
            "representative_to_allowed_members": [
                [1, [1, 2]],
                [5, [5]],
            ],
        },
        "concept_source": {
            "mode": "explicit_whitelist",
            "domain": [2, 5],
            "source_to_representative": [[2, 1], [5, 5]],
        },
    }
    feature_contract_fingerprint = _sha256_json(semantic)
    run_root = output_paths.run_output_root()
    run_root.mkdir(parents=True)
    output_paths.feature_contract_json().write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_contract_fingerprint": feature_contract_fingerprint,
                "semantic": semantic,
                "provenance": {},
            },
        ),
        encoding="utf-8",
    )
    candidate_rows = [{"feature_index": 1}, {"feature_index": 5}]
    candidate_path = output_paths.candidate_concepts_json()
    candidate_path.write_text(json.dumps(candidate_rows), encoding="utf-8")
    output_paths.candidate_concepts_cluster_provenance_json().write_text(
        json.dumps(
            {
                "schema_version": 1,
                "feature_contract_fingerprint": feature_contract_fingerprint,
                "candidate_concepts_sha256": hashlib.sha256(
                    candidate_path.read_bytes(),
                ).hexdigest(),
                "concept_projection_version": ("mask_first_min_allowed_representative_v1"),
                "counts": {
                    "raw_source_ids": 2,
                    "mask_allowed_source_ids": 2,
                    "unique_representatives": 2,
                    "representatives_with_statistics": 2,
                    "count_survivors": 2,
                    "description_fetch_survivors": 2,
                    "described_survivors": 2,
                },
                "representatives": [
                    {
                        "representative_id": 1,
                        "allowed_member_ids": [1, 2],
                        "projecting_source_ids": [2],
                        "statistics_source_id": 1,
                        "outcome": "candidate_emitted",
                    },
                    {
                        "representative_id": 5,
                        "allowed_member_ids": [5],
                        "projecting_source_ids": [5],
                        "statistics_source_id": 5,
                        "outcome": "candidate_emitted",
                    },
                ],
            },
        ),
        encoding="utf-8",
    )

    concept = _concept(0)
    concepts_path = output_paths.concepts_json()
    concepts_path.write_text(
        json.dumps([asdict(concept)]),
        encoding="utf-8",
    )
    consumed_generation_inputs = {
        "schema_version": 1,
        "feature_contract_fingerprint": feature_contract_fingerprint,
        "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
        "artifact_sha256": {
            "concepts.json": hashlib.sha256(concepts_path.read_bytes()).hexdigest(),
        },
        "composition_sentinel": "payload-consumed-by-generate-all",
    }
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "contexts_fingerprint": "c" * 64,
            "feature_contract_fingerprint": feature_contract_fingerprint,
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "_cluster_generation_inputs",
        lambda *args, **kwargs: consumed_generation_inputs,
    )
    intervention = _patch_run_intervention(monkeypatch)
    clustered_target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=cast(
            Any,
            SimpleNamespace(
                feature_contract_fingerprint=feature_contract_fingerprint,
                consolidation=_RepresentativeConsolidation(),
            ),
        ),
    )

    generations_dir = generate_all(
        cfg,
        concepts=[concept],
        targets=[clustered_target],
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[],
    )

    marker_path = output_paths.generations_feature_contract_json()
    batch_path = generations_dir / "batch_000000.parquet"
    assert marker_path.is_file()
    assert batch_path.is_file()
    assert intervention.call_count == 1
    marker_payload = json.loads(marker_path.read_text(encoding="utf-8"))
    batch_rows = load_existing(generations_dir, schema=GENERATIONS_SCHEMA).to_pylist()
    assert len(batch_rows) == 1

    assert marker_payload["generation_inputs"] == consumed_generation_inputs, (
        "the real marker must contain the exact generation-inputs payload consumed by generate_all"
    )
    assert (
        batch_rows[0]["seed_pool_fingerprint"]
        == (marker_payload["generation_inputs"]["seed_pool_fingerprint"])
    ), "the real parquet batch and real marker must bind the same generation inputs"

    unclustered_cfg = cfg.replace(conditioning=None)
    unclustered_target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
    )
    with pytest.raises(
        AxBenchBaseError,
        match="Unclustered Stage 4 rejects cluster-stamped generations",
    ):
        generate_all(
            unclustered_cfg,
            concepts=[concept],
            targets=[unclustered_target],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )


class _ContextProvider:
    def __init__(
        self,
        *,
        metadata: dict[str, Any],
        segments: tuple[Any, ...] | None,
    ) -> None:
        self.manifest = {"metadata": metadata}
        self.segments = segments
        self.seen: list[list[int]] = []

    def get(self, token_ids: Sequence[int]) -> tuple[Any, ...] | None:
        self.seen.append(list(token_ids))
        return self.segments


class _RepresentativeConsolidation:
    def assert_representatives_only(
        self,
        feature_ids: t.Tensor,
        *,
        surface: str,
    ) -> None:
        del surface
        non_representative = [
            feature_id for feature_id in feature_ids.tolist() if feature_id not in {1, 2}
        ]
        if non_representative:
            raise FeatureClustersError(
                f"clustered test contains non-representative feature ID {non_representative[0]}"
            )


class _FakeClusterContract:
    feature_contract_fingerprint = "a" * 64
    consolidation = _RepresentativeConsolidation()


def _fake_decoder_checkpoint_pin(
    *,
    step: int = 7,
    model_sha256: str = "d" * 64,
) -> Any:
    payload = {
        "resolved_tag": "d-test",
        "step": step,
        "model_sha256": model_sha256,
        "metadata_sha256": "e" * 64,
    }
    return SimpleNamespace(
        step=step,
        semantic_dict=lambda: dict(payload),
    )


def _patch_bound_sources(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seed_fingerprint: str = "4" * 64,
) -> None:
    monkeypatch.setattr(
        generation_mod,
        "_bind_cluster_generation_sources",
        lambda *_args, **_kwargs: ("1" * 64, seed_fingerprint),
    )


def test_cluster_generation_inputs_preflights_scalar_artifacts_and_scale(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    _patch_bound_sources(monkeypatch)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_decoder_only"])
    assert cfg.run_tag is not None
    concept = _concept(1)
    contract = cast(Any, _FakeClusterContract())
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
        checkpoint_files_pin=_fake_decoder_checkpoint_pin(),
    )
    method = next(method for method in ALL_METHODS if method.name == "diffmean_decoder_only")
    plan = Stage4MethodPlan(
        method=method,
        target_name="decoder_only",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=[_seed_row()],
        coefficients=[0.2, 0.4],
        concept_features_path=tmp_path / "unused.safetensors",
        cluster_source_corpus_sha256="7" * 64,
    )
    loaded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_scalar_load(*args: Any, **kwargs: Any) -> None:
        loaded.append((args, kwargs))

    monkeypatch.setattr(
        generation_mod,
        "_load_scalar_direction_artifact",
        record_scalar_load,
    )
    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: {
            "final concepts": "1" * 64,
            "Stage 3e manifest": "2" * 64,
            "diffmean_decoder_only direction for concept 1": "3" * 64,
        }[surface],
    )
    payload = generation_mod._cluster_generation_inputs(
        cfg,
        cluster_contract=contract,
        concepts=[concept],
        plans=[plan],
        eligible_by_mode={
            SteeringMode.POSITIVE: [concept],
            SteeringMode.NEGATIVE: [],
        },
        seed_pool_fingerprint="4" * 64,
        consumed_eval_seeds=[_seed_row()],
        steering_scale_identity="dirmaxact:cluster-test",
    )

    assert loaded == [
        (
            (
                paths.AxBenchBasePaths.from_config(cfg).diffmean_direction(
                    "decoder_only",
                    SteeringMode.POSITIVE,
                    concept.concept_id,
                ),
            ),
            {
                "method": method,
                "concept": concept,
                "target_model_name": "decoder_only",
                "layer": cfg.layer_hooks.diffmean_and_probe_layer,
                "output_paths": paths.AxBenchBasePaths.from_config(cfg),
                "cluster_contract": target.cluster_contract,
                "source_corpus_sha256": "7" * 64,
            },
        ),
    ]
    assert payload["artifact_sha256"] == {
        "concepts.json": "1" * 64,
        "scalar:diffmean_decoder_only:concept:1": "3" * 64,
        "stage4_contexts_manifest.json": "2" * 64,
    }
    assert payload["source_corpus_sha256"] == {"positive": "7" * 64}
    assert payload["steering_scale_identity"] == "dirmaxact:cluster-test"
    assert payload["method_plans"][0]["coefficients"] == [0.2, 0.4]


def test_cluster_generation_inputs_preflights_ours_rows_and_context_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    _patch_bound_sources(monkeypatch)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["ours_extract_and_add_encdec"],
    )
    concept = _concept(0)
    contract = cast(Any, _FakeClusterContract())
    provider = _ContextProvider(metadata={}, segments=())
    target = TargetBundle(
        name="encdec",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=provider,
        cluster_contract=contract,
    )
    method = next(method for method in ALL_METHODS if method.name == "ours_extract_and_add_encdec")
    concept_features_path = tmp_path / "concept_features.safetensors"
    plan = Stage4MethodPlan(
        method=method,
        target_name="encdec",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=[_seed_row()],
        coefficients=[0, 1],
        concept_features_path=concept_features_path,
    )
    loaded: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_concept_feature_load(*args: Any, **kwargs: Any) -> list[int]:
        loaded.append((args, kwargs))
        return [1, 2]

    monkeypatch.setattr(
        generation_mod,
        "_load_top_k_concept_fids",
        record_concept_feature_load,
    )
    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: {
            "final concepts": "1" * 64,
            "Stage 3e manifest": "2" * 64,
            "positive concept features": "3" * 64,
        }[surface],
    )

    payload = generation_mod._cluster_generation_inputs(
        cfg,
        cluster_contract=contract,
        concepts=[concept],
        plans=[plan],
        eligible_by_mode={
            SteeringMode.POSITIVE: [concept],
            SteeringMode.NEGATIVE: [],
        },
        seed_pool_fingerprint="4" * 64,
        consumed_eval_seeds=[_seed_row()],
        steering_scale_identity=None,
    )

    assert loaded == [
        (
            (concept_features_path, concept.concept_id),
            {
                "expected_tokenizer_id": cfg.extraction_tokenizer_id,
                "cluster_contract": target.cluster_contract,
                "expected_representative_id": concept.feature_index,
            },
        ),
    ]
    assert payload["artifact_sha256"] == {
        "concept_features:positive": "3" * 64,
        "concepts.json": "1" * 64,
        "stage4_contexts_manifest.json": "2" * 64,
    }
    assert payload["context_cache_manifest_sha256"] == {
        "encdec": generation_mod.sha256_bytes(
            generation_mod.canonical_json(provider.manifest).encode("utf-8")
        )
    }
    assert payload["steering_scale_identity"] is None
    assert payload["method_plans"][0]["coefficients"] == [0, 1]


def test_cluster_generation_inputs_executes_real_ours_artifact_identity_gate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The real clustered OURS preflight binds the row to its representative."""

    _set_base_dir(monkeypatch, tmp_path)
    _patch_bound_sources(monkeypatch)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["ours_extract_and_add_encdec"],
    )
    concept = replace(_concept(0), feature_index=1)
    contract = cast(Any, _FakeClusterContract())
    provider = _ContextProvider(metadata={}, segments=())
    target = TargetBundle(
        name="encdec",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=provider,
        cluster_contract=contract,
    )
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "ours_extract_and_add_encdec"
    )
    concept_features_path = tmp_path / "concept_features.safetensors"
    save_file(
        {
            "feature_ids": t.tensor([[1, -1, -1]], dtype=t.int32),
            "percentile_deltas": t.zeros(1, 3),
            "canonical_feature_id": t.tensor(
                [concept.feature_index],
                dtype=t.int32,
            ),
            "canonical_rank": t.tensor([0], dtype=t.int32),
        },
        str(concept_features_path),
        metadata={
            "mode": "canonical",
            "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
        },
    )
    plan = Stage4MethodPlan(
        method=method,
        target_name="encdec",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=[_seed_row()],
        coefficients=[0],
        concept_features_path=concept_features_path,
    )
    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: {
            "final concepts": "1" * 64,
            "Stage 3e manifest": "2" * 64,
            "positive concept features": "3" * 64,
        }[surface],
    )
    call_kwargs = {
        "cluster_contract": contract,
        "concepts": [concept],
        "plans": [plan],
        "eligible_by_mode": {
            SteeringMode.POSITIVE: [concept],
            SteeringMode.NEGATIVE: [],
        },
        "seed_pool_fingerprint": "4" * 64,
        "consumed_eval_seeds": [_seed_row()],
        "steering_scale_identity": None,
    }

    payload = generation_mod._cluster_generation_inputs(cfg, **call_kwargs)

    assert payload["artifact_sha256"]["concept_features:positive"] == "3" * 64

    real_loader = generation_mod._load_top_k_concept_fids

    def omit_representative_binding(
        concept_features_path: Path,
        concept_id: int,
        *,
        expected_tokenizer_id: str | None = None,
        cluster_contract: Any = None,
        expected_representative_id: int | None = None,
    ) -> list[int]:
        del expected_representative_id
        return real_loader(
            concept_features_path,
            concept_id,
            expected_tokenizer_id=expected_tokenizer_id,
            cluster_contract=cluster_contract,
        )

    monkeypatch.setattr(
        generation_mod,
        "_load_top_k_concept_fids",
        omit_representative_binding,
    )
    with pytest.raises(
        AxBenchBaseError,
        match="requires the expected Concept.feature_index representative",
    ):
        generation_mod._cluster_generation_inputs(cfg, **call_kwargs)


def test_cluster_context_manifest_hash_matches_loaded_provider_bytes(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        '{"metadata":{"feature_contract_fingerprint":"aaaaaaaa"}}\n',
        encoding="utf-8",
    )
    loaded = {"metadata": {"feature_contract_fingerprint": "aaaaaaaa"}}
    provider = SimpleNamespace(
        manifest_path=manifest_path,
        manifest=loaded,
    )
    bundle = TargetBundle(
        name="encdec",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=cast(Any, provider),
        cluster_contract=cast(Any, _FakeClusterContract()),
    )

    assert generation_mod._context_cache_manifest_sha256(bundle) == (
        generation_mod.sha256_bytes(manifest_path.read_bytes())
    )
    provider.manifest = {"metadata": {"feature_contract_fingerprint": "changed"}}
    with pytest.raises(AxBenchBaseError, match="changed after.*provider loaded"):
        generation_mod._context_cache_manifest_sha256(bundle)


def test_cluster_context_manifest_fused_read_rejects_aba_bytes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The bytes parsed must be the same bytes whose digest was frozen."""

    manifest_path = tmp_path / "manifest.json"
    original_bytes = b'{"metadata":{"version":"A"}}\n'
    transient_bytes = b'{"metadata":{"version":"B"}}\n'
    manifest_path.write_bytes(original_bytes)
    provider = SimpleNamespace(
        manifest_path=manifest_path,
        manifest={"metadata": {"version": "A"}},
    )
    bundle = TargetBundle(
        name="encdec",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=cast(Any, provider),
        cluster_contract=cast(Any, _FakeClusterContract()),
    )
    real_read_bytes = Path.read_bytes
    real_read_text = Path.read_text

    def _aba_read_bytes(path: Path) -> bytes:
        if path != manifest_path:
            return real_read_bytes(path)
        path.write_bytes(transient_bytes)
        payload = real_read_bytes(path)
        path.write_bytes(original_bytes)
        return payload

    def _aba_read_text(
        path: Path,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        if path != manifest_path:
            return real_read_text(path, encoding=encoding, errors=errors)
        path.write_bytes(transient_bytes)
        payload = real_read_text(path, encoding=encoding, errors=errors)
        path.write_bytes(original_bytes)
        return payload

    monkeypatch.setattr(Path, "read_bytes", _aba_read_bytes)
    monkeypatch.setattr(Path, "read_text", _aba_read_text)

    with pytest.raises(AxBenchBaseError, match="changed while.*being loaded"):
        generation_mod._context_cache_manifest_sha256(bundle)

    assert real_read_bytes(manifest_path) == original_bytes


def _validate_context_cache(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    metadata: dict[str, Any] | None = None,
    segments: tuple[Any, ...] | None = None,
    record_tokens: list[int] | None = None,
    cluster_contract: Any | None = None,
) -> _ContextProvider:
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_encdec"])
    model = _make_target_module()
    default_metadata = {
        "model_max_seq_len": _model_sequence_len(model),
        "contexts_fingerprint": "c" * 64,
    }
    if cluster_contract is not None:
        cluster_metadata = {
            "feature_contract_fingerprint": (cluster_contract.feature_contract_fingerprint)
        }
        monkeypatch.setattr(
            generation_mod,
            "clustered_context_cache_metadata_for_contract",
            lambda _contract: cluster_metadata,
        )
        default_metadata.update(cluster_metadata)
    provider = _ContextProvider(
        metadata=metadata or default_metadata,
        segments=segments,
    )
    bundle = TargetBundle(
        name="encdec",
        model=model,
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=provider,
        cluster_contract=cluster_contract,
    )
    seed = _seed_row()
    concept = _concept(0)
    method = next(method for method in ALL_METHODS if method.name == "diffmean_encdec")
    plan = Stage4MethodPlan(
        method=method,
        target_name="encdec",
        bundle=bundle,
        target_meta=bundle.meta,
        seeds_for_mode=[seed],
        coefficients=[1.0],
        concept_features_path=tmp_path / "unused.safetensors",
    )
    monkeypatch.setattr(
        generation_mod,
        "validate_stage4_contexts_for_generation",
        lambda *args, **kwargs: (
            {"contexts_fingerprint": "c" * 64},
            [
                {
                    "context_key": "context-key",
                    "token_ids": record_tokens or [1, 2, 3],
                }
            ],
        ),
    )
    _validate_stage4_context_cache(
        cfg,
        plans=[plan],
        concepts=[concept],
        eval_seeds_positive=[seed],
        eval_seeds_negative=[],
    )
    return provider


def test_stage4_context_cache_accepts_every_expected_valid_segment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    provider = _validate_context_cache(
        monkeypatch,
        tmp_path,
        segments=(
            ContextFeatureSegment(
                feature_ids=(2,),
                buckets=(1,),
                char_start=0,
                char_end=3,
            ),
        ),
    )
    assert provider.seen == [[1, 2, 3]]


def test_clustered_stage4_cache_preserves_sampler_order_and_rejects_nonreps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = _FakeClusterContract()
    provider = _validate_context_cache(
        monkeypatch,
        tmp_path,
        cluster_contract=contract,
        segments=(
            ContextFeatureSegment(
                feature_ids=(2, 1),
                buckets=(1, 0),
                char_start=0,
                char_end=3,
            ),
        ),
    )
    assert provider.seen == [[1, 2, 3]]

    with pytest.raises(
        FeatureClustersError,
        match="non-representative feature ID 3",
    ):
        _validate_context_cache(
            monkeypatch,
            tmp_path,
            cluster_contract=contract,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(3,),
                    buckets=(1,),
                    char_start=0,
                    char_end=3,
                ),
            ),
        )


def test_unclustered_stage4_cache_preserves_training_sampler_order(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Active-sampler order followed by inactive-sampler order is semantic."""

    provider = _validate_context_cache(
        monkeypatch,
        tmp_path,
        segments=(
            ContextFeatureSegment(
                feature_ids=(3720, 7806, 3345, 191, 4284),
                buckets=(0, 0, 0, 1, 1),
                char_start=0,
                char_end=3,
            ),
        ),
    )
    assert provider.seen == [[1, 2, 3]]


def _unchecked_context_feature_segment(
    *,
    feature_ids: tuple[Any, ...],
    buckets: tuple[Any, ...],
    char_start: Any,
    char_end: Any,
) -> ContextFeatureSegment:
    """Build malformed cache ingress without tripping constructor validation.

    These cases exercise Stage 4's defense-in-depth cache validator. The public
    constructor now rejects them earlier, so the fixture must model malformed
    persisted input explicitly instead of failing during test collection.
    """

    segment = object.__new__(ContextFeatureSegment)
    object.__setattr__(segment, "feature_ids", feature_ids)
    object.__setattr__(segment, "buckets", buckets)
    object.__setattr__(segment, "char_start", char_start)
    object.__setattr__(segment, "char_end", char_end)
    return segment


@pytest.mark.parametrize(
    ("metadata", "error_match"),
    [
        (
            {"model_max_seq_len": 129, "contexts_fingerprint": "c" * 64},
            "model_max_seq_len",
        ),
        (
            {"model_max_seq_len": 128, "contexts_fingerprint": "d" * 64},
            "contexts_fingerprint",
        ),
    ],
)
def test_stage4_context_cache_rejects_stale_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    metadata: dict[str, Any],
    error_match: str,
) -> None:
    with pytest.raises(AxBenchBaseError, match=error_match):
        _validate_context_cache(
            monkeypatch,
            tmp_path,
            metadata=metadata,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(2,),
                    buckets=(1,),
                    char_start=0,
                    char_end=3,
                ),
            ),
        )


@pytest.mark.parametrize(
    ("segments", "error_match"),
    [
        (None, "missing a non-empty segment tuple"),
        (
            (
                ContextFeatureSegment(
                    feature_ids=(2, 2),
                    buckets=(1, 1),
                    char_start=0,
                    char_end=3,
                ),
            ),
            "structurally invalid segment",
        ),
        (
            (
                _unchecked_context_feature_segment(
                    feature_ids=(True,),
                    buckets=(1,),
                    char_start=0,
                    char_end=3,
                ),
            ),
            "structurally invalid segment",
        ),
        (
            (
                _unchecked_context_feature_segment(
                    feature_ids=(2,),
                    buckets=(False,),
                    char_start=0,
                    char_end=3,
                ),
            ),
            "structurally invalid segment",
        ),
        (
            (
                _unchecked_context_feature_segment(
                    feature_ids=(2,),
                    buckets=(1,),
                    char_start=False,
                    char_end=3,
                ),
            ),
            "structurally invalid segment",
        ),
        (
            (
                _unchecked_context_feature_segment(
                    feature_ids=(2,),
                    buckets=(1,),
                    char_start=0,
                    char_end=True,
                ),
            ),
            "structurally invalid segment",
        ),
    ],
)
def test_stage4_context_cache_rejects_missing_or_invalid_segments(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    segments: tuple[Any, ...] | None,
    error_match: str,
) -> None:
    with pytest.raises(AxBenchBaseError, match=error_match):
        _validate_context_cache(
            monkeypatch,
            tmp_path,
            segments=segments,
        )


@pytest.mark.parametrize(
    "segment_kwargs",
    [
        {"feature_ids": (True,), "buckets": (1,), "char_start": 0, "char_end": 3},
        {"feature_ids": (2,), "buckets": (False,), "char_start": 0, "char_end": 3},
        {"feature_ids": (2,), "buckets": (1,), "char_start": False, "char_end": 3},
        {"feature_ids": (2,), "buckets": (1,), "char_start": 0, "char_end": True},
    ],
)
def test_context_feature_segment_rejects_bool_typed_fields_at_construction(
    segment_kwargs: dict[str, Any],
) -> None:
    """Bool-typed fields are rejected at construction before validation."""
    with pytest.raises(BeartypeCallHintParamViolation):
        ContextFeatureSegment(**segment_kwargs)


def test_stage4_context_cache_accepts_reverse_alignment_within_window(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Preflight mirrors the consumer's valid tail-clipping behaviour."""

    provider = _validate_context_cache(
        monkeypatch,
        tmp_path,
        segments=(
            ContextFeatureSegment(
                feature_ids=(2,),
                buckets=(1,),
                char_start=0,
                char_end=12,
            ),
        ),
    )
    assert provider.seen == [[1, 2, 3]]


def test_stage4_context_cache_rejects_unalignable_segments_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """Large coverage drift must fail during preflight, not first generation."""

    with pytest.raises(AxBenchBaseError, match="cannot be aligned.*before generation"):
        _validate_context_cache(
            monkeypatch,
            tmp_path,
            segments=(
                ContextFeatureSegment(
                    feature_ids=(2,),
                    buckets=(1,),
                    char_start=0,
                    char_end=3,
                ),
            ),
            record_tokens=list(range(20)),
        )


def _make_targets(
    target_models: dict[str, nn.Module],
    metas: dict[str, TargetModelMetadata],
    tokenizers: dict[str, Any],
    *,
    include_encdec_provider: bool = True,
) -> list[TargetBundle]:
    """Build a list of :class:`TargetBundle` from the old dict-style args."""
    return [
        TargetBundle(
            name=name,
            model=target_models[name],
            meta=metas[name],
            tokenizer=tokenizers[name],
            context_provider=_ContextProvider(metadata={}, segments=())
            if include_encdec_provider and name == "encdec"
            else None,
        )
        for name in target_models
    ]


def _patch_run_intervention(
    monkeypatch: pytest.MonkeyPatch,
    *,
    factory: Any | None = None,
) -> MagicMock:
    """Patch the module-level ``run_intervention`` with a tracked mock.

    ``factory`` lets callers supply a per-call generator of
    ``GenerationResult`` outputs; defaults to a fixed result per call.
    """
    if factory is None:
        mock = MagicMock(
            side_effect=itertools.cycle(
                [
                    GenerationResult(
                        text="gen",
                        token_count=3,
                        stop_reason="bos_token",
                    ),
                ],
            ),
        )
    else:
        mock = MagicMock(side_effect=factory)
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.run_intervention",
        mock,
    )
    return mock


# ---------------------------------------------------------------------------
# Test fixtures (pytest)
# ---------------------------------------------------------------------------


_TEST_SEED_FINGERPRINT = "f" * 64
_REAL_BIND_CLUSTER_GENERATION_SOURCES = generation_mod._bind_cluster_generation_sources


@pytest.fixture(autouse=True)
def _strict_stage4_prerequisites(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep outer-loop tests focused while preserving strict production gates."""

    monkeypatch.setattr(
        generation_mod,
        "load_seed_pool_fingerprint",
        lambda _run_tag: _TEST_SEED_FINGERPRINT,
    )
    monkeypatch.setattr(
        generation_mod,
        "_bind_cluster_generation_sources",
        lambda *_args, **_kwargs: ("1" * 64, _TEST_SEED_FINGERPRINT),
    )
    monkeypatch.setattr(
        generation_mod,
        "_validate_stage4_context_cache",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_mod,
        "load_prompting_descriptions",
        lambda _cfg, _mode: {index: f"description {index}" for index in range(10_000)},
    )


@pytest.fixture
def tokenizers() -> dict[str, Any]:
    tk = _Tokenizer()
    return {"decoder_only": tk, "encdec": tk}


@pytest.fixture
def target_models() -> dict[str, nn.Module]:
    return {
        "decoder_only": _make_target_module(),
        "encdec": _make_target_module(),
    }


@pytest.mark.parametrize(
    "failure_point",
    [
        None,
        "context_cache",
        "generation_inputs",
        "resume_validation",
        "resume_metadata_read",
        "marker_commit",
    ],
)
def test_clustered_run_preflight_finishes_before_first_forward(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure_point: str | None,
) -> None:
    """Mutation-proof the exact clustered preflight-to-forward ordering."""

    _set_base_dir(monkeypatch, tmp_path)
    events: list[str] = []
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    contract = cast(Any, _FakeClusterContract())
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
        checkpoint_files_pin=_fake_decoder_checkpoint_pin(),
    )

    def load_manifest(_path: Path) -> dict[str, str]:
        events.append("manifest")
        return {
            "contexts_fingerprint": "c" * 64,
            "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        }

    monkeypatch.setattr(generation_mod, "load_stage4_contexts_manifest", load_manifest)

    def _maybe_raise(point: str, result: Any = None) -> Any:
        events.append(point)
        if failure_point == point:
            raise AxBenchBaseError(f"injected preflight failure: {point}")
        return result

    monkeypatch.setattr(
        generation_mod,
        "_validate_stage4_context_cache",
        lambda *args, **kwargs: _maybe_raise("context_cache", None),
    )
    monkeypatch.setattr(
        generation_mod,
        "_cluster_generation_inputs",
        lambda *args, **kwargs: _maybe_raise(
            "generation_inputs",
            {
                "schema_version": 1,
                "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
                "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
            },
        ),
    )
    monkeypatch.setattr(
        generation_mod,
        "validate_cluster_generation_resume",
        lambda *args, **kwargs: _maybe_raise("resume_validation"),
    )
    monkeypatch.setattr(
        generation_mod,
        "_load_existing_generation_metadata",
        lambda *args, **kwargs: _maybe_raise("resume_metadata_read", {}),
    )
    scan_number = 0

    def scan(_path: Path) -> list[int]:
        nonlocal scan_number
        scan_number += 1
        events.append(f"resume_scan_{scan_number}")
        return []

    monkeypatch.setattr(generation_mod, "_scan_batch_indices", scan)
    monkeypatch.setattr(
        generation_mod,
        "commit_cluster_generation_marker",
        lambda *args, **kwargs: _maybe_raise("marker_commit"),
    )

    class RecordingWriter:
        def __init__(self, **_kwargs: Any) -> None:
            events.append("writer_init")

        def __enter__(self) -> RecordingWriter:
            events.append("writer_enter")
            return self

        def __exit__(self, *_args: Any) -> None:
            events.append("writer_exit")

        def append(self, _row: dict[str, Any]) -> None:
            events.append("writer_append")

    monkeypatch.setattr(generation_mod, "ParquetAppendWriter", RecordingWriter)

    def forward(*_args: Any, **_kwargs: Any) -> GenerationResult:
        events.append("forward")
        return GenerationResult(
            text="gen",
            token_count=3,
            stop_reason="bos_token",
        )

    monkeypatch.setattr(generation_mod, "run_intervention", forward)
    expected = [
        "manifest",
        "context_cache",
        "generation_inputs",
        "resume_validation",
        "resume_scan_1",
        "resume_metadata_read",
        "resume_scan_2",
        "marker_commit",
        "writer_init",
        "writer_enter",
        "forward",
        "writer_append",
        "writer_exit",
    ]

    if failure_point is None:
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[target],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )
        assert events == expected
    else:
        with pytest.raises(AxBenchBaseError, match="injected preflight failure"):
            generate_all(
                cfg,
                concepts=[_concept(0)],
                targets=[target],
                eval_seeds_positive=[_seed_row()],
                eval_seeds_negative=[],
            )
        assert events == expected[: expected.index(failure_point) + 1]


def test_clustered_generation_binds_cached_concepts_and_eval_seeds_to_marker(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """A separately valid disk snapshot B cannot stamp cached runtime inputs A."""

    _set_base_dir(monkeypatch, tmp_path)
    stats_path = tmp_path / "feature_statistics.safetensors"
    save_file(
        {"count": t.zeros(16_384, dtype=t.int64)},
        str(stats_path),
        metadata={"n_sae_features": "16384"},
    )
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["control_decoder_only"],
    ).replace(
        feature_statistics_path=str(stats_path),
        seed_continuation_nanochat_tokens=4,
        seed_prefix_max_nanochat_tokens=16,
    )
    assert cfg.run_tag is not None
    tokenizer = _ArtifactTokenizer()
    concept_a = _concept(0, "cached concept A")
    seed_a = _artifact_seed_row(doc_idx=1, prefix="cached-A")
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_path.write_text(
        json.dumps([asdict(concept_a)]),
        encoding="utf-8",
    )
    write_committed_seed_pair(
        cfg,
        tokenizer=tokenizer,
        train_rows=[seed_a],
        eval_rows=[seed_a],
    )
    monkeypatch.setattr(
        runtime_state_mod,
        "_nanochat_get_tokenizer",
        lambda: tokenizer,
    )
    state = RuntimeState(cfg=cfg)
    cached_concepts = state.concepts()
    cached_eval_seeds = state.seeds_eval()

    concept_b = _concept(0, "separately valid swapped concept B")
    seed_b = _artifact_seed_row(doc_idx=2, prefix="swapped-B")
    concepts_path.write_text(
        json.dumps([asdict(concept_b)]),
        encoding="utf-8",
    )
    write_committed_seed_pair(
        cfg.replace(seed_pool_seed=cfg.seed_pool_seed + 1),
        tokenizer=tokenizer,
        train_rows=[seed_b],
        eval_rows=[seed_b],
    )
    monkeypatch.setattr(seed_contract_mod, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(
        generation_mod,
        "load_seed_pool_fingerprint",
        load_seed_pool_fingerprint,
    )
    monkeypatch.setattr(
        generation_mod,
        "_bind_cluster_generation_sources",
        _REAL_BIND_CLUSTER_GENERATION_SOURCES,
    )

    contract = cast(Any, _FakeClusterContract())
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
        checkpoint_files_pin=_fake_decoder_checkpoint_pin(),
    )
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "control_decoder_only"
    )
    plan = Stage4MethodPlan(
        method=method,
        target_name="decoder_only",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=cached_eval_seeds,
        coefficients=[0],
        concept_features_path=tmp_path / "unused.safetensors",
    )
    contexts_fingerprint = "c" * 64
    manifest_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_manifest()
    manifest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "contexts_fingerprint": contexts_fingerprint,
            "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "_build_stage4_method_plans",
        lambda *args, **kwargs: [plan],
    )
    monkeypatch.setattr(
        generation_mod,
        "_checkpoint_refs_for_plans",
        lambda *_args, **_kwargs: {
            "decoder_only": {"source": "test", "model_tag": "test", "step": 7}
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "validate_cluster_generation_resume",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_mod,
        "_load_existing_generation_metadata",
        lambda *args, **kwargs: {},
    )
    marker_inputs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        generation_mod,
        "commit_cluster_generation_marker",
        lambda *args, generation_inputs, **kwargs: marker_inputs.append(generation_inputs),
    )
    forward = _patch_run_intervention(monkeypatch)

    try:
        generate_all(
            cfg,
            concepts=cached_concepts,
            targets=[target],
            eval_seeds_positive=cached_eval_seeds,
            eval_seeds_negative=[],
        )
    except AxBenchBaseError as exc:
        message = str(exc)
        assert "do not match the in-memory inputs consumed by generation" in message
        assert "concepts.json" in message
        assert "seeds_eval.safetensors" in message
    else:
        assert marker_inputs
        committed = marker_inputs[0]
        pytest.fail(
            "stale snapshot B was accepted: "
            f"marker concepts_sha256="
            f"{committed['artifact_sha256']['concepts.json']}; "
            f"marker concepts={committed['concepts']}; "
            f"marker seed_pool_fingerprint="
            f"{committed['seed_pool_fingerprint']}; "
            f"consumed concepts={[asdict(concept) for concept in cached_concepts]}; "
            f"consumed eval seeds={cached_eval_seeds}; "
            f"forwards={forward.call_count}"
        )
    finally:
        state.close()


def test_clustered_run_rejects_context_fingerprint_change_during_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The Stage 3e identity must agree before any input hash or marker write."""

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    contract = cast(Any, _FakeClusterContract())
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
    )
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "contexts_fingerprint": "c" * 64,
            "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "_validate_stage4_context_cache",
        lambda *args, **kwargs: "d" * 64,
    )
    input_spy = MagicMock(
        return_value={
            "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
        }
    )
    marker_spy = MagicMock()
    writer_spy = MagicMock()
    forward_spy = _patch_run_intervention(monkeypatch)
    monkeypatch.setattr(
        generation_mod,
        "_cluster_generation_inputs",
        input_spy,
    )
    monkeypatch.setattr(
        generation_mod,
        "validate_cluster_generation_resume",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_mod,
        "_load_existing_generation_metadata",
        lambda *args, **kwargs: {},
    )
    monkeypatch.setattr(
        generation_mod,
        "commit_cluster_generation_marker",
        marker_spy,
    )
    monkeypatch.setattr(generation_mod, "ParquetAppendWriter", writer_spy)

    with pytest.raises(
        AxBenchBaseError,
        match="context identity changed during clustered preflight",
    ):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[target],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )

    input_spy.assert_not_called()
    marker_spy.assert_not_called()
    writer_spy.assert_not_called()
    forward_spy.assert_not_called()


def test_clustered_run_rejects_batch_change_during_resume_metadata_read(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """The run-level scan bracket blocks marker, writer, and forwarding."""

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    contract = cast(Any, _FakeClusterContract())
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
    )
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "contexts_fingerprint": "c" * 64,
            "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "_cluster_generation_inputs",
        lambda *args, **kwargs: {
            "schema_version": 1,
            "feature_contract_fingerprint": (contract.feature_contract_fingerprint),
            "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "validate_cluster_generation_resume",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_mod,
        "_load_existing_generation_metadata",
        lambda *args, **kwargs: {},
    )
    scan = MagicMock(side_effect=[[], [0]])
    monkeypatch.setattr(generation_mod, "_scan_batch_indices", scan)
    marker_spy = MagicMock()
    writer_spy = MagicMock()
    forward_spy = _patch_run_intervention(monkeypatch)
    monkeypatch.setattr(
        generation_mod,
        "commit_cluster_generation_marker",
        marker_spy,
    )
    monkeypatch.setattr(generation_mod, "ParquetAppendWriter", writer_spy)

    with pytest.raises(
        AxBenchBaseError,
        match="generation batches changed while resume metadata was being loaded",
    ):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=[target],
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )

    assert scan.call_count == 2
    marker_spy.assert_not_called()
    writer_spy.assert_not_called()
    forward_spy.assert_not_called()


def test_unclustered_generate_all_uses_exact_legacy_writer_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["control_decoder_only"])
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
    )
    original_writer = generation_mod.ParquetAppendWriter
    writer_kwargs: list[dict[str, Any]] = []

    class _WriterSpy:
        def __init__(self, **kwargs: Any) -> None:
            writer_kwargs.append(dict(kwargs))
            self._inner = original_writer(**kwargs)

        def __enter__(self):
            return self._inner.__enter__()

        def __exit__(self, *args: Any) -> None:
            self._inner.__exit__(*args)

    monkeypatch.setattr(generation_mod, "ParquetAppendWriter", _WriterSpy)
    forward_spy = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=[target],
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[],
    )

    assert forward_spy.call_count == 1
    assert len(writer_kwargs) == 1
    assert writer_kwargs[0]["immutable_batches"] is False
    assert writer_kwargs[0]["expected_existing_indices"] is None


def test_clustered_steering_scale_identity_agrees_across_both_paths(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_decoder_only"])
    contract = cast(Any, _FakeClusterContract())
    concept = _concept(1)
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
        checkpoint_files_pin=_fake_decoder_checkpoint_pin(),
    )
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "diffmean_decoder_only"
    )
    plan = Stage4MethodPlan(
        method=method,
        target_name="decoder_only",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=[_seed_row()],
        coefficients=[0.2],
        concept_features_path=tmp_path / "unused.safetensors",
        cluster_source_corpus_sha256="7" * 64,
    )
    monkeypatch.setattr(
        generation_mod,
        "load_stage4_contexts_manifest",
        lambda _path: {
            "contexts_fingerprint": "c" * 64,
            "feature_contract_fingerprint": contract.feature_contract_fingerprint,
        },
    )
    monkeypatch.setattr(
        generation_mod,
        "_build_stage4_method_plans",
        lambda *args, **kwargs: [plan],
    )
    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: {
            "final concepts": "1" * 64,
            "Stage 3e manifest": "2" * 64,
            "diffmean_decoder_only direction for concept 1": "3" * 64,
        }[surface],
    )
    scalar_loads: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_scalar_load(*args: Any, **kwargs: Any) -> None:
        scalar_loads.append((args, kwargs))

    monkeypatch.setattr(
        generation_mod,
        "_load_scalar_direction_artifact",
        record_scalar_load,
    )
    monkeypatch.setattr(
        generation_mod,
        "validate_cluster_generation_resume",
        lambda *args, **kwargs: None,
    )
    monkeypatch.setattr(
        generation_mod,
        "_load_existing_generation_metadata",
        lambda *args, **kwargs: {},
    )
    marker_inputs: list[dict[str, Any]] = []
    monkeypatch.setattr(
        generation_mod,
        "commit_cluster_generation_marker",
        lambda *args, generation_inputs, **kwargs: marker_inputs.append(generation_inputs),
    )
    enforced_identities: list[str] = []
    monkeypatch.setattr(
        generation_mod,
        "enforce_steering_scale_identity",
        lambda _path, *, identity, resume_active: enforced_identities.append(identity),
    )
    _patch_run_intervention(monkeypatch)

    preflight = MagicMock()
    preflight.identity = "dirmaxact:distinct-two-path-proof"
    preflight_spy = MagicMock(return_value=preflight)
    monkeypatch.setattr(
        generation_mod,
        "preflight_direction_scales",
        preflight_spy,
    )
    generate_all(
        cfg,
        concepts=[concept],
        targets=[target],
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[],
        apply_max_act_scale=True,
    )

    preflight_spy.assert_called_once()
    assert len(marker_inputs) == 1
    assert marker_inputs[0]["steering_scale_identity"] == (
        "dirmaxact:distinct-two-path-proof"
    )
    assert enforced_identities == ["dirmaxact:distinct-two-path-proof"]
    assert marker_inputs[0]["steering_scale_identity"] == enforced_identities[0]
    assert (
        scalar_loads
        == [
            (
                (
                    paths.AxBenchBasePaths.from_config(cfg).diffmean_direction(
                        "decoder_only",
                        SteeringMode.POSITIVE,
                        concept.concept_id,
                    ),
                ),
                {
                    "method": method,
                    "concept": concept,
                    "target_model_name": "decoder_only",
                    "layer": cfg.layer_hooks.diffmean_and_probe_layer,
                    "output_paths": paths.AxBenchBasePaths.from_config(cfg),
                    "cluster_contract": contract,
                    "source_corpus_sha256": "7" * 64,
                },
            ),
        ]
        * 2
    )


def test_cluster_generation_inputs_rejects_missing_checkpoint_ref_key(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["control_decoder_only"],
    ).replace(
        decoder_only={"source": "base", "step": 7},
    )
    contract = cast(Any, _FakeClusterContract())
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=contract,
    )
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "control_decoder_only"
    )
    plan = Stage4MethodPlan(
        method=method,
        target_name="decoder_only",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=[_seed_row()],
        coefficients=[None],
        concept_features_path=tmp_path / "unused.safetensors",
    )
    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: "1" * 64,
    )

    with pytest.raises(
        AxBenchBaseError,
        match="checkpoint reference.*missing.*model_tag",
    ):
        generation_mod._cluster_generation_inputs(
            cfg,
            cluster_contract=contract,
            concepts=[_concept(0)],
            plans=[plan],
            eligible_by_mode={
                SteeringMode.POSITIVE: [_concept(0)],
                SteeringMode.NEGATIVE: [],
            },
            seed_pool_fingerprint="f" * 64,
            consumed_eval_seeds=[_seed_row()],
            steering_scale_identity=None,
        )


def test_cluster_generation_inputs_fail_loud_on_missing_run_tag_pin_and_scalar_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    contract = cast(Any, _FakeClusterContract())
    _patch_bound_sources(monkeypatch)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["diffmean_decoder_only"],
    )
    concept = _concept(1)
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "diffmean_decoder_only"
    )

    def plan_for(
        *,
        pin: Any = None,
        corpus_sha256: str | None = "7" * 64,
    ) -> Stage4MethodPlan:
        target = TargetBundle(
            name="decoder_only",
            model=_make_target_module(),
            meta=_meta(),
            tokenizer=_Tokenizer(),
            context_provider=None,
            cluster_contract=contract,
            checkpoint_files_pin=pin,
        )
        return Stage4MethodPlan(
            method=method,
            target_name="decoder_only",
            bundle=target,
            target_meta=target.meta,
            seeds_for_mode=[_seed_row()],
            coefficients=[0.2],
            concept_features_path=tmp_path / "unused.safetensors",
            cluster_source_corpus_sha256=corpus_sha256,
        )

    kwargs = {
        "cluster_contract": contract,
        "concepts": [concept],
        "eligible_by_mode": {
            SteeringMode.POSITIVE: [concept],
            SteeringMode.NEGATIVE: [],
        },
        "seed_pool_fingerprint": "4" * 64,
        "consumed_eval_seeds": [_seed_row()],
        "steering_scale_identity": None,
    }
    with pytest.raises(AxBenchBaseError, match="identity requires cfg.run_tag"):
        generation_mod._cluster_generation_inputs(
            cfg.replace(run_tag=None),
            plans=[plan_for(pin=_fake_decoder_checkpoint_pin())],
            **kwargs,
        )

    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: "1" * 64,
    )
    scalar_loads: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

    def record_scalar_load(*args: Any, **kwargs: Any) -> None:
        scalar_loads.append((args, kwargs))

    monkeypatch.setattr(
        generation_mod,
        "_load_scalar_direction_artifact",
        record_scalar_load,
    )
    with pytest.raises(AxBenchBaseError, match="lacks frozen source-corpus"):
        generation_mod._cluster_generation_inputs(
            cfg,
            plans=[plan_for(pin=_fake_decoder_checkpoint_pin(), corpus_sha256=None)],
            **kwargs,
        )
    with pytest.raises(AxBenchBaseError, match="direction-scale identity"):
        generation_mod._cluster_generation_inputs(
            cfg,
            plans=[plan_for(pin=_fake_decoder_checkpoint_pin())],
            **kwargs,
        )
    assert scalar_loads == [
        (
            (
                paths.AxBenchBasePaths.from_config(cfg).diffmean_direction(
                    "decoder_only",
                    SteeringMode.POSITIVE,
                    concept.concept_id,
                ),
            ),
            {
                "method": method,
                "concept": concept,
                "target_model_name": "decoder_only",
                "layer": cfg.layer_hooks.diffmean_and_probe_layer,
                "output_paths": paths.AxBenchBasePaths.from_config(cfg),
                "cluster_contract": contract,
                "source_corpus_sha256": "7" * 64,
            },
        ),
    ]

    with pytest.raises(AxBenchBaseError, match="one exact loaded checkpoint pin"):
        generation_mod._cluster_generation_inputs(
            cfg.replace(enabled_methods=["control_decoder_only"]),
            plans=[
                Stage4MethodPlan(
                    method=next(
                        candidate
                        for candidate in ALL_METHODS
                        if candidate.name == "control_decoder_only"
                    ),
                    target_name="decoder_only",
                    bundle=plan_for().bundle,
                    target_meta=_meta(),
                    seeds_for_mode=[_seed_row()],
                    coefficients=[None],
                    concept_features_path=tmp_path / "unused.safetensors",
                )
            ],
            cluster_contract=contract,
            concepts=[],
            eligible_by_mode={
                SteeringMode.POSITIVE: [],
                SteeringMode.NEGATIVE: [],
            },
            seed_pool_fingerprint="4" * 64,
            consumed_eval_seeds=[_seed_row()],
            steering_scale_identity=None,
        )


def test_cluster_generation_inputs_rejects_unsupported_method_family(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _make_cfg(tmp_path, enabled_methods=["prompting_decoder_only"])
    _patch_bound_sources(monkeypatch)
    target = TargetBundle(
        name="decoder_only",
        model=_make_target_module(),
        meta=_meta(),
        tokenizer=_Tokenizer(),
        context_provider=None,
        cluster_contract=cast(Any, _FakeClusterContract()),
        checkpoint_files_pin=_fake_decoder_checkpoint_pin(),
    )
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "prompting_decoder_only"
    )
    plan = Stage4MethodPlan(
        method=method,
        target_name="decoder_only",
        bundle=target,
        target_meta=target.meta,
        seeds_for_mode=[_seed_row()],
        coefficients=[None],
        concept_features_path=tmp_path / "unused.safetensors",
    )
    monkeypatch.setattr(
        generation_mod,
        "_stable_cluster_artifact_sha256",
        lambda _path, *, surface: "1" * 64,
    )

    with pytest.raises(AxBenchBaseError, match="unsupported method family"):
        generation_mod._cluster_generation_inputs(
            cfg,
            cluster_contract=cast(Any, _FakeClusterContract()),
            concepts=[_concept(0)],
            plans=[plan],
            eligible_by_mode={
                SteeringMode.POSITIVE: [_concept(0)],
                SteeringMode.NEGATIVE: [],
            },
            seed_pool_fingerprint="4" * 64,
            consumed_eval_seeds=[_seed_row()],
            steering_scale_identity=None,
        )


def test_decoder_checkpoint_refs_bind_loaded_bytes_and_resolved_latest_step(
    tmp_path: Path,
) -> None:
    checkpoint_dir = tmp_path / "base_checkpoints" / "d-test"
    checkpoint_dir.mkdir(parents=True)
    model_path = checkpoint_dir / "model_000007.pt"
    (checkpoint_dir / "meta_000007.json").write_text("{}", encoding="utf-8")
    t.save({"weight": t.tensor([1.0])}, model_path)
    pin_a, _metadata, _state = resolve_checkpoint_files_pin(
        checkpoint_dir,
        step=None,
        resolved_tag="d-test",
    )
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["control_decoder_only"],
    ).replace(
        decoder_only={"source": "base", "model_tag": "d-test", "step": None},
    )
    method = next(
        candidate for candidate in ALL_METHODS if candidate.name == "control_decoder_only"
    )

    def refs_for(pin: Any) -> dict[str, dict[str, Any]]:
        target = TargetBundle(
            name="decoder_only",
            model=_make_target_module(),
            meta=_meta(),
            tokenizer=_Tokenizer(),
            context_provider=None,
            cluster_contract=cast(Any, _FakeClusterContract()),
            checkpoint_files_pin=pin,
        )
        plan = Stage4MethodPlan(
            method=method,
            target_name="decoder_only",
            bundle=target,
            target_meta=target.meta,
            seeds_for_mode=[_seed_row()],
            coefficients=[None],
            concept_features_path=tmp_path / "unused.safetensors",
        )
        return generation_mod._checkpoint_refs_for_plans(cfg, [plan])

    refs_a = refs_for(pin_a)
    t.save({"weight": t.tensor([2.0])}, model_path)
    with pytest.raises(
        ClusterFeatureContractError,
        match="checkpoint model bytes changed after they were pinned",
    ):
        assert_checkpoint_files_pin_unchanged(pin_a)
    pin_b, _metadata, _state = resolve_checkpoint_files_pin(
        checkpoint_dir,
        step=None,
        resolved_tag="d-test",
    )
    refs_b = refs_for(pin_b)

    assert refs_a["decoder_only"]["step"] == 7
    assert refs_b["decoder_only"]["step"] == 7
    assert refs_a != refs_b
    assert (
        refs_a["decoder_only"]["loaded_checkpoint"]["model_sha256"]
        != refs_b["decoder_only"]["loaded_checkpoint"]["model_sha256"]
    )


# ---------------------------------------------------------------------------
# 1. Default — iterate all methods × concepts × seeds × coefficients
# ---------------------------------------------------------------------------


def test_iterates_all_methods_when_enabled_methods_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """One concept, one seed per mode, negative_selection non-None on both
    targets → every method's grid is fully expanded.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path)
    metas = {
        "decoder_only": _meta(negative_selection={"strategy": "uniform"}),
        "encdec": _meta(negative_selection={"strategy": "uniform"}),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]
    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    # Expected call count: sum over all methods of their grid size
    # (single seed, single concept).
    expected_calls = 0
    for method in ALL_METHODS:
        coefs = method.coefficient_grid_fn(metas[method.target_model], cfg)
        expected_calls += len(coefs)
    assert mock.call_count == expected_calls


# ---------------------------------------------------------------------------
# 2. Methods requiring negative sampling are skipped on a non-neg target
# ---------------------------------------------------------------------------


def test_skips_methods_requiring_negative_sampling_on_target(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """encdec target has ``negative_selection is None`` → the 3
    ``requires_negative_sampling=True`` methods are filtered out.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path)
    metas = {
        "decoder_only": _meta(negative_selection=None),
        "encdec": _meta(negative_selection=None),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row()]
    neg_seeds = [_seed_row()]
    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    # Collect method names that actually got called.
    called_method_names = {call.kwargs["method"].name for call in mock.call_args_list}
    # The three requires_negative_sampling methods are filtered.
    forbidden = {
        "neg_ours_add_to_negatives_encdec",
        "neg_ours_replace_random_negatives_encdec",
        "neg_ours_replace_all_negatives_encdec",
    }
    assert forbidden.isdisjoint(called_method_names)
    # And every other method is called at least once.
    remaining = {m.name for m in ALL_METHODS} - forbidden
    assert remaining == called_method_names


# ---------------------------------------------------------------------------
# 3. Resume skips already-done generation_keys
# ---------------------------------------------------------------------------


def test_already_done_skip_on_resume(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Pre-populate the generations dir with 2 rows for one method;
    re-run and verify those rows are skipped.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_decoder_only"])
    assert cfg.run_tag is not None
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]

    # The diffmean_scalar_grid has 10 values. We'll pre-populate the
    # first 3 as already-done.
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    diffmean_coefs = [0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=10,
    ) as writer:
        for coef in diffmean_coefs[:3]:
            writer.append(
                {
                    "generation_key": build_generation_key(
                        method_name="diffmean_decoder_only",
                        concept_id=0,
                        eval_seed=0,
                        coefficient=coef,
                        target_model_name="decoder_only",
                    ),
                    "concept_id": 0,
                    "method_name": "diffmean_decoder_only",
                    "mode": "positive",
                    "eval_seed": 0,
                    "coefficient": str(coef),
                    "generation_text": "prev",
                    "target_model": "decoder_only",
                    "prefix_text": pos_seeds[0].prefix_text,
                    "prompt_token_key": prompt_token_key([1, 1, 2, 3]),
                    "prompt_token_count": 4,
                    "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
                    "token_count": 2,
                    "stop_reason": "bos_token",
                },
            )

    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    # 10 total coefficients in the grid, 3 already done → 7 new calls.
    assert mock.call_count == 7
    # Existing rows still present.
    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.num_rows == 10
    all_keys = set(table.column("generation_key").to_pylist())
    assert len(all_keys) == 10


@pytest.mark.parametrize(
    ("stale_field", "stale_value"),
    [
        pytest.param("prefix_text", "stale prefix", id="prefix-text"),
        pytest.param(
            "prompt_token_key",
            prompt_token_key([1, 99, 2, 3]),
            id="same-text-different-exact-token-ids",
        ),
        pytest.param("prompt_token_count", 5, id="prompt-token-count"),
        pytest.param(
            "seed_pool_fingerprint",
            "e" * 64,
            id="seed-pool-fingerprint",
        ),
    ],
)
def test_resume_rejects_matching_key_with_different_exact_identity(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
    stale_field: str,
    stale_value: Any,
) -> None:
    """Index-based keys must not hide any stale prompt/seed identity."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_decoder_only"])
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    current_seed = _seed_row(doc_idx=0, text="current prefix")
    existing_identity: dict[str, Any] = {
        "prefix_text": current_seed.prefix_text,
        "prompt_token_key": prompt_token_key([1, 1, 2, 3]),
        "prompt_token_count": 4,
        "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
    }
    existing_identity[stale_field] = stale_value
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=10,
    ) as writer:
        writer.append(
            {
                "generation_key": build_generation_key(
                    method_name="diffmean_decoder_only",
                    concept_id=0,
                    eval_seed=0,
                    coefficient=0.2,
                    target_model_name="decoder_only",
                ),
                "concept_id": 0,
                "method_name": "diffmean_decoder_only",
                "mode": "positive",
                "eval_seed": 0,
                "coefficient": "0.2",
                "generation_text": "prev",
                "target_model": "decoder_only",
                "prefix_text": existing_identity["prefix_text"],
                "prompt_token_key": existing_identity["prompt_token_key"],
                "prompt_token_count": existing_identity["prompt_token_count"],
                "seed_pool_fingerprint": existing_identity["seed_pool_fingerprint"],
                "token_count": 2,
                "stop_reason": "bos_token",
            },
        )

    mock = _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="exact resume identity"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=[current_seed],
            eval_seeds_negative=[],
        )

    assert mock.call_count == 0


@pytest.mark.parametrize(
    ("malformed_field", "malformed_value"),
    [
        pytest.param("generation_key", None, id="null-generation-key"),
        pytest.param("generation_key", "wrong-key", id="inconsistent-key"),
        pytest.param("concept_id", -1, id="negative-concept-id"),
        pytest.param("method_name", "unknown_method", id="unknown-method"),
        pytest.param("mode", "negative", id="wrong-mode"),
        pytest.param("eval_seed", -1, id="negative-eval-seed"),
        pytest.param("coefficient", None, id="null-coefficient"),
        pytest.param("generation_text", None, id="null-generation-text"),
        pytest.param("target_model", "encdec", id="wrong-target-model"),
        pytest.param("prefix_text", None, id="null-prefix-text"),
        pytest.param("prompt_token_key", "not-a-token-digest", id="prompt-key"),
        pytest.param("prompt_token_count", 0, id="prompt-count"),
        pytest.param("seed_pool_fingerprint", "not-a-digest", id="seed-fingerprint"),
        pytest.param("token_count", -1, id="negative-token-count"),
        pytest.param("stop_reason", "unknown", id="unknown-stop-reason"),
    ],
)
def test_resume_rejects_malformed_exact_identity_before_generation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    malformed_field: str,
    malformed_value: Any,
) -> None:
    """Nullable Arrow fields must not turn corrupt rows into resume sentinels."""
    _set_base_dir(monkeypatch, tmp_path)
    identity: dict[str, Any] = {
        "generation_key": "diffmean_decoder_only|c0|s0|k0.2|decoder_only",
        "concept_id": 0,
        "method_name": "diffmean_decoder_only",
        "mode": "positive",
        "eval_seed": 0,
        "coefficient": "0.2",
        "generation_text": "prev",
        "target_model": "decoder_only",
        "prefix_text": "current prefix",
        "prompt_token_key": prompt_token_key([1, 1, 2, 3]),
        "prompt_token_count": 4,
        "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
        "token_count": 2,
        "stop_reason": "bos_token",
    }
    identity[malformed_field] = malformed_value
    gen_dir = test_paths.default_paths("rt").generations_dir()
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=1,
    ) as writer:
        writer.append(identity)

    with pytest.raises(AxBenchBaseError, match="malformed exact generation identity"):
        generation_mod._load_existing_generation_metadata(
            gen_dir,
            expected_max_new_tokens=4,
        )


@pytest.mark.parametrize(
    ("token_count", "stop_reason"),
    [
        pytest.param(3, "max_tokens", id="early-max-token-stop"),
        pytest.param(4, "bos_token", id="bos-at-budget"),
        pytest.param(5, "bos_token", id="bos-over-budget"),
    ],
)
def test_resume_rejects_result_termination_inconsistent_with_current_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    token_count: int,
    stop_reason: str,
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    gen_dir = test_paths.default_paths("rt").generations_dir()
    with ParquetAppendWriter(
        out_dir=gen_dir,
        schema=GENERATIONS_SCHEMA,
        batch_size=1,
    ) as writer:
        writer.append(
            {
                "generation_key": ("diffmean_decoder_only|c0|s0|k0.2|decoder_only"),
                "concept_id": 0,
                "method_name": "diffmean_decoder_only",
                "mode": "positive",
                "eval_seed": 0,
                "coefficient": "0.2",
                "generation_text": "previous",
                "target_model": "decoder_only",
                "prefix_text": "prefix",
                "prompt_token_key": prompt_token_key([1, 2, 3]),
                "prompt_token_count": 3,
                "seed_pool_fingerprint": _TEST_SEED_FINGERPRINT,
                "token_count": token_count,
                "stop_reason": stop_reason,
            }
        )

    with pytest.raises(AxBenchBaseError, match="malformed exact generation identity"):
        generation_mod._load_existing_generation_metadata(
            gen_dir,
            expected_max_new_tokens=4,
        )


def test_encdec_stage4_requires_context_provider(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Encoder-decoder Stage 4 must fail before dispatch without full context."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_encdec"])
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    mock = _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="context_provider.*full-context"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(
                target_models,
                metas,
                tokenizers,
                include_encdec_provider=False,
            ),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )

    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# 4. All rows written to the parquet directory-dataset
# ---------------------------------------------------------------------------


def test_all_rows_written_to_generations_parquet(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Post-run, ``load_existing`` returns every written row with the
    expected schema and non-empty columns."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=["diffmean_decoder_only", "prompting_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0, text="p0"), _seed_row(doc_idx=1, text="p1")]
    neg_seeds = [_seed_row(doc_idx=2, text="n0")]

    _patch_run_intervention(monkeypatch)

    gen_dir = generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    assert gen_dir == paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    # Expected: 10 (diffmean positive) * 2 seeds * 1 concept + 1 (prompting
    # positive) * 2 seeds * 1 concept = 20 + 2 = 22 rows.
    assert table.num_rows == 22
    # Every row has the expected columns populated.
    rows = table.to_pylist()
    for row in rows:
        assert set(row.keys()) == set(GENERATIONS_SCHEMA.names)
        assert row["generation_text"] == "gen"
        assert row["target_model"] == "decoder_only"
        assert row["token_count"] == 3
        assert row["stop_reason"] == "bos_token"


# ---------------------------------------------------------------------------
# 6. Invalid method name in enabled_methods raises
# ---------------------------------------------------------------------------


def test_invalid_method_in_enabled_list_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["nonexistent_method"])
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="nonexistent_method"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[_seed_row()],
        )


def test_invalid_method_in_disabled_list_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, disabled_methods=["typo_method_name"])
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="typo_method_name"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[_seed_row()],
        )


# ---------------------------------------------------------------------------
# 7. Empty concepts list produces an empty (or absent-row) generations dir
# ---------------------------------------------------------------------------


def test_empty_concepts_produces_empty_generations_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path)
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    mock = _patch_run_intervention(monkeypatch)

    gen_dir = generate_all(
        cfg,
        concepts=[],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[_seed_row()],
    )

    assert mock.call_count == 0
    assert gen_dir.exists()
    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.num_rows == 0


# ---------------------------------------------------------------------------
# 8. Run tag not set raises AxBenchBaseError
# ---------------------------------------------------------------------------


def test_run_tag_not_set_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, run_tag=None)
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="run_tag"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[_seed_row()],
        )


# ---------------------------------------------------------------------------
# 9. Generation key format is deterministic and readable
# ---------------------------------------------------------------------------


def test_generation_key_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Spot-check one row's generation_key matches the expected format."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["prompting_decoder_only"])
    assert cfg.run_tag is not None
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    concepts = [_concept(42)]
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]
    _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).generations_dir(),
        schema=GENERATIONS_SCHEMA,
    )
    rows = table.to_pylist()
    # prompting_decoder_only has a single-condition grid, so one row
    # per concept * seed. Our fixture has 1 concept * 1 seed = 1 row.
    assert len(rows) == 1
    expected_key = "prompting_decoder_only|c42|s0|kNone|decoder_only"
    assert rows[0]["generation_key"] == expected_key
    assert rows[0]["coefficient"] == "None"


def test_build_generation_key_helper() -> None:
    """Direct unit test for the key helper, independent of generate_all."""
    key = build_generation_key(
        method_name="diffmean_encdec",
        concept_id=7,
        eval_seed=2,
        coefficient=1.5,
        target_model_name="encdec",
    )
    assert key == "diffmean_encdec|c7|s2|k1.5|encdec"

    # None coefficient → "None" suffix
    key_none = build_generation_key(
        method_name="prompting_encdec",
        concept_id=1,
        eval_seed=0,
        coefficient=None,
        target_model_name="encdec",
    )
    assert key_none == "prompting_encdec|c1|s0|kNone|encdec"

    # Integer coefficient (SAE bucket)
    key_int = build_generation_key(
        method_name="ours_replace_random_encdec",
        concept_id=3,
        eval_seed=5,
        coefficient=2,
        target_model_name="encdec",
    )
    assert key_int == "ours_replace_random_encdec|c3|s5|k2|encdec"


# ---------------------------------------------------------------------------
# 10. Rows carry the exact expected schema
# ---------------------------------------------------------------------------


def test_rows_have_correct_schema(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["lora_decoder_only"])
    assert cfg.run_tag is not None
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[_seed_row()],
    )

    # Verify schema of the on-disk batch matches GENERATIONS_SCHEMA.
    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).generations_dir(),
        schema=GENERATIONS_SCHEMA,
    )
    assert table.schema.equals(GENERATIONS_SCHEMA)
    assert table.schema.field("generation_key").type == pa.string()
    assert table.schema.field("concept_id").type == pa.int32()
    assert table.schema.field("eval_seed").type == pa.int32()
    assert table.schema.field("token_count").type == pa.int32()
    assert table.schema.field("coefficient").type == pa.string()
    assert table.schema.field("stop_reason").type == pa.string()


# ---------------------------------------------------------------------------
# 11. generations_dir created if missing
# ---------------------------------------------------------------------------


def test_generations_dir_created_if_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["lora_decoder_only"])
    assert cfg.run_tag is not None
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    # Ensure it does NOT exist up front.
    assert not gen_dir.exists()
    _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[_seed_row()],
    )
    assert gen_dir.exists()


# ---------------------------------------------------------------------------
# 12. Exception in run_intervention propagates and leaves partial rows on disk
# ---------------------------------------------------------------------------


def test_exception_in_run_intervention_propagates(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """run_intervention raises on the 3rd call → generate_all bubbles up
    the exception; the first 2 rows are still flushed to disk (writer
    buffer_size=100 but the context manager triggers a close() which
    flushes the partial buffer on exit, even on exception)."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["prompting_decoder_only"])
    assert cfg.run_tag is not None
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=i, text=f"p{i}") for i in range(3)]
    neg_seeds = [_seed_row()]

    # prompting has one coefficient per seed; fail on the 3rd logical row.
    results: list[Any] = [
        GenerationResult(text=f"r{i}", token_count=3, stop_reason="bos_token") for i in range(2)
    ]
    results.append(RuntimeError("boom"))
    _patch_run_intervention(monkeypatch, factory=iter(results))

    with pytest.raises(RuntimeError, match="boom"):
        generate_all(
            cfg,
            concepts=concepts,
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=pos_seeds,
            eval_seeds_negative=neg_seeds,
        )

    # The first 2 rows were buffered and flushed on __exit__ via close().
    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).generations_dir(),
        schema=GENERATIONS_SCHEMA,
    )
    assert table.num_rows == 2


# ---------------------------------------------------------------------------
# 13. Method referencing a missing target fails loudly
# ---------------------------------------------------------------------------


def test_missing_target_model_fails_loudly(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """An enabled method must never disappear because its target is absent."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["diffmean_encdec"])
    # Only decoder_only — encdec missing on purpose.
    target_models_only_decoder: dict[str, nn.Module] = {
        "decoder_only": _make_target_module(),
    }

    metas = {
        "decoder_only": _meta(),
    }
    tokenizers_only_decoder: dict[str, Any] = {"decoder_only": _Tokenizer()}
    with pytest.raises(
        AxBenchBaseError,
        match="requires target 'encdec'.*Refusing to silently omit",
    ):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models_only_decoder, metas, tokenizers_only_decoder),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[_seed_row()],
        )


# ---------------------------------------------------------------------------
# 14. Multiple concepts and multiple eval seeds per mode multiply correctly
# ---------------------------------------------------------------------------


def test_multiple_concepts_and_seeds_multiply(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """3 concepts * 4 positive seeds * 1 method (single_condition_grid=1) = 12 rows."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["prompting_decoder_only"])
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    concepts = [_concept(i) for i in range(3)]
    pos_seeds = [_seed_row(doc_idx=i) for i in range(4)]
    neg_seeds = [_seed_row(doc_idx=5)]
    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    assert mock.call_count == 12
    # Collect (concept_id, eval_seed) pairs from the mock calls.
    seen_pairs = {
        (call.kwargs["concept"].concept_id, call.kwargs["eval_seed"])
        for call in mock.call_args_list
    }
    expected_pairs = {(concept_id, seed_idx) for concept_id in range(3) for seed_idx in range(4)}
    assert seen_pairs == expected_pairs


# ---------------------------------------------------------------------------
# 15. Disabled methods are filtered out
# ---------------------------------------------------------------------------


def test_disabled_methods_filtered(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=[
            "diffmean_decoder_only",
            "prompting_decoder_only",
        ],
        disabled_methods=["prompting_decoder_only"],
    )
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[_seed_row()],
    )

    table = load_existing(
        paths.AxBenchBasePaths.from_config(cfg).generations_dir(),
        schema=GENERATIONS_SCHEMA,
    )
    # Only diffmean_decoder_only rows survive; 10 coefficients.
    assert table.num_rows == 10
    method_names = set(table.column("method_name").to_pylist())
    assert method_names == {"diffmean_decoder_only"}


# ---------------------------------------------------------------------------
# 16. Unknown mode raises (defensive)
# ---------------------------------------------------------------------------


def test_unknown_mode_raises_in_seed_router(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """A synthetic Method with an unknown mode should trip the seed
    router's defensive check. We reach the router by patching
    ``ALL_METHODS`` to include one such entry."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path, enabled_methods=["bogus_mode_method"])
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }

    from steering.eval.axbench_base.coefficient_grids import single_condition_grid
    from steering.eval.axbench_base.method_registry import BaselineKind, Method

    # Deliberate smuggle of a bogus mode string into a ``SteeringMode``
    # field — the dispatch's ``_eval_seeds_for_mode`` raise branch is
    # what's being exercised here.
    bogus_method = Method(
        name="bogus_mode_method",
        mode=cast(SteeringMode, "sideways"),
        baseline=BaselineKind.DIFFMEAN,
        target_model="decoder_only",
        requires_negative_sampling=False,
        coefficient_grid_fn=single_condition_grid,
        partition_modifier_name=None,
    )

    # Patch get_method_by_name for enabled-list validation and
    # ALL_METHODS for the iteration.
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.ALL_METHODS",
        [bogus_method],
    )
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.get_method_by_name",
        lambda name: bogus_method,
    )
    _patch_run_intervention(monkeypatch)

    with pytest.raises(AxBenchBaseError, match="unknown method mode"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[_seed_row()],
        )


# ---------------------------------------------------------------------------
# 17. Missing target metadata treats the method as lacking negative sampling
# ---------------------------------------------------------------------------


def test_missing_target_metadata_is_not_negative_sampling(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """If the target-meta dict is missing a target entry entirely, the
    filter returns False (no negative sampling). Combined with the
    target_models check from test #13, the end result is that the
    method is filtered out cleanly."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(
        tmp_path,
        enabled_methods=[
            "neg_ours_add_to_negatives_encdec",
        ],
    )
    # Don't populate encdec in metas at all.
    metas = {
        "decoder_only": _meta(),
    }
    mock = _patch_run_intervention(monkeypatch)

    # Use a decoder_only-only target setup so we don't have to supply
    # an encdec model.
    decoder_only_targets: dict[str, nn.Module] = {
        "decoder_only": _make_target_module(),
    }

    decoder_only_tokenizers: dict[str, Any] = {"decoder_only": _Tokenizer()}

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(decoder_only_targets, metas, decoder_only_tokenizers),
        eval_seeds_positive=[_seed_row()],
        eval_seeds_negative=[_seed_row()],
    )

    # Method filtered out (requires negative sampling + no meta).
    assert mock.call_count == 0


# ---------------------------------------------------------------------------
# 18. Schema matches the judge_all expected schema exactly
# ---------------------------------------------------------------------------


def test_schema_matches_judge_expected_schema() -> None:
    """Integrity check: Stage 4 and Stage 5 must agree on the schema
    byte-for-byte, otherwise load_existing will raise on the stage
    boundary."""
    from steering.eval.axbench_base.judge import _expected_generations_schema

    assert GENERATIONS_SCHEMA.equals(_expected_generations_schema())


# ---------------------------------------------------------------------------
# eval_seeds_per_concept cap
# ---------------------------------------------------------------------------


def test_eval_seeds_capped_by_generation_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """The full eval pool is thousands of rows in production; Stage 4
    must cap iteration at ``cfg.generation.eval_seeds_per_concept_positive``
    so the generation matrix stays bounded.

    Passes 20 eval seeds but a 5-row cap; expects 1 concept * 5 seeds * 1
    method * 1 coefficient = 5 rows regardless of the pool size.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = make_valid_config(
        run_tag="rt",
        output_root=str(tmp_path / "eval_out"),
        enabled_methods=["prompting_decoder_only"],
        generation=GenerationConfig(
            max_new_tokens=4,
            temperature=0.7,
            top_k_sampling=50,
            master_seed=13,
            eval_seeds_per_concept_positive=5,
        ),
    )
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=i) for i in range(20)]
    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=[],
    )

    assert mock.call_count == 5, (
        f"expected 5 dispatches (1 concept x 5 seeds x 1 method x 1 coef), "
        f"got {mock.call_count} — the cap on eval_seeds_per_concept_positive "
        f"is not being applied"
    )
    seed_indices_seen = {call.kwargs["eval_seed"] for call in mock.call_args_list}
    assert seed_indices_seen == set(range(5)), (
        f"Cap should take the head of the pool; got seed indices {seed_indices_seen}"
    )


def test_eval_seeds_negative_cap_applied(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Negative-mode methods cap on ``eval_seeds_per_concept_negative``,
    not the positive field. Passes 20 negative seeds with a cap of 3.
    """
    _set_base_dir(monkeypatch, tmp_path)
    cfg = make_valid_config(
        run_tag="rt",
        output_root=str(tmp_path / "eval_out"),
        enabled_methods=["neg_prompting_decoder_only"],
        run_negative_mode=True,
        generation=GenerationConfig(
            max_new_tokens=4,
            temperature=0.7,
            top_k_sampling=50,
            master_seed=13,
            eval_seeds_per_concept_positive=20,
            eval_seeds_per_concept_negative=3,
        ),
    )
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    concepts = [_concept(0)]
    neg_seeds = [_seed_row(doc_idx=i) for i in range(20)]
    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[],
        eval_seeds_negative=neg_seeds,
    )

    assert mock.call_count == 3, f"expected 3 dispatches, got {mock.call_count}"


def test_eval_seeds_cap_does_not_exceed_pool(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """If the cap is greater than the pool size, every row in the pool
    runs — the cap is a max, not a required count."""
    _set_base_dir(monkeypatch, tmp_path)
    cfg = make_valid_config(
        run_tag="rt",
        output_root=str(tmp_path / "eval_out"),
        enabled_methods=["prompting_decoder_only"],
        generation=GenerationConfig(
            max_new_tokens=4,
            temperature=0.7,
            top_k_sampling=50,
            master_seed=13,
            eval_seeds_per_concept_positive=100,
        ),
    )
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=i) for i in range(4)]
    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=[],
    )

    assert mock.call_count == 4


# ---------------------------------------------------------------------------
# _eval_seeds_for_mode direct tests — cap=0, cap=None, negative cap
# ---------------------------------------------------------------------------


def test_eval_seeds_for_mode_cap_zero_returns_empty_pool() -> None:
    """``cap=0`` must mean "zero seeds" (empty list), not "no cap" (full
    pool). The old implementation had ``if cap is not None and cap > 0``
    which treated 0 as a falsy sentinel — an operator trying to disable
    one mode's evaluation would get the full pool instead. Pin the
    post-fix contract.
    """
    from steering.eval.axbench_base.generation import _eval_seeds_for_mode
    from steering.eval.axbench_base.types import SteeringMode

    pool = [_seed_row(doc_idx=i) for i in range(5)]
    out = _eval_seeds_for_mode(
        SteeringMode.POSITIVE,
        eval_seeds_positive=pool,
        eval_seeds_negative=[],
        cap=0,
    )
    assert out == []


def test_eval_seeds_for_mode_cap_none_returns_full_pool() -> None:
    """``cap=None`` is the "no cap" sentinel — the full pool passes
    through unchanged."""
    from steering.eval.axbench_base.generation import _eval_seeds_for_mode
    from steering.eval.axbench_base.types import SteeringMode

    pool = [_seed_row(doc_idx=i) for i in range(5)]
    out = _eval_seeds_for_mode(
        SteeringMode.POSITIVE,
        eval_seeds_positive=pool,
        eval_seeds_negative=[],
        cap=None,
    )
    assert out == pool


def test_eval_seeds_for_mode_negative_cap_raises() -> None:
    """``cap < 0`` is never a sensible argument — callers who want "no
    cap" pass ``None``. Raise loudly so a config bug doesn't silently
    produce a slice-from-the-end (pool[:-3]) subsample."""
    from steering.eval.axbench_base.generation import _eval_seeds_for_mode
    from steering.eval.axbench_base.types import AxBenchBaseError, SteeringMode

    pool = [_seed_row(doc_idx=i) for i in range(5)]
    with pytest.raises(AxBenchBaseError, match="non-negative"):
        _eval_seeds_for_mode(
            SteeringMode.POSITIVE,
            eval_seeds_positive=pool,
            eval_seeds_negative=[],
            cap=-1,
        )


# ---------------------------------------------------------------------------
# Empty coefficient grid → warning + skip
# ---------------------------------------------------------------------------


def test_empty_coefficient_grid_logs_warning_and_skips(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """A method whose ``coefficient_grid_fn`` returns ``[]`` must surface
    a WARNING log line so operators debugging a missing leaderboard
    entry get a signal. Without the warning the method silently
    disappears from Stage 4 output and then from the Stage 6 report.
    """
    from io import StringIO

    from loguru import logger

    from steering.eval.axbench_base.method_registry import BaselineKind, Method

    empty_grid_method = Method(
        name="empty_grid_method",
        mode=SteeringMode.POSITIVE,
        baseline=BaselineKind.DIFFMEAN,
        target_model="decoder_only",
        requires_negative_sampling=False,
        coefficient_grid_fn=lambda _meta, _cfg: [],  # empty!
        partition_modifier_name=None,
    )
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.ALL_METHODS",
        [empty_grid_method],
    )
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.get_method_by_name",
        lambda name: empty_grid_method,
    )
    mock = _patch_run_intervention(monkeypatch)

    _set_base_dir(monkeypatch, tmp_path)
    cfg = _make_cfg(tmp_path)
    metas = {"decoder_only": _meta(), "encdec": _meta()}

    sink = StringIO()
    handler_id = logger.add(sink, format="{level}|{message}", level="WARNING")
    try:
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=[_seed_row()],
            eval_seeds_negative=[],
        )
    finally:
        logger.remove(handler_id)

    assert mock.call_count == 0, "Empty grid must NOT call run_intervention; method should skip."
    out = sink.getvalue()
    assert "empty_grid_method" in out, out
    assert "empty coefficient grid" in out, out
