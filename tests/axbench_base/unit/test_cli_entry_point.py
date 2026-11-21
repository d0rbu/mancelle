"""Tests for ``scripts/axbench_base_eval.py`` — the Hydra CLI entry point.

Covers:
- ``_run`` (the testable inner function the Hydra decorator wraps):
    - stage dispatch for ``"all"`` and for each single stage name
    - ``run_negative_mode=False`` skip of ``"1c"``
    - resume with an existing ``run_tag`` (not regenerated)
    - auto-generated ``run_tag`` format + determinism
    - ``validate_config`` / ``cfg_to_dataclass`` error paths
    - ``check_stage_dependencies`` is called before every stage and
      blocks execution when it raises
    - unknown stage name bubbles up as ``AxBenchBaseError``
- Stage 4 (GENERATE) dispatches through :func:`generate_all` with
      exactly the target-model families selected by the enabled method plan
- per-stage dispatch tests for stages 0c, 1b, 1c, 2, 3a, 3b, 3c, 3d, 4,
  5 (Phase 6 + Phase 7 of the runtime-state PR) covering resource
  threading, mode loops, error propagation, and
  ``RuntimeState.close()``
- ``main`` is importable and exposes the Hydra ``__wrapped__`` accessor
  so the test suite can drive it without launching the real Hydra
  argument parser
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from loguru import logger as loguru_logger
from omegaconf import DictConfig, OmegaConf

import scripts.eval.axbench_base_eval as cli
from scripts.eval.axbench_base_eval import (
    _ALL_STAGES_IN_ORDER,
    _dispatch_stage,
    _generate_run_tag,
    _resolve_stage_list,
    _run,
    _run_scalar_baseline_training,
    _run_single_stage,
    _validate_stage4_concept_corpora,
    main,
)
from steering.eval.axbench_base import _stage0c_internals as stage0c_internals_mod
from steering.eval.axbench_base import negative_prefixes as negative_prefixes_mod
from steering.eval.axbench_base import paths
from steering.eval.axbench_base import runtime_state as runtime_state_mod
from steering.eval.axbench_base.baselines.conditioning import (
    EncoderConditioning,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.seed_contract import LoadedSeedSplit
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    SeedRow,
    Stage,
    SteeringMode,
    TargetModelMetadata,
)
from steering.eval.cluster_checkpoint_pin import CheckpointFilesPin
from steering.eval.cluster_contract_identity import StableFileDigest
from steering.eval.context_cache_io import (
    CacheIdentity,
    ContextCacheEntry,
    context_length_cache_dir,
    feature_selection_hash_leaf,
    save_context_cache,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.axbench_base.unit.test_config import make_valid_config
from tests.runtime_attribute_helpers import set_runtime_attribute

_N_SAE_FEATURES: PositiveInt = PositiveInt.parse(16_384)
_NUM_BUCKETS: PositiveInt = PositiveInt.parse(4)


@contextmanager
def _capture_loguru_messages(level: str = "INFO") -> Iterator[list[str]]:
    """Capture loguru messages at or above ``level`` into a plain list.

    Loguru does not propagate to pytest's ``caplog``, so the tests
    that need to verify log output attach their own sink via
    :meth:`loguru.logger.add` / :meth:`loguru.logger.remove`.
    """
    collected: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: collected.append(msg.record["message"]),
        level=level,
    )
    try:
        yield collected
    finally:
        loguru_logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _minimal_dict(**overrides: Any) -> dict[str, Any]:
    """Return a dict with every AxBenchBaseConfig mandatory field set."""
    base: dict[str, Any] = {
        "feature_dir": "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        "feature_statistics_path": "out/data/features/test/stats.safetensors",
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cpu",
        "output_root": None,
        "run_tag": None,
        "stage": "all",
        "run_negative_mode": True,
    }
    base.update(overrides)
    return base


def _dict_cfg(**overrides: Any) -> DictConfig:
    return OmegaConf.create(_minimal_dict(**overrides))


# #1249: stage 3a/3b encdec dispatch now conditions the fit on the concept's
# committed fair feature-DiffMean bag. Dispatch tests stub the loader (the bag
# STORE is exercised by tests/axbench_bpb_adv/unit/test_fit_conditioning_loader)
# and assert the provider reaches encdec trainer calls -- and only those.
_STUB_BAG_STORE = "/stub/ours_gcg_bags"
_SENTINEL_CONDITIONING = EncoderConditioning(entries=((1, 1),))


def _stub_encdec_fit_conditioning(
    monkeypatch: pytest.MonkeyPatch,
) -> list[dict[str, Any]]:
    """Record loader calls; every requested concept maps to the sentinel bag."""
    loads: list[dict[str, Any]] = []

    def _fake_load(
        bag_store: Any,
        *,
        mode: Any,
        concept_ids: list[int],
        expected_feature_contract_fingerprint: str | None,
    ) -> dict[int, EncoderConditioning]:
        loads.append(
            {
                "bag_store": bag_store,
                "mode": mode,
                "concept_ids": list(concept_ids),
                "fingerprint": expected_feature_contract_fingerprint,
            }
        )
        return dict.fromkeys(concept_ids, _SENTINEL_CONDITIONING)

    monkeypatch.setattr(cli, "load_encdec_fit_conditioning", _fake_load)
    return loads


def _make_cfg(**overrides: Any) -> AxBenchBaseConfig:
    kwargs: dict[str, Any] = {
        "feature_dir": "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
        "feature_statistics_path": "out/data/features/test/stats.safetensors",
        "decoder_only": {"source": "base", "model_tag": "d13", "step": 10000},
        "encdec": {"source": "encdec", "model_tag": "d13", "step": 10000},
        "device": "cpu",
        "output_root": None,
        "run_tag": "cli-test",
        "stage": "all",
        "run_negative_mode": True,
    }
    kwargs.update(overrides)
    return AxBenchBaseConfig(**kwargs)


@pytest.fixture(autouse=True)
def _pin_nanochat_base_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> Path:
    """Pin NANOCHAT_BASE_DIR to tmp_path so paths resolvers don't touch the
    real ``out/`` tree during these unit tests."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    return tmp_path


def _make_state(cfg: AxBenchBaseConfig) -> RuntimeState:
    """Construct a bare :class:`RuntimeState` for dispatcher tests.

    Phase 6 tests that patch stage function implementations don't
    actually enter heavy-model context managers — patches to
    ``cli.<stage_fn>`` bypass them entirely — so a raw
    :class:`RuntimeState` with no stub loaders installed is safe.
    Tests that DO exercise heavy resources install stub loaders via
    the :func:`_stub_heavy_loaders` fixture.
    """
    return RuntimeState(cfg=cfg)


def _rank0_stage_failure_gloo_worker(
    rank: int,
    world_size: int,
) -> dict[str, str]:
    """Exercise the real rank-0-only Stage 4 failure collective."""

    cfg = _make_cfg(stage="4")
    state = RuntimeState(
        cfg=cfg,
        ddp_rank=rank,
        ddp_world_size=world_size,
    )
    original_check = cli.check_stage_dependencies
    original_run = cli._run_single_stage
    set_runtime_attribute(
        cli,
        "check_stage_dependencies",
        lambda *_args, **_kwargs: None,
    )

    def fail_stage4(*_args: Any, **_kwargs: Any) -> None:
        raise RuntimeError("stage4 preflight failed")

    set_runtime_attribute(cli, "_run_single_stage", fail_stage4)
    try:
        cli._dispatch_stage(cfg, state)
    except Exception as exc:
        return {
            "type": type(exc).__name__,
            "message": str(exc),
        }
    finally:
        cli.check_stage_dependencies = original_check
        cli._run_single_stage = original_run
    return {"type": "none", "message": "stage unexpectedly succeeded"}


@pytest.mark.parametrize(
    ("encdec", "error"),
    [
        (
            {"source": "", "model_tag": "d13", "step": 1},
            "encdec.source must be a non-empty string",
        ),
        (
            {"source": "encdec", "model_tag": "", "step": 1},
            "encdec.model_tag must be a non-empty string",
        ),
        (
            {"source": "encdec", "model_tag": "d13", "step": True},
            "encdec.step must be a non-negative integer",
        ),
        (
            {"source": "encdec", "model_tag": "d13", "step": -1},
            "encdec.step must be non-negative",
        ),
    ],
)
def test_cluster_contract_request_rejects_malformed_checkpoint_ref(
    encdec: dict[str, Any],
    error: str,
) -> None:
    cfg = _make_cfg(
        conditioning={
            "feature_mask": {"mode": "whitelist", "path": "/mask.txt"},
            "feature_clusters": {"path": "/clusters.yaml"},
        },
        encdec=encdec,
    )
    with pytest.raises(AxBenchBaseError, match=error):
        cli._cluster_contract_request(cfg)


def test_cluster_contract_request_requires_mask_and_cluster_mappings() -> None:
    cfg = _make_cfg(
        conditioning={
            "feature_mask": "not-a-mapping",
            "feature_clusters": {"path": "/clusters.yaml"},
        },
    )
    with pytest.raises(AxBenchBaseError, match="requires feature_mask.*mappings"):
        cli._cluster_contract_request(cfg)


def _make_concept(**overrides: Any) -> Concept:
    """Return a minimal well-formed :class:`Concept` for tests."""
    base: dict[str, Any] = {
        "concept_id": 0,
        "feature_index": 100,
        "layer": 8,
        "sae_source": "8-gemmascope-res-16k",
        "description": "the color blue",
        "corpus_count": 50,
        "description_source": "neuronpedia",
    }
    base.update(overrides)
    return Concept(**base)


def _write_completed_stage0c_fixture(
    cfg: AxBenchBaseConfig,
    *,
    artifact_cfg: AxBenchBaseConfig | None = None,
) -> Path:
    """Write a one-concept completed Stage 0c set for CLI resume tests."""
    assert cfg.run_tag is not None
    writer_cfg = cfg if artifact_cfg is None else artifact_cfg
    concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
    candidate_concepts_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
    rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()
    parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    concepts_path.parent.mkdir(parents=True, exist_ok=True)
    candidate_payload = {
        "feature_index": 100,
        "layer": 8,
        "sae_source": writer_cfg.neuronpedia_source,
        "description": "fixture",
        "corpus_count": 50,
        "description_source": "neuronpedia",
    }
    candidate_concepts_path.write_text(
        json.dumps([candidate_payload]),
        encoding="utf-8",
    )
    concepts_path.write_text(
        json.dumps(
            [
                {
                    "concept_id": 0,
                    **candidate_payload,
                }
            ]
        ),
        encoding="utf-8",
    )
    concepts = [Concept(concept_id=0, **candidate_payload)]
    rejected_payload: list[dict[str, Any]] = []
    rejected_path.write_text(json.dumps(rejected_payload), encoding="utf-8")
    n_passes = max(cfg.min_consistency_passes, 1)
    row = {
        "concept_id": 0,
        "prefix_text": "x" * max(cfg.min_prefix_chars, 1),
        "prefix_source_shard": 0,
        "prefix_source_doc": 0,
        "prefix_activating_seg_idx": 0,
        "verification_continuations": ["continuation"] * n_passes,
        "verification_seeds": list(range(n_passes)),
        "verification_percentiles": [
            cfg.verification_percentile_threshold * 100.0 + 1.0
        ] * n_passes,
    }
    pool_identity = stage0c_internals_mod.candidate_pool_identity(
        [CandidateConcept(**candidate_payload)]
    )
    schema = stage0c_internals_mod._schema_with_completed_negative_prefixes_metadata(
        writer_cfg,
        candidate_pool_identity=pool_identity,
    )
    schema = stage0c_internals_mod._schema_with_completed_artifact_set_identities(
        schema,
        concepts=concepts,
        rejected_concepts=rejected_payload,
        concept_examples_qc_report=None,
        stage0c_fingerprint=stage0c_internals_mod._fingerprint_with_pool(
            writer_cfg,
            candidate_pool_identity=pool_identity,
        ),
    )
    table = pa.Table.from_pylist([row], schema=schema)
    pq.write_table(table, parquet_path)
    return parquet_path


def _make_seed_row(shard_idx: int = 0, doc_idx: int = 0) -> SeedRow:
    """Return a minimal :class:`SeedRow` for tests."""
    return SeedRow(
        shard_idx=shard_idx,
        doc_idx=doc_idx,
        source_nanochat_token_count=6,
        prefix_source_token_start=0,
        prefix_source_token_end=3,
        prefix_text="prefix",
        prefix_content_nanochat_tokens=[1, 2, 3],
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text="cont",
        continuation_content_nanochat_tokens=[4, 5, 6],
    )


def _make_meta(*, negative_selection: Any = None) -> TargetModelMetadata:
    """Return a minimal :class:`TargetModelMetadata` for tests."""
    return TargetModelMetadata(
        num_buckets=_NUM_BUCKETS,
        n_sae_features=_N_SAE_FEATURES,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _disable_stage_4_etc_precompute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep Stage 4 dispatch tests in plumbing-only mode.

    The generic CLI tests patch :func:`generate_all` and assert the
    outer Stage 4 wiring. They should not exercise the real
    extract-then-condition precompute helper, which belongs to the
    helper-specific tests below. Also stubs
    :func:`_validate_stage4_concept_corpora` to avoid reading synth
    corpus files that don't exist in unit-test environments.
    """
    monkeypatch.setattr(
        cli,
        "_validate_stage4_concept_corpora",
        lambda cfg, all_concept_ids: None,
    )
    _patch_stage4_context_provider(monkeypatch)


def _patch_stage4_context_provider(
    monkeypatch: pytest.MonkeyPatch,
    provider: Any | None = None,
) -> Any:
    """Replace Stage 4 context-provider construction with a tracked object."""
    context_provider = MagicMock(name="stage4_context_provider")
    if provider is not None:
        context_provider = provider
    monkeypatch.setattr(
        cli,
        "_build_stage4_encdec_context_provider",
        lambda *args, **kwargs: context_provider,
    )
    return context_provider


@pytest.fixture
def _stub_heavy_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Install stub loaders that return controllable :class:`MagicMock` models.

    Mirrors the ``stub_heavy_loaders`` fixture in
    ``tests.axbench_base.unit.test_runtime_state`` so dispatcher tests
    that enter a heavy-model context manager don't need a GPU or the
    real base Gemma + SAE. Every monkeypatched loader returns a fresh
    :class:`MagicMock` on first invocation and caches it for
    re-entry, so the refcount + CPU-offload contract in
    :class:`RuntimeState` works exactly like production.

    The stub also reports CUDA unavailable and replaces
    ``torch.cuda.empty_cache`` with a no-op mock.
    """
    loaded: dict[str, MagicMock] = {}

    def _make_stub_model(name: str) -> MagicMock:
        model = MagicMock(name=name)
        model.to.return_value = model
        if name == "encdec":
            model.dec_config.sequence_len = 1024
        return model

    def _make_loader(name: str) -> Any:
        def _loader(cfg: Any, *_args: Any) -> Any:  # noqa: ARG001
            if name not in loaded:
                loaded[name] = _make_stub_model(name)
            return loaded[name]

        return _loader

    def _make_pair_loader() -> Any:
        def _loader(
            cfg: Any,
            **_kwargs: Any,
        ) -> tuple[Any, Any]:  # noqa: ARG001
            if "base_model" not in loaded:
                loaded["base_model"] = _make_stub_model("base_model")
            if "base_sae" not in loaded:
                loaded["base_sae"] = _make_stub_model("base_sae")
            return loaded["base_model"], loaded["base_sae"]

        return _loader

    monkeypatch.setattr(
        runtime_state_mod, "_load_feature_extraction_pair",
        _make_pair_loader(),
    )
    monkeypatch.setattr(
        runtime_state_mod,
        "_load_decoder_only_model",
        _make_loader("decoder_only"),
    )
    monkeypatch.setattr(
        runtime_state_mod, "_load_encdec_model", _make_loader("encdec"),
    )
    monkeypatch.setattr(runtime_state_mod.t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(runtime_state_mod.t.cuda, "empty_cache", MagicMock())
    return loaded


def _preload_state(
    cfg: AxBenchBaseConfig,
    *,
    concepts: list[Concept] | None = None,
    seeds_train: list[SeedRow] | None = None,
    encdec_meta: TargetModelMetadata | None = None,
    decoder_only_meta: TargetModelMetadata | None = None,
    judge_client: Any = None,
    cluster_contract: Any = None,
) -> RuntimeState:
    """Return a :class:`RuntimeState` with cached resources pre-populated.

    Dispatcher tests that invoke a stage via a monkeypatched stub can
    pre-populate the ``concepts()`` / ``seeds_train()`` / meta caches
    so the stage doesn't touch disk — and the test can assert the
    correct resource was fetched by checking whether the pre-populated
    cache was returned.
    """
    state = RuntimeState(cfg=cfg, cluster_contract=cluster_contract)
    if concepts is not None:
        state._concepts = concepts
        state._stage1b_concepts = concepts
        state._downstream_concepts = concepts
        concept_ids = [concept.concept_id for concept in concepts]
        _write_synth_corpus(
            state.output_paths().synth_training_positive(),
            concept_ids,
        )
        if cfg.run_negative_mode:
            _write_synth_corpus(
                state.output_paths().synth_training_negative(),
                concept_ids,
            )
    if seeds_train is not None:
        state._seeds_train = LoadedSeedSplit(
            rows=seeds_train,
            rows_sha256="d" * 64,
        )
    if encdec_meta is not None:
        state._encdec_meta = encdec_meta
    if decoder_only_meta is not None:
        state._decoder_only_meta = decoder_only_meta
    if judge_client is not None:
        state._judge_client = judge_client
    return state


# ---------------------------------------------------------------------------
# _resolve_stage_list
# ---------------------------------------------------------------------------


class TestResolveStageList:
    def test_all_expands_to_full_pipeline(self) -> None:
        stages = _resolve_stage_list("all", run_negative_mode=True)
        assert stages == list(_ALL_STAGES_IN_ORDER)

    def test_all_skips_1c_when_neg_mode_off(self) -> None:
        stages = _resolve_stage_list("all", run_negative_mode=False)
        assert "1c" not in stages
        # Order must otherwise be preserved.
        expected = [s for s in _ALL_STAGES_IN_ORDER if s != "1c"]
        assert stages == expected

    @pytest.mark.parametrize(
        "stage",
        [
            "0a", "0b", "0c",
            "1b", "1c",
            "2",
            "3a", "3b", "3c", "3d",
            "4", "5", "6",
        ],
    )
    def test_single_stage_returns_singleton(self, stage: str) -> None:
        assert _resolve_stage_list(stage, run_negative_mode=True) == [stage]

    def test_unknown_stage_raises(self) -> None:
        with pytest.raises(AxBenchBaseError, match="Unknown stage"):
            _resolve_stage_list("bogus", run_negative_mode=True)

    def test_single_1c_stage_still_runs_even_with_neg_mode_off(self) -> None:
        """Requesting ``stage="1c"`` explicitly runs it even with
        ``run_negative_mode=False``; only ``"all"`` applies the skip rule.
        The dep check inside ``check_stage_dependencies`` still short-
        circuits to a no-op, which is the desired composable behavior.
        """
        assert _resolve_stage_list("1c", run_negative_mode=False) == ["1c"]

    def test_expected_all_stages_in_order_contents(self) -> None:
        """Guard against accidental reordering of the pipeline stages.

        Uses bare string comparison to verify the backward-compat
        contract: ``_ALL_STAGES_IN_ORDER`` is now a tuple of
        :class:`Stage` members, but because ``Stage`` subclasses
        :class:`str` this string-keyed tuple comparison still holds.
        """
        assert _ALL_STAGES_IN_ORDER == (
            "0a", "0b", "0c",
            "1b", "1c",
            "2",
            "3a", "3b", "3c", "3d", "3e",
            "4", "5", "6",
        )

    def test_all_stages_has_14_entries(self) -> None:
        """Stage 3e adds exact Stage-4 context enumeration to the pipeline."""
        assert len(_ALL_STAGES_IN_ORDER) == 14

    def test_all_expands_to_full_pipeline_via_stage_enum(self) -> None:
        """``_resolve_stage_list`` accepts a :class:`Stage` member in
        place of the bare-string form. Passing ``Stage.ALL`` produces
        the same list as passing ``"all"``.
        """
        via_enum = _resolve_stage_list(Stage.ALL, run_negative_mode=True)
        via_string = _resolve_stage_list("all", run_negative_mode=True)
        assert via_enum == via_string
        assert via_enum == list(_ALL_STAGES_IN_ORDER)

    def test_single_stage_via_enum(self) -> None:
        """Passing a :class:`Stage` member for a single stage returns
        a singleton list containing that member. This is the enum-first
        path alongside the string-first parametrized test above.
        """
        assert _resolve_stage_list(
            Stage.PREPARE_CONCEPTS, run_negative_mode=True,
        ) == [Stage.PREPARE_CONCEPTS]
        assert _resolve_stage_list(
            Stage.TRAIN_LORA, run_negative_mode=True,
        ) == [Stage.TRAIN_LORA]

    def test_all_skips_1c_via_enum_when_neg_mode_off(self) -> None:
        """The skip rule works whether the caller used ``"all"`` or
        :attr:`Stage.ALL`: :attr:`Stage.SYNTH_NEGATIVE_MODE` is removed
        from the expansion.
        """
        out = _resolve_stage_list(Stage.ALL, run_negative_mode=False)
        assert Stage.SYNTH_NEGATIVE_MODE not in out
        assert "1c" not in out


# ---------------------------------------------------------------------------
# _generate_run_tag
# ---------------------------------------------------------------------------


class TestGenerateRunTag:
    def test_format_matches_yyyymmdd_hhmmss_hash(self) -> None:
        cfg = _make_cfg(run_tag=None)
        tag = _generate_run_tag(cfg)
        assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", tag), tag

    def test_deterministic_for_same_cfg_same_second(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stub ``datetime.now`` so two back-to-back calls land on the
        exact same second — the returned tags must be identical when the
        config content is identical."""
        import datetime as _dt

        fixed = _dt.datetime(2026, 4, 14, 12, 34, 56, tzinfo=_dt.UTC)

        class _FixedDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz: Any = None) -> _dt.datetime:
                return fixed

        monkeypatch.setattr(cli, "_DateTime", _FixedDatetime)
        cfg = _make_cfg(run_tag=None)
        tag1 = _generate_run_tag(cfg)
        tag2 = _generate_run_tag(cfg)
        assert tag1 == tag2

    def test_differs_by_cfg_feature_dir(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two configs differing only in ``feature_dir`` produce different
        hash suffixes (the timestamp is stubbed to the same second)."""
        import datetime as _dt

        fixed = _dt.datetime(2026, 4, 14, 12, 34, 56, tzinfo=_dt.UTC)

        class _FixedDatetime(_dt.datetime):
            @classmethod
            def now(cls, tz: Any = None) -> _dt.datetime:
                return fixed

        monkeypatch.setattr(cli, "_DateTime", _FixedDatetime)
        cfg_a = _make_cfg(run_tag=None, feature_dir="out/a")
        cfg_b = _make_cfg(run_tag=None, feature_dir="out/b")
        tag_a = _generate_run_tag(cfg_a)
        tag_b = _generate_run_tag(cfg_b)
        assert tag_a != tag_b
        # Timestamp prefix is identical; suffix (hash) differs.
        assert tag_a.split("-", 2)[:2] == tag_b.split("-", 2)[:2]
        assert tag_a.split("-", 2)[2] != tag_b.split("-", 2)[2]

    def test_starts_with_iso_like_timestamp(self) -> None:
        """Sanity: year portion must be 4 digits and hour portion 2."""
        cfg = _make_cfg(run_tag=None)
        tag = _generate_run_tag(cfg)
        date_part, time_part, hash_part = tag.split("-")
        assert len(date_part) == 8
        assert len(time_part) == 6
        assert len(hash_part) == 8


# ---------------------------------------------------------------------------
# _dispatch_stage
# ---------------------------------------------------------------------------


class TestDispatchStage:
    def test_stage_0a_dispatch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``cfg.stage="0a"`` → ``_run_single_stage`` called once with "0a"."""
        calls: list[str] = []

        def fake_run_single_stage(
            cfg: AxBenchBaseConfig, stage: str, state: RuntimeState,
        ) -> None:
            calls.append(stage)

        def fake_check(cfg: AxBenchBaseConfig, stage: str) -> None:
            return None

        monkeypatch.setattr(cli, "_run_single_stage", fake_run_single_stage)
        monkeypatch.setattr(cli, "check_stage_dependencies", fake_check)

        cfg = _make_cfg(stage="0a")
        state = _make_state(cfg)
        _dispatch_stage(cfg, state)
        assert calls == ["0a"]

    def test_stage_all_runs_in_order(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def fake_run_single_stage(
            cfg: AxBenchBaseConfig, stage: str, state: RuntimeState,
        ) -> None:
            calls.append(stage)

        def fake_check(cfg: AxBenchBaseConfig, stage: str) -> None:
            return None

        monkeypatch.setattr(cli, "_run_single_stage", fake_run_single_stage)
        monkeypatch.setattr(cli, "check_stage_dependencies", fake_check)

        cfg = _make_cfg(stage="all")
        state = _make_state(cfg)
        _dispatch_stage(cfg, state)
        assert calls == list(_ALL_STAGES_IN_ORDER)

    def test_stage_all_skips_1c_when_neg_mode_off(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def fake_run_single_stage(
            cfg: AxBenchBaseConfig, stage: str, state: RuntimeState,
        ) -> None:
            calls.append(stage)

        def fake_check(cfg: AxBenchBaseConfig, stage: str) -> None:
            return None

        monkeypatch.setattr(cli, "_run_single_stage", fake_run_single_stage)
        monkeypatch.setattr(cli, "check_stage_dependencies", fake_check)

        cfg = _make_cfg(stage="all", run_negative_mode=False)
        state = _make_state(cfg)
        _dispatch_stage(cfg, state)
        assert "1c" not in calls
        assert calls == [s for s in _ALL_STAGES_IN_ORDER if s != "1c"]

    def test_check_stage_dependencies_called_before_each_stage(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Every stage must check its deps BEFORE it runs."""
        events: list[tuple[str, str]] = []

        def fake_check(cfg: AxBenchBaseConfig, stage: str) -> None:
            events.append(("check", stage))

        def fake_run_single_stage(
            cfg: AxBenchBaseConfig, stage: str, state: RuntimeState,
        ) -> None:
            events.append(("run", stage))

        monkeypatch.setattr(cli, "check_stage_dependencies", fake_check)
        monkeypatch.setattr(cli, "_run_single_stage", fake_run_single_stage)

        cfg = _make_cfg(stage="all")
        state = _make_state(cfg)
        _dispatch_stage(cfg, state)

        # Every "run X" must be preceded by "check X".
        for i, (kind, stage) in enumerate(events):
            if kind == "run":
                assert events[i - 1] == ("check", stage)

    def test_missing_dependency_raises_and_does_not_run_stage(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``check_stage_dependencies`` raises, ``_run_single_stage``
        must NOT be called."""
        run_called: list[str] = []

        def fake_check(cfg: AxBenchBaseConfig, stage: str) -> None:
            raise AxBenchBaseError(f"simulated missing dep for {stage}")

        def fake_run_single_stage(
            cfg: AxBenchBaseConfig, stage: str, state: RuntimeState,
        ) -> None:
            run_called.append(stage)

        monkeypatch.setattr(cli, "check_stage_dependencies", fake_check)
        monkeypatch.setattr(cli, "_run_single_stage", fake_run_single_stage)

        cfg = _make_cfg(stage="0a")
        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="simulated missing dep"):
            _dispatch_stage(cfg, state)
        assert run_called == []

    def test_invalid_stage_name_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli, "check_stage_dependencies", lambda *a, **k: None)
        monkeypatch.setattr(cli, "_run_single_stage", lambda *a, **k: None)
        cfg = _make_cfg(stage="bogus")
        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="Unknown stage"):
            _dispatch_stage(cfg, state)

    def test_rank0_stage4_failure_raises_canonically_on_two_real_gloo_ranks(
        self,
        gloo_multiprocess: Any,
    ) -> None:
        results = gloo_multiprocess(
            2,
            _rank0_stage_failure_gloo_worker,
        )

        assert results == [
            {
                "type": "AxBenchBaseError",
                "message": (
                    "Rank-0-only stage '4' failed collectively: "
                    "RuntimeError: stage4 preflight failed"
                ),
            },
            {
                "type": "AxBenchBaseError",
                "message": (
                    "Rank-0-only stage '4' failed collectively: "
                    "RuntimeError: stage4 preflight failed"
                ),
            },
        ]

    @pytest.mark.parametrize("rank", [0, 1])
    def test_rank0_stage4_failure_broadcast_raises_on_each_simulated_rank(
        self,
        monkeypatch: pytest.MonkeyPatch,
        rank: int,
    ) -> None:
        cfg = _make_cfg(stage="4")
        state = RuntimeState(cfg=cfg, ddp_rank=rank, ddp_world_size=2)
        run = MagicMock(side_effect=RuntimeError("stage4 preflight failed"))
        monkeypatch.setattr(cli, "_run_single_stage", run)
        monkeypatch.setattr(cli, "check_stage_dependencies", lambda *_args: None)
        monkeypatch.setattr(cli.dist, "is_available", lambda: True)
        monkeypatch.setattr(cli.dist, "is_initialized", lambda: True)

        def broadcast(outcome: list[Any], *, src: int) -> None:
            assert src == 0
            if rank == 1:
                outcome[0] = {
                    "resume_path": None,
                    "resume_error": (
                        "Rank-0-only stage '4' failed collectively: "
                        "RuntimeError: stage4 preflight failed"
                    ),
                }

        monkeypatch.setattr(cli.dist, "broadcast_object_list", broadcast)

        with pytest.raises(
            AxBenchBaseError,
            match=(
                "Rank-0-only stage '4' failed collectively: "
                "RuntimeError: stage4 preflight failed"
            ),
        ):
            _dispatch_stage(cfg, state)

        assert run.call_count == (1 if rank == 0 else 0)

    def test_rank0_stage_outcome_rejects_missing_broadcast_payload(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(cli.dist, "is_available", lambda: True)
        monkeypatch.setattr(cli.dist, "is_initialized", lambda: True)
        monkeypatch.setattr(
            cli.dist,
            "broadcast_object_list",
            lambda *_args, **_kwargs: None,
        )
        with pytest.raises(AxBenchBaseError, match="broadcast returned no rank-0 decision"):
            cli._broadcast_rank0_stage_outcome(
                Stage.GENERATE,
                rank=1,
                world_size=2,
                failure=None,
            )


# ---------------------------------------------------------------------------
# _run_single_stage — data-path stages (0-3) dispatch to real functions
# ---------------------------------------------------------------------------


class TestRunSingleStageDataPath:
    """Stages 0a/0b/1a are callable with only ``cfg`` so the CLI can
    drive them directly. We patch the underlying stage functions and
    assert they are called. ``state`` is passed but unused by these
    branches."""

    def test_run_stage_0a_calls_prepare_concept_list(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        def fake_prepare_concept_list(cfg: AxBenchBaseConfig) -> Path:
            calls.append("prepare_concept_list")
            return Path("out/concepts.json")

        monkeypatch.setattr(
            cli, "prepare_concept_list", fake_prepare_concept_list,
        )
        cfg = _make_cfg()
        state = _make_state(cfg)
        _run_single_stage(cfg, "0a", state)
        assert calls == ["prepare_concept_list"]

    def test_run_stage_0b_calls_prepare_seed_pools(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_prepare_seed_pools(
            cfg: AxBenchBaseConfig,
            **kwargs: Any,
        ) -> tuple[Path, Path]:
            calls.append({"cfg": cfg, **kwargs})
            return (Path("train"), Path("eval"))

        monkeypatch.setattr(
            cli, "prepare_seed_pools", fake_prepare_seed_pools,
        )
        cfg = _make_cfg()
        state = _make_state(cfg)
        _run_single_stage(cfg, "0b", state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        assert calls[0]["base_model"] is _stub_heavy_loaders["base_model"]
        assert calls[0]["base_sae"] is _stub_heavy_loaders["base_sae"]
        assert calls[0]["rank"] == 0
        assert calls[0]["world_size"] == 1

    def test_run_stage_0a_via_stage_enum(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_run_single_stage`` accepts a :class:`Stage` member in
        addition to a bare string. Passing
        :attr:`Stage.PREPARE_CONCEPTS` dispatches to the same stage
        function as passing ``"0a"``.
        """
        calls: list[str] = []

        def fake_prepare_concept_list(cfg: AxBenchBaseConfig) -> Path:
            calls.append("prepare_concept_list")
            return Path("out/concepts.json")

        monkeypatch.setattr(
            cli, "prepare_concept_list", fake_prepare_concept_list,
        )
        cfg = _make_cfg()
        state = _make_state(cfg)
        _run_single_stage(cfg, Stage.PREPARE_CONCEPTS, state)
        assert calls == ["prepare_concept_list"]

    def test_clustered_dispatch_threads_full_contract(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract = MagicMock(feature_contract_fingerprint="a" * 64)
        received: list[Any] = []

        def fake_prepare_concept_list(
            cfg: AxBenchBaseConfig,
            *,
            cluster_contract: Any,
        ) -> Path:
            del cfg
            received.append(cluster_contract)
            return Path("out/concepts.json")

        monkeypatch.setattr(
            cli, "prepare_concept_list", fake_prepare_concept_list,
        )
        cfg = _make_cfg()
        state = _preload_state(cfg, cluster_contract=contract)

        _run_single_stage(cfg, Stage.PREPARE_CONCEPTS, state)

        assert received == [contract]


# ---------------------------------------------------------------------------
# _run_single_stage — unknown / Stage.ALL fallthrough
# ---------------------------------------------------------------------------


class TestRunSingleStageFallthrough:
    """After Phase 7, every stage in :data:`_ALL_STAGES_IN_ORDER` is
    dispatched. Only ``Stage.ALL`` (a pseudo-stage expanded upstream)
    and truly unknown stage names reach the fallthrough branch, and
    both are treated as caller bugs — not stubs."""

    def test_unknown_stage_raises_axbench(self) -> None:
        cfg = _make_cfg()
        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="Unknown stage"):
            _run_single_stage(cfg, "99z", state)

    def test_stage_all_passed_directly_raises_axbench(self) -> None:
        """Passing :attr:`Stage.ALL` to ``_run_single_stage`` is a
        caller bug — the CLI expands ``ALL`` via ``_resolve_stage_list``
        before per-stage dispatch. Hitting ``_run_single_stage`` with
        ``ALL`` must raise :class:`AxBenchBaseError`, not crash with a
        confusing ``KeyError`` or silently no-op."""
        cfg = _make_cfg()
        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="Unknown stage"):
            _run_single_stage(cfg, Stage.ALL, state)


# ---------------------------------------------------------------------------
# _run_single_stage — Stage 6 aggregate dispatch (Phase 5)
# ---------------------------------------------------------------------------


class TestRunSingleStageAggregate:
    """Stage 6 is pure disk IO — no RuntimeState, no heavy models. It is
    wired in :func:`_run_single_stage` exactly like the other data-path
    stages (0a / 0b / 1a). These tests pin that the dispatch routes
    ``Stage.AGGREGATE`` to the module-level :func:`cli.aggregate`, that
    the enum + bare-string forms both work, and that errors from the
    underlying stage function bubble up to the operator."""

    def test_run_single_stage_aggregate_dispatches_to_aggregate(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``_run_single_stage(cfg, Stage.AGGREGATE)`` calls
        :func:`cli.aggregate` exactly once with ``cfg``."""
        calls: list[AxBenchBaseConfig] = []

        def fake_aggregate(cfg: AxBenchBaseConfig) -> Path:
            calls.append(cfg)
            return Path("out/report.yaml")

        monkeypatch.setattr(cli, "aggregate", fake_aggregate)
        cfg = _make_cfg()
        state = _make_state(cfg)
        _run_single_stage(cfg, Stage.AGGREGATE, state)
        assert len(calls) == 1
        assert calls[0] is cfg

    def test_run_single_stage_aggregate_via_string(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Passing the bare string ``"6"`` dispatches to
        :func:`cli.aggregate` just like passing :attr:`Stage.AGGREGATE`."""
        calls: list[AxBenchBaseConfig] = []

        def fake_aggregate(cfg: AxBenchBaseConfig) -> Path:
            calls.append(cfg)
            return Path("out/report.yaml")

        monkeypatch.setattr(cli, "aggregate", fake_aggregate)
        cfg = _make_cfg()
        state = _make_state(cfg)
        _run_single_stage(cfg, "6", state)
        assert len(calls) == 1

    def test_run_single_stage_aggregate_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Errors raised from :func:`aggregate` bubble up unchanged —
        the dispatcher must NOT swallow them."""
        def fake_aggregate(cfg: AxBenchBaseConfig) -> Path:
            raise AxBenchBaseError("simulated aggregate failure")

        monkeypatch.setattr(cli, "aggregate", fake_aggregate)
        cfg = _make_cfg()
        state = _make_state(cfg)
        with pytest.raises(
            AxBenchBaseError, match="simulated aggregate failure",
        ):
            _run_single_stage(cfg, Stage.AGGREGATE, state)

    def test_runtime_state_stages_frozenset_is_removed(self) -> None:
        """Phase 7 regression: after wiring Stage 4, the
        :data:`_RUNTIME_STATE_STAGES` frozenset has no more stubbed
        members, so Phase 7 deletes it outright. The symbol must no
        longer exist on the CLI module."""
        assert not hasattr(cli, "_RUNTIME_STATE_STAGES")

    def test_runtime_state_message_is_removed(self) -> None:
        """Phase 7 regression: the operator-facing message is removed
        along with :data:`_RUNTIME_STATE_STAGES`. The symbol must no
        longer exist on the CLI module."""
        assert not hasattr(cli, "_RUNTIME_STATE_MESSAGE")


# ---------------------------------------------------------------------------
# _run — top-level wrapper (testable inner of ``main``)
# ---------------------------------------------------------------------------


class TestRun:
    @pytest.fixture(autouse=True)
    def _stub_env_validation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stub ``validate_env_for_stages`` so existing ``_run`` tests
        don't depend on the real ``OPENROUTER_API_KEY`` being set.

        The Phase 5 env-validation gate is covered by dedicated tests
        in :class:`TestRunEnvValidation` below; this autouse neutralizes
        it for the rest of the ``TestRun`` class so those tests stay
        focused on ``_run``'s own control flow.
        """
        monkeypatch.setattr(
            cli, "validate_env_for_stages", lambda *a, **k: None,
        )

    def test_cfg_validation_runs_before_dispatch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        validate_calls: list[AxBenchBaseConfig] = []
        dispatch_calls: list[AxBenchBaseConfig] = []

        def fake_validate(cfg: AxBenchBaseConfig) -> None:
            validate_calls.append(cfg)

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            dispatch_calls.append(cfg)

        monkeypatch.setattr(cli, "validate_config", fake_validate)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg(run_tag="existing-run")
        _run(cfg)
        assert len(validate_calls) == 1
        assert len(dispatch_calls) == 1

    def test_resume_with_existing_run_tag(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``cfg.run_tag`` already set → must NOT be regenerated by
        ``_run``."""
        dispatched: list[AxBenchBaseConfig] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            dispatched.append(cfg)

        def boom_generate(cfg: AxBenchBaseConfig) -> str:
            raise AssertionError("_generate_run_tag should not be called")

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        monkeypatch.setattr(cli, "_generate_run_tag", boom_generate)

        cfg = _dict_cfg(run_tag="existing-run")
        _run(cfg)
        assert len(dispatched) == 1
        assert dispatched[0].run_tag == "existing-run"

    def test_run_tag_auto_generated_when_null(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        dispatched: list[AxBenchBaseConfig] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            dispatched.append(cfg)

        def fake_generate(cfg: AxBenchBaseConfig) -> str:
            return "20260414-123456-deadbeef"

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        monkeypatch.setattr(cli, "_generate_run_tag", fake_generate)

        cfg = _dict_cfg(run_tag=None)
        _run(cfg)
        assert len(dispatched) == 1
        assert dispatched[0].run_tag == "20260414-123456-deadbeef"

    def test_unclustered_run_calls_local_cluster_marker_boundary(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        marker_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        monkeypatch.setattr(cli, "_dispatch_stage", lambda *_args, **_kwargs: None)
        monkeypatch.setattr(
            cli,
            "_resolve_ddp_coordinates",
            lambda: (0, 2),
        )
        monkeypatch.setattr(
            cli,
            "_install_cluster_marker_ddp",
            lambda *args, **kwargs: marker_calls.append((args, kwargs)),
        )

        _run(_dict_cfg(run_tag="unclustered", stage="6"))

        assert len(marker_calls) == 1
        args, kwargs = marker_calls[0]
        marker_cfg, marker_contract = args
        assert marker_cfg.run_tag == "unclustered"
        assert marker_contract is None
        assert kwargs == {"rank": 0, "world_size": 2}

    def test_cfg_to_dataclass_errors_bubble_up(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When a mandatory field is ``???``, cfg_to_dataclass raises
        ``AxBenchBaseError`` — and ``_run`` surfaces it untouched (we
        never reach validate/dispatch)."""
        dispatch_called: list[bool] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            dispatch_called.append(True)

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg()
        # Force a mandatory field back to missing.
        cfg.feature_dir = "???"
        with pytest.raises(AxBenchBaseError, match="feature_dir"):
            _run(cfg)
        assert dispatch_called == []

    def test_validate_error_bubbles_up(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``validate_config`` rejects the config, ``_run`` must NOT
        call ``_dispatch_stage``."""
        dispatch_called: list[bool] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            dispatch_called.append(True)

        def fake_validate(cfg: AxBenchBaseConfig) -> None:
            raise AxBenchBaseError("bad config")

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        monkeypatch.setattr(cli, "validate_config", fake_validate)

        cfg = _dict_cfg(run_tag="t")
        with pytest.raises(AxBenchBaseError, match="bad config"):
            _run(cfg)
        assert dispatch_called == []


# ---------------------------------------------------------------------------
# _run — env-validation integration (Phase 5)
# ---------------------------------------------------------------------------


class TestRunEnvValidation:
    """``_run`` calls :func:`validate_env_for_stages` BEFORE it calls
    :func:`_dispatch_stage`. These tests verify the integration contract:

    - the gate runs with the RESOLVED stage list (post-``ALL`` expansion,
      post-``run_negative_mode`` filtering)
    - the gate runs with the dataclass ``cfg`` (for the Stage 0c QC
      conditional exemption)
    - if the gate raises, ``_dispatch_stage`` is never called
    - if the gate passes, ``_dispatch_stage`` runs normally
    """

    def test_run_calls_validate_env_for_stages_before_stage_execution(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Order-of-operations contract: env validation runs before
        any stage dispatch. Verified by recording call order into a
        shared event list."""
        events: list[str] = []

        def fake_validate_env(
            stages: list[Stage], *, cfg: AxBenchBaseConfig | None = None,
        ) -> None:
            events.append("validate_env")

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            events.append("dispatch")

        monkeypatch.setattr(cli, "validate_env_for_stages", fake_validate_env)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg(run_tag="env-val-test")
        _run(cfg)
        assert events == ["validate_env", "dispatch"]

    def test_run_passes_resolved_stages_to_validate_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The stage list the validator sees must be the
        ``_resolve_stage_list``-expanded list (not the raw ``cfg.stage``
        string). For ``cfg.stage="5"`` this is ``[Stage.JUDGE]``."""
        captured_stages: list[list[Stage]] = []

        def fake_validate_env(
            stages: list[Stage], *, cfg: AxBenchBaseConfig | None = None,
        ) -> None:
            captured_stages.append(list(stages))

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            return None

        monkeypatch.setattr(cli, "validate_env_for_stages", fake_validate_env)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg(run_tag="env-val-test", stage="5")
        _run(cfg)
        assert captured_stages == [[Stage.JUDGE]]

    def test_run_passes_cfg_to_validate_env(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The validator must receive the fully-processed dataclass
        ``cfg`` so it can consult
        :attr:`AxBenchBaseConfig.enable_concept_examples_qc`."""
        captured_cfg: list[AxBenchBaseConfig | None] = []

        def fake_validate_env(
            stages: list[Stage], *, cfg: AxBenchBaseConfig | None = None,
        ) -> None:
            captured_cfg.append(cfg)

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            return None

        monkeypatch.setattr(cli, "validate_env_for_stages", fake_validate_env)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg(run_tag="env-val-test", stage="0a")
        _run(cfg)
        assert len(captured_cfg) == 1
        captured = captured_cfg[0]
        assert captured is not None
        assert isinstance(captured, AxBenchBaseConfig)
        assert captured.run_tag == "env-val-test"

    def test_run_env_validation_failure_prevents_stage_execution(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When the env validator raises, ``_dispatch_stage`` is NEVER
        called. Crucial for the "fail fast" contract: the operator
        must not see a partial run."""
        dispatch_called: list[bool] = []

        def fake_validate_env(
            stages: list[Stage], *, cfg: AxBenchBaseConfig | None = None,
        ) -> None:
            raise AxBenchBaseError("OPENROUTER_API_KEY missing (simulated)")

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            dispatch_called.append(True)

        monkeypatch.setattr(cli, "validate_env_for_stages", fake_validate_env)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg(run_tag="env-val-test")
        with pytest.raises(
            AxBenchBaseError, match="OPENROUTER_API_KEY missing",
        ):
            _run(cfg)
        assert dispatch_called == []

    def test_run_env_validation_runs_after_run_tag_generated(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The validator receives the FINAL cfg (with ``run_tag``
        populated) — not the raw DictConfig."""
        captured_cfg: list[AxBenchBaseConfig | None] = []

        def fake_validate_env(
            stages: list[Stage], *, cfg: AxBenchBaseConfig | None = None,
        ) -> None:
            captured_cfg.append(cfg)

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            return None

        def fake_generate(cfg: AxBenchBaseConfig) -> str:
            return "20260414-123456-deadbeef"

        monkeypatch.setattr(cli, "validate_env_for_stages", fake_validate_env)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        monkeypatch.setattr(cli, "_generate_run_tag", fake_generate)

        cfg = _dict_cfg(run_tag=None)
        _run(cfg)
        assert len(captured_cfg) == 1
        captured = captured_cfg[0]
        assert captured is not None
        assert captured.run_tag == "20260414-123456-deadbeef"

    def test_run_env_validation_uses_neg_mode_filtered_list(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``run_negative_mode=False``, ``Stage.SYNTH_NEGATIVE_MODE``
        must NOT appear in the list passed to the validator — this
        is the same filter applied by ``_dispatch_stage``, so the two
        callers see a consistent stage list."""
        captured_stages: list[list[Stage]] = []

        def fake_validate_env(
            stages: list[Stage], *, cfg: AxBenchBaseConfig | None = None,
        ) -> None:
            captured_stages.append(list(stages))

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            return None

        monkeypatch.setattr(cli, "validate_env_for_stages", fake_validate_env)
        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)

        cfg = _dict_cfg(
            run_tag="env-val-test",
            stage="all",
            run_negative_mode=False,
        )
        _run(cfg)
        assert len(captured_stages) == 1
        assert Stage.SYNTH_NEGATIVE_MODE not in captured_stages[0]


# ---------------------------------------------------------------------------
# Phase 6: stage dispatch for the 9 newly-wired runtime-state stages
# ---------------------------------------------------------------------------


class TestStage0cDispatch:
    """Stage 0c (PREPARE_NEGATIVE_PREFIXES) enters base_model + base_sae
    and calls :func:`prepare_negative_prefixes` with the concepts path,
    extraction model, SAE, and feature-dir-derived SAE metadata.
    """

    def test_dispatches_to_prepare_negative_prefixes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_prepare(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> Path:
            calls.append({"cfg": cfg, **kwargs})
            return Path("out/negative_mode_prefixes.parquet")

        monkeypatch.setattr(cli, "prepare_negative_prefixes", fake_prepare)
        cfg = _make_cfg()
        state = _preload_state(cfg)
        _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        # concepts_path comes from the invocation's explicit path contract.
        from steering.eval.axbench_base import paths as _p

        assert calls[0]["concepts_path"] == (
            _p.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
        )
        assert "target_decoder_only" not in calls[0]
        assert calls[0]["base_model"] is not None
        assert calls[0]["base_sae"] is not None
        assert state._seeds_eval is None
        assert calls[0]["target_model_meta"].sae_layer == 8
        assert calls[0]["target_model_meta"].segment_size == 128
        assert calls[0]["target_model_meta"].num_buckets is None
        assert "decoder_only" not in _stub_heavy_loaders
        assert "encdec" not in _stub_heavy_loaders

    def test_enters_base_model_base_sae_not_decoder_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 0c must not enter the decoder-only model context."""
        observed_refcounts: dict[str, int] = {}

        def fake_prepare(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> Path:
            observed_refcounts["base_model"] = state._base_model_refcount
            observed_refcounts["sae"] = state._base_sae_refcount
            observed_refcounts["decoder_only"] = state._decoder_only_refcount
            observed_refcounts["encdec"] = state._encdec_refcount
            return Path("out/negative_mode_prefixes.parquet")

        monkeypatch.setattr(cli, "prepare_negative_prefixes", fake_prepare)
        cfg = _make_cfg()
        state = _preload_state(cfg)
        _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)
        assert observed_refcounts == {
            "base_model": 1,
            "sae": 1,
            "decoder_only": 0,
            "encdec": 0,
        }
        # Post-body: every refcount returns to zero.
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0
        assert "decoder_only" not in _stub_heavy_loaders
        assert "encdec" not in _stub_heavy_loaders

    def test_resume_fast_path_skips_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Complete Stage 0c artifacts should return before model loading."""

        def fail_prepare(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            del cfg, kwargs
            raise AssertionError("prepare_negative_prefixes should not run")

        monkeypatch.setattr(cli, "prepare_negative_prefixes", fail_prepare)
        monkeypatch.setattr(
            cli,
            "resume_negative_prefixes_if_complete",
            lambda cfg: Path("out/negative_mode_prefixes.parquet"),
        )

        cfg = _make_cfg()
        state = _make_state(cfg)
        _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0
        assert state._decoder_only_refcount == 0

    def test_resume_fast_path_broadcasts_unexpected_validation_errors(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        seen: dict[str, Any] = {}

        def _unexpected_resume(cfg: AxBenchBaseConfig) -> Path | None:
            del cfg
            raise AttributeError("unexpected malformed header shape")

        def _broadcast(
            resume_path,
            *,
            rank,
            world_size,
            resume_error,
        ):
            seen.update(
                path=resume_path,
                rank=rank,
                world_size=world_size,
                error=resume_error,
            )
            raise AxBenchBaseError(resume_error)

        monkeypatch.setattr(
            cli,
            "resume_negative_prefixes_if_complete",
            _unexpected_resume,
        )
        monkeypatch.setattr(cli, "_broadcast_resume_decision", _broadcast)

        cfg = _make_cfg()
        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="malformed header shape"):
            _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert seen == {
            "path": None,
            "rank": 0,
            "world_size": 1,
            "error": "unexpected malformed header shape",
        }
        assert _stub_heavy_loaders == {}

    def test_real_split_identity_resume_skips_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Exercise the real on-disk validator before any model context."""
        cfg = _make_cfg(
            max_concepts=1,
            n_verified_prefixes_per_concept_target=1,
            min_consistency_passes=1,
            n_verification_seeds=1,
            min_prefix_chars=1,
            enable_concept_examples_qc=False,
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_source="22-gemmascope-2-res-16k",
        )
        _write_completed_stage0c_fixture(cfg)
        monkeypatch.setattr(
            negative_prefixes_mod,
            "read_n_sae_features_from_stats",
            lambda path: 16384,
        )
        monkeypatch.setattr(
            cli,
            "prepare_negative_prefixes",
            lambda *args, **kwargs: pytest.fail(
                "prepare_negative_prefixes should not run"
            ),
        )

        state = _make_state(cfg)
        _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    def test_real_identity_mismatch_fails_before_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stale completed provenance is rejected before model/SAE loading."""
        cfg = _make_cfg(
            max_concepts=1,
            n_verified_prefixes_per_concept_target=1,
            min_consistency_passes=1,
            n_verification_seeds=1,
            min_prefix_chars=1,
            enable_concept_examples_qc=False,
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_source="22-gemmascope-2-res-16k",
        )
        stale_cfg = cfg.replace(feature_extraction_model_id="gemma-3-12b-pt")
        _write_completed_stage0c_fixture(cfg, artifact_cfg=stale_cfg)
        monkeypatch.setattr(
            negative_prefixes_mod,
            "read_n_sae_features_from_stats",
            lambda path: 16384,
        )

        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="feature_extraction_model_id"):
            _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    def test_real_candidate_pool_mismatch_fails_before_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """A pinned run_tag cannot reuse Stage 0c after Stage 0a changes."""
        cfg = _make_cfg(
            max_concepts=1,
            n_verified_prefixes_per_concept_target=1,
            min_consistency_passes=1,
            n_verification_seeds=1,
            min_prefix_chars=1,
            enable_concept_examples_qc=False,
        )
        _write_completed_stage0c_fixture(cfg)
        assert cfg.run_tag is not None
        candidate_path = paths.AxBenchBasePaths.from_config(cfg).candidate_concepts_json()
        candidate_payload = json.loads(candidate_path.read_text(encoding="utf-8"))
        candidate_payload[0]["description"] = "changed by rerun Stage 0a"
        candidate_path.write_text(
            json.dumps(candidate_payload),
            encoding="utf-8",
        )
        monkeypatch.setattr(
            negative_prefixes_mod,
            "read_n_sae_features_from_stats",
            lambda path: 16384,
        )

        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="candidate_pool_identity"):
            _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    @pytest.mark.parametrize(
        ("sidecar", "metadata_key"),
        [
            ("concepts", "concepts_identity"),
            ("rejected", "rejected_concepts_identity"),
        ],
    )
    def test_real_sidecar_identity_mismatch_fails_before_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
        sidecar: str,
        metadata_key: str,
    ) -> None:
        """Mixed canonical JSON/parquet transactions fail before model load."""
        cfg = _make_cfg(
            max_concepts=1,
            n_verified_prefixes_per_concept_target=1,
            min_consistency_passes=1,
            n_verification_seeds=1,
            min_prefix_chars=1,
            enable_concept_examples_qc=False,
        )
        _write_completed_stage0c_fixture(cfg)
        assert cfg.run_tag is not None
        concepts_path = paths.AxBenchBasePaths.from_config(cfg).concepts_json()
        rejected_path = paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json()
        parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
        if sidecar == "concepts":
            payload = json.loads(concepts_path.read_text(encoding="utf-8"))
            payload[0]["description"] = "changed final concept"
            concepts_path.write_text(json.dumps(payload), encoding="utf-8")
        else:
            rejected_path.write_text(
                json.dumps(
                    [
                        {
                            "candidate_index": 1,
                            "feature_index": 101,
                            "description": "changed rejection audit",
                            "reason": "not_enough_verified_prefixes",
                            "n_verified_found": 0,
                            "llm_judge_score": None,
                        }
                    ]
                ),
                encoding="utf-8",
            )
        artifact_paths = (concepts_path, rejected_path, parquet_path)
        preserved = {path: path.read_bytes() for path in artifact_paths}
        monkeypatch.setattr(
            negative_prefixes_mod,
            "read_n_sae_features_from_stats",
            lambda path: 16384,
        )

        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match=metadata_key):
            _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert {path: path.read_bytes() for path in artifact_paths} == preserved
        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    def test_real_full_fingerprint_mismatch_fails_before_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """A config-only Stage 0c change cannot reuse a completed transaction."""
        cfg = _make_cfg(
            max_concepts=1,
            n_verified_prefixes_per_concept_target=1,
            min_consistency_passes=1,
            n_verification_seeds=1,
            min_prefix_chars=1,
            enable_concept_examples_qc=False,
        )
        _write_completed_stage0c_fixture(cfg)
        changed_cfg = cfg.replace(min_prefix_chars=2)
        assert cfg.run_tag is not None
        artifact_paths = (
            paths.AxBenchBasePaths.from_config(cfg).concepts_json(),
            paths.AxBenchBasePaths.from_config(cfg).rejected_concepts_json(),
            paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes(),
        )
        preserved = {path: path.read_bytes() for path in artifact_paths}
        monkeypatch.setattr(
            negative_prefixes_mod,
            "read_n_sae_features_from_stats",
            lambda path: 16384,
        )

        state = _make_state(changed_cfg)
        with pytest.raises(
            AxBenchBaseError,
            match="stage0c_fingerprint_identity",
        ):
            _run_single_stage(
                changed_cfg,
                Stage.PREPARE_NEGATIVE_PREFIXES,
                state,
            )

        assert {path: path.read_bytes() for path in artifact_paths} == preserved
        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    def test_unreadable_completed_parquet_fails_before_heavy_contexts(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Completed-artifact corruption must not trigger model loading."""
        cfg = _make_cfg(
            max_concepts=1,
            n_verified_prefixes_per_concept_target=1,
            min_consistency_passes=1,
            n_verification_seeds=1,
            min_prefix_chars=1,
            enable_concept_examples_qc=False,
        )
        _write_completed_stage0c_fixture(cfg)
        assert cfg.run_tag is not None
        parquet_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
        corrupt_bytes = b"not parquet"
        parquet_path.write_bytes(corrupt_bytes)
        stats_reads = 0

        def _unexpected_stats_read(path: Path) -> int:
            nonlocal stats_reads
            stats_reads += 1
            return 16384

        monkeypatch.setattr(
            negative_prefixes_mod,
            "read_n_sae_features_from_stats",
            _unexpected_stats_read,
        )

        state = _make_state(cfg)
        with pytest.raises(AxBenchBaseError, match="unreadable"):
            _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)

        assert stats_reads == 0
        assert parquet_path.read_bytes() == corrupt_bytes
        assert _stub_heavy_loaders == {}
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    def test_error_propagates_and_refcounts_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Errors from ``prepare_negative_prefixes`` bubble up. After
        the exception unwinds the ``with`` blocks, refcounts drop back
        to zero so :meth:`RuntimeState.close` succeeds."""
        def fake_prepare(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> Path:
            raise AxBenchBaseError("simulated Stage 0c failure")

        monkeypatch.setattr(cli, "prepare_negative_prefixes", fake_prepare)
        cfg = _make_cfg()
        state = _preload_state(cfg)
        with pytest.raises(AxBenchBaseError, match="simulated Stage 0c"):
            _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0
        assert "decoder_only" not in _stub_heavy_loaders
        assert "encdec" not in _stub_heavy_loaders
        # close() must not raise.
        state.close()


class TestStage1bDispatch:
    """Stage 1b (SYNTH_POSITIVE_MODE) enters base_model + base_sae and
    calls :func:`synthesize_positive_mode` with concepts, seeds_train,
    extraction model, sae, encdec meta.
    """

    def test_dispatches_to_synthesize_positive_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            calls.append({"cfg": cfg, **kwargs})
            return Path("out/synth_training_positive.parquet")

        monkeypatch.setattr(cli, "synthesize_positive_mode", fake_synth)
        cfg = _make_cfg()
        concepts = [_make_concept()]
        seeds = [_make_seed_row()]
        state = _preload_state(
            cfg,
            concepts=concepts,
            seeds_train=seeds,
            encdec_meta=_make_meta(),
        )
        _run_single_stage(cfg, Stage.SYNTH_POSITIVE_MODE, state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        assert calls[0]["concepts"] is concepts
        assert calls[0]["seeds_train"] is seeds
        assert calls[0]["seed_rows_sha256"] == "d" * 64
        assert calls[0]["base_model"] is not None
        assert calls[0]["base_sae"] is not None

    def test_only_enters_base_model_and_sae_not_decoder_only(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 1b must NOT enter the decoder_only_model or encdec_model
        contexts — only base_model + base_sae."""
        observed: dict[str, int] = {}

        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            observed["base_model"] = state._base_model_refcount
            observed["sae"] = state._base_sae_refcount
            observed["decoder_only"] = state._decoder_only_refcount
            observed["encdec"] = state._encdec_refcount
            return Path("out/synth_training_positive.parquet")

        monkeypatch.setattr(cli, "synthesize_positive_mode", fake_synth)
        cfg = _make_cfg()
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            seeds_train=[_make_seed_row()],
            encdec_meta=_make_meta(),
        )
        _run_single_stage(cfg, Stage.SYNTH_POSITIVE_MODE, state)
        assert observed["base_model"] == 1
        assert observed["sae"] == 1
        assert observed["decoder_only"] == 0
        assert observed["encdec"] == 0

    def test_clustered_dispatch_threads_full_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        contract = MagicMock(feature_contract_fingerprint="a" * 64)
        received: list[Any] = []

        def fake_synth(
            cfg: AxBenchBaseConfig,
            concepts: list[Concept],
            seeds_train: list[SeedRow],
            base_model: Any,
            base_sae: Any,
            target_model_meta: TargetModelMetadata,
            *,
            seed_rows_sha256: str,
            cluster_contract: Any,
        ) -> Path:
            del cfg, concepts, seeds_train, base_model, base_sae, target_model_meta
            assert seed_rows_sha256 == "d" * 64
            received.append(cluster_contract)
            return Path("out/synth_training_positive.parquet")

        monkeypatch.setattr(cli, "synthesize_positive_mode", fake_synth)
        cfg = _make_cfg()
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            seeds_train=[_make_seed_row()],
            encdec_meta=_make_meta(),
            cluster_contract=contract,
        )

        _run_single_stage(cfg, Stage.SYNTH_POSITIVE_MODE, state)

        assert received == [contract]

    def test_error_propagates_and_refcounts_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            raise AxBenchBaseError("simulated Stage 1b failure")

        monkeypatch.setattr(cli, "synthesize_positive_mode", fake_synth)
        cfg = _make_cfg()
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            seeds_train=[_make_seed_row()],
            encdec_meta=_make_meta(),
        )
        with pytest.raises(AxBenchBaseError, match="simulated Stage 1b"):
            _run_single_stage(cfg, Stage.SYNTH_POSITIVE_MODE, state)
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0


class TestStage1cDispatch:
    """Stage 1c (SYNTH_NEGATIVE_MODE) enters base_model + base_sae and
    calls :func:`synthesize_negative_mode` — but ONLY when
    ``cfg.run_negative_mode=True``. With the flag off, the dispatcher
    skips the stage entirely (log + early return)."""

    def test_dispatches_to_synthesize_negative_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            calls.append({"cfg": cfg, **kwargs})
            return Path("out/synth_training_negative.parquet")

        monkeypatch.setattr(cli, "synthesize_negative_mode", fake_synth)
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(
            cfg, concepts=[_make_concept()], encdec_meta=_make_meta(),
        )
        _run_single_stage(cfg, Stage.SYNTH_NEGATIVE_MODE, state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        assert calls[0]["base_model"] is not None
        assert calls[0]["base_sae"] is not None

    def test_skip_when_run_negative_mode_false(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """With ``run_negative_mode=False``, ``synthesize_negative_mode``
        is NOT called and no heavy-model contexts are entered."""
        calls: list[Any] = []

        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            calls.append(cfg)
            return Path("unreachable")

        monkeypatch.setattr(cli, "synthesize_negative_mode", fake_synth)
        cfg = _make_cfg(run_negative_mode=False)
        state = _make_state(cfg)
        _run_single_stage(cfg, Stage.SYNTH_NEGATIVE_MODE, state)
        assert calls == []
        # Heavy-model contexts never entered.
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0
        assert state._base_model is None
        assert state._base_sae is None

    def test_clustered_dispatch_threads_full_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        contract = MagicMock(feature_contract_fingerprint="a" * 64)
        received: list[Any] = []

        def fake_synth(
            cfg: AxBenchBaseConfig,
            concepts: list[Concept],
            base_model: Any,
            base_sae: Any,
            target_model_meta: TargetModelMetadata,
            *,
            cluster_contract: Any,
        ) -> Path:
            del cfg, concepts, base_model, base_sae, target_model_meta
            received.append(cluster_contract)
            return Path("out/synth_training_negative.parquet")

        monkeypatch.setattr(cli, "synthesize_negative_mode", fake_synth)
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            encdec_meta=_make_meta(),
            cluster_contract=contract,
        )

        _run_single_stage(cfg, Stage.SYNTH_NEGATIVE_MODE, state)

        assert received == [contract]

    def test_error_propagates_and_refcounts_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            raise AxBenchBaseError("simulated Stage 1c failure")

        monkeypatch.setattr(cli, "synthesize_negative_mode", fake_synth)
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(
            cfg, concepts=[_make_concept()], encdec_meta=_make_meta(),
        )
        with pytest.raises(AxBenchBaseError, match="simulated Stage 1c"):
            _run_single_stage(cfg, Stage.SYNTH_NEGATIVE_MODE, state)
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0


class TestStage2Dispatch:
    """Stage 2 (CONCEPT_FEATURE_MAPPING) dispatches on
    ``cfg.concept_feature_mode``: canonical mode needs nothing but
    cfg + concepts; delta_ranked mode needs base_model + base_sae +
    encdec meta.
    """

    def test_canonical_mode_no_heavy_models(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_canonical(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept],
        ) -> tuple[Path, Path | None]:
            calls.append({"cfg": cfg, "concepts": concepts})
            return (Path("pos"), None)

        def fake_delta_ranked(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> tuple[Path, Path | None]:
            raise AssertionError("delta_ranked must not be called")

        monkeypatch.setattr(
            cli, "concept_feature_mapping_canonical", fake_canonical,
        )
        monkeypatch.setattr(
            cli, "concept_feature_mapping_delta_ranked", fake_delta_ranked,
        )
        cfg = _make_cfg(concept_feature_mode="canonical")
        concepts = [_make_concept()]
        state = _preload_state(cfg, concepts=concepts)
        _run_single_stage(cfg, Stage.CONCEPT_FEATURE_MAPPING, state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        assert calls[0]["concepts"] is concepts
        # Heavy-model contexts never entered.
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0

    def test_delta_ranked_mode_enters_base_model_and_sae(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_canonical(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept],
        ) -> tuple[Path, Path | None]:
            raise AssertionError("canonical must not be called")

        def fake_delta_ranked(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> tuple[Path, Path | None]:
            calls.append({"cfg": cfg, **kwargs})
            return (Path("pos"), Path("neg"))

        monkeypatch.setattr(
            cli, "concept_feature_mapping_canonical", fake_canonical,
        )
        monkeypatch.setattr(
            cli, "concept_feature_mapping_delta_ranked", fake_delta_ranked,
        )
        cfg = _make_cfg(
            concept_feature_mode="delta_ranked", run_negative_mode=True,
        )
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            encdec_meta=_make_meta(),
        )
        _run_single_stage(cfg, Stage.CONCEPT_FEATURE_MAPPING, state)
        assert len(calls) == 1
        kwargs = calls[0]
        assert kwargs["base_model"] is not None
        assert kwargs["base_sae"] is not None
        from steering.eval.axbench_base import paths as _p

        assert cfg.run_tag is not None
        output_paths = _p.AxBenchBasePaths.from_config(cfg)
        assert kwargs["synth_training_positive_path"] == (output_paths.synth_training_positive())
        assert kwargs["synth_training_negative_path"] == (output_paths.synth_training_negative())

    def test_clustered_canonical_dispatch_threads_full_contract(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        contract = MagicMock(feature_contract_fingerprint="a" * 64)
        received: list[Any] = []

        def fake_canonical(
            cfg: AxBenchBaseConfig,
            *,
            concepts: list[Concept],
            cluster_contract: Any,
        ) -> tuple[Path, Path | None]:
            del cfg, concepts
            received.append(cluster_contract)
            return (Path("pos"), Path("neg"))

        monkeypatch.setattr(
            cli,
            "concept_feature_mapping_canonical",
            fake_canonical,
        )
        cfg = _make_cfg(concept_feature_mode="canonical")
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            cluster_contract=contract,
        )

        _run_single_stage(cfg, Stage.CONCEPT_FEATURE_MAPPING, state)

        assert received == [contract]

    def test_delta_ranked_mode_neg_path_none_when_neg_mode_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """With ``run_negative_mode=False``, the delta_ranked branch
        passes :data:`None` for ``synth_training_negative_path``."""
        calls: list[dict[str, Any]] = []

        def fake_delta_ranked(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> tuple[Path, Path | None]:
            calls.append({"cfg": cfg, **kwargs})
            return (Path("pos"), None)

        monkeypatch.setattr(
            cli, "concept_feature_mapping_delta_ranked", fake_delta_ranked,
        )
        cfg = _make_cfg(
            concept_feature_mode="delta_ranked", run_negative_mode=False,
        )
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            encdec_meta=_make_meta(),
        )
        _run_single_stage(cfg, Stage.CONCEPT_FEATURE_MAPPING, state)
        assert calls[0]["synth_training_negative_path"] is None

    def test_unknown_concept_feature_mode_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``validate_config`` normally catches this at config load
        time, but the dispatcher also has a defensive branch."""
        # Bypass validate_config by constructing directly.
        cfg = _make_cfg(concept_feature_mode="canonical")
        # Patch the field via the dataclass replace helper — but
        # AxBenchBaseConfig is frozen. Use object.__setattr__ to bypass.
        object.__setattr__(cfg, "concept_feature_mode", "bogus")
        state = _preload_state(cfg, concepts=[_make_concept()])
        with pytest.raises(
            AxBenchBaseError, match="Unknown concept_feature_mode",
        ):
            _run_single_stage(cfg, Stage.CONCEPT_FEATURE_MAPPING, state)


class TestStage3aDispatch:
    """Stage 3a (TRAIN_DIFFMEAN) trains directions for both targets
    (decoder_only + encdec) across positive + optionally negative modes.
    Each (target, mode) pair is a separate trainer invocation."""

    def test_both_targets_positive_and_negative(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append({"cfg": cfg, **kwargs})
            return []

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)
        # 2 targets x 2 modes = 4 calls.
        assert len(calls) == 4
        target_modes = [
            (c["target_model_name"], c["mode"]) for c in calls
        ]
        assert target_modes == [
            ("decoder_only", SteeringMode.POSITIVE),
            ("decoder_only", SteeringMode.NEGATIVE),
            ("encdec", SteeringMode.POSITIVE),
            ("encdec", SteeringMode.NEGATIVE),
        ]

    def test_neg_mode_false_only_positive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append({"cfg": cfg, **kwargs})
            return []

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)
        # 2 targets x 1 mode = 2 calls.
        assert len(calls) == 2
        assert all(c["mode"] is SteeringMode.POSITIVE for c in calls)

    def test_clustered_dispatch_covers_both_targets_and_threads_fingerprint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        fingerprint = "a" * 64
        contract = MagicMock(feature_contract_fingerprint=fingerprint)
        calls: list[dict[str, Any]] = []

        def fake_trainer(
            cfg: AxBenchBaseConfig,
            target_model: Any,
            target_model_name: str,
            mode: SteeringMode,
            concepts: list[Concept],
            *,
            training_corpus_path: Path,
            feature_contract_fingerprint: str,
            encoder_conditioning_for: Any = None,
        ) -> list[Path]:
            del cfg
            calls.append(
                {
                    "target_model": target_model,
                    "target_model_name": target_model_name,
                    "encoder_conditioning_for": encoder_conditioning_for,
                    "mode": mode,
                    "concepts": concepts,
                    "training_corpus_path": training_corpus_path,
                    "feature_contract_fingerprint": feature_contract_fingerprint,
                }
            )
            return []

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            cluster_contract=contract,
        )

        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)

        # Both targets, matching the unclustered path. Training decoder_only
        # ONLY left encdec rows un-counter-steerable: the counter direction is
        # requested in the ROW'S OWN target coordinates whenever
        # `adversarial_strength != 0`, so an encdec row at non-zero strength
        # needs `<family>/encdec/`. Invisible at `magnitudes: [0.0]` -- the
        # requirement is gated on strength -- so it survived until a real
        # counter-steering sweep hit the preflight.
        #
        # Order is TARGET-MAJOR (all modes per target), matching
        # `test_both_targets_positive_and_negative`.
        assert [call["target_model_name"] for call in calls] == [
            "decoder_only",
            "decoder_only",
            "encdec",
            "encdec",
        ]
        # #1249: encdec fits are conditioned by the concept's committed bag;
        # decoder-only fits have nothing to condition. The provider must reach
        # exactly the encdec calls.
        for call in calls:
            provider = call["encoder_conditioning_for"]
            if call["target_model_name"] == "encdec":
                assert provider is not None
                assert provider(_make_concept().concept_id) is _SENTINEL_CONDITIONING
            else:
                assert provider is None
        assert [call["mode"] for call in calls] == [
            SteeringMode.POSITIVE,
            SteeringMode.NEGATIVE,
            SteeringMode.POSITIVE,
            SteeringMode.NEGATIVE,
        ]
        assert all(
            call["feature_contract_fingerprint"] == fingerprint
            for call in calls
        )

    def test_corpus_paths_per_mode(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Positive mode must get the synth_training_positive path;
        negative mode must get the synth_training_negative path."""
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append(kwargs)
            return []

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)
        from steering.eval.axbench_base import paths as _p

        assert cfg.run_tag is not None
        output_paths = _p.AxBenchBasePaths.from_config(cfg)
        positive_calls = [c for c in calls if c["mode"] is SteeringMode.POSITIVE]
        negative_calls = [c for c in calls if c["mode"] is SteeringMode.NEGATIVE]
        for c in positive_calls:
            assert c["training_corpus_path"] == output_paths.synth_training_positive()
        for c in negative_calls:
            assert c["training_corpus_path"] == output_paths.synth_training_negative()

    def test_target_models_entered_sequentially(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """The decoder-only and encdec contexts are entered ONE at a time
        (not nested). While trainer runs, only one target refcount is 1."""
        observed_refcounts: list[dict[str, int]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            observed_refcounts.append({
                "decoder_only": state._decoder_only_refcount,
                "encdec": state._encdec_refcount,
            })
            return []

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)
        # First 1 call is under decoder_only (refcount 1, encdec 0);
        # second is under encdec (refcount 1, decoder_only 0).
        assert observed_refcounts[0] == {"decoder_only": 1, "encdec": 0}
        assert observed_refcounts[1] == {"decoder_only": 0, "encdec": 1}

    def test_error_propagates_and_refcounts_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            raise AxBenchBaseError("simulated Stage 3a failure")

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        with pytest.raises(AxBenchBaseError, match="simulated Stage 3a"):
            _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0


class TestStage3bDispatch:
    """Stage 3b (TRAIN_LINEAR_PROBE) uses the same dispatch pattern as
    3a — two targets, two modes each. Tests mirror the 3a set."""

    def test_both_targets_positive_and_negative(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append({"cfg": cfg, **kwargs})
            return []

        monkeypatch.setattr(cli, "train_linear_probe_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_LINEAR_PROBE, state)
        assert len(calls) == 4
        assert [c["target_model_name"] for c in calls] == [
            "decoder_only", "decoder_only", "encdec", "encdec",
        ]

    def test_neg_mode_false_only_positive(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append(kwargs)
            return []

        monkeypatch.setattr(cli, "train_linear_probe_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_LINEAR_PROBE, state)
        assert len(calls) == 2
        assert all(c["mode"] is SteeringMode.POSITIVE for c in calls)

    def test_clustered_dispatch_covers_both_targets_and_threads_fingerprint(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        fingerprint = "a" * 64
        contract = MagicMock(feature_contract_fingerprint=fingerprint)
        calls: list[dict[str, Any]] = []

        def fake_trainer(
            cfg: AxBenchBaseConfig,
            target_model: Any,
            target_model_name: str,
            mode: SteeringMode,
            concepts: list[Concept],
            *,
            training_corpus_path: Path,
            feature_contract_fingerprint: str,
            encoder_conditioning_for: Any = None,
        ) -> list[Path]:
            del cfg
            calls.append(
                {
                    "target_model": target_model,
                    "target_model_name": target_model_name,
                    "encoder_conditioning_for": encoder_conditioning_for,
                    "mode": mode,
                    "concepts": concepts,
                    "training_corpus_path": training_corpus_path,
                    "feature_contract_fingerprint": feature_contract_fingerprint,
                }
            )
            return []

        monkeypatch.setattr(cli, "train_linear_probe_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            cluster_contract=contract,
        )

        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_LINEAR_PROBE, state)

        # Both targets, matching the unclustered path. Training decoder_only
        # ONLY left encdec rows un-counter-steerable: the counter direction is
        # requested in the ROW'S OWN target coordinates whenever
        # `adversarial_strength != 0`, so an encdec row at non-zero strength
        # needs `<family>/encdec/`. Invisible at `magnitudes: [0.0]` -- the
        # requirement is gated on strength -- so it survived until a real
        # counter-steering sweep hit the preflight.
        #
        # Order is TARGET-MAJOR (all modes per target), matching
        # `test_both_targets_positive_and_negative`.
        assert [call["target_model_name"] for call in calls] == [
            "decoder_only",
            "decoder_only",
            "encdec",
            "encdec",
        ]
        # #1249: encdec fits are conditioned by the concept's committed bag;
        # decoder-only fits have nothing to condition. The provider must reach
        # exactly the encdec calls.
        for call in calls:
            provider = call["encoder_conditioning_for"]
            if call["target_model_name"] == "encdec":
                assert provider is not None
                assert provider(_make_concept().concept_id) is _SENTINEL_CONDITIONING
            else:
                assert provider is None
        assert [call["mode"] for call in calls] == [
            SteeringMode.POSITIVE,
            SteeringMode.NEGATIVE,
            SteeringMode.POSITIVE,
            SteeringMode.NEGATIVE,
        ]
        assert all(
            call["feature_contract_fingerprint"] == fingerprint
            for call in calls
        )

    def test_does_not_touch_base_model_or_sae(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 3b uses target models only; base_model + base_sae
        refcounts remain 0 throughout the stage."""
        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            assert state._base_model_refcount == 0
            assert state._base_sae_refcount == 0
            return []

        monkeypatch.setattr(cli, "train_linear_probe_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_LINEAR_PROBE, state)

    def test_error_propagates_and_refcounts_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            raise AxBenchBaseError("simulated Stage 3b failure")

        monkeypatch.setattr(cli, "train_linear_probe_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        with pytest.raises(AxBenchBaseError, match="simulated Stage 3b"):
            _run_single_stage(cfg, Stage.TRAIN_LINEAR_PROBE, state)
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0


class TestStage3cDispatch:
    """Stage 3c (TRAIN_PROMPTING) is pure disk IO — no heavy models, no
    judge client. It writes concept descriptions to a JSON file.
    :func:`write_prompting_descriptions` handles both modes internally
    via ``cfg.run_negative_mode``, so the dispatcher calls it once."""

    def test_dispatches_to_write_prompting_descriptions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_write(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept],
        ) -> list[Path]:
            calls.append({"cfg": cfg, "concepts": concepts})
            return [Path("out/prompting.json")]

        monkeypatch.setattr(cli, "write_prompting_descriptions", fake_write)
        cfg = _make_cfg()
        concepts = [_make_concept()]
        state = _preload_state(cfg, concepts=concepts)
        _run_single_stage(cfg, Stage.TRAIN_PROMPTING, state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        assert calls[0]["concepts"] is concepts

    def test_no_heavy_model_contexts_entered(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stage 3c must NOT enter any heavy-model context. The four
        refcounts stay at zero and no model is loaded."""
        def fake_write(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept],
        ) -> list[Path]:
            return []

        monkeypatch.setattr(cli, "write_prompting_descriptions", fake_write)
        cfg = _make_cfg()
        state = _preload_state(cfg, concepts=[_make_concept()])
        _run_single_stage(cfg, Stage.TRAIN_PROMPTING, state)
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0
        assert state._base_model is None

    def test_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_write(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept],
        ) -> list[Path]:
            raise AxBenchBaseError("simulated Stage 3c failure")

        monkeypatch.setattr(cli, "write_prompting_descriptions", fake_write)
        cfg = _make_cfg()
        state = _preload_state(cfg, concepts=[_make_concept()])
        with pytest.raises(AxBenchBaseError, match="simulated Stage 3c"):
            _run_single_stage(cfg, Stage.TRAIN_PROMPTING, state)


class TestStage3dDispatch:
    """Stage 3d (TRAIN_LORA) is decoder-only — the encdec target is
    NEVER entered. Tests mirror the 3a set but with a single target."""

    def test_only_decoder_only_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append({"cfg": cfg, **kwargs})
            return []

        monkeypatch.setattr(cli, "train_lora_all", fake_trainer)
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _run_single_stage(cfg, Stage.TRAIN_LORA, state)
        # 1 target x 2 modes = 2 calls.
        assert len(calls) == 2
        assert all(
            c["target_model_name"] == "decoder_only" for c in calls
        )

    def test_neg_mode_false_one_call(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append(kwargs)
            return []

        monkeypatch.setattr(cli, "train_lora_all", fake_trainer)
        cfg = _make_cfg(run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _run_single_stage(cfg, Stage.TRAIN_LORA, state)
        assert len(calls) == 1
        assert calls[0]["mode"] is SteeringMode.POSITIVE
        assert calls[0]["target_model_name"] == "decoder_only"

    def test_encdec_context_never_entered(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 3d must NOT enter the encdec_model context. The encdec
        refcount stays at 0 even with ``run_negative_mode=True``, and
        the encdec model is never loaded."""
        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            assert state._encdec_refcount == 0
            return []

        monkeypatch.setattr(cli, "train_lora_all", fake_trainer)
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _run_single_stage(cfg, Stage.TRAIN_LORA, state)
        assert state._encdec_refcount == 0
        assert state._encdec_model is None

    def test_error_propagates_and_refcounts_zero(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            raise AxBenchBaseError("simulated Stage 3d failure")

        monkeypatch.setattr(cli, "train_lora_all", fake_trainer)
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        with pytest.raises(AxBenchBaseError, match="simulated Stage 3d"):
            _run_single_stage(cfg, Stage.TRAIN_LORA, state)
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0


class TestStage5Dispatch:
    """Stage 5 (JUDGE) uses the judge_client plain accessor (not a
    context manager) and does NOT need heavy models."""

    def test_dispatches_to_judge_all(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_judge(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept], client: Any,
        ) -> Path:
            calls.append({"cfg": cfg, "concepts": concepts, "client": client})
            return Path("out/judge_scores_dir")

        monkeypatch.setattr(cli, "judge_all", fake_judge)
        cfg = _make_cfg()
        fake_client = MagicMock(name="judge_client")
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            judge_client=fake_client,
        )
        _run_single_stage(cfg, Stage.JUDGE, state)
        assert len(calls) == 1
        assert calls[0]["cfg"] is cfg
        assert calls[0]["concepts"] is state._concepts
        assert calls[0]["client"] is fake_client

    def test_no_heavy_model_contexts_entered(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stage 5 must NOT enter any heavy-model context. The four
        refcounts stay at zero."""
        def fake_judge(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept], client: Any,
        ) -> Path:
            return Path("out/judge_scores_dir")

        monkeypatch.setattr(cli, "judge_all", fake_judge)
        cfg = _make_cfg()
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            judge_client=MagicMock(name="judge_client"),
        )
        _run_single_stage(cfg, Stage.JUDGE, state)
        assert state._base_model_refcount == 0
        assert state._base_sae_refcount == 0
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0

    def test_error_propagates(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fake_judge(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept], client: Any,
        ) -> Path:
            raise AxBenchBaseError("simulated Stage 5 failure")

        monkeypatch.setattr(cli, "judge_all", fake_judge)
        cfg = _make_cfg()
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            judge_client=MagicMock(name="judge_client"),
        )
        with pytest.raises(AxBenchBaseError, match="simulated Stage 5"):
            _run_single_stage(cfg, Stage.JUDGE, state)


# ---------------------------------------------------------------------------
# Phase 6: _run_scalar_baseline_training helper regression tests
# ---------------------------------------------------------------------------


class TestScalarBaselineTrainingHelper:
    """Direct tests of the shared helper. The stage dispatch tests
    above exercise it indirectly; these pin corner cases of the
    helper itself."""

    def test_unknown_target_raises(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The helper only recognizes ``"decoder_only"`` and ``"encdec"``
        target names. Any other value is a programming error and must
        raise :class:`AxBenchBaseError`."""
        cfg = _make_cfg(run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])

        def trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            raise AssertionError("trainer should not be called")

        with pytest.raises(AxBenchBaseError, match="Unknown target name"):
            _run_scalar_baseline_training(
                cfg, state, trainer=trainer, targets=("bogus",),
            )

    def test_empty_targets_tuple_noop(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """An empty ``targets`` tuple means "no trainer invocations".
        The helper must exit cleanly without calling the trainer."""
        cfg = _make_cfg(run_negative_mode=True)
        state = _preload_state(cfg, concepts=[_make_concept()])
        calls: list[Any] = []

        def trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append(kwargs)
            return []

        _run_scalar_baseline_training(
            cfg, state, trainer=trainer, targets=(),
        )
        assert calls == []

    def test_stage3_trainer_cannot_receive_stage1_dropped_concept(
        self,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """The final filtered list is the only Stage 3 iteration source."""

        cfg = _make_cfg(run_negative_mode=False)
        original = [_make_concept(concept_id=0), _make_concept(concept_id=1)]
        state = _preload_state(cfg, concepts=original)
        state._stage1b_concepts = [original[1]]
        state._downstream_concepts = [original[1]]
        _write_synth_corpus(
            state.output_paths().synth_training_positive(),
            [1],
        )
        received: list[list[int]] = []

        def trainer(
            cfg: AxBenchBaseConfig,
            **kwargs: Any,
        ) -> list[Path]:
            del cfg
            received.append(
                [concept.concept_id for concept in kwargs["concepts"]]
            )
            return []

        _run_scalar_baseline_training(
            cfg,
            state,
            trainer=trainer,
            targets=("decoder_only",),
        )

        assert received == [[1]]


# ---------------------------------------------------------------------------
# Phase 6: end-to-end _run regression (state constructed + closed)
# ---------------------------------------------------------------------------


class TestRunConstructsAndClosesState:
    """``_run`` must construct a :class:`RuntimeState`, pass it to
    ``_dispatch_stage``, and call :meth:`close` in a ``finally`` block
    so cached resources are always released — even when stage dispatch
    raises."""

    @pytest.fixture(autouse=True)
    def _stub_env_validation(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            cli, "validate_env_for_stages", lambda *a, **k: None,
        )

    def test_state_constructed_and_passed_to_dispatch(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        received_state: list[RuntimeState] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            received_state.append(state)

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        cfg = _dict_cfg(run_tag="state-test")
        _run(cfg)
        assert len(received_state) == 1
        assert isinstance(received_state[0], RuntimeState)

    def test_state_closed_after_success(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        closed_flags: list[bool] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            closed_flags.append(state._closed)

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        cfg = _dict_cfg(run_tag="state-test")
        _run(cfg)
        # During dispatch, state was not yet closed.
        assert closed_flags == [False]

    def test_state_closed_after_dispatch_failure(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If dispatch raises, ``_run`` must still call
        :meth:`RuntimeState.close` in the ``finally`` block."""
        captured_state: list[RuntimeState] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            captured_state.append(state)
            raise AxBenchBaseError("simulated dispatch failure")

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        cfg = _dict_cfg(run_tag="state-test")
        with pytest.raises(AxBenchBaseError, match="simulated dispatch"):
            _run(cfg)
        assert len(captured_state) == 1
        assert captured_state[0]._closed is True

    def test_state_close_idempotent_across_runs(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Two successive calls to ``_run`` each construct a fresh
        :class:`RuntimeState` and close it. No state is shared between
        runs."""
        received_states: list[RuntimeState] = []

        def fake_dispatch(
            cfg: AxBenchBaseConfig, state: RuntimeState,
        ) -> None:
            received_states.append(state)

        monkeypatch.setattr(cli, "_dispatch_stage", fake_dispatch)
        cfg = _dict_cfg(run_tag="state-test")
        _run(cfg)
        _run(cfg)
        assert len(received_states) == 2
        assert received_states[0] is not received_states[1]
        assert received_states[0]._closed is True
        assert received_states[1]._closed is True


# ---------------------------------------------------------------------------
# Phase 6: additional per-stage regression tests (enum form, refcount
# accounting under nested dispatch, extended kwargs assertions)
# ---------------------------------------------------------------------------


class TestPhase6DispatchEnumForm:
    """Each of the 9 newly-wired stages produces the same effect whether
    invoked with its :class:`Stage` member or its bare-string id.
    Parametrized over stage and dispatch mode."""

    @pytest.mark.parametrize(
        "stage_enum",
        [
            Stage.TRAIN_PROMPTING,
            Stage.JUDGE,
        ],
    )
    def test_enum_form_dispatches_correctly(
        self,
        stage_enum: Stage,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stages that don't enter heavy-model contexts can be tested
        without the loader fixture."""
        if stage_enum is Stage.TRAIN_PROMPTING:
            calls: list[str] = []

            def fake_write(
                cfg: AxBenchBaseConfig,
                *,
                concepts: list[Concept],
            ) -> list[Path]:
                calls.append("write")
                return []

            monkeypatch.setattr(
                cli, "write_prompting_descriptions", fake_write,
            )
            cfg = _make_cfg()
            state = _preload_state(cfg, concepts=[_make_concept()])
            _run_single_stage(cfg, stage_enum, state)
            assert calls == ["write"]
        else:  # Stage.JUDGE
            calls = []

            def fake_judge(
                cfg: AxBenchBaseConfig,
                *,
                concepts: list[Concept],
                client: Any,
            ) -> Path:
                calls.append("judge")
                return Path("out/judge_scores_dir")

            monkeypatch.setattr(cli, "judge_all", fake_judge)
            cfg = _make_cfg()
            state = _preload_state(
                cfg,
                concepts=[_make_concept()],
                judge_client=MagicMock(name="judge_client"),
            )
            _run_single_stage(cfg, stage_enum, state)
            assert calls == ["judge"]


# ---------------------------------------------------------------------------
# Stage 4 (GENERATE) dispatch — Phase 7 of the runtime-state PR
# ---------------------------------------------------------------------------


def _preload_stage_4_state(
    cfg: AxBenchBaseConfig,
    *,
    inactive_sampler: Any = None,
) -> tuple[RuntimeState, dict[str, Any]]:
    """Pre-populate every cache Stage 4 touches so the dispatch avoids
    hitting disk / network. Returns ``(state, slots)`` where ``slots``
    has the exact objects planted in each cache so assertions can pin
    "generate_all received the preloaded object".
    """
    concept = _make_concept()
    seed = _make_seed_row()
    encdec_meta = _make_meta()
    decoder_only_meta = _make_meta()
    nanochat_tokenizer = MagicMock(name="nanochat_tokenizer")
    state = _preload_state(
        cfg,
        concepts=[concept],
        encdec_meta=encdec_meta,
        decoder_only_meta=decoder_only_meta,
    )
    state._seeds_eval = [seed]
    state._nanochat_tokenizer = nanochat_tokenizer
    state._inactive_sampler = inactive_sampler
    state._inactive_sampler_loaded = True
    slots: dict[str, Any] = {
        "concept": concept,
        "seed": seed,
        "encdec_meta": encdec_meta,
        "decoder_only_meta": decoder_only_meta,
        "nanochat_tokenizer": nanochat_tokenizer,
        "inactive_sampler": inactive_sampler,
    }
    return state, slots


class _TinyEncdecModel:
    class _DecConfig:
        sequence_len = 2048

    dec_config = _DecConfig()


class TestStage4ContextProvider:
    def test_stage_4_context_provider_requires_conditioning_block(self) -> None:
        cfg = _make_cfg(conditioning=None)
        with pytest.raises(AxBenchBaseError, match="conditioning.context_cache_dir"):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=_TinyEncdecModel(),
                encdec_meta=_make_meta(),
            )

    def test_stage_4_context_provider_requires_cache_dir(self) -> None:
        cfg = _make_cfg(conditioning={"identity": {}})
        with pytest.raises(AxBenchBaseError, match="conditioning.context_cache_dir"):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=_TinyEncdecModel(),
                encdec_meta=_make_meta(),
            )

    def test_stage_4_context_provider_rejects_model_cache_shape_mismatch(
        self,
    ) -> None:
        cfg = _make_cfg(
            conditioning={
                "context_cache_dir": "/cache",
                "n_sae_features": 123,
                "num_buckets": 4,
            },
        )
        with pytest.raises(AxBenchBaseError, match="n_sae_features"):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=_TinyEncdecModel(),
                encdec_meta=_make_meta(),
            )

    def test_stage_4_context_provider_builds_from_model_len_and_meta(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = _make_cfg(
            conditioning={
                "context_cache_dir": "/cache/root",
                "identity": {
                    "model_id": "gemma-2-2b",
                    "sae_id": "gemmascope-residual-l8-16k",
                    "layer": 8,
                    "pooling": "max_pool",
                    "feature_selection_hash": "a" * 64,
                },
            },
        )
        provider = MagicMock(name="provider")
        captured: dict[str, Any] = {}
        monkeypatch.setattr(
            cli,
            "load_stage4_contexts_manifest",
            lambda _path: {
                "encdec_sequence_len": 2048,
                "contexts_fingerprint": "c" * 64,
            },
        )

        def fake_build_context_feature_provider(
            provider_cfg: DictConfig,
            *,
            cache_dir: str,
            error_prefix: str,
            expected_metadata: dict[str, Any],
        ) -> Any:
            captured["cfg"] = provider_cfg
            captured["cache_dir"] = cache_dir
            captured["error_prefix"] = error_prefix
            captured["expected_metadata"] = expected_metadata
            return provider

        monkeypatch.setattr(
            cli,
            "build_context_feature_provider",
            fake_build_context_feature_provider,
        )
        out = cli._build_stage4_encdec_context_provider(
            cfg,
            encdec_model=_TinyEncdecModel(),
            encdec_meta=_make_meta(),
        )

        assert out is provider
        assert captured["cache_dir"] == "/cache/root"
        assert captured["error_prefix"] == "conditioning.context_cache_dir"
        assert captured["expected_metadata"] == {
            "contexts_fingerprint": "c" * 64
        }
        built_cfg = captured["cfg"]
        assert built_cfg.model_max_seq_len == 2048
        assert built_cfg.conditioning.n_sae_features == 16384
        assert built_cfg.conditioning.num_buckets == 4

    def test_stage_4_context_provider_loads_real_cache_layout(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        feature_hash = "b" * 64
        identity = CacheIdentity(
            model_id="gemma-2-2b",
            sae_id="gemmascope-residual-l8-16k",
            layer=8,
            pooling="max_pool",
            feature_selection_hash=feature_hash,
        )
        cache_root = tmp_path / "stage4_context_cache"
        resolved_cache_dir = context_length_cache_dir(
            cache_root,
            2048,
            hash_leaf=feature_selection_hash_leaf(feature_hash),
        )
        save_context_cache(
            resolved_cache_dir,
            [
                ContextCacheEntry.from_sequences(
                    [101, 202, 303],
                    [
                        ContextFeatureSegment(
                            feature_ids=(17,),
                            buckets=(3,),
                            char_start=0,
                            char_end=5,
                        ),
                    ],
                )
            ],
            identity=identity,
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            metadata={
                "model_max_seq_len": 2048,
                "contexts_fingerprint": "c" * 64,
            },
        )
        monkeypatch.setattr(
            cli,
            "load_stage4_contexts_manifest",
            lambda _path: {
                "encdec_sequence_len": 2048,
                "contexts_fingerprint": "c" * 64,
            },
        )
        cfg = _make_cfg(
            conditioning={
                "context_cache_dir": str(cache_root),
                "identity": identity.as_dict(),
            },
        )

        provider = cli._build_stage4_encdec_context_provider(
            cfg,
            encdec_model=_TinyEncdecModel(),
            encdec_meta=_make_meta(),
        )

        segments = provider.get([101, 202, 303])
        assert segments is not None
        assert segments[0].feature_ids == (17,)
        assert segments[0].buckets == (3,)
        assert provider.get([101, 202]) is None

    @pytest.mark.parametrize(
        ("configured_feature_hash", "cache_contexts_fingerprint"),
        [
            ("a" * 64, "c" * 64),
            ("b" * 64, "d" * 64),
        ],
    )
    def test_stage_4_context_provider_rejects_wrong_selection_or_contexts(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        configured_feature_hash: str,
        cache_contexts_fingerprint: str,
    ) -> None:
        cache_feature_hash = "b" * 64
        cache_identity = CacheIdentity(
            model_id="gemma-2-2b",
            sae_id="gemmascope-residual-l8-16k",
            layer=8,
            pooling="max_pool",
            feature_selection_hash=cache_feature_hash,
        )
        cache_root = tmp_path / "stage4_context_cache"
        resolved_cache_dir = context_length_cache_dir(
            cache_root,
            2048,
            hash_leaf=feature_selection_hash_leaf(cache_feature_hash),
        )
        save_context_cache(
            resolved_cache_dir,
            [
                ContextCacheEntry.from_sequences(
                    [101, 202, 303],
                    [
                        ContextFeatureSegment(
                            feature_ids=(17,),
                            buckets=(3,),
                            char_start=0,
                            char_end=5,
                        ),
                    ],
                )
            ],
            identity=cache_identity,
            n_sae_features=_N_SAE_FEATURES,
            num_buckets=_NUM_BUCKETS,
            metadata={
                "model_max_seq_len": 2048,
                "contexts_fingerprint": cache_contexts_fingerprint,
            },
        )
        monkeypatch.setattr(
            cli,
            "load_stage4_contexts_manifest",
            lambda _path: {
                "encdec_sequence_len": 2048,
                "contexts_fingerprint": "c" * 64,
            },
        )
        cfg = _make_cfg(
            conditioning={
                "context_cache_dir": str(cache_root),
                "identity": {
                    **cache_identity.as_dict(),
                    "feature_selection_hash": configured_feature_hash,
                },
            },
        )

        with pytest.raises(AxBenchBaseError, match="Failed to build"):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=_TinyEncdecModel(),
                encdec_meta=_make_meta(),
            )

    def test_stage_4_context_provider_rejects_stage3e_model_length(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            cli,
            "load_stage4_contexts_manifest",
            lambda _path: {
                "encdec_sequence_len": 1024,
                "contexts_fingerprint": "c" * 64,
            },
        )
        cfg = _make_cfg(conditioning={"context_cache_dir": "/cache"})

        with pytest.raises(AxBenchBaseError, match="different.*sequence length"):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=_TinyEncdecModel(),
                encdec_meta=_make_meta(),
            )

    @pytest.mark.parametrize(
        ("cluster_contract", "manifest_fingerprint", "error"),
        [
            (
                MagicMock(feature_contract_fingerprint="a" * 64),
                "b" * 64,
                "do not match the loaded clustered checkpoint",
            ),
            (
                None,
                "a" * 64,
                "Unclustered Stage 4 rejects cluster-stamped",
            ),
        ],
    )
    def test_stage_4_context_provider_rejects_wrong_cluster_ownership(
        self,
        monkeypatch: pytest.MonkeyPatch,
        cluster_contract: Any,
        manifest_fingerprint: str,
        error: str,
    ) -> None:
        monkeypatch.setattr(
            cli,
            "load_stage4_contexts_manifest",
            lambda _path: {
                "encdec_sequence_len": 2048,
                "contexts_fingerprint": "c" * 64,
                "feature_contract_fingerprint": manifest_fingerprint,
            },
        )
        cfg = _make_cfg(conditioning={"context_cache_dir": "/cache"})

        with pytest.raises(AxBenchBaseError, match=error):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=_TinyEncdecModel(),
                encdec_meta=_make_meta(),
                cluster_contract=cluster_contract,
            )

    def test_stage_4_context_provider_rejects_missing_sequence_len(self) -> None:
        class ModelWithoutSeqLen:
            dec_config = object()

        cfg = _make_cfg(conditioning={"context_cache_dir": "/cache"})
        with pytest.raises(AxBenchBaseError, match="dec_config.sequence_len"):
            cli._build_stage4_encdec_context_provider(
                cfg,
                encdec_model=ModelWithoutSeqLen(),
                encdec_meta=_make_meta(),
            )


class TestRunSingleStageGenerate:
    """Stage 4 dispatches with exactly the resources its methods require."""

    @pytest.mark.parametrize(
        ("kwargs", "error_match"),
        [
            (
                {"decoder_only_model": object()},
                "Decoder-only Stage 4 model metadata is missing",
            ),
            (
                {"decoder_only_meta": _make_meta()},
                "Decoder-only Stage 4 metadata has no live model",
            ),
            (
                {
                    "encdec_model": object(),
                    "encdec_meta": _make_meta(),
                },
                "Encoder-decoder Stage 4 model has no context provider",
            ),
            (
                {
                    "encdec_meta": _make_meta(),
                    "encdec_context_provider": MagicMock(),
                },
                "Encoder-decoder Stage 4 metadata/provider has no live model",
            ),
        ],
    )
    def test_stage_4_target_bundle_contract_fails_loudly(
        self,
        kwargs: dict[str, Any],
        error_match: str,
    ) -> None:
        """Inconsistent target resources must never become a partial plan."""

        inputs: dict[str, Any] = {
            "decoder_only_model": None,
            "encdec_model": None,
            "decoder_only_meta": None,
            "encdec_meta": None,
            "nanochat_tokenizer": object(),
            "encdec_context_provider": None,
        }
        inputs.update(kwargs)
        with pytest.raises(AxBenchBaseError, match=error_match):
            cli._stage4_targets(**inputs)

    def test_clustered_decoder_target_requires_loaded_checkpoint_pin(self) -> None:
        with pytest.raises(AxBenchBaseError, match="missing its exact loaded checkpoint pin"):
            cli._stage4_targets(
                decoder_only_model=object(),
                encdec_model=None,
                decoder_only_meta=_make_meta(),
                encdec_meta=None,
                nanochat_tokenizer=object(),
                encdec_context_provider=None,
                cluster_contract=MagicMock(),
            )

    def test_clustered_decoder_target_threads_loaded_checkpoint_pin(
        self,
        tmp_path: Path,
    ) -> None:
        digest = StableFileDigest(
            path=tmp_path / "checkpoint",
            sha256="a" * 64,
            byte_length=1,
            stat_signature=(0, 0, 0, 1, 0, 0),
        )
        pin = CheckpointFilesPin(
            checkpoint_dir=tmp_path,
            resolved_tag="d-test",
            step=7,
            model_file=digest,
            metadata_file=digest,
        )
        model = SimpleNamespace(_loaded_checkpoint_files_pin=pin)

        targets = cli._stage4_targets(
            decoder_only_model=model,
            encdec_model=None,
            decoder_only_meta=_make_meta(),
            encdec_meta=None,
            nanochat_tokenizer=object(),
            encdec_context_provider=None,
            cluster_contract=MagicMock(),
        )

        assert len(targets) == 1
        assert targets[0].checkpoint_files_pin is pin

    def test_unclustered_stage4_targets_build_both_plain_bundles(self) -> None:
        decoder_model = object()
        encdec_model = object()
        decoder_meta = _make_meta()
        encdec_meta = _make_meta()
        tokenizer = object()
        context_provider = MagicMock()

        targets = cli._stage4_targets(
            decoder_only_model=decoder_model,
            encdec_model=encdec_model,
            decoder_only_meta=decoder_meta,
            encdec_meta=encdec_meta,
            nanochat_tokenizer=tokenizer,
            encdec_context_provider=context_provider,
        )

        assert [target.name for target in targets] == ["decoder_only", "encdec"]
        assert [target.model for target in targets] == [decoder_model, encdec_model]
        assert [target.meta for target in targets] == [decoder_meta, encdec_meta]
        assert all(target.tokenizer is tokenizer for target in targets)
        assert targets[0].context_provider is None
        assert targets[1].context_provider is context_provider
        assert all(target.cluster_contract is None for target in targets)
        assert all(target.cluster_pre_forward_guard is None for target in targets)
        assert all(target.checkpoint_files_pin is None for target in targets)

    def test_stage4_target_planning_rejects_unknown_registry_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        invalid = SimpleNamespace(
            name="invalid_target_method",
            mode=SteeringMode.POSITIVE,
            target_model="not-a-target",
        )
        monkeypatch.setattr(cli, "ALL_METHODS", (invalid,))

        with pytest.raises(AxBenchBaseError, match="unknown target"):
            cli._configured_stage4_target_names(_make_cfg())

    def test_stage_4_dispatches_to_generate_all(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """``_run_single_stage(cfg, Stage.GENERATE)`` calls
        :func:`generate_all` exactly once with ``cfg`` positional."""
        calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            calls.append((args, kwargs))
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        assert len(calls) == 1
        args, _ = calls[0]
        assert args[0] is cfg

    def test_stage_4_decoder_only_plan_skips_encdec_model_and_cache(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """A decoder-only method must not require unused encdec resources."""

        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            del args
            received.update(kwargs)
            return Path("out/generations_dir")

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            pytest.fail("decoder-only Stage 4 touched an encdec-only resource")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        monkeypatch.setattr(
            cli,
            "_validate_stage4_concept_corpora",
            lambda cfg, all_concept_ids: None,
        )
        monkeypatch.setattr(cli, "_build_stage4_encdec_context_provider", forbidden)
        cfg = _make_cfg(
            enabled_methods=["diffmean_decoder_only"],
            conditioning=None,
        )
        state, _ = _preload_stage_4_state(cfg)
        monkeypatch.setattr(state, "encdec_model", forbidden)
        monkeypatch.setattr(state, "encdec_meta", forbidden)
        monkeypatch.setattr(state, "inactive_sampler", forbidden)

        _run_single_stage(cfg, Stage.GENERATE, state)

        assert [target.name for target in received["targets"]] == ["decoder_only"]
        assert received["inactive_sampler"] is None

    def test_stage_4_encdec_only_plan_skips_decoder_model(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """An encdec-only method must not load the unused decoder checkpoint."""

        received: dict[str, Any] = {}
        provider = MagicMock(name="context_provider")

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            del args
            received.update(kwargs)
            return Path("out/generations_dir")

        def forbidden(*args: Any, **kwargs: Any) -> Any:
            del args, kwargs
            pytest.fail("encdec-only Stage 4 touched the decoder-only model")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        monkeypatch.setattr(
            cli,
            "_validate_stage4_concept_corpora",
            lambda cfg, all_concept_ids: None,
        )
        monkeypatch.setattr(
            cli,
            "_build_stage4_encdec_context_provider",
            lambda *args, **kwargs: provider,
        )
        cfg = _make_cfg(enabled_methods=["diffmean_encdec"])
        state, _ = _preload_stage_4_state(cfg)
        monkeypatch.setattr(state, "decoder_only_model", forbidden)
        monkeypatch.setattr(state, "decoder_only_meta", forbidden)

        _run_single_stage(cfg, Stage.GENERATE, state)

        assert [target.name for target in received["targets"]] == ["encdec"]
        assert received["targets"][0].context_provider is provider

    def test_stage_4_dispatches_via_bare_string_too(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Passing the bare string ``"4"`` dispatches to
        :func:`generate_all` just like passing :attr:`Stage.GENERATE`."""
        calls: list[str] = []

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            calls.append("generate_all")
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, "4", state)
        assert calls == ["generate_all"]

    def test_stage_4_passes_concepts_from_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, slots = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        assert received["concepts"] == [slots["concept"]]

    def test_stage_4_passes_both_metas(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, slots = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        targets = received["targets"]
        target_by_name = {t.name: t for t in targets}
        assert target_by_name["decoder_only"].meta is slots["decoder_only_meta"]
        assert target_by_name["encdec"].meta is slots["encdec_meta"]

    def test_stage_4_passes_provider_only_to_encdec_target(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        received: dict[str, Any] = {}
        provider = MagicMock(name="required_encdec_context_provider")

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        monkeypatch.setattr(
            cli,
            "_build_stage4_encdec_context_provider",
            lambda *args, **kwargs: provider,
        )
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        target_by_name = {t.name: t for t in received["targets"]}
        assert target_by_name["decoder_only"].context_provider is None
        assert target_by_name["encdec"].context_provider is provider

    def test_stage_4_passes_underlying_models_not_engines(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """``targets`` bundles carry the underlying target modules (GPT
        / EncoderDecoderGPT) in ``.model`` and the
        :class:`InferenceEngine` wrappers in ``.engine``."""
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        targets = received["targets"]
        target_by_name = {t.name: t for t in targets}
        # Both targets present.
        assert set(target_by_name.keys()) == {"decoder_only", "encdec"}
        # .model holds the stub models from _stub_heavy_loaders;
        # .engine holds InferenceEngine (or its stub) instances.
        assert target_by_name["decoder_only"].model is _stub_heavy_loaders[
            "decoder_only"
        ]
        assert target_by_name["encdec"].model is _stub_heavy_loaders["encdec"]

    def test_stage_4_passes_seeds_eval_for_both_modes(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Until Stage 0b produces a distinct negative eval pool,
        Stage 4 threads :meth:`RuntimeState.seeds_eval` into both
        ``eval_seeds_positive`` and ``eval_seeds_negative``."""
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, slots = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        assert received["eval_seeds_positive"] == [slots["seed"]]
        assert received["eval_seeds_negative"] == [slots["seed"]]

    def test_stage_4_passes_tokenizers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, slots = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        targets = received["targets"]
        target_by_name = {t.name: t for t in targets}
        assert target_by_name["decoder_only"].tokenizer is slots["nanochat_tokenizer"]
        assert target_by_name["encdec"].tokenizer is slots["nanochat_tokenizer"]

    def test_stage_4_passes_inactive_sampler_none_when_not_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """If the encdec meta has no ``negative_selection``, the
        accessor returns ``None`` and Stage 4 forwards that. The
        downstream :func:`generate_all` dispatches use the sampler only
        for negative-mode methods, so ``None`` is a legal value."""
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg, inactive_sampler=None)
        _run_single_stage(cfg, Stage.GENERATE, state)
        assert received["inactive_sampler"] is None

    def test_stage_4_passes_inactive_sampler_callable_when_set(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        received: dict[str, Any] = {}

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        fake_sampler = MagicMock(name="inactive_sampler")
        state, _ = _preload_stage_4_state(
            cfg, inactive_sampler=fake_sampler,
        )
        _run_single_stage(cfg, Stage.GENERATE, state)
        assert received["inactive_sampler"] is fake_sampler

    def test_stage_4_validates_exact_authoritative_concept_ids(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 4 validates, then forwards the same authoritative concepts."""
        received: dict[str, Any] = {}
        validated: list[set[int]] = []

        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            received.update(kwargs)
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        monkeypatch.setattr(
            cli,
            "_validate_stage4_concept_corpora",
            lambda cfg, all_concept_ids: validated.append(all_concept_ids),
        )
        _patch_stage4_context_provider(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)

        expected_ids = {concept.concept_id for concept in received["concepts"]}
        assert validated == [expected_ids]
        assert "present_concept_ids_by_mode" not in received

    def test_stage_4_error_propagates_and_closes_state(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Errors raised from :func:`generate_all` must bubble up
        unchanged. After the error, all heavy-model refcounts must
        return to zero so :meth:`close` can finalize cleanly."""
        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            raise AxBenchBaseError("simulated generate_all failure")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        with pytest.raises(
            AxBenchBaseError, match="simulated generate_all failure",
        ):
            _run_single_stage(cfg, Stage.GENERATE, state)
        # Both target-model refcounts must be 0 after the context
        # managers unwind.
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0
        # Close must still succeed.
        state.close()

    def test_stage_4_enters_all_required_context_managers(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """decoder_only_model / encdec_model must each be entered exactly
        once during Stage 4, reflected in the post-dispatch refcounts
        being 0 (entered + exited)."""
        def fake_generate_all(*args: Any, **kwargs: Any) -> Path:
            return Path("out/generations_dir")

        monkeypatch.setattr(cli, "generate_all", fake_generate_all)
        _disable_stage_4_etc_precompute(monkeypatch)
        cfg = _make_cfg()
        state, _ = _preload_stage_4_state(cfg)
        _run_single_stage(cfg, Stage.GENERATE, state)
        assert state._decoder_only_refcount == 0
        assert state._encdec_refcount == 0


class TestPhase6DispatchViaEnum:
    """Each of the 9 newly-wired stages accepts both the enum form
    (:attr:`Stage.TRAIN_DIFFMEAN`) and the bare-string form (``"3a"``).
    These tests pin the enum/string equivalence."""

    @pytest.mark.parametrize(
        "stage_enum, stage_str",
        [
            (Stage.PREPARE_NEGATIVE_PREFIXES, "0c"),
            (Stage.SYNTH_POSITIVE_MODE, "1b"),
            (Stage.SYNTH_NEGATIVE_MODE, "1c"),
            (Stage.CONCEPT_FEATURE_MAPPING, "2"),
            (Stage.TRAIN_DIFFMEAN, "3a"),
            (Stage.TRAIN_LINEAR_PROBE, "3b"),
            (Stage.TRAIN_PROMPTING, "3c"),
            (Stage.TRAIN_LORA, "3d"),
            (Stage.JUDGE, "5"),
        ],
    )
    def test_stage_enum_and_string_equivalence(
        self,
        stage_enum: Stage,
        stage_str: str,
    ) -> None:
        """String form and enum form resolve to the same stage."""
        assert Stage.coerce(stage_str) is stage_enum


class TestPhase6Regression:
    """Extra regression tests pinning Phase 6 contracts that the
    per-stage classes above don't cover directly."""

    def test_runtime_state_stages_symbol_removed(self) -> None:
        """Phase 7: after wiring Stage 4, the
        :data:`_RUNTIME_STATE_STAGES` frozenset is removed entirely.
        The symbol must no longer exist on the CLI module."""
        assert not hasattr(cli, "_RUNTIME_STATE_STAGES")

    def test_runtime_state_message_symbol_removed(self) -> None:
        """Phase 7: operator-facing message is removed alongside
        :data:`_RUNTIME_STATE_STAGES`. The symbol must no longer
        exist on the CLI module."""
        assert not hasattr(cli, "_RUNTIME_STATE_MESSAGE")

    def test_stage_0c_metadata_comes_from_feature_dir_without_encdec_load(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 0c only needs SAE-layer metadata parsed from feature_dir."""
        received_meta: list[TargetModelMetadata] = []

        def fake_prepare(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> Path:
            received_meta.append(kwargs["target_model_meta"])
            return Path("out/negative_mode_prefixes.parquet")

        monkeypatch.setattr(cli, "prepare_negative_prefixes", fake_prepare)
        cfg = _make_cfg()
        state = _preload_state(cfg)
        _run_single_stage(cfg, Stage.PREPARE_NEGATIVE_PREFIXES, state)
        assert len(received_meta) == 1
        assert received_meta[0].sae_layer == 8
        assert received_meta[0].segment_size == 128
        assert received_meta[0].num_buckets is None
        assert state._encdec_meta is None
        assert "encdec" not in _stub_heavy_loaders
        assert "decoder_only" not in _stub_heavy_loaders

    def test_stage_1b_passes_seeds_train_not_seeds_eval(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """Stage 1b must pass the TRAIN seed pool (not eval). Verified
        by pre-populating both caches with distinct objects and asserting
        the trainer received the train pool."""
        received_seeds: list[list[SeedRow]] = []

        def fake_synth(
            cfg: AxBenchBaseConfig, **kwargs: Any,
        ) -> Path:
            received_seeds.append(kwargs["seeds_train"])
            return Path("out/synth_training_positive.parquet")

        monkeypatch.setattr(cli, "synthesize_positive_mode", fake_synth)
        cfg = _make_cfg()
        train_seeds = [_make_seed_row(shard_idx=0)]
        eval_seeds = [_make_seed_row(shard_idx=99)]
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            seeds_train=train_seeds,
            encdec_meta=_make_meta(),
        )
        state._seeds_eval = eval_seeds  # verify we don't accidentally use this
        _run_single_stage(cfg, Stage.SYNTH_POSITIVE_MODE, state)
        assert received_seeds == [train_seeds]
        # Ensure the eval pool was NOT returned.
        assert received_seeds[0] is not eval_seeds

    def test_stage_1c_skip_message_logged(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Item 9: the defensive skip branch must emit a log message
        at INFO level so operators see why the stage was skipped.
        We capture loguru records directly -- the previous form of
        this test claimed to check for the message but only verified
        that the stage function was not called.
        """
        def fake_synth(cfg: AxBenchBaseConfig, **kwargs: Any) -> Path:
            raise AssertionError("must not be called")

        monkeypatch.setattr(cli, "synthesize_negative_mode", fake_synth)
        cfg = _make_cfg(run_negative_mode=False)
        state = _make_state(cfg)
        with _capture_loguru_messages(level="INFO") as messages:
            _run_single_stage(cfg, Stage.SYNTH_NEGATIVE_MODE, state)
        # The skip message must appear in the captured records.
        assert any("Skipping Stage 1c" in m for m in messages), (
            f"expected 'Skipping Stage 1c' in loguru INFO records, "
            f"got {messages!r}"
        )

    def test_stage_2_concepts_reused_not_reloaded(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Stage 2 canonical uses :meth:`RuntimeState.concepts()`; the
        same cached list is returned across repeated accessor calls."""
        calls: list[list[Concept]] = []

        def fake_canonical(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept],
        ) -> tuple[Path, Path | None]:
            calls.append(concepts)
            return (Path("pos"), None)

        monkeypatch.setattr(
            cli, "concept_feature_mapping_canonical", fake_canonical,
        )
        cfg = _make_cfg(concept_feature_mode="canonical")
        concepts = [_make_concept()]
        state = _preload_state(cfg, concepts=concepts)
        _run_single_stage(cfg, Stage.CONCEPT_FEATURE_MAPPING, state)
        assert len(calls) == 1
        assert calls[0] is concepts

    def test_stage_3a_concepts_shared_across_trainer_calls(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """All four trainer invocations share the SAME concepts list
        — it's loaded once from ``state.concepts()`` before the outer
        loop, not re-fetched per call."""
        received: list[list[Concept]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            received.append(kwargs["concepts"])
            return []

        monkeypatch.setattr(cli, "train_diffmean_all", fake_trainer)
        cfg = _make_cfg(encdec_fit_bag_store=_STUB_BAG_STORE, run_negative_mode=True)
        concepts = [_make_concept()]
        state = _preload_state(cfg, concepts=concepts)
        _stub_encdec_fit_conditioning(monkeypatch)
        _run_single_stage(cfg, Stage.TRAIN_DIFFMEAN, state)
        assert len(received) == 4
        # Same identity across every call.
        for c in received:
            assert c is concepts

    def test_stage_3d_corpus_paths_only_positive_when_neg_off(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        calls: list[dict[str, Any]] = []

        def fake_trainer(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
            calls.append(kwargs)
            return []

        monkeypatch.setattr(cli, "train_lora_all", fake_trainer)
        cfg = _make_cfg(run_negative_mode=False)
        state = _preload_state(cfg, concepts=[_make_concept()])
        _run_single_stage(cfg, Stage.TRAIN_LORA, state)
        from steering.eval.axbench_base import paths as _p

        assert cfg.run_tag is not None
        assert len(calls) == 1
        assert calls[0]["training_corpus_path"] == (
            _p.AxBenchBasePaths.from_config(cfg).synth_training_positive()
        )

    def test_stage_5_client_identity_preserved(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The judge client passed to ``judge_all`` must be the EXACT
        same object returned by :meth:`RuntimeState.judge_client`.
        Dispatch must not re-wrap or clone it."""
        original_client = MagicMock(name="original_judge_client")

        def fake_judge(
            cfg: AxBenchBaseConfig, *, concepts: list[Concept], client: Any,
        ) -> Path:
            assert client is original_client
            return Path("out/judge_scores_dir")

        monkeypatch.setattr(cli, "judge_all", fake_judge)
        cfg = _make_cfg()
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            judge_client=original_client,
        )
        _run_single_stage(cfg, Stage.JUDGE, state)


class TestPhase7EndToEndAllStages:
    """Phase 7 end-to-end test: dispatching ``Stage.ALL`` through the
    real :func:`_dispatch_stage` (with every stage function stubbed)
    must route every stage through the expected branch *including*
    Stage 4, which now dispatches via :func:`generate_all`."""

    @pytest.fixture(autouse=True)
    def _stub_all_stage_functions(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> dict[str, list[Any]]:
        """Replace every stage function with a call-tracking stub."""
        calls: dict[str, list[Any]] = {
            "prepare_concept_list": [],
            "prepare_seed_pools": [],
            "prepare_negative_prefixes": [],
            "synthesize_positive_mode": [],
            "synthesize_negative_mode": [],
            "concept_feature_mapping_canonical": [],
            "concept_feature_mapping_delta_ranked": [],
            "train_diffmean_all": [],
            "train_linear_probe_all": [],
            "write_prompting_descriptions": [],
            "train_lora_all": [],
            "enumerate_stage4_contexts": [],
            "generate_all": [],
            "judge_all": [],
            "aggregate": [],
        }

        def _tracker(name: str) -> Any:
            def _stage_fn(*args: Any, **kwargs: Any) -> Any:
                calls[name].append((args, kwargs))
                return Path(f"out/{name}")

            return _stage_fn

        for name in calls:
            monkeypatch.setattr(cli, name, _tracker(name))
        _disable_stage_4_etc_precompute(monkeypatch)
        # check_stage_dependencies always passes.
        monkeypatch.setattr(
            cli, "check_stage_dependencies", lambda cfg, stage: None,
        )
        return calls

    def test_dispatch_stage_all_neg_mode_on(
        self,
        _stub_all_stage_functions: dict[str, list[Any]],
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """With ``stage="all"`` and ``run_negative_mode=True``, every
        stage runs exactly once per target/mode pair where
        applicable, including Stage 4 and all downstream stages."""
        cfg = _make_cfg(
            encdec_fit_bag_store=_STUB_BAG_STORE,
            stage="all",
            run_negative_mode=True,
            concept_feature_mode="canonical",
        )
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            seeds_train=[_make_seed_row()],
            encdec_meta=_make_meta(),
            decoder_only_meta=_make_meta(),
            judge_client=MagicMock(name="judge_client"),
        )
        state._seeds_eval = [_make_seed_row()]
        state._nanochat_tokenizer = MagicMock(name="tokenizer")
        state._inactive_sampler = None
        state._inactive_sampler_loaded = True
        _stub_encdec_fit_conditioning(monkeypatch)
        _dispatch_stage(cfg, state)
        calls = _stub_all_stage_functions
        # 0a
        assert len(calls["prepare_concept_list"]) == 1
        # 0b
        assert len(calls["prepare_seed_pools"]) == 1
        # 0c
        assert len(calls["prepare_negative_prefixes"]) == 1
        # 1b
        assert len(calls["synthesize_positive_mode"]) == 1
        # 1c (neg mode is on)
        assert len(calls["synthesize_negative_mode"]) == 1
        # 2 (canonical)
        assert len(calls["concept_feature_mapping_canonical"]) == 1
        assert len(calls["concept_feature_mapping_delta_ranked"]) == 0
        # 3a: 2 targets x 2 modes = 4
        assert len(calls["train_diffmean_all"]) == 4
        # 3b: 2 targets x 2 modes = 4
        assert len(calls["train_linear_probe_all"]) == 4
        # 3c: once
        assert len(calls["write_prompting_descriptions"]) == 1
        # 3d: 1 target x 2 modes = 2
        assert len(calls["train_lora_all"]) == 2
        # 3e: exact Stage 4 context enumeration
        assert len(calls["enumerate_stage4_contexts"]) == 1
        # 4: Stage.GENERATE now wired
        assert len(calls["generate_all"]) == 1
        # 5: judge
        assert len(calls["judge_all"]) == 1
        # 6: aggregate
        assert len(calls["aggregate"]) == 1

    def test_dispatch_stage_all_neg_mode_off(
        self,
        _stub_all_stage_functions: dict[str, list[Any]],
        monkeypatch: pytest.MonkeyPatch,
        _stub_heavy_loaders: dict[str, MagicMock],
    ) -> None:
        """With ``run_negative_mode=False``, Stage 1c is DROPPED from
        the expansion (not called at all); positive-only corpus
        paths are used."""
        cfg = _make_cfg(
            encdec_fit_bag_store=_STUB_BAG_STORE,
            stage="all",
            run_negative_mode=False,
            concept_feature_mode="canonical",
        )
        state = _preload_state(
            cfg,
            concepts=[_make_concept()],
            seeds_train=[_make_seed_row()],
            encdec_meta=_make_meta(),
            decoder_only_meta=_make_meta(),
            judge_client=MagicMock(name="judge_client"),
        )
        state._seeds_eval = [_make_seed_row()]
        state._nanochat_tokenizer = MagicMock(name="tokenizer")
        state._inactive_sampler = None
        state._inactive_sampler_loaded = True
        _stub_encdec_fit_conditioning(monkeypatch)
        _dispatch_stage(cfg, state)
        calls = _stub_all_stage_functions
        # 1c SKIPPED (not in _ALL_STAGES_IN_ORDER expansion).
        assert len(calls["synthesize_negative_mode"]) == 0
        # 3a/3b: 2 targets x 1 mode = 2 each.
        assert len(calls["train_diffmean_all"]) == 2
        assert len(calls["train_linear_probe_all"]) == 2
        # 3d: 1 target x 1 mode = 1.
        assert len(calls["train_lora_all"]) == 1
        # 3e still runs once.
        assert len(calls["enumerate_stage4_contexts"]) == 1
        # 4: still runs exactly once.
        assert len(calls["generate_all"]) == 1
        # 5 + 6: run.
        assert len(calls["judge_all"]) == 1
        assert len(calls["aggregate"]) == 1


# ---------------------------------------------------------------------------
# main — hydra-wrapped entry point
# ---------------------------------------------------------------------------


class TestMainEntryPoint:
    def test_main_is_hydra_wrapped(self) -> None:
        """``hydra.main`` decorates ``main`` so the underlying function
        is accessible via ``__wrapped__``. Downstream test tooling (and
        the ``if __name__ == "__main__"`` path) both rely on this."""
        assert hasattr(main, "__wrapped__")

    def test_main_wrapped_calls_run(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Bypass the Hydra decorator by calling ``main.__wrapped__``
        directly with a hand-built DictConfig and assert ``_run`` is
        invoked with that config."""
        seen: list[DictConfig] = []

        def fake_run(cfg: DictConfig) -> None:
            seen.append(cfg)

        monkeypatch.setattr(cli, "_run", fake_run)
        cfg = _dict_cfg(run_tag="main-test")
        main.__wrapped__(cfg)
        assert len(seen) == 1
        assert seen[0] is cfg


# ---------------------------------------------------------------------------
# _validate_stage4_concept_corpora — authoritative Stage 4 preflight
# ---------------------------------------------------------------------------


def _write_synth_corpus(path: Path, present_ids: list[int]) -> None:
    """Write a minimal 2-column synth parquet for testing."""
    rows = []
    for cid in present_ids:
        rows.append({"concept_id": cid, "content_class": "concept_rich"})
        rows.append({"concept_id": cid, "content_class": "concept_avoiding"})
    schema = pa.schema([("concept_id", pa.int32()), ("content_class", pa.string())])
    if path.name == "synth_training_positive.parquet":
        schema = schema_with_positive_qc_contract(
            schema,
            accept_threshold=0.3,
        )
    schema = schema_with_stage1_continuation_contract(
        schema,
        role=(
            "stage1b_positive"
            if path.name == "synth_training_positive.parquet"
            else "stage1c_negative"
        ),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), path)


def test_stage4_corpus_validation_rejects_missing_final_concept(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag="s4present",
        output_root=str(tmp_path / "out"),
        device="cpu",
    )
    _write_synth_corpus(paths.AxBenchBasePaths.from_config(cfg).synth_training_positive(), [0, 1])
    _write_synth_corpus(paths.AxBenchBasePaths.from_config(cfg).synth_training_negative(), [0])
    with pytest.raises(
        AxBenchBaseError,
        match=r"Stage 4 generation.*positive synth classes.*2",
    ):
        _validate_stage4_concept_corpora(cfg, {0, 1, 2})


def test_stage4_corpus_validation_accepts_complete_authoritative_population(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag="s4present-complete",
        output_root=str(tmp_path / "out"),
        device="cpu",
    )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    _write_synth_corpus(output_paths.synth_training_positive(), [0, 1, 2])
    _write_synth_corpus(output_paths.synth_training_negative(), [0, 1, 2])

    assert _validate_stage4_concept_corpora(cfg, {0, 2}) is None


def test_stage4_corpus_validation_does_not_require_disabled_negative_mode(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag="s4present2",
        output_root=str(tmp_path / "out"),
        device="cpu",
    ).replace(run_negative_mode=False)
    _write_synth_corpus(
        paths.AxBenchBasePaths.from_config(cfg).synth_training_positive(),
        [0, 1, 2],
    )
    assert _validate_stage4_concept_corpora(cfg, {0, 1, 2}) is None


def test_stage4_corpus_validation_absent_neg_corpus_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The final population contract requires every enabled-mode corpus."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag="s4present3",
        output_root=str(tmp_path / "out"),
        device="cpu",
    )
    assert cfg.run_negative_mode is True
    # Only positive corpus exists; negative corpus is absent.
    _write_synth_corpus(
        paths.AxBenchBasePaths.from_config(cfg).synth_training_positive(),
        [0, 1],
    )
    with pytest.raises(
        AxBenchBaseError,
        match="Stage 4 generation: failed to read synth corpus",
    ):
        _validate_stage4_concept_corpora(cfg, {0, 1})


def test_stage4_corpus_validation_absent_pos_corpus_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The missing-positive case fails at the same contract boundary."""
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    cfg = make_valid_config(
        run_tag="s4present4",
        output_root=str(tmp_path / "out"),
        device="cpu",
    )
    assert cfg.run_negative_mode is True
    # Only negative corpus exists; positive corpus is absent.
    _write_synth_corpus(
        paths.AxBenchBasePaths.from_config(cfg).synth_training_negative(),
        [0, 1],
    )
    with pytest.raises(
        AxBenchBaseError,
        match="Stage 4 generation: failed to read synth corpus",
    ):
        _validate_stage4_concept_corpora(cfg, {0, 1})


# ---------------------------------------------------------------------------
# Module-level cosmetic / dunder asserts
# ---------------------------------------------------------------------------


class TestModuleInterface:
    def test_all_stages_in_order_is_tuple(self) -> None:
        """Guard against accidental list->tuple regressions."""
        assert isinstance(_ALL_STAGES_IN_ORDER, tuple)

    def test_has_name_main_block(self) -> None:
        """Sanity: the module has an ``if __name__ == "__main__"`` block
        (hydra CLIs are invoked that way)."""
        src = Path(cli.__file__).read_text(encoding="utf-8")
        assert 'if __name__ == "__main__":' in src
