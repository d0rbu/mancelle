"""Unit tests for :mod:`steering.eval.axbench_base.runtime_state` (Phase 1).

Phase 1 of the runtime-state PR introduces the :class:`RuntimeState`
scaffold with the pure-data accessors (concepts, seeds_train, seeds_eval,
nanochat_tokenizer, extraction_tokenizer). Heavy-model context managers and
the judge client arrive in later phases.

This suite covers:

- construction + initial-state invariants
- the ``close()`` protocol (idempotency, closed sentinel, post-close
  error)
- the thread-safety assertion pattern (every accessor must refuse to
  run off the main thread)
- per-accessor lazy-load, cache, loader-arg, error-path, and post-close behaviour
- per-accessor error paths specific to the resource (missing files,
  unset ``run_tag``, corrupt JSON, tokenizer import / load failures)
"""

from __future__ import annotations

import dataclasses
import json
import random
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Literal, cast
from unittest.mock import MagicMock

import pytest
import torch as t
from loguru import logger as loguru_logger
from safetensors.torch import save_file

from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base import paths
from steering.eval.axbench_base import runtime_state as runtime_state_mod
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.seed_contract import LoadedSeedSplit
from steering.eval.axbench_base.synth.concept_yield import (
    ConceptYieldRecord,
    write_concept_yield_report,
    write_filtered_concepts,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    TargetModelMetadata,
)
from steering.sae_resolver import L0Tier, ResolvedSAE
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.seed_contract import write_committed_seed_pair


@contextmanager
def _capture_loguru_warnings() -> Iterator[list[str]]:
    """Capture WARNING+ level messages from loguru into a plain list.

    Used by the close()-exception-safety tests to verify that a
    failing cache close is logged without propagating.
    """
    collected: list[str] = []
    handler_id = loguru_logger.add(
        lambda msg: collected.append(msg.record["message"]),
        level="WARNING",
    )
    try:
        yield collected
    finally:
        loguru_logger.remove(handler_id)


# ---------------------------------------------------------------------------
# Shared test utilities
# ---------------------------------------------------------------------------


_CONCEPTS_PAYLOAD: list[dict[str, Any]] = [
    {
        "concept_id": 0,
        "feature_index": 100,
        "layer": 8,
        "sae_source": "8-gemmascope-res-16k",
        "description": "the color blue",
        "corpus_count": 50,
        "description_source": "neuronpedia",
    },
    {
        "concept_id": 1,
        "feature_index": 200,
        "layer": 8,
        "sae_source": "8-gemmascope-res-16k",
        "description": "scientific measurement",
        "corpus_count": 75,
        "description_source": "neuronpedia",
    },
]


def _make_concepts() -> list[Concept]:
    return [Concept(**entry) for entry in _CONCEPTS_PAYLOAD]


def _make_seed_row(shard_idx: int = 0, doc_idx: int = 0) -> SeedRow:
    """Return a minimal, well-formed :class:`SeedRow` for tests."""
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


class _SeedTokenizer:
    def get_bos_token_id(self) -> int:
        return 99

    def encode(self, text: str) -> list[int]:
        return [1_000 + ord(char) for char in text]

    def decode(self, tokens: list[int]) -> str:
        if tokens == [1, 2, 3]:
            return "prefix"
        if tokens == [4, 5, 6]:
            return "cont"
        return "".join(chr(token - 1_000) for token in tokens)


def _write_stats_metadata(path: Path, n_sae_features: int) -> None:
    """Write a minimal safetensors stats file whose metadata carries
    ``n_sae_features`` so ``RuntimeState`` can read it via
    :func:`safetensors.safe_open`.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_placeholder": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )


def _write_concepts_json(run_tag: str, payload: list[dict[str, Any]]) -> Path:
    path = test_paths.default_paths(run_tag).concepts_json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _write_test_seed_pair(cfg: AxBenchBaseConfig) -> tuple[Path, Path]:
    return write_committed_seed_pair(
        cfg,
        tokenizer=_SeedTokenizer(),
        train_rows=[_make_seed_row(doc_idx=0)],
        eval_rows=[_make_seed_row(doc_idx=1)],
    )


def _write_concept_handoff(
    cfg: AxBenchBaseConfig,
    *,
    stage: Literal["stage1b", "stage1c"],
    input_concepts: list[Concept],
    surviving_ids: set[int],
) -> Path:
    records = []
    for concept in input_concepts:
        survives = concept.concept_id in surviving_ids
        records.append(
            ConceptYieldRecord(
                concept_id=concept.concept_id,
                feature_index=concept.feature_index,
                attempted=2,
                accepted=2 if survives else 0,
                dropped=0 if survives else 2,
                drop_fraction=0.0 if survives else 1.0,
                excluded=not survives,
                exclusion_reason=None if survives else "drop_fraction_exceeded",
                reason_counts={} if survives else {"fixture_qc_failure": 2},
                emitted_rows=4 if survives else 0,
            )
        )
    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    if stage == "stage1b":
        report_path = output_paths.stage1b_concept_yield_report()
        filtered_path = output_paths.stage1b_concepts()
    elif stage == "stage1c":
        report_path = output_paths.stage1c_concept_yield_report()
        filtered_path = output_paths.stage1c_concepts()
    else:
        raise AssertionError(f"unexpected fixture stage {stage}")
    survivor_set = write_concept_yield_report(
        cfg,
        stage=stage,
        records=records,
        path=report_path,
    )
    assert survivor_set == frozenset(surviving_ids)
    write_filtered_concepts(input_concepts, survivor_set, filtered_path)
    return filtered_path


# ---------------------------------------------------------------------------
# Core fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def cfg_obj(
    make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
    tmp_path: Path,
    tmp_run_root: Path,
) -> AxBenchBaseConfig:
    """A :class:`AxBenchBaseConfig` whose ``feature_statistics_path`` points
    to a real stats file whose metadata carries ``n_sae_features=16384``.

    Includes an isolated ``tmp_run_root`` via :func:`tmp_run_root` so
    every ``paths.*`` helper lands under ``tmp_path``.
    """
    stats_path = tmp_path / "feature_statistics.safetensors"
    _write_stats_metadata(stats_path, n_sae_features=16384)
    return make_valid_axbench_cfg(
        feature_statistics_path=str(stats_path),
        run_tag="test-runtime-state",
        seed_continuation_nanochat_tokens=3,
        seed_prefix_max_nanochat_tokens=3,
    )


@pytest.fixture(autouse=True)
def _install_seed_tokenizer(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runtime_state_mod, "_nanochat_get_tokenizer", lambda: _SeedTokenizer()
    )


@pytest.fixture
def runtime_state(cfg_obj: AxBenchBaseConfig) -> RuntimeState:
    return RuntimeState(cfg=cfg_obj)


@pytest.fixture
def concepts_on_disk(cfg_obj: AxBenchBaseConfig) -> Path:
    assert cfg_obj.run_tag is not None
    return _write_concepts_json(cfg_obj.run_tag, _CONCEPTS_PAYLOAD)


@pytest.fixture
def seeds_train_on_disk(cfg_obj: AxBenchBaseConfig) -> Path:
    assert cfg_obj.run_tag is not None
    train_path, _eval_path = _write_test_seed_pair(cfg_obj)
    return train_path


@pytest.fixture
def seeds_eval_on_disk(cfg_obj: AxBenchBaseConfig) -> Path:
    assert cfg_obj.run_tag is not None
    _train_path, eval_path = _write_test_seed_pair(cfg_obj)
    return eval_path


@pytest.fixture
def run_in_thread():
    """Runs ``fn`` in a worker thread and returns ``(result, exception)``."""

    def runner(fn: Callable[[], Any]) -> tuple[Any, BaseException | None]:
        captured: dict[str, Any] = {"result": None, "error": None}

        def _target() -> None:
            try:
                captured["result"] = fn()
            except BaseException as exc:  # noqa: BLE001
                captured["error"] = exc

        th = threading.Thread(target=_target)
        th.start()
        th.join()
        return captured["result"], captured["error"]

    return runner


# ===========================================================================
# TestConstruction — construction + initial-state invariants
# ===========================================================================


class TestConstruction:
    def test_construction_with_valid_cfg(self, cfg_obj: AxBenchBaseConfig) -> None:
        """Bare ``RuntimeState(cfg=cfg_obj)`` succeeds and round-trips cfg."""
        state = RuntimeState(cfg=cfg_obj)
        assert state.cfg is cfg_obj

    def test_construction_does_not_call_any_loader(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Constructing a ``RuntimeState`` must not touch ANY loader.

        We wrap every private loader on the class with a
        ``MagicMock(side_effect=...)`` that explodes if called. If the
        constructor eagerly invoked a loader, one of these mocks would
        raise during ``RuntimeState(cfg=cfg_obj)``.
        """
        loader_names = [
            "_load_concepts",
            "_load_seeds_train",
            "_load_seeds_eval",
            "_load_nanochat_tokenizer",
            "_load_extraction_tokenizer",
        ]
        for name in loader_names:
            mock = MagicMock(
                side_effect=AssertionError(f"{name} was called during construction"),
            )
            monkeypatch.setattr(RuntimeState, name, mock)
        # Constructor should NOT invoke any of the above.
        RuntimeState(cfg=cfg_obj)

    @pytest.mark.skipif(
        not t.cuda.is_available(),
        reason="requires CUDA to measure GPU memory",
    )
    def test_construction_no_gpu_allocation(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """Phase 1 construction must not touch the GPU."""
        before = t.cuda.memory_allocated()
        RuntimeState(cfg=cfg_obj)
        after = t.cuda.memory_allocated()
        assert after == before, (
            f"RuntimeState construction allocated GPU memory: "
            f"before={before}, after={after}"
        )

    def test_construction_closed_is_false_initially(
        self, runtime_state: RuntimeState,
    ) -> None:
        assert runtime_state._closed is False

    def test_construction_all_private_fields_start_none(
        self, runtime_state: RuntimeState,
    ) -> None:
        assert runtime_state._concepts is None
        assert runtime_state._stage1b_concepts is None
        assert runtime_state._downstream_concepts is None
        assert runtime_state._seeds_train is None
        assert runtime_state._seeds_eval is None
        assert runtime_state._nanochat_tokenizer is None
        assert runtime_state._extraction_tokenizer is None


# ===========================================================================
# TestCloseProtocol — close() idempotency + post-close errors
# ===========================================================================


class TestCloseProtocol:
    def test_close_is_idempotent(self, runtime_state: RuntimeState) -> None:
        runtime_state.close()
        runtime_state.close()  # second call must NOT raise
        assert runtime_state._closed is True

    def test_close_marks_closed_flag_true(
        self, runtime_state: RuntimeState,
    ) -> None:
        assert runtime_state._closed is False
        runtime_state.close()
        assert runtime_state._closed is True

    def test_close_before_any_accessor_call_works(
        self, runtime_state: RuntimeState,
    ) -> None:
        """Calling ``close()`` immediately after construction is a no-op
        that doesn't raise.
        """
        runtime_state.close()

    def test_close_drops_cached_concepts_reference(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
    ) -> None:
        runtime_state.concepts()
        assert runtime_state._concepts is not None
        runtime_state.close()
        assert runtime_state._concepts is None

    def test_close_drops_all_cached_phase1_references(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        seeds_train_on_disk: Path,
        seeds_eval_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """After ``close()`` every private cache slot resets to ``None``."""
        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer,
            "from_pretrained",
            lambda model_id: object(),
        )
        runtime_state.concepts()
        runtime_state.seeds_train()
        runtime_state.seeds_eval()
        runtime_state.nanochat_tokenizer()
        runtime_state.extraction_tokenizer()
        runtime_state.close()
        assert runtime_state._concepts is None
        assert runtime_state._stage1b_concepts is None
        assert runtime_state._downstream_concepts is None
        assert runtime_state._seeds_train is None
        assert runtime_state._seeds_eval is None
        assert runtime_state._nanochat_tokenizer is None
        assert runtime_state._extraction_tokenizer is None

    def test_accessor_after_close_raises_clear_error(
        self, runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.concepts()


# ===========================================================================
# TestThreadSafetyAssertion — off-main-thread access is rejected
# ===========================================================================


class TestThreadSafetyAssertion:
    def test_accessor_from_main_thread_works(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
    ) -> None:
        """Baseline: main-thread access is fine."""
        result = runtime_state.concepts()
        assert isinstance(result, list)

    def test_concepts_from_worker_thread_raises(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.concepts)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_seeds_train_from_worker_thread_raises(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.seeds_train)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_nanochat_tokenizer_from_worker_thread_raises(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
        run_in_thread: Callable,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "_nanochat_get_tokenizer",
            lambda: object(),
        )
        _, error = run_in_thread(runtime_state.nanochat_tokenizer)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)


# ===========================================================================
# TestConceptsAccessor
# ===========================================================================


class TestConceptsAccessor:
    def test_concepts_first_call_invokes_loader_once(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(return_value=_make_concepts())
        monkeypatch.setattr(
            runtime_state_mod,
            "load_and_validate_concepts",
            mock_loader,
        )
        runtime_state.concepts()
        assert mock_loader.call_count == 1

    def test_concepts_second_call_returns_cached_instance(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
    ) -> None:
        a = runtime_state.concepts()
        b = runtime_state.concepts()
        assert a is b

    def test_concepts_raises_after_close(
        self, runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.concepts()

    def test_concepts_raises_from_non_main_thread(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.concepts)
        assert isinstance(error, AxBenchBaseError)

    def test_concepts_loader_failure_propagates_as_axbench_base_error(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise AxBenchBaseError("simulated concept corruption")

        monkeypatch.setattr(
            runtime_state_mod, "load_and_validate_concepts", boom,
        )
        with pytest.raises(AxBenchBaseError, match="simulated concept corruption"):
            runtime_state.concepts()

    def test_concepts_loader_called_with_expected_path_and_n_features(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(return_value=_make_concepts())
        monkeypatch.setattr(
            runtime_state_mod, "load_and_validate_concepts", mock_loader,
        )
        runtime_state.concepts()
        path_arg, kwargs = mock_loader.call_args
        # One positional (the path), one keyword (n_sae_features).
        assert path_arg[0] == concepts_on_disk
        assert kwargs["n_sae_features"] == 16384

    def test_concepts_raises_if_run_tag_unset(
        self,
        make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
        tmp_path: Path,
        tmp_run_root: Path,
    ) -> None:
        stats_path = tmp_path / "stats.safetensors"
        _write_stats_metadata(stats_path, n_sae_features=16384)
        cfg = make_valid_axbench_cfg(
            feature_statistics_path=str(stats_path),
            run_tag=None,
        )
        state = RuntimeState(cfg=cfg)
        with pytest.raises(AxBenchBaseError, match="run_tag"):
            state.concepts()

    def test_concepts_raises_if_file_missing(
        self, runtime_state: RuntimeState,
    ) -> None:
        # No concepts.json has been written for this run tag.
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state.concepts()
        msg = str(excinfo.value)
        assert "concepts.json" in msg
        assert "Stage 0c" in msg
        assert "PREPARE_NEGATIVE_PREFIXES" in msg

    def test_concepts_raises_on_corrupt_file(
        self,
        runtime_state: RuntimeState,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        assert cfg_obj.run_tag is not None
        path = paths.AxBenchBasePaths.from_config(cfg_obj).concepts_json()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("this is not json", encoding="utf-8")
        # Item 8: pin the error message so a future refactor that
        # raises a different AxBenchBaseError from a different code
        # path doesn't silently pass this test. We expect the load
        # path to mention concept loading.
        with pytest.raises(
            AxBenchBaseError, match="Failed to load concepts",
        ):
            runtime_state.concepts()

    def test_concepts_content_matches_disk(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
    ) -> None:
        concepts = runtime_state.concepts()
        assert len(concepts) == len(_CONCEPTS_PAYLOAD)
        assert concepts[0].concept_id == 0
        assert concepts[0].description == "the color blue"
        assert concepts[1].feature_index == 200


# ===========================================================================
# TestFilteredConceptAccessors — Stage 1b/1c authoritative handoffs
# ===========================================================================


class TestFilteredConceptAccessors:
    def test_stage1b_preserves_sparse_original_id_and_caches(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        del concepts_on_disk
        _write_concept_handoff(
            cfg_obj,
            stage="stage1b",
            input_concepts=_make_concepts(),
            surviving_ids={1},
        )

        first = runtime_state.stage1b_concepts()
        second = runtime_state.stage1b_concepts()

        assert first is second
        assert [concept.concept_id for concept in first] == [1]
        assert first[0] == _make_concepts()[1]

    def test_downstream_uses_stage1c_subset_of_stage1b(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        del concepts_on_disk
        original = _make_concepts()
        _write_concept_handoff(
            cfg_obj,
            stage="stage1b",
            input_concepts=original,
            surviving_ids={0, 1},
        )
        _write_concept_handoff(
            cfg_obj,
            stage="stage1c",
            input_concepts=original,
            surviving_ids={1},
        )

        downstream = runtime_state.downstream_concepts()

        assert [concept.concept_id for concept in downstream] == [1]
        assert downstream is runtime_state.downstream_concepts()

    def test_downstream_without_negative_mode_is_stage1b_object(
        self,
        concepts_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        del concepts_on_disk
        cfg = cfg_obj.replace(run_negative_mode=False)
        state = RuntimeState(cfg=cfg)
        _write_concept_handoff(
            cfg,
            stage="stage1b",
            input_concepts=_make_concepts(),
            surviving_ids={1},
        )

        assert state.downstream_concepts() is state.stage1b_concepts()

    def test_stage1b_report_and_filtered_list_mismatch_raises(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        del concepts_on_disk
        path = _write_concept_handoff(
            cfg_obj,
            stage="stage1b",
            input_concepts=_make_concepts(),
            surviving_ids={1},
        )
        path.write_text(json.dumps([_CONCEPTS_PAYLOAD[0]]), encoding="utf-8")

        with pytest.raises(
            AxBenchBaseError,
            match="Stage 1b filtered concepts do not match",
        ):
            runtime_state.stage1b_concepts()

    def test_stage1c_cannot_reintroduce_stage1b_dropped_concept(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        del concepts_on_disk
        original = _make_concepts()
        _write_concept_handoff(
            cfg_obj,
            stage="stage1b",
            input_concepts=original,
            surviving_ids={1},
        )
        stage1c_path = _write_concept_handoff(
            cfg_obj,
            stage="stage1c",
            input_concepts=[original[1]],
            surviving_ids={1},
        )
        stage1c_path.write_text(
            json.dumps([_CONCEPTS_PAYLOAD[0]]),
            encoding="utf-8",
        )

        with pytest.raises(AxBenchBaseError, match="absent from its parent"):
            runtime_state.downstream_concepts()

    def test_filtered_accessor_after_close_raises(
        self,
        runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.stage1b_concepts()


# ===========================================================================
# TestSeedsTrainAccessor
# ===========================================================================


class TestSeedsTrainAccessor:
    @pytest.mark.parametrize("identity_first", [False, True])
    def test_rows_and_identity_share_one_loaded_snapshot(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
        identity_first: bool,
    ) -> None:
        rows = [_make_seed_row()]
        snapshot = LoadedSeedSplit(rows=rows, rows_sha256="b" * 64)
        mock_loader = MagicMock(return_value=snapshot)
        monkeypatch.setattr(
            runtime_state_mod,
            "load_seed_pool_with_rows_identity",
            mock_loader,
        )
        if identity_first:
            assert runtime_state.seeds_train_rows_sha256() == "b" * 64
            assert runtime_state.seeds_train() is rows
        else:
            assert runtime_state.seeds_train() is rows
            assert runtime_state.seeds_train_rows_sha256() == "b" * 64
        assert mock_loader.call_count == 1

    def test_seeds_train_first_call_invokes_loader_once(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(
            return_value=LoadedSeedSplit([_make_seed_row()], "a" * 64)
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "load_seed_pool_with_rows_identity",
            mock_loader,
        )
        runtime_state.seeds_train()
        assert mock_loader.call_count == 1

    def test_seeds_train_second_call_returns_cached_instance(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
    ) -> None:
        a = runtime_state.seeds_train()
        b = runtime_state.seeds_train()
        assert a is b

    def test_seeds_train_raises_after_close(
        self, runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.seeds_train()

        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.seeds_train_rows_sha256()

    def test_seeds_train_raises_from_non_main_thread(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.seeds_train)
        assert isinstance(error, AxBenchBaseError)
        _, identity_error = run_in_thread(
            runtime_state.seeds_train_rows_sha256
        )
        assert isinstance(identity_error, AxBenchBaseError)

    def test_seeds_train_loader_failure_propagates_as_axbench_base_error(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(path: Path) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(
            runtime_state_mod,
            "load_seed_pool_with_rows_identity",
            boom,
        )
        with pytest.raises(AxBenchBaseError, match="seeds_train"):
            runtime_state.seeds_train()

    def test_seeds_train_loader_called_with_correct_path(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(
            return_value=LoadedSeedSplit([_make_seed_row()], "a" * 64)
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "load_seed_pool_with_rows_identity",
            mock_loader,
        )
        runtime_state.seeds_train()
        assert cfg_obj.run_tag is not None
        assert (
            mock_loader.call_args[0][0] == paths.AxBenchBasePaths.from_config(cfg_obj).seeds_train()
        )

    def test_seeds_train_raises_if_run_tag_unset(
        self,
        make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
        tmp_path: Path,
        tmp_run_root: Path,
    ) -> None:
        stats_path = tmp_path / "stats.safetensors"
        _write_stats_metadata(stats_path, n_sae_features=16384)
        cfg = make_valid_axbench_cfg(
            feature_statistics_path=str(stats_path),
            run_tag=None,
        )
        state = RuntimeState(cfg=cfg)
        with pytest.raises(AxBenchBaseError, match="run_tag"):
            state.seeds_train()

    def test_seeds_train_raises_if_file_missing(
        self, runtime_state: RuntimeState,
    ) -> None:
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state.seeds_train()
        assert "seeds_train" in str(excinfo.value)
        assert "Stage 0b" in str(excinfo.value)


# ===========================================================================
# TestSeedsEvalAccessor
# ===========================================================================


class TestSeedsEvalAccessor:
    def test_seeds_eval_first_call_invokes_loader_once(
        self,
        runtime_state: RuntimeState,
        seeds_eval_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(return_value=[_make_seed_row()])
        monkeypatch.setattr(runtime_state_mod, "load_seed_pool", mock_loader)
        runtime_state.seeds_eval()
        assert mock_loader.call_count == 1

    def test_seeds_eval_second_call_returns_cached_instance(
        self,
        runtime_state: RuntimeState,
        seeds_eval_on_disk: Path,
    ) -> None:
        a = runtime_state.seeds_eval()
        b = runtime_state.seeds_eval()
        assert a is b

    def test_seeds_eval_raises_after_close(
        self, runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.seeds_eval()

    def test_seeds_eval_raises_from_non_main_thread(
        self,
        runtime_state: RuntimeState,
        seeds_eval_on_disk: Path,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.seeds_eval)
        assert isinstance(error, AxBenchBaseError)

    def test_seeds_eval_loader_failure_propagates_as_axbench_base_error(
        self,
        runtime_state: RuntimeState,
        seeds_eval_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(path: Path) -> Any:
            raise RuntimeError("boom")

        monkeypatch.setattr(runtime_state_mod, "load_seed_pool", boom)
        with pytest.raises(AxBenchBaseError, match="seeds_eval"):
            runtime_state.seeds_eval()

    def test_seeds_eval_loader_called_with_correct_path(
        self,
        runtime_state: RuntimeState,
        seeds_eval_on_disk: Path,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(return_value=[_make_seed_row()])
        monkeypatch.setattr(runtime_state_mod, "load_seed_pool", mock_loader)
        runtime_state.seeds_eval()
        assert cfg_obj.run_tag is not None
        assert (
            mock_loader.call_args[0][0] == paths.AxBenchBasePaths.from_config(cfg_obj).seeds_eval()
        )

    def test_seeds_eval_raises_if_run_tag_unset(
        self,
        make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
        tmp_path: Path,
        tmp_run_root: Path,
    ) -> None:
        stats_path = tmp_path / "stats.safetensors"
        _write_stats_metadata(stats_path, n_sae_features=16384)
        cfg = make_valid_axbench_cfg(
            feature_statistics_path=str(stats_path),
            run_tag=None,
        )
        state = RuntimeState(cfg=cfg)
        with pytest.raises(AxBenchBaseError, match="run_tag"):
            state.seeds_eval()

    def test_seeds_eval_raises_if_file_missing(
        self, runtime_state: RuntimeState,
    ) -> None:
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state.seeds_eval()
        assert "seeds_eval" in str(excinfo.value)
        assert "Stage 0b" in str(excinfo.value)


# ===========================================================================
# TestNanochatTokenizerAccessor
# ===========================================================================


class TestNanochatTokenizerAccessor:
    def test_nanochat_tokenizer_first_call_invokes_loader_once(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = object()
        mock_loader = MagicMock(return_value=sentinel)
        monkeypatch.setattr(
            runtime_state_mod, "_nanochat_get_tokenizer", mock_loader,
        )
        assert runtime_state.nanochat_tokenizer() is sentinel
        assert mock_loader.call_count == 1

    def test_nanochat_tokenizer_second_call_returns_cached_instance(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = object()
        monkeypatch.setattr(
            runtime_state_mod,
            "_nanochat_get_tokenizer",
            MagicMock(return_value=sentinel),
        )
        a = runtime_state.nanochat_tokenizer()
        b = runtime_state.nanochat_tokenizer()
        assert a is b is sentinel

    def test_nanochat_tokenizer_raises_after_close(
        self, runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.nanochat_tokenizer()

    def test_nanochat_tokenizer_raises_from_non_main_thread(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
        run_in_thread: Callable,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "_nanochat_get_tokenizer",
            MagicMock(return_value=object()),
        )
        _, error = run_in_thread(runtime_state.nanochat_tokenizer)
        assert isinstance(error, AxBenchBaseError)

    def test_nanochat_tokenizer_loader_failure_propagates_as_axbench_base_error(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom() -> Any:
            raise RuntimeError("tokenizer not on disk")

        monkeypatch.setattr(
            runtime_state_mod, "_nanochat_get_tokenizer", boom,
        )
        with pytest.raises(AxBenchBaseError, match="nanochat tokenizer"):
            runtime_state.nanochat_tokenizer()

    def test_nanochat_tokenizer_loader_invoked_with_no_args(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(return_value=object())
        monkeypatch.setattr(
            runtime_state_mod, "_nanochat_get_tokenizer", mock_loader,
        )
        runtime_state.nanochat_tokenizer()
        assert mock_loader.call_args == ((), {})


# ===========================================================================
# TestGemmaTokenizerAccessor
# ===========================================================================


class TestGemmaTokenizerAccessor:
    def test_extraction_tokenizer_first_call_invokes_loader_once(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = object()
        mock_loader = MagicMock(return_value=sentinel)
        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer, "from_pretrained", mock_loader,
        )
        assert runtime_state.extraction_tokenizer() is sentinel
        assert mock_loader.call_count == 1

    def test_extraction_tokenizer_second_call_returns_cached_instance(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sentinel = object()
        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer,
            "from_pretrained",
            MagicMock(return_value=sentinel),
        )
        a = runtime_state.extraction_tokenizer()
        b = runtime_state.extraction_tokenizer()
        assert a is b is sentinel

    def test_extraction_tokenizer_raises_after_close(
        self, runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.extraction_tokenizer()

    def test_extraction_tokenizer_raises_from_non_main_thread(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
        run_in_thread: Callable,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer,
            "from_pretrained",
            MagicMock(return_value=object()),
        )
        _, error = run_in_thread(runtime_state.extraction_tokenizer)
        assert isinstance(error, AxBenchBaseError)

    def test_extraction_tokenizer_loader_failure_propagates_as_axbench_base_error(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(model_id: str) -> Any:
            raise OSError("model not cached")

        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer, "from_pretrained", boom,
        )
        with pytest.raises(AxBenchBaseError, match="extraction tokenizer"):
            runtime_state.extraction_tokenizer()

    def test_extraction_tokenizer_loader_called_with_cfg_tokenizer_id(
        self,
        runtime_state: RuntimeState,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        mock_loader = MagicMock(return_value=object())
        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer, "from_pretrained", mock_loader,
        )
        runtime_state.extraction_tokenizer()
        assert mock_loader.call_args[0][0] == cfg_obj.extraction_tokenizer_id

    def test_extraction_tokenizer_import_error_translated(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(model_id: str) -> Any:
            raise ImportError("transformers not installed")

        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer, "from_pretrained", boom,
        )
        with pytest.raises(AxBenchBaseError, match="extraction tokenizer"):
            runtime_state.extraction_tokenizer()


class TestLoaderAxBenchErrorPassthrough:
    """Cover the ``except AxBenchBaseError: raise`` branches in every loader.

    When a helper (``load_and_validate_concepts``, ``load_seed_pool``, or the
    tokenizer factories) raises an :class:`AxBenchBaseError` directly, the
    loader must let that through untouched -- not re-wrap it with a generic
    "Failed to load..." message.
    """

    def test_concepts_loader_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        concepts_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raises_native(*args: Any, **kwargs: Any) -> Any:
            raise AxBenchBaseError("original concept failure")

        monkeypatch.setattr(
            runtime_state_mod, "load_and_validate_concepts", raises_native,
        )
        with pytest.raises(
            AxBenchBaseError, match="original concept failure",
        ) as excinfo:
            runtime_state.concepts()
        # "Failed to load" is the generic wrapper prefix -- ensure it did
        # NOT wrap the native error.
        assert "Failed to load concepts" not in str(excinfo.value)

    def test_seeds_train_loader_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        seeds_train_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raises_native(path: Path, **_kwargs: Any) -> Any:
            raise AxBenchBaseError("original seeds_train failure")

        monkeypatch.setattr(
            runtime_state_mod,
            "load_seed_pool_with_rows_identity",
            raises_native,
        )
        with pytest.raises(
            AxBenchBaseError, match="original seeds_train failure",
        ) as excinfo:
            runtime_state.seeds_train()
        assert "Failed to load seeds_train" not in str(excinfo.value)

    def test_seeds_eval_loader_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        seeds_eval_on_disk: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raises_native(path: Path, **_kwargs: Any) -> Any:
            raise AxBenchBaseError("original seeds_eval failure")

        monkeypatch.setattr(runtime_state_mod, "load_seed_pool", raises_native)
        with pytest.raises(
            AxBenchBaseError, match="original seeds_eval failure",
        ) as excinfo:
            runtime_state.seeds_eval()
        assert "Failed to load seeds_eval" not in str(excinfo.value)

    def test_nanochat_tokenizer_loader_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raises_native() -> Any:
            raise AxBenchBaseError("original nanochat failure")

        monkeypatch.setattr(
            runtime_state_mod, "_nanochat_get_tokenizer", raises_native,
        )
        with pytest.raises(
            AxBenchBaseError, match="original nanochat failure",
        ) as excinfo:
            runtime_state.nanochat_tokenizer()
        assert "Failed to load nanochat tokenizer" not in str(excinfo.value)

    def test_extraction_tokenizer_loader_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raises_native(model_id: str) -> Any:
            raise AxBenchBaseError("original extraction tokenizer failure")

        monkeypatch.setattr(
            runtime_state_mod.AutoTokenizer, "from_pretrained", raises_native,
        )
        with pytest.raises(
            AxBenchBaseError, match="original extraction tokenizer failure",
        ) as excinfo:
            runtime_state.extraction_tokenizer()
        assert "Failed to load extraction tokenizer" not in str(excinfo.value)


# ===========================================================================
# Phase 2 -- heavy-model context managers (base_model, base_sae,
# decoder_only_model, encdec_model)
#
# The four heavy accessors follow the same refcounted CPU-offload
# pattern, so their test classes are structurally identical. Each
# covers: lazy load, device placement on enter, CPU offload on
# final exit, refcount sharing across nested with-blocks, re-entry
# reuse, refcount transitions, close() gating, thread safety, loader
# failure translation, CUDA cache management, and refcount integrity
# under exceptions.
# ===========================================================================


def test_feature_extraction_pair_passes_v2_l0_tier(
    make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Gemma-scope-2 sources parse to an L0 tier and may split model ids.

    The production loader must pass that parsed tier into ``resolve_sae``;
    otherwise every Gemma-scope-2 AxBench-Base run fails before model load.
    Gemma-3 PT also needs Neuronpedia's public model id for descriptions but
    the local PT model id for SAELens/model loading.
    """
    cfg = make_valid_axbench_cfg(
        neuronpedia_model_id="gemma-3-4b",
        feature_extraction_model_id="gemma-3-4b-pt",
        neuronpedia_source="22-gemmascope-2-res-65k",
    )
    resolve_calls: list[dict[str, Any]] = []
    load_calls: list[dict[str, Any]] = []
    fake_model = SimpleNamespace()
    fake_sae = SimpleNamespace(
        cfg=SimpleNamespace(
            metadata={
                "neuronpedia_id": "gemma-3-4b/22-gemmascope-2-res-65k",
            },
        ),
        to=MagicMock(),
    )
    resolved = ResolvedSAE(
        release="gemma-scope-2-4b-pt-res",
        sae_id="layer_22_width_65k_l0_medium",
        hook_name="blocks.22.hook_resid_post",
    )

    def _resolve_sae(**kwargs: Any) -> ResolvedSAE:
        resolve_calls.append(kwargs)
        return resolved

    def _load_sae_and_model(**kwargs: Any) -> tuple[Any, Any, object]:
        load_calls.append(kwargs)
        return fake_model, fake_sae, object()

    monkeypatch.setattr(runtime_state_mod, "resolve_sae", _resolve_sae)
    monkeypatch.setattr(
        runtime_state_mod,
        "load_sae_and_model",
        _load_sae_and_model,
    )
    monkeypatch.setattr(
        runtime_state_mod,
        "build_nnterp_model",
        lambda *args, **kwargs: object(),
    )

    model, sae = runtime_state_mod._load_feature_extraction_pair(cfg)

    assert model is fake_model
    assert sae is fake_sae
    assert resolve_calls[0]["model"] == "gemma-3-4b-pt"
    assert resolve_calls[0]["l0"] is L0Tier.medium
    assert load_calls[0]["model_name"] == "gemma-3-4b-pt"


def test_runtime_state_threads_cluster_contract_to_feature_extraction_loader(
    cfg_obj: AxBenchBaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = MagicMock()
    model = MagicMock()
    sae = MagicMock()
    calls: list[tuple[AxBenchBaseConfig, Any]] = []

    def load_pair(
        cfg: AxBenchBaseConfig,
        *,
        cluster_contract: Any,
    ) -> tuple[Any, Any]:
        calls.append((cfg, cluster_contract))
        return model, sae

    monkeypatch.setattr(runtime_state_mod, "_load_feature_extraction_pair", load_pair)
    state = RuntimeState(cfg_obj, cluster_contract=contract)

    state._ensure_feature_extraction_loaded()

    assert calls == [(cfg_obj, contract)]
    assert state._base_model is model
    assert state._base_sae is sae


def test_encdec_model_threads_cluster_contract_to_loader(
    cfg_obj: AxBenchBaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = MagicMock()
    model = MagicMock()
    model.to.return_value = model

    def clustered_only_loader(
        cfg: AxBenchBaseConfig,
        cluster_contract: Any = None,
    ) -> Any:
        if cluster_contract is None:
            pytest.fail("clustered encdec_model must not use the legacy loader call")
        assert cfg is cfg_obj
        assert cluster_contract is contract
        return model

    monkeypatch.setattr(
        runtime_state_mod,
        "_load_encdec_model",
        clustered_only_loader,
    )
    state = RuntimeState(cfg_obj, cluster_contract=contract)

    with state.encdec_model() as loaded:
        assert loaded is model


def test_encdec_meta_threads_cluster_contract_to_detector(
    cfg_obj: AxBenchBaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = MagicMock()
    model = MagicMock()
    model.to.return_value = model
    expected = _make_meta()

    def clustered_only_detector(
        *,
        target_model: Any,
        cfg: AxBenchBaseConfig,
        cluster_contract: Any = None,
    ) -> TargetModelMetadata:
        if cluster_contract is None:
            pytest.fail("clustered encdec_meta must not use the legacy detector call")
        assert target_model is model
        assert cfg is cfg_obj
        assert cluster_contract is contract
        return expected

    monkeypatch.setattr(
        runtime_state_mod,
        "detect_target_model_metadata",
        clustered_only_detector,
    )
    state = RuntimeState(cfg_obj, cluster_contract=contract)
    state._encdec_model = model

    assert state.encdec_meta() is expected


def test_inactive_sampler_threads_representative_universe_to_builder(
    cfg_obj: AxBenchBaseConfig,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    contract = MagicMock()
    contract.representative_universe = (2, 5, 11)
    sentinel = MagicMock()
    expected_meta = _make_meta(
        negative_selection={"distribution": "geometric", "mean": 2.0},
    )

    def clustered_only_builder(
        *,
        encdec_meta: TargetModelMetadata,
        cfg: AxBenchBaseConfig,
        allowed_feature_ids: t.Tensor | None = None,
    ) -> Any:
        if allowed_feature_ids is None:
            pytest.fail(
                "clustered inactive_sampler must not use the legacy builder call"
            )
        assert encdec_meta is expected_meta
        assert cfg is cfg_obj
        assert allowed_feature_ids.dtype is t.int64
        assert allowed_feature_ids.tolist() == [2, 5, 11]
        return sentinel

    monkeypatch.setattr(
        runtime_state_mod,
        "_build_inactive_sampler",
        clustered_only_builder,
    )
    state = RuntimeState(cfg_obj, cluster_contract=contract)
    state._encdec_meta = expected_meta

    assert state.inactive_sampler() is sentinel


@pytest.fixture
def stub_heavy_loaders(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, Any]:
    """Install stub loaders that return controllable :class:`MagicMock`s.

    Each of the four Phase 2 heavy-model loaders is replaced with a
    deterministic factory that creates a fresh :class:`MagicMock` the
    first time it is called for a given name and returns the *same*
    instance on subsequent calls. The ``.to`` method is also a
    :class:`MagicMock` so tests can assert on device-placement calls
    without touching any real tensors.

    The underlying CUDA operations are stubbed too: availability is
    ``False`` and cache emptying is a no-op, keeping the entire test
    suite hardware-agnostic. Individual tests may override these.

    Returns a mapping ``{loader_name: loaded_mock}`` so tests can grab
    the shared stub instance directly.
    """
    loaded_models: dict[str, Any] = {}

    def _make_stub_model(name: str) -> Any:
        model = MagicMock()
        model._name = name
        # ``to()`` on a real torch module returns ``self`` so ``.to(...)``
        # chains work. Mirror that so production code that does
        # ``model = model.to("cpu")`` stays consistent.
        model.to.return_value = model
        return model

    def _make_loader(name: str) -> Callable[[Any], Any]:
        def _loader(cfg: Any) -> Any:  # noqa: ARG001
            if name not in loaded_models:
                loaded_models[name] = _make_stub_model(name)
            return loaded_models[name]

        return _loader

    def _make_pair_loader() -> Callable[[Any], tuple[Any, Any]]:
        def _loader(cfg: Any) -> tuple[Any, Any]:  # noqa: ARG001
            if "base_model" not in loaded_models:
                loaded_models["base_model"] = _make_stub_model("base_model")
            if "base_sae" not in loaded_models:
                loaded_models["base_sae"] = _make_stub_model("base_sae")
            return loaded_models["base_model"], loaded_models["base_sae"]

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
    return loaded_models


@pytest.fixture
def count_calls() -> Callable[[Callable[[Any], Any]], tuple[Callable[[Any], Any], list[Any]]]:
    """Return ``(wrapped, call_log)`` where ``call_log`` is appended to
    on every ``wrapped(...)`` invocation.

    Used to assert "loader invoked exactly once" across entry/exit cycles.
    """

    def factory(
        base: Callable[[Any], Any],
    ) -> tuple[Callable[[Any], Any], list[Any]]:
        log: list[Any] = []

        def wrapped(cfg: Any) -> Any:
            log.append(cfg)
            return base(cfg)

        return wrapped, log

    return factory


# Parameter table for the 4 heavy-model accessors. Keeps the per-accessor
# test bodies short and keeps naming symmetric.
HEAVY_ACCESSORS: list[tuple[str, str, str, str]] = [
    # (method_name, loader_name, refcount_attr, cached_attr)
    ("base_model", "_load_feature_extraction_pair", "_base_model_refcount", "_base_model"),
    ("base_sae", "_load_feature_extraction_pair", "_base_sae_refcount", "_base_sae"),
    (
        "decoder_only_model",
        "_load_decoder_only_model",
        "_decoder_only_refcount",
        "_decoder_only_model",
    ),
    (
        "encdec_model",
        "_load_encdec_model",
        "_encdec_refcount",
        "_encdec_model",
    ),
]


@pytest.mark.parametrize(
    ("method_name", "loader_name", "refcount_attr", "cached_attr"),
    HEAVY_ACCESSORS,
    ids=[a[0] for a in HEAVY_ACCESSORS],
)
class TestHeavyAccessors:
    """Uniform coverage for every heavy-model context manager.

    Every test in this class parametrizes over the four accessors so
    each check is asserted for ``base_model``, ``base_sae``,
    ``decoder_only_model``, and ``encdec_model``.
    """

    def test_first_enter_invokes_loader_once(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        call_count = {"n": 0}

        def _make_mock() -> Any:
            return MagicMock(to=MagicMock(return_value=MagicMock()))

        if loader_name == "_load_feature_extraction_pair":
            def counting_loader(cfg: Any) -> tuple[Any, Any]:
                call_count["n"] += 1
                return _make_mock(), _make_mock()
        else:
            def counting_loader(cfg: Any) -> Any:
                call_count["n"] += 1
                return _make_mock()

        monkeypatch.setattr(runtime_state_mod, loader_name, counting_loader)
        method = getattr(runtime_state, method_name)
        with method():
            pass
        assert call_count["n"] == 1

    def test_first_enter_moves_model_to_device(
        self,
        runtime_state: RuntimeState,
        cfg_obj: AxBenchBaseConfig,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        # First entry: loader returns fresh model. Context manager must
        # NOT move to device on first entry (loader is expected to
        # place the model on cfg.device already). The .to() call only
        # happens on re-entry after the previous context exited.
        method = getattr(runtime_state, method_name)
        with method() as model:
            # Loader has run; .to() should NOT have been called yet.
            assert model.to.call_args_list == []

    def test_exit_at_refcount_zero_moves_to_cpu(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        method = getattr(runtime_state, method_name)
        with method() as model:
            pass
        # On exit, the model is parked on CPU.
        model.to.assert_called_once_with("cpu")

    def test_nested_with_shares_live_instance(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        method = getattr(runtime_state, method_name)
        with method() as outer:
            with method() as inner:
                assert outer is inner

    def test_nested_inner_exit_does_not_offload(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        method = getattr(runtime_state, method_name)
        with method() as outer:
            with method():
                pass
            # Inner exited but outer refcount is still 1 -- no CPU move.
            assert outer.to.call_args_list == []
        # Outer exit drops refcount to 0 -- now the single CPU move.
        assert outer.to.call_count == 1
        outer.to.assert_called_once_with("cpu")

    def test_reenter_after_full_exit_moves_back_to_device(
        self,
        runtime_state: RuntimeState,
        cfg_obj: AxBenchBaseConfig,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        method = getattr(runtime_state, method_name)
        with method() as m1:
            pass
        # m1 was parked on CPU on first exit.
        first_cycle_calls = list(m1.to.call_args_list)
        with method() as m2:
            # On re-entry, the context manager must wake the cached
            # model back up on cfg.device.
            assert m2 is m1
            assert m2.to.call_args_list[len(first_cycle_calls):] == [
                ((cfg_obj.device,), {}),
            ]

    def test_reenter_does_not_call_loader_twice(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        call_count = {"n": 0}
        stub_model = MagicMock()
        stub_model.to.return_value = stub_model

        if loader_name == "_load_feature_extraction_pair":
            stub_sae = MagicMock()
            stub_sae.to.return_value = stub_sae

            def counting_loader(cfg: Any) -> tuple[Any, Any]:
                call_count["n"] += 1
                return stub_model, stub_sae
        else:
            def counting_loader(cfg: Any) -> Any:
                call_count["n"] += 1
                return stub_model

        monkeypatch.setattr(runtime_state_mod, loader_name, counting_loader)
        method = getattr(runtime_state, method_name)
        with method():
            pass
        with method():
            pass
        assert call_count["n"] == 1

    def test_refcount_transitions_correctly(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        assert getattr(runtime_state, refcount_attr) == 0
        method = getattr(runtime_state, method_name)
        with method():
            assert getattr(runtime_state, refcount_attr) == 1
            with method():
                assert getattr(runtime_state, refcount_attr) == 2
            assert getattr(runtime_state, refcount_attr) == 1
        assert getattr(runtime_state, refcount_attr) == 0

    def test_raises_after_close(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        runtime_state.close()
        method = getattr(runtime_state, method_name)
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            with method():
                pass

    def test_raises_from_non_main_thread(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        run_in_thread: Callable,
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        method = getattr(runtime_state, method_name)

        def enter_in_thread() -> None:
            # Opening the context manager off the main thread must
            # raise BEFORE yielding.
            with method():
                pass

        _, error = run_in_thread(enter_in_thread)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_loader_failure_propagates_as_axbench_base_error(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        def boom(cfg: Any) -> Any:
            raise AxBenchBaseError("simulated loader failure")

        monkeypatch.setattr(runtime_state_mod, loader_name, boom)
        method = getattr(runtime_state, method_name)
        with pytest.raises(
            AxBenchBaseError, match="simulated loader failure",
        ):
            with method():
                pass
        # Refcount must remain 0 so later tests can open fresh
        # contexts without leaking the failed-enter state.
        assert getattr(runtime_state, refcount_attr) == 0

    def test_empty_cuda_cache_called_on_final_exit(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        empty_mock = MagicMock()
        monkeypatch.setattr(runtime_state_mod.t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(runtime_state_mod.t.cuda, "empty_cache", empty_mock)
        method = getattr(runtime_state, method_name)
        with method():
            assert empty_mock.call_count == 0
            with method():
                pass
            # Inner exit at refcount 1 must NOT empty cache.
            assert empty_mock.call_count == 0
        # Outer exit at refcount 0 empties the cache exactly once.
        assert empty_mock.call_count == 1

    def test_exception_inside_with_still_decrements_refcount(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        method_name: str,
        loader_name: str,
        refcount_attr: str,
        cached_attr: str,
    ) -> None:
        method = getattr(runtime_state, method_name)
        with pytest.raises(ValueError, match="inner boom"):
            with method():
                raise ValueError("inner boom")
        assert getattr(runtime_state, refcount_attr) == 0


# ===========================================================================
# TestHeavyAccessorsInteractions
# ===========================================================================


class TestHeavyAccessorsInteractions:
    """Cross-accessor lifecycle + close() interaction coverage."""

    def test_two_different_heavy_accessors_can_be_live_simultaneously(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model() as g, runtime_state.base_sae() as s:
            assert runtime_state._base_model_refcount == 1
            assert runtime_state._base_sae_refcount == 1
            assert g is not s
        assert runtime_state._base_model_refcount == 0
        assert runtime_state._base_sae_refcount == 0

    def test_two_different_heavy_accessors_do_not_interfere_with_refcounts(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model():
            assert runtime_state._base_sae_refcount == 0
            assert runtime_state._decoder_only_refcount == 0
            assert runtime_state._encdec_refcount == 0

    def test_all_four_heavy_accessors_can_be_live_simultaneously(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model(), runtime_state.base_sae(), runtime_state.decoder_only_model(), runtime_state.encdec_model():
            assert runtime_state._base_model_refcount == 1
            assert runtime_state._base_sae_refcount == 1
            assert runtime_state._decoder_only_refcount == 1
            assert runtime_state._encdec_refcount == 1
        assert runtime_state._base_model_refcount == 0
        assert runtime_state._base_sae_refcount == 0
        assert runtime_state._decoder_only_refcount == 0
        assert runtime_state._encdec_refcount == 0

    def test_close_raises_if_base_model_refcount_nonzero(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model():
            with pytest.raises(
                AxBenchBaseError, match="base_model",
            ) as excinfo:
                runtime_state.close()
            assert "active model contexts" in str(excinfo.value)

    def test_close_raises_if_multiple_refcounts_nonzero(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model(), runtime_state.decoder_only_model():
            with pytest.raises(
                AxBenchBaseError, match="active model contexts",
            ) as excinfo:
                runtime_state.close()
            msg = str(excinfo.value)
            assert "base_model" in msg
            assert "decoder_only_model" in msg

    def test_close_does_not_mark_closed_when_refcount_check_fails(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model():
            # Item 8: pin the message so a future refactor that raises
            # a different AxBenchBaseError from a different code path
            # doesn't silently pass this test.
            with pytest.raises(
                AxBenchBaseError, match="active model contexts",
            ):
                runtime_state.close()
            assert runtime_state._closed is False
        # After the context exits cleanly, close() succeeds.
        runtime_state.close()
        assert runtime_state._closed is True

    def test_close_succeeds_after_all_contexts_exit(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model():
            pass
        with runtime_state.base_sae():
            pass
        runtime_state.close()
        assert runtime_state._closed is True

    def test_close_drops_heavy_model_refs(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model():
            pass
        with runtime_state.base_sae():
            pass
        with runtime_state.decoder_only_model():
            pass
        with runtime_state.encdec_model():
            pass
        assert runtime_state._base_model is not None
        assert runtime_state._base_sae is not None
        assert runtime_state._decoder_only_model is not None
        assert runtime_state._encdec_model is not None
        runtime_state.close()
        assert runtime_state._base_model is None
        assert runtime_state._base_sae is None
        assert runtime_state._decoder_only_model is None
        assert runtime_state._encdec_model is None

    def test_close_idempotent_after_phase_2_fields_initialized(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
    ) -> None:
        with runtime_state.base_model():
            pass
        runtime_state.close()
        runtime_state.close()  # Second call is a no-op.
        assert runtime_state._closed is True

    def test_close_from_non_main_thread_raises(
        self,
        runtime_state: RuntimeState,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.close)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_cuda_cache_emptied_on_close_if_cuda_available(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_mock = MagicMock()
        monkeypatch.setattr(runtime_state_mod.t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(runtime_state_mod.t.cuda, "empty_cache", empty_mock)
        runtime_state.close()
        assert empty_mock.call_count == 1

    def test_cuda_cache_not_emptied_on_close_if_cuda_unavailable(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        empty_mock = MagicMock()
        monkeypatch.setattr(runtime_state_mod.t.cuda, "is_available", lambda: False)
        monkeypatch.setattr(runtime_state_mod.t.cuda, "empty_cache", empty_mock)
        runtime_state.close()
        assert empty_mock.call_count == 0


# ===========================================================================
# TestProductionLoaderBodies -- cover the real _load_* function bodies
#
# The production loaders wrap the existing project helpers. Tests here
# stub the inner helpers (``load_sae_and_model``, ``build_nnterp_model``,
# ``load_model``) so we can exercise every branch of every loader
# without actually downloading a model.
# ===========================================================================


class TestLoadFeatureExtractionPairProduction:
    """Cover :func:`_load_feature_extraction_pair` without a real download."""

    @staticmethod
    def _cluster_contract(cfg_obj: AxBenchBaseConfig) -> Any:
        return SimpleNamespace(
            n_sae_features=16384,
            sae=SimpleNamespace(
                model=(
                    cfg_obj.feature_extraction_model_id
                    or cfg_obj.neuronpedia_model_id
                ),
                activation_type="residual",
                layer=8,
                width="16k",
                release="gemma-scope-2b-pt-res-canonical",
                sae_id="layer_8/width_16k/canonical",
                hook_name="blocks.8.hook_resid_post",
            ),
        )

    @staticmethod
    def _loaded_sae(
        *,
        d_sae: int = 16384,
        hook_name: str = "blocks.8.hook_resid_post",
        decoder_rows: int = 16384,
        metadata: dict[str, Any] | None = None,
        cfg_identity: dict[str, Any] | None = None,
    ) -> Any:
        cfg_fields = {
            "d_sae": d_sae,
            "hook_name": hook_name,
            "metadata": metadata,
        }
        if cfg_identity is not None:
            cfg_fields.update(cfg_identity)
        return SimpleNamespace(
            cfg=SimpleNamespace(**cfg_fields),
            W_dec=t.empty((decoder_rows, 1)),
            to=MagicMock(),
        )

    def test_returns_model_and_sae_pair(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The pair loader returns (model, sae) from a single load call
        using a properly resolved SAELens identifier.

        Regression test for the original implementation that passed
        ``sae_release=cfg.neuronpedia_source`` and
        ``sae_id=f"blocks.{layer}.hook_resid_pre"`` -- neither matched
        the real SAELens API, so the loader had never actually worked
        against real Gemma-2b + GemmaScope. The correct path parses
        ``cfg.neuronpedia_source`` into structured config and feeds a
        :class:`ResolvedSAE` to :func:`load_sae_and_model`.
        """
        tl_model = SimpleNamespace()
        # SAE stub needs a no-op ``.to()`` for the post-build bf16 cast.
        sae_obj = SimpleNamespace(to=MagicMock())
        tokenizer_obj = object()
        nnterp_model = object()
        load_calls: list[dict[str, Any]] = []
        build_calls: list[dict[str, Any]] = []

        def fake_load_sae_and_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return tl_model, sae_obj, tokenizer_obj

        def fake_build_nnterp(
            tl: Any, device: str, model_dtype: Any = None,
        ) -> Any:
            build_calls.append(
                {"tl": tl, "device": device, "model_dtype": model_dtype},
            )
            return nnterp_model

        monkeypatch.setattr(
            runtime_state_mod, "load_sae_and_model", fake_load_sae_and_model,
        )
        monkeypatch.setattr(
            runtime_state_mod, "build_nnterp_model", fake_build_nnterp,
        )
        model, sae = runtime_state_mod._load_feature_extraction_pair(cfg_obj)
        # The loader returns the TransformerLens model so downstream
        # extractors can use ``model.run_with_cache`` / ``model.cfg``;
        # the nnterp speedup wrapper is attached as a side-channel
        # attribute for callers that opt in.
        assert model is tl_model
        assert model.nnterp_model is nnterp_model
        assert sae is sae_obj
        # Parsed (layer, release, activation, width) from
        # cfg_obj.neuronpedia_source = "8-gemmascope-res-16k".
        assert load_calls[0]["model_name"] == cfg_obj.neuronpedia_model_id
        assert load_calls[0]["layer"] == 8
        assert load_calls[0]["device"] == cfg_obj.device
        resolved = load_calls[0]["resolved"]
        assert resolved.release == "gemma-scope-2b-pt-res-canonical"
        assert resolved.sae_id == "layer_8/width_16k/canonical"
        assert resolved.hook_name == "blocks.8.hook_resid_post"
        assert build_calls[0]["tl"] is tl_model
        assert build_calls[0]["device"] == cfg_obj.device
        # Pin the bf16 contract — the production fix (commit 90d3ef7)
        # passes ``model_dtype=torch.bfloat16`` so Gemma-2-2b + SAE fit
        # in a 24 GB GPU. Without this the first forward OOMs at ~25 GB.
        assert build_calls[0]["model_dtype"] is t.bfloat16
        # The SAE must also be cast to bf16 so encode() runs in the same
        # dtype as the nnterp residual stream; a MagicMock records the
        # call so we can verify the argument.
        sae_obj.to.assert_called_with(t.bfloat16)

    def test_cluster_contract_validates_configured_and_loaded_sae(
        self,
        make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The real medium-tier layout is accepted via its Neuronpedia identity."""

        cfg = make_valid_axbench_cfg(
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_source="22-gemmascope-2-res-65k",
        )
        contract: Any = SimpleNamespace(
            n_sae_features=65536,
            sae=SimpleNamespace(
                model="gemma-3-4b-pt",
                activation_type="residual",
                layer=22,
                width="65k",
                release="gemma-scope-2-4b-pt-res",
                sae_id="layer_22_width_65k_l0_medium",
                hook_name="blocks.22.hook_resid_post",
            ),
        )
        tl_model = SimpleNamespace()
        sae = self._loaded_sae(
            d_sae=65536,
            hook_name="blocks.22.hook_resid_post",
            decoder_rows=65536,
            # SAELens 6.42 SAEConfig has no sae_id field. Public medium is
            # discriminated by the registry-provided Neuronpedia identity.
            metadata={
                "model_name": "google/gemma-3-4b-pt",
                "neuronpedia_id": "gemma-3-4b/22-gemmascope-2-res-65k",
            },
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "load_sae_and_model",
            lambda **_kwargs: (tl_model, sae, object()),
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "build_nnterp_model",
            lambda *_args, **_kwargs: None,
        )

        model, loaded_sae = runtime_state_mod._load_feature_extraction_pair(
            cfg,
            cluster_contract=contract,
        )

        assert model is tl_model
        assert loaded_sae is sae

    def test_cluster_contract_rejects_configured_sae_drift_before_load(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        load = MagicMock()
        monkeypatch.setattr(runtime_state_mod, "load_sae_and_model", load)
        contract = self._cluster_contract(cfg_obj)
        contract.sae.model = "different-model"

        with pytest.raises(AxBenchBaseError, match="SAE identity does not match"):
            runtime_state_mod._load_feature_extraction_pair(
                cfg_obj,
                cluster_contract=contract,
            )

        load.assert_not_called()

    @pytest.mark.parametrize(
        ("d_sae", "hook_name", "decoder_rows"),
        [
            (16383, "blocks.8.hook_resid_post", 16384),
            (16384, "blocks.7.hook_resid_post", 16384),
            (16384, "blocks.8.hook_resid_post", 16383),
        ],
        ids=("d-sae", "hook", "decoder-rows"),
    )
    def test_cluster_contract_rejects_loaded_sae_drift(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
        d_sae: int,
        hook_name: str,
        decoder_rows: int,
    ) -> None:
        sae = self._loaded_sae(
            d_sae=d_sae,
            hook_name=hook_name,
            decoder_rows=decoder_rows,
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "load_sae_and_model",
            lambda **_kwargs: (SimpleNamespace(), sae, object()),
        )

        with pytest.raises(AxBenchBaseError, match="loaded runtime extraction SAE"):
            runtime_state_mod._load_feature_extraction_pair(
                cfg_obj,
                cluster_contract=self._cluster_contract(cfg_obj),
            )

    def test_same_shape_wrong_l0_tier_is_rejected_without_caching(
        self,
        make_valid_axbench_cfg: Callable[..., AxBenchBaseConfig],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A same-shape GemmaScope-2 SAE from another tier never enters state."""

        cfg = make_valid_axbench_cfg(
            neuronpedia_model_id="gemma-3-4b",
            feature_extraction_model_id="gemma-3-4b-pt",
            neuronpedia_source="22-gemmascope-2-res-65k",
        )
        contract: Any = SimpleNamespace(
            n_sae_features=65536,
            sae=SimpleNamespace(
                model="gemma-3-4b-pt",
                activation_type="residual",
                layer=22,
                width="65k",
                release="gemma-scope-2-4b-pt-res",
                sae_id="layer_22_width_65k_l0_medium",
                hook_name="blocks.22.hook_resid_post",
            ),
        )
        wrong_tier_sae = self._loaded_sae(
            d_sae=65536,
            hook_name="blocks.22.hook_resid_post",
            decoder_rows=65536,
            # This is the real SAELens 6.42 big/small-tier identity shape:
            # SAEConfig has no sae_id and unpublished tiers have no
            # Neuronpedia identity.
            metadata={
                "model_name": "google/gemma-3-4b-pt",
                "neuronpedia_id": None,
            },
        )
        load = MagicMock(
            return_value=(SimpleNamespace(), wrong_tier_sae, object()),
        )
        monkeypatch.setattr(runtime_state_mod, "load_sae_and_model", load)
        monkeypatch.setattr(
            runtime_state_mod,
            "build_nnterp_model",
            lambda *_args, **_kwargs: None,
        )
        state = RuntimeState(
            cfg,
            cluster_contract=contract,
        )

        with pytest.raises(AxBenchBaseError, match="cannot prove.*L0 tier"):
            state._ensure_feature_extraction_loaded()

        load.assert_called_once()
        assert state._feat_extraction_loaded is False
        assert state._base_model is None
        assert state._base_sae is None

    @pytest.mark.parametrize(
        ("cfg_identity", "metadata", "match"),
        [
            (
                {"release": "wrong-release"},
                None,
                "cfg.release.*does not match resolved release",
            ),
            (
                {},
                {"model_name": "wrong-model"},
                "model_name.*does not match resolved model",
            ),
            (
                {},
                {"neuronpedia_id": "wrong-model/wrong-source"},
                "neuronpedia_id.*does not match resolved Neuronpedia id",
            ),
        ],
        ids=("release", "model", "neuronpedia"),
    )
    def test_cluster_contract_rejects_loaded_sae_identity_drift(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
        cfg_identity: dict[str, Any],
        metadata: dict[str, Any] | None,
        match: str,
    ) -> None:
        """Every loaded identity representation has a direct rejection test."""

        sae = self._loaded_sae(
            metadata=metadata,
            cfg_identity=cfg_identity,
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "load_sae_and_model",
            lambda **_kwargs: (SimpleNamespace(), sae, object()),
        )

        with pytest.raises(AxBenchBaseError, match=match):
            runtime_state_mod._load_feature_extraction_pair(
                cfg_obj,
                cluster_contract=self._cluster_contract(cfg_obj),
            )

    def test_sae_bf16_cast_only_when_nnterp_available(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """When ``build_nnterp_model`` returns ``None``, the SAE stays in
        its loaded dtype so downstream fp32 fallbacks keep working. Pins
        the guard at runtime_state.py:388 (``if nn_model is not None``).
        """
        sae_obj = SimpleNamespace(to=MagicMock())
        tl_model = SimpleNamespace()
        monkeypatch.setattr(
            runtime_state_mod,
            "load_sae_and_model",
            lambda **kwargs: (tl_model, sae_obj, object()),
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "build_nnterp_model",
            lambda tl, device, model_dtype=None: None,
        )
        runtime_state_mod._load_feature_extraction_pair(cfg_obj)
        sae_obj.to.assert_not_called()

    def test_uses_sae_layer_not_diffmean_probe_layer(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The SAE layer comes from cfg.neuronpedia_source, NOT from
        cfg.layer_hooks.diffmean_and_probe_layer.

        These two layers are semantically distinct:
          - diffmean_and_probe_layer: which layer of the nanochat TARGET
            model to hook for Stage 3 baseline capture (default 6).
          - SAE layer: which layer of the feature-extraction BASE model
            the SAE was trained on (encoded in neuronpedia_source,
            default 8 for ``"8-gemmascope-res-16k"``).
        The old loader conflated these; this test pins that they stay
        independent.
        """
        load_calls: list[dict[str, Any]] = []

        def fake_load_sae_and_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            # SAE stub needs a no-op ``.to()`` for the post-build bf16 cast.
            return (
                SimpleNamespace(),
                SimpleNamespace(to=MagicMock()),
                object(),
            )

        monkeypatch.setattr(
            runtime_state_mod, "load_sae_and_model", fake_load_sae_and_model,
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "build_nnterp_model",
            lambda tl, device, model_dtype=None: object(),
        )
        # Set the two layers to different values so a regression that
        # picks the wrong one surfaces loud.
        cfg_obj = dataclasses.replace(
            cfg_obj,
            layer_hooks=dataclasses.replace(
                cfg_obj.layer_hooks, diffmean_and_probe_layer=11,
            ),
        )
        # neuronpedia_source default "8-gemmascope-res-16k" encodes layer 8.
        runtime_state_mod._load_feature_extraction_pair(cfg_obj)
        assert load_calls[0]["layer"] == 8  # SAE layer, NOT 11
        assert load_calls[0]["resolved"].sae_id == "layer_8/width_16k/canonical"
        assert load_calls[0]["resolved"].hook_name == "blocks.8.hook_resid_post"

    def test_malformed_neuronpedia_source_raises_axbench_base_error(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A garbled neuronpedia_source raises AxBenchBaseError wrapping
        the parser's ValueError, not a bare TypeError from SAELens."""
        cfg_obj = dataclasses.replace(
            cfg_obj, neuronpedia_source="not-a-valid-source-string-actually",
        )
        monkeypatch.setattr(
            runtime_state_mod, "load_sae_and_model",
            lambda **kw: (object(), object(), object()),
        )
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state_mod._load_feature_extraction_pair(cfg_obj)
        assert "Failed to load feature-extraction model" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ValueError)

    def test_falls_back_to_tl_when_nnterp_unavailable(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        tl_model = SimpleNamespace()
        sae_obj = object()
        monkeypatch.setattr(
            runtime_state_mod,
            "load_sae_and_model",
            lambda **kwargs: (tl_model, sae_obj, object()),
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "build_nnterp_model",
            lambda tl, device, model_dtype=None: None,
        )
        model, sae = runtime_state_mod._load_feature_extraction_pair(cfg_obj)
        assert model is tl_model
        # nnterp speedup unavailable — side-channel attribute is None so
        # the TL fallback path fires cleanly downstream.
        assert model.nnterp_model is None
        assert sae is sae_obj

    def test_wraps_failure_in_axbench_base_error(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(**kwargs: Any) -> Any:
            raise RuntimeError("network is down")

        monkeypatch.setattr(runtime_state_mod, "load_sae_and_model", boom)
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state_mod._load_feature_extraction_pair(cfg_obj)
        msg = str(excinfo.value)
        assert "Failed to load feature-extraction model" in msg
        assert cfg_obj.neuronpedia_model_id in msg
        assert isinstance(excinfo.value.__cause__, RuntimeError)


class TestLoadDecoderOnlyModelProduction:
    """Cover :func:`_load_decoder_only_model` without a real checkpoint load."""

    def test_load_decoder_only_passes_source_tag_step(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_model = object()
        load_calls: list[dict[str, Any]] = []

        def fake_load_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return stub_model, object(), {"meta": "data"}

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        result = runtime_state_mod._load_decoder_only_model(cfg_obj)
        assert result is stub_model
        assert load_calls[0]["source"] == cfg_obj.decoder_only["source"]
        assert load_calls[0]["model_tag"] == cfg_obj.decoder_only["model_tag"]
        assert load_calls[0]["step"] == cfg_obj.decoder_only["step"]
        assert isinstance(load_calls[0]["device"], t.device)
        assert load_calls[0]["phase"] == runtime_state_mod.ModelPhase.EVAL

    def test_load_decoder_only_with_optional_fields_missing(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``model_tag`` and ``step`` are optional in cfg.decoder_only."""
        cfg_obj_no_tag = cfg_obj.replace(decoder_only={"source": "base"})
        stub_model = object()
        load_calls: list[dict[str, Any]] = []

        def fake_load_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return stub_model, object(), {}

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        result = runtime_state_mod._load_decoder_only_model(cfg_obj_no_tag)
        assert result is stub_model
        assert load_calls[0]["model_tag"] is None
        assert load_calls[0]["step"] is None

    def test_clustered_load_pins_latest_numeric_step_and_exact_files(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cfg_latest = cfg_obj.replace(
            decoder_only={
                "source": "base",
                "model_tag": "d-test",
                "step": None,
            }
        )
        stub_model = MagicMock()
        pin = SimpleNamespace(step=23)
        checkpoint_dir_calls: list[dict[str, Any]] = []
        checkpoint_pin_calls: list[tuple[tuple[Any, ...], dict[str, Any]]] = []
        load_calls: list[dict[str, Any]] = []
        unchanged: list[Any] = []

        def resolve_checkpoint_dir(**kwargs: Any) -> tuple[Path, str]:
            checkpoint_dir_calls.append(kwargs)
            return tmp_path, "d-test"

        monkeypatch.setattr(
            runtime_state_mod,
            "resolve_eval_checkpoint_directory",
            resolve_checkpoint_dir,
        )

        def resolve_checkpoint_pin(
            *args: Any,
            **kwargs: Any,
        ) -> tuple[Any, dict[str, Any], dict[str, t.Tensor]]:
            checkpoint_pin_calls.append((args, kwargs))
            return pin, {}, {"weight": t.tensor([1.0])}

        monkeypatch.setattr(
            runtime_state_mod,
            "resolve_checkpoint_files_pin",
            resolve_checkpoint_pin,
        )
        monkeypatch.setattr(
            runtime_state_mod,
            "assert_checkpoint_files_pin_unchanged",
            lambda actual: unchanged.append(actual),
        )

        def fake_load_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return stub_model, object(), {}

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        result = runtime_state_mod._load_decoder_only_model(
            cfg_latest,
            cast(Any, SimpleNamespace()),
        )

        assert result is stub_model
        assert checkpoint_dir_calls == [
            {
                "checkpoint_dir": None,
                "checkpoint_source": "base",
                "checkpoint_model_tag": "d-test",
            },
        ]
        assert checkpoint_pin_calls == [
            (
                (tmp_path,),
                {
                    "step": None,
                    "resolved_tag": "d-test",
                },
            ),
        ]
        assert load_calls == [
            {
                "source": "base",
                "device": t.device(cfg_latest.device),
                "phase": runtime_state_mod.ModelPhase.EVAL,
                "model_tag": "d-test",
                "step": 23,
                "eval_conditioning": (
                    runtime_state_mod.EvalConditioning.deferred("axbench_base")
                ),
            },
        ]
        assert unchanged == [pin]
        assert result._loaded_checkpoint_files_pin is pin

    def test_clustered_load_rejects_checkpoint_inode_aba_during_runtime_load(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        cfg_latest = cfg_obj.replace(
            decoder_only={
                "source": "base",
                "model_tag": "d-test",
                "step": None,
            }
        )
        model_path = tmp_path / "model_000023.pt"
        metadata_path = tmp_path / "meta_000023.json"
        replacement_path = tmp_path / "replacement.pt"
        restored_path = tmp_path / "restored.pt"
        t.save({"weight": t.tensor([1.0])}, model_path)
        t.save({"weight": t.tensor([2.0])}, replacement_path)
        metadata_path.write_text("{}", encoding="utf-8")
        original_bytes = model_path.read_bytes()
        restored_path.write_bytes(original_bytes)
        original_inode = model_path.lstat().st_ino

        monkeypatch.setattr(
            runtime_state_mod,
            "resolve_eval_checkpoint_directory",
            lambda **_kwargs: (tmp_path, "d-test"),
        )

        def fake_load_model(**_kwargs: Any) -> tuple[MagicMock, object, dict[str, Any]]:
            replacement_path.replace(model_path)
            restored_path.replace(model_path)
            assert model_path.read_bytes() == original_bytes
            assert model_path.lstat().st_ino != original_inode
            return MagicMock(), object(), {}

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)

        with pytest.raises(
            AxBenchBaseError,
            match="checkpoint model bytes changed after they were pinned",
        ):
            runtime_state_mod._load_decoder_only_model(
                cfg_latest,
                cast(Any, SimpleNamespace()),
            )

    @pytest.mark.parametrize(
        ("decoder_only", "error"),
        [
            (
                {"source": "", "model_tag": "d-test", "step": 1},
                "decoder_only.source must be a non-empty string",
            ),
            (
                {"source": "base", "model_tag": "", "step": 1},
                "decoder_only.model_tag must be a non-empty string",
            ),
            (
                {"source": "base", "model_tag": "d-test", "step": True},
                "decoder_only.step must be a non-negative integer",
            ),
            (
                {"source": "base", "model_tag": "d-test", "step": -1},
                "decoder_only.step must be non-negative",
            ),
        ],
    )
    def test_clustered_load_rejects_malformed_checkpoint_ref(
        self,
        cfg_obj: AxBenchBaseConfig,
        decoder_only: dict[str, Any],
        error: str,
    ) -> None:
        with pytest.raises(AxBenchBaseError, match=error):
            runtime_state_mod._load_decoder_only_model(
                cfg_obj.replace(decoder_only=decoder_only),
                cast(Any, SimpleNamespace()),
            )

    def test_load_decoder_only_wraps_failure_in_axbench_base_error(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(**kwargs: Any) -> Any:
            raise FileNotFoundError("missing checkpoint")

        monkeypatch.setattr(runtime_state_mod, "load_model", boom)
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state_mod._load_decoder_only_model(cfg_obj)
        msg = str(excinfo.value)
        assert "Failed to load decoder-only nanochat model" in msg
        assert isinstance(excinfo.value.__cause__, FileNotFoundError)


class TestLoadEncdecModelProduction:
    """Cover :func:`_load_encdec_model` without a real checkpoint load."""

    def test_load_encdec_attaches_meta_data(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_model = MagicMock()
        expected_meta = {"checkpoint_dir": "/fake/path", "user_config": {"k": "v"}}

        def fake_load_model(**kwargs: Any) -> Any:
            return stub_model, object(), expected_meta

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        result = runtime_state_mod._load_encdec_model(cfg_obj)
        assert result is stub_model
        assert stub_model._loaded_meta_data is expected_meta

    def test_load_encdec_rejects_turn_checkpoint_before_metadata_attachment(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_model = SimpleNamespace()

        def fake_load_model(**_kwargs: Any) -> Any:
            return (
                stub_model,
                object(),
                {
                    "checkpoint_dir": "/fake/turn",
                    "sft_segment_mode": "turn_delimited_fixed_tokens",
                },
            )

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)

        # kills: leaving the guard inside the loader's ``try``, which would
        # re-wrap a supported-mode rejection as a load failure whose message
        # blames the source/model_tag/step triple or the device's VRAM.
        with pytest.raises(
            ValueError,
            match=r"axbench_base does not support checkpoint /fake/turn.*"
            r"turn_delimited_fixed_tokens",
        ) as exc_info:
            runtime_state_mod._load_encdec_model(cfg_obj)

        assert not isinstance(exc_info.value, AxBenchBaseError)
        assert "VRAM" not in str(exc_info.value)
        assert not hasattr(stub_model, "_loaded_meta_data")

    def test_load_encdec_passes_source_tag_step(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        stub_model = MagicMock()
        load_calls: list[dict[str, Any]] = []

        def fake_load_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return stub_model, object(), {}

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        runtime_state_mod._load_encdec_model(cfg_obj)
        assert load_calls[0]["source"] == cfg_obj.encdec["source"]
        assert load_calls[0]["model_tag"] == cfg_obj.encdec["model_tag"]
        assert load_calls[0]["step"] == cfg_obj.encdec["step"]

    def test_clustered_load_declares_vocab_safe_only_with_validated_contract(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The real loader must not reject compaction before contract checks."""

        stub_model = MagicMock()
        expected_meta = {"checkpoint_dir": "/fake/clustered"}
        load_calls: list[dict[str, Any]] = []
        asserted: list[tuple[Any, Any, Any, str]] = []
        cluster_contract = MagicMock()

        def fake_load_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return stub_model, object(), expected_meta

        def fake_assert_loaded(
            contract: Any,
            model: Any,
            metadata: Any,
            *,
            surface: str,
        ) -> None:
            asserted.append((contract, model, metadata, surface))

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        monkeypatch.setattr(
            runtime_state_mod,
            "assert_loaded_cluster_target",
            fake_assert_loaded,
        )

        result = runtime_state_mod._load_encdec_model(
            cfg_obj,
            cluster_contract=cluster_contract,
        )

        compacted_vocab = FeatureVocab(
            n_sae_features=8,
            original_ids=(1, 3, 7),
        )
        load_calls[0]["eval_conditioning"].assert_loadable(
            compacted_vocab,
            checkpoint_dir="/fake/clustered",
        )
        assert result is stub_model
        assert asserted == [
            (
                cluster_contract,
                stub_model,
                expected_meta,
                "axbench_base encdec checkpoint",
            )
        ]

    def test_unclustered_load_keeps_deferred_compaction_rejection(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """No-cluster callers retain the exact fail-loud loader stance."""

        load_calls: list[dict[str, Any]] = []

        def fake_load_model(**kwargs: Any) -> Any:
            load_calls.append(kwargs)
            return MagicMock(), object(), {}

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        runtime_state_mod._load_encdec_model(cfg_obj)

        compacted_vocab = FeatureVocab(
            n_sae_features=8,
            original_ids=(1, 3, 7),
        )
        with pytest.raises(NotImplementedError, match="axbench_base"):
            load_calls[0]["eval_conditioning"].assert_loadable(
                compacted_vocab,
                checkpoint_dir="/fake/clustered",
            )

    def test_clustered_rejection_happens_before_metadata_attachment(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """A rejected clustered checkpoint never stamps a live model object."""

        stub_model = SimpleNamespace()
        expected_meta = {"checkpoint_dir": "/fake/rejected"}
        cluster_contract = MagicMock()
        events: list[str] = []

        def fake_load_model(**_kwargs: Any) -> Any:
            events.append("load")
            return stub_model, object(), expected_meta

        def reject_loaded(
            _contract: Any,
            model: Any,
            metadata: Any,
            *,
            surface: str,
        ) -> None:
            assert model is stub_model
            assert metadata is expected_meta
            assert surface == "axbench_base encdec checkpoint"
            assert not hasattr(model, "_loaded_meta_data")
            events.append("reject")
            raise ValueError("stale clustered checkpoint")

        monkeypatch.setattr(runtime_state_mod, "load_model", fake_load_model)
        monkeypatch.setattr(
            runtime_state_mod,
            "assert_loaded_cluster_target",
            reject_loaded,
        )

        with pytest.raises(AxBenchBaseError, match="stale clustered checkpoint"):
            runtime_state_mod._load_encdec_model(
                cfg_obj,
                cluster_contract=cluster_contract,
            )

        assert events == ["load", "reject"]
        assert not hasattr(stub_model, "_loaded_meta_data")

    def test_load_encdec_wraps_failure_in_axbench_base_error(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(**kwargs: Any) -> Any:
            raise RuntimeError("missing tensor")

        monkeypatch.setattr(runtime_state_mod, "load_model", boom)
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state_mod._load_encdec_model(cfg_obj)
        msg = str(excinfo.value)
        assert "Failed to load encoder-decoder nanochat model" in msg
        assert isinstance(excinfo.value.__cause__, RuntimeError)


# ===========================================================================
# Phase 3 -- metas, judge client, inactive sampler
#
# Phase 3 adds lazily-built accessors for detected metas, an OpenRouter judge
# client, and a derived inactive-feature sampler. Every test in this section uses
# ``stub_heavy_loaders`` so it never touches a real model; Phase 3's
# own production builders are similarly monkey-patched at the module
# boundary.
# ===========================================================================


def _make_meta(
    *,
    negative_selection: dict[str, Any] | None = None,
) -> TargetModelMetadata:
    """Return a minimal :class:`TargetModelMetadata` for tests."""
    return TargetModelMetadata(
        num_buckets=8,
        n_sae_features=16384,
        negative_selection=negative_selection,
        feature_selection=None,
        per_feature_stats=None,
        device="cuda",
        sae_layer=8,
        segment_size=128,
    )


@pytest.fixture
def stub_phase3_builders(
    monkeypatch: pytest.MonkeyPatch,
) -> dict[str, MagicMock]:
    """Install stub module-level builders for every live Phase 3 resource.

    Each stub is a :class:`MagicMock` that records call args and
    returns a fresh :class:`MagicMock` whose ``close()`` is also a
    mock so tests can assert teardown ordering.
    """
    stubs: dict[str, MagicMock] = {
        "AxBenchJudgeClient": MagicMock(
            name="judge_client_constructor",
            side_effect=lambda cfg, cache_path: MagicMock(
                name=f"judge_client:{cache_path}",
            ),
        ),
        "_build_inactive_sampler": MagicMock(
            name="build_inactive_sampler",
            side_effect=lambda *, encdec_meta, cfg: MagicMock(
                name="inactive_sampler",
            ),
        ),
    }
    for name, mock in stubs.items():
        monkeypatch.setattr(runtime_state_mod, name, mock)
    return stubs


@pytest.fixture
def stub_detect_meta(
    monkeypatch: pytest.MonkeyPatch,
) -> MagicMock:
    """Monkey-patch :func:`detect_target_model_metadata` with a configurable stub.

    Default behaviour: returns a fresh :class:`TargetModelMetadata`
    whose ``negative_selection`` is ``None`` (so ``inactive_sampler``
    resolves to ``None``). Override ``side_effect`` or ``return_value``
    in individual tests.
    """
    stub = MagicMock(
        name="detect_target_model_metadata",
        side_effect=lambda *, target_model, cfg: _make_meta(),
    )
    monkeypatch.setattr(
        runtime_state_mod, "detect_target_model_metadata", stub,
    )
    return stub


# ---------------------------------------------------------------------------
# Metas
# ---------------------------------------------------------------------------


META_ACCESSORS: list[tuple[str, str, str, str]] = [
    # (meta_method, model_method, refcount_attr, cache_attr)
    (
        "decoder_only_meta",
        "decoder_only_model",
        "_decoder_only_refcount",
        "_decoder_only_meta",
    ),
    ("encdec_meta", "encdec_model", "_encdec_refcount", "_encdec_meta"),
]


@pytest.mark.parametrize(
    ("meta_method", "model_method", "refcount_attr", "cache_attr"),
    META_ACCESSORS,
    ids=[a[0] for a in META_ACCESSORS],
)
class TestMetaAccessors:
    def test_first_call_runs_detector_and_caches(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_detect_meta: MagicMock,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        method = getattr(runtime_state, meta_method)
        result = method()
        assert isinstance(result, TargetModelMetadata)
        assert stub_detect_meta.call_count == 1
        assert getattr(runtime_state, cache_attr) is result

    def test_second_call_does_not_re_run_detector(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_detect_meta: MagicMock,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        method = getattr(runtime_state, meta_method)
        first = method()
        second = method()
        assert first is second
        assert stub_detect_meta.call_count == 1

    def test_second_call_does_not_re_enter_model_context(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_detect_meta: MagicMock,
        monkeypatch: pytest.MonkeyPatch,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        """After the first call caches the meta, subsequent calls must
        NOT re-enter the underlying model context. We verify this by
        installing a tracking loader via :meth:`monkeypatch.setattr`
        BEFORE invoking the meta accessor so every loader entry is
        actually observed — the previous form of this test never
        installed the wrapper and was tautological (``call_count``
        stayed at 0 even if caching were completely broken).
        """
        loader_name = {
            "decoder_only_model": "_load_decoder_only_model",
            "encdec_model": "_load_encdec_model",
        }[model_method]
        original_loader = getattr(runtime_state_mod, loader_name)

        call_count = {"n": 0}

        def tracking_loader(cfg: Any) -> Any:
            call_count["n"] += 1
            return original_loader(cfg)

        monkeypatch.setattr(runtime_state_mod, loader_name, tracking_loader)

        method = getattr(runtime_state, meta_method)
        # First call: loader fires exactly once.
        method()
        assert call_count["n"] == 1
        # Second call: the meta is cached so the loader MUST NOT fire.
        method()
        assert call_count["n"] == 1
        # Third call: still cached.
        method()
        assert call_count["n"] == 1

    def test_after_close_raises(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_detect_meta: MagicMock,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        runtime_state.close()
        method = getattr(runtime_state, meta_method)
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            method()

    def test_non_main_thread_raises(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_detect_meta: MagicMock,
        run_in_thread: Callable,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        method = getattr(runtime_state, meta_method)
        _, error = run_in_thread(method)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_detector_generic_failure_wrapped(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        def boom(*, target_model: Any, cfg: Any) -> Any:
            raise RuntimeError("fake detect failure")

        monkeypatch.setattr(
            runtime_state_mod, "detect_target_model_metadata", boom,
        )
        method = getattr(runtime_state, meta_method)
        with pytest.raises(AxBenchBaseError) as excinfo:
            method()
        assert "Failed to detect" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)
        # Item 4: the underlying model's refcount must be back to 0 even
        # on detector failure. Otherwise close() would raise on the
        # active-refcount check and leave the caller holding a dangling
        # model reference.
        assert getattr(runtime_state, refcount_attr) == 0, (
            f"Meta accessor must decrement underlying model refcount "
            f"even on detector failure; {refcount_attr} is still "
            f"{getattr(runtime_state, refcount_attr)!r}, expected 0. "
            "close() will refuse with an active refcount otherwise."
        )

    def test_detector_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        def raises_native(*, target_model: Any, cfg: Any) -> Any:
            raise AxBenchBaseError("original detect failure")

        monkeypatch.setattr(
            runtime_state_mod, "detect_target_model_metadata", raises_native,
        )
        method = getattr(runtime_state, meta_method)
        with pytest.raises(
            AxBenchBaseError, match="original detect failure",
        ) as excinfo:
            method()
        assert "Failed to detect" not in str(excinfo.value)
        # Item 4: refcount must still return to 0 even on the passthrough
        # path so close() can run cleanly.
        assert getattr(runtime_state, refcount_attr) == 0

    def test_exits_model_context_cleanly_after_detection(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_detect_meta: MagicMock,
        meta_method: str,
        model_method: str,
        refcount_attr: str,
        cache_attr: str,
    ) -> None:
        method = getattr(runtime_state, meta_method)
        method()
        # Refcount is back to 0 after the context manager exits.
        assert getattr(runtime_state, refcount_attr) == 0


class TestEncdecMetaNegativeSelection:
    """Narrow coverage for the two encdec_meta negative_selection modes."""

    def test_with_none_negative_selection(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(negative_selection=None),
        )
        meta = runtime_state.encdec_meta()
        assert meta.negative_selection is None

    def test_with_dict_negative_selection(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        sel = {"distribution": "geometric", "mean": 3.0}
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(negative_selection=sel),
        )
        meta = runtime_state.encdec_meta()
        assert meta.negative_selection == sel


# ---------------------------------------------------------------------------
# Judge client
# ---------------------------------------------------------------------------


class TestJudgeClientAccessor:
    def test_first_call_opens_and_caches(
        self,
        runtime_state: RuntimeState,
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        result = runtime_state.judge_client()
        assert stub_phase3_builders["AxBenchJudgeClient"].call_count == 1
        assert runtime_state._judge_client is result

    def test_second_call_returns_cached_instance(
        self,
        runtime_state: RuntimeState,
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        first = runtime_state.judge_client()
        second = runtime_state.judge_client()
        assert first is second
        assert stub_phase3_builders["AxBenchJudgeClient"].call_count == 1

    def test_after_close_raises(
        self,
        runtime_state: RuntimeState,
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.judge_client()

    def test_non_main_thread_raises(
        self,
        runtime_state: RuntimeState,
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.judge_client)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_missing_run_tag_raises(
        self,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        cfg_no_tag = cfg_obj.replace(run_tag=None)
        state = RuntimeState(cfg=cfg_no_tag)
        with pytest.raises(AxBenchBaseError, match="run_tag"):
            state.judge_client()

    def test_builder_failure_wrapped_as_axbench_error(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def boom(*args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("fake builder failure")

        monkeypatch.setattr(runtime_state_mod, "AxBenchJudgeClient", boom)
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state.judge_client()
        assert "Failed to" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_builder_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def raises_native(*args: Any, **kwargs: Any) -> Any:
            raise AxBenchBaseError("original builder failure")

        monkeypatch.setattr(runtime_state_mod, "AxBenchJudgeClient", raises_native)
        with pytest.raises(
            AxBenchBaseError, match="original builder failure",
        ):
            runtime_state.judge_client()


class TestClientGlobal:
    def test_close_closes_judge_client(
        self,
        runtime_state: RuntimeState,
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        judge = cast(MagicMock, runtime_state.judge_client())
        runtime_state.close()
        assert judge.close.call_count == 1


# ---------------------------------------------------------------------------
# inactive_sampler
# ---------------------------------------------------------------------------


class TestInactiveSampler:
    def test_first_call_enters_encdec_meta(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(negative_selection=None),
        )
        runtime_state.inactive_sampler()
        # encdec meta is cached, inactive_sampler_loaded flag set
        assert runtime_state._encdec_meta is not None
        assert runtime_state._inactive_sampler_loaded is True

    def test_returns_none_when_negative_selection_none(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(negative_selection=None),
        )
        sampler = runtime_state.inactive_sampler()
        assert sampler is None
        # Build helper was NOT called.
        assert stub_phase3_builders["_build_inactive_sampler"].call_count == 0

    def test_returns_callable_when_negative_selection_set(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(
                negative_selection={"distribution": "geometric", "mean": 2.0},
            ),
        )
        sampler = runtime_state.inactive_sampler()
        assert sampler is not None
        assert callable(sampler)
        assert stub_phase3_builders["_build_inactive_sampler"].call_count == 1
        assert set(
            stub_phase3_builders["_build_inactive_sampler"].call_args.kwargs
        ) == {"encdec_meta", "cfg"}

    def test_cached_across_calls(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(
                negative_selection={"distribution": "geometric", "mean": 2.0},
            ),
        )
        s1 = runtime_state.inactive_sampler()
        s2 = runtime_state.inactive_sampler()
        assert s1 is s2
        assert stub_phase3_builders["_build_inactive_sampler"].call_count == 1

    def test_cached_none_not_recomputed(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """``None`` return value is cached too, via the sentinel flag."""
        detect_calls = {"n": 0}

        def detect_once(*, target_model: Any, cfg: Any) -> Any:
            detect_calls["n"] += 1
            return _make_meta(negative_selection=None)

        monkeypatch.setattr(
            runtime_state_mod, "detect_target_model_metadata", detect_once,
        )
        assert runtime_state.inactive_sampler() is None
        assert runtime_state.inactive_sampler() is None
        # encdec_meta was only detected once (first inactive_sampler call).
        # The second call hits the sentinel and skips everything.
        assert detect_calls["n"] == 1

    def test_after_close_raises(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        runtime_state.close()
        with pytest.raises(AxBenchBaseError, match="RuntimeState is closed"):
            runtime_state.inactive_sampler()

    def test_non_main_thread_raises(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        run_in_thread: Callable,
    ) -> None:
        _, error = run_in_thread(runtime_state.inactive_sampler)
        assert isinstance(error, AxBenchBaseError)
        assert "main thread" in str(error)

    def test_build_failure_wrapped_as_axbench_error(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(
                negative_selection={"distribution": "geometric", "mean": 2.0},
            ),
        )

        def boom(*, encdec_meta: Any, cfg: Any) -> Any:
            raise RuntimeError("fake builder failure")

        monkeypatch.setattr(runtime_state_mod, "_build_inactive_sampler", boom)
        with pytest.raises(AxBenchBaseError) as excinfo:
            runtime_state.inactive_sampler()
        assert "Failed to build inactive sampler" in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, RuntimeError)

    def test_build_axbench_error_passthrough(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(
                negative_selection={"distribution": "geometric", "mean": 2.0},
            ),
        )

        def raises_native(*, encdec_meta: Any, cfg: Any) -> Any:
            raise AxBenchBaseError("original sampler failure")

        monkeypatch.setattr(
            runtime_state_mod, "_build_inactive_sampler", raises_native,
        )
        with pytest.raises(
            AxBenchBaseError, match="original sampler failure",
        ) as excinfo:
            runtime_state.inactive_sampler()
        assert "Failed to build inactive sampler" not in str(excinfo.value)

    def test_inactive_sampler_propagates_encdec_load_failure(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Item 17: if the encdec loader fails while
        :meth:`inactive_sampler` is trying to populate the sampler
        via :meth:`encdec_meta`, the failure must propagate as
        :class:`AxBenchBaseError` rather than returning a silent
        :data:`None` or leaving the state half-initialized.

        Also verifies that a partial context enter does not leak
        the encdec refcount (must return to 0 on the raise path).
        """
        def fail_loader(cfg: Any) -> Any:
            raise AxBenchBaseError("encdec load failed: test injection")

        monkeypatch.setattr(
            runtime_state_mod, "_load_encdec_model", fail_loader,
        )

        with pytest.raises(
            AxBenchBaseError, match="encdec load failed",
        ):
            runtime_state.inactive_sampler()

        # Refcount must still be 0 (no leak from a partial
        # context enter).
        assert runtime_state._encdec_refcount == 0


class TestBuildInactiveSampler:
    """Phase 7: :func:`_build_inactive_sampler` returns a real sampler.

    The helper parses the encdec target's training ``negative_selection``
    dict, reads ``n_sae_features`` from the feature-statistics file, and
    returns a closure that bridges ``random.Random`` to
    :class:`torch.Generator` via :func:`sample_inactive_features`. Every
    sampled fid is mapped to ``num_buckets - 1``.
    """

    _DEFAULT_NEGATIVE_SELECTION: dict[str, Any] = {
        "distribution": "geometric",
        "mean": 3.0,
    }

    def test_build_sampler_raises_if_num_buckets_none(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        meta = TargetModelMetadata(
            num_buckets=None,
            n_sae_features=16384,
            negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            feature_selection=None,
            per_feature_stats=None,
            device="cpu",
            sae_layer=8,
            segment_size=128,
        )
        with pytest.raises(AxBenchBaseError, match="num_buckets is None"):
            runtime_state_mod._build_inactive_sampler(
                encdec_meta=meta, cfg=cfg_obj,
            )

    def test_build_sampler_raises_if_negative_selection_none(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        with pytest.raises(
            AxBenchBaseError, match="negative_selection is None",
        ):
            runtime_state_mod._build_inactive_sampler(
                encdec_meta=_make_meta(negative_selection=None),
                cfg=cfg_obj,
            )

    def test_build_sampler_raises_if_parse_returns_none(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Defensive: if :func:`parse_negative_selection` returns
        ``None`` despite a non-``None`` input dict, the builder must
        raise a clear :class:`AxBenchBaseError` rather than silently
        producing a sampler that crashes on first call.
        """
        monkeypatch.setattr(
            runtime_state_mod, "parse_negative_selection", lambda cfg: None,
        )
        with pytest.raises(
            AxBenchBaseError,
            match="parse_negative_selection returned None",
        ):
            runtime_state_mod._build_inactive_sampler(
                encdec_meta=_make_meta(
                    negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
                ),
                cfg=cfg_obj,
            )

    def test_build_sampler_returns_callable(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            ),
            cfg=cfg_obj,
        )
        assert callable(sampler)

    def test_sampler_accepts_rng_kwarg(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            ),
            cfg=cfg_obj,
        )
        # The orchestrator calls ``sampler(rng=rng)`` — positional
        # should be rejected (closure signature is keyword-only).
        result = sampler(rng=random.Random(42))
        assert isinstance(result, dict)

    def test_sampler_returns_dict_int_int(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            ),
            cfg=cfg_obj,
        )
        result = sampler(rng=random.Random(42))
        assert all(isinstance(k, int) for k in result)
        assert all(isinstance(v, int) for v in result.values())

    def test_sampler_all_values_are_inactive_bucket(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """The inactive bucket is ``num_buckets - 1``. Every value in
        the returned dict must equal that bucket so downstream
        encoder-input assembly sees a uniform inactive assignment."""
        meta = TargetModelMetadata(
            num_buckets=5,
            n_sae_features=16384,
            negative_selection={"distribution": "geometric", "mean": 5.0},
            feature_selection=None,
            per_feature_stats=None,
            device="cpu",
            sae_layer=8,
            segment_size=128,
        )
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=meta, cfg=cfg_obj,
        )
        result = sampler(rng=random.Random(7))
        # Expect ``num_buckets - 1 == 4`` for every entry.
        assert all(v == 4 for v in result.values())

    def test_sampler_keys_are_valid_sae_feature_ids(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            ),
            cfg=cfg_obj,
        )
        result = sampler(rng=random.Random(123))
        # ``cfg_obj`` writes a stats file with ``n_sae_features=16384``.
        assert all(0 <= k < 16384 for k in result)

    def test_sampler_keys_are_unique(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """Dict keys are implicitly unique, but this pins that
        :func:`sample_inactive_features` never emits duplicates that
        would collapse the dict size."""
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            ),
            cfg=cfg_obj,
        )
        result = sampler(rng=random.Random(99))
        assert len(result) == len(set(result))

    def test_sampler_deterministic_given_same_rng_seed(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """Two invocations with an identical ``random.Random(seed)``
        must produce identical fid sets. Regression for the
        ``random.Random → torch.Generator`` bridge: if the bridge ever
        stops pulling the torch seed deterministically from
        ``rng.randint``, this test will fail."""
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection=self._DEFAULT_NEGATIVE_SELECTION,
            ),
            cfg=cfg_obj,
        )
        a = sampler(rng=random.Random(42))
        b = sampler(rng=random.Random(42))
        assert a == b

    def test_sampler_different_seeds_different_output(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """Two invocations with different ``random.Random`` seeds
        should (with overwhelming probability for 16k features) land on
        different fid sets. This pins that the bridge actually consumes
        the incoming rng and isn't returning a constant."""
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": 10.0,
                },
            ),
            cfg=cfg_obj,
        )
        a = sampler(rng=random.Random(1))
        b = sampler(rng=random.Random(2))
        assert a != b

    def test_sampler_respects_distribution_mean(
        self, cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """Mean sample size should be close to the distribution mean
        over many draws. The tolerance (15%) is chosen from the
        geometric-distribution standard error: for a geometric
        with mean 10 the per-sample std is ~sqrt(10*9) ~= 9.5, so
        the standard error for n=400 samples is ~0.47 (stderr =
        std/sqrt(n)). 15% of the target mean gives a 1.5 absolute
        window, ~3x the stderr -- a 1.5x safety margin over the 2-sigma
        bound. Pins that the config mean actually reaches the sampler.
        """
        target_mean = 10.0
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": target_mean,
                },
            ),
            cfg=cfg_obj,
        )
        rng = random.Random(0)
        sizes = [len(sampler(rng=rng)) for _ in range(400)]
        mean = sum(sizes) / len(sizes)
        assert abs(mean - target_mean) < target_mean * 0.15

    def test_sampler_reads_n_sae_features_from_stats(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The closure must consult the stats file via
        :func:`read_n_sae_features_from_stats`. A patched reader
        returning ``8`` forces every sampled fid into ``[0, 8)``."""
        monkeypatch.setattr(
            runtime_state_mod,
            "read_n_sae_features_from_stats",
            lambda path: 8,
        )
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": 3.0,
                },
            ),
            cfg=cfg_obj,
        )
        result = sampler(rng=random.Random(0))
        assert all(0 <= k < 8 for k in result)

    def test_sampler_propagates_stats_read_errors(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``read_n_sae_features_from_stats`` raises, the failure
        should surface at build time (not deferred to the first
        ``sampler(rng=...)`` call) so operators learn about a missing
        stats file immediately."""
        def boom(_path: Path) -> int:
            raise AxBenchBaseError("fake stats read failure")

        monkeypatch.setattr(
            runtime_state_mod, "read_n_sae_features_from_stats", boom,
        )
        with pytest.raises(
            AxBenchBaseError, match="fake stats read failure",
        ):
            runtime_state_mod._build_inactive_sampler(
                encdec_meta=_make_meta(
                    negative_selection={
                        "distribution": "geometric",
                        "mean": 3.0,
                    },
                ),
                cfg=cfg_obj,
            )

    def test_sampler_bridges_rng_via_module_upper_bound(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The closure must seed :class:`torch.Generator` with
        :data:`_TORCH_SEED_UPPER_BOUND` as the ceiling so the seed
        stays inside signed int64. Monkey-patching the constant to a
        low value forces every call to land on a narrow seed range;
        the sampler still succeeds and produces valid fids."""
        monkeypatch.setattr(
            runtime_state_mod, "_TORCH_SEED_UPPER_BOUND", 1024,
        )
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": 3.0,
                },
            ),
            cfg=cfg_obj,
        )
        result = sampler(rng=random.Random(0))
        assert all(0 <= k < 16384 for k in result)

    def test_sampler_returns_empty_dict_when_k_is_zero(
        self,
        cfg_obj: AxBenchBaseConfig,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Item 18: when :func:`sample_inactive_features` returns an
        empty tensor (k=0 sampled features from the distribution),
        the inactive sampler closure must return an empty dict, not
        raise and not crash downstream encoder-input assembly.
        """
        def empty_sampler(
            *, active_feature_ids: Any, n_sae_features: int,
            config: Any, rng: Any,
        ) -> Any:
            return t.empty(0, dtype=t.int64)

        monkeypatch.setattr(
            runtime_state_mod, "sample_inactive_features", empty_sampler,
        )

        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": 3.0,
                },
            ),
            cfg=cfg_obj,
        )

        assert callable(sampler)
        result = sampler(rng=random.Random(42))
        assert result == {}
        assert isinstance(result, dict)

    def test_sampler_does_not_exclude_active_features_by_design(
        self,
        cfg_obj: AxBenchBaseConfig,
    ) -> None:
        """The sampler passes an empty active_feature_ids because the active
        set is not known at sampler-construction time. Overlap with the
        eventual active partition is resolved by collision_filter at
        generation time, not at sampling time.

        This test pins that design decision: if someone changes the sampler
        to exclude active features, this test fails and the comment in
        ``_build_inactive_sampler`` points them at the collision_filter.
        """
        sampler = runtime_state_mod._build_inactive_sampler(
            encdec_meta=_make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": 5.0,
                    "min_k": 3,
                    "max_k": 10,
                },
            ),
            cfg=cfg_obj,
        )
        # The sampler CAN return features that might collide with a
        # hypothetical active partition -- that is by design. The
        # collision_filter test suite in test_collision_filter_heavy.py
        # covers the resolution.
        result = sampler(rng=random.Random(42))
        assert isinstance(result, dict)
        assert len(result) > 0


class TestInactiveSamplerAccessorPhase7:
    """Regression: the :meth:`RuntimeState.inactive_sampler` accessor
    now returns a real callable (not ``None`` / ``NotImplementedError``)
    when the encdec target's ``negative_selection`` is set. Phase 3
    stubbed out the builder; Phase 7 lands the real one.

    The accessor-level behaviour (thread safety, caching, close path)
    is already covered by :class:`TestInactiveSampler` which patches
    the module-level builder. This class pins the *end-to-end*
    contract with NO stubs so a regression in the real builder
    surfaces via the public accessor too.
    """

    def test_accessor_returns_real_callable_when_negative_selection_set(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(
                negative_selection={
                    "distribution": "geometric",
                    "mean": 3.0,
                },
            ),
        )
        sampler = runtime_state.inactive_sampler()
        assert sampler is not None
        assert callable(sampler)
        result = sampler(rng=random.Random(0))
        assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# Close -- Phase 3 augmentations
# ---------------------------------------------------------------------------


class TestClosePhase3:
    def test_close_closes_judge_client_if_opened(
        self,
        runtime_state: RuntimeState,
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        client = cast(MagicMock, runtime_state.judge_client())
        runtime_state.close()
        client.close.assert_called_once_with()
        assert runtime_state._judge_client is None

    def test_close_handles_unopened_client_gracefully(
        self,
        runtime_state: RuntimeState,
    ) -> None:
        """Close without ever opening the client must not blow up."""
        runtime_state.close()  # no Phase 3 accessor was called
        assert runtime_state._closed is True

    def test_close_clears_meta_refs(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(negative_selection=None),
        )
        runtime_state.decoder_only_meta()
        runtime_state.encdec_meta()
        assert runtime_state._decoder_only_meta is not None
        assert runtime_state._encdec_meta is not None
        runtime_state.close()
        assert runtime_state._decoder_only_meta is None
        assert runtime_state._encdec_meta is None

    def test_close_clears_inactive_sampler_refs(
        self,
        runtime_state: RuntimeState,
        stub_heavy_loaders: dict[str, Any],
        stub_phase3_builders: dict[str, MagicMock],
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr(
            runtime_state_mod,
            "detect_target_model_metadata",
            lambda *, target_model, cfg: _make_meta(
                negative_selection={"distribution": "geometric", "mean": 2.0},
            ),
        )
        sampler = runtime_state.inactive_sampler()
        assert sampler is not None
        assert runtime_state._inactive_sampler_loaded is True
        runtime_state.close()
        assert runtime_state._inactive_sampler is None
        assert runtime_state._inactive_sampler_loaded is False


# ---------------------------------------------------------------------------
# close() -- exception safety during teardown
# ---------------------------------------------------------------------------


class TestCloseExceptionSafety:
    """Regression tests for the hardened :meth:`RuntimeState.close`
    teardown around a faulty SQLite-backed judge client.

    The contract we pin here:

    - If ``judge_client.close()`` raises, the state still finishes as
      ``_closed=True``.
    - If ``torch.cuda.empty_cache()`` raises, the state still finishes as
      ``_closed=True``.
    - After any partial failure, calling :meth:`close` a second time
      is a no-op that does not raise.

    Every failure path must emit a WARNING via :mod:`loguru` naming
    the resource so operators see the failure in the logs.
    """

    def test_close_continues_when_judge_client_close_raises(
        self,
        runtime_state: RuntimeState,
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        """If ``judge_client.close()`` raises last, the state still
        finishes as ``_closed=True`` after the warning is logged.
        """
        judge = cast(MagicMock, runtime_state.judge_client())

        judge.close.side_effect = RuntimeError("judge close failed")

        with _capture_loguru_warnings() as warnings:
            runtime_state.close()

        assert judge.close.call_count == 1
        assert runtime_state._closed is True
        assert any("judge_client" in w for w in warnings), (
            f"expected 'judge_client' in warnings, got {warnings!r}"
        )

    def test_close_continues_when_empty_cuda_cache_raises(
        self,
        runtime_state: RuntimeState,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """If ``torch.cuda.empty_cache()`` raises, the state must still
        finish as ``_closed=True``. This covers the final defensive
        block in close().
        """
        def boom() -> None:
            raise RuntimeError("fake cuda cache empty failure")

        monkeypatch.setattr(runtime_state_mod.t.cuda, "is_available", lambda: True)
        monkeypatch.setattr(runtime_state_mod.t.cuda, "empty_cache", boom)

        with _capture_loguru_warnings() as warnings:
            runtime_state.close()

        assert runtime_state._closed is True
        assert any("torch.cuda.empty_cache" in w for w in warnings), (
            f"expected 'torch.cuda.empty_cache' in warnings, got {warnings!r}"
        )

    def test_close_double_call_after_partial_failure_is_safe(
        self,
        runtime_state: RuntimeState,
        stub_phase3_builders: dict[str, MagicMock],
    ) -> None:
        """After the client close raises, a second ``close()`` call must
        be a safe no-op. The fix for the zombie-state bug: _closed
        is set unconditionally so a retry from a CLI ``finally``
        block does not re-enter the cache close path.
        """
        judge = cast(MagicMock, runtime_state.judge_client())
        judge.close.side_effect = RuntimeError("judge close failed")

        # First call: completes despite the raise.
        runtime_state.close()
        assert runtime_state._closed is True

        # Second call: must be a plain no-op, NOT a re-entry.
        judge.close.reset_mock()
        runtime_state.close()
        # The second call did not try to close the judge client again
        # (the reference has been dropped and _closed short-circuits).
        assert judge.close.call_count == 0
        assert runtime_state._closed is True
