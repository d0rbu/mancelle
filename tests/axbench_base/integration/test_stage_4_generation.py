"""Stage 4 ``generate_all`` end-to-end integration tests.

Exercises :func:`steering.eval.axbench_base.generation.generate_all`
with :func:`run_intervention` monkey-patched to return canned
:class:`GenerationResult` values. The outer loop is covered at unit-test
depth under ``tests/axbench_base/unit/test_generate_all.py``; these
integration tests verify the wider dispatch integration with the
parquet directory-dataset and method registry:

* All registered methods are called exactly once each on a minimal
  1-concept / 1-seed / 1-coefficient-per-method setup.
* Text rows land in the generations directory-dataset with non-empty
  ``generation_text`` columns.
* Methods whose ``requires_negative_sampling=True`` are filtered out
  when both target models have ``negative_selection is None``.
* The returned directory path is correct and schema-compliant.
* Invalid method names in ``cfg.enabled_methods`` raise
  :class:`AxBenchBaseError` from the filter helper.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch.nn as nn

from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
)
from steering.eval.axbench_base.generation import (
    GENERATIONS_SCHEMA,
)
from steering.eval.axbench_base.generation import (
    generate_all as _generate_all,
)
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.method_registry import ALL_METHODS
from steering.eval.axbench_base.parquet_writer import load_existing
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    TargetBundle,
    TargetModelMetadata,
)
from tests.axbench_base import paths as test_paths
from tests.axbench_base.fixtures.stage4_contract import (
    Stage4TestContextProvider,
    commit_seed_contract,
    commit_stage4_context_contract,
)


def generate_all(*args: Any, apply_max_act_scale: bool = False, **kwargs: Any) -> Path:
    return _generate_all(
        *args,
        apply_max_act_scale=apply_max_act_scale,
        **kwargs,
    )


def _make_targets(
    target_models: dict[str, nn.Module],
    metas: dict[str, TargetModelMetadata],
    tokenizers: dict[str, Any],
    *,
    context_provider: Stage4TestContextProvider | None = None,
) -> list[TargetBundle]:
    """Build a list of :class:`TargetBundle` from the old dict-style args."""
    return [
        TargetBundle(
            name=name,
            model=target_models[name],
            meta=metas[name],
            tokenizer=tokenizers[name],
            context_provider=context_provider if name == "encdec" else None,
        )
        for name in target_models
    ]


pytestmark = pytest.mark.slow
_TARGET_SEQUENCE_LEN = 2048


# ---------------------------------------------------------------------------
# Local helpers
# ---------------------------------------------------------------------------


def _concept(concept_id: int = 0, description: str = "the color blue") -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id * 7,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=description,
        corpus_count=100,
        description_source="stub",
    )


def _seed_row(doc_idx: int = 0, text: str = "once upon a time") -> SeedRow:
    tokenizer = _StubTokenizer()
    continuation_text = "cont"
    prefix_tokens = tokenizer.encode(text)
    continuation_tokens = tokenizer.encode(continuation_text)
    return SeedRow(
        shard_idx=0,
        doc_idx=doc_idx,
        source_nanochat_token_count=len(prefix_tokens) + len(continuation_tokens),
        prefix_source_token_start=0,
        prefix_source_token_end=len(prefix_tokens),
        prefix_text=text,
        prefix_content_nanochat_tokens=prefix_tokens,
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text=continuation_text,
        continuation_content_nanochat_tokens=continuation_tokens,
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


def _make_cfg(
    tmp_path: Path,
    *,
    run_tag: str = "stage4_int",
    enabled_methods: list[str] | None = None,
) -> AxBenchBaseConfig:
    return AxBenchBaseConfig(
        feature_dir=(
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        ),
        feature_statistics_path="stats.safetensors",
        decoder_only={"source": "base", "model_tag": "d13", "step": 10000},
        encdec={"source": "encdec", "model_tag": "d13", "step": 10000},
        device="cpu",
        output_root=str(tmp_path / "eval_out"),
        run_tag=run_tag,
        generation=GenerationConfig(
            max_new_tokens=4,
            temperature=0.7,
            top_k_sampling=50,
            master_seed=13,
        ),
        seed_continuation_nanochat_tokens=4,
        enabled_methods=enabled_methods,
    )


class _StubTokenizer:
    """Character-hash tokenizer; only needed by the Prompting path."""

    def encode(self, text: str, prepend: int | None = None) -> list[int]:
        tokens = [ord(char) + 2 for char in text]
        return [prepend, *tokens] if prepend is not None else tokens

    def decode(self, ids: list[int]) -> str:
        return "".join("<bos>" if token == 1 else chr(token - 2) for token in ids)

    def get_bos_token_id(self) -> int:
        return 1


class _FakeTargetModule(nn.Linear):
    config: SimpleNamespace
    dec_config: SimpleNamespace

    def __init__(self) -> None:
        super().__init__(2, 2)
        self.config = SimpleNamespace(sequence_len=_TARGET_SEQUENCE_LEN)
        self.dec_config = SimpleNamespace(sequence_len=_TARGET_SEQUENCE_LEN)


def _fake_target_module() -> nn.Module:
    return _FakeTargetModule()


def _patch_run_intervention(
    monkeypatch: pytest.MonkeyPatch,
    *,
    return_text: str = "gen text",
    token_count: int = 3,
    stop_reason: str = "bos_token",
) -> MagicMock:
    """Replace ``generation.run_intervention`` with a tracked mock.

    Every call returns an identical ``GenerationResult`` — enough for
    integration tests that only care about invocation count /
    schema / resume semantics.
    """
    mock = MagicMock(
        side_effect=itertools.cycle(
            [
                GenerationResult(
                    text=return_text,
                    token_count=token_count,
                    stop_reason=stop_reason,
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.run_intervention",
        mock,
    )
    return mock


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def tokenizers() -> dict[str, Any]:
    tok = _StubTokenizer()
    return {"decoder_only": tok, "encdec": tok}


@pytest.fixture
def target_models() -> dict[str, nn.Module]:
    return {
        "decoder_only": _fake_target_module(),
        "encdec": _fake_target_module(),
    }


def _prepare_targets(
    *,
    cfg: AxBenchBaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    concepts: list[Concept],
    pos_seeds: list[SeedRow],
    neg_seeds: list[SeedRow],
    target_models: dict[str, nn.Module],
    metas: dict[str, TargetModelMetadata],
    tokenizers: dict[str, Any],
) -> list[TargetBundle]:
    _, context_provider = commit_stage4_context_contract(
        cfg,
        monkeypatch=monkeypatch,
        tokenizer=tokenizers["encdec"],
        concepts=concepts,
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
        encdec_meta=metas["encdec"],
        encdec_sequence_len=_TARGET_SEQUENCE_LEN,
    )
    return _make_targets(
        target_models,
        metas,
        tokenizers,
        context_provider=context_provider,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_4_all_methods_called_once_each(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """With 1 concept, 1 eval seed per mode, and a single-condition grid
    forced across every method, ``generate_all`` calls
    ``run_intervention`` once per registry entry — assuming both targets
    have negative sampling enabled so none of the 3
    ``requires_negative_sampling=True`` methods are filtered out.
    """
    cfg = _make_cfg(tmp_run_root)
    metas = {
        "decoder_only": _meta(
            num_buckets=2, negative_selection={"strategy": "uniform"},
        ),
        "encdec": _meta(
            num_buckets=2, negative_selection={"strategy": "uniform"},
        ),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0, text="pos prefix")]
    neg_seeds = [_seed_row(doc_idx=1, text="neg prefix")]
    mock = _patch_run_intervention(monkeypatch)
    targets = _prepare_targets(
        cfg=cfg,
        monkeypatch=monkeypatch,
        concepts=concepts,
        pos_seeds=pos_seeds,
        neg_seeds=neg_seeds,
        target_models=target_models,
        metas=metas,
        tokenizers=tokenizers,
    )

    generate_all(
        cfg,
        concepts=concepts,
        targets=targets,
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    # Expected: sum over every method of its grid size.
    expected = 0
    for method in ALL_METHODS:
        coefs = method.coefficient_grid_fn(metas[method.target_model], cfg)
        expected += len(coefs)
    assert mock.call_count == expected
    # Each registered method appears in at least one call.
    called_names = {
        call.kwargs["method"].name for call in mock.call_args_list
    }
    assert called_names == {m.name for m in ALL_METHODS}
    assert len(called_names) == len(ALL_METHODS)


def test_stage_4_output_non_empty_per_method(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Every row written by ``generate_all`` carries the mocked
    ``gen text`` in its ``generation_text`` column, and the method
    coverage matches the full registry."""
    cfg = _make_cfg(tmp_run_root, run_tag="stage4_nonempty")
    metas = {
        "decoder_only": _meta(
            num_buckets=2, negative_selection={"strategy": "uniform"},
        ),
        "encdec": _meta(
            num_buckets=2, negative_selection={"strategy": "uniform"},
        ),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]
    _patch_run_intervention(monkeypatch, return_text="gen text")
    targets = _prepare_targets(
        cfg=cfg,
        monkeypatch=monkeypatch,
        concepts=concepts,
        pos_seeds=pos_seeds,
        neg_seeds=neg_seeds,
        target_models=target_models,
        metas=metas,
        tokenizers=tokenizers,
    )

    gen_dir = generate_all(
        cfg,
        concepts=concepts,
        targets=targets,
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    assert gen_dir == test_paths.default_paths("stage4_nonempty").generations_dir()
    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.num_rows > 0
    rows = table.to_pylist()
    seen_methods: set[str] = set()
    for row in rows:
        assert row["generation_text"] == "gen text"
        assert row["token_count"] == 3
        assert row["stop_reason"] == "bos_token"
        seen_methods.add(row["method_name"])
    assert seen_methods == {m.name for m in ALL_METHODS}


def test_stage_4_skips_d12_negative_sampling_methods_when_meta_negative_selection_none(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """When every target has ``negative_selection is None`` the 3
    ``requires_negative_sampling=True`` methods must be filtered out.

    The registry count goes from 25 down to 22 — d12 because those
    3 methods are the ``neg_ours_add_to_negatives_encdec``,
    ``neg_ours_replace_random_negatives_encdec``, and
    ``neg_ours_replace_all_negatives_encdec`` variants that need the
    inactive sampler to have anything to draw from.
    """
    cfg = _make_cfg(tmp_run_root, run_tag="stage4_no_neg_sampling")
    metas = {
        "decoder_only": _meta(num_buckets=2, negative_selection=None),
        "encdec": _meta(num_buckets=2, negative_selection=None),
    }
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]
    mock = _patch_run_intervention(monkeypatch)
    targets = _prepare_targets(
        cfg=cfg,
        monkeypatch=monkeypatch,
        concepts=concepts,
        pos_seeds=pos_seeds,
        neg_seeds=neg_seeds,
        target_models=target_models,
        metas=metas,
        tokenizers=tokenizers,
    )

    generate_all(
        cfg,
        concepts=concepts,
        targets=targets,
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    called_names = {
        call.kwargs["method"].name for call in mock.call_args_list
    }
    forbidden = {
        "neg_ours_add_to_negatives_encdec",
        "neg_ours_replace_random_negatives_encdec",
        "neg_ours_replace_all_negatives_encdec",
    }
    assert forbidden.isdisjoint(called_names)
    # Every other method in the registry is still called at least once.
    remaining = {m.name for m in ALL_METHODS} - forbidden
    assert remaining == called_names
    assert len(called_names) == len(ALL_METHODS) - len(forbidden)


def test_stage_4_writes_generations_directory_dataset(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """The returned path is a directory containing at least one
    ``batch_NNNNNN.parquet`` file with the full generations schema."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage4_writes_dir",
        enabled_methods=["diffmean_decoder_only", "lora_decoder_only"],
    )
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)
    concepts = [_concept(0)]
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]
    commit_seed_contract(
        cfg,
        monkeypatch=monkeypatch,
        tokenizer=tokenizers["decoder_only"],
        eval_rows=[*pos_seeds, *neg_seeds],
    )

    gen_dir = generate_all(
        cfg,
        concepts=concepts,
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=pos_seeds,
        eval_seeds_negative=neg_seeds,
    )

    assert gen_dir.is_dir()
    batch_files = sorted(gen_dir.glob("batch_*.parquet"))
    assert len(batch_files) >= 1
    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.schema.equals(GENERATIONS_SCHEMA)
    # diffmean has 10 coefs; lora has 1. 1 concept x 1 seed each = 11 rows.
    assert table.num_rows == 11


def test_stage_4_invalid_method_in_enabled_list_raises(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """An unknown method name surfaces as :class:`AxBenchBaseError`
    rather than silently being dropped by the filter."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage4_bad_method",
        enabled_methods=["not_a_real_method_xyz"],
    )
    metas = {
        "decoder_only": _meta(),
        "encdec": _meta(),
    }
    _patch_run_intervention(monkeypatch)
    pos_seeds = [_seed_row(doc_idx=0)]
    neg_seeds = [_seed_row(doc_idx=1)]
    commit_seed_contract(
        cfg,
        monkeypatch=monkeypatch,
        tokenizer=tokenizers["decoder_only"],
        eval_rows=[*pos_seeds, *neg_seeds],
    )

    with pytest.raises(AxBenchBaseError, match="not_a_real_method_xyz"):
        generate_all(
            cfg,
            concepts=[_concept(0)],
            targets=_make_targets(target_models, metas, tokenizers),
            eval_seeds_positive=pos_seeds,
            eval_seeds_negative=neg_seeds,
        )
