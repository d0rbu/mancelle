"""Stage 4 resume integration tests.

Exercises the crash-safe resume path of
:func:`steering.eval.axbench_base.generation.generate_all`:

* Pre-populated rows with matching ``generation_key`` values are
  skipped on the next call.
* The final directory contains both the pre-existing rows AND the
  newly-generated rows (nothing is clobbered).
* An empty / non-existent generations dir triggers a normal full run.
* A "partial" ``batch_NNNNNN.parquet`` (non-zero index, gap-free
  schema, but only a subset of the total-run rows) is read back on
  resume and survives unchanged, with the writer's zero-padded
  cursor advancing to ``max(index) + 1``.
"""

from __future__ import annotations

import itertools
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch as t
import torch.nn as nn
from safetensors.torch import save_file

from steering.eval.axbench_base import paths
from steering.eval.axbench_base.config import (
    AxBenchBaseConfig,
    GenerationConfig,
)
from steering.eval.axbench_base.direction_scale import (
    STEERING_SCALE_MARKER_NAME,
    direction_scale_identity,
    encode_max_activation,
)
from steering.eval.axbench_base.generation import (
    GENERATIONS_SCHEMA,
    build_generation_key,
)
from steering.eval.axbench_base.generation import (
    generate_all as _generate_all,
)
from steering.eval.axbench_base.generation_helpers import GenerationResult
from steering.eval.axbench_base.parquet_writer import (
    ParquetAppendWriter,
    load_existing,
)
from steering.eval.axbench_base.stage4_prompts import (
    build_stage4_prompt_tokens,
    prompt_token_key,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    Concept,
    SeedRow,
    SteeringMode,
    TargetBundle,
    TargetModelMetadata,
)
from tests.axbench_base.fixtures.stage4_contract import commit_seed_contract


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
) -> list[TargetBundle]:
    """Build a list of :class:`TargetBundle` from the old dict-style args."""
    return [
        TargetBundle(
            name=name,
            model=target_models[name],
            meta=metas[name],
            tokenizer=tokenizers[name],
            context_provider=None,
        )
        for name in target_models
    ]


pytestmark = pytest.mark.slow
_TARGET_SEQUENCE_LEN = 2048


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_DIFFMEAN_COEFS: tuple[float, ...] = (
    0.2, 0.4, 0.6, 0.8, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
)


def _concept(concept_id: int = 0) -> Concept:
    return Concept(
        concept_id=concept_id,
        feature_index=concept_id * 7,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description="concept description",
        corpus_count=100,
        description_source="stub",
    )


def _seed_row(doc_idx: int = 0, text: str = "prefix") -> SeedRow:
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


def _meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=4,
        n_sae_features=16384,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=8,
        segment_size=128,
    )


def _make_cfg(
    tmp_path: Path,
    *,
    run_tag: str = "stage4_resume",
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
    return_text: str = "new gen",
) -> MagicMock:
    mock = MagicMock(
        side_effect=itertools.cycle(
            [
                GenerationResult(
                    text=return_text,
                    token_count=3,
                    stop_reason="bos_token",
                ),
            ],
        ),
    )
    monkeypatch.setattr(
        "steering.eval.axbench_base.generation.run_intervention",
        mock,
    )
    return mock


def _write_existing_rows(
    gen_dir: Path,
    *,
    method_name: str,
    concept_id: int,
    target_model_name: str,
    mode: str,
    coefs: list[float],
    seed_pool_fingerprint: str,
    generation_text: str = "old gen",
    prefix_text: str = "prefix",
    start_batch_index: int | None = None,
) -> None:
    """Pre-populate the generations directory with rows for ``coefs``.

    When ``start_batch_index`` is provided, the underlying batch file
    is renamed to ``batch_{start_batch_index:06d}.parquet`` so the
    writer's resume cursor has to skip past it. That path exercises the
    "partial batch NNNNNN" case in
    :func:`test_stage_4_resume_ignores_partial_batch_file`.
    """
    gen_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = _StubTokenizer()
    seed = _seed_row(text=prefix_text)
    prompt_tokens = build_stage4_prompt_tokens(
        tokenizer=tokenizer,
        seed_content_tokens=seed.prefix_content_nanochat_tokens,
    )
    with ParquetAppendWriter(
        out_dir=gen_dir, schema=GENERATIONS_SCHEMA, batch_size=1000,
    ) as writer:
        for coef in coefs:
            writer.append(
                {
                    "generation_key": build_generation_key(
                        method_name=method_name,
                        concept_id=concept_id,
                        eval_seed=0,
                        coefficient=coef,
                        target_model_name=target_model_name,
                    ),
                    "concept_id": concept_id,
                    "method_name": method_name,
                    "mode": mode,
                    "eval_seed": 0,
                    "coefficient": str(coef),
                    "generation_text": generation_text,
                    "target_model": target_model_name,
                    "prefix_text": prefix_text,
                    "prompt_token_key": prompt_token_key(prompt_tokens),
                    "prompt_token_count": len(prompt_tokens),
                    "seed_pool_fingerprint": seed_pool_fingerprint,
                    "token_count": 2,
                    "stop_reason": "bos_token",
                },
            )

    if start_batch_index is None:
        return

    # Rename the on-disk batch so the writer's cursor has to skip past
    # ``start_batch_index`` — a crash-mid-sweep simulation.
    existing = sorted(gen_dir.glob("batch_*.parquet"))
    assert len(existing) == 1, (
        f"expected exactly one batch, got {len(existing)} in {gen_dir}"
    )
    renamed = gen_dir / f"batch_{start_batch_index:06d}.parquet"
    existing[0].rename(renamed)


# ---------------------------------------------------------------------------
# Fixtures
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


def _commit_test_seed_contract(
    cfg: AxBenchBaseConfig,
    monkeypatch: pytest.MonkeyPatch,
    tokenizer: Any,
) -> str:
    return commit_seed_contract(
        cfg,
        monkeypatch=monkeypatch,
        tokenizer=tokenizer,
        eval_rows=[_seed_row(doc_idx=0), _seed_row(doc_idx=1)],
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_stage_4_resume_skips_completed_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """Pre-populate 3 of the 10 diffmean rows and verify only the
    remaining 7 trigger a new ``run_intervention`` call on resume."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage4_resume_skip",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])

    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:3]),
        seed_pool_fingerprint=seed_fingerprint,
    )

    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row(doc_idx=0)],
        eval_seeds_negative=[_seed_row(doc_idx=1)],
    )

    # 10 coefs total, 3 pre-existing → only 7 new calls.
    assert mock.call_count == 7


def test_stage_4_resume_preserves_existing_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """After resume, the generations dir contains ALL 10 rows: 3
    pre-existing (with ``old gen`` text) + 7 new (with ``new gen``
    text), keyed by unique generation_key."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage4_resume_preserve",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])

    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:3]),
        seed_pool_fingerprint=seed_fingerprint,
        generation_text="old gen",
    )

    _patch_run_intervention(monkeypatch, return_text="new gen")

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row(doc_idx=0)],
        eval_seeds_negative=[_seed_row(doc_idx=1)],
    )

    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.num_rows == 10
    rows = table.to_pylist()
    all_keys = {row["generation_key"] for row in rows}
    assert len(all_keys) == 10

    # Pre-existing rows carry ``old gen``, new rows carry ``new gen``.
    old_text_count = sum(1 for r in rows if r["generation_text"] == "old gen")
    new_text_count = sum(1 for r in rows if r["generation_text"] == "new gen")
    assert old_text_count == 3
    assert new_text_count == 7

    # Every coefficient in the full grid is represented exactly once.
    coef_strings = {r["coefficient"] for r in rows}
    assert coef_strings == {str(c) for c in _DIFFMEAN_COEFS}


def test_stage_4_resume_empty_dir_writes_from_scratch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """An empty generations dir behaves as a normal first-time run —
    every row in the diffmean 10-coef grid is generated."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage4_resume_empty",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])

    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    gen_dir.mkdir(parents=True, exist_ok=True)
    # No batch files at all — just the directory.
    assert list(gen_dir.iterdir()) == []

    mock = _patch_run_intervention(monkeypatch)

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row(doc_idx=0)],
        eval_seeds_negative=[_seed_row(doc_idx=1)],
    )

    assert mock.call_count == 10
    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.num_rows == 10


def test_stage_4_resume_ignores_partial_batch_file(
    monkeypatch: pytest.MonkeyPatch,
    tmp_run_root: Path,
    tokenizers: dict[str, Any],
    target_models: dict[str, nn.Module],
) -> None:
    """A pre-existing batch file at a non-zero index (e.g. `batch_000005`)
    is still read on resume, and the writer's cursor advances past it
    without clobbering the file or losing its rows."""
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="stage4_resume_partial",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])

    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    # Two pre-existing rows, forced to live in `batch_000005.parquet`.
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:2]),
        seed_pool_fingerprint=seed_fingerprint,
        generation_text="partial gen",
        start_batch_index=5,
    )
    # Sanity check: the partial file is at index 5, not 0.
    assert (gen_dir / "batch_000005.parquet").exists()
    assert not (gen_dir / "batch_000000.parquet").exists()

    mock = _patch_run_intervention(monkeypatch, return_text="resume gen")

    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=_make_targets(target_models, metas, tokenizers),
        eval_seeds_positive=[_seed_row(doc_idx=0)],
        eval_seeds_negative=[_seed_row(doc_idx=1)],
    )

    # 10 coefs total, 2 pre-existing inside batch_000005.parquet → 8 new calls.
    assert mock.call_count == 8

    # The pre-existing batch file is untouched.
    assert (gen_dir / "batch_000005.parquet").exists()

    table = load_existing(gen_dir, schema=GENERATIONS_SCHEMA)
    assert table.num_rows == 10
    rows = table.to_pylist()
    partial_text_count = sum(
        1 for r in rows if r["generation_text"] == "partial gen"
    )
    resume_text_count = sum(
        1 for r in rows if r["generation_text"] == "resume gen"
    )
    assert partial_text_count == 2
    assert resume_text_count == 8


# ---------------------------------------------------------------------------
# Direction-owned scale resume guard
# ---------------------------------------------------------------------------


def _write_direction_artifact(cfg: AxBenchBaseConfig, value: float) -> str:
    path = paths.AxBenchBasePaths.from_config(cfg).diffmean_direction(
        "decoder_only", SteeringMode.POSITIVE, 0
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"direction": t.tensor([1.0, 0.0])},
        str(path),
        metadata={
            "concept_id": "0",
            "mode": "positive",
            "target_model": "decoder_only",
            "layer": str(cfg.layer_hooks.diffmean_and_probe_layer),
            "n_pos": "2",
            "n_neg": "2",
            "max_activation": encode_max_activation(value),
        },
    )
    return direction_scale_identity(
        {("diffmean", "decoder_only", "positive", 0): value}
    )


def _run(cfg: AxBenchBaseConfig, *, targets: list[TargetBundle]) -> None:
    generate_all(
        cfg,
        concepts=[_concept(0)],
        targets=targets,
        eval_seeds_positive=[_seed_row(doc_idx=0)],
        eval_seeds_negative=[_seed_row(doc_idx=1)],
        apply_max_act_scale=True,
    )


def test_resume_guard_fresh_run_stamps_marker(
    monkeypatch, tmp_run_root, tmp_path, tokenizers, target_models
):
    cfg = _make_cfg(
        tmp_run_root, run_tag="guard_fresh", enabled_methods=["diffmean_decoder_only"]
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])
    _patch_run_intervention(monkeypatch)
    identity = _write_direction_artifact(cfg, 44.0)

    _run(
        cfg,
        targets=_make_targets(target_models, metas, tokenizers),
    )

    marker = paths.AxBenchBasePaths.from_config(cfg).generations_dir() / STEERING_SCALE_MARKER_NAME
    assert marker.read_text() == identity


def test_resume_guard_matching_identity_resumes_cleanly(
    monkeypatch, tmp_run_root, tmp_path, tokenizers, target_models
):
    cfg = _make_cfg(tmp_run_root, run_tag="guard_match", enabled_methods=["diffmean_decoder_only"])
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:3]),
        seed_pool_fingerprint=seed_fingerprint,
    )
    identity = _write_direction_artifact(cfg, 44.0)
    # Pre-stamp the marker with the SAME identity this run resolves.
    (gen_dir / STEERING_SCALE_MARKER_NAME).write_text(identity)
    mock = _patch_run_intervention(monkeypatch)

    _run(
        cfg,
        targets=_make_targets(target_models, metas, tokenizers),
    )
    # Clean resume: only the 7 missing rows are (re)generated.
    assert mock.call_count == 7


def test_resume_guard_mismatched_identity_raises(
    monkeypatch, tmp_run_root, tmp_path, tokenizers, target_models
):
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="guard_mismatch",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:3]),
        seed_pool_fingerprint=seed_fingerprint,
    )
    # A stale marker from a DIFFERENT scale regime.
    (gen_dir / STEERING_SCALE_MARKER_NAME).write_text("dirmaxact:staleidentity0")
    _patch_run_intervention(monkeypatch)
    _write_direction_artifact(cfg, 44.0)

    with pytest.raises(AxBenchBaseError, match="DIFFERENT steering-scale regime"):
        _run(
            cfg,
            targets=_make_targets(target_models, metas, tokenizers),
        )


def test_resume_guard_old_maxact_marker_raises(
    monkeypatch, tmp_run_root, tokenizers, target_models
):
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="guard_old_maxact",
        enabled_methods=["diffmean_decoder_only"],
    )
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(
        cfg, monkeypatch, tokenizers["decoder_only"]
    )
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:3]),
        seed_pool_fingerprint=seed_fingerprint,
    )
    (gen_dir / STEERING_SCALE_MARKER_NAME).write_text(
        "maxact:0123456789abcdef"
    )
    _write_direction_artifact(cfg, 44.0)
    _patch_run_intervention(monkeypatch)

    with pytest.raises(
        AxBenchBaseError,
        match=r"pre-#1228 SAE max_act regime.*fresh run_tag",
    ):
        _run(
            cfg,
            targets=_make_targets(target_models, metas, tokenizers),
        )


def test_resume_guard_preguard_rows_without_marker_raise(
    monkeypatch, tmp_run_root, tmp_path, tokenizers, target_models
):
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="guard_preguard",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    # Existing DiffMean rows (resume_active) but NO marker == scored pre-#588.
    _write_existing_rows(
        gen_dir,
        method_name="diffmean_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=list(_DIFFMEAN_COEFS[:3]),
        seed_pool_fingerprint=seed_fingerprint,
    )
    assert not (gen_dir / STEERING_SCALE_MARKER_NAME).exists()
    _patch_run_intervention(monkeypatch)
    _write_direction_artifact(cfg, 44.0)

    with pytest.raises(AxBenchBaseError, match="predate the issue #1228"):
        _run(
            cfg,
            targets=_make_targets(target_models, metas, tokenizers),
        )


def test_resume_guard_nonscaled_existing_rows_do_not_block(
    monkeypatch, tmp_run_root, tmp_path, tokenizers, target_models
):
    # Prior rows are for a NON-scaled method (control); adding a diffmean run must
    # NOT be blocked (resume_active is precise: only scaled methods count).
    cfg = _make_cfg(
        tmp_run_root,
        run_tag="guard_ours_ok",
        enabled_methods=["diffmean_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    seed_fingerprint = _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])
    gen_dir = paths.AxBenchBasePaths.from_config(cfg).generations_dir()
    _write_existing_rows(
        gen_dir,
        method_name="control_decoder_only",
        concept_id=0,
        target_model_name="decoder_only",
        mode="positive",
        coefs=[1.0],
        seed_pool_fingerprint=seed_fingerprint,
    )
    assert not (gen_dir / STEERING_SCALE_MARKER_NAME).exists()
    _patch_run_intervention(monkeypatch)
    identity = _write_direction_artifact(cfg, 44.0)

    # No raise: the control rows are not a scaled determinant. The fresh diffmean
    # run stamps the marker.
    _run(
        cfg,
        targets=_make_targets(target_models, metas, tokenizers),
    )
    assert (gen_dir / STEERING_SCALE_MARKER_NAME).read_text() == identity


def test_resume_guard_skipped_for_non_scaled_run(
    monkeypatch, tmp_run_root, tmp_path, tokenizers, target_models
):
    # A run that enables ONLY a non-scaled method (control) has scaling_active
    # False, so the guard is skipped and NO marker is stamped. Teeth for the
    # `if scaling_active:` gate: a mutation
    # that always ran the guard would stamp a spurious marker here.
    cfg = _make_cfg(
        tmp_run_root, run_tag="guard_unscaled",
        enabled_methods=["control_decoder_only"],
    )
    assert cfg.run_tag is not None
    metas = {"decoder_only": _meta(), "encdec": _meta()}
    _commit_test_seed_contract(cfg, monkeypatch, tokenizers["decoder_only"])
    _patch_run_intervention(monkeypatch)
    _run(
        cfg,
        targets=_make_targets(target_models, metas, tokenizers),
    )
    assert not (
        paths.AxBenchBasePaths.from_config(cfg).generations_dir() / STEERING_SCALE_MARKER_NAME
    ).exists()
