"""Integration tests for the axbench_base pipeline dispatcher (Item 13).

What this test file tests
-------------------------
These tests exercise the ``_run`` / ``_dispatch_stage`` /
``_run_single_stage`` **wiring** end-to-end with a real
:class:`RuntimeState` constructed inside the CLI. Every stage
function is **stubbed out** to a no-op that writes the canonical
artifact(s) to disk so downstream stages can still resolve their
dependencies. The four heavy-model loaders are also stubbed via
``stub_heavy_loaders`` so no real Gemma / SAE / target checkpoint is
ever downloaded.

What these tests catch
~~~~~~~~~~~~~~~~~~~~~~
* Wiring bugs in ``_run`` dispatching stages: wrong stage order,
  missing stage branches, bad args to stage functions.
* ``RuntimeState`` construction + teardown: correct ``close()`` on
  successful runs and on crash paths.
* ``validate_env_for_stages`` gating the pipeline on missing
  OpenRouter credentials before any heavy load.
* ``_resolve_stage_list`` correctly expanding ``Stage.ALL`` with /
  without ``run_negative_mode``.
* Auto-generation of ``run_tag`` via ``_generate_run_tag`` and the
  format of the resulting tag.
* The refcount + CPU-offload contract on heavy-model contexts under
  the full dispatcher (see :func:`test_full_pipeline_memory_thrash_observable`).

What these tests do NOT catch
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
* Bugs in the stage function bodies themselves. Stage-body tests
  live in ``tests/axbench_base/integration/test_stage_*.py`` and
  ``tests/axbench_base/unit/test_*.py``. This file does NOT re-verify
  that ``synthesize_positive_mode`` actually synthesizes anything,
  only that ``_dispatch_stage`` calls the stubbed function with the
  right runtime-state accessors.
* Real model weights, real tokenization, real LLM calls.

Scope of stubbing
-----------------
* The four heavy-model loaders (``_load_feature_extraction_pair``,
  ``_load_decoder_only_model``, ``_load_encdec_model``) are replaced
  at module scope via the ``stub_heavy_loaders`` fixture in
  ``tests/axbench_base/conftest.py``. Every ``with state.base_model()``
  body drives the same refcount + CPU-offload contract as production.
* Every top-level stage function on the cli module is monkey-patched
  to a no-op that writes the canonical stage artifact(s) to the
  dep-check-expected location. Pre-existing per-stage smoke coverage
  already verified each stage function on its own; this file focuses
  on the orchestrator glue.
* ``check_stage_dependencies`` is NOT patched — it runs for real
  against the pre-populated artifacts so tests see the real dep-check
  short-circuiting behavior on resume paths.
* the production cached ``requests.Session.post`` is replaced by the
  ``mock_openrouter_wire``
  fixture so any stray OpenRouter call (e.g. from a stage function
  we forgot to stub) fails loudly rather than hitting the network.

Fixture composition
-------------------
Each test pulls the three session-independent fixtures via dependency
injection:

1. ``stub_heavy_loaders`` — CPU-only stubs for base Gemma / base SAE /
   decoder-only / encdec.
2. ``mock_openrouter_wire`` — canned session-level responses + the
   ``OPENROUTER_API_KEY`` env var.
3. ``tiny_feature_statistics_path`` — a minimal safetensors header
   with ``n_sae_features=16384`` so :func:`check_stage_dependencies`
   can resolve the Stage 4 per-concept enumeration.

Runtime budget
--------------
Each test is under 5 seconds on a modern CPU; the full suite below
runs in well under 30 s. All tests are marked ``@pytest.mark.slow``
because they construct a real ``RuntimeState`` and exercise the full
dispatch loop.
"""

from __future__ import annotations

import functools
import json
import re
from pathlib import Path
from typing import Any, Literal
from unittest.mock import MagicMock

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save_file

import scripts.eval.axbench_base_eval as cli
from scripts.eval.axbench_base_eval import _run
from steering.eval.axbench_base import (
    _stage0c_internals as stage0c_internals_mod,
)
from steering.eval.axbench_base import (
    paths,
)
from steering.eval.axbench_base import (
    runtime_state as rs_mod,
)
from steering.eval.axbench_base.artifact_metadata import (
    schema_with_extraction_tokenizer_metadata,
)
from steering.eval.axbench_base.config import AxBenchBaseConfig
from steering.eval.axbench_base.generation import GENERATIONS_SCHEMA
from steering.eval.axbench_base.judge import _JUDGE_SCORES_SCHEMA
from steering.eval.axbench_base.positive_qc_contract import (
    schema_with_positive_qc_contract,
)
from steering.eval.axbench_base.runtime_state import RuntimeState
from steering.eval.axbench_base.synth.concept_yield import (
    ConceptYieldRecord,
    write_concept_yield_report,
    write_filtered_concepts,
)
from steering.eval.axbench_base.synth.negative_mode import NEG_TRAINING_SCHEMA
from steering.eval.axbench_base.synth.positive_mode import TRAINING_SCHEMA
from steering.eval.axbench_base.synth_continuation_contract import (
    schema_with_stage1_continuation_contract,
)
from steering.eval.axbench_base.types import (
    AxBenchBaseError,
    CandidateConcept,
    Concept,
    SeedRow,
    SteeringMode,
)
from tests.axbench_base.fixtures.seed_contract import write_committed_seed_pair

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Dev-tier constants
# ---------------------------------------------------------------------------


_DEV_TIER_N_CONCEPTS: int = 5
_DEV_TIER_N_RANK_STORE: int = 4
_SEED_TOKEN_WIDTH: int = 64


# ---------------------------------------------------------------------------
# Stub artifact writers (one per stage output)
# ---------------------------------------------------------------------------


def _concepts(n: int) -> list[Concept]:
    """Return ``n`` trivial concepts with deterministic fields."""
    return [
        Concept(
            concept_id=i,
            feature_index=100 + i,
            layer=8,
            sae_source="8-gemmascope-res-16k",
            description=f"concept {i} description",
            corpus_count=500,
            description_source="stub",
        )
        for i in range(n)
    ]


class _SeedTokenizer:
    def get_bos_token_id(self) -> int:
        return 1

    def encode(self, text: str) -> list[int]:
        return [ord(char) + 2 for char in text]

    def decode(self, token_ids: list[int]) -> str:
        return "".join(chr(token_id - 2) for token_id in token_ids if token_id != 0)


def _padded_seed_tokens(text: str) -> list[int]:
    tokens = _SeedTokenizer().encode(text)
    if len(tokens) > _SEED_TOKEN_WIDTH:
        raise AssertionError("test seed text exceeds fixed fixture width")
    return [*tokens, *([0] * (_SEED_TOKEN_WIDTH - len(tokens)))]


def _make_seed_row(doc_idx: int) -> SeedRow:
    """Return a minimal :class:`SeedRow` with non-empty continuation."""
    text = f"prefix text {doc_idx}"
    return SeedRow(
        shard_idx=0,
        doc_idx=doc_idx,
        source_nanochat_token_count=2 * _SEED_TOKEN_WIDTH,
        prefix_source_token_start=0,
        prefix_source_token_end=_SEED_TOKEN_WIDTH,
        prefix_text=text,
        prefix_content_nanochat_tokens=_padded_seed_tokens(text),
        sae_feature_ids=[],
        sae_feature_mags=[],
        continuation_text=f"continuation {doc_idx}",
        continuation_content_nanochat_tokens=_padded_seed_tokens(
            f"continuation {doc_idx}"
        ),
    )


def _write_concepts_json(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
) -> Path:
    """Write concepts.json at the canonical path for ``run_tag``."""
    out_path = output_paths.concepts_json()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "concept_id": c.concept_id,
            "feature_index": c.feature_index,
            "layer": c.layer,
            "sae_source": c.sae_source,
            "description": c.description,
            "corpus_count": c.corpus_count,
            "description_source": c.description_source,
        }
        for c in concepts
    ]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _write_candidate_concepts_json(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
) -> Path:
    """Write new-schema candidate_concepts.json at the canonical path."""
    out_path = output_paths.candidate_concepts_json()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {
            "feature_index": c.feature_index,
            "layer": c.layer,
            "sae_source": c.sae_source,
            "description": c.description,
            "corpus_count": c.corpus_count,
            "description_source": c.description_source,
        }
        for c in concepts
    ]
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _write_rejected_concepts_json(
    output_paths: paths.AxBenchBasePaths,
) -> Path:
    out_path = output_paths.rejected_concepts_json()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text("[]", encoding="utf-8")
    return out_path


def _write_seed_pools(cfg: AxBenchBaseConfig) -> tuple[Path, Path]:
    """Write a committed schema-v2 train/eval seed pair."""
    assert cfg.run_tag is not None
    train_rows = [_make_seed_row(i) for i in range(3)]
    eval_rows = [_make_seed_row(10 + i) for i in range(2)]
    return write_committed_seed_pair(
        cfg,
        tokenizer=_SeedTokenizer(),
        train_rows=train_rows,
        eval_rows=eval_rows,
    )


def _write_negative_prefixes(
    cfg: AxBenchBaseConfig,
    concepts: list[Concept],
) -> Path:
    """Write negative_mode_prefixes.parquet with one verified prefix
    per concept."""
    assert cfg.run_tag is not None
    n_passes = max(cfg.min_consistency_passes, 1)
    percentile = min(
        99.5,
        cfg.verification_percentile_threshold * 100.0 + 1.0,
    )
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        for row_idx in range(cfg.n_verified_prefixes_per_concept_target):
            rows.append(
                {
                    "concept_id": concept.concept_id,
                    "prefix_text": (
                        f"concept-{concept.concept_id}-{row_idx}:".ljust(
                            cfg.min_prefix_chars,
                            "x",
                        )
                    ),
                    "prefix_source_shard": 0,
                    "prefix_source_doc": (
                        concept.concept_id
                        * cfg.n_verified_prefixes_per_concept_target
                        + row_idx
                    ),
                    "prefix_activating_seg_idx": row_idx,
                    "verification_continuations": [
                        "verified continuation "
                        f"{concept.concept_id}-{row_idx}-{pass_idx}"
                        for pass_idx in range(n_passes)
                    ],
                    "verification_seeds": [
                        row_idx * n_passes + pass_idx
                        for pass_idx in range(n_passes)
                    ],
                    "verification_percentiles": [percentile] * n_passes,
                }
            )
    candidate_concepts = [
        CandidateConcept(
            feature_index=concept.feature_index,
            layer=concept.layer,
            sae_source=concept.sae_source,
            description=concept.description,
            corpus_count=concept.corpus_count,
            description_source=concept.description_source,
        )
        for concept in concepts
    ]
    pool_identity = stage0c_internals_mod.candidate_pool_identity(
        candidate_concepts
    )
    schema = (
        stage0c_internals_mod._schema_with_completed_negative_prefixes_metadata(
            cfg,
            candidate_pool_identity=pool_identity,
        )
    )
    schema = stage0c_internals_mod._schema_with_completed_artifact_set_identities(
        schema,
        concepts=concepts,
        rejected_concepts=[],
        concept_examples_qc_report=None,
        stage0c_fingerprint=stage0c_internals_mod._fingerprint_with_pool(
            cfg,
            candidate_pool_identity=pool_identity,
        ),
    )
    out_path = paths.AxBenchBasePaths.from_config(cfg).negative_mode_prefixes()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows, schema=schema), out_path)
    return out_path


def _pos_row(
    concept_id: int,
    content_class: str,
    output_text: str,
) -> dict[str, Any]:
    """Return a row matching the :data:`TRAINING_SCHEMA` positive schema."""
    return {
        "concept_id": concept_id,
        "category": "positive" if content_class == "concept_rich" else "negative",
        "content_class": content_class,
        "prefix_text": f"prefix for c{concept_id}",
        "output_text": output_text,
        "output_concept": f"concept {concept_id}",
        "n_prefix_nanochat_tokens": 6,
        "gen_source": "stub",
        "gen_seed": 0,
        "qc_feature_id": 100 + concept_id if concept_id >= 0 else -1,
        "qc_seed_percentile": 0.1,
        "qc_output_percentile": 0.8,
        "qc_delta": 0.7,
        "qc_attempts": 1,
        "qc_accepted": True,
        "qc_zero_activation_verified": content_class == "concept_avoiding",
    }


def _neg_row(
    concept_id: int,
    content_class: str,
    output_text: str,
) -> dict[str, Any]:
    """Return a row matching the :data:`NEG_TRAINING_SCHEMA` schema."""
    base = _pos_row(concept_id, content_class, output_text)
    base.update(
        {
            "prefix_source_shard": 0,
            "prefix_source_doc": concept_id,
            "prefix_source_seg_idx": 0,
            "qc_zero_activation_verified": content_class == "concept_avoiding",
        },
    )
    return base


def _write_synth_training_positive(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
) -> Path:
    """Write synth_training_positive.parquet (stage 1b product).

    Each accepted ``concept_rich`` row is immediately followed by a paired
    ``concept_avoiding`` row of the same prefix (the class-margin refactor
    removed the Stage 1a global-negative pool, so there are no
    ``concept_id == -1`` rows).
    """
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        for k in range(3):
            rows.append(
                _pos_row(
                    concept.concept_id,
                    "concept_rich",
                    f"positive sample c{concept.concept_id}_{k}",
                ),
            )
            rows.append(
                _pos_row(
                    concept.concept_id,
                    "concept_avoiding",
                    f"avoiding sample c{concept.concept_id}_{k}",
                ),
            )
    out_path = output_paths.synth_training_positive()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=schema_with_stage1_continuation_contract(
                schema_with_positive_qc_contract(
                    schema_with_extraction_tokenizer_metadata(
                        TRAINING_SCHEMA,
                        "google/gemma-2-2b",
                    ),
                    accept_threshold=0.3,
                ),
                role="stage1b_positive",
            ),
        ),
        out_path,
    )
    return out_path


def _write_synth_training_negative(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
) -> Path:
    """Write synth_training_negative.parquet (stage 1c product)."""
    rows: list[dict[str, Any]] = []
    for concept in concepts:
        for k in range(2):
            rows.append(
                _neg_row(
                    concept.concept_id,
                    "concept_rich",
                    f"concept rich c{concept.concept_id}_{k}",
                ),
            )
            rows.append(
                _neg_row(
                    concept.concept_id,
                    "concept_avoiding",
                    f"concept avoiding c{concept.concept_id}_{k}",
                ),
            )
    out_path = output_paths.synth_training_negative()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(
        pa.Table.from_pylist(
            rows,
            schema=schema_with_stage1_continuation_contract(
                schema_with_extraction_tokenizer_metadata(
                    NEG_TRAINING_SCHEMA,
                    "google/gemma-2-2b",
                ),
                role="stage1c_negative",
            ),
        ),
        out_path,
    )
    return out_path


def _write_successful_stage1_handoff(
    cfg: AxBenchBaseConfig,
    *,
    stage: Literal["stage1b", "stage1c"],
    concepts: list[Concept],
) -> None:
    """Write an all-surviving Stage 1 report/list pair for dispatcher stubs."""

    output_paths = paths.AxBenchBasePaths.from_config(cfg)
    records = [
        ConceptYieldRecord(
            concept_id=concept.concept_id,
            feature_index=concept.feature_index,
            attempted=2,
            accepted=2,
            dropped=0,
            drop_fraction=0.0,
            excluded=False,
            exclusion_reason=None,
            reason_counts={},
            emitted_rows=4,
        )
        for concept in concepts
    ]
    if stage == "stage1b":
        report_path = output_paths.stage1b_concept_yield_report()
        filtered_path = output_paths.stage1b_concepts()
    elif stage == "stage1c":
        report_path = output_paths.stage1c_concept_yield_report()
        filtered_path = output_paths.stage1c_concepts()
    else:
        raise AssertionError(f"unexpected Stage 1 handoff {stage!r}")
    survivors = write_concept_yield_report(
        cfg,
        stage=stage,
        records=records,
        path=report_path,
    )
    write_filtered_concepts(concepts, survivors, filtered_path)


def _write_concept_features(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    *,
    n_rank_store: int,
    write_neg: bool,
) -> tuple[Path, Path | None]:
    """Write concept_features{,_neg}.safetensors (stage 2 products).

    When ``write_neg`` is :data:`False` (negative-mode-off path) only
    the positive file is emitted — matching what the real stage 2 does
    when ``cfg.run_negative_mode=False``.
    """
    n_concepts = max(concept.concept_id for concept in concepts) + 1
    feature_ids = t.full(
        (n_concepts, n_rank_store),
        fill_value=-1,
        dtype=t.int32,
    )
    percentile_deltas = t.full(
        (n_concepts, n_rank_store),
        fill_value=float("nan"),
        dtype=t.float32,
    )
    canonical_feature_id = t.full((n_concepts,), -1, dtype=t.int32)
    canonical_rank = t.full(
        (n_concepts,),
        n_rank_store,
        dtype=t.int32,
    )
    for concept in concepts:
        row = concept.concept_id
        feature_ids[row, 0] = concept.feature_index
        percentile_deltas[row, 0] = 0.0
        canonical_feature_id[row] = concept.feature_index
        canonical_rank[row] = 0

    tensors: dict[str, t.Tensor] = {
        "feature_ids": feature_ids,
        "percentile_deltas": percentile_deltas,
        "canonical_feature_id": canonical_feature_id,
        "canonical_rank": canonical_rank,
    }
    metadata = {"mode": "canonical"}

    pos_path = output_paths.concept_features()
    pos_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, str(pos_path), metadata=metadata)

    neg_path: Path | None = None
    if write_neg:
        neg_path = output_paths.concept_features_neg()
        save_file(tensors, str(neg_path), metadata=metadata)
    return pos_path, neg_path


def _write_direction_artifacts(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    *,
    run_negative_mode: bool,
) -> list[Path]:
    """Write stub DiffMean + LinearProbe direction files for every
    (target_model, mode, concept) triple expected by Stage 4 dep check.
    """
    hidden = 8
    written: list[Path] = []
    modes: list[SteeringMode] = [SteeringMode.POSITIVE]
    if run_negative_mode:
        modes.append(SteeringMode.NEGATIVE)
    for target in ("decoder_only", "encdec"):
        for mode in modes:
            for concept in concepts:
                direction = t.zeros(hidden, dtype=t.float32)
                dm_path = output_paths.diffmean_direction(
                    target_model=target,
                    mode=mode,
                    concept_id=concept.concept_id,
                )
                dm_path.parent.mkdir(parents=True, exist_ok=True)
                save_file({"direction": direction}, str(dm_path))
                written.append(dm_path)

                lp_path = output_paths.linear_probe_direction(
                    target_model=target,
                    mode=mode,
                    concept_id=concept.concept_id,
                )
                lp_path.parent.mkdir(parents=True, exist_ok=True)
                save_file(
                    {
                        "direction": direction.clone(),
                        "coef": direction.clone().unsqueeze(0),
                        "intercept": t.zeros(1, dtype=t.float32),
                    },
                    str(lp_path),
                )
                written.append(lp_path)
    return written


def _write_lora_adapters(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    *,
    run_negative_mode: bool,
) -> list[Path]:
    """Write stub LoRA adapter .pt files (decoder_only only, per mode
    × concept)."""
    written: list[Path] = []
    modes: list[SteeringMode] = [SteeringMode.POSITIVE]
    if run_negative_mode:
        modes.append(SteeringMode.NEGATIVE)
    for mode in modes:
        for concept in concepts:
            out_path = output_paths.lora_adapter(
                mode=mode,
                concept_id=concept.concept_id,
            )
            out_path.parent.mkdir(parents=True, exist_ok=True)
            t.save({"layer_0_stub": t.zeros(1)}, out_path)
            written.append(out_path)
    return written


def _write_prompting_descriptions(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    *,
    run_negative_mode: bool,
) -> list[Path]:
    """Write prompting baseline description JSON files."""
    written: list[Path] = []
    modes: list[SteeringMode] = [SteeringMode.POSITIVE]
    if run_negative_mode:
        modes.append(SteeringMode.NEGATIVE)
    for mode in modes:
        p = output_paths.prompting_descriptions(mode)
        p.parent.mkdir(parents=True, exist_ok=True)
        mapping = {str(c.concept_id): c.description for c in concepts}
        p.write_text(
            json.dumps(mapping, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        written.append(p)
    return written


def _write_generations_dir(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    *,
    run_negative_mode: bool,
) -> Path:
    """Write generations/batch_000000.parquet with a row per
    (method, concept, mode) tuple — enough for Stage 5 / 6 to operate on.
    """
    method_names = ["diffmean_decoder_only", "ours_extract_and_add_encdec"]
    modes = ["positive"]
    if run_negative_mode:
        modes.append("negative")
    rows: list[dict[str, Any]] = []
    for method in method_names:
        for concept in concepts:
            for mode in modes:
                rows.append(
                    {
                        "generation_key": (
                            f"{method}|c{concept.concept_id}|s0|kNone|decoder_only"
                        ),
                        "concept_id": concept.concept_id,
                        "method_name": method,
                        "mode": mode,
                        "eval_seed": 0,
                        "coefficient": "None",
                        "generation_text": f"gen text c{concept.concept_id}",
                        "target_model": "decoder_only",
                        "prefix_text": f"prefix c{concept.concept_id}",
                        "token_count": 4,
                        "stop_reason": "bos_token",
                    },
                )
    out_dir = output_paths.generations_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_000000.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=GENERATIONS_SCHEMA),
        out_path,
    )
    return out_dir


def _write_judge_scores_dir(
    output_paths: paths.AxBenchBasePaths,
    concepts: list[Concept],
    *,
    run_negative_mode: bool,
) -> Path:
    """Write judge_scores/batch_000000.parquet mirroring the generations."""
    method_names = ["diffmean_decoder_only", "ours_extract_and_add_encdec"]
    modes = ["positive"]
    if run_negative_mode:
        modes.append("negative")
    rows: list[dict[str, Any]] = []
    for method in method_names:
        for concept in concepts:
            for mode in modes:
                rows.append(
                    {
                        "generation_key": (
                            f"{method}|c{concept.concept_id}|s0|kNone|decoder_only"
                        ),
                        "concept_id": concept.concept_id,
                        "method_name": method,
                        "mode": mode,
                        "eval_seed": 0,
                        "coefficient": "None",
                        "concept_score": 1,
                        "concept_attempts": 1,
                        "concept_invalid": False,
                        "fluency_score": 2,
                        "fluency_attempts": 1,
                        "fluency_invalid": False,
                        "judge_model": "mock-judge",
                    },
                )
    out_dir = output_paths.judge_scores_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "batch_000000.parquet"
    pq.write_table(
        pa.Table.from_pylist(rows, schema=_JUDGE_SCORES_SCHEMA),
        out_path,
    )
    return out_dir


# ---------------------------------------------------------------------------
# Stage stubs
#
# Each stub writes ONLY the artifacts the corresponding real stage
# would produce, and accepts the same call signature via ``**kwargs``.
# A shared ``stub_all_stages`` fixture installs every stub at
# module-level (on the cli module) so ``_run`` reaches the real
# dispatch loop and the stubs run as the "stage body". Production
# heavy-model context managers still fire (via ``stub_heavy_loaders``)
# so refcount / offload contracts are exercised end-to-end.
# ---------------------------------------------------------------------------


def _install_stage_stubs(
    monkeypatch: pytest.MonkeyPatch,
    *,
    concepts: list[Concept],
) -> dict[str, int]:
    """Replace every stage function on the cli module with a stub that
    writes the canonical artifacts to disk. Returns a shared counter
    dict tests can inspect to see which stages actually fired.
    """
    counts: dict[str, int] = dict.fromkeys(
        (
            "prepare_concept_list",
            "prepare_seed_pools",
            "prepare_negative_prefixes",
            "synthesize_positive_mode",
            "synthesize_negative_mode",
            "concept_feature_mapping_canonical",
            "concept_feature_mapping_delta_ranked",
            "train_diffmean_all",
            "train_linear_probe_all",
            "write_prompting_descriptions",
            "train_lora_all",
            "enumerate_stage4_contexts",
            "generate_all",
            "judge_all",
            "aggregate",
        ),
        0,
    )

    def fake_prepare_concept_list(cfg: AxBenchBaseConfig) -> Path:
        counts["prepare_concept_list"] += 1
        assert cfg.run_tag is not None
        return _write_candidate_concepts_json(
            paths.AxBenchBasePaths.from_config(cfg),
            concepts,
        )

    def fake_prepare_seed_pools(cfg: AxBenchBaseConfig, **kwargs: Any) -> tuple[Path, Path]:
        del kwargs
        counts["prepare_seed_pools"] += 1
        assert cfg.run_tag is not None
        return _write_seed_pools(cfg)

    def fake_prepare_negative_prefixes(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        counts["prepare_negative_prefixes"] += 1
        assert cfg.run_tag is not None
        output_paths = paths.AxBenchBasePaths.from_config(cfg)
        _write_concepts_json(output_paths, concepts)
        _write_rejected_concepts_json(output_paths)
        return _write_negative_prefixes(cfg, concepts)

    def fake_synthesize_positive_mode(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        counts["synthesize_positive_mode"] += 1
        assert cfg.run_tag is not None
        output = _write_synth_training_positive(
            paths.AxBenchBasePaths.from_config(cfg), concepts,
        )
        _write_successful_stage1_handoff(
            cfg,
            stage="stage1b",
            concepts=concepts,
        )
        return output

    def fake_synthesize_negative_mode(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        counts["synthesize_negative_mode"] += 1
        assert cfg.run_tag is not None
        output = _write_synth_training_negative(
            paths.AxBenchBasePaths.from_config(cfg), concepts,
        )
        _write_successful_stage1_handoff(
            cfg,
            stage="stage1c",
            concepts=concepts,
        )
        return output

    def fake_canonical(
        cfg: AxBenchBaseConfig,
        *,
        concepts: list[Concept],
    ) -> tuple[Path, Path | None]:
        del concepts
        counts["concept_feature_mapping_canonical"] += 1
        assert cfg.run_tag is not None
        stub_concepts = _concepts(_DEV_TIER_N_CONCEPTS)
        pos, neg = _write_concept_features(
            paths.AxBenchBasePaths.from_config(cfg),
            stub_concepts,
            n_rank_store=_DEV_TIER_N_RANK_STORE,
            write_neg=cfg.run_negative_mode,
        )
        return pos, neg

    def fake_delta_ranked(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> tuple[Path, Path | None]:
        del kwargs
        counts["concept_feature_mapping_delta_ranked"] += 1
        assert cfg.run_tag is not None
        stub_concepts = _concepts(_DEV_TIER_N_CONCEPTS)
        pos, neg = _write_concept_features(
            paths.AxBenchBasePaths.from_config(cfg),
            stub_concepts,
            n_rank_store=_DEV_TIER_N_RANK_STORE,
            write_neg=cfg.run_negative_mode,
        )
        return pos, neg

    def fake_train_diffmean_all(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> list[Path]:
        del kwargs
        counts["train_diffmean_all"] += 1
        assert cfg.run_tag is not None
        # Each (target, mode) call adds one set of per-concept files;
        # _write_direction_artifacts produces the full matrix in one
        # call, which is idempotent (overwrites on repeat) so all four
        # trainer calls across (decoder_only, encdec) x (pos, neg)
        # collectively materialize the whole matrix exactly once.
        _write_direction_artifacts(
            paths.AxBenchBasePaths.from_config(cfg),
            _concepts(_DEV_TIER_N_CONCEPTS),
            run_negative_mode=cfg.run_negative_mode,
        )
        return []

    def fake_train_linear_probe_all(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> list[Path]:
        del kwargs
        counts["train_linear_probe_all"] += 1
        assert cfg.run_tag is not None
        _write_direction_artifacts(
            paths.AxBenchBasePaths.from_config(cfg),
            _concepts(_DEV_TIER_N_CONCEPTS),
            run_negative_mode=cfg.run_negative_mode,
        )
        return []

    def fake_write_prompting_descriptions(
        cfg: AxBenchBaseConfig,
        *,
        concepts: list[Concept],
    ) -> list[Path]:
        counts["write_prompting_descriptions"] += 1
        assert cfg.run_tag is not None
        return _write_prompting_descriptions(
            paths.AxBenchBasePaths.from_config(cfg),
            concepts,
            run_negative_mode=cfg.run_negative_mode,
        )

    def fake_train_lora_all(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> list[Path]:
        del kwargs
        counts["train_lora_all"] += 1
        assert cfg.run_tag is not None
        _write_lora_adapters(
            paths.AxBenchBasePaths.from_config(cfg),
            _concepts(_DEV_TIER_N_CONCEPTS),
            run_negative_mode=cfg.run_negative_mode,
        )
        return []

    def fake_enumerate_stage4_contexts(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        counts["enumerate_stage4_contexts"] += 1
        assert cfg.run_tag is not None
        out_path = paths.AxBenchBasePaths.from_config(cfg).stage4_contexts_manifest()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}\n", encoding="utf-8")
        return out_path

    def fake_generate_all(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        counts["generate_all"] += 1
        assert cfg.run_tag is not None
        return _write_generations_dir(
            paths.AxBenchBasePaths.from_config(cfg),
            _concepts(_DEV_TIER_N_CONCEPTS),
            run_negative_mode=cfg.run_negative_mode,
        )

    def fake_build_stage4_encdec_context_provider(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> object:
        # Stage 4's encdec context provider is cache-backed: production
        # requires a real ``conditioning.context_cache_dir`` and builds
        # a cache-matched provider. This wiring test stubs ``generate_all``
        # (which is the only consumer of the provider), so return a
        # sentinel the stubbed generation loop never actually uses.
        del cfg, kwargs
        return object()

    def fake_judge_all(
        cfg: AxBenchBaseConfig,
        **kwargs: Any,
    ) -> Path:
        del kwargs
        counts["judge_all"] += 1
        assert cfg.run_tag is not None
        return _write_judge_scores_dir(
            paths.AxBenchBasePaths.from_config(cfg),
            _concepts(_DEV_TIER_N_CONCEPTS),
            run_negative_mode=cfg.run_negative_mode,
        )

    # Stage 6 uses the REAL aggregate so the test can assert the
    # leaderboard / report.yaml shapes produced by production.
    monkeypatch.setattr(cli, "prepare_concept_list", fake_prepare_concept_list)
    monkeypatch.setattr(cli, "prepare_seed_pools", fake_prepare_seed_pools)
    monkeypatch.setattr(
        cli,
        "prepare_negative_prefixes",
        fake_prepare_negative_prefixes,
    )
    monkeypatch.setattr(
        cli,
        "synthesize_positive_mode",
        fake_synthesize_positive_mode,
    )
    monkeypatch.setattr(
        cli,
        "synthesize_negative_mode",
        fake_synthesize_negative_mode,
    )
    monkeypatch.setattr(
        cli,
        "concept_feature_mapping_canonical",
        fake_canonical,
    )
    monkeypatch.setattr(
        cli,
        "concept_feature_mapping_delta_ranked",
        fake_delta_ranked,
    )
    monkeypatch.setattr(cli, "train_diffmean_all", fake_train_diffmean_all)
    monkeypatch.setattr(
        cli,
        "train_linear_probe_all",
        fake_train_linear_probe_all,
    )
    monkeypatch.setattr(
        cli,
        "write_prompting_descriptions",
        fake_write_prompting_descriptions,
    )
    monkeypatch.setattr(cli, "train_lora_all", fake_train_lora_all)
    monkeypatch.setattr(
        cli, "enumerate_stage4_contexts", fake_enumerate_stage4_contexts
    )
    monkeypatch.setattr(cli, "generate_all", fake_generate_all)
    monkeypatch.setattr(
        cli,
        "_build_stage4_encdec_context_provider",
        fake_build_stage4_encdec_context_provider,
    )
    monkeypatch.setattr(cli, "judge_all", fake_judge_all)
    return counts


# ---------------------------------------------------------------------------
# Config builder
# ---------------------------------------------------------------------------


def _build_dev_cfg(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    *,
    run_tag: str | None,
    run_negative_mode: bool,
    stage: str = "all",
) -> DictConfig:
    """Build a fresh :class:`DictConfig` for the dev-tier pipeline.

    Returning a plain :class:`DictConfig` so callers can pass it
    straight into :func:`_run`. The config points at the shared tiny
    feature-statistics safetensors, uses CPU device, and disables
    ``enable_concept_examples_qc`` so Stage 0c never tries to
    construct a real OpenRouter QC judge.
    """
    base: dict[str, Any] = {
        "feature_dir": (
            "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base"
        ),
        "feature_statistics_path": str(tiny_feature_statistics_path),
        "decoder_only": {
            "source": "base",
            "model_tag": "d0",
            "step": 0,
        },
        "encdec": {
            "source": "encdec",
            "model_tag": "d0",
            "step": 0,
        },
        "device": "cpu",
        "output_root": str(tmp_run_root / "eval_out"),
        "run_tag": run_tag,
        "stage": stage,
        "run_negative_mode": run_negative_mode,
        "enable_concept_examples_qc": False,
        "n_rank_store": _DEV_TIER_N_RANK_STORE,
        "max_concepts": _DEV_TIER_N_CONCEPTS,
        "validation_shards": [0],
        "seed_prefix_max_nanochat_tokens": _SEED_TOKEN_WIDTH,
        "seed_continuation_nanochat_tokens": _SEED_TOKEN_WIDTH,
    }
    return OmegaConf.create(base)


# ---------------------------------------------------------------------------
# Test 1: full stage=ALL happy path
# ---------------------------------------------------------------------------


def test_full_pipeline_stage_all_with_tiny_models(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Full ``stage="all"`` run with real ``_run`` orchestration.

    Asserts every stage fires in order, every canonical artifact lands
    on disk, and Stage 6 ``aggregate`` produces a non-empty
    leaderboard + report.yaml + aggregation_report.yaml.
    """
    del stub_heavy_loaders, mock_openrouter_wire  # pulled in for side effects
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    counts = _install_stage_stubs(monkeypatch, concepts=concepts)

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag="test-all-happy",
        run_negative_mode=True,
    )
    _run(cfg)

    run_tag = "test-all-happy"
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", run_tag)
    legacy_run = tmp_run_root / "eval/axbench_base" / run_tag
    # Every data-producing stage fired exactly once.
    assert counts["prepare_concept_list"] == 1
    assert counts["prepare_seed_pools"] == 1
    assert counts["prepare_negative_prefixes"] == 1
    assert counts["synthesize_positive_mode"] == 1
    assert counts["synthesize_negative_mode"] == 1
    assert counts["concept_feature_mapping_canonical"] == 1
    assert counts["concept_feature_mapping_delta_ranked"] == 0
    # DiffMean + LinearProbe fire once per (target, mode) = 2 * 2 = 4
    assert counts["train_diffmean_all"] == 4
    assert counts["train_linear_probe_all"] == 4
    assert counts["write_prompting_descriptions"] == 1
    # LoRA fires once per mode (decoder-only only), pos + neg = 2
    assert counts["train_lora_all"] == 2
    assert counts["enumerate_stage4_contexts"] == 1
    assert counts["generate_all"] == 1
    assert counts["judge_all"] == 1

    # Every artifact landed.
    assert output_paths.candidate_concepts_json().exists()
    candidate_rows = json.loads(output_paths.candidate_concepts_json().read_text())
    assert candidate_rows
    assert all("concept_id" not in row for row in candidate_rows)
    assert output_paths.concepts_json().exists()
    assert output_paths.rejected_concepts_json().exists()
    assert output_paths.seeds_train().exists()
    assert output_paths.seeds_eval().exists()
    assert output_paths.negative_mode_prefixes().exists()
    assert output_paths.synth_training_positive().exists()
    assert output_paths.synth_training_negative().exists()
    assert output_paths.concept_features().exists()
    assert output_paths.concept_features_neg().exists()
    assert output_paths.generations_dir().is_dir()
    assert output_paths.judge_scores_dir().is_dir()
    assert not legacy_run.exists()

    # Stage 6 (real aggregate) products.
    report_path = output_paths.report_yaml()
    assert report_path.exists()
    leaderboard_path = output_paths.leaderboard_csv()
    assert leaderboard_path.exists()
    aggregation_path = output_paths.aggregation_report()
    assert aggregation_path.exists()

    leaderboard_lines = leaderboard_path.read_text(
        encoding="utf-8",
    ).splitlines()
    # Header + at least one data row.
    assert len(leaderboard_lines) >= 2
    assert leaderboard_lines[0].startswith("method_name")


# ---------------------------------------------------------------------------
# Test 2: resume after stage 2 crash
# ---------------------------------------------------------------------------


def test_full_pipeline_resume_after_stage_2_crash(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Item 10: verify resume semantics end-to-end.

    The test drives the dispatcher twice with the same ``run_tag``.
    After the first run every artifact for stages 0a-2 + 3a-6 is
    present. On the second run the per-stage stubs run again
    (``_dispatch_stage`` iterates all stages unconditionally and most
    dependency checks validate inputs, not existence of outputs), but
    Stage 0c has an explicit completed-artifact resume short-circuit.
    The test asserts that:

    1. Idempotent stage stubs fire on both runs (exact per-stage counts
       pinned), while Stage 0c reuses its validated completed artifact.
    2. Stage 3e context enumeration is part of both complete runs.
    3. The final artifacts remain intact after the second run
       (aggregate Stage 6 writes a fresh report + leaderboard).

    This pins the observable resume-safety contract: a second run
    with the same ``run_tag`` works, stages still dispatch, every
    dep check passes, and Stage 6 can re-aggregate without losing
    prior artifacts.
    """
    del stub_heavy_loaders, mock_openrouter_wire
    run_tag = "test-resume"
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)

    # First run: drive every stage from scratch.
    counts = _install_stage_stubs(monkeypatch, concepts=concepts)
    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=run_tag,
        run_negative_mode=True,
    )
    _run(cfg)
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", run_tag)

    # Snapshot first-run counts.
    first_run_counts = dict(counts)
    assert first_run_counts["prepare_concept_list"] == 1
    assert first_run_counts["prepare_seed_pools"] == 1
    assert first_run_counts["prepare_negative_prefixes"] == 1
    assert first_run_counts["concept_feature_mapping_canonical"] == 1
    assert first_run_counts["train_diffmean_all"] == 4
    assert first_run_counts["enumerate_stage4_contexts"] == 1
    assert first_run_counts["generate_all"] == 1
    assert first_run_counts["judge_all"] == 1

    # Verify every canonical artifact landed from the first run.
    assert output_paths.candidate_concepts_json().exists()
    assert output_paths.concepts_json().exists()
    assert output_paths.rejected_concepts_json().exists()
    assert output_paths.seeds_train().exists()
    assert output_paths.seeds_eval().exists()
    assert output_paths.negative_mode_prefixes().exists()
    assert output_paths.synth_training_positive().exists()
    assert output_paths.synth_training_negative().exists()
    assert output_paths.concept_features().exists()
    assert output_paths.concept_features_neg().exists()
    assert output_paths.report_yaml().exists()
    assert output_paths.leaderboard_csv().exists()

    # Second run with the same run_tag: every stage re-dispatches and
    # every dependency check passes. Stage 0c validates and reuses its
    # completed transaction; the idempotent stages execute again.
    _run(cfg)

    assert counts["prepare_concept_list"] == 2
    assert counts["prepare_seed_pools"] == 2
    assert counts["prepare_negative_prefixes"] == 1
    assert counts["concept_feature_mapping_canonical"] == 2
    assert counts["train_diffmean_all"] == 8
    assert counts["train_linear_probe_all"] == 8
    assert counts["train_lora_all"] == 4
    assert counts["enumerate_stage4_contexts"] == 2
    assert counts["generate_all"] == 2
    assert counts["judge_all"] == 2

    # Final aggregate artifacts still present after the second run.
    assert output_paths.report_yaml().exists()
    assert output_paths.leaderboard_csv().exists()


# ---------------------------------------------------------------------------
# Test 3: neg_mode=False skips 1c and neg_mode methods
# ---------------------------------------------------------------------------


def test_full_pipeline_neg_mode_off_skips_1c_and_neg_methods(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With ``run_negative_mode=False``:

    - Stage 1c (``synthesize_negative_mode``) is dropped from the
      expansion and its stub is never called.
    - Per-concept neg-mode artifacts are never required by the dep
      check for Stages 3a/3b/3d or Stage 4.
    - The final leaderboard rows carry only ``mode="positive"``.
    """
    del stub_heavy_loaders, mock_openrouter_wire
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    counts = _install_stage_stubs(monkeypatch, concepts=concepts)

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag="test-neg-off",
        run_negative_mode=False,
    )
    _run(cfg)

    run_tag = "test-neg-off"
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", run_tag)
    # 1c never ran.
    assert counts["synthesize_negative_mode"] == 0
    # 3a + 3b ran only for positive mode: 2 targets x 1 mode = 2
    assert counts["train_diffmean_all"] == 2
    assert counts["train_linear_probe_all"] == 2
    # 3d ran only positive: 1 target x 1 mode = 1
    assert counts["train_lora_all"] == 1
    assert counts["enumerate_stage4_contexts"] == 1
    # concept_features_neg was never written.
    assert not output_paths.concept_features_neg().exists()
    # Leaderboard has only positive rows.
    leaderboard_lines = (
        output_paths.leaderboard_csv()
        .read_text(
            encoding="utf-8",
        )
        .splitlines()
    )
    assert len(leaderboard_lines) >= 2
    data_rows = leaderboard_lines[1:]
    for row in data_rows:
        parts = row.split(",")
        assert parts[1] == "positive", f"Non-positive row in leaderboard: {row}"


# ---------------------------------------------------------------------------
# Test 4: stage=6 only, with no model loads
# ---------------------------------------------------------------------------


def test_full_pipeline_stage_6_only_no_model_load(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-populate through Stage 5, run ``stage="6"`` only, patch the
    heavy-model loaders to raise-if-called, verify they aren't invoked.

    Stage 6 is pure disk IO — it should never touch base Gemma, base
    SAE, or the target models.
    """
    del mock_openrouter_wire
    run_tag = "test-stage-6"
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", run_tag)

    # Pre-populate every artifact Stage 6's real aggregate reads.
    _write_concepts_json(output_paths, concepts)
    _write_generations_dir(output_paths, concepts, run_negative_mode=True)
    _write_judge_scores_dir(output_paths, concepts, run_negative_mode=True)

    def _raise(cfg: Any) -> Any:
        del cfg
        raise AssertionError("loader should not be called for stage=6")

    monkeypatch.setattr(rs_mod, "_load_feature_extraction_pair", _raise)
    # _load_feature_extraction_pair already patched above
    monkeypatch.setattr(rs_mod, "_load_decoder_only_model", _raise)
    monkeypatch.setattr(rs_mod, "_load_encdec_model", _raise)
    monkeypatch.setattr(rs_mod.t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rs_mod.t.cuda, "empty_cache", MagicMock())

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=run_tag,
        run_negative_mode=True,
        stage="6",
    )
    _run(cfg)

    assert output_paths.report_yaml().exists()
    assert output_paths.leaderboard_csv().exists()
    assert output_paths.aggregation_report().exists()


# ---------------------------------------------------------------------------
# Test 5: double run: second is all no-ops
# ---------------------------------------------------------------------------


def test_full_pipeline_double_run_second_is_noop(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run the full ``stage="all"`` pipeline twice in a row. Both runs
    succeed and all artifacts still exist at the end. This pins the
    contract that re-running a completed pipeline is safe (stages
    overwrite their artifacts idempotently, aggregation rewrites
    report.yaml).
    """
    del stub_heavy_loaders, mock_openrouter_wire
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    counts = _install_stage_stubs(monkeypatch, concepts=concepts)

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag="test-double",
        run_negative_mode=True,
    )
    _run(cfg)
    _run(cfg)

    run_tag = "test-double"
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", run_tag)
    # Each stage fired twice.
    assert counts["prepare_concept_list"] == 2
    assert counts["prepare_negative_prefixes"] == 1
    assert counts["enumerate_stage4_contexts"] == 2
    assert counts["generate_all"] == 2
    assert counts["judge_all"] == 2
    # Final artifacts still exist.
    assert output_paths.report_yaml().exists()
    assert output_paths.leaderboard_csv().exists()


# ---------------------------------------------------------------------------
# Test 6: missing OPENROUTER_API_KEY fails fast before model load
# ---------------------------------------------------------------------------


def test_full_pipeline_missing_openrouter_fails_fast_before_model_load(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unset ``OPENROUTER_API_KEY``, patch every loader to raise if
    called, run ``stage="all"`` — the env check must raise BEFORE any
    loader fires.
    """
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)

    def _raise(cfg: Any) -> Any:
        del cfg
        raise AssertionError("loader should not be called")

    monkeypatch.setattr(rs_mod, "_load_feature_extraction_pair", _raise)
    # _load_feature_extraction_pair already patched above
    monkeypatch.setattr(rs_mod, "_load_decoder_only_model", _raise)
    monkeypatch.setattr(rs_mod, "_load_encdec_model", _raise)
    monkeypatch.setattr(rs_mod.t.cuda, "is_available", lambda: False)
    monkeypatch.setattr(rs_mod.t.cuda, "empty_cache", MagicMock())

    # We still patch stage functions to deterministic no-ops so that
    # _run fails for the env-validator reason and not some
    # intermediate parse error that would confuse the assertion.
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    _install_stage_stubs(monkeypatch, concepts=concepts)

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag="test-no-env",
        run_negative_mode=True,
    )
    with pytest.raises(AxBenchBaseError, match="OPENROUTER_API_KEY"):
        _run(cfg)


# ---------------------------------------------------------------------------
# Test 7: stage 3 splits run individually
# ---------------------------------------------------------------------------


def test_full_pipeline_stage_3_splits_run_individually(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run each of stages 3a, 3b, 3c, 3d individually.

    Asserts:

    - only the corresponding trainer fires per invocation
    - 3a/3b/3d produce per-concept direction / adapter files in the
      paths Stage 4's dep check expects
    - 3c produces ``{mode}_descriptions.json`` files
    """
    del stub_heavy_loaders, mock_openrouter_wire
    run_tag = "test-stage-3-splits"
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", run_tag)

    # Pre-populate every prior stage output — the dep check
    # for each 3x stage requires concepts.json +
    # synth_training_positive + synth_training_negative.
    _write_concepts_json(output_paths, concepts)
    _write_synth_training_positive(output_paths, concepts)
    _write_synth_training_negative(output_paths, concepts)

    counts = _install_stage_stubs(monkeypatch, concepts=concepts)

    # Stage 3a — DiffMean only.
    cfg_3a = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=run_tag,
        run_negative_mode=True,
        stage="3a",
    )
    cfg_3a_obj = cli.cfg_to_dataclass(cfg_3a)
    _write_successful_stage1_handoff(
        cfg_3a_obj,
        stage="stage1b",
        concepts=concepts,
    )
    _write_successful_stage1_handoff(
        cfg_3a_obj,
        stage="stage1c",
        concepts=concepts,
    )
    _run(cfg_3a)
    assert counts["train_diffmean_all"] == 4
    assert counts["train_linear_probe_all"] == 0
    assert counts["write_prompting_descriptions"] == 0
    assert counts["train_lora_all"] == 0
    # All pos+neg directions on disk.
    for c in concepts:
        for target in ("decoder_only", "encdec"):
            for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
                assert output_paths.diffmean_direction(
                    target_model=target,
                    mode=mode,
                    concept_id=c.concept_id,
                ).exists()

    # Stage 3b — LinearProbe only.
    cfg_3b = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=run_tag,
        run_negative_mode=True,
        stage="3b",
    )
    _run(cfg_3b)
    assert counts["train_linear_probe_all"] == 4
    for c in concepts:
        for target in ("decoder_only", "encdec"):
            for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
                assert output_paths.linear_probe_direction(
                    target_model=target,
                    mode=mode,
                    concept_id=c.concept_id,
                ).exists()

    # Stage 3c — Prompting only.
    cfg_3c = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=run_tag,
        run_negative_mode=True,
        stage="3c",
    )
    _run(cfg_3c)
    assert counts["write_prompting_descriptions"] == 1
    assert output_paths.prompting_descriptions(SteeringMode.POSITIVE).exists()
    assert output_paths.prompting_descriptions(SteeringMode.NEGATIVE).exists()

    # Stage 3d — LoRA only.
    cfg_3d = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=run_tag,
        run_negative_mode=True,
        stage="3d",
    )
    _run(cfg_3d)
    assert counts["train_lora_all"] == 2
    for c in concepts:
        for mode in (SteeringMode.POSITIVE, SteeringMode.NEGATIVE):
            assert output_paths.lora_adapter(
                mode=mode,
                concept_id=c.concept_id,
            ).exists()


# ---------------------------------------------------------------------------
# Test 8: memory thrash observable
# ---------------------------------------------------------------------------


def test_full_pipeline_memory_thrash_observable(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Instrument :class:`RuntimeState` so every CPU-offload transition
    is recorded. After a full ``stage="all"`` run verify:

    - base_model was woken up and parked on CPU at least once per
      stage that needs it (Stage 0c + 1b + 1c + 2 delta_ranked path).
    - Every entry has a matching exit (no leaks).

    The Phase 8 contract is the "refcount drops to 0 when the last
    ``with`` block exits, and the live model is offloaded to CPU".
    """
    del stub_heavy_loaders, mock_openrouter_wire
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    _install_stage_stubs(monkeypatch, concepts=concepts)

    transitions: list[tuple[str, str]] = []

    real_init = RuntimeState.__init__

    # Track every ``to(device)`` call on every stub via the per-model
    # name. We do this by wrapping the refcounted context managers in
    # a subclass-less instrumentation: we simply ask each stub to
    # append to ``transitions`` on ``.to()``. The stub models already
    # return ``self`` from ``.to()`` — here we wrap their ``.to``
    # method after the fact.
    states: list[RuntimeState] = []

    def instrumented_init(
        self: RuntimeState,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_init(self, *args, **kwargs)
        states.append(self)

    monkeypatch.setattr(RuntimeState, "__init__", instrumented_init)

    # Every heavy-model context manager calls ``.to("cpu")`` on exit
    # at refcount 0 and ``.to(device)`` on wake-up. To observe this
    # without coupling to the stub class we wrap each loader's return
    # value AFTER the fixture's stubs are installed. The single-loader
    # refactor merged ``_load_base_model`` / ``_load_base_sae`` into
    # :func:`_load_feature_extraction_pair` (returning ``(model, sae)``),
    # so the feature-extraction loader is wrapped once and each element
    # of the returned tuple gets its own tracking ``.to``.
    def _track(name: str, model: Any) -> Any:
        inner_to = model.to

        def tracking_to(device: Any) -> Any:
            transitions.append((name, str(device)))
            return inner_to(device)

        model.to = tracking_to
        return model

    def _wrap_single_loader(name: str) -> Any:
        original = getattr(rs_mod, f"_load_{name}")

        @functools.wraps(original)
        def wrapped(cfg: Any) -> Any:
            return _track(name, original(cfg))

        return wrapped

    original_pair = rs_mod._load_feature_extraction_pair

    @functools.wraps(original_pair)
    def _wrapped_pair(cfg: Any) -> tuple[Any, Any]:
        model, sae = original_pair(cfg)
        return _track("base_model", model), _track("base_sae", sae)

    monkeypatch.setattr(
        rs_mod,
        "_load_feature_extraction_pair",
        _wrapped_pair,
    )
    for _name in ("decoder_only_model", "encdec_model"):
        monkeypatch.setattr(
            rs_mod,
            f"_load_{_name}",
            _wrap_single_loader(_name),
        )

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag="test-thrash",
        run_negative_mode=True,
    )
    _run(cfg)

    # Every stage that used base_model must have parked it to CPU at
    # least once. Stages 0b, 0c, 1b, and 1c use base_model + base_sae.
    # Canonical Stage 2 does not touch Gemma. Target-model transitions
    # come from Stages 3a-3e and Stage 4.
    names_seen = {name for name, _ in transitions}
    # base_model is entered by stages that need extraction.
    assert "base_model" in names_seen, (
        f"base_model never cycled; transitions={transitions}"
    )
    assert "base_sae" in names_seen
    assert "decoder_only_model" in names_seen
    assert "encdec_model" in names_seen

    # Item 12: pin EXACT CPU-park counts per stub so a stage addition
    # or removal shows up as a test failure.
    #
    # For ``run_negative_mode=True`` + canonical concept_feature_mode
    # + stage="all", the observed per-stub CPU park count is pinned
    # at:
    #
    # * base_model         : 7
    # * base_sae           : 8
    # * decoder_only_model : 7
    # * encdec_model       : 9
    #
    # The exact counts depend on the refcount mechanics: every
    # :meth:`RuntimeState.base_model` context that exits at
    # ``refcount == 0`` calls ``.to("cpu")``, and each of the Phase 6
    # dispatch branches (0b, 0c, 1b, 1c) acquire and release the base
    # extraction contexts independently. The target-model counts
    # reflect the 3a/3b/3d scalar-baseline helper, Stage 3e's encdec
    # context enumeration, and Stage 4's nested model+engine acquisition.
    # Stages 0b and 0c deliberately do not load target models.
    #
    # base_sae is 1 higher than base_model because the
    # single-feature-extraction-load refactor populates both
    # ``_base_model`` and ``_base_sae`` when either accessor fires its
    # ``_ensure_feature_extraction_loaded`` branch — so on that first
    # stage the "other" accessor then falls into the
    # ``refcount == 0 → .to(device)`` wake-up branch. Under
    # ``device=cpu`` the wake-up ``.to("cpu")`` is picked up by this
    # test's park filter, producing the one-count asymmetry. It is
    # harmless in production (the wake-up is a no-op on a model
    # already resident on-device) and would disappear if the test ran
    # on CUDA.
    #
    # A CHANGE TO THIS COUNT MEANS A CHANGE TO THE PIPELINE'S
    # HEAVY-MODEL ACCESS PATTERN. Update the counts and document the
    # new access pattern before rubber-stamping the new numbers; a
    # runaway retry or a stale with-block could also cause
    # unexpected parks, and those are bugs.
    cpu_park_counts = {
        name: sum(1 for rec in transitions if rec == (name, "cpu"))
        for name in ("base_model", "base_sae", "decoder_only_model", "encdec_model")
    }
    expected_cpu_park_counts = {
        "base_model": 7,
        "base_sae": 8,
        "decoder_only_model": 7,
        "encdec_model": 9,
    }
    assert cpu_park_counts == expected_cpu_park_counts, (
        f"CPU park counts do not match the pinned expectation. "
        f"Expected {expected_cpu_park_counts}, got {cpu_park_counts}. "
        f"Full transitions:\n{transitions}"
    )

    # The one state we built has closed cleanly.
    assert len(states) == 1
    state = states[0]
    assert state._closed is True
    refcount_names = [
        ("base_model", "_base_model_refcount"),
        ("base_sae", "_base_sae_refcount"),
        ("decoder_only_model", "_decoder_only_refcount"),
        ("encdec_model", "_encdec_refcount"),
    ]
    for label, attr in refcount_names:
        value = getattr(state, attr, None)
        assert value == 0, f"{label}.{attr} is {value!r}, expected 0 after close()"


# ---------------------------------------------------------------------------
# Test 9: close idempotent after crash
# ---------------------------------------------------------------------------


def test_full_pipeline_close_idempotent_after_crash(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Force Stage 3d (LoRA training) to raise, verify:

    - the exception bubbles out of ``_run``
    - ``RuntimeState.close()`` still ran (state recorded via an
      instrumentation spy on the constructor)
    - all heavy-model refcounts are zero when ``close()`` finalizes
    """
    del stub_heavy_loaders, mock_openrouter_wire
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    _install_stage_stubs(monkeypatch, concepts=concepts)

    def boom(cfg: AxBenchBaseConfig, **kwargs: Any) -> list[Path]:
        del cfg, kwargs
        raise AxBenchBaseError("simulated Stage 3d crash")

    monkeypatch.setattr(cli, "train_lora_all", boom)

    built_states: list[RuntimeState] = []
    real_init = RuntimeState.__init__

    def instrumented_init(
        self: RuntimeState,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        real_init(self, *args, **kwargs)
        built_states.append(self)

    monkeypatch.setattr(RuntimeState, "__init__", instrumented_init)

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag="test-crash",
        run_negative_mode=True,
    )
    with pytest.raises(AxBenchBaseError, match="simulated Stage 3d crash"):
        _run(cfg)

    assert len(built_states) == 1
    state = built_states[0]
    # ``close()`` ran in the ``finally`` of ``_run`` after the
    # exception unwound, so the state is closed and every refcount
    # is zero.
    assert state._closed is True
    assert state._base_model_refcount == 0
    assert state._base_sae_refcount == 0
    assert state._decoder_only_refcount == 0
    assert state._encdec_refcount == 0


# ---------------------------------------------------------------------------
# Test 10: auto-generated run_tag
# ---------------------------------------------------------------------------


def test_full_pipeline_run_tag_auto_generation(
    tmp_run_root: Path,
    tiny_feature_statistics_path: Path,
    stub_heavy_loaders: dict[str, Any],
    mock_openrouter_wire: dict[str, Any],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Run with ``run_tag=None``. Verify:

    - ``_generate_run_tag`` was called and produced a timestamp-based
      tag
    - the generated tag appears in the output directory structure
      under ``output_root``
    - artifacts land under that resolved path
    """
    del stub_heavy_loaders, mock_openrouter_wire
    concepts = _concepts(_DEV_TIER_N_CONCEPTS)
    _install_stage_stubs(monkeypatch, concepts=concepts)

    captured_tag: dict[str, str] = {}
    real_generate = cli._generate_run_tag

    def spy_generate(cfg: AxBenchBaseConfig) -> str:
        tag = real_generate(cfg)
        captured_tag["tag"] = tag
        return tag

    monkeypatch.setattr(cli, "_generate_run_tag", spy_generate)

    cfg = _build_dev_cfg(
        tmp_run_root,
        tiny_feature_statistics_path,
        run_tag=None,
        run_negative_mode=True,
    )
    _run(cfg)

    assert "tag" in captured_tag
    generated = captured_tag["tag"]
    output_paths = paths.AxBenchBasePaths(tmp_run_root / "eval_out", generated)
    # Format: YYYYMMDD-HHMMSS-<8-char-hex>
    assert re.fullmatch(r"\d{8}-\d{6}-[0-9a-f]{8}", generated), generated
    # The report YAML for the generated tag exists.
    assert output_paths.report_yaml().exists()
    assert output_paths.leaderboard_csv().exists()
