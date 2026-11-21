"""Issue #1057: explicit filtered-population consistency integration test.

A concept that appears in immutable Stage 0c ``concepts.json`` but is excluded
by Stage 1b must be:

  1. Absent from the explicit Stage 1b concept list handed to
     ``train_diffmean_all``—the trainer never receives an empty-data concept.
  2. NOT listed by ``_stage_4_missing_artifacts`` as a required (missing)
     artifact—the DAG enumerates the same explicit final list.

This test exercises both properties together with a real (tiny CPU) GPT to
confirm the Stage 3 → DAG hand-off is consistent.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch as t
from safetensors.torch import save_file

import nanochat.flash_attention as _fa_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.eval.axbench_base import paths
from steering.eval.axbench_base.baselines import _capture as capture_mod
from steering.eval.axbench_base.baselines.diffmean import train_diffmean_all
from steering.eval.axbench_base.config import LayerHooksConfig, LoraConfig
from steering.eval.axbench_base.dag import _stage_4_missing_artifacts
from steering.eval.axbench_base.synth.concept_yield import (
    ConceptYieldRecord,
    write_concept_yield_report,
    write_filtered_concepts,
)
from steering.eval.axbench_base.types import Concept, SteeringMode
from tests.axbench_base.unit.test_config import make_valid_config
from tests.axbench_base.unit.test_diffmean import (
    TINY_LAYER,
    TINY_N_EMBD,
    _install_stub_tokenizer,
    _make_tiny_gpt,
    _pos_row,
    _write_positive_corpus,
)

pytestmark = pytest.mark.slow


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _concept(cid: int) -> Concept:
    return Concept(
        concept_id=cid,
        feature_index=100 + cid,
        layer=8,
        sae_source="8-gemmascope-res-16k",
        description=f"c{cid}",
        corpus_count=50,
        description_source="neuronpedia",
    )


def _write_stats(path: Path, n_sae_features: int = 16384) -> None:
    """Write a minimal feature-statistics safetensors that the DAG dep check can read."""
    path.parent.mkdir(parents=True, exist_ok=True)
    save_file(
        {"_p": t.zeros(1, dtype=t.int64)},
        str(path),
        metadata={"n_sae_features": str(n_sae_features)},
    )


def _make_config(tmp_path: Path, stats_path: Path) -> Any:
    """Return a minimal CPU config with negative mode disabled."""
    return make_valid_config(
        run_tag="absent508",
        output_root=str(tmp_path / "out"),
        device="cpu",
        feature_statistics_path=str(stats_path),
    ).replace(
        run_negative_mode=False,
        layer_hooks=LayerHooksConfig(diffmean_and_probe_layer=TINY_LAYER),
        lora=LoraConfig(min_training_rows=2),
    )


# ---------------------------------------------------------------------------
# Core integration test
# ---------------------------------------------------------------------------


def test_filtered_concept_excluded_across_stage3_and_dag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concept 1 is absent from Stage 1b's final list; concept 0 has rows.

    Assertions:
      (a) ``train_diffmean_all`` receives only concept 0 and writes one path.
      (b) ``_stage_4_missing_artifacts`` does NOT include any of concept
          1's per-concept artifact paths.
    """
    # ---- environment ----
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))

    # Stub tokenizer in the diffmean trainer module.
    _install_stub_tokenizer(monkeypatch)

    # Force SDPA: FlexAttention cannot backward on CPU (lora training needs backward).
    old_impl = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    try:
        _run_test_body(tmp_path, monkeypatch)
    finally:
        _fa_mod._override_impl = old_impl


def _run_test_body(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stats = tmp_path / "stats.safetensors"
    _write_stats(stats)
    cfg = _make_config(tmp_path, stats)
    assert cfg.run_tag is not None
    output_paths = paths.AxBenchBasePaths.from_config(cfg)

    # concepts.json: concept 0 AND concept 1
    concepts = [_concept(0), _concept(1)]
    concepts_json_path = output_paths.concepts_json()
    concepts_json_path.parent.mkdir(parents=True, exist_ok=True)
    concepts_json_path.write_text(
        json.dumps(
            [
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
        )
    )
    final_concepts = [concepts[0]]
    survivors = write_concept_yield_report(
        cfg,
        stage="stage1b",
        records=[
            ConceptYieldRecord(
                concept_id=0,
                feature_index=100,
                attempted=2,
                accepted=2,
                dropped=0,
                drop_fraction=0.0,
                excluded=False,
                exclusion_reason=None,
                reason_counts={},
                emitted_rows=4,
            ),
            ConceptYieldRecord(
                concept_id=1,
                feature_index=101,
                attempted=2,
                accepted=0,
                dropped=2,
                drop_fraction=1.0,
                excluded=True,
                exclusion_reason="drop_fraction_exceeded",
                reason_counts={"rich_qc_failed": 2},
                emitted_rows=0,
            ),
        ],
        path=output_paths.stage1b_concept_yield_report(),
    )
    assert survivors == frozenset({0})
    assert write_filtered_concepts(
        concepts,
        survivors,
        output_paths.stage1b_concepts(),
    ) == final_concepts

    # Positive corpus: rows for concept 0 ONLY.
    rows: list[dict[str, Any]] = [
        _pos_row(0, "concept_rich", "POS a", "out a", 1),
        _pos_row(0, "concept_rich", "POS b", "out b", 1),
        _pos_row(0, "concept_avoiding", "NEG a", "out c", 1),
        _pos_row(0, "concept_avoiding", "NEG b", "out d", 1),
    ]
    pos_corpus = output_paths.synth_training_positive()
    pos_corpus.parent.mkdir(parents=True, exist_ok=True)
    _write_positive_corpus(pos_corpus, rows)

    # Stub out the heavy capture so the test runs fast on CPU.
    vec = t.zeros(TINY_N_EMBD)
    vec[0] = 1.0

    def _stub_capture(
        *,
        model: Any,
        layer: int,
        tokenizer: Any,
        prefix_text: str,
        output_text: str,
        n_prefix_nanochat_tokens: int,
    ) -> tuple[t.Tensor, t.Tensor]:
        del model, layer, tokenizer, output_text, n_prefix_nanochat_tokens
        mean = vec.clone() if "POS" in prefix_text else (vec.clone() * -1.0)
        return mean, mean.unsqueeze(0)

    monkeypatch.setattr(capture_mod, "_capture_per_row_continuation", _stub_capture)

    model = _make_tiny_gpt()

    # ---- assertion (a): Stage 3 train_diffmean_all ----
    dm_paths = train_diffmean_all(
        cfg=cfg,
        target_model=model,
        target_model_name="decoder_only",
        mode=SteeringMode.POSITIVE,
        concepts=final_concepts,
        training_corpus_path=pos_corpus,
    )
    assert len(dm_paths) == 1, (
        f"Expected exactly 1 direction (concept 0 only); got {len(dm_paths)}: {dm_paths}"
    )
    assert not output_paths.diffmean_direction(
        "decoder_only",
        SteeringMode.POSITIVE,
        1,
    ).is_file(), "concept 1's diffmean artifact must NOT have been written"

    # ---- seed artifact deps for Stage 4 ----
    # Generic Stage 4 non-per-concept deps (with run_negative_mode=False):
    #   concepts.json (done), concept_features.safetensors, seeds_eval.safetensors
    output_paths.concept_features().parent.mkdir(parents=True, exist_ok=True)
    output_paths.concept_features().write_bytes(b"stub")
    output_paths.seeds_eval().write_bytes(b"stub")
    output_paths.seed_pool_manifest().write_text("{}")
    output_paths.stage4_contexts_manifest().write_text("{}")

    # Prompting description (one per mode, NOT per concept).
    prompting_pos = output_paths.prompting_descriptions(SteeringMode.POSITIVE)
    prompting_pos.parent.mkdir(parents=True, exist_ok=True)
    prompting_pos.write_text("{}")

    # Per-concept × per-target artifacts for concept 0 only.
    for target in ("decoder_only", "encdec"):
        dm0 = output_paths.diffmean_direction(target, SteeringMode.POSITIVE, 0)
        dm0.parent.mkdir(parents=True, exist_ok=True)
        dm0.write_bytes(b"stub")
        lp0 = output_paths.linear_probe_direction(
            target,
            SteeringMode.POSITIVE,
            0,
        )
        lp0.parent.mkdir(parents=True, exist_ok=True)
        lp0.write_bytes(b"stub")

    lora0 = output_paths.lora_adapter(SteeringMode.POSITIVE, 0)
    lora0.parent.mkdir(parents=True, exist_ok=True)
    lora0.write_bytes(b"stub")

    # ---- assertion (b): DAG does NOT require concept 1's artifacts ----
    missing = _stage_4_missing_artifacts(cfg)
    concept_1_entries = [p for p in missing if "0001" in str(p)]
    assert not concept_1_entries, (
        f"Concept 1 was wrongly listed as required by the DAG: {concept_1_entries}"
    )
    # Sanity: nothing at all should be missing now that we seeded all of concept 0.
    assert not missing, (
        f"Stage 4 dep check still reports missing artifacts: {missing}"
    )
