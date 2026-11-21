"""Functional coverage for the BPB-Adv clean-source CLI."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch as t

import scripts.eval.axbench_bpb_adv_eval as bpb_adv_cli
import scripts.eval.clean_axbench_source as clean_cli
from steering.eval.axbench_base.types import TargetModelMetadata
from steering.eval.axbench_bpb.types import SteeringMode
from steering.eval.axbench_bpb_adv.attack_rows import BpbAttackWorkSpec
from steering.eval.axbench_bpb_adv.config import (
    AxBenchBpbAdvConfig,
    FasterGcgConfig,
    GeneralValBpbConfig,
    Stage,
)
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import run_model_backed_bpb_candidate_stage
from steering.eval.axbench_bpb_adv.paths import AxBenchBpbAdvPaths
from steering.eval.axbench_bpb_adv.schemas import CANDIDATES_SCHEMA, MACRO_AGG_SCHEMA, SCORE_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import build_bpb_adv_score_table
from tests.axbench_bpb_adv.functional.test_cli_stage_chain import _dual_score_pair
from tests.axbench_bpb_adv.source_run_builders import (
    both_class_rows,
    one_sided_rows,
    write_scalar_direction_baselines,
    write_stage_0_3_source_run,
)
from tests.axbench_bpb_adv.zero_candidate_callbacks import (
    zero_gradient_batch,
    zero_objective,
)

_VOCAB = 128


class _TinyTokenizer:
    def get_bos_token_id(self) -> int:
        return 0

    def get_vocab_size(self) -> int:
        return _VOCAB

    def encode(self, text: str) -> list[int]:
        return [ord(ch) % _VOCAB for ch in text]

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


def _meta() -> TargetModelMetadata:
    return TargetModelMetadata(
        num_buckets=None,
        n_sae_features=8,
        negative_selection=None,
        feature_selection=None,
        per_feature_stats=None,
        device="cpu",
        sae_layer=6,
        segment_size=128,
    )


def _write_pairing_source(root: Path) -> None:
    write_stage_0_3_source_run(
        root,
        positive_rows=both_class_rows("p-ok-1")
        + both_class_rows("p-ok-2")
        + one_sided_rows("p-rich-only", content_class="concept_rich"),
        negative_rows=both_class_rows("n-ok-1")
        + both_class_rows("n-ok-2")
        + one_sided_rows("n-avoid-only", content_class="concept_avoiding"),
        baselines_builder=write_scalar_direction_baselines,
    )


def test_clean_source_cli_default_output_dir(tmp_path: Path) -> None:
    source = tmp_path / "source"
    _write_pairing_source(source)

    result = clean_cli.main(["--source-run-dir", str(source)])

    assert result.output_dir == tmp_path / "source_paired"
    assert result.manifest_path == tmp_path / "source_paired" / "cleaning_manifest.json"
    assert result.manifest_path.exists()
    assert result.validation.n_dropped == 0


def test_clean_source_cli_default_output_dir_from_dot(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    _write_pairing_source(source)
    monkeypatch.chdir(source)

    result = clean_cli.main(["--source-run-dir", "."])

    assert result.output_dir == tmp_path / "source_paired"
    assert result.manifest_path == tmp_path / "source_paired" / "cleaning_manifest.json"
    assert result.manifest_path.exists()
    assert result.validation.n_dropped == 0


def _config(
    *,
    source_run_dir: Path,
    output_root: Path,
    run_tag: str,
) -> AxBenchBpbAdvConfig:
    return AxBenchBpbAdvConfig(
        source_run_dir=str(source_run_dir),
        output_root=str(output_root),
        run_tag=run_tag,
        device="cpu",
        feature_dir=str(output_root / "features"),
        feature_statistics_path=str(output_root / "stats.safetensors"),
        steering_metric="bpb_class_margin",
        target_models=["decoder_only"],
        adversarial_methods=[
            FasterGcgConfig(
                optimized_token_counts=(0,),
                max_steps=0,
            )
        ],
        # This tiny CPU pipeline doesn't mock score_general_val, and general_val_bpb
        # now defaults to enabled=True (it needs a real decoder-only model to
        # score) -- keep it off, matching this test's un-mocked stages.
        general_val_bpb=GeneralValBpbConfig(enabled=False),
        scatter_baseline_x="baseline_loss",
    )


def test_clean_source_cli_repairs_source_that_eval_rejects(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    cleaned = tmp_path / "source_paired"
    _write_pairing_source(source)

    monkeypatch.setattr(
        bpb_adv_cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg: ({"decoder_only": _meta()}, _TinyTokenizer()),
    )

    def _fake_work_specs(**_kwargs: Any) -> list[BpbAttackWorkSpec]:
        return [
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
            BpbAttackWorkSpec(
                method_name="neg_diffmean_decoder_only",
                mode=SteeringMode.NEGATIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
        ]

    monkeypatch.setattr(bpb_adv_cli, "work_specs_from_method_registry", _fake_work_specs)

    dirty_run_tag = "dirty_warn"
    dirty_cfg = _config(
        source_run_dir=source,
        output_root=tmp_path / "dirty_out",
        run_tag=dirty_run_tag,
    )
    with pytest.raises(AxBenchBpbAdvError, match="missing a class side"):
        bpb_adv_cli._stage_validate_source(dirty_cfg)

    result = clean_cli.main(["--source-run-dir", str(source), "--output-dir", str(cleaned)])
    assert result.output_dir == cleaned.resolve()
    assert len(result.positive.dropped_incomplete_groups) == 1
    assert len(result.negative.dropped_incomplete_groups) == 1

    clean_run_tag = "clean_error"
    clean_cfg = _config(
        source_run_dir=cleaned,
        output_root=tmp_path / "clean_out",
        run_tag=clean_run_tag,
    )
    bpb_adv_cli._stage_validate_source(clean_cfg)
    bpb_adv_cli._stage_prepare_attack_rows(clean_cfg)

    clean_paths = AxBenchBpbAdvPaths(
        output_root=clean_cfg.output_root,
        run_tag=clean_run_tag,
    )
    clean_rows = pq.read_table(clean_paths.attack_rows_path())
    assert clean_rows.num_rows == 2

    clean_contract = json.loads(clean_paths.source_contract().read_text(encoding="utf-8"))
    assert clean_contract["n_dropped"] == 0


def test_cleaned_source_runs_tiny_cpu_pipeline_through_plot(
    tmp_path: Path, monkeypatch
) -> None:
    source = tmp_path / "source"
    cleaned = tmp_path / "source_paired"
    _write_pairing_source(source)
    result = clean_cli.main(["--source-run-dir", str(source), "--output-dir", str(cleaned)])
    assert result.validation.n_dropped == 0

    tokenizer = _TinyTokenizer()
    monkeypatch.setattr(
        bpb_adv_cli,
        "_target_runtime_info_from_runtime",
        lambda _cfg: ({"decoder_only": _meta()}, tokenizer),
    )

    def _fake_work_specs(**_kwargs: Any) -> list[BpbAttackWorkSpec]:
        return [
            BpbAttackWorkSpec(
                method_name="diffmean_decoder_only",
                mode=SteeringMode.POSITIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
            BpbAttackWorkSpec(
                method_name="neg_diffmean_decoder_only",
                mode=SteeringMode.NEGATIVE,
                target_model="decoder_only",
                coefficient_str="1.0",
                baseline_family="diffmean",
            ),
        ]

    monkeypatch.setattr(bpb_adv_cli, "work_specs_from_method_registry", _fake_work_specs)
    monkeypatch.setattr(
        bpb_adv_cli,
        "_stage_prepare_context_cache",
        lambda *_args, **_kwargs: None,
    )

    def _fake_candidate_stage(*, cfg: AxBenchBpbAdvConfig, paths: AxBenchBpbAdvPaths) -> int:
        return run_model_backed_bpb_candidate_stage(
            cfg=cfg,
            paths=paths,
            tokenizer=tokenizer,
            embedding_weight=t.zeros(_VOCAB, 1),
            test_objective=zero_objective,
            test_gradient_provider_batch=zero_gradient_batch,
        )

    monkeypatch.setattr(bpb_adv_cli, "run_model_backed_bpb_candidate_stage", _fake_candidate_stage)

    def _fake_score_stage(
        *,
        cfg: AxBenchBpbAdvConfig,
        paths: AxBenchBpbAdvPaths,
        candidates: pa.Table,
        stats: dict[str, int] | None = None,
    ) -> pa.Table:
        del cfg, paths, stats
        scores = build_bpb_adv_score_table(
            candidates=candidates,
            context_segments_by_prefix_hash={},
            tokenizer=tokenizer,
            token_bytes=t.ones(_VOCAB, dtype=t.int64),
            max_seq_len_by_target={"decoder_only": 32},
            score_pair=_dual_score_pair,
        )
        rows = scores.to_pylist()
        for mode, method_name in (
            ("positive", "feature_diffmean_encdec"),
            ("negative", "neg_feature_diffmean_encdec"),
        ):
            base = next(row for row in rows if row["mode"] == mode and row["is_control"])
            rows.append(
                {
                    **base,
                    "candidate_id": f"feature-diffmean-target-{mode}",
                    "method_name": method_name,
                    "target_model": "encdec",
                    "coefficient_str": "None",
                }
            )
        return pa.Table.from_pylist(rows, schema=SCORE_SCHEMA)

    monkeypatch.setattr(bpb_adv_cli, "run_model_backed_bpb_adv_score_stage", _fake_score_stage)

    cfg = _config(
        source_run_dir=cleaned,
        output_root=tmp_path / "clean_full_out",
        run_tag="clean_full",
    )
    paths = AxBenchBpbAdvPaths(output_root=cfg.output_root, run_tag="clean_full")
    for stage in bpb_adv_cli._resolve_stages(Stage.ALL):
        bpb_adv_cli._run_stage(stage, cfg, ddp_rank=0, ddp_world_size=1)

    assert pq.read_table(paths.candidates_path()).schema.equals(CANDIDATES_SCHEMA)
    assert pq.read_table(paths.scores_path()).schema.equals(SCORE_SCHEMA)
    assert pq.read_table(paths.aggregates_dir() / "macro.parquet").schema.equals(
        MACRO_AGG_SCHEMA
    )
    assert list(paths.plots_dir().rglob("*.png"))
