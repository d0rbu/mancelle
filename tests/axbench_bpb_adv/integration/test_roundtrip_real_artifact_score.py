"""Manual real-artifact roundtrip score smoke test.

Default CI intentionally skips this file via the ``very_slow`` marker. To run it
on a GPU host with real AxBench-BPB-Adv artifacts:

    AXBENCH_BPB_ADV_ROUNDTRIP_REAL_CONFIG=/path/to/config.yaml \
      uv run pytest tests/axbench_bpb_adv/integration/test_roundtrip_real_artifact_score.py \
      -m very_slow -v

The config must point at an existing run with `candidates/candidates.parquet`
already built. The test runs the production score stage with
`steering_metric=roundtrip` and validates the real roundtrip score schema.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import pyarrow.parquet as pq
import pytest
from omegaconf import DictConfig, OmegaConf

import scripts.eval.axbench_bpb_adv_eval as cli
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb.source_paths import SourcePaths
from steering.eval.axbench_bpb_adv.aggregation import aggregate_roundtrip_adv_scores
from steering.eval.axbench_bpb_adv.config import SteeringMetric, validate_config
from steering.eval.axbench_bpb_adv.config_loader import cfg_to_dataclass
from steering.eval.axbench_bpb_adv.schemas import ROUNDTRIP_SCORE_SCHEMA
from steering.eval.axbench_bpb_adv.scoring import _read_concepts

pytestmark = [pytest.mark.very_slow, pytest.mark.requires_feature_data]


def test_roundtrip_score_stage_on_real_artifacts() -> None:
    config_path = os.environ.get("AXBENCH_BPB_ADV_ROUNDTRIP_REAL_CONFIG")
    if not config_path:
        pytest.skip("set AXBENCH_BPB_ADV_ROUNDTRIP_REAL_CONFIG to run real-artifact smoke")

    raw_cfg = OmegaConf.load(Path(config_path))
    if not isinstance(raw_cfg, DictConfig):
        pytest.fail("real-artifact config must be a mapping/DictConfig")
    cfg = cfg_to_dataclass(raw_cfg)
    cfg.steering_metric = SteeringMetric.ROUNDTRIP.value
    validate_config(cfg)
    paths = cli._paths(cfg)
    if not paths.candidates_path().exists():
        pytest.skip(
            "real-artifact smoke requires prebuilt candidates at "
            f"{paths.candidates_path()}"
        )

    cli._stage_score(cfg)

    scores = pq.read_table(paths.roundtrip_scores_path())
    assert scores.schema.equals(ROUNDTRIP_SCORE_SCHEMA)
    assert scores.num_rows > 0
    candidates = pq.read_table(paths.candidates_path())
    assert scores.num_rows == (
        candidates.num_rows * cfg.roundtrip_generation.samples_per_candidate
    )
    concepts_by_id = _read_concepts(SourcePaths(Path(cfg.source_run_dir)))
    for row in scores.to_pylist():
        assert row["generated_text"] != ""
        assert row["control_generated_text"] != ""
        expected_generation_budget = int(row["reference_segment_token_count"])
        assert 0 < row["n_generated_tokens"] <= expected_generation_budget
        assert 0 < row["control_n_generated_tokens"] <= expected_generation_budget
        assert token_sequence_digest(row["roundtrip_prompt_tokens"]) == (
            row["roundtrip_prompt_token_digest"]
        )
        assert token_sequence_digest(row["control_roundtrip_prompt_tokens"]) == (
            row["control_roundtrip_prompt_token_digest"]
        )
        concept = concepts_by_id[int(row["concept_id"])]
        assert row["target_feature_id"] == concept.feature_index
        for field in (
            "loss_concept_rich",
            "loss_concept_avoiding",
            "roundtrip_pre_act",
            "roundtrip_post_act",
            "control_roundtrip_pre_act",
            "control_roundtrip_post_act",
        ):
            assert math.isfinite(float(row[field]))
        for field in ("roundtrip_percentile", "control_roundtrip_percentile"):
            value = float(row[field])
            assert 0.0 <= value <= 100.0

    macro = aggregate_roundtrip_adv_scores(
        paths.roundtrip_scores_path(),
        paths.roundtrip_aggregates_dir(),
    )
    assert macro.num_rows > 0
