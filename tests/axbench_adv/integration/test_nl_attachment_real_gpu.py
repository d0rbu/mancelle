"""Opt-in exact NL producer replay on the full masked step-1125 model.

The operator supplies the experiment's Hydra source config and an existing judge
cache through AXBENCH_ADV_NL_PILOT_CONFIG. This test never admits paid calls.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch as t
from hydra import compose, initialize_config_dir
from loguru import logger

from scripts.eval.axbench_adv.run import build_config
from steering.eval.axbench_adv.best_point import select_best_points
from steering.eval.axbench_adv.counter import CounterDirection
from steering.eval.axbench_adv.pipeline import _cell_rows
from steering.eval.axbench_adv.self_conditioning import SelfConditioningMethodRuntime
from steering.eval.axbench_adv.source_contract import validate_source_contract
from steering.eval.axbench_adv.types import NL_METHOD
from steering.eval.axbench_adv.worker_client import AxBenchAdvJudgeClient


@pytest.mark.slow
def test_full_masked_nl_source_and_counter_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = os.environ.get("AXBENCH_ADV_NL_PILOT_CONFIG")
    if config_path is None:
        pytest.skip("set AXBENCH_ADV_NL_PILOT_CONFIG for real NL source qualification")
    assert t.cuda.is_available(), "explicit real-model qualification requires CUDA"
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    path = Path(config_path).resolve()
    with initialize_config_dir(config_dir=str(path.parent), version_base=None):
        raw_config = compose(config_name=path.stem)
    config = build_config(raw_config)
    assert config.source_expectations.project_target_step == 1125
    assert config.source_expectations.max_concepts == 50
    assert config.judge_cache.is_file()
    validated = validate_source_contract(
        paths=config.source_paths,
        expectations=config.source_expectations,
        expected_hashes=config.source_hashes,
        output_dir=tmp_path / "qualification",
        reserved_paths=(config.judge_cache, config.reflection_cache),
        nl_attachment_paths=config.nl_attachment_paths,
        nl_attachment_expectations=config.nl_attachment_expectations,
    )
    assert validated.runtime_hashes["target_model"] == (
        "795a5b32687b0c6badb9b849fcadb9acf3d66eca54265ef24bc68ba3b2493da0"
    )
    attachment = validated.nl_attachment
    assert attachment is not None
    factor = select_best_points(attachment.steering_entries, methods=(NL_METHOD,))[
        NL_METHOD
    ].factor
    concept, _workers, examples = _cell_rows(
        validated, method=NL_METHOD, concept_id=0, factor=factor,
    )
    concept_frame = attachment.selected_frame.loc[
        attachment.selected_frame.concept_id.eq(0)
    ]
    selected = concept_frame.factor.eq(factor).to_numpy()
    reference = concept_frame.loc[selected]
    assert reference.input_id.tolist() == list(range(10))
    block = attachment.steering_entries[0]["results"]["LMJudgeEvaluator"][NL_METHOD]
    expected_concept = [
        value for keep, value in zip(
            selected, block["raw_relevance_concept_ratings"], strict=True,
        ) if keep
    ]
    expected_fluency = [
        value for keep, value in zip(
            selected, block["raw_fluency_ratings"], strict=True,
        ) if keep
    ]
    with SelfConditioningMethodRuntime(
        setup=validated.method_setup,
        training=validated.training_texts,
        main_frame=validated.main_frame,
        max_concepts=50,
        max_new_tokens=128,
        master_seed=config.master_seed,
        device=config.device,
        extraction_batch_size=validated.extraction_batch_size,
        judge_factory=AxBenchAdvJudgeClient,
        judge_cache=config.judge_cache,
        judge_model="openai/gpt-5.6-luna",
        api_base_url=config.api_base_url,
        cache_only=True,
        timeout_seconds=300,
        stderr_dir=tmp_path / "worker-logs",
        nl_target_scores=validated.nl_target_scores,
    ) as runtime:
        model = runtime._target_model
        assert sum(parameter.numel() for parameter in model.parameters()) > 2_500_000_000
        assert next(model.parameters()).device.type == "cuda"
        assert model.dec_config.n_layer == 27 and model.dec_config.n_embd == 1664
        assert len(runtime._originals) == 500
        assert set(runtime._nl_target_scores) == set(range(50))
        free_bytes, total_bytes = t.cuda.mem_get_info(config.device)
        assert free_bytes >= 2 * 1024**3, "full runtime must retain 2 GiB of GPU headroom"
        logger.info(
            "NL full-model qualification: factor={}, free_bytes={}, total_bytes={}, peak_allocated={}",
            factor, free_bytes, total_bytes, t.cuda.max_memory_allocated(config.device),
        )
        with runtime.bind_cell(
            method=NL_METHOD, concept_id=0, concept=concept, factor=factor, examples=examples,
        ) as cell:
            control = cell.evaluate("", list(range(10)))
            assert [row.generation for row in control] == reference[
                f"{NL_METHOD}_steered_generation"
            ].tolist()
            assert [row.concept_rating for row in control] == expected_concept
            assert [row.fluency_rating for row in control] == expected_fluency
            assert sum(row.paid_judge_cache_misses for row in control) == 0
            sentinel = CounterDirection(t.ones(1664), {"layer": 21})
            assert cell.evaluate_counter(sentinel, 0.0, list(range(10))) == control
            assert model._post_cross_attention_eval_callback is None
