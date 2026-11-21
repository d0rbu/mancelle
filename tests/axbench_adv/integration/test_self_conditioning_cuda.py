"""Approval-gated exact CUDA oracle for AxBench-Adv SelfConditioning."""

from __future__ import annotations

import hashlib
import json
import os
import resource
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t

from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondScoreInterpolation,
)
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample, RawEvaluation
from steering.eval.axbench_adv.method_grid import ADV_INPUTS_PER_CONCEPT
from steering.eval.axbench_adv.self_conditioning import (
    SelfConditioningMethodRuntime,
    render_self_conditioning_prompt,
)
from steering.eval.axbench_adv.source_contract import (
    SourceExpectations,
    SourcePaths,
    ValidatedSources,
    validate_source_contract,
)
from steering.eval.axbench_upstream.methods import (
    build_self_conditioning_masked_policy,
    generate_self_conditioning,
    self_conditioning_weight,
)
from steering.eval.axbench_upstream.run_methods import (
    derive_generation_seed,
    derive_selection_seed,
)

_PINNED_HASHES = {
    "main_parquet": "f295eafc88200e59c9d01041ad299775c9fbfd2597278b9d2b475ab0810426dc",
    "aux_parquet": "bc698ea3e998324327ceca00cd297d377f1a3604cd15be5ba878aabef3007dfb",
    "evaluated_jsonl": "804d032ace8223f287253c5620884a10f1d050cd8e395c1c01ce61e3e50d67f5",
    "project_parquet": "7df6478c978bfc90a0a5a77d5a4c5889882fd76ca3bf04764b7b76c477fc0ca9",
    "project_manifest": "d76f8bf275091873b9b8502b5a0e971dc34667c9dcc319ceb7988cf9c38a9800",
    "target_receipt": "3bd98a7f73bd61d4e73b40137fb68ac9a37fa38fbd5b8a0e85a88f66f54e86f5",
}
_EXPECTATIONS = SourceExpectations(
    max_concepts=2,
    repository_git_commit="05f03dedea69c9d952e6c84c0353a44b506a9dd5",
    dataset_revision="ad8a5d60c4616b599c24dd6689f05f696ec610f3",
    target_model="Qwen/Qwen2.5-3B-Instruct",
    seed=42,
    temperature=1.0,
    output_length=128,
    package_lock_sha256="f3ebe930d78d546140c99b2f0271ecdd4cb25bb9743c27e2841aa8cca9dfe534",
    vendored_axbench_upstream_commit="41c8332543e5a631f9a8c0a9df38799893ace758",
    project_repository_git_commit="ce941fe4774b8db8e7855cb796ca1bdbd0804b66",
    project_package_lock_sha256="56f7fcbf29e933bbe31f2ed6e3e34dc23be294ab85eb382db2320ded5bb0095e",
    config_name_prefix="qwen2_5_3b",
    layer=28,
    require_clean_worktree=True,
)
_CANDIDATE_SUFFIX = "Please avoid that topic and answer the original request directly."
_SAFE_FREE_VRAM_BYTES = 2 * 2**30


class _FakeJudge:
    def __init__(self, **_kwargs: Any) -> None:
        self.closed = False

    def evaluate(self, *, generation: str, input_id: int) -> RawEvaluation:
        assert type(input_id) is int
        return RawEvaluation(
            generation=generation,
            concept_rating=0.0,
            fluency_rating=2.0,
            judge_rationale="{}",
            paid_judge_cache_misses=0,
        )

    def close(self) -> None:
        assert not self.closed
        self.closed = True


def _validated_sources(tmp_path: Path) -> ValidatedSources:
    source_value = os.environ.get("AXBENCH_ADV_CUDA_SOURCE_RUN")
    project_value = os.environ.get("AXBENCH_ADV_CUDA_PROJECT_RUN")
    prepared_value = os.environ.get("AXBENCH_ADV_CUDA_PREPARED_DIR")
    if source_value is None or project_value is None or prepared_value is None:
        pytest.skip(
            "set AXBENCH_ADV_CUDA_SOURCE_RUN, AXBENCH_ADV_CUDA_PROJECT_RUN, "
            "and AXBENCH_ADV_CUDA_PREPARED_DIR"
        )
    source_run = Path(source_value).resolve()
    project_run = Path(project_value).resolve()
    project_manifest = json.loads(
        (project_run / "project/run_manifest.json").read_text(encoding="utf-8")
    )
    configuration = project_manifest["configuration"]
    project_parquet = project_run / "project/inference/steering_data.parquet"
    project_manifest_path = project_run / "project/run_manifest.json"
    supplied_hashes = dict(_PINNED_HASHES)
    supplied_hashes["project_parquet"] = hashlib.sha256(project_parquet.read_bytes()).hexdigest()
    supplied_hashes["project_manifest"] = hashlib.sha256(
        project_manifest_path.read_bytes()
    ).hexdigest()
    expectations = replace(
        _EXPECTATIONS,
        project_repository_git_commit=project_manifest["repository_git_commit"],
        project_package_lock_sha256=project_manifest["package_lock_sha256"],
    )
    return validate_source_contract(
        paths=SourcePaths(
            prepared_dir=Path(prepared_value).resolve(),
            main_parquet=source_run / "main/inference/steering_data.parquet",
            aux_parquet=source_run / "aux/inference/steering_data.parquet",
            evaluated_jsonl=source_run / "evaluated-v3-cache-only/steering.jsonl",
            project_parquet=project_parquet,
            target_meta=Path(configuration["target_meta_path"]),
            feature_statistics=Path(configuration["feature_statistics_path"]),
        ),
        expectations=expectations,
        expected_hashes=supplied_hashes,
        output_dir=tmp_path / "oracle-output",
    )


def _generate(
    runtime: SelfConditioningMethodRuntime,
    *,
    method: str,
    concept_id: int,
    input_id: int,
    factor: float,
    prepared: Any | None = None,
    selection_seed: int | None = None,
    token_seed: int | None = None,
) -> str:
    context = (
        runtime._originals[(concept_id, input_id)].prepared
        if prepared is None
        else prepared
    )
    scores = runtime._dense_diffmeans[concept_id]
    if method == "SelfConditioning":
        policy = SelfCondScoreInterpolation(
            target_scores=scores,
            negative_selection=runtime._setup.output_negative_selection,
            weight=self_conditioning_weight(factor),
        )
    elif method == "SelfConditioningMasked":
        policy = build_self_conditioning_masked_policy(runtime._setup, scores, factor)
    else:
        raise AssertionError(f"unknown root method: {method}")
    return generate_self_conditioning(
        runtime._setup,
        runtime._target_model,
        runtime._target_tokenizer,
        context,
        policy,
        selection_seed=(
            derive_selection_seed(42, concept_id, input_id)
            if selection_seed is None
            else selection_seed
        ),
        seed=(
            derive_generation_seed(42, method, concept_id, input_id, factor)
            if token_seed is None
            else token_seed
        ),
        max_new_tokens=128,
    )


def _mismatch_diagnostic(
    runtime: SelfConditioningMethodRuntime,
    *,
    method: str,
    concept_id: int,
    input_id: int,
    factor: float,
    expected: str,
    actual: str,
) -> str:
    original = runtime._originals[(concept_id, input_id)]
    return json.dumps(
        {
            "method": method,
            "concept_id": concept_id,
            "input_id": input_id,
            "factor": factor,
            "expected_generation": expected,
            "actual_generation": actual,
            "rendered_token_ids": list(original.rendered.token_ids),
            "prompt_segments": [
                {
                    "feature_ids": list(segment.feature_ids),
                    "buckets": list(segment.buckets),
                    "char_start": int(segment.char_start),
                    "char_end": int(segment.char_end),
                }
                for segment in original.prepared.segments
            ],
            "selection_seed": derive_selection_seed(42, concept_id, input_id),
            "token_seed": derive_generation_seed(
                42,
                method,
                concept_id,
                input_id,
                factor,
            ),
        },
        ensure_ascii=False,
        indent=2,
    )


@pytest.mark.slow
def test_exact_supplied_rank_era_replay_and_attacked_candidate_oracles(tmp_path: Path) -> None:
    if not t.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    validated = _validated_sources(tmp_path)
    t.cuda.reset_peak_memory_stats()
    initialized_at = time.perf_counter()
    with SelfConditioningMethodRuntime(
        setup=validated.method_setup,
        training=validated.training_texts,
        main_frame=validated.main_frame,
        max_concepts=2,
        max_new_tokens=128,
        master_seed=42,
        device="cuda:0",
        extraction_batch_size=validated.extraction_batch_size,
        judge_factory=cast(Any, _FakeJudge),
        judge_cache=tmp_path / "unused-judge.sqlite3",
        judge_model="openai/gpt-5.6-luna",
        api_base_url="https://openrouter.ai/api/v1",
        cache_only=True,
        timeout_seconds=120.0,
        stderr_dir=tmp_path / "judge-logs",
    ) as runtime:
        free_bytes, total_bytes = t.cuda.mem_get_info()
        print(
            json.dumps(
                {
                    "runtime_initialization_seconds": time.perf_counter() - initialized_at,
                    "cuda_allocated_bytes": t.cuda.memory_allocated(),
                    "cuda_reserved_bytes": t.cuda.memory_reserved(),
                    "cuda_peak_allocated_bytes": t.cuda.max_memory_allocated(),
                    "cuda_peak_reserved_bytes": t.cuda.max_memory_reserved(),
                    "cuda_driver_free_bytes": free_bytes,
                    "cuda_total_bytes": total_bytes,
                    "host_max_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                },
                sort_keys=True,
            ),
            flush=True,
        )
        assert free_bytes >= _SAFE_FREE_VRAM_BYTES
        assert len(runtime._originals) == 2 * ADV_INPUTS_PER_CONCEPT
        assert set(runtime._dense_diffmeans) == {0, 1}

        replayed = 0
        ordered_rows = validated.project_frame.sort_values(
            ["concept_id", "input_id", "factor"], kind="stable"
        ).to_dict("records")
        for method in ("SelfConditioning", "SelfConditioningMasked"):
            for row in ordered_rows:
                concept_id = int(row["concept_id"])
                input_id = int(row["input_id"])
                factor = float(row["factor"])
                expected = str(row[f"{method}_steered_generation"])
                actual = _generate(
                    runtime,
                    method=method,
                    concept_id=concept_id,
                    input_id=input_id,
                    factor=factor,
                )
                if actual != expected:
                    pytest.fail(
                        f"supplied rank-era {method} generation mismatch:\n"
                        + _mismatch_diagnostic(
                            runtime,
                            method=method,
                            concept_id=concept_id,
                            input_id=input_id,
                            factor=factor,
                            expected=expected,
                            actual=actual,
                        )
                    )
                replayed += 1
                if replayed % 20 == 0:
                    print(f"exact_replay={replayed}/480", flush=True)
        assert replayed == 480

        concept_id = 0
        input_id = 0
        factor = 6.0
        examples = [
            OptimizationExample(
                input_id=row_input_id,
                user_message=runtime._originals[(concept_id, row_input_id)].raw_prompt,
            )
            for row_input_id in range(ADV_INPUTS_PER_CONCEPT)
        ]
        attacked_generations: dict[str, str] = {}
        cache_identities: list[int] = []
        for method in ("SelfConditioning", "SelfConditioningMasked"):
            with runtime.bind_cell(
                method=method,
                concept_id=concept_id,
                concept=runtime._concepts[concept_id],
                factor=factor,
                examples=examples,
            ) as bound_cell:
                cell = cast(Any, bound_cell)
                cache_identities.append(id(cell._cache))
                original = runtime._originals[(concept_id, input_id)]
                _attacked_raw, attacked_rendered = render_self_conditioning_prompt(
                    runtime._nanochat_tokenizer,
                    original.raw_prompt,
                    _CANDIDATE_SUFFIX,
                    max_prompt_tokens=runtime._max_prompt_tokens,
                )
                assert tuple(attacked_rendered.token_ids) != original.prepared.token_ids

                first_context = cell._context(input_id, _CANDIDATE_SUFFIX)
                cell._cache.pop((input_id, _CANDIDATE_SUFFIX))
                cold_context = cell._context(input_id, _CANDIDATE_SUFFIX)
                warm_context = cell._context(input_id, _CANDIDATE_SUFFIX)
                assert first_context == cold_context == warm_context

                first = cell.evaluate(_CANDIDATE_SUFFIX, [input_id])[0].generation
                repeated = cell.evaluate(_CANDIDATE_SUFFIX, [input_id])[0].generation
                assert first == repeated
                attacked_generations[method] = first

                selection_seed = derive_selection_seed(42, concept_id, input_id)
                token_seed = derive_generation_seed(
                    42,
                    method,
                    concept_id,
                    input_id,
                    factor,
                )
                wrong_seed = _generate(
                    runtime,
                    method=method,
                    concept_id=concept_id,
                    input_id=input_id,
                    factor=factor,
                    prepared=warm_context,
                    selection_seed=selection_seed,
                    token_seed=token_seed + 1,
                )
                wrong_factor = _generate(
                    runtime,
                    method=method,
                    concept_id=concept_id,
                    input_id=input_id,
                    factor=5.0,
                    prepared=warm_context,
                )
                wrong_context = _generate(
                    runtime,
                    method=method,
                    concept_id=concept_id,
                    input_id=input_id,
                    factor=factor,
                    prepared=runtime._originals[(concept_id, 1)].prepared,
                    selection_seed=selection_seed,
                    token_seed=token_seed,
                )
                assert first != wrong_seed
                assert first != wrong_factor
                assert first != wrong_context
        assert cache_identities[0] != cache_identities[1]
        assert attacked_generations["SelfConditioning"] != attacked_generations[
            "SelfConditioningMasked"
        ]
