"""Opt-in real-model NL runtime qualification, separate from judged-source replay.

Set NL_STEERING_GPU_CONFIG to the audited Concept500/MPNet qualification config
documented in docs/evaluation/axbench-adv.md. This loads real Gemma/SAE and 3B
checkpoints, retains the complete training contrast for concept zero, and uses
ten actual prepared evaluation prompts. Only the judge is replaced. Explicit
qualification fails if its assets or CUDA are unavailable.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
import pytest
import torch as t

from steering.conditional.encoder_decoder.self_conditioned import SelfCondScoreInterpolation
from steering.eval.axbench_adv.gepa_optimization import OptimizationExample
from steering.eval.axbench_adv.method_grid import ADV_FULL_FACTORS, ADV_INPUTS_PER_CONCEPT
from steering.eval.axbench_adv.self_conditioning import SelfConditioningMethodRuntime
from steering.eval.axbench_upstream.data import load_metadata, select_training_texts
from steering.eval.axbench_upstream.methods import generate_self_conditioning, load_method_setup
from steering.eval.axbench_upstream.nl_target_scores import build_retrieval_target_scores
from steering.eval.axbench_upstream.seeds import derive_generation_seed, derive_selection_seed
from steering.refined_types import UnitInterval
from tests.axbench_adv.integration.test_self_conditioning_cuda import _FakeJudge

pytestmark = pytest.mark.slow
_SUFFIX = "Please avoid that topic and answer the original request directly."


def _digest(tensor: t.Tensor) -> str:
    value = tensor.detach().contiguous().cpu()
    assert bool(t.isfinite(value).all())
    return hashlib.sha256(value.view(t.uint8).numpy().tobytes()).hexdigest()


@pytest.mark.parametrize("checkpoint_name", ["fixed", "turn"])
def test_real_3b_nl_runtime_matches_direct_generator(
    checkpoint_name: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    config_path = os.environ.get("NL_STEERING_GPU_CONFIG")
    if config_path is None:
        pytest.skip("set NL_STEERING_GPU_CONFIG to authenticated real qualification assets")
    assert t.cuda.is_available()
    config = json.loads(Path(config_path).read_text())
    checkpoint = config[checkpoint_name]
    statistics = Path(checkpoint["statistics_path"])
    assert hashlib.sha256(statistics.read_bytes()).hexdigest() == checkpoint["statistics_sha256"]
    setup = load_method_setup(
        Path(checkpoint["meta_path"]), statistics, require_exact_statistics_identity=True,
    )
    assert setup.sae_model == "gemma-3-4b-pt" and setup.sae_layer == 22
    assert setup.n_sae_features == 65_536
    assert setup.segment_mode.value == checkpoint["segment_mode"]
    if checkpoint_name == "turn":
        assert setup.turn_self_cond_segment_size_rustbpe == 33

    receipt_path = Path(config["receipt_path"])
    receipt = json.loads(receipt_path.read_text())
    assert receipt["schema_version"] == 1
    descriptions = receipt["description_embeddings"]
    queries = receipt["concept_embeddings"]
    assert descriptions["embedding_space"] == queries["embedding_space"]
    texts_bytes = Path(config["description_texts_path"]).read_bytes()
    assert hashlib.sha256(texts_bytes).hexdigest() == descriptions["source_texts_sha256"]
    description_path = receipt_path.parent / descriptions["path"]
    with np.load(description_path, allow_pickle=False) as archive:
        assert archive["fids"].tolist() == [row[0] for row in json.loads(texts_bytes)]
        assert archive["emb"].shape[1] == 768
    prepared_dir = Path(config["prepared_dir"])
    metadata = load_metadata(prepared_dir / "generate/metadata.jsonl")
    assert len(metadata) == 500
    concepts = {cid: row["concept"] for cid, row in enumerate(metadata)}
    targets = build_retrieval_target_scores(
        setup=setup, description_embeddings_path=description_path,
        concept_embeddings_path=receipt_path.parent / queries["path"], concepts=concepts,
        expected_hashes={name: receipt[name]["sha256"] for name in (
            "description_embeddings", "concept_embeddings",
        )},
    )
    assert set(targets) == set(range(500))
    training = select_training_texts(prepared_dir, max_concepts=1)
    assert len(training[0].positive) == len(training[0].negative) == 72
    evaluation = pd.read_parquet(prepared_dir / "generate/latent_eval_data.parquet")
    assert evaluation["concept_id"].nunique() == 500 and len(evaluation) >= 38_000
    prompts = evaluation.loc[evaluation["concept_id"].eq(0), "input"].head(ADV_INPUTS_PER_CONCEPT).tolist()
    assert len(prompts) == ADV_INPUTS_PER_CONCEPT
    # These are real prompt inputs to the runtime, not a forged project source.
    frame = pd.DataFrame([
        {"concept_id": 0, "input_id": iid, "factor": factor,
         "input_concept": concepts[0], "original_prompt": prompt}
        for iid, prompt in enumerate(prompts) for factor in ADV_FULL_FACTORS
    ])
    examples = [OptimizationExample(input_id=iid, user_message=prompt) for iid, prompt in enumerate(prompts)]
    t.cuda.reset_peak_memory_stats()
    with SelfConditioningMethodRuntime(
        setup=setup, training=training, main_frame=frame, max_concepts=1,
        max_new_tokens=96, master_seed=42, device="cuda:0", extraction_batch_size=4,
        judge_factory=cast(Any, _FakeJudge), judge_cache=tmp_path / "unused-judge.sqlite3",
        judge_model="openai/gpt-5.6-luna", api_base_url="http://127.0.0.1:1",
        cache_only=True, timeout_seconds=30, stderr_dir=tmp_path / "logs",
        nl_target_scores={0: targets[0]},
    ) as runtime:
        model = runtime._target_model
        assert sum(parameter.numel() for parameter in model.parameters()) >= 2_500_000_000
        assert next(model.parameters()).device.type == "cuda"
        assert len(runtime._originals) == ADV_INPUTS_PER_CONCEPT
        assert runtime._nl_target_scores[0].numpy().tobytes() == targets[0].numpy().tobytes()
        traces: list[Any] = []
        forward = model.forward

        def capture_forward(*args: Any, **kwargs: Any) -> Any:
            output = forward(*args, **kwargs)
            values = output if isinstance(output, tuple) else (output,)
            traces.append((_digest(args[0]), tuple(_digest(value) for value in values)))
            return output

        monkeypatch.setattr(model, "forward", capture_forward)
        evidence: dict[str, Any] = {}
        with t.inference_mode():
            for factor, suffix in ((0.0, ""), (3.0, ""), (6.0, ""), (6.0, _SUFFIX)):
                with runtime.bind_cell(
                    method="NLSteering", concept_id=0, concept=concepts[0],
                    factor=factor, examples=examples,
                ) as bound:
                    cell = cast(Any, bound)
                    context = cell._context(0, suffix)
                    if suffix:
                        assert context.token_ids != runtime._originals[(0, 0)].prepared.token_ids
                        cell._cache.pop((0, suffix))
                        assert context == cell._context(0, suffix) == cell._context(0, suffix)
                    traces.clear()
                    actual = cell.evaluate(suffix, [0])[0].generation
                    actual_trace = tuple(traces)
                    traces.clear()
                    direct = generate_self_conditioning(
                        setup, model, runtime._target_tokenizer, context,
                        SelfCondScoreInterpolation(
                            target_scores=runtime._dense_diffmeans[0] if factor == 0 else targets[0],
                            negative_selection=setup.output_negative_selection,
                            weight=UnitInterval.parse(factor / 6),
                        ),
                        selection_seed=derive_selection_seed(42, 0, 0),
                        seed=derive_generation_seed(42, "NLSteering", 0, 0, factor),
                        max_new_tokens=96,
                    )
                    assert actual and actual_trace
                    assert (actual, actual_trace) == (direct, tuple(traces))
                    if suffix:
                        traces.clear()
                        assert cell.evaluate(suffix, [0])[0].generation == actual
                        assert tuple(traces) == actual_trace
                    evidence[f"{factor}:{bool(suffix)}"] = {"generation": actual, "trace": actual_trace}
        evidence["peak_cuda_bytes"] = t.cuda.max_memory_allocated()
        evidence["target_sha256"] = _digest(targets[0])
        evidence["fit_rows"] = {"positive": 72, "negative": 72}
        evidence["extraction_batch_size"] = 4
        (tmp_path / f"{checkpoint_name}-adv-nl-runtime.json").write_text(json.dumps(evidence, sort_keys=True))
