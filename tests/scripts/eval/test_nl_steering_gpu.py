"""Opt-in NL qualification on Concept500, real Gemma/SAE, and 3B decoders.

Set NL_STEERING_GPU_CONFIG to a JSON document containing prepared_dir,
receipt_path, description_texts_path, and checkpoint records named fixed/turn
with meta_path, statistics_path, statistics_sha256, and segment_mode. The
description source is the exact canonical JSON population hashed by the
producer receipt. Explicit qualification fails rather than skipping missing
assets. No model, tokenizer, extraction, generation, or CUDA kernel is faked.

Every Concept500 target is compared byte-for-byte with the frozen retrieval
arithmetic. Six evenly spaced concepts additionally exercise complete real
training contrasts, prompt extraction, and 96-token generation budgets at
factors 0, 3, and 6. This is numerical qualification, not a judged campaign.
"""

from __future__ import annotations

import copy
import gc
import hashlib
import json
import os
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch as t

from nanochat.tokenizer import get_tokenizer
from scripts.eval.axbench_upstream.merge_methods import FULL_FACTORS, NL_PROJECT_METHODS
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfConditionedFeatureProvider,
    SelfCondScoreInterpolation,
)
from steering.eval.axbench_upstream import methods, run_methods
from steering.eval.axbench_upstream.cell_store import CellResultStore
from steering.eval.axbench_upstream.data import load_metadata, select_training_texts
from steering.eval.axbench_upstream.nl_target_scores import build_retrieval_target_scores
from steering.eval.axbench_upstream.nl_target_source import load_nl_target_source_receipt
from steering.eval.axbench_upstream.seeds import derive_generation_seed, derive_selection_seed
from steering.eval.axbench_upstream.work_grid import CellKey, build_project_cell_grid
from steering.eval.chat_generation import render_prompt_for_chat_completion
from steering.refined_types import UnitInterval
from steering.segmentation import SegmentMode

pytestmark = pytest.mark.slow
CONCEPT_IDS = (0, 100, 200, 300, 400, 499)


def _digest(value: t.Tensor) -> str:
    tensor = value.detach().contiguous().cpu()
    assert bool(t.isfinite(tensor).all())
    return hashlib.sha256(
        str((tuple(tensor.shape), tensor.dtype)).encode() + tensor.view(t.uint8).numpy().tobytes()
    ).hexdigest()


@pytest.fixture(params=("fixed", "turn"))
def real_case(request):
    path = os.environ.get("NL_STEERING_GPU_CONFIG")
    if path is None:
        pytest.skip("set NL_STEERING_GPU_CONFIG to authenticated real qualification assets")
    assert t.cuda.is_available()
    gc.collect()
    t.cuda.empty_cache()
    config = json.loads(Path(path).read_text())
    checkpoint = config[request.param]
    statistics = Path(checkpoint["statistics_path"])
    assert hashlib.sha256(statistics.read_bytes()).hexdigest() == checkpoint["statistics_sha256"]
    setup = methods.load_method_setup(
        Path(checkpoint["meta_path"]),
        statistics,
        require_exact_statistics_identity=True,
    )
    assert setup.sae_model == "gemma-3-4b-pt" and setup.sae_layer == 22
    assert setup.n_sae_features == 65_536
    assert setup.segment_mode.value == checkpoint["segment_mode"]
    if request.param == "fixed":
        # The production loader above authenticates decoded-content identity;
        # the separately pinned raw digest also fixes serialization bytes.
        assert setup.prompt_statistics_raw_sha256 is None
    else:
        assert setup.turn_self_cond_segment_size_rustbpe == 33
        assert setup.prompt_statistics_raw_sha256 == checkpoint["statistics_sha256"]
    receipt = load_nl_target_source_receipt(Path(config["receipt_path"]))
    texts_bytes = Path(config["description_texts_path"]).read_bytes()
    assert (
        hashlib.sha256(texts_bytes).hexdigest()
        == receipt.description_embeddings.source_texts_sha256
    )
    description_texts = json.loads(texts_bytes)
    with np.load(receipt.description_embeddings.path, allow_pickle=False) as archive:
        assert archive["fids"].tolist() == [row[0] for row in description_texts]
        assert archive["emb"].shape == (len(description_texts), 768)
    metadata = load_metadata(Path(config["prepared_dir"]) / "generate/metadata.jsonl")
    assert len(metadata) == 500
    concepts = {index: row["concept"] for index, row in enumerate(metadata)}
    targets = build_retrieval_target_scores(
        setup=setup,
        description_embeddings_path=receipt.description_embeddings.path,
        concept_embeddings_path=receipt.concept_embeddings.path,
        concepts=concepts,
        expected_hashes=receipt.expected_hashes,
    )
    yield config, request.param, setup, receipt, concepts, targets
    gc.collect()
    t.cuda.empty_cache()


def _legacy_targets(setup, receipt, concepts):
    """Frozen branch arithmetic, independently constructing the unit table.

    Retain scalar float32 additions in ascending original-ID order: replacing
    this reduction with a matrix multiply changes the legacy bit pattern.
    """
    assert setup.feature_vocab is not None and setup.consolidation is not None
    original_ids = tuple(setup.feature_vocab.original_ids)
    with np.load(receipt.description_embeddings.path, allow_pickle=False) as archive:
        vectors = archive["emb"].astype(np.float32)
        vectors /= np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-8)
        descriptions = dict(zip(archive["fids"].tolist(), vectors, strict=True))
    rows = {fid: row for row, fid in enumerate(original_ids)}
    sums = np.zeros((len(original_ids), vectors.shape[1]), dtype=np.float32)
    counts = np.zeros(len(original_ids), dtype=np.int64)
    for fid in t.nonzero(setup.consolidation.allowed_universe).flatten().tolist():
        if fid in descriptions:
            row = rows[int(setup.consolidation.rep_of[fid])]
            sums[row] += descriptions[fid]
            counts[row] += 1
    norms = np.linalg.norm(sums, axis=1, keepdims=True)
    covered = (counts > 0) & (norms[:, 0] > 1e-8)
    assert covered.any()
    sums[covered] /= norms[covered]
    sums[~covered] = 0
    table = t.from_numpy(sums)
    with np.load(receipt.concept_embeddings.path, allow_pickle=False) as archive:
        queries = dict(
            zip(archive["texts"].tolist(), archive["emb"].astype(np.float32), strict=True)
        )
    result = {}
    for concept_id, concept in concepts.items():
        raw = table @ t.from_numpy(queries[concept])
        centered = raw - raw[covered].median()
        scale = centered[covered].abs().max()
        scores = centered / scale if scale > 0 else t.zeros_like(centered)
        scores[~covered] = 0
        result[concept_id] = scores.clamp(-1, 1)
    return result, covered


def test_real_concept500_retrieval_is_byte_exact(real_case, tmp_path):
    _, name, setup, receipt, concepts, targets = real_case
    reference, covered = _legacy_targets(setup, receipt, concepts)
    assert set(targets) == set(range(500))
    for concept_id, expected in reference.items():
        actual = targets[concept_id]
        assert actual.device.type == "cpu" and actual.dtype is t.float32
        assert actual.shape == (len(setup.feature_vocab.original_ids),)
        assert actual.is_contiguous() and bool(t.isfinite(actual).all())
        assert actual.numpy().tobytes() == expected.numpy().tobytes(), concept_id
        assert t.equal(actual[~covered], t.zeros_like(actual[~covered]))
        assert float(actual.abs().max()) == 1.0
    (tmp_path / f"{name}-all500-targets.json").write_text(
        json.dumps(
            {
                "covered_units": int(covered.sum()),
                "total_units": len(covered),
                "target_sha256": {cid: _digest(value) for cid, value in targets.items()},
            },
            sort_keys=True,
        )
    )


def _assert_real_old_cohorts_unchanged(
    setup, model, tokenizer, prepared, diffmeans, targets, monkeypatch, tmp_path
):
    """Real driver dispatch and Parquet resume, with unchanged physical batches."""
    frame = pd.DataFrame(
        {"concept_id": [0] * 12, "input_id": [0] * 12, "factor": list(FULL_FACTORS)}
    )
    old = build_project_cell_grid(frame, seed=42)
    combined = build_project_cell_grid(frame, seed=42, methods=NL_PROJECT_METHODS)
    old_set = set(old)
    assert tuple(cell for cell in combined if cell in old_set) == old
    rows = tuple(frame.itertuples(index=False))
    scalar, batched = run_methods._generate_project_cell, run_methods._generate_project_cells
    calls = []

    def capture_scalar(cell, **kwargs):
        result = scalar(cell, **kwargs)
        calls.append((cell,))
        return result

    def capture_batched(cells, **kwargs):
        result = batched(cells, **kwargs)
        calls.append(tuple(cells))
        return result

    monkeypatch.setattr(run_methods, "_generate_project_cell", capture_scalar)
    monkeypatch.setattr(run_methods, "_generate_project_cells", capture_batched)

    def run(label, cells, existing):
        root = tmp_path / label
        store_args: dict[str, Any] = {
            "root": root,
            "cells": cells,
            "tripwire_cells": cells,
            "semantic_identity": {"qualification": label},
            "rank": 0,
            "world_size": 1,
        }
        # Close and reopen to exercise actual persisted resume batches.
        with CellResultStore(**store_args) as store:
            for cell, generation in existing.items():
                store.append(cell, generation)
        calls.clear()
        with CellResultStore(**store_args) as store:
            run_methods._append_missing_project_cells(
                store,
                cells,
                rows=rows,
                rendered_tokens_by_pair={(0, 0): prepared[0].token_ids},
                prepared_by_tokens={prepared[0].token_ids: prepared[0]},
                dense_diffmeans=diffmeans,
                nl_targets=targets,
                setup=setup,
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=34,
            )
            results = store.complete_results()
        with CellResultStore(**store_args) as store:
            assert store.complete_results() == results
            assert store.missing() == ()
        return results, tuple(call for call in calls if call[0] in old_set)

    with t.inference_mode():
        original, original_calls = run("old-full-grid", old, {})
        enabled, enabled_calls = run("nl-preserves-full-grid", combined, {})
        expected_calls = (
            (old,)
            if setup.segment_mode is SegmentMode.fixed_tokens
            else tuple((cell,) for cell in old)
        )
        assert original_calls == enabled_calls == expected_calls
        assert original == {cell: enabled[cell] for cell in old}
        existing = {cell: original[cell] for cell in old[:-1]}
        resumed, resumed_calls = run("old-resumed-singleton", old, existing)
        nl_resumed, nl_resumed_calls = run("nl-preserves-resumed-singleton", combined, existing)
        assert resumed_calls == nl_resumed_calls == ((old[-1],),)
        assert resumed == {cell: nl_resumed[cell] for cell in old}
    return {
        "old_cells": len(old),
        "old_batch_width": len(old) if setup.segment_mode is SegmentMode.fixed_tokens else 1,
        "resumed_old_cells": 1,
        "nl_cells": len(combined) - len(old),
        "old_generation_exact": True,
    }


def test_real_3b_generation_matches_existing_adapter_and_selection_oracle(
    real_case,
    monkeypatch,
    tmp_path,
):
    config, name, setup, receipt, concepts, targets = real_case
    reference, _ = _legacy_targets(setup, receipt, concepts)
    prepared_dir = Path(config["prepared_dir"])
    training = select_training_texts(prepared_dir, max_concepts=500)
    evaluation = pd.read_parquet(prepared_dir / "generate/latent_eval_data.parquet")
    assert evaluation["concept_id"].nunique() == 500 and len(evaluation) >= 38_000
    tokenizer = get_tokenizer()
    base, sae = methods.load_extraction_stack(setup, "cuda")
    prepared, diffmeans, fit_counts = {}, {}, {}
    with t.inference_mode():
        for cid in CONCEPT_IDS:
            selected = training[cid]
            assert selected.positive and selected.negative
            fit_counts[cid] = (len(selected.positive), len(selected.negative))
            diffmeans[cid] = methods.fit_dense_diffmean(
                setup,
                base,
                sae,
                selected.positive,
                selected.negative,
                concept_id=cid,
                extraction_batch_size=8,
            )
            prompt = evaluation.loc[evaluation["concept_id"].eq(cid), "input"].iloc[0]
            rendered = render_prompt_for_chat_completion(tokenizer, prompt)
            prepared[cid] = methods.prepare_prompt_context(
                setup,
                base,
                sae,
                tokenizer,
                token_ids=rendered.token_ids,
                assistant_end_id=rendered.assistant_end_id,
            )
            assert prepared[cid].token_ids == tuple(rendered.token_ids)
            assert prepared[cid].segment_mode is setup.segment_mode
            assert prepared[cid].segments
    # Avoid requiring the extraction model and 3B decoder to coexist in VRAM.
    if hasattr(base, "nnterp_model"):
        del base.nnterp_model
    del base, sae
    gc.collect()
    t.cuda.empty_cache()
    model, tokenizer = methods.load_target_model(setup, "cuda")
    assert sum(parameter.numel() for parameter in model.parameters()) >= 2_500_000_000
    assert next(model.parameters()).device.type == "cuda"
    original_ids = tuple(setup.feature_vocab.original_ids)
    traces, selections, score_rows = [], [], []
    forward = model.forward

    def capture_forward(*args, **kwargs):
        output = forward(*args, **kwargs)
        values = output if isinstance(output, tuple) else (output,)
        traces.append((_digest(args[0]), tuple(_digest(value) for value in values)))
        return output

    monkeypatch.setattr(model, "forward", capture_forward)
    step = SelfConditionedFeatureProvider._step_score_interpolation
    finalize = SelfConditionedFeatureProvider._finalize_two_sided_selection

    def capture_step(provider, state, fp_logits_tail, **kwargs):
        assert provider.cfg.pool_mode.value == "max_pool"
        pooled = fp_logits_tail.amax(dim=1).detach().float()
        policy = provider.cfg.score_policy
        target = policy.target_scores.to(pooled.device).unsqueeze(0)
        weight = float(policy.weight)
        scores = (
            pooled
            if weight == 0
            else target
            if weight == 1
            else t.lerp(
                2 * t.sigmoid(pooled) - 1,
                target,
                weight,
            )
        )
        score_rows.append(scores[0].cpu().tolist())
        return step(provider, state, fp_logits_tail, **kwargs)

    def capture_selection(provider, state, positive_ids, negative_ids, **kwargs):
        positive, negative = positive_ids.tolist(), negative_ids.tolist()
        scores = dict(zip(original_ids, score_rows[-1], strict=True))
        assert positive == sorted(original_ids, key=lambda fid: -scores[fid])[: len(positive)]
        remaining = [fid for fid in original_ids if fid not in positive]
        assert negative == sorted(remaining, key=lambda fid: scores[fid])[: len(negative)]
        assert not set(positive) & set(negative)
        result = finalize(provider, state, positive_ids, negative_ids, **kwargs)
        buckets = {**dict.fromkeys(positive, 0), **dict.fromkeys(negative, 1)}
        assert provider.last_appended_segment == buckets
        assert result.target_slot_idx == state.target_slot_idx + 1
        selections.append((len(traces), positive, negative, buckets))
        return result

    monkeypatch.setattr(SelfConditionedFeatureProvider, "_step_score_interpolation", capture_step)
    monkeypatch.setattr(
        SelfConditionedFeatureProvider, "_finalize_two_sided_selection", capture_selection
    )
    evidence: dict[str, Any] = {"fit_rows": fit_counts, "cells": {}}
    with t.inference_mode():
        for cid in CONCEPT_IDS:
            for factor in (0.0, 3.0, 6.0):
                selection_seed = derive_selection_seed(42, cid, 0)
                token_seed = derive_generation_seed(42, "NLSteering", cid, 0, factor)
                cell = CellKey(
                    "project", "NLSteering", 0, cid, 0, factor.hex(), selection_seed, token_seed
                )
                actual_policy = run_methods._project_score_policy(
                    cell,
                    setup=setup,
                    diffmean=diffmeans[cid],
                    factor=factor,
                    nl_target=targets[cid],
                )
                # Zero compares against the actual old DiffMean policy; other
                # strengths compare the unchanged adapter with legacy targets.
                expected_policy = SelfCondScoreInterpolation(
                    target_scores=diffmeans[cid] if factor == 0 else reference[cid],
                    negative_selection=setup.output_negative_selection,
                    weight=UnitInterval.parse(factor / 6),
                )
                records = []
                for policy in (expected_policy, actual_policy):
                    traces.clear()
                    selections.clear()
                    score_rows.clear()
                    generation = methods.generate_self_conditioning(
                        setup,
                        model,
                        tokenizer,
                        prepared[cid],
                        policy,
                        selection_seed=selection_seed,
                        seed=token_seed,
                        max_new_tokens=96,
                    )
                    assert traces and selections
                    records.append(copy.deepcopy((generation, traces, selections)))
                assert records[0] == records[1], (name, cid, factor)
                evidence["cells"][f"{cid}:{factor}"] = records[1]
    if setup.segment_mode is SegmentMode.turn_delimited_fixed_tokens:
        refreshes = [entry[2] for entry in evidence["cells"].values()]
        assert any(len(entries) >= 3 for entries in refreshes), (
            "must cross two real 33-token refresh boundaries"
        )
        for entries in refreshes:
            positions = [entry[0] for entry in entries]
            assert all(right - left == 33 for left, right in pairwise(positions))
    else:
        assert all(len(entry[2]) == 1 for entry in evidence["cells"].values())
    assert any(
        evidence["cells"][f"{cid}:0.0"][2] != evidence["cells"][f"{cid}:6.0"][2]
        for cid in CONCEPT_IDS
    )
    # The additional driver gate includes masking and batched forwards;
    # restore scalar-interpolation-specific capture before entering it.
    monkeypatch.setattr(model, "forward", forward)
    monkeypatch.setattr(SelfConditionedFeatureProvider, "_step_score_interpolation", step)
    monkeypatch.setattr(SelfConditionedFeatureProvider, "_finalize_two_sided_selection", finalize)
    evidence["old_cohorts"] = _assert_real_old_cohorts_unchanged(
        setup,
        model,
        tokenizer,
        prepared,
        diffmeans,
        targets,
        monkeypatch,
        tmp_path,
    )
    evidence["peak_cuda_bytes"] = t.cuda.max_memory_allocated()
    (tmp_path / f"{name}-real-generation.json").write_text(json.dumps(evidence, sort_keys=True))
