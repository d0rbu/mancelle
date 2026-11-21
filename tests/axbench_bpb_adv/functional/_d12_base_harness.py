"""Shared d12_base model harness for real-model GCG integration tests.

Provides the model loader, stub scorer, objective factory, and single-pair
attack-row builder shared by ``test_gcg_refactor_real_determinism.py`` and
``test_gcg_exact_speedup_integration.py``. Each test keeps its own ``_cfg()``
because the two tests need different ``FasterGcgConfig`` settings.

Leading underscore: this module is not collected by pytest.
"""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import torch as t

from nanochat.checkpoint_manager import load_model
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_gcg.edit_primitives import token_sequence_digest
from steering.eval.axbench_bpb.teacher_force import TargetModelBundle
from steering.eval.axbench_bpb.types import Intervention
from steering.eval.axbench_bpb_adv.model_optimize import BpbObjectiveCallbacks
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA

NANOCHAT_CACHE = Path.home() / ".cache" / "nanochat"
D12_DIR = NANOCHAT_CACHE / "base_checkpoints"
D12_TAG = "d12_base"
D12_STEP = 54600


def load_d12_base() -> TargetModelBundle:
    """Load the d12_base model from the shared cache.

    Sets ``NANOCHAT_BASE_DIR`` so the framework locates both the checkpoint
    and the tokenizer alongside it.
    """
    os.environ.setdefault("NANOCHAT_BASE_DIR", str(NANOCHAT_CACHE))
    model, tokenizer, _meta = load_model(
        source="base",
        device="cuda",
        phase="eval",
        model_tag=D12_TAG,
        step=D12_STEP,
        eval_conditioning=EvalConditioning.deferred("d12_base_harness"),
    )
    model.eval()
    sequence_len = getattr(model.config, "sequence_len", None)
    assert isinstance(sequence_len, int)
    return TargetModelBundle(
        model=model,
        tokenizer=tokenizer,
        max_seq_len=sequence_len,
        target_name="decoder_only",
    )


class StubPairScorer:
    """A pair scorer that always returns the no-op (none) intervention.

    Keeps tests self-contained (no concept artifacts) while still driving a
    real fwd+bwd through d12_base.
    """

    concepts_by_id = {1: object()}

    def _base_steered_intervention(self, **_kwargs: Any) -> Intervention:
        return Intervention.none()


def make_objective(
    bundle: TargetModelBundle,
    token_bytes: t.Tensor,
    device: t.device,
) -> BpbObjectiveCallbacks:
    """Construct a BpbObjectiveCallbacks backed by the real d12_base model."""
    return BpbObjectiveCallbacks(
        target_bundle=bundle,
        target_model_meta=None,
        tokenizer=bundle.tokenizer,
        token_bytes=token_bytes,
        source_paths=cast(Any, SimpleNamespace()),
        pair_scorer=cast(Any, StubPairScorer()),
        context_segments_by_prefix_hash={},
        device=device,
        lora=None,
    )


def single_pair_attack_rows(
    prompt_tokens: list[int],
    editable_start: int,
    editable_count: int,
) -> pa.Table:
    """One (rich, avoid) attack-row pair: the minimal non-degenerate group."""
    assert editable_start + editable_count == len(prompt_tokens)
    editable = prompt_tokens[editable_start : editable_start + editable_count]
    rich_output = [264, 412, 99]
    avoid_output = [171, 333]
    return pa.Table.from_pylist(
        [
            {
                "attack_row_id": "integ-row",
                "core_prefix_id": "coreprefix000001",
                "concept_rich_sample_id": "rich",
                "concept_avoiding_sample_id": "avoid",
                "concept_id": 1,
                "mode": "positive",
                "method_name": "diffmean_decoder_only",
                "target_model": "decoder_only",
                "coefficient_str": "1.0",
                "adversarial_method": "faster_gcg",
                "adversarial_strength": float(editable_count),
                "strength_unit": "optimized_tokens",
                "strength_index": 1,
                "context_signal_mode": "not_applicable",
                "prefix_text": "the quick brown fox",
                "prefix_hash": "hash",
                "source_prompt_tokens": prompt_tokens,
                "source_prompt_token_digest": token_sequence_digest(prompt_tokens),
                "concept_rich_output_tokens": rich_output,
                "concept_avoiding_output_tokens": avoid_output,
                "concept_rich_output_digest": token_sequence_digest(rich_output),
                "concept_avoiding_output_digest": token_sequence_digest(avoid_output),
                "editable_token_start": editable_start,
                "editable_token_count": editable_count,
                "original_editable_tokens": editable,
            }
        ],
        schema=ATTACK_ROWS_SCHEMA,
    )
