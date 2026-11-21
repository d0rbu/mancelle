"""Correctness contracts for Faster-GCG shared-prefix KV candidate scoring."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pyarrow as pa
import pytest
import torch as t

import nanochat.flash_attention as flash_attention
import steering.eval.axbench_bpb_adv.model_optimize as model_optimize
import steering.eval.axbench_bpb_adv.surrogate as surrogate
from nanochat.engine import KVCache
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import GPT, GPTConfig
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_gcg.edit_primitives import (
    EditableSpan,
    token_sequence_digest,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.axbench_base.config import LoraConfig
from steering.eval.axbench_bpb.teacher_force import (
    TargetModelBundle,
    build_output_only_mask,
)
from steering.eval.axbench_bpb.types import Intervention, LoraParams
from steering.eval.axbench_bpb_adv.config import FasterGcgConfig
from steering.eval.axbench_bpb_adv.errors import AxBenchBpbAdvError
from steering.eval.axbench_bpb_adv.model_optimize import BpbObjectiveCallbacks
from steering.eval.axbench_bpb_adv.optimize import build_universal_gcg_candidates
from steering.eval.axbench_bpb_adv.schemas import ATTACK_ROWS_SCHEMA
from steering.eval.axbench_bpb_adv.surrogate import (
    _build_teacher_forced_layout,
    _canonical_shared_prefix_device,
    _new_shared_prefix_cache,
    _prepare_shared_prefix_groups,
    teacher_forced_bpb_loss_shared_prefix_kv_batch,
)
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from tests.axbench_bpb_adv.full_sequence_reference import (
    teacher_forced_bpb_loss_full_sequence_reference,
)
from tests.conftest import (
    TINY_N_EMBD,
    TINY_N_SAE_FEATURES,
    TINY_NUM_BUCKETS,
    TINY_VOCAB,
    _gpt_blocks,
    _make_encdec,
    _make_gpt,
)


class _CharTokenizer:
    bos_token_id = 0

    def decode(self, ids: list[int]) -> str:
        return "".join(chr(int(token_id)) for token_id in ids if int(token_id) != 0)


@pytest.fixture(scope="module", autouse=True)
def _force_sdpa() -> Iterator[None]:
    previous = flash_attention._override_impl
    flash_attention._override_impl = FlashAttnImpl.SDPA
    try:
        yield
    finally:
        flash_attention._override_impl = previous


@pytest.fixture(scope="module")
def decoder_bundle() -> TargetModelBundle:
    t.manual_seed(1227)
    model = _make_gpt()
    # GPT.init_weights deliberately zero-initializes attention output
    # projections. Make the tiny fixture cache-sensitive so exact parity cannot
    # pass vacuously while K/V is ignored.
    with t.no_grad():
        generator = t.Generator().manual_seed(1227)
        for block in _gpt_blocks(model):
            block.attn.c_proj.weight.copy_(
                t.randn(
                    block.attn.c_proj.weight.shape,
                    generator=generator,
                    dtype=block.attn.c_proj.weight.dtype,
                )
                * 0.02
            )
    return TargetModelBundle(
        model=model,
        tokenizer=_CharTokenizer(),
        max_seq_len=64,
        target_name="decoder_only",
    )


@pytest.fixture(scope="module")
def encdec_bundle() -> TargetModelBundle:
    t.manual_seed(1228)
    model = _make_encdec()
    with t.no_grad():
        generator = t.Generator().manual_seed(1228)
        for block in model._decoder_blocks:
            block.attn.c_proj.weight.copy_(
                t.randn(
                    block.attn.c_proj.weight.shape,
                    generator=generator,
                    dtype=block.attn.c_proj.weight.dtype,
                )
                * 0.02
            )
    return TargetModelBundle(
        model=model,
        tokenizer=_CharTokenizer(),
        max_seq_len=64,
        target_name="encdec",
    )


def _segments(
    prompts: list[list[int]],
) -> list[tuple[ContextFeatureSegment, ...]]:
    tokenizer = _CharTokenizer()
    return [
        (
            ContextFeatureSegment(
                feature_ids=(1, 2),
                buckets=(1, 2),
                char_start=0,
                char_end=len(tokenizer.decode(prompt)),
            ),
        )
        for prompt in prompts
    ]


def _losses(
    *,
    bundle: TargetModelBundle,
    prompts: list[list[int]],
    source_prompts: list[list[int]],
    outputs: list[list[int]],
    spans: list[EditableSpan],
    intervention: Intervention,
    batch_size: int,
    segments: list[tuple[ContextFeatureSegment, ...]] | None = None,
    lora: LoraParams | None = None,
) -> tuple[list[float], list[float]]:
    context_segments = segments or [() for _ in prompts]
    common = {
        "target_bundle": bundle,
        "prompts": prompts,
        "outputs": outputs,
        "prefix_context_segments_list": context_segments,
        "output_token_bytes": t.ones(TINY_VOCAB, dtype=t.int64),
        "intervention": intervention,
        "device": t.device("cpu"),
        "device_batch_size": batch_size,
        "lora": lora,
    }
    full = teacher_forced_bpb_loss_full_sequence_reference(
        **common  # ty: ignore[invalid-argument-type]
    )
    shared = teacher_forced_bpb_loss_shared_prefix_kv_batch(
        source_prompts=source_prompts,
        editable_spans=spans,
        **common,  # ty: ignore[invalid-argument-type]
    )
    return full, shared


@pytest.mark.parametrize("k", [1, 2, 4, 8, 16, 32])
def test_real_gpt_exact_parity_for_every_production_k(
    decoder_bundle: TargetModelBundle,
    k: int,
) -> None:
    fixed = [2, 3, 4]
    source = [*fixed, *[(20 + index) % TINY_VOCAB for index in range(k)]]
    prompts = [
        [*fixed, *[(40 + offset + index) % TINY_VOCAB for index in range(k)]]
        for offset in (0, 7, 13)
    ]
    outputs = [[90], [91, 92], [93, 94, 95]]
    full, shared = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=outputs,
        spans=[EditableSpan(start=len(fixed), count=k) for _ in prompts],
        intervention=Intervention.none(),
        batch_size=2,
    )
    assert shared == full


@pytest.mark.parametrize("k", [1, 4, 32])
def test_real_encdec_exact_parity_for_multiple_k_and_mixed_lengths(
    encdec_bundle: TargetModelBundle,
    k: int,
) -> None:
    fixed = [2, 3, 4]
    source = [*fixed, *[(20 + index) % TINY_VOCAB for index in range(k)]]
    prompts = [
        [*fixed, *[(50 + offset + index) % TINY_VOCAB for index in range(k)]]
        for offset in (0, 9, 17)
    ]
    outputs = [[90, 91], [92], [93, 94, 95]]
    full, shared = _losses(
        bundle=encdec_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=outputs,
        spans=[EditableSpan(start=len(fixed), count=k) for _ in prompts],
        intervention=Intervention.ours(feature_conditions={3: 1, 4: 2}),
        batch_size=2,
        segments=_segments(prompts),
    )
    assert shared == full


def test_real_gpt_output_token_losses_are_exact_not_only_reduced_bpb(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exact scalar BPB parity must not hide offsetting per-token errors."""

    prompt = [2, 3, 20, 21]
    source = [2, 3, 10, 11]
    output = [60, 61, 62]
    layout = _build_teacher_forced_layout(
        target_bundle=decoder_bundle,
        prompt_tokens=prompt,
        output_tokens=output,
        prefix_context_segments=(),
        intervention=Intervention.none(),
    )
    original_forward = decoder_bundle.model.forward
    recorded_losses: list[t.Tensor] = []

    def _recording_forward(*args: Any, **kwargs: Any) -> Any:
        result = original_forward(*args, **kwargs)
        if kwargs.get("loss_reduction") == "none":
            recorded_losses.append(result.detach().clone())
        return result

    monkeypatch.setattr(decoder_bundle.model, "forward", _recording_forward)
    full, shared = _losses(
        bundle=decoder_bundle,
        prompts=[prompt],
        source_prompts=[source],
        outputs=[output],
        spans=[EditableSpan(2, 2)],
        intervention=Intervention.none(),
        batch_size=1,
    )
    assert shared == full
    assert len(recorded_losses) == 3  # full reference, prefill, cached suffix
    full_mask = build_output_only_mask(
        n_full_prefix=PositiveInt.parse(layout.n_full_prefix),
        n_output=PositiveInt.parse(len(output)),
    )
    suffix_mask = build_output_only_mask(
        n_full_prefix=PositiveInt.parse(2),
        n_output=PositiveInt.parse(len(output)),
    )
    full_output_losses = recorded_losses[0].view(-1)[full_mask]
    cached_output_losses = recorded_losses[2].view(-1)[suffix_mask]
    assert t.equal(cached_output_losses, full_output_losses)


def test_sliding_window_decoder_layers_preserve_exact_cached_parity(
    decoder_bundle: TargetModelBundle,
) -> None:
    original_windows = list(decoder_bundle.model.window_sizes)
    decoder_bundle.model.window_sizes = [(2, 0), original_windows[-1]]
    try:
        fixed = [2, 3, 4, 5, 6, 7, 8, 9]
        source = [*fixed, 10, 11]
        prompts = [[*fixed, 20, 21], [*fixed, 22, 23]]
        full, shared = _losses(
            bundle=decoder_bundle,
            prompts=prompts,
            source_prompts=[source, source],
            outputs=[[60, 61, 62], [63, 64]],
            spans=[EditableSpan(len(fixed), 2), EditableSpan(len(fixed), 2)],
            intervention=Intervention.none(),
            batch_size=2,
        )
        assert shared == full
    finally:
        decoder_bundle.model.window_sizes = original_windows


def test_encdec_prompt_prefix_multiple_segments_preserves_exact_routing(
    encdec_bundle: TargetModelBundle,
) -> None:
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20, 21], [2, 3, 22, 23]]
    segments: list[tuple[ContextFeatureSegment, ...]] = [
        (
            ContextFeatureSegment((1,), (1,), 0, 2),
            ContextFeatureSegment((2,), (2,), 2, 4),
        )
        for _ in prompts
    ]
    full, shared = _losses(
        bundle=encdec_bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=[[60, 61], [62]],
        spans=[EditableSpan(2, 2), EditableSpan(2, 2)],
        intervention=Intervention.prompting(
            target_model="encdec", prompt_prefix_tokens=[70, 71]
        ),
        batch_size=2,
        segments=segments,
    )
    assert shared == full


def test_compacted_encdec_checkpoint_preserves_exact_cached_parity() -> None:
    base = _make_encdec()
    vocab = FeatureVocab(
        n_sae_features=int(TINY_N_SAE_FEATURES),
        original_ids=(1, 2, 3, 4),
    )
    model = EncoderDecoderGPT(
        dec_config=base.dec_config,
        enc_config=base.enc_config,
        xattn_config=base.xattn_config,
        n_sae_features=int(TINY_N_SAE_FEATURES),
        num_buckets=int(TINY_NUM_BUCKETS),
        pad_vocab_size_to=TINY_VOCAB,
        feature_vocab=vocab,
    )
    model.init_weights()
    with t.no_grad():
        generator = t.Generator().manual_seed(1229)
        for block in model._decoder_blocks:
            block.attn.c_proj.weight.copy_(
                t.randn(
                    block.attn.c_proj.weight.shape,
                    generator=generator,
                    dtype=block.attn.c_proj.weight.dtype,
                )
                * 0.02
            )
    bundle = TargetModelBundle(
        model=model,
        tokenizer=_CharTokenizer(),
        max_seq_len=64,
        target_name="encdec",
    )
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20, 21], [2, 3, 22, 23]]
    full, shared = _losses(
        bundle=bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=[[60, 61], [62]],
        spans=[EditableSpan(2, 2), EditableSpan(2, 2)],
        intervention=Intervention.ours(feature_conditions={3: 1, 4: 2}),
        batch_size=2,
        segments=_segments(prompts),
    )
    assert shared == full


def test_tiny_real_gpt_full_gcg_trajectory_matches_test_oracle_exactly(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Exercise initialization, gradients, ranking, acceptance, and retention."""

    prompt = [0, 2, 3, 10, 11]
    rich = [60, 61]
    avoid = [62, 63]
    row = {
        "attack_row_id": "shared-prefix-trajectory-row",
        "core_prefix_id": "shared-prefix-trajectory-prefix",
        "concept_rich_sample_id": "rich",
        "concept_avoiding_sample_id": "avoid",
        "concept_id": 1,
        "mode": "positive",
        "method_name": "diffmean_decoder_only",
        "target_model": "decoder_only",
        "coefficient_str": "1.0",
        "adversarial_method": "faster_gcg",
        "adversarial_strength": 2.0,
        "strength_unit": "optimized_tokens",
        "strength_index": 1,
        "context_signal_mode": "not_applicable",
        "prefix_text": "tiny trajectory",
        "prefix_hash": "tiny-trajectory-hash",
        "source_prompt_tokens": prompt,
        "source_prompt_token_digest": token_sequence_digest(prompt),
        "concept_rich_output_tokens": rich,
        "concept_avoiding_output_tokens": avoid,
        "concept_rich_output_digest": token_sequence_digest(rich),
        "concept_avoiding_output_digest": token_sequence_digest(avoid),
        "editable_token_start": len(prompt) - 2,
        "editable_token_count": 2,
        "original_editable_tokens": prompt[-2:],
    }
    attack_rows = pa.Table.from_pylist([row], schema=ATTACK_ROWS_SCHEMA)
    cfg = FasterGcgConfig(
        optimized_token_counts=(2,),
        max_steps=2,
        candidate_batch_size=4,
        top_k=8,
        distance_weight=10.0,
        temperature=0.1,
        master_seed=1227,
        universal_minibatch_size=1,
    )

    class _NoopPairScorer:
        concepts_by_id = {1: object()}

        def _base_steered_intervention(self, **_kwargs: Any) -> Intervention:
            return Intervention.none()

    def _callbacks() -> BpbObjectiveCallbacks:
        return BpbObjectiveCallbacks(
            target_bundle=decoder_bundle,
            target_model_meta=None,
            tokenizer=decoder_bundle.tokenizer,
            token_bytes=t.ones(TINY_VOCAB, dtype=t.int64),
            source_paths=cast(Any, SimpleNamespace()),
            pair_scorer=cast(Any, _NoopPairScorer()),
            context_segments_by_prefix_hash={},
            device=t.device("cpu"),
            device_batch_size=2,
        )

    def _run() -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        callbacks = _callbacks()
        events: list[dict[str, Any]] = []
        candidates, triggers = build_universal_gcg_candidates(
            attack_rows=attack_rows,
            tokenizer=decoder_bundle.tokenizer,
            cfg=cfg,
            embedding_weight=callbacks.embedding_weight(),
            allowed_token_ids=list(range(2, TINY_VOCAB)),
            objective=callbacks.objective_batch,
            gradient_provider_batch=callbacks.gradient_provider_batch,
            progress_callback=events.append,
            group_boundary_callback=callbacks.clear_eval_cache,
        )
        return candidates.to_pylist(), triggers, events

    def _full_reference_adapter(**kwargs: Any) -> list[float]:
        kwargs.pop("source_prompts")
        kwargs.pop("editable_spans")
        return teacher_forced_bpb_loss_full_sequence_reference(**kwargs)

    with monkeypatch.context() as reference_patch:
        reference_patch.setattr(
            model_optimize,
            "teacher_forced_bpb_loss_shared_prefix_kv_batch",
            _full_reference_adapter,
        )
        reference = _run()
    shared = _run()
    assert shared == reference


def test_prompting_prefix_and_nondivisible_chunking_are_exact(
    decoder_bundle: TargetModelBundle,
) -> None:
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20 + 2 * index, 21 + 2 * index] for index in range(5)]
    outputs = [[60 + index] * (1 + index % 3) for index in range(5)]
    full, shared = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=outputs,
        spans=[EditableSpan(2, 2) for _ in prompts],
        intervention=Intervention.prompting(
            target_model="decoder_only", prompt_prefix_tokens=[70, 71]
        ),
        batch_size=2,
    )
    assert shared == full


def test_direction_hook_is_active_across_prefill_and_suffix(
    decoder_bundle: TargetModelBundle,
) -> None:
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20, 21], [2, 3, 22, 23]]
    outputs = [[60, 61], [62]]
    direction = t.linspace(-1.0, 1.0, TINY_N_EMBD)
    intervention = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=direction,
        direction_layer=0,
        coefficient=3.0,
    )
    full, shared = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=outputs,
        spans=[EditableSpan(2, 2), EditableSpan(2, 2)],
        intervention=intervention,
        batch_size=2,
    )
    no_intervention, _ = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=outputs,
        spans=[EditableSpan(2, 2), EditableSpan(2, 2)],
        intervention=Intervention.none(),
        batch_size=2,
    )
    assert shared == full
    assert shared != no_intervention


def test_lora_parity_and_teardown() -> None:
    fixture_root = Path(__file__).parents[2] / "axbench_bpb" / "fixtures"
    checkpoint = t.load(
        fixture_root / "tiny_decoder_only" / "ckpt.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = GPTConfig(**checkpoint["config"])
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=checkpoint["pad_vocab_size_to"])
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(checkpoint["model"], strict=True, assign=True)
    model.eval()
    bundle = TargetModelBundle(
        model=model,
        tokenizer=_CharTokenizer(),
        max_seq_len=config.sequence_len,
        target_name="decoder_only",
    )
    adapter_path = (
        fixture_root
        / "tiny_decoder_only"
        / "lora_adapter_concept_0001.pt"
    )
    lora = LoraParams(cfg=LoraConfig(rank=4, alpha=8.0), layer=1)
    before = {
        name: tensor.detach().clone()
        for name, tensor in model.state_dict().items()
    }
    full, shared = _losses(
        bundle=bundle,
        prompts=[[2, 3, 20, 21], [2, 3, 22, 23]],
        source_prompts=[[2, 3, 10, 11], [2, 3, 10, 11]],
        outputs=[[60, 61], [62]],
        spans=[EditableSpan(2, 2), EditableSpan(2, 2)],
        intervention=Intervention.lora(adapter_path=adapter_path),
        batch_size=2,
        lora=lora,
    )

    assert shared == full
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(t.equal(before[name], after[name]) for name in before)


def test_result_order_is_restored_across_compatibility_groups(
    decoder_bundle: TargetModelBundle,
) -> None:
    # Different output lengths force different left truncation and therefore
    # different shared-prefix lengths/groups. Interleave the two groups.
    short_bundle = TargetModelBundle(
        model=decoder_bundle.model,
        tokenizer=decoder_bundle.tokenizer,
        max_seq_len=7,
        target_name="decoder_only",
    )
    source = [2, 3, 4, 5, 10, 11]
    prompts = [
        [2, 3, 4, 5, 20, 21],
        [2, 3, 4, 5, 22, 23],
        [2, 3, 4, 5, 24, 25],
        [2, 3, 4, 5, 26, 27],
    ]
    outputs = [[60], [61, 62, 63], [64], [65, 66, 67]]
    prepared, groups = _prepare_shared_prefix_groups(
        target_bundle=short_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=outputs,
        prefix_context_segments_list=[() for _ in prompts],
        editable_spans=[EditableSpan(4, 2) for _ in prompts],
        intervention=Intervention.none(),
    )
    assert len(prepared) == 4
    assert len(groups) == 2
    assert [group.sample_indices for group in groups] == [[0, 2], [1, 3]]
    full, shared = _losses(
        bundle=short_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=outputs,
        spans=[EditableSpan(4, 2) for _ in prompts],
        intervention=Intervention.none(),
        batch_size=2,
    )
    assert shared == full


def test_no_truncation_rich_avoid_share_one_encdec_prefill(
    encdec_bundle: TargetModelBundle,
) -> None:
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20, 21], [2, 3, 22, 23]]
    outputs = [[60], [61, 62]]
    _prepared, groups = _prepare_shared_prefix_groups(
        target_bundle=encdec_bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=outputs,
        prefix_context_segments_list=_segments(prompts),
        editable_spans=[EditableSpan(2, 2), EditableSpan(2, 2)],
        intervention=Intervention.ours(feature_conditions={3: 1}),
    )
    assert len(groups) == 1


def test_encdec_rich_avoid_truncation_splits_exact_compatibility_groups(
    encdec_bundle: TargetModelBundle,
) -> None:
    short_bundle = TargetModelBundle(
        model=encdec_bundle.model,
        tokenizer=encdec_bundle.tokenizer,
        max_seq_len=7,
        target_name="encdec",
    )
    source = [2, 3, 4, 5, 10, 11]
    prompts = [[2, 3, 4, 5, 20, 21], [2, 3, 4, 5, 20, 21]]
    outputs = [[60], [61, 62, 63]]
    segments = _segments(prompts)
    _prepared, groups = _prepare_shared_prefix_groups(
        target_bundle=short_bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=outputs,
        prefix_context_segments_list=segments,
        editable_spans=[EditableSpan(4, 2), EditableSpan(4, 2)],
        intervention=Intervention.ours(feature_conditions={3: 1}),
    )
    assert len(groups) == 2
    full, shared = _losses(
        bundle=short_bundle,
        prompts=prompts,
        source_prompts=[source, source],
        outputs=outputs,
        spans=[EditableSpan(4, 2), EditableSpan(4, 2)],
        intervention=Intervention.ours(feature_conditions={3: 1}),
        batch_size=2,
        segments=segments,
    )
    assert shared == full


def test_bare_cuda_device_resolves_to_indexed_model_device() -> None:
    assert _canonical_shared_prefix_device(
        requested=t.device("cuda"),
        model_device=t.device("cuda:0"),
    ) == t.device("cuda:0")
    assert _canonical_shared_prefix_device(
        requested=t.device("cuda"),
        model_device=t.device("cuda:3"),
    ) == t.device("cuda:3")
    with pytest.raises(AxBenchBpbAdvError, match="cuda:1.*cuda:0"):
        _canonical_shared_prefix_device(
            requested=t.device("cuda:0"),
            model_device=t.device("cuda:1"),
        )


def test_compacted_vocabulary_cached_key_is_built_once_per_call(
    encdec_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    allowed_ids = t.tensor([1, 2, 3, 4], dtype=t.int64)
    monkeypatch.setattr(
        surrogate,
        "feature_vocab_allowed_ids",
        lambda _model: allowed_ids,
    )
    original_cached_key = surrogate._cpu_tensor_cached_structural_key
    vocabulary_key_calls = 0

    def _recording_cached_key(
        tensor: t.Tensor,
    ) -> surrogate._CachedCpuTensorStructuralKey:
        nonlocal vocabulary_key_calls
        vocabulary_key_calls += 1
        return original_cached_key(tensor)

    monkeypatch.setattr(
        surrogate,
        "_cpu_tensor_cached_structural_key",
        _recording_cached_key,
    )
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20 + 2 * index, 21 + 2 * index] for index in range(6)]
    prepared, groups = _prepare_shared_prefix_groups(
        target_bundle=encdec_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=[[60 + index] for index in range(6)],
        prefix_context_segments_list=_segments(prompts),
        editable_spans=[EditableSpan(2, 2) for _ in prompts],
        intervention=Intervention.ours(feature_conditions={3: 1}),
    )

    assert len(prepared) == 6
    assert groups
    assert vocabulary_key_calls == 1


@pytest.mark.parametrize(
    ("prompts", "source_prompts", "outputs", "spans", "max_seq_len", "match"),
    [
        ([[2, 3]], [[2, 3]], [[4]], [EditableSpan(2, 0)], 8, "k > 0"),
        ([[2, 3, 4]], [[2, 3, 4]], [[5]], [EditableSpan(1, 1)], 8, "prompt boundary"),
        ([[2, 3, 4]], [[2, 3]], [[5]], [EditableSpan(1, 2)], 8, "changed prompt length"),
        ([[9, 3, 4]], [[2, 3, 4]], [[5]], [EditableSpan(1, 2)], 8, "fixed prefix"),
        ([[2, 3, 4, 5]], [[2, 3, 4, 5]], [[6, 7, 8, 9]], [EditableSpan(2, 2)], 5, "dropped"),
        ([[2, 3]], [[2, 3]], [[4]], [EditableSpan(0, 2)], 8, "no fixed-prefix"),
    ],
)
def test_invalid_shared_prefix_layouts_fail_loudly(
    decoder_bundle: TargetModelBundle,
    prompts: list[list[int]],
    source_prompts: list[list[int]],
    outputs: list[list[int]],
    spans: list[EditableSpan],
    max_seq_len: int,
    match: str,
) -> None:
    bundle = TargetModelBundle(
        model=decoder_bundle.model,
        tokenizer=decoder_bundle.tokenizer,
        max_seq_len=max_seq_len,
        target_name="decoder_only",
    )
    with pytest.raises(AxBenchBpbAdvError, match=match):
        _losses(
            bundle=bundle,
            prompts=prompts,
            source_prompts=source_prompts,
            outputs=outputs,
            spans=spans,
            intervention=Intervention.none(),
            batch_size=2,
        )


def test_inconsistent_k_and_malformed_lengths_fail_before_forward(
    decoder_bundle: TargetModelBundle,
) -> None:
    with pytest.raises(AxBenchBpbAdvError, match="inconsistent"):
        _prepare_shared_prefix_groups(
            target_bundle=decoder_bundle,
            prompts=[[2, 3, 4], [2, 3, 4]],
            source_prompts=[[2, 3, 4], [2, 3, 4]],
            outputs=[[5], [6]],
            prefix_context_segments_list=[(), ()],
            editable_spans=[EditableSpan(2, 1), EditableSpan(1, 2)],
            intervention=Intervention.none(),
        )
    with pytest.raises(AxBenchBpbAdvError, match="malformed batch lengths"):
        _prepare_shared_prefix_groups(
            target_bundle=decoder_bundle,
            prompts=[[2, 3, 4]],
            source_prompts=[],
            outputs=[[5]],
            prefix_context_segments_list=[()],
            editable_spans=[EditableSpan(2, 1)],
            intervention=Intervention.none(),
        )


def test_destination_fanout_is_materialized_and_rows_are_independent(
    decoder_bundle: TargetModelBundle,
) -> None:
    source = _new_shared_prefix_cache(
        target_bundle=decoder_bundle,
        batch_size=1,
        max_seq_len=8,
        device=t.device("cpu"),
    )
    source.k_cache[:, :, :3].copy_(
        t.arange(source.k_cache[:, :, :3].numel(), dtype=source.k_cache.dtype).view_as(
            source.k_cache[:, :, :3]
        )
    )
    source.v_cache[:, :, :3].copy_(source.k_cache[:, :, :3] + 1)
    source.advance(3)
    destination = _new_shared_prefix_cache(
        target_bundle=decoder_bundle,
        batch_size=3,
        max_seq_len=8,
        device=t.device("cpu"),
    )
    destination.prefill(source)
    assert t.equal(destination.k_cache[:, 0, :3], destination.k_cache[:, 1, :3])
    assert destination.k_cache[:, 0].data_ptr() != destination.k_cache[:, 1].data_ptr()
    before = destination.k_cache[:, 1, :3].clone()
    destination.k_cache[:, 0, :3].add_(1000)
    assert t.equal(destination.k_cache[:, 1, :3], before)


def test_full_cache_is_reset_reused_and_tail_uses_smaller_cache(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[int, int]] = []
    original_prefill = KVCache.prefill

    class _RecordingCache(KVCache):
        _next_serial = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            self.serial = self._next_serial
            type(self)._next_serial += 1

        def prefill(self, other: KVCache) -> None:
            calls.append((self.serial, self.batch_size))
            original_prefill(self, other)

    monkeypatch.setattr(surrogate, "KVCache", _RecordingCache)
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20 + 2 * i, 21 + 2 * i] for i in range(5)]
    _full, shared = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source for _ in prompts],
        outputs=[[60 + i] for i in range(5)],
        spans=[EditableSpan(2, 2) for _ in prompts],
        intervention=Intervention.none(),
        batch_size=2,
    )
    assert len(shared) == 5
    assert [batch for _cache_id, batch in calls] == [2, 2, 1]
    assert calls[0][0] == calls[1][0]
    assert calls[2][0] != calls[1][0]


def test_full_sequence_test_oracle_never_constructs_kv_cache(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("legacy path constructed KVCache")

    monkeypatch.setattr(surrogate, "KVCache", _forbidden)
    result = teacher_forced_bpb_loss_full_sequence_reference(
        target_bundle=decoder_bundle,
        prompts=[[2, 3, 4]],
        outputs=[[5, 6]],
        prefix_context_segments_list=[()],
        output_token_bytes=t.ones(TINY_VOCAB, dtype=t.int64),
        intervention=Intervention.none(),
        device=t.device("cpu"),
        device_batch_size=1,
    )
    assert len(result) == 1


def test_cache_position_mismatch_fails_before_suffix_forward(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_prefill = KVCache.prefill

    def _bad_prefill(self: KVCache, other: KVCache) -> None:
        original_prefill(self, other)
        self.advance(1)

    monkeypatch.setattr(KVCache, "prefill", _bad_prefill)
    with pytest.raises(AxBenchBpbAdvError, match="KV cache position"):
        _losses(
            bundle=decoder_bundle,
            prompts=[[2, 3, 20, 21]],
            source_prompts=[[2, 3, 10, 11]],
            outputs=[[60]],
            spans=[EditableSpan(2, 2)],
            intervention=Intervention.none(),
            batch_size=1,
        )


def test_encdec_cached_suffix_checks_offset_plus_length(
    encdec_bundle: TargetModelBundle,
) -> None:
    model = encdec_bundle.model
    rotary_length = int(model.cos.size(1))
    cache = _new_shared_prefix_cache(
        target_bundle=encdec_bundle,
        batch_size=1,
        max_seq_len=rotary_length,
        device=t.device("cpu"),
    )
    cache.advance(rotary_length - 1)
    before = tuple(tensor.clone() for tensor in (cache.k_cache, cache.v_cache, cache.cache_seqlens))
    with pytest.raises(ValueError, match="RoPE capacity exceeded"):
        model(
            t.tensor([[2, 3]], dtype=t.long),
            t.full((1, 2), -1, dtype=t.long),
            enc_feature_ids=t.tensor([[[1]]], dtype=t.long),
            enc_magnitudes=t.tensor([[[1]]], dtype=t.long),
            enc_mask=t.tensor([[[True]]]),
            seg_map=t.zeros((1, 2), dtype=t.long),
            kv_cache=cache,
            loss_reduction="none",
        )
    assert all(t.equal(actual, expected) for actual, expected in zip(
        (cache.k_cache, cache.v_cache, cache.cache_seqlens), before, strict=True,
    ))


def test_corrupting_one_prefix_k_slot_breaks_exact_parity(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = [2, 3, 10, 11]
    prompts = [[2, 3, 20, 21]]
    outputs = [[60, 61]]
    full, clean = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source],
        outputs=outputs,
        spans=[EditableSpan(2, 2)],
        intervention=Intervention.none(),
        batch_size=1,
    )
    assert clean == full
    original_prefill = KVCache.prefill

    def _corrupt_prefill(self: KVCache, other: KVCache) -> None:
        original_prefill(self, other)
        self.k_cache[:, :, 0].add_(100.0)

    monkeypatch.setattr(KVCache, "prefill", _corrupt_prefill)
    _full_again, corrupted = _losses(
        bundle=decoder_bundle,
        prompts=prompts,
        source_prompts=[source],
        outputs=outputs,
        spans=[EditableSpan(2, 2)],
        intervention=Intervention.none(),
        batch_size=1,
    )
    assert corrupted != full


def test_training_state_restored_after_injected_suffix_failure(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = decoder_bundle.model
    original_forward = model.forward
    calls = 0

    def _fail_second_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected suffix failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", _fail_second_forward)
    model.train()
    try:
        with pytest.raises(RuntimeError, match="injected suffix failure"):
            _losses(
                bundle=decoder_bundle,
                prompts=[[2, 3, 20, 21]],
                source_prompts=[[2, 3, 10, 11]],
                outputs=[[60]],
                spans=[EditableSpan(2, 2)],
                intervention=Intervention.none(),
                batch_size=1,
            )
        assert model.training is True
    finally:
        model.eval()


def test_direction_hook_removed_after_injected_suffix_failure(
    decoder_bundle: TargetModelBundle,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = decoder_bundle.model
    block = _gpt_blocks(model)[0]
    hooks_before = set(block._forward_hooks)
    original_forward = model.forward
    calls = 0

    def _fail_second_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected hooked suffix failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", _fail_second_forward)
    intervention = Intervention.from_direction(
        kind="diffmean",
        target_model="decoder_only",
        direction=t.ones(TINY_N_EMBD),
        direction_layer=0,
        coefficient=1.0,
    )
    with pytest.raises(RuntimeError, match="injected hooked suffix failure"):
        teacher_forced_bpb_loss_shared_prefix_kv_batch(
            target_bundle=decoder_bundle,
            prompts=[[2, 3, 20, 21]],
            source_prompts=[[2, 3, 10, 11]],
            outputs=[[60]],
            prefix_context_segments_list=[()],
            editable_spans=[EditableSpan(2, 2)],
            output_token_bytes=t.ones(TINY_VOCAB, dtype=t.int64),
            intervention=intervention,
            device=t.device("cpu"),
            device_batch_size=1,
        )
    assert set(block._forward_hooks) == hooks_before


def test_lora_detached_after_injected_suffix_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fixture_root = Path(__file__).parents[2] / "axbench_bpb" / "fixtures"
    checkpoint = t.load(
        fixture_root / "tiny_decoder_only" / "ckpt.pt",
        map_location="cpu",
        weights_only=False,
    )
    config = GPTConfig(**checkpoint["config"])
    with t.device("meta"):
        model = GPT(config, pad_vocab_size_to=checkpoint["pad_vocab_size_to"])
    model.to_empty(device="cpu")
    model.init_buffers()
    model.load_state_dict(checkpoint["model"], strict=True, assign=True)
    bundle = TargetModelBundle(
        model=model,
        tokenizer=_CharTokenizer(),
        max_seq_len=config.sequence_len,
        target_name="decoder_only",
    )
    before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    original_forward = model.forward
    calls = 0

    def _fail_second_forward(*args: Any, **kwargs: Any) -> Any:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("injected LoRA suffix failure")
        return original_forward(*args, **kwargs)

    monkeypatch.setattr(model, "forward", _fail_second_forward)
    with pytest.raises(RuntimeError, match="injected LoRA suffix failure"):
        teacher_forced_bpb_loss_shared_prefix_kv_batch(
            target_bundle=bundle,
            prompts=[[2, 3, 20, 21]],
            source_prompts=[[2, 3, 10, 11]],
            outputs=[[60]],
            prefix_context_segments_list=[()],
            editable_spans=[EditableSpan(2, 2)],
            output_token_bytes=t.ones(TINY_VOCAB, dtype=t.int64),
            intervention=Intervention.lora(
                adapter_path=(
                    fixture_root
                    / "tiny_decoder_only"
                    / "lora_adapter_concept_0001.pt"
                )
            ),
            lora=LoraParams(cfg=LoraConfig(rank=4, alpha=8.0), layer=1),
            device=t.device("cpu"),
            device_batch_size=1,
        )
    after = model.state_dict()
    assert before.keys() == after.keys()
    assert all(t.equal(before[name], after[name]) for name in before)
