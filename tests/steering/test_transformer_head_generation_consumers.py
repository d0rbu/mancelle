"""Learned-head cache histories through real multisegment and AxBench consumers."""

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch as t

from nanochat import flash_attention
from nanochat.engine import KVCache
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
    SelfConditionedFeatureProvider,
    SelfCondState,
)
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import (
    NegativeSamplingSelection,
    SamplingMode,
    SamplingSelection,
    ThresholdSelection,
)
from steering.conditional.thresholding import ThresholdType
from steering.eval import multiseg_generator
from steering.eval.axbench_adv import self_conditioning
from steering.eval.axbench_upstream.methods import PreparedPromptContext
from steering.eval.multiseg_generator import MultisegInputs
from steering.eval.target_conditioning import ContextFeatureSegment
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer


@pytest.fixture(autouse=True)
def sdpa(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.SDPA)


def _learned_model():
    t.manual_seed(1719)
    model = build_tiny_model(
        n_sae_features=8, num_buckets=2, seq_len=64, n_layer=2,
        xattn_frequency=1, vocab_size=512,
        feature_pred_config=FeaturePredictionConfig(
            block=FeaturePredictionBlock.TRANSFORMER, head_layers=2,
            loss_mode=FeaturePredLossMode.MAX_POOL,
        ),
    ).eval()
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if "c_proj.weight" in name:
                parameter.normal_(0, 0.03)
    return model


def _observe_cached_history(model, monkeypatch):
    """Replay each call without a cache, preserving its exact conditioning history."""
    original = model.forward
    tokens, slots, records = [], [], []

    def forward(ids, *args, **kwargs):
        cache = kwargs["kv_cache"]
        assert isinstance(cache, KVCache)
        start = int(cache.get_pos())
        tokens[start:] = ids[0].tolist()
        slots[start:] = kwargs["seg_map"][0].tolist()
        actual = original(ids, *args, **kwargs)
        assert int(cache.get_pos()) == start + ids.shape[1]
        with t.no_grad():
            expected_lm, expected_fp = original(
                t.tensor([tokens]), None, seg_map=t.tensor([slots]),
                encoder_out=kwargs["encoder_out"], encoder_strict=True,
                return_fp_logits=True,
            )
        if kwargs.get("fp_logits_only"):
            pairs = [(actual, expected_fp[:, start:])]
        elif kwargs.get("return_fp_logits"):
            pairs = list(zip(actual, (expected_lm[:, start:], expected_fp[:, start:]), strict=True))
        else:
            pairs = [(actual, expected_lm[:, start:])]
        for observed, expected in pairs:
            t.testing.assert_close(observed, expected, atol=3e-6, rtol=3e-6)
        records.append((start, kwargs, cache, expected_fp.detach()))
        return actual

    monkeypatch.setattr(model, "forward", forward)
    return records


@pytest.mark.parametrize("cadence", [1, 2])
def test_multisegment_learned_head_matches_replayed_conditioning_across_fixed_commits(monkeypatch, cadence):
    model = _learned_model()
    config = SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(cadence),
        segment_mode=SegmentMode.fixed_tokens,
        pool_mode=FeaturePredLossMode.MAX_POOL,
        n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(2),
        selection=ThresholdSelectionConfig(ThresholdSelection(ThresholdType.segment_topk, 2), None),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
    )
    provider = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
    records = _observe_cached_history(model, monkeypatch)
    prefix = [16, 17, 18, 19, 20, 21]
    inputs = MultisegInputs(
        enc_feature_ids=t.tensor([[[1, 2], [3, 4]]]),
        enc_magnitudes=t.tensor([[[0, 1], [1, 0]]]),
        enc_mask=t.ones(1, 2, 2, dtype=t.bool),
        seg_map=t.tensor([[0, 0, 1, 1, 1, 1]]),
    )
    generated = list(multiseg_generator.generate_completion_stream_self_conditioned(
        model, prefix, provider, inputs, 6, t.device("cpu"),
        temperature=0, bos_id=511,
    ))
    assert len(generated) == 6
    assert records[0][0] == 0 and records[1][0] == len(prefix) - 1
    cache = records[-1][2]
    assert cache.n_layers == model.num_cache_layers == 4
    assert cache.get_pos() == len(prefix) + len(generated) - 1
    for position in range(len(prefix), int(cache.get_pos())):
        calls = [kwargs for start, kwargs, *_ in records if start == position]
        boundary = (position - len(prefix) + 1) % cadence == 0
        assert len(calls) == 1 + int(boundary)
        if boundary:
            assert calls[0]["fp_logits_only"]
            assert not calls[1].get("return_fp_logits", False)
            assert int(calls[1]["seg_map"].item()) == int(calls[0]["seg_map"].item()) + 1
    for slot in range(cache.n_layers):
        assert t.count_nonzero(cache.k_cache[slot, :, :cache.get_pos()])


def test_axbench_gcg_prompt_seed_uses_learned_head_cache_and_matches_full_prefix(monkeypatch):
    model, tokenizer = _learned_model(), ByteChatTokenizer()
    setup = cast(Any, SimpleNamespace(
        segment_mode=SegmentMode.fixed_tokens,
        output_selection=SamplingSelection(mean=2, std=2, min_k=1, max_k=3, mode=SamplingMode.top_k_percentile),
        output_negative_selection=NegativeSamplingSelection(mean=1, std=1, min_k=1, max_k=2),
        percentile_table=t.zeros(8, 99), n_sae_features=8,
        feature_mask=None, consolidation=None,
    ))
    cell = object.__new__(self_conditioning._BoundSelfConditioningCell)
    cell._method, cell._concept_id, cell._factor = "SelfConditioning", 0, 0.0
    cell._runtime = cast(Any, SimpleNamespace(
        _target_model=model, _target_tokenizer=tokenizer, _nanochat_tokenizer=tokenizer,
        _max_prompt_tokens=48, _prompt_rng=t.Generator().manual_seed(10),
        _setup=setup, _extraction_model=None, _sae=None,
        _dense_diffmeans={0: t.zeros(8)},
    ))

    def extracted_prompt(_setup, _extractor, _sae, _tokenizer, *, token_ids, assistant_end_id):
        # Only the external SAE producer is replaced: routing, encoding, head
        # prediction, feature selection and cache ownership all remain real.
        return PreparedPromptContext(
            tuple(token_ids), assistant_end_id,
            (ContextFeatureSegment((1, 2), (0, 1), 0, len(tokenizer.decode(token_ids))),),
        )

    monkeypatch.setattr(self_conditioning, "prepare_prompt_context", extracted_prompt)
    records = _observe_cached_history(model, monkeypatch)
    rng = cell._runtime._prompt_rng.get_state().clone()
    row, seed = cell._prepare_gcg_row(
        prompt="hello", response="yes", sample_id=31, seed_text="!", objective="response_margin",
    )
    assert seed == tuple(tokenizer.encode("!"))
    assert t.equal(cell._runtime._prompt_rng.get_state(), rng)
    assert len(records) == 1
    _start, kwargs, cache, full_prefix_fp = records[0]
    assert cache.n_layers == model.num_cache_layers == 4
    assert cache.get_pos() == len(row.clean_prompt) - 1
    for slot in range(cache.n_layers):
        assert t.count_nonzero(cache.k_cache[slot, :, :cache.get_pos()])
    config = self_conditioning.build_self_conditioning_config(
        setup, model, cell._policy(), selection_seed=31,
    )
    oracle = SelfConditionedFeatureProvider(model, config, t.device("cpu"))
    source_rows = row.clean_seg_map[0, :-1] == row.source_slot
    initial = kwargs["encoder_out"]
    with t.no_grad():
        expected = oracle.seed_from_prompt(
            SelfCondState(initial, initial.layout.n_segments - 1), full_prefix_fp[:, source_rows],
        )
    assert row.target_slot == expected.target_slot_idx
    assert t.equal(row.encoder_out.values, expected.encoder_out.values)
    assert row.response_byte_weights is not None
    assert t.equal(row.response_byte_weights, t.ones(len(row.response)))
