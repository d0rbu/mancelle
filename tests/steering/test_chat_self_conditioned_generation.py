"""Real FP/KV forwards with instrumented source rows and scripted token choices."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import pytest
import torch as t

import nanochat.engine as scalar_engine
import steering.eval.chat_self_conditioned_generation as chat
import steering.eval.inference_engine as inference
from scripts.eval.context_cache_common import cache_identity_from_config
from steering.conditional.encoder_decoder.config import (
    FeatureBucketProducer,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
    FeaturePredLossMode,
    FeatureSelectionProducer,
)
from steering.conditional.encoder_decoder.eval_adapter import TargetAwareEncoderDecoderEvalWrapper
from steering.conditional.encoder_decoder.self_conditioned import (
    SelfCondConfig,
)
from steering.conditional.feature_pipeline import ThresholdSelectionConfig
from steering.conditional.sampling import ThresholdSelection
from steering.conditional.thresholding import ThresholdType
from steering.eval.context_cache_io import (
    context_length_cache_dir,
    feature_selection_hash_leaf,
    save_context_cache,
)
from steering.eval.context_enumeration import enumerate_chat_target_contexts
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    ContextRoutingMode,
    SingleEntryContextProvider,
    SingleTargetSegment,
)
from steering.eval.target_real_runner import build_context_feature_provider
from steering.eval.turn_context_routing import prepare_turn_context
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.test_turn_context_cache_build import (
    StraddlingSourceTokenizer,
    UnroutedStraddlingSourceTokenizer,
    build_fixture,
    convert,
)
from tests.steering.test_turn_context_cache_validation import FineSourceTokenizer

TURN = SegmentMode.turn_delimited_fixed_tokens
FIXED = SegmentMode.fixed_tokens


class WideSourceTokenizer(StraddlingSourceTokenizer):
    """One wide source token makes the final prompt slot wider than K=33."""

    def encode(self, text):
        pieces = [self.decode([token]) for token in super().encode(text)[1:]]
        combined = []
        for piece in pieces:
            if combined and set(piece) == {"a"} and set(combined[-1]) == {"a"}:
                combined[-1] += piece
            else:
                combined.append(piece)
        return [self.bos_token_id, *(self._id_for_piece(piece) for piece in combined)]


@dataclass
class Fixture:
    engine: Any
    tokenizer: Any
    source: Any
    contract: Any
    prompt: list[int]
    provider: Any
    model: Any


@dataclass
class Forward:
    position: int
    ids: list[int]
    slots: list[int]
    fp: t.Tensor | None
    logits: t.Tensor | None


def make_fixture(tmp_path, monkeypatch, *, mode=TURN, K=2, source=None, prompt_text=None, full_context=False, head_depth=0, head_block="mlp"):
    tokenizer = ByteChatTokenizer()
    conversation = {"messages": [
        {"role": "user", "content": prompt_text or "café " + "a" * 80},
        {"role": "assistant", "content": "REFERENCE MUST NEVER BE FED"},
    ]}
    prompt = tokenizer.render_for_completion(conversation)
    capacity = len(prompt) if full_context else 2048
    contract = None
    if mode is TURN:
        # Prefix-only serialized schema-5 cache: there is no reference row to read.
        records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[conversation],
            max_problems=None, task_name="unit", segment_mode=TURN)
        cfg, records, paths, contract, tokenizer, source = build_fixture(
            tmp_path, monkeypatch, records=records, tokenizer=tokenizer,
            source=source or StraddlingSourceTokenizer(),
        )
        contract = replace(contract, model_max_seq_len=capacity)
        entries = [entry for entry in convert(records, paths, contract, tokenizer) if entry.token_ids == tuple(prompt)]
        assert len(entries) == 1
        identity = cache_identity_from_config({}, turn_contract=contract)
        cache_dir = context_length_cache_dir(cfg.output_dir, capacity,
            hash_leaf=feature_selection_hash_leaf(identity.feature_selection_hash))
        save_context_cache(cache_dir, entries, identity=identity, n_sae_features=PositiveInt.parse(8),
            num_buckets=PositiveInt.parse(4), source_segment_mode=TURN, turn_contract=contract)
        provider = build_context_feature_provider({"model_max_seq_len": capacity, "conditioning": {"identity": {}}},
            cache_dir=cfg.output_dir, turn_contract=contract)
    else:
        provider = SingleEntryContextProvider(prompt, (
            ContextFeatureSegment(feature_ids=(1, 2), buckets=(0, 1), char_start=0, char_end=len(tokenizer.decode(prompt))),
        ))
    t.manual_seed(418)
    model = build_tiny_model(n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
        feature_pred_config=FeaturePredictionConfig(loss_mode=FeaturePredLossMode.MAX_POOL, head_layers=head_depth, block=FeaturePredictionBlock(head_block)),
        seq_len=capacity, n_layer=2, xattn_frequency=1, vocab_size=512)
    model.eval()
    with t.no_grad():
        for name, param in model.named_parameters():
            if "cross_attn" in name and param.ndim >= 2:
                param.normal_(std=0.04)
            if name.startswith("feature_pred_head.residual_blocks."):
                param.normal_(std=0.03)
    config = SelfCondConfig(
        self_cond_segment_size=PositiveInt.parse(K), segment_mode=mode,
        pool_mode=FeaturePredLossMode.MAX_POOL, n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
        selection=ThresholdSelectionConfig(ThresholdSelection(ThresholdType.segment_topk, 2), None),
        selection_producer=FeatureSelectionProducer.FULL_PERCENTILE_RANK,
        bucket_producer=FeatureBucketProducer.UNIFORM_PERCENTILE,
    )
    engine = inference.build_engine_self_conditioned(model, tokenizer, t.device("cpu"), config,
        context_provider=provider, context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE if mode is TURN else ContextRoutingMode.DECODER_ALIGNED,
        turn_contract=contract)
    return Fixture(engine, tokenizer, source, contract, prompt, provider, model)


def instrument(fixture, monkeypatch):
    forwards, appends = [], []
    original = fixture.model.forward

    def forward(ids, *args, **kwargs):
        position = kwargs["kv_cache"].get_pos()
        result = original(ids, *args, **kwargs)
        logits, fp = (result if kwargs.get("return_fp_logits") else
                      (None, result) if kwargs.get("fp_logits_only") else (result, None))
        forwards.append(Forward(position, ids[0].tolist(), kwargs["seg_map"][0].tolist(),
            None if fp is None else fp.detach().clone(), None if logits is None else logits.detach().clone()))
        return result

    monkeypatch.setattr(fixture.model, "forward", forward)
    provider = fixture.engine._model._self_cond.provider
    for name in ("seed_from_prompt", "step", "append_tool_window"):
        original_append = getattr(provider, name)

        def append(state, rows, *args, _name=name, _original=original_append, **kwargs):
            result = _original(state, rows, *args, **kwargs)
            appends.append((_name, rows.detach().clone(), result.target_slot_idx, kwargs))
            return result

        monkeypatch.setattr(provider, name, append)
    return forwards, appends


def script_tokens(monkeypatch, module, tokens):
    calls = []

    def sample(logits, *args):
        assert t.isfinite(logits).all()
        token = tokens[min(len(calls), len(tokens) - 1)]
        calls.append(logits.detach().clone())
        return t.tensor([[token]])

    monkeypatch.setattr(module, "sample_next_token", sample)
    return calls


@pytest.mark.parametrize("mode", [TURN, FIXED])
@pytest.mark.parametrize("K", [1, 2, 33])
@pytest.mark.parametrize("offset", [-1, 0, 1, 4])
@pytest.mark.parametrize("terminal", [False, True])
def test_real_cadence_seed_rows_and_budget(tmp_path, monkeypatch, mode, K, offset, terminal):
    fixture = make_fixture(tmp_path, monkeypatch, mode=mode, K=K, source=WideSourceTokenizer())
    forwards, appends = instrument(fixture, monkeypatch)
    N = max(1, K + offset) if offset != 4 else 3 * K + 1
    token = fixture.tokenizer.encode("x")[0]
    script_tokens(monkeypatch, chat, [token] * N + [fixture.tokenizer.encode_special("<assistant_end>")])
    result = chat.generate_self_conditioned_chat(fixture.engine, fixture.prompt, max_tokens=N + int(terminal),
        extraction_tokenizer=fixture.source)
    assert result.token_ids == tuple(fixture.prompt + [token] * N)
    assert result.mask == (0,) * len(fixture.prompt) + (1,) * N
    consumed = N if terminal else N - 1
    expected = 1 + (consumed + int(mode is TURN)) // K
    assert len(appends) == expected
    assert appends[0][0] == "seed_from_prompt"
    assert sum(name == "seed_from_prompt" for name, *_ in appends) == 1
    prefix, marker = forwards[:2]
    assert prefix.ids == fixture.prompt[:-1]
    assert marker.ids == fixture.prompt[-1:]
    assert prefix.fp is not None
    final_slot = prefix.slots[-1]
    selected = t.tensor(prefix.slots) == final_slot
    assert int(selected.sum()) > K
    assert t.equal(appends[0][1], prefix.fp[:, selected, :])
    assert marker.slots == [appends[0][2]]
    if mode is TURN:
        assert marker.fp is not None
        assert [forward.position for forward in forwards] == [0, *range(len(fixture.prompt) - 1, len(fixture.prompt) + consumed)]
        assert [token for forward in forwards[2:] for token in forward.ids] == [token] * consumed
        fp_inputs = [forward.fp for forward in forwards[1:]]
    else:
        assert marker.fp is None
        fp_inputs = [forward.fp for forward in forwards[2:] if forward.fp is not None]
        # Fixed Kth input is measured then replayed into the SAME KV position.
        for position in range(len(fixture.prompt), len(fixture.prompt) + consumed):
            at_position = [forward for forward in forwards if forward.position == position]
            boundary = (position - len(fixture.prompt) + 1) % K == 0
            assert len(at_position) == 1 + int(boundary)
            if boundary:
                assert at_position[0].logits is None
                assert at_position[1].fp is None
                assert at_position[1].slots == [at_position[0].slots[0] + 1]
    for index, (_name, rows, _slot, _kw) in enumerate(appends[1:]):
        assert t.equal(rows, t.cat(fp_inputs[index * K:(index + 1) * K], dim=1))


@pytest.mark.parametrize("mode", [TURN, FIXED])
@pytest.mark.parametrize("source_type", [StraddlingSourceTokenizer, UnroutedStraddlingSourceTokenizer, FineSourceTokenizer])
@pytest.mark.parametrize("K", [1, 2, 33])
def test_categorical_prefill_uses_s0_and_full_context(tmp_path, monkeypatch, mode, source_type, K):
    fixture = make_fixture(tmp_path, monkeypatch, mode=mode, K=K, source=source_type(), full_context=True)
    wrapper = fixture.engine._model
    forwards, appends = instrument(fixture, monkeypatch)
    monkeypatch.setattr(inference, "sample_next_token", lambda *a: pytest.fail("categorical sampled"))
    monkeypatch.setattr(chat, "sample_next_token", lambda *a: pytest.fail("categorical sampled"))
    prefill = fixture.engine.prepare_self_conditioned_prefill(fixture.prompt)
    assert prefill.kv_cache.get_pos() == prefill.kv_cache.max_seq_len == len(fixture.prompt)
    assert len(forwards) == 2
    assert t.equal(prefill.logits, forwards[-1].logits[:, -1, :])
    letters = fixture.tokenizer.encode("ABCD")
    assert prefill.logits[:, letters].argmax().item() == forwards[-1].logits[:, -1, letters].argmax().item()
    assert forwards[-1].slots == [appends[0][2]]
    if mode is TURN:
        raw = fixture.provider.get(fixture.prompt)
        prepared = prepare_turn_context(fixture.prompt, raw, fixture.tokenizer,
            source_bos_char_len=fixture.contract.semantic["source_bos_char_len"])
        assert len(prepared.segments) < len(raw)
        assert forwards[0].slots == prepared.token_assignments.tolist()
        assert len(wrapper._prepared_turn_contexts) == 1
        if source_type is FineSourceTokenizer:
            assert set(forwards[0].slots) != set(range(len(prepared.segments)))
            assert appends[0][2] == len(prepared.segments)
    with pytest.raises(RuntimeError, match="requires a kv_cache"):
        wrapper(t.tensor([fixture.prompt]))


@pytest.mark.parametrize("mode", [TURN, FIXED])
@pytest.mark.parametrize("K", [1, 2, 6, 33])
@pytest.mark.parametrize("case", ["valid", "failed", "empty", "long", "budget", "exact", "repeat", "bos"])
@pytest.mark.parametrize("head_depth,head_block", [(0, "mlp"), (2, "transformer")])
def test_calculator_transcript_fp_windows_and_resets(tmp_path, monkeypatch, mode, K, case, head_depth, head_block):
    fixture = make_fixture(tmp_path, monkeypatch, mode=mode, K=K, head_depth=head_depth, head_block=head_block)
    tok = fixture.tokenizer
    special = tok.encode_special
    expression = "1/0" if case == "failed" else "" if case == "empty" else "123456789*123456789" if case == "long" else "1+1"
    answer = "15241578750190521" if case == "long" else "2"
    assistant = [special("<python_start>"), *tok.encode(expression), special("<python_end>")]
    if case == "exact":
        assistant = tok.encode("x" * ((-1 - len(assistant)) % K)) + assistant
    forced = [] if case in ("failed", "empty") else [special("<output_start>"), *tok.encode(answer), special("<output_end>")]
    generated = assistant + forced + tok.encode(" resumed")
    if case == "repeat":
        generated += assistant + forced + tok.encode(" done")
    stop = special("<bos>" if case == "bos" else "<assistant_end>")
    script = generated + [stop]
    budget = len(assistant) + 2 if case == "budget" else 4096
    expected = generated[:budget]
    forwards, appends = instrument(fixture, monkeypatch)
    sampled_logits = script_tokens(monkeypatch, chat, script)
    output = chat.generate_self_conditioned_chat(fixture.engine, fixture.prompt, max_tokens=budget, extraction_tokenizer=fixture.source)
    assert output.token_ids == tuple(fixture.prompt + expected)
    assert all(stop not in f.ids for f in forwards[1:])
    consumed = len(expected) - int(case == "budget")
    assert max(f.position for f in forwards) == len(fixture.prompt) + consumed - 1
    if mode is TURN and forced:
        tool_appends = [a for a in appends if a[0] == "append_tool_window"]
        if case == "budget":
            assert tool_appends == []
        else:
            assert tool_appends
            if case == "long" and K in (1, 2):
                assert max(a[1].shape[1] for a in tool_appends) > K
            for _, rows, slot, _ in tool_appends:
                calls = [f for f in forwards if f.slots == [slot - 1]]
                assert t.equal(rows, t.cat([f.fp for f in calls], dim=1))
            first_tool = next(f for f in forwards if f.ids == [special("<output_start>")])
            assistant_inputs = 1 + len(assistant)
            expected_incoming = appends[0][2] + (assistant_inputs + K - 1) // K
            assert first_tool.slots == [expected_incoming]
            partial = assistant_inputs % K
            if partial:
                outgoing = [f for f in forwards[1:] if f.position < first_tool.position][-partial:]
                boundary_append = next(a for a in appends if a[2] == expected_incoming)
                assert boundary_append[0] == "step"
                assert t.equal(boundary_append[1], t.cat([f.fp for f in outgoing], dim=1))
            final_tool = next(f for f in forwards if f.ids == [special("<output_end>")])
            resumed = next(f for f in forwards if f.position == final_tool.position + 1)
            assert resumed.slots == [final_tool.slots[0] + 1]
            assert t.equal(sampled_logits[final_tool.position + 1 - len(fixture.prompt)], final_tool.logits[:, -1, :])
    # The current scalar calculator is the transcript/mask oracle. Its extra
    # budget-end forward is allowed here, but never in the new driver above.
    with monkeypatch.context() as patch:
        # Restore the real model method so the oracle doesn't pollute traces.
        patch.setattr(fixture.model, "forward", type(fixture.model).forward.__get__(fixture.model))
        static = TargetAwareEncoderDecoderEvalWrapper(fixture.model, tok, SingleTargetSegment({1: 0}), fixture.provider,
            context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE if mode is TURN else ContextRoutingMode.DECODER_ALIGNED,
            turn_contract=fixture.contract)
        script_tokens(patch, scalar_engine, script)
        oracle_ids, oracle_masks = scalar_engine.Engine(static, tok).generate_batch(fixture.prompt,
            max_tokens=min(budget, 2048 - len(fixture.prompt)), temperature=0)
    assert output.token_ids == tuple(oracle_ids[0])
    assert output.mask == tuple(oracle_masks[0])
