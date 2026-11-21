"""Static turn prompt packing and KV-position parity against training routing."""

from dataclasses import replace

import pytest
import torch as t

import tests.conftest as shared_fixtures
from nanochat.engine import KVCache
from scripts.eval import build_eval_context_cache as builder
from scripts.eval import context_cache_common as common
from steering.conditional.encoder_decoder.eval_adapter import TargetAwareEncoderDecoderEvalWrapper
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval import turn_context_routing
from steering.eval.context_enumeration import enumerate_chat_target_contexts
from steering.eval.inference_engine import build_wrapper_for_features
from steering.eval.target_conditioning import (
    ContextFeatureSegment,
    ContextRoutingMode,
    LockstepTargetRealConfig,
    MultiTargetSegments,
    PooledTargetRealConfig,
    SelfConditionedTarget,
    SingleEntryContextProvider,
    SingleTargetSegment,
    TargetConditioningMode,
)
from steering.eval.target_real_runner import (
    build_context_feature_provider,
    build_target_real_artifacts_for_example,
)
from steering.eval.turn_context_routing import prepare_turn_context
from steering.refined_types import PositiveInt
from steering.segmentation import (
    CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
    SegmentMode,
    turn_segments_for_document,
)
from tests.conftest import build_tiny_decoder, build_tiny_encdec
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer
from tests.steering.test_context_cache_in_memory_conversion import _DifferentSourceTokenizer
from tests.steering.test_turn_context_cache_build import (
    StraddlingSourceTokenizer,
    UnroutedStraddlingSourceTokenizer,
    build_fixture,
    synthetic_shards,
)
from tests.steering.test_turn_context_cache_validation import FineSourceTokenizer


def turn_model(monkeypatch, *, contract=None):
    monkeypatch.setattr(shared_fixtures, "TINY_SEQ_LEN", 2048)
    monkeypatch.setattr(shared_fixtures, "TINY_VOCAB", 512)
    monkeypatch.setattr(shared_fixtures, "TINY_N_SAE_FEATURES", PositiveInt.parse(8))
    t.manual_seed(551)
    model = build_tiny_encdec()
    if contract is not None:
        model = EncoderDecoderGPT(model.dec_config, model.enc_config, model.xattn_config,
            n_sae_features=contract.lineage.n_sae_features, num_buckets=contract.lineage.num_buckets,
            feature_vocab=contract.feature_vocab, pad_vocab_size_to=512)
        model.init_weights()
        model.eval()
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if "cross_attn" in name and parameter.ndim >= 2:
                parameter.normal_(std=0.04)
    return model


def wrapper_fixture(tmp_path, monkeypatch, *, lockstep=False, mode=TargetConditioningMode.CURRENT, straddle=True, unrouted=False):
    tokenizer = ByteChatTokenizer()
    conversation = {"messages": [{"role": "user", "content": "A short prompt café"},
        {"role": "assistant", "content": [{"type": "python", "text": "1+1"},
            {"type": "python_output", "text": "2"}, {"type": "text", "text": "Tool result then #### 2"}]}]}
    records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[conversation], max_problems=None,
        task_name="unit", segment_mode=SegmentMode.turn_delimited_fixed_tokens)
    cfg, records, _, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch, records=records, tokenizer=tokenizer,
        source=UnroutedStraddlingSourceTokenizer() if unrouted else StraddlingSourceTokenizer() if straddle else None)
    builder.build_cache_from_config(cfg)
    provider = build_context_feature_provider({"model_max_seq_len": 2048, "conditioning": {"identity": {}}},
        cache_dir=cfg.output_dir, turn_contract=contract)
    model = turn_model(monkeypatch)
    full_ids, mask = tokenizer.render_conversation(conversation)
    trc = (LockstepTargetRealConfig("lockstep", tmp_path) if lockstep else PooledTargetRealConfig("pooled", tmp_path))
    artifacts = build_target_real_artifacts_for_example(model=model, tokenizer=tokenizer, trc=trc,
        provider=provider, full_token_ids=full_ids, mask=mask, target_mode=mode, turn_contract=contract)
    return artifacts, model, tokenizer, provider, contract, full_ids, mask


@pytest.mark.parametrize("lockstep,mode", [(False, TargetConditioningMode.CURRENT),
    (False, TargetConditioningMode.NEXT), (True, TargetConditioningMode.CURRENT)])
@pytest.mark.parametrize("straddle,unrouted", [(False, False), (True, False), (True, True)],
    ids=["exact-boundary", "gemma-straddle", "unrouted-straddle"])
def test_turn_prompt_and_target_forwards_match_training(tmp_path, monkeypatch, lockstep, mode, straddle, unrouted):
    # kills: clipped full-row prompt, retained marker slot, fixed-width next offset, mask-zero tool fill.
    artifacts, model, tokenizer, provider, contract, full_ids, mask = wrapper_fixture(
        tmp_path, monkeypatch, lockstep=lockstep, mode=mode, straddle=straddle, unrouted=unrouted)
    wrapper = artifacts.wrapper
    context_ids = artifacts.context_token_ids
    p0 = len(context_ids) - 1
    segments = provider.get(context_ids)
    assert segments is not None
    raw_oracle = build_full_sequence_segment_routing(ids=context_ids, tokenizer=tokenizer,
        seg_features=([t.tensor(seg.feature_ids) for seg in segments], [t.tensor(seg.buckets) for seg in segments],
                      [seg.char_start for seg in segments], [seg.char_end for seg in segments]), source_bos_char_len=3)
    keep = int(raw_oracle.seg_map[:p0].max())
    prompt_slots = [dict(zip(seg.feature_ids, seg.buckets, strict=True)) for seg in segments[:keep]]
    prepared = prepare_turn_context(context_ids, segments, tokenizer, source_bos_char_len=3)
    assert prepared.segments == segments[:keep]
    assert t.equal(prepared.token_assignments, raw_oracle.seg_map[:p0] - 1)
    assert keep < len(segments)
    marker_start = source_aligned_char_offsets(tokenizer, context_ids, source_bos_char_len=3)[p0]
    if unrouted:
        assert keep == 3
        assert set(range(keep, len(segments))) == {3, 4}
        assert int(raw_oracle.seg_map[p0]) - 1 == keep + 1 == 4
        assert keep not in (raw_oracle.seg_map - 1).tolist()
        assert (segments[keep].char_start, segments[keep].char_end) == (marker_start - 5, marker_start + 1)
        assert segments[keep + 1].char_start == marker_start + 1
        assert all(seg.char_end > marker_start for seg in segments[keep:])
        # This is exactly the stage-6 precondition that rejected 217 real rows.
        assert any(seg.char_start < marker_start for seg in segments[keep:])
    elif straddle:
        # Regression witness: both clauses of the old char-equality precondition fail.
        assert not any(seg.char_start == marker_start for seg in segments)
        assert segments[keep - 1].char_start < marker_start < segments[keep - 1].char_end
        assert segments[keep - 1].char_end == segments[keep].char_start == marker_start + 1
        assert prepared.segments[-1] == segments[keep - 1]
    else:
        assert segments[keep - 1].char_end == segments[keep].char_start == marker_start
    if lockstep:
        assert isinstance(artifacts.target, MultiTargetSegments)
        target_slots = artifacts.target.segments
        schedule = artifacts.target.token_assignments
    else:
        assert isinstance(artifacts.target, SingleTargetSegment)
        target_slots = [artifacts.target.features]
        schedule = t.zeros(len(full_ids) - p0, dtype=t.int64)
    expected_slots = (prompt_slots[1:] if mode is TargetConditioningMode.NEXT else prompt_slots) + target_slots
    target_base = len(expected_slots) - len(target_slots)
    captured = []
    original = model.encoder.forward

    def capture_encoder(ids, buckets, active):
        captured.append([dict(zip(row_ids[row_mask].tolist(), row_buckets[row_mask].tolist(), strict=True))
            for row_ids, row_buckets, row_mask in zip(ids[0], buckets[0], active[0], strict=True)])
        return original(ids, buckets, active)

    monkeypatch.setattr(model.encoder, "forward", capture_encoder)
    with wrapper.conditioning_context(target_mask=t.tensor([mask], dtype=t.bool), context_token_ids=[context_ids]), t.no_grad():
        logits = wrapper(t.tensor([full_ids]))
        _encoder, routing = wrapper._build_conditioning(t.tensor([full_ids]))
    assert t.isfinite(logits).all()
    assert captured == [expected_slots, expected_slots]
    assert t.equal(routing[0, :p0], raw_oracle.seg_map[:p0] - 1)
    assert t.equal(routing[0, p0:], target_base + schedule)
    assert wrapper.context_provider.get(context_ids) == segments
    assert wrapper.context_provider.turn_contract is contract
    assert len(wrapper._prepared_turn_contexts) == 1
    # A second non-KV pass must reuse the prepared exact row and raw assignments.
    monkeypatch.setattr(wrapper, "_lookup_context_segments", lambda *_: pytest.fail("prompt prepared twice"))
    with wrapper.conditioning_context(context_token_ids=[context_ids]), t.no_grad():
        again = wrapper(t.tensor([full_ids]))
    assert t.equal(logits, again)


def test_turn_lockstep_absolute_decode_and_reference_clamp(tmp_path, monkeypatch):
    # kills: prefill-relative origin, repeated assignment zero, skipping tool positions, missing clamp.
    artifacts, model, _tokenizer, _provider, _contract, full_ids, _mask = wrapper_fixture(tmp_path, monkeypatch, lockstep=True)
    wrapper = artifacts.wrapper
    context_ids = artifacts.context_token_ids
    p0 = len(context_ids) - 1
    assert isinstance(artifacts.target, MultiTargetSegments)
    assignments = artifacts.target.token_assignments
    cache = KVCache(batch_size=1, num_heads=model.dec_config.n_kv_head,
        seq_len=2048, head_dim=model.dec_config.n_embd // model.dec_config.n_head,
        num_layers=model.dec_config.n_layer, device=t.device("cpu"), dtype=t.float32)
    captured = []
    original = wrapper._build_conditioning

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append((cache.get_pos(), result[1].clone()))
        return result

    monkeypatch.setattr(wrapper, "_build_conditioning", capture)
    with wrapper.conditioning_context(context_token_ids=[context_ids]), t.no_grad():
        wrapper(t.tensor([context_ids]), kv_cache=cache)
        state = wrapper._generation_encoder_state
        assert state.target_start_pos == p0
        assert captured[0][1][0, -1] == state.target_base_idx + assignments[0]
        # Exercise a multi-input decode too: every absolute position needs its own assignment.
        continuation = full_ids[len(context_ids):] + [ord("x") + 16] * 4
        for start in range(0, len(continuation), 3):
            wrapper(t.tensor([continuation[start:start + 3]]), kv_cache=cache)
    assert captured[1][0] == p0 + 1
    for absolute, routing in captured[1:]:
        expected = assignments[(t.arange(absolute, absolute + routing.shape[1]) - p0).clamp_max(len(assignments) - 1)]
        assert t.equal(routing[0], state.target_base_idx + expected)
    assert cache.get_pos() == len(full_ids) + 4
    assert captured[-1][1][0, -1] == state.target_base_idx + assignments[-1]


def test_turn_lockstep_next_rejected(tmp_path, monkeypatch):
    # kills: accidentally admitting multiple target slots with NEXT timing.
    with pytest.raises(ValueError, match=r"MultiTargetSegments with len\(segments\) > 1 is not supported "
                                         r"under target_conditioning_mode=NEXT"):
        wrapper_fixture(tmp_path, monkeypatch, lockstep=True, mode=TargetConditioningMode.NEXT)


@pytest.mark.parametrize("mutation,message", [
    # `missing-contract` and `fixed-provider` are indistinguishable by message: the
    # production guard is one disjunction over both inputs
    # (steering/conditional/encoder_decoder/eval_adapter.py:369-370).
    ("missing-contract", "raw-source turn routing requires a checkpoint contract and a turn-capable provider"),
    ("fixed-provider", "raw-source turn routing requires a checkpoint contract and a turn-capable provider"),
    ("wrong-lineage", "turn context contract disagrees with checkpoint-derived expectations"),
    ("implicit-mode", "turn provider/contract requires explicit raw-source turn routing"),
    ("selfcond", "raw-source ChatCORE routing supports static target-real conditioning only"),
    ("wrong-size", "turn wrapper model sequence_len disagrees with checkpoint contract"),
    ("wrong-vocab", "turn wrapper model feature vocabulary disagrees with checkpoint contract"),
    ("wrong-width", "turn wrapper model n_sae_features disagrees with checkpoint contract"),
    ("wrong-buckets", "turn wrapper model num_buckets disagrees with checkpoint contract"),
    ("wrong-tokenizer", "turn wrapper decoder tokenizer disagrees with checkpoint contract"),
])
def test_raw_source_wrapper_requires_explicit_validated_contract(tmp_path, monkeypatch, mutation, message):
    # kills: bypassing mode/lineage/model/tokenizer admission through a lower-level wrapper constructor.
    artifacts, model, tokenizer, _provider, contract, _full, _mask = wrapper_fixture(tmp_path, monkeypatch)
    provider = artifacts.wrapper.context_provider
    supplied_contract = contract
    routing_mode = ContextRoutingMode.TURN_RAW_SOURCE
    target = artifacts.target
    if mutation == "missing-contract":
        supplied_contract = None
    elif mutation == "fixed-provider":
        provider = SingleEntryContextProvider(artifacts.context_token_ids, provider.get(artifacts.context_token_ids))
    elif mutation == "wrong-lineage":
        supplied_contract = replace(contract, semantic={**contract.semantic, "statistics_raw_sha256": "f" * 64})
    elif mutation == "implicit-mode":
        routing_mode = ContextRoutingMode.DECODER_ALIGNED
    elif mutation == "selfcond":
        target = SelfConditionedTarget()
    elif mutation == "wrong-size":
        supplied_contract = replace(contract, model_max_seq_len=128)
    elif mutation == "wrong-vocab":
        model.encoder.feature_vocab = FeatureVocab(8, (0, 2, 3, 5, 7))
        assert model.encoder.config.n_sae_features == contract.lineage.n_sae_features == 8
        assert model.encoder.config.num_buckets == contract.lineage.num_buckets == 4
    elif mutation == "wrong-width":
        model.encoder.config.n_sae_features = 16
        assert model.feature_vocab == contract.feature_vocab
        assert model.encoder.config.num_buckets == contract.lineage.num_buckets == 4
    elif mutation == "wrong-buckets":
        model.encoder.config.num_buckets = 2
        assert model.feature_vocab == contract.feature_vocab
        assert model.encoder.config.n_sae_features == contract.lineage.n_sae_features == 8
    else:
        monkeypatch.setattr(tokenizer, "encode", lambda *args, **kwargs: [99])
    with pytest.raises(ValueError, match=message):
        TargetAwareEncoderDecoderEvalWrapper(model, tokenizer, target, provider,
            turn_contract=supplied_contract, context_routing_mode=routing_mode)


def test_turn_provider_cannot_enter_default_artifact_builder(tmp_path, monkeypatch):
    # kills: implicit schema-5 admission through shared fixed consumers.
    artifacts, model, tokenizer, provider, _contract, full_ids, mask = wrapper_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match="turn provider requires an explicit checkpoint-derived turn contract"):
        build_target_real_artifacts_for_example(model=model, tokenizer=tokenizer,
            trc=PooledTargetRealConfig("pooled", tmp_path), provider=provider, full_token_ids=full_ids, mask=mask)
    with pytest.raises(ValueError, match="turn provider/contract requires explicit raw-source turn routing"):
        build_wrapper_for_features(model, tokenizer, target=artifacts.target, context_provider=artifacts.wrapper.context_provider)


def test_turn_prefix_geometry_and_exact_input_guards(tmp_path, monkeypatch):
    # kills: accepting a fixed-style prefix or a stale context key despite a valid target.
    artifacts, _model, tokenizer, provider, _contract, full_ids, _mask = wrapper_fixture(tmp_path, monkeypatch)
    context = artifacts.context_token_ids
    segments = provider.get(context)
    assert segments is not None
    # A marker may share the last prompt slot; no exact-boundary slot is required.
    shared = (replace(segments[0], char_end=segments[-1].char_end),)
    prepared = prepare_turn_context(context, shared, tokenizer, source_bos_char_len=3)
    assert prepared.segments == shared
    assert prepared.token_assignments.tolist() == [0] * (len(context) - 1)
    # A short input and a rewritten head legitimately share the exact-key guard.
    with artifacts.wrapper.conditioning_context(context_token_ids=[context]):
        with pytest.raises(ValueError, match="turn wrapper input must begin with the exact completion-prefix key"):
            artifacts.wrapper(t.tensor([full_ids[:3]]))
        with pytest.raises(ValueError, match="turn wrapper input must begin with the exact completion-prefix key"):
            artifacts.wrapper(t.tensor([[0, 1, 100] + full_ids[3:]]))


# A slot index at or past ``len(per_seg_fids)`` and a marker routed before the
# last retained slot are unreachable by the primitive's cummax construction, so
# the guards that rejected them are gone and only these mutations remain.
@pytest.mark.parametrize("mutation,match", [
    ("predecessor-gap", "Gap between"), ("unrouted", "unrouted"),
    ("dropped-prompt", "ends at or before the assistant marker"), ("tail-overflow", "delta=21, window=20"),
])
def test_prompt_predecessor_and_routed_slot_guards(tmp_path, monkeypatch, mutation, match):
    artifacts, _, tokenizer, provider, _, _, _ = wrapper_fixture(tmp_path, monkeypatch)
    prefix = artifacts.context_token_ids
    segments = provider.get(prefix)
    assert segments is not None
    prepared = prepare_turn_context(prefix, segments, tokenizer, source_bos_char_len=3)
    first_marker = len(prepared.segments)
    if mutation == "predecessor-gap":
        changed = list(segments)
        changed[first_marker - 1] = replace(changed[first_marker - 1],
            char_end=changed[first_marker - 1].char_end - 1)
        segments = tuple(changed)
    elif mutation == "tail-overflow":
        segments = (*segments[:-1], replace(segments[-1],
            char_end=segments[-1].char_end + CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1))
    else:
        original = turn_context_routing.build_full_sequence_segment_routing

        def corrupt_routing(**kwargs):
            routing = original(**kwargs)
            assignments = routing.seg_map.clone()
            if mutation == "dropped-prompt":
                assignments[:-1].clamp_max_(first_marker - 2)
            else:
                assignments[1] = 0
            return replace(routing, seg_map=assignments)

        monkeypatch.setattr(turn_context_routing, "build_full_sequence_segment_routing", corrupt_routing)
    with pytest.raises(ValueError, match=match):
        prepare_turn_context(prefix, segments, tokenizer, source_bos_char_len=3)


class TailResidueSourceTokenizer(_DifferentSourceTokenizer):
    """One character per source token, with post-marker tokenizer residue."""

    def __init__(self, residue):
        super().__init__()
        self.residue = residue

    def encode(self, text):
        return [self.bos_token_id, *(self._id_for_piece(char) for char in text + "~" * self.residue)]


@pytest.mark.parametrize("residue", [1, CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS,
                                   CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS + 1])
def test_turn_prefix_tail_residue_matches_sft_alignment(residue):
    tokenizer = ByteChatTokenizer()
    conversation = {"messages": [{"role": "user", "content": "x" * 200},
                                 {"role": "assistant", "content": "answer"}]}
    records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[conversation],
        max_problems=None, task_name="unit", segment_mode=SegmentMode.turn_delimited_fixed_tokens)
    prefix = tokenizer.render_for_completion(conversation)
    record = next(row for row in records if list(row.token_ids) == prefix)
    assert record.turn_char_starts is not None
    source = TailResidueSourceTokenizer(residue)
    source_ids = source.encode(record.context_text)
    if residue > CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS:
        with pytest.raises(ValueError, match="Turn-delimited tokenizer round-trip mismatch for document .*"
                                             "tail_delta=21, window=20"):
            turn_segments_for_document(source_ids, source, record.context_text,
                record.turn_char_starts, PositiveInt.parse(32))
        # Independently constructed raw payload also fails consumer admission.
        accepted = TailResidueSourceTokenizer(CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS)
        _, starts, ends, _ = turn_segments_for_document(accepted.encode(record.context_text),
            accepted, record.context_text, record.turn_char_starts, PositiveInt.parse(32))
        ends[-1] += 1
    else:
        _, starts, ends, drift = turn_segments_for_document(source_ids, source,
            record.context_text, record.turn_char_starts, PositiveInt.parse(32))
        assert drift == residue
    segments = tuple(ContextFeatureSegment((index,), (1,), start, end)
                     for index, (start, end) in enumerate(zip(starts, ends, strict=True)))
    seg_features = ([t.tensor(seg.feature_ids) for seg in segments],
                    [t.tensor(seg.buckets) for seg in segments], starts, ends)
    if residue > CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS:
        # Both entry points legitimately share the reverse-mismatch guard; naming it
        # keeps `delta=` from also matching the round-trip guard's `tail_delta=`.
        reverse = ("Reverse tokenizer-mismatch delta exceeds end-marker window: "
                   "decoded_len=242, coverage_end=263, delta=21, window=20")
        with pytest.raises(ValueError, match=reverse):
            build_full_sequence_segment_routing(ids=prefix, tokenizer=tokenizer,
                source_bos_char_len=3, seg_features=seg_features)
        with pytest.raises(ValueError, match=reverse):
            prepare_turn_context(prefix, segments, tokenizer, source_bos_char_len=3)
        return
    oracle = build_full_sequence_segment_routing(ids=prefix, tokenizer=tokenizer,
        source_bos_char_len=3, seg_features=seg_features)
    prepared = prepare_turn_context(prefix, segments, tokenizer, source_bos_char_len=3)
    keep = int(oracle.seg_map[:-1].max())
    assert prepared.segments == segments[:keep]
    assert prepared.token_assignments.tolist() == (oracle.seg_map[:-1] - 1).tolist()
    assert prepared.token_ids == tuple(prefix)
    assert ends[-1] == source_aligned_char_offsets(tokenizer, prefix, source_bos_char_len=3)[-1] + residue
    if residue == CROSS_TOKENIZER_END_MARKER_WINDOW_CHARS:
        assert starts[-1] >= source_aligned_char_offsets(tokenizer, prefix, source_bos_char_len=3)[-1]


@pytest.mark.parametrize("end_delta", [-1, 0], ids=["before-marker", "at-marker"])
def test_turn_prefix_rejects_unrouted_prompt_window(tmp_path, monkeypatch, end_delta):
    artifacts, _, tokenizer, provider, _, _, _ = wrapper_fixture(tmp_path, monkeypatch, unrouted=True)
    prefix = artifacts.context_token_ids
    segments = provider.get(prefix)
    assert segments is not None
    prepared = prepare_turn_context(prefix, segments, tokenizer, source_bos_char_len=3)
    keep = len(prepared.segments)
    marker_start = source_aligned_char_offsets(tokenizer, prefix, source_bos_char_len=3)[-2]
    tail = segments[keep]
    changed = (*segments[:keep], replace(tail, char_end=marker_start + end_delta),
               replace(tail, char_start=marker_start + end_delta), *segments[keep + 1:])
    # Both split offsets legitimately hit the one dropped-prefix-slot guard.
    with pytest.raises(ValueError, match="dropped turn prefix slot ends at or before the assistant marker"):
        prepare_turn_context(prefix, changed, tokenizer, source_bos_char_len=3)


@pytest.mark.parametrize("with_contract", [False, True])
def test_raw_routing_rejects_decoder_only_model(tmp_path, monkeypatch, with_contract):
    _, _, tokenizer, _, contract, _, _ = wrapper_fixture(tmp_path, monkeypatch)
    model = build_tiny_decoder()
    # Both routing modes legitimately reach the one encoder-decoder admission guard.
    with pytest.raises(ValueError, match="turn raw-source routing requires an encoder-decoder model"):
        build_wrapper_for_features(model, tokenizer, context_provider=None,
            context_routing_mode=(ContextRoutingMode.DECODER_ALIGNED if with_contract else ContextRoutingMode.TURN_RAW_SOURCE),
            turn_contract=contract if with_contract else None)


def test_multiple_marker_slots_unused_ordinals_and_zero_width_inputs(tmp_path, monkeypatch):
    # kills: discarding just one marker slot, compressing unused source ordinals, snapping tiny tails.
    tokenizer = ByteChatTokenizer()
    conversation = {"messages": [{"role": "user", "content": "é🦋"}, {"role": "assistant", "content": "答"}]}
    records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[conversation], max_problems=None,
        task_name="unicode", segment_mode=SegmentMode.turn_delimited_fixed_tokens)
    cfg, records, _, _, tokenizer, _ = build_fixture(tmp_path, monkeypatch, records=records, tokenizer=tokenizer)
    source = FineSourceTokenizer()
    synthetic_shards(tmp_path / "features", records, source)
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda _: source)
    contract, _ = common.turn_contract_from_config(cfg, tokenizer)
    builder.build_cache_from_config(cfg)
    provider = build_context_feature_provider({"model_max_seq_len": 2048, "conditioning": {}},
        cache_dir=cfg.output_dir, turn_contract=contract)
    full_ids, mask = tokenizer.render_conversation(conversation)
    model = turn_model(monkeypatch)
    artifacts = build_target_real_artifacts_for_example(model=model, tokenizer=tokenizer,
        trc=LockstepTargetRealConfig("lockstep", tmp_path), provider=provider,
        full_token_ids=full_ids, mask=mask, turn_contract=contract)
    assert isinstance(artifacts.target, MultiTargetSegments)
    assert isinstance(artifacts.wrapper, TargetAwareEncoderDecoderEvalWrapper)
    prefix = artifacts.context_token_ids
    with artifacts.wrapper.conditioning_context(context_token_ids=[prefix]), t.no_grad():
        logits = artifacts.wrapper(t.tensor([full_ids]))
        encoded, routing = artifacts.wrapper._build_conditioning(t.tensor([full_ids]))
    assert t.isfinite(logits).all()
    prepared = artifacts.wrapper._prepared_turn_contexts[tuple(prefix)]
    cached = provider.get(prefix)
    assert cached is not None
    assert len(cached) - len(prepared.segments) > 1
    assert len(set(prepared.token_assignments.tolist())) < len(prepared.segments)
    assert len(set(artifacts.target.token_assignments.tolist())) < len(artifacts.target.segments)
    assert encoded.layout.n_segments == len(prepared.segments) + len(artifacts.target.segments)
    # The marker selects the first retained target; preceding unused slots are trimmed.
    assert artifacts.target.token_assignments[0] == 0
    assert routing[0, len(prefix) - 1] == len(prepared.segments) + artifacts.target.token_assignments[0]


def test_single_slot_lockstep_next_remains_supported(tmp_path, monkeypatch):
    # kills: blanket lockstep-NEXT rejection and token-width offsets instead of one source ordinal.
    artifacts, model, tokenizer, _provider, contract, full_ids, _mask = wrapper_fixture(tmp_path, monkeypatch)
    target = MultiTargetSegments(segments=[{1: 2}], token_assignments=t.zeros(3, dtype=t.int64))
    wrapper = build_wrapper_for_features(model, tokenizer, target=target,
        context_provider=artifacts.wrapper.context_provider, target_mode=TargetConditioningMode.NEXT,
        context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE, turn_contract=contract)
    context = artifacts.context_token_ids
    with wrapper.conditioning_context(context_token_ids=[context]), t.no_grad():
        logits = wrapper(t.tensor([full_ids]))
        _encoder, routing = wrapper._build_conditioning(t.tensor([full_ids]))
    assert t.isfinite(logits).all()
    prepared = wrapper._prepared_turn_contexts[tuple(context)]
    assert (routing[0, len(context) - 1:] == len(prepared.segments) - 1).all()


def test_raw_source_batching_and_fixed_provider_artifacts_rejected(tmp_path, monkeypatch):
    # kills: implicit turn schema admission or encoder-decoder batch generation via the raw wrapper.
    artifacts, model, tokenizer, _provider, contract, full_ids, mask = wrapper_fixture(tmp_path, monkeypatch)
    with artifacts.wrapper.conditioning_context(context_token_ids=[artifacts.context_token_ids] * 2):
        with pytest.raises(ValueError, match="turn target-real wrapper requires batch size 1"):
            artifacts.wrapper(t.tensor([full_ids, full_ids]))
    fixed_provider = SingleEntryContextProvider(artifacts.context_token_ids, artifacts.wrapper.context_provider.get(artifacts.context_token_ids))
    with pytest.raises(ValueError, match="turn target-real requires a validated schema-5 provider"):
        build_target_real_artifacts_for_example(model=model, tokenizer=tokenizer,
            trc=PooledTargetRealConfig("pooled", tmp_path), provider=fixed_provider,
            full_token_ids=full_ids, mask=mask, turn_contract=contract)


def test_clustered_wrapper_rejects_same_cardinality_vocabulary(tmp_path, monkeypatch):
    _, records, _, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch, clustered=True)
    model = turn_model(monkeypatch, contract=contract)
    assert contract.feature_vocab == FeatureVocab(8, (0, 2, 3, 6, 7))
    model.encoder.feature_vocab = FeatureVocab(8, (0, 1, 3, 6, 7))
    assert model.feature_vocab is not None
    assert model.feature_vocab.size == contract.feature_vocab.size
    provider = SingleEntryContextProvider(records[0].token_ids, (), turn_contract=contract)
    with pytest.raises(ValueError, match="turn wrapper model feature vocabulary disagrees with checkpoint contract"):
        TargetAwareEncoderDecoderEvalWrapper(model, tokenizer, SingleTargetSegment(features={2: 0}), provider,
            turn_contract=contract, context_routing_mode=ContextRoutingMode.TURN_RAW_SOURCE)
