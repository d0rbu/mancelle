"""CPU integration through real turn caches, model forwards, task scoring and ChatCORE."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch as t
import yaml
from datasets import Dataset
from omegaconf import DictConfig, OmegaConf
from safetensors.torch import save as safetensors_save

import steering.overlay_io as overlay_io
import tests.conftest as shared_fixtures
from errors import FeatureClustersError, FeatureVocabError
from nanochat import checkpoint_manager
from nanochat.checkpoint_manager import ModelPhase, ModelSource
from nanochat.common import COMPUTE_DTYPE
from nanochat.engine import Engine
from scripts.eval import build_eval_context_cache as builder
from scripts.eval import chat_eval
from scripts.eval import context_cache_common as common
from scripts.eval import validate_eval_context_cache as validator
from scripts.eval.build_eval_context_cache import write_context_parquet, write_context_records
from steering.conditional.encoder_decoder.eval_adapter import TargetAwareEncoderDecoderEvalWrapper
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
)
from steering.conditional.encoder_decoder.sft_selection import (
    SegmentSelectionContext,
    select_conversation_segments,
)
from steering.conditional.eval_conditioning import EvalConditioning
from steering.conditional.feature_mask import FeatureMask, FeatureMaskMode
from steering.conditional.feature_pipeline import build_feature_pipeline_from_cfg_block
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.missing_percentile import (
    MissingPercentilePolicy,
    MissingPercentileResolver,
    percentile_table_coverage,
)
from steering.conditional.sampling import SamplingBundle
from steering.eval import context_cache_io as context_io
from steering.eval import target_real_runner
from steering.eval.cache_identity import cache_identity_from_config
from steering.eval.context_cache_build import (
    build_context_entry_from_extracted_segments,
    turn_source_segments,
)
from steering.eval.context_cache_io import (
    ContextCacheEntry,
    SafetensorsContextFeatureProvider,
    save_context_cache,
)
from steering.eval.context_enumeration import (
    dedupe_context_records,
    enumerate_chat_target_contexts,
    turn_supervision_interval,
)
from steering.eval.feature_config import resolve_next_segment_conditioning_offset
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    MultiTargetSegments,
    PooledTargetRealConfig,
    SingleEntryContextProvider,
    SingleTargetSegment,
    TargetConditioningMode,
)
from steering.eval.target_real_runner import build_target_real_artifacts_for_example
from steering.eval.target_segments import pool_segments_to_dict
from steering.eval.turn_context_contract import turn_context_contract_hash
from steering.eval.turn_context_routing import prepare_turn_context
from steering.feature_io import load_features
from steering.refined_types import NonNegativeInt, PositiveInt
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY
from tasks import arc, gsm8k, humaneval, mmlu, spellingbee
from tests.conftest import TEST_CACHE_IDENTITY, build_tiny_decoder, build_tiny_encdec
from tests.steering.chatcore_cache_helpers import ByteChatTokenizer, ChatTokenizer
from tests.steering.chatcore_checkpoint_helpers import (
    PRODUCTION_MODEL_TAG,
    PRODUCTION_TAG,
    SFT_LEAF,
    stage_checkpoint,
)
from tests.steering.test_context_cache_in_memory_conversion import _pipeline
from tests.steering.test_turn_context_cache_build import (
    StraddlingSourceTokenizer,
    UnroutedStraddlingSourceTokenizer,
    build_fixture,
)
from tests.steering.test_turn_eval_adapter import turn_model


def stable_safetensors_save(tensors, metadata=None):
    """Freeze Rust HashMap header ordering, an independent serialization entropy source."""
    raw = safetensors_save(tensors, metadata=metadata)
    size = int.from_bytes(raw[:8], "little")
    header = json.dumps(json.loads(raw[8:8 + size]), sort_keys=True, separators=(",", ":")).encode()
    header += b" " * (-len(header) % 8)
    return len(header).to_bytes(8, "little") + header + raw[8 + size:]


def test_fixed_prechange_payload(tmp_path, monkeypatch):
    # kills: any change to fixed enumeration, conversion, routing or model outputs.
    monkeypatch.setattr(context_io.time, "strftime", lambda *args: "2026-09-10T00:00:00Z")
    monkeypatch.setattr(overlay_io, "safetensors_save", stable_safetensors_save)
    tokenizer = ChatTokenizer()
    conversation = {"messages": [
        {"role": "user", "content": "Q"},
        {"role": "assistant", "content": "AB"},
    ]}
    records = enumerate_chat_target_contexts(
        tokenizer=tokenizer, task_object=[conversation], max_problems=None, task_name="fixed",
    )
    jsonl = write_context_records(tmp_path / "contexts.jsonl", records)
    parquet = write_context_parquet(tmp_path / "parquet", records)
    entries = [build_context_entry_from_extracted_segments(
        token_ids=record.token_ids,
        segment_feature_ids=[t.tensor([1, 2, 3])],
        segment_feature_magnitudes=[t.tensor([0.2, 0.5, 0.9])],
        segment_char_starts=[0], segment_char_ends=[len(record.context_text) + 5],
        pipeline=_pipeline(), n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(4), tokenizer=tokenizer,
        source_bos_char_len=NonNegativeInt.parse(5),
    ) for record in records]
    cache = tmp_path / "cache" / "c128" / ("0" * 16)
    save_context_cache(cache, entries, identity=TEST_CACHE_IDENTITY,
                       n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
                       metadata={"model_max_seq_len": 128})
    provider = SafetensorsContextFeatureProvider(cache, identity=TEST_CACHE_IDENTITY)
    t.manual_seed(812)
    model = build_tiny_encdec()
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if "cross_attn" in name and parameter.ndim >= 2:
                parameter.normal_(std=0.04)
    full_ids, mask = tokenizer.render_conversation(conversation)
    context_ids = list(records[1].token_ids)
    artifacts = build_target_real_artifacts_for_example(
        model=model, tokenizer=tokenizer, provider=provider,
        trc=PooledTargetRealConfig(name="fixed", cache_dir=cache, pooling_op=PoolingOp.BUCKET_MAX),
        full_token_ids=full_ids, mask=mask, context_token_ids=context_ids,
    )
    wrapper = artifacts.wrapper
    assert isinstance(wrapper, TargetAwareEncoderDecoderEvalWrapper)
    assert isinstance(artifacts.target, SingleTargetSegment)
    routing = []
    original = wrapper._build_conditioning

    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        routing.append(result[1].tolist())
        return result

    monkeypatch.setattr(wrapper, "_build_conditioning", capture)
    with wrapper.conditioning_context(target_mask=t.tensor([mask], dtype=t.bool), context_token_ids=[context_ids]), t.no_grad():
        logits = wrapper(t.tensor([full_ids]))
    assert t.isfinite(logits).all()
    # Upstream added transient ownership fields; fixed entries keep their defaults.
    for entry in entries:
        assert entry.segment_mode is SegmentMode.fixed_tokens
        assert entry.token_assignments is None
    payload = {
        "jsonl": hashlib.sha256(jsonl.read_bytes()).hexdigest(),
        "parquet": hashlib.sha256(parquet.read_bytes()).hexdigest(),
        "entries": [{
            "token_ids": entry.token_ids,
            "segments": [asdict(segment) for segment in entry.segments],
        } for entry in entries],
        "routing": routing,
        "logits": hashlib.sha256(logits.contiguous().numpy().tobytes()).hexdigest(),
        "greedy": logits.argmax(-1).tolist(),
        "target": asdict(artifacts.target),
    }
    # Frozen from ORIGINAL production HEAD 884dd39e4 before any implementation edits.
    expected = json.loads(
        '{"entries": [{"segments": [{"buckets": [0, 1], "char_end": 62, "char_start": 0, "feature_ids": [3, 5]}], "token_ids": [0, 1, 97, 2, 3, 81, 82, 4]}, {"segments": [{"buckets": [0, 1], "char_end": 45, "char_start": 0, "feature_ids": [3, 5]}], "token_ids": [0, 1, 97, 2, 3]}], "greedy": [[23, 17, 38, 65, 17, 18, 9, 124]], "jsonl": "827a99c93ecfdd2820ab105807c233cb8df64b68c3e08613308044be0622b3a4", "logits": "65bf399d1de123b7ac351239e97293c855ff00dc8c1c5669f31d73f55818fa0a", "parquet": "5e755d613af930b5b7f95282d5749e19016e7ad92025d25f84b51688eb7a828e", "routing": [[[0, 0, 0, 0, 1, 1, 1, 1]]], "target": {"features": {"3": 0, "5": 1}}}'
    )
    # The model computes in the process-wide COMPUTE_DTYPE (NANOCHAT_DTYPE, else
    # CUDA availability); only the logits digest depends on it. An unknown dtype
    # fails loud rather than silently skipping the fixed-path gate.
    frozen_logits = {
        str(t.bfloat16): "65bf399d1de123b7ac351239e97293c855ff00dc8c1c5669f31d73f55818fa0a",
        str(t.float32): "3790d42f73a0d122c333cc856d6382ece6a7a5fd630dd2b1fd13ce2428dc5204",
    }
    assert str(COMPUTE_DTYPE) in frozen_logits, f"golden not frozen for {COMPUTE_DTYPE}; freeze it"
    expected["logits"] = frozen_logits[str(COMPUTE_DTYPE)]
    assert json.loads(json.dumps(payload, sort_keys=True)) == expected
    # kills: fixed serialization or KV decode changing despite teacher-forced parity.
    with wrapper.conditioning_context(context_token_ids=[context_ids]), t.no_grad():
        generated, masks = Engine(wrapper, tokenizer).generate_batch(context_ids, max_tokens=3, temperature=0.0)
    serialized = {
        "manifest": hashlib.sha256((cache / "manifest.json").read_bytes()).hexdigest(),
        "shard": hashlib.sha256((cache / "contexts_00000.safetensors.zst").read_bytes()).hexdigest(),
        "generated": generated, "masks": masks, "routing": routing,
    }
    # Additional golden from an untouched git archive of 884dd39e4.
    assert serialized == json.loads('{"generated": [[0, 1, 97, 2, 3, 17, 66, 114]], "manifest": "14c6ccf1418e7738aac7a9910eda7cbb6369c355853815d3565a66978f428ef9", "masks": [[0, 0, 0, 0, 0, 1, 1, 1]], "routing": [[[0, 0, 0, 0, 1, 1, 1, 1]], [[0, 0, 0, 0, 1]], [[1]], [[1]], [[1]]], "shard": "e6ead498f7cce4dc6341130304812304cf3dc87f8c442621b8206da38447dcd8"}')


def local_chat_tasks(tmp_path, monkeypatch, *, gsm_problems=1):
    """Replace dataset acquisition only; use each real task's renderer and evaluator."""
    rows = {
        gsm8k: [{"question": "What is 1+1?", "answer": "<<1+1=2>> so #### 2"}],
        mmlu: [{"question": "Choose one", "choices": ["one", "two", "three", "four"], "answer": 0, "subject": "unit"}],
        arc: [{"question": "Choose ARC", "choices": {"text": ["yes", "no"], "label": ["A", "B"]}, "answerKey": "B"}],
        humaneval: [{"prompt": "def f():", "canonical_solution": "    return 2", "entry_point": "f", "test": "def check(fn):\n    assert fn() == 2"}],
    }
    if gsm_problems != 1:
        rows[gsm8k] = [{"question": f"What is {index}+{index}?", "answer": f"#### {2 * index}"}
                      for index in range(1, gsm_problems + 1)]
    for module, local_rows in rows.items():
        dataset = Dataset.from_list(local_rows)
        monkeypatch.setattr(module, "load_dataset", lambda *args, _ds=dataset, **kwargs: _ds)
    monkeypatch.setattr(arc, "load_dataset", lambda _name, subset, **kwargs:
        Dataset.from_list([{**rows[arc][0], "question": f"Choose {subset}"}]))
    word_path = tmp_path / "words.txt"
    word_path.write_text("bee\n")
    monkeypatch.setattr(spellingbee, "resolve_word_list_path", lambda: word_path)
    return {"GSM8K": gsm8k.GSM8K("main", "test"), "MMLU": mmlu.MMLU("all", "test"),
            "ARC-Easy": arc.ARC("ARC-Easy", "test"), "ARC-Challenge": arc.ARC("ARC-Challenge", "test"),
            "HumanEval": humaneval.HumanEval(), "SpellingBee": spellingbee.SpellingBee(size=1, split="test")}


@pytest.mark.parametrize("straddle,unrouted", [(False, False), (True, False), (True, True)],
    ids=["exact-boundary", "gemma-straddle", "unrouted-straddle"])
def test_turn_cache_pipeline_with_all_real_chat_tasks(tmp_path, monkeypatch, straddle, unrouted):
    # kills: task-shaped mocks, missing full/prefix rows, tool-mask compression, cadence-dependent static selection.
    tokenizer = ChatTokenizer()
    tasks = local_chat_tasks(tmp_path, monkeypatch)
    records = []
    for name, task in tasks.items():
        records.extend(enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=task,
            max_problems=1, task_name=name, segment_mode=SegmentMode.turn_delimited_fixed_tokens))
    cfg, records, _paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch, records=records,
        source=UnroutedStraddlingSourceTokenizer() if unrouted else StraddlingSourceTokenizer() if straddle else None)
    manifest = builder.build_cache_from_config(cfg)
    standalone = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": False})
    assert isinstance(standalone, DictConfig)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    summary = validator.validate_cache_from_config(standalone)
    builder.validate_cache_from_config(cfg)
    assert summary["cached_contexts"] == len(records)
    assert len(records) == 12
    assert {row.task for row in records} == set(tasks)
    assert len(summary["turn_target_checks"]) == 6
    provider = SafetensorsContextFeatureProvider(manifest.parent,
        identity=common.cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=SegmentMode.turn_delimited_fixed_tokens, expected_turn_contract=contract)
    by_key = {row.key: row for row in records}
    for check in summary["turn_target_checks"]:
        full = by_key[check["key"]]
        segments = provider.get(full.token_ids)
        assert segments is not None
        oracle = build_full_sequence_segment_routing(ids=full.token_ids, tokenizer=tokenizer,
            seg_features=([t.tensor(seg.feature_ids) for seg in segments], [t.tensor(seg.buckets) for seg in segments],
                          [seg.char_start for seg in segments], [seg.char_end for seg in segments]),
            source_bos_char_len=3)
        start = check["target_start_pos"]
        _, end = turn_supervision_interval(full.token_ids, tokenizer)
        first_source, last_source = int(oracle.seg_map[start]) - 1, int(oracle.seg_map[end - 1]) - 1
        target_segments = segments[first_source:last_source + 1]
        assert check["lockstep_assignments"] == (oracle.seg_map[start:end] - 1 - first_source).tolist()
        assert check["lockstep_segments"] == [dict(zip(seg.feature_ids, seg.buckets, strict=True)) for seg in target_segments]
        assert check["pooled_features"] == pool_segments_to_dict(target_segments, op=PoolingOp.BUCKET_MAX)
        prefix = full.token_ids[:start + 1]
        prefix_segments = provider.get(prefix)
        assert prefix_segments is not None
        prefix_oracle = build_full_sequence_segment_routing(ids=prefix, tokenizer=tokenizer,
            seg_features=([t.tensor(seg.feature_ids) for seg in prefix_segments],
                          [t.tensor(seg.buckets) for seg in prefix_segments],
                          [seg.char_start for seg in prefix_segments], [seg.char_end for seg in prefix_segments]),
            source_bos_char_len=3)
        keep = int(prefix_oracle.seg_map[:start].max())
        prompt = prepare_turn_context(prefix, prefix_segments, tokenizer, source_bos_char_len=3)
        assert prompt.segments == prefix_segments[:keep]
        assert t.equal(prompt.token_assignments, prefix_oracle.seg_map[:start] - 1)
        assert keep < len(prefix_segments)
        marker_start = check["conditioning_char_start"]
        assert segments[first_source].char_start == prefix_segments[keep + int(unrouted)].char_start == marker_start + int(straddle)
        if unrouted:
            assert first_source == keep + 1
            assert set(range(keep, len(prefix_segments))) == {keep, keep + 1}
            assert keep not in (prefix_oracle.seg_map - 1).tolist()
            assert keep not in (oracle.seg_map - 1).tolist()
            assert segments[keep].char_start == prefix_segments[keep].char_start == marker_start - 5
            assert segments[keep].char_end == prefix_segments[keep].char_end == marker_start + 1
            assert all(seg.char_end > marker_start for seg in prefix_segments[keep:])
            assert any(seg.char_start < marker_start for seg in prefix_segments[keep:])
        elif straddle:
            assert not any(seg.char_start == marker_start for seg in prefix_segments)
            assert prompt.segments[-1].char_start < marker_start < prompt.segments[-1].char_end
        assert len(check["plans"]) == 3  # real pooled-current, pooled-next and lockstep-current plans
        if full.task in {"GSM8K", "SpellingBee"}:
            output = full.token_ids.index(tokenizer.encode_special("<output_start>"))
            output_end = full.token_ids.index(tokenizer.encode_special("<output_end>"))
            assert len(check["lockstep_assignments"]) > output_end - start
            assert oracle.seg_map[output] != oracle.seg_map[output - 1]
    meta_path = cfg.checkpoint_meta_path
    meta = json.loads(Path(meta_path).read_text())
    meta["turn_self_cond_segment_size_rustbpe"] = 7
    with open(meta_path, "w") as handle:
        json.dump(meta, handle)
    # Cadence does not affect semantics, but edited metadata is a new producer snapshot.
    with pytest.raises(ValueError, match="checkpoint.*sha256"):
        validator.validate_cache_from_config(standalone)
    assert common.turn_contract_from_config(cfg, tokenizer)[0].semantic == contract.semantic
    # The complete entry-point tests below retain these actual task scorers.
    assert tasks["GSM8K"].evaluate(tasks["GSM8K"][0], "#### 2")
    assert tasks["MMLU"].evaluate(tasks["MMLU"][0], "A")
    assert tasks["ARC-Easy"].evaluate(tasks["ARC-Easy"][0], "B")
    assert tasks["ARC-Challenge"].evaluate(tasks["ARC-Challenge"][0], "B")


class TaskChatTokenizer(ChatTokenizer):
    """Compact alphabet for real task rows, disjoint from every chat special ID."""

    def __init__(self, tasks):
        alphabet = set()
        for task in tasks.values():
            for message in task[0]["messages"]:
                content = message["content"]
                if isinstance(content, str):
                    alphabet.update(content)
                else:
                    for part in content:
                        alphabet.update(part["text"])
        self.characters = {char: index + 16 for index, char in enumerate(sorted(alphabet))}
        self.inverse = {value: key for key, value in self.characters.items()}
        assert max(self.inverse) < 128

    def encode(self, text, **kwargs):
        return [self.characters[char] for char in text]

    def decode(self, ids):
        specials = {value: key for key, value in self._specials.items()}
        return "".join(specials.get(token, self.inverse.get(token, f"<unused{token}>")) for token in ids)


def test_fixed_chat_eval_main_prechange_payload(tmp_path, monkeypatch):
    # kills: fixed task dispatch, categorical restrictions, generation/scoring or result aggregation drift.
    tasks = local_chat_tasks(tmp_path, monkeypatch)
    tokenizer = TaskChatTokenizer(tasks)
    records = dedupe_context_records([row for name, task in tasks.items() for row in
        enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=task, max_problems=1, task_name=name)])
    entries = [build_context_entry_from_extracted_segments(
        token_ids=record.token_ids, segment_feature_ids=[t.tensor([1, 2, 3])],
        segment_feature_magnitudes=[t.tensor([0.2, 0.5, 0.9])],
        segment_char_starts=[0], segment_char_ends=[len(record.context_text) + 5],
        pipeline=_pipeline(), n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
        tokenizer=tokenizer, source_bos_char_len=NonNegativeInt.parse(5),
    ) for record in records]
    cache = tmp_path / "cache" / "c2048" / ("0" * 16)
    save_context_cache(cache, entries, identity=TEST_CACHE_IDENTITY, n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(4), metadata={"model_max_seq_len": 2048})
    monkeypatch.setattr(shared_fixtures, "TINY_SEQ_LEN", 2048)
    monkeypatch.setattr(shared_fixtures, "TINY_N_SAE_FEATURES", PositiveInt.parse(8))
    t.manual_seed(551)
    model = build_tiny_encdec()
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if "cross_attn" in name and parameter.ndim >= 2:
                parameter.normal_(std=0.04)
    meta = {"checkpoint_dir": str(tmp_path), "user_config": {"next_segment_conditioning_offset": False}}
    (tmp_path / "meta_001116.json").write_text(json.dumps(meta))
    monkeypatch.setattr(chat_eval, "get_checkpoint_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(chat_eval, "load_model", lambda *args, **kwargs: (model, tokenizer, meta))
    monkeypatch.setattr(chat_eval, "compute_init", lambda _: (False, 0, 0, 1, t.device("cpu")))
    monkeypatch.setattr(chat_eval, "compute_cleanup", lambda: None)
    monkeypatch.setattr(chat_eval, "get_report", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setattr(target_real_runner, "get_report", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    forwards = []
    original_forward = model.forward

    def capture_forward(*args, **kwargs):
        logits = original_forward(*args, **kwargs)
        assert t.isfinite(logits).all()
        forwards.append(hashlib.sha256(logits.detach().contiguous().numpy().tobytes()).hexdigest())
        return logits

    monkeypatch.setattr(model, "forward", capture_forward)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/eval/chat_eval.yaml")
    cfg.source = "encdec"
    cfg.step = 1116
    cfg.device_type = "cpu"
    cfg.dtype = "float32"
    cfg.max_problems = 1
    cfg.max_new_tokens = 3
    cfg.model_max_seq_len = 2048
    cfg.conditioning = {"identity": TEST_CACHE_IDENTITY.as_dict(), "n_sae_features": 8, "num_buckets": 4}
    cfg.target_real_configs = [{"name": "pooled", "mode": "pooled", "cache_dir": str(tmp_path / "cache")}]
    chat_eval.main.__wrapped__(cfg)
    output = (tmp_path / "chat_eval/target_real/pooled.yaml").read_text()
    result = yaml.safe_load(output)
    assert result["kind"] == "target_real_pooled"
    assert result["target_conditioning_mode"] == "current"
    frozen = {"yaml": output.replace(str(tmp_path), "<fixture>"), "forwards": forwards}
    # Frozen by executing this fixture against untouched git archive 884dd39e4.
    expected = {"yaml": "kind: target_real_pooled\ntarget_conditioning_mode: current\ncache_dir: <fixture>/cache\npooling_op: bucket_max\nseeds:\n  adhoc:\n    results:\n      ARC-Easy: 1.0\n      ARC-Challenge: 1.0\n      MMLU: 0.0\n      GSM8K: 0.0\n      HumanEval: 0.0\n      SpellingBee: 0.0\n    chatcore_metric: 0.2777777777777778\n"}
    frozen_forwards = {
        str(t.bfloat16): ["7f064157cd42f8988b3c8b35989ac3219e1f0224e3be9a47be389c12cde08016", "fb3a36bd971e39b47869ce283f0c29d685361e16576db8e4d093d14f0f58a507", "ba58a3ea929ace7858b9a4a3840e91602c5269bd4038faf5bf7ea0b076d37b45", "93ce34bb93142a83f51c236e0e23c02dc57dfbfc9936f73bb76568121a13f48c", "219c580260f8d9d2841ee6585ee74c0ee67280a79b6689c2c8f7c1dee4fc24be", "6625ba85c9aabbb199bad9d3e7947331cdbba417506e2096808bed5474349f11", "f825c6548b197568fb4427209f608f87bc4f34c9a5fc03290f0589fe3a183d9e", "6e6d6127733823f63b6e59bceea1e72f314cd53902139ec8be493eeb1f3ca64c", "219c580260f8d9d2841ee6585ee74c0ee67280a79b6689c2c8f7c1dee4fc24be", "6625ba85c9aabbb199bad9d3e7947331cdbba417506e2096808bed5474349f11", "f825c6548b197568fb4427209f608f87bc4f34c9a5fc03290f0589fe3a183d9e", "131563ce3680c2745d006bdf4c179c71b49dcae954c26725c8b0f2bff5f8fc6a", "219c580260f8d9d2841ee6585ee74c0ee67280a79b6689c2c8f7c1dee4fc24be", "6625ba85c9aabbb199bad9d3e7947331cdbba417506e2096808bed5474349f11", "f825c6548b197568fb4427209f608f87bc4f34c9a5fc03290f0589fe3a183d9e"],
        str(t.float32): ["6e0bbd209651cb7fd9ad78d4829ea5b14f27d8d6408d3c6ffd241833e508b1d3", "922774da4c45872b9c846752832eaeb56709b819514a6efcb5e538370372e973", "6ba04ef0f6887a2803d16137d88ac68c5c24e2d18d3b57f690e9a659a5c0896e", "7a1440cd110254c6235f49049cefdadbcc11496a1e8325a7c8b5d242c614538d", "fe75c60e2dab2e026f84dfd154ec29106f011e5942163d9dcd91b3867b7b14a4", "3191bd4b523dd8dac3737d649dfc242424af143152fd9cf5a4b2d0fa9e4096de", "50dade058265e4c071b22cb014c57891aed27407d9bc02654f4edda21b93a226", "f63f51c522fc1792bde479991f161cd8633d315c41dd57c8312496168ebe2608", "fe75c60e2dab2e026f84dfd154ec29106f011e5942163d9dcd91b3867b7b14a4", "3191bd4b523dd8dac3737d649dfc242424af143152fd9cf5a4b2d0fa9e4096de", "50dade058265e4c071b22cb014c57891aed27407d9bc02654f4edda21b93a226", "1491ff35a0f56eef2f1cb12efbb34220e1d347c5767905cc1047fceea8ff34d3", "fe75c60e2dab2e026f84dfd154ec29106f011e5942163d9dcd91b3867b7b14a4", "3191bd4b523dd8dac3737d649dfc242424af143152fd9cf5a4b2d0fa9e4096de", "50dade058265e4c071b22cb014c57891aed27407d9bc02654f4edda21b93a226"],
    }
    assert str(COMPUTE_DTYPE) in frozen_forwards, f"golden not frozen for {COMPUTE_DTYPE}; freeze it"
    expected["forwards"] = frozen_forwards[str(COMPUTE_DTYPE)]
    assert frozen == expected
    assert len(result["seeds"]["adhoc"]["results"]) == 6
    assert result["seeds"]["adhoc"]["chatcore_metric"] == (1 + 1 - 1 / 3) / 6


def test_fixed_base_source_main_loads_and_writes_the_configured_directory(tmp_path, monkeypatch):
    # kills: a fixed run only ever proven against a stubbed load_model, and any
    # fixed load/output that follows the RESOLVED instead of the configured path.
    tasks = {"MMLU": local_chat_tasks(tmp_path, monkeypatch)["MMLU"]}
    tokenizer = TaskChatTokenizer(tasks)
    records = dedupe_context_records([row for name, task in tasks.items() for row in
        enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=task, max_problems=1, task_name=name)])
    entries = [build_context_entry_from_extracted_segments(
        token_ids=record.token_ids, segment_feature_ids=[t.tensor([1, 2, 3])],
        segment_feature_magnitudes=[t.tensor([0.2, 0.5, 0.9])],
        segment_char_starts=[0], segment_char_ends=[len(record.context_text) + 5],
        pipeline=_pipeline(), n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4),
        tokenizer=tokenizer, source_bos_char_len=NonNegativeInt.parse(5),
    ) for record in records]
    cache = tmp_path / "cache" / "c2048" / ("0" * 16)
    save_context_cache(cache, entries, identity=TEST_CACHE_IDENTITY, n_sae_features=PositiveInt.parse(8),
        num_buckets=PositiveInt.parse(4), metadata={"model_max_seq_len": 2048})
    model = turn_model(monkeypatch)
    meta = {
        "step": 1116, "sft_segment_mode": SegmentMode.fixed_tokens.value,
        "encoder_config": asdict(model.enc_config), "xattn_config": asdict(model.xattn_config),
        "num_buckets": int(model.encoder.config.num_buckets),
        "user_config": {"next_segment_conditioning_offset": False},
        ONLINE_SAE_LINEAGE_META_KEY: {
            "schema_version": 1, "source_segment_mode": SegmentMode.fixed_tokens.value,
            "negative_selection": {"distribution": "geometric", "mean": 1.0, "std": 1.0,
                                   "min_k": 1, "max_k": 2},
        },
    }
    checkpoint_path, loaded = stage_checkpoint(tmp_path, monkeypatch, meta, model, tokenizer, real_loader=True)
    # base_checkpoints/<tag> is the operator's configured tree; its physical
    # files live under the encdec root, so configured != resolved.
    encdec_root = checkpoint_path.parents[2]
    encdec_root.with_name("base_checkpoints").symlink_to(encdec_root, target_is_directory=True)
    configured = encdec_root.with_name("base_checkpoints") / PRODUCTION_MODEL_TAG
    assert configured.resolve() == checkpoint_path.parent != configured
    monkeypatch.setattr(chat_eval, "compute_init", lambda _: (False, 0, 0, 1, t.device("cpu")))
    monkeypatch.setattr(chat_eval, "compute_cleanup", lambda: None)
    monkeypatch.setattr(chat_eval, "get_report", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setattr(target_real_runner, "get_report", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    assert chat_eval.load_model is checkpoint_manager.load_model
    real_load, metas = chat_eval.load_model, []

    def capture(*args, **kwargs):
        built, loaded_tokenizer, loaded_meta = real_load(*args, **kwargs)
        assert built is not model  # Real construction, not the fixture's stand-in loader.
        metas.append(loaded_meta)
        return built, loaded_tokenizer, loaded_meta

    monkeypatch.setattr(chat_eval, "load_model", capture)
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/eval/chat_eval.yaml")
    cfg.source = ModelSource.BASE.value
    cfg.model_tag = PRODUCTION_MODEL_TAG
    cfg.step = 1116
    cfg.task_name = "MMLU"
    cfg.device_type = "cpu"
    cfg.dtype = "float32"
    cfg.max_problems = 1
    cfg.max_new_tokens = 3
    cfg.model_max_seq_len = 2048
    cfg.conditioning = {"identity": TEST_CACHE_IDENTITY.as_dict(), "n_sae_features": 8, "num_buckets": 4}
    cfg.target_real_configs = [{"name": "pooled", "mode": "pooled", "cache_dir": str(tmp_path / "cache")}]
    chat_eval.main.__wrapped__(cfg)
    assert loaded == [(configured, 1116)]
    assert metas[0]["checkpoint_dir"] == str(configured)
    body = yaml.safe_load((configured / "chat_eval/target_real/pooled.yaml").read_text())["seeds"]["adhoc"]
    assert set(body["results"]) == {"MMLU"}
    # The fixed body publishes neither the turn receipt nor shard ranges.
    assert set(body) == {"results"}


def turn_main_fixture(tmp_path, monkeypatch, *, lockstep=False, mode="current", production=False, clustered=False,
                      straddle=True, gsm_problems=1, cache_max_problems=1, real_loader=False):
    tasks = local_chat_tasks(tmp_path, monkeypatch, gsm_problems=gsm_problems)
    if production:
        tasks = {"MMLU": tasks["MMLU"]}
    tokenizer = ByteChatTokenizer()
    records = [row for name, task in tasks.items() for row in enumerate_chat_target_contexts(
        tokenizer=tokenizer, task_object=task, max_problems=cache_max_problems, task_name=name,
        segment_mode=SegmentMode.turn_delimited_fixed_tokens)]
    cache_cfg, records, paths, contract, tokenizer, source = build_fixture(
        tmp_path, monkeypatch, records=records, tokenizer=tokenizer, production=production, clustered=clustered,
        source=StraddlingSourceTokenizer() if straddle else None)
    model = turn_model(monkeypatch, contract=contract if production or clustered else None)
    meta = json.loads(Path(cache_cfg.checkpoint_meta_path).read_text())
    meta["step"] = 1116
    meta["user_config"]["next_segment_conditioning_offset"] = mode == "next"
    checkpoint_path, loaded = stage_checkpoint(tmp_path, monkeypatch, meta, model, tokenizer, real_loader=real_loader)
    cache_cfg.checkpoint_meta_path = str(checkpoint_path)
    manifest = builder.build_cache_from_config(cache_cfg)
    standalone = OmegaConf.merge(cache_cfg, {"cache_dir": cache_cfg.output_dir, "enumerate_evals": False})
    assert isinstance(standalone, DictConfig)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    summary = validator.validate_cache_from_config(standalone)
    builder.validate_cache_from_config(cache_cfg)
    meta["checkpoint_dir"] = str(checkpoint_path.parent)
    monkeypatch.setattr(chat_eval, "load_model", checkpoint_manager.load_model)
    monkeypatch.setattr(chat_eval, "compute_init", lambda _: (False, 0, 0, 1, t.device("cpu")))
    monkeypatch.setattr(chat_eval, "compute_cleanup", lambda: None)
    monkeypatch.setattr(chat_eval, "get_report", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    monkeypatch.setattr(target_real_runner, "get_report", lambda **kwargs: SimpleNamespace(log=lambda **kwargs: None))
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/eval/chat_eval.yaml")
    cfg.source = ModelSource.ENCDEC.value
    cfg.model_tag = PRODUCTION_MODEL_TAG
    cfg.step = 1116
    cfg.device_type = "cpu"
    cfg.dtype = "float32"
    cfg.max_problems = 1
    cfg.max_new_tokens = 3
    cfg.model_max_seq_len = 2048
    cfg.segment_mode = SegmentMode.turn_delimited_fixed_tokens.value
    cfg.conditioning = {"identity": {}}
    cfg.target_real_configs = [{"name": "target", "mode": "lockstep" if lockstep else "pooled", "cache_dir": cache_cfg.output_dir}]
    return SimpleNamespace(cfg=cfg, model=model, tokenizer=tokenizer, meta=meta, tasks=tasks,
        records=records, paths=paths, cache_cfg=cache_cfg, cache=manifest.parent, contract=contract, summary=summary,
        checkpoint_path=checkpoint_path, loaded=loaded, source=source)


@pytest.mark.parametrize("lockstep,mode", [(False, TargetConditioningMode.CURRENT),
    (False, TargetConditioningMode.NEXT), (True, TargetConditioningMode.CURRENT)])
def test_clustered_turn_chat_eval_main(tmp_path, monkeypatch, lockstep, mode):
    # kills: rejecting clustered turn metadata before mode resolution, or mask-only consumer/model vocabularies.
    fixture = turn_main_fixture(tmp_path, monkeypatch, lockstep=lockstep, mode=mode.value, clustered=True)
    assert fixture.model.feature_vocab == fixture.contract.feature_vocab
    assert fixture.model.feature_vocab_size == 5
    assert fixture.model.encoder.config.n_sae_features == 8
    assert fixture.model.encoder.feature_embed.weight.shape[0] == 5
    fixture.model.assert_vocab_consistent()
    translations = []
    original_to_compact = fixture.model.encoder._to_compact

    def capture_translation(ids, active):
        compact = original_to_compact(ids, active)
        original_ids, rows = ids[active].tolist(), compact[active].tolist()
        vocab = fixture.model.feature_vocab
        assert vocab is not None
        assert rows == [vocab.original_ids.index(fid) for fid in original_ids]
        translations.extend(zip(original_ids, rows, strict=True))
        return compact

    monkeypatch.setattr(fixture.model.encoder, "_to_compact", capture_translation)
    forwards = []
    original = fixture.model.forward

    def capture(*args, **kwargs):
        logits = original(*args, **kwargs)
        assert t.isfinite(logits).all()
        forwards.append(logits.shape)
        return logits

    monkeypatch.setattr(fixture.model, "forward", capture)
    chat_eval.main.__wrapped__(fixture.cfg)
    payload = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    assert set(payload["seeds"]["adhoc"]["results"]) == set(fixture.tasks)
    assert payload["target_conditioning_mode"] == mode.value
    assert payload["kind"] == ("target_real_lockstep" if lockstep else "target_real_pooled")
    assert forwards
    assert any(fid >= 5 and fid != row for fid, row in translations)
    assert fixture.loaded == [(fixture.checkpoint_path.parent, 1116)]


def test_fixed_compacted_chat_eval_retains_exact_deferred_error(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    fixed_meta = {key: value for key, value in fixture.meta.items() if key != ONLINE_SAE_LINEAGE_META_KEY}
    fixed_meta["sft_segment_mode"] = SegmentMode.fixed_tokens.value
    fixture.checkpoint_path.write_text(json.dumps(fixed_meta))
    # A CLI request for turn mode must never grant a fixed checkpoint admission.
    assert fixture.cfg.segment_mode == SegmentMode.turn_delimited_fixed_tokens.value
    with pytest.raises(NotImplementedError) as expected:
        EvalConditioning.deferred("chat_eval").assert_loadable(
            fixture.model.feature_vocab, checkpoint_dir=str(fixture.checkpoint_path.parent))
    with pytest.raises(NotImplementedError) as actual:
        chat_eval.main.__wrapped__(fixture.cfg)
    assert str(actual.value) == str(expected.value)
    assert "feature vocabulary of 5 of 8 SAE features" in str(actual.value)
    assert not fixture.loaded


def test_compacted_turn_admission_uses_checkpoint_mode_without_config_override(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    fixture.cfg.segment_mode = None
    fixture.cfg.task_name = "MMLU"
    chat_eval.main.__wrapped__(fixture.cfg)
    assert fixture.loaded == [(fixture.checkpoint_path.parent, 1116)]
    payload = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    assert set(payload["seeds"]["adhoc"]["results"]) == {"MMLU"}


def test_turn_main_rejects_mode_change_during_load(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    original = chat_eval.load_model

    def changed_load(*args, **kwargs):
        model, tokenizer, meta = original(*args, **kwargs)
        del meta[ONLINE_SAE_LINEAGE_META_KEY]
        meta["sft_segment_mode"] = SegmentMode.fixed_tokens.value
        return model, tokenizer, meta

    monkeypatch.setattr(chat_eval, "load_model", changed_load)
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *args, **kwargs: pytest.fail("changed mode reached encoder"))
    with pytest.raises(ValueError, match="segment_mode changed during model load"):
        chat_eval.main.__wrapped__(fixture.cfg)


def test_turn_main_asserts_loaded_vocab_buffers_before_forward(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    weights_path = fixture.checkpoint_path.with_name("model_001116.pt")
    weights = t.load(weights_path, weights_only=True)
    # Same shape and Python vocabulary, different persistent row identity. Only
    # the complete loaded-model assertion detects this before encoder forward.
    weights["state_dict"]["encoder.feature_vocab_ids"][1] = 1
    t.save(weights, weights_path)
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *args, **kwargs: pytest.fail("invalid vocabulary reached encoder"))
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("invalid vocabulary reached decoder"))
    with pytest.raises(FeatureVocabError, match="encoder.feature_vocab_ids disagrees"):
        chat_eval.main.__wrapped__(fixture.cfg)


@pytest.mark.parametrize("field", ["feature_mask", "feature_clusters_canonical"])
def test_turn_main_checks_cache_metadata_against_loaded_contract(tmp_path, monkeypatch, field):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    path = fixture.cache / "manifest.json"
    manifest = json.loads(path.read_text())
    # Keep the semantic turn contract intact; corrupt only the duplicated cache
    # metadata that conditioned_eval_cache_contract_metadata must bind.
    manifest["metadata"][field] = {}
    path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *args, **kwargs: pytest.fail("bad metadata reached encoder"))
    with pytest.raises(RuntimeError, match=field):
        chat_eval.main.__wrapped__(fixture.cfg)


def test_turn_main_rejects_oov_id_in_unselected_cache_row(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    path = fixture.cache / "manifest.json"
    manifest = json.loads(path.read_text())
    shard = manifest["shards"][-1]
    shard_path = fixture.cache / shard["file"]
    tensors = overlay_io.load_overlay_shard_bytes(shard_path.read_bytes())
    tensors[context_io.T_FEATURE_IDS][-1] = 1  # consolidated away, but inside SAE width
    overlay_io.save_overlay_shard(tensors, shard_path)
    payload = shard_path.read_bytes()
    shard["sha256"] = hashlib.sha256(payload).hexdigest()
    path.write_text(json.dumps(manifest))
    fixture.cfg.task_name = "MMLU"  # The bad final row belongs to SpellingBee.
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *args, **kwargs: pytest.fail("bad cache reached encoder"))
    with pytest.raises(RuntimeError, match="cached feature ID 1 is outside"):
        chat_eval.main.__wrapped__(fixture.cfg)


@pytest.mark.parametrize("lockstep", [False, True], ids=["pooled", "lockstep"])
def test_turn_main_rejects_constructed_oov_target_before_forward(tmp_path, monkeypatch, lockstep):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True, lockstep=lockstep)
    original = target_real_runner.build_turn_target_for_real_config

    def inject_oov(**kwargs):
        target = original(**kwargs)
        if isinstance(target, SingleTargetSegment):
            return replace(target, features={**target.features, 1: 0})
        assert isinstance(target, MultiTargetSegments)
        return replace(target, segments=[*target.segments[:-1], {**target.segments[-1], 1: 0}])

    monkeypatch.setattr(target_real_runner, "build_turn_target_for_real_config", inject_oov)
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *args, **kwargs: pytest.fail("bad target reached encoder"))
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("bad target reached decoder"))
    with pytest.raises(RuntimeError, match="target segment .*outside.*vocabulary"):
        chat_eval.main.__wrapped__(fixture.cfg)


@pytest.mark.parametrize("lockstep", [False, True], ids=["pooled", "lockstep"])
def test_turn_pack_rejects_oov_context_after_provider_validation(tmp_path, monkeypatch, lockstep):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True, lockstep=lockstep)
    fixture.cfg.task_name = "MMLU"
    validated = []
    original_load = SafetensorsContextFeatureProvider._load_all_shards

    def load(provider):
        original_load(provider)
        validated.append(provider)

    original_pack = TargetAwareEncoderDecoderEvalWrapper._pack_encoder_inputs
    injected = []

    def inject_context(wrapper, row_payloads, *, device):
        assert validated
        payload = row_payloads[0]
        index = next(i for i, seg in enumerate(payload.context_segments) if seg.feature_ids)
        segment = payload.context_segments[index]
        assert 1 not in fixture.model.feature_vocab.original_ids
        assert set(segment.feature_ids) <= set(fixture.model.feature_vocab.original_ids)
        segments = list(payload.context_segments)
        segments[index] = replace(segment, feature_ids=(1, *segment.feature_ids[1:]))
        poisoned = replace(payload, context_segments=tuple(segments))
        injected.append(poisoned)
        return original_pack(wrapper, [poisoned, *row_payloads[1:]], device=device)

    monkeypatch.setattr(SafetensorsContextFeatureProvider, "_load_all_shards", load)
    monkeypatch.setattr(TargetAwareEncoderDecoderEvalWrapper, "_pack_encoder_inputs", inject_context)
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *a, **kw: pytest.fail("OOV context reached encoder entry"))
    with pytest.raises(RuntimeError, match="encoder segment .*outside.*vocabulary"):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert injected
    assert not (fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").exists()


def test_fixed_chat_eval_keeps_cluster_guard(tmp_path, monkeypatch):
    # kills: bypassing the existing fixed-path clustered metadata guard.
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    fixed_meta = {key: value for key, value in fixture.meta.items()
                  if key != ONLINE_SAE_LINEAGE_META_KEY}
    fixed_meta["sft_segment_mode"] = SegmentMode.fixed_tokens.value
    fixture.checkpoint_path.write_text(json.dumps(fixed_meta))
    fixture.cfg.segment_mode = SegmentMode.fixed_tokens.value
    monkeypatch.setattr(chat_eval, "load_model", lambda *args, **kwargs: (fixture.model, fixture.tokenizer, fixed_meta))
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("fixed clustered checkpoint reached forward"))
    with pytest.raises(FeatureClustersError, match="cannot consume a checkpoint/artifact trained with static"):
        chat_eval.main.__wrapped__(fixture.cfg)


@pytest.mark.parametrize("mutation,message", [
    ("seed", "conditioning.seed conflicts with checkpoint turn contract"),
    ("clusters", "turn ChatCORE does not support clustered/delta selection overrides"),
    ("mask-vocab", "feature vocabulary does not match"),
    ("same-size-vocab", "feature vocabulary does not match"),  # same vocabulary guard as mask-vocab
    ("cache-clusters", "turn context contract disagrees with checkpoint-derived expectations"),
    ("cache-seed", "turn context contract disagrees with checkpoint-derived expectations"),  # same contract guard
    ("marker", "turn cache is missing the expected clustered selection semantics"),
])
def test_clustered_turn_main_binds_checkpoint_cache_and_selection(tmp_path, monkeypatch, mutation, message):
    # kills: CLI cluster/seed authority, admitting changed clusters/seed at an unchanged path, or stale cache semantics.
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    if mutation == "seed":
        fixture.cfg.conditioning.seed = fixture.contract.lineage.selection_seed + 1
    elif mutation == "clusters":
        fixture.cfg.conditioning.feature_clusters = {"path": str(tmp_path / "clusters.json")}
    elif mutation == "same-size-vocab":
        wrong_vocab = FeatureVocab(8, (0, 1, 3, 6, 7))
        monkeypatch.setattr(type(fixture.model), "feature_vocab", property(lambda _: wrong_vocab))
    elif mutation == "mask-vocab":
        monkeypatch.setattr(type(fixture.model), "feature_vocab", property(lambda _: None))
    else:
        path = fixture.cache / "manifest.json"
        manifest = json.loads(path.read_text())
        if mutation == "marker":
            del manifest["metadata"][context_io.CLUSTER_SELECTION_SEMANTICS_KEY]
        elif mutation == "cache-seed":
            manifest["turn_contract"]["source_lineage"]["selection_seed"] += 1
        else:
            manifest["turn_contract"]["feature_clusters_canonical"]["clusters"] = [[1, 3], [2, 4, 5]]
        path.write_text(json.dumps(manifest))
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("invalid cluster binding reached forward"))
    with pytest.raises((ValueError, RuntimeError, FeatureVocabError), match=message):
        chat_eval.main.__wrapped__(fixture.cfg)


def test_production_contract_identity_and_consumer_admission(tmp_path, monkeypatch):
    # Production dimensions and compact vocabulary, with a tiny decoder and synthetic SAE magnitudes.
    fixture = turn_main_fixture(tmp_path, monkeypatch, production=True, clustered=True)
    contract = fixture.contract
    assert fixture.meta["user_config"]["sae"] == {
        "release": "gemmascope", "model": "gemma-2-2b", "activation_type": "residual",
        "layer": 8, "width": "16k",
    }
    assert (contract.lineage.model, contract.lineage.release, contract.lineage.activation_type,
            contract.lineage.layer, contract.lineage.width, contract.lineage.l0) == (
        "gemma-3-4b-pt", "gemmascope_2", "residual", 22, "65k", "medium")
    assert (contract.lineage.n_sae_features, contract.lineage.num_buckets,
            contract.lineage.source_segment_size, contract.lineage.missing_percentile_policy,
            contract.model_max_seq_len) == (65536, 2, 32, "lowest", 2048)
    assert contract.lineage.selection_seed == 42
    assert contract.consolidation is not None
    expected_vocab = FeatureVocab(65536, tuple(i for i in range(847) if i not in {1, 4, 5}))
    assert contract.feature_vocab == expected_vocab
    assert fixture.meta["feature_vocab"] == expected_vocab.metadata_dict()
    assert len(expected_vocab.original_ids) == 844
    # Construct the SFT context directly: do not share either turn context factory.
    lineage = contract.lineage
    pipeline = build_feature_pipeline_from_cfg_block(
        {"feature_selection": dict(lineage.feature_selection), "negative_selection": lineage.negative_selection},
        per_feature_stats=contract.statistics.per_feature_stats, n_sae_features=65536,
        num_buckets=2, seed=42, feature_mask=contract.feature_mask)
    assert isinstance(pipeline.selection_config, SamplingBundle)
    table = pipeline.selection_config.percentile_table
    assert "0" not in contract.statistics.per_feature_stats
    assert not percentile_table_coverage(table)[0]
    assert pipeline.negative_bundle is not None
    oracle = SegmentSelectionContext(
        threshold_cfg=None, precomputed_thresholds=None, sampling_bundle=pipeline.selection_config,
        negative_bundle=replace(pipeline.negative_bundle,
            feature_mask=FeatureMask(FeatureMaskMode.whitelist, expected_vocab.original_ids)),
        boundaries=pipeline.boundaries, fp_percentile_table=table, cluster_threshold_percentile_table=table,
        feature_mask=contract.feature_mask, consolidation=contract.consolidation,
        missing_percentile=MissingPercentileResolver(MissingPercentilePolicy.LOWEST, percentile_table_coverage(table)))
    assert oracle.missing_percentile.ranks(t.tensor([0]), t.tensor([0.11]), table).tolist() == [0.5]
    provider = SafetensorsContextFeatureProvider(fixture.cache,
        identity=cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=SegmentMode.turn_delimited_fixed_tokens, expected_turn_contract=contract)
    assert provider.manifest["metadata"]["cluster_selection_semantics"] == "pre_mask_representative_max_percentile_v2"
    seen_negatives = set()
    for index, (path, record) in enumerate(zip(fixture.paths, fixture.records, strict=True)):
        raw = turn_source_segments(load_features(path), 0)
        assert all({0, 1, 4} <= set(segment.feature_ids.tolist()) for segment in raw)
        with pytest.raises(ValueError, match="Percentile statistics are missing.*\\[0\\]"):
            select_conversation_segments(raw, ctx=replace(oracle, missing_percentile=MissingPercentileResolver.error()),
                example_idx=index)
        expected = select_conversation_segments(raw, ctx=oracle, example_idx=index)
        selected = provider.get(record.token_ids)
        assert selected is not None
        for got, want in zip(selected, expected, strict=True):
            assert got.feature_ids == tuple(want.feature_ids.tolist())
            assert got.buckets == tuple(want.buckets.tolist())
            selected_map = dict(zip(got.feature_ids, got.buckets, strict=True))
            assert {fid: selected_map[fid] for fid in (0, 2, 3)} == {0: 0, 2: 0, 3: 0}
            assert not ({1, 4, 5} & selected_map.keys())
            negatives = {fid for fid, bucket in selected_map.items() if bucket == 1}
            assert negatives and negatives <= set(expected_vocab.original_ids) - {0, 2, 3}
            seen_negatives.update(negatives)
    assert seen_negatives
    expected_hash = hashlib.sha256(json.dumps(contract.semantic, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert turn_context_contract_hash(contract) == expected_hash
    identity = cache_identity_from_config({}, turn_contract=contract)
    assert identity.feature_selection_hash == expected_hash
    assert (identity.model_id, identity.sae_id, identity.layer, identity.pooling) == (
        "gemma-3-4b-pt", "layer_22_width_65k_l0_medium", 22, "max_pool")
    assert fixture.model.encoder.config.n_sae_features == 65536
    assert fixture.model.encoder.config.num_buckets == 2
    assert fixture.model.feature_vocab == contract.feature_vocab
    assert fixture.model.feature_vocab_size == 844
    admitted = []
    original = chat_eval.build_target_real_artifacts_for_example

    def capture(**kwargs):
        artifacts = original(**kwargs)
        assert isinstance(artifacts.wrapper, TargetAwareEncoderDecoderEvalWrapper)
        assert artifacts.wrapper.turn_contract is not None
        assert artifacts.wrapper.turn_contract.semantic == contract.semantic
        assert kwargs["provider"].identity == identity
        assert isinstance(artifacts.wrapper.context_provider, SingleEntryContextProvider)
        assert artifacts.wrapper.context_provider.turn_contract is not None
        assert artifacts.wrapper.context_provider.turn_contract.semantic == contract.semantic
        admitted.append(artifacts)
        return artifacts

    monkeypatch.setattr(chat_eval, "build_target_real_artifacts_for_example", capture)
    fixture.cfg.task_name = "MMLU"
    chat_eval.main.__wrapped__(fixture.cfg)
    assert len(admitted) == 1
    assert admitted[0].wrapper._prepared_turn_contexts
    payload = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    assert set(payload["seeds"]["adhoc"]["results"]) == {"MMLU"}


@pytest.mark.parametrize("lockstep,mode", [(False, "current"), (False, "next"), (True, "current")])
@pytest.mark.parametrize("straddle", [False, True], ids=["exact-boundary", "gemma-straddle"])
def test_turn_chat_eval_main_real_tasks_and_cadence_parity(tmp_path, monkeypatch, lockstep, mode, straddle):
    # kills: bypassing real task evaluation, default fixed cache admission, FP substitution, cadence dependencies.
    fixture = turn_main_fixture(tmp_path, monkeypatch, lockstep=lockstep, mode=mode, straddle=straddle)
    original_forward = fixture.model.forward
    forwards = []

    def capture_forward(*args, **kwargs):
        logits = original_forward(*args, **kwargs)
        assert t.isfinite(logits).all()
        forwards.append((args[0].tolist(), kwargs["seg_map"].tolist(),
            hashlib.sha256(logits.detach().contiguous().numpy().tobytes()).hexdigest()))
        return logits

    monkeypatch.setattr(fixture.model, "forward", capture_forward)
    original_generate = Engine.generate_batch
    transcripts = []

    def capture_generate(self, *args, **kwargs):
        result = original_generate(self, *args, **kwargs)
        transcripts.append(result)
        return result

    monkeypatch.setattr(Engine, "generate_batch", capture_generate)
    original_build = chat_eval.build_target_real_artifacts_for_example
    wrappers = []

    def capture_wrapper(**kwargs):
        artifacts = original_build(**kwargs)
        wrappers.append(artifacts)
        return artifacts

    monkeypatch.setattr(chat_eval, "build_target_real_artifacts_for_example", capture_wrapper)
    chat_eval.main.__wrapped__(fixture.cfg)
    result_path = fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml"
    output = result_path.read_bytes()
    payload = yaml.safe_load(output)
    assert payload["kind"] == ("target_real_lockstep" if lockstep else "target_real_pooled")
    assert payload["target_conditioning_mode"] == mode
    results = payload["seeds"]["adhoc"]["results"]
    assert set(results) == set(fixture.tasks)
    assert len(wrappers) == 6
    assert all(not artifacts.wrapper.is_self_conditioned for artifacts in wrappers)
    assert len(transcripts) == 3  # GSM8K, HumanEval and SpellingBee use the real scalar Engine.
    assert payload["seeds"]["adhoc"]["chatcore_metric"] == sum(
        (score - chat_eval.BASELINE_ACCURACIES[name]) / (1 - chat_eval.BASELINE_ACCURACIES[name])
        for name, score in results.items()) / 6
    assert all(artifacts.wrapper._prepared_turn_contexts for artifacts in wrappers)
    original_forwards, original_transcripts = list(forwards), list(transcripts)
    fixture.meta["turn_self_cond_segment_size_rustbpe"] = 17  # Not s4, nor constrained by the 3-token budget.
    fixture.checkpoint_path.write_text(json.dumps(fixture.meta))
    forwards.clear()
    transcripts.clear()
    chat_eval.main.__wrapped__(fixture.cfg)
    updated = yaml.safe_load(result_path.read_bytes())
    previous_checkpoint = payload["seeds"]["adhoc"]["run_identity"]["checkpoint"]
    updated_checkpoint = updated["seeds"]["adhoc"]["run_identity"]["checkpoint"]
    assert updated_checkpoint["metadata_sha256"] != previous_checkpoint["metadata_sha256"]
    assert updated_checkpoint["metadata_sha256"] == hashlib.sha256(fixture.checkpoint_path.read_bytes()).hexdigest()
    # Cadence metadata changes checkpoint provenance, but not static conditioning,
    # generation, task results or any other serialized field.
    updated_checkpoint["metadata_sha256"] = previous_checkpoint["metadata_sha256"]
    assert yaml.safe_dump(updated, default_flow_style=False, sort_keys=False).encode() == output
    assert forwards == original_forwards
    assert transcripts == original_transcripts


@pytest.mark.parametrize("mutation,match", [("fixed-cache", "schema_version"),
    ("wrong-size", "turn context contract disagrees with checkpoint-derived expectations"), ("wrong-lineage", "contract"),
    ("wrong-model-length", "sequence_len"), ("wrong-config-length", "model_max_seq_len"),
    ("cadence", "positive integer"), ("samples", "num_samples"), ("temperature", "temperature"),
    ("distributed", "single-process"), ("microbatch", "scalar Engine"),
    ("static", "empty feature_configs"), ("mode", "segment_mode"),
    ("extraction-tokenizer", "extraction_tokenizer_id conflicts"),
    ("next-offset", "next_segment_conditioning_offset disagrees with checkpoint metadata"),
    ("decoder-only", "turn target-real requires an encoder-decoder")])
def test_turn_main_rejects_invalid_settings_and_caches_before_forward(tmp_path, monkeypatch, mutation, match):
    # kills: trusting cache authority, accepting cross-mode/size/lineage caches, unsupported turn work before admission.
    fixture = turn_main_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("invalid turn eval reached model forward"))
    if mutation in {"fixed-cache", "wrong-size", "wrong-lineage"}:
        manifest_path = fixture.cache / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        if mutation == "fixed-cache":
            manifest["schema_version"] = 4
            manifest["source_segment_mode"] = SegmentMode.fixed_tokens.value
        elif mutation == "wrong-size":
            manifest["turn_contract"]["source_segment_size"] = 32
            manifest["source_segment_size"] = 32
        else:
            manifest["turn_contract"]["statistics_raw_sha256"] = "f" * 64
        manifest_path.write_text(json.dumps(manifest))
    elif mutation == "wrong-model-length":
        fixture.model.dec_config.sequence_len = 1024
    elif mutation == "wrong-config-length":
        fixture.cfg.model_max_seq_len = 1024
    elif mutation == "cadence":
        fixture.meta["turn_self_cond_segment_size_rustbpe"] = True
        fixture.checkpoint_path.write_text(json.dumps(fixture.meta))
    elif mutation == "samples":
        fixture.cfg.num_samples = 2
    elif mutation == "temperature":
        fixture.cfg.temperature = 0.5
    elif mutation == "distributed":
        monkeypatch.setattr(chat_eval, "compute_init", lambda _: (True, 0, 0, 2, t.device("cpu")))
    elif mutation == "microbatch":
        fixture.cfg.greedy_microbatch_size = 2
    elif mutation == "static":
        fixture.cfg.feature_configs = ["unused.yaml"]
    elif mutation == "extraction-tokenizer":
        fixture.cfg.extraction_tokenizer_id = "google/gemma-2-2b"
    elif mutation == "next-offset":
        fixture.cfg.next_segment_conditioning_offset = not resolve_next_segment_conditioning_offset(
            {"next_segment_conditioning_offset": "auto"}, fixture.meta)
    elif mutation == "decoder-only":
        decoder = build_tiny_decoder()
        monkeypatch.setattr(decoder, "forward", lambda *args, **kwargs: pytest.fail("invalid decoder reached forward"))
        monkeypatch.setattr(chat_eval, "load_model", lambda *args, **kwargs: (decoder, fixture.tokenizer, fixture.meta))
    else:
        fixture.cfg.segment_mode = SegmentMode.fixed_tokens.value
    with pytest.raises((ValueError, RuntimeError), match=match):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert not (fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").exists()


@pytest.mark.parametrize("substitute,message", [
    ("fixed", "Context cache schema has changed: manifest schema_version=4, expected 5"),
    ("size", "turn context contract disagrees with checkpoint-derived expectations"),
    ("lineage", "turn context contract disagrees with checkpoint-derived expectations"),  # same contract guard
])
def test_main_rejects_genuine_incompatible_cache_even_after_leaf_rename(tmp_path, monkeypatch, substitute, message):
    # kills: admission based only on directory names or geometry labels, without complete schema/lineage checks.
    fixture = turn_main_fixture(tmp_path, monkeypatch)
    replacement = tmp_path / "replacement" / "c2048" / fixture.cache.name
    if substitute == "fixed":
        provider = target_real_runner.build_context_feature_provider(fixture.cfg,
            cache_dir=fixture.cache_cfg.output_dir, turn_contract=fixture.contract)
        entries = []
        for record in fixture.records:
            segments = provider.get(record.token_ids)
            assert segments is not None
            entries.append(ContextCacheEntry(record.token_ids, segments))
        save_context_cache(replacement, entries, identity=provider.identity,
            n_sae_features=PositiveInt.parse(8), num_buckets=PositiveInt.parse(4), metadata={"model_max_seq_len": 2048})
    else:
        alternate, _, _, _, _, _ = build_fixture(tmp_path / "alternate", monkeypatch,
            records=fixture.records, tokenizer=fixture.tokenizer,
            source=StraddlingSourceTokenizer(),
            size=8 if substitute == "size" else 4, seed=23 if substitute == "lineage" else 19)
        manifest = builder.build_cache_from_config(alternate)
        builder.validate_cache_from_config(alternate)
        # Hiding its different semantic hash in a matching leaf must not grant admission.
        shutil.copytree(manifest.parent, replacement)
    fixture.cfg.target_real_configs[0].cache_dir = str(replacement)
    monkeypatch.setattr(fixture.model, "forward", lambda *args, **kwargs: pytest.fail("incompatible cache reached forward"))
    with pytest.raises(RuntimeError, match=message):
        chat_eval.main.__wrapped__(fixture.cfg)


def test_real_categorical_tasks_select_marker_logits_and_available_letters(tmp_path, monkeypatch):
    # kills: whole-answer teacher forcing, wrong prompt position, or unrestricted vocabulary argmax.
    fixture = turn_main_fixture(tmp_path, monkeypatch)
    names = ["MMLU", "ARC-Easy", "ARC-Challenge"]
    fixture.cfg.task_name = "|".join(names)
    by_prompt = {}
    for name in names:
        conversation = fixture.tasks[name][0]
        answer = conversation["messages"][-1]["content"]
        wrong = next(letter for letter in conversation["letters"] if letter != answer)
        by_prompt[tuple(chat_eval.render_chat_completion(fixture.tokenizer, conversation).token_ids)] = (answer, wrong)
    original = fixture.model.forward
    calls = []

    def force_letter_logits(ids, *args, **kwargs):
        logits = original(ids, *args, **kwargs)
        assert t.isfinite(logits).all()
        answer, wrong = by_prompt[tuple(ids[0].tolist())]
        scripted = t.zeros_like(logits)
        scripted[:, :, fixture.tokenizer.encode("0")[0]] = 100
        scripted[:, :-1, fixture.tokenizer.encode(wrong)[0]] = 10
        scripted[:, -1, fixture.tokenizer.encode(answer)[0]] = 10
        calls.append(ids[0, -1].item())
        return scripted

    monkeypatch.setattr(fixture.model, "forward", force_letter_logits)
    chat_eval.main.__wrapped__(fixture.cfg)
    result = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    assert result["seeds"]["adhoc"]["results"] == dict.fromkeys(names, 1.0)
    assert calls == [fixture.tokenizer.encode_special("<assistant_start>")] * 3


def test_production_checkpoint_leaf_resolution(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, production=True, clustered=True)
    leaf = fixture.checkpoint_path.parent
    assert sorted(path.name for path in leaf.iterdir()) == ["meta_001116.json", "model_001116.pt"]
    assert not list(leaf.parent.glob("model_*.pt"))
    assert not list(leaf.parent.glob("meta_*.json"))
    assert Path(checkpoint_manager.get_checkpoint_dir(
        ModelSource.ENCDEC, model_tag=PRODUCTION_MODEL_TAG, step=1116)) == leaf
    bad_source = f"{ModelSource.ENCDEC}/standard_cond"
    assert Path(checkpoint_manager.get_checkpoint_dir(
        bad_source, model_tag=PRODUCTION_TAG, step=1116)) == leaf / SFT_LEAF
    with pytest.raises(FileNotFoundError, match="standard_sft_conditioned_checkpoints/standard_sft_conditioned_checkpoints"):
        checkpoint_manager.load_model(bad_source, "cpu", ModelPhase.EVAL,
            model_tag=PRODUCTION_TAG, step=1116, eval_conditioning=EvalConditioning.deferred("test"))
    fixture.cfg.task_name = "MMLU"
    chat_eval.main.__wrapped__(fixture.cfg)
    assert fixture.loaded == [(leaf, 1116)]
    newer_meta = json.loads(fixture.checkpoint_path.read_bytes())
    newer_meta["step"] = 1117
    (leaf / "meta_001117.json").write_text(json.dumps(newer_meta))
    weights = t.load(leaf / "model_001116.pt", weights_only=True)
    weights["step"] = 1117
    t.save(weights, leaf / "model_001117.pt")
    chat_eval.main.__wrapped__(fixture.cfg)
    assert fixture.loaded == [(leaf, 1116), (leaf, 1116)]
    fixture.cfg.step = None
    chat_eval.main.__wrapped__(fixture.cfg)
    assert fixture.loaded == [(leaf, 1116), (leaf, 1116), (leaf, 1117)]
