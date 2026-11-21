"""Marker-aware target construction and explicit turn-consumer admission."""

import hashlib
import json
from dataclasses import replace

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.eval import build_eval_context_cache as builder
from steering.conditional.encoder_decoder.full_sequence_routing import (
    build_full_sequence_segment_routing,
    source_aligned_char_offsets,
)
from steering.eval import eval_cache_paths, target_real_runner
from steering.eval.context_cache_io import ContextCacheEntry
from steering.eval.pooling_op import PoolingOp
from steering.eval.target_conditioning import (
    LockstepTargetRealConfig,
    MultiTargetSegments,
    PooledTargetRealConfig,
    SingleTargetSegment,
)
from steering.eval.target_real_dispatch import build_turn_target_for_real_config
from steering.eval.target_real_runner import (
    build_context_feature_provider,
    parse_target_real_configs_from_cfg,
)
from steering.segmentation import token_to_char_offsets
from tests.steering.test_turn_context_cache_build import (
    StraddlingSourceTokenizer,
    UnroutedStraddlingSourceTokenizer,
    build_fixture,
    convert,
)


def dispatch_fixture(tmp_path, monkeypatch, *, straddle=False, unrouted=False):
    cfg, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch,
        source=UnroutedStraddlingSourceTokenizer() if unrouted else StraddlingSourceTokenizer() if straddle else None)
    entries = convert(records, paths, contract, tokenizer)
    full, prefix = records
    kwargs = {"entry": entries[0], "full_token_ids": full.token_ids,
        "context_token_ids": prefix.token_ids, "tokenizer": tokenizer, "turn_contract": contract,
        "target_char_start": full.metadata["target_char_start"], "target_char_end": full.metadata["target_char_end"],
        "target_token_loss_mask": [0] * len(full.token_ids)}
    return cfg, contract, entries, kwargs


def test_pooled_target_excludes_unrouted_source_window_inside_the_span(tmp_path, monkeypatch):
    # kills: pooling the contiguous ordinal range, which folds a source window no
    # target input routes to (a tool-boundary straddle) into the reference bag.
    _, contract, entries, kwargs = dispatch_fixture(tmp_path, monkeypatch)
    tokenizer, full_ids = kwargs["tokenizer"], kwargs["full_token_ids"]
    p0 = len(kwargs["context_token_ids"]) - 1
    bos = contract.semantic["source_bos_char_len"]
    offsets = source_aligned_char_offsets(tokenizer, full_ids, source_bos_char_len=bos)
    segments = list(entries[0].segments)

    def routing_for(windows):
        return build_full_sequence_segment_routing(ids=full_ids, tokenizer=tokenizer,
            seg_features=([t.tensor(seg.feature_ids) for seg in windows], [t.tensor(seg.buckets) for seg in windows],
                          [seg.char_start for seg in windows], [seg.char_end for seg in windows]),
            source_bos_char_len=bos).seg_map - 1

    baseline = routing_for(segments)
    first, last = int(baseline[p0]), int(baseline[-1])
    # Carve a one-character window out of an in-span window, inside a multi-character
    # decoder token (the tool markers), so that no decoder token routes to it.
    split = None
    for index in range(first + 1, last + 1):
        window = segments[index]
        for cut in range(window.char_start + 1, window.char_end - 2):
            owners = [i for i in range(p0, len(full_ids)) if offsets[i] < cut + 1 and offsets[i + 1] > cut]
            if len(owners) != 1 or offsets[owners[0]] >= cut or offsets[owners[0] + 1] <= cut + 1:
                continue
            candidate = [*segments[:index],
                         replace(window, char_end=cut), replace(window, char_start=cut, char_end=cut + 1),
                         replace(window, char_start=cut + 1), *segments[index + 1:]]
            if index + 1 not in routing_for(candidate).tolist():
                split = (index + 1, candidate)
                break
        if split is not None:
            break
    assert split is not None, "fixture has no in-span multi-character token to carve"
    unrouted, windows = split
    changed = [replace(seg, feature_ids=(1,), buckets=(0,)) for seg in windows]
    changed[unrouted] = replace(changed[unrouted], feature_ids=(9,), buckets=(3,))
    routing = routing_for(changed)
    assert unrouted not in routing.tolist()
    assert int(routing[p0]) < unrouted <= int(routing[-1])  # strictly inside the target span
    kwargs["entry"] = ContextCacheEntry(tuple(full_ids), tuple(changed))
    pooled = build_turn_target_for_real_config(cfg=PooledTargetRealConfig("pooled", tmp_path, PoolingOp.BUCKET_MAX), **kwargs)
    assert isinstance(pooled, SingleTargetSegment)
    assert pooled.features == {1: 0}
    lockstep = build_turn_target_for_real_config(cfg=LockstepTargetRealConfig("lockstep", tmp_path), **kwargs)
    assert isinstance(lockstep, MultiTargetSegments)
    assert lockstep.segments[unrouted - int(routing[p0])] == {9: 3}
    assert unrouted - int(routing[p0]) not in lockstep.token_assignments.tolist()


@pytest.mark.parametrize("pooling,expected", [
    (PoolingOp.BUCKET_MAX, {7: 2, 1: 0}), (PoolingOp.BUCKET_MEAN, {7: 1, 1: 0}),
])
@pytest.mark.parametrize("straddle,unrouted", [(False, False), (True, False), (True, True)],
    ids=["exact-boundary", "gemma-straddle", "unrouted-straddle"])
def test_turn_marker_pool_and_reference_input_schedule(tmp_path, monkeypatch, pooling, expected, straddle, unrouted):
    # kills: BOS double shift, target-only routing, marker exclusion, tool-mask forward fill.
    _, contract, entries, kwargs = dispatch_fixture(tmp_path, monkeypatch, straddle=straddle, unrouted=unrouted)
    tokenizer = kwargs["tokenizer"]
    full_ids = kwargs["full_token_ids"]
    p0 = len(kwargs["context_token_ids"]) - 1
    source_offsets = source_aligned_char_offsets(tokenizer, full_ids, source_bos_char_len=3)
    segments = entries[0].segments
    raw_oracle = build_full_sequence_segment_routing(ids=full_ids, tokenizer=tokenizer,
        seg_features=([t.tensor(seg.feature_ids) for seg in segments], [t.tensor(seg.buckets) for seg in segments],
                      [seg.char_start for seg in segments], [seg.char_end for seg in segments]),
        source_bos_char_len=contract.semantic["source_bos_char_len"])
    first = int(raw_oracle.seg_map[p0]) - 1
    last = int(raw_oracle.seg_map[-1]) - 1
    if straddle:
        assert segments[first].char_start == source_offsets[p0] + 1
        assert next(i for i, seg in enumerate(segments) if seg.char_end > source_offsets[p0]) == first - 1
    if unrouted:
        keep = int(raw_oracle.seg_map[:p0].max())
        assert keep == 5
        assert first == keep + 1 == 6
        assert set(range(keep, first)) == {5}
        assert keep not in (raw_oracle.seg_map - 1).tolist()
        assert (segments[keep].char_start, segments[keep].char_end) == (source_offsets[p0] - 5, source_offsets[p0] + 1)
        # Stage 6 would reject this prefix tail even though no input uses it.
        assert segments[keep].char_start < source_offsets[p0]
    # Feature 7 occurs in the marker slot at bucket 2 and in the body at bucket 0.
    changed = [replace(seg, feature_ids=(1,), buckets=(0,)) for seg in segments]
    changed[first - 1] = replace(changed[first - 1], feature_ids=(6,), buckets=(3,))
    changed[first] = replace(changed[first], feature_ids=(7,), buckets=(2,))
    changed[first + 1] = replace(changed[first + 1], feature_ids=(7,), buckets=(0,))
    kwargs["entry"] = ContextCacheEntry(tuple(full_ids), tuple(changed))
    pooled = build_turn_target_for_real_config(cfg=PooledTargetRealConfig("pooled", tmp_path, pooling), **kwargs)
    assert isinstance(pooled, SingleTargetSegment)
    assert pooled.features == expected
    lockstep = build_turn_target_for_real_config(cfg=LockstepTargetRealConfig("lockstep", tmp_path), **kwargs)
    oracle = build_full_sequence_segment_routing(ids=full_ids, tokenizer=tokenizer,
        seg_features=([t.tensor(seg.feature_ids) for seg in changed], [t.tensor(seg.buckets) for seg in changed],
                      [seg.char_start for seg in changed], [seg.char_end for seg in changed]),
        source_bos_char_len=contract.semantic["source_bos_char_len"])
    assert isinstance(lockstep, MultiTargetSegments)
    assert t.equal(lockstep.token_assignments, oracle.seg_map[p0:] - 1 - first)
    assert lockstep.segments == [dict(zip(seg.feature_ids, seg.buckets, strict=True)) for seg in changed[first:last + 1]]
    assert lockstep.token_assignments[0] == 0
    assert lockstep.segments[0] == {7: 2}
    assert 6 not in pooled.features
    assert all(6 not in slot for slot in lockstep.segments)
    output = full_ids.index(tokenizer.encode_special("<output_start>"))
    assert lockstep.token_assignments[output - p0] != lockstep.token_assignments[output - p0 - 1]
    assert len(lockstep.token_assignments) == len(full_ids) - p0
    # The renderer's supervision mask cannot suppress forced decoder input positions.
    kwargs["target_token_loss_mask"] = [1] * len(full_ids)
    again = build_turn_target_for_real_config(cfg=LockstepTargetRealConfig("lockstep", tmp_path), **kwargs)
    assert isinstance(again, MultiTargetSegments)
    assert t.equal(again.token_assignments, lockstep.token_assignments)
    assert source_offsets[p0] != token_to_char_offsets(tokenizer, list(full_ids))[p0]


@pytest.mark.parametrize("end_delta", [-1, 0], ids=["before-marker", "at-marker"])
@pytest.mark.parametrize("lockstep", [False, True], ids=["pooled", "lockstep"])
def test_turn_target_rejects_unrouted_prompt_window(tmp_path, monkeypatch, end_delta, lockstep):
    _, _, entries, kwargs = dispatch_fixture(tmp_path, monkeypatch, unrouted=True)
    p0 = len(kwargs["context_token_ids"]) - 1
    marker_start = source_aligned_char_offsets(kwargs["tokenizer"], kwargs["full_token_ids"], source_bos_char_len=3)[p0]
    segments = entries[0].segments
    index = next(i for i, seg in enumerate(segments) if seg.char_start == marker_start - 5)
    tail = segments[index]
    # Split the unused straddle into an unused prompt window and a straddle.
    changed = (*segments[:index], replace(tail, char_end=marker_start + end_delta),
               replace(tail, char_start=marker_start + end_delta), *segments[index + 1:])
    kwargs["entry"] = replace(entries[0], segments=changed)
    cfg = LockstepTargetRealConfig("lockstep", tmp_path) if lockstep else PooledTargetRealConfig("pooled", tmp_path)
    # All four cases legitimately share the one unused-prompt-window guard; the
    # point is that neither target mode and neither split offset can slip past it.
    with pytest.raises(ValueError, match="unused turn source slot before target ends at or before the assistant marker"):
        build_turn_target_for_real_config(cfg=cfg, **kwargs)


def test_turn_target_allows_full_reference_beyond_checkpoint_limit(tmp_path, monkeypatch):
    _, contract, _, kwargs = dispatch_fixture(tmp_path, monkeypatch)
    prefix_length = len(kwargs["context_token_ids"])
    assert len(kwargs["full_token_ids"]) > prefix_length
    kwargs["turn_contract"] = replace(contract, model_max_seq_len=prefix_length)
    target = build_turn_target_for_real_config(
        cfg=LockstepTargetRealConfig("lockstep", tmp_path), **kwargs
    )
    assert isinstance(target, MultiTargetSegments)


@pytest.mark.parametrize("mutation,message", [
    # `wrong-prefix` and `wrong-entry` share the exact-prefix guard: a mutated prefix
    # and a prefix taken from the other cached row are both non-prefixes of the row.
    ("wrong-prefix", "turn target-real full row and completion prefix disagree"),
    ("not-marker", "turn target-real requires a marker-only terminal <assistant_start> turn"),
    ("wrong-entry", "turn target-real full row and completion prefix disagree"),
    ("mask-length", "full_token_ids and target_token_loss_mask length mismatch"),
    # `body-start` and `body-end` share the answer-body provenance guard (both ends).
    ("body-start", "turn target-real answer-body character provenance disagrees with the prefix/full row"),
    ("body-end", "turn target-real answer-body character provenance disagrees with the prefix/full row"),
    ("oversize", "turn target-real completion prefix exceeds checkpoint model_max_seq_len"),
    ("gap", r"Gap between segment 0 \(end=\d+\) and segment 1 \(start=\d+\)"),
    ("unrouted", r"Forward tokenizer-mismatch delta exceeds end-marker window: "
                 r"decoded_len=\d+, coverage_end=\d+, delta=\d+, window=\d+"),
])
def test_turn_dispatch_fails_loud(tmp_path, monkeypatch, mutation, message):
    # kills: unchecked exact keys, truncated/missing geometry, invalid provenance, zero-to-minus-one routing.
    _, contract, entries, kwargs = dispatch_fixture(tmp_path, monkeypatch)
    if mutation == "wrong-prefix":
        prefix = list(kwargs["context_token_ids"])
        prefix[2] += 1
        kwargs["context_token_ids"] = prefix
    elif mutation == "not-marker":
        kwargs["context_token_ids"] = kwargs["context_token_ids"][:-1]
    elif mutation == "wrong-entry":
        kwargs["entry"] = entries[1]
    elif mutation == "mask-length":
        kwargs["target_token_loss_mask"] = []
    elif mutation == "body-start":
        kwargs["target_char_start"] += 1
    elif mutation == "body-end":
        kwargs["target_char_end"] += 1
    elif mutation == "oversize":
        kwargs["turn_contract"] = replace(contract, model_max_seq_len=1)
    elif mutation == "gap":
        segments = list(entries[0].segments)
        segments[1] = replace(segments[1], char_start=segments[1].char_start + 1)
        kwargs["entry"] = replace(entries[0], segments=tuple(segments))
    else:
        # Leave enough missing text to exceed the shared end-marker tolerance.
        kwargs["entry"] = replace(entries[0], segments=entries[0].segments[:1])
    with pytest.raises(ValueError, match=message):
        build_turn_target_for_real_config(cfg=LockstepTargetRealConfig("lockstep", tmp_path), **kwargs)


def test_turn_provider_and_auto_resolution_are_checkpoint_bound(tmp_path, monkeypatch):
    # kills: schema-5 opt-in lost in runner, default c1024/SAE identity, explicit hash escape hatch.
    cfg, contract, _, _ = dispatch_fixture(tmp_path, monkeypatch)
    monkeypatch.setattr(eval_cache_paths, "get_base_dir", lambda: str(tmp_path))
    builder.build_cache_from_config(cfg)
    consumer = OmegaConf.create({"model_max_seq_len": 2048, "conditioning": {"identity": {}},
        "target_real_configs": [{"name": "pooled", "mode": "pooled", "cache_dir": None}]})
    provider = build_context_feature_provider(consumer, cache_dir=cfg.output_dir, turn_contract=contract)
    assert provider.turn_contract is contract
    trc = parse_target_real_configs_from_cfg(consumer, suite="chat_target", turn_contract=contract)[0]
    expected_hash = hashlib.sha256(json.dumps(contract.semantic, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    assert provider.identity.feature_selection_hash == expected_hash
    assert trc.cache_dir == (tmp_path /
        "eval_caches/chat_target/m_gemma-3-4b-pt/l22/a_res/p_max_pool/f65536/c2048" / expected_hash[:16])
    consumer.conditioning.identity.feature_selection_hash = "f" * 64
    with pytest.raises(ValueError, match="explicit turn feature_selection_hash does not match checkpoint contract"):
        build_context_feature_provider(consumer, cache_dir=cfg.output_dir, turn_contract=contract)


@pytest.mark.parametrize("field,value", [("n_sae_features", 16), ("num_buckets", 2),
    ("source_segment_size", 32), ("seed", 42), ("feature_clusters", {}),
    ("feature_selection", {"threshold": 0.1}), ("source_context_window", 2048),
    ("source_bos_char_len", 0), ("negative_selection", {}), ("stats_source_sha256", "f" * 64),
    ("feature_mask", {"mode": "whitelist", "ids": [1]}), ("missing_percentile_policy", "lowest"),
    ("feature_delta", {})])
def test_turn_consumer_overrides_rejected_before_cache_io(tmp_path, monkeypatch, field, value):
    # kills: legacy defaults or override settings silently overriding checkpoint authority.
    _, contract, _, _ = dispatch_fixture(tmp_path, monkeypatch)
    cfg = {"model_max_seq_len": 2048, "conditioning": {"identity": {}, field: value}}
    # Clustered/delta selection has its own guard; every other field must name itself.
    expected = ("turn ChatCORE does not support clustered/delta selection overrides"
                if field in {"feature_clusters", "feature_delta"}
                else f"conditioning.{field} conflicts with checkpoint turn contract")
    with pytest.raises(ValueError, match=expected):
        build_context_feature_provider(cfg, cache_dir=tmp_path / "missing", turn_contract=contract)


def test_turn_other_suites_and_wrong_context_length_rejected(tmp_path, monkeypatch):
    # kills: CORE/progression accidentally gaining turn support and c1024 auto-resolution.
    _, contract, _, _ = dispatch_fixture(tmp_path, monkeypatch)
    cfg = {"model_max_seq_len": 1024, "conditioning": {"identity": {}},
           "target_real_configs": [{"name": "pooled", "mode": "pooled"}]}
    with pytest.raises(ValueError, match="turn target-real supports only the chat_target suite"):
        parse_target_real_configs_from_cfg(cfg, suite="core_target", turn_contract=contract)
    with pytest.raises(ValueError, match="model_max_seq_len must equal the turn checkpoint sequence_len"):
        parse_target_real_configs_from_cfg(cfg, suite="chat_target", turn_contract=contract)


@pytest.mark.parametrize("rank,world_size", [(0, 2), (1, 1)])
def test_single_process_guard_reads_the_live_launcher_environment(monkeypatch, rank, world_size):
    # kills: trusting the caller's reported rank/world over the live launcher environment
    # (the guard now runs once, at turn admission in chat_eval._resolve_chat_turn_contract).
    for key, value in {"RANK": rank, "LOCAL_RANK": rank, "WORLD_SIZE": world_size}.items():
        monkeypatch.setenv(key, str(value))
    # Both env combinations legitimately hit the one admission guard; the case split
    # is over which launcher variable disagrees with the caller's reported values.
    with pytest.raises(ValueError, match="turn target-real ChatCORE requires single-process execution"):
        target_real_runner.require_single_process_turn_target_real(rank=0, world_size=1)


def test_one_rank_ddp_is_single_process(tmp_path, monkeypatch):
    for key, value in {"RANK": "0", "LOCAL_RANK": "0", "WORLD_SIZE": "1"}.items():
        monkeypatch.setenv(key, value)
    assert target_real_runner.get_dist_info() == (True, 0, 0, 1)
    target_real_runner.require_single_process_turn_target_real()
