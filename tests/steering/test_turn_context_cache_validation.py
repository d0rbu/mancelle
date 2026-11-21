"""Deep source geometry, replay, and shared validation-entrypoint guards."""

import json
from dataclasses import replace
from itertools import pairwise
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import torch as t
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from scripts.eval import build_eval_context_cache as builder
from scripts.eval import context_cache_common as common
from scripts.eval import validate_eval_context_cache as validator
from steering.conditional.encoder_decoder.full_sequence_routing import source_aligned_char_offsets
from steering.eval import context_cache_validation
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.context_cache_validation import (
    validate_context_cache_records,
    validate_turn_context_cache_records,
    validate_turn_source_geometry,
)
from steering.eval.context_enumeration import enumerate_chat_target_contexts
from steering.eval.target_conditioning import MultiTargetSegments, SingleTargetSegment
from steering.extract_features import (
    _read_text_and_turn_chunks,
    _validate_extraction_parquet_schema,
    _validate_segment_mode_dataset_phase,
)
from steering.feature_io import load_features, save_features
from steering.feature_paths import check_sft_data_ready
from steering.segmentation import SegmentMode, token_to_char_offsets
from tests.steering.chatcore_cache_helpers import ChatTokenizer
from tests.steering.test_context_cache_in_memory_conversion import _DifferentSourceTokenizer
from tests.steering.test_turn_context_cache_build import (
    StraddlingSourceTokenizer,
    UnroutedStraddlingSourceTokenizer,
    build_fixture,
    synthetic_shards,
)
from tests.steering.test_turn_context_contract import checkpoint_fixture, resolve

TURN = SegmentMode.turn_delimited_fixed_tokens


def loaded_fixture(tmp_path, monkeypatch):
    cfg, records, paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch)
    manifest = builder.build_cache_from_config(cfg)
    provider = SafetensorsContextFeatureProvider(manifest.parent,
        identity=common.cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=TURN, expected_turn_contract=contract)
    return cfg, records, paths, contract, tokenizer, source, provider


def test_both_validation_entrypoints_share_deep_checks(tmp_path, monkeypatch):
    # kills: builder validate checking only cache non-emptiness, or standalone skipping source geometry.
    cfg, _, paths, _, tokenizer, _, _ = loaded_fixture(tmp_path, monkeypatch)
    standalone = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": False})
    assert isinstance(standalone, DictConfig)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    summary = validator.validate_cache_from_config(standalone)
    assert summary["turn_target_checks"][0]["target_start_pos"] > 1
    data = load_features(paths[0])
    data["segment_positions"][1] += 1
    save_features(data, paths[0])
    for invoke, config in ((builder.validate_cache_from_config, cfg), (validator.validate_cache_from_config, standalone)):
        # Both entrypoints must reject the edited shard through the same digest guard.
        with pytest.raises(ValueError, match="turn cache ordered feature_files provenance disagrees"):
            invoke(config)


@pytest.mark.parametrize("target_type,corrupt_assignments", [
    (SingleTargetSegment, False), (MultiTargetSegments, False), (MultiTargetSegments, True),
])
@pytest.mark.parametrize("source_class", [StraddlingSourceTokenizer, UnroutedStraddlingSourceTokenizer],
    ids=["gemma-straddle", "unrouted-straddle"])
def test_validation_checks_targets_against_independent_routing(tmp_path, monkeypatch, target_type, corrupt_assignments, source_class):
    cfg, _, _, _, _, _ = build_fixture(tmp_path, monkeypatch, source=source_class())
    builder.build_cache_from_config(cfg)
    builder.validate_cache_from_config(cfg)
    original = context_cache_validation.build_turn_target_for_real_config

    def corrupt_target(**kwargs):
        target = original(**kwargs)
        if not isinstance(target, target_type):
            return target
        if isinstance(target, SingleTargetSegment):
            assert target.features
            return replace(target, features={})
        assert isinstance(target, MultiTargetSegments)
        if corrupt_assignments:
            assert target.token_assignments.max() > 0
            return replace(target, token_assignments=t.zeros_like(target.token_assignments))
        return replace(target, segments=[{}, *target.segments[1:]])

    monkeypatch.setattr(context_cache_validation, "build_turn_target_for_real_config", corrupt_target)
    # Emptied lockstep segments and zeroed assignments share the lockstep comparison;
    # the pooled/lockstep split is what the message must distinguish.
    plan = "pooled" if target_type is SingleTargetSegment else "lockstep"
    with pytest.raises(ValueError, match=f"{plan} target disagrees with full-sequence routing"):
        builder.validate_cache_from_config(cfg)


def test_validation_rejects_unrouted_prompt_window_before_target_dispatch(tmp_path, monkeypatch):
    cfg, records, _, _, tokenizer, _ = build_fixture(tmp_path, monkeypatch, source=UnroutedStraddlingSourceTokenizer())
    builder.build_cache_from_config(cfg)
    full, prefix = records
    p0 = len(prefix.token_ids) - 1
    marker_start = source_aligned_char_offsets(tokenizer, full.token_ids, source_bos_char_len=3)[p0]
    original = context_cache_validation.build_full_sequence_segment_routing

    def corrupt_full_routing(**kwargs):
        routing = original(**kwargs)
        if tuple(kwargs["ids"]) == full.token_ids:
            assignments = routing.seg_map.clone()
            keep = int(assignments[:p0].max())
            assert kwargs["seg_features"][3][keep - 1] <= marker_start
            assignments[:p0].clamp_max_(keep - 1)
            return replace(routing, seg_map=assignments)
        return routing

    monkeypatch.setattr(context_cache_validation, "build_full_sequence_segment_routing", corrupt_full_routing)
    monkeypatch.setattr(context_cache_validation, "build_turn_target_for_real_config",
        lambda **kwargs: pytest.fail("unrouted prompt window reached target dispatch"))
    with pytest.raises(ValueError, match="unused turn source slot before target ends at or before the assistant marker"):
        builder.validate_cache_from_config(cfg)


@pytest.mark.parametrize("field", ["segment_positions", "segment_char_starts", "segment_char_ends", "doc_token_counts"])
def test_geometry_reconstruction_before_any_output(tmp_path, monkeypatch, field):
    # kills: trusting plausible source geometry, snapping boundaries, wrong document truncation counts.
    cfg, _, paths, _, _, _ = build_fixture(tmp_path, monkeypatch)
    data = load_features(paths[0])
    data[field][0] += 1
    save_features(data, paths[0])
    # The three geometry arrays share the reconstruction guard; only a wrong document
    # token count is caught by the separate per-document count comparison.
    message = ("doc 0 source token count disagrees" if field == "doc_token_counts"
               else "doc 0 reconstructed turn geometry disagrees")
    with pytest.raises(ValueError, match=message):
        builder.build_cache_from_config(cfg)
    assert not (tmp_path / "cache").exists()


def test_valid_geometry_with_large_extraction_truncation_is_rejected(tmp_path, monkeypatch):
    # kills: extrapolating missing source extraction across an entire answer/tool turn.
    _, records, _, _, tokenizer, source = build_fixture(tmp_path, monkeypatch)
    meta, _, _ = checkpoint_fixture(tmp_path / "short", window=8)
    contract = resolve(meta)
    paths = synthetic_shards(tmp_path / "short-features", records, source, window=8)
    with pytest.raises(ValueError, match=r"Forward tokenizer-mismatch delta exceeds end-marker window: "
                                         r"decoded_len=\d+, coverage_end=\d+, delta=\d+, window=\d+"):
        validate_turn_source_geometry(records=records, feature_shard_paths=paths, contract=contract,
            extraction_tokenizer=source, tokenizer=tokenizer)


def test_full_reference_may_exceed_checkpoint_limit(tmp_path, monkeypatch):
    # Full rows supply extraction/target provenance and never enter the decoder.
    _, records, paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch)
    full, prefix = records
    assert len(full.token_ids) > len(prefix.token_ids)
    contract = replace(contract, model_max_seq_len=len(prefix.token_ids))
    validate_turn_source_geometry(records=records, feature_shard_paths=paths, contract=contract,
        extraction_tokenizer=source, tokenizer=tokenizer)


def test_completion_prefix_exceeds_checkpoint_limit(tmp_path, monkeypatch):
    # The independently rendered prefix is the actual decoder input.
    _, records, paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch)
    _, prefix = records
    contract = replace(contract, model_max_seq_len=len(prefix.token_ids) - 1)
    with pytest.raises(ValueError, match="exceeds checkpoint model_max_seq_len"):
        validate_turn_source_geometry(records=records, feature_shard_paths=paths, contract=contract,
            extraction_tokenizer=source, tokenizer=tokenizer)


@pytest.mark.parametrize("mutation,message", [
    # The first four legitimately share the replay guard: bucket, feature-id, span and
    # whole-row swaps all stay in-domain, so only re-running the source can reject them.
    ("buckets", "cached selection/raw spans disagree with source replay"),
    ("feature-id", "cached selection/raw spans disagree with source replay"),
    ("spans", "cached selection/raw spans disagree with source replay"),
    ("row-swap", "cached selection/raw spans disagree with source replay"),
    ("missing-row", "turn cache must contain exactly the enumerated rows"),
    ("contract", "turn context contract disagrees with checkpoint-derived expectations"),
    ("default-validator", "turn cache validation requires an explicit checkpoint contract"),
])
def test_deep_replay_rejects_wrong_cache_before_forward(tmp_path, monkeypatch, mutation, message):
    # kills: treating geometry alone as selected-cache parity, missing prefix lookup, or implicit turn admission.
    _, records, paths, contract, tokenizer, source, provider = loaded_fixture(tmp_path, monkeypatch)
    if mutation in {"buckets", "feature-id", "spans", "missing-row"}:
        key = next(iter(provider._rows))
        tokens, segments = provider._rows[key]
        if mutation == "missing-row":
            del provider._rows[key]
        else:
            # Every mutation stays inside the cached vocabulary and geometry
            # domains, so only the source replay can reject it.
            flipped = next(fid for fid in range(contract.lineage.n_sae_features)
                           if fid not in segments[0].feature_ids)
            changed = replace(segments[0], **{
                "buckets": {"buckets": tuple(0 for _ in segments[0].buckets)},
                "feature-id": {"feature_ids": (flipped, *segments[0].feature_ids[1:])},
                "spans": {"char_end": segments[0].char_end + 1},
            }[mutation])
            assert changed != segments[0]
            provider._rows[key] = (tokens, (changed, *segments[1:]))
    elif mutation == "row-swap":
        (first_key, (first_tokens, first_payload)), (second_key, (second_tokens, second_payload)) = (
            provider._rows.items())
        assert first_payload != second_payload
        provider._rows[first_key] = (first_tokens, second_payload)
        provider._rows[second_key] = (second_tokens, first_payload)
    elif mutation == "contract":
        contract = replace(contract, semantic={**contract.semantic, "source_context_window": 8})
    with pytest.raises(ValueError, match=message):
        if mutation == "default-validator":
            validate_context_cache_records(records=records, provider=provider, tokenizer=tokenizer)
        else:
            validate_turn_context_cache_records(records=records, provider=provider, tokenizer=tokenizer,
                turn_contract=contract, extraction_tokenizer=source, feature_shard_paths=paths)


def test_extraction_admission_and_real_parquet_reader(tmp_path, monkeypatch):
    # kills: phase-free extraction, omitted turn starts, and claiming BOS exclusion removes BOS coordinates.
    cfg, records, _, contract, _, _ = build_fixture(tmp_path, monkeypatch)
    parquet = pq.ParquetFile(str(cfg.context_parquet_dir) + "/shard_00000.parquet")
    phase = "standard_sft_val"
    assert check_sft_data_ready(cfg.context_parquet_dir, phase)[0]
    _validate_segment_mode_dataset_phase(TURN, phase)
    for wrong_phase in (None, "base"):
        # Both a missing and a pretraining phase hit the one dataset-phase guard.
        with pytest.raises(ValueError, match="turn-delimited segmentation requires an SFT dataset_phase"):
            _validate_segment_mode_dataset_phase(TURN, wrong_phase)
    _validate_extraction_parquet_schema(parquet_path="fixture", schema_names=parquet.schema_arrow.names,
        dataset_phase=phase, segment_mode=TURN)
    texts, starts = _read_text_and_turn_chunks(parquet, range(parquet.num_row_groups), 1, TURN)
    assert texts == [[row.context_text] for row in records]
    assert starts == [[list(row.turn_char_starts)] for row in records]
    assert contract.semantic["ignore_bos_features"] is True
    assert contract.semantic["source_bos_char_len"] == 3


def test_summary_file_and_wrong_enumeration(tmp_path, monkeypatch):
    # kills: dropping validator summary output or accepting a reordered live task enumeration.
    cfg, records, _, _, tokenizer, _, _ = loaded_fixture(tmp_path, monkeypatch)
    cfg.summary_file = str(tmp_path / "summary.json")
    builder.validate_cache_from_config(cfg)
    assert json.loads((tmp_path / "summary.json").read_text())["cached_contexts"] == 2
    standalone = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": True})
    assert isinstance(standalone, DictConfig)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(validator, "enumerate_expected_eval_contexts", lambda *_: list(reversed(records)))
    with pytest.raises(ValueError, match="turn selected eval enumeration differs from ordered contexts_file"):
        validator.validate_cache_from_config(standalone)


class FineSourceTokenizer(_DifferentSourceTokenizer):
    """Fine source windows, with an atomic prompt terminator to avoid unused prompt tails."""

    def encode(self, text):
        pieces = []
        for index, part in enumerate(text.split("<user_end>")):
            if index:
                pieces.append("<user_end>")
            pieces.extend(part)
        return [self.bos_token_id, *(self._id_for_piece(piece) for piece in pieces)]


class ByteChatTokenizer(ChatTokenizer):
    """Multibyte Unicode decodes through zero-width partial-byte positions."""

    def encode(self, text, **kwargs):
        return [256 + byte for byte in text.encode("utf-8")]

    def decode(self, ids):
        inverse = {value: key for key, value in self._specials.items()}
        pieces, buffer = [], bytearray()
        for token in ids:
            if token in inverse:
                pieces.append(buffer.decode("utf-8", errors="ignore"))
                buffer.clear()
                pieces.append(inverse[token])
            else:
                buffer.append(token - 256)
        pieces.append(buffer.decode("utf-8", errors="ignore"))
        return "".join(pieces)


def test_raw_unused_slots_utf8_zero_width_and_short_tail(tmp_path, monkeypatch):
    # kills: tiny-tail reassignment, dropping unused internal slots, double BOS shift, zero-width division.
    tokenizer = ByteChatTokenizer()
    source = FineSourceTokenizer()
    records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[{"messages": [
        {"role": "user", "content": "é🦋"}, {"role": "assistant", "content": "答"}]}],
        max_problems=1, task_name="unicode", segment_mode=TURN)
    cfg, records, _, _, _, _ = build_fixture(tmp_path, monkeypatch, records=records)
    synthetic_shards(tmp_path / "features", records, source)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda _: source)
    contract, _ = common.turn_contract_from_config(cfg, tokenizer)
    manifest = builder.build_cache_from_config(cfg)
    summary = common.validate_turn_cache_from_config(cfg, tokenizer, cache_root=cfg.output_dir)
    provider = SafetensorsContextFeatureProvider(manifest.parent,
        identity=common.cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=TURN, expected_turn_contract=contract)
    target = summary["turn_target_checks"][0]
    assert len(set(target["lockstep_assignments"])) < len(target["lockstep_segments"])
    segments = provider.get(records[0].token_ids)
    assert segments is not None
    assert len(segments) > len(set(target["lockstep_assignments"]))
    offsets = token_to_char_offsets(tokenizer, list(records[0].token_ids))
    assert any(left == right for left, right in pairwise(offsets))


def test_turn_presets_clear_fixed_defaults():
    # kills: inheriting legacy threshold/SAE identity defaults into the turn CLI runbook.
    config_dir = str(Path(__file__).resolve().parents[2] / "configs/eval")
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        build = compose(config_name="build_eval_context_cache_turn")
        validate = compose(config_name="validate_eval_context_cache_turn")
    assert build.segment_mode == validate.segment_mode == TURN.value
    assert build.feature_selection is None
    assert build.stats_path is None
    assert build.n_sae_features is None
    assert all(value is None for value in build.identity.values())
    assert build.evals == validate.evals == "chat_target"


@pytest.mark.parametrize("field", ["parquet_files", "jsonl_sha256", "feature_files"])
def test_provenance_tampering(tmp_path, monkeypatch, field):
    # kills: accepting reordered or relabeled input files without checking their committed digests.
    cfg, _, _, _, tokenizer, _, provider = loaded_fixture(tmp_path, monkeypatch)
    manifest = json.loads(provider.path.read_text())
    manifest["metadata"]["input_provenance"][field] = "wrong"
    provider.path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match=f"turn cache ordered {field} provenance disagrees"):
        common.validate_turn_cache_from_config(cfg, tokenizer, cache_root=cfg.output_dir)


@pytest.mark.parametrize("field", ["extraction_tokenizer", "feature_shard_paths"])
def test_partial_validation_inputs_and_unsupported_targets(tmp_path, monkeypatch, field):
    # kills: shallow turn validation when source inputs or supported target policy are absent.
    cfg, records, paths, contract, tokenizer, source, provider = loaded_fixture(tmp_path, monkeypatch)
    args = {"records": records, "provider": provider, "tokenizer": tokenizer, "turn_contract": contract,
            "extraction_tokenizer": source, "feature_shard_paths": paths}
    del args[field]
    with pytest.raises(TypeError, match=f"missing 1 required keyword-only argument: '{field}'"):
        validate_turn_context_cache_records(**args)
    cfg.target_feature_configs = []
    with pytest.raises(ValueError, match="turn validation supports target-real configs only"):
        builder.validate_cache_from_config(cfg)


@pytest.mark.parametrize("entrypoint", ["builder", "standalone"])
@pytest.mark.parametrize("field,delta", [
    ("target_char_end", -1), ("target_char_end", 1),
    ("target_char_start", -1), ("target_char_start", 1),
    ("target_start", -1), ("target_start", 1),
])
def test_shared_validator_authenticates_target_interval(tmp_path, monkeypatch, entrypoint, field, delta):
    cfg, records, paths, _, tokenizer, _, _ = loaded_fixture(tmp_path, monkeypatch)
    _, _, provenance = common.load_turn_context_inputs(cfg, tokenizer)
    full = records[0]
    changed = dict(full.metadata)
    if field == "target_start":
        changed[field] += delta
    else:
        offsets = token_to_char_offsets(tokenizer, list(full.token_ids))
        index = offsets.index(changed[field])
        changed[field] = offsets[index + delta] if index + delta < len(offsets) else offsets[-1] + 1
    records = [replace(full, metadata=changed), *records[1:]]
    # Keep inventory fingerprints fixed to isolate the shared deep validator;
    # provenance rejection is separately covered by test_provenance_tampering.
    monkeypatch.setattr(common, "load_turn_context_inputs", lambda *_: (records, paths, provenance))
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    invoke = builder.validate_cache_from_config if entrypoint == "builder" else validator.validate_cache_from_config
    if entrypoint == "standalone":
        cfg = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": False})
    assert isinstance(cfg, DictConfig)
    # Both deltas of a field share that field's comparison, but the message must name
    # the field that moved (target_start is not a substring of target_char_start).
    with pytest.raises(ValueError, match=f"{field} disagrees with renderer supervision interval"):
        invoke(cfg)


class TruncatedToolTokenizer(ChatTokenizer):
    def render_conversation(self, conversation, max_tokens=2048):
        return super().render_conversation(conversation, max_tokens=13)


@pytest.mark.parametrize("entrypoint", ["builder", "standalone"])
@pytest.mark.parametrize("boundary", ["shortened", "extended"])
def test_truncated_tool_interval_entrypoints(tmp_path, monkeypatch, entrypoint, boundary):
    tokenizer = TruncatedToolTokenizer()
    conversation = {"messages": [{"role": "user", "content": "Q"}, {"role": "assistant", "content": [
        {"type": "python", "text": "1+1"}, {"type": "python_output", "text": "2345"},
        {"type": "text", "text": "AB"}]}]}
    records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[conversation],
        max_problems=1, task_name="truncated", segment_mode=TURN)
    cfg, records, paths, _, _, _ = build_fixture(tmp_path, monkeypatch, records=records, tokenizer=tokenizer)
    builder.build_cache_from_config(cfg)
    monkeypatch.setattr(validator, "get_tokenizer", lambda: tokenizer)
    invoke = builder.validate_cache_from_config if entrypoint == "builder" else validator.validate_cache_from_config
    if entrypoint == "standalone":
        cfg = OmegaConf.merge(cfg, {"cache_dir": cfg.output_dir, "enumerate_evals": False})
    assert isinstance(cfg, DictConfig)
    invoke(cfg)  # The correctly truncated reference is valid in both entrypoints.
    _, _, provenance = common.load_turn_context_inputs(cfg, tokenizer)
    full = records[0]
    offsets = token_to_char_offsets(tokenizer, list(full.token_ids))
    end = offsets.index(full.metadata["target_char_end"])
    changed_end = offsets[end - 1] if boundary == "shortened" else offsets[-1]
    assert changed_end != full.metadata["target_char_end"]
    records = [replace(full, metadata={**full.metadata, "target_char_end": changed_end}), records[1]]
    monkeypatch.setattr(common, "load_turn_context_inputs", lambda *_: (records, paths, provenance))
    # Both boundaries move the same field, so both legitimately hit its comparison.
    with pytest.raises(ValueError, match="target_char_end disagrees with renderer supervision interval"):
        invoke(cfg)


def test_exact_pooled_current_next_and_lockstep_plans(tmp_path, monkeypatch):
    tokenizer, source = ChatTokenizer(), FineSourceTokenizer()
    records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[{"messages": [
        {"role": "user", "content": "Q"}, {"role": "assistant", "content": "a" * 45}]}],
        max_problems=1, task_name="plans", segment_mode=TURN)
    cfg, records, _, _, _, _ = build_fixture(tmp_path, monkeypatch, records=records, size=32)
    paths = synthetic_shards(tmp_path / "features", records, source, size=32)
    meta_path = Path(cfg.checkpoint_meta_path)
    meta = json.loads(meta_path.read_text())
    lineage = meta["structured_sft_online_sae_lineage_v1"]
    lineage["negative_selection"] = None
    lineage["feature_selection"].update(min_k=1, max_k=2)
    meta_path.write_text(json.dumps(meta))
    # Full-row prompt, prefix prompt, target slots and prefix marker all differ.
    for path, fids in zip(paths, ([0, 3, 4, 5], [1, 7]), strict=True):
        data = load_features(path)
        assert data["n_segments"] == len(fids)
        data["feature_ids"] = t.tensor(fids, dtype=t.int32)
        data["feature_mags"] = t.ones(len(fids), dtype=t.float32)
        data["offsets"] = t.arange(len(fids) + 1, dtype=t.int64)
        save_features(data, path)
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda _: source)
    builder.build_cache_from_config(cfg)
    summary = common.validate_turn_cache_from_config(cfg, tokenizer, cache_root=cfg.output_dir)
    check = summary["turn_target_checks"][0]
    assert check["pooled_features"] == {3: 3, 4: 3, 5: 3}
    assert check["lockstep_segments"] == [{3: 3}, {4: 3}, {5: 3}]
    assert check["plans"] == [
        {"encoder_slots": [{1: 3}, {3: 3, 4: 3, 5: 3}], "generation_slot": 1},
        {"encoder_slots": [{3: 3, 4: 3, 5: 3}], "generation_slot": 0},
        {"encoder_slots": [{1: 3}, {3: 3}, {4: 3}, {5: 3}], "generation_slot": 1},
    ]


@pytest.mark.parametrize("flag,message", [
    ("require_nonempty_context_features", "no selected context features"),
    ("require_nonempty_context_segments", "empty selected context segment"),
])
def test_turn_nonempty_selection_requirements(tmp_path, monkeypatch, flag, message):
    cfg, _, _, _, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    meta_path = Path(cfg.checkpoint_meta_path)
    meta = json.loads(meta_path.read_text())
    meta["structured_sft_online_sae_lineage_v1"]["negative_selection"] = None
    meta["structured_sft_online_sae_lineage_v1"]["feature_selection"].update(min_k=0, max_k=1)
    meta_path.write_text(json.dumps(meta))
    builder.build_cache_from_config(cfg)
    cfg.require_nonempty_context_features = False
    cfg.require_nonempty_context_segments = False
    common.validate_turn_cache_from_config(cfg, tokenizer, cache_root=cfg.output_dir)
    cfg[flag] = True
    with pytest.raises(ValueError, match=message):
        common.validate_turn_cache_from_config(cfg, tokenizer, cache_root=cfg.output_dir)
