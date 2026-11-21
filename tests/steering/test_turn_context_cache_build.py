"""Real raw-shard conversion, lineage RNG and ordered-file provenance."""

import json
import re
from dataclasses import replace
from typing import Any

import pytest
import torch as t
from omegaconf import OmegaConf

from scripts.eval import build_eval_context_cache as builder
from scripts.eval import context_cache_common as common
from steering.conditional.encoder_decoder.sft_selection import select_conversation_segments
from steering.conditional.feature_vocab import FeatureVocab
from steering.eval.context_cache_build import (
    build_turn_cache_entries,
    turn_source_segments,
)
from steering.eval.context_cache_io import SafetensorsContextFeatureProvider
from steering.eval.context_cache_validation import validate_turn_source_geometry
from steering.eval.context_enumeration import dedupe_context_records, enumerate_chat_target_contexts
from steering.feature_io import load_features, save_features
from steering.refined_types import PositiveInt
from steering.segmentation import SegmentMode, turn_segments_for_document
from tests.steering.chatcore_cache_helpers import ChatTokenizer
from tests.steering.test_clustered_eval_context_cache import _cluster_state
from tests.steering.test_context_cache_in_memory_conversion import (
    _DifferentSourceTokenizer,
    _pipeline,
)
from tests.steering.test_turn_context_contract import (
    checkpoint_fixture,
    cluster_checkpoint,
    production_checkpoint_fixture,
    production_clustered_checkpoint_fixture,
)

TURN = SegmentMode.turn_delimited_fixed_tokens


class StraddlingSourceTokenizer(_DifferentSourceTokenizer):
    """Model Gemma's boundary geometry with end-marker tokens ending in ``><``."""

    def encode(self, text):
        pieces = [self.decode([token]) for token in super().encode(text)[1:]]
        for index in range(1, len(pieces)):
            if pieces[index - 1].endswith(">") and pieces[index].startswith("<"):
                pieces[index - 1] += "<"
                pieces[index] = pieces[index][1:]
        return [self.bos_token_id, *(self._id_for_piece(piece) for piece in pieces)]


class UnroutedStraddlingSourceTokenizer(StraddlingSourceTokenizer):
    """Put ``_end><`` in its own four-token-window tail before the marker.

    The decoder's whole user-end token routes to the preceding window by
    midpoint; the assistant marker routes to the following window. Splitting
    the earlier ``<user`` piece aligns the unused tail without changing text.
    """

    def encode(self, text):
        pieces = []
        for token in super().encode(text)[1:]:
            piece = self.decode([token])
            if piece == "<user_end><":
                # The first turn includes BOS. These fixtures have one user turn.
                split = (-(1 + len(pieces)) - 1) % 4
                pieces.extend([*"<user"[:split], "<user"[split:], "_end><"])
            else:
                pieces.append(piece)
        return [self.bos_token_id, *(self._id_for_piece(piece) for piece in pieces)]


def synthetic_shards(directory, records, source, *, size=4, split=1, window=4096):
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for shard_index, begin in enumerate(range(0, len(records), split)):
        rows = records[begin:begin + split]
        docs, indices, positions, starts, ends, counts, fids, mags = [], [], [], [], [], [], [], []
        for doc, row in enumerate(rows):
            ids = source.encode(row.context_text)
            segments, cs, ce, _ = turn_segments_for_document(ids[:window], source, row.context_text,
                row.turn_char_starts, PositiveInt.parse(size))
            counts.append(len(ids))
            for index, ((position, _end), start, end) in enumerate(zip(segments, cs, ce, strict=True)):
                docs.append(doc)
                indices.append(index)
                positions.append(position)
                starts.append(start)
                ends.append(end)
                fids.extend([0, 1, 2, 3, 4])
                mags.extend([0.11, 0.25, 0.53, 0.71, 0.92])
        data = {"offsets": t.arange(0, len(fids) + 1, 5, dtype=t.int64),
                "feature_ids": t.tensor(fids, dtype=t.int32), "feature_mags": t.tensor(mags, dtype=t.float32),
                "doc_ids": t.tensor(docs, dtype=t.int32), "segment_indices": t.tensor(indices, dtype=t.int32),
                "segment_positions": t.tensor(positions, dtype=t.int32),
                "segment_char_starts": t.tensor(starts, dtype=t.int32), "segment_char_ends": t.tensor(ends, dtype=t.int32),
                "doc_token_counts": t.tensor(counts, dtype=t.int32), "n_docs": len(rows), "n_segments": len(docs),
                "layer": 22, "context_window": window, "pooling": "max_pool", "segment_mode": TURN.value,
                "segment_size": size, "context_independent_features": True, "ignore_bos_features": True,
                "l0_tier": "medium"}
        path = directory / f"shard_{shard_index:05d}_features.safetensors.zst"
        save_features(data, path)
        paths.append(path)
    return paths


def build_fixture(tmp_path, monkeypatch, *, records=None, mask=None, tokenizer=None, source=None, size=4, seed=19, production=False, clustered=False):
    tokenizer = ChatTokenizer() if tokenizer is None else tokenizer
    source = _DifferentSourceTokenizer() if source is None else source
    if production:
        factory = production_clustered_checkpoint_fixture if clustered else production_checkpoint_fixture
        meta, meta_path, _ = factory(tmp_path)
        size = 32
    else:
        meta, meta_path, _ = checkpoint_fixture(tmp_path, size=size, seed=seed)
    if mask is not None:
        meta["feature_mask_canonical"] = {"mode": "whitelist", "ids": mask}
        meta["feature_vocab"] = FeatureVocab(8, tuple(mask)).metadata_dict()
        meta_path.write_text(json.dumps(meta))
    if clustered and not production:
        cluster_checkpoint(meta, tmp_path)
        meta_path.write_text(json.dumps(meta))
    if records is None:
        records = enumerate_chat_target_contexts(tokenizer=tokenizer, task_object=[{"messages": [
            {"role": "user", "content": "a prompt with unequal token widths"},
            {"role": "assistant", "content": [
                {"type": "python", "text": "1+1"}, {"type": "python_output", "text": "2"},
                {"type": "text", "text": "#### 2"},
            ]}]}], max_problems=None, task_name="unit", segment_mode=TURN)
    records = dedupe_context_records(records)
    contexts = builder.write_context_records(tmp_path / "rows.jsonl", records)
    builder.write_context_parquet(tmp_path / "parquet", records)
    paths = synthetic_shards(tmp_path / "features", records, source, size=size)
    monkeypatch.setattr(builder, "get_tokenizer", lambda: tokenizer)
    monkeypatch.setattr(common.AutoTokenizer, "from_pretrained", lambda _: source)
    cfg = OmegaConf.create({
        "segment_mode": TURN.value, "checkpoint_meta_path": str(meta_path), "model_max_seq_len": 2048,
        "contexts_file": str(contexts), "context_parquet_dir": str(tmp_path / "parquet"),
        "feature_dir": str(tmp_path / "features"), "output_dir": str(tmp_path / "cache"),
        "shard_size": 1, "overwrite": False, "identity": {}, "metadata": {},
    })
    contract, _ = common.turn_contract_from_config(cfg, tokenizer)
    return cfg, records, paths, contract, tokenizer, source


def convert(records, paths, contract, tokenizer):
    return build_turn_cache_entries(
        context_records=records, feature_shard_paths=paths,
        turn_contract=contract, tokenizer=tokenizer,
    )


@pytest.mark.parametrize("mask", [None, [0, 2, 3, 5, 6]])
def test_raw_spans_selection_and_shard_boundary_parity(tmp_path, monkeypatch, mask):
    # kills: legacy selector/default seed, RNG reset per shard, snapped spans, omitted negatives/mask.
    cfg, records, paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch, mask=mask)
    entries = convert(records, list(reversed(paths)), contract, tokenizer)
    # Schema-5 rows carry their own producing geometry, not the fixed default.
    assert all(entry.segment_mode is TURN and entry.token_assignments is None for entry in entries)
    expected_context = contract.build_selection_context()
    for index, (path, entry) in enumerate(zip(paths, entries, strict=True)):
        raw = turn_source_segments(load_features(path), 0)
        selected = select_conversation_segments(raw, ctx=expected_context, example_idx=index)
        assert [(seg.char_start, seg.char_end) for seg in entry.segments] == [(seg.char_start, seg.char_end) for seg in raw]
        for got, expected in zip(entry.segments, selected, strict=True):
            assert got.feature_ids == tuple(expected.feature_ids.tolist())
            assert got.buckets == tuple(expected.buckets.tolist())
            assert 3 in got.buckets
            if mask is not None:
                assert set(got.feature_ids) <= set(mask)
    joined = synthetic_shards(tmp_path / "joined", records, source, split=len(records))
    assert convert(records, joined, contract, tokenizer) == entries
    manifest = builder.build_cache_from_config(cfg)
    provider = SafetensorsContextFeatureProvider(manifest.parent, identity=common.cache_identity_from_config({}, turn_contract=contract),
        expected_source_segment_mode=TURN, expected_turn_contract=contract)
    assert [provider.get(row.token_ids) for row in records] == [entry.segments for entry in entries]
    metadata = json.loads(manifest.read_text())["metadata"]
    assert len(metadata["contexts_fingerprint"]) == 64
    assert len(metadata["input_provenance"]["feature_files"]) == 2
    assert metadata["training_feature_dir"] != metadata["input_provenance"]["evaluation_feature_dir"]


@pytest.mark.parametrize("field,value", [
    ("segment_mode", "fixed_tokens"), ("segment_size", 32), ("context_window", 2048),
    ("layer", 8), ("pooling", "mean_pool"), ("l0_tier", "small"),
    ("context_independent_features", False), ("ignore_bos_features", False),
])
def test_shard_setting_rejections(tmp_path, monkeypatch, field, value):
    # kills: accepting source extraction settings that differ from the checkpoint contract.
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    data = load_features(paths[-1])
    data[field] = value
    save_features(data, paths[-1])
    # One shard-settings guard per field; the echoed observed value keeps the eight cases distinct.
    with pytest.raises(ValueError, match=re.escape(f"turn shard {field}={value!r}")):
        convert(records, paths, contract, tokenizer)


@pytest.mark.parametrize("mutation,message", [
    ("more", "turn feature shards contain more docs than context_records"),
    ("fewer", "turn feature shards contain 1 docs but context_records has 2"),
    ("indices", "turn source doc 0 has missing/duplicate segment indices"),
    ("raw-id", "raw feature ID outside checkpoint SAE width"),
    ("annotations", "Turn text/boundaries disagree with exact tokens for tokens_sha256:"),
])
def test_source_row_and_selection_rejections(tmp_path, monkeypatch, mutation, message):
    # kills: missing/extra documents, duplicate source ordinals, invalid raw IDs, stale annotations.
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    if mutation == "more":
        paths = paths + paths
    elif mutation == "fewer":
        paths = paths[:1]
    elif mutation == "annotations":
        records = [replace(records[0], context_text="X" + records[0].context_text), records[1]]
    else:
        data = load_features(paths[0])
        data["segment_indices" if mutation == "indices" else "feature_ids"][0] = 100
        save_features(data, paths[0])
    with pytest.raises(ValueError, match=re.escape(message)):
        convert(records, paths, contract, tokenizer)


# "seed", "feature_selection" and "model_max_seq_len" share the CLI-override guard, which names the
# offending config key; "identity" and "extraction_tokenizer_id" hit their own guards.
@pytest.mark.parametrize("override,message", [
    ("seed", "configured seed conflicts with checkpoint turn contract; use the turn cache preset"),
    ("feature_selection", "configured feature_selection conflicts with checkpoint turn contract; use the turn cache preset"),
    ("model_max_seq_len", "configured model_max_seq_len conflicts with checkpoint turn contract; use the turn cache preset"),
    ("identity", "explicit turn feature_selection_hash does not match checkpoint contract"),
    ("extraction_tokenizer_id", "extraction_tokenizer_id conflicts with checkpoint SAE model"),
])
def test_cli_assertions_fail_before_context_io(tmp_path, monkeypatch, override, message):
    # kills: silent legacy defaults/CLI overrides and writing output before checkpoint validation.
    cfg, _, _, _, _, _ = build_fixture(tmp_path, monkeypatch)
    cfg[override] = {"seed": 42, "feature_selection": {"threshold": 2048.9}, "model_max_seq_len": 1024,
                     "identity": {"feature_selection_hash": "0" * 64}, "extraction_tokenizer_id": "other/model"}[override]
    monkeypatch.setattr(builder, "load_turn_context_inputs", lambda *_: pytest.fail("override reached context IO"))
    with pytest.raises(ValueError, match=re.escape(message)):
        builder.build_cache_from_config(cfg)
    assert not (tmp_path / "cache").exists()


def test_parquet_order_disagreement(tmp_path, monkeypatch):
    # kills: selecting shards in a different order from the committed enumeration inputs.
    cfg, records, _, _, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    builder.write_context_parquet(cfg.context_parquet_dir, list(reversed(records)))
    with pytest.raises(ValueError, match=re.escape(
            "turn JSONL and parquet row order/keys/text/boundaries disagree")):
        common.load_turn_context_inputs(cfg, tokenizer)


@pytest.mark.parametrize("bound", ["negative", "upper"])
@pytest.mark.parametrize("surface", ["converter", "geometry"])
def test_orphan_source_segment_rejected(tmp_path, monkeypatch, bound, surface):
    _, records, paths, contract, tokenizer, source = build_fixture(tmp_path, monkeypatch)
    data = load_features(paths[0])
    at_front = bound == "negative"
    for field in ("doc_ids", "segment_indices", "segment_positions", "segment_char_starts", "segment_char_ends"):
        row = data[field][:1].clone()
        if field == "doc_ids":
            row[0] = -1 if at_front else data["n_docs"]
        data[field] = t.cat((row, data[field]) if at_front else (data[field], row))
    count = int(data["offsets"][1])
    for field in ("feature_ids", "feature_mags"):
        row = data[field][:count].clone()
        data[field] = t.cat((row, data[field]) if at_front else (data[field], row))
    data["offsets"] = (t.cat((t.tensor([0]), data["offsets"] + count)) if at_front else
                       t.cat((data["offsets"], data["offsets"][-1:] + count)))
    data["n_segments"] += 1
    save_features(data, paths[0])
    # Both bounds trip the same range guard on both surfaces; the pinned interval proves the guard
    # saw a single-document shard rather than some other doc_id complaint.
    with pytest.raises(ValueError, match=re.escape("source segment doc_id outside [0, 1)")):
        if surface == "converter":
            convert(records, paths, contract, tokenizer)
        else:
            validate_turn_source_geometry(records=records, feature_shard_paths=paths, contract=contract,
                extraction_tokenizer=source, tokenizer=tokenizer)


def test_duplicate_converter_enumeration(tmp_path, monkeypatch):
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    with pytest.raises(ValueError, match=re.escape("turn conversion requires deduplicated enumeration rows")):
        convert([*records, records[0]], paths, contract, tokenizer)


@pytest.mark.parametrize("field", ["pipeline", "feature_clusters", "n_sae_features",
    "num_buckets", "source_bos_char_len"])
def test_turn_converter_admits_no_selection_or_geometry_override(tmp_path, monkeypatch, field):
    # kills: re-opening a turn conversion argument the checkpoint contract owns.
    _, records, paths, contract, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    args: dict[str, Any] = {"context_records": records, "feature_shard_paths": paths,
                            "turn_contract": contract, "tokenizer": tokenizer}
    args[field] = (_pipeline() if field == "pipeline" else
                   _cluster_state(tmp_path) if field == "feature_clusters" else 7)
    with pytest.raises(TypeError, match=f"unexpected keyword argument '{field}'"):
        build_turn_cache_entries(**args)


@pytest.mark.parametrize("field,message", [
    ("context_parquet_dir", "turn context_parquet_dir contains no parquet files"),
    ("feature_dir", "turn feature_dir contains no extracted shards"),
])
def test_missing_input_inventory(tmp_path, monkeypatch, field, message):
    cfg, _, _, _, tokenizer, _ = build_fixture(tmp_path, monkeypatch)
    cfg[field] = str(tmp_path / "absent")
    with pytest.raises(ValueError, match=re.escape(message)):
        common.load_turn_context_inputs(cfg, tokenizer)
