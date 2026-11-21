"""CPU guards for fixed-reachable behavior identified by the stage-7 audit."""

import json
from pathlib import Path

import pytest
import torch as t
from omegaconf import OmegaConf

from nanochat.checkpoint_manager import ModelSource
from nanochat.core_eval import TaskMeta, TaskType
from scripts.eval import chat_eval
from steering.conditional.encoder_decoder.eval_adapter import build_segmented_feature_encoder_inputs
from steering.eval.context_enumeration import enumerate_selected_core_target_contexts
from steering.segmentation import SegmentMode
from tests.conftest import MockTokenizer, build_tiny_decoder


@pytest.mark.parametrize("checkpoint_mode,configured_mode", [
    (SegmentMode.turn_delimited_fixed_tokens, None),
    (SegmentMode.fixed_tokens, SegmentMode.turn_delimited_fixed_tokens),
])
def test_decoder_only_admission_precedes_task_and_cache_io(tmp_path, monkeypatch, checkpoint_mode, configured_mode):
    model = build_tiny_decoder()
    meta = {"sft_segment_mode": checkpoint_mode.value}
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/eval/chat_eval.yaml")
    cfg.device_type, cfg.dtype = "cpu", "float32"
    cfg.source = ModelSource.BASE.value
    cfg.step = 0
    (tmp_path / "meta_000000.json").write_text(json.dumps(meta))
    t.save(model.state_dict(), tmp_path / "model_000000.pt")
    monkeypatch.setattr(chat_eval, "get_checkpoint_dir", lambda *args, **kwargs: str(tmp_path))
    cfg.segment_mode = None if configured_mode is None else configured_mode.value
    monkeypatch.setattr(chat_eval, "compute_init", lambda _: (False, 0, 0, 1, t.device("cpu")))
    monkeypatch.setattr(chat_eval, "load_model", lambda *args, **kwargs: (model, MockTokenizer(), meta))
    monkeypatch.setattr(chat_eval, "parse_feature_configs", lambda *args: pytest.fail("invalid decoder reached task/config IO"))
    monkeypatch.setattr(chat_eval, "parse_target_real_configs_from_cfg", lambda *args, **kwargs: pytest.fail("invalid decoder reached cache IO"))
    with pytest.raises(ValueError, match="turn target-real requires an encoder-decoder model"):
        chat_eval.main.__wrapped__(cfg)


def test_malformed_mode_precedes_fixed_cluster_guard(tmp_path, monkeypatch):
    cfg = OmegaConf.load(Path(__file__).resolve().parents[2] / "configs/eval/chat_eval.yaml")
    cfg.device_type, cfg.dtype = "cpu", "float32"
    cfg.source = ModelSource.BASE.value
    model = build_tiny_decoder()
    meta = {"sft_segment_mode": "malformed", "feature_clusters_canonical": {"invalid": True}}
    cfg.step = 0
    (tmp_path / "meta_000000.json").write_text(json.dumps(meta))
    monkeypatch.setattr(chat_eval, "get_checkpoint_dir", lambda *args, **kwargs: str(tmp_path))
    monkeypatch.setattr(chat_eval, "compute_init", lambda _: (False, 0, 0, 1, t.device("cpu")))
    monkeypatch.setattr(chat_eval, "load_model", lambda *args, **kwargs: (model, MockTokenizer(), meta))
    monkeypatch.setattr(chat_eval, "reject_feature_clusters_metadata", lambda *args: pytest.fail("cluster guard preceded mode validation"))
    with pytest.raises(ValueError, match="sft_segment_mode"):
        chat_eval.main.__wrapped__(cfg)


def test_core_enumeration_role_json_literal():
    rows = enumerate_selected_core_target_contexts(
        tokenizer=MockTokenizer(), data=[{"context": "ab", "continuation": "c"}],
        task_meta=TaskMeta(task_type=TaskType.LANGUAGE_MODELING, num_fewshot=0, continuation_delimiter=" "),
        seed=42, task_label="literal", max_seq_len=2048)
    assert json.dumps([row.metadata["target_real_role"] for row in rows]) == '["full", "context_prefix"]'
    assert [row.suite for row in rows] == ["core_target", "core_target"]


@pytest.mark.parametrize("conditions,expected_ids,expected_buckets,expected_mask", [
    ([], [[[0]]], [[[0]]], [[[False]]]),
    ([{}, {}], [[[0], [0]]], [[[0], [0]]], [[[False], [False]]]),
    ([{7: 2, 1: 0}, {}, {3: 1}], [[[7, 1], [0, 0], [3, 0]]],
     [[[2, 0], [0, 0], [1, 0]]], [[[True, True], [False, False], [True, False]]]),
])
def test_exported_packing_helper_literal(conditions, expected_ids, expected_buckets, expected_mask):
    ids, buckets, mask = build_segmented_feature_encoder_inputs(conditions, t.device("cpu"))
    assert ids.dtype == buckets.dtype == t.int64
    assert mask.dtype == t.bool
    assert (ids.tolist(), buckets.tolist(), mask.tolist()) == (expected_ids, expected_buckets, expected_mask)


def test_exported_packing_helper_rejects_missing_signal():
    with pytest.raises(ValueError, match="Missing segment conditioning signal at index 1"):
        build_segmented_feature_encoder_inputs([{1: 0}, None], t.device("cpu"))
