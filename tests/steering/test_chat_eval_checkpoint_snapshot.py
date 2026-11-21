"""Receipts describe the loaded snapshot even when checkpoint files race."""

import hashlib
import json
import re
import shutil
from copy import deepcopy

import pytest
import yaml

from nanochat import checkpoint_manager
from scripts.eval import chat_eval
from steering.eval import turn_checkpoint_provenance as provenance
from steering.eval.chat_eval_result_identity import TurnChatEvalCheckpointSnapshot
from steering.segmentation import SegmentMode
from steering.sft.segment_mode_identity import ONLINE_SAE_LINEAGE_META_KEY
from tests.steering.test_chat_eval_turn_cache_integration import turn_main_fixture


@pytest.mark.parametrize("clustered", [False, True])
def test_real_loader_extends_sft_metadata_and_receipt_binds_original_bytes(tmp_path, monkeypatch, clustered):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=clustered, real_loader=True)
    fixture.cfg.task_name = "MMLU"
    payload = fixture.checkpoint_path.read_bytes()
    before = json.loads(payload)
    assert before["user_config"]["negative_selection"] is None
    assert before["user_config"]["num_buckets"] is None
    assert "window_pattern" not in before["model_config"]
    assert "encoder" not in before["user_config"]
    loaded = []
    original = chat_eval.load_model

    def capture(*args, **kwargs):
        model, tokenizer, meta = original(*args, **kwargs)
        assert model is not fixture.model  # Real construction, not the fixture's stand-in loader.
        loaded.append(meta)
        return model, tokenizer, meta

    monkeypatch.setattr(chat_eval, "load_model", capture)
    chat_eval.main.__wrapped__(fixture.cfg)
    assert fixture.loaded == [(fixture.checkpoint_path.parent, 1116)]
    assert len(loaded) == 1
    after = loaded[0]
    assert after["model_config"]["window_pattern"] == "L"
    assert after["user_config"]["negative_selection"] == before[ONLINE_SAE_LINEAGE_META_KEY]["negative_selection"]
    assert after["user_config"]["feature_prediction"] is not None
    assert after["user_config"]["num_buckets"] == before["num_buckets"]
    assert after["user_config"]["encoder"] and after["user_config"]["cross_attention"]
    assert fixture.checkpoint_path.read_bytes() == payload
    receipt = yaml.safe_load((fixture.checkpoint_path.parent / "chat_eval/target_real/target.yaml").read_text())
    identity = receipt["seeds"]["adhoc"]["run_identity"]["checkpoint"]
    assert identity["metadata_sha256"] == hashlib.sha256(payload).hexdigest()
    assert identity["model_sha256"] == hashlib.sha256(fixture.checkpoint_path.with_name("model_001116.pt").read_bytes()).hexdigest()
    assert identity["metadata_sha256"] != hashlib.sha256(json.dumps(after).encode()).hexdigest()
    assert set(receipt["seeds"]["adhoc"]["results"]) == {"MMLU"}


@pytest.mark.parametrize("path", [("model_config", "sequence_len"), (ONLINE_SAE_LINEAGE_META_KEY, "selection_seed")])
def test_real_loader_non_null_mutation_rejects_before_evaluation(tmp_path, monkeypatch, path):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True, real_loader=True)
    original = chat_eval.load_model

    def changed_load(*args, **kwargs):
        model, tokenizer, meta = original(*args, **kwargs)
        meta[path[0]][path[1]] += 1
        return model, tokenizer, meta

    monkeypatch.setattr(chat_eval, "load_model", changed_load)
    monkeypatch.setattr(chat_eval, "_chat_eval_target_real_per_config", lambda **kw: pytest.fail("changed metadata reached evaluation"))
    with pytest.raises(ValueError, match=path[1]):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert fixture.loaded == [(fixture.checkpoint_path.parent, 1116)]
    assert not list(tmp_path.rglob("target.yaml"))


def snapshot_fixture(tmp_path, metadata):
    path = tmp_path / "meta_000100.json"
    payload = json.dumps(metadata).encode()
    path.write_bytes(payload)
    path.with_name("model_000100.pt").write_bytes(b"snapshot weights")
    snapshot = TurnChatEvalCheckpointSnapshot.capture(path, payload, 100)
    return snapshot, {**deepcopy(metadata), "checkpoint_dir": str(tmp_path)}


@pytest.mark.parametrize("before,after,path", [
    ({"width": 64}, {"width": 32}, "user_config.width"),
    ({"width": 64}, {}, "user_config.width"),
    ({"width": 64}, {"width": None}, "user_config.width"),
    ({"enabled": 1}, {"enabled": True}, "user_config.enabled"),
    ({"weight": 1.0}, {"weight": 1}, "user_config.weight"),
    ({"encoder": {}}, {"encoder": None}, "user_config.encoder"),
    ({"ids": [1, 2]}, {"ids": [2, 1]}, "user_config.ids[0]"),
    ({"ids": [1, 2]}, {"ids": [1, 2, 3]}, "user_config.ids"),
    ({"ids": [None]}, {"ids": [1]}, "user_config.ids[0]"),
    ({"optional": None}, {}, "user_config.optional"),
])
def test_non_contract_values_cannot_change_or_disappear(tmp_path, before, after, path):
    snapshot, loaded = snapshot_fixture(tmp_path, {
        ONLINE_SAE_LINEAGE_META_KEY: {"selection_seed": 19}, "user_config": before,
    })
    loaded["user_config"] = after
    with pytest.raises(ValueError, match=re.escape(f"metadata at {path}")):
        snapshot.bind_loaded(loaded)


@pytest.mark.parametrize("path,value", [
    (("sft_segment_mode",), SegmentMode.turn_delimited_fixed_tokens.value),
    ((ONLINE_SAE_LINEAGE_META_KEY,), {"selection_seed": 19}),
    (("selfcond_data",), {"sft_segment_mode": SegmentMode.turn_delimited_fixed_tokens.value}),
    (("turn_self_cond_segment_size_rustbpe",), 32),
    (("feature_vocab",), {"ids": [1, 2]}),
    (("feature_mask_canonical",), {}),
    (("feature_clusters_canonical",), {}),
    (("feature_clusters_descriptor",), {}),
    (("clustered_threshold_selection_semantics",), "changed"),
    (("model_config", "sequence_len"), 2048),
    (("num_buckets",), 4),
    (("n_sae_features",), 8),
])
@pytest.mark.parametrize("present", [False, True], ids=["missing", "null"])
def test_loader_cannot_fill_contract_fields(tmp_path, path, value, present):
    meta: dict = {ONLINE_SAE_LINEAGE_META_KEY: {"selection_seed": 19}}
    container = meta
    for key in path[:-1]:
        container[key] = {}
        container = container[key]
    if present:
        container[path[-1]] = None
    else:
        container.pop(path[-1], None)
    snapshot, loaded = snapshot_fixture(tmp_path, meta)
    container = loaded
    for key in path[:-1]:
        container = container[key]
    container[path[-1]] = value
    with pytest.raises(ValueError, match=re.escape("metadata at " + ".".join(path))):
        snapshot.bind_loaded(loaded)


def test_loader_cannot_extend_lineage_block(tmp_path):
    snapshot, loaded = snapshot_fixture(tmp_path, {ONLINE_SAE_LINEAGE_META_KEY: {"selection_seed": 19}})
    loaded[ONLINE_SAE_LINEAGE_META_KEY]["new_field"] = 1
    with pytest.raises(ValueError, match=re.escape(f"{ONLINE_SAE_LINEAGE_META_KEY}.new_field")):
        snapshot.bind_loaded(loaded)


@pytest.mark.parametrize("block", ["user_config", "pretrain_conditioning"])
def test_recovered_bucket_copy_cannot_change_contract(tmp_path, block):
    snapshot, loaded = snapshot_fixture(tmp_path, {
        ONLINE_SAE_LINEAGE_META_KEY: {"selection_seed": 19},
        "num_buckets": 4, block: {"num_buckets": None},
    })
    loaded[block]["num_buckets"] = 8
    with pytest.raises(ValueError, match=re.escape(f"{block}.num_buckets")):
        snapshot.bind_loaded(loaded)


def test_discovery_change_loads_initial_directory_and_step(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    initial = fixture.checkpoint_path.parent
    other = initial.parent.parent / "new-discovery"
    shutil.copytree(initial, other)
    fixture.cfg.model_tag = None
    fixture.cfg.step = None
    fixture.cfg.task_name = "MMLU"
    discoveries = []

    def discover(root):
        discoveries.append(root)
        return str(initial if len(discoveries) == 1 else other)

    monkeypatch.setattr(checkpoint_manager, "find_largest_model", discover)
    chat_eval.main.__wrapped__(fixture.cfg)
    assert len(discoveries) == 1
    assert fixture.loaded == [(initial, 1116)]
    receipt = yaml.safe_load((initial / "chat_eval/target_real/target.yaml").read_text())
    assert receipt["seeds"]["adhoc"]["run_identity"]["checkpoint"]["metadata_sha256"] == hashlib.sha256(
        fixture.checkpoint_path.read_bytes()).hexdigest()


@pytest.mark.parametrize("replace_file", [False, True])
def test_same_mode_loaded_metadata_change_rejects_before_evaluation(tmp_path, monkeypatch, replace_file):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    before = json.loads(fixture.checkpoint_path.read_bytes())
    before["step_time"] = 1.0
    fixture.checkpoint_path.write_text(json.dumps(before))
    original = chat_eval.load_model

    def changed_load(*args, **kwargs):
        if replace_file:
            changed = json.loads(fixture.checkpoint_path.read_bytes())
            changed["step_time"] = 123.0
            fixture.checkpoint_path.write_text(json.dumps(changed))
        model, tokenizer, meta = original(*args, **kwargs)
        meta["step_time"] = 123.0
        return model, tokenizer, meta

    monkeypatch.setattr(chat_eval, "load_model", changed_load)
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *a, **kw: pytest.fail("metadata race reached encoder"))
    with pytest.raises(ValueError, match="loaded checkpoint metadata differs"):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert not list(tmp_path.rglob("target.yaml"))


def test_same_mode_file_extension_still_rejects_before_evaluation(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True, real_loader=True)
    original = chat_eval.load_model

    def changed_load(*args, **kwargs):
        changed = json.loads(fixture.checkpoint_path.read_bytes())
        changed["new_diagnostic"] = 1
        fixture.checkpoint_path.write_text(json.dumps(changed))
        return original(*args, **kwargs)

    monkeypatch.setattr(chat_eval, "load_model", changed_load)
    monkeypatch.setattr(chat_eval, "_chat_eval_target_real_per_config", lambda **kw: pytest.fail("metadata race reached evaluation"))
    with pytest.raises(ValueError, match="checkpoint metadata changed between load and receipt"):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert not list(tmp_path.rglob("target.yaml"))


def test_weights_replaced_between_load_and_hash_reject_before_evaluation(tmp_path, monkeypatch):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    original = chat_eval.load_model
    weights = fixture.checkpoint_path.with_name("model_001116.pt")

    def changed_load(*args, **kwargs):
        result = original(*args, **kwargs)
        replacement = weights.with_suffix(".replacement")
        replacement.write_bytes(weights.read_bytes())
        replacement.replace(weights)  # Same bytes, different inode: still a snapshot change.
        return result

    monkeypatch.setattr(chat_eval, "load_model", changed_load)
    monkeypatch.setattr(fixture.model.encoder, "forward", lambda *a, **kw: pytest.fail("weight race reached encoder"))
    with pytest.raises(ValueError, match="checkpoint model changed between load and receipt"):
        chat_eval.main.__wrapped__(fixture.cfg)
    assert not list(tmp_path.rglob("target.yaml"))


@pytest.mark.parametrize("replacement", ["metadata", "weights", None])
def test_snapshot_fence_spans_evaluation_and_hashes_once(tmp_path, monkeypatch, replacement):
    fixture = turn_main_fixture(tmp_path, monkeypatch, clustered=True)
    fixture.cfg.task_name = "MMLU"
    fixture.cfg.target_real_configs.append({**dict(fixture.cfg.target_real_configs[0]), "name": "second"})
    original_digest = provenance._model_digest
    digests = []

    def digest(path):
        digests.append(path)
        return original_digest(path)

    original_eval = chat_eval._chat_eval_target_real_per_config

    def evaluate(**kwargs):
        result = original_eval(**kwargs)
        if replacement == "metadata":
            fixture.checkpoint_path.write_bytes(fixture.checkpoint_path.read_bytes() + b"\n")
        elif replacement == "weights":
            weights = fixture.checkpoint_path.with_name("model_001116.pt")
            weights.write_bytes(weights.read_bytes() + b"changed")
        return result

    monkeypatch.setattr(provenance, "_model_digest", digest)
    monkeypatch.setattr(chat_eval, "_chat_eval_target_real_per_config", evaluate)
    if replacement is None:
        chat_eval.main.__wrapped__(fixture.cfg)
        assert len(list(tmp_path.rglob("target.yaml"))) == 1
        assert len(list(tmp_path.rglob("second.yaml"))) == 1
    else:
        with pytest.raises(RuntimeError, match="failed during evaluation") as error:
            chat_eval.main.__wrapped__(fixture.cfg)
        assert "changed between load and receipt" in str(error.value.__cause__)
        assert not list(tmp_path.rglob("target.yaml"))
    assert digests == [fixture.checkpoint_path.with_name("model_001116.pt")]
