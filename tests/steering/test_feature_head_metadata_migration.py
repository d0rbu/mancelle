"""Explicit migration, strict schema, and architecture admission tests."""

from __future__ import annotations

import copy
import hashlib
import json
import subprocess
import sys

import pytest
import torch as t
from omegaconf import OmegaConf

from nanochat import flash_attention
from scripts.train import conditional_encdec_sft as sft
from scripts.train import conditional_encdec_train as pretrain
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    feature_prediction_config_from_metadata,
    parse_feature_prediction_config,
    serialize_feature_prediction_config,
    validate_feature_prediction_architecture,
    validate_feature_prediction_resume,
)
from steering.tools import migrate_feature_head_metadata as migration
from tests.steering.test_encdec_train_vocab_wiring import _cfg as _pretrain_cfg
from tests.steering.test_encdec_train_vocab_wiring import _shared as _pretrain_shared
from tests.steering.test_feature_head_checkpoint_depth import READERS, _metadata, _restore
from tests.steering.test_residual_feature_head_integration import _model
from tests.steering.test_transformer_feature_head import _transformer_model


@pytest.fixture(autouse=True)
def sdpa(monkeypatch):
    monkeypatch.setattr(flash_attention, "_override_impl", flash_attention.FlashAttnImpl.SDPA)


def _hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _legacy(tmp_path, depth=2):
    model = _model(depth=depth or 0)
    with t.no_grad():
        for name, parameter in model.named_parameters():
            if name.startswith("feature_pred_head.residual_blocks"):
                parameter.normal_(0, 0.03)
    meta = _metadata(model)
    for raw in (meta["user_config"]["feature_prediction"], meta["feature_prediction_config"]):
        raw.pop("block")
        raw.pop("head_layers")
        if depth is not None:
            raw["mlp_layers"] = depth
    meta["user_config"]["feature_prediction_mlp_layers"] = depth or 0
    meta["unrelated"] = {"nested": [1, "historical provenance", None]}
    source = tmp_path / "meta_000002.json"
    weights = tmp_path / "model_000002.pt"
    source.write_text(json.dumps(meta))
    t.save(model.state_dict(), weights)
    return model, meta, source, weights, tmp_path / "canonical.json"


@pytest.mark.parametrize("depth", [None, 0, 2])
@pytest.mark.parametrize("mirror", ["both", "user", "top"])
def test_conversion_changes_only_metadata_and_loads_exact_learned_tensors(tmp_path, monkeypatch, depth, mirror):
    model, meta, source, weights, output = _legacy(tmp_path, depth)
    if mirror == "user":
        meta.pop("feature_prediction_config")
    elif mirror == "top":
        meta["user_config"].pop("feature_prediction")
    source.write_text(json.dumps(meta))
    before = (_hash(source), _hash(weights))
    original_load = migration.t.load
    calls = []

    def load(*args, **kwargs):
        calls.append(kwargs)
        return original_load(*args, **kwargs)

    monkeypatch.setattr(migration.t, "load", load)
    converted = migration.migrate_feature_head_metadata(source, weights, output)
    assert calls == [{"map_location": "meta", "weights_only": True, "mmap": True}]
    assert before == (_hash(source), _hash(weights))
    assert json.loads(output.read_text()) == converted
    assert converted["unrelated"] == meta["unrelated"]
    assert converted["feature_prediction_config"] == converted["user_config"]["feature_prediction"]
    assert converted["user_config"]["feature_prediction_block"] == "mlp"
    assert converted["user_config"]["feature_prediction_head_layers"] == (depth or 0)
    assert "feature_prediction_mlp_layers" not in converted["user_config"]
    hidden = t.randn(1, 5, model.dec_config.n_embd)
    for reader in READERS:
        restored = _restore(model, converted, model.state_dict(), reader)
        assert t.equal(restored.apply_feature_prediction_head(hidden), model.apply_feature_prediction_head(hidden))
        assert all(t.equal(tensor, restored.state_dict()[name]) for name, tensor in model.state_dict().items())


@pytest.mark.parametrize("corruption", ["missing_depth", "wrong_depth", "wrong_shape", "missing_bias", "unexpected_gate", "conflict", "partial", "mixed", "bad_override", "null_mirror"])
def test_conversion_rejects_metadata_and_tensor_disagreements_without_writes(tmp_path, corruption):
    model, meta, source, weights, output = _legacy(tmp_path)
    state = model.state_dict()
    if corruption == "missing_depth":
        for raw in (meta["feature_prediction_config"], meta["user_config"]["feature_prediction"]):
            raw.pop("mlp_layers")
    elif corruption == "wrong_depth":
        for raw in (meta["feature_prediction_config"], meta["user_config"]["feature_prediction"]):
            raw["mlp_layers"] = 3
    elif corruption == "wrong_shape":
        state["feature_pred_head.residual_blocks.1.c_fc.weight"] = t.zeros(1, 1)
    elif corruption == "missing_bias":
        state.pop("feature_pred_head.bias")
    elif corruption == "unexpected_gate":
        state["feature_pred_head.residual_blocks.0.attn.ve_gate.weight"] = t.zeros(1, 1)
    elif corruption == "conflict":
        meta["feature_prediction_config"]["mlp_layers"] = 1
    elif corruption in ("partial", "mixed"):
        meta["feature_prediction_config"]["block"] = "mlp"
        if corruption == "mixed":
            meta["feature_prediction_config"]["head_layers"] = 2
    elif corruption == "bad_override":
        meta["user_config"]["feature_prediction_mlp_layers"] = 1
    elif corruption == "null_mirror":
        meta["feature_prediction_config"] = None
    source.write_text(json.dumps(meta))
    t.save(state, weights)
    before = (_hash(source), _hash(weights))
    with pytest.raises(ValueError):
        migration.migrate_feature_head_metadata(source, weights, output)
    assert not output.exists() and before == (_hash(source), _hash(weights))


def test_conversion_refuses_existing_destination_and_racing_writer(tmp_path, monkeypatch):
    _, _, source, weights, output = _legacy(tmp_path)
    output.write_text("original")
    with pytest.raises(FileExistsError):
        migration.migrate_feature_head_metadata(source, weights, output)
    output.unlink()
    original_link = migration.os.link

    def competing_writer(src, dst):
        output.write_text("other writer")
        original_link(src, dst)

    monkeypatch.setattr(migration.os, "link", competing_writer)
    with pytest.raises(FileExistsError):
        migration.migrate_feature_head_metadata(source, weights, output)
    assert output.read_text() == "other writer"
    assert not list(tmp_path.glob(".canonical.json.*"))


@pytest.mark.parametrize("corruption", [
    "non_mapping_state", "non_string_key", "duplicate_head_key", "non_tensor_head",
    "missing_projection", "empty_projection", "decoder_width", "non_mapping_metadata",
    "non_mapping_user", "headless", "canonical_override", "integer_tensor",
])
def test_conversion_rejects_invalid_checkpoint_envelopes_and_tensor_metadata(tmp_path, corruption):
    model, meta, source, weights, output = _legacy(tmp_path)
    state = dict(model.state_dict())
    if corruption == "non_mapping_state":
        state = list(state.values())
    elif corruption == "non_string_key":
        state[123] = t.ones(1)
    elif corruption == "duplicate_head_key":
        state["_orig_mod.feature_pred_head.weight"] = state["feature_pred_head.weight"]
    elif corruption == "non_tensor_head":
        state["feature_pred_head.weight"] = "invalid"
    elif corruption == "missing_projection":
        state.pop("feature_pred_head.weight")
    elif corruption == "empty_projection":
        state["feature_pred_head.weight"] = t.empty(0, model.dec_config.n_embd)
    elif corruption == "decoder_width":
        meta["model_config"]["n_embd"] += 1
    elif corruption == "non_mapping_metadata":
        meta = []
    elif corruption == "non_mapping_user":
        meta["user_config"] = []
    elif corruption == "headless":
        meta = {"user_config": {"feature_prediction": None}, "feature_prediction_config": None}
        state = {}
    elif corruption == "canonical_override":
        meta["user_config"]["feature_prediction_head_layers"] = 2
    elif corruption == "integer_tensor":
        state["feature_pred_head.bias"] = state["feature_pred_head.bias"].long()
    source.write_text(json.dumps(meta))
    t.save(state, weights)
    before = (_hash(source), _hash(weights))
    with pytest.raises(ValueError):
        migration.migrate_feature_head_metadata(source, weights, output)
    assert not output.exists() and before == (_hash(source), _hash(weights))


def test_conversion_accepts_compiled_keys_without_inventing_sft_overrides(tmp_path):
    model, meta, source, weights, output = _legacy(tmp_path)
    meta["user_config"].pop("feature_prediction_mlp_layers")
    source.write_text(json.dumps(meta))
    t.save({f"_orig_mod.{key}": value for key, value in model.state_dict().items()}, weights)
    converted = migration.migrate_feature_head_metadata(source, weights, output)
    assert "feature_prediction_head_layers" not in converted["user_config"]
    assert feature_prediction_config_from_metadata(converted) == model.feature_pred_config


def test_conversion_preserves_the_legacy_canonical_readers_missing_loss_mode_default(tmp_path):
    _, meta, source, weights, output = _legacy(tmp_path)
    meta["feature_prediction_config"].pop("loss_mode")
    meta["user_config"]["feature_prediction"].pop("loss_mode")
    source.write_text(json.dumps(meta))
    converted = migration.migrate_feature_head_metadata(source, weights, output)
    assert converted["feature_prediction_config"]["loss_mode"] == "best_token"
    assert converted["user_config"]["feature_prediction"] == converted["feature_prediction_config"]


@pytest.mark.parametrize("metadata", [
    {"user_config": None}, {"feature_prediction_config": []},
    {"user_config": {"feature_prediction": "invalid"}},
])
def test_canonical_parser_rejects_non_mapping_envelopes(metadata):
    with pytest.raises(ValueError, match="mapping"):
        feature_prediction_config_from_metadata(metadata)


@pytest.mark.parametrize("reader", READERS)
@pytest.mark.parametrize("corruption", ["legacy", "missing_block", "missing_depth", "mixed", "null_block", "bad_block"])
def test_runtime_never_infers_legacy_architecture(reader, corruption):
    raw = serialize_feature_prediction_config(FeaturePredictionConfig())
    assert raw is not None
    if corruption == "legacy":
        raw.pop("block")
        raw["mlp_layers"] = raw.pop("head_layers")
    elif corruption == "missing_block":
        raw.pop("block")
    elif corruption == "missing_depth":
        raw.pop("head_layers")
    elif corruption == "mixed":
        raw["mlp_layers"] = 0
    else:
        raw["block"] = None if corruption == "null_block" else "attention"
    with pytest.raises(ValueError):
        reader({"feature_prediction_config": raw})


@pytest.mark.parametrize("saved_block", ["mlp", "transformer"])
@pytest.mark.parametrize("requested_block", ["mlp", "transformer"])
@pytest.mark.parametrize("saved_depth", [0, 2])
@pytest.mark.parametrize("requested_depth", [0, 2, 4, 6, 8, 10])
@pytest.mark.parametrize("resuming", [False, True])
def test_architecture_transitions_include_zero_depth_block_identity(saved_block, requested_block, saved_depth, requested_depth, resuming):
    saved = FeaturePredictionConfig(block=saved_block, head_layers=saved_depth)
    permitted = (saved_block, saved_depth) == (requested_block, requested_depth) or (not resuming and saved_depth == 0)
    if permitted:
        validate_feature_prediction_architecture(saved, requested_block, requested_depth, resuming=resuming)
    else:
        with pytest.raises(ValueError, match="does not match checkpoint"):
            validate_feature_prediction_architecture(saved, requested_block, requested_depth, resuming=resuming)


@pytest.mark.parametrize("block,depth", [("mlp", 0), ("mlp", 2), ("transformer", 0), ("transformer", 3)])
def test_checkpoint_roundtrip_and_every_resume_guard_preserve_effective_architecture(block, depth):
    model = _transformer_model(depth) if block == "transformer" and depth else _model(depth=depth)
    model.feature_pred_config = replace_config = FeaturePredictionConfig(block=block, head_layers=depth)
    meta = _metadata(model)
    assert feature_prediction_config_from_metadata(meta) == replace_config
    validate_feature_prediction_resume(meta, replace_config)
    changed = copy.deepcopy(meta)
    for raw in (changed["feature_prediction_config"], changed["user_config"]["feature_prediction"]):
        raw["block"] = "mlp" if block == "transformer" else "transformer"
    with pytest.raises(ValueError, match="does not match checkpoint"):
        validate_feature_prediction_resume(changed, replace_config)
    with pytest.raises(ValueError, match="presence"):
        validate_feature_prediction_resume(meta, None)
    hidden = t.randn(1, 7, model.dec_config.n_embd)
    for reader in READERS:
        restored = _restore(model, meta, model.state_dict(), reader)
        assert t.equal(restored.apply_feature_prediction_head(hidden), model.apply_feature_prediction_head(hidden))


@pytest.mark.parametrize("override", ["+feature_prediction_mlp_layers=2", "+feature_prediction={mlp_layers:2}"])
def test_removed_hydra_keys_fail_before_any_compute_or_checkpoint_load(monkeypatch, override):
    monkeypatch.setattr(sft.sys, "argv", ["sft", override])
    monkeypatch.setattr(sft, "compute_init", lambda *_: pytest.fail("removed key reached compute initialization"))
    with pytest.raises(ValueError, match="removed"):
        sft.main()


def test_fresh_defaults_do_not_apply_to_checkpoint_parsing():
    assert parse_feature_prediction_config({}) == FeaturePredictionConfig()
    with pytest.raises(ValueError, match="explicit block and head_layers"):
        parse_feature_prediction_config({}, checkpoint=True)
    assert feature_prediction_config_from_metadata({}) is None


@pytest.mark.parametrize("depth", [0, 2])
@pytest.mark.parametrize("schema,block", [
    ("canonical", "transformer"), ("legacy", "transformer"),
    ("wrong_block", "transformer"), ("wrong_depth", "transformer"),
    ("headless", "transformer"), ("converted", "mlp"),
])
def test_real_pretraining_resume_checks_architecture_before_applying_state(
    tmp_path, monkeypatch, hydra_compose, depth, schema, block,
):
    cfg = _pretrain_cfg(hydra_compose, None)
    for key, value in {
        "depth": 2, "aspect_ratio": 32, "head_dim": 32, "max_seq_len": 16,
        "resume_from_step": 1,
        "feature_prediction": {"block": block, "head_layers": depth},
    }.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    model = pretrain.build_encdec_meta_model_from_training_cfg(
        cfg, base_vocab=128, n_sae_features=64, feature_vocab=None,
    )
    model.to_empty(device="cpu")
    model.init_weights()
    shared = _pretrain_shared(model, cfg, str(tmp_path))
    meta = _metadata(model)
    for raw in (meta["feature_prediction_config"], meta["user_config"]["feature_prediction"]):
        if schema in ("legacy", "converted"):
            raw.pop("block")
            raw["mlp_layers"] = raw.pop("head_layers")
        elif schema == "wrong_block":
            raw["block"] = "mlp"
        elif schema == "wrong_depth":
            raw["head_layers"] = depth + 1
    if schema == "headless":
        meta["feature_prediction_config"] = meta["user_config"]["feature_prediction"] = None
    before = copy.deepcopy(model.state_dict())
    parameters = dict(model.named_parameters())
    saved = copy.deepcopy(before)
    saved["feature_pred_head.bias"].fill_(17)
    if schema == "converted":
        source = tmp_path / "legacy.json"
        weights = tmp_path / "model.pt"
        source.write_text(json.dumps(meta))
        t.save(saved, weights)
        meta = migration.migrate_feature_head_metadata(source, weights, tmp_path / "canonical.json")
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path / "base"))
    monkeypatch.setattr(pretrain, "load_checkpoint", lambda *_args, **_kwargs: (saved, {}, meta))
    original_load = model.load_state_dict
    applied = []

    def apply_state(*args, **kwargs):
        applied.append(True)
        original_load(*args, **kwargs)
        raise RuntimeError("accepted resume reached model state application")

    monkeypatch.setattr(model, "load_state_dict", apply_state)
    monkeypatch.setattr(model, "setup_optimizer", lambda *_args, **_kwargs: pytest.fail("unexpected optimizer application"))
    if schema in ("canonical", "converted"):
        with pytest.raises(RuntimeError, match="accepted resume"):
            pretrain.train_one_config(cfg, shared, None)
        assert applied and t.equal(model.state_dict()["feature_pred_head.bias"], saved["feature_pred_head.bias"])
    else:
        with pytest.raises(ValueError, match="removed|does not match checkpoint|presence"):
            pretrain.train_one_config(cfg, shared, None)
        assert not applied
        assert all(parameter is parameters[name] for name, parameter in model.named_parameters())
        assert all(t.equal(value, before[name]) for name, value in model.state_dict().items())


def test_real_hydra_migration_cli(tmp_path):
    _, _, source, weights, output = _legacy(tmp_path)
    result = subprocess.run([
        sys.executable, "-m", "steering.tools.migrate_feature_head_metadata",
        f"source_metadata={source}", f"weights={weights}", f"output_metadata={output}",
    ], text=True, capture_output=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr
    restored = feature_prediction_config_from_metadata(json.loads(output.read_text()))
    assert restored is not None and restored.head_layers == 2
