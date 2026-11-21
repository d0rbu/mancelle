"""Exercise cooldown through the real pretraining loop and checkpoint writer."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from omegaconf import DictConfig, OmegaConf

import nanochat.flash_attention as attention
import scripts.train.conditional_encdec_train as pretrain
from nanochat.common import DummyWandb
from nanochat.flash_attention import FlashAttnImpl
from tests.steering.selfcond.helpers import make_batch
from tests.steering.test_encdec_train_vocab_wiring import _cfg, _shared


def _runtime(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hydra_compose,
    *,
    splice_junctions: t.Tensor | None = None,
):
    monkeypatch.setenv("NANOCHAT_BASE_DIR", str(tmp_path))
    monkeypatch.setattr(attention, "_override_impl", FlashAttnImpl.SDPA)
    monkeypatch.setattr(
        pretrain, "get_report", lambda **_kwargs: SimpleNamespace(log=lambda **_fields: None)
    )
    cfg = _cfg(hydra_compose, None)
    for key, value in {
        "model_tag": "cooldown-test",
        "depth": 2,
        "aspect_ratio": 16,
        "head_dim": 16,
        "window_pattern": "L",
        "encoder": {"n_embd": 32, "depth": 1, "n_head": 2, "n_kv_head": 2, "head_dim": 16},
        "cross_attention.frequency": 1,
        "max_seq_len": 8,
        "device_batch_size": 2,
        "total_batch_size": 32,
        "num_iterations": 5,
        "eval_every": 0,
        "core_metric_every": 0,
        "sample_every": 0,
        "steering_eval_every": 0,
        "save_every": 1,
        "compile": False,
        "feature_delta": None,
        "feature_selection.delta_weights": None,
        "feature_prediction": None,
        "initialize_encoder_embeddings_from_sae": None,
        "conditioning_cooldown": {
            "enabled": True,
            "start_fraction": 1.0,
            "end_fraction": 0.0,
            "start_step": 1,
            "end_step": 3,
            "cooldown_tokens": 20_000_000_000,
            "granularity": "segment",
        },
    }.items():
        OmegaConf.update(cfg, key, value, merge=False, force_add=True)
    model = pretrain.build_encdec_meta_model_from_training_cfg(
        cfg, base_vocab=128, n_sae_features=64, feature_vocab=None
    )
    model.to_empty(device="cpu")
    shared = _shared(model, cfg, str(tmp_path))
    shared.per_feature_stats = {}
    shared.feature_index = SimpleNamespace()  # ty: ignore[invalid-assignment]
    batch = make_batch([[2, 2], [1, 3]], T=8, F=2, n_sae=64, seed=19)
    cursors: list[Any] = []
    fetched: list[int] = []
    masks: list[t.Tensor] = []

    def batches(**kwargs):
        assert kwargs["label_dropout"] == 0.0
        cursor = kwargs["resume_state_dict"]
        cursors.append(copy.deepcopy(cursor))
        offset = 0 if cursor is None else cursor["batch"]
        while True:
            fetched.append(offset)
            state = {"epoch": 1, "batch": offset, "ddp_world_size": 1}
            if kwargs.get("include_splice_mask", False):
                state["conditioning_splice_identity"] = None
                state["conditioning_splice_mask"] = (
                    splice_junctions
                    if splice_junctions is not None
                    else t.zeros_like(batch.seg_doc_id, dtype=t.bool)
                ).tolist()
            yield (
                batch.inputs,
                batch.targets,
                batch.enc_fids,
                batch.enc_mags,
                batch.enc_mask,
                batch.enc_pcts,
                batch.seg_map,
                batch.seg_doc_id,
                None,
                state,
            )
            offset += 1

    def capture_mask(_model, args):
        masks.append(args[4].clone())

    monkeypatch.setattr(pretrain, "encoder_decoder_data_loader_with_state", batches)
    model.register_forward_pre_hook(capture_mask)
    return cfg, shared, batch, cursors, fetched, masks


def _run(cfg: DictConfig, shared: pretrain.SharedState) -> None:
    with t.compiler.set_stance("force_eager"):
        pretrain.train_one_config(cfg, shared, DummyWandb())


def _assert_tree_equal(actual: Any, expected: Any) -> None:
    if isinstance(expected, t.Tensor):
        assert isinstance(actual, t.Tensor)
        assert actual.dtype == expected.dtype
        assert t.equal(actual, expected)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_tree_equal(actual[key], expected[key])
    elif isinstance(expected, (list, tuple)):
        assert type(actual) is type(expected)
        assert len(actual) == len(expected)
        for left, right in zip(actual, expected, strict=True):
            _assert_tree_equal(left, right)
    else:
        assert actual == expected


@pytest.mark.parametrize("granularity", ["segment", "document"])
def test_actual_loop_uses_update_schedule_for_every_accumulated_microbatch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose, granularity: str
) -> None:
    cfg, shared, batch, cursors, fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    cfg.conditioning_cooldown.granularity = granularity
    pristine = batch.enc_mask.clone()
    _run(cfg, shared)

    # Five updates, two accumulated microbatches each; update 1 starts at 100%,
    # update 2 is 50%, update 3 reaches zero, update 4 holds zero.
    assert len(masks) == 10
    assert all(t.equal(mask, pristine) for mask in masks[:4])
    assert any(mask.any() for mask in masks[4:6])
    assert any(not t.equal(mask, pristine) for mask in masks[4:6])
    assert all(not mask.any() for mask in masks[6:])
    assert t.equal(batch.enc_mask, pristine), "the consumer mutated a reusable loader tensor"
    assert cursors == [None]
    assert fetched == list(range(11)), "the last pending batch remains unconsumed"
    if granularity == "document":
        for mask in masks:
            for row in range(2):
                for document in range(2):
                    selected = batch.seg_doc_id[row] == document
                    decisions = mask[row, selected].any(dim=-1)
                    assert t.equal(decisions, decisions[:1].expand_as(decisions))

    (metadata_path,) = tmp_path.rglob("meta_000005.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["conditioning_cooldown"]["start_step"] == 1
    assert metadata["conditioning_cooldown"]["end_step"] == 3
    assert metadata["conditioning_cooldown"]["granularity"] == granularity
    assert metadata["dataloader_state_dict"]["rank_states"][0]["batch"] == 10
    assert all(t.isfinite(parameter).all() for parameter in shared.orig_model.parameters())


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("horizon", ["explicit", "flops", "ratio"])
def test_actual_loop_resolves_auto_batch_and_derived_horizons_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    hydra_compose,
    enabled: bool,
    horizon: str,
) -> None:
    cfg, shared, _batch, _cursors, fetched, masks = _runtime(
        tmp_path, monkeypatch, hydra_compose
    )
    cfg.conditioning_cooldown.enabled = enabled
    cfg.total_batch_size = -1
    # A reference token count equal to the configured target makes the auto
    # batch exactly B_REF. The ratio horizon also exercises a partial last batch.
    target_tokens = 176 if horizon == "ratio" else 240
    scaling_params = pretrain._scaling_parameter_count(shared.model)
    cfg.target_param_data_ratio = target_tokens / scaling_params
    shared.d_ref = float(int(cfg.target_param_data_ratio * scaling_params))
    shared.b_ref = 32
    feature_selection = pretrain.parse_feature_selection(
        pretrain._cfg_dict(cfg.feature_selection, name="feature_selection")
    )
    negative_selection = pretrain.parse_negative_selection(
        pretrain._optional_cfg_dict(cfg, "negative_selection")
    )
    avg_features = pretrain.avg_features_per_segment_estimate(feature_selection, negative_selection)
    original_estimate = shared.orig_model.estimate_flops
    flops = original_estimate(avg_features, cfg.segment_size)
    cfg.num_iterations = 5 if horizon == "explicit" else -1
    cfg.target_flops = -1 if horizon == "ratio" else (9 if horizon == "explicit" else 5) * flops * 32
    calls = []

    def estimate_once(features, segment_size):
        calls.append((features, segment_size))
        return original_estimate(features, segment_size)

    monkeypatch.setattr(shared.orig_model, "estimate_flops", estimate_once)
    _run(cfg, shared)

    assert calls == [(avg_features, cfg.segment_size)]
    assert len(masks) == 10  # Five updates, two 16-token microbatches per update.
    assert fetched == list(range(11))
    (metadata_path,) = tmp_path.rglob("meta_000005.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["step"] == 5
    if enabled:
        assert metadata["conditioning_cooldown"]["num_iterations"] == 5
        assert metadata["conditioning_cooldown"]["total_batch_size"] == 32
    else:
        assert "conditioning_cooldown" not in metadata


def test_actual_checkpoint_resume_reproduces_masks_weights_and_optimizer_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose
) -> None:
    cfg, shared, _batch, cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    _run(cfg, shared)
    expected_masks = [mask.clone() for mask in masks[4:]]
    expected_weights = copy.deepcopy(shared.orig_model.state_dict())
    (checkpoint_path,) = tmp_path.rglob("meta_000002.json")
    source_bytes = checkpoint_path.read_bytes()
    optimizer_path = checkpoint_path.with_name("optim_000005_rank0.pt")
    expected_optimizer = t.load(optimizer_path, map_location="cpu", weights_only=False)
    masks.clear()
    with t.no_grad():
        for parameter in shared.orig_model.parameters():
            parameter.fill_(float("nan"))
    t.manual_seed(999_123)
    t.rand(200)
    cfg.resume_from_step = 2
    _run(cfg, shared)
    assert checkpoint_path.read_bytes() == source_bytes
    assert cursors[-1]["batch"] == 4
    _assert_tree_equal(masks, expected_masks)
    _assert_tree_equal(shared.orig_model.state_dict(), expected_weights)
    _assert_tree_equal(
        t.load(optimizer_path, map_location="cpu", weights_only=False), expected_optimizer
    )


@pytest.mark.parametrize("granularity", ["segment", "document"])
def test_actual_loop_drops_junctions_last_and_documents_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose, granularity: str
) -> None:
    junctions = t.tensor([[False, False, True, False], [False, True, False, False]])
    cfg, shared, batch, _cursors, _fetched, masks = _runtime(
        tmp_path, monkeypatch, hydra_compose, splice_junctions=junctions
    )
    cfg.conditioning_cooldown.granularity = granularity
    cfg.conditioning_cooldown.end_fraction = 0.125
    _run(cfg, shared)
    # The 0.5625 midpoint can retain all protected units. By 0.125 the
    # unprotected tier is fully dropped, and only protected units may remain.
    protected = junctions.clone()
    if granularity == "document":
        for row in range(2):
            for segment in junctions[row].nonzero().flatten().tolist():
                protected[row] |= batch.seg_doc_id[row] == batch.seg_doc_id[row, segment]
    for mask in masks[4:6]:
        assert t.equal(mask[protected], batch.enc_mask[protected])
    for mask in masks[6:]:
        assert not mask[~protected].any()
    if granularity == "document":
        for mask in masks:
            for row in range(2):
                for doc in (0, 1):
                    decisions = mask[row, batch.seg_doc_id[row] == doc].any(dim=-1)
                    assert t.equal(decisions, decisions[:1].expand_as(decisions))
    (metadata_path,) = tmp_path.rglob("meta_000005.json")
    metadata = json.loads(metadata_path.read_text())
    assert metadata["conditioning_cooldown"]["splice_policy"] == "junctions_last"
    assert (
        metadata["dataloader_state_dict"]["rank_states"][0]["conditioning_splice_mask"]
        == junctions.tolist()
    )


def test_legacy_checkpoint_can_start_cooldown_without_overwriting_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose
) -> None:
    cfg, shared, _batch, cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    cfg.conditioning_cooldown.enabled = False
    cfg.num_iterations = 2
    _run(cfg, shared)
    (checkpoint_path,) = tmp_path.rglob("meta_000002.json")
    before = {path: path.read_bytes() for path in checkpoint_path.parent.glob("*000002*")}
    assert "conditioning_cooldown" not in json.loads(checkpoint_path.read_text())
    cfg.conditioning_cooldown.enabled = True
    cfg.conditioning_cooldown.start_step = 2
    cfg.conditioning_cooldown.end_step = 4
    cfg.num_iterations = 5
    cfg.resume_from_step = 2
    masks.clear()
    _run(cfg, shared)
    assert cursors[-1]["batch"] == 4
    assert all(path.read_bytes() == data for path, data in before.items())
    assert len(masks) == 6
    assert masks[0].any() and masks[1].any()
    assert not masks[-1].any() and not masks[-2].any()


def test_disabled_cooldown_matches_absent_config_exactly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose
) -> None:
    cfg, shared, _batch, _cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    cfg.conditioning_cooldown.enabled = False
    cfg.num_iterations = 2
    _run(cfg, shared)
    first_weights = copy.deepcopy(shared.orig_model.state_dict())
    first_masks = [mask.clone() for mask in masks]
    first_rng = t.get_rng_state().clone()
    (metadata_path,) = tmp_path.rglob("meta_000002.json")
    first_metadata = json.loads(metadata_path.read_text())
    assert "conditioning_cooldown" not in first_metadata["user_config"]
    OmegaConf.set_struct(cfg, False)
    del cfg.conditioning_cooldown
    masks.clear()
    _run(cfg, shared)
    _assert_tree_equal(shared.orig_model.state_dict(), first_weights)
    _assert_tree_equal(masks, first_masks)
    assert t.equal(t.get_rng_state(), first_rng)
    last_metadata = json.loads(metadata_path.read_text())
    assert last_metadata["user_config"] == first_metadata["user_config"]
    assert last_metadata["dataloader_state_dict"] == first_metadata["dataloader_state_dict"]


def test_completed_cooldown_checkpoint_fails_before_loading_batches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose
) -> None:
    cfg, shared, _batch, cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    _run(cfg, shared)
    weights = copy.deepcopy(shared.orig_model.state_dict())
    cfg.resume_from_step = cfg.num_iterations
    masks.clear()
    with pytest.raises(ValueError, match="no remaining.*updates"):
        _run(cfg, shared)
    assert cursors == [None]
    assert masks == []
    _assert_tree_equal(shared.orig_model.state_dict(), weights)


def test_active_resume_requires_splice_identity_before_model_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose
) -> None:
    cfg, shared, _batch, cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    _run(cfg, shared)
    (checkpoint_path,) = tmp_path.rglob("meta_000002.json")
    metadata = json.loads(checkpoint_path.read_text())
    del metadata["dataloader_state_dict"]["rank_states"][0]["conditioning_splice_identity"]
    checkpoint_path.write_text(json.dumps(metadata))
    with t.no_grad():
        for parameter in shared.orig_model.parameters():
            parameter.fill_(71.0)
    weights = copy.deepcopy(shared.orig_model.state_dict())
    cfg.resume_from_step = 2
    masks.clear()
    with pytest.raises(ValueError, match="conditioning_splice_identity"):
        _run(cfg, shared)
    assert cursors == [None]
    assert masks == []
    _assert_tree_equal(shared.orig_model.state_dict(), weights)


def test_real_validation_remains_fully_conditioned_and_does_not_change_training(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose
) -> None:
    cfg, shared, batch, _cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    _run(cfg, shared)
    expected_weights = copy.deepcopy(shared.orig_model.state_dict())
    expected_train_masks = [mask.clone() for mask in masks]
    masks.clear()
    observed_training: list[t.Tensor] = []
    observed_validation: list[t.Tensor] = []

    def capture_by_mode(model, args):
        destination = observed_training if model.training else observed_validation
        destination.append(args[4].clone())

    shared.orig_model.register_forward_pre_hook(capture_by_mode)
    shared.token_bytes = t.ones(128, dtype=t.int64)
    cfg.eval_every = 1
    cfg.eval_tokens = 16
    _run(cfg, shared)
    assert len(observed_validation) == 6  # step zero, each intermediate save, final
    assert all(t.equal(mask, batch.enc_mask) for mask in observed_validation)
    _assert_tree_equal(observed_training, expected_train_masks)
    _assert_tree_equal(shared.orig_model.state_dict(), expected_weights)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("device_batch_size", 1),
        ("total_batch_size", 64),
        ("seed", 43),
        ("conditioning_cooldown.enabled", False),
        ("conditioning_cooldown.end_fraction", 0.25),
    ],
)
def test_actual_resume_rejects_changed_schedule_or_geometry_before_model_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose, key: str, value: Any
) -> None:
    cfg, shared, _batch, cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    _run(cfg, shared)
    with t.no_grad():
        for parameter in shared.orig_model.parameters():
            parameter.fill_(71.0)
    weights = copy.deepcopy(shared.orig_model.state_dict())
    cfg.resume_from_step = 2
    OmegaConf.update(cfg, key, value, merge=False)
    masks.clear()
    with pytest.raises(ValueError, match="preserve the resolved schedule"):
        _run(cfg, shared)
    assert cursors == [None]
    assert masks == []
    _assert_tree_equal(shared.orig_model.state_dict(), weights)


@pytest.mark.parametrize("malformed", [None, [[0, 0, 0, 0], [0, 0, 0, 0]], [[False]]])
def test_actual_loop_rejects_missing_or_malformed_splice_flags_before_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, hydra_compose, malformed: Any
) -> None:
    cfg, shared, _batch, _cursors, _fetched, masks = _runtime(tmp_path, monkeypatch, hydra_compose)
    real_batches = pretrain.encoder_decoder_data_loader_with_state

    def malformed_batches(**kwargs):
        for batch in real_batches(**kwargs):
            state = dict(batch[9])
            if malformed is None:
                del state["conditioning_splice_mask"]
            else:
                state["conditioning_splice_mask"] = malformed
            yield (*batch[:9], state)

    monkeypatch.setattr(pretrain, "encoder_decoder_data_loader_with_state", malformed_batches)
    with pytest.raises(ValueError, match="splice_mask"):
        _run(cfg, shared)
    assert masks == []
