"""T-S2/T-S3: startup validation and checkpoint-identity precedence."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import scripts.train.conditional_encdec_selfcond as conditional_encdec_selfcond_module
from errors import FeatureVocabError, SelfCondTrainingError
from scripts.train.conditional_encdec_selfcond import (
    _TrainLoaderLookahead,
    assert_base_dir_has_checkpoint,
    assert_checkpoint_feature_universe,
    assert_resume_selfcond_matches,
    assert_training_window,
    explicit_override_roots,
    resolve_adam_betas,
    resolve_checkpoint_num_buckets,
    resolve_dataloader_resume,
    resolve_identity_config,
    resolve_selfcond_train_config,
    resolve_val_dirs,
    validate_selfcond_startup,
)
from scripts.train.conditional_encdec_selfcond import (
    resolve_grad_accum_steps as _resolve_grad_accum_steps,
)
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    SelfCondDataSource,
    SelfCondTrainConfig,
    SelfCondTrainScope,
)
from steering.conditional.feature_vocab import FeatureVocab
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.selfcond.helpers import tiny_head_model


def resolve_grad_accum_steps(*args: Any, **kwargs: Any) -> Any:
    """Pass raw startup values through the production beartype boundary."""
    return _resolve_grad_accum_steps(*args, **kwargs)


class _LoggerSpy:
    def __init__(self) -> None:
        self.warnings: list[str] = []

    def warning(self, message: str) -> None:
        self.warnings.append(message)


def _meta() -> dict:
    return {
        "user_config": {
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "max_pool",
                "loss_fn": "bce",
            },
            "feature_dir": "/checkpoint/features",
            "feature_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "min_k": 2,
                "max_k": 4,
            },
            "negative_selection": {
                "distribution": "geometric",
                "mean": 2.0,
                "std": 2.0,
                "min_k": 1,
                "max_k": 3,
            },
            "seed": 42,
            "segment_size": 128,
            "sae": {"width": "16k", "layer": 8},
            "num_buckets": 2,
            "pooling": "max",
            "dataset_phase": "train",
            "feature_mask": None,
            "feature_clusters": None,
            "feature_delta": None,
            "label_dropout": 0.0,
            "next_segment_conditioning_offset": False,
            "use_precomputed_batches": True,
        }
    }


def _sft_shaped_meta() -> dict:
    """Meta as written by conditional_encdec_sft's checkpoint saver."""
    return {
        "step": 500,
        "dataset": "safety",
        "sft_loss_mask": "full_sequence",
        "missing_percentile_policy": "error",
        "model_config": {"sequence_len": 64},
        "encoder_config": {"depth": 1},
        "xattn_config": {"frequency": 1},
        "num_buckets": 2,
        "feature_prediction_config": {
            "block": "mlp",
            "head_layers": 0,
            "weight": 0.1,
            "layer": -1,
            "loss_mode": "max_pool",
            "loss_fn": "bce",
        },
        "pretrain_conditioning": {
            "negative_selection": {
                "distribution": "geometric",
                "mean": 12.0,
                "std": 12.0,
                "min_k": 0,
                "max_k": 49,
            },
            "feature_delta": None,
            "delta_weights": None,
        },
        "loop_state": {
            "sft_cursor_semantics": "committed_v1",
            "sft_consumed": 40,
            "smooth_train_loss": 1.0,
            "smooth_fp_loss": 0.0,
            "min_val_bpb": 2.0,
            "total_training_time": 1.0,
            "sft_artifact_coverage": {"dataset_size": 20},
        },
        "user_config": {
            "num_buckets": None,
            "feature_dir": (
                "/data/f/s128/l8/a_residual/p_max/f16k/"
                "m_gemma-2-2b/safety_sft"
            ),
            "feature_selection": {
                "distribution": "geometric",
                "mean": 10.0,
                "std": 10.0,
                "min_k": 1,
                "max_k": 40,
                "mode": "top_k_percentile",
                "delta_weights": None,
            },
            "segment_size": 128,
            "sae": {
                "release": "gemmascope",
                "model": "gemma-2-2b",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
            },
            "sft_loss_mask": "full_sequence",
            "min_loss_tokens": 8,
            "feature_source_bos_char_len": 5,
            "alignment_probe_samples": 8,
            "encoder_embedding_lr": 0.2,
            "encoder_matrix_lr": 0.02,
            "xattn_matrix_lr": 0.02,
            "feature_prediction": {
                "block": "mlp",
                "head_layers": 0,
                "weight": 0.1,
                "layer": -1,
                "loss_mode": "max_pool",
                "loss_fn": "bce",
            },
        },
    }


def _config() -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=FeaturePredLossMode.MAX_POOL,
        train_scope=SelfCondTrainScope.HEAD_ONLY,
        ste_form=SteForm.DENSE,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=2,
            max_k=4,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=2.0,
            std=2.0,
            min_k=1,
            max_k=3,
        ),
        seed=42,
    )


def _runtime() -> dict:
    return {
        "core_metric_every": 0,
        "sample_every": 0,
        "steering_eval_every": 0,
        "label_dropout": 0.0,
    }


@pytest.mark.parametrize(
    ("mutation", "match"),
    [
        (lambda meta: meta["user_config"].update(num_buckets=3), "C3"),
        (lambda meta: meta["user_config"].update(negative_selection=None), "C4"),
        (lambda meta: meta["user_config"].update(feature_clusters={"path": "x"}), "C5"),
        (
            lambda meta: meta["user_config"]["feature_selection"].update(
                gradient={"mode": "pct_grad_abs_max"}
            ),
            "C5",
        ),
        (lambda meta: meta["user_config"].update(label_dropout=0.1), "C5"),
        (
            lambda meta: meta["user_config"].update(next_segment_conditioning_offset=True),
            "C5",
        ),
        (lambda meta: meta["user_config"].update(use_precomputed_batches=False), "C6"),
        (
            lambda meta: meta["user_config"].update(
                feature_delta={"mode": "splice_counterfactual"}
            ),
            "C8",
        ),
    ],
)
def test_startup_delegates_each_metadata_constraint(
    mutation: Callable[[dict], None],
    match: str,
) -> None:
    meta = _meta()
    mutation(meta)
    with pytest.raises(SelfCondTrainingError, match=match):
        validate_selfcond_startup(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=100),
            meta,
            _config(),
            16,
            _runtime(),
        )


def test_startup_c1_c2_and_unwired_cadences() -> None:
    headless = build_tiny_model(n_sae_features=16, num_buckets=2, feature_pred_config=None)
    with pytest.raises(SelfCondTrainingError, match="C1"):
        validate_selfcond_startup(headless, _meta(), _config(), 16, _runtime())

    template = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=101)
    compacted = EncoderDecoderGPT(
        template.dec_config,
        template.enc_config,
        template.xattn_config,
        n_sae_features=16,
        num_buckets=2,
        feature_pred_config=template.feature_pred_config,
        feature_vocab=FeatureVocab(n_sae_features=16, original_ids=(0, 1, 2, 3)),
    )
    compacted.init_weights()
    with pytest.raises(NotImplementedError, match="compact feature-prediction"):
        validate_selfcond_startup(compacted, _meta(), _config(), 16, _runtime())

    runtime = _runtime()
    runtime["sample_every"] = 1
    with pytest.raises(SelfCondTrainingError, match="sample_every"):
        validate_selfcond_startup(template, _meta(), _config(), 16, runtime)


def test_recorded_num_buckets_is_strict_and_feature_universe_matches() -> None:
    assert resolve_checkpoint_num_buckets(_meta()) == 2
    with pytest.raises(SelfCondTrainingError, match="num_buckets.*absent"):
        resolve_checkpoint_num_buckets({"user_config": {}})
    assert_checkpoint_feature_universe(16, 16)
    with pytest.raises(FeatureVocabError, match="n_sae_features"):
        assert_checkpoint_feature_universe(16, 32)


def test_recorded_num_buckets_falls_through_user_config_none() -> None:
    assert resolve_checkpoint_num_buckets(_sft_shaped_meta()) == 2


def test_identity_resolution_reads_sft_shaped_meta() -> None:
    identity = resolve_identity_config({}, _sft_shaped_meta(), set())
    assert identity["feature_selection"]["mean"] == 10.0
    # Identity resolution stays STRICT (an explicit user_config null does not
    # fall through — review finding); num_buckets is owned by the dedicated
    # null-tolerant resolver, and main() overwrites the identity entry with it.
    assert "num_buckets" not in identity
    assert resolve_checkpoint_num_buckets(_sft_shaped_meta()) == 2


def test_c4_reads_negative_selection_from_pretrain_conditioning_block() -> None:
    validate_selfcond_startup(
        tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=105),
        _sft_shaped_meta(),
        _config(),
        8,
        _runtime(),
    )


def test_c4_still_raises_when_conditioning_block_has_no_negatives() -> None:
    meta = _sft_shaped_meta()
    meta["pretrain_conditioning"]["negative_selection"] = None
    with pytest.raises(SelfCondTrainingError, match="C4"):
        validate_selfcond_startup(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=106),
            meta,
            _config(),
            8,
            _runtime(),
        )


def test_c8_reads_feature_delta_from_pretrain_conditioning_block() -> None:
    meta = _sft_shaped_meta()
    meta["pretrain_conditioning"]["feature_delta"] = {
        "mode": "splice_counterfactual"
    }
    with pytest.raises(SelfCondTrainingError, match="C8"):
        validate_selfcond_startup(
            tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=107),
            meta,
            _config(),
            8,
            _runtime(),
        )


def test_identity_meta_wins_unless_explicit_override_is_logged() -> None:
    cli_config = {
        "feature_dir": "/default/features",
        "feature_selection": {"distribution": "geometric", "mean": 12.0},
        "negative_selection": None,
        "seed": 9,
        "segment_size": 128,
        "sae": None,
        "num_buckets": None,
        "pooling": None,
        "dataset_phase": None,
    }
    resolved = resolve_identity_config(cli_config, _meta(), set())
    assert resolved["feature_dir"] == "/checkpoint/features"
    assert resolved["seed"] == 42
    assert resolved["negative_selection"] == _meta()["user_config"]["negative_selection"]

    spy = _LoggerSpy()
    overridden = dict(cli_config)
    overridden["feature_dir"] = "/explicit/features"
    resolved = resolve_identity_config(
        overridden,
        _meta(),
        explicit_override_roots(["+feature_dir=/explicit/features"]),
        log=spy,
    )
    assert resolved["feature_dir"] == "/explicit/features"
    assert spy.warnings and "feature_dir" in spy.warnings[0]

    mismatch = dict(cli_config)
    mismatch["segment_size"] = 64
    with pytest.raises(SelfCondTrainingError, match="segment_size"):
        resolve_identity_config(mismatch, _meta(), {"segment_size"})


def test_selfcond_policy_inherits_checkpoint_counts_and_pool_mode() -> None:
    identity = resolve_identity_config({}, _meta(), set())
    resolved = resolve_selfcond_train_config(
        _meta(),
        identity,
        {
            "train_scope": "head_only",
            "ste_form": "dense",
            "pool_mode": None,
            "positive_counts": None,
            "negative_counts": None,
        },
        16,
    )
    assert resolved.pool_mode is FeaturePredLossMode.MAX_POOL
    assert resolved.positive_counts == SamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=2,
        max_k=4,
    )
    assert resolved.negative_counts == NegativeSamplingSelection(
        distribution=Distribution.geometric,
        mean=2.0,
        std=2.0,
        min_k=1,
        max_k=3,
    )


def test_startup_c9_rejects_a_model_with_zero_cross_attention_layers() -> None:
    """C9: without cross-attention the STE never reaches the loss — fail at startup.

    xattn_frequency=3 with a 2-layer decoder places no cross-attention layer
    ((i+1) % 3 is never 0 for i in {0, 1}), the degenerate-but-constructible
    state where encoder_out is silently unused: head-only backward would raise
    mid-step and full scope would hand DistMuonAdamW None encoder grads.
    """
    fp_config = FeaturePredictionConfig(
        weight=0.1,
        layer=-1,
        loss_mode=FeaturePredLossMode.MAX_POOL,
        loss_fn=FeaturePredLossFn.BCE,
    )
    no_xattn = build_tiny_model(
        n_sae_features=16,
        num_buckets=2,
        feature_pred_config=fp_config,
        n_layer=2,
        xattn_frequency=3,
    )
    assert len(no_xattn.cross_attn_layers) == 0
    with pytest.raises(SelfCondTrainingError, match="C9.*cross-attention"):
        validate_selfcond_startup(no_xattn, _meta(), _config(), 16, _runtime())


def _recorded_selfcond_block() -> dict:
    return {
        "action_ranking_policy": "pooled_fp_logits_v1",
        "train_scope": "head_only",
        "ste_form": "dense",
        "pool_mode": "max_pool",
        "positive_counts": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 2,
            "max_k": 4,
        },
        "negative_counts": {
            "distribution": "geometric",
            "mean": 2.0,
            "std": 2.0,
            "min_k": 1,
            "max_k": 3,
        },
        "seed": 42,
        "original_head_loss_mode": "max_pool",
    }


def test_resume_refuses_a_changed_selfcond_policy() -> None:
    """Resuming under a different pool_mode/counts/seed than the checkpoint raises."""
    matching = {"selfcond": _recorded_selfcond_block()}
    assert_resume_selfcond_matches(matching, _config())  # identical policy: no raise

    for mutate, expectation in [
        (lambda block: block.update(pool_mode="last_token"), "pool_mode"),
        (
            lambda block: block["positive_counts"].update(mean=9.0, std=9.0),
            "positive_counts",
        ),
        (lambda block: block.update(seed=7), "seed"),
        (
            lambda block: block.update(action_ranking_policy="legacy_sigmoid_v0"),
            "action_ranking_policy",
        ),
    ]:
        block = _recorded_selfcond_block()
        mutate(block)
        with pytest.raises(SelfCondTrainingError, match=expectation):
            assert_resume_selfcond_matches({"selfcond": block}, _config())

    with pytest.raises(SelfCondTrainingError, match="no 'selfcond' block"):
        assert_resume_selfcond_matches({}, _config())

    legacy = _recorded_selfcond_block()
    del legacy["action_ranking_policy"]
    with pytest.raises(SelfCondTrainingError, match="action_ranking_policy"):
        assert_resume_selfcond_matches({"selfcond": legacy}, _config())


def test_c6_and_offset_accept_integer_truthy_metadata() -> None:
    """Legacy/hand-repaired meta recording bools as 1/0 must validate (the old
    `is not True` identity checks spuriously rejected valid checkpoints)."""
    meta = _meta()
    meta["user_config"]["use_precomputed_batches"] = 1
    meta["user_config"]["next_segment_conditioning_offset"] = 0
    validate_selfcond_startup(
        tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=104),
        meta,
        _config(),
        16,
        _runtime(),
    )  # must not raise


def test_resolved_identity_gradient_override_is_guarded_in_main() -> None:
    """An explicit feature_selection override with a gradient block must be
    rejected against the RESOLVED identity, not just the checkpoint meta
    (review finding: the meta-only C5 check was bypassable from the CLI).
    The guard is inline in main() right after resolve_identity_config; pin its
    presence and predicate at the source level (the repo's convention for
    main()-inline contracts)."""
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    assert "resolved_feature_selection" in source
    assert 'get("gradient") is not None' in source
    assert "RESOLVED feature_selection carries a gradient-attribution" in source
    # And the guard must sit AFTER identity resolution (it gates the resolved
    # value, not the raw CLI/meta inputs).
    assert source.index("resolve_identity_config") < source.index(
        "RESOLVED feature_selection carries a gradient-attribution"
    )


def test_base_dir_checkpoint_assertion_names_leaf_dirs(tmp_path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(SelfCondTrainingError, match="no encoder-decoder checkpoint"):
        assert_base_dir_has_checkpoint(empty)

    leafed = tmp_path / "leafed"
    (leafed / "bl_0011223344556677").mkdir(parents=True)
    (leafed / "bl_0011223344556677" / "model_000100.pt").write_bytes(b"x")
    with pytest.raises(
        SelfCondTrainingError,
        match=r"(?s)bl_0011223344556677.*exact leaf.*base_checkpoint_dir",
    ):
        assert_base_dir_has_checkpoint(leafed)

    plain = tmp_path / "plain"
    plain.mkdir()
    (plain / "model_000100.pt").write_bytes(b"x")
    assert_base_dir_has_checkpoint(plain)  # must not raise


def test_resolve_val_dirs_fails_loud_without_val_rows_and_finds_sibling(tmp_path) -> None:
    """The dead-eval state (val guard silently never firing) must be impossible:
    eval_every > 0 with no val_* sibling raises at startup; a real sibling with
    shards resolves (review finding #1)."""
    from types import SimpleNamespace

    train_dir = tmp_path / "c64" / "sall"
    train_dir.mkdir(parents=True)
    discovery = SimpleNamespace(
        train_packed_dir=train_dir,
        val_packed_dir=None,
        val_selection_dir=None,
    )
    with pytest.raises(SelfCondTrainingError, match=r"(?s)eval_every > 0.*val_"):
        resolve_val_dirs(discovery, 100)

    # eval disabled: no raise, no val dir
    assert resolve_val_dirs(discovery, 0) == (None, None)

    # a val sibling with at least one overlay shard (*.safetensors.zst —
    # what glob_overlay_shards actually matches) resolves
    val_dir = tmp_path / "c64" / "val_sall"
    val_dir.mkdir(parents=True)
    (val_dir / "shard_00000.safetensors.zst").write_bytes(b"x")
    resolved_val, resolved_sel = resolve_val_dirs(discovery, 100)
    assert resolved_val == val_dir
    assert resolved_sel is None


def test_resolve_adam_betas_reads_this_runs_config_not_checkpoint_meta() -> None:
    """A CLI adam_beta1=/adam_beta2= override must take effect: the composed
    config carries these keys via base_train.yaml, and reading them from the
    base checkpoint's recorded user_config silently ignored the override
    (review finding)."""
    assert resolve_adam_betas(SimpleNamespace(adam_beta1=0.5, adam_beta2=0.9)) == (0.5, 0.9)
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    assert 'meta_user.get("adam_beta1"' not in source
    assert 'meta_user.get("adam_beta2"' not in source
    assert source.count("resolve_adam_betas(cfg)") == 2  # head_only + full scopes


def test_resolve_grad_accum_steps_arithmetic_and_guards() -> None:
    assert resolve_grad_accum_steps(1, 64, 1, 128) == 2
    assert resolve_grad_accum_steps(2, 64, 1, 128) == 1
    assert resolve_grad_accum_steps(1, 64, 2, 0) == 1  # <= 0 falls back to micro tokens
    with pytest.raises(BeartypeCallHintParamViolation, match="device_batch_size"):
        resolve_grad_accum_steps(0, 64, 1, 128)
    with pytest.raises(SelfCondTrainingError, match="divisible"):
        resolve_grad_accum_steps(1, 64, 1, 96)


def test_resolve_dataloader_resume_is_fail_loud_on_corrupt_resume_meta() -> None:
    """A resume checkpoint without a well-formed loader cursor must raise, not
    silently restart from epoch 1 / batch 0 (review finding)."""
    assert resolve_dataloader_resume(None) is None
    assert resolve_dataloader_resume({"dataloader_state_dict": {"cursor": 3}}) == {"cursor": 3}
    for corrupted in (
        {},
        {"dataloader_state_dict": None},
        {"dataloader_state_dict": []},
        {"dataloader_state_dict": {}},
    ):
        with pytest.raises(SelfCondTrainingError, match="dataloader_state_dict"):
            resolve_dataloader_resume(corrupted)


def test_train_loader_lookahead_checkpoints_next_unconsumed_batch() -> None:
    def raw_batch(cursor: int) -> tuple:
        marker = t.tensor([cursor])
        return (
            marker,
            marker,
            marker,
            marker,
            marker.bool(),
            marker.float(),
            marker,
            marker,
            None,
            {"ddp_world_size": 1, "batch_idx": cursor},
        )

    lookahead = _TrainLoaderLookahead(
        iter([raw_batch(10), raw_batch(11), raw_batch(12)])
    )
    assert lookahead.next_state_dict["batch_idx"] == 10

    consumed = lookahead.pop()
    assert consumed.state_dict["batch_idx"] == 10
    assert lookahead.next_state_dict["batch_idx"] == 11

    consumed = lookahead.pop()
    assert consumed.state_dict["batch_idx"] == 11
    assert lookahead.next_state_dict["batch_idx"] == 12


def test_selfcond_save_gathers_next_loader_state_on_all_ranks() -> None:
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    save_predicate = source.index("if run_save:")
    gather = source.index("build_dataloader_state_envelope(", save_predicate)
    checkpoint = source.index("save_checkpoint(", gather)
    assert save_predicate < gather < checkpoint
    assert "train_lookahead.next_state_dict" in source[gather:checkpoint]


def test_assert_training_window_rejects_a_loop_that_would_never_run() -> None:
    assert_training_window(2, 3)  # normal window passes
    for start_step, num_iterations in ((0, 0), (3, 3), (4, 3)):
        with pytest.raises(SelfCondTrainingError, match="nothing to train"):
            assert_training_window(start_step, num_iterations)


def test_completion_log_is_gated_on_a_written_checkpoint() -> None:
    """main() previously claimed 'complete ... use the checkpoint directory'
    even when save_every=0 meant nothing was ever written (review finding).
    Pin the gate at source level, the repo's convention for main()-inline
    contracts."""
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    assert "wrote_checkpoint = True" in source
    assert "WITHOUT writing any checkpoint" in source
    assert source.index("save_checkpoint(") < source.index("wrote_checkpoint = True")
    assert "assert_training_window(start_step, int(cfg.num_iterations))" in source


def test_startup_c6_is_source_aware() -> None:
    """Packed-row C6 applies only to the pretraining data source."""
    meta = _meta()
    meta["user_config"]["use_precomputed_batches"] = False
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=108)

    with pytest.raises(SelfCondTrainingError, match="C6"):
        validate_selfcond_startup(
            model,
            meta,
            _config(),
            16,
            _runtime(),
            data_source=SelfCondDataSource.PRETRAINING,
        )

    validate_selfcond_startup(
        model,
        meta,
        _config(),
        16,
        _runtime(),
        data_source=SelfCondDataSource.SFT,
    )


def test_main_rejects_sft_prefetch_before_source_construction() -> None:
    """SFT buffered_v2 cursor semantics forbid train prefetch."""
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    data_source = source.index("resolve_selfcond_data_source(selfcond_block)")
    prefetch_guard = source.index('cfg.get("train_prefetch_batches") is not None')
    source_branch = source.index("if data_source is SelfCondDataSource.SFT", data_source)
    source_resolution = source.index("resolve_selfcond_sft_sources(")

    assert data_source < source_branch <= prefetch_guard < source_resolution
    assert (
        conditional_encdec_selfcond_module.SELFCOND_SFT_CURSOR_SEMANTICS
        in source[prefetch_guard:source_resolution]
    )


def test_main_rejects_pretraining_mode_on_sft_shaped_base_early() -> None:
    """An SFT leaf needs an explicit SFT source, not packed-row discovery.

    The guard sits BEFORE the resume load and model construction (the review
    restored the pre-branch resume structure): a wrong-mode base fails before
    any heavy work.
    """
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    guard = source.index("this base is an SFT checkpoint; set selfcond.data_source=sft")
    resume_load = source.index("resume_model_data, resume_optimizer_data, resume_meta")
    startup = source.index("validate_selfcond_startup(")
    pretraining_discovery = source.index("discover_packed_selection(")

    assert guard < resume_load < startup < pretraining_discovery


def test_main_sft_wiring_follows_the_ratified_nine_point_order() -> None:
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    data_source = source.index("resolve_selfcond_data_source(selfcond_block)")
    startup = source.index("validate_selfcond_startup(", data_source)
    pretraining_base_guard = source.index(
        "this base is an SFT checkpoint; set selfcond.data_source=sft"
    )
    resume_load = source.index("resume_model_data, resume_optimizer_data, resume_meta")
    sources = source.index("resolve_selfcond_sft_sources(")
    initial_batch = source.index("pending_sft_batch = next(train_gen)")
    step_batches = source.index("def step_batches()")
    train_step = source.index("run_selfcond_train_step(", step_batches)
    committed_cursor = source.index(
        "committed_consumed = sft_state.consumed",
        train_step,
    )
    eval_branch = source.index("if run_eval:", committed_cursor)
    save_branch = source.index("if run_save:", eval_branch)

    # Restored pre-branch resume structure (review finding): the SFT-shape
    # guard precedes the resume load + policy assert, which precede the single
    # model-state load and the startup validation; SFT source resolution and
    # the pre-pulled initial batch follow.
    assert (
        data_source
        < pretraining_base_guard
        < resume_load
        < startup
        < sources
        < initial_batch
        < step_batches
        < train_step
        < committed_cursor
        < eval_branch
        < save_branch
    )


def test_main_sft_pending_batch_and_dynamic_segment_contract_are_pinned() -> None:
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    step_batches = source.index("def step_batches()")
    train_step = source.index("run_selfcond_train_step(", step_batches)
    closure = source[step_batches:train_step]

    assert "nonlocal pending_sft_batch" in closure
    assert "if pending_sft_batch is not None:" in closure
    assert "raw, pending_sft_batch = pending_sft_batch, None" in closure
    assert '"selfcond SFT training batch construction"' in closure
    assert "raw = next(train_gen)" in closure
    assert "yield SelfCondBatch.from_sft(raw)" in closure
    assert "static_max_seg_size = None" in source[:step_batches]
    assert "static_max_seg_size=static_max_seg_size" in source[
        train_step : source.index("step_loss = train_result.loss", train_step)
    ]


def test_main_sft_eval_and_save_record_source_identity_without_loader_state() -> None:
    source = inspect.getsource(conditional_encdec_selfcond_module.main)
    eval_branch = source.index("if run_eval:")
    save_branch = source.index("if run_save:", eval_branch)
    eval_body = source[eval_branch:save_branch]
    save_body = source[save_branch:source.index("if master_process:", save_branch)]

    assert "run_sft_validation_cycle(" in eval_body
    assert "with disable_fp8(model):" in eval_body
    assert 'f"data_source={data_source.value}' in eval_body
    assert 'f"dataset={dataset_identity}' in eval_body
    assert '"data_source": data_source.value' in eval_body
    assert '"dataset": dataset_identity' in eval_body

    assert '"sft_cursor_semantics"' in save_body
    assert "SELFCOND_SFT_CURSOR_SEMANTICS" in save_body
    assert '"sft_consumed": committed_consumed' in save_body
    # buffered_v2 requires the per-rank buffer envelope beside the cursor;
    # without it the generator refuses to resume at all.
    assert '"sft_dataloader_state"' in save_body
    assert '"sft_artifact_coverage"' in save_body
    assert "data_source=data_source" in save_body
    assert "selfcond_data=selfcond_data_dict" in save_body
    assert "dataloader_state_dict=None" in save_body
