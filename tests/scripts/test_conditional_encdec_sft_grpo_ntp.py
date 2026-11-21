from __future__ import annotations

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.train import conditional_encdec_sft_grpo as trainer
from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import BASE_MODEL_DIM
from steering.conditional.encoder_decoder.sft_grpo import SFTGRPOError
from steering.conditional.encoder_decoder.sft_grpo_runtime import (
    GRPOBatchGeometry,
    assert_grpo_resume_identity,
    build_grpo_resume_identity,
)
from steering.conditional.encoder_decoder.sft_grpo_training import NTPObjective
from tests.encdec_test_helpers import build_tiny_model
from tests.scripts.test_conditional_encdec_sft_grpo import _policy

CONFIG_DIR = str(Path("configs/training").resolve())


def _block(**overrides: object) -> dict[str, object]:
    block: dict[str, object] = {
        "trajectories_per_group": 1,
        "loss_weight": 1.0,
        "decay_start": 0.8,
        "fp_head_lr": None,
    }
    block.update(overrides)
    return block


def _cfg(*, ntp: object = "default", lr_decay_start: object = 0.8) -> dict[str, object]:
    return {
        "ntp": _block() if ntp == "default" else ntp,
        "lr_decay_start": lr_decay_start,
    }


def test_absent_ntp_block_disables_the_objective() -> None:
    """Absence IS the disable mechanism -- there is no separate enable flag."""
    assert trainer.resolve_ntp_settings(_cfg(ntp=None), group_size=4) is None
    assert trainer.objective_weights(None, completed_step=0, num_iterations=10) is None


def test_ntp_block_must_be_a_mapping_or_null() -> None:
    with pytest.raises(SFTGRPOError, match="ntp must be a mapping or null"):
        trainer.resolve_ntp_settings(_cfg(ntp=1.0), group_size=4)


@pytest.mark.parametrize(
    ("block", "match"),
    [
        ({}, "missing="),
        (_block(extra=1), "unknown="),
        ({k: v for k, v in _block().items() if k != "loss_weight"}, "missing="),
    ],
)
def test_ntp_block_must_define_exactly_its_fields(block: object, match: str) -> None:
    """No silent defaults inside the block: a typo must not fall back to 1.0."""
    with pytest.raises(SFTGRPOError, match=match):
        trainer.resolve_ntp_settings(_cfg(ntp=block), group_size=4)


def test_every_block_key_lands_in_the_field_it_names() -> None:
    """MUTUALLY DISTINCT values, so no key can be routed to another field.

    The block's defaults are all 1.0 or None -- the multiplicative identity -- so
    with them a misrouted key, a dropped weight, or a hard-coded ``k`` are all
    invisible. Distinct values are what make those mutations observable.
    """
    settings = trainer.resolve_ntp_settings(
        _cfg(
            ntp=_block(
                trajectories_per_group=2,
                loss_weight=3.0,
                decay_start=0.25,
                fp_head_lr=0.5,
            )
        ),
        group_size=8,
    )
    assert settings is not None
    assert (
        int(settings.trajectories_per_group),
        float(settings.loss_weight),
        float(settings.decay_start),
        float(settings.fp_head_lr),
    ) == (2, 3.0, 0.25, 0.5)


def test_objective_weights_apply_the_ntp_base_and_carry_k() -> None:
    """At step 0 ``sft_flat_decay`` is exactly 1.0, so the weight is the base."""
    settings = trainer.resolve_ntp_settings(
        _cfg(ntp=_block(trajectories_per_group=2, loss_weight=3.0)),
        group_size=8,
    )
    assert trainer.objective_weights(settings, completed_step=0, num_iterations=10) == NTPObjective(
        trajectories_per_group=2, loss_weight=3.0
    )


def test_ntp_weight_is_flat_until_decay_start_then_linear_to_zero() -> None:
    """`decay_start` says WHEN decay begins, as a fraction of the run."""
    settings = trainer.resolve_ntp_settings(_cfg(ntp=_block(decay_start=0.8)), group_size=4)
    weights = [
        trainer.objective_weights(settings, completed_step=step, num_iterations=10).loss_weight
        for step in range(10)
    ]
    assert weights == pytest.approx([1.0] * 9 + [0.5])
    half = trainer.resolve_ntp_settings(_cfg(ntp=_block(decay_start=0.5)), group_size=4)
    assert [
        trainer.objective_weights(half, completed_step=step, num_iterations=10).loss_weight
        for step in range(10)
    ] == pytest.approx([1.0] * 5 + [1.0, 0.8, 0.6, 0.4, 0.2])
    never = trainer.resolve_ntp_settings(_cfg(ntp=_block(decay_start=1.0)), group_size=4)
    assert all(
        trainer.objective_weights(never, completed_step=step, num_iterations=10).loss_weight == 1.0
        for step in range(10)
    )


@pytest.mark.parametrize("start", [0.0, 0.5, 0.8, 1.0])
def test_lr_decay_start_is_read_without_an_ntp_block(start: float) -> None:
    """The RL run's LR decay is independent of the auxiliary objective."""
    assert trainer.resolve_lr_decay_start(_cfg(ntp=None, lr_decay_start=start)) == pytest.approx(
        start
    )


@pytest.mark.parametrize("value", [-0.1, 1.1, float("nan"), True])
def test_lr_decay_start_must_be_a_run_fraction(value: object) -> None:
    with pytest.raises(SFTGRPOError, match="invalid lr_decay_start"):
        trainer.resolve_lr_decay_start(_cfg(lr_decay_start=value))


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"trajectories_per_group": 5}, "no greater than group_size=4"),
        ({"trajectories_per_group": 0}, "invalid NTP setting"),
        ({"loss_weight": -1.0}, "invalid NTP setting"),
        ({"decay_start": 1.5}, "invalid NTP setting"),
        ({"loss_weight": float("nan")}, "invalid NTP setting"),
        ({"fp_head_lr": 0.0}, "invalid NTP setting"),
        ({"trajectories_per_group": True}, "invalid NTP setting"),
        ({"loss_weight": True}, "invalid NTP setting"),
    ],
)
def test_enabled_ntp_rejects_invalid_block_values(
    overrides: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(SFTGRPOError, match=match):
        trainer.resolve_ntp_settings(_cfg(ntp=_block(**overrides)), group_size=4)


def test_config_composes_a_disabled_ntp_block() -> None:
    cfg = _compose_grpo("conditional_encdec_sft_grpo")
    assert cfg["ntp"] is None
    assert cfg["lr_decay_start"] == 0.8
    assert trainer.resolve_ntp_settings(cfg, group_size=8) is None
    assert trainer.resolve_lr_decay_start(cfg) == 0.8


def _compose_grpo(name: str) -> dict[str, object]:
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        cfg = compose(
            config_name=name,
            overrides=[
                "base_checkpoint_dir=/base",
                "train_artifact_dir=/train",
                "validation_artifact_dir=/val",
                "global_conversations_per_step=2",
                "group_size=8",
                "device_conversation_batch_size=1",
                "trajectory_microbatch_size=1",
                "num_iterations=1",
            ],
        )
    resolved = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(resolved, dict)
    return resolved


def _identity(*, ntp: dict[str, object] | None = None) -> dict[str, object]:
    return build_grpo_resume_identity(
        base_checkpoint_sha256="a" * 64,
        train_artifact_sha256="b" * 64,
        validation_artifact_sha256="c" * 64,
        geometry=GRPOBatchGeometry(2, 4, 1, 1, 2),
        parameter_manifest=("x",),
        policy=_policy(),
        segmentation={"segment_length": 256},
        initial_conditioning_mode="none",
        sae_loaded=False,
        kl_beta=0,
        reference_sha256=None,
        optimization={"num_iterations": 100, "weight_decay": 0.0},
        ntp=ntp,
    )


def test_legacy_identity_is_key_exact_and_sha_pinned() -> None:
    identity = _identity()
    assert "ntp" not in identity
    assert identity["sha256"] == "41cd82e4ec87461ec83cc330e706c4ae39a0aacf7be5c38d7104adb0e715891d"
    assert set(identity) == {
        "version",
        "base_checkpoint_sha256",
        "train_artifact_sha256",
        "validation_artifact_sha256",
        "geometry",
        "parameter_manifest",
        "action_mode",
        "action_policy_version",
        "policy_seed",
        "positive_counts",
        "negative_counts",
        "segmentation",
        "initial_conditioning_mode",
        "sae_loaded",
        "kl_beta",
        "reference_sha256",
        "optimization",
        "kl_direction",
        "kl_reduction",
        "sha256",
    }


def test_ntp_identity_adds_key_and_rejects_every_field_drift() -> None:
    block = {
        "version": 1,
        "trajectories_per_group": 1,
        "loss_weight": 1.0,
        "decay_start": 0.8,
        "fp_head_lr": None,
    }
    baseline = _identity()
    enabled = _identity(ntp=block)
    assert enabled["ntp"] == block
    assert enabled["sha256"] != baseline["sha256"]
    # Pin the ENABLED hash to a literal, exactly as the legacy path is pinned.
    # Without it, a change to how the ntp block is serialized would silently
    # invalidate every existing NTP checkpoint with no test failure.
    assert enabled["sha256"] == ("0c98e5e60297b2424dcc420cefb8bf84255cd4d4dc9a41c226a41687c6686c6b")
    for name, value in block.items():
        drifted = dict(block)
        drifted[name] = 2 if value != 2 else 3
        with pytest.raises(SFTGRPOError, match="resume identity mismatch"):
            assert_grpo_resume_identity(enabled, _identity(ntp=drifted))


def test_settings_serialize_to_the_identity_block_field_for_field() -> None:
    """Distinct values, so a field routed to the wrong key is observable.

    ``identity_dict`` is what pins the enabled resume identity. Built inline from
    same-valued defaults, a swapped pair would hash self-consistently and two
    genuinely different configurations would cross-resume into each other.
    """
    settings = trainer.resolve_ntp_settings(
        _cfg(
            ntp=_block(
                trajectories_per_group=2,
                loss_weight=3.0,
                decay_start=0.25,
                fp_head_lr=0.5,
            )
        ),
        group_size=8,
    )
    assert settings is not None
    assert settings.identity_dict() == {
        "version": 1,
        "trajectories_per_group": 2,
        "loss_weight": 3.0,
        "decay_start": 0.25,
        "fp_head_lr": 0.5,
    }


def test_pilot_config_inherits_the_disabled_ntp_surface() -> None:
    """The pilot extends the base config, so it must carry the same two keys.

    ``resolve_ntp_settings`` reads raw ``cfg["ntp"]``, so a config that stopped
    inheriting would raise ``KeyError`` rather than the intended fail-loud error.
    """
    cfg = _compose_grpo("conditional_encdec_sft_grpo_pilot")
    assert cfg["ntp"] is None
    assert cfg["lr_decay_start"] == 0.8
    assert trainer.resolve_ntp_settings(cfg, group_size=256) is None


def _optimizer_model():
    model = build_tiny_model(
        n_sae_features=8,
        num_buckets=2,
        feature_pred_config=FeaturePredictionConfig(
            layer=-1,
            loss_mode=FeaturePredLossMode.MAX_POOL,
            loss_fn=FeaturePredLossFn.BCE,
        ),
        n_layer=2,
        n_embd=64,
        seq_len=16,
        xattn_frequency=2,
    )
    model.requires_grad_(True)
    return model


def _optimizer_cfg(**overrides: object):
    base = {
        "weight_decay": 0.0,
        "lr_multiplier": 1.0,
        "init_lr_frac": 1.0,
        "unembedding_lr": 0.004,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "encoder_embedding_lr": 0.2,
        "encoder_matrix_lr": 0.02,
        "xattn_matrix_lr": 0.02,
    }
    base.update(overrides)
    return OmegaConf.create(base)


def _head_rates(optimizer, model) -> dict[str, tuple[float, float]]:
    rates: dict[str, tuple[float, float]] = {}
    for group in optimizer.param_groups:
        for name, parameter in model.named_parameters():
            if name not in {"lm_head.weight", "feature_pred_head.weight"}:
                continue
            if any(parameter is entry for entry in group["params"]):
                rates[name] = (float(group["lr"]), float(group["initial_lr"]))
    return rates


def test_build_optimizer_applies_lr_multiplier_and_init_frac_to_the_fp_head() -> None:
    """The only trainer branch that is line-covered but never value-covered.

    ``_build_optimizer`` threads ``fp_head_lr * lr_multiplier`` into
    ``setup_optimizer`` and then scales every group by ``init_lr_frac``. Nothing
    exercised that arithmetic with a non-``None`` value, so the FP head could
    train at the wrong rate in every NTP run while the CUDA test -- which calls
    ``setup_optimizer`` directly and bypasses the trainer -- stayed green.
    Distinct multiplier and fraction values make a dropped factor observable.
    """
    model = _optimizer_model()
    optimizer = trainer._build_optimizer(
        model,
        _optimizer_cfg(lr_multiplier=2.0, init_lr_frac=0.5),
        fp_head_lr=0.007,
    )
    scale = (model.dec_config.n_embd / BASE_MODEL_DIM) ** -0.5
    rates = _head_rates(optimizer, model)
    assert rates["feature_pred_head.weight"][0] == pytest.approx(0.007 * 2.0 * scale * 0.5)
    assert rates["feature_pred_head.weight"][1] == pytest.approx(
        rates["feature_pred_head.weight"][0]
    )
    assert rates["lm_head.weight"][0] == pytest.approx(0.004 * 2.0 * scale * 0.5)
    assert rates["lm_head.weight"][0] != rates["feature_pred_head.weight"][0]


def test_build_optimizer_shares_the_unembedding_rate_when_fp_head_lr_is_absent() -> None:
    model = _optimizer_model()
    optimizer = trainer._build_optimizer(model, _optimizer_cfg(), fp_head_lr=None)
    rates = _head_rates(optimizer, model)
    assert rates["lm_head.weight"][0] == pytest.approx(rates["feature_pred_head.weight"][0])
