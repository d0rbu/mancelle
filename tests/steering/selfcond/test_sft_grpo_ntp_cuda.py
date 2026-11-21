from __future__ import annotations

import copy
from dataclasses import replace

import pytest
import torch as t

from steering.conditional.encoder_decoder.config import (
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import BASE_MODEL_DIM
from steering.conditional.encoder_decoder.sft_grpo_training import (
    NoneInitialConditioningProvider,
    SeedContext,
    assert_grpo_gradients_ready,
    backward_ntp_group,
    backward_replay_group,
    collect_conversation_group,
    configure_grpo_trainable_parameters,
)
from tests.encdec_test_helpers import build_tiny_model
from tests.steering.selfcond.test_sft_grpo_training import _config, _environment


@pytest.fixture(autouse=True)
def _release_the_compile_cache():
    """Leave the Dynamo compile cache as clean as we found it.

    These tests compile many distinct optimizer and encoder shapes, and one of
    them raises ``recompile_limit`` to get there. ``monkeypatch`` restores the
    limit but NOT the cache, so without this a later test in the same process
    that compiles one more shape trips the restored limit of 8 and fails for a
    reason that has nothing to do with it. Measured: dropping this fixture makes
    ``test_training_integration.py`` fail in a combined ``tests/steering/selfcond``
    run. The trainer resets the compiler every step for the same reason.
    """
    yield
    t.compiler.reset()


def _model():
    return (
        build_tiny_model(
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
        .cuda()
        .eval()
    )


def _group(model, *, microbatch_size: int = 1):
    return collect_conversation_group(
        model,
        _environment(),
        group_size=4,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=microbatch_size,
    )


def _missing(model) -> set[str]:
    return {name for name, parameter in model.named_parameters() if parameter.grad is None}


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_real_model_gradient_sources_match_the_head_disjointness_table() -> None:
    t.manual_seed(201)
    source = _model()
    group = _group(source)
    policy = copy.deepcopy(source)
    ntp = copy.deepcopy(source)
    configure_grpo_trainable_parameters(policy, train_lm_head=True)
    configure_grpo_trainable_parameters(ntp, train_lm_head=True)
    backward_replay_group(
        policy,
        group,
        config=_config(),
        trajectory_microbatch_size=1,
    )
    backward_ntp_group(ntp, group, trajectory_count=1, loss_scale=1.0)
    assert _missing(policy) == {"lm_head.weight"}
    assert _missing(ntp) == {
        "feature_pred_head.weight",
        "feature_pred_head.bias",
    }


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_real_collect_replay_ntp_and_readiness_complete() -> None:
    t.manual_seed(202)
    model = _model()
    group = _group(model, microbatch_size=1)
    manifest = configure_grpo_trainable_parameters(model, train_lm_head=True)
    backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=1,
    )
    ntp_loss = backward_ntp_group(model, group, trajectory_count=1, loss_scale=1.0)
    # The NTP loss is the reward's exact negation by construction: both call
    # ``masked_mean_token_ce`` on the same conditioning at the same parameters.
    # On CPU that holds bit-for-bit. On CUDA it does not, and the cause is
    # measured rather than assumed: the no-grad forward is bitwise repeatable,
    # the grad-enabled forward is bitwise repeatable, and the two differ from
    # each other by ~9.3e-7 relative. Enabling autograd changes kernel
    # selection -- the same effect that makes ``_fp_forward_replay`` recompute
    # its visible forward under ``no_grad``. The tolerance is the replay
    # tolerance this policy already ships (``GRPOPolicyConfig.replay_atol`` /
    # ``replay_rtol`` = 2e-5), which leaves ~20x headroom over the measurement.
    assert ntp_loss == pytest.approx(
        -group.trajectories[0].reward,
        rel=2e-5,
        abs=2e-5,
    )
    assert_grpo_gradients_ready(model, expected_manifest=manifest)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_real_optimizer_step_moves_both_heads(monkeypatch: pytest.MonkeyPatch) -> None:
    # ``muon_step_fused`` / ``adamw_step_fused`` are ``@torch.compile(
    # fullgraph=True)`` and recompile once per distinct parameter shape, against
    # a default ``recompile_limit`` of 8. This deliberately tiny model carries 15
    # distinct shapes, so a real optimizer step exceeds that limit. Measured to
    # be pre-existing and unrelated to this objective: the policy-only path that
    # ``origin/main`` runs (``train_lm_head=False``, no NTP backward) hits the
    # identical ``FailOnRecompileLimitHit``. The limit is a compile-cache guard,
    # not a correctness guard, so raise it for this test only.
    monkeypatch.setattr(t._dynamo.config, "recompile_limit", 64)
    t.manual_seed(203)
    model = _model()
    group = _group(model)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    optimizer = model.setup_optimizer(
        trainable_only=True,
        unembedding_lr=0.004,
        fp_head_lr=0.007,
    )
    before_lm = model.lm_head.weight.detach().clone()
    before_fp = model.feature_pred_head.weight.detach().clone()
    backward_replay_group(
        model,
        group,
        config=_config(),
        trajectory_microbatch_size=1,
    )
    backward_ntp_group(model, group, trajectory_count=1, loss_scale=1.0)

    # The two heads see disjoint gradient sources, so each must land in its own
    # AdamW group at its own learning rate -- distinct values here, otherwise the
    # `fp_head_lr` seam is a no-op and a swap would be unobservable.
    head_lrs = {
        label: group_lr
        for label, parameter in (
            ("lm_head", model.lm_head.weight),
            ("fp_head", model.feature_pred_head.weight),
        )
        for group_lr in (
            optimizer_group["lr"]
            for optimizer_group in optimizer.param_groups
            if any(entry is parameter for entry in optimizer_group["params"])
        )
    }
    # Exact values, not merely "distinct": exchanging the two rates would send
    # the NTP readout and the policy head to each other's objective's rate and
    # still satisfy a distinctness check.
    scale = (model.dec_config.n_embd / BASE_MODEL_DIM) ** -0.5
    assert head_lrs["lm_head"] == pytest.approx(0.004 * scale)
    assert head_lrs["fp_head"] == pytest.approx(0.007 * scale)

    optimizer.step()
    assert not t.equal(model.lm_head.weight, before_lm)
    assert not t.equal(model.feature_pred_head.weight, before_fp)


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_ntp_matches_the_reward_when_collection_used_a_wider_microbatch() -> None:
    """Pin the one geometry difference between NTP and the reward forward.

    ``backward_ntp_group`` always forwards at ``B=1`` with a conditioning width
    taken from a SINGLE history, while collection pads to the widest history in
    its microbatch. The two agree only because the packed encoder drops masked
    padding slots entirely -- if it ever stopped doing so, the reward and the NTP
    loss would diverge by far more than kernel-selection noise. The CPU stub is
    per-position independent, so batch and width are structurally invisible
    there; this is the only place the claim can be tested.
    """
    t.manual_seed(204)
    model = _model()
    group = _group(model, microbatch_size=4)
    configure_grpo_trainable_parameters(model, train_lm_head=True)
    widths = {
        len(action.feature_ids)
        for trajectory in group.trajectories
        for action in trajectory.conditioning
    }
    assert len(widths) > 1, (
        "fixture must contain histories of DIFFERENT widths, otherwise the "
        "microbatch padding this test exists to check never happens"
    )
    ntp_loss = backward_ntp_group(model, group, trajectory_count=1, loss_scale=1.0)
    assert ntp_loss == pytest.approx(
        -group.trajectories[0].reward,
        rel=2e-5,
        abs=2e-5,
    )


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA is required")
def test_mean_over_k_is_exact_in_fp32_and_only_approximate_for_bf16_tables() -> None:
    """The ``/k`` mean is EXACT in the parameter dtype; BF16 tables accumulate.

    Real conditioned-SFT checkpoints store 15 tensors in BF16 (``transformer.wte``,
    ``value_embeds.*``, the encoder feature/magnitude embeddings); everything else
    is FP32. ``.grad`` follows the parameter, so every per-trajectory ``+=`` into a
    BF16 table rounds. That is NOT specific to this objective -- the pre-existing
    policy replay accumulates one backward per trajectory microbatch into the same
    tables and drifts by the same order of magnitude -- so this pins both halves:
    a broken REDUCTION shows up as FP32 inexactness, while BF16 re-rounding is
    bounded and held to the policy loop's own standard.
    """
    t.manual_seed(301)
    source = _model()
    source.transformer["wte"].to(dtype=t.bfloat16)
    for embedding in source.value_embeds.values():
        embedding.to(dtype=t.bfloat16)
    group_size = 16
    group = collect_conversation_group(
        source,
        _environment(),
        group_size=group_size,
        config=_config(),
        seed_context=SeedContext("train", 0, 0),
        initial_provider=NoneInitialConditioningProvider(),
        trajectory_microbatch_size=1,
    )
    identical = replace(
        group,
        trajectories=(group.trajectories[0],) * group_size,
        advantages=(0.0,) * group_size,
    )

    def ntp_grads(count: int) -> tuple[t.Tensor, t.Tensor]:
        model = copy.deepcopy(source)
        configure_grpo_trainable_parameters(model, train_lm_head=True)
        backward_ntp_group(model, identical, trajectory_count=count, loss_scale=1.0)
        assert model.lm_head.weight.grad is not None
        assert model.transformer["wte"].weight.grad is not None
        return (
            model.lm_head.weight.grad.clone(),
            model.transformer["wte"].weight.grad.clone(),
        )

    def relative(actual: t.Tensor, expected: t.Tensor) -> float:
        return (actual.float() - expected.float()).norm().item() / max(
            expected.float().norm().item(), 1e-30
        )

    fp32_one, bf16_one = ntp_grads(1)
    fp32_many, bf16_many = ntp_grads(group_size)
    assert fp32_one.dtype == t.float32
    assert bf16_one.dtype == t.bfloat16
    # The reduction itself is exact: identical rollouts, k=1 vs k=K, FP32 param.
    assert t.equal(fp32_many, fp32_one)
    ntp_drift = relative(bf16_many, bf16_one)

    def policy_wte_grad(microbatch: int) -> t.Tensor:
        model = copy.deepcopy(source)
        configure_grpo_trainable_parameters(model, train_lm_head=True)
        backward_replay_group(
            model,
            group,
            config=_config(),
            trajectory_microbatch_size=microbatch,
        )
        grad = model.transformer["wte"].weight.grad
        assert grad is not None
        return grad.clone()

    policy_drift = relative(policy_wte_grad(1), policy_wte_grad(group_size))
    assert policy_drift > 0.0, "fixture must actually exercise BF16 accumulation"
    assert ntp_drift < 0.05, f"NTP BF16 accumulation drifted {ntp_drift:.2%}"
    assert ntp_drift < 20 * policy_drift + 0.01, (
        f"NTP drift {ntp_drift:.2%} is materially worse than the pre-existing "
        f"policy-loop drift {policy_drift:.2%}"
    )
