"""Integration contracts for self-conditioning training helpers."""

from __future__ import annotations

import copy
import inspect
import math
import re
from collections.abc import Iterator
from typing import Any

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation

import nanochat.flash_attention as _fa_mod
import scripts.train.conditional_encdec_selfcond as conditional_encdec_selfcond_module
from errors import SelfCondTrainingError
from nanochat.flash_attention import FlashAttnImpl
from nanochat.gpt import OptimizerKind
from nanochat.optim import MuonAdamW
from scripts.train.conditional_encdec_selfcond import (
    evaluate_selfcond_bpb,
)
from scripts.train.conditional_encdec_selfcond import (
    run_selfcond_train_step as _run_selfcond_train_step,
)
from steering.conditional.encoder_decoder.config import FeaturePredLossMode
from steering.conditional.encoder_decoder.dataloader import EncDecTrainBatch
from steering.conditional.encoder_decoder.encoder import SteForm
from steering.conditional.encoder_decoder.selfcond_training import (
    VAL_ROLLOUT_STEP_SENTINEL,
    MutableConditioning,
    RolloutResult,
    SelfCondTrainConfig,
    SelfCondTrainScope,
    _step_metrics,
    build_head_only_optimizer,
    freeze_for_head_only,
    rollout,
    selfcond_step_seed,
)
from steering.conditional.feature_pipeline import FeaturePipelineConfig
from steering.conditional.sampling import (
    Distribution,
    NegativeSamplingSelection,
    SamplingSelection,
)
from steering.refined_types import PositiveInt
from tests.encdec_test_helpers import VOCAB_SIZE
from tests.steering.selfcond.helpers import SelfcondBatch, make_batch, tiny_head_model


def run_selfcond_train_step(*args: Any, **kwargs: Any) -> Any:
    """Pass raw integration values through the production beartype boundary."""
    return _run_selfcond_train_step(*args, **kwargs)


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _config(
    scope: SelfCondTrainScope,
    ste_form: SteForm,
    *,
    seed: int = 123,
    pool_mode: FeaturePredLossMode = FeaturePredLossMode.MAX_POOL,
) -> SelfCondTrainConfig:
    return SelfCondTrainConfig(
        pool_mode=pool_mode,
        train_scope=scope,
        ste_form=ste_form,
        positive_counts=SamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=1,
            max_k=2,
        ),
        negative_counts=NegativeSamplingSelection(
            distribution=Distribution.geometric,
            mean=1.0,
            std=1.0,
            min_k=0,
            max_k=1,
        ),
        seed=seed,
    )


def _plain_optimizer(
    model,
    scope: SelfCondTrainScope,
    *,
    lr: float,
) -> t.optim.AdamW:
    if scope is SelfCondTrainScope.HEAD_ONLY:
        freeze_for_head_only(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = t.optim.AdamW(parameters, lr=lr, weight_decay=0.0)
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
        group["kind"] = OptimizerKind.ADAMW
    return optimizer


def _rollout_ce(
    model,
    batch: SelfcondBatch,
    cfg: SelfCondTrainConfig,
) -> float:
    rng = t.Generator().manual_seed(selfcond_step_seed(cfg.seed, 0, VAL_ROLLOUT_STEP_SENTINEL))
    result = rollout(model, batch, cfg, rng, None)
    with t.no_grad():
        loss = model(
            batch.inputs,
            batch.targets,
            result.conditioning.fids,
            result.conditioning.mags,
            result.conditioning.mask,
            batch.seg_map,
        )
    return float(loss.item())


def _eval_loader(batch: SelfcondBatch) -> Iterator[EncDecTrainBatch]:
    typed = EncDecTrainBatch(
        inputs=batch.inputs,
        targets=batch.targets,
        enc_fids=batch.enc_fids,
        enc_mags=batch.enc_mags,
        enc_mask=batch.enc_mask,
        enc_pcts=batch.enc_pcts,
        seg_map=batch.seg_map,
        seg_doc_id=batch.seg_doc_id,
        fp_targets=None,
        state_dict={"cursor": 1},
    )
    yield typed


def _assert_nested_equal(left, right) -> None:
    if isinstance(left, t.Tensor):
        assert isinstance(right, t.Tensor)
        assert t.equal(left, right)
        return
    if isinstance(left, dict):
        assert isinstance(right, dict)
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
        return
    if isinstance(left, (tuple, list)):
        assert isinstance(right, type(left))
        assert len(left) == len(right)
        for left_item, right_item in zip(left, right, strict=True):
            _assert_nested_equal(left_item, right_item)
        return
    assert left == right


def test_head_only_optimizer_group_step_and_state_roundtrip(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """T-I4: one complete AdamW group owns exactly the FP-head parameters."""
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=90)
    assert model.feature_pred_head is not None
    unembedding_lr = 0.004
    betas = (0.71, 0.89)
    optimizer = build_head_only_optimizer(
        model,
        unembedding_lr=unembedding_lr,
        adam_betas=betas,
    )

    assert isinstance(optimizer, MuonAdamW)
    assert len(optimizer.param_groups) == 1
    group = optimizer.param_groups[0]
    assert set(group) == {
        "kind",
        "params",
        "lr",
        "betas",
        "eps",
        "weight_decay",
        "initial_lr",
    }
    head_parameters = list(model.feature_pred_head.parameters())
    assert len(head_parameters) == 2
    assert len(group["params"]) == 2
    assert all(
        actual is expected
        for actual, expected in zip(group["params"], head_parameters, strict=True)
    )
    expected_lr = unembedding_lr * (model.dec_config.n_embd / 768) ** -0.5
    assert group["kind"] is OptimizerKind.ADAMW
    assert group["lr"] == expected_lr
    assert group["initial_lr"] == expected_lr
    assert group["betas"] == betas
    assert group["eps"] == 1e-10
    assert group["weight_decay"] == 0.0

    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for parameter in model.parameters():
        parameter.grad = t.ones_like(parameter)
    optimizer.step()
    for name, parameter in model.named_parameters():
        if name.startswith("feature_pred_head."):
            assert not t.equal(parameter, before[name])
        else:
            assert t.equal(parameter, before[name])
    for parameter in head_parameters:
        assert bool((optimizer.state[parameter]["exp_avg"] != 0).all())

    saved = copy.deepcopy(optimizer.state_dict())
    restored_model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=91)
    assert restored_model.feature_pred_head is not None
    restored = build_head_only_optimizer(
        restored_model,
        unembedding_lr=unembedding_lr,
        adam_betas=betas,
    )
    restored.load_state_dict(saved)
    for original, replacement in zip(
        head_parameters,
        restored_model.feature_pred_head.parameters(),
        strict=True,
    ):
        assert t.equal(
            optimizer.state[original]["exp_avg"],
            restored.state[replacement]["exp_avg"],
        )


@pytest.mark.parametrize("scope", list(SelfCondTrainScope))
@pytest.mark.parametrize("ste_form", list(SteForm))
def test_real_train_step_is_finite_and_exercises_predicted_slots(
    scope: SelfCondTrainScope,
    ste_form: SteForm,
) -> None:
    """T-I1: the script's actual step runs for all scope/form combinations."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=120)
    cfg = _config(scope, ste_form)
    optimizer = _plain_optimizer(model, scope, lr=0.002)
    batch = make_batch([[2]], T=8, F=3, n_sae=8, seed=121)

    result = run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        cfg,
        global_step=0,
        ddp_rank=0,
        num_iterations=4,
        grad_clip_max_norm=1.0,
        static_max_seg_size=None,
    )

    assert math.isfinite(result.loss)
    assert result.metrics.frac_predicted_slots > 0.0
    assert result.metrics.mean_k == 1.0
    assert result.metrics.mean_l == 0.0


def test_head_only_fixed_batch_overfit_reduces_rollout_ce() -> None:
    """T-I2: fifty seeded real steps reduce fixed-rollout CE by a loose margin."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=130)
    with t.no_grad():
        model.lm_head.weight.mul_(10.0)
    cfg = _config(SelfCondTrainScope.HEAD_ONLY, SteForm.DENSE, seed=131)
    batch = make_batch([[2]], T=8, F=3, n_sae=8, seed=132)
    continuation_tokens = batch.seg_map[0] == 1
    candidate_log_probs: list[t.Tensor] = []
    with t.no_grad():
        for feature_id in range(8):
            fids = batch.enc_fids.clone()
            mags = batch.enc_mags.clone()
            mask = batch.enc_mask.clone()
            fids[:, 1] = 0
            mags[:, 1] = 0
            mask[:, 1] = False
            fids[:, 1, 0] = feature_id
            mask[:, 1, 0] = True
            logits = model(batch.inputs, None, fids, mags, mask, batch.seg_map)
            candidate_log_probs.append(logits.log_softmax(dim=-1))
        stacked = t.stack(candidate_log_probs)
        advantages = stacked[1:, 0, continuation_tokens] - stacked[0, 0, continuation_tokens]
        best_feature = int(advantages.max(dim=-1).values.mean(dim=-1).argmax().item()) + 1
        target_advantage = (
            stacked[best_feature, 0, continuation_tokens] - stacked[0, 0, continuation_tokens]
        )
        batch.targets.fill_(-1)
        batch.targets[0, continuation_tokens] = target_advantage.argmax(dim=-1)
        assert model.feature_pred_head is not None
        model.feature_pred_head.weight.zero_()
        model.feature_pred_head.bias.zero_()

    optimizer = _plain_optimizer(model, cfg.train_scope, lr=0.1)
    before = _rollout_ce(model, batch, cfg)

    for step in range(50):
        run_selfcond_train_step(
            model,
            optimizer,
            [batch],
            cfg,
            global_step=step,
            ddp_rank=0,
            num_iterations=50,
            grad_clip_max_norm=1.0,
            static_max_seg_size=None,
        )

    after = _rollout_ce(model, batch, cfg)
    assert after < before * 0.99, (before, after)


def test_selfcond_validation_is_deterministic_no_grad_and_optimizer_untouched() -> None:
    """T-I6: fixed validation seed is exact and evaluation is side-effect free."""
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=140)
    cfg = _config(SelfCondTrainScope.HEAD_ONLY, SteForm.SELECTED, seed=141)
    optimizer = _plain_optimizer(model, cfg.train_scope, lr=0.01)
    batch = make_batch([[3]], T=9, F=3, n_sae=8, seed=142)
    run_selfcond_train_step(
        model,
        optimizer,
        [batch],
        cfg,
        global_step=0,
        ddp_rank=0,
        num_iterations=2,
        grad_clip_max_norm=1.0,
        static_max_seg_size=None,
    )
    optimizer_before = copy.deepcopy(optimizer.state_dict())
    model_before = {name: value.detach().clone() for name, value in model.state_dict().items()}
    assert all(parameter.grad is None for parameter in model.parameters())
    token_bytes = t.ones(VOCAB_SIZE, dtype=t.int64)

    first = evaluate_selfcond_bpb(
        model,
        _eval_loader(batch),
        1,
        token_bytes,
        cfg,
        ddp_rank=0,
        static_max_seg_size=None,
    )
    second = evaluate_selfcond_bpb(
        model,
        _eval_loader(batch),
        1,
        token_bytes,
        cfg,
        ddp_rank=0,
        static_max_seg_size=None,
    )

    assert first == second
    assert math.isfinite(first)
    assert all(parameter.grad is None for parameter in model.parameters())
    for name, value in model.state_dict().items():
        assert t.equal(value, model_before[name])
    _assert_nested_equal(optimizer.state_dict(), optimizer_before)


def test_eval_block_logs_both_val_metrics_together() -> None:
    """T-I6 (strengthened): the script's eval cadence must emit val/bpb AND
    val/bpb_selfcond in the same wandb log call — reviewer finding: exercising
    evaluate_selfcond_bpb in isolation never pinned the paired logging."""
    source = inspect.getsource(conditional_encdec_selfcond_module)
    eval_log = re.search(
        r"wandb_run\.log\(\s*\{[^}]*\"val/bpb\": val_bpb,[^}]*"
        r"\"val/bpb_selfcond\": val_bpb_selfcond,[^}]*\}",
        source,
    )
    assert eval_log is not None, (
        "the eval cadence must log val/bpb and val/bpb_selfcond in one wandb call"
    )


def test_full_scope_runs_with_the_real_setup_optimizer(
    monkeypatch: pytest.MonkeyPatch,
    request: pytest.FixtureRequest,
) -> None:
    """Review finding: FULL scope was never exercised through the REAL
    model.setup_optimizer — a refactor of its group schema (the "kind" key,
    initial_lr population) would pass the whole suite and break the first
    production full-scope run. Two real steps with the production optimizer."""
    for name in ("RANK", "LOCAL_RANK", "WORLD_SIZE"):
        monkeypatch.delenv(name, raising=False)
    # The real Muon step is torch.compile(fullgraph=True)d per group shape;
    # mirror the production script's dynamo cache bound (it sets 64 before
    # training for exactly this reason) so the shared test process's already-
    # consumed cache entries don't trip FailOnRecompileLimitHit here.
    # Clear these variants before monkeypatch restores the lower cache limit.
    request.addfinalizer(t.compiler.reset)
    monkeypatch.setattr(t._dynamo.config, "cache_size_limit", 64)
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=16, seed=93)
    batch = make_batch([[3], [2, 1]], T=12, F=4, n_sae=16, seed=94)
    optimizer = model.setup_optimizer()
    assert all("initial_lr" in group for group in optimizer.param_groups)
    muon_groups = [
        group for group in optimizer.param_groups if group.get("kind") == OptimizerKind.MUON
    ]
    assert muon_groups, "setup_optimizer must produce Muon groups for full scope"

    cfg = _config(SelfCondTrainScope.FULL, SteForm.DENSE, seed=95)
    before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
    for step in range(2):
        result = run_selfcond_train_step(
            model,
            optimizer,
            [batch],
            cfg,
            global_step=step,
            ddp_rank=0,
            num_iterations=2,
            grad_clip_max_norm=1.0,
            static_max_seg_size=None,
        )
        assert math.isfinite(result.loss)
    # The production schedule ran against the real Muon groups (momentum is
    # only written for kind == MUON — the exact schema a refactor could break).
    assert all("momentum" in group for group in muon_groups)
    changed = [
        name for name, parameter in model.named_parameters() if not t.equal(parameter, before[name])
    ]
    assert any(name.startswith("transformer.") for name in changed)
    assert any(name.startswith("encoder.") for name in changed)
    assert any(name.startswith("feature_pred_head.") for name in changed)


def test_train_prefetch_batches_resolver_and_val_invariant() -> None:
    from scripts.train.conditional_encdec_selfcond import (
        VAL_EVAL_PREFETCH_BATCHES,
        resolve_train_prefetch_batches,
    )

    assert VAL_EVAL_PREFETCH_BATCHES == 0  # eval loaders must stay unwrapped
    grad_accum_steps = PositiveInt.parse(7)
    assert resolve_train_prefetch_batches(None, grad_accum_steps) == 7  # null = parity
    assert resolve_train_prefetch_batches(3, grad_accum_steps) == 3
    assert resolve_train_prefetch_batches(0, grad_accum_steps) == 0  # explicit synchronous
    with pytest.raises(TypeError):
        resolve_train_prefetch_batches(-1, grad_accum_steps)


def test_bpb_accumulation_helper_matches_hand_computation() -> None:
    """The shared val/bpb + val/bpb_selfcond accumulation contract, pinned
    against a hand computation with ignore_index and zero-byte tokens."""
    from steering.conditional.encoder_decoder.loss_eval import (
        accumulate_bpb_nats_bytes,
        reduce_bpb,
    )

    token_bytes = t.tensor([0, 2, 3, 1], dtype=t.int64)  # token 0 has zero bytes
    loss = t.tensor([1.0, 2.0, 4.0, 8.0])
    targets = t.tensor([1, 0, -1, 3])  # -1 = ignore_index; token 0 zero-byte
    nats, num_bytes = accumulate_bpb_nats_bytes(loss, targets, token_bytes)
    # only targets 1 (2 bytes) and 3 (1 byte) contribute nats AND bytes
    assert float(nats) == 1.0 + 8.0
    assert int(num_bytes) == 2 + 1
    assert reduce_bpb(nats.clone(), num_bytes.clone()) == pytest.approx(9.0 / (math.log(2) * 3))
    # no-negative fast path is value-identical to the masked path. The masked
    # invocation must ACTUALLY take the masked branch: append a -1 target with
    # a nonzero loss and demand the ignored element contributes exactly nothing
    # (the original form sliced the -1 off before the call, comparing the fast
    # path to itself — reviewer finding).
    clean_targets = t.tensor([1, 2, 3, 1])
    fast = accumulate_bpb_nats_bytes(loss, clean_targets, token_bytes)
    padded_targets = t.cat([clean_targets, t.tensor([-1])])
    padded_loss = t.cat([loss, t.tensor([123.0])])
    assert bool((padded_targets.int() < 0).any())  # masked branch engaged
    masked = accumulate_bpb_nats_bytes(padded_loss, padded_targets, token_bytes)
    assert t.equal(fast[0], masked[0]) and t.equal(fast[1], masked[1])
    # zero total bytes -> inf convention
    assert reduce_bpb(t.tensor(0.0), t.tensor(0)) == float("inf")


def test_grad_accum_two_micro_batches_is_deterministic_and_consumes_both() -> None:
    """grad_accum_steps > 1 is production's common case yet was previously
    untested (review finding): two distinct micro-batches, one optimizer step,
    bitwise-reproducible outcome."""

    def one_run() -> tuple[float, t.Tensor]:
        model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=140)
        cfg = _config(SelfCondTrainScope.HEAD_ONLY, SteForm.DENSE)
        optimizer = _plain_optimizer(model, SelfCondTrainScope.HEAD_ONLY, lr=0.002)
        batches = iter(
            [
                make_batch([[2]], T=8, F=3, n_sae=8, seed=141),
                make_batch([[3]], T=9, F=3, n_sae=8, seed=142),
            ]
        )
        result = run_selfcond_train_step(
            model,
            optimizer,
            batches,
            cfg,
            global_step=0,
            ddp_rank=0,
            num_iterations=4,
            grad_clip_max_norm=1.0,
            static_max_seg_size=None,
            grad_accum_steps=2,
        )
        with pytest.raises(StopIteration):
            next(batches)  # both micro-batches were consumed
        assert model.feature_pred_head is not None
        return result.loss, model.feature_pred_head.weight.detach().clone()

    first_loss, first_weight = one_run()
    second_loss, second_weight = one_run()
    assert math.isfinite(first_loss)
    assert first_loss == second_loss
    assert t.equal(first_weight, second_weight)


def test_train_step_guards_raise_loudly() -> None:
    model = tiny_head_model(FeaturePredLossMode.MAX_POOL, n_sae=8, seed=150)
    cfg = _config(SelfCondTrainScope.HEAD_ONLY, SteForm.DENSE)
    optimizer = _plain_optimizer(model, SelfCondTrainScope.HEAD_ONLY, lr=0.002)
    batch = make_batch([[2]], T=8, F=3, n_sae=8, seed=151)
    common = {
        "global_step": 0,
        "ddp_rank": 0,
        "grad_clip_max_norm": 1.0,
        "static_max_seg_size": None,
    }
    with pytest.raises(BeartypeCallHintParamViolation, match="grad_accum_steps"):
        run_selfcond_train_step(
            model, optimizer, [batch], cfg, num_iterations=4, grad_accum_steps=0, **common
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="grad_accum_steps"):
        run_selfcond_train_step(
            model,
            optimizer,
            [batch],
            cfg,
            num_iterations=4,
            grad_accum_steps=True,
            **common,
        )
    with pytest.raises(BeartypeCallHintParamViolation, match="num_iterations"):
        run_selfcond_train_step(model, optimizer, [batch], cfg, num_iterations=0, **common)
    with pytest.raises(BeartypeCallHintParamViolation, match="grad_clip_max_norm"):
        run_selfcond_train_step(
            model,
            optimizer,
            [batch],
            cfg,
            num_iterations=4,
            global_step=0,
            ddp_rank=0,
            grad_clip_max_norm=0.0,
            static_max_seg_size=None,
        )
    with pytest.raises(SelfCondTrainingError, match="ended before grad_accum_steps"):
        run_selfcond_train_step(
            model, optimizer, [batch], cfg, num_iterations=4, grad_accum_steps=2, **common
        )


def test_precomputed_loader_rejects_boolean_prefetch_depth(tmp_path) -> None:
    pipeline = object.__new__(FeaturePipelineConfig)
    with pytest.raises(BeartypeCallHintParamViolation, match="prefetch_batches"):
        conditional_encdec_selfcond_module._make_precomputed_loader(
            packed_dir=tmp_path,
            selection_dir=None,
            batch_size=PositiveInt.parse(1),
            pipeline=pipeline,
            device="cpu",
            pad_f=None,
            ddp_rank=0,
            ddp_world_size=1,
            resume_state_dict=None,
            prefetch_batches=False,
        )


def test_mean_selected_pos_score_matches_hand_computation() -> None:
    """The wandb-logged selection diagnostic had zero numeric verification
    (review finding): pin gather/mask semantics against hand-placed scores,
    including a negative-sign column that must NOT contaminate the mean."""
    batch = make_batch([[3]], T=6, F=4, n_sae=8, seed=160)
    state = MutableConditioning(
        fids=t.zeros(1, 3, 4, dtype=t.int64),
        mags=t.zeros(1, 3, 4, dtype=t.int64),
        mask=t.zeros(1, 3, 4, dtype=t.bool),
        sign=t.zeros(1, 3, 4, dtype=t.float32),
    )
    state.fids[0, 1, 0] = 5
    state.sign[0, 1, 0] = 1.0
    state.fids[0, 1, 1] = 2
    state.sign[0, 1, 1] = 1.0
    state.fids[0, 2, 0] = 3
    state.sign[0, 2, 0] = -1.0  # negative slot column: excluded from the mean
    state.fids[0, 2, 1] = 7
    state.sign[0, 2, 1] = 1.0
    result = RolloutResult(
        conditioning=state,
        predicted=t.tensor([[False, True, True]]),
        k=t.tensor([[0, 2, 1]]),
        l=t.tensor([[0, 0, 1]]),
    )
    shifted = t.zeros(1, 3, 8)
    shifted[0, 1, 5] = 0.9
    shifted[0, 1, 2] = 0.4
    shifted[0, 2, 7] = 0.6
    shifted[0, 2, 3] = 0.99  # sits under the sign=-1 column; must not leak in

    metrics = _step_metrics(t.tensor(1.5), batch, result, shifted)
    assert metrics.mean_selected_pos_score == pytest.approx((0.9 + 0.4 + 0.6) / 3, rel=1e-6)
    assert metrics.lm_loss == pytest.approx(1.5)
    assert metrics.frac_predicted_slots == pytest.approx(2 / 3)
    assert metrics.mean_k == pytest.approx(1.5)
    assert metrics.mean_l == pytest.approx(0.5)
