"""Residual-head mathematics, legacy parity, upgrade, and optimizer contracts."""

from pathlib import Path

import pytest
import torch as t
import torch.nn.functional as F
from omegaconf import OmegaConf
from torchao.float8 import Float8LinearConfig, convert_to_float8_training

from nanochat.gpt import MLP, GPTConfig, Linear, OptimizerKind
from scripts.train.conditional_encdec_sft import _setup_sft_optimizer
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionBlock,
    FeaturePredictionConfig,
)
from steering.conditional.encoder_decoder.feature_prediction_head import (
    ResidualFeaturePredictionHead,
)
from steering.conditional.encoder_decoder.model import EncoderDecoderGPT
from steering.conditional.encoder_decoder.selfcond_training import (
    build_head_only_optimizer,
    freeze_for_head_only,
)
from steering.conditional.feature_vocab import FeatureVocab

_WIDTH = 64
_N_SAE = 32
_VOCAB = FeatureVocab(n_sae_features=_N_SAE, original_ids=(1, 3, 6, 12, 17, 25, 31))
_SFT_CONFIG_DIR = Path(__file__).resolve().parents[2] / "configs" / "training"


def _model(depth=0, *, compact=False, block=FeaturePredictionBlock.MLP):
    with t.device("cpu"):
        model = EncoderDecoderGPT(
            GPTConfig(
                sequence_len=16,
                vocab_size=64,
                n_layer=2,
                n_head=2,
                n_kv_head=2,
                n_embd=_WIDTH,
                window_pattern="L",
            ),
            EncoderConfig(n_embd=32, depth=1, n_head=2, n_kv_head=2, head_dim=16),
            CrossAttentionConfig(frequency=2),
            n_sae_features=_N_SAE,
            num_buckets=4,
            feature_pred_config=FeaturePredictionConfig(head_layers=depth, block=block),
            feature_vocab=_VOCAB if compact else None,
        )
        model.init_weights()
    return model


def _head(depth=2, *, width=16, device="cpu", dtype=t.float32):
    with t.device(device):
        projection = Linear(width, 7).to(dtype=dtype)
    return ResidualFeaturePredictionHead(projection, depth)


def _learned_head(head):
    """Ensure every branch contributes, so the oracle can detect missing work."""
    with t.no_grad():
        for index, block in enumerate(head.residual_blocks):
            block.c_proj.weight.normal_(mean=0.01 * (index + 1), std=0.03)
    return head


def _oracle(x, head):
    """Explicit algebra independent of the production block and norm functions."""
    for block in head.residual_blocks:
        normalized = F.rms_norm(x, (head.in_features,))
        hidden = normalized @ block.c_fc.weight.to(x.dtype).T
        hidden = hidden.clamp_min(0).pow(2)
        x = x + hidden @ block.c_proj.weight.to(x.dtype).T
    return F.linear(x, head.weight.to(x.dtype), head.bias.to(x.dtype))


@pytest.mark.parametrize("value", [True, False, -1, 1.0, 1.5, "2", None, [], {}])
def test_depth_rejects_coercion(value):
    with pytest.raises(ValueError, match="head_layers must be a nonnegative integer"):
        FeaturePredictionConfig(head_layers=value)


@pytest.mark.parametrize("depth", [0, 1, 2])
def test_depth_config_retains_exact_integer(depth):
    assert FeaturePredictionConfig(head_layers=depth).head_layers == depth
    assert FeaturePredictionConfig().head_layers == 0


def test_residual_class_requires_positive_depth():
    with pytest.raises(ValueError, match="requires positive head_layers"):
        _head(0)


def test_bias_free_projection_dtype_and_parameter_identity_are_preserved():
    projection = Linear(16, 7, bias=False).double()
    head = ResidualFeaturePredictionHead(projection, 2)
    x = t.randn(2, 16)
    assert head.bias is None
    assert "bias" not in head.state_dict()
    assert head.weight is projection.weight
    assert all(parameter.dtype == t.float64 for parameter in head.parameters())
    assert t.equal(head(x), projection(x))


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_zero_residual_is_exact_linear_and_preserves_rng_parameters_and_input(depth, dtype):
    t.manual_seed(19)
    projection = Linear(16, 7)
    projection.eval()
    x = t.randn(2, 5, 16).to(dtype).requires_grad_()
    original_x = x.detach().clone()
    expected = projection(x)
    rng_before = t.get_rng_state().clone()
    head = ResidualFeaturePredictionHead(projection, depth)
    assert t.equal(t.get_rng_state(), rng_before)
    assert head.weight is projection.weight
    assert head.bias is projection.bias
    assert not head.training and all(not block.training for block in head.residual_blocks)
    assert t.equal(head(x), expected)
    assert t.equal(x.detach(), original_x)
    expected_gradient = t.autograd.grad(expected.sum(), x, retain_graph=True)[0]
    assert t.equal(t.autograd.grad(head(x).sum(), x)[0], expected_gradient)


@pytest.mark.parametrize("depth", [1, 2])
def test_architecture_and_state_registration(depth):
    head = _head(depth)
    assert not isinstance(head, t.nn.Linear)
    assert head.in_features == 16 and head.out_features == 7
    assert len(head.residual_blocks) == depth
    assert len({id(block) for block in head.residual_blocks}) == depth
    expected_keys = {"weight", "bias"}
    for index, block in enumerate(head.residual_blocks):
        assert type(block) is MLP
        assert block.c_fc.weight.shape == (64, 16)
        assert block.c_proj.weight.shape == (16, 64)
        assert block.c_fc.bias is None and block.c_proj.bias is None
        assert t.count_nonzero(block.c_fc.weight) > 0
        assert block.c_fc.weight.abs().max() <= (3 / 16) ** 0.5
        assert t.count_nonzero(block.c_proj.weight) == 0
        expected_keys.update(
            {f"residual_blocks.{index}.c_fc.weight", f"residual_blocks.{index}.c_proj.weight"}
        )
    assert set(head.state_dict()) == expected_keys
    ids = [id(parameter) for _, parameter in head.named_parameters(remove_duplicate=False)]
    assert len(ids) == len(set(ids)) == 2 + 2 * depth
    assert sum(p.numel() for p in head.parameters()) == 7 * 17 + depth * 8 * 16**2


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_learned_residual_output_and_input_gradient_match_independent_algebra(depth, dtype):
    t.manual_seed(72)
    head = _learned_head(_head(depth))
    x = t.randn(2, 5, 16).to(dtype).requires_grad_()
    expected = _oracle(x, head)
    actual = head(x)
    assert t.equal(actual, expected)
    actual_gradient = t.autograd.grad(actual.sum(), x)[0]
    expected_gradient = t.autograd.grad(expected.sum(), x)[0]
    assert t.equal(actual_gradient, expected_gradient)
    assert not t.equal(actual, F.linear(x, head.weight.to(dtype), head.bias.to(dtype)))


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("compact", [False, True])
def test_model_construction_preserves_legacy_tensors_and_rng(depth, compact):
    t.manual_seed(23)
    baseline = _model(0, compact=compact)
    rng_before = t.get_rng_state().clone()
    t.manual_seed(23)
    candidate = _model(depth, compact=compact)
    assert t.equal(t.get_rng_state(), rng_before)
    for name, expected in baseline.state_dict().items():
        assert t.equal(candidate.state_dict()[name], expected), name
    if depth == 0:
        assert type(candidate.feature_pred_head) is Linear
        assert set(candidate.state_dict()) == set(baseline.state_dict())
    else:
        assert type(candidate.feature_pred_head) is ResidualFeaturePredictionHead
    assert candidate.feature_pred_head.out_features == (_VOCAB.size if compact else _N_SAE)
    baseline.assert_vocab_consistent()
    candidate.assert_vocab_consistent()


@pytest.mark.parametrize("depth", [1, 2])
@pytest.mark.parametrize("compact", [False, True])
def test_wrapping_projection_preserves_all_source_tensors_parameter_identity_and_rng(depth, compact):
    model = _model(compact=compact)
    original_parameters = dict(model.named_parameters())
    original_state = {name: value.clone() for name, value in model.state_dict().items()}
    original_vocab = model.feature_vocab
    x = t.randn(2, 3, _WIDTH)
    expected = model.feature_pred_head(x)
    rng_before = t.get_rng_state().clone()
    model.feature_pred_head = ResidualFeaturePredictionHead(model.feature_pred_head, depth)
    assert t.equal(t.get_rng_state(), rng_before)
    assert model.feature_vocab is original_vocab
    assert t.equal(model.feature_pred_head(x), expected)
    for name, parameter in original_parameters.items():
        assert dict(model.named_parameters())[name] is parameter, name
        assert t.equal(model.state_dict()[name], original_state[name]), name
    model.assert_vocab_consistent()


@pytest.mark.parametrize("depth", [0, 1, 3])
def test_strict_loading_rejects_architecture_depth_mismatch(depth):
    state = _model(2).state_dict()
    with pytest.raises(RuntimeError, match="(Unexpected|Missing) key"):
        _model(depth).load_state_dict(state, strict=True)


@pytest.mark.parametrize("key", ["weight", "bias", "residual_blocks.1.c_proj.weight"])
def test_strict_loading_rejects_incomplete_head(key):
    state = _model(2).state_dict()
    del state[f"feature_pred_head.{key}"]
    with pytest.raises(RuntimeError, match="Missing key"):
        _model(2).load_state_dict(state, strict=True)


def test_head_only_training_freezes_trunk_and_includes_every_residual_parameter():
    model = _model(2)
    freeze_for_head_only(model)
    head_ids = {id(p) for p in model.feature_pred_head.parameters()}
    assert {id(p) for p in model.parameters() if p.requires_grad} == head_ids
    optimizer = build_head_only_optimizer(model, unembedding_lr=0.004, adam_betas=(0.8, 0.95))
    assert {id(p) for group in optimizer.param_groups for p in group["params"]} == head_ids
    trainable_optimizer = model.setup_optimizer(trainable_only=True)
    assert {id(p) for group in trainable_optimizer.param_groups for p in group["params"]} == head_ids
    trunk_before = {name: p.detach().clone() for name, p in model.named_parameters() if id(p) not in head_ids}
    F.mse_loss(model.feature_pred_head(t.randn(2, _WIDTH)), t.ones(2, _N_SAE)).backward()
    optimizer.step()
    for name, expected in trunk_before.items():
        assert t.equal(dict(model.named_parameters())[name], expected), name


@pytest.mark.parametrize("depth", [0, 1, 2])
@pytest.mark.parametrize("compact", [False, True])
def test_parameter_and_flop_counts_include_each_weight_matrix_only(depth, compact):
    baseline = _model(0, compact=compact)
    model = _model(depth, compact=compact)
    added = depth * 8 * _WIDTH**2
    output_width = _VOCAB.size if compact else _N_SAE
    assert model.num_scaling_params()["feature_pred_head"] == output_width * (_WIDTH + 1) + added
    assert model.num_scaling_params()["total"] - baseline.num_scaling_params()["total"] == added
    assert model.estimate_flops(3.0, 4) - baseline.estimate_flops(3.0, 4) == 6 * added


def test_float8_conversion_cannot_replace_nonlinear_head():
    head = _learned_head(_head(width=64))
    projection_weight = head.weight
    residual_weight = head.residual_blocks[1].c_proj.weight
    converted = convert_to_float8_training(head, config=Float8LinearConfig())
    assert converted is head
    assert type(converted) is ResidualFeaturePredictionHead
    assert converted.weight is projection_weight
    assert converted.residual_blocks[1].c_proj.weight is residual_weight
    assert len(converted.residual_blocks) == 2


@pytest.mark.slow
@pytest.mark.skipif(not t.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [t.float32, t.bfloat16])
def test_cuda_upgrade_preserves_cpu_and_device_rng_and_exact_logits(dtype):
    with t.device("cuda"):
        projection = Linear(64, 32)
        x = t.randn(2, 5, 64).to(dtype)
    expected = projection(x)
    cpu_state = t.get_rng_state().clone()
    cuda_state = t.cuda.get_rng_state().clone()
    head = ResidualFeaturePredictionHead(projection, 2)
    assert t.equal(t.get_rng_state(), cpu_state)
    assert t.equal(t.cuda.get_rng_state(), cuda_state)
    assert t.equal(head(x), expected)
    assert all(p.device == projection.weight.device for p in head.parameters())


def _group_for_parameter(optimizer, parameter):
    groups = [
        group
        for group in optimizer.param_groups
        if any(candidate is parameter for candidate in group["params"])
    ]
    assert len(groups) == 1
    return groups[0]


def _head_block_matrices(model):
    head = model.feature_pred_head
    assert isinstance(head, ResidualFeaturePredictionHead)
    return [parameter for parameter in head.residual_blocks.parameters() if parameter.ndim == 2]


@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
@t.random.fork_rng()
def test_residual_head_block_matrices_use_muon_and_projection_uses_adamw(block):
    model = _model(2, block=block)
    optimizer = model.setup_optimizer()
    head = model.feature_pred_head
    assert isinstance(head, ResidualFeaturePredictionHead)
    block_matrices = _head_block_matrices(model)
    assert len(block_matrices) == (4 if block is FeaturePredictionBlock.MLP else 8)

    scale = (_WIDTH / 768) ** -0.5
    assert head.bias is not None
    projection_params = [head.weight, head.bias]
    for parameter in projection_params:
        group = _group_for_parameter(optimizer, parameter)
        assert group["kind"] == OptimizerKind.ADAMW
        assert group["lr"] == pytest.approx(0.004 * scale)
    assert {
        id(parameter) for parameter in _group_for_parameter(optimizer, head.weight)["params"]
    } == {id(parameter) for parameter in projection_params}

    for parameter in block_matrices:
        group = _group_for_parameter(optimizer, parameter)
        assert group["kind"] == OptimizerKind.MUON
        assert group["lr"] == pytest.approx(0.02)
        assert group["momentum"] == pytest.approx(0.95)
        assert group["ns_steps"] == 5
        assert group["beta2"] == pytest.approx(0.95)
        assert group["weight_decay"] == pytest.approx(0.0)
        assert not any(
            candidate is parameter
            for adamw_group in optimizer.param_groups
            if adamw_group["kind"] == OptimizerKind.ADAMW
            for candidate in adamw_group["params"]
        )

    grouped_head = [
        parameter
        for group in optimizer.param_groups
        for parameter in group["params"]
        if any(parameter is head_parameter for head_parameter in head.parameters())
    ]
    assert len(grouped_head) == len({id(parameter) for parameter in grouped_head})
    assert {id(parameter) for parameter in grouped_head} == {
        id(parameter) for parameter in head.parameters()
    }


@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
@t.random.fork_rng()
def test_fp_head_matrix_lr_override_moves_only_residual_block_matrices(block):
    model = _model(2, block=block)
    baseline = model.setup_optimizer()
    overridden = model.setup_optimizer(fp_head_matrix_lr=0.005)
    block_matrix_ids = {id(parameter) for parameter in _head_block_matrices(model)}

    for parameter in model.parameters():
        before = _group_for_parameter(baseline, parameter)
        after = _group_for_parameter(overridden, parameter)
        assert after["kind"] == before["kind"]
        if id(parameter) in block_matrix_ids:
            assert before["lr"] == pytest.approx(0.02)
            assert after["lr"] == pytest.approx(0.005)
        else:
            assert after["lr"] == pytest.approx(before["lr"])


@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
@t.random.fork_rng()
def test_depth_zero_head_retains_single_adamw_group(block):
    model = _model(0, block=block)
    optimizer = model.setup_optimizer(fp_head_matrix_lr=0.005)
    head = model.feature_pred_head
    head_ids = {id(parameter) for parameter in head.parameters()}
    head_groups = [
        group
        for group in optimizer.param_groups
        if any(id(parameter) in head_ids for parameter in group["params"])
    ]
    assert len(head_groups) == 1
    assert head_groups[0]["kind"] == OptimizerKind.ADAMW
    assert {id(parameter) for parameter in head_groups[0]["params"]} == head_ids
    assert not any(
        group["kind"] == OptimizerKind.MUON
        and any(id(parameter) in head_ids for parameter in group["params"])
        for group in optimizer.param_groups
    )


@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
@t.random.fork_rng()
def test_trainable_only_head_yields_projection_adamw_and_block_muon_groups(block):
    model = _model(2, block=block)
    model.requires_grad_(False)
    model.feature_pred_head.requires_grad_(True)
    optimizer = model.setup_optimizer(trainable_only=True)
    head = model.feature_pred_head
    head_ids = {id(parameter) for parameter in head.parameters()}
    grouped = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(grouped) == len({id(parameter) for parameter in grouped}) == len(head_ids)
    assert {id(parameter) for parameter in grouped} == head_ids
    assert _group_for_parameter(optimizer, head.weight)["kind"] == OptimizerKind.ADAMW
    assert all(
        _group_for_parameter(optimizer, parameter)["kind"] == OptimizerKind.MUON
        for parameter in _head_block_matrices(model)
    )


@pytest.mark.parametrize("head_layers", [0, 2])
@pytest.mark.parametrize("block", list(FeaturePredictionBlock))
def test_sft_optimizer_fp_head_lr_moves_only_the_projection_group(head_layers, block):
    """SFT ``fp_head_lr`` moves the retained projection without moving block matrices."""
    base = {
        "unembedding_lr": 0.004,
        "embedding_lr": 0.3,
        "matrix_lr": 0.02,
        "fp_head_matrix_lr": 0.02,
        "weight_decay": 0.0,
        "encoder_embedding_lr": 0.2,
        "encoder_matrix_lr": 0.02,
        "xattn_matrix_lr": 0.02,
        "lr_multiplier": 2.5,
        "init_lr_frac": 0.3,
    }

    def group_lrs(fp_head_lr):
        model = _model(head_layers, block=block)
        optimizer = _setup_sft_optimizer(model, OmegaConf.create(base | {"fp_head_lr": fp_head_lr}))
        head = model.feature_pred_head
        projection = _group_for_parameter(optimizer, head.weight)
        lm = [
            g for g in optimizer.param_groups if any(p is model.lm_head.weight for p in g["params"])
        ]
        assert all(group["initial_lr"] == group["lr"] for group in optimizer.param_groups)
        matrix_lrs = (
            {
                _group_for_parameter(optimizer, parameter)["lr"]
                for parameter in _head_block_matrices(model)
            }
            if head_layers
            else set()
        )
        return projection["lr"], lm[0]["lr"], matrix_lrs

    shared_head, shared_lm, shared_matrices = group_lrs(None)
    own_head, own_lm, own_matrices = group_lrs(0.04)
    assert shared_head == pytest.approx(shared_lm)
    assert own_head == pytest.approx(10 * shared_head)
    assert own_head == pytest.approx(0.04 * (_WIDTH / 768) ** -0.5 * 2.5 * 0.3)
    assert own_lm == pytest.approx(shared_lm)
    assert own_matrices == shared_matrices
    if head_layers:
        assert len(own_matrices) == 1
        assert next(iter(own_matrices)) == pytest.approx(0.02 * 2.5 * 0.3)
    for invalid in (0.0, -1, True, float("nan"), float("inf"), "0.04"):
        with pytest.raises(TypeError, match="Could not parse.*PositiveFiniteNumber"):
            group_lrs(invalid)


@t.random.fork_rng()
def test_sft_config_defaults_and_scales_fp_head_matrix_lr(hydra_compose):
    cfg = hydra_compose(str(_SFT_CONFIG_DIR), "conditional_encdec_sft")
    assert cfg.fp_head_matrix_lr == pytest.approx(0.02)
    user_config = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(user_config, dict)
    assert user_config["fp_head_matrix_lr"] == pytest.approx(0.02)
    cfg.lr_multiplier = 2.5
    cfg.init_lr_frac = 0.3
    model = _model(2, block=FeaturePredictionBlock.TRANSFORMER)
    optimizer = _setup_sft_optimizer(model, cfg)
    matrix_lrs = {
        _group_for_parameter(optimizer, parameter)["lr"]
        for parameter in _head_block_matrices(model)
    }
    assert len(matrix_lrs) == 1
    assert next(iter(matrix_lrs)) == pytest.approx(0.02 * 2.5 * 0.3)
