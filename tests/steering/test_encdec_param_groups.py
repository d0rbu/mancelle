"""Optimizer parameter-group tests for EncoderDecoderGPT.setup_optimizer.

Verifies that every model parameter lands in exactly one optimizer group,
that each group uses the correct optimizer kind (AdamW vs Muon), that LR
scaling follows the documented rules, and that Muon groups respect the
per-shape invariant.

Also verifies that the training script correctly applies batch_lr_scale
to encoder and cross-attention LRs.
"""

from collections import Counter

import pytest
import torch as t

from nanochat.gpt import GPTConfig, OptimizerKind
from nanochat.optim import MuonAdamW
from steering.conditional.encoder_decoder.config import (
    CrossAttentionConfig,
    EncoderConfig,
    FeaturePredictionConfig,
    FeaturePredLossFn,
    FeaturePredLossMode,
)
from steering.conditional.encoder_decoder.model import (
    BASE_MODEL_DIM,
    EncoderDecoderGPT,
    _coalesce_compatible_muon_groups,
)
from tests.encdec_test_helpers import build_tiny_model

# ── Small model dimensions (matches test_encdec_fp8.py) ──────────────────
DEC_N_EMBD = 256
DEC_N_LAYER = 4
DEC_N_HEAD = 4
HEAD_DIM = 64
ENC_N_EMBD = 128
ENC_DEPTH = 2
XATTN_FREQ = 2  # cross-attention at decoder layers 1 and 3
N_SAE_FEATURES = 16384
NUM_BUCKETS = 10
VOCAB_SIZE = 256
T = 128

# ── Default LR values used by setup_optimizer ─────────────────────────────
DEFAULT_UNEMBEDDING_LR = 0.004
DEFAULT_EMBEDDING_LR = 0.2
DEFAULT_MATRIX_LR = 0.02
DEFAULT_SCALAR_LR = 0.5
DEFAULT_ENCODER_EMBEDDING_LR = 0.2
DEFAULT_ENCODER_MATRIX_LR = 0.02
DEFAULT_XATTN_MATRIX_LR = 0.02
DEFAULT_WEIGHT_DECAY = 0.0
DEFAULT_ADAM_BETAS = (0.8, 0.95)

DMODEL_LR_SCALE = (DEC_N_EMBD / 768) ** -0.5


def _muon_group(params, *, lr=0.02):
    return {
        "kind": OptimizerKind.MUON,
        "params": params,
        "lr": lr,
        "momentum": 0.95,
        "ns_steps": 5,
        "beta2": 0.95,
        "weight_decay": 0.0,
    }


def test_muon_coalescing_requires_identical_shape_and_hyperparameters():
    same_a = t.nn.Parameter(t.empty(2, 3))
    same_b = t.nn.Parameter(t.empty(2, 3))
    other_lr = t.nn.Parameter(t.empty(2, 3))
    other_shape = t.nn.Parameter(t.empty(3, 2))

    groups = _coalesce_compatible_muon_groups(
        [
            _muon_group([same_a]),
            _muon_group([same_b]),
            _muon_group([other_lr], lr=0.1),
            _muon_group([other_shape]),
        ]
    )

    assert [[id(parameter) for parameter in group["params"]] for group in groups] == [
        [id(same_a), id(same_b)],
        [id(other_lr)],
        [id(other_shape)],
    ]


def test_muon_coalescing_preserves_two_step_parameter_updates():
    baseline_params = [
        t.nn.Parameter(t.tensor([[0.1, -0.2], [0.3, -0.4]])),
        t.nn.Parameter(t.tensor([[0.5, -0.6], [0.7, -0.8]])),
    ]
    candidate_params = [t.nn.Parameter(parameter.detach().clone()) for parameter in baseline_params]
    baseline = MuonAdamW([_muon_group([parameter]) for parameter in baseline_params])
    candidate = MuonAdamW(
        _coalesce_compatible_muon_groups(
            [_muon_group([parameter]) for parameter in candidate_params]
        )
    )

    for step in range(2):
        for index, (baseline_param, candidate_param) in enumerate(
            zip(baseline_params, candidate_params, strict=True)
        ):
            grad = t.full_like(baseline_param, (step + 1) * (index + 1) * 0.01)
            baseline_param.grad = grad.clone()
            candidate_param.grad = grad.clone()
        baseline.step()
        candidate.step()

    for baseline_param, candidate_param in zip(baseline_params, candidate_params, strict=True):
        t.testing.assert_close(candidate_param, baseline_param, rtol=0, atol=0)


# ── Fixtures ──────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def model():
    """Build a small EncoderDecoderGPT on CPU and initialise weights."""
    dec_config = GPTConfig(
        sequence_len=T,
        vocab_size=VOCAB_SIZE,
        n_layer=DEC_N_LAYER,
        n_head=DEC_N_HEAD,
        n_kv_head=DEC_N_HEAD,
        n_embd=DEC_N_EMBD,
        window_pattern="L",
    )
    enc_config = EncoderConfig(n_embd=ENC_N_EMBD, depth=ENC_DEPTH, head_dim=HEAD_DIM)
    xattn_config = CrossAttentionConfig(frequency=XATTN_FREQ)

    m = EncoderDecoderGPT(
        dec_config,
        enc_config,
        xattn_config,
        n_sae_features=N_SAE_FEATURES,
        num_buckets=NUM_BUCKETS,
    )
    m.init_weights()
    return m


@pytest.fixture(scope="module")
def optimizer_and_groups(model):
    """Return (optimizer, param_groups_raw) where param_groups_raw is the
    list of group dicts *before* the optimizer strips the ``kind`` key."""
    # Call with explicit values so tests are deterministic
    opt = model.setup_optimizer(
        unembedding_lr=DEFAULT_UNEMBEDDING_LR,
        embedding_lr=DEFAULT_EMBEDDING_LR,
        matrix_lr=DEFAULT_MATRIX_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        adam_betas=DEFAULT_ADAM_BETAS,
        scalar_lr=DEFAULT_SCALAR_LR,
        encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR,
        encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR,
        xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR,
    )
    return opt


# ── Helpers ───────────────────────────────────────────────────────────────


def _ptr_to_group_map(optimizer):
    """Map data_ptr -> list of (group_index, group_dict)."""
    mapping: dict[int, list[tuple[int, dict]]] = {}
    for idx, group in enumerate(optimizer.param_groups):
        for p in group["params"]:
            mapping.setdefault(p.data_ptr(), []).append((idx, group))
    return mapping


def _model_ptr_set(model):
    """Set of data_ptr() for every model parameter."""
    return {p.data_ptr() for p in model.parameters()}


def _named_param_ptr_map(model):
    """Map data_ptr -> parameter name."""
    return {p.data_ptr(): name for name, p in model.named_parameters()}


def _groups_for_prefix(optimizer, model, prefix):
    """Return optimizer groups that contain any parameter whose name starts with *prefix*."""
    name_map = _named_param_ptr_map(model)
    result = []
    for group in optimizer.param_groups:
        for p in group["params"]:
            pname = name_map.get(p.data_ptr(), "")
            if pname.startswith(prefix):
                result.append(group)
                break
    return result


def _params_for_prefix(model, prefix):
    """Return data_ptrs of parameters whose name starts with *prefix*."""
    return {p.data_ptr() for name, p in model.named_parameters() if name.startswith(prefix)}


# ══════════════════════════════════════════════════════════════════════════
# 1. Parameter Group Completeness
# ══════════════════════════════════════════════════════════════════════════


class TestParameterGroupCompleteness:
    """Every model parameter must appear in exactly one optimizer group."""

    def test_every_param_in_exactly_one_group(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        ptr_map = _ptr_to_group_map(opt)

        for name, param in model.named_parameters():
            ptr = param.data_ptr()
            occurrences = ptr_map.get(ptr, [])
            assert len(occurrences) == 1, (
                f"Parameter '{name}' (ptr={ptr}) appears in {len(occurrences)} "
                f"group(s), expected exactly 1"
            )

    def test_no_orphaned_parameters(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        grouped_ptrs = set()
        for group in opt.param_groups:
            for p in group["params"]:
                grouped_ptrs.add(p.data_ptr())

        orphans = []
        for name, param in model.named_parameters():
            if param.data_ptr() not in grouped_ptrs:
                orphans.append(name)
        assert len(orphans) == 0, f"Orphaned parameters (not in any group): {orphans}"

    def test_no_duplicated_parameters(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        ptr_counts = Counter()
        for group in opt.param_groups:
            for p in group["params"]:
                ptr_counts[p.data_ptr()] += 1

        name_map = _named_param_ptr_map(model)
        duplicates = [
            name_map.get(ptr, f"<ptr={ptr}>") for ptr, count in ptr_counts.items() if count > 1
        ]
        assert len(duplicates) == 0, f"Parameters in multiple groups: {duplicates}"

    def test_param_count_matches_model(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        grouped_count = sum(len(g["params"]) for g in opt.param_groups)
        model_count = sum(1 for _ in model.parameters())
        assert grouped_count == model_count, (
            f"Grouped param count ({grouped_count}) != model param count ({model_count})"
        )


# ══════════════════════════════════════════════════════════════════════════
# 2. Parameter Categorization (AdamW vs Muon)
# ══════════════════════════════════════════════════════════════════════════


class TestParameterCategorization:
    """Parameters must land in the correct optimizer kind based on their role."""

    def test_decoder_embeddings_use_adamw(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "transformer.wte")
        assert len(groups) > 0, "No optimizer group contains transformer.wte params"
        for g in groups:
            assert g["kind"] == OptimizerKind.ADAMW, (
                f"transformer.wte group has kind={g['kind']}, expected adamw"
            )

    def test_decoder_value_embeds_use_adamw(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "value_embeds")
        assert len(groups) > 0, "No optimizer group contains value_embeds params"
        for g in groups:
            assert g["kind"] == OptimizerKind.ADAMW, (
                f"value_embeds group has kind={g['kind']}, expected adamw"
            )

    def test_decoder_lm_head_uses_adamw(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "lm_head")
        assert len(groups) > 0, "No optimizer group contains lm_head params"
        for g in groups:
            assert g["kind"] == OptimizerKind.ADAMW, (
                f"lm_head group has kind={g['kind']}, expected adamw"
            )

    def test_decoder_scalars_use_adamw(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        for prefix in ("resid_lambdas", "x0_lambdas"):
            groups = _groups_for_prefix(opt, model, prefix)
            assert len(groups) > 0, f"No optimizer group contains {prefix} params"
            for g in groups:
                assert g["kind"] == OptimizerKind.ADAMW, (
                    f"{prefix} group has kind={g['kind']}, expected adamw"
                )

    def test_decoder_blocks_use_muon(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "transformer.h")
        assert len(groups) > 0, "No optimizer group contains transformer.h params"
        for g in groups:
            assert g["kind"] == OptimizerKind.MUON, (
                f"transformer.h group has kind={g['kind']}, expected muon"
            )

    def test_encoder_embeddings_use_adamw(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        for prefix in (
            "encoder.feature_embed",
            "encoder.magnitude_embed",
            "encoder.value_embeds",
        ):
            groups = _groups_for_prefix(opt, model, prefix)
            assert len(groups) > 0, f"No optimizer group contains {prefix} params"
            for g in groups:
                assert g["kind"] == OptimizerKind.ADAMW, (
                    f"{prefix} group has kind={g['kind']}, expected adamw"
                )

    def test_encoder_blocks_use_muon(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "encoder.blocks")
        assert len(groups) > 0, "No optimizer group contains encoder.blocks params"
        for g in groups:
            assert g["kind"] == OptimizerKind.MUON, (
                f"encoder.blocks group has kind={g['kind']}, expected muon"
            )

    def test_cross_attention_uses_muon(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "cross_attn_layers")
        assert len(groups) > 0, "No optimizer group contains cross_attn_layers params"
        for g in groups:
            assert g["kind"] == OptimizerKind.MUON, (
                f"cross_attn_layers group has kind={g['kind']}, expected muon"
            )


# ══════════════════════════════════════════════════════════════════════════
# 3. LR Scaling
# ══════════════════════════════════════════════════════════════════════════


class TestLRScaling:
    """Verify learning rate scaling rules for each group."""

    def test_adamw_groups_have_dmodel_lr_scale(self, model, optimizer_and_groups):
        """Non-scalar AdamW groups should have LRs scaled by (model_dim/768)**-0.5."""
        opt = optimizer_and_groups
        name_map = _named_param_ptr_map(model)

        # Map each group to a representative param name for identification
        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.ADAMW:
                continue
            rep_name = name_map.get(group["params"][0].data_ptr(), "")

            # Skip scalar groups (resid_lambdas, x0_lambdas) -- they have custom LR
            if "resid_lambdas" in rep_name or "x0_lambdas" in rep_name:
                continue

            lr = group["lr"]
            # The LR should be base_lr * dmodel_lr_scale for some base_lr
            # Check that lr / dmodel_lr_scale yields a recognisable base LR
            base_lr = lr / DMODEL_LR_SCALE
            # Must be one of the known base LRs
            known_base_lrs = {
                DEFAULT_UNEMBEDDING_LR,
                DEFAULT_EMBEDDING_LR,
                DEFAULT_ENCODER_EMBEDDING_LR,
            }
            assert any(abs(base_lr - k) < 1e-10 for k in known_base_lrs), (
                f"Group with representative param '{rep_name}' has lr={lr}, "
                f"which gives base_lr={base_lr} (dmodel_lr_scale={DMODEL_LR_SCALE}). "
                f"Expected one of {known_base_lrs}"
            )

    def test_muon_groups_no_dmodel_lr_scale(self, model, optimizer_and_groups):
        """Muon groups use the raw LR without dmodel_lr_scale."""
        opt = optimizer_and_groups
        name_map = _named_param_ptr_map(model)
        known_muon_lrs = {
            DEFAULT_MATRIX_LR,
            DEFAULT_ENCODER_MATRIX_LR,
            DEFAULT_XATTN_MATRIX_LR,
        }
        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.MUON:
                continue
            lr = group["lr"]
            rep_name = name_map.get(group["params"][0].data_ptr(), "")
            assert any(abs(lr - k) < 1e-10 for k in known_muon_lrs), (
                f"Muon group with representative param '{rep_name}' has lr={lr}, "
                f"expected one of {known_muon_lrs} (raw, without dmodel_lr_scale)"
            )

    def test_x0_lambdas_use_special_betas(self, model, optimizer_and_groups):
        """x0_lambdas group must use betas=(0.96, 0.95)."""
        opt = optimizer_and_groups
        groups = _groups_for_prefix(opt, model, "x0_lambdas")
        assert len(groups) == 1, f"Expected 1 group for x0_lambdas, got {len(groups)}"
        betas = groups[0]["betas"]
        assert betas == (0.96, 0.95), f"x0_lambdas betas={betas}, expected (0.96, 0.95)"

    def test_all_adamw_groups_no_weight_decay(self, model, optimizer_and_groups):
        """All AdamW groups should have weight_decay=0.0."""
        opt = optimizer_and_groups
        for idx, group in enumerate(opt.param_groups):
            if group["kind"] != OptimizerKind.ADAMW:
                continue
            assert group["weight_decay"] == 0.0, (
                f"AdamW group {idx} has weight_decay={group['weight_decay']}, expected 0.0"
            )

    def test_muon_groups_have_weight_decay(self, model, optimizer_and_groups):
        """Muon groups should carry the passed weight_decay value."""
        opt = optimizer_and_groups
        for idx, group in enumerate(opt.param_groups):
            if group["kind"] != OptimizerKind.MUON:
                continue
            assert group["weight_decay"] == DEFAULT_WEIGHT_DECAY, (
                f"Muon group {idx} has weight_decay={group['weight_decay']}, "
                f"expected {DEFAULT_WEIGHT_DECAY}"
            )


# ══════════════════════════════════════════════════════════════════════════
# 4. Muon Group Shape Invariant
# ══════════════════════════════════════════════════════════════════════════


class TestMuonGroupShapeInvariant:
    """Each Muon group must contain parameters of a single shape,
    and every unique shape must have its own group."""

    def test_each_muon_group_has_uniform_shape(self, model, optimizer_and_groups):
        opt = optimizer_and_groups
        for idx, group in enumerate(opt.param_groups):
            if group["kind"] != OptimizerKind.MUON:
                continue
            shapes = {p.shape for p in group["params"]}
            assert len(shapes) == 1, f"Muon group {idx} has mixed shapes: {shapes}"

    def test_muon_groups_cover_all_shapes(self, model, optimizer_and_groups):
        """For decoder blocks, encoder blocks, and cross-attention, every unique
        parameter shape must have a dedicated Muon group."""
        opt = optimizer_and_groups
        name_map = _named_param_ptr_map(model)

        # Collect expected shapes per component
        component_prefixes = {
            "decoder_blocks": "transformer.h",
            "encoder_blocks": "encoder.blocks",
            "cross_attention": "cross_attn_layers",
        }

        for component_name, prefix in component_prefixes.items():
            # Expected: set of unique shapes for this component
            expected_shapes = set()
            for name, param in model.named_parameters():
                if name.startswith(prefix):
                    expected_shapes.add(param.shape)

            if not expected_shapes:
                continue  # skip if component has no parameters

            # Actual: shapes covered by Muon groups for this component
            covered_shapes = set()
            for group in opt.param_groups:
                if group["kind"] != OptimizerKind.MUON:
                    continue
                # Check if any param in this group belongs to the component
                group_belongs_to_component = False
                for p in group["params"]:
                    pname = name_map.get(p.data_ptr(), "")
                    if pname.startswith(prefix):
                        group_belongs_to_component = True
                        break
                if group_belongs_to_component:
                    for p in group["params"]:
                        covered_shapes.add(p.shape)

            missing = expected_shapes - covered_shapes
            assert not missing, f"{component_name}: shapes {missing} have no Muon group"


# ══════════════════════════════════════════════════════════════════════════
# 5. Batch LR Scale (encoder & cross-attention)
# ══════════════════════════════════════════════════════════════════════════


class TestBatchLRScaleBug:
    """Verifies that the batch_lr_scale fix in scripts/conditional_encdec_train.py
    correctly passes encoder_embedding_lr, encoder_matrix_lr, and xattn_matrix_lr
    with batch_lr_scale applied."""

    def test_encoder_lr_not_batch_scaled(self, model):
        """Simulate the training script call with batch_lr_scale=2.0.

        After the fix, the training script passes encoder and cross-attention LRs
        multiplied by batch_lr_scale, just like decoder LRs.
        """
        batch_lr_scale = 2.0

        # This mirrors the fixed training script (encoder/xattn args included)
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            adam_betas=DEFAULT_ADAM_BETAS,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        name_map = _named_param_ptr_map(model)

        # Encoder embedding LR should be scaled
        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.ADAMW:
                continue
            rep_name = name_map.get(group["params"][0].data_ptr(), "")
            if "encoder.feature_embed" in rep_name or "encoder.magnitude_embed" in rep_name:
                expected_lr = DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
                actual_lr = group["lr"]
                assert abs(actual_lr - expected_lr) < 1e-10, (
                    f"Encoder embedding LR should be {expected_lr} "
                    f"(base * batch_lr_scale * dmodel_scale) but is {actual_lr}"
                )

        # Encoder matrix LR should be scaled
        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.MUON:
                continue
            rep_name = name_map.get(group["params"][0].data_ptr(), "")
            if rep_name.startswith("encoder.blocks"):
                expected_lr = DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale
                actual_lr = group["lr"]
                assert abs(actual_lr - expected_lr) < 1e-10, (
                    f"Encoder matrix LR should be {expected_lr} "
                    f"(base * batch_lr_scale) but is {actual_lr}"
                )

        # Cross-attention matrix LR should be scaled
        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.MUON:
                continue
            rep_name = name_map.get(group["params"][0].data_ptr(), "")
            if rep_name.startswith("cross_attn_layers"):
                expected_lr = DEFAULT_XATTN_MATRIX_LR * batch_lr_scale
                actual_lr = group["lr"]
                assert abs(actual_lr - expected_lr) < 1e-10, (
                    f"Cross-attention matrix LR should be {expected_lr} "
                    f"(base * batch_lr_scale) but is {actual_lr}"
                )


# ══════════════════════════════════════════════════════════════════════════
# 6. Batch LR Scale — Parametrized Verification
# ══════════════════════════════════════════════════════════════════════════


class TestBatchLRScaleParametrized:
    """Verify that ALL parameter groups have correct LRs when batch_lr_scale
    is applied uniformly to every base LR passed to setup_optimizer."""

    @pytest.mark.parametrize("batch_lr_scale", [1.0, 2.0, 0.5, 0.25])
    def test_all_group_lrs_correct(self, model, batch_lr_scale):
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            adam_betas=DEFAULT_ADAM_BETAS,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        # 1. Decoder lm_head LR
        lm_head_groups = _groups_for_prefix(opt, model, "lm_head")
        assert len(lm_head_groups) > 0
        expected = DEFAULT_UNEMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
        for g in lm_head_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"lm_head lr={g['lr']}, expected {expected} (batch_lr_scale={batch_lr_scale})"
            )

        # 2. Decoder embedding (wte) LR
        wte_groups = _groups_for_prefix(opt, model, "transformer.wte")
        assert len(wte_groups) > 0
        expected = DEFAULT_EMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
        for g in wte_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"wte lr={g['lr']}, expected {expected} (batch_lr_scale={batch_lr_scale})"
            )

        # 3. Decoder value_embeds LR
        ve_groups = _groups_for_prefix(opt, model, "value_embeds")
        assert len(ve_groups) > 0
        expected = DEFAULT_EMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
        for g in ve_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"value_embeds lr={g['lr']}, expected {expected} (batch_lr_scale={batch_lr_scale})"
            )

        # 4. Decoder resid_lambdas LR (scalar_lr * 0.01)
        resid_groups = _groups_for_prefix(opt, model, "resid_lambdas")
        assert len(resid_groups) == 1
        expected = DEFAULT_SCALAR_LR * batch_lr_scale * 0.01
        assert resid_groups[0]["lr"] == pytest.approx(expected), (
            f"resid_lambdas lr={resid_groups[0]['lr']}, expected {expected} "
            f"(batch_lr_scale={batch_lr_scale})"
        )

        # 5. Decoder x0_lambdas LR (scalar_lr, no 0.01 factor)
        x0_groups = _groups_for_prefix(opt, model, "x0_lambdas")
        assert len(x0_groups) == 1
        expected = DEFAULT_SCALAR_LR * batch_lr_scale
        assert x0_groups[0]["lr"] == pytest.approx(expected), (
            f"x0_lambdas lr={x0_groups[0]['lr']}, expected {expected} "
            f"(batch_lr_scale={batch_lr_scale})"
        )

        # 6. Decoder matrix (transformer.h) LR — Muon, no dmodel_lr_scale
        dec_matrix_groups = _groups_for_prefix(opt, model, "transformer.h")
        assert len(dec_matrix_groups) > 0
        expected = DEFAULT_MATRIX_LR * batch_lr_scale
        for g in dec_matrix_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"transformer.h lr={g['lr']}, expected {expected} (batch_lr_scale={batch_lr_scale})"
            )

        # 7. Encoder embedding LR
        enc_embed_groups = _groups_for_prefix(opt, model, "encoder.feature_embed")
        enc_embed_groups += _groups_for_prefix(opt, model, "encoder.magnitude_embed")
        enc_embed_groups += _groups_for_prefix(opt, model, "encoder.value_embeds")
        # Deduplicate: all encoder embeddings share one group
        seen_ids = set()
        unique_enc_embed_groups = []
        for g in enc_embed_groups:
            gid = id(g)
            if gid not in seen_ids:
                seen_ids.add(gid)
                unique_enc_embed_groups.append(g)
        assert len(unique_enc_embed_groups) > 0
        expected = DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
        for g in unique_enc_embed_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"encoder embedding lr={g['lr']}, expected {expected} "
                f"(batch_lr_scale={batch_lr_scale})"
            )

        # 8. Encoder matrix LR — Muon, no dmodel_lr_scale
        enc_matrix_groups = _groups_for_prefix(opt, model, "encoder.blocks")
        assert len(enc_matrix_groups) > 0
        expected = DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale
        for g in enc_matrix_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"encoder.blocks lr={g['lr']}, expected {expected} "
                f"(batch_lr_scale={batch_lr_scale})"
            )

        # 9. Cross-attention matrix LR — Muon, no dmodel_lr_scale
        xattn_groups = _groups_for_prefix(opt, model, "cross_attn_layers")
        assert len(xattn_groups) > 0
        expected = DEFAULT_XATTN_MATRIX_LR * batch_lr_scale
        for g in xattn_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"cross_attn_layers lr={g['lr']}, expected {expected} "
                f"(batch_lr_scale={batch_lr_scale})"
            )


# ══════════════════════════════════════════════════════════════════════════
# 7. Non-Default Base LRs
# ══════════════════════════════════════════════════════════════════════════


class TestNonDefaultBaseLRs:
    """Verify that custom (non-default) base LRs are correctly applied,
    confirming the values passed to setup_optimizer actually take effect
    rather than silently falling back to defaults."""

    def test_custom_encoder_embedding_lr(self, model):
        custom_lr = 0.1  # not the default 0.2
        batch_lr_scale = 2.0
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=custom_lr * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )
        expected = custom_lr * batch_lr_scale * DMODEL_LR_SCALE
        enc_embed_groups = _groups_for_prefix(opt, model, "encoder.feature_embed")
        assert len(enc_embed_groups) > 0
        for g in enc_embed_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"encoder embedding lr={g['lr']}, expected {expected} "
                f"(custom_lr={custom_lr}, batch_lr_scale={batch_lr_scale})"
            )
            # Verify it differs from what the default would produce
            default_lr = DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale * DMODEL_LR_SCALE
            assert g["lr"] != pytest.approx(default_lr), (
                f"encoder embedding lr={g['lr']} matches default {default_lr} — "
                f"custom_lr={custom_lr} was ignored"
            )

    def test_custom_encoder_matrix_lr(self, model):
        custom_lr = 0.05  # not the default 0.02
        batch_lr_scale = 2.0
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=custom_lr * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )
        expected = custom_lr * batch_lr_scale
        enc_matrix_groups = _groups_for_prefix(opt, model, "encoder.blocks")
        assert len(enc_matrix_groups) > 0
        for g in enc_matrix_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"encoder.blocks lr={g['lr']}, expected {expected} "
                f"(custom_lr={custom_lr}, batch_lr_scale={batch_lr_scale})"
            )
            default_lr = DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale
            assert g["lr"] != pytest.approx(default_lr), (
                f"encoder.blocks lr={g['lr']} matches default {default_lr} — "
                f"custom_lr={custom_lr} was ignored"
            )

    def test_custom_xattn_matrix_lr(self, model):
        custom_lr = 0.01  # not the default 0.02
        batch_lr_scale = 2.0
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=custom_lr * batch_lr_scale,
        )
        expected = custom_lr * batch_lr_scale
        xattn_groups = _groups_for_prefix(opt, model, "cross_attn_layers")
        assert len(xattn_groups) > 0
        for g in xattn_groups:
            assert g["lr"] == pytest.approx(expected), (
                f"cross_attn_layers lr={g['lr']}, expected {expected} "
                f"(custom_lr={custom_lr}, batch_lr_scale={batch_lr_scale})"
            )
            default_lr = DEFAULT_XATTN_MATRIX_LR * batch_lr_scale
            assert g["lr"] != pytest.approx(default_lr), (
                f"cross_attn_layers lr={g['lr']} matches default {default_lr} — "
                f"custom_lr={custom_lr} was ignored"
            )


# ══════════════════════════════════════════════════════════════════════════
# 8. LR Proportionality — Ratio Invariant Under batch_lr_scale
# ══════════════════════════════════════════════════════════════════════════


class TestLRProportionality:
    """For any batch_lr_scale, the ratio between corresponding decoder and encoder
    LRs should equal the ratio of their base LRs. batch_lr_scale must cancel out."""

    @pytest.mark.parametrize("batch_lr_scale", [0.25, 0.5, 1.0, 2.0, 4.0])
    def test_matrix_lr_ratio_invariant(self, model, batch_lr_scale):
        """decoder_matrix_lr / encoder_matrix_lr should always equal
        DEFAULT_MATRIX_LR / DEFAULT_ENCODER_MATRIX_LR regardless of batch_lr_scale."""
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        name_map = _named_param_ptr_map(model)
        dec_matrix_lr = None
        enc_matrix_lr = None
        xattn_lr = None

        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.MUON:
                continue
            names = [name_map.get(parameter.data_ptr(), "") for parameter in group["params"]]
            if any(name.startswith("transformer.h") for name in names):
                dec_matrix_lr = group["lr"]
            if any(name.startswith("encoder.blocks") for name in names):
                enc_matrix_lr = group["lr"]
            if any(name.startswith("cross_attn_layers") for name in names):
                xattn_lr = group["lr"]

        assert dec_matrix_lr is not None, "No Muon group found for transformer.h"
        assert enc_matrix_lr is not None, "No Muon group found for encoder.blocks"
        assert xattn_lr is not None, "No Muon group found for cross_attn_layers"

        # decoder / encoder ratio should be constant
        expected_ratio = DEFAULT_MATRIX_LR / DEFAULT_ENCODER_MATRIX_LR
        actual_ratio = dec_matrix_lr / enc_matrix_lr
        assert actual_ratio == pytest.approx(expected_ratio), (
            f"dec/enc matrix LR ratio = {actual_ratio}, expected {expected_ratio} "
            f"(batch_lr_scale={batch_lr_scale})"
        )

        # decoder / xattn ratio should be constant
        expected_xattn_ratio = DEFAULT_MATRIX_LR / DEFAULT_XATTN_MATRIX_LR
        actual_xattn_ratio = dec_matrix_lr / xattn_lr
        assert actual_xattn_ratio == pytest.approx(expected_xattn_ratio), (
            f"dec/xattn matrix LR ratio = {actual_xattn_ratio}, "
            f"expected {expected_xattn_ratio} (batch_lr_scale={batch_lr_scale})"
        )

    @pytest.mark.parametrize("batch_lr_scale", [0.25, 0.5, 1.0, 2.0, 4.0])
    def test_embedding_lr_ratio_invariant(self, model, batch_lr_scale):
        """decoder_embedding_lr / encoder_embedding_lr should always equal
        DEFAULT_EMBEDDING_LR / DEFAULT_ENCODER_EMBEDDING_LR regardless of
        batch_lr_scale. Both sides include dmodel_lr_scale so it cancels."""
        opt = model.setup_optimizer(
            unembedding_lr=DEFAULT_UNEMBEDDING_LR * batch_lr_scale,
            embedding_lr=DEFAULT_EMBEDDING_LR * batch_lr_scale,
            scalar_lr=DEFAULT_SCALAR_LR * batch_lr_scale,
            matrix_lr=DEFAULT_MATRIX_LR * batch_lr_scale,
            weight_decay=DEFAULT_WEIGHT_DECAY,
            encoder_embedding_lr=DEFAULT_ENCODER_EMBEDDING_LR * batch_lr_scale,
            encoder_matrix_lr=DEFAULT_ENCODER_MATRIX_LR * batch_lr_scale,
            xattn_matrix_lr=DEFAULT_XATTN_MATRIX_LR * batch_lr_scale,
        )

        name_map = _named_param_ptr_map(model)
        dec_embed_lr = None
        enc_embed_lr = None

        for group in opt.param_groups:
            if group["kind"] != OptimizerKind.ADAMW:
                continue
            rep = name_map.get(group["params"][0].data_ptr(), "")
            if rep.startswith("transformer.wte") and dec_embed_lr is None:
                dec_embed_lr = group["lr"]
            elif (
                rep.startswith(
                    (
                        "encoder.feature_embed",
                        "encoder.magnitude_embed",
                        "encoder.value_embeds",
                    )
                )
                and enc_embed_lr is None
            ):
                enc_embed_lr = group["lr"]

        assert dec_embed_lr is not None, "No AdamW group found for transformer.wte"
        assert enc_embed_lr is not None, "No AdamW group found for encoder embeddings"

        expected_ratio = DEFAULT_EMBEDDING_LR / DEFAULT_ENCODER_EMBEDDING_LR
        actual_ratio = dec_embed_lr / enc_embed_lr
        assert actual_ratio == pytest.approx(expected_ratio), (
            f"dec/enc embedding LR ratio = {actual_ratio}, expected {expected_ratio} "
            f"(batch_lr_scale={batch_lr_scale})"
        )


def _head_model():
    return build_tiny_model(
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


def _group_signature(optimizer, model):
    names = _named_param_ptr_map(model)
    return [
        (
            group["kind"],
            tuple(names[parameter.data_ptr()] for parameter in group["params"]),
            group["lr"],
            group.get("initial_lr"),
        )
        for group in optimizer.param_groups
    ]


def test_fp_head_lr_none_is_optimizer_group_identity() -> None:
    model = _head_model()
    implicit = model.setup_optimizer()
    explicit = model.setup_optimizer(fp_head_lr=None)
    assert _group_signature(explicit, model) == _group_signature(implicit, model)


def test_custom_fp_head_lr_changes_only_fp_head_group() -> None:
    model = _head_model()
    baseline = model.setup_optimizer(unembedding_lr=0.004)
    candidate = model.setup_optimizer(unembedding_lr=0.004, fp_head_lr=0.007)
    baseline_groups = _group_signature(baseline, model)
    candidate_groups = _group_signature(candidate, model)
    assert len(candidate_groups) == len(baseline_groups)
    changed = []
    for before, after in zip(baseline_groups, candidate_groups, strict=True):
        assert before[:2] == after[:2]
        if before[2:] != after[2:]:
            changed.append(after)
    assert len(changed) == 1
    assert set(changed[0][1]) == {
        "feature_pred_head.weight",
        "feature_pred_head.bias",
    }


def test_trainable_only_keeps_both_heads_at_distinct_lrs() -> None:
    model = _head_model()
    model.requires_grad_(True)
    optimizer = model.setup_optimizer(
        unembedding_lr=0.004,
        fp_head_lr=0.007,
        trainable_only=True,
    )
    groups = _group_signature(optimizer, model)
    head_lrs = {
        names: lr
        for _kind, names, lr, _initial_lr in groups
        if names
        in {
            ("lm_head.weight",),
            ("feature_pred_head.weight", "feature_pred_head.bias"),
        }
    }
    assert set(head_lrs) == {
        ("lm_head.weight",),
        ("feature_pred_head.weight", "feature_pred_head.bias"),
    }
    # Assert the exact mapping, not merely that the two differ: exchanging the
    # two head learning rates would satisfy "distinct" while sending the NTP
    # readout and the policy head to each other's objective's rate.
    scale = (model.dec_config.n_embd / BASE_MODEL_DIM) ** -0.5
    assert head_lrs[("lm_head.weight",)] == pytest.approx(0.004 * scale)
    assert head_lrs[("feature_pred_head.weight", "feature_pred_head.bias")] == pytest.approx(
        0.007 * scale
    )
