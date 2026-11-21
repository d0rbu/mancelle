from __future__ import annotations

import inspect
import types
from pathlib import Path
from typing import Any, cast

import pytest
import torch as t
from beartype.roar import BeartypeCallHintParamViolation
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

import nanochat.flash_attention as _fa_mod
import scripts.train.conditional_encdec_train as train_mod
import steering.conditional.encoder_decoder.sae_embedding_init as init_mod
from nanochat.flash_attention import FlashAttnImpl
from steering.conditional.encoder_decoder.sae_embedding_init import (
    EncoderEmbeddingClusterVectors,
    EncoderEmbeddingInit,
    EncoderEmbeddingInitResult,
    average_row_norm,
    coerce_sae_l0,
    initialize_encoder_feature_embeddings_from_matrix,
    load_sae_decoder_matrix_from_cfg,
    maybe_initialize_encoder_feature_embeddings_from_sae,
    normalize_to_average_row_norm,
    orient_sae_decoder_matrix,
    parse_encoder_embedding_cluster_vectors,
    parse_encoder_embedding_init,
)
from steering.conditional.encoder_decoder.sae_embedding_init import (
    project_sae_decoder_matrix as _project_sae_decoder_matrix,
)
from steering.refined_types import PositiveInt
from steering.sae_resolver import L0Tier, SAERelease
from tests.encdec_test_helpers import build_tiny_model

CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "configs" / "training"


def project_sae_decoder_matrix(
    matrix: t.Tensor,
    encoder_dim: int,
    mode: EncoderEmbeddingInit,
    *,
    seed: int,
    rows_are_feature_vocab: bool = False,
) -> t.Tensor:
    """Pass raw test values through the production beartype boundary."""
    return _project_sae_decoder_matrix(
        matrix,
        cast(PositiveInt, encoder_dim),
        mode,
        seed=seed,
        rows_are_feature_vocab=rows_are_feature_vocab,
    )


@pytest.fixture(autouse=True)
def _force_sdpa():
    old = _fa_mod._override_impl
    _fa_mod._override_impl = FlashAttnImpl.SDPA
    yield
    _fa_mod._override_impl = old


def _simple_embedding_model(n_sae_features: int = 8, encoder_dim: int = 4):
    """A duck-typed stand-in for EncoderDecoderGPT.

    Since #695 the SAE init reads ``encoder.config.n_sae_features`` (the ORIGINAL
    dictionary width) and ``encoder.feature_vocab`` rather than inferring the width
    from ``feature_embed``'s row count — which is ``V``, not ``n_sae``, once the encoder
    is compacted. The fake must expose the same contract as the real
    ``FeatureEncoder``; ``feature_vocab=None`` is the uncompacted case these tests
    exercise.
    """
    return types.SimpleNamespace(
        encoder=types.SimpleNamespace(
            feature_embed=t.nn.Embedding(n_sae_features, encoder_dim),
            magnitude_embed=t.nn.Embedding(3, encoder_dim),
            feature_vocab=None,
            config=types.SimpleNamespace(n_sae_features=n_sae_features),
        )
    )


def _forward_inputs(model, B: int = 2, T: int = 8, n_seg: int = 2, F: int = 3):
    idx = t.randint(0, model.dec_config.vocab_size, (B, T))
    targets = t.randint(0, model.dec_config.vocab_size, (B, T))
    enc_fids = t.randint(0, model.encoder.config.n_sae_features, (B, n_seg, F))
    enc_mags = t.randint(0, model.encoder.config.num_buckets, (B, n_seg, F))
    enc_mask = t.ones(B, n_seg, F, dtype=t.bool)
    seg_size = T // n_seg
    seg_map = (t.arange(T).unsqueeze(0).expand(B, -1) // seg_size).clamp(max=n_seg - 1)
    return idx, targets, enc_fids, enc_mags, enc_mask, seg_map


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, None),
        ("null", None),
        ("none", None),
        ("random", EncoderEmbeddingInit.RANDOM),
        ("RANDOM", EncoderEmbeddingInit.RANDOM),
        ("svd", EncoderEmbeddingInit.SVD),
        ("SVD", EncoderEmbeddingInit.SVD),
    ],
)
def test_parse_encoder_embedding_init(raw, expected):
    assert parse_encoder_embedding_init(raw) is expected


def test_parse_encoder_embedding_init_rejects_invalid_value():
    with pytest.raises(ValueError, match="null, random, or svd"):
        parse_encoder_embedding_init("pca")


def test_parse_encoder_embedding_init_rejects_wrong_type():
    with pytest.raises(ValueError, match="int"):
        parse_encoder_embedding_init(1)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        (None, EncoderEmbeddingClusterVectors.MEAN),
        ("null", EncoderEmbeddingClusterVectors.MEAN),
        ("none", EncoderEmbeddingClusterVectors.MEAN),
        ("representative", EncoderEmbeddingClusterVectors.REPRESENTATIVE),
        ("REPRESENTATIVE", EncoderEmbeddingClusterVectors.REPRESENTATIVE),
        ("mean", EncoderEmbeddingClusterVectors.MEAN),
        ("MEAN", EncoderEmbeddingClusterVectors.MEAN),
    ],
)
def test_parse_encoder_embedding_cluster_vectors(raw, expected):
    assert parse_encoder_embedding_cluster_vectors(raw) is expected


def test_parse_encoder_embedding_cluster_vectors_treats_absent_key_as_mean():
    cfg: dict[str, object] = {}
    assert (
        parse_encoder_embedding_cluster_vectors(
            cfg.get("initialize_encoder_embeddings_cluster_vectors")
        )
        is EncoderEmbeddingClusterVectors.MEAN
    )


def test_parse_encoder_embedding_cluster_vectors_rejects_invalid_value():
    with pytest.raises(ValueError, match="representative or mean"):
        parse_encoder_embedding_cluster_vectors("median")


def test_parse_encoder_embedding_cluster_vectors_rejects_wrong_type():
    with pytest.raises(ValueError, match="int"):
        parse_encoder_embedding_cluster_vectors(1)


def test_conditional_encdec_train_default_config_exposes_null_initializer():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="conditional_encdec_train")
    assert cfg.initialize_encoder_embeddings_from_sae is None
    assert cfg.initialize_encoder_embeddings_cluster_vectors == "mean"
    assert cfg.sae.l0 is None


def test_conditional_encdec_train_config_accepts_svd_override_case_insensitive():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(
            config_name="conditional_encdec_train",
            overrides=["initialize_encoder_embeddings_from_sae=SVD"],
        )
    assert parse_encoder_embedding_init(cfg.initialize_encoder_embeddings_from_sae) is (
        EncoderEmbeddingInit.SVD
    )


def test_gemma3_training_config_provides_l0_tier_for_sae_load():
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base=None):
        cfg = compose(config_name="conditional_encdec_train_gemma3")
    assert cfg.sae.release == "gemmascope_2"
    assert cfg.sae.l0 == "medium"


def test_coerce_sae_l0_handles_v1_and_v2_values():
    assert coerce_sae_l0(SAERelease.gemmascope, None) is None
    assert coerce_sae_l0(SAERelease.gemmascope, 176) == 176
    assert coerce_sae_l0(SAERelease.gemmascope_2, None) is None
    assert coerce_sae_l0(SAERelease.gemmascope_2, "medium") is L0Tier.medium
    assert coerce_sae_l0(SAERelease.gemmascope_2, L0Tier.big) is L0Tier.big


def test_coerce_sae_l0_rejects_cross_release_values():
    with pytest.raises(ValueError, match="average-L0"):
        coerce_sae_l0(SAERelease.gemmascope, "medium")
    with pytest.raises(ValueError, match="average-L0"):
        coerce_sae_l0(SAERelease.gemmascope, L0Tier.medium)
    with pytest.raises(ValueError, match="gemma-scope-2 tier"):
        coerce_sae_l0(SAERelease.gemmascope_2, 176)


def test_load_sae_decoder_matrix_resolves_cfg_and_accepts_tuple_return(monkeypatch):
    calls = {}

    def fake_resolve_sae(**kwargs):
        calls["resolve"] = kwargs
        return types.SimpleNamespace(
            release="fake-release",
            sae_id="fake-sae-id",
            hook_name="blocks.8.hook_resid_post",
        )

    matrix = t.arange(12, dtype=t.float64).reshape(3, 4).requires_grad_()

    def fake_from_pretrained(**kwargs):
        calls["from_pretrained"] = kwargs
        return (types.SimpleNamespace(W_dec=matrix), {"cfg": "ignored"}, None)

    monkeypatch.setattr(init_mod, "resolve_sae", fake_resolve_sae)
    monkeypatch.setattr(init_mod.SAE, "from_pretrained", staticmethod(fake_from_pretrained))

    cfg = OmegaConf.create(
        {
            "release": "gemmascope_2",
            "model": "gemma-3-1b-pt",
            "activation_type": "residual",
            "layer": 22,
            "width": "16k",
            "l0": "medium",
        }
    )
    loaded = load_sae_decoder_matrix_from_cfg(cfg)

    assert calls["resolve"]["l0"] is L0Tier.medium
    assert calls["from_pretrained"] == {
        "release": "fake-release",
        "sae_id": "fake-sae-id",
        "device": "cpu",
    }
    assert loaded.dtype == t.float32
    assert loaded.device.type == "cpu"
    assert not loaded.requires_grad
    assert t.equal(loaded, matrix.detach().float())


def test_load_sae_decoder_matrix_rejects_sae_without_w_dec(monkeypatch):
    monkeypatch.setattr(
        init_mod,
        "resolve_sae",
        lambda **_kwargs: types.SimpleNamespace(
            release="fake-release",
            sae_id="fake-sae-id",
            hook_name="hook",
        ),
    )
    monkeypatch.setattr(
        init_mod.SAE,
        "from_pretrained",
        staticmethod(lambda **_kwargs: types.SimpleNamespace()),
    )
    cfg = OmegaConf.create(
        {
            "release": "gemmascope",
            "model": "gemma-2-2b",
            "activation_type": "residual",
            "layer": 8,
            "width": "16k",
            "l0": None,
        }
    )
    with pytest.raises(ValueError, match="does not expose W_dec"):
        load_sae_decoder_matrix_from_cfg(cfg)


def test_orient_sae_decoder_matrix_accepts_canonical_and_transpose():
    canonical = t.arange(24, dtype=t.float32).reshape(6, 4)
    assert t.equal(orient_sae_decoder_matrix(canonical, 6), canonical)
    assert t.equal(orient_sae_decoder_matrix(canonical.T, 6), canonical)


def test_orient_sae_decoder_matrix_rejects_shape_without_feature_axis():
    with pytest.raises(ValueError, match="n_sae_features=6"):
        orient_sae_decoder_matrix(t.zeros(5, 4), 6)


def test_orient_sae_decoder_matrix_rejects_non_matrix_input():
    with pytest.raises(ValueError, match="rank-2"):
        orient_sae_decoder_matrix(t.zeros(2, 3, 4), 6)


def test_random_projection_is_deterministic_and_seeded():
    matrix = t.arange(30, dtype=t.float32).reshape(6, 5) / 10
    a = project_sae_decoder_matrix(matrix, 3, EncoderEmbeddingInit.RANDOM, seed=123)
    b = project_sae_decoder_matrix(matrix, 3, EncoderEmbeddingInit.RANDOM, seed=123)
    c = project_sae_decoder_matrix(matrix, 3, EncoderEmbeddingInit.RANDOM, seed=124)
    assert t.equal(a, b)
    assert not t.equal(a, c)


def test_random_projection_does_not_advance_global_rng():
    matrix = t.arange(30, dtype=t.float32).reshape(6, 5) / 10
    t.manual_seed(12345)
    before = t.random.get_rng_state()
    project_sae_decoder_matrix(matrix, 3, EncoderEmbeddingInit.RANDOM, seed=999)
    after = t.random.get_rng_state()
    assert t.equal(after, before)


def test_svd_projection_matches_top_right_singular_projection():
    matrix = t.tensor(
        [
            [1.0, 0.0, 2.0, 0.0],
            [0.0, 3.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 4.0],
            [1.0, 1.0, 0.0, 0.0],
        ]
    )
    actual = project_sae_decoder_matrix(matrix, 2, EncoderEmbeddingInit.SVD, seed=0)
    _eigvals, eigvecs = t.linalg.eigh(matrix.T @ matrix)
    expected = matrix @ eigvecs[:, -2:].flip(dims=(1,))
    assert t.allclose(actual.abs(), expected.abs(), atol=1e-6)


def test_svd_projection_rejects_encoder_dim_larger_than_sae_dim():
    """rows=6, sae_dim=4, encoder_dim=5: rank(cov) <= min(6, 4) = 4 < 5."""
    with pytest.raises(ValueError, match="SAE decoder dimension is the binding constraint"):
        project_sae_decoder_matrix(t.randn(6, 4), 5, EncoderEmbeddingInit.SVD, seed=0)


def test_projection_rejects_bad_encoder_dim_and_unknown_mode():
    with pytest.raises(BeartypeCallHintParamViolation, match="encoder_dim"):
        project_sae_decoder_matrix(t.randn(6, 4), 0, EncoderEmbeddingInit.RANDOM, seed=0)
    with pytest.raises(BeartypeCallHintParamViolation, match="mode"):
        project_sae_decoder_matrix(
            t.randn(6, 4),
            2,
            cast(EncoderEmbeddingInit, "pca"),
            seed=0,
        )


def test_average_row_norm_rejects_bad_shapes_and_nonfinite_values():
    with pytest.raises(ValueError, match="rank-2"):
        average_row_norm(t.ones(3))
    bad = t.ones(2, 3)
    bad[0, 0] = float("nan")
    with pytest.raises(ValueError, match="not finite"):
        average_row_norm(bad)


def test_normalize_to_average_row_norm_matches_target():
    matrix = t.randn(10, 4)
    target = t.tensor(7.5)
    normalized = normalize_to_average_row_norm(matrix, target)
    assert average_row_norm(normalized).item() == pytest.approx(7.5, rel=1e-6)


def test_normalize_to_average_row_norm_rejects_zero_projection_and_bad_target():
    with pytest.raises(ValueError, match="zero average"):
        normalize_to_average_row_norm(t.zeros(10, 4), t.tensor(1.0))
    with pytest.raises(ValueError, match="not finite"):
        normalize_to_average_row_norm(t.ones(10, 4), t.tensor(float("nan")))
    with pytest.raises(ValueError, match="must be positive"):
        normalize_to_average_row_norm(t.ones(10, 4), t.tensor(0.0))


def test_initialize_from_matrix_preserves_random_average_norm_and_other_embeddings():
    t.manual_seed(0)
    model = _simple_embedding_model(n_sae_features=8, encoder_dim=4)
    original_feature = model.encoder.feature_embed.weight.detach().clone()
    original_magnitude = model.encoder.magnitude_embed.weight.detach().clone()
    target_norm = average_row_norm(model.encoder.feature_embed.weight).item()

    result = initialize_encoder_feature_embeddings_from_matrix(
        model,
        sae_decoder_matrix=t.randn(8, 6),
        mode=EncoderEmbeddingInit.RANDOM,
        seed=11,
    )

    assert result.target_avg_norm == pytest.approx(target_norm)
    assert result.final_avg_norm == pytest.approx(target_norm, rel=1e-6)
    assert not t.equal(model.encoder.feature_embed.weight, original_feature)
    assert t.equal(model.encoder.magnitude_embed.weight, original_magnitude)


def test_initialize_from_matrix_preserves_feature_embedding_dtype():
    t.manual_seed(0)
    model = _simple_embedding_model(n_sae_features=8, encoder_dim=4)
    model.encoder.feature_embed = model.encoder.feature_embed.to(dtype=t.bfloat16)
    target_norm = average_row_norm(model.encoder.feature_embed.weight).item()

    result = initialize_encoder_feature_embeddings_from_matrix(
        model,
        sae_decoder_matrix=t.randn(8, 6),
        mode=EncoderEmbeddingInit.RANDOM,
        seed=11,
    )

    assert model.encoder.feature_embed.weight.dtype == t.bfloat16
    assert result.target_avg_norm == pytest.approx(target_norm)
    assert result.final_avg_norm == pytest.approx(target_norm, rel=2e-3)


def test_initialize_from_matrix_rejects_zero_projection_without_overwriting():
    t.manual_seed(0)
    model = _simple_embedding_model(n_sae_features=8, encoder_dim=4)
    before = model.encoder.feature_embed.weight.detach().clone()

    with pytest.raises(ValueError, match="zero average row norm"):
        initialize_encoder_feature_embeddings_from_matrix(
            model,
            sae_decoder_matrix=t.zeros(8, 6),
            mode=EncoderEmbeddingInit.RANDOM,
            seed=11,
        )

    assert t.equal(model.encoder.feature_embed.weight, before)


def test_initialize_from_matrix_rejects_internal_shape_mismatch_without_overwriting(
    monkeypatch,
):
    t.manual_seed(0)
    model = _simple_embedding_model(n_sae_features=8, encoder_dim=4)
    before = model.encoder.feature_embed.weight.detach().clone()

    monkeypatch.setattr(
        init_mod,
        "project_sae_decoder_matrix",
        lambda *_args, **_kwargs: t.zeros(7, 4),
    )
    with pytest.raises(ValueError, match="shape mismatch"):
        init_mod.initialize_encoder_feature_embeddings_from_matrix(
            model,
            sae_decoder_matrix=t.randn(8, 6),
            mode=EncoderEmbeddingInit.RANDOM,
            seed=11,
        )

    assert t.equal(model.encoder.feature_embed.weight, before)


def test_initialize_from_matrix_accepts_transposed_sae_decoder_matrix():
    t.manual_seed(0)
    model = _simple_embedding_model(n_sae_features=8, encoder_dim=4)
    target_norm = average_row_norm(model.encoder.feature_embed.weight).item()
    initialize_encoder_feature_embeddings_from_matrix(
        model,
        sae_decoder_matrix=t.randn(6, 8),
        mode=EncoderEmbeddingInit.SVD,
        seed=0,
    )
    assert average_row_norm(model.encoder.feature_embed.weight).item() == pytest.approx(
        target_norm, rel=1e-6
    )


def test_maybe_initialize_null_is_noop_and_does_not_load_sae(monkeypatch):
    model = _simple_embedding_model()
    before = model.encoder.feature_embed.weight.detach().clone()

    def fail_loader(_sae_cfg):
        raise AssertionError("SAE loader should not run for null initializer")

    monkeypatch.setattr(init_mod, "load_sae_decoder_matrix_from_cfg", fail_loader)
    cfg = OmegaConf.create({"initialize_encoder_embeddings_from_sae": None, "sae": {}})

    assert maybe_initialize_encoder_feature_embeddings_from_sae(model, cfg, seed=0) is None
    assert t.equal(model.encoder.feature_embed.weight, before)


def test_maybe_initialize_null_accepts_plain_config_objects(monkeypatch):
    model = _simple_embedding_model()
    before = model.encoder.feature_embed.weight.detach().clone()

    class GetOnlyCfg:
        def get(self, key, default=None):
            assert key in {
                "initialize_encoder_embeddings_from_sae",
                "initialize_encoder_embeddings_cluster_vectors",
            }
            return default

    monkeypatch.setattr(
        init_mod,
        "load_sae_decoder_matrix_from_cfg",
        lambda _sae_cfg: pytest.fail("SAE loader should not run for null initializer"),
    )
    assert maybe_initialize_encoder_feature_embeddings_from_sae(model, GetOnlyCfg(), seed=0) is None
    assert (
        maybe_initialize_encoder_feature_embeddings_from_sae(
            model,
            types.SimpleNamespace(initialize_encoder_embeddings_from_sae=None),
            seed=0,
        )
        is None
    )
    assert t.equal(model.encoder.feature_embed.weight, before)


def test_cluster_vector_mean_with_svd_loads_sae_then_threads_clusters(monkeypatch):
    """svd + mean is a supported row, and loading still precedes initialization."""
    model = _simple_embedding_model()
    feature_clusters = cast(Any, object())
    matrix = t.ones(8, 6)
    events: list[str] = []
    cfg = OmegaConf.create(
        {
            "initialize_encoder_embeddings_from_sae": "svd",
            "initialize_encoder_embeddings_cluster_vectors": "mean",
            "sae": {},
        }
    )
    expected = EncoderEmbeddingInitResult(
        mode=EncoderEmbeddingInit.SVD,
        n_sae_features=8,
        sae_dim=6,
        encoder_dim=4,
        target_avg_norm=1.0,
        projected_avg_norm=1.0,
        final_avg_norm=1.0,
        vocab_size=8,
        cluster_vectors=EncoderEmbeddingClusterVectors.MEAN,
        clusters_averaged=1,
        members_averaged=2,
    )

    def fake_load(sae_cfg):
        events.append("load")
        assert sae_cfg is cfg.sae
        return matrix

    def fake_initialize(actual_model, loaded, mode, **kwargs):
        events.append("initialize")
        assert actual_model is model
        assert loaded is matrix
        assert mode is EncoderEmbeddingInit.SVD
        assert kwargs["cluster_vectors"] is EncoderEmbeddingClusterVectors.MEAN
        assert kwargs["feature_clusters"] is feature_clusters
        return expected

    monkeypatch.setattr(init_mod, "load_sae_decoder_matrix_from_cfg", fake_load)
    monkeypatch.setattr(
        init_mod,
        "initialize_encoder_feature_embeddings_from_matrix",
        fake_initialize,
    )

    result = maybe_initialize_encoder_feature_embeddings_from_sae(
        model,
        cfg,
        seed=0,
        feature_clusters=feature_clusters,
    )

    assert result is expected
    assert events == ["load", "initialize"]


def test_invalid_cluster_vector_policy_still_fails_before_sae_load(monkeypatch):
    model = _simple_embedding_model()
    cfg = OmegaConf.create(
        {
            "initialize_encoder_embeddings_from_sae": "svd",
            "initialize_encoder_embeddings_cluster_vectors": "centroid",
            "sae": {},
        }
    )
    monkeypatch.setattr(
        init_mod,
        "load_sae_decoder_matrix_from_cfg",
        lambda _sae_cfg: pytest.fail("policy parsing must run before the SAE load"),
    )

    with pytest.raises(ValueError, match="representative or mean"):
        maybe_initialize_encoder_feature_embeddings_from_sae(model, cfg, seed=0)


def test_cluster_vector_mean_with_null_init_is_an_inert_no_op(monkeypatch):
    """mean + null init: no SAE table exists, so the default cannot raise — it no-ops."""
    model = _simple_embedding_model()
    before = model.encoder.feature_embed.weight.clone()
    cfg = OmegaConf.create(
        {
            "initialize_encoder_embeddings_from_sae": None,
            "initialize_encoder_embeddings_cluster_vectors": "mean",
            "sae": {},
        }
    )
    monkeypatch.setattr(
        init_mod,
        "load_sae_decoder_matrix_from_cfg",
        lambda _sae_cfg: pytest.fail("null init mode must never load an SAE"),
    )
    assert (
        maybe_initialize_encoder_feature_embeddings_from_sae(
            model,
            cfg,
            seed=0,
            feature_clusters=cast(Any, object()),
        )
        is None
    )
    assert t.equal(model.encoder.feature_embed.weight, before)


def test_svd_with_clusters_and_explicit_representative_still_initializes(monkeypatch):
    """Explicit representative keeps threading the unchanged svd policy."""
    model = _simple_embedding_model()
    cfg = OmegaConf.create(
        {
            "initialize_encoder_embeddings_from_sae": "svd",
            "initialize_encoder_embeddings_cluster_vectors": "representative",
            "sae": {},
        }
    )
    monkeypatch.setattr(
        init_mod,
        "load_sae_decoder_matrix_from_cfg",
        lambda _sae_cfg: t.ones(8, 6),
    )
    seen = {}

    def fake_initialize(*args, **kwargs):
        seen["kwargs"] = kwargs
        return EncoderEmbeddingInitResult(
            mode=EncoderEmbeddingInit.SVD,
            n_sae_features=8,
            sae_dim=6,
            encoder_dim=4,
            target_avg_norm=1.0,
            projected_avg_norm=1.0,
            final_avg_norm=1.0,
            vocab_size=8,
            cluster_vectors=EncoderEmbeddingClusterVectors.REPRESENTATIVE,
        )

    monkeypatch.setattr(
        init_mod,
        "initialize_encoder_feature_embeddings_from_matrix",
        fake_initialize,
    )
    result = maybe_initialize_encoder_feature_embeddings_from_sae(
        model,
        cfg,
        seed=0,
        feature_clusters=cast(Any, object()),
    )
    assert result is not None
    assert seen["kwargs"]["cluster_vectors"] is EncoderEmbeddingClusterVectors.REPRESENTATIVE


@pytest.mark.parametrize(
    ("init_mode", "cluster_vectors", "feature_clusters", "expected_policy"),
    [
        ("random", "representative", None, "representative"),
        ("random", "mean", object(), "mean"),
        ("random", None, object(), "mean"),
        ("random", None, None, "mean"),
        ("svd", "mean", object(), "mean"),
    ],
)
def test_non_null_cluster_vector_validation_matrix_threads_valid_rows(
    monkeypatch,
    init_mode,
    cluster_vectors,
    feature_clusters,
    expected_policy,
):
    model = _simple_embedding_model()
    cfg = OmegaConf.create(
        {
            "initialize_encoder_embeddings_from_sae": init_mode,
            "initialize_encoder_embeddings_cluster_vectors": cluster_vectors,
            "sae": {},
        }
    )
    seen = {}
    fake_result = EncoderEmbeddingInitResult(
        mode=EncoderEmbeddingInit(init_mode),
        n_sae_features=8,
        sae_dim=6,
        encoder_dim=4,
        target_avg_norm=1.0,
        projected_avg_norm=1.0,
        final_avg_norm=1.0,
        vocab_size=8,
        cluster_vectors=EncoderEmbeddingClusterVectors(expected_policy),
    )

    monkeypatch.setattr(
        init_mod,
        "load_sae_decoder_matrix_from_cfg",
        lambda _sae_cfg: t.ones(8, 6),
    )

    def fake_initialize(*args, **kwargs):
        seen["args"] = args
        seen["kwargs"] = kwargs
        return fake_result

    monkeypatch.setattr(
        init_mod,
        "initialize_encoder_feature_embeddings_from_matrix",
        fake_initialize,
    )
    result = maybe_initialize_encoder_feature_embeddings_from_sae(
        model,
        cfg,
        seed=3,
        feature_clusters=cast(Any, feature_clusters),
    )

    assert result is fake_result
    assert seen["args"][2] is EncoderEmbeddingInit(init_mode)
    assert seen["kwargs"]["cluster_vectors"].value == expected_policy
    assert seen["kwargs"]["feature_clusters"] is feature_clusters


def test_maybe_initialize_non_null_requires_sae_cfg():
    model = _simple_embedding_model()
    cfg = OmegaConf.create({"initialize_encoder_embeddings_from_sae": "random"})
    with pytest.raises(ValueError, match="requires a configured sae"):
        maybe_initialize_encoder_feature_embeddings_from_sae(model, cfg, seed=0)


def test_maybe_initialize_random_uses_loader_and_cfg(monkeypatch):
    model = _simple_embedding_model(n_sae_features=8, encoder_dim=4)

    monkeypatch.setattr(
        init_mod,
        "load_sae_decoder_matrix_from_cfg",
        lambda sae_cfg: t.randn(8, 6),
    )
    cfg = OmegaConf.create(
        {
            "initialize_encoder_embeddings_from_sae": "random",
            "sae": {
                "release": "gemmascope",
                "model": "gemma-2-2b",
                "activation_type": "residual",
                "layer": 8,
                "width": "16k",
                "l0": None,
            },
        }
    )
    target_norm = average_row_norm(model.encoder.feature_embed.weight).item()
    result = maybe_initialize_encoder_feature_embeddings_from_sae(model, cfg, seed=77)
    assert result is not None
    assert result.mode is EncoderEmbeddingInit.RANDOM
    assert average_row_norm(model.encoder.feature_embed.weight).item() == pytest.approx(
        target_norm, rel=1e-6
    )


@pytest.mark.parametrize("mode", [EncoderEmbeddingInit.RANDOM, EncoderEmbeddingInit.SVD])
def test_initialized_tiny_encdec_forward_backward(mode):
    t.manual_seed(0)
    # n_sae must be >= the encoder dim (32 = N_EMBD // 2): the svd eigenbasis has rank
    # min(n_sae, sae_dim), so a 16-feature dictionary would make HALF of the 32 selected
    # eigenvectors null-space directions and silently emit 16 identically-zero columns.
    # (Real dictionaries are 16,384 / 65,536 against a 384-dim encoder, so this only ever
    # bites unrealistically tiny fixtures — but the fixture was producing garbage.)
    model = build_tiny_model(n_sae_features=64, num_buckets=4)
    model.train()
    target_norm = average_row_norm(model.encoder.feature_embed.weight).item()
    initialize_encoder_feature_embeddings_from_matrix(
        model,
        sae_decoder_matrix=t.randn(64, 40),
        mode=mode,
        seed=5,
    )

    idx, targets, enc_fids, enc_mags, enc_mask, seg_map = _forward_inputs(model)
    loss = model(idx, targets, enc_fids, enc_mags, enc_mask, seg_map)
    loss.backward()

    assert t.isfinite(loss)
    assert average_row_norm(model.encoder.feature_embed.weight).item() == pytest.approx(
        target_norm, rel=1e-6
    )
    assert model.encoder.feature_embed.weight.grad is not None
    assert model.encoder.feature_embed.weight.grad.abs().sum().item() > 0


def test_training_wiring_applies_initializer_after_fresh_init_before_optimizer():
    src = inspect.getsource(train_mod.train_one_config)
    init_idx = src.index("orig_model.init_weights()")
    sae_init_idx = src.index("maybe_initialize_encoder_feature_embeddings_from_sae(")
    optimizer_idx = src.index("optimizer = model.setup_optimizer")
    assert init_idx < sae_init_idx < optimizer_idx
    assert "if not resuming:\n        orig_model.init_weights()\n        maybe_initialize" in src
    assert "feature_clusters=shared.feature_clusters," in src
