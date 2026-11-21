"""Unit tests for model_detection (Task 9)."""

from __future__ import annotations

import dataclasses
from types import SimpleNamespace
from typing import cast

import pytest

from errors import RemovedFeatureError
from steering.eval.axbench_base import model_detection as model_detection_module
from steering.eval.axbench_base.model_detection import detect_target_model_metadata
from steering.eval.axbench_base.types import AxBenchBaseError, TargetModelMetadata
from steering.eval.cluster_feature_contract import ClusterFeatureContract
from tests.axbench_base.unit.test_config import make_valid_config


def _make_encdec_model(
    *,
    num_buckets: int = 8,
    n_sae_features: int = 16384,
    meta_data: dict | None = None,
    attach_meta: bool = True,
) -> SimpleNamespace:
    """Build a lightweight stand-in for an EncoderDecoderGPT.

    Only the attributes ``detect_target_model_metadata`` reads are set:
    ``encoder.config.num_buckets``, ``encoder.config.n_sae_features``,
    and ``_loaded_meta_data``.
    """
    encoder_config = SimpleNamespace(
        num_buckets=num_buckets,
        n_sae_features=n_sae_features,
    )
    encoder = SimpleNamespace(config=encoder_config)
    model = SimpleNamespace(encoder=encoder)
    if attach_meta:
        model._loaded_meta_data = meta_data if meta_data is not None else {}
    return model


def _make_decoder_only_model() -> SimpleNamespace:
    """Stand-in for a GPT (decoder-only) — deliberately lacks ``.encoder``."""
    return SimpleNamespace(transformer=SimpleNamespace(h=[]))


def test_detect_returns_target_model_metadata_instance():
    """Both branches must return a TargetModelMetadata dataclass instance."""
    cfg = make_valid_config(device="cpu")
    dec_meta = detect_target_model_metadata(_make_decoder_only_model(), cfg)
    enc_meta = detect_target_model_metadata(_make_encdec_model(), cfg)
    assert isinstance(dec_meta, TargetModelMetadata)
    assert isinstance(enc_meta, TargetModelMetadata)


def test_detect_decoder_only_returns_none_for_num_buckets():
    model = _make_decoder_only_model()
    cfg = make_valid_config(device="cpu")
    meta = detect_target_model_metadata(model, cfg)
    assert meta.num_buckets is None
    assert meta.n_sae_features is None
    assert meta.negative_selection is None
    assert meta.feature_selection is None
    assert meta.per_feature_stats is None
    assert meta.device == "cpu"
    # Review-fix 1: sae_layer/segment_size are parsed from cfg.feature_dir
    # on both branches. make_valid_config uses
    # "out/data/features/s128/l8/a_res/p_max_pool/f16k/m_gemma-2-2b/base".
    assert meta.sae_layer == 8
    assert meta.segment_size == 128


def test_detect_encdec_reads_num_buckets_from_encoder_config():
    model = _make_encdec_model(num_buckets=8, n_sae_features=16384)
    cfg = make_valid_config(device="cuda")
    meta = detect_target_model_metadata(model, cfg)
    assert meta.num_buckets == 8
    assert meta.n_sae_features == 16384
    assert meta.device == "cuda"
    # Review-fix 1: same source as the decoder-only branch.
    assert meta.sae_layer == 8
    assert meta.segment_size == 128


def test_detect_encdec_reads_negative_selection_from_meta_data():
    neg_sel = {"strategy": "global", "n_inactive": 64}
    model = _make_encdec_model(
        meta_data={"user_config": {"negative_selection": neg_sel}},
    )
    cfg = make_valid_config()
    meta = detect_target_model_metadata(model, cfg)
    assert meta.negative_selection == neg_sel


def test_detect_encdec_reads_feature_selection_from_meta_data():
    feat_sel = {"threshold_type": "percentile", "threshold": 0.95}
    model = _make_encdec_model(
        meta_data={"user_config": {"feature_selection": feat_sel}},
    )
    cfg = make_valid_config()
    meta = detect_target_model_metadata(model, cfg)
    assert meta.feature_selection == feat_sel


def test_detect_raises_on_encdec_without_loaded_meta_data():
    model = _make_encdec_model(attach_meta=False)
    cfg = make_valid_config()
    with pytest.raises(AxBenchBaseError, match="lacks _loaded_meta_data"):
        detect_target_model_metadata(model, cfg)


def test_detect_d12_num_buckets_1_no_negative_sampling():
    """d12-style: num_buckets=1, no negative selection configured."""
    model = _make_encdec_model(
        num_buckets=1,
        meta_data={"user_config": {}},
    )
    cfg = make_valid_config()
    meta = detect_target_model_metadata(model, cfg)
    assert meta.num_buckets == 1
    assert meta.negative_selection is None
    assert meta.feature_selection is None


def test_detect_d13_num_buckets_8_with_negative_sampling():
    """d13-style: num_buckets=8, negative selection configured."""
    model = _make_encdec_model(
        num_buckets=8,
        meta_data={
            "user_config": {
                "negative_selection": {"strategy": "global", "n": 216},
                "feature_selection": {
                    "threshold_type": "percentile",
                    "threshold": 0.9,
                },
            },
        },
    )
    cfg = make_valid_config()
    meta = detect_target_model_metadata(model, cfg)
    assert meta.num_buckets == 8
    assert meta.negative_selection is not None
    assert meta.negative_selection["strategy"] == "global"
    assert meta.feature_selection is not None
    assert meta.feature_selection["threshold"] == 0.9


def test_detect_encdec_user_config_missing():
    """If meta_data has no user_config key, the fallback dict is used."""
    model = _make_encdec_model(meta_data={"other_field": "ignored"})
    cfg = make_valid_config()
    meta = detect_target_model_metadata(model, cfg)
    assert meta.negative_selection is None
    assert meta.feature_selection is None


def test_detect_encdec_user_config_is_none():
    """If meta_data['user_config'] is explicitly None, the fallback dict is used."""
    model = _make_encdec_model(meta_data={"user_config": None})
    cfg = make_valid_config()
    meta = detect_target_model_metadata(model, cfg)
    assert meta.negative_selection is None
    assert meta.feature_selection is None


def test_detect_sae_layer_parsed_from_feature_dir():
    """sae_layer should reflect the l{layer} component of cfg.feature_dir."""
    cfg = make_valid_config(
        feature_dir="out/data/features/s64/l12/a_res/p_max_pool/f16k/m_gemma-2-2b/base",
    )
    meta = detect_target_model_metadata(_make_decoder_only_model(), cfg)
    assert meta.sae_layer == 12
    assert meta.segment_size == 64


def test_detect_result_fields_are_stable():
    """Both branches must return the same set of dataclass fields."""
    expected = {
        "num_buckets",
        "n_sae_features",
        "negative_selection",
        "feature_selection",
        "feature_mask",
        "per_feature_stats",
        "device",
        # Review-fix 1: added by the batch refactor
        "sae_layer",
        "segment_size",
    }
    cfg = make_valid_config(device="cpu")
    dec_meta = detect_target_model_metadata(_make_decoder_only_model(), cfg)
    enc_meta = detect_target_model_metadata(_make_encdec_model(), cfg)
    dec_fields = {f.name for f in dataclasses.fields(dec_meta)}
    enc_fields = {f.name for f in dataclasses.fields(enc_meta)}
    assert dec_fields == expected
    assert enc_fields == expected


def test_detect_encdec_rejects_removed_feature_clustering_metadata():
    model = _make_encdec_model(
        meta_data={"feature_clustering_canonical": {"clusters": [[1, 2]]}},
    )
    cfg = make_valid_config()
    with pytest.raises(RemovedFeatureError, match="feature clustering pipeline was removed"):
        detect_target_model_metadata(model, cfg)


def test_detect_encdec_cluster_contract_validates_resume_metadata(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The clustered branch validates canonical metadata instead of rejecting it."""

    metadata = {
        "feature_clusters_descriptor": {"content_sha256": "a" * 64},
        "feature_clusters_canonical": {
            "schema_version": 1,
            "clusters": [[1, 2]],
        },
        "user_config": {"negative_selection": {"n": 3}},
    }
    resolved = object()
    contract = cast(
        ClusterFeatureContract,
        SimpleNamespace(consolidation=SimpleNamespace(resolved=resolved)),
    )
    calls: list[tuple[object, object, str]] = []

    def assert_match(
        resolved_arg: object,
        metadata_arg: object,
        *,
        checkpoint_dir: str,
    ) -> None:
        calls.append((resolved_arg, metadata_arg, checkpoint_dir))

    monkeypatch.setattr(
        model_detection_module,
        "assert_resume_feature_clusters_match",
        assert_match,
    )

    result = detect_target_model_metadata(
        _make_encdec_model(meta_data=metadata),
        make_valid_config(),
        cluster_contract=contract,
    )

    assert result.negative_selection == {"n": 3}
    assert calls == [
        (resolved, metadata, "axbench_base target model metadata"),
    ]
